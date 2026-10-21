"""Serial benchmark runner. Does not import any historical campaign launcher."""
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from cilight.io import isolated_config, save, sha


def configure(settings):
    # Set before numerical libraries are imported; also inherited by solver workers.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    sys.dont_write_bytecode = True
    home = os.environ.get('SUMO_HOME')
    if not home or not (Path(home) / 'tools').is_dir():
        raise RuntimeError('Set SUMO_HOME to an Eclipse SUMO 1.27.1 installation.')
    sys.path.insert(0, str(Path(home) / 'tools'))
    import sumolib
    binary = sumolib.checkBinary('sumo')
    version = subprocess.run([binary, '--version'], capture_output=True, text=True, check=True, timeout=15).stdout
    if 'sumo 1.27.1' not in version:
        raise RuntimeError('The reference profile requires SUMO 1.27.1, found: ' + version.splitlines()[0])
    import torch
    import config
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    config.CIMPC_IMPLEMENTATION_STAGE = 5
    config.CIMPC_INTERVAL = float(settings['interval'])
    config.CIMPC_HORIZON = settings['horizon']
    config.CIMPC_MIN_GREEN = float(settings['min_green'])
    config.EXTERNAL_CIMPC_MIN_GREEN = float(settings['min_green'])
    config.EXTERNAL_CIMPC_MAX_GREEN = float(settings['max_green'])
    config.CIMPC_YELLOW = float(settings['yellow'])
    config.CIMPC_PROCESS_WORKERS = settings['solver_workers']
    config.CIMPC_PARALLEL_WORKERS = 1
    config.STEP_LENGTH = 1.
    config.L_CIMPC_V2_MODEL_PATH = None
    config.EXTERNAL_CIMPC_MODEL_PATH = None
    config.COLLECT_EMISSIONS = False


def admission(output, stop=None):
    import psutil
    import shutil
    if stop and Path(stop).exists():
        raise RuntimeError('STOP requested; partial results retained.')
    if psutil.virtual_memory().available < 12 * 2**30:
        raise RuntimeError('Less than 12 GiB physical memory available; no run started/continued.')
    if shutil.disk_usage(output).free < 10 * 2**30:
        raise RuntimeError('Less than 10 GiB free on output volume.')
    # Unknown external SUMO instances are not admitted. Never terminate them.
    own = {p.pid for p in psutil.Process().children(recursive=True)}
    for p in psutil.process_iter(['name']):
        if (p.info['name'] or '').lower() in ('sumo', 'sumo.exe', 'sumo-gui', 'sumo-gui.exe') and p.pid not in own:
            raise RuntimeError('Another SUMO instance is active; run this release serially.')


def load_critic(path, expected_sha256):
    """Load the numeric final-weight archive bound to the network profile."""
    if Path(path).suffix != '.npz':
        raise ValueError('Final weights must be a numeric .npz archive')
    if sha(path) != expected_sha256:
        raise ValueError('Final weights digest mismatch')
    import numpy as np
    import torch
    from rl.terminal_value import TerminalCritic
    critic = TerminalCritic()
    with np.load(path, allow_pickle=False) as arrays:
        state = {name: torch.from_numpy(arrays[name].copy()) for name in arrays.files}
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Non-finite final weight tensor')
    critic.load_state_dict(state, strict=True)
    critic.eval()
    critic.requires_grad_(False)
    return critic


def demand_count(source, seconds):
    # Mirrors native mainline counting for explicit vehicles/trips and periodic flows.
    tree = ET.parse(source)
    begin = tree.find('time/begin')
    origin = float(begin.get('value')) if begin is not None else 0.
    total = 0
    for filename in tree.find('input/route-files').get('value').split(','):
        for node in ET.parse(Path(source).parent / filename.strip()).getroot():
            tag = node.tag.rsplit('}', 1)[-1]
            if tag in ('vehicle', 'trip'):
                depart = float(node.get('depart')) - origin
                total += int(0 <= depart < seconds)
            elif tag == 'flow':
                start, end = float(node.get('begin', '0')), float(node.get('end', '86400'))
                if start >= origin + seconds or end <= origin:
                    continue
                if 'probability' in node.attrib or 'number' in node.attrib:
                    raise ValueError('Unsupported stochastic/count flow in exact demand accounting')
                period = float(node.get('period')) if 'period' in node.attrib else 3600 / float(node.get('vehsPerHour'))
                if not math.isfinite(period) or period <= 0:
                    raise ValueError('Invalid flow period')
                total += sum(0 <= start + k * period - origin < seconds
                             for k in range(max(0, math.ceil((min(end, origin + seconds) - start) / period - 1e-12))))
    return total


def run_episode(settings, source, destination, seed, critic=None, beta=0., seconds=3600, stop=None):
    import numpy as np
    import torch
    import traci.constants as tc
    import config
    from controllers.terminal_value import TerminalValueCIMPC
    from utils.sumo_utils import start_sumo, reset_graph_cache
    from utils.sumo_statistics import summarize_statistics, verify_statistics_against_trips
    from cilight.aql import AQL, controlled_incoming_lanes

    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Refusing to overwrite an existing run: ' + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    admission(destination.parent, stop)
    destination.mkdir()
    source = isolated_config(source, destination)
    config.configure_scenario(str(source), str(destination))
    config.RANDOM_SEED = int(seed)
    reset_graph_cache()
    before = {k: v.clone() for k, v in critic.state_dict().items()} if critic is not None else {}
    save(destination / 'request.json', dict(network=settings['network'], seed=seed, seconds=seconds, beta=beta,
                                          method='CI-MPC' if critic is None else 'CILight'))
    conn, controller = None, None
    arrivals = teleports = 0
    started = time.perf_counter()
    try:
        conn = start_sumo(seed=seed, label=f'cilight_{os.getpid()}_{seed}', sim_seconds=seconds,
                          tripinfo_output=str(destination / 'tripinfo.xml'), tripinfo_write_unfinished=True,
                          statistics_output=str(destination / 'statistics.xml'))
        version = conn.getVersion()
        if len(conn.trafficlight.getIDList()) != settings['signal_count']:
            raise ValueError('Live signal count mismatch')
        begin = conn.simulation.getTime()
        aql = AQL(controlled_incoming_lanes(config.NET_FILE), begin=begin)
        variable = tc.LAST_STEP_VEHICLE_HALTING_NUMBER
        for lane in aql.lanes:
            conn.lane.subscribe(lane, (variable,))
        controller = TerminalValueCIMPC(conn, critic=critic, beta=beta, seconds=seconds)
        with (destination / 'aql_samples.jsonl').open('x', encoding='utf-8') as stream:
            import json
            for t in range(seconds):
                if t % int(settings['interval']) == 0:
                    admission(destination, stop)
                controller.step(float(t))
                conn.simulationStep()
                values = conn.lane.getAllSubscriptionResults()
                counts = {lane: values[lane][variable] for lane in aql.lanes}
                now = conn.simulation.getTime()
                aql.add(now, counts)
                stream.write(json.dumps(dict(time=now, halting_by_lane=counts)) + '\n')
                arrivals += conn.simulation.getArrivedNumber()
                teleports += conn.simulation.getStartingTeleportNumber()
                if (t + 1) % 100 == 0:
                    save(destination / 'progress.json', dict(simulation_s=t+1, seconds=seconds, seed=seed))
        diagnostics = list(controller.diagnostics)
        trace = controller.actual_phase_trace
        failures = sum(not row['success'] for row in controller.local_diagnostics)
    finally:
        # Native close and output flush; cleanup failures propagate, never count as success.
        try:
            if controller is not None:
                controller.close()
        finally:
            try:
                if conn is not None:
                    child = getattr(conn, '_process', None)
                    conn.close(wait=True)
                    if child is not None and child.wait(timeout=30) != 0:
                        raise RuntimeError('SUMO did not exit successfully')
            finally:
                reset_graph_cache()
    if failures:
        raise RuntimeError(f'{failures} local solver failures; retained as partial, not complete')
    expected = seconds // int(settings['interval'])
    if len(diagnostics) != expected:
        raise ValueError('Incomplete decision horizon')
    observed = {row['time']: row['nodes'] for row in trace}
    for row in diagnostics:
        for item in row['phase_choices'].split(';'):
            junction, phase = item.rsplit(':', 1)
            actual = observed[row['simulation_time']][junction]
            if actual['phase'] != int(phase) and not (actual['yellow'] and actual['pending'] == int(phase)):
                raise ValueError('Suggested and executed phase mismatch')
    if critic is not None and any(not torch.equal(v, critic.state_dict()[k]) for k, v in before.items()):
        raise ValueError('Rollout modified critic weights')
    metrics = summarize_statistics(destination / 'statistics.xml')
    verify_statistics_against_trips(metrics, destination / 'tripinfo.xml')
    metrics.update(aql.result(begin + seconds))
    times = [row['decision_time_s'] for row in diagnostics]
    metrics.update(network=settings['network'], method='CI-MPC' if critic is None else 'CILight', seed=seed,
                   seconds=seconds, beta=beta, decision_mean_ms=1000 * statistics.mean(times),
                   decision_p95_ms=1000 * float(np.quantile(times, .95)), decision_count=len(times),
                   wall_time_s=time.perf_counter()-started, completed_arrivals=arrivals, teleports=teleports,
                   solver_failures=failures, suggestion_execution_mismatches=0,
                   demand=demand_count(source, seconds), sumo_version=list(version))
    actions = [dict(time=r['simulation_time'], phases=r['phase_choices']) for r in diagnostics]
    save(destination / 'actions.json', actions)
    save(destination / 'observed_phase_trace.json', trace)
    save(destination / 'metrics.json', metrics)
    files = ('metrics.json', 'actions.json', 'observed_phase_trace.json', 'statistics.xml', 'tripinfo.xml',
             'aql_samples.jsonl', 'request.json')
    save(destination / 'complete.json', dict(sha256={name: sha(destination / name) for name in files}))
    print(f"{settings['name']} / {metrics['method']} / {seed}: ATT={metrics['average_trip_time_s']}", flush=True)
    return metrics
