"""Fresh terminal-critic training with the original validation-only stop rule."""
import copy
import math
from pathlib import Path
import random
import shutil
import statistics

from cilight.io import save, sha, verify_result
from cilight.runtime import run_episode
from cilight.selection_policy import Policy, decide, validate_record


def beta_for_episode(episode):
    return 0. if episode <= 2 else {3: .05, 4: .10, 5: .15}.get(episode, .20)


def select(history, policy):
    # Historical launch_v1 adapter: retain actual teleport counts in raw history,
    # but exclude this diagnostic from checkpoint selection.
    for record in history:
        validate_record(record, policy)
    decisions = copy.deepcopy(history)
    for record in decisions:
        for row in record['seeds']:
            row['teleports'] = 0
    return decide(decisions, policy)


def train(settings, source, output):
    import numpy as np
    import torch
    from rl.terminal_value import TerminalCritic, fit_terminal
    output = Path(output).resolve()
    model_seed = settings['model_seed']
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    model = TerminalCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    policy = Policy(tuple(settings['validation_seeds']), tuple(settings['test_seeds']),
                    tuple(range(settings['seed_base']+1, settings['seed_base']+301)), 300)
    protocol_sha = sha(output/'protocol.json')
    history, records = [], []
    (output/'checkpoints').mkdir()
    for episode in range(1, 301):
        k = episode - 1
        torch.manual_seed(model_seed + 100*k)
        model.train()
        metrics, rows = run_episode(settings, source, output/'training'/f'episode_{episode:04d}',
                                    policy.training_seeds[k], model, beta_for_episode(episode), stop=output/'STOP')
        fit = fit_terminal(model, optimizer, rows, settings['update_seed']+k)
        if not math.isfinite(fit['loss']):
            raise ValueError('Non-finite critic update')
        reward = -100. * sum(float(np.sum(row['cost'])) for row in rows)
        records.append(dict(episode=episode, reward=reward, update=fit, metrics=metrics))
        checkpoint = output/'checkpoints'/f'episode_{episode:04d}.pt'
        tmp = checkpoint.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), completed=episode,
                        torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                        network=settings['network'], protocol_sha256=protocol_sha, records=records), tmp)
        tmp.replace(checkpoint)
        save(output/'training_records.json', records)
        if episode % 10:
            continue
        model.eval()
        observations = []
        for seed in policy.validation_seeds:
            directory = output/'validation'/f'episode_{episode:04d}'/f'seed_{seed}'
            m, _ = run_episode(settings, source, directory, seed, model, .5, stop=output/'STOP')
            verify_result(directory)
            marker = directory/'complete.json'
            observations.append(dict(seed=seed, att=m['average_trip_time_s'], awt=m['average_waiting_time_s'],
                                     adt=m['average_delay_time_s'], completed=m['completed_arrivals'], demand=m['demand'],
                                     teleports=m['teleports'], artifact=str(marker.relative_to(output)), artifact_sha256=sha(marker)))
        history.append(dict(episode=episode, split='validation', status='complete', artifact_verified=True,
                            checkpoint_sha256=sha(checkpoint), protocol_sha256=protocol_sha, seeds=observations,
                            training_reward_mean=statistics.mean(r['reward'] for r in records[-20:])))
        for record in history:
            for row in record['seeds']:
                marker = output/row['artifact']
                if sha(marker) != row['artifact_sha256']:
                    raise ValueError('Prior validation marker changed')
                verify_result(marker.parent)
        decision = select(history, policy)
        save(output/'validation_history.json', history)
        save(output/'early_stop_decision.json', decision)
        if decision['action'] == 'stop_at_committed_checkpoint':
            selected = output/'checkpoints'/f"episode_{decision['selected_episode']:04d}.pt"
            target = output/'selected.pt'
            shutil.copy2(selected, target)
            save(output/'selection.json', dict(network=settings['network'], selected_episode=decision['selected_episode'],
                                              checkpoint='selected.pt', sha256=sha(target), last_episode=episode,
                                              test_used_for_selection=False, reason=decision['reason']))
            print('Training complete. Selected checkpoint: ' + str(target), flush=True)
            return
    raise RuntimeError('No valid terminal selection decision')
