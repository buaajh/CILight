import ast
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch

from cilight.aql import AQL
from cilight.io import ROOT, NETWORKS, isolated_config, save, sha, spec, verify_code, verify_result
from cilight.runtime import demand_count
from cilight.summary import FIELDS, summarize
from rl.terminal_value import TerminalCritic, NumpyTerminalValue, terminal_features, GAMMA, COST_SCALE
from utils.sumo_statistics import summarize_statistics, verify_statistics_against_trips


class ReleaseTests(unittest.TestCase):
    def test_source_and_weight_hashes(self):
        verify_code()

    def test_all_python_syntax(self):
        for path in ROOT.rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))

    def test_seed_panels(self):
        for name in NETWORKS:
            cfg = spec(name)
            self.assertEqual(len(cfg['test_seeds']), 4)
            self.assertEqual(len(set(cfg['test_seeds'])), 4)
            self.assertTrue(all(isinstance(seed, int) and seed >= 0 for seed in cfg['test_seeds']))

    def test_exported_weights_are_numeric_and_bound(self):
        import hashlib
        from cilight.io import read
        manifest = read(ROOT/'provenance/sources.json')['files']
        for name in NETWORKS:
            cfg = spec(name)
            item = next(r for r in manifest if r['path'] == cfg['weights'])
            with np.load(ROOT/cfg['weights'], allow_pickle=False) as arrays:
                self.assertEqual(len(arrays.files), 6)
                for key in arrays.files:
                    self.assertEqual(arrays[key].dtype, np.float32)
                    self.assertTrue(np.isfinite(arrays[key]).all())
                    self.assertEqual(hashlib.sha256(arrays[key].tobytes()).hexdigest(), item['tensor_hashes'][key])

    def test_cli_is_evaluation_only(self):
        from run import parser
        import argparse
        cli = parser()
        commands = next(a for a in cli._actions if isinstance(a, argparse._SubParsersAction))
        self.assertEqual(set(commands.choices), {'check', 'evaluate', 'summarize'})
        args = cli.parse_args(['evaluate', '--network', 'grid16', '--output', 'runs/test'])
        self.assertEqual(args.data_root, Path(__import__('os').environ.get('CILIGHT_DATA_ROOT', ROOT/'scenarios')))
        self.assertEqual(args.seconds, 3600)

    def test_critic_terminal_zero(self):
        model = TerminalCritic()
        features = torch.ones(5, 12)
        features[:, -1] = 0
        self.assertTrue(torch.equal(model(features), torch.zeros(5)))

    def test_numpy_critic_matches_torch(self):
        from types import SimpleNamespace
        torch.manual_seed(7)
        model = TerminalCritic()
        topo = SimpleNamespace(capacity=np.array([12., 20.]), service=np.eye(2))
        queue, arrival = np.array([5., 3.]), np.array([1., 2.])
        features = terminal_features(topo, queue, arrival, 0, 120.)
        expected = .5 * GAMMA**3 * COST_SCALE * float(np.clip(model(torch.tensor(features)).detach().numpy(), 0, 100))
        actual = NumpyTerminalValue(model.numpy_weights(), .5, 120., 3)(topo, queue, arrival, 0)
        self.assertAlmostEqual(expected, actual, places=5)
        self.assertEqual(NumpyTerminalValue(model.numpy_weights(), 0, 120., 3)(topo, queue, arrival, 0), 0)
        self.assertEqual(NumpyTerminalValue(model.numpy_weights(), .5, 0., 3)(topo, queue, arrival, 0), 0)

    def test_final_weight_loading_is_bound_and_frozen(self):
        from cilight.runtime import load_critic
        for name in NETWORKS:
            cfg = spec(name)
            model = load_critic(ROOT/cfg['weights'], cfg['weights_sha256'])
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            with np.load(ROOT/cfg['weights'], allow_pickle=False) as arrays:
                for key, value in model.state_dict().items():
                    np.testing.assert_array_equal(value.numpy(), arrays[key])
            with self.assertRaises(ValueError):
                load_critic(ROOT/cfg['weights'], '0'*64)
        with self.assertRaises(ValueError):
            load_critic(ROOT/'unknown.pt', '0'*64)

    def test_aql_and_missing_sample(self):
        aql = AQL(('a', 'b'))
        aql.add(1., {'a': 2, 'b': 0})
        aql.add(2., {'a': 2, 'b': 2})
        self.assertEqual(aql.result(2.)['aql_vehicles_per_lane'], 1.5)
        with self.assertRaises(ValueError):
            aql.add(4., {'a': 0, 'b': 0})

    def test_native_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p/'stats.xml').write_text('<statistics><vehicleTripStatistics count="2" totalTravelTime="60" waitingTime="5" timeLoss="8"/></statistics>')
            (p/'trip.xml').write_text('<tripinfos><tripinfo duration="20" waitingTime="4" timeLoss="6"/><tripinfo duration="40" waitingTime="6" timeLoss="10"/></tripinfos>')
            m = summarize_statistics(p/'stats.xml')
            verify_statistics_against_trips(m, p/'trip.xml')
            self.assertEqual(m['average_trip_time_s'], 30.)
            m['average_trip_time_s'] = 31.
            with self.assertRaises(ValueError):
                verify_statistics_against_trips(m, p/'trip.xml')

    def test_isolated_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p/'in').mkdir()
            (p/'out').mkdir()
            cfg = p/'in'/'test.sumocfg'
            cfg.write_text('<configuration><input><net-file value="net.xml"/><route-files value="route.xml"/><additional-files value="det.xml"/></input></configuration>')
            extra = p/'in'/'det.xml'
            extra.write_text('<additional><e1Detector id="a" file="shared.xml"/><e1Detector id="b" file="shared.xml"/></additional>')
            before = sha(extra)
            target = isolated_config(cfg, p/'out')
            result = ET.parse(target)
            self.assertTrue(Path(result.find('input/net-file').get('value')).is_absolute())
            detectors = ET.parse(p/'out'/'additional_0.xml').getroot()
            self.assertEqual(detectors[0].get('file'), detectors[1].get('file'))
            self.assertEqual(Path(detectors[0].get('file')).parent, p/'out')
            self.assertEqual(sha(extra), before)

    def test_demand_time_origin(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p/'c.xml').write_text('<configuration><input><route-files value="r.xml"/></input><time><begin value="100"/></time></configuration>')
            (p/'r.xml').write_text('<routes><vehicle id="x" depart="100"/><vehicle id="y" depart="110"/><flow id="f" begin="100" end="110" period="2"/></routes>')
            self.assertEqual(demand_count(p/'c.xml', 10), 6)

    def test_summary_uses_sample_standard_deviation_and_verifies(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            for i, value in enumerate((1., 2., 3., 4.)):
                d = p/'cilight'/f'seed_{i}'
                row = dict(network='grid16', method='CILight', seed=i, seconds=3600, **{k: value for k in FIELDS})
                save(d/'metrics.json', row)
                save(d/'complete.json', {'sha256': {'metrics.json': sha(d/'metrics.json')}})
            rows = summarize(p, p/'summary.csv')
            self.assertAlmostEqual(rows[0]['average_trip_time_s_sample_std'], math.sqrt(5/3))
            self.assertEqual(rows[0]['average_trip_time_s_mean'], 2.5)
            save(p/'cilight/seed_0/metrics.json', {})
            with self.assertRaises(ValueError):
                verify_result(p/'cilight/seed_0')

    def test_summary_rejects_partial_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p/'cilight/seed_1').mkdir(parents=True)
            with self.assertRaises(ValueError):
                summarize(p, p/'out.csv')


if __name__ == '__main__':
    unittest.main()
