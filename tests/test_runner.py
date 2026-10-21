"""Mocked TraCI lifecycle test; never starts SUMO."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cilight.io import spec, verify_result


@unittest.skipUnless(os.environ.get('SUMO_HOME'), 'SUMO_HOME/tools required for helper import; no simulation is run')
class MockedRunnerTests(unittest.TestCase):
    def test_episode_outputs_and_native_close(self):
        import utils.sumo_utils  # honors SUMO_HOME/tools before importing traci
        import traci.constants as tc
        from cilight.runtime import run_episode
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'scenario.sumocfg'
            source.write_text('<configuration><input><net-file value="net.xml"/><route-files value="route.xml"/></input></configuration>')
            (root/'net.xml').write_text('<net><edge id="e"><lane id="e_0" index="0"/></edge><connection from="e" fromLane="0" tl="tl"/></net>')
            (root/'route.xml').write_text('<routes><vehicle id="v" depart="0"/></routes>')
            destination = root/'run'
            calls = []

            class FakeConnection:
                def __init__(self):
                    self.now = 0
                    self.trafficlight = SimpleNamespace(getIDList=lambda: ['tl'])
                    self.simulation = SimpleNamespace(getTime=lambda: self.now, getArrivedNumber=lambda: int(self.now == 10),
                                                      getStartingTeleportNumber=lambda: 0)
                    self.lane = SimpleNamespace(subscribe=lambda *a: None,
                        getAllSubscriptionResults=lambda: {'e_0': {tc.LAST_STEP_VEHICLE_HALTING_NUMBER: 1}})
                    self._process = SimpleNamespace(wait=lambda timeout: 0)

                def getVersion(self):
                    return 22, 'SUMO 1.27.1 (mock)'

                def simulationStep(self):
                    self.now += 1

                def close(self, wait):
                    calls.append(('close', wait))
                    (destination/'statistics.xml').write_text('<statistics><vehicleTripStatistics count="1" totalTravelTime="10" waitingTime="2" timeLoss="3"/></statistics>')
                    (destination/'tripinfo.xml').write_text('<tripinfos><tripinfo duration="10" waitingTime="2" timeLoss="3"/></tripinfos>')

            class FakeController:
                def __init__(self, *args, **kwargs):
                    self.diagnostics = []
                    self.actual_phase_trace = []
                    self.local_diagnostics = [{'success': True}]

                def step(self, t):
                    if t == 0:
                        self.diagnostics.append(dict(simulation_time=t, phase_choices='tl:0', decision_time_s=.001))
                    self.actual_phase_trace.append(dict(time=t, nodes={'tl': dict(phase=0, pending=None, yellow=False)}))

                def close(self):
                    calls.append(('controller_close', True))

            settings = spec('grid16')
            settings['signal_count'] = 1
            connection = FakeConnection()
            with patch('cilight.runtime.admission'), patch('utils.sumo_utils.start_sumo', return_value=connection) as start, \
                    patch('controllers.terminal_value.TerminalValueCIMPC', FakeController):
                metrics = run_episode(settings, source, destination, 1, seconds=10)
            self.assertTrue(start.call_args.kwargs['tripinfo_write_unfinished'])
            self.assertEqual(calls, [('controller_close', True), ('close', True)])
            self.assertEqual(metrics['samples'], 10)
            self.assertEqual(metrics['decision_count'], 1)
            self.assertEqual(metrics['demand'], 1)
            self.assertFalse((destination/'learning.pt').exists())
            self.assertEqual(verify_result(destination)['average_trip_time_s'], 10)


if __name__ == '__main__':
    unittest.main()
