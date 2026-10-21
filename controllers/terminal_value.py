"""CILight terminal-value components extracted without algorithm changes."""
import numpy as np
import config
from controllers.held_executable_joint_l_cimpc import HeldExecutionV2
from controllers.executable_joint_budget_l_cimpc import PhaseObservationMixin
from rl.terminal_value import terminal_features, NumpyTerminalValue, COST_SCALE


class TerminalValueCIMPC(PhaseObservationMixin, HeldExecutionV2):
    def __init__(self, conn, critic=None, beta=0., seconds=3600, progress=None):
        super().__init__(conn)
        self.beta, self.seconds, self.progress = float(beta), seconds, progress
        self.weights = critic.numpy_weights() if critic is not None else None
        self.learning_rows = []
        self.ab_diagnostics = []

    def _adaptive_parameters_for_update(self, sim_time, queues, forecasts, current_green):
        features, costs = [], []
        for tl in self.order:
            topo = self.topology[tl]
            arrival = self._arrival_forecast(tl, forecasts)[:, 0]
            features.append(terminal_features(topo, queues[tl], arrival,
                                              current_green[tl], self.seconds-sim_time))
            occ = queues[tl]/np.maximum(topo.capacity, 1.)
            # Tail target is queue congestion in the same units as the queue
            # portion of the original objective; it excludes flow rewards.
            costs.append(((config.CIMPC_QUEUE_WEIGHT+config.CIMPC_TERMINAL_QUEUE_WEIGHT)
                *np.sum(occ**2)+config.CIMPC_BALANCE_WEIGHT*np.sum(occ**4)
                +config.CIMPC_PROJECTION_LINEAR_QUEUE_WEIGHT*np.sum(queues[tl]))/COST_SCALE)
        self.learning_rows.append({'time': float(sim_time), 'features': np.asarray(features),
                                   'cost': np.asarray(costs)})
        if self.progress is not None and int(sim_time) % 100 == 0:
            self.progress(sim_time)
        if self.weights is None or self.beta == 0:
            return None
        return {'terminal_value': NumpyTerminalValue(self.weights, self.beta,
            self.seconds-sim_time-config.CIMPC_HORIZON*config.CIMPC_INTERVAL,
            config.CIMPC_HORIZON)}

