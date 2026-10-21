"""CILight terminal-value integration for fixed-weight evaluation."""
import config
from controllers.held_executable_joint_l_cimpc import HeldExecutionV2
from controllers.executable_joint_budget_l_cimpc import PhaseObservationMixin
from rl.terminal_value import NumpyTerminalValue


class TerminalValueCIMPC(PhaseObservationMixin, HeldExecutionV2):
    def __init__(self, conn, critic=None, beta=0., seconds=3600, progress=None):
        super().__init__(conn)
        self.beta, self.seconds, self.progress = float(beta), seconds, progress
        self.weights = critic.numpy_weights() if critic is not None else None
        self.ab_diagnostics = []

    def _adaptive_parameters_for_update(self, sim_time, queues, forecasts, current_green):
        if self.progress is not None and int(sim_time) % 100 == 0:
            self.progress(sim_time)
        if self.weights is None or self.beta == 0:
            return None
        return {'terminal_value': NumpyTerminalValue(self.weights, self.beta,
            self.seconds-sim_time-config.CIMPC_HORIZON*config.CIMPC_INTERVAL,
            config.CIMPC_HORIZON)}

