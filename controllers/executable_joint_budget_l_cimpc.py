"""Execution-consistent nominal witness, separate from frozen original V2.

This fixes prediction/actuation consistency only; it does not certify stale
neighbor flows or establish recursive feasibility. Total budget is unchanged.
"""
import config
from controllers.structured_l_cimpc import FastCertificateLCIMPC
from controllers.joint_budget_l_cimpc import JointBudgetLCIMPC


def executable_phase(proposed, current, elapsed_green, min_green, pending=False):
    """Actuator cannot change green before minimum dwell or during clearance."""
    if pending or elapsed_green < min_green:
        return int(current)
    return int(proposed)


class PhaseObservationMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actual_phase_trace = []

    def step(self, sim_time):
        elapsed = super().step(sim_time)
        # Observe after the controller call, before the next SUMO integration.
        self.actual_phase_trace.append(dict(time=float(sim_time), nodes={
            t: dict(phase=int(self.conn.trafficlight.getPhase(t)),
                    pending=self.pending_phase.get(t), elapsed_green=float(self.time_in_phase[t]),
                    yellow='y' in self.conn.trafficlight.getRedYellowGreenState(t))
            for t in self.tls_ids}))
        return elapsed


class ExecutionConsistentV2(FastCertificateLCIMPC):
    """Normalize nominal phases before downstream budget/control coordination.

Use the EXISTING fixed-phase inner solution for a forced hold. Do not clip
flow, modify saturation/receiving constraints, recompute/loosen total budget,
or update the robust error EWMA a second time.
"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.execution_repairs = []

    def _postprocess_phase_choices(self, sim_time, queues, current_green, forecasts,
                                   phase_choices, payloads, solutions):
        base, result = super()._postprocess_phase_choices(
            sim_time, queues, current_green, forecasts, phase_choices, payloads, solutions)
        executable = {t: executable_phase(base[t], current_green[t], self.time_in_phase[t],
                      config.CIMPC_MIN_GREEN, self.pending_phase.get(t) is not None) for t in self.order}
        repaired = [t for t in self.order if executable[t] != base[t]]
        previous_prediction = float(self.last_lyapunov_applied)
        if repaired:
            result = {t: result[t].copy() for t in self.order}
            for t in repaired:
                result[t][:, 0] = self._first_phase_flow(t, solutions[t], executable[t], forecasts)
            flows = {t: result[t][:, 0] for t in self.order}
            self.last_lyapunov_applied = self._network_queue_lyapunov(queues, forecasts, flows)
            self._finalize_lyapunov_projection(self.last_lyapunov_applied)
            self.last_safety_violation = max(self.last_lyapunov_applied-self.last_lyapunov_limit, 0.)
        self.execution_repairs.append(dict(time=float(sim_time), repaired_nodes=repaired,
            proposed={t:int(base[t]) for t in self.order}, executable=executable,
            original_prediction=previous_prediction, executed_prediction=float(self.last_lyapunov_applied),
            unchanged_total_budget=float(self.last_lyapunov_limit)))
        return executable, result


class ExecutableJointBudgetLCIMPC(JointBudgetLCIMPC, ExecutionConsistentV2):
    """MRO installs the execution-consistent witness BEFORE joint optimization."""
    pass
