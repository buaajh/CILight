"""Keep a newly applied steady green alive until the next MPC update.

The original actuator sets a pending target after clearance but leaves the
source program's short phase duration active. It can expire before MPC's next
decision. This separate ablation preserves yellow transitions and extends
only a steady green that the controller actually applied.
"""
import config
from controllers.executable_joint_budget_l_cimpc import ExecutionConsistentV2
from controllers.joint_budget_l_cimpc import JointBudgetLCIMPC


class HeldExecutionV2(ExecutionConsistentV2):
    def _arm_steady_green(self, tl, target):
        if self.pending_phase.get(tl) is None and target in self.topology[tl].green_phases:
            self.conn.trafficlight.setPhaseDuration(tl, config.CIMPC_INTERVAL+config.STEP_LENGTH)

    def _apply_pending_after_clearance(self, tl):
        target=self.pending_phase.get(tl)
        super()._apply_pending_after_clearance(tl)
        if target is not None and self.pending_phase.get(tl) is None:
            self._arm_steady_green(tl,target)

    def _safe_set_phase(self, tl, phase_idx):
        super()._safe_set_phase(tl,phase_idx)
        self._arm_steady_green(tl,int(phase_idx)%self.graph.num_phases[tl])


class HeldJointBudgetLCIMPC(JointBudgetLCIMPC,HeldExecutionV2):
    pass
