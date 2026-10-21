"""Safety-filtered external residual reinforcement learning for CI-MPC.

CI-MPC remains the nominal action provider.  The learned policy only reranks
Stage-5 first-phase candidates inside a small MPC-cost budget.  A deterministic
shield enforces signal timing and refuses candidates that worsen predicted
queue overflow or violate the already-tightened downstream receiving bounds.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass

import numpy as np

import config
from controllers.ci_mpc import CIMPCController


@dataclass
class ResidualProjection:
    phase: int
    proposed_phase: int
    reason: str
    eligible: np.ndarray
    cost_budget: float
    budget_type: str = "mpc_cost"
    lyapunov_limit: float = np.nan
    base_lyapunov: float = np.nan
    applied_lyapunov: float = np.nan


def queue_lyapunov(queue, capacity, overflow_weight=0.0):
    """Dimensionless quadratic queue/storage Lyapunov function."""

    queue = np.maximum(np.asarray(queue, dtype=float), 0.0)
    capacity = np.maximum(np.asarray(capacity, dtype=float), 1.0)
    occupancy = queue / capacity
    overflow = np.maximum(occupancy - 1.0, 0.0)
    return float(
        np.sum(occupancy ** 2)
        + float(overflow_weight) * np.sum(overflow ** 2)
    )


def lyapunov_budget_limit(
    current_value,
    nominal_next_value,
    descent_retention=0.5,
    slack=0.0,
):
    """Preserve a fraction of the nominal MPC one-step Lyapunov decrease.

    If nominal CI-MPC decreases V, the returned limit retains at least
    ``descent_retention`` of that decrease (up to ``slack``).  If nominal
    CI-MPC does not decrease V, a corrected action may not worsen its predicted
    next value by more than ``slack``.
    """

    retention = float(descent_retention)
    if not 0.0 <= retention <= 1.0:
        raise ValueError("descent_retention must be in [0, 1]")
    current = float(current_value)
    nominal = float(nominal_next_value)
    if not np.isfinite(current) or not np.isfinite(nominal):
        raise ValueError("Lyapunov values must be finite")
    nominal_decrease = max(current - nominal, 0.0)
    return float(
        nominal + (1.0 - retention) * nominal_decrease + max(float(slack), 0.0)
    )


def project_residual_phase(
    base_phase,
    candidate_costs,
    residual_scores,
    feasible_mask,
    current_phase,
    elapsed_green,
    min_green,
    max_green,
    control_interval=0.0,
    relative_cost_budget=0.05,
    absolute_cost_budget=1.0,
    residual_gain=1.0,
    candidate_lyapunov_values=None,
    lyapunov_limit=None,
):
    """Project a bounded residual action onto safe near-MPC candidates."""

    costs = np.asarray(candidate_costs, dtype=float).reshape(-1)
    scores = np.asarray(residual_scores, dtype=float).reshape(-1)
    feasible = np.asarray(feasible_mask, dtype=bool).reshape(-1)
    if costs.size == 0 or scores.size != costs.size or feasible.size != costs.size:
        raise ValueError("candidate costs, scores, and mask must have equal size")
    base_phase = int(base_phase)
    current_phase = int(current_phase)
    if not 0 <= base_phase < costs.size or not 0 <= current_phase < costs.size:
        raise ValueError("base/current phase is outside the candidate vector")

    use_lyapunov_budget = candidate_lyapunov_values is not None
    lyapunov = None
    if use_lyapunov_budget:
        lyapunov = np.asarray(candidate_lyapunov_values, dtype=float).reshape(-1)
        if lyapunov.size != costs.size:
            raise ValueError("candidate Lyapunov values must match candidate costs")
        if lyapunov_limit is None or not np.isfinite(float(lyapunov_limit)):
            raise ValueError("a finite Lyapunov limit is required")
    hard = feasible & np.isfinite(costs)
    if use_lyapunov_budget:
        hard &= np.isfinite(lyapunov)
    proposed = int(np.argmax(np.where(hard, scores, -np.inf))) if np.any(hard) else base_phase

    if float(elapsed_green) + 1e-9 < float(min_green):
        eligible = np.zeros_like(hard)
        if hard[current_phase]:
            eligible[current_phase] = True
            return ResidualProjection(
                current_phase,
                proposed,
                "min_green_hold",
                eligible,
                0.0,
                budget_type="lyapunov" if use_lyapunov_budget else "mpc_cost",
                lyapunov_limit=(
                    float(lyapunov_limit) if use_lyapunov_budget else np.nan
                ),
                base_lyapunov=(
                    float(lyapunov[base_phase]) if use_lyapunov_budget else np.nan
                ),
                applied_lyapunov=(
                    float(lyapunov[current_phase]) if use_lyapunov_budget else np.nan
                ),
            )

    # Decisions occur on a coarse receding-horizon clock.  Anticipate the
    # next interval so the physical green cannot overshoot the hard maximum
    # between two decision instants.
    maximum_reached = (
        float(elapsed_green) + float(control_interval)
        > float(max_green) + 1e-9
    )
    if maximum_reached and hard.size > 1:
        hard[current_phase] = False

    if not np.any(hard):
        fallback = base_phase if feasible[base_phase] else current_phase
        return ResidualProjection(
            fallback,
            proposed,
            "no_safe_candidate",
            hard,
            0.0,
            budget_type="lyapunov" if use_lyapunov_budget else "mpc_cost",
            lyapunov_limit=(
                float(lyapunov_limit) if use_lyapunov_budget else np.nan
            ),
            base_lyapunov=(
                float(lyapunov[base_phase]) if use_lyapunov_budget else np.nan
            ),
            applied_lyapunov=(
                float(lyapunov[fallback]) if use_lyapunov_budget else np.nan
            ),
        )

    if use_lyapunov_budget:
        limit = float(lyapunov_limit)
        eligible = hard & (lyapunov <= limit + 1e-12)
        base_lyapunov = float(lyapunov[base_phase])
        lyapunov_budget = max(limit - base_lyapunov, 0.0)
        if not np.any(eligible):
            # Timing can remove the nominal hold phase at maximum green even
            # when no alternative satisfies the Lyapunov certificate.  Choose
            # the least-V safe alternative, but expose the relaxed certificate
            # explicitly in diagnostics rather than silently claiming it.
            choice = int(np.argmin(np.where(hard, lyapunov, np.inf)))
            return ResidualProjection(
                choice,
                proposed,
                "max_green_lyapunov_relaxed"
                if maximum_reached
                else "no_lyapunov_candidate",
                hard,
                lyapunov_budget,
                budget_type="lyapunov",
                lyapunov_limit=limit,
                base_lyapunov=base_lyapunov,
                applied_lyapunov=float(lyapunov[choice]),
            )
        if np.max(np.abs(scores)) <= 1e-12 and eligible[base_phase]:
            choice = base_phase
        else:
            denominator = max(lyapunov_budget, 1e-9)
            normalized_gap = np.maximum(lyapunov - base_lyapunov, 0.0) / denominator
            combined = normalized_gap - float(residual_gain) * np.clip(
                scores, -1.0, 1.0
            )
            choice = int(np.argmin(np.where(eligible, combined, np.inf)))
        if maximum_reached:
            reason = "max_green_force"
        elif choice == base_phase:
            reason = "mpc_retained"
        else:
            reason = "rl_override"
        return ResidualProjection(
            choice,
            proposed,
            reason,
            eligible,
            lyapunov_budget,
            budget_type="lyapunov",
            lyapunov_limit=limit,
            base_lyapunov=base_lyapunov,
            applied_lyapunov=float(lyapunov[choice]),
        )

    reference_cost = float(costs[base_phase])
    if not np.isfinite(reference_cost):
        reference_cost = float(np.min(costs[hard]))
    budget = max(
        float(absolute_cost_budget),
        float(relative_cost_budget) * max(abs(reference_cost), 1.0),
    )
    eligible = hard & (costs <= reference_cost + budget + 1e-12)
    if not np.any(eligible):
        # Maximum green is a hard timing constraint. If its removal of the
        # current phase empties the near-MPC set, use the least-cost remaining
        # safe alternative instead of silently extending green forever.
        choice = int(np.argmin(np.where(hard, costs, np.inf)))
        reason = "max_green_force" if maximum_reached else "safe_fallback"
        return ResidualProjection(choice, proposed, reason, hard, budget)

    normalized_gap = np.maximum(costs - reference_cost, 0.0) / max(budget, 1e-9)
    combined = normalized_gap - float(residual_gain) * np.clip(scores, -1.0, 1.0)
    choice = int(np.argmin(np.where(eligible, combined, np.inf)))
    if maximum_reached:
        reason = "max_green_force"
    elif choice == base_phase:
        reason = "mpc_retained"
    else:
        reason = "rl_override"
    return ResidualProjection(choice, proposed, reason, eligible, budget)


class _ResidualPolicy:
    def __init__(self, action_size, model_path=None, deterministic=True):
        self.action_size = int(action_size)
        self.model = None
        self.deterministic = bool(deterministic)
        if model_path:
            candidate_exists = os.path.isfile(model_path) or os.path.isfile(
                f"{model_path}.zip"
            )
            if not candidate_exists:
                raise FileNotFoundError(
                    f"External RL-CI-MPC model not found: {model_path}"
                )
            from stable_baselines3 import PPO

            self.model = PPO.load(model_path)

    @property
    def source(self):
        return "ppo" if self.model is not None else "zero_residual_fallback"

    def predict(self, observation):
        if self.model is None:
            return np.zeros(self.action_size, dtype=np.float32)
        action, _ = self.model.predict(
            np.asarray(observation, dtype=np.float32),
            deterministic=self.deterministic,
        )
        return np.asarray(action, dtype=np.float32).reshape(-1)


class ExternalCooperativeCIMPCController(CIMPCController):
    """Stage-5 CI-MPC followed by a bounded, hard-shielded RL correction."""

    name = "External_RL_CI_MPC"
    diagnostic_prefix = "external_rl_ci_mpc_solver"
    features_per_junction = 12

    def __init__(
        self,
        conn,
        model_path=None,
        external_actions=False,
        budget_mode=None,
    ):
        if config.CIMPC_IMPLEMENTATION_STAGE != 5:
            raise ValueError("External cooperative CI-MPC requires --cimpc-stage 5")
        self.external_actions = bool(external_actions)
        self.budget_mode = str(
            config.EXTERNAL_CIMPC_BUDGET_MODE
            if budget_mode is None
            else budget_mode
        )
        if self.budget_mode not in ("cost", "lyapunov"):
            raise ValueError("budget_mode must be 'cost' or 'lyapunov'")
        self.external_diagnostics = []
        self.last_external_observation = None
        self.last_external_action = None
        self.last_correction_rate = 0.0
        self.last_blocked_rate = 0.0
        self.last_safety_violation = 0.0
        self.last_correction_by_tl = {}
        self.last_lyapunov_nominal_limit = np.nan
        self.last_lyapunov_robust_reserve = 0.0
        self.last_lyapunov_prediction_error = 0.0
        self.last_lyapunov_prediction_error_ewma = 0.0
        self.previous_lyapunov_prediction = np.nan
        super().__init__(conn)

        self.action_slices = {}
        offset = 0
        for tl in sorted(self.tls_ids):
            count = len(self.topology[tl].green_phases)
            self.action_slices[tl] = slice(offset, offset + count)
            offset += count
        self.action_size = offset
        self.external_action = np.zeros(self.action_size, dtype=np.float32)
        self.last_external_action = self.external_action.copy()
        self.residual_policy = None
        if not self.external_actions:
            self.residual_policy = _ResidualPolicy(
                self.action_size,
                model_path=model_path or config.EXTERNAL_CIMPC_MODEL_PATH,
                deterministic=config.EXTERNAL_CIMPC_DETERMINISTIC,
            )

    @property
    def observation_size(self):
        return len(self.tls_ids) * self.features_per_junction + 2

    def set_external_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != self.action_size:
            raise ValueError(
                f"external residual action has {action.size} entries; "
                f"expected {self.action_size}"
            )
        self.external_action = np.clip(action, -1.0, 1.0)

    def build_external_observation(
        self, sim_time, queues, forecasts, current_green
    ):
        features = []
        per_step_capacity = max(
            config.CIMPC_SATURATION_PER_LANE * config.CIMPC_INTERVAL, 1.0
        )
        for tl in sorted(self.tls_ids):
            topo = self.topology[tl]
            queue = np.maximum(np.asarray(queues[tl], dtype=float), 0.0)
            occupancy = queue / np.maximum(topo.capacity, 1.0)
            arrivals = self._arrival_forecast(tl, forecasts)[:, 0]
            arrivals_scaled = np.clip(
                arrivals / (2.0 * per_step_capacity), 0.0, 1.0
            )
            receiving = self._receiving_capacity_forecast(tl, queues, forecasts)
            ratios = []
            for group_index, group in enumerate(topo.receiving_groups):
                downstream = self.topology[group.downstream_tl]
                storage = float(np.sum(downstream.capacity[group.target_indices]))
                ratios.append(
                    np.clip(
                        float(receiving[group_index, 0]) / max(storage, 1.0),
                        0.0,
                        1.0,
                    )
                )
            ratios = np.asarray(ratios if ratios else [1.0], dtype=float)
            phase_denominator = max(len(topo.green_phases) - 1, 1)
            features.extend(
                [
                    float(np.mean(np.clip(occupancy / 2.0, 0.0, 1.0))),
                    float(np.max(np.clip(occupancy / 2.0, 0.0, 1.0), initial=0.0)),
                    float(
                        np.clip(
                            np.sum(queue) / max(2.0 * np.sum(topo.capacity), 1.0),
                            0.0,
                            1.0,
                        )
                    ),
                    float(np.mean(arrivals_scaled)),
                    float(np.max(arrivals_scaled, initial=0.0)),
                    float(np.min(ratios, initial=1.0)),
                    float(np.mean(ratios)),
                    float(np.mean(occupancy >= config.RLCIMPC_SAFE_OCCUPANCY)),
                    float(np.mean(ratios <= 0.10)),
                    float(current_green[tl] / phase_denominator),
                    float(
                        np.clip(
                            self.time_in_phase[tl]
                            / max(config.EXTERNAL_CIMPC_MAX_GREEN, 1.0),
                            0.0,
                            1.0,
                        )
                    ),
                    float(self.last_correction_by_tl.get(tl, 0.0)),
                ]
            )
        angle = 2.0 * math.pi * (float(sim_time) % 3600.0) / 3600.0
        features.extend([math.sin(angle), math.cos(angle)])
        observation = np.asarray(features, dtype=np.float32)
        if observation.size != self.observation_size:
            raise RuntimeError(
                f"external observation size {observation.size} != "
                f"{self.observation_size}"
            )
        return observation

    def observe_rl_state(self, sim_time):
        queues = self._measure_queues()
        current_green = {
            tl: self._current_green_index(tl) for tl in self.tls_ids
        }
        forecasts = {
            tl: np.hstack((flow[:, 1:], flow[:, -1:]))
            for tl, flow in self.flow_forecast.items()
        }
        return self.build_external_observation(
            sim_time, queues, forecasts, current_green
        )

    def _first_phase_flow(self, tl, solution, phase, forecasts):
        """Return the certified first-step flow for one local green phase."""

        candidate_flows = np.asarray(solution.first_phase_flows, dtype=float)
        expected = (len(self.topology[tl].lanes), len(self.topology[tl].green_phases))
        if candidate_flows.shape == expected and 0 <= int(phase) < expected[1]:
            flow = candidate_flows[:, int(phase)]
            if np.all(np.isfinite(flow)):
                return np.maximum(flow, 0.0).copy()
        return np.maximum(np.asarray(forecasts[tl][:, 0], dtype=float), 0.0).copy()

    def _compute_lyapunov_limit(self, current_value, nominal_next_value):
        """Return the active Lyapunov limit; subclasses may robustly tighten it."""

        limit = lyapunov_budget_limit(
            current_value,
            nominal_next_value,
            config.EXTERNAL_CIMPC_LYAPUNOV_DESCENT_RETENTION,
            config.EXTERNAL_CIMPC_LYAPUNOV_SLACK,
        )
        self.last_lyapunov_nominal_limit = float(limit)
        self.last_lyapunov_robust_reserve = 0.0
        self.last_lyapunov_prediction_error = 0.0
        return float(limit)

    def _finalize_lyapunov_projection(self, applied_value):
        """Persist the one-step prediction for next-cycle model-error auditing."""

        self.previous_lyapunov_prediction = float(applied_value)

    def _network_queue_lyapunov(self, queues, forecasts, first_flows=None):
        """Predict the next network queue Lyapunov value from first-step flows.

        Internal transfers are reconstructed through the same routing matrices
        used by CI-MPC, so moving a queue to a downstream junction is not
        mistaken for removing it from the controlled network.
        """

        if first_flows is None:
            return float(
                sum(
                    queue_lyapunov(
                        queues[tl],
                        self.topology[tl].capacity,
                        config.EXTERNAL_CIMPC_LYAPUNOV_OVERFLOW_WEIGHT,
                    )
                    for tl in self.tls_ids
                )
            )
        trial_forecasts = {tl: flow.copy() for tl, flow in forecasts.items()}
        for tl, first_flow in first_flows.items():
            trial_forecasts[tl][:, 0] = np.asarray(first_flow, dtype=float)
        value = 0.0
        for tl in self.tls_ids:
            arrivals = self._arrival_forecast(tl, trial_forecasts)[:, 0]
            next_queue = np.maximum(
                np.asarray(queues[tl], dtype=float)
                + arrivals
                - np.asarray(first_flows[tl], dtype=float),
                0.0,
            )
            value += queue_lyapunov(
                next_queue,
                self.topology[tl].capacity,
                config.EXTERNAL_CIMPC_LYAPUNOV_OVERFLOW_WEIGHT,
            )
        return float(value)

    def _candidate_feasibility(self, tl, payload, solution, base_phase):
        topo = self.topology[tl]
        queue, arrivals, receiving = payload[1], payload[2], payload[3]
        costs = np.asarray(solution.first_phase_objectives, dtype=float)
        flows = np.asarray(solution.first_phase_flows, dtype=float)
        phase_count = len(topo.green_phases)
        if costs.size != phase_count or flows.shape != (len(queue), phase_count):
            mask = np.zeros(phase_count, dtype=bool)
            mask[int(base_phase)] = True
            return mask, np.full(phase_count, np.nan), np.full(phase_count, np.nan)

        available = np.maximum(queue + arrivals[:, 0], 0.0)
        next_queues = np.maximum(available[:, None] - flows, 0.0)
        overflows = np.sum(
            np.maximum(next_queues - topo.capacity[:, None], 0.0), axis=0
        )
        base_overflow = float(overflows[int(base_phase)])
        feasible = np.isfinite(costs) & (
            overflows
            <= base_overflow + config.EXTERNAL_CIMPC_OVERFLOW_TOLERANCE
        )
        receiving_margins = np.full(phase_count, np.inf, dtype=float)
        for phase_index in range(phase_count):
            margins = []
            for group_index, group in enumerate(topo.receiving_groups):
                discharge = float(
                    np.sum(flows[np.asarray(group.source_indices, dtype=int), phase_index])
                )
                margins.append(float(receiving[group_index, 0]) - discharge)
            if margins:
                receiving_margins[phase_index] = min(margins)
                if receiving_margins[phase_index] < -config.EXTERNAL_CIMPC_RECEIVING_TOLERANCE:
                    feasible[phase_index] = False
        # The nominal MPC choice is the certified fallback.  Numerical noise
        # in the reconstructed diagnostics must never mask it.
        feasible[int(base_phase)] = True
        return feasible, overflows, receiving_margins

    def _postprocess_phase_choices(
        self,
        sim_time,
        queues,
        current_green,
        forecasts,
        phase_choices,
        latest_payload_by_tl,
        latest_solutions,
    ):
        observation = self.build_external_observation(
            sim_time, queues, forecasts, current_green
        )
        if self.external_actions:
            action = self.external_action.copy()
            policy_source = "external_training_action"
        else:
            action = self.residual_policy.predict(observation)
            policy_source = self.residual_policy.source
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.last_external_observation = observation.copy()
        self.last_external_action = action.astype(np.float32)

        use_lyapunov_budget = self.budget_mode == "lyapunov"
        selected_first_flows = {}
        current_lyapunov = np.nan
        nominal_next_lyapunov = np.nan
        lyapunov_limit = np.nan
        if use_lyapunov_budget:
            # Build the all-nominal first-step network prediction before
            # applying learned corrections. Minimum-green holds are included
            # because they are part of the physically executable controller.
            nominal_choices = dict(phase_choices)
            for tl in self.tls_ids:
                if self.time_in_phase[tl] + 1e-9 < config.CIMPC_MIN_GREEN:
                    nominal_choices[tl] = int(current_green[tl])
            selected_first_flows = {
                tl: self._first_phase_flow(
                    tl, latest_solutions[tl], nominal_choices[tl], forecasts
                )
                for tl in self.tls_ids
            }
            current_lyapunov = self._network_queue_lyapunov(queues, forecasts)
            nominal_next_lyapunov = self._network_queue_lyapunov(
                queues, forecasts, selected_first_flows
            )
            lyapunov_limit = self._compute_lyapunov_limit(
                current_lyapunov, nominal_next_lyapunov
            )

        corrected = 0
        blocked = 0
        max_violation = 0.0
        new_choices = dict(phase_choices)
        new_forecasts = {tl: flow.copy() for tl, flow in forecasts.items()}
        correction_by_tl = {}

        for tl in sorted(self.tls_ids):
            solution = latest_solutions[tl]
            payload = latest_payload_by_tl[tl]
            base_phase = int(phase_choices[tl])
            scores = action[self.action_slices[tl]]
            feasible, overflows, receiving_margins = self._candidate_feasibility(
                tl, payload, solution, base_phase
            )
            costs = np.asarray(solution.first_phase_objectives, dtype=float)
            phase_count = len(self.topology[tl].green_phases)
            candidate_lyapunov = np.full(phase_count, np.inf, dtype=float)
            candidate_first_flows = []
            for phase_index in range(phase_count):
                candidate_flow = self._first_phase_flow(
                    tl, solution, phase_index, forecasts
                )
                candidate_first_flows.append(candidate_flow)
                if use_lyapunov_budget:
                    trial_first_flows = dict(selected_first_flows)
                    trial_first_flows[tl] = candidate_flow
                    candidate_lyapunov[phase_index] = self._network_queue_lyapunov(
                        queues, forecasts, trial_first_flows
                    )
            projection_kwargs = {}
            if use_lyapunov_budget:
                projection_kwargs = {
                    "candidate_lyapunov_values": candidate_lyapunov,
                    "lyapunov_limit": lyapunov_limit,
                }
            projection = project_residual_phase(
                base_phase=base_phase,
                candidate_costs=costs,
                residual_scores=scores,
                feasible_mask=feasible,
                current_phase=current_green[tl],
                elapsed_green=self.time_in_phase[tl],
                min_green=config.CIMPC_MIN_GREEN,
                max_green=config.EXTERNAL_CIMPC_MAX_GREEN,
                control_interval=config.CIMPC_INTERVAL,
                relative_cost_budget=config.EXTERNAL_CIMPC_RELATIVE_COST_BUDGET,
                absolute_cost_budget=config.EXTERNAL_CIMPC_ABSOLUTE_COST_BUDGET,
                residual_gain=config.EXTERNAL_CIMPC_RESIDUAL_GAIN,
                **projection_kwargs,
            )
            choice = int(projection.phase)
            new_choices[tl] = choice
            if use_lyapunov_budget:
                selected_first_flows[tl] = candidate_first_flows[choice]
            correction = int(choice != base_phase)
            corrected += correction
            correction_by_tl[tl] = correction
            blocked += int(
                projection.proposed_phase != choice
                and projection.proposed_phase != base_phase
            )

            new_forecasts[tl][:, 0] = candidate_first_flows[choice]

            selected_margin = float(receiving_margins[choice])
            selected_overflow = float(overflows[choice])
            base_overflow = float(overflows[base_phase])
            violation = max(-selected_margin, 0.0) if np.isfinite(selected_margin) else 0.0
            violation += max(selected_overflow - base_overflow, 0.0)
            lyapunov_violation = (
                max(
                    float(projection.applied_lyapunov) - float(lyapunov_limit),
                    0.0,
                )
                if use_lyapunov_budget
                else 0.0
            )
            max_violation = max(max_violation, violation, lyapunov_violation)
            self.external_diagnostics.append(
                {
                    "simulation_time": float(sim_time),
                    "tl": tl,
                    "policy_source": policy_source,
                    "mpc_phase": base_phase,
                    "rl_proposed_phase": int(projection.proposed_phase),
                    "applied_phase": choice,
                    "corrected": correction,
                    "projection_reason": projection.reason,
                    "eligible_phases": ";".join(
                        str(index)
                        for index, value in enumerate(projection.eligible)
                        if value
                    ),
                    "budget_type": projection.budget_type,
                    "projection_budget_mode": self.budget_mode,
                    "cost_budget": float(projection.cost_budget),
                    "mpc_cost": float(costs[base_phase]),
                    "applied_cost": float(costs[choice]),
                    "cost_gap": float(costs[choice] - costs[base_phase]),
                    "base_overflow": base_overflow,
                    "applied_overflow": selected_overflow,
                    "receiving_margin": selected_margin,
                    "elapsed_green": float(self.time_in_phase[tl]),
                    "lyapunov_current": current_lyapunov,
                    "lyapunov_nominal_next": nominal_next_lyapunov,
                    "lyapunov_limit": lyapunov_limit,
                    "lyapunov_nominal_limit": self.last_lyapunov_nominal_limit,
                    "lyapunov_robust_reserve": self.last_lyapunov_robust_reserve,
                    "lyapunov_prediction_error": self.last_lyapunov_prediction_error,
                    "lyapunov_prediction_error_ewma": (
                        self.last_lyapunov_prediction_error_ewma
                    ),
                    "lyapunov_candidate_base": float(
                        candidate_lyapunov[base_phase]
                    ) if use_lyapunov_budget else np.nan,
                    "lyapunov_applied": float(projection.applied_lyapunov),
                    "lyapunov_violation": lyapunov_violation,
                    "physical_safety_violation": violation,
                    "safety_violation": max(violation, lyapunov_violation),
                }
            )

        junctions = max(len(self.tls_ids), 1)
        self.last_correction_rate = corrected / junctions
        self.last_blocked_rate = blocked / junctions
        self.last_correction_by_tl = correction_by_tl
        self.last_lyapunov_current = current_lyapunov
        self.last_lyapunov_nominal_next = nominal_next_lyapunov
        self.last_lyapunov_limit = lyapunov_limit
        self.last_lyapunov_applied = (
            self._network_queue_lyapunov(queues, forecasts, selected_first_flows)
            if use_lyapunov_budget
            else np.nan
        )
        self.last_safety_violation = max_violation
        if use_lyapunov_budget:
            self.last_safety_violation = max(
                max_violation,
                self.last_lyapunov_applied - lyapunov_limit,
                0.0,
            )
            self._finalize_lyapunov_projection(self.last_lyapunov_applied)
        return new_choices, new_forecasts

    def export_diagnostics(self, output_dir):
        if self.external_diagnostics:
            path = os.path.join(output_dir, "external_cooperation_diagnostics.csv")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(self.external_diagnostics[0].keys())
                )
                writer.writeheader()
                writer.writerows(self.external_diagnostics)
        super().export_diagnostics(output_dir)
