"""Complementarity-Implicit distributed MPC for SUMO traffic signals.

The signal plan is the outer MPC decision.  For every candidate plan, the
movement discharge is defined implicitly by the KKT system of a strictly
concave network-flow allocation problem.  Demand, signal saturation, and
downstream storage are all represented by complementary primal/dual pairs.
The MPEC is solved as a smooth NLP with a Scholtes product relaxation and only
the first (projected) safe SUMO phase is applied.

This implementation is intentionally independent of both MPC baselines in the
repository.  In particular, it does not call the deterministic DMPC MILP or
the paper-inspired stochastic MPC implementation.
"""

from __future__ import annotations

import csv
import gc
import multiprocessing as mp
import os
import queue as queue_module
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

import config
from controllers.base_controller import BaseController


def _load_casadi():
    """Load the native NLP runtime only for stages that actually use IPOPT."""

    global ca
    if "ca" not in globals():
        import casadi as casadi_module

        ca = casadi_module


@dataclass
class ReceivingGroup:
    """One downstream link-storage constraint seen by a local junction."""

    downstream_tl: str
    target_edge: str
    source_indices: list[int]
    target_indices: list[int]


@dataclass
class CIJunctionTopology:
    tl: str
    lanes: list[str]
    green_phases: list[int]
    service: np.ndarray
    capacity: np.ndarray
    boundary_indices: list[int]
    upstream_routes: dict[str, np.ndarray] = field(default_factory=dict)
    receiving_groups: list[ReceivingGroup] = field(default_factory=list)


@dataclass
class LocalCISolution:
    success: bool
    status: str
    flow: np.ndarray
    phase: np.ndarray
    max_complementarity: float
    mean_complementarity: float
    max_stationarity_residual: float
    max_primal_violation: float
    phase_integrality_gap: float
    solve_time_s: float
    iterations: int
    candidate_sequences: int = 0
    candidate_nodes: int = 0
    horizon_objective: float = np.nan
    # Stage-5 exposes the best complete-horizon objective conditional on each
    # possible first green phase.  External residual policies can therefore
    # make a bounded post-MPC correction without replacing the optimiser.
    first_phase_objectives: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    first_phase_flows: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=float)
    )
    # Opt-in research output. The ordinary Stage-5 solve and warm start are
    # unchanged; these are short-cost-best traces, not a top-K/global optimum.
    first_phase_rollouts: dict = field(default_factory=dict)


class ComplementarityImplicitNLP:
    """Local smooth NLP obtained from a Scholtes-relaxed flow KKT system."""

    def __init__(self, topo: CIJunctionTopology):
        _load_casadi()
        self.topo = topo
        self.n_lanes = len(topo.lanes)
        self.n_phases = len(topo.green_phases)
        self.n_groups = len(topo.receiving_groups)
        self.horizon = config.CIMPC_HORIZON
        self.gamma = np.zeros((self.n_groups, self.n_lanes))
        for group_index, group in enumerate(topo.receiving_groups):
            self.gamma[group_index, group.source_indices] = 1.0
        self.has_overlapping_receiving_groups = bool(
            self.n_groups
            and np.any(np.sum(self.gamma > 0.0, axis=0) > 1)
        )

        self._last_x = None
        self._build_nlp()

    @staticmethod
    def _vec(array: np.ndarray) -> np.ndarray:
        return np.asarray(array, dtype=float).reshape(-1, order="F")

    def _build_nlp(self):
        L, P, G, H = self.n_lanes, self.n_phases, self.n_groups, self.horizon

        # Outer variables: queues, movement flows, relaxed safe-phase weights.
        q = ca.MX.sym("q", L, H + 1)
        f = ca.MX.sym("f", L, H)
        phase = ca.MX.sym("phase", P, H)

        # Inner flow-allocation KKT multipliers.
        lam_d = ca.MX.sym("lam_d", L, H)
        lam_c = ca.MX.sym("lam_c", L, H)
        eta = ca.MX.sym("eta", L, H)
        nu = ca.MX.sym("nu", G, H)

        # State-capacity slack is outside the implicit flow problem.  It keeps
        # the controller recursively feasible under unexpectedly high demand.
        spill = ca.MX.sym("spill", L, H)

        initial_queue = ca.MX.sym("initial_queue", L)
        arrivals = ca.MX.sym("arrivals", L, H)
        receiving_capacity = ca.MX.sym("receiving_capacity", G, H)
        previous_phase = ca.MX.sym("previous_phase", P)

        variables = [q, f, phase, lam_d, lam_c, eta, nu, spill]
        parameters = [initial_queue, arrivals, receiving_capacity, previous_phase]
        self._symbols = {
            "q": q,
            "f": f,
            "phase": phase,
            "lam_d": lam_d,
            "lam_c": lam_c,
            "eta": eta,
            "nu": nu,
            "spill": spill,
        }

        self._slices = {}
        offset = 0
        for name, rows, cols in (
            ("q", L, H + 1),
            ("f", L, H),
            ("phase", P, H),
            ("lam_d", L, H),
            ("lam_c", L, H),
            ("eta", L, H),
            ("nu", G, H),
            ("spill", L, H),
        ):
            size = rows * cols
            self._slices[name] = (slice(offset, offset + size), (rows, cols))
            offset += size

        x = ca.vertcat(*(ca.vec(item) for item in variables))
        p = ca.vertcat(*(ca.vec(item) for item in parameters))
        constraints = []
        lower_g = []
        upper_g = []

        def equality(expr):
            flat = ca.vec(expr)
            constraints.append(flat)
            lower_g.extend([0.0] * int(flat.numel()))
            upper_g.extend([0.0] * int(flat.numel()))

        def nonnegative(expr):
            flat = ca.vec(expr)
            constraints.append(flat)
            lower_g.extend([0.0] * int(flat.numel()))
            upper_g.extend([ca.inf] * int(flat.numel()))

        equality(q[:, 0] - initial_queue)
        objective = 0
        capacity = ca.DM(self.topo.capacity)
        inverse_capacity = 1.0 / ca.fmax(capacity, 1.0)
        service = ca.DM(self.topo.service)
        gamma = ca.DM(self.gamma)

        demand_slacks = []
        signal_slacks = []
        receiving_slacks = []
        stationarity_terms = []

        for k in range(H):
            equality(ca.sum1(phase[:, k]) - 1.0)
            demand = q[:, k] + arrivals[:, k]
            phase_reference = previous_phase if k == 0 else phase[:, k - 1]
            # A held phase uses the complete interval.  A one-hot phase change
            # loses exactly CIMPC_YELLOW seconds; the smooth squared distance
            # is also meaningful during the relaxed NLP iterations.
            switch_fraction = 0.5 * ca.sumsqr(phase[:, k] - phase_reference)
            effective_green = (
                config.CIMPC_INTERVAL - config.CIMPC_YELLOW * switch_fraction
            )
            signal_capacity = (
                config.CIMPC_SATURATION_PER_LANE
                * effective_green
                * ca.mtimes(service, phase[:, k])
            )
            demand_slack = demand - f[:, k]
            signal_slack = signal_capacity - f[:, k]
            demand_slacks.append(demand_slack)
            signal_slacks.append(signal_slack)

            nonnegative(demand_slack)
            nonnegative(signal_slack)
            equality(q[:, k + 1] - demand + f[:, k])
            nonnegative(capacity + spill[:, k] - q[:, k + 1])

            # Cumulative receiving-space constraints prevent using the same
            # initially free downstream storage again at each horizon step.
            if G:
                cumulative_inflow = 0
                for tau in range(k + 1):
                    cumulative_inflow += ca.mtimes(gamma, f[:, tau])
                recv_slack = receiving_capacity[:, k] - cumulative_inflow
                receiving_slacks.append(recv_slack)
                nonnegative(recv_slack)

            # Stationarity of the full-horizon strictly concave allocation.
            weight = 1.0 + config.CIMPC_INNER_PRESSURE_WEIGHT * (
                demand * inverse_capacity
            )
            future_receiving_dual = ca.MX.zeros(L, 1)
            if G:
                for tau in range(k, H):
                    future_receiving_dual += ca.mtimes(gamma.T, nu[:, tau])
            stationarity = (
                config.CIMPC_INNER_REGULARIZATION * f[:, k]
                - weight
                + lam_d[:, k]
                + lam_c[:, k]
                + future_receiving_dual
                - eta[:, k]
            )
            stationarity_terms.append(stationarity)
            equality(stationarity)

            # Scholtes relaxation: both factors remain nonnegative through
            # variable bounds/primal constraints and their product is <= rho.
            nonnegative(
                config.CIMPC_SCHOLTES_RHO - lam_d[:, k] * demand_slack
            )
            nonnegative(
                config.CIMPC_SCHOLTES_RHO - lam_c[:, k] * signal_slack
            )
            nonnegative(config.CIMPC_SCHOLTES_RHO - eta[:, k] * f[:, k])
            if G:
                nonnegative(
                    config.CIMPC_SCHOLTES_RHO - nu[:, k] * receiving_slacks[k]
                )

            occupancy = q[:, k + 1] * inverse_capacity
            objective += config.CIMPC_QUEUE_WEIGHT * ca.sumsqr(occupancy)
            objective += config.CIMPC_BALANCE_WEIGHT * ca.sum1(occupancy ** 4)
            objective -= config.CIMPC_FLOW_REWARD * ca.sum1(f[:, k])
            objective += config.CIMPC_SPILL_PENALTY * ca.sum1(spill[:, k] ** 2)

            objective += config.CIMPC_SWITCH_PENALTY * ca.sumsqr(
                phase[:, k] - phase_reference
            )
            objective += config.CIMPC_ONE_HOT_PENALTY * ca.sum1(
                phase[:, k] * (1.0 - phase[:, k])
            )

        terminal_occupancy = q[:, H] * inverse_capacity
        objective += config.CIMPC_TERMINAL_QUEUE_WEIGHT * ca.sumsqr(
            terminal_occupancy
        )

        g = ca.vertcat(*constraints)
        nlp = {"x": x, "p": p, "f": objective, "g": g}
        options = {
            "print_time": False,
            "error_on_fail": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": config.CIMPC_IPOPT_MAX_ITER,
            "ipopt.max_cpu_time": config.CIMPC_LOCAL_TIME_LIMIT,
            "ipopt.tol": config.CIMPC_IPOPT_TOL,
            "ipopt.acceptable_tol": config.CIMPC_IPOPT_ACCEPTABLE_TOL,
            "ipopt.acceptable_iter": 5,
            "ipopt.hessian_approximation": "limited-memory",
        }
        self.solver = ca.nlpsol(f"ci_{self.topo.tl}", "ipopt", nlp, options)
        self.lower_g = np.asarray(lower_g, dtype=float)
        self.upper_g = np.asarray(upper_g, dtype=float)

        # All NLP variables are nonnegative; only phase weights have an upper
        # bound.  Multipliers receive a finite guard against numerical blow-up.
        self.lower_x = np.zeros(int(x.numel()))
        self.upper_x = np.full(int(x.numel()), np.inf)
        phase_slice, _ = self._slices["phase"]
        self.upper_x[phase_slice] = 1.0
        for multiplier in ("lam_d", "lam_c", "eta", "nu"):
            multiplier_slice, _ = self._slices[multiplier]
            self.upper_x[multiplier_slice] = config.CIMPC_DUAL_UPPER_BOUND

    def _unpack(self, vector: np.ndarray, name: str) -> np.ndarray:
        variable_slice, shape = self._slices[name]
        return np.asarray(vector[variable_slice], dtype=float).reshape(
            shape, order="F"
        )

    def _initial_guess(
        self,
        queue: np.ndarray,
        arrivals: np.ndarray,
        receiving_capacity: np.ndarray,
        current_green: int,
    ) -> np.ndarray:
        L, P, G, H = self.n_lanes, self.n_phases, self.n_groups, self.horizon
        if self._last_x is not None and np.all(np.isfinite(self._last_x)):
            guess = self._last_x.copy()
            q_slice, q_shape = self._slices["q"]
            old_q = guess[q_slice].reshape(q_shape, order="F")
            old_q[:, 0] = queue
            guess[q_slice] = self._vec(old_q)
            # A pure one-hot warm start can trap the non-convex phase penalty
            # at the previously selected vertex.  A small deterministic
            # interior reset lets the traffic objective rank other safe phases.
            phase_slice, phase_shape = self._slices["phase"]
            old_phase = guess[phase_slice].reshape(phase_shape, order="F")
            old_phase = (
                config.CIMPC_WARM_PHASE_MEMORY * old_phase
                + (1.0 - config.CIMPC_WARM_PHASE_MEMORY) / P
            )
            guess[phase_slice] = self._vec(old_phase)
            return np.clip(guess, self.lower_x, self.upper_x)

        phase = np.zeros((P, H))
        phase[current_green, :] = 1.0
        q = np.zeros((L, H + 1))
        q[:, 0] = queue
        f = np.zeros((L, H))
        spill = np.zeros((L, H))
        cumulative_group_flow = np.zeros(G)
        effective_green = config.CIMPC_INTERVAL

        for k in range(H):
            demand = q[:, k] + arrivals[:, k]
            signal_capacity = (
                config.CIMPC_SATURATION_PER_LANE
                * effective_green
                * (self.topo.service @ phase[:, k])
            )
            f[:, k] = np.minimum(demand, signal_capacity)
            for group_index in range(G):
                members = self.gamma[group_index] > 0.5
                requested = float(np.sum(f[members, k]))
                available = max(
                    receiving_capacity[group_index, k]
                    - cumulative_group_flow[group_index],
                    0.0,
                )
                if requested > available and requested > 0:
                    f[members, k] *= available / requested
                cumulative_group_flow[group_index] += float(np.sum(f[members, k]))
            q[:, k + 1] = np.maximum(demand - f[:, k], 0.0)
            spill[:, k] = np.maximum(q[:, k + 1] - self.topo.capacity, 0.0)

        lam_d = np.zeros((L, H))
        lam_c = np.zeros((L, H))
        eta = np.zeros((L, H))
        nu = np.zeros((G, H))
        for k in range(H):
            demand = q[:, k] + arrivals[:, k]
            signal_capacity = (
                config.CIMPC_SATURATION_PER_LANE
                * effective_green
                * (self.topo.service @ phase[:, k])
            )
            rhs = np.maximum(
                1.0
                + config.CIMPC_INNER_PRESSURE_WEIGHT
                * demand
                / np.maximum(self.topo.capacity, 1.0)
                - config.CIMPC_INNER_REGULARIZATION * f[:, k],
                0.0,
            )
            demand_active = np.abs(demand - f[:, k]) < 1e-5
            signal_active = np.abs(signal_capacity - f[:, k]) < 1e-5
            lam_d[demand_active, k] = rhs[demand_active]
            use_signal = ~demand_active & signal_active
            lam_c[use_signal, k] = rhs[use_signal]

        blocks = [q, f, phase, lam_d, lam_c, eta, nu, spill]
        return np.concatenate([self._vec(item) for item in blocks])

    def solve(
        self,
        queue: np.ndarray,
        arrivals: np.ndarray,
        receiving_capacity: np.ndarray,
        current_green: int,
        adaptive_parameters=None,
    ) -> LocalCISolution:
        L, P, G, H = self.n_lanes, self.n_phases, self.n_groups, self.horizon
        queue = np.maximum(np.asarray(queue, dtype=float), 0.0)
        arrivals = np.maximum(np.asarray(arrivals, dtype=float), 0.0)
        receiving_capacity = np.maximum(
            np.asarray(receiving_capacity, dtype=float), 0.0
        ).reshape((G, H))
        previous_phase = np.zeros(P)
        previous_phase[current_green] = 1.0
        parameter_vector = np.concatenate(
            (
                self._vec(queue),
                self._vec(arrivals),
                self._vec(receiving_capacity),
                self._vec(previous_phase),
            )
        )
        initial_guess = self._initial_guess(
            queue, arrivals, receiving_capacity, current_green
        )

        started = time.perf_counter()
        try:
            result = self.solver(
                x0=initial_guess,
                p=parameter_vector,
                lbx=self.lower_x,
                ubx=self.upper_x,
                lbg=self.lower_g,
                ubg=self.upper_g,
            )
            vector = np.asarray(result["x"]).reshape(-1)
            constraint_values = np.asarray(result["g"]).reshape(-1)
            stats = self.solver.stats()
            status = str(stats.get("return_status", "unknown"))
            iterations = int(stats.get("iter_count", 0) or 0)
        except Exception as exc:
            vector = initial_guess
            constraint_values = np.full_like(self.lower_g, np.nan)
            status = f"exception:{type(exc).__name__}"
            iterations = 0
        solve_time = time.perf_counter() - started

        q = self._unpack(vector, "q")
        f = self._unpack(vector, "f")
        phase = self._unpack(vector, "phase")
        lam_d = self._unpack(vector, "lam_d")
        lam_c = self._unpack(vector, "lam_c")
        eta = self._unpack(vector, "eta")
        nu = self._unpack(vector, "nu")

        demand = q[:, :H] + arrivals
        signal_capacity = np.zeros((L, H))
        for k in range(H):
            reference = previous_phase if k == 0 else phase[:, k - 1]
            switch_fraction = 0.5 * float(np.sum((phase[:, k] - reference) ** 2))
            effective_green = (
                config.CIMPC_INTERVAL - config.CIMPC_YELLOW * switch_fraction
            )
            signal_capacity[:, k] = (
                config.CIMPC_SATURATION_PER_LANE
                * effective_green
                * (self.topo.service @ phase[:, k])
            )
        complementarity = [
            (lam_d * (demand - f)).reshape(-1),
            (lam_c * (signal_capacity - f)).reshape(-1),
            (eta * f).reshape(-1),
        ]
        stationarity_columns = []
        receiving_margins = []
        for k in range(H):
            future_dual = np.zeros(L)
            if G:
                future_dual = self.gamma.T @ np.sum(nu[:, k:], axis=1)
                margin = receiving_capacity[:, k] - self.gamma @ np.sum(
                    f[:, : k + 1], axis=1
                )
                receiving_margins.append(margin)
                complementarity.append((nu[:, k] * margin).reshape(-1))
            weight = 1.0 + config.CIMPC_INNER_PRESSURE_WEIGHT * (
                demand[:, k] / np.maximum(self.topo.capacity, 1.0)
            )
            stationarity_columns.append(
                config.CIMPC_INNER_REGULARIZATION * f[:, k]
                - weight
                + lam_d[:, k]
                + lam_c[:, k]
                + future_dual
                - eta[:, k]
            )
        products = np.concatenate(complementarity)
        finite_products = products[np.isfinite(products)]
        max_complementarity = (
            float(np.max(np.abs(finite_products))) if finite_products.size else np.inf
        )
        mean_complementarity = (
            float(np.mean(np.abs(finite_products))) if finite_products.size else np.inf
        )
        max_stationarity = float(
            np.max(np.abs(np.column_stack(stationarity_columns)))
        )

        primal_margins = [demand - f, signal_capacity - f, f, phase, 1.0 - phase]
        if receiving_margins:
            primal_margins.append(np.column_stack(receiving_margins))
        min_primal_margin = min(float(np.min(item)) for item in primal_margins)
        max_primal_violation = max(-min_primal_margin, 0.0)
        if constraint_values.size and np.all(np.isfinite(constraint_values)):
            lower_violation = np.maximum(self.lower_g - constraint_values, 0.0)
            finite_upper = np.isfinite(self.upper_g)
            upper_violation = np.zeros_like(constraint_values)
            upper_violation[finite_upper] = np.maximum(
                constraint_values[finite_upper] - self.upper_g[finite_upper], 0.0
            )
            max_primal_violation = max(
                max_primal_violation,
                float(np.max(lower_violation)),
                float(np.max(upper_violation)),
            )
        phase_integrality_gap = float(np.mean(1.0 - np.max(phase, axis=0)))

        accepted_statuses = {
            "Solve_Succeeded",
            "Solved_To_Acceptable_Level",
            "Feasible_Point_Found",
        }
        finite = np.all(np.isfinite(vector))
        success = bool(
            finite
            and max_primal_violation <= config.CIMPC_ACCEPTED_PRIMAL_VIOLATION
            and max_stationarity <= config.CIMPC_ACCEPTED_STATIONARITY_RESIDUAL
            and (
                status in accepted_statuses
                or status in {"Maximum_Iterations_Exceeded", "Maximum_CpuTime_Exceeded"}
            )
        )
        if success:
            self._last_x = vector.copy()

        return LocalCISolution(
            success=success,
            status=status,
            flow=np.maximum(f, 0.0),
            phase=np.clip(phase, 0.0, 1.0),
            max_complementarity=max_complementarity,
            mean_complementarity=mean_complementarity,
            max_stationarity_residual=max_stationarity,
            max_primal_violation=max_primal_violation,
            phase_integrality_gap=phase_integrality_gap,
            solve_time_s=solve_time,
            iterations=iterations,
        )


def _failed_local_solution(topo, current_green, status):
    phase = np.zeros((len(topo.green_phases), config.CIMPC_HORIZON))
    phase[current_green, :] = 1.0
    return LocalCISolution(
        success=False,
        status=status,
        flow=np.zeros((len(topo.lanes), config.CIMPC_HORIZON)),
        phase=phase,
        max_complementarity=np.inf,
        mean_complementarity=np.inf,
        max_stationarity_residual=np.inf,
        max_primal_violation=np.inf,
        phase_integrality_gap=0.0,
        solve_time_s=0.0,
        iterations=0,
    )


class FixedPhaseComplementaritySolver:
    """Strict phase enumeration with exact stagewise KKT water filling.

    For stage 4 the signal action is no longer a relaxed NLP variable.  Given
    each safe one-hot phase, the strictly concave implicit flow allocation has
    a unique capped water-filling solution.  This directly solves its
    complementarity system and avoids a general-purpose IPOPT call. Stage 5
    evaluates every complete phase sequence over the prediction horizon;
    stage 4 retains its original stage-greedy selection for ablation.
    """

    def __init__(self, topo, enumerate_sequences=False):
        self.topo = topo
        self.n_lanes = len(topo.lanes)
        self.n_phases = len(topo.green_phases)
        self.n_groups = len(topo.receiving_groups)
        self.horizon = config.CIMPC_HORIZON
        self.gamma = np.zeros((self.n_groups, self.n_lanes))
        for group_index, group in enumerate(topo.receiving_groups):
            self.gamma[group_index, group.source_indices] = 1.0
        self.has_overlapping_receiving_groups = bool(
            self.n_groups
            and np.any(np.sum(self.gamma > 0.0, axis=0) > 1)
        )
        self.last_flow = None
        self.last_phase = None
        self.enumerate_sequences = bool(enumerate_sequences)
        self.active_parameters = {}

    def _allocate(self, available, receiving_remaining, phase_index, previous):
        switched = phase_index != previous
        effective_green = config.CIMPC_INTERVAL - (
            config.CIMPC_YELLOW if switched else 0.0
        )
        signal_capacity = (
            config.CIMPC_SATURATION_PER_LANE
            * effective_green
            * self.topo.service[:, phase_index]
        )
        upper = np.minimum(available, signal_capacity)
        weights = 1.0 + config.CIMPC_INNER_PRESSURE_WEIGHT * (
            available / np.maximum(self.topo.capacity, 1.0)
        )
        flow = upper.copy()
        receiving_dual = np.zeros(self.n_groups)
        eps = config.CIMPC_INNER_REGULARIZATION

        if not self.has_overlapping_receiving_groups:
            # Preserve the original exact and faster one-pass water filling
            # when every movement lane belongs to at most one receiving group.
            for group_index, group in enumerate(self.topo.receiving_groups):
                members = np.asarray(group.source_indices, dtype=int)
                group_capacity = max(
                    float(receiving_remaining[group_index]), 0.0
                )
                if float(np.sum(flow[members])) <= group_capacity + 1e-10:
                    continue
                member_weights = weights[members]
                member_upper = upper[members]
                low = (
                    float(np.min(member_weights - eps * member_upper)) - 1.0
                )
                high = float(np.max(member_weights)) + 1.0
                for _ in range(60):
                    dual = 0.5 * (low + high)
                    allocation = np.clip(
                        (member_weights - dual) / eps, 0.0, member_upper
                    )
                    if float(np.sum(allocation)) > group_capacity:
                        low = dual
                    else:
                        high = dual
                receiving_dual[group_index] = high
                flow[members] = np.clip(
                    (member_weights - high) / eps, 0.0, member_upper
                )
        else:
            # A lane can feed multiple downstream edges on irregular networks.
            # Sequentially overwriting flow[members] is then invalid: solving a
            # later group can violate an earlier group. Coordinate minimisation
            # of the convex dual enforces all coupled receiving constraints:
            #   f(nu)=clip((w-Gamma^T nu)/eps, 0, upper), nu>=0.
            # Each scalar dual update is an exact monotone water-fill solve.
            capacities = np.maximum(
                np.asarray(receiving_remaining, dtype=float), 0.0
            )
            for _ in range(400):
                previous_dual = receiving_dual.copy()
                for group_index, group in enumerate(
                    self.topo.receiving_groups
                ):
                    members = np.asarray(group.source_indices, dtype=int)
                    other_dual = (
                        self.gamma[:, members].T @ receiving_dual
                        - receiving_dual[group_index]
                    )
                    effective_weights = weights[members] - other_dual
                    member_upper = upper[members]
                    group_capacity = float(capacities[group_index])

                    allocation_at_zero = np.clip(
                        effective_weights / eps, 0.0, member_upper
                    )
                    if (
                        float(np.sum(allocation_at_zero))
                        <= group_capacity + 1e-12
                    ):
                        receiving_dual[group_index] = 0.0
                        continue

                    low = 0.0
                    high = max(float(np.max(effective_weights)), 0.0) + 1.0
                    for _ in range(80):
                        dual = 0.5 * (low + high)
                        allocation = np.clip(
                            (effective_weights - dual) / eps,
                            0.0,
                            member_upper,
                        )
                        if float(np.sum(allocation)) > group_capacity:
                            low = dual
                        else:
                            high = dual
                    receiving_dual[group_index] = high

                flow = np.clip(
                    (weights - self.gamma.T @ receiving_dual) / eps,
                    0.0,
                    upper,
                )
                margins = capacities - self.gamma @ flow
                complementarity = receiving_dual * margins
                if (
                    float(np.max(np.maximum(-margins, 0.0))) <= 1e-10
                    and float(np.max(np.abs(complementarity))) <= 1e-9
                    and float(np.max(np.abs(receiving_dual - previous_dual)))
                    <= 1e-10
                ):
                    break
        return flow, signal_capacity, receiving_dual

    def _stage_cost(
        self,
        next_queue,
        candidate_flow,
        receiving_dual,
        phase_index,
        previous,
        horizon_index,
    ):
        """Use the stage-4 objective unchanged so stages 4 and 5 are comparable."""

        occupancy = next_queue / np.maximum(self.topo.capacity, 1.0)
        queue_weight = float(
            self.active_parameters.get("alpha_q", config.CIMPC_QUEUE_WEIGHT)
        )
        flow_reward = float(
            self.active_parameters.get("alpha_f", config.CIMPC_FLOW_REWARD)
        )
        spill_weight = float(self.active_parameters.get("alpha_spill", 0.0))
        switch_weight = float(
            self.active_parameters.get(
                "alpha_delta_phase", config.CIMPC_PROJECTION_SWITCH_PENALTY
            )
        )
        safety_weight = float(self.active_parameters.get("lambda_safe", 0.0))
        safety_excess = np.maximum(
            occupancy - config.RLCIMPC_SAFE_OCCUPANCY, 0.0
        )
        cost = (
            (queue_weight + config.CIMPC_TERMINAL_QUEUE_WEIGHT)
            * float(np.sum(occupancy ** 2))
            + config.CIMPC_BALANCE_WEIGHT
            * float(np.sum(occupancy ** 4))
            + config.CIMPC_PROJECTION_LINEAR_QUEUE_WEIGHT
            * float(np.sum(next_queue))
            - flow_reward * float(np.sum(candidate_flow))
            + switch_weight * float(phase_index != previous)
            + spill_weight * float(np.sum(np.maximum(receiving_dual, 0.0)))
            + safety_weight * float(np.sum(safety_excess ** 2))
        )
        if self.last_flow is not None:
            cost += config.CIMPC_PROX_FLOW_WEIGHT * float(
                np.sum((candidate_flow - self.last_flow[:, horizon_index]) ** 2)
            )
        if self.last_phase is not None:
            cost += config.CIMPC_PROX_PHASE_WEIGHT * float(
                phase_index != self.last_phase[horizon_index]
            )
        return cost

    def solve(
        self,
        queue,
        arrivals,
        receiving_capacity,
        current_green,
        adaptive_parameters=None,
    ):
        started = time.perf_counter()
        self.active_parameters = dict(adaptive_parameters or {})
        queue = np.maximum(np.asarray(queue, dtype=float), 0.0)
        arrivals = np.maximum(np.asarray(arrivals, dtype=float), 0.0)
        receiving_capacity = np.maximum(
            np.asarray(receiving_capacity, dtype=float), 0.0
        ).reshape((self.n_groups, self.horizon))

        q = np.zeros((self.n_lanes, self.horizon + 1))
        flow = np.zeros((self.n_lanes, self.horizon))
        phase = np.zeros((self.n_phases, self.horizon))
        signal_capacity = np.zeros_like(flow)
        receiving_duals = np.zeros((self.n_groups, self.horizon))
        q[:, 0] = queue
        candidate_sequences = 0
        candidate_nodes = 0
        horizon_objective = 0.0
        first_phase_objectives = np.full(self.n_phases, np.inf, dtype=float)
        first_phase_flows = np.zeros((self.n_lanes, self.n_phases), dtype=float)
        first_phase_rollouts = {}
        retain_rollouts = bool(self.active_parameters.get("retain_candidate_rollouts", False))
        terminal_value = self.active_parameters.get("terminal_value")
        if terminal_value is not None and not self.enumerate_sequences:
            raise ValueError("Learned terminal value requires full sequence enumeration")

        if self.enumerate_sequences:
            complete_sequences = self.n_phases ** self.horizon
            if complete_sequences > config.CIMPC_SEQUENCE_ENUMERATION_LIMIT:
                raise ValueError(
                    "phase-sequence space exceeds CIMPC_SEQUENCE_ENUMERATION_LIMIT: "
                    f"{self.n_phases}^{self.horizon}={complete_sequences}"
                )

            best_objective = np.inf
            best_rollout = None

            def enumerate_prefix(
                k,
                local_queue,
                cumulative_inflow,
                previous,
                objective,
                flow_prefix,
                capacity_prefix,
                dual_prefix,
                queue_prefix,
                phase_prefix,
            ):
                nonlocal candidate_nodes, candidate_sequences
                nonlocal best_objective, best_rollout

                if k == self.horizon:
                    candidate_sequences += 1
                    if terminal_value is not None:
                        extra = float(terminal_value(self.topo, local_queue,
                                                    arrivals[:, -1], phase_prefix[-1]))
                        if not np.isfinite(extra):
                            raise ValueError("Non-finite learned terminal value")
                        objective += extra
                    first_phase = int(phase_prefix[0])
                    if objective < first_phase_objectives[first_phase] - 1e-12:
                        first_phase_objectives[first_phase] = float(objective)
                        first_phase_flows[:, first_phase] = flow_prefix[0]
                        if retain_rollouts:
                            first_phase_rollouts[first_phase] = {
                                "queue": np.asarray(queue_prefix).copy(),
                                "flow": np.asarray(flow_prefix).copy(),
                                "phase": np.asarray(phase_prefix, dtype=int),
                            }
                    if objective < best_objective - 1e-12:
                        best_objective = objective
                        best_rollout = (
                            [item.copy() for item in flow_prefix],
                            [item.copy() for item in capacity_prefix],
                            [item.copy() for item in dual_prefix],
                            [item.copy() for item in queue_prefix],
                            list(phase_prefix),
                        )
                    return

                available = local_queue + arrivals[:, k]
                remaining = (
                    receiving_capacity[:, k] - cumulative_inflow
                    if self.n_groups
                    else np.zeros(0)
                )
                for phase_index in range(self.n_phases):
                    candidate_nodes += 1
                    local_flow, local_capacity, local_dual = self._allocate(
                        available, remaining, phase_index, previous
                    )
                    next_queue = np.maximum(available - local_flow, 0.0)
                    stage_cost = self._stage_cost(
                        next_queue,
                        local_flow,
                        local_dual,
                        phase_index,
                        previous,
                        k,
                    )
                    next_cumulative = (
                        cumulative_inflow + self.gamma @ local_flow
                        if self.n_groups
                        else cumulative_inflow
                    )
                    enumerate_prefix(
                        k + 1,
                        next_queue,
                        next_cumulative,
                        phase_index,
                        objective + stage_cost,
                        flow_prefix + [local_flow],
                        capacity_prefix + [local_capacity],
                        dual_prefix + [local_dual],
                        queue_prefix + [next_queue],
                        phase_prefix + [phase_index],
                    )

            enumerate_prefix(
                0,
                queue,
                np.zeros(self.n_groups),
                current_green,
                0.0,
                [],
                [],
                [],
                [],
                [],
            )
            if best_rollout is None:
                raise RuntimeError("phase-sequence enumeration returned no rollout")
            best_flows, best_capacities, best_duals, best_queues, best_phases = (
                best_rollout
            )
            horizon_objective = float(best_objective)
            for k, choice in enumerate(best_phases):
                phase[choice, k] = 1.0
                flow[:, k] = best_flows[k]
                signal_capacity[:, k] = best_capacities[k]
                receiving_duals[:, k] = best_duals[k]
                q[:, k + 1] = best_queues[k]
        else:
            cumulative_inflow = np.zeros(self.n_groups)
            previous = current_green
            for k in range(self.horizon):
                available = q[:, k] + arrivals[:, k]
                remaining = (
                    receiving_capacity[:, k] - cumulative_inflow
                    if self.n_groups
                    else np.zeros(0)
                )
                best = None
                for phase_index in range(self.n_phases):
                    candidate_nodes += 1
                    candidate_flow, candidate_capacity, candidate_dual = (
                        self._allocate(
                            available, remaining, phase_index, previous
                        )
                    )
                    next_queue = np.maximum(available - candidate_flow, 0.0)
                    cost = self._stage_cost(
                        next_queue,
                        candidate_flow,
                        candidate_dual,
                        phase_index,
                        previous,
                        k,
                    )
                    if k == 0:
                        first_phase_objectives[phase_index] = float(cost)
                        first_phase_flows[:, phase_index] = candidate_flow
                    if best is None or cost < best[0] - 1e-12:
                        best = (
                            cost,
                            phase_index,
                            candidate_flow,
                            candidate_capacity,
                            candidate_dual,
                            next_queue,
                        )

                cost, choice, local_flow, local_capacity, local_dual, next_queue = (
                    best
                )
                horizon_objective += cost
                phase[choice, k] = 1.0
                flow[:, k] = local_flow
                signal_capacity[:, k] = local_capacity
                receiving_duals[:, k] = local_dual
                q[:, k + 1] = next_queue
                if self.n_groups:
                    cumulative_inflow += self.gamma @ local_flow
                previous = choice

        # Reconstruct stagewise KKT multipliers for auditable exact
        # complementarity and stationarity diagnostics.
        products = []
        stationarity_residuals = []
        primal_margins = [flow, q]
        cumulative_inflow = np.zeros(self.n_groups)
        for k in range(self.horizon):
            demand = q[:, k] + arrivals[:, k]
            demand_slack = demand - flow[:, k]
            capacity_slack = signal_capacity[:, k] - flow[:, k]
            weights = 1.0 + config.CIMPC_INNER_PRESSURE_WEIGHT * (
                demand / np.maximum(self.topo.capacity, 1.0)
            )
            group_dual = receiving_duals[:, k]
            downstream_dual = (
                self.gamma.T @ group_dual
                if self.n_groups
                else np.zeros(self.n_lanes)
            )
            required_upper_dual = (
                weights
                - config.CIMPC_INNER_REGULARIZATION * flow[:, k]
                - downstream_dual
            )
            lam_d = np.zeros(self.n_lanes)
            lam_c = np.zeros(self.n_lanes)
            eta = np.zeros(self.n_lanes)
            for lane_index in range(self.n_lanes):
                value = required_upper_dual[lane_index]
                if value >= 0:
                    if demand_slack[lane_index] <= capacity_slack[lane_index] + 1e-8:
                        lam_d[lane_index] = value
                    else:
                        lam_c[lane_index] = value
                else:
                    eta[lane_index] = -value
            stationarity = (
                config.CIMPC_INNER_REGULARIZATION * flow[:, k]
                - weights
                + lam_d
                + lam_c
                + downstream_dual
                - eta
            )
            stationarity_residuals.append(stationarity)
            products.extend(
                [lam_d * demand_slack, lam_c * capacity_slack, eta * flow[:, k]]
            )
            primal_margins.extend([demand_slack, capacity_slack])
            if self.n_groups:
                cumulative_inflow += self.gamma @ flow[:, k]
                receiving_margin = receiving_capacity[:, k] - cumulative_inflow
                products.append(group_dual * receiving_margin)
                primal_margins.append(receiving_margin)

        products_array = np.concatenate([np.asarray(item).reshape(-1) for item in products])
        stationarity_array = np.concatenate(stationarity_residuals)
        min_margin = min(float(np.min(item)) for item in primal_margins)
        max_primal_violation = max(-min_margin, 0.0)
        finite = np.all(np.isfinite(q)) and np.all(np.isfinite(flow))
        success = bool(finite and max_primal_violation <= 1e-7)

        self.last_flow = flow.copy()
        self.last_phase = np.argmax(phase, axis=0)
        return LocalCISolution(
            success=success,
            status=(
                "Phase_Sequence_Enumeration_Exact"
                if success and self.enumerate_sequences
                else "Specialised_Complementarity_Exact"
                if success
                else "fast_infeasible"
            ),
            flow=flow,
            phase=phase,
            max_complementarity=float(np.max(np.abs(products_array))),
            mean_complementarity=float(np.mean(np.abs(products_array))),
            max_stationarity_residual=float(np.max(np.abs(stationarity_array))),
            max_primal_violation=max_primal_violation,
            phase_integrality_gap=0.0,
            solve_time_s=time.perf_counter() - started,
            iterations=candidate_nodes,
            candidate_sequences=candidate_sequences,
            candidate_nodes=candidate_nodes,
            horizon_objective=horizon_objective,
            first_phase_objectives=first_phase_objectives,
            first_phase_flows=first_phase_flows,
            first_phase_rollouts=first_phase_rollouts,
        )


def _ci_process_worker(
    worker_id,
    topology,
    input_queue,
    output_queue,
    fast_solver=False,
    enumerate_sequences=False,
):
    """Own a fixed topology shard and its warm-started CasADi solvers."""

    if fast_solver:
        agents = {
            tl: FixedPhaseComplementaritySolver(
                topo, enumerate_sequences=enumerate_sequences
            )
            for tl, topo in topology.items()
        }
    else:
        agents = {
            tl: ComplementarityImplicitNLP(topo)
            for tl, topo in topology.items()
        }
    output_queue.put(("ready", worker_id, os.getpid()))
    while True:
        message = input_queue.get()
        if message is None:
            break
        task_id, payloads = message
        results = []
        for payload in payloads:
            (
                tl,
                local_queue,
                arrivals,
                receiving,
                current_green,
                adaptive_parameters,
            ) = payload
            try:
                solution = agents[tl].solve(
                    local_queue,
                    arrivals,
                    receiving,
                    current_green,
                    adaptive_parameters,
                )
            except Exception as exc:
                solution = _failed_local_solution(
                    topology[tl],
                    current_green,
                    f"worker_exception:{type(exc).__name__}",
                )
            results.append((tl, solution))
        output_queue.put(("result", task_id, worker_id, results))

    # CasADi 3.7.x can corrupt the Windows heap during threaded/native DLL
    # finalisation. Queues are flushed before bypassing only that finalisation.
    output_queue.put(("stopped", worker_id, os.getpid()))
    output_queue.close()
    output_queue.join_thread()
    os._exit(0)


class PersistentCIProcessPool:
    """Eight fixed shards; every junction keeps its solver and warm start."""

    def __init__(
        self,
        topology,
        tls_ids,
        workers,
        fast_solver=False,
        enumerate_sequences=False,
    ):
        self.context = mp.get_context("spawn")
        self.output_queue = self.context.Queue()
        self.input_queues = []
        self.processes = []
        self.worker_for_tl = {}
        self.task_id = 0

        worker_count = min(max(int(workers), 1), len(tls_ids))
        shards = [dict() for _ in range(worker_count)]
        for index, tl in enumerate(sorted(tls_ids)):
            worker_id = index % worker_count
            shards[worker_id][tl] = topology[tl]
            self.worker_for_tl[tl] = worker_id

        self.worker_pids = {}
        # Starting eight CasADi runtimes simultaneously creates a large
        # transient memory spike on Windows. Start fixed shards one at a time;
        # this is a cold-start cost only and keeps steady-state parallelism 8.
        try:
            for worker_id, shard in enumerate(shards):
                input_queue = self.context.Queue()
                process = self.context.Process(
                    target=_ci_process_worker,
                    args=(
                        worker_id,
                        shard,
                        input_queue,
                        self.output_queue,
                        fast_solver,
                        enumerate_sequences,
                    ),
                    name=f"ci-mpc-worker-{worker_id}",
                )
                process.daemon = True
                process.start()
                self.input_queues.append(input_queue)
                self.processes.append(process)

                deadline = time.monotonic() + 45.0
                while worker_id not in self.worker_pids:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"CI-MPC process worker {worker_id} did not initialise"
                        )
                    kind, ready_worker, value = self.output_queue.get(
                        timeout=remaining
                    )
                    if kind == "ready":
                        self.worker_pids[ready_worker] = value
        except Exception:
            self.shutdown()
            raise

    def solve(self, payloads):
        self.task_id += 1
        task_id = self.task_id
        by_worker = defaultdict(list)
        for payload in payloads:
            by_worker[self.worker_for_tl[payload[0]]].append(payload)
        for worker_id, worker_payloads in by_worker.items():
            self.input_queues[worker_id].put((task_id, worker_payloads))

        results = []
        received = 0
        deadline = time.monotonic() + config.CIMPC_PROCESS_TIMEOUT
        while received < len(by_worker):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                kind, received_task, worker_id, worker_results = (
                    self.output_queue.get(timeout=remaining)
                )
            except queue_module.Empty:
                break
            if kind == "result" and received_task == task_id:
                results.extend(worker_results)
                received += 1

        returned = {tl for tl, _ in results}
        for payload in payloads:
            tl = payload[0]
            current_green = payload[4]
            if tl not in returned:
                results.append(
                    (
                        tl,
                        _failed_local_solution(
                            # Each payload's topology belongs to exactly one
                            # fixed worker; the parent adds a timeout fallback.
                            self._topology_for_timeout[tl],
                            current_green,
                            "worker_timeout",
                        ),
                    )
                )
        return results

    def attach_topology(self, topology):
        self._topology_for_timeout = topology

    def worker_memory_mb(self):
        try:
            import psutil

            return sum(
                psutil.Process(pid).memory_info().rss
                for pid in self.worker_pids.values()
                if psutil.pid_exists(pid)
            ) / (1024.0 ** 2)
        except Exception:
            return np.nan

    def shutdown(self):
        for input_queue in getattr(self, "input_queues", []):
            try:
                input_queue.put(None)
            except Exception:
                pass
        for process in getattr(self, "processes", []):
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        for input_queue in getattr(self, "input_queues", []):
            try:
                input_queue.close()
            except Exception:
                pass
        try:
            self.output_queue.close()
        except Exception:
            pass


class CIMPCController(BaseController):
    """Distributed complementarity-implicit MPC with neighbour messages."""

    name = "CI_MPC"
    diagnostic_prefix = "ci_mpc"

    def __init__(self, conn):
        super().__init__(conn)
        self.topology = self._build_topology()
        self.agents = {}
        self.executor = None
        self.process_pool = None
        if config.CIMPC_IMPLEMENTATION_STAGE >= 2:
            self.process_pool = PersistentCIProcessPool(
                self.topology,
                self.tls_ids,
                config.CIMPC_PROCESS_WORKERS,
                fast_solver=config.CIMPC_IMPLEMENTATION_STAGE >= 4,
                enumerate_sequences=config.CIMPC_IMPLEMENTATION_STAGE >= 5,
            )
            self.process_pool.attach_topology(self.topology)
        else:
            self.agents = {
                tl: ComplementarityImplicitNLP(self.topology[tl])
                for tl in self.tls_ids
            }
            self.executor = ThreadPoolExecutor(
                max_workers=min(config.CIMPC_PARALLEL_WORKERS, len(self.tls_ids)),
                thread_name_prefix="ci-mpc-junction",
            )
        self.flow_forecast = {
            tl: np.zeros((len(self.topology[tl].lanes), config.CIMPC_HORIZON))
            for tl in self.tls_ids
        }
        self.boundary_lanes = {
            topo.lanes[index]
            for topo in self.topology.values()
            for index in topo.boundary_indices
        }
        self.boundary_counts = defaultdict(float)
        self.boundary_rates = defaultdict(
            lambda: config.CIMPC_INITIAL_BOUNDARY_RATE
        )
        self.last_vehicle_ids = set()
        self.next_decision = 0.0
        self.diagnostics = []
        self.local_diagnostics = []
        self.current_adaptive_parameters = None
        self._closed = False

    def _build_topology(self):
        lanes_by_tl = {}
        lane_owner = {}
        for tl in self.tls_ids:
            lanes = sorted(
                {
                    lane
                    for approach_lanes in self.graph.tl_approach_lanes[tl].values()
                    for lane in approach_lanes
                }
            )
            lanes_by_tl[tl] = lanes
            for lane in lanes:
                lane_owner[lane] = tl

        node_to_tl = {
            node.getID(): tl
            for tl, nodes in self.graph.tl_nodes.items()
            for node in nodes
        }
        topology = {}
        for tl in self.tls_ids:
            lanes = lanes_by_tl[tl]
            lane_index = {lane: index for index, lane in enumerate(lanes)}
            phases = self.graph.green_phase_indices[tl]
            service = np.zeros((len(lanes), len(phases)))
            for phase_index, phase in enumerate(phases):
                state = self.graph.phase_states[tl][phase]
                for link_index, character in enumerate(state):
                    if (
                        character not in ("G", "g")
                        or link_index >= len(self.graph.tl_lanes[tl])
                    ):
                        continue
                    lane = self.graph.tl_lanes[tl][link_index]
                    if lane in lane_index:
                        service[lane_index[lane], phase_index] = 1.0

            capacity = np.array(
                [
                    max(self.graph.net.getLane(lane).getLength(), 1.0)
                    / config.CIMPC_EFFECTIVE_VEHICLE_LENGTH
                    for lane in lanes
                ],
                dtype=float,
            )
            boundary_indices = []
            for index, lane in enumerate(lanes):
                edge = self.graph.net.getLane(lane).getEdge()
                if edge.getFromNode().getID() not in node_to_tl:
                    boundary_indices.append(index)
            topology[tl] = CIJunctionTopology(
                tl=tl,
                lanes=lanes,
                green_phases=phases,
                service=service,
                capacity=capacity,
                boundary_indices=boundary_indices,
            )

        # Build both message-routing matrices and downstream storage groups
        # directly from SUMO controlled connections.  Movement lanes therefore
        # retain their observed turning identity; no equal turn assumption is
        # introduced by the CI-MPC model.
        targets_by_source = defaultdict(set)
        for upstream_tl in self.tls_ids:
            for link_group in self.graph.tl_controlled_links[upstream_tl]:
                for link in link_group:
                    if len(link) >= 2 and link[0] and link[1] in lane_owner:
                        targets_by_source[link[0]].add(link[1])

        receive_members = defaultdict(lambda: {"sources": set(), "targets": set()})
        for source_lane, target_lanes in targets_by_source.items():
            upstream_tl = lane_owner.get(source_lane)
            if upstream_tl is None:
                continue
            source_index = lanes_by_tl[upstream_tl].index(source_lane)
            valid_targets = [lane for lane in target_lanes if lane in lane_owner]
            if not valid_targets:
                continue
            share = 1.0 / len(valid_targets)
            for target_lane in valid_targets:
                downstream_tl = lane_owner[target_lane]
                if downstream_tl == upstream_tl:
                    continue
                target_index = lanes_by_tl[downstream_tl].index(target_lane)
                routing = topology[downstream_tl].upstream_routes.setdefault(
                    upstream_tl,
                    np.zeros(
                        (
                            len(lanes_by_tl[downstream_tl]),
                            len(lanes_by_tl[upstream_tl]),
                        )
                    ),
                )
                routing[target_index, source_index] += share
                target_edge = self.graph.net.getLane(target_lane).getEdge().getID()
                key = (upstream_tl, downstream_tl, target_edge)
                receive_members[key]["sources"].add(source_index)
                receive_members[key]["targets"].add(target_index)

        for (upstream_tl, downstream_tl, target_edge), members in receive_members.items():
            # Storage is shared by every movement lane on the same physical
            # receiving edge, including lanes without a direct connection from
            # this particular upstream junction.
            target_indices = [
                index
                for index, lane in enumerate(lanes_by_tl[downstream_tl])
                if self.graph.net.getLane(lane).getEdge().getID() == target_edge
            ]
            topology[upstream_tl].receiving_groups.append(
                ReceivingGroup(
                    downstream_tl=downstream_tl,
                    target_edge=target_edge,
                    source_indices=sorted(members["sources"]),
                    target_indices=target_indices or sorted(members["targets"]),
                )
            )
        for topo in topology.values():
            topo.receiving_groups.sort(
                key=lambda group: (group.downstream_tl, group.target_edge)
            )
        return topology

    def _observe_boundary_entries(self):
        current = set(self.conn.vehicle.getIDList())
        for vehicle_id in current - self.last_vehicle_ids:
            try:
                lane = self.conn.vehicle.getLaneID(vehicle_id)
            except Exception:
                continue
            if lane in self.boundary_lanes:
                self.boundary_counts[lane] += 1.0
        self.last_vehicle_ids = current

    def _update_boundary_rates(self):
        alpha = config.CIMPC_DEMAND_EWMA_ALPHA
        for lane in self.boundary_lanes:
            observed_rate = self.boundary_counts[lane] / config.CIMPC_INTERVAL
            self.boundary_rates[lane] = (
                (1.0 - alpha) * self.boundary_rates[lane]
                + alpha * observed_rate
            )
            self.boundary_counts[lane] = 0.0

    def _measure_queues(self):
        return {
            tl: np.array(
                [
                    self.conn.lane.getLastStepVehicleNumber(lane)
                    for lane in topo.lanes
                ],
                dtype=float,
            )
            for tl, topo in self.topology.items()
        }

    def _current_green_index(self, tl: str):
        topo = self.topology[tl]
        actual = self.conn.trafficlight.getPhase(tl)
        if actual in topo.green_phases:
            return topo.green_phases.index(actual)
        pending = self.pending_phase.get(tl)
        if pending in topo.green_phases:
            return topo.green_phases.index(pending)
        previous = (actual - 1) % self.graph.num_phases[tl]
        if previous in topo.green_phases:
            return topo.green_phases.index(previous)
        return 0

    def _arrival_forecast(self, tl: str, flows: dict[str, np.ndarray]):
        topo = self.topology[tl]
        arrivals = np.zeros((len(topo.lanes), config.CIMPC_HORIZON))
        for index in topo.boundary_indices:
            arrivals[index, :] = (
                self.boundary_rates[topo.lanes[index]] * config.CIMPC_INTERVAL
            )
        for upstream_tl, routing in topo.upstream_routes.items():
            arrivals += routing @ flows[upstream_tl]
        return arrivals

    def _receiving_capacity_forecast(
        self,
        tl: str,
        queues: dict[str, np.ndarray],
        flows: dict[str, np.ndarray],
    ):
        topo = self.topology[tl]
        receiving = np.zeros(
            (len(topo.receiving_groups), config.CIMPC_HORIZON), dtype=float
        )
        for group_index, group in enumerate(topo.receiving_groups):
            downstream_topo = self.topology[group.downstream_tl]
            targets = group.target_indices
            free_now = float(
                np.sum(
                    np.maximum(
                        downstream_topo.capacity[targets]
                        - queues[group.downstream_tl][targets],
                        0.0,
                    )
                )
            )
            predicted_departures = np.sum(
                flows[group.downstream_tl][targets, :], axis=0
            )
            receiving[group_index, :] = free_now + np.cumsum(
                np.maximum(predicted_departures, 0.0)
            )
        if config.CIMPC_IMPLEMENTATION_STAGE >= 1:
            # Cross-cycle messages are one update old. Tightening receiving
            # storage preserves spillback safety without a same-cycle barrier.
            receiving = np.maximum(
                receiving - config.CIMPC_ASYNC_STORAGE_MARGIN, 0.0
            )
        return receiving

    def _adaptive_parameters_for_update(
        self, sim_time, queues, forecasts, current_green
    ):
        """Hook for policies that adapt objective parameters, not phases."""

        return None

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
        """Hook for a safety-filtered residual policy after MPC has solved.

        The nominal controller returns its phase choices and neighbour flow
        messages unchanged.  Subclasses may select another MPC-enumerated
        first phase and must reconcile the first flow message at the same
        time, before any signal command is sent to SUMO.
        """

        return phase_choices, forecasts

    def _solve_agent(self, payload):
        tl, queue, arrivals, receiving, current_green, adaptive_parameters = payload
        return tl, self.agents[tl].solve(
            queue,
            arrivals,
            receiving,
            current_green,
            adaptive_parameters,
        )

    def _solve_payloads(self, payloads):
        if self.process_pool is not None:
            return self.process_pool.solve(payloads)
        return list(self.executor.map(self._solve_agent, payloads))

    def _strict_phase_projection(
        self,
        tl: str,
        queue: np.ndarray,
        arrivals: np.ndarray,
        receiving: np.ndarray,
        current_green: int,
        relaxed_phase: np.ndarray,
    ):
        """Restore the first relaxed action to the best safe one-hot phase.

        Each candidate uses the same strictly concave inner allocation as the
        KKT model.  Since this is a single interval with disjoint receiving
        groups, its solution is a capped water-filling calculation rather than
        another NLP solve.
        """

        topo = self.topology[tl]
        available = np.maximum(queue + arrivals[:, 0], 0.0)
        lane_weights = 1.0 + config.CIMPC_INNER_PRESSURE_WEIGHT * (
            available / np.maximum(topo.capacity, 1.0)
        )
        best_choice = current_green
        best_flow = np.zeros_like(available)
        best_cost = np.inf

        for phase_index in range(len(topo.green_phases)):
            switched = phase_index != current_green
            effective_green = config.CIMPC_INTERVAL - (
                config.CIMPC_YELLOW if switched else 0.0
            )
            upper = np.minimum(
                available,
                config.CIMPC_SATURATION_PER_LANE
                * effective_green
                * topo.service[:, phase_index],
            )
            flow = upper.copy()

            for group_index, group in enumerate(topo.receiving_groups):
                members = np.asarray(group.source_indices, dtype=int)
                group_capacity = max(float(receiving[group_index, 0]), 0.0)
                if float(np.sum(flow[members])) <= group_capacity + 1e-9:
                    continue
                # KKT water filling for
                # max w^T f - eps/2||f||^2, 0<=f<=upper, sum(f)<=R.
                weights = lane_weights[members]
                member_upper = upper[members]
                low = float(np.min(weights - config.CIMPC_INNER_REGULARIZATION * member_upper)) - 1.0
                high = float(np.max(weights)) + 1.0
                for _ in range(60):
                    dual = 0.5 * (low + high)
                    allocation = np.clip(
                        (weights - dual) / config.CIMPC_INNER_REGULARIZATION,
                        0.0,
                        member_upper,
                    )
                    if float(np.sum(allocation)) > group_capacity:
                        low = dual
                    else:
                        high = dual
                flow[members] = np.clip(
                    (weights - high) / config.CIMPC_INNER_REGULARIZATION,
                    0.0,
                    member_upper,
                )

            next_queue = np.maximum(available - flow, 0.0)
            occupancy = next_queue / np.maximum(topo.capacity, 1.0)
            cost = (
                (config.CIMPC_QUEUE_WEIGHT + config.CIMPC_TERMINAL_QUEUE_WEIGHT)
                * float(np.sum(occupancy ** 2))
                + config.CIMPC_BALANCE_WEIGHT * float(np.sum(occupancy ** 4))
                + config.CIMPC_PROJECTION_LINEAR_QUEUE_WEIGHT
                * float(np.sum(next_queue))
                - config.CIMPC_FLOW_REWARD * float(np.sum(flow))
                + config.CIMPC_PROJECTION_SWITCH_PENALTY * float(switched)
                + config.CIMPC_RELAXED_PHASE_PROJECTION_WEIGHT
                * (1.0 - float(relaxed_phase[phase_index, 0]))
            )
            if cost < best_cost - 1e-10:
                best_cost = cost
                best_choice = phase_index
                best_flow = flow
        return best_choice, best_flow

    def _coordinate(self, sim_time: float):
        queues = self._measure_queues()
        current_green = {
            tl: self._current_green_index(tl) for tl in self.tls_ids
        }
        forecasts = {
            tl: np.hstack((flow[:, 1:], flow[:, -1:]))
            for tl, flow in self.flow_forecast.items()
        }
        adaptive_parameters = self._adaptive_parameters_for_update(
            sim_time, queues, forecasts, current_green
        )
        self.current_adaptive_parameters = (
            dict(adaptive_parameters) if adaptive_parameters is not None else None
        )

        residual = float("inf")
        iterations = 0
        all_solved = True
        phase_choices = current_green.copy()
        latest_solutions = {}
        latest_payload_by_tl = {}
        local_residuals = {tl: 0.0 for tl in self.tls_ids}
        event_trigger_seeds = []
        event_corrected_agents = set()
        started = time.perf_counter()
        coordination_limit = (
            1
            if config.CIMPC_IMPLEMENTATION_STAGE >= 1
            else config.CIMPC_COORDINATION_ITERATIONS
        )
        for iterations in range(1, coordination_limit + 1):
            payloads = []
            for tl in self.tls_ids:
                payloads.append(
                    (
                        tl,
                        queues[tl],
                        self._arrival_forecast(tl, forecasts),
                        self._receiving_capacity_forecast(tl, queues, forecasts),
                        current_green[tl],
                        self.current_adaptive_parameters,
                    )
                )
            solved_pairs = self._solve_payloads(payloads)
            latest_payload_by_tl = {payload[0]: payload for payload in payloads}
            latest_solutions = dict(solved_pairs)
            iteration_solved = all(solution.success for solution in latest_solutions.values())
            all_solved = all_solved and iteration_solved

            proposed = {}
            for tl in self.tls_ids:
                solution = latest_solutions[tl]
                proposed[tl] = solution.flow if solution.success else forecasts[tl]
                if solution.success:
                    payload = latest_payload_by_tl[tl]
                    if config.CIMPC_IMPLEMENTATION_STAGE >= 4:
                        phase_choices[tl] = int(np.argmax(solution.phase[:, 0]))
                        strict_first_flow = solution.flow[:, 0]
                    else:
                        phase_choices[tl], strict_first_flow = self._strict_phase_projection(
                            tl,
                            payload[1],
                            payload[2],
                            payload[3],
                            payload[4],
                            solution.phase,
                        )
                    # The first neighbour message must match the action that is
                    # actually sent to SUMO, not the optimistic relaxed phase.
                    proposed[tl] = proposed[tl].copy()
                    proposed[tl][:, 0] = strict_first_flow

            residual = max(
                float(np.max(np.abs(proposed[tl] - forecasts[tl])))
                for tl in self.tls_ids
            )
            local_residuals = {
                tl: float(np.max(np.abs(proposed[tl] - forecasts[tl])))
                for tl in self.tls_ids
            }
            damping = config.CIMPC_COORDINATION_DAMPING
            forecasts = {
                tl: damping * proposed[tl] + (1.0 - damping) * forecasts[tl]
                for tl in self.tls_ids
            }
            if residual <= config.CIMPC_COORDINATION_TOL:
                break

        # Stage 3: a second solve is event-triggered and local. Select at most
        # two high-risk seeds, expand to their one-hop neighbourhood, and cap
        # the correction subgraph so unrelated junctions never wait.
        if config.CIMPC_IMPLEMENTATION_STAGE >= 3:
            risk_rows = []
            for tl in self.tls_ids:
                payload = latest_payload_by_tl[tl]
                receiving = payload[3]
                low_receiving = bool(
                    receiving.size
                    and float(np.min(receiving[:, 0]))
                    < config.CIMPC_EVENT_RECEIVING_THRESHOLD
                )
                max_downstream_occupancy = 0.0
                for group in self.topology[tl].receiving_groups:
                    downstream = self.topology[group.downstream_tl]
                    indices = group.target_indices
                    occupancy = queues[group.downstream_tl][indices] / np.maximum(
                        downstream.capacity[indices], 1.0
                    )
                    if occupancy.size:
                        max_downstream_occupancy = max(
                            max_downstream_occupancy, float(np.max(occupancy))
                        )
                triggered = (
                    local_residuals[tl]
                    > config.CIMPC_EVENT_RESIDUAL_THRESHOLD
                    or low_receiving
                    or max_downstream_occupancy
                    > config.CIMPC_EVENT_OCCUPANCY_THRESHOLD
                )
                if triggered:
                    risk_score = (
                        local_residuals[tl]
                        + 2.0 * max_downstream_occupancy
                        + (1.0 if low_receiving else 0.0)
                    )
                    risk_rows.append((risk_score, tl))

            risk_rows.sort(reverse=True)
            event_trigger_seeds = [
                tl for _, tl in risk_rows[: config.CIMPC_EVENT_MAX_SEEDS]
            ]
            ordered_correction = []
            for seed in event_trigger_seeds:
                for candidate in [seed, *self.graph.adjacency[seed]]:
                    if candidate not in ordered_correction:
                        ordered_correction.append(candidate)
            event_corrected_agents = set(
                ordered_correction[: config.CIMPC_EVENT_MAX_AGENTS]
            )

            if event_corrected_agents:
                iterations = 2
                correction_payloads = []
                for tl in sorted(event_corrected_agents):
                    correction_payloads.append(
                        (
                            tl,
                            queues[tl],
                            self._arrival_forecast(tl, forecasts),
                            self._receiving_capacity_forecast(
                                tl, queues, forecasts
                            ),
                            current_green[tl],
                            self.current_adaptive_parameters,
                        )
                    )
                correction_pairs = self._solve_payloads(correction_payloads)
                correction_solutions = dict(correction_pairs)
                all_solved = all_solved and all(
                    solution.success
                    for solution in correction_solutions.values()
                )
                corrected_proposed = {}
                for payload in correction_payloads:
                    tl = payload[0]
                    solution = correction_solutions[tl]
                    latest_solutions[tl] = solution
                    latest_payload_by_tl[tl] = payload
                    corrected_proposed[tl] = (
                        solution.flow.copy()
                        if solution.success
                        else forecasts[tl].copy()
                    )
                    if solution.success:
                        if config.CIMPC_IMPLEMENTATION_STAGE >= 4:
                            phase_choices[tl] = int(
                                np.argmax(solution.phase[:, 0])
                            )
                            strict_first_flow = solution.flow[:, 0]
                        else:
                            phase_choices[tl], strict_first_flow = (
                                self._strict_phase_projection(
                                    tl,
                                    payload[1],
                                    payload[2],
                                    payload[3],
                                    payload[4],
                                    solution.phase,
                                )
                            )
                        corrected_proposed[tl][:, 0] = strict_first_flow
                    local_residuals[tl] = float(
                        np.max(np.abs(corrected_proposed[tl] - forecasts[tl]))
                    )

                damping = config.CIMPC_COORDINATION_DAMPING
                for tl in event_corrected_agents:
                    forecasts[tl] = (
                        damping * corrected_proposed[tl]
                        + (1.0 - damping) * forecasts[tl]
                    )
                residual = max(local_residuals.values())

        phase_choices, forecasts = self._postprocess_phase_choices(
            sim_time,
            queues,
            current_green,
            forecasts,
            phase_choices,
            latest_payload_by_tl,
            latest_solutions,
        )
        self.flow_forecast = forecasts
        switched = 0
        for tl, choice in phase_choices.items():
            target = self.topology[tl].green_phases[choice]
            actual = self.conn.trafficlight.getPhase(tl)
            if target != actual and self.time_in_phase[tl] >= config.CIMPC_MIN_GREEN:
                self._safe_set_phase(tl, target)
                switched += 1
            else:
                # SUMO's source program is static and would automatically move
                # to its next yellow after the nominal 10-second green.  An MPC
                # hold action must therefore explicitly extend the selected
                # steady green until the next receding-horizon update.
                state = self.conn.trafficlight.getRedYellowGreenState(tl)
                if "y" not in state and self.pending_phase.get(tl) is None:
                    self.conn.trafficlight.setPhaseDuration(
                        tl, config.CIMPC_INTERVAL + config.STEP_LENGTH
                    )

        solutions = list(latest_solutions.values())
        status_counts = defaultdict(int)
        for tl, solution in latest_solutions.items():
            status_counts[solution.status] += 1
            self.local_diagnostics.append(
                {
                    "simulation_time": sim_time,
                    "tl": tl,
                    "coordination_iteration": iterations,
                    "event_corrected": int(tl in event_corrected_agents),
                    "success": int(solution.success),
                    "status": solution.status,
                    "solve_time_s": solution.solve_time_s,
                    "ipopt_iterations": solution.iterations,
                    "max_complementarity_product": solution.max_complementarity,
                    "mean_complementarity_product": solution.mean_complementarity,
                    "max_stationarity_residual": solution.max_stationarity_residual,
                    "max_primal_violation": solution.max_primal_violation,
                    "phase_integrality_gap": solution.phase_integrality_gap,
                    "candidate_sequences": solution.candidate_sequences,
                    "candidate_nodes": solution.candidate_nodes,
                    "horizon_objective": solution.horizon_objective,
                }
            )
        self.diagnostics.append(
            {
                "simulation_time": sim_time,
                "implementation_stage": config.CIMPC_IMPLEMENTATION_STAGE,
                "coordination_mode": (
                    "event_triggered_async"
                    if config.CIMPC_IMPLEMENTATION_STAGE >= 3
                    else (
                        "cross_cycle_async"
                        if config.CIMPC_IMPLEMENTATION_STAGE >= 1
                        else "same_cycle_synchronous"
                    )
                ),
                "execution_backend": (
                    "persistent_processes_phase_sequence_enumeration"
                    if config.CIMPC_IMPLEMENTATION_STAGE >= 5
                    else "persistent_processes_fast_complementarity"
                    if config.CIMPC_IMPLEMENTATION_STAGE >= 4
                    else "persistent_processes"
                    if self.process_pool is not None
                    else "thread_pool"
                ),
                "worker_processes": (
                    len(self.process_pool.processes)
                    if self.process_pool is not None
                    else 0
                ),
                "worker_rss_mb": (
                    self.process_pool.worker_memory_mb()
                    if self.process_pool is not None
                    else 0.0
                ),
                "coordination_iterations": iterations,
                "coordination_residual": residual,
                "event_trigger_seed_count": len(event_trigger_seeds),
                "event_trigger_seeds": ";".join(sorted(event_trigger_seeds)),
                "event_corrected_agent_count": len(event_corrected_agents),
                "event_corrected_agents": ";".join(
                    sorted(event_corrected_agents)
                ),
                "all_local_nlps_solved": int(all_solved),
                "successful_local_nlps": sum(solution.success for solution in solutions),
                "local_nlps": len(solutions),
                "phase_switches": switched,
                "measured_vehicles_on_controlled_lanes": float(
                    sum(map(np.sum, queues.values()))
                ),
                "max_complementarity_product": max(
                    (solution.max_complementarity for solution in solutions),
                    default=np.nan,
                ),
                "mean_complementarity_product": float(
                    np.mean([solution.mean_complementarity for solution in solutions])
                )
                if solutions
                else np.nan,
                "max_stationarity_residual": max(
                    (solution.max_stationarity_residual for solution in solutions),
                    default=np.nan,
                ),
                "max_primal_violation": max(
                    (solution.max_primal_violation for solution in solutions),
                    default=np.nan,
                ),
                "mean_phase_integrality_gap": float(
                    np.mean([solution.phase_integrality_gap for solution in solutions])
                )
                if solutions
                else np.nan,
                "max_local_solve_time_s": max(
                    (solution.solve_time_s for solution in solutions), default=0.0
                ),
                "status_counts": ";".join(
                    f"{key}:{value}" for key, value in sorted(status_counts.items())
                ),
                "phase_choices": ";".join(
                    f"{tl}:{self.topology[tl].green_phases[choice]}"
                    for tl, choice in sorted(phase_choices.items())
                ),
                "decision_time_s": time.perf_counter() - started,
            }
        )

    def step(self, sim_time: float) -> float:
        started = time.perf_counter()
        for tl in self.tls_ids:
            self.time_in_phase[tl] += config.STEP_LENGTH
            self._apply_pending_after_clearance(tl)
        self._observe_boundary_entries()

        if sim_time + 1e-9 >= self.next_decision:
            if sim_time > 0:
                self._update_boundary_rates()
            self._coordinate(sim_time)
            self.next_decision = sim_time + config.CIMPC_INTERVAL
        return time.perf_counter() - started

    def decide(self, tl: str, sim_time: float):
        return None

    def export_diagnostics(self, output_dir: str):
        if self.diagnostics:
            path = os.path.join(
                output_dir, f"{self.diagnostic_prefix}_diagnostics.csv"
            )
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(self.diagnostics[0].keys())
                )
                writer.writeheader()
                writer.writerows(self.diagnostics)
        if self.local_diagnostics:
            path = os.path.join(
                output_dir, f"{self.diagnostic_prefix}_local_diagnostics.csv"
            )
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(self.local_diagnostics[0].keys())
                )
                writer.writeheader()
                writer.writerows(self.local_diagnostics)
        self.close()

    def close(self):
        """Release worker resources without requiring diagnostics export."""

        if self._closed:
            return
        self._closed = True
        # Explicit shutdown matters on Windows: the non-daemon IPOPT worker
        # threads would otherwise keep the benchmark process alive after all
        # result files had already been written.
        if self.executor is not None:
            self.executor.shutdown(wait=True)
        if self.process_pool is not None:
            self.process_pool.shutdown()
        self.agents.clear()
        gc.collect()
