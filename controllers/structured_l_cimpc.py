"""Experimental joint-candidate actors, with an unmodified V2 fallback.

Candidates are sparse joint actions, not independent junction samples. Full
local traces are heuristic features only; the certificate is recomputed on
the first-step network. No neural output modifies physical flow predictions.
"""
from __future__ import annotations

import time
import numpy as np
import config
from controllers.l_cimpc_v2 import LCIMPCV2Controller


class FastCertificateLCIMPC(LCIMPCV2Controller):
    """Exact-arithmetic V2 with memoized repeated certificates.

Dense vectorization changed summation order by ~1e-15, which can change
argmin ties in exceptional baseline fallbacks. Cache the ORIGINAL calculation
instead; never use a numerically different result to choose a baseline phase.
"""

    def __init__(self, conn, **kwargs):
        super().__init__(conn, external_actions=True, **kwargs)
        self.order = sorted(self.tls_ids)
        self.offsets = {}
        offset = 0
        for tl in self.order:
            n = len(self.topology[tl].lanes)
            self.offsets[tl] = slice(offset, offset + n)
            offset += n
        self.n_lanes = offset
        self.storage = np.concatenate([self.topology[t].capacity for t in self.order])
        self.routing = np.zeros((offset, offset))
        for tl in self.order:
            for upstream, routing in self.topology[tl].upstream_routes.items():
                self.routing[self.offsets[tl], self.offsets[upstream]] += routing
        self._certificate_cache = None
        self._exact_values = {}

    def _coordinate(self, sim_time):
        self._certificate_cache = None
        self._exact_values = {}
        return super()._coordinate(sim_time)

    def _certificate_context(self, queues):
        if self._certificate_cache is None:
            q = np.concatenate([queues[t] for t in self.order])
            boundary = np.zeros(self.n_lanes)
            for tl in self.order:
                topo = self.topology[tl]
                for i in topo.boundary_indices:
                    boundary[self.offsets[tl].start + i] = (
                        self.boundary_rates[topo.lanes[i]] * config.CIMPC_INTERVAL
                    )
            self._certificate_cache = (q, boundary)
        return self._certificate_cache

    def _v(self, q):
        occupancy = np.maximum(q, 0) / np.maximum(self.storage, 1)
        return float(np.sum(occupancy**2) +
                     config.EXTERNAL_CIMPC_LYAPUNOV_OVERFLOW_WEIGHT *
                     np.sum(np.maximum(occupancy - 1, 0)**2))

    def _network_queue_lyapunov(self, queues, forecasts, first_flows=None):
        key = (id(queues), id(forecasts), None if first_flows is None else
               tuple(np.asarray(first_flows[t], dtype=float).tobytes() for t in self.order))
        if key not in self._exact_values:
            self._exact_values[key] = super()._network_queue_lyapunov(queues, forecasts, first_flows)
        return self._exact_values[key]

    def _vectorized_network_queue_lyapunov(self, queues, forecasts, first_flows=None):
        """Diagnostic only: mathematically equal, NOT bitwise equivalent."""
        q, boundary = self._certificate_context(queues)
        if first_flows is None:
            return self._v(q)
        f = np.concatenate([first_flows[t] for t in self.order])
        return self._v(q + boundary + self.routing @ f - f)


class StructuredLCIMPC(FastCertificateLCIMPC):
    """A callback selects one *already certified* joint candidate exactly once."""

    def __init__(self, conn, selector=None, candidate_limit=16):
        super().__init__(conn)
        self.selector = selector
        self.candidate_limit = int(candidate_limit)
        self.last_packet = None
        self.candidate_diagnostics = []
        self.groups = []
        assigned = set()
        for tl in self.order:
            for g, group in enumerate(self.topology[tl].receiving_groups):
                members = np.asarray(group.source_indices) + self.offsets[tl].start
                if assigned.intersection(members.tolist()):
                    raise ValueError("Differentiable reference requires disjoint receiving groups")
                assigned.update(members.tolist())
                self.groups.append((tl, g, members))
        for i in range(self.n_lanes):
            if i not in assigned:
                self.groups.append((None, None, np.asarray([i])))
        self.group_index = np.zeros(self.n_lanes, dtype=np.int64)
        for g, (_, _, members) in enumerate(self.groups):
            self.group_index[members] = g

    def _adaptive_parameters_for_update(self, *args):
        return {"retain_candidate_rollouts": True}

    def _postprocess_phase_choices(self, sim_time, queues, current_green,
                                   forecasts, phase_choices, payloads, solutions):
        started = time.perf_counter()
        # Calling the original zero-residual projection is deliberate: it
        # preserves timing, exceptional fallbacks and the causal robust state.
        base, base_forecasts = super()._postprocess_phase_choices(
            sim_time, queues, current_green, forecasts, phase_choices, payloads, solutions
        )
        if self.selector is None:
            return base, base_forecasts
        q, boundary = self._certificate_context(queues)
        base_f = np.concatenate([base_forecasts[t][:, 0] for t in self.order])
        base_next = np.maximum(q + boundary + self.routing @ base_f - base_f, 0)
        base_v = self._v(base_next)
        limit = self.last_lyapunov_limit
        costs = {t: np.asarray(solutions[t].first_phase_objectives) for t in self.order}
        base_cost = sum(costs[t][base[t]] for t in self.order)
        scale_cost = max(abs(base_cost), 1.0)
        scale_flow = max(float(np.sum(base_f)), 1.0)
        candidates = [dict(base)]
        candidate_flows = [base_f]
        seen = {tuple(base[t] for t in self.order)}

        def make_flow(choices):
            return np.concatenate([
                self._first_phase_flow(t, solutions[t], choices[t], forecasts)
                for t in self.order
            ])

        def admissible(choices, f):
            next_q = np.maximum(q + boundary + self.routing @ f - f, 0)
            if self._v(next_q) > limit + 1e-9:
                return False
            if np.any(np.maximum(next_q - self.storage, 0) >
                      np.maximum(base_next - self.storage, 0) +
                      config.EXTERNAL_CIMPC_OVERFLOW_TOLERANCE):
                return False
            if np.any(f > q + boundary + self.routing @ f + 1e-7):
                return False
            # Recompute receiving bounds using the entire trial action, not
            # the nominal upstream messages used to generate local traces.
            trial = {t: base_forecasts[t].copy() for t in self.order}
            for t in self.order:
                trial[t][:, 0] = f[self.offsets[t]]
            for t in self.order:
                receiving = self._receiving_capacity_forecast(t, queues, trial)
                for g, group in enumerate(self.topology[t].receiving_groups):
                    sent = np.sum(f[self.offsets[t]][group.source_indices])
                    if sent > receiving[g, 0] + config.EXTERNAL_CIMPC_RECEIVING_TOLERANCE:
                        return False
            return True

        alternatives = []
        # A deterministic rotating scan gives every junction coverage. No
        # future SUMO states or test outcomes enter candidate construction.
        rotation = int(sim_time / config.CIMPC_INTERVAL) % len(self.order)
        rotated = self.order[rotation:] + self.order[:rotation]
        for t in rotated:
            mask, _, _ = self._candidate_feasibility(t, payloads[t], solutions[t], base[t])
            if self.time_in_phase[t] + 1e-9 < config.CIMPC_MIN_GREEN:
                continue
            for p in np.argsort(costs[t], kind="stable"):
                p = int(p)
                if p == base[t] or not mask[p] or not np.isfinite(costs[t][p]):
                    continue
                if (p == current_green[t] and self.time_in_phase[t] +
                        config.CIMPC_INTERVAL > config.EXTERNAL_CIMPC_MAX_GREEN + 1e-9):
                    continue
                trial = dict(base)
                trial[t] = p
                f = make_flow(trial)
                if admissible(trial, f):
                    alternatives.append((t, p, trial, f))
        # Prefer breadth (one alternative per junction) before extra phases.
        alternatives.sort(key=lambda a: (rotated.index(a[0]), costs[a[0]][a[1]]))
        first, rest, used = [], [], set()
        for item in alternatives:
            (rest if item[0] in used else first).append(item)
            used.add(item[0])
        for _, _, trial, f in first + rest:
            if len(candidates) >= self.candidate_limit - 3:
                break
            key = tuple(trial[t] for t in self.order)
            if key not in seen:
                seen.add(key)
                candidates.append(trial)
                candidate_flows.append(f)
        compound = dict(base)
        for t, p, _, _ in first:
            trial = dict(compound)
            trial[t] = p
            f = make_flow(trial)
            key = tuple(trial[x] for x in self.order)
            if admissible(trial, f):
                compound = trial
                if key not in seen and len(candidates) < self.candidate_limit:
                    seen.add(key)
                    candidates.append(dict(trial))
                    candidate_flows.append(f)

        features, uppers, terminals, nexts = [], [], [], []
        group_caps = []
        for t, g, members in self.groups:
            group_caps.append(max(float(payloads[t][3][g, 0]), 0) if t is not None
                              else float(self.storage[members[0]] + q[members[0]] + 100))
        for choices, f in zip(candidates, candidate_flows):
            terminal, upper = [], []
            for t in self.order:
                topo, payload = self.topology[t], payloads[t]
                p = choices[t]
                roll = solutions[t].first_phase_rollouts.get(p)
                if roll is None:
                    raise RuntimeError("Stage-5 full candidate trace missing")
                terminal.extend(roll["queue"][-1])
                green = config.CIMPC_INTERVAL - (
                    config.CIMPC_YELLOW if p != current_green[t] else 0)
                upper.extend(np.minimum(np.maximum(payload[1] + payload[2][:, 0], 0),
                             config.CIMPC_SATURATION_PER_LANE * green * topo.service[:, p]))
            terminal = np.asarray(terminal) / np.maximum(self.storage, 1)
            next_q = np.maximum(q + boundary + self.routing @ f - f, 0)
            occ = next_q / np.maximum(self.storage, 1)
            cost = sum(costs[t][choices[t]] for t in self.order)
            features.append([
                (cost - base_cost) / scale_cost,
                (self._v(next_q) - base_v) / max(base_v, 1),
                np.mean(occ), np.max(occ), np.mean(terminal), np.max(terminal),
                (np.sum(f) - np.sum(base_f)) / scale_flow,
                np.mean([choices[t] != current_green[t] for t in self.order]),
                np.mean([choices[t] != base[t] for t in self.order]),
                np.mean(terminal**2), np.mean(occ**2),
                (limit - self._v(next_q)) / max(base_v, 1),
            ])
            terminals.append(terminal)
            nexts.append(occ)
            uppers.append(upper)
        packet = {
            "state": self.last_external_observation.copy(),
            "features": np.asarray(features, dtype=np.float32),
            "terminal": np.asarray(terminals, dtype=np.float32),
            "next": np.asarray(nexts, dtype=np.float32),
            "flow": np.asarray(candidate_flows, dtype=np.float32) / 10,
            "upper": np.asarray(uppers, dtype=np.float32) / 10,
            "group_caps": np.asarray(group_caps, dtype=np.float32) / 10,
            "group_index": self.group_index.copy(),
            "lane_state": np.stack([q / np.maximum(self.storage, 1),
                boundary / 10, base_f / 10, self.storage / 100], axis=-1).astype(np.float32),
        }
        self.last_packet = packet
        inference_start = time.perf_counter()
        selected = int(self.selector(packet))
        inference_s = time.perf_counter() - inference_start
        if not 0 <= selected < len(candidates):
            raise ValueError("Actor selected an invalid candidate")
        chosen, f = candidates[selected], candidate_flows[selected]
        result = {t: base_forecasts[t].copy() for t in self.order}
        for t in self.order:
            result[t][:, 0] = f[self.offsets[t]]
        # Preserve tail messages and solver warm starts for exact candidate-0
        # equivalence. Alternative local tail traces are features, not messages.
        if selected:
            self.last_lyapunov_applied = self._network_queue_lyapunov(
                queues, forecasts, {t: f[self.offsets[t]] for t in self.order})
            self._finalize_lyapunov_projection(self.last_lyapunov_applied)
            self.last_correction_by_tl = {t: int(chosen[t] != phase_choices[t]) for t in self.order}
            self.last_correction_rate = float(np.mean(list(self.last_correction_by_tl.values())))
            self.last_safety_violation = max(self.last_lyapunov_applied - limit, 0)
        self.candidate_diagnostics.append({
            "simulation_time": sim_time, "candidate_count": len(candidates),
            "admissible_single_alternatives": len(alternatives), "selected": selected,
            "changed_junctions": sum(chosen[t] != base[t] for t in self.order),
            "baseline_lyapunov": base_v, "selected_lyapunov": self.last_lyapunov_applied,
            "lyapunov_limit": limit, "inherited_baseline_violation": max(base_v - limit, 0),
            "candidate_violation": max(self.last_lyapunov_applied - limit, 0) if selected else 0,
            "actor_inference_s": inference_s,
            "postprocess_total_s": time.perf_counter() - started,
        })
        return chosen, result
