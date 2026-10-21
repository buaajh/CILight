"""Joint budget/control coordination with auditable first-step certificates.

Stage 1 research implementation: bounded discrete neighborhood optimization,
exact conditional budget projection and a fixed predicted-cost acceptance
guard. No global discrete optimum or recursive-feasibility claim. Frozen V2
is called once; its inherited warnings are preserved, never certified anew.
"""
from __future__ import annotations
import time
import numpy as np
import config
from controllers.structured_l_cimpc import FastCertificateLCIMPC
from controllers.external_cooperative_ci_mpc import queue_lyapunov


def project_budget(desired, lower, total, tolerance=1e-9):
    """Euclidean projection onto b>=lower, sum(b)=total, conditional on action.

    Returns primal/dual certificate. The discrete action itself is not
    differentiated. None means this action has no feasible budget allocation.
    """
    desired, lower = np.asarray(desired, float), np.asarray(lower, float)
    if desired.shape != lower.shape or desired.ndim != 1 or not desired.size:
        raise ValueError('Budget shape mismatch')
    if not (np.isfinite(desired).all() and np.isfinite(lower).all() and np.isfinite(total)):
        raise ValueError('Nonfinite budget')
    if (lower < 0).any() or total < 0:
        raise ValueError('Negative budget')
    if lower.sum() > total:
        return None
    lo = float(np.min(desired-lower)-total-1)
    hi = float(np.max(desired-lower)+1)
    for _ in range(64):
        tau = .5*(lo+hi)
        if np.maximum(lower, desired-tau).sum() > total:
            lo = tau
        else:
            hi = tau
    tau = hi
    b = np.maximum(lower, desired-tau)
    mu = np.maximum(lower-desired+tau, 0)
    residual = max(abs(float(b.sum()-total)), float(np.max(np.maximum(lower-b, 0))),
                   float(np.max(np.abs(b-desired+tau-mu))), float(np.max(np.abs(mu*(b-lower)))))
    if residual > tolerance*max(1., total):
        raise RuntimeError('Budget KKT certificate failed')
    return dict(budget=b, equality_dual=tau, local_duals=mu,
                distortion=float(.5*np.sum((b-desired)**2)), kkt_residual=residual)


def desired_budget(base_values, total, residual):
    base = np.maximum(np.asarray(base_values, float), 0)
    residual = np.asarray(residual, float)
    if residual.shape != base.shape or not np.isfinite(residual).all():
        raise ValueError('Bad coordination preference')
    prior = base + max(.02*total/len(base), 1e-12)
    logits = np.log(prior)+np.clip(residual, -4., 4.)
    weights = np.exp(logits-logits.max())
    return total*weights/weights.sum()


class JointBudgetLCIMPC(FastCertificateLCIMPC):
    """Neural preferences enter a joint (phase,budget) optimization objective.

    The teacher minimizes fixed physical proxy J. The zero/learned versions
    minimize J plus budget projection distortion subject to J<=J(V2).
    Search effort, constraints and J are fixed across all variants.
    """
    def __init__(self, conn, selector=None, mode='teacher', proposal_cap=64):
        super().__init__(conn)
        self.selector, self.mode = selector, mode
        self.proposal_cap = int(proposal_cap)
        self.candidate_diagnostics, self.joint_diagnostics, self.teacher_records = [], [], []
        self.node_index = {t: i for i, t in enumerate(self.order)}
        self.adjacency = np.zeros((len(self.order), len(self.order)), np.float32)
        self.edges = []
        self.receiving_shared = {}
        for t in self.order:
            for upstream in self.topology[t].upstream_routes:
                self.adjacency[self.node_index[t], self.node_index[upstream]] = 1
            for group in self.topology[t].receiving_groups:
                key = (group.downstream_tl, tuple(sorted(group.target_indices)))
                self.receiving_shared.setdefault(key, []).append((t, tuple(group.source_indices)))
        for i in range(len(self.order)):
            for j in range(i+1, len(self.order)):
                if self.adjacency[i, j] or self.adjacency[j, i]:
                    self.edges.append((self.order[i], self.order[j]))

    def _adaptive_parameters_for_update(self, *args):
        return {'retain_candidate_rollouts': True}

    def _postprocess_phase_choices(self, sim_time, queues, current_green, forecasts,
                                   phase_choices, payloads, solutions):
        started = time.perf_counter()
        base, base_forecasts = super()._postprocess_phase_choices(
            sim_time, queues, current_green, forecasts, phase_choices, payloads, solutions)
        limit = float(self.last_lyapunov_limit)
        base_flows = {t: base_forecasts[t][:, 0].copy() for t in self.order}
        def network(flows):
            trial = {t: base_forecasts[t].copy() for t in self.order}
            for t in self.order:
                trial[t][:, 0] = flows[t]
            available = {t: queues[t]+self._arrival_forecast(t, trial)[:, 0] for t in self.order}
            nxt = {t: np.maximum(available[t]-flows[t], 0) for t in self.order}
            local = np.asarray([queue_lyapunov(nxt[t], self.topology[t].capacity,
                        config.EXTERNAL_CIMPC_LYAPUNOV_OVERFLOW_WEIGHT) for t in self.order])
            return nxt, local, available, trial
        base_next, base_local, _, _ = network(base_flows)
        current_l = sum(queue_lyapunov(queues[t], self.topology[t].capacity,
                         config.EXTERNAL_CIMPC_LYAPUNOV_OVERFLOW_WEIGHT) for t in self.order)
        scale_l, scale_q = max(current_l, 1.), max(sum(np.sum(queues[t]) for t in self.order), 1.)
        base_cost = sum(float(solutions[t].first_phase_objectives[base[t]]) for t in self.order)
        scale_cost = max(abs(base_cost), 1.)
        def objective(nxt, local, choices):
            # Joint first-step backlog/Lyapunov plus the existing local horizon
            # cost heuristic. Local tails are not jointly certified forecasts.
            horizon = sum(float(solutions[t].first_phase_objectives[choices[t]]) for t in self.order)
            return .5*float(local.sum())/scale_l + .5*sum(float(np.sum(nxt[t])) for t in self.order)/scale_q + .05*(horizon-base_cost)/scale_cost
        base_j = objective(base_next, base_local, base)
        packet = {'node_features': np.asarray([[base_local[i]/max(limit, 1.),
            float(np.mean(queues[t]/np.maximum(self.topology[t].capacity, 1))),
            float(np.max(queues[t]/np.maximum(self.topology[t].capacity, 1))),
            float(np.sum(base_flows[t]))/max(float(np.sum(queues[t])), 1),
            float(np.mean(self._arrival_forecast(t, base_forecasts)[:, 0]))/10,
            min(self.time_in_phase[t]/60., 2.), np.log1p(max(limit, 0)),
            max(limit-float(base_local.sum()), 0)/max(limit, 1.)]
            for i, t in enumerate(self.order)], np.float32),
            'adjacency': self.adjacency.copy(), 'base_values': base_local.astype(np.float32), 'total': max(limit, 0.)}
        raw = np.asarray(self.selector(packet), float) if self.selector else np.zeros(len(self.order))
        desired = desired_budget(base_local, max(limit, 0.), raw)
        options = {}
        for t in self.order:
            opts = []
            # Do not prune by stale neighbor receiving capacity. All proposals
            # are checked on the actual joint flow below.
            costs = np.asarray(solutions[t].first_phase_objectives)
            for p in np.argsort(costs, kind='stable'):
                p = int(p)
                if not np.isfinite(costs[p]):
                    continue
                if p != current_green[t] and self.time_in_phase[t]+1e-9 < config.CIMPC_MIN_GREEN:
                    continue
                if p == current_green[t] and self.time_in_phase[t]+config.CIMPC_INTERVAL > config.EXTERNAL_CIMPC_MAX_GREEN+1e-9:
                    continue
                opts.append(p)
            options[t] = opts
        rejected = dict(physical=0, lyapunov=0, performance=0)
        evaluated, feasible = {}, []
        baseline_failures = []
        baseline_details = {}
        def evaluate(choices, initial=False):
            key = tuple(choices[t] for t in self.order)
            if key in evaluated:
                return evaluated[key]
            flows = base_flows if initial else {t: self._first_phase_flow(t, solutions[t], choices[t], forecasts) for t in self.order}
            nxt, local, available, trial = network(flows)
            physical = True
            failures = set()
            for t in self.order:
                if initial:
                    baseline_details[t] = dict(phase=int(choices[t]), current_phase=int(current_green[t]),
                        elapsed_green=float(self.time_in_phase[t]), allowed_phases=options[t],
                        finite_cost=bool(np.isfinite(solutions[t].first_phase_objectives[choices[t]])),
                        available_excess=max(float(np.max(flows[t]-available[t])), 0.),
                        min_green_violation=bool(choices[t] != current_green[t] and self.time_in_phase[t]+1e-9 < config.CIMPC_MIN_GREEN),
                        max_green_violation=bool(choices[t] == current_green[t] and self.time_in_phase[t]+config.CIMPC_INTERVAL > config.EXTERNAL_CIMPC_MAX_GREEN+1e-9))
                if choices[t] not in options[t]:
                    physical = False
                    failures.add('timing_or_nonfinite_cost')
                green = config.CIMPC_INTERVAL-(config.CIMPC_YELLOW if choices[t] != current_green[t] else 0.)
                cap = config.CIMPC_SATURATION_PER_LANE*green*self.topology[t].service[:, choices[t]]
                f = flows[t]
                if np.any(f < -1e-7) or np.any(f > cap+1e-7) or np.any(f > available[t]+1e-7):
                    physical = False
                    if np.any(f < -1e-7): failures.add('negative_flow')
                    if np.any(f > cap+1e-7): failures.add('signal_capacity')
                    if np.any(f > available[t]+1e-7): failures.add('available_vehicles')
                if np.any(np.maximum(nxt[t]-self.topology[t].capacity, 0) >
                    np.maximum(base_next[t]-self.topology[t].capacity, 0)+config.EXTERNAL_CIMPC_OVERFLOW_TOLERANCE):
                    physical = False
                    failures.add('overflow')
                receiving = self._receiving_capacity_forecast(t, queues, trial)
                for g, group in enumerate(self.topology[t].receiving_groups):
                    if sum(f[group.source_indices]) > receiving[g, 0]+config.EXTERNAL_CIMPC_RECEIVING_TOLERANCE:
                        physical = False
                        failures.add('individual_receiving')
            # Shared receivers: several upstream senders cannot each consume
            # the same free space. Deduplicate repeated source-lane entries.
            for (downstream, targets), senders in self.receiving_shared.items():
                targets = np.asarray(targets, int)
                free = np.maximum(self.topology[downstream].capacity[targets]-queues[downstream][targets], 0).sum()
                capacity = max(float(free+flows[downstream][targets].sum()-config.CIMPC_ASYNC_STORAGE_MARGIN), 0.)
                lanes = {(t, lane) for t, indices in senders for lane in indices}
                sent = sum(float(flows[t][lane]) for t, lane in lanes)
                if sent > capacity+config.EXTERNAL_CIMPC_RECEIVING_TOLERANCE:
                    physical = False
                    failures.add('shared_receiving')
            exact = self._network_queue_lyapunov(queues, forecasts, flows)
            proj = project_budget(desired, local, limit) if limit >= 0 else None
            j = objective(nxt, local, choices)
            reason = 'physical' if not physical else ('lyapunov' if proj is None or exact > limit+1e-9 else
                     ('performance' if j > base_j+1e-9 else None))
            if initial:
                baseline_failures.extend(sorted(failures))
                if proj is None: baseline_failures.append('local_sum_exceeds_budget')
                if exact > limit+1e-9: baseline_failures.append('exact_network_exceeds_budget')
            if reason:
                rejected[reason] += 1
                evaluated[key] = None
                return None
            value = j if self.mode == 'teacher' else j+.2*proj['distortion']/max(limit*limit, 1e-6)
            result = dict(choices=dict(choices), flows=flows, local=local, exact=exact, j=j, value=value, projection=proj)
            evaluated[key] = result
            feasible.append(result)
            return result
        witness = evaluate(base, initial=True)
        # If the inherited baseline is not certified, preserve it and report
        # the warning; never pretend its existence proves feasibility.
        chosen = witness
        if witness is not None:
            rotation = int(sim_time/config.CIMPC_INTERVAL) % len(self.order)
            rotated = self.order[rotation:]+self.order[:rotation]
            for t in rotated:
                for p in options[t]:
                    if p == chosen['choices'][t] or len(evaluated) >= self.proposal_cap:
                        continue
                    proposal = dict(chosen['choices']); proposal[t] = p
                    candidate = evaluate(proposal)
                    if candidate is not None and candidate['value'] < chosen['value']-1e-12:
                        chosen = candidate
            # Neighbor-pair moves can traverse a barrier that no single
            # junction can pass while respecting the network budget.
            edges = self.edges[rotation % max(len(self.edges), 1):]+self.edges[:rotation % max(len(self.edges), 1)]
            for a, b in edges:
                for pa in options[a]:
                    for pb in options[b]:
                        if len(evaluated) >= self.proposal_cap:
                            break
                        if pa == chosen['choices'][a] or pb == chosen['choices'][b]:
                            continue
                        proposal = dict(chosen['choices']); proposal[a] = pa; proposal[b] = pb
                        candidate = evaluate(proposal)
                        if candidate is not None and candidate['value'] < chosen['value']-1e-12:
                            chosen = candidate
            # Select the best visited feasible pair; only this finite visited
            # set is covered by the reported objective certificate.
            chosen = min(feasible, key=lambda a: a['value'])
        changed = sum(chosen['choices'][t] != base[t] for t in self.order) if chosen else 0
        result = base_forecasts
        if chosen is not None and changed:
            result = {t: base_forecasts[t].copy() for t in self.order}
            for t in self.order:
                result[t][:, 0] = chosen['flows'][t]
            self.last_lyapunov_applied = chosen['exact']
            self._finalize_lyapunov_projection(chosen['exact'])
            self.last_correction_by_tl = {t: int(chosen['choices'][t] != phase_choices[t]) for t in self.order}
            self.last_correction_rate = float(np.mean(list(self.last_correction_by_tl.values())))
            self.last_safety_violation = max(chosen['exact']-limit, 0.)
        projection = chosen['projection'] if chosen else None
        if self.mode == 'teacher' and chosen is not None and limit > 1e-8:
            self.teacher_records.append(dict(node_features=packet['node_features'], adjacency=self.adjacency,
                 base_values=packet['base_values'], total=limit,
                 target_share=(projection['budget']/limit).astype(np.float32),
                 changed=bool(changed), model_cost_gain=base_j-chosen['j']))
        diagnostic = dict(time=sim_time, baseline_certified=witness is not None,
            baseline_failures=baseline_failures, changed_junctions=changed,
            baseline_details=baseline_details, baseline_local_sum_gap=float(base_local.sum()-limit),
            evaluated=len(evaluated), feasible=len(feasible), rejected=rejected, limit=limit,
            desired=desired.tolist(), baseline_local=base_local.tolist(),
            local=chosen['local'].tolist() if chosen else None,
            budget=projection['budget'].tolist() if projection else None,
            local_duals=projection['local_duals'].tolist() if projection else None,
            kkt_residual=projection['kkt_residual'] if projection else None,
            budget_sum_error=float(projection['budget'].sum()-limit) if projection else None,
            predicted_base_cost=base_j, predicted_selected_cost=chosen['j'] if chosen else base_j,
            restricted_objective_gap=chosen['value']-min(c['value'] for c in feasible) if chosen else None,
            certificate_scope='first-step joint model; no recursive-feasibility or full-network optimum claim')
        self.joint_diagnostics.append(diagnostic)
        self.candidate_diagnostics.append(dict(simulation_time=sim_time, candidate_count=max(len(feasible), 1),
            selected=int(bool(changed)), changed_junctions=changed,
            inherited_baseline_violation=max(float(base_local.sum())-limit, 0.),
            candidate_violation=max(chosen['exact']-limit, 0.) if changed else 0.,
            actor_inference_s=0., postprocess_total_s=time.perf_counter()-started))
        return chosen['choices'] if chosen else base, result
