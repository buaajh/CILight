"""Offline validation-only early-stop decisions, not a training launcher.

The threshold is practical, not a statistical significance test. Checkpoints
are selected only from the frozen validation panel. Caller must verify artifacts
and implement safe checkpoint-boundary shutdown before enabling this policy.
"""
from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class Policy:
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    training_seeds: tuple[int, ...]
    max_episodes: int
    interval: int = 10
    minimum_episodes: int = 60
    patience: int = 6
    relative_att_improvement: float = .01
    secondary_tolerance: float = .02
    completion_tolerance: float = .01
    min_improved_seeds: int = 3

    def __post_init__(self):
        panels = [self.validation_seeds, self.test_seeds, self.training_seeds]
        if any(not p or len(set(p)) != len(p) for p in panels):
            raise ValueError('Empty or duplicate seed panel')
        if any(set(panels[i]) & set(panels[j]) for i in range(3) for j in range(i)):
            raise ValueError('Training, validation and test seed panels must be disjoint')
        if any(type(v) is not int for v in (self.interval, self.minimum_episodes, self.max_episodes, self.patience, self.min_improved_seeds)):
            raise ValueError('Episode/count policy values must be integers')
        if (self.interval <= 0 or self.minimum_episodes < self.interval
                or self.max_episodes < self.minimum_episodes or self.patience < 1
                or not 1 <= self.min_improved_seeds <= len(self.validation_seeds)):
            raise ValueError('Invalid budget or patience')
        if self.max_episodes % self.interval or self.minimum_episodes % self.interval:
            raise ValueError('Budget and warmup must align with validation boundaries')
        if not 0 < self.relative_att_improvement < 1:
            raise ValueError('Invalid improvement threshold')
        if not 0 <= self.secondary_tolerance < 1 or not 0 <= self.completion_tolerance < 1:
            raise ValueError('Invalid quality tolerance')


def digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_record(record, policy):
    if record['split'] != 'validation':
        raise ValueError('Only validation observations may influence early stopping')
    if record.get('status') != 'complete' or not record.get('artifact_verified'):
        raise ValueError('Incomplete or unverified validation is not a plateau observation')
    if not digest(record['checkpoint_sha256']) or not digest(record['protocol_sha256']):
        raise ValueError('Missing checkpoint/protocol binding')
    if not isinstance(record['episode'], int) or record['episode'] <= 0:
        raise ValueError('Invalid episode')
    rows = record['seeds']
    if len(rows) != len(policy.validation_seeds) or sorted(r['seed'] for r in rows) != sorted(policy.validation_seeds):
        raise ValueError('Validation panel changed, missing or duplicated')
    for row in rows:
        if not digest(row['artifact_sha256']):
            raise ValueError('Missing artifact binding')
        for metric in ('att', 'awt', 'adt', 'completed', 'demand', 'teleports'):
            if not isinstance(row[metric], (int, float)) or not math.isfinite(row[metric]) or row[metric] < 0:
                raise ValueError('Missing/nonfinite/negative metric')
        if row['demand'] <= 0 or row['completed'] <= 0 or row['completed'] > row['demand']:
            raise ValueError('Invalid demand/completion counts')
        for count in ('completed', 'demand', 'teleports'):
            if row[count] != int(row[count]):
                raise ValueError('Counts must be integral')
    reward = record.get('training_reward_mean')
    if reward is not None and not math.isfinite(reward):
        raise ValueError('Nonfinite training reward requires diagnosis')


def mean(record, metric):
    return sum(r[metric] for r in record['seeds']) / len(record['seeds'])


def decide(history, policy):
    """Recompute deterministically from append-only records; fail closed on errors.

    First validation is the reference, not proof of superiority to any baseline.
    Upper limits must be explicitly supplied by the caller; none are implicit.
    No file, process, simulator, model or scheduler is changed by this function.
    """
    if not history:
        return {'action': 'continue', 'reason': 'await_first_validation'}
    best = None
    stale = 0
    seen_checkpoints = set()
    protocol = history[0]['protocol_sha256']
    demand = None
    trace = []
    terminal = None
    for index, record in enumerate(history):
        if terminal:
            raise ValueError('History continues after a stop decision')
        validate_record(record, policy)
        if record['episode'] != (index + 1) * policy.interval:
            raise ValueError('Missing, repeated or out-of-order validation point')
        if record['episode'] > policy.max_episodes:
            raise ValueError('History exceeds configured budget')
        if record['protocol_sha256'] != protocol:
            raise ValueError('Frozen protocol changed')
        if record['checkpoint_sha256'] in seen_checkpoints:
            raise ValueError('Duplicate checkpoint is not an independent validation point')
        seen_checkpoints.add(record['checkpoint_sha256'])
        panel = {r['seed']: r['demand'] for r in record['seeds']}
        if demand is not None and panel != demand:
            raise ValueError('Per-seed demand changed')
        demand = panel
        if best is None:
            best = record
            trace.append({'episode': record['episode'], 'accepted': True, 'reason': 'initial_reference'})
        else:
            previous = {r['seed']: r for r in best['seeds']}
            improved = sum(r['att'] < previous[r['seed']]['att'] for r in record['seeds'])
            att_ok = mean(record, 'att') < mean(best, 'att') and mean(record, 'att') <= mean(best, 'att') * (1-policy.relative_att_improvement)
            # Guard both the incumbent and initial reference to prevent cumulative
            # degradation through individually small accepted steps.
            guards = []
            for reference in (best, history[0]):
                guards.extend(mean(record, m) <= mean(reference, m) * (1+policy.secondary_tolerance) for m in ('awt', 'adt'))
                guards.append(mean(record, 'completed') >= mean(reference, 'completed') * (1-policy.completion_tolerance))
                ref_rows = {r['seed']: r for r in reference['seeds']}
                guards.append(all(r['teleports'] <= ref_rows[r['seed']]['teleports'] for r in record['seeds']))
            accepted = att_ok and improved >= policy.min_improved_seeds and all(guards)
            trace.append({'episode': record['episode'], 'accepted': accepted,
                          'att_improved_seeds': improved, 'quality_guards_pass': all(guards)})
            if accepted:
                best, stale = record, 0
            else:
                stale += 1
        if record['episode'] >= policy.max_episodes:
            terminal = 'budget_cap_not_convergence'
        elif record['episode'] >= policy.minimum_episodes and stale >= policy.patience:
            terminal = 'validation_plateau_not_statistical_convergence'
    return {'action': 'stop_at_committed_checkpoint' if terminal else 'continue',
            'reason': terminal or 'validation_monitoring', 'last_episode': history[-1]['episode'],
            'selected_episode': best['episode'], 'selected_checkpoint_sha256': best['checkpoint_sha256'],
            'stale_validations': stale, 'trace': trace}
