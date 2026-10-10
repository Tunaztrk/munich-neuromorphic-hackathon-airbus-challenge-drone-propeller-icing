"""Shared causal drop detector and retrospective clean-training intervals.

An electrical drop is a shedding *candidate*, not proof of ice detachment.
No camera channel is inspected. Confirmation uses only already observed rows.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class EventPolicy:
    minimum_drop_ampere: float = 1.0
    drop_window_seconds: float = 1.0
    confirmation_seconds: float = 2.0
    persistence_fraction: float = 0.8
    cooldown_seconds: float = 3.0
    command_rpm_absolute: float = 100.0
    command_rpm_relative: float = 0.05
    command_throttle_percent: float = 5.0
    motor_minimum_rpm: float = 100.0


class DropDetector:
    def __init__(self, policy=None):
        self.policy = policy or EventPolicy()
        self.history = deque()
        self.pending = None
        self.cooldown_until = -np.inf
        self.last_time = None
        self.audit = []

    def update(self, timestamp, row):
        p = self.policy
        if self.last_time is not None and timestamp <= self.last_time:
            raise ValueError('Detector timestamps must increase.')
        self.last_time = float(timestamp)
        row = np.asarray(row, dtype=float)
        if not np.isfinite(row[0]):
            raise ValueError('Reset detector at missing current.')
        previous = list(self.history)
        self.history.append((float(timestamp), row.copy()))
        while self.history and timestamp-self.history[0][0] > 5.0:
            self.history.popleft()
        recent = [(t, r) for t, r in previous if timestamp-t <= p.drop_window_seconds+1e-8]
        if self.pending is None and timestamp >= self.cooldown_until and len(recent) >= 3:
            baseline = float(np.median([r[0] for _, r in recent]))
            if baseline-row[0] > p.minimum_drop_ampere:
                samples = recent+[(timestamp, row)]
                steps = np.diff([r[0] for _, r in samples])
                worst = int(np.argmin(steps))
                # The interval endpoint is an observed timestamp, not an oracle.
                event_time = float(samples[worst+1][0])
                self.pending = {'event_time': event_time, 'candidate_time': float(timestamp),
                                'baseline_current': baseline, 'before': recent,
                                'largest_step': float(steps[worst])}
        if self.pending is None:
            return None
        q = self.pending
        if len(previous):
            step = float(row[0]-previous[-1][1][0])
            if timestamp-q['candidate_time'] <= p.drop_window_seconds and step < q['largest_step']:
                q['event_time'], q['largest_step'] = float(timestamp), step
        if timestamp-q['candidate_time'] < p.confirmation_seconds-1e-8:
            return None
        post = [(t, r) for t, r in self.history if t >= q['candidate_time']+0.5-1e-8]
        values = np.asarray([r[0] for _, r in post])
        drop = float(q['baseline_current']-np.median(values)) if len(values) else 0.
        persistent = len(values) > 0 and float(np.mean(q['baseline_current']-values > p.minimum_drop_ampere)) >= p.persistence_fraction
        surrounding = [r for _, r in q['before']]+[r for _, r in post]
        settings = np.asarray(surrounding)
        target = settings[:, 2][np.isfinite(settings[:, 2])]
        throttle = settings[:, 3][np.isfinite(settings[:, 3])]
        commanded = len(target) > 1 and np.ptp(target) > max(p.command_rpm_absolute, p.command_rpm_relative*abs(float(np.median(target))))
        # Throttle is often a control response to shedding. Use it as a command
        # proxy only when the requested RPM is unavailable, not when it is steady.
        if len(target) == 0 and len(throttle) > 1:
            commanded = np.ptp(throttle) > p.command_throttle_percent
        rpm = settings[:, 1][np.isfinite(settings[:, 1])]
        motor_stop = len(rpm) > 0 and np.min(rpm) < p.motor_minimum_rpm
        status = ('command_change' if commanded else 'motor_stop' if motor_stop else
                  'confirmed_drop' if persistent and drop > p.minimum_drop_ampere else 'transient')
        result = {k: v for k, v in q.items() if k not in {'before', 'largest_step'}}
        result.update(detected_time=float(timestamp), drop_ampere=drop, status=status,
                      confirmation_delay_seconds=float(timestamp-q['event_time']))
        self.audit.append(result)
        self.pending = None
        self.cooldown_until = timestamp+p.cooldown_seconds
        return result


def detect_segments(segments, policy=None):
    events, audit = [], []
    for segment in segments:
        detector = DropDetector(policy)
        for t, row in zip(segment.time, segment.raw):
            event = detector.update(float(t), row)
            if event is not None:
                event.update(parent_trial_id=int(segment.trial), source_segment=segment.key)
                audit.append(event.copy())
                if event['status'] == 'confirmed_drop':
                    events.append(event.copy())
    for i, event in enumerate(events, 1):
        event['event_id'] = f'DROP_{i:03d}'
    return events, audit


def jitter(time, current):
    """Detrended 5–95 percentile residual span, in amperes."""
    if len(current) < 5:
        return None
    residual = current-np.polyval(np.polyfit(time-time[0], current, 1), time-time[0])
    return float(np.quantile(residual, .95)-np.quantile(residual, .05))


def transition_policy(segments, events, training_trials):
    lookup = {s.key: s for s in segments}
    profiles = []
    for e in events:
        s = lookup[e['source_segment']]
        t, y = s.time, s.raw[:, 0]
        def span(a, b, t=t, y=y):
            mask = (t >= a) & (t < b)
            return jitter(t[mask], y[mask])
        base = span(e['event_time']-15., e['event_time']-5.)
        before = span(e['event_time']-3., e['event_time'])
        after = span(e['detected_time'], e['detected_time']+3.)
        threshold = max(.5, 3.*(base or 0.))
        profiles.append({'event_id': e['event_id'], 'parent_trial_id': e['parent_trial_id'],
                             'baseline_jitter_ampere': base, 'pre_jitter_ampere': before,
                             'post_jitter_ampere': after, 'threshold_ampere': threshold,
                             'pre_oscillation': before is not None and before > threshold,
                             'post_oscillation': after is not None and after > threshold})
    training = [r for r in profiles if r['parent_trial_id'] in training_trials]
    # Repeated pre-event oscillation supports targeted retrospective exclusions,
    # rather than deleting three seconds from every otherwise quiet event.
    enough = len(training) >= 3
    pre_fraction = sum(r['pre_oscillation'] for r in training)/max(1, len(training))
    post_fraction = sum(r['post_oscillation'] for r in training)/max(1, len(training))
    policy = {'pre_guard_seconds': 3. if sum(r['pre_oscillation'] for r in training) >= 3 else 0.,
                  'pre_guard_scope': 'flagged_events_only',
                  'post_guard_seconds': 3. if enough and post_fraction >= .5 else 0.,
                  'training_event_count': len(training), 'pre_oscillation_fraction': pre_fraction,
                  'post_oscillation_fraction': post_fraction,
                  'criterion': 'Pre: at least three flagged training events, then exclude only flagged events. Post: at least three training events and >=50% flagged. Jitter > max(0.5 A, 3x baseline).'}
    return policy, profiles


def event_intervals(events, guard, profiles=None):
    flags = {r['event_id']: r['pre_oscillation'] for r in (profiles or [])}
    return [{'parent_trial_id': e['parent_trial_id'], 'event_id': e['event_id'],
                 'start': e['event_time']-(guard['pre_guard_seconds'] if flags.get(e['event_id'], False) else 0.),
                 'end': e['detected_time']+guard['post_guard_seconds']} for e in events]


def clean_ranges(segment, intervals):
    """Return index ranges. Original parent IDs and raw channels stay intact."""
    eligible = np.ones(len(segment.time), dtype=bool)
    for interval in intervals:
        if interval['parent_trial_id'] == segment.trial:
            eligible &= ~((segment.time >= interval['start']-1e-8) & (segment.time <= interval['end']+1e-8))
    edges = np.diff(np.r_[False, eligible, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def policy_dict(policy=None):
    return asdict(policy or EventPolicy())
