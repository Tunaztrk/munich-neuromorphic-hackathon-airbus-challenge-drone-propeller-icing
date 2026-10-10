"""Delayed supervised adaptation of the final linear decoder only."""
from __future__ import annotations

import copy
from collections import deque

import numpy as np
import torch


class DelayedHead:
    """Targets arrive one observation at a time; issued forecasts never change.

    Encoder representations are detached and cached. No SNN weights, decay,
    first decoder layer, feature statistics, or previous predictions are updated.
    Call observe(), predict()/issue(), then maybe_update() in that order.
    """
    def __init__(self, linear, horizon, dt, target_mean, target_scale, learning_rate=0.,
                 batch_size=16, minimum_examples=8, buffer_size=128,
                 update_seconds=5., maximum_parameter_change=.25):
        self.head = copy.deepcopy(linear).cpu()
        for parameter in self.head.parameters():
            parameter.requires_grad_(True)
        self.reference = copy.deepcopy(self.head.state_dict())
        self.horizon, self.dt = int(horizon), float(dt)
        self.target_mean, self.target_scale = float(target_mean), float(target_scale)
        self.learning_rate = float(learning_rate)
        self.batch_size, self.minimum_examples = batch_size, minimum_examples
        self.buffer_size, self.update_seconds = buffer_size, float(update_seconds)
        self.maximum_parameter_change = float(maximum_parameter_change)
        self.version = 0
        self.update_log = []
        self.total_updates = 0
        self.event_resets = 0
        self.reset_buffers()

    def reset_buffers(self, clean_after=-np.inf, restore=True):
        if restore:
            self.head.load_state_dict(self.reference)
        self.optimizer = torch.optim.Adam(self.head.parameters(), lr=max(self.learning_rate, 1e-12))
        self.pending = deque()
        self.recent = deque(maxlen=self.buffer_size)
        self.clean_after = float(clean_after)
        self.last_observation = None
        self.last_update = -np.inf

    def event(self, detected_time, post_guard=0.):
        self.reset_buffers(detected_time+post_guard)
        self.version += 1
        self.event_resets += 1

    def observe(self, timestamp, current):
        if self.last_observation is not None and not np.isclose(timestamp-self.last_observation, self.dt, atol=1e-6):
            raise ValueError('Feedback expects a regular observed grid; reset at gaps.')
        self.last_observation = float(timestamp)
        for sample in self.pending:
            if timestamp > sample['origin_time']+1e-8:
                sample['targets'].append(float(current))
                sample['target_end_time'] = float(timestamp)
        while self.pending and len(self.pending[0]['targets']) >= self.horizon:
            sample = self.pending.popleft()
            if len(sample['targets']) != self.horizon:
                raise AssertionError('A future target was skipped or replayed.')
            if sample['eligible']:
                self.recent.append(sample)

    @torch.no_grad()
    def predict(self, representation, anchor):
        z = torch.as_tensor(representation, dtype=torch.float32).reshape(1, -1)
        a = float(anchor)
        return (a+self.head(z)[0].numpy())*self.target_scale+self.target_mean

    def issue(self, timestamp, representation, anchor, history_start, eligible=True):
        forecast = self.predict(representation, anchor).copy()
        self.pending.append({'origin_time': float(timestamp), 'representation': np.asarray(representation, dtype=np.float32).copy(),
                                 'anchor': float(anchor), 'targets': [],
                                 'eligible': bool(eligible and history_start > self.clean_after+1e-8),
                                 'issued_version': self.version, 'issued_forecast': forecast.copy()})
        return forecast, self.version

    def maybe_update(self, timestamp, blocked=False):
        if self.learning_rate <= 0 or blocked or len(self.recent) < self.minimum_examples or timestamp-self.last_update < self.update_seconds-1e-8:
            return False
        samples = list(self.recent)[-self.batch_size:]
        if any(s['target_end_time'] > timestamp+1e-8 for s in samples):
            raise AssertionError('Unobserved target in feedback.')
        z = torch.from_numpy(np.stack([s['representation'] for s in samples]))
        anchor = torch.tensor([s['anchor'] for s in samples]).unsqueeze(1)
        target = torch.tensor([s['targets'] for s in samples])
        target = (target-self.target_mean)/self.target_scale
        self.optimizer.zero_grad(set_to_none=True)
        loss = (anchor+self.head(z)-target).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite online loss.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.head.parameters(), .1)
        self.optimizer.step()
        with torch.no_grad():
            difference = [p-self.reference[name] for name, p in self.head.named_parameters()]
            norm = torch.sqrt(sum(d.square().sum() for d in difference))
            factor = min(1., self.maximum_parameter_change/max(float(norm), 1e-12))
            for (name, p), delta in zip(self.head.named_parameters(), difference):
                p.copy_(self.reference[name]+factor*delta)
        self.last_update = float(timestamp)
        self.version += 1
        self.total_updates += 1
        self.update_log.append({'update_time': float(timestamp), 'maximum_target_time': max(s['target_end_time'] for s in samples),
                                    'samples': len(samples), 'loss': float(loss.detach()), 'version': self.version})
        return True
