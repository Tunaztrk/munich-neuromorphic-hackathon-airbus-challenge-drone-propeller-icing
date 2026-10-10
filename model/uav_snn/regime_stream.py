"""Live causal window/stateful SNN with delayed final-head supervision."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .data import FeatureBuilder
from .events import DropDetector
from .feedback import DelayedHead
from .network import CurrentSNN


class RegimeForecaster:
    def __init__(self, checkpoint_path, feedback_learning_rate=None):
        cp = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if cp.get('model_family') != 'regime_v1':
            raise ValueError('Expected a regime_v1 checkpoint.')
        self.cfg = cp['config']
        self.names = cp['feature_names']
        self.guard = cp['event_guard']
        self.imputer = np.asarray(cp['imputer'], dtype=np.float32)
        self.center = np.asarray(cp['center'], dtype=np.float32)
        self.scale = np.asarray(cp['scale'], dtype=np.float32)
        self.target_mean, self.target_scale = cp['target_mean'], cp['target_scale']
        self.dt = self.cfg['sample_interval_seconds']
        self.horizon = round(self.cfg['forecast_seconds']/self.dt)
        j = self.names.index('current')
        self.model = CurrentSNN(len(self.names), self.horizon, self.cfg, j,
                                self.scale[j]/self.target_scale,
                                (self.center[j]-self.target_mean)/self.target_scale)
        self.model.load_state_dict(cp['state_dict'])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.lr = self.cfg.get('feedback_learning_rate', 0.) if feedback_learning_rate is None else feedback_learning_rate
        self.width = round((self.cfg.get('window_seconds') or 1.)/self.dt)
        self.reset()

    def reset(self):
        """Start a new independent parent trial, restoring the offline adapter."""
        self.builder, self.detector = FeatureBuilder(self.cfg), DropDetector()
        self.context = deque(maxlen=self.width)
        self.state = None
        self.adapter = DelayedHead(self.model.decoder[-1], self.horizon, self.dt,
                                   self.target_mean, self.target_scale, learning_rate=self.lr)
        self.last_time = None
        self.clean_after = -np.inf
        self.previous_transition = False
        self.count = 0
        self.last_event = None
        self.last_prediction_version = 0
        self.prediction_ledger = deque(maxlen=512)

    def clear_context(self):
        self.builder = FeatureBuilder(self.cfg)
        self.context.clear()
        self.state = None

    def update(self, timestamp, measurements):
        """One 5 Hz as-of observation in RAW_COLUMNS order, current in A.

        Forecasts continue during confirmations and short histories. A short
        context or transition returns explicit persistence, not a missing output.
        Forecast Feedback runs after producing the current forecast and only with matured labels.
        """
        row = np.asarray(measurements, dtype=float)
        if row.shape != (12,) or not np.isfinite(row[0]):
            raise ValueError('Supply 12 telemetry channels and finite current; reset at missing-current gaps.')
        if self.last_time is not None and not np.isclose(timestamp-self.last_time, self.dt, atol=1e-6):
            raise ValueError('Expected 0.2 s as-of grid; reset at gaps or parent boundaries.')
        event = self.detector.update(float(timestamp), row)
        if event and event['status'] == 'confirmed_drop':
            self.last_event = event.copy()
            self.clean_after = timestamp+self.guard['post_guard_seconds']
            self.clear_context()
            self.adapter.event(float(timestamp), self.guard['post_guard_seconds'])
        transition = timestamp <= self.clean_after+1e-8
        if self.previous_transition and not transition:
            self.clear_context()
        self.adapter.observe(float(timestamp), float(row[0]))
        features = self.builder.step(float(timestamp), row)
        x = (np.where(np.isfinite(features), features, self.imputer)-self.center)/self.scale
        self.context.append((float(timestamp), x.astype(np.float32)))
        with torch.no_grad():
            if self.cfg['inference_mode'] == 'window':
                window = torch.from_numpy(np.stack([x for _, x in self.context])).unsqueeze(0)
                z, anchor, _ = self.model.window_encode(window)
            else:
                z, anchor, self.state, _ = self.model.encode_step(torch.from_numpy(x.astype(np.float32)).unsqueeze(0), self.state)
        fallback = transition or len(self.context) < 5
        representation, a = z[0].numpy(), float(anchor[0, 0])
        prediction = np.full(self.horizon, row[0], dtype=np.float32) if fallback else self.adapter.predict(representation, a)
        version = self.adapter.version
        # Match the established 1-second issue cadence. update() still returns a
        # trajectory at every sensor tick; only cadenced forecasts train the head.
        stride = round(self.cfg['forecast_stride_seconds']/self.dt)
        warmup = round(self.cfg.get('feedback_issue_warmup_seconds', 20.)/self.dt)
        if self.count >= warmup-1 and (self.count-(warmup-1)) % stride == 0:
            self.adapter.issue(float(timestamp), representation, a, self.context[0][0], eligible=not fallback)
        self.prediction_ledger.append({'origin_time': float(timestamp), 'model_version': version,
                                            'forecast': prediction.copy(), 'fallback': bool(fallback)})
        self.adapter.maybe_update(float(timestamp), blocked=transition or self.detector.pending is not None)
        self.last_time = float(timestamp)
        self.previous_transition = transition
        self.count += 1
        self.last_prediction_version = version
        return prediction.copy()
