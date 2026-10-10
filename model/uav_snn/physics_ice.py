"""Causal physical load features and compact neural ice-area observers.

The observer runs at 1 Hz and never receives camera measurements. Its hidden
state and slow telemetry histories persist across electrically detected shedding.
"""
from __future__ import annotations

import copy
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .ice_area import IceAreaEstimator, slope, trial_weights

FEATURE_NAMES = [
    'temperature_c', 'lwc', 'current_a', 'current_slope_5s', 'rpm',
    'current_slope_30s', 'current_mean_30s', 'current_std_5s',
    'voltage_v', 'normalized_power_w', 'normalized_power_slope_5s',
    'normalized_power_slope_30s', 'normalized_power_mean_30s',
    'normalized_power_std_5s', 'rpm_tracking_error', 'rpm_slope_5s',
    'clean_power_residual_w', 'normalized_clean_power_residual_w',
    'load_ewm_120s', 'load_ewm_300s', 'exposure_120s', 'exposure_300s',
    'front_strain_per_speed_squared', 'back_strain_per_speed_squared',
    'strain_difference', 'front_strain_missing', 'back_strain_missing',
]
FEATURE_GROUPS = {
    'core': [0, 1, 2, 3],
    'history': list(range(8)),
    'normalized_instant': [0, 1, 4, 8, 9],
    'normalized_history': list(range(16)),
    'clean_residual': list(range(18)),
    'persistent': list(range(22)),
    'dms': list(range(27)),
}


class CleanPowerReference:
    """Training-only clean calibration; coefficients are not motor constants."""

    @staticmethod
    def design(raw):
        raw = np.atleast_2d(raw)
        n, voltage, temperature = raw[:, 1]/3000., raw[:, 4], raw[:, 5]
        return np.column_stack([n, n*n, n*n*n, voltage, temperature])

    def fit(self, raw, parents):
        x = self.design(raw)
        w = trial_weights(parents)
        self.mean = w @ x
        self.scale = np.maximum(np.sqrt(w @ ((x-self.mean)**2)), 1e-6)
        z = np.column_stack([np.ones(len(x)), (x-self.mean)/self.scale])
        penalty = np.eye(z.shape[1])*.01
        penalty[0, 0] = 0
        self.coef = np.linalg.solve(z.T @ (w[:, None]*z)+penalty,
                                    z.T @ (w*raw[:, 0]*raw[:, 4]))
        self.lower, self.upper = x.min(0), x.max(0)
        return self

    def predict(self, raw):
        x = self.design(raw)
        z = np.column_stack([np.ones(len(x)), (x-self.mean)/self.scale])
        return np.maximum(0, z @ self.coef)

    def payload(self):
        return {k: getattr(self, k).tolist() for k in ['mean', 'scale', 'coef', 'lower', 'upper']}

    @classmethod
    def from_payload(cls, data):
        obj = cls()
        for k, value in data.items():
            setattr(obj, k, np.asarray(value))
        return obj


class PhysicalFeatureBuilder:
    """One-second telemetry in RAW_COLUMNS order; no camera or trial ID input."""

    def __init__(self, reference):
        self.reference = reference
        self.history = deque(maxlen=30)
        self.last_time = None
        self.ewm = None
        self.exposure = np.zeros(2)
        self.last_raw = None

    def step(self, timestamp, raw):
        raw = np.asarray(raw, float)
        if raw.shape != (12,):
            raise ValueError('Expected exactly 12 telemetry channels, without ice.')
        if not np.isfinite(raw[[0, 1, 4, 5, 6]]).all() or raw[1] < 100:
            raise ValueError('Finite current, spinning RPM, voltage, temperature and LWC required.')
        if self.last_time is not None and not np.isclose(timestamp-self.last_time, 1., atol=1e-6):
            raise ValueError('Ice observer expects 1 Hz telemetry; reset at gaps.')
        power = raw[0]*raw[4]
        normalized = power*(3000./raw[1])**3
        residual = power-float(self.reference.predict(raw)[0])
        self.history.append((raw.copy(), normalized))
        if self.ewm is None:
            self.ewm = np.full(2, normalized)
        decay = np.exp(-1./np.array([120., 300.]))
        self.ewm = decay*self.ewm+(1-decay)*normalized
        # Finite-memory exposure proxy in arbitrary units, not accumulated mass.
        self.exposure = decay*self.exposure + raw[6]*(raw[1]/3000.)
        self.last_time, self.last_raw = float(timestamp), raw.copy()
        if len(self.history) < 30:
            return None
        readings = np.stack([r for r, _ in self.history])
        current, rpm = readings[:, 0], readings[:, 1]
        load = np.asarray([p for _, p in self.history])
        strain = raw[[9, 10]]
        missing = ~np.isfinite(strain)
        strain = np.where(missing, 0., strain)
        return np.array([
            raw[5], raw[6], raw[0], slope(current[-5:]), raw[1],
            slope(current), current.mean(), current[-5:].std(), raw[4],
            normalized, slope(load[-5:]), slope(load), load.mean(), load[-5:].std(),
            raw[2]-raw[1] if np.isfinite(raw[2]) else 0., slope(rpm[-5:]),
            residual, residual*(3000./raw[1])**3,
            *self.ewm, *self.exposure, *(strain*(3000./raw[1])**2),
            strain[0]-strain[1], *missing.astype(float),
        ])

    def rollout(self, current_forecast):
        """Advance a copy at +1..+H seconds; all other channels held at origin."""
        if len(self.history) < 30:
            raise ValueError('Thirty observed one-second samples are required.')
        clone = copy.deepcopy(self)
        features = []
        for current in np.asarray(current_forecast, float):
            raw = self.last_raw.copy()
            raw[0] = current
            features.append(clone.step(clone.last_time+1., raw))
        return np.stack(features)


class PhysicalBasisEstimator(IceAreaEstimator):
    def __init__(self, spec):
        self.spec = dict(spec)
        self.indices = np.asarray(FEATURE_GROUPS[spec['features']])


class IceGRU(nn.Module):
    """Persistent neural observer with a telemetry skip connection and mm² output."""

    def __init__(self, inputs, hidden=16):
        super().__init__()
        self.gru = nn.GRU(inputs, hidden, batch_first=True)
        self.readout = nn.Linear(hidden+inputs, 1)

    def forward(self, x, state=None):
        h, state = self.gru(x, state)
        # Targets/loss are in thousands of mm²; softplus keeps area nonnegative.
        area = torch.nn.functional.softplus(self.readout(torch.cat([h, x], -1))).squeeze(-1)
        return area, state


class IceObserver:
    """Serialized regression or neural ensemble, with an identical live feature API."""

    def __init__(self, path):
        path = Path(path)
        manifest = json.loads((path/'observer.json').read_text())
        self.manifest = manifest
        self.reference = CleanPowerReference.from_payload(manifest['clean_reference'])
        self.kind = manifest['kind']
        if self.kind == 'basis':
            self.basis = PhysicalBasisEstimator.load(path/'observer.npz')
        elif self.kind == 'gru':
            self.models, self.transforms = [], []
            for name in manifest['checkpoints']:
                cp = torch.load(path/name, map_location='cpu', weights_only=True)
                model = IceGRU(len(cp['indices']), cp['hidden_size'])
                model.load_state_dict(cp['state_dict'])
                model.eval()
                self.models.append(model)
                self.transforms.append((np.asarray(cp['indices']), np.asarray(cp['mean']),
                                        np.asarray(cp['scale'])))
        else:
            raise ValueError('Unknown ice-observer model family.')
        self.reset()

    def reset(self):
        self.builder = PhysicalFeatureBuilder(self.reference)
        self.states = [None]*len(getattr(self, 'models', []))
        self.area_now = None

    @torch.no_grad()
    def predict_features(self, features, states=None):
        features = np.atleast_2d(features)
        if self.kind == 'basis':
            return self.basis.predict(features), []
        states = [None]*len(self.models) if states is None else states
        predictions, next_states = [], []
        for model, (indices, mean, scale), state in zip(self.models, self.transforms, states):
            x = torch.as_tensor(((features[:, indices]-mean)/scale).astype(np.float32))[None]
            area, new_state = model(x, state)
            predictions.append(area[0].numpy()*1000.)
            next_states.append(new_state)
        return np.mean(predictions, axis=0), next_states

    def update(self, timestamp, raw):
        features = self.builder.step(timestamp, raw)
        if features is None:
            return None
        prediction, self.states = self.predict_features(features, self.states)
        self.area_now = float(prediction[0])
        return self.area_now

    def forecast(self, current_forecast):
        features = self.builder.rollout(current_forecast)
        # Forward calls return new states; never overwrite the observed state.
        return self.predict_features(features, self.states)[0]
