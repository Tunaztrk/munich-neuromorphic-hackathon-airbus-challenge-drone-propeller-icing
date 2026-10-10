"""Telemetry-only ice-area estimators, independent of the current SNN.

All histories are trailing, one-second as-of observations. Camera measurements
are targets only. The serialized estimator is a NumPy archive, not pickle.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FEATURES = [
    'temperature_c', 'lwc', 'current_a', 'current_slope_5s_a_per_s',
    'rpm', 'current_slope_30s_a_per_s', 'current_mean_30s_a',
    'current_std_5s_a',
]
FEATURE_SETS = {'current': [2], 'settings': [0, 1], 'core': [0, 1, 2, 3],
                'operating': [0, 1, 2, 3, 4], 'history': list(range(8)),
                'current_rpm': [2, 4]}
FEATURE_SETS.update({
    'history_no_rpm': [0, 1, 2, 3, 5, 6, 7],
    'history_no_slopes': [0, 1, 2, 4, 6, 7],
    'history_no_temperature': [1, 2, 3, 4, 5, 6, 7],
    'history_no_lwc': [0, 2, 3, 4, 5, 6, 7],
})
# Columns in native records. Ice columns are deliberately absent from features.
COLUMNS = ['timestamp', 'temp_test_sec', 'lwc', 'current', 'rpm',
           'target_rpm', 'Total_Ice_Area_mm2', 'Total_Ice_Pixels']


def asof(records, query):
    """Return last observations, with a 0.5-second maximum age."""
    query = np.asarray(query, dtype=float)
    index = np.searchsorted(records[:, 0], query + 1e-8, side='right') - 1
    result = records[np.maximum(index, 0)].copy()
    valid = (index >= 0) & (query - result[..., 0] <= 0.5 + 1e-8)
    result[~valid] = np.nan
    return result


def slope(values):
    axis = np.arange(values.shape[-1], dtype=float)
    axis -= axis.mean()
    return (values @ axis) / (axis @ axis)


def from_history(history, temperature, lwc, rpm):
    """30 current observations at one-second spacing, ending at prediction time."""
    history = np.asarray(history, dtype=float)
    if history.shape[-1] != 30:
        raise ValueError('Exactly 30 one-second current observations are required.')
    return np.column_stack([
        temperature, lwc, history[:, -1], slope(history[:, -5:]), rpm,
        slope(history), history.mean(axis=1), history[:, -5:].std(axis=1),
    ])


def telemetry_at(records, query):
    query = np.asarray(query, dtype=float)
    past = asof(records, query[:, None] - np.arange(29, -1, -1)[None, :])
    last = past[:, -1]
    history = past[:, :, 3]
    features = from_history(history, last[:, 1], last[:, 2], last[:, 4])
    spinning = np.all(past[:, :, 4] >= 100, axis=1)
    commanded = np.all(np.isnan(past[:, :, 5]) | (past[:, :, 5] > 0), axis=1)
    valid = np.isfinite(features).all(axis=1) & spinning & commanded
    return features, valid, history


def trial_weights(trials):
    _, index, counts = np.unique(trials, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[index]
    return weights / weights.sum()


def design(z, kind, centers, width):
    if kind == 'constant':
        return np.ones((len(z), 1))
    terms = [np.ones((len(z), 1)), z]
    if kind == 'quadratic':
        terms.append(np.column_stack([z[:, i] * z[:, j]
                                     for i in range(z.shape[1])
                                     for j in range(i, z.shape[1])]))
    elif kind == 'rbf':
        distance = np.maximum((z*z).sum(1)[:, None] + (centers*centers).sum(1)[None, :]
                              - 2*z @ centers.T, 0)
        terms.append(np.exp(-distance / (2 * width**2)))
    elif kind != 'linear':
        raise ValueError(kind)
    return np.column_stack(terms)


class IceAreaEstimator:
    """A nonnegative, smooth-basis regression; area is measured in mm²."""

    def __init__(self, spec):
        self.spec = dict(spec)
        self.indices = np.array(FEATURE_SETS[spec['features']], dtype=int)

    def fit(self, features, target, trials):
        x = np.asarray(features[:, self.indices], dtype=float)
        if not np.isfinite(x).all() or not np.isfinite(target).all():
            raise ValueError('Training inputs and targets must be finite.')
        weights = trial_weights(trials)
        self.mean = weights @ x
        self.scale = np.sqrt(weights @ ((x-self.mean)**2))
        self.scale = np.maximum(self.scale, 1e-6)
        z = (x-self.mean)/self.scale
        self.lower = x.min(0)
        self.upper = x.max(0)
        self.centers = np.empty((0, x.shape[1]))
        if self.spec['kind'] == 'rbf':
            # Equal representation of training parents, with no target access.
            rng = np.random.default_rng(2026)
            parents = np.unique(trials)
            chosen = [rng.choice(np.flatnonzero(trials == parents[i % len(parents)]))
                      for i in range(128)]
            self.centers = z[chosen]
        phi = design(z, self.spec['kind'], self.centers, self.spec.get('width', 1.0))
        penalty = np.eye(phi.shape[1]) * self.spec.get('alpha', 0.001)
        penalty[0, 0] = 0
        self.coefficients = np.linalg.solve(phi.T @ (weights[:, None]*phi) + penalty,
                                            phi.T @ (weights*target/1000.0))
        return self

    def predict(self, features):
        x = np.asarray(features[:, self.indices], dtype=float)
        if not np.isfinite(x).all():
            raise ValueError('Prediction inputs must be finite; do not fill with ice measurements.')
        predictions = []
        for start in range(0, len(x), 4096):
            z = (x[start:start+4096]-self.mean)/self.scale
            phi = design(z, self.spec['kind'], self.centers, self.spec.get('width', 1.0))
            predictions.append(np.maximum(0, 1000.0 * (phi @ self.coefficients)))
        return np.concatenate(predictions) if predictions else np.empty(0)

    def outside_training_range(self, features):
        x = features[:, self.indices]
        return np.any((x < self.lower) | (x > self.upper), axis=1)

    def predict_current_rpm(self, current, rpm):
        """Two-input convenience API; preserve the broadcast shape of the inputs."""
        if self.spec['features'] != 'current_rpm':
            raise ValueError('This API requires a fitted current_rpm estimator.')
        current, rpm = np.broadcast_arrays(np.asarray(current, float), np.asarray(rpm, float))
        features = np.zeros((current.size, len(FEATURES)))
        features[:, 2], features[:, 4] = current.ravel(), rpm.ravel()
        return self.predict(features).reshape(current.shape)

    def save(self, path, metadata):
        np.savez_compressed(path, spec=json.dumps(self.spec), metadata=json.dumps(metadata),
                            mean=self.mean, scale=self.scale, centers=self.centers,
                            coefficients=self.coefficients, lower=self.lower, upper=self.upper)

    @classmethod
    def load(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as data:
            estimator = cls(json.loads(str(data['spec'])))
            for name in ['mean', 'scale', 'centers', 'coefficients', 'lower', 'upper']:
                setattr(estimator, name, data[name].copy())
        return estimator
