"""Causal as-of sampling, incremental features, and trial-isolated scaling."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_ROOT = Path(__file__).resolve().parents[1]
RAW_COLUMNS = [
    'current', 'rpm', 'target_rpm', 'throttle_pct', 'voltage', 'temp_test_sec',
    'lwc', 'dms_temp_cal_front', 'dms_temp_cal_back', 'dms_front', 'dms_back',
    'dms_avg',
]
ICE_TARGET_COLUMNS = ['Total_Ice_Pixels', 'Total_Ice_Area_mm2']
RELATION_COLUMNS = [
    'current_smooth', 'current_slope', 'rpm_slope', 'throttle_slope',
    'voltage_slope', 'rpm_error',
    'relative_rpm_error', 'electrical_power', 'current_per_krpm',
    'freezing_lwc_interaction', 'temperature_current_interaction',
    'dms_front_back_difference', 'current_mean_1s', 'current_std_1s',
    'current_mean_5s', 'current_std_5s',
]


def input_columns(cfg):
    # Missing schema identifies archived checkpoints, whose input contract must
    # remain reproducible. New training explicitly requires telemetry_v2.
    return RAW_COLUMNS + (ICE_TARGET_COLUMNS if cfg.get('input_schema', 'legacy_ice_v1') == 'legacy_ice_v1' else [])


@dataclass
class Segment:
    trial: int
    key: str
    time: np.ndarray
    raw: np.ndarray
    features: np.ndarray | None = None
    ice_targets: np.ndarray | None = None


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if cfg.get('input_schema', 'legacy_ice_v1') not in {'telemetry_v2', 'legacy_ice_v1'}:
        raise ValueError('Unknown input_schema.')
    for key in ['sample_interval_seconds', 'max_observation_age_seconds',
                'forecast_seconds', 'forecast_stride_seconds',
                'smooth_time_constant_seconds', 'hold_feature_seconds']:
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be positive and finite.')
    dt = cfg['sample_interval_seconds']
    for key in ['forecast_seconds', 'forecast_stride_seconds', 'warmup_seconds']:
        if cfg[key] < 0 or not math.isclose(cfg[key]/dt, round(cfg[key]/dt), abs_tol=1e-6):
            raise ValueError(f'{key} must be a nonnegative multiple of the sample interval.')
    if set(cfg['validation_trials']) & set(cfg['test_trials']):
        raise ValueError('Validation and test trial sets overlap.')
    for key in ['hidden_size', 'epochs', 'batch_size', 'chunk_steps', 'patience', 'cpu_threads']:
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer.')
    for key in ['learning_rate', 'spike_threshold', 'surrogate_slope', 'trace_time_constant_seconds', 'gradient_clip_norm']:
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be positive and finite.')
    for key in ['weight_decay', 'activity_regularization', 'ridge_regularization']:
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f'{key} must be nonnegative and finite.')
    if len(cfg['neuron_time_constants_seconds']) != 2 or any(t <= 0 or not math.isfinite(t) for t in cfg['neuron_time_constants_seconds']):
        raise ValueError('Exactly two positive finite neuron time constants are required.')


def source_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_segments(source: Path, cfg: dict) -> list[Segment]:
    """At grid t, select the last source row at or before t, without interpolation.

    Targets are original current readings held to the grid, never averaged or
    smoothed. Bound observation age and preserve polish1 current-segment IDs.
    """
    groups = defaultdict(list)
    with source.open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = ['data_id', 'timestamp', 'current_segment_id', *RAW_COLUMNS]
        missing = set(required)-set(reader.fieldnames)
        if missing:
            raise ValueError(f'Source columns missing: {sorted(missing)}')
        for row in reader:
            values = []
            for name in RAW_COLUMNS+ICE_TARGET_COLUMNS:
                try:
                    value = float(row.get(name, ''))
                except (ValueError, TypeError):
                    value = math.nan
                values.append(value if math.isfinite(value) else math.nan)
            segment = row['current_segment_id']
            segment_number = int(segment.split(':')[-1]) if segment else -1
            groups[int(row['data_id'])].append([float(row['timestamp']), segment_number, *values])
    dt = cfg['sample_interval_seconds']
    segments = []
    for trial, records in sorted(groups.items()):
        arr = np.asarray(records, dtype=np.float64)
        t = arr[:, 0]
        if np.any(np.diff(t) <= 0):
            raise ValueError(f'Trial {trial}: timestamps must strictly increase.')
        grid = np.arange(math.ceil(t[0]/dt-1e-9), math.floor(t[-1]/dt+1e-9)+1)*dt
        indices = np.searchsorted(t, grid+1e-9, side='right')-1
        source_values = arr[indices, 2:]
        values = source_values[:, :len(input_columns(cfg))]
        ice_targets = source_values[:, len(RAW_COLUMNS):]
        ids = arr[indices, 1]
        valid = (indices >= 0) & np.isfinite(values[:, 0]) & (ids >= 0)
        valid &= grid-t[indices] <= cfg['max_observation_age_seconds']+1e-9
        start = None
        serial = 0
        for i in range(len(grid)+1):
            boundary = i == len(grid) or not valid[i]
            if i < len(grid) and i > 0 and valid[i] and valid[i-1] and ids[i] != ids[i-1]:
                boundary = True
            if boundary and start is not None:
                serial += 1
                segments.append(Segment(trial, f'{trial}:{serial}', grid[start:i].copy(),
                                        values[start:i].copy(), ice_targets=ice_targets[start:i].copy()))
                start = None
            if i < len(grid) and valid[i] and start is None:
                start = i
    if cfg.get('clean_intervals_path'):
        from .events import clean_ranges
        intervals = json.loads(Path(cfg['clean_intervals_path']).read_text())
        clean = []
        serials = defaultdict(int)
        for segment in segments:
            for a, b in clean_ranges(segment, intervals):
                serials[segment.trial] += 1
                clean.append(Segment(segment.trial, f'{segment.trial}.{serials[segment.trial]}',
                                     segment.time[a:b].copy(), segment.raw[a:b].copy(),
                                     ice_targets=segment.ice_targets[a:b].copy()))
        return clean
    return segments


class FeatureBuilder:
    """One causal feature state per segment, shared by training and live inference."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.columns = input_columns(cfg)
        self.previous = None
        self.previous_time = None
        self.held = np.full(len(self.columns), np.nan)
        self.held_time = np.full(len(self.columns), -np.inf)
        self.smooth = None
        self.history = deque()
        relations = RELATION_COLUMNS.copy()
        if len(self.columns) > len(RAW_COLUMNS):
            relations[5:5] = ['ice_area_slope', 'ice_pixels_slope']
        self.names = self.columns+relations+[f'{name}_missing' for name in self.columns]

    def step(self, timestamp, measurements):
        x = np.asarray(measurements, dtype=np.float64).copy()
        if x.shape != (len(self.columns),):
            raise ValueError(f'Expected {len(self.columns)} input measurements in {self.columns} order.')
        if not np.isfinite(x[0]):
            raise ValueError('Missing current requires a segment reset, not imputation.')
        dt = self.cfg['sample_interval_seconds'] if self.previous_time is None else timestamp-self.previous_time
        if dt <= 0:
            raise ValueError('Feature timestamps must strictly increase.')
        mask = ~np.isfinite(x)
        observed = ~mask
        self.held[observed] = x[observed]
        self.held_time[observed] = timestamp
        can_hold = mask & (timestamp-self.held_time <= self.cfg['hold_feature_seconds'])
        x[can_hold] = self.held[can_hold]
        self.smooth = x[0] if self.smooth is None else self.smooth-math.expm1(-dt/self.cfg['smooth_time_constant_seconds'])*(x[0]-self.smooth)
        slopes = np.zeros(len(x)) if self.previous is None else (x-self.previous)/dt
        self.history.append((timestamp, x[0]))
        while self.history and timestamp-self.history[0][0] >= 5.-1e-8:
            self.history.popleft()
        recent_1 = [value for t, value in self.history if timestamp-t < 1.-1e-8]
        recent_5 = [value for _, value in self.history]
        rpm_error = x[2]-x[1]
        relative_error = rpm_error/x[2] if np.isfinite(x[2]) and x[2] > 100 else np.nan
        load = x[0]/(x[1]/1000) if np.isfinite(x[1]) and x[1] > 100 else np.nan
        relations = [self.smooth, slopes[0], slopes[1], slopes[3], slopes[4],
                     rpm_error, relative_error, x[0]*x[4], load,
                     max(0., -x[5])*x[6] if np.isfinite(x[5]) else np.nan,
                     x[5]*x[0], x[9]-x[10], np.mean(recent_1), np.std(recent_1),
                     np.mean(recent_5), np.std(recent_5)]
        if len(self.columns) > len(RAW_COLUMNS):
            relations[5:5] = [slopes[13], slopes[12]]
        self.previous = x.copy()
        self.previous_time = timestamp
        return np.asarray([*x, *relations, *mask.astype(float)], dtype=np.float32)


def feature_indices(names, feature_set):
    if feature_set == 'all':
        return np.arange(len(names))
    if feature_set == 'raw':
        return np.asarray([i for i, name in enumerate(names) if name in RAW_COLUMNS+ICE_TARGET_COLUMNS or name.endswith('_missing')])
    if feature_set == 'no_camera':
        return np.asarray([i for i, name in enumerate(names) if 'ice' not in name.lower()])
    raise ValueError('feature_set must be all, raw, or no_camera.')


def prepare(cfg, output=None):
    output = Path(output or MODEL_ROOT/'data'/'prepared.npz')
    output.parent.mkdir(parents=True, exist_ok=True)
    source = (MODEL_ROOT/cfg['source_csv']).resolve()
    segments = read_segments(source, cfg)
    builder = FeatureBuilder(cfg)
    selection = feature_indices(builder.names, cfg['feature_set'])
    names = [builder.names[i] for i in selection]
    warmup = round(cfg['warmup_seconds']/cfg['sample_interval_seconds'])
    horizon = round(cfg['forecast_seconds']/cfg['sample_interval_seconds'])
    usable = []
    for s in segments:
        if len(s.time) < max(warmup, 1)+horizon:
            continue
        state = FeatureBuilder(cfg)
        s.features = np.stack([state.step(t, r)[selection] for t, r in zip(s.time, s.raw)])
        usable.append(s)
    validation, test = set(cfg['validation_trials']), set(cfg['test_trials'])
    all_trials = {s.trial for s in usable}
    if not validation <= all_trials or not test <= all_trials:
        raise ValueError('Configured validation/test trials have no usable segment.')
    train = [s for s in usable if s.trial not in validation | test]
    if not train:
        raise ValueError('No training segments.')
    train_features = np.concatenate([s.features for s in train])
    finite_count = np.isfinite(train_features).sum(axis=0)
    imputer = np.asarray([np.median(train_features[np.isfinite(train_features[:, i]), i])
                          if finite_count[i] else 0. for i in range(len(names))], dtype=np.float32)
    imputed = np.where(np.isfinite(train_features), train_features, imputer)
    center = imputed.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = imputed.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.
    train_current = np.concatenate([s.raw[:, 0] for s in train])
    target_mean, target_scale = float(train_current.mean()), float(train_current.std())
    if target_scale < 1e-6:
        target_scale = 1.
    lengths = np.asarray([len(s.time) for s in usable])
    offsets = np.r_[0, np.cumsum(lengths)]
    payload = {
        'x': np.concatenate([np.where(np.isfinite(s.features), s.features, imputer) for s in usable]),
        'y': np.concatenate([s.raw[:, 0] for s in usable]).astype(np.float32),
        'time': np.concatenate([s.time for s in usable]), 'offsets': offsets,
        'trials': np.asarray([s.trial for s in usable]), 'keys': np.asarray([s.key for s in usable]),
        'feature_names': np.asarray(names), 'imputer': imputer, 'center': center, 'scale': scale,
        'target_mean': np.asarray(target_mean), 'target_scale': np.asarray(target_scale),
        'ice_targets': np.concatenate([s.ice_targets for s in usable]).astype(np.float32),
        'ice_target_valid': np.concatenate([np.isfinite(s.ice_targets) for s in usable]),
        'ice_target_names': np.asarray(ICE_TARGET_COLUMNS),
    }
    np.savez_compressed(output, **payload)
    manifest = {
        'config': cfg, 'source': str(source), 'source_sha256': source_hash(source),
        'sampling': 'Last original row at/before each absolute grid tick; no averaging, interpolation or smoothing of targets.',
        'target': 'Original current sampled as-of at grid timestamps, in amperes (A); units confirmed by the dataset owner.',
        'ice_targets': 'Separate original as-of observations with validity masks; never imputed, held across missing readings, or used as telemetry_v2 inputs. Available for future supervised heads; the current decoder predicts current only.',
        'ice_target_names': ICE_TARGET_COLUMNS,
        'input_schema': cfg.get('input_schema', 'legacy_ice_v1'),
        'max_target_observation_age_seconds': cfg['max_observation_age_seconds'],
        'feature_names': names, 'training_trials': sorted(all_trials-validation-test),
        'validation_trials': sorted(validation), 'test_trials': sorted(test),
        'segments': len(usable), 'discarded_short_segments': len(segments)-len(usable),
        'samples': int(offsets[-1]), 'feature_count': len(names),
        'scaler_fit_scope': 'Training trials only; validation/test observations never used.',
        'all_missing_training_features': [names[i] for i in np.flatnonzero(finite_count == 0)],
        'warmup_steps': warmup, 'forecast_steps': horizon,
    }
    output.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in ['segments', 'samples', 'feature_count', 'warmup_steps', 'forecast_steps']}, indent=2), flush=True)
    return output


class PreparedData:
    def __init__(self, path, cfg):
        with np.load(path, allow_pickle=False) as loaded:
            for key in loaded.files:
                setattr(self, key, loaded[key].copy())
        self.cfg = cfg
        self.x = ((self.x-self.center)/self.scale).astype(np.float32)
        self.target_mean = float(self.target_mean)
        self.target_scale = float(self.target_scale)
        self.horizon = round(cfg['forecast_seconds']/cfg['sample_interval_seconds'])
        self.warmup = round(cfg['warmup_seconds']/cfg['sample_interval_seconds'])
        self.stride = round(cfg['forecast_stride_seconds']/cfg['sample_interval_seconds'])
        self.current_index = list(self.feature_names).index('current')

    def indices(self, split):
        validation, test = set(self.cfg['validation_trials']), set(self.cfg['test_trials'])
        return [i for i, t in enumerate(self.trials) if
                (split == 'train' and t not in validation | test) or
                (split == 'validation' and t in validation) or
                (split == 'test' and t in test)]

    def segment(self, i):
        a, b = self.offsets[i:i+2]
        return self.x[a:b], self.y[a:b], self.time[a:b]

    def origins(self, n):
        return np.arange(max(self.warmup-1, 0), n-self.horizon, self.stride)

    def standardized_target(self, current):
        return (current-self.target_mean)/self.target_scale
