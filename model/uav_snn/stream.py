"""Incremental forecasting from causally sampled sensor rows, without future data."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import FeatureBuilder, PreparedData, feature_indices
from .network import CurrentSNN, build_model


class StreamingForecaster:
    """One step per resampled observation; caller resets at trial/segment boundaries.

    New input measurements follow the 12-channel RAW_COLUMNS order (no ice).
    Archived checkpoints retain their original 14-channel contract. All values
    must be available at the
    supplied timestamp. Original irregular observations need causal as-of
    sampling first; update() intentionally does not interpolate raw telemetry.
    """
    def __init__(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint.get('model_family') == 'regime_v1':
            from .regime_stream import RegimeForecaster
            self.regime = RegimeForecaster(checkpoint_path)
            self.cfg = self.regime.cfg
            self.model = self.regime.model
            return
        self.cfg = checkpoint['config']
        self.names = checkpoint['feature_names']
        self.imputer = np.asarray(checkpoint['imputer'], dtype=np.float32)
        self.center = np.asarray(checkpoint['center'], dtype=np.float32)
        self.scale = np.asarray(checkpoint['scale'], dtype=np.float32)
        self.target_mean = checkpoint['target_mean']
        self.target_scale = checkpoint['target_scale']
        current_index = self.names.index('current')
        horizon = round(self.cfg['forecast_seconds']/self.cfg['sample_interval_seconds'])
        self.model = CurrentSNN(len(self.names), horizon, self.cfg, current_index,
                                self.scale[current_index]/self.target_scale,
                                (self.center[current_index]-self.target_mean)/self.target_scale)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.reset()

    def reset(self):
        if hasattr(self, 'regime'):
            return self.regime.reset()
        self.builder = FeatureBuilder(self.cfg)
        self.selection = feature_indices(self.builder.names, self.cfg['feature_set'])
        self.state = None
        self.count = 0

    @torch.no_grad()
    def update(self, timestamp, measurements):
        if hasattr(self, 'regime'):
            # Online adaptation needs gradients for its separate linear head.
            with torch.enable_grad():
                return self.regime.update(timestamp, measurements)
        features = self.builder.step(timestamp, measurements)[self.selection]
        features = np.where(np.isfinite(features), features, self.imputer)
        standardized = (features-self.center)/self.scale
        prediction, self.state, _ = self.model.step(torch.from_numpy(standardized).unsqueeze(0), self.state)
        self.count += 1
        if self.count < round(self.cfg['warmup_seconds']/self.cfg['sample_interval_seconds']):
            return None
        return prediction[0].numpy()*self.target_scale+self.target_mean


def replay(run_dir, trial=28):
    """Replay a held-out trial with the same incremental network state as evaluation."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir/'config.json').read_text())
    torch.set_num_threads(cfg['cpu_threads'])
    data = PreparedData(run_dir/'prepared.npz', cfg)
    checkpoint = torch.load(run_dir/'best.pt', map_location='cpu', weights_only=True)
    model = build_model(data, cfg)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    latencies, values, origins, actual = [], [], [], []
    with torch.no_grad():
        for i in data.indices('test'):
            if int(data.trials[i]) != trial:
                continue
            x, current, timestamp = data.segment(i)
            state = None
            accepted_origins = set(data.origins(len(x)).tolist())
            for j, sample in enumerate(x):
                start = time.perf_counter()
                forecast, state, _ = model.step(torch.from_numpy(sample).unsqueeze(0), state)
                elapsed = time.perf_counter()-start
                if j >= data.warmup:
                    latencies.append(elapsed)
                if j in accepted_origins:
                    origins.append(timestamp[j])
                    values.append(forecast[0].numpy()*data.target_scale+data.target_mean)
                    actual.append(current[j+1:j+1+data.horizon])
    if not values:
        raise ValueError(f'Trial {trial} has no held-out replay data.')
    output = run_dir/f'replay_trial_{trial}.npz'
    np.savez_compressed(output, origin_time=np.asarray(origins), forecast=np.asarray(values), actual=np.asarray(actual))
    metrics = {'trial': trial, 'forecast_count': len(values), 'incremental_inference': True,
               'latency_scope': 'CPU network step and readout; excludes sensor IO and preprocessing',
               'median_step_ms': float(np.median(latencies)*1000),
               'p95_step_ms': float(np.quantile(latencies, .95)*1000),
               'sample_interval_ms': cfg['sample_interval_seconds']*1000}
    # Match forecasts by timestamp against independently batched test evaluation.
    with np.load(run_dir/'test_predictions.npz') as batched:
        mask = batched['trials'] == trial
        order_a = np.argsort(origins)
        order_b = np.argsort(batched['origin_time'][mask])
        if not np.allclose(np.asarray(origins)[order_a], batched['origin_time'][mask][order_b]):
            raise AssertionError('Replay and evaluation forecast origins differ.')
        error = np.max(np.abs(np.asarray(values)[order_a]-batched['snn'][mask][order_b]))
        metrics['max_replay_evaluation_difference'] = float(error)
        if error > 1e-3:
            raise AssertionError(f'Replay differs from batched evaluation: {error}')
    (run_dir/f'replay_trial_{trial}.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics, indent=2), flush=True)
    return output
