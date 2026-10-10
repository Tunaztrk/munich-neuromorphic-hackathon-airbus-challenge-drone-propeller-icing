"""Verify selected-checkpoint live replay, immutable weights and CPU latency."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import MODEL_ROOT, read_segments
from .regime_data import write_json
from .stream import StreamingForecaster


def check_runtime(root, trial=28):
    root = Path(root)
    selection = json.loads((root/'selection.json').read_text())
    cp = torch.load(root/'selected.pt', map_location='cpu', weights_only=True)
    cfg = cp['config']
    torch.set_num_threads(cfg['cpu_threads'])
    segments = [s for s in read_segments(MODEL_ROOT/cfg['source_csv'], dict(cfg, clean_intervals_path=None)) if s.trial == trial]
    if not segments:
        raise ValueError('No source observations for requested trial.')
    variant = 'fb_positive' if selection['chosen_feedback_learning_rate'] > 0 else 'frozen'
    path = root/selection['chosen_window_stage']/f'{variant}_test_predictions.npz'
    with np.load(path) as saved:
        mask = saved['trials'] == trial
        expected = {(round(float(t), 6)): y.copy() for t, y in zip(saved['origin_time'][mask], saved['snn'][mask])}
        versions = {(round(float(t), 6)): int(v) for t, v in zip(saved['origin_time'][mask], saved['model_version'][mask])}
    runtime = StreamingForecaster(root/'selected.pt')
    encoder_before = {k: v.clone() for k, v in runtime.model.state_dict().items()}
    elapsed, difference, fallback, event_clocks = [], [], 0, []
    for s in segments:
        runtime.reset()
        last_event = None
        for t, row in zip(s.time, s.raw):
            start = time.perf_counter()
            forecast = runtime.update(float(t), row)
            elapsed.append((time.perf_counter()-start)*1000.)
            fallback += int(runtime.regime.prediction_ledger[-1]['fallback'])
            key = round(float(t), 6)
            if key in expected:
                difference.append(float(np.max(np.abs(expected[key]-forecast))))
                if runtime.regime.last_prediction_version != versions[key]:
                    raise AssertionError('Issued live model version differs from chronological evaluation.')
            event = runtime.regime.last_event
            if event is not None and event['detected_time'] != last_event:
                event_clocks.append({'event_time': event['event_time'], 'detected_time': event['detected_time']})
                last_event = event['detected_time']
    for name, value in runtime.model.state_dict().items():
        if not torch.equal(value, encoder_before[name]):
            raise AssertionError(f'Frozen model weights changed: {name}')
    maximum_difference = max(difference, default=np.inf)
    if maximum_difference > 1e-3 or len(difference) != len(expected):
        raise AssertionError(f'Live replay mismatch: maximum={maximum_difference}, count={len(difference)}/{len(expected)}')
    report = {'trial': trial, 'source_ticks': len(elapsed), 'matched_forecasts': len(difference),
              'maximum_live_saved_forecast_difference_ampere': maximum_difference,
              'issued_versions_match': True, 'all_offline_model_weights_unchanged': True,
              'adaptive_parameter_count': sum(p.numel() for p in runtime.regime.adapter.head.parameters()),
              'fallback_tick_count': fallback, 'event_clocks': event_clocks,
              'feedback_learning_rate': selection['chosen_feedback_learning_rate'],
              'latency_scope': 'CPU event detection, feature processing, full window encoder, readout, ledger and due head update; excludes sensor IO and as-of resampling.',
              'cpu_threads': cfg['cpu_threads'], 'median_update_ms': float(np.median(elapsed)),
              'p95_update_ms': float(np.quantile(elapsed, .95)), 'maximum_update_ms': float(max(elapsed)),
              'sample_interval_ms': cfg['sample_interval_seconds']*1000.,
              'fraction_exceeding_sample_interval': float(np.mean(np.asarray(elapsed) > cfg['sample_interval_seconds']*1000.))}
    write_json(root/'runtime_check.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='regime_v1')
    parser.add_argument('--trial', type=int, default=28)
    args = parser.parse_args()
    check_runtime(MODEL_ROOT/'runs'/args.run, args.trial)


if __name__ == '__main__':
    main()
