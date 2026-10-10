"""Divided Inputs -> Sliding Window -> Forecast Feedback, measured as separate reproducible stages.

Offline labels select clean training segments only. Evaluation always replays
complete observed segments, confirming events causally rather than reading labels.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import (
    MODEL_ROOT,
    FeatureBuilder,
    PreparedData,
    prepare,
    read_segments,
    source_hash,
)
from .events import DropDetector
from .experiment import batches, metric_block, seed_everything
from .feedback import DelayedHead
from .network import build_model
from .regime_data import prepare_divided_input, write_json


@dataclass
class Trace:
    segment: object
    x: np.ndarray
    starts: np.ndarray
    fallback: np.ndarray
    blocked: np.ndarray
    events: dict


def causal_traces(segments, data, guard):
    """Features/context resets depend on detections available at the current tick."""
    traces = []
    for s in segments:
        builder, detector = FeatureBuilder(data.cfg), DropDetector()
        features, starts, fallback, blocked, events = [], [], [], [], {}
        start, clean_after, previous_transition = 0, -np.inf, False
        for j, (t, row) in enumerate(zip(s.time, s.raw)):
            event = detector.update(float(t), row)
            if event and event['status'] == 'confirmed_drop':
                events[j] = event
                clean_after = t+guard['post_guard_seconds']
                start, builder = j, FeatureBuilder(data.cfg)
            transition = t <= clean_after+1e-8
            if previous_transition and not transition:
                start, builder = j, FeatureBuilder(data.cfg)
            features.append(builder.step(float(t), row))
            starts.append(start)
            fallback.append(transition or j-start+1 < 5)
            blocked.append(transition or detector.pending is not None)
            previous_transition = transition
        x = np.asarray(features, dtype=np.float32)
        x = np.where(np.isfinite(x), x, data.imputer)
        x = ((x-data.center)/data.scale).astype(np.float32)
        traces.append(Trace(s, x, np.asarray(starts), np.asarray(fallback), np.asarray(blocked), events))
    return traces


def replay_records(traces, data, split, context_seconds=None):
    """Use the archived 20-second warmup and same 1-second origins for comparison."""
    if split == 'train':
        parents = {int(t.segment.trial) for t in traces}-set(data.cfg['validation_trials'])-set(data.cfg['test_trials'])
    elif split in {'validation', 'test'}:
        parents = set(data.cfg['validation_trials'] if split == 'validation' else data.cfg['test_trials'])
    else:
        raise ValueError('split must be train, validation, or test.')
    width = None if context_seconds is None else round(context_seconds/data.cfg['sample_interval_seconds'])
    result = []
    for i, trace in enumerate(traces):
        if trace.segment.trial not in parents:
            continue
        for o in range(99, len(trace.x)-data.horizon, data.stride):
            begin = int(trace.starts[o])
            if width is not None:
                begin = max(begin, o-width+1)
            result.append((i, o, begin))
    return result


def collate_windows(data, records, width=None):
    windows, targets = [], []
    for segment, origin in records:
        x, y, _ = data.segment(segment)
        begin = max(0, origin-width+1) if width is not None else 0
        windows.append(x[begin:origin+1])
        targets.append(data.standardized_target(y[origin+1:origin+1+data.horizon]))
    lengths = np.asarray([len(w) for w in windows])
    padded = np.zeros((len(windows), int(lengths.max()), len(data.feature_names)), dtype=np.float32)
    for j, window in enumerate(windows):
        padded[j, :len(window)] = window
    return torch.from_numpy(padded), torch.from_numpy(lengths), torch.from_numpy(np.asarray(targets, dtype=np.float32))


def training_windows(data, epoch):
    rng = np.random.default_rng(data.cfg['seed']+epoch)
    records = [(i, int(o)) for i in data.indices('train') for o in data.origins(len(data.segment(i)[0]))]
    rng.shuffle(records)
    width = round(data.cfg['window_seconds']/data.cfg['sample_interval_seconds'])
    contexts = [k for k in [5, 25, 50, 100, width] if k <= width]
    for a in range(0, len(records), data.cfg['window_batch_size']):
        # Teach short contexts after a reset as well as complete windows.
        context = width if rng.random() >= .25 else int(rng.choice(contexts))
        yield collate_windows(data, records[a:a+data.cfg['window_batch_size']], context)


@torch.no_grad()
def representations(model, data, traces, split, mode, width=None):
    records = replay_records(traces, data, split, width if mode == 'window' else None)
    z = np.zeros((len(records), data.cfg['hidden_size']), dtype=np.float32)
    anchors = np.zeros(len(records), dtype=np.float32)
    if mode == 'window':
        order = sorted(range(len(records)), key=lambda k: records[k][1]-records[k][2])
        batch_size = data.cfg.get('evaluation_window_batch_size', data.cfg['window_batch_size'])
        for a in range(0, len(order), batch_size):
            indices = order[a:a+batch_size]
            lengths = np.asarray([records[k][1]-records[k][2]+1 for k in indices])
            x = np.zeros((len(indices), int(lengths.max()), len(data.feature_names)), dtype=np.float32)
            for lane, k in enumerate(indices):
                i, o, b = records[k]
                x[lane, :lengths[lane]] = traces[i].x[b:o+1]
            rep, anchor, _ = model.window_encode(torch.from_numpy(x), torch.from_numpy(lengths))
            z[indices], anchors[indices] = rep.numpy(), anchor[:, 0].numpy()
    else:
        mapping = {(i, o): k for k, (i, o, _) in enumerate(records)}
        relevant = {i for i, _, _ in records}
        for i in sorted(relevant):
            trace, state = traces[i], None
            for j, sample in enumerate(trace.x):
                if trace.starts[j] == j:
                    state = None
                rep, anchor, state, _ = model.encode_step(torch.from_numpy(sample).unsqueeze(0), state)
                k = mapping.get((i, j))
                if k is not None:
                    z[k], anchors[k] = rep[0].numpy(), float(anchor[0, 0])
    return records, z, anchors


def forecast_arrays(model, data, traces, records, z, anchors, feedback_lr=0., guard=None):
    """Replay observations and delayed labels in timestamp order, even with cached z.

    Representations can be cached because the encoder and first decoder never
    adapt. Feedback targets are accumulated from observations, not read ahead.
    """
    actual = np.asarray([traces[i].segment.raw[o+1:o+1+data.horizon, 0] for i, o, _ in records])
    raw_current = np.asarray([traces[i].segment.raw[o, 0] for i, o, _ in records])
    trials = np.asarray([traces[i].segment.trial for i, _, _ in records])
    times = np.asarray([traces[i].segment.time[o] for i, o, _ in records])
    starts = np.asarray([traces[i].segment.time[b] for i, _, b in records])
    fallback = np.asarray([traces[i].fallback[o] for i, o, _ in records])
    with torch.no_grad():
        frozen = (torch.from_numpy(anchors).unsqueeze(1)+model.decoder[-1](torch.from_numpy(z))).numpy()*data.target_scale+data.target_mean
    frozen[fallback] = raw_current[fallback, None]
    arrays = {'actual': actual, 'trials': trials, 'origin_time': times, 'history_start_time': starts,
                  'snn': frozen.copy(), 'persistence': np.repeat(raw_current[:, None], data.horizon, axis=1),
                  'fallback': fallback, 'model_version': np.zeros(len(records), dtype=np.int64)}
    diagnostics = {'feedback_learning_rate': feedback_lr, 'update_count': 0,
                       'adaptive_parameters': sum(p.numel() for p in model.decoder[-1].parameters()),
                       'feature_statistics_frozen': True, 'encoder_frozen': True, 'update_log': []}
    if feedback_lr <= 0:
        return arrays, diagnostics
    guard = guard or {'post_guard_seconds': 0.}
    mapping = {(i, o): k for k, (i, o, _) in enumerate(records)}
    relevant = sorted({i for i, _, _ in records})
    for i in relevant:
        trace = traces[i]
        adapter = DelayedHead(model.decoder[-1], data.horizon, data.cfg['sample_interval_seconds'],
                              data.target_mean, data.target_scale, learning_rate=feedback_lr)
        # Restore the offline head and optimizer at every source segment; neither
        # past held-out trials nor pre-gap state can train the next segment.
        for j, (t, row) in enumerate(zip(trace.segment.time, trace.segment.raw)):
            if j in trace.events:
                adapter.event(float(t), guard['post_guard_seconds'])
            adapter.observe(float(t), float(row[0]))
            k = mapping.get((i, j))
            if k is not None:
                prediction, version = adapter.issue(float(t), z[k], anchors[k], starts[k], eligible=not fallback[k])
                arrays['snn'][k] = raw_current[k] if fallback[k] else prediction
                arrays['model_version'][k] = version
            adapter.maybe_update(float(t), blocked=bool(trace.blocked[j]))
        diagnostics['update_count'] += adapter.total_updates
        diagnostics['update_log'].extend(dict(parent_trial_id=int(trace.segment.trial), source_segment=trace.segment.key, **r) for r in adapter.update_log)
    return arrays, diagnostics


def summarize(arrays, split, events):
    actual, trial = arrays['actual'], arrays['trials']
    free = np.ones(len(actual), dtype=bool)
    for e in events:
        free &= ~((trial == e['parent_trial_id']) & (arrays['origin_time'] < e['event_time']) &
                  (arrays['origin_time']+10. >= e['event_time']))
    metrics = {'split': split, 'current_unit': 'A', 'forecast_count': len(actual), 'forecast_coverage': 1.,
                   'short_context_persistence_count': int(arrays['fallback'].sum()),
                   'target_horizon_event_free_count': int(free.sum()), 'models': {}}
    for name in ['snn', 'persistence', 'previous_snn', 'linear']:
        if name not in arrays:
            continue
        pred = arrays[name]
        block = metric_block(actual, pred)
        block['by_trial'] = {str(t): metric_block(actual[trial == t], pred[trial == t]) for t in np.unique(trial)}
        block['macro_trial_mae'] = float(np.mean([b['mae'] for b in block['by_trial'].values()]))
        block['target_horizon_event_free'] = metric_block(actual[free], pred[free]) if free.any() else None
        block['target_horizon_crosses_event'] = metric_block(actual[~free], pred[~free]) if (~free).any() else None
        metrics['models'][name] = block
    return metrics


def attach_previous(arrays, split):
    path = MODEL_ROOT/'runs'/'telemetry_tau10'/f'{split}_predictions.npz'
    with np.load(path) as old:
        mapping = {(int(t), round(float(s), 6)): k for k, (t, s) in enumerate(zip(old['trials'], old['origin_time']))}
        keep = [k for k, (t, s) in enumerate(zip(arrays['trials'], arrays['origin_time'])) if (int(t), round(float(s), 6)) in mapping]
        old_indices = [mapping[(int(arrays['trials'][k]), round(float(arrays['origin_time'][k]), 6))] for k in keep]
        count_before = len(arrays['trials'])
        arrays = {name: value[keep] for name, value in arrays.items()}
        if not np.allclose(arrays['actual'], old['actual'][old_indices], atol=1e-5):
            raise AssertionError('Source targets differ from the archived comparison.')
        arrays['previous_snn'] = old['snn'][old_indices]
        arrays['linear'] = old['linear'][old_indices]
    return arrays, count_before


def checkpoint_payload(model, data, cfg, epoch, guard):
    return {'state_dict': model.state_dict(), 'config': cfg, 'epoch': epoch,
                'feature_names': data.feature_names.tolist(), 'imputer': data.imputer.tolist(),
                'center': data.center.tolist(), 'scale': data.scale.tolist(), 'target_mean': data.target_mean,
                'target_scale': data.target_scale, 'event_guard': guard,
                'model_family': 'regime_v1', 'online_update_scope': 'final linear decoder only'}


def train_stage(data, traces, cfg, directory, guard, events):
    directory.mkdir()
    seed_everything(cfg['seed'], cfg['cpu_threads'])
    # PreparedData holds the training policy as well as scaling; each stage keeps
    # its own configuration without mutating normalization arrays.
    data.cfg = cfg
    model = build_model(data, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    history, best, stale = [], np.inf, 0
    started = time.perf_counter()
    for epoch in range(cfg['epochs']):
        model.train()
        count, total, state = 0, 0., None
        if cfg['inference_mode'] == 'window':
            iterator = training_windows(data, epoch)
            for x, lengths, y in iterator:
                optimizer.zero_grad(set_to_none=True)
                z, anchor, activity = model.window_encode(x, lengths)
                pred = anchor+model.decoder[-1](z)
                loss = (pred-y).square().mean()+cfg['activity_regularization']*activity.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite window training loss.')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip_norm'])
                optimizer.step()
                count += len(x)
                total += float(loss.detach())*len(x)
        else:
            for _, reset, x, y, mask, _ in batches(data, 'train', True, cfg['seed']+epoch):
                state = None if reset else model.detach_state(state)
                optimizer.zero_grad(set_to_none=True)
                prediction, state, activity = model(x, state)
                n = int(mask.sum())
                if not n:
                    continue
                loss = (prediction[mask]-y[mask]).square().mean()+cfg['activity_regularization']*activity[mask].mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite segmented training loss.')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip_norm'])
                optimizer.step()
                count += n
                total += float(loss.detach())*n
        model.eval()
        records, z, anchors = representations(model, data, traces, 'validation', cfg['inference_mode'], cfg.get('window_seconds'))
        arrays, _ = forecast_arrays(model, data, traces, records, z, anchors)
        arrays, _ = attach_previous(arrays, 'validation')
        metrics = summarize(arrays, 'validation', events)
        score = metrics['models']['snn']['macro_trial_mae']
        row = {'epoch': epoch+1, 'training_loss': total/max(count, 1), 'training_forecasts': count,
                   'validation_macro_trial_mae': score, 'elapsed_seconds': time.perf_counter()-started}
        history.append(row)
        print(json.dumps(dict(stage=directory.name, **row)), flush=True)
        if score < best:
            best, stale = score, 0
            torch.save(checkpoint_payload(model, data, cfg, epoch+1, guard), directory/'best.pt')
            write_json(directory/'validation_metrics.json', metrics)
            np.savez_compressed(directory/'validation_predictions.npz', **arrays)
        else:
            stale += 1
        write_json(directory/'history.json', history)
        if stale >= cfg['patience']:
            break
    write_json(directory/'config.json', cfg)
    return best


def load_stage(directory, data):
    cp = torch.load(directory/'best.pt', map_location='cpu', weights_only=True)
    model = build_model(data, cp['config'])
    model.load_state_dict(cp['state_dict'])
    model.eval()
    return model, cp


def run_pipeline(run_name='regime_v1', epochs=4, windows=(20., 30., 60.)):
    if Path(run_name).name != run_name or run_name in {'.', '..'}:
        raise ValueError('Run name must be one folder name.')
    if not windows or any(w < 1 or not np.isfinite(w) or not np.isclose(w/.2, round(w/.2)) for w in windows):
        raise ValueError('Window durations must be >=1 s and multiples of 0.2 s.')
    root = MODEL_ROOT/'runs'/run_name
    root.mkdir(exist_ok=False)
    cfg = json.loads((MODEL_ROOT/'config'/'default.json').read_text())
    cfg.update(warmup_seconds=1., epochs=epochs,
               patience=2, window_batch_size=128, model_family='regime_v1',
               feedback_issue_warmup_seconds=20.)
    dataset_path = MODEL_ROOT/'data'/'divided_input_v1'
    if not dataset_path.exists():
        prepare_divided_input(MODEL_ROOT/cfg['source_csv'], cfg, dataset_path)
    manifest = json.loads((dataset_path/'manifest.json').read_text())
    if manifest['source_sha256'] != source_hash(MODEL_ROOT/cfg['source_csv']) or manifest['validation_trials'] != cfg['validation_trials'] or manifest['test_trials'] != cfg['test_trials']:
        raise ValueError('Cached Divided Inputs source or parent splits changed; regenerate the dataset before training.')
    cfg['clean_intervals_path'] = str(dataset_path/'intervals.json')
    events = json.loads((dataset_path/'events.json').read_text())
    guard = manifest['transition_guard']
    prepared = prepare(cfg, root/'prepared.npz')
    data = PreparedData(prepared, cfg)
    seed_everything(cfg['seed'], cfg['cpu_threads'])
    segments = read_segments(MODEL_ROOT/cfg['source_csv'], dict(cfg, clean_intervals_path=None))
    traces = causal_traces(segments, data, guard)
    scores = {}
    stage_cfg = dict(cfg, inference_mode='stateful', window_seconds=None)
    scores['divided_input'] = train_stage(data, traces, stage_cfg, root/'divided_input', guard, events)
    for width in windows:
        name = f'pencere_{width:g}s'
        stage_cfg = dict(cfg, inference_mode='window', window_seconds=float(width))
        scores[name] = train_stage(data, traces, stage_cfg, root/name, guard, events)
    return finish_pipeline(root, data, traces, scores, events, guard)


def finish_pipeline(root, data, traces, scores, events, guard):
    chosen_window = min((k for k in scores if k.startswith('pencere_')), key=scores.get)
    model, checkpoint = load_stage(root/chosen_window, data)
    # Binary spikes can amplify tiny batched/single-lane GEMM differences.
    # Final scores use the same single-window arithmetic as the live runtime.
    checkpoint['config']['evaluation_window_batch_size'] = 1
    data.cfg = checkpoint['config']
    val_records, val_z, val_anchor = representations(model, data, traces, 'validation', 'window', data.cfg['window_seconds'])
    feedback_scores = {}
    for lr in [0., 1e-5, 5e-5]:
        arrays, diagnostic = forecast_arrays(model, data, traces, val_records, val_z, val_anchor, lr, guard)
        arrays, count_before = attach_previous(arrays, 'validation')
        metrics = summarize(arrays, 'validation', events)
        metrics.update(diagnostic, available_forecast_origins=count_before)
        name = f'fb_lr_{lr:g}'
        feedback_scores[name] = metrics['models']['snn']['macro_trial_mae']
        write_json(root/f'{name}_validation_metrics.json', metrics)
        np.savez_compressed(root/f'{name}_validation_predictions.npz', **arrays)
        print(json.dumps({'feedback_lr': lr, 'validation_macro_trial_mae': feedback_scores[name], 'updates': diagnostic['update_count']}), flush=True)
    chosen_fb = min(feedback_scores, key=feedback_scores.get)
    chosen_lr = float(chosen_fb.split('_')[-1])
    # A positive-rate candidate is reported even if disabling Forecast Feedback wins validation.
    positive_fb = min((k for k in feedback_scores if k != 'fb_lr_0'), key=feedback_scores.get)
    positive_lr = float(positive_fb.split('_')[-1])
    for stage in ['divided_input', chosen_window]:
        stage_model, cp = load_stage(root/stage, data)
        data.cfg = cp['config']
        data.cfg['evaluation_window_batch_size'] = 1
        records, z, anchor = representations(stage_model, data, traces, 'test', data.cfg['inference_mode'], data.cfg.get('window_seconds'))
        for label, lr in ([('frozen', 0.)] if stage == 'divided_input' else [('frozen', 0.), ('fb_positive', positive_lr)]):
            arrays, diagnostic = forecast_arrays(stage_model, data, traces, records, z, anchor, lr, guard)
            arrays, count_before = attach_previous(arrays, 'test')
            metrics = summarize(arrays, 'test', events)
            metrics.update(diagnostic, available_forecast_origins=count_before,
                           matched_archived_origin_coverage=len(arrays['trials'])/count_before)
            write_json(root/stage/f'{label}_test_metrics.json', metrics)
            np.savez_compressed(root/stage/f'{label}_test_predictions.npz', **arrays)
    checkpoint['config']['feedback_learning_rate'] = chosen_lr
    torch.save(checkpoint, root/'selected.pt')
    selection = {'stage_validation_macro_mae': scores, 'chosen_window_stage': chosen_window,
                     'feedback_validation_macro_mae': feedback_scores, 'chosen_feedback_learning_rate': chosen_lr,
                     'positive_feedback_candidate_learning_rate': positive_lr,
                     'selection_scope': 'Validation parents only. Test is exploratory because it has been inspected previously.',
                     'checkpoint': 'selected.pt', 'archived_default_unchanged': True,
                     'training_epochs_limit': data.cfg['epochs'], 'current_unit': 'A',
                     'evaluation_arithmetic': 'Final selected forecasts use single-window CPU arithmetic as in the live API; offline training and epoch/window selection use batched arithmetic.'}
    write_json(root/'selection.json', selection)
    create_regime_report(root)
    print(f'Completed: {root}', flush=True)
    return root


def refresh_evaluation(root):
    """Recompute final reports from saved offline weights, without retraining."""
    root = Path(root)
    selection = json.loads((root/'selection.json').read_text())
    cp = torch.load(root/'selected.pt', map_location='cpu', weights_only=True)
    cfg = cp['config']
    seed_everything(cfg['seed'], cfg['cpu_threads'])
    data = PreparedData(root/'prepared.npz', cfg)
    segments = read_segments(MODEL_ROOT/cfg['source_csv'], dict(cfg, clean_intervals_path=None))
    traces = causal_traces(segments, data, cp['event_guard'])
    events = json.loads((MODEL_ROOT/'data'/'divided_input_v1'/'events.json').read_text())
    return finish_pipeline(root, data, traces, selection['stage_validation_macro_mae'], events, cp['event_guard'])


def create_regime_report(root):
    os.environ.setdefault('MPLCONFIGDIR', str(MODEL_ROOT/'.cache'/'matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(MODEL_ROOT/'.cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    choice = json.loads((root/'selection.json').read_text())
    window = choice['chosen_window_stage']
    rows = []
    for stage, variant in [('divided_input', 'frozen'), (window, 'frozen'), (window, 'fb_positive')]:
        metrics = json.loads((root/stage/f'{variant}_test_metrics.json').read_text())
        b = metrics['models']['snn']
        rows.append(f"| {stage}/{variant} | {b['mae']:.6f} | {b['macro_trial_mae']:.6f} | {b['endpoint_mae']:.6f} | {metrics['update_count']} |")
    with np.load(root/window/'frozen_test_predictions.npz') as frozen, np.load(root/window/'fb_positive_test_predictions.npz') as online:
        for trial in [28, 32]:
            mask = frozen['trials'] == trial
            time_target = frozen['origin_time'][mask]+10.
            fig, axes = plt.subplots(2, 1, figsize=(12, 7))
            for ax in axes:
                ax.plot(time_target, frozen['actual'][mask, -1], label='Measured current', lw=1)
                ax.plot(time_target, frozen['previous_snn'][mask, -1], label='Archived SNN, 10 s ahead', lw=1, alpha=.7)
                ax.plot(time_target, frozen['snn'][mask, -1], label=f'{window}, frozen, 10 s ahead', lw=1)
                ax.plot(time_target, online['snn'][mask, -1], label='Window SNN + delayed Forecast Feedback, 10 s ahead', lw=1)
                ax.set_ylabel('Current (A)')
                ax.grid(alpha=.2)
            events = json.loads((MODEL_ROOT/'data'/'divided_input_v1'/'events.json').read_text())
            e = next((e for e in events if e['parent_trial_id'] == trial), None)
            if e:
                for ax in axes:
                    ax.axvline(e['event_time'], color='black', ls='--', label='Event time')
                    ax.axvline(e['detected_time'], color='gray', ls=':', label='Detection time')
                axes[1].set_xlim(e['event_time']-25, e['event_time']+60)
            axes[0].legend(fontsize=8)
            axes[1].set_xlabel('Target timestamp (s): forecast was issued 10 s earlier')
            fig.suptitle(f'Trial {trial}: unchanged raw targets, equal forecast origins')
            fig.tight_layout()
            fig.savefig(root/f'trial_{trial}_comparison.png', dpi=150)
            plt.close(fig)
        baseline = {name: metric_block(frozen['actual'], frozen[name]) for name in ['previous_snn', 'persistence', 'linear']}
    adaptive = json.loads((root/window/'fb_positive_test_metrics.json').read_text())
    event_rows = []
    for name in ['previous_snn', 'persistence', 'linear', 'snn']:
        block = adaptive['models'][name]
        event_rows.append(f"| {name} | {block['target_horizon_event_free']['mae']:.6f} | {block['target_horizon_crosses_event']['mae']:.6f} |")
    crossing_count = adaptive['forecast_count']-adaptive['target_horizon_event_free_count']
    document = f'''# Divided Inputs / Sliding Window / Forecast Feedback prototype results

Current unit: A. Target: original as-of current at 50 leads, 0.2–10 s.
Ice remains a separate target and is absent from all 40 inputs.
All stages preserve 31 parent training trials, 7 validation and 8 test trials.

## Matched test origins

| Stage | Pooled MAE (A) | Macro parent MAE (A) | 10 s endpoint MAE (A) | Online updates |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Archived SNN pooled MAE: {baseline['previous_snn']['mae']:.6f} A.
Persistence pooled MAE: {baseline['persistence']['mae']:.6f} A.
Archived linear baseline pooled MAE: {baseline['linear']['mae']:.6f} A.

## Ordinary trajectories versus crossing-event horizons

| Model | Event-free target-horizon MAE (A) | Crossing-event target-horizon MAE (A) |
| --- | ---: | ---: |
{chr(10).join(event_rows)}

There are {crossing_count} overlapping crossing-event forecast windows. These
are not independent physical events or trials. Improvement is concentrated in
ordinary trajectories; crossing-event error remains high, and this result does
not establish early fracture prediction. The frozen linear baseline still has
lower pooled error than the adaptive SNN. The SNN uses Forecast Feedback only after actual
labels arrive; frozen and adaptive results must remain separately identified.

## Selection and interpretation

Window chosen by validation macro parent MAE: `{window}`.
Default Forecast Feedback learning rate chosen by validation: {choice['chosen_feedback_learning_rate']}.
Positive Forecast Feedback candidate: {choice['positive_feedback_candidate_learning_rate']}.
Zero learning rate is an explicit control: Forecast Feedback stays disabled if it wins validation.
`selected.pt` contains the chosen window and Forecast Feedback policy. The previous default
checkpoint remains available unchanged. All checkpoint/epoch/window/Forecast Feedback choices
use validation parents, not these test plots.

Offline training and epoch/window selection use batched arithmetic; final
selected-window evaluation uses single-window CPU arithmetic matching the live
API. This avoids binary spike differences near numerical thresholds. The
arithmetic audit and final live replay check are stored separately.

These are short prototype runs, not a converged architecture search. Test trials
have been inspected repeatedly in earlier work, so results are exploratory.
Segmentation improves the training objective's focus on ordinary trajectories;
it does not teach the model when unseen physical shedding will happen. Every
drop remains in the main test targets and forecasts are produced throughout
events. Early post-reset contexts shorter than 1 s use explicit persistence.
Metrics report this fallback count and the origin intersection with the archive.

`target_horizon_event_free` and `target_horizon_crosses_event` metrics are
retrospective evaluation strata only, never inputs or online reset triggers.
Predictions in NPZ files are immutable forecasts saved when issued; the target
timestamps are origin+lead. Forecast Feedback consumes each actual target only after it arrives,
updates a detached copy of the final 48→50 decoder, and restores the offline
head/optimizer/buffers at detections and original observation gaps. Its learning
uses held-out observations that are already past, so adaptive metrics must be
reported separately from frozen held-out inference.

## Artifacts

- `selection.json`: validation-only stage and Forecast Feedback choices.
- `prepared.npz/json`: clean training segments and training-only normalization.
- Stage folders: immutable checkpoints, training history and forecast arrays.
- Forecast Feedback metrics: update audit with update_time and maximum_target_time.
- `trial_28_comparison.png`, `trial_32_comparison.png`: aligned 10 s forecasts.
- `../../data/divided_input_v1/DATASET.md`: event rules and data preservation.

See `../../REGIME_APPROACHES.md` for architecture, commands, clocks and limitations.
'''
    (root/'RESULTS.md').write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='regime_v1')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--windows', type=float, nargs='+', default=[20., 30., 60.])
    parser.add_argument('--refresh-evaluation', action='store_true', help='Recompute final evaluations from existing immutable offline weights.')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.refresh_evaluation:
        refresh_evaluation(MODEL_ROOT/'runs'/args.run)
    else:
        run_pipeline(args.run, args.epochs, args.windows)


if __name__ == '__main__':
    main()
