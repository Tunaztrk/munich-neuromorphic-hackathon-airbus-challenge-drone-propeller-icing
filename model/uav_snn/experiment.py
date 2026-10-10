"""Stateful chunk training, leakage-safe baselines and multi-horizon metrics."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .data import MODEL_ROOT, PreparedData, prepare
from .network import build_model


def seed_everything(seed, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def batches(data, split, shuffle=False, seed=0):
    """Stream separate segments in parallel lanes, preserving their time order.

    Recurrent state carries across consecutive chunks; gradients are truncated.
    Padding appears only after a lane ends, receives no loss, and is never reused.
    """
    indices = sorted(data.indices(split), key=lambda i: data.offsets[i+1]-data.offsets[i])
    bsize, chunk = data.cfg['batch_size'], data.cfg['chunk_steps']
    groups = [indices[i:i+bsize] for i in range(0, len(indices), bsize)]
    if shuffle:
        random.Random(seed).shuffle(groups)
    for group_number, group in enumerate(groups):
        series = [data.segment(i) for i in group]
        maxlen = max(len(x) for x, _, _ in series)
        for start in range(0, maxlen, chunk):
            steps = min(chunk, maxlen-start)
            x = np.zeros((len(group), steps, len(data.feature_names)), dtype=np.float32)
            targets = np.zeros((len(group), steps, data.horizon), dtype=np.float32)
            mask = np.zeros((len(group), steps), dtype=bool)
            records = []
            for lane, (features, current, timestamp) in enumerate(series):
                n = max(0, min(steps, len(features)-start))
                if n:
                    x[lane, :n] = features[start:start+n]
                origins = data.origins(len(features))
                origins = origins[(origins >= start) & (origins < start+steps)]
                for origin in origins:
                    local = origin-start
                    targets[lane, local] = data.standardized_target(current[origin+1:origin+1+data.horizon])
                    mask[lane, local] = True
                    records.append((lane, local, group[lane], int(origin)))
            yield group_number, start == 0, torch.from_numpy(x), torch.from_numpy(targets), torch.from_numpy(mask), records


def linear_features(data, x, origin):
    """Latest multivariate observation plus causal current slopes (1/5/10 s)."""
    slopes = []
    dt = data.cfg['sample_interval_seconds']
    for duration in [1., 5., 10.]:
        previous = max(0, origin-round(duration/dt))
        slopes.append((x[origin, data.current_index]-x[previous, data.current_index])/
                      max((origin-previous)*dt, dt))
    return np.r_[1., x[origin], slopes].astype(np.float64)


def fit_linear(data, run_dir):
    count = len(data.feature_names)+4
    xtx = np.zeros((count, count))
    xty = np.zeros((count, data.horizon))
    for i in data.indices('train'):
        x, current, _ = data.segment(i)
        origins = data.origins(len(x))
        features = np.stack([linear_features(data, x, origin) for origin in origins])
        targets = np.stack([data.standardized_target(current[o+1:o+1+data.horizon])-
                            data.standardized_target(current[o]) for o in origins])
        # Give each independent training trial segment equal total weight.
        xtx += features.T@features/len(origins)
        xty += features.T@targets/len(origins)
    regularizer = np.eye(count)*data.cfg['ridge_regularization']
    regularizer[0, 0] = 0.
    coefficients = np.linalg.solve(xtx+regularizer, xty)
    np.savez(run_dir/'linear_baseline.npz', coefficients=coefficients)
    return coefficients


def metric_block(actual, predictions):
    error = predictions-actual
    return {'mae': float(np.abs(error).mean()), 'rmse': float(np.sqrt(np.square(error).mean())),
            'endpoint_mae': float(np.abs(error[:, -1]).mean()),
            'mae_by_lead': np.abs(error).mean(axis=0).tolist(),
            'rmse_by_lead': np.sqrt(np.square(error).mean(axis=0)).tolist()}


def evaluate(model, data, split, coefficients=None):
    model.eval()
    device = next(model.parameters()).device
    predictions, targets, persistent, linear, trials, times, activity = [], [], [], [], [], [], []
    state = None
    with torch.no_grad():
        for _, reset, x, y, mask, records in batches(data, split):
            if reset:
                state = None
            forecast, state, spikes = model(x.to(device), state)
            forecast = forecast.cpu().numpy()*data.target_scale+data.target_mean
            if mask.any():
                activity.append(spikes.cpu().numpy()[mask.numpy()])
            for lane, local, segment, origin in records:
                features, current, timestamp = data.segment(segment)
                predictions.append(forecast[lane, local])
                targets.append(current[origin+1:origin+1+data.horizon])
                persistent.append(np.full(data.horizon, current[origin]))
                if coefficients is not None:
                    delta = linear_features(data, features, origin)@coefficients
                    linear.append(current[origin]+delta*data.target_scale)
                trials.append(int(data.trials[segment]))
                times.append(float(timestamp[origin]))
    if not targets:
        raise ValueError(f'No valid forecast origins in {split}.')
    actual = np.asarray(targets)
    pred = {'snn': np.asarray(predictions), 'persistence': np.asarray(persistent)}
    if linear:
        pred['linear'] = np.asarray(linear)
    trial_arr = np.asarray(trials)
    metrics = {'split': split, 'forecast_count': len(actual), 'current_units': 'original dataset units',
               'lead_seconds': ((np.arange(data.horizon)+1)*data.cfg['sample_interval_seconds']).tolist(),
               'models': {}, 'spike_fraction_by_layer': np.concatenate(activity).mean(axis=0).tolist()}
    # Future target values are used only for evaluation stratification, never inputs.
    abrupt = np.min(actual-np.asarray(persistent), axis=1) <= -2.
    for name, forecasts in pred.items():
        block = metric_block(actual, forecasts)
        block['by_trial'] = {str(t): metric_block(actual[trial_arr == t], forecasts[trial_arr == t])
                             for t in np.unique(trial_arr)}
        block['macro_trial_mae'] = float(np.mean([v['mae'] for v in block['by_trial'].values()]))
        block['abrupt_drop_windows'] = metric_block(actual[abrupt], forecasts[abrupt]) if abrupt.any() else None
        block['abrupt_drop_window_count'] = int(abrupt.sum())
        metrics['models'][name] = block
    arrays = {'actual': actual, 'trials': trial_arr, 'origin_time': np.asarray(times), **pred}
    return metrics, arrays


def train(cfg, run_name):
    if cfg.get('input_schema') != 'telemetry_v2':
        raise ValueError('New training requires input_schema=telemetry_v2; legacy ice-input configurations are replay-only.')
    if Path(run_name).name != run_name or run_name in {'.', '..'}:
        raise ValueError('Run name must be a single folder name.')
    run_dir = MODEL_ROOT/'runs'/run_name
    if run_dir.exists():
        raise FileExistsError(f'Run already exists: {run_dir}; choose another run name.')
    run_dir.mkdir(parents=True)
    (run_dir/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    seed_everything(cfg['seed'], cfg['cpu_threads'])
    prepared = prepare(cfg, run_dir/'prepared.npz')
    data = PreparedData(prepared, cfg)
    device = torch.device(cfg['device'])
    model = build_model(data, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=.5)
    coefficients = fit_linear(data, run_dir)
    best, stale, history = float('inf'), 0, []
    start_time = time.perf_counter()
    for epoch in range(cfg['epochs']):
        model.train()
        state = None
        sum_loss, forecasts = 0., 0
        for batch_index, (_, reset, x, y, mask, _) in enumerate(batches(data, 'train', True, cfg['seed']+epoch)):
            if reset:
                state = None
            elif state is not None:
                state = model.detach_state(state)
            optimizer.zero_grad(set_to_none=True)
            pred, state, activity = model(x.to(device), state)
            valid = mask.to(device)
            n = int(mask.sum())
            if not n:
                continue
            errors = pred[valid]-y.to(device)[valid]
            loss = errors.square().mean()+cfg['activity_regularization']*activity[valid].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip_norm'])
            optimizer.step()
            sum_loss += float(loss.detach())*n
            forecasts += n
        validation, _ = evaluate(model, data, 'validation')
        score = validation['models']['snn']['macro_trial_mae']
        scheduler.step(score)
        row = {'epoch': epoch+1, 'training_loss': sum_loss/max(forecasts, 1),
               'validation_macro_trial_mae': score,
               'validation_mae': validation['models']['snn']['mae'],
               'learning_rate': optimizer.param_groups[0]['lr'],
               'elapsed_seconds': time.perf_counter()-start_time}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best:
            best, stale = score, 0
            torch.save({'state_dict': model.state_dict(), 'config': cfg, 'epoch': epoch+1,
                        'feature_names': data.feature_names.tolist(), 'imputer': data.imputer.tolist(),
                        'center': data.center.tolist(), 'scale': data.scale.tolist(),
                        'target_mean': data.target_mean, 'target_scale': data.target_scale}, run_dir/'best.pt')
        else:
            stale += 1
        (run_dir/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        if stale >= cfg['patience']:
            break
    checkpoint = torch.load(run_dir/'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    for split in ['validation', 'test']:
        metrics, arrays = evaluate(model, data, split, coefficients)
        metrics['selected_epoch'] = checkpoint['epoch']
        metrics['parameters'] = sum(p.numel() for p in model.parameters())
        (run_dir/f'{split}_metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
        np.savez_compressed(run_dir/f'{split}_predictions.npz', **arrays)
    from .report import create_report
    create_report(run_dir)
    print(f'Completed run: {run_dir}', flush=True)
    return run_dir
