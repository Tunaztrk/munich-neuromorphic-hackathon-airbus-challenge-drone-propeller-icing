"""Standalone forecast visualizations and an English results report."""
import json
import os
from pathlib import Path

import numpy as np

from .data import MODEL_ROOT

os.environ.setdefault('MPLCONFIGDIR', str(MODEL_ROOT/'.mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME', str(MODEL_ROOT/'.cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def create_report(run_dir):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir/'config.json').read_text())
    metrics = json.loads((run_dir/'test_metrics.json').read_text())
    with np.load(run_dir/'test_predictions.npz') as source:
        arrays = {key: source[key].copy() for key in source.files}
    lead = np.asarray(metrics['lead_seconds'])
    fig, ax = plt.subplots(figsize=(9, 4))
    for name in ['snn', 'persistence', 'linear']:
        ax.plot(lead, metrics['models'][name]['mae_by_lead'], label=name)
    ax.set(xlabel='Forecast lead time (s)', ylabel='MAE (original current units)',
           title='Held-out trials: current forecast error by lead time')
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(run_dir/'error_by_lead.png', dpi=160); plt.close(fig)
    records = []
    # Include origins before confirmed events, avoiding a demo restricted to easy post-event forecasts.
    events = {28: 1159.15, 32: 1195.72}
    for trial, event in events.items():
        subset = np.flatnonzero(arrays['trials'] == trial)
        if not len(subset):
            continue
        for offset in [-5., 1.]:
            index = subset[np.argmin(np.abs(arrays['origin_time'][subset]-(event+offset)))]
            origin = float(arrays['origin_time'][index])
            fig, ax = plt.subplots(figsize=(9, 4))
            for name in ['actual', 'snn', 'persistence', 'linear']:
                ax.plot(origin+lead, arrays[name][index], label=name)
            ax.axvline(event, color='#b45309', linestyle='--', label='Confirmed shedding')
            ax.set(xlabel='Trial timestamp (s)', ylabel='Current (original units)',
                   title=f'Trial {trial}: forecast issued at {origin:.1f} s')
            ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
            filename = f'trial_{trial}_{"before" if offset < 0 else "after"}_event.png'
            fig.savefig(run_dir/filename, dpi=160); plt.close(fig)
            records.append((trial, origin, filename))
    lines = ['# Prototype results', '',
             f"Forecast duration: {cfg['forecast_seconds']} s; sampling interval: {cfg['sample_interval_seconds']} s.",
             f"Selected epoch: {metrics['selected_epoch']}; parameters: {metrics['parameters']:,}.",
             'The checkpoint was selected using validation macro-trial MAE; held-out test trials were not used for selection.', '',
             '| Model | MAE | RMSE | Macro-trial MAE | Endpoint MAE |',
             '|---|---:|---:|---:|---:|']
    for name, block in metrics['models'].items():
        lines.append(f"| {name} | {block['mae']:.4f} | {block['rmse']:.4f} | {block['macro_trial_mae']:.4f} | {block['endpoint_mae']:.4f} |")
    lines += ['', 'Errors are in the original current units; units require confirmation from experiment records.', '',
              f"Spike fractions at valid forecast origins, by layer: {metrics['spike_fraction_by_layer']}.",
              'Software spike activity is not measured neuromorphic energy consumption.', '',
              '![Error by lead time](error_by_lead.png)', '', '## Confirmed event examples', '',
              'Forecasts before shedding may fail to anticipate an abrupt event. After-event adaptation and pre-event forecasting are separate behaviors.']
    for _, _, filename in records:
        lines += ['', f'![Forecast example]({filename})']
    lines += ['', '## Interpretation limits', '',
              'This is one software prototype, one fixed trial split, and one random seed. It is not a hardware deployment or a safety certification.',
              'Abrupt-drop metrics select future target decreases of at least 2 original current units for diagnostics only; they do not label every selected window as ice shedding.',
              ('Inputs are telemetry only: 12 raw channels, 16 relations and 12 missingness flags. Ice pixels/area are stored separately as targets and never enter the model. The decoder currently forecasts current only.'
               if cfg.get('input_schema') == 'telemetry_v2' else
               'Archived legacy run: inputs include camera-derived ice measures. It does not satisfy the current telemetry-only input requirement.')]
    (run_dir/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    create_html(run_dir, arrays, cfg)


def create_html(run_dir, arrays, cfg):
    trials = sorted(set(arrays['trials'].tolist()))
    packed = []
    # Load compressed arrays once; indexing NpzFile repeatedly decompresses entire arrays.
    with np.load(run_dir/'prepared.npz') as source:
        prepared = {key: source[key] for key in ['trials', 'offsets', 'time', 'y', 'x', 'feature_names']}
    snapshot_names = ['rpm', 'throttle_pct', 'temp_test_sec', 'lwc', 'Total_Ice_Area_mm2']
    feature_indices = {name: i for i, name in enumerate(prepared['feature_names'].tolist())}
    for trial in trials:
        mask = arrays['trials'] == trial
        order = np.argsort(arrays['origin_time'][mask])
        segments = [tuple(prepared['offsets'][i:i+2])
                    for i, t in enumerate(prepared['trials']) if int(t) == trial]
        series = [{'t': prepared['time'][a:b].tolist(), 'y': prepared['y'][a:b].tolist()}
                  for a, b in segments]
        snapshots = []
        for timestamp in arrays['origin_time'][mask][order]:
            for a, b in segments:
                if prepared['time'][a] <= timestamp <= prepared['time'][b-1]:
                    index = a+np.searchsorted(prepared['time'][a:b], timestamp)
                    snapshot = []
                    for name in snapshot_names:
                        available = name in feature_indices and prepared['x'][index, feature_indices[name+'_missing']] < .5
                        snapshot.append(float(prepared['x'][index, feature_indices[name]]) if available else None)
                    snapshots.append(snapshot)
                    break
            else:
                raise ValueError(f'Forecast origin {timestamp} in trial {trial} has no matching segment.')
        packed.append({'trial': trial, 'origin': arrays['origin_time'][mask][order].tolist(),
                       'snn': arrays['snn'][mask][order].tolist(),
                       'actual': arrays['actual'][mask][order].tolist(),
                       'persistence': arrays['persistence'][mask][order].tolist(),
                       'linear': arrays['linear'][mask][order].tolist(), 'history': series,
                       'sensor_snapshots': snapshots})
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>UAV current forecast replay</title>
<style>body{font:16px system-ui;margin:30px;background:#f5f7fa;color:#182434}canvas{width:100%;height:440px;background:white;border-radius:12px}input{width:70%}button,select{padding:9px;margin:12px}#info{margin:15px 0}</style>
<h1>UAV icing · Current trajectory forecast</h1><p>Held-out recorded trials. Ground truth is shown for evaluation; it is never supplied to the forecaster.</p>
<select id="trial"></select><button id="play">Play / pause</button><input id="frame" type="range" min="0" value="0"><div id="info"></div><div id="sensors"></div><canvas id="plot"></canvas>
<p>Blue: observed current history. Orange: SNN forecast. Green: true future. Gray: persistence. Red: linear baseline. Dashed line: forecast origin.</p>
<script>const data=__DATA__,dt=__DT__;const tr=document.getElementById('trial'),fr=document.getElementById('frame');let timer=null;for(const d of data){let o=document.createElement('option');o.value=d.trial;o.textContent='Trial '+d.trial;tr.append(o)}
function render(){const d=data.find(v=>v.trial===Number(tr.value));fr.max=d.origin.length-1;const i=Math.min(Number(fr.value),d.origin.length-1),origin=d.origin[i];const series=d.history.find(s=>origin>=s.t[0]&&origin<=s.t[s.t.length-1]);
const c=document.getElementById('plot'),w=c.clientWidth,h=440,r=devicePixelRatio||1;c.width=w*r;c.height=h*r;const g=c.getContext('2d');g.scale(r,r);let points=[];if(series)for(let j=0;j<series.t.length;j++)if(series.t[j]>=origin-30&&series.t[j]<=origin)points.push([series.t[j],series.y[j]]);
const future=name=>d[name][i].map((v,k)=>[origin+(k+1)*dt,v]);let values=[...points,...future('actual'),...future('snn'),...future('persistence'),...future('linear')].map(p=>p[1]);let lo=Math.min(...values)-.5,hi=Math.max(...values)+.5;const x=t=>60+(t-origin+30)/(30+d.snn[i].length*dt)*(w-85),y=v=>h-40-(v-lo)/(hi-lo)*(h-70);
g.font='13px system-ui';for(let j=0;j<=5;j++){let v=lo+(hi-lo)*j/5;g.strokeStyle='#e5e9ef';g.beginPath();g.moveTo(60,y(v));g.lineTo(w-25,y(v));g.stroke();g.fillStyle='#526074';g.fillText(v.toFixed(2),8,y(v)+4);let t=origin-30+(30+d.snn[i].length*dt)*j/5;g.fillText(t.toFixed(1),x(t)-12,h-15)}
for(const [p,color] of [[points,'#2563eb'],[future('persistence'),'#9ca3af'],[future('linear'),'#dc2626'],[future('actual'),'#059669'],[future('snn'),'#ea580c']]){g.strokeStyle=color;g.lineWidth=2;g.beginPath();p.forEach((q,j)=>j?g.lineTo(x(q[0]),y(q[1])):g.moveTo(x(q[0]),y(q[1])));g.stroke()}
g.strokeStyle='#111827';g.setLineDash([5,5]);g.beginPath();g.moveTo(x(origin),25);g.lineTo(x(origin),h-40);g.stroke();g.setLineDash([]);document.getElementById('info').textContent=`Trial ${d.trial} · forecast origin ${origin.toFixed(1)} s · horizon ${(d.snn[i].length*dt).toFixed(1)} s`;const labels=['RPM','Throttle %','Temperature','LWC','Ice area mm²'];document.getElementById('sensors').textContent=d.sensor_snapshots[i].map((v,j)=>labels[j]+': '+(v===null?'unavailable':v.toFixed(2))).join(' · ')}
tr.onchange=()=>{fr.value=0;render()};fr.oninput=render;document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null}else timer=setInterval(()=>{fr.value=Number(fr.value)>=Number(fr.max)?0:Number(fr.value)+1;render()},200)};window.onresize=render;render();</script></html>'''
    (run_dir/'forecast_replay.html').write_text(html.replace('__DATA__', json.dumps(packed, separators=(',', ':'))).replace('__DT__', str(cfg['sample_interval_seconds'])))


def compare_runs():
    root = MODEL_ROOT/'runs'
    lines = ['# Experiment comparison', '',
             'Runs share a trial split and seed. Checkpoints are selected using validation macro-trial MAE.',
             'Different horizons address different forecasting difficulties; do not rank them as interchangeable tasks.', '',
             '| Run | τ (s) | Input schema / features | η | Selected epoch | Validation macro MAE | Test MAE | Test RMSE |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    records = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder/'test_metrics.json').exists():
            continue
        cfg = json.loads((folder/'config.json').read_text())
        validation = json.loads((folder/'validation_metrics.json').read_text())
        test = json.loads((folder/'test_metrics.json').read_text())
        record = {'run': folder.name, 'forecast_seconds': cfg['forecast_seconds'],
                  'input_schema': cfg.get('input_schema', 'legacy_ice_v1'),
                  'feature_set': cfg['feature_set'], 'learning_rate': cfg['learning_rate'],
                  'validation_macro_trial_mae': validation['models']['snn']['macro_trial_mae'],
                  'test_mae': test['models']['snn']['mae'], 'test_rmse': test['models']['snn']['rmse']}
        records.append(record)
        lines.append(f"| [{folder.name}]({folder.name}/RESULTS.md) | {cfg['forecast_seconds']:g} | {record['input_schema']} / {cfg['feature_set']} | {cfg['learning_rate']:g} | {test['selected_epoch']} | {record['validation_macro_trial_mae']:.4f} | {record['test_mae']:.4f} | {record['test_rmse']:.4f} |")
    lines += ['', 'These are first-prototype comparisons, not a broad hyperparameter search or multi-seed significance study.',
              'telemetry_v2 excludes all ice features: raw-only has 12 channels plus missingness indicators; all has 40 features. Archived legacy_ice_v1 runs used 14 channels, including ice. These are different input contracts.',
              'Use validation results for subsequent configuration choices; held-out test results describe generalization.']
    eligible = [r for r in records if r['input_schema'] == 'telemetry_v2' and r['forecast_seconds'] == 10.]
    if eligible:
        selected = min(eligible, key=lambda r:r['validation_macro_trial_mae'])
        (root/'selected_run.json').write_text(json.dumps({'run': selected['run'], 'input_schema': 'telemetry_v2',
            'selection': 'Lowest validation macro-trial MAE among completed telemetry_v2 10-second SNN runs only.',
            'eligible_runs': [r['run'] for r in eligible]}, indent=2)+'\n')
        lines += ['', f"Current telemetry-only default: [{selected['run']}]({selected['run']}/RESULTS.md). Legacy runs remain available for explicit replay and are excluded from default selection."]
    (root/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    (root/'comparison.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(records, indent=2), flush=True)
