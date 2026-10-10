"""Command-line entry point for preparation, training, evaluation, and replay."""
import argparse
import json
from pathlib import Path

from .data import MODEL_ROOT, load_config, prepare, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'replay', 'report', 'compare'])
    parser.add_argument('--config', default=str(MODEL_ROOT/'config'/'default.json'))
    selected = MODEL_ROOT/'runs'/'selected_run.json'
    parser.add_argument('--run-name', default=json.loads(selected.read_text())['run'] if selected.exists() else 'telemetry_tau10')
    parser.add_argument('--trial', type=int, default=28)
    parser.add_argument('--tau', type=float)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--feature-set', choices=['all', 'raw', 'no_camera'])
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in {'.', '..'}:
        parser.error('--run-name must be a folder name without path components.')
    cfg = load_config(args.config)
    for name, key in [('tau', 'forecast_seconds'), ('learning_rate', 'learning_rate'),
                      ('epochs', 'epochs'), ('feature_set', 'feature_set')]:
        value = getattr(args, name)
        if value is not None:
            cfg[key] = value
    validate_config(cfg)
    dt = cfg['sample_interval_seconds']
    if cfg['forecast_seconds'] <= 0 or abs(cfg['forecast_seconds']/dt-round(cfg['forecast_seconds']/dt)) > 1e-6:
        parser.error('--tau must be a positive multiple of the sample interval.')
    if args.command == 'prepare':
        prepare(cfg)
    elif args.command == 'train':
        from .experiment import train
        train(cfg, args.run_name)
    elif args.command == 'replay':
        from .stream import replay
        replay(MODEL_ROOT/'runs'/args.run_name, args.trial)
    elif args.command == 'report':
        from .report import create_report
        create_report(MODEL_ROOT/'runs'/args.run_name)
    elif args.command == 'compare':
        from .report import compare_runs
        compare_runs()


if __name__ == '__main__':
    main()
