"""Build Divided Inputs artifacts without altering source rows or environmental data."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .data import read_segments, source_hash
from .events import detect_segments, event_intervals, policy_dict, transition_policy


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_table(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_divided_input(source, cfg, output, overwrite=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=overwrite)
    source = Path(source).resolve()
    segments = read_segments(source, dict(cfg, clean_intervals_path=None))
    events, audit = detect_segments(segments)
    train_ids = sorted({s.trial for s in segments}-set(cfg['validation_trials'])-set(cfg['test_trials']))
    guard, profiles = transition_policy(segments, events, train_ids)
    intervals = event_intervals(events, guard, profiles)
    write_json(output/'intervals.json', intervals)
    write_json(output/'events.json', events)
    write_table(output/'event_catalog.csv', events)
    write_table(output/'candidate_audit.csv', audit)
    write_table(output/'oscillation_audit.csv', profiles)
    by_trial = defaultdict(list)
    for interval in intervals:
        by_trial[interval['parent_trial_id']].append(interval)
    metadata = ['parent_trial_id', 'clean_segment_id', 'training_eligible',
                'exclusion_reason', 'event_id', 'event_time', 'detected_time']
    counts = Counter()
    serial = Counter()
    active_key = {}
    with source.open(newline='') as input_stream, (output/'marked_dataset.csv').open('w', newline='') as marked_stream, (output/'clean_training_view.csv').open('w', newline='') as clean_stream:
        reader = csv.DictReader(input_stream)
        fields = reader.fieldnames+metadata
        marked = csv.DictWriter(marked_stream, fields)
        clean = csv.DictWriter(clean_stream, fields)
        marked.writeheader()
        clean.writeheader()
        event_lookup = {e['event_id']: e for e in events}
        for row in reader:
            trial, t = int(row['data_id']), float(row['timestamp'])
            interval = next((i for i in by_trial[trial] if i['start']-1e-8 <= t <= i['end']+1e-8), None)
            try:
                valid = math.isfinite(float(row['current'])) and bool(row['current_segment_id'])
            except ValueError:
                valid = False
            eligible = valid and interval is None
            if eligible:
                source_key = row['current_segment_id']
                if active_key.get(trial) != source_key:
                    serial[trial] += 1
                active_key[trial] = source_key
                key = f'{trial}.{serial[trial]}'
            else:
                active_key.pop(trial, None)
                key = ''
            event = event_lookup[interval['event_id']] if interval else None
            row.update(parent_trial_id=trial, clean_segment_id=key, training_eligible=int(eligible),
                       exclusion_reason='current_gap' if not valid else 'event_transition' if interval else '',
                       event_id=event['event_id'] if event else '',
                       event_time=event['event_time'] if event else '',
                       detected_time=event['detected_time'] if event else '')
            marked.writerow(row)
            counts['source_rows'] += 1
            if eligible:
                clean.writerow(row)
                counts['eligible_rows'] += 1
            else:
                counts['excluded_rows'] += 1
    manifest = {'source': str(source), 'source_sha256': source_hash(source), 'current_unit': 'A',
                    'detector': policy_dict(), 'transition_guard': guard,
                    'counts': dict(counts), 'confirmed_drop_count': len(events),
                    'candidate_status_counts': dict(Counter(e['status'] for e in audit)),
                    'parent_training_trials': train_ids, 'validation_trials': cfg['validation_trials'],
                    'test_trials': cfg['test_trials'], 'clean_segments_by_parent': dict(serial),
                    'environmental_columns': 'All source columns preserved verbatim in both CSV outputs; only metadata is appended.',
                    'interpretation': 'Current-drop candidates consistent with shedding, not independently camera-confirmed ice events.',
                    'detection_sampling': 'Causal 0.2 s as-of grid; event timestamps have sampling uncertainty. Detector runs on all trials; guard frequency fitted to training parents only.',
                    'eligibility': 'Retrospective preprocessing metadata, never an inference input. All descendants inherit the original parent split. Training history and complete future horizon must remain inside a clean segment.'}
    write_json(output/'manifest.json', manifest)
    document = f'''# Divided Inputs dataset

Source: `{source.name}`; current is measured in amperes (user confirmed).
The source file is unchanged. All original column values, including environmental
settings and separate ice targets, are copied verbatim. Ice is never an input.

`marked_dataset.csv` retains all {counts['source_rows']:,} rows and appends metadata.
`clean_training_view.csv` retains {counts['eligible_rows']:,} eligible rows.
Its name indicates eligibility; validation/test parents remain separately marked
by the manifest and must never be used for offline training.
Segment IDs such as `19.1` and `19.2` are strings, not new independent trials.

## Event clocks and criteria

A decrease strictly greater than 1 A relative to the previous one-second median
opens a candidate. Confirmation waits two seconds and requires at least 80% of
observations from candidate+0.5 s to confirmation to remain more than 1 A below
the pre-drop median. Requested RPM changes above max(100 RPM, 5%) or a motor
stop below 100 RPM exclude the candidate. Throttle is a command proxy only if
requested RPM is unavailable, because throttle may respond to shedding.

`event_time` is the timestamp of the largest observed downward step around
candidate initiation. `detected_time` is the first timestamp satisfying the
confirmation check. Both clocks use the causal 5 Hz grid. {len(events)} confirmed
electrical drops were found; these are not automatically verified ice fractures.
All rejected candidate reasons are retained in `candidate_audit.csv`.

## Oscillation check

Jitter is the 5–95 percentile span after removing a local linear trend. Compare
the 3 s before the event and the 3 s after confirmation with a baseline at
event−15 to event−5 s. Flag jitter above max(0.5 A, three times baseline).
Enable a targeted 3 s pre-guard once at least three training events exhibit
pre-event oscillation; apply it only to individually flagged events. A shared
post-guard requires at least three training events and a flagged fraction >=50%.
Held-out trials do not choose these rules. Selected pre-guard:
{guard['pre_guard_seconds']} s for flagged events only; post-guard:
{guard['post_guard_seconds']} s. The interval event_time−applicable_pre_guard through
detected_time+post_guard is excluded inclusively from clean training.

## Files and limitations

- `event_catalog.csv` / `events.json`: retrospective event ledger and clocks.
- `candidate_audit.csv`: accepted and rejected causal candidates.
- `oscillation_audit.csv`: measured pre/post jitter for each confirmed drop.
- `intervals.json`: exact exclusions consumed by the training preparation code.
- `manifest.json`: thresholds, splits, counts, source hash and preservation contract.

Training preparation uses the original CSV and these intervals so 0.2 s sampling
and missing-current gaps remain identical to the previous implementation.
The live detector receives telemetry only. Retrospective interval labels and
future event clocks are never supplied to the live network or Forecast Feedback update logic.
Gradual load decreases, partial shedding below 1 A and unobserved physical causes
can escape or confound this detector. Original trajectories remain the evaluation
targets, including every retained drop and transition.
'''
    (output/'DATASET.md').write_text(document)
    print(json.dumps({'divided_input': str(output), 'events': len(events), 'rows': dict(counts), 'guard': guard}), flush=True)
    return manifest
