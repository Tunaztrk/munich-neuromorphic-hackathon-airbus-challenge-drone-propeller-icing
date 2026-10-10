# Telemetry-based icing forecasting prototype

A software prototype combining a spiking current forecaster, an ice-area observer,
and explicit threshold-based warning logic.

This source-only edition contains no experimental measurements, trained weights,
fitted normalization statistics, experiment outputs, plots, notebooks or original
Git history. Its tests construct artificial signals. No experimental performance
claims are made in this edition.

## Method

- **Sliding Window:** up to 30 seconds of causal telemetry history processed by
  two recurrent leaky integrate-and-fire layers, with 48 neurons per layer.
- **Divided Inputs:** training segments separated around persistent current drops.
  This focuses ordinary trajectory learning; it does not establish fracture anticipation.
- **Forecast Feedback:** delayed true measurements supervise constrained updates
  to the final linear decoder. The spiking backbone stays fixed.
- **Ice observer:** a separate three-member GRU ensemble, with 16 hidden units per
  member, estimates area from telemetry and history. Camera measurements are
  supervision targets, not live inputs.
- **Decision:** current >14 A OR estimated ice area >800 mm², now or within the
  next 10 seconds. These are example project criteria, not certified safety limits.

The current network produces 50 values at 0.2-second spacing. Ice forecasting
uses the predicted current while holding other telemetry at its latest value.
The software provides warnings, not flight-control actions.

## Setup and synthetic tests

From this edition's root, using Python 3.12 or newer:

```sh
python3 -m venv model/.venv
model/.venv/bin/python -m pip install -r model/requirements.lock.txt
model/.venv/bin/python -m unittest discover -s model/tests -v
```

Tests run without the experimental dataset or pretrained weights. The historical experiment comparison routines require independently generated
local runs; experiment reproduction artifacts are not bundled. The example
configuration uses placeholder validation/test IDs and an absent private dataset
path; replace these locally if you have authorized training data.

## Inference interface

```python
from model.uav_snn.danger import IcingDangerForecaster
from model.uav_snn.data import RAW_COLUMNS

# Supply your own independently trained or authorized private artifacts.
forecaster = IcingDangerForecaster(
    "path/to/private/current_checkpoint.pt",
    "path/to/private/ice_observer_bundle",
)
# result = forecaster.update(timestamp_seconds, telemetry_in_RAW_COLUMNS_order)
```

Call with a causally aligned sensor tick every 0.2 seconds. The ice observer
issues at 1 Hz after 30 valid one-second observations. Reset between independent
trials and across observation gaps. Missing warmup information is unavailable,
not Safe.

## Publication boundary

Keep datasets and trained artifacts outside this directory. Create a new Git
repository for this edition; do not copy `.git` from an experimental repository.
Ignore rules do not remove information already stored in Git history.

The repository uses an allowlist so only the reviewed source layout can be
tracked by normal `git add`. Avoid force-adding experimental files. Embedded
figures and notebook outputs can disclose data even without a CSV file.

Publication is pending the owner's confirmation of visibility, GitHub destination,
and permission to distribute the source code. No open-source license has been
assigned automatically.
