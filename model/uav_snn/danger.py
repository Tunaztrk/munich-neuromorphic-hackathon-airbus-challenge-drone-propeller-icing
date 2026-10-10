"""Forecast current, infer ice area and expose the specified Danger/Safe output.

The decision head is deterministic: the user-defined thresholds are not learned.
Current forecasts remain produced by the existing SNN, and camera targets never
enter inference. Ice forecasts use a separately validated neural/regression model.
"""
from __future__ import annotations

import numpy as np

from .physics_ice import IceObserver
from .stream import StreamingForecaster


def danger_output(timestamp, current_now, area_now, current_forecast, area_forecast,
                  current_threshold=14., area_threshold=800.):
    """Current is sampled at 5 Hz; area at +1..+H seconds.

    Each one-second decision includes any current crossing in that interval and
    the estimated area at its endpoint. Equality alone does not exceed a threshold.
    This function does not convert scores into calibrated probabilities.
    """
    current_forecast = np.asarray(current_forecast, float)
    area_forecast = np.asarray(area_forecast, float)
    if current_forecast.ndim != 1 or area_forecast.ndim != 1 or not len(area_forecast):
        raise ValueError('Supply nonempty one-dimensional forecast paths.')
    if len(current_forecast) != 5*len(area_forecast):
        raise ValueError('Expected five current samples per one-second area prediction.')
    if not np.isfinite(np.r_[current_now, area_now, current_forecast, area_forecast]).all():
        raise ValueError('Missing or nonfinite values cannot be classified Safe.')
    current_peak = current_forecast.reshape(-1, 5).max(-1)
    current_crossing = current_peak > current_threshold
    ice_crossing = area_forecast > area_threshold
    danger = current_crossing | ice_crossing
    danger_now = current_now > current_threshold or area_now > area_threshold
    return {
        'ready': True, 'origin_time': float(timestamp),
        'future_seconds': np.arange(1, len(area_forecast)+1, dtype=float),
        'current_forecast_a': current_forecast.copy(),
        'current_peak_per_second_a': current_peak,
        'ice_area_now_mm2': float(area_now), 'ice_area_forecast_mm2': area_forecast.copy(),
        'danger_forecast': danger, 'status_forecast': np.where(danger, 'Danger', 'Safe'),
        'current_trigger': current_crossing, 'ice_trigger': ice_crossing,
        'status_now': 'Danger' if danger_now else 'Safe',
        'status': 'Danger' if danger_now or danger.any() else 'Safe',
        'current_threshold_a': float(current_threshold), 'area_threshold_mm2': float(area_threshold),
        'decision_rule': f'Danger when measured/forecast current >{current_threshold:g} A OR estimated area >{area_threshold:g} mm²',
        'future_settings_assumption': 'All non-current telemetry held at the origin.',
    }


class IcingDangerForecaster:
    """A system-level current/area/decision API, receiving only 12 telemetry values.

    Call update at every 0.2 s sensor tick. The area observer runs at the established
    1 Hz issue cadence. Before 30 observed one-second samples, status is unavailable
    rather than Safe. Reset at a new parent or measurement gap, never just because
    the current SNN cleared its event context.
    """

    def __init__(self, current_checkpoint, ice_bundle):
        self.current = StreamingForecaster(current_checkpoint)
        if self.current.cfg.get('input_schema') != 'telemetry_v2':
            raise ValueError('Danger integration requires a camera-free telemetry_v2 checkpoint.')
        if not np.isclose(self.current.cfg['sample_interval_seconds'], .2):
            raise ValueError('This integration expects the existing 5 Hz current forecaster.')
        self.ice = IceObserver(ice_bundle)
        self.reset()

    def reset(self):
        self.current.reset()
        self.ice.reset()
        self.count = 0

    def update(self, timestamp, measurements):
        measurements = np.asarray(measurements, float)
        if measurements.shape != (12,):
            raise ValueError('Supply RAW_COLUMNS telemetry only; ice camera channels are prohibited.')
        current = self.current.update(timestamp, measurements)
        issue = self.count % 5 == 4
        self.count += 1
        if not issue:
            return None
        area = self.ice.update(timestamp, measurements)
        if area is None or current is None:
            return {'ready': False, 'origin_time': float(timestamp), 'status': None,
                    'current_forecast_a': current, 'reason': 'Ice observer needs 30 valid one-second observations.'}
        area_path = self.ice.forecast(np.asarray(current)[4::5])
        return danger_output(timestamp, measurements[0], area, current, area_path)
