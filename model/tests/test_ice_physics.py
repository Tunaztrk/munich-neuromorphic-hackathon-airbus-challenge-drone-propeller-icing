"""Physical scaling, recurrent replay, future-state isolation and danger semantics."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from model.uav_snn.danger import danger_output
from model.uav_snn.physics_ice import (
    FEATURE_GROUPS,
    CleanPowerReference,
    IceGRU,
    IceObserver,
    PhysicalFeatureBuilder,
)


class PhysicsIceTests(unittest.TestCase):
    def raw(self, t):
        rpm = 2900.+2*t
        return np.array([4.+.01*t, rpm, 3000., 45., 20., -10., 1.,
                         np.nan, np.nan, -.2, -.3, -.25])

    def reference(self):
        raw = np.stack([self.raw(t) for t in range(50)])
        return CleanPowerReference().fit(raw, np.repeat([1, 2], 25))

    def builder(self):
        builder = PhysicalFeatureBuilder(self.reference())
        for t in range(35):
            builder.step(t, self.raw(t))
        return builder

    def test_rotational_cube_normalization(self):
        builder = PhysicalFeatureBuilder(self.reference())
        for t in range(35):
            raw = self.raw(t)
            raw[0] = 5.*(raw[1]/3000.)**3
            feature = builder.step(t, raw)
        self.assertAlmostEqual(feature[9], 100., places=9)
        self.assertAlmostEqual(feature[10], 0., places=9)
        self.assertAlmostEqual(feature[11], 0., places=9)

    def test_rollout_isolated_from_live_history(self):
        builder = self.builder()
        before_time, before_exposure = builder.last_time, builder.exposure.copy()
        before_raw = builder.last_raw.copy()
        path = before_raw[0]+.2*np.arange(1, 11)
        future = builder.rollout(path)
        self.assertAlmostEqual(future[-1, 3], .2, places=10)
        self.assertEqual(builder.last_time, before_time)
        np.testing.assert_array_equal(builder.exposure, before_exposure)
        np.testing.assert_array_equal(builder.last_raw, before_raw)
        with self.assertRaises(ValueError):
            builder.step(before_time+2., before_raw)
        with self.assertRaises(ValueError):
            builder.step(before_time+1., np.r_[before_raw, 800.])

    def test_serialized_neural_observer_matches_sequence_and_keeps_state(self):
        torch.manual_seed(11)
        indices = FEATURE_GROUPS['normalized_history']
        ref = self.reference()
        builder = PhysicalFeatureBuilder(ref)
        rows = []
        for t in range(40):
            feature = builder.step(t, self.raw(t))
            if feature is not None:
                rows.append(feature)
        x = np.stack(rows)
        mean = x[:, indices].mean(0)
        scale = np.maximum(x[:, indices].std(0), 1.)
        model = IceGRU(len(indices), 4).eval()
        cp = {'indices': indices, 'mean': mean.tolist(), 'scale': scale.tolist(),
              'hidden_size': 4, 'state_dict': model.state_dict()}
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            torch.save(cp, folder/'ice.pt')
            (folder/'observer.json').write_text(json.dumps({'kind': 'gru',
                'clean_reference': ref.payload(), 'checkpoints': ['ice.pt']}))
            observer = IceObserver(folder)
            predicted = [observer.update(t, self.raw(t)) for t in range(40)]
            with torch.no_grad():
                expected, state = model(torch.as_tensor(((x[:, indices]-mean)/scale).astype(np.float32))[None])
            np.testing.assert_allclose(predicted[29:], expected[0].numpy()*1000., atol=2e-4, rtol=1e-6)
            state_before = observer.states[0].clone()
            forecast = observer.forecast(np.full(10, 4.4))
            future_x = observer.builder.rollout(np.full(10, 4.4))
            with torch.no_grad():
                expected_future, _ = model(torch.as_tensor(((future_x[:, indices]-mean)/scale).astype(np.float32))[None], state)
            np.testing.assert_allclose(forecast, expected_future[0].numpy()*1000., atol=2e-4, rtol=1e-6)
            torch.testing.assert_close(observer.states[0], state_before)

    def test_danger_uses_both_thresholds_and_the_full_current_path(self):
        current = np.full(50, 14.)
        area = np.full(10, 800.)
        result = danger_output(0, 14., 800., current, area)
        self.assertEqual(result['status'], 'Safe')
        current[1] = 14.1  # A crossing between the one-second output timestamps.
        area[7] = 801.
        result = danger_output(0, 5., 200., current, area)
        self.assertTrue(result['danger_forecast'][0])
        self.assertTrue(result['danger_forecast'][7])
        self.assertEqual(result['status'], 'Danger')
        self.assertEqual(result['status_now'], 'Safe')
        with self.assertRaises(ValueError):
            danger_output(0, 5., np.nan, current, area)


if __name__ == '__main__':
    unittest.main()
