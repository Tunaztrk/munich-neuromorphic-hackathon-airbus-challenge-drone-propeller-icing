"""Check physical time alignment and exclusion of camera/future information."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model.uav_snn.ice_area import IceAreaEstimator, from_history, telemetry_at


class IceAreaTests(unittest.TestCase):
    def records(self):
        t = np.arange(0, 80, .1)
        return np.column_stack([t, np.full(len(t), -10.), np.ones(len(t)),
                                4 + .01*t, np.full(len(t), 3000.),
                                np.full(len(t), 3000.), 20*t, 200*t])

    def test_camera_and_future_cannot_change_features(self):
        records = self.records()
        query = np.array([35., 40.])
        before, valid, _ = telemetry_at(records, query)
        changed = records.copy()
        changed[:, 6:] = 1e9  # Camera values, including prior area, are never inputs.
        changed[changed[:, 0] > 40.001, 1:6] = -1e6
        after, after_valid, _ = telemetry_at(changed, query)
        np.testing.assert_array_equal(valid, after_valid)
        np.testing.assert_array_equal(before, after)
        self.assertTrue(valid.all())

    def test_seconds_derivative_and_gaps(self):
        records = self.records()
        features, valid, _ = telemetry_at(records, np.array([40.]))
        self.assertTrue(valid[0])
        self.assertAlmostEqual(features[0, 3], .01, places=9)
        self.assertAlmostEqual(features[0, 5], .01, places=9)
        gap = records[(records[:, 0] < 29) | (records[:, 0] > 32)]
        _, valid_gap, _ = telemetry_at(gap, np.array([40.]))
        self.assertFalse(valid_gap[0])

    def test_future_current_path_uses_correct_derivative(self):
        # At +10 s a 30-s window has 20 historical and 10 predicted samples.
        history = np.full((1, 30), 5.)
        forecast = 5 + .2*np.arange(1, 11)[None, :]
        future_history = np.column_stack([history[:, 10:], forecast])
        features = from_history(future_history, [-10], [1], [3000])
        self.assertAlmostEqual(features[0, 2], 7.)
        self.assertAlmostEqual(features[0, 3], .2)

    def test_serialization_and_fixed_prediction_state(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(90, 8))
        target = 1000 + 100*x[:, 2] - 20*x[:, 0]
        trial = np.repeat([1, 2, 3], 30)
        model = IceAreaEstimator({'features': 'core', 'kind': 'linear', 'alpha': .0001})
        model.fit(x, target, trial)
        before = model.coefficients.copy()
        prediction = model.predict(x)
        self.assertLess(np.abs(prediction-target).mean(), 1.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.npz'
            model.save(path, {'test': True})
            np.testing.assert_array_equal(prediction, IceAreaEstimator.load(path).predict(x))
        np.testing.assert_array_equal(before, model.coefficients)

    def test_current_rpm_needs_no_other_inputs(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(100, 8))
        target = 1000 + 100*x[:, 2] - 30*x[:, 4]
        model = IceAreaEstimator({'features': 'current_rpm', 'kind': 'linear', 'alpha': .001})
        model.fit(x, target, np.repeat(np.arange(5), 20))
        expected = model.predict(x)
        x[:, [0, 1, 3, 5, 6, 7]] = np.nan
        np.testing.assert_array_equal(expected, model.predict(x))
        np.testing.assert_array_equal(expected, model.predict_current_rpm(x[:, 2], x[:, 4]))
        # A single-row BLAS call may round differently from a full batch.
        np.testing.assert_allclose(expected[:1], model.predict_current_rpm(x[:1, 2], x[0, 4]),
                                   atol=1e-10, rtol=0)

    def test_two_input_api_rejects_four_input_model(self):
        model = IceAreaEstimator({'features': 'core', 'kind': 'linear', 'alpha': .001})
        with self.assertRaises(ValueError):
            model.predict_current_rpm(10, 3500)


if __name__ == '__main__':
    unittest.main()
