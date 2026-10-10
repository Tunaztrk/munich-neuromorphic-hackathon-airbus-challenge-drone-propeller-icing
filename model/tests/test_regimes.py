"""Causal event clocks, independent windows and delayed head-only learning."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from model.uav_snn.data import MODEL_ROOT, FeatureBuilder, Segment
from model.uav_snn.events import DropDetector, clean_ranges
from model.uav_snn.feedback import DelayedHead
from model.uav_snn.network import CurrentSNN


def telemetry(current, target=3000.):
    row = np.ones(12)
    row[0], row[1], row[2], row[3] = current, 3000., target, 40.
    return row


class RegimeChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(1)
        self.cfg = json.loads((MODEL_ROOT/'config'/'default.json').read_text())

    def series(self, transient=False, command=False):
        detector = DropDetector()
        events = []
        for j in range(60):
            low = j >= 20 and (not transient or j < 22)
            result = detector.update(j*.2, telemetry(5. if low else 8., 2000. if command and j >= 20 else 3000.))
            if result:
                events.append(result)
        return events

    def test_drop_clock_and_confirmation_are_causal(self):
        event = next(e for e in self.series() if e['status'] == 'confirmed_drop')
        self.assertAlmostEqual(event['event_time'], 4.)
        self.assertAlmostEqual(event['detected_time'], 6.)
        self.assertGreaterEqual(event['detected_time'], event['candidate_time']+2.)
        detector = DropDetector()
        for j in range(30):
            self.assertIsNone(detector.update(j*.2, telemetry(8. if j < 20 else 5.)))
        self.assertEqual(detector.update(6., telemetry(5.))['status'], 'confirmed_drop')

    def test_transient_and_command_drop_are_not_shedding_labels(self):
        self.assertFalse(any(e['status'] == 'confirmed_drop' for e in self.series(transient=True)))
        self.assertFalse(any(e['status'] == 'confirmed_drop' for e in self.series(command=True)))
        self.assertTrue(any(e['status'] == 'command_change' for e in self.series(command=True)))

    def test_clean_segments_preserve_parent_and_cannot_cross_event(self):
        t = np.arange(100)*.2
        segment = Segment(19, '19:1', t, np.stack([telemetry(8.) for _ in t]))
        intervals = [{'parent_trial_id': 19, 'start': 8., 'end': 10.}]
        ranges = clean_ranges(segment, intervals)
        self.assertEqual(len(ranges), 2)
        horizon = 10
        for a, b in ranges:
            for o in range(a, b-horizon):
                self.assertFalse(np.any((t[o:o+1+horizon] >= 8.) & (t[o:o+1+horizon] <= 10.)))
        np.testing.assert_array_equal(segment.raw[:, 0], 8.)
        self.assertEqual(segment.trial, 19)

    def test_window_padding_and_overlap_have_independent_states(self):
        model = CurrentSNN(40, 50, self.cfg)
        with torch.no_grad():
            model.decoder[-1].weight.normal_(0., .05)
            x = torch.randn(2, 12, 40)
            z, anchor, _ = model.window_encode(x, torch.tensor([5, 12]))
            batch = anchor+model.decoder[-1](z)
            for lane, length in [(0, 5), (1, 12)]:
                expected = model(x[lane:lane+1, :length])[0][0, -1]
                torch.testing.assert_close(batch[lane], expected)
            altered = x.clone()
            altered[0, 5:] = 1e5
            z2, a2, _ = model.window_encode(altered, torch.tensor([5, 12]))
            torch.testing.assert_close(batch, a2+model.decoder[-1](z2))
            previous = model.window_encode(x[:, :8])
            independent = model.window_encode(x[:, 4:12])
            repeat = model.window_encode(x[:, 4:12])
            torch.testing.assert_close(independent[0], repeat[0])
            self.assertEqual(previous[0].shape, independent[0].shape)

    def test_feedback_waits_for_targets_and_only_updates_its_head(self):
        model = CurrentSNN(40, 3, self.cfg)
        original = copy.deepcopy(model.state_dict())
        adapter = DelayedHead(model.decoder[-1], 3, .2, 0., 1., learning_rate=.01, minimum_examples=1, update_seconds=.2)
        adapter.observe(0., 1.)
        issued, version = adapter.issue(0., np.ones(48), 1., 0.)
        saved = issued.copy()
        for t in [.2, .4]:
            adapter.observe(t, 2.)
            self.assertFalse(adapter.maybe_update(t))
        adapter.observe(.6, 2.)
        self.assertTrue(adapter.maybe_update(.6))
        self.assertEqual(version, 0)
        np.testing.assert_array_equal(issued, saved)
        self.assertGreater(adapter.predict(np.ones(48), 1.)[0], issued[0])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name])
        self.assertEqual(sum(p.numel() for p in adapter.head.parameters()), 147)
        self.assertLessEqual(adapter.update_log[0]['maximum_target_time'], .6+1e-8)

    def test_feedback_event_flushes_stale_predictions_and_restores_head(self):
        head = torch.nn.Linear(2, 2)
        adapter = DelayedHead(head, 2, .2, 0., 1., learning_rate=.01, minimum_examples=1)
        adapter.observe(0., 1.)
        adapter.issue(0., np.ones(2), 1., 0.)
        adapter.observe(.2, 1.)
        adapter.event(.2)
        self.assertEqual(len(adapter.pending), 0)
        self.assertEqual(len(adapter.recent), 0)
        for name, p in adapter.head.state_dict().items():
            torch.testing.assert_close(p, head.state_dict()[name])
        adapter.observe(.4, 1.)
        adapter.issue(.4, np.ones(2), 1., .2)
        adapter.observe(.6, 1.)
        adapter.observe(.8, 1.)
        self.assertFalse(adapter.maybe_update(.8))

    def test_future_values_cannot_revise_issued_forecasts(self):
        head = torch.nn.Linear(2, 2)
        def replay(future):
            adapter = DelayedHead(head, 2, .2, 0., 1., learning_rate=.01, minimum_examples=1)
            results = []
            for j in range(8):
                adapter.observe(j*.2, 1. if j < 4 else future)
                results.append(adapter.issue(j*.2, np.ones(2), 1., 0.)[0])
                adapter.maybe_update(j*.2)
            return np.stack(results)
        np.testing.assert_array_equal(replay(2.)[:5], replay(100.)[:5])

    def test_live_runtime_matches_frozen_window_and_ignores_future(self):
        from model.uav_snn.regime_stream import RegimeForecaster
        model = CurrentSNN(40, 50, self.cfg)
        cfg = dict(self.cfg, inference_mode='window', window_seconds=2.)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'cp.pt'
            torch.save({'state_dict': model.state_dict(), 'config': cfg, 'feature_names': ['current']+[f'x{i}' for i in range(39)],
                            'imputer': [0.]*40, 'center': [0.]*40, 'scale': [1.]*40, 'target_mean': 0., 'target_scale': 1.,
                            'model_family': 'regime_v1', 'event_guard': {'post_guard_seconds': 0.}}, path)
            runtime = RegimeForecaster(path)
            predictions = [runtime.update(j*.2, telemetry(8.)) for j in range(20)]
            self.assertTrue(all(p.shape == (50,) for p in predictions))
            self.assertEqual(len(runtime.prediction_ledger), 20)
            saved = runtime.prediction_ledger[10]['forecast'].copy()
            runtime.update(4., telemetry(100.))
            np.testing.assert_array_equal(runtime.prediction_ledger[10]['forecast'], saved)
            x = torch.from_numpy(np.stack([x for _, x in list(runtime.context)])).unsqueeze(0)
            with torch.no_grad():
                z, anchor, _ = runtime.model.window_encode(x)
                expected = anchor[0].numpy()+runtime.model.decoder[-1](z)[0].numpy()
            np.testing.assert_allclose(runtime.prediction_ledger[-1]['forecast'], expected, atol=1e-5)

    def test_cached_adaptive_evaluation_matches_live_observation_replay(self):
        from model.uav_snn.regime_experiment import (
            causal_traces,
            forecast_arrays,
            representations,
        )
        from model.uav_snn.regime_stream import RegimeForecaster
        cfg = dict(self.cfg, inference_mode='window', window_seconds=2., test_trials=[1],
                   validation_trials=[], window_batch_size=32, feedback_learning_rate=.001)
        names = FeatureBuilder(cfg).names
        data = SimpleNamespace(cfg=cfg, feature_names=np.asarray(names), imputer=np.zeros(40), center=np.zeros(40),
                               scale=np.ones(40), horizon=50, stride=5, target_mean=0., target_scale=1.)
        model = CurrentSNN(40, 50, cfg)
        with torch.no_grad():
            model.decoder[-1].weight.normal_(0., .02)
        t = np.arange(350)*.2
        rows = np.stack([telemetry(8.+.01*j if j < 200 else 5.+.005*(j-200)) for j in range(350)])
        segment = Segment(1, '1:1', t, rows)
        guard = {'post_guard_seconds': 0.}
        traces = causal_traces([segment], data, guard)
        records, z, anchor = representations(model, data, traces, 'test', 'window', 2.)
        arrays, diagnostic = forecast_arrays(model, data, traces, records, z, anchor, .001, guard)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'cp.pt'
            torch.save({'state_dict': model.state_dict(), 'config': cfg, 'feature_names': names,
                            'imputer': [0.]*40, 'center': [0.]*40, 'scale': [1.]*40, 'target_mean': 0., 'target_scale': 1.,
                            'model_family': 'regime_v1', 'event_guard': guard}, path)
            runtime = RegimeForecaster(path)
            forecasts = [runtime.update(float(s), row) for s, row in zip(t, rows)]
            origins = [o for _, o, _ in records]
            np.testing.assert_allclose(np.asarray(forecasts)[origins], arrays['snn'], atol=2e-5)
            self.assertEqual(diagnostic['update_count'], runtime.adapter.total_updates)
            self.assertGreater(diagnostic['update_count'], 0)
            self.assertEqual(runtime.adapter.event_resets, 1)


if __name__ == '__main__':
    unittest.main()
