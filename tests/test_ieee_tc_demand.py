"""No-GPU exact contract tests for the arrival fraction in IEEE Eq. (4)."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from faaslora.experiment.hotness_tracker import HotnessTracker
from faaslora.experiment.experiment_stack import ExperimentStack


class ExactDemandTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.tracker = HotnessTracker(None, window_seconds=10, clock=lambda: self.now)

    def test_distribution_not_per_adapter_ewma_or_doubled_probability(self):
        for adapter in ['a', 'a', 'b']:
            self.tracker.record_arrival(adapter)
        snap = self.tracker.snapshot()
        self.assertEqual(snap.total_arrivals, 3)
        self.assertEqual(snap.fraction('a'), 2/3)
        self.assertEqual(snap.fraction('b'), 1/3)
        self.assertEqual(sum(snap.fraction(a) for a in snap.counts), 1)

    def test_left_open_right_closed_and_idle_expiry(self):
        self.tracker.record_arrival('a')
        self.now = 105
        self.tracker.record_arrival('b')
        self.now = 110
        self.assertEqual(self.tracker.get_hotness('a'), 0)
        self.assertEqual(self.tracker.get_hotness('b'), 1)
        self.now = 115
        self.assertEqual(self.tracker.get_hotness('b'), 0)
        self.assertEqual(self.tracker.get_top_k(3), [])
        self.assertEqual(self.tracker.snapshot().total_arrivals, 0)

    def test_late_startup_attachment_keeps_original_arrival_age(self):
        self.now = 120
        self.tracker.record_arrival('expired', observed_at=100)
        self.tracker.record_arrival('left-boundary', observed_at=110)
        self.tracker.record_arrival('live', observed_at=111)
        snap = self.tracker.snapshot()
        self.assertEqual(dict(snap.counts), {'live':1})
        self.now = 121
        self.assertEqual(self.tracker.snapshot().total_arrivals, 0)

    def test_delayed_observation_must_be_ordered_and_not_future(self):
        self.tracker.record_arrival('a', observed_at=95)
        for value in (94, 101, float('nan')):
            with self.assertRaisesRegex(ValueError, 'not in the future'):
                self.tracker.record_arrival('b', observed_at=value)
        self.assertEqual(dict(self.tracker.snapshot().counts), {'a':1})

    def test_no_silent_5000_arrival_truncation(self):
        for i in range(6000):
            self.tracker.record_arrival('a' if i < 3000 else 'b')
        self.assertEqual(self.tracker.snapshot().total_arrivals, 6000)
        self.assertEqual(self.tracker.get_hotness('a'), .5)

    def test_frozen_epoch_not_changed_by_future_arrivals(self):
        self.tracker.record_arrival('a')
        snap = self.tracker.snapshot()
        self.tracker.record_arrival('b')
        self.assertEqual(snap.fraction('a'), 1)
        with self.assertRaises(TypeError):
            snap.counts['a'] = 99

    def test_no_stale_registry_prior(self):
        stack = ExperimentStack.__new__(ExperimentStack)
        stack.hotness_tracker = self.tracker
        stack.registry = SimpleNamespace(get_artifact=lambda _: SimpleNamespace(hotness_score=1))
        self.assertEqual(stack._online_artifact_hotness('a'), 0)

    def test_resolved_access_does_not_count_second_arrival(self):
        stack = ExperimentStack.__new__(ExperimentStack)
        stack.hotness_tracker = self.tracker
        measured = []
        stack.registry = SimpleNamespace(update_access_stats=lambda *a, **kw: measured.append((a, kw)))
        stack._schedule_host_promotion_from_nvme = lambda _: None
        stack.record_arrival('a')
        stack.record_access('a', load_time_ms=7, hit=False)
        stack.record_access('a', load_time_ms=2, hit=True)
        self.assertEqual(self.tracker.snapshot().total_arrivals, 1)
        self.assertEqual(len(measured), 2)

    def test_arrival_hook_precedes_admission_and_not_completion(self):
        source = Path(__file__).resolve().parents[1] / 'scripts/run_all_experiments.py'
        tree = ast.parse(source.read_text())
        run_one = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == 'run_one')
        calls = [(n.func.attr, n.lineno) for n in ast.walk(run_one)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        arrivals = [line for name, line in calls if name == 'record_arrival']
        self.assertEqual(len(arrivals), 1)
        self.assertLess(arrivals[0], next(line for name, line in calls if name == '_acquire_dispatch_admission'))

    def test_invalid_clock_and_window_rejected(self):
        self.tracker.record_arrival('a')
        self.now = 99
        with self.assertRaises(ValueError):
            self.tracker.snapshot()
        for value in [0, -1, float('inf'), float('nan')]:
            with self.assertRaises(ValueError):
                HotnessTracker(None, window_seconds=value)

    def test_ties_have_stable_adapter_order(self):
        self.tracker.record_arrival('b')
        self.tracker.record_arrival('a')
        self.assertEqual(self.tracker.get_top_k(1), ['a'])


if __name__ == '__main__':
    unittest.main()
