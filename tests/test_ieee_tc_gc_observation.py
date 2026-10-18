import gc
import unittest
from unittest.mock import patch
from scripts.ieee_tc.gc_observation import GCObservation


class GCObservationTests(unittest.TestCase):
    def test_actual_tiny_collection_and_policy_restored(self):
        before = (gc.isenabled(), gc.get_threshold(), gc.get_freeze_count(), list(gc.callbacks))
        with GCObservation() as observer:
            observer.mark('test_only', 0, 'tiny_cycle')
            cycle = []
            cycle.append(cycle)
            del cycle
            gc.collect(0)  # Unit witness ONLY; no forced GC in native diagnostic.
        result = observer.report()
        self.assertTrue(result['gc_policy_unchanged'])
        self.assertTrue(any(row['stage']=='test_only' for row in result['events']))
        self.assertTrue(all(row['wall_ns']>=0 for row in result['events']))
        self.assertEqual(before,(gc.isenabled(),gc.get_threshold(),gc.get_freeze_count(),list(gc.callbacks)))

    def test_callback_removed_on_exception(self):
        before = list(gc.callbacks)
        with self.assertRaisesRegex(ValueError, 'deliberate'):
            with GCObservation():
                raise ValueError('deliberate')
        self.assertEqual(before, gc.callbacks)

    def test_invalid_or_truncated_events_fail_not_silent(self):
        for limit, phases in ((1, ['stop']), (0, ['start','stop'])):
            observer = GCObservation(limit)
            with observer:
                for phase in phases:
                    observer._callback(phase,dict(generation=0,collected=0,uncollectable=0))
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                observer.report()

    def test_active_observer_cannot_publish_partial_result(self):
        with GCObservation() as observer:
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                observer.report()


if __name__ == '__main__':
    unittest.main()
