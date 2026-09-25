"""Physical-time algebra; synthetic owner events are not model qualification."""
import unittest

from faaslora.metrics.metrics_collector import PhysicalGPULedger


class PhysicalLifecycle(unittest.TestCase):
    def setUp(self):
        self.ledger = PhysicalGPULedger(clock_id='clock', deployment_notice_s=0.)

    def acquire(self, lease='runtime', at=0., devices=('GPU-A',), owner='owner'):
        self.ledger.acquire(lease_id=lease, owner_id=owner, gpu_uuids=devices,
                            at=at, clock_id='clock', evidence_id='allocation:'+lease)

    def release(self, lease='runtime', at=10., owner='owner'):
        self.ledger.release(lease_id=lease, owner_id=owner, at=at,
                            clock_id='clock', evidence_id='native-return:'+lease)

    def report(self, **kw):
        args = dict(observed_until_s=10., arrival_start_s=2., arrival_end_s=5.,
                    last_terminal_s=8., n_plan=4, n_terminal=4, n_correct=4)
        return self.ledger.summarize(**(args | kw))

    def test_startup_and_preparation_overlap_is_ten_not_sixteen(self):
        self.acquire('startup', at=0.)
        self.acquire('preparation', at=2.)
        self.release('preparation', at=8.)
        self.release('startup', at=10.)
        report = self.report()
        self.assertEqual(report['gpu_seconds'], 10.)
        self.assertEqual(report['gpu_seconds_per_correct_request'], 2.5)
        self.assertEqual(report['window_gpu_seconds'],
                         dict(pre_arrival=2., arrival=3., drain=3., cleanup=2.))

    def test_tp_and_same_card_processes_count_physical_union(self):
        self.acquire('tensor-parallel', at=0., devices=('GPU-A','GPU-B'))
        self.acquire('same-card-helper', at=1., devices=('GPU-B',))
        self.release('tensor-parallel', at=8.)
        self.release('same-card-helper', at=10.)
        report = self.report()
        self.assertEqual(report['gpu_seconds'], 18.)
        self.assertEqual(report['device_intervals'], {'GPU-A':[(0.,8.)], 'GPU-B':[(0.,10.)]})

    def test_reallocation_preserves_unallocated_gap(self):
        self.acquire('first')
        self.release('first', at=2.)
        self.acquire('second', at=5.)
        self.release('second', at=10.)
        self.assertEqual(self.report()['gpu_seconds'], 7.)

    def test_cpu_only_preparation_is_not_gpu_allocation(self):
        self.acquire(at=4.)
        self.release(at=10.)
        self.assertEqual(self.report()['gpu_seconds'], 6.)

    def test_worker_or_daemon_not_released_stays_censored(self):
        self.acquire()
        report = self.report()
        self.assertEqual(report['gpu_seconds_observed'], 10.)
        self.assertIsNone(report['gpu_seconds'])
        self.assertIsNone(report['gpu_seconds_per_correct_request'])
        self.assertFalse(report['measurement_complete'])
        self.assertEqual(report['open_lease_ids'], ['runtime'])

    def test_failure_keeps_full_offered_denominator(self):
        self.acquire()
        self.release(at=4.)
        report = self.report(observed_until_s=4., n_terminal=1, n_correct=0,
                             last_terminal_s=None)
        self.assertEqual(report['n_plan'], 4)
        self.assertEqual(report['gpu_seconds_per_offered_observed'], 1.)
        self.assertIsNone(report['gpu_seconds_per_correct_request'])
        self.assertFalse(report['eligible_correctness'])
        self.assertEqual(report['window_gpu_seconds'],
                         dict(pre_arrival=2., arrival=2., drain=0., cleanup=0.))

    def test_all_failed_zero_correct_cannot_produce_finite_optimum(self):
        self.acquire()
        self.release()
        report = self.report(n_correct=0)
        self.assertEqual(report['gpu_seconds'], 10.)
        self.assertIsNone(report['gpu_seconds_per_correct_request'])
        self.assertFalse(report['eligible_correctness'])

    def test_missing_allocation_not_a_zero_cost_win(self):
        with self.assertRaisesRegex(ValueError, 'allocation evidence'):
            self.report()

    def test_clock_missing_evidence_and_event_order_fail_closed(self):
        for kwargs in (dict(clock_id='foreign'), dict(evidence_id=''), dict(at=-1.)):
            args = dict(lease_id='x', owner_id='o', gpu_uuids=['GPU-A'], at=1.,
                        clock_id='clock', evidence_id='proof') | kwargs
            with self.assertRaises(ValueError):
                self.ledger.acquire(**args)
        self.acquire(at=2.)
        with self.assertRaises(ValueError):
            self.release(at=1.)

    def test_duplicate_foreign_and_unproven_release_reject(self):
        self.acquire()
        with self.assertRaises(ValueError):
            self.acquire()
        with self.assertRaises(ValueError):
            self.release(owner='not-owner')
        self.release()
        with self.assertRaises(ValueError):
            self.release()
        with self.assertRaises(ValueError):
            self.acquire(at=11.)

    def test_logical_indices_and_duplicate_devices_reject(self):
        for devices in ([], ['0'], ['GPU-A','GPU-A'], ['MIG-device']):
            with self.assertRaises(ValueError):
                self.acquire(devices=devices)

    def test_invalid_population_and_time_boundaries_reject(self):
        self.acquire()
        self.release()
        for kw in (dict(n_correct=5), dict(n_terminal=3), dict(last_terminal_s=None),
                   dict(arrival_end_s=1.), dict(observed_until_s=9.),
                   dict(last_terminal_s=float('nan')), dict(n_plan=0)):
            with self.assertRaises(ValueError):
                self.report(**kw)


if __name__ == '__main__':
    unittest.main()
