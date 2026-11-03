"""Synthetic clock/epoch cases, never performance or numerical qualification."""
import unittest
from scripts.ieee_tc.audit_d389_scaleup_handoff_labels import activation_first_services


class FirstActualDispatch(unittest.TestCase):
    def epoch(self, identity='a', category='natural_scaleout'):
        return dict(activation_id=identity, target_replica=identity, category=category,
                    policy='full', clock_id='test', trigger_at=1., engine_ready_at=3.)

    def dispatch(self, rid='r1', when=5., token=6., success=True):
        return dict(request_id=rid, instance_id='a', clock_id='test',
                    dispatch_monotonic_s=when, arrival_monotonic_s=4.,
                    first_token_monotonic_s=token, success=success)

    def test_failed_first_is_not_replaced_by_later_success(self):
        report = activation_first_services([self.epoch()], [self.dispatch('later', 7., 8.),
            self.dispatch(token=None, success=False)], clock_id='test')
        first = report['records'][0]
        self.assertEqual(first['first_request']['request_id'], 'r1')
        self.assertIsNone(first['first_request_ttft_ms'])
        self.assertEqual(report['summary']['natural_scaleout']['with_first_token'], 0)

    def test_unused_initial_epoch_retained_and_times_separate(self):
        report = activation_first_services([self.epoch(), self.epoch('unused', 'initial')],
                                           [self.dispatch()], clock_id='test')
        first = report['records'][0]
        self.assertEqual(first['engine_activation_ms'], 2000.)
        self.assertEqual(first['first_request_ttft_ms'], 2000.)
        self.assertEqual(first['activation_to_first_token_ms'], 5000.)
        self.assertEqual(report['summary']['initial']['unused_rate'], 1.)

    def test_equal_clock_samples_preserve_ambiguity(self):
        report = activation_first_services([self.epoch()], [self.dispatch(),
            self.dispatch('r2')], clock_id='test')
        self.assertTrue(report['records'][0]['first_dispatch_ambiguous'])
        self.assertIsNone(report['records'][0]['first_request'])

    def test_missing_dispatch_or_mixed_clock_is_not_ignored(self):
        for changed in ({'dispatch_monotonic_s': None}, {'clock_id': 'other'},
                        {'first_token_monotonic_s': 4.5}, {'dispatch_monotonic_s': 2.}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                activation_first_services([self.epoch()], [self.dispatch() | changed], clock_id='test')


if __name__ == '__main__':
    unittest.main()
