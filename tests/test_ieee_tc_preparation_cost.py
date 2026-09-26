"""Preparation d excludes pre-load waiting; fixtures are not model profiles."""
import unittest
from faaslora.clock import local_monotonic_clock_id
from faaslora.preloading.preloading_planner import observed_preparation_interval


class PreparationIntervals(unittest.TestCase):
    def inputs(self, tier='nvme', native_source=False):
        clock = local_monotonic_clock_id()
        admission = dict(source=dict(tier=tier, native=native_source, owner_id='worker' if native_source else 'files',
            path=None if tier == 'remote' else '/existing/a'),
            admitted_monotonic_s=100., service_class=dict(tier=tier))
        native = dict(acquired=True, clock_id=clock, owner_id='worker', lease_id='lease',
            lora_name='a', lora_path='/existing/a', native_load_invoked=True,
            source_tier_before_acquisition='host' if native_source else 'file',
            native_load_started_monotonic_s=150., native_load_completed_monotonic_s=153.,
            acquired_monotonic_s=153.)
        remote = dict(artifact_id='a', state='published', content_verified=True,
            loading_clock_id=clock, transfer_id='transfer', target_path='/existing/a',
            loading_started_monotonic_s=111., published_monotonic_s=140.)
        return dict(adapter_id='a', request_id='r', admission=admission, native=native,
                    remote=remote if tier == 'remote' else None)

    def test_host_and_file_exclude_rpc_queue_and_capacity_wait(self):
        for tier, native in (('host', True), ('host', False), ('nvme', False)):
            with self.subTest(tier=tier, native=native):
                sample = observed_preparation_interval(**self.inputs(tier, native))
                self.assertTrue(sample['profile_eligible'])
                self.assertEqual(sample['d_ms'], 3000.)
                self.assertEqual(sample['excluded_before_loading_ms'], 50000.)

    def test_remote_keeps_wait_between_loading_stages_not_before_first_stage(self):
        sample = observed_preparation_interval(**self.inputs('remote'))
        self.assertEqual(sample['d_ms'], 42000.)
        self.assertEqual(sample['native_loading_ms'], 3000.)
        self.assertEqual(sample['after_remote_publication_ms'], 13000.)
        self.assertEqual(sample['excluded_before_loading_ms'], 11000.)

    def test_shared_and_changed_sources_do_not_become_zero_cost_samples(self):
        args = self.inputs('remote')
        args['remote'] = None
        result = observed_preparation_interval(**args)
        self.assertFalse(result['profile_eligible'])
        self.assertIsNone(result['d_ms'])
        self.assertEqual(result['reason'], 'shared_file_preparation_reused')
        for field, value in (('native_load_invoked', False), ('source_tier_before_acquisition', 'host')):
            args = self.inputs()
            args['native'][field] = value
            result = observed_preparation_interval(**args)
            self.assertFalse(result['profile_eligible'])
            self.assertIsNone(result['d_ms'])

    def test_wrong_native_owner_clock_identity_and_boundaries_rejected(self):
        for field, value in (('clock_id', 'other-clock'), ('lora_name', 'other'),
                ('lora_path', '/other'), ('lease_id', ''), ('native_load_invoked', None),
                ('native_load_started_monotonic_s', None), ('native_load_started_monotonic_s', 99.),
                ('native_load_completed_monotonic_s', 154.)):
            args = self.inputs()
            args['native'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                observed_preparation_interval(**args)

    def test_remote_identity_success_and_stage_order_are_required(self):
        for field, value in (('loading_clock_id', 'other'), ('artifact_id', 'other'),
                ('state', 'not_published'), ('content_verified', False), ('transfer_id', ''),
                ('target_path', '/other'), ('loading_started_monotonic_s', 99.),
                ('published_monotonic_s', 151.)):
            args = self.inputs('remote')
            args['remote'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                observed_preparation_interval(**args)

    def test_gpu_zero_is_a_definition_not_a_completed_load_sample(self):
        with self.assertRaisesRegex(ValueError, 'non-executable'):
            observed_preparation_interval(**self.inputs('gpu', True))
