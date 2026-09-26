"""Preparation d excludes pre-load waiting; fixtures are not model profiles."""
import unittest
import copy
import hashlib
import json
import tempfile
from pathlib import Path
from dataclasses import asdict
from faaslora.clock import local_monotonic_clock_id
from faaslora.preloading.preloading_planner import (
    observed_preparation_interval, PreparationClass, PreparationCostModel, FrozenPreparationProfiles)


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


class PreparationCosts(unittest.TestCase):
    def setUp(self):
        self.key = PreparationClass('nvme', 'verified_regular_file_tree_v1', 'fixture-layout', 0)
        self.model = PreparationCostModel({self.key: 10.}, beta=.5, profile_id='fixture-not-measurement')

    def sample(self):
        args = PreparationIntervals().inputs()
        args['admission']['service_class']['representation'] = self.key.representation
        sample = observed_preparation_interval(**args)
        sample['preparation_class'] = asdict(self.key)
        return sample

    def test_completed_d_updates_without_preload_wait_and_inheritance_is_frozen(self):
        old_seq, old_view = self.model.snapshot()
        self.assertTrue(self.model.record_completed_load(self.key, self.sample()))
        sequence, view = self.model.snapshot()
        self.assertEqual((old_seq, sequence), (0, 1))
        self.assertEqual(view[self.key], 1505.)  # (10 + actual 3000) / 2, not D=53000.
        self.assertEqual(old_view[self.key], 10.)
        self.assertEqual(self.model.new_replica().snapshot()[1][self.key], 10.)
        with self.assertRaises(TypeError):
            view[self.key] = 0

    def test_duplicate_or_unsupported_samples_do_not_update(self):
        sample = self.sample()
        self.model.record_completed_load(self.key, sample)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.model.record_completed_load(self.key, sample)
        other = PreparationClass('nvme', self.key.representation, 'another-layout', 0)
        sample['preparation_class'] = asdict(other)
        with self.assertRaises(KeyError):
            self.model.record_completed_load(other, sample)
        self.assertEqual(self.model.snapshot()[0], 1)

    def test_changed_class_invalid_boundaries_or_service_D_rejected(self):
        for field, value in (('preparation_class', asdict(self.key) | {'size_bin': 1}),
                ('d_ms', 53000.), ('executable_monotonic_s', 149.),
                ('native_lease_id', ''), ('profile_eligible', None)):
            sample = self.sample()
            sample[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.model.record_completed_load(self.key, sample)
        self.assertEqual(self.model.snapshot()[0], 0)

    def test_ineligible_sample_not_zero_or_online_initialization(self):
        sample = self.sample() | dict(profile_eligible=False, d_ms=None,
                                    reason='shared_file_preparation_reused')
        self.assertFalse(self.model.record_completed_load(self.key, sample))
        self.assertEqual(self.model.snapshot()[0], 0)
        sample['d_ms'] = 0.
        with self.assertRaises(ValueError):
            self.model.record_completed_load(self.key, sample)

    def test_profile_and_class_validation(self):
        for value in (True, -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                PreparationCostModel({self.key: value}, beta=.5, profile_id='fixture')
        with self.assertRaises(ValueError):
            PreparationClass('host', '', 'layout', 0)
        gpu = PreparationClass('gpu', 'slots', 'layout', 0)
        with self.assertRaises(ValueError):
            PreparationCostModel({gpu: 1.}, beta=.5, profile_id='fixture')


class FrozenPreparationInitialization(unittest.TestCase):
    """Synthetic contract fixtures; these files are NOT measured model profiles."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'preparation.json'
        self.model = dict(model_path='/existing/model', dtype='float16', tensor_parallel_size=1,
            ieee_gpu_references=True, timing_contract='ieee_tc_native_v1',
            generation_contract='fixed_length_greedy_v1')
        self.context = dict(backend_environment_sha256='a'*64,
            resource_envelope_sha256='b'*64, input_contract_sha256='c'*64)
        args = PreparationIntervals().inputs('host', True)
        source = args['admission']['source']
        source.update(expected_content_sha256='d'*64, footprint_bytes=512,
            representation='native_cpu_dense_ab_v1:torch.float16:unpinned')
        args['admission']['clock_id'] = args['native']['clock_id'] = 'previous-boot-fixture-clock'
        args['admission']['service_class']['representation'] = source['representation']
        self.sample = dict(adapter_id='a', request_id='r', attempt_id='attempt-1',
            correct=True, source_run_sha256='e'*64,
            admission=args['admission'], native=args['native'], remote=None)
        self.payload = dict(kind='native_preparation_profiles_v1', layout_partition='exact_content_v1',
            context=self.context, model_config=self.model, size_edges_bytes=[512], samples=[self.sample])

    def write(self, payload=None):
        raw = json.dumps(self.payload if payload is None else payload, sort_keys=True).encode()
        self.path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def load(self, payload=None, **changes):
        arguments = dict(expected_sha256=self.write(payload), model_config=self.model,
            expected_context=self.context, beta=.5)
        return FrozenPreparationProfiles.load(self.path, **(arguments | changes))

    def test_means_use_recomputed_load_boundaries_with_original_clock(self):
        second = copy.deepcopy(self.sample)
        second.update(request_id='r2')
        second['native'].update(lease_id='lease2', native_load_started_monotonic_s=152.)
        profile = self.load(self.payload | {'samples': [self.sample, second]})
        key = profile.classify_source(self.sample['admission']['source'])
        self.assertEqual(profile.profiles[key], 2000.)  # (3000 + 1000)/2, never service D.
        self.assertEqual(profile.sample_counts[key], 2)
        self.assertEqual(key.size_bin, 0)
        self.assertEqual(key.layout_id, 'exact_content_sha256:'+'d'*64)
        self.assertEqual(profile.identity()['initial_samples'], 2)

    def test_unsupported_content_representation_and_size_cannot_borrow_cost(self):
        profile = self.load()
        for changes in ({'expected_content_sha256': 'f'*64}, {'footprint_bytes': 513},
                        {'representation': 'different-native-layout'}):
            with self.subTest(changes=changes), self.assertRaises(KeyError):
                profile.classify_source(self.sample['admission']['source'] | changes)

    def test_profile_hash_context_backend_contract_and_size_edges_reject(self):
        for changes in ({'expected_sha256': 'f'*64}, {'model_config': self.model | {'dtype': 'bfloat16'}},
                {'expected_context': self.context | {'resource_envelope_sha256': 'f'*64}}, {'beta': 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.load(**changes)
        for field, value in (('layout_partition', 'rank_only'), ('size_edges_bytes', [512, 512]),
                             ('samples', []), ('kind', 'native_service_profiles_v1')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load(self.payload | {field: value})

    def test_incomplete_changed_clock_and_duplicate_loads_reject(self):
        mutations = [lambda s: s.update(correct=False),
            lambda s: s['admission'].update(clock_id='other'),
            lambda s: s['native'].update(native_load_invoked=False),
            lambda s: s['admission']['source'].update(footprint_bytes=0),
            lambda s: s['admission']['service_class'].update(representation='other')]
        for change in mutations:
            sample = copy.deepcopy(self.sample)
            change(sample)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load(self.payload | {'samples': [sample]})
        for second in (self.sample, self.sample | {'request_id': 'r2', 'source_run_sha256': 'f'*64}):
            with self.assertRaises(ValueError):
                self.load(self.payload | {'samples': [self.sample, second]})

    def test_actual_pool_initializes_each_runtime_without_inheriting_online_costs(self):
        from types import SimpleNamespace
        from faaslora.experiment.instance_pool import InstancePool
        profile = self.load()
        pool = InstancePool(max_instances=4, preparation_profiles=profile)
        engine = SimpleNamespace(model_cfg=self.model)
        first = pool.get_slot(pool.add_instance(engine, None))
        key = profile.classify_source(self.sample['admission']['source'])
        args = PreparationIntervals().inputs('host', True)
        args['native']['native_load_started_monotonic_s'] = 152.
        args['admission']['service_class']['representation'] = key.representation
        sample = observed_preparation_interval(**args) | {'preparation_class': asdict(key)}
        first.preparation_cost_model.record_completed_load(key, sample)
        second = pool.get_slot(pool.add_instance(SimpleNamespace(model_cfg=self.model | {'device_id': 1}), None))
        self.assertEqual(first.preparation_cost_model.estimate(key), 2000.)
        self.assertEqual(second.preparation_cost_model.estimate(key), 3000.)
        with self.assertRaisesRegex(ValueError, 'alias'):
            pool.add_instance(engine, None)
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            pool.add_instance(SimpleNamespace(model_cfg=self.model | {'dtype': 'bfloat16'}), None)

    def test_actual_runner_loads_profile_and_reports_frozen_identity(self):
        from tests.test_ieee_tc_service_routing import FrozenMeasuredInitialization
        fixture = FrozenMeasuredInitialization()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.model = {k: v for k, v in fixture.model.items() if k not in ('device_id', 'visible_device_ids')}
        self.payload['model_config'] = self.model
        spec = dict(path=str(self.path), sha256=self.write(), context=self.context, ewma_beta=.5)
        runner = fixture.runner(coord_changes={'ieee_preparation_profile': spec})
        slot = runner.instance_pool.get_slots()[0]
        key = runner._preparation_profiles.classify_source(self.sample['admission']['source'])
        self.assertEqual(slot.preparation_cost_model.estimate(key), 3000.)
        self.assertEqual(runner._current_coord_metrics()['ieee_preparation_profile']['profile_sha256'], spec['sha256'])
        with self.assertRaisesRegex(ValueError, 'invalid IEEE measured preparation profile'):
            fixture.runner(coord_changes={'ieee_preparation_profile': {'path': 'missing'}})
        other_context = self.context | {'resource_envelope_sha256': 'f'*64}
        other_spec = spec | {'context': other_context,
            'sha256': self.write(self.payload | {'context': other_context})}
        with self.assertRaisesRegex(ValueError, 'measurement contexts differ'):
            fixture.runner(coord_changes={'ieee_preparation_profile': other_spec})
