"""Deterministic formula tests; no inference, remote traffic or new workload."""
import dataclasses
import asyncio
import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from faaslora.experiment.instance_pool import (
    InstancePool, ReplicaRoutingSnapshot, Router, ServiceClassBins,
    ServiceComponents, ServiceCostModel, ServiceIntervalObservation,
    ServiceObservationClass, NativeSourceSnapshot, InstanceSlot, FrozenServiceProfiles,
    confirmed_source_class,
)


def klass(tier='remote', admitted=0, representation='safetensors-fp16'):
    return ServiceObservationClass(tier, 0, 0, 0, 0, representation, admitted)


def candidate(replica='a', **changes):
    fields = dict(epoch=1, request_id='r', replica_id=replica, adapter_id='lora-a',
                  runtime_ready=True, available_slots=2, admitted_requests=1,
                  active_adapters=frozenset({'lora-a'}), max_active_loras=2,
                  pending_loads=0, gpu_utilization_pct=0., last_dispatch_at=0.,
                  service_class=klass(), service=ServiceComponents(10., 20., 30.))
    fields.update(changes)
    return ReplicaRoutingSnapshot(**fields)


def source_payload():
    return dict(kind='native_lora_sources_v1', owner_id='owner', epoch=1, clock_id='clock',
                captured_monotonic_s=10., slot_adapter_ids=[4, None],
                registered_cpu_adapter_ids=[4], unknown_native_adapter_ids=[],
                unconfirmed_gpu_adapter_ids=[], complete_for_native_caches=True,
                snapshot_holds_reference=False,
                sources=[dict(adapter_int_id=4, adapter_id='a', lora_path='/existing/a',
                              rank=8, cpu_registered=True, gpu_slot=0,
                              gpu_confirmed_monotonic_s=9.)])


def measured_source_payload():
    payload = source_payload()
    payload['native_footprints'] = dict(uniform_slot_layout=True,
        host_footprint_scope='native_registered_tensor_storage_capacity', host_budget_reserved=False,
        host_allocator_overhead_included=False, slot_adapter_ids=[4, None],
        registered_cpu_adapter_ids=[4], slot_capacity_bytes=1024, pool_allocated_bytes=2048,
        host_tensor_storage_bytes=512, host_allocations=[dict(allocation_id=0,
            allocated_bytes=512, adapter_ids=[4], pinned=False)],
        host_adapter_footprints=[dict(adapter_int_id=4, allocation_ids=[0], storage_bytes=512,
            exclusive_storage_bytes=512, dtypes=['torch.float16'],
            representation='native_cpu_dense_ab_v1', has_packed_modules=False)],
        pool_tensor_views=[dict(dtype='torch.float16')])
    return payload


class CommittedNativeSources(unittest.TestCase):
    def parse(self, payload=None):
        return NativeSourceSnapshot.from_native(payload or source_payload(),
            expected_clock_id='clock', received_monotonic_s=20.)

    def test_input_mutation_cannot_change_the_committed_state(self):
        payload = source_payload()
        state = self.parse(payload)
        payload['sources'][0]['adapter_id'] = 'changed'
        payload['slot_adapter_ids'].clear()
        self.assertEqual(state.sources[0].adapter_id, 'a')
        self.assertEqual(state.slot_adapter_ids, (4, None))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            state.sources[0].rank = 16

    def test_native_miss_is_not_a_remote_claim_and_cpu_source_survives(self):
        payload = source_payload()
        payload['slot_adapter_ids'] = [None, None]
        payload['sources'][0].update(gpu_slot=None, gpu_confirmed_monotonic_s=None)
        state = self.parse(payload)
        self.assertEqual(state.find(adapter_id='a', adapter_int_id=4, lora_path='/existing/a').tier, 'host')
        self.assertIsNone(state.find(adapter_id='b', adapter_int_id=5, lora_path='/existing/b'))

    def test_unowned_native_id_is_not_attached_to_an_arbitrary_adapter(self):
        payload = source_payload()
        payload.update(sources=[], unknown_native_adapter_ids=[4], complete_for_native_caches=False)
        state = self.parse(payload)
        with self.assertRaisesRegex(ValueError, 'without confirmed source'):
            state.find(adapter_id='a', adapter_int_id=4, lora_path='/existing/a')

    def test_source_lookup_rejects_native_id_name_and_path_collisions(self):
        state = self.parse()
        for changes in ({'adapter_id': 'b'}, {'adapter_int_id': 5}, {'lora_path': '/wrong'}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'identity'):
                state.find(**(dict(adapter_id='a', adapter_int_id=4, lora_path='/existing/a') | changes))

    def test_raw_slot_without_completed_copy_cannot_be_gpu_ready(self):
        payload = source_payload()
        payload['sources'][0]['gpu_confirmed_monotonic_s'] = None
        with self.assertRaisesRegex(ValueError, 'completed-copy'):
            self.parse(payload)
        payload['sources'][0]['gpu_slot'] = None
        payload.update(unconfirmed_gpu_adapter_ids=[4], complete_for_native_caches=False)
        self.assertEqual(self.parse(payload).sources[0].tier, 'host')

    def test_wrong_clock_future_copy_and_inconsistent_coverage_reject(self):
        for change in ({'clock_id': 'other'}, {'captured_monotonic_s': 30.},
                       {'registered_cpu_adapter_ids': []}, {'unknown_native_adapter_ids': [4]},
                       {'complete_for_native_caches': False}, {'epoch': True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.parse(source_payload() | change)
        payload = source_payload()
        payload['sources'][0]['gpu_confirmed_monotonic_s'] = 11.
        with self.assertRaises(ValueError):
            self.parse(payload)

    def test_delayed_reply_does_not_replace_newer_state_or_mutate_legacy_hints(self):
        slot = InstanceSlot('replica', object(), object())
        first = self.parse()
        slot.commit_native_sources(first)
        newer = dataclasses.replace(first, epoch=2, captured_monotonic_s=11.)
        self.assertTrue(slot.commit_native_sources(newer))
        self.assertFalse(slot.commit_native_sources(first))
        self.assertIs(slot.native_source_state, newer)
        self.assertFalse(slot.gpu_resident_adapters)
        self.assertFalse(slot.host_cached_adapters)

    def test_owner_change_and_same_epoch_different_content_require_reconciliation(self):
        slot = InstanceSlot('replica', object(), object())
        first = self.parse()
        slot.commit_native_sources(first)
        with self.assertRaisesRegex(ValueError, 'owner changed'):
            slot.commit_native_sources(dataclasses.replace(first, owner_id='new-worker'))
        changed = dataclasses.replace(first, sources=(dataclasses.replace(first.sources[0], rank=16),))
        with self.assertRaisesRegex(ValueError, 'same native epoch'):
            slot.commit_native_sources(changed)
        self.assertIs(slot.native_source_state, first)

    def test_native_source_class_uses_gpu_slot_or_actual_host_storage(self):
        payload = measured_source_payload()
        bins = ServiceClassBins((16,), (32,), (8,), (512,), (1,))
        features = dict(prompt_tokens=16, declared_output_tokens=32, admitted_after_accept=2)
        state = self.parse(payload)
        gpu = state.sources[0].service_class(bins, **features)
        self.assertEqual((gpu.tier, gpu.footprint_bin, gpu.admitted_bin), ('gpu', 1, 1))
        self.assertEqual(gpu.representation, 'native_gpu_dense_slot_v1:torch.float16')
        self.assertEqual(state.host_tensor_storage_bytes, 512)
        self.assertEqual(state.gpu_pool_storage_bytes, 2048)
        payload['sources'][0].update(gpu_slot=None, gpu_confirmed_monotonic_s=None)
        payload.update(slot_adapter_ids=[None, None])
        payload['native_footprints']['slot_adapter_ids'] = [None, None]
        host = self.parse(payload).sources[0].service_class(bins, **features)
        self.assertEqual((host.tier, host.footprint_bin), ('host', 0))
        self.assertEqual(host.representation, 'native_cpu_dense_ab_v1:torch.float16:unpinned')
        payload['native_footprints']['host_allocations'][0]['pinned'] = True
        pinned = self.parse(payload).sources[0].service_class(bins, **features)
        self.assertNotEqual(host, pinned)
        self.assertTrue(pinned.representation.endswith(':pinned'))

    def test_readiness_only_snapshot_cannot_invent_service_cost_class(self):
        with self.assertRaisesRegex(ValueError, 'lacks measured footprint'):
            self.parse().sources[0].service_class(ServiceClassBins((), (), (), (), ()),
                prompt_tokens=8, declared_output_tokens=8, admitted_after_accept=1)

    def test_malformed_storage_totals_edges_or_wrong_native_state_reject(self):
        for field, value in (('host_tensor_storage_bytes', 1024), ('pool_allocated_bytes', 1024),
                             ('slot_adapter_ids', [None, None]), ('registered_cpu_adapter_ids', []),
                             ('uniform_slot_layout', False)):
            payload = measured_source_payload()
            payload['native_footprints'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.parse(payload)
        for field, value in (('allocation_ids', [0, 0]), ('exclusive_storage_bytes', 0),
                             ('storage_bytes', 123), ('dtypes', [])):
            payload = measured_source_payload()
            payload['native_footprints']['host_adapter_footprints'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.parse(payload)

    def test_shared_host_storage_total_is_not_sum_of_adapter_footprints(self):
        payload = measured_source_payload()
        payload.update(registered_cpu_adapter_ids=[4, 5], unknown_native_adapter_ids=[5],
                       complete_for_native_caches=False)
        fp = payload['native_footprints']
        fp['registered_cpu_adapter_ids'] = [4, 5]
        fp['host_allocations'][0]['adapter_ids'] = [4, 5]
        fp['host_adapter_footprints'][0]['exclusive_storage_bytes'] = 0
        fp['host_adapter_footprints'].append(fp['host_adapter_footprints'][0] | {'adapter_int_id': 5})
        state = self.parse(payload)
        self.assertEqual(state.host_tensor_storage_bytes, 512)
        self.assertEqual(state.sources[0].host_storage_bytes, 512)
        payload['native_footprints']['host_tensor_storage_bytes'] = 1024
        with self.assertRaisesRegex(ValueError, 'distinct storage union'):
            self.parse(payload)


class ConfirmedTierComposition(unittest.TestCase):
    def setUp(self):
        self.identity = dict(adapter_id='a', rank=8, content_sha256='a'*64,
            remote_payload_bytes=10000, remote_representation='tar_gzip_verified_file_tree_v1')
        self.files = dict(kind='confirmed_file_sources_v1', owner_id='files', epoch=2,
            clock_id='clock', adapter_id='a', snapshot_holds_reference=False, sources=[])
        self.bins = ServiceClassBins((), (), (8,), (512, 1024, 4096), ())

    def classify(self, payload, **changes):
        args = dict(native=NativeSourceSnapshot.from_native(payload,
            expected_clock_id='clock', received_monotonic_s=20.), files=self.files,
            identity=self.identity, adapter_int_id=4, bins=self.bins,
            prompt_tokens=2, declared_output_tokens=4, admitted_after_accept=1)
        return confirmed_source_class(**(args | changes))

    def file(self, tier):
        return dict(tier=tier, adapter_id='a', content_sha256='a'*64, content_verified=True,
            representation='verified_regular_file_tree_v1', allocated_file_bytes=4096,
            path=f'/managed/{tier}/a')

    def empty_native(self):
        payload = source_payload()
        payload.update(sources=[], slot_adapter_ids=[None, None], registered_cpu_adapter_ids=[])
        return payload

    def test_fastest_confirmed_native_source_preserves_tensor_representation(self):
        self.files['sources'] = [self.file('host'), self.file('nvme')]
        payload = measured_source_payload()
        key, evidence = self.classify(payload)
        self.assertEqual((key.tier, key.footprint_bin), ('gpu', 1))
        self.assertTrue(evidence['native'])
        self.assertNotIn('content_sha256', evidence)  # expected ID is not native byte verification
        payload['slot_adapter_ids'] = [None, None]
        payload['sources'][0].update(gpu_slot=None, gpu_confirmed_monotonic_s=None)
        payload['native_footprints']['slot_adapter_ids'] = [None, None]
        key, evidence = self.classify(payload)
        self.assertEqual((key.tier, key.footprint_bin), ('host', 0))
        self.assertTrue(key.representation.startswith('native_cpu_'))

    def test_file_host_nvme_and_remote_are_not_tensor_or_wire_footprints(self):
        for tier, sources, footprint_bin in [('host', [self.file('nvme'), self.file('host')], 2),
                ('nvme', [self.file('nvme')], 2), ('remote', [], 3)]:
            self.files['sources'] = sources
            key, evidence = self.classify(self.empty_native())
            self.assertEqual((key.tier, key.footprint_bin), (tier, footprint_bin))
            self.assertFalse(evidence['native'])
            self.assertNotIn('native_', key.representation)

    def test_bad_file_content_rejects_even_when_a_native_hit_exists(self):
        for change in ({'content_sha256': 'b'*64}, {'content_verified': False},
                       {'allocated_file_bytes': 0}, {'path': 'relative'}, {'tier': 'gpu'}):
            self.files['sources'] = [self.file('host') | change]
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.classify(measured_source_payload())

    def test_unknown_native_id_and_wrong_native_name_or_rank_reject(self):
        payload = source_payload()
        payload.update(sources=[], unknown_native_adapter_ids=[4], complete_for_native_caches=False)
        with self.assertRaisesRegex(ValueError, 'unowned'):
            self.classify(payload)
        for change in ({'adapter_id': 'wrong'}, {'rank': 16}):
            payload = measured_source_payload()
            payload['sources'][0].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'identity/rank'):
                self.classify(payload)


class FrozenMeasuredInitialization(unittest.TestCase):
    """Small invented contract fixtures only; never model profiling evidence."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'profile.json'
        self.model = dict(model_path='/existing/model', dtype='float16',
                          tensor_parallel_size=1, visible_device_ids=[0, 1],
                          timing_contract='ieee_tc_native_v1', ieee_gpu_references=True,
                          generation_contract='fixed_length_greedy_v1')
        self.context = dict(backend_environment_sha256='a'*64,
                            resource_envelope_sha256='b'*64, input_contract_sha256='c'*64)
        self.features = dict(tier='host', prompt_tokens=16, declared_output_tokens=32,
                             adapter_rank=8, footprint_bytes=512,
                             representation='native_cpu_dense_ab_v1:torch.float16:unpinned',
                             admitted_after_accept=1)
        sample = dict(source_measurement='native_admission_acquisition_token_events_v1',
                      correct=True, source_run_sha256='d'*64, request_id='fixture-1',
                      attempt_id='attempt-1', native_clock_id='test-clock',
                      admission_clock_id='test-clock', class_features=self.features,
                      native_output_tokens=32, admitted_monotonic_s=10.,
                      acquired_monotonic_s=10.125, first_token_monotonic_s=10.5,
                      last_token_monotonic_s=11.)
        self.payload = dict(kind='native_service_profiles_v1', context=self.context,
                            model_config=FrozenServiceProfiles.model_identity(self.model),
                            bins=dict(prompt_tokens=[16], declared_output_tokens=[32],
                                      adapter_rank=[8], footprint_bytes=[512], admitted_requests=[1]),
                            samples=[sample])

    def write_profile(self, payload=None):
        data = json.dumps(self.payload if payload is None else payload, sort_keys=True).encode()
        self.path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def load(self, payload=None, **changes):
        kwargs = dict(expected_sha256=self.write_profile(payload), model_config=self.model,
                      expected_context=self.context, beta=.25)
        return FrozenServiceProfiles.load(self.path, **(kwargs | changes))

    def test_means_come_from_native_boundaries_and_preserve_class_counts(self):
        second = copy.deepcopy(self.payload['samples'][0])
        second.update(request_id='fixture-2', acquired_monotonic_s=10.375,
                      first_token_monotonic_s=11., last_token_monotonic_s=12.5)
        self.payload['samples'].append(second)
        profiles = self.load()
        key = profiles.bins.classify(**self.features)
        self.assertEqual(profiles.profiles[key], ServiceComponents(250., 500., 1000.))
        self.assertEqual(profiles.sample_counts[key], 2)
        self.assertEqual(profiles.identity()['initial_samples'], 2)
        self.assertEqual(profiles.identity()['source_run_sha256'], ['d'*64])
        with self.assertRaises(TypeError):
            profiles.profiles[key] = ServiceComponents(0, 0, 0)

    def test_sha_and_model_or_backend_configuration_changes_reject(self):
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.load(expected_sha256='0'*64)
        for field, value in [('model_path', '/other'), ('dtype', 'bfloat16'),
                             ('tensor_parallel_size', 2),
                             ('ieee_native_host_allocator_policy', 'uncached_v1')]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'context mismatch'):
                self.load(model_config=self.model | {field: value})
        self.load(model_config=self.model | {'visible_device_ids': [3], 'device_id': 3})
        for field in self.context:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'context mismatch'):
                self.load(expected_context=self.context | {field: 'f'*64})

    def test_context_bins_and_fixed_generation_are_required(self):
        for bad in ({}, self.context | {'extra': 'f'*64},
                    self.context | {'backend_environment_sha256': 'unlocked'}):
            with self.assertRaisesRegex(ValueError, 'frozen environment'):
                self.load(expected_context=bad)
        bad = copy.deepcopy(self.payload)
        bad['bins']['prompt_tokens'] = [16, 16]
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            self.load(bad)
        for field in ('generation_contract', 'timing_contract'):
            model = self.model | {field: 'legacy'}
            bad = self.payload | {'model_config': FrozenServiceProfiles.model_identity(model)}
            with self.assertRaisesRegex(ValueError, 'native timing'):
                self.load(bad, model_config=model)

    def test_wrong_clock_duplicates_incorrect_and_non_native_samples_reject(self):
        for changes in ({'admission_clock_id': 'other-clock'}, {'correct': False},
                        {'request_id': ''}, {'source_run_sha256': ''},
                        {'source_measurement': 'http_completion'}, {'attempt_id': ''}):
            bad = copy.deepcopy(self.payload)
            bad['samples'][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.load(bad)
        bad = copy.deepcopy(self.payload)
        bad['samples'].append(copy.deepcopy(bad['samples'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.load(bad)

    def test_native_count_and_interval_order_are_not_estimated_or_clamped(self):
        for changes in ({'native_output_tokens': 31}, {'native_output_tokens': 32.},
                        {'acquired_monotonic_s': 9.}, {'last_token_monotonic_s': 10.25},
                        {'first_token_monotonic_s': math.nan}, {'admitted_monotonic_s': True}):
            bad = copy.deepcopy(self.payload)
            bad['samples'][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.load(bad)

    def test_gpu_requires_protection_at_admission_and_single_token_has_no_decode(self):
        sample = self.payload['samples'][0]
        sample['class_features']['tier'] = 'gpu'
        with self.assertRaisesRegex(ValueError, 'protected at admission'):
            self.load()
        sample.update(protected_at_admission=True)
        with self.assertRaisesRegex(ValueError, 'protected at admission'):
            self.load()
        sample.update(acquired_monotonic_s=10., native_output_tokens=1)
        sample['class_features']['declared_output_tokens'] = 1
        with self.assertRaisesRegex(ValueError, 'O=0'):
            self.load()
        sample['last_token_monotonic_s'] = sample['first_token_monotonic_s']
        profiles = self.load()
        self.assertEqual(next(iter(profiles.profiles.values())), ServiceComponents(0, 500, 0))

    def test_missing_classes_cannot_borrow_a_nearby_profile(self):
        profiles = self.load()
        for change in ({'prompt_tokens': 17}, {'admitted_after_accept': 2},
                       {'tier': 'gpu'}, {'footprint_bytes': 513}):
            with self.subTest(change=change), self.assertRaises(KeyError):
                profiles.new_replica().estimate(profiles.bins.classify(**(self.features | change)))

    def test_pool_scaleout_resets_learning_and_does_not_alias_the_runtime(self):
        profiles = self.load()
        key = profiles.bins.classify(**self.features)
        pool = InstancePool(service_profiles=profiles)
        first_engine = SimpleNamespace(model_cfg=self.model)
        first = pool.get_slot(pool.add_instance(first_engine, None))
        observation = ServiceIntervalObservation(first.service_cost_model, key, 20.)
        observation.acquire(20.5)
        self.assertEqual(first.service_cost_model.estimate(key).d_ms, 218.75)
        second = pool.get_slot(pool.add_instance(SimpleNamespace(model_cfg=self.model | {'device_id': 1}), None))
        self.assertEqual(second.service_cost_model.estimate(key).d_ms, 125.)
        self.assertEqual(second.service_cost_model.sample_counts(key), {'d_ms': 0, 't_ms': 0, 'o_ms': 0})
        self.assertIs(first.service_class_bins, profiles.bins)
        with self.assertRaisesRegex(ValueError, 'alias one physical runtime'):
            pool.add_instance(first_engine, None)
        with self.assertRaisesRegex(ValueError, 'differs from its measured'):
            pool.add_instance(SimpleNamespace(model_cfg=self.model | {'dtype': 'bfloat16'}), None)
        self.assertEqual(pool.count(), 2)

    def runner(self, coord_changes=None, model_changes=None):
        from scripts.run_all_experiments import ScenarioRunner
        model = self.model | (model_changes or {})
        spec = dict(path=str(self.path), sha256=self.write_profile(),
                    context=self.context, ewma_beta=.25)
        coord = dict(instance_mode='dedicated', routing_policy='ieee_confirmed',
                     service_bin_ms=10., ieee_service_profile=spec)
        coord.update(coord_changes or {})
        return ScenarioRunner(name='fixture', baseline_type='faaslora_full', adapter_info={},
            traces=[], remote_dir=Path(self.tmp.name), nvme_dir=Path(self.tmp.name),
            bandwidth_mbps=100., hardware_cfg={'gpu_device_ids': [0, 1]}, cost_model={},
            engine=SimpleNamespace(device_id=0, model_cfg=model), runner_model_cfg=model,
            preload_cfg={}, workload_cfg={'generation_contract': 'fixed_length_greedy_v1'},
            coord_cfg=coord)

    def test_actual_runner_initializes_profiles_bins_and_summary_identity(self):
        runner = self.runner()
        slot = runner.instance_pool.get_slots()[0]
        key = slot.service_class_bins.classify(**self.features)
        self.assertEqual(slot.service_cost_model.estimate(key), ServiceComponents(125, 375, 500))
        self.assertEqual(runner.router.service_bin_ms, 10.)
        identity = runner._current_coord_metrics()['ieee_service_profile']
        self.assertEqual(identity['profile_sha256'], slot.service_cost_model.profile_id)
        self.assertEqual(identity['initial_samples'], 1)

    def test_service_profile_alone_cannot_enter_legacy_activation_or_warmup(self):
        runner = self.runner()
        engine = SimpleNamespace(model_cfg=self.model | {'dtype': 'bfloat16'}, shutdown=AsyncMock())
        runner.engine_factory = AsyncMock(return_value=(engine, None))
        runner._warmup_engine_hot_set = AsyncMock()
        with self.assertRaisesRegex(ValueError, 'real owners and frozen measurements'):
            asyncio.run(runner._add_dedicated_instance_slot(True, reserved_device_id=1))
        runner.engine_factory.assert_not_awaited()
        engine.shutdown.assert_not_awaited()
        runner._warmup_engine_hot_set.assert_not_called()
        self.assertEqual(runner.instance_pool.count(), 1)

    def test_actual_runner_rejects_absent_profiles_and_shared_runtime(self):
        for changes, message in [({'ieee_service_profile': None}, 'measured ieee_service_profile'),
                                 ({'instance_mode': 'shared'}, 'distinct physical runtime'),
                                 ({'service_bin_ms': None}, 'bin width')]:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                self.runner(coord_changes=changes)
        with self.assertRaisesRegex(ValueError, 'invalid IEEE measured service profile'):
            self.runner(model_changes={'ieee_gpu_references': False})


class ServiceMeasurementContract(unittest.TestCase):
    def test_class_uses_declared_features_and_inclusive_edges(self):
        bins = ServiceClassBins((16, 32), (32, 64), (8,), (1024,), (1, 4))
        features = dict(tier='remote', prompt_tokens=16, declared_output_tokens=64,
                        adapter_rank=8, footprint_bytes=1025, representation='r',
                        admitted_after_accept=2)
        key = bins.classify(**features)
        self.assertEqual((key.prompt_bin, key.output_limit_bin, key.rank_bin,
                          key.footprint_bin, key.admitted_bin), (0, 1, 0, 1, 1))
        self.assertNotEqual(key, bins.classify(**{**features, 'admitted_after_accept': 5}))
        with self.assertRaises(ValueError):
            ServiceClassBins((2, 2), (), (), (), ())
        with self.assertRaises(ValueError):
            bins.classify(**{**features, 'admitted_after_accept': 0})
        with self.assertRaises(TypeError):
            bins.classify(**features, actual_generated_tokens=2)

    def test_first_sample_updates_profile_with_ewma_not_replace_or_average(self):
        model = ServiceCostModel({klass(): ServiceComponents(100, 200, 300)},
                                 beta=.25, profile_id='measured-profile')
        model.record_interval(klass(), 'd_ms', 20)
        self.assertEqual(model.estimate(klass()).d_ms, 80)
        model.record_interval(klass(), 'd_ms', 40)
        self.assertEqual(model.estimate(klass()).d_ms, 70)
        self.assertEqual(model.sample_counts(klass()), {'d_ms': 2, 't_ms': 0, 'o_ms': 0})
        self.assertEqual(model.new_replica().estimate(klass()).d_ms, 100)

    def test_three_nonoverlapping_intervals_and_admission_class(self):
        remote, gpu = klass(), klass('gpu')
        model = ServiceCostModel({remote: ServiceComponents(10, 20, 30),
                                  gpu: ServiceComponents(0, 20, 30)},
                                 beta=1, profile_id='p')
        obs = ServiceIntervalObservation(model, remote, 10)
        obs.acquire(10.125)
        obs.first_token(10.5)
        obs.last_token(12)
        self.assertEqual(model.estimate(remote), ServiceComponents(125, 375, 1500))
        self.assertEqual(model.estimate(remote).total_ms, 2000)
        # Resolving to executable GPU does not reclassify the admitted request.
        self.assertEqual(model.sample_counts(gpu), {'d_ms': 0, 't_ms': 0, 'o_ms': 0})
        with self.assertRaises(ValueError):
            obs.last_token(12.2)  # completion notification is not another last token

    def test_protected_gpu_hit_and_single_token_have_zero_intervals(self):
        key = klass('gpu')
        model = ServiceCostModel({key: ServiceComponents(0, 20, 30)}, beta=1, profile_id='p')
        obs = ServiceIntervalObservation(model, key, 1)
        self.assertEqual(obs.acquired_at, 1)
        obs.first_token(1.25)
        obs.last_token(1.25)
        self.assertEqual(model.estimate(key), ServiceComponents(0, 250, 0))

    def test_cancellation_preserves_completed_intervals_without_inventing_tail(self):
        model = ServiceCostModel({klass(): ServiceComponents(1, 2, 3)}, beta=1, profile_id='p')
        obs = ServiceIntervalObservation(model, klass(), 1)
        obs.acquire(1.25)
        obs.cancel()
        self.assertEqual(model.estimate(klass()), ServiceComponents(250, 2, 3))
        self.assertEqual(model.sample_counts(klass()), {'d_ms': 1, 't_ms': 0, 'o_ms': 0})
        with self.assertRaises(ValueError):
            obs.first_token(2)

    def test_uninitialized_or_invalid_costs_fail_without_zero_fallback(self):
        for beta in (0, -1, 1.1, math.nan):
            with self.assertRaises(ValueError):
                ServiceCostModel({klass(): ServiceComponents(1, 2, 3)}, beta=beta, profile_id='p')
        with self.assertRaises(ValueError):
            ServiceCostModel({klass('gpu'): ServiceComponents(1, 2, 3)}, beta=1, profile_id='p')
        model = ServiceCostModel({klass(): ServiceComponents(1, 2, 3)}, beta=1, profile_id='p')
        with self.assertRaises(KeyError):
            model.estimate(klass('host'))
        obs = ServiceIntervalObservation(model, klass(), 2)
        for timestamp in (1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                obs.acquire(timestamp)
        with self.assertRaises(ValueError):
            obs.first_token(3)
        self.assertEqual(model.sample_counts(klass()), {'d_ms': 0, 't_ms': 0, 'o_ms': 0})


class CommittedRoutingContract(unittest.TestCase):
    def test_strict_feasible_set_empty_means_queue(self):
        rows = (candidate('a', available_slots=0),
                candidate('b', active_adapters=frozenset({'other'}), max_active_loras=1),
                candidate('c', runtime_ready=False))
        self.assertIsNone(Router.select_ieee_snapshot(rows, 10))
        self.assertIsNone(Router.select_ieee_snapshot((), 10))
        allowed = candidate('d', max_active_loras=1)
        self.assertEqual(Router.select_ieee_snapshot(rows + (allowed,), 10), allowed)

    def test_service_bin_then_load_not_unbinned_cost(self):
        expensive_idle = candidate('b', admitted_requests=0, active_adapters=frozenset(),
                                   service=ServiceComponents(10, 20, 39))
        cheaper_busy = candidate('a')
        self.assertEqual(Router.select_ieee_snapshot((cheaper_busy, expensive_idle), 10).replica_id, 'b')
        next_bin = dataclasses.replace(expensive_idle, service=ServiceComponents(10, 20, 40))
        self.assertEqual(Router.select_ieee_snapshot((cheaper_busy, next_bin), 10).replica_id, 'a')

    def test_Q_order_and_stable_id_are_exact(self):
        better = candidate('z')
        for field, value in [('admitted_requests', 2), ('pending_loads', 1),
                             ('gpu_utilization_pct', 1.), ('last_dispatch_at', 1.)]:
            worse = dataclasses.replace(candidate('a'), **{field: value})
            self.assertEqual(Router.select_ieee_snapshot((worse, better), 10), better)
        a = candidate('a')
        self.assertEqual(Router.select_ieee_snapshot((better, a), 10), a)

    def test_snapshot_rejects_mixed_epochs_requests_duplicates_and_mutability(self):
        row = candidate()
        for rows in ((row, dataclasses.replace(row, replica_id='b', epoch=2)),
                     (row, dataclasses.replace(row, replica_id='b', request_id='x')),
                     (row, row), [row]):
            with self.assertRaises(ValueError):
                Router.select_ieee_snapshot(rows, 10)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            row.pending_loads = 10
        with self.assertRaises(ValueError):
            candidate(active_adapters={'lora-a'})

    def test_AA_shadow_selection_is_pure(self):
        rows = (candidate('b'), candidate('a'))
        router = Router(SimpleNamespace(get_slots=lambda: []), 'ieee_confirmed', service_bin_ms=10)
        first = router.select_ieee_snapshot(rows, 10)
        self.assertEqual(first, router.select_ieee_snapshot(rows, 10))
        self.assertEqual(router.selection_count, 0)
        self.assertIsNone(router.last_ieee_decision)

    def test_live_router_uses_only_explicit_snapshot_not_handoff_or_old_cost(self):
        pool = InstancePool(max_instances=2)
        a = pool.add_instance(None, None)
        b = pool.add_instance(None, None)
        pool.get_slot(a).scaleup_handoff_request_budget = 100
        pool.get_slot(a).scaleup_handoff_planned_adapter_ranks = {'different': 0}
        rows = (candidate(b), candidate(a))
        router = Router(pool, 'ieee_confirmed', service_bin_ms=10)
        self.assertEqual(router.select_instance('lora-a', ieee_snapshot=rows).instance_id, a)
        self.assertFalse(hasattr(pool.get_slot(a), 'scaleup_handoff_assigned_requests'))
        self.assertEqual(router.last_ieee_decision.replica_id, a)
        with self.assertRaises(ValueError):
            router.select_instance('lora-a')
        with self.assertRaises(ValueError):
            router.select_instance('different', ieee_snapshot=rows)
        pool.get_slot(b).status = 'draining'
        with self.assertRaises(ValueError):
            router.select_instance('lora-a', ieee_snapshot=rows)

    def test_missing_profile_does_not_make_feasible_replica_artificially_fast(self):
        with self.assertRaises(ValueError):
            candidate(service=None)
        self.assertFalse(candidate(available_slots=0, service=None).feasible)
        with self.assertRaises(ValueError):
            candidate(service_class=klass('gpu'))
        for delta in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                Router.select_ieee_snapshot((), delta)


if __name__ == '__main__':
    unittest.main()
