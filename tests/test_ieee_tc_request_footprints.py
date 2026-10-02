"""Fresh per-request classes with complete native identities; CPU-only tests."""
import asyncio
import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from faaslora.clock import local_monotonic_clock_id
from faaslora.experiment.instance_pool import InstanceSlot, NativeSourceSnapshot, ServiceClassBins
from faaslora.memory import gpu_monitor
from scripts.run_all_experiments import InferenceEngine, SubprocessInferenceEngineProxy
from tests.test_ieee_tc_service_routing import measured_source_payload
from tests import test_ieee_tc_worker_observation as observation
from tests import test_ieee_tc_gpu_references as references
from tests import test_ieee_tc_request_lifecycle as lifecycle


def full_payload():
    p = measured_source_payload()
    p['device_uuid'] = 'GPU-fixture'
    p['registered_cpu_adapter_ids'] = [4, 5]
    p['sources'].append({**p['sources'][0], 'adapter_int_id': 5, 'adapter_id': 'b',
        'lora_path': '/existing/b', 'source_id': 'copy-b', 'gpu_slot': None,
        'gpu_confirmed_monotonic_s': None})
    f = p['native_footprints']
    f['registered_cpu_adapter_ids'] = [4, 5]
    f['host_allocations'].append(dict(allocation_id=1, allocated_bytes=1024, adapter_ids=[5], pinned=True))
    f['host_tensor_storage_bytes'] = 1536
    f['host_adapter_footprints'].append({**f['host_adapter_footprints'][0], 'adapter_int_id': 5,
        'allocation_ids': [1], 'storage_bytes': 1024, 'exclusive_storage_bytes': 1024})
    return p


def scoped_payload(ids=(4,)):
    p = full_payload()
    f = p.pop('native_footprints')
    adapters = [a for a in f['host_adapter_footprints'] if a['adapter_int_id'] in ids]
    indices = sorted({i for a in adapters for i in a['allocation_ids']})
    remap = {old: new for new, old in enumerate(indices)}
    f['host_allocations'] = [{**f['host_allocations'][old], 'allocation_id': remap[old]} for old in indices]
    for a in adapters:
        a['allocation_ids'] = [remap[i] for i in a['allocation_ids']]
        a['within_observation_exclusive_bytes'] = a.pop('exclusive_storage_bytes')
    f['host_adapter_footprints'] = adapters
    f.pop('host_tensor_storage_bytes')
    f['requested_host_storage_bytes'] = sum(a['allocated_bytes'] for a in f['host_allocations'])
    f['host_footprint_scope'] = 'native_requested_tensor_storage_capacity'
    f['host_requested_adapter_ids'] = list(ids)
    p['native_request_footprints'] = f
    return p


def parse(ids=(4,), payload=None):
    return NativeSourceSnapshot.from_request_native(payload or scoped_payload(ids), requested_adapter_ids=ids,
        expected_clock_id='clock', received_monotonic_s=20.)


class RequestFootprintSchema(unittest.TestCase):
    def test_target_class_and_all_identities_match_full_observation(self):
        full = NativeSourceSnapshot.from_native(full_payload(), expected_clock_id='clock', received_monotonic_s=20.)
        for ids in ((4,), (5,), (4, 5), (), (99,)):
            with self.subTest(ids=ids):
                state = parse(ids)
                self.assertEqual(state.identity_view(), full.identity_view())
                self.assertIsNone(state.host_tensor_storage_bytes)
                wire = state.request_wire(device_uuid='GPU-fixture')
                self.assertEqual(NativeSourceSnapshot.from_request_wire(wire, requested_adapter_ids=ids,
                    expected_clock_id='clock', received_monotonic_s=20.), state)
                for source in state.sources:
                    if source.adapter_int_id in ids:
                        self.assertEqual(source, next(s for s in full.sources if s.adapter_int_id == source.adapter_int_id))
                    else:
                        self.assertIsNone(source.host_storage_bytes)
                        with self.assertRaisesRegex(ValueError, 'measured footprint'):
                            source.service_class(ServiceClassBins((), (), (), (), ()),
                                prompt_tokens=2, declared_output_tokens=4, admitted_after_accept=1)

    def test_scoped_view_cannot_be_published_as_full_inventory(self):
        with self.assertRaisesRegex(ValueError, 'full routing'):
            parse().routing_wire(device_uuid='GPU-fixture')
        with self.assertRaisesRegex(ValueError, 'scoped measured'):
            parse().identity_view().request_wire(device_uuid='GPU-fixture')

    def test_invalid_scope_is_rejected_not_normalized(self):
        for ids in (None, [True], [0], [4, 4], [5, 4], '4'):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                NativeSourceSnapshot.from_request_native(scoped_payload(), requested_adapter_ids=ids,
                    expected_clock_id='clock', received_monotonic_s=20.)

    def test_scope_and_coverage_must_match_both_endpoints(self):
        p = scoped_payload()
        with self.assertRaisesRegex(ValueError, 'scope'):
            parse((5,), p)
        p['native_request_footprints']['host_adapter_footprints'] = []
        with self.assertRaises(ValueError):
            parse(payload=p)
        wire = parse().request_wire(device_uuid='GPU-fixture')
        for edit in (lambda w: w['request_footprints'].update(requested_adapter_ids=[5]),
                     lambda w: w['request_footprints'].update(sources=[]),
                     lambda w: w['request_footprints'].update(host_tensor_storage_bytes=512),
                     lambda w: w['request_footprints']['sources'][0].update(host_storage_bytes=0),
                     lambda w: w['request_footprints']['sources'][0].update(gpu_slot_capacity_bytes=9)):
            bad = copy.deepcopy(wire)
            edit(bad)
            with self.assertRaises(ValueError):
                NativeSourceSnapshot.from_request_wire(bad, requested_adapter_ids=[4],
                    expected_clock_id='clock', received_monotonic_s=20.)

    def test_global_capacity_or_reclaimable_names_fail_closed(self):
        for mutation in ('union', 'exclusive'):
            p = scoped_payload()
            f = p['native_request_footprints']
            if mutation == 'union':
                f['host_tensor_storage_bytes'] = 512
            else:
                f['host_adapter_footprints'][0]['exclusive_storage_bytes'] = 512
            with self.assertRaises(ValueError):
                parse(payload=p)

    def test_malformed_identity_is_not_hidden_by_scope(self):
        p = scoped_payload()
        p['sources'][1]['rank'] = 0
        with self.assertRaisesRegex(ValueError, 'identity/rank'):
            parse(payload=p)


class RequestFootprintFreshness(unittest.TestCase):
    def test_interleaved_scopes_retain_same_epoch_overlap_consistency(self):
        slot = InstanceSlot('r', None, None)
        self.assertTrue(slot.commit_native_sources(parse((4,))))
        self.assertTrue(slot.commit_native_sources(replace(parse((5,)), captured_monotonic_s=11.)))
        self.assertEqual(set(slot.native_source_footprint_frontier), {4, 5})
        third = replace(parse((4,)), captured_monotonic_s=12.)
        bad = replace(third, sources=(replace(third.sources[0], host_storage_bytes=999), third.sources[1]))
        with self.assertRaisesRegex(ValueError, 'footprint'):
            slot.commit_native_sources(bad)
        self.assertTrue(slot.commit_native_sources(third))
        self.assertIsNone(slot.native_source_state.sources[1].host_storage_bytes)

    def test_new_epoch_resets_consistency_only_not_identity_frontier(self):
        slot = InstanceSlot('r', None, None)
        slot.commit_native_sources(parse((4,)))
        newer = replace(parse((5,)), epoch=2, captured_monotonic_s=11.)
        self.assertTrue(slot.commit_native_source_identity(newer.identity_view()))
        self.assertFalse(slot.commit_native_sources(parse((5,))))
        self.assertTrue(slot.commit_native_sources(newer))
        self.assertEqual(set(slot.native_source_footprint_frontier), {5})

    def test_older_capture_and_same_epoch_identity_change_reject(self):
        slot = InstanceSlot('r', None, None)
        slot.commit_native_sources(replace(parse((4,)), captured_monotonic_s=12.))
        self.assertFalse(slot.commit_native_sources(parse((5,))))
        bad = replace(parse((5,)), captured_monotonic_s=13.,
            sources=(replace(parse((5,)).sources[0], source_id='changed'), parse((5,)).sources[1]))
        with self.assertRaisesRegex(ValueError, 'same native epoch'):
            slot.commit_native_sources(bad)

    def test_full_and_scoped_transitions_preserve_all_known_footprints(self):
        full = NativeSourceSnapshot.from_native(full_payload(), expected_clock_id='clock', received_monotonic_s=20.)
        slot = InstanceSlot('r', None, None)
        slot.commit_native_sources(full)
        self.assertTrue(slot.commit_native_sources(replace(parse((5,)), captured_monotonic_s=11.)))
        self.assertTrue(slot.commit_native_sources(replace(full, captured_monotonic_s=12.)))

    def test_full_union_consistency_survives_interleaved_scoped_read(self):
        full = NativeSourceSnapshot.from_native(full_payload(), expected_clock_id='clock', received_monotonic_s=20.)
        slot = InstanceSlot('r', None, None)
        slot.commit_native_sources(full)
        slot.commit_native_sources(replace(parse((5,)), captured_monotonic_s=11.))
        # Target footprint totals cannot recover a union when aliases change.
        # Do not discard the independently observed global capacity frontier.
        bad = replace(full, host_tensor_storage_bytes=1024, captured_monotonic_s=12.)
        with self.assertRaisesRegex(ValueError, 'observed capacity'):
            slot.commit_native_sources(bad)
        self.assertTrue(slot.commit_native_sources(replace(bad, epoch=2)))
        self.assertEqual(slot.native_source_capacity_frontier['host_tensor_storage_bytes'], 1024)


class RequestFootprintObservation(unittest.TestCase):
    def test_native_tensors_scoped_without_fictitious_reclaimable_credit(self):
        native, _ = observation.NativeHostFootprint().native_models()
        result = gpu_monitor._ieee_lora_host_inventory(native, requested_adapter_ids=[8], include_tensor_views=False)
        self.assertEqual(result['host_footprint_scope'], 'native_requested_tensor_storage_capacity')
        self.assertNotIn('host_tensor_storage_bytes', result)
        self.assertEqual(result['requested_host_storage_bytes'], 512)
        self.assertEqual(len(result['host_adapter_footprints']), 1)
        self.assertNotIn('exclusive_storage_bytes', result['host_adapter_footprints'][0])
        missing = gpu_monitor._ieee_lora_host_inventory(native, requested_adapter_ids=[99], include_tensor_views=False)
        self.assertEqual(missing['requested_host_storage_bytes'], 0)
        self.assertEqual(missing['host_adapter_footprints'], [])

    def test_actual_native_owner_stays_fresh_and_checks_unrelated_source_mutation(self):
        case = references.NativeDemandTransactions()
        case.setUp()
        case.demand()
        worker = gpu_monitor.IEEEWorkerObservationExtension()
        worker.device, worker.rank = SimpleNamespace(type='cuda'), 0
        worker.model_runner = SimpleNamespace(lora_manager=SimpleNamespace(_adapter_manager=case.manager))
        worker._ieee_gpu_reference_owner = case.owner
        worker._ieee_host_allocator_policy = {'verified': False}
        torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda _:
            SimpleNamespace(uuid=SimpleNamespace(bytes=list(range(16))))))
        with patch.object(gpu_monitor, 'torch', torch), \
             patch.object(gpu_monitor, '_ieee_lora_host_inventory', return_value={'requested_host_storage_bytes': 16}) as host, \
             patch.object(gpu_monitor, '_ieee_lora_pool_inventory', return_value={'slot_capacity_bytes': 32}), \
             patch.object(gpu_monitor, '_ieee_pinned_host_observation') as allocator:
            first = worker.ieee_gpu_reference(operation='request_source_snapshot', requested_adapter_ids=[4])
            host.assert_called_once_with(case.manager, include_tensor_views=False, requested_adapter_ids=[4])
            self.assertNotIn('native_footprints', first)
            self.assertIn('staged_sources', first)
            case.release('cold-1')
            case.demand('second', aid=5)
            second = worker.ieee_gpu_reference(operation='request_source_snapshot', requested_adapter_ids=[4])
            self.assertGreater(second['epoch'], first['epoch'])
            self.assertIn(5, second['registered_cpu_adapter_ids'])
            allocator.assert_not_called()
            case.manager._registered_adapters[5] = object()
            with self.assertRaisesRegex(RuntimeError, 'source object'):
                worker.ieee_gpu_reference(operation='request_source_snapshot', requested_adapter_ids=[4])
            self.assertEqual(host.call_count, 2)

    def test_frontend_and_proxy_forward_exact_scope(self):
        p = scoped_payload()
        p['clock_id'] = local_monotonic_clock_id()
        engine = SimpleNamespace(ieee_gpu_reference=AsyncMock(return_value=p))
        value = asyncio.run(InferenceEngine.ieee_request_sources(engine, requested_adapter_ids=[4]))
        self.assertEqual(value['kind'], 'native_lora_request_sources_v1')
        engine.ieee_gpu_reference.assert_awaited_once_with(operation='request_source_snapshot', requested_adapter_ids=[4])
        proxy = SimpleNamespace(_rpc=AsyncMock(return_value=value))
        self.assertEqual(asyncio.run(SubprocessInferenceEngineProxy.ieee_request_sources(proxy, requested_adapter_ids=[4])), value)
        proxy._rpc.assert_awaited_once_with('ieee_request_sources', requested_adapter_ids=[4])

    def test_interleaved_different_targets_do_not_share_missing_footprints(self):
        runner, slots, _, _, _ = lifecycle.PredecisionRoutingIntegration().build()
        async def run():
            values = await asyncio.gather(*(runner._ieee_collect_native_sources(tuple(slots), requested_adapter_ids=ids)
                for ids in ([4], [5], [4])))
            self.assertEqual([v[1]['requested_adapter_ids'] for v in values], [[4], [5], [4]])
            self.assertEqual(runner._ieee_source_observation_stats['collections'], 2)
            self.assertEqual(runner._ieee_source_observation_stats['joined'], 1)
            self.assertEqual(sum(s.engine.ieee_request_sources.await_count for s in slots), 4)
            self.assertEqual(runner._ieee_source_observation_waves, {})
        asyncio.run(run())

    def test_last_reader_cancel_for_one_scope_does_not_cancel_another_scope(self):
        runner, slots, _, _, _ = lifecycle.PredecisionRoutingIntegration().build()
        async def run():
            entered = {4: asyncio.Event(), 5: asyncio.Event()}
            released = asyncio.Event()
            cancelled = []
            original = slots[0].engine.ieee_request_sources.side_effect
            async def held(*, requested_adapter_ids):
                aid = requested_adapter_ids[0]
                entered[aid].set()
                try:
                    await released.wait()
                except asyncio.CancelledError:
                    cancelled.append(aid)
                    raise
                return await original(requested_adapter_ids=requested_adapter_ids)
            slots[0].engine.ieee_request_sources.side_effect = held
            tasks = [asyncio.create_task(runner._ieee_collect_native_sources(tuple(slots),
                requested_adapter_ids=[aid])) for aid in (4, 5)]
            try:
                await asyncio.wait_for(asyncio.gather(*(e.wait() for e in entered.values())), 1.)
                tasks[0].cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await tasks[0]
                self.assertEqual(cancelled, [4])
                self.assertFalse(tasks[1].done())
                self.assertEqual(len(runner._ieee_source_observation_waves), 1)
                released.set()
                value = await asyncio.wait_for(tasks[1], 1.)
                self.assertEqual(value[1]['requested_adapter_ids'], [5])
                self.assertEqual(runner._ieee_source_observation_waves, {})
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        asyncio.run(run())


if __name__ == '__main__':
    unittest.main()
