"""Fresh readiness-only rechecks; physical/routing observations remain complete."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from faaslora.clock import local_monotonic_clock_id
from faaslora.experiment.instance_pool import InstanceSlot, NativeSourceSnapshot, ServiceClassBins
from faaslora.memory import gpu_monitor
from scripts.run_all_experiments import InferenceEngine, SubprocessInferenceEngineProxy
from tests.test_ieee_tc_service_routing import measured_source_payload
from tests import test_ieee_tc_request_lifecycle as lifecycle


class IdentityRecheck(unittest.TestCase):
    def build(self):
        state = NativeSourceSnapshot.from_native(measured_source_payload(),
            expected_clock_id='clock', received_monotonic_s=20.)
        slot = InstanceSlot('replica', engine=None, coordinator=None)
        self.assertTrue(slot.commit_native_sources(state))
        return slot, state, state.identity_view()

    def test_identity_projection_keeps_source_and_cannot_supply_service_class(self):
        _, full, identity = self.build()
        self.assertEqual(identity.sources[0].source_id, full.sources[0].source_id)
        self.assertEqual(identity.slot_adapter_ids, full.slot_adapter_ids)
        self.assertIsNone(identity.host_tensor_storage_bytes)
        self.assertIsNone(identity.gpu_pool_storage_bytes)
        with self.assertRaisesRegex(ValueError, 'measured footprint'):
            identity.sources[0].service_class(ServiceClassBins((), (), (), (), ()),
                prompt_tokens=2, declared_output_tokens=4, admitted_after_accept=1)

    def test_same_epoch_identity_does_not_replace_measured_routing_view(self):
        slot, full, identity = self.build()
        newer = replace(identity, captured_monotonic_s=11.)
        self.assertTrue(slot.commit_native_source_identity(newer))
        self.assertIs(slot.native_source_state, full)
        self.assertIs(slot.native_source_identity_state, newer)
        self.assertEqual(full.host_tensor_storage_bytes, 512)
        self.assertFalse(slot.accepts_native_sources(full))  # older capture
        fresh = replace(full, captured_monotonic_s=12.)
        self.assertTrue(slot.commit_native_sources(fresh))

    def test_identity_frontier_rejects_delayed_full_epoch(self):
        slot, full, identity = self.build()
        self.assertTrue(slot.commit_native_source_identity(replace(identity, epoch=2,
                                                                  captured_monotonic_s=12.)))
        self.assertFalse(slot.commit_native_sources(replace(full, captured_monotonic_s=11.)))
        self.assertIs(slot.native_source_state, full)
        self.assertTrue(slot.commit_native_sources(replace(full, epoch=2, captured_monotonic_s=13.)))

    def test_same_epoch_copy_identity_change_rejected_in_both_directions(self):
        for newer_is_full in (False, True):
            slot, full, identity = self.build()
            if newer_is_full:
                slot.commit_native_source_identity(identity)
                row = replace(full.sources[0], source_id='wrong-copy')
                # Clear the full view to isolate the independently committed identity frontier.
                slot.native_source_state = None
                with self.assertRaisesRegex(ValueError, 'same native epoch'):
                    slot.commit_native_sources(replace(full, sources=(row,)))
            else:
                row = replace(identity.sources[0], source_id='wrong-copy')
                with self.assertRaisesRegex(ValueError, 'same native epoch'):
                    slot.commit_native_source_identity(replace(identity, sources=(row,)))

    def test_owner_clock_capture_and_stale_identity_rejected(self):
        for change in ({'owner_id': 'wrong'}, {'clock_id': 'wrong'},
                       {'epoch': 2, 'captured_monotonic_s': 9.}):
            slot, _, identity = self.build()
            with self.assertRaises(ValueError):
                slot.commit_native_source_identity(replace(identity, **change))
        slot, _, identity = self.build()
        slot.commit_native_source_identity(replace(identity, epoch=2, captured_monotonic_s=12.))
        self.assertFalse(slot.commit_native_source_identity(identity))

    def test_full_inventory_cannot_enter_identity_only_publication(self):
        slot, full, _ = self.build()
        with self.assertRaisesRegex(ValueError, 'footprint evidence'):
            slot.commit_native_source_identity(full)
        self.assertIsNone(slot.native_source_identity_state)

    def test_full_same_epoch_footprint_inconsistency_still_rejected(self):
        slot, full, identity = self.build()
        slot.commit_native_source_identity(identity)
        with self.assertRaisesRegex(ValueError, 'same native epoch'):
            slot.commit_native_sources(replace(full, host_tensor_storage_bytes=999))

    def test_frontend_queries_native_identity_endpoint_and_rejects_inventory(self):
        payload = measured_source_payload()
        payload.pop('native_footprints')
        payload.update(clock_id=local_monotonic_clock_id(), device_uuid='GPU-fixture')
        engine = SimpleNamespace(ieee_gpu_reference=AsyncMock(return_value=payload))
        result = asyncio.run(InferenceEngine.ieee_source_identities(engine))
        self.assertIsNone(result['routing_footprints'])
        engine.ieee_gpu_reference.assert_awaited_once_with(operation='source_identity_snapshot')
        payload['native_footprints'] = {}
        with self.assertRaisesRegex(ValueError, 'physical inventory'):
            asyncio.run(InferenceEngine.ieee_source_identities(engine))

    def test_worker_identity_query_uses_live_owner_without_any_inventory(self):
        payload = measured_source_payload()
        payload.pop('native_footprints')
        manager = object()
        owner = SimpleNamespace(manager=manager, source_snapshot=Mock(return_value=payload))
        worker = gpu_monitor.IEEEWorkerObservationExtension()
        worker.device = SimpleNamespace(type='cuda')
        worker.rank = 0
        worker.model_runner = SimpleNamespace(lora_manager=SimpleNamespace(_adapter_manager=manager))
        worker._ieee_gpu_reference_owner = owner
        worker._ieee_host_allocator_policy = {'verified': False}
        torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=Mock(
            return_value=SimpleNamespace(uuid=SimpleNamespace(bytes=list(range(16)))))))
        with patch.object(gpu_monitor, 'torch', torch), \
             patch.object(gpu_monitor, '_ieee_lora_host_inventory') as host, \
             patch.object(gpu_monitor, '_ieee_lora_pool_inventory') as pool, \
             patch.object(gpu_monitor, '_ieee_pinned_host_observation') as allocator:
            first = worker.ieee_gpu_reference(operation='source_identity_snapshot')
            self.assertNotIn('native_footprints', first)
            payload['epoch'] = 2
            second = worker.ieee_gpu_reference(operation='source_identity_snapshot')
            self.assertEqual(second['epoch'], 2)
            self.assertEqual(first['epoch'], 1)
            owner.source_snapshot.side_effect = RuntimeError('owner invariant violation')
            with self.assertRaisesRegex(RuntimeError, 'owner invariant'):
                worker.ieee_gpu_reference(operation='source_identity_snapshot')
            host.assert_not_called(); pool.assert_not_called(); allocator.assert_not_called()
            self.assertEqual(owner.source_snapshot.call_count, 3)

    def test_proxy_uses_explicit_identity_rpc(self):
        proxy = SimpleNamespace(_rpc=AsyncMock(return_value={'identity': 'fixture'}))
        self.assertEqual(asyncio.run(SubprocessInferenceEngineProxy.ieee_source_identities(proxy)),
                         {'identity': 'fixture'})
        proxy._rpc.assert_awaited_once_with('ieee_source_identities')

    def test_faster_copy_appearing_at_recheck_forces_new_routing(self):
        case = lifecycle.SelectedSourceAdmissionIntegration()
        try:
            runner, slot, trace, plan, owner, observed = case.build('remote')
            case.file_source(runner, trace, publish=True)
            calls = 0
            async def appear(*, operation, **kwargs):
                nonlocal calls
                if operation == 'source_identity_snapshot':
                    calls += 1
                    if calls == 1:
                        lifecycle.ControllerNativeReferenceLifecycle.preload_native(owner)
                return await observed(operation=operation, **kwargs)
            slot.engine.ieee_gpu_reference.side_effect = appear
            result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
            self.assertTrue(result.success, result.error)
            self.assertEqual(result.readiness_tier_before_dispatch, 'gpu')
            self.assertEqual(calls, 1)
            self.assertIn('faster native source',
                result.gpu_reference_evidence['prior_routing_attempts'][0]['native_source_conflict'])
            self.assertEqual(owner.snapshot()['live_leases'], 0)
        finally:
            case.doCleanups()
