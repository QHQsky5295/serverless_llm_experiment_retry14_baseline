"""Real runner admission lifetime with fake inference; no GPU performance claims."""
import asyncio
from dataclasses import asdict, replace
import hashlib
import json
import socket
import tempfile
import threading
import time
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from faaslora.experiment.instance_pool import InstanceSlot
from scripts.run_all_experiments import (
    ScenarioRunner, RequestExecutionPlan, RuntimeRequestReservation, ScenarioResult, aggregate_runs,
    InferenceEngine, SubprocessInferenceEngineProxy)


def fixture():
    runner = ScenarioRunner.__new__(ScenarioRunner)
    runner.model_cfg = {'runtime_concurrency_cap': 2, 'max_num_seqs': 2,
                        'max_loras': 2, 'timing_contract': 'legacy'}
    runner.baseline_type = 'faaslora_full'
    runner.cost_model = {}
    runner._stack = None
    runner._unsettled_runtime_reservations = {}
    runner._remote_transfer_evidence = []
    runner.adapter_info = {'adapter-a': {'size_mb': 30.}}
    runner._prune_dead_instance_slots = AsyncMock()
    runner._refresh_all_slot_runtime_hints = Mock()
    runner._refresh_slot_runtime_hints = Mock()
    runner._release_live_waiting_trace = Mock()
    runner._observe_live_started_lora = Mock()
    runner._slot_predicted_total_busy_ms = Mock(return_value=100.)
    runner._notify_dispatch_capacity_changed = AsyncMock()
    runner._schedule_all_runtime_gpu_forward = Mock()
    runner._mark_slot_adapter_tier = Mock()
    runner._begin_scaleup_runtime_request_labels = Mock(return_value={})
    engine = SimpleNamespace(generate_prepared=AsyncMock(
        return_value=(10., 2., 4, {'runtime_estimated_e2e_ms': 16.})))
    slot = InstanceSlot('inst-a', engine=engine, coordinator=None)
    runner.router = SimpleNamespace(select_instance=lambda *a, **k: slot)
    runner._resolve_lora = AsyncMock(return_value=('adapter-a', '/existing/a', 1., 'nvme', 0., 0.))
    trace = SimpleNamespace(request_id='req-a', adapter_id='adapter-a', is_burst=False,
                            expected_output_tokens=4, prompt='hello')
    plan = RequestExecutionPlan('hello', 2, 4)
    return runner, slot, trace, plan


class PredecisionRoutingIntegration(unittest.TestCase):
    """Actual runner/router calls with explicit native-measurement fixtures, no GPU."""
    def test_ieee_full_cannot_silently_start_legacy_preloading(self):
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._routing_policy = 'ieee_confirmed'
        runner._stack = SimpleNamespace(start=AsyncMock())
        with self.assertRaisesRegex(RuntimeError, 'legacy priority/warmup is forbidden'):
            asyncio.run(runner._preload_full_stack())
        runner._stack.start.assert_not_awaited()

    def build(self):
        from faaslora.clock import local_monotonic_clock_id
        from faaslora.experiment.instance_pool import Router, ServiceClassBins, ServiceCostModel, ServiceComponents
        from tests.test_ieee_tc_service_routing import source_payload
        runner, a, trace, plan = fixture()
        runner.model_cfg.update(timing_contract='ieee_tc_native_v1', ieee_gpu_references=True)
        runner._routing_policy = 'ieee_confirmed'
        runner._ieee_routing_epoch = 0
        runner._ieee_nvml_initialized = False
        runner._runtime_hints_refresh_interval_s = 1.
        runner._ieee_artifact_identities = {trace.adapter_id: dict(adapter_id=trace.adapter_id,
            rank=8, content_sha256='a'*64, remote_payload_bytes=8192,
            remote_representation='tar_gzip_verified_file_tree_v1')}
        files = Mock()
        files.source_snapshot.side_effect = lambda aid: dict(kind='confirmed_file_sources_v1',
            owner_id='files', epoch=1, clock_id=local_monotonic_clock_id(), adapter_id=aid,
            snapshot_holds_reference=False, captured_monotonic_s=time.monotonic(), sources=[])
        runner._stack = SimpleNamespace(residency_manager=SimpleNamespace(local_source_references=files))
        b = InstanceSlot('inst-b', engine=SimpleNamespace(), coordinator=None)
        slots = [a, b]
        runner.instance_pool = SimpleNamespace(get_slots=lambda: list(slots))
        runner.router = Router(runner.instance_pool, 'ieee_confirmed', service_bin_ms=10.)
        bins = ServiceClassBins((), (), (), (), ())
        key = bins.classify(tier='remote', prompt_tokens=2, declared_output_tokens=4,
            adapter_rank=8, footprint_bytes=8192, representation='tar_gzip_verified_file_tree_v1',
            admitted_after_accept=1)
        for index, slot in enumerate(slots):
            slot.device_id = index
            slot.service_class_bins = bins
            slot.service_cost_model = ServiceCostModel(beta=.25, profile_id='test-fixture-only',
                profiles={key: ServiceComponents(100. if index == 0 else 10., 20., 30.)})
            async def snapshot(*, operation, replica=slot.instance_id, ordinal=index):
                self.assertEqual(operation, 'source_snapshot')
                return source_payload() | dict(owner_id=replica, clock_id=local_monotonic_clock_id(),
                    captured_monotonic_s=time.monotonic(), sources=[],
                    device_uuid=f'GPU-00010203-0405-0607-0809-0a0b0c0d0e0{ordinal}',
                    slot_adapter_ids=[None, None], registered_cpu_adapter_ids=[])
            slot.engine.ieee_gpu_reference = AsyncMock(side_effect=snapshot)
        runner._sample_ieee_gpu_utilization = Mock(side_effect=lambda slot, **kw: dict(
            source='fixture-native-busy', device_id=slot.device_id, device_uuid=kw['device_uuid'],
            gpu_utilization_pct=0., sampled_monotonic_s=time.monotonic()))
        return runner, slots, trace, plan, files

    def test_actual_predecision_view_and_selection_ignore_legacy_affinity(self):
        runner, (a, b), trace, plan, files = self.build()
        a.gpu_resident_adapters.add(trace.adapter_id)
        a.scaleup_handoff_request_budget = 100
        rows, evidence = asyncio.run(runner._ieee_request_snapshot(trace, plan))
        self.assertEqual([row.service_class.tier for row in rows], ['remote', 'remote'])
        self.assertIs(runner.router.select_instance(trace.adapter_id, ieee_snapshot=rows), b)
        self.assertEqual(evidence[b.instance_id]['source']['tier'], 'remote')
        self.assertEqual(a.active_requests, 0)
        self.assertEqual(b.active_requests, 0)
        self.assertEqual(rows[0].epoch, rows[1].epoch)
        runner._refresh_all_slot_runtime_hints.assert_not_called()
        files.source_snapshot.assert_called_once_with(trace.adapter_id)

    def test_live_counts_make_infeasible_rows_without_fabricating_missing_cost(self):
        runner, (a, b), trace, plan, _ = self.build()
        a.active_requests = 2
        a.service_cost_model = None
        b.ieee_pending_load_ids.add('already-loading')
        rows, _ = asyncio.run(runner._ieee_request_snapshot(trace, plan))
        self.assertFalse(rows[0].feasible)
        self.assertIsNone(rows[0].service)
        self.assertEqual(rows[1].pending_loads, 1)
        self.assertIs(runner.router.select_instance(trace.adapter_id, ieee_snapshot=rows), b)
        b.runtime_forwarding_active = 1
        with self.assertRaisesRegex(ValueError, 'untracked legacy forwarding'):
            asyncio.run(runner._ieee_request_snapshot(trace, plan))

    def test_membership_change_during_collection_retries_without_selection(self):
        runner, slots, trace, plan, _ = self.build()
        original = slots[0].engine.ieee_gpu_reference.side_effect
        async def changing(**kwargs):
            result = await original(**kwargs)
            slots.pop()
            return result
        slots[0].engine.ieee_gpu_reference.side_effect = changing
        self.assertIsNone(asyncio.run(runner._ieee_request_snapshot(trace, plan)))
        self.assertEqual(runner.router.selection_count, 0)
        self.assertEqual(slots[0].active_requests, 0)

    def test_delayed_native_epoch_cannot_be_used_after_newer_commit(self):
        runner, (a, _), trace, plan, _ = self.build()
        asyncio.run(runner._ieee_request_snapshot(trace, plan))
        a.native_source_state = replace(a.native_source_state, epoch=2)
        self.assertIsNone(asyncio.run(runner._ieee_request_snapshot(trace, plan)))
        self.assertEqual(a.native_source_state.epoch, 2)

    def test_duplicate_native_devices_do_not_masquerade_as_scaleout(self):
        runner, (a, b), trace, plan, _ = self.build()
        original = b.engine.ieee_gpu_reference.side_effect
        async def duplicate(**kwargs):
            view = await original(**kwargs)
            view['device_uuid'] = 'GPU-00010203-0405-0607-0809-0a0b0c0d0e00'
            return view
        b.engine.ieee_gpu_reference.side_effect = duplicate
        with self.assertRaisesRegex(ValueError, 'distinct native physical GPU'):
            asyncio.run(runner._ieee_request_snapshot(trace, plan))
        self.assertEqual(runner.router.selection_count, 0)

    def test_actual_request_reserves_selected_view_before_resolve_then_releases(self):
        runner, (a, b), trace, plan, _ = self.build()
        reservation = RuntimeRequestReservation(trace.request_id)
        async def stop_at_resolution(*args, **kwargs):
            self.assertIs(reservation.slot, b)
            self.assertEqual(reservation.ieee_routing_evidence['selected_replica_id'], 'inst-b')
            self.assertEqual(b.active_requests, 1)
            self.assertIn(trace.request_id, b.ieee_pending_load_ids)
            self.assertEqual(a.active_requests, 0)
            raise RuntimeError('end of routing boundary witness')
        runner._ieee_prepare_selected_adapter = AsyncMock(side_effect=stop_at_resolution)
        async def run():
            try:
                with self.assertRaisesRegex(RuntimeError, 'boundary witness'):
                    await runner._exec_request_in_reservation(trace, 4, 0.,
                        request_plan=plan, _reservation=reservation)
            finally:
                await runner._finish_runtime_request_reservation(reservation)
        asyncio.run(run())
        self.assertTrue(reservation.released)
        self.assertFalse(b.ieee_pending_load_ids)
        self.assertEqual(b.active_requests, 0)
        self.assertGreater(b.ieee_last_dispatch_at, 0.)

    def test_empty_feasible_set_waits_and_never_uses_primary_fallback(self):
        runner, (a, b), trace, plan, _ = self.build()
        a.active_requests = b.active_requests = 2
        runner.engine = SimpleNamespace(generate_prepared=AsyncMock())
        async def run():
            task = asyncio.create_task(runner._exec_request(trace, 4, 0., request_plan=plan))
            for _ in range(20):
                await asyncio.sleep(0)
                if runner.router.selection_count:
                    break
            self.assertGreater(runner.router.selection_count, 0)
            self.assertFalse(task.done())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(run())
        runner.engine.generate_prepared.assert_not_called()
        runner._resolve_lora.assert_not_called()
        self.assertEqual((a.active_requests, b.active_requests), (2, 2))

    def test_unknown_load_retains_pending_and_controller_ownership(self):
        runner, (a, _), trace, _, _ = self.build()
        a.active_requests = 1
        a.ieee_pending_load_ids.add(trace.request_id)
        reservation = RuntimeRequestReservation(trace.request_id)
        reservation.bind(a, trace.adapter_id, False)
        reservation.ieee_load_pending = True
        reservation.gpu_reference_evidence['state'] = 'acquiring'
        asyncio.run(runner._finish_runtime_request_reservation(reservation))
        self.assertFalse(reservation.released)
        self.assertEqual(a.ieee_pending_load_ids, {trace.request_id})
        self.assertEqual(a.active_requests, 1)
        self.assertEqual(a.status, 'draining')

    def test_native_busy_sample_not_memory_fraction_and_cadence_is_recorded(self):
        runner, (a, _), _, _, _ = self.build()
        del runner._sample_ieee_gpu_utilization
        native = SimpleNamespace(nvmlInit=Mock(), nvmlShutdown=Mock(),
            nvmlDeviceGetHandleByUUID=Mock(return_value='handle'),
            nvmlDeviceGetUtilizationRates=Mock(return_value=SimpleNamespace(gpu=13, memory=91)),
            nvmlDeviceGetHandleByIndex=Mock(side_effect=AssertionError('index is not native identity')))
        a.utilization_percent = 99.
        device_uuid = 'GPU-00010203-0405-0607-0809-0a0b0c0d0e0f'
        with patch.dict('sys.modules', {'pynvml': native}):
            first = runner._sample_ieee_gpu_utilization(a, device_uuid=device_uuid)
            second = runner._sample_ieee_gpu_utilization(a, device_uuid=device_uuid)
            self.assertEqual(first, second)
            self.assertEqual(first['gpu_utilization_pct'], 13.)
            self.assertEqual(first['device_uuid'], device_uuid)
            native.nvmlDeviceGetHandleByUUID.assert_called_once_with(device_uuid)
            native.nvmlDeviceGetUtilizationRates.assert_called_once_with('handle')
            with self.assertRaisesRegex(ValueError, 'physical GPU changed'):
                runner._sample_ieee_gpu_utilization(a,
                    device_uuid='GPU-00010203-0405-0607-0809-0a0b0c0d0e00')
            a.ieee_utilization_sample = None
            native.nvmlDeviceGetUtilizationRates.return_value.gpu = float('nan')
            with self.assertRaisesRegex(ValueError, 'invalid native GPU'):
                runner._sample_ieee_gpu_utilization(a, device_uuid=device_uuid)


class RequestOwnershipLifetime(unittest.TestCase):
    def test_resolution_failure_releases_original_request_and_adapter_counts(self):
        runner, slot, trace, plan = fixture()
        runner._resolve_lora.side_effect = RuntimeError('artifact unavailable')
        with self.assertRaisesRegex(RuntimeError, 'artifact unavailable'):
            asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertEqual(slot.active_requests, 0)
        self.assertEqual(slot.active_adapter_counts, {})
        self.assertEqual(slot.inflight_request_deadlines, {})
        runner._notify_dispatch_capacity_changed.assert_awaited_once()

    def test_cancellation_during_resolution_releases_before_native_work(self):
        runner, slot, trace, plan = fixture()
        async def check():
            entered = asyncio.Event()
            async def wait_for_artifact(*args, **kwargs):
                entered.set()
                await asyncio.Future()
            runner._resolve_lora.side_effect = wait_for_artifact
            task = asyncio.create_task(runner._exec_request(trace, 4, 0., request_plan=plan))
            await entered.wait()
            self.assertEqual(slot.active_requests, 1)
            self.assertEqual(slot.active_adapter_counts, {'adapter-a': 1})
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(check())
        self.assertEqual(slot.active_requests, 0)
        self.assertFalse(slot.active_adapter_counts)
        slot.engine.generate_prepared.assert_not_awaited()

    def test_failure_immediately_after_reserve_does_not_leak_capacity(self):
        runner, slot, trace, plan = fixture()
        runner._release_live_waiting_trace.side_effect = RuntimeError('invalid arrival ownership')
        with self.assertRaisesRegex(RuntimeError, 'invalid arrival ownership'):
            asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertEqual(slot.active_requests, 0)
        self.assertFalse(slot.active_adapter_counts)

    def test_release_uses_original_adapter_even_if_resolution_changes_identity(self):
        runner, slot, trace, plan = fixture()
        runner._resolve_lora.return_value = (None, None, 0., 'backbone', 0., 0.)
        # This legacy behavior must not leak adapter-a's count. Native fixed-work
        # qualification still rejects actual adapter-to-backbone substitution.
        asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertEqual(slot.active_requests, 0)
        self.assertFalse(slot.active_adapter_counts)

    def test_completion_after_inference_updates_batch_pressure_exactly_once(self):
        runner, slot, trace, plan = fixture()
        coordinator = SimpleNamespace(notify_batch_start=Mock(), notify_batch_end=Mock())
        slot.coordinator = coordinator
        # A post-generation result-processing error still has one batch lifetime.
        slot.engine.generate_prepared.return_value = ('invalid', 2., 4, {})
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertFalse(result.success)
        coordinator.notify_batch_start.assert_called_once_with(2, 4)
        coordinator.notify_batch_end.assert_called_once_with(2, 4)
        self.assertEqual(slot.active_requests, 0)

    def test_legacy_generation_cancel_also_closes_batch_ownership(self):
        runner, slot, trace, plan = fixture()
        coordinator = SimpleNamespace(notify_batch_start=Mock(), notify_batch_end=Mock())
        slot.coordinator = coordinator
        async def check():
            entered = asyncio.Event()
            async def generate(**kwargs):
                entered.set()
                await asyncio.Future()
            slot.engine.generate_prepared.side_effect = generate
            task = asyncio.create_task(runner._exec_request(trace, 4, 0., request_plan=plan))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(check())
        coordinator.notify_batch_end.assert_called_once_with(2, 4)
        self.assertEqual(slot.active_requests, 0)

    def test_native_cancel_without_terminal_does_not_fabricate_free_capacity(self):
        runner, slot, trace, plan = fixture()
        runner.model_cfg['timing_contract'] = 'ieee_tc_native_v1'
        async def check():
            entered = asyncio.Event()
            async def generate(**kwargs):
                entered.set()
                await asyncio.Future()
            slot.engine.generate_prepared.side_effect = generate
            task = asyncio.create_task(runner._exec_request(trace, 4, 0., request_plan=plan))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(check())
        self.assertEqual(slot.active_requests, 1)
        self.assertEqual(slot.active_adapter_counts, {'adapter-a': 1})
        self.assertEqual(slot.status, 'draining')
        self.assertEqual(runner._try_reserve_runtime_request_slot(slot, 'adapter-a'), (False, False))
        self.assertFalse(runner._slot_can_accept_runtime_request(slot, 'adapter-a'))
        pending = runner._unsettled_runtime_reservations['req-a']
        self.assertFalse(pending.released)
        self.assertFalse(pending.native_terminal_observed)
        result = ScenarioResult('case', 'faaslora_full', total=1)
        runner._attach_control_path_background_metrics(result)
        self.assertFalse(result.runtime_request_ownership['all_native_requests_settled'])
        self.assertEqual(result.runtime_request_ownership['unsettled'][0]['request_id'], 'req-a')
        combined = aggregate_runs([result, result])
        self.assertEqual(combined.runtime_request_ownership['runs'],
                         [result.runtime_request_ownership, result.runtime_request_ownership])

    def test_shared_last_timing_cannot_supply_a_native_terminal(self):
        runner, slot, trace, plan = fixture()
        runner.model_cfg['timing_contract'] = 'ieee_tc_native_v1'
        slot.engine.last_timing = {'native_terminal_observed': True}
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertFalse(result.success)
        self.assertIn('terminal acknowledgement', result.error)
        self.assertEqual(slot.active_requests, 1)
        self.assertIn('req-a', runner._unsettled_runtime_reservations)

    def test_release_is_idempotent_but_counter_underflow_is_not_hidden(self):
        runner, slot, trace, plan = fixture()
        reservation = RuntimeRequestReservation('r')
        reserved, adapter_reserved = runner._try_reserve_runtime_request_slot(slot, 'adapter-a')
        self.assertTrue(reserved)
        reservation.bind(slot, 'adapter-a', adapter_reserved)
        with self.assertRaisesRegex(RuntimeError, 'rebound'):
            reservation.bind(slot, 'adapter-a', adapter_reserved)
        asyncio.run(runner._finish_runtime_request_reservation(reservation))
        asyncio.run(runner._finish_runtime_request_reservation(reservation))
        self.assertEqual(slot.active_requests, 0)
        runner._notify_dispatch_capacity_changed.assert_awaited_once()
        impossible = RuntimeRequestReservation('another')
        impossible.bind(slot, 'adapter-a', True)
        with self.assertRaisesRegex(RuntimeError, 'underflow'):
            asyncio.run(runner._finish_runtime_request_reservation(impossible))

    def test_failed_adapter_reservation_does_not_increment_request_count(self):
        runner, slot, trace, plan = fixture()
        slot.begin_active_adapter = Mock(side_effect=ValueError('invalid adapter counter'))
        with self.assertRaisesRegex(ValueError, 'invalid adapter counter'):
            runner._try_reserve_runtime_request_slot(slot, 'adapter-a')
        self.assertEqual(slot.active_requests, 0)
        self.assertFalse(slot.active_adapter_counts)

    def test_fixed_output_resolution_cannot_turn_into_backbone_inference(self):
        runner, slot, trace, plan = fixture()
        runner.model_cfg['generation_contract'] = 'fixed_length_greedy_v1'
        runner._resolve_lora.return_value = (None, None, 0., 'backbone', 0., 0.)
        with self.assertRaisesRegex(ValueError, 'changed or lost'):
            asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        slot.engine.generate_prepared.assert_not_awaited()
        self.assertEqual(slot.active_requests, 0)
        self.assertFalse(slot.active_adapter_counts)

    def test_cancelling_one_of_two_same_adapter_requests_releases_only_its_share(self):
        runner, slot, trace, plan = fixture()
        async def check():
            entered = asyncio.Event()
            started = 0
            async def wait_for_artifact(*args, **kwargs):
                nonlocal started
                started += 1
                if started == 2:
                    entered.set()
                await asyncio.Future()
            runner._resolve_lora.side_effect = wait_for_artifact
            other = SimpleNamespace(**{**vars(trace), 'request_id': 'req-b'})
            tasks = [asyncio.create_task(runner._exec_request(t, 4, 0., request_plan=plan))
                     for t in (trace, other)]
            await entered.wait()
            self.assertEqual(slot.active_requests, 2)
            self.assertEqual(slot.active_adapter_counts, {'adapter-a': 2})
            for remaining, task in zip((1, 0), tasks):
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(slot.active_requests, remaining)
                self.assertEqual(slot.active_adapter_counts, {'adapter-a': 1} if remaining else {})
        asyncio.run(check())
        self.assertFalse(slot.inflight_request_deadlines)

    def test_observed_native_terminal_releases_even_when_result_analysis_fails(self):
        runner, slot, trace, plan = fixture()
        runner.model_cfg['timing_contract'] = 'ieee_tc_native_v1'
        slot.engine.generate_prepared.return_value = (10., 2., 4, {'native_terminal_observed': True})
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertFalse(result.success)  # Missing timing; not a valid performance sample.
        self.assertEqual(slot.active_requests, 0)
        self.assertFalse(slot.active_adapter_counts)
        self.assertFalse(runner._unsettled_runtime_reservations)
        self.assertTrue(result.failure_observation['native_terminal_observed'])
        self.assertIsNone(result.ttft_ms)
        self.assertFalse(result.output_contract_match)

    def test_native_generation_failure_keeps_input_and_dispatch_but_not_fake_latency(self):
        runner, slot, trace, plan = fixture()
        runner.model_cfg['timing_contract'] = 'ieee_tc_native_v1'
        runner._generation_contract = 'fixed_length_greedy_v1'
        slot.engine.generate_prepared.side_effect = RuntimeError('native stream failed')
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertFalse(result.success)
        self.assertEqual(result.request_id, trace.request_id)
        self.assertEqual(result.adapter_id, trace.adapter_id)
        self.assertEqual(result.requested_completion_tokens, 4)
        self.assertEqual(result.canonical_prompt_sha256, hashlib.sha256(b'hello').hexdigest())
        self.assertEqual(result.timing_contract, 'ieee_tc_native_v1')
        self.assertTrue(result.readiness_tier_before_dispatch)
        for field in ('ttft_ms', 'tpot_ms', 'e2e_ms', 'cost_usd', 'output_tokens',
                      'overall_ttft_ms', 'service_ttft_ms', 'completed_offset_s'):
            self.assertIsNone(getattr(result, field))
        self.assertFalse(result.output_contract_match)
        self.assertFalse(result.failure_observation['native_terminal_observed'])
        self.assertEqual(slot.active_requests, 1)


def native_reference_fixture():
    from faaslora.clock import local_monotonic_clock_id
    from faaslora.memory.residency_manager import IEEEBackendGPUReferences
    from tests.test_ieee_tc_gpu_references import NativeManager, NativeAdapter
    runner, slot, trace, plan = fixture()
    runner.model_cfg.update(timing_contract='ieee_tc_native_v1', ieee_gpu_references=True)
    manager = NativeManager()
    for aid in (1, 2, 3):
        manager.remove_adapter(aid)
    def load(**kwargs):
        aid = kwargs['adapter_int_id']
        if aid not in manager._registered_adapters:
            manager._registered_adapters[aid] = NativeAdapter()
        manager.activate(aid)
    owner = IEEEBackendGPUReferences(manager, Mock(), demand_loader=load)
    async def rpc(*, operation, **kwargs):
        return {**getattr(owner, operation)(**kwargs), 'clock_id': local_monotonic_clock_id()}
    slot.engine.ieee_gpu_reference = AsyncMock(side_effect=rpc)
    return runner, slot, trace, plan, owner, rpc


class SelectedSourceAdmissionIntegration(unittest.TestCase):
    """Real request/router/owner and event path; tiny fixtures, no GPU timing claim."""
    def bind_preparation_fixture(self, runner, slot):
        from faaslora.preloading.preloading_planner import PreparationClass, FrozenPreparationProfiles
        content = runner._ieee_artifact_identities['adapter-a']['content_sha256']
        keys = [PreparationClass(tier, representation, 'exact_content_sha256:'+content, 0)
                for tier, representation in (
                    ('host', 'native_cpu_dense_ab_v1:torch.float16:unpinned'),
                    ('host', 'verified_regular_file_tree_v1'),
                    ('nvme', 'verified_regular_file_tree_v1'),
                    ('remote', 'tar_gzip_verified_file_tree_v1'))]
        profile = FrozenPreparationProfiles((), {key: 10. for key in keys},
            {key: 1 for key in keys}, 'fixture-not-measured-profile', ('fixture-run',), .5, '{}')
        runner._preparation_profiles = profile
        slot.preparation_cost_model = profile.new_replica()
        return profile

    def build(self, tier='gpu'):
        from collections import defaultdict
        from faaslora.clock import local_monotonic_clock_id
        from faaslora.experiment.instance_pool import (Router, ServiceClassBins, ServiceCostModel,
                                                     ServiceComponents, ServiceObservationClass)
        from faaslora.memory.residency_manager import ResidencyManager
        from faaslora.preloading.preloading_manager import PreloadingManager
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        host, nvme = root / 'host', root / 'nvme'
        host.mkdir()
        nvme.mkdir()
        manager = ResidencyManager({'memory': {'host': {'cache_dir': str(host)},
            'nvme': {'cache_dir': str(nvme)}}}, Mock(), Mock())
        runner._stack = SimpleNamespace(residency_manager=manager, record_access=Mock(),
            preloading_manager=PreloadingManager({}, Mock(), manager, Mock()))
        runner._routing_policy = 'ieee_confirmed'
        runner._ieee_routing_epoch = 0
        runner._access_count = defaultdict(int)
        runner._generation_contract = 'fixed_length_greedy_v1'
        runner._ieee_artifact_identities = {trace.adapter_id: dict(adapter_id=trace.adapter_id,
            rank=8, content_sha256='a'*64, remote_payload_bytes=8192,
            remote_representation='tar_gzip_verified_file_tree_v1')}
        runner.instance_pool = SimpleNamespace(get_slots=lambda: [slot])
        runner.router = Router(runner.instance_pool, 'ieee_confirmed', service_bin_ms=10.)
        runner._sample_ieee_gpu_utilization = Mock(return_value=dict(gpu_utilization_pct=0.))
        runner._resolve_lora.side_effect = AssertionError('IEEE must not use legacy resolve/admission')
        runner._remote_materialize_locks = defaultdict(asyncio.Lock)
        runner.nvme_dir = nvme
        runner._nvme_cache = {}
        aid = InferenceEngine._lora_int_id(trace.adapter_id)
        if tier in ('gpu', 'host'):
            ControllerNativeReferenceLifecycle.preload_native(owner)
            if tier == 'host':
                owner.manager.deactivate(aid)
        representations = dict(gpu='native_gpu_dense_slot_v1:torch.float16',
            host='native_cpu_dense_ab_v1:torch.float16:unpinned',
            nvme='verified_regular_file_tree_v1', remote='tar_gzip_verified_file_tree_v1')
        slot.service_class_bins = ServiceClassBins((), (), (), (), (1,))
        profiles = {ServiceObservationClass(t, 0, 0, 0, 0, representation, count):
                    ServiceComponents(0. if t == 'gpu' else 100., 200., 300.)
                    for t, representation in representations.items() for count in (0, 1)}
        profiles.update({ServiceObservationClass('host', 0, 0, 0, 0,
            'verified_regular_file_tree_v1', count): ServiceComponents(100., 200., 300.) for count in (0, 1)})
        slot.service_cost_model = ServiceCostModel(profiles, beta=.5, profile_id='test-fixtures-not-profile-data')

        async def observed(*, operation, **kwargs):
            value = await rpc(operation=operation, **kwargs)
            if operation == 'source_snapshot':
                ids, slots = value['registered_cpu_adapter_ids'], value['slot_adapter_ids']
                value['device_uuid'] = 'GPU-00010203-0405-0607-0809-0a0b0c0d0e0f'
                value['native_footprints'] = dict(uniform_slot_layout=True,
                    host_footprint_scope='native_registered_tensor_storage_capacity', host_budget_reserved=False,
                    host_allocator_overhead_included=False, slot_adapter_ids=slots,
                    registered_cpu_adapter_ids=ids, slot_capacity_bytes=1024, pool_allocated_bytes=2048,
                    host_tensor_storage_bytes=512*len(ids),
                    host_allocations=[dict(allocation_id=i, allocated_bytes=512, adapter_ids=[a], pinned=False)
                                      for i, a in enumerate(ids)],
                    host_adapter_footprints=[dict(adapter_int_id=a, allocation_ids=[i], storage_bytes=512,
                        exclusive_storage_bytes=512, dtypes=['torch.float16'],
                        representation='native_cpu_dense_ab_v1', has_packed_modules=False) for i, a in enumerate(ids)],
                    pool_tensor_views=[dict(dtype='torch.float16')])
            return value
        slot.engine.ieee_gpu_reference.side_effect = observed

        async def generate(**kwargs):
            from tests.test_ieee_tc_service_events import event
            reference = kwargs['gpu_reference']
            receive = kwargs['native_event_observer']
            dispatch = time.monotonic()
            owner.begin_use(lease_id=reference['lease_id'], expected_owner_id=owner.owner_id,
                adapter_int_id=aid, backend_request_id='native-test', lora_name=trace.adapter_id,
                lora_path=reference['lora_path'])
            first = time.monotonic()
            fields = dict(adapter_id=trace.adapter_id, native_clock_id=local_monotonic_clock_id(),
                backend_request_id='native-test', gpu_reference_owner_id=owner.owner_id,
                gpu_reference_lease_id=reference['lease_id'], gpu_reference_adapter_int_id=aid)
            receive(event(**fields, timestamp_monotonic_s=first))
            await asyncio.sleep(0)
            last = time.monotonic()
            receive(event(2, **fields, token_count=4, timestamp_monotonic_s=last))
            owner.end_use(lease_id=reference['lease_id'], expected_owner_id=owner.owner_id,
                backend_request_id='native-test')
            timing = dict(timing_contract='ieee_tc_native_v1', native_clock_id=local_monotonic_clock_id(),
                native_terminal_observed=True, native_dispatch_monotonic_s=dispatch,
                native_queued_monotonic_s=dispatch, native_scheduled_monotonic_s=dispatch,
                native_first_token_monotonic_s=first, native_last_token_monotonic_s=last,
                worker_completed_monotonic_s=time.monotonic(), native_output_tokens=4,
                native_tpot_ms=(last-first)*1000/3,
                gpu_reference_owner_id=owner.owner_id, gpu_reference_lease_id=reference['lease_id'],
                gpu_reference_adapter_int_id=aid)
            return (first-dispatch)*1000, (last-first)*1000/3, 4, timing
        slot.engine.generate_prepared.side_effect = generate
        return runner, slot, trace, plan, owner, observed

    def test_explicit_profile_measurement_reuses_actual_admission_without_cost_defaults(self):
        for tier in ('gpu', 'host'):
            runner, slot, trace, plan, owner, _ = self.build(tier)
            protect = runner._ieee_protect_selected_source
            async def collect(reservation, source, key):
                return await protect(reservation, source, key, collect_profile_only=True)
            runner._ieee_protect_selected_source = collect
            result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
            self.assertTrue(result.success, result.error)
            evidence = result.gpu_reference_evidence
            self.assertTrue(evidence['source_admission']['profile_collection_only'])
            self.assertEqual(evidence['service_intervals']['service_class']['tier'], tier)
            self.assertEqual(len(evidence['service_events']), 2)
            for key in slot.service_cost_model._profiles:
                self.assertEqual(slot.service_cost_model.sample_counts(key), dict(d_ms=0, t_ms=0, o_ms=0))
            self.assertEqual(owner.snapshot()['live_leases'], 0)
            self.assertEqual(owner.snapshot()['live_host_source_leases'], 0)

    def profile_case(self, runner, slot, trace, plan, source):
        slot.engine._lora_int_id = InferenceEngine._lora_int_id
        slot.engine.prepare_request = Mock(return_value=plan)
        # This fixture supplies synthetic inference only, never production data.
        return dict(case_id='profile-case', source_request_id=trace.request_id,
            adapter_id=trace.adapter_id, source=source, source_row_sha256='b'*64,
            row=dict(expected_output_tokens=4, expected_input_tokens=2,
                     body=dict(messages=[dict(role='user', content='hello')])) )

    def test_source_profile_collector_covers_all_five_observed_representations(self):
        from scripts.ieee_tc_preflight import collect_native_source_wave
        for source in ('gpu', 'native_host', 'file_host', 'nvme', 'remote'):
            with self.subTest(source=source):
                runner, slot, trace, plan, owner, _ = self.build(
                    'host' if source == 'native_host' else 'gpu' if source == 'gpu' else 'remote')
                if source in ('file_host', 'nvme', 'remote'):
                    self.file_source(runner, trace, publish=source != 'remote', host=source == 'file_host')
                case = self.profile_case(runner, slot, trace, plan, source)
                result = dict(requests=[])
                asyncio.run(collect_native_source_wave(runner, slot, [case], result))
                sample = result['requests'][0]
                self.assertTrue(sample['pass'])
                self.assertTrue(sample['reservation_released'])
                self.assertEqual(sample['requested_source'], source)
                self.assertEqual(sample['actual_tokens'], 4)
                self.assertEqual(sample['class_features']['admitted_after_accept'], 1)
                self.assertEqual(sample['class_features']['representation'],
                                 sample['service_class']['representation'])
                self.assertTrue(sample['source_evidence']['source_admission']['profile_collection_only'])
                self.assertEqual(slot.active_requests, 0)
                self.assertEqual(owner.snapshot()['live_leases'], 0)
                self.assertEqual(owner.snapshot()['live_host_source_leases'], 0)
                for key in slot.service_cost_model._profiles:
                    self.assertEqual(slot.service_cost_model.sample_counts(key), dict(d_ms=0, t_ms=0, o_ms=0))

    def test_profile_source_change_is_preserved_not_primed_or_relabelled(self):
        from scripts.ieee_tc_preflight import collect_native_source_wave
        runner, slot, trace, plan, owner, _ = self.build('gpu')
        case = self.profile_case(runner, slot, trace, plan, 'remote')
        result = dict(requests=[])
        with self.assertRaisesRegex(RuntimeError, 'controlled source changed'):
            asyncio.run(collect_native_source_wave(runner, slot, [case], result))
        self.assertFalse(result['requests'][0]['pass'])
        self.assertTrue(result['requests'][0]['reservation_released'])
        slot.engine.generate_prepared.assert_not_awaited()
        self.assertEqual(slot.active_requests, 0)
        self.assertEqual(owner.snapshot()['live_leases'], 0)

    def test_profile_wrong_output_fails_with_native_reference_cleanup(self):
        from scripts.ieee_tc_preflight import collect_native_source_wave
        runner, slot, trace, plan, owner, _ = self.build('gpu')
        case = self.profile_case(runner, slot, trace, plan, 'gpu')
        generate = slot.engine.generate_prepared.side_effect
        async def wrong(**kwargs):
            value = await generate(**kwargs)
            return value[:2]+(3, value[3])
        slot.engine.generate_prepared.side_effect = wrong
        result = dict(requests=[])
        with self.assertRaisesRegex(RuntimeError, 'matching native completion'):
            asyncio.run(collect_native_source_wave(runner, slot, [case], result))
        self.assertFalse(result['requests'][0]['pass'])
        self.assertTrue(result['requests'][0]['reservation_released'])
        self.assertEqual(owner.snapshot()['live_leases'], 0)

    def test_profile_known_stale_view_retires_full_reservation_before_reselection(self):
        from scripts.ieee_tc_preflight import collect_native_source_wave
        runner, slot, trace, plan, owner, _ = self.build('gpu')
        case = self.profile_case(runner, slot, trace, plan, 'gpu')
        observed = slot.engine.ieee_gpu_reference.side_effect
        attempts = []
        async def concurrent_epoch(*, operation, **kwargs):
            if operation == 'demand_load_and_acquire':
                attempts.append(kwargs['lease_id'])
                self.assertEqual(slot.active_requests, 1)
                if len(attempts) <= 2:
                    owner.epoch += 1  # Controlled unrelated native transition.
            return await observed(operation=operation, **kwargs)
        slot.engine.ieee_gpu_reference.side_effect = concurrent_epoch
        result = dict(requests=[])
        asyncio.run(collect_native_source_wave(runner, slot, [case], result))
        sample = result['requests'][0]
        self.assertTrue(sample['pass'])
        self.assertEqual(len(attempts), 3)
        self.assertEqual(len(set(attempts)), 3)
        self.assertEqual(len(sample['rejected_source_views']), 2)
        for rejected in sample['rejected_source_views']:
            self.assertTrue(rejected['reservation_released'])
            self.assertEqual(rejected['source_evidence']['state'], 'rejected')
            self.assertFalse(rejected['source_evidence']['last_conflict']['acquired'])
        self.assertTrue(sample['reservation_released'])
        self.assertEqual(slot.active_requests, 0)
        self.assertFalse(runner._unsettled_runtime_reservations)
        self.assertEqual(owner.snapshot()['live_leases'], 0)

    def test_profile_rejects_duplicate_misses_before_any_native_operation(self):
        from scripts.ieee_tc_preflight import collect_native_source_wave
        runner, slot, trace, plan, owner, _ = self.build('remote')
        case = self.profile_case(runner, slot, trace, plan, 'remote')
        with self.assertRaisesRegex(ValueError, 'distinct existing adapters'):
            asyncio.run(collect_native_source_wave(runner, slot, [case, case], dict(requests=[])))
        slot.engine.ieee_gpu_reference.assert_not_awaited()

    def test_profile_full_pending_intent_precedes_protection_and_reaches_generation(self):
        from scripts.ieee_tc_preflight import collect_native_source_wave
        from faaslora.clock import local_monotonic_clock_id
        runner, slot, trace, plan, owner, _ = self.build('host')
        runner.model_cfg['ieee_admission_profile'] = {'test_only': True}
        case = self.profile_case(runner, slot, trace, plan, 'native_host')
        events, intents = [], []
        original_rpc = slot.engine.ieee_gpu_reference.side_effect
        original_generate = slot.engine.generate_prepared.side_effect

        async def register(**kwargs):
            self.assertEqual((kwargs['prompt'], kwargs['max_tokens'], kwargs['adapter_id']),
                             (plan.prompt, plan.max_tokens, trace.adapter_id))
            events.append('register')
            intents.append(kwargs['intent_id'])
            return dict(kind='ieee_pending_admission_v1', intent_id=kwargs['intent_id'],
                state='pending', clock_id=local_monotonic_clock_id(), physical_kv_reservation=False)

        async def source(**kwargs):
            if kwargs['operation'] == 'hold_host_source':
                self.assertEqual(events, ['register'])
                self.assertIn(case['case_id'], slot.ieee_pending_load_ids)
                events.append('protect')
            return await original_rpc(**kwargs)

        async def generate(**kwargs):
            self.assertEqual(kwargs['pending_admission_id'], intents[0])
            self.assertFalse(slot.ieee_pending_load_ids)
            events.append('generate')
            return await original_generate(**kwargs)

        async def close(**kwargs):
            self.assertEqual(kwargs['intent_id'], intents[0])
            self.assertEqual(slot.active_requests, 1)
            events.append('close')
            return dict(intent_id=kwargs['intent_id'], closed=True)

        slot.engine.ieee_register_pending = AsyncMock(side_effect=register)
        slot.engine.ieee_close_pending = AsyncMock(side_effect=close)
        slot.engine.ieee_gpu_reference.side_effect = source
        slot.engine.generate_prepared.side_effect = generate
        result = dict(requests=[])
        asyncio.run(collect_native_source_wave(runner, slot, [case], result))
        self.assertEqual(events, ['register', 'protect', 'generate', 'close'])
        self.assertTrue(result['requests'][0]['pass'])
        self.assertEqual(result['requests'][0]['source_evidence']['pending_kv_admission']['state'], 'closed')
        self.assertEqual(slot.active_requests, 0)
        self.assertFalse(slot.ieee_pending_load_ids)
        self.assertEqual(owner.snapshot()['live_leases'], 0)

    def test_profile_full_known_conflict_closes_old_pending_before_new_intent(self):
        from scripts.ieee_tc_preflight import collect_native_source_wave
        from faaslora.clock import local_monotonic_clock_id
        runner, slot, trace, plan, owner, _ = self.build('gpu')
        runner.model_cfg['ieee_admission_profile'] = {'test_only': True}
        case = self.profile_case(runner, slot, trace, plan, 'gpu')
        original_rpc = slot.engine.ieee_gpu_reference.side_effect
        original_generate = slot.engine.generate_prepared.side_effect
        active, registered, closed = set(), [], []
        attempts = []

        async def register(**kwargs):
            self.assertFalse(active)
            intent = kwargs['intent_id']
            active.add(intent)
            registered.append(intent)
            return dict(kind='ieee_pending_admission_v1', intent_id=intent, state='pending',
                clock_id=local_monotonic_clock_id(), physical_kv_reservation=False)

        async def source(**kwargs):
            if kwargs['operation'] == 'demand_load_and_acquire':
                self.assertEqual(len(active), 1)
                attempts.append(kwargs['lease_id'])
                if len(attempts) == 1:
                    owner.epoch += 1
            return await original_rpc(**kwargs)

        async def generate(**kwargs):
            self.assertEqual(kwargs['pending_admission_id'], registered[-1])
            return await original_generate(**kwargs)

        async def close(**kwargs):
            intent = kwargs['intent_id']
            active.remove(intent)
            closed.append(intent)
            return dict(intent_id=intent, closed=True)

        slot.engine.ieee_register_pending = AsyncMock(side_effect=register)
        slot.engine.ieee_close_pending = AsyncMock(side_effect=close)
        slot.engine.ieee_gpu_reference.side_effect = source
        slot.engine.generate_prepared.side_effect = generate
        result = dict(requests=[])
        asyncio.run(collect_native_source_wave(runner, slot, [case], result))
        self.assertEqual(len(registered), 2)
        self.assertEqual(len(set(registered)), 2)
        self.assertEqual(closed, registered)
        self.assertFalse(active)
        sample = result['requests'][0]
        self.assertTrue(sample['pass'])
        rejected = sample['rejected_source_views'][0]
        self.assertTrue(rejected['reservation_released'])
        self.assertEqual(rejected['source_evidence']['pending_kv_admission']['state'], 'closed')
        self.assertEqual(slot.active_requests, 0)
        self.assertFalse(runner._unsettled_runtime_reservations)

    def test_profile_lost_pending_reply_does_not_generate_or_release_unknown_ownership(self):
        from scripts.ieee_tc_preflight import collect_native_source_wave
        runner, slot, trace, plan, owner, _ = self.build('gpu')
        runner.model_cfg['ieee_admission_profile'] = {'test_only': True}
        case = self.profile_case(runner, slot, trace, plan, 'gpu')
        slot.engine.ieee_register_pending = AsyncMock(side_effect=RuntimeError('lost pending reply'))
        slot.engine.ieee_close_pending = AsyncMock(side_effect=RuntimeError('pending owner unresolved'))
        result = dict(requests=[])
        with self.assertRaisesRegex(RuntimeError, 'pending owner unresolved'):
            asyncio.run(collect_native_source_wave(runner, slot, [case], result))
        sample = result['requests'][0]
        self.assertFalse(sample['pass'])
        self.assertIn('lost pending reply', sample['error'])
        self.assertIn('pending owner unresolved', sample['cleanup_error'])
        slot.engine.generate_prepared.assert_not_awaited()
        self.assertEqual(slot.active_requests, 1)
        self.assertIn(case['case_id'], runner._unsettled_runtime_reservations)
        self.assertEqual(owner.snapshot()['live_leases'], 0)

    def test_profile_concurrent_wave_joins_sibling_even_when_one_output_is_wrong(self):
        from scripts.ieee_tc_preflight import collect_native_source_wave
        from faaslora.clock import local_monotonic_clock_id
        from tests.test_ieee_tc_service_events import event
        runner, slot, trace, plan, owner, _ = self.build('gpu')
        first = self.profile_case(runner, slot, trace, plan, 'gpu')
        second = {**first, 'case_id': 'profile-second', 'adapter_id': 'adapter-b'}
        runner._ieee_artifact_identities['adapter-b'] = {
            **runner._ieee_artifact_identities[trace.adapter_id], 'adapter_id': 'adapter-b'}
        state = owner.snapshot()
        receipt = owner.demand_load_and_acquire(lease_id='setup-b',
            adapter_int_id=InferenceEngine._lora_int_id('adapter-b'), lora_name='adapter-b',
            lora_path='/existing/a', expected_owner_id=state['owner_id'], expected_epoch=state['epoch'])
        owner.release(lease_id=receipt['lease_id'], expected_owner_id=owner.owner_id)
        result = dict(requests=[])
        async def run():
            arrived, barrier = [], asyncio.Event()
            async def generate(**kwargs):
                reference, aid = kwargs['gpu_reference'], kwargs['adapter_id']
                arrived.append(aid)
                if len(arrived) == 2:
                    barrier.set()
                await asyncio.wait_for(barrier.wait(), .5)
                first_at = time.monotonic()
                fields = dict(adapter_id=aid, native_clock_id=local_monotonic_clock_id(),
                    backend_request_id=aid, gpu_reference_owner_id=owner.owner_id,
                    gpu_reference_lease_id=reference['lease_id'],
                    gpu_reference_adapter_int_id=reference['adapter_int_id'])
                kwargs['native_event_observer'](event(**fields, timestamp_monotonic_s=first_at))
                last_at = time.monotonic()
                kwargs['native_event_observer'](event(2, **fields, token_count=4, timestamp_monotonic_s=last_at))
                return 0., 0., 3 if aid == trace.adapter_id else 4, dict(
                    native_clock_id=local_monotonic_clock_id(), native_terminal_observed=True,
                    native_first_token_monotonic_s=first_at, native_last_token_monotonic_s=last_at)
            slot.engine.generate_prepared.side_effect = generate
            with self.assertRaisesRegex(RuntimeError, 'matching native completion'):
                await collect_native_source_wave(runner, slot, [first, second], result)
            self.assertEqual(set(arrived), {'adapter-a', 'adapter-b'})
        asyncio.run(run())
        self.assertEqual(len(result['requests']), 2)
        self.assertEqual(sum(row['pass'] for row in result['requests']), 1)
        self.assertTrue(all(row['reservation_released'] for row in result['requests']))
        self.assertEqual(slot.active_requests, 0)
        self.assertEqual(owner.snapshot()['live_leases'], 0)
        self.assertIn(2, [row['class_features']['admitted_after_accept'] for row in result['requests']])

    def test_profile_matrix_executes_real_file_setup_and_native_eviction_between_waves(self):
        from scripts.ieee_tc_preflight import qualify_native_source_matrix
        runner, slot, trace, plan, owner, _ = self.build('remote')
        self.file_source(runner, trace)
        base = self.profile_case(runner, slot, trace, plan, 'remote')
        waves = [[{**base, 'case_id': 'profile-'+source, 'source': source}]
                 for source in ('remote', 'nvme', 'file_host', 'gpu', 'remote')]
        result = dict(requests=[])
        asyncio.run(qualify_native_source_matrix(
            slot.engine, runner, slot.service_class_bins, waves, result))
        self.assertTrue(all(row['pass'] for row in result['requests']))
        self.assertEqual([row['requested_source'] for row in result['requests']],
                         ['remote', 'nvme', 'file_host', 'gpu', 'remote'])
        self.assertEqual(len(result['profile_waves']), 5)
        self.assertEqual(len(runner._remote_transfer_evidence), 5)
        self.assertEqual(owner.snapshot()['live_leases'], 0)
        self.assertTrue(all(row['complete'] for row in result['profile_waves']))
        self.assertEqual(result['profile_waves'][0]['native_evictions'], {})
        self.assertTrue(all(row['native_evictions'] for row in result['profile_waves'][1:]))

    def test_actual_request_loads_update_the_admission_fixed_preparation_class(self):
        for tier in ('gpu', 'native_host', 'host', 'nvme', 'remote'):
            with self.subTest(tier=tier):
                runner, slot, trace, plan, owner, _ = self.build(
                    'host' if tier == 'native_host' else 'gpu' if tier == 'gpu' else 'remote')
                if tier in ('host', 'nvme', 'remote'):
                    self.file_source(runner, trace, publish=tier != 'remote', host=tier == 'host')
                profile = self.bind_preparation_fixture(runner, slot)
                result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
                self.assertTrue(result.success, result.error)
                admission = result.gpu_reference_evidence['source_admission']
                self.assertGreater(admission['source']['footprint_bytes'], 0)
                self.assertEqual(admission['source']['representation'], admission['service_class']['representation'])
                if tier == 'gpu':
                    self.assertNotIn('preparation_interval', result.gpu_reference_evidence)
                    self.assertEqual(slot.preparation_cost_model.snapshot()[0], 0)
                    continue
                key = profile.classify_source(admission['source'])
                interval = result.gpu_reference_evidence['preparation_interval']
                self.assertTrue(interval['cost_model_updated'])
                self.assertEqual(interval['preparation_class'], asdict(key))
                self.assertEqual(slot.preparation_cost_model.estimate(key), (10.+interval['d_ms'])/2)
                self.assertEqual(slot.preparation_cost_model.snapshot()[0], 1)
                for other in profile.profiles:
                    if other != key:
                        self.assertEqual(slot.preparation_cost_model.estimate(other), 10.)
                self.assertEqual(owner.snapshot()['live_leases'], 0)
                self.assertEqual(owner.snapshot()['live_host_source_leases'], 0)

    def test_unknown_preparation_class_fails_before_holding_or_loading_source(self):
        runner, slot, trace, plan, owner, _ = self.build('host')
        profile = self.bind_preparation_fixture(runner, slot)
        runner._preparation_profiles = replace(profile, profiles={
            key: value for key, value in profile.profiles.items() if key.tier == 'remote'})
        with self.assertRaisesRegex(KeyError, 'no measured preparation profile'):
            asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        slot.engine.generate_prepared.assert_not_awaited()
        self.assertEqual(owner.snapshot()['live_leases'], 0)
        self.assertEqual(owner.snapshot()['live_host_source_leases'], 0)
        calls = [call.kwargs.get('operation') for call in slot.engine.ieee_gpu_reference.await_args_list]
        self.assertNotIn('hold_host_source', calls)
        self.assertNotIn('demand_load_and_acquire', calls)

    def test_actual_completed_load_cost_reaches_next_stack_planning_epoch(self):
        from faaslora.preloading.preloading_planner import PreparationClass, PreparationOption, PreloadingPlanner
        from faaslora.experiment.experiment_stack import ExperimentStack
        from faaslora.experiment.hotness_tracker import HotnessTracker
        from faaslora.registry.schema import StorageTier
        runner, slot, trace, plan, owner, _ = self.build('host')
        profile = self.bind_preparation_fixture(runner, slot)
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertTrue(result.success, result.error)
        # Controlled cache-fixture transition: fresh received HOST state after
        # releasing the actual request, not a stale post-load HOST assumption.
        owner.manager.deactivate(InferenceEngine._lora_int_id(trace.adapter_id))
        _, evidence = asyncio.run(runner._ieee_request_snapshot(trace, plan))
        source = evidence[slot.instance_id]['source']
        key = profile.classify_source(source)
        stack = ExperimentStack.__new__(ExperimentStack)
        stack.hotness_tracker = HotnessTracker(None)
        stack.hotness_tracker.record_arrival(trace.adapter_id)
        stack.preloading_planner = PreloadingPlanner.__new__(PreloadingPlanner)
        stack.preloading_planner.max_dp_buffer_bytes = 16*1024**2
        target = PreparationClass('gpu', 'native_gpu_dense_slot_v1:torch.float16', key.layout_id, 0)
        footprint = slot.native_source_state.sources[0].gpu_slot_capacity_bytes
        epoch = stack.plan_ieee_preparation(mode='handoff',
            options=[PreparationOption(trace.adapter_id, key, target, footprint)],
            budgets={StorageTier.GPU: footprint, StorageTier.HOST: 0, StorageTier.NVME: 0},
            costs=slot.preparation_cost_model, source_snapshot_id=str(source['epoch']))
        selected = epoch['selected']['gpu'][0]
        self.assertEqual(epoch['cost_sequence'], 1)
        self.assertEqual(selected.source_load_ms, slot.preparation_cost_model.estimate(key))
        self.assertEqual(selected.benefit_ms, selected.source_load_ms)
        self.assertFalse(epoch['physical_resources_reserved'])

    def test_actual_gpu_and_host_requests_use_admission_fixed_class_and_native_events(self):
        for tier in ('gpu', 'host'):
            with self.subTest(tier=tier):
                runner, slot, trace, plan, owner, _ = self.build(tier)
                result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
                self.assertTrue(result.success, result.error)
                evidence = result.gpu_reference_evidence
                intervals = evidence['service_intervals']
                self.assertEqual(intervals['service_class']['tier'], tier)
                d = intervals['acquired_monotonic_s'] - intervals['admitted_monotonic_s']
                self.assertEqual(d == 0., tier == 'gpu')
                if tier == 'host':
                    preparation = evidence['preparation_interval']
                    self.assertTrue(preparation['profile_eligible'])
                    self.assertLessEqual(preparation['d_ms'], d * 1000.)
                    self.assertAlmostEqual(preparation['d_ms'] + preparation['excluded_before_loading_ms'],
                                           d * 1000.)
                else:
                    self.assertNotIn('preparation_interval', evidence)
                self.assertEqual(result.readiness_tier_before_dispatch, tier)
                self.assertTrue(evidence['confirmed_dispatch_snapshot'])
                self.assertEqual(len(evidence['service_events']), 2)
                key = next(key for key in slot.service_cost_model._profiles
                    if asdict(key) == intervals['service_class'])
                self.assertEqual(slot.service_cost_model.sample_counts(key), dict(d_ms=1, t_ms=1, o_ms=1))
                self.assertEqual(slot.active_requests, 0)
                self.assertFalse(slot.active_adapter_counts)
                self.assertFalse(slot.ieee_pending_load_ids)
                self.assertEqual(owner.snapshot()['live_leases'], 0)
                self.assertEqual(owner.snapshot()['live_host_source_leases'], 0)
                runner._resolve_lora.assert_not_awaited()

    def test_changed_gpu_source_reselects_whole_router_and_does_not_record_gpu_zero(self):
        runner, slot, trace, plan, owner, rpc = self.build('gpu')
        attempts = []
        async def invalidate(*, operation, **kwargs):
            if operation == 'demand_load_and_acquire':
                attempts.append(kwargs)
                if len(attempts) == 1:
                    owner.manager.deactivate(InferenceEngine._lora_int_id(trace.adapter_id))
            return await rpc(operation=operation, **kwargs)
        slot.engine.ieee_gpu_reference.side_effect = invalidate
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertTrue(result.success, result.error)
        self.assertEqual(runner.router.selection_count, 2)
        self.assertEqual(result.readiness_tier_before_dispatch, 'host')
        history = result.gpu_reference_evidence['prior_routing_attempts']
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['state'], 'rejected')
        self.assertFalse(history[0]['last_conflict']['acquired'])
        for key in slot.service_cost_model._profiles:
            if key.tier == 'gpu':
                self.assertEqual(slot.service_cost_model.sample_counts(key), dict(d_ms=0, t_ms=0, o_ms=0))

    def test_admission_class_uses_live_post_accept_count_after_host_guard(self):
        runner, slot, trace, plan, owner, rpc = self.build('host')
        slot.active_requests = 1  # Another backbone request ends during native RPC.
        async def completing(*, operation, **kwargs):
            value = await rpc(operation=operation, **kwargs)
            if operation == 'hold_host_source':
                slot.active_requests -= 1
            return value
        slot.engine.ieee_gpu_reference.side_effect = completing
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertTrue(result.success, result.error)
        evidence = result.gpu_reference_evidence
        self.assertEqual(evidence['routing_snapshot']['candidates'][0]['service_class']['admitted_bin'], 1)
        self.assertEqual(evidence['source_admission']['service_class']['admitted_bin'], 0)

    def test_lost_host_hold_ack_keeps_controller_and_cpu_source_owned(self):
        runner, slot, trace, plan, owner, rpc = self.build('host')
        async def lost(*, operation, **kwargs):
            value = await rpc(operation=operation, **kwargs)
            if operation == 'hold_host_source':
                raise ConnectionError('test lost HOST acknowledgement')
            return value
        slot.engine.ieee_gpu_reference.side_effect = lost
        with self.assertRaises(ConnectionError):
            asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        pending = runner._unsettled_runtime_reservations[trace.request_id]
        self.assertIsNone(pending.ieee_observation)
        self.assertEqual(slot.active_requests, 1)
        self.assertEqual(slot.status, 'draining')
        self.assertEqual(owner.snapshot()['live_host_source_leases'], 1)
        self.assertFalse(owner.evict(adapter_int_id=InferenceEngine._lora_int_id(trace.adapter_id))['evicted'])

    def test_cancel_after_host_admission_does_not_invent_preparation_or_token_intervals(self):
        runner, slot, trace, plan, owner, _ = self.build('host')
        entered = asyncio.Event()
        async def wait_for_cancel(reservation):
            self.assertIsNotNone(reservation.ieee_observation)
            entered.set()
            await asyncio.Future()
        runner._ieee_prepare_selected_adapter = AsyncMock(side_effect=wait_for_cancel)
        async def run():
            task = asyncio.create_task(runner._exec_request(trace, 4, 0., request_plan=plan))
            await asyncio.wait_for(entered.wait(), .5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(run())
        self.assertEqual(slot.active_requests, 0)
        self.assertEqual(owner.snapshot()['live_host_source_leases'], 0)
        for key in slot.service_cost_model._profiles:
            self.assertEqual(slot.service_cost_model.sample_counts(key), dict(d_ms=0, t_ms=0, o_ms=0))

    def test_lost_host_release_ack_retains_gpu_and_request_until_reconciliation(self):
        runner, slot, trace, plan, owner, rpc = self.build('host')
        async def lost(*, operation, **kwargs):
            result = await rpc(operation=operation, **kwargs)
            if operation == 'release_host_source':
                raise ConnectionError('test lost HOST release acknowledgement')
            return result
        slot.engine.ieee_gpu_reference.side_effect = lost
        with self.assertRaises(ConnectionError):
            asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        reservation = runner._unsettled_runtime_reservations[trace.request_id]
        self.assertEqual(reservation.gpu_reference_evidence['native_host_source']['state'], 'release_pending')
        self.assertEqual(slot.active_requests, 1)
        self.assertEqual(owner.snapshot()['live_leases'], 1)
        self.assertEqual(owner.snapshot()['live_host_source_leases'], 0)
        self.assertIsNotNone(reservation.ieee_observation.acquired_at)
        self.assertIsNone(reservation.ieee_observation.first_at)

    def file_source(self, runner, trace, *, publish=False, host=False):
        from faaslora.storage.http_artifact_store import HttpArtifactStoreClient
        from tests.test_http_artifact_store import content_manifest, archive_bytes, SizedResponse
        from faaslora.registry.schema import StorageTier
        payload = {'adapter_config.json': b'{"r":8}', 'weights': b'fixture-not-a-model'}
        client = HttpArtifactStoreClient(endpoint='http://127.0.0.1:1')
        client.configure_content_manifest(content_manifest(artifact_id=trace.adapter_id, files=payload))
        client._opener = Mock()
        client._opener.open.side_effect = lambda *a, **kw: SizedResponse(archive_bytes(list(payload.items())))
        runner._remote_artifact_client = client
        runner._ieee_artifact_identities[trace.adapter_id] = dict(client.routing_identity(
            trace.adapter_id, payload['adapter_config.json']))
        if publish:
            ok, _ = runner._materialize_remote_adapter(trace.adapter_id, runner.nvme_dir / trace.adapter_id)
            self.assertTrue(ok)
            if host:
                runner._stack.residency_manager._materialize_into_tier_dir(trace.adapter_id,
                    str(runner.nvme_dir / trace.adapter_id), StorageTier.HOST)
        return client

    def test_file_and_remote_sources_load_from_confirmed_reference_not_legacy_hint(self):
        for tier in ('remote', 'nvme', 'host'):
            with self.subTest(tier=tier):
                runner, slot, trace, plan, owner, rpc = self.build('remote')
                client = self.file_source(runner, trace, publish=tier != 'remote', host=tier == 'host')
                load = owner.demand_loader
                def guarded_load(**kwargs):
                    reference_owner = runner._stack.residency_manager.local_source_references
                    self.assertEqual(len(reference_owner.leases), 1)
                    self.assertFalse(runner._stack.residency_manager._delete_path(kwargs['lora_path']))
                    load(**kwargs)
                owner.demand_loader = guarded_load
                result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
                self.assertTrue(result.success, result.error)
                evidence = result.gpu_reference_evidence
                self.assertEqual(result.readiness_tier_before_dispatch, tier)
                self.assertEqual(evidence['service_intervals']['service_class']['tier'], tier)
                self.assertEqual(evidence['local_source_reference']['state'], 'released')
                self.assertTrue(evidence['local_source_reference']['content_verified'])
                preparation = evidence['preparation_interval']
                self.assertTrue(preparation['profile_eligible'])
                spans = evidence['service_intervals']
                self.assertAlmostEqual(preparation['d_ms'] + preparation['excluded_before_loading_ms'],
                    1000. * (spans['acquired_monotonic_s'] - spans['admitted_monotonic_s']))
                if tier == 'remote':
                    self.assertEqual(preparation['remote_transfer_id'], evidence['remote_preparation']['transfer_id'])
                    self.assertLessEqual(preparation['remote_published_monotonic_s'], preparation['native_started_monotonic_s'])
                self.assertEqual(client._opener.open.call_count, 1)
                runner._resolve_lora.assert_not_awaited()
                self.assertEqual(slot.active_requests, 0)

    def test_native_publication_after_remote_routing_reselects_before_admission(self):
        runner, slot, trace, plan, owner, rpc = self.build('remote')
        calls = 0
        async def becomes_native(*, operation, **kwargs):
            nonlocal calls
            if operation == 'source_snapshot':
                calls += 1
                if calls == 2:
                    ControllerNativeReferenceLifecycle.preload_native(owner)
            return await rpc(operation=operation, **kwargs)
        slot.engine.ieee_gpu_reference.side_effect = becomes_native
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.readiness_tier_before_dispatch, 'gpu')
        self.assertEqual(runner.router.selection_count, 2)
        self.assertEqual(len(result.gpu_reference_evidence['prior_routing_attempts']), 1)
        for key in slot.service_cost_model._profiles:
            if key.tier == 'remote':
                self.assertEqual(slot.service_cost_model.sample_counts(key), dict(d_ms=0, t_ms=0, o_ms=0))

    def test_published_http_delivery_keeps_measured_remote_path_and_subsequent_gpu_reuse(self):
        """Actual HTTP/file owner/router path; inference remains a CPU fixture."""
        from remote_artifact_node.server import ArtifactServer, ArtifactHandler, prepare_delivery_cache
        from faaslora.storage.http_artifact_store import HttpArtifactStoreClient
        from tests.test_http_artifact_store import content_manifest
        runner, slot, trace, plan, owner, _ = self.build('remote')
        root = runner.nvme_dir.parent
        origin, cache = root/'published-origin', root/'delivery-cache'
        payload = {'adapter_config.json': b'{"r":8}', 'weights': b'fixture-not-a-model'}
        for name, data in payload.items():
            target = origin/trace.adapter_id/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        index = root/'content.json'
        index.write_text(json.dumps(content_manifest(artifact_id=trace.adapter_id, files=payload)))
        publication = prepare_delivery_cache(origin, index, cache)
        records = []
        server = ArtifactServer(('127.0.0.1', 0), ArtifactHandler,
            root=origin, delivery_cache=cache, event_sink=records.append)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = HttpArtifactStoreClient(endpoint=f'http://127.0.0.1:{server.server_port}',
                required_delivery_mode='prepublished_gzip_v1')
            client.configure_content_manifest(json.loads(index.read_text()))
            runner._remote_artifact_client = client
            runner._ieee_artifact_identities[trace.adapter_id] = client.routing_identity(
                trace.adapter_id, payload['adapter_config.json'])
            self.bind_preparation_fixture(runner, slot)
            with asyncio.Runner() as event_loop, patch('remote_artifact_node.server._sha_file',
                       side_effect=AssertionError('request must not scan published/source objects')):
                cold = event_loop.run(runner._exec_request(trace, 4, 0., request_plan=plan))
                self.assertTrue(cold.success, cold.error)
                evidence = cold.gpu_reference_evidence
                transfer = evidence['remote_preparation']
                self.assertEqual(cold.readiness_tier_before_dispatch, 'remote')
                self.assertEqual(transfer['remote_delivery_mode'], 'prepublished_gzip_v1')
                self.assertFalse(transfer['remote_pack_performed'])
                self.assertTrue(transfer['published_archive_verified'])
                self.assertTrue(transfer['content_verified'])
                self.assertEqual(transfer['remote_archive_sha256'],
                                 publication['artifacts'][0]['archive_sha256'])
                self.assertEqual(transfer['transferred_bytes'],
                                 publication['artifacts'][0]['archive_bytes'])
                self.assertEqual(evidence['local_source_reference']['state'], 'released')
                preparation, service = evidence['preparation_interval'], evidence['service_intervals']
                self.assertTrue(preparation['cost_model_updated'])
                self.assertEqual(preparation['remote_transfer_id'], transfer['transfer_id'])
                self.assertLessEqual(transfer['published_monotonic_s'],
                                     preparation['native_started_monotonic_s'])
                self.assertAlmostEqual(preparation['d_ms']+preparation['excluded_before_loading_ms'],
                    1000*(service['acquired_monotonic_s']-service['admitted_monotonic_s']))
                # Only information already obtained by the first actual request.
                trace.request_id = 'req-published-gpu-hit'
                warm = event_loop.run(runner._exec_request(trace, 4, 0., request_plan=plan))
                self.assertTrue(warm.success, warm.error)
                self.assertEqual(warm.readiness_tier_before_dispatch, 'gpu')
                self.assertNotIn('remote_preparation', warm.gpu_reference_evidence)
                self.assertEqual(len(runner._remote_transfer_evidence), 1)
                runner._resolve_lora.assert_not_awaited()
                self.assertEqual(slot.active_requests, 0)
                self.assertEqual(owner.snapshot()['live_leases'], 0)
                self.assertFalse(runner._stack.residency_manager.local_source_references.materializations)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['transfer_id'], transfer['http_transfer_id'])
        self.assertEqual(records[0]['bytes_written'], transfer['transferred_bytes'])
        self.assertFalse(records[0]['pack_performed'])
        self.assertFalse(records[0]['temporary_created'])
        for tier in ('remote', 'gpu'):
            self.assertEqual(sum(slot.service_cost_model.sample_counts(key)['t_ms']
                for key in slot.service_cost_model._profiles if key.tier == tier), 1)

    def test_file_epoch_conflict_retries_selection_without_leaking_reference(self):
        runner, slot, trace, plan, owner, _ = self.build('remote')
        self.file_source(runner, trace, publish=True)
        files = runner._stack.residency_manager.local_source_references
        acquire = files.acquire_confirmed
        calls = 0
        def conflict_once(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                files.source_epoch += 1
            return acquire(**kwargs)
        with patch.object(files, 'acquire_confirmed', side_effect=conflict_once):
            result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertTrue(result.success, result.error)
        self.assertEqual(runner.router.selection_count, 2)
        self.assertIn('owner/epoch changed', result.gpu_reference_evidence['prior_routing_attempts'][0]['file_source_conflict'])
        self.assertFalse(files.leases)

    def test_wrong_event_final_timestamp_cannot_pass_completed_request(self):
        runner, slot, trace, plan, owner, _ = self.build('gpu')
        generate = slot.engine.generate_prepared.side_effect
        async def wrong(**kwargs):
            ttft, tpot, count, timing = await generate(**kwargs)
            timing['native_first_token_monotonic_s'] += .001
            return ttft, tpot, count, timing
        slot.engine.generate_prepared.side_effect = wrong
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertFalse(result.success)
        self.assertIn('native interval events', result.error)
        self.assertEqual(owner.snapshot()['live_leases'], 0)

    def test_cancellation_during_decode_retains_only_completed_intervals(self):
        from faaslora.clock import local_monotonic_clock_id
        from tests.test_ieee_tc_service_events import event
        runner, slot, trace, plan, owner, _ = self.build('host')
        entered = asyncio.Event()
        async def partial(**kwargs):
            reference = kwargs['gpu_reference']
            owner.begin_use(lease_id=reference['lease_id'], expected_owner_id=owner.owner_id,
                adapter_int_id=reference['adapter_int_id'], backend_request_id='partial',
                lora_name=trace.adapter_id, lora_path=reference['lora_path'])
            kwargs['native_event_observer'](event(adapter_id=trace.adapter_id,
                backend_request_id='partial', native_clock_id=local_monotonic_clock_id(),
                gpu_reference_owner_id=owner.owner_id, gpu_reference_lease_id=reference['lease_id'],
                gpu_reference_adapter_int_id=reference['adapter_int_id'],
                timestamp_monotonic_s=time.monotonic()))
            entered.set()
            await asyncio.Future()
        async def retire(*, gpu_reference, abort):
            self.assertTrue(abort)
            owner.end_use(lease_id=gpu_reference['lease_id'], expected_owner_id=owner.owner_id,
                backend_request_id='partial')
            return dict(gpu_reference_owner_id=owner.owner_id,
                gpu_reference_lease_id=gpu_reference['lease_id'], native_retirement={'retired': True})
        slot.engine.generate_prepared.side_effect = partial
        slot.engine.ieee_retire_generation = AsyncMock(side_effect=retire)
        async def run():
            task = asyncio.create_task(runner._exec_request(trace, 4, 0., request_plan=plan))
            await asyncio.wait_for(entered.wait(), .5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(run())
        observed = [slot.service_cost_model.sample_counts(key) for key in slot.service_cost_model._profiles
                    if slot.service_cost_model.sample_counts(key)['t_ms']]
        self.assertEqual(observed, [dict(d_ms=1, t_ms=1, o_ms=0)])
        self.assertEqual(owner.snapshot()['live_leases'], 0)
        self.assertEqual(slot.active_requests, 0)


class LocalSourceOwnership(unittest.TestCase):
    def setUp(self):
        from faaslora.memory.residency_manager import ResidencyManager
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.host, self.nvme = self.root / 'host', self.root / 'nvme'
        self.host.mkdir()
        self.nvme.mkdir()
        self.source = self.nvme / 'a'
        self.source.mkdir()
        (self.source / 'weights').write_bytes(b'tiny-test-fixture')
        self.manager = ResidencyManager({'memory': {
            'host': {'cache_dir': str(self.host)}, 'nvme': {'cache_dir': str(self.nvme)}}}, Mock(), Mock())

    def acquire(self, lease_id='r', path=None):
        return self.manager.acquire_local_source(path=str(path or self.source), adapter_id='a', lease_id=lease_id)

    def release(self, receipt):
        self.manager.release_local_source(lease_id=receipt['lease_id'], expected_owner_id=receipt['owner_id'])

    def test_remote_space_is_held_before_body_and_includes_previous_copy(self):
        owner = self.manager.local_source_references
        files = {'adapter_model.safetensors': (17, '0' * 64)}
        before = owner.inventory()['tiers']['nvme']['allocated_file_bytes']
        with owner.materializing(self.source) as transfer:
            with owner.transfer_workspace(transfer) as staging:
                receipt = owner.prepare_transfer(transfer, staging, 123, files,
                                                 limit_bytes=before + 8192)
                self.assertEqual(receipt['reserved_file_bytes'], 8192)
                self.assertEqual(receipt['used_file_bytes_before'], before)
                self.assertEqual((staging.parent / 'artifact.tar.gz').stat().st_size, 123)
                self.assertEqual((staging / 'adapter_model.safetensors').stat().st_size, 17)
                view = owner.inventory()
                self.assertEqual(view['transfer_held_file_bytes'], 8192)
                self.assertEqual(view['tiers']['nvme']['allocated_file_bytes'], before + 8192)
                self.assertEqual((self.source / 'weights').read_bytes(), b'tiny-test-fixture')
        self.assertEqual(owner.inventory()['allocated_file_bytes'], before)

    def test_concurrent_transfers_cannot_spend_the_same_remaining_space(self):
        owner = self.manager.local_source_references
        files = {'weights': (17, '0' * 64)}
        before = owner.inventory()['tiers']['nvme']['allocated_file_bytes']
        with owner.materializing(self.source) as first, owner.materializing(self.nvme / 'b') as second:
            with owner.transfer_workspace(first) as one, owner.transfer_workspace(second) as two:
                owner.prepare_transfer(first, one, 123, files, limit_bytes=before + 8192)
                with self.assertRaisesRegex(RuntimeError, 'capacity conflict'):
                    owner.prepare_transfer(second, two, 123, files, limit_bytes=before + 8192)
                self.assertFalse((two.parent / 'artifact.tar.gz').exists())

    def test_preallocation_failure_is_not_replaced_with_sparse_truncate(self):
        owner = self.manager.local_source_references
        with owner.materializing(self.source) as transfer:
            with owner.transfer_workspace(transfer) as staging:
                with patch('os.posix_fallocate', side_effect=OSError('unsupported filesystem')):
                    with self.assertRaisesRegex(OSError, 'unsupported filesystem'):
                        owner.prepare_transfer(transfer, staging, 123, {'weights': (17, '0' * 64)},
                                               limit_bytes=32768)
        self.assertEqual(sorted(p.name for p in self.nvme.iterdir()), ['a'])

    def test_preallocated_writer_changes_content_not_allocation_identity(self):
        owner = self.manager.local_source_references
        with owner.materializing(self.source) as transfer:
            with owner.transfer_workspace(transfer) as staging:
                owner.prepare_transfer(transfer, staging, 123, {'weights': (17, '0' * 64)},
                                       limit_bytes=32768)
                with (staging / 'weights').open('r+b') as writer:
                    writer.write(b'changed')
                owner.inventory()  # Managed content timestamps may change.
                with (staging / 'weights').open('ab') as writer:
                    writer.write(b'illegal growth')
                with self.assertRaisesRegex(RuntimeError, 'reserved file changed'):
                    owner.inventory()

    def test_two_writer_threads_reserve_one_available_transfer(self):
        owner = self.manager.local_source_references
        before = owner.inventory()['tiers']['nvme']['allocated_file_bytes']
        ready, allocated = threading.Barrier(2), threading.Barrier(2)
        outcomes, errors = [], []
        def transfer(name):
            try:
                with owner.materializing(self.nvme / name) as token:
                    with owner.transfer_workspace(token) as staging:
                        ready.wait(2)
                        try:
                            owner.prepare_transfer(token, staging, 123, {'weights': (17, '0' * 64)},
                                                   limit_bytes=before + 8192)
                            outcomes.append('reserved')
                        except RuntimeError as error:
                            if 'capacity conflict' not in str(error):
                                raise
                            outcomes.append('conflict')
                        finally:
                            allocated.wait(2)  # Neither releases before both decisions.
            except BaseException as error:
                errors.append(error)
        workers = [threading.Thread(target=transfer, args=(name,)) for name in ('b', 'c')]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(3)
        self.assertFalse(any(worker.is_alive() for worker in workers))
        self.assertFalse(errors, errors)
        self.assertCountEqual(outcomes, ['reserved', 'conflict'])
        self.assertEqual(owner.inventory()['allocated_file_bytes'], before)

    def test_reserved_allocation_failure_records_exact_observation_without_relaxing_guard(self):
        from faaslora.memory import residency_manager as module
        owner = self.manager.local_source_references
        original_inventory = module._local_file_inventory
        with owner.materializing(self.source) as transfer:
            with owner.transfer_workspace(transfer) as staging:
                owner.prepare_copy(transfer, staging, {'weights': (17, '0' * 64)}, limit_bytes=32768)
                key, expected = next(iter(owner._prepared_transfers[transfer]['files'].items()))
                def observed_change(*args, **kwargs):
                    view = original_inventory(*args, **kwargs)
                    for item in view['allocations']:
                        if (item['device'], item['inode']) == key:
                            item['allocated_bytes'] += 4096
                    return view
                with patch.object(module, '_local_file_inventory', side_effect=observed_change):
                    with self.assertRaisesRegex(RuntimeError, 'reserved file changed') as caught:
                        owner.inventory()
                detail = json.loads(str(caught.exception).split(': ', 1)[1])
                self.assertEqual(detail['transfer_id'], transfer)
                self.assertEqual(detail['tier'], 'nvme')
                self.assertEqual(detail['expected']['allocated_bytes'], expected[1])
                self.assertEqual(detail['observed']['allocated_bytes'], expected[1] + 4096)
                self.assertEqual(detail['changed_fields'], ['allocated_bytes'])
                self.assertEqual(owner._prepared_transfers[transfer]['files'][key], expected)
                owner.inventory()  # Failure diagnostics did not mutate ownership.

    def test_reserved_missing_file_failure_distinguishes_identity_from_size(self):
        owner = self.manager.local_source_references
        with owner.materializing(self.source) as transfer:
            with owner.transfer_workspace(transfer) as staging:
                owner.prepare_copy(transfer, staging, {'weights': (17, '0' * 64)}, limit_bytes=32768)
                (staging / 'weights').unlink()
                with self.assertRaisesRegex(RuntimeError, 'reserved file changed') as caught:
                    owner.inventory()
                detail = json.loads(str(caught.exception).split(': ', 1)[1])
                self.assertIsNone(detail['observed'])
                self.assertEqual(detail['changed_fields'], ['missing_reserved_inode'])

    def test_transfer_budget_is_frozen_and_private_writer_cannot_be_reclaimed(self):
        owner = self.manager.local_source_references
        with owner.materializing(self.source) as transfer:
            with owner.transfer_workspace(transfer) as staging:
                owner.prepare_transfer(transfer, staging, 123, {'weights': (17, '0' * 64)},
                                       limit_bytes=32768)
                self.assertFalse(self.manager._delete_path(staging.parent))
                self.assertFalse(self.manager._delete_path(self.source))
        with owner.materializing(self.source) as transfer:
            with owner.transfer_workspace(transfer) as staging:
                with self.assertRaisesRegex(ValueError, 'budget cannot change'):
                    owner.prepare_transfer(transfer, staging, 123, {'weights': (17, '0' * 64)},
                                           limit_bytes=65536)

    def test_failed_workspace_cleanup_leaves_real_bytes_in_next_budget(self):
        owner = self.manager.local_source_references
        before = owner.inventory()['allocated_file_bytes']
        with patch('shutil.rmtree', side_effect=OSError('cleanup failure')):
            with self.assertRaisesRegex(OSError, 'cleanup failure'):
                with owner.materializing(self.source) as transfer:
                    with owner.transfer_workspace(transfer) as staging:
                        owner.prepare_transfer(transfer, staging, 123, {'weights': (17, '0' * 64)},
                                               limit_bytes=before + 8192)
        self.assertFalse(owner.materializations)
        self.assertEqual(owner.inventory()['allocated_file_bytes'], before + 8192)
        with owner.materializing(self.nvme / 'b') as transfer:
            with owner.transfer_workspace(transfer) as staging:
                with self.assertRaisesRegex(RuntimeError, 'capacity conflict'):
                    owner.prepare_transfer(transfer, staging, 123, {'weights': (17, '0' * 64)},
                                           limit_bytes=before + 8192)

    def test_duplicate_target_does_not_allocate_another_workspace(self):
        owner = self.manager.local_source_references
        with owner.materializing(self.source):
            with self.assertRaisesRegex(RuntimeError, 'already has an active transfer'):
                with owner.materializing(self.source):
                    self.fail('duplicate destination was admitted')
            self.assertEqual(len(owner.materializations), 1)

    def test_inventory_counts_retained_copies_and_private_workspace(self):
        import shutil
        shutil.copytree(self.source, self.host / 'a')
        workspace = self.nvme / '.a.staging-test'
        workspace.mkdir()
        (workspace / 'archive').write_bytes(b'partial-archive')
        view = self.manager.local_file_inventory()
        size = (self.source / 'weights').stat().st_size
        self.assertEqual(view['logical_file_bytes'], 2 * size + 15)
        self.assertEqual(view['tiers']['host']['logical_file_bytes'], size)
        self.assertEqual(view['tiers']['nvme']['logical_file_bytes'], size + 15)
        self.assertFalse(view['capacity_reserved'])
        self.assertFalse(view['physical_release_proven'])

    def test_inventory_deduplicates_shared_inodes_but_not_equal_content(self):
        import os
        import shutil
        second = self.nvme / 'b'
        second.mkdir()
        os.link(self.source / 'weights', second / 'weights')
        third = self.host / 'c'
        shutil.copytree(self.source, third)
        view = self.manager.local_file_inventory()
        size = (self.source / 'weights').stat().st_size
        self.assertEqual(view['logical_file_bytes'], 2 * size)
        self.assertEqual(view['file_path_bytes'], 3 * size)
        self.assertEqual(view['unique_file_count'], 2)
        shared = [item for item in view['allocations'] if len(item['paths']) == 2]
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0]['external_link_count'], 0)

    def test_cross_tier_hardlinks_are_one_owner_allocation_not_additive_tiers(self):
        import os
        target = self.host / 'a'
        target.mkdir()
        os.link(self.source / 'weights', target / 'weights')
        view = self.manager.local_file_inventory()
        size = (self.source / 'weights').stat().st_size
        self.assertEqual(view['logical_file_bytes'], size)
        self.assertFalse(view['tier_totals_additive'])
        self.assertEqual(view['tiers']['host']['file_path_bytes'], size)
        self.assertEqual(view['tiers']['nvme']['file_path_bytes'], size)

    def test_sparse_file_retains_logical_and_allocated_sizes_separately(self):
        sparse = self.source / 'sparse'
        with sparse.open('wb') as handle:
            handle.truncate(1024 * 1024)
        view = self.manager.local_file_inventory()
        item = next(item for item in view['allocations'] if str(sparse) in item['paths'])
        self.assertEqual(item['logical_bytes'], 1024 * 1024)
        self.assertEqual(item['allocated_bytes'], sparse.stat().st_blocks * 512)
        self.assertLess(item['allocated_bytes'], item['logical_bytes'])

    def test_external_hardlink_is_not_reported_as_reclaimable_capacity(self):
        import os
        os.link(self.source / 'weights', self.root / 'external-weights')
        view = self.manager.local_file_inventory()
        item = next(item for item in view['allocations'] if item['kind'] == 'file')
        self.assertEqual(item['external_link_count'], 1)
        self.assertFalse(view['physical_release_proven'])
        self.assertTrue(self.manager._delete_path(str(self.source)))
        self.assertEqual((self.root / 'external-weights').read_bytes(), b'tiny-test-fixture')

    def test_live_transfer_cannot_be_labelled_a_complete_capacity_snapshot(self):
        with self.manager.local_source_references.materializing(self.source):
            with self.assertRaisesRegex(RuntimeError, 'quiescent'):
                self.manager.local_file_inventory()
            self.release(self.acquire())  # Old completed source still readable.
        self.assertGreater(self.manager.local_file_inventory()['allocated_bytes'], 0)

    def test_acquired_source_carries_measured_file_representation_not_model_size(self):
        receipt = self.acquire()
        footprint = receipt['file_footprint']
        self.assertEqual(footprint['logical_file_bytes'], len(b'tiny-test-fixture'))
        self.assertEqual(footprint['unique_file_count'], 1)
        self.assertEqual(footprint['scope'], 'linked_inode_storage_v1')
        self.assertFalse(footprint['content_verified'])
        self.assertNotIn('allocations', footprint)  # Do not repeat full trees per request.

    def test_file_links_missing_roots_and_nested_roots_are_not_hidden(self):
        from faaslora.memory.residency_manager import LocalSourceReferences
        with self.assertRaisesRegex(ValueError, 'nonoverlapping'):
            LocalSourceReferences({'nvme': self.nvme, 'host': self.source})
        (self.source / 'linked').symlink_to(self.root / 'missing')
        with self.assertRaisesRegex(ValueError, 'links/special'):
            self.acquire()
        self.assertFalse(self.manager.local_source_references.leases)
        (self.source / 'linked').unlink()
        self.host.rmdir()
        with self.assertRaises(FileNotFoundError):
            self.manager.local_file_inventory()

    def test_noncooperative_file_change_invalidates_footprint_scan(self):
        from unittest.mock import patch
        original = Path.lstat
        weights = self.source / 'weights'
        observed = 0
        def changing(path, *args, **kwargs):
            nonlocal observed
            if path == weights:
                observed += 1
                if observed == 2:
                    weights.write_bytes(b'changed-during-scan')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'lstat', changing):
            with self.assertRaisesRegex(RuntimeError, 'changed while collecting'):
                self.manager.local_file_inventory()

    def test_two_readers_release_only_their_own_share(self):
        first, second = self.acquire('first'), self.acquire('second')
        self.release(first)
        self.release(first)
        self.assertFalse(self.manager._delete_path(str(self.source)))
        self.assertTrue((self.source / 'weights').exists())
        self.release(second)
        self.assertTrue(self.manager._delete_path(str(self.source)))

    def test_parent_and_child_mutations_cannot_bypass_live_reference(self):
        self.acquire()
        self.assertFalse(self.manager._delete_path(str(self.nvme)))
        self.assertFalse(self.manager._delete_path(str(self.source / 'weights')))
        self.assertTrue((self.source / 'weights').exists())

    def test_replacement_is_deferred_but_same_copy_and_other_tier_remain_usable(self):
        from faaslora.registry.schema import StorageTier
        receipt = self.acquire()
        other = self.host / 'a'
        other.mkdir()
        (other / 'weights').write_bytes(b'replacement-fixture')
        self.assertIsNone(self.manager._materialize_into_tier_dir('a', str(other), StorageTier.NVME))
        self.assertEqual((self.source / 'weights').read_bytes(), b'tiny-test-fixture')
        self.assertEqual(self.manager._materialize_into_tier_dir('a', str(self.source), StorageTier.NVME), str(self.source))
        self.assertEqual(self.manager._materialize_into_tier_dir('a', str(self.source), StorageTier.HOST), str(other))
        self.release(receipt)
        self.assertEqual(self.manager._materialize_into_tier_dir('a', str(other), StorageTier.NVME), str(self.source))

    def test_lease_identity_cannot_be_rebound_or_revived(self):
        receipt = self.acquire()
        self.assertEqual(receipt, self.acquire())
        with self.assertRaisesRegex(ValueError, 'owner changed'):
            self.manager.release_local_source(lease_id='r', expected_owner_id='wrong')
        (self.nvme / 'b').mkdir()
        with self.assertRaisesRegex(ValueError, 'rebound'):
            self.acquire(path=self.nvme / 'b')
        self.release(receipt)
        with self.assertRaisesRegex(ValueError, 'unused lease'):
            self.acquire()

    def test_unmanaged_or_missing_source_is_not_silently_registered(self):
        with self.assertRaisesRegex(ValueError, 'managed tier'):
            self.acquire(path=self.root)
        with self.assertRaises(FileNotFoundError):
            self.acquire(path=self.nvme / 'missing')
        self.manager.set_storage_manager(Mock())
        with self.assertRaisesRegex(RuntimeError, 'does not share'):
            self.acquire()

    def test_failed_physical_delete_does_not_publish_success(self):
        from unittest.mock import patch
        with patch('faaslora.memory.residency_manager.shutil.rmtree', side_effect=OSError('busy')):
            self.assertFalse(self.manager._delete_path(str(self.source)))
        self.assertTrue(self.source.exists())

    def test_rejected_eviction_leaves_tier_accounting_unchanged(self):
        from faaslora.registry.schema import StorageTier
        receipt = self.acquire()
        metadata = SimpleNamespace(storage_tier=StorageTier.NVME, size_bytes=17,
                                   storage_path=str(self.source))
        self.manager.registry.get_artifact.return_value = metadata
        self.manager.tier_artifacts[StorageTier.NVME].add('a')
        self.manager.tier_capacities[StorageTier.NVME].used_bytes = 17
        self.assertFalse(asyncio.run(self.manager.evict_artifact('a', StorageTier.REMOTE)))
        self.assertIn('a', self.manager.tier_artifacts[StorageTier.NVME])
        self.assertEqual(self.manager.tier_capacities[StorageTier.NVME].used_bytes, 17)
        self.release(receipt)

    def test_private_transfer_retains_tier_but_leaves_old_copy_readable(self):
        owner = self.manager.local_source_references
        with owner.materializing(self.source):
            self.assertEqual(len(owner.materializations), 1)
            self.assertFalse(self.manager._delete_path(str(self.nvme)))
            self.release(self.acquire())
        self.assertFalse(owner.materializations)
        self.assertTrue(self.manager._delete_path(str(self.nvme)))

    def test_failed_tier_copy_preserves_destination_and_removes_private_stage(self):
        from unittest.mock import patch
        from faaslora.registry.schema import StorageTier
        target = self.host / 'a'
        target.mkdir()
        (target / 'old').write_bytes(b'valid-before-copy')
        def partial(src, dst):
            dst.mkdir()
            (dst / 'partial').write_bytes(b'incomplete')
            raise OSError('copy failed')
        with patch('faaslora.memory.residency_manager.shutil.copytree', side_effect=partial):
            self.assertIsNone(self.manager._materialize_into_tier_dir('a', str(self.source), StorageTier.HOST))
        self.assertEqual((target / 'old').read_bytes(), b'valid-before-copy')
        self.assertEqual(sorted(p.name for p in self.host.iterdir()), ['a'])

    def test_copy_excludes_concurrent_source_reclamation_until_completion(self):
        import shutil
        from unittest.mock import patch
        from faaslora.registry.schema import StorageTier
        original = shutil.copytree
        entered, proceed, attempting, finished = (threading.Event() for _ in range(4))
        results = {}
        def copying(src, dst):
            entered.set()
            if not proceed.wait(2):
                raise RuntimeError('test copy barrier timeout')
            return original(src, dst)
        def transfer():
            results['copy'] = self.manager._materialize_into_tier_dir('a', str(self.source), StorageTier.HOST)
        def reclaim():
            attempting.set()
            results['delete'] = self.manager._delete_path(str(self.source))
            finished.set()
        with patch('faaslora.memory.residency_manager.shutil.copytree', side_effect=copying):
            copy_thread = threading.Thread(target=transfer)
            delete_thread = threading.Thread(target=reclaim)
            copy_thread.start()
            try:
                self.assertTrue(entered.wait(1))
                delete_thread.start()
                self.assertTrue(attempting.wait(1))
                self.assertFalse(finished.wait(.02))
            finally:
                proceed.set()
                copy_thread.join(2)
                if delete_thread.ident is not None:
                    delete_thread.join(2)
        self.assertEqual(results['copy'], str(self.host / 'a'))
        self.assertTrue(results['delete'])
        self.assertEqual((self.host / 'a' / 'weights').read_bytes(), b'tiny-test-fixture')


class ConfirmedFilePublication(unittest.TestCase):
    """Actual download/owner publication with tiny files, not inference evidence."""
    def setUp(self):
        from faaslora.memory.residency_manager import ResidencyManager
        from faaslora.storage.http_artifact_store import HttpArtifactStoreClient
        from tests.test_http_artifact_store import content_manifest
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.nvme, self.host = self.root / 'nvme', self.root / 'host'
        self.nvme.mkdir()
        self.host.mkdir()
        self.manager = ResidencyManager({'memory': {'nvme': {'cache_dir': str(self.nvme)},
            'host': {'cache_dir': str(self.host)}}}, Mock(), Mock())
        self.owner = self.manager.local_source_references
        self.client = HttpArtifactStoreClient(endpoint='http://127.0.0.1:1')
        self.payload = {'adapter_config.json': b'{"r":8}', 'nested/weights': b'tiny-test-fixture'}
        self.client.configure_content_manifest(content_manifest(files=self.payload))
        self.client._opener = Mock()
        self.runner = ScenarioRunner.__new__(ScenarioRunner)
        self.runner.model_cfg = {'ieee_gpu_references': True}
        self.runner._stack = SimpleNamespace(residency_manager=self.manager)
        self.runner._remote_artifact_client = self.client
        self.runner._remote_transfer_evidence = []

    def fetch(self, payload=None):
        from tests.test_http_artifact_store import archive_bytes, SizedResponse
        self.client._opener.open.return_value = SizedResponse(archive_bytes(list(
            (self.payload if payload is None else payload).items())))
        return self.runner._materialize_remote_adapter('a', self.nvme / 'a')

    def test_actual_runner_publishes_verified_snapshot_and_protects_exact_copy(self):
        from faaslora.clock import local_monotonic_clock_id
        self.assertEqual(self.owner.source_snapshot('a')['sources'], [])
        self.assertTrue(self.fetch()[0])
        state = self.owner.source_snapshot('a')
        self.assertEqual(len(state['sources']), 1)
        source = state['sources'][0]
        self.assertEqual((source['tier'], source['adapter_id']), ('nvme', 'a'))
        self.assertTrue(source['content_verified'])
        self.assertEqual(source['file_path_bytes'], sum(map(len, self.payload.values())))
        self.assertEqual(source['allocated_file_bytes'], 8192)
        transfer = self.runner._remote_transfer_evidence[-1]
        self.assertEqual(transfer['loading_clock_id'], local_monotonic_clock_id())
        self.assertLessEqual(transfer['loading_started_monotonic_s'], transfer['published_monotonic_s'])
        evidence = self.runner._remote_transfer_evidence[-1]['confirmed_file_publication']
        self.assertEqual(evidence['epoch'], state['epoch'])
        self.assertEqual(evidence['content_sha256'], source['content_sha256'])
        lease = self.owner.acquire_confirmed(path=source['path'], adapter_id='a', lease_id='r',
            expected_owner_id=state['owner_id'], expected_epoch=state['epoch'],
            expected_content_sha256=source['content_sha256'])
        self.assertTrue(lease['content_verified'])
        with self.owner.mutation(self.nvme / 'a') as allowed:
            self.assertFalse(allowed)
        self.owner.release(lease_id='r', expected_owner_id=state['owner_id'])
        source['tier'] = 'gpu'
        self.assertEqual(self.owner.source_snapshot('a')['sources'][0]['tier'], 'nvme')

    def test_directory_existence_is_not_confirmed_local_or_remote(self):
        (self.nvme / 'a').mkdir()
        (self.nvme / 'a' / 'unknown').write_bytes(b'x')
        with self.assertRaisesRegex(RuntimeError, 'without verified source publication'):
            self.owner.source_snapshot('a')
        self.assertTrue(self.fetch()[0])
        self.assertEqual(len(self.owner.source_snapshot('a')['sources']), 1)

    def test_bad_transfer_preserves_old_confirmed_copy_and_epoch(self):
        from faaslora.storage.http_artifact_store import RemoteArtifactError
        self.fetch()
        before = self.owner.source_snapshot('a')
        with self.assertRaises(RemoteArtifactError):
            self.fetch(self.payload | {'nested/weights': b'wrong-data'})
        after = self.owner.source_snapshot('a')
        self.assertEqual(after['sources'], before['sources'])
        self.assertEqual(after['epoch'], before['epoch'])
        self.assertFalse(self.owner.materializations)

    def test_delete_withdraws_before_reuse_and_failed_rename_restores_identity(self):
        import shutil
        self.fetch()
        before = self.owner.source_snapshot('a')
        original = Path.rename
        def fail_new(path, destination):
            if path.name == 'payload':
                raise OSError('publication-test-failure')
            return original(path, destination)
        with patch.object(Path, 'rename', fail_new), self.assertRaises(OSError):
            self.fetch()
        restored = self.owner.source_snapshot('a')
        self.assertEqual(restored['sources'], before['sources'])
        self.assertGreater(restored['epoch'], before['epoch'])
        with self.owner.mutation(self.nvme / 'a') as allowed:
            self.assertTrue(allowed)
            with self.assertRaisesRegex(RuntimeError, 'without verified source publication'):
                self.owner.source_snapshot('a')
            shutil.rmtree(self.nvme / 'a')
        self.assertEqual(self.owner.source_snapshot('a')['sources'], [])

    def test_observed_file_metadata_change_withdraws_confirmation(self):
        import os
        self.fetch()
        epoch = self.owner.source_snapshot('a')['epoch']
        path = self.nvme / 'a' / 'nested' / 'weights'
        before = path.stat()
        path.write_bytes(b'X'*len(self.payload['nested/weights']))
        # Post-publication external writers are outside the cooperative owner.
        # Check observable invalidation deterministically, not timestamp precision.
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        with self.assertRaisesRegex(RuntimeError, 'changed outside'):
            self.owner.source_snapshot('a')
        self.assertGreater(self.owner.source_epoch, epoch)
        with self.assertRaisesRegex(RuntimeError, 'without verified source publication'):
            self.owner.source_snapshot('a')

    def test_verified_host_copy_preserves_content_and_lower_tier_after_eviction(self):
        from faaslora.registry.schema import StorageTier
        self.fetch()
        destination = self.manager._materialize_into_tier_dir('a', str(self.nvme / 'a'), StorageTier.HOST)
        self.assertEqual(destination, str(self.host / 'a'))
        sources = self.owner.source_snapshot('a')['sources']
        self.assertEqual({row['tier'] for row in sources}, {'host', 'nvme'})
        self.assertEqual(len({row['content_sha256'] for row in sources}), 1)
        self.assertTrue(self.manager._delete_path(destination))
        self.assertEqual([row['tier'] for row in self.owner.source_snapshot('a')['sources']], ['nvme'])

    def test_corrupt_tier_copy_does_not_publish_fast_tier_or_invalidate_source(self):
        from faaslora.registry.schema import StorageTier
        self.fetch()
        original = self.owner.publish_transfer
        changed = []
        def corrupt(transfer, staging, target, publish, **kwargs):
            payload = Path(staging) / 'nested' / 'weights'
            if payload.is_file():
                payload.write_bytes(b'X'*len(self.payload['nested/weights']))
                changed.append(str(payload))
            return original(transfer, staging, target, publish, **kwargs)
        with patch.object(self.owner, 'publish_transfer', corrupt):
            destination = self.manager._materialize_into_tier_dir('a', str(self.nvme / 'a'), StorageTier.HOST)
        self.assertIsNone(destination)
        self.assertEqual(len(changed), 1)
        self.assertFalse((self.host / 'a').exists())
        self.assertEqual([row['tier'] for row in self.owner.source_snapshot('a')['sources']], ['nvme'])

    def test_actual_async_copy_preallocates_payload_only_and_publishes_same_content(self):
        from faaslora.registry.schema import StorageTier
        self.fetch()
        self.manager.tier_capacities[StorageTier.HOST].total_bytes = 8192
        before = self.manager.local_file_budgets()
        self.assertFalse(before['snapshot_reserves_capacity'])
        self.assertFalse(before['total_host_memory_covered'])
        original = self.owner.prepare_copy
        def reserve(*args, **kwargs):
            receipt = original(*args, **kwargs)
            self.assertFalse((args[1].parent / 'artifact.tar.gz').exists())
            view = self.manager.local_file_budgets()['tiers']['host']
            self.assertEqual((view['used_bytes'], view['remaining_bytes'], view['pending_increment_bytes']),
                             (8192, 0, 0))
            self.assertEqual(view['active_transfers'], 1)
            self.assertFalse(self.manager._delete_path(str(self.nvme / 'a')))
            return receipt
        with patch.object(self.owner, 'prepare_copy', reserve), patch(
                'shutil.copytree', side_effect=AssertionError('unbudgeted fallback')):
            receipt = asyncio.run(self.runner._materialize_confirmed_source_async(
                'a', self.nvme / 'a', StorageTier.HOST))
        self.assertEqual(receipt['state'], 'published')
        self.assertEqual(receipt['file_reservation']['transfer_kind'], 'local_verified_copy')
        self.assertEqual(receipt['file_reservation']['reserved_file_bytes'], 8192)
        self.assertEqual(receipt['copied_bytes'], sum(map(len, self.payload.values())))
        self.assertTrue(receipt['source_reference_released'])
        self.assertLessEqual(receipt['started_at'], receipt['ready_at'])
        self.assertLessEqual(receipt['ready_at'], receipt['finished_at'])
        self.assertEqual(self.manager.local_transfer_evidence, [receipt])
        self.assertEqual(len({r['content_sha256'] for r in self.owner.source_snapshot('a')['sources']}), 1)
        self.assertFalse(self.owner.leases or self.owner.materializations)
        view = self.manager.local_file_budgets()['tiers']['host']
        self.assertEqual((view['used_bytes'], view['active_transfers']), (8192, 0))
        # Reverse movement replaces an old NVMe copy, not an assumed one-way cache.
        reverse = asyncio.run(self.runner._materialize_confirmed_source_async(
            'a', self.host / 'a', StorageTier.NVME))
        self.assertEqual(reverse['state'], 'published')
        self.assertEqual(reverse['file_reservation']['used_file_bytes_before'], 8192)
        self.assertEqual(reverse['file_reservation']['allocated_file_bytes_after'], 16384)
        self.assertEqual(self.manager.local_file_budgets()['tiers']['nvme']['used_bytes'], 8192)
        self.assertEqual(len({r['content_sha256'] for r in self.owner.source_snapshot('a')['sources']}), 1)

    def test_replacement_requires_both_old_and_new_payload_space(self):
        from faaslora.registry.schema import StorageTier
        self.fetch()
        self.manager.tier_capacities[StorageTier.HOST].total_bytes = 16383
        self.manager.materialize_confirmed_source('a', self.nvme / 'a', StorageTier.HOST)
        before = self.owner.source_snapshot('a')
        with self.assertRaisesRegex(RuntimeError, 'capacity conflict'):
            self.manager.materialize_confirmed_source('a', self.nvme / 'a', StorageTier.HOST)
        self.assertEqual(self.owner.source_snapshot('a')['sources'], before['sources'])
        self.assertEqual(self.owner.source_epoch, before['epoch'])
        self.assertEqual(self.manager.local_transfer_evidence[-1]['state'], 'not_published')
        self.assertEqual(self.manager.local_transfer_evidence[-1]['copied_bytes'], 0)
        self.assertEqual(sorted(p.name for p in self.host.iterdir()), ['a'])
        self.assertFalse(self.owner.leases or self.owner.materializations)

    def test_local_copy_shares_capacity_with_another_preallocated_transfer(self):
        from faaslora.registry.schema import StorageTier
        self.fetch()
        self.manager.tier_capacities[StorageTier.HOST].total_bytes = 8192
        with self.owner.materializing(self.host / 'b') as token:
            with self.owner.transfer_workspace(token) as staging:
                self.owner.prepare_copy(token, staging, {'weights': (1, '0'*64)}, limit_bytes=8192)
                before = self.manager.local_file_budgets()['tiers']['host']
                self.assertEqual(before['used_bytes'], 4096)
                with self.assertRaisesRegex(RuntimeError, 'capacity conflict'):
                    self.manager.materialize_confirmed_source('a', self.nvme / 'a', StorageTier.HOST)
                self.assertEqual(self.manager.local_file_budgets()['tiers']['host'], before)
        self.assertEqual(self.manager.local_file_budgets()['tiers']['host']['used_bytes'], 0)
        self.assertFalse(self.owner.leases)

    def test_local_copy_cancellation_joins_writer_before_releasing_source(self):
        from faaslora.registry.schema import StorageTier
        self.fetch()
        entered, leave = threading.Event(), threading.Event()
        original = self.owner.prepare_copy
        def reserve(*args, **kwargs):
            result = original(*args, **kwargs)
            entered.set()
            if not leave.wait(5):
                raise RuntimeError('test copy gate timed out')
            return result
        async def run():
            task = asyncio.create_task(self.runner._materialize_confirmed_source_async(
                'a', self.nvme / 'a', StorageTier.HOST))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                self.assertFalse(self.manager._delete_path(str(self.nvme / 'a')))
                self.assertTrue(self.owner.leases and self.owner.materializations)
            finally:
                leave.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        with patch.object(self.owner, 'prepare_copy', reserve):
            asyncio.run(run())
        evidence = self.manager.local_transfer_evidence[-1]
        self.assertEqual((evidence['state'], evidence['copied_bytes']), ('not_published', 0))
        self.assertTrue(evidence['source_reference_released'])
        self.assertFalse(self.owner.leases or self.owner.materializations)
        self.assertFalse(list(self.host.iterdir()))
        self.assertEqual([s['tier'] for s in self.owner.source_snapshot('a')['sources']], ['nvme'])

    def test_reader_on_old_destination_prevents_replacement_not_source_release(self):
        from faaslora.registry.schema import StorageTier
        self.fetch()
        self.manager.materialize_confirmed_source('a', self.nvme / 'a', StorageTier.HOST)
        before = self.owner.source_snapshot('a')
        held = self.owner.acquire(path=str(self.host / 'a'), adapter_id='a', lease_id='old-reader')
        try:
            with self.assertRaisesRegex(RuntimeError, 'live source reference'):
                self.manager.materialize_confirmed_source('a', self.nvme / 'a', StorageTier.HOST)
            self.assertEqual(list(self.owner.leases), ['old-reader'])
            self.assertEqual(self.owner.source_snapshot('a')['sources'], before['sources'])
        finally:
            self.manager.release_local_source(lease_id=held['lease_id'], expected_owner_id=held['owner_id'])
        self.assertEqual(sorted(p.name for p in self.host.iterdir()), ['a'])

    def test_failed_copy_cleanup_retains_capacity_and_failure_evidence(self):
        from faaslora.registry.schema import StorageTier
        self.fetch()
        self.manager.tier_capacities[StorageTier.HOST].total_bytes = 8192
        with patch.object(self.owner, 'publish_transfer', side_effect=RuntimeError('injected publication error')), \
                patch('shutil.rmtree', side_effect=OSError('injected cleanup error')):
            with self.assertRaisesRegex(OSError, 'injected cleanup error'):
                self.manager.materialize_confirmed_source('a', self.nvme / 'a', StorageTier.HOST)
        view = self.manager.local_file_budgets()['tiers']['host']
        self.assertEqual((view['used_bytes'], view['remaining_bytes'], view['active_transfers']), (8192, 0, 0))
        self.assertEqual(self.manager.local_transfer_evidence[-1]['state'], 'not_published')
        self.assertEqual(self.manager.local_transfer_evidence[-1]['error_type'], 'OSError')
        self.assertEqual(self.manager.local_transfer_evidence[-1]['error_chain'], ['OSError', 'RuntimeError'])
        self.assertFalse(self.owner.leases or self.owner.materializations)
        with self.assertRaisesRegex(RuntimeError, 'capacity conflict'):
            self.manager.materialize_confirmed_source('a', self.nvme / 'a', StorageTier.HOST)

    def test_local_file_snapshot_cannot_change_frozen_budget_or_admit_unknown_copy(self):
        from faaslora.registry.schema import StorageTier
        self.fetch()
        self.manager.local_file_budgets()
        self.manager.tier_capacities[StorageTier.HOST].total_bytes += 1
        with self.assertRaisesRegex(ValueError, 'budget cannot change'):
            self.manager.local_file_budgets()
        (self.host / 'b').mkdir()
        with self.assertRaisesRegex(ValueError, 'exact confirmed source'):
            self.manager.materialize_confirmed_source('b', self.host / 'b', StorageTier.NVME)
        self.assertFalse((self.nvme / 'b').exists())
        self.assertEqual(self.manager.local_transfer_evidence[-1]['state'], 'rejected')
        self.assertFalse(self.owner.leases or self.owner.materializations)

    def test_stale_snapshot_and_wrong_identity_do_not_create_read_leases(self):
        self.fetch()
        state = self.owner.source_snapshot('a')
        source = state['sources'][0]
        args = dict(path=source['path'], adapter_id='a', lease_id='r',
            expected_owner_id=state['owner_id'], expected_epoch=state['epoch'],
            expected_content_sha256=source['content_sha256'])
        for change in ({'expected_epoch': state['epoch'] - 1}, {'expected_epoch': True},
                       {'expected_owner_id': 'old-owner'}, {'adapter_id': 'other'},
                       {'expected_content_sha256': '0'*64}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                self.owner.acquire_confirmed(**(args | change))
        self.assertFalse(self.owner.leases)
        with self.assertRaisesRegex(ValueError, 'verified adapter identity'):
            self.owner.acquire(path=source['path'], adapter_id='wrong', lease_id='r')

    def test_post_verification_mutation_rejects_before_publication(self):
        from faaslora.storage import http_artifact_store
        original = http_artifact_store._extract_verified
        def mutate(tar, target, expected, check, **kwargs):
            receipt = original(tar, target, expected, check, **kwargs)
            (target / 'nested' / 'weights').write_bytes(b'X'*len(self.payload['nested/weights']))
            return receipt
        with patch.object(http_artifact_store, '_extract_verified', mutate):
            with self.assertRaisesRegex(RuntimeError, 'changed before source publication'):
                self.fetch()
        self.assertFalse((self.nvme / 'a').exists())
        self.assertEqual(self.owner.source_snapshot('a')['sources'], [])
        self.assertEqual(self.runner._remote_transfer_evidence[-1]['state'], 'not_published')

    def test_existing_localhost_server_feeds_actual_runner_confirmed_publication(self):
        from remote_artifact_node.server import ArtifactServer, ArtifactHandler
        from faaslora.storage.http_artifact_store import HttpArtifactStoreClient
        from tests.test_http_artifact_store import content_manifest
        origin = self.root / 'origin'
        for name, data in self.payload.items():
            path = origin / 'a' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        server = ArtifactServer(('127.0.0.1', 0), ArtifactHandler, root=origin, token=None)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = HttpArtifactStoreClient(endpoint=f'http://127.0.0.1:{server.server_port}')
            client.configure_content_manifest(content_manifest(files=self.payload))
            self.runner._remote_artifact_client = client
            ok, _ = asyncio.run(self.runner._materialize_remote_adapter_async('a', self.nvme / 'a'))
            self.assertTrue(ok)
            self.assertEqual(len(self.owner.source_snapshot('a')['sources']), 1)
            self.assertFalse(self.owner.materializations)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class ControllerNativeReferenceLifecycle(unittest.TestCase):
    @staticmethod
    def preload_native(owner, adapter_id='adapter-a'):
        aid = InferenceEngine._lora_int_id(adapter_id)
        snapshot = owner.snapshot()
        owner.demand_load_and_acquire(lease_id='preload', adapter_int_id=aid,
            lora_name=adapter_id, lora_path='/existing/a',
            expected_owner_id=snapshot['owner_id'], expected_epoch=snapshot['epoch'])
        owner.release(lease_id='preload', expected_owner_id=owner.owner_id)
        return aid

    def test_cached_native_weights_do_not_require_the_original_file(self):
        for tier in ('gpu', 'host'):
            with self.subTest(tier=tier):
                runner, slot, trace, plan, owner, rpc = native_reference_fixture()
                aid = InferenceEngine._lora_int_id(trace.adapter_id)
                before = owner.snapshot()
                owner.demand_load_and_acquire(lease_id='preload', adapter_int_id=aid,
                    lora_name=trace.adapter_id, lora_path='/evicted/file/adapter-a',
                    expected_owner_id=before['owner_id'], expected_epoch=before['epoch'])
                owner.release(lease_id='preload', expected_owner_id=owner.owner_id)
                if tier == 'host':
                    owner.manager.deactivate(aid)
                # Faithful native cache-hit branch: activation reuses CPU tensors.
                owner.demand_loader = lambda **kwargs: owner.manager.activate(kwargs['adapter_int_id'])
                runner._stack = SimpleNamespace(record_access=Mock(), residency_manager=SimpleNamespace(
                    acquire_local_source=Mock(side_effect=AssertionError('file lease on cached tensors'))))
                runner._resolve_lora.side_effect = AssertionError('file resolution on a cache hit')
                runner._begin_scaleup_runtime_request_labels.side_effect = ValueError('stop before generation')
                result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
                self.assertIn('stop before generation', result.error)
                runner._resolve_lora.assert_not_awaited()
                runner._stack.residency_manager.acquire_local_source.assert_not_called()
                receipt = result.gpu_reference_evidence['receipt']
                self.assertEqual(receipt['source_tier_before_acquisition'], tier)
                self.assertEqual(receipt['lora_path'], '/evicted/file/adapter-a')
                self.assertNotIn('local_source_reference', result.gpu_reference_evidence)
                self.assertEqual(result.gpu_reference_evidence['state'], 'released')
                self.assertFalse(result.gpu_reference_evidence['confirmed_dispatch_snapshot'])
                self.assertEqual(slot.active_requests, 0)
                self.assertEqual(owner.snapshot()['live_leases'], 0)

    def test_cached_source_change_is_reobserved_before_any_file_read(self):
        for changed_to in ('host', 'unconfirmed_gpu', 'file'):
            with self.subTest(changed_to=changed_to):
                runner, slot, trace, plan, owner, rpc = native_reference_fixture()
                aid = self.preload_native(owner)
                probes = []
                async def changed_once(*, operation, **kwargs):
                    if operation == 'demand_load_and_acquire':
                        probes.append(dict(kwargs))
                        if len(probes) == 1:
                            owner.manager.deactivate(aid)
                            if changed_to == 'file':
                                owner.evict(adapter_int_id=aid)
                            elif changed_to == 'unconfirmed_gpu':
                                owner.manager.activate(aid)
                    return await rpc(operation=operation, **kwargs)
                slot.engine.ieee_gpu_reference.side_effect = changed_once
                runner._begin_scaleup_runtime_request_labels.side_effect = ValueError('pre-generation stop')
                result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
                self.assertIn('pre-generation stop', result.error)
                self.assertEqual(len(probes), 2)
                self.assertEqual(probes[0]['required_source_tier'], 'gpu')
                if changed_to == 'file':
                    runner._resolve_lora.assert_awaited_once()
                    self.assertNotIn('required_source_tier', probes[1])
                else:
                    runner._resolve_lora.assert_not_awaited()
                    self.assertEqual(probes[1]['required_source_tier'], 'host')
                evidence = result.gpu_reference_evidence
                self.assertEqual(len(evidence['cache_source_rechecks']), 1)
                self.assertFalse(evidence['cache_source_rechecks'][0]['conflict']['acquired'])
                self.assertEqual(evidence['receipt']['source_tier_before_acquisition'],
                                 'file' if changed_to == 'file' else 'host')
                self.assertEqual(evidence['state'], 'released')
                self.assertEqual(owner.snapshot()['live_leases'], 0)
                self.assertEqual(slot.active_requests, 0)

    def test_lost_cached_acquisition_retains_native_not_file_ownership(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        self.preload_native(owner)
        async def check():
            entered = asyncio.Event()
            async def lost(*, operation, **kwargs):
                value = await rpc(operation=operation, **kwargs)
                if operation == 'demand_load_and_acquire':
                    entered.set()
                    await asyncio.Future()
                return value
            slot.engine.ieee_gpu_reference.side_effect = lost
            task = asyncio.create_task(runner._exec_request(trace, 4, 0., request_plan=plan))
            await asyncio.wait_for(entered.wait(), .5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(check())
        evidence = runner._unsettled_runtime_reservations[trace.request_id].gpu_reference_evidence
        self.assertEqual(evidence['state'], 'acquiring')
        self.assertNotIn('local_source_reference', evidence)
        self.assertEqual(evidence['intent']['required_source_tier'], 'gpu')
        runner._resolve_lora.assert_not_awaited()
        self.assertEqual(owner.snapshot()['live_leases'], 1)
        self.assertEqual(slot.active_requests, 1)
        self.assertEqual(slot.status, 'draining')

    def test_cached_native_identity_collision_does_not_resolve_or_generate(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        aid = self.preload_native(owner)
        owner._sources[aid] = ('different-adapter', '/existing/a')
        with self.assertRaisesRegex(ValueError, 'another adapter'):
            asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        runner._resolve_lora.assert_not_awaited()
        slot.engine.generate_prepared.assert_not_awaited()
        self.assertEqual(owner.snapshot()['live_leases'], 0)
        self.assertEqual(slot.active_requests, 0)

    def test_native_http_capacity_conflict_rejects_before_reading_body(self):
        from faaslora.memory.residency_manager import ResidencyManager
        from faaslora.registry.schema import StorageTier
        from faaslora.storage.http_artifact_store import HttpArtifactStoreClient
        from tests.test_http_artifact_store import archive_bytes, content_manifest, SizedResponse
        runner, *_ = native_reference_fixture()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'a'
            source.mkdir()
            (source / 'old').write_bytes(b'previous-copy')
            manager = ResidencyManager({'memory': {'nvme': {'cache_dir': directory}}}, Mock(), Mock())
            used = manager.local_file_inventory()['allocated_file_bytes']
            manager.tier_capacities[StorageTier.NVME].total_bytes = used + 4096
            runner._stack = SimpleNamespace(residency_manager=manager)
            client = HttpArtifactStoreClient(endpoint='http://127.0.0.1:1')
            client.configure_content_manifest(content_manifest())
            response = SizedResponse(archive_bytes())
            response.read = Mock(wraps=response.read)
            client._opener = Mock()
            client._opener.open.return_value = response
            runner._remote_artifact_client = client
            with self.assertRaisesRegex(RuntimeError, 'capacity conflict'):
                asyncio.run(runner._materialize_remote_adapter_async('a', source))
            response.read.assert_not_called()
            self.assertEqual((source / 'old').read_bytes(), b'previous-copy')
            self.assertFalse(manager.local_source_references.materializations)
            self.assertEqual(manager.local_file_inventory()['allocated_file_bytes'], used)
            self.assertEqual(runner._remote_transfer_evidence[-1]['state'], 'not_published')
            self.assertEqual(runner._remote_transfer_evidence[-1]['transferred_bytes'], 0)

    def test_remote_writer_runs_off_loop_and_joins_actual_thread_on_repeated_cancel(self):
        async def check():
            entered, proceed, exited = (threading.Event() for _ in range(3))
            controller_thread = threading.get_ident()
            observed = {}
            def io_work(cancellation):
                observed['thread'] = threading.get_ident()
                observed['cancellation'] = cancellation
                entered.set()
                try:
                    if not proceed.wait(2):
                        raise RuntimeError('test barrier timeout')
                    return 'completed'
                finally:
                    exited.set()
            task = asyncio.create_task(ScenarioRunner._owned_artifact_io(io_work))
            try:
                while not entered.is_set():
                    await asyncio.sleep(0)
                self.assertNotEqual(observed['thread'], controller_thread)
                task.cancel()
                await asyncio.sleep(.01)
                self.assertTrue(observed['cancellation'].is_set())
                self.assertFalse(task.done())
                task.cancel()
                await asyncio.sleep(.01)
                self.assertFalse(task.done())
            finally:
                proceed.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertTrue(exited.is_set())
        asyncio.run(check())

    def test_native_http_publication_obeys_read_owner_and_does_not_hide_failure(self):
        from faaslora.memory.residency_manager import ResidencyManager
        from faaslora.storage.http_artifact_store import HttpArtifactStoreClient
        from tests.test_http_artifact_store import archive_bytes, content_manifest, SizedResponse
        import io
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'a'
            source.mkdir()
            (source / 'old').write_bytes(b'previous-copy')
            manager = ResidencyManager({'memory': {'nvme': {'cache_dir': directory}}}, Mock(), Mock())
            reference = manager.acquire_local_source(path=str(source), adapter_id='a', lease_id='reader')
            runner._stack = SimpleNamespace(residency_manager=manager)
            client = HttpArtifactStoreClient(endpoint='http://127.0.0.1:1')
            client.configure_content_manifest(content_manifest())
            client._opener = Mock()
            client._opener.open.side_effect = lambda *a, **k: SizedResponse(archive_bytes())
            runner._remote_artifact_client = client
            with self.assertRaisesRegex(RuntimeError, 'live source reference'):
                asyncio.run(runner._materialize_remote_adapter_async('a', source))
            self.assertEqual((source / 'old').read_bytes(), b'previous-copy')
            self.assertFalse(manager.local_source_references.materializations)
            manager.release_local_source(lease_id='reader', expected_owner_id=reference['owner_id'])
            ok, elapsed = asyncio.run(runner._materialize_remote_adapter_async('a', source))
            self.assertTrue(ok)
            self.assertGreater(elapsed, 0)
            self.assertTrue((source / 'adapter_model.safetensors').exists())
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ['a'])
            self.assertEqual([row['state'] for row in runner._remote_transfer_evidence],
                             ['not_published', 'published'])
            self.assertTrue(runner._remote_transfer_evidence[-1]['content_verified'])
            reservation = runner._remote_transfer_evidence[-1]['file_reservation']
            self.assertEqual(reservation['scope'], 'preallocated_regular_files_v1')
            self.assertEqual(reservation['reserved_file_bytes'], 8192)
            self.assertEqual(reservation['used_file_bytes_before'], 4096)
            self.assertNotEqual(runner._remote_transfer_evidence[0]['transfer_id'],
                                runner._remote_transfer_evidence[1]['transfer_id'])
            self.assertEqual(runner._remote_transfer_evidence[-1]['local_source_owner_id'],
                             manager.local_source_references.owner_id)

    def test_actual_http_client_cancel_waits_for_reader_and_cleans_without_publication(self):
        from faaslora.memory.residency_manager import ResidencyManager
        from faaslora.storage.http_artifact_store import HttpArtifactStoreClient
        from tests.test_http_artifact_store import archive_bytes, content_manifest, SizedResponse
        import io
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'a'
            source.mkdir()
            (source / 'old').write_bytes(b'previous-copy')
            manager = ResidencyManager({'memory': {'nvme': {'cache_dir': directory}}}, Mock(), Mock())
            runner._stack = SimpleNamespace(residency_manager=manager)
            client = HttpArtifactStoreClient(endpoint='http://127.0.0.1:1')
            client.configure_content_manifest(content_manifest())
            entered, proceed = threading.Event(), threading.Event()
            class HeldResponse(SizedResponse):
                def read(inner, *args):
                    entered.set()
                    if not proceed.wait(2):
                        raise RuntimeError('test barrier timeout')
                    return super().read(*args)
            client._opener = Mock()
            client._opener.open.return_value = HeldResponse(archive_bytes())
            runner._remote_artifact_client = client
            async def check():
                task = asyncio.create_task(runner._materialize_remote_adapter_async('a', source))
                try:
                    while not entered.is_set():
                        await asyncio.sleep(0)
                    task.cancel()
                    await asyncio.sleep(.01)
                    self.assertFalse(task.done())
                    self.assertTrue(manager.local_source_references.materializations)
                    self.assertEqual(manager.local_file_inventory()['transfer_held_file_bytes'], 8192)
                    self.assertFalse(manager._delete_path(directory))
                finally:
                    proceed.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
            asyncio.run(check())
            self.assertFalse(manager.local_source_references.materializations)
            self.assertEqual((source / 'old').read_bytes(), b'previous-copy')
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ['a'])

    def test_lost_load_reply_retains_managed_file_until_native_reconciliation(self):
        from faaslora.memory.residency_manager import ResidencyManager
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'adapter-a'
            source.mkdir()
            manager = ResidencyManager({'memory': {'nvme': {'cache_dir': directory}}}, Mock(), Mock())
            runner._stack = SimpleNamespace(residency_manager=manager, record_access=Mock())
            runner._resolve_lora.return_value = ('adapter-a', str(source), 1., 'nvme', 0., 0.)
            async def check():
                entered = asyncio.Event()
                async def lost(*, operation, **kwargs):
                    value = await rpc(operation=operation, **kwargs)
                    if operation == 'demand_load_and_acquire':
                        entered.set()
                        await asyncio.Future()
                    return value
                slot.engine.ieee_gpu_reference.side_effect = lost
                task = asyncio.create_task(runner._exec_request(trace, 4, 0., request_plan=plan))
                await asyncio.wait_for(entered.wait(), .5)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            asyncio.run(check())
            pending = runner._unsettled_runtime_reservations[trace.request_id]
            self.assertEqual(pending.gpu_reference_evidence['local_source_reference']['state'], 'held')
            self.assertFalse(manager._delete_path(str(source)))
            self.assertTrue(source.exists())
            self.assertEqual(owner.snapshot()['live_leases'], 1)

    def test_read_only_snapshot_failure_releases_file_without_claiming_native_work(self):
        from faaslora.memory.residency_manager import ResidencyManager
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'adapter-a'
            source.mkdir()
            manager = ResidencyManager({'memory': {'nvme': {'cache_dir': directory}}}, Mock(), Mock())
            runner._stack = SimpleNamespace(residency_manager=manager, record_access=Mock())
            runner._resolve_lora.return_value = ('adapter-a', str(source), 1., 'nvme', 0., 0.)
            snapshots = []
            async def fail_after_file_resolution(*, operation, **kwargs):
                if operation == 'source_snapshot':
                    snapshots.append(operation)
                    if len(snapshots) == 2:
                        raise RuntimeError('snapshot unavailable')
                return await rpc(operation=operation, **kwargs)
            slot.engine.ieee_gpu_reference.side_effect = fail_after_file_resolution
            result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
            self.assertEqual(result.gpu_reference_evidence['local_source_reference']['state'], 'released')
            self.assertTrue(manager._delete_path(str(source)))
            self.assertEqual(slot.active_requests, 0)
            self.assertEqual(owner.snapshot()['live_leases'], 0)

    def test_managed_file_is_retained_during_native_load_and_released_at_ack(self):
        from faaslora.memory.residency_manager import ResidencyManager
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'nvme' / 'adapter-a'
            source.mkdir(parents=True)
            (source / 'weights').write_bytes(b'tiny-test-fixture')
            manager = ResidencyManager({'memory': {
                'host': {'cache_dir': str(root / 'host')},
                'nvme': {'cache_dir': str(root / 'nvme')}}}, Mock(), Mock())
            runner._stack = SimpleNamespace(residency_manager=manager, record_access=Mock())
            runner._resolve_lora.return_value = ('adapter-a', str(source), 1., 'nvme', 0., 0.)
            async def observing(*, operation, **kwargs):
                if operation == 'demand_load_and_acquire':
                    self.assertFalse(manager._delete_path(str(source)))
                    self.assertTrue(source.exists())
                return await rpc(operation=operation, **kwargs)
            slot.engine.ieee_gpu_reference.side_effect = observing
            def after_load(**kwargs):
                self.assertTrue(manager._delete_path(str(source)))
                raise ValueError('stop after acknowledged copy')
            runner._begin_scaleup_runtime_request_labels.side_effect = after_load
            result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
            self.assertIn('acknowledged copy', result.error)
            self.assertEqual(result.gpu_reference_evidence['local_source_reference']['state'], 'released')
            self.assertEqual(result.gpu_reference_evidence['local_source_reference']['file_footprint']
                             ['logical_file_bytes'], len(b'tiny-test-fixture'))
            self.assertEqual(owner.snapshot()['live_leases'], 0)

    def test_measured_footprints_reach_request_without_duplicating_tensor_inventory(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        runner._begin_scaleup_runtime_request_labels.side_effect = ValueError('pre-generation stop')
        async def measured(*, operation, **kwargs):
            value = await rpc(operation=operation, **kwargs)
            if operation == 'source_snapshot':
                value['native_footprints'] = dict(uniform_slot_layout=True,
                    host_footprint_scope='native_registered_tensor_storage_capacity',
                    host_budget_reserved=False, host_allocator_overhead_included=False,
                    slot_adapter_ids=[None, None], registered_cpu_adapter_ids=[],
                    slot_capacity_bytes=32, pool_allocated_bytes=64,
                    host_tensor_storage_bytes=0, host_allocations=[], host_adapter_footprints=[],
                    pool_tensor_views=[dict(dtype='torch.float16')])
            return value
        slot.engine.ieee_gpu_reference.side_effect = measured
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        snapshot = result.gpu_reference_evidence['snapshot_before_acquisition']
        self.assertNotIn('native_footprints', snapshot)
        self.assertIsNone(snapshot['selected_source_footprint'])  # Correctly cold before this load.
        self.assertEqual(snapshot['host_tensor_storage_bytes'], 0)
        self.assertEqual(snapshot['gpu_pool_storage_bytes'], 64)
        self.assertEqual(slot.native_source_state.gpu_pool_storage_bytes, 64)
        self.assertEqual(result.gpu_reference_evidence['state'], 'released')

    def test_actual_runner_acquires_before_generation_and_releases_after_terminal(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        async def generate(**kwargs):
            reference = kwargs['gpu_reference']
            aid = InferenceEngine._lora_int_id(trace.adapter_id)
            self.assertEqual(reference['adapter_int_id'], aid)
            self.assertEqual(owner.snapshot()['reference_counts'], {str(aid): 1})
            self.assertEqual(slot.active_requests, 1)
            owner.begin_use(lease_id=reference['lease_id'], expected_owner_id=reference['owner_id'],
                            adapter_int_id=aid, backend_request_id='native-1',
                            lora_name=trace.adapter_id, lora_path='/existing/a')
            owner.end_use(lease_id=reference['lease_id'], expected_owner_id=reference['owner_id'],
                          backend_request_id='native-1')
            return 10., 2., 4, {'native_terminal_observed': True,
                'gpu_reference_owner_id': reference['owner_id'],
                'gpu_reference_lease_id': reference['lease_id'],
                'gpu_reference_adapter_int_id': aid}
        slot.engine.generate_prepared.side_effect = generate
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertEqual(owner.snapshot()['released_leases'], 1)
        self.assertEqual(owner.snapshot()['live_leases'], 0)
        self.assertEqual(slot.active_requests, 0)
        self.assertFalse(result.success)  # Deliberately missing native token timing.
        self.assertEqual(result.gpu_reference_evidence['state'], 'released')
        self.assertFalse(result.gpu_reference_evidence['confirmed_dispatch_snapshot'])
        self.assertFalse(result.gpu_reference_evidence['receipt']['gpu_resident_before_load'])
        self.assertEqual(slot.native_source_state.owner_id, owner.owner_id)
        self.assertFalse(slot.native_source_state.sources)  # Captured before this cold load.

    def test_cancel_during_acquire_retains_unknown_native_ownership(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        async def check():
            entered = asyncio.Event()
            async def delayed_reply(*, operation, **kwargs):
                value = await rpc(operation=operation, **kwargs)
                if operation == 'demand_load_and_acquire':
                    entered.set()
                    await asyncio.Future()
                return value
            slot.engine.ieee_gpu_reference.side_effect = delayed_reply
            task = asyncio.create_task(runner._exec_request(trace, 4, 0., request_plan=plan))
            # The prechange runner never calls the native reference interface.
            await asyncio.wait_for(entered.wait(), .25)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(check())
        self.assertEqual(owner.snapshot()['live_leases'], 1)
        self.assertEqual(slot.active_requests, 1)
        self.assertEqual(slot.status, 'draining')
        pending = runner._unsettled_runtime_reservations[trace.request_id]
        self.assertEqual(pending.gpu_reference_evidence['state'], 'acquiring')
        self.assertFalse(pending.generation_started)
        slot.engine.generate_prepared.assert_not_awaited()

    def test_unacknowledged_generation_retains_both_controller_and_native_owners(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        slot.engine.generate_prepared.side_effect = RuntimeError('stream lost')
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertFalse(result.success)
        self.assertEqual(owner.snapshot()['live_leases'], 1)
        self.assertEqual(slot.active_requests, 1)
        self.assertEqual(slot.status, 'draining')
        self.assertEqual(result.gpu_reference_evidence['state'], 'acquired')

    def test_error_after_acquisition_before_generate_releases_the_unused_reference(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        runner._begin_scaleup_runtime_request_labels.side_effect = ValueError('bad activation identity')
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertFalse(result.success)
        self.assertEqual(result.gpu_reference_evidence['state'], 'released')
        self.assertEqual(owner.snapshot()['released_leases'], 1)
        self.assertEqual(slot.active_requests, 0)
        slot.engine.generate_prepared.assert_not_awaited()

    def test_worker_conflict_does_not_load_or_generate_or_claim_an_unknown_lease(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        async def reject(*, operation, **kwargs):
            if operation == 'source_snapshot':
                return await rpc(operation=operation)
            snapshot = await rpc(operation='snapshot')
            return snapshot if operation == 'snapshot' else {
                **snapshot, 'acquired': False, 'reason': 'all_gpu_slots_pinned'}
        slot.engine.ieee_gpu_reference.side_effect = reject
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertFalse(result.success)
        self.assertEqual(result.gpu_reference_evidence['state'], 'rejected')
        self.assertEqual(slot.active_requests, 0)
        self.assertEqual(owner.snapshot()['live_leases'], 0)
        self.assertFalse(runner._unsettled_runtime_reservations)
        slot.engine.generate_prepared.assert_not_awaited()

    def test_stale_epoch_rechecks_native_state_without_repeating_a_load(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        epochs = []
        async def change_once(*, operation, **kwargs):
            if operation == 'demand_load_and_acquire':
                epochs.append(kwargs['expected_epoch'])
                if len(epochs) == 1:
                    owner.epoch += 1  # Another serialized native owner event.
            return await rpc(operation=operation, **kwargs)
        slot.engine.ieee_gpu_reference.side_effect = change_once
        runner._begin_scaleup_runtime_request_labels.side_effect = ValueError('end test before generate')
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertEqual(len(epochs), 2)
        self.assertEqual(epochs[1], epochs[0]+1)
        self.assertEqual(result.gpu_reference_evidence['stale_rechecks'], 1)
        self.assertEqual(owner.snapshot()['released_leases'], 1)

    def test_wrong_terminal_reference_does_not_release_a_different_request(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        async def wrong(**kwargs):
            ref = kwargs['gpu_reference']
            return 10., 2., 4, {'native_terminal_observed': True,
                'gpu_reference_owner_id': ref['owner_id'], 'gpu_reference_lease_id': 'other',
                'gpu_reference_adapter_int_id': ref['adapter_int_id']}
        slot.engine.generate_prepared.side_effect = wrong
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertIn('another GPU reference', result.error)
        self.assertEqual(owner.snapshot()['live_leases'], 1)
        self.assertEqual(slot.active_requests, 1)
        self.assertFalse(runner._unsettled_runtime_reservations[trace.request_id].native_terminal_observed)

    def test_lost_release_reply_keeps_controller_capacity_and_intent(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        runner._begin_scaleup_runtime_request_labels.side_effect = ValueError('pre-generation stop')
        async def lose_release(*, operation, **kwargs):
            value = await rpc(operation=operation, **kwargs)
            if operation == 'release':
                raise OSError('release reply lost')
            return value
        slot.engine.ieee_gpu_reference.side_effect = lose_release
        with self.assertRaisesRegex(OSError, 'release reply lost'):
            asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertEqual(owner.snapshot()['live_leases'], 0)
        self.assertEqual(slot.active_requests, 1)  # Caller has no release acknowledgement.
        pending = runner._unsettled_runtime_reservations[trace.request_id]
        self.assertEqual(pending.gpu_reference_evidence['state'], 'release_pending')
        self.assertEqual(slot.status, 'draining')

    def test_mismatched_clock_snapshot_has_no_mutation(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        async def wrong_clock(*, operation, **kwargs):
            return {**await rpc(operation=operation, **kwargs), 'clock_id': 'different-host'}
        slot.engine.ieee_gpu_reference.side_effect = wrong_clock
        with self.assertRaisesRegex(ValueError, 'clock identity'):
            asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        runner._resolve_lora.assert_not_awaited()
        self.assertEqual(owner.snapshot()['live_leases'], 0)
        self.assertEqual(slot.active_requests, 0)
        slot.engine.generate_prepared.assert_not_awaited()

    def test_malformed_acquire_reply_cannot_assert_that_no_native_work_occurred(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        async def broken(*, operation, **kwargs):
            value = await rpc(operation=operation, **kwargs)
            if operation == 'demand_load_and_acquire':
                value['lora_name'] = 'wrong-source'
            return value
        slot.engine.ieee_gpu_reference.side_effect = broken
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertFalse(result.success)
        self.assertEqual(owner.snapshot()['live_leases'], 1)
        self.assertEqual(slot.active_requests, 1)
        self.assertEqual(result.gpu_reference_evidence['state'], 'acquiring')
        slot.engine.generate_prepared.assert_not_awaited()

    def test_native_owner_can_reject_release_despite_a_controller_terminal_flag(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        async def premature(**kwargs):
            ref = kwargs['gpu_reference']
            owner.begin_use(lease_id=ref['lease_id'], expected_owner_id=ref['owner_id'],
                            adapter_int_id=ref['adapter_int_id'], backend_request_id='still-running',
                            lora_name=trace.adapter_id, lora_path='/existing/a')
            return 10., 2., 4, {'native_terminal_observed': True,
                'gpu_reference_owner_id': ref['owner_id'], 'gpu_reference_lease_id': ref['lease_id'],
                'gpu_reference_adapter_int_id': ref['adapter_int_id']}
        slot.engine.generate_prepared.side_effect = premature
        with self.assertRaisesRegex(RuntimeError, 'release lacks matching acknowledgement'):
            asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertEqual(owner.snapshot()['live_leases'], 1)
        self.assertEqual(slot.active_requests, 1)
        self.assertEqual(slot.status, 'draining')

    def test_two_controller_requests_share_native_pins_but_not_lease_identity(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        async def check():
            reservations = []
            for name in ('request-a', 'request-b'):
                reservation = RuntimeRequestReservation(name)
                success, adapter_reserved = runner._try_reserve_runtime_request_slot(slot, 'adapter-a')
                self.assertTrue(success)
                reservation.bind(slot, 'adapter-a', adapter_reserved)
                reservations.append(reservation)
                await runner._acquire_runtime_gpu_reference(reservation, slot.engine, 'adapter-a', '/existing/a')
            receipts = [r.gpu_reference_evidence['receipt'] for r in reservations]
            self.assertNotEqual(receipts[0]['lease_id'], receipts[1]['lease_id'])
            aid = receipts[0]['adapter_int_id']
            self.assertEqual(owner.snapshot()['reference_counts'], {str(aid): 2})
            self.assertFalse(receipts[0]['gpu_resident_before_load'])
            self.assertTrue(receipts[1]['gpu_resident_before_load'])
            await runner._finish_runtime_request_reservation(reservations[0])
            self.assertEqual(owner.snapshot()['reference_counts'], {str(aid): 1})
            self.assertEqual(slot.active_requests, 1)
            await runner._finish_runtime_request_reservation(reservations[1])
            self.assertFalse(owner.snapshot()['reference_counts'])
            self.assertEqual(slot.active_requests, 0)
        asyncio.run(check())


class NativeCapacityWait(unittest.TestCase):
    """Real runner/reference owner, CPU LRU fixtures; no GPU performance claim."""
    def bind(self, runner, engine, name, adapter):
        slot = InstanceSlot(name, engine=engine, coordinator=None)
        ok, reserved = runner._try_reserve_runtime_request_slot(slot, adapter)
        self.assertTrue(ok)
        reservation = RuntimeRequestReservation(name)
        reservation.bind(slot, adapter, reserved)
        return reservation

    async def acquire(self, runner, engine, name, adapter):
        reservation = self.bind(runner, engine, name, adapter)
        await runner._acquire_runtime_gpu_reference(reservation, engine, adapter, '/existing/'+adapter)
        return reservation

    def test_waits_for_last_shared_reference_not_first_release(self):
        async def check():
            runner, slot, _, _, owner, rpc = native_reference_fixture()
            engine = slot.engine
            a = await self.acquire(runner, engine, 'a', 'adapter-a')
            a2 = await self.acquire(runner, engine, 'a2', 'adapter-a')
            b = await self.acquire(runner, engine, 'b', 'adapter-b')
            entered = asyncio.Event()
            calls = []
            async def observe(*, operation, **kwargs):
                value = await rpc(operation=operation, **kwargs)
                if value.get('reason') == 'all_gpu_slots_pinned':
                    calls.append(value)
                    entered.set()
                return value
            engine.ieee_gpu_reference.side_effect = observe
            c = self.bind(runner, engine, 'c', 'adapter-c')
            task = asyncio.create_task(runner._acquire_runtime_gpu_reference(c, engine, 'adapter-c', '/existing/adapter-c'))
            await asyncio.wait_for(entered.wait(), 1)
            await runner._finish_runtime_request_reservation(a)
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertEqual(len(calls), 1)  # No native polling after a partial release.
            await runner._finish_runtime_request_reservation(a2)
            receipt = await asyncio.wait_for(task, 1)
            self.assertTrue(receipt['acquired'])
            self.assertEqual(c.gpu_reference_evidence['capacity_waits'][0]['outcome'], 'release_observed')
            await runner._finish_runtime_request_reservation(b)
            await runner._finish_runtime_request_reservation(c)
            self.assertEqual(owner.snapshot()['live_leases'], 0)
        asyncio.run(check())

    def test_release_before_conflict_reply_is_not_a_lost_wakeup(self):
        async def check():
            runner, slot, _, _, owner, rpc = native_reference_fixture()
            a = await self.acquire(runner, slot.engine, 'a', 'adapter-a')
            b = await self.acquire(runner, slot.engine, 'b', 'adapter-b')
            async def early_release(*, operation, **kwargs):
                value = await rpc(operation=operation, **kwargs)
                if value.get('reason') == 'all_gpu_slots_pinned':
                    await runner._finish_runtime_request_reservation(a)
                return value
            slot.engine.ieee_gpu_reference.side_effect = early_release
            c = await asyncio.wait_for(self.acquire(runner, slot.engine, 'c', 'adapter-c'), 1)
            self.assertEqual(len(c.gpu_reference_evidence['capacity_waits']), 1)
            await runner._finish_runtime_request_reservation(b)
            await runner._finish_runtime_request_reservation(c)
            self.assertEqual(owner.snapshot()['live_leases'], 0)
        asyncio.run(check())

    def test_cancelled_waiter_does_not_cancel_owners_or_retain_fake_acquisition(self):
        async def check():
            runner, slot, _, _, owner, rpc = native_reference_fixture()
            a = await self.acquire(runner, slot.engine, 'a', 'adapter-a')
            b = await self.acquire(runner, slot.engine, 'b', 'adapter-b')
            entered = asyncio.Event()
            async def observe(*, operation, **kwargs):
                value = await rpc(operation=operation, **kwargs)
                if value.get('reason') == 'all_gpu_slots_pinned':
                    entered.set()
                return value
            slot.engine.ieee_gpu_reference.side_effect = observe
            c = self.bind(runner, slot.engine, 'c', 'adapter-c')
            task = asyncio.create_task(runner._acquire_runtime_gpu_reference(c, slot.engine, 'adapter-c', '/existing/adapter-c'))
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await runner._finish_runtime_request_reservation(c)
            self.assertTrue(c.released)
            self.assertFalse(runner._unsettled_runtime_reservations)
            self.assertEqual(owner.snapshot()['live_leases'], 2)
            for request in (a, b):
                intent = request.gpu_reference_evidence['intent']
                witness = runner._native_reference_witnesses[(owner.owner_id, intent['lease_id'])]
                self.assertFalse(witness['completion'].done())
                await runner._finish_runtime_request_reservation(request)
            self.assertEqual(owner.snapshot()['live_leases'], 0)
        asyncio.run(check())

    def test_unknown_release_is_not_free_capacity(self):
        async def check():
            runner, slot, _, _, owner, rpc = native_reference_fixture()
            a = await self.acquire(runner, slot.engine, 'a', 'adapter-a')
            b = await self.acquire(runner, slot.engine, 'b', 'adapter-b')
            entered = asyncio.Event()
            async def observe(*, operation, **kwargs):
                value = await rpc(operation=operation, **kwargs)
                if value.get('reason') == 'all_gpu_slots_pinned':
                    entered.set()
                return value
            slot.engine.ieee_gpu_reference.side_effect = observe
            c = self.bind(runner, slot.engine, 'c', 'adapter-c')
            task = asyncio.create_task(runner._acquire_runtime_gpu_reference(c, slot.engine, 'adapter-c', '/existing/adapter-c'))
            await asyncio.wait_for(entered.wait(), 1)
            for request in (a, b):
                runner._retain_runtime_request_reservation(request)
            with self.assertRaisesRegex(RuntimeError, 'no acknowledged release path'):
                await asyncio.wait_for(task, 1)
            await runner._finish_runtime_request_reservation(c)
            self.assertEqual(owner.snapshot()['live_leases'], 2)
            self.assertEqual(c.gpu_reference_evidence['state'], 'rejected')
            # End the fixture through actual successful releases, not forged frees.
            for request in (a, b):
                await runner._finish_runtime_request_reservation(request)
        asyncio.run(check())

    def test_full_host_request_includes_capacity_wait_in_its_fixed_source_interval(self):
        async def check():
            setup = SelectedSourceAdmissionIntegration()
            try:
                runner, slot, trace, plan, owner, rpc = setup.build('host')
                # Native owners share a runtime. Source admission does not
                # disguise another logical slot's physical pin as free capacity.
                runner._stack, stack = None, runner._stack
                a = await self.acquire(runner, slot.engine, 'busy-b', 'adapter-b')
                b = await self.acquire(runner, slot.engine, 'busy-c', 'adapter-c')
                runner._stack = stack
                entered = asyncio.Event()
                async def observe(*, operation, **kwargs):
                    value = await rpc(operation=operation, **kwargs)
                    if value.get('reason') == 'all_gpu_slots_pinned':
                        entered.set()
                    return value
                slot.engine.ieee_gpu_reference.side_effect = observe
                task = asyncio.create_task(runner._exec_request(trace, 4, 0., request_plan=plan))
                await asyncio.wait_for(entered.wait(), 1)
                self.assertFalse(task.done())
                await runner._finish_runtime_request_reservation(a)
                result = await asyncio.wait_for(task, 1)
                self.assertTrue(result.success, result.error)
                evidence = result.gpu_reference_evidence
                interval, wait = evidence['service_intervals'], evidence['capacity_waits'][0]
                self.assertEqual(result.readiness_tier_before_dispatch, 'host')
                self.assertLessEqual(interval['admitted_monotonic_s'], wait['start_monotonic_s'])
                self.assertLessEqual(wait['end_monotonic_s'], interval['acquired_monotonic_s'])
                self.assertEqual(owner.snapshot()['live_host_source_leases'], 0)
                await runner._finish_runtime_request_reservation(b)
                self.assertEqual(owner.snapshot()['live_leases'], 0)
                runner._resolve_lora.assert_not_awaited()
            finally:
                setup.doCleanups()
        asyncio.run(check())

    def test_cpu_cache_waits_for_host_reference_release(self):
        async def check():
            runner, slot, _, _, owner, rpc = native_reference_fixture()
            holders = []
            for name in ('a', 'b', 'c'):
                adapter = 'adapter-'+name
                loaded = await self.acquire(runner, slot.engine, 'load-'+name, adapter)
                await runner._finish_runtime_request_reservation(loaded)
                aid = InferenceEngine._lora_int_id(adapter)
                owner.manager.deactivate(aid)
                held = self.bind(runner, slot.engine, 'hold-'+name, adapter)
                held.gpu_reference_engine = slot.engine
                intent = dict(lease_id='host-'+name, adapter_int_id=aid, lora_name=adapter,
                    lora_path='/existing/'+adapter, expected_owner_id=owner.owner_id,
                    expected_epoch=owner.snapshot()['epoch'])
                held.gpu_reference_evidence['native_host_source'] = dict(state='held', intent=intent,
                    receipt=await rpc(operation='hold_host_source', **intent))
                runner._track_native_reference_intent(held, intent, 'host')
                holders.append(held)
            entered = asyncio.Event()
            async def observe(*, operation, **kwargs):
                value = await rpc(operation=operation, **kwargs)
                if value.get('reason') == 'all_cpu_entries_pinned':
                    self.assertEqual(value['capacity_blockers']['tier'], 'host')
                    entered.set()
                return value
            slot.engine.ieee_gpu_reference.side_effect = observe
            task = asyncio.create_task(self.acquire(runner, slot.engine, 'd', 'adapter-d'))
            await asyncio.wait_for(entered.wait(), 1)
            self.assertFalse(task.done())
            await runner._finish_runtime_request_reservation(holders[0])
            loaded = await asyncio.wait_for(task, 1)
            self.assertEqual(loaded.gpu_reference_evidence['capacity_waits'][0]['reason'], 'all_cpu_entries_pinned')
            for held in holders[1:]+[loaded]:
                await runner._finish_runtime_request_reservation(held)
            self.assertEqual(owner.snapshot()['live_host_source_leases'], 0)
            self.assertEqual(owner.snapshot()['live_leases'], 0)
        asyncio.run(check())

    def test_unowned_native_pins_fail_explicitly_without_stealing_capacity(self):
        async def check():
            runner, slot, _, _, owner, _ = native_reference_fixture()
            for name in ('a', 'b'):
                loaded = await self.acquire(runner, slot.engine, 'load-'+name, 'adapter-'+name)
                await runner._finish_runtime_request_reservation(loaded)
                aid = InferenceEngine._lora_int_id('adapter-'+name)
                owner.manager._registered_adapters.pin(aid)
                owner.manager._active_adapters.pin(aid)
            before = owner.snapshot()
            blocked = self.bind(runner, slot.engine, 'c', 'adapter-c')
            with self.assertRaisesRegex(RuntimeError, 'no acknowledged release path'):
                await asyncio.wait_for(runner._acquire_runtime_gpu_reference(
                    blocked, slot.engine, 'adapter-c', '/existing/adapter-c'), 1)
            self.assertEqual(owner.snapshot(), before)
            await runner._finish_runtime_request_reservation(blocked)
            self.assertTrue(blocked.released)
            self.assertEqual(len(owner.manager._active_adapters.pinned_items), 2)
        asyncio.run(check())

    def test_wake_does_not_grant_capacity_and_two_waiters_recheck_native_victims(self):
        async def check():
            runner, slot, _, _, owner, rpc = native_reference_fixture()
            a = await self.acquire(runner, slot.engine, 'a', 'adapter-a')
            b = await self.acquire(runner, slot.engine, 'b', 'adapter-b')
            entered, conflicts = asyncio.Event(), []
            async def observe(*, operation, **kwargs):
                value = await rpc(operation=operation, **kwargs)
                if value.get('reason') == 'all_gpu_slots_pinned':
                    conflicts.append(value)
                    if len(conflicts) >= 2:
                        entered.set()
                return value
            slot.engine.ieee_gpu_reference.side_effect = observe
            tasks = [asyncio.create_task(self.acquire(runner, slot.engine, name, 'adapter-'+name))
                     for name in ('c', 'd')]
            await asyncio.wait_for(entered.wait(), 1)
            await runner._finish_runtime_request_reservation(a)
            done, pending = await asyncio.wait(tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED)
            self.assertEqual(len(done), 1)
            first = next(iter(done)).result()
            self.assertEqual(owner.snapshot()['live_leases'], 2)
            await runner._finish_runtime_request_reservation(first)
            remaining = await asyncio.wait_for(next(iter(pending)), 1)
            for request in (remaining, b):
                await runner._finish_runtime_request_reservation(request)
            self.assertEqual(owner.snapshot()['live_leases'], 0)
        asyncio.run(check())

    def test_native_qualification_keeps_first_distinct_input_rows_and_actual_release_path(self):
        from scripts.ieee_tc_preflight import qualify_native_capacity_wait
        from faaslora.clock import local_monotonic_clock_id
        async def check():
            runner, slot, _, _, owner, rpc = native_reference_fixture()
            engine = slot.engine
            engine._lora_int_id = InferenceEngine._lora_int_id
            engine.prepare_request = lambda prompt, target, inputs, **kw: RequestExecutionPlan('canonical', 2, target)
            async def generate(**kwargs):
                await asyncio.sleep(0)  # Fixture scheduling; not production waiting logic.
                ref = kwargs['gpu_reference']
                return 1., 1., kwargs['request_plan'].max_tokens, dict(native_terminal_observed=True,
                    native_clock_id=local_monotonic_clock_id(), gpu_reference_owner_id=ref['owner_id'],
                    gpu_reference_lease_id=ref['lease_id'], gpu_reference_adapter_int_id=ref['adapter_int_id'])
            engine.generate_prepared.side_effect = generate
            entries = [SimpleNamespace(request_id='req-'+str(i), source_sha256=str(i)*64,
                source_json=json.dumps(dict(adapter_id='adapter-'+name, expected_output_tokens=4,
                    expected_input_tokens=2, body=dict(messages=[dict(role='user', content='existing')]))))
                for i, name in enumerate(('a', 'a', 'b', 'c', 'd'))]
            result = dict(model_config=runner.model_cfg, requests=[], sources_before=await rpc(operation='source_snapshot'))
            adapters = {'adapter-'+name: dict(path='/existing/adapter-'+name) for name in ('a','b','c','d')}
            await qualify_native_capacity_wait(engine, SimpleNamespace(entries=entries), adapters, result)
            self.assertEqual(result['selected_request_ids'], ['req-0', 'req-2', 'req-3'])
            self.assertTrue(all(row['pass'] for row in result['requests']))
            self.assertEqual(result['capacity_ownership_after']['live_leases'], 0)
            self.assertFalse(result['router_qualified'])
            self.assertFalse(result['physical_capacity_qualified'])
            waits = result['requests'][-1]['source_evidence']['capacity_waits']
            self.assertEqual(waits[0]['outcome'], 'release_observed')
            self.assertTrue(all(row['source_evidence']['state'] == 'released' for row in result['requests']))
        asyncio.run(check())


class NativeRPCOwnership(unittest.TestCase):
    def proxy(self):
        proxy = SubprocessInferenceEngineProxy.__new__(SubprocessInferenceEngineProxy)
        proxy.model_cfg = {'timing_contract': 'ieee_tc_native_v1'}
        proxy._engine_dead = False
        proxy._native_rpc_uncertain = {}
        proxy._rpc_channel_queue = None
        proxy._rpc_channels = []
        proxy._process = SimpleNamespace(poll=Mock(return_value=None))
        proxy._acquire_rpc_channel = AsyncMock(return_value=object())
        proxy._open_rpc_channel = AsyncMock(return_value=object())
        proxy._drop_rpc_channel = AsyncMock()
        proxy._release_rpc_channel = AsyncMock()
        proxy._with_worker_log_context = lambda value: value
        return proxy

    def test_unknown_native_outcome_is_not_reexecuted_by_transport_retry(self):
        proxy = self.proxy()
        proxy._blocking_rpc_roundtrip = Mock(side_effect=OSError('reply lost after submit'))
        with self.assertRaisesRegex(RuntimeError, 'reply lost after submit'):
            asyncio.run(proxy._rpc('generate', prompt='existing'))
        self.assertEqual(proxy._blocking_rpc_roundtrip.call_count, 1)
        proxy._open_rpc_channel.assert_not_awaited()
        proxy._release_rpc_channel.assert_not_awaited()

    def test_cancelled_thread_roundtrip_cannot_return_its_channel_to_the_pool(self):
        proxy = self.proxy()
        started, finish = threading.Event(), threading.Event()
        def blocking(*args):
            started.set()
            if not finish.wait(2.):
                raise RuntimeError('test roundtrip was not joined')
            return b'{"ok": true, "result": {}}\n', 0., 0., time.time()
        proxy._blocking_rpc_roundtrip = Mock(side_effect=blocking)
        async def check():
            task = asyncio.create_task(proxy._rpc('ieee_gpu_reference', operation='acquire'))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 1.))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                finish.set()
        asyncio.run(check())
        proxy._release_rpc_channel.assert_not_awaited()
        proxy._drop_rpc_channel.assert_awaited_once()
        self.assertFalse(proxy._engine_dead)
        self.assertEqual(len(proxy._native_rpc_uncertain), 1)

    def test_real_socket_shutdown_unblocks_cancelled_receiver_without_pool_reuse(self):
        proxy = self.proxy()
        client, server = socket.socketpair()
        channel = SimpleNamespace(sock=client, recv_buffer=bytearray())
        proxy._acquire_rpc_channel.return_value = channel
        proxy._open_rpc_channel.return_value = channel
        proxy._drop_rpc_channel = MethodType(SubprocessInferenceEngineProxy._drop_rpc_channel, proxy)
        # The production blocking receiver is used, with no responding backend.
        async def check():
            task = asyncio.create_task(proxy._rpc('ieee_gpu_reference', operation='acquire'))
            try:
                raw = await asyncio.wait_for(asyncio.to_thread(server.recv, 4096), 1.)
                self.assertEqual(json.loads(raw)['cmd'], 'ieee_gpu_reference')
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(await asyncio.wait_for(asyncio.to_thread(server.recv, 1), 1.), b'')
            finally:
                task.cancel()
                server.close()
                client.close()
        asyncio.run(check())
        proxy._release_rpc_channel.assert_not_awaited()
        self.assertFalse(proxy._engine_dead)
        self.assertEqual(len(proxy._native_rpc_uncertain), 1)


def replay_fixture():
    runner = ScenarioRunner.__new__(ScenarioRunner)
    runner.model_cfg = {'timing_contract': 'ieee_tc_native_v1'}
    runner._generation_contract = 'fixed_length_greedy_v1'
    runner._coordination_enabled = False
    runner.baseline_type = 'vllm'
    runner.name = 'failure-identity-test'
    runner._ttft_slo_ms = 1000.
    runner.wl_cfg = {'generation_contract': 'fixed_length_greedy_v1'}
    runner.engine = SimpleNamespace(backend='vllm')
    runner._stack = None
    runner._external_replay = None
    runner.traces = [SimpleNamespace(request_id=f'req-{i}', adapter_id=f'adapter-{i}',
                    is_burst=False, expected_output_tokens=4, prompt='existing fixture')
                    for i in range(2)]
    plans = {t.request_id: RequestExecutionPlan(t.prompt, 2, 4) for t in runner.traces}
    runner._prepare_request_execution_plan_cache = Mock(return_value=plans)
    runner._scheduled_offset = Mock(return_value=0.)
    runner._live_scale_eval_period_s = Mock(return_value=.1)
    for name in ('_assert_clean_gpu_environment', '_begin_instance_lifecycle_tracking',
                 '_observe_live_arrived_lora', '_observe_live_waiting_trace',
                 '_release_live_started_lora', '_release_live_waiting_trace',
                 '_release_live_arrived_lora', '_emit_live_snapshot',
                 '_attach_control_path_background_metrics'):
        setattr(runner, name, Mock())
    for name in ('_ensure_min_instances', '_await_trace_arrival',
                 '_acquire_dispatch_admission', '_release_dispatch_admission',
                 '_maybe_run_live_scale_control_evaluation',
                 '_wait_for_pending_scale_up_tasks', '_cancel_runtime_gpu_forward_tasks',
                 '_cleanup_extra_instances'):
        setattr(runner, name, AsyncMock())
    for name in ('_backlog_depth', '_active_request_count', '_busy_instance_ratio',
                 '_arrived_request_count'):
        setattr(runner, name, Mock(return_value=0))
    runner._waiting_visible_trace_queue = Mock(return_value=[])
    runner._coordinator_metric_views = Mock(return_value=[])
    runner._exec_request = AsyncMock(side_effect=RuntimeError('artifact unavailable'))
    return runner


class ReplayFailureIdentity(unittest.TestCase):
    def test_outer_exceptions_keep_offered_identity_and_missing_measurements(self):
        runner = replay_fixture()
        result, _ = asyncio.run(runner.run())
        self.assertEqual([r.request_id for r in result.requests], ['req-0', 'req-1'])
        self.assertEqual([r.adapter_id for r in result.requests], ['adapter-0', 'adapter-1'])
        self.assertEqual((result.total, result.completed, result.failed), (2, 0, 2))
        for row in result.requests:
            self.assertFalse(row.success)
            self.assertFalse(row.output_contract_match)
            self.assertIsNone(row.ttft_ms)
            self.assertIsNone(row.tpot_ms)
            self.assertIsNone(row.output_tokens)
            self.assertIsNone(row.cost_usd)
            self.assertEqual(row.requested_completion_tokens, 4)
            self.assertEqual(row.canonical_prompt_sha256,
                             hashlib.sha256(b'existing fixture').hexdigest())
            self.assertGreaterEqual(row.failure_observation['observed_offset_s'], 0.)
            self.assertFalse(row.failure_observation['native_completion_inferred'])
            self.assertIsNone(row.completed_offset_s)
            self.assertIsNone(json.loads(json.dumps(asdict(row)))['output_tokens'])

    def test_individual_task_cancellation_is_not_anonymous_or_whole_replay_abort(self):
        runner = replay_fixture()
        runner._exec_request.side_effect = asyncio.CancelledError('request cancelled')
        result, _ = asyncio.run(runner.run())
        self.assertEqual([r.request_id for r in result.requests], ['req-0', 'req-1'])
        self.assertTrue(all(not r.success for r in result.requests))
        self.assertTrue(all(r.failure_observation['exception_type'] == 'CancelledError'
                            for r in result.requests))

    def test_global_replay_cancellation_still_propagates(self):
        runner = replay_fixture()
        async def check():
            entered = asyncio.Event()
            async def never_complete(*args, **kwargs):
                entered.set()
                await asyncio.Future()
            runner._exec_request.side_effect = never_complete
            task = asyncio.create_task(runner.run())
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(check())
        runner._attach_control_path_background_metrics.assert_not_called()
        evidence=runner._interrupted_replay_evidence[-1]
        self.assertFalse(evidence['complete'])
        self.assertEqual(evidence['planned_request_count'],2)
        self.assertEqual(len(evidence['requests']),evidence['submitted_count'])
        self.assertTrue(all(not r['success'] for r in evidence['requests']))
        self.assertFalse(evidence['collection_errors'])

    def test_collector_rejects_unfinished_future_or_wrong_result_identity(self):
        runner = replay_fixture()
        async def check():
            trace = runner.traces[0]
            plan = RequestExecutionPlan(trace.prompt, 2, 4)
            future = asyncio.Future()
            with self.assertRaisesRegex(RuntimeError, 'unfinished'):
                runner._collect_request_task_result(future, trace, time.perf_counter(), plan)
            future.set_exception(RuntimeError('original'))
            row = runner._collect_request_task_result(future, trace, time.perf_counter() - .1, plan)
            for changed in (replace(row, request_id='another'), replace(row, adapter_id='other')):
                future = asyncio.Future()
                future.set_result(changed)
                with self.assertRaisesRegex(RuntimeError, 'identity'):
                    runner._collect_request_task_result(future, trace, time.perf_counter() - .1, plan)
        asyncio.run(check())

    def test_failure_cannot_be_manufactured_before_arrival_or_without_fixed_input(self):
        runner = replay_fixture()
        async def check():
            trace = runner.traces[0]
            future = asyncio.Future()
            future.set_exception(RuntimeError('original'))
            with self.assertRaisesRegex(RuntimeError, 'before its offered arrival'):
                runner._collect_request_task_result(future, trace, time.perf_counter() + 60., None)
            with self.assertRaisesRegex(RuntimeError, 'prepared input identity'):
                runner._collect_request_task_result(future, trace, time.perf_counter() - .1, None)
        asyncio.run(check())

    def test_external_failure_keeps_original_arrival_evidence(self):
        runner = replay_fixture()
        async def check():
            trace = runner.traces[0]
            record = {'request_id':trace.request_id, 'server_received_s': time.perf_counter()}
            runner._external_replay = SimpleNamespace(records={trace.request_id:record})
            future = asyncio.Future()
            future.set_exception(RuntimeError('original'))
            row = runner._collect_request_task_result(future, trace, time.perf_counter() - .1,
                                                       RequestExecutionPlan(trace.prompt, 2, 4))
            self.assertEqual(row.external_arrival_timing, record)
            self.assertIsNot(row.external_arrival_timing, record)
            self.assertEqual(row.arrival_contract, 'external_frozen_trace_v1')
        asyncio.run(check())

    def test_incomplete_publisher_does_not_manufacture_future_failure_rows(self):
        runner = replay_fixture()
        async def receive():
            yield 0, {'server_received_s':time.perf_counter()}
        runner._external_replay = SimpleNamespace(
            context={'replay_t0_s':time.perf_counter() - .1},
            plan=SimpleNamespace(entries=[0,1]), receive=receive,
            records={'req-0':{'request_id':'req-0'}})
        with self.assertRaisesRegex(RuntimeError, 'incomplete replay'):
            asyncio.run(runner.run())
        self.assertEqual(runner._exec_request.await_count, 1)
        evidence=runner._interrupted_replay_evidence[-1]
        self.assertEqual(evidence['planned_request_count'],2)
        self.assertEqual(evidence['submitted_count'],1)
        self.assertEqual([r['request_id'] for r in evidence['requests']],['req-0'])
        self.assertEqual(evidence['unsubmitted_request_ids'],['req-1'])
        self.assertFalse(evidence['complete'])

    def test_control_failure_preserves_previously_observed_rows_and_original_error(self):
        runner=replay_fixture()
        runner._maybe_run_live_scale_control_evaluation.side_effect=RuntimeError('control failure')
        with self.assertRaisesRegex(RuntimeError,'control failure'):
            asyncio.run(runner.run())
        evidence=runner._interrupted_replay_evidence[-1]
        self.assertEqual(evidence['error_type'],'RuntimeError')
        self.assertEqual(evidence['submitted_count'],2)
        self.assertEqual([r['request_id'] for r in evidence['requests']],['req-0','req-1'])
        self.assertFalse(evidence['collection_errors'])
        runner._attach_control_path_background_metrics.assert_not_called()

    def test_duplicate_or_empty_input_identity_is_not_silently_skipped(self):
        runner = replay_fixture()
        runner._prepare_request_execution_plan = Mock(return_value=RequestExecutionPlan('p', 2, 4))
        for traces in ([runner.traces[0], runner.traces[0]],
                       [SimpleNamespace(request_id='')]):
            with self.assertRaisesRegex(ValueError, 'unique request IDs'):
                ScenarioRunner._prepare_request_execution_plan_cache(runner, runner.engine, traces, 4)


if __name__ == '__main__':
    unittest.main()
