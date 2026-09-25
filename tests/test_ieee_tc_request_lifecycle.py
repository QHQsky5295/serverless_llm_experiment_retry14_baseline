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
from unittest.mock import AsyncMock, Mock

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
        manager._registered_adapters[aid] = NativeAdapter()
        manager.activate(aid)
    owner = IEEEBackendGPUReferences(manager, Mock(), demand_loader=load)
    async def rpc(*, operation, **kwargs):
        return {**getattr(owner, operation)(**kwargs), 'clock_id': local_monotonic_clock_id()}
    slot.engine.ieee_gpu_reference = AsyncMock(side_effect=rpc)
    return runner, slot, trace, plan, owner, rpc


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


class ControllerNativeReferenceLifecycle(unittest.TestCase):
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
        from tests.test_http_artifact_store import archive_bytes
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
            client._opener = Mock()
            client._opener.open.side_effect = lambda *a, **k: io.BytesIO(archive_bytes())
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

    def test_actual_http_client_cancel_waits_for_reader_and_cleans_without_publication(self):
        from faaslora.memory.residency_manager import ResidencyManager
        from faaslora.storage.http_artifact_store import HttpArtifactStoreClient
        from tests.test_http_artifact_store import archive_bytes
        import io
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'a'
            source.mkdir()
            (source / 'old').write_bytes(b'previous-copy')
            manager = ResidencyManager({'memory': {'nvme': {'cache_dir': directory}}}, Mock(), Mock())
            runner._stack = SimpleNamespace(residency_manager=manager)
            client = HttpArtifactStoreClient(endpoint='http://127.0.0.1:1')
            entered, proceed = threading.Event(), threading.Event()
            class HeldResponse(io.BytesIO):
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
            slot.engine.ieee_gpu_reference.side_effect = RuntimeError('snapshot unavailable')
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
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertIn('clock identity', result.error)
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


class NativeRPCOwnership(unittest.TestCase):
    def proxy(self):
        proxy = SubprocessInferenceEngineProxy.__new__(SubprocessInferenceEngineProxy)
        proxy.model_cfg = {'timing_contract': 'ieee_tc_native_v1'}
        proxy._engine_dead = False
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
            asyncio.run(proxy._rpc('generate_prepared', request_plan={'prompt': 'existing'}))
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
        self.assertTrue(proxy._engine_dead)

    def test_real_socket_shutdown_unblocks_cancelled_receiver_without_pool_reuse(self):
        proxy = self.proxy()
        client, server = socket.socketpair()
        channel = SimpleNamespace(sock=client, recv_buffer=bytearray())
        proxy._acquire_rpc_channel.return_value = channel
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
        self.assertTrue(proxy._engine_dead)


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

    def test_duplicate_or_empty_input_identity_is_not_silently_skipped(self):
        runner = replay_fixture()
        runner._prepare_request_execution_plan = Mock(return_value=RequestExecutionPlan('p', 2, 4))
        for traces in ([runner.traces[0], runner.traces[0]],
                       [SimpleNamespace(request_id='')]):
            with self.assertRaisesRegex(ValueError, 'unique request IDs'):
                ScenarioRunner._prepare_request_execution_plan_cache(runner, runner.engine, traces, 4)


if __name__ == '__main__':
    unittest.main()
