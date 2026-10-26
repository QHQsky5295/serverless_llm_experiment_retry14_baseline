"""D352: transaction-local storage evidence, always fresh allocator occupancy.

CPU contract tests; no claim about CUDA correctness or Full replay performance.
"""
import asyncio
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from faaslora.memory import gpu_monitor as monitor
from tests import test_ieee_tc_gpu_references as references
from tests import test_ieee_tc_worker_observation as observations


def stats(allocated=256, active=128):
    return {'allocated_bytes.current': allocated, 'active_bytes.current': active,
            'allocations.current': 4, 'active_requests.current': 2}


class HitHostTransaction(unittest.TestCase):
    def setUp(self):
        self.case = references.NativeDemandTransactions()
        self.case.setUp()
        self.owner = self.case.owner
        self.rows = [dict(pinned=False, allocated_bytes=64, adapter_ids=[4])]
        self.observe = Mock(return_value=stats())
        self.addCleanup(patch.stopall)
        patch.object(monitor, 'torch', SimpleNamespace(cuda=SimpleNamespace(
            host_memory_stats=self.observe))).start()

        def result(observation, tensor_budget_bytes):
            return dict(admitted=observation['accounted_tensor_bytes'] <= tensor_budget_bytes,
                        before=observation, tensor_budget_bytes=tensor_budget_bytes)

        def full(*, tensor_budget_bytes, **kwargs):
            return result(monitor._ieee_pinned_host_observation(dict(
                host_allocations=self.rows, host_staged_adapter_ids=[])), tensor_budget_bytes)

        def recheck(*, before_check, tensor_budget_bytes):
            return result(monitor._ieee_reobserve_pinned_host_allocator(before_check['before']),
                          tensor_budget_bytes)

        self.full, self.recheck = Mock(side_effect=full), Mock(side_effect=recheck)
        self.owner.host_allocation_check = self.full
        self.owner.host_allocation_recheck = self.recheck
        self.owner.configure_host_budget(expected_owner_id=self.owner.owner_id,
                                         tensor_budget_bytes=1024)
        self.reset_counts()

    def reset_counts(self):
        self.full.reset_mock()
        self.recheck.reset_mock()
        self.observe.reset_mock()
        self.case.fence.reset_mock()

    def warm(self):
        self.case.demand(lease='initial')
        self.case.release('initial')
        self.reset_counts()

    def test_confirmed_hit_one_graph_two_allocator_reads_and_fence(self):
        self.warm()
        old_loads = len(self.case.loads)
        result = self.case.demand(lease='hit')
        self.assertTrue(result['acquired'])
        self.assertTrue(result['gpu_confirmed_before_acquisition'])
        self.assertEqual(len(self.case.loads), old_loads)
        self.full.assert_called_once()
        self.recheck.assert_called_once()
        self.assertEqual(self.observe.call_count, 2)
        self.case.fence.assert_called_once()
        self.assertEqual(result['host_allocation']['after_storage_observation'],
                         'same_transaction_confirmed_gpu_hit_v1')
        self.assertEqual(result['host_allocation']['before'], result['host_allocation']['after'])
        self.assertIn(4, self.case.manager._registered_adapters.pinned_items)
        self.assertIn(4, self.case.manager._active_adapters.pinned_items)
        self.assertEqual(self.case.demand(lease='hit'), result)  # transport retry
        self.assertEqual(self.observe.call_count, 2)
        self.case.release('hit')
        self.assertEqual(self.owner.snapshot()['live_leases'], 0)

    def test_cold_host_activation_and_unconfirmed_hit_keep_full_walks(self):
        for mode in ('cold', 'host', 'unconfirmed'):
            with self.subTest(mode=mode):
                if mode != 'cold':
                    self.case.release('cold-1')
                    if mode == 'host':
                        self.case.manager.deactivate(4)
                    else:
                        self.owner._gpu_confirmations.pop(4)
                    self.reset_counts()
                result = self.case.demand(lease='cold-1' if mode == 'cold' else mode)
                self.assertTrue(result['acquired'])
                self.assertEqual(self.full.call_count, 2)
                self.assertEqual(self.observe.call_count, 2)
                self.recheck.assert_not_called()
                self.assertEqual(result['host_allocation']['after_storage_observation'],
                                 'fresh_inventory')
                if mode == 'host':
                    self.case.release('host')

    def test_no_qualified_recheck_keeps_original_double_walk(self):
        self.warm()
        self.owner.host_allocation_recheck = None
        self.case.demand(lease='hit')
        self.assertEqual(self.full.call_count, 2)
        self.recheck.assert_not_called()

    def test_allocator_growth_after_fence_is_rejected_and_poisons(self):
        self.warm()
        self.observe.side_effect = [stats(), stats(2048)]
        with self.assertRaisesRegex(RuntimeError, 'exceeded its HOST'):
            self.case.demand(lease='hit')
        self.assertEqual(self.observe.call_count, 2)
        with self.assertRaisesRegex(RuntimeError, 'invalidated'):
            self.owner.snapshot()

    def test_precheck_failure_does_not_fence_or_acquire(self):
        self.warm()
        self.observe.return_value = stats(2048)
        result = self.case.demand(lease='hit')
        self.assertFalse(result['acquired'])
        self.assertEqual(result['reason'], 'native_host_tensor_budget')
        self.case.fence.assert_not_called()
        self.recheck.assert_not_called()
        self.assertEqual(self.owner.snapshot()['live_leases'], 0)

    def test_fence_failure_cannot_publish_reused_evidence(self):
        self.warm()
        self.case.fence.side_effect = RuntimeError('CUDA fence fixture failure')
        with self.assertRaisesRegex(RuntimeError, 'CUDA fence'):
            self.case.demand(lease='hit')
        self.recheck.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'invalidated'):
            self.owner.snapshot()

    def test_every_new_operation_remeasures_storage_even_same_copy(self):
        self.warm()
        first = self.case.demand(lease='first')
        self.case.release('first')
        self.rows[0]['allocated_bytes'] = 512
        second = self.case.demand(lease='second')
        self.assertEqual(first['host_allocation']['after']['before']['accounted_tensor_bytes'], 320)
        self.assertEqual(second['host_allocation']['after']['before']['accounted_tensor_bytes'], 768)
        self.assertEqual(self.full.call_count, 2)
        self.assertEqual(self.recheck.call_count, 2)

    def protected_host(self, hold='host-copy'):
        self.warm()
        self.case.manager.deactivate(4)
        self.owner.snapshot()
        source_id = self.owner._source_incarnations[4]
        held = self.owner.hold_host_source(lease_id=hold, adapter_int_id=4,
            lora_name='adapter-4', lora_path='/existing/adapter-4',
            expected_owner_id=self.owner.owner_id, expected_epoch=self.owner.epoch,
            expected_source_id=source_id)
        self.assertTrue(held['held'])
        self.witness = Mock(return_value={'kind': 'test-transaction', 'source': source_id})
        self.owner.host_activation_storage_witness = self.witness
        self.reset_counts()
        return dict(required_source_tier='host', expected_source_id=source_id)

    def test_protected_host_activation_one_walk_two_fresh_allocator_reads(self):
        args = self.protected_host()
        loads = len(self.case.loads)
        result = self.case.demand(lease='host-activate', **args)
        self.assertTrue(result['acquired'])
        self.assertTrue(result['native_load_invoked'])
        self.assertFalse(result['gpu_confirmed_before_acquisition'])
        self.assertEqual(len(self.case.loads), loads + 1)
        self.assertEqual(self.full.call_count, 1)
        self.assertEqual(self.recheck.call_count, 1)
        self.assertEqual(self.observe.call_count, 2)
        self.assertEqual(self.witness.call_count, 2)
        self.assertEqual(self.witness.call_args.kwargs['previous'], self.witness.return_value)
        self.assertEqual(result['host_allocation']['after_storage_observation'],
                         'same_transaction_registered_host_activation_v1')
        self.assertEqual(self.case.demand(lease='host-activate', **args), result)
        self.assertEqual(self.witness.call_count, 2)  # no work on transport replay
        self.owner.release_host_source(lease_id='host-copy', expected_owner_id=self.owner.owner_id)
        self.assertIn(4, self.case.manager._registered_adapters.pinned_items)
        self.case.release('host-activate')
        self.assertNotIn(4, self.case.manager._registered_adapters.pinned_items)

    def test_host_without_live_copy_protection_keeps_full_walks(self):
        args = self.protected_host()
        self.owner.release_host_source(lease_id='host-copy', expected_owner_id=self.owner.owner_id)
        result = self.case.demand(lease='unheld', **args)
        self.assertTrue(result['acquired'])
        self.witness.assert_not_called()
        self.assertEqual(self.full.call_count, 2)
        self.recheck.assert_not_called()

    def test_host_lease_transfer_fences_before_release_and_retains_gpu_pins(self):
        args = self.protected_host()
        def fence():
            self.assertIn('host-copy', self.owner._host_leases)
        self.case.fence.side_effect = fence
        result = self.case.demand(lease='transfer', release_host_lease_id='host-copy', **args)
        ack = result['host_lease_transfer']
        self.assertEqual(ack['kind'], 'same_owner_host_to_gpu_v1')
        self.assertEqual(ack['gpu_lease_id'], 'transfer')
        self.assertEqual(ack['host_source_id'], args['expected_source_id'])
        self.assertLess(ack['acquired_epoch'], ack['epoch'])
        self.assertEqual(result['epoch'], ack['epoch'])
        self.assertEqual(result['post_load_source_snapshot']['epoch'], ack['epoch'])
        self.assertGreaterEqual(ack['released_monotonic_s'], result['acquired_monotonic_s'])
        self.assertEqual(self.owner.snapshot()['live_host_source_leases'], 0)
        self.assertEqual(self.owner.snapshot()['live_leases'], 1)
        for cache in self.owner._caches():
            self.assertIn(4, cache.pinned_items)
        self.case.fence.side_effect = None
        self.case.release('transfer')
        for cache in self.owner._caches():
            self.assertNotIn(4, cache.pinned_items)

    def test_host_lease_transfer_replay_does_not_load_fence_or_release_twice(self):
        args = self.protected_host() | {'release_host_lease_id': 'host-copy'}
        result = self.case.demand(lease='transfer', **args)
        loads, fences, epoch = len(self.case.loads), self.case.fence.call_count, self.owner.epoch
        self.assertEqual(self.case.demand(lease='transfer', **args), result)
        self.assertEqual((len(self.case.loads), self.case.fence.call_count, self.owner.epoch),
                         (loads, fences, epoch))
        for changed in (None, 'another-lease'):
            with self.assertRaisesRegex(ValueError, 'different demand load'):
                self.case.demand(lease='transfer', **(args | {'release_host_lease_id': changed}))

    def test_host_lease_transfer_retires_only_its_own_source_reader(self):
        args = self.protected_host()
        other = self.owner.hold_host_source(lease_id='other-reader', adapter_int_id=4,
            lora_name='adapter-4', lora_path='/existing/adapter-4',
            expected_owner_id=self.owner.owner_id, expected_epoch=self.owner.epoch,
            expected_source_id=args['expected_source_id'])
        self.assertTrue(other['held'])
        self.case.demand(lease='transfer', release_host_lease_id='host-copy', **args)
        self.case.release('transfer')
        self.assertIn(4, self.case.manager._registered_adapters.pinned_items)
        self.assertEqual(self.owner.snapshot()['live_host_source_leases'], 1)
        self.owner.release_host_source(lease_id='other-reader', expected_owner_id=self.owner.owner_id)
        self.assertNotIn(4, self.case.manager._registered_adapters.pinned_items)

    def test_host_lease_transfer_failure_preserves_source_until_known_outcome(self):
        for failure in ('precheck', 'fence'):
            with self.subTest(failure=failure):
                # Fresh fixture; poison is intentionally irreversible.
                other = HitHostTransaction()
                other.setUp()
                try:
                    args = other.protected_host() | {'release_host_lease_id': 'host-copy'}
                    if failure == 'precheck':
                        other.observe.return_value = stats(2048)
                        result = other.case.demand(lease='transfer', **args)
                        self.assertFalse(result['acquired'])
                    else:
                        other.case.fence.side_effect = RuntimeError('copy failed')
                        with self.assertRaisesRegex(RuntimeError, 'copy failed'):
                            other.case.demand(lease='transfer', **args)
                    self.assertIn('host-copy', other.owner._host_leases)
                    self.assertNotIn('host-copy', other.owner._host_released)
                finally:
                    other.doCleanups()

    def test_host_lease_transfer_rejects_wrong_source_before_loading(self):
        args = self.protected_host()
        loads = len(self.case.loads)
        for lease in ('unknown', '', 7):
            with self.subTest(lease=lease), self.assertRaises(ValueError):
                self.case.demand(lease='transfer', release_host_lease_id=lease, **args)
        self.assertEqual(len(self.case.loads), loads)
        self.assertIn('host-copy', self.owner._host_leases)

    def test_host_lease_transfer_can_follow_same_copy_gpu_promotion(self):
        args = self.protected_host()
        self.case.demand(lease='concurrent', **args)
        self.case.release('concurrent')
        stale = self.case.demand(lease='transfer', release_host_lease_id='host-copy', **args)
        self.assertFalse(stale['acquired'])
        self.assertIn('host-copy', self.owner._host_leases)
        result = self.case.demand(lease='transfer', required_source_tier='gpu',
            expected_source_id=self.owner._gpu_source_incarnations[4],
            release_host_lease_id='host-copy')
        self.assertTrue(result['acquired'])
        self.assertFalse(result['native_load_invoked'])
        self.assertEqual(result['host_lease_transfer']['host_source_id'], args['expected_source_id'])

    def test_worker_capacity_witness_transfers_and_replays_complete_ack(self):
        from faaslora.scheduling.resource_coordinator import (
            CompletedLengthSnapshot, NativeIterationObservation, NativeTransferObservation)
        from tests import test_ieee_tc_scheduler_observation as fixture
        args = self.protected_host()
        worker = monitor.IEEEWorkerObservationExtension()
        worker.device, worker.rank = SimpleNamespace(type='cuda'), 0
        worker.model_runner = SimpleNamespace(lora_manager=SimpleNamespace(
            _adapter_manager=self.case.manager))
        worker._ieee_gpu_reference_owner = self.owner
        steps = NativeIterationObservation()
        steps.scheduled(fixture.iteration(r=32))
        scheduler = fixture.scheduler()
        scheduler._ieee_transfers = NativeTransferObservation(steps, 2)
        scheduler._ieee_transfers.event(operation='attach_domain', transfer_id='files')
        observation = fixture.observe(scheduler, steps)
        lengths = CompletedLengthSnapshot('model/backend', 'profile',
            observation['captured_at'], {0: 64., 1: 128., 2: 256.})
        pool = dict(slot_adapter_ids=list(self.case.manager.lora_index_to_id),
            pool_allocated_bytes=200, slot_capacity_bytes=100, occupied_slot_capacity_bytes=100)
        monitor.torch.cuda.mem_get_info = lambda _: (100, 1000)
        command = dict(operation='admit_and_demand_load', lease_id='transfer', adapter_int_id=4,
            lora_name='adapter-4', lora_path='/existing/adapter-4',
            expected_owner_id=self.owner.owner_id, expected_epoch=self.owner.epoch,
            capacity_only=True, scheduler_observation=observation, lengths=lengths,
            release_host_lease_id='host-copy', **args)
        with patch.object(monitor, '_ieee_lora_pool_inventory', return_value=pool):
            result = worker.ieee_gpu_reference(**command)
            replay = worker.ieee_gpu_reference(**command)
        self.assertTrue(result['acquired'])
        self.assertEqual(result['admission']['backend_receipt_epoch'], result['epoch'])
        self.assertEqual(result['admission']['post_observation_owner_epoch'], result['epoch'])
        self.assertEqual(result['host_lease_transfer']['epoch'], result['epoch'])
        self.assertEqual(replay, result)
        self.assertEqual(self.owner.snapshot()['live_host_source_leases'], 0)

    def test_host_without_source_identity_keeps_full_walks(self):
        self.protected_host()
        result = self.case.demand(lease='unqualified', required_source_tier='host')
        self.assertTrue(result['acquired'])
        self.witness.assert_not_called()
        self.assertEqual(self.full.call_count, 2)

    def test_host_without_witness_callback_keeps_full_walks(self):
        args = self.protected_host()
        self.owner.host_activation_storage_witness = None
        self.case.demand(lease='unqualified', **args)
        self.assertEqual(self.full.call_count, 2)
        self.recheck.assert_not_called()

    def test_host_pre_witness_failure_precedes_mutation(self):
        args = self.protected_host()
        loads = len(self.case.loads)
        self.witness.side_effect = ValueError('unaudited native path')
        with self.assertRaisesRegex(ValueError, 'unaudited'):
            self.case.demand(lease='bad', **args)
        self.assertEqual(len(self.case.loads), loads)
        self.case.fence.assert_not_called()
        self.assertEqual(self.owner.snapshot()['live_leases'], 0)

    def test_host_post_witness_failure_invalidates_owner(self):
        args = self.protected_host()
        self.witness.side_effect = [self.witness.return_value, {'kind': 'changed'}]
        with self.assertRaisesRegex(RuntimeError, 'changed its storage witness'):
            self.case.demand(lease='bad', **args)
        self.recheck.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'invalidated'):
            self.owner.snapshot()

    def test_host_fresh_post_allocator_can_reject(self):
        args = self.protected_host()
        self.observe.side_effect = [stats(), stats(2048)]
        with self.assertRaisesRegex(RuntimeError, 'exceeded its HOST'):
            self.case.demand(lease='bad', **args)
        self.assertEqual(self.observe.call_count, 2)
        with self.assertRaisesRegex(RuntimeError, 'invalidated'):
            self.owner.snapshot()

    def test_host_storage_measured_again_on_next_activation(self):
        args = self.protected_host()
        first = self.case.demand(lease='first-host', **args)
        self.case.release('first-host')
        self.case.manager.deactivate(4)
        self.rows[0]['allocated_bytes'] = 512
        second = self.case.demand(lease='second-host', **args)
        self.assertEqual(first['host_allocation']['after']['before']['accounted_tensor_bytes'], 320)
        self.assertEqual(second['host_allocation']['after']['before']['accounted_tensor_bytes'], 768)
        self.assertEqual(self.full.call_count, 2)
        self.assertEqual(self.recheck.call_count, 2)
        self.assertEqual(self.witness.call_count, 4)


class ProactiveHostTransaction(unittest.TestCase):
    """D675: private proactive transaction, not an unprotected demand hit."""
    def setUp(self):
        self.accounting = HitHostTransaction()
        self.accounting.setUp()
        self.addCleanup(self.accounting.doCleanups)
        self.accounting.warm()
        self.case = self.accounting.case
        self.owner = self.case.owner
        self.case.manager.deactivate(4)
        self.owner.snapshot()
        self.owner.preparation_loader = self.case.loader
        self.witness = Mock(return_value={'kind': 'test-proactive-transaction'})
        self.owner.proactive_host_activation_storage_witness = self.witness
        self.accounting.reset_counts()

    def prepare(self, lease='prepare-4', decide=None):
        before = self.owner.snapshot()
        return self.owner.proactive_host_prepare_and_acquire(
            lease_id=lease, adapter_int_id=4, lora_name='adapter-4',
            lora_path='/existing/adapter-4', expected_owner_id=before['owner_id'],
            expected_epoch=before['epoch'],
            capacity_only=False, decide=decide or (lambda *args: dict(admit=True, reason='test-admit')))

    def test_proactive_uses_fresh_precheck_then_rechecks_allocator_and_witness(self):
        self.assertFalse(self.owner._host_references)
        loads = len(self.case.loads)
        result = self.prepare()
        self.assertTrue(result['acquired'])
        self.assertEqual(result['host_allocation']['after_storage_observation'],
                         'same_transaction_proactive_host_activation_v1')
        self.accounting.full.assert_called_once()
        self.accounting.recheck.assert_called_once()
        self.assertEqual(self.accounting.observe.call_count, 2)
        self.assertEqual(self.witness.call_count, 2)
        self.assertIs(self.witness.call_args.kwargs['loader'], self.case.loader)
        self.assertEqual(result['expected_source_id'], self.owner._source_incarnations[4])
        self.assertEqual(len(self.case.loads), loads + 1)
        self.assertEqual(self.prepare(), result)  # no work on transport retry
        self.assertEqual(self.witness.call_count, 2)
        self.assertIn(4, self.case.manager._active_adapters.pinned_items)
        self.assertIn(4, self.case.manager._registered_adapters.pinned_items)
        self.case.release('prepare-4')
        self.assertEqual(self.owner.snapshot()['live_leases'], 0)

    def test_demand_cannot_use_proactive_callback_even_with_same_loader(self):
        result = self.case.demand(lease='ordinary', required_source_tier='host',
            expected_source_id=self.owner._source_incarnations[4])
        self.assertTrue(result['acquired'])
        self.witness.assert_not_called()
        self.accounting.recheck.assert_not_called()
        self.assertEqual(self.accounting.full.call_count, 2)

    def test_no_qualified_proactive_callback_keeps_full_walks(self):
        self.owner.proactive_host_activation_storage_witness = None
        result = self.prepare()
        self.assertEqual(result['host_allocation']['after_storage_observation'], 'fresh_inventory')
        self.accounting.recheck.assert_not_called()
        self.assertEqual(self.accounting.full.call_count, 2)

    def test_no_recheck_keeps_full_walks(self):
        self.owner.host_allocation_recheck = None
        self.prepare()
        self.witness.assert_not_called()
        self.assertEqual(self.accounting.full.call_count, 2)

    def test_precheck_is_after_policy_not_cost_provider_inventory(self):
        def decide(*args):
            self.accounting.rows[0]['allocated_bytes'] = 512
            return dict(admit=True, reason='test-admit')
        result = self.prepare(decide=decide)
        self.assertEqual(result['host_allocation']['before']['before']['accounted_tensor_bytes'], 768)
        self.assertEqual(result['host_allocation']['after']['before']['accounted_tensor_bytes'], 768)

    def test_each_new_activation_takes_new_graph(self):
        self.prepare()
        self.case.release('prepare-4')
        self.case.manager.deactivate(4)
        self.accounting.rows[0]['allocated_bytes'] = 512
        result = self.prepare(lease='prepare-again')
        self.assertEqual(result['host_allocation']['after']['before']['accounted_tensor_bytes'], 768)
        self.assertEqual(self.accounting.full.call_count, 2)
        self.assertEqual(self.accounting.recheck.call_count, 2)

    def test_pre_witness_failure_precedes_copy(self):
        loads = len(self.case.loads)
        self.witness.side_effect = RuntimeError('unqualified copy')
        with self.assertRaisesRegex(RuntimeError, 'unqualified copy'):
            self.prepare()
        self.assertEqual(len(self.case.loads), loads)
        self.accounting.recheck.assert_not_called()

    def test_post_witness_change_invalidates_owner(self):
        self.witness.side_effect = [self.witness.return_value, {'kind': 'changed'}]
        with self.assertRaisesRegex(RuntimeError, 'changed its storage witness'):
            self.prepare()
        self.accounting.recheck.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'invalidated'):
            self.owner.snapshot()

    def test_post_allocator_growth_still_rejects(self):
        self.accounting.observe.side_effect = [stats(), stats(2048)]
        with self.assertRaisesRegex(RuntimeError, 'exceeded its HOST'):
            self.prepare()
        self.assertEqual(self.accounting.observe.call_count, 2)
        with self.assertRaisesRegex(RuntimeError, 'invalidated'):
            self.owner.snapshot()

    def test_copy_failure_invalidates_without_reusing_evidence(self):
        self.owner.preparation_loader = Mock(side_effect=RuntimeError('copy failed'))
        with self.assertRaisesRegex(RuntimeError, 'copy failed'):
            self.prepare()
        self.accounting.recheck.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'invalidated'):
            self.owner.snapshot()


class RegisteredProactiveHostTransaction(unittest.TestCase):
    def test_call_local_precheck_runs_after_gpu_only_removal(self):
        case = references.NativeObjectiveReplacement()
        case.setUp()
        self.addCleanup(case.doCleanups)
        owner = case.owner
        owner.host_allocation_check = Mock(return_value={'admitted': True})
        owner.host_allocation_recheck = Mock(return_value={'admitted': True})
        owner.proactive_host_activation_storage_witness = Mock(return_value={'kind': 'fixture'})
        owner.configure_host_budget(expected_owner_id=owner.owner_id, tensor_budget_bytes=1024)
        owner.host_allocation_check.reset_mock()
        registered = dict(case.manager._registered_adapters.cache)
        def precheck(**kwargs):
            self.assertEqual(dict(case.manager._registered_adapters.cache), registered)
            self.assertNotIn(3, case.manager._active_adapters)
            return dict(admitted=True, before={'fixture': 1},
                        before_storage_observation='same_transaction_registered_preparation_v1')
        check = Mock(side_effect=precheck)
        result = case.prepare(case.epoch(), _preparation_host_precheck=check)
        self.assertTrue(result['acquired'])
        check.assert_called_once_with(tensor_budget_bytes=1024)
        owner.host_allocation_check.assert_not_called()
        owner.host_allocation_recheck.assert_called_once()
        self.assertEqual(result['host_allocation']['before']['before_storage_observation'],
                         'same_transaction_registered_preparation_v1')

    def test_registered_objective_replacement_keeps_exact_victim_and_fallback(self):
        case = references.NativeObjectiveReplacement()
        case.setUp()
        self.addCleanup(case.doCleanups)
        owner = case.owner
        owner.host_allocation_check = Mock(return_value={'admitted': True, 'before': {'fixture': 1}})
        owner.host_allocation_recheck = Mock(return_value={'admitted': True})
        owner.proactive_host_activation_storage_witness = Mock(return_value={'kind': 'fixture'})
        owner.configure_host_budget(expected_owner_id=owner.owner_id, tensor_budget_bytes=1024)
        epoch = case.epoch()
        owner.register_preparation_plan(plan_id='registered', objective=epoch,
            target_adapter_ids=[4], expected_owner_id=owner.owner_id)
        fallback = case.manager._registered_adapters.cache[3]
        result = case.prepare(epoch, preparation_plan_id='registered')
        self.assertTrue(result['acquired'])
        self.assertEqual(result['replacement']['victim_adapter_ids'], [3])
        self.assertIs(case.manager._registered_adapters.cache[3], fallback)
        self.assertEqual(result['host_allocation']['after_storage_observation'],
                         'same_transaction_proactive_host_activation_v1')
        owner.release(lease_id='objective-prepare', expected_owner_id=owner.owner_id)
        owner.close_preparation_plan(plan_id='registered', expected_owner_id=owner.owner_id)

    def test_staged_registration_precedes_fresh_graph_and_readonly_activation(self):
        from tests import test_ieee_tc_transfer_pressure as pressure
        # Reuse the fixture's serialized inline planner, not its optional real
        # process startup. This test concerns storage/reference ownership;
        # process placement and startup are separate existing qualifications.
        pressure.setUpModule()
        self.addCleanup(pressure.tearDownModule)
        case = pressure.IntegratedNativeHostReplacement()
        self.addCleanup(case.doCleanups)
        factory, (fixture, runner, queue, slot, owner, snapshot, loads) = case.make()
        def graph():
            return {'registered': tuple(sorted(owner.manager._registered_adapters)),
                    'staged': tuple(sorted(owner.staged_models()))}
        def check(**kwargs):
            return {'admitted': True, 'before': graph()}
        def recheck(*, before_check, **kwargs):
            self.assertEqual(before_check['before'], graph())
            return {'admitted': True}
        def witness(*, adapter_int_id, loader, previous=None):
            self.assertIs(loader, owner.preparation_loader)
            self.assertIn(adapter_int_id, owner.manager._registered_adapters)
            self.assertNotIn(adapter_int_id, owner.staged_models())
            current = dict(graph(), source_id=owner._source_incarnations[adapter_int_id])
            if previous is not None:
                self.assertEqual(previous, current)
            return current
        owner.host_allocation_check = Mock(side_effect=check)
        owner.host_allocation_recheck = Mock(side_effect=recheck)
        owner.proactive_host_activation_storage_witness = Mock(side_effect=witness)
        async def run():
            try:
                await asyncio.wait_for(factory.execute(runner, slot, mode='residency'), 5)
                receipts = [a['receipt'] for p in runner._ieee_gpu_preparation_plans for a in p['attempts']]
                staged = [r for r in receipts if r['transaction_scope'] == 'budgeted_staging_joint_cpu_gpu_commit']
                self.assertTrue(staged)
                self.assertTrue(any(r['replacement']['host_victim_adapter_ids'] for r in staged))
                for row in staged:
                    self.assertEqual(row['host_allocation']['after_storage_observation'],
                                     'same_transaction_proactive_host_activation_v1')
                self.assertGreater(owner.host_allocation_recheck.call_count, 0)
                factory.check_clean(fixture, runner, owner)
            finally:
                await queue.close()
        asyncio.run(run())


class NativeHostActivationWitness(unittest.TestCase):
    """Native method binding and transaction identity, no CUDA performance claim."""
    def setUp(self):
        from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
        from vllm.lora.model_manager import LRUCacheLoRAModelManager
        self.manager = object.__new__(LRUCacheLoRAModelManager)
        self.manager._registered_adapters = SimpleNamespace(cache={4: object()})
        self.manager.modules = {'layer': object()}
        self.manager.moe_ep_load_spec = None
        self.loader = object.__new__(LRUCacheWorkerLoRAManager)
        self.loader._adapter_manager = self.manager
        self.addCleanup(patch.stopall)
        patch('vllm.__version__', '0.30.0').start()
        self.copy = patch.object(monitor, '_ieee_host_copy_contract', return_value=[]).start()

    def witness(self, previous=None, staged=None):
        return monitor._ieee_host_activation_storage_witness(
            self.loader, self.manager, 4, {} if staged is None else staged, previous=previous)

    def test_native_path_qualified_once_and_membership_rechecked(self):
        before = self.witness()
        self.assertEqual(self.witness(previous=before), before)
        self.copy.assert_called_once_with(self.manager, 4)
        self.manager._registered_adapters.cache[4] = object()
        with self.assertRaisesRegex(RuntimeError, 'membership changed'):
            self.witness(previous=before)

    def test_staged_source_and_overridden_path_rejected(self):
        with self.assertRaisesRegex(ValueError, 'non-staged'):
            self.witness(staged={4: object()})
        self.loader.add_adapter = Mock()
        with self.assertRaisesRegex(RuntimeError, 'overridden'):
            self.witness()
        self.copy.assert_not_called()

    def test_native_copy_contract_failure_not_hidden(self):
        self.copy.side_effect = ValueError('source needs conversion')
        with self.assertRaisesRegex(ValueError, 'needs conversion'):
            self.witness()


class NativeRegisteredPreparationWitness(unittest.TestCase):
    """Actual native removal callbacks with tiny CPU objects; no GPU timings."""
    def setUp(self):
        from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
        from vllm.lora.model_manager import AdapterLRUCache, LRUCacheLoRAModelManager
        from faaslora.memory.residency_manager import IEEEBackendGPUReferences
        self.manager = object.__new__(LRUCacheLoRAModelManager)
        self.manager.lora_config = SimpleNamespace(max_loras=2, max_cpu_loras=4)
        self.manager.modules, self.manager.moe_ep_load_spec = {}, None
        self.manager.lora_index_to_id = [1, None]
        self.manager._registered_adapters = AdapterLRUCache(4, self.manager.deactivate_adapter)
        self.manager._active_adapters = AdapterLRUCache(2, self.manager._deactivate_adapter)
        self.manager._registered_adapters[1] = SimpleNamespace(id=1)
        self.manager._registered_adapters[2] = SimpleNamespace(id=2)
        self.manager._active_adapters[1] = None
        self.loader = object.__new__(LRUCacheWorkerLoRAManager)
        self.loader._adapter_manager = self.manager
        self.owner = IEEEBackendGPUReferences(self.manager, lambda: None)
        self.addCleanup(patch.stopall)
        patch('vllm.__version__', '0.30.0').start()

    def witness(self, previous=None):
        return monitor._ieee_registered_preparation_storage_witness(
            self.loader, self.manager, self.owner, 2, previous=previous)

    def test_native_gpu_removal_preserves_host_graph_boundary(self):
        before = self.witness()
        cpu = dict(self.manager._registered_adapters.cache)
        self.manager._active_adapters.pop(1)
        self.owner._refresh()
        self.assertEqual(self.manager.lora_index_to_id, [None, None])
        self.assertEqual(dict(self.manager._registered_adapters.cache), cpu)
        self.assertEqual(self.witness(previous=before), before)

    def test_removal_wrapper_and_captured_original_both_qualified(self):
        gpu = self.manager._active_adapters
        saved = gpu._on_remove
        gpu._on_remove = Mock(wraps=saved)
        with self.assertRaisesRegex(RuntimeError, 'callback chain'):
            self.witness()
        gpu._on_remove = saved
        cache, wrapped, _ = self.owner._native_removal_bindings[id(gpu)]
        self.owner._native_removal_bindings[id(gpu)] = (cache, wrapped, Mock())
        with self.assertRaisesRegex(RuntimeError, 'callback chain'):
            self.witness()

    def test_cpu_deactivation_callback_or_overridden_pop_rejected(self):
        gpu = self.manager._active_adapters
        saved = gpu.deactivate_fn
        gpu.deactivate_fn = self.manager.deactivate_adapter
        with self.assertRaisesRegex(RuntimeError, 'callback chain'):
            self.witness()
        gpu.deactivate_fn = saved
        gpu.pop = Mock()
        with self.assertRaisesRegex(RuntimeError, 'removal method'):
            self.witness()

    def test_staged_construction_membership_and_thread_changes_rejected(self):
        before = self.witness()
        self.owner._native_host_build = object()
        with self.assertRaisesRegex(RuntimeError, 'staging or construction'):
            self.witness(previous=before)
        self.owner._native_host_build = None
        self.owner._staged_host[3] = {'model': object()}
        with self.assertRaisesRegex(RuntimeError, 'staging or construction'):
            self.witness(previous=before)
        self.owner._staged_host.clear()
        self.manager._registered_adapters[2] = SimpleNamespace(id=2)
        with self.assertRaisesRegex(RuntimeError, 'membership changed'):
            self.witness(previous=before)
        self.owner.thread_id = -1
        with self.assertRaisesRegex(RuntimeError, 'qualified native owner'):
            self.witness()


class AllocatorReobservation(unittest.TestCase):
    def test_exact_graph_projection_with_staging_alias_and_fresh_counters(self):
        rows = [dict(pinned=False, allocated_bytes=64, adapter_ids=[1]),
                dict(pinned=False, allocated_bytes=128, adapter_ids=[2]),
                dict(pinned=True, allocated_bytes=128, adapter_ids=[2]),
                dict(pinned=False, allocated_bytes=32, adapter_ids=[1, 2])]
        inventory = dict(host_allocations=rows, host_staged_adapter_ids=[2])
        observe = Mock(return_value=stats())
        with patch.object(monitor, 'torch', SimpleNamespace(cuda=SimpleNamespace(
                host_memory_stats=observe))):
            before = monitor._ieee_pinned_host_observation(inventory)
            saved = copy.deepcopy(before)
            observe.return_value = stats(allocated=1024, active=512)
            after = monitor._ieee_reobserve_pinned_host_allocator(before)
            full = monitor._ieee_pinned_host_observation(inventory)
            self.assertEqual(after, full)
            self.assertEqual(before, saved)  # no receipt/previous counter mutation
            self.assertEqual(after['accounted_tensor_bytes'], 1248)
            self.assertEqual(after['registered_exclusive_storage_bytes'], 64)
            self.assertEqual(observe.call_count, 3)
            observe.return_value = stats(active=0)
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                monitor._ieee_reobserve_pinned_host_allocator(before)
            observe.return_value = {}
            self.assertFalse(monitor._ieee_reobserve_pinned_host_allocator(before)['available'])

    def test_incomplete_or_target_scoped_storage_cannot_supply_budget(self):
        self.addCleanup(patch.stopall)
        patch.object(monitor, 'torch', SimpleNamespace(cuda=SimpleNamespace(
            host_memory_stats=lambda: stats()))).start()
        with self.assertRaisesRegex(ValueError, 'storage totals'):
            monitor._ieee_reobserve_pinned_host_allocator({})
        fields = ('registered_pinned_storage_bytes', 'registered_pageable_storage_bytes',
                  'staged_only_pinned_storage_bytes', 'staged_only_pageable_storage_bytes',
                  'registered_exclusive_storage_bytes')
        with self.assertRaisesRegex(ValueError, 'complete storage scope'):
            monitor._ieee_reobserve_pinned_host_allocator(dict.fromkeys(fields, 0))


class NativeWorkerRecheck(unittest.TestCase):
    def test_actual_worker_rechecks_workspace_counters_and_loader_qualification(self):
        import torch
        import vllm
        native, _ = observations.NativeHostFootprint().native_models()
        native.modules, native.moe_ep_load_spec = {}, None
        loader = SimpleNamespace(_adapter_manager=native)
        worker = monitor.IEEEWorkerObservationExtension()
        worker.device, worker.rank = SimpleNamespace(type='cuda'), 0
        worker.model_runner = SimpleNamespace(lora_manager=loader)
        worker._ieee_host_allocator_policy = {'verified': False}
        owner = SimpleNamespace(manager=native, snapshot=Mock(return_value={}),
            staged_models=lambda: {}, host_workspace_check=Mock(return_value=None),
            host_construction_reserved_bytes=Mock(return_value=0))
        with patch('faaslora.memory.residency_manager.IEEEBackendGPUReferences',
                   return_value=owner) as factory, \
             patch.object(vllm, '__version__', '0.30.0'), \
             patch.object(torch, '__version__', '2.13.0+cu130'), \
             patch.dict('sys.modules', {'vllm.utils.torch_utils': SimpleNamespace(PIN_MEMORY=True)}), \
             patch.object(torch.cuda, 'host_memory_stats', return_value=stats(), create=True) as allocator, \
             patch.object(monitor, '_ieee_lora_host_inventory',
                          wraps=monitor._ieee_lora_host_inventory) as inventory:
            worker.ieee_gpu_reference(operation='snapshot')
            proactive = factory.call_args.kwargs['proactive_host_activation_storage_witness']
            owner.preparation_loader = factory.call_args.kwargs['preparation_loader']
            owner.demand_loader = factory.call_args.kwargs['demand_loader']
            owner.completion_fence = factory.call_args.args[1]
            registered = factory.call_args.kwargs['registered_preparation_storage_witness']
            with patch.object(monitor, '_ieee_registered_preparation_storage_witness',
                              return_value={'kind': 'registered-fixture'}) as witness:
                saved = registered(adapter_int_id=4)
                self.assertEqual(registered(adapter_int_id=4, previous=saved), saved)
                fence = owner.completion_fence
                owner.completion_fence = Mock()
                with self.assertRaisesRegex(RuntimeError, 'worker binding changed'):
                    registered(adapter_int_id=4)
                owner.completion_fence = fence
                self.assertEqual(witness.call_count, 2)
            with patch.object(monitor, '_ieee_host_activation_storage_witness',
                              return_value={'kind': 'native-fixture'}) as witness:
                saved = proactive(adapter_int_id=4, loader=owner.preparation_loader)
                self.assertEqual(proactive(adapter_int_id=4, loader=owner.preparation_loader,
                                           previous=saved), saved)
                self.assertEqual(witness.call_count, 2)
                with self.assertRaisesRegex(RuntimeError, 'loader binding changed'):
                    proactive(adapter_int_id=4, loader=Mock())
                original = owner.preparation_loader
                owner.preparation_loader = Mock()
                with self.assertRaisesRegex(RuntimeError, 'loader binding changed'):
                    proactive(adapter_int_id=4, loader=owner.preparation_loader)
                owner.preparation_loader = original
                self.assertEqual(witness.call_count, 2)
            full = factory.call_args.kwargs['host_allocation_check']
            recheck = factory.call_args.kwargs['host_allocation_recheck']
            before = full(lora_path=None, reuse=True, tensor_budget_bytes=2048)
            after = recheck(before_check=before, tensor_budget_bytes=2048)
            self.assertEqual(before, after)
            inventory.assert_called_once()
            self.assertEqual(allocator.call_count, 2)
            self.assertEqual(owner.host_workspace_check.call_count, 2)
            self.assertEqual(owner.host_construction_reserved_bytes.call_count, 2)
            graph = monitor._ieee_lora_host_inventory(native, include_tensor_views=False)
            inventory.reset_mock()
            shared = full(lora_path=None, reuse=True, tensor_budget_bytes=2048,
                          _registered_preparation_inventory=graph)
            inventory.assert_not_called()
            self.assertEqual(shared['before'], before['before'])
            self.assertEqual(shared['before_storage_observation'],
                             'same_transaction_registered_preparation_v1')
            allocator.return_value = stats(allocated=4096)
            self.assertFalse(full(lora_path=None, reuse=True, tensor_budget_bytes=2048,
                                  _registered_preparation_inventory=graph)['admitted'])
            self.assertFalse(recheck(before_check=before, tensor_budget_bytes=2048)['admitted'])
            allocator.return_value = stats()
            owner.host_workspace_check.return_value = dict(admitted=False, reason='workspace-fixture')
            denied = recheck(before_check=before, tensor_budget_bytes=2048)
            self.assertFalse(denied['admitted'])
            self.assertEqual(denied['reason'], 'workspace-fixture')
            allocator.return_value = {}
            with self.assertRaisesRegex(RuntimeError, 'lacks allocator occupancy'):
                recheck(before_check=before, tensor_budget_bytes=2048)
            worker.model_runner.lora_manager = object()
            with self.assertRaisesRegex(RuntimeError, 'qualified dense'):
                recheck(before_check=before, tensor_budget_bytes=2048)


if __name__ == '__main__':
    unittest.main()
