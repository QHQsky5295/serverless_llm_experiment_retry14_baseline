"""Initialized planning uses fresh complete graphs, not descriptive reports.

Metadata/real-owner and actual transport tests, not native model performance.
"""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

from scripts.run_all_experiments import InferenceEngine, SubprocessInferenceEngineProxy
from scripts import dedicated_engine_worker as worker
from tests import test_ieee_tc_transfer_pressure as fixtures
from faaslora.preloading.planning_cpu import execution_preparation_bundle


def projection(native):
    value = copy.deepcopy(native)
    value.pop('native_staging_footprints', None)
    value.pop('native_host_allocator', None)
    value['native_footprints'].pop('host_tensor_views', None)
    return value


def without_provenance(value):
    if isinstance(value, dict):
        return {key: without_provenance(item) for key, item in value.items()
                if key not in ('plan_sha256', 'planning_sha256', 'source_snapshot_id',
                               'native_staging_footprints', 'native_host_allocator', 'host_tensor_views',
                               'file_plan_id', 'captured_at')}
    if isinstance(value, (tuple, list)):
        return type(value)(without_provenance(item) for item in value)
    return value


class PlanningSnapshot(unittest.TestCase):
    def make(self):
        factory = fixtures.OwnedPreparationPlanning()
        self.addCleanup(factory.doCleanups)
        data = factory.make()
        fixture, runner, queue, slot, native = data
        self.addCleanup(lambda: asyncio.run(queue.close()))
        native['native_footprints']['host_tensor_views'] = [{'fixture_description': True}]
        native['native_staging_footprints'] = {'fixture_description': True}
        native['native_host_allocator'] = {'fixture_description': True}
        original = slot.engine.ieee_gpu_reference.side_effect
        async def observe(**kw):
            value = await original(**kw)
            return projection(value) if kw['operation'] == 'routing_source_snapshot' else value
        slot.engine.ieee_gpu_reference.side_effect = observe
        slot.engine.ieee_gpu_reference.reset_mock()
        return factory, data

    def test_initialized_planner_uses_complete_native_graph_not_compact_wire(self):
        factory, (fixture, runner, queue, slot, native) = self.make()
        slot.engine.ieee_routing_sources = AsyncMock(side_effect=AssertionError('compact wire'))
        slot.engine.ieee_request_sources = AsyncMock(side_effect=AssertionError('target-scoped capacity'))
        plan = factory.plan(runner, slot)
        slot.engine.ieee_gpu_reference.assert_awaited_once_with(operation='routing_source_snapshot')
        actual = plan['source_view']['native']
        self.assertEqual(actual, projection(native))
        self.assertEqual(actual['native_footprints']['host_tensor_storage_bytes'], 1024)
        self.assertEqual(len(actual['native_footprints']['host_allocations']), 2)
        self.assertEqual(actual['replacement_protected_adapter_ids'], [])
        self.assertEqual([c.artifact_id for c in plan['selected']['gpu']], ['a', 'd'])
        self.assertEqual(plan['remaining_bytes'], dict(gpu=1048576, host=0, nvme=90112))
        self.assertFalse(plan['physical_resources_reserved'])

    def test_full_and_projected_runner_plans_objectives_match_in_both_modes(self):
        factory, (fixture, runner, queue, slot, native) = self.make()
        endpoint = slot.engine.ieee_gpu_reference.side_effect
        for mode in ('handoff', 'residency'):
            with self.subTest(mode=mode):
                outputs = []
                for full in (True, False):
                    async def observe(**kw):
                        self.assertEqual(kw, {'operation': 'routing_source_snapshot'})
                        return copy.deepcopy(native) if full else projection(native)
                    slot.engine.ieee_gpu_reference.side_effect = observe
                    async def run():
                        plan = await runner._plan_ieee_preparation_for_slot(slot=slot, mode=mode)
                        return await execution_preparation_bundle(runner._stack, plan,
                            size_edges_bytes=runner._preparation_profiles.size_edges_bytes)
                    outputs.append(asyncio.run(run()))
                self.assertEqual(without_provenance(outputs[0]), without_provenance(outputs[1]))
                self.assertNotEqual(outputs[0][0]['source_snapshot_id'], outputs[1][0]['source_snapshot_id'])
        slot.engine.ieee_gpu_reference.side_effect = endpoint

    def test_every_epoch_reads_fresh_protection_and_never_reuses_selection(self):
        factory, (fixture, runner, queue, slot, native) = self.make()
        first = factory.plan(runner, slot)
        victim = native['slot_adapter_ids'][0]
        native['epoch'] += 1
        native['replacement_protected_adapter_ids'] = [victim]
        second = factory.plan(runner, slot)
        self.assertEqual(slot.engine.ieee_gpu_reference.await_count, 2)
        self.assertEqual(first['source_view']['native']['replacement_protected_adapter_ids'], [])
        self.assertEqual(second['source_view']['native']['replacement_protected_adapter_ids'], [victim])
        self.assertGreater(second['source_view']['native']['epoch'], first['source_view']['native']['epoch'])
        self.assertNotEqual(first['selected']['gpu'], second['selected']['gpu'])

    def test_global_union_alias_credit_and_protection_corruption_are_rejected(self):
        factory, (fixture, runner, queue, slot, native) = self.make()
        original = copy.deepcopy(native)
        mutations = {
            'union': lambda x: x['native_footprints'].__setitem__('host_tensor_storage_bytes', 0),
            'alias': lambda x: x['native_footprints']['host_allocations'][0].__setitem__('adapter_ids', []),
            'credit': lambda x: x['native_footprints']['host_adapter_footprints'][0].__setitem__('exclusive_storage_bytes', 0),
            'gpu': lambda x: x['native_footprints'].__setitem__('pool_allocated_bytes', 1),
            'missing_protection': lambda x: x.pop('replacement_protected_adapter_ids'),
        }
        for name, corrupt in mutations.items():
            with self.subTest(name=name):
                native.clear(); native.update(copy.deepcopy(original))
                corrupt(native)
                with self.assertRaises(ValueError):
                    factory.plan(runner, slot)
        self.assertFalse(fixture.owner.leases)
        self.assertFalse(fixture.owner.materializations)

    def test_cancel_observation_never_submits_plan_or_reserves_movement(self):
        factory, (fixture, runner, queue, slot, native) = self.make()
        async def run():
            entered = asyncio.Event()
            async def wait(**kw):
                self.assertEqual(kw, {'operation': 'routing_source_snapshot'})
                entered.set()
                await asyncio.Future()
            slot.engine.ieee_gpu_reference.side_effect = wait
            with patch.object(runner._stack, 'plan_ieee_owned_preparation_async', new_callable=AsyncMock) as planner:
                task = asyncio.create_task(runner._plan_ieee_preparation_for_slot(slot=slot, mode='residency'))
                await entered.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError): await task
                planner.assert_not_awaited()
            self.assertFalse(fixture.owner.leases)
            self.assertFalse(fixture.owner.materializations)
        asyncio.run(run())


class CapacityConsumer(unittest.IsolatedAsyncioTestCase):
    async def test_deferred_capacity_still_requires_full_allocator_observation(self):
        fixture = fixtures.DeferredNativeHostCapacity()
        runner, queue, engine, state, calls = await fixture.make()
        try:
            await runner._refresh_ieee_deferred_host_capacity()
            engine.ieee_gpu_reference.assert_awaited_once_with(operation='source_snapshot')
            state['accounted_tensor_bytes'] = 800
            await runner._refresh_ieee_deferred_host_capacity()
            await asyncio.sleep(0)
            self.assertEqual(len(calls), 4)
        finally:
            await queue.close()


class PlanningTransport(unittest.IsolatedAsyncioTestCase):
    async def roundtrip(self, cancelled=False):
        ready = asyncio.get_running_loop().create_future()
        entered, proceed, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = []
        observed = dict(kind='native_lora_sources_v1', owner_id='fixture-owner', epoch=7,
            native_footprints={'host_allocations': [{'allocation_id': 0}],
                               'pool_tensor_views': [{'dtype': 'torch.float16'}]},
            replacement_protected_adapter_ids=[123])
        class FakeEngine:
            # Real public forwarder; only native collective computation is stubbed.
            ieee_gpu_reference = InferenceEngine.ieee_gpu_reference
            def __init__(self, cfg, *args):
                self.model_cfg = dict(cfg, ieee_gpu_references=True)
                self.backend, self._engine_dead = 'vllm', False
                self.engine = NS(collective_rpc=self.collective)
            async def collective(self, method, *, kwargs):
                calls.append((method, kwargs))
                entered.set()
                if cancelled: await proceed.wait()
                completed.set()
                return [copy.deepcopy(observed)]
            async def initialize(self): pass
            async def shutdown(self): pass
        with tempfile.TemporaryDirectory() as root:
            payload = Path(root)/'payload.json'
            payload.write_text(json.dumps(dict(repo_root=str(Path.cwd()), model_cfg={}, cost_model={})))
            with patch('scripts.run_all_experiments.InferenceEngine', FakeEngine), \
                 patch.object(worker, '_write_ready', side_effect=lambda path, value: ready.set_result(value)):
                serving = asyncio.create_task(worker._run_worker(payload, Path(root)/'ready.json'))
                address = await asyncio.wait_for(ready, 2.)
                proxy = SubprocessInferenceEngineProxy(process=NS(poll=Mock(return_value=None)),
                    host=address['host'], port=address['port'], model_cfg={'timing_contract':'ieee_tc_native_v1'},
                    cost_model={}, device_id=0, workdir=Path(root), log_path=Path(root)/'none')
                task = asyncio.create_task(proxy.ieee_gpu_reference(operation='routing_source_snapshot'))
                try:
                    if cancelled:
                        await asyncio.wait_for(entered.wait(), 2.)
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError): await task
                        self.assertFalse(proxy._native_rpc_uncertain)  # Read-only, not a resource lease.
                        proceed.set()
                        await asyncio.wait_for(completed.wait(), 2.)
                        result = await proxy.ieee_gpu_reference(operation='routing_source_snapshot')
                    else:
                        result = await asyncio.wait_for(task, 2.)
                    self.assertEqual({k: result[k] for k in observed}, observed)
                    self.assertTrue(all(call == ('ieee_gpu_reference', {'operation':'routing_source_snapshot'}) for call in calls))
                    self.assertEqual(len(calls), 2 if cancelled else 1)
                finally:
                    proceed.set()
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    for channel in list(proxy._rpc_channels): await proxy._drop_rpc_channel(channel)
                    await proxy._rpc('shutdown')
                    await asyncio.wait_for(serving, 2.)

    async def test_actual_worker_proxy_and_engine_keep_complete_graph(self):
        await self.roundtrip()

    async def test_cancelled_read_has_no_resource_claim_and_fresh_read_still_works(self):
        await self.roundtrip(cancelled=True)


if __name__ == '__main__':
    unittest.main()
