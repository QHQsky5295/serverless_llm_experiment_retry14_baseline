"""Only the fresh read-only planning transport changes representation."""
import asyncio
import copy
from dataclasses import FrozenInstanceError, replace
import json
import pickle
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import msgspec

from faaslora.preloading.planning_wire import FrozenPlanningSnapshot, decode_planning_rpc_reply


class PlanningWire(unittest.TestCase):
    def test_exact_types_float_bits_and_detached_timing(self):
        value = dict(epoch=2**80, owner='α', captured_monotonic_s=15.25,
                     sources=[None, True, -0., 5e-324, 1.7976931348623157e308],
                     timing={'worker_response_ready_wall_time': 100., 'native_clock_id': 'test'})
        raw = bytearray(json.dumps(dict(ok=True, result=value)).encode())
        wire = decode_planning_rpc_reply(raw)['result']
        raw[:] = b'changed'
        self.assertIs(type(wire.payload), bytes)
        self.assertEqual(wire.decode(), value)
        self.assertEqual(wire.decode()['sources'][2].hex(), (-0.).hex())
        self.assertEqual(wire.decode()['sources'][3].hex(), (5e-324).hex())
        changed = wire.timing_copy()
        changed['native_clock_id'] = 'later'
        self.assertEqual(wire.decode(), value)
        transported = pickle.loads(pickle.dumps(wire, protocol=5))
        self.assertEqual(transported.decode(), value)
        with self.assertRaises(FrozenInstanceError):
            wire.payload = b'{}'
        with self.assertRaises(TypeError):
            FrozenPlanningSnapshot(bytearray(b'{}'), pickle.dumps(None))

    def test_error_and_invalid_inputs_never_become_valid_state(self):
        self.assertEqual(decode_planning_rpc_reply(b'{"ok":false,"error":"rejected"}'),
                         {'ok': False, 'error': 'rejected'})
        for raw in (b'{"ok":true,"result":', b'{"ok":true,"result":{"x":NaN}}',
                    b'{"ok":true,"result":{}}extra'):
            with self.subTest(raw=raw), self.assertRaises(msgspec.DecodeError):
                decode_planning_rpc_reply(raw)
        # Lexical scanning can defer numeric overflow/UTF-8 validation; the full
        # decode must reject before owned_preparation_inputs or publication.
        for raw in (b'{"ok":true,"result":{"x":1e309}}',
                    b'{"ok":true,"result":{"x":"\xff"}}'):
            with self.subTest(raw=raw), self.assertRaises((msgspec.DecodeError, UnicodeError)):
                decode_planning_rpc_reply(raw)['result'].decode()


class PlanningWireRPC(unittest.IsolatedAsyncioTestCase):
    def proxy(self):
        from tests.test_ieee_tc_request_lifecycle import NativeRPCOwnership
        return NativeRPCOwnership().proxy()

    async def test_same_wire_status_timing_and_unchanged_regular_api(self):
        proxy = self.proxy()
        value = dict(owner_id='owner', epoch=3, captured_monotonic_s=12.,
                     native_footprints={'allocations': [1, 2, 3]},
                     timing={'worker_response_ready_wall_time': 98., 'native_clock_id': 'clock'})
        raw = json.dumps(dict(ok=True, result=value)).encode()+b'\n'
        payloads = []
        async def exchange(channel, payload, **kwargs):
            payloads.append(json.loads(payload))
            return raw, 2., 7., 100.
        proxy._native_rpc_roundtrip = AsyncMock(side_effect=exchange)
        with patch('scripts.run_all_experiments.time.time', return_value=100.), \
             patch('scripts.run_all_experiments.time.perf_counter', return_value=10.):
            ordinary = await proxy.ieee_gpu_reference(operation='routing_source_snapshot')
            with patch('scripts.run_all_experiments._decode_native_rpc_frame',
                       side_effect=AssertionError('parent materialized full graph')):
                deferred = await proxy.ieee_planning_sources()
        self.assertIs(type(deferred), FrozenPlanningSnapshot)
        self.assertEqual(deferred.decode(), ordinary)
        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual(payloads[1]['kwargs'], {'operation': 'routing_source_snapshot'})
        self.assertFalse(proxy._native_rpc_uncertain)
        self.assertEqual(proxy._drop_rpc_channel.await_count, 2)  # Fresh read-only channel each time.

    async def test_only_readonly_planning_can_opt_in(self):
        proxy = self.proxy()
        proxy._native_rpc_roundtrip = AsyncMock()
        for cmd, kwargs in [('generate', {}), ('ieee_gpu_reference', {'operation': 'release'}),
                            ('ieee_gpu_reference', {'operation': 'routing_source_snapshot', 'extra': 1})]:
            with self.subTest(cmd=cmd, kwargs=kwargs), self.assertRaisesRegex(ValueError, 'read-only'):
                await proxy._rpc(cmd, _planning_snapshot_wire=True, **kwargs)
        proxy.model_cfg['timing_contract'] = 'legacy'
        with self.assertRaisesRegex(ValueError, 'read-only'):
            await proxy.ieee_planning_sources()
        proxy._open_rpc_channel.assert_not_awaited()
        proxy._native_rpc_roundtrip.assert_not_awaited()

    async def test_remote_failure_and_cancel_keep_readonly_ownership(self):
        for response in ('error', 'cancel'):
            proxy = self.proxy()
            proxy._native_rpc_roundtrip = (AsyncMock(side_effect=asyncio.CancelledError)
                if response == 'cancel' else AsyncMock(return_value=(
                    b'{"ok":false,"error":"native planning rejected"}', 0., 0., time.time())))
            with self.assertRaises(asyncio.CancelledError if response == 'cancel' else RuntimeError):
                await proxy.ieee_planning_sources()
            self.assertFalse(proxy._native_rpc_uncertain)
            proxy._native_rpc_roundtrip.assert_awaited_once()
            proxy._drop_rpc_channel.assert_awaited_once()


class PlanningWireWorker(unittest.IsolatedAsyncioTestCase):
    async def test_runner_uses_wire_only_for_native_subprocess_planning(self):
        from faaslora.experiment.instance_pool import FrozenServiceProfiles
        from tests.test_ieee_tc_planning_cpu import PlanningCPU
        from tests.test_ieee_tc_request_lifecycle import NativeRPCOwnership
        case = PlanningCPU()
        self.addCleanup(case.doCleanups)
        args, runner, slot = await asyncio.to_thread(case.inputs)
        native = args['native_snapshot']
        wire = decode_planning_rpc_reply(json.dumps(dict(ok=True, result=native)).encode())['result']
        proxy = NativeRPCOwnership().proxy()
        proxy.model_cfg = dict(slot.engine.model_cfg)
        slot.engine = proxy
        state = object()
        plan = SimpleNamespace(native_source_publication=lambda: (state, 'GPU-test'))
        for contract in ('ieee_tc_native_v1', 'legacy'):
            proxy.model_cfg['timing_contract'] = contract
            # Each synthetic transport fixture has its own matching runtime
            # identity; do not disable the production profile-identity check.
            runner._preparation_profiles = replace(runner._preparation_profiles,
                model_config_json=json.dumps(FrozenServiceProfiles.model_identity(proxy.model_cfg),
                                            sort_keys=True, allow_nan=False))
            proxy.ieee_planning_sources = AsyncMock(return_value=wire)
            proxy.ieee_gpu_reference = AsyncMock(return_value=native)
            with patch.object(runner, '_attach_ieee_host_budget', new=AsyncMock()), \
                 patch.object(runner._stack, 'plan_ieee_owned_preparation_async',
                              new=AsyncMock(return_value=plan)) as compute, \
                 patch.object(runner, '_ieee_publish_validated_planning_sources') as publish:
                self.assertIs(await runner._plan_ieee_preparation_for_slot(slot=slot, mode='residency'), plan)
            self.assertIs(compute.call_args.kwargs['native_snapshot'],
                          wire if contract == 'ieee_tc_native_v1' else native)
            self.assertEqual(proxy.ieee_planning_sources.await_count, int(contract == 'ieee_tc_native_v1'))
            self.assertEqual(proxy.ieee_gpu_reference.await_count, int(contract == 'legacy'))
            publish.assert_called_once_with(slot, state, device_uuid='GPU-test', expected_engine=proxy)

    async def test_existing_spawned_selector_plan_and_publication_match(self):
        from tests.test_ieee_tc_planning_cpu import PlanningCPU
        from faaslora.preloading.planning_cpu import IEEEPlanningCPU, execute_planning_message
        case = PlanningCPU()
        self.addCleanup(case.doCleanups)
        args, runner, _ = await asyncio.to_thread(case.inputs)
        limit = runner._stack.preloading_planner.max_dp_buffer_bytes
        native = args['native_snapshot']
        # Both APIs attach a timing dict. Use the same frozen values in each arm.
        native.setdefault('timing', {})
        expected, receipt = execute_planning_message(pickle.dumps(('owned_execution_epoch', limit, args), protocol=5))
        wire = decode_planning_rpc_reply(json.dumps(dict(ok=True, result=native)).encode())['result']
        candidate_args = dict(args, native_snapshot=wire)
        worker = IEEEPlanningCPU()
        self.addAsyncCleanup(worker.close)
        actual = await worker.run('owned_execution_epoch', limit, candidate_args)
        original_plan, selected, objectives = pickle.loads(expected)
        plan, chosen, actual_objectives = actual.execution_bundle_copy()
        self.assertEqual((plan, chosen), (original_plan, selected))
        self.assertEqual(actual.native_source_publication(), receipt['_native_source_publication'])
        self.assertEqual(actual_objectives['gpu'], objectives['gpu'])
        # File plan UUIDs are deliberately newly allocated, not cached results.
        self.assertEqual(actual_objectives['size_edges_bytes'], objectives['size_edges_bytes'])
        self.assertNotEqual(actual_objectives['file_plan_id'], objectives['file_plan_id'])
        broken = copy.deepcopy(native)
        broken['native_footprints']['host_tensor_storage_bytes'] = -1
        bad = decode_planning_rpc_reply(json.dumps(dict(ok=True, result=broken)).encode())['result']
        with self.assertRaises(ValueError):
            await worker.run('owned_execution_epoch', limit, dict(args, native_snapshot=bad))
        self.assertEqual(worker.events[-1]['state'], 'failed')

    async def test_wire_cancel_joins_existing_owner_and_never_publishes(self):
        from tests.test_ieee_tc_planning_cpu import PlanningCPU
        from faaslora.preloading.planning_cpu import IEEEPlanningCPU
        case = PlanningCPU()
        self.addCleanup(case.doCleanups)
        args, runner, _ = await asyncio.to_thread(case.inputs)
        args['native_snapshot'] = decode_planning_rpc_reply(
            json.dumps(dict(ok=True, result=args['native_snapshot'])).encode())['result']
        worker = IEEEPlanningCPU()
        self.addAsyncCleanup(worker.close)
        task = asyncio.create_task(worker.run('owned_execution_epoch',
            runner._stack.preloading_planner.max_dp_buffer_bytes, args))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(worker.events[-1]['state'], 'cancelled_result_discarded')
        self.assertTrue(worker.events[-1]['frozen_execution_validated'])


if __name__ == '__main__':
    unittest.main()
