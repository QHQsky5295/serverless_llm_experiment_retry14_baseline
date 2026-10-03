"""Ordered pending/GPU protection: synthetic fixtures, not GPU performance."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

from faaslora.clock import local_monotonic_clock_id
from scripts.run_all_experiments import InferenceEngine, SubprocessInferenceEngineProxy
from scripts import dedicated_engine_worker as worker
from tests import test_ieee_tc_request_lifecycle as lifecycle


def pending_receipt(intent_id):
    return dict(kind='ieee_pending_admission_v1', intent_id=intent_id,
        state='pending', clock_id=local_monotonic_clock_id(), physical_kv_reservation=False)


def command():
    return dict(intent_id='pending-test', prompt='canonical', max_tokens=4, adapter_id='adapter-a',
        gpu_reference_intent=dict(lease_id='gpu-test', adapter_int_id=InferenceEngine._lora_int_id('adapter-a'),
            lora_name='adapter-a', lora_path='/test/adapter-a', expected_owner_id='owner-test',
            expected_epoch=1, required_source_tier='gpu', expected_source_id='copy-test'))


class FrontendSequence(unittest.IsolatedAsyncioTestCase):
    def bridge(self):
        return NS(_lora_int_id=InferenceEngine._lora_int_id,
            ieee_register_pending=AsyncMock(side_effect=lambda **kw: pending_receipt(kw['intent_id'])),
            ieee_gpu_reference=AsyncMock(return_value={'native_receipt': 'unchanged'}))

    async def test_await_registration_before_exact_native_acquire(self):
        bridge = self.bridge()
        seen, proceed = asyncio.Event(), asyncio.Event()
        async def register(**kw):
            seen.set()
            await proceed.wait()
            return pending_receipt(kw['intent_id'])
        bridge.ieee_register_pending.side_effect = register
        task = asyncio.create_task(InferenceEngine.ieee_register_pending_gpu_reference(bridge, **command()))
        await seen.wait()
        bridge.ieee_gpu_reference.assert_not_awaited()
        proceed.set()
        result = await task
        self.assertEqual(result['kind'], 'ieee_pending_gpu_reference_v1')
        self.assertEqual(result['reference'], {'native_receipt': 'unchanged'})
        bridge.ieee_gpu_reference.assert_awaited_once_with(operation='demand_load_and_acquire',
            **command()['gpu_reference_intent'])

    async def test_reject_non_gpu_or_inexact_identity_before_register(self):
        for field, value in (('required_source_tier', 'host'), ('expected_source_id', ''),
                ('expected_owner_id', ''), ('expected_epoch', True), ('expected_epoch', 0),
                ('adapter_int_id', 0), ('lora_name', 'different'), ('lora_path', 'relative'),
                ('lease_id', None), ('extra_field', 'unrecognized')):
            with self.subTest(field=field, value=value):
                bridge, request = self.bridge(), command()
                request['gpu_reference_intent'][field] = value
                with self.assertRaisesRegex(ValueError, 'exact selected GPU'):
                    await InferenceEngine.ieee_register_pending_gpu_reference(bridge, **request)
                bridge.ieee_register_pending.assert_not_awaited()
                bridge.ieee_gpu_reference.assert_not_awaited()

    async def test_invalid_pending_ack_cannot_start_gpu_mutation(self):
        for field, value in (('kind', 'wrong'), ('intent_id', 'wrong'), ('state', 'registering'),
                ('clock_id', 'wrong'), ('physical_kv_reservation', True)):
            with self.subTest(field=field):
                bridge = self.bridge()
                bridge.ieee_register_pending.side_effect = None
                bridge.ieee_register_pending.return_value = pending_receipt('pending-test') | {field: value}
                with self.assertRaisesRegex(ValueError, 'pending acknowledgement'):
                    await InferenceEngine.ieee_register_pending_gpu_reference(bridge, **command())
                bridge.ieee_gpu_reference.assert_not_awaited()

    async def test_explicit_source_conflict_is_returned_not_retried_or_loaded(self):
        bridge = self.bridge()
        conflict = dict(acquired=False, reason='required_source_changed')
        bridge.ieee_gpu_reference.return_value = conflict
        result = await InferenceEngine.ieee_register_pending_gpu_reference(bridge, **command())
        self.assertIs(result['reference'], conflict)
        self.assertEqual(bridge.ieee_gpu_reference.await_count, 1)


class ControllerSequence(unittest.TestCase):
    def build(self, tier='gpu'):
        runner, slot, trace, plan, owner, observed = lifecycle.SelectedSourceAdmissionIntegration.build(self, tier)
        runner.model_cfg['ieee_admission_profile'] = {'test_only': True}
        slot.engine._lora_int_id = InferenceEngine._lora_int_id
        slot.engine.ieee_register_pending = AsyncMock(side_effect=lambda **kw: pending_receipt(kw['intent_id']))
        slot.engine.ieee_close_pending = AsyncMock(side_effect=lambda **kw: dict(**kw, closed=True))
        slot.engine.ieee_register_pending_gpu_reference = AsyncMock(side_effect=MethodType(
            InferenceEngine.ieee_register_pending_gpu_reference, slot.engine))
        return runner, slot, trace, plan, owner, observed

    def test_actual_gpu_request_preserves_pending_and_reference_until_retirement(self):
        runner, slot, trace, plan, owner, observed = self.build()
        events = []
        async def register(**kw):
            events.append('register')
            return pending_receipt(kw['intent_id'])
        async def reference(**kw):
            if kw['operation'] in ('demand_load_and_acquire', 'release'):
                events.append(kw['operation'])
            return await observed(**kw)
        generate = slot.engine.generate_prepared.side_effect
        async def infer(**kw):
            self.assertEqual(kw['pending_admission_id'], slot.engine.ieee_register_pending.await_args.kwargs['intent_id'])
            events.append('generate')
            return await generate(**kw)
        async def close(**kw):
            events.append('close')
            self.assertEqual(owner.snapshot()['live_leases'], 1)
            return dict(**kw, closed=True)
        slot.engine.ieee_register_pending.side_effect = register
        slot.engine.ieee_gpu_reference.side_effect = reference
        slot.engine.generate_prepared.side_effect = infer
        slot.engine.ieee_close_pending.side_effect = close
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertTrue(result.success, result.error)
        self.assertEqual(events, ['register', 'demand_load_and_acquire', 'generate', 'close', 'release'])
        self.assertEqual(slot.engine.ieee_register_pending_gpu_reference.await_count, 1)
        self.assertEqual(result.readiness_tier_before_dispatch, 'gpu')
        self.assertFalse(result.gpu_reference_evidence['receipt']['native_load_invoked'])
        self.assertEqual(result.gpu_reference_evidence['pending_kv_admission']['state'], 'closed')
        self.assertEqual(owner.snapshot()['live_leases'], 0)
        self.assertEqual(slot.active_requests, 0)
        self.assertFalse(slot.ieee_pending_load_ids)

    def test_host_path_keeps_separate_registration_and_pending_load_update(self):
        runner, slot, trace, plan, owner, observed = self.build('host')
        async def protect(**kw):
            if kw['operation'] == 'hold_host_source':
                self.assertEqual(slot.engine.ieee_register_pending.await_count, 1)
                self.assertEqual(slot.ieee_pending_load_ids, {trace.request_id})
            return await observed(**kw)
        slot.engine.ieee_gpu_reference.side_effect = protect
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertTrue(result.success, result.error)
        slot.engine.ieee_register_pending_gpu_reference.assert_not_awaited()
        self.assertEqual(owner.snapshot()['live_leases'], 0)
        self.assertFalse(slot.ieee_pending_load_ids)

    def test_source_changed_after_registration_retries_whole_router(self):
        runner, slot, trace, plan, owner, observed = self.build()
        registrations = []
        async def register(**kw):
            registrations.append(kw['intent_id'])
            if len(registrations) == 1:
                owner.manager.deactivate(InferenceEngine._lora_int_id(trace.adapter_id))
            return pending_receipt(kw['intent_id'])
        slot.engine.ieee_register_pending.side_effect = register
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.readiness_tier_before_dispatch, 'host')
        self.assertEqual(len(set(registrations)), 2)
        self.assertEqual(slot.engine.ieee_close_pending.await_count, 2)
        self.assertEqual(slot.engine.ieee_register_pending_gpu_reference.await_count, 1)
        first = result.gpu_reference_evidence['prior_routing_attempts'][0]
        self.assertFalse(first['last_conflict']['acquired'])
        self.assertEqual(first['pending_kv_admission']['state'], 'closed')
        self.assertEqual(owner.snapshot()['live_leases'], 0)

    def test_lost_or_corrupt_combined_reply_retains_possible_gpu_ownership(self):
        for mode in ('lost', 'pending', 'reference'):
            with self.subTest(mode=mode):
                runner, slot, trace, plan, owner, observed = self.build()
                async def corrupt(**kw):
                    result = await InferenceEngine.ieee_register_pending_gpu_reference(slot.engine, **kw)
                    if mode == 'lost':
                        raise ConnectionError('lost combined reply')
                    if mode == 'pending':
                        result['pending']['intent_id'] = 'wrong'
                    else:
                        result['reference']['lease_id'] = 'wrong'
                    return result
                slot.engine.ieee_register_pending_gpu_reference.side_effect = corrupt
                with self.assertRaises((ValueError, ConnectionError)):
                    asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
                self.assertEqual(owner.snapshot()['live_leases'], 1)
                self.assertEqual(slot.active_requests, 1)
                self.assertEqual(slot.status, 'draining')
                self.assertEqual(len(runner._unsettled_runtime_reservations), 1)
                slot.engine.ieee_close_pending.assert_not_awaited()
                slot.engine.generate_prepared.assert_not_awaited()


class WorkerTransport(unittest.IsolatedAsyncioTestCase):
    async def roundtrip(self, cancel_stage=None):
        ready = asyncio.get_running_loop().create_future()
        entered, proceed, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = []
        class FakeEngine:
            _lora_int_id = staticmethod(InferenceEngine._lora_int_id)
            ieee_register_pending_gpu_reference = InferenceEngine.ieee_register_pending_gpu_reference
            def __init__(self, cfg, *args):
                self.model_cfg = cfg
            async def initialize(self):
                pass
            async def shutdown(self):
                pass
            async def ieee_register_pending(self, **kw):
                calls.append('pending')
                if cancel_stage == 'pending':
                    entered.set()
                    await proceed.wait()
                return pending_receipt(kw['intent_id'])
            async def ieee_gpu_reference(self, **kw):
                calls.append('gpu')
                if cancel_stage == 'gpu':
                    entered.set()
                    await proceed.wait()
                completed.set()
                return dict(acquired=True, lease_id=kw['lease_id'])
            async def ieee_close_pending(self, **kw):
                return dict(**kw, closed=True)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'payload.json'
            path.write_text(json.dumps(dict(repo_root=str(Path.cwd()), model_cfg={}, cost_model={})))
            with patch('scripts.run_all_experiments.InferenceEngine', FakeEngine), \
                 patch.object(worker, '_write_ready', side_effect=lambda path, value: ready.set_result(value)):
                serving = asyncio.create_task(worker._run_worker(path, Path(temp) / 'ready.json'))
                address = await asyncio.wait_for(ready, 2.)
                proxy = SubprocessInferenceEngineProxy.__new__(SubprocessInferenceEngineProxy)
                proxy.model_cfg = dict(timing_contract='ieee_tc_native_v1')
                proxy._engine_dead, proxy._native_rpc_uncertain = False, {}
                proxy._process = NS(poll=Mock(return_value=None))
                proxy._host, proxy._port = address['host'], address['port']
                proxy._rpc_pool_size, proxy._rpc_channel_queue = 1, None
                proxy._rpc_channel_init_lock, proxy._rpc_channels = asyncio.Lock(), []
                proxy._with_worker_log_context = lambda value: value
                task = asyncio.create_task(proxy.ieee_register_pending_gpu_reference(**command()))
                try:
                    if cancel_stage:
                        await asyncio.wait_for(entered.wait(), 2.)
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                        uncertainty = list(proxy._native_rpc_uncertain.values())
                        self.assertEqual(len(uncertainty), 1)
                        self.assertEqual((uncertainty[0]['owner_id'], uncertainty[0]['lease_id'],
                            uncertainty[0]['intent_id']), ('owner-test', 'gpu-test', 'pending-test'))
                        proceed.set()
                        await asyncio.wait_for(completed.wait(), 2.)
                        await proxy.ieee_close_pending(intent_id='pending-test')
                        self.assertEqual(list(proxy._native_rpc_uncertain.values()), uncertainty)
                        with self.assertRaises(RuntimeError):
                            await proxy._rpc('generate')
                    else:
                        reply = await asyncio.wait_for(task, 2.)
                        self.assertEqual(reply['kind'], 'ieee_pending_gpu_reference_v1')
                        self.assertEqual(reply['pending'], pending_receipt('pending-test'))
                        self.assertEqual(reply['reference'], dict(acquired=True, lease_id='gpu-test'))
                        self.assertFalse(proxy._native_rpc_uncertain)
                    self.assertEqual(calls, ['pending', 'gpu'])
                finally:
                    proceed.set()
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    for channel in list(proxy._rpc_channels):
                        await proxy._drop_rpc_channel(channel)
                    await proxy._rpc('shutdown')
                    await asyncio.wait_for(serving, 2.)

    async def test_actual_worker_and_proxy_preserve_both_native_receipts(self):
        await self.roundtrip()

    async def test_cancel_during_pending_does_not_clear_unknown_gpu_pin(self):
        await self.roundtrip('pending')

    async def test_cancel_during_gpu_does_not_clear_unknown_gpu_pin(self):
        await self.roundtrip('gpu')


if __name__ == '__main__':
    unittest.main()
