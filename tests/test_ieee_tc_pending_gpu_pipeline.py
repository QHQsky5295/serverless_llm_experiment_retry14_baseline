"""Ordered pending/GPU protection: synthetic fixtures, not GPU performance."""
import asyncio
import json
from pathlib import Path
import tempfile
import sys
from types import MethodType, ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

from faaslora.clock import local_monotonic_clock_id
from scripts.run_all_experiments import InferenceEngine, SubprocessInferenceEngineProxy
from scripts import dedicated_engine_worker as worker
from tests import test_ieee_tc_request_lifecycle as lifecycle
from tests.test_ieee_tc_native_retirement import frontend_type
from tests.test_ieee_tc_scheduler_observation import NativeHookWiring, scheduler
from tests.test_ieee_tc_pending_admission import descriptor, utility_call


def pending_receipt(intent_id):
    return dict(kind='ieee_pending_admission_v1', intent_id=intent_id,
        state='pending', clock_id=local_monotonic_clock_id(), physical_kv_reservation=False)


def command():
    return dict(intent_id='pending-test', prompt='canonical', max_tokens=4, adapter_id='adapter-a',
        gpu_reference_intent=dict(lease_id='gpu-test', adapter_int_id=InferenceEngine._lora_int_id('adapter-a'),
            lora_name='adapter-a', lora_path='/test/adapter-a', expected_owner_id='owner-test',
            expected_epoch=1, required_source_tier='gpu', expected_source_id='copy-test'))


async def fake_native_registration(bridge, kw):
    """Synthetic core sequencing for the existing controller/transport fixtures.

    The real native-core and frontend methods are exercised separately below.
    """
    pending = pending_receipt(kw['intent_id'])
    if 'gpu_reference_intent' not in kw:
        return pending
    reference = await bridge.ieee_gpu_reference(operation='demand_load_and_acquire',
                                               **kw['gpu_reference_intent'])
    return dict(kind='ieee_pending_gpu_reference_v1', pending=pending, reference=reference,
                execution_scope='single_native_core_turn_v1')


class FrontendSequence(unittest.IsolatedAsyncioTestCase):
    def bridge(self):
        bridge = NS(_lora_int_id=InferenceEngine._lora_int_id, model_cfg={'ieee_gpu_references': True},
            ieee_gpu_reference=AsyncMock(return_value={'native_receipt': 'unchanged'}))
        async def register(**kw):
            return await fake_native_registration(bridge, kw)
        bridge.ieee_register_pending = AsyncMock(side_effect=register)
        return bridge

    async def test_await_registration_before_exact_native_acquire(self):
        bridge = self.bridge()
        seen, proceed = asyncio.Event(), asyncio.Event()
        async def register(**kw):
            seen.set()
            await proceed.wait()
            return await fake_native_registration(bridge, kw)
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

    async def test_digest_bearing_reference_is_forwarded_and_validated(self):
        bridge = self.bridge()
        request = command()
        request['gpu_reference_intent']['expected_content_sha256'] = 'a' * 64
        result = await InferenceEngine.ieee_register_pending_gpu_reference(bridge, **request)
        self.assertEqual(result['kind'], 'ieee_pending_gpu_reference_v1')
        bridge.ieee_gpu_reference.assert_awaited_once_with(
            operation='demand_load_and_acquire', **request['gpu_reference_intent'])

        for digest in ('', 'a' * 63, 'a' * 65, 'g' * 64, 123):
            with self.subTest(digest=digest):
                bridge = self.bridge()
                request = command()
                request['gpu_reference_intent']['expected_content_sha256'] = digest
                with self.assertRaisesRegex(ValueError, 'valid content identity'):
                    await InferenceEngine.ieee_register_pending_gpu_reference(bridge, **request)
                bridge.ieee_register_pending.assert_not_awaited()
                bridge.ieee_gpu_reference.assert_not_awaited()

    async def test_invalid_combined_ack_is_rejected_without_further_mutation(self):
        for field, value in (('kind', 'wrong'), ('intent_id', 'wrong'), ('state', 'registering'),
                ('clock_id', 'wrong'), ('physical_kv_reservation', True)):
            with self.subTest(field=field):
                bridge = self.bridge()
                bridge.ieee_register_pending.side_effect = None
                bridge.ieee_register_pending.return_value = dict(kind='ieee_pending_gpu_reference_v1',
                    pending=pending_receipt('pending-test') | {field: value}, reference={},
                    execution_scope='single_native_core_turn_v1')
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
        slot.engine.model_cfg = {'ieee_gpu_references': True}
        async def register(**kw):
            return await fake_native_registration(slot.engine, kw)
        slot.engine.ieee_register_pending = AsyncMock(side_effect=register)
        slot.engine.ieee_close_pending = AsyncMock(side_effect=lambda **kw: dict(**kw, closed=True))
        slot.engine.ieee_register_pending_gpu_reference = AsyncMock(side_effect=MethodType(
            InferenceEngine.ieee_register_pending_gpu_reference, slot.engine))
        return runner, slot, trace, plan, owner, observed

    def test_actual_gpu_request_preserves_pending_and_reference_until_retirement(self):
        runner, slot, trace, plan, owner, observed = self.build()
        events = []
        async def register(**kw):
            events.append('register')
            return await fake_native_registration(slot.engine, kw)
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
            return await fake_native_registration(slot.engine, kw)
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


class NativeCoreTransaction(unittest.TestCase):
    def make(self):
        module, _, core_type = NativeHookWiring().load_adapter()
        core = core_type()
        core.scheduler = module.IEEENativeAsyncScheduler(scheduler())
        uniproc = ModuleType('vllm.v1.executor.uniproc_executor')
        uniproc.UniProcExecutor = type('UniProcExecutor', (), {})
        core.model_executor = uniproc.UniProcExecutor()
        core.collective_rpc = Mock(return_value=[{'acquired': True, 'lease_id': 'gpu-test'}])
        return core, patch.dict(sys.modules, {uniproc.__name__: uniproc})

    def invoke(self, core, *, reference=None, desc=None):
        ref = command()['gpu_reference_intent'] if reference is None else reference
        desc = descriptor(adapter=ref.get('adapter_int_id')) if desc is None else desc
        return utility_call(core.ieee_pending_gpu_reference, 'pending-test', desc, ref)

    def test_single_serialized_command_registers_before_exact_cached_gpu_protection(self):
        core, modules = self.make()
        def protect(method, *, kwargs):
            self.assertEqual(method, 'ieee_gpu_reference')
            self.assertEqual(kwargs, {'operation': 'demand_load_and_acquire', **command()['gpu_reference_intent']})
            rows = core.scheduler._ieee_pending_admissions.snapshot()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['demand_owner'], 'controller_pending')
            return [{'acquired': True, 'lease_id': 'gpu-test'}]
        core.collective_rpc.side_effect = protect
        with modules:
            reply = self.invoke(core)
        self.assertEqual(reply['execution_scope'], 'single_native_core_turn_v1')
        self.assertFalse(reply['pending']['physical_kv_reservation'])
        core.collective_rpc.assert_called_once()

    def test_invalid_command_or_executor_cannot_register_pending_demand(self):
        for field, value in (('required_source_tier', 'host'), ('adapter_int_id', True),
                ('expected_epoch', 0), ('expected_source_id', ''), ('expected_content_sha256', 'bad'),
                ('extra', 1), ('lora_path', 'relative')):
            with self.subTest(field=field):
                core, modules = self.make()
                with modules, self.assertRaises(ValueError):
                    self.invoke(core, reference=command()['gpu_reference_intent'] | {field: value})
                self.assertFalse(core.scheduler._ieee_pending_admissions.entries)
                core.collective_rpc.assert_not_called()
        core, modules = self.make()
        core.model_executor = object()
        with modules, self.assertRaisesRegex(RuntimeError, 'UniProc'):
            self.invoke(core)
        self.assertFalse(core.scheduler._ieee_pending_admissions.entries)

    def test_invalid_pending_receipt_never_reaches_gpu_owner(self):
        for field, value in (('kind', 'wrong'), ('intent_id', 'wrong'), ('state', 'withdrawn'),
                ('clock_id', 'other'), ('physical_kv_reservation', True)):
            core, modules = self.make()
            core.scheduler._ieee_pending_admissions.register = Mock(
                return_value=pending_receipt('pending-test') | {field: value})
            with modules, self.assertRaisesRegex(ValueError, 'pending acknowledgement'):
                self.invoke(core)
            core.collective_rpc.assert_not_called()

    def test_known_source_conflict_returns_both_receipts_without_reload_or_retry(self):
        core, modules = self.make()
        conflict = dict(acquired=False, reason='required_source_changed')
        core.collective_rpc.return_value = [conflict]
        with modules:
            reply = self.invoke(core)
        self.assertIs(reply['reference'], conflict)
        self.assertEqual(len(core.scheduler._ieee_pending_admissions.snapshot()), 1)
        core.collective_rpc.assert_called_once()

    def test_worker_error_or_bad_reply_keeps_registered_demand_owned(self):
        for failure in ('exception', 'malformed'):
            core, modules = self.make()
            if failure == 'exception':
                core.collective_rpc.side_effect = RuntimeError('lost worker response')
            else:
                core.collective_rpc.return_value = []
            with modules, self.assertRaises(RuntimeError):
                self.invoke(core)
            self.assertEqual(len(core.scheduler._ieee_pending_admissions.snapshot()), 1)
            with modules, self.assertRaisesRegex(ValueError, 'already used'):
                self.invoke(core)
            core.collective_rpc.assert_called_once()


class NativeFrontendTransaction(unittest.IsolatedAsyncioTestCase):
    def make(self):
        core, modules = NativeCoreTransaction().make()
        front = frontend_type()()
        front.get_supported_tasks = AsyncMock(return_value=('generate',))
        front.input_processor = NS(process_inputs_async=AsyncMock(return_value=NS(
            prompt_token_ids=[1, 20, 30], sampling_params=NS(max_tokens=4))))
        async def call(method, *args):
            with modules:
                return utility_call(getattr(core, method), *args)
        front.engine_core.call_utility_async = AsyncMock(side_effect=call)
        return front, core

    async def register(self, front, ref=None):
        ref = command()['gpu_reference_intent'] if ref is None else ref
        return await front.ieee_register_pending('pending-test', 'canonical', NS(max_tokens=4),
            command()['gpu_reference_intent']['adapter_int_id'], gpu_reference_intent=ref)

    async def test_actual_frontend_and_core_use_one_utility_then_normal_withdrawal(self):
        front, core = self.make()
        result = await self.register(front)
        self.assertTrue(result['reference']['acquired'])
        self.assertEqual(front.ieee_pending['pending-test']['state'], 'pending')
        front.engine_core.call_utility_async.assert_awaited_once_with('ieee_pending_gpu_reference',
            'pending-test', descriptor(limit=4, adapter=command()['gpu_reference_intent']['adapter_int_id']),
            command()['gpu_reference_intent'])
        self.assertEqual((await front.ieee_close_pending('pending-test'))['state'], 'withdrawn')
        self.assertFalse(core.scheduler._ieee_pending_admissions.snapshot())

    async def test_cancellation_joins_registration_before_withdrawal_and_does_not_unpin(self):
        front, core = self.make()
        entered, proceed = asyncio.Event(), asyncio.Event()
        original = front.engine_core.call_utility_async.side_effect
        async def delayed(method, *args):
            result = await original(method, *args)
            if method == 'ieee_pending_gpu_reference':
                entered.set()
                await proceed.wait()
            return result
        front.engine_core.call_utility_async.side_effect = delayed
        task = asyncio.create_task(self.register(front))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        close = asyncio.create_task(front.ieee_close_pending('pending-test'))
        await asyncio.sleep(0)
        self.assertFalse(close.done())
        self.assertEqual(len(core.scheduler._ieee_pending_admissions.snapshot()), 1)
        proceed.set()
        self.assertEqual((await close)['state'], 'withdrawn')
        core.collective_rpc.assert_called_once()  # withdrawal cannot release a GPU pin

    async def test_command_is_copied_before_cancellable_tokenization(self):
        front, core = self.make()
        entered, proceed = asyncio.Event(), asyncio.Event()
        async def tokenize(*args, **kwargs):
            entered.set()
            await proceed.wait()
            return NS(prompt_token_ids=[1, 20, 30], sampling_params=NS(max_tokens=4))
        front.input_processor.process_inputs_async.side_effect = tokenize
        ref = command()['gpu_reference_intent']
        task = asyncio.create_task(self.register(front, ref))
        await entered.wait()
        ref['expected_source_id'] = 'mutated'
        proceed.set()
        await task
        self.assertEqual(core.collective_rpc.call_args.kwargs['kwargs']['expected_source_id'], 'copy-test')

    async def test_invalid_native_combined_ack_is_not_promoted_to_pending(self):
        for field, value in (('kind', 'wrong'), ('execution_scope', 'wrong'), ('reference', None),
                ('pending', None)):
            front, core = self.make()
            original = front.engine_core.call_utility_async.side_effect
            async def corrupt(method, *args):
                return (await original(method, *args)) | {field: value}
            front.engine_core.call_utility_async.side_effect = corrupt
            with self.assertRaises(ValueError):
                await self.register(front)
            self.assertEqual(front.ieee_pending['pending-test']['state'], 'registering')
            self.assertEqual(len(core.scheduler._ieee_pending_admissions.snapshot()), 1)
            core.collective_rpc.assert_called_once()


class WorkerTransport(unittest.IsolatedAsyncioTestCase):
    async def roundtrip(self, cancel_stage=None):
        ready = asyncio.get_running_loop().create_future()
        entered, proceed, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = []
        class FakeEngine:
            _lora_int_id = staticmethod(InferenceEngine._lora_int_id)
            ieee_register_pending_gpu_reference = InferenceEngine.ieee_register_pending_gpu_reference
            def __init__(self, cfg, *args):
                self.model_cfg = cfg | {'ieee_gpu_references': True}
            async def initialize(self):
                pass
            async def shutdown(self):
                pass
            async def ieee_register_pending(self, **kw):
                calls.append('pending')
                if cancel_stage == 'pending':
                    entered.set()
                    await proceed.wait()
                return await fake_native_registration(self, kw)
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

    async def test_capacity_witness_reply_reconciles_exact_lost_rpc(self):
        proxy = SubprocessInferenceEngineProxy.__new__(SubprocessInferenceEngineProxy)
        proxy._native_rpc_uncertain = {
            'lost-witness': {
                'cmd': 'ieee_admit_and_demand_load',
                'owner_id': 'owner-test', 'lease_id': 'gpu-test'}}
        proxy._rpc = AsyncMock(return_value={
            'request_lease_id': 'gpu-test', 'owner_id': 'owner-test',
            'acquired': False, 'reason': 'backend_load_conflict'})
        command = dict(lease_id='gpu-test', expected_owner_id='owner-test',
                       capacity_only=True)
        result = await proxy.ieee_admit_and_demand_load(**command)
        self.assertFalse(result['acquired'])
        self.assertFalse(proxy._native_rpc_uncertain)
        proxy._rpc.assert_awaited_once_with('ieee_admit_and_demand_load', **command)


if __name__ == '__main__':
    unittest.main()
