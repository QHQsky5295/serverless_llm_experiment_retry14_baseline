"""Cancellation-ID and owner boundary tests; no model, CUDA or performance claim."""
import asyncio
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

from faaslora.clock import local_monotonic_clock_id
from scripts.run_all_experiments import InferenceEngine, SubprocessInferenceEngineProxy
from tests.test_ieee_tc_request_lifecycle import native_reference_fixture


def frontend_type():
    class Native:
        def __init__(self):
            self.sent = []
            self.entered, self.release = asyncio.Event(), asyncio.Event()
            self.engine_core = NS(call_utility_async=AsyncMock())
            self.abort = AsyncMock()
        async def _add_request(self, request, *args):
            self.entered.set()
            await self.release.wait()
            self.sent.append(request.request_id)
    modules = {name: ModuleType(name) for name in ('vllm', 'vllm.v1.engine.async_llm')}
    modules['vllm'].__version__ = '0.30.0'
    modules['vllm.v1.engine.async_llm'].AsyncLLM = Native
    path = Path(__file__).parents[1] / 'faaslora/scheduling/vllm_ieee_frontend.py'
    spec = importlib.util.spec_from_file_location('_tested_ieee_frontend', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module.IEEENativeAsyncLLM


class NativeRetirement(unittest.TestCase):
    def test_cancel_joins_original_add_and_uses_exact_internal_id(self):
        async def check():
            frontend = frontend_type()()
            req = NS(external_req_id='external', request_id='random-native-id')
            task = asyncio.create_task(frontend._add_request(req, 'p', None, 0, object()))
            await frontend.entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertEqual(frontend.sent, [])
            frontend.release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            frontend.engine_core.call_utility_async.return_value = {
                'kind': 'ieee_native_request_retirement_v1', 'retired': True,
                'request_id': 'random-native-id', 'clock_id': local_monotonic_clock_id()}
            receipt = await frontend.ieee_retire_request('external', abort=True)
            self.assertEqual(frontend.sent, ['random-native-id'])
            self.assertEqual(receipt['external_request_id'], 'external')
            frontend.abort.assert_awaited_once_with('random-native-id', internal=True)
            frontend.engine_core.call_utility_async.assert_awaited_once_with(
                'ieee_request_retirement', 'random-native-id', True)
            with self.assertRaisesRegex(ValueError, 'unobserved'):
                await frontend.ieee_retire_request('not-submitted', abort=True)
        asyncio.run(check())

    def test_wrong_native_reply_cannot_release(self):
        async def check():
            frontend = frontend_type()()
            frontend.release.set()
            await frontend._add_request(NS(external_req_id='e', request_id='i'), 'p', None, 0, None)
            frontend.engine_core.call_utility_async.return_value = {
                'kind': 'ieee_native_request_retirement_v1', 'retired': True,
                'request_id': 'someone-else', 'clock_id': local_monotonic_clock_id()}
            with self.assertRaisesRegex(RuntimeError, 'identity/clock'):
                await frontend.ieee_retire_request('e', abort=True)
        asyncio.run(check())

    def test_engine_marks_end_use_only_after_matching_retirement(self):
        async def check():
            engine = InferenceEngine({'ieee_gpu_references': True}, {})
            engine._ieee_generation_refs['lease'] = {'owner_id': 'owner', 'backend_request_id': 'external'}
            engine.ieee_gpu_reference = AsyncMock()
            engine.engine = NS(ieee_retire_request=AsyncMock(return_value={
                'retired': True, 'external_request_id': 'wrong'}))
            ref = {'lease_id': 'lease', 'owner_id': 'owner'}
            with self.assertRaisesRegex(ValueError, 'bound request'):
                await engine.ieee_retire_generation(gpu_reference=ref, abort=True)
            engine.ieee_gpu_reference.assert_not_awaited()
            engine.engine.ieee_retire_request.return_value['external_request_id'] = 'external'
            receipt = await engine.ieee_retire_generation(gpu_reference=ref, abort=True)
            engine.ieee_gpu_reference.assert_awaited_once_with(operation='end_use', lease_id='lease',
                expected_owner_id='owner', backend_request_id='external')
            self.assertTrue(receipt['native_retirement']['retired'])
            await engine.ieee_retire_generation(gpu_reference=ref, abort=True)
            self.assertEqual(engine.ieee_gpu_reference.await_count, 1)
        asyncio.run(check())

    def test_controller_cancel_remains_failed_but_reconciles_its_ownership(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        async def check():
            entered = asyncio.Event()
            async def generate(**kwargs):
                ref = kwargs['gpu_reference']
                owner.begin_use(lease_id=ref['lease_id'], expected_owner_id=ref['owner_id'],
                    adapter_int_id=ref['adapter_int_id'], backend_request_id='exact',
                    lora_name=trace.adapter_id, lora_path='/existing/a')
                entered.set()
                await asyncio.Future()
            # Use the source identity actually bound by the controller fixture.
            async def retire(*, gpu_reference, abort):
                self.assertTrue(abort)
                owner.end_use(lease_id=gpu_reference['lease_id'],
                    expected_owner_id=gpu_reference['owner_id'], backend_request_id='exact')
                return {'gpu_reference_owner_id': gpu_reference['owner_id'],
                    'gpu_reference_lease_id': gpu_reference['lease_id'],
                    'native_retirement': {'retired': True}}
            slot.engine.generate_prepared.side_effect = generate
            slot.engine.ieee_retire_generation = AsyncMock(side_effect=retire)
            task = asyncio.create_task(runner._exec_request(trace, 4, 0., request_plan=plan))
            await asyncio.wait_for(entered.wait(), .5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(check())
        self.assertEqual(slot.active_requests, 0)
        self.assertEqual(owner.snapshot()['live_leases'], 0)
        self.assertFalse(runner._unsettled_runtime_reservations)

    def test_proxy_retirement_keeps_dispatch_identity(self):
        proxy = SubprocessInferenceEngineProxy.__new__(SubprocessInferenceEngineProxy)
        proxy._rpc = AsyncMock(return_value={'native_retirement': {'retired': True}})
        ref = {'lease_id': 'lease', 'owner_id': 'owner'}
        asyncio.run(proxy.ieee_retire_generation(gpu_reference=ref, abort=True))
        proxy._rpc.assert_awaited_once_with('ieee_retire_generation', gpu_reference=ref, abort=True)
