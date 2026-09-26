"""Cancellation-ID and owner boundary tests; no model, CUDA or performance claim."""
import asyncio
import importlib.util
from pathlib import Path
import sys
import json
import tempfile
import threading
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

    def test_concurrent_normal_and_abort_retirement_commit_end_use_once(self):
        async def check():
            engine = InferenceEngine({'ieee_gpu_references': True}, {})
            engine._ieee_generation_refs['lease'] = {'owner_id': 'owner', 'backend_request_id': 'external'}
            entered, finish = asyncio.Event(), asyncio.Event()
            async def retire(*args, **kwargs):
                entered.set()
                await finish.wait()
                return {'retired': True, 'external_request_id': 'external'}
            engine.engine = NS(ieee_retire_request=AsyncMock(side_effect=retire))
            engine.ieee_gpu_reference = AsyncMock()
            ref = {'owner_id': 'owner', 'lease_id': 'lease'}
            one = asyncio.create_task(engine.ieee_retire_generation(gpu_reference=ref, abort=False))
            await entered.wait()
            two = asyncio.create_task(engine.ieee_retire_generation(gpu_reference=ref, abort=True))
            await asyncio.sleep(0)
            self.assertEqual(engine.engine.ieee_retire_request.await_count, 1)
            finish.set()
            first, second = await asyncio.gather(one, two)
            self.assertEqual(first, second)
            self.assertEqual(engine.ieee_gpu_reference.await_count, 1)
        asyncio.run(check())

    def test_cancelled_channel_open_joins_and_closes_the_actual_socket(self):
        async def check():
            proxy = SubprocessInferenceEngineProxy.__new__(SubprocessInferenceEngineProxy)
            started, finish = threading.Event(), threading.Event()
            channel = object()
            def opening():
                started.set()
                finish.wait(2.)
                return channel
            proxy._open_blocking_rpc_channel = opening
            proxy._drop_rpc_channel = AsyncMock()
            task = asyncio.create_task(proxy._open_rpc_channel())
            self.assertTrue(await asyncio.to_thread(started.wait, 1.))
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            proxy._drop_rpc_channel.assert_awaited_once_with(channel)
        asyncio.run(check())

    def test_proxy_retirement_keeps_dispatch_identity(self):
        proxy = SubprocessInferenceEngineProxy.__new__(SubprocessInferenceEngineProxy)
        proxy._native_rpc_uncertain = {}
        proxy._rpc = AsyncMock(return_value={'native_retirement': {'retired': True},
            'gpu_reference_owner_id': 'owner', 'gpu_reference_lease_id': 'lease'})
        ref = {'lease_id': 'lease', 'owner_id': 'owner'}
        asyncio.run(proxy.ieee_retire_generation(gpu_reference=ref, abort=True))
        proxy._rpc.assert_awaited_once_with('ieee_retire_generation', gpu_reference=ref, abort=True)

    def test_actual_tcp_cancel_preserves_survivor_and_independent_control(self):
        # Actual TCP and blocking roundtrips; synthetic owner, not GPU proof.
        async def check(root):
            entered = {name: asyncio.Event() for name in ('a', 'b')}
            release = {name: asyncio.Event() for name in ('a', 'b')}
            submissions = []
            handlers = set()
            async def handle(reader, writer):
                handlers.add(asyncio.current_task())
                try:
                    while line := await reader.readline():
                        payload = json.loads(line)
                        cmd, kwargs = payload['cmd'], payload['kwargs']
                        ref = kwargs.get('gpu_reference', {})
                        if cmd == 'generate':
                            name = ref['lease_id']
                            submissions.append(name)
                            entered[name].set()
                            await release[name].wait()
                            result = {'survived': name}
                        elif cmd == 'ieee_retire_generation':
                            release[ref['lease_id']].set()
                            result = {'gpu_reference_owner_id': ref['owner_id'],
                                'gpu_reference_lease_id': ref['lease_id'],
                                'native_retirement': {'retired': True}}
                        else:
                            raise AssertionError(cmd)
                        writer.write((json.dumps({'ok': True, 'result': result})+'\n').encode())
                        await writer.drain()
                except (ConnectionError, asyncio.CancelledError):
                    pass
                finally:
                    writer.close()
                    await writer.wait_closed()
                    handlers.discard(asyncio.current_task())
            server = await asyncio.start_server(handle, '127.0.0.1', 0)
            port = server.sockets[0].getsockname()[1]
            proxy = SubprocessInferenceEngineProxy(process=NS(poll=lambda: None), host='127.0.0.1',
                port=port, model_cfg={'timing_contract': 'ieee_tc_native_v1'}, cost_model={},
                device_id=0, workdir=Path(root), log_path=Path(root)/'absent.log')
            proxy._rpc_pool_size = 2
            refs = {n: {'owner_id': 'owner', 'lease_id': n} for n in entered}
            tasks = {n: asyncio.create_task(proxy._rpc('generate', gpu_reference=refs[n])) for n in entered}
            try:
                await asyncio.wait_for(asyncio.gather(*(e.wait() for e in entered.values())), 2.)
                tasks['a'].cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await tasks['a']
                self.assertFalse(proxy._engine_dead)
                self.assertEqual(len(proxy._native_rpc_uncertain), 1)
                with self.assertRaisesRegex(RuntimeError, 'ownership unresolved'):
                    await proxy._rpc('generate', gpu_reference=refs['a'])
                await asyncio.wait_for(proxy.ieee_retire_generation(gpu_reference=refs['a'], abort=True), 2.)
                self.assertFalse(proxy._native_rpc_uncertain)
                self.assertFalse(tasks['b'].done())
                release['b'].set()
                self.assertEqual((await asyncio.wait_for(tasks['b'], 2.))['survived'], 'b')
                self.assertCountEqual(submissions, ['a', 'b'])
                # Lost pool position is replenishable, not a forever-empty queue.
                one = await asyncio.wait_for(proxy._acquire_rpc_channel(), 1.)
                two = await asyncio.wait_for(proxy._acquire_rpc_channel(), 1.)
                self.assertIsNot(one, two)
                await proxy._release_rpc_channel(one)
                await proxy._release_rpc_channel(two)
            finally:
                for e in release.values(): e.set()
                for task in tasks.values(): task.cancel()
                await asyncio.gather(*tasks.values(), return_exceptions=True)
                await proxy._close_all_rpc_channels()
                server.close()
                await server.wait_closed()
                await asyncio.gather(*list(handlers), return_exceptions=True)
        with tempfile.TemporaryDirectory() as root:
            asyncio.run(check(root))

    def test_wrong_or_other_retirement_cannot_clear_unknown_ownership(self):
        async def check():
            proxy = SubprocessInferenceEngineProxy.__new__(SubprocessInferenceEngineProxy)
            proxy._native_rpc_uncertain = {'a': {'cmd': 'generate', 'owner_id': 'o', 'lease_id': 'a'},
                'load': {'cmd': 'ieee_gpu_reference', 'owner_id': 'o', 'lease_id': 'a'}}
            proxy._rpc = AsyncMock(return_value={'gpu_reference_owner_id': 'o',
                'gpu_reference_lease_id': 'b', 'native_retirement': {'retired': True}})
            with self.assertRaisesRegex(ValueError, 'dispatch identity'):
                await proxy.ieee_retire_generation(gpu_reference={'owner_id': 'o', 'lease_id': 'a'}, abort=True)
            self.assertEqual(len(proxy._native_rpc_uncertain), 2)
            await proxy.ieee_retire_generation(gpu_reference={'owner_id': 'o', 'lease_id': 'b'}, abort=True)
            self.assertEqual(len(proxy._native_rpc_uncertain), 2)
            proxy._rpc.return_value['gpu_reference_lease_id'] = 'a'
            await proxy.ieee_retire_generation(gpu_reference={'owner_id': 'o', 'lease_id': 'a'}, abort=True)
            self.assertEqual(list(proxy._native_rpc_uncertain), ['load'])
        asyncio.run(check())
