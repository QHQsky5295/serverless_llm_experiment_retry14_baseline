"""Real native input method + synthetic owners; no GPU/performance qualification."""
import ast
import asyncio
from concurrent.futures import Future
from enum import Enum
import logging
from pathlib import Path
import queue
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

from faaslora.scheduling.core_interleave import (
    AtomicControlInterleave, POLICY, HOST_CONSTRUCTION_POLICY, _NATIVE_INPUT_AST_SHA,
    install_core_control_interleave, native_input_fingerprint,
)

NATIVE = Path('/home/qhq/.venvs/primelora_vllm0300_tc_20260925/lib/python3.12/site-packages/vllm/v1/engine/core.py')


class Kind(Enum):
    ADD=1; ABORT=2; UTILITY=3; WAKEUP=4; EXECUTOR_FAILED=5


def utility(identity, method='ieee_prepare_host', args=()):
    return Kind.UTILITY, [0, identity, method, list(args)]


def native_input():
    tree=ast.parse(NATIVE.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EngineCoreProc')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_process_input_queue')
    namespace=dict(queue=queue,logger=logging.getLogger(__name__),DEBUG=logging.DEBUG)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(NATIVE),'exec'),namespace)
    return namespace['_process_input_queue']


def fixture(*, enabled=True, host=False, work=True, requests=()):
    config=NS(additional_config={'ieee_tc_control_interleave':POLICY} if enabled else {},
              model_config=NS(is_moe=False))
    if host:
        config.additional_config['ieee_tc_host_construction'] = HOST_CONSTRUCTION_POLICY
    scheduler=NS(vllm_config=config,parallel_config=NS(tensor_parallel_size=1,
        pipeline_parallel_size=1,data_parallel_size=1))
    class Executor: pass
    class Core:
        _process_input_queue=native_input()
        def __init__(self):
            self.scheduler=scheduler;self.model_executor=Executor()
            self.input_queue=queue.Queue();self.aborts_queue=queue.Queue()
            self.work=work;self.running=True;self.process_input_queue_block=False
            self.batch_work=False;self.calls=[];self.callbacks=0
            self.utility_hook=None
            for request in requests:self.input_queue.put(request)
        def has_work(self):return self.work or self.batch_work
        def is_running(self):return self.running
        def shutdown(self):self.calls.append(('shutdown', None))
        def _notify_idle_state_callbacks(self):self.callbacks+=1
        def _handle_client_request(self,kind,payload):
            self.calls.append((kind,payload[1] if kind is Kind.UTILITY else payload))
            if kind is Kind.ADD:self.work=True
            if kind is Kind.ABORT:self.work=False
            if kind is Kind.EXECUTOR_FAILED:raise RuntimeError('executor failed')
            if kind is Kind.UTILITY and self.utility_hook:self.utility_hook(payload)
        def _process_engine_step(self):
            self.calls.append(('step',None))
            # Native abort processing remains owned by step, not this adapter.
            while not self.aborts_queue.empty():self.aborts_queue.get_nowait();self.work=False
    state=install_core_control_interleave(Core,scheduler,executor_type=Executor,
        request_types=Kind,native_version='0.30.0')
    return Core(),state,Core,scheduler,Executor


class AtomicControlFairness(unittest.TestCase):
    def test_d872_native_future_callback_does_not_return_to_core_thread(self):
        # Execute the installed native utility method, just as native_input()
        # above extracts the installed input loop. No GPU/model initialization.
        from typing import Callable
        tree = ast.parse(NATIVE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'EngineCoreProc')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == '_invoke_utility_method')
        method.decorator_list = []
        native = NS()
        namespace = dict(Future=Future, Callable=Callable, UtilityOutput=object,
                         UtilityResult=lambda value: value, EngineCoreProc=native,
                         logger=logging.getLogger(__name__))
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(NATIVE), 'exec'), namespace)
        native._invoke_utility_method = namespace['_invoke_utility_method']
        result = Future()
        output = NS(result=None)
        replies = []
        core_thread = threading.get_ident()
        native._invoke_utility_method('build', lambda: result, output,
            lambda value: replies.append((threading.get_ident(), value.result)))
        self.assertEqual(replies, [])
        worker = threading.Thread(target=lambda: result.set_result('private_cpu_object'))
        worker.start()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0][1], 'private_cpu_object')
        self.assertNotEqual(replies[0][0], core_thread)

    def test_native_source_and_disabled_path_are_exact(self):
        self.assertEqual(native_input_fingerprint(native_input()),_NATIVE_INPUT_AST_SHA)
        core,state,_,_,_=fixture(enabled=False,requests=[utility(1),utility(2)])
        self.assertIsNone(state)
        core._process_input_queue()
        self.assertEqual(core.calls,[(Kind.UTILITY,1),(Kind.UTILITY,2)])

    def test_work_rounds_preserve_fifo_and_original_step(self):
        core,state,_,_,_=fixture(requests=[utility(i) for i in range(3)])
        for i in range(3):
            core._process_input_queue()
            self.assertEqual(core.input_queue.qsize(),2-i)
            self.assertEqual(len([c for c in core.calls if c[0]=='step']),i)
            core._process_engine_step()
        self.assertEqual(core.calls,[(Kind.UTILITY,0),('step',None),
            (Kind.UTILITY,1),('step',None),(Kind.UTILITY,2),('step',None)])
        self.assertEqual(state.snapshot(),dict(kind='ieee_native_control_interleave_v1',
            policy=POLICY,input_turns=3,handled_controls=3,
            yields_to_native_loop=3,yields_with_queued_input=2))

    def test_idle_drains_and_add_transitions_to_work_without_losing_input(self):
        core,state,_,_,_=fixture(work=False,requests=[utility(1),utility(2)])
        core._process_input_queue()
        self.assertEqual(core.calls,[(Kind.UTILITY,1),(Kind.UTILITY,2)])
        self.assertEqual(state.yields,0)
        core.input_queue.put((Kind.ADD,'request'))
        core.input_queue.put(utility(3));core.input_queue.put(utility(4))
        core._process_input_queue()
        self.assertEqual(core.calls[-2:],[(Kind.ADD,'request'),(Kind.UTILITY,3)])
        self.assertEqual(core.input_queue.qsize(),1)
        self.assertEqual(state.yields,1)

    def test_idle_callback_and_empty_abort_queue_match_native(self):
        for enabled in (False,True):
            core,_,_,_,_=fixture(enabled=enabled,work=False)
            core.aborts_queue.put(['old-abort'])
            core._process_input_queue()
            self.assertEqual(core.callbacks,1)
            self.assertTrue(core.aborts_queue.empty())
            self.assertEqual(core.calls,[])

    def test_blocking_idle_wakeup_keeps_native_wait_contract(self):
        core,_,_,_,_=fixture(work=False,requests=[utility(1),(Kind.ADD,'r'),utility(2)])
        core.process_input_queue_block=True
        core._process_input_queue()
        self.assertEqual(core.calls,[(Kind.UTILITY,1),(Kind.ADD,'r'),(Kind.UTILITY,2)])
        self.assertTrue(core.work)

    def test_atomic_utility_is_never_split_and_exception_identity_survives(self):
        core,state,_,_,_=fixture(requests=[utility(1),utility(2)])
        transitions=[]
        def transaction(_):
            transitions.extend(['pending_registered','reference_acquired'])
            self.assertNotIn(('step',None),core.calls)
        core.utility_hook=transaction;core._process_input_queue()
        self.assertEqual(transitions,['pending_registered','reference_acquired'])
        failure=ValueError('atomic error')
        def fail(_):raise failure
        core.utility_hook=fail
        with self.assertRaises(ValueError) as caught:core._process_input_queue()
        self.assertIs(caught.exception,failure)
        self.assertEqual(state.controls,1)

    def test_real_pending_gpu_bridge_remains_one_control_transaction(self):
        from tests.test_ieee_tc_pending_gpu_pipeline import NativeCoreTransaction
        real=NativeCoreTransaction()
        backend,modules=real.make()
        core,state,_,_,_=fixture(requests=[utility(1,'ieee_pending_gpu_reference'),utility(2)])
        receipts=[]
        def protect(method, *, kwargs):
            self.assertNotIn(('step',None),core.calls)
            self.assertEqual(len(backend.scheduler._ieee_pending_admissions.snapshot()),1)
            return [{'acquired':True,'lease_id':'gpu-test'}]
        backend.collective_rpc.side_effect=protect
        def transaction(_):receipts.append(real.invoke(backend))
        core.utility_hook=transaction
        with modules:core._process_input_queue()
        self.assertEqual(state.yields,1)
        self.assertEqual(receipts[0]['execution_scope'],'single_native_core_turn_v1')
        self.assertEqual(core.input_queue.qsize(),1)

    def test_nonprime_utility_is_not_reordered_or_counted(self):
        core,state,_,_,_=fixture(requests=[utility(1,'check_health'),
            (Kind.ADD,'r'),utility(2,'collective_rpc',['other']),
            utility(3,'collective_rpc',['ieee_gpu_reference']),utility(4)])
        core._process_input_queue()
        self.assertEqual(core.calls,[(Kind.UTILITY,1),(Kind.ADD,'r'),(Kind.UTILITY,2),(Kind.UTILITY,3)])
        self.assertEqual(state.controls,1)
        self.assertEqual(core.input_queue.qsize(),1)

    def test_abort_and_fault_are_handled_by_native_paths(self):
        core,state,_,_,_=fixture(requests=[(Kind.ABORT,'r'),utility(1),utility(2)])
        core._process_input_queue()
        self.assertEqual(state.yields,0)
        self.assertEqual(core.calls,[(Kind.ABORT,'r'),(Kind.UTILITY,1),(Kind.UTILITY,2)])
        core.work=True;core.aborts_queue.put(['r'])
        core.input_queue.put(utility(3));core.input_queue.put((Kind.ABORT,'r'))
        core._process_input_queue();core._process_engine_step();core._process_input_queue()
        self.assertFalse(core.work)
        self.assertIn((Kind.ABORT,'r'),core.calls)
        core.input_queue.put((Kind.EXECUTOR_FAILED,None))
        with self.assertRaisesRegex(RuntimeError,'executor failed'):core._process_input_queue()

    def test_shutdown_does_not_add_fairness_yields(self):
        core,state,_,_,_=fixture(requests=[utility(1),utility(2)])
        core.running=False
        core._process_input_queue()
        self.assertEqual(len(core.calls),2)
        self.assertEqual(state.yields,0)

    def test_batch_inflight_is_work_and_future_completion_is_not_faked(self):
        core,state,_,_,_=fixture(work=False,requests=[utility(1,'ieee_request_retirement'),utility(2)])
        core.batch_work=True
        pending=Future();replies=[]
        def hook(_):pending.add_done_callback(lambda f:replies.append(f.result()))
        core.utility_hook=hook;core._process_input_queue()
        self.assertEqual(state.yields,1);self.assertEqual(replies,[])
        pending.set_result('retired')
        self.assertEqual(replies,['retired'])

    def test_owner_collision_and_unqualified_executor_fail(self):
        core,state,cls,scheduler,executor=fixture()
        self.assertIs(install_core_control_interleave(cls,scheduler,executor_type=executor,
            request_types=Kind,native_version='0.30.0'),state)
        core.model_executor=object()
        with self.assertRaisesRegex(RuntimeError,'single native owner'):core._process_input_queue()
        core.model_executor=executor();core._process_input_queue()
        other=cls()
        with self.assertRaisesRegex(RuntimeError,'single native owner'):other._process_input_queue()
        failures=[]
        def foreign_thread():
            try:core._process_input_queue()
            except RuntimeError as e:failures.append(str(e))
        thread=threading.Thread(target=foreign_thread);thread.start();thread.join()
        self.assertEqual(len(failures),1)

    def test_changed_source_version_policy_and_topology_reject(self):
        for case in ('version','policy','dp','moe','source'):
            core,_,cls,scheduler,executor=fixture(enabled=False)
            scheduler.vllm_config.additional_config['ieee_tc_control_interleave']=POLICY
            version='0.30.0'
            if case=='version':version='0.10.2'
            if case=='policy':scheduler.vllm_config.additional_config['ieee_tc_control_interleave']='unknown'
            if case=='dp':scheduler.parallel_config.data_parallel_size=2
            if case=='moe':scheduler.vllm_config.model_config.is_moe=True
            if case=='source':cls._process_input_queue=lambda self:None
            before=cls._process_input_queue
            with self.subTest(case=case),self.assertRaises(ValueError):
                install_core_control_interleave(cls,scheduler,executor_type=executor,
                    request_types=Kind,native_version=version)
            self.assertIs(cls._process_input_queue,before)

    def test_configuration_reaches_backend_and_rejects_unowned_modes(self):
        from scripts import run_all_experiments as runner
        cfg=dict(ieee_scheduler_observation=True,ieee_input_upper_bounds=[20,100],
                 ieee_gpu_references=True,ieee_control_interleave=POLICY)
        native=object();args_mock=Mock(side_effect=lambda **kw:NS(**kw))
        frontend=ModuleType('faaslora.scheduling.vllm_ieee_frontend')
        frontend.IEEENativeAsyncLLM=NS(from_engine_args=Mock(return_value=native))
        for case in ('valid','missing_references','missing_scheduler','wrong_policy',
                     'host','host_no_workspace','host_no_interleave','host_wrong_policy','host_no_lora'):
            current=dict(cfg)
            if case=='missing_references':current['ieee_gpu_references']=False
            if case=='missing_scheduler':current['ieee_scheduler_observation']=False
            if case=='wrong_policy':current['ieee_control_interleave']='x'
            if case.startswith('host'):
                current.update(ieee_host_construction=HOST_CONSTRUCTION_POLICY,
                               ieee_native_host_workspace={'kind': 'native_host_workspace_contract_v1'})
            if case=='host_no_workspace':current.pop('ieee_native_host_workspace')
            if case=='host_no_interleave':current.pop('ieee_control_interleave')
            if case=='host_wrong_policy':current['ieee_host_construction']='wrong'
            engine=runner.InferenceEngine(current,{})
            engine._resolve_vllm_visible_devices=Mock(return_value='0')
            engine._resolve_vllm_executor_backend=Mock(return_value=None)
            with patch.object(runner,'_lazy_import_vllm',return_value=True), \
                 patch.object(runner,'AsyncEngineArgs',args_mock), \
                 patch.dict('sys.modules',{'faaslora.scheduling.vllm_ieee_frontend':frontend}):
                if case in ('valid','host'):
                    actual=asyncio.run(engine._try_create_engine('local',1,.8,1024,True,True,2,16))
                    self.assertIs(actual,native)
                    self.assertEqual(args_mock.call_args.kwargs['additional_config']['ieee_tc_control_interleave'],POLICY)
                    self.assertEqual(args_mock.call_args.kwargs['additional_config'].get('ieee_tc_host_construction'),
                                     HOST_CONSTRUCTION_POLICY if case=='host' else None)
                else:
                    with self.assertRaisesRegex(RuntimeError,'control interleave|HOST construction'):
                        asyncio.run(engine._try_create_engine('local',1,.8,1024,True,case!='host_no_lora',2,16))


class PrivateHostCoreBridge(unittest.TestCase):
    """Real reference owner + worker bridge + installed native input/Future code."""

    def setUp(self):
        from tests.test_ieee_tc_gpu_references import PrivateNativeHostConstruction
        from faaslora.memory.gpu_monitor import IEEEWorkerObservationExtension
        self.f = PrivateNativeHostConstruction()
        self.f.setUp()
        self.core, self.state, self.cls, scheduler, _ = fixture(host=True)
        self.worker = NS(vllm_config=scheduler.vllm_config,
            model_runner=NS(lora_manager=NS(_adapter_manager=self.f.manager)),
            _ieee_gpu_reference_owner=self.f.owner, rank=0,
            _ieee_host_allocator_policy={'verified': True})
        self.rpc_calls = []
        def rpc(method, *, args):
            self.rpc_calls.append((method, threading.get_ident()))
            return [getattr(IEEEWorkerObservationExtension, method)(self.worker, *args)]
        self.core.collective_rpc = rpc

    def tearDown(self):
        self.f.finish.set()
        build = self.f.owner._native_host_build
        if build is not None and build._thread is not None and build._thread.ident is not None:
            build._thread.join(5)

    def begin(self):
        result = self.core.ieee_prepare_file_host(self.f.command())
        self.assertTrue(self.f.entered.wait(5))
        return result, self.state.host_construction.active[0]

    def finish(self, build):
        self.f.finish.set()
        self.assertTrue(build.done.wait(5))
        build._thread.join(5)  # Enqueue finished; commit still belongs to core.
        self.core._process_input_queue()

    def test_private_work_yields_native_steps_and_commit_keeps_fifo(self):
        command = self.f.command()
        replies = []
        self.core.utility_hook = lambda payload: replies.append(
            self.core.ieee_prepare_file_host(payload[3][0]))
        self.core.input_queue.put(utility(1, 'ieee_prepare_file_host', [command]))
        self.core.input_queue.put(utility(2, 'check_health'))
        self.core._process_input_queue()
        future = replies[0]
        build = self.state.host_construction.active[0]
        self.assertTrue(self.f.entered.wait(5))
        self.assertFalse(future.done())
        self.core._process_engine_step()
        self.assertFalse(self.f.owner.source_snapshot()['sources'])
        self.assertEqual(self.f.owner.host_construction_reserved_bytes(), 300)
        self.core.utility_hook = None
        callbacks = []
        future.add_done_callback(lambda value: callbacks.append(threading.get_ident()))
        self.finish(build)
        self.assertTrue(future.result()['held'])
        self.assertEqual(callbacks, [threading.get_ident()])
        self.assertEqual(self.core.calls, [(Kind.UTILITY, 1), ('step', None), (Kind.UTILITY, 2)])
        self.assertEqual(self.state.yields, 2)
        self.assertTrue(all(tid == threading.get_ident() for _, tid in self.rpc_calls))
        self.assertNotEqual(self.f.loader_threads[0], threading.get_ident())
        self.assertEqual(self.state.host_construction.snapshot()['settled'], 1)
        self.assertIn('clock_id', future.result())
        self.assertIsNone(build.model)

    def test_native_utility_reply_runs_on_core_not_constructor(self):
        from typing import Callable
        tree = ast.parse(NATIVE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EngineCoreProc')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_invoke_utility_method')
        method.decorator_list = []
        native = NS()
        namespace = dict(Future=Future, Callable=Callable, UtilityOutput=object,
            UtilityResult=lambda value: value, EngineCoreProc=native, logger=logging.getLogger(__name__))
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(NATIVE), 'exec'), namespace)
        native._invoke_utility_method = namespace['_invoke_utility_method']
        output, replies = NS(result=None), []
        native._invoke_utility_method('host', lambda: self.core.ieee_prepare_file_host(self.f.command()), output,
            lambda value: replies.append((threading.get_ident(), value.result)))
        build = self.state.host_construction.active[0]
        self.assertTrue(self.f.entered.wait(5))
        self.assertEqual(replies, [])
        self.finish(build)
        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0][0], threading.get_ident())
        self.assertTrue(replies[0][1]['held'])

    def test_pending_duplicate_and_cancel_cannot_release_work(self):
        command = self.f.command()
        future = self.core.ieee_prepare_file_host(command)
        build = self.state.host_construction.active[0]
        self.assertFalse(future.cancel())
        self.assertIs(self.core.ieee_prepare_file_host(command), future)
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.core.ieee_prepare_file_host({**command, 'lora_path': '/changed'})
        other = self.core.ieee_prepare_file_host(self.f.command(lease='other', aid=5))
        self.assertEqual(other['reason'], 'native_host_construction_busy')
        self.assertEqual(self.state.host_construction.started, 1)
        self.finish(build)
        self.assertTrue(future.result()['held'])

    def test_idle_completion_is_wakeup_not_fake_model_work(self):
        future, build = self.begin()
        self.core.work = False
        self.finish(build)
        self.assertFalse(self.core.has_work())
        self.assertTrue(future.result()['held'])
        self.assertEqual(self.state.yields, 0)
        self.assertFalse(self.core.calls)
        # Ordinary native wakeups remain ordinary dispatches.
        self.core.input_queue.put((Kind.WAKEUP, None))
        self.core._process_input_queue()
        self.assertEqual(self.core.calls, [(Kind.WAKEUP, None)])

    def test_shutdown_joins_and_discards_before_native_worker_destruction(self):
        future, build = self.begin()
        join = build.join_for_shutdown
        def release_at_join():
            self.assertFalse(future.done())
            self.assertNotIn(('shutdown', None), self.core.calls)
            self.f.finish.set()
            join()
        self.core.running = False
        with patch.object(build, 'join_for_shutdown', side_effect=release_at_join):
            self.core.shutdown()
        self.assertEqual(self.core.calls, [('shutdown', None)])
        self.assertEqual(future.result()['reason'], 'construction_cancelled')
        self.assertTrue(future.result()['private_construction']['discarded'])
        self.assertFalse(self.f.manager._registered_adapters)
        self.assertIsNone(build.model)
        self.core._process_input_queue()  # Already-settled wake does not settle twice.
        self.assertEqual(self.state.host_construction.settled, 1)
        with self.assertRaisesRegex(RuntimeError, 'shutting down'):
            self.core.ieee_prepare_file_host(self.f.command(lease='later'))

    def test_construct_failure_is_error_not_ready_or_released(self):
        self.f.owner.file_host_constructor = Mock(side_effect=RuntimeError('private failure'))
        future = self.core.ieee_prepare_file_host(self.f.command())
        build = self.state.host_construction.active[0]
        self.finish(build)
        with self.assertRaisesRegex(RuntimeError, 'private failure'):
            future.result()
        self.assertTrue(self.f.owner._poisoned)
        self.assertIs(self.f.owner._native_host_build, build)
        self.assertFalse(build.settled)
        self.assertFalse(self.f.manager._registered_adapters)
        self.assertEqual(self.state.host_construction.failed, 1)
        self.core.shutdown()
        self.assertEqual(self.core.calls, [('shutdown', None)])

    def test_thread_start_failure_is_settled_as_error_without_joining_unstarted_thread(self):
        with patch('threading.Thread.start', side_effect=RuntimeError('start failure')):
            future = self.core.ieee_prepare_file_host(self.f.command())
        self.core._process_input_queue()
        with self.assertRaisesRegex(RuntimeError, 'start failure'):
            future.result()
        self.assertTrue(self.f.owner._poisoned)
        self.assertFalse(self.f.manager._registered_adapters)

    def test_policy_requires_interleave_and_cannot_change_after_install(self):
        with self.assertRaisesRegex(ValueError, 'requires native control interleave'):
            fixture(enabled=False, host=True)
        self.core.scheduler.vllm_config.additional_config.pop('ieee_tc_host_construction')
        with self.assertRaisesRegex(ValueError, 'changed after installation'):
            install_core_control_interleave(self.cls, self.core.scheduler,
                executor_type=type(self.core.model_executor), request_types=Kind, native_version='0.30.0')


class HostPreparationFrontend(unittest.IsolatedAsyncioTestCase):
    async def test_optin_routes_only_cpu_preparation_and_validates_terminal_reply(self):
        from scripts.run_all_experiments import InferenceEngine
        utility_call = AsyncMock(return_value={'held': True})
        worker_call = AsyncMock(return_value=[{'original': True}])
        engine = NS(model_cfg={'ieee_gpu_references': True, 'ieee_host_construction': HOST_CONSTRUCTION_POLICY},
            backend='vllm', _engine_dead=False,
            engine=NS(engine_core=NS(call_utility_async=utility_call), collective_rpc=worker_call))
        result = await InferenceEngine.ieee_gpu_reference(engine, operation='prepare_file_host_and_hold', lease_id='l')
        self.assertEqual(result, {'held': True})
        utility_call.assert_awaited_once_with('ieee_prepare_file_host', {'lease_id': 'l'})
        worker_call.assert_not_awaited()
        self.assertEqual(await InferenceEngine.ieee_gpu_reference(engine, operation='source_snapshot'), {'original': True})
        worker_call.assert_awaited_once_with('ieee_gpu_reference', kwargs={'operation': 'source_snapshot'})
        utility_call.return_value = ['not a core receipt']
        with self.assertRaisesRegex(RuntimeError, 'core-owned HOST'):
            await InferenceEngine.ieee_gpu_reference(engine, operation='prepare_file_host_and_hold')
        engine.model_cfg.pop('ieee_host_construction')
        self.assertEqual(await InferenceEngine.ieee_gpu_reference(engine, operation='prepare_file_host_and_hold'),
                         {'original': True})


if __name__=='__main__':unittest.main()
