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
from unittest.mock import Mock, patch

from faaslora.scheduling.core_interleave import (
    AtomicControlInterleave, POLICY, _NATIVE_INPUT_AST_SHA,
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


def fixture(*, enabled=True, work=True, requests=()):
    config=NS(additional_config={'ieee_tc_control_interleave':POLICY} if enabled else {},
              model_config=NS(is_moe=False))
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
        for case in ('valid','missing_references','missing_scheduler','wrong_policy'):
            current=dict(cfg)
            if case=='missing_references':current['ieee_gpu_references']=False
            if case=='missing_scheduler':current['ieee_scheduler_observation']=False
            if case=='wrong_policy':current['ieee_control_interleave']='x'
            engine=runner.InferenceEngine(current,{})
            engine._resolve_vllm_visible_devices=Mock(return_value='0')
            engine._resolve_vllm_executor_backend=Mock(return_value=None)
            with patch.object(runner,'_lazy_import_vllm',return_value=True), \
                 patch.object(runner,'AsyncEngineArgs',args_mock), \
                 patch.dict('sys.modules',{'faaslora.scheduling.vllm_ieee_frontend':frontend}):
                if case=='valid':
                    actual=asyncio.run(engine._try_create_engine('local',1,.8,1024,True,True,2,16))
                    self.assertIs(actual,native)
                    self.assertEqual(args_mock.call_args.kwargs['additional_config']['ieee_tc_control_interleave'],POLICY)
                else:
                    with self.assertRaisesRegex(RuntimeError,'control interleave'):
                        asyncio.run(engine._try_create_engine('local',1,.8,1024,True,True,2,16))


if __name__=='__main__':unittest.main()
