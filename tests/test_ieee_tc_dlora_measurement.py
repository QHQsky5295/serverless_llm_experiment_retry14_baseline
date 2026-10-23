"""CPU-only execution of retained native methods; no Ray/GPU inference claim."""
import ast
import asyncio
import copy
import importlib.util
import logging
import json
import os
from pathlib import Path
import sys
import tempfile
import math
import time
from enum import Enum
from types import MethodType
from unittest.mock import Mock
from types import SimpleNamespace as NS
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
NATIVE = ROOT/'vendor_new_baselines/dLoRA_artifact_main_20260519'
sys.path.insert(0, str(MAIN))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load('tc_dlora_measurement', ROOT/'scripts/ieee_tc_serverless_measurement.py')
p = load('tc_dlora_prepare', ROOT/'scripts/prepare_ieee_tc_serverless_stack.py')
REPLACEMENTS = p.dlora_measurement_sources(NATIVE)


class Sequence:
    def __init__(self):
        self.data = NS(prompt_token_ids=[1, 2])
        self.ids, self.seq_id = [], 17
        self.output_logprobs, self.output_text = {}, ''
        self.prompt, self.status = 'prompt', 'running'

    def get_output_token_ids(self):
        return self.ids

    def get_cumulative_logprob(self):
        return 0.


class Group:
    def __init__(self, target=3):
        self.request_id, self.model_id = 'request-1', 7
        self.seqs = [Sequence()]
        self.sampling_params = NS(n=1, best_of=1, temperature=0, top_p=1,
            use_beam_search=False, ignore_eos=True, stop=[], max_tokens=target, logprobs=None)

    def get_seqs(self):
        return self.seqs

    def is_finished(self):
        return self.seqs[0].status == 'length'


def native_nodes(source, names, namespace):
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<native-method-test>', 'exec'), namespace)
    return namespace


OUTPUT_NS = native_nodes(REPLACEMENTS['vllm/outputs.py'],
    {'CompletionOutput', 'RequestOutput'}, dict(Dict=Dict, List=List, Optional=Optional,
    SequenceGroup=Group, SequenceStatus=NS(get_finished_reason=lambda status:
        'length' if status == 'length' else None), dlora_output_snapshot=m.dlora_output_snapshot))


def initialized(target=3):
    group = Group(target)
    with patch.object(m.time, 'perf_counter', return_value=10.):
        m.dlora_register_sequence_group(group)
    with patch.object(m.time, 'perf_counter', return_value=11.):
        m.dlora_scheduled([group], 0)
    return group


def token(group, token_id, at, engine=0):
    group.seqs[0].ids.append(token_id)
    m.dlora_tokens_available([group], engine, at)


def completed(target=3):
    group = initialized(target)
    for i in range(target):
        token(group, 100+i, 12.+i*.25)
    group.seqs[0].status = 'length'
    return group, OUTPUT_NS['RequestOutput'].from_seq_group(group)


def validated(output, **changes):
    params = dict(request_id='request-1', model_id=7, prompt_ids=[1, 2],
        target=3, dispatched_at=9., completed_at=20.,
        clock_id=output.tc_dlora_native['native_clock_id'])
    params.update(changes)
    return m.dlora_validate_final_output(output, **params)


class DloraNativeTimingTests(unittest.TestCase):
    def test_native_events_not_final_http_time(self):
        group, output = completed()
        result = validated(output)
        self.assertEqual(result['native_first_token_monotonic_s'], 12.)
        self.assertEqual(result['native_last_token_monotonic_s'], 12.5)
        self.assertEqual(result['native_tpot_ms'], 250.)
        self.assertEqual(result['worker_completion_notification_ms'], 7500.)
        self.assertEqual(result['completion_token_ids'], [100, 101, 102])
        self.assertFalse(result['lora_numerical_correctness_qualified'])
        self.assertEqual(group.seqs[0].ids, [100, 101, 102])

    def test_single_token_has_no_tpot(self):
        _, output = completed(1)
        self.assertIsNone(validated(output, target=1)['native_tpot_ms'])

    def test_migration_deepcopy_preserves_first_boundary_and_native_insert(self):
        group = initialized()
        token(group, 100, 12.)
        migrated = copy.deepcopy(group)  # Existing EngineManager's carrier.
        engine_class = next(n for n in ast.parse(REPLACEMENTS['vllm/engine/llm_engine.py']).body
                            if isinstance(n, ast.ClassDef) and n.name == 'LLMEngine')
        method = next(n for n in engine_class.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'insert_seq_groups')
        ns = dict(List=List, SequenceGroup=Group)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<native-insert>', 'exec'), ns)
        inserted = []
        owner = NS(seq_counter=iter([99]), scheduler=NS(insert_seq_group=inserted.append))
        self.assertEqual(ns['insert_seq_groups'](owner, [migrated]), [migrated])
        self.assertEqual(migrated.seqs[0].seq_id, 99)
        with patch.object(m.time, 'perf_counter', return_value=13.):
            m.dlora_scheduled([migrated], 1)
        token(migrated, 101, 14., engine=1)
        token(migrated, 102, 15., engine=1)
        migrated.seqs[0].status = 'length'
        result = validated(OUTPUT_NS['RequestOutput'].from_seq_group(migrated))
        self.assertEqual(result['native_first_token_monotonic_s'], 12.)
        self.assertEqual(result['native_scheduled_monotonic_s'], 11.)
        self.assertEqual(result['native_engine_path'], [0, 1])
        self.assertEqual(result['native_tpot_ms'], 1500.)  # Includes migration gap.
        self.assertEqual(group._tc_dlora['token_ids'], [100])
        self.assertIsNot(group._tc_dlora, migrated._tc_dlora)

    def test_snapshots_do_not_alias_future_native_updates(self):
        group = initialized()
        token(group, 100, 12.)
        early = OUTPUT_NS['RequestOutput'].from_seq_group(group)
        token(group, 101, 13.)
        self.assertEqual(early.tc_dlora_native['token_ids'], [100])
        self.assertEqual(early.tc_dlora_native['last_at'], 12.)

    def test_reject_duplicate_missing_or_rewritten_tokens(self):
        for kind in ('duplicate', 'skip', 'rewrite', 'negative', 'backward', 'wrong_owner'):
            group = initialized()
            token(group, 100, 12.)
            at, owner = 13., 0
            if kind == 'skip': group.seqs[0].ids.extend([101, 102])
            elif kind == 'rewrite': group.seqs[0].ids[:] = [200, 101]
            elif kind == 'negative': group.seqs[0].ids.append(-1)
            elif kind == 'backward':
                group.seqs[0].ids.append(101)
                at = 11.5
            elif kind == 'wrong_owner':
                group.seqs[0].ids.append(101)
                owner = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                m.dlora_tokens_available([group], owner, at)

    def test_reject_binding_clock_count_and_completion_changes(self):
        for key, value in (('request_id', 'different'), ('model_id', 8),
                           ('prompt_ids', [1, 3]), ('target', 2),
                           ('clock_id', 'different-host'), ('completed_at', 11.),
                           ('dispatched_at', 11.)):
            _, output = completed()
            with self.subTest(key=key), self.assertRaises(ValueError):
                validated(output, **{key: value})
        for field, value in (('first_at', None), ('last_at', float('nan'))):
            _, output = completed()
            output.tc_dlora_native[field] = value
            with self.assertRaises(ValueError): validated(output)

    def test_invalid_generation_never_qualified(self):
        for key, value in (('n', 2), ('best_of', 2), ('temperature', 1.),
                           ('top_p', .5), ('use_beam_search', True),
                           ('ignore_eos', False), ('stop', ['stop']), ('max_tokens', 0)):
            group = Group()
            setattr(group.sampling_params, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                m.dlora_register_sequence_group(group)

    def test_no_output_ignored_request_cannot_be_success(self):
        group = initialized()
        group.seqs[0].status = 'length'
        output = OUTPUT_NS['RequestOutput'].from_seq_group(group)
        with self.assertRaises(ValueError): validated(output)

    def test_source_hooks_are_before_text_and_preserve_core_policies(self):
        source = REPLACEMENTS['vllm/engine/llm_engine.py']
        self.assertLess(source.index('tc_produced_at = time.perf_counter()'),
                        source.index('seq_groups = self.scheduler.update(output)'))
        self.assertLess(source.index('dlora_tokens_available(seq_groups'),
                        source.index('self._decode_sequences(seq_groups)'))
        self.assertEqual(set(REPLACEMENTS), {'vllm/engine/llm_engine.py',
            'vllm/outputs.py', 'vllm/entrypoints/api_server.py'})
        # Scheduler/placement/migration/AsyncStream and all arithmetic are untouched.
        for path, expected in p.DLORA_OBSERVATION_SOURCES.items():
            self.assertEqual(p.sha((NATIVE/path).read_bytes()), expected)
        for source in REPLACEMENTS.values():
            self.assertNotIn('cuda.synchronize', source)

    def test_native_terminal_only_tracker_and_stream_transfer_unchanged(self):
        namespace = dict(asyncio=asyncio, Dict=Dict, List=List, Tuple=Tuple, Set=Set,
            ExecType=NS(LORA='lora'), RequestOutput=OUTPUT_NS['RequestOutput'],
            logger=logging.getLogger('dlora_test'))
        native_nodes((NATIVE/'vllm/engine/async_llm_engine.py').read_text(),
                     {'AsyncStream', 'RequestTracker'}, namespace)
        tracker = namespace['RequestTracker']('lora')
        stream = tracker.add_request('request-1')
        group = initialized()
        token(group, 100, 12.)
        tracker.process_request_output(0, OUTPUT_NS['RequestOutput'].from_seq_group(group))
        self.assertTrue(stream._queue.empty())
        migrated = namespace['RequestTracker']('lora')
        migrated.add_requests(tracker.pop_requests(['request-1']))
        self.assertIs(migrated._request_streams['request-1'], stream)
        token(group, 101, 13.)
        token(group, 102, 14.)
        group.seqs[0].status = 'length'
        output = OUTPUT_NS['RequestOutput'].from_seq_group(group)
        migrated.process_request_output(1, output)
        self.assertIs(stream._queue.get_nowait(), output)
        self.assertIs(stream._queue.get_nowait(), StopIteration)

    def test_final_api_retains_native_ids_and_event_times(self):
        namespace = native_nodes(REPLACEMENTS['vllm/entrypoints/api_server.py'],
            {'_request_output_payload'}, {})
        _, output = completed()
        payload = namespace['_request_output_payload'](output)
        self.assertEqual(payload['token_ids'], [[100, 101, 102]])
        self.assertEqual(payload['prompt_token_ids'], [1, 2])
        self.assertEqual(payload['finish_reasons'], ['length'])
        self.assertEqual(payload['tc_dlora_native']['first_at'], 12.)
        self.assertEqual(payload['usage']['completion_tokens'], 3)

    def test_exclusive_source_view_preserves_vendor_and_core_files(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'view'
            result = p.prepare_dlora_measurement_view(NATIVE, out, MAIN)
            self.assertFalse(result['performance_run_authorized'])
            self.assertTrue((out/'vllm/engine/engine_manager.py').is_symlink())
            self.assertEqual((out/'vllm/engine/engine_manager.py').resolve(),
                             NATIVE/'vllm/engine/engine_manager.py')
            self.assertFalse((out/'vllm/engine/llm_engine.py').is_symlink())
            self.assertTrue(result['terminal_only_delivery_preserved'])
            with self.assertRaises(FileExistsError):
                p.prepare_dlora_measurement_view(NATIVE, out, MAIN)


CANONICAL = p.dlora_measurement_sources(NATIVE, canonical_input=True)
replayer = load('tc_dlora_replayer', ROOT/'scripts/replay_vllm_engine_trace.py')
sampling = load('tc_dlora_sampling', NATIVE/'vllm/sampling_params.py')


class Exec(Enum):
    LORA = 1
    REPLICATED = 2


class Migration(Enum):
    PERIOD_MIG = 3
    DISPATCH_MIG = 2


def method(source, cls, name, namespace):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<retained-native-method>', 'exec'), namespace)
    return namespace[name]


native_select = method(CANONICAL['vllm/engine/engine_manager.py'], 'EngineManager', 'select_engine',
    dict(AsyncLLMEngine=object, Optional=Optional, List=List, ExecType=Exec, MigrationType=Migration, math=math))
native_generate = method((NATIVE/'vllm/engine/async_llm_engine.py').read_text(),
    'AsyncLLMEngine', 'generate', dict(Optional=Optional, List=List, SamplingParams=sampling.SamplingParams,
                                   RequestOutput=object, time=time))
tracker_namespace = native_nodes((NATIVE/'vllm/engine/async_llm_engine.py').read_text(),
    {'AsyncStream', 'RequestTracker'}, dict(asyncio=asyncio, Dict=Dict, List=List, Tuple=Tuple,
        Set=Set, ExecType=Exec, RequestOutput=object, logger=logging.getLogger('dlora_input_test')))


def common_body():
    from faaslora.clock import local_monotonic_clock_id
    return dict(model='3b', request_id='request-1', canonical_prompt='canonical text',
        input_tokens=[1, 2], lora_adapter_name='adapter-a', max_tokens=3,
        temperature=0, top_p=1, ignore_eos=True, stream=False, stop=[], stop_token_ids=[],
        _sllm_internal_metrics=dict(tc_clock_id=local_monotonic_clock_id()))


def fake_manager(*, pending=False, wrong_binding=False):
    """Execute native selection/generate/tracker, substituting only GPU output.

    Deliberately nonalphabetic loader order and a lower-cost second engine make
    sorted-ID rebinding or replacing native selection with RR observable.
    """
    manager = NS(exec_type=Exec.LORA, migration_type=Migration.PERIOD_MIG, is_running=True,
        select_lock=asyncio.Lock(), num_models=2, expected_lora_distribution=[.5, .5],
        model_engine_mapping={0:[0, 1], 1:[0, 1]}, engine_model_mapping={0:[0, 1], 1:[0, 1]},
        engine_num_requests=[1, 0], engine_exec_cost={0:1., 1:0.},
        engine_req_model_cnt={0:{0:0, 1:1}, 1:{0:0, 1:0}}, calls=[], aborts=[],
        registered=asyncio.Event(), engines=[])
    async def refresh():
        pass
    manager.get_engine_req_model_cnt = refresh
    manager.select_engine = MethodType(native_select, manager)
    for index in range(2):
        engine = NS(engine_id=index, lora_config=NS(num_models=2,
            adapter_ids=['adapter-z', 'adapter-a'], adapter_paths=['/pool/adapter-z', '/pool/adapter-a']),
            request_tracker=tracker_namespace['RequestTracker'](Exec.LORA))
        def abort(rid, engine=engine):
            manager.aborts.append((engine.engine_id, rid))
            engine.request_tracker.abort_request(rid)
        engine._abort = abort
        async def add(rid, mid, prompt, params, *, prompt_token_ids, arrival_time, engine=engine):
            manager.calls.append(dict(engine_id=engine.engine_id, request_id=rid, model_id=mid,
                prompt=prompt, prompt_token_ids=list(prompt_token_ids), sampling=params, arrival=arrival_time))
            stream = engine.request_tracker.add_request(rid)
            manager.registered.set()
            if not pending:
                group = Group(params.max_tokens)
                group.request_id, group.model_id = rid, mid+(1 if wrong_binding else 0)
                group.sampling_params = params
                group.seqs[0].data.prompt_token_ids = list(prompt_token_ids)
                m.dlora_register_sequence_group(group)
                m.dlora_scheduled([group], engine.engine_id)
                for i in range(params.max_tokens):
                    token(group, 100+i, time.perf_counter(), engine.engine_id)
                group.seqs[0].status = 'length'
                engine.request_tracker.process_request_output(engine.engine_id,
                    OUTPUT_NS['RequestOutput'].from_seq_group(group))
            return stream
        engine.add_request = add
        engine.generate = MethodType(native_generate, engine)
        manager.engines.append(engine)
    return manager


def handler(manager):
    from faaslora.clock import local_monotonic_clock_id
    return replayer.NativeDloraRequestHandler(manager=manager, sampling_factory=sampling.SamplingParams,
        model='3b', vocab_size=1000, max_model_len=1024, clock_id=local_monotonic_clock_id())


class DloraCanonicalInputTests(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, {'ieee_tc_serverless_measurement': m})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_loader_order_not_sorted_order_and_disagreeing_engines_rejected(self):
        manager = fake_manager()
        self.assertEqual(m.dlora_native_adapter_bindings(manager)['adapter-a'], (1, '/pool/adapter-a'))
        manager.engines[1].lora_config.adapter_ids.reverse()
        with self.assertRaises(ValueError): m.dlora_native_adapter_bindings(manager)

    def test_invalid_or_base_only_loader_identity_rejected(self):
        for kind in ('missing', 'duplicate', 'relative', 'wrong_count', 'replicated'):
            manager = fake_manager()
            cfg = manager.engines[0].lora_config
            if kind == 'missing': cfg.adapter_paths.clear()
            if kind == 'duplicate': cfg.adapter_ids[:] = ['same', 'same']
            if kind == 'relative': cfg.adapter_paths[0] = 'relative-path'
            if kind == 'wrong_count': cfg.num_models = 3
            if kind == 'replicated': manager.exec_type = Exec.REPLICATED
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                m.dlora_native_adapter_bindings(manager)

    def test_fixed_sampling_and_copied_canonical_ids(self):
        body = common_body()
        request = m.dlora_prepare_canonical_request(body, model='3b',
            adapters=m.dlora_native_adapter_bindings(fake_manager()), vocab_size=1000, max_model_len=1024)
        body['input_tokens'].append(5)
        self.assertEqual(request['prompt_token_ids'], [1, 2])
        params = sampling.SamplingParams(**request['sampling_kwargs'])
        self.assertEqual((params.n, params.best_of, params.top_k, params.max_tokens), (1, 1, -1, 3))
        self.assertTrue(params.ignore_eos)
        self.assertEqual(params.stop, [])

    def test_invalid_common_requests_never_reach_native_selection(self):
        async def check():
            for key, value in (('model','wrong'), ('request_id',''), ('lora_adapter_name','absent'),
                    ('canonical_prompt',''), ('input_tokens',[True]), ('input_tokens',[1000]),
                    ('input_tokens',[1]*1023), ('max_tokens',257), ('max_tokens',True),
                    ('temperature',1), ('top_p',.9), ('ignore_eos',False), ('stream',True),
                    ('stop',['stop']), ('stop_token_ids',[2]), ('n',2), ('best_of',2),
                    ('use_beam_search',True)):
                manager, body = fake_manager(), common_body()
                body[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    await handler(manager).generate(body)
                self.assertEqual(manager.calls, [])
        asyncio.run(check())

    def test_native_selection_generate_and_final_ids_share_canonical_work(self):
        async def check():
            manager = fake_manager()
            result = await handler(manager).generate(common_body())
            call, = manager.calls
            self.assertEqual(call['engine_id'], 1)  # Native cost choice, NOT round-robin.
            self.assertEqual(call['model_id'], 1)   # Native loader's unsorted index.
            self.assertEqual(call['prompt_token_ids'], [1, 2])
            self.assertEqual(call['prompt'], 'canonical text')
            measured = result['metrics']['ieee_tc']
            self.assertEqual(measured['native_prompt_token_ids'], [1, 2])
            self.assertEqual(measured['native_adapter_path'], '/pool/adapter-a')
            self.assertEqual(measured['native_model_id'], 1)
            self.assertEqual(measured['completion_token_ids'], [100, 101, 102])
            self.assertEqual(result['usage']['completion_tokens'], 3)
            self.assertFalse(measured['lora_numerical_correctness_qualified'])
            self.assertEqual(manager.aborts, [])
        asyncio.run(check())

    def test_output_binding_failure_not_relabelled_success(self):
        async def check():
            manager = fake_manager(wrong_binding=True)
            with self.assertRaises(ValueError): await handler(manager).generate(common_body())
            self.assertEqual(manager.aborts, [(1, 'request-1')])
        asyncio.run(check())

    def test_cancel_after_migration_aborts_current_not_initial_engine(self):
        async def check():
            manager = fake_manager(pending=True)
            task = asyncio.create_task(handler(manager).generate(common_body()))
            await manager.registered.wait()
            async with manager.select_lock:
                streams = manager.engines[1].request_tracker.pop_requests(['request-1'])
                manager.engines[0].request_tracker.add_requests(streams)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
            self.assertEqual(manager.aborts, [(0, 'request-1')])
            self.assertTrue(streams['request-1'].finished)
        asyncio.run(check())

    def test_duplicate_and_clock_mismatch_have_no_second_dispatch(self):
        async def check():
            manager, body = fake_manager(), common_body()
            service = handler(manager)
            body['_sllm_internal_metrics']['tc_clock_id'] = 'other-boot'
            with self.assertRaises(ValueError): await service.generate(body)
            self.assertEqual(manager.calls, [])
            await service.generate(common_body())
            with self.assertRaises(ValueError): await service.generate(common_body())
            self.assertEqual(len(manager.calls), 1)
        asyncio.run(check())

    def test_only_forwarding_changes_native_manager_and_opt_in_view(self):
        path = 'vllm/engine/engine_manager.py'
        source = CANONICAL[path].replace(
            'model_id: int,\n                            prompt_token_ids: Optional[List[int]] = None)',
            'model_id: int)').replace(
            'model_id,\n                                            prompt_token_ids=prompt_token_ids)', 'model_id)')
        self.assertEqual(source, (NATIVE/path).read_text())
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'view'
            manifest = p.prepare_dlora_measurement_view(NATIVE, out, MAIN, canonical_input=True)
            self.assertEqual(manifest['input_contract'], 'canonical_token_ids_v1')
            self.assertFalse((out/path).is_symlink())
            self.assertTrue((out/'vllm/engine/async_llm_engine.py').is_symlink())
            self.assertFalse(manifest['performance_run_authorized'])


class DloraOwnedRuntimeTests(unittest.TestCase):
    def test_private_ray_never_uses_discovery_and_store_is_service_total(self):
        cfg = dict(ray_root='/tmp/tc-dlora-test/ray', gpu_ids=[0, 1],
                   ray_object_store_bytes=8*1024**3, ray_num_cpus=40)
        with patch.dict('os.environ', {'RAY_ADDRESS':'10.0.0.1:9999'}):
            options = replayer.dlora_private_ray_options(cfg)
        self.assertEqual(options['address'], 'local')
        self.assertEqual(options['object_store_memory'], 8*1024**3)
        self.assertEqual(options['num_gpus'], 2)
        self.assertEqual(options['num_cpus'], 40)
        self.assertEqual(options['_system_config'],
                         dict(enable_worker_prestart=False, prestart_worker_first_driver=False))
        self.assertEqual(options['_node_ip_address'], '127.0.0.1')
        for change in (dict(gpu_ids=[0,1,2,3]), dict(ray_object_store_bytes=16*1024**3),
                       dict(ray_root='relative')):
            with self.assertRaises(ValueError):
                replayer.dlora_private_ray_options(dict(cfg, **change))

    def test_native_ray_guard_runs_before_worker_or_distributed_initialization(self):
        sources = p.dlora_measurement_sources(NATIVE, canonical_input=True, worker_lifecycle=True)
        tree = ast.parse(sources['vllm/engine/ray_utils.py'])
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'RayWorker')
        namespace = dict(TorchDistributedWorker=object)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<native-RayWorker>', 'exec'), namespace)
        calls = []
        support = NS(dlora_worker_launch_guard=lambda:calls.append('guard'),
                     dlora_native_worker_receipt=lambda worker: ('receipt', worker))
        with patch.dict(sys.modules, {'vllm.tc_measurement':support}):
            worker = namespace['RayWorker']()
            self.assertEqual(calls, ['guard'])
            self.assertIsNone(worker.worker)
            worker.worker = 'native-model'
            self.assertEqual(worker.tc_worker_receipt(), ('receipt','native-model'))
        # No modification of worker initialization, native NCCL, or arithmetic.
        self.assertNotIn('vllm/worker/worker.py', sources)
        self.assertNotIn('vllm/worker/lora_engine.py', sources)
        with self.assertRaises(ValueError): p.dlora_measurement_sources(NATIVE, worker_lifecycle=True)

    def test_receipt_reads_actual_capacity_and_native_loaded_order(self):
        cfg = dict(adapters={'z':[0,'/pool/z'], 'a':[1,'/pool/a']}, engine_args={'model':'/model'})
        guard = NS(gpu_process_identity=lambda pid:dict(pid=pid, start_ticks=3))
        native = NS(exec_type=NS(name='LORA'), model_config=NS(model='/model',use_dummy_weights=False,
                    hf_config=NS(vocab_size=128256)),
            parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1),
            lora_config=NS(num_models=2, adapter_ids=['z','a'], adapter_paths=['/pool/z','/pool/a'],
                           gpu_capacity=1),
            lora_engine=NS(cpu_lora_weights=[tuple(NS(lora_As=NS(shape=(2,3,4)),
                lora_Bs=NS(shape=(2,4,3))) for _ in range(4)) for _ in range(3)],
                num_hidden_layers=3,lora_cnt=4,gpu_lora_models=[1]), cache_engine=object(),
            cache_config=NS(num_gpu_blocks=300,num_cpu_blocks=900), rank=1)
        torch = NS(cuda=NS(device_count=lambda:1,current_device=lambda:0),__version__='test-only')
        with patch.object(m, 'dlora_worker_launch_guard', return_value=(cfg,guard)), \
                patch.dict(sys.modules, {'torch':torch}), patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES':'1'}):
            receipt = m.dlora_native_worker_receipt(native)
            self.assertEqual(receipt['actual_gpu_capacity'], 1)
            self.assertEqual(receipt['gpu_lora_models'], [1])
            self.assertEqual(receipt['adapter_bindings'], cfg['adapters'])
            self.assertEqual(receipt['cpu_lora_layers'],3)
            self.assertEqual(receipt['cpu_adapter_capacity'],2)
            self.assertFalse(receipt['numerical_correctness_qualified'])
            native.model_config.use_dummy_weights = True
            with self.assertRaises(ValueError): m.dlora_native_worker_receipt(native)

    def test_receipt_uses_native_layer_major_constructor_not_synthetic_adapter_list(self):
        source=(NATIVE/'vllm/worker/lora_engine.py').read_text()
        tree=ast.parse(source)
        classes=[]
        for name in ('LoRaEngine','LlamaLoRaEngine'):
            node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name))
            node.body=[n for n in node.body if isinstance(n,ast.Assign)
                       or isinstance(n,ast.FunctionDef) and n.name=='__init__']
            classes.append(node)
        # Execute the actual native constructor with shape-only tensor holders;
        # no weight data, CUDA allocation or altered native layout is introduced.
        class ShapeWeight:
            def __init__(self,engine,count,inputs,rank,outputs,dtype,device):
                self.lora_As=NS(shape=(count,inputs,rank))
                self.lora_Bs=NS(shape=(count,rank,outputs))
        fake_torch=NS(device=object,cuda=NS(Stream=object,current_stream=lambda:None,Event=object,
            device_count=lambda:1,current_device=lambda:0),__version__='shape-only')
        ns=dict(Tuple=Tuple,List=List,LoRaWeight=ShapeWeight,torch=fake_torch,
                ModelConfig=object,LoRaConfig=object,ParallelConfig=object,LlamaConfig=object)
        exec(compile(ast.Module(body=classes,type_ignores=[]),'<native-LoRA-constructors>','exec'),ns)
        config=NS(hidden_size=3072,num_hidden_layers=28,intermediate_size=8192,
                  num_attention_heads=24,num_key_value_heads=8,vocab_size=128256)
        lora=NS(num_models=500,max_r=16,gpu_capacity=250,
                adapter_ids=[f'a{i}' for i in range(500)],adapter_paths=[f'/pool/a{i}' for i in range(500)])
        parallel=NS(tensor_parallel_size=1,pipeline_parallel_size=1)
        model=NS(model='/model',use_dummy_weights=False,dtype='half',hf_config=config)
        engine=ns['LlamaLoRaEngine'](config,model,lora,parallel,'cuda:0')
        self.assertEqual(len(engine.cpu_lora_weights),28)
        self.assertEqual(engine.cpu_lora_weights[0][0].lora_As.shape,(500,3072,48))
        engine.gpu_lora_models=list(range(250))
        worker=NS(exec_type=NS(name='LORA'),model_config=model,parallel_config=parallel,
            lora_config=lora,lora_engine=engine,cache_engine=object(),
            cache_config=NS(num_gpu_blocks=100,num_cpu_blocks=100),rank=0)
        cfg=dict(engine_args={'model':'/model'},adapters={aid:[i,lora.adapter_paths[i]]
                                                        for i,aid in enumerate(lora.adapter_ids)})
        guard=NS(gpu_process_identity=lambda pid:dict(pid=pid,start_ticks=1))
        with patch.object(m,'dlora_worker_launch_guard',return_value=(cfg,guard)), \
                patch.dict(sys.modules,{'torch':fake_torch}),patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0'):
            receipt=m.dlora_native_worker_receipt(worker)
            self.assertEqual(receipt['cpu_lora_layers'],28)
            self.assertEqual(receipt['cpu_adapter_capacity'],500)
            engine.cpu_lora_weights[0][0].lora_As.shape=(499,3072,48)
            with self.assertRaises(ValueError):m.dlora_native_worker_receipt(worker)

    def test_runtime_shutdown_on_native_construction_failure(self):
        async def check():
            with tempfile.TemporaryDirectory() as directory:
                cfg = dict(ray_root=str(Path(directory)/'ray'), gpu_ids=[0,1],
                    ray_object_store_bytes=8*1024**3,ray_num_cpus=40,source_view=directory,
                    adapters={'a':[0,'/pool/a']}, engine_args=dict(num_groups=2,
                    tensor_parallel_size=1,pipeline_parallel_size=1,use_dummy_weights=False,
                    worker_use_ray=True,engine_use_ray=True,exec_type=Exec.LORA.value,num_models=1))
                ray = NS(__version__='2.51.2',is_initialized=lambda:False,init=Mock(),shutdown=Mock(),
                    nodes=lambda:[dict(Alive=True,Resources=dict(GPU=2,CPU=40,object_store_memory=8*1024**3))])
                factory = Mock(side_effect=RuntimeError('native constructor failed'))
                with patch.object(replayer,'dlora_worker_context',return_value=(cfg,None)), \
                        patch.object(replayer,'verify_dlora_source_view'), \
                        patch.object(replayer.inspect,'getfile',return_value=str(Path(directory)/'engine_manager.py')), \
                        patch.dict(sys.modules, {'ray':ray,'vllm.engine.arg_utils':NS(AsyncEngineArgs=NS),
                            'vllm.engine.engine_manager':NS(EngineManager=factory,MigrationType=Migration),
                            'vllm.config':NS(ExecType=Exec)}):
                    with self.assertRaisesRegex(RuntimeError,'native constructor failed'):
                        async with replayer.dlora_native_runtime('config'):
                            self.fail('failed constructor cannot publish ready')
                self.assertEqual(ray.init.call_args.kwargs['address'], 'local')
                ray.shutdown.assert_called_once_with()
        asyncio.run(check())

    def test_runtime_shutdown_on_partial_ray_initialization_failure(self):
        async def check():
            with tempfile.TemporaryDirectory() as directory:
                cfg = dict(ray_root=str(Path(directory)/'ray'),gpu_ids=[0,1],
                           ray_object_store_bytes=8*1024**3,ray_num_cpus=40)
                ray = NS(__version__='2.51.2',is_initialized=lambda:False,
                         init=Mock(side_effect=RuntimeError('private Ray failed')),shutdown=Mock())
                with patch.object(replayer,'dlora_worker_context',return_value=(cfg,None)), \
                        patch.object(replayer,'verify_dlora_source_view'), patch.dict(sys.modules,{'ray':ray}):
                    with self.assertRaisesRegex(RuntimeError,'private Ray failed'):
                        async with replayer.dlora_native_runtime('config'):
                            self.fail('failed Ray cannot yield a native manager')
                ray.shutdown.assert_called_once_with()
        asyncio.run(check())

    def test_lifecycle_view_records_sources_without_overwriting_legacy_view(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'view'
            result = p.prepare_dlora_measurement_view(NATIVE,out,MAIN,
                canonical_input=True,worker_lifecycle=True)
            self.assertTrue(result['worker_lifecycle'])
            self.assertEqual(result['original_sha256']['vllm/engine/ray_utils.py'],p.DLORA_RAY_SOURCE_SHA256)
            self.assertFalse((out/'vllm/engine/ray_utils.py').is_symlink())
            self.assertTrue((out/'vllm/worker/worker.py').is_symlink())


class DloraZeroMLPViewTests(unittest.TestCase):
    def make_adapter(self, directory):
        root = Path(directory)
        config = dict(target_modules=['q_proj','k_proj','v_proj','o_proj'], bias='none')
        metadata = {f'base_model.model.model.layers.0.self_attn.{module}.lora_{kind}.weight':
                    dict(dtype='F16',shape=[2,2],data_offsets=[0,8])
                    for module in config['target_modules'] for kind in ('A','B')}
        self.write_adapter(root, config, metadata)
        return root, config, metadata

    def write_adapter(self, root, config, metadata):
        (root/'adapter_config.json').write_text(json.dumps(config))
        header = json.dumps(metadata).encode()
        (root/'adapter_model.safetensors').write_bytes(len(header).to_bytes(8,'little')+header+b'\0'*64)

    def test_attention_only_pool_metadata_without_weight_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, _ = self.make_adapter(directory)
            result = m.dlora_attention_only_pool_metadata([str(root)],1)
            self.assertEqual(result['adapter_count'],1)
            self.assertEqual(result['adapters'][0]['tensor_count'],8)
            self.assertFalse(result['full_weight_content_verified'])

    def test_declared_or_serialized_mlp_is_not_silently_zeroed(self):
        with tempfile.TemporaryDirectory() as directory:
            root, cfg, header = self.make_adapter(directory)
            cfg['target_modules'].append('gate_proj')
            self.write_adapter(root,cfg,header)
            with self.assertRaisesRegex(ValueError,'attention-only'):
                m.dlora_attention_only_pool_metadata([str(root)],1)
            cfg['target_modules'].remove('gate_proj')
            header['base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight'] = {}
            self.write_adapter(root,cfg,header)
            with self.assertRaisesRegex(ValueError,'unsupported tensor'):
                m.dlora_attention_only_pool_metadata([str(root)],1)

    def test_incomplete_pool_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, cfg, header = self.make_adapter(directory)
            header.pop(next(iter(header)))
            self.write_adapter(root,cfg,header)
            with self.assertRaisesRegex(ValueError,'incomplete'):
                m.dlora_attention_only_pool_metadata([str(root)],1)

    def test_representation_only_changes_llama_cpu_mlp_allocation(self):
        key = 'vllm/worker/lora_engine.py'
        self.assertNotIn(key,REPLACEMENTS)
        replacements = p.dlora_measurement_sources(NATIVE,cpu_zero_mlp_views=True)
        self.assertEqual(set(replacements)-set(REPLACEMENTS),{key})
        old, new = ast.parse((NATIVE/key).read_text()),ast.parse(replacements[key])
        old_classes = {node.name:node for node in old.body if isinstance(node,ast.ClassDef)}
        new_classes = {node.name:node for node in new.body if isinstance(node,ast.ClassDef)}
        for name in ('LoRaWeight','LoRaEngine','OPTLoRaEngine'):
            self.assertEqual(ast.dump(old_classes[name]),ast.dump(new_classes[name]))
        before = {node.name:node for node in old_classes['LlamaLoRaEngine'].body if isinstance(node,ast.FunctionDef)}
        after = {node.name:node for node in new_classes['LlamaLoRaEngine'].body if isinstance(node,ast.FunctionDef)}
        for name in before.keys()-{'__init__'}:
            self.assertEqual(ast.dump(before[name]),ast.dump(after[name]))
        constructors = [node.func.id for node in ast.walk(after['__init__'])
                        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                        and node.func.id in ('LoRaWeight','TCZeroLoRaWeight')]
        self.assertEqual(constructors,['LoRaWeight','LoRaWeight','TCZeroLoRaWeight','TCZeroLoRaWeight'])

    def test_new_view_discloses_layout_without_modifying_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'view'
            result = p.prepare_dlora_measurement_view(NATIVE,out,MAIN,
                canonical_input=True,worker_lifecycle=True,cpu_zero_mlp_views=True)
            self.assertEqual(result['cpu_lora_representation'],'attention_only_cpu_mlp_zero_views_v1')
            self.assertFalse(result['cpu_kv_capacity_changed'])
            self.assertFalse(result['gpu_representation_changed'])
            self.assertFalse((out/'vllm/worker/lora_engine.py').is_symlink())


def dlora_configs():
    http = dict(schema='ieee_tc_dlora_http_replay_v1', backbone='/model', model='3b',
        url='http://127.0.0.1:19090/v1/chat/completions', physical_lifecycle='dlora_native_pool_v1',
        ingress_mode='service_pre_ready_v1', artifact_preparation='native_full_pool_before_service_v1',
        remote_artifacts={'real':True}, main_repo=str(MAIN))
    args = dict(model='/model', tokenizer='/model', use_dummy_weights=False, use_np_weights=False,
        dtype='half', worker_use_ray=True, engine_use_ray=True, tensor_parallel_size=1,
        pipeline_parallel_size=1, num_groups=2, num_models=500, exec_type=3,
        gpu_memory_utilization=.92, max_num_seqs=4, max_num_batched_tokens=1024,
        gpu_capacity=8, swap_space=8)
    cfg = dict(schema='ieee_tc_dlora_native_service_v1', python=sys.executable,
        native_python=sys.executable, source_view='/source/view', routing='native_engine_manager',
        worker_startup='native_dp2_manager_v1', generation_contract='fixed_length_greedy_v1',
        gpu_ids=[0,1], migration_type=3, router_port=19091, worker_ports=[19092],
        engine_args=args, request_max_model_len=1024, ray_root='/tmp/tc-dlora-test/ray',
        ray_object_store_bytes=8*1024**3, ray_num_cpus=40)
    return cfg, http


def dlora_service_context(manager):
    from faaslora.clock import local_monotonic_clock_id
    now, clock = time.perf_counter(), local_monotonic_clock_id()
    prep = {aid:dict(tc_artifact_id=aid, tc_artifact_request_id='dlora-full-pool:'+aid,
        tc_artifact_cache_hit=False, tc_artifact_content_sha256='a'*64,
        tc_artifact_manifest_sha256='b'*64, tc_artifact_receipt_sha256='c'*64,
        tc_artifact_transfer_id='d'*32, tc_artifact_transferred_bytes=1024,
        tc_artifact_delivery_mode='prepublished_gzip_v1', tc_artifact_resolve_started_s=now,
        tc_artifact_resolve_completed_s=now) for aid in manager.engines[0].lora_config.adapter_ids}
    return dict(preparation=prep, deployment_notice_s=now-60.,
        receipt=dict(tc_worker_receipt_id='e'*32, tc_worker_receipt_sha256='f'*64, tc_worker_ready_s=now),
        workers={i:dict(engine_id=i,pid=100+i,start_ticks=200+i,clock_id=clock) for i in range(2)})


class DloraHTTPServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, {'ieee_tc_serverless_measurement':m,
            'prepare_ieee_tc_serverless_stack':p})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_native_config_and_environment_never_select_shared_cluster_or_rr(self):
        cfg, http = dlora_configs()
        self.assertEqual(replayer.validate_native_service_config(cfg,http),19090)
        for change in (dict(routing='round_robin'), dict(gpu_ids=[0,1,2,3]),
                       dict(worker_ports=[19092,19093]), dict(migration_type=1)):
            with self.assertRaises(ValueError): replayer.validate_native_service_config(dict(cfg,**change),http)
        bad = copy.deepcopy(cfg);bad['engine_args']['use_dummy_weights']=True
        with self.assertRaises(ValueError): replayer.validate_native_service_config(bad,http)
        env = replayer.dlora_worker_environment(dict(cfg,main_repo=str(MAIN)),
            dict(RAY_ADDRESS='other-cluster',PYTHONPATH='/dirty',PYTHONHOME='/dirty',PATH='/usr/bin'))
        self.assertNotIn('RAY_ADDRESS',env)
        self.assertNotIn('PYTHONHOME',env)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0,1')
        self.assertTrue(env['PYTHONPATH'].startswith('/source/view:'))
        self.assertNotIn('/dirty',env['PYTHONPATH'])

    async def test_ingress_waits_without_choosing_native_engine(self):
        body = common_body();clock = body['_sllm_internal_metrics']['tc_clock_id']
        gate = replayer.NativeManagerIngress(['request-1'],clock)
        task = asyncio.create_task(gate.route(body,time.perf_counter()))
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        gate.ready[0].set()
        index, forwarded = await task
        self.assertEqual(index,0)  # API manager, not GPU engine 0.
        control = forwarded['_sllm_internal_metrics']
        self.assertNotIn('instance_id',control)
        self.assertNotIn('tc_instance_assigned_s',control)
        with self.assertRaises(ValueError): await gate.route(body,time.perf_counter())

    async def response(self):
        manager, body = fake_manager(), common_body()
        h = handler(manager)
        context = dlora_service_context(manager);h.service_context=context
        clock = body['_sllm_internal_metrics']['tc_clock_id']
        gate = replayer.NativeManagerIngress([body['request_id']],clock);gate.ready[0].set()
        now=time.perf_counter()
        event=dict(planned_arrival_s=now,task_created_s=now,client_submit_s=now)
        _,body=await gate.route(body,time.perf_counter())
        result=await h.generate(body)
        prepared=dict(request_id=body['request_id'], adapter_id=body['lora_adapter_name'],
            input_token_ids=body['input_tokens'], target_tokens=3, native_backend='dlora',
            canonical_prompt_sha256=replayer.hashlib.sha256(body['canonical_prompt'].encode()).hexdigest(),
            native_prompt_token_ids_sha256=m._native_ids_digest(body['input_tokens']),
            artifact_content_sha256='a'*64, artifact_manifest_sha256='b'*64,
            worker_lifetime_required=True, deployment_notice_s=context['deployment_notice_s'])
        return result,prepared,event,clock,manager

    async def test_native_selection_to_shared_timeline_and_physical_terminal(self):
        result,prepared,event,clock,manager=await self.response()
        completed_at=time.perf_counter()
        summary=p.validate_http_observation(prepared,result,event,completed_at,clock)
        self.assertTrue(summary['protocol_valid']);self.assertFalse(summary['lora_numerical_correctness_qualified'])
        native=result['metrics']['ieee_tc'];control=native['control_observation']
        self.assertEqual(manager.calls[0]['engine_id'],1)  # Native cost selected engine 1, not RR.
        self.assertEqual(control['instance_id'],'dlora-1')
        self.assertEqual(native['native_adapter_index'],1)  # Native nonsorted loader order.
        self.assertEqual(native['native_engine_workers'][0]['pid'],101)
        self.assertLessEqual(control['tc_manager_entry_s'],control['tc_instance_assigned_s'])
        self.assertLess(native['native_first_token_monotonic_s'],completed_at)
        from faaslora.datasets.workload_generator import FrozenReplayPlan
        from faaslora.metrics.metrics_collector import PhysicalGPUDeployment
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);trace=root/'trace.json'
            trace.write_text(json.dumps(dict(requests=[dict(request_id='request-1',arrival_time_s=0.,
                adapter_id='adapter-a',expected_output_tokens=3)])))
            plan=FrozenReplayPlan.load(trace)
            deployment=PhysicalGPUDeployment(root=root/'physical',plan=plan,context=dict(
                deployment_notice_s=prepared['deployment_notice_s'],replay_t0_s=event['planned_arrival_s'],
                clock_id=clock,plan=plan.identity()))
            replay=root/'replay.jsonl'
            rows=[dict(event='request_contract',source_item_sha256=plan.entries[0].source_sha256,**prepared),
                dict(event='http_raw_response',request_id='request-1',body=result,status=200),
                dict(event='http_response',request_id='request-1',response=summary,client_completed_s=completed_at)]
            replay.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            p.record_http_physical_terminals(deployment,replay)
            self.assertTrue(deployment.terminals['request-1']['native_contract_matched'])
            self.assertEqual(deployment.terminals['request-1']['completion_token_source'],'dlora_native_sequence_token_ids')

    async def test_shared_contract_rejects_wrong_binding_or_missing_lifetime(self):
        result,prepared,event,clock,_=await self.response()
        for field,value in [('native_model_id',0),('native_engine_workers',[]),
                            ('native_adapter_id','other'),('native_first_token_monotonic_s',float('nan'))]:
            bad=copy.deepcopy(result);bad['metrics']['ieee_tc'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                p.validate_http_observation(prepared,bad,event,time.perf_counter(),clock)
        bad=copy.deepcopy(result);bad['metrics']['ieee_tc']['control_observation'].pop('tc_worker_receipt_id')
        with self.assertRaises(ValueError):p.validate_http_observation(prepared,bad,event,time.perf_counter(),clock)

    async def test_shared_full_pool_primitive_keeps_native_preload_identity(self):
        now=time.perf_counter();paths={f'a{i:03}':f'/pool/a{i:03}' for i in range(500)}
        calls=[]
        async def resolve(aid,path,rid):
            calls.append((aid,path,rid))
            return dict(tc_artifact_id=aid,tc_artifact_request_id=rid,tc_artifact_cache_hit=False,
                tc_artifact_resolve_started_s=time.perf_counter(),tc_artifact_resolve_completed_s=time.perf_counter())
        prepared=await p.prepare_slora_full_pool(NS(resolve=resolve),paths,now,backend='dlora')
        self.assertEqual(len(prepared),500)
        self.assertTrue(all(rid=='dlora-full-pool:'+aid for aid,_,rid in calls))

    def test_ready_binding_covers_actual_gpu_workers_not_manager_pid(self):
        cfg,_=dlora_configs();cfg.update(source_view_manifest_sha256='a'*64,adapters={'a':[0,'/pool/a']})
        identity=dict(pid=99,start_ticks=1);ids={i:dict(pid=i,start_ticks=i) for i in (100,101)}
        allocation=NS(owner_id='lease',gpu_uuids=('GPU-a','GPU-b'))
        receipt=dict(schema='ieee_tc_dlora_backend_ready_v1',identity=identity,index=0,clock_id='clock',
            allocation_lease_id='lease',source_view_manifest_sha256='a'*64,preparation_sha256='b'*64,
            ready_boundary='native_EngineManager_model_LoRA_placement_and_KV_initialized',
            workers=[dict(engine_id=i,identity=ids[100+i],cuda_visible=str(i),clock_id='clock',
                adapter_bindings=cfg['adapters'],model='/model',tensor_parallel_size=1,pipeline_parallel_size=1,
                cuda_device_count=1,num_gpu_blocks=100,gpu_lora_models=[0],actual_gpu_capacity=1) for i in range(2)])
        kw=dict(cfg=cfg,identity=identity,worker_identity=lambda pid:ids[pid],allocation=allocation,
                clock_id='clock',preparation_sha256='b'*64)
        bindings=replayer.validate_dlora_ready_receipt(receipt,**kw)
        self.assertEqual({w['pid'] for w in bindings},{100,101})
        receipt['workers'][1]['cuda_visible']='0'
        with self.assertRaises(ValueError):replayer.validate_dlora_ready_receipt(receipt,**kw)

    def test_cli_worker_entry_uses_current_native_service_not_legacy_replay(self):
        with patch.object(replayer,'native_http_main',return_value=0) as entry, \
                patch.object(sys,'argv',['replay','native-dlora-worker','--config','/cfg','--index','0']):
            self.assertEqual(replayer.main(),0)
            entry.assert_called_once_with(sys.argv[1:])


if __name__ == '__main__':
    unittest.main()
