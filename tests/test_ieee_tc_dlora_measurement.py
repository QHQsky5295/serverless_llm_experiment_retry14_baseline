"""CPU-only execution of retained native methods; no Ray/GPU inference claim."""
import ast
import asyncio
import copy
import importlib.util
import logging
from pathlib import Path
import sys
import tempfile
import math
import time
from enum import Enum
from types import MethodType
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


if __name__ == '__main__':
    unittest.main()
