"""CPU-only S-LoRA observations: native methods, synthetic tensors, no inference."""
import ast
import asyncio
import collections
import copy
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
from typing import Dict, List, Union, AsyncGenerator
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
NATIVE = ROOT/'repos/S-LoRA'
sys.path.insert(0, str(MAIN))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


measurement = load('tc_slora_measurement', ROOT/'scripts/ieee_tc_serverless_measurement.py')
launch = load('tc_slora_source', ROOT/'scripts/prepare_ieee_tc_serverless_stack.py')
sampling = load('tc_slora_sampling', NATIVE/'slora/server/sampling_params.py')


def fixture(rid='r', target=3, path='/artifacts/a'):
    params = sampling.SamplingParams(do_sample=False, temperature=0, ignore_eos=True,
                                     max_new_tokens=target)
    params.verify()
    return dict(request_id=rid, adapter_dir=path, input_id=[1, 2],
                output_len=target, sampling_param=params.to_dict())


def server_fixture(bmm=False):
    params = NS(dummy=False, no_lora=False, no_kernel=False, no_lora_compute=False,
                scheduler='slora', bmm=bmm)
    # Unified paging moves weights into the shared pool, then offloads staging
    # tensors. is_on_gpu=False is therefore VALID on the packed path.
    adapter = NS(lora_dir='/artifacts/a', is_on_gpu=lambda: bmm)
    server = NS(world_size=1, tp_rank=0, input_params=params, model=object(),
                adapters=[adapter], adapter_id={'/artifacts/a': 0},
                infer_adapter=NS(adapter_dirs=['/artifacts/a']))
    name, module = ('LoraBmmInfer', 'slora.models.bmm.lora_bmm_infer') if bmm else (
        'LoraUnorderedBatchInfer', 'slora.models.peft.lora_unordered_batch_infer')
    engine = type(name, (), {'__module__': module})()
    engine.base_model, engine.infer_adapter = server.model, server.infer_adapter
    return server, engine


def native_payloads(target=3):
    request = fixture(target=target)
    server, engine = server_fixture()
    with patch.object(measurement.time, 'perf_counter', return_value=10.):
        measurement.slora_register_requests(server, [request])
    batch = NS(requests=[request], adapter_dirs=[request['adapter_dir']])
    with patch.object(measurement.time, 'perf_counter', return_value=11.):
        measurement.slora_forward_started(server, batch, engine)
    result = []
    for i in range(target):
        request['input_id'].append(100+i)
        event = measurement.slora_token_event(request, request['adapter_dir'], 100+i, 12.+i*.25)
        result.append(dict(token=dict(id=100+i), finished=i == target-1, tc_slora_native_token=event))
    return request, result


def consumer(target=3):
    return measurement.SloraNativeRequestObservation(request_id='r', adapter_path='/artifacts/a',
        prompt_ids=[1, 2], target=target,
        clock_id=measurement._slora_process_identity(__import__('os').getpid())['clock_id'])


class SloraHostBoundTests(unittest.TestCase):
    def facts(self, aid='a', rank=8):
        return dict(adapter_id=aid, rank=rank, dtype='F16', tensor_shapes={
            f'base_model.model.model.layers.0.self_attn.{module}.lora_{side}.weight':
            ([rank, 64] if side == 'A' else [64, rank])
            for module in ('q_proj', 'k_proj', 'v_proj', 'o_proj') for side in ('A', 'B')})

    def bound(self, rows, replicas=4, limit=80*1024**3):
        return launch.slora_host_storage_bound(rows,
            dict(model_type='llama', hidden_size=64, num_hidden_layers=1,
                 num_attention_heads=4, num_key_value_heads=4), replicas, limit)

    def test_two_native_copies_and_logical_not_content_deduplication(self):
        result = self.bound([self.facts('a'), self.facts('b')])
        self.assertEqual(result['single_copy_pool_bytes'], 2*8*64*8*2)
        self.assertEqual(result['pinned_adapter_bytes_lower_bound'], 2*4*2*8*64*8*2)
        self.assertFalse(result['deployment_memory_qualified'])
        self.assertFalse(result['performance_run_authorized'])

    def test_necessary_not_sufficient_and_boundary(self):
        n = self.bound([self.facts()])['pinned_adapter_bytes_lower_bound']
        self.assertFalse(self.bound([self.facts()], limit=n)['exceeds_memory_max'])
        self.assertTrue(self.bound([self.facts()], limit=n-1)['exceeds_memory_max'])
        self.assertFalse(self.bound([self.facts()], limit=n+1)['deployment_memory_qualified'])

    def test_reject_missing_extra_dtype_and_rank(self):
        for change in ('missing', 'extra', 'dtype', 'rank', 'shape'):
            row = self.facts()
            key = next(iter(row['tensor_shapes']))
            if change == 'missing':
                del row['tensor_shapes'][key]
            elif change == 'extra':
                row['tensor_shapes']['lm_head.weight'] = [8, 64]
            elif change == 'shape':
                row['tensor_shapes'][key] = [1, 1]
            else:
                row[change] = 'F32' if change == 'dtype' else 16
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.bound([row])

    def test_reject_duplicate_empty_and_invalid_resources(self):
        for rows in ([], [self.facts(), self.facts()]):
            with self.assertRaises(ValueError):
                self.bound(rows)
        for replicas, limit in ((True, 80), (0, 80), (5, 80), (1, 0), (1, True)):
            with self.assertRaises(ValueError):
                self.bound([self.facts()], replicas, limit)

    def test_native_layout_identity(self):
        for path, expected in launch.SLORA_HOST_LAYOUT_SOURCES.items():
            self.assertEqual(launch.sha((NATIVE/path).read_bytes()), expected)

    def test_actual_native_loader_retains_independent_cpu_storages(self):
        import torch
        native = load('tc_slora_native_layer', NATIVE/
            'slora/models/peft/layer_weights/lora_layer_weight.py')
        row = self.facts()
        weights = {k: torch.arange(__import__('math').prod(shape), dtype=torch.float16).reshape(shape)
                   for k, shape in row['tensor_shapes'].items()}
        model = dict(hidden_size=64, num_attention_heads=4, num_key_value_heads=4)
        layer = native.LoraLayerWeight(0, 0, 1, dict(r=8), model)
        # Exercise unmodified loader arithmetic/retention without CUDA pinning.
        # This is a storage-alias test, not measured pinned RSS or a GPU check.
        initialized_before = torch.cuda.is_initialized()
        with patch.object(torch.Tensor, 'pin_memory', lambda value: value.clone()):
            layer.load_hf_weights(weights, swap=True)
        homes = [getattr(layer, f'{module}_lora_{side}_home')
                 for side in ('A', 'B') for module in ('q', 'k', 'v', 'o')]
        all_tensors = homes + [layer.w_combined_home]
        self.assertEqual(len({v.untyped_storage().data_ptr() for v in all_tensors}), 9)
        actual_bytes = sum(v.untyped_storage().nbytes() for v in all_tensors)
        self.assertEqual(actual_bytes, self.bound([row], replicas=1)['pinned_adapter_bytes_lower_bound'])
        expected = torch.concat([v.T.reshape(8, 4, -1) for v in homes]).reshape(2, 32, 4, -1)
        self.assertTrue(torch.equal(layer.w_combined_home, expected))
        self.assertEqual(torch.cuda.is_initialized(), initialized_before)


class SloraObservationTests(unittest.TestCase):
    def test_mutable_prompt_snapshot_and_completion_tail(self):
        request, payloads = native_payloads()
        self.assertEqual(request['input_id'], [1, 2, 100, 101, 102])
        self.assertEqual(payloads[0]['tc_slora_native_token']['first_event']['native_prompt_token_ids'], [1, 2])
        self.assertNotIn('first_event', payloads[1]['tc_slora_native_token'])
        observed = consumer()
        for item in payloads:
            observed.observe(json.loads(json.dumps(item)))
        result = observed.finish(20.)
        self.assertEqual(result['native_decode_ms'], 500.)
        self.assertEqual(result['native_tpot_ms'], 250.)
        self.assertEqual(result['worker_completion_notification_ms'], 7500.)
        self.assertEqual(result['native_prompt_token_ids'], [1, 2])
        self.assertEqual(result['completion_token_ids'], [100, 101, 102])
        self.assertNotIn('native_queued_monotonic_s', result)
        self.assertFalse(result['lora_numerical_correctness_qualified'])

    def test_single_token_has_no_tpot(self):
        _, items = native_payloads(1)
        observed = consumer(1)
        observed.observe(items[0])
        self.assertIsNone(observed.finish(13.)['native_tpot_ms'])

    def test_registration_rejects_partial_batch_and_fallback(self):
        for field, value in [('dummy', True), ('no_lora', True), ('no_kernel', True),
                             ('no_lora_compute', True), ('scheduler', 'peft')]:
            server, _ = server_fixture()
            setattr(server.input_params, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                measurement.slora_register_requests(server, [fixture()])
        server, _ = server_fixture()
        first, second = fixture(), fixture()
        with self.assertRaises(ValueError):
            measurement.slora_register_requests(server, [first, second])
        self.assertNotIn('_tc_slora', first)
        server.world_size = 2
        with self.assertRaises(ValueError):
            measurement.slora_register_requests(server, [fixture()])

    def test_registration_rejects_wrong_sampler_and_token_types(self):
        for change in ('do_sample', 'top_k', 'presence_penalty', 'tokens', 'target'):
            server, _ = server_fixture()
            req = fixture()
            if change == 'tokens':
                req['input_id'] = [True]
            elif change == 'target':
                req['output_len'] = 257
            else:
                req['sampling_param'][change] = 2
            with self.subTest(change=change), self.assertRaises(ValueError):
                measurement.slora_register_requests(server, [req])

    def test_packed_pool_binding_and_first_forward_only(self):
        for bmm in (False, True):
            server, engine = server_fixture(bmm)
            req = fixture()
            measurement.slora_register_requests(server, [req])
            batch = NS(requests=[req], adapter_dirs=['/artifacts/a'])
            measurement.slora_forward_started(server, batch, engine)
            first = req['_tc_slora']['scheduled_monotonic_s']
            measurement.slora_forward_started(server, batch, engine)
            self.assertEqual(first, req['_tc_slora']['scheduled_monotonic_s'])
            if bmm:
                server.adapters[0].is_on_gpu = lambda: False
            else:
                server.infer_adapter.adapter_dirs.clear()
            with self.subTest(bmm=bmm), self.assertRaises(ValueError):
                measurement.slora_forward_started(server, batch, engine)

    def test_unrelated_engine_and_wrong_pool_rejected(self):
        server, engine = server_fixture()
        req = fixture()
        measurement.slora_register_requests(server, [req])
        batch = NS(requests=[req], adapter_dirs=['/artifacts/a'])
        for wrong in (NS(base_model=server.model), engine):
            engine.infer_adapter = NS(adapter_dirs=['/artifacts/a'])
            with self.assertRaises(ValueError):
                measurement.slora_forward_started(server, batch, wrong)

    def test_native_append_is_required_and_count_is_bounded(self):
        req, items = native_payloads(1)
        with self.assertRaises(ValueError):
            measurement.slora_token_event(req, req['adapter_dir'], 100, 13.)
        server, engine = server_fixture()
        req = fixture()
        measurement.slora_register_requests(server, [req])
        measurement.slora_forward_started(server, NS(requests=[req], adapter_dirs=[req['adapter_dir']]), engine)
        with self.assertRaises(ValueError):
            measurement.slora_token_event(req, req['adapter_dir'], 100, __import__('time').perf_counter())

    def test_corrupt_first_event_rejected(self):
        _, items = native_payloads()
        for field, value in [('native_prompt_token_ids', [9]), ('target_tokens', 2),
                             ('scheduled_monotonic_s', 20.), ('registered_monotonic_s', float('nan')),
                             ('engine_class', 'base_model')]:
            bad = copy.deepcopy(items[0])
            bad['tc_slora_native_token']['first_event'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                consumer().observe(bad)
        bad = copy.deepcopy(items[0])
        bad['tc_slora_native_token']['first_event']['process']['clock_id'] = 'other-host'
        with self.assertRaises(ValueError):
            consumer().observe(bad)

    def test_stream_missing_duplicate_reordered_or_wrong_id_rejected(self):
        _, items = native_payloads()
        for item in (items[1], {**items[0], 'token': {'id': True}},
                     {**items[0], 'token': {'id': 999}},
                     {**items[0], 'finished': True}, {'token': {'id': 100}}):
            with self.subTest(item=item), self.assertRaises(ValueError):
                consumer().observe(item)
        observed = consumer()
        observed.observe(items[0])
        with self.assertRaises(ValueError):
            observed.observe(items[0])
        with self.assertRaises(ValueError):
            observed.finish(20.)
        bad = copy.deepcopy(items[1])
        bad['tc_slora_native_token']['produced_monotonic_s'] = 11.
        with self.assertRaises(ValueError):
            observed.observe(bad)
        for item in items[1:]:
            observed.observe(item)
        with self.assertRaises(ValueError):
            observed.finish(12.)
        with self.assertRaises(ValueError):
            observed.observe(items[-1])


class StripObservation(ast.NodeTransformer):
    """Remove only the five explicitly inserted observation statements/import."""
    def visit_ImportFrom(self, node):
        return None if node.module == 'slora.server.tc_measurement' else node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id in ('slora_register_requests', 'slora_forward_started'):
                return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'tc_slora_produced_at':
                return None
            if (isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                    and target.slice.value == 'tc_slora_native_token'):
                return None
        return self.generic_visit(node)


class SloraSourceTests(unittest.TestCase):
    def test_only_observation_ast_changes(self):
        replacements = launch.slora_measurement_sources(NATIVE)
        self.assertEqual(len(replacements), 2)
        for path, source in replacements.items():
            before = ast.parse((NATIVE/path).read_text())
            restored = StripObservation().visit(ast.parse(source))
            self.assertEqual(ast.dump(before), ast.dump(restored), path)
            compile(source, path, 'exec')
        rpc = replacements['slora/server/router/model_infer/model_rpc.py']
        self.assertIn('numpy()\n        tc_slora_produced_at = time.perf_counter()', rpc)

    def test_exclusive_view_preserves_original_and_unmodified_files(self):
        before = {p: launch.sha((NATIVE/p).read_bytes()) for p in launch.SLORA_OBSERVATION_SOURCES}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'view'
            manifest = launch.prepare_slora_measurement_view(NATIVE, out, MAIN)
            self.assertFalse(manifest['performance_run_authorized'])
            self.assertFalse((out/'slora/server/api_server.py').is_symlink())
            self.assertTrue((out/'slora/server/router/model_infer/infer_batch.py').is_symlink())
            self.assertEqual((out/'slora/server/tc_measurement.py').resolve(), ROOT/'scripts/ieee_tc_serverless_measurement.py')
            self.assertFalse((out/'sitecustomize.py').exists())
            with self.assertRaises(FileExistsError):
                launch.prepare_slora_measurement_view(NATIVE, out, MAIN)
        self.assertEqual(before, {p: launch.sha((NATIVE/p).read_bytes()) for p in before})

    def test_source_drift_fails_before_creating_view(self):
        with tempfile.TemporaryDirectory() as directory:
            src, out = Path(directory)/'source', Path(directory)/'view'
            for p in launch.SLORA_OBSERVATION_SOURCES:
                (src/p).parent.mkdir(parents=True, exist_ok=True)
                (src/p).symlink_to(NATIVE/p)
            changed = src/'slora/server/api_server.py'
            changed.unlink()
            changed.write_text((NATIVE/'slora/server/api_server.py').read_text()+'\n# drift\n')
            with self.assertRaisesRegex(ValueError, 'source drift'):
                launch.prepare_slora_measurement_view(src, out, MAIN)
            self.assertFalse(out.exists())


class CpuTensor(np.ndarray):
    def cuda(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def new_empty(self, shape):
        return np.empty(shape, dtype=self.dtype).view(CpuTensor)


def cpu_torch():
    # Explicit fake allocation backend for executing unchanged native Python
    # methods; this is neither CUDA execution nor a numerical model witness.
    def alloc(fn):
        def run(*args, **kwargs):
            kwargs.pop('device', None)
            return fn(*args, **kwargs).view(CpuTensor)
        return run
    return NS(Tensor=CpuTensor, dtype=object, device=object, float16=np.float16,
        int32=np.int32, int64=np.int64, long=np.int64,
        no_grad=lambda: lambda fn: fn, cuda=NS(current_device=lambda: 0),
        tensor=alloc(np.array), empty=alloc(np.zeros), zeros=alloc(np.zeros),
        arange=alloc(np.arange), log=alloc(np.log), max=np.max, sum=np.sum,
        cat=lambda items, dim: np.concatenate(items, axis=dim).view(CpuTensor),
        cumsum=lambda a, dim, dtype: np.cumsum(a, axis=dim, dtype=dtype).view(CpuTensor))


class NativeBatchTests(unittest.TestCase):
    def test_actual_native_init_forward_filter_merge_preserve_observation(self):
        torch = cpu_torch()
        scope = dict(torch=torch, np=np, collections=collections, dataclass=dataclass,
                     List=List, Dict=Dict, MemoryManager=object, setting={'max_req_total_len': 16})
        source = (NATIVE/'slora/server/router/model_infer/infer_batch.py').read_text()
        nodes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'native_infer_batch', 'exec'), scope)
        Batch = scope['InferBatch']
        memory = NS(free=Mock())
        server, engine = server_fixture()
        server.model = NS(mem_manager=memory, vocab_size=200)
        engine.base_model = server.model
        engine.forward = Mock(return_value='actual-native-forward-boundary')
        engine_class = lambda *args, **kwargs: engine
        replacements = launch.slora_measurement_sources(NATIVE)
        rpc = ast.parse(replacements['slora/server/router/model_infer/model_rpc.py'])
        methods = next(n for n in rpc.body if isinstance(n, ast.ClassDef) and n.name == 'ModelRpcServer').body
        chosen = [n for n in methods if isinstance(n, ast.FunctionDef) and n.name in ('exposed_add_batch', 'forward')]
        scope.update(InferBatch=Batch, time=measurement.time,
            slora_register_requests=measurement.slora_register_requests,
            slora_forward_started=measurement.slora_forward_started,
            slora_token_event=measurement.slora_token_event, LoraUnorderedBatchInfer=engine_class,
            sample=lambda logits, batch: (torch.tensor([100]*len(batch)), torch.tensor([.5]*len(batch))))
        exec(compile(ast.Module(body=chosen, type_ignores=[]), 'observed_native_rpc', 'exec'), scope)
        server.cache = {}
        reqs = [fixture('r'), fixture('s'), fixture('t')]
        with patch.dict(sys.modules, {'torch': torch}):
            scope['exposed_add_batch'](server, 'one', reqs[:2], 'fp16')
            scope['exposed_add_batch'](server, 'two', reqs[2:], 'fp16')
        self.assertIs(server.cache['one'].all_input_ids[0], reqs[0]['input_id'])
        first = scope['forward'](server, 'one', True)
        scope['forward'](server, 'two', True)
        filtered = server.cache['one'].filter(['r'])
        merged = Batch.merge(filtered, server.cache['two'])
        self.assertIs(merged.requests[0], reqs[0])
        self.assertIs(merged.requests[1], reqs[2])
        self.assertEqual([r['_tc_slora']['count'] for r in merged.requests], [1, 1])
        server.cache['one'] = merged
        second = scope['forward'](server, 'one', False)
        self.assertEqual(second['r'][1]['tc_slora_native_token']['output_index'], 2)
        self.assertEqual(second['t'][1]['tc_slora_native_token']['output_index'], 2)
        self.assertEqual(first['r'][1]['tc_slora_native_token']['first_event']['native_prompt_token_ids'], [1, 2])
        self.assertEqual(reqs[1]['_tc_slora']['count'], 1)
        self.assertEqual(engine.forward.call_count, 3)


class NativeSSETests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_detokenizer_and_fifo_preserve_burst_event_identity(self):
        """Execute existing transport methods; only sockets/tokenizer are fakes."""
        class BatchTokenIdOut:
            pass
        class ReqDetokenizationState:
            pass
        class AbortReq:
            pass
        class BatchAbortReq:
            pass
        class BatchStrOut:
            def __init__(self):
                self.reqs_infs = []
        router_to_detok, detok_to_http = asyncio.Queue(), asyncio.Queue()
        scope = dict(asyncio=asyncio, Union=Union, BatchTokenIdOut=BatchTokenIdOut,
            ReqDetokenizationState=ReqDetokenizationState, AbortReq=AbortReq,
            BatchAbortReq=BatchAbortReq, BatchStrOut=BatchStrOut,
            decode_token=lambda tokenizer, state, token, **kw: 'x'*len(state.output_ids))
        for path, name in [('slora/server/detokenization/manager.py', 'DeTokenizationManager'),
                           ('slora/server/httpserver/manager.py', 'HttpServerManager')]:
            node = next(n for n in ast.parse((NATIVE/path).read_text()).body
                        if isinstance(n, ast.ClassDef) and n.name == name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), scope)
        detok = object.__new__(scope['DeTokenizationManager'])
        detok.tokenizer, detok.req_id_to_out = object(), {}
        detok.recv_from_router = NS(recv_pyobj=router_to_detok.get)
        detok.send_to_httpserver = NS(send_pyobj=detok_to_http.put_nowait)
        manager = object.__new__(scope['HttpServerManager'])
        manager.tokenizer = NS(encode=lambda _: [1, 2])
        manager.total_token_num, manager.max_req_input_len, manager.max_req_total_len = 512, 32, 512
        manager.req_id_to_out_inf = {}
        manager.send_to_router = NS(send_pyobj=Mock())
        manager.recv_from_detokenization = NS(recv_pyobj=detok_to_http.get)
        async def consume():
            observed = consumer()
            params = sampling.SamplingParams(ignore_eos=True, max_new_tokens=3)
            async for text, metadata, finished in manager.generate('/artifacts/a', 'prompt', params, 'r'):
                observed.observe(dict(token=dict(id=metadata['id']), finished=finished,
                                      tc_slora_native_token=metadata['tc_slora_native_token']))
                # Let the entire producer burst run ahead of the HTTP consumer.
                await asyncio.sleep(0)
            return observed.finish(20.)
        tasks = [asyncio.create_task(detok.handle_loop()), asyncio.create_task(manager.handle_loop())]
        output = asyncio.create_task(consume())
        try:
            for _ in range(10):
                if 'r' in manager.req_id_to_out_inf:
                    break
                await asyncio.sleep(0)
            self.assertIn('r', manager.req_id_to_out_inf)
            state = ReqDetokenizationState()
            state.request_id, state.output_ids, state.gen_metadata, state.output_str = 'r', [], {}, ''
            router_to_detok.put_nowait(state)
            _, items = native_payloads()
            burst = BatchTokenIdOut()
            burst.reqs_infs = [('r', p['token']['id'],
                               dict(id=p['token']['id'], tc_slora_native_token=p['tc_slora_native_token']),
                               p['finished'], False) for p in items]
            router_to_detok.put_nowait(burst)
            result = await asyncio.wait_for(output, timeout=2.)
            self.assertEqual(result['completion_token_ids'], [100, 101, 102])
            self.assertEqual(result['native_tpot_ms'], 250.)
            self.assertNotIn('r', manager.req_id_to_out_inf)
            self.assertNotIn('r', detok.req_id_to_out)
        finally:
            for task in tasks+[output]:
                task.cancel()
            await asyncio.gather(*tasks, output, return_exceptions=True)

    async def test_actual_api_preserves_native_event_with_slow_consumer(self):
        _, items = native_payloads()
        async def generate(*args):
            for item in items:
                yield 'x', dict(id=item['token']['id'], logprob=0.,
                               tc_slora_native_token=item['tc_slora_native_token']), item['finished']
        source = launch.slora_measurement_sources(NATIVE)['slora/server/api_server.py']
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'generate_stream')
        node.decorator_list = []
        scope = dict(Request=object, Response=object, AsyncGenerator=AsyncGenerator,
            isFirst=False, asyncio=asyncio, json=json, SamplingParams=sampling.SamplingParams,
            httpserver_manager=NS(generate=generate, abort=Mock()),
            BackgroundTasks=lambda: NS(add_task=Mock()), StreamingResponse=lambda stream, **kw: stream)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'native_sse', 'exec'), scope)
        async def body():
            return dict(req_id='r', lora_dir='/artifacts/a', inputs='prompt',
                        parameters=dict(do_sample=False, temperature=0, ignore_eos=True, max_new_tokens=3))
        stream = await scope['generate_stream'](NS(json=body))
        observed = consumer()
        async for data in stream:
            await asyncio.sleep(0)
            observed.observe(json.loads(data.decode().removeprefix('data:')))
        self.assertEqual(observed.finish(20.)['native_tpot_ms'], 250.)


if __name__ == '__main__':
    unittest.main()
