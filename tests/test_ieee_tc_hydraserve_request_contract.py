"""CPU-only native source/input plumbing. NOT a GPU performance qualification."""
import ast
import asyncio
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from HydraServe_project.runtime import request_contract as rc
from tests.test_ieee_tc_hydraserve_platform_port import NATIVE


def contract(**kw):
    values = dict(request_id='req_00001', adapter_id='adapter_0001', adapter_index=2,
        adapter_path='/cache/adapter_0001', canonical_prompt='unchanged prompt',
        prompt_token_ids=(1, 42, 73), target_tokens=3)
    values.update(kw)
    return rc.FrozenRequest(**values)


def response(ids, cumulative, *, finish=None, stamp=12.0, **kw):
    c = contract()
    event = dict(request_id=c.request_id, native_request_id='native-0',
        adapter_id=c.adapter_id, adapter_index=c.adapter_index, adapter_path=c.adapter_path,
        prompt_token_ids=list(c.prompt_token_ids), delta_token_ids=ids,
        cumulative_tokens=cumulative, output_monotonic_s=stamp,
        process=dict(pid=12, start_ticks=34, boot_id='test-boot'))
    event.update(kw)
    row = dict(choices=[dict(index=0, text='' if cumulative == 1 else 'multi',
                            finish_reason=finish)], hydraserve_native=event,
        usage=(dict(completion_tokens=cumulative, prompt_tokens=3, total_tokens=3+cumulative)
               if finish else None))
    return NS(model_dump=lambda: row)


class Stream:
    def __init__(self, responses):
        self.responses, self.closed = responses, False

    def __iter__(self):
        yield from self.responses

    def close(self):
        self.closed = True


def load_native(overlays):
    # Only dependency imports are replaced by doubles; original definitions and
    # the actual source overlay execute. No K8s/Podman/model startup occurs here.
    sources = [overlays[rc.ENGINE], overlays[rc.platform_port.SOURCE_DIR+'/scheduler.py']]
    ns = dict(__name__=__name__, OpenAI=lambda **kw: NS(), AsyncOpenAI=lambda **kw: NS(),
              ResourceUsage=object, ResourceManager=object, AllocatedResource=object)
    for source in sources:
        tree = ast.parse(source)
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom)
            and n.module in ('openai', 'engine', 'resource_manager', 'cache_manager', 'utils'))]
        exec(compile(tree, 'rendered-native-hydraserve', 'exec'), ns)
    return ns


class ContractTests(unittest.TestCase):
    def test_fixed_arguments_no_template_or_text_retokenization(self):
        c = contract(target_tokens=256)
        args = c.completion_arguments()
        self.assertEqual(args['model'], c.adapter_id)
        self.assertEqual(args['prompt'], [1, 42, 73])
        self.assertNotIn('messages', args)
        self.assertEqual(args['max_tokens'], 256)
        self.assertEqual((args['temperature'], args['top_p'], args['stop']), (0, 1, []))
        self.assertEqual(args['extra_body'], dict(ignore_eos=True, stop_token_ids=[]))
        self.assertEqual(len(c.prompt_sha256), 64)

    def test_strict_input_types_and_paths(self):
        for changes in (dict(target_tokens=True), dict(target_tokens=257),
                dict(adapter_index=0), dict(prompt_token_ids=(True,)),
                dict(prompt_token_ids=[]), dict(adapter_path='/cache/../adapter_0001'),
                dict(adapter_id='../../secret'), dict(request_id='')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                contract(**changes)

    def test_empty_text_multitoken_chunk_native_count(self):
        b = rc.BoundRequest(contract())
        stream = Stream([response([11], 1), response([12, 13], 3, finish='length', stamp=13.0)])
        out = list(rc.CompletionAsChatStream(stream, b))
        self.assertTrue(stream.closed and b.terminal)
        self.assertEqual(out[-1].usage.completion_tokens, 3)
        self.assertEqual(len(out), 2)  # 2 text chunks != 3 native tokens.
        self.assertEqual(out[0].choices[0].delta.content, '')

    def test_no_adapter_fallback_or_invented_terminal(self):
        for rows in ([response([11], 1, adapter_id='base')],
                [response([True], 1)], [response([11], 1)],
                [response([11], 1), response([12, 13], 3, finish='stop')],
                [response([11], 1), response([12, 13], 3, finish='length', stamp=1.0)],
                [response([11], 1), response([12, 13], 3, finish='length', native_request_id='other')]):
            stream = Stream(rows)
            with self.assertRaises(ValueError):
                list(rc.CompletionAsChatStream(stream, rc.BoundRequest(contract())))
            self.assertTrue(stream.closed)

    def test_policy_methods_ast_unchanged(self):
        overlays = rc.render_overlays(NATIVE)
        path = rc.platform_port.SOURCE_DIR+'/scheduler.py'
        before = ast.parse(rc.platform_port.render_overlays(NATIVE)[path])
        after = ast.parse(overlays[path])
        classes = lambda t: {n.name: n for n in t.body if isinstance(n, ast.ClassDef)}
        old, new = classes(before), classes(after)
        # Whole original scheduling/planner/Request/Model classes, including
        # autoscaler, admission count and binding assignment are unchanged.
        for name in old.keys()-{'Instance'}:
            self.assertEqual(ast.dump(old[name]), ast.dump(new[name]), name)
        methods = lambda n: {m.name: m for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
        a, b = methods(old['Instance']), methods(new['Instance'])
        for name in a.keys()-{'chat', 'resp_generator'}:
            self.assertEqual(ast.dump(a[name]), ast.dump(b[name]), name)
        # Success-path profiling is verbatim, merely protected by finally.
        self.assertEqual([ast.dump(n) for n in a['resp_generator'].body[:-3]],
                         [ast.dump(n) for n in b['resp_generator'].body[0].body])
        self.assertEqual([ast.dump(n) for n in a['resp_generator'].body[-3:]],
                         [ast.dump(n) for n in b['resp_generator'].body[0].finalbody])

    def test_native_annotation_uses_output_snapshot(self):
        req = NS(user=rc.PREFIX+'req_00001', n=1, echo=False, ignore_eos=True,
            temperature=0, top_p=1, stop=[], stop_token_ids=[], model='adapter_0001')
        res = NS(lora_request=NS(lora_name=req.model, lora_int_id=2,
            lora_local_path='/cache/adapter_0001'), request_id='native-0',
            prompt_token_ids=[1, 42, 73], tc_output_monotonic_s=123.0)
        out = NS(token_ids=[11, 12, 13])
        data = json.loads(rc.annotate_native_event('{}', req, res, out, [12, 13]))
        self.assertEqual(data['hydraserve_native']['output_monotonic_s'], 123.0)
        self.assertEqual(data['hydraserve_native']['delta_token_ids'], [12, 13])
        res.lora_request = None
        with self.assertRaises(ValueError):
            rc.annotate_native_event('{}', req, res, out, [12, 13])

    def test_output_copies_ids_before_buffering(self):
        source = rc.render_overlays(NATIVE)[rc.OUTPUTS]
        tree = ast.parse(source)
        # Execute CompletionOutput only without importing GPU dependencies.
        node = next(n for n in tree.body if getattr(n, 'name', None) == 'CompletionOutput')
        import typing
        ns = dict(List=typing.List, Optional=typing.Optional, Union=typing.Union,
                  SampleLogprobs=list, LoRARequest=object)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'native-output', 'exec'), ns)
        ids = [3]
        out = ns['CompletionOutput'](0, '', ids, 0, None)
        ids.append(4)
        self.assertEqual(out.token_ids, [3])

    def test_source_drift_fails_closed(self):
        original = Path.read_bytes
        def read(path):
            raw = original(path)
            return raw+b'\n' if path.name == 'engine.py' else raw
        with patch.object(Path, 'read_bytes', read), self.assertRaises(ValueError):
            rc.render_overlays(NATIVE)

    def test_registry_metadata_only_and_native_stage_registration(self):
        from HydraServe_project.runtime import native_delivery
        overlays = native_delivery.render_overlays(NATIVE, request_contract=True)
        self.assertEqual(overlays[native_delivery.APP].decode().count('lora_modules=frozen_lora_modules()'), 2)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'registry.json'
            rows = [dict(adapter_id='adapter_0001', adapter_index=1,
                         adapter_path='/not-materialized/adapter_0001')]
            raw = json.dumps(rows).encode()
            path.write_bytes(raw)
            with patch.dict(os.environ, dict(HYDRASERVE_LORA_REGISTRY=str(path),
                    HYDRASERVE_LORA_REGISTRY_SHA256=rc.sha256(raw).hexdigest())), \
                    patch.dict(sys.modules, {'vllm.entrypoints.openai.serving_engine': NS(LoRAModulePath=NS)}):
                modules = rc.frozen_lora_modules()
                self.assertEqual(modules[0].local_path, '/not-materialized/adapter_0001')
                path.write_bytes(raw+b' ')
                with self.assertRaisesRegex(ValueError, 'identity drift'):
                    rc.frozen_lora_modules()


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ns = load_native(rc.render_overlays(NATIVE))
        self.model = self.ns['Model']('llama2_7b/0', (14, 10000, 100, 1, 1))
        self.instance = self.ns['Instance'](self.model, 1, False, 0,
                                           ['127.0.0.1'], [object()])
        self.calls = []
        def create(**kw):
            self.calls.append(kw)
            return Stream([response([11], 1), response([12, 13], 3, finish='length', stamp=13.0)])
        self.instance.chat_engine.client.completions = NS(create=create)

    async def test_original_request_to_instance_to_engine(self):
        b = rc.BoundRequest(contract())
        request = self.ns['Request'](b.contract.request_id, self.model.model_id, b, True, 0)
        with redirect_stdout(io.StringIO()):
            request.allocate_instance(self.instance)
            reply = await request.resp_func
            self.assertEqual(self.instance.num_reqs, 1)
            self.assertEqual(list(reply), ['', 'multi', '#3'])
        self.assertIs(request.prompt, b)
        self.assertTrue(b.terminal)
        self.assertEqual(self.instance.num_reqs, 0)
        self.assertEqual(self.instance.num_coldstart_reqs, 0)
        self.assertEqual(self.model.model_stats.num_output_tokens, 3)
        self.assertEqual(self.calls, [b.contract.completion_arguments()])

    async def test_bad_stream_releases_native_counters(self):
        self.instance.chat_engine.client.completions.create = lambda **kw: Stream([response([2], 1, adapter_id='base')])
        self.instance.num_reqs = 1
        reply = await self.instance.chat(1, rc.BoundRequest(contract()), True)
        with self.assertRaises(ValueError):
            list(reply)
        self.assertEqual((self.instance.num_reqs, self.instance.num_coldstart_reqs), (0, 0))

    async def test_submission_failure_propagates_releases_counts(self):
        def create(**kw):
            raise RuntimeError('test submission error')
        self.instance.chat_engine.client.completions.create = create
        self.instance.num_reqs = 1
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'submission'):
            await self.instance.chat(1, rc.BoundRequest(contract()), True)
        self.assertEqual((self.instance.num_reqs, self.instance.num_coldstart_reqs), (0, 0))

    async def test_cancel_while_waiting_for_native_init(self):
        self.instance.inited = False
        self.instance.num_reqs = 1
        task = asyncio.create_task(self.instance.chat(1, rc.BoundRequest(contract()), True))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual((self.instance.num_reqs, self.instance.num_coldstart_reqs), (0, 0))

    async def test_cold_stream_close_releases_once(self):
        self.instance.pp_size = 2
        self.instance.num_reqs = 1
        reply = await self.instance.chat(1, rc.BoundRequest(contract()), True)
        next(reply)
        self.assertEqual((self.instance.num_reqs, self.instance.num_coldstart_reqs), (1, 1))
        reply.close()
        reply.close()
        self.assertEqual((self.instance.num_reqs, self.instance.num_coldstart_reqs), (0, 0))


if __name__ == '__main__':
    unittest.main()
