"""Source integrity plus opt-in CPU execution of the actual measured native methods.

No GPU arithmetic/performance evidence. Run native tests in loquetier_20260520
with TC_TEST_LOQUETIER_NATIVE=1; skipping those tests is not qualification.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest

from scripts import prepare_ieee_tc_serverless_stack as source

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT/'vendor_new_baselines/Loquetier_main_20260520'
MAIN = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')


class RemoveObservations(ast.NodeTransformer):
    def visit_Import(self, node):
        return None if any(n.asname == 'tc_time' for n in node.names) else node

    def visit_ImportFrom(self, node):
        return None if node.module == 'tc_measurement' else node

    def visit_Expr(self, node):
        call = node.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id in ('loquetier_register_requests', 'loquetier_forward_started',
                                     'loquetier_token_event', 'loquetier_finished')):
            return None
        return self.generic_visit(node)


class LoquetierSourceTests(unittest.TestCase):
    def test_packed_query_rows_cover_every_token_head_once(self):
        # Validate the actual index derivation independently of CUDA compilation.
        for group in (1, 2, 3, 4, 8):
            for frags in (1, 2):
                rows_per_tile = 4*frags*16
                for length in (1, 17, 257, 759):
                    written = []
                    for tile in range(math.ceil(length*group/rows_per_tile)):
                        for warp in range(4):
                            base = tile*rows_per_tile+warp*frags*16
                            for fragment in range(frags):
                                for half in range(2):
                                    for lane_row in range(8):
                                        packed = base+fragment*16+half*8+lane_row
                                        token, head = divmod(packed, group)
                                        if token < length:
                                            written.append((token, head))
                                        next_token, next_head = divmod(packed+8, group)
                                        stride_n, stride_h = 8*group*128, 128
                                        step = (next_token-token)*stride_n+(next_head-head)*stride_h
                                        self.assertEqual(token*stride_n+head*stride_h+step,
                                                         next_token*stride_n+next_head*stride_h)
                    self.assertEqual(sorted(written), [(t,h) for t in range(length) for h in range(group)])

    def test_configured_llama3_view_preserves_native_sources(self):
        cfg = MAIN/'models/LLM-Research--Llama-3.2-3B-Instruct/config.json'
        native_headers = NATIVE/'kernel_src/third_party/flashinfer/include/flashinfer'
        before = {name:source.sha((native_headers/name).read_bytes())
                  for name in ('prefill.cuh', 'decode.cuh')}
        with tempfile.TemporaryDirectory() as tmp:
            view = Path(tmp)/'kernel'
            receipt = source.prepare_loquetier_kernel_view(NATIVE, view, cfg)
            self.assertFalse(receipt['compiled'])
            self.assertFalse(receipt['model_qualified'])
            self.assertEqual(receipt['model_config_sha256'], source.sha(cfg.read_bytes()))
            pre = (view/'third_party/flashinfer/include/flashinfer/prefill.cuh').read_text()
            self.assertNotIn('(8 / group_size)', pre)
            self.assertNotIn('(16 / group_size)', pre)
            self.assertIn('(qo_idx_base + fx * 16', pre)
            self.assertIn('qo_idx % group_size', pre)
            self.assertTrue((view/'csrc/smlm_forward/smlm_forward_cutlass.cu').is_symlink())
            self.assertFalse((view/'csrc/flashinfer_adapter/generated').exists())
            self.assertFalse((view/'src/loquetier/_build_meta.py').exists())
            with self.assertRaises(ValueError):
                source.prepare_loquetier_kernel_view(NATIVE, view, cfg)
        self.assertEqual(before, {name:source.sha((native_headers/name).read_bytes()) for name in before})
        with self.assertRaises(ValueError):
            source.loquetier_llama3_kernel_sources(NATIVE, {'rope_scaling': None})

    def test_remove_only_observers_recovers_exact_native_ast(self):
        measured = source.loquetier_measurement_sources(NATIVE)['model_generation.py']
        original = (NATIVE/'src/loquetier_src/model_generation.py').read_text()
        self.assertEqual(ast.dump(RemoveObservations().visit(ast.parse(measured))),
                         ast.dump(ast.parse(original)))
        self.assertEqual(measured.count('loquetier_token_event(self, r,'), 1)
        self.assertNotIn('cuda.synchronize', measured)

    def test_source_drift_rejected_before_view_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'native'
            for relative in source.LOQUETIER_OBSERVATION_SOURCES:
                p = root/'src/loquetier_src'/relative
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes((NATIVE/'src/loquetier_src'/relative).read_bytes())
            p = root/'src/loquetier_src/model_generation.py'
            p.write_text(p.read_text()+'\n# drift\n')
            with self.assertRaisesRegex(ValueError, 'source drift'):
                source.prepare_loquetier_measurement_view(root, Path(tmp)/'view', MAIN)
            self.assertFalse((Path(tmp)/'view').exists())

    def test_exclusive_view_preserves_originals_and_links_only_unmodified_files(self):
        before = {p: source.sha((NATIVE/'src/loquetier_src'/p).read_bytes())
                  for p in source.LOQUETIER_OBSERVATION_SOURCES}
        with tempfile.TemporaryDirectory() as tmp:
            view = Path(tmp)/'view'
            receipt = source.prepare_loquetier_measurement_view(NATIVE, view, MAIN)
            self.assertFalse(receipt['performance_run_authorized'])
            self.assertFalse(receipt['added_per_token_cuda_synchronization'])
            self.assertFalse((view/'loquetier_src/model_generation.py').is_symlink())
            self.assertTrue((view/'loquetier_src/models/mixed_lora.py').is_symlink())
            self.assertEqual(receipt['measured_sha256']['model_generation.py'],
                source.sha((view/'loquetier_src/model_generation.py').read_bytes()))
            with self.assertRaises(FileExistsError):
                source.prepare_loquetier_measurement_view(NATIVE, view, MAIN)
        self.assertEqual(before, {p: source.sha((NATIVE/'src/loquetier_src'/p).read_bytes())
                                 for p in before})


@unittest.skipUnless(os.environ.get('TC_TEST_LOQUETIER_NATIVE') == '1',
                     'explicit existing Loquetier native environment required')
class NativeLoquetierObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This process imports only this new view, never a second native package.
        cls.tmp = tempfile.TemporaryDirectory()
        view = Path(cls.tmp.name)/'view'
        source.prepare_loquetier_measurement_view(NATIVE, view, MAIN)
        sys.path.insert(0, str(view))
        import torch
        from transformers import GenerationConfig, LogitsProcessorList
        from loquetier_src.model_generation import ModelGeneration, InputRequest, InferStatus
        from loquetier_src.models.mixed_lora import (
            MixedLoraModel, MixedLoraLinear, MixedLoraContext, NamedLoraLayer)
        from loquetier_src import tc_measurement
        import loquetier_src.model_generation as native
        if Path(native.__file__).resolve() != view/'loquetier_src/model_generation.py':
            raise RuntimeError('native package imported before the explicit observation view')
        cls.torch, cls.GenerationConfig = torch, GenerationConfig
        cls.LogitsProcessorList, cls.ModelGeneration = LogitsProcessorList, ModelGeneration
        cls.InputRequest, cls.InferStatus = InputRequest, InferStatus
        cls.MixedLoraModel, cls.MixedLoraLinear = MixedLoraModel, MixedLoraLinear
        cls.MixedLoraContext, cls.NamedLoraLayer = MixedLoraContext, NamedLoraLayer
        cls.observer, cls.native = tc_measurement, native

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def fixture(self):
        t = self.torch
        model = object.__new__(self.MixedLoraModel)
        t.nn.Module.__init__(model)
        model.dtype, model.device, model.rank = t.float16, t.device('cpu'), 2
        model.training_loras = set()
        model.lora_name_to_id = {'a': 0, 'b': 1}
        model.root_context = self.MixedLoraContext()
        layers = [self.NamedLoraLayer(n, t.zeros((2, 4), dtype=t.float16),
            t.zeros((4, 2), dtype=t.float16), t.nn.Identity()) for n in ('a', 'b')]
        model.adapter_lists = {a.adapter_name: [a] for a in layers}
        model.target_module_saves = {'q': layers}
        model.model = t.nn.Module()
        model.model.q = self.MixedLoraLinear(model.root_context,
            t.nn.Linear(4, 4, bias=False, dtype=t.float16), layers, 2, t.float16, True)
        gen = object.__new__(self.ModelGeneration)
        gen.model, gen.device = model, t.device('cpu')
        gen.generation_config = self.GenerationConfig(do_sample=False, num_beams=1, eos_token_id=None)
        gen.mode, gen.eos_token_ids = gen.generation_config.get_generation_mode(), [None]
        gen.requests, gen.prefill_requests, gen.eval_requests, gen.train_requests = [], [], [], []
        gen.finished_mark_ids, gen.output_with_mark_ids = set(), {}
        gen.decode_request_changed = False
        gen.new_decode_request, gen.cached_decode_kv = False, []
        receipt = self.observer.loquetier_bind_static_registry(gen, {'a': '/artifacts/a', 'b': '/artifacts/b'})
        self.assertTrue(receipt['pointer_tables_verified'])
        self.assertFalse(receipt['numerical_correctness_qualified'])
        return gen

    def request(self, mark=1, target=3, aid='a'):
        r = self.InputRequest(self.InferStatus.Prefill,
            self.torch.tensor([1, 5, 6], dtype=self.torch.long), [1, 5, 6], mark, aid,
            max_length=3+target, logits_processor=self.LogitsProcessorList())
        r._tc_request_id = f'request-{mark}'
        r.kvcache = NS(releases=0)
        def release():
            r.kvcache.releases += 1
        r.kvcache.release = release
        return r

    def register(self, gen, requests):
        self.observer.loquetier_register_requests(gen, requests)
        for r in requests:
            r.lora_id = gen.model._get_lora_id(r.lora_name)
        gen.requests = requests

    def step(self, gen, tokens):
        ids, lengths = [], []
        for r in gen.requests:
            if not ids or ids[-1] != r.lora_id:
                ids.append(r.lora_id)
                lengths.append(len(r.next_input_ids))
            else:
                lengths[-1] += len(r.next_input_ids)
        gen.model.root_context.update(lora_ids=ids)
        self.observer.loquetier_forward_started(gen, ids, lengths)
        gen.output_with_mark_ids, gen.finished_mark_ids = {}, set()
        gen.post_process(self.torch.tensor(tokens, dtype=self.torch.long))

    def test_actual_measured_postprocess_fixed_lengths_and_native_timing(self):
        for target in (1, 2, 256):
            gen, r = self.fixture(), self.request(target=target)
            self.register(gen, [r])
            for _ in range(target):
                self.step(gen, [2])
            state = r._tc_loquetier
            self.assertEqual(state['output_ids'], [2]*target)
            self.assertEqual(state['forward_count'], target)
            self.assertTrue(state['finished'])
            self.assertEqual(gen.requests, [])
            self.assertEqual(r.kvcache.releases, 1)
            self.assertLessEqual(state['registered_monotonic_s'], state['scheduled_monotonic_s'])
            self.assertLessEqual(state['scheduled_monotonic_s'], state['first_token_monotonic_s'])
            self.assertLessEqual(state['first_token_monotonic_s'], state['last_token_monotonic_s'])
            if target == 1:
                self.assertEqual(state['first_token_monotonic_s'], state['last_token_monotonic_s'])

    def test_mixed_batch_terminal_sort_and_identity(self):
        gen, a, b = self.fixture(), self.request(1, 3, 'b'), self.request(2, 1, 'a')
        self.register(gen, [a, b])
        self.step(gen, [21, 22])
        self.assertEqual([r.mark_id for r in gen.requests], [1])
        self.step(gen, [23])
        self.step(gen, [24])
        self.assertEqual(a._tc_loquetier['output_ids'], [21, 23, 24])
        self.assertEqual(b._tc_loquetier['output_ids'], [22])
        self.assertEqual(a._tc_loquetier['adapter_index'], 1)
        self.assertEqual(b._tc_loquetier['adapter_index'], 0)

    def test_native_terminal_consumed_once_and_process_identity_required(self):
        gen, r = self.fixture(), self.request(target=2)
        self.register(gen, [r])
        process = self.observer._slora_process_identity(os.getpid())
        with self.assertRaises(ValueError):
            self.observer.loquetier_request_result(gen, 1, time.perf_counter(), process)
        self.step(gen, [10])
        self.step(gen, [11])
        with self.assertRaises(ValueError):
            self.observer.loquetier_request_result(gen, 1, time.perf_counter(), dict(process, pid=1))
        result = self.observer.loquetier_request_result(gen, 1, time.perf_counter(), process)
        self.assertEqual(result['completion_token_ids'], [10, 11])
        self.assertEqual(result['native_forward_count'], 2)
        self.assertEqual(result['native_adapter_index'], 0)
        self.assertEqual(result['native_tpot_ms'], result['native_decode_ms'])
        self.assertFalse(result['lora_numerical_correctness_qualified'])
        with self.assertRaises(KeyError):
            self.observer.loquetier_request_result(gen, 1, time.perf_counter(), process)

    def test_cancel_releases_only_selected_cache_and_preserves_partial_tokens(self):
        gen, a, b = self.fixture(), self.request(1, 3), self.request(2, 3, 'b')
        self.register(gen, [a, b])
        self.step(gen, [10, 11])
        gen.cached_decode_kv = [a.kvcache, b.kvcache]
        cancelled = self.observer.loquetier_cancel_request(gen, 1)
        self.assertEqual(cancelled['completion_token_ids'], [10])
        self.assertFalse(cancelled['native_terminal_observed'])
        self.assertEqual(a.kvcache.releases, 1)
        self.assertEqual(b.kvcache.releases, 0)
        self.assertEqual(gen.cached_decode_kv, [b.kvcache])
        self.assertEqual(gen.requests, [b])
        self.step(gen, [12])
        self.step(gen, [13])
        self.assertEqual(b._tc_loquetier['output_ids'], [11, 12, 13])
        self.assertEqual(b.kvcache.releases, 1)
        with self.assertRaises(KeyError):
            self.observer.loquetier_cancel_request(gen, 1)

    def test_cancel_prefill_before_forward_is_not_success(self):
        gen, r = self.fixture(), self.request()
        self.register(gen, [r])
        gen.requests, gen.prefill_requests = [], [r]
        result = self.observer.loquetier_cancel_request(gen, r.mark_id)
        self.assertEqual(result['native_output_tokens'], 0)
        self.assertIsNone(result['native_first_token_monotonic_s'])
        self.assertEqual(gen.requests_len, 0)
        self.assertEqual(gen.cached_decode_kv, [])
        self.assertEqual(r.kvcache.releases, 1)

    def test_native_empty_batch_stale_cache_reproduction_and_lifecycle_adaptation(self):
        from loquetier.utils import KvPool
        for adapt in (False, True):
            gen = self.fixture()
            gen.kvpool = KvPool(1, 1, 2, 16, self.torch.float16, self.torch.device('cpu'))
            first = self.request(1, 2)
            gen.add_requests([first])  # Actual native KV, not the request fixture's mock.
            gen.pre_process()
            self.step(gen, [10])
            gen.pre_process()
            self.step(gen, [11])
            self.assertEqual(gen.requests_len, 0)
            self.assertEqual(gen.kvpool.num_pages, 0)
            self.assertEqual(gen.cached_decode_kv, [first.kvcache])
            if adapt:
                self.assertEqual(self.observer.loquetier_clear_released_decode_cache(gen), 1)
            second = self.request(2, 1)
            gen.add_requests([second])
            gen.pre_process()
            # Original pre_process grows a page for the already-ended request.
            self.assertEqual(first.kvcache.num_pages, 0 if adapt else 1)
            self.assertEqual(gen.kvpool.num_pages, 1 if adapt else 2)
            first.kvcache.release()
            second.kvcache.release()

    def test_actual_native_bridge_uses_owning_thread_and_preserves_work(self):
        import threading
        from scripts.replay_vllm_engine_trace import LoquetierNativeBatchBridge
        from loquetier.utils import KvPool
        gen = self.fixture()
        gen.kvpool = KvPool(1, 1, 2, 16, self.torch.float16, self.torch.device('cpu'))
        def request_factory(mark, rid, aid, ids, target):
            request = self.request(mark, target, aid)
            request._tc_request_id = rid
            request.input_ids = self.torch.tensor(ids, dtype=self.torch.long)
            request.next_input_ids, request.max_length = ids, len(ids)+target
            return request
        def generate():
            gen.pre_process()
            self.step(gen, [20+r.mark_id for r in gen.requests])
        # Only model arithmetic is replaced by controlled integer tokens;
        # real add/pre/post/KV/observations and the real bridge execute.
        gen.generate = generate
        bridge = LoquetierNativeBatchBridge(gen, request_factory, 2)
        a = bridge.add('a-request', 'a', [1, 2, 3], 2)
        b = bridge.add('b-request', 'b', [1, 2, 3], 3)
        with self.assertRaises(ValueError):
            bridge.add('over-capacity', 'a', [1], 2)
        self.assertEqual(bridge.step(), {})
        result = bridge.step()
        self.assertEqual(result[a]['completion_token_ids'], [20+a]*2)
        cancelled = bridge.cancel(b)
        self.assertEqual(cancelled['native_output_tokens'], 2)
        self.assertEqual(gen.kvpool.num_pages, 0)
        failures = []
        def foreign_thread():
            try:
                bridge.step()
            except ValueError as exc:
                failures.append(str(exc))
        thread = threading.Thread(target=foreign_thread)
        thread.start()
        thread.join()
        self.assertEqual(len(failures), 1)
        c = bridge.add('later-request', 'a', [1, 2, 3], 2)
        self.assertEqual(bridge.step(), {})
        self.assertEqual(bridge.step()[c]['native_output_tokens'], 2)
        self.assertEqual(gen.cached_decode_kv, [])
        self.assertEqual(gen.kvpool.num_pages, 0)
        self.assertEqual(bridge.close(), [])

    def test_capacity_schedules_share_native_registry_without_reusing_request_ids(self):
        from scripts.replay_vllm_engine_trace import LoquetierNativeBatchBridge
        from Loquetier_project.scripts.run_loquetier_trace_gate import _capacity_mixed_forward
        from loquetier.utils import KvPool
        gen = self.fixture()
        gen.kvpool = KvPool(1, 1, 2, 16, self.torch.float16, self.torch.device('cpu'))
        def factory(mark, rid, aid, ids, target):
            request = self.request(mark, target, aid)
            request._tc_request_id = rid
            request.input_ids = self.torch.tensor(ids, dtype=self.torch.long)
            request.next_input_ids, request.max_length = list(ids), len(ids)+target
            return request
        def generate():
            gen.pre_process()
            self.step(gen, [8+r.lora_id for r in gen.requests])
        # Model arithmetic alone is synthetic; registration, native lists/KV,
        # postprocessing, observations, bridge and capacity helper are real.
        gen.generate = generate
        bridge = LoquetierNativeBatchBridge(gen, factory, 2)
        cases = [dict(source_request_id='source-'+aid, adapter_id=aid,
            prompt_ids=[1, 2, 3], target_tokens=3, retained_output_ids=[8+i]*3)
            for i, aid in enumerate(('a', 'b'))]
        for staggered in (False, True):
            rows = _capacity_mixed_forward(bridge, cases, lambda *_: None, staggered)
            self.assertEqual([r['source_request_id'] for r in rows], ['source-a', 'source-b'])
            self.assertTrue(all(r['matches_retained_output'] for r in rows))
            self.assertEqual(gen.kvpool.num_pages, 0)
            self.assertEqual(gen._tc_loquetier_active, {})
        self.assertEqual(len(gen._tc_loquetier_request_ids), 4)
        self.assertEqual(gen._tc_loquetier_seen, {0, 1, 2, 3})
        # Repeating a schedule is still rejected: do not reset the native guard.
        with self.assertRaisesRegex(ValueError, 'fixed input/static binding contract'):
            _capacity_mixed_forward(bridge, cases, lambda *_: None, False)
        self.assertEqual(gen.kvpool.num_pages, 0)
        self.assertEqual(gen._tc_loquetier_active, {})
        self.assertEqual(bridge.close(), [])

    def test_capacity_score_observer_preserves_actual_greedy_tensor_and_native_prefix(self):
        from Loquetier_project.scripts.run_loquetier_trace_gate import _capacity_score_observer
        gen, r, other = self.fixture(), self.request(1, 3, 'a'), self.request(2, 3, 'b')
        self.register(gen, [r, other])
        self.step(gen, [8, 9])
        gen.model.root_context.update(lora_ids=[0, 1])
        self.observer.loquetier_forward_started(gen, [0, 1], [1, 1])
        outputs = []
        def greedy(scores):
            result = self.torch.argmax(scores, dim=-1)
            outputs.append(result)
            return result
        gen.generation_strategy = greedy
        scores = self.torch.zeros((2, 12), dtype=self.torch.float32)
        scores[0, 8:10], scores[1, 9] = 2, 3
        before = scores.clone()
        target = dict(case=dict(source_request_id='source-a', adapter_id='a',
            prompt_ids=[1, 5, 6], retained_output_ids=[8, 8, 8]), token_index=1,
            retained_token=8, observed_token=9)
        with tempfile.TemporaryDirectory() as tmp:
            records = []
            with _capacity_score_observer(gen, target, Path(tmp), 'case', records):
                chosen = gen.generation_strategy(scores)
                self.assertIs(chosen, outputs[0])
                self.assertTrue(self.torch.equal(scores, before))
                self.assertEqual(chosen.tolist(), [8, 9])
            self.assertIs(gen.generation_strategy, greedy)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertTrue(record['prefix_matches_retained'])
            self.assertEqual(record['actual_prefix_token_ids'], [8])
            self.assertEqual(record['native_segments'], [0, 1])
            self.assertEqual(record['maximum_tie_count'], 2)
            self.assertEqual(record['top_two_margin'], 0)
            payload = Path(record['score_path']).read_bytes()
            self.assertEqual(payload, before[0].numpy().astype('<f4', copy=False).tobytes())
            self.assertEqual(hashlib.sha256(payload).hexdigest(), record['score_sha256'])
            self.assertEqual(record['score_bytes'], 48)
            # Same path cannot silently overwrite the first observed vector.
            with self.assertRaises(FileExistsError):
                with _capacity_score_observer(gen, target, Path(tmp), 'case', []):
                    gen.generation_strategy(scores)
            self.assertIs(gen.generation_strategy, greedy)
            # Native input/prefix inconsistency is rejected, not repaired.
            r._tc_loquetier['output_ids'][0] = 10
            with self.assertRaisesRegex(ValueError, 'identity/prefix'):
                with _capacity_score_observer(gen, target, Path(tmp), 'bad-prefix', []):
                    gen.generation_strategy(scores)
            self.assertIs(gen.generation_strategy, greedy)
            r._tc_loquetier['output_ids'][0] = 8
            bad = scores.clone()
            bad[0, 1] = float('nan')
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                with _capacity_score_observer(gen, target, Path(tmp), 'nonfinite', []):
                    gen.generation_strategy(bad)
            self.assertIs(gen.generation_strategy, greedy)
        gen.post_process(chosen)
        self.assertEqual(r._tc_loquetier['output_ids'], [8, 8])
        self.assertEqual(other._tc_loquetier['output_ids'], [9, 9])

    def test_capacity_multi_score_observer_keeps_two_native_rows_and_original_decision(self):
        from Loquetier_project.scripts.run_loquetier_trace_gate import _capacity_score_observer
        gen, a, b = self.fixture(), self.request(1, 3, 'a'), self.request(2, 3, 'b')
        self.register(gen, [a, b])
        self.step(gen, [8, 9])
        gen.model.root_context.update(lora_ids=[0, 1])
        self.observer.loquetier_forward_started(gen, [0, 1], [1, 1])
        decisions = []
        def greedy(scores):
            decisions.append(self.torch.argmax(scores, dim=-1))
            return decisions[-1]
        gen.generation_strategy = greedy
        targets = [dict(case=dict(source_request_id='source-'+aid, adapter_id=aid,
            prompt_ids=[1, 5, 6], retained_output_ids=[prefix, 8, 8]), token_index=1,
            retained_token=8, observed_token=9) for aid, prefix in [('a', 8), ('b', 9)]]
        scores = self.torch.zeros((2, 12), dtype=self.torch.float32)
        scores[0, 8:10], scores[1, 9] = 2, 3
        before = scores.clone()
        with tempfile.TemporaryDirectory() as tmp:
            records = []
            with _capacity_score_observer(gen, targets, Path(tmp), 'both', records):
                chosen = gen.generation_strategy(scores)
                self.assertIs(chosen, decisions[0])
                self.assertTrue(self.torch.equal(scores, before))
            self.assertIs(gen.generation_strategy, greedy)
            self.assertEqual([r['source_request_id'] for r in records], ['source-a', 'source-b'])
            self.assertEqual([r['selected_token'] for r in records], [8, 9])
            self.assertTrue(all(r['prefix_matches_retained'] for r in records))
            for i, record in enumerate(records):
                self.assertEqual(Path(record['score_path']).read_bytes(), before[i].numpy().astype('<f4', copy=False).tobytes())
            for invalid in ([], [targets[0], targets[0]]):
                with self.assertRaisesRegex(ValueError, 'empty or ambiguous'):
                    with _capacity_score_observer(gen, invalid, Path(tmp), 'invalid', []):
                        self.fail('ambiguous target must fail before changing strategy')
                self.assertIs(gen.generation_strategy, greedy)
        gen.post_process(chosen)
        self.assertEqual(a._tc_loquetier['output_ids'], [8, 8])
        self.assertEqual(b._tc_loquetier['output_ids'], [9, 9])

    def test_empty_batch_cleanup_rejects_live_pages_and_leaves_active_batch_unchanged(self):
        gen = self.fixture()
        live = NS(num_pages=1, seqlen=1)
        gen.cached_decode_kv = [live]
        with self.assertRaisesRegex(ValueError, 'live decode pages'):
            self.observer.loquetier_clear_released_decode_cache(gen)
        self.assertEqual(gen.cached_decode_kv, [live])
        gen.requests = [self.request()]
        self.assertEqual(self.observer.loquetier_clear_released_decode_cache(gen), 0)
        self.assertEqual(gen.cached_decode_kv, [live])

    def test_unknown_and_duplicate_rejected_atomically_before_native_allocation(self):
        for requests in ([self.request(), self.request(2, aid='missing')],
                         [self.request(), self.request()],
                         [self.request(target=257)]):
            gen = self.fixture()
            with self.assertRaises(ValueError):
                gen.add_requests(requests)  # Actual measured native entry.
            self.assertEqual(gen._tc_loquetier_active, {})
            self.assertFalse(any(hasattr(r, '_tc_loquetier') for r in requests))

    def test_wrong_adapter_or_segment_cannot_create_token_evidence(self):
        gen, r = self.fixture(), self.request()
        self.register(gen, [r])
        gen.model.root_context.update(lora_ids=[0])
        with self.assertRaises(ValueError):
            self.observer.loquetier_forward_started(gen, [0], [999])
        self.assertEqual(r._tc_loquetier['forward_count'], 0)
        r.lora_id = 1
        with self.assertRaises(ValueError):
            self.observer.loquetier_forward_started(gen, [1], [3])
        self.assertEqual(r._tc_loquetier['output_ids'], [])

    def test_pointer_table_and_module_mapping_rejected_at_startup(self):
        for change in ('pointer', 'layout'):
            gen = self.fixture()
            del gen._tc_loquetier_registry
            if change == 'pointer':
                gen.model.model.q.wa_ptrs[0] = 0
            else:
                gen.model.target_module_saves['q'].reverse()
            with self.assertRaises(ValueError):
                self.observer.loquetier_bind_static_registry(gen, {'a': '/artifacts/a', 'b': '/artifacts/b'})

    def test_early_terminal_duplicate_and_backward_token_rejected(self):
        gen, r = self.fixture(), self.request(target=2)
        self.register(gen, [r])
        self.step(gen, [20])
        state = copy.deepcopy(r._tc_loquetier)
        with self.assertRaises(ValueError):
            self.observer.loquetier_token_event(gen, r, 20, state['last_token_monotonic_s'])
        self.assertEqual(state, r._tc_loquetier)
        gen.finished_mark_ids = {r.mark_id}
        with self.assertRaises(ValueError):
            self.observer.loquetier_finished(gen)
        gen.model.root_context.update(lora_ids=[0])
        self.observer.loquetier_forward_started(gen, [0], [1])
        with self.assertRaises(ValueError):
            self.observer.loquetier_token_event(gen, r, 20, state['last_token_monotonic_s']-1.)


if __name__ == '__main__':
    unittest.main()
