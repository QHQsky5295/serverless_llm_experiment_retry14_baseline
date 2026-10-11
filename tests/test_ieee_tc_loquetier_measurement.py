"""Source integrity plus opt-in CPU execution of the actual measured native methods.

No GPU arithmetic/performance evidence. Run native tests in loquetier_20260520
with TC_TEST_LOQUETIER_NATIVE=1; skipping those tests is not qualification.
"""
import ast
import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
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
