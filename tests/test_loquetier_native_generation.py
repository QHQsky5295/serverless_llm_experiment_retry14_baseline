"""Opt-in CPU checks of the installed, unmodified native generation rules.

Run in loquetier_20260520 with TC_TEST_LOQUETIER_NATIVE=1. These tests do not
perform model arithmetic, qualify GPU memory, or supply latency measurements.
"""
import os
import gc
import weakref
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


@unittest.skipUnless(os.environ.get('TC_TEST_LOQUETIER_NATIVE') == '1',
                     'requires the explicitly selected existing Loquetier native environment')
class NativeGenerationRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vendor = Path(__file__).resolve().parents[1]/'vendor_new_baselines/Loquetier_main_20260520/src'
        sys.path.insert(0, str(vendor))
        import torch
        from transformers import GenerationConfig
        from transformers.generation.utils import GenerationMode
        from loquetier_src.model_generation import InputRequest, ModelGeneration
        from loquetier_src.models.mixed_lora import MixedLoraModel
        from loquetier_src.enums import InferStatus
        cls.torch, cls.GenerationConfig, cls.GenerationMode = torch, GenerationConfig, GenerationMode
        cls.InputRequest, cls.ModelGeneration, cls.MixedLoraModel = InputRequest, ModelGeneration, MixedLoraModel
        cls.InferStatus = InferStatus

    def native_request(self, mark, target, adapter_id):
        request = self.InputRequest(self.InferStatus.Prefill,
            self.torch.tensor([1, 10, 11], dtype=self.torch.long), [1, 10, 11],
            mark, f'adapter-{adapter_id}', lora_id=adapter_id, max_length=3+target)
        request.kvcache = SimpleNamespace(releases=0)
        def release():
            request.kvcache.releases += 1
        request.kvcache.release = release
        return request

    def native_generation(self, requests):
        model = object.__new__(self.ModelGeneration)
        model.requests = requests
        model.eos_token_ids = [None]
        model.output_with_mark_ids = {}
        model.finished_mark_ids = set()
        model.decode_request_changed = False
        return model

    def test_native_max_length_exactly_counts_1_2_256_with_eos_disabled(self):
        for target in (1, 2, 256):
            with self.subTest(target=target):
                request = self.native_request(7, target, 3)
                model = self.native_generation([request])
                count = 0
                while model.requests:
                    model.output_with_mark_ids = {}
                    model.finished_mark_ids = set()
                    # Native post_process sees the ordinary EOS integer, but
                    # disabled EOS must retain the declared fixed token count.
                    model.post_process(self.torch.tensor([2], dtype=self.torch.long))
                    count += len(model.output_with_mark_ids[7])
                    self.assertLessEqual(count, target)
                self.assertEqual(count, target)
                self.assertEqual(model.finished_mark_ids, {7})
                self.assertEqual(request.kvcache.releases, 1)

    def test_batch_completion_and_sort_preserve_native_adapter_ids(self):
        first, second = self.native_request(1, 1, 5), self.native_request(2, 3, 2)
        model = self.native_generation([first, second])
        model.post_process(self.torch.tensor([20, 21], dtype=self.torch.long))
        self.assertEqual(model.finished_mark_ids, {1})
        self.assertEqual(model.output_with_mark_ids, {1: [20], 2: [21]})
        self.assertEqual([(r.mark_id, r.lora_id) for r in model.requests], [(2, 2)])
        self.assertEqual(first.kvcache.releases, 1)
        self.assertEqual(second.kvcache.releases, 0)

    def test_native_eos_and_greedy_configuration(self):
        cfg = self.GenerationConfig(do_sample=False, num_beams=1,
                                    eos_token_id=None, max_length=1024)
        self.assertEqual(cfg.get_generation_mode(), self.GenerationMode.GREEDY_SEARCH)
        self.assertIsNone(cfg.eos_token_id)

    def test_documented_unknown_adapter_fallback_requires_wrapper_rejection(self):
        model = object.__new__(self.MixedLoraModel)
        model.lora_name_to_id = {'known': 0, 'other': 1}
        self.assertEqual(model._get_lora_id('missing'), 0)
        self.assertNotIn('missing', model.lora_name_to_id)
        self.assertEqual(model._get_lora_id('other'), 1)

    def _prepared_owner_witness(self, raw_dtype, expect_storage_alias, *,
                                reject_training=False, retain_owner=False):
        """Actual virtual/PEFT/preparation ownership, CPU synthetic tensors only.

        The reference products below are NOT SMLM or model-forward qualification.
        In particular, releasing a Python Parameter need not free its storage:
        native preparation can keep an FP16 view of the same allocation.
        """
        from peft import LoraConfig, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM
        from loquetier_src import create_virtual_model
        t = self.torch
        with t.random.fork_rng(devices=[]):
            t.manual_seed(559)
            base = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=2, max_position_embeddings=32)).to(dtype=t.float16)
            original_base = {n: (p.data_ptr(), p.detach().clone())
                             for n, p in base.named_parameters()}

            def make_model(rank):
                model = get_peft_model(create_virtual_model(base), LoraConfig(
                    task_type='CAUSAL_LM', r=rank, lora_alpha=2*rank,
                    target_modules=['q_proj'], bias='none', inference_mode=True))
                # Explicit nonzero synthetic fixture; no experimental pool changes.
                for name, parameter in model.named_parameters():
                    if '.lora_' in name:
                        parameter.data = t.randn(parameter.shape, dtype=raw_dtype)
                return model

            models = [make_model(1), make_model(2)]
            def owner_refs():
                objects = [obj for model in models
                    for obj in (model, model.base_model, model.base_model.model)]
                parameters = [p for model in models for n, p in model.named_parameters()
                              if '.lora_' in n]
                self.assertEqual(len(parameters), 4)
                self.assertTrue(all(p.dtype == raw_dtype for p in parameters))
                return ([weakref.ref(x) for x in objects],
                        [weakref.ref(x) for x in parameters],
                        {p.untyped_storage().data_ptr() for p in parameters})
            object_refs, parameter_refs, raw_storage = owner_refs()
            from Loquetier_project.scripts.run_loquetier_trace_gate import _ownership_refs
            probed_owners, probed_parameters, probed_storage, probed_dtypes = _ownership_refs(models)
            self.assertEqual(len(probed_owners), len(object_refs))
            self.assertEqual(len(probed_parameters), len(parameter_refs))
            self.assertEqual(set(probed_storage), raw_storage)
            self.assertEqual(probed_dtypes, {str(raw_dtype): 4})
            mixed = self.MixedLoraModel(create_virtual_model(base),
                {f'adapter-{i}': m.base_model for i, m in enumerate(models)},
                dtype=t.float16, device=t.device('cpu'), apply_scaling='A')
            self.assertEqual(mixed.lora_name_to_id, {'adapter-0': 0, 'adapter-1': 1})
            self.assertFalse(mixed.training_loras)
            layers = mixed.target_module_saves['model.layers.0.self_attn.q_proj']
            prepared = [w for layer in layers for w in (layer.wa, layer.wb)]
            pointers = [w.data_ptr() for w in prepared]
            contents = [w.clone() for w in prepared]
            aliases = raw_storage & {w.untyped_storage().data_ptr() for w in prepared}
            self.assertEqual(bool(aliases), expect_storage_alias)
            linear = mixed.model.model.layers[0].self_attn.q_proj
            self.assertEqual(linear.wa_ptrs.tolist(), [a.wa.data_ptr() for a in layers])
            self.assertEqual(linear.wb_ptrs.tolist(), [a.wb.data_ptr() for a in layers])
            x = t.randn((3, 16), dtype=t.float32)
            def reference_products():
                # CPU reference arithmetic only; GPU native forward is a later gate.
                return [t.nn.functional.linear(x, linear.base_layer.weight.float())
                    + t.nn.functional.linear(t.nn.functional.linear(x, a.wa.float()),
                                              a.wb.float()) for a in layers]
            before = reference_products()
            from scripts.replay_vllm_engine_trace import release_loquetier_inference_owners
            if reject_training:
                mixed.training_loras.add('adapter-0')
                with self.assertRaisesRegex(ValueError, 'pure-inference'):
                    release_loquetier_inference_owners(models, base, mixed)
                self.assertEqual(len(models), 2)
                mixed.training_loras.clear()
            if retain_owner:
                extra_owner = models[0]
                with self.assertRaisesRegex(ValueError, 'owners remain'):
                    release_loquetier_inference_owners(models, base, mixed)
                del extra_owner
                gc.collect()
            else:
                receipt = release_loquetier_inference_owners(models, base, mixed)
                self.assertEqual(receipt['remaining_owners'], 0)
                self.assertEqual(receipt['remaining_parameters'], 0)
                self.assertTrue(receipt['retained_layout_unchanged'])
                self.assertFalse(receipt['runtime_content_hash_checked'])
                self.assertEqual(receipt['owner_count'], 6)
                self.assertEqual(receipt['parameter_count'], 4)
                self.assertEqual(bool(receipt['aliased_storage_bytes']), expect_storage_alias)
                self.assertEqual(receipt['raw_storage_bytes'],
                    receipt['releasable_storage_bytes']+receipt['aliased_storage_bytes'])
            self.assertEqual(models, [])
            self.assertTrue(all(ref() is None for ref in object_refs))
            self.assertTrue(all(ref() is None for ref in parameter_refs))
            self.assertTrue(all(ref() is None for ref in probed_owners + probed_parameters))
            self.assertEqual([w.data_ptr() for w in prepared], pointers)
            self.assertTrue(all(t.equal(w, old) for w, old in zip(prepared, contents)))
            self.assertTrue(all(t.equal(a, b) for a, b in zip(before, reference_products())))
            self.assertEqual(linear.wa_ptrs.tolist(), [a.wa.data_ptr() for a in layers])
            self.assertEqual(linear.wb_ptrs.tolist(), [a.wb.data_ptr() for a in layers])
            self.assertEqual(set(dict(base.named_parameters())), set(original_base))
            for name, parameter in base.named_parameters():
                ptr, content = original_base[name]
                self.assertEqual(parameter.data_ptr(), ptr)
                self.assertTrue(t.equal(parameter, content))

    def test_prepared_fp16_copy_survives_fp32_peft_owner_collection(self):
        self._prepared_owner_witness(self.torch.float32, expect_storage_alias=False)

    def test_prepared_alias_retains_storage_after_fp16_parameter_collection(self):
        self._prepared_owner_witness(self.torch.float16, expect_storage_alias=True)

    def test_training_rejected_before_owner_list_mutation(self):
        self._prepared_owner_witness(self.torch.float32, False, reject_training=True)

    def test_external_raw_owner_fails_closed_instead_of_claiming_release(self):
        self._prepared_owner_witness(self.torch.float32, False, retain_owner=True)

    def test_owner_inventory_tracks_content_storage_and_releases_input(self):
        from Loquetier_project.scripts.run_loquetier_trace_gate import _tensor_inventory
        tensor = self.torch.arange(24, dtype=self.torch.float16).reshape(4, 6)
        ref = weakref.ref(tensor)
        first = _tensor_inventory([('weight', tensor)])
        self.assertEqual(first['logical_bytes'], 48)
        self.assertEqual(first['unique_storage_bytes'], 48)
        self.assertEqual(first, _tensor_inventory([('weight', tensor)]))
        clone = _tensor_inventory([('weight', tensor.clone())])
        self.assertEqual(first['content_sha256'], clone['content_sha256'])
        self.assertNotEqual(first['layout_sha256'], clone['layout_sha256'])
        tensor[0, 0] += 1
        changed = _tensor_inventory([('weight', tensor)])
        self.assertNotEqual(first['content_sha256'], changed['content_sha256'])
        self.assertEqual(first['layout_sha256'], changed['layout_sha256'])
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            _tensor_inventory([('bad', tensor.T)])
        del tensor
        gc.collect()
        self.assertIsNone(ref())


if __name__ == '__main__':
    unittest.main()
