"""Opt-in CPU checks of the installed, unmodified native generation rules.

Run in loquetier_20260520 with TC_TEST_LOQUETIER_NATIVE=1. These tests do not
perform model arithmetic, qualify GPU memory, or supply latency measurements.
"""
import os
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


if __name__ == '__main__':
    unittest.main()
