"""CPU equivalence only: does NOT qualify CUDA publication or model outputs."""
import hashlib
import importlib.util
from pathlib import Path
import random
import sys
import unittest

from faaslora.utils.lora_metadata import lower_dense_lora_metadata as lower


def native_request_mapping(ids=(7, 0, 9), order=(2, 0, 1), counts=(2, 3, 1)):
    """Actual installed V2 producer and native LoRAMapping; no model/GPU/file I/O."""
    import numpy as np
    import torch
    from vllm.lora.layers import LoRAMapping
    from vllm.lora.request import LoRARequest
    from vllm.v1.worker.gpu.lora_utils import LoraState
    path = Path(sys.modules[LoraState.__module__].__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != '1e90eddd9ddb4ca55e574bb28e87ae3a6f6750175a62d373102426543f4ccd66':
        raise RuntimeError('native request-mapping producer identity changed')
    if torch.cuda.is_initialized():
        raise RuntimeError('native CPU fixture must not initialize CUDA')
    state = LoraState(len(ids))
    for i, adapter in enumerate(ids):
        request = LoRARequest(str(adapter), adapter, '/unit-test-only/not-opened') if adapter else None
        state.add_request(str(i), i, request)
    prompt, tokens, active = state.make_lora_inputs(
        [str(i) for i in order], np.array(order, dtype=np.int32),
        np.array(counts, dtype=np.int32))
    return LoRAMapping(tokens, prompt, is_prefill=True), active


class DenseMetadataLowering(unittest.TestCase):
    def call(self, **changes):
        args = dict(index_mapping=(11, 11, 0, 7, 11), prompt_mapping=(7, 0, 11),
                    slot_layout=(7, None, 11), vocab_size=32000,
                    max_num_tokens=8, captured_lora_counts=(1, 2, 4))
        args.update(changes)
        return lower(**args)

    def test_exact_stable_rows_and_two_mappings(self):
        value = self.call()
        self.assertEqual(value.base_indices, (2, 2, -1, 0, 2))
        self.assertEqual(value.sampler_indices, (0, -1, 2))
        self.assertEqual(value.sampler_indices_padded, (0, 10, 8))
        self.assertEqual(value.embeddings_indices, (64000, 64000, 0, 0, 64000))
        self.assertEqual(value.indices_len, (5, 3, 3, 5))
        kernel = value.token_kernel
        self.assertEqual(kernel.stable_order, (2, 3, 0, 1, 4))
        self.assertEqual(kernel.active_slots, (-1, 0, 2, -1))
        self.assertEqual(kernel.counts, (1, 1, 3, 0))
        self.assertEqual(kernel.starts, (0, 1, 2, 5, 0))
        self.assertEqual(kernel.launch_groups, 4)

    def test_current_slot_layout_not_cached(self):
        old = self.call()
        new = self.call(slot_layout=(11, None, 7))
        self.assertNotEqual(old.base_indices, new.base_indices)
        self.assertEqual(new.base_indices, (0, 0, -1, 2, 0))
        self.assertEqual(old.base_indices, (2, 2, -1, 0, 2))

    def test_empty_all_base_and_negative_ids(self):
        for rows in ((), (0,), (-1, 0, -2)):
            value = self.call(index_mapping=rows, prompt_mapping=rows)
            self.assertTrue(value.token_kernel.no_lora)
            self.assertEqual(value.token_kernel.launch_groups, 0)
            self.assertEqual(value.token_kernel.slot_by_row, ())
            self.assertEqual(value.token_kernel.stable_order, ())
            self.assertEqual(value.token_kernel.counts, (0, 0, 0, 0))
            self.assertEqual(value.token_kernel.starts, (0, 0, 0, 0, 0))

    def test_no_specialization_and_base_group_count(self):
        self.assertEqual(self.call(captured_lora_counts=()).token_kernel.launch_groups, 3)
        self.assertEqual(self.call(captured_lora_counts=(4,)).token_kernel.launch_groups, 4)

    def test_missing_adapter_never_falls_back(self):
        for change in (dict(index_mapping=(23,)), dict(prompt_mapping=(23,))):
            with self.assertRaisesRegex(ValueError, 'no current physical slot'):
                self.call(**change)

    def test_exact_cpu_integer_representations(self):
        import numpy as np
        for dtype in (np.int8, np.int16, np.int32, np.int64,
                      np.uint8, np.uint16, np.uint32, np.uint64):
            with self.subTest(dtype=dtype):
                value = self.call(index_mapping=tuple(map(dtype, (11, 11, 0, 7, 11))),
                                  prompt_mapping=tuple(map(dtype, (7, 0, 11))),
                                  captured_lora_counts=tuple(map(dtype, (1, 2, 4))))
                self.assertEqual(value, self.call())
                self.assertTrue(all(type(x) is int for x in value.sampler_indices_padded))

    def test_reject_implicit_integer_coercions_without_device_reads(self):
        import numpy as np
        import torch
        from unittest.mock import patch
        class IndexOnly:
            def __index__(self):
                raise AssertionError('arbitrary index coercion is forbidden')
        values = (True, np.bool_(True), 7.0, np.float32(7), '7',
                  np.array(7), torch.tensor(7), IndexOnly(), np.timedelta64(7, 's'))
        with patch.object(torch.Tensor, '__index__', side_effect=AssertionError('tensor read')):
            for field in ('index_mapping', 'prompt_mapping', 'captured_lora_counts'):
                for value in values:
                    with self.subTest(field=field, value=type(value)), self.assertRaises(ValueError):
                        self.call(**{field: (value,)})

    def test_native_v2_producer_preserves_order_and_current_slots(self):
        import numpy as np
        mapping, active = native_request_mapping()
        self.assertEqual({item.lora_int_id for item in active}, {7, 9})
        self.assertTrue(all(type(x) is np.int32 for x in mapping.index_mapping + mapping.prompt_mapping))
        self.assertEqual(mapping.index_mapping, (9, 9, 7, 7, 7, 0))
        self.assertEqual(mapping.prompt_mapping, (9, 7, 0))
        for layout, base, prompt in (((7, 9), (1, 1, 0, 0, 0, -1), (1, 0, -1)),
                                     ((9, 7), (0, 0, 1, 1, 1, -1), (0, 1, -1))):
            value = self.call(index_mapping=mapping.index_mapping, prompt_mapping=mapping.prompt_mapping,
                              slot_layout=layout, captured_lora_counts=(1, 2, 3))
            self.assertEqual(value.base_indices, base)
            self.assertEqual(value.sampler_indices, prompt)
        with self.assertRaisesRegex(ValueError, 'no current physical slot'):
            self.call(index_mapping=mapping.index_mapping, slot_layout=(7, None), captured_lora_counts=())

    def test_native_v2_all_base_empty_and_full_slot_occupancy(self):
        for ids, order, counts, layout in (
                ((0, 0), (1, 0), (1, 2), (None, None)),
                ((7,), (), (), (7,)),
                (tuple(range(1, 15)), tuple(reversed(range(14))), (1,) * 14, tuple(range(1, 15)))):
            mapping, _ = native_request_mapping(ids, order, counts)
            args = dict(index_mapping=mapping.index_mapping, prompt_mapping=mapping.prompt_mapping,
                        slot_layout=layout, max_num_tokens=1024, captured_lora_counts=())
            value = self.call(**args)
            self.assertEqual(value, self.call(**{**args,
                'index_mapping': tuple(int(x) for x in mapping.index_mapping),
                'prompt_mapping': tuple(int(x) for x in mapping.prompt_mapping)}))

    def test_reject_mutable_ambiguous_and_out_of_capacity(self):
        invalid = [dict(index_mapping=[11]), dict(prompt_mapping=(True,)),
                   dict(slot_layout=(7, 11, 11)), dict(slot_layout=(False, 11)),
                   dict(slot_layout=()), dict(max_num_tokens=4),
                   dict(max_num_tokens=True), dict(max_num_tokens=0),
                   dict(max_num_tokens=2**31), dict(vocab_size=0),
                   dict(vocab_size=2**63), dict(captured_lora_counts=(2, 1)),
                   dict(captured_lora_counts=(1, 1)), dict(captured_lora_counts=(0,)),
                   dict(captured_lora_counts=(5,))]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.call(**change)


class InstalledNativeCPUOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        if not str(torch.__version__).startswith('2.13.') or torch.cuda.is_initialized():
            raise RuntimeError('use the qualified Torch2.13 environment, CUDA hidden')
        path = Path(sys.prefix) / 'lib/python3.12/site-packages/vllm/lora/ops/triton_ops/lora_kernel_metadata.py'
        if hashlib.sha256(path.read_bytes()).hexdigest() != '7669eeeb491e89df7ef4ff050a79d3934ce7a63b146fe93239f081d176c68881':
            raise RuntimeError('native oracle identity changed')
        spec = importlib.util.spec_from_file_location('_d520_native_metadata_oracle', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.native = module.LoRAKernelMeta

    def check_kernel(self, expected, rows, capacity, captured):
        torch = self.torch
        native = self.native.make(capacity, 80, 'cpu', list(captured))
        # Unused all-base buffers deliberately have a sentinel. Do not require
        # them to equal invented zeros or publish stale rows as active evidence.
        native.token_lora_mapping.fill_(12345)
        native.token_indices_sorted_by_lora_ids.fill_(12345)
        native.prepare_tensors(torch.tensor(rows, dtype=torch.long))
        self.assertEqual(native.no_lora_flag_cpu.item(), expected.no_lora)
        self.assertEqual(native.num_active_loras_cpu.item(), expected.launch_groups)
        self.assertEqual(tuple(native.active_lora_ids.tolist()), expected.active_slots)
        self.assertEqual(tuple(native.num_tokens_per_lora.tolist()), expected.counts)
        self.assertEqual(tuple(native.lora_token_start_loc.tolist()), expected.starts)
        if not expected.no_lora:
            self.assertEqual(tuple(native.token_lora_mapping[:len(rows)].tolist()), expected.slot_by_row)
            self.assertEqual(tuple(native.token_indices_sorted_by_lora_ids[:len(rows)].tolist()), expected.stable_order)
        else:
            self.assertTrue(torch.all(native.token_lora_mapping == 12345).item())

    def test_native_metadata_equivalence(self):
        # Bounded synthetic *unit-test metadata*, not a new workload or weights.
        rng = random.Random(520)
        for capacity in (1, 2, 4, 8):
            layout = tuple(100 + 3 * i if i % 3 != 1 else None for i in range(capacity))
            choices = (0, -1) + tuple(x for x in layout if x is not None)
            captures = ((), (capacity + 1,), tuple(n for n in range(1, capacity + 2)
                                                   if n & (n - 1) == 0 or n == capacity + 1))
            for captured in captures:
                for length in (0, 1, 7, 64):
                    index = tuple(rng.choice(choices) for _ in range(length))
                    prompt = tuple(rng.choice(choices) for _ in range(min(length, 8)))
                    value = lower(index_mapping=index, prompt_mapping=prompt, slot_layout=layout,
                                  vocab_size=32000, max_num_tokens=80, captured_lora_counts=captured)
                    # Independent base conversion follows native list-index semantics.
                    base = tuple(layout.index(x) if x > 0 else -1 for x in index)
                    sampler = tuple(layout.index(x) if x > 0 else -1 for x in prompt)
                    self.assertEqual(value.base_indices, base)
                    self.assertEqual(value.sampler_indices, sampler)
                    self.assertEqual(value.sampler_indices_padded,
                                     tuple(i + (capacity if x == -1 else x) * len(sampler)
                                           for i, x in enumerate(sampler)))
                    self.check_kernel(value.token_kernel, base, capacity, captured)
                    self.check_kernel(value.sampler_kernel, sampler, capacity, captured)
        self.assertFalse(self.torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
