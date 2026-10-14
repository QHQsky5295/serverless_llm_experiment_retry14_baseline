"""CPU ownership/byte-contract tests, NOT CUDA or inference qualification."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from faaslora.memory import lora_metadata_publication as m
from faaslora.utils.lora_metadata import lower_dense_lora_metadata


class Event:
    def __init__(self, blocking):
        assert blocking
        self.done = False
        self.waits = self.records = 0
        self.on_wait = lambda: None

    def query(self):
        return self.done

    def synchronize(self):
        self.on_wait()
        self.waits += 1
        self.done = True

    def record(self, stream):
        self.done = False
        self.records += 1


def fixture(rows=8, slots=2):
    stream = NS(device=torch.device('cpu'))
    cuda = NS(Event=Event, current_stream=lambda device: stream,
              is_current_stream_capturing=lambda: False)
    fake = NS(int64=torch.int64, int32=torch.int32, bool=torch.bool, device=torch.device,
              cuda=cuda, empty=lambda n, **kw: torch.empty(n, **{k: v for k, v in kw.items()
                                                              if k != 'pin_memory'}))
    wrapper = NS(indices_len=[None] * 4, is_prefill=False, update_metadata=lambda *a: None)
    for name in m.BASE:
        setattr(wrapper, name, torch.full((rows,), -999, dtype=torch.int64))
    for field in ('token_mapping_meta', 'prompt_mapping_meta'):
        meta = NS(captured_lora_counts=[1, 2, 3])
        for name, size in zip(m.KERNEL, (rows, rows, slots + 1, slots + 1, slots + 2)):
            setattr(meta, name, torch.full((size,), -999, dtype=torch.int32))
        meta.no_lora_flag_cpu = torch.tensor([False])
        meta.num_active_loras_cpu = torch.tensor([0], dtype=torch.int32)
        meta.default_num_active_loras_cpu = torch.tensor([slots + 1], dtype=torch.int32)
        setattr(wrapper, field, meta)
    publisher = m.DenseMetadataPublisher(wrapper, rows=rows, slots=slots, banks=2,
                                        stream=stream, torch_module=fake)
    return publisher, wrapper, fake


def mapping(ids=(7, 0, 9, 7), prompts=(9, 7)):
    return NS(index_mapping=ids, prompt_mapping=prompts, is_prefill=True)


class PublicationTests(unittest.TestCase):
    def test_native_v2_mapping_reaches_publisher_without_type_coercion_fixture(self):
        from tests.test_ieee_tc_lora_metadata_lowering import native_request_mapping
        p, w, _ = fixture()
        inp, _ = native_request_mapping()
        p.update_metadata(inp, [7, 9], 3, 32000)
        self.assertEqual(w._token_lora_indices[:6].tolist(), [1, 1, 0, 0, 0, -1])
        self.assertEqual(w._sampler_indices[:3].tolist(), [1, 0, -1])
        self.assertEqual(w._sampler_indices_padded[:3].tolist(), [3, 1, 8])
        self.assertEqual(w.token_mapping_meta.num_tokens_per_lora.tolist(), [1, 3, 2])
        self.assertEqual(p.updates, 1)
        self.assertFalse(p.poisoned)
        self.assertFalse(torch.cuda.is_initialized())

    def test_bytes_and_fixed_allocations(self):
        p, _, _ = fixture()
        self.assertEqual(p.staging_bytes, 2 * (48 * 8 + 24 * 2 + 32))
        pointers = [x.data_ptr() for x in p._slabs]
        for _ in range(9):
            p.update_metadata(mapping(), [7, 9], 3, 32000)
        self.assertEqual(pointers, [x.data_ptr() for x in p._slabs])
        self.assertEqual(p.reuse_waits, 7)
        self.assertEqual(p.updates, 9)
        self.assertEqual(len(p._slabs), 4)
        for args in ((0, 2, 2), (8, True, 2), (8, 2, -1)):
            with self.assertRaises(ValueError):
                m.staging_bytes(*args)

    def test_payloads_and_preserved_native_bindings(self):
        p, w, _ = fixture()
        original = [(value, ptr) for _, _, value, ptr in p._bindings]
        for ids, prompts, layout in (((7, 0, 9, 7), (9, 7), [7, 9]),
                                     ((7, 9), (7,), [9, 7]),
                                     ((0, -1), (0,), [7, None]),
                                     ((), (), [None, None])):
            inp = mapping(ids, prompts)
            expect = lower_dense_lora_metadata(index_mapping=ids, prompt_mapping=prompts,
                slot_layout=tuple(layout), vocab_size=32000, max_num_tokens=8,
                captured_lora_counts=(1, 2, 3))
            p.update_metadata(inp, layout, 3, 32000)
            for name, values in zip(m.BASE, (expect.base_indices, expect.sampler_indices,
                                           expect.sampler_indices_padded, expect.embeddings_indices)):
                self.assertEqual(getattr(w, name)[:len(values)].tolist(), list(values))
            for meta, rows in zip(p.metas, (expect.token_kernel, expect.sampler_kernel)):
                for name, values in zip(m.KERNEL, (rows.slot_by_row, rows.stable_order,
                                                  rows.active_slots, rows.counts, rows.starts)):
                    self.assertEqual(getattr(meta, name)[:len(values)].tolist(), list(values))
                self.assertEqual(meta.no_lora_flag_cpu.item(), rows.no_lora)
                self.assertEqual(meta.num_active_loras_cpu.item(), rows.launch_groups)
                self.assertEqual(meta.default_num_active_loras_cpu.item(), 3)
            self.assertEqual(w.indices_len, list(expect.indices_len))
        self.assertTrue(all(value.data_ptr() == ptr for value, ptr in original))

    def test_bank_not_overwritten_before_copy_completion(self):
        p, _, _ = fixture()
        p.update_metadata(mapping(), [7, 9], 3, 32000)
        bank = p._banks[0]
        before = bank.views[0].copy()
        bank.event.on_wait = lambda: self.assertEqual(bank.views[0].tolist(), before.tolist())
        p.update_metadata(mapping((9,), (9,)), [7, 9], 3, 32000)
        p.update_metadata(mapping((9, 9), (9,)), [7, 9], 3, 32000)
        self.assertEqual(bank.event.waits, 1)
        self.assertEqual(bank.event.records, 2)

    def test_completed_bank_does_not_wait(self):
        p, _, _ = fixture()
        for _ in range(2):
            p.update_metadata(mapping(), [7, 9], 3, 32000)
        p._banks[0].event.done = True
        p.update_metadata(mapping(), [7, 9], 3, 32000)
        self.assertEqual(p.reuse_waits, 0)

    def test_all_base_leaves_unused_arrays(self):
        p, _, _ = fixture()
        p.update_metadata(mapping(), [7, 9], 3, 32000)
        old = p.metas[0].token_lora_mapping.clone()
        p.update_metadata(mapping((0, 0), (0,)), [7, 9], 3, 32000)
        self.assertTrue(torch.equal(p.metas[0].token_lora_mapping, old))
        self.assertEqual(p.metas[0].active_lora_ids.tolist(), [-1] * 3)
        self.assertEqual(p.metas[0].lora_token_start_loc.tolist(), [0] * 4)

    def test_unsupported_boundary_rejected_before_copy(self):
        for change in ('stream', 'capture', 'thread', 'binding', 'slots', 'missing'):
            p, w, t = fixture()
            if change == 'stream':
                t.cuda.current_stream = lambda device: object()
            if change == 'capture':
                t.cuda.is_current_stream_capturing = lambda: True
            if change == 'thread':
                p.thread_id = -1
            if change == 'binding':
                w._token_lora_indices = w._token_lora_indices.clone()
            with self.assertRaises((RuntimeError, ValueError)):
                p.update_metadata(mapping(), [7, None] if change == 'missing' else [7, 9],
                                  4 if change == 'slots' else 3, 32000)
            self.assertEqual(p.copied_bytes, 0)

    def test_partial_copy_poison_never_falls_back(self):
        p, _, _ = fixture()
        original = torch.Tensor.copy_
        count = 0
        def fail_second(dest, source, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError('injected copy failure')
            return original(dest, source, **kwargs)
        with patch.object(torch.Tensor, 'copy_', fail_second):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                p.update_metadata(mapping(), [7, 9], 3, 32000)
            with self.assertRaisesRegex(RuntimeError, 'previously failed'):
                p.update_metadata(mapping(), [7, 9], 3, 32000)
        self.assertTrue(p.poisoned)
        self.assertEqual(count, 2)


class InstallationTests(unittest.TestCase):
    def setup_install(self, budget=100000):
        p, w, t = fixture()
        manager = object()
        worker = NS(model_runner=NS(vllm_config=NS(additional_config={
            'ieee_tc_lora_metadata_policy': m.POLICY})))
        result = dict(configured=True, tensor_budget_bytes=budget,
                      workspace_partition=dict(registered_upper_bytes=1000, demand_load_peak_bytes=300),
                      allocation=dict(before=dict(accounted_tensor_bytes=100), allocator_policy=dict(
                          policy='uncached_background_v1', verified=True)))
        owner = NS(_caches=lambda: ({}, {}), staged_models=lambda: {},
                   host_allocation_check=lambda **kw: dict(admitted=True,
                       before=dict(accounted_tensor_bytes=100 + p.staging_bytes)))
        boundary = (t, w, 8, 2, 2, p.stream)
        return worker, manager, owner, result, boundary

    def test_off_has_no_allocation_or_boundary_probe(self):
        worker = NS(model_runner=NS(vllm_config=NS(additional_config={})))
        with patch.object(m, '_qualified_boundary', side_effect=AssertionError('must not probe')):
            result = {}
            self.assertIs(m.install_after_host_budget(worker, None, None, result), result)

    def test_install_and_idempotence(self):
        worker, manager, owner, result, boundary = self.setup_install()
        with patch.object(m, '_qualified_boundary', return_value=boundary):
            first = m.install_after_host_budget(worker, manager, owner, result)
            publisher = worker._ieee_metadata_publisher
            second = m.install_after_host_budget(worker, manager, owner, result)
        self.assertIs(worker._ieee_metadata_publisher, publisher)
        self.assertTrue(first['metadata_publication']['installed'])
        self.assertEqual(second['metadata_publication']['staging_bytes'], publisher.staging_bytes)
        self.assertEqual(boundary[1].update_metadata, publisher.update_metadata)

    def test_budget_failure_before_any_pinned_allocation(self):
        worker, manager, owner, result, boundary = self.setup_install(budget=1700)
        with patch.object(m, '_qualified_boundary', return_value=boundary), \
                patch.object(m, 'DenseMetadataPublisher', side_effect=AssertionError('must not allocate')):
            with self.assertRaisesRegex(RuntimeError, 'exceeds HOST'):
                m.install_after_host_budget(worker, manager, owner, result)

    def test_failed_postcheck_cannot_install_on_retry(self):
        worker, manager, owner, result, boundary = self.setup_install()
        original = boundary[1].update_metadata
        owner.host_allocation_check = lambda **kw: dict(admitted=False,
            before=dict(accounted_tensor_bytes=1000000))
        with patch.object(m, '_qualified_boundary', return_value=boundary):
            with self.assertRaises(RuntimeError):
                m.install_after_host_budget(worker, manager, owner, result)
            with self.assertRaises(RuntimeError):
                m.install_after_host_budget(worker, manager, owner, result)
        self.assertIs(boundary[1].update_metadata, original)

    def test_invalid_policy_and_owner_state(self):
        for condition in ('policy', 'loaded', 'staged', 'allocator', 'workspace'):
            worker, manager, owner, result, boundary = self.setup_install()
            if condition == 'policy':
                worker.model_runner.vllm_config.additional_config['ieee_tc_lora_metadata_policy'] = 'unknown'
            if condition == 'loaded':
                owner._caches = lambda: ({7: object()}, {})
            if condition == 'staged':
                owner.staged_models = lambda: {7: object()}
            if condition == 'allocator':
                result['allocation']['allocator_policy']['verified'] = False
            if condition == 'workspace':
                result['workspace_partition'] = None
            with patch.object(m, '_qualified_boundary', return_value=boundary), \
                    patch.object(m, 'DenseMetadataPublisher', side_effect=AssertionError('must not allocate')):
                with self.assertRaises((RuntimeError, ValueError)):
                    m.install_after_host_budget(worker, manager, owner, result)


if __name__ == '__main__':
    unittest.main()
