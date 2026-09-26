"""Tiny CPU serialization fixtures; run with the pinned native torch environment.

These are test arrays, not generated LoRA weights or experiment/model artifacts.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

spec = importlib.util.spec_from_file_location(
    'tc_checkpoint', Path(__file__).resolve().parents[1] / 'scripts/prepare_ieee_tc_serverless_stack.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class NativeCheckpointTests(unittest.TestCase):
    def fixture(self, root):
        native, source = root / 'native', root / 'source'
        native.mkdir()
        source.mkdir()
        rank = native / 'rank_0'
        rank.mkdir()
        config = dict(model_type='llama', architectures=['LlamaForCausalLM'], tie_word_embeddings=True)
        for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json',
                     'generation_config.json', 'special_tokens_map.json'):
            content = json.dumps(config if name == 'config.json' else {})
            (source / name).write_text(content)
            (native / name).write_text(content)
        names = ['model.embed_tokens.weight', 'model.layers.0.self_attn.q_proj.weight',
                 'model.layers.0.self_attn.k_proj.weight', 'model.layers.0.self_attn.v_proj.weight',
                 'model.layers.0.mlp.gate_proj.weight', 'model.layers.0.mlp.up_proj.weight']
        tensors = {name: (torch.arange(6).reshape(2, 3) + i / 4).to(torch.bfloat16)
                   for i, name in enumerate(names)}
        save_file(tensors, str(source / 'model.safetensors'))
        (source / 'model.safetensors.index.json').write_text(json.dumps(
            dict(weight_map={name: 'model.safetensors' for name in names})))
        # Construct independently from the auditor's mapping.
        parts = [(names[0], [names[0]]),
                 ('model.layers.0.self_attn.qkv_proj.weight', names[1:4]),
                 ('model.layers.0.mlp.gate_up_proj.weight', names[4:6])]
        index, offset = {}, 0
        with (rank / 'tensor.data_0').open('wb') as handle:
            for name, components in parts:
                tensor = torch.cat([tensors[p] for p in components], dim=0).to(torch.float16)
                data = tensor.numpy().tobytes()
                index[name] = [offset, len(data), list(tensor.shape), list(tensor.stride()), 'torch.float16']
                handle.write(data)
                offset += len(data)
        (rank / 'tensor_index.json').write_text(json.dumps(index))
        return native, source

    def test_all_current_bytes_match_packed_fp16_without_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            native, source = self.fixture(Path(directory))
            before = launch.stream_sha(native / 'rank_0/tensor.data_0')
            result = launch.audit_checkpoint(native, source)
            self.assertTrue(result['passed'])
            self.assertEqual(result['native_tensor_count'], 3)
            self.assertEqual(result['source_tensor_count'], 6)
            self.assertEqual(result['native_data_sha256'], before)
            self.assertTrue(all(r['native_sha256'] == r['reference_fp16_sha256'] for r in result['tensors']))
            self.assertFalse(result['native_loader_qualified'])
            self.assertFalse(result['performance_run_authorized'])
            self.assertFalse(torch.cuda.is_initialized())

    def test_wrong_weight_and_packed_order_fail(self):
        for reorder in (False, True):
            with self.subTest(reorder=reorder), tempfile.TemporaryDirectory() as directory:
                native, source = self.fixture(Path(directory))
                data = native / 'rank_0/tensor.data_0'
                raw = bytearray(data.read_bytes())
                if reorder:
                    raw[12:24], raw[24:36] = raw[24:36], raw[12:24]
                else:
                    raw[0] ^= 1
                data.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, 'exact FP16 checkpoint mismatch'):
                    launch.audit_checkpoint(native, source)

    def test_changed_tokenizer_and_omitted_source_tensor_fail(self):
        for change in ('tokenizer', 'index'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                native, source = self.fixture(Path(directory))
                if change == 'tokenizer':
                    (native / 'tokenizer.json').write_text('{"changed":1}')
                else:
                    path = source / 'model.safetensors.index.json'
                    index = json.loads(path.read_text())
                    del index['weight_map']['model.embed_tokens.weight']
                    path.write_text(json.dumps(index))
                with self.assertRaises(ValueError):
                    launch.audit_checkpoint(native, source)


if __name__ == '__main__':
    unittest.main()
