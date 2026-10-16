"""CPU metadata checks only; synthetic fixtures are not performance evidence."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from Loquetier_project.scripts.run_loquetier_trace_gate import (
    _adapter_footprint, _static_capacity_shards, _tensor_header,
    _ownership_prompt, _ownership_forward, _parse_args)


class LoquetierCapacityTests(unittest.TestCase):
    def fixture(self, aid='a', rank=8):
        config = dict(peft_type='LORA', bias='none', r=rank, base_model_name_or_path='/model')
        tensors = {
            'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight':
                dict(shape=[rank, 16], dtype='F16'),
            'base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight':
                dict(shape=[16, rank], dtype='F16')}
        return config, tensors, _adapter_footprint(aid, config, tensors)

    def test_actual_shape_rank_not_config_rank_only(self):
        config, tensors, _ = self.fixture(rank=16)
        config['r'] = 8
        row = _adapter_footprint('a', config, tensors)
        self.assertEqual(row['native_max_rank'], 16)
        self.assertEqual(row['actual_ranks'], [16])
        self.assertEqual(row['peft_original_fp32_bytes'], 2048)
        self.assertEqual(row['prepared_fp16_bytes_per_rank'], 64)
        self.assertEqual(row['pointer_bytes'], 16)

    def test_rank_pair_and_unsupported_weights_rejected(self):
        config, tensors, _ = self.fixture()
        for modified in ({next(iter(tensors)): tensors[next(iter(tensors))]},
                         dict(tensors, **{'embed_tokens.weight': dict(shape=[8, 16])})):
            with self.assertRaises(ValueError):
                _adapter_footprint('a', config, modified)
        invalid = copy.deepcopy(tensors)
        invalid[list(invalid)[1]]['shape'][1] = 16
        with self.assertRaises(ValueError):
            _adapter_footprint('a', config, invalid)
        with self.assertRaises(ValueError):
            _adapter_footprint('a', dict(config, use_dora=True), tensors)

    def test_deterministic_full_disjoint_mapping_without_request_popularity(self):
        adapters = [self.fixture(str(i), 8 if i % 2 else 16)[2] for i in range(20)]
        result = _static_capacity_shards(adapters, 4)
        self.assertEqual(result, _static_capacity_shards(list(reversed(adapters)), 4))
        self.assertEqual(sorted(a for r in result for a in r['adapter_ids']), sorted(a['id'] for a in adapters))
        self.assertEqual(sum(r['adapter_count'] for r in result), 20)
        for row in result:
            self.assertEqual(row['prepared_fp16_bytes'], row['adapter_count'] * row['native_max_rank'] * 64)

    def test_duplicate_or_heterogeneous_module_mapping_rejected(self):
        a, b = self.fixture('a')[2], self.fixture('b')[2]
        with self.assertRaises(ValueError):
            _static_capacity_shards([a, a], 2)
        b['module_signature'][0][1] = 32
        with self.assertRaises(ValueError):
            _static_capacity_shards([a, b], 2)
        for count in (0, 5, True):
            with self.assertRaises(ValueError):
                _static_capacity_shards([a], count)

    def test_header_reads_metadata_and_checks_actual_file_extent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weights.safetensors'
            data = json.dumps({'a': dict(shape=[2, 3], dtype='F16', data_offsets=[0, 12])}).encode()
            payload = struct.pack('<Q', len(data)) + data + bytes(12)
            path.write_bytes(payload)
            rows, digest = _tensor_header(path)
            self.assertEqual(rows['a']['shape'], [2, 3])
            self.assertEqual(len(digest), 64)
            for invalid in (b'bad', payload[:-1], payload + b'bad', struct.pack('<Q', 2**63)):
                path.write_bytes(invalid)
                with self.assertRaises(ValueError):
                    _tensor_header(path)

    def test_owner_prompt_reuses_first_native_input_and_verifies_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'native.jsonl'
            first = dict(event='loquetier_native_completed', request_id='first',
                observation=dict(native_terminal_observed=True, native_prompt_token_ids=list(range(64))))
            second = dict(first, request_id='later')
            path.write_text('\n'.join(json.dumps(x) for x in (dict(event='start'), first, second))+'\n')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            result = _ownership_prompt(path, digest)
            self.assertEqual(result['source_request_id'], 'first')
            self.assertEqual(result['prompt_ids'], list(range(32)))
            self.assertEqual(result['source_prompt_tokens'], 64)
            self.assertFalse(result['performance_input'])
            with self.assertRaisesRegex(ValueError, 'SHA'):
                _ownership_prompt(path, '0'*64)
            first['observation']['native_terminal_observed'] = False
            path.write_text(json.dumps(first)+'\n')
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                _ownership_prompt(path, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_owner_cli_requires_explicit_optin_and_source_identity(self):
        base = ['gate', '--base-model', '/base', '--adapter-root', '/pool', '--output', '/new',
                '--native-capacity-probe', '--static-capacity-plan', '/plan',
                '--measurement-view', '/view', '--measurement-view-sha256', 'a'*64,
                '--physical-gpu-id', '0', '--main-repo', '/main']
        with patch('sys.argv', base):
            self.assertFalse(_parse_args().inference_owner_probe)
        inputs = ['--ownership-input-journal', '/journal', '--ownership-input-sha256', 'b'*64]
        for extra in (['--inference-owner-probe'], inputs,
                      ['--inference-owner-probe', *inputs[:-1], 'bad']):
            with patch('sys.argv', base+extra), self.assertRaises(SystemExit):
                _parse_args()
        with patch('sys.argv', base+['--inference-owner-probe', *inputs]):
            self.assertTrue(_parse_args().inference_owner_probe)

    def test_owner_forward_covers_every_id_with_same_grouping_and_rejects_wrong_binding(self):
        class Bridge:
            batch_size = 2
            def __init__(self, wrong=False):
                self.generator = SimpleNamespace(requests_len=0)
                self.objects, self.groups, self.mark, self.steps = {}, [], 0, 0
                self.wrong = wrong
            def add(self, rid, aid, ids, target):
                mark = self.mark
                self.mark += 1
                self.objects[mark] = (aid, ids)
                self.generator.requests_len += 1
                return mark
            def step(self):
                self.steps += 1
                if self.steps % 2:
                    self.groups.append([aid for aid, _ in self.objects.values()])
                    return {}
                result = {mark: dict(native_adapter_id='bad' if self.wrong else aid,
                    native_prompt_token_ids=ids, native_output_tokens=2, native_forward_count=2,
                    native_adapter_index=mark, completion_token_ids=[3, 4])
                    for mark, (aid, ids) in self.objects.items()}
                self.objects.clear()
                self.generator.requests_len = 0
                return result
        bridge = Bridge()
        rows = _ownership_forward(bridge, ['a', 'b', 'c'], [1, 2], 'before')
        self.assertEqual([row['adapter_id'] for row in rows], ['a', 'b', 'c'])
        self.assertEqual(bridge.groups, [['a', 'b'], ['c']])
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            _ownership_forward(Bridge(wrong=True), ['a'], [1, 2], 'after')


if __name__ == '__main__':
    unittest.main()
