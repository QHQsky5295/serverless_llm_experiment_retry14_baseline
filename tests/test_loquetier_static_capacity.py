"""CPU metadata checks only; synthetic fixtures are not performance evidence."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

from Loquetier_project.scripts.run_loquetier_trace_gate import (
    _adapter_footprint, _static_capacity_shards, _tensor_header)


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


if __name__ == '__main__':
    unittest.main()
