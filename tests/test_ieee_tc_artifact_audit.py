import json
from pathlib import Path
import tempfile
import unittest
import copy
import hashlib
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file

from scripts import ieee_tc_preflight as p


class NativeExecutionMetadataAudit(unittest.TestCase):
    """Pure metadata fixtures, never GPU or model-numerical qualification."""
    @staticmethod
    def meta(slots, capacity=4):
        active = sorted(set(slots))
        all_base = active == [-1]
        counts = [] if all_base else [slots.count(a) for a in active]
        starts = [0]
        for count in counts:
            starts.append(starts[-1] + count)
        return dict(token_lora_mapping=list(slots),
                    token_indices_sorted_by_lora_ids=sorted(range(len(slots)), key=slots.__getitem__),
                    active_lora_ids=([] if all_base else active) + [-1] * (capacity + 1 - (0 if all_base else len(active))),
                    num_tokens_per_lora=counts + [0] * (capacity + 1 - len(counts)),
                    lora_token_start_loc=starts + [0] * (capacity + 2 - len(starts)),
                    no_lora=all_base, launch_lora_count=capacity + 1)

    def observation(self, ids=(11, 22, 0), counts=(2, 1, 2), layout=(22, None, 11, 33)):
        bindings = {f'r{i}': aid for i, aid in enumerate(ids)}
        slots = [layout.index(aid) if aid else -1 for aid in ids]
        tokens = [s for s, n in zip(slots, counts) for _ in range(n)]
        return dict(kind='native_lora_forward_metadata_v1', slot_adapter_ids=list(layout),
                    requests=[dict(backend_request_id=f'r{i}', adapter_int_id=aid,
                                   scheduled_tokens=n, sampled_tokens=1)
                              for i, (aid, n) in enumerate(zip(ids, counts))],
                    token_slot_indices=tokens, sampler_slot_indices=slots,
                    token_kernel_meta=self.meta(tokens), sampler_kernel_meta=self.meta(slots)), bindings

    def test_mixed_prefill_decode_and_base_rows(self):
        obs, bindings = self.observation()
        before = copy.deepcopy(obs)
        result = p.validate_native_lora_execution_metadata(obs, bindings)
        self.assertTrue(result['metadata_matches'])
        self.assertEqual(result['token']['real_rows'], 5)
        self.assertEqual(result['sampler']['real_rows'], 3)
        self.assertFalse(result['gpu_observation_qualified'])
        self.assertFalse(result['kernel_arithmetic_qualified'])
        self.assertFalse(result['full_pool_qualified'])
        self.assertIsNone(result['n_correct'])
        self.assertEqual(obs, before)

    def test_same_logical_mapping_new_slot_layout_requires_new_metadata(self):
        old, bindings = self.observation()
        new, _ = self.observation(layout=(11, 22, 33, None))
        self.assertTrue(p.validate_native_lora_execution_metadata(new, bindings)['metadata_matches'])
        for field in ('token_slot_indices', 'sampler_slot_indices', 'token_kernel_meta', 'sampler_kernel_meta'):
            stale = copy.deepcopy(new)
            stale[field] = old[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.validate_native_lora_execution_metadata(stale, bindings)

    def test_batch_reordering_uses_request_ids_not_row_numbers(self):
        obs, _ = self.observation(ids=(22, 0, 11), counts=(1, 2, 2))
        obs['requests'][0]['backend_request_id'] = 'r1'
        obs['requests'][1]['backend_request_id'] = 'r2'
        obs['requests'][2]['backend_request_id'] = 'r0'
        bindings = {'r0': 11, 'r1': 22, 'r2': 0, 'not_scheduled': 33}
        self.assertTrue(p.validate_native_lora_execution_metadata(obs, bindings)['metadata_matches'])

    def test_external_binding_catches_self_consistent_wrong_adapter(self):
        obs, bindings = self.observation(ids=(22,), counts=(4,))
        bindings['r0'] = 11
        with self.assertRaisesRegex(ValueError, 'identity'):
            p.validate_native_lora_execution_metadata(obs, bindings)

    def test_stale_payload_is_unused_only_for_real_all_base_branch(self):
        obs, bindings = self.observation(ids=(0,), counts=(2,))
        for field in ('token_kernel_meta', 'sampler_kernel_meta'):
            obs[field]['token_lora_mapping'] = [99] * len(obs[field]['token_lora_mapping'])
            obs[field]['token_indices_sorted_by_lora_ids'] = [99] * len(obs[field]['token_indices_sorted_by_lora_ids'])
        result = p.validate_native_lora_execution_metadata(obs, bindings)
        self.assertTrue(result['token']['skipped'])
        obs['token_kernel_meta']['num_tokens_per_lora'][0] = 1
        with self.assertRaisesRegex(ValueError, 'not reset'):
            p.validate_native_lora_execution_metadata(obs, bindings)

    def test_wrong_groups_and_silent_no_lora_branch_rejected(self):
        mutations = [
            ('token_lora_mapping', [2, 2, -1, -1, -1]),
            ('token_indices_sorted_by_lora_ids', [3, 3, 2, 0, 1]),
            ('token_indices_sorted_by_lora_ids', [3, 4, 0, 2, 1]),
            ('token_indices_sorted_by_lora_ids', [3, 5, 2, 0, 1]),
            ('active_lora_ids', [-1, 2, 0, -1, -1]),
            ('num_tokens_per_lora', [2, 1, 1, 0, 0]),
            ('num_tokens_per_lora', [2, 1, 3, 0, 0]),
            ('lora_token_start_loc', [0, 1, 3, 5, 0, 0]),
            ('no_lora', True), ('launch_lora_count', 2),
        ]
        for field, value in mutations:
            obs, bindings = self.observation()
            obs['token_kernel_meta'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                p.validate_native_lora_execution_metadata(obs, bindings)

    def test_wrong_sampler_groups_are_not_hidden_by_correct_token_groups(self):
        obs, bindings = self.observation()
        obs['sampler_kernel_meta']['active_lora_ids'][1] = 2
        with self.assertRaisesRegex(ValueError, 'sampler'):
            p.validate_native_lora_execution_metadata(obs, bindings)

    def test_missing_or_duplicate_physical_identity(self):
        for slots in ([22, None, None, 33], [22, 11, 11, 33], [22, None, True, 33]):
            obs, bindings = self.observation()
            obs['slot_adapter_ids'] = slots
            with self.assertRaises(ValueError):
                p.validate_native_lora_execution_metadata(obs, bindings)

    def test_unsupported_or_malformed_rows(self):
        for field, value in [('backend_request_id', 'unknown'), ('adapter_int_id', True),
                             ('scheduled_tokens', 0), ('scheduled_tokens', 10**9),
                             ('sampled_tokens', 2), ('sampled_tokens', True)]:
            obs, bindings = self.observation()
            obs['requests'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.validate_native_lora_execution_metadata(obs, bindings)
        obs, bindings = self.observation()
        obs['requests'][1]['backend_request_id'] = 'r0'
        with self.assertRaises(ValueError):
            p.validate_native_lora_execution_metadata(obs, bindings)

    def test_missing_schema_fields_fail_instead_of_zero_fill(self):
        obs, bindings = self.observation()
        for field in obs:
            broken = copy.deepcopy(obs)
            del broken[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.validate_native_lora_execution_metadata(broken, bindings)
        for field in obs['token_kernel_meta']:
            broken = copy.deepcopy(obs)
            del broken['token_kernel_meta'][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.validate_native_lora_execution_metadata(broken, bindings)


class CheckpointSlotAudit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.weights = Path(self.tmp.name)/'adapter_model.safetensors'
        self.config = dict(peft_type='LORA', bias='none', r=1, lora_alpha=2,
                           target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
        self.base = dict(model_type='llama', hidden_size=2, intermediate_size=3,
                         num_hidden_layers=1, vocab_size=5,
                         num_attention_heads=1, num_key_value_heads=1)
        self.tensors = {}
        for i, name in enumerate(self.config['target_modules'], 1):
            prefix = f'base_model.model.model.layers.0.self_attn.{name}'
            self.tensors[prefix+'.lora_A.weight'] = np.array([[i, -i]], dtype=np.float32)
            self.tensors[prefix+'.lora_B.weight'] = np.array([[i], [-i]], dtype=np.float32)
        save_file(self.tensors, str(self.weights))

    def expected(self):
        return p.checkpoint_slot_fingerprints(self.weights, self.config, self.base, 4)

    @staticmethod
    def observed(expected):
        return [dict(r, all_finite=True, exact_value_match=True, mismatched_elements=0,
                     actual_nonzero_elements=r['expected_nonzero_elements'],
                     expected_padded_sha256=r['checkpoint_padded_sha256'],
                     actual_slot_sha256=r['checkpoint_padded_sha256']) for r in expected]

    def test_all_modules_padding_qkv_order_and_scaling(self):
        before = self.weights.read_bytes()
        rows = self.expected()
        self.assertEqual(len(rows), 18)
        for index in range(3):
            row = next(r for r in rows if r['module'].endswith('qkv_proj')
                       and r['slice'] == index and r['side'] == 'b')
            correct = np.zeros((2, 4), dtype='<f2')
            correct[:, 0] = [2*(index+1), -2*(index+1)]
            self.assertEqual(row['checkpoint_padded_sha256'],
                             hashlib.sha256(correct.tobytes()).hexdigest())
        self.assertEqual(sum(r['source_absent'] for r in rows), 10)
        self.assertEqual(before, self.weights.read_bytes())
        result = p.validate_checkpoint_slot_snapshot(rows, self.observed(rows))
        self.assertTrue(result['passed'])
        self.assertFalse(result['per_token_execution_mapping_qualified'])

    def test_cast_precedes_scaling(self):
        key = next(k for k in self.tensors if 'q_proj.lora_B' in k)
        self.tensors[key][:] = 1.00049
        save_file(self.tensors, str(self.weights))
        self.config['lora_alpha'] = 3
        row = next(r for r in self.expected() if r['checkpoint_key'] == key)
        padded = np.zeros((2, 4), dtype='<f2')
        padded[:, 0] = np.float16(float(np.float16(1.00049))*3)
        self.assertEqual(row['checkpoint_padded_sha256'],
                         hashlib.sha256(padded.tobytes()).hexdigest())

    def test_wrong_source_scaled_value_slice_or_padding_is_rejected(self):
        rows = self.expected()
        observed = self.observed(rows)
        index = next(i for i, r in enumerate(rows) if r['module'].endswith('qkv_proj')
                     and r['slice'] == 0 and r['side'] == 'b')
        for field, bad in [('actual_slot_sha256', 'f'*64),
                           ('expected_padded_sha256', 'e'*64),
                           ('source_absent', True), ('shape', [4, 2]),
                           ('source_shape', [1, 2]), ('all_finite', False),
                           ('actual_nonzero_elements', 3)]:
            with self.subTest(field=field):
                broken = copy.deepcopy(observed)
                broken[index][field] = bad
                self.assertFalse(p.validate_checkpoint_slot_snapshot(rows, broken)['passed'])
        # A real wrong Q/K assignment has a valid digest, not just malformed text.
        broken = copy.deepcopy(observed)
        other = next(r for r in rows if r['module'].endswith('qkv_proj')
                     and r['slice'] == 1 and r['side'] == 'b')
        broken[index]['actual_slot_sha256'] = other['checkpoint_padded_sha256']
        self.assertFalse(p.validate_checkpoint_slot_snapshot(rows, broken)['passed'])

    def test_missing_duplicate_and_extra_snapshots_rejected(self):
        rows = self.expected()
        obs = self.observed(rows)
        for broken in (obs[:-1], obs+[obs[0]], []):
            with self.assertRaises(ValueError):
                p.validate_checkpoint_slot_snapshot(rows, broken)

    def test_checkpoint_missing_or_extra_tensor_rejected(self):
        for tensors in (dict(list(self.tensors.items())[:-1]),
                        dict(self.tensors, surprise=np.zeros((1,), dtype=np.float32))):
            save_file(tensors, str(self.weights))
            with self.assertRaisesRegex(ValueError, 'inventory'):
                self.expected()

    def test_unsupported_config_and_geometry_rejected(self):
        for field, value in [('use_rslora', True), ('rank_pattern', {'q_proj':2}),
                             ('r', 5), ('lora_alpha', float('nan')),
                             ('target_modules', ['q_proj'])]:
            original = self.config.copy()
            self.config[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.expected()
            self.config = original
        self.base['num_key_value_heads'] = 2
        with self.assertRaisesRegex(ValueError, 'geometry'):
            self.expected()

    def test_nonfinite_checkpoint_not_repaired(self):
        self.tensors[next(iter(self.tensors))][:] = np.nan
        save_file(self.tensors, str(self.weights))
        with self.assertRaisesRegex(ValueError, 'shape/dtype/value'):
            self.expected()


class ExistingArtifactAudit(unittest.TestCase):
    def pool(self, root, tensors, ids=('a', 'b')):
        (root / '.publicmix_generation_manifest.json').write_text(
            json.dumps({'adapters': [{'id': a} for a in ids]}))
        for adapter in ids:
            directory = root / adapter
            directory.mkdir()
            save_file(tensors, str(directory / 'adapter_model.safetensors'))
            (directory / 'adapter_config.json').write_text(json.dumps({'r': 1}))
            (directory / 'adapter_data.bin').write_bytes(b'\0' * 13)

    def facts(self, root):
        with patch.object(p, 'check_plan', return_value='fixture'):
            return p.audit_artifact_pools([root], 2)

    def test_zero_pool_is_complete_audit_not_semantic_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.pool(root, {'layer.lora_A.weight': np.zeros((1, 2), dtype=np.float16),
                             'layer.lora_B.weight': np.zeros((3, 1), dtype=np.float16)})
            before = p.protected_entries([root])
            result = self.facts(root)
            self.assertTrue(result['audit_complete'])
            self.assertFalse(result['semantic_adapter_qualification'])
            pool = result['pools'][0]
            self.assertEqual(pool['distinct_weight_sha256'], 1)
            self.assertEqual(pool['all_zero_tensor_adapters'], 2)
            self.assertEqual(pool['provably_zero_ab_update_adapters'], 2)
            self.assertEqual(pool['logical_padding_bytes'], 26)
            self.assertEqual(before, p.protected_entries([root]))

    def test_zero_b_with_nonzero_a_still_proves_zero_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.pool(root, {'layer.lora_A.weight': np.ones((1, 2), dtype=np.float16),
                             'layer.lora_B.weight': np.zeros((3, 1), dtype=np.float16)})
            pool = self.facts(root)['pools'][0]
            self.assertEqual(pool['all_zero_tensor_adapters'], 0)
            self.assertEqual(pool['provably_zero_ab_update_adapters'], 2)

    def test_nonzero_operands_not_automatically_nonzero_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.pool(root, {'layer.lora_A.weight': np.ones((2, 1), dtype=np.float16),
                             'layer.lora_B.weight': np.array([[1, -1]], dtype=np.float16)})
            result = self.facts(root)
            self.assertEqual(result['pools'][0]['provably_zero_ab_update_adapters'], 0)
            self.assertFalse(result['semantic_adapter_qualification'])

    def test_nonfinite_is_reported_not_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.pool(root, {'layer.lora_A.weight': np.array([[np.nan]], dtype=np.float16),
                             'layer.lora_B.weight': np.zeros((1, 1), dtype=np.float16)})
            before = p.protected_entries([root])
            pool = self.facts(root)['pools'][0]
            self.assertEqual(pool['nonfinite_adapters'], 2)
            self.assertEqual(pool['provably_zero_ab_update_adapters'], 0)
            self.assertEqual(before, p.protected_entries([root]))

    def test_missing_weight_retains_failed_adapter_and_incomplete_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.pool(root, {'layer.lora_A.weight': np.zeros((1, 2), dtype=np.float16)})
            (root / 'b' / 'adapter_model.safetensors').unlink()
            result = self.facts(root)
            self.assertFalse(result['audit_complete'])
            pool = result['pools'][0]
            self.assertEqual(pool['inspected_adapters'], 1)
            self.assertFalse(pool['rows'][1]['inspected'])
            self.assertFalse(next(iter(result['weights_by_sha256'].values()))[
                'all_ab_updates_provably_zero'])

    def test_duplicate_and_path_manifest_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for ids in (['a', 'a'], ['../escape'], ['.'], ['']):
                (root / '.publicmix_generation_manifest.json').write_text(
                    json.dumps({'adapters': [{'id': a} for a in ids]}))
                with self.assertRaisesRegex(ValueError, 'unique safe'):
                    self.facts(root)

    def test_changed_content_is_not_accepted_as_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.pool(root, {'layer.lora_A.weight': np.zeros((1, 2), dtype=np.float16),
                             'layer.lora_B.weight': np.zeros((3, 1), dtype=np.float16)})
            real = p.artifact_tensor_facts
            def change_after_read(path):
                facts = real(path)
                (path.parent / 'adapter_data.bin').write_bytes(b'changed')
                return facts
            with patch.object(p, 'artifact_tensor_facts', side_effect=change_after_read):
                result = self.facts(root)
            self.assertFalse(result['audit_complete'])
            self.assertFalse(result['pools'][0]['rows'][0]['inspected'])


if __name__ == '__main__':
    unittest.main()
