import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file

from scripts import ieee_tc_preflight as p


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
