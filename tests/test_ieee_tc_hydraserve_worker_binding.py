"""Load-boundary integration invariants; no GPU/performance qualification."""
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from HydraServe_project.runtime import lora_worker_binding as binding

NATIVE = Path(__file__).resolve().parents[1] / 'vendor_new_baselines/HydraServe_main_20260520'


class WorkerBindingTests(unittest.TestCase):
    def test_exact_native_inverse_and_profiler_coverage(self):
        overlay, removed = binding.render_overlay(NATIVE)
        binding.verify_inverse(overlay, removed, NATIVE)
        model = next(n for n in ast.parse(overlay).body if isinstance(n, ast.ClassDef) and n.name == 'ModelRunner')
        for name, helper in [('load_model', '_hydra_bind_initial'), ('load_dest_model', '_hydra_bind_destination')]:
            method = next(n for n in model.body if getattr(n, 'name', None) == name)
            self.assertIsInstance(method.body[0], ast.With)
            call = method.body[0].body[-1].value
            self.assertEqual(call.func.id, helper)

    def test_coordinates_and_hash_are_authoritative(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'partition.json'
            raw = json.dumps(dict(schema='hydraserve_native_partition_v1', total_layers=32,
                                  partitions={'2': [16, 32], '1': [32]})).encode()
            path.write_bytes(raw)
            env = dict(HYDRASERVE_PARTITION_MANIFEST=str(path),
                       HYDRASERVE_PARTITION_SHA256=sha256(raw).hexdigest(),
                       PP_RANK='1', PP_SIZE='2', DEST_PP_RANK='0', DEST_PP_SIZE='1')
            with patch.dict(os.environ, env):
                model = NS(pp_rank=1, pp_size=2, config=NS(num_hidden_layers=16))
                stage, _ = binding.read_stage(model)
                self.assertEqual(stage.global_layers, tuple(range(16, 32)))
                destination = NS(pp_rank=0, pp_size=1, config=NS(num_hidden_layers=32))
                self.assertEqual(binding.read_stage(destination, True)[0].global_layers, tuple(range(32)))
                with self.assertRaisesRegex(ValueError, 'rank differs'):
                    binding.read_stage(destination)
                model.config.num_hidden_layers = 15
                with self.assertRaisesRegex(ValueError, 'config differs'):
                    binding.read_stage(model)
                path.write_bytes(raw + b' ')
                with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                    binding.read_stage(model)

    def test_no_lora_leaves_native_path_unmodified(self):
        runner = NS(lora_config=None)
        binding.bind_initial(runner)
        binding.bind_destination(runner)
        self.assertEqual(vars(runner), {'lora_config': None})


if __name__ == '__main__':
    unittest.main()
