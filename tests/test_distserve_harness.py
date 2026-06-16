import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from scripts.replay_distserve_trace import parse_token_timestamps
from scripts.validate_distserve_semantics import common_prefix_length


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = (
    ROOT / "workspaces/distserve_official_20260615"
)


def load_converter_module():
    logger_module = types.ModuleType("distserve.logger")
    logger_module.init_logger = lambda name: None
    package = types.ModuleType("distserve")
    package.__path__ = []
    sys.modules["distserve"] = package
    sys.modules["distserve.logger"] = logger_module

    converter_path = WORKSPACE / "distserve/downloader/converter.py"
    spec = importlib.util.spec_from_file_location(
        "distserve_converter_harness_test",
        converter_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DistServeHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.converter = load_converter_module()

    def test_safetensors_shards_are_merged_and_preferred_over_bin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_file(
                {"model.embed_tokens.weight": torch.tensor([1.0])},
                root / "model-00001-of-00002.safetensors",
            )
            save_file(
                {"lm_head.weight": torch.tensor([2.0])},
                root / "model-00002-of-00002.safetensors",
            )
            torch.save(
                {"partial_wrong_format": torch.tensor([3.0])},
                root / "partial.bin",
            )

            weight_format, files = self.converter.discover_weight_files(
                str(root)
            )
            state = self.converter.load_hf_state_dict(str(root))

            self.assertEqual(weight_format, "safetensors")
            self.assertEqual(len(files), 2)
            self.assertEqual(
                set(state),
                {"model.embed_tokens.weight", "lm_head.weight"},
            )

    def test_token_timestamps_match_distserve_api_clock(self):
        ttft, tpot, e2e, token_ids = parse_token_timestamps(
            {
                "timestamps": [100.1, 100.12, 100.15],
                "token_ids": [10, 11, 12],
            },
            dispatch_perf=100.0,
        )
        self.assertAlmostEqual(ttft, 100.0)
        self.assertAlmostEqual(tpot, 25.0)
        self.assertAlmostEqual(e2e, 150.0)
        self.assertEqual(token_ids, [10, 11, 12])

    def test_token_observability_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "observability mismatch"):
            parse_token_timestamps(
                {
                    "timestamps": [100.1, 100.2],
                    "token_ids": [10],
                },
                dispatch_perf=100.0,
            )

    def test_semantic_comparison_uses_contiguous_token_prefix(self):
        self.assertEqual(common_prefix_length([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(common_prefix_length([1, 2], [9, 2]), 0)
        self.assertEqual(common_prefix_length([1, 2], [1, 2, 3]), 2)

    def test_bin_checkpoint_remains_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save({"weight": torch.tensor([4.0])}, root / "model.bin")
            weight_format, files = self.converter.discover_weight_files(
                str(root)
            )
            state = self.converter.load_hf_state_dict(str(root))
            self.assertEqual(weight_format, "bin")
            self.assertEqual(len(files), 1)
            self.assertEqual(state["weight"].item(), 4.0)

    def test_missing_checkpoint_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                ValueError,
                r"\*\.safetensors or \*\.bin",
            ):
                self.converter.discover_weight_files(tmp)

    def test_tied_llama_embeddings_materialize_output_projection(self):
        tensor_dict = {
            "model.embed_tokens.weight": torch.arange(8).reshape(4, 2),
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(128, 2),
            "model.layers.0.self_attn.k_proj.weight": torch.zeros(128, 2),
            "model.layers.0.self_attn.v_proj.weight": torch.zeros(128, 2),
            "model.layers.0.self_attn.o_proj.weight": torch.zeros(2, 128),
        }
        converted, _, _ = self.converter.preprocess_llama2(tensor_dict)
        self.assertIn("lm_head.weight", converted)
        self.assertTrue(
            torch.equal(
                converted["lm_head.weight"],
                converted["model.embed_tokens.weight"],
            )
        )

    def test_patch_records_both_converter_and_downloader_changes(self):
        patch = (
            ROOT / "patches/distserve_relayserve_compat.patch"
        ).read_text(encoding="utf-8")
        self.assertIn("discover_weight_files", patch)
        self.assertIn("load_hf_state_dict", patch)
        self.assertIn("*.safetensors.index.json", patch)
        self.assertIn("Prefer safetensors", patch)
        self.assertIn("tensor_dict[LM_HEAD] = tensor_dict[EMBED_TOKENS]", patch)
        self.assertIn('"token_ids": token_ids', patch)

    def test_formal_runner_fails_closed_on_semantic_mismatch(self):
        runner = (
            ROOT / "scripts/run_distserve_relayserve_continuation.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("validate_distserve_semantics.py", runner)
        self.assertIn("--required-exact-fraction 1.0", runner)
        self.assertIn("--required-first-token-fraction 1.0", runner)
        self.assertIn("--required-token-fraction 1.0", runner)
        self.assertIn("--context-tensor-parallel-size 2", runner)
        self.assertIn("--decoding-tensor-parallel-size 2", runner)
        self.assertIn("DISTSERVE_MIN_AVAILABLE_MEMORY_GB", runner)
        self.assertIn("refusing to overwrite existing DistServe run directory", runner)


if __name__ == "__main__":
    unittest.main()
