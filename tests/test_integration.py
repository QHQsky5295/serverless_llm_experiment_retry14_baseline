#!/usr/bin/env python3
"""Integration-level smoke tests for current mainline entrypoints."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import yaml

from scripts.generate_lora_adapters import resolve_generation_defaults


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_CONFIG = PROJECT_ROOT / "configs" / "experiments.yaml"


class EntryPointIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with EXPERIMENTS_CONFIG.open("r", encoding="utf-8") as f:
            cls.experiments = yaml.safe_load(f)

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *args],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_download_model_list_mentions_current_second_family(self) -> None:
        result = self._run("scripts/download_model.py", "--list")

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("mistralai/Mistral-7B-Instruct-v0.3", result.stdout)
        self.assertIn("mistralai/Mistral-Nemo-Instruct-2407", result.stdout)

    def test_generate_adapter_help_is_available(self) -> None:
        result = self._run("scripts/generate_lora_adapters.py", "--help")

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--use-peft", result.stdout)
        self.assertIn("--finetune", result.stdout)

    def test_generator_defaults_align_with_active_profile_selection(self) -> None:
        defaults = resolve_generation_defaults(EXPERIMENTS_CONFIG)
        active_model = self.experiments["model_profiles"]["qwen_14b_tp2"]["model"]["name"]

        self.assertEqual(defaults["model"], active_model)
        self.assertEqual(defaults["num_adapters"], 100)
        self.assertEqual(defaults["generation_mode"], "peft_finetune")

    def test_active_qwen14_profile_uses_mp_tp2(self) -> None:
        active_model = self.experiments["model_profiles"]["qwen_14b_tp2"]["model"]

        self.assertEqual(active_model["tensor_parallel_size"], 2)
        self.assertEqual(active_model["distributed_executor_backend"], "mp")
        self.assertEqual(active_model["gpu_memory_utilization"], 0.85)

    def test_current_mistral_mainline_profile_uses_500_adapters(self) -> None:
        workload = self.experiments["workload_profiles"]["mistral_7b_auto500_main"]

        self.assertEqual(workload["lora_adapters"]["selected_num_adapters"], 500)
        self.assertEqual(workload["lora_adapters"]["full_num_adapters"], 500)
        self.assertEqual(workload["resource_coordination"]["max_instances"], 2)

    def test_paper_mainline_defaults_to_one_shot_preparation(self) -> None:
        adapters = self.experiments["lora_adapters"]

        self.assertEqual(adapters["generation_mode"], "peft_finetune")
        self.assertEqual(adapters["preparation_mode"], "one_shot")


if __name__ == "__main__":
    unittest.main()
