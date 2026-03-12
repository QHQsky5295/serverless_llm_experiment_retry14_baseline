"""Basic smoke tests that run in the stable mainline environment."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from faaslora.utils.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_CONFIG = PROJECT_ROOT / "configs" / "experiments.yaml"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class MainlineConfigSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with EXPERIMENTS_CONFIG.open("r", encoding="utf-8") as f:
            cls.experiments = yaml.safe_load(f)

    def test_mainline_defaults_match_frozen_path(self) -> None:
        model = self.experiments["model"]
        resource = self.experiments["resource_coordination"]
        workload = self.experiments["workload"]

        self.assertEqual(resource["instance_mode"], "auto")
        self.assertEqual(resource["max_instances"], 2)
        self.assertEqual(workload["sampling_strategy"], "representative")
        self.assertEqual(workload["total_requests"], 1000)
        self.assertEqual(workload["concurrency"], 8)
        self.assertEqual(workload["time_scale_factor"], 0.02)
        self.assertEqual(model["max_model_len"], 2048)
        self.assertEqual(model["max_loras"], 8)
        self.assertEqual(model["max_num_seqs"], 8)
        self.assertEqual(model["max_num_batched_tokens"], 4096)
        self.assertEqual(model["runtime_concurrency_cap"], 8)

    def test_scale_preset_500_matches_mainline_serving_parameters(self) -> None:
        preset = self.experiments["lora_adapters"]["scale_presets"]["500"]["model"]

        self.assertEqual(preset["max_loras"], 8)
        self.assertEqual(preset["max_num_seqs"], 8)
        self.assertEqual(preset["max_num_batched_tokens"], 4096)
        self.assertEqual(preset["runtime_concurrency_cap"], 8)


class ConfigSmokeTests(unittest.TestCase):
    def test_default_config_accepts_env_override(self) -> None:
        with patch.dict(os.environ, {"FAASLORA_API_HTTP_PORT": "18080"}, clear=False):
            cfg = Config(str(DEFAULT_CONFIG))
        self.assertEqual(cfg.get("api.http.port"), 18080)


class CliSmokeTests(unittest.TestCase):
    def _run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "faaslora.cli", *args],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_top_level_help_is_available(self) -> None:
        result = self._run_cli("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("coordinator", result.stdout)

    def test_coordinator_help_is_available(self) -> None:
        result = self._run_cli("coordinator", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--config", result.stdout)
        self.assertIn("--host", result.stdout)
        self.assertIn("--port", result.stdout)


if __name__ == "__main__":
    unittest.main()
