"""Basic smoke tests that run in the stable mainline environment."""

from __future__ import annotations

import os
import asyncio
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from faaslora.api.http_server import HTTPServer
from faaslora.memory.gpu_monitor import GPUMemoryMonitor
from faaslora.api.grpc_server import GRPCServer
from faaslora.coordination.coordinator import Coordinator
from faaslora.datasets.huggingface_adapter import HuggingFaceAdapter
from faaslora.registry.artifact_registry import ArtifactRegistry
from faaslora.scheduling.resource_coordinator import ResourceCoordinator
from faaslora.serving.inference_engine import InferenceEngine
from faaslora.serving.vllm_wrapper import VLLMWrapper
from faaslora.storage.remote_client import RemoteStorageClient
from faaslora.storage.s3_client import S3Client
from faaslora.storage.storage_manager import StorageManager
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
        full = next(
            scenario for scenario in self.experiments["scenarios"]
            if scenario["name"] == "faaslora_full"
        )

        self.assertEqual(resource["instance_mode"], "auto")
        self.assertEqual(resource["max_instances"], 2)
        self.assertTrue(resource["effective_capacity_admission_enabled"])
        self.assertEqual(workload["sampling_strategy"], "representative")
        self.assertEqual(workload["total_requests"], 1000)
        self.assertEqual(workload["concurrency"], 8)
        self.assertEqual(workload["time_scale_factor"], 0.02)
        self.assertEqual(model["max_model_len"], 2048)
        self.assertEqual(model["max_loras"], 8)
        self.assertEqual(model["max_num_seqs"], 8)
        self.assertEqual(model["max_num_batched_tokens"], 4096)
        self.assertEqual(model["runtime_concurrency_cap"], 8)
        self.assertTrue(
            full["resource_coordination"]["effective_capacity_admission_enabled"]
        )

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


class CoordinationSmokeTests(unittest.TestCase):
    def test_residency_manager_available_memory_still_reserves_load_headroom(self) -> None:
        class DummyResidencyManager:
            @staticmethod
            def get_tier_status(_tier):
                return {"capacity": {"free_bytes": 1024 * 1024 * 1024}}

        coord = ResourceCoordinator(
            config={"gpu_budget_mb": 1000, "lora_load_reserve_ratio": 0.10},
            residency_manager=DummyResidencyManager(),
        )
        self.assertAlmostEqual(coord._available_mb(), 924.0, places=3)


class LifecycleCompatibilityTests(unittest.TestCase):
    def test_components_expose_stop_for_top_level_coordinator(self) -> None:
        self.assertTrue(hasattr(GPUMemoryMonitor, "stop"))
        self.assertTrue(hasattr(ArtifactRegistry, "stop"))
        self.assertTrue(hasattr(InferenceEngine, "stop"))

    def test_grpc_server_does_not_claim_bindings_exist(self) -> None:
        class DummyCoordinator:
            pass

        server = GRPCServer(DummyCoordinator(), Config(str(DEFAULT_CONFIG)))
        self.assertFalse(server._servicer_registration_available())

    def test_http_server_start_fails_fast_without_fastapi(self) -> None:
        class DummyCoordinator:
            is_running = False

        server = HTTPServer(DummyCoordinator(), Config(str(DEFAULT_CONFIG)))
        with patch("faaslora.api.http_server.FASTAPI_AVAILABLE", False):
            with self.assertRaisesRegex(RuntimeError, "FastAPI not available"):
                asyncio.run(server.start())

    def test_coordinator_raises_when_inference_engine_fails_startup(self) -> None:
        class DummyRegistry:
            def __init__(self, _config) -> None:
                pass

            async def start(self) -> None:
                return None

            async def stop(self) -> None:
                return None

        class DummyGPUMonitor:
            enabled = True

            def __init__(self, _config) -> None:
                pass

            async def start(self) -> None:
                return None

            async def stop(self) -> None:
                return None

        class DummyMetricsCollector:
            enabled = True

            def __init__(self, _config) -> None:
                pass

            async def start(self) -> None:
                return None

            async def stop(self) -> None:
                return None

        class DummyResidencyManager:
            def __init__(self, **_kwargs) -> None:
                pass

            async def start(self) -> None:
                return None

            async def stop(self) -> None:
                return None

        class DummyPreloadingPlanner:
            def __init__(self, **_kwargs) -> None:
                pass

        class DummyPreloadingManager:
            def __init__(self, **_kwargs) -> None:
                pass

            async def start(self) -> None:
                return None

            async def stop(self) -> None:
                return None

        class DummyInferenceEngine:
            def __init__(self, **_kwargs) -> None:
                pass

            async def start(self) -> bool:
                return False

            async def stop(self) -> None:
                return None

        class DummyAutoScaler:
            enabled = True

            def __init__(self, **_kwargs) -> None:
                pass

            async def start(self) -> None:
                return None

            async def stop(self) -> None:
                return None

        with patch("faaslora.coordination.coordinator.ArtifactRegistry", DummyRegistry), \
             patch("faaslora.coordination.coordinator.GPUMemoryMonitor", DummyGPUMonitor), \
             patch("faaslora.coordination.coordinator.MetricsCollector", DummyMetricsCollector), \
             patch("faaslora.coordination.coordinator.ResidencyManager", DummyResidencyManager), \
             patch("faaslora.coordination.coordinator.PreloadingPlanner", DummyPreloadingPlanner), \
             patch("faaslora.coordination.coordinator.PreloadingManager", DummyPreloadingManager), \
             patch("faaslora.coordination.coordinator.InferenceEngine", DummyInferenceEngine), \
             patch("faaslora.coordination.coordinator.AutoScaler", DummyAutoScaler):
            coordinator = Coordinator(Config(str(DEFAULT_CONFIG)))
            with self.assertRaisesRegex(RuntimeError, "Inference engine failed to start"):
                asyncio.run(coordinator._initialize_components())


class DependencyFailFastTests(unittest.TestCase):
    def test_vllm_wrapper_initialize_fails_when_vllm_missing(self) -> None:
        wrapper = VLLMWrapper(Config(str(DEFAULT_CONFIG)), registry=None, gpu_monitor=None)
        with patch("faaslora.serving.vllm_wrapper.VLLM_AVAILABLE", False):
            ok = asyncio.run(wrapper.initialize())
        self.assertFalse(ok)

    def test_huggingface_adapter_does_not_create_mock_hub_client(self) -> None:
        with patch("faaslora.datasets.huggingface_adapter.HF_HUB_AVAILABLE", False):
            adapter = HuggingFaceAdapter(Config(str(DEFAULT_CONFIG)))
        self.assertIsNone(adapter.hf_api)

    def test_s3_client_initialize_fails_fast_when_boto3_missing(self) -> None:
        client = S3Client(Config(str(DEFAULT_CONFIG)))
        with patch("faaslora.storage.s3_client.BOTO3_AVAILABLE", False):
            with self.assertRaisesRegex(RuntimeError, "boto3 not available"):
                asyncio.run(client.initialize())

    def test_storage_manager_skips_s3_when_remote_backend_is_not_s3(self) -> None:
        class DummyLocalCache:
            def __init__(self, _config) -> None:
                pass

            async def initialize(self) -> None:
                return None

            async def cleanup(self) -> None:
                return None

        class DummyRemoteClient:
            initialized = False

            def __init__(self, _config) -> None:
                pass

            async def initialize(self) -> None:
                self.initialized = True

            async def cleanup(self) -> None:
                return None

        class DummyS3Client:
            def __init__(self, _config) -> None:
                pass

            async def initialize(self) -> None:
                raise AssertionError("S3 backend should not be initialized")

            async def cleanup(self) -> None:
                return None

        cfg = Config(str(DEFAULT_CONFIG))
        cfg.set("storage.provider", "none")
        cfg.set("storage.remote.provider", "none")

        with patch("faaslora.storage.storage_manager.LocalCache", DummyLocalCache), \
             patch("faaslora.storage.storage_manager.RemoteStorageClient", DummyRemoteClient), \
             patch("faaslora.storage.storage_manager.S3Client", DummyS3Client), \
             patch.object(StorageManager, "_update_stats", autospec=True) as update_stats:
            update_stats.return_value = None
            manager = StorageManager(cfg)
            manager.auto_cleanup_enabled = False
            manager.prefetch_enabled = False
            asyncio.run(manager.initialize())
            self.assertEqual(manager.remote_backend, "remote_client")
            self.assertTrue(manager.is_running)
            asyncio.run(manager.shutdown())

    def test_storage_manager_default_config_uses_s3_backend_without_remote_client(self) -> None:
        class DummyLocalCache:
            def __init__(self, _config) -> None:
                pass

            async def initialize(self) -> None:
                return None

            async def cleanup(self) -> None:
                return None

        class DummyS3Client:
            initialized = False

            def __init__(self, _config) -> None:
                pass

            async def initialize(self) -> None:
                self.initialized = True

            async def cleanup(self) -> None:
                return None

        with patch("faaslora.storage.storage_manager.LocalCache", DummyLocalCache), \
             patch("faaslora.storage.storage_manager.S3Client", DummyS3Client), \
             patch("faaslora.storage.storage_manager.RemoteStorageClient", side_effect=AssertionError("RemoteStorageClient should not be constructed for s3 backend")), \
             patch.object(StorageManager, "_update_stats", autospec=True) as update_stats:
            update_stats.return_value = None
            manager = StorageManager(Config(str(DEFAULT_CONFIG)))
            manager.auto_cleanup_enabled = False
            manager.prefetch_enabled = False
            self.assertEqual(manager.remote_backend, "s3")
            self.assertIsNone(manager.remote_client)
            asyncio.run(manager.initialize())
            self.assertTrue(manager.is_running)
            asyncio.run(manager.shutdown())

    def test_remote_client_rejects_unknown_provider(self) -> None:
        cfg = Config(str(DEFAULT_CONFIG))
        cfg.set("storage.remote.provider", "bad_provider")
        with self.assertRaisesRegex(ValueError, "Unsupported remote storage provider"):
            RemoteStorageClient(cfg)

    def test_remote_client_huggingface_requires_dependency(self) -> None:
        cfg = Config(str(DEFAULT_CONFIG))
        cfg.set("storage.remote.provider", "huggingface")
        client = RemoteStorageClient(cfg)
        with patch("faaslora.storage.remote_client.importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "requires huggingface_hub"):
                asyncio.run(client.initialize())

    def test_s3_client_accepts_legacy_storage_remote_config(self) -> None:
        cfg = Config(str(DEFAULT_CONFIG))
        cfg.set("storage.remote.provider", "s3")
        cfg.set("storage.remote.bucket", "legacy-bucket")
        cfg.set("storage.remote.region", "ap-southeast-1")
        cfg.set("storage.remote.access_key", "legacy-ak")
        cfg.set("storage.remote.secret_key", "legacy-sk")

        client = S3Client(cfg)

        self.assertEqual(client.bucket_name, "legacy-bucket")
        self.assertEqual(client.region, "ap-southeast-1")
        self.assertEqual(client.access_key, "legacy-ak")
        self.assertEqual(client.secret_key, "legacy-sk")


if __name__ == "__main__":
    unittest.main()
