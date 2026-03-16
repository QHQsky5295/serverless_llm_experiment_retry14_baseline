"""Basic smoke tests that run in the stable mainline environment."""

from __future__ import annotations

import os
import asyncio
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from tempfile import TemporaryDirectory

import yaml

from faaslora.api.http_server import HTTPServer
from faaslora.memory.gpu_monitor import GPUMemoryMonitor
from faaslora.api.grpc_server import GRPCServer
from faaslora.coordination.coordinator import Coordinator
from faaslora.datasets.huggingface_adapter import HuggingFaceAdapter
from faaslora.datasets.azure_functions_adapter import AzureFunctionsAdapter
from faaslora.datasets.azure_llm_adapter import AzureLLMAdapter
from faaslora.datasets.dataset_loader import WorkloadDataset
from faaslora.registry.artifact_registry import ArtifactRegistry
from faaslora.scheduling.resource_coordinator import ResourceCoordinator
from faaslora.serving.inference_engine import InferenceEngine
from faaslora.serving.vllm_wrapper import VLLMWrapper
from faaslora.storage.remote_client import RemoteStorageClient
from faaslora.storage.s3_client import S3Client
from faaslora.storage.storage_manager import StorageManager
from faaslora.utils.config import Config
from scripts.generate_lora_adapters import resolve_generation_defaults


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_CONFIG = PROJECT_ROOT / "configs" / "experiments.yaml"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class MainlineConfigSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with EXPERIMENTS_CONFIG.open("r", encoding="utf-8") as f:
            cls.experiments = yaml.safe_load(f)

    def test_mainline_defaults_match_frozen_path(self) -> None:
        model, workload, resource, lora_cfg = self._resolve_active_profiles()

        self.assertEqual(resource["instance_mode"], "auto")
        self.assertEqual(resource["max_instances"], 1)
        self.assertTrue(resource["effective_capacity_admission_enabled"])
        self.assertEqual(workload["sampling_strategy"], "representative")
        self.assertEqual(workload["total_requests"], 1000)
        self.assertEqual(workload["concurrency"], 2)
        self.assertEqual(workload["time_scale_factor"], 0.02)
        self.assertEqual(model["name"], self.experiments["model_profiles"]["qwen_14b_tp2"]["model"]["name"])
        self.assertEqual(model["tensor_parallel_size"], 2)
        self.assertEqual(model["max_model_len"], 1024)
        self.assertEqual(model["max_loras"], 2)
        self.assertEqual(model["max_num_seqs"], 2)
        self.assertEqual(model["max_num_batched_tokens"], 1024)
        self.assertEqual(model["runtime_concurrency_cap"], 2)
        self.assertEqual(lora_cfg["full_num_adapters"], 100)

    def test_scale_preset_500_matches_mainline_serving_parameters(self) -> None:
        preset = self.experiments["lora_adapters"]["scale_presets"]["500"]["model"]

        self.assertEqual(preset["max_loras"], 8)
        self.assertEqual(preset["max_num_seqs"], 8)
        self.assertEqual(preset["max_num_batched_tokens"], 4096)
        self.assertEqual(preset["runtime_concurrency_cap"], 8)

    def test_dataset_controls_are_yaml_driven(self) -> None:
        datasets = self.experiments["datasets"]

        self.assertEqual(datasets["arrival_source"], "azure_llm")
        self.assertEqual(datasets["token_source"], "azure_llm")
        self.assertEqual(datasets["prompt_source"], "sharegpt_auto")
        self.assertEqual(datasets["sharegpt_max_records"], 5000)

    def test_profile_selection_defaults_exist(self) -> None:
        selection = self.experiments["profile_selection"]
        model_profiles = self.experiments["model_profiles"]
        dataset_profiles = self.experiments["dataset_profiles"]
        workload_profiles = self.experiments["workload_profiles"]

        self.assertEqual(selection["model"], "qwen_14b_tp2")
        self.assertEqual(selection["dataset"], "azure_sharegpt_rep1000")
        self.assertEqual(selection["workload"], "qwen_14b_tp2_main")
        self.assertIn("qwen_14b_tp2", model_profiles)
        self.assertIn("azure_sharegpt_rep4000", dataset_profiles)
        self.assertIn("qwen_14b_tp2_main", workload_profiles)

    def test_scale_preset_can_be_disabled_for_new_model_profiles(self) -> None:
        self.assertTrue(self.experiments["lora_adapters"]["apply_scale_preset"])
        qwen14 = self.experiments["workload_profiles"]["qwen_14b_tp2_main"]
        self.assertFalse(qwen14["lora_adapters"]["apply_scale_preset"])

    def test_paper_mainline_defaults_to_peft_finetune_artifacts(self) -> None:
        adapters = self.experiments["lora_adapters"]

        self.assertEqual(adapters["generation_mode"], "peft_finetune")
        self.assertEqual(adapters["preparation_mode"], "one_shot")
        self.assertFalse(adapters["generate_synthetic"])

    def test_generator_defaults_follow_selected_profile(self) -> None:
        defaults = resolve_generation_defaults(EXPERIMENTS_CONFIG)

        self.assertEqual(
            defaults["model"],
            self.experiments["model_profiles"]["qwen_14b_tp2"]["model"]["name"],
        )
        self.assertEqual(defaults["num_adapters"], 100)
        self.assertEqual(defaults["generation_mode"], "peft_finetune")

    def test_peft_batch_generator_loads_base_model_once(self) -> None:
        from scripts import generate_lora_adapters as generator

        class FakeTokenizer:
            def save_pretrained(self, dest: str) -> None:
                Path(dest, "tokenizer_config.json").write_text("{}", encoding="utf-8")

        class FakePeftModel:
            def __init__(self) -> None:
                self.adapters = []
                self.saved = []

            def add_adapter(self, adapter_name, _config, low_cpu_mem_usage=False):
                self.adapters.append(adapter_name)

            def set_adapter(self, adapter_name):
                self.adapters.append(f"active:{adapter_name}")

            def save_pretrained(self, save_directory, selected_adapters=None, **_kwargs):
                dest = Path(save_directory)
                dest.mkdir(parents=True, exist_ok=True)
                Path(dest, "adapter_config.json").write_text("{}", encoding="utf-8")
                self.saved.append(tuple(selected_adapters or []))

            def delete_adapter(self, adapter_name):
                self.adapters.append(f"deleted:{adapter_name}")

        fake_model = FakePeftModel()

        class FakePeftModule:
            @staticmethod
            def get_peft_model(_base_model, _config, adapter_name="default", **_kwargs):
                fake_model.adapters.append(f"init:{adapter_name}")
                return fake_model

        adapter_specs = [
            {"adapter_id": "a1", "rank": 8},
            {"adapter_id": "a2", "rank": 8},
            {"adapter_id": "a3", "rank": 8},
        ]

        with TemporaryDirectory() as tmpdir:
            with patch.object(
                generator,
                "_load_peft_base_model",
                return_value=(object(), FakeTokenizer(), ["q_proj"]),
            ) as load_once, patch.object(
                generator,
                "_create_lora_config",
                return_value=object(),
            ), patch.object(
                generator,
                "ensure_adapter_support_files",
                return_value=None,
            ), patch.dict(
                sys.modules,
                {"peft": FakePeftModule},
            ):
                timings = generator.generate_adapters_with_peft(
                    model_name="dummy-model",
                    output_dir=Path(tmpdir),
                    adapter_specs=adapter_specs,
                    finetune=False,
                )

        load_once.assert_called_once()
        self.assertEqual(set(timings.keys()), {"a1", "a2", "a3"})
        self.assertEqual(fake_model.saved, [("a1",), ("a2",), ("a3",)])

    def test_runner_one_shot_preparation_builds_generator_command(self) -> None:
        from scripts import run_all_experiments as runner

        adapters_cfg = {
            "generation_mode": "peft_finetune",
            "preparation_mode": "one_shot",
            "_manifest_path": "configs/generated/lora_manifest_1000.json",
            "adapters": [{"id": "finance_lora", "lora_rank": 8}],
        }

        with TemporaryDirectory() as tmpdir:
            remote_dir = Path(tmpdir)
            with patch.object(runner.subprocess, "run") as mocked_run:
                mocked_run.return_value.returncode = 0
                runner._auto_prepare_peft_artifacts(
                    adapters_cfg=adapters_cfg,
                    remote_dir=remote_dir,
                    model_name="dummy-model",
                    generation_mode="peft_finetune",
                    pending_adapters=adapters_cfg["adapters"],
                    model_cfg={"visible_device_ids": [0, 1]},
                )

        mocked_run.assert_called_once()
        args, kwargs = mocked_run.call_args
        cmd = args[0]
        env = kwargs["env"]

        self.assertTrue(str(cmd[1]).endswith("scripts/generate_lora_adapters.py"))
        self.assertIn("--model", cmd)
        self.assertIn("dummy-model", cmd)
        self.assertIn("--num-adapters", cmd)
        self.assertIn("1", cmd)
        self.assertIn("--use-peft", cmd)
        self.assertIn("--finetune", cmd)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")

    def _resolve_active_profiles(self):
        def merge(base, override):
            merged = dict(base)
            for key, value in override.items():
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    merged[key] = merge(merged[key], value)
                else:
                    merged[key] = value
            return merged

        selection = self.experiments["profile_selection"]
        model = merge(
            self.experiments["model"],
            self.experiments["model_profiles"][selection["model"]]["model"],
        )
        workload = merge(
            self.experiments["workload"],
            self.experiments["dataset_profiles"][selection["dataset"]]["workload"],
        )
        workload = merge(
            workload,
            self.experiments["workload_profiles"][selection["workload"]]["workload"],
        )
        resource = merge(
            self.experiments["resource_coordination"],
            self.experiments["workload_profiles"][selection["workload"]]["resource_coordination"],
        )
        lora_cfg = merge(
            self.experiments["lora_adapters"],
            self.experiments["workload_profiles"][selection["workload"]].get("lora_adapters", {}),
        )
        return model, workload, resource, lora_cfg


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


class DatasetParsingTests(unittest.TestCase):
    def test_workload_dataset_can_disable_azure_and_force_embedded_prompts(self) -> None:
        dataset = WorkloadDataset()

        stats = dataset.initialize(
            load_azure=False,
            max_sgpt=16,
            prompt_source="embedded",
        )

        self.assertFalse(dataset.has_real_azure_data())
        self.assertFalse(dataset.has_real_sharegpt_data())
        self.assertEqual(stats["azure"]["total_records"], 0)
        self.assertEqual(stats["sharegpt"]["source"], "embedded")
        self.assertGreater(stats["sharegpt"]["total_records"], 0)

    def test_workload_dataset_rejects_unknown_prompt_source(self) -> None:
        dataset = WorkloadDataset()

        with self.assertRaisesRegex(ValueError, "Unsupported ShareGPT prompt source"):
            dataset.initialize(prompt_source="unknown_source")

    def test_azure_functions_adapter_parses_string_booleans(self) -> None:
        adapter = AzureFunctionsAdapter(Config(str(DEFAULT_CONFIG)))
        invocation = adapter._parse_invocation_row(
            {
                "timestamp": "0",
                "success": "False",
                "cold_start": "1",
            }
        )
        self.assertIsNotNone(invocation)
        self.assertFalse(invocation.success)
        self.assertTrue(invocation.cold_start)

    def test_azure_llm_adapter_parses_string_booleans(self) -> None:
        adapter = AzureLLMAdapter(Config(str(DEFAULT_CONFIG)))
        request = adapter._parse_request_data(
            {
                "timestamp": "0",
                "success": "0",
                "prompt_tokens": 1,
                "completion_tokens": 1,
            }
        )
        self.assertIsNotNone(request)
        self.assertFalse(request.success)

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
