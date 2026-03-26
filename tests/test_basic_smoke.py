"""Basic smoke tests that run in the stable mainline environment."""

from __future__ import annotations

import os
import asyncio
import json
import subprocess
import sys
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import yaml

from faaslora.api.http_server import HTTPServer
from faaslora.memory.gpu_monitor import GPUMemoryInfo, GPUMemoryMonitor
from faaslora.memory.residency_manager import ResidencyManager
from faaslora.api.grpc_server import GRPCServer
from faaslora.coordination.autoscaler import AutoScaler
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
from faaslora.experiment.experiment_stack import ExperimentStack
from faaslora.preloading.preloading_planner import PreloadingPlanner
from faaslora.registry.schema import ArtifactMetadata, ArtifactStatus, StorageTier
from faaslora.utils.model_assets import ensure_adapter_support_files
from faaslora.utils.config import Config
from scripts.generate_lora_adapters import (
    _build_adapter_specs,
    _normalize_saved_adapter_weights,
    resolve_generation_defaults,
)
from scripts.prepare_publicmix_pool import (
    _build_publicmix_pool,
    build_publicmix_manifest,
    model_refs_match,
    scan_public_adapter_pool,
)
from scripts.run_all_experiments import (
    RequestResult,
    ScenarioResult,
    ScenarioRunner,
    _apply_tp_instance_capacity_guard,
    _adapter_matches_model,
    _build_local_tp_runtime_env_updates,
    _is_comparable_request,
    _normalize_lora_preparation_mode,
    _prepare_dedicated_subprocess_model_cfg,
    _resolve_vllm_runtime_guards,
    _select_adaptive_retry_gpu_util,
    _should_spawn_dedicated_engine_subprocess,
)


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
        selected_model = self.experiments["model_profiles"]["qwen_14b_tp2_v2_publicmix"]["model"]

        self.assertEqual(resource["instance_mode"], "auto")
        self.assertEqual(resource["max_instances"], 1)
        self.assertTrue(resource["effective_capacity_admission_enabled"])
        self.assertEqual(workload["sampling_strategy"], "representative")
        self.assertEqual(workload["total_requests"], 1000)
        self.assertEqual(workload["concurrency"], 2)
        self.assertEqual(workload["time_scale_factor"], 0.02)
        self.assertEqual(model["name"], selected_model["name"])
        self.assertEqual(model["tensor_parallel_size"], 2)
        self.assertEqual(model["max_model_len"], 1024)
        self.assertEqual(model["max_loras"], 2)
        self.assertEqual(model["max_num_seqs"], 2)
        self.assertEqual(model["max_num_batched_tokens"], 1024)
        self.assertEqual(model["runtime_concurrency_cap"], 2)
        self.assertEqual(lora_cfg["full_num_adapters"], 500)
        faaslora_full = next(s for s in self.experiments["scenarios"] if s["name"] == "faaslora_full")
        self.assertTrue(faaslora_full["preloading"]["dynamic_forwarding_enabled"])

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

        self.assertEqual(selection["model"], "qwen_14b_tp2_v2_publicmix")
        self.assertEqual(selection["dataset"], "azure_sharegpt_rep1000")
        self.assertEqual(selection["workload"], "qwen_14b_tp2_a500_main")
        self.assertIn("qwen_14b_tp2_v2_publicmix", model_profiles)
        self.assertIn("azure_sharegpt_rep4000", dataset_profiles)
        self.assertIn("qwen_14b_tp2_a500_main", workload_profiles)

    def test_scale_preset_can_be_disabled_for_new_model_profiles(self) -> None:
        self.assertTrue(self.experiments["lora_adapters"]["apply_scale_preset"])
        qwen14 = self.experiments["workload_profiles"]["qwen_14b_tp2_main"]
        self.assertFalse(qwen14["lora_adapters"]["apply_scale_preset"])

    def test_paper_mainline_defaults_to_peft_finetune_artifacts(self) -> None:
        adapters = self.experiments["lora_adapters"]

        self.assertEqual(adapters["generation_mode"], "peft_finetune")
        self.assertEqual(adapters["preparation_mode"], "two_phase")
        self.assertEqual(_normalize_lora_preparation_mode(adapters), "two_phase")
        self.assertFalse(adapters["generate_synthetic"])

    def test_active_profile_uses_model_specific_frozen_remote_dir(self) -> None:
        selection = self.experiments["profile_selection"]
        model_profile = self.experiments["model_profiles"][selection["model"]]
        storage = model_profile.get("storage", {})

        self.assertEqual(storage["remote_dir"], "artifacts/frozen/qwen_14b_a500_v2_publicmix")

    def test_mistral_nemo_profile_uses_conservative_vllm_path(self) -> None:
        model = self.experiments["model_profiles"]["mistral_nemo_12b_tp2"]["model"]
        guards = _resolve_vllm_runtime_guards(model)

        self.assertEqual(model["max_loras"], 1)
        self.assertEqual(model["max_num_seqs"], 1)
        self.assertEqual(model["runtime_concurrency_cap"], 1)
        self.assertEqual(guards["tokenizer_mode"], "mistral")
        self.assertFalse(guards["enable_chunked_prefill"])
        self.assertFalse(guards["enable_prefix_caching"])
        self.assertEqual(guards["env_updates"]["VLLM_USE_V1"], "0")
        self.assertEqual(guards["env_updates"]["VLLM_ATTENTION_BACKEND"], "FLASH_ATTN")
        self.assertEqual(guards["env_updates"]["VLLM_USE_FLASHINFER_SAMPLER"], "0")

    def test_tp_capacity_guard_caps_dual_gpu_tp2_to_single_instance(self) -> None:
        model = {"tensor_parallel_size": 2, "visible_device_ids": [0, 1]}
        coord = {"min_instances": 1, "max_instances": 2}

        guarded, meta = _apply_tp_instance_capacity_guard(
            model,
            coord,
            fallback_gpu_count=2,
        )

        self.assertEqual(guarded["max_instances"], 1)
        self.assertEqual(guarded["min_instances"], 1)
        self.assertEqual(meta["max_tp_instances"], 1)

    def test_tp_runtime_env_prefers_loopback_for_local_mp(self) -> None:
        env = _build_local_tp_runtime_env_updates(tp=2, executor_backend="mp")

        self.assertEqual(env["MASTER_ADDR"], "127.0.0.1")
        self.assertEqual(env["VLLM_HOST_IP"], "127.0.0.1")
        self.assertEqual(env["GLOO_SOCKET_IFNAME"], "lo")
        self.assertEqual(env["NCCL_SOCKET_IFNAME"], "lo")

    def test_adaptive_retry_gpu_util_preserves_requested_value_with_sufficient_headroom(self) -> None:
        self.assertEqual(_select_adaptive_retry_gpu_util(0.85, 0.98), 0.85)

    def test_adaptive_retry_gpu_util_tracks_observed_headroom(self) -> None:
        self.assertEqual(_select_adaptive_retry_gpu_util(0.85, 0.728), 0.72)

    def test_tp1_dedicated_vllm_uses_subprocess_isolation(self) -> None:
        model = {"backend": "vllm", "tensor_parallel_size": 1}

        self.assertTrue(
            _should_spawn_dedicated_engine_subprocess(model, instance_mode="auto")
        )
        self.assertTrue(
            _should_spawn_dedicated_engine_subprocess(model, instance_mode="dedicated")
        )
        self.assertFalse(
            _should_spawn_dedicated_engine_subprocess(model, instance_mode="shared")
        )

    def test_prepare_dedicated_subprocess_model_cfg_remaps_tp1_gpu_to_local_zero(self) -> None:
        model = {
            "backend": "vllm",
            "tensor_parallel_size": 1,
            "device_id": 0,
            "visible_device_ids": [0, 1],
        }

        local_model, env = _prepare_dedicated_subprocess_model_cfg(
            model,
            device_id=1,
        )

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["FAASLORA_VISIBLE_DEVICES"], "1")
        self.assertEqual(local_model["device_id"], 0)
        self.assertEqual(local_model["visible_device_ids"], [0])

    def test_auto_scale_up_does_not_fallback_to_shared_slot_after_dedicated_failure(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.instance_pool = SimpleNamespace(count=lambda: 1, max_instances=2)
        runner._instance_mode = "auto"
        runner.engine_factory = object()
        calls = []

        async def fake_add_dedicated(coord_enabled: bool):
            calls.append(("dedicated", coord_enabled))
            return None

        async def fake_add_shared(coord_enabled: bool):
            calls.append(("shared", coord_enabled))
            return {"event_type": "logical_scale_up"}

        runner._add_dedicated_instance_slot = fake_add_dedicated
        runner._add_shared_instance_slot = fake_add_shared

        result = asyncio.run(runner._scale_up_instance_pool(coord_enabled=True))

        self.assertIsNone(result)
        self.assertEqual(calls, [("dedicated", True)])

    def test_ensure_min_instances_stops_after_dedicated_failure_in_auto_mode(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.instance_pool = SimpleNamespace(count=lambda: 1, min_instances=2)
        runner._instance_mode = "auto"
        runner.engine_factory = object()
        calls = []

        async def fake_add_dedicated(coord_enabled: bool):
            calls.append(("dedicated", coord_enabled))
            return None

        async def fake_add_shared(coord_enabled: bool):
            calls.append(("shared", coord_enabled))
            return {"event_type": "logical_scale_up"}

        runner._add_dedicated_instance_slot = fake_add_dedicated
        runner._add_shared_instance_slot = fake_add_shared

        asyncio.run(runner._ensure_min_instances(coord_enabled=False))

        self.assertEqual(calls, [("dedicated", False)])

    def test_refresh_slot_runtime_hints_syncs_real_resident_and_shared_local_tiers(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        sync_calls = []
        runner._stack = SimpleNamespace(
            sync_local_tier_paths=lambda: sync_calls.append("sync"),
            _host_paths={"host_a": "/tmp/host_a"},
            _nvme_paths={"nvme_b": "/tmp/nvme_b"},
        )
        runner._gpu_runtime_snapshot = lambda device_id: (21.0, 24.0, 88.0) if device_id == 1 else None
        runner._instance_mode = "auto"

        slot = SimpleNamespace(
            coordinator=SimpleNamespace(
                get_summary_metrics=lambda: {
                    "queued_loads": 2,
                    "current_lora_resident_mb": 96.0,
                    "current_gpu_utilization_pct": 41.0,
                },
                _get_resident_loras=lambda: {"gpu_hot": 32.0},
            ),
            device_id=1,
            gpu_resident_adapters={"stale_gpu"},
            host_cached_adapters=set(),
            nvme_cached_adapters=set(),
        )

        def _update_runtime_hints(metrics):
            slot.metrics = dict(metrics)

        slot.update_runtime_hints = _update_runtime_hints

        runner._refresh_slot_runtime_hints(slot)

        self.assertEqual(sync_calls, ["sync"])
        self.assertEqual(slot.gpu_resident_adapters, {"gpu_hot"})
        self.assertEqual(slot.host_cached_adapters, {"host_a"})
        self.assertEqual(slot.nvme_cached_adapters, {"nvme_b"})
        self.assertEqual(slot.metrics["queued_loads"], 2)
        self.assertEqual(slot.metrics["current_gpu_utilization_pct"], 88.0)
        self.assertEqual(slot.metrics["physical_gpu_utilization_pct"], 88.0)

    def test_refresh_slot_runtime_hints_keeps_higher_logical_utilization(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._stack = None
        runner._gpu_runtime_snapshot = lambda _device_id: (10.0, 24.0, 35.0)
        runner._instance_mode = "auto"

        slot = SimpleNamespace(
            coordinator=SimpleNamespace(
                get_summary_metrics=lambda: {
                    "queued_loads": 0,
                    "current_lora_resident_mb": 64.0,
                    "current_gpu_utilization_pct": 61.0,
                },
                _get_resident_loras=lambda: {"gpu_hot": 32.0},
            ),
            device_id=0,
            gpu_resident_adapters=set(),
            host_cached_adapters=set(),
            nvme_cached_adapters=set(),
        )

        def _update_runtime_hints(metrics):
            slot.metrics = dict(metrics)

        slot.update_runtime_hints = _update_runtime_hints

        runner._refresh_slot_runtime_hints(slot)

        self.assertEqual(slot.metrics["current_gpu_utilization_pct"], 61.0)
        self.assertEqual(slot.metrics["physical_gpu_utilization_pct"], 35.0)

    def test_exec_request_preserves_source_cache_tier_after_gpu_promotion(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.router = None
        runner.engine = None
        runner._refresh_all_slot_runtime_hints = lambda: None
        runner._refresh_slot_runtime_hints = lambda _slot: None
        marks = []
        runner._mark_slot_adapter_tier = lambda _slot, aid, tier: marks.append((aid, tier))
        runner.adapter_info = {"finance_lora": {"size_mb": 30.0}}
        runner.baseline_type = "faaslora_full"
        runner._access_count = {"finance_lora": 0}
        runner.cost_model = {}

        class FakeRegistry:
            def __init__(self):
                self.updates = []

            def update_artifact(self, adapter_id: str, payload):
                self.updates.append((adapter_id, dict(payload)))

        access_log = []
        runner._stack = SimpleNamespace(
            registry=FakeRegistry(),
            residency_manager=SimpleNamespace(admit_artifact=lambda *args, **kwargs: None),
            record_access=lambda adapter_id, load_time_ms=0.0, hit=True: access_log.append((adapter_id, load_time_ms, hit)),
        )

        class FakeCoord:
            def __init__(self):
                self._residency_manager = None
                self.marked = []

            def notify_batch_start(self, _tokens: int):
                return None

            def notify_batch_end(self, _tokens: int):
                return None

            async def _mark_resident(self, adapter_id: str, size_mb: float):
                self.marked.append((adapter_id, size_mb))
                return True

        class FakeEngine:
            async def load_lora_to_gpu_and_measure(self, lora_path: str, adapter_id: str):
                return 5.0, True

            async def generate(self, prompt, lora_path, adapter_id, max_tokens, input_tokens, temperature):
                return 10.0, 1.0, 4

        async def fake_resolve(adapter_id, _is_burst, _expected_input_tokens, coordinator=None):
            return adapter_id, "/tmp/fake_lora", 123.0, "host", 0.0, 0.0

        runner._resolve_lora = fake_resolve
        runner.coordinator = FakeCoord()
        runner.engine = FakeEngine()

        trace = SimpleNamespace(
            request_id="req_00000",
            adapter_id="finance_lora",
            is_burst=False,
            expected_input_tokens=16,
            prompt="hello",
        )

        result = asyncio.run(runner._exec_request(trace, max_tokens=8, temperature=0.0))

        self.assertTrue(result.success)
        self.assertEqual(result.cache_tier, "host")
        self.assertTrue(result.cache_hit)
        self.assertEqual(result.lora_io_ms, 5.0)
        self.assertIn(("finance_lora", "host"), marks)
        self.assertIn(("finance_lora", "gpu"), marks)
        self.assertEqual(runner.coordinator.marked, [("finance_lora", 30.0)])
        self.assertEqual(access_log, [("finance_lora", 5.0, True)])

    def test_generator_defaults_follow_selected_profile(self) -> None:
        defaults = resolve_generation_defaults(EXPERIMENTS_CONFIG)

        self.assertEqual(
            defaults["model"],
            self.experiments["model_profiles"]["qwen_14b_tp2_v2_publicmix"]["model"]["name"],
        )
        self.assertEqual(defaults["num_adapters"], 500)
        self.assertEqual(defaults["generation_mode"], "peft_finetune")
        self.assertEqual(defaults["artifact_pool_profile"], "standardized_v1")

    def test_host_admission_materializes_real_host_path_and_syncs_stack_views(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            remote_dir = root / "remote"
            nvme_dir = root / "nvme"
            host_dir = root / "host"
            adapter_id = "finance_lora"
            src = remote_dir / adapter_id
            src.mkdir(parents=True, exist_ok=True)
            Path(src, "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct"}),
                encoding="utf-8",
            )
            Path(src, "adapter_model.bin").write_bytes(b"ok")

            stack = ExperimentStack(
                adapter_info={adapter_id: {"size_mb": 32.0, "hotness": 0.9}},
                hardware_cfg={"gpu_budget_mb": 24000},
                coord_cfg={"min_instances": 1, "max_instances": 2},
                preload_cfg={"strategy": "hybrid", "nvme_capacity_mb": 20480, "host_capacity_mb": 4096},
                remote_dir=remote_dir,
                nvme_dir=nvme_dir,
                host_dir=host_dir,
            )
            stack._ensure_registered()

            admitted = asyncio.run(stack.residency_manager.admit_artifact(adapter_id, StorageTier.HOST))
            self.assertTrue(admitted)

            meta = stack.registry.get_artifact(adapter_id)
            self.assertEqual(meta.storage_tier, StorageTier.HOST)
            self.assertTrue(str(meta.storage_path).startswith(str(host_dir)))
            self.assertTrue((host_dir / adapter_id).exists())

            stack.sync_local_tier_paths()
            self.assertIn(adapter_id, stack._host_paths)
            self.assertNotIn(adapter_id, stack._nvme_paths)

    def test_explicit_nvme_hit_schedules_host_promotion_without_utility_gate(self) -> None:
        stack = ExperimentStack.__new__(ExperimentStack)
        stack.config = SimpleNamespace(
            get=lambda key, default=None: {"max_concurrent_operations": 2}
            if key == "preloading"
            else default
        )
        stack.residency_manager = SimpleNamespace(
            get_tier_status=lambda _tier: {"capacity": {"total_bytes": 4096 * 1024 * 1024}}
        )
        stack.sync_local_tier_paths = lambda: None
        stack._nvme_paths = {"finance_lora": "/tmp/nvme/finance_lora"}
        stack._host_paths = {}
        stack._pending_host_promotions = {}
        stack._artifact_metadata = lambda _aid: SimpleNamespace(size_bytes=32 * 1024 * 1024)
        stack._effective_forward_budget_bytes = lambda *_args, **_kwargs: 0
        stack._forward_utility = lambda *_args, **_kwargs: 0.0
        stack._select_host_forward_candidate = lambda: None

        async def fake_promote(_aid: str) -> None:
            return None

        stack._promote_nvme_hit_to_host = fake_promote

        def fake_create_task(coro):
            coro.close()
            return SimpleNamespace(kind="task")

        with patch("faaslora.experiment.experiment_stack.asyncio.create_task", side_effect=fake_create_task):
            scheduled = stack._schedule_host_promotion_from_nvme("finance_lora")

        self.assertTrue(scheduled)
        self.assertIn("finance_lora", stack._pending_host_promotions)

    def test_knapsack_host_plan_uses_scaled_units_and_returns_candidates(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir, "planner.yaml")
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "system": {"name": "faaslora", "version": "1.0.0"},
                        "redis": {"host": "localhost", "port": 6379},
                        "memory": {
                            "gpu": {"enabled": True, "capacity_gb": 24},
                            "host": {"capacity_gb": 64},
                            "nvme": {"cache_size_gb": 50},
                        },
                        "serving": {"vllm": {"model_name": "stub-model", "tensor_parallel_size": 1}},
                        "api": {"http": {"host": "127.0.0.1", "port": 8000}},
                        "registry": {"backend": "memory"},
                        "preloading": {
                            "strategy": "hybrid",
                            "max_plan_size_gb": 4,
                            "min_hotness_threshold": 0.1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = Config(str(config_path))
            registry = ArtifactRegistry(config)
            planner = PreloadingPlanner(config=config, registry=registry)

            for idx, size_mb in enumerate((32, 48, 64), start=1):
                registry.register_artifact(
                    ArtifactMetadata(
                        artifact_id=f"a{idx}",
                        name=f"a{idx}",
                        size_bytes=size_mb * 1024 * 1024,
                        storage_tier=StorageTier.NVME,
                        storage_path=f"/tmp/a{idx}",
                        status=ArtifactStatus.AVAILABLE,
                        hotness_score=0.9,
                        value_per_byte=0.05,
                        predicted_load_time_ms=10.0,
                    )
                )

            plan = planner.generate_preloading_plan(
                target_tier=StorageTier.HOST,
                capacity_bytes=96 * 1024 * 1024,
                scaling_event={"type": "initial"},
            )

            self.assertGreaterEqual(len(plan.selected_artifacts), 1)
            self.assertLessEqual(plan.total_size_bytes, 96 * 1024 * 1024)

    def test_nvme_hit_schedules_async_host_promotion(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            remote_dir = root / "remote"
            nvme_dir = root / "nvme"
            host_dir = root / "host"
            adapter_id = "medical_lora"
            src = remote_dir / adapter_id
            src.mkdir(parents=True, exist_ok=True)
            Path(src, "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct"}),
                encoding="utf-8",
            )
            Path(src, "adapter_model.bin").write_bytes(b"ok")

            stack = ExperimentStack(
                adapter_info={adapter_id: {"size_mb": 32.0, "hotness": 0.9}},
                hardware_cfg={"gpu_budget_mb": 24000},
                coord_cfg={"min_instances": 1, "max_instances": 2},
                preload_cfg={
                    "strategy": "hybrid",
                    "nvme_capacity_mb": 20480,
                    "host_capacity_mb": 4096,
                    "min_hotness": 0.3,
                    "dynamic_forwarding_enabled": True,
                },
                remote_dir=remote_dir,
                nvme_dir=nvme_dir,
                host_dir=host_dir,
            )
            stack._ensure_registered()
            stack.coordinator.compute_faaslora_host_load_ms = lambda _size_mb: 4.0
            stack.coordinator.compute_faaslora_nvme_load_ms = lambda _size_mb: 8.0
            admitted = asyncio.run(stack.residency_manager.admit_artifact(adapter_id, StorageTier.NVME))
            self.assertTrue(admitted)
            stack.sync_local_tier_paths()
            self.assertIn(adapter_id, stack._nvme_paths)
            self.assertNotIn(adapter_id, stack._host_paths)

            class FakeCoord:
                def _is_resident(self, _aid: str) -> bool:
                    return False

                def record_gpu_ready_hit(self, _adapter_id: str) -> None:
                    return None

                def record_warm_pool_hit(self, _adapter_id: str = "") -> None:
                    return None

                def compute_faaslora_nvme_load_ms(self, _size_mb: float) -> float:
                    return 1.0

                async def request_lora_load(self, *_args, **_kwargs):
                    return 0.0, 0.0

            async def run_case() -> None:
                _, tier, _, _, _ = await stack.resolve_lora(
                    adapter_id,
                    32.0,
                    False,
                    lambda _aid: "",
                    coordinator=FakeCoord(),
                )
                self.assertEqual(tier, "nvme")
                if stack._pending_host_promotions:
                    await asyncio.gather(*stack._pending_host_promotions.values())
                stack.sync_local_tier_paths()

            asyncio.run(run_case())

            self.assertIn(adapter_id, stack._host_paths)
            self.assertTrue((host_dir / adapter_id).exists())

    def test_dynamic_forwarding_selects_best_gpu_candidate(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            remote_dir = root / "remote"
            nvme_dir = root / "nvme"
            host_dir = root / "host"
            adapter_a = "hot_host"
            adapter_b = "warm_nvme"
            for aid in (adapter_a, adapter_b):
                src = remote_dir / aid
                src.mkdir(parents=True, exist_ok=True)
                Path(src, "adapter_config.json").write_text(
                    json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct"}),
                    encoding="utf-8",
                )
                Path(src, "adapter_model.bin").write_bytes(b"ok")

            stack = ExperimentStack(
                adapter_info={
                    adapter_a: {"size_mb": 16.0, "hotness": 0.9},
                    adapter_b: {"size_mb": 32.0, "hotness": 0.6},
                },
                hardware_cfg={"gpu_budget_mb": 24000},
                coord_cfg={"min_instances": 1, "max_instances": 2},
                preload_cfg={
                    "strategy": "hybrid",
                    "nvme_capacity_mb": 20480,
                    "host_capacity_mb": 4096,
                    "min_hotness": 0.3,
                    "dynamic_forwarding_enabled": True,
                },
                remote_dir=remote_dir,
                nvme_dir=nvme_dir,
                host_dir=host_dir,
            )
            stack._ensure_registered()
            asyncio.run(stack.residency_manager.admit_artifact(adapter_a, StorageTier.HOST))
            asyncio.run(stack.residency_manager.admit_artifact(adapter_b, StorageTier.NVME))
            stack.sync_local_tier_paths()

            class FakeCoord:
                def _contention_pressure(self) -> float:
                    return 0.0

                def compute_faaslora_host_load_ms(self, _size_mb: float) -> float:
                    return 4.0

                def compute_faaslora_nvme_load_ms(self, _size_mb: float) -> float:
                    return 8.0

                def evaluate_gpu_admission(self, _adapter_id: str, _size_mb: float, tier: str = "nvme"):
                    return {"admit": True, "tier": tier}

            candidate = stack.select_gpu_forward_candidate(
                gpu_resident_adapters=set(),
                coordinator=FakeCoord(),
            )
            self.assertIsNotNone(candidate)
            self.assertEqual(candidate["adapter_id"], adapter_a)
            self.assertEqual(candidate["source_tier"], "host")

    def test_dynamic_forwarding_selects_positive_gain_replacement(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            remote_dir = root / "remote"
            nvme_dir = root / "nvme"
            host_dir = root / "host"
            adapter_hot = "hot_host"
            adapter_weak = "weak_gpu"
            for aid in (adapter_hot, adapter_weak):
                src = remote_dir / aid
                src.mkdir(parents=True, exist_ok=True)
                Path(src, "adapter_config.json").write_text(
                    json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct"}),
                    encoding="utf-8",
                )
                Path(src, "adapter_model.bin").write_bytes(b"ok")

            stack = ExperimentStack(
                adapter_info={
                    adapter_hot: {"size_mb": 16.0, "hotness": 0.9},
                    adapter_weak: {"size_mb": 32.0, "hotness": 0.2},
                },
                hardware_cfg={"gpu_budget_mb": 24000},
                coord_cfg={"min_instances": 1, "max_instances": 2},
                preload_cfg={
                    "strategy": "hybrid",
                    "nvme_capacity_mb": 20480,
                    "host_capacity_mb": 4096,
                    "min_hotness": 0.3,
                    "dynamic_forwarding_enabled": True,
                },
                remote_dir=remote_dir,
                nvme_dir=nvme_dir,
                host_dir=host_dir,
            )
            stack._ensure_registered()
            asyncio.run(stack.residency_manager.admit_artifact(adapter_hot, StorageTier.HOST))
            asyncio.run(stack.residency_manager.admit_artifact(adapter_weak, StorageTier.NVME))
            stack.sync_local_tier_paths()

            class FakeCoord:
                def _contention_pressure(self) -> float:
                    return 0.0

                def compute_faaslora_host_load_ms(self, _size_mb: float) -> float:
                    return 4.0

                def compute_faaslora_nvme_load_ms(self, _size_mb: float) -> float:
                    return 8.0

                def evaluate_gpu_admission(self, _adapter_id: str, _size_mb: float, tier: str = "nvme"):
                    return {
                        "admit": False,
                        "should_attempt": True,
                        "effective_capacity_mb": 0.0,
                        "tier": tier,
                    }

            candidate = stack.select_gpu_forward_candidate(
                gpu_resident_adapters={adapter_weak},
                coordinator=FakeCoord(),
            )
            self.assertIsNotNone(candidate)
            self.assertEqual(candidate["adapter_id"], adapter_hot)
            self.assertIsNotNone(candidate["replace"])
            self.assertEqual(candidate["replace"]["adapter_id"], adapter_weak)
            self.assertEqual(candidate["replace"]["target_tier"], "nvme")

    def test_runtime_gpu_forward_executes_value_based_replacement(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.model_cfg = {"runtime_concurrency_cap": 2}
        runner.wl_cfg = {"concurrency": 2}
        runner._runtime_gpu_forward_tasks = {}
        runner._runtime_forward_task_key = lambda _slot: ("slot",)
        runner._refresh_slot_runtime_hints = lambda _slot: None
        runner._schedule_runtime_gpu_forward = lambda _slot: False
        marks = []
        runner._mark_slot_adapter_tier = lambda _slot, aid, tier: marks.append((aid, tier))

        updates = []

        class FakeRegistry:
            def update_artifact(self, adapter_id: str, payload):
                updates.append((adapter_id, dict(payload)))

        class FakeResidency:
            def __init__(self):
                self.evictions = []
                self.admits = []

            async def evict_artifact(self, adapter_id: str, tier):
                self.evictions.append((adapter_id, tier))
                return True

            async def admit_artifact(self, adapter_id: str, tier, force=False):
                self.admits.append((adapter_id, tier, force))
                return True

        residency = FakeResidency()
        runner._stack = SimpleNamespace(
            registry=FakeRegistry(),
            residency_manager=residency,
            _host_paths={"old": "/tmp/old", "new": "/tmp/new"},
            _nvme_paths={},
        )

        class FakeEngine:
            def __init__(self):
                self.unloaded = []
                self.loaded = []

            async def load_lora_to_gpu_and_measure(self, lora_path: str, adapter_id: str):
                self.loaded.append((lora_path, adapter_id))
                return 5.0, True

            async def unload_lora_adapter(self, adapter_id: str):
                self.unloaded.append(adapter_id)
                return True

        class FakeCoord:
            def __init__(self):
                self._residency_manager = object()
                self._loading_semaphore = asyncio.Semaphore(2)

            def _loads_in_flight_ratio(self) -> float:
                return 0.0

            def evaluate_gpu_admission(self, _adapter_id: str, _size_mb: float, tier: str = "nvme"):
                return {
                    "admit": False,
                    "should_attempt": True,
                    "effective_capacity_mb": 0.0,
                    "tier": tier,
                }

        slot = SimpleNamespace(
            engine=FakeEngine(),
            coordinator=FakeCoord(),
            active_requests=1,
            load_queue_depth=0,
        )
        candidate = {
            "adapter_id": "new",
            "path": "/tmp/new",
            "source_tier": "host",
            "size_mb": 16.0,
            "replace": {
                "adapter_id": "old",
                "target_tier": "host",
                "size_mb": 8.0,
                "utility": 0.1,
            },
        }

        asyncio.run(runner._run_runtime_gpu_forward(slot, candidate))

        self.assertEqual(slot.engine.unloaded, ["old"])
        self.assertEqual(slot.engine.loaded, [("/tmp/new", "new")])
        self.assertIn(("old", StorageTier.HOST), residency.evictions)
        self.assertIn(("new", StorageTier.GPU, True), residency.admits)
        self.assertIn(("old", "host"), marks)
        self.assertIn(("new", "gpu"), marks)
        self.assertTrue(any(aid == "new" for aid, _ in updates))

    def test_online_dynamic_forwarding_allows_controlled_background_activity(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.model_cfg = {"runtime_concurrency_cap": 2}
        runner.wl_cfg = {"concurrency": 2}

        slot = SimpleNamespace(active_requests=1, load_queue_depth=0)

        class CoordIdle:
            def _loads_in_flight_ratio(self) -> float:
                return 0.0

        class CoordWarm:
            def _loads_in_flight_ratio(self) -> float:
                return 0.5

        class CoordSaturated:
            def _loads_in_flight_ratio(self) -> float:
                return 1.0

        self.assertTrue(runner._runtime_forward_has_capacity(slot, CoordIdle()))
        slot.load_queue_depth = 1
        self.assertTrue(runner._runtime_forward_has_capacity(slot, CoordWarm()))
        slot.load_queue_depth = 2
        self.assertFalse(runner._runtime_forward_has_capacity(slot, CoordWarm()))
        slot.load_queue_depth = 0
        slot.active_requests = 2
        self.assertFalse(runner._runtime_forward_has_capacity(slot, CoordIdle()))
        slot.active_requests = 1
        self.assertFalse(runner._runtime_forward_has_capacity(slot, CoordSaturated()))

    def test_runtime_forward_capacity_limit_falls_back_to_engine_model_cfg(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.engine = SimpleNamespace(model_cfg={"runtime_concurrency_cap": 3})
        runner.wl_cfg = {"concurrency": 1}

        self.assertEqual(runner._runtime_forward_capacity_limit(), 3)

    def test_formal_a500_profiles_exist_for_four_models(self) -> None:
        workloads = self.experiments["workload_profiles"]

        self.assertEqual(workloads["qwen_7b_auto500_main"]["lora_adapters"]["selected_num_adapters"], 500)
        self.assertEqual(workloads["qwen_7b_tp2_compare_a500_main"]["lora_adapters"]["selected_num_adapters"], 500)
        self.assertEqual(workloads["qwen_14b_tp2_a500_main"]["lora_adapters"]["selected_num_adapters"], 500)
        self.assertEqual(workloads["mistral_7b_auto500_main"]["lora_adapters"]["selected_num_adapters"], 500)
        self.assertEqual(workloads["mistral_nemo_12b_tp2_main"]["lora_adapters"]["selected_num_adapters"], 500)

    def test_realistic_v2_pool_planner_creates_mixed_specs(self) -> None:
        manifest = {
            "adapters": [{"id": f"a{i:03d}", "size_mb": 32} for i in range(20)]
        }
        profiles = self.experiments["lora_adapters"]["artifact_pool_profiles"]
        specs = _build_adapter_specs(
            manifest=manifest,
            artifact_pool_profile="realistic_v2",
            artifact_pool_profiles=profiles,
            artifact_pool_seed=42,
            ranks_fallback=[8],
        )
        self.assertEqual(len(specs), 20)
        self.assertGreater(len({spec["rank"] for spec in specs}), 1)
        self.assertGreater(
            len({tuple(spec["target_modules"]) for spec in specs}),
            1,
        )

    def test_publicmix_validator_accepts_only_runtime_compatible_public_adapters(self) -> None:
        with TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir, "public")
            source_dir.mkdir(parents=True, exist_ok=True)

            valid = source_dir / "valid_public_lora"
            valid.mkdir()
            Path(valid, "adapter_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
                        "peft_type": "LORA",
                        "r": 8,
                        "target_modules": ["q_proj", "v_proj"],
                        "modules_to_save": None,
                    }
                ),
                encoding="utf-8",
            )
            Path(valid, "adapter_model.bin").write_bytes(b"ok")

            invalid = source_dir / "bad_public_lora"
            invalid.mkdir()
            Path(invalid, "adapter_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": "Qwen/Qwen2.5-14B-Instruct",
                        "peft_type": "LORA",
                        "r": 8,
                        "target_modules": ["lm_head"],
                        "modules_to_save": ["lm_head"],
                    }
                ),
                encoding="utf-8",
            )
            Path(invalid, "adapter_model.bin").write_bytes(b"bad")

            report = scan_public_adapter_pool(source_dir, "Qwen/Qwen2.5-7B-Instruct")
            self.assertEqual(report["accepted_count"], 1)
            self.assertEqual(report["rejected_count"], 1)
            self.assertEqual(report["accepted"][0]["source_id"], "valid_public_lora")
            self.assertIn(
                "base_model_name_or_path does not match expected model",
                report["rejected"][0]["reasons"],
            )

    def test_publicmix_validator_accepts_fused_target_modules_when_runtime_supports_them(self) -> None:
        with TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir, "public")
            source_dir.mkdir(parents=True, exist_ok=True)

            fused = source_dir / "fused_public_lora"
            fused.mkdir()
            Path(fused, "adapter_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
                        "peft_type": "LORA",
                        "r": 16,
                        "target_modules": ["qkv_proj", "gate_up_proj", "down_proj", "o_proj"],
                        "modules_to_save": None,
                    }
                ),
                encoding="utf-8",
            )
            Path(fused, "adapter_model.bin").write_bytes(b"ok")

            report = scan_public_adapter_pool(source_dir, "Qwen/Qwen2.5-7B-Instruct")
            self.assertEqual(report["accepted_count"], 1)
            self.assertEqual(report["rejected_count"], 0)
            self.assertEqual(report["accepted"][0]["source_id"], "fused_public_lora")

    def test_publicmix_validator_rejects_extreme_rank_and_size(self) -> None:
        with TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir, "public")
            source_dir.mkdir(parents=True, exist_ok=True)

            huge = source_dir / "huge_public_lora"
            huge.mkdir()
            Path(huge, "adapter_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.3",
                        "peft_type": "LORA",
                        "r": 512,
                        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                        "modules_to_save": None,
                    }
                ),
                encoding="utf-8",
            )
            Path(huge, "adapter_model.bin").write_bytes(b"x" * 1024)

            report = scan_public_adapter_pool(source_dir, "mistralai/Mistral-7B-Instruct-v0.3")
            self.assertEqual(report["accepted_count"], 0)
            self.assertEqual(report["rejected_count"], 1)
            reasons = report["rejected"][0]["reasons"]
            self.assertTrue(any("rank exceeds publicmix limit" in reason for reason in reasons))

    def test_publicmix_validator_rejects_dora_public_adapters(self) -> None:
        with TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir, "public")
            source_dir.mkdir(parents=True, exist_ok=True)

            dora = source_dir / "dora_public_lora"
            dora.mkdir()
            Path(dora, "adapter_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.3",
                        "peft_type": "LORA",
                        "r": 32,
                        "target_modules": ["qkv_proj", "gate_up_proj", "down_proj", "o_proj"],
                        "modules_to_save": None,
                        "use_dora": True,
                    }
                ),
                encoding="utf-8",
            )
            Path(dora, "adapter_model.bin").write_bytes(b"ok")

            report = scan_public_adapter_pool(source_dir, "mistralai/Mistral-7B-Instruct-v0.3")
            self.assertEqual(report["accepted_count"], 0)
            self.assertEqual(report["rejected_count"], 1)
            reasons = report["rejected"][0]["reasons"]
            self.assertTrue(any("use_dora=true is unsupported" in reason for reason in reasons))

    def test_publicmix_manifest_preserves_canonical_order_and_records_fill(self) -> None:
        report = {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "accepted_count": 2,
            "rejected_count": 1,
            "accepted": [
                {
                    "source_id": "pub_a",
                    "local_path": "/tmp/pub_a",
                    "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
                    "rank": 8,
                    "size_mb": 12.5,
                    "target_modules": ["q_proj", "v_proj"],
                    "dtype": "float16",
                },
                {
                    "source_id": "pub_b",
                    "local_path": "/tmp/pub_b",
                    "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
                    "rank": 16,
                    "size_mb": 18.0,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "dtype": "float16",
                },
            ],
            "rejected": [{"source_id": "pub_bad"}],
        }

        manifest = build_publicmix_manifest(
            validated_report=report,
            target_count=4,
            topup_profile="realistic_v2",
            topup_seed=42,
        )
        self.assertEqual(manifest["public_count"], 2)
        self.assertEqual(manifest["generated_fill_count"], 2)
        self.assertEqual(manifest["adapters"][0]["source_type"], "public")
        self.assertEqual(manifest["adapters"][0]["public_adapter_id"], "pub_a")
        self.assertEqual(manifest["adapters"][1]["source_type"], "public")
        self.assertEqual(manifest["adapters"][2]["source_type"], "generated_fill")
        self.assertEqual(manifest["adapters"][2]["generation_profile"], "realistic_v2")

    def test_publicmix_build_materializes_public_dirs_and_invokes_generator(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            public_dir = root / "public_a"
            public_dir.mkdir()
            Path(public_dir, "adapter_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
                        "peft_type": "LORA",
                        "r": 8,
                        "target_modules": ["q_proj", "v_proj"],
                        "modules_to_save": None,
                    }
                ),
                encoding="utf-8",
            )
            Path(public_dir, "adapter_model.bin").write_bytes(b"ok")

            manifest = {
                "model_name": "Qwen/Qwen2.5-7B-Instruct",
                "num_adapters": 2,
                "topup_profile": "realistic_v2",
                "topup_seed": 42,
                "adapters": [
                    {
                        "id": "finance_lora",
                        "source_type": "public",
                        "local_path": str(public_dir),
                    },
                    {
                        "id": "medical_lora",
                        "source_type": "generated_fill",
                    },
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output_dir = root / "frozen_pool"
            model_override = str(root / "models" / "Qwen--Qwen2.5-7B-Instruct")
            Path(model_override).mkdir(parents=True, exist_ok=True)

            with patch("scripts.prepare_publicmix_pool.subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(args=["python"], returncode=0)
                summary = _build_publicmix_pool(
                    manifest_path=manifest_path,
                    output_dir=output_dir,
                    generation_mode="synthetic",
                    python_bin=sys.executable,
                    model_override=model_override,
                    link_mode="copy",
                    force_public=False,
                )

            self.assertEqual(summary["public_created"], 1)
            self.assertTrue((output_dir / "finance_lora").exists())
            self.assertFalse((output_dir / "finance_lora").is_symlink())
            normalized_cfg = json.loads((output_dir / "finance_lora" / "adapter_config.json").read_text(encoding="utf-8"))
            self.assertEqual(normalized_cfg["base_model_name_or_path"], model_override)
            called_cmd = mock_run.call_args.kwargs["args"] if "args" in mock_run.call_args.kwargs else mock_run.call_args.args[0]
            self.assertIn("generate_lora_adapters.py", " ".join(called_cmd))
            self.assertIn("--artifact-pool-profile", called_cmd)
            self.assertIn("realistic_v2", called_cmd)
            self.assertIn("--synthetic", called_cmd)
            self.assertIn(model_override, called_cmd)

    def test_publicmix_build_revalidates_and_rejects_stale_dora_manifest_entries(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            public_dir = root / "public_dora"
            public_dir.mkdir()
            Path(public_dir, "adapter_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.3",
                        "peft_type": "LORA",
                        "r": 32,
                        "target_modules": ["qkv_proj", "gate_up_proj", "down_proj", "o_proj"],
                        "modules_to_save": None,
                        "use_dora": True,
                    }
                ),
                encoding="utf-8",
            )
            Path(public_dir, "adapter_model.bin").write_bytes(b"ok")

            manifest = {
                "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
                "num_adapters": 1,
                "topup_profile": "realistic_v2",
                "topup_seed": 42,
                "adapters": [
                    {
                        "id": "finance_lora",
                        "source_type": "public",
                        "local_path": str(public_dir),
                    }
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output_dir = root / "frozen_pool"

            with self.assertRaisesRegex(ValueError, "runtime-compatible"):
                _build_publicmix_pool(
                    manifest_path=manifest_path,
                    output_dir=output_dir,
                    generation_mode="synthetic",
                    python_bin=sys.executable,
                    model_override="",
                    link_mode="copy",
                    force_public=False,
                )

    def test_adapter_match_accepts_publicmix_lora_with_mlp_shapes(self) -> None:
        try:
            import torch
            from safetensors.torch import save_file
        except Exception as exc:  # pragma: no cover - environment-specific fallback
            self.skipTest(f"safetensors/torch unavailable: {exc}")

        with TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir, "finance_lora")
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model_name = "/home/qhq/serverless_llm_experiment/models/mistralai--Mistral-Nemo-Instruct-2407"
            Path(adapter_dir, "adapter_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": model_name,
                        "peft_type": "LORA",
                        "r": 16,
                        "target_modules": [
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            tensors = {
                "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": torch.zeros((16, 14336), dtype=torch.float16),
                "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": torch.zeros((5120, 16), dtype=torch.float16),
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros((16, 5120), dtype=torch.float16),
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros((4096, 16), dtype=torch.float16),
            }
            save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))

            self.assertTrue(
                _adapter_matches_model(
                    adapter_dir,
                    model_name,
                    {"hidden_size": 5120},
                )
            )

    def test_model_ref_matching_accepts_local_and_repo_style_names(self) -> None:
        self.assertTrue(
            model_refs_match(
                "/home/qhq/serverless_llm_experiment/models/Qwen--Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
            )
        )
        self.assertFalse(
            model_refs_match(
                "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-14B-Instruct",
            )
        )
        self.assertTrue(
            model_refs_match(
                "mistralai/Mistral-Nemo-Instruct-2407",
                "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
            )
        )

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
                adapter_name = (selected_adapters or ["default"])[0]
                nested = dest / adapter_name
                nested.mkdir(parents=True, exist_ok=True)
                Path(nested, "adapter_config.json").write_text("{}", encoding="utf-8")
                Path(nested, "adapter_model.safetensors").write_text("x", encoding="utf-8")
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
            stale_dir = Path(tmpdir, "a1")
            stale_dir.mkdir(parents=True, exist_ok=True)
            Path(stale_dir, "stale.txt").write_text("old", encoding="utf-8")
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
            self.assertFalse(Path(tmpdir, "a1", "stale.txt").exists())
            self.assertTrue(Path(tmpdir, "a1", "adapter_config.json").exists())
            self.assertFalse(Path(tmpdir, "a1", "a1").exists())

        load_once.assert_called_once()
        self.assertEqual(set(timings.keys()), {"a1", "a2", "a3"})
        self.assertEqual(fake_model.saved, [("a1",), ("a2",), ("a3",)])

    def test_saved_adapter_weights_are_normalized_to_fp16(self) -> None:
        try:
            import torch
            from safetensors.torch import load_file, save_file
        except Exception as exc:  # pragma: no cover - environment-specific fallback
            self.skipTest(f"torch/safetensors unavailable: {exc}")

        with TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir)
            save_file(
                {"x": torch.ones(2, 2, dtype=torch.float32)},
                str(dest / "adapter_model.safetensors"),
            )
            _normalize_saved_adapter_weights(dest, target_dtype="float16")
            state = load_file(str(dest / "adapter_model.safetensors"))
            self.assertEqual({str(v.dtype) for v in state.values()}, {"torch.float16"})

    def test_support_files_repair_broken_symlink_after_pool_move(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_root = root / "models" / "mistralai--Mistral-7B-Instruct-v0.3"
            model_root.mkdir(parents=True, exist_ok=True)
            (model_root / "config.json").write_text('{"ok": true}', encoding="utf-8")

            adapter_dir = root / "artifacts" / "frozen" / "mistral_7b_a500_v2_publicmix" / "medical_lora"
            adapter_dir.mkdir(parents=True, exist_ok=True)

            broken_target = Path("/definitely/missing/config.json")
            (adapter_dir / "config.json").symlink_to(broken_target)

            created = ensure_adapter_support_files(adapter_dir, str(model_root))

            self.assertIn("config.json", created)
            repaired = adapter_dir / "config.json"
            self.assertTrue(repaired.exists())
            self.assertTrue(repaired.is_symlink())
            self.assertEqual(repaired.resolve(), (model_root / "config.json").resolve())

    def test_support_files_include_mistral_tokenizer_assets(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_root = root / "models" / "mistralai--Mistral-7B-Instruct-v0.3"
            model_root.mkdir(parents=True, exist_ok=True)
            (model_root / "config.json").write_text("{}", encoding="utf-8")
            (model_root / "tokenizer.model.v3").write_text("tok", encoding="utf-8")
            (model_root / "chat_template.jinja").write_text("tmpl", encoding="utf-8")

            adapter_dir = root / "artifacts" / "frozen" / "mistral_7b_a500_v2_publicmix" / "finance_lora"
            adapter_dir.mkdir(parents=True, exist_ok=True)

            created = ensure_adapter_support_files(adapter_dir, str(model_root))

            self.assertIn("tokenizer.model.v3", created)
            self.assertIn("chat_template.jinja", created)
            self.assertTrue((adapter_dir / "tokenizer.model.v3").exists())
            self.assertTrue((adapter_dir / "chat_template.jinja").exists())

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

    def test_gpu_ready_hits_are_distinct_from_warm_pool_hits(self) -> None:
        coord = ResourceCoordinator(config={"gpu_budget_mb": 1000, "model_weights_mb": 100})
        coord._resident_loras = {"warm": 10.0, "cold": 12.0}
        coord._warm_pool = {"warm"}

        coord.record_gpu_ready_hit("warm")
        coord.record_gpu_ready_hit("cold")

        metrics = coord.get_summary_metrics()
        self.assertEqual(metrics["gpu_ready_hits"], 2)
        self.assertEqual(metrics["warm_pool_hits"], 1)

    def test_dynamic_working_set_pressure_reduces_effective_capacity(self) -> None:
        coord = ResourceCoordinator(config={
            "gpu_budget_mb": 1000,
            "model_weights_mb": 100,
            "lora_load_reserve_ratio": 0.0,
            "effective_capacity_admission_enabled": True,
            "idle_timeout_s": 10.0,
        })
        base_capacity = coord._effective_capacity_mb(pressure=0.0)

        now = time.time()
        coord._active_tokens = 4000
        coord._active_batches = 2
        coord._recent_batch_tokens_ewma = 3000.0
        coord._adapter_sizes_mb["hot_a"] = 120.0
        coord._adapter_sizes_mb["hot_b"] = 140.0
        coord._access_log["hot_a"].append(now)
        coord._access_log["hot_b"].append(now)

        dynamic_capacity = coord._effective_capacity_mb()
        decision = coord.evaluate_gpu_admission("candidate", 64.0, tier="nvme")

        self.assertLess(dynamic_capacity, base_capacity)
        self.assertGreater(decision["future_reserve_mb"], 0.0)
        self.assertGreaterEqual(decision["working_set_pressure"], 0.0)

    def test_working_set_gap_is_reserved_once_for_gpu_admission(self) -> None:
        coord = ResourceCoordinator(config={
            "gpu_budget_mb": 1000,
            "model_weights_mb": 100,
            "lora_load_reserve_ratio": 0.0,
            "effective_capacity_admission_enabled": True,
            "idle_timeout_s": 10.0,
        })
        now = time.time()
        for idx in range(3):
            aid = f"hot_{idx}"
            coord._adapter_sizes_mb[aid] = 220.0
            coord._access_log[aid].append(now)
        coord._adapter_sizes_mb["candidate"] = 64.0
        coord._access_log["candidate"].append(now)

        decision = coord.evaluate_gpu_admission("candidate", 64.0, tier="nvme")

        self.assertGreater(decision["working_set_pressure"], 0.75)
        self.assertEqual(decision["pressure"], 0.0)
        self.assertGreater(decision["effective_capacity_mb"], 150.0)
        self.assertTrue(decision["should_attempt"])
        self.assertTrue(decision["admit"])

    def test_latency_model_constants_can_be_explicitly_configured(self) -> None:
        coord = ResourceCoordinator(config={
            "gpu_load_overhead_ms": 80.0,
            "pcie_bw_mbps": 20000.0,
            "nvme_bw_mbps": 4000.0,
            "serverlessllm_overhead_ratio": 0.25,
            "host_locality_factor": 1.2,
            "cpu_locality_factor": 0.95,
            "nvme_locality_factor": 0.7,
            "remote_locality_factor": 0.4,
        })

        self.assertAlmostEqual(coord.compute_slora_load_ms(100.0), 100.0 / 20.0 + 80.0)
        self.assertAlmostEqual(
            coord.compute_serverlessllm_load_ms(100.0),
            100.0 / 4.0 + 100.0 / 20.0 + 80.0 * 0.25,
        )
        self.assertAlmostEqual(coord._locality_factor("host"), 1.2)
        self.assertAlmostEqual(coord._locality_factor("cpu"), 0.95)
        self.assertAlmostEqual(coord._locality_factor("nvme"), 0.7)
        self.assertAlmostEqual(coord._locality_factor("remote"), 0.4)
        self.assertAlmostEqual(coord._locality_factor("unknown"), 0.4)

    def test_trigger_scale_down_prefers_high_reload_value_adapter(self) -> None:
        coord = ResourceCoordinator(config={"gpu_budget_mb": 1000, "model_weights_mb": 100})
        now = time.time()
        coord._resident_loras = {"hot": 16.0, "cold": 32.0}
        coord._adapter_sizes_mb.update({"hot": 16.0, "cold": 32.0})
        coord._adapter_last_source_tier.update({"hot": "nvme", "cold": "host"})
        coord._access_log["hot"].extend([now - 1, now - 2, now - 3])
        coord._access_log["cold"].append(now - 180)

        warm = asyncio.run(coord.trigger_scale_down(warm_pool_size=1))

        self.assertEqual(warm, {"hot"})
        self.assertIn("hot", coord._resident_loras)
        self.assertNotIn("cold", coord._resident_loras)


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



class RuntimeAccountingAndMetricsSmokeTests(unittest.TestCase):
    def test_scale_up_preload_capacity_prefers_explicit_config(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.preload_cfg = {
            "scale_up_preload_mb": 1536,
            "host_capacity_mb": 4096,
        }

        capacity_bytes = runner._scale_up_preload_capacity_bytes()

        self.assertEqual(capacity_bytes, 1536 * 1024 * 1024)

    def test_residency_manager_sync_gpu_capacity_tracks_active_devices(self) -> None:
        class DummyConfig:
            def __init__(self) -> None:
                self._data = {
                    "memory": {
                        "gpu": {"total_memory_gb": 24},
                        "host": {"total_memory_gb": 64},
                        "nvme": {"cache_size_gb": 100},
                    },
                    "storage": {},
                }

            def get(self, key, default=None):
                return self._data.get(key, default)

        infos = {
            0: GPUMemoryInfo(device_id=0, timestamp=0.0, total_bytes=10, used_bytes=4, free_bytes=6, active_bytes=1, cached_bytes=2),
            1: GPUMemoryInfo(device_id=1, timestamp=0.0, total_bytes=12, used_bytes=5, free_bytes=7, active_bytes=2, cached_bytes=3),
        }
        monitor = SimpleNamespace(
            enabled=True,
            devices=[0, 1],
            get_all_devices_memory_info=lambda: infos,
            get_current_memory_info=lambda device_id=0: infos.get(device_id),
        )
        manager = ResidencyManager(DummyConfig(), SimpleNamespace(), monitor)
        manager.set_tracked_gpu_device_ids([0, 1])

        manager._sync_gpu_capacity_once()

        gpu_capacity = manager.tier_capacities[StorageTier.GPU]
        self.assertEqual(gpu_capacity.total_bytes, 22)
        self.assertEqual(gpu_capacity.used_bytes, 9)

    def test_scenario_result_aggregate_computes_ttft_decomposition(self) -> None:
        result = ScenarioResult("demo", "faaslora_full", total=4)
        result.ttft_slo_ms = 180.0
        result.requests = [
            RequestResult("r1", "a", False, "normal", True, "gpu", 10.0, 80.0, 100.0, 5.0, 5.0, 20.0, 120.0, 10, 5, 0.0, True, "inst_1", False),
            RequestResult("r2", "b", False, "normal", True, "host", 60.0, 120.0, 200.0, 10.0, 10.0, 25.0, 240.0, 10, 5, 0.0, True, "inst_1", False),
            RequestResult("r3", "c", False, "normal", False, "remote", 0.0, 140.0, 150.0, 5.0, 5.0, 30.0, 190.0, 10, 5, 0.0, True, "inst_2", True),
            RequestResult("r4", "d", False, "normal", True, "gpu", 20.0, 160.0, 220.0, 10.0, 10.0, 35.0, 260.0, 10, 5, 0.0, True, "inst_2", False),
        ]
        result.scale_up_events = [{"request_index": 2, "instance_id": "inst_2"}]

        result.aggregate(4.0)

        self.assertAlmostEqual(result.avg_comparable_ttft_ms, (100.0 + 200.0 + 220.0) / 3.0)
        self.assertAlmostEqual(result.avg_serverless_overhead_ms, (20.0 + 80.0 + 10.0 + 40.0) / 4.0)
        self.assertAlmostEqual(result.avg_gpu_ready_ttft_ms, (100.0 + 220.0) / 2.0)
        self.assertAlmostEqual(result.avg_scaleup_affected_ttft_ms, 150.0)
        self.assertAlmostEqual(result.throughput_tok_per_s, 5.0)
        self.assertAlmostEqual(result.slo_attainment, 0.5)
        self.assertEqual(result.cold_starts_after_scale_up, [1])

    def test_live_result_stats_includes_ttft_decomposition(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._live_scale_up_events = [{"request_index": 2}]
        runner._ttft_slo_ms = 180.0
        results = [
            RequestResult("r1", "a", False, "normal", True, "gpu", 10.0, 80.0, 100.0, 5.0, 5.0, 20.0, 120.0, 10, 5, 0.0, True, "inst_1", False),
            RequestResult("r2", "b", False, "normal", True, "host", 60.0, 120.0, 200.0, 10.0, 10.0, 25.0, 240.0, 10, 5, 0.0, True, "inst_1", False),
            RequestResult("r3", "c", False, "normal", False, "remote", 0.0, 140.0, 150.0, 5.0, 5.0, 30.0, 190.0, 10, 5, 0.0, True, "inst_2", True),
            RequestResult("r4", "d", False, "normal", True, "gpu", 20.0, 160.0, 220.0, 10.0, 10.0, 35.0, 260.0, 10, 5, 0.0, True, "inst_2", False),
        ]

        stats = runner._live_result_stats(results)

        self.assertAlmostEqual(stats["avg_comparable_ttft_ms"], (100.0 + 200.0 + 220.0) / 3.0)
        self.assertAlmostEqual(stats["avg_serverless_overhead_ms"], (20.0 + 80.0 + 10.0 + 40.0) / 4.0)
        self.assertAlmostEqual(stats["avg_gpu_ready_ttft_ms"], (100.0 + 220.0) / 2.0)
        self.assertAlmostEqual(stats["avg_scaleup_affected_ttft_ms"], 150.0)
        self.assertEqual(stats["total_output_tokens"], 20)
        self.assertAlmostEqual(stats["slo_attainment"], 0.5)

    def test_live_result_stats_handles_empty_comparable_subset(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._live_scale_up_events = [{"request_index": 0}]
        runner._ttft_slo_ms = 5000.0
        results = [
            RequestResult("r1", "a", False, "normal", False, "remote", 30.0, 120.0, 180.0, 5.0, 5.0, 20.0, 220.0, 10, 5, 0.0, True, "inst_2", True),
            RequestResult("r2", "b", False, "normal", True, "host", 40.0, 140.0, 210.0, 10.0, 10.0, 25.0, 250.0, 10, 5, 0.0, True, "inst_2", True),
        ]

        stats = runner._live_result_stats(results)

        self.assertEqual(stats["avg_comparable_ttft_ms"], 0.0)
        self.assertEqual(stats["p95_comparable_ttft_ms"], 0.0)
        self.assertEqual(stats["p99_comparable_ttft_ms"], 0.0)

    def test_comparable_request_requires_local_tier_and_non_scaleup(self) -> None:
        local_hit = RequestResult("r1", "a", False, "normal", True, "host", 40.0, 140.0, 210.0, 10.0, 10.0, 25.0, 250.0, 10, 5, 0.0, True, "inst_1", False)
        scaleup_local = RequestResult("r2", "b", False, "normal", True, "host", 30.0, 120.0, 180.0, 5.0, 5.0, 20.0, 220.0, 10, 5, 0.0, True, "inst_2", True)
        remote_fetch = RequestResult("r3", "c", False, "normal", False, "remote", 30.0, 120.0, 180.0, 5.0, 5.0, 20.0, 220.0, 10, 5, 0.0, True, "inst_1", False)
        legacy_scaleup = SimpleNamespace(success=True, cache_tier="host")

        self.assertTrue(_is_comparable_request(local_hit))
        self.assertFalse(_is_comparable_request(scaleup_local))
        self.assertFalse(_is_comparable_request(remote_fetch))
        self.assertFalse(_is_comparable_request(legacy_scaleup, 3, {3}))

    def test_aggregate_counts_cold_starts_by_scaled_instance_id(self) -> None:
        result = ScenarioResult("demo", "faaslora_full", total=5)
        result.requests = [
            RequestResult("r1", "a", False, "normal", True, "gpu", 0.0, 80.0, 80.0, 0.0, 0.0, 10.0, 110.0, 10, 5, 0.0, True, "inst_1", False),
            RequestResult("r2", "b", False, "normal", True, "host", 40.0, 120.0, 160.0, 0.0, 0.0, 12.0, 190.0, 10, 5, 0.0, True, "inst_2", True),
            RequestResult("r3", "c", False, "normal", True, "gpu", 0.0, 90.0, 90.0, 0.0, 0.0, 12.0, 120.0, 10, 5, 0.0, True, "inst_2", False),
            RequestResult("r4", "d", False, "normal", True, "nvme", 50.0, 140.0, 190.0, 0.0, 0.0, 15.0, 220.0, 10, 5, 0.0, True, "inst_3", True),
            RequestResult("r5", "e", False, "normal", True, "host", 35.0, 130.0, 165.0, 0.0, 0.0, 14.0, 200.0, 10, 5, 0.0, True, "inst_3", True),
        ]
        result.scale_up_events = [
            {"request_index": 1, "instance_id": "inst_2"},
            {"request_index": 3, "instance_id": "inst_3"},
        ]

        result.aggregate(5.0)

        self.assertEqual(result.cold_starts_after_scale_up, [1, 2])

    def test_sync_stack_gpu_accounting_tracks_only_stack_runtime_devices(self) -> None:
        tracked = []
        sync_calls = []
        runner = ScenarioRunner.__new__(ScenarioRunner)
        shared_coord = SimpleNamespace(name="shared")
        runner._stack = SimpleNamespace(
            residency_manager=SimpleNamespace(
                set_tracked_gpu_device_ids=lambda ids: tracked.append(list(ids)),
                _sync_gpu_capacity_once=lambda: sync_calls.append("sync"),
            )
        )
        runner.coordinator = shared_coord
        runner.instance_pool = SimpleNamespace(
            get_slots=lambda: [
                SimpleNamespace(device_id=0, coordinator=shared_coord),
                SimpleNamespace(device_id=1, coordinator=SimpleNamespace(name="dedicated")),
                SimpleNamespace(device_id=1, coordinator=SimpleNamespace(name="dedicated-2")),
            ]
        )
        runner.engine = SimpleNamespace(device_id=0, model_cfg={"tensor_parallel_size": 1})
        runner._available_device_ids = lambda: [0, 1]

        runner._sync_stack_gpu_accounting()

        self.assertEqual(tracked, [[0]])
        self.assertEqual(sync_calls, ["sync"])

    def test_sync_stack_gpu_accounting_expands_tp_devices_for_primary_runtime(self) -> None:
        tracked = []
        sync_calls = []
        runner = ScenarioRunner.__new__(ScenarioRunner)
        shared_coord = SimpleNamespace(name="shared")
        runner._stack = SimpleNamespace(
            residency_manager=SimpleNamespace(
                set_tracked_gpu_device_ids=lambda ids: tracked.append(list(ids)),
                _sync_gpu_capacity_once=lambda: sync_calls.append("sync"),
            )
        )
        runner.coordinator = shared_coord
        runner.instance_pool = SimpleNamespace(
            get_slots=lambda: [SimpleNamespace(device_id=0, coordinator=shared_coord)]
        )
        runner.engine = SimpleNamespace(
            device_id=0,
            model_cfg={"tensor_parallel_size": 2, "visible_device_ids": [0, 1]},
        )
        runner._available_device_ids = lambda: [0, 1]

        runner._sync_stack_gpu_accounting()

        self.assertEqual(tracked, [[0, 1]])
        self.assertEqual(sync_calls, ["sync"])

    def test_autoscaler_collect_metrics_uses_max_gpu_util_across_devices(self) -> None:
        class DummyConfig:
            def get(self, key, default=None):
                if key == "coordination":
                    return {"autoscaling": {"enabled": True}}
                return default

        infos = {
            0: GPUMemoryInfo(device_id=0, timestamp=0.0, total_bytes=100, used_bytes=40, free_bytes=60, active_bytes=0, cached_bytes=0),
            1: GPUMemoryInfo(device_id=1, timestamp=0.0, total_bytes=100, used_bytes=90, free_bytes=10, active_bytes=0, cached_bytes=0),
        }
        autoscaler = AutoScaler(
            config=DummyConfig(),
            registry=SimpleNamespace(get_all_artifacts=lambda: []),
            gpu_monitor=SimpleNamespace(
                enabled=True,
                get_all_devices_memory_info=lambda: infos,
                get_current_memory_info=lambda device_id=0: infos.get(device_id),
            ),
        )
        autoscaler.get_current_instances = lambda: [
            SimpleNamespace(load_score=0.0, gpu_memory_used=0.0, active_requests=0),
            SimpleNamespace(load_score=0.0, gpu_memory_used=0.0, active_requests=1),
        ]

        metrics = asyncio.run(autoscaler._collect_current_metrics())

        self.assertAlmostEqual(metrics.gpu_utilization, 90.0)


if __name__ == "__main__":
    unittest.main()
