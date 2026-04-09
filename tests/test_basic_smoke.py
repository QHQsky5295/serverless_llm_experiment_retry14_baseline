"""Basic smoke tests that run in the stable mainline environment."""

from __future__ import annotations

import os
import asyncio
import json
import subprocess
import sys
import time
import unittest
from collections import defaultdict
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import yaml

from faaslora.api.http_server import HTTPServer
from faaslora.memory.gpu_monitor import GPUMemoryInfo, GPUMemoryMonitor
from faaslora.memory.memory_coordinator import MemoryCoordinator
from faaslora.memory.residency_manager import ResidencyManager
from faaslora.api.grpc_server import GRPCServer
from faaslora.coordination.autoscaler import AutoScaler, ScalingAction
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
from faaslora.experiment.instance_pool import (
    InstancePool,
    InstanceSlot,
    ObservedRequestCost,
    Router,
)
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
    InferenceEngine as ScriptInferenceEngine,
    RequestResult,
    ScenarioResult,
    ScenarioRunner,
    SubprocessInferenceEngineProxy,
    _assert_official_workload_sources_available,
    _apply_tp_instance_capacity_guard,
    _adapter_matches_model,
    _build_comparison_table,
    _build_scenario_summaries,
    _kill_stale_dedicated_worker_process_groups,
    _build_local_tp_runtime_env_updates,
    _classify_workload_timing_mode,
    _foreign_gpu_consumers_from_rows,
    _is_fatal_engine_error_message,
    _is_comparable_request,
    _normalize_lora_preparation_mode,
    _prepare_dedicated_subprocess_model_cfg,
    _resolve_runtime_gpu_device_ids,
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
        selected_model = self.experiments["model_profiles"]["qwen_7b_main_v2_publicmix"]["model"]

        self.assertEqual(resource["instance_mode"], "auto")
        self.assertEqual(resource["max_instances"], 4)
        self.assertTrue(resource["effective_capacity_admission_enabled"])
        self.assertEqual(resource["arrival_window_s"], 5.0)
        self.assertEqual(resource["scale_eval_interval_s"], 5.0)
        self.assertEqual(resource["scale_cooldown_s"], 5.0)
        self.assertEqual(workload["sampling_strategy"], "representative")
        self.assertEqual(workload["total_requests"], 500)
        self.assertEqual(workload["concurrency"], 4)
        self.assertEqual(workload["time_scale_factor"], 1.0)
        self.assertEqual(model["name"], selected_model["name"])
        self.assertEqual(model["tensor_parallel_size"], 1)
        self.assertEqual(model["visible_device_ids"], [0, 1, 2, 3])
        self.assertEqual(model["max_model_len"], 1024)
        self.assertEqual(model["max_loras"], 6)
        self.assertEqual(model["max_num_seqs"], 2)
        self.assertEqual(model["max_num_batched_tokens"], 1024)
        self.assertEqual(model["runtime_concurrency_cap"], 2)
        self.assertFalse(model["enable_chunked_prefill"])
        self.assertFalse(model["enable_prefix_caching"])
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

    def test_official_workload_guard_rejects_missing_azure_trace(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "Refusing to fall back to synthetic_poisson or fixed-default token lengths",
        ):
            _assert_official_workload_sources_available(
                arrival_source="azure_llm",
                token_source="azure_llm",
                prompt_source="sharegpt_auto",
                has_azure=False,
                has_sgpt=True,
            )

    def test_official_workload_guard_allows_explicit_synthetic_path(self) -> None:
        _assert_official_workload_sources_available(
            arrival_source="synthetic_poisson",
            token_source="fixed_default",
            prompt_source="embedded",
            has_azure=False,
            has_sgpt=False,
        )

    def test_official_workload_guard_rejects_missing_sharegpt_data(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "Refusing to fall back to embedded prompts",
        ):
            _assert_official_workload_sources_available(
                arrival_source="azure_llm",
                token_source="azure_llm",
                prompt_source="sharegpt_auto",
                has_azure=True,
                has_sgpt=False,
            )

    def test_workload_timing_mode_classifies_real_and_accelerated_replay(self) -> None:
        self.assertEqual(
            _classify_workload_timing_mode(
                use_azure_replay=True,
                time_scale_factor=1.0,
            ),
            "azure_real_time",
        )
        self.assertEqual(
            _classify_workload_timing_mode(
                use_azure_replay=True,
                time_scale_factor=0.02,
            ),
            "azure_accelerated_replay",
        )
        self.assertEqual(
            _classify_workload_timing_mode(
                use_azure_replay=False,
                time_scale_factor=1.0,
            ),
            "poisson_synthetic",
        )

    def test_profile_selection_defaults_exist(self) -> None:
        selection = self.experiments["profile_selection"]
        model_profiles = self.experiments["model_profiles"]
        dataset_profiles = self.experiments["dataset_profiles"]
        workload_profiles = self.experiments["workload_profiles"]

        self.assertEqual(selection["model"], "qwen_7b_main_v2_publicmix")
        self.assertEqual(selection["dataset"], "azure_sharegpt_rep500")
        self.assertEqual(selection["workload"], "qwen_7b_auto500_main")
        self.assertIn("qwen_7b_main_v2_publicmix", model_profiles)
        self.assertIn("azure_sharegpt_rep500", dataset_profiles)
        self.assertIn("azure_sharegpt_rep4000", dataset_profiles)
        self.assertIn("qwen_7b_auto500_main", workload_profiles)

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

        self.assertEqual(storage["remote_dir"], "artifacts/frozen/qwen_7b_a500_v2_publicmix")

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

    def test_tp_capacity_guard_allows_quad_gpu_tp2_two_instances(self) -> None:
        model = {"tensor_parallel_size": 2, "visible_device_ids": [0, 1, 2, 3]}
        coord = {"min_instances": 1, "max_instances": 4}

        guarded, meta = _apply_tp_instance_capacity_guard(
            model,
            coord,
            fallback_gpu_count=4,
        )

        self.assertEqual(guarded["max_instances"], 2)
        self.assertEqual(guarded["min_instances"], 1)
        self.assertEqual(meta["max_tp_instances"], 2)

    def test_tp_runtime_env_prefers_loopback_for_local_mp(self) -> None:
        env = _build_local_tp_runtime_env_updates(tp=2, executor_backend="mp")

        self.assertEqual(env["MASTER_ADDR"], "127.0.0.1")
        self.assertTrue(int(env["MASTER_PORT"]) > 0)
        self.assertEqual(env["VLLM_HOST_IP"], "127.0.0.1")
        self.assertEqual(env["GLOO_SOCKET_IFNAME"], "lo")
        self.assertEqual(env["NCCL_SOCKET_IFNAME"], "lo")

    def test_tp_runtime_env_reuses_existing_master_port(self) -> None:
        with patch.dict(os.environ, {"MASTER_PORT": "29611"}, clear=False):
            env = _build_local_tp_runtime_env_updates(tp=2, executor_backend="mp")

        self.assertEqual(env["MASTER_PORT"], "29611")

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

    def test_tp2_dedicated_vllm_uses_subprocess_isolation(self) -> None:
        model = {"backend": "vllm", "tensor_parallel_size": 2}

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
        self.assertTrue(local_model["skip_stale_gpu_cleanup"])

    def test_prepare_dedicated_subprocess_model_cfg_remaps_tp2_gpu_group_to_local_indices(self) -> None:
        model = {
            "backend": "vllm",
            "tensor_parallel_size": 2,
            "device_id": 0,
            "visible_device_ids": [0, 1, 2, 3],
        }

        local_model, env = _prepare_dedicated_subprocess_model_cfg(
            model,
            device_id=2,
        )

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,3")
        self.assertEqual(env["FAASLORA_VISIBLE_DEVICES"], "2,3")
        self.assertEqual(local_model["device_id"], 0)
        self.assertEqual(local_model["visible_device_ids"], [0, 1])
        self.assertTrue(local_model["skip_stale_gpu_cleanup"])

    def test_subprocess_spawn_uses_dedicated_worker_cleanup_not_global_gpu_cleanup(self) -> None:
        class FakeProcess:
            def __init__(self, pid: int):
                self.pid = pid

            def poll(self):
                return None

        calls = {"dedicated_cleanup": 0}
        SubprocessInferenceEngineProxy._startup_cleanup_done = False

        with TemporaryDirectory() as tmpdir:
            worker_root = Path(tmpdir) / "faaslora_worker_test"
            worker_root.mkdir(parents=True, exist_ok=True)

            def fake_popen(cmd, **kwargs):
                ready_path = Path(cmd[cmd.index("--ready-file") + 1])
                ready_path.write_text(
                    json.dumps({"status": "ready", "host": "127.0.0.1", "port": 18080}),
                    encoding="utf-8",
                )
                return FakeProcess(pid=123456)

            async def run_spawn():
                with patch(
                    "scripts.run_all_experiments._kill_stale_dedicated_worker_process_groups",
                    side_effect=lambda *args, **kwargs: calls.__setitem__("dedicated_cleanup", calls["dedicated_cleanup"] + 1) or 0,
                ), patch(
                    "scripts.run_all_experiments._kill_stale_gpu_processes",
                    side_effect=AssertionError("global gpu cleanup should not run during dedicated spawn"),
                ), patch(
                    "scripts.run_all_experiments._prepare_dedicated_subprocess_model_cfg",
                    return_value=({"backend": "vllm", "tensor_parallel_size": 2, "device_id": 0}, {}),
                ), patch(
                    "scripts.run_all_experiments.tempfile.mkdtemp",
                    return_value=str(worker_root),
                ), patch(
                    "scripts.run_all_experiments.subprocess.Popen",
                    side_effect=fake_popen,
                ), patch.object(
                    SubprocessInferenceEngineProxy,
                    "_write_process_meta",
                    return_value=None,
                ):
                    proxy = await SubprocessInferenceEngineProxy.spawn(
                        model_cfg={"backend": "vllm", "tensor_parallel_size": 2},
                        cost_model={},
                        device_id=2,
                    )
                return proxy

            proxy = asyncio.run(run_spawn())

        self.assertEqual(calls["dedicated_cleanup"], 1)
        self.assertEqual(proxy.device_id, 2)
        self.assertEqual(proxy._host, "127.0.0.1")
        self.assertEqual(proxy._port, 18080)

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

    def test_refresh_slot_runtime_hints_attaches_scaleup_handoff_plan(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._stack = None
        runner._gpu_runtime_snapshot = lambda _device_id: None
        runner._instance_mode = "auto"
        runner._scaleup_runtime_handoff_plans = {
            "inst_2": {
                "planned_adapters": ["adapter_b", "adapter_a"],
                "first_service_request_count": 2,
            }
        }

        slot = SimpleNamespace(
            instance_id="inst_2",
            coordinator=SimpleNamespace(
                get_summary_metrics=lambda: {},
                _get_resident_loras=lambda: {},
            ),
            device_id=0,
            gpu_resident_adapters=set(),
            host_cached_adapters=set(),
            nvme_cached_adapters=set(),
            scaleup_handoff_assigned_requests=1,
        )

        runner._refresh_slot_runtime_hints(slot)

        self.assertEqual(slot.scaleup_handoff_planned_adapters, ["adapter_b", "adapter_a"])
        self.assertEqual(
            slot.scaleup_handoff_planned_adapter_ranks,
            {"adapter_b": 0, "adapter_a": 1},
        )
        self.assertEqual(slot.scaleup_handoff_request_budget, 2)
        self.assertEqual(slot.scaleup_handoff_assigned_requests, 1)

    def test_exec_request_preserves_source_cache_tier_after_gpu_promotion(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.router = None
        runner.engine = None
        runner._refresh_all_slot_runtime_hints = lambda: None
        runner._refresh_slot_runtime_hints = lambda _slot: None
        marks = []
        events = []
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
            record_access=lambda adapter_id, load_time_ms=0.0, hit=True: (
                events.append(("record_access", adapter_id, load_time_ms, hit)),
                access_log.append((adapter_id, load_time_ms, hit)),
            )[-1],
        )

        class FakeCoord:
            def __init__(self):
                self._residency_manager = None
                self.marked = []

            def notify_batch_start(self, _tokens: int, _output_tokens_hint: int = 0):
                return None

            def notify_batch_end(self, _tokens: int, _output_tokens_hint: int = 0):
                return None

            async def _mark_resident(self, adapter_id: str, size_mb: float):
                self.marked.append((adapter_id, size_mb))
                return True

        class FakeEngine:
            async def load_lora_to_gpu_and_measure(self, lora_path: str, adapter_id: str):
                return 5.0, True

            async def generate(self, prompt, lora_path, adapter_id, max_tokens, input_tokens, temperature):
                events.append(("generate", adapter_id))
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
        self.assertLess(
            events.index(("record_access", "finance_lora", 5.0, True)),
            events.index(("generate", "finance_lora")),
        )

    def test_generator_defaults_follow_selected_profile(self) -> None:
        defaults = resolve_generation_defaults(EXPERIMENTS_CONFIG)

        self.assertEqual(
            defaults["model"],
            self.experiments["model_profiles"]["qwen_7b_main_v2_publicmix"]["model"]["name"],
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
            stack.record_access(adapter_a, load_time_ms=0.0, hit=True)

            class FakeCoord:
                def _contention_pressure(self) -> float:
                    return 0.0

                def compute_faaslora_host_load_ms(self, _size_mb: float) -> float:
                    return 4.0

                def compute_faaslora_nvme_load_ms(self, _size_mb: float) -> float:
                    return 8.0

                def evaluate_gpu_admission(
                    self,
                    _adapter_id: str,
                    _size_mb: float,
                    tier: str = "nvme",
                    utility_override=None,
                ):
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
            stack.record_access(adapter_hot, load_time_ms=0.0, hit=True)

            class FakeCoord:
                def _contention_pressure(self) -> float:
                    return 0.0

                def compute_faaslora_host_load_ms(self, _size_mb: float) -> float:
                    return 4.0

                def compute_faaslora_nvme_load_ms(self, _size_mb: float) -> float:
                    return 8.0

                def evaluate_gpu_admission(
                    self,
                    _adapter_id: str,
                    _size_mb: float,
                    tier: str = "nvme",
                    utility_override=None,
                ):
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
        schedule_all_calls = []
        runner._schedule_all_runtime_gpu_forward = lambda: schedule_all_calls.append("called") or 0
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

            def evaluate_gpu_admission(
                self,
                _adapter_id: str,
                _size_mb: float,
                tier: str = "nvme",
                utility_override=None,
            ):
                return {
                    "admit": False,
                    "should_attempt": True,
                    "effective_capacity_mb": 0.0,
                    "tier": tier,
                }

        slot = SimpleNamespace(
            engine=FakeEngine(),
            coordinator=FakeCoord(),
            active_requests=0,
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
        self.assertEqual(schedule_all_calls, [])

    def test_dynamic_forwarding_uses_online_hotness_for_cold_instance_admission(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            remote_dir = root / "remote"
            nvme_dir = root / "nvme"
            host_dir = root / "host"
            adapter_hot = "hot_host"
            src = remote_dir / adapter_hot
            src.mkdir(parents=True, exist_ok=True)
            Path(src, "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct"}),
                encoding="utf-8",
            )
            Path(src, "adapter_model.bin").write_bytes(b"ok")

            stack = ExperimentStack(
                adapter_info={adapter_hot: {"size_mb": 16.0, "hotness": 0.9}},
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
            stack.sync_local_tier_paths()
            stack.record_access(adapter_hot, load_time_ms=0.0, hit=True)

            observed = {}

            class FakeCoord:
                def _contention_pressure(self) -> float:
                    return 0.0

                def _locality_factor(self, tier: str) -> float:
                    return 1.0 if tier == "host" else 0.75

                def compute_faaslora_host_load_ms(self, _size_mb: float) -> float:
                    return 4.0

                def compute_faaslora_nvme_load_ms(self, _size_mb: float) -> float:
                    return 8.0

                def evaluate_gpu_admission(
                    self,
                    _adapter_id: str,
                    _size_mb: float,
                    tier: str = "nvme",
                    utility_override=None,
                ):
                    observed["utility_override"] = utility_override
                    return {
                        "admit": float(utility_override or 0.0) > 0.5,
                        "should_attempt": True,
                        "effective_capacity_mb": 1024.0,
                        "tier": tier,
                    }

            candidate = stack.select_gpu_forward_candidate(
                gpu_resident_adapters=set(),
                coordinator=FakeCoord(),
            )

            self.assertIsNotNone(candidate)
            self.assertEqual(candidate["adapter_id"], adapter_hot)
            self.assertGreater(float(observed.get("utility_override", 0.0) or 0.0), 0.5)
            self.assertGreater(float(candidate.get("admission_utility", 0.0) or 0.0), 0.5)

    def test_dynamic_forwarding_can_be_restricted_to_preferred_gpu_hotset(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            remote_dir = root / "remote"
            nvme_dir = root / "nvme"
            host_dir = root / "host"
            adapter_hot = "hot_host"
            adapter_cold = "cold_host"
            for adapter_id in (adapter_hot, adapter_cold):
                src = remote_dir / adapter_id
                src.mkdir(parents=True, exist_ok=True)
                Path(src, "adapter_config.json").write_text(
                    json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct"}),
                    encoding="utf-8",
                )
                Path(src, "adapter_model.bin").write_bytes(b"ok")

            stack = ExperimentStack(
                adapter_info={
                    adapter_hot: {"size_mb": 16.0, "hotness": 0.9},
                    adapter_cold: {"size_mb": 16.0, "hotness": 0.1},
                },
                hardware_cfg={"gpu_budget_mb": 24000},
                coord_cfg={"min_instances": 1, "max_instances": 2},
                preload_cfg={
                    "strategy": "hybrid",
                    "nvme_capacity_mb": 20480,
                    "host_capacity_mb": 4096,
                    "min_hotness": 0.0,
                    "dynamic_forwarding_enabled": True,
                },
                remote_dir=remote_dir,
                nvme_dir=nvme_dir,
                host_dir=host_dir,
            )
            stack._ensure_registered()
            asyncio.run(stack.residency_manager.admit_artifact(adapter_hot, StorageTier.HOST))
            asyncio.run(stack.residency_manager.admit_artifact(adapter_cold, StorageTier.HOST))
            stack.sync_local_tier_paths()
            stack.record_access(adapter_cold, load_time_ms=0.0, hit=True)

            class FakeCoord:
                def _contention_pressure(self) -> float:
                    return 0.0

                def _locality_factor(self, tier: str) -> float:
                    return 1.0 if tier == "host" else 0.75

                def compute_faaslora_host_load_ms(self, _size_mb: float) -> float:
                    return 4.0

                def compute_faaslora_nvme_load_ms(self, _size_mb: float) -> float:
                    return 8.0

                def evaluate_gpu_admission(
                    self,
                    _adapter_id: str,
                    _size_mb: float,
                    tier: str = "nvme",
                    utility_override=None,
                ):
                    return {
                        "admit": True,
                        "should_attempt": True,
                        "effective_capacity_mb": 1024.0,
                        "tier": tier,
                    }

            candidate = stack.select_gpu_forward_candidate(
                gpu_resident_adapters=set(),
                coordinator=FakeCoord(),
                preferred_gpu_adapters={adapter_cold},
            )

            self.assertIsNotNone(candidate)
            self.assertEqual(candidate["adapter_id"], adapter_cold)

    def test_online_dynamic_forwarding_allows_controlled_background_activity(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.model_cfg = {"runtime_concurrency_cap": 2}
        runner.wl_cfg = {"concurrency": 2}

        slot = SimpleNamespace(active_requests=0, load_queue_depth=0)

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
        slot.active_requests = 1
        self.assertFalse(runner._runtime_forward_has_capacity(slot, CoordIdle()))
        slot.active_requests = 0
        self.assertFalse(runner._runtime_forward_has_capacity(slot, CoordSaturated()))

    def test_schedule_runtime_gpu_forward_skips_when_preferred_hotset_is_already_mirrored(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._dynamic_forwarding_enabled = True
        runner._stack = SimpleNamespace(
            select_gpu_forward_candidate=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("select_gpu_forward_candidate should not be called")
            )
        )
        runner._runtime_gpu_forward_tasks = {}
        runner._runtime_forward_task_key = lambda _slot: ("slot",)
        runner._runtime_forward_has_capacity = lambda _slot, _coord: True
        runner._runtime_forward_preferred_gpu_adapters = lambda _slot: {"hot_a", "hot_b"}

        slot = SimpleNamespace(
            engine=SimpleNamespace(load_lora_to_gpu_and_measure=lambda *_args, **_kwargs: None),
            coordinator=object(),
            gpu_resident_adapters={"hot_a", "hot_b"},
        )

        self.assertFalse(runner._schedule_runtime_gpu_forward(slot))

    def test_runtime_forward_preferred_gpu_adapters_filters_stale_sibling_residents(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        now = time.time()
        metadata = {
            "hot_a": SimpleNamespace(last_accessed_at=now - 5.0),
            "stale_b": SimpleNamespace(last_accessed_at=now - 600.0),
        }
        runner._stack = SimpleNamespace(
            hotness_tracker=SimpleNamespace(window_seconds=300.0),
            _artifact_metadata=lambda adapter_id: metadata.get(adapter_id),
        )
        current_slot = SimpleNamespace(
            runtime_group_key=lambda: ("self",),
            gpu_resident_adapters={"self_only"},
        )
        sibling_slot = SimpleNamespace(
            runtime_group_key=lambda: ("sib",),
            gpu_resident_adapters={"hot_a", "stale_b"},
        )
        runner.instance_pool = SimpleNamespace(
            get_runtime_groups=lambda: [[current_slot], [sibling_slot]]
        )

        preferred = runner._runtime_forward_preferred_gpu_adapters(current_slot)

        self.assertEqual(preferred, ["hot_a"])

    def test_runtime_forward_preferred_gpu_adapters_prioritize_handoff_and_waiting_queue(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        now = time.time()
        runner._scaleup_runtime_instance_ids = {"inst_2"}
        runner._runtime_forward_capacity_limit = lambda: 2
        runner._last_scale_up_handoff_plan = {
            "planned_adapters": ["handoff_first", "handoff_second"],
            "ordered_handoff_adapters": [
                "handoff_first",
                "handoff_second",
                "handoff_third",
            ],
        }
        runner._live_waiting_traces_by_id = {
            "req_1": SimpleNamespace(adapter_id="queue_first"),
            "req_2": SimpleNamespace(adapter_id="queue_second"),
            "req_3": SimpleNamespace(adapter_id="queue_only"),
        }
        metadata = {
            "handoff_first": SimpleNamespace(last_accessed_at=0.0),
            "handoff_second": SimpleNamespace(last_accessed_at=0.0),
            "handoff_third": SimpleNamespace(last_accessed_at=0.0),
            "queue_first": SimpleNamespace(last_accessed_at=now - 5.0),
            "queue_second": SimpleNamespace(last_accessed_at=now - 3.0),
            "queue_only": SimpleNamespace(last_accessed_at=0.0),
            "hot_a": SimpleNamespace(last_accessed_at=now - 5.0),
        }
        runner._stack = SimpleNamespace(
            hotness_tracker=SimpleNamespace(window_seconds=300.0),
            _artifact_metadata=lambda adapter_id: metadata.get(adapter_id),
        )
        current_slot = SimpleNamespace(
            instance_id="inst_2",
            runtime_group_key=lambda: ("self",),
            gpu_resident_adapters={"handoff_first"},
        )
        sibling_slot = SimpleNamespace(
            runtime_group_key=lambda: ("sib",),
            gpu_resident_adapters={"hot_a", "queue_second", "queue_first"},
        )
        runner.instance_pool = SimpleNamespace(
            get_runtime_groups=lambda: [[current_slot], [sibling_slot]]
        )

        preferred = runner._runtime_forward_preferred_gpu_adapters(current_slot)

        self.assertEqual(
            preferred,
            ["handoff_second", "queue_first", "queue_second", "hot_a"],
        )

    def test_runtime_forward_preferred_gpu_adapters_use_instance_scoped_handoff_tail(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        now = time.time()
        runner._scaleup_runtime_instance_ids = {"inst_2"}
        runner._scaleup_runtime_handoff_plans = {
            "inst_2": {
                "planned_adapters": ["handoff_first", "handoff_second"],
                "ordered_handoff_adapters": [
                    "handoff_first",
                    "handoff_second",
                    "handoff_tail_1",
                    "handoff_tail_2",
                    "handoff_tail_3",
                ],
            }
        }
        runner._last_scale_up_handoff_plan = {
            "planned_adapters": ["wrong_global"],
            "ordered_handoff_adapters": ["wrong_global", "wrong_tail"],
        }
        runner._runtime_forward_capacity_limit = lambda: 2
        runner._live_waiting_traces_by_id = {
            "req_1": SimpleNamespace(adapter_id="queue_first"),
            "req_2": SimpleNamespace(adapter_id="hot_a"),
        }
        metadata = {
            "handoff_tail_1": SimpleNamespace(last_accessed_at=now - 4.0),
            "handoff_tail_2": SimpleNamespace(last_accessed_at=now - 3.0),
            "queue_first": SimpleNamespace(last_accessed_at=now - 2.0),
            "hot_a": SimpleNamespace(last_accessed_at=now - 1.0),
        }
        runner._stack = SimpleNamespace(
            hotness_tracker=SimpleNamespace(window_seconds=300.0),
            _artifact_metadata=lambda adapter_id: metadata.get(adapter_id),
        )
        current_slot = SimpleNamespace(
            instance_id="inst_2",
            runtime_group_key=lambda: ("self",),
            gpu_resident_adapters={"handoff_first"},
        )
        sibling_slot = SimpleNamespace(
            runtime_group_key=lambda: ("sib",),
            gpu_resident_adapters={"handoff_tail_1", "handoff_tail_2", "queue_first", "hot_a"},
        )
        runner.instance_pool = SimpleNamespace(
            get_runtime_groups=lambda: [[current_slot], [sibling_slot]]
        )

        preferred = runner._runtime_forward_preferred_gpu_adapters(current_slot)

        self.assertEqual(
            preferred,
            ["handoff_second", "handoff_tail_1", "handoff_tail_2", "queue_first", "hot_a"],
        )

    def test_runtime_forward_preferred_gpu_adapters_do_not_expand_waiting_queue_without_gpu_signal(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        now = time.time()
        runner._scaleup_runtime_instance_ids = set()
        runner._last_scale_up_handoff_plan = {}
        runner._live_waiting_traces_by_id = {
            "req_1": SimpleNamespace(adapter_id="queue_only"),
            "req_2": SimpleNamespace(adapter_id="hot_a"),
        }
        metadata = {
            "queue_only": SimpleNamespace(last_accessed_at=0.0),
            "hot_a": SimpleNamespace(last_accessed_at=now - 5.0),
        }
        runner._stack = SimpleNamespace(
            hotness_tracker=SimpleNamespace(window_seconds=300.0),
            _artifact_metadata=lambda adapter_id: metadata.get(adapter_id),
        )
        current_slot = SimpleNamespace(
            instance_id="inst_1",
            runtime_group_key=lambda: ("self",),
            gpu_resident_adapters=set(),
        )
        sibling_slot = SimpleNamespace(
            runtime_group_key=lambda: ("sib",),
            gpu_resident_adapters={"hot_a"},
        )
        runner.instance_pool = SimpleNamespace(
            get_runtime_groups=lambda: [[current_slot], [sibling_slot]]
        )

        preferred = runner._runtime_forward_preferred_gpu_adapters(current_slot)

        self.assertEqual(preferred, ["hot_a"])

    def test_runtime_forward_preferred_gpu_adapters_rank_opportunistic_tail_by_observed_utility(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        now = time.time()
        size_mb = {
            "small_wait": 12.0,
            "large_wait": 120.0,
        }
        runner._scaleup_runtime_instance_ids = set()
        runner._last_scale_up_handoff_plan = {}
        runner._live_waiting_traces_by_id = {
            "req_1": SimpleNamespace(adapter_id="large_wait"),
            "req_2": SimpleNamespace(adapter_id="small_wait"),
        }
        runner._stack = SimpleNamespace(
            hotness_tracker=SimpleNamespace(window_seconds=300.0),
            _artifact_metadata=lambda _adapter_id: SimpleNamespace(last_accessed_at=now - 5.0),
            _artifact_size_mb=lambda adapter_id: size_mb[adapter_id],
            _online_artifact_hotness=lambda _adapter_id: 1.0,
        )
        current_slot = SimpleNamespace(
            instance_id="inst_2",
            runtime_group_key=lambda: ("self",),
            gpu_resident_adapters=set(),
            predicted_lora_io_ms=lambda **kwargs: 240.0,
        )
        sibling_slot = SimpleNamespace(
            runtime_group_key=lambda: ("sib",),
            gpu_resident_adapters={"large_wait", "small_wait"},
        )
        runner.instance_pool = SimpleNamespace(
            get_runtime_groups=lambda: [[current_slot], [sibling_slot]]
        )

        preferred = runner._runtime_forward_preferred_gpu_adapters(current_slot)

        self.assertEqual(preferred, ["small_wait", "large_wait"])

    def test_schedule_runtime_gpu_forward_skips_when_only_stale_sibling_gpu_set_remains(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._dynamic_forwarding_enabled = True
        runner._stack = SimpleNamespace(
            hotness_tracker=SimpleNamespace(window_seconds=300.0),
            _artifact_metadata=lambda _adapter_id: SimpleNamespace(last_accessed_at=time.time() - 600.0),
            select_gpu_forward_candidate=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("select_gpu_forward_candidate should not be called")
            ),
        )
        runner._runtime_gpu_forward_tasks = {}
        current_slot = SimpleNamespace(
            runtime_group_key=lambda: ("self",),
            engine=SimpleNamespace(load_lora_to_gpu_and_measure=lambda *_args, **_kwargs: None),
            coordinator=object(),
            gpu_resident_adapters=set(),
        )
        sibling_slot = SimpleNamespace(
            runtime_group_key=lambda: ("sib",),
            gpu_resident_adapters={"stale_only"},
        )
        runner.instance_pool = SimpleNamespace(get_runtime_groups=lambda: [[current_slot], [sibling_slot]])
        runner._runtime_forward_has_capacity = lambda _slot, _coord: True

        self.assertFalse(runner._schedule_runtime_gpu_forward(current_slot))

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

    def test_notify_batch_tracks_decode_hint_in_active_tokens(self) -> None:
        coord = ResourceCoordinator(config={"kv_per_1k_tokens_mb": 2.0})

        coord.notify_batch_start(100, 300)
        self.assertEqual(coord._active_tokens, 400)
        self.assertEqual(coord._active_batches, 1)
        self.assertAlmostEqual(coord._recent_batch_tokens_ewma, 400.0)

        coord.notify_batch_end(100, 300)
        self.assertEqual(coord._active_tokens, 0)
        self.assertEqual(coord._active_batches, 0)

    def test_record_memory_sample_includes_kv_tokens(self) -> None:
        coord = ResourceCoordinator(config={
            "gpu_budget_mb": 1000,
            "model_weights_mb": 100,
            "kv_per_1k_tokens_mb": 2.0,
        })
        coord._resident_loras["adapter_a"] = 100.0
        coord._active_tokens = 2000

        coord._record_memory_sample()

        self.assertAlmostEqual(coord.metrics.peak_memory_utilization, 0.204)
        self.assertAlmostEqual(coord.metrics.avg_memory_utilization, 0.204)

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
    def test_foreign_gpu_consumers_filter_own_process_family_and_small_residuals(self) -> None:
        offenders = _foreign_gpu_consumers_from_rows(
            [0, 1],
            gpu_uuid_rows=[
                "0, GPU-AAA",
                "1, GPU-BBB",
            ],
            compute_app_rows=[
                "GPU-AAA, 101, /home/qhq/anaconda3/envs/LLM_vllm0102/bin/python, 17366 MiB",
                "GPU-AAA, 202, /app/.venv/bin/python, 6506 MiB",
                "GPU-BBB, 303, /usr/bin/python3, 15 MiB",
            ],
            allowed_pids={101},
            min_used_mib=256,
        )

        self.assertEqual(
            offenders,
            [
                {
                    "gpu_index": 0,
                    "pid": 202,
                    "process_name": "/app/.venv/bin/python",
                    "used_mib": 6506,
                }
            ],
        )

    def test_runner_gpu_environment_guard_detects_foreign_consumer(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._gpu_environment_guard_last_at = 0.0
        runner._gpu_environment_guard_interval_s = 30.0
        runner._available_device_ids = lambda: [0, 1]

        with patch("scripts.run_all_experiments._assert_clean_gpu_environment") as guard:
            runner._assert_clean_gpu_environment(context="live_progress", force=True)

        guard.assert_called_once()

    def test_fatal_engine_error_message_detects_oom_and_background_loop(self) -> None:
        self.assertTrue(_is_fatal_engine_error_message("CUDA out of memory. Tried to allocate 34 MiB"))
        self.assertTrue(_is_fatal_engine_error_message("vLLM: Background loop has errored already."))
        self.assertFalse(_is_fatal_engine_error_message("temporary timeout while polling"))

    def test_retire_failed_slot_reassigns_primary_instance(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.instance_pool = InstancePool(min_instances=1, max_instances=2)
        inst_1 = runner.instance_pool.add_instance(
            SimpleNamespace(_engine_dead=True),
            SimpleNamespace(),
            owns_engine=True,
            device_id=0,
        )
        inst_2 = runner.instance_pool.add_instance(
            SimpleNamespace(_engine_dead=False),
            SimpleNamespace(),
            owns_engine=True,
            device_id=1,
        )
        runner._primary_instance_id = inst_1
        runner._slot_retire_lock = asyncio.Lock()
        runner._scaleup_runtime_instance_ids = set()
        runner._runtime_gpu_forward_tasks = {}

        async def _cleanup_removed_slot(_slot) -> None:
            return None

        runner._cleanup_removed_slot = _cleanup_removed_slot

        retired = asyncio.run(runner._retire_failed_slot(runner.instance_pool.get_slot(inst_1), "oom"))

        self.assertTrue(retired)
        self.assertEqual(runner._primary_instance_id, inst_2)
        self.assertEqual([slot.instance_id for slot in runner.instance_pool.get_slots()], [inst_2])

    def test_runtime_gpu_device_ids_use_bound_device_for_tp1(self) -> None:
        device_ids = _resolve_runtime_gpu_device_ids(
            {
                "tensor_parallel_size": 1,
                "device_id": 0,
                "visible_device_ids": [0, 1],
            }
        )

        self.assertEqual(device_ids, [0])

    def test_available_device_ids_use_runner_scope_not_primary_local_subprocess_mask(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.model_cfg = {"visible_device_ids": [0, 1]}
        runner.hw = {}
        runner.engine = SimpleNamespace(model_cfg={"visible_device_ids": [0]})

        self.assertEqual(runner._available_device_ids(), [0, 1])

    def test_scenario_runner_init_preserves_global_scope_when_primary_engine_is_local_subprocess(self) -> None:
        runner = ScenarioRunner(
            name="s",
            baseline_type="faaslora_full",
            adapter_info={},
            traces=[],
            remote_dir=PROJECT_ROOT,
            nvme_dir=PROJECT_ROOT,
            bandwidth_mbps=100.0,
            hardware_cfg={"gpu_device_ids": [0, 1]},
            cost_model={},
            engine=SimpleNamespace(device_id=0, model_cfg={"visible_device_ids": [0], "tensor_parallel_size": 1}),
            runner_model_cfg={"visible_device_ids": [0, 1], "tensor_parallel_size": 1},
            preload_cfg={},
            workload_cfg={},
            coord_cfg={},
        )

        self.assertEqual(runner.model_cfg["visible_device_ids"], [0, 1])
        self.assertEqual(runner.hw["gpu_device_ids"], [0, 1])
        self.assertEqual(runner._available_device_ids(), [0, 1])

    def test_script_inference_engine_cleanup_respects_skip_flag(self) -> None:
        engine = ScriptInferenceEngine({"skip_stale_gpu_cleanup": True}, {})

        with patch("scripts.run_all_experiments._kill_stale_gpu_processes") as kill:
            engine._maybe_kill_stale_gpu_processes()

        kill.assert_not_called()

    def test_script_inference_engine_generate_prefers_vllm_request_metrics_for_ttft_and_tpot(self) -> None:
        class FakeVllmEngine:
            async def generate(self, **_kwargs):
                yield SimpleNamespace(
                    outputs=[SimpleNamespace(token_ids=[11, 12, 13], text="")],
                    metrics=SimpleNamespace(
                        arrival_time=10.0,
                        first_token_time=10.5,
                        finished_time=11.1,
                        last_token_time=11.1,
                    ),
                )

        engine = ScriptInferenceEngine.__new__(ScriptInferenceEngine)
        engine.model_cfg = {"backend": "vllm"}
        engine.cost_model = {}
        engine.engine = FakeVllmEngine()
        engine.backend = "vllm"
        engine.device_id = 0
        engine._counter = 0
        engine._lock = asyncio.Lock()
        engine._reinit_lock = asyncio.Lock()
        engine._engine_dead = False
        engine._lora_in_engine = False
        engine._reinit_attempted = False
        engine._prepare_vllm_prompt = (
            lambda prompt, max_tokens, input_tokens_hint: (
                prompt,
                input_tokens_hint,
                max_tokens,
            )
        )

        with patch(
            "scripts.run_all_experiments.SamplingParams",
            side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
        ), patch(
            "scripts.run_all_experiments.time.perf_counter",
            side_effect=[100.0, 180.0, 250.0],
        ):
            ttft_ms, tpot_ms, output_tokens = asyncio.run(
                engine.generate(
                    prompt="hello",
                    lora_path=None,
                    adapter_id=None,
                    max_tokens=8,
                    input_tokens=4,
                    temperature=0.7,
                    top_p=0.9,
                )
            )

        self.assertEqual(output_tokens, 3)
        self.assertAlmostEqual(ttft_ms, 500.0)
        self.assertAlmostEqual(tpot_ms, 300.0)

    def test_runtime_gpu_device_ids_use_local_visible_set_for_tp2(self) -> None:
        device_ids = _resolve_runtime_gpu_device_ids(
            {
                "tensor_parallel_size": 2,
                "device_id": 0,
                "visible_device_ids": [0, 1],
            }
        )

        self.assertEqual(device_ids, [0, 1])

    def test_runtime_gpu_device_ids_use_group_anchor_for_tp2_scale_out(self) -> None:
        device_ids = _resolve_runtime_gpu_device_ids(
            {
                "tensor_parallel_size": 2,
                "device_id": 2,
                "visible_device_ids": [0, 1, 2, 3],
            }
        )

        self.assertEqual(device_ids, [2, 3])

    def test_schedule_all_runtime_gpu_forward_prioritizes_idle_runtime(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        scheduled = []

        idle_slot = SimpleNamespace(
            instance_id="inst_1",
            active_requests=0,
            last_selected_at=10.0,
            created_at=1.0,
        )
        busy_slot = SimpleNamespace(
            instance_id="inst_2",
            active_requests=2,
            last_selected_at=20.0,
            created_at=2.0,
        )
        runner.instance_pool = SimpleNamespace(get_slots=lambda: [busy_slot, idle_slot])
        runner._schedule_runtime_gpu_forward = lambda slot: scheduled.append(slot.instance_id) or True

        scheduled_count = runner._schedule_all_runtime_gpu_forward()

        self.assertEqual(scheduled_count, 2)
        self.assertEqual(scheduled, ["inst_1", "inst_2"])

    def test_scale_up_preload_capacity_prefers_explicit_config(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.preload_cfg = {
            "scale_up_preload_mb": 1536,
            "host_capacity_mb": 4096,
        }

        capacity_bytes = runner._scale_up_preload_capacity_bytes()

        self.assertEqual(capacity_bytes, 1536 * 1024 * 1024)

    def test_scale_up_warmup_preferred_gpu_adapters_uses_recent_live_gpu_union(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        slot_a = SimpleNamespace(gpu_resident_adapters={"hot_a", "stale_a"})
        slot_b = SimpleNamespace(gpu_resident_adapters={"hot_b"})
        runner.instance_pool = SimpleNamespace(get_runtime_groups=lambda: [[slot_a], [slot_b]])
        runner._stack = SimpleNamespace(
            sync_local_tier_paths=lambda: None,
            _host_paths={"host_only": "/tmp/host_only"},
            _nvme_paths={"nvme_only": "/tmp/nvme_only"},
        )
        runner._adapter_is_recently_active = lambda aid, now=None: aid not in {"stale_a", "nvme_only"}

        preferred = runner._scale_up_warmup_preferred_gpu_adapters()

        self.assertEqual(preferred, {"hot_a", "hot_b"})

    def test_waiting_visible_trace_queue_prefers_live_waiting_queue_order(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        waiting_a = SimpleNamespace(request_id="r_wait_1", adapter_id="waiting_a")
        waiting_b = SimpleNamespace(request_id="r_wait_2", adapter_id="waiting_b")
        runner._live_waiting_traces_by_id = {
            "r_wait_1": waiting_a,
            "r_wait_2": waiting_b,
        }
        runner._live_arrived_lora_counts = defaultdict(int, {"waiting_a": 1, "waiting_b": 1})
        runner._live_started_lora_counts = defaultdict(int)

        queue = runner._waiting_visible_trace_queue(
            visible_traces=[SimpleNamespace(adapter_id="other")]
        )

        self.assertEqual(queue, [waiting_a, waiting_b])

    def test_update_dynamic_scaling_live_state_uses_waiting_and_started_adapters(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._baseline_rps_ewma_beta = 0.5
        runner._baseline_rps = 2.0
        runner._scale_down_beta = 0.3
        runner._scale_up_t_min = 5.0
        runner._scale_up_alpha = 0.5
        runner._dynamic_scaling = True
        runner._low_load_since = None
        runner._active_loras_ewma = 0.0
        runner._live_waiting_traces_by_id = {
            "r1": SimpleNamespace(request_id="r1", adapter_id="wait_a"),
            "r2": SimpleNamespace(request_id="r2", adapter_id="wait_b"),
        }
        runner._live_started_lora_counts = defaultdict(int, {"wait_a": 1, "exec_c": 1})

        overrides = runner._update_dynamic_scaling_live_state(10.0, 20.0)

        self.assertEqual(runner._current_live_active_adapter_count(), 3)
        self.assertAlmostEqual(runner._active_loras_ewma, 3.0)
        self.assertAlmostEqual(runner._baseline_rps, 6.0)
        self.assertEqual(
            overrides,
            {
                "scale_up_threshold_rps": 9.0,
                "scale_down_threshold_rps": 1.7999999999999998,
            },
        )

    def test_predict_scale_up_handoff_plan_trims_incumbent_progress_and_caps_prefix_by_headroom(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.coord_cfg = {"max_concurrent_loads": 2}
        runner._scale_up_target_runtime_headroom_mb = lambda: 700.0
        runner._scale_up_bootstrap_latency_ms = lambda: 120.0
        runner._scale_up_initial_admission_request_budget = lambda: 2
        runner._runtime_forward_capacity_limit = lambda: 2
        runner._scale_up_ready_candidate_queue = lambda **kwargs: (
            [
                SimpleNamespace(adapter_id="already_started"),
                SimpleNamespace(adapter_id="waiting_a"),
                SimpleNamespace(adapter_id="future_b"),
                SimpleNamespace(adapter_id="future_c"),
            ],
            4,
        )
        runner._scale_up_incumbent_started_request_count = lambda **kwargs: 1
        runner._predicted_scale_up_load_ms = lambda adapter_ids, max_concurrent_loads=1: 0.0
        runner._scale_up_warmup_preferred_gpu_adapters = lambda: {"fallback_only"}
        runner._stack = SimpleNamespace(
            _artifact_size_mb=lambda aid: {
                "waiting_a": 400.0,
                "future_b": 300.0,
                "future_c": 500.0,
            }[aid]
        )

        plan = runner._predict_scale_up_handoff_plan(
            replay_t0=0.0,
            arrived_request_count=4,
            queue_visible_request_count=2,
            visible_traces=[],
        )

        self.assertEqual(plan["ordered_handoff_adapters"], ["waiting_a", "future_b", "future_c"])
        self.assertEqual(plan["planned_adapters"], ["waiting_a", "future_b"])
        self.assertEqual(plan["queue_at_ready_request_count"], 3)
        self.assertEqual(plan["first_service_request_count"], 2)
        self.assertEqual(plan["first_service_adapter_count"], 2)
        self.assertEqual(plan["initial_admission_request_budget"], 2)
        self.assertEqual(plan["incumbent_started_request_count"], 1)
        self.assertEqual(plan["exact_prefix_bytes"], 700 * 1024 * 1024)
        self.assertEqual(
            [trace.adapter_id for trace in plan["_queue_at_ready_traces"]],
            ["waiting_a", "future_b", "future_c"],
        )

    def test_route_aware_scale_up_first_service_prefix_follows_router_selected_requests(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._nvme_cache = {}
        runner._gpu_warmed = set()
        runner.adapter_info = {
            "finance": {"size_mb": 50.0},
            "support": {"size_mb": 50.0},
        }
        runner.cost_model = {"bandwidth_mbps": 1000.0}
        runner._refresh_slot_runtime_hints = lambda slot: None
        coordinator = SimpleNamespace(
            metrics={},
            compute_faaslora_host_load_ms=lambda size_mb: float(size_mb) * 2.0,
            compute_faaslora_nvme_load_ms=lambda size_mb: float(size_mb) * 4.0,
            compute_cold_start_load_ms=lambda size_mb, bandwidth: float(size_mb) * 8.0,
        )
        runner.coordinator = coordinator
        runner._stack = SimpleNamespace(
            _host_paths={"finance": "/tmp/finance", "support": "/tmp/support"},
            _nvme_paths={},
            sync_local_tier_paths=lambda: None,
        )
        incumbent = InstanceSlot("inst_1", None, coordinator, created_at=1.0)
        incumbent.host_cached_adapters = {"finance", "support"}
        incumbent.gpu_resident_adapters = {"finance"}
        incumbent.observed_runtime_ttft_ms = 1000.0
        incumbent.observed_runtime_samples = 1
        incumbent.observed_request_costs = {
            "lora_gpu": ObservedRequestCost(
                samples=1,
                avg_lora_io_ms=0.0,
                avg_runtime_ttft_ms=1000.0,
                avg_tail_service_ms=0.0,
            ),
            "lora_host": ObservedRequestCost(
                samples=1,
                avg_lora_io_ms=100.0,
                avg_runtime_ttft_ms=1000.0,
                avg_tail_service_ms=0.0,
            ),
            "lora_any": ObservedRequestCost(
                samples=2,
                avg_lora_io_ms=50.0,
                avg_runtime_ttft_ms=1000.0,
                avg_tail_service_ms=0.0,
            ),
        }
        pool = SimpleNamespace(get_slots=lambda: [incumbent])
        runner.instance_pool = pool
        runner.router = Router(pool, policy="least_connections", runtime_concurrency_cap=2)
        routed = runner._route_aware_scale_up_first_service_prefix_traces(
            [
                SimpleNamespace(adapter_id="finance"),
                SimpleNamespace(adapter_id="support"),
            ],
            ["finance"],
            request_budget=1,
        )

        self.assertEqual([trace.adapter_id for trace in routed], ["finance"])

    def test_predict_scale_up_handoff_plan_uses_router_aware_first_service_prefix(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.coord_cfg = {"max_concurrent_loads": 1}
        runner._scale_up_target_runtime_headroom_mb = lambda: 60.0
        runner._scale_up_bootstrap_latency_ms = lambda: 0.0
        runner._scale_up_initial_admission_request_budget = lambda: 1
        runner._runtime_forward_capacity_limit = lambda: 2
        runner._predicted_scale_up_load_ms = lambda adapter_ids, max_concurrent_loads=1: 0.0
        runner._scale_up_ready_candidate_queue = lambda **kwargs: (
            [
                SimpleNamespace(adapter_id="finance"),
                SimpleNamespace(adapter_id="support"),
            ],
            2,
        )
        runner._scale_up_incumbent_started_request_count = lambda **kwargs: 0
        runner._scale_up_warmup_preferred_gpu_adapters = lambda: set()
        runner._nvme_cache = {}
        runner._gpu_warmed = set()
        runner.adapter_info = {
            "finance": {"size_mb": 50.0},
            "support": {"size_mb": 50.0},
        }
        runner.cost_model = {"bandwidth_mbps": 1000.0}
        runner._refresh_slot_runtime_hints = lambda slot: None
        coordinator = SimpleNamespace(
            metrics={},
            compute_faaslora_host_load_ms=lambda size_mb: float(size_mb) * 2.0,
            compute_faaslora_nvme_load_ms=lambda size_mb: float(size_mb) * 4.0,
            compute_cold_start_load_ms=lambda size_mb, bandwidth: float(size_mb) * 8.0,
        )
        runner.coordinator = coordinator
        runner._stack = SimpleNamespace(
            _host_paths={"finance": "/tmp/finance", "support": "/tmp/support"},
            _nvme_paths={},
            sync_local_tier_paths=lambda: None,
            _artifact_size_mb=lambda aid: {"finance": 50.0, "support": 50.0}[aid],
        )
        incumbent = InstanceSlot("inst_1", None, coordinator, created_at=1.0)
        incumbent.host_cached_adapters = {"finance", "support"}
        incumbent.gpu_resident_adapters = {"finance"}
        incumbent.observed_runtime_ttft_ms = 1000.0
        incumbent.observed_runtime_samples = 1
        incumbent.observed_request_costs = {
            "lora_gpu": ObservedRequestCost(
                samples=1,
                avg_lora_io_ms=0.0,
                avg_runtime_ttft_ms=1000.0,
                avg_tail_service_ms=0.0,
            ),
            "lora_host": ObservedRequestCost(
                samples=1,
                avg_lora_io_ms=100.0,
                avg_runtime_ttft_ms=1000.0,
                avg_tail_service_ms=0.0,
            ),
            "lora_any": ObservedRequestCost(
                samples=2,
                avg_lora_io_ms=50.0,
                avg_runtime_ttft_ms=1000.0,
                avg_tail_service_ms=0.0,
            ),
        }
        pool = SimpleNamespace(get_slots=lambda: [incumbent])
        runner.instance_pool = pool
        runner.router = Router(pool, policy="least_connections", runtime_concurrency_cap=2)

        plan = runner._predict_scale_up_handoff_plan(
            replay_t0=0.0,
            arrived_request_count=2,
            queue_visible_request_count=2,
            visible_traces=[],
        )

        self.assertEqual(plan["ordered_handoff_adapters"], ["finance", "support"])
        self.assertEqual(plan["planned_adapters"], ["finance"])
        self.assertEqual(plan["first_service_request_count"], 1)
        self.assertEqual(plan["first_service_adapter_count"], 1)

    def test_scale_up_first_service_prefix_traces_skip_backbone_when_forming_warmup_slice(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._scale_up_initial_admission_request_budget = lambda: 1

        prefix = runner._scale_up_first_service_prefix_traces(
            [
                SimpleNamespace(adapter_id=None),
                SimpleNamespace(adapter_id="a1"),
                SimpleNamespace(adapter_id="a2"),
            ]
        )

        self.assertEqual([trace.adapter_id for trace in prefix], ["a1"])

    def test_scale_up_initial_admission_request_budget_tracks_incremental_dispatch_capacity(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._runtime_forward_capacity_limit = lambda: 2
        runner._configured_workload_concurrency = lambda: 8
        runner._current_runtime_group_count = lambda: 3

        self.assertEqual(runner._scale_up_initial_admission_request_budget(), 2)

        runner._configured_workload_concurrency = lambda: 5
        self.assertEqual(runner._scale_up_initial_admission_request_budget(), 1)

    def test_refined_scale_up_target_instances_keeps_scale_count_under_autoscaler_control(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.instance_pool = SimpleNamespace(max_instances=4)

        refined = runner._refined_scale_up_target_instances(
            current_instances=1,
            decision=SimpleNamespace(target_instances=2),
            handoff_plan={
                "queue_at_ready_request_count": 7,
                "first_service_request_count": 3,
                "initial_admission_request_budget": 2,
            },
        )

        self.assertEqual(refined, 2)

    def test_build_scale_up_runtime_handoff_plans_slices_queue_by_first_service_budget(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._predicted_scale_up_load_ms = lambda adapter_ids, max_concurrent_loads=1: float(len(adapter_ids) * 10)
        runner._stack = SimpleNamespace(
            _artifact_size_mb=lambda aid: {
                "a1": 100.0,
                "a2": 100.0,
                "a3": 100.0,
                "a4": 100.0,
            }[aid]
        )
        plans = runner._build_scale_up_runtime_handoff_plans(
            {
                "mode": "dynamic_headroom_exact_handoff_prefix",
                "_queue_at_ready_traces": [
                    SimpleNamespace(adapter_id="a1"),
                    SimpleNamespace(adapter_id="a2"),
                    SimpleNamespace(adapter_id="a3"),
                    SimpleNamespace(adapter_id="a4"),
                ],
                "ordered_handoff_adapters": ["a1", "a2", "a3", "a4"],
                "queue_at_ready_request_count": 4,
                "queue_at_ready_adapter_count": 4,
                "first_service_request_count": 2,
                "initial_admission_request_budget": 2,
                "projected_arrived_request_count": 6,
                "incumbent_started_request_count": 1,
                "bootstrap_latency_ms": 50.0,
                "target_headroom_bytes": 250 * 1024 * 1024,
                "configured_max_concurrent_loads": 2,
            },
            additional_instances=2,
        )

        self.assertEqual(len(plans), 2)
        self.assertEqual(plans[0]["ordered_handoff_adapters"], ["a1", "a2"])
        self.assertEqual(plans[0]["planned_adapters"], ["a1", "a2"])
        self.assertEqual(plans[1]["ordered_handoff_adapters"], ["a3", "a4"])
        self.assertEqual(plans[1]["planned_adapters"], ["a3", "a4"])

    def test_build_scale_up_runtime_handoff_plans_skip_backbone_when_forming_lora_handoff_slice(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._predicted_scale_up_load_ms = lambda adapter_ids, max_concurrent_loads=1: float(len(adapter_ids) * 10)
        runner._stack = SimpleNamespace(
            _artifact_size_mb=lambda aid: {
                "a1": 100.0,
                "a2": 100.0,
            }[aid]
        )

        plans = runner._build_scale_up_runtime_handoff_plans(
            {
                "mode": "dynamic_headroom_exact_handoff_prefix",
                "_queue_at_ready_traces": [
                    SimpleNamespace(adapter_id=None),
                    SimpleNamespace(adapter_id="a1"),
                    SimpleNamespace(adapter_id="a2"),
                ],
                "ordered_handoff_adapters": ["a1", "a2"],
                "queue_at_ready_request_count": 3,
                "queue_at_ready_adapter_count": 2,
                "first_service_request_count": 1,
                "initial_admission_request_budget": 1,
                "projected_arrived_request_count": 6,
                "incumbent_started_request_count": 1,
                "bootstrap_latency_ms": 50.0,
                "target_headroom_bytes": 250 * 1024 * 1024,
                "configured_max_concurrent_loads": 2,
            },
            additional_instances=1,
        )

        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0]["ordered_handoff_adapters"], ["a1"])
        self.assertEqual(plans[0]["planned_adapters"], ["a1"])
        self.assertEqual(plans[0]["first_service_request_count"], 1)
        self.assertEqual(plans[0]["first_service_adapter_count"], 1)
        self.assertEqual(plans[0]["first_service_scanned_request_count"], 2)

    def test_build_scale_up_runtime_handoff_plans_keeps_request_budget_on_initial_admission(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._predicted_scale_up_load_ms = lambda adapter_ids, max_concurrent_loads=1: float(len(adapter_ids) * 10)
        runner._stack = SimpleNamespace(
            _artifact_size_mb=lambda aid: {
                "a1": 100.0,
                "a2": 100.0,
                "a3": 100.0,
            }[aid]
        )

        plans = runner._build_scale_up_runtime_handoff_plans(
            {
                "mode": "dynamic_headroom_exact_handoff_prefix",
                "_queue_at_ready_traces": [
                    SimpleNamespace(adapter_id="a1"),
                    SimpleNamespace(adapter_id="a2"),
                    SimpleNamespace(adapter_id="a3"),
                ],
                "ordered_handoff_adapters": ["a1", "a2", "a3"],
                "queue_at_ready_request_count": 3,
                "queue_at_ready_adapter_count": 3,
                "first_service_request_count": 2,
                "initial_admission_request_budget": 1,
                "effective_initial_warmup_parallelism": 2,
                "projected_arrived_request_count": 6,
                "incumbent_started_request_count": 1,
                "bootstrap_latency_ms": 50.0,
                "target_headroom_bytes": 250 * 1024 * 1024,
                "configured_max_concurrent_loads": 2,
            },
            additional_instances=1,
        )

        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0]["ordered_handoff_adapters"], ["a1"])
        self.assertEqual(plans[0]["planned_adapters"], ["a1"])
        self.assertEqual(plans[0]["first_service_request_count"], 1)
        self.assertEqual(plans[0]["effective_initial_warmup_parallelism"], 2)

    def test_build_scale_up_runtime_handoff_plans_prefers_route_aware_runtime_slices_when_available(self) -> None:
        class DummySlot:
            def __init__(self, instance_id: str) -> None:
                self.instance_id = instance_id
                self.active_requests = 0
                self.load_queue_depth = 0
                self.last_selected_at = 0.0
                self.gpu_resident_adapters = set()
                self.host_cached_adapters = set()
                self.nvme_cached_adapters = set()
                self.observed_request_costs = {}
                self.scaleup_handoff_planned_adapters = []
                self.scaleup_handoff_planned_adapter_ranks = {}

            def mark_adapter_tier(self, adapter_id: str, tier: str) -> None:
                if tier == "gpu":
                    self.gpu_resident_adapters.add(adapter_id)

        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.router = SimpleNamespace(policy="least_connections", runtime_concurrency_cap=2)
        runner.instance_pool = SimpleNamespace(get_slots=lambda: [DummySlot("inst_1")])
        runner._predicted_scale_up_load_ms = (
            lambda adapter_ids, max_concurrent_loads=1: float(len(adapter_ids) * 10)
        )
        runner._stack = SimpleNamespace(
            _artifact_size_mb=lambda aid: {
                "education": 100.0,
                "finance": 100.0,
                "legal": 100.0,
            }[aid]
        )

        def _copy_slot(slot):
            copied = DummySlot(slot.instance_id)
            copied.active_requests = int(getattr(slot, "active_requests", 0) or 0)
            copied.load_queue_depth = int(getattr(slot, "load_queue_depth", 0) or 0)
            copied.last_selected_at = float(getattr(slot, "last_selected_at", 0.0) or 0.0)
            copied.gpu_resident_adapters = set(getattr(slot, "gpu_resident_adapters", set()) or set())
            copied.host_cached_adapters = set(getattr(slot, "host_cached_adapters", set()) or set())
            copied.nvme_cached_adapters = set(getattr(slot, "nvme_cached_adapters", set()) or set())
            copied.observed_request_costs = dict(getattr(slot, "observed_request_costs", {}) or {})
            copied.scaleup_handoff_planned_adapters = list(
                getattr(slot, "scaleup_handoff_planned_adapters", []) or []
            )
            copied.scaleup_handoff_planned_adapter_ranks = dict(
                getattr(slot, "scaleup_handoff_planned_adapter_ranks", {}) or {}
            )
            return copied

        def _make_slot(planned_gpu_adapters, request_budget=0, instance_id=None):
            slot = DummySlot(str(instance_id or "__scaleup_sim__"))
            slot.scaleup_handoff_planned_adapters = list(planned_gpu_adapters or [])
            slot.scaleup_handoff_planned_adapter_ranks = {
                aid: idx for idx, aid in enumerate(slot.scaleup_handoff_planned_adapters)
            }
            for aid in slot.scaleup_handoff_planned_adapters:
                slot.mark_adapter_tier(aid, "gpu")
            return slot

        route_aware_calls = []

        def _route_aware_state(queue_at_ready, simulated_slots, new_slot, request_budget):
            route_aware_calls.append(
                (new_slot.instance_id, [getattr(t, "adapter_id", None) for t in queue_at_ready])
            )
            scanned = [queue_at_ready[0]]
            routed = [queue_at_ready[0]]
            remaining = list(queue_at_ready[1:])
            return scanned, routed, remaining

        runner._copy_scale_up_routing_simulation_slot = _copy_slot
        runner._make_scale_up_routing_simulated_slot = _make_slot
        runner._route_aware_scale_up_first_service_prefix_state = _route_aware_state

        plans = runner._build_scale_up_runtime_handoff_plans(
            {
                "mode": "dynamic_headroom_exact_handoff_prefix",
                "_queue_at_ready_traces": [
                    SimpleNamespace(adapter_id="finance"),
                    SimpleNamespace(adapter_id="legal"),
                ],
                "ordered_handoff_adapters": ["education", "legal"],
                "queue_at_ready_request_count": 2,
                "queue_at_ready_adapter_count": 2,
                "first_service_request_count": 1,
                "initial_admission_request_budget": 1,
                "projected_arrived_request_count": 3,
                "incumbent_started_request_count": 0,
                "bootstrap_latency_ms": 50.0,
                "target_headroom_bytes": 300 * 1024 * 1024,
                "configured_max_concurrent_loads": 2,
            },
            additional_instances=2,
        )

        self.assertEqual(len(route_aware_calls), 4)
        self.assertEqual(len(plans), 2)
        self.assertEqual(plans[0]["ordered_handoff_adapters"], ["finance"])
        self.assertEqual(plans[0]["planned_adapters"], ["finance"])
        self.assertEqual(plans[1]["ordered_handoff_adapters"], ["legal"])
        self.assertEqual(plans[1]["planned_adapters"], ["legal"])

    def test_scale_up_first_service_request_budget_matches_initial_admission_slice(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._scale_up_initial_admission_request_budget = lambda: 1
        runner._runtime_forward_capacity_limit = lambda: 1
        runner._live_scale_eval_period_s = lambda: 15.0
        runner._scale_up_trace_total_busy_estimate_ms = lambda trace: {
            "first": 8200.0,
            "second": 7800.0,
            "third": 9000.0,
        }[trace.adapter_id]

        budget = runner._scale_up_first_service_request_budget(
            [
                SimpleNamespace(adapter_id="first"),
                SimpleNamespace(adapter_id="second"),
                SimpleNamespace(adapter_id="third"),
            ]
        )

        self.assertEqual(budget, 1)

    def test_scale_up_first_service_request_budget_does_not_expand_beyond_initial_admission(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._scale_up_initial_admission_request_budget = lambda: 1
        runner._runtime_forward_capacity_limit = lambda: 2
        runner._live_scale_eval_period_s = lambda: 5.0
        runner._scale_up_trace_total_busy_estimate_ms = lambda trace: 9000.0

        budget = runner._scale_up_first_service_request_budget(
            [
                SimpleNamespace(adapter_id="first"),
                SimpleNamespace(adapter_id="second"),
                SimpleNamespace(adapter_id="third"),
            ]
        )

        self.assertEqual(budget, 1)

    def test_scale_up_first_service_request_budget_does_not_expand_from_parallelism_only(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.coord_cfg = {"max_concurrent_loads": 2}
        runner._scale_up_initial_admission_request_budget = lambda: 1

        budget = runner._scale_up_first_service_request_budget(
            [
                SimpleNamespace(adapter_id="first"),
                SimpleNamespace(adapter_id="second"),
                SimpleNamespace(adapter_id="third"),
            ],
            configured_max_concurrent_loads=2,
        )

        self.assertEqual(budget, 1)

    def test_predicted_scale_up_load_ms_respects_max_concurrent_loads(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        metadata = {
            "a": SimpleNamespace(predicted_load_time_ms=100.0),
            "b": SimpleNamespace(predicted_load_time_ms=80.0),
            "c": SimpleNamespace(predicted_load_time_ms=40.0),
        }
        runner._stack = SimpleNamespace(
            _artifact_metadata=lambda aid: metadata[aid],
            _artifact_size_mb=lambda aid: 1.0,
            _host_paths={},
            _nvme_paths={},
        )
        runner.coordinator = SimpleNamespace(
            compute_faaslora_host_load_ms=lambda _size_mb: 0.0,
            compute_faaslora_nvme_load_ms=lambda _size_mb: 0.0,
            compute_cold_start_load_ms=lambda _size_mb, _bw: 0.0,
        )
        runner.cost_model = {}

        predicted_ms = runner._predicted_scale_up_load_ms(
            ["a", "b", "c"],
            max_concurrent_loads=2,
        )

        self.assertEqual(predicted_ms, 120.0)

    def test_predicted_scale_up_load_ms_respects_source_tier_busy_floor(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._stack = SimpleNamespace(
            _artifact_metadata=lambda _aid: SimpleNamespace(predicted_load_time_ms=22.0),
            _artifact_size_mb=lambda _aid: 30.0,
            _host_paths={},
            _nvme_paths={"a": "/tmp/a"},
        )
        runner.coordinator = SimpleNamespace(
            compute_faaslora_host_load_ms=lambda _size_mb: 40.0,
            compute_faaslora_nvme_load_ms=lambda _size_mb: 120.0,
            compute_cold_start_load_ms=lambda _size_mb, _bw: 500.0,
        )
        runner.cost_model = {"bandwidth_mbps": 100.0}

        predicted_ms = runner._predicted_scale_up_load_ms(["a"])

        self.assertEqual(predicted_ms, 120.0)

    def test_live_scaleup_preferred_adapters_store_exact_handoff_plan(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._predict_scale_up_handoff_plan = lambda **kwargs: {
            "mode": "dynamic_headroom_exact_handoff_prefix",
            "planned_adapters": ["waiting_a", "future_b"],
            "ordered_handoff_adapters": ["waiting_a", "future_b", "future_c"],
            "exact_prefix_bytes": 700 * 1024 * 1024,
        }

        preferred = runner._live_scale_up_preferred_gpu_adapters(
            replay_t0=0.0,
            arrived_request_count=4,
            queue_visible_request_count=2,
            visible_traces=[],
        )
        frontier = runner._live_scale_up_arrived_demand_frontier(
            replay_t0=0.0,
            arrived_request_count=4,
            queue_visible_request_count=2,
            visible_traces=[],
        )

        self.assertEqual(preferred, ["waiting_a", "future_b"])
        self.assertEqual(frontier, ["waiting_a", "future_b", "future_c"])
        self.assertEqual(
            runner._last_scale_up_handoff_plan["mode"],
            "dynamic_headroom_exact_handoff_prefix",
        )

    def test_begin_scaleup_runtime_request_labels_tracks_first_service_and_plan_match(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._scaleup_runtime_instance_ids = {"inst_2"}
        runner._scaleup_runtime_lora_request_ordinals = {}
        runner._scaleup_runtime_handoff_plan = lambda _instance_id: {
            "planned_adapters": ["adapter_a", "adapter_b"],
            "first_service_request_count": 2,
        }

        first = runner._begin_scaleup_runtime_request_labels(
            slot=SimpleNamespace(instance_id="inst_2"),
            adapter_id="adapter_a",
            cache_tier="gpu",
        )
        second = runner._begin_scaleup_runtime_request_labels(
            slot=SimpleNamespace(instance_id="inst_2"),
            adapter_id="adapter_x",
            cache_tier="host",
        )
        third = runner._begin_scaleup_runtime_request_labels(
            slot=SimpleNamespace(instance_id="inst_2"),
            adapter_id="adapter_b",
            cache_tier="gpu",
        )

        self.assertTrue(first["on_scaleup_runtime"])
        self.assertFalse(first["scaleup_affected"])
        self.assertTrue(first["scaleup_first_service"])
        self.assertTrue(first["scaleup_planned_adapter_match"])

        self.assertTrue(second["on_scaleup_runtime"])
        self.assertTrue(second["scaleup_affected"])
        self.assertTrue(second["scaleup_first_service"])
        self.assertFalse(second["scaleup_planned_adapter_match"])

        self.assertTrue(third["on_scaleup_runtime"])
        self.assertFalse(third["scaleup_first_service"])
        self.assertTrue(third["scaleup_planned_adapter_match"])

    def test_scale_up_preload_capacity_uses_exact_handoff_prefix_budget(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.preload_cfg = {
            "scale_up_dynamic_budget_enabled": True,
            "scale_up_preload_mb": 1024,
        }
        runner.hw = {
            "gpu_budget_mb": 4096,
            "model_weights_mb": 1024,
        }
        runner.coord_cfg = {"lora_load_reserve_ratio": 0.25}
        runner.model_cfg = {"tensor_parallel_size": 1, "visible_device_ids": [0], "device_id": 0}
        runner.engine = SimpleNamespace(device_id=0)
        runner.coordinator = SimpleNamespace(_predicted_kv_growth_mb=lambda: 256.0)
        runner.adapter_info = {
            "hot_a": {"size_mb": 400.0},
            "hot_b": {"size_mb": 500.0},
        }
        runner._stack = SimpleNamespace(
            _host_paths={"hot_a": "/tmp/hot_a"},
            _nvme_paths={"hot_b": "/tmp/hot_b"},
            _artifact_size_mb=lambda aid: {"hot_a": 400.0, "hot_b": 500.0}[aid],
            sync_local_tier_paths=lambda: None,
        )
        runner._last_scale_up_handoff_plan = {
            "mode": "dynamic_headroom_exact_handoff_prefix",
            "planned_adapters": ["hot_a", "hot_b"],
            "exact_prefix_bytes": 900 * 1024 * 1024,
            "queue_at_ready_request_count": 5,
            "queue_at_ready_adapter_count": 2,
            "projected_arrived_request_count": 7,
            "incumbent_started_request_count": 1,
            "bootstrap_latency_ms": 150.0,
            "plan_load_latency_ms": 320.0,
            "ready_delay_ms": 470.0,
            "configured_max_concurrent_loads": 2,
        }

        capacity_bytes = runner._scale_up_preload_capacity_bytes(["hot_a", "hot_b"])

        self.assertEqual(capacity_bytes, 900 * 1024 * 1024)
        self.assertEqual(
            runner._last_scale_up_preload_budget["mode"],
            "dynamic_headroom_exact_handoff_prefix",
        )
        self.assertEqual(
            runner._last_scale_up_preload_budget["exact_prefix_bytes"],
            900 * 1024 * 1024,
        )
        self.assertEqual(runner._last_scale_up_preload_budget["queue_at_ready_request_count"], 5)

    def test_scale_up_preload_capacity_falls_back_to_preferred_prefix_not_live_hotset(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.preload_cfg = {
            "scale_up_dynamic_budget_enabled": True,
            "scale_up_preload_mb": 1024,
        }
        runner.hw = {}
        runner.coord_cfg = {}
        runner.model_cfg = {"tensor_parallel_size": 1, "visible_device_ids": [0], "device_id": 0}
        runner.engine = SimpleNamespace(device_id=0)
        runner.coordinator = SimpleNamespace(_predicted_kv_growth_mb=lambda: 0.0)
        runner.adapter_info = {
            "hot_a": {"size_mb": 300.0},
            "recent_b": {"size_mb": 500.0},
        }
        runner._stack = SimpleNamespace(
            _host_paths={"hot_a": "/tmp/hot_a", "recent_b": "/tmp/recent_b"},
            _nvme_paths={},
            _artifact_size_mb=lambda aid: {"hot_a": 300.0, "recent_b": 500.0}[aid],
            sync_local_tier_paths=lambda: None,
        )
        runner._last_scale_up_handoff_plan = {}
        runner._scale_up_target_runtime_headroom_mb = lambda: 600.0

        capacity_bytes = runner._scale_up_preload_capacity_bytes(["hot_a", "recent_b"])

        self.assertEqual(capacity_bytes, 300 * 1024 * 1024)
        self.assertEqual(
            runner._last_scale_up_preload_budget["mode"],
            "dynamic_headroom_preferred_prefix",
        )
        self.assertEqual(
            runner._last_scale_up_preload_budget["exact_prefix_bytes"],
            300 * 1024 * 1024,
        )
        self.assertEqual(
            runner._last_scale_up_preload_budget["live_hotset_bytes"],
            800 * 1024 * 1024,
        )
        self.assertEqual(runner._last_scale_up_preload_budget["recent_working_set_bytes"], 0)

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

    def test_residency_manager_sync_gpu_capacity_defaults_to_first_visible_device_without_tracked_ids(self) -> None:
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
            device_count=2,
            get_all_devices_memory_info=lambda: infos,
            get_current_memory_info=lambda device_id=0: infos.get(device_id),
        )
        manager = ResidencyManager(DummyConfig(), SimpleNamespace(), monitor)

        manager._sync_gpu_capacity_once()

        gpu_capacity = manager.tier_capacities[StorageTier.GPU]
        self.assertEqual(gpu_capacity.total_bytes, 10)
        self.assertEqual(gpu_capacity.used_bytes, 4)

    def test_residency_manager_sync_gpu_capacity_aggregates_explicit_configured_device_ids(self) -> None:
        class DummyConfig:
            def __init__(self) -> None:
                self._data = {
                    "memory": {
                        "gpu": {"total_memory_gb": 24, "device_ids": [0, 1]},
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
        manager = ResidencyManager(
            DummyConfig(),
            SimpleNamespace(),
            SimpleNamespace(
                enabled=True,
                devices=[0, 1],
                device_count=2,
                get_all_devices_memory_info=lambda: infos,
                get_current_memory_info=lambda device_id=0: infos.get(device_id),
            ),
        )

        manager._sync_gpu_capacity_once()

        gpu_capacity = manager.tier_capacities[StorageTier.GPU]
        self.assertEqual(gpu_capacity.total_bytes, 22)
        self.assertEqual(gpu_capacity.used_bytes, 9)

    def test_memory_coordinator_initial_budget_aggregates_active_devices(self) -> None:
        class DummyConfig:
            def __init__(self) -> None:
                self._data = {
                    "memory.coordinator": {
                        "inference_memory_ratio": 0.7,
                        "artifact_memory_ratio": 0.2,
                        "safety_margin_ratio": 0.1,
                    }
                }

            def get(self, key, default=None):
                return self._data.get(key, default)

        infos = {
            0: GPUMemoryInfo(device_id=0, timestamp=0.0, total_bytes=10, used_bytes=4, free_bytes=6, active_bytes=1, cached_bytes=2),
            1: GPUMemoryInfo(device_id=1, timestamp=0.0, total_bytes=12, used_bytes=5, free_bytes=7, active_bytes=2, cached_bytes=3),
        }
        coordinator = MemoryCoordinator(
            config=DummyConfig(),
            gpu_monitor=SimpleNamespace(
                enabled=True,
                devices=[0, 1],
                device_count=2,
                get_all_devices_memory_info=lambda: infos,
                get_current_memory_info=lambda device_id=0: infos.get(device_id),
            ),
            residency_manager=SimpleNamespace(_gpu_device_ids_for_accounting=lambda: [0, 1]),
            registry=SimpleNamespace(),
        )

        asyncio.run(coordinator._initialize_memory_budget())

        self.assertIsNotNone(coordinator.memory_budget)
        self.assertEqual(coordinator.memory_budget.total_bytes, 22)

    def test_memory_coordinator_defaults_to_first_visible_device_without_explicit_scope(self) -> None:
        class DummyConfig:
            def __init__(self) -> None:
                self._data = {
                    "memory.coordinator": {
                        "inference_memory_ratio": 0.7,
                        "artifact_memory_ratio": 0.2,
                        "safety_margin_ratio": 0.1,
                    }
                }

            def get(self, key, default=None):
                return self._data.get(key, default)

        infos = {
            0: GPUMemoryInfo(device_id=0, timestamp=0.0, total_bytes=10, used_bytes=4, free_bytes=6, active_bytes=1, cached_bytes=2),
            1: GPUMemoryInfo(device_id=1, timestamp=0.0, total_bytes=12, used_bytes=5, free_bytes=7, active_bytes=2, cached_bytes=3),
        }
        coordinator = MemoryCoordinator(
            config=DummyConfig(),
            gpu_monitor=SimpleNamespace(
                enabled=True,
                devices=[0, 1],
                device_count=2,
                get_all_devices_memory_info=lambda: infos,
                get_current_memory_info=lambda device_id=0: infos.get(device_id),
            ),
            residency_manager=SimpleNamespace(),
            registry=SimpleNamespace(),
        )

        asyncio.run(coordinator._initialize_memory_budget())

        self.assertIsNotNone(coordinator.memory_budget)
        self.assertEqual(coordinator.memory_budget.total_bytes, 10)

    def test_vllm_wrapper_memory_stats_aggregate_visible_devices(self) -> None:
        cfg = Config(str(DEFAULT_CONFIG))
        cfg.set("serving.vllm.visible_device_ids", [0, 1])
        infos = {
            0: GPUMemoryInfo(device_id=0, timestamp=0.0, total_bytes=10, used_bytes=4, free_bytes=6, active_bytes=1, cached_bytes=2),
            1: GPUMemoryInfo(device_id=1, timestamp=0.0, total_bytes=12, used_bytes=5, free_bytes=7, active_bytes=2, cached_bytes=3),
        }
        wrapper = VLLMWrapper(
            cfg,
            registry=SimpleNamespace(),
            gpu_monitor=SimpleNamespace(
                enabled=True,
                devices=[0, 1],
                get_all_devices_memory_info=lambda: infos,
                get_current_memory_info=lambda device_id=0: infos.get(device_id),
            ),
        )

        stats = wrapper.get_memory_stats()

        self.assertEqual(stats["gpu_memory_total_bytes"], 22)
        self.assertEqual(stats["gpu_memory_used_bytes"], 9)
        self.assertEqual(stats["gpu_memory_free_bytes"], 13)
        self.assertAlmostEqual(stats["gpu_memory_utilization"], 9 / 22)

    def test_vllm_wrapper_memory_stats_prefer_local_visible_devices_over_global_config(self) -> None:
        cfg = Config(str(DEFAULT_CONFIG))
        cfg.set("serving.vllm.visible_device_ids", [0, 1])
        infos = {
            0: GPUMemoryInfo(device_id=0, timestamp=0.0, total_bytes=10, used_bytes=4, free_bytes=6, active_bytes=1, cached_bytes=2),
            1: GPUMemoryInfo(device_id=1, timestamp=0.0, total_bytes=12, used_bytes=5, free_bytes=7, active_bytes=2, cached_bytes=3),
        }
        wrapper = VLLMWrapper(
            cfg,
            registry=SimpleNamespace(),
            gpu_monitor=SimpleNamespace(
                enabled=True,
                devices=[0],
                get_all_devices_memory_info=lambda: infos,
                get_current_memory_info=lambda device_id=0: infos.get(device_id),
            ),
        )

        stats = wrapper.get_memory_stats()

        self.assertEqual(stats["gpu_memory_total_bytes"], 10)
        self.assertEqual(stats["gpu_memory_used_bytes"], 4)
        self.assertEqual(stats["gpu_memory_free_bytes"], 6)
        self.assertAlmostEqual(stats["gpu_memory_utilization"], 4 / 10)

    def test_resource_coordinator_scales_budget_with_runtime_device_count(self) -> None:
        coord = ResourceCoordinator(
            config={
                "gpu_budget_mb": 24576,
                "gpu_device_ids": [0, 1],
                "model_weights_mb": 14336,
                "kv_per_1k_tokens_mb": 2.0,
            },
            coordination_enabled=True,
        )

        self.assertEqual(coord.gpu_device_count, 2)
        self.assertEqual(coord.gpu_budget_mb, 49152)

    def test_stack_resolve_lora_dedicated_coord_ignores_shared_gpu_tier(self) -> None:
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
                hardware_cfg={"gpu_budget_mb": 24000, "gpu_device_ids": [0]},
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
            admitted = asyncio.run(stack.residency_manager.admit_artifact(adapter_id, StorageTier.NVME))
            self.assertTrue(admitted)
            admitted_gpu = asyncio.run(stack.residency_manager.admit_artifact(adapter_id, StorageTier.GPU, force=True))
            self.assertTrue(admitted_gpu)
            stack.sync_local_tier_paths()

            class DedicatedCoord:
                _residency_manager = None

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
                    coordinator=DedicatedCoord(),
                )
                self.assertEqual(tier, "nvme")

            asyncio.run(run_case())

    def test_scenario_result_aggregate_computes_ttft_decomposition(self) -> None:
        result = ScenarioResult("demo", "faaslora_full", total=4)
        result.ttft_slo_ms = 180.0
        result.requests = [
            RequestResult("r1", "a", False, "normal", True, "gpu", 10.0, 80.0, 100.0, 5.0, 5.0, 20.0, 120.0, 10, 5, 0.10, True, "inst_1", False),
            RequestResult("r2", "b", False, "normal", True, "host", 60.0, 120.0, 200.0, 10.0, 10.0, 25.0, 240.0, 10, 5, 0.20, True, "inst_1", False),
            RequestResult(
                "r3", "c", False, "normal", False, "remote",
                0.0, 140.0, 150.0, 5.0, 5.0, 30.0, 190.0, 10, 5, 0.30, True,
                "inst_2", True, None, True, True, False,
            ),
            RequestResult(
                "r4", "d", False, "normal", True, "gpu",
                20.0, 160.0, 220.0, 10.0, 10.0, 0.0, 260.0, 10, 5, 0.40, True,
                "inst_2", False, None, True, False, True,
            ),
        ]
        result.scale_up_events = [{
            "request_index": 2,
            "instance_id": "inst_2",
            "runtime_kind": "dedicated",
            "cold_start_latency_ms": 320.0,
        }]

        result.aggregate(4.0)

        self.assertAlmostEqual(result.avg_comparable_ttft_ms, (100.0 + 200.0 + 220.0) / 3.0)
        self.assertAlmostEqual(result.avg_serverless_overhead_ms, (20.0 + 80.0 + 10.0 + 40.0) / 4.0)
        self.assertAlmostEqual(result.avg_runtime_ttft_ms, (80.0 + 120.0 + 140.0 + 160.0) / 4.0)
        self.assertAlmostEqual(result.avg_gpu_ready_ttft_ms, (100.0 + 220.0) / 2.0)
        self.assertAlmostEqual(result.avg_warm_standard_ttft_ms, (100.0 + 220.0) / 2.0)
        self.assertAlmostEqual(result.p95_warm_standard_ttft_ms, 220.0)
        self.assertAlmostEqual(result.p99_warm_standard_ttft_ms, 220.0)
        self.assertAlmostEqual(result.avg_scaleup_affected_ttft_ms, 150.0)
        self.assertAlmostEqual(result.avg_scaleup_runtime_ttft_ms, (150.0 + 220.0) / 2.0)
        self.assertEqual(result.scaleup_runtime_lora_request_count, 2)
        self.assertAlmostEqual(result.scaleup_runtime_gpu_hit_rate, 0.5)
        self.assertAlmostEqual(result.avg_scaleup_first_service_ttft_ms, 150.0)
        self.assertEqual(result.scaleup_first_service_request_count, 1)
        self.assertAlmostEqual(result.scaleup_first_service_gpu_hit_rate, 0.0)
        self.assertAlmostEqual(result.scaleup_first_service_planned_match_rate, 0.0)
        self.assertAlmostEqual(result.avg_cold_start_latency_ms, 320.0)
        self.assertAlmostEqual(result.p95_cold_start_latency_ms, 320.0)
        self.assertAlmostEqual(result.throughput_tok_per_s, 5.0)
        self.assertAlmostEqual(result.slo_attainment, 0.5)
        self.assertAlmostEqual(result.slo_goodput_rps, 0.5)
        self.assertAlmostEqual(result.slo_goodput_tok_per_s, 2.5)
        self.assertAlmostEqual(result.avg_tpot_ms, 25.0)
        self.assertAlmostEqual(result.tpot_observed_request_ratio, 0.75)
        self.assertAlmostEqual(result.qpr, 5.0 / (0.25 * 0.1675))
        self.assertAlmostEqual(result.qpr_rps_legacy, 1.0 / (0.25 * 0.1675))
        self.assertAlmostEqual(result.cost_effectiveness_e2e, 1.0 / (0.2025 * 0.25))
        self.assertEqual(result.cold_starts_after_scale_up, [1])

    def test_live_result_stats_includes_ttft_decomposition(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._live_scale_up_events = [{"request_index": 2}]
        runner._ttft_slo_ms = 180.0
        results = [
            RequestResult("r1", "a", False, "normal", True, "gpu", 10.0, 80.0, 100.0, 5.0, 5.0, 20.0, 120.0, 10, 5, 0.10, True, "inst_1", False),
            RequestResult("r2", "b", False, "normal", True, "host", 60.0, 120.0, 200.0, 10.0, 10.0, 25.0, 240.0, 10, 5, 0.20, True, "inst_1", False),
            RequestResult(
                "r3", "c", False, "normal", False, "remote",
                0.0, 140.0, 150.0, 5.0, 5.0, 30.0, 190.0, 10, 5, 0.30, True,
                "inst_2", True, None, True, True, False,
            ),
            RequestResult(
                "r4", "d", False, "normal", True, "gpu",
                20.0, 160.0, 220.0, 10.0, 10.0, 0.0, 260.0, 10, 5, 0.40, True,
                "inst_2", False, None, True, False, True,
            ),
        ]

        stats = runner._live_result_stats(results)

        self.assertAlmostEqual(stats["avg_comparable_ttft_ms"], (100.0 + 200.0 + 220.0) / 3.0)
        self.assertAlmostEqual(stats["avg_serverless_overhead_ms"], (20.0 + 80.0 + 10.0 + 40.0) / 4.0)
        self.assertAlmostEqual(stats["avg_runtime_ttft_ms"], (80.0 + 120.0 + 140.0 + 160.0) / 4.0)
        self.assertAlmostEqual(stats["avg_gpu_ready_ttft_ms"], (100.0 + 220.0) / 2.0)
        self.assertAlmostEqual(stats["avg_warm_standard_ttft_ms"], (100.0 + 220.0) / 2.0)
        self.assertAlmostEqual(stats["p95_warm_standard_ttft_ms"], 220.0)
        self.assertAlmostEqual(stats["p99_warm_standard_ttft_ms"], 220.0)
        self.assertAlmostEqual(stats["avg_scaleup_affected_ttft_ms"], 150.0)
        self.assertAlmostEqual(stats["avg_scaleup_runtime_ttft_ms"], (150.0 + 220.0) / 2.0)
        self.assertEqual(stats["scaleup_runtime_lora_request_count"], 2)
        self.assertAlmostEqual(stats["scaleup_runtime_gpu_hit_rate"], 0.5)
        self.assertAlmostEqual(stats["avg_scaleup_first_service_ttft_ms"], 150.0)
        self.assertEqual(stats["scaleup_first_service_request_count"], 1)
        self.assertAlmostEqual(stats["scaleup_first_service_gpu_hit_rate"], 0.0)
        self.assertAlmostEqual(stats["scaleup_first_service_planned_match_rate"], 0.0)
        self.assertAlmostEqual(stats["tpot_observed_request_ratio"], 0.75)
        self.assertEqual(stats["total_output_tokens"], 20)
        self.assertAlmostEqual(stats["slo_attainment"], 0.5)
        self.assertAlmostEqual(stats["cost_effectiveness_e2e"], 1.0 / (0.2025 * 0.25))

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
        self.assertEqual(stats["avg_warm_standard_ttft_ms"], 0.0)
        self.assertEqual(stats["p95_warm_standard_ttft_ms"], 0.0)
        self.assertEqual(stats["p99_warm_standard_ttft_ms"], 0.0)

    def test_metric_exports_include_layered_standard_and_scaling_metrics(self) -> None:
        result = ScenarioResult("demo", "faaslora_full", total=4)
        result.ttft_slo_ms = 180.0
        result.requests = [
            RequestResult("r1", "a", False, "normal", True, "gpu", 10.0, 80.0, 100.0, 5.0, 5.0, 20.0, 120.0, 10, 5, 0.10, True, "inst_1", False),
            RequestResult("r2", "b", False, "normal", True, "host", 60.0, 120.0, 200.0, 10.0, 10.0, 25.0, 240.0, 10, 5, 0.20, True, "inst_1", False),
            RequestResult(
                "r3", "c", False, "normal", False, "remote",
                0.0, 140.0, 150.0, 5.0, 5.0, 30.0, 190.0, 10, 5, 0.30, True,
                "inst_2", True, None, True, True, False,
            ),
            RequestResult(
                "r4", "d", False, "normal", True, "gpu",
                20.0, 160.0, 220.0, 10.0, 10.0, 0.0, 260.0, 10, 5, 0.40, True,
                "inst_2", False, None, True, False, True,
            ),
        ]
        result.scale_up_events = [{"request_index": 2, "instance_id": "inst_2"}]
        result.aggregate(4.0)

        comparison_row = _build_comparison_table([result])[0]
        self.assertIn("standard_serving_metrics", comparison_row)
        self.assertIn("serverless_deployment_metrics", comparison_row)
        self.assertIn("scaling_metrics", comparison_row)
        self.assertIn("mechanism_metrics", comparison_row)
        self.assertAlmostEqual(comparison_row["TTFT_warm_standard_avg_ms"], 160.0)
        self.assertAlmostEqual(comparison_row["SLO_goodput_RPS"], 0.5)
        self.assertAlmostEqual(
            comparison_row["standard_serving_metrics"]["TTFT_warm_standard_avg_ms"], 160.0
        )
        self.assertAlmostEqual(
            comparison_row["scaling_metrics"]["SLO_goodput_RPS"], 0.5
        )
        self.assertAlmostEqual(comparison_row["TTFT_scaleup_runtime_avg_ms"], 185.0)
        self.assertAlmostEqual(comparison_row["TTFT_scaleup_first_service_avg_ms"], 150.0)
        self.assertAlmostEqual(comparison_row["TPOT_observed_ratio"], 0.75)
        self.assertEqual(comparison_row["ScaleUp_first_service_requests"], 1)
        self.assertAlmostEqual(
            comparison_row["standard_serving_metrics"]["Cost_effectiveness_e2e"],
            round(1.0 / (0.2025 * 0.25), 4),
        )

        summaries = _build_scenario_summaries(
            [result],
            {
                "backend": "vllm",
                "instance_mode": "shared",
                "routing_policy": "adapter_affinity",
                "num_adapters": 4,
                "active_adapter_cap": None,
                "hotset_rotation_requests": 0,
                "min_instances": 1,
                "max_instances": 1,
                "arrival_window_s": 5.0,
                "scale_eval_interval_s": 15.0,
                "scenario_coordination": {
                    "demo": {
                        "instance_mode": "auto",
                        "routing_policy": "least_connections",
                        "min_instances": 2,
                        "max_instances": 4,
                        "arrival_window_s": 7.5,
                        "scale_eval_interval_s": 12.0,
                    }
                },
            },
        )
        summary = summaries["demo"]
        self.assertIn("standard_serving_metrics", summary)
        self.assertIn("serverless_deployment_metrics", summary)
        self.assertIn("scaling_metrics", summary)
        self.assertIn("mechanism_metrics", summary)
        self.assertEqual(summary["instance_mode"], "auto")
        self.assertEqual(summary["routing_policy"], "least_connections")
        self.assertEqual(summary["min_instances"], 2)
        self.assertEqual(summary["max_instances"], 4)
        self.assertEqual(summary["arrival_window_s"], 7.5)
        self.assertEqual(summary["scale_eval_interval_s"], 12.0)
        self.assertAlmostEqual(summary["avg_warm_standard_ttft_ms"], 160.0)
        self.assertAlmostEqual(summary["slo_goodput_rps"], 0.5)
        self.assertAlmostEqual(summary["scaling_metrics"]["SLO_goodput_RPS"], 0.5)
        self.assertAlmostEqual(summary["avg_scaleup_runtime_ttft_ms"], 185.0)
        self.assertAlmostEqual(summary["avg_scaleup_first_service_ttft_ms"], 150.0)
        self.assertAlmostEqual(summary["tpot_observed_request_ratio"], 0.75)

    def test_router_prefers_lower_observed_backbone_runtime_cost(self) -> None:
        fast = InstanceSlot("inst_fast", None, None)
        slow = InstanceSlot("inst_slow", None, None)
        slow.record_runtime_ttft(4500.0, is_backbone=True)
        fast.record_runtime_ttft(120.0, is_backbone=True)
        pool = SimpleNamespace(get_slots=lambda: [slow, fast])
        router = Router(pool, policy="least_connections")

        selected = router.select_instance(None)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.instance_id, "inst_fast")

    def test_predicted_lora_io_uses_exact_bucket_before_fallback(self) -> None:
        slot = InstanceSlot("inst_a", None, None)
        slot.mark_adapter_tier("adapter_a", "host")
        slot.record_request_cost(
            adapter_id="adapter_a",
            cache_tier="host",
            lora_io_ms=240.0,
            runtime_ttft_ms=900.0,
        )

        predicted = slot.predicted_lora_io_ms(
            adapter_id="adapter_a",
            fallback_lora_io_ms=1200.0,
        )

        self.assertEqual(predicted, 240.0)

    def test_backbone_routing_does_not_inherit_lora_runtime_bias(self) -> None:
        backbone_only = InstanceSlot("inst_backbone", None, None)
        lora_only = InstanceSlot("inst_lora", None, None)
        backbone_only.record_runtime_ttft(3600.0, is_backbone=True)
        lora_only.record_request_cost(
            adapter_id="adapter_a",
            cache_tier="gpu",
            lora_io_ms=0.0,
            runtime_ttft_ms=7800.0,
        )
        lora_only.active_requests = 1
        pool = SimpleNamespace(get_slots=lambda: [backbone_only, lora_only])
        router = Router(pool, policy="least_connections")

        selected = router.select_instance(None)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.instance_id, "inst_lora")

    def test_router_prefers_lower_observed_lora_total_cost(self) -> None:
        fast = InstanceSlot("inst_fast", None, None)
        slow = InstanceSlot("inst_slow", None, None)
        fast.mark_adapter_tier("adapter_a", "nvme")
        slow.mark_adapter_tier("adapter_a", "gpu")
        slow.record_request_cost(
            adapter_id="adapter_a",
            cache_tier="gpu",
            lora_io_ms=0.0,
            runtime_ttft_ms=5200.0,
        )
        fast.record_request_cost(
            adapter_id="adapter_a",
            cache_tier="nvme",
            lora_io_ms=300.0,
            runtime_ttft_ms=600.0,
        )
        pool = SimpleNamespace(get_slots=lambda: [slow, fast])
        router = Router(pool, policy="adapter_affinity")

        selected = router.select_instance("adapter_a", adapter_size_mb=30.0)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.instance_id, "inst_fast")

    def test_router_prefers_gpu_slot_when_observed_total_cost_is_lower(self) -> None:
        nvme = InstanceSlot("inst_nvme", None, None)
        local = InstanceSlot("inst_local", None, None)
        local.mark_adapter_tier("adapter_a", "gpu")
        nvme.mark_adapter_tier("adapter_a", "nvme")
        local.record_request_cost(
            adapter_id="adapter_a",
            cache_tier="gpu",
            lora_io_ms=0.0,
            runtime_ttft_ms=900.0,
        )
        nvme.record_request_cost(
            adapter_id="adapter_a",
            cache_tier="nvme",
            lora_io_ms=500.0,
            runtime_ttft_ms=700.0,
        )
        pool = SimpleNamespace(get_slots=lambda: [local, nvme])
        router = Router(pool, policy="adapter_affinity")

        selected = router.select_instance("adapter_a", adapter_size_mb=30.0)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.instance_id, "inst_local")

    def test_router_accounts_for_tail_service_occupancy(self) -> None:
        busy_gpu = InstanceSlot("inst_busy", None, None)
        idle_gpu = InstanceSlot("inst_idle", None, None)
        busy_gpu.mark_adapter_tier("adapter_a", "gpu")
        idle_gpu.mark_adapter_tier("adapter_a", "gpu")
        busy_gpu.active_requests = 1
        busy_gpu.record_request_cost(
            adapter_id="adapter_a",
            cache_tier="gpu",
            lora_io_ms=0.0,
            runtime_ttft_ms=7000.0,
            tail_service_ms=12000.0,
        )
        idle_gpu.record_request_cost(
            adapter_id="adapter_a",
            cache_tier="gpu",
            lora_io_ms=0.0,
            runtime_ttft_ms=7600.0,
            tail_service_ms=1500.0,
        )
        pool = SimpleNamespace(get_slots=lambda: [busy_gpu, idle_gpu])
        router = Router(pool, policy="adapter_affinity", runtime_concurrency_cap=2)

        selected = router.select_instance("adapter_a", adapter_size_mb=30.0)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.instance_id, "inst_idle")

    def test_router_accounts_for_idle_backbone_tail_cost(self) -> None:
        low_ttft_high_tail = InstanceSlot("inst_low_ttft_high_tail", None, None)
        balanced = InstanceSlot("inst_balanced", None, None)
        low_ttft_high_tail.record_request_cost(
            adapter_id=None,
            cache_tier="backbone",
            lora_io_ms=0.0,
            runtime_ttft_ms=120.0,
            tail_service_ms=9000.0,
        )
        balanced.record_request_cost(
            adapter_id=None,
            cache_tier="backbone",
            lora_io_ms=0.0,
            runtime_ttft_ms=400.0,
            tail_service_ms=1200.0,
        )
        pool = SimpleNamespace(get_slots=lambda: [low_ttft_high_tail, balanced])
        router = Router(pool, policy="least_connections")

        selected = router.select_instance(None)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.instance_id, "inst_balanced")

    def test_router_does_not_trade_gpu_hit_for_small_load_difference(self) -> None:
        host_slot = InstanceSlot("inst_host", None, None)
        gpu_slot = InstanceSlot("inst_gpu", None, None)
        host_slot.mark_adapter_tier("adapter_a", "host")
        gpu_slot.mark_adapter_tier("adapter_a", "gpu")
        host_slot.active_requests = 0
        gpu_slot.active_requests = 1
        host_slot.record_request_cost(
            adapter_id="adapter_a",
            cache_tier="host",
            lora_io_ms=9500.0,
            runtime_ttft_ms=16000.0,
        )
        gpu_slot.record_request_cost(
            adapter_id="adapter_a",
            cache_tier="gpu",
            lora_io_ms=0.0,
            runtime_ttft_ms=7200.0,
        )
        pool = SimpleNamespace(get_slots=lambda: [host_slot, gpu_slot])
        router = Router(pool, policy="adapter_affinity")

        selected = router.select_instance("adapter_a", adapter_size_mb=30.0)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.instance_id, "inst_gpu")

    def test_router_penalizes_saturated_hot_slot_with_large_live_queue_wait(self) -> None:
        cool_host = InstanceSlot("inst_host", None, None)
        hot_gpu = InstanceSlot("inst_gpu", None, None)
        cool_host.mark_adapter_tier("adapter_a", "host")
        hot_gpu.mark_adapter_tier("adapter_a", "gpu")
        cool_host.record_request_cost(
            adapter_id="adapter_a",
            cache_tier="host",
            lora_io_ms=3500.0,
            runtime_ttft_ms=2500.0,
            tail_service_ms=1500.0,
        )
        hot_gpu.record_request_cost(
            adapter_id="adapter_a",
            cache_tier="gpu",
            lora_io_ms=0.0,
            runtime_ttft_ms=3000.0,
            tail_service_ms=2000.0,
        )
        hot_gpu.active_requests = 2
        hot_gpu.record_inflight_request_estimate(
            "live_req_a",
            24000.0,
            now_monotonic=100.0,
        )
        hot_gpu.record_inflight_request_estimate(
            "live_req_b",
            26000.0,
            now_monotonic=100.0,
        )
        pool = SimpleNamespace(get_slots=lambda: [cool_host, hot_gpu])
        router = Router(pool, policy="adapter_affinity", runtime_concurrency_cap=2)

        with patch("faaslora.experiment.instance_pool.time.perf_counter", return_value=100.0):
            selected = router.select_instance("adapter_a", adapter_size_mb=30.0)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.instance_id, "inst_host")

    def test_router_does_not_override_lower_cost_runtime_for_planned_handoff_adapter(self) -> None:
        existing = InstanceSlot("inst_existing", None, None)
        scaleup = InstanceSlot("inst_scaleup", None, None)
        existing.mark_adapter_tier("adapter_a", "gpu")
        scaleup.mark_adapter_tier("adapter_a", "gpu")
        existing.record_request_cost(
            adapter_id="adapter_a",
            cache_tier="gpu",
            lora_io_ms=0.0,
            runtime_ttft_ms=3000.0,
        )
        scaleup.record_request_cost(
            adapter_id="adapter_a",
            cache_tier="gpu",
            lora_io_ms=0.0,
            runtime_ttft_ms=6000.0,
        )
        scaleup.scaleup_handoff_planned_adapters = ["adapter_a"]
        scaleup.scaleup_handoff_planned_adapter_ranks = {"adapter_a": 0}
        scaleup.scaleup_handoff_request_budget = 1
        scaleup.scaleup_handoff_assigned_requests = 0
        pool = SimpleNamespace(get_slots=lambda: [existing, scaleup])
        router = Router(pool, policy="adapter_affinity")

        selected = router.select_instance("adapter_a", adapter_size_mb=30.0)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.instance_id, "inst_existing")
        self.assertEqual(scaleup.scaleup_handoff_assigned_requests, 0)

    def test_router_uses_planned_handoff_as_tiebreak_when_costs_are_equal(self) -> None:
        existing = InstanceSlot("inst_existing", None, None)
        scaleup = InstanceSlot("inst_scaleup", None, None)
        existing.mark_adapter_tier("adapter_a", "gpu")
        scaleup.mark_adapter_tier("adapter_a", "gpu")
        existing.record_request_cost(
            adapter_id="adapter_a",
            cache_tier="gpu",
            lora_io_ms=0.0,
            runtime_ttft_ms=3000.0,
        )
        scaleup.record_request_cost(
            adapter_id="adapter_a",
            cache_tier="gpu",
            lora_io_ms=0.0,
            runtime_ttft_ms=3000.0,
        )
        scaleup.scaleup_handoff_planned_adapters = ["adapter_a"]
        scaleup.scaleup_handoff_planned_adapter_ranks = {"adapter_a": 0}
        scaleup.scaleup_handoff_request_budget = 1
        scaleup.scaleup_handoff_assigned_requests = 0
        pool = SimpleNamespace(get_slots=lambda: [existing, scaleup])
        router = Router(pool, policy="adapter_affinity")

        selected = router.select_instance("adapter_a", adapter_size_mb=30.0)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.instance_id, "inst_scaleup")
        self.assertEqual(scaleup.scaleup_handoff_assigned_requests, 1)

    def test_router_avoids_scaleup_runtime_for_unplanned_adapter_during_handoff(self) -> None:
        existing = InstanceSlot("inst_existing", None, None)
        scaleup = InstanceSlot("inst_scaleup", None, None)
        existing.mark_adapter_tier("adapter_b", "host")
        existing.record_request_cost(
            adapter_id="adapter_b",
            cache_tier="host",
            lora_io_ms=200.0,
            runtime_ttft_ms=1200.0,
        )
        scaleup.record_request_cost(
            adapter_id="adapter_b",
            cache_tier="remote",
            lora_io_ms=0.0,
            runtime_ttft_ms=900.0,
        )
        scaleup.scaleup_handoff_planned_adapters = ["adapter_a"]
        scaleup.scaleup_handoff_planned_adapter_ranks = {"adapter_a": 0}
        scaleup.scaleup_handoff_request_budget = 1
        scaleup.scaleup_handoff_assigned_requests = 0
        scaleup.active_requests = 1
        pool = SimpleNamespace(get_slots=lambda: [existing, scaleup])
        router = Router(pool, policy="adapter_affinity", runtime_concurrency_cap=2)

        selected = router.select_instance("adapter_b", adapter_size_mb=30.0)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.instance_id, "inst_existing")
        self.assertEqual(scaleup.scaleup_handoff_assigned_requests, 0)

    def test_router_allows_unplanned_adapter_on_spare_scaleup_lane(self) -> None:
        existing = InstanceSlot("inst_existing", None, None)
        scaleup = InstanceSlot("inst_scaleup", None, None)
        existing.mark_adapter_tier("adapter_b", "host")
        existing.record_request_cost(
            adapter_id="adapter_b",
            cache_tier="host",
            lora_io_ms=600.0,
            runtime_ttft_ms=1400.0,
        )
        scaleup.record_request_cost(
            adapter_id="adapter_b",
            cache_tier="remote",
            lora_io_ms=0.0,
            runtime_ttft_ms=100.0,
        )
        scaleup.scaleup_handoff_planned_adapters = ["adapter_a"]
        scaleup.scaleup_handoff_planned_adapter_ranks = {"adapter_a": 0}
        scaleup.scaleup_handoff_request_budget = 1
        scaleup.scaleup_handoff_assigned_requests = 0
        pool = SimpleNamespace(get_slots=lambda: [existing, scaleup])
        router = Router(pool, policy="adapter_affinity", runtime_concurrency_cap=2)

        selected = router.select_instance("adapter_b", adapter_size_mb=30.0)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.instance_id, "inst_scaleup")
        self.assertEqual(scaleup.scaleup_handoff_assigned_requests, 0)

    def test_router_avoids_scaleup_runtime_for_backbone_when_only_reserved_lane_remains(self) -> None:
        existing = InstanceSlot("inst_existing", None, None)
        scaleup = InstanceSlot("inst_scaleup", None, None)
        existing.record_request_cost(
            adapter_id=None,
            cache_tier="backbone",
            lora_io_ms=0.0,
            runtime_ttft_ms=500.0,
        )
        scaleup.record_request_cost(
            adapter_id=None,
            cache_tier="backbone",
            lora_io_ms=0.0,
            runtime_ttft_ms=100.0,
        )
        existing.mark_adapter_tier("adapter_a", "gpu")
        scaleup.mark_adapter_tier("adapter_a", "gpu")
        existing.record_request_cost(
            adapter_id="adapter_a",
            cache_tier="gpu",
            lora_io_ms=0.0,
            runtime_ttft_ms=1200.0,
        )
        scaleup.record_request_cost(
            adapter_id="adapter_a",
            cache_tier="gpu",
            lora_io_ms=0.0,
            runtime_ttft_ms=1200.0,
        )
        scaleup.scaleup_handoff_planned_adapters = ["adapter_a"]
        scaleup.scaleup_handoff_planned_adapter_ranks = {"adapter_a": 0}
        scaleup.scaleup_handoff_request_budget = 1
        scaleup.scaleup_handoff_assigned_requests = 0
        scaleup.active_requests = 1
        pool = SimpleNamespace(get_slots=lambda: [existing, scaleup])
        router = Router(pool, policy="least_connections", runtime_concurrency_cap=2)

        selected = router.select_instance(None)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.instance_id, "inst_existing")
        self.assertEqual(scaleup.scaleup_handoff_assigned_requests, 0)

    def test_router_backbone_request_can_use_spare_scaleup_lane_without_consuming_budget(self) -> None:
        existing = InstanceSlot("inst_existing", None, None)
        scaleup = InstanceSlot("inst_scaleup", None, None)
        existing.record_request_cost(
            adapter_id=None,
            cache_tier="backbone",
            lora_io_ms=0.0,
            runtime_ttft_ms=500.0,
        )
        scaleup.record_request_cost(
            adapter_id=None,
            cache_tier="backbone",
            lora_io_ms=0.0,
            runtime_ttft_ms=100.0,
        )
        existing.mark_adapter_tier("adapter_a", "gpu")
        scaleup.mark_adapter_tier("adapter_a", "gpu")
        existing.record_request_cost(
            adapter_id="adapter_a",
            cache_tier="gpu",
            lora_io_ms=0.0,
            runtime_ttft_ms=1200.0,
        )
        scaleup.record_request_cost(
            adapter_id="adapter_a",
            cache_tier="gpu",
            lora_io_ms=0.0,
            runtime_ttft_ms=1200.0,
        )
        scaleup.scaleup_handoff_planned_adapters = ["adapter_a"]
        scaleup.scaleup_handoff_planned_adapter_ranks = {"adapter_a": 0}
        scaleup.scaleup_handoff_request_budget = 1
        scaleup.scaleup_handoff_assigned_requests = 0
        pool = SimpleNamespace(get_slots=lambda: [existing, scaleup])
        router = Router(pool, policy="adapter_affinity", runtime_concurrency_cap=2)

        backbone_selected = router.select_instance(None)
        self.assertIsNotNone(backbone_selected)
        self.assertEqual(backbone_selected.instance_id, "inst_scaleup")
        self.assertEqual(scaleup.scaleup_handoff_assigned_requests, 0)

        lora_selected = router.select_instance("adapter_a", adapter_size_mb=30.0)
        self.assertIsNotNone(lora_selected)
        self.assertEqual(lora_selected.instance_id, "inst_scaleup")
        self.assertEqual(scaleup.scaleup_handoff_assigned_requests, 1)

    def test_comparable_request_requires_local_tier_and_non_scaleup(self) -> None:
        local_hit = RequestResult("r1", "a", False, "normal", True, "host", 40.0, 140.0, 210.0, 10.0, 10.0, 25.0, 250.0, 10, 5, 0.0, True, "inst_1", False)
        scaleup_local = RequestResult("r2", "b", False, "normal", True, "host", 30.0, 120.0, 180.0, 5.0, 5.0, 20.0, 220.0, 10, 5, 0.0, True, "inst_2", True)
        remote_fetch = RequestResult("r3", "c", False, "normal", False, "remote", 30.0, 120.0, 180.0, 5.0, 5.0, 20.0, 220.0, 10, 5, 0.0, True, "inst_1", False)
        backbone = RequestResult("r4", None, False, "normal", False, "backbone", 0.0, 90.0, 90.0, 0.0, 0.0, 20.0, 140.0, 10, 5, 0.0, True, "inst_1", False)
        legacy_scaleup = SimpleNamespace(success=True, cache_tier="host")

        self.assertTrue(_is_comparable_request(local_hit))
        self.assertFalse(_is_comparable_request(scaleup_local))
        self.assertFalse(_is_comparable_request(remote_fetch))
        self.assertFalse(_is_comparable_request(backbone))
        self.assertFalse(_is_comparable_request(legacy_scaleup, 3, {3}))

    def test_aggregate_hit_rates_ignore_backbone_only_requests(self) -> None:
        result = ScenarioResult("demo", "faaslora_full", total=3)
        result.requests = [
            RequestResult("r1", None, False, "normal", False, "backbone", 0.0, 90.0, 90.0, 0.0, 0.0, 15.0, 120.0, 10, 5, 0.0, True, "inst_1", False),
            RequestResult("r2", "a", False, "normal", True, "gpu", 0.0, 100.0, 100.0, 0.0, 0.0, 15.0, 130.0, 10, 5, 0.0, True, "inst_1", False),
            RequestResult("r3", "b", False, "normal", False, "remote", 50.0, 150.0, 200.0, 0.0, 0.0, 15.0, 240.0, 10, 5, 0.0, True, "inst_2", False),
        ]

        result.aggregate(6.0)

        self.assertAlmostEqual(result.cache_hit_rate, 0.5)
        self.assertAlmostEqual(result.gpu_hit_rate, 0.5)

    def test_live_result_stats_hit_ratio_ignores_backbone_only_requests(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._live_scale_up_events = []
        runner._ttft_slo_ms = 5000.0
        results = [
            RequestResult("r1", None, False, "normal", False, "backbone", 0.0, 90.0, 90.0, 0.0, 0.0, 15.0, 120.0, 10, 5, 0.0, True, "inst_1", False),
            RequestResult("r2", "a", False, "normal", True, "gpu", 0.0, 100.0, 100.0, 0.0, 0.0, 15.0, 130.0, 10, 5, 0.0, True, "inst_1", False),
            RequestResult("r3", "b", False, "normal", False, "remote", 50.0, 150.0, 200.0, 0.0, 0.0, 15.0, 240.0, 10, 5, 0.0, True, "inst_2", False),
        ]

        stats = runner._live_result_stats(results)

        self.assertAlmostEqual(stats["cache_hit_ratio"], 0.5)

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

    def test_register_scale_up_event_preserves_cold_start_fields(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.instance_pool = SimpleNamespace(count=lambda: 2)
        runner._live_scale_up_events = []
        runner._scaleup_runtime_instance_ids = set()
        runner._last_scale_up_handoff_plan = {
            "planned_adapters": ["finance_lora"],
            "ordered_handoff_adapters": ["finance_lora", "medical_lora"],
            "queue_at_ready_request_count": 12,
            "queue_at_ready_adapter_count": 2,
            "first_service_request_count": 1,
            "first_service_adapter_count": 1,
            "initial_admission_request_budget": 2,
            "effective_initial_warmup_parallelism": 2,
            "projected_arrived_request_count": 40,
            "incumbent_started_request_count": 28,
            "configured_max_concurrent_loads": 1,
            "bootstrap_latency_ms": 50000.0,
            "plan_load_latency_ms": 120.0,
            "ready_delay_ms": 50120.0,
        }
        runner._last_scale_up_preload_budget = {
            "mode": "dynamic_headroom_exact_handoff_prefix",
            "capacity_bytes": 32 * 1024 * 1024,
            "exact_prefix_bytes": 32 * 1024 * 1024,
            "live_hotset_bytes": 64 * 1024 * 1024,
            "recent_working_set_bytes": 0,
            "target_headroom_bytes": 256 * 1024 * 1024,
        }
        result = ScenarioResult("demo", "faaslora_full", total=0)
        decision = SimpleNamespace(reason="scale", target_instances=2)

        runner._register_scale_up_event(
            result,
            decision=decision,
            current_instances=1,
            request_index=25,
            completed_request_count=12,
            queue_visible_request_count=40,
            submitted_request_count=25,
            arrived_request_count=40,
            scale_event={
                "event_type": "physical_scale_up",
                "instance_id": "inst_2",
                "device_id": 1,
                "runtime_kind": "dedicated",
                "cold_start_latency_ms": 250.0,
                "warmed_adapters": 14,
            },
        )

        self.assertEqual(len(result.scale_up_events), 1)
        event = result.scale_up_events[0]
        self.assertEqual(event["request_index"], 25)
        self.assertEqual(event["queue_visible_request_count"], 40)
        self.assertEqual(event["completed_request_count"], 12)
        self.assertEqual(event["submitted_request_count"], 25)
        self.assertEqual(event["arrived_request_count"], 40)
        self.assertAlmostEqual(event["cold_start_latency_ms"], 250.0)
        self.assertEqual(event["warmed_adapters"], 14)
        self.assertEqual(event["planned_adapters"], ["finance_lora"])
        self.assertEqual(event["ordered_handoff_adapters"], ["finance_lora", "medical_lora"])
        self.assertEqual(event["warmup_plan_adapter_count"], 1)
        self.assertEqual(event["queue_at_ready_request_count"], 12)
        self.assertEqual(event["queue_at_ready_adapter_count"], 2)
        self.assertEqual(event["first_service_request_count"], 1)
        self.assertEqual(event["first_service_adapter_count"], 1)
        self.assertEqual(event["initial_admission_request_budget"], 2)
        self.assertEqual(event["effective_initial_warmup_parallelism"], 2)
        self.assertEqual(event["incumbent_started_request_count"], 28)
        self.assertEqual(event["budget_mode"], "dynamic_headroom_exact_handoff_prefix")
        self.assertEqual(event["exact_prefix_bytes"], 32 * 1024 * 1024)
        self.assertIn("inst_2", runner._scaleup_runtime_instance_ids)

    def test_scale_up_incumbent_started_request_count_respects_dispatch_capacity_limit(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.adapter_info = {}
        runner._runtime_forward_capacity_limit = lambda: 2
        runner._current_dispatch_capacity_limit = lambda: 1
        slots = [
            SimpleNamespace(
                instance_id=f"inst_{idx}",
                active_requests=0,
                predicted_cache_tier=lambda _adapter_id: "remote",
                predicted_request_cost_ms=lambda **_kwargs: 1000.0,
                predicted_tail_service_ms=lambda **_kwargs: 0.0,
                observed_request_costs={},
                observed_runtime_ttft_ms=0.0,
                observed_backbone_ttft_ms=0.0,
            )
            for idx in range(1, 4)
        ]
        runner.instance_pool = SimpleNamespace(get_slots=lambda: slots)

        started = runner._scale_up_incumbent_started_request_count(
            candidate_queue=[
                SimpleNamespace(adapter_id="a"),
                SimpleNamespace(adapter_id="b"),
                SimpleNamespace(adapter_id="c"),
            ],
            ready_delay_ms=1500.0,
        )

        self.assertEqual(started, 2)

    def test_scale_up_incumbent_started_request_count_uses_observed_busy_floor(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.adapter_info = {}
        runner._runtime_forward_capacity_limit = lambda: 1
        runner._current_dispatch_capacity_limit = lambda: 1
        slot = InstanceSlot("inst_1", None, None)
        slot.record_request_cost(
            adapter_id="finance_lora",
            cache_tier="host",
            lora_io_ms=100.0,
            runtime_ttft_ms=900.0,
            tail_service_ms=1100.0,
        )
        slot.predicted_request_cost_ms = lambda **_kwargs: 0.0
        slot.predicted_tail_service_ms = lambda **_kwargs: 0.0
        runner.instance_pool = SimpleNamespace(get_slots=lambda: [slot])

        started = runner._scale_up_incumbent_started_request_count(
            candidate_queue=[
                SimpleNamespace(adapter_id="finance_lora"),
                SimpleNamespace(adapter_id="finance_lora"),
            ],
            ready_delay_ms=1000.0,
        )

        self.assertEqual(started, 1)

    def test_scale_up_incumbent_started_request_count_does_not_invent_progress_without_observations(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.adapter_info = {}
        runner._runtime_forward_capacity_limit = lambda: 2
        runner._current_dispatch_capacity_limit = lambda: 2
        runner.instance_pool = SimpleNamespace(
            get_slots=lambda: [
                SimpleNamespace(
                    instance_id="inst_1",
                    active_requests=0,
                    predicted_cache_tier=lambda _adapter_id: "remote",
                    predicted_request_cost_ms=lambda **_kwargs: 0.0,
                    predicted_tail_service_ms=lambda **_kwargs: 0.0,
                    observed_request_costs={},
                    observed_runtime_ttft_ms=0.0,
                    observed_backbone_ttft_ms=0.0,
                )
            ]
        )

        started = runner._scale_up_incumbent_started_request_count(
            candidate_queue=[
                SimpleNamespace(adapter_id="a"),
                SimpleNamespace(adapter_id="b"),
            ],
            ready_delay_ms=30000.0,
        )

        self.assertEqual(started, 0)

    def test_scale_up_bootstrap_latency_ms_uses_engine_startup_signal(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._observed_scale_up_cold_start_latencies_ms = [51000.0, 53000.0]
        runner._live_scale_up_events = []
        runner.instance_pool = SimpleNamespace(
            get_slots=lambda: [
                SimpleNamespace(engine=SimpleNamespace(startup_latency_ms=26000.0))
            ]
        )
        runner.engine = SimpleNamespace(startup_latency_ms=26000.0)

        bootstrap_ms = runner._scale_up_bootstrap_latency_ms()

        self.assertEqual(bootstrap_ms, 26000.0)

    def test_add_dedicated_instance_slot_records_cold_start_latency(self) -> None:
        mark_calls = []
        refresh_calls = []
        schedule_all_calls = []
        order = []
        slot = SimpleNamespace(
            instance_id="inst_2",
            mark_adapter_tier=lambda aid, tier: mark_calls.append((aid, tier)),
        )
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._last_scale_up_handoff_plan = {"planned_adapters": ["hot_a", "hot_b"]}
        runner._observed_scale_up_cold_start_latencies_ms = []
        runner._scaleup_runtime_handoff_plans = {}
        runner.instance_pool = SimpleNamespace(
            count=lambda: 1,
            max_instances=2,
            add_instance=lambda engine, coord, owns_engine, owns_coordinator, device_id: (
                order.append("add_instance") or "inst_2"
            ),
            get_slot=lambda instance_id: slot,
        )
        async def fake_engine_factory(device_id=None):
            order.append("engine_factory")
            return SimpleNamespace(device_id=device_id), SimpleNamespace()

        async def fake_warmup(engine, coordinator=None):
            order.append("warmup")
            return {"hot_a", "hot_b"}

        runner.engine_factory = fake_engine_factory
        runner._select_dedicated_device_id = lambda: 1
        runner._prime_slot_cache_view = lambda slot_obj, include_gpu=False: None
        runner._sync_stack_gpu_accounting = lambda: None
        runner._refresh_slot_runtime_hints = lambda slot_obj: refresh_calls.append(slot_obj.instance_id)
        runner._warmup_engine_hot_set = fake_warmup
        runner._schedule_all_runtime_gpu_forward = lambda: schedule_all_calls.append("called") or 0
        async def fake_notify():
            self.assertIn("inst_2", runner._scaleup_runtime_instance_ids)
            self.assertEqual(
                runner._scaleup_runtime_handoff_plans["inst_2"]["planned_adapters"],
                ["hot_a", "hot_b"],
            )
            order.append("notify")
        runner._notify_dispatch_capacity_changed = fake_notify

        with patch("scripts.run_all_experiments.time.perf_counter", side_effect=[10.0, 10.25]):
            event = asyncio.run(runner._add_dedicated_instance_slot(coord_enabled=True))

        self.assertIsNotNone(event)
        self.assertEqual(event["runtime_kind"], "dedicated")
        self.assertAlmostEqual(event["cold_start_latency_ms"], 250.0)
        self.assertEqual(event["warmed_adapters"], 2)
        self.assertEqual(refresh_calls, ["inst_2"])
        self.assertEqual(len(mark_calls), 2)
        self.assertEqual(schedule_all_calls, [])
        self.assertEqual(order, ["engine_factory", "warmup", "add_instance", "notify"])
        self.assertEqual(runner._observed_scale_up_cold_start_latencies_ms, [250.0])
        self.assertIn("inst_2", runner._scaleup_runtime_instance_ids)
        self.assertEqual(
            runner._scaleup_runtime_handoff_plans["inst_2"]["planned_adapters"],
            ["hot_a", "hot_b"],
        )

    def test_select_dedicated_device_id_uses_tp_group_anchors(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.model_cfg = {"tensor_parallel_size": 2, "visible_device_ids": [0, 1, 2, 3]}
        runner.hw = {}
        runner.engine = None
        runner.instance_pool = SimpleNamespace(
            get_slots=lambda: [SimpleNamespace(device_id=0)],
            count=lambda: 1,
        )

        self.assertEqual(runner._select_dedicated_device_id(), 2)

    def test_dispatch_capacity_tracks_live_runtime_groups_under_workload_limit(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.wl_cfg = {"concurrency": 6}
        runner._workload_concurrency_limit = 6
        runner.model_cfg = {"runtime_concurrency_cap": 2}
        runner.engine = SimpleNamespace()
        runner.instance_pool = SimpleNamespace(
            get_runtime_groups=lambda: [
                [SimpleNamespace(instance_id="inst_1")],
                [SimpleNamespace(instance_id="inst_2")],
                [SimpleNamespace(instance_id="inst_3")],
            ],
            count=lambda: 3,
        )

        self.assertEqual(runner._current_dispatch_capacity_limit(), 6)

    def test_dispatch_capacity_respects_workload_cap_even_if_runtime_capacity_is_higher(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.wl_cfg = {"concurrency": 4}
        runner._workload_concurrency_limit = 4
        runner.model_cfg = {"runtime_concurrency_cap": 2}
        runner.engine = SimpleNamespace()
        runner.instance_pool = SimpleNamespace(
            get_runtime_groups=lambda: [
                [SimpleNamespace(instance_id="inst_1")],
                [SimpleNamespace(instance_id="inst_2")],
                [SimpleNamespace(instance_id="inst_3")],
                [SimpleNamespace(instance_id="inst_4")],
            ],
            count=lambda: 4,
        )

        self.assertEqual(runner._current_dispatch_capacity_limit(), 4)

    def test_live_scale_control_evaluation_uses_time_and_pressure(self) -> None:
        preload_calls = []
        scaleup_calls = []
        register_calls = []
        captured_overrides = []
        captured_metrics = []

        async def fake_trigger_scaling_preload(capacity_bytes: int, preferred_gpu_adapters=None):
            preload_calls.append((capacity_bytes, set(preferred_gpu_adapters or set())))
            return "plan"

        async def fake_scale_up_instance_pool(_coord_enabled: bool):
            scaleup_calls.append("scale")
            return {"instance_id": "inst_2", "runtime_kind": "dedicated"}

        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.baseline_type = "faaslora_full"
        runner.instance_pool = SimpleNamespace(count=lambda: 1, max_instances=2)
        runner._stack = SimpleNamespace(
            autoscaler=SimpleNamespace(
                decision_interval=15.0,
                make_scaling_decision_with_metrics=lambda metrics, current_instances, overrides=None: (
                    captured_metrics.append(metrics) or
                    captured_overrides.append(dict(overrides or {})) or
                    SimpleNamespace(
                        action=ScalingAction.SCALE_UP,
                        target_instances=2,
                        reason="Scale up triggered by: requests_per_second:scale_up",
                        metrics=metrics,
                        current_instances=current_instances,
                        overrides=overrides,
                    )
                ),
            ),
            trigger_scaling_preload=fake_trigger_scaling_preload,
        )
        runner._autoscaler_gpu_signal = lambda: 72.0
        runner._scale_up_preload_capacity_bytes = lambda preferred_gpu_adapters=None: 512 * 1024 * 1024
        runner._live_scale_up_preferred_gpu_adapters = lambda **kwargs: ["hot_a", "hot_b"]
        runner._runtime_forward_capacity_limit = lambda: 2
        runner._scale_up_instance_pool = fake_scale_up_instance_pool
        runner._dynamic_scaling = True
        runner._baseline_rps = 1.0
        runner._baseline_rps_ewma_beta = 0.25
        runner._scale_up_alpha = 0.3
        runner._scale_up_t_min = 1.0
        runner._scale_down_beta = 0.4
        runner._scale_down_duration_s = 45.0
        runner._low_load_since = None
        runner._register_scale_up_event = lambda result, decision, current_instances, request_index, scale_event, **counts: register_calls.append(
            {
                "request_index": request_index,
                "current_instances": current_instances,
                "scale_event": dict(scale_event),
                "reason": decision.reason,
                **counts,
            }
        )
        runner._arrival_rps = lambda _replay_t0: 40.0
        runner._arrived_request_count = lambda _replay_t0: 7
        runner._live_scale_eval_last_at = 0.0
        runner._live_scale_overrides = {"scale_up_threshold_rps": 5.0}
        runner._active_loras_ewma = 0.0
        runner._live_waiting_traces_by_id = {}
        runner._live_started_lora_counts = defaultdict(int)

        result = ScenarioResult(
            scenario_name="faaslora_full",
            baseline_type="faaslora_full",
            total=10,
            ttft_slo_ms=5000.0,
        )
        results_view = [
            RequestResult("r1", None, False, "normal", False, "backbone", 0.0, 100.0, 100.0, 0.0, 0.0, 20.0, 200.0, 10, 5, 0.0, True, "inst_1", False),
        ]

        with patch("scripts.run_all_experiments.time.perf_counter", return_value=20.0):
            triggered = asyncio.run(
                runner._maybe_run_live_scale_control_evaluation(
                    result=result,
                    coord_enabled=True,
                    replay_t0=0.0,
                    results_view=results_view,
                    backlog=3,
                    active_requests=2,
                    busy_ratio=1.0,
                    completed_count=7,
                    queue_visible_request_count=25,
                    visible_traces=[
                        SimpleNamespace(adapter_id="hot_a"),
                        SimpleNamespace(adapter_id="hot_b"),
                    ],
                )
            )

        self.assertTrue(triggered)
        self.assertEqual(scaleup_calls, ["scale"])
        self.assertEqual(preload_calls, [(512 * 1024 * 1024, {"hot_a", "hot_b"})])
        self.assertEqual(register_calls[0]["request_index"], 25)
        self.assertEqual(register_calls[0]["completed_request_count"], 7)
        self.assertEqual(register_calls[0]["submitted_request_count"], 25)
        self.assertEqual(register_calls[0]["arrived_request_count"], 7)
        self.assertEqual(runner._live_scale_overrides, captured_overrides[0])
        self.assertAlmostEqual(captured_metrics[0].avg_ttft_ms, 100.0)
        self.assertAlmostEqual(captured_metrics[0].avg_response_time_ms, 200.0)
        self.assertEqual(len(captured_overrides), 1)
        self.assertAlmostEqual(captured_overrides[0]["scale_up_threshold_rps"], 13.975)
        self.assertAlmostEqual(captured_overrides[0]["scale_down_threshold_rps"], 4.3)

    def test_live_scale_eval_period_prefers_explicit_scale_eval_interval(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._scale_eval_interval_s = 7.5
        runner._stack = SimpleNamespace(
            autoscaler=SimpleNamespace(decision_interval=99.0)
        )

        self.assertEqual(runner._live_scale_eval_period_s(), 7.5)

    def test_live_scale_up_evaluation_respects_time_interval(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.baseline_type = "faaslora_full"
        runner.instance_pool = SimpleNamespace(count=lambda: 1, max_instances=2)
        runner._stack = SimpleNamespace(
            autoscaler=SimpleNamespace(
                decision_interval=15.0,
                make_scaling_decision_with_metrics=lambda *args, **kwargs: SimpleNamespace(
                    action=ScalingAction.SCALE_UP,
                    target_instances=2,
                    reason="unexpected",
                ),
            ),
            trigger_scaling_preload=None,
        )
        runner._live_scale_eval_last_at = 10.0

        with patch("scripts.run_all_experiments.time.perf_counter", return_value=20.0):
            should_eval = runner._should_attempt_live_scale_control_eval(
                now_monotonic=time.perf_counter(),
                arrival_rps=40.0,
                backlog=3,
                active_requests=2,
                busy_ratio=1.0,
            )

        self.assertFalse(should_eval)

    def test_arrival_rps_decays_after_replay_finishes(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._scheduled_arrivals = [0.0, 1.0, 2.0]
        runner._arrival_window_s = 1.0

        with patch("scripts.run_all_experiments.time.perf_counter", return_value=10.0):
            self.assertEqual(runner._arrival_rps(0.0), 0.0)

    def test_arrival_rps_uses_recent_window_without_cumulative_boost(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._scheduled_arrivals = [0.0, 0.1, 0.2, 100.0]
        runner._arrival_window_s = 5.0

        with patch("scripts.run_all_experiments.time.perf_counter", return_value=4.0):
            self.assertAlmostEqual(runner._arrival_rps(0.0), 2.0 / 5.0)

    def test_live_scale_control_eval_can_run_on_backlog_without_recent_arrivals(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.baseline_type = "faaslora_full"
        runner.instance_pool = SimpleNamespace(count=lambda: 1, max_instances=2, min_instances=1)
        runner._stack = SimpleNamespace()
        runner._live_scale_eval_last_at = 0.0
        runner._scale_eval_interval_s = 15.0

        with patch("scripts.run_all_experiments.time.perf_counter", return_value=20.0):
            should_eval = runner._should_attempt_live_scale_control_eval(
                now_monotonic=time.perf_counter(),
                arrival_rps=0.0,
                backlog=3,
                active_requests=0,
                busy_ratio=0.0,
            )

        self.assertTrue(should_eval)

    def test_scale_down_one_instance_require_idle_skips_busy_slots(self) -> None:
        removed_ids = []
        idle_slot = SimpleNamespace(
            instance_id="inst_idle",
            active_requests=0,
            load_queue_depth=0,
            last_selected_at=5.0,
            created_at=1.0,
        )
        busy_slot = SimpleNamespace(
            instance_id="inst_busy",
            active_requests=1,
            load_queue_depth=0,
            last_selected_at=10.0,
            created_at=2.0,
        )
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._primary_instance_id = "inst_1"
        runner.instance_pool = SimpleNamespace(
            min_instances=1,
            count=lambda: 3,
            get_slots=lambda: [
                SimpleNamespace(instance_id="inst_1", active_requests=0, load_queue_depth=0),
                busy_slot,
                idle_slot,
            ],
            remove_instance=lambda instance_id: removed_ids.append(instance_id) or idle_slot,
        )

        async def fake_cleanup(_slot):
            return None

        runner._cleanup_removed_slot = fake_cleanup

        event = asyncio.run(runner._scale_down_one_instance(require_idle=True))

        self.assertIsNotNone(event)
        self.assertEqual(removed_ids, ["inst_idle"])
        self.assertEqual(event["instance_id"], "inst_idle")

    def test_live_scale_control_evaluation_can_trigger_online_scale_down(self) -> None:
        scale_down_calls = []
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.baseline_type = "faaslora_full"
        runner.instance_pool = SimpleNamespace(count=lambda: 2, max_instances=4, min_instances=1)
        runner._stack = SimpleNamespace(
            autoscaler=SimpleNamespace(
                decision_interval=15.0,
                make_scaling_decision_with_metrics=lambda *args, **kwargs: SimpleNamespace(
                    action=ScalingAction.SCALE_DOWN,
                    target_instances=1,
                    reason="Scale down triggered by: requests_per_second:scale_down",
                ),
            ),
        )
        runner._dynamic_scaling = True
        runner._baseline_rps = 1.0
        runner._baseline_rps_ewma_beta = 0.25
        runner._scale_up_alpha = 0.3
        runner._scale_up_t_min = 1.0
        runner._scale_down_beta = 0.4
        runner._scale_down_duration_s = 45.0
        runner._low_load_since = 0.0
        runner._active_loras_ewma = 0.0
        runner._live_waiting_traces_by_id = {}
        runner._live_started_lora_counts = defaultdict(int)
        runner._live_scale_eval_last_at = 0.0
        runner._live_scale_overrides = None
        runner._arrival_rps = lambda _replay_t0: 0.2
        runner._arrived_request_count = lambda _replay_t0: 8
        runner._autoscaler_gpu_signal = lambda: 5.0
        runner._scale_down_one_instance = lambda require_idle=False: scale_down_calls.append(require_idle) or asyncio.sleep(0, result={
            "event_type": "physical_scale_down",
            "instance_id": "inst_2",
            "device_id": 1,
            "runtime_kind": "dedicated",
        })

        result = ScenarioResult(
            scenario_name="faaslora_full",
            baseline_type="faaslora_full",
            total=10,
            ttft_slo_ms=5000.0,
        )

        with patch("scripts.run_all_experiments.time.perf_counter", return_value=60.0):
            triggered = asyncio.run(
                runner._maybe_run_live_scale_control_evaluation(
                    result=result,
                    coord_enabled=True,
                    replay_t0=0.0,
                    results_view=[],
                    backlog=0,
                    active_requests=0,
                    busy_ratio=0.0,
                    completed_count=8,
                    queue_visible_request_count=8,
                    visible_traces=[],
                )
            )

        self.assertTrue(triggered)
        self.assertEqual(scale_down_calls, [True])
        self.assertEqual(result.scale_down_events, 1)
        self.assertEqual(result.scale_down_event_log[0]["instance_id"], "inst_2")
        self.assertEqual(result.scale_down_event_log[0]["queue_visible_request_count"], 8)
        self.assertEqual(result.scale_down_event_log[0]["arrived_request_count"], 8)

    def test_live_scale_control_evaluation_blocks_scale_down_while_pressure_present(self) -> None:
        scale_down_calls = []
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.baseline_type = "faaslora_full"
        runner.instance_pool = SimpleNamespace(count=lambda: 2, max_instances=4, min_instances=1)
        runner._stack = SimpleNamespace(
            autoscaler=SimpleNamespace(
                decision_interval=15.0,
                make_scaling_decision_with_metrics=lambda *args, **kwargs: SimpleNamespace(
                    action=ScalingAction.SCALE_DOWN,
                    target_instances=1,
                    reason="Scale down triggered by: ttft_latency:scale_down",
                ),
            ),
        )
        runner._dynamic_scaling = True
        runner._baseline_rps = 1.0
        runner._baseline_rps_ewma_beta = 0.25
        runner._scale_up_alpha = 0.3
        runner._scale_up_t_min = 1.0
        runner._scale_down_beta = 0.4
        runner._scale_down_duration_s = 45.0
        runner._low_load_since = 0.0
        runner._active_loras_ewma = 0.0
        runner._live_waiting_traces_by_id = {}
        runner._live_started_lora_counts = defaultdict(int)
        runner._live_scale_eval_last_at = 0.0
        runner._live_scale_overrides = None
        runner._arrival_rps = lambda _replay_t0: 0.2
        runner._arrived_request_count = lambda _replay_t0: 8
        runner._autoscaler_gpu_signal = lambda: 5.0
        runner._scale_down_one_instance = (
            lambda require_idle=False: scale_down_calls.append(require_idle)
            or asyncio.sleep(0, result={"instance_id": "inst_2"})
        )

        result = ScenarioResult(
            scenario_name="faaslora_full",
            baseline_type="faaslora_full",
            total=10,
            ttft_slo_ms=5000.0,
        )

        with patch("scripts.run_all_experiments.time.perf_counter", return_value=60.0):
            triggered = asyncio.run(
                runner._maybe_run_live_scale_control_evaluation(
                    result=result,
                    coord_enabled=True,
                    replay_t0=0.0,
                    results_view=[],
                    backlog=2,
                    active_requests=1,
                    busy_ratio=0.25,
                    completed_count=5,
                    queue_visible_request_count=8,
                    visible_traces=[],
                )
            )

        self.assertFalse(triggered)
        self.assertEqual(scale_down_calls, [])
        self.assertEqual(result.scale_down_events, 0)

    def test_dynamic_scale_down_uses_monotonic_clock(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._dynamic_scaling = True
        runner._scale_down_duration_s = 45.0
        runner._low_load_since = None
        runner._baseline_rps = 10.0
        runner._baseline_rps_ewma_beta = 0.25
        runner._scale_up_alpha = 0.3
        runner._scale_up_t_min = 1.0
        runner._scale_down_beta = 0.4

        overrides = runner._observe_dynamic_scaling_rps(0.5, 100.0)

        self.assertAlmostEqual(overrides["scale_up_threshold_rps"], 9.9125)
        self.assertAlmostEqual(overrides["scale_down_threshold_rps"], 3.05)
        self.assertEqual(runner._low_load_since, 100.0)

        with patch("scripts.run_all_experiments.time.perf_counter", return_value=120.0):
            self.assertFalse(runner._should_trigger_scale_down())
        with patch("scripts.run_all_experiments.time.perf_counter", return_value=146.0):
            self.assertTrue(runner._should_trigger_scale_down())

    def test_scaleup_gpu_candidates_ignore_global_min_hotness_gate(self) -> None:
        stack = ExperimentStack.__new__(ExperimentStack)
        now = time.time()
        metas = {
            "warm_a": SimpleNamespace(last_accessed_at=now, hotness_score=0.95, value_per_byte=0.1, size_bytes=32 * 1024 * 1024),
            "warm_b": SimpleNamespace(last_accessed_at=now - 5.0, hotness_score=0.05, value_per_byte=0.05, size_bytes=32 * 1024 * 1024),
        }
        stack.sync_local_tier_paths = lambda: None
        stack._host_paths = {"warm_a": "/tmp/warm_a", "warm_b": "/tmp/warm_b"}
        stack._nvme_paths = {}
        stack.registry = SimpleNamespace(get_artifact=lambda aid: metas.get(aid))
        stack.preloading_planner = SimpleNamespace(min_hotness_threshold=0.9)

        selected = stack._select_scaleup_gpu_candidates(64 * 1024 * 1024)

        self.assertEqual(set(selected), {"warm_a", "warm_b"})

    def test_scaleup_gpu_candidates_prefer_live_gpu_hotset_when_capacity_limited(self) -> None:
        stack = ExperimentStack.__new__(ExperimentStack)
        now = time.time()
        metas = {
            "live_gpu": SimpleNamespace(
                last_accessed_at=now - 60.0,
                hotness_score=0.2,
                value_per_byte=0.02,
                size_bytes=32 * 1024 * 1024,
            ),
            "recent_only": SimpleNamespace(
                last_accessed_at=now,
                hotness_score=0.95,
                value_per_byte=0.1,
                size_bytes=32 * 1024 * 1024,
            ),
        }
        stack.sync_local_tier_paths = lambda: None
        stack._host_paths = {"live_gpu": "/tmp/live_gpu", "recent_only": "/tmp/recent_only"}
        stack._nvme_paths = {}
        stack.registry = SimpleNamespace(get_artifact=lambda aid: metas.get(aid))

        selected = stack._select_scaleup_gpu_candidates(
            32 * 1024 * 1024,
            preferred_gpu_adapters={"live_gpu"},
        )

        self.assertEqual(selected, ["live_gpu"])

    def test_scaleup_gpu_candidates_allow_live_frontier_without_recent_access(self) -> None:
        stack = ExperimentStack.__new__(ExperimentStack)
        now = time.time()
        metas = {
            "frontier_a": SimpleNamespace(
                last_accessed_at=0.0,
                hotness_score=0.01,
                value_per_byte=0.01,
                size_bytes=32 * 1024 * 1024,
            ),
            "recent_b": SimpleNamespace(
                last_accessed_at=now,
                hotness_score=0.95,
                value_per_byte=0.1,
                size_bytes=32 * 1024 * 1024,
            ),
        }
        stack.sync_local_tier_paths = lambda: None
        stack._host_paths = {"frontier_a": "/tmp/frontier_a", "recent_b": "/tmp/recent_b"}
        stack._nvme_paths = {}
        stack.registry = SimpleNamespace(get_artifact=lambda aid: metas.get(aid))

        selected = stack._select_scaleup_gpu_candidates(
            32 * 1024 * 1024,
            preferred_gpu_adapters=["frontier_a"],
        )

        self.assertEqual(selected, ["frontier_a"])

    def test_scaleup_gpu_candidates_respect_live_frontier_order(self) -> None:
        stack = ExperimentStack.__new__(ExperimentStack)
        metas = {
            "frontier_first": SimpleNamespace(
                last_accessed_at=0.0,
                hotness_score=0.01,
                value_per_byte=0.01,
                size_bytes=32 * 1024 * 1024,
            ),
            "frontier_second": SimpleNamespace(
                last_accessed_at=0.0,
                hotness_score=0.99,
                value_per_byte=0.2,
                size_bytes=32 * 1024 * 1024,
            ),
        }
        stack.sync_local_tier_paths = lambda: None
        stack._host_paths = {
            "frontier_first": "/tmp/frontier_first",
            "frontier_second": "/tmp/frontier_second",
        }
        stack._nvme_paths = {}
        stack.registry = SimpleNamespace(get_artifact=lambda aid: metas.get(aid))

        selected = stack._select_scaleup_gpu_candidates(
            32 * 1024 * 1024,
            preferred_gpu_adapters=["frontier_first", "frontier_second"],
        )

        self.assertEqual(selected, ["frontier_first"])

    def test_gpu_forward_candidates_respect_preferred_order(self) -> None:
        stack = ExperimentStack.__new__(ExperimentStack)
        stack._dynamic_forwarding_enabled = True
        stack.sync_local_tier_paths = lambda: None
        stack._host_paths = {
            "frontier_first": "/tmp/frontier_first",
            "frontier_second": "/tmp/frontier_second",
        }
        stack._nvme_paths = {}
        stack._artifact_size_mb = lambda _aid: 32.0
        stack._artifact_metadata = lambda _aid: SimpleNamespace(last_accessed_at=0.0)
        stack._global_admission_utility = lambda *_args, **_kwargs: 1.0
        utility = {"frontier_first": 1.0, "frontier_second": 100.0}
        stack._forward_utility = lambda aid, *_args, **_kwargs: utility[aid]

        class FakeCoord:
            def evaluate_gpu_admission(
                self,
                _adapter_id: str,
                _size_mb: float,
                tier: str = "host",
                utility_override=None,
            ):
                return {
                    "admit": True,
                    "should_attempt": True,
                    "effective_capacity_mb": 1024.0,
                    "tier": tier,
                }

        candidate = stack.select_gpu_forward_candidate(
            gpu_resident_adapters=set(),
            coordinator=FakeCoord(),
            preferred_gpu_adapters=["frontier_first", "frontier_second"],
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["adapter_id"], "frontier_first")

    def test_scaleup_gpu_candidates_rank_by_hotness_before_small_recency_deltas(self) -> None:
        stack = ExperimentStack.__new__(ExperimentStack)
        now = time.time()
        metas = {
            "hot_recent": SimpleNamespace(
                last_accessed_at=now - 3.0,
                hotness_score=0.95,
                value_per_byte=0.05,
                size_bytes=32 * 1024 * 1024,
            ),
            "cold_newer": SimpleNamespace(
                last_accessed_at=now - 1.0,
                hotness_score=0.05,
                value_per_byte=0.05,
                size_bytes=32 * 1024 * 1024,
            ),
        }
        stack.sync_local_tier_paths = lambda: None
        stack._host_paths = {"hot_recent": "/tmp/hot_recent", "cold_newer": "/tmp/cold_newer"}
        stack._nvme_paths = {}
        stack.registry = SimpleNamespace(get_artifact=lambda aid: metas.get(aid))
        stack.hotness_tracker = SimpleNamespace(window_seconds=300.0, get_hotness=lambda aid: 0.0)
        stack.adapter_info = {"hot_recent": {"hotness": 0.95}, "cold_newer": {"hotness": 0.05}}

        selected = stack._select_scaleup_gpu_candidates(32 * 1024 * 1024)

        self.assertEqual(selected, ["hot_recent"])

    def test_online_artifact_hotness_ignores_static_prior_without_observed_access(self) -> None:
        stack = ExperimentStack.__new__(ExperimentStack)
        stack.registry = SimpleNamespace(
            get_artifact=lambda aid: SimpleNamespace(
                last_accessed_at=0.0,
                access_count=0,
                hit_count=0,
                miss_count=0,
                hotness_score=0.95,
            )
        )
        stack.hotness_tracker = SimpleNamespace(window_seconds=300.0, get_hotness=lambda aid: 0.0)
        stack.adapter_info = {"static_only": {"hotness": 0.95}}

        self.assertEqual(stack._online_artifact_hotness("static_only"), 0.0)
        self.assertEqual(stack._artifact_hotness("static_only"), 0.95)

    def test_adapter_recently_active_uses_derived_online_window_when_tracker_missing(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        now = time.time()
        runner._online_hotness_window_s = 15.0
        runner._stack = SimpleNamespace(
            hotness_tracker=None,
            _artifact_metadata=lambda adapter_id: SimpleNamespace(last_accessed_at=now - 20.0),
        )

        self.assertFalse(runner._adapter_is_recently_active("stale", now=now))

    def test_scaleup_preload_prefers_recent_live_runtime_gpu_union(self) -> None:
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._gpu_warmed = {"boot_hot"}
        runner._stack = SimpleNamespace(
            sync_local_tier_paths=lambda: None,
            _host_paths={"host_recent": "/tmp/host_recent", "host_stale": "/tmp/host_stale"},
            _nvme_paths={},
        )
        runner.instance_pool = SimpleNamespace(
            get_runtime_groups=lambda: [
                [SimpleNamespace(gpu_resident_adapters={"a", "b"})],
                [SimpleNamespace(gpu_resident_adapters={"b", "c", "stale"})],
            ]
        )
        runner._adapter_is_recently_active = lambda aid, now=None: aid not in {"stale", "host_stale"}

        self.assertEqual(
            runner._scale_up_warmup_preferred_gpu_adapters(),
            {"a", "b", "c"},
        )

    def test_scaleup_gpu_candidates_filter_stale_recent_window_instead_of_hour_threshold(self) -> None:
        stack = ExperimentStack.__new__(ExperimentStack)
        now = time.time()
        metas = {
            "recent_a": SimpleNamespace(
                last_accessed_at=now - 30.0,
                hotness_score=0.9,
                value_per_byte=0.1,
                size_bytes=32 * 1024 * 1024,
            ),
            "stale_b": SimpleNamespace(
                last_accessed_at=now - 600.0,
                hotness_score=0.95,
                value_per_byte=0.1,
                size_bytes=32 * 1024 * 1024,
            ),
        }
        stack.sync_local_tier_paths = lambda: None
        stack._host_paths = {"recent_a": "/tmp/recent_a", "stale_b": "/tmp/stale_b"}
        stack._nvme_paths = {}
        stack.registry = SimpleNamespace(get_artifact=lambda aid: metas.get(aid))
        stack.hotness_tracker = SimpleNamespace(window_seconds=300.0)

        selected = stack._select_scaleup_gpu_candidates(64 * 1024 * 1024)

        self.assertEqual(selected, ["recent_a"])

    def test_sync_stack_gpu_accounting_tracks_primary_runtime_only_for_tp1_scale_out(self) -> None:
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

    def test_scaleup_warmup_uses_curated_admission_utility(self) -> None:
        class DummyEngine:
            async def load_lora_to_gpu_and_measure(self, _path: str, _aid: str):
                return 12.0, True

        decisions = []

        class DummyCoordinator:
            effective_capacity_admission_enabled = True
            _residency_manager = SimpleNamespace()

            def evaluate_gpu_admission(self, adapter_id: str, size_mb: float, tier: str = "nvme", utility_override=None):
                decisions.append((adapter_id, size_mb, tier, utility_override))
                return {"admit": True}

        registry_updates = []
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.preload_cfg = {"gpu_warmup_hotness": 0.6}
        runner.adapter_info = {"warm_a": {"size_mb": 30.0, "hotness": 0.2}}
        runner._stack = SimpleNamespace(
            _host_paths={"warm_a": "/tmp/warm_a"},
            _nvme_paths={},
            consume_scaleup_gpu_plan=lambda: ["warm_a"],
            registry=SimpleNamespace(update_artifact=lambda aid, payload: registry_updates.append((aid, payload))),
        )

        warmed = asyncio.run(runner._warmup_engine_hot_set(DummyEngine(), coordinator=DummyCoordinator()))

        self.assertEqual(warmed, {"warm_a"})
        self.assertEqual(decisions, [("warm_a", 30.0, "host", 1.0)])
        self.assertEqual(registry_updates[0][0], "warm_a")

    def test_scaleup_warmup_plan_does_not_expand_with_global_hot_adapters(self) -> None:
        loaded = []

        class DummyEngine:
            async def load_lora_to_gpu_and_measure(self, path: str, aid: str):
                loaded.append((aid, path))
                return 10.0, True

        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.preload_cfg = {"gpu_warmup_hotness": 0.6}
        runner.adapter_info = {
            "planned_a": {"size_mb": 30.0, "hotness": 0.2},
            "extra_hot": {"size_mb": 30.0, "hotness": 0.95},
        }
        runner._stack = SimpleNamespace(
            _host_paths={
                "planned_a": "/tmp/planned_a",
                "extra_hot": "/tmp/extra_hot",
            },
            _nvme_paths={},
            consume_scaleup_gpu_plan=lambda: ["planned_a"],
            registry=SimpleNamespace(update_artifact=lambda *_args, **_kwargs: None),
        )

        warmed = asyncio.run(runner._warmup_engine_hot_set(DummyEngine(), coordinator=None))

        self.assertEqual(warmed, {"planned_a"})
        self.assertEqual(loaded, [("planned_a", "/tmp/planned_a")])

    def test_scaleup_warmup_without_plan_does_not_fallback_to_static_hotset(self) -> None:
        loaded = []

        class DummyEngine:
            async def load_lora_to_gpu_and_measure(self, path: str, aid: str):
                loaded.append((aid, path))
                return 10.0, True

        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.preload_cfg = {"gpu_warmup_hotness": 0.6}
        runner.adapter_info = {
            "extra_hot": {"size_mb": 30.0, "hotness": 0.95},
        }
        runner._stack = SimpleNamespace(
            _host_paths={"extra_hot": "/tmp/extra_hot"},
            _nvme_paths={},
            consume_scaleup_gpu_plan=lambda: [],
            registry=SimpleNamespace(update_artifact=lambda *_args, **_kwargs: None),
        )

        warmed = asyncio.run(runner._warmup_engine_hot_set(DummyEngine(), coordinator=None))

        self.assertEqual(warmed, set())
        self.assertEqual(loaded, [])

    def test_scaleup_warmup_uses_explicit_instance_plan_over_pending_global_plan(self) -> None:
        loaded = []

        class DummyEngine:
            async def load_lora_to_gpu_and_measure(self, path: str, aid: str):
                loaded.append((aid, path))
                return 10.0, True

        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.preload_cfg = {"gpu_warmup_hotness": 0.6}
        runner.adapter_info = {
            "planned_a": {"size_mb": 30.0, "hotness": 0.2},
            "wrong_global": {"size_mb": 30.0, "hotness": 0.9},
        }
        runner._stack = SimpleNamespace(
            _host_paths={
                "planned_a": "/tmp/planned_a",
                "wrong_global": "/tmp/wrong_global",
            },
            _nvme_paths={},
            consume_scaleup_gpu_plan=lambda: ["wrong_global"],
            registry=SimpleNamespace(update_artifact=lambda *_args, **_kwargs: None),
        )

        warmed = asyncio.run(
            runner._warmup_engine_hot_set(
                DummyEngine(),
                coordinator=None,
                planned_aids=["planned_a"],
            )
        )

        self.assertEqual(warmed, {"planned_a"})
        self.assertEqual(loaded, [("planned_a", "/tmp/planned_a")])

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
