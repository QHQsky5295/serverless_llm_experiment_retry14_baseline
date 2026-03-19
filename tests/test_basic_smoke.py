"""Basic smoke tests that run in the stable mainline environment."""

from __future__ import annotations

import os
import asyncio
import json
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
    _apply_tp_instance_capacity_guard,
    _adapter_matches_model,
    _build_local_tp_runtime_env_updates,
    _normalize_lora_preparation_mode,
    _resolve_vllm_runtime_guards,
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
        selected_model = self.experiments["model_profiles"]["mistral_nemo_12b_tp2_v2_publicmix"]["model"]

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
        self.assertEqual(model["max_loras"], 1)
        self.assertEqual(model["max_num_seqs"], 1)
        self.assertEqual(model["max_num_batched_tokens"], 1024)
        self.assertEqual(model["runtime_concurrency_cap"], 1)
        self.assertEqual(lora_cfg["full_num_adapters"], 500)

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

        self.assertEqual(selection["model"], "mistral_nemo_12b_tp2_v2_publicmix")
        self.assertEqual(selection["dataset"], "azure_sharegpt_rep1000")
        self.assertEqual(selection["workload"], "mistral_nemo_12b_tp2_main")
        self.assertIn("mistral_nemo_12b_tp2_v2_publicmix", model_profiles)
        self.assertIn("azure_sharegpt_rep4000", dataset_profiles)
        self.assertIn("mistral_nemo_12b_tp2_main", workload_profiles)

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

        self.assertEqual(storage["remote_dir"], "artifacts/frozen/mistral_nemo_12b_a500_v2_publicmix")

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

    def test_generator_defaults_follow_selected_profile(self) -> None:
        defaults = resolve_generation_defaults(EXPERIMENTS_CONFIG)

        self.assertEqual(
            defaults["model"],
            self.experiments["model_profiles"]["mistral_nemo_12b_tp2_v2_publicmix"]["model"]["name"],
        )
        self.assertEqual(defaults["num_adapters"], 500)
        self.assertEqual(defaults["generation_mode"], "peft_finetune")
        self.assertEqual(defaults["artifact_pool_profile"], "standardized_v1")

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
