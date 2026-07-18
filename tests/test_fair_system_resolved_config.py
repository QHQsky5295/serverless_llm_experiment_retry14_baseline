from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MAIN_REPO = Path("/home/qhq/serverless_llm_experiment_retry14_baseline")
MODULE_PATH = ROOT / "scripts" / "fair_system_resolved_config.py"
SPEC = importlib.util.spec_from_file_location("fair_system_resolved_config", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class FairSystemResolvedConfigTests(unittest.TestCase):
    def _args(
        self,
        root: Path,
        *,
        seed: int,
        total: int,
        role: str,
        run_tag: str = "run",
        trace_path: str = "/trace.json",
        subset_path: str = "/subset.json",
        execution_order: str = "faaslora slora serverlessllm",
        output_name: str = "sidecar.json",
        bandwidth: str = "250",
        zipf: str = "",
        rotation: str = "",
        time_scale: str = "",
    ):
        return module.parse_args(
            [
                "--baselines-root",
                str(ROOT),
                "--main-repo",
                str(MAIN_REPO),
                "--model-profile",
                "llama2_7b_main_v2_publicmix",
                "--dataset-profile",
                "azure_sharegpt_rep4000",
                "--workload-profile",
                "llama2_7b_auto500_formal4000_s8",
                "--total-requests",
                str(total),
                "--selected-num-adapters",
                "500",
                "--sampling-seed",
                str(seed),
                "--time-scale-factor",
                time_scale,
                "--trace-role",
                role,
                "--generation-contract",
                "fixed_length_greedy_v1",
                "--fixed-output-max-tokens",
                "256",
                "--fixed-prompt-max-tokens",
                "759",
                "--storage-bandwidth-mib-s",
                bandwidth,
                "--zipf-exponent",
                zipf,
                "--hotset-rotation-requests",
                rotation,
                "--faaslora-scenario",
                "v2_full",
                "--gpu-ids",
                "0,1,2,3",
                "--run-tag",
                run_tag,
                "--trace-path",
                trace_path,
                "--adapter-subset-path",
                subset_path,
                "--execution-order",
                execution_order,
                "--output",
                str(root / output_name),
                "--registry",
                str(root / "registry.json"),
            ]
        )

    def _build(self, args, env=None):
        clean = {
            "source_clean_for_formal": True,
            "baseline_tracked_dirty_paths": [],
            "baseline_tracked_clean": True,
            "faaslora_tracked_dirty_paths": [],
            "faaslora_allowed_tracked_dirty_paths": [
                "configs/generated/lora_manifest_1000.json"
            ],
            "faaslora_disallowed_tracked_dirty_paths": [],
            "faaslora_tracked_clean_for_formal": True,
            "untracked_files_checked": False,
        }
        with mock.patch.object(module, "_source_identity", return_value={"source": "a" * 64}), mock.patch.object(
            module, "_git_commit", return_value="b" * 40
        ), mock.patch.object(module, "source_cleanliness", return_value=clean):
            return module.build_sidecar(args, env or {})

    def test_hash_excludes_seed_trace_subset_order_tag_and_trace_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validation = self._build(
                self._args(
                    root,
                    seed=41,
                    total=1000,
                    role="validation",
                    run_tag="validation-seed41",
                    trace_path="/trace-seed41.json",
                    subset_path="/subset-seed41.json",
                    execution_order="faaslora slora serverlessllm",
                )
            )
            heldout = self._build(
                self._args(
                    root,
                    seed=43,
                    total=4000,
                    role="heldout",
                    run_tag="heldout-seed43",
                    trace_path="/trace-seed43.json",
                    subset_path="/subset-seed43.json",
                    execution_order="serverlessllm slora faaslora",
                    output_name="heldout.json",
                )
            )
        self.assertEqual(
            validation["system_resolved_config_sha256"],
            heldout["system_resolved_config_sha256"],
        )
        self.assertEqual(
            validation["configuration_family_id"], heldout["configuration_family_id"]
        )
        module._assert_no_forbidden_hash_keys(validation["hashed_config"])

    def test_each_system_runtime_override_changes_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(Path(tmp), seed=41, total=1000, role="validation")
            base = self._build(args, {})["system_resolved_config_sha256"]
            for env in (
                {"FAASLORA_MAX_INSTANCES": "3"},
                {"SLORA_MAX_TOTAL_TOKEN_NUM": "16000"},
                {"SLLM_DEPLOY_TARGET": "7"},
            ):
                with self.subTest(env=env):
                    self.assertNotEqual(
                        base, self._build(args, env)["system_resolved_config_sha256"]
                    )

    def test_sensitivity_axes_change_full_identity_but_not_system_config_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nominal = self._build(
                self._args(
                    root,
                    seed=43,
                    total=4000,
                    role="heldout",
                    bandwidth="250",
                    zipf="1.0",
                    rotation="500",
                    time_scale="8.0",
                ),
                {
                    "FAASLORA_STORAGE_BANDWIDTH_MIB_S": "250",
                    "FAASLORA_ZIPF_EXPONENT": "1.0",
                    "FAASLORA_HOTSET_ROTATION_REQUESTS": "500",
                    "SLLM_REMOTE_ARTIFACT_STAGE_BANDWIDTH_MIB_S": "250",
                    "SLLM_ZIPF_EXPONENT": "1.0",
                },
            )
            sensitivity = self._build(
                self._args(
                    root,
                    seed=44,
                    total=4000,
                    role="heldout",
                    bandwidth="11.9209",
                    zipf="1.4",
                    rotation="100",
                    time_scale="4.0",
                    output_name="sensitivity.json",
                ),
                {
                    "FAASLORA_STORAGE_BANDWIDTH_MIB_S": "11.9209",
                    "FAASLORA_ZIPF_EXPONENT": "1.4",
                    "FAASLORA_HOTSET_ROTATION_REQUESTS": "100",
                    "SLLM_REMOTE_ARTIFACT_STAGE_BANDWIDTH_MIB_S": "11.9209",
                    "SLLM_ZIPF_EXPONENT": "1.4",
                },
            )
        self.assertEqual(
            nominal["system_resolved_config_sha256"],
            sensitivity["system_resolved_config_sha256"],
        )
        self.assertEqual(
            nominal["configuration_family_id"],
            sensitivity["configuration_family_id"],
        )
        self.assertNotEqual(
            nominal["full_run_identity_sha256"],
            sensitivity["full_run_identity_sha256"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scale8 = self._build(
                self._args(
                    root,
                    seed=41,
                    total=1000,
                    role="validation",
                    time_scale="8.0",
                ),
                {"FAASLORA_TIME_SCALE_FACTOR": "8.0", "SLLM_TIME_SCALE_FACTOR": "8.0"},
            )
            scale4 = self._build(
                self._args(
                    root,
                    seed=41,
                    total=1000,
                    role="validation",
                    time_scale="4.0",
                    output_name="scale4.json",
                ),
                {"FAASLORA_TIME_SCALE_FACTOR": "4.0", "SLLM_TIME_SCALE_FACTOR": "4.0"},
            )
        self.assertEqual(
            scale8["system_resolved_config_sha256"],
            scale4["system_resolved_config_sha256"],
        )
        self.assertNotEqual(
            scale8["full_run_identity_sha256"],
            scale4["full_run_identity_sha256"],
        )

    def test_provenance_environment_does_not_change_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(Path(tmp), seed=41, total=1000, role="validation")
            base = self._build(args, {})["system_resolved_config_sha256"]
            provenance_env = {
                "SLLM_RUN_TAG": "different",
                "SLLM_SHARED_TRACE_PATH": "/different-trace",
                "SLLM_SHARED_ADAPTER_SUBSET_PATH": "/different-subset",
                "SLLM_SAMPLING_SEED": "99",
                "FAASLORA_RESULTS_TAG": "different",
                "FAASLORA_GENERATION_SEED": "99",
                "FAIR_ROUND_EXECUTION_ORDER": "slora faaslora",
            }
            self.assertEqual(
                base,
                self._build(args, provenance_env)["system_resolved_config_sha256"],
            )

    def test_trace_role_gate_is_strict_for_v2(self):
        self.assertEqual(
            module.validate_trace_role(41, "validation", "v2_full", 1000),
            "validation",
        )
        self.assertEqual(
            module.validate_trace_role(43, "heldout", "v2_full", 4000),
            "heldout",
        )
        with self.assertRaisesRegex(ValueError, "trace_role mismatch"):
            module.validate_trace_role(41, "heldout", "v2_full", 1000)
        with self.assertRaisesRegex(ValueError, "exactly 4,000"):
            module.validate_trace_role(44, "heldout", "v2_full", 1000)
        with self.assertRaisesRegex(ValueError, "permits only seeds"):
            module.validate_trace_role(46, "heldout", "v2_full", 4000)

    def test_heldout_registry_rejects_config_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validation_a = self._build(
                self._args(
                    root,
                    seed=41,
                    total=1000,
                    role="validation",
                    output_name="validation-a.json",
                ),
                {},
            )
            validation_b = self._build(
                self._args(
                    root,
                    seed=41,
                    total=1000,
                    role="validation",
                    output_name="validation-b.json",
                ),
                {"SLLM_DEPLOY_TARGET": "9"},
            )
            seed43 = self._build(
                self._args(root, seed=43, total=4000, role="heldout"), {}
            )
            seed44 = self._build(
                self._args(
                    root,
                    seed=44,
                    total=4000,
                    role="heldout",
                    output_name="seed44.json",
                ),
                {"SLLM_DEPLOY_TARGET": "9"},
            )
            module.register_sidecar(validation_a, root / "registry.json")
            module.register_sidecar(validation_b, root / "registry.json")
            module.register_sidecar(seed43, root / "registry.json")
            with self.assertRaisesRegex(ValueError, "held-out resolved configuration changed"):
                module.register_sidecar(seed44, root / "registry.json")

    def test_heldout_registry_requires_seed41_validation_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed43 = self._build(
                self._args(root, seed=43, total=4000, role="heldout"), {}
            )
            with self.assertRaisesRegex(ValueError, "not observed on seed 41 validation"):
                module.register_sidecar(seed43, root / "registry.json")

            validation = self._build(
                self._args(
                    root,
                    seed=41,
                    total=1000,
                    role="validation",
                    output_name="validation.json",
                ),
                {"SLLM_DEPLOY_TARGET": "11"},
            )
            module.register_sidecar(validation, root / "registry.json")
            with self.assertRaisesRegex(ValueError, "not observed on seed 41 validation"):
                module.register_sidecar(seed43, root / "registry.json")

            matching_validation = self._build(
                self._args(
                    root,
                    seed=41,
                    total=1000,
                    role="validation",
                    output_name="matching-validation.json",
                ),
                {},
            )
            seed44 = self._build(
                self._args(
                    root,
                    seed=44,
                    total=4000,
                    role="heldout",
                    output_name="seed44.json",
                ),
                {},
            )
            module.register_sidecar(matching_validation, root / "registry.json")
            module.register_sidecar(seed43, root / "registry.json")
            registry = module.register_sidecar(seed44, root / "registry.json")
            heldout = registry["families"][seed43["configuration_family_id"]]["heldout"]
            self.assertEqual(heldout["seeds"], [43, 44])

    def test_formal_source_gate_ignores_untracked_and_allows_only_faas_manifest(self):
        with mock.patch.object(
            module,
            "_tracked_dirty_paths",
            side_effect=[[], ["configs/generated/lora_manifest_1000.json"]],
        ):
            status = module.require_formal_source_cleanliness(ROOT, MAIN_REPO)
        self.assertTrue(status["source_clean_for_formal"])
        self.assertFalse(status["untracked_files_checked"])

        with mock.patch.object(
            module,
            "_tracked_dirty_paths",
            side_effect=[["scripts/run_full_fair_round.sh"], []],
        ):
            with self.assertRaisesRegex(ValueError, "formal source gate failed"):
                module.require_formal_source_cleanliness(ROOT, MAIN_REPO)


if __name__ == "__main__":
    unittest.main()
