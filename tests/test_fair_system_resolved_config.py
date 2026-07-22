from __future__ import annotations

import importlib.util
import json
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
        trace_path: str | None = None,
        subset_path: str | None = None,
        systems: str = "slora faaslora",
        execution_order: str | None = None,
        output_name: str = "sidecar.json",
        bandwidth: str = "250",
        zipf: str = "",
        rotation: str = "",
        time_scale: str = "",
        formal: str = "0",
        campaign_kind: str = "v2_c5_matched_output",
        model_profile: str = "llama2_7b_main_v2_publicmix",
        generation_contract: str = "fixed_length_greedy_v1",
    ):
        root.mkdir(parents=True, exist_ok=True)
        if trace_path is None:
            trace = root / "trace.json"
            if not trace.exists():
                trace.write_text('{"requests": []}\n', encoding="utf-8")
            trace_path = str(trace)
        if subset_path is None:
            subset = root / "subset.json"
            if not subset.exists():
                subset.write_text("[]\n", encoding="utf-8")
            subset_path = str(subset)
        if execution_order is None:
            first = "slora" if seed == 42 or seed % 2 else "faaslora"
            if model_profile == "llama32_3b_main_modelscope":
                first = "faaslora" if first == "slora" else "slora"
            execution_order = f"{first} {'faaslora' if first == 'slora' else 'slora'}"
        return module.parse_args(
            [
                "--baselines-root",
                str(ROOT),
                "--main-repo",
                str(MAIN_REPO),
                "--model-profile",
                model_profile,
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
                "--formal-run",
                formal,
                "--trace-role",
                role,
                "--generation-contract",
                generation_contract,
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
                "--systems",
                systems,
                "--execution-order",
                execution_order,
                "--campaign-kind",
                campaign_kind,
                "--output",
                str(root / output_name),
                "--registry",
                str(root / "registry.json"),
            ]
        )

    def _build(self, args, env=None, worktrees=None):
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
        default_worktree = {
            "commit": "c" * 40,
            "tracked_dirty_paths": [],
            "has_tracked_patch": False,
            "head_to_worktree_binary_diff_sha256": module.hashlib.sha256(b"").hexdigest(),
            "head_to_worktree_binary_diff_bytes": 0,
            "staged_binary_diff_sha256": module.hashlib.sha256(b"").hexdigest(),
            "staged_binary_diff_bytes": 0,
            "unstaged_binary_diff_sha256": module.hashlib.sha256(b"").hexdigest(),
            "unstaged_binary_diff_bytes": 0,
            "dirty_tracked_files": {},
            "unmerged_entries": [],
            "untracked_files_included": False,
        }
        identities = worktrees or [default_worktree, default_worktree]
        with mock.patch.object(
            module, "_source_identity", return_value={"source": "a" * 64}
        ), mock.patch.object(
            module, "_git_commit", return_value="b" * 40
        ), mock.patch.object(
            module, "_git_worktree_identity", side_effect=identities
        ), mock.patch.object(
            module, "source_cleanliness", return_value=clean
        ):
            return module.build_sidecar(args, env or {})

    def _write_and_mark_validation(
        self,
        sidecar,
        registry_path: Path,
        *,
        status: str = "complete",
        manifest_name: str | None = None,
    ):
        sidecar_path = Path(sidecar["sidecar_path"])
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest_path = sidecar_path.parent / (
            manifest_name or f"{sidecar_path.stem}-MANIFEST.json"
        )
        manifest = {
            "status": status,
            "formal_run": True,
            "source_clean_for_formal": True,
            "trace_role": "validation",
            "sampling_seed": 41,
            "total_requests": 1000,
            "campaign_kind": sidecar.get("campaign_kind"),
            "model_profile": sidecar["model_profile"],
            "systems": sidecar["systems"],
            "execution_order": sidecar["full_run_identity"]["execution_order"],
            "shared_trace_path": sidecar["excluded_from_hash_audit"]["trace_path"],
            "shared_trace_sha256": sidecar["full_run_identity"]["trace_sha256"],
            "shared_adapter_subset_path": sidecar["excluded_from_hash_audit"][
                "adapter_subset_path"
            ],
            "shared_adapter_subset_sha256": sidecar["full_run_identity"][
                "adapter_subset_sha256"
            ],
            "system_resolved_config_path": str(sidecar_path.resolve()),
            "system_resolved_config_sidecar_sha256": module._file_sha256(
                sidecar_path
            ),
            "system_resolved_config_sidecar_bytes": sidecar_path.stat().st_size,
            "system_resolved_config_family_id": sidecar[
                "configuration_family_id"
            ],
            "system_resolved_config_sha256": sidecar[
                "system_resolved_config_sha256"
            ],
            "baseline_git": {
                "commit": sidecar["source_commits"]["baselines"],
                "tracked_dirty_paths": [],
            },
            "faaslora_git": {
                "commit": sidecar["source_commits"]["faaslora"],
                "tracked_dirty_paths": [],
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return module.mark_successful_validation(
            sidecar_path=sidecar_path,
            registry_path=registry_path,
            manifest_path=manifest_path,
        )

    def test_upstream_patch_identity_records_content_and_changes_system_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "upstream"
            repo.mkdir()
            module.subprocess.run(["git", "init", "-q", str(repo)], check=True)
            module.subprocess.run(
                ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
                check=True,
            )
            module.subprocess.run(
                ["git", "-C", str(repo), "config", "user.name", "test"],
                check=True,
            )
            source = repo / "runtime.py"
            source.write_text("value = 1\n", encoding="utf-8")
            module.subprocess.run(["git", "-C", str(repo), "add", "runtime.py"], check=True)
            module.subprocess.run(
                ["git", "-C", str(repo), "commit", "-q", "-m", "base"],
                check=True,
            )
            source.write_text("value = 2\n", encoding="utf-8")
            identity_a = module._git_worktree_identity(repo)
            self.assertEqual(identity_a["tracked_dirty_paths"], ["runtime.py"])
            self.assertTrue(identity_a["has_tracked_patch"])
            self.assertEqual(
                identity_a["dirty_tracked_files"]["runtime.py"]["content_sha256"],
                module.hashlib.sha256(source.read_bytes()).hexdigest(),
            )

            source.write_text("value = 3\n", encoding="utf-8")
            identity_b = module._git_worktree_identity(repo)
            self.assertNotEqual(
                identity_a["head_to_worktree_binary_diff_sha256"],
                identity_b["head_to_worktree_binary_diff_sha256"],
            )

            fixed = {
                "commit": "d" * 40,
                "tracked_dirty_paths": [],
                "has_tracked_patch": False,
                "head_to_worktree_binary_diff_sha256": module.hashlib.sha256(b"").hexdigest(),
                "head_to_worktree_binary_diff_bytes": 0,
                "staged_binary_diff_sha256": module.hashlib.sha256(b"").hexdigest(),
                "staged_binary_diff_bytes": 0,
                "unstaged_binary_diff_sha256": module.hashlib.sha256(b"").hexdigest(),
                "unstaged_binary_diff_bytes": 0,
                "dirty_tracked_files": {},
                "unmerged_entries": [],
                "untracked_files_included": False,
            }
            sidecar_a = self._build(
                self._args(root, seed=41, total=1000, role="validation"),
                worktrees=[identity_a, fixed],
            )
            sidecar_b = self._build(
                self._args(root, seed=41, total=1000, role="validation"),
                worktrees=[identity_b, fixed],
            )
            self.assertNotEqual(
                sidecar_a["system_resolved_config_sha256"],
                sidecar_b["system_resolved_config_sha256"],
            )

    def test_slora_patch_artifact_must_match_live_tracked_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baselines_root = root / "baselines"
            patch_dir = baselines_root / "patches"
            patch_dir.mkdir(parents=True)
            repo = root / "S-LoRA"
            repo.mkdir()
            module.subprocess.run(["git", "init", "-q", str(repo)], check=True)
            module.subprocess.run(
                ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
                check=True,
            )
            module.subprocess.run(
                ["git", "-C", str(repo), "config", "user.name", "test"],
                check=True,
            )
            source = repo / "runtime.py"
            source.write_text("value = 1\n", encoding="utf-8")
            module.subprocess.run(["git", "-C", str(repo), "add", "runtime.py"], check=True)
            module.subprocess.run(
                ["git", "-C", str(repo), "commit", "-q", "-m", "base"],
                check=True,
            )
            source.write_text("value = 2\n", encoding="utf-8")
            diff = module._git_output(repo, ["diff", "--binary"])
            artifact = patch_dir / "S-LoRA_local_changes.patch.gz"
            artifact.write_bytes(module.gzip.compress(diff, mtime=0))

            identity = module._slora_patch_artifact_identity(baselines_root, repo)
            self.assertTrue(identity["matches_live_worktree"])
            self.assertEqual(
                identity["decompressed_diff_sha256"],
                module.hashlib.sha256(diff).hexdigest(),
            )

            source.write_text("value = 3\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "worktree drifted"):
                module._slora_patch_artifact_identity(baselines_root, repo)

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
                "FAASLORA_WORKLOAD_SEED": "99",
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
                    formal="1",
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
                    formal="1",
                ),
                {"SLLM_DEPLOY_TARGET": "9"},
            )
            seed43 = self._build(
                self._args(
                    root, seed=43, total=4000, role="heldout", formal="1"
                ),
                {},
            )
            seed44 = self._build(
                self._args(
                    root,
                    seed=44,
                    total=4000,
                    role="heldout",
                    output_name="seed44/protocol/system_resolved_config.json",
                    formal="1",
                ),
                {"SLLM_DEPLOY_TARGET": "9"},
            )
            registry_path = root / "registry.json"
            module.register_sidecar(validation_a, registry_path)
            module.register_sidecar(validation_b, registry_path)
            self._write_and_mark_validation(validation_a, registry_path)
            with self.assertRaisesRegex(ValueError, "already selected"):
                self._write_and_mark_validation(validation_b, registry_path)
            module.register_sidecar(seed43, root / "registry.json")
            with self.assertRaisesRegex(ValueError, "not completed successfully"):
                module.register_sidecar(seed44, root / "registry.json")

    def test_heldout_registry_requires_seed41_validation_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed43 = self._build(
                self._args(
                    root, seed=43, total=4000, role="heldout", formal="1"
                ),
                {},
            )
            with self.assertRaisesRegex(ValueError, "not completed successfully"):
                module.register_sidecar(seed43, root / "registry.json")

            validation = self._build(
                self._args(
                    root,
                    seed=41,
                    total=1000,
                    role="validation",
                    output_name="validation.json",
                    formal="1",
                ),
                {"SLLM_DEPLOY_TARGET": "11"},
            )
            registry_path = root / "registry.json"
            module.register_sidecar(validation, registry_path)
            self._write_and_mark_validation(validation, registry_path)
            with self.assertRaisesRegex(ValueError, "not completed successfully"):
                module.register_sidecar(seed43, root / "registry.json")

            matching_root = root / "matching"
            matching_registry = matching_root / "registry.json"
            matching_validation = self._build(
                self._args(
                    matching_root,
                    seed=41,
                    total=1000,
                    role="validation",
                    output_name="matching-validation.json",
                    formal="1",
                ),
                {},
            )
            seed44 = self._build(
                self._args(
                    matching_root,
                    seed=44,
                    total=4000,
                    role="heldout",
                    output_name="seed44/protocol/system_resolved_config.json",
                    formal="1",
                ),
                {},
            )
            seed43_matching = self._build(
                self._args(
                    matching_root,
                    seed=43,
                    total=4000,
                    role="heldout",
                    output_name="seed43/protocol/system_resolved_config.json",
                    formal="1",
                ),
                {},
            )
            module.register_sidecar(matching_validation, matching_registry)
            self._write_and_mark_validation(matching_validation, matching_registry)
            module.register_sidecar(seed43_matching, matching_registry)
            registry = module.register_sidecar(seed44, matching_registry)
            heldout = registry["families"][seed43_matching["configuration_family_id"]][
                "heldout"
            ]
            self.assertEqual(heldout["seeds"], [43, 44])

    def test_registered_but_incomplete_validation_does_not_unlock_heldout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "registry.json"
            validation = self._build(
                self._args(
                    root,
                    seed=41,
                    total=1000,
                    role="validation",
                    output_name="validation.json",
                    formal="1",
                ),
                {},
            )
            heldout = self._build(
                self._args(
                    root, seed=43, total=4000, role="heldout", formal="1"
                ),
                {},
            )
            module.register_sidecar(validation, registry_path)
            with self.assertRaisesRegex(ValueError, "not completed successfully"):
                module.register_sidecar(heldout, registry_path)
            with self.assertRaisesRegex(ValueError, "status=complete"):
                self._write_and_mark_validation(
                    validation, registry_path, status="incomplete"
                )
            registry = self._write_and_mark_validation(validation, registry_path)
            successes = registry["families"][validation["configuration_family_id"]][
                "successful_validation"
            ]
            self.assertEqual(len(successes), 1)
            module.register_sidecar(heldout, registry_path)

    def test_successful_validation_rejects_manifest_sidecar_sha_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "registry.json"
            validation = self._build(
                self._args(
                    root,
                    seed=41,
                    total=1000,
                    role="validation",
                    output_name="validation.json",
                    formal="1",
                ),
                {},
            )
            module.register_sidecar(validation, registry_path)
            sidecar_path = Path(validation["sidecar_path"])
            sidecar_path.write_text(json.dumps(validation), encoding="utf-8")
            manifest_path = root / "MANIFEST.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "formal_run": True,
                        "source_clean_for_formal": True,
                        "trace_role": "validation",
                        "sampling_seed": 41,
                        "campaign_kind": validation.get("campaign_kind"),
                        "system_resolved_config_path": str(sidecar_path.resolve()),
                        "system_resolved_config_sidecar_sha256": "0" * 64,
                        "system_resolved_config_family_id": validation[
                            "configuration_family_id"
                        ],
                        "system_resolved_config_sha256": validation[
                            "system_resolved_config_sha256"
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sidecar SHA"):
                module.mark_successful_validation(
                    sidecar_path=sidecar_path,
                    registry_path=registry_path,
                    manifest_path=manifest_path,
                )

    def test_heldout_revalidates_seed41_bytes_and_writes_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "registry.json"
            validation = self._build(
                self._args(
                    root,
                    seed=41,
                    total=1000,
                    role="validation",
                    output_name="validation.json",
                    formal="1",
                ),
                {},
            )
            heldout43 = self._build(
                self._args(
                    root,
                    seed=43,
                    total=4000,
                    role="heldout",
                    output_name="heldout43/protocol/system_resolved_config.json",
                    formal="1",
                ),
                {},
            )
            heldout44 = self._build(
                self._args(
                    root,
                    seed=44,
                    total=4000,
                    role="heldout",
                    output_name="heldout44/protocol/system_resolved_config.json",
                    formal="1",
                ),
                {},
            )
            module.register_sidecar(validation, registry_path)
            self._write_and_mark_validation(validation, registry_path)
            module.register_sidecar(heldout43, registry_path)

            evidence_path = (
                root / "heldout43" / "protocol" / "seed41_validation_evidence.json"
            )
            self.assertTrue(evidence_path.is_file())
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(
                evidence["schema_version"],
                module.VALIDATION_EVIDENCE_SCHEMA_VERSION,
            )
            self.assertEqual(evidence["heldout"]["sampling_seed"], 43)
            self.assertEqual(evidence["seed41_validation"]["sampling_seed"], 41)
            self.assertEqual(evidence["seed41_validation"]["total_requests"], 1000)

            family = json.loads(registry_path.read_text(encoding="utf-8"))[
                "families"
            ][validation["configuration_family_id"]]
            manifest_path = Path(family["successful_validation"][0]["manifest"])
            manifest_path.write_text(
                manifest_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "manifest bytes"):
                module.register_sidecar(heldout44, registry_path)
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(
                registry["families"][validation["configuration_family_id"]][
                    "heldout"
                ]["seeds"],
                [43],
            )

    def test_heldout_rejects_missing_selected_validation_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / "registry.json"
            validation = self._build(
                self._args(
                    root,
                    seed=41,
                    total=1000,
                    role="validation",
                    output_name="validation.json",
                    formal="1",
                ),
                {},
            )
            heldout = self._build(
                self._args(
                    root,
                    seed=43,
                    total=4000,
                    role="heldout",
                    output_name="heldout.json",
                    formal="1",
                ),
                {},
            )
            module.register_sidecar(validation, registry_path)
            self._write_and_mark_validation(validation, registry_path)
            Path(validation["sidecar_path"]).unlink()
            with self.assertRaisesRegex(ValueError, "validation sidecar is missing"):
                module.register_sidecar(heldout, registry_path)

    def test_resume_sidecar_is_immutable_for_config_and_full_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(
                root,
                seed=41,
                total=1000,
                role="validation",
                output_name="protocol/system_resolved_config.json",
                formal="1",
            )
            original = self._build(args, {})
            path = Path(original["sidecar_path"])
            module.write_or_validate_immutable_sidecar(path, original)
            original_bytes = path.read_bytes()
            self.assertEqual(
                module.write_or_validate_immutable_sidecar(path, original), original
            )
            self.assertEqual(path.read_bytes(), original_bytes)

            config_drift = self._build(args, {"SLLM_DEPLOY_TARGET": "9"})
            with self.assertRaisesRegex(ValueError, "resume identity drift"):
                module.write_or_validate_immutable_sidecar(path, config_drift)
            self.assertEqual(path.read_bytes(), original_bytes)

            full_drift_args = self._args(
                root,
                seed=41,
                total=1000,
                role="validation",
                output_name="protocol/system_resolved_config.json",
                run_tag="different-run-tag",
                formal="1",
            )
            full_drift = self._build(full_drift_args, {})
            with self.assertRaisesRegex(ValueError, "resume identity drift"):
                module.write_or_validate_immutable_sidecar(path, full_drift)
            self.assertEqual(path.read_bytes(), original_bytes)

    def test_campaign_protocol_enforces_exact_systems_contract_and_order(self):
        main = module.validate_campaign_protocol(
            campaign_kind="v2_full_vs_serverless",
            formal_run=True,
            trace_role="heldout",
            model_profile="llama2_7b_main_v2_publicmix",
            sampling_seed=43,
            faaslora_scenario="v2_full",
            systems=["serverlessllm", "faaslora"],
            execution_order=["faaslora", "serverlessllm"],
            generation_contract="legacy",
        )
        self.assertEqual(
            main["expected_execution_order"], ["faaslora", "serverlessllm"]
        )
        c5 = module.validate_campaign_protocol(
            campaign_kind="v2_c5_matched_output",
            formal_run=True,
            trace_role="heldout",
            model_profile="llama32_3b_main_modelscope",
            sampling_seed=44,
            faaslora_scenario="v2_full",
            systems=["slora", "faaslora"],
            execution_order=["slora", "faaslora"],
            generation_contract="fixed_length_greedy_v1",
        )
        self.assertEqual(c5["expected_execution_order"], ["slora", "faaslora"])

        with self.assertRaisesRegex(ValueError, "exact systems"):
            module.validate_campaign_protocol(
                campaign_kind="v2_c5_matched_output",
                formal_run=True,
                trace_role="heldout",
                model_profile="llama2_7b_main_v2_publicmix",
                sampling_seed=43,
                faaslora_scenario="v2_full",
                systems=["slora", "faaslora", "serverlessllm"],
                execution_order=["slora", "faaslora", "serverlessllm"],
                generation_contract="fixed_length_greedy_v1",
            )
        with self.assertRaisesRegex(ValueError, "execution order mismatch"):
            module.validate_campaign_protocol(
                campaign_kind="v2_full_vs_serverless",
                formal_run=True,
                trace_role="heldout",
                model_profile="llama2_7b_main_v2_publicmix",
                sampling_seed=43,
                faaslora_scenario="v2_full",
                systems=["serverlessllm", "faaslora"],
                execution_order=["serverlessllm", "faaslora"],
                generation_contract="legacy",
            )
        with self.assertRaisesRegex(ValueError, "requires FAIR_CAMPAIGN_KIND"):
            module.validate_campaign_protocol(
                campaign_kind="",
                formal_run=True,
                trace_role="heldout",
                model_profile="llama2_7b_main_v2_publicmix",
                sampling_seed=43,
                faaslora_scenario="v2_full",
                systems=["faaslora"],
                execution_order=["faaslora"],
                generation_contract="legacy",
            )

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
