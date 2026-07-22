from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import faaslora_ablation_validation_registry as registry


class FaaSLoRAAblationValidationRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "config", "user.email", "test@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "config", "user.name", "Test"],
            check=True,
        )
        self.config = self.repo / "configs" / "experiments.yaml"
        self.config.parent.mkdir(parents=True)
        self.config.write_text("profiles: {}\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(self.repo), "add", "configs/experiments.yaml"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-qm", "fixture"],
            check=True,
        )
        self.commit = subprocess.check_output(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True
        ).strip()
        self.family = registry.configuration_family(
            model_profile="llama2_7b_main_v2_publicmix",
            dataset_profile="azure_sharegpt_rep4000",
            workload_profile="llama2_7b_auto500_formal4000_s8",
            selected_num_adapters=500,
            gpu_ids="0,1,2,3",
            generation_contract="legacy",
        )
        self.registry_path = self.root / "protocol" / "registry.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _manifest(
        self,
        name: str,
        frozen_hash: str,
        **overrides: object,
    ) -> Path:
        path = self.root / name / "MANIFEST.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "complete",
            "formal_run": True,
            "trace_role": "validation",
            "scenarios": ["v2_full"],
            "shared_trace": {"sampling_seed": 41, "requests": 1000},
            "code_snapshot": {
                "source_clean_for_formal": True,
                "git_commit": self.commit,
            },
            "config_snapshot": {
                "path": str(self.config.resolve()),
                "sha256": self._sha(self.config),
                "bytes": self.config.stat().st_size,
            },
            "configuration_family": self.family,
            "non_feature_frozen_config_sha256": frozen_hash,
            "non_feature_frozen_config_consistent": True,
        }
        payload.update(overrides)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    def _register(self, manifest: Path) -> dict:
        return registry.register_successful_validation(
            registry_path=self.registry_path,
            manifest_path=manifest,
            repo=self.repo,
            config_path=self.config,
            family=self.family,
        )

    def _resolve(
        self,
        *,
        seed: int = 43,
        requests: int = 4000,
        expected: str = "",
        family: dict | None = None,
        evidence_name: str = "evidence.json",
    ) -> dict:
        return registry.resolve_heldout_validation(
            registry_path=self.registry_path,
            evidence_path=self.root / evidence_name,
            repo=self.repo,
            config_path=self.config,
            family=family or self.family,
            sampling_seed=seed,
            total_requests=requests,
            round_dir=self.root / f"heldout_seed{seed}",
            expected_non_feature_sha256=expected,
        )

    def test_incomplete_or_unregistered_validation_cannot_unlock_heldout(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing validation registry"):
            self._resolve()
        incomplete = self._manifest(
            "incomplete", "a" * 64, status="incomplete"
        )
        with self.assertRaisesRegex(ValueError, "status=complete"):
            self._register(incomplete)
        self.assertFalse(self.registry_path.exists())

    def test_complete_seed41_validation_registers_and_freezes_heldout(self) -> None:
        validation = self._manifest("valid", "a" * 64)
        self._register(validation)
        evidence = self._resolve(seed=43)
        self.assertEqual(
            evidence["selected_non_feature_frozen_config_sha256"], "a" * 64
        )
        self.assertEqual(evidence["successful_validation"]["manifest"], str(validation))
        self.assertEqual(evidence["heldout_seed"], 43)
        stored = json.loads(self.registry_path.read_text(encoding="utf-8"))
        only_family = next(iter(stored["families"].values()))
        self.assertEqual(
            only_family["heldout_frozen_non_feature_sha256"], "a" * 64
        )

    def test_source_drift_is_rejected(self) -> None:
        validation = self._manifest("valid", "a" * 64)
        self._register(validation)

        drift = self.repo / "source.txt"
        drift.write_text("new commit\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(self.repo), "add", "source.txt"], check=True
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-qm", "drift"], check=True
        )
        with self.assertRaisesRegex(ValueError, "current source commit/config SHA"):
            self._resolve()

    def test_config_drift_is_rejected(self) -> None:
        validation = self._manifest("valid", "a" * 64)
        self._register(validation)
        self.config.write_text("profiles:\n  changed: true\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "current source commit/config SHA"):
            self._resolve()

    def test_registered_validation_manifest_byte_drift_is_rejected(self) -> None:
        validation = self._manifest("valid", "a" * 64)
        self._register(validation)
        validation.write_text(
            validation.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "byte count changed"):
            self._resolve()

    def test_multiple_hashes_require_selection_and_frozen_hash_cannot_switch(self) -> None:
        self._register(self._manifest("candidate_a", "a" * 64))
        self._register(self._manifest("candidate_b", "b" * 64))
        with self.assertRaisesRegex(ValueError, "multiple successful validation hashes"):
            self._resolve()
        selected = self._resolve(expected="a" * 64)
        self.assertEqual(
            selected["selected_non_feature_frozen_config_sha256"], "a" * 64
        )
        with self.assertRaisesRegex(ValueError, "already frozen hash"):
            self._resolve(seed=44, expected="b" * 64, evidence_name="seed44.json")

    def test_existing_per_round_evidence_is_immutable_as_registry_grows(self) -> None:
        self._register(self._manifest("valid", "a" * 64))
        evidence_path = self.root / "evidence.json"
        self._resolve(seed=43)
        original_bytes = evidence_path.read_bytes()
        self._resolve(seed=44, evidence_name="seed44.json")
        resumed = self._resolve(seed=43)
        self.assertEqual(evidence_path.read_bytes(), original_bytes)
        self.assertEqual(resumed, json.loads(original_bytes))

        tampered = json.loads(evidence_path.read_text(encoding="utf-8"))
        tampered["heldout_requests"] = 3999
        evidence_path.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from the requested run"):
            self._resolve(seed=43)

    def test_family_seed_request_and_validation_contract_are_fail_closed(self) -> None:
        cases = (
            ({"scenarios": ["v2_elastic_only"]}, "exactly the v2_full"),
            ({"shared_trace": {"sampling_seed": 42, "requests": 1000}}, "seed 41"),
            ({"shared_trace": {"sampling_seed": 41, "requests": 999}}, "1,000"),
            ({"configuration_family": {**self.family, "selected_num_adapters": 499}}, "family mismatch"),
        )
        for index, (override, message) in enumerate(cases):
            with self.subTest(message=message):
                manifest = self._manifest(f"bad_{index}", "a" * 64, **override)
                with self.assertRaisesRegex(ValueError, message):
                    self._register(manifest)

        self._register(self._manifest("valid", "a" * 64))
        with self.assertRaisesRegex(ValueError, "seeds 43/44/45"):
            self._resolve(seed=42)
        with self.assertRaisesRegex(ValueError, "4,000 requests"):
            self._resolve(requests=3999)
        wrong_family = {**self.family, "workload_profile": "other"}
        with self.assertRaisesRegex(ValueError, "no successful seed-41 validation"):
            self._resolve(family=wrong_family)


if __name__ == "__main__":
    unittest.main()
