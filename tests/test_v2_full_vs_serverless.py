from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from scripts import analyze_v2_full_vs_serverless as analyzer
from scripts import plot_paper_figures


def _run(model: str, seed: int, system: str) -> analyzer.FormalRun:
    multiplier = 0.8 if system == "faaslora" else 1.0
    e2e_avg_ms = 500.0 * multiplier + seed
    cost_req_usd = 0.01 * multiplier
    metrics = {
        "ttft_avg_ms": 100.0 * multiplier + seed,
        "ttft_p95_ms": 150.0 * multiplier + seed,
        "e2e_avg_ms": e2e_avg_ms,
        "e2e_p95_ms": 650.0 * multiplier + seed,
        "tpot_avg_ms": 12.0 * multiplier,
        "tpot_p95_ms": 18.0 * multiplier,
        "tok_s": 120.0 / multiplier,
        "cost_req_usd": cost_req_usd,
        "ce": 1.0 / ((e2e_avg_ms / 1000.0) * cost_req_usd),
    }
    model_char = "a" if model == "llama2_7b" else "b"
    return analyzer.FormalRun(
        model=model,
        seed=seed,
        system=system,
        scenario=analyzer.FORMAL_SCENARIOS[system],
        source=Path(f"/{model}/seed{seed}/{system}.json"),
        metadata_sampling_seed=seed,
        total=4000,
        completed=4000,
        generation_contract="legacy",
        selected_num_adapters=500,
        bandwidth_mib_s=250.0,
        configured_time_scale_factor=8.0,
        effective_time_scale_factor=8.0,
        zipf_exponent=1.0,
        active_adapter_cap=48,
        hotset_rotation_requests=500,
        hotset_rotation_mode="legacy",
        hotset_overlap_fraction=0.75,
        trace_sha256=(model_char * 62) + f"{seed % 100:02d}",
        adapter_subset_sha256=(model_char * 63) + "e",
        system_resolved_config_sha256=(model_char + "c") * 32,
        metrics=metrics,
    )


def _matrix() -> list[analyzer.FormalRun]:
    return [
        _run(model, seed, system)
        for model in analyzer.FORMAL_MODELS
        for seed in analyzer.FORMAL_SEEDS
        for system in analyzer.FORMAL_SYSTEMS
    ]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _execution_order(model: str, seed: int) -> list[str]:
    if seed in analyzer.FORMAL_SEEDS:
        return list(analyzer.FORMAL_EXECUTION_ORDERS[(model, seed)])
    if model == "llama2_7b":
        return ["faaslora", "serverlessllm"]
    return ["serverlessllm", "faaslora"]


def _write_seed41_validation(
    root: Path,
    *,
    model: str,
    config_sha: str,
    family_id: str,
    source_commits: dict[str, str],
) -> tuple[Path, Path, Path, Path]:
    round_dir = root / "validation" / model
    trace_path = round_dir / "shared_artifacts" / "trace.json"
    subset_path = round_dir / "shared_artifacts" / "subset.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text('{"requests": []}\n', encoding="utf-8")
    subset_path.write_text("[]\n", encoding="utf-8")
    trace_sha = _file_sha256(trace_path)
    subset_sha = _file_sha256(subset_path)
    order = _execution_order(model, 41)
    profile = analyzer.FORMAL_MODEL_PROFILES[model]
    identity = {
        "system_resolved_config_sha256": config_sha,
        "model_profile": profile,
        "total_requests": 1000,
        "sampling_seed": 41,
        "trace_sha256": trace_sha,
        "adapter_subset_sha256": subset_sha,
        "systems": list(analyzer.FORMAL_SYSTEMS),
        "execution_order": order,
        "source_commits": source_commits,
    }
    sidecar_path = round_dir / "protocol" / "system_resolved_config.json"
    _write_json(
        sidecar_path,
        {
            "formal_run": True,
            "trace_role": "validation",
            "sampling_seed": 41,
            "campaign_kind": analyzer.FORMAL_CAMPAIGN_KIND,
            "model_profile": profile,
            "systems": list(analyzer.FORMAL_SYSTEMS),
            "source_commits": source_commits,
            "configuration_family_id": family_id,
            "system_resolved_config_sha256": config_sha,
            "full_run_identity": identity,
            "full_run_identity_sha256": analyzer._canonical_sha256(identity),
        },
    )
    manifest_path = round_dir / "MANIFEST.json"
    _write_json(
        manifest_path,
        {
            "status": "complete",
            "formal_run": True,
            "source_clean_for_formal": True,
            "trace_role": "validation",
            "sampling_seed": 41,
            "total_requests": 1000,
            "campaign_kind": analyzer.FORMAL_CAMPAIGN_KIND,
            "model_profile": profile,
            "systems": list(analyzer.FORMAL_SYSTEMS),
            "execution_order": order,
            "system_resolved_config_family_id": family_id,
            "system_resolved_config_sha256": config_sha,
            "system_resolved_config_path": str(sidecar_path.resolve()),
            "system_resolved_config_sidecar_bytes": sidecar_path.stat().st_size,
            "system_resolved_config_sidecar_sha256": _file_sha256(sidecar_path),
            "shared_trace_path": str(trace_path.resolve()),
            "shared_trace_sha256": trace_sha,
            "shared_adapter_subset_path": str(subset_path.resolve()),
            "shared_adapter_subset_sha256": subset_sha,
            "baseline_git": {"commit": source_commits["baselines"]},
            "faaslora_git": {"commit": source_commits["faaslora"]},
        },
    )
    return manifest_path, sidecar_path, trace_path, subset_path


def _protocol_matrix(
    root: Path,
) -> tuple[
    list[analyzer.FormalRun],
    dict[Path, Path],
    dict[tuple[str, int], Path],
    dict[tuple[str, int], Path],
]:
    runs: list[analyzer.FormalRun] = []
    manifest_by_source: dict[Path, Path] = {}
    manifests: dict[tuple[str, int], Path] = {}
    sidecars: dict[tuple[str, int], Path] = {}
    source_commits = {"baselines": "1" * 40, "faaslora": "2" * 40}
    base = _matrix()
    by_identity = {(run.model, run.seed, run.system): run for run in base}
    for model in analyzer.FORMAL_MODELS:
        config_sha = by_identity[(model, 43, "faaslora")].system_resolved_config_sha256
        family_id = hashlib.sha256(f"family-{model}".encode()).hexdigest()
        validation_manifest, validation_sidecar, trace_path, subset_path = (
            _write_seed41_validation(
                root,
                model=model,
                config_sha=config_sha,
                family_id=family_id,
                source_commits=source_commits,
            )
        )
        validation_order = _execution_order(model, 41)
        validation_evidence = {
            "sampling_seed": 41,
            "total_requests": 1000,
            "sidecar_path": str(validation_sidecar.resolve()),
            "sidecar_bytes": validation_sidecar.stat().st_size,
            "sidecar_sha256": _file_sha256(validation_sidecar),
            "manifest_path": str(validation_manifest.resolve()),
            "manifest_bytes": validation_manifest.stat().st_size,
            "manifest_sha256": _file_sha256(validation_manifest),
            "campaign_kind": analyzer.FORMAL_CAMPAIGN_KIND,
            "model_profile": analyzer.FORMAL_MODEL_PROFILES[model],
            "configuration_family_id": family_id,
            "system_resolved_config_sha256": config_sha,
            "systems": list(analyzer.FORMAL_SYSTEMS),
            "execution_order": validation_order,
            "source_commits": source_commits,
            "shared_trace_path": str(trace_path.resolve()),
            "shared_trace_sha256": _file_sha256(trace_path),
            "shared_adapter_subset_path": str(subset_path.resolve()),
            "shared_adapter_subset_sha256": _file_sha256(subset_path),
        }
        for seed in analyzer.FORMAL_SEEDS:
            round_dir = root / "heldout" / f"{model}-seed{seed}"
            order = _execution_order(model, seed)
            model_runs: list[analyzer.FormalRun] = []
            for system in analyzer.FORMAL_SYSTEMS:
                source = round_dir / "raw" / system / f"{system}_result.json"
                _write_json(source, {})
                run = replace(
                    by_identity[(model, seed, system)], source=source.resolve()
                )
                runs.append(run)
                model_runs.append(run)
            identity = {
                "system_resolved_config_sha256": config_sha,
                "model_profile": analyzer.FORMAL_MODEL_PROFILES[model],
                "total_requests": 4000,
                "selected_num_adapters": 500,
                "sampling_seed": seed,
                "trace_sha256": model_runs[0].trace_sha256,
                "adapter_subset_sha256": model_runs[0].adapter_subset_sha256,
                "systems": list(analyzer.FORMAL_SYSTEMS),
                "execution_order": order,
                "generation_contract": "legacy",
                "storage_bandwidth_mib_s": 250.0,
                "time_scale_factor": 8.0,
                "workload_overrides": {
                    "zipf_exponent": 1.0,
                    "active_adapter_cap": 48,
                    "hotset_rotation_requests": 500,
                    "hotset_rotation_mode": "legacy",
                    "hotset_overlap_fraction": 0.75,
                },
                "faaslora_scenario": "v2_full",
                "campaign_kind": analyzer.FORMAL_CAMPAIGN_KIND,
                "source_commits": source_commits,
            }
            identity_sha = analyzer._canonical_sha256(identity)
            sidecar_path = round_dir / "protocol" / "system_resolved_config.json"
            _write_json(
                sidecar_path,
                {
                    "formal_run": True,
                    "trace_role": "heldout",
                    "sampling_seed": seed,
                    "campaign_kind": analyzer.FORMAL_CAMPAIGN_KIND,
                    "model_profile": analyzer.FORMAL_MODEL_PROFILES[model],
                    "systems": list(analyzer.FORMAL_SYSTEMS),
                    "source_commits": source_commits,
                    "configuration_family_id": family_id,
                    "system_resolved_config_sha256": config_sha,
                    "full_run_identity": identity,
                    "full_run_identity_sha256": identity_sha,
                },
            )
            evidence_path = round_dir / "protocol" / "seed41_validation_evidence.json"
            _write_json(
                evidence_path,
                {
                    "schema_version": "eurosys27_v2_seed41_validation_evidence_v1",
                    "heldout": {
                        "sampling_seed": seed,
                        "campaign_kind": analyzer.FORMAL_CAMPAIGN_KIND,
                        "model_profile": analyzer.FORMAL_MODEL_PROFILES[model],
                        "configuration_family_id": family_id,
                        "system_resolved_config_sha256": config_sha,
                        "full_run_identity_sha256": identity_sha,
                        "source_commits": source_commits,
                    },
                    "seed41_validation": validation_evidence,
                },
            )
            manifest_path = round_dir / "MANIFEST.json"
            _write_json(
                manifest_path,
                {
                    "campaign_kind": analyzer.FORMAL_CAMPAIGN_KIND,
                    "model_profile": analyzer.FORMAL_MODEL_PROFILES[model],
                    "sampling_seed": seed,
                    "total_requests": 4000,
                    "selected_num_adapters": 500,
                    "systems": list(analyzer.FORMAL_SYSTEMS),
                    "supported_systems": list(analyzer.FORMAL_SYSTEMS),
                    "execution_order": order,
                    "generation_contract": "legacy",
                    "bandwidth_mib_s": 250.0,
                    "faaslora_scenario": "v2_full",
                    "workload_overrides": dict(identity["workload_overrides"]),
                    "shared_trace_sha256": model_runs[0].trace_sha256,
                    "shared_adapter_subset_sha256": model_runs[0].adapter_subset_sha256,
                    "system_resolved_config_family_id": family_id,
                    "system_resolved_config_sha256": config_sha,
                    "full_run_identity_sha256": identity_sha,
                    "system_resolved_config_path": str(sidecar_path.resolve()),
                    "system_resolved_config_sidecar_bytes": sidecar_path.stat().st_size,
                    "system_resolved_config_sidecar_sha256": _file_sha256(sidecar_path),
                    "seed41_validation_evidence_path": str(evidence_path.resolve()),
                    "seed41_validation_evidence_bytes": evidence_path.stat().st_size,
                    "seed41_validation_evidence_sha256": _file_sha256(evidence_path),
                    "baseline_git": {"commit": source_commits["baselines"]},
                    "faaslora_git": {"commit": source_commits["faaslora"]},
                },
            )
            manifests[(model, seed)] = manifest_path
            sidecars[(model, seed)] = sidecar_path
            for run in model_runs:
                manifest_by_source[run.source.resolve()] = manifest_path.resolve()
    return runs, manifest_by_source, manifests, sidecars


class FullVsServerlessFormalTests(unittest.TestCase):
    def test_exact_matrix_and_output(self) -> None:
        runs = _matrix()
        self.assertEqual(
            analyzer.FORMAL_SCENARIOS["serverlessllm"],
            "serverlessllm_fair",
        )
        analyzer.validate_formal_matrix(runs)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new-output"
            analyzer.write_outputs(
                runs,
                output,
                inputs=[Path("campaign")],
                manifest_paths=[Path("campaign/MANIFEST.json")],
            )
            with (output / "full_vs_serverless_per_run.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                per_run = list(csv.DictReader(handle))
            with (output / "full_vs_serverless_paired_per_seed.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                paired = list(csv.DictReader(handle))
            with (output / "full_vs_serverless_paired_summary.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(len(per_run), 12)
            self.assertEqual(len(paired), 54)
            self.assertEqual(len(summary), 18)
            self.assertEqual({row["paired_seed_count"] for row in summary}, {"3"})
            for field in (
                "ttft_avg_ms",
                "e2e_avg_ms",
                "tpot_avg_ms",
                "tok_s",
                "cost_req_usd",
                "ce",
            ):
                self.assertIn(field, per_run[0])
            with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                analyzer.write_outputs(
                    runs,
                    output,
                    inputs=[],
                    manifest_paths=[],
                )

    def test_missing_extra_and_duplicate_identities_are_rejected(self) -> None:
        runs = _matrix()
        with self.assertRaisesRegex(SystemExit, "missing="):
            analyzer.validate_formal_matrix(runs[:-1])
        with self.assertRaisesRegex(SystemExit, "extra="):
            analyzer.validate_formal_matrix(
                [*runs, replace(runs[0], seed=42, source=Path("/extra.json"))]
            )
        with self.assertRaisesRegex(SystemExit, "duplicate="):
            analyzer.validate_formal_matrix([*runs, runs[0]])

    def test_wrong_axes_scenario_and_generation_contract_are_rejected(self) -> None:
        runs = _matrix()
        for replacement, pattern in (
            ({"bandwidth_mib_s": 119.2093}, "wrong bandwidth_mib_s"),
            ({"hotset_rotation_mode": "abrupt"}, "wrong hotset_rotation_mode"),
            ({"hotset_overlap_fraction": 0.0}, "wrong hotset_overlap_fraction"),
            ({"selected_num_adapters": 100}, "wrong selected_num_adapters"),
            (
                {"generation_contract": "fixed_length_greedy_v1"},
                "wrong generation_contract",
            ),
            ({"scenario": "faaslora_full"}, "must use scenario=v2_full"),
        ):
            with self.subTest(replacement=replacement):
                with self.assertRaisesRegex(SystemExit, pattern):
                    analyzer.validate_formal_matrix(
                        [replace(runs[0], **replacement), *runs[1:]]
                    )

    def test_config_drift_and_pair_hash_mismatch_are_rejected(self) -> None:
        runs = _matrix()
        with self.assertRaisesRegex(SystemExit, "frozen configuration drift"):
            analyzer.validate_formal_matrix(
                [
                    replace(runs[0], system_resolved_config_sha256="f" * 64),
                    *runs[1:],
                ]
            )
        with self.assertRaisesRegex(SystemExit, "trace/subset mismatch"):
            analyzer.validate_formal_matrix(
                [replace(runs[0], trace_sha256="f" * 64), *runs[1:]]
            )

    def test_result_seed_and_recomputed_ce_are_enforced(self) -> None:
        runs = _matrix()
        with self.assertRaisesRegex(SystemExit, "result seed mismatch"):
            analyzer.validate_formal_matrix(
                [replace(runs[0], metadata_sampling_seed=44), *runs[1:]]
            )
        wrong_metrics = dict(runs[0].metrics)
        wrong_metrics["ce"] = 1e99
        with self.assertRaisesRegex(SystemExit, "reported CE does not match"):
            analyzer.validate_formal_matrix(
                [replace(runs[0], metrics=wrong_metrics), *runs[1:]]
            )

    def test_campaign_protocol_accepts_exact_shared_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs, manifests_by_source, _, _ = _protocol_matrix(Path(tmp))
            analyzer.validate_formal_matrix(runs)
            analyzer.validate_formal_campaign_protocol(runs, manifests_by_source)

    def test_wrong_campaign_order_and_manifest_seed_are_rejected(self) -> None:
        mutations = (
            ("campaign_kind", "v2_c5_matched_output", "campaign_kind"),
            ("execution_order", ["serverlessllm", "faaslora"], "execution_order"),
            ("sampling_seed", 44, "manifest seed mismatch"),
        )
        for field, value, pattern in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                runs, manifest_by_source, manifests, _ = _protocol_matrix(Path(tmp))
                path = manifests[("llama2_7b", 43)]
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload[field] = value
                _write_json(path, payload)
                with self.assertRaisesRegex(SystemExit, pattern):
                    analyzer.validate_formal_campaign_protocol(runs, manifest_by_source)

    def test_pair_must_share_manifest_and_resolved_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs, manifest_by_source, manifests, _ = _protocol_matrix(Path(tmp))
            serverless = next(
                run
                for run in runs
                if run.model == "llama2_7b"
                and run.seed == 43
                and run.system == "serverlessllm"
            )
            manifest_by_source[serverless.source.resolve()] = manifests[
                ("llama2_7b", 44)
            ].resolve()
            with self.assertRaisesRegex(
                SystemExit, "must share one fair-round manifest"
            ):
                analyzer.validate_formal_campaign_protocol(runs, manifest_by_source)

        runs = _matrix()
        with self.assertRaisesRegex(SystemExit, "frozen configuration drift"):
            analyzer.validate_formal_matrix(
                [
                    replace(
                        runs[1],
                        system_resolved_config_sha256="f" * 64,
                    ),
                    runs[0],
                    *runs[2:],
                ]
            )

    def test_sidecar_full_identity_and_evidence_are_revalidated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs, manifest_by_source, manifests, sidecars = _protocol_matrix(Path(tmp))
            manifest_path = manifests[("llama2_7b", 43)]
            sidecar_path = sidecars[("llama2_7b", 43)]
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["full_run_identity"]["trace_sha256"] = "f" * 64
            sidecar["full_run_identity_sha256"] = analyzer._canonical_sha256(
                sidecar["full_run_identity"]
            )
            _write_json(sidecar_path, sidecar)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["full_run_identity_sha256"] = sidecar["full_run_identity_sha256"]
            manifest["system_resolved_config_sidecar_bytes"] = (
                sidecar_path.stat().st_size
            )
            manifest["system_resolved_config_sidecar_sha256"] = _file_sha256(
                sidecar_path
            )
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                SystemExit, "full-run trace/subset identity mismatch"
            ):
                analyzer.validate_formal_campaign_protocol(runs, manifest_by_source)

        with tempfile.TemporaryDirectory() as tmp:
            runs, manifest_by_source, manifests, _ = _protocol_matrix(Path(tmp))
            manifest_path = manifests[("llama2_7b", 43)]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            evidence_path = Path(manifest["seed41_validation_evidence_path"])
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["seed41_validation"]["sampling_seed"] = 42
            _write_json(evidence_path, evidence)
            manifest["seed41_validation_evidence_bytes"] = evidence_path.stat().st_size
            manifest["seed41_validation_evidence_sha256"] = _file_sha256(evidence_path)
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(SystemExit, "validation must bind seed 41"):
                analyzer.validate_formal_campaign_protocol(runs, manifest_by_source)

    def test_v2_seed_prefers_metadata_sampling_seed(self) -> None:
        payload = {
            "metadata": {
                "sampling_seed": 43,
                "generation_seed": 99,
                "workload_seed": 99,
            }
        }
        self.assertEqual(
            plot_paper_figures._v2_seed(
                payload,
                Path("/tmp/model_seed99_result.json"),
                "run_seed99",
            ),
            43,
        )

    def test_formal_analyze_rejects_loose_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loose = root / "loose.json"
            loose.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(
                SystemExit, "loose raw JSON input is forbidden"
            ):
                analyzer.analyze([loose], root / "output")


if __name__ == "__main__":
    unittest.main()
