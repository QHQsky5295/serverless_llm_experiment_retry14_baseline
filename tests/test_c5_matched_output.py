from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import analyze_c5_matched_output as c5


TRACE_SHA = "a" * 64
SUBSET_SHA = "b" * 64


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _request(
    system: str,
    index: int,
    *,
    dispatch_ms: float,
    service_ttft_ms: float,
    tpot_ms: float,
) -> dict:
    completion_tokens = 5
    decode_ms = tpot_ms * (completion_tokens - 1)
    service_e2e_ms = service_ttft_ms + decode_ms
    result = {
        "request_id": f"req-{index}",
        "adapter_id": f"adapter-{index % 2}",
        "success": True,
        "error": None,
        "generation_contract": c5.CONTRACT,
        "source_expected_output_tokens": completion_tokens,
        "requested_completion_tokens": completion_tokens,
        "completion_tokens": completion_tokens,
        "completion_token_ids_sha256": _digest(f"tokens-{system}-{index}"),
        "canonical_prompt_sha256": _digest(f"prompt-{index}"),
        "output_contract_match": True,
        "dispatch_admission_wait_ms": dispatch_ms,
        "service_ttft_ms": service_ttft_ms,
        "service_e2e_ms": service_e2e_ms,
        "overall_ttft_ms": dispatch_ms + service_ttft_ms,
        "ttft_ms": dispatch_ms + service_ttft_ms,
        "overall_e2e_ms": dispatch_ms + service_e2e_ms,
        "e2e_ms": dispatch_ms + service_e2e_ms,
        "tpot_ms": tpot_ms,
        "tpot_observed": True,
    }
    if system == "prime":
        result.update(
            {
                "scheduled_arrival_offset_s": float(index),
                "canonical_prompt_tokens": 20 + index,
                "completion_token_source": "vllm_token_ids",
            }
        )
    else:
        result.update(
            {
                "arrival_time_s": float(index),
                "guard_prompt_tokens": 20 + index,
                "completion_token_source": "slora_native_sse_token_id",
                "prompt_token_source": "local_tokenizer_guard",
                "native_sse_integer_token_id_count": completion_tokens,
                "native_sse_invalid_token_id_count": 0,
                "final_empty_success": False,
            }
        )
    return result


def _write_result(
    path: Path,
    *,
    system: str,
    seed: int,
    expected_requests: int = 3,
    dispatch_ms: float = 2.0,
    service_ttft_ms: float = 10.0,
    tpot_ms: float = 2.0,
    model: str = "llama2_7b",
) -> None:
    requests = [
        _request(
            system,
            index,
            dispatch_ms=dispatch_ms + index * 0.1,
            service_ttft_ms=service_ttft_ms + index * 0.2,
            tpot_ms=tpot_ms + index * 0.05,
        )
        for index in range(expected_requests)
    ]
    model_profile = (
        "llama32_3b_main_modelscope"
        if model == "llama32_3b"
        else "llama2_7b_main_v2_publicmix"
    )
    metadata = {
        "shared_trace_sha256": TRACE_SHA,
        "shared_adapter_subset_sha256": SUBSET_SHA,
        "model_profile": model_profile,
        "sampling_seed": seed,
    }
    if system == "prime":
        metadata.update({"generation_contract": c5.CONTRACT, "generation_seed": seed})
        scenario = "v2_full"
    else:
        metadata["sampling_seed"] = seed
        scenario = "slora_fair"
    avg_e2e_ms = sum(float(request["e2e_ms"]) for request in requests) / len(requests)
    monetary_cost = 0.02 if system == "prime" else 0.03
    monetary_ce = 1.0 / (monetary_cost * (avg_e2e_ms / 1000.0))
    payload = {
        "metadata": metadata,
        "scenario_summaries": {
            scenario: {
                "completion_token_source_counts": {
                    (
                        "vllm_token_ids"
                        if system == "prime"
                        else "slora_native_sse_token_id"
                    ): expected_requests
                },
                "throughput_tok_per_s": 120.0 if system == "prime" else 100.0,
                "monetary_cost_per_request_usd": monetary_cost,
                "monetary_ce": monetary_ce,
            }
        },
        "detailed_results": {
            scenario: {
                "total": expected_requests,
                "completed": expected_requests,
                "failed": 0,
                "requests": requests,
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _formal_order(model: str, seed: int) -> list[str]:
    return list(c5.FORMAL_EXECUTION_ORDERS[(model, seed)])


def _write_formal_round(root: Path, *, model: str, seed: int) -> tuple[Path, list[c5.RunSpec]]:
    round_dir = root / f"{model}-seed{seed}"
    prime_path = round_dir / "raw" / "faaslora" / "formal_faaslora_result.json"
    slora_path = round_dir / "raw" / "replay" / "formal_slora_dp4_tp1_summary.json"
    prime_path.parent.mkdir(parents=True)
    slora_path.parent.mkdir(parents=True)
    _write_result(prime_path, system="prime", seed=seed, expected_requests=1, model=model)
    _write_result(slora_path, system="slora", seed=seed, expected_requests=1, model=model)

    config_sha = _digest(f"formal-config-{model}")
    family_id = _digest(f"formal-family-{model}")
    source_commits = {"baselines": "c" * 40, "faaslora": "d" * 40}
    execution_order = _formal_order(model, seed)
    model_profile = (
        "llama32_3b_main_modelscope"
        if model == "llama32_3b"
        else "llama2_7b_main_v2_publicmix"
    )
    full_identity = {
        "system_resolved_config_sha256": config_sha,
        "model_profile": model_profile,
        "total_requests": c5.FORMAL_REQUESTS,
        "time_scale_factor": c5.FORMAL_TIME_SCALE,
        "selected_num_adapters": c5.FORMAL_ADAPTERS,
        "sampling_seed": seed,
        "trace_sha256": TRACE_SHA,
        "adapter_subset_sha256": SUBSET_SHA,
        "execution_order": execution_order,
        "generation_contract": c5.CONTRACT,
        "fixed_output_max_tokens": 256,
        "fixed_prompt_max_tokens": 759,
        "storage_bandwidth_mib_s": c5.FORMAL_BANDWIDTH_MIB_S,
        "workload_overrides": dict(c5.FORMAL_WORKLOAD_AXES),
        "faaslora_scenario": "v2_full",
    }
    full_identity_sha = c5._canonical_sha256(full_identity)
    sidecar = round_dir / "protocol" / "system_resolved_config.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(
        json.dumps(
            {
                "formal_run": True,
                "trace_role": "heldout",
                "sampling_seed": seed,
                "campaign_kind": c5.FORMAL_CAMPAIGN_KIND,
                "model_profile": model_profile,
                "configuration_family_id": family_id,
                "system_resolved_config_sha256": config_sha,
                "full_run_identity": full_identity,
                "full_run_identity_sha256": full_identity_sha,
                "source_commits": source_commits,
            }
        ),
        encoding="utf-8",
    )

    # Keep the referenced seed-41 manifest outside the held-out round tree;
    # formal provenance discovery intentionally treats every nested
    # MANIFEST.json as a campaign input.
    validation_dir = root / "_seed41_validation_fixtures" / f"{model}-for-seed{seed}"
    validation_dir.mkdir(parents=True)
    validation_trace = validation_dir / "trace.json"
    validation_subset = validation_dir / "adapter_subset.json"
    validation_trace.write_text('{"seed":41,"kind":"trace"}', encoding="utf-8")
    validation_subset.write_text('{"seed":41,"kind":"subset"}', encoding="utf-8")
    validation_order = (
        ["slora", "faaslora"]
        if model == "llama2_7b"
        else ["faaslora", "slora"]
    )
    validation_identity = {
        "total_requests": 1000,
        "execution_order": validation_order,
        "trace_sha256": c5._sha256_file(validation_trace),
        "adapter_subset_sha256": c5._sha256_file(validation_subset),
    }
    validation_sidecar = validation_dir / "system_resolved_config.json"
    validation_sidecar.write_text(
        json.dumps(
            {
                "formal_run": True,
                "source_clean_for_formal": True,
                "trace_role": "validation",
                "sampling_seed": 41,
                "campaign_kind": c5.FORMAL_CAMPAIGN_KIND,
                "model_profile": model_profile,
                "configuration_family_id": family_id,
                "system_resolved_config_sha256": config_sha,
                "full_run_identity": validation_identity,
                "source_commits": source_commits,
            }
        ),
        encoding="utf-8",
    )
    validation_manifest = validation_dir / "MANIFEST.json"
    validation_manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "formal_run": True,
                "source_clean_for_formal": True,
                "trace_role": "validation",
                "sampling_seed": 41,
                "campaign_kind": c5.FORMAL_CAMPAIGN_KIND,
                "model_profile": model_profile,
                "system_resolved_config_family_id": family_id,
                "system_resolved_config_sha256": config_sha,
                "systems": ["slora", "faaslora"],
                "execution_order": validation_order,
                "shared_trace_path": str(validation_trace),
                "shared_trace_sha256": c5._sha256_file(validation_trace),
                "shared_adapter_subset_path": str(validation_subset),
                "shared_adapter_subset_sha256": c5._sha256_file(validation_subset),
                "system_resolved_config_path": str(validation_sidecar),
                "system_resolved_config_sidecar_sha256": c5._sha256_file(
                    validation_sidecar
                ),
                "system_resolved_config_sidecar_bytes": validation_sidecar.stat().st_size,
                "baseline_git": {"commit": source_commits["baselines"]},
                "faaslora_git": {"commit": source_commits["faaslora"]},
            }
        ),
        encoding="utf-8",
    )
    evidence = round_dir / "protocol" / "seed41_validation_evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": c5.SEED41_EVIDENCE_SCHEMA,
                "heldout": {
                    "sampling_seed": seed,
                    "campaign_kind": c5.FORMAL_CAMPAIGN_KIND,
                    "model_profile": model_profile,
                    "configuration_family_id": family_id,
                    "system_resolved_config_sha256": config_sha,
                    "full_run_identity_sha256": full_identity_sha,
                    "source_commits": source_commits,
                },
                "seed41_validation": {
                    "sampling_seed": 41,
                    "total_requests": 1000,
                    "sidecar_path": str(validation_sidecar),
                    "sidecar_bytes": validation_sidecar.stat().st_size,
                    "sidecar_sha256": c5._sha256_file(validation_sidecar),
                    "manifest_path": str(validation_manifest),
                    "manifest_bytes": validation_manifest.stat().st_size,
                    "manifest_sha256": c5._sha256_file(validation_manifest),
                    "campaign_kind": c5.FORMAL_CAMPAIGN_KIND,
                    "model_profile": model_profile,
                    "configuration_family_id": family_id,
                    "system_resolved_config_sha256": config_sha,
                    "systems": ["slora", "faaslora"],
                    "execution_order": validation_order,
                    "source_commits": source_commits,
                    "shared_trace_path": str(validation_trace),
                    "shared_trace_sha256": c5._sha256_file(validation_trace),
                    "shared_adapter_subset_path": str(validation_subset),
                    "shared_adapter_subset_sha256": c5._sha256_file(validation_subset),
                },
            }
        ),
        encoding="utf-8",
    )

    def record(path: Path) -> dict:
        return {"bytes": path.stat().st_size, "sha256": c5._sha256_file(path)}

    manifest = {
        "status": "complete",
        "formal_run": True,
        "trace_role": "heldout",
        "source_clean_for_formal": True,
        "campaign_kind": c5.FORMAL_CAMPAIGN_KIND,
        "baseline_git": {"commit": source_commits["baselines"]},
        "faaslora_git": {"commit": source_commits["faaslora"]},
        "model_profile": model_profile,
        "sampling_seed": seed,
        "total_requests": c5.FORMAL_REQUESTS,
        "selected_num_adapters": c5.FORMAL_ADAPTERS,
        "shared_trace_sha256": TRACE_SHA,
        "shared_adapter_subset_sha256": SUBSET_SHA,
        "systems": ["slora", "faaslora"],
        "supported_systems": ["slora", "faaslora"],
        "generation_contract": c5.CONTRACT,
        "fixed_output_max_tokens": 256,
        "fixed_prompt_max_tokens": 759,
        "bandwidth_mib_s": c5.FORMAL_BANDWIDTH_MIB_S,
        "faaslora_scenario": "v2_full",
        "execution_order": execution_order,
        "system_resolved_config_sha256": config_sha,
        "full_run_identity_sha256": full_identity_sha,
        "system_resolved_config_path": str(sidecar),
        "system_resolved_config_sidecar_sha256": c5._sha256_file(sidecar),
        "system_resolved_config_sidecar_bytes": sidecar.stat().st_size,
        "seed41_validation_evidence_path": str(evidence),
        "seed41_validation_evidence_sha256": c5._sha256_file(evidence),
        "seed41_validation_evidence_bytes": evidence.stat().st_size,
        "source_files": {
            str(prime_path.relative_to(round_dir)): record(prime_path),
            str(slora_path.relative_to(round_dir)): record(slora_path),
        },
    }
    manifest_path = round_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return round_dir, c5.specs_from_round(round_dir)


def _fake_formal_run(spec: c5.RunSpec, **_kwargs: object) -> c5.ValidatedRun:
    system = c5._normalize_system(spec.system)
    model = c5._canonical_model_identity(spec.model)
    scenario = "v2_full" if system == "prime" else "slora_fair"
    e2e = 20.0 if system == "prime" else 25.0
    cost = 0.02 if system == "prime" else 0.03
    metrics = {field: 1.0 for field in c5.RUN_METRIC_FIELDS}
    metrics.update(
        {
            "dispatch_admission_mean_ms": 2.0 if system == "prime" else 3.0,
            "service_ttft_mean_ms": 8.0 if system == "prime" else 10.0,
            "decode_mean_ms": 10.0 if system == "prime" else 12.0,
            "overall_ttft_mean_ms": 10.0 if system == "prime" else 13.0,
            "service_e2e_mean_ms": 18.0 if system == "prime" else 22.0,
            "e2e_mean_ms": e2e,
            "throughput_tok_per_s": 120.0 if system == "prime" else 100.0,
            "monetary_cost_per_request_usd": cost,
            "monetary_ce": 1.0 / (cost * (e2e / 1000.0)),
        }
    )
    request_map = [{"request_id": "formal-map"}]
    return c5.ValidatedRun(
        system=system,
        model=model,
        seed=spec.seed,
        path=spec.path.resolve(),
        scenario=scenario,
        requests=[{}] * c5.FORMAL_REQUESTS,
        request_map=request_map,
        request_map_sha256=c5._canonical_sha256(request_map),
        source_sha256=c5._sha256_file(spec.path),
        trace_sha256=TRACE_SHA,
        adapter_subset_sha256=SUBSET_SHA,
        metrics=metrics,
        round_manifest=spec.round_manifest.resolve() if spec.round_manifest else None,
    )


class C5MatchedOutputTests(unittest.TestCase):
    def test_three_seed_analysis_uses_seed_level_paired_ci(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = []
            for seed_index, seed in enumerate((43, 44, 45)):
                prime = root / f"prime-{seed}.json"
                slora = root / f"slora-{seed}.json"
                _write_result(
                    prime,
                    system="prime",
                    seed=seed,
                    dispatch_ms=2.0 + seed_index,
                    service_ttft_ms=9.0 + seed_index,
                    tpot_ms=1.5 + seed_index * 0.1,
                )
                _write_result(
                    slora,
                    system="slora",
                    seed=seed,
                    dispatch_ms=4.0 + seed_index,
                    service_ttft_ms=12.0 + seed_index,
                    tpot_ms=2.0 + seed_index * 0.1,
                )
                specs.extend(
                    [
                        c5.RunSpec("prime", "llama2_7b", seed, prime),
                        c5.RunSpec("slora", "llama2_7b", seed, slora),
                    ]
                )

            output = root / "publication"
            manifest = c5.analyze(specs, output_dir=output, expected_requests=3)

            self.assertFalse(
                manifest["statistical_method"]["request_records_are_independent_repetitions"]
            )
            self.assertEqual(
                manifest["statistical_method"]["model_seed_sets"]["llama2_7b"],
                [43, 44, 45],
            )
            expected_files = {
                "c5_per_run_metrics.csv",
                "c5_paired_seed_differences.csv",
                "c5_seed_level_ci.csv",
                "c5_matched_output_manifest.json",
                "c5_matched_output_stage_decomposition.pdf",
                "c5_matched_output_stage_decomposition.png",
            }
            self.assertTrue(expected_files.issubset({path.name for path in output.iterdir()}))
            with (output / "c5_seed_level_ci.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            e2e = next(
                row
                for row in rows
                if row["model"] == "llama2_7b" and row["metric"] == "e2e_mean_ms"
            )
            self.assertEqual(e2e["n_seeds"], "3")
            self.assertEqual(e2e["ci_unit"], "seed")
            self.assertAlmostEqual(float(e2e["prime_minus_slora_mean"]), -7.0)
            self.assertGreaterEqual(float(e2e["slora_minus_prime_improvement_mean"]), 7.0)
            throughput = next(
                row
                for row in rows
                if row["model"] == "llama2_7b"
                and row["metric"] == "throughput_tok_per_s"
            )
            self.assertEqual(throughput["metric_direction"], "higher-is-better")
            self.assertAlmostEqual(float(throughput["prime_advantage_mean"]), 20.0)
            cost = next(
                row
                for row in rows
                if row["model"] == "llama2_7b"
                and row["metric"] == "monetary_cost_per_request_usd"
            )
            self.assertEqual(cost["metric_direction"], "lower-is-better")
            self.assertAlmostEqual(float(cost["prime_advantage_mean"]), 0.01)

    def test_cross_system_request_map_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prime = root / "prime.json"
            slora = root / "slora.json"
            _write_result(prime, system="prime", seed=43)
            _write_result(slora, system="slora", seed=43)
            payload = json.loads(slora.read_text(encoding="utf-8"))
            payload["detailed_results"]["slora_fair"]["requests"][1][
                "canonical_prompt_sha256"
            ] = _digest("different prompt")
            slora.write_text(json.dumps(payload), encoding="utf-8")

            specs = [
                c5.RunSpec("prime", "llama2_7b", 43, prime),
                c5.RunSpec("slora", "llama2_7b", 43, slora),
            ]
            with self.assertRaisesRegex(c5.ValidationError, "request/adapter/arrival/target/prompt"):
                c5.analyze(specs, output_dir=root / "out", expected_requests=3)
            self.assertFalse((root / "out").exists())

    def test_missing_formal_seeds_are_rejected_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prime = root / "prime.json"
            slora = root / "slora.json"
            _write_result(prime, system="prime", seed=43)
            _write_result(slora, system="slora", seed=43)
            specs = [
                c5.RunSpec("prime", "llama2_7b", 43, prime),
                c5.RunSpec("slora", "llama2_7b", 43, slora),
            ]
            with self.assertRaisesRegex(c5.ValidationError, "incomplete formal seed matrix"):
                c5.analyze(specs, output_dir=root / "out", expected_requests=3)
            self.assertFalse((root / "out").exists())

    def test_incomplete_and_token_contract_runs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prime = root / "prime.json"
            _write_result(prime, system="prime", seed=43)
            payload = json.loads(prime.read_text(encoding="utf-8"))
            payload["detailed_results"]["v2_full"]["completed"] = 2
            payload["detailed_results"]["v2_full"]["failed"] = 1
            prime.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(c5.ValidationError, "incomplete formal run"):
                c5.validate_run(c5.RunSpec("prime", "llama2_7b", 43, prime), expected_requests=3)

            _write_result(prime, system="prime", seed=43)
            payload = json.loads(prime.read_text(encoding="utf-8"))
            payload["detailed_results"]["v2_full"]["requests"][0]["completion_tokens"] = 4
            prime.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(c5.ValidationError, "actual_tokens != target_tokens"):
                c5.validate_run(c5.RunSpec("prime", "llama2_7b", 43, prime), expected_requests=3)

    def test_latency_identity_tpot_and_fallback_gates_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slora = root / "slora.json"
            _write_result(slora, system="slora", seed=43)
            payload = json.loads(slora.read_text(encoding="utf-8"))
            request = payload["detailed_results"]["slora_fair"]["requests"][0]
            request["overall_e2e_ms"] += 1.01
            request["e2e_ms"] += 1.01
            slora.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(c5.ValidationError, "decomposition error"):
                c5.validate_run(c5.RunSpec("slora", "llama2_7b", 43, slora), expected_requests=3)

            _write_result(slora, system="slora", seed=43)
            payload = json.loads(slora.read_text(encoding="utf-8"))
            payload["detailed_results"]["slora_fair"]["requests"][0]["tpot_ms"] += 1.01
            slora.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(c5.ValidationError, "TPOT recomputation error"):
                c5.validate_run(c5.RunSpec("slora", "llama2_7b", 43, slora), expected_requests=3)

            _write_result(slora, system="slora", seed=43)
            payload = json.loads(slora.read_text(encoding="utf-8"))
            payload["scenario_summaries"]["slora_fair"]["completion_token_source_counts"][
                "trace_expected"
            ] = 1
            slora.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(c5.ValidationError, "forbidden fallback count"):
                c5.validate_run(c5.RunSpec("slora", "llama2_7b", 43, slora), expected_requests=3)

    def test_nonfresh_output_and_incomplete_round_manifest_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out"
            output.mkdir()
            (output / "old.csv").write_text("old", encoding="utf-8")
            with self.assertRaisesRegex(c5.ValidationError, "not fresh/empty"):
                c5._prepare_output_dir(output)

            round_dir = root / "round"
            round_dir.mkdir()
            (round_dir / "MANIFEST.json").write_text(
                json.dumps(
                    {
                        "status": "incomplete",
                        "generation_contract": c5.CONTRACT,
                        "model_profile": "llama2_7b_main_v2_publicmix",
                        "sampling_seed": 43,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(c5.ValidationError, "manifest is incomplete"):
                c5.specs_from_round(round_dir)

    def test_formal_mode_accepts_only_exact_two_model_matrix_and_emits_headlines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            round_dirs: list[Path] = []
            specs: list[c5.RunSpec] = []
            for model in c5.FORMAL_MODELS:
                for seed in c5.FORMAL_SEEDS:
                    round_dir, round_specs = _write_formal_round(
                        root, model=model, seed=seed
                    )
                    round_dirs.append(round_dir)
                    specs.extend(round_specs)

            with mock.patch.object(c5, "validate_run", side_effect=_fake_formal_run):
                manifest = c5.analyze(
                    specs,
                    output_dir=root / "formal-publication",
                    formal_mode=True,
                    provenance_inputs=round_dirs,
                )

            self.assertTrue(manifest["formal_mode"])
            self.assertEqual(len(manifest["runs"]), 12)
            self.assertEqual(len(manifest["formal_provenance_manifests"]), 6)
            self.assertEqual(
                manifest["validation_gates"]["formal_prime_scenario"], "v2_full"
            )
            per_run = manifest["runs"][0]
            for field in (
                "throughput_tok_per_s",
                "monetary_cost_per_request_usd",
                "monetary_ce",
            ):
                self.assertIn(field, per_run)
            ci_by_metric = {
                row["metric"]: row
                for row in manifest["paired_seed_statistics"]
                if row["model"] == "llama2_7b"
            }
            self.assertEqual(
                ci_by_metric["throughput_tok_per_s"]["metric_direction"],
                "higher-is-better",
            )
            self.assertEqual(ci_by_metric["throughput_tok_per_s"]["n_seeds"], 3)
            self.assertGreater(
                ci_by_metric["throughput_tok_per_s"]["prime_advantage_mean"], 0
            )

    def test_formal_mode_rejects_loose_json_wrong_order_and_incomplete_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            round_dirs: list[Path] = []
            specs: list[c5.RunSpec] = []
            for model in c5.FORMAL_MODELS:
                for seed in c5.FORMAL_SEEDS:
                    round_dir, round_specs = _write_formal_round(
                        root, model=model, seed=seed
                    )
                    round_dirs.append(round_dir)
                    specs.extend(round_specs)

            with self.assertRaisesRegex(SystemExit, "loose raw JSON input is forbidden"):
                c5.analyze(
                    specs,
                    output_dir=root / "loose-out",
                    formal_mode=True,
                    provenance_inputs=[specs[0].path],
                )

            with mock.patch.object(c5, "validate_run", side_effect=_fake_formal_run):
                with self.assertRaisesRegex(c5.ValidationError, "exactly 2 models"):
                    c5.analyze(
                        specs[:-2],
                        output_dir=root / "missing-out",
                        formal_mode=True,
                        provenance_inputs=round_dirs,
                    )
            self.assertFalse((root / "missing-out").exists())

            bad_manifest_path = round_dirs[0] / "MANIFEST.json"
            bad_manifest = json.loads(bad_manifest_path.read_text(encoding="utf-8"))
            bad_manifest["execution_order"] = list(
                reversed(bad_manifest["execution_order"])
            )
            bad_manifest_path.write_text(json.dumps(bad_manifest), encoding="utf-8")
            with mock.patch.object(c5, "validate_run", side_effect=_fake_formal_run):
                with self.assertRaisesRegex(c5.ValidationError, "execution_order"):
                    c5.analyze(
                        specs,
                        output_dir=root / "wrong-order-out",
                        formal_mode=True,
                        provenance_inputs=round_dirs,
                    )
            self.assertFalse((root / "wrong-order-out").exists())

    def test_formal_mode_rejects_workload_axis_and_non_v2_full_prime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            round_dirs: list[Path] = []
            specs: list[c5.RunSpec] = []
            for model in c5.FORMAL_MODELS:
                for seed in c5.FORMAL_SEEDS:
                    round_dir, round_specs = _write_formal_round(
                        root, model=model, seed=seed
                    )
                    round_dirs.append(round_dir)
                    specs.extend(round_specs)

            manifest_path = round_dirs[0] / "MANIFEST.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            sidecar_path = Path(manifest["system_resolved_config_path"])
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["full_run_identity"]["workload_overrides"]["zipf_exponent"] = 1.4
            sidecar["full_run_identity_sha256"] = c5._canonical_sha256(
                sidecar["full_run_identity"]
            )
            sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
            manifest["full_run_identity_sha256"] = sidecar["full_run_identity_sha256"]
            manifest["system_resolved_config_sidecar_sha256"] = c5._sha256_file(sidecar_path)
            manifest["system_resolved_config_sidecar_bytes"] = sidecar_path.stat().st_size
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.object(c5, "validate_run", side_effect=_fake_formal_run):
                with self.assertRaisesRegex(c5.ValidationError, "zipf_exponent"):
                    c5.analyze(
                        specs,
                        output_dir=root / "bad-axis-out",
                        formal_mode=True,
                        provenance_inputs=round_dirs,
                    )

            def wrong_prime_scenario(spec: c5.RunSpec, **kwargs: object) -> c5.ValidatedRun:
                run = _fake_formal_run(spec, **kwargs)
                if run.system == "prime":
                    run.scenario = "faaslora_full"
                return run

            # Restore the axis before exercising the independent scenario gate.
            sidecar["full_run_identity"]["workload_overrides"]["zipf_exponent"] = 1.0
            sidecar["full_run_identity_sha256"] = c5._canonical_sha256(
                sidecar["full_run_identity"]
            )
            sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
            manifest["full_run_identity_sha256"] = sidecar["full_run_identity_sha256"]
            manifest["system_resolved_config_sidecar_sha256"] = c5._sha256_file(sidecar_path)
            manifest["system_resolved_config_sidecar_bytes"] = sidecar_path.stat().st_size
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.object(c5, "validate_run", side_effect=wrong_prime_scenario):
                with self.assertRaisesRegex(c5.ValidationError, "scenario='v2_full'"):
                    c5.analyze(
                        specs,
                        output_dir=root / "bad-scenario-out",
                        formal_mode=True,
                        provenance_inputs=round_dirs,
                    )

    def test_formal_mode_rejects_campaign_and_seed41_evidence_tampering(self) -> None:
        def build_matrix(root: Path) -> tuple[list[Path], list[c5.RunSpec]]:
            round_dirs: list[Path] = []
            specs: list[c5.RunSpec] = []
            for model in c5.FORMAL_MODELS:
                for seed in c5.FORMAL_SEEDS:
                    round_dir, round_specs = _write_formal_round(
                        root, model=model, seed=seed
                    )
                    round_dirs.append(round_dir)
                    specs.extend(round_specs)
            return round_dirs, specs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            round_dirs, specs = build_matrix(root / "wrong-campaign")
            manifest_path = round_dirs[0] / "MANIFEST.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["campaign_kind"] = "v2_full_vs_serverless"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.object(c5, "validate_run", side_effect=_fake_formal_run):
                with self.assertRaisesRegex(c5.ValidationError, "campaign_kind"):
                    c5.analyze(
                        specs,
                        output_dir=root / "wrong-campaign-out",
                        formal_mode=True,
                        provenance_inputs=round_dirs,
                    )

            round_dirs, specs = build_matrix(root / "missing-evidence")
            manifest_path = round_dirs[0] / "MANIFEST.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("seed41_validation_evidence_path")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.object(c5, "validate_run", side_effect=_fake_formal_run):
                with self.assertRaisesRegex(c5.ValidationError, "seed41 validation evidence"):
                    c5.analyze(
                        specs,
                        output_dir=root / "missing-evidence-out",
                        formal_mode=True,
                        provenance_inputs=round_dirs,
                    )

            round_dirs, specs = build_matrix(root / "damaged-seed41-artifact")
            manifest = json.loads(
                (round_dirs[0] / "MANIFEST.json").read_text(encoding="utf-8")
            )
            evidence = json.loads(
                Path(manifest["seed41_validation_evidence_path"]).read_text(
                    encoding="utf-8"
                )
            )
            trace_path = Path(
                evidence["seed41_validation"]["shared_trace_path"]
            )
            trace_path.write_text("tampered", encoding="utf-8")
            with mock.patch.object(c5, "validate_run", side_effect=_fake_formal_run):
                with self.assertRaisesRegex(c5.ValidationError, "bytes/SHA mismatch"):
                    c5.analyze(
                        specs,
                        output_dir=root / "damaged-seed41-out",
                        formal_mode=True,
                        provenance_inputs=round_dirs,
                    )


if __name__ == "__main__":
    unittest.main()
