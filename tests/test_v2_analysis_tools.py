from __future__ import annotations

import json
import csv
import hashlib
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from scripts import analyze_service_readiness as readiness
from scripts import eurosys27_v2_provenance as v2_provenance
from scripts import plot_paper_figures

sys.modules.setdefault("plot_paper_figures", plot_paper_figures)
from scripts import plot_paper_sensitivity  # noqa: E402


def _request(
    request_id: str,
    *,
    dispatch_tier: str | None,
    service_tier: str,
) -> dict:
    record = {
        "request_id": request_id,
        "adapter_id": "adapter-a",
        "success": True,
        "cache_tier": service_tier,
        "overall_ttft_ms": 100.0,
        "adapter_gpu_ready_before_dispatch": dispatch_tier == "gpu",
        "adapter_local_ready_before_dispatch": dispatch_tier in ("host", "nvme"),
        "adapter_remote_cold_before_dispatch": dispatch_tier == "remote",
        "adapter_replica_mismatch": dispatch_tier != "gpu",
        "remote_mismatch": dispatch_tier == "remote",
    }
    if dispatch_tier is not None:
        record["readiness_tier_before_dispatch"] = dispatch_tier
    return record


def _write_result(
    path: Path,
    *,
    model: str,
    run_tag: str,
    requests: list[dict],
    diagnostic: dict | None = None,
) -> None:
    metadata = {
        "model": model,
        "results_tag": run_tag,
    }
    detail: dict = {"requests": requests}
    if diagnostic is not None:
        total = int(diagnostic.get("total", len(requests)))
        phase_count = int(diagnostic.get("phase_count", 8))
        phase_size = total // phase_count if phase_count > 0 else 0
        phase_results = [
            {
                "phase": index,
                "total": phase_size,
                "completed": phase_size,
            }
            for index in range(phase_count)
        ]
        detail.update(
            {
                "total": total,
                "completed": int(diagnostic.get("completed", total)),
                "failed": int(diagnostic.get("failed", 0)),
                "multi_cycle_phase_results": diagnostic.get(
                    "phase_results", phase_results
                ),
                "scale_up_events": [
                    {"event": index}
                    for index in range(int(diagnostic.get("scale_up_events", 8)))
                ],
            }
        )
        metadata.update(
            {
                "generation_seed": int(diagnostic.get("generation_seed", 43)),
                "total_requests": total,
                "generation_contract": "fixed_length_greedy_v1",
                "shared_trace_sha256": "a" * 64,
                "shared_adapter_subset_sha256": "b" * 64,
                "scenario_coordination": {
                    "faaslora_full": {
                        "cold_cache_reset_before_run": diagnostic.get(
                            "cold_cache_reset_before_run", True
                        ),
                        "feature_activation": {
                            "scale_up_event_count": int(
                                diagnostic.get("scale_up_events", 8)
                            )
                        },
                    }
                },
            }
        )
    path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "detailed_results": {
                    "faaslora_full": detail
                },
            }
        ),
        encoding="utf-8",
    )


def _diagnostic_requests(total: int, *, first_service: int) -> list[dict]:
    records: list[dict] = []
    for index in range(total):
        record = _request(f"request-{index}", dispatch_tier="gpu", service_tier="gpu")
        record["scaleup_first_service"] = index < first_service
        record["scaleup_planned_adapter_match"] = index < first_service
        records.append(record)
    return records


def _write_v2_ablation_result(
    path: Path,
    *,
    model: str,
    scenario: str,
    seed: int,
    ttft_p95: float,
    e2e_avg: float,
    cost: float,
    ce: float,
    trace_sha: str = "a" * 64,
    subset_sha: str = "b" * 64,
    corrupt_contract_map: bool = False,
    missing_dispatch_tier: bool = False,
    corrupt_activation: bool = False,
    total: int = 4,
    non_feature_hash: str = "e" * 64,
    bandwidth_mib_s: float = 250.0,
    generation_contract: str = "legacy",
    admission_outcomes: tuple[int, int, int] | None = None,
) -> None:
    contract = generation_contract
    requests = []
    for index in range(total):
        requests.append(
            {
                "request_id": f"r{index}",
                "adapter_id": f"adapter-{index % 2}",
                "success": True,
                "scheduled_arrival_offset_s": float(index),
                "source_expected_output_tokens": 64,
                "requested_completion_tokens": 64,
                "canonical_prompt_sha256": "c" * 64,
                "canonical_prompt_tokens": 32,
                "generation_contract": contract,
                "readiness_tier_before_dispatch": "gpu",
            }
        )
    if missing_dispatch_tier:
        requests[0].pop("readiness_tier_before_dispatch")
    contract_rows = [
        {
            "request_id": request["request_id"],
            "adapter_id": request["adapter_id"],
            "arrival_time_s": request["scheduled_arrival_offset_s"],
            "source_expected_output_tokens": request["source_expected_output_tokens"],
            "requested_completion_tokens": request["requested_completion_tokens"],
            "canonical_prompt_sha256": request["canonical_prompt_sha256"],
            "canonical_prompt_tokens": request["canonical_prompt_tokens"],
        }
        for request in requests
    ]
    contract_map_sha = hashlib.sha256(
        json.dumps(
            contract_rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if corrupt_contract_map:
        contract_map_sha = "d" * 64

    gate_map = {
        "v2_elastic_only": (False, False, False, False, False),
        "v2_hit_aware_preparation": (True, True, False, False, False),
        "v2_hierarchical_no_coord": (True, True, True, False, False),
        "v2_full": (True, True, True, True, True),
    }
    readiness, handoff, hierarchy, coordination, admission = gate_map[scenario]
    if admission_outcomes is None:
        admission_outcomes = (1, 0, 0) if admission else (0, 0, 0)
    admission_admits, admission_defers, admission_rejects = admission_outcomes
    activation = {
        "successful_request_count": total,
        "routing_decision_count": total,
        "routing_selection_attempt_count": total,
        "readiness_aware_routing_decision_count": total if readiness else 0,
        "load_only_routing_decision_count": 0 if readiness else total,
        "scale_up_event_count": 1 if handoff else 0,
        "scale_up_events_with_planned_adapters": 1 if handoff else 0,
        "scaleup_first_service_request_count": 1 if handoff else 0,
        "scaleup_first_service_planned_match_count": 1 if handoff else 0,
        "scaleup_first_service_planned_match_rate": 1.0 if handoff else 0.0,
        "gpu_admission_observed_request_count": 1 if admission else 0,
        "gpu_admission_decision_count": 1 if admission else 0,
        "gpu_admission_admit_count": admission_admits,
        "gpu_admission_defer_count": admission_defers,
        "gpu_admission_reject_count": admission_rejects,
        "dispatch_tier_counts": {"gpu": total, "host": 0, "nvme": 0, "remote": 0},
        "dispatch_to_service_transition_count": 1 if hierarchy else 0,
        "initial_or_current_nvme_adapter_count": 2 if handoff else 0,
        "initial_or_current_host_adapter_count": 1 if hierarchy else 0,
        "host_promotion_scheduled_count": 1 if hierarchy else 0,
        "host_promotion_completed_count": 1 if hierarchy else 0,
        "runtime_gpu_forward_attempt_count": 1 if hierarchy else 0,
        "runtime_gpu_forward_success_count": 1 if hierarchy else 0,
    }
    if corrupt_activation:
        activation["readiness_aware_routing_decision_count"] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "model": model,
                    "generation_seed": seed,
                    "results_tag": f"v2_seed{seed}_{scenario}",
                    "shared_trace_sha256": trace_sha,
                    "shared_adapter_subset_sha256": subset_sha,
                    "generation_contract": contract,
                    "non_feature_frozen_config_sha256": non_feature_hash,
                    "num_adapters": 500,
                    "bandwidth_mib_s": bandwidth_mib_s,
                    "configured_time_scale_factor": 8.0,
                    "effective_time_scale_factor": 8.0,
                    "active_adapter_cap": 48,
                    "hotset_rotation_requests": 500,
                    "hotset_rotation_mode": "legacy",
                    "hotset_overlap_fraction": 0.75,
                    "shared_trace_load_profile": {
                        "zipf_exponent": 1.0,
                        "active_adapter_cap": 48,
                        "hotset_rotation_requests": 500,
                        "rotation_mode": "legacy",
                        "hotset_overlap_fraction": 0.75,
                        "num_adapters": 500,
                    },
                    "generation_contract_request_map_sha256": {
                        scenario: contract_map_sha,
                    },
                    "scenario_coordination": {
                        scenario: {
                            "feature_gates": {
                                "readiness_routing_enabled": readiness,
                                "scale_up_handoff_enabled": handoff,
                                "hierarchical_residency_enabled": hierarchy,
                                "coordination_enabled": coordination,
                                "effective_capacity_admission_enabled": admission,
                            },
                            "feature_activation": activation,
                        }
                    },
                },
                "detailed_results": {
                    scenario: {
                        "total": total,
                        "completed": total,
                        "requests": requests,
                        "avg_overall_ttft_ms": ttft_p95 * 0.7,
                        "p95_overall_ttft_ms": ttft_p95,
                        "avg_overall_e2e_ms": e2e_avg,
                        "avg_tpot_ms": 12.0,
                        "throughput_tok_per_s": 100.0 + ce / 100.0,
                        "monetary_cost_per_request_usd": cost,
                        "monetary_ce": ce,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_sensitivity_result(
    path: Path,
    *,
    system: str,
    scenario: str,
    seed: int,
    sequence: list[str],
    rotation_mode: str = "abrupt",
    rotation_requests: int = 4,
    zipf: float = 1.0,
    overlap: float = 0.0,
    bandwidth: dict | None = None,
    trace_sha: str = "a" * 64,
    subset_sha: str = "b" * 64,
    model: str = "/models/Llama-2-7B",
) -> None:
    requests = []
    for index, adapter in enumerate(sequence):
        requests.append(
            {
                "request_id": f"r{index}",
                "adapter_id": adapter,
                "success": True,
                "scheduled_arrival_offset_s": float(index),
                "tpot_ms": 10.0 + index,
                "tpot_observed": True,
                "adapter_remote_cold_before_dispatch": index % 4 == 0,
            }
        )
    metadata = {
        "system": system,
        "model": model,
        "generation_seed": seed,
        "results_tag": f"{system}_{scenario}_seed{seed}_{path.stem}",
        "workload_profile": "v2_workload_test",
        "sampling_stats": {
            "shared_trace_metadata": {
                "load_profile": {
                    "rotation_mode": rotation_mode,
                    "hotset_rotation_requests": rotation_requests,
                    "zipf_exponent": zipf,
                    "hotset_overlap_fraction": overlap,
                    "active_adapter_cap": 3,
                    "num_adapters": 500,
                }
            }
        },
        "shared_trace_sha256": trace_sha,
        "shared_adapter_subset_sha256": subset_sha,
    }
    if bandwidth is not None:
        metadata.update(
            {
                "bandwidth_mib_s": bandwidth.get("configured_mib_s"),
                "bandwidth_gbit_s": bandwidth.get("configured_gbit_s"),
                "bandwidth_limit_mode": bandwidth["limit_mode"],
                "scenario_coordination": {
                    scenario: {
                        "aggregate_bandwidth": bandwidth,
                    }
                },
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "detailed_results": {
                    scenario: {
                        "total": len(sequence),
                        "completed": len(sequence),
                        "requests": requests,
                        "avg_overall_ttft_ms": 100.0,
                        "p95_overall_ttft_ms": 150.0,
                        "avg_overall_e2e_ms": 500.0,
                        "p95_overall_e2e_ms": 650.0,
                        "avg_tpot_ms": 12.0,
                        "throughput_tok_per_s": 100.0,
                        "monetary_cost_per_request_usd": 0.01,
                        "monetary_ce": 200.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_formal_ablation_matrix(root: Path) -> None:
    matrix = {
        "Llama-2-7B": plot_paper_figures.V2_ABLATION_SCENARIOS,
        "Llama-3.2-3B": ("v2_elastic_only", "v2_full"),
    }
    for model, scenarios in matrix.items():
        for seed in (43, 44, 45):
            for scenario_index, scenario in enumerate(scenarios):
                _write_v2_ablation_result(
                    root
                    / model
                    / f"seed_{seed}"
                    / f"{scenario}_result.json",
                    model=model,
                    scenario=scenario,
                    seed=seed,
                    ttft_p95=100.0 - 5.0 * scenario_index,
                    e2e_avg=200.0 - 5.0 * scenario_index,
                    cost=0.01 - 0.0001 * scenario_index,
                    ce=500.0 + 10.0 * scenario_index,
                    total=4000,
                    non_feature_hash=("e" if "7B" in model else "f") * 64,
                )


def _bandwidth_metadata(mib_s: float | None, *, serverless: bool) -> dict:
    if mib_s is None:
        return {
            "limit_mode": "file_no_delay" if serverless else "local_sim_no_delay",
            "configured_mib_s": None,
            "configured_gbit_s": None,
            "transfer_count": 8,
            "total_bytes": 8 * 1024 * 1024,
            "reservation_span_s": 0.1,
            "total_injected_wait_s": 0.0,
            "achieved_reserved_mib_s": 80.0,
        }
    return {
        "limit_mode": (
            "file_aggregate_reservation"
            if serverless
            else "aggregate_application_layer_local_sim"
        ),
        "configured_mib_s": mib_s,
        "configured_gbit_s": mib_s * 8 * 1024 * 1024 / 1_000_000_000,
        "transfer_count": 8,
        "total_bytes": 8 * 1024 * 1024,
        "reservation_span_s": 1.0,
        "total_injected_wait_s": 0.5,
        "achieved_reserved_mib_s": min(mib_s, 8.0),
    }


def _write_formal_bandwidth_matrix(root: Path) -> None:
    model_points = {
        "/models/Llama-2-7B": {
            43: (11.9209, 29.8023, 59.6046, 119.2093, 250.0, None),
            44: (11.9209, 119.2093, None),
            45: (11.9209, 119.2093, None),
        },
        "/models/Llama-3.2-3B": {
            43: (11.9209, 119.2093, None),
        },
    }
    sequence = ["a", "a", "b", "a", "c", "b", "a", "c"]
    for model, by_seed in model_points.items():
        model_slug = "7b" if "7B" in model else "3b"
        for seed, points in by_seed.items():
            for point_index, mib_s in enumerate(points):
                point_slug = "nodelay" if mib_s is None else str(mib_s).replace(".", "p")
                for system, scenario in (
                    ("PrimeLoRA", "v2_full"),
                    ("ServerlessLLM", "serverlessllm_new"),
                ):
                    serverless = system == "ServerlessLLM"
                    _write_sensitivity_result(
                        root
                        / model_slug
                        / f"seed_{seed}"
                        / f"{system}_{point_slug}_{point_index}_result.json",
                        system=system,
                        scenario=scenario,
                        seed=seed,
                        sequence=sequence,
                        bandwidth=_bandwidth_metadata(
                            mib_s,
                            serverless=serverless,
                        ),
                        model=model,
                    )


def _write_formal_workload_matrix(root: Path) -> None:
    profiles = {
        # The mode, not the retained interval field, disables rotation.
        "stationary": ("stationary", 500, 1.0, 0.0),
        "rotation100": ("abrupt", 100, 1.0, 0.0),
        "submitted_main_legacy_rotation500": ("legacy", 500, 1.0, 0.75),
        "abrupt_rotation500_overlap0": ("abrupt", 500, 1.0, 0.0),
        "rotation2000": ("abrupt", 2000, 1.0, 0.0),
        "zipf06": ("abrupt", 500, 0.6, 0.0),
        "zipf14": ("abrupt", 500, 1.4, 0.0),
        "gradual": ("gradual", 500, 1.0, 0.5),
    }
    replicated = {"stationary", "rotation100", "submitted_main_legacy_rotation500"}
    sequence = ["a", "a", "b", "a", "c", "b", "a", "c"]
    for profile_name, (mode, rotation, zipf, overlap) in profiles.items():
        full_seeds = (43, 44, 45) if profile_name in replicated else (43,)
        for seed in full_seeds:
            for system, scenario in (
                ("PrimeLoRA", "v2_full"),
                ("ServerlessLLM", "serverlessllm_new"),
            ):
                _write_sensitivity_result(
                    root
                    / profile_name
                    / f"seed_{seed}"
                    / f"{system}_result.json",
                    system=system,
                    scenario=scenario,
                    seed=seed,
                    sequence=sequence,
                    rotation_mode=mode,
                    rotation_requests=rotation,
                    zipf=zipf,
                    overlap=overlap,
                )
        if profile_name in replicated:
            for seed in (43, 44, 45):
                _write_sensitivity_result(
                    root
                    / profile_name
                    / f"seed_{seed}"
                    / "ElasticOnly_result.json",
                    system="PrimeLoRA",
                    scenario="v2_elastic_only",
                    seed=seed,
                    sequence=sequence,
                    rotation_mode=mode,
                    rotation_requests=rotation,
                    zipf=zipf,
                    overlap=overlap,
                )


class ReadinessV2Tests(unittest.TestCase):
    def test_strict_dispatch_validation_checks_every_successful_lora(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "strict_result.json"
            _write_result(
                path,
                model="model-a",
                run_tag="run-43",
                requests=[
                    _request("good", dispatch_tier="gpu", service_tier="gpu"),
                    _request("missing", dispatch_tier=None, service_tier="host"),
                ],
            )
            with self.assertRaisesRegex(SystemExit, "missing readiness_tier_before_dispatch"):
                readiness.load_scenarios(path, require_dispatch_tier=True)

    def test_same_scenario_is_preserved_by_model_and_run_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for run_tag in ("run-43", "run-44"):
                _write_result(
                    root / f"{run_tag}_result.json",
                    model="/models/model-a",
                    run_tag=run_tag,
                    requests=[_request(run_tag, dispatch_tier="gpu", service_tier="gpu")],
                )
            scenarios = readiness.load_scenarios(root, require_dispatch_tier=True)
            self.assertEqual(
                [(item.model, item.name, item.run_tag) for item in scenarios],
                [
                    ("model-a", "faaslora_full", "run-43"),
                    ("model-a", "faaslora_full", "run-44"),
                ],
            )
            aggregate = readiness.aggregate_scenarios(scenarios)
            self.assertEqual(len(aggregate), 1)
            self.assertEqual(aggregate[0].run_tag, "aggregate:2runs")
            self.assertEqual(len(aggregate[0].records), 2)
            across = readiness.build_across_run_summary(readiness.build_summary(scenarios))
            self.assertEqual(len(across), 1)
            self.assertEqual(across[0]["run_count"], 2)
            self.assertEqual(across[0]["ci_unit"], "independent run_tag")

    def test_diagnostic_gate_accepts_complete_single_cold_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "single_result.json"
            _write_result(
                path,
                model="model-a",
                run_tag="cold-1",
                requests=_diagnostic_requests(40, first_service=20),
                diagnostic={"total": 40},
            )
            scenarios = readiness.load_scenarios(path)
            report = readiness.validate_diagnostic_evidence(
                scenarios,
                aggregate_runs=False,
                expected_total=40,
            )
            self.assertTrue(report["passed"])
            self.assertFalse(report["rerun_required"])
            self.assertEqual(report["groups"][0]["first_service_n"], 20)
            self.assertTrue(report["groups"][0]["runs"][0]["passed_integrity"])

    def test_diagnostic_gate_pools_at_most_two_identical_cold_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for run_tag in ("cold-1", "cold-2"):
                _write_result(
                    root / f"{run_tag}_result.json",
                    model="model-a",
                    run_tag=run_tag,
                    requests=_diagnostic_requests(40, first_service=10),
                    diagnostic={"total": 40, "generation_seed": 43},
                )
            scenarios = readiness.load_scenarios(root)
            report = readiness.validate_diagnostic_evidence(
                scenarios,
                aggregate_runs=True,
                expected_total=40,
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["groups"][0]["run_count"], 2)
            self.assertEqual(report["groups"][0]["first_service_n"], 20)
            self.assertTrue(
                report["groups"][0]["configuration_fingerprints_identical"]
            )

            _write_result(
                root / "cold-3_result.json",
                model="model-a",
                run_tag="cold-3",
                requests=_diagnostic_requests(40, first_service=10),
                diagnostic={"total": 40, "generation_seed": 43},
            )
            too_many = readiness.validate_diagnostic_evidence(
                readiness.load_scenarios(root),
                aggregate_runs=True,
                expected_total=40,
            )
            self.assertFalse(too_many["passed"])
            self.assertTrue(
                any("maximum is 2" in error for error in too_many["integrity_errors"])
            )

    def test_diagnostic_gate_rejects_nonidentical_cold_run_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for run_tag, seed in (("cold-1", 43), ("cold-2", 44)):
                _write_result(
                    root / f"{run_tag}_result.json",
                    model="model-a",
                    run_tag=run_tag,
                    requests=_diagnostic_requests(40, first_service=10),
                    diagnostic={"total": 40, "generation_seed": seed},
                )
            report = readiness.validate_diagnostic_evidence(
                readiness.load_scenarios(root),
                aggregate_runs=True,
                expected_total=40,
            )
            self.assertFalse(report["passed"])
            self.assertFalse(report["rerun_required"])
            self.assertTrue(
                any(
                    "configurations are not identical" in error
                    for error in report["integrity_errors"]
                )
            )

    def test_diagnostic_cli_first_service_shortfall_exits_and_requests_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "short_result.json"
            output = root / "analysis"
            _write_result(
                result_path,
                model="model-a",
                run_tag="cold-1",
                requests=_diagnostic_requests(4000, first_service=19),
                diagnostic={"total": 4000},
            )
            argv = [
                "analyze_service_readiness.py",
                "--input",
                str(result_path),
                "--output",
                str(output),
                "--diagnostic-gates",
            ]
            with mock.patch("sys.argv", argv):
                with self.assertRaisesRegex(SystemExit, "rerun_required=true"):
                    readiness.main()
            report = json.loads(
                (output / "readiness_diagnostic_gate.json").read_text(encoding="utf-8")
            )
            self.assertFalse(report["passed"])
            self.assertTrue(report["rerun_required"])
            self.assertEqual(report["groups"][0]["first_service_n"], 19)

    def test_diagnostic_gate_checks_completion_phases_scaleups_and_tier_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid_result.json"
            requests = _diagnostic_requests(40, first_service=20)
            requests[0]["adapter_gpu_ready_before_dispatch"] = False
            _write_result(
                path,
                model="model-a",
                run_tag="cold-1",
                requests=requests,
                diagnostic={
                    "total": 40,
                    "completed": 39,
                    "failed": 1,
                    "phase_count": 7,
                    "scale_up_events": 7,
                },
            )
            report = readiness.validate_diagnostic_evidence(
                readiness.load_scenarios(path),
                aggregate_runs=False,
                expected_total=40,
            )
            self.assertFalse(report["passed"])
            errors = "\n".join(report["integrity_errors"])
            self.assertIn("completed=39", errors)
            self.assertIn("failed=1", errors)
            self.assertIn("phase_count=7", errors)
            self.assertIn("scale_up_events=7", errors)
            self.assertIn("tier invariant conflicts=1", errors)

    def test_transition_csv_separates_observation_from_staleness_claim(self) -> None:
        scenario = readiness.ScenarioRecords(
            name="faaslora_full",
            model="model-a",
            run_tag="run-43",
            source=Path("source.json"),
            records=[
                _request("stable", dispatch_tier="gpu", service_tier="gpu"),
                _request("promoted", dispatch_tier="host", service_tier="gpu"),
            ],
            dispatch_tier_records=2,
        )
        summary, audit = readiness.build_transition_outputs([scenario])
        self.assertEqual(sum(row["n"] for row in summary), 2)
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["transition"], "promoted_before_service")
        self.assertTrue(audit[0]["potentially_stale_signal"])

    def test_manifest_proxy_caveat_is_conditional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            strict = readiness.ScenarioRecords(
                name="faaslora_full",
                model="model-a",
                run_tag="run-43",
                source=Path("strict.json"),
                records=[_request("strict", dispatch_tier="gpu", service_tier="gpu")],
                dispatch_tier_records=1,
            )
            strict_manifest = root / "strict_manifest.json"
            readiness.write_manifest(
                strict_manifest,
                [strict],
                [],
                {},
                require_dispatch_tier=True,
                aggregate_runs=False,
            )
            self.assertIsNone(json.loads(strict_manifest.read_text())["field_caveat"])

            proxy = readiness.ScenarioRecords(
                name="faaslora_full",
                model="model-a",
                run_tag="legacy",
                source=Path("legacy.json"),
                records=[_request("legacy", dispatch_tier=None, service_tier="host")],
                dispatch_tier_records=0,
            )
            proxy_manifest = root / "proxy_manifest.json"
            readiness.write_manifest(
                proxy_manifest,
                [proxy],
                [],
                {},
                require_dispatch_tier=False,
                aggregate_runs=False,
            )
            self.assertIn(
                "service-time readiness proxy",
                json.loads(proxy_manifest.read_text())["field_caveat"],
            )

    def test_fresh_output_and_publish_paths_refuse_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            occupied = root / "occupied"
            occupied.mkdir()
            (occupied / "old.pdf").write_text("old", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                readiness._prepare_output_dir(occupied)

            output = root / "output"
            readiness._prepare_output_dir(output)
            (output / "figure.pdf").write_text("new", encoding="utf-8")
            publish = root / "publish"
            readiness._publish_figures(output, publish, ["figure.pdf"])
            with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                readiness._publish_figures(output, publish, ["figure.pdf"])


class FigureFormulaTests(unittest.TestCase):
    def test_lower_is_better_uses_conventional_relative_delta(self) -> None:
        self.assertAlmostEqual(
            plot_paper_figures._improvement_pct(100.0, 80.0, higher_is_better=False),
            20.0,
        )
        self.assertAlmostEqual(
            plot_paper_figures._improvement_pct(100.0, 120.0, higher_is_better=False),
            -20.0,
        )

    def test_higher_is_better_remains_relative_to_reference(self) -> None:
        self.assertAlmostEqual(
            plot_paper_figures._improvement_pct(100.0, 120.0, higher_is_better=True),
            20.0,
        )

    def test_idle_factor_changes_only_serverless_lifecycle_cost(self) -> None:
        serverless = plot_paper_figures.MainSystemData(
            key="faaslora",
            label="PrimeLoRA",
            source=Path("prime.json"),
            metrics={
                "is_serverless": 1.0,
                "infra_startup_gpu_seconds": 10.0,
                "infra_active_gpu_seconds": 30.0,
                "infra_idle_ready_gpu_seconds": 60.0,
                "gpu_cost_rate_usd_per_s": 2.0,
                "completed": 10.0,
                "cost_invocation_usd": 0.5,
                "cost_req_usd": 20.5,
                "e2e_avg_ms": 1000.0,
            },
        )
        serverful = plot_paper_figures.MainSystemData(
            key="sglang",
            label="SGLang",
            source=Path("sglang.json"),
            metrics={
                "is_serverless": 0.0,
                "cost_req_usd": 7.0,
                "e2e_avg_ms": 1000.0,
            },
        )
        self.assertAlmostEqual(plot_paper_figures._cost_at_idle_factor(serverless, 0.0), 8.5)
        self.assertAlmostEqual(plot_paper_figures._cost_at_idle_factor(serverless, 1.0), 20.5)
        self.assertAlmostEqual(plot_paper_figures._cost_at_idle_factor(serverful, 0.0), 7.0)
        self.assertAlmostEqual(plot_paper_figures._cost_at_idle_factor(serverful, 1.0), 7.0)

    def test_v2_fig9_uses_seed_level_ci_and_paired_relative_formula(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            for seed in (43, 44, 45):
                for scenario, ttft, e2e, cost, ce in (
                    ("v2_elastic_only", 100.0 + (seed - 44), 200.0, 0.0100, 500.0),
                    ("v2_hit_aware_preparation", 92.0, 192.0, 0.0098, 535.0),
                    ("v2_hierarchical_no_coord", 86.0, 186.0, 0.0094, 565.0),
                    ("v2_full", 80.0 + 0.8 * (seed - 44), 180.0, 0.0090, 600.0),
                ):
                    _write_v2_ablation_result(
                        inputs / f"seed_{seed}" / f"{scenario}_result.json",
                        model="model-7b",
                        scenario=scenario,
                        seed=seed,
                        ttft_p95=ttft,
                        e2e_avg=e2e,
                        cost=cost,
                        ce=ce,
                    )
            loaded = plot_paper_figures.load_v2_ablation_results([inputs])
            self.assertEqual(len(loaded), 12)
            self.assertEqual(
                plot_paper_figures.V2_ABLATION_SCENARIOS,
                (
                    "v2_elastic_only",
                    "v2_hit_aware_preparation",
                    "v2_hierarchical_no_coord",
                    "v2_full",
                ),
            )
            output = root / "output"
            plot_paper_figures.plot_v2_fig9_ablation([inputs], output)
            with (output / "fig9_v2_ablation_relative_summary.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            ttft = next(
                row
                for row in rows
                if row["scenario"] == "v2_full" and row["metric"] == "p95_overall_ttft_ms"
            )
            ce = next(
                row
                for row in rows
                if row["scenario"] == "v2_full" and row["metric"] == "monetary_ce"
            )
            self.assertEqual(int(ttft["paired_seed_count"]), 3)
            self.assertAlmostEqual(float(ttft["improvement_pct_mean"]), 20.0)
            self.assertAlmostEqual(float(ce["improvement_pct_mean"]), 20.0)
            self.assertTrue((output / "fig9_v2_ablation.pdf").is_file())
            with (output / "fig9_v2_ablation_adjacent_increment_summary.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                adjacent = list(csv.DictReader(handle))
            self.assertEqual({row["mechanism"] for row in adjacent}, {"M1", "M2", "M3"})
            self.assertTrue(all(int(row["paired_seed_count"]) == 3 for row in adjacent))
            references = {
                row["mechanism"]: row["reference_scenario"]
                for row in adjacent
            }
            self.assertEqual(references["M1"], "v2_elastic_only")
            self.assertEqual(references["M2"], "v2_hit_aware_preparation")
            self.assertEqual(references["M3"], "v2_hierarchical_no_coord")
            with (output / "fig9_v2_ablation_per_seed.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                full_table_row = next(csv.DictReader(handle))
            for field in (
                "avg_overall_ttft_ms",
                "avg_tpot_ms",
                "throughput_tok_per_s",
                "routing_decision_count",
                "scale_up_events_with_planned_adapters",
                "host_promotion_completed_count",
                "gpu_admission_decision_count",
                "gpu_admission_admit_count",
                "gpu_admission_defer_count",
                "gpu_admission_reject_count",
            ):
                self.assertIn(field, full_table_row)

    def test_v2_fig9_rejects_provenance_dispatch_contract_and_activation_errors(self) -> None:
        def write_pair(root: Path, **full_overrides: object) -> Path:
            inputs = root / "inputs"
            _write_v2_ablation_result(
                inputs / "elastic_result.json",
                model="model-7b",
                scenario="v2_elastic_only",
                seed=43,
                ttft_p95=100.0,
                e2e_avg=200.0,
                cost=0.01,
                ce=500.0,
            )
            _write_v2_ablation_result(
                inputs / "full_result.json",
                model="model-7b",
                scenario="v2_full",
                seed=43,
                ttft_p95=80.0,
                e2e_avg=180.0,
                cost=0.009,
                ce=600.0,
                **full_overrides,
            )
            return inputs

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SystemExit, "trace_hashes"):
                plot_paper_figures.load_v2_ablation_results(
                    [write_pair(Path(tmp), trace_sha="e" * 64)]
                )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SystemExit, "dispatch-time"):
                plot_paper_figures.load_v2_ablation_results(
                    [write_pair(Path(tmp), missing_dispatch_tier=True)]
                )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SystemExit, "request-map SHA mismatch"):
                plot_paper_figures.load_v2_ablation_results(
                    [write_pair(Path(tmp), corrupt_contract_map=True)]
                )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SystemExit, "readiness-aware routing activation"):
                plot_paper_figures.load_v2_ablation_results(
                    [write_pair(Path(tmp), corrupt_activation=True)]
                )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SystemExit, "admission outcome invariant"):
                plot_paper_figures.load_v2_ablation_results(
                    [write_pair(Path(tmp), admission_outcomes=(1, 1, 0))]
                )
        with tempfile.TemporaryDirectory() as tmp:
            inputs = Path(tmp) / "inputs"
            _write_v2_ablation_result(
                inputs / "elastic_result.json",
                model="model-7b",
                scenario="v2_elastic_only",
                seed=43,
                ttft_p95=100.0,
                e2e_avg=200.0,
                cost=0.01,
                ce=500.0,
                admission_outcomes=(1, 0, 0),
            )
            with self.assertRaisesRegex(SystemExit, "disabled admission"):
                plot_paper_figures.load_v2_ablation_results([inputs])

    def test_v2_formal_ablation_matrix_accepts_exact_and_rejects_missing_or_extra(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inputs = Path(tmp) / "formal_ablation"
            _write_formal_ablation_matrix(inputs)
            results = plot_paper_figures.load_v2_ablation_results(
                [inputs], formal_matrix=True
            )
            self.assertEqual(len(results), 18)

            with self.assertRaisesRegex(SystemExit, "formal A2/A3.*missing"):
                plot_paper_figures.validate_v2_ablation_formal_matrix(results[:-1])

            extra = replace(
                results[0],
                seed=42,
                source=Path("unexpected_seed42_result.json"),
            )
            with self.assertRaisesRegex(SystemExit, "formal A2/A3.*extra"):
                plot_paper_figures.validate_v2_ablation_formal_matrix(
                    [*results, extra]
                )

    def test_v2_formal_ablation_rejects_wrong_axes_model_contract_and_nonfeature_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inputs = Path(tmp) / "formal_ablation"
            _write_formal_ablation_matrix(inputs)
            results = plot_paper_figures.load_v2_ablation_results([inputs])

            with self.assertRaisesRegex(SystemExit, "unsupported model identity"):
                plot_paper_figures.validate_v2_ablation_formal_matrix(
                    [replace(results[0], model="Qwen-2.5-7B"), *results[1:]]
                )
            with self.assertRaisesRegex(SystemExit, "expected 4000/4000"):
                plot_paper_figures.validate_v2_ablation_formal_matrix(
                    [replace(results[0], total=1000, completed=1000), *results[1:]]
                )
            wrong_axes = dict(results[0].formal_axes)
            wrong_axes["bandwidth_mib_s"] = 119.2093
            with self.assertRaisesRegex(SystemExit, "wrong bandwidth_mib_s"):
                plot_paper_figures.validate_v2_ablation_formal_matrix(
                    [replace(results[0], formal_axes=wrong_axes), *results[1:]]
                )
            with self.assertRaisesRegex(SystemExit, "generation_contract must be legacy"):
                plot_paper_figures.validate_v2_ablation_formal_matrix(
                    [replace(results[0], generation_contract="fixed_length_greedy_v1"), *results[1:]]
                )
            with self.assertRaisesRegex(SystemExit, "non-feature frozen configuration drift"):
                plot_paper_figures.validate_v2_ablation_formal_matrix(
                    [
                        replace(
                            results[0],
                            non_feature_frozen_config_sha256="9" * 64,
                        ),
                        *results[1:],
                    ]
                )


class SensitivityV2Tests(unittest.TestCase):
    def test_formal_bandwidth_matrix_accepts_exact_and_rejects_missing_or_extra(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inputs = Path(tmp) / "formal_bandwidth"
            _write_formal_bandwidth_matrix(inputs)
            observations = plot_paper_sensitivity.load_v2_sensitivity_observations(
                [inputs]
            )
            self.assertEqual(len(observations), 30)
            plot_paper_sensitivity.validate_formal_bandwidth_matrix(observations)

            with self.assertRaisesRegex(SystemExit, "formal A6.*missing"):
                plot_paper_sensitivity.validate_formal_bandwidth_matrix(
                    observations[:-1]
                )
            extra = replace(
                observations[0],
                seed=42,
                source=Path("unexpected_seed42_bandwidth_result.json"),
            )
            with self.assertRaisesRegex(SystemExit, "formal A6.*extra"):
                plot_paper_sensitivity.validate_formal_bandwidth_matrix(
                    [*observations, extra]
                )

    def test_formal_workload_matrix_accepts_exact_and_rejects_missing_or_extra(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inputs = Path(tmp) / "formal_workload"
            _write_formal_workload_matrix(inputs)
            observations = plot_paper_sensitivity.load_v2_sensitivity_observations(
                [inputs]
            )
            self.assertEqual(len(observations), 37)
            plot_paper_sensitivity.validate_formal_workload_matrix(observations)

            with self.assertRaisesRegex(SystemExit, "formal C4.*missing"):
                plot_paper_sensitivity.validate_formal_workload_matrix(
                    observations[:-1]
                )
            extra = replace(
                observations[0],
                seed=42,
                source=Path("unexpected_seed42_workload_result.json"),
            )
            with self.assertRaisesRegex(SystemExit, "formal C4.*extra"):
                plot_paper_sensitivity.validate_formal_workload_matrix(
                    [*observations, extra]
                )

    def test_workload_characteristics_and_system_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            sequence = ["a", "a", "b", "a", "c", "b", "a", "c"]
            _write_sensitivity_result(
                inputs / "prime_result.json",
                system="PrimeLoRA",
                scenario="v2_full",
                seed=43,
                sequence=sequence,
            )
            _write_sensitivity_result(
                inputs / "serverless_result.json",
                system="ServerlessLLM",
                scenario="serverlessllm_new",
                seed=43,
                sequence=sequence,
            )
            observations = plot_paper_sensitivity.load_v2_sensitivity_observations([inputs])
            self.assertEqual({item.system_key for item in observations}, {"faaslora", "serverlessllm"})
            output = root / "output"
            plot_paper_sensitivity.plot_workload_sensitivity([inputs], output)
            with (output / "workload_characteristics_per_seed.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(int(row["actual_unique_adapters"]), 3)
            self.assertAlmostEqual(float(row["first_touch_ratio"]), 3 / 8)
            self.assertAlmostEqual(
                float(row["empirical_observed_set_turnover_mean"]), 1 / 3
            )
            expected_entropy = -(4 / 8 * math.log(4 / 8) + 2 * (2 / 8 * math.log(2 / 8)))
            self.assertAlmostEqual(float(row["entropy_nats"]), expected_entropy)
            self.assertTrue((output / "fig_workload_sensitivity.pdf").is_file())
            self.assertTrue((output / "table_workload_sensitivity.tex").is_file())

    def test_workload_rejects_missing_or_mismatched_shared_hashes(self) -> None:
        sequence = ["a", "a", "b", "a"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            _write_sensitivity_result(
                inputs / "prime_result.json",
                system="PrimeLoRA",
                scenario="v2_full",
                seed=43,
                sequence=sequence,
                trace_sha="c" * 64,
            )
            _write_sensitivity_result(
                inputs / "serverless_result.json",
                system="ServerlessLLM",
                scenario="serverlessllm_new",
                seed=43,
                sequence=sequence,
                trace_sha="d" * 64,
            )
            with self.assertRaisesRegex(SystemExit, "trace_hashes"):
                plot_paper_sensitivity.plot_workload_sensitivity(
                    [inputs], root / "output"
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            _write_sensitivity_result(
                inputs / "prime_result.json",
                system="PrimeLoRA",
                scenario="v2_full",
                seed=43,
                sequence=sequence,
                trace_sha="",
            )
            with self.assertRaisesRegex(SystemExit, "shared_trace_sha256"):
                plot_paper_sensitivity.plot_workload_sensitivity(
                    [inputs], root / "output"
                )

    def test_bandwidth_strict_hash_and_aggregate_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            sequence = ["a", "a", "b", "a", "c", "b", "a", "c"]
            limited_mib = 11.9209
            prime_limited = {
                "limit_mode": "aggregate_application_layer_local_sim",
                "configured_mib_s": limited_mib,
                "configured_gbit_s": limited_mib * 8 * 1024 * 1024 / 1_000_000_000,
                "transfer_count": 8,
                "total_bytes": 8 * 1024 * 1024,
                "reservation_span_s": 1.0,
                "total_injected_wait_s": 0.5,
                "achieved_reserved_mib_s": 8.0,
            }
            serverless_limited = {
                **prime_limited,
                "limit_mode": "file_aggregate_reservation",
            }
            prime_no_delay = {
                "limit_mode": "local_sim_no_delay",
                "configured_mib_s": None,
                "configured_gbit_s": None,
                "transfer_count": 8,
                "total_bytes": 8 * 1024 * 1024,
                "reservation_span_s": 0.1,
                "total_injected_wait_s": 0.0,
                "achieved_reserved_mib_s": 80.0,
            }
            serverless_no_delay = {**prime_no_delay, "limit_mode": "file_no_delay"}
            for system, scenario in (
                ("PrimeLoRA", "v2_full"),
                ("ServerlessLLM", "serverlessllm_new"),
            ):
                limited = prime_limited if system == "PrimeLoRA" else serverless_limited
                no_delay = prime_no_delay if system == "PrimeLoRA" else serverless_no_delay
                _write_sensitivity_result(
                    inputs / f"{system}_limited_result.json",
                    system=system,
                    scenario=scenario,
                    seed=43,
                    sequence=sequence,
                    bandwidth=limited,
                )
                _write_sensitivity_result(
                    inputs / f"{system}_nodelay_result.json",
                    system=system,
                    scenario=scenario,
                    seed=43,
                    sequence=sequence,
                    bandwidth=no_delay,
                )
            output = root / "output"
            plot_paper_sensitivity.plot_bandwidth_sensitivity([inputs], output)
            with (output / "bandwidth_sensitivity_per_seed.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["bandwidth_key"] for row in rows}, {"mib_s=11.920900", "no-delay"})
            self.assertEqual({row["limit_mode"] for row in rows}, {"aggregate", "no-delay"})
            self.assertEqual(
                {row["raw_limit_mode"] for row in rows},
                {
                    "aggregate_application_layer_local_sim",
                    "file_aggregate_reservation",
                    "local_sim_no_delay",
                    "file_no_delay",
                },
            )
            self.assertTrue((output / "fig_bandwidth_sensitivity.pdf").is_file())
            self.assertTrue((output / "table_bandwidth_sensitivity.tex").is_file())

            broken = root / "broken"
            _write_sensitivity_result(
                broken / "prime_result.json",
                system="PrimeLoRA",
                scenario="v2_full",
                seed=43,
                sequence=sequence,
                bandwidth=prime_limited,
                trace_sha="c" * 64,
            )
            _write_sensitivity_result(
                broken / "serverless_result.json",
                system="ServerlessLLM",
                scenario="serverlessllm_new",
                seed=43,
                sequence=sequence,
                bandwidth=serverless_limited,
                trace_sha="d" * 64,
            )
            with self.assertRaisesRegex(SystemExit, "trace_hashes"):
                plot_paper_sensitivity.plot_bandwidth_sensitivity(
                    [broken], root / "broken_output"
                )


class FormalProvenanceGateTests(unittest.TestCase):
    @staticmethod
    def _integrity(path: Path) -> dict:
        return {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def _write_ablation_manifest(
        self,
        path: Path,
        records: list[tuple[Path, str]],
        **overrides: object,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        non_feature_hash = str(
            overrides.get("non_feature_frozen_config_sha256") or "e" * 64
        )
        heldout_seed = next(
            (
                seed
                for seed in (43, 44, 45)
                if f"seed{seed}" in str(path)
                or any(f"seed{seed}" in str(source) for source, _ in records)
            ),
            43,
        )
        family = {
            "campaign_kind": "v2_a2_a3_ablation",
            "model_profile": "llama2_7b_main_v2_publicmix",
            "dataset_profile": "azure_sharegpt_rep4000",
            "workload_profile": "llama2_7b_auto500_formal4000_s8",
            "selected_num_adapters": 500,
            "gpu_ids": ["0", "1", "2", "3"],
            "generation_contract": "legacy",
        }
        config = path.parent / "frozen_experiments.yaml"
        config.write_text("profiles: {}\n", encoding="utf-8")
        config_identity = {"path": str(config.resolve()), **self._integrity(config)}

        protocol = path.parent / "protocol"
        protocol.mkdir(parents=True, exist_ok=True)
        validation_round = protocol / "seed41_validation"
        validation_round.mkdir(parents=True, exist_ok=True)
        validation_result = validation_round / "seed41_v2_full_result.json"
        validation_result.write_text(
            json.dumps(
                {
                    "metadata": {
                        "formal_run": True,
                        "trace_role": "validation",
                        "non_feature_frozen_config_sha256": non_feature_hash,
                    }
                }
            ),
            encoding="utf-8",
        )
        validation_manifest = validation_round / "MANIFEST.json"
        validation_payload = {
            "status": "complete",
            "formal_run": True,
            "trace_role": "validation",
            "scenarios": ["v2_full"],
            "configuration_family": family,
            "config_snapshot": config_identity,
            "non_feature_frozen_config_sha256": non_feature_hash,
            "non_feature_frozen_config_consistent": True,
            "code_snapshot": {
                "git_commit": "faaslora-commit",
                "source_clean_for_formal": True,
            },
            "shared_trace": {"sampling_seed": 41, "requests": 1000},
            "entries": [
                {
                    "scenario": "v2_full",
                    "result_json": str(validation_result.resolve()),
                    "non_feature_frozen_config_sha256": non_feature_hash,
                    **self._integrity(validation_result),
                }
            ],
        }
        validation_manifest.write_text(
            json.dumps(validation_payload), encoding="utf-8"
        )
        successful_validation = {
            "seed": 41,
            "non_feature_frozen_config_sha256": non_feature_hash,
            "manifest": str(validation_manifest.resolve()),
            "manifest_sha256": hashlib.sha256(
                validation_manifest.read_bytes()
            ).hexdigest(),
            "manifest_bytes": validation_manifest.stat().st_size,
            "source_commit": "faaslora-commit",
            "config_path": str(config.resolve()),
            "config_sha256": config_identity["sha256"],
        }
        family_id = hashlib.sha256(
            json.dumps(
                family, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        evidence = protocol / "seed41_validation_evidence.json"
        evidence_payload = {
            "schema_version": "eurosys27_v2_faaslora_ablation_validation_evidence_v1",
            "configuration_family": family,
            "configuration_family_id": family_id,
            "selected_non_feature_frozen_config_sha256": non_feature_hash,
            "successful_validation": successful_validation,
            "heldout_seed": heldout_seed,
            "heldout_requests": 4000,
            "heldout_round_dir": str(path.parent.resolve()),
            "source_commit": "faaslora-commit",
            "config_path": str(config.resolve()),
            "config_sha256": config_identity["sha256"],
            "registry_path": str((path.parent / "registry.json").resolve()),
            "registry_sha256_after_freeze": "f" * 64,
        }
        evidence.write_text(json.dumps(evidence_payload), encoding="utf-8")
        payload = {
            "status": "complete",
            "formal_run": True,
            "trace_role": "heldout",
            "round_dir": str(path.parent.resolve()),
            "scenarios": [source.stem for source, _ in records],
            "configuration_family": family,
            "config_snapshot": config_identity,
            "non_feature_frozen_config_sha256": non_feature_hash,
            "non_feature_frozen_config_consistent": True,
            "shared_trace": {"sampling_seed": heldout_seed, "requests": 4000},
            "seed41_validation_evidence": {
                "path": str(evidence.resolve()),
                **self._integrity(evidence),
                "selected_non_feature_frozen_config_sha256": non_feature_hash,
                "successful_validation_manifest": str(
                    validation_manifest.resolve()
                ),
                "successful_validation_manifest_sha256": successful_validation[
                    "manifest_sha256"
                ],
                "successful_validation_manifest_bytes": successful_validation[
                    "manifest_bytes"
                ],
                "source_commit": "faaslora-commit",
                "config_path": str(config.resolve()),
                "config_sha256": config_identity["sha256"],
                "configuration_family_id": family_id,
                "registry_path": evidence_payload["registry_path"],
                "registry_sha256_after_freeze": evidence_payload[
                    "registry_sha256_after_freeze"
                ],
            },
            "code_snapshot": {
                "git_commit": "faaslora-commit",
                "source_clean_for_formal": True,
            },
            "entries": [
                {
                    "scenario": source.stem,
                    "result_json": str(source.resolve()),
                    "system_resolved_config_sha256": config_hash,
                    "non_feature_frozen_config_sha256": non_feature_hash,
                    **self._integrity(source),
                }
                for source, config_hash in records
            ],
        }
        payload.update(overrides)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _write_fair_manifest(
        self,
        path: Path,
        records: list[Path],
        config_hash: str,
        **overrides: object,
    ) -> Path:
        round_dir = path.parent
        payload = {
            "status": "complete",
            "formal_run": True,
            "trace_role": "heldout",
            "source_clean_for_formal": True,
            "system_resolved_config_sha256": config_hash,
            "baseline_git": {"commit": "baseline-commit"},
            "faaslora_git": {"commit": "faaslora-commit"},
            "source_files": {
                str(source.resolve().relative_to(round_dir.resolve())): self._integrity(
                    source
                )
                for source in records
            },
        }
        payload.update(overrides)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_formal_provenance_accepts_both_campaign_manifest_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_hash = "c" * 64
            ablation_sources = []
            for seed in (43, 44, 45):
                source = root / "ablation" / f"seed{seed}_result.json"
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
                self._write_ablation_manifest(
                    source.parent / f"seed{seed}" / "MANIFEST.json",
                    [(source, config_hash)],
                )
                ablation_sources.append(source)

            index = v2_provenance.build_formal_provenance_index(
                [root / "ablation"]
            )
            v2_provenance.validate_formal_analysis_sources(
                index,
                (
                    v2_provenance.FormalAnalysisIdentity(
                        source=source,
                        model="7b",
                        variant="v2_full",
                        seed=seed,
                    )
                    for seed, source in zip((43, 44, 45), ablation_sources)
                ),
                analysis_label="test A2",
            )

            fair_dir = root / "fair"
            prime = fair_dir / "raw" / "prime_result.json"
            serverless = fair_dir / "raw" / "serverless_summary.json"
            prime.parent.mkdir(parents=True, exist_ok=True)
            prime.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "formal_run": True,
                            "trace_role": "heldout",
                            "system_resolved_config_sha256": config_hash,
                        }
                    }
                ),
                encoding="utf-8",
            )
            serverless.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
            self._write_fair_manifest(
                fair_dir / "MANIFEST.json",
                [prime, serverless],
                config_hash,
            )
            fair_index = v2_provenance.build_formal_provenance_index([fair_dir])
            v2_provenance.validate_formal_analysis_sources(
                fair_index,
                [
                    v2_provenance.FormalAnalysisIdentity(
                        prime, "7b", "faaslora", 43
                    ),
                    v2_provenance.FormalAnalysisIdentity(
                        serverless, "7b", "serverlessllm", 43
                    ),
                ],
                analysis_label="test A6",
            )

    def test_formal_provenance_rejects_loose_raw_and_bad_manifest_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw_result.json"
            source.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "loose raw JSON"):
                v2_provenance.build_formal_provenance_index([source])

            cases = (
                ({"status": "incomplete"}, "status must be 'complete'"),
                ({"formal_run": False}, "formal_run must be true"),
                ({"trace_role": "validation"}, "trace_role must be 'heldout'"),
                (
                    {
                        "code_snapshot": {
                            "git_commit": "faaslora-commit",
                            "source_clean_for_formal": False,
                        }
                    },
                    "source_clean_for_formal must be true",
                ),
                (
                    {
                        "code_snapshot": {
                            "git_commit": "",
                            "source_clean_for_formal": True,
                        }
                    },
                    "committed source revision is empty",
                ),
            )
            for index, (override, message) in enumerate(cases):
                manifest = self._write_ablation_manifest(
                    root / f"case{index}" / "MANIFEST.json",
                    [(source, "a" * 64)],
                    **override,
                )
                with self.subTest(message=message), self.assertRaisesRegex(
                    SystemExit, message
                ):
                    v2_provenance.build_formal_provenance_index([manifest])

    def test_formal_provenance_rejects_uncovered_tampered_and_config_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source43 = root / "seed43_result.json"
            source44 = root / "seed44_result.json"
            uncovered = root / "seed45_result.json"
            for source in (source43, source44, uncovered):
                source.write_text("{}", encoding="utf-8")
            manifest = self._write_ablation_manifest(
                root / "MANIFEST.json",
                [(source43, "a" * 64), (source44, "b" * 64)],
            )
            index = v2_provenance.build_formal_provenance_index([manifest])

            with self.assertRaisesRegex(SystemExit, "no MANIFEST.json record"):
                v2_provenance.validate_formal_analysis_sources(
                    index,
                    [
                        v2_provenance.FormalAnalysisIdentity(
                            uncovered, "7b", "v2_full", 45
                        )
                    ],
                    analysis_label="test",
                )

            with self.assertRaisesRegex(SystemExit, "configuration drift"):
                v2_provenance.validate_formal_analysis_sources(
                    index,
                    [
                        v2_provenance.FormalAnalysisIdentity(
                            source43, "7b", "v2_full", 43
                        ),
                        v2_provenance.FormalAnalysisIdentity(
                            source44, "7b", "v2_full", 44
                        ),
                    ],
                    analysis_label="test",
                )

            source43.write_text('{"tampered": true}', encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "byte-count mismatch|SHA-256 mismatch"):
                v2_provenance.validate_formal_analysis_sources(
                    index,
                    [
                        v2_provenance.FormalAnalysisIdentity(
                            source43, "7b", "v2_full", 43
                        )
                    ],
                    analysis_label="test",
                )

    def test_formal_provenance_revalidates_seed41_freeze_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "result.json"
            source.write_text("{}", encoding="utf-8")

            missing = self._write_ablation_manifest(
                root / "missing" / "MANIFEST.json",
                [(source, "a" * 64)],
                seed41_validation_evidence=None,
            )
            with self.assertRaisesRegex(SystemExit, "missing seed41_validation_evidence"):
                v2_provenance.build_formal_provenance_index([missing])

            tampered_evidence_manifest = self._write_ablation_manifest(
                root / "tampered_evidence" / "MANIFEST.json",
                [(source, "a" * 64)],
            )
            tampered_payload = json.loads(
                tampered_evidence_manifest.read_text(encoding="utf-8")
            )
            evidence_path = Path(tampered_payload["seed41_validation_evidence"]["path"])
            evidence_path.write_text(
                evidence_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "byte-count mismatch|SHA-256 mismatch"):
                v2_provenance.build_formal_provenance_index(
                    [tampered_evidence_manifest]
                )

            tampered_validation_manifest = self._write_ablation_manifest(
                root / "tampered_validation" / "MANIFEST.json",
                [(source, "a" * 64)],
            )
            heldout = json.loads(
                tampered_validation_manifest.read_text(encoding="utf-8")
            )
            validation_path = Path(
                heldout["seed41_validation_evidence"][
                    "successful_validation_manifest"
                ]
            )
            validation_path.write_text(
                validation_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "byte-count mismatch|SHA-256 mismatch"):
                v2_provenance.build_formal_provenance_index(
                    [tampered_validation_manifest]
                )

            wrong_hash = self._write_ablation_manifest(
                root / "wrong_hash" / "MANIFEST.json",
                [(source, "a" * 64)],
            )
            wrong_payload = json.loads(wrong_hash.read_text(encoding="utf-8"))
            wrong_payload["non_feature_frozen_config_sha256"] = "d" * 64
            wrong_hash.write_text(json.dumps(wrong_payload), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "differs from seed41 selection"):
                v2_provenance.build_formal_provenance_index([wrong_hash])

    def test_formal_plot_entry_points_preflight_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loose = root / "loose_result.json"
            loose.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "loose raw JSON"):
                plot_paper_figures.plot_v2_fig9_ablation(
                    [loose], root / "fig9", formal_matrix=True
                )
            with self.assertRaisesRegex(SystemExit, "loose raw JSON"):
                plot_paper_sensitivity.plot_bandwidth_sensitivity(
                    [loose], root / "a6", formal_matrix=True
                )
            with self.assertRaisesRegex(SystemExit, "loose raw JSON"):
                plot_paper_sensitivity.plot_workload_sensitivity(
                    [loose], root / "c4", formal_matrix=True
                )


class FormalMatrixCliTests(unittest.TestCase):
    def test_fig9_cli_forwards_formal_matrix_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(
                sys,
                "argv",
                [
                    "plot_paper_figures.py",
                    "--figure",
                    "fig9_v2_ablation",
                    "--input",
                    str(root / "inputs"),
                    "--out-dir",
                    str(root / "output"),
                    "--formal-matrix",
                ],
            ), mock.patch.object(
                plot_paper_figures, "plot_v2_fig9_ablation"
            ) as plotter:
                plot_paper_figures.main()
            self.assertTrue(plotter.call_args.kwargs["formal_matrix"])

    def test_sensitivity_cli_forwards_formal_matrix_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(
                sys,
                "argv",
                [
                    "plot_paper_sensitivity.py",
                    "--figure",
                    "bandwidth",
                    "--input",
                    str(root / "inputs"),
                    "--out-dir",
                    str(root / "output"),
                    "--formal-matrix",
                ],
            ), mock.patch.object(
                plot_paper_sensitivity, "plot_bandwidth_sensitivity"
            ) as plotter:
                plot_paper_sensitivity.main()
            self.assertTrue(plotter.call_args.kwargs["formal_matrix"])


if __name__ == "__main__":
    unittest.main()
