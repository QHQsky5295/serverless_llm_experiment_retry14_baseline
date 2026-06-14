#!/usr/bin/env python3
"""Verify a SLINFER formal table from raw requests and frozen snapshots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


EXPECTED_MODELS = {
    "3b": ("Llama-3.2 3B", 180.0, 14.0),
    "7b": ("Llama-2 7B", 440.0, 32.0),
}


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise AssertionError("empty percentile sample")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: float, expected: float, tolerance: float = 1e-6) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"value mismatch: actual={actual}, expected={expected}")


def verify_hash(row: dict[str, str], prefix: str) -> Path:
    path = Path(row[f"{prefix}_path"])
    if not path.is_file() or sha256(path) != row[f"{prefix}_sha256"]:
        raise AssertionError(f"{prefix} evidence mismatch: {path}")
    return path


def scenario_from(summary: dict) -> dict:
    summaries = summary["scenario_summaries"]
    return summaries[0] if isinstance(summaries, list) else next(iter(summaries.values()))


def verify_row(row: dict[str, str]) -> None:
    model_key = row["model_key"]
    if model_key not in EXPECTED_MODELS:
        raise AssertionError(f"unexpected model key: {model_key}")
    model, ttft_slo, tpot_slo = EXPECTED_MODELS[model_key]
    if row["system"] != "SLINFER" or row["system_key"] != "slinfer":
        raise AssertionError(f"{model_key}: system mismatch")
    if row["model"] != model:
        raise AssertionError(f"{model_key}: model mismatch")
    if row["trace_role"] != "formal4000" or int(row["formal_requests"]) != 4000:
        raise AssertionError(f"{model_key}: formal contract mismatch")
    if float(row["replay_rate"]) != 1.0 or row["slo_profile"] != "paper_nominal":
        raise AssertionError(f"{model_key}: workload or SLO profile mismatch")
    if int(row["gpu_budget"]) != 4:
        raise AssertionError(f"{model_key}: GPU budget mismatch")
    if row["system_mode"] != "official_sota_gpu_only":
        raise AssertionError(f"{model_key}: system mode mismatch")
    if float(row["monitor_tail_s"]) != float(row["keep_alive_s"]) + 2.0:
        raise AssertionError(f"{model_key}: monitor tail does not cover keep-alive")

    raw_path = verify_hash(row, "raw_records")
    summary_path = verify_hash(row, "source_summary")
    manifest_path = verify_hash(row, "manifest")
    trace_path = verify_hash(row, "trace")
    raw = json.loads(raw_path.read_text())
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    scenario = scenario_from(summary)
    requests = [item for item in raw["results"] if item.get("success")]
    failures = [item for item in raw["results"] if not item.get("success")]
    if len(requests) != 4000 or failures:
        raise AssertionError(
            f"{model_key}: expected clean 4000/4000, "
            f"success={len(requests)}, failures={len(failures)}"
        )
    if int(scenario["completed_requests"]) != 4000:
        raise AssertionError(f"{model_key}: summary completion mismatch")
    if Path(manifest["trace_path"]).resolve() != trace_path.resolve():
        raise AssertionError(f"{model_key}: manifest trace mismatch")
    if manifest["trace_sha256"] != row["trace_sha256"]:
        raise AssertionError(f"{model_key}: manifest trace hash mismatch")
    adaptation = json.loads(
        (Path(manifest["frozen_config_dir"]) / "hardware_adaptation.json").read_text()
    )
    if int(row["workers_per_gpu"]) != int(adaptation["workers_per_gpu"]):
        raise AssertionError(f"{model_key}: worker topology mismatch")
    identity = adaptation["model_identity"]
    if identity["config_identity_verified"] is not True:
        raise AssertionError(f"{model_key}: model identity not verified")
    if identity["source_config_sha256"] != identity["converted_config_sha256"]:
        raise AssertionError(f"{model_key}: source/converted model mismatch")
    for field in (
        "source_model_realpath",
        "source_config_sha256",
        "converted_config_sha256",
    ):
        if row[field] != identity[field]:
            raise AssertionError(f"{model_key}: model identity field mismatch: {field}")

    ttfts = [float(item["ttft_ms"]) for item in requests]
    tpots = [float(item["tpot_ms"]) for item in requests]
    ttft_p95 = percentile(ttfts, 0.95)
    tpot_p95 = percentile(tpots, 0.95)
    ttft_attainment = sum(value <= ttft_slo for value in ttfts) / len(ttfts)
    tpot_attainment = sum(value <= tpot_slo for value in tpots) / len(tpots)
    joint_attainment = sum(
        float(item["ttft_ms"]) <= ttft_slo
        and float(item["tpot_ms"]) <= tpot_slo
        for item in requests
    ) / len(requests)
    expected_gate = ttft_p95 <= ttft_slo and tpot_p95 <= tpot_slo
    close(float(row["ttft_target_ms"]), ttft_slo)
    close(float(row["tpot_target_ms"]), tpot_slo)
    close(float(row["ttft_avg_ms"]), float(scenario["avg_ttft_ms"]))
    close(float(row["ttft_p95_ms"]), ttft_p95)
    close(float(row["tpot_avg_ms"]), float(scenario["avg_tpot_ms"]))
    close(float(row["tpot_p95_ms"]), tpot_p95)
    close(float(row["ttft_slo_attainment"]), ttft_attainment)
    close(float(row["tpot_slo_attainment"]), tpot_attainment)
    close(float(row["joint_slo_attainment"]), joint_attainment)
    if (row["paper_slo_gate_pass"] == "true") != expected_gate:
        raise AssertionError(f"{model_key}: SLO gate status mismatch")
    for field in (
        "e2e_avg_ms",
        "e2e_p95_ms",
        "ce",
        "monetary_cost_per_request_usd",
        "monetary_cost_total_usd",
        "allocated_gpu_seconds",
        "active_gpu_seconds",
        "idle_gpu_seconds",
        "startup_gpu_seconds",
    ):
        scenario_field = {
            "e2e_avg_ms": "avg_e2e_ms",
            "e2e_p95_ms": "p95_e2e_ms",
        }.get(field, field)
        close(float(row[field]), float(scenario[scenario_field]))
    close(
        float(row["predeploy_startup_sec"]),
        float(manifest["initial_runtime_startup_sec"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    args = parser.parse_args()
    with args.table.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["model_key"]: row for row in rows}
    if set(indexed) != set(EXPECTED_MODELS) or len(rows) != len(EXPECTED_MODELS):
        raise AssertionError("expected exactly one 3b and one 7b SLINFER row")
    for model_key in ("3b", "7b"):
        verify_row(indexed[model_key])
    print("PASS system=SLINFER models=3b,7b formal_requests=4000")


if __name__ == "__main__":
    main()
