#!/usr/bin/env python3
"""Verify a ServerlessLLM formal table from immutable raw evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


EXPECTED_MODELS = {
    "3b": ("Llama-3.2 3B", 180.0, 14.0, 16),
    "7b": ("Llama-2 7B", 440.0, 32.0, 8),
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
    if not path.is_file():
        raise AssertionError(f"missing {prefix}: {path}")
    if sha256(path) != row[f"{prefix}_sha256"]:
        raise AssertionError(f"{prefix} hash mismatch: {path}")
    return path


def verify_row(row: dict[str, str]) -> None:
    model_key = row["model_key"]
    if model_key not in EXPECTED_MODELS:
        raise AssertionError(f"unexpected model key: {model_key}")
    model, ttft_slo, tpot_slo, target = EXPECTED_MODELS[model_key]
    if row["system_key"] != "serverlessllm" or row["system"] != "ServerlessLLM":
        raise AssertionError(f"{model_key}: unexpected system")
    if row["model"] != model:
        raise AssertionError(f"{model_key}: model label mismatch")
    if row["trace_role"] != "formal4000":
        raise AssertionError(f"{model_key}: trace role mismatch")
    if row["slo_profile"] != "paper_nominal":
        raise AssertionError(f"{model_key}: SLO profile mismatch")
    if float(row["replay_rate"]) != 1.0:
        raise AssertionError(f"{model_key}: replay rate mismatch")
    if int(row["formal_requests"]) != 4000:
        raise AssertionError(f"{model_key}: formal request count mismatch")
    if int(row["gpu_budget"]) != 4:
        raise AssertionError(f"{model_key}: GPU budget mismatch")
    if int(row["target_concurrency"]) != target:
        raise AssertionError(f"{model_key}: frozen target mismatch")
    if int(row["keep_alive_s"]) != 60:
        raise AssertionError(f"{model_key}: keep-alive mismatch")
    if int(row["min_instances"]) != 1 or int(row["max_instances"]) != 4:
        raise AssertionError(f"{model_key}: instance bounds mismatch")

    raw_path = verify_hash(row, "raw_records")
    summary_path = verify_hash(row, "source_summary")
    manifest_path = verify_hash(row, "manifest")
    trace_path = verify_hash(row, "trace")
    raw = json.loads(raw_path.read_text())
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    result = summary["comparison_table"][0]
    exact = next(iter(summary["scenario_summaries"].values()))
    requests = [item for item in raw["results"] if item.get("success")]
    failures = [item for item in raw["results"] if not item.get("success")]
    if len(requests) != 4000 or failures:
        raise AssertionError(
            f"{model_key}: expected clean 4000/4000, "
            f"success={len(requests)}, failures={len(failures)}"
        )
    if int(result["completed"]) != 4000 or int(result["total"]) != 4000:
        raise AssertionError(f"{model_key}: summary request totals mismatch")
    if manifest["trace_role"] != "formal4000":
        raise AssertionError(f"{model_key}: manifest trace role mismatch")
    if int(manifest["max_requests"]) != 4000:
        raise AssertionError(f"{model_key}: manifest request count mismatch")
    if Path(manifest["trace_path"]).resolve() != trace_path.resolve():
        raise AssertionError(f"{model_key}: manifest trace path mismatch")
    if manifest["trace_sha256"] != row["trace_sha256"]:
        raise AssertionError(f"{model_key}: manifest trace hash mismatch")

    tpots = [
        float(item["tpot_ms"])
        for item in requests
        if item.get("tpot_ms") is not None
    ]
    ttft_p95 = percentile(
        [float(item["ttft_ms"]) for item in requests],
        0.95,
    )
    tpot_p95 = percentile(tpots, 0.95)
    ttft_attainment = sum(
        float(item["ttft_ms"]) <= ttft_slo for item in requests
    ) / len(requests)
    tpot_attainment = sum(value <= tpot_slo for value in tpots) / len(tpots)
    joint_attainment = sum(
        float(item["ttft_ms"]) <= ttft_slo
        and item.get("tpot_ms") is not None
        and float(item["tpot_ms"]) <= tpot_slo
        for item in requests
    ) / len(requests)
    expected_gate = ttft_p95 <= ttft_slo and tpot_p95 <= tpot_slo

    if int(row["total_requests"]) != 4000:
        raise AssertionError(f"{model_key}: table total mismatch")
    if int(row["ok_requests"]) != 4000 or int(row["failed_requests"]) != 0:
        raise AssertionError(f"{model_key}: table completion mismatch")
    close(float(row["ttft_target_ms"]), ttft_slo)
    close(float(row["tpot_target_ms"]), tpot_slo)
    close(float(row["ttft_avg_ms"]), float(exact["avg_ttft_ms"]))
    close(float(row["ttft_p95_ms"]), ttft_p95)
    close(float(row["tpot_avg_ms"]), sum(tpots) / len(tpots))
    close(float(row["tpot_p95_ms"]), tpot_p95)
    close(float(row["ttft_slo_attainment"]), ttft_attainment)
    close(float(row["tpot_slo_attainment"]), tpot_attainment)
    close(float(row["joint_slo_attainment"]), joint_attainment)
    if (row["paper_slo_gate_pass"].lower() == "true") != expected_gate:
        raise AssertionError(f"{model_key}: SLO gate status mismatch")
    close(float(row["e2e_avg_ms"]), float(exact["avg_e2e_ms"]))
    close(float(row["e2e_p95_ms"]), float(exact["p95_e2e_ms"]))
    close(float(row["ce"]), float(exact["cost_effectiveness_e2e"]))
    for field in (
        "monetary_cost_per_request_usd",
        "monetary_cost_total_usd",
        "infra_gpu_seconds_total",
        "infra_active_gpu_seconds",
        "infra_idle_ready_gpu_seconds",
    ):
        close(float(row[field]), float(exact[field]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    args = parser.parse_args()
    with args.table.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["model_key"]: row for row in rows}
    if set(indexed) != set(EXPECTED_MODELS) or len(rows) != len(EXPECTED_MODELS):
        raise AssertionError(
            f"expected one 3b and one 7b row, got {[row['model_key'] for row in rows]}"
        )
    for model_key in ("3b", "7b"):
        verify_row(indexed[model_key])
    print("PASS system=ServerlessLLM models=3b,7b formal_requests=4000")


if __name__ == "__main__":
    main()
