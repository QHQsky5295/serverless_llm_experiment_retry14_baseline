#!/usr/bin/env python3
"""Verify a Llumnix formal table from immutable raw evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

MODELS = {
    "3b": ("Llama-3.2 3B", 180.0, 14.0),
    "7b": ("Llama-2 7B", 440.0, 32.0),
}
FORMAL_TRACE_ROLE = "formal4000"


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


def close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
        raise AssertionError(f"value mismatch: {actual} != {expected}")


def evidence(row: dict[str, str], prefix: str) -> Path:
    path = Path(row[f"{prefix}_path"])
    if not path.is_file() or sha256(path) != row[f"{prefix}_sha256"]:
        raise AssertionError(f"{prefix} evidence mismatch: {path}")
    return path


def verify_row(row: dict[str, str]) -> None:
    model_key = row["model_key"]
    model, ttft_target, tpot_target = MODELS[model_key]
    if row["system"] != "Llumnix" or row["system_key"] != "llumnix":
        raise AssertionError(f"{model_key}: unexpected system")
    if row["model"] != model:
        raise AssertionError(f"{model_key}: model label mismatch")
    if row["trace_role"] != FORMAL_TRACE_ROLE:
        raise AssertionError(f"{model_key}: trace role mismatch")
    manifest_trace_role = row.get("manifest_trace_role")
    if not manifest_trace_role or not manifest_trace_role.startswith(FORMAL_TRACE_ROLE):
        raise AssertionError(f"{model_key}: manifest trace role mismatch")
    if float(row["replay_rate"]) != 1.0:
        raise AssertionError(f"{model_key}: replay rate mismatch")
    if row["slo_profile"] != "paper_nominal":
        raise AssertionError(f"{model_key}: SLO profile mismatch")
    if int(row["formal_requests"]) != 4000 or int(row["gpu_budget"]) != 4:
        raise AssertionError(f"{model_key}: formal contract mismatch")
    if row.get("llumnix_variant") not in {
        "official_routine_migration",
        "official_no_routine_migration",
    }:
        raise AssertionError(f"{model_key}: Llumnix variant mismatch")
    routine_enabled = row.get("routine_migration_enabled")
    if routine_enabled not in {"true", "false"}:
        raise AssertionError(f"{model_key}: routine migration flag mismatch")

    raw_path = evidence(row, "raw_records")
    summary_path = evidence(row, "source_summary")
    manifest_path = evidence(row, "manifest")
    trace_path = evidence(row, "trace")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = list(raw["results"])
    ok = [record for record in results if record.get("success")]
    failed = [record for record in results if not record.get("success")]
    if len(results) != 4000 or len(ok) != 4000 or failed:
        raise AssertionError(f"{model_key}: formal execution is not clean")
    if (
        manifest["trace_role"] != manifest_trace_role
        or not str(manifest["trace_role"]).startswith(FORMAL_TRACE_ROLE)
        or int(manifest["max_requests"]) != 4000
    ):
        raise AssertionError(f"{model_key}: manifest formal contract mismatch")
    manifest_routine_enabled = bool(
        manifest.get("launch_profile", {}).get("enable_routine_migration")
    )
    if (routine_enabled == "true") != manifest_routine_enabled:
        raise AssertionError(f"{model_key}: routine migration evidence mismatch")
    expected_variant = (
        "official_routine_migration"
        if manifest_routine_enabled
        else "official_no_routine_migration"
    )
    if row["llumnix_variant"] != expected_variant:
        raise AssertionError(f"{model_key}: Llumnix variant evidence mismatch")
    if Path(manifest["trace_path"]).resolve() != trace_path.resolve():
        raise AssertionError(f"{model_key}: trace path mismatch")
    if manifest["trace_sha256"] != row["trace_sha256"]:
        raise AssertionError(f"{model_key}: trace hash mismatch")
    if not summary["formal_main_comparison_eligible"]:
        raise AssertionError(f"{model_key}: summary eligibility mismatch")

    ttfts = [float(record["ttft_ms"]) for record in ok]
    tpots = [float(record["tpot_ms"]) for record in ok]
    e2es = [float(record["e2e_ms"]) for record in ok]
    lifecycle = summary["lifecycle_cost"]
    paper_pass = (
        percentile(ttfts, 0.95) <= ttft_target
        and percentile(tpots, 0.95) <= tpot_target
    )
    close(float(row["ttft_target_ms"]), ttft_target)
    close(float(row["tpot_target_ms"]), tpot_target)
    close(float(row["ttft_avg_ms"]), sum(ttfts) / len(ttfts))
    close(float(row["ttft_p95_ms"]), percentile(ttfts, 0.95))
    close(float(row["tpot_avg_ms"]), sum(tpots) / len(tpots))
    close(float(row["tpot_p95_ms"]), percentile(tpots, 0.95))
    close(float(row["e2e_avg_ms"]), sum(e2es) / len(e2es))
    close(float(row["e2e_p95_ms"]), percentile(e2es, 0.95))
    close(float(row["ce"]), float(lifecycle["monetary_ce"]))
    close(
        float(row["monetary_cost_per_request_usd"]),
        float(lifecycle["monetary_cost_per_request_usd"]),
    )
    close(
        float(row["monetary_cost_total_usd"]),
        float(lifecycle["monetary_cost_total_usd"]),
    )
    close(
        float(row["infra_gpu_seconds_total"]),
        float(lifecycle["infra_gpu_seconds_total"]),
    )
    close(
        float(row["client_prewarm_sec"]),
        float(raw["client_prewarm_sec_excluded_from_workload_clock"]),
    )
    expected_gpu_seconds = (
        float(row["predeploy_startup_sec"])
        + float(row["client_prewarm_sec"])
        + float(raw["elapsed_sec"])
    ) * int(row["gpu_budget"])
    close(float(row["infra_gpu_seconds_total"]), expected_gpu_seconds)
    if (row["paper_slo_gate_pass"] == "true") != paper_pass:
        raise AssertionError(f"{model_key}: paper SLO gate mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    args = parser.parse_args()
    with args.table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["model_key"]: row for row in rows}
    if len(rows) != 2 or set(indexed) != set(MODELS):
        raise AssertionError("expected exactly one 3B and one 7B row")
    for model_key in ("3b", "7b"):
        verify_row(indexed[model_key])
    print("PASS system=Llumnix models=3b,7b formal_requests=4000")


if __name__ == "__main__":
    main()
