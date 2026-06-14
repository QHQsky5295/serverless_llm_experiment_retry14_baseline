#!/usr/bin/env python3
"""Build and verify the frozen SLINFER cross-rate calibration matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


EXPECTED_RATES = {0.67, 1.0, 1.3}
FIELDNAMES = [
    "model_key",
    "rate_scale",
    "trace_role",
    "workers_per_gpu",
    "keep_alive_s",
    "completed",
    "total",
    "failed",
    "success_rate",
    "ttft_slo_ms",
    "tpot_slo_ms",
    "ttft_p95_ms",
    "tpot_p95_ms",
    "ttft_p95_pass",
    "tpot_p95_pass",
    "joint_p95_pass",
    "avg_e2e_ms",
    "p95_e2e_ms",
    "ce",
    "monetary_cost_per_request_usd",
    "avg_allocated_gpus",
    "peak_allocated_gpus",
    "cold_start_requests",
    "memory_guard_min_available_gib",
    "scheduler_ttft_baseline_s",
    "scheduler_ttft_max_threshold_s",
    "scheduler_tpot_s",
    "raw_records_path",
    "raw_records_sha256",
    "source_summary_path",
    "source_summary_sha256",
    "manifest_path",
    "manifest_sha256",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile from an empty sample")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def parse_run(value: str) -> tuple[str, float, Path]:
    identity, separator, run_dir = value.partition("=")
    model_key, colon, rate_text = identity.partition(":")
    if not separator or not colon or model_key not in {"3b", "7b"}:
        raise argparse.ArgumentTypeError(
            "run must be MODEL_KEY:RATE_SCALE=RUN_DIR"
        )
    try:
        rate_scale = float(rate_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rate scale must be numeric") from exc
    return model_key, rate_scale, Path(run_dir).resolve()


def scenario_from(summary: dict[str, object]) -> dict[str, object]:
    scenarios = summary["scenario_summaries"]
    if isinstance(scenarios, list):
        return scenarios[0]
    return next(iter(scenarios.values()))


def build_row(model_key: str, rate_scale: float, run_dir: Path) -> dict[str, object]:
    raw_path = run_dir / "raw_records.json"
    summary_path = run_dir / "source_summary.json"
    manifest_path = run_dir / "manifest.json"
    raw = json.loads(raw_path.read_text())
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    adaptation_path = Path(manifest["frozen_config_dir"]) / "hardware_adaptation.json"
    adaptation = json.loads(adaptation_path.read_text())
    scenario = scenario_from(summary)
    records = raw["results"]
    successes = [record for record in records if record.get("success")]
    ttft_values = [float(record["ttft_ms"]) for record in successes]
    tpot_values = [float(record["tpot_ms"]) for record in successes]
    ttft_p95 = percentile(ttft_values, 0.95)
    tpot_p95 = percentile(tpot_values, 0.95)
    ttft_slo = float(scenario["ttft_slo_ms"])
    tpot_slo = float(scenario["tpot_slo_ms"])
    scheduler = manifest["scheduler_deadline_contract"]
    completed = len(successes)
    total = len(records)
    if manifest.get("execution_status") != "passed_clean":
        raise ValueError(f"run is not a clean calibration artifact: {run_dir}")
    return {
        "model_key": model_key,
        "rate_scale": rate_scale,
        "trace_role": manifest["trace_role"],
        "workers_per_gpu": int(adaptation["workers_per_gpu"]),
        "keep_alive_s": float(manifest["keep_alive_s"]),
        "completed": completed,
        "total": total,
        "failed": total - completed,
        "success_rate": completed / total,
        "ttft_slo_ms": ttft_slo,
        "tpot_slo_ms": tpot_slo,
        "ttft_p95_ms": ttft_p95,
        "tpot_p95_ms": tpot_p95,
        "ttft_p95_pass": ttft_p95 <= ttft_slo,
        "tpot_p95_pass": tpot_p95 <= tpot_slo,
        "joint_p95_pass": ttft_p95 <= ttft_slo and tpot_p95 <= tpot_slo,
        "avg_e2e_ms": float(scenario["avg_e2e_ms"]),
        "p95_e2e_ms": float(scenario["p95_e2e_ms"]),
        "ce": float(scenario["ce"]),
        "monetary_cost_per_request_usd": float(
            scenario["monetary_cost_per_request_usd"]
        ),
        "avg_allocated_gpus": float(scenario["avg_allocated_gpus"]),
        "peak_allocated_gpus": float(scenario["peak_allocated_gpus"]),
        "cold_start_requests": sum(
            record.get("cold_start") is True for record in successes
        ),
        "memory_guard_min_available_gib": (
            float(manifest["memory_guard_min_available_kb"]) / 1024 / 1024
        ),
        "scheduler_ttft_baseline_s": float(scheduler["ttft_baseline_s"]),
        "scheduler_ttft_max_threshold_s": float(
            scheduler["ttft_max_threshold_s"]
        ),
        "scheduler_tpot_s": float(scheduler["tpot_s"]),
        "raw_records_path": str(raw_path),
        "raw_records_sha256": sha256(raw_path),
        "source_summary_path": str(summary_path),
        "source_summary_sha256": sha256(summary_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
    }


def close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
        raise AssertionError(f"value mismatch: actual={actual}, expected={expected}")


def verify_matrix(path: Path, require_rate_grid: bool) -> None:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError("SLINFER rate matrix is empty")
    frozen_by_model: dict[str, set[tuple[str, ...]]] = {}
    rates_by_model: dict[str, set[float]] = {}
    for row in rows:
        for prefix in ("raw_records", "source_summary", "manifest"):
            evidence_path = Path(row[f"{prefix}_path"])
            if sha256(evidence_path) != row[f"{prefix}_sha256"]:
                raise AssertionError(f"{prefix} hash mismatch: {evidence_path}")
        raw = json.loads(Path(row["raw_records_path"]).read_text())
        summary = json.loads(Path(row["source_summary_path"]).read_text())
        manifest = json.loads(Path(row["manifest_path"]).read_text())
        scenario = scenario_from(summary)
        successes = [record for record in raw["results"] if record.get("success")]
        ttft_p95 = percentile(
            [float(record["ttft_ms"]) for record in successes], 0.95
        )
        tpot_p95 = percentile(
            [float(record["tpot_ms"]) for record in successes], 0.95
        )
        close(float(row["ttft_p95_ms"]), ttft_p95)
        close(float(row["tpot_p95_ms"]), tpot_p95)
        close(float(row["ce"]), float(scenario["ce"]))
        if int(row["completed"]) != len(successes):
            raise AssertionError("completed count does not match raw evidence")
        if int(row["total"]) != len(raw["results"]):
            raise AssertionError("total count does not match raw evidence")
        if row["trace_role"] != manifest["trace_role"]:
            raise AssertionError("trace role does not match manifest")
        model_key = row["model_key"]
        rates_by_model.setdefault(model_key, set()).add(float(row["rate_scale"]))
        frozen_by_model.setdefault(model_key, set()).add(
            (
                row["workers_per_gpu"],
                row["keep_alive_s"],
                row["scheduler_ttft_baseline_s"],
                row["scheduler_ttft_max_threshold_s"],
                row["scheduler_tpot_s"],
            )
        )
    for model_key, configurations in frozen_by_model.items():
        if len(configurations) != 1:
            raise AssertionError(
                f"{model_key} rate rows do not share one frozen configuration"
            )
        if require_rate_grid and rates_by_model[model_key] != EXPECTED_RATES:
            raise AssertionError(
                f"{model_key} rate grid mismatch: {rates_by_model[model_key]}"
            )
    print(f"PASS rows={len(rows)} models={len(frozen_by_model)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-rate-grid", action="store_true")
    args = parser.parse_args()
    rows = [
        build_row(model_key, rate_scale, run_dir)
        for model_key, rate_scale, run_dir in sorted(
            args.run, key=lambda item: (item[0], item[1])
        )
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDNAMES, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        str(value).lower()
                        if isinstance(value, bool)
                        else value
                    )
                    for key, value in row.items()
                }
            )
    verify_matrix(args.output, args.require_rate_grid)


if __name__ == "__main__":
    main()
