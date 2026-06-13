#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute percentile of an empty sample")
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


def verify_row(row: dict[str, str]) -> tuple[float, float, float, float]:
    raw_path = Path(row["raw_records_path"])
    summary_path = Path(row["source_summary_path"])
    if sha256(raw_path) != row["raw_records_sha256"]:
        raise AssertionError(f"raw hash mismatch: {raw_path}")
    if sha256(summary_path) != row["source_summary_sha256"]:
        raise AssertionError(f"summary hash mismatch: {summary_path}")

    raw = json.loads(raw_path.read_text())
    summary = json.loads(summary_path.read_text())
    result = summary["comparison_table"][0]
    exact = next(iter(summary["scenario_summaries"].values()))
    requests = [item for item in raw["results"] if item.get("success")]
    tpots = [
        float(item["tpot_ms"])
        for item in requests
        if item.get("tpot_ms") is not None
    ]
    ttft_slo = float(row["ttft_slo_ms"])
    tpot_slo = float(row["tpot_slo_ms"])
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

    if int(row["completed"]) != int(result["completed"]):
        raise AssertionError(f"completed mismatch: {summary_path}")
    if int(row["total"]) != int(result["total"]):
        raise AssertionError(f"total mismatch: {summary_path}")
    close(float(row["ttft_p95_ms"]), float(exact["p95_ttft_ms"]))
    close(float(row["tpot_p95_ms"]), tpot_p95)
    close(float(row["ttft_slo_attainment"]), ttft_attainment)
    close(float(row["tpot_slo_attainment"]), tpot_attainment)
    close(float(row["joint_slo_attainment"]), joint_attainment)
    close(float(row["e2e_avg_ms"]), float(exact["avg_e2e_ms"]))
    close(float(row["ce"]), float(exact["cost_effectiveness_e2e"]))
    close(
        float(row["monetary_cost_per_request_usd"]),
        float(exact["monetary_cost_per_request_usd"]),
    )

    worst_ratio = max(
        float(row["ttft_p95_ms"]) / ttft_slo,
        float(row["tpot_p95_ms"]) / tpot_slo,
    )
    close(float(row["worst_normalized_p95_violation"]), worst_ratio)
    return (
        worst_ratio,
        -ttft_attainment,
        -joint_attainment,
        -float(row["ce"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    args = parser.parse_args()

    with args.table.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("calibration table is empty")

    ranking = [(verify_row(row), row) for row in rows]
    selected = [row for _, row in ranking if row["selected"].lower() == "true"]
    if len(selected) != 1:
        raise AssertionError(f"expected one selected row, found {len(selected)}")
    winner = min(ranking, key=lambda item: item[0])[1]
    if winner["target_concurrency"] != selected[0]["target_concurrency"]:
        raise AssertionError(
            "selected target does not match the frozen calibration policy: "
            f"selected={selected[0]['target_concurrency']} "
            f"expected={winner['target_concurrency']}"
        )

    print(
        "PASS "
        f"model={selected[0]['model_key']} "
        f"selected_target={selected[0]['target_concurrency']} "
        f"candidates={len(rows)}"
    )


if __name__ == "__main__":
    main()
