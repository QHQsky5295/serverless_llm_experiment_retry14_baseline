#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
        raise AssertionError(f"value mismatch: actual={actual}, expected={expected}")


def verify_row(row: dict[str, str]) -> tuple[float, float, float, int]:
    for prefix in ("raw_records", "source_summary", "manifest"):
        path = Path(row[f"{prefix}_path"])
        if sha256(path) != row[f"{prefix}_sha256"]:
            raise AssertionError(f"{prefix} hash mismatch: {path}")

    raw = json.loads(Path(row["raw_records_path"]).read_text())
    summary = json.loads(Path(row["source_summary_path"]).read_text())
    manifest = json.loads(Path(row["manifest_path"]).read_text())
    scenario_summaries = summary["scenario_summaries"]
    scenario = (
        scenario_summaries[0]
        if isinstance(scenario_summaries, list)
        else next(iter(scenario_summaries.values()))
    )
    requests = [item for item in raw["results"] if item.get("success")]
    ttfts = [float(item["ttft_ms"]) for item in requests]
    tpots = [float(item["tpot_ms"]) for item in requests]
    ttft_slo = float(row["ttft_slo_ms"])
    tpot_slo = float(row["tpot_slo_ms"])
    ttft_p95 = percentile(ttfts, 0.95)
    tpot_p95 = percentile(tpots, 0.95)
    ttft_attainment = sum(value <= ttft_slo for value in ttfts) / len(ttfts)
    tpot_attainment = sum(value <= tpot_slo for value in tpots) / len(tpots)
    joint_attainment = sum(
        ttft <= ttft_slo and tpot <= tpot_slo
        for ttft, tpot in zip(ttfts, tpots, strict=True)
    ) / len(requests)

    close(float(row["ttft_p95_ms"]), ttft_p95)
    close(float(row["tpot_p95_ms"]), tpot_p95)
    close(float(row["ttft_slo_attainment"]), ttft_attainment)
    close(float(row["tpot_slo_attainment"]), tpot_attainment)
    close(float(row["joint_slo_attainment"]), joint_attainment)
    close(float(row["e2e_avg_ms"]), float(scenario["avg_e2e_ms"]))
    close(float(row["ce"]), float(scenario["ce"]))
    close(
        float(row["monetary_cost_per_request_usd"]),
        float(scenario["monetary_cost_per_request_usd"]),
    )
    close(float(row["node_memory_gb"]), float(manifest["node_memory_gb"]))
    close(float(row["monitor_tail_s"]), float(manifest["monitor_tail_s"]))
    worst_ratio = max(ttft_p95 / ttft_slo, tpot_p95 / tpot_slo)
    close(float(row["worst_normalized_p95_violation"]), worst_ratio)
    return (
        worst_ratio,
        -joint_attainment,
        -float(row["ce"]),
        int(row["keep_alive_s"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    args = parser.parse_args()
    with args.table.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ranking = [(verify_row(row), row) for row in rows]
    selected = [row for _, row in ranking if row["selected"] == "true"]
    if len(selected) != 1:
        raise AssertionError(f"expected one selected row, found {len(selected)}")
    winner = min(ranking, key=lambda item: item[0])[1]
    if winner["keep_alive_s"] != selected[0]["keep_alive_s"]:
        raise AssertionError(
            "selected keep-alive does not match the frozen policy: "
            f"selected={selected[0]['keep_alive_s']} "
            f"expected={winner['keep_alive_s']}"
        )
    print(
        "PASS "
        f"model={selected[0]['model_key']} "
        f"selected_keep_alive_s={selected[0]['keep_alive_s']} "
        f"candidates={len(rows)}"
    )


if __name__ == "__main__":
    main()
