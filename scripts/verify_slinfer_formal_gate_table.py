#!/usr/bin/env python3
"""Verify the corrected SLINFER formal eligibility table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


TARGETS = {
    "3b": (180.0, 14.0),
    "7b": (440.0, 32.0),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise AssertionError("empty percentile sample")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
        raise AssertionError(f"value mismatch: {actual} != {expected}")


def verify_evidence(row: dict[str, str], prefix: str) -> Path:
    path = Path(row[f"{prefix}_path"])
    if not path.is_file() or sha256(path) != row[f"{prefix}_sha256"]:
        raise AssertionError(f"{prefix} evidence mismatch: {path}")
    return path


def verify_row(row: dict[str, str]) -> None:
    model_key = row["model_key"]
    ttft_target, tpot_target = TARGETS[model_key]
    raw_path = verify_evidence(row, "raw_records")
    summary_path = verify_evidence(row, "source_summary")
    manifest_path = verify_evidence(row, "manifest")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = list(raw["results"])
    ok = [item for item in results if item.get("success")]
    failed = [item for item in results if not item.get("success")]
    if row["formal_attempted"] != "true" or len(results) != 4000:
        raise AssertionError(f"{model_key}: formal cardinality mismatch")
    if manifest["trace_role"] != "formal4000" or int(manifest["max_requests"]) != 4000:
        raise AssertionError(f"{model_key}: manifest formal contract mismatch")
    strict_pass = len(ok) == 4000 and not failed
    ttft_p95 = percentile([float(item["ttft_ms"]) for item in ok], 0.95)
    tpot_p95 = percentile([float(item["tpot_ms"]) for item in ok], 0.95)
    paper_pass = strict_pass and ttft_p95 <= ttft_target and tpot_p95 <= tpot_target
    close(float(row["ttft_p95_ms"]), ttft_p95)
    close(float(row["tpot_p95_ms"]), tpot_p95)
    if (row["strict_zero_failure_pass"] == "true") != strict_pass:
        raise AssertionError(f"{model_key}: strict status mismatch")
    if (row["paper_slo_gate_pass"] == "true") != paper_pass:
        raise AssertionError(f"{model_key}: paper SLO status mismatch")
    if (row["external_main_comparison_eligible"] == "true") != strict_pass:
        raise AssertionError(f"{model_key}: main-comparison eligibility mismatch")
    summaries = summary["scenario_summaries"]
    scenario = summaries[0] if isinstance(summaries, list) else next(iter(summaries.values()))
    close(float(row["ce"]), float(scenario["ce"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    args = parser.parse_args()
    with args.table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["model_key"]: row for row in rows}
    if len(rows) != 2 or set(indexed) != set(TARGETS):
        raise AssertionError("expected exactly one 3B and one 7B gate row")
    for model_key in ("3b", "7b"):
        verify_row(indexed[model_key])
    print("PASS system=SLINFER formal_gate models=3b,7b")


if __name__ == "__main__":
    main()
