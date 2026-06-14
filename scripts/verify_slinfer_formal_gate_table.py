#!/usr/bin/env python3
"""Verify the SLINFER formal eligibility table against raw evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
        raise AssertionError(f"value mismatch: {actual} != {expected}")


def scenario_from(summary: dict) -> dict:
    summaries = summary["scenario_summaries"]
    return summaries[0] if isinstance(summaries, list) else next(iter(summaries.values()))


def verify_evidence(row: dict[str, str], prefix: str) -> Path:
    path = Path(row[f"{prefix}_path"])
    if not path.is_file() or sha256(path) != row[f"{prefix}_sha256"]:
        raise AssertionError(f"{prefix} evidence mismatch: {path}")
    return path


def verify_3b(row: dict[str, str]) -> None:
    raw_path = verify_evidence(row, "raw_records")
    summary_path = verify_evidence(row, "source_summary")
    manifest_path = verify_evidence(row, "manifest")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenario = scenario_from(summary)
    results = list(raw["results"])
    ok = [item for item in results if item.get("success")]
    failed = [item for item in results if not item.get("success")]
    if (len(results), len(ok), len(failed)) != (4000, 3755, 245):
        raise AssertionError("unexpected SLINFER 3B formal cardinality")
    if {item.get("failure_reason") for item in failed} != {"deadline_violation"}:
        raise AssertionError("unexpected SLINFER failure reason")
    if row["formal_attempted"] != "true":
        raise AssertionError("3B formal attempt missing")
    if row["strict_zero_failure_pass"] != "false":
        raise AssertionError("3B strict gate must fail")
    if row["external_main_comparison_eligible"] != "false":
        raise AssertionError("failed formal run cannot enter the main comparison")
    if manifest["summary_scope"] != "partial_success_diagnostic_only":
        raise AssertionError("failed formal summary scope mismatch")
    close(float(row["ttft_p95_ms"]), float(scenario["p95_ttft_ms"]))
    close(float(row["tpot_p95_ms"]), float(scenario["p95_tpot_ms"]))
    close(float(row["diagnostic_ce_success_subset"]), float(scenario["ce"]))


def verify_7b(row: dict[str, str]) -> None:
    if row["formal_attempted"] != "false":
        raise AssertionError("7B formal run must remain blocked")
    if row["formal_status"] != "blocked_by_calibration_gate":
        raise AssertionError("7B formal block reason mismatch")
    if row["raw_records_path"] or row["source_summary_path"]:
        raise AssertionError("blocked 7B row must not invent evidence paths")
    if row["external_main_comparison_eligible"] != "false":
        raise AssertionError("blocked 7B row cannot enter the main comparison")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    args = parser.parse_args()
    with args.table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["model_key"]: row for row in rows}
    if len(rows) != 2 or set(indexed) != {"3b", "7b"}:
        raise AssertionError("expected exactly one 3B and one 7B gate row")
    verify_3b(indexed["3b"])
    verify_7b(indexed["7b"])
    print("PASS system=SLINFER formal_gate 3b=failed 7b=blocked")


if __name__ == "__main__":
    main()
