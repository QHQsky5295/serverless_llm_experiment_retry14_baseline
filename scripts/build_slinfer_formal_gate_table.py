#!/usr/bin/env python3
"""Build the audited SLINFER formal eligibility table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import yaml


MODELS = {
    "3b": ("Llama-3.2 3B", 180.0, 14.0),
    "7b": ("Llama-2 7B", 440.0, 32.0),
}

FIELDS = [
    "system",
    "model",
    "model_key",
    "calibration_status",
    "formal_status",
    "formal_attempted",
    "total_requests",
    "ok_requests",
    "failed_requests",
    "failure_reasons",
    "ttft_target_ms",
    "tpot_target_ms",
    "ttft_p95_ms",
    "tpot_p95_ms",
    "ce",
    "strict_zero_failure_pass",
    "paper_slo_gate_pass",
    "external_main_comparison_eligible",
    "raw_records_path",
    "raw_records_sha256",
    "source_summary_path",
    "source_summary_sha256",
    "manifest_path",
    "manifest_sha256",
    "notes",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def scenario_from(summary: dict) -> dict:
    summaries = summary["scenario_summaries"]
    return summaries[0] if isinstance(summaries, list) else next(iter(summaries.values()))


def parse_run(value: str) -> tuple[str, Path]:
    model_key, separator, run_dir_text = value.partition("=")
    if not separator or model_key not in MODELS:
        raise argparse.ArgumentTypeError("run must be 3b=RUN_DIR or 7b=RUN_DIR")
    return model_key, Path(run_dir_text).resolve()


def comparison_eligible(manifest: dict, failed_requests: int) -> bool:
    return bool(manifest["formal_main_comparison_eligible"]) and failed_requests == 0


def attempted_row(model_key: str, run_dir: Path, policy: dict) -> dict[str, object]:
    model, ttft_target, tpot_target = MODELS[model_key]
    raw_path = run_dir / "raw_records.json"
    summary_path = run_dir / "source_summary.json"
    manifest_path = run_dir / "manifest.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenario = scenario_from(summary)
    results = list(raw.get("results") or [])
    ok = [item for item in results if bool(item.get("success"))]
    failed = [item for item in results if not bool(item.get("success"))]
    if manifest["trace_role"] != "formal4000" or int(manifest["max_requests"]) != 4000:
        raise ValueError(f"{run_dir}: run is not formal4000")
    ttft_p95 = percentile([float(item["ttft_ms"]) for item in ok], 0.95)
    tpot_p95 = percentile([float(item["tpot_ms"]) for item in ok], 0.95)
    strict_pass = bool(manifest["strict_zero_failure_pass"]) and not failed
    paper_slo_pass = (
        strict_pass
        and ttft_p95 <= ttft_target
        and tpot_p95 <= tpot_target
    )
    main_eligible = comparison_eligible(manifest, len(failed))
    if main_eligible != strict_pass:
        raise ValueError(f"{run_dir}: formal eligibility and strict status disagree")
    note = (
        "Clean formal4000 execution; eligible for the external main comparison. "
        + (
            "The external paper SLO gate also passed."
            if paper_slo_pass
            else "The external paper SLO gate failed and remains reported separately."
        )
        if strict_pass
        else (
            "Formal execution failed the strict zero-failure contract. CE is "
            "diagnostic over successful requests only and is excluded from the "
            "main comparison."
        )
    )
    return {
        "system": "SLINFER",
        "model": model,
        "model_key": model_key,
        "calibration_status": policy["models"][model_key]["calibration_status"],
        "formal_status": (
            "completed_clean_formal4000"
            if strict_pass
            else "failed_strict_validation"
        ),
        "formal_attempted": "true",
        "total_requests": len(results),
        "ok_requests": len(ok),
        "failed_requests": len(failed),
        "failure_reasons": json.dumps(
            manifest.get("failure_reasons") or {}, sort_keys=True
        ),
        "ttft_target_ms": ttft_target,
        "tpot_target_ms": tpot_target,
        "ttft_p95_ms": ttft_p95,
        "tpot_p95_ms": tpot_p95,
        "ce": scenario["ce"],
        "strict_zero_failure_pass": str(strict_pass).lower(),
        "paper_slo_gate_pass": str(paper_slo_pass).lower(),
        "external_main_comparison_eligible": str(main_eligible).lower(),
        "raw_records_path": str(raw_path.resolve()),
        "raw_records_sha256": sha256(raw_path),
        "source_summary_path": str(summary_path.resolve()),
        "source_summary_sha256": sha256(summary_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "notes": note,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/relayserve_slinfer_calibration_policy_v2.yaml"),
    )
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    indexed = dict(args.run)
    if set(indexed) != set(MODELS) or len(args.run) != len(MODELS):
        raise SystemExit("exactly one 3b run and one 7b run are required")
    policy = yaml.safe_load(args.policy.read_text(encoding="utf-8")) or {}
    if policy.get("schema") != "relayserve_slinfer_calibration_policy_v2":
        raise SystemExit("formal gate requires the corrected v2 policy")
    rows = [
        attempted_row(model_key, indexed[model_key], policy)
        for model_key in ("3b", "7b")
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
