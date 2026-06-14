#!/usr/bin/env python3
"""Build the audited SLINFER formal eligibility table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
    "diagnostic_ce_success_subset",
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


def scenario_from(summary: dict) -> dict:
    summaries = summary["scenario_summaries"]
    return summaries[0] if isinstance(summaries, list) else next(iter(summaries.values()))


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
    paper_slo_pass = (
        not failed
        and float(scenario["p95_ttft_ms"]) <= ttft_target
        and float(scenario["p95_tpot_ms"]) <= tpot_target
    )
    return {
        "system": "SLINFER",
        "model": model,
        "model_key": model_key,
        "calibration_status": policy["models"][model_key]["calibration_status"],
        "formal_status": policy["models"][model_key]["formal_status"],
        "formal_attempted": "true",
        "total_requests": len(results),
        "ok_requests": len(ok),
        "failed_requests": len(failed),
        "failure_reasons": json.dumps(
            manifest.get("failure_reasons") or {}, sort_keys=True
        ),
        "ttft_target_ms": ttft_target,
        "tpot_target_ms": tpot_target,
        "ttft_p95_ms": scenario["p95_ttft_ms"],
        "tpot_p95_ms": scenario["p95_tpot_ms"],
        "diagnostic_ce_success_subset": scenario["ce"],
        "strict_zero_failure_pass": str(
            bool(manifest["strict_zero_failure_pass"])
        ).lower(),
        "paper_slo_gate_pass": str(paper_slo_pass).lower(),
        "external_main_comparison_eligible": str(
            bool(manifest["formal_main_comparison_eligible"]) and paper_slo_pass
        ).lower(),
        "raw_records_path": str(raw_path.resolve()),
        "raw_records_sha256": sha256(raw_path),
        "source_summary_path": str(summary_path.resolve()),
        "source_summary_sha256": sha256(summary_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "notes": (
            "Formal trace completed, but native deadline violations failed the "
            "strict zero-failure contract. CE is diagnostic over successful "
            "requests only and is excluded from the main comparison."
        ),
    }


def blocked_row(model_key: str, policy: dict) -> dict[str, object]:
    model, ttft_target, tpot_target = MODELS[model_key]
    model_policy = policy["models"][model_key]
    return {
        "system": "SLINFER",
        "model": model,
        "model_key": model_key,
        "calibration_status": model_policy["calibration_status"],
        "formal_status": model_policy["formal_status"],
        "formal_attempted": "false",
        "total_requests": "",
        "ok_requests": "",
        "failed_requests": "",
        "failure_reasons": "",
        "ttft_target_ms": ttft_target,
        "tpot_target_ms": tpot_target,
        "ttft_p95_ms": "",
        "tpot_p95_ms": "",
        "diagnostic_ce_success_subset": "",
        "strict_zero_failure_pass": "false",
        "paper_slo_gate_pass": "false",
        "external_main_comparison_eligible": "false",
        "raw_records_path": "",
        "raw_records_sha256": "",
        "source_summary_path": "",
        "source_summary_sha256": "",
        "manifest_path": "",
        "manifest_sha256": "",
        "notes": (
            "Formal4000 was not run because no 7B calibration candidate "
            "completed a clean 512/512 replay."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/relayserve_slinfer_calibration_policy_v1.yaml"),
    )
    parser.add_argument("--run-3b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = yaml.safe_load(args.policy.read_text(encoding="utf-8")) or {}
    rows = [
        attempted_row("3b", args.run_3b.resolve(), policy),
        blocked_row("7b", policy),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
