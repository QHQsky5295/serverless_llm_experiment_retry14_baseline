#!/usr/bin/env python3
"""Build the SLINFER external-main formal4000 source table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


FIELDS = [
    "system",
    "system_key",
    "model",
    "model_key",
    "formal_requests",
    "trace_role",
    "replay_rate",
    "slo_profile",
    "gpu_budget",
    "total_requests",
    "ok_requests",
    "failed_requests",
    "ttft_target_ms",
    "tpot_target_ms",
    "ttft_avg_ms",
    "ttft_p95_ms",
    "tpot_avg_ms",
    "tpot_p95_ms",
    "paper_slo_gate_pass",
    "e2e_avg_ms",
    "e2e_p95_ms",
    "ce",
    "monetary_cost_per_request_usd",
    "monetary_cost_total_usd",
    "raw_records_path",
    "raw_records_sha256",
    "source_summary_path",
    "source_summary_sha256",
    "manifest_path",
    "manifest_sha256",
]


TARGETS = {
    "3b": ("Llama-3.2 3B", 180.0, 14.0),
    "7b": ("Llama-2 7B", 440.0, 32.0),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_run(value: str) -> tuple[str, Path]:
    model_key, separator, run_dir_text = value.partition("=")
    if not separator or model_key not in TARGETS:
        raise argparse.ArgumentTypeError("run must be 3b=RUN_DIR or 7b=RUN_DIR")
    return model_key, Path(run_dir_text).resolve()


def scenario_from(summary: dict) -> dict:
    summaries = summary["scenario_summaries"]
    return summaries[0] if isinstance(summaries, list) else next(iter(summaries.values()))


def build_row(model_key: str, run_dir: Path) -> dict[str, object]:
    model, ttft_target, tpot_target = TARGETS[model_key]
    raw_path = run_dir / "raw_records.json"
    summary_path = run_dir / "source_summary.json"
    manifest_path = run_dir / "manifest.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenario = scenario_from(summary)
    if manifest.get("trace_role") != "formal4000" or int(manifest.get("max_requests", 0)) != 4000:
        raise ValueError(f"{run_dir}: run is not formal4000")
    ok_requests = int(manifest.get("ok_requests", 0))
    failed_requests = int(manifest.get("failed_requests", -1))
    total_requests = int(scenario["total_requests"])
    if total_requests != 4000 or ok_requests != 4000 or failed_requests != 0:
        raise ValueError(f"{run_dir}: run is not a clean 4000/4000 formal run")
    paper_slo_pass = (
        float(scenario["p95_ttft_ms"]) <= ttft_target
        and float(scenario["p95_tpot_ms"]) <= tpot_target
    )
    return {
        "system": "SLINFER",
        "system_key": "slinfer",
        "model": model,
        "model_key": model_key,
        "formal_requests": 4000,
        "trace_role": "formal4000",
        "replay_rate": 1.0,
        "slo_profile": "paper_nominal",
        "gpu_budget": int(scenario.get("peak_allocated_gpus") or 4),
        "total_requests": total_requests,
        "ok_requests": ok_requests,
        "failed_requests": failed_requests,
        "ttft_target_ms": ttft_target,
        "tpot_target_ms": tpot_target,
        "ttft_avg_ms": scenario["avg_ttft_ms"],
        "ttft_p95_ms": scenario["p95_ttft_ms"],
        "tpot_avg_ms": scenario["avg_tpot_ms"],
        "tpot_p95_ms": scenario["p95_tpot_ms"],
        "paper_slo_gate_pass": str(paper_slo_pass).lower(),
        "e2e_avg_ms": scenario["avg_e2e_ms"],
        "e2e_p95_ms": scenario["p95_e2e_ms"],
        "ce": scenario["ce"],
        "monetary_cost_per_request_usd": scenario["monetary_cost_per_request_usd"],
        "monetary_cost_total_usd": scenario["monetary_cost_total_usd"],
        "raw_records_path": str(raw_path),
        "raw_records_sha256": sha256(raw_path),
        "source_summary_path": str(summary_path),
        "source_summary_sha256": sha256(summary_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    indexed = dict(args.run)
    if set(indexed) != set(TARGETS) or len(args.run) != len(TARGETS):
        raise SystemExit("exactly one 3b run and one 7b run are required")
    rows = [build_row(model_key, indexed[model_key]) for model_key in ("3b", "7b")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
