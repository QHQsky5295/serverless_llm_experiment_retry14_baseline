#!/usr/bin/env python3
"""Build the audited Mooncake 3B/7B formal4000 comparison table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


MODELS = {
    "3b": ("Llama-3.2 3B", 180.0, 14.0),
    "7b": ("Llama-2 7B", 440.0, 32.0),
}

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
    "topology",
    "protocol",
    "total_requests",
    "ok_requests",
    "failed_requests",
    "observable_tpot_requests",
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
    "infra_gpu_seconds_total",
    "predeploy_startup_sec",
    "client_prewarm_sec",
    "trace_path",
    "trace_sha256",
    "raw_records_path",
    "raw_records_sha256",
    "source_summary_path",
    "source_summary_sha256",
    "manifest_path",
    "manifest_sha256",
    "mooncake_git_commit",
    "vllm_git_commit",
    "baseline_harness_git_commit",
    "relayserve_git_commit",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_run(value: str) -> tuple[str, Path]:
    model_key, separator, run_dir_text = value.partition("=")
    if not separator or model_key not in MODELS:
        raise argparse.ArgumentTypeError("run must be 3b=RUN_DIR or 7b=RUN_DIR")
    return model_key, Path(run_dir_text).resolve()


def scenario_from(summary: dict[str, Any]) -> dict[str, Any]:
    scenarios = summary["scenario_summaries"]
    return scenarios[0] if isinstance(scenarios, list) else next(iter(scenarios.values()))


def build_row(model_key: str, run_dir: Path) -> dict[str, object]:
    model, ttft_target, tpot_target = MODELS[model_key]
    raw_path = run_dir / "raw_records.json"
    summary_path = run_dir / "source_summary.json"
    manifest_path = run_dir / "manifest.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = list(raw.get("results") or [])
    ok = [record for record in results if record.get("success")]
    failed = [record for record in results if not record.get("success")]
    observable_tpot = [
        record
        for record in ok
        if record.get("tpot_ms") is not None and record.get("tpot_observed")
    ]
    scenario = scenario_from(summary)

    if manifest.get("system") != "Mooncake":
        raise ValueError(f"{run_dir}: unexpected system")
    if manifest.get("trace_role") != "formal4000":
        raise ValueError(f"{run_dir}: trace role is not formal4000")
    if int(manifest.get("max_requests") or 0) != 4000:
        raise ValueError(f"{run_dir}: formal request count is not 4000")
    if len(results) != 4000 or len(ok) != 4000 or failed:
        raise ValueError(
            f"{run_dir}: expected clean 4000/4000, "
            f"total={len(results)}, ok={len(ok)}, failed={len(failed)}"
        )
    if len(observable_tpot) != 4000:
        raise ValueError(f"{run_dir}: TPOT is not observable for all requests")
    if not bool(summary.get("formal_main_comparison_eligible")):
        raise ValueError(f"{run_dir}: source summary is not formal eligible")
    if not bool(manifest.get("strict_zero_failure_pass")):
        raise ValueError(f"{run_dir}: manifest strict gate did not pass")
    launch = manifest["launch_profile"]
    if launch["topology"] != "2P+2D" or launch["protocol"] != "tcp":
        raise ValueError(f"{run_dir}: unexpected Mooncake launch profile")
    if float(scenario["ttft_target_ms"]) != ttft_target:
        raise ValueError(f"{run_dir}: TTFT target mismatch")
    if float(scenario["tpot_target_ms"]) != tpot_target:
        raise ValueError(f"{run_dir}: TPOT target mismatch")

    lifecycle = summary["lifecycle_cost"]
    return {
        "system": "Mooncake",
        "system_key": "mooncake",
        "model": model,
        "model_key": model_key,
        "formal_requests": 4000,
        "trace_role": manifest["trace_role"],
        "replay_rate": 1.0,
        "slo_profile": "paper_nominal",
        "gpu_budget": 4,
        "topology": launch["topology"],
        "protocol": launch["protocol"],
        "total_requests": len(results),
        "ok_requests": len(ok),
        "failed_requests": len(failed),
        "observable_tpot_requests": len(observable_tpot),
        "ttft_target_ms": ttft_target,
        "tpot_target_ms": tpot_target,
        "ttft_avg_ms": scenario["ttft_ms"]["avg"],
        "ttft_p95_ms": scenario["ttft_ms"]["p95"],
        "tpot_avg_ms": scenario["tpot_ms"]["avg"],
        "tpot_p95_ms": scenario["tpot_ms"]["p95"],
        "paper_slo_gate_pass": str(
            bool(summary["paper_slo_gate_pass"])
        ).lower(),
        "e2e_avg_ms": scenario["e2e_ms"]["avg"],
        "e2e_p95_ms": scenario["e2e_ms"]["p95"],
        "ce": lifecycle["monetary_ce"],
        "monetary_cost_per_request_usd": lifecycle[
            "monetary_cost_per_request_usd"
        ],
        "monetary_cost_total_usd": lifecycle["monetary_cost_total_usd"],
        "infra_gpu_seconds_total": lifecycle["infra_gpu_seconds_total"],
        "predeploy_startup_sec": manifest["initial_runtime_startup_sec"],
        "client_prewarm_sec": summary[
            "client_prewarm_sec_excluded_from_workload_clock"
        ],
        "trace_path": manifest["trace_path"],
        "trace_sha256": manifest["trace_sha256"],
        "raw_records_path": str(raw_path),
        "raw_records_sha256": sha256(raw_path),
        "source_summary_path": str(summary_path),
        "source_summary_sha256": sha256(summary_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "mooncake_git_commit": manifest["mooncake_git_commit"],
        "vllm_git_commit": manifest["vllm_git_commit"],
        "baseline_harness_git_commit": manifest["baseline_harness_git_commit"],
        "relayserve_git_commit": manifest["relayserve_git_commit"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    indexed = dict(args.run)
    if set(indexed) != set(MODELS) or len(args.run) != len(MODELS):
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
