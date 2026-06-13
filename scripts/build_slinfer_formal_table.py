#!/usr/bin/env python3
"""Build the audited SLINFER 3B/7B formal4000 comparison table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import yaml


MODEL_SPECS = {
    "3b": ("Llama-3.2 3B", 180.0, 14.0),
    "7b": ("Llama-2 7B", 440.0, 32.0),
}

FIELDNAMES = [
    "system",
    "system_key",
    "model",
    "model_key",
    "formal_requests",
    "trace_role",
    "replay_rate",
    "slo_profile",
    "gpu_budget",
    "system_mode",
    "keep_alive_s",
    "node_memory_gb",
    "monitor_tail_s",
    "total_requests",
    "ok_requests",
    "failed_requests",
    "ttft_target_ms",
    "tpot_target_ms",
    "ttft_avg_ms",
    "ttft_p95_ms",
    "tpot_avg_ms",
    "tpot_p95_ms",
    "ttft_slo_attainment",
    "tpot_slo_attainment",
    "joint_slo_attainment",
    "paper_slo_gate_pass",
    "e2e_avg_ms",
    "e2e_p95_ms",
    "ce",
    "monetary_cost_per_request_usd",
    "monetary_cost_total_usd",
    "allocated_gpu_seconds",
    "active_gpu_seconds",
    "idle_gpu_seconds",
    "startup_gpu_seconds",
    "predeploy_startup_sec",
    "source_model_realpath",
    "source_config_sha256",
    "converted_config_sha256",
    "trace_path",
    "trace_sha256",
    "raw_records_path",
    "raw_records_sha256",
    "source_summary_path",
    "source_summary_sha256",
    "manifest_path",
    "manifest_sha256",
    "slinfer_git_commit",
    "slinfer_worktree_diff_sha256",
    "baseline_harness_git_commit",
    "relayserve_git_commit",
]


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


def parse_run(value: str) -> tuple[str, Path]:
    model_key, separator, run_dir_text = value.partition("=")
    if not separator or model_key not in MODEL_SPECS:
        raise argparse.ArgumentTypeError("run must be 3b=RUN_DIR or 7b=RUN_DIR")
    return model_key, Path(run_dir_text).resolve()


def scenario_from(summary: dict) -> dict:
    summaries = summary["scenario_summaries"]
    return summaries[0] if isinstance(summaries, list) else next(iter(summaries.values()))


def build_row(model_key: str, run_dir: Path, policy: dict) -> dict[str, object]:
    model, ttft_slo, tpot_slo = MODEL_SPECS[model_key]
    contract = policy["comparison_contract"]
    selected_keep_alive = policy["models"][model_key]["selected_keep_alive_s"]
    if selected_keep_alive is None:
        raise ValueError(f"{model_key}: calibration policy is not frozen")
    raw_path = run_dir / "raw_records.json"
    summary_path = run_dir / "source_summary.json"
    manifest_path = run_dir / "manifest.json"
    raw = json.loads(raw_path.read_text())
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    scenario = scenario_from(summary)
    requests = [item for item in raw["results"] if item.get("success")]
    ttfts = [float(item["ttft_ms"]) for item in requests]
    tpots = [float(item["tpot_ms"]) for item in requests]
    ttft_p95 = percentile(ttfts, 0.95)
    tpot_p95 = percentile(tpots, 0.95)
    ttft_attainment = sum(value <= ttft_slo for value in ttfts) / len(ttfts)
    tpot_attainment = sum(value <= tpot_slo for value in tpots) / len(tpots)
    joint_attainment = sum(
        float(item["ttft_ms"]) <= ttft_slo
        and float(item["tpot_ms"]) <= tpot_slo
        for item in requests
    ) / len(requests)
    if manifest["trace_role"] != "formal4000" or int(manifest["max_requests"]) != 4000:
        raise ValueError(f"{run_dir}: run is not formal4000")
    if float(manifest["keep_alive_s"]) != float(selected_keep_alive):
        raise ValueError(
            f"{run_dir}: keep_alive={manifest['keep_alive_s']}, "
            f"frozen={selected_keep_alive}"
        )
    if len(requests) != 4000 or int(scenario["completed_requests"]) != 4000:
        raise ValueError(f"{run_dir}: expected clean 4000/4000 formal evidence")
    adaptation_path = Path(manifest["frozen_config_dir"]) / "hardware_adaptation.json"
    adaptation = json.loads(adaptation_path.read_text())
    identity = adaptation["model_identity"]
    if identity["config_identity_verified"] is not True:
        raise ValueError(f"{run_dir}: model identity is not verified")
    return {
        "system": "SLINFER",
        "system_key": "slinfer",
        "model": model,
        "model_key": model_key,
        "formal_requests": 4000,
        "trace_role": manifest["trace_role"],
        "replay_rate": 1.0,
        "slo_profile": "paper_nominal",
        "gpu_budget": int(contract["gpu_budget"]),
        "system_mode": contract["system_mode"],
        "keep_alive_s": float(manifest["keep_alive_s"]),
        "node_memory_gb": float(manifest["node_memory_gb"]),
        "monitor_tail_s": float(manifest["monitor_tail_s"]),
        "total_requests": int(scenario["total_requests"]),
        "ok_requests": int(scenario["completed_requests"]),
        "failed_requests": int(scenario["failed_requests"]),
        "ttft_target_ms": ttft_slo,
        "tpot_target_ms": tpot_slo,
        "ttft_avg_ms": float(scenario["avg_ttft_ms"]),
        "ttft_p95_ms": ttft_p95,
        "tpot_avg_ms": float(scenario["avg_tpot_ms"]),
        "tpot_p95_ms": tpot_p95,
        "ttft_slo_attainment": ttft_attainment,
        "tpot_slo_attainment": tpot_attainment,
        "joint_slo_attainment": joint_attainment,
        "paper_slo_gate_pass": str(
            ttft_p95 <= ttft_slo and tpot_p95 <= tpot_slo
        ).lower(),
        "e2e_avg_ms": float(scenario["avg_e2e_ms"]),
        "e2e_p95_ms": float(scenario["p95_e2e_ms"]),
        "ce": float(scenario["ce"]),
        "monetary_cost_per_request_usd": float(
            scenario["monetary_cost_per_request_usd"]
        ),
        "monetary_cost_total_usd": float(scenario["monetary_cost_total_usd"]),
        "allocated_gpu_seconds": float(scenario["allocated_gpu_seconds"]),
        "active_gpu_seconds": float(scenario["active_gpu_seconds"]),
        "idle_gpu_seconds": float(scenario["idle_gpu_seconds"]),
        "startup_gpu_seconds": float(scenario["startup_gpu_seconds"]),
        "predeploy_startup_sec": float(manifest["initial_runtime_startup_sec"]),
        "source_model_realpath": identity["source_model_realpath"],
        "source_config_sha256": identity["source_config_sha256"],
        "converted_config_sha256": identity["converted_config_sha256"],
        "trace_path": manifest["trace_path"],
        "trace_sha256": manifest["trace_sha256"],
        "raw_records_path": str(raw_path),
        "raw_records_sha256": sha256(raw_path),
        "source_summary_path": str(summary_path),
        "source_summary_sha256": sha256(summary_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "slinfer_git_commit": manifest["slinfer_git_commit"],
        "slinfer_worktree_diff_sha256": manifest["slinfer_worktree_diff_sha256"],
        "baseline_harness_git_commit": manifest["baseline_harness_git_commit"],
        "relayserve_git_commit": manifest["relayserve_git_commit"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/relayserve_slinfer_calibration_policy_v1.yaml"),
    )
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    indexed = dict(args.run)
    if set(indexed) != set(MODEL_SPECS) or len(args.run) != len(MODEL_SPECS):
        raise SystemExit("exactly one 3b run and one 7b run are required")
    policy = yaml.safe_load(args.policy.read_text()) or {}
    if policy.get("schema") != "relayserve_slinfer_calibration_policy_v1":
        raise SystemExit("unexpected SLINFER calibration policy schema")
    rows = [
        build_row(model_key, indexed[model_key], policy)
        for model_key in ("3b", "7b")
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
