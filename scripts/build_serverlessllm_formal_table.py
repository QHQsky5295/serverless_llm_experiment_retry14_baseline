#!/usr/bin/env python3
"""Build the audited ServerlessLLM 3B/7B formal4000 comparison table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import yaml


MODEL_SPECS = {
    "3b": {
        "model": "Llama-3.2 3B",
        "ttft_slo_ms": 180.0,
        "tpot_slo_ms": 14.0,
    },
    "7b": {
        "model": "Llama-2 7B",
        "ttft_slo_ms": 440.0,
        "tpot_slo_ms": 32.0,
    },
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
    "target_concurrency",
    "keep_alive_s",
    "min_instances",
    "max_instances",
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
    "infra_gpu_seconds_total",
    "infra_active_gpu_seconds",
    "infra_idle_ready_gpu_seconds",
    "predeploy_startup_sec",
    "trace_path",
    "trace_sha256",
    "raw_records_path",
    "raw_records_sha256",
    "source_summary_path",
    "source_summary_sha256",
    "manifest_path",
    "manifest_sha256",
    "serverlessllm_git_commit",
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


def load_policy(path: Path) -> dict:
    policy = yaml.safe_load(path.read_text()) or {}
    if policy.get("schema") != "relayserve_serverlessllm_calibration_policy_v1":
        raise ValueError(f"unexpected calibration policy schema: {path}")
    return policy


def build_row(model_key: str, run_dir: Path, policy: dict) -> dict[str, object]:
    spec = MODEL_SPECS[model_key]
    model_policy = policy["models"][model_key]
    contract = policy["comparison_contract"]
    raw_path = run_dir / "raw_records.json"
    summary_path = run_dir / "source_summary.json"
    manifest_path = run_dir / "manifest.json"
    raw = json.loads(raw_path.read_text())
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    result = summary["comparison_table"][0]
    exact = next(iter(summary["scenario_summaries"].values()))
    requests = [item for item in raw["results"] if item.get("success")]
    tpots = [
        float(item["tpot_ms"])
        for item in requests
        if item.get("tpot_ms") is not None
    ]
    ttft_slo = float(spec["ttft_slo_ms"])
    tpot_slo = float(spec["tpot_slo_ms"])
    ttft_p95 = float(exact["p95_ttft_ms"])
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
    expected_target = int(model_policy["selected_target_concurrency"])
    expected_keep_alive = int(contract["keep_alive_s"])
    if manifest["trace_role"] != "formal4000":
        raise ValueError(f"{run_dir}: trace role is not formal4000")
    if int(manifest["max_requests"]) != 4000:
        raise ValueError(f"{run_dir}: formal request count is not 4000")
    deploy = json.loads(Path(manifest["deploy_path"]).read_text())
    actual_target = int(deploy["auto_scaling_config"]["target"])
    keep_alive = int(deploy["auto_scaling_config"]["keep_alive"])
    if actual_target != expected_target:
        raise ValueError(
            f"{run_dir}: target={actual_target}, frozen target={expected_target}"
        )
    if keep_alive != expected_keep_alive:
        raise ValueError(
            f"{run_dir}: keep_alive={keep_alive}, frozen={expected_keep_alive}"
        )
    total = int(result["total"])
    completed = int(result["completed"])
    if total != 4000 or completed != 4000 or len(requests) != 4000:
        raise ValueError(
            f"{run_dir}: expected clean 4000/4000, got "
            f"summary={completed}/{total}, raw_success={len(requests)}"
        )
    return {
        "system": "ServerlessLLM",
        "system_key": "serverlessllm",
        "model": spec["model"],
        "model_key": model_key,
        "formal_requests": 4000,
        "trace_role": manifest["trace_role"],
        "replay_rate": 1.0,
        "slo_profile": "paper_nominal",
        "gpu_budget": int(contract["gpu_budget"]),
        "target_concurrency": actual_target,
        "keep_alive_s": keep_alive,
        "min_instances": int(contract["min_instances"]),
        "max_instances": int(contract["max_instances"]),
        "total_requests": total,
        "ok_requests": completed,
        "failed_requests": total - completed,
        "ttft_target_ms": ttft_slo,
        "tpot_target_ms": tpot_slo,
        "ttft_avg_ms": float(exact["avg_ttft_ms"]),
        "ttft_p95_ms": ttft_p95,
        "tpot_avg_ms": sum(tpots) / len(tpots),
        "tpot_p95_ms": tpot_p95,
        "ttft_slo_attainment": ttft_attainment,
        "tpot_slo_attainment": tpot_attainment,
        "joint_slo_attainment": joint_attainment,
        "paper_slo_gate_pass": str(
            ttft_p95 <= ttft_slo and tpot_p95 <= tpot_slo
        ).lower(),
        "e2e_avg_ms": float(exact["avg_e2e_ms"]),
        "e2e_p95_ms": float(exact["p95_e2e_ms"]),
        "ce": float(exact["cost_effectiveness_e2e"]),
        "monetary_cost_per_request_usd": float(
            exact["monetary_cost_per_request_usd"]
        ),
        "monetary_cost_total_usd": float(exact["monetary_cost_total_usd"]),
        "infra_gpu_seconds_total": float(exact["infra_gpu_seconds_total"]),
        "infra_active_gpu_seconds": float(exact["infra_active_gpu_seconds"]),
        "infra_idle_ready_gpu_seconds": float(
            exact["infra_idle_ready_gpu_seconds"]
        ),
        "predeploy_startup_sec": float(manifest["initial_runtime_startup_sec"]),
        "trace_path": manifest["trace_path"],
        "trace_sha256": manifest["trace_sha256"],
        "raw_records_path": str(raw_path),
        "raw_records_sha256": sha256(raw_path),
        "source_summary_path": str(summary_path),
        "source_summary_sha256": sha256(summary_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "serverlessllm_git_commit": manifest["serverlessllm_git_commit"],
        "baseline_harness_git_commit": manifest["baseline_harness_git_commit"],
        "relayserve_git_commit": manifest["relayserve_git_commit"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/relayserve_serverlessllm_calibration_policy_v1.yaml"),
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        help="repeat exactly once for 3b=RUN_DIR and 7b=RUN_DIR",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    indexed = dict(args.run)
    if set(indexed) != set(MODEL_SPECS) or len(args.run) != len(MODEL_SPECS):
        raise SystemExit("exactly one 3b run and one 7b run are required")
    policy = load_policy(args.policy)
    rows = [
        build_row(model_key, indexed[model_key], policy)
        for model_key in ("3b", "7b")
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
