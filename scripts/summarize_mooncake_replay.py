#!/usr/bin/env python3
"""Summarize an official Mooncake replay under the RelayServe paper contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def stats(records: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    values = [
        float(record[field])
        for record in records
        if record.get(field) is not None
    ]
    return {
        "avg": sum(values) / len(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def failure_reason(record: dict[str, Any]) -> str:
    error = str(record.get("error") or "").strip()
    if not error:
        return "unknown"
    return error.split(":", 1)[0][:120]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-key", choices=("3b", "7b"), required=True)
    parser.add_argument("--gpu-budget", type=int, default=4)
    parser.add_argument("--startup-sec", type=float, default=0.0)
    parser.add_argument("--deployment-duration-sec", type=float, default=0.0)
    parser.add_argument("--gpu-cost-per-second-usd", type=float, default=0.0008)
    parser.add_argument("--tpot-slo-ms", type=float, required=True)
    args = parser.parse_args()

    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    results = list(replay.get("results") or [])
    ok = [record for record in results if record.get("success")]
    failed = [record for record in results if not record.get("success")]
    expected = int(replay.get("expected_requests") or len(results))
    elapsed_sec = float(replay.get("elapsed_sec") or 0.0)
    client_prewarm_sec = float(
        replay.get("client_prewarm_sec_excluded_from_workload_clock") or 0.0
    )
    deployment_duration_sec = (
        float(args.deployment_duration_sec)
        if args.deployment_duration_sec > 0
        else float(args.startup_sec) + client_prewarm_sec + elapsed_sec
    )
    gpu_seconds = max(0.0, deployment_duration_sec) * max(0, args.gpu_budget)
    monetary_total = gpu_seconds * max(0.0, args.gpu_cost_per_second_usd)
    denominator = len(ok) if ok else len(results)
    monetary_per_request = (
        monetary_total / denominator if denominator > 0 else 0.0
    )
    e2e = stats(ok, "e2e_ms")
    average_e2e_sec = float(e2e["avg"] or 0.0) / 1000.0
    ce_denominator = monetary_per_request * average_e2e_sec
    ce = 1.0 / ce_denominator if ce_denominator > 1e-12 else 0.0
    observable_tpot = [
        record
        for record in ok
        if record.get("tpot_ms") is not None and record.get("tpot_observed")
    ]
    strict_pass = (
        len(results) == expected
        and len(ok) == expected
        and not failed
        and len(observable_tpot) == expected
        and bool(replay.get("force_stream"))
    )
    ttft_target = float(replay["ttft_slo_ms"])
    tpot_target = float(args.tpot_slo_ms)
    ttft = stats(ok, "ttft_ms")
    tpot = stats(observable_tpot, "tpot_ms")
    service_ttft = stats(ok, "service_ttft_ms")
    service_e2e = stats(ok, "service_e2e_ms")
    dispatch_wait = stats(ok, "dispatch_admission_wait_ms")
    paper_slo_pass = (
        strict_pass
        and float(ttft["p95"] or math.inf) <= ttft_target
        and float(tpot["p95"] or math.inf) <= tpot_target
    )
    target_met = [
        record
        for record in observable_tpot
        if float(record.get("ttft_ms") or math.inf) <= ttft_target
        and float(record.get("tpot_ms") or math.inf) <= tpot_target
    ]
    prompt_tokens = sum(int(record.get("prompt_tokens") or 0) for record in ok)
    completion_tokens = sum(
        int(record.get("completion_tokens") or 0) for record in ok
    )
    scenario = {
        "scenario_name": "mooncake_relayserve_continuation",
        "system": "Mooncake",
        "model": args.model,
        "model_key": args.model_key,
        "total_requests": len(results),
        "ok_requests": len(ok),
        "failed_requests": len(failed),
        "observable_tpot_requests": len(observable_tpot),
        "strict_zero_failure_pass": strict_pass,
        "paper_slo_gate_pass": paper_slo_pass,
        "ttft_target_ms": ttft_target,
        "tpot_target_ms": tpot_target,
        "ttft_ms": ttft,
        "service_ttft_ms": service_ttft,
        "tpot_ms": tpot,
        "e2e_ms": e2e,
        "service_e2e_ms": service_e2e,
        "dispatch_wait_ms": dispatch_wait,
        "target_met_requests": len(target_met),
        "target_met_fraction": len(target_met) / len(results) if results else 0.0,
        "prompt_tokens_total": prompt_tokens,
        "completion_tokens_total": completion_tokens,
        "gpu_budget": args.gpu_budget,
        "deployment_duration_sec": deployment_duration_sec,
        "startup_sec": float(args.startup_sec),
        "client_prewarm_sec": client_prewarm_sec,
        "gpu_seconds_total": gpu_seconds,
        "monetary_cost_total_usd": monetary_total,
        "monetary_cost_per_request_usd": monetary_per_request,
        "ce": ce,
    }
    payload = {
        "schema": "relayserve_external_baseline_summary_v1",
        "system": "Mooncake",
        "system_key": "mooncake",
        "runtime_class": "static_disaggregated_serverful",
        "model": args.model,
        "model_key": args.model_key,
        "trace_role": "formal4000" if expected == 4000 else "calibration",
        "replay_rate": 1.0,
        "slo_profile": "paper_nominal",
        "total_requests": len(results),
        "ok_requests": len(ok),
        "failed_requests": len(failed),
        "observable_tpot_requests": len(observable_tpot),
        "strict_zero_failure_pass": strict_pass,
        "formal_main_comparison_eligible": strict_pass and expected == 4000,
        "paper_slo_gate_pass": paper_slo_pass,
        "failure_reasons": dict(Counter(failure_reason(record) for record in failed)),
        "measured_replay_duration_s": elapsed_sec,
        "client_prewarm_sec_excluded_from_workload_clock": client_prewarm_sec,
        "prompt_tokens": {"total": prompt_tokens},
        "completion_tokens": {"total": completion_tokens},
        "lifecycle_cost": {
            "runtime_class": "static_disaggregated_serverful",
            "gpu_cost_per_second_usd": args.gpu_cost_per_second_usd,
            "infra_gpu_seconds_total": gpu_seconds,
            "monetary_equivalent_gpu_seconds": gpu_seconds,
            "monetary_cost_total_usd": monetary_total,
            "monetary_cost_per_request_usd": monetary_per_request,
            "monetary_ce": ce,
        },
        "scenario_summaries": [scenario],
        "raw_replay_path": str(args.replay.resolve()),
        "raw_replay_sha256": sha256(args.replay),
        "trace_source": replay.get("trace_source"),
        "metric_schema_version": replay.get("metric_schema_version"),
        "metric_definitions": replay.get("metric_definitions"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "total": len(results),
                "ok": len(ok),
                "failed": len(failed),
                "observable_tpot": len(observable_tpot),
                "strict_zero_failure_pass": strict_pass,
                "paper_slo_gate_pass": paper_slo_pass,
                "ce": ce,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
