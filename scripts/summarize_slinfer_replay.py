#!/usr/bin/env python3
"""Summarize SLINFER replay output using the RelayServe paper metric contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(1.0, q / 100.0)) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rounded(value: float | None, digits: int = 6) -> float | None:
    return round(float(value), digits) if value is not None else None


def _flatten_sum(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        return sum(_flatten_sum(item) for item in value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _gpu_lifecycle(logs: dict[str, Any]) -> dict[str, float]:
    usage = list(((logs.get("node_usage") or {}).get("gpu") or []))
    density = list(((logs.get("node_density") or {}).get("gpu") or []))
    batches = list(((logs.get("batch") or {}).get("gpu") or []))
    sample_count = max(len(usage), len(density), len(batches))
    startup = 0.0
    active = 0.0
    idle = 0.0
    off = 0.0
    allocated_total = 0.0
    peak_allocated = 0.0
    node_count = 0

    for index in range(sample_count):
        allocated = float(usage[index]) if index < len(usage) else 0.0
        allocated_total += allocated
        peak_allocated = max(peak_allocated, allocated)
        density_row = density[index] if index < len(density) else []
        batch_row = batches[index] if index < len(batches) else []
        if isinstance(density_row, list):
            node_count = max(node_count, len(density_row))
        if isinstance(batch_row, list):
            node_count = max(node_count, len(batch_row))
        nodes = max(
            len(density_row) if isinstance(density_row, list) else 0,
            len(batch_row) if isinstance(batch_row, list) else 0,
        )
        explicitly_allocated = 0
        for node_index in range(nodes):
            node_density = (
                _flatten_sum(density_row[node_index])
                if isinstance(density_row, list) and node_index < len(density_row)
                else 0.0
            )
            node_batch = (
                _flatten_sum(batch_row[node_index])
                if isinstance(batch_row, list) and node_index < len(batch_row)
                else 0.0
            )
            node_is_allocated = node_density > 0 or node_batch > 0
            if not node_is_allocated:
                off += 1.0
            elif node_batch > 0:
                explicitly_allocated += 1
                active += 1.0
            else:
                explicitly_allocated += 1
                idle += 1.0
        startup += max(0.0, allocated - explicitly_allocated)

    return {
        "monitor_samples": float(sample_count),
        "monitor_node_count": float(node_count),
        "startup_gpu_seconds": startup,
        "active_gpu_seconds": active,
        "idle_gpu_seconds": idle,
        "off_gpu_seconds": off,
        "allocated_gpu_seconds": allocated_total,
        "avg_allocated_gpus": allocated_total / sample_count if sample_count else 0.0,
        "peak_allocated_gpus": peak_allocated,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-key", choices=["3b", "7b"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario-name", default="slinfer_relayserve_continuation")
    args = parser.parse_args()

    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    profile_name = (
        "relayserve_3b_paper_nominal"
        if args.model_key == "3b"
        else "relayserve_7b_paper_nominal"
    )
    workload = dict(
        ((config.get("workload_profiles") or {}).get(profile_name) or {}).get(
            "workload"
        )
        or {}
    )
    cost_model = dict(config.get("cost_model") or {})
    ttft_slo_ms = float(workload["ttft_slo_ms"])
    tpot_slo_ms = float(workload["tpot_slo_ms"])
    gpu_price = float(cost_model["gpu_cost_per_second_usd"])
    idle_factor = float(cost_model["serverless_idle_gpu_cost_factor"])

    results = list(replay.get("results") or [])
    ok = [record for record in results if bool(record.get("success"))]
    failed = [record for record in results if not bool(record.get("success"))]
    ttft = [float(record["ttft_ms"]) for record in ok if record.get("ttft_ms") is not None]
    service_ttft = [
        float(record["service_ttft_ms"])
        for record in ok
        if record.get("service_ttft_ms") is not None
    ]
    tpot = [float(record["tpot_ms"]) for record in ok if record.get("tpot_ms") is not None]
    e2e = [float(record["e2e_ms"]) for record in ok if record.get("e2e_ms") is not None]
    service_e2e = [
        float(record["service_e2e_ms"])
        for record in ok
        if record.get("service_e2e_ms") is not None
    ]
    dispatch_wait = [
        float(record["dispatch_wait_ms"])
        for record in ok
        if record.get("dispatch_wait_ms") is not None
    ]
    elapsed_s = float(replay.get("elapsed_sec") or 0.0)
    prompt_tokens = sum(int(record.get("prompt_tokens") or 0) for record in ok)
    completion_tokens = sum(int(record.get("completion_tokens") or 0) for record in ok)
    lifecycle = _gpu_lifecycle(dict(replay.get("gateway_logs") or {}))
    full_price_gpu_seconds = (
        lifecycle["startup_gpu_seconds"] + lifecycle["active_gpu_seconds"]
    )
    discounted_idle_gpu_seconds = lifecycle["idle_gpu_seconds"] * idle_factor
    monetary_cost_total = (
        full_price_gpu_seconds + discounted_idle_gpu_seconds
    ) * gpu_price
    monetary_cost_per_request = (
        monetary_cost_total / len(ok) if ok else 0.0
    )
    avg_e2e_s = (_avg(e2e) or 0.0) / 1000.0
    ce = (
        1.0 / (monetary_cost_per_request * avg_e2e_s)
        if monetary_cost_per_request > 0 and avg_e2e_s > 0
        else 0.0
    )
    slo_good = [
        record
        for record in ok
        if float(record.get("ttft_ms") or math.inf) <= ttft_slo_ms
        and float(record.get("tpot_ms") or math.inf) <= tpot_slo_ms
    ]
    throughput_rps = len(ok) / elapsed_s if elapsed_s > 0 else 0.0
    throughput_tokps = completion_tokens / elapsed_s if elapsed_s > 0 else 0.0

    scenario = {
        "scenario_name": args.scenario_name,
        "system": "SLINFER",
        "baseline_type": "external_serverless_llm",
        "backend": "slinfer_official_modified_vllm",
        "model_key": args.model_key,
        "system_config": "sota_gpu_only",
        "total_requests": len(results),
        "completed_requests": len(ok),
        "failed_requests": len(failed),
        "elapsed_sec": _rounded(elapsed_s),
        "avg_ttft_ms": _rounded(_avg(ttft)),
        "p95_ttft_ms": _rounded(_pct(ttft, 95)),
        "p99_ttft_ms": _rounded(_pct(ttft, 99)),
        "avg_service_ttft_ms": _rounded(_avg(service_ttft)),
        "p95_service_ttft_ms": _rounded(_pct(service_ttft, 95)),
        "avg_tpot_ms": _rounded(_avg(tpot)),
        "p95_tpot_ms": _rounded(_pct(tpot, 95)),
        "avg_e2e_ms": _rounded(_avg(e2e)),
        "p95_e2e_ms": _rounded(_pct(e2e, 95)),
        "p99_e2e_ms": _rounded(_pct(e2e, 99)),
        "avg_service_e2e_ms": _rounded(_avg(service_e2e)),
        "avg_dispatch_wait_ms": _rounded(_avg(dispatch_wait)),
        "p95_dispatch_wait_ms": _rounded(_pct(dispatch_wait, 95)),
        "throughput_rps": _rounded(throughput_rps),
        "throughput_tok_per_s": _rounded(throughput_tokps),
        "total_prompt_tokens": prompt_tokens,
        "total_completion_tokens": completion_tokens,
        "ttft_slo_ms": ttft_slo_ms,
        "tpot_slo_ms": tpot_slo_ms,
        "slo_attainment": _rounded(len(slo_good) / len(ok) if ok else 0.0),
        "slo_goodput_rps": _rounded(len(slo_good) / elapsed_s if elapsed_s > 0 else 0.0),
        "startup_gpu_seconds": lifecycle["startup_gpu_seconds"],
        "active_gpu_seconds": lifecycle["active_gpu_seconds"],
        "idle_gpu_seconds": lifecycle["idle_gpu_seconds"],
        "allocated_gpu_seconds": lifecycle["allocated_gpu_seconds"],
        "avg_allocated_gpus": _rounded(lifecycle["avg_allocated_gpus"]),
        "peak_allocated_gpus": lifecycle["peak_allocated_gpus"],
        "gpu_cost_per_second_usd": gpu_price,
        "serverless_idle_gpu_cost_factor": idle_factor,
        "monetary_cost_total_usd": _rounded(monetary_cost_total, 9),
        "monetary_cost_per_request_usd": _rounded(
            monetary_cost_per_request,
            12,
        ),
        "ce": _rounded(ce),
        "raw_records_path": str(args.replay.resolve()),
        "prompt_token_source": "local_guarded_prompt_token_ids",
        "completion_token_source": "slinfer_completed_length",
    }
    summary = {
        "schema_version": 4,
        "metric_schema_version": "e2e_v3",
        "metric_definitions": replay.get("metric_definitions") or {},
        "metadata": {
            "system": "SLINFER",
            "model_key": args.model_key,
            "trace_source": replay.get("trace_source"),
            "config_path": str(args.config.resolve()),
            "cost_model": (
                "startup/active GPU-seconds at full price plus ready-idle "
                "GPU-seconds at the frozen serverless idle factor"
            ),
        },
        "comparison_table": [scenario],
        "scenario_summaries": [scenario],
        "detailed_results": {
            "gateway_config": replay.get("gateway_config"),
            "gpu_lifecycle": lifecycle,
            "failures": failed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[slinfer-summary] completed={len(ok)}/{len(results)} "
        f"ttft_p95={scenario['p95_ttft_ms']}ms "
        f"tpot_avg={scenario['avg_tpot_ms']}ms CE={scenario['ce']}"
    )
    return 0 if len(ok) == len(results) and results else 1


if __name__ == "__main__":
    raise SystemExit(main())
