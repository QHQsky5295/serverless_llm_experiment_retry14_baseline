#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_profiles(
    cfg: Dict[str, Any],
    model_profile: str,
    dataset_profile: str,
    workload_profile: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    model_cfg = dict(cfg.get("model", {}) or {})
    adapters_cfg = dict(cfg.get("lora_adapters", {}) or {})
    datasets_cfg = dict(cfg.get("datasets", {}) or {})
    workload_cfg = dict(cfg.get("workload", {}) or {})
    coord_cfg = dict(cfg.get("resource_coordination", {}) or {})
    for bucket_name, selected in [
        ("model_profiles", model_profile),
        ("dataset_profiles", dataset_profile),
        ("workload_profiles", workload_profile),
    ]:
        profile = dict((cfg.get(bucket_name, {}) or {}).get(selected) or {})
        if not profile:
            raise KeyError(f"unknown profile '{selected}' in {bucket_name}")
        model_cfg = _deep_merge(model_cfg, profile.get("model", {}) or {})
        adapters_cfg = _deep_merge(adapters_cfg, profile.get("lora_adapters", {}) or {})
        datasets_cfg = _deep_merge(datasets_cfg, profile.get("datasets", {}) or {})
        workload_cfg = _deep_merge(workload_cfg, profile.get("workload", {}) or {})
        coord_cfg = _deep_merge(coord_cfg, profile.get("resource_coordination", {}) or {})
    return model_cfg, adapters_cfg, datasets_cfg, workload_cfg, coord_cfg


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(1.0, float(q) / 100.0)) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _build_metric_structure() -> Dict[str, List[str]]:
    return {
        "standard_serving_metrics": [
            "TTFT_avg_ms",
            "TTFT_P95_ms",
            "TTFT_P99_ms",
            "TTFT_warm_standard_avg_ms",
            "TTFT_warm_standard_P95_ms",
            "TTFT_warm_standard_P99_ms",
            "TPOT_avg_ms",
            "Throughput_RPS",
            "Throughput_TOKPS",
            "Cost_effectiveness_e2e",
        ],
        "serverless_deployment_metrics": [
            "TTFT_overall_avg_ms",
            "TTFT_comparable_avg_ms",
            "TTFT_comparable_P95_ms",
            "TTFT_comparable_P99_ms",
            "TTFT_gpu_ready_avg_ms",
            "TTFT_gpu_ready_P95_ms",
            "TTFT_scaleup_affected_avg_ms",
            "TTFT_scaleup_affected_P95_ms",
            "Runtime_TTFT_avg_ms",
            "Runtime_TTFT_P95_ms",
            "Cold_start_avg_ms",
            "Cold_start_P95_ms",
            "TTFT_serverless_overhead_avg_ms",
            "TTFT_serverless_overhead_P95_ms",
            "E2E_avg_ms",
            "E2E_P95_ms",
            "E2E_P99_ms",
            "Monetary_cost_avg_usd",
            "Monetary_cost_total_usd",
        ],
        "scaling_metrics": [
            "TTFT_SLO_ms",
            "SLO_attainment",
            "SLO_goodput_RPS",
            "SLO_goodput_TOKPS",
            "scale_up_events",
            "scale_down_events",
        ],
        "mechanism_metrics": [
            "CE",
            "QPR_TOKPS_TTFT_legacy",
            "QPR_RPS_legacy",
            "cache_hit_rate",
            "GPU_hit_rate",
            "LoRA_IO_avg_ms",
            "GPU_ready_hits",
            "Warm_pool_hits",
            "C3_contention_events",
            "C3_contention_penalty_ms",
            "C3_defer_delay_ms",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize a Punica replay into the common paper metric schema.")
    ap.add_argument("--main-repo", type=Path, default=Path("/home/qhq/serverless_llm_experiment_retry14_baseline"))
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--model-profile", required=True)
    ap.add_argument("--dataset-profile", required=True)
    ap.add_argument("--workload-profile", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--scenario-name", default="punica_fair")
    args = ap.parse_args()

    main_repo = args.main_repo.resolve()
    cfg_path = args.config.resolve() if args.config else (main_repo / "configs/experiments.yaml")
    cfg = _load_yaml(cfg_path)
    model_cfg, _adapters_cfg, _datasets_cfg, workload_cfg, coord_cfg = _resolve_profiles(
        cfg, args.model_profile, args.dataset_profile, args.workload_profile
    )

    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    trace = json.loads(args.trace.read_text(encoding="utf-8"))

    results = list(replay.get("results", []) or [])
    total = int(trace.get("total_requests", len(results)) or len(results))
    ok = [r for r in results if bool(r.get("success"))]
    failed = [r for r in results if not bool(r.get("success"))]

    ttft = [float(r["ttft_ms"]) for r in ok if r.get("ttft_ms") is not None]
    e2e = [float(r["e2e_ms"]) for r in ok if r.get("e2e_ms") is not None]
    tpot = [float(r["tpot_ms"]) for r in ok if r.get("tpot_ms") is not None]
    lora_io = [float(r["lora_load_ms"]) for r in ok if r.get("lora_load_ms") is not None]
    costs = [float(r.get("cost_usd", 0.0) or 0.0) for r in ok]
    completion_tokens = [int(r.get("completion_tokens", 0) or 0) for r in ok]
    elapsed_sec = float((replay.get("metadata", {}) or {}).get("elapsed_sec", 0.0) or 0.0)

    avg_ttft_ms = sum(ttft) / len(ttft) if ttft else 0.0
    avg_e2e_ms = sum(e2e) / len(e2e) if e2e else 0.0
    avg_tpot_ms = (sum(tpot) / len(tpot)) if tpot else None
    avg_cost_usd = sum(costs) / len(costs) if costs else 0.0
    total_cost_usd = sum(costs)
    throughput_rps = len(ok) / max(elapsed_sec, 1e-6) if elapsed_sec > 0 else 0.0
    throughput_tokps = sum(completion_tokens) / max(elapsed_sec, 1e-6) if elapsed_sec > 0 else 0.0
    ttft_slo_ms = float(workload_cfg.get("ttft_slo_ms", coord_cfg.get("ttft_slo_ms", 5000.0)) or 5000.0)
    slo_attainment = (sum(1 for r in ok if float(r.get("ttft_ms") or 0.0) <= ttft_slo_ms) / len(ok)) if ok else 0.0
    legacy_denom = avg_cost_usd * (avg_ttft_ms / 1000.0)
    qpr_tokps_ttft_legacy = throughput_tokps / legacy_denom if legacy_denom > 1e-12 else 0.0
    qpr_rps_legacy = throughput_rps / legacy_denom if legacy_denom > 1e-12 else 0.0
    ce_denom = avg_cost_usd * (avg_e2e_ms / 1000.0)
    ce = 1.0 / ce_denom if ce_denom > 1e-12 else 0.0
    cache_hit_rate = (
        sum(1 for r in ok if bool(r.get("cache_hit"))) / len(ok)
    ) if ok else None

    summary = {
        "scenario_name": args.scenario_name,
        "baseline_type": "punica",
        "backend": "punica_official",
        "instance_mode": "single_gpu",
        "routing_policy": "fcfs_batching_wrapper",
        "num_adapters": int(trace.get("selected_num_adapters", 0) or 0),
        "active_adapter_cap": workload_cfg.get("active_adapter_cap"),
        "hotset_rotation_requests": workload_cfg.get("hotset_rotation_requests"),
        "min_instances": 1,
        "max_instances": 1,
        "arrival_window_s": None,
        "scale_eval_interval_s": None,
        "total_requests": total,
        "completed_requests": len(ok),
        "failed_requests": len(failed),
        "elapsed_sec": _round(elapsed_sec),
        "avg_ttft_ms": _round(avg_ttft_ms),
        "p95_ttft_ms": _round(_pct(ttft, 95)),
        "p99_ttft_ms": _round(_pct(ttft, 99)),
        "avg_tpot_ms": _round(avg_tpot_ms),
        "avg_e2e_ms": _round(avg_e2e_ms),
        "p95_e2e_ms": _round(_pct(e2e, 95)),
        "p99_e2e_ms": _round(_pct(e2e, 99)),
        "throughput_rps": _round(throughput_rps, 6),
        "throughput_tok_per_s": _round(throughput_tokps, 6),
        "slo_attainment": _round(slo_attainment, 6),
        "ttft_slo_ms": _round(ttft_slo_ms),
        "avg_cost_usd": _round(avg_cost_usd, 8),
        "total_cost_usd": _round(total_cost_usd, 8),
        "ce": _round(ce, 6),
        "qpr": _round(ce, 6),
        "qpr_tokps_ttft_legacy": _round(qpr_tokps_ttft_legacy, 6),
        "qpr_rps_legacy": _round(qpr_rps_legacy, 6),
        "cache_hit_rate": _round(cache_hit_rate, 6),
        "gpu_hit_rate": None,
        "avg_lora_io_ms": _round(sum(lora_io) / len(lora_io) if lora_io else None),
        "avg_comparable_ttft_ms": None,
        "p95_comparable_ttft_ms": None,
        "p99_comparable_ttft_ms": None,
        "avg_warm_standard_ttft_ms": None,
        "p95_warm_standard_ttft_ms": None,
        "p99_warm_standard_ttft_ms": None,
        "avg_serverless_overhead_ms": None,
        "p95_serverless_overhead_ms": None,
        "avg_runtime_ttft_ms": None,
        "p95_runtime_ttft_ms": None,
        "avg_gpu_ready_ttft_ms": None,
        "p95_gpu_ready_ttft_ms": None,
        "avg_scaleup_affected_ttft_ms": None,
        "p95_scaleup_affected_ttft_ms": None,
        "avg_scaleup_runtime_ttft_ms": None,
        "p95_scaleup_runtime_ttft_ms": None,
        "scaleup_runtime_lora_request_count": 0,
        "scaleup_runtime_gpu_hit_rate": None,
        "avg_scaleup_first_service_ttft_ms": None,
        "scaleup_first_service_request_count": 0,
        "scaleup_first_service_gpu_hit_rate": None,
        "avg_cold_start_latency_ms": None,
        "p95_cold_start_latency_ms": None,
        "scale_up_events": None,
        "scale_down_events": None,
        "avg_prompt_tokens": _round(sum(int(r.get("prompt_tokens", 0) or 0) for r in ok) / len(ok) if ok else 0.0),
        "avg_completion_tokens": _round(sum(completion_tokens) / len(completion_tokens) if completion_tokens else 0.0),
    }

    metric_structure = _build_metric_structure()
    comparison_table = {
        "TTFT_overall_ms": _round(avg_ttft_ms),
        "TTFT_comp_ms": None,
        "TTFT_warm_ms": None,
        "TPOT_ms": _round(avg_tpot_ms),
        "Tok/s": _round(throughput_tokps, 4),
        "E2E_ms": _round(avg_e2e_ms),
        "Cost/req_USD": _round(avg_cost_usd, 8),
        "CE": _round(ce, 6),
        "SLO_attainment": _round(slo_attainment, 6),
        "Completed": len(ok),
        "Failed": len(failed),
    }
    scenario_summaries = {args.scenario_name: summary}

    payload = {
        "schema_version": 1,
        "metadata": {
            "system": "punica",
            "run_tag": (replay.get("metadata", {}) or {}).get("run_tag"),
            "model_profile": args.model_profile,
            "dataset_profile": args.dataset_profile,
            "workload_profile": args.workload_profile,
        },
        "metric_structure": metric_structure,
        "comparison_table": comparison_table,
        "sota_comparison": {},
        "scenario_summaries": scenario_summaries,
        "detailed_results": {
            "requests": results,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"punica summary -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
