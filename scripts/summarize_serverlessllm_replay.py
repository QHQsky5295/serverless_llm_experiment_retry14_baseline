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


def _cell(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _build_metric_structure() -> Dict[str, List[str]]:
    return {
        "standard_serving_metrics": [
            "TTFT_avg_ms",
            "TTFT_P95_ms",
            "TTFT_P99_ms",
            "TTFT_e2e_avg_ms",
            "TTFT_e2e_P95_ms",
            "TTFT_e2e_P99_ms",
            "TTFT_service_avg_ms",
            "TTFT_service_P95_ms",
            "TTFT_service_P99_ms",
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
            "TTFT_e2e_avg_ms",
            "TTFT_service_avg_ms",
            "Dispatch_admission_wait_avg_ms",
            "Dispatch_admission_wait_P95_ms",
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
            "TTFT_service_overhead_avg_ms",
            "TTFT_service_overhead_P95_ms",
            "E2E_avg_ms",
            "E2E_P95_ms",
            "E2E_P99_ms",
            "E2E_e2e_avg_ms",
            "E2E_service_avg_ms",
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


def _known_bool_rate(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    """Return a boolean rate only when the backend actually reported the field."""
    known = [record for record in records if record.get(key) is not None]
    if not known:
        return None
    return sum(1 for record in known if bool(record.get(key))) / len(known)


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize a ServerlessLLM replay into the paper metric schema.")
    ap.add_argument("--main-repo", type=Path, default=Path("/home/qhq/serverless_llm_experiment_retry14_baseline"))
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--adapter-subset", type=Path, default=None)
    ap.add_argument("--deploy", type=Path, default=None)
    ap.add_argument("--model-profile", required=True)
    ap.add_argument("--dataset-profile", required=True)
    ap.add_argument("--workload-profile", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--scenario-name", default="serverlessllm_fair")
    ap.add_argument("--baseline-type", default="serverlessllm")
    ap.add_argument("--backend-label", default="serverlessllm_official")
    ap.add_argument("--system-name", default="ServerlessLLM")
    ap.add_argument("--instance-mode", default="auto")
    ap.add_argument("--routing-policy", default="round_robin")
    args = ap.parse_args()

    main_repo = args.main_repo.resolve()
    cfg_path = args.config.resolve() if args.config else (main_repo / "configs/experiments.yaml")
    cfg = _load_yaml(cfg_path)
    cost_model = dict(cfg.get("cost_model", {}) or {})
    model_cfg, _adapters_cfg, _datasets_cfg, workload_cfg, coord_cfg = _resolve_profiles(
        cfg, args.model_profile, args.dataset_profile, args.workload_profile
    )

    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    deploy = json.loads(args.deploy.read_text(encoding="utf-8")) if args.deploy else {}

    results = list(replay.get("results", []) or [])
    total = int(trace.get("total_requests", len(results)) or len(results))
    ok = [r for r in results if bool(r.get("success"))]
    failed = [r for r in results if not bool(r.get("success"))]

    ttft = [float(r["ttft_ms"]) for r in ok if r.get("ttft_ms") is not None]
    e2e = [float(r["e2e_ms"]) for r in ok if r.get("e2e_ms") is not None]
    service_ttft = [
        float(r.get("service_ttft_ms", r.get("ttft_ms")))
        for r in ok
        if r.get("service_ttft_ms", r.get("ttft_ms")) is not None
    ]
    service_e2e = [
        float(r.get("service_e2e_ms", r.get("e2e_ms")))
        for r in ok
        if r.get("service_e2e_ms", r.get("e2e_ms")) is not None
    ]
    dispatch_wait = [
        float(r.get("dispatch_admission_wait_ms", 0.0) or 0.0)
        for r in ok
    ]
    tpot = [float(r["tpot_ms"]) for r in ok if r.get("tpot_ms") is not None]
    runtime_ttft = [
        float(r["runtime_ttft_ms"])
        for r in ok
        if r.get("runtime_ttft_ms") is not None
    ]
    overhead = [
        float(r["serverless_overhead_ms"])
        for r in ok
        if r.get("serverless_overhead_ms") is not None
    ]
    service_overhead = [
        float(r["service_overhead_ms"])
        for r in ok
        if r.get("service_overhead_ms") is not None
    ]
    lora_io = [
        float(r["lora_load_ms"])
        for r in ok
        if r.get("lora_load_ms") is not None
    ]
    comparable_ttft = [
        float(r["ttft_ms"])
        for r in ok
        if bool(r.get("comparable_request"))
    ]
    warm_standard_ttft = [
        float(r["ttft_ms"])
        for r in ok
        if bool(r.get("warm_standard_request"))
    ]
    gpu_ready_ttft = [
        float(r["ttft_ms"])
        for r in ok
        if bool(r.get("gpu_ready_request"))
    ]
    scaleup_affected_ttft = [
        float(r["ttft_ms"])
        for r in ok
        if bool(r.get("scaleup_affected"))
    ]
    scaleup_runtime_ttft = [
        float(r["runtime_ttft_ms"])
        for r in ok
        if bool(r.get("scaleup_affected")) and r.get("runtime_ttft_ms") is not None
    ]
    scaleup_runtime_lora = [r for r in ok if bool(r.get("scaleup_affected"))]
    scaleup_first_service = [r for r in ok if bool(r.get("scaleup_first_service"))]
    scaleup_first_service_ttft = [
        float(r["ttft_ms"])
        for r in scaleup_first_service
        if r.get("ttft_ms") is not None
    ]
    cold_start = [
        float(r["cold_start_latency_ms"])
        for r in ok
        if r.get("cold_start_latency_ms") is not None
    ]
    observed_tpot_count = sum(1 for r in ok if bool(r.get("tpot_observed")))
    costs = [float(r.get("cost_usd", 0.0) or 0.0) for r in ok]
    completion_tokens = [int(r.get("completion_tokens", 0) or 0) for r in ok]
    prompt_tokens = [int(r.get("prompt_tokens", 0) or 0) for r in ok]
    total_tokens = [int(r.get("total_tokens", 0) or 0) for r in ok]
    elapsed_sec = max((float(r.get("completion_offset_s", 0.0) or 0.0) for r in results), default=0.0)
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
    multi_token_ok = [r for r in ok if int(r.get("completion_tokens", 0) or 0) > 1]
    tpot_observed_ratio = (observed_tpot_count / len(multi_token_ok)) if multi_token_ok else 0.0
    cache_hit_rate = _known_bool_rate(ok, "cache_hit")
    gpu_hit_rate = _known_bool_rate(ok, "gpu_ready_request")

    summary = {
        "scenario_name": args.scenario_name,
        "baseline_type": str(args.baseline_type),
        "backend": str(args.backend_label),
        "instance_mode": str(args.instance_mode),
        "routing_policy": str(args.routing_policy),
        "num_adapters": int(trace.get("selected_num_adapters", 0) or 0),
        "active_adapter_cap": workload_cfg.get("active_adapter_cap"),
        "hotset_rotation_requests": workload_cfg.get("hotset_rotation_requests"),
        "min_instances": int(((deploy.get("auto_scaling_config", {}) or {}).get("min_instances", 0) or 0)),
        "max_instances": int(((deploy.get("auto_scaling_config", {}) or {}).get("max_instances", 0) or 0)),
        "arrival_window_s": None,
        "scale_eval_interval_s": 1.0,
        "total_requests": total,
        "completed_requests": len(ok),
        "failed_requests": len(failed),
        "elapsed_sec": _round(elapsed_sec),
        "metric_schema_version": "e2e_v3",
        "primary_ttft_definition": "scheduled trace arrival to client-observed first output token/chunk",
        "primary_e2e_definition": "scheduled trace arrival to client-observed response completion",
        "service_ttft_definition": "common system-ingress to first output token/chunk; ingress is request release/dispatch into the target serving system and excludes scheduled-arrival wait",
        "slo_metric": "TTFT_e2e",
        "avg_ttft_ms": _round(avg_ttft_ms),
        "p95_ttft_ms": _round(_pct(ttft, 95)),
        "p99_ttft_ms": _round(_pct(ttft, 99)),
        "avg_overall_ttft_ms": _round(avg_ttft_ms),
        "p50_overall_ttft_ms": _round(_pct(ttft, 50)),
        "p95_overall_ttft_ms": _round(_pct(ttft, 95)),
        "p99_overall_ttft_ms": _round(_pct(ttft, 99)),
        "avg_service_ttft_ms": _round(sum(service_ttft) / len(service_ttft) if service_ttft else None),
        "p95_service_ttft_ms": _round(_pct(service_ttft, 95)),
        "p99_service_ttft_ms": _round(_pct(service_ttft, 99)),
        "avg_tpot_ms": _round(avg_tpot_ms),
        "avg_e2e_ms": _round(avg_e2e_ms),
        "p95_e2e_ms": _round(_pct(e2e, 95)),
        "p99_e2e_ms": _round(_pct(e2e, 99)),
        "avg_overall_e2e_ms": _round(avg_e2e_ms),
        "p95_overall_e2e_ms": _round(_pct(e2e, 95)),
        "p99_overall_e2e_ms": _round(_pct(e2e, 99)),
        "avg_service_e2e_ms": _round(sum(service_e2e) / len(service_e2e) if service_e2e else None),
        "p95_service_e2e_ms": _round(_pct(service_e2e, 95)),
        "p99_service_e2e_ms": _round(_pct(service_e2e, 99)),
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
        "gpu_hit_rate": _round(gpu_hit_rate, 6),
        "avg_lora_io_ms": _round(sum(lora_io) / len(lora_io) if lora_io else None),
        "avg_comparable_ttft_ms": _round(sum(comparable_ttft) / len(comparable_ttft) if comparable_ttft else None),
        "p95_comparable_ttft_ms": _round(_pct(comparable_ttft, 95)),
        "p99_comparable_ttft_ms": _round(_pct(comparable_ttft, 99)),
        "avg_warm_standard_ttft_ms": _round(sum(warm_standard_ttft) / len(warm_standard_ttft) if warm_standard_ttft else None),
        "p95_warm_standard_ttft_ms": _round(_pct(warm_standard_ttft, 95)),
        "p99_warm_standard_ttft_ms": _round(_pct(warm_standard_ttft, 99)),
        "avg_serverless_overhead_ms": _round(sum(overhead) / len(overhead) if overhead else None),
        "p95_serverless_overhead_ms": _round(_pct(overhead, 95)),
        "avg_service_overhead_ms": _round(sum(service_overhead) / len(service_overhead) if service_overhead else None),
        "p95_service_overhead_ms": _round(_pct(service_overhead, 95)),
        "avg_dispatch_admission_wait_ms": _round(sum(dispatch_wait) / len(dispatch_wait) if dispatch_wait else None),
        "p95_dispatch_admission_wait_ms": _round(_pct(dispatch_wait, 95)),
        "avg_runtime_ttft_ms": _round(sum(runtime_ttft) / len(runtime_ttft) if runtime_ttft else None),
        "p95_runtime_ttft_ms": _round(_pct(runtime_ttft, 95)),
        "avg_gpu_ready_ttft_ms": _round(sum(gpu_ready_ttft) / len(gpu_ready_ttft) if gpu_ready_ttft else None),
        "p95_gpu_ready_ttft_ms": _round(_pct(gpu_ready_ttft, 95)),
        "avg_scaleup_affected_ttft_ms": _round(sum(scaleup_affected_ttft) / len(scaleup_affected_ttft) if scaleup_affected_ttft else None),
        "p95_scaleup_affected_ttft_ms": _round(_pct(scaleup_affected_ttft, 95)),
        "avg_scaleup_runtime_ttft_ms": _round(sum(scaleup_runtime_ttft) / len(scaleup_runtime_ttft) if scaleup_runtime_ttft else None),
        "p95_scaleup_runtime_ttft_ms": _round(_pct(scaleup_runtime_ttft, 95)),
        "scaleup_runtime_lora_request_count": len(scaleup_runtime_lora),
        "scaleup_runtime_gpu_hit_rate": _round(_known_bool_rate(scaleup_runtime_lora, "gpu_ready_request"), 6),
        "avg_scaleup_first_service_ttft_ms": _round(sum(scaleup_first_service_ttft) / len(scaleup_first_service_ttft) if scaleup_first_service_ttft else None),
        "scaleup_first_service_request_count": len(scaleup_first_service),
        "scaleup_first_service_gpu_hit_rate": _round(_known_bool_rate(scaleup_first_service, "gpu_ready_request"), 6),
        "scaleup_first_service_planned_match_rate": None,
        "avg_cold_start_latency_ms": _round(sum(cold_start) / len(cold_start) if cold_start else None),
        "p95_cold_start_latency_ms": _round(_pct(cold_start, 95)),
        "cost_effectiveness_e2e": _round(ce, 6),
        "slo_goodput_rps": _round(throughput_rps * slo_attainment, 6),
        "slo_goodput_tok_per_s": _round(throughput_tokps * slo_attainment, 6),
        "tpot_observed_request_ratio": _round(tpot_observed_ratio, 6),
        "contention_events": None,
        "avg_contention_ms": None,
        "avg_defer_ms": None,
        "gpu_ready_hits": None,
        "warm_pool_hits": None,
        "scale_up_events": None,
        "scale_down_events": None,
        "scale_down_event_log": None,
        "cold_starts_after_scale_up": None,
        "standard_serving_metrics": {
            "TTFT_avg_ms": _round(avg_ttft_ms),
            "TTFT_P95_ms": _round(_pct(ttft, 95)),
            "TTFT_P99_ms": _round(_pct(ttft, 99)),
            "TTFT_e2e_avg_ms": _round(avg_ttft_ms),
            "TTFT_e2e_P95_ms": _round(_pct(ttft, 95)),
            "TTFT_e2e_P99_ms": _round(_pct(ttft, 99)),
            "TTFT_service_avg_ms": _round(sum(service_ttft) / len(service_ttft) if service_ttft else None),
            "TTFT_service_P95_ms": _round(_pct(service_ttft, 95)),
            "TTFT_service_P99_ms": _round(_pct(service_ttft, 99)),
            "TTFT_warm_standard_avg_ms": _round(sum(warm_standard_ttft) / len(warm_standard_ttft) if warm_standard_ttft else None),
            "TTFT_warm_standard_P95_ms": _round(_pct(warm_standard_ttft, 95)),
            "TTFT_warm_standard_P99_ms": _round(_pct(warm_standard_ttft, 99)),
            "TPOT_avg_ms": _round(avg_tpot_ms),
            "Throughput_RPS": _round(throughput_rps),
            "Throughput_TOKPS": _round(throughput_tokps),
            "Cost_effectiveness_e2e": _round(ce),
        },
        "serverless_deployment_metrics": {
            "TTFT_overall_avg_ms": _round(avg_ttft_ms),
            "TTFT_e2e_avg_ms": _round(avg_ttft_ms),
            "TTFT_service_avg_ms": _round(sum(service_ttft) / len(service_ttft) if service_ttft else None),
            "Dispatch_admission_wait_avg_ms": _round(sum(dispatch_wait) / len(dispatch_wait) if dispatch_wait else None),
            "Dispatch_admission_wait_P95_ms": _round(_pct(dispatch_wait, 95)),
            "TTFT_comparable_avg_ms": _round(sum(comparable_ttft) / len(comparable_ttft) if comparable_ttft else None),
            "TTFT_comparable_P95_ms": _round(_pct(comparable_ttft, 95)),
            "TTFT_comparable_P99_ms": _round(_pct(comparable_ttft, 99)),
            "TTFT_gpu_ready_avg_ms": _round(sum(gpu_ready_ttft) / len(gpu_ready_ttft) if gpu_ready_ttft else None),
            "TTFT_gpu_ready_P95_ms": _round(_pct(gpu_ready_ttft, 95)),
            "TTFT_scaleup_affected_avg_ms": _round(sum(scaleup_affected_ttft) / len(scaleup_affected_ttft) if scaleup_affected_ttft else None),
            "TTFT_scaleup_affected_P95_ms": _round(_pct(scaleup_affected_ttft, 95)),
            "Runtime_TTFT_avg_ms": _round(sum(runtime_ttft) / len(runtime_ttft) if runtime_ttft else None),
            "Runtime_TTFT_P95_ms": _round(_pct(runtime_ttft, 95)),
            "Cold_start_avg_ms": _round(sum(cold_start) / len(cold_start) if cold_start else None),
            "Cold_start_P95_ms": _round(_pct(cold_start, 95)),
            "TTFT_serverless_overhead_avg_ms": _round(sum(overhead) / len(overhead) if overhead else None),
            "TTFT_serverless_overhead_P95_ms": _round(_pct(overhead, 95)),
            "TTFT_service_overhead_avg_ms": _round(sum(service_overhead) / len(service_overhead) if service_overhead else None),
            "TTFT_service_overhead_P95_ms": _round(_pct(service_overhead, 95)),
            "E2E_avg_ms": _round(avg_e2e_ms),
            "E2E_P95_ms": _round(_pct(e2e, 95)),
            "E2E_P99_ms": _round(_pct(e2e, 99)),
            "E2E_e2e_avg_ms": _round(avg_e2e_ms),
            "E2E_service_avg_ms": _round(sum(service_e2e) / len(service_e2e) if service_e2e else None),
            "Monetary_cost_avg_usd": _round(avg_cost_usd, 8),
            "Monetary_cost_total_usd": _round(total_cost_usd, 8),
        },
        "scaling_metrics": {
            "TTFT_SLO_ms": _round(ttft_slo_ms),
            "SLO_attainment": _round(slo_attainment),
            "SLO_goodput_RPS": _round(throughput_rps * slo_attainment),
            "SLO_goodput_TOKPS": _round(throughput_tokps * slo_attainment),
            "scale_up_events": None,
            "scale_down_events": None,
        },
        "mechanism_metrics": {
            "CE": _round(ce),
            "QPR_TOKPS_TTFT_legacy": _round(qpr_tokps_ttft_legacy),
            "QPR_RPS_legacy": _round(qpr_rps_legacy),
            "TPOT_observed_ratio": _round(tpot_observed_ratio),
            "cache_hit_rate": _round(cache_hit_rate),
            "GPU_hit_rate": _round(gpu_hit_rate),
            "LoRA_IO_avg_ms": _round(sum(lora_io) / len(lora_io) if lora_io else None),
            "ScaleUp_runtime_lora_requests": len(scaleup_runtime_lora),
            "ScaleUp_runtime_gpu_hit_rate": _round(
                (
                    sum(1 for r in scaleup_runtime_lora if bool(r.get("gpu_ready_request")))
                    / len(scaleup_runtime_lora)
                ) if scaleup_runtime_lora else None
            ),
            "ScaleUp_first_service_requests": len(scaleup_first_service),
            "ScaleUp_first_service_gpu_hit_rate": _round(
                (
                    sum(1 for r in scaleup_first_service if bool(r.get("gpu_ready_request")))
                    / len(scaleup_first_service)
                ) if scaleup_first_service else None
            ),
            "ScaleUp_first_service_plan_match_rate": None,
            "GPU_ready_hits": None,
            "Warm_pool_hits": None,
            "C3_contention_events": None,
            "C3_contention_penalty_ms": None,
            "C3_defer_delay_ms": None,
        },
    }

    comparison_row = {
        "scenario": args.scenario_name,
        "baseline_type": str(args.baseline_type),
        "completed": len(ok),
        "total": total,
        "TTFT_avg_ms": _cell(avg_ttft_ms),
        "TTFT_P50_ms": _cell(_pct(ttft, 50)),
        "TTFT_P95_ms": _cell(_pct(ttft, 95)),
        "TTFT_P99_ms": _cell(_pct(ttft, 99)),
        "TTFT_e2e_avg_ms": _cell(avg_ttft_ms),
        "TTFT_e2e_P95_ms": _cell(_pct(ttft, 95)),
        "TTFT_e2e_P99_ms": _cell(_pct(ttft, 99)),
        "TTFT_service_avg_ms": _cell(sum(service_ttft) / len(service_ttft) if service_ttft else None),
        "TTFT_service_P95_ms": _cell(_pct(service_ttft, 95)),
        "TTFT_service_P99_ms": _cell(_pct(service_ttft, 99)),
        "TTFT_comparable_avg_ms": _cell(sum(comparable_ttft) / len(comparable_ttft) if comparable_ttft else None),
        "TTFT_comparable_P95_ms": _cell(_pct(comparable_ttft, 95)),
        "TTFT_comparable_P99_ms": _cell(_pct(comparable_ttft, 99)),
        "TTFT_warm_standard_avg_ms": _cell(sum(warm_standard_ttft) / len(warm_standard_ttft) if warm_standard_ttft else None),
        "TTFT_warm_standard_P95_ms": _cell(_pct(warm_standard_ttft, 95)),
        "TTFT_warm_standard_P99_ms": _cell(_pct(warm_standard_ttft, 99)),
        "Runtime_TTFT_avg_ms": _cell(sum(runtime_ttft) / len(runtime_ttft) if runtime_ttft else None),
        "Runtime_TTFT_P95_ms": _cell(_pct(runtime_ttft, 95)),
        "TTFT_gpu_ready_avg_ms": _cell(sum(gpu_ready_ttft) / len(gpu_ready_ttft) if gpu_ready_ttft else None),
        "TTFT_scaleup_affected_avg_ms": _cell(sum(scaleup_affected_ttft) / len(scaleup_affected_ttft) if scaleup_affected_ttft else None),
        "TTFT_scaleup_runtime_avg_ms": _cell(sum(scaleup_runtime_ttft) / len(scaleup_runtime_ttft) if scaleup_runtime_ttft else None),
        "TTFT_scaleup_first_service_avg_ms": _cell(sum(scaleup_first_service_ttft) / len(scaleup_first_service_ttft) if scaleup_first_service_ttft else None),
        "Cold_start_avg_ms": _cell(sum(cold_start) / len(cold_start) if cold_start else None),
        "Cold_start_P95_ms": _cell(_pct(cold_start, 95)),
        "TTFT_serverless_overhead_avg_ms": _cell(sum(overhead) / len(overhead) if overhead else None),
        "TTFT_service_overhead_avg_ms": _cell(sum(service_overhead) / len(service_overhead) if service_overhead else None),
        "Dispatch_admission_wait_avg_ms": _cell(sum(dispatch_wait) / len(dispatch_wait) if dispatch_wait else None),
        "Dispatch_admission_wait_P95_ms": _cell(_pct(dispatch_wait, 95)),
        "TPOT_avg_ms": _cell(avg_tpot_ms),
        "TPOT_observed_ratio": _cell(tpot_observed_ratio, 4),
        "E2E_avg_ms": _cell(avg_e2e_ms),
        "E2E_P95_ms": _cell(_pct(e2e, 95)),
        "E2E_P99_ms": _cell(_pct(e2e, 99)),
        "E2E_e2e_avg_ms": _cell(avg_e2e_ms),
        "E2E_service_avg_ms": _cell(sum(service_e2e) / len(service_e2e) if service_e2e else None),
        "throughput_RPS": _cell(throughput_rps, 3),
        "throughput_TOKPS": _cell(throughput_tokps, 3),
        "SLO_attainment": _cell(slo_attainment, 4),
        "SLO_goodput_RPS": _cell(throughput_rps * slo_attainment, 4),
        "SLO_goodput_TOKPS": _cell(throughput_tokps * slo_attainment, 4),
        "TTFT_SLO_ms": _cell(ttft_slo_ms, 1),
        "avg_cost_USD": _cell(avg_cost_usd, 7),
        "total_cost_USD": _cell(total_cost_usd, 5),
        "CE": _cell(ce, 4),
        "Cost_effectiveness_e2e": _cell(ce, 4),
        "QPR_TOKPS_TTFT_legacy": _cell(qpr_tokps_ttft_legacy, 1),
        "QPR_RPS_legacy": _cell(qpr_rps_legacy, 1),
        "cache_hit_rate": _cell(cache_hit_rate, 4),
        "GPU_hit_rate": _cell(gpu_hit_rate, 4),
        "LoRA_IO_avg_ms": _cell(sum(lora_io) / len(lora_io) if lora_io else None),
        "ScaleUp_runtime_lora_requests": len(scaleup_runtime_lora),
        "ScaleUp_runtime_gpu_hit_rate": _cell(
            (
                sum(1 for r in scaleup_runtime_lora if bool(r.get("gpu_ready_request")))
                / len(scaleup_runtime_lora)
            ) if scaleup_runtime_lora else None,
            4,
        ),
        "ScaleUp_first_service_requests": len(scaleup_first_service),
        "ScaleUp_first_service_gpu_hit_rate": _cell(
            (
                sum(1 for r in scaleup_first_service if bool(r.get("gpu_ready_request")))
                / len(scaleup_first_service)
            ) if scaleup_first_service else None,
            4,
        ),
        "ScaleUp_first_service_plan_match_rate": None,
        "C3_contention_events": None,
        "C3_contention_penalty_ms": None,
        "C3_defer_delay_ms": None,
        "C3_gpu_ready_hits": None,
        "C3_warm_pool_hits": None,
        "standard_serving_metrics": summary["standard_serving_metrics"],
        "serverless_deployment_metrics": summary["serverless_deployment_metrics"],
        "scaling_metrics": summary["scaling_metrics"],
        "mechanism_metrics": summary["mechanism_metrics"],
        "E1_scale_down_events": None,
        "E1_scale_down_event_log": None,
        "E1_scale_up_events": None,
        "E1_cold_starts_after_scale_up": None,
    }

    detailed_results = {
        args.scenario_name: {
            "scenario_name": args.scenario_name,
            "baseline_type": str(args.baseline_type),
            "total": total,
            "completed": len(ok),
            "failed": len(failed),
            "elapsed_sec": elapsed_sec,
            "requests": results,
        }
    }

    output = {
        "schema_version": 4,
        "metric_schema_version": "e2e_v3",
        "metric_definitions": {
            "primary_ttft": "scheduled trace arrival to client-observed first output token/chunk",
            "primary_e2e": "scheduled trace arrival to client-observed response completion",
            "service_ttft": "common system-ingress to first output token/chunk; ingress is request release/dispatch into the target serving system and excludes scheduled-arrival wait",
            "service_e2e": "common system-ingress to response completion; ingress is request release/dispatch into the target serving system and excludes scheduled-arrival wait",
            "tpot": "(final_token_time - first_token_time) / max(output_tokens - 1, 1)",
            "slo_metric": "TTFT_e2e",
        },
        "metadata": {
            "system": str(args.system_name),
            "metric_schema_version": "e2e_v3",
            "primary_ttft_definition": "scheduled trace arrival to client-observed first output token/chunk",
            "primary_e2e_definition": "scheduled trace arrival to client-observed response completion",
            "service_ttft_definition": "common system-ingress to first output token/chunk; ingress is request release/dispatch into the target serving system and excludes scheduled-arrival wait",
            "slo_metric": "TTFT_e2e",
            "main_repo": str(main_repo),
            "config_path": str(cfg_path),
            "model_profile": args.model_profile,
            "dataset_profile": args.dataset_profile,
            "workload_profile": args.workload_profile,
            "trace_source": str(args.trace),
            "adapter_subset_path": str(args.adapter_subset) if args.adapter_subset else None,
            "replay_source": str(args.replay),
            "deploy_config": str(args.deploy) if args.deploy else None,
            "model_name": str(model_cfg.get("name")),
            "selected_num_adapters": trace.get("selected_num_adapters"),
            "sampling_seed": trace.get("sampling_seed"),
        },
        "metric_structure": _build_metric_structure(),
        "comparison_table": [comparison_row],
        "sota_comparison": [comparison_row],
        "scenario_summaries": {args.scenario_name: summary},
        "detailed_results": detailed_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    tpot_display = f"{avg_tpot_ms:.1f}ms" if avg_tpot_ms is not None else "n/a"
    print(
        f"Summary[{args.scenario_name}] "
        f"TTFT={avg_ttft_ms:.1f}ms TPOT={tpot_display} "
        f"Tok/s={throughput_tokps:.2f} E2E={avg_e2e_ms:.1f}ms "
        f"Cost/req=${avg_cost_usd:.6f} CE={ce:.3f}"
    )
    print(f"wrote summary -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
