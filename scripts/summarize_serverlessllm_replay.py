#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

METRIC_DEF_PRIMARY_TTFT = (
    "scheduled trace arrival to system-observed first generated output token/chunk"
)
METRIC_DEF_PRIMARY_E2E = (
    "scheduled trace arrival to system-observed response completion when available; "
    "otherwise client-observed completion"
)
METRIC_DEF_SERVICE_TTFT = (
    "service-path start to first generated output token/chunk; service-path start "
    "is the system-specific admitted/backend-execution start and excludes scheduled-arrival "
    "and upstream queue/admission wait"
)
METRIC_DEF_SERVICE_E2E = (
    "service-path start to response completion; service-path start is the system-specific "
    "admitted/backend-execution start and excludes scheduled-arrival and upstream "
    "queue/admission wait"
)


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


def _cost_per_million_tokens(total_cost_usd: float, token_count: int) -> float:
    try:
        cost = float(total_cost_usd or 0.0)
        tokens = int(token_count or 0)
    except Exception:
        return 0.0
    if cost <= 0.0 or tokens <= 0:
        return 0.0
    return cost * 1_000_000.0 / float(tokens)


def _load_structured_file(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    raw = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        parsed = yaml.safe_load(raw) or {}
        return parsed if isinstance(parsed, dict) else {}


def _gpu_cost_per_second_usd(cost_model: Dict[str, Any]) -> float:
    per_second = cost_model.get("gpu_cost_per_second_usd")
    if per_second is not None:
        try:
            return max(0.0, float(per_second))
        except Exception:
            pass
    per_hour = cost_model.get("gpu_hour_cost_usd")
    if per_hour is not None:
        try:
            return max(0.0, float(per_hour)) / 3600.0
        except Exception:
            pass
    return 0.0008


def _serverless_idle_gpu_cost_factor(cost_model: Dict[str, Any]) -> float:
    for key in (
        "serverless_idle_gpu_cost_factor",
        "idle_gpu_cost_factor",
        "gpu_idle_cost_factor",
    ):
        value = cost_model.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
        except Exception:
            continue
        if math.isfinite(parsed):
            return min(1.0, max(0.0, parsed))
    # Alibaba Function Compute Tesla GPU CU factors: idle 0.5 / active 2.1.
    return 0.5 / 2.1


def _serverless_invocation_cost_per_request_usd(cost_model: Dict[str, Any]) -> float:
    for key in (
        "serverless_invocation_cost_per_request_usd",
        "invocation_cost_per_request_usd",
        "function_invocation_cost_usd",
    ):
        value = cost_model.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
        except Exception:
            continue
        if math.isfinite(parsed):
            return max(0.0, parsed)
    return 0.0


def _cost_model_deployment_idle_tail_s(cost_model: Dict[str, Any]) -> float:
    for key in (
        "deployment_idle_tail_s",
        "deployment_lifecycle_idle_tail_s",
        "post_workload_idle_tail_s",
    ):
        value = cost_model.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
        except Exception:
            continue
        if math.isfinite(parsed):
            return max(0.0, parsed)
    return 300.0


def _cost_model_serverless_idle_retention_s(cost_model: Dict[str, Any]) -> float:
    for key in (
        "serverless_idle_retention_s",
        "serverless_keepalive_s",
        "function_keepalive_s",
    ):
        value = cost_model.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
        except Exception:
            continue
        if math.isfinite(parsed):
            return max(0.0, parsed)
    return _cost_model_deployment_idle_tail_s(cost_model)


def _is_serverful_cost_runtime(lifecycles: List[Dict[str, Any]]) -> bool:
    runtime_kinds = [
        str((item or {}).get("runtime_kind", "") or "").lower()
        for item in lifecycles or []
    ]
    return bool(runtime_kinds) and all("static_serverful" in kind for kind in runtime_kinds)


def _summarize_monetary_cost_from_resource_seconds(
    *,
    cost_model: Dict[str, Any],
    completed_requests: int,
    total_requests: int,
    avg_e2e_ms: float,
    lifecycles: List[Dict[str, Any]],
    infra_gpu_seconds_total: float,
    infra_startup_gpu_seconds: float,
    infra_active_gpu_seconds: float,
    infra_idle_ready_gpu_seconds: float,
) -> Dict[str, Any]:
    per_second = _gpu_cost_per_second_usd(cost_model)
    idle_factor = _serverless_idle_gpu_cost_factor(cost_model)
    invocation_cost = _serverless_invocation_cost_per_request_usd(cost_model)
    completed = max(0, int(completed_requests or 0))
    denominator = completed if completed > 0 else max(0, int(total_requests or 0))
    serverful = _is_serverful_cost_runtime(lifecycles)

    total_gpu_s = max(0.0, float(infra_gpu_seconds_total or 0.0))
    startup_gpu_s = max(0.0, float(infra_startup_gpu_seconds or 0.0))
    active_gpu_s = max(0.0, float(infra_active_gpu_seconds or 0.0))
    idle_ready_gpu_s = max(0.0, float(infra_idle_ready_gpu_seconds or 0.0))

    if serverful:
        monetary_equivalent_gpu_seconds = total_gpu_s
        charged_idle_factor = 1.0
        active_charge_gpu_seconds = total_gpu_s
        idle_charge_gpu_seconds = 0.0
    else:
        active_charge_gpu_seconds = startup_gpu_s + active_gpu_s
        idle_charge_gpu_seconds = idle_ready_gpu_s * idle_factor
        monetary_equivalent_gpu_seconds = active_charge_gpu_seconds + idle_charge_gpu_seconds
        charged_idle_factor = idle_factor

    invocation_total = invocation_cost * float(completed)
    total_cost = monetary_equivalent_gpu_seconds * per_second + invocation_total
    per_request = total_cost / float(denominator) if denominator > 0 else 0.0
    avg_e2e_s = max(0.0, float(avg_e2e_ms or 0.0)) / 1000.0
    ce_denom = per_request * avg_e2e_s
    ce = 1.0 / ce_denom if ce_denom > 1e-12 else 0.0
    return {
        "monetary_cost_total_usd": total_cost,
        "monetary_cost_per_request_usd": per_request,
        "monetary_ce": ce,
        "monetary_equivalent_gpu_seconds": monetary_equivalent_gpu_seconds,
        "monetary_active_charge_gpu_seconds": active_charge_gpu_seconds,
        "monetary_idle_charge_gpu_seconds": idle_charge_gpu_seconds,
        "serverless_idle_gpu_cost_factor": charged_idle_factor,
        "serverless_invocation_cost_total_usd": invocation_total,
        "serverless_invocation_cost_per_request_usd": invocation_cost,
        "monetary_pricing_runtime_class": "serverful" if serverful else "serverless",
    }


def _capped_weighted_interval_seconds(
    intervals: List[Tuple[float, float, float]],
    max_weight: Optional[float] = None,
) -> float:
    events: List[Tuple[float, float]] = []
    for start, end, weight in intervals or []:
        try:
            s = float(start)
            e = float(end)
            w = float(weight)
        except Exception:
            continue
        if not (math.isfinite(s) and math.isfinite(e) and math.isfinite(w)):
            continue
        if e <= s or w <= 0.0:
            continue
        events.append((s, w))
        events.append((e, -w))
    if not events:
        return 0.0

    cap: Optional[float] = None
    if max_weight is not None:
        try:
            parsed = float(max_weight)
            if math.isfinite(parsed) and parsed > 0.0:
                cap = parsed
        except Exception:
            cap = None

    total = 0.0
    current = 0.0
    previous: Optional[float] = None
    idx = 0
    events.sort(key=lambda item: item[0])
    while idx < len(events):
        at = events[idx][0]
        if previous is not None and at > previous:
            effective = min(current, cap) if cap is not None else current
            total += (at - previous) * max(0.0, effective)
        while idx < len(events) and events[idx][0] == at:
            current += events[idx][1]
            idx += 1
        previous = at
    return max(0.0, total)


def _replay_start_wall_time(results: List[Dict[str, Any]]) -> Optional[float]:
    candidates: List[float] = []
    for record in results:
        sm = dict(record.get("server_metrics") or {})
        request_received_at = sm.get("request_received_at")
        dispatch_offset_s = record.get("dispatch_offset_s")
        if request_received_at is not None and dispatch_offset_s is not None:
            try:
                candidates.append(float(request_received_at) - float(dispatch_offset_s))
            except Exception:
                pass
            continue
        finished_at = sm.get("finished_at") or sm.get("last_token_at") or sm.get("response_sent_to_client_ts")
        completion_offset_s = record.get("completion_offset_s")
        if finished_at is not None and completion_offset_s is not None:
            try:
                candidates.append(float(finished_at) - float(completion_offset_s))
            except Exception:
                pass
    if candidates:
        return min(candidates)
    return None


def _serverlessllm_runtime_gpu_count(deploy: Dict[str, Any], model_cfg: Dict[str, Any]) -> int:
    deploy_count = deploy.get("num_gpus")
    backend_cfg = dict(deploy.get("backend_config", {}) or {})
    tp = backend_cfg.get("tensor_parallel_size", model_cfg.get("tensor_parallel_size", 1))
    candidates = []
    for value in (deploy_count, tp, model_cfg.get("tensor_parallel_size", 1)):
        try:
            candidates.append(int(value))
        except Exception:
            continue
    return max([v for v in candidates if v > 0] or [1])


def _sglang_runtime_gpu_count(deploy: Dict[str, Any], model_cfg: Dict[str, Any]) -> int:
    for key in ("num_gpus", "runtime_gpu_count"):
        value = deploy.get(key)
        if value is not None:
            try:
                parsed = int(value)
                if parsed > 0:
                    return parsed
            except Exception:
                pass
    try:
        dp = int(deploy.get("data_parallel_replicas", deploy.get("dp", 1)) or 1)
        tp = int(deploy.get("tp", deploy.get("tensor_parallel_size", model_cfg.get("tensor_parallel_size", 1))) or 1)
        if dp > 0 and tp > 0:
            return dp * tp
    except Exception:
        pass
    for key in ("tp", "tensor_parallel_size"):
        value = deploy.get(key)
        if value is not None:
            try:
                parsed = int(value)
                if parsed > 0:
                    return parsed
            except Exception:
                pass
    try:
        return max(1, int(model_cfg.get("tensor_parallel_size", 1) or 1))
    except Exception:
        return 1


def _static_runtime_lifecycle(
    *,
    elapsed_sec: float,
    startup_sec: float = 0.0,
    gpu_count: int,
    label: str,
    model_name: str,
    replica_count: int = 1,
    gpu_per_replica: Optional[int] = None,
) -> List[Dict[str, Any]]:
    safe_elapsed = max(0.0, float(elapsed_sec or 0.0))
    safe_startup = max(0.0, float(startup_sec or 0.0))
    replicas = max(1, int(replica_count or 1))
    per_replica = (
        max(1, int(gpu_per_replica or 1))
        if gpu_per_replica is not None
        else max(1, int(math.ceil(max(1, int(gpu_count or 1)) / replicas)))
    )
    return [
        {
            "instance_id": f"{label}_static_runtime_{idx}",
            "runtime_kind": "static_serverful",
            "gpu_count": per_replica,
            "model_name": str(model_name or ""),
            "lifecycle_source": "static_runtime_window_with_startup",
            "created_offset_s": 0.0,
            "ready_offset_s": safe_startup,
            "removed_offset_s": safe_startup + safe_elapsed,
        }
        for idx in range(replicas)
    ]


def _reconstruct_serverless_instance_lifecycles(
    *,
    results: List[Dict[str, Any]],
    elapsed_sec: float,
    gpu_count: int,
    model_name: str,
) -> List[Dict[str, Any]]:
    replay_start_wall = _replay_start_wall_time(results)
    if replay_start_wall is None:
        return []
    instances: Dict[str, Dict[str, Any]] = {}
    for record in results:
        sm = dict(record.get("server_metrics") or {})
        instance_id = str(sm.get("instance_id") or "").strip()
        if not instance_id:
            continue
        created_at = sm.get("instance_created_at")
        ready_at = sm.get("instance_ready_at")
        request_received_at = sm.get("request_received_at")
        backend_started_at = sm.get("backend_started_at")
        finished_at = (
            sm.get("finished_at")
            or sm.get("last_token_at")
            or sm.get("response_sent_to_client_ts")
            or sm.get("first_token_at")
            or backend_started_at
            or request_received_at
        )
        try:
            created_abs = float(
                created_at
                if created_at is not None
                else request_received_at
                if request_received_at is not None
                else backend_started_at
                if backend_started_at is not None
                else finished_at
            )
            ready_abs = float(
                ready_at
                if ready_at is not None
                else backend_started_at
                if backend_started_at is not None
                else request_received_at
                if request_received_at is not None
                else created_abs
            )
            finished_abs = float(finished_at)
        except Exception:
            continue
        rec = instances.setdefault(
            instance_id,
            {
                "instance_id": instance_id,
                "runtime_kind": "serverless_runtime",
                "gpu_count": max(1, int(gpu_count or 1)),
                "model_name": str(model_name or ""),
                "lifecycle_source": "server_metrics_reconstructed",
                "created_abs": created_abs,
                "ready_abs": ready_abs,
                "last_finished_abs": finished_abs,
            },
        )
        rec["created_abs"] = min(float(rec.get("created_abs", created_abs)), created_abs)
        rec["ready_abs"] = min(float(rec.get("ready_abs", ready_abs)), ready_abs)
        rec["last_finished_abs"] = max(float(rec.get("last_finished_abs", finished_abs)), finished_abs)

    lifecycles: List[Dict[str, Any]] = []
    for rec in instances.values():
        created_offset_s = max(0.0, float(rec["created_abs"]) - replay_start_wall)
        ready_offset_s = max(created_offset_s, float(rec["ready_abs"]) - replay_start_wall)
        removed_offset_s = max(
            ready_offset_s,
            float(rec["last_finished_abs"]) - replay_start_wall,
        )
        lifecycles.append(
            {
                "instance_id": rec["instance_id"],
                "runtime_kind": rec["runtime_kind"],
                "gpu_count": rec["gpu_count"],
                "model_name": rec["model_name"],
                "lifecycle_source": rec["lifecycle_source"],
                "created_offset_s": created_offset_s,
                "ready_offset_s": ready_offset_s,
                "removed_offset_s": removed_offset_s,
                "last_finished_offset_s": max(0.0, float(rec["last_finished_abs"]) - replay_start_wall),
            }
        )
    lifecycles.sort(key=lambda item: (float(item.get("created_offset_s", 0.0) or 0.0), str(item.get("instance_id") or "")))
    return lifecycles


def _summarize_infra_from_lifecycles(
    lifecycles: List[Dict[str, Any]],
    *,
    elapsed_sec: float,
    completed_requests: int,
    total_requests: int,
    avg_e2e_ms: float,
    gpu_cost_per_second_usd: float,
    deployment_idle_tail_s: float = 300.0,
    serverless_idle_retention_s: float = 300.0,
    max_billing_gpus: Optional[float] = None,
) -> Dict[str, Any]:
    normalized: List[Dict[str, Any]] = []
    total_gpu_seconds = 0.0
    startup_gpu_seconds = 0.0
    ready_gpu_seconds = 0.0
    safe_elapsed = max(0.0, float(elapsed_sec or 0.0))
    safe_tail = max(0.0, float(deployment_idle_tail_s or 0.0))
    safe_retention = max(0.0, float(serverless_idle_retention_s or 0.0))
    billing_elapsed = safe_elapsed + safe_tail
    lifetime_intervals: List[Tuple[float, float, float]] = []
    startup_intervals: List[Tuple[float, float, float]] = []
    ready_intervals: List[Tuple[float, float, float]] = []
    for raw in lifecycles or []:
        item = dict(raw or {})
        instance_id = str(item.get("instance_id") or "").strip()
        if not instance_id:
            continue
        runtime_kind = str(item.get("runtime_kind") or "").lower()
        remove_reason = str(item.get("remove_reason") or "").lower()
        gpu_count = max(1, int(item.get("gpu_count", 1) or 1))
        created_offset_s = min(max(0.0, float(item.get("created_offset_s", 0.0) or 0.0)), billing_elapsed)
        ready_offset_s = min(
            max(created_offset_s, float(item.get("ready_offset_s", created_offset_s) or created_offset_s)),
            billing_elapsed,
        )
        removed_offset_s = max(ready_offset_s, float(item.get("removed_offset_s", safe_elapsed) or safe_elapsed))
        last_finished_offset_s = item.get("last_finished_offset_s")
        try:
            last_finished_offset_s = float(last_finished_offset_s)
        except Exception:
            last_finished_offset_s = None
        if "static_serverful" in runtime_kind:
            billing_removed_offset_s = billing_elapsed
        elif remove_reason == "scale_down":
            billing_removed_offset_s = removed_offset_s
        else:
            live_until = max(
                removed_offset_s,
                float(last_finished_offset_s)
                if last_finished_offset_s is not None
                else removed_offset_s,
            )
            billing_removed_offset_s = live_until + safe_retention
        removed_offset_s = min(max(ready_offset_s, billing_removed_offset_s), billing_elapsed)
        lifetime_sec = max(0.0, removed_offset_s - created_offset_s)
        startup_sec = max(0.0, ready_offset_s - created_offset_s)
        ready_sec = max(0.0, removed_offset_s - ready_offset_s)
        lifetime_gpu_seconds = lifetime_sec * gpu_count
        startup_gpu_seconds_i = startup_sec * gpu_count
        ready_gpu_seconds_i = ready_sec * gpu_count
        total_gpu_seconds += lifetime_gpu_seconds
        startup_gpu_seconds += startup_gpu_seconds_i
        ready_gpu_seconds += ready_gpu_seconds_i
        lifetime_intervals.append((created_offset_s, removed_offset_s, float(gpu_count)))
        startup_intervals.append((created_offset_s, ready_offset_s, float(gpu_count)))
        ready_intervals.append((ready_offset_s, removed_offset_s, float(gpu_count)))
        item.update(
            {
                "created_offset_s": _round(created_offset_s, 6),
                "ready_offset_s": _round(ready_offset_s, 6),
                "removed_offset_s": _round(removed_offset_s, 6),
                "billing_scope": "deployment_lifecycle",
                "deployment_idle_tail_s": _round(safe_tail, 6),
                "serverless_idle_retention_s": _round(safe_retention, 6),
                "lifetime_sec": _round(lifetime_sec, 6),
                "startup_sec": _round(startup_sec, 6),
                "ready_sec": _round(ready_sec, 6),
                "lifetime_gpu_seconds": _round(lifetime_gpu_seconds, 6),
                "startup_gpu_seconds": _round(startup_gpu_seconds_i, 6),
                "ready_gpu_seconds": _round(ready_gpu_seconds_i, 6),
            }
        )
        normalized.append(item)

    if max_billing_gpus is not None and float(max_billing_gpus or 0.0) > 0.0:
        total_gpu_seconds = _capped_weighted_interval_seconds(lifetime_intervals, max_billing_gpus)
        startup_gpu_seconds = min(
            _capped_weighted_interval_seconds(startup_intervals, max_billing_gpus),
            total_gpu_seconds,
        )
        ready_gpu_seconds = min(
            _capped_weighted_interval_seconds(ready_intervals, max_billing_gpus),
            total_gpu_seconds,
        )

    infra_cost_total_usd = total_gpu_seconds * max(0.0, float(gpu_cost_per_second_usd or 0.0))
    per_request_denominator = int(completed_requests or 0) if int(completed_requests or 0) > 0 else max(0, int(total_requests or 0))
    infra_cost_per_request_usd = infra_cost_total_usd / per_request_denominator if per_request_denominator > 0 else 0.0
    avg_e2e_s = max(0.0, float(avg_e2e_ms or 0.0)) / 1000.0
    infra_ce_denom = infra_cost_per_request_usd * avg_e2e_s
    infra_ce = 1.0 / infra_ce_denom if infra_ce_denom > 1e-12 else 0.0
    return {
        "instance_lifecycle_log": normalized,
        "infra_gpu_seconds_total": total_gpu_seconds,
        "infra_startup_gpu_seconds": startup_gpu_seconds,
        "infra_ready_gpu_seconds": ready_gpu_seconds,
        "infra_cost_total_usd": infra_cost_total_usd,
        "infra_cost_per_request_usd": infra_cost_per_request_usd,
        "infra_ce": infra_ce,
        "infra_billing_elapsed_sec": billing_elapsed,
        "deployment_idle_tail_s": safe_tail,
        "serverless_idle_retention_s": safe_retention,
        "infra_max_billing_gpus": max(0.0, float(max_billing_gpus or 0.0)),
        "billing_scope": "deployment_lifecycle",
    }


def _union_interval_seconds(intervals: List[Tuple[float, float]]) -> float:
    finite = sorted(
        (float(start), float(end))
        for start, end in intervals
        if math.isfinite(float(start)) and math.isfinite(float(end)) and float(end) > float(start)
    )
    if not finite:
        return 0.0
    total = 0.0
    cur_start, cur_end = finite[0]
    for start, end in finite[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
            continue
        total += cur_end - cur_start
        cur_start, cur_end = start, end
    total += cur_end - cur_start
    return max(0.0, total)


def _max_concurrent_lifecycles(
    lifecycles: List[Dict[str, Any]],
    *,
    elapsed_sec: float,
    ready_only: bool,
    weight_key: Optional[str] = None,
) -> float:
    safe_elapsed = max(0.0, float(elapsed_sec or 0.0))
    events: List[Tuple[float, int, float]] = []
    for raw in lifecycles or []:
        item = dict(raw or {})
        start_key = "ready_offset_s" if ready_only else "created_offset_s"
        start = max(0.0, min(float(item.get(start_key, 0.0) or 0.0), safe_elapsed))
        end = max(start, min(float(item.get("removed_offset_s", safe_elapsed) or safe_elapsed), safe_elapsed))
        if end <= start:
            continue
        weight = float(item.get(weight_key, 1.0) or 1.0) if weight_key else 1.0
        events.append((start, 1, weight))
        events.append((end, -1, weight))
    current = 0.0
    peak = 0.0
    for _, kind, weight in sorted(events, key=lambda ev: (ev[0], ev[1])):
        current += kind * weight
        peak = max(peak, current)
    return max(0.0, peak)


def _summarize_resource_efficiency(
    *,
    results: List[Dict[str, Any]],
    lifecycles: List[Dict[str, Any]],
    elapsed_sec: float,
    ttft_slo_ms: float,
    static_startup_sec: float = 0.0,
    max_allocated_gpus: Optional[float] = None,
    max_allocated_replicas: Optional[float] = None,
    target_url_to_instance_id: Optional[Dict[str, str]] = None,
    default_instance_id: str = "__static_runtime_0",
) -> Dict[str, float]:
    safe_elapsed = max(0.0, float(elapsed_sec or 0.0))
    url_to_instance = dict(target_url_to_instance_id or {})
    gpu_count_by_instance = {
        str(item.get("instance_id") or ""): max(1, int(item.get("gpu_count", 1) or 1))
        for item in lifecycles or []
        if str(item.get("instance_id") or "")
    }
    intervals_by_instance: Dict[str, List[Tuple[float, float]]] = {}
    active_weighted_intervals: List[Tuple[float, float, float]] = []
    completed = 0
    output_tokens = 0
    slo_completed = 0
    slo_output_tokens = 0
    offset_shift = max(0.0, float(static_startup_sec or 0.0))
    for record in results or []:
        if not bool(record.get("success")):
            continue
        completed += 1
        tokens = max(0, int(record.get("completion_tokens", record.get("output_tokens", 0)) or 0))
        output_tokens += tokens
        try:
            ttft_ms = float(record.get("ttft_ms", record.get("overall_ttft_ms", 0.0)) or 0.0)
        except Exception:
            ttft_ms = 0.0
        if ttft_ms <= float(ttft_slo_ms or 5000.0):
            slo_completed += 1
            slo_output_tokens += tokens
        target_url = str(record.get("target_base_url") or "")
        server_metrics = dict(record.get("server_metrics") or {})
        instance_id = (
            url_to_instance.get(target_url)
            or str(server_metrics.get("instance_id") or "")
            or default_instance_id
        )
        try:
            end = float(record.get("completion_offset_s", 0.0) or 0.0) + offset_shift
        except Exception:
            continue
        try:
            service_e2e_ms = float(record.get("service_e2e_ms", record.get("e2e_ms", 0.0)) or 0.0)
        except Exception:
            service_e2e_ms = 0.0
        start = max(0.0, end - max(0.0, service_e2e_ms) / 1000.0)
        # The denominator lifecycles include startup for static serverful systems.
        # Clamp active serving to the same billing window.
        billing_elapsed = safe_elapsed + offset_shift
        start = max(0.0, min(start, billing_elapsed))
        end = max(start, min(end, billing_elapsed))
        if end > start:
            intervals_by_instance.setdefault(instance_id, []).append((start, end))
            active_weighted_intervals.append(
                (start, end, float(gpu_count_by_instance.get(instance_id, 1)))
            )

    active_gpu_seconds = 0.0
    for instance_id, intervals in intervals_by_instance.items():
        active_gpu_seconds += _union_interval_seconds(intervals) * gpu_count_by_instance.get(instance_id, 1)
    active_gpu_seconds = _capped_weighted_interval_seconds(
        active_weighted_intervals,
        max_allocated_gpus,
    )

    total_gpu_seconds = _capped_weighted_interval_seconds(
        [
            (
                float(item.get("created_offset_s", 0.0) or 0.0),
                float(item.get("removed_offset_s", 0.0) or 0.0),
                float(item.get("gpu_count", 1) or 1),
            )
            for item in lifecycles or []
        ],
        max_allocated_gpus,
    )
    startup_gpu_seconds = _capped_weighted_interval_seconds(
        [
            (
                float(item.get("created_offset_s", 0.0) or 0.0),
                float(item.get("ready_offset_s", 0.0) or 0.0),
                float(item.get("gpu_count", 1) or 1),
            )
            for item in lifecycles or []
        ],
        max_allocated_gpus,
    )
    ready_gpu_seconds = _capped_weighted_interval_seconds(
        [
            (
                float(item.get("ready_offset_s", 0.0) or 0.0),
                float(item.get("removed_offset_s", 0.0) or 0.0),
                float(item.get("gpu_count", 1) or 1),
            )
            for item in lifecycles or []
        ],
        max_allocated_gpus,
    )
    idle_ready_gpu_seconds = max(0.0, ready_gpu_seconds - active_gpu_seconds)
    billing_elapsed = safe_elapsed + offset_shift
    summary = {
        "infra_active_gpu_seconds": active_gpu_seconds,
        "infra_idle_ready_gpu_seconds": idle_ready_gpu_seconds,
        "infra_avg_allocated_gpus": total_gpu_seconds / billing_elapsed if billing_elapsed > 1e-12 else 0.0,
        "infra_avg_ready_gpus": ready_gpu_seconds / billing_elapsed if billing_elapsed > 1e-12 else 0.0,
        "infra_max_allocated_gpus": _max_concurrent_lifecycles(
            lifecycles,
            elapsed_sec=billing_elapsed,
            ready_only=False,
            weight_key="gpu_count",
        ),
        "infra_max_ready_gpus": _max_concurrent_lifecycles(
            lifecycles,
            elapsed_sec=billing_elapsed,
            ready_only=True,
            weight_key="gpu_count",
        ),
        "infra_avg_replicas": (
            sum(float(item.get("lifetime_sec", 0.0) or 0.0) for item in lifecycles or []) / billing_elapsed
            if billing_elapsed > 1e-12 else 0.0
        ),
        "infra_max_replicas": _max_concurrent_lifecycles(
            lifecycles,
            elapsed_sec=billing_elapsed,
            ready_only=False,
            weight_key=None,
        ),
        "infra_active_gpu_ratio": active_gpu_seconds / total_gpu_seconds if total_gpu_seconds > 1e-12 else 0.0,
        "infra_ready_active_gpu_ratio": active_gpu_seconds / ready_gpu_seconds if ready_gpu_seconds > 1e-12 else 0.0,
        "infra_idle_ready_gpu_ratio": idle_ready_gpu_seconds / total_gpu_seconds if total_gpu_seconds > 1e-12 else 0.0,
        "infra_startup_gpu_ratio": startup_gpu_seconds / total_gpu_seconds if total_gpu_seconds > 1e-12 else 0.0,
        "completed_requests_per_gpu_second": completed / total_gpu_seconds if total_gpu_seconds > 1e-12 else 0.0,
        "output_tokens_per_gpu_second": output_tokens / total_gpu_seconds if total_gpu_seconds > 1e-12 else 0.0,
        "goodput_requests_per_gpu_second": slo_completed / total_gpu_seconds if total_gpu_seconds > 1e-12 else 0.0,
        "goodput_tokens_per_gpu_second": slo_output_tokens / total_gpu_seconds if total_gpu_seconds > 1e-12 else 0.0,
    }
    if max_allocated_gpus is not None and float(max_allocated_gpus or 0.0) > 0.0:
        cap = max(0.0, float(max_allocated_gpus or 0.0))
        summary["infra_max_allocated_gpus"] = min(summary["infra_max_allocated_gpus"], cap)
        summary["infra_max_ready_gpus"] = min(summary["infra_max_ready_gpus"], cap)
    if max_allocated_replicas is not None and float(max_allocated_replicas or 0.0) > 0.0:
        replica_cap = max(0.0, float(max_allocated_replicas or 0.0))
        summary["infra_avg_replicas"] = min(summary["infra_avg_replicas"], replica_cap)
        summary["infra_max_replicas"] = min(summary["infra_max_replicas"], replica_cap)
    return summary


def _build_metric_structure() -> Dict[str, List[str]]:
    return {
        "standard_serving_metrics": [
            "TTFT_avg_ms",
            "TTFT_P50_ms",
            "TTFT_P95_ms",
            "TTFT_P99_ms",
            "TTFT_e2e_avg_ms",
            "TTFT_e2e_P50_ms",
            "TTFT_e2e_P95_ms",
            "TTFT_e2e_P99_ms",
            "TTFT_service_avg_ms",
            "TTFT_service_P50_ms",
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
            "Dispatch_admission_wait_P50_ms",
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
            "E2E_P50_ms",
            "E2E_P95_ms",
            "E2E_P99_ms",
            "E2E_e2e_avg_ms",
            "E2E_service_avg_ms",
            "E2E_service_P50_ms",
            "E2E_service_P95_ms",
            "E2E_service_P99_ms",
            "Monetary_cost_avg_usd",
            "Monetary_cost_total_usd",
            "Infra_GPU_seconds_total",
            "Infra_GPU_seconds_startup",
            "Infra_GPU_seconds_ready",
            "Infra_GPU_seconds_active",
            "Infra_GPU_seconds_idle_ready",
            "Infra_avg_allocated_GPUs",
            "Infra_avg_ready_GPUs",
            "Infra_max_allocated_GPUs",
            "Infra_max_ready_GPUs",
            "Infra_avg_replicas",
            "Infra_max_replicas",
            "Infra_active_GPU_ratio",
            "Infra_ready_active_GPU_ratio",
            "Infra_idle_ready_GPU_ratio",
            "Infra_startup_GPU_ratio",
            "Completed_requests_per_GPU_second",
            "Output_tokens_per_GPU_second",
            "Goodput_requests_per_GPU_second",
            "Goodput_tokens_per_GPU_second",
            "Infra_cost_total_usd",
            "Infra_cost_per_request_usd",
            "Monetary_cost_total_usd",
            "Monetary_cost_per_request_usd",
            "Monetary_equivalent_GPU_seconds",
            "Monetary_active_charge_GPU_seconds",
            "Monetary_idle_charge_GPU_seconds",
            "Serverless_idle_GPU_cost_factor",
            "Monetary_pricing_runtime_class",
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
            "Monetary_CE",
            "Infra_CE",
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
    ap.add_argument("--static-startup-sec", type=float, default=0.0)
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
    deploy = _load_structured_file(args.deploy.resolve()) if args.deploy else {}

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
    tpot = [
        float(r["tpot_ms"])
        for r in ok
        if r.get("tpot_ms") is not None and bool(r.get("tpot_observed", True))
    ]
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
    metrics_source_counts = Counter(str(r.get("metrics_source") or "missing") for r in ok)
    prompt_token_source_counts = Counter(
        str(r.get("prompt_token_source") or "missing")
        for r in ok
    )
    completion_token_source_counts = Counter(
        str(r.get("completion_token_source") or "missing")
        for r in ok
    )
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
    infra_gpu_cost_per_second_usd = _gpu_cost_per_second_usd(cost_model)
    static_startup_sec = max(0.0, float(args.static_startup_sec or 0.0))
    deployment_idle_tail_s = _cost_model_deployment_idle_tail_s(cost_model)
    serverless_idle_retention_s = _cost_model_serverless_idle_retention_s(cost_model)
    infra_billing_elapsed_sec = elapsed_sec
    backend_cfg = dict(deploy.get("backend_config", {}) or {})
    max_billing_gpus = 1.0
    max_billing_replicas = 1.0
    static_serverful_baseline = str(args.baseline_type) in ("sglang", "vllm", "slora")
    if static_serverful_baseline:
        tensor_parallel_size = int(deploy.get("tp", deploy.get("tensor_parallel_size", model_cfg.get("tensor_parallel_size", 1))) or 1)
        data_parallel_replicas = int(deploy.get("data_parallel_replicas", deploy.get("dp", 1)) or 1)
        runtime_gpu_count = _sglang_runtime_gpu_count(deploy, model_cfg)
        max_billing_gpus = float(max(1, runtime_gpu_count))
        max_billing_replicas = float(max(1, data_parallel_replicas))
        parallelism_topology = str(
            deploy.get("parallelism_topology")
            or (
                f"data_parallel_dp{data_parallel_replicas}_tp{tensor_parallel_size}"
                if data_parallel_replicas > 1
                else (
                    f"model_parallel_tp{tensor_parallel_size}"
                    if tensor_parallel_size > 1
                    else "single_gpu"
                )
            )
        )
        instance_lifecycle_log = _static_runtime_lifecycle(
            elapsed_sec=elapsed_sec,
            startup_sec=static_startup_sec,
            gpu_count=runtime_gpu_count,
            label=str(args.scenario_name),
            model_name=str(model_cfg.get("name", "")),
            replica_count=data_parallel_replicas,
            gpu_per_replica=tensor_parallel_size,
        )
        infra_billing_elapsed_sec = elapsed_sec + static_startup_sec
    else:
        tensor_parallel_size = int(backend_cfg.get("tensor_parallel_size", deploy.get("tensor_parallel_size", model_cfg.get("tensor_parallel_size", 1))) or 1)
        data_parallel_replicas = 1
        runtime_gpu_count = _serverlessllm_runtime_gpu_count(deploy, model_cfg)
        max_instances_cfg = int(((deploy.get("auto_scaling_config", {}) or {}).get("max_instances", 1)) or 1)
        max_billing_gpus = float(max(1, max_instances_cfg) * max(1, runtime_gpu_count))
        max_billing_replicas = float(max(1, max_instances_cfg))
        parallelism_topology = (
            f"model_parallel_tp{tensor_parallel_size}"
            if tensor_parallel_size > 1
            else "single_gpu"
        )
        instance_lifecycle_log = _reconstruct_serverless_instance_lifecycles(
            results=results,
            elapsed_sec=elapsed_sec,
            gpu_count=runtime_gpu_count,
            model_name=str(model_cfg.get("name", "")),
        )
    infra_summary = _summarize_infra_from_lifecycles(
        instance_lifecycle_log,
        elapsed_sec=infra_billing_elapsed_sec,
        completed_requests=len(ok),
        total_requests=total,
        avg_e2e_ms=avg_e2e_ms,
        gpu_cost_per_second_usd=infra_gpu_cost_per_second_usd,
        deployment_idle_tail_s=deployment_idle_tail_s,
        serverless_idle_retention_s=serverless_idle_retention_s,
        max_billing_gpus=max_billing_gpus,
    )
    target_url_to_instance_id = {}
    if static_serverful_baseline:
        base_urls = list(deploy.get("base_urls") or [])
        target_url_to_instance_id = {
            str(url): f"{args.scenario_name}_static_runtime_{idx}"
            for idx, url in enumerate(base_urls)
        }
    resource_summary = _summarize_resource_efficiency(
        results=results,
        lifecycles=infra_summary["instance_lifecycle_log"],
        elapsed_sec=elapsed_sec + deployment_idle_tail_s,
        ttft_slo_ms=ttft_slo_ms,
        static_startup_sec=static_startup_sec if static_serverful_baseline else 0.0,
        max_allocated_gpus=max_billing_gpus,
        max_allocated_replicas=max_billing_replicas,
        target_url_to_instance_id=target_url_to_instance_id,
        default_instance_id=f"{args.scenario_name}_static_runtime_0",
    )
    monetary_summary = _summarize_monetary_cost_from_resource_seconds(
        cost_model=cost_model,
        completed_requests=len(ok),
        total_requests=total,
        avg_e2e_ms=avg_e2e_ms,
        lifecycles=infra_summary["instance_lifecycle_log"],
        infra_gpu_seconds_total=infra_summary["infra_gpu_seconds_total"],
        infra_startup_gpu_seconds=infra_summary["infra_startup_gpu_seconds"],
        infra_active_gpu_seconds=resource_summary["infra_active_gpu_seconds"],
        infra_idle_ready_gpu_seconds=resource_summary["infra_idle_ready_gpu_seconds"],
    )

    token_avg_cost_usd = avg_cost_usd
    token_total_cost_usd = total_cost_usd
    token_ce = ce
    avg_cost_usd = monetary_summary["monetary_cost_per_request_usd"]
    total_cost_usd = monetary_summary["monetary_cost_total_usd"]
    ce = monetary_summary["monetary_ce"]
    total_prompt_tokens = sum(prompt_tokens)
    total_completion_tokens = sum(completion_tokens)
    total_served_tokens = total_prompt_tokens + total_completion_tokens
    cost_per_1m_total_tokens_usd = _cost_per_million_tokens(total_cost_usd, total_served_tokens)
    cost_per_1m_output_tokens_usd = _cost_per_million_tokens(total_cost_usd, total_completion_tokens)
    token_proxy_cost_per_1m_total_tokens_usd = _cost_per_million_tokens(
        token_total_cost_usd,
        total_served_tokens,
    )
    token_proxy_cost_per_1m_output_tokens_usd = _cost_per_million_tokens(
        token_total_cost_usd,
        total_completion_tokens,
    )

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
        "primary_ttft_definition": METRIC_DEF_PRIMARY_TTFT,
        "primary_e2e_definition": METRIC_DEF_PRIMARY_E2E,
        "service_ttft_definition": METRIC_DEF_SERVICE_TTFT,
        "service_e2e_definition": METRIC_DEF_SERVICE_E2E,
        "slo_metric": "TTFT_e2e",
        "avg_ttft_ms": _round(avg_ttft_ms),
        "p95_ttft_ms": _round(_pct(ttft, 95)),
        "p99_ttft_ms": _round(_pct(ttft, 99)),
        "avg_overall_ttft_ms": _round(avg_ttft_ms),
        "p50_overall_ttft_ms": _round(_pct(ttft, 50)),
        "p95_overall_ttft_ms": _round(_pct(ttft, 95)),
        "p99_overall_ttft_ms": _round(_pct(ttft, 99)),
        "avg_service_ttft_ms": _round(sum(service_ttft) / len(service_ttft) if service_ttft else None),
        "p50_service_ttft_ms": _round(_pct(service_ttft, 50)),
        "p95_service_ttft_ms": _round(_pct(service_ttft, 95)),
        "p99_service_ttft_ms": _round(_pct(service_ttft, 99)),
        "avg_tpot_ms": _round(avg_tpot_ms),
        "avg_e2e_ms": _round(avg_e2e_ms),
        "p95_e2e_ms": _round(_pct(e2e, 95)),
        "p99_e2e_ms": _round(_pct(e2e, 99)),
        "avg_overall_e2e_ms": _round(avg_e2e_ms),
        "p50_overall_e2e_ms": _round(_pct(e2e, 50)),
        "p95_overall_e2e_ms": _round(_pct(e2e, 95)),
        "p99_overall_e2e_ms": _round(_pct(e2e, 99)),
        "avg_service_e2e_ms": _round(sum(service_e2e) / len(service_e2e) if service_e2e else None),
        "p50_service_e2e_ms": _round(_pct(service_e2e, 50)),
        "p95_service_e2e_ms": _round(_pct(service_e2e, 95)),
        "p99_service_e2e_ms": _round(_pct(service_e2e, 99)),
        "throughput_rps": _round(throughput_rps, 6),
        "throughput_tok_per_s": _round(throughput_tokps, 6),
        "slo_attainment": _round(slo_attainment, 6),
        "ttft_slo_ms": _round(ttft_slo_ms),
        "avg_cost_usd": _round(avg_cost_usd, 8),
        "total_cost_usd": _round(total_cost_usd, 8),
        "total_input_tokens": int(total_prompt_tokens),
        "total_output_tokens": int(total_completion_tokens),
        "total_tokens": int(total_served_tokens),
        "cost_per_1m_total_tokens_usd": _round(cost_per_1m_total_tokens_usd, 8),
        "cost_per_1m_output_tokens_usd": _round(cost_per_1m_output_tokens_usd, 8),
        "token_avg_cost_usd": _round(token_avg_cost_usd, 8),
        "token_total_cost_usd": _round(token_total_cost_usd, 8),
        "token_proxy_cost_per_1m_total_tokens_usd": _round(token_proxy_cost_per_1m_total_tokens_usd, 8),
        "token_proxy_cost_per_1m_output_tokens_usd": _round(token_proxy_cost_per_1m_output_tokens_usd, 8),
        "token_ce": _round(token_ce, 6),
        "max_model_len": model_cfg.get("max_model_len"),
        "max_input_len": model_cfg.get("max_input_len"),
        "max_output_tokens_cap": model_cfg.get("max_output_tokens_cap"),
        "max_num_seqs": model_cfg.get("max_num_seqs"),
        "runtime_concurrency_cap": model_cfg.get("runtime_concurrency_cap"),
        "max_loras": model_cfg.get("max_loras"),
        "avg_prompt_tokens": _round((sum(prompt_tokens) / len(prompt_tokens)) if prompt_tokens else None),
        "p95_prompt_tokens": _round(_pct(prompt_tokens, 95)),
        "max_prompt_tokens": max(prompt_tokens) if prompt_tokens else None,
        "avg_completion_tokens": _round((sum(completion_tokens) / len(completion_tokens)) if completion_tokens else None),
        "p95_completion_tokens": _round(_pct(completion_tokens, 95)),
        "p99_completion_tokens": _round(_pct(completion_tokens, 99)),
        "max_completion_tokens": max(completion_tokens) if completion_tokens else None,
        "avg_total_tokens": _round((sum(total_tokens) / len(total_tokens)) if total_tokens else None),
        "max_total_tokens": max(total_tokens) if total_tokens else None,
        "metrics_source_counts": dict(sorted(metrics_source_counts.items())),
        "prompt_token_source_counts": dict(sorted(prompt_token_source_counts.items())),
        "completion_token_source_counts": dict(sorted(completion_token_source_counts.items())),
        "infra_gpu_seconds_total": _round(infra_summary["infra_gpu_seconds_total"], 6),
        "infra_startup_gpu_seconds": _round(infra_summary["infra_startup_gpu_seconds"], 6),
        "infra_ready_gpu_seconds": _round(infra_summary["infra_ready_gpu_seconds"], 6),
        "infra_active_gpu_seconds": _round(resource_summary["infra_active_gpu_seconds"], 6),
        "infra_idle_ready_gpu_seconds": _round(resource_summary["infra_idle_ready_gpu_seconds"], 6),
        "infra_avg_allocated_gpus": _round(resource_summary["infra_avg_allocated_gpus"], 6),
        "infra_avg_ready_gpus": _round(resource_summary["infra_avg_ready_gpus"], 6),
        "infra_max_allocated_gpus": _round(resource_summary["infra_max_allocated_gpus"], 6),
        "infra_max_ready_gpus": _round(resource_summary["infra_max_ready_gpus"], 6),
        "infra_avg_replicas": _round(resource_summary["infra_avg_replicas"], 6),
        "infra_max_replicas": _round(resource_summary["infra_max_replicas"], 6),
        "infra_active_gpu_ratio": _round(resource_summary["infra_active_gpu_ratio"], 6),
        "infra_ready_active_gpu_ratio": _round(resource_summary["infra_ready_active_gpu_ratio"], 6),
        "infra_idle_ready_gpu_ratio": _round(resource_summary["infra_idle_ready_gpu_ratio"], 6),
        "infra_startup_gpu_ratio": _round(resource_summary["infra_startup_gpu_ratio"], 6),
        "completed_requests_per_gpu_second": _round(resource_summary["completed_requests_per_gpu_second"], 6),
        "output_tokens_per_gpu_second": _round(resource_summary["output_tokens_per_gpu_second"], 6),
        "goodput_requests_per_gpu_second": _round(resource_summary["goodput_requests_per_gpu_second"], 6),
        "goodput_tokens_per_gpu_second": _round(resource_summary["goodput_tokens_per_gpu_second"], 6),
        "infra_cost_total_usd": _round(infra_summary["infra_cost_total_usd"], 8),
        "infra_cost_per_request_usd": _round(infra_summary["infra_cost_per_request_usd"], 8),
        "infra_ce": _round(infra_summary["infra_ce"], 6),
        "monetary_cost_total_usd": _round(monetary_summary["monetary_cost_total_usd"], 8),
        "monetary_cost_per_request_usd": _round(monetary_summary["monetary_cost_per_request_usd"], 8),
        "monetary_ce": _round(monetary_summary["monetary_ce"], 6),
        "monetary_equivalent_gpu_seconds": _round(monetary_summary["monetary_equivalent_gpu_seconds"], 6),
        "monetary_active_charge_gpu_seconds": _round(monetary_summary["monetary_active_charge_gpu_seconds"], 6),
        "monetary_idle_charge_gpu_seconds": _round(monetary_summary["monetary_idle_charge_gpu_seconds"], 6),
        "serverless_idle_gpu_cost_factor": _round(monetary_summary["serverless_idle_gpu_cost_factor"], 6),
        "serverless_invocation_cost_total_usd": _round(monetary_summary["serverless_invocation_cost_total_usd"], 8),
        "serverless_invocation_cost_per_request_usd": _round(monetary_summary["serverless_invocation_cost_per_request_usd"], 10),
        "monetary_pricing_runtime_class": monetary_summary["monetary_pricing_runtime_class"],
        "infra_billing_elapsed_sec": _round(infra_summary["infra_billing_elapsed_sec"], 6),
        "infra_max_billing_gpus": _round(infra_summary["infra_max_billing_gpus"], 6),
        "deployment_idle_tail_s": _round(infra_summary["deployment_idle_tail_s"], 6),
        "serverless_idle_retention_s": _round(infra_summary["serverless_idle_retention_s"], 6),
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
        "p50_dispatch_admission_wait_ms": _round(_pct(dispatch_wait, 50)),
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
            "TTFT_P50_ms": _round(_pct(ttft, 50)),
            "TTFT_P95_ms": _round(_pct(ttft, 95)),
            "TTFT_P99_ms": _round(_pct(ttft, 99)),
            "TTFT_e2e_avg_ms": _round(avg_ttft_ms),
            "TTFT_e2e_P50_ms": _round(_pct(ttft, 50)),
            "TTFT_e2e_P95_ms": _round(_pct(ttft, 95)),
            "TTFT_e2e_P99_ms": _round(_pct(ttft, 99)),
            "TTFT_service_avg_ms": _round(sum(service_ttft) / len(service_ttft) if service_ttft else None),
            "TTFT_service_P50_ms": _round(_pct(service_ttft, 50)),
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
            "Dispatch_admission_wait_P50_ms": _round(_pct(dispatch_wait, 50)),
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
            "E2E_P50_ms": _round(_pct(e2e, 50)),
            "E2E_P95_ms": _round(_pct(e2e, 95)),
            "E2E_P99_ms": _round(_pct(e2e, 99)),
            "E2E_e2e_avg_ms": _round(avg_e2e_ms),
            "E2E_service_avg_ms": _round(sum(service_e2e) / len(service_e2e) if service_e2e else None),
            "E2E_service_P50_ms": _round(_pct(service_e2e, 50)),
            "E2E_service_P95_ms": _round(_pct(service_e2e, 95)),
            "E2E_service_P99_ms": _round(_pct(service_e2e, 99)),
            "Monetary_cost_avg_usd": _round(avg_cost_usd, 8),
            "Monetary_cost_total_usd": _round(total_cost_usd, 8),
            "Monetary_cost_per_1M_total_tokens_usd": _round(cost_per_1m_total_tokens_usd, 8),
            "Monetary_cost_per_1M_output_tokens_usd": _round(cost_per_1m_output_tokens_usd, 8),
            "Token_proxy_cost_avg_usd": _round(token_avg_cost_usd, 8),
            "Token_proxy_cost_total_usd": _round(token_total_cost_usd, 8),
            "Token_proxy_cost_per_1M_total_tokens_usd": _round(token_proxy_cost_per_1m_total_tokens_usd, 8),
            "Token_proxy_cost_per_1M_output_tokens_usd": _round(token_proxy_cost_per_1m_output_tokens_usd, 8),
            "Token_proxy_CE": _round(token_ce, 6),
            "Infra_GPU_seconds_total": _round(infra_summary["infra_gpu_seconds_total"], 6),
            "Infra_GPU_seconds_startup": _round(infra_summary["infra_startup_gpu_seconds"], 6),
            "Infra_GPU_seconds_ready": _round(infra_summary["infra_ready_gpu_seconds"], 6),
            "Infra_GPU_seconds_active": _round(resource_summary["infra_active_gpu_seconds"], 6),
            "Infra_GPU_seconds_idle_ready": _round(resource_summary["infra_idle_ready_gpu_seconds"], 6),
            "Infra_avg_allocated_GPUs": _round(resource_summary["infra_avg_allocated_gpus"], 6),
            "Infra_avg_ready_GPUs": _round(resource_summary["infra_avg_ready_gpus"], 6),
            "Infra_max_allocated_GPUs": _round(resource_summary["infra_max_allocated_gpus"], 6),
            "Infra_max_ready_GPUs": _round(resource_summary["infra_max_ready_gpus"], 6),
            "Infra_avg_replicas": _round(resource_summary["infra_avg_replicas"], 6),
            "Infra_max_replicas": _round(resource_summary["infra_max_replicas"], 6),
            "Infra_active_GPU_ratio": _round(resource_summary["infra_active_gpu_ratio"], 6),
            "Infra_ready_active_GPU_ratio": _round(resource_summary["infra_ready_active_gpu_ratio"], 6),
            "Infra_idle_ready_GPU_ratio": _round(resource_summary["infra_idle_ready_gpu_ratio"], 6),
            "Infra_startup_GPU_ratio": _round(resource_summary["infra_startup_gpu_ratio"], 6),
            "Completed_requests_per_GPU_second": _round(resource_summary["completed_requests_per_gpu_second"], 6),
            "Output_tokens_per_GPU_second": _round(resource_summary["output_tokens_per_gpu_second"], 6),
            "Goodput_requests_per_GPU_second": _round(resource_summary["goodput_requests_per_gpu_second"], 6),
            "Goodput_tokens_per_GPU_second": _round(resource_summary["goodput_tokens_per_gpu_second"], 6),
            "Infra_cost_total_usd": _round(infra_summary["infra_cost_total_usd"], 8),
            "Infra_cost_per_request_usd": _round(infra_summary["infra_cost_per_request_usd"], 8),
            "Monetary_cost_total_usd": _round(monetary_summary["monetary_cost_total_usd"], 8),
            "Monetary_cost_per_request_usd": _round(monetary_summary["monetary_cost_per_request_usd"], 8),
            "Monetary_equivalent_GPU_seconds": _round(monetary_summary["monetary_equivalent_gpu_seconds"], 6),
            "Monetary_active_charge_GPU_seconds": _round(monetary_summary["monetary_active_charge_gpu_seconds"], 6),
            "Monetary_idle_charge_GPU_seconds": _round(monetary_summary["monetary_idle_charge_gpu_seconds"], 6),
            "Serverless_idle_GPU_cost_factor": _round(monetary_summary["serverless_idle_gpu_cost_factor"], 6),
            "Monetary_pricing_runtime_class": monetary_summary["monetary_pricing_runtime_class"],
            "Infra_billing_elapsed_sec": _round(infra_summary["infra_billing_elapsed_sec"], 6),
            "Infra_max_billing_GPUs": _round(infra_summary["infra_max_billing_gpus"], 6),
            "Deployment_idle_tail_s": _round(infra_summary["deployment_idle_tail_s"], 6),
            "Serverless_idle_retention_s": _round(infra_summary["serverless_idle_retention_s"], 6),
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
            "Monetary_CE": _round(monetary_summary["monetary_ce"]),
            "Infra_CE": _round(infra_summary["infra_ce"]),
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
        "TTFT_e2e_P50_ms": _cell(_pct(ttft, 50)),
        "TTFT_e2e_P95_ms": _cell(_pct(ttft, 95)),
        "TTFT_e2e_P99_ms": _cell(_pct(ttft, 99)),
        "TTFT_service_avg_ms": _cell(sum(service_ttft) / len(service_ttft) if service_ttft else None),
        "TTFT_service_P50_ms": _cell(_pct(service_ttft, 50)),
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
        "Dispatch_admission_wait_P50_ms": _cell(_pct(dispatch_wait, 50)),
        "Dispatch_admission_wait_P95_ms": _cell(_pct(dispatch_wait, 95)),
        "TPOT_avg_ms": _cell(avg_tpot_ms),
        "TPOT_observed_ratio": _cell(tpot_observed_ratio, 4),
        "E2E_avg_ms": _cell(avg_e2e_ms),
        "E2E_P50_ms": _cell(_pct(e2e, 50)),
        "E2E_P95_ms": _cell(_pct(e2e, 95)),
        "E2E_P99_ms": _cell(_pct(e2e, 99)),
        "E2E_e2e_avg_ms": _cell(avg_e2e_ms),
        "E2E_service_avg_ms": _cell(sum(service_e2e) / len(service_e2e) if service_e2e else None),
        "E2E_service_P50_ms": _cell(_pct(service_e2e, 50)),
        "E2E_service_P95_ms": _cell(_pct(service_e2e, 95)),
        "E2E_service_P99_ms": _cell(_pct(service_e2e, 99)),
        "throughput_RPS": _cell(throughput_rps, 3),
        "throughput_TOKPS": _cell(throughput_tokps, 3),
        "SLO_attainment": _cell(slo_attainment, 4),
        "SLO_goodput_RPS": _cell(throughput_rps * slo_attainment, 4),
        "SLO_goodput_TOKPS": _cell(throughput_tokps * slo_attainment, 4),
        "TTFT_SLO_ms": _cell(ttft_slo_ms, 1),
        "avg_cost_USD": _cell(avg_cost_usd, 7),
        "total_cost_USD": _cell(total_cost_usd, 5),
        "cost_per_1m_total_tokens_usd": _cell(cost_per_1m_total_tokens_usd, 8),
        "cost_per_1m_output_tokens_usd": _cell(cost_per_1m_output_tokens_usd, 8),
        "token_avg_cost_usd": _cell(token_avg_cost_usd, 8),
        "token_total_cost_usd": _cell(token_total_cost_usd, 8),
        "token_proxy_cost_per_1m_total_tokens_usd": _cell(token_proxy_cost_per_1m_total_tokens_usd, 8),
        "token_proxy_cost_per_1m_output_tokens_usd": _cell(token_proxy_cost_per_1m_output_tokens_usd, 8),
        "token_ce": _cell(token_ce, 6),
        "infra_gpu_seconds_total": _cell(infra_summary["infra_gpu_seconds_total"], 6),
        "infra_startup_gpu_seconds": _cell(infra_summary["infra_startup_gpu_seconds"], 6),
        "infra_ready_gpu_seconds": _cell(infra_summary["infra_ready_gpu_seconds"], 6),
        "infra_active_gpu_seconds": _cell(resource_summary["infra_active_gpu_seconds"], 6),
        "infra_idle_ready_gpu_seconds": _cell(resource_summary["infra_idle_ready_gpu_seconds"], 6),
        "infra_avg_allocated_gpus": _cell(resource_summary["infra_avg_allocated_gpus"], 6),
        "infra_avg_ready_gpus": _cell(resource_summary["infra_avg_ready_gpus"], 6),
        "infra_max_allocated_gpus": _cell(resource_summary["infra_max_allocated_gpus"], 6),
        "infra_max_ready_gpus": _cell(resource_summary["infra_max_ready_gpus"], 6),
        "infra_avg_replicas": _cell(resource_summary["infra_avg_replicas"], 6),
        "infra_max_replicas": _cell(resource_summary["infra_max_replicas"], 6),
        "infra_active_gpu_ratio": _cell(resource_summary["infra_active_gpu_ratio"], 6),
        "infra_ready_active_gpu_ratio": _cell(resource_summary["infra_ready_active_gpu_ratio"], 6),
        "infra_idle_ready_gpu_ratio": _cell(resource_summary["infra_idle_ready_gpu_ratio"], 6),
        "infra_startup_gpu_ratio": _cell(resource_summary["infra_startup_gpu_ratio"], 6),
        "completed_requests_per_gpu_second": _cell(resource_summary["completed_requests_per_gpu_second"], 6),
        "output_tokens_per_gpu_second": _cell(resource_summary["output_tokens_per_gpu_second"], 6),
        "goodput_requests_per_gpu_second": _cell(resource_summary["goodput_requests_per_gpu_second"], 6),
        "goodput_tokens_per_gpu_second": _cell(resource_summary["goodput_tokens_per_gpu_second"], 6),
        "infra_cost_total_usd": _cell(infra_summary["infra_cost_total_usd"], 7),
        "infra_cost_per_request_usd": _cell(infra_summary["infra_cost_per_request_usd"], 7),
        "monetary_cost_total_usd": _cell(monetary_summary["monetary_cost_total_usd"], 7),
        "monetary_cost_per_request_usd": _cell(monetary_summary["monetary_cost_per_request_usd"], 7),
        "monetary_ce": _cell(monetary_summary["monetary_ce"], 4),
        "monetary_equivalent_gpu_seconds": _cell(monetary_summary["monetary_equivalent_gpu_seconds"], 6),
        "monetary_active_charge_gpu_seconds": _cell(monetary_summary["monetary_active_charge_gpu_seconds"], 6),
        "monetary_idle_charge_gpu_seconds": _cell(monetary_summary["monetary_idle_charge_gpu_seconds"], 6),
        "serverless_idle_gpu_cost_factor": _cell(monetary_summary["serverless_idle_gpu_cost_factor"], 6),
        "serverless_invocation_cost_total_usd": _cell(monetary_summary["serverless_invocation_cost_total_usd"], 8),
        "serverless_invocation_cost_per_request_usd": _cell(monetary_summary["serverless_invocation_cost_per_request_usd"], 10),
        "monetary_pricing_runtime_class": monetary_summary["monetary_pricing_runtime_class"],
        "infra_billing_elapsed_sec": _cell(infra_summary["infra_billing_elapsed_sec"], 6),
        "infra_max_billing_gpus": _cell(infra_summary["infra_max_billing_gpus"], 6),
        "deployment_idle_tail_s": _cell(infra_summary["deployment_idle_tail_s"], 6),
        "serverless_idle_retention_s": _cell(infra_summary["serverless_idle_retention_s"], 6),
        "CE": _cell(ce, 4),
        "Monetary_CE": _cell(monetary_summary["monetary_ce"], 4),
        "Infra_CE": _cell(infra_summary["infra_ce"], 4),
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
            "instance_lifecycle_log": infra_summary["instance_lifecycle_log"],
        }
    }

    output = {
        "schema_version": 4,
        "metric_schema_version": "e2e_v3",
        "metric_definitions": {
            "primary_ttft": METRIC_DEF_PRIMARY_TTFT,
            "primary_e2e": METRIC_DEF_PRIMARY_E2E,
            "service_ttft": METRIC_DEF_SERVICE_TTFT,
            "service_e2e": METRIC_DEF_SERVICE_E2E,
            "tpot": "(service_completion_time - service_first_token_time) / max(output_tokens - 1, 1)",
            "infra_cost_model": "flat deployment lifecycle GPU-second audit; every allocated GPU-second has the same price, independent of active vs idle state",
            "monetary_cost_model": "main cloud monetary billing; serverful runtimes pay full-price lifecycle GPU-seconds, while serverless runtimes pay full-price startup/active GPU-seconds plus discounted ready-but-idle GPU-seconds",
            "primary_cost": "workload-level monetary cost normalized as Cost/req = total monetary cost / completed requests",
            "primary_ce": "1 / (avg E2E_e2e in seconds * Cost/req); this is the main paper-facing cost-effectiveness metric",
            "cost_per_1m_total_tokens": "workload-level lifecycle monetary cost divided by total served tokens (input + output), scaled to one million tokens",
            "token_proxy_cost": "legacy request/token pricing proxy derived from base + per-input-token + per-output-token charges; retained only as a diagnostic view and never used as the headline cost",
            "infra_resource_efficiency": "active/idle GPU-seconds and replica counts derived from runtime lifecycles plus request service intervals; goodput per GPU-second uses TTFT-SLO-satisfied requests/tokens divided by allocated GPU-seconds",
            "slo_metric": "TTFT_e2e",
        },
        "metadata": {
            "system": str(args.system_name),
            "metric_schema_version": "e2e_v3",
            "primary_ttft_definition": METRIC_DEF_PRIMARY_TTFT,
            "primary_e2e_definition": METRIC_DEF_PRIMARY_E2E,
            "service_ttft_definition": METRIC_DEF_SERVICE_TTFT,
            "service_e2e_definition": METRIC_DEF_SERVICE_E2E,
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
            "max_model_len": model_cfg.get("max_model_len"),
            "max_input_len": model_cfg.get("max_input_len"),
            "max_output_tokens_cap": model_cfg.get("max_output_tokens_cap"),
            "max_num_seqs": model_cfg.get("max_num_seqs"),
            "runtime_concurrency_cap": model_cfg.get("runtime_concurrency_cap"),
            "max_loras": model_cfg.get("max_loras"),
            "tensor_parallel_size": tensor_parallel_size,
            "data_parallel_replicas": data_parallel_replicas,
            "gpu_per_request": max(1, tensor_parallel_size),
            "runtime_gpu_count": runtime_gpu_count,
            "parallelism_topology": parallelism_topology,
            "infra_cost_model": {
                "name": "simulated_gpu_second_deployment_lifecycle_v3_capped",
                "gpu_cost_per_second_usd": round(infra_gpu_cost_per_second_usd, 8),
                "gpu_hour_cost_usd_equivalent": round(infra_gpu_cost_per_second_usd * 3600.0, 6),
                "scope": "deployment_lifecycle_with_idle_tail",
                "billing_unit": "gpu_second",
                "static_startup_sec": round(static_startup_sec, 6),
                "deployment_idle_tail_s": round(deployment_idle_tail_s, 6),
                "serverless_idle_retention_s": round(serverless_idle_retention_s, 6),
                "infra_billing_elapsed_sec": round(infra_summary["infra_billing_elapsed_sec"], 6),
                "infra_max_billing_gpus": round(infra_summary["infra_max_billing_gpus"], 6),
            },
            "monetary_cost_model": {
                "name": "serverless_active_idle_differential_billing_v1",
                "gpu_cost_per_second_usd": round(infra_gpu_cost_per_second_usd, 8),
                "serverless_idle_gpu_cost_factor": round(_serverless_idle_gpu_cost_factor(cost_model), 10),
                "serverless_invocation_cost_per_request_usd": round(
                    _serverless_invocation_cost_per_request_usd(cost_model),
                    10,
                ),
                "serverful_policy": "full-price lifecycle GPU-seconds",
                "serverless_policy": "startup/active GPU-seconds at full price; ready-but-idle GPU-seconds at idle factor",
            },
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
        f"Cost/req=${avg_cost_usd:.6f} CE={ce:.3f} Cost/1MTok=${cost_per_1m_total_tokens_usd:.2f} "
        f"TokenProxy/req=${token_avg_cost_usd:.6f} TokenProxyCE={token_ce:.3f} "
        f"InfraCost/req=${infra_summary['infra_cost_per_request_usd']:.6f} "
        f"InfraCE={infra_summary['infra_ce']:.3f} "
        f"MonetaryEqGPU={monetary_summary['monetary_equivalent_gpu_seconds']:.2f}s "
        f"IdleFactor={monetary_summary['serverless_idle_gpu_cost_factor']:.3f} "
        f"PricingClass={monetary_summary['monetary_pricing_runtime_class']} "
        f"ActiveGPU={resource_summary['infra_active_gpu_seconds']:.2f}s "
        f"IdleReadyGPU={resource_summary['infra_idle_ready_gpu_seconds']:.2f}s "
        f"GoodTok/GPU-s={resource_summary['goodput_tokens_per_gpu_second']:.2f}"
    )
    print(f"wrote summary -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
