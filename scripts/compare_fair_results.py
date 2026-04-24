#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STRICT_KEY_METRICS = [
    ("TTFT_e2e_ms", "avg_overall_ttft_ms"),
    ("E2E_e2e_ms", "avg_overall_e2e_ms"),
    ("TPOT_ms", "avg_tpot_ms"),
    ("Tok/s", "throughput_tok_per_s"),
    ("RPS", "throughput_rps"),
    ("SLO", "slo_attainment"),
    ("Cost_req_usd", "avg_cost_usd"),
    ("Cost_1MTok_usd", "cost_per_1m_total_tokens_usd"),
    ("CE", "ce"),
    ("MonetaryCost_req_usd", "monetary_cost_per_request_usd"),
    ("MonetaryCE", "monetary_ce"),
    ("TokenProxy_req_usd", "token_avg_cost_usd"),
    ("TokenProxy_1MTok_usd", "token_proxy_cost_per_1m_total_tokens_usd"),
    ("TokenProxyCE", "token_ce"),
    ("InfraCost_req_usd", "infra_cost_per_request_usd"),
    ("InfraCE", "infra_ce"),
]

DIAGNOSTIC_KEY_METRICS = [
    ("TTFT_service_ms", "avg_service_ttft_ms"),
    ("E2E_service_ms", "avg_service_e2e_ms"),
    ("DispatchWait_ms", "avg_dispatch_admission_wait_ms"),
    ("TTFT_comp_ms", "avg_comparable_ttft_ms"),
    ("TTFT_warm_ms", "avg_warm_standard_ttft_ms"),
    ("ColdStart_ms", "avg_cold_start_latency_ms"),
    ("ScaleUpAffected_ms", "avg_scaleup_affected_ttft_ms"),
    ("ScaleUpFirstService_ms", "avg_scaleup_first_service_ttft_ms"),
    ("InfraGPU_s", "infra_gpu_seconds_total"),
    ("InfraStartupGPU_s", "infra_startup_gpu_seconds"),
    ("InfraReadyGPU_s", "infra_ready_gpu_seconds"),
    ("InfraBilling_s", "infra_billing_elapsed_sec"),
    ("InfraMaxBillingGPU", "infra_max_billing_gpus"),
]

RESOURCE_KEY_METRICS = [
    ("ActiveGPU_s", "infra_active_gpu_seconds"),
    ("IdleReadyGPU_s", "infra_idle_ready_gpu_seconds"),
    ("AvgRep", "infra_avg_replicas"),
    ("MaxRep", "infra_max_replicas"),
    ("AvgGPU", "infra_avg_allocated_gpus"),
    ("MaxGPU", "infra_max_allocated_gpus"),
    ("ActiveGPU%", "infra_active_gpu_ratio"),
    ("IdleReadyGPU%", "infra_idle_ready_gpu_ratio"),
    ("Req/GPU-s", "completed_requests_per_gpu_second"),
    ("Tok/GPU-s", "output_tokens_per_gpu_second"),
    ("GoodReq/GPU-s", "goodput_requests_per_gpu_second"),
    ("GoodTok/GPU-s", "goodput_tokens_per_gpu_second"),
    ("MonetaryEqGPU_s", "monetary_equivalent_gpu_seconds"),
    ("MonetaryActiveChargeGPU_s", "monetary_active_charge_gpu_seconds"),
    ("MonetaryIdleChargeGPU_s", "monetary_idle_charge_gpu_seconds"),
    ("IdleGPUCostFactor", "serverless_idle_gpu_cost_factor"),
]

DEFAULT_DEADLINE_MS = 300_000.0
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


def _metric_value(summary: Dict[str, Any], key: str) -> Any:
    if key == "ce":
        for candidate in ("ce", "qpr", "cost_effectiveness_e2e"):
            if summary.get(candidate) is not None:
                return summary.get(candidate)
        return None
    if key == "avg_overall_ttft_ms":
        return summary.get("avg_overall_ttft_ms")
    if key == "avg_overall_e2e_ms":
        return summary.get("avg_overall_e2e_ms")
    if key == "infra_ce":
        return summary.get("infra_ce")
    return summary.get(key)


ENVELOPE_KEY_METRICS = [
    ("MaxModelLen", "max_model_len"),
    ("MaxInputLen", "max_input_len"),
    ("MaxOutputCap", "max_output_tokens_cap"),
    ("RuntimeCap", "runtime_concurrency_cap"),
    ("MaxSeqs", "max_num_seqs"),
    ("MaxLoras", "max_loras"),
]

TOPOLOGY_KEY_METRICS = [
    ("TP", "tensor_parallel_size"),
    ("GPU/Req", "gpu_per_request"),
    ("RuntimeGPUs", "runtime_gpu_count"),
    ("VisibleGPUs", "visible_gpu_count"),
    ("Topology", "parallelism_topology"),
]


def _as_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_deploy_config(path_value: Any) -> Dict[str, Any]:
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore

            parsed = yaml.safe_load(text)
        except Exception:
            parsed = {}
    return dict(parsed or {}) if isinstance(parsed, dict) else {}


def _visible_gpu_count(value: Any) -> Optional[int]:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, str):
        return len([item for item in value.split(",") if item.strip()])
    return _as_int(value)


def _attach_topology_fields(summary: Dict[str, Any]) -> None:
    deploy = _load_deploy_config(summary.get("deploy_config"))
    backend_cfg = dict(deploy.get("backend_config", {}) or {})
    tp = (
        _as_int(summary.get("tensor_parallel_size"))
        or _as_int(deploy.get("tp"))
        or _as_int(deploy.get("tensor_parallel_size"))
        or _as_int(backend_cfg.get("tensor_parallel_size"))
        or 1
    )
    visible_gpu_count = (
        _visible_gpu_count(summary.get("visible_device_ids"))
        or _visible_gpu_count(deploy.get("gpu_ids"))
        or _visible_gpu_count(deploy.get("visible_device_ids"))
    )
    runtime_gpu_count = (
        _as_int(summary.get("runtime_gpu_count"))
        or _as_int(deploy.get("num_gpus"))
        or _as_int(backend_cfg.get("num_gpus"))
        or tp
    )
    gpu_per_request = max(1, tp)
    deploy_topology = deploy.get("parallelism_topology")
    if deploy_topology:
        topology = str(deploy_topology)
    elif tp > 1:
        topology = f"model_parallel_tp{tp}"
    elif visible_gpu_count and visible_gpu_count > 1:
        topology = "single_gpu_scaleout"
    else:
        topology = "single_gpu"
    summary.setdefault("tensor_parallel_size", tp)
    summary.setdefault("gpu_per_request", gpu_per_request)
    summary.setdefault("runtime_gpu_count", runtime_gpu_count)
    summary.setdefault("visible_gpu_count", visible_gpu_count)
    summary.setdefault("parallelism_topology", topology)


def _detail_requests(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    detailed = data.get("detailed_results") or {}
    for scenario in detailed.values():
        requests = scenario.get("requests") if isinstance(scenario, dict) else None
        if isinstance(requests, list):
            return [req for req in requests if isinstance(req, dict)]
    return []


def _request_metric(request: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = request.get(key)
        if value is not None:
            return value
    return None


def _deadline_row(label: str, requests: List[Dict[str, Any]], deadline_ms: float) -> List[str]:
    total = len(requests)
    ok = [req for req in requests if bool(req.get("success"))]
    e2e_ok = [
        req for req in ok
        if (_request_metric(req, "overall_e2e_ms", "e2e_ms") is not None)
        and float(_request_metric(req, "overall_e2e_ms", "e2e_ms")) <= deadline_ms
    ]
    ttft_ok = [
        req for req in ok
        if (_request_metric(req, "overall_ttft_ms", "ttft_ms") is not None)
        and float(_request_metric(req, "overall_ttft_ms", "ttft_ms")) <= deadline_ms
    ]
    total_tokens = sum(int(req.get("completion_tokens", req.get("output_tokens", 0)) or 0) for req in ok)
    deadline_tokens = sum(int(req.get("completion_tokens", req.get("output_tokens", 0)) or 0) for req in e2e_ok)
    deadline_s = max(float(deadline_ms) / 1000.0, 1e-9)
    return [
        label,
        f"{len(e2e_ok)}/{total}",
        f"{(len(e2e_ok) / total) if total else 0.0:.4f}",
        f"{len(ttft_ok)}/{total}",
        f"{(len(ttft_ok) / total) if total else 0.0:.4f}",
        f"{(deadline_tokens / total_tokens) if total_tokens else 0.0:.4f}",
        f"{len(e2e_ok) / deadline_s:.4f}",
        f"{deadline_tokens / deadline_s:.4f}",
    ]


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _token_row(label: str, requests: List[Dict[str, Any]]) -> List[str]:
    ok = [req for req in requests if bool(req.get("success"))]
    prompt_tokens = [
        int(req.get("prompt_tokens", req.get("input_tokens", 0)) or 0)
        for req in ok
    ]
    completion_tokens = [
        int(req.get("completion_tokens", req.get("output_tokens", 0)) or 0)
        for req in ok
    ]
    return [
        label,
        f"{_avg([float(v) for v in prompt_tokens]):.2f}" if prompt_tokens else "0.00",
        str(max(prompt_tokens) if prompt_tokens else 0),
        f"{_avg([float(v) for v in completion_tokens]):.2f}" if completion_tokens else "0.00",
        str(max(completion_tokens) if completion_tokens else 0),
        str(sum(1 for value in completion_tokens if value > 256)),
    ]


def _load_summary(path: Path) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    schema = data.get("metric_schema_version") or (data.get("metadata", {}) or {}).get("metric_schema_version")
    if schema != "e2e_v3":
        raise RuntimeError(
            f"{path} is not metric_schema_version=e2e_v3 "
            f"(found {schema!r}). Refusing to compare mixed TTFT/E2E definitions."
        )
    summaries = dict(data.get("scenario_summaries", {}) or {})
    if not summaries:
        raise RuntimeError(f"no scenario_summaries found in {path}")
    scenario_name, summary = next(iter(summaries.items()))
    summary_schema = summary.get("metric_schema_version")
    if summary_schema != "e2e_v3":
        raise RuntimeError(
            f"{path}:{scenario_name} is not metric_schema_version=e2e_v3 "
            f"(found {summary_schema!r})."
        )
    required = ("avg_overall_ttft_ms", "avg_service_ttft_ms", "avg_overall_e2e_ms", "avg_service_e2e_ms")
    missing = [key for key in required if summary.get(key) is None]
    if missing:
        raise RuntimeError(f"{path}:{scenario_name} missing required e2e_v3 metrics: {', '.join(missing)}")
    metadata = dict(data.get("metadata", {}) or {})
    for key in (
        "max_model_len",
        "max_input_len",
        "max_output_tokens_cap",
        "requested_runtime_concurrency_cap",
        "runtime_concurrency_cap",
        "max_num_seqs",
        "max_loras",
        "tensor_parallel_size",
        "gpu_per_request",
        "runtime_gpu_count",
        "visible_gpu_count",
        "parallelism_topology",
        "deploy_config",
    ):
        if summary.get(key) is None and metadata.get(key) is not None:
            summary[key] = metadata.get(key)
    _attach_topology_fields(summary)
    system_name = str((data.get("metadata", {}) or {}).get("system") or summary.get("baseline_type") or path.stem)
    label = f"{system_name}:{scenario_name}"
    return label, summary, _detail_requests(data)


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _render(row: List[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    lines = [_render(headers), "-+-".join("-" * width for width in widths)]
    lines.extend(_render(row) for row in rows)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Print a direct comparison table for fair-round result JSONs.")
    ap.add_argument("--result", type=Path, action="append", required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--deadline-ms", type=float, default=DEFAULT_DEADLINE_MS)
    args = ap.parse_args()

    rows: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]] = [
        _load_summary(path.resolve()) for path in args.result
    ]
    strict_headers = ["System"] + [name for name, _ in STRICT_KEY_METRICS]
    strict_rows: List[List[str]] = []
    for label, summary, _ in rows:
        row = [label] + [_fmt(_metric_value(summary, field)) for _, field in STRICT_KEY_METRICS]
        strict_rows.append(row)

    diagnostic_headers = ["System"] + [name for name, _ in DIAGNOSTIC_KEY_METRICS]
    diagnostic_rows: List[List[str]] = []
    for label, summary, _ in rows:
        row = [label] + [_fmt(_metric_value(summary, field)) for _, field in DIAGNOSTIC_KEY_METRICS]
        diagnostic_rows.append(row)

    resource_headers = ["System"] + [name for name, _ in RESOURCE_KEY_METRICS]
    resource_rows: List[List[str]] = []
    for label, summary, _ in rows:
        row = [label] + [_fmt(_metric_value(summary, field)) for _, field in RESOURCE_KEY_METRICS]
        resource_rows.append(row)

    envelope_headers = ["System"] + [name for name, _ in ENVELOPE_KEY_METRICS]
    envelope_rows: List[List[str]] = []
    for label, summary, _ in rows:
        row = [label] + [_fmt(_metric_value(summary, field)) for _, field in ENVELOPE_KEY_METRICS]
        envelope_rows.append(row)

    topology_headers = ["System"] + [name for name, _ in TOPOLOGY_KEY_METRICS]
    topology_rows: List[List[str]] = []
    for label, summary, _ in rows:
        row = [label] + [_fmt(_metric_value(summary, field)) for _, field in TOPOLOGY_KEY_METRICS]
        topology_rows.append(row)

    deadline_headers = [
        "System",
        f"E2E<= {args.deadline_ms / 1000.0:.0f}s",
        "CompletionRate",
        f"TTFT<= {args.deadline_ms / 1000.0:.0f}s",
        "TTFTRate",
        "TokenCoverage",
        "EffRPS",
        "EffTok/s",
    ]
    deadline_rows = [
        _deadline_row(label, requests, float(args.deadline_ms))
        for label, _, requests in rows
    ]

    token_headers = [
        "System",
        "PromptTokAvg",
        "PromptTokMax",
        "OutTokAvg",
        "OutTokMax",
        "OutTok>256",
    ]
    token_rows = [_token_row(label, requests) for label, _, requests in rows]

    print("[Strict Main Metrics]")
    print(_render_table(strict_headers, strict_rows))
    print()
    print("[Diagnostic Breakdown Metrics]")
    print(_render_table(diagnostic_headers, diagnostic_rows))
    print()
    print("[Resource Efficiency Metrics]")
    print(_render_table(resource_headers, resource_rows))
    print()
    print("[Execution Envelope Audit]")
    print(_render_table(envelope_headers, envelope_rows))
    print()
    print("[Parallelism Topology Audit]")
    print(_render_table(topology_headers, topology_rows))
    print()
    print("[Deadline / Effective Service Metrics]")
    print(_render_table(deadline_headers, deadline_rows))
    print()
    print("[Served Token Distribution Audit]")
    print(_render_table(token_headers, token_rows))

    if args.output is not None:
        payload = {
            "metric_schema_version": "e2e_v3",
            "metric_definitions": {
                "primary_ttft": METRIC_DEF_PRIMARY_TTFT,
                "primary_e2e": METRIC_DEF_PRIMARY_E2E,
                "service_ttft": METRIC_DEF_SERVICE_TTFT,
                "service_e2e": METRIC_DEF_SERVICE_E2E,
                "resource_efficiency": (
                    "active/idle GPU-seconds, replica counts, and goodput per GPU-second "
                    "derived from each system's runtime lifecycle and request service intervals"
                ),
                "ce": (
                    "main monetary cost efficiency: 1 / (E2E_e2e_seconds * monetary "
                    "cost per completed request). Serverful runtimes pay full-price "
                    "lifecycle GPU-seconds; serverless runtimes pay full-price "
                    "startup/active GPU-seconds plus discounted idle-ready GPU-seconds."
                ),
                "infra_ce": (
                    "resource-normalized audit CE using flat deployment-lifecycle "
                    "GPU-second cost where every allocated GPU-second has the same price"
                ),
                "token_ce": (
                    "diagnostic legacy token-proxy CE retained only for auditing old cost accounting"
                ),
            },
            "strict_headers": strict_headers,
            "strict_rows": strict_rows,
            "diagnostic_headers": diagnostic_headers,
            "diagnostic_rows": diagnostic_rows,
            "resource_headers": resource_headers,
            "resource_rows": resource_rows,
            "envelope_headers": envelope_headers,
            "envelope_rows": envelope_rows,
            "topology_headers": topology_headers,
            "topology_rows": topology_rows,
            "deadline_ms": float(args.deadline_ms),
            "deadline_headers": deadline_headers,
            "deadline_rows": deadline_rows,
            "token_headers": token_headers,
            "token_rows": token_rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote comparison table -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
