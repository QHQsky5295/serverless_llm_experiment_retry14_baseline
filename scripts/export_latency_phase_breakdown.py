#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PHASE_SCHEMA_VERSION = "latency_phase_v2"


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _first_float(mapping: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        parsed = _as_float(mapping.get(key))
        if parsed is not None:
            return parsed
    return None


def _percentile(values: Sequence[float], pct: float) -> Optional[float]:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def _stats(values: Sequence[float], total: int) -> Dict[str, Any]:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    return {
        "observed": len(xs),
        "total": int(total),
        "observed_ratio": (len(xs) / total) if total else 0.0,
        "avg_ms": (sum(xs) / len(xs)) if xs else None,
        "p50_ms": _percentile(xs, 50.0),
        "p95_ms": _percentile(xs, 95.0),
        "p99_ms": _percentile(xs, 99.0),
        "max_ms": max(xs) if xs else None,
    }


def _detail_requests(data: Dict[str, Any]) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    summaries = data.get("scenario_summaries") or {}
    if not isinstance(summaries, dict) or not summaries:
        raise RuntimeError("result JSON has no scenario_summaries")
    scenario_name, summary = next(iter(summaries.items()))
    detailed = data.get("detailed_results") or {}
    scenario = detailed.get(scenario_name)
    if not isinstance(scenario, dict) or not isinstance(scenario.get("requests"), list):
        raise RuntimeError(f"result JSON has no detailed request list for {scenario_name}")
    requests = [req for req in scenario["requests"] if isinstance(req, dict)]
    return str(scenario_name), summary if isinstance(summary, dict) else {}, requests


def _load_result(path: Path) -> Tuple[str, str, Dict[str, Any], List[Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    schema = data.get("metric_schema_version") or (data.get("metadata") or {}).get("metric_schema_version")
    if schema != "e2e_v3":
        raise RuntimeError(f"{path} is not metric_schema_version=e2e_v3 (found {schema!r})")
    scenario_name, summary, requests = _detail_requests(data)
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    system = str(metadata.get("system") or summary.get("baseline_type") or path.stem)
    return system, scenario_name, summary, requests


def _adapter_prepare_ms(req: Dict[str, Any]) -> Optional[float]:
    base = _first_float(req, ("lora_io_ms", "lora_load_ms"))
    if base is None:
        return None
    contention = _first_float(req, ("contention_ms",)) or 0.0
    defer = _first_float(req, ("defer_ms",)) or 0.0
    return max(0.0, base + contention + defer)


def _phase_values(requests: Iterable[Dict[str, Any]]) -> Dict[str, List[float]]:
    phases: Dict[str, List[float]] = {
        "dispatch_admission_wait_ms": [],
        "service_ttft_ms": [],
        "decode_tail_ms": [],
        "service_e2e_ms": [],
        "overall_ttft_ms": [],
        "overall_e2e_ms": [],
        "adapter_prepare_observed_ms": [],
        "first_token_compute_residual_observed_ms": [],
        "runtime_estimated_e2e_ms": [],
        "worker_wall_e2e_ms": [],
        "parent_rpc_overhead_ms": [],
        "service_path_residual_ms": [],
    }
    for req in requests:
        dispatch = _first_float(req, ("dispatch_admission_wait_ms", "queue_wait_ms"))
        service_ttft = _first_float(req, ("service_ttft_ms", "server_service_ttft_ms"))
        service_e2e = _first_float(req, ("service_e2e_ms", "server_service_e2e_ms"))
        overall_ttft = _first_float(req, ("overall_ttft_ms", "ttft_ms"))
        overall_e2e = _first_float(req, ("overall_e2e_ms", "e2e_ms"))
        runtime_estimated_e2e = _first_float(req, ("runtime_estimated_e2e_ms",))
        worker_wall_e2e = _first_float(req, ("worker_wall_e2e_ms",))
        parent_rpc_overhead = _first_float(req, ("parent_rpc_overhead_ms",))
        service_path_residual = _first_float(req, ("service_path_residual_ms",))

        if dispatch is not None:
            phases["dispatch_admission_wait_ms"].append(max(0.0, dispatch))
        if service_ttft is not None:
            phases["service_ttft_ms"].append(max(0.0, service_ttft))
        if service_e2e is not None:
            phases["service_e2e_ms"].append(max(0.0, service_e2e))
        if service_ttft is not None and service_e2e is not None:
            phases["decode_tail_ms"].append(max(0.0, service_e2e - service_ttft))
        if overall_ttft is not None:
            phases["overall_ttft_ms"].append(max(0.0, overall_ttft))
        if overall_e2e is not None:
            phases["overall_e2e_ms"].append(max(0.0, overall_e2e))
        if runtime_estimated_e2e is not None:
            phases["runtime_estimated_e2e_ms"].append(max(0.0, runtime_estimated_e2e))
        if worker_wall_e2e is not None:
            phases["worker_wall_e2e_ms"].append(max(0.0, worker_wall_e2e))
        if parent_rpc_overhead is not None:
            phases["parent_rpc_overhead_ms"].append(max(0.0, parent_rpc_overhead))
        if service_path_residual is not None:
            phases["service_path_residual_ms"].append(max(0.0, service_path_residual))

        adapter_prepare = _adapter_prepare_ms(req)
        if adapter_prepare is not None:
            phases["adapter_prepare_observed_ms"].append(adapter_prepare)
            if service_ttft is not None:
                phases["first_token_compute_residual_observed_ms"].append(
                    max(0.0, service_ttft - adapter_prepare)
                )
    return phases


def _deadline_metrics(requests: Sequence[Dict[str, Any]], deadline_ms: float) -> Dict[str, Any]:
    ok = [req for req in requests if bool(req.get("success"))]
    e2e_ok = [
        req for req in ok
        if (_first_float(req, ("overall_e2e_ms", "e2e_ms")) is not None)
        and _first_float(req, ("overall_e2e_ms", "e2e_ms")) <= deadline_ms
    ]
    ttft_ok = [
        req for req in ok
        if (_first_float(req, ("overall_ttft_ms", "ttft_ms")) is not None)
        and _first_float(req, ("overall_ttft_ms", "ttft_ms")) <= deadline_ms
    ]
    total_tokens = sum(int(req.get("completion_tokens", req.get("output_tokens", 0)) or 0) for req in ok)
    deadline_tokens = sum(int(req.get("completion_tokens", req.get("output_tokens", 0)) or 0) for req in e2e_ok)
    return {
        "deadline_ms": float(deadline_ms),
        "completed_within_deadline": len(e2e_ok),
        "ttft_within_deadline": len(ttft_ok),
        "total": len(requests),
        "completion_rate": (len(e2e_ok) / len(requests)) if requests else 0.0,
        "ttft_rate": (len(ttft_ok) / len(requests)) if requests else 0.0,
        "token_coverage": (deadline_tokens / total_tokens) if total_tokens else 0.0,
    }


def _build_rows(path: Path, deadline_ms: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    system, scenario, summary, requests = _load_result(path)
    phases = _phase_values(requests)
    total = len(requests)
    rows: List[Dict[str, Any]] = []
    for stack_name, phase_names in (
        (
            "overall_e2e_stack",
            ("dispatch_admission_wait_ms", "service_ttft_ms", "decode_tail_ms"),
        ),
        (
            "service_detail_observed_stack",
            ("adapter_prepare_observed_ms", "first_token_compute_residual_observed_ms", "decode_tail_ms"),
        ),
        (
            "reference_totals",
            ("overall_ttft_ms", "overall_e2e_ms", "service_e2e_ms"),
        ),
        (
            "service_runtime_residual_stack",
            (
                "runtime_estimated_e2e_ms",
                "service_path_residual_ms",
            ),
        ),
        (
            "runtime_diagnosis_reference_totals",
            (
                "parent_rpc_overhead_ms",
                "worker_wall_e2e_ms",
            ),
        ),
    ):
        for phase in phase_names:
            row = {
                "system": system,
                "scenario": scenario,
                "result_path": str(path),
                "stack": stack_name,
                "phase": phase,
            }
            row.update(_stats(phases[phase], total))
            rows.append(row)

    payload = {
        "system": system,
        "scenario": scenario,
        "result_path": str(path),
        "summary": summary,
        "deadline": _deadline_metrics(requests, deadline_ms),
        "phase_stats": {
            phase: _stats(values, total)
            for phase, values in phases.items()
        },
    }
    return rows, payload


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "system",
        "scenario",
        "stack",
        "phase",
        "avg_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
        "observed",
        "total",
        "observed_ratio",
        "result_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export latency phase breakdowns from fair-round e2e_v3 result JSONs."
    )
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--deadline-ms", type=float, default=300_000.0)
    args = parser.parse_args()

    all_rows: List[Dict[str, Any]] = []
    payload: Dict[str, Any] = {
        "phase_schema_version": PHASE_SCHEMA_VERSION,
        "metric_schema_version": "e2e_v3",
        "phase_definitions": {
            "dispatch_admission_wait_ms": (
                "scheduled trace arrival to admitted/backend service start; includes replay "
                "dispatch and backend queue/admission wait"
            ),
            "service_ttft_ms": "admitted/backend service start to first generated output token/chunk",
            "decode_tail_ms": "service completion minus service TTFT",
            "adapter_prepare_observed_ms": (
                "observed adapter movement/preparation time only when the system reports "
                "lora_io_ms or lora_load_ms; missing values are not interpreted as zero"
            ),
            "first_token_compute_residual_observed_ms": (
                "service TTFT minus observed adapter preparation time, only for requests with "
                "adapter preparation observations"
            ),
            "runtime_estimated_e2e_ms": (
                "backend-native runtime estimate: runtime TTFT plus runtime TPOT times generated token tail"
            ),
            "worker_wall_e2e_ms": "wall-clock generate time observed inside the dedicated inference worker",
            "parent_rpc_overhead_ms": "parent-side RPC wall time minus worker generate wall time",
            "service_path_residual_ms": (
                "service E2E minus backend-native runtime estimate; indicates serving orchestration, "
                "proxy, and runtime-wall residual outside adapter preparation"
            ),
        },
        "results": [],
    }

    for result_path in args.result:
        rows, result_payload = _build_rows(result_path.resolve(), float(args.deadline_ms))
        all_rows.extend(rows)
        payload["results"].append(result_payload)

    _write_csv(args.output_csv, all_rows)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote latency phase CSV -> {args.output_csv}")
    if args.output_json is not None:
        print(f"wrote latency phase JSON -> {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
