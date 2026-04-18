#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


STRICT_KEY_METRICS = [
    ("TTFT_e2e_ms", "avg_overall_ttft_ms"),
    ("E2E_e2e_ms", "avg_overall_e2e_ms"),
    ("TPOT_ms", "avg_tpot_ms"),
    ("Tok/s", "throughput_tok_per_s"),
    ("RPS", "throughput_rps"),
    ("SLO", "slo_attainment"),
    ("Cost_req_usd", "avg_cost_usd"),
    ("CE", "ce"),
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
]


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
    return summary.get(key)


def _load_summary(path: Path) -> Tuple[str, Dict[str, Any]]:
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
    system_name = str((data.get("metadata", {}) or {}).get("system") or summary.get("baseline_type") or path.stem)
    label = f"{system_name}:{scenario_name}"
    return label, summary


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
    args = ap.parse_args()

    rows: List[Tuple[str, Dict[str, Any]]] = [_load_summary(path.resolve()) for path in args.result]
    strict_headers = ["System"] + [name for name, _ in STRICT_KEY_METRICS]
    strict_rows: List[List[str]] = []
    for label, summary in rows:
        row = [label] + [_fmt(_metric_value(summary, field)) for _, field in STRICT_KEY_METRICS]
        strict_rows.append(row)

    diagnostic_headers = ["System"] + [name for name, _ in DIAGNOSTIC_KEY_METRICS]
    diagnostic_rows: List[List[str]] = []
    for label, summary in rows:
        row = [label] + [_fmt(_metric_value(summary, field)) for _, field in DIAGNOSTIC_KEY_METRICS]
        diagnostic_rows.append(row)

    print("[Strict Main Metrics]")
    print(_render_table(strict_headers, strict_rows))
    print()
    print("[Diagnostic Breakdown Metrics]")
    print(_render_table(diagnostic_headers, diagnostic_rows))

    if args.output is not None:
        payload = {
            "metric_schema_version": "e2e_v3",
            "metric_definitions": {
                "primary_ttft": "scheduled trace arrival to client-observed first output token/chunk",
                "primary_e2e": "scheduled trace arrival to client-observed response completion",
                "service_ttft": "common system-ingress to first output token/chunk; ingress is request release/dispatch into the target serving system and excludes scheduled-arrival wait",
                "service_e2e": "common system-ingress to response completion; ingress is request release/dispatch into the target serving system and excludes scheduled-arrival wait",
            },
            "strict_headers": strict_headers,
            "strict_rows": strict_rows,
            "diagnostic_headers": diagnostic_headers,
            "diagnostic_rows": diagnostic_rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote comparison table -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
