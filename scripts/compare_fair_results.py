#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


KEY_METRICS = [
    ("CE", "ce"),
    ("TTFT_overall_ms", "avg_ttft_ms"),
    ("TTFT_comp_ms", "avg_comparable_ttft_ms"),
    ("TTFT_warm_ms", "avg_warm_standard_ttft_ms"),
    ("TPOT_ms", "avg_tpot_ms"),
    ("Tok/s", "throughput_tok_per_s"),
    ("E2E_ms", "avg_e2e_ms"),
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
    return summary.get(key)


def _load_summary(path: Path) -> Tuple[str, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summaries = dict(data.get("scenario_summaries", {}) or {})
    if not summaries:
        raise RuntimeError(f"no scenario_summaries found in {path}")
    scenario_name, summary = next(iter(summaries.items()))
    system_name = str((data.get("metadata", {}) or {}).get("system") or summary.get("baseline_type") or path.stem)
    label = f"{system_name}:{scenario_name}"
    return label, summary


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    ap = argparse.ArgumentParser(description="Print a direct comparison table for fair-round result JSONs.")
    ap.add_argument("--result", type=Path, action="append", required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    rows: List[Tuple[str, Dict[str, Any]]] = [_load_summary(path.resolve()) for path in args.result]
    headers = ["System"] + [name for name, _ in KEY_METRICS]
    table_rows: List[List[str]] = []
    for label, summary in rows:
        row = [label] + [_fmt(_metric_value(summary, field)) for _, field in KEY_METRICS]
        table_rows.append(row)

    widths = [len(h) for h in headers]
    for row in table_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _render(row: List[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    print(_render(headers))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(_render(row))

    if args.output is not None:
        payload = {
            "headers": headers,
            "rows": table_rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote comparison table -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
