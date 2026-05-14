#!/usr/bin/env python3
"""Compare remote-fair summaries against an existing paper CSV.

This helper is intentionally read-only: it does not rerun experiments, rewrite
summaries, or change metric definitions.  It is used to audit whether a new
local-sim or true-remote round remains aligned with the already closed-loop
paper results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


SYSTEM_ALIASES = {
    "SGLang": "SGLang",
    "vLLM": "vLLM",
    "S-LoRA": "S-LoRA",
    "ServerlessLLM": "ServerlessLLM",
    "FaaSLoRA": "PrimeLoRA",
    "PrimeLoRA": "PrimeLoRA",
}

SYSTEM_FROM_KEY = {
    "sglang": "SGLang",
    "vllm": "vLLM",
    "slora": "S-LoRA",
    "serverlessllm": "ServerlessLLM",
    "faaslora": "PrimeLoRA",
}

METRICS = (
    ("ttft_avg_ms", "TTFT Avg"),
    ("ttft_p95_ms", "TTFT P95"),
    ("e2e_avg_ms", "E2E Avg"),
    ("e2e_p95_ms", "E2E P95"),
    ("tpot_avg_ms", "TPOT Avg"),
    ("tok_s", "Throughput"),
    ("cost_req_musd", "Cost/req"),
    ("ce", "CE"),
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def _pick(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _model_from_path(path: Path) -> str:
    text = str(path)
    if "llama32_3b" in text:
        return "Llama-3.2 3B"
    if "llama2_13b" in text:
        return "Llama-2 13B"
    if "llama2_7b" in text:
        return "Llama-2 7B"
    return "unknown"


def _system_from_path(path: Path) -> str:
    name = path.name.lower()
    for key, label in SYSTEM_FROM_KEY.items():
        if f"_{key}_" in name or name.endswith(f"_{key}_summary.json") or f"_{key}_result" in name:
            return label
    return "unknown"


def _metric_from_summary(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    model = _model_from_path(path)
    system = _system_from_path(path)

    if payload.get("comparison_table"):
        row = payload["comparison_table"][0]
        system = SYSTEM_ALIASES.get(str(row.get("System", system)), system)
        return {
            "model": model,
            "system": system,
            "source": str(path),
            "completed": int(_float(row.get("completed", row.get("Completed", 0)))),
            "total": int(_float(row.get("total", row.get("Total", 0)))),
            "ttft_avg_ms": _float(_pick(row, "TTFT_avg_ms")),
            "ttft_p95_ms": _float(_pick(row, "TTFT_p95_ms", "TTFT_P95_ms")),
            "e2e_avg_ms": _float(_pick(row, "E2E_avg_ms")),
            "e2e_p95_ms": _float(_pick(row, "E2E_p95_ms", "E2E_P95_ms")),
            "tpot_avg_ms": _float(_pick(row, "TPOT_avg_ms")),
            "tok_s": _float(_pick(row, "Tok_s", "throughput_TOKPS")),
            "cost_req_musd": _float(_pick(row, "Cost_req_usd", "avg_cost_USD", "monetary_cost_per_request_usd")) * 1000.0,
            "ce": _float(row.get("CE")),
        }

    scenario_summaries = payload.get("scenario_summaries") or {}
    summary = scenario_summaries.get("faaslora_full")
    if summary:
        return {
            "model": model,
            "system": "PrimeLoRA",
            "source": str(path),
            "completed": int(_float(summary.get("total_requests", summary.get("successful_requests", 0)))),
            "total": int(_float(summary.get("total_requests", 0))),
            "ttft_avg_ms": _float(summary.get("avg_ttft_ms")),
            "ttft_p95_ms": _float(summary.get("p95_ttft_ms")),
            "e2e_avg_ms": _float(summary.get("avg_e2e_ms")),
            "e2e_p95_ms": _float(summary.get("p95_e2e_ms")),
            "tpot_avg_ms": _float(summary.get("avg_tpot_ms")),
            "tok_s": _float(summary.get("throughput_tok_s")),
            "cost_req_musd": _float(summary.get("cost_per_request_usd")) * 1000.0,
            "ce": _float(summary.get("cost_efficiency")),
        }

    if len(scenario_summaries) == 1:
        summary = next(iter(scenario_summaries.values()))
        return {
            "model": model,
            "system": system,
            "source": str(path),
            "completed": int(_float(summary.get("completed_requests", summary.get("total_requests", 0)))),
            "total": int(_float(summary.get("total_requests", 0))),
            "ttft_avg_ms": _float(summary.get("avg_ttft_ms", summary.get("avg_overall_ttft_ms"))),
            "ttft_p95_ms": _float(summary.get("p95_ttft_ms", summary.get("p95_overall_ttft_ms"))),
            "e2e_avg_ms": _float(summary.get("avg_e2e_ms", summary.get("avg_overall_e2e_ms"))),
            "e2e_p95_ms": _float(summary.get("p95_e2e_ms", summary.get("p95_overall_e2e_ms"))),
            "tpot_avg_ms": _float(summary.get("avg_tpot_ms")),
            "tok_s": _float(summary.get("throughput_tok_per_s", summary.get("throughput_tok_s"))),
            "cost_req_musd": _float(
                summary.get(
                    "monetary_cost_per_request_usd",
                    summary.get("infra_cost_per_request_usd", summary.get("avg_cost_usd")),
                )
            )
            * 1000.0,
            "ce": _float(summary.get("ce", summary.get("cost_efficiency", summary.get("monetary_ce")))),
        }

    raise SystemExit(f"{path}: unsupported summary schema")


def _read_reference(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = str(row.get("model") or row.get("Model") or "").strip()
            system = str(row.get("system") or row.get("System") or "").strip()
            if not model or not system:
                continue
            out[(model, system)] = {
                "ttft_avg_ms": _float(row.get("ttft_avg_ms") or row.get("TTFT Avg")),
                "ttft_p95_ms": _float(row.get("ttft_p95_ms") or row.get("TTFT P95")),
                "e2e_avg_ms": _float(row.get("e2e_avg_ms") or row.get("E2E Avg")),
                "e2e_p95_ms": _float(row.get("e2e_p95_ms") or row.get("E2E P95")),
                "tpot_avg_ms": _float(row.get("tpot_avg_ms") or row.get("TPOT Avg")),
                "tok_s": _float(row.get("tok_s") or row.get("Throughput")),
                "cost_req_musd": (
                    _float(row.get("cost_req_usd")) * 1000.0
                    if row.get("cost_req_usd") not in (None, "")
                    else _float(row.get("Cost/req"))
                ),
                "ce": _float(row.get("ce") or row.get("CE")),
            }
    return out


def _scan_summaries(round_dir: Path) -> list[Path]:
    patterns = ("raw/replay/*summary.json", "raw/faaslora/*result.json")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(round_dir.glob(pattern))
    return sorted(paths)


def _rows(round_dir: Path, reference_csv: Path) -> list[dict[str, Any]]:
    reference = _read_reference(reference_csv)
    rows: list[dict[str, Any]] = []
    for path in _scan_summaries(round_dir):
        current = _metric_from_summary(path)
        ref = reference.get((current["model"], current["system"]))
        base = {
            "model": current["model"],
            "system": current["system"],
            "completed": current["completed"],
            "total": current["total"],
            "source": current["source"],
            "reference_found": bool(ref),
        }
        for key, label in METRICS:
            value = current[key]
            base[f"{key}_current"] = value
            if ref:
                base[f"{key}_reference"] = ref[key]
                base[f"{key}_delta"] = value - ref[key]
                base[f"{key}_delta_pct"] = (value / ref[key] - 1.0) * 100.0 if ref[key] else math.nan
            else:
                base[f"{key}_reference"] = math.nan
                base[f"{key}_delta"] = math.nan
                base[f"{key}_delta_pct"] = math.nan
        rows.append(base)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Remote-Fair Result Comparison",
        "",
        "| Model | System | Completed | TTFT Avg Δ | E2E Avg Δ | Cost Δ | CE Δ | Source |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        def fmt_delta(key: str) -> str:
            pct = row.get(f"{key}_delta_pct")
            delta = row.get(f"{key}_delta")
            if pct is None or math.isnan(float(pct)):
                return "n/a"
            return f"{delta:.2f} ({pct:+.2f}%)"

        lines.append(
            "| {model} | {system} | {completed}/{total} | {ttft} | {e2e} | {cost} | {ce} | `{source}` |".format(
                model=row["model"],
                system=row["system"],
                completed=row["completed"],
                total=row["total"],
                ttft=fmt_delta("ttft_avg_ms"),
                e2e=fmt_delta("e2e_avg_ms"),
                cost=fmt_delta("cost_req_musd"),
                ce=fmt_delta("ce"),
                source=row["source"],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a remote-fair round against a closed-loop reference CSV.")
    parser.add_argument("--round-dir", required=True, type=Path)
    parser.add_argument("--reference-csv", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    rows = _rows(args.round_dir.resolve(), args.reference_csv.resolve())
    _write_csv(args.out_csv.resolve(), rows)
    if args.out_md:
        _write_md(args.out_md.resolve(), rows)
    for row in rows:
        print(
            f"{row['model']} {row['system']}: {row['completed']}/{row['total']} "
            f"TTFTavg={row['ttft_avg_ms_current']:.2f}ms "
            f"E2Eavg={row['e2e_avg_ms_current']:.2f}ms "
            f"CE={row['ce_current']:.2f}"
        )


if __name__ == "__main__":
    main()
