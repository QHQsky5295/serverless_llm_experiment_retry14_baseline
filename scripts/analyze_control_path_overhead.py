#!/usr/bin/env python3
"""Generate a PrimeLoRA-only control-path overhead audit.

This script intentionally does not compare control-path overhead across
different serving systems. Baselines expose different wrapper, runtime, and
admission boundaries, so the paper-facing audit is scoped to PrimeLoRA's
additional online decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


OPERATIONS = [
    (
        "Routing +\ntier lookup",
        "routing_decision_us",
        "avg_routing_decision_us",
        "p95_routing_decision_us",
        False,
    ),
    (
        "Adapter-path\nresolution",
        "adapter_path_resolution_us",
        "avg_adapter_path_resolution_us",
        "p95_adapter_path_resolution_us",
        False,
    ),
    (
        "GPU-admission\ncheck",
        "gpu_admission_decision_us",
        "avg_gpu_admission_decision_us",
        "p95_gpu_admission_decision_us",
        True,
    ),
    (
        "Online control\ntotal",
        "control_path_total_us",
        "avg_control_path_total_us",
        "p95_control_path_total_us",
        False,
    ),
]


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def percentile(values: Sequence[float], p: float) -> float:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return 0.0
    if len(finite) == 1:
        return finite[0]
    rank = max(0.0, min(1.0, p / 100.0)) * (len(finite) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return finite[lo]
    frac = rank - lo
    return finite[lo] * (1.0 - frac) + finite[hi] * frac


def iter_json_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    for path in sorted(input_path.rglob("*.json")):
        yield path


def load_payloads(input_path: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    for path in iter_json_files(input_path):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            yield path, payload


def scenario_records(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    detailed = payload.get("detailed_results")
    if isinstance(detailed, dict):
        for scenario_name, scenario in detailed.items():
            if isinstance(scenario, dict):
                item = dict(scenario)
                item.setdefault("scenario_name", scenario_name)
                records.append(item)
    elif isinstance(detailed, list):
        for item in detailed:
            if isinstance(item, dict):
                records.append(dict(item))
    if not records and isinstance(payload.get("requests"), list):
        records.append(dict(payload))
    return records


def scenario_summary(payload: Dict[str, Any], scenario_name: str) -> Dict[str, Any]:
    summaries = payload.get("scenario_summaries")
    if not isinstance(summaries, dict):
        return {}
    if scenario_name in summaries and isinstance(summaries[scenario_name], dict):
        return summaries[scenario_name]
    for item in summaries.values():
        if isinstance(item, dict) and item.get("scenario_name") == scenario_name:
            return item
    return {}


def select_scenario(
    payloads: Iterable[Tuple[Path, Dict[str, Any]]],
    scenario_filter: Optional[str],
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    candidates: List[Tuple[Path, Dict[str, Any], Dict[str, Any]]] = []
    for path, payload in payloads:
        for scenario in scenario_records(payload):
            name = str(scenario.get("scenario_name") or path.stem)
            baseline = str(scenario.get("baseline_type") or "").lower()
            if scenario_filter and scenario_filter.lower() not in name.lower():
                continue
            if scenario_filter is None and baseline not in {"faaslora_full", "primelora"}:
                continue
            summary = scenario_summary(payload, name)
            candidates.append((path, scenario, summary))
    if not candidates:
        raise SystemExit(
            "No PrimeLoRA/FaaSLoRA scenario with control-path fields found. "
            "Run a new PrimeLoRA result after the control-path instrumentation."
        )
    def score(candidate: Tuple[Path, Dict[str, Any], Dict[str, Any]]) -> Tuple[int, int]:
        _, scenario, summary = candidate
        requests = scenario.get("requests") if isinstance(scenario.get("requests"), list) else []
        has_request_fields = int(
            any(_finite_float(req.get("control_path_total_us"), -1.0) >= 0.0 for req in requests)
        )
        has_summary_fields = int(_finite_float(summary.get("avg_control_path_total_us"), -1.0) >= 0.0)
        return has_request_fields + has_summary_fields, len(requests)
    return max(candidates, key=score)


def summarize_operation(
    scenario: Dict[str, Any],
    summary: Dict[str, Any],
    request_field: str,
    avg_field: str,
    p95_field: str,
    positive_only: bool = False,
) -> Tuple[float, float, int]:
    requests = scenario.get("requests")
    values = []
    if isinstance(requests, list):
        for req in requests:
            if not isinstance(req, dict) or not bool(req.get("success", True)):
                continue
            value = _finite_float(req.get(request_field), -1.0)
            if value < 0.0:
                continue
            if positive_only and value <= 0.0:
                continue
            values.append(value)
    if values:
        return sum(values) / len(values), percentile(values, 95), len(values)
    avg = _finite_float(summary.get(avg_field), 0.0)
    p95 = _finite_float(summary.get(p95_field), 0.0)
    count = int(_finite_float(summary.get("completed_requests"), 0.0))
    if avg <= 0.0 and p95 <= 0.0:
        raise SystemExit(
            f"Missing control-path field {request_field}; archived results cannot produce this audit."
        )
    return avg, p95, count


def summarize_background(summary: Dict[str, Any]) -> Tuple[float, float, int]:
    avg = _finite_float(summary.get("avg_background_planning_us"), 0.0)
    p95 = _finite_float(summary.get("p95_background_planning_us"), 0.0)
    count = int(_finite_float(summary.get("background_planning_event_count"), 0.0))
    return avg, p95, count


def write_csv(rows: List[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: List[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Control-path overhead of PrimeLoRA. All values are in milliseconds. Online total is measured per request; GPU-admission and background-planning rows report triggered events.}",
        r"\label{tab:control_path_overhead}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Operation & Avg & P95 & Events \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['operation_latex']} & {row['avg_ms']:.3f} & {row['p95_ms']:.3f} & {int(row['events'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(rows: List[Dict[str, Any]], output: Path, source: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "control_path_overhead",
        "scope": "PrimeLoRA-only diagnostic audit; not a cross-system comparison",
        "source": str(source),
        "operations": [
            {
                "operation": row["operation_label"].replace("\n", " "),
                "avg_ms": row["avg_ms"],
                "p95_ms": row["p95_ms"],
                "events": int(row["events"]),
            }
            for row in rows
        ],
        "field_note": (
            "Online control total is routing_decision_us + "
            "adapter_path_resolution_us + gpu_admission_decision_us. "
            "Background handoff planning is reported separately when present."
        ),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot(rows: List[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["operation_label"]) for row in rows]
    avg_ms = [float(row["avg_ms"]) for row in rows]
    p95_ms = [float(row["p95_ms"]) for row in rows]
    y = list(range(len(rows)))
    y_avg = [idx - 0.055 for idx in y]
    y_p95 = [idx + 0.055 for idx in y]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.labelsize": 10.8,
        "xtick.labelsize": 10.2,
        "ytick.labelsize": 10.2,
        "legend.fontsize": 9.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.45, 2.72))
    ax.hlines(y, avg_ms, p95_ms, color="#a8b4c4", linewidth=1.15, zorder=1)
    ax.scatter(avg_ms, y_avg, s=42, marker="o", color="#2b6cb0", label="Avg", zorder=3)
    ax.scatter(p95_ms, y_p95, s=48, marker="D", color="#c53030", label="P95", zorder=3)
    for idx, p95 in enumerate(p95_ms):
        ax.annotate(
            f"{p95:.3f}",
            (p95, y_p95[idx]),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.8,
            color="#1f2937",
            bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            zorder=4,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Overhead (ms)")
    xmax = max(max(p95_ms), max(avg_ms), 0.01) * 1.28
    ax.set_xlim(left=0, right=xmax)
    ax.grid(axis="x", linestyle="--", linewidth=0.55, alpha=0.35)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.35, borderaxespad=0.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.25)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Result JSON file or directory")
    parser.add_argument("--output", default="figs/paper/control_path", help="Output directory")
    parser.add_argument("--scenario", default=None, help="Optional scenario-name substring")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    source, scenario, summary = select_scenario(load_payloads(input_path), args.scenario)

    rows: List[Dict[str, Any]] = []
    for label, request_field, avg_field, p95_field, positive_only in OPERATIONS:
        avg_us, p95_us, count = summarize_operation(
            scenario,
            summary,
            request_field,
            avg_field,
            p95_field,
            positive_only=positive_only,
        )
        rows.append({
            "operation_label": label,
            "operation_latex": label.replace("\n", " "),
            "avg_us": avg_us,
            "p95_us": p95_us,
            "avg_ms": avg_us / 1000.0,
            "p95_ms": p95_us / 1000.0,
            "events": count,
            "source": str(source),
        })
    bg_avg_us, bg_p95_us, bg_count = summarize_background(summary)
    if bg_count > 0 or bg_avg_us > 0.0 or bg_p95_us > 0.0:
        rows.append({
            "operation_label": "Background\nhandoff plan",
            "operation_latex": "Background handoff plan",
            "avg_us": bg_avg_us,
            "p95_us": bg_p95_us,
            "avg_ms": bg_avg_us / 1000.0,
            "p95_ms": bg_p95_us / 1000.0,
            "events": bg_count,
            "source": str(source),
        })

    write_csv(rows, output_dir / "control_path_overhead_summary.csv")
    write_latex(rows, output_dir / "tables" / "table_control_path_overhead.tex")
    write_manifest(rows, output_dir / "control_path_overhead_manifest.json", source)
    plot(rows, output_dir / "fig_control_path_overhead.pdf")
    paper_copy = Path("figs/fig_control_path_overhead.pdf").resolve()
    paper_copy.parent.mkdir(parents=True, exist_ok=True)
    plot(rows, paper_copy)

    print(f"source={source}")
    for row in rows:
        print(
            f"{row['operation_label'].replace(chr(10), ' ')}: "
            f"avg={row['avg_ms']:.3f}ms p95={row['p95_ms']:.3f}ms events={int(row['events'])}"
        )
    print(f"wrote {output_dir / 'fig_control_path_overhead.pdf'}")


if __name__ == "__main__":
    main()
