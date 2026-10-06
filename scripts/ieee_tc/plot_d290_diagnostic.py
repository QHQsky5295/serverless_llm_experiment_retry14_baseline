#!/usr/bin/env python3
"""Create a paper-facing diagnostic figure for the D290 bounded PrimeLoRA run.

This is deliberately a diagnostic, not a comparison plot: D290 is a valid
native-contract replay, but it does not qualify the frozen G1/G2 gates.  The
script derives all values from the immutable result JSON and records its SHA in
the manifest so the figure cannot silently drift from the experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def percentile(values: list[float], q: float) -> float:
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    # Type-1 percentile, matching the controller/result convention.
    idx = max(0, min(len(values) - 1, int(np.ceil(q * len(values))) - 1))
    return values[idx]


def load(path: Path) -> tuple[dict, dict]:
    result = json.loads(path.read_text(encoding="utf-8"))
    scenario = next(iter(result["detailed_results"].values()))
    requests = [r for r in scenario["requests"] if r.get("success")]
    if len(requests) != scenario.get("completed"):
        raise ValueError("diagnostic expects every completed request to be successful")
    return result, scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.result_json.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result, scenario = load(result_path)
    requests = [r for r in scenario["requests"] if r.get("success")]

    # TTFT identity: arrival release lateness + dispatch/admission wait is the
    # external queue component; service TTFT is the selected runtime component.
    queue = [
        float(r.get("arrival_release_lateness_ms", 0.0) or 0.0)
        + float(r.get("dispatch_admission_wait_ms", 0.0) or 0.0)
        for r in requests
    ]
    service = [float(r.get("service_ttft_ms", 0.0) or 0.0) for r in requests]
    overall = [float(r.get("overall_ttft_ms", r.get("ttft_ms", 0.0)) or 0.0) for r in requests]
    startup_affected = [
        float(r.get("overall_ttft_ms", 0.0) or 0.0)
        for r in requests
        if r.get("scaleup_affected") or r.get("on_scaleup_runtime")
    ]
    tiers = {}
    for r in requests:
        tier = str(r.get("readiness_tier_before_dispatch") or "unknown")
        tiers[tier] = tiers.get(tier, 0) + 1
    frontier_rows = 0
    cold_reads = 0
    for r in requests:
        evidence = r.get("gpu_reference_evidence") or {}
        routing = evidence.get("routing_snapshot") or {}
        sources = routing.get("sources") if isinstance(routing, dict) else None
        if isinstance(sources, dict):
            row_frontier = False
            for source in sources.values():
                collection = (source or {}).get("source_collection") if isinstance(source, dict) else None
                if not isinstance(collection, dict):
                    continue
                row_frontier = row_frontier or bool(collection.get("frontier_absence_replica_indexes"))
                cold_reads += int(collection.get("cold_full_read_replica_count", 0) or 0)
            frontier_rows += int(row_frontier)

    stage_rows = []
    for metric, values in (
        ("queue/admission", queue),
        ("service", service),
        ("overall TTFT", overall),
    ):
        stage_rows.append({
            "metric": metric,
            "mean_ms": mean(values) if values else 0.0,
            "p95_ms": percentile(values, 0.95),
            "n": len(values),
        })
    (out_dir / "stage_summary.csv").write_text("", encoding="utf-8")
    with (out_dir / "stage_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean_ms", "p95_ms", "n"])
        writer.writeheader()
        writer.writerows(stage_rows)

    # IEEE single-column geometry; Times New Roman is requested by the paper
    # style, with a deterministic serif fallback when it is unavailable.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.titlesize": 8.2,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    colors = {"queue": "#D55E00", "service": "#0072B2", "overall": "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(3.42, 2.22), gridspec_kw={"width_ratios": [1.18, 1.0]})
    ax = axes[0]
    labels = ["Queue/\nadmission", "Service\nTTFT", "Overall\nTTFT"]
    means = [row["mean_ms"] for row in stage_rows]
    p95s = [row["p95_ms"] for row in stage_rows]
    x = np.arange(3)
    width = 0.34
    ax.bar(x - width / 2, means, width, color=[colors["queue"], colors["service"], colors["overall"]], label="mean")
    ax.bar(x + width / 2, p95s, width, facecolor="white", edgecolor="#333333", linewidth=0.85, hatch="//", label="P95")
    ax.set_xticks(x, labels)
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("(a) Latency decomposition", fontweight="bold", pad=4)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper left", borderaxespad=0.2, handlelength=1.2, columnspacing=0.8)
    ymax = max(p95s) * 1.16 if p95s else 1.0
    ax.set_ylim(0, ymax)
    for xx, value in zip(x - width / 2, means):
        ax.text(xx, value + ymax * 0.018, f"{value/1000:.1f}s", ha="center", va="bottom", fontsize=6.8)

    ax = axes[1]
    counts = [len(startup_affected), frontier_rows, cold_reads]
    labels = ["scale-up\naffected", "frontier\nrows", "cold-read\nobservations"]
    bar_colors = ["#E69F00", "#56B4E9", "#CC79A7"]
    x = np.arange(3)
    bars = ax.bar(x, counts, color=bar_colors, width=0.62)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Count")
    ax.set_title("(b) Control-path evidence", fontweight="bold", pad=4)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(counts + [1]) * 1.25)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(counts + [1]) * 0.035, str(value), ha="center", va="bottom", fontsize=7.0)
    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.24, top=0.83, wspace=0.34)
    fig.savefig(out_dir / "d290_latency_control_diagnostic.pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(out_dir / "d290_latency_control_diagnostic.png", dpi=300, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

    manifest = {
        "figure": "d290_latency_control_diagnostic",
        "source_result": str(result_path),
        "source_sha256": sha256(result_path),
        "scenario": scenario.get("scenario_name"),
        "completed": scenario.get("completed"),
        "failed": scenario.get("failed"),
        "mean_overall_ttft_ms": scenario.get("avg_overall_ttft_ms"),
        "p95_overall_ttft_ms": scenario.get("p95_overall_ttft_ms"),
        "mean_service_ttft_ms": scenario.get("avg_service_ttft_ms"),
        "p95_service_ttft_ms": scenario.get("p95_service_ttft_ms"),
        "mean_queue_admission_ms": mean(queue) if queue else 0.0,
        "p95_queue_admission_ms": percentile(queue, 0.95),
        "scaleup_affected_count": len(startup_affected),
        "frontier_rows": frontier_rows,
        "cold_read_observations": cold_reads,
        "readiness_tier_counts": tiers,
        "formal_qualification": False,
        "interpretation": "diagnostic_only_startup_and_dispatch_dominate",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
