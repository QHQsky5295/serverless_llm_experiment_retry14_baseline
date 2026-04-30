#!/usr/bin/env python3
"""Analyze selected-replica adapter readiness from completed FaaSLoRA rounds."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCENARIO_ORDER = ("faaslora_nvme", "faaslora_no_coord", "faaslora_full")
SCENARIO_LABELS = {
    "faaslora_nvme": "PrimeLoRA-NVMe",
    "faaslora_no_coord": "PrimeLoRA-NoCoord",
    "faaslora_full": "PrimeLoRA",
}
SHORT_LABELS = {
    "faaslora_nvme": "NVMe",
    "faaslora_no_coord": "NoCoord",
    "faaslora_full": "PrimeLoRA",
}
TIER_ORDER = ("gpu", "host", "nvme", "remote")
TIER_LABELS = {
    "gpu": "GPU-ready",
    "host": "HOST-prepared",
    "nvme": "NVMe-prepared",
    "remote": "Remote-cold",
}
TIER_COLORS = {
    "gpu": "#78B87A",
    "host": "#7FA7D9",
    "nvme": "#F2B36D",
    "remote": "#C84D4D",
}
METRIC_CMAP = "YlGnBu"


@dataclass
class ScenarioRecords:
    name: str
    source: Path
    records: List[Dict[str, Any]]
    used_dispatch_before_tier: bool


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 8.8,
            "axes.linewidth": 0.65,
            "grid.linewidth": 0.55,
            "lines.linewidth": 1.35,
            "patch.linewidth": 0.45,
        }
    )


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def percentile(values: Sequence[float], q: float) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.percentile(np.asarray(vals, dtype=float), q))


def mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.mean(np.asarray(vals, dtype=float)))


def readiness_tier(record: Dict[str, Any]) -> str:
    tier = str(record.get("readiness_tier_before_dispatch") or "").strip().lower()
    if not tier:
        tier = str(record.get("cache_tier") or "unknown").strip().lower()
    return tier


def ttft_ms(record: Dict[str, Any]) -> float:
    return as_float(record.get("overall_ttft_ms"), as_float(record.get("ttft_ms"), 0.0))


def adapter_prep_ms(record: Dict[str, Any]) -> float:
    return as_float(record.get("lora_io_ms")) + as_float(record.get("defer_ms"))


def runtime_ttft_ms(record: Dict[str, Any]) -> float:
    return as_float(record.get("vllm_ttft_ms"), as_float(record.get("service_ttft_ms"), 0.0))


def dispatch_wait_ms(record: Dict[str, Any]) -> float:
    return sum(
        as_float(record.get(key))
        for key in (
            "ingress_queue_wait_ms",
            "dispatch_admission_wait_ms",
            "dispatch_window_wait_ms",
            "runtime_slot_wait_ms",
        )
    )


def load_scenarios(input_path: Path) -> List[ScenarioRecords]:
    if input_path.is_file():
        candidates = [input_path]
    else:
        candidates = sorted(input_path.rglob("*_result.json"))
    scenarios: Dict[str, ScenarioRecords] = {}
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        detailed = payload.get("detailed_results")
        if not isinstance(detailed, dict):
            continue
        for name, detail in detailed.items():
            if not isinstance(detail, dict) or not isinstance(detail.get("requests"), list):
                continue
            records = [
                dict(item)
                for item in detail["requests"]
                if as_bool(item.get("success")) and str(item.get("adapter_id") or "").strip()
            ]
            if not records:
                continue
            used_dispatch_before_tier = any(
                str(item.get("readiness_tier_before_dispatch") or "").strip()
                for item in records
            )
            scenarios[name] = ScenarioRecords(
                name=name,
                source=path,
                records=records,
                used_dispatch_before_tier=used_dispatch_before_tier,
            )
    ordered = [scenarios[name] for name in SCENARIO_ORDER if name in scenarios]
    ordered.extend(v for k, v in sorted(scenarios.items()) if k not in SCENARIO_ORDER)
    if not ordered:
        raise SystemExit(f"no FaaSLoRA request records found under {input_path}")
    return ordered


def build_summary(scenarios: Sequence[ScenarioRecords]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        records = scenario.records
        n = len(records)
        tier_counts = {tier: 0 for tier in TIER_ORDER}
        for record in records:
            tier = readiness_tier(record)
            if tier in tier_counts:
                tier_counts[tier] += 1
        gpu_records = [r for r in records if readiness_tier(r) == "gpu"]
        mismatch_records = [r for r in records if readiness_tier(r) != "gpu"]
        scaleout_first = [r for r in records if as_bool(r.get("scaleup_first_service"))]
        scaleout_mismatch = [
            r
            for r in records
            if as_bool(r.get("scaleup_first_service"))
            and not as_bool(r.get("scaleup_planned_adapter_match"))
        ]
        rows.append(
            {
                "scenario": scenario.name,
                "label": SCENARIO_LABELS.get(scenario.name, scenario.name),
                "short_label": SHORT_LABELS.get(scenario.name, scenario.name),
                "source": str(scenario.source),
                "n": n,
                "used_dispatch_before_tier": scenario.used_dispatch_before_tier,
                "gpu_ready_pct": 100.0 * tier_counts["gpu"] / n,
                "host_pct": 100.0 * tier_counts["host"] / n,
                "nvme_pct": 100.0 * tier_counts["nvme"] / n,
                "host_nvme_pct": 100.0 * (tier_counts["host"] + tier_counts["nvme"]) / n,
                "remote_cold_pct": 100.0 * tier_counts["remote"] / n,
                "mismatch_pct": 100.0 * len(mismatch_records) / n,
                "ttft_p50_gpu_ready_ms": percentile([ttft_ms(r) for r in gpu_records], 50),
                "ttft_p95_gpu_ready_ms": percentile([ttft_ms(r) for r in gpu_records], 95),
                "ttft_p50_mismatch_ms": percentile([ttft_ms(r) for r in mismatch_records], 50),
                "ttft_p95_mismatch_ms": percentile([ttft_ms(r) for r in mismatch_records], 95),
                "adapter_prep_p95_all_ms": percentile([adapter_prep_ms(r) for r in records], 95),
                "adapter_prep_p95_mismatch_ms": percentile(
                    [adapter_prep_ms(r) for r in mismatch_records], 95
                ),
                "dispatch_wait_p95_mismatch_ms": percentile(
                    [dispatch_wait_ms(r) for r in mismatch_records], 95
                ),
                "runtime_ttft_p95_mismatch_ms": percentile(
                    [runtime_ttft_ms(r) for r in mismatch_records], 95
                ),
                "scaleout_first_service_n": len(scaleout_first),
                "scaleout_mismatch_n": len(scaleout_mismatch),
            }
        )
    return rows


def build_by_tier(scenarios: Sequence[ScenarioRecords]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        records = scenario.records
        n = len(records)
        for tier in TIER_ORDER:
            group = [r for r in records if readiness_tier(r) == tier]
            if not group:
                continue
            rows.append(
                {
                    "scenario": scenario.name,
                    "label": SCENARIO_LABELS.get(scenario.name, scenario.name),
                    "tier": tier,
                    "tier_label": TIER_LABELS[tier],
                    "n": len(group),
                    "share_pct": 100.0 * len(group) / n,
                    "ttft_p50_ms": percentile([ttft_ms(r) for r in group], 50),
                    "ttft_p95_ms": percentile([ttft_ms(r) for r in group], 95),
                    "adapter_prep_p95_ms": percentile([adapter_prep_ms(r) for r in group], 95),
                    "dispatch_wait_p95_ms": percentile([dispatch_wait_ms(r) for r in group], 95),
                    "runtime_ttft_p95_ms": percentile([runtime_ttft_ms(r) for r in group], 95),
                    "ttft_mean_ms": mean([ttft_ms(r) for r in group]),
                    "adapter_prep_mean_ms": mean([adapter_prep_ms(r) for r in group]),
                    "dispatch_wait_mean_ms": mean([dispatch_wait_ms(r) for r in group]),
                    "runtime_ttft_mean_ms": mean([runtime_ttft_ms(r) for r in group]),
                }
            )
    return rows


def build_scaleout(scenarios: Sequence[ScenarioRecords]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        records = scenario.records
        first = [r for r in records if as_bool(r.get("scaleup_first_service"))]
        match = [r for r in first if as_bool(r.get("scaleup_planned_adapter_match"))]
        miss = [r for r in first if not as_bool(r.get("scaleup_planned_adapter_match"))]
        rows.append(
            {
                "scenario": scenario.name,
                "label": SCENARIO_LABELS.get(scenario.name, scenario.name),
                "first_service_n": len(first),
                "planned_match_n": len(match),
                "planned_miss_n": len(miss),
                "planned_match_rate_pct": 100.0 * len(match) / len(first) if first else float("nan"),
                "gpu_ready_rate_pct": 100.0
                * sum(1 for r in first if readiness_tier(r) == "gpu")
                / len(first)
                if first
                else float("nan"),
                "remote_cold_rate_pct": 100.0
                * sum(1 for r in first if readiness_tier(r) == "remote")
                / len(first)
                if first
                else float("nan"),
                "ttft_p50_match_ms": percentile([ttft_ms(r) for r in match], 50),
                "ttft_p95_match_ms": percentile([ttft_ms(r) for r in match], 95),
                "ttft_p50_miss_ms": percentile([ttft_ms(r) for r in miss], 50),
                "ttft_p95_miss_ms": percentile([ttft_ms(r) for r in miss], 95),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_num(value: Any, digits: int = 1) -> str:
    try:
        val = float(value)
    except Exception:
        return "--"
    if not math.isfinite(val):
        return "--"
    return f"{val:.{digits}f}"


def write_latex_table(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    include_remote = bool(
        rows and max(float(row.get("remote_cold_pct", 0.0) or 0.0) for row in rows) > 0.0
    )
    colspec = "lrrrrrrr" if include_remote else "lrrrrrr"
    header = (
        r"System & \shortstack{GPU\\(\%)} & \shortstack{HOST/NVMe\\(\%)} & "
    )
    if include_remote:
        header += r"\shortstack{Remote\\(\%)} & "
    header += (
        r"\shortstack{Mismatch\\(\%)} & \shortstack{TTFT p95\\GPU (ms)} & "
        r"\shortstack{TTFT p95\\Non-GPU (ms)} & \shortstack{Prep p95\\Non-GPU (ms)} \\"
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Service-readiness analysis on the representative Llama-2-7B multi-LoRA workload.}",
        r"\label{tab:service_readiness}",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\hline",
        header,
        r"\hline",
    ]
    for row in rows:
        cells = [
            str(row["label"]),
            fmt_num(row["gpu_ready_pct"], 2),
            fmt_num(row["host_nvme_pct"], 2),
        ]
        if include_remote:
            cells.append(fmt_num(row["remote_cold_pct"], 2))
        cells.extend(
            [
                fmt_num(row["mismatch_pct"], 2),
                fmt_num(row["ttft_p95_gpu_ready_ms"], 1),
                fmt_num(row["ttft_p95_mismatch_ms"], 1),
                fmt_num(row["adapter_prep_p95_mismatch_ms"], 1),
            ]
        )
        lines.append(
            " & ".join(cells)
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_summary(out_path: Path, summary_rows: Sequence[Dict[str, Any]]) -> None:
    labels = [str(row["short_label"]) for row in summary_rows]
    y = np.arange(len(summary_rows))
    fig = plt.figure(figsize=(3.45, 4.15))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.12],
        width_ratios=[1.04, 1.0],
        hspace=0.70,
        wspace=0.34,
    )

    ax_gpu = fig.add_subplot(grid[0, 0])
    ax_mix = fig.add_subplot(grid[0, 1], sharey=ax_gpu)
    ax_tail = fig.add_subplot(grid[1, :])

    gpu_vals = [float(row["gpu_ready_pct"]) for row in summary_rows]
    ax_gpu.barh(y, gpu_vals, height=0.55, color=TIER_COLORS["gpu"], edgecolor="white", linewidth=0.55)
    ax_gpu.set_yticks(y)
    ax_gpu.set_yticklabels(labels)
    ax_gpu.invert_yaxis()
    ax_gpu.set_xlim(0, 100)
    ax_gpu.set_xlabel("GPU-ready (%)", labelpad=2)
    ax_gpu.grid(axis="x", alpha=0.25)
    for yi, value in zip(y, gpu_vals):
        ax_gpu.text(value - 1.2, yi, f"{value:.2f}", ha="right", va="center", fontsize=8.6, color="white")

    bottoms = np.zeros(len(summary_rows))
    tier_values: Dict[str, List[float]] = {}
    for tier in TIER_ORDER:
        vals = []
        for row in summary_rows:
            if tier == "gpu":
                vals.append(float(row["gpu_ready_pct"]))
            elif tier == "host":
                vals.append(float(row["host_pct"]))
            elif tier == "nvme":
                vals.append(float(row["nvme_pct"]))
            else:
                vals.append(float(row["remote_cold_pct"]))
        tier_values[tier] = vals
    visible_tiers = [
        tier
        for tier in ("host", "nvme", "remote")
        if tier != "remote" or max(tier_values[tier]) > 0.0
    ]
    for tier in visible_tiers:
        vals = tier_values[tier]
        ax_mix.barh(
            y,
            vals,
            left=bottoms,
            height=0.55,
            color=TIER_COLORS[tier],
            edgecolor="white",
            linewidth=0.55,
            label=TIER_LABELS[tier],
        )
        for yi, left, value in zip(y, bottoms, vals):
            if value <= 0:
                continue
            x_mid = left + value / 2.0
            if value >= 0.55:
                ax_mix.text(
                    x_mid,
                    yi,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color="#1F2A35",
                )
            else:
                ax_mix.text(
                    left + value + 0.08,
                    yi,
                    f"{value:.2f}",
                    ha="left",
                    va="center",
                    fontsize=7.8,
                    color="#1F2A35",
                )
        bottoms += np.asarray(vals)
    xmax_mix = max(5.6, float(max(bottoms)) * 1.28)
    ax_mix.set_xlim(0, xmax_mix)
    ax_mix.set_xlabel("Non-GPU tier (%)", labelpad=2)
    ax_mix.grid(axis="x", alpha=0.25)
    ax_mix.tick_params(axis="y", left=False, labelleft=False)
    ax_mix.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    fig.text(0.5, 0.545, "(a) Selected-replica adapter-readiness distribution", ha="center", va="top", fontsize=10.0)

    ax = ax_tail
    gpu_s = [float(row["ttft_p95_gpu_ready_ms"]) / 1000.0 for row in summary_rows]
    mis_s = [float(row["ttft_p95_mismatch_ms"]) / 1000.0 for row in summary_rows]
    for yi, gv, mv in zip(y, gpu_s, mis_s):
        ax.plot([gv, mv], [yi, yi], color="#A0A0A0", lw=1.0, zorder=1)
    ax.scatter(gpu_s, y, marker="o", s=34, color=TIER_COLORS["gpu"], label="GPU-ready", zorder=2)
    ax.scatter(mis_s, y, marker="D", s=34, color="#C74343", label="Non-GPU", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("TTFT p95 (s)")
    ax.grid(axis="x", alpha=0.25)
    xmax = max(mis_s + gpu_s) * 1.65
    ax.set_xlim(0, xmax)
    ax.set_ylim(len(summary_rows) - 0.45, -0.45)
    ax.legend(
        loc="center right",
        bbox_to_anchor=(0.98, 0.50),
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#D0D0D0",
        fontsize=8.2,
        handlelength=1.25,
        handletextpad=0.45,
        borderpad=0.35,
    )
    for yi, gv, mv in zip(y, gpu_s, mis_s):
        ax.text(gv + xmax * 0.018, yi - 0.14, f"{gv:.2f}", fontsize=8.8, color="#2F6F3E")
        ax.text(mv + xmax * 0.018, yi - 0.14, f"{mv:.2f}", fontsize=8.8, color="#8A2E2E")
    ax.text(0.5, -0.36, "(b) Tail penalty when not GPU-ready", transform=ax.transAxes, ha="center", va="top", fontsize=10.0)

    fig.subplots_adjust(left=0.22, right=0.98, top=0.92, bottom=0.16)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_mechanism_matrix(out_path: Path, summary_rows: Sequence[Dict[str, Any]]) -> None:
    labels = [str(row["short_label"]) for row in summary_rows]
    metric_rows = [
        ("Non-GPU dispatch (%)", [float(row["mismatch_pct"]) for row in summary_rows], "{:.2f}"),
        ("Prep p95, mismatch (ms)", [float(row["adapter_prep_p95_mismatch_ms"]) for row in summary_rows], "{:.1f}"),
        ("Mismatch TTFT p95 (s)", [float(row["ttft_p95_mismatch_ms"]) / 1000.0 for row in summary_rows], "{:.2f}"),
    ]
    remote_values = [float(row["remote_cold_pct"]) for row in summary_rows]
    if max(remote_values) > 0.0:
        metric_rows.insert(1, ("Remote-cold (%)", remote_values, "{:.2f}"))
    raw = np.asarray([vals for _, vals, _ in metric_rows], dtype=float)
    norm = np.zeros_like(raw)
    for i in range(raw.shape[0]):
        row = raw[i]
        lo, hi = float(np.nanmin(row)), float(np.nanmax(row))
        norm[i] = 0.45 if abs(hi - lo) < 1e-12 else (row - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(3.45, 3.1))
    ax.imshow(norm, cmap=METRIC_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(metric_rows)))
    ax.set_yticklabels([item[0] for item in metric_rows])
    for i, (_, vals, fmt) in enumerate(metric_rows):
        for j, value in enumerate(vals):
            text_color = "white" if norm[i, j] > 0.62 else "#1F2A35"
            ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=9.4, color=text_color)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(metric_rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)
    ax.text(0.5, -0.20, "Lower is better for all cells", transform=ax.transAxes, ha="center", va="top", fontsize=9.4)
    fig.subplots_adjust(left=0.48, right=0.99, top=0.96, bottom=0.16)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_full_cdf(out_path: Path, scenarios: Sequence[ScenarioRecords]) -> bool:
    full = next((scenario for scenario in scenarios if scenario.name == "faaslora_full"), None)
    if full is None:
        return False
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    plotted = False
    for tier in TIER_ORDER:
        group = [ttft_ms(r) / 1000.0 for r in full.records if readiness_tier(r) == tier]
        if len(group) < 5:
            continue
        values = np.sort(np.asarray(group, dtype=float))
        cdf = np.arange(1, len(values) + 1) / len(values)
        p95 = percentile(values, 95)
        ax.plot(values, cdf, label=f"{TIER_LABELS[tier]} (n={len(values)}, p95={p95:.2f}s)", color=TIER_COLORS[tier], lw=1.35)
        ax.axvline(p95, color=TIER_COLORS[tier], lw=0.8, ls="--", alpha=0.55)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel("TTFT (s)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=True, framealpha=0.94, fontsize=7.6)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.98, bottom=0.22)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def write_manifest(
    out_path: Path,
    scenarios: Sequence[ScenarioRecords],
    generated: Sequence[str],
    skipped: Dict[str, str],
) -> None:
    payload = {
        "analysis": "service_readiness",
        "input_sources": [
            {
                "scenario": scenario.name,
                "source": str(scenario.source),
                "requests_used": len(scenario.records),
            }
            for scenario in scenarios
        ],
        "filter": "success == True and adapter_id non-empty",
        "tier_field": (
            "readiness_tier_before_dispatch when present; otherwise cache_tier"
        ),
        "dispatch_before_tier_available": all(s.used_dispatch_before_tier for s in scenarios),
        "field_caveat": (
            "Current archived results do not contain readiness_tier_before_dispatch. "
            "cache_tier is used as a selected-replica service-time readiness proxy."
        ),
        "generated": list(generated),
        "skipped": skipped,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="FaaSLoRA result file or completed round directory")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    args = parser.parse_args()

    configure_matplotlib()
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir = out_dir / "tables"

    scenarios = load_scenarios(args.input)
    summary_rows = build_summary(scenarios)
    tier_rows = build_by_tier(scenarios)
    scaleout_rows = build_scaleout(scenarios)

    write_csv(out_dir / "service_readiness_summary.csv", summary_rows)
    write_csv(out_dir / "service_readiness_by_tier.csv", tier_rows)
    write_csv(out_dir / "scaleout_first_service_summary.csv", scaleout_rows)
    write_latex_table(table_dir / "table_service_readiness.tex", summary_rows)

    generated: List[str] = []
    plot_summary(out_dir / "fig_service_readiness_summary.pdf", summary_rows)
    generated.append("fig_service_readiness_summary.pdf")
    plot_mechanism_matrix(out_dir / "fig_mechanism_gap_ablation.pdf", summary_rows)
    generated.append("fig_mechanism_gap_ablation.pdf")
    if plot_full_cdf(out_dir / "fig_ttft_breakdown_readiness.pdf", scenarios):
        generated.append("fig_ttft_breakdown_readiness.pdf")

    skipped: Dict[str, str] = {}
    min_first_service = min((int(row["first_service_n"]) for row in scaleout_rows), default=0)
    if min_first_service < 20:
        skipped["fig_scaleout_first_service.pdf"] = (
            "Skipped as a paper figure because scaleup_first_service has fewer than 20 samples "
            f"per scenario (minimum observed n={min_first_service})."
        )
    skipped["fig_control_plane_overhead.pdf"] = (
        "Skipped because archived results do not contain routing_decision_us, "
        "gpu_admission_decision_us, or control_plane_total_us."
    )
    (out_dir / "service_readiness_warnings.txt").write_text(
        "\n".join(f"{name}: {reason}" for name, reason in skipped.items()) + "\n",
        encoding="utf-8",
    )
    write_manifest(out_dir / "service_readiness_manifest.json", scenarios, generated, skipped)

    paper_fig_dir = Path("figs")
    paper_fig_dir.mkdir(parents=True, exist_ok=True)
    for fig_name in generated:
        src = out_dir / fig_name
        if src.exists():
            shutil.copy2(src, paper_fig_dir / fig_name)

    print("Service-readiness summary")
    for row in summary_rows:
        print(
            f"  {row['label']}: gpu={row['gpu_ready_pct']:.2f}% "
            f"mismatch={row['mismatch_pct']:.2f}% "
            f"ttft_p95_gpu={row['ttft_p95_gpu_ready_ms']:.1f}ms "
            f"ttft_p95_mismatch={row['ttft_p95_mismatch_ms']:.1f}ms"
        )
    print(f"wrote outputs -> {out_dir}")


if __name__ == "__main__":
    main()
