#!/usr/bin/env python3
"""Generate PrimeLoRA sensitivity figures from multiple completed rounds."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_paper_figures import (
    LEGEND_FONTSIZE,
    SYSTEM_COLORS,
    SYSTEM_LABELS,
    SYSTEM_ORDER,
    TICK_FONTSIZE,
    _as_float,
    _load_json,
    _main_round_data,
    _row_dict,
    _style_axes,
    _style_xgrid_axes,
    _system_key,
    _write_csv,
    _xlabel_with_panel,
)


LOWER_BETTER_METRICS = {
    "cost_req_usd",
    "ttft_avg_ms",
    "ttft_p95_ms",
    "e2e_avg_ms",
    "e2e_p95_ms",
    "tpot_avg_ms",
    "tpot_p95_ms",
}
HIGHER_BETTER_METRICS = {"ce", "tok_s"}
MATRIX_CELL_FONTSIZE = 8.0
MATRIX_TICK_FONTSIZE = 8.3
ADAPTER_POOL_MARKERS = {
    "faaslora": "D",
    "sglang": "s",
    "vllm": "^",
    "slora": "o",
    "serverlessllm": "X",
}
ADAPTER_POOL_LINESTYLES = {
    "faaslora": "-",
    "sglang": (0, (3.0, 1.4)),
    "vllm": (0, (1.2, 1.2)),
    "slora": (0, (4.0, 1.4, 1.2, 1.4)),
    "serverlessllm": "-.",
}


def _compare_json(round_dir: Path) -> Path:
    manifest = _load_json(round_dir / "MANIFEST.json")
    compare = manifest.get("compare_json")
    if compare and Path(str(compare)).exists():
        return Path(str(compare))
    matches = sorted((round_dir / "compare").glob("*five_system_compare.json"))
    if len(matches) != 1:
        raise SystemExit(f"{round_dir}: expected one five-system compare JSON, found {len(matches)}")
    return matches[0]


def _time_scale(round_dir: Path) -> float:
    manifest = _load_json(round_dir / "MANIFEST.json")
    run_tag = str(manifest.get("run_tag") or round_dir.name)
    match = re.search(r"_s([0-9]+(?:p[0-9]+)?)_", run_tag)
    if match:
        return float(match.group(1).replace("p", "."))
    env_path = round_dir / "round.env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("export SLLM_TIME_SCALE_FACTOR="):
                return float(line.split("=", 1)[1].strip())
    raise SystemExit(f"{round_dir}: cannot infer time scale from run_tag or round.env")


def _strict_rps(round_dir: Path) -> float:
    compare = _load_json(_compare_json(round_dir))
    headers = compare.get("strict_headers") or []
    rows = [_row_dict(headers, row) for row in compare.get("strict_rows") or []]
    for row in rows:
        if _system_key(str(row.get("System"))) == "faaslora":
            return _as_float(row.get("RPS"), f"{round_dir}.faaslora.RPS")
    raise SystemExit(f"{round_dir}: missing FaaSLoRA strict RPS")


def _adapter_pool_size(round_dir: Path) -> int:
    manifest = _load_json(round_dir / "MANIFEST.json")
    run_tag = str(manifest.get("run_tag") or round_dir.name)
    match = re.search(r"_a([0-9]+)_", run_tag)
    if match:
        return int(match.group(1))
    env_path = round_dir / "round.env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("export SLLM_SELECTED_NUM_ADAPTERS="):
                return int(line.split("=", 1)[1].strip())
    raise SystemExit(f"{round_dir}: cannot infer adapter pool size from run_tag or round.env")


def _collect(round_dirs: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for round_dir in round_dirs:
        systems = _main_round_data(round_dir)
        scale = _time_scale(round_dir)
        rps = _strict_rps(round_dir)
        for system in systems:
            row: Dict[str, Any] = {
                "round_dir": str(round_dir),
                "time_scale": scale,
                "nominal_rps": rps,
                "system_key": system.key,
                "system": system.label,
            }
            row.update(system.metrics)
            rows.append(row)
    rows.sort(key=lambda row: (row["nominal_rps"], SYSTEM_ORDER.index(row["system_key"])))
    return rows


def _collect_adapter_pool(round_dirs: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for round_dir in round_dirs:
        systems = _main_round_data(round_dir)
        adapter_pool = _adapter_pool_size(round_dir)
        scale = _time_scale(round_dir)
        for system in systems:
            row: Dict[str, Any] = {
                "round_dir": str(round_dir),
                "time_scale": scale,
                "adapter_pool_size": adapter_pool,
                "system_key": system.key,
                "system": system.label,
            }
            row.update(system.metrics)
            rows.append(row)
    rows.sort(key=lambda row: (row["adapter_pool_size"], SYSTEM_ORDER.index(row["system_key"])))
    return rows


def _series(rows: Sequence[Dict[str, Any]], system_key: str, metric: str) -> tuple[List[float], List[float]]:
    selected = [row for row in rows if row["system_key"] == system_key]
    selected.sort(key=lambda row: row["nominal_rps"])
    return [row["nominal_rps"] for row in selected], [row[metric] for row in selected]


def _series_by_adapter_pool(rows: Sequence[Dict[str, Any]], system_key: str, metric: str) -> tuple[List[float], List[float]]:
    selected = [row for row in rows if row["system_key"] == system_key]
    selected.sort(key=lambda row: row["adapter_pool_size"])
    return [row["adapter_pool_size"] for row in selected], [row[metric] for row in selected]


def _add_axis_arrows(ax: plt.Axes) -> None:
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", length=2.6, width=0.65, color="#333333")
    ax.annotate(
        "",
        xy=(1.025, 0.0),
        xytext=(0.0, 0.0),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "linewidth": 0.7, "color": "#333333", "shrinkA": 0.0, "shrinkB": 0.0},
        annotation_clip=False,
    )
    ax.annotate(
        "",
        xy=(0.0, 1.035),
        xytext=(0.0, 0.0),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "linewidth": 0.7, "color": "#333333", "shrinkA": 0.0, "shrinkB": 0.0},
        annotation_clip=False,
    )


def _plot_lines(
    ax: plt.Axes,
    rows: Sequence[Dict[str, Any]],
    systems: Sequence[str],
    metric: str,
    ylabel: str,
    panel_caption: str,
    *,
    scale: float = 1.0,
    xlabel: str = "Replay rate on 4 GPUs (req/s)",
    compact: bool = False,
) -> None:
    for key in systems:
        xs, ys = _series(rows, key, metric)
        ys = [value * scale for value in ys]
        label = SYSTEM_LABELS[key]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=2.9 if compact else 4.5,
            linewidth=1.05 if compact else 1.55,
            color=SYSTEM_COLORS[key],
            label=label,
        )
    _xlabel_with_panel(ax, xlabel, panel_caption)
    ax.set_ylabel(ylabel)
    loads = sorted({row["nominal_rps"] for row in rows})
    if len(loads) > 1:
        xpad = (max(loads) - min(loads)) * 0.16
        ax.set_xlim(min(loads) - xpad, max(loads) + xpad)
    ax.set_xticks(loads)
    ax.set_xticklabels([f"{load:.2f}" for load in loads])
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    _style_axes(ax)
    _add_axis_arrows(ax)


def _rank_shade(rank: int) -> str:
    if rank == 1:
        return "#B9DFBA"
    if rank == 2:
        return "#DCEEDC"
    if rank == 3:
        return "#F1F1F1"
    if rank == 4:
        return "#F7D9D7"
    return "#EFB3AF"


def _format_metric_value(value: float, fmt: str) -> str:
    if fmt in {"seconds", "one_decimal", "integer", "cost"}:
        return f"{value:.3f}"
    raise ValueError(f"unknown metric format {fmt!r}")


def _metric_load_matrix_panel(ax: plt.Axes, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metric_specs = [
        ("CE\n$1/(\\bar{L}\\bar{C})$", "ce", "higher_is_better", 1.0, "one_decimal"),
        ("Cost\n(mUSD)", "cost_req_usd", "lower_is_better", 1000.0, "cost"),
        ("TTFT\navg (s)", "ttft_avg_ms", "lower_is_better", 0.001, "seconds"),
        ("TTFT\np95 (s)", "ttft_p95_ms", "lower_is_better", 0.001, "seconds"),
        ("E2E\navg (s)", "e2e_avg_ms", "lower_is_better", 0.001, "seconds"),
        ("E2E\np95 (s)", "e2e_p95_ms", "lower_is_better", 0.001, "seconds"),
        ("TPOT\navg (ms)", "tpot_avg_ms", "lower_is_better", 1.0, "one_decimal"),
        ("TPOT\np95 (ms)", "tpot_p95_ms", "lower_is_better", 1.0, "one_decimal"),
        ("Throughput\n(tok/s)", "tok_s", "higher_is_better", 1.0, "one_decimal"),
    ]
    systems = [key for key in SYSTEM_ORDER if any(row["system_key"] == key for row in rows)]
    loads = sorted({float(row["nominal_rps"]) for row in rows})
    row_lookup = {(row["system_key"], float(row["nominal_rps"])): row for row in rows}
    display_values: Dict[tuple[str, str, float], float] = {}
    for system_key in systems:
        for _, metric, _, display_scale, _ in metric_specs:
            for load in loads:
                row = row_lookup[(system_key, load)]
                display_values[(system_key, metric, load)] = float(row[metric]) * display_scale

    row_specs = [(system_key, load) for system_key in systems for load in loads]
    out_rows: List[Dict[str, Any]] = []
    ax.set_xlim(0, len(metric_specs))
    ax.set_ylim(0, len(row_specs))
    for xi, (label, metric, direction, _, fmt) in enumerate(metric_specs):
        for yi, (system_key, load) in enumerate(row_specs):
            row = row_lookup[(system_key, load)]
            ordered_systems = sorted(
                systems,
                key=lambda key: display_values[(key, metric, load)],
                reverse=direction == "higher_is_better",
            )
            color_rank = {key: rank + 1 for rank, key in enumerate(ordered_systems)}
            value = display_values[(system_key, metric, load)]
            rect = plt.Rectangle((xi, yi), 1, 1, facecolor=_rank_shade(color_rank[system_key]), edgecolor="white", linewidth=0.8)
            ax.add_patch(rect)
            ax.text(
                xi + 0.5,
                yi + 0.5,
                _format_metric_value(value, fmt),
                ha="center",
                va="center",
                fontsize=MATRIX_CELL_FONTSIZE,
            )
            out_rows.append(
                {
                    "system_key": system_key,
                    "system": SYSTEM_LABELS[system_key],
                    "time_scale": row["time_scale"],
                    "load": f"{load:.2f}",
                    "load_rate_req_s": load,
                    "metric": metric,
                    "metric_label": label.replace("\n", " "),
                    "direction": direction,
                    "display_unit": (
                        "s"
                        if fmt == "seconds"
                        else ("milli-USD" if fmt == "cost" else ("tok/s" if metric == "tok_s" else "ms" if metric.startswith("tpot_") else "native"))
                    ),
                    "display_value": value,
                    "display_text": _format_metric_value(value, fmt),
                    "color_rank_within_metric_load": color_rank[system_key],
                }
            )
    ax.set_xticks(np.arange(len(metric_specs)) + 0.5, [item[0] for item in metric_specs])
    ylabels: List[str] = []
    for system_key, load in row_specs:
        load_label = f"{load:.2f}"
        ylabels.append(f"{SYSTEM_LABELS[system_key]} {load_label}" if load == loads[0] else f"  {load_label}")
    ax.set_yticks(np.arange(len(row_specs)) + 0.5, ylabels)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=MATRIX_TICK_FONTSIZE, length=0, pad=1.6)
    ax.tick_params(axis="y", labelsize=MATRIX_TICK_FONTSIZE, length=0, pad=1.0)
    for group_idx, system_key in enumerate(systems):
        if group_idx > 0:
            ax.axhline(group_idx * len(loads), color="white", linewidth=2.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _xlabel_with_panel(ax, "Rows list replay rate on 4 GPUs (req/s)", "(c) Metric values")
    ax.xaxis.label.set_size(TICK_FONTSIZE)
    return out_rows


def _write_main_metric_table(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    selected = sorted(rows, key=lambda row: (row["nominal_rps"], SYSTEM_ORDER.index(row["system_key"])))
    lines = [
        "% Auto-generated by scripts/plot_paper_sensitivity.py. Verify caption wording before final submission.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Operating-load sensitivity on the representative Llama-2 7B workload. Load is the average trace replay rate on the fixed 4-GPU testbed, not fleet-wide production QPS. TTFT, E2E, and TPOT are in milliseconds, throughput is in tok/s, and Cost/req is in USD. Lower is better for latency and cost; higher is better for throughput and CE.}",
        "\\label{tab:load_sensitivity_metrics}",
        "\\begin{tabular}{llrrrrrrrrr}",
        "\\hline",
        "Replay rate (req/s) & System & TTFT Avg (ms) & TTFT p95 (ms) & E2E Avg (ms) & E2E p95 (ms) & TPOT Avg (ms) & TPOT p95 (ms) & Throughput (tok/s) & Cost/req (USD) & CE \\\\",
        "\\hline",
    ]
    for row in selected:
        load = float(row["nominal_rps"])
        load_cell = f"{load:.2f}"
        lines.append(
            f"{load_cell} & {SYSTEM_LABELS[row['system_key']]} & "
            f"{row['ttft_avg_ms']:.0f} & {row['ttft_p95_ms']:.0f} & "
            f"{row['e2e_avg_ms']:.0f} & {row['e2e_p95_ms']:.0f} & "
            f"{row['tpot_avg_ms']:.1f} & {row['tpot_p95_ms']:.1f} & "
            f"{row['tok_s']:.1f} & {row['cost_req_usd']:.6f} & {row['ce']:.1f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_adapter_pool_metric_table(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    selected = sorted(rows, key=lambda row: (row["adapter_pool_size"], SYSTEM_ORDER.index(row["system_key"])))
    lines = [
        "% Auto-generated by scripts/plot_paper_sensitivity.py. Verify caption wording before final submission.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.0pt}",
        "\\caption{Adapter-pool sensitivity on the representative Llama-2 7B workload. All points use 4,000 requests, Zipf adapter popularity, the same replay scale, 100\\% LoRA-bound requests, and hot-set rotation every 500 requests. TTFT, E2E, and TPOT are in milliseconds, throughput is in tok/s, and Cost/req is in USD. Lower is better for latency and cost; higher is better for throughput and CE.}",
        "\\label{tab:adapter_pool_sensitivity_metrics}",
        "\\begin{tabular}{rlrrrrrrrrr}",
        "\\hline",
        "Adapters & System & TTFT Avg & TTFT p95 & E2E Avg & E2E p95 & TPOT Avg & TPOT p95 & Throughput (tok/s) & Cost/req & CE \\\\",
        "\\hline",
    ]
    for row in selected:
        lines.append(
            f"{int(row['adapter_pool_size'])} & {SYSTEM_LABELS[row['system_key']]} & "
            f"{row['ttft_avg_ms']:.0f} & {row['ttft_p95_ms']:.0f} & "
            f"{row['e2e_avg_ms']:.0f} & {row['e2e_p95_ms']:.0f} & "
            f"{row['tpot_avg_ms']:.1f} & {row['tpot_p95_ms']:.1f} & "
            f"{row['tok_s']:.1f} & {row['cost_req_usd']:.6f} & {row['ce']:.1f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _draw_load_trend_panels(
    fig: plt.Figure,
    axes: Sequence[plt.Axes],
    rows: Sequence[Dict[str, Any]],
    *,
    compact: bool = False,
) -> None:
    if compact:
        xlabel = "Replay rate\n(req/s)"
        _plot_lines(axes[0], rows, SYSTEM_ORDER, "ce", "CE", "(a) CE", xlabel=xlabel, compact=True)
        _plot_lines(axes[1], rows, SYSTEM_ORDER, "cost_req_usd", "Cost/req\n(mUSD)", "(b) Cost", scale=1000.0, xlabel=xlabel, compact=True)
        for ax in axes:
            ax.xaxis.label.set_size(7.0)
            ax.yaxis.label.set_size(7.1)
            ax.tick_params(axis="both", labelsize=6.8)
            ax.yaxis.labelpad = 1.0
    else:
        _plot_lines(axes[0], rows, SYSTEM_ORDER, "ce", "CE (higher is better)", "(a) CE vs load")
        _plot_lines(axes[1], rows, SYSTEM_ORDER, "cost_req_usd", "Cost/req (mUSD)", "(b) Cost vs load", scale=1000.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=6.5 if compact else 8.4,
        ncols=3 if compact else 5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=0.8 if compact else 1.3,
        handlelength=1.2 if compact else 1.6,
    )


def _plot_adapter_pool_lines(
    ax: plt.Axes,
    rows: Sequence[Dict[str, Any]],
    systems: Sequence[str],
    metric: str,
    ylabel: str,
    panel_caption: str,
    *,
    scale: float = 1.0,
    compact: bool = False,
) -> None:
    for key in systems:
        xs, ys = _series_by_adapter_pool(rows, key, metric)
        ys = [value * scale for value in ys]
        ax.plot(
            xs,
            ys,
            marker=ADAPTER_POOL_MARKERS[key],
            markersize=3.15 if compact else 4.7,
            linewidth=1.05 if compact else 1.55,
            linestyle=ADAPTER_POOL_LINESTYLES[key],
            color=SYSTEM_COLORS[key],
            markerfacecolor="white",
            markeredgecolor=SYSTEM_COLORS[key],
            markeredgewidth=0.85 if compact else 1.0,
            label=SYSTEM_LABELS[key],
        )
    pools = sorted({int(row["adapter_pool_size"]) for row in rows})
    _xlabel_with_panel(ax, "Adapters", panel_caption)
    ax.set_ylabel(ylabel)
    if len(pools) > 1:
        xpad = (max(pools) - min(pools)) * 0.08
        ax.set_xlim(min(pools) - xpad, max(pools) + xpad)
    ax.set_xticks(pools)
    ax.set_xticklabels([str(pool) for pool in pools])
    ax.tick_params(axis="both", labelsize=6.8 if compact else TICK_FONTSIZE)
    if compact:
        ax.xaxis.label.set_size(7.0)
        ax.yaxis.label.set_size(7.1)
        ax.yaxis.labelpad = 1.0
    _style_axes(ax)
    _add_axis_arrows(ax)


def plot_adapter_pool_sensitivity(round_dirs: Sequence[Path], out_dir: Path) -> None:
    rows = _collect_adapter_pool([Path(path).resolve() for path in round_dirs])
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(3.62, 1.76), constrained_layout=False)
    _plot_adapter_pool_lines(axes[0], rows, SYSTEM_ORDER, "ce", "CE", "(a) CE", compact=True)
    _plot_adapter_pool_lines(
        axes[1],
        rows,
        SYSTEM_ORDER,
        "cost_req_usd",
        "Cost/req\n(mUSD)",
        "(b) Cost",
        scale=1000.0,
        compact=True,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=6.5,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=0.8,
        handlelength=1.2,
    )
    fig.subplots_adjust(left=0.15, right=0.99, top=0.70, bottom=0.34, wspace=0.42)

    pdf = out_dir / "fig9_adapter_pool_sensitivity.pdf"
    csv_path = out_dir / "fig9_adapter_pool_sensitivity_data.csv"
    table_path = out_dir / "table_fig9_adapter_pool_sensitivity_metrics.tex"
    manifest = out_dir / "fig9_adapter_pool_sensitivity_manifest.json"
    fig.savefig(pdf)
    plt.close(fig)
    _write_csv(csv_path, rows)
    _write_adapter_pool_metric_table(table_path, rows)
    manifest_payload = {
        "figure": "fig9_adapter_pool_sensitivity",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pdf": str(pdf),
        "csv": str(csv_path),
        "table_tex": str(table_path),
        "round_dirs": [str(Path(path).resolve()) for path in round_dirs],
        "note": "Adapter-pool sensitivity combines the completed 100/200/300/400-adapter queue with the closed 500-adapter Llama-2 7B main round. The figure follows the scaling style used in multi-LoRA systems papers: adapter pool size is the x-axis, and all five systems are shown on the same axes. The full metric table reports TTFT, E2E, TPOT, throughput, Cost/req, and CE.",
    }
    manifest.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_load_sensitivity(round_dirs: Sequence[Path], out_dir: Path) -> None:
    rows = _collect([Path(path).resolve() for path in round_dirs])
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.16, 4.60), constrained_layout=False)
    gridspec = fig.add_gridspec(2, 2, height_ratios=[0.95, 2.15], width_ratios=[1.0, 1.0])
    axes = [fig.add_subplot(gridspec[0, 0]), fig.add_subplot(gridspec[0, 1]), fig.add_subplot(gridspec[1, :])]
    _draw_load_trend_panels(fig, axes[:2], rows)
    value_rows = _metric_load_matrix_panel(axes[2], rows)
    fig.subplots_adjust(left=0.13, right=0.995, top=0.91, bottom=0.14, hspace=0.57, wspace=0.16)

    trend_fig, trend_axes = plt.subplots(1, 2, figsize=(3.62, 1.76), constrained_layout=False)
    _draw_load_trend_panels(trend_fig, trend_axes, rows, compact=True)
    trend_fig.subplots_adjust(left=0.15, right=0.99, top=0.70, bottom=0.34, wspace=0.42)

    pdf = out_dir / "fig8_load_sensitivity.pdf"
    trend_pdf = out_dir / "fig8_load_sensitivity_trends.pdf"
    csv_path = out_dir / "fig8_load_sensitivity_data.csv"
    value_csv_path = out_dir / "fig8_load_sensitivity_metric_values.csv"
    table_path = out_dir / "table_fig8_load_sensitivity_metrics.tex"
    manifest = out_dir / "fig8_load_sensitivity_manifest.json"
    fig.savefig(pdf)
    plt.close(fig)
    trend_fig.savefig(trend_pdf)
    plt.close(trend_fig)
    _write_csv(csv_path, rows)
    _write_csv(value_csv_path, value_rows)
    _write_main_metric_table(table_path, rows)
    manifest_payload = {
        "figure": "fig8_load_sensitivity",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pdf": str(pdf),
        "trend_pdf": str(trend_pdf),
        "csv": str(csv_path),
        "metric_values_csv": str(value_csv_path),
        "table_tex": str(table_path),
        "round_dirs": [str(Path(path).resolve()) for path in round_dirs],
        "note": "Panels (a) and (b) compare all five systems on CE and lifecycle cost across average trace replay rates of 0.67, 0.81, and 1.01 req/s on the fixed 4-GPU testbed. This is deployment-local replay intensity, not fleet-wide production QPS. Panel (c) reports concrete metric values for all five systems and all three load points; all cell values are rounded to three decimals, and colors mark within-metric, within-load favorability with lower-is-better for latency/cost and higher-is-better for CE/throughput. Units: TTFT/E2E in seconds, TPOT in ms, throughput in tok/s, and cost in milli-USD. The accompanying table reports all primary metrics for all systems and load points. The trend_pdf contains only panels (a) and (b) for a cleaner main-text option paired with the LaTeX table.",
    }
    manifest.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PrimeLoRA sensitivity figures from completed rounds.")
    parser.add_argument("--round-dir", action="append", required=True, type=Path, help="Completed fair round directory; pass once per load point.")
    parser.add_argument("--out-dir", type=Path, default=Path("figs/paper/sensitivity"))
    parser.add_argument("--figure", choices=["load", "adapter_pool"], default="load")
    args = parser.parse_args()
    if args.figure == "adapter_pool":
        plot_adapter_pool_sensitivity(args.round_dir, args.out_dir.resolve())
        print(f"generated fig9_adapter_pool_sensitivity -> {args.out_dir.resolve()}")
    else:
        plot_load_sensitivity(args.round_dir, args.out_dir.resolve())
        print(f"generated fig8_load_sensitivity -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
