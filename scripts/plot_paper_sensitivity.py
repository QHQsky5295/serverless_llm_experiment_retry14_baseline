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


def _series(rows: Sequence[Dict[str, Any]], system_key: str, metric: str) -> tuple[List[float], List[float]]:
    selected = [row for row in rows if row["system_key"] == system_key]
    selected.sort(key=lambda row: row["nominal_rps"])
    return [row["nominal_rps"] for row in selected], [row[metric] for row in selected]


def _load_label_map(rows: Sequence[Dict[str, Any]]) -> Dict[float, str]:
    loads = sorted({float(row["nominal_rps"]) for row in rows})
    if len(loads) == 3:
        names = ["Low", "Medium", "Nominal"]
    else:
        names = [f"Load {idx + 1}" for idx in range(len(loads))]
    return {load: name for load, name in zip(loads, names)}


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
) -> None:
    for key in systems:
        xs, ys = _series(rows, key, metric)
        ys = [value * scale for value in ys]
        label = SYSTEM_LABELS[key]
        ax.plot(xs, ys, marker="o", markersize=4.5, linewidth=1.55, color=SYSTEM_COLORS[key], label=label)
    _xlabel_with_panel(ax, "Offered load (req/s)", panel_caption)
    ax.set_ylabel(ylabel)
    loads = sorted({row["nominal_rps"] for row in rows})
    load_labels = _load_label_map(rows)
    if len(loads) > 1:
        xpad = (max(loads) - min(loads)) * 0.16
        ax.set_xlim(min(loads) - xpad, max(loads) + xpad)
    ax.set_xticks(loads)
    ax.set_xticklabels([f"{load_labels[load]}\n{load:.2f}" for load in loads])
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
        ("CE", "ce", "higher_is_better", 1.0, "one_decimal"),
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
    load_labels = _load_label_map(rows)
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
                    "load": load_labels[load],
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
        load_label = load_labels[load]
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
    _xlabel_with_panel(ax, "Rows use Low, Medium, and Nominal load points", "(c) Metric values")
    ax.xaxis.label.set_size(TICK_FONTSIZE)
    return out_rows


def _write_main_metric_table(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    selected = sorted(rows, key=lambda row: (row["nominal_rps"], SYSTEM_ORDER.index(row["system_key"])))
    load_labels = _load_label_map(rows)
    lines = [
        "% Auto-generated by scripts/plot_paper_sensitivity.py. Verify caption wording before final submission.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Operating-load sensitivity on the representative Llama-2 7B workload. Low, Medium, and Nominal denote average arrival rates of 0.67, 0.81, and 1.01 req/s on the 4-GPU testbed. TTFT, E2E, and TPOT are in milliseconds, throughput is in tok/s, and Cost/req is in USD. Lower is better for latency and cost; higher is better for throughput and CE.}",
        "\\label{tab:load_sensitivity_metrics}",
        "\\begin{tabular}{llrrrrrrrrr}",
        "\\hline",
        "Load & System & TTFT Avg (ms) & TTFT p95 (ms) & E2E Avg (ms) & E2E p95 (ms) & TPOT Avg (ms) & TPOT p95 (ms) & Throughput (tok/s) & Cost/req (USD) & CE \\\\",
        "\\hline",
    ]
    for row in selected:
        load = float(row["nominal_rps"])
        load_cell = f"{load_labels[load]} ({load:.2f})"
        lines.append(
            f"{load_cell} & {SYSTEM_LABELS[row['system_key']]} & "
            f"{row['ttft_avg_ms']:.0f} & {row['ttft_p95_ms']:.0f} & "
            f"{row['e2e_avg_ms']:.0f} & {row['e2e_p95_ms']:.0f} & "
            f"{row['tpot_avg_ms']:.1f} & {row['tpot_p95_ms']:.1f} & "
            f"{row['tok_s']:.1f} & {row['cost_req_usd']:.6f} & {row['ce']:.1f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_load_sensitivity(round_dirs: Sequence[Path], out_dir: Path) -> None:
    rows = _collect([Path(path).resolve() for path in round_dirs])
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.16, 4.60), constrained_layout=False)
    gridspec = fig.add_gridspec(2, 2, height_ratios=[0.95, 2.15], width_ratios=[1.0, 1.0])
    axes = [fig.add_subplot(gridspec[0, 0]), fig.add_subplot(gridspec[0, 1]), fig.add_subplot(gridspec[1, :])]
    _plot_lines(axes[0], rows, SYSTEM_ORDER, "ce", "CE (higher is better)", "(a) CE vs load")
    _plot_lines(axes[1], rows, SYSTEM_ORDER, "cost_req_usd", "Cost/req (mUSD)", "(b) Cost vs load", scale=1000.0)
    value_rows = _metric_load_matrix_panel(axes[2], rows)
    fig.subplots_adjust(left=0.13, right=0.995, top=0.91, bottom=0.14, hspace=0.43, wspace=0.16)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8.4,
        ncols=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
    )

    pdf = out_dir / "fig8_load_sensitivity.pdf"
    csv_path = out_dir / "fig8_load_sensitivity_data.csv"
    value_csv_path = out_dir / "fig8_load_sensitivity_metric_values.csv"
    table_path = out_dir / "table_fig8_load_sensitivity_metrics.tex"
    manifest = out_dir / "fig8_load_sensitivity_manifest.json"
    fig.savefig(pdf)
    plt.close(fig)
    _write_csv(csv_path, rows)
    _write_csv(value_csv_path, value_rows)
    _write_main_metric_table(table_path, rows)
    manifest_payload = {
        "figure": "fig8_load_sensitivity",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pdf": str(pdf),
        "csv": str(csv_path),
        "metric_values_csv": str(value_csv_path),
        "table_tex": str(table_path),
        "round_dirs": [str(Path(path).resolve()) for path in round_dirs],
        "note": "Panels (a) and (b) compare all five systems on CE and lifecycle cost across Low, Medium, and Nominal operating-load points. These labels correspond to average arrival rates of 0.67, 0.81, and 1.01 req/s on the 4-GPU testbed. Panel (c) reports concrete metric values for all five systems and all three load points using the same load labels; all cell values are rounded to three decimals, and colors mark within-metric, within-load favorability with lower-is-better for latency/cost and higher-is-better for CE/throughput. Units: TTFT/E2E in seconds, TPOT in ms, throughput in tok/s, and cost in milli-USD. The accompanying table reports all primary metrics for all systems and load points.",
    }
    manifest.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PrimeLoRA sensitivity figures from completed rounds.")
    parser.add_argument("--round-dir", action="append", required=True, type=Path, help="Completed fair round directory; pass once per load point.")
    parser.add_argument("--out-dir", type=Path, default=Path("figs/paper/sensitivity"))
    args = parser.parse_args()
    plot_load_sensitivity(args.round_dir, args.out_dir.resolve())
    print(f"generated fig8_load_sensitivity -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
