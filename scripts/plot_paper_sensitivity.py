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
    _xlabel_with_panel(ax, "Nominal replay rate (req/s)", panel_caption)
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted({row["nominal_rps"] for row in rows}))
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    _style_axes(ax)


def _ratio_color(metric: str, ratio: float) -> str:
    favorable = ratio < 1.0 if metric in LOWER_BETTER_METRICS else ratio > 1.0
    distance = min(abs(ratio - 1.0), 1.0)
    if distance < 0.05:
        return "#F1F1F1"
    if favorable:
        return "#DCEEDC" if distance < 0.25 else "#B9DFBA"
    return "#F7D9D7" if distance < 0.75 else "#EFB3AF"


def _metric_ratio_panel(ax: plt.Axes, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metric_specs = [
        ("CE", "ce", "↑"),
        ("Cost", "cost_req_usd", "↓"),
        ("TTFT avg", "ttft_avg_ms", "↓"),
        ("TTFT p95", "ttft_p95_ms", "↓"),
        ("E2E avg", "e2e_avg_ms", "↓"),
        ("E2E p95", "e2e_p95_ms", "↓"),
        ("TPOT avg", "tpot_avg_ms", "↓"),
        ("TPOT p95", "tpot_p95_ms", "↓"),
        ("Tok/s", "tok_s", "↑"),
    ]
    scales = sorted({row["time_scale"] for row in rows}, reverse=True)
    out_rows: List[Dict[str, Any]] = []
    ax.set_xlim(0, len(metric_specs))
    ax.set_ylim(0, len(scales))
    for yi, scale in enumerate(scales):
        scale_rows = [row for row in rows if row["time_scale"] == scale]
        faas = next(row for row in scale_rows if row["system_key"] == "faaslora")
        ref = next(row for row in scale_rows if row["system_key"] == "sglang")
        for xi, (label, metric, direction) in enumerate(metric_specs):
            ratio = faas[metric] / ref[metric]
            rect = plt.Rectangle((xi, yi), 1, 1, facecolor=_ratio_color(metric, ratio), edgecolor="white", linewidth=1.0)
            ax.add_patch(rect)
            ax.text(xi + 0.5, yi + 0.58, f"{ratio:.2f}x", ha="center", va="center", fontsize=8.3)
            ax.text(xi + 0.5, yi + 0.28, direction, ha="center", va="center", fontsize=7.8, color="#555555")
            out_rows.append(
                {
                    "time_scale": scale,
                    "nominal_rps": faas["nominal_rps"],
                    "reference_system": "SGLang",
                    "metric": metric,
                    "metric_label": label,
                    "direction": "lower_is_better" if metric in LOWER_BETTER_METRICS else "higher_is_better",
                    "faaslora_value": faas[metric],
                    "reference_value": ref[metric],
                    "faaslora_over_sglang": ratio,
                }
            )
    ax.set_xticks(np.arange(len(metric_specs)) + 0.5, [item[0] for item in metric_specs], rotation=28, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(scales)) + 0.5, [f"s{int(scale)}" if float(scale).is_integer() else f"s{scale:g}" for scale in scales])
    ax.tick_params(axis="both", labelsize=8.1, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("(c) Primary metric ratio: FaaSLoRA / SGLang")
    return out_rows


def _write_main_metric_table(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    selected = sorted(rows, key=lambda row: (-row["time_scale"], SYSTEM_ORDER.index(row["system_key"])))
    lines = [
        "% Auto-generated by scripts/plot_paper_sensitivity.py. Verify caption wording before final submission.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Operating-load sensitivity on the representative Llama-2 7B workload. Lower is better for latency and cost; higher is better for Tok/s and CE.}",
        "\\label{tab:load_sensitivity_metrics}",
        "\\begin{tabular}{llrrrrrrrrr}",
        "\\hline",
        "Load & System & TTFT Avg & TTFT p95 & E2E Avg & E2E p95 & TPOT Avg & TPOT p95 & Tok/s & Cost/req & CE \\\\",
        "\\hline",
    ]
    for row in selected:
        scale = int(row["time_scale"]) if float(row["time_scale"]).is_integer() else row["time_scale"]
        lines.append(
            f"s{scale} & {SYSTEM_LABELS[row['system_key']]} & "
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

    fig = plt.figure(figsize=(7.16, 3.35), constrained_layout=True)
    gridspec = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.55])
    axes = [fig.add_subplot(gridspec[0, 0]), fig.add_subplot(gridspec[0, 1]), fig.add_subplot(gridspec[0, 2])]
    _plot_lines(axes[0], rows, SYSTEM_ORDER, "ce", "CE (higher is better)", "(a) CE across operating load")
    _plot_lines(axes[1], rows, SYSTEM_ORDER, "cost_req_usd", "Cost/req (milli-USD)", "(b) Lifecycle cost", scale=1000.0)
    ratio_rows = _metric_ratio_panel(axes[2], rows)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        ncols=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
    )

    pdf = out_dir / "fig8_load_sensitivity.pdf"
    csv_path = out_dir / "fig8_load_sensitivity_data.csv"
    ratio_csv_path = out_dir / "fig8_load_sensitivity_ratios.csv"
    table_path = out_dir / "table_fig8_load_sensitivity_metrics.tex"
    manifest = out_dir / "fig8_load_sensitivity_manifest.json"
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    _write_csv(csv_path, rows)
    _write_csv(ratio_csv_path, ratio_rows)
    _write_main_metric_table(table_path, rows)
    manifest_payload = {
        "figure": "fig8_load_sensitivity",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pdf": str(pdf),
        "csv": str(csv_path),
        "ratio_csv": str(ratio_csv_path),
        "table_tex": str(table_path),
        "round_dirs": [str(Path(path).resolve()) for path in round_dirs],
        "note": "Panels (a) and (b) compare all five systems on CE and lifecycle cost. Panel (c) audits all primary metrics as FaaSLoRA/SGLang ratios because SGLang is the strongest CE baseline in these operating-load rounds. The accompanying table reports all primary metrics for all systems and load points.",
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
