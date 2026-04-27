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
    AXIS_SYSTEM_LABELS,
    LEGEND_FONTSIZE,
    PANEL_TITLE_FONTSIZE,
    SYSTEM_COLORS,
    SYSTEM_LABELS,
    SYSTEM_ORDER,
    TICK_FONTSIZE,
    _as_float,
    _load_json,
    _main_round_data,
    _row_dict,
    _style_axes,
    _system_key,
    _write_csv,
)


LATENCY_SYSTEMS = ("faaslora", "sglang", "vllm", "slora")


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
    title: str,
    *,
    scale: float = 1.0,
) -> None:
    for key in systems:
        xs, ys = _series(rows, key, metric)
        ys = [value * scale for value in ys]
        label = SYSTEM_LABELS[key]
        ax.plot(xs, ys, marker="o", markersize=4.5, linewidth=1.55, color=SYSTEM_COLORS[key], label=label)
    ax.set_xlabel("Nominal replay rate (req/s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    ax.set_xticks(sorted({row["nominal_rps"] for row in rows}))
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    _style_axes(ax)


def plot_load_sensitivity(round_dirs: Sequence[Path], out_dir: Path) -> None:
    rows = _collect([Path(path).resolve() for path in round_dirs])
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.15), constrained_layout=True)
    axes = axes.ravel()
    _plot_lines(axes[0], rows, LATENCY_SYSTEMS, "ttft_avg_ms", "TTFT avg (ms)", "(a) First-token latency")
    _plot_lines(axes[1], rows, LATENCY_SYSTEMS, "e2e_avg_ms", "E2E avg (ms)", "(b) End-to-end latency")
    _plot_lines(axes[2], rows, SYSTEM_ORDER, "cost_req_usd", "Cost/req (milli-USD)", "(c) Lifecycle cost", scale=1000.0)
    _plot_lines(axes[3], rows, SYSTEM_ORDER, "ce", "CE (higher is better)", "(d) Cost efficiency")

    handles, labels = axes[2].get_legend_handles_labels()
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
    manifest = out_dir / "fig8_load_sensitivity_manifest.json"
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    _write_csv(csv_path, rows)
    manifest_payload = {
        "figure": "fig8_load_sensitivity",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pdf": str(pdf),
        "csv": str(csv_path),
        "round_dirs": [str(Path(path).resolve()) for path in round_dirs],
        "note": "Latency panels intentionally exclude ServerlessLLM because its queueing delay is orders of magnitude larger; CE/cost panels include all systems.",
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
