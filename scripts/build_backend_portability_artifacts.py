#!/usr/bin/env python3
"""Build PrimeLoRA-SGLang backend portability artifacts.

This script does not rerun the formal main-experiment chain.  It reads measured
SGLang, vLLM, PrimeLoRA-vLLM, and PrimeLoRA-SGLang results that were produced
on the same shared replay and adapter subset, then emits the paper table and
lifecycle figure for the backend-portability extension.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_main_7b13b_artifacts as main_artifacts  # noqa: E402
import plot_paper_figures as ppf  # noqa: E402


DEFAULT_BASELINES_ROOT = Path("/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison")
DEFAULT_7B_ROUND = DEFAULT_BASELINES_ROOT / "20260424_104050_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1"
DEFAULT_3B_ROUND = DEFAULT_BASELINES_ROOT / "20260509_201205_llama32_3b_main_s8_v2"
DEFAULT_3B_PRIME = Path(
    "/home/qhq/serverless_llm_experiment_retry14_baseline/results/"
    "experiment_results_full_vllm_auto_a500_r4000_c4_faaslora_full_"
    "llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_max2_auto.json"
)
DEFAULT_7B_PRIME_SGLANG = Path(
    "/home/qhq/serverless_llm_experiment_retry14_baseline/results/"
    "experiment_results_full_sglang_auto_a500_r4000_c4_faaslora_full_"
    "llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_primelora_sglang_actual_v1.json"
)
DEFAULT_3B_PRIME_SGLANG = Path(
    "/home/qhq/serverless_llm_experiment_retry14_baseline/results/"
    "experiment_results_full_sglang_auto_a500_r4000_c4_faaslora_full_"
    "llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_primelora_sglang_actual_v1.json"
)


@dataclass(frozen=True)
class SourceSet:
    model_key: str
    model_label: str
    round_dir: Path
    sglang_summary: Path
    prime_sglang_summary: Path
    vllm_summary: Path
    prime_vllm_summary: Path


@dataclass(frozen=True)
class PortabilityRow:
    model_key: str
    model_label: str
    system_key: str
    system_label: str
    source: Path
    method: str
    metrics: dict[str, float]
    diagnostics: dict[str, float]


SYSTEM_ORDER = ("vllm", "primelora_vllm", "sglang", "primelora_sglang")
SYSTEM_LABELS = {
    "sglang": "SGLang",
    "primelora_sglang": "PrimeLoRA-SGLang",
    "vllm": "vLLM",
    "primelora_vllm": "PrimeLoRA-vLLM",
}
SYSTEM_COLORS = {
    "vllm": "#85C1B9",
    "primelora_vllm": "#D88C5A",
    "sglang": "#6E9CCF",
    "primelora_sglang": "#4F9D69",
}
BACKEND_LABELS = {
    "vllm": "vLLM",
    "primelora_vllm": "vLLM",
    "sglang": "SGLang",
    "primelora_sglang": "SGLang",
}
SYSTEM_SHORT_LABELS = {
    "vllm": "Standalone",
    "primelora_vllm": "PrimeLoRA",
    "sglang": "Standalone",
    "primelora_sglang": "PrimeLoRA",
}


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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


def _round_run_tag(round_dir: Path) -> str:
    return main_artifacts._round_run_tag(round_dir)


def _source_set(
    model_key: str,
    model_label: str,
    round_dir: Path,
    *,
    prime_sglang_summary: Path,
    prime_vllm_override: Path | None = None,
) -> SourceSet:
    run_tag = _round_run_tag(round_dir)
    sglang = ppf._main_summary_path(round_dir, run_tag, "sglang")
    vllm = ppf._main_summary_path(round_dir, run_tag, "vllm")
    prime = prime_vllm_override or ppf._main_summary_path(round_dir, run_tag, "faaslora")
    return SourceSet(
        model_key=model_key,
        model_label=model_label,
        round_dir=round_dir,
        sglang_summary=sglang,
        prime_sglang_summary=prime_sglang_summary,
        vllm_summary=vllm,
        prime_vllm_summary=prime,
    )


def _measured_row(model: SourceSet, system_key: str, source: Path, measured_key: str) -> PortabilityRow:
    metrics = main_artifacts._metrics_from_summary(source, measured_key)
    diagnostics = main_artifacts._diagnostics_from_summary(source, measured_key)
    return PortabilityRow(
        model_key=model.model_key,
        model_label=model.model_label,
        system_key=system_key,
        system_label=SYSTEM_LABELS[system_key],
        source=source,
        method="measured",
        metrics=metrics,
        diagnostics=diagnostics,
    )


def _rows_for_models(models: Sequence[SourceSet]) -> list[PortabilityRow]:
    rows: list[PortabilityRow] = []
    for model in models:
        rows.append(_measured_row(model, "sglang", model.sglang_summary, "sglang"))
        rows.append(_measured_row(model, "primelora_sglang", model.prime_sglang_summary, "faaslora"))
        rows.append(_measured_row(model, "vllm", model.vllm_summary, "vllm"))
        rows.append(_measured_row(model, "primelora_vllm", model.prime_vllm_summary, "faaslora"))
    return rows


def _csv_rows(rows: Sequence[PortabilityRow]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        data = {
            "model_key": row.model_key,
            "model": row.model_label,
            "system_key": row.system_key,
            "system": row.system_label,
            "method": row.method,
            "source": str(row.source),
        }
        data.update(row.metrics)
        data.update({f"diag_{k}": v for k, v in row.diagnostics.items()})
        out.append(data)
    return out


def _best_keys(rows: Sequence[PortabilityRow], model_key: str, metric: str, *, higher: bool = False) -> set[str]:
    vals = {row.system_key: float(row.metrics[metric]) for row in rows if row.model_key == model_key}
    target = max(vals.values()) if higher else min(vals.values())
    return {key for key, value in vals.items() if math.isclose(value, target, rel_tol=1e-9, abs_tol=1e-9)}


def _fmt(value: float, fmt: str, *, bold: bool = False) -> str:
    text = fmt.format(value)
    return f"\\textbf{{{text}}}" if bold else text


def _fmt_ms(value: float, *, bold: bool = False) -> str:
    text = f"{value:.1f}"
    return f"\\textbf{{{text}}}" if bold else text


def write_main_table(rows: Sequence[PortabilityRow], out_dir: Path) -> None:
    _write_csv(out_dir / "table_backend_portability_data.csv", _csv_rows(rows))
    by_model: dict[str, list[PortabilityRow]] = {}
    model_labels: dict[str, str] = {}
    for row in rows:
        by_model.setdefault(row.model_key, []).append(row)
        model_labels[row.model_key] = row.model_label

    lines = [
        "% Auto-generated by scripts/build_backend_portability_artifacts.py.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Backend sensitivity using measured vLLM-backed and SGLang-backed PrimeLoRA runs on the same shared replay and adapter subset.}",
        "\\label{tab:backend_portability}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3.8pt}",
        "\\renewcommand{\\arraystretch}{1.12}",
        "\\begin{tabular}{lllrrrr}",
        "\\toprule",
        "Model & Backend & System & "
        "\\shortstack{TTFT Avg\\\\(ms)} & "
        "\\shortstack{Throughput\\\\(Tok/s)} & "
        "\\shortstack{Cost/req\\\\(mUSD)} & "
        "\\shortstack{CE\\\\$(\\bar{L}\\bar{C})^{-1}$} \\\\",
        "\\midrule",
    ]
    for mi, model_key in enumerate(by_model):
        if mi:
            lines.append("\\midrule")
        best = {
            "ttft_avg_ms": _best_keys(rows, model_key, "ttft_avg_ms"),
            "tok_s": _best_keys(rows, model_key, "tok_s", higher=True),
            "cost_req_usd": _best_keys(rows, model_key, "cost_req_usd"),
            "ce": _best_keys(rows, model_key, "ce", higher=True),
        }
        ordered = sorted(by_model[model_key], key=lambda row: SYSTEM_ORDER.index(row.system_key))
        for ri, row in enumerate(ordered):
            m = row.metrics
            model_cell = model_labels[model_key] if ri == 0 else ""
            lines.append(
                " & ".join(
                    [
                        model_cell,
                        BACKEND_LABELS[row.system_key],
                        SYSTEM_SHORT_LABELS[row.system_key],
                        _fmt_ms(m["ttft_avg_ms"], bold=row.system_key in best["ttft_avg_ms"]),
                        _fmt(m["tok_s"], "{:.1f}", bold=row.system_key in best["tok_s"]),
                        _fmt(m["cost_req_usd"] * 1000.0, "{:.3f}", bold=row.system_key in best["cost_req_usd"]),
                        _fmt(m["ce"], "{:.2f}", bold=row.system_key in best["ce"]),
                    ]
                )
                + " \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (out_dir / "table_backend_portability.tex").write_text("\n".join(lines), encoding="utf-8")


def write_decomposition_table(rows: Sequence[PortabilityRow], out_dir: Path) -> None:
    data = []
    for row in rows:
        data.append(
            {
                "model_key": row.model_key,
                "model": row.model_label,
                "system_key": row.system_key,
                "system": row.system_label,
                "method": row.method,
                "ttft_avg_ms": row.metrics["ttft_avg_ms"],
                "service_ttft_ms": row.diagnostics["service_ttft_ms"],
                "dispatch_wait_ms": row.diagnostics["dispatch_wait_ms"],
                "tpot_avg_ms": row.metrics["tpot_avg_ms"],
                "source": str(row.source),
            }
        )
    _write_csv(out_dir / "table_backend_portability_ttft_decomposition_data.csv", data)

    lines = [
        "% Auto-generated by scripts/build_backend_portability_artifacts.py.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Backend-portability first-token decomposition using measured SGLang-backed and vLLM-backed PrimeLoRA runs.}",
        "\\label{tab:backend_portability_decomposition}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3.0pt}",
        "\\renewcommand{\\arraystretch}{1.10}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Model & System & "
        "\\shortstack{TTFT Avg\\\\(ms)} & "
        "\\shortstack{Service TTFT\\\\(ms)} & "
        "\\shortstack{Dispatch Wait\\\\(ms)} & "
        "\\shortstack{TPOT Avg\\\\(ms)} \\\\",
        "\\midrule",
    ]
    last_model = None
    for row in rows:
        if last_model is not None and row.model_label != last_model:
            lines.append("\\midrule")
        model_label = row.model_label if row.model_label != last_model else ""
        last_model = row.model_label
        lines.append(
            f"{model_label} & {row.system_label} & "
            f"{row.metrics['ttft_avg_ms']:.1f} & {row.diagnostics['service_ttft_ms']:.1f} & "
            f"{row.diagnostics['dispatch_wait_ms']:.1f} & {row.metrics['tpot_avg_ms']:.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (out_dir / "table_backend_portability_ttft_decomposition.tex").write_text("\n".join(lines), encoding="utf-8")


def _two_line_model(label: str) -> str:
    parts = label.rsplit(" ", 1)
    if len(parts) == 2:
        return f"{parts[0]}\n{parts[1]}"
    return label


def write_lifecycle_figure(rows: Sequence[PortabilityRow], out_dir: Path) -> None:
    _write_csv(out_dir / "fig_backend_portability_lifecycle_cost_data.csv", _csv_rows(rows))
    labels: list[str] = []
    ordered: list[PortabilityRow] = []
    y_positions: list[float] = []
    group_centers: list[tuple[str, float]] = []
    y = 0.0
    model_order = []
    for row in rows:
        if row.model_key not in model_order:
            model_order.append(row.model_key)
    for model_key in model_order:
        model_rows = [row for row in rows if row.model_key == model_key]
        start = y
        for key in SYSTEM_ORDER:
            row = next(item for item in model_rows if item.system_key == key)
            labels.append(row.system_label)
            ordered.append(row)
            y_positions.append(y)
            y += 1.0
        group_centers.append((_two_line_model(model_rows[0].model_label), (start + y - 1.0) / 2.0))
        y += 0.52

    y_arr = np.asarray(y_positions, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(3.50, 2.78), constrained_layout=False)
    components = [
        ("Startup", ("cost_startup_usd",), "#A9C4E8"),
        ("Active", ("cost_active_usd", "cost_invocation_usd"), "#A7D3A8"),
        ("Idle-ready", ("cost_idle_ready_usd",), "#F4C58D"),
    ]
    bottom = np.zeros(len(ordered))
    handles = []
    legend_labels = []
    for name, keys, color in components:
        vals = np.asarray([sum(row.metrics.get(key, 0.0) for key in keys) * 1000.0 for row in ordered])
        bars = axes[0].barh(y_arr, vals, left=bottom, height=0.52, color=color, edgecolor="#555555", linewidth=0.25)
        handles.append(bars[0])
        legend_labels.append(name)
        bottom += vals
    axes[0].set_yticks(y_arr)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_ylim(float(y_arr[-1] + 0.24), float(y_arr[0] - 0.28))
    ppf._xlabel_with_panel(axes[0], "Cost/req (mUSD)", "(a) Cost")
    ppf._style_axes(axes[0])

    gpu_components = [
        ("Startup", "infra_startup_gpu_seconds", "#A9C4E8"),
        ("Active", "infra_active_gpu_seconds", "#A7D3A8"),
        ("Idle-ready", "infra_idle_ready_gpu_seconds", "#F4C58D"),
    ]
    bottom = np.zeros(len(ordered))
    for _, key, color in gpu_components:
        vals = np.asarray([row.metrics.get(key, 0.0) / max(row.metrics["completed"], 1.0) for row in ordered])
        axes[1].barh(y_arr, vals, left=bottom, height=0.52, color=color, edgecolor="#555555", linewidth=0.25)
        bottom += vals
    axes[1].set_yticks(y_arr)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_ylim(float(y_arr[-1] + 0.24), float(y_arr[0] - 0.28))
    ppf._xlabel_with_panel(axes[1], "GPU-s/req", "(b) GPU time")
    ppf._style_axes(axes[1])

    for ax in axes:
        ax.tick_params(axis="both", labelsize=6.55)
        ax.xaxis.label.set_size(7.2)
        ax.xaxis.labelpad = 3.6
        ax.grid(axis="x", color="#E7E7E7", linewidth=0.45)
        for _, center in group_centers[:-1]:
            ax.axhline(center + 2.47, color="#B8B8B8", linewidth=0.5, linestyle=":")
    for text, center in group_centers:
        lines = text.split("\n", 1)
        x_label = -1.43
        if len(lines) == 2:
            axes[0].text(
                x_label,
                center - 0.29,
                lines[0],
                transform=axes[0].get_yaxis_transform(),
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight="bold",
                clip_on=False,
            )
            axes[0].text(
                x_label,
                center + 0.29,
                lines[1],
                transform=axes[0].get_yaxis_transform(),
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight="bold",
                clip_on=False,
            )
        else:
            axes[0].text(
                x_label,
                center,
                text,
                transform=axes[0].get_yaxis_transform(),
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight="bold",
                clip_on=False,
            )

    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        fontsize=6.7,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.61, 0.925),
        columnspacing=0.55,
        handlelength=1.1,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.505, right=0.99, top=0.835, bottom=0.135, wspace=0.23)
    pdf = out_dir / "fig_backend_portability_lifecycle_cost.pdf"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def write_manifest(rows: Sequence[PortabilityRow], out_dir: Path) -> None:
    payload = {
        "artifact": "primelora_sglang_backend_portability",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Measured SGLang/vLLM baselines, PrimeLoRA-vLLM, and PrimeLoRA-SGLang "
            "runs are read from result JSON files produced on the same shared replay "
            "and adapter subset."
        ),
        "rows": [
            {
                "model": row.model_label,
                "system": row.system_label,
                "method": row.method,
                "source": str(row.source),
                "ce": row.metrics["ce"],
                "e2e_avg_ms": row.metrics["e2e_avg_ms"],
                "cost_req_usd": row.metrics["cost_req_usd"],
            }
            for row in rows
        ],
        "outputs": {
            "table": str(out_dir / "table_backend_portability.tex"),
            "decomposition": str(out_dir / "table_backend_portability_ttft_decomposition.tex"),
            "figure": str(out_dir / "fig_backend_portability_lifecycle_cost.pdf"),
            "prime_sglang_results": [
                str(row.source)
                for row in rows
                if row.system_key == "primelora_sglang"
            ],
        },
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    for name in (
        "backend_portability_manifest.json",
        "table_backend_portability_manifest.json",
        "table_backend_portability_ttft_decomposition_manifest.json",
        "fig_backend_portability_lifecycle_cost_manifest.json",
    ):
        (out_dir / name).write_text(text, encoding="utf-8")


def build(models: Sequence[SourceSet], out_dir: Path, copy_root_fig: bool = True) -> list[PortabilityRow]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows_for_models(models)
    write_main_table(rows, out_dir)
    write_decomposition_table(rows, out_dir)
    write_lifecycle_figure(rows, out_dir)
    write_manifest(rows, out_dir)
    if copy_root_fig:
        root_fig = Path("figs") / "fig_backend_portability_lifecycle_cost.pdf"
        root_fig.parent.mkdir(parents=True, exist_ok=True)
        root_fig.write_bytes((out_dir / "fig_backend_portability_lifecycle_cost.pdf").read_bytes())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PrimeLoRA-SGLang backend portability artifacts.")
    parser.add_argument("--round-7b", type=Path, default=DEFAULT_7B_ROUND)
    parser.add_argument("--round-3b", type=Path, default=DEFAULT_3B_ROUND)
    parser.add_argument("--prime-sglang-7b-summary", type=Path, default=DEFAULT_7B_PRIME_SGLANG)
    parser.add_argument("--prime-sglang-3b-summary", type=Path, default=DEFAULT_3B_PRIME_SGLANG)
    parser.add_argument(
        "--prime-7b-summary",
        type=Path,
        default=None,
        help="Optional PrimeLoRA-vLLM 7B override; defaults to the baseline round faaslora summary.",
    )
    parser.add_argument("--prime-3b-summary", type=Path, default=DEFAULT_3B_PRIME)
    parser.add_argument("--out-dir", type=Path, default=Path("figs/paper/backend_portability"))
    args = parser.parse_args()

    models = [
        _source_set(
            "llama2_7b",
            "Llama-2 7B",
            args.round_7b.resolve(),
            prime_sglang_summary=args.prime_sglang_7b_summary.resolve(),
            prime_vllm_override=args.prime_7b_summary.resolve() if args.prime_7b_summary else None,
        ),
        _source_set(
            "llama32_3b",
            "Llama-3.2 3B",
            args.round_3b.resolve(),
            prime_sglang_summary=args.prime_sglang_3b_summary.resolve(),
            prime_vllm_override=args.prime_3b_summary.resolve(),
        ),
    ]
    rows = build(models, args.out_dir.resolve())
    print(f"wrote backend portability artifacts to {args.out_dir.resolve()}")
    for row in rows:
        print(
            f"{row.model_label:14s} {row.system_label:18s} "
            f"E2E={row.metrics['e2e_avg_ms']:.1f}ms Cost={row.metrics['cost_req_usd']*1000.0:.3f}mUSD "
            f"CE={row.metrics['ce']:.2f} method={row.method}"
        )


if __name__ == "__main__":
    main()
