#!/usr/bin/env python3
"""Build paper artifacts that merge Llama-2 7B and 13B main results."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

import plot_paper_figures as ppf  # noqa: E402


@dataclass(frozen=True)
class ModelRound:
    key: str
    label: str
    round_dir: Path
    systems: list[ppf.MainSystemData]


SYSTEM_ORDER = ppf.SYSTEM_ORDER
DISPLAY_ORDER = ("sglang", "vllm", "slora", "serverlessllm", "faaslora")
SYSTEM_LABELS = ppf.SYSTEM_LABELS


def _fmt(value: float, fmt: str, *, bold: bool = False) -> str:
    text = fmt.format(value)
    return f"\\textbf{{{text}}}" if bold else text


def _best_keys(systems: Sequence[ppf.MainSystemData], metric: str, *, higher: bool = False) -> set[str]:
    values = {system.key: float(system.metrics[metric]) for system in systems}
    best = max(values.values()) if higher else min(values.values())
    return {key for key, value in values.items() if math.isclose(value, best, rel_tol=1e-9, abs_tol=1e-9)}


def _system_rows(model: ModelRound) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for system in _ordered_systems(model.systems):
        metrics = system.metrics
        row = {
            "model": model.label,
            "system_key": system.key,
            "system": system.label,
            "source": str(system.source),
        }
        for key, value in metrics.items():
            row[key] = value
        rows.append(row)
    return rows


def _ordered_systems(systems: Sequence[ppf.MainSystemData]) -> list[ppf.MainSystemData]:
    by_key = {system.key: system for system in systems}
    return [by_key[key] for key in DISPLAY_ORDER if key in by_key]


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
    manifest_path = round_dir / "MANIFEST.json"
    if manifest_path.exists():
        manifest = ppf._load_json(manifest_path)
        if manifest.get("metric_schema_version") != "e2e_v3":
            raise SystemExit(f"{manifest_path}: metric_schema_version must be e2e_v3")
        run_tag = str(manifest.get("run_tag") or "")
        if run_tag:
            return run_tag

    env_path = round_dir / "round.env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("export SLLM_RUN_TAG="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    shared = sorted((round_dir / "shared_artifacts").glob("*_trace.json"))
    if len(shared) == 1:
        return shared[0].name.removesuffix("_trace.json")
    raise SystemExit(f"{round_dir}: unable to determine SLLM_RUN_TAG")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shared_artifact_hashes(round_dir: Path) -> dict[str, str]:
    shared_dir = round_dir / "shared_artifacts"
    traces = sorted(shared_dir.glob("*_trace.json"))
    adapters = sorted(shared_dir.glob("*_adapter_subset.json"))
    if len(traces) != 1 or len(adapters) != 1:
        raise SystemExit(f"{round_dir}: expected exactly one shared trace and adapter subset")
    return {
        "trace": _sha256(traces[0]),
        "adapter_subset": _sha256(adapters[0]),
    }


def _validate_same_shared_artifacts(base_round: Path, override_round: Path, *, model_key: str, system_key: str) -> None:
    base = _shared_artifact_hashes(base_round)
    other = _shared_artifact_hashes(override_round)
    if base != other:
        raise SystemExit(
            f"{model_key}:{system_key}: override round does not use the same shared trace/adapter subset as {base_round}"
        )


def _parse_system_round_overrides(specs: Sequence[str]) -> dict[str, dict[str, Path]]:
    overrides: dict[str, dict[str, Path]] = {}
    for spec in specs:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            raise SystemExit(
                "--system-round-override must use MODEL_KEY:SYSTEM_KEY:ROUND_DIR, "
                "for example llama2_13b:faaslora:/path/to/round"
            )
        model_key, system_key, round_dir = parts
        if model_key not in {"llama2_7b", "llama2_13b"}:
            raise SystemExit(f"{spec}: unknown model key {model_key!r}")
        if system_key not in SYSTEM_ORDER:
            raise SystemExit(f"{spec}: unknown system key {system_key!r}")
        overrides.setdefault(model_key, {})[system_key] = Path(round_dir).expanduser().resolve()
    return overrides


def _metrics_from_summary(source: Path, key: str) -> dict[str, float]:
    raw = ppf._main_row_from_summary(source, key)
    completed = int(ppf._as_float(raw.get("completed"), f"{key}.completed"))
    total = int(ppf._as_float(raw.get("total"), f"{key}.total"))
    if completed <= 0 or completed != total:
        raise SystemExit(f"{source}: invalid completion completed={completed} total={total}")

    metrics = {
        "completed": float(completed),
        "ttft_avg_ms": ppf._as_float(raw.get("TTFT_avg_ms"), f"{key}.TTFT_avg_ms"),
        "ttft_p95_ms": ppf._as_float(raw.get("TTFT_p95_ms"), f"{key}.TTFT_p95_ms"),
        "e2e_avg_ms": ppf._as_float(raw.get("E2E_avg_ms"), f"{key}.E2E_avg_ms"),
        "e2e_p95_ms": ppf._as_float(raw.get("E2E_p95_ms"), f"{key}.E2E_p95_ms"),
        "tpot_avg_ms": ppf._as_float(raw.get("TPOT_avg_ms"), f"{key}.TPOT_avg_ms"),
        "tpot_p95_ms": ppf._as_float(raw.get("TPOT_p95_ms"), f"{key}.TPOT_p95_ms"),
        "tok_s": ppf._as_float(raw.get("Tok_s"), f"{key}.Tok_s"),
        "cost_req_usd": ppf._as_float(raw.get("Cost_req_usd"), f"{key}.Cost_req_usd"),
        "ce": ppf._as_float(raw.get("CE"), f"{key}.CE"),
        "cost_1mtok_usd": ppf._as_float(raw.get("cost_per_1m_total_tokens_usd"), f"{key}.cost_per_1m_total_tokens_usd"),
        "monetary_cost_total_usd": ppf._as_float(raw.get("monetary_cost_total_usd"), f"{key}.monetary_cost_total_usd"),
        "monetary_active_charge_gpu_seconds": ppf._as_float(
            raw.get("monetary_active_charge_gpu_seconds"),
            f"{key}.monetary_active_charge_gpu_seconds",
        ),
        "monetary_idle_charge_gpu_seconds": ppf._as_float(
            raw.get("monetary_idle_charge_gpu_seconds"),
            f"{key}.monetary_idle_charge_gpu_seconds",
        ),
        "infra_active_gpu_seconds": ppf._as_float(raw.get("infra_active_gpu_seconds"), f"{key}.infra_active_gpu_seconds"),
        "infra_idle_ready_gpu_seconds": ppf._as_float(
            raw.get("infra_idle_ready_gpu_seconds"),
            f"{key}.infra_idle_ready_gpu_seconds",
        ),
        "infra_startup_gpu_seconds": ppf._as_float(raw.get("infra_startup_gpu_seconds"), f"{key}.infra_startup_gpu_seconds"),
        "serverless_invocation_cost_per_request_usd": ppf._as_float(
            raw.get("serverless_invocation_cost_per_request_usd"),
            f"{key}.serverless_invocation_cost_per_request_usd",
        ),
    }
    active_rate = metrics["monetary_cost_total_usd"] / max(
        metrics["monetary_active_charge_gpu_seconds"] + metrics["monetary_idle_charge_gpu_seconds"], 1e-12
    )
    startup = metrics["infra_startup_gpu_seconds"] * active_rate / metrics["completed"]
    idle = metrics["monetary_idle_charge_gpu_seconds"] * active_rate / metrics["completed"]
    invocation = metrics["serverless_invocation_cost_per_request_usd"]
    active = max(metrics["cost_req_usd"] - startup - idle - invocation, 0.0)
    metrics.update(
        {
            "tpot_ms": metrics["tpot_avg_ms"],
            "cost_startup_usd": startup,
            "cost_active_usd": active,
            "cost_idle_ready_usd": idle,
            "cost_invocation_usd": invocation,
        }
    )
    return metrics


def _diagnostics_from_summary(source: Path, key: str) -> dict[str, float]:
    payload = ppf._load_json(source)
    if payload.get("metric_schema_version") != "e2e_v3":
        raise SystemExit(f"{source}: metric_schema_version must be e2e_v3")

    if key == "faaslora":
        summary = (payload.get("scenario_summaries") or {}).get("faaslora_full")
        if not summary:
            raise SystemExit(f"{source}: missing scenario_summaries.faaslora_full")
        return {
            "service_ttft_ms": ppf._as_float(summary.get("avg_service_ttft_ms"), f"{key}.avg_service_ttft_ms"),
            "dispatch_wait_ms": ppf._as_float(
                summary.get("avg_dispatch_admission_wait_ms"),
                f"{key}.avg_dispatch_admission_wait_ms",
            ),
            "service_e2e_ms": ppf._as_float(summary.get("avg_service_e2e_ms"), f"{key}.avg_service_e2e_ms"),
        }

    table = payload.get("comparison_table") or []
    if len(table) != 1:
        raise SystemExit(f"{source}: expected one comparison_table row, found {len(table)}")
    row = table[0]
    return {
        "service_ttft_ms": ppf._as_float(row.get("TTFT_service_avg_ms"), f"{key}.TTFT_service_avg_ms"),
        "dispatch_wait_ms": ppf._as_float(
            row.get("Dispatch_admission_wait_avg_ms"),
            f"{key}.Dispatch_admission_wait_avg_ms",
        ),
        "service_e2e_ms": ppf._as_float(row.get("E2E_service_avg_ms"), f"{key}.E2E_service_avg_ms"),
    }


def build_main_table(models: Sequence[ModelRound], out_dir: Path) -> None:
    rows = [row for model in models for row in _system_rows(model)]
    _write_csv(out_dir / "table1_end_to_end_data.csv", rows)

    lines = [
        "% Auto-generated by scripts/build_main_7b13b_artifacts.py. Verify caption wording before final submission.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Main comparison on Llama-2 7B and 13B multi-LoRA workloads. Latency and TPOT are in milliseconds, Cost/req is in mUSD, and throughput is in tokens/s. Lower latency and cost are better; higher throughput and CE are better.}",
        "\\label{tab:main_results}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{1.75pt}",
        "\\renewcommand{\\arraystretch}{1.10}",
        "\\begin{tabular}{lrrrrrrrrr}",
        "\\toprule",
        "System & "
        "\\shortstack{TTFT Avg\\\\(ms)} & "
        "\\shortstack{TTFT P95\\\\(ms)} & "
        "\\shortstack{E2E Avg\\\\(ms)} & "
        "\\shortstack{E2E P95\\\\(ms)} & "
        "\\shortstack{TPOT Avg\\\\(ms)} & "
        "\\shortstack{TPOT P95\\\\(ms)} & "
        "\\shortstack{Throughput\\\\(Tok/s)} & "
        "\\shortstack{Cost/req\\\\(mUSD)} & "
        "\\shortstack{CE\\\\$(\\bar{L}\\bar{C})^{-1}$} \\\\",
        "\\midrule",
    ]
    for mi, model in enumerate(models):
        if mi:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{model.label}}}}} \\\\")
        best = {
            "ttft_avg_ms": _best_keys(model.systems, "ttft_avg_ms"),
            "ttft_p95_ms": _best_keys(model.systems, "ttft_p95_ms"),
            "e2e_avg_ms": _best_keys(model.systems, "e2e_avg_ms"),
            "e2e_p95_ms": _best_keys(model.systems, "e2e_p95_ms"),
            "tpot_avg_ms": _best_keys(model.systems, "tpot_avg_ms"),
            "tpot_p95_ms": _best_keys(model.systems, "tpot_p95_ms"),
            "tok_s": _best_keys(model.systems, "tok_s", higher=True),
            "cost_req_usd": _best_keys(model.systems, "cost_req_usd"),
            "ce": _best_keys(model.systems, "ce", higher=True),
        }
        for system in _ordered_systems(model.systems):
            m = system.metrics
            lines.append(
                " & ".join(
                    [
                        system.label,
                        _fmt(m["ttft_avg_ms"], "{:.1f}", bold=system.key in best["ttft_avg_ms"]),
                        _fmt(m["ttft_p95_ms"], "{:.1f}", bold=system.key in best["ttft_p95_ms"]),
                        _fmt(m["e2e_avg_ms"], "{:.1f}", bold=system.key in best["e2e_avg_ms"]),
                        _fmt(m["e2e_p95_ms"], "{:.1f}", bold=system.key in best["e2e_p95_ms"]),
                        _fmt(m["tpot_avg_ms"], "{:.1f}", bold=system.key in best["tpot_avg_ms"]),
                        _fmt(m["tpot_p95_ms"], "{:.1f}", bold=system.key in best["tpot_p95_ms"]),
                        _fmt(m["tok_s"], "{:.1f}", bold=system.key in best["tok_s"]),
                        _fmt(m["cost_req_usd"] * 1000.0, "{:.3f}", bold=system.key in best["cost_req_usd"]),
                        _fmt(m["ce"], "{:.2f}", bold=system.key in best["ce"]),
                    ]
                )
                + " \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (out_dir / "table1_end_to_end.tex").write_text("\n".join(lines), encoding="utf-8")


def build_decomposition_table(models: Sequence[ModelRound], out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for model in models:
        for system in _ordered_systems(model.systems):
            d = _diagnostics_from_summary(system.source, system.key)
            rows.append(
                {
                    "model": model.label,
                    "system_key": system.key,
                    "system": system.label,
                    "ttft_avg_ms": system.metrics["ttft_avg_ms"],
                    "service_ttft_ms": d["service_ttft_ms"],
                    "dispatch_wait_ms": d["dispatch_wait_ms"],
                    "tpot_avg_ms": system.metrics["tpot_avg_ms"],
                }
            )
    _write_csv(out_dir / "table_ttft_decomposition_data.csv", rows)

    lines = [
        "% Auto-generated by scripts/build_main_7b13b_artifacts.py.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Diagnostic first-token decomposition on Llama-2 7B and 13B multi-LoRA workloads. TTFT, Service TTFT, Dispatch Wait, and TPOT are in milliseconds.}",
        "\\label{tab:ttft_decomposition}",
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
        if last_model is not None and row["model"] != last_model:
            lines.append("\\midrule")
        model_label = row["model"] if row["model"] != last_model else ""
        last_model = row["model"]
        lines.append(
            f"{model_label} & {row['system']} & "
            f"{row['ttft_avg_ms']:.1f} & {row['service_ttft_ms']:.1f} & "
            f"{row['dispatch_wait_ms']:.1f} & {row['tpot_avg_ms']:.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (out_dir / "table_ttft_decomposition.tex").write_text("\n".join(lines), encoding="utf-8")


def build_lifecycle_figure(models: Sequence[ModelRound], out_dir: Path) -> None:
    rows = [row for model in models for row in _system_rows(model)]
    _write_csv(out_dir / "fig7_lifecycle_cost_data.csv", rows)

    labels: list[str] = []
    systems: list[ppf.MainSystemData] = []
    y_positions: list[float] = []
    y = 0.0
    group_centers: list[tuple[str, float]] = []
    for model in models:
        start = y
        for system in _ordered_systems(model.systems):
            labels.append(system.label)
            systems.append(system)
            y_positions.append(y)
            y += 1.0
        group_centers.append((model.label.replace("Llama-2 ", ""), (start + y - 1.0) / 2.0))
        y += 0.58

    y_arr = np.asarray(y_positions, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(3.50, 3.42), constrained_layout=False)
    components = [
        ("Startup", "cost_startup_usd", "#A9C4E8"),
        ("Active", "cost_active_usd", "#A7D3A8"),
        ("Idle-ready", "cost_idle_ready_usd", "#F4C58D"),
        ("Invocation", "cost_invocation_usd", "#D8B6D9"),
    ]
    bottom = np.zeros(len(systems))
    legend_handles = []
    legend_labels = []
    for name, key, color in components:
        vals = np.asarray([system.metrics[key] * 1000.0 for system in systems])
        bars = axes[0].barh(y_arr, vals, left=bottom, height=0.54, color=color, edgecolor="#555555", linewidth=0.25)
        if np.any(vals > 0):
            legend_handles.append(bars[0])
            legend_labels.append(name)
        bottom += vals
    axes[0].set_yticks(y_arr)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_ylim(float(y_arr[-1] + 0.30), float(y_arr[0] - 0.295))
    ppf._xlabel_with_panel(axes[0], "Cost/req (mUSD)", "(a) Cost")
    ppf._style_axes(axes[0])

    gpu_components = [
        ("Startup", "infra_startup_gpu_seconds", "#A9C4E8"),
        ("Active serving", "infra_active_gpu_seconds", "#A7D3A8"),
        ("Idle-ready", "infra_idle_ready_gpu_seconds", "#F4C58D"),
    ]
    bottom = np.zeros(len(systems))
    for _, key, color in gpu_components:
        vals = np.asarray([system.metrics[key] / system.metrics["completed"] for system in systems])
        axes[1].barh(y_arr, vals, left=bottom, height=0.54, color=color, edgecolor="#555555", linewidth=0.25)
        bottom += vals
    axes[1].set_yticks(y_arr)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_ylim(float(y_arr[-1] + 0.30), float(y_arr[0] - 0.295))
    ppf._xlabel_with_panel(axes[1], "GPU-s/req", "(b) GPU time")
    ppf._style_axes(axes[1])

    for ax in axes:
        ax.tick_params(axis="both", labelsize=7.0)
        ax.xaxis.label.set_size(7.3)
        ax.xaxis.labelpad = 4.0
        ax.grid(axis="x", color="#E7E7E7", linewidth=0.45)
        xmin, xmax = ax.get_xlim()
        for _, center in group_centers[:-1]:
            ax.axhline(center + 2.86, color="#B8B8B8", linewidth=0.5, linestyle=":")
        ax.set_xlim(xmin, xmax)
    for text, center in group_centers:
        axes[0].text(
            -0.72,
            center,
            text,
            transform=axes[0].get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            clip_on=False,
        )

    fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        fontsize=6.7,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.61, 0.900),
        columnspacing=0.64,
        handlelength=1.1,
    )
    fig.subplots_adjust(left=0.43, right=0.99, top=0.858, bottom=0.125, wspace=0.23)

    pdf = out_dir / "fig7_lifecycle_cost.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def write_manifest(models: Sequence[ModelRound], out_dir: Path) -> None:
    payload = {
        "artifact": "main_7b13b_tables_and_lifecycle_figure",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": [
            {
                "label": model.label,
                "round_dir": str(model.round_dir),
                "sources": [str(system.source) for system in model.systems],
            }
            for model in models
        ],
        "outputs": {
            "table1": str(out_dir / "table1_end_to_end.tex"),
            "ttft_decomposition": str(out_dir / "table_ttft_decomposition.tex"),
            "fig7": str(out_dir / "fig7_lifecycle_cost.pdf"),
        },
    }
    (out_dir / "main_7b13b_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_model_round(
    key: str,
    label: str,
    round_dir: Path,
    *,
    system_round_overrides: dict[str, Path] | None = None,
    allow_missing_systems: set[str] | None = None,
) -> ModelRound:
    system_round_overrides = system_round_overrides or {}
    allow_missing_systems = allow_missing_systems or set()
    systems: list[ppf.MainSystemData] = []
    missing: list[str] = []
    for system_key in SYSTEM_ORDER:
        source_round_dir = system_round_overrides.get(system_key, round_dir)
        if source_round_dir != round_dir:
            _validate_same_shared_artifacts(round_dir, source_round_dir, model_key=key, system_key=system_key)
        run_tag = _round_run_tag(source_round_dir)
        try:
            source = ppf._main_summary_path(source_round_dir, run_tag, system_key)
        except SystemExit:
            missing.append(system_key)
            continue
        if not source.exists():
            missing.append(system_key)
            continue
        systems.append(
            ppf.MainSystemData(
                system_key,
                SYSTEM_LABELS[system_key],
                source,
                _metrics_from_summary(source, system_key),
            )
        )
    required_missing = [system for system in missing if system not in allow_missing_systems]
    if required_missing:
        override_note = ""
        if system_round_overrides:
            override_note = " overrides=" + json.dumps(
                {system: str(path) for system, path in sorted(system_round_overrides.items())},
                ensure_ascii=False,
            )
        raise SystemExit(f"{round_dir}: missing required system summaries: {required_missing}{override_note}")
    return ModelRound(key=key, label=label, round_dir=round_dir, systems=systems)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined Llama-2 7B/13B main comparison artifacts.")
    parser.add_argument("--round-7b", required=True, type=Path)
    parser.add_argument("--round-13b", required=True, type=Path)
    parser.add_argument(
        "--system-round-override",
        action="append",
        default=[],
        help="Use one system summary from a separate fair round after verifying shared trace/subset hashes. "
        "Format: MODEL_KEY:SYSTEM_KEY:ROUND_DIR.",
    )
    parser.add_argument(
        "--allow-missing-system",
        action="append",
        default=[],
        choices=SYSTEM_ORDER,
        help="Explicitly allow a missing system while building intermediate artifacts.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("figs/paper/main"))
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = _parse_system_round_overrides(args.system_round_override)
    allow_missing_systems = set(args.allow_missing_system)
    models = [
        load_model_round(
            "llama2_7b",
            "Llama-2 7B",
            args.round_7b.resolve(),
            system_round_overrides=overrides.get("llama2_7b", {}),
            allow_missing_systems=allow_missing_systems,
        ),
        load_model_round(
            "llama2_13b",
            "Llama-2 13B",
            args.round_13b.resolve(),
            system_round_overrides=overrides.get("llama2_13b", {}),
            allow_missing_systems=allow_missing_systems,
        ),
    ]
    build_main_table(models, out_dir)
    build_decomposition_table(models, out_dir)
    build_lifecycle_figure(models, out_dir)
    write_manifest(models, out_dir)
    print(f"wrote combined main artifacts to {out_dir}")


if __name__ == "__main__":
    main()
