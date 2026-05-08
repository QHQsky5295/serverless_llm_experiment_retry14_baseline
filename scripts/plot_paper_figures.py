#!/usr/bin/env python3
"""Generate publication figures from PrimeLoRA paper experiment rounds.

The script intentionally reads only completed, audited JSON results. It writes
the PDF figure, a CSV data dump, and a small manifest for each generated figure.
Missing fields fail fast instead of being silently converted to zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _configure_matplotlib() -> None:
    """Use a compact, system-paper style shared by all generated figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.titlesize": 10.4,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 9.0,
            "axes.linewidth": 0.65,
            "grid.linewidth": 0.55,
            "lines.linewidth": 1.35,
            "patch.linewidth": 0.45,
        }
    )


_configure_matplotlib()


SCENARIOS = ("faaslora_nvme", "faaslora_no_coord", "faaslora_full")
SCENARIO_LABELS = {
    "faaslora_nvme": "NVMe",
    "faaslora_no_coord": "NoCoord",
    "faaslora_full": "Full",
}
COLORS = {
    "faaslora_nvme": "#7FA7D9",
    "faaslora_no_coord": "#F2B36D",
    "faaslora_full": "#78B87A",
    "avg": "#7FA7D9",
    "p95": "#E88989",
}

SYSTEM_ORDER = ("faaslora", "sglang", "vllm", "slora", "serverlessllm")
SYSTEM_LABELS = {
    "faaslora": "PrimeLoRA",
    "sglang": "SGLang",
    "vllm": "vLLM",
    "slora": "S-LoRA",
    "serverlessllm": "ServerlessLLM",
}
SYSTEM_COLORS = {
    "faaslora": "#78B87A",
    "sglang": "#7FA7D9",
    "vllm": "#F2B36D",
    "slora": "#8FD0C7",
    "serverlessllm": "#E88989",
}
AXIS_SYSTEM_LABELS = {
    "faaslora": "PrimeLoRA",
    "sglang": "SGLang",
    "vllm": "vLLM",
    "slora": "S-LoRA",
    "serverlessllm": "ServerlessLLM",
}
METRIC_COLORS = {
    "ttft": "#7FA7D9",
    "e2e": "#8FD0C7",
    "cost": "#F2B36D",
    "ce": "#78B87A",
}
DOUBLE_COL_FIGSIZE = (7.16, 3.15)
DOUBLE_COL_TALL_FIGSIZE = (7.16, 5.9)
SINGLE_COL_MOTIVATION_FIGSIZE = (3.45, 4.45)
PANEL_TITLE_FONTSIZE = 10.4
TICK_FONTSIZE = 9.2
LEGEND_FONTSIZE = 9.0
ANNOTATION_FONTSIZE = 9.2
MOTIVATION_LABEL_FONTSIZE = 10.8
MOTIVATION_TICK_FONTSIZE = 9.8
MOTIVATION_LEGEND_FONTSIZE = 9.0
MOTIVATION_ANNOTATION_FONTSIZE = 9.5
MOTIVATION_SMALL_TEXT_FONTSIZE = 8.7


@dataclass
class ScenarioData:
    name: str
    source: Path
    summary: Dict[str, Any]
    requests: List[Dict[str, Any]]


@dataclass
class MainSystemData:
    key: str
    label: str
    source: Path
    metrics: Dict[str, float]


def _require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"required file not found: {path}")


def _load_json(path: Path) -> Dict[str, Any]:
    _require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any, label: str) -> float:
    if value is None:
        raise SystemExit(f"missing numeric field: {label}")
    try:
        out = float(value)
    except Exception as exc:  # pragma: no cover - defensive CLI path
        raise SystemExit(f"non-numeric field {label}: {value!r}") from exc
    if not math.isfinite(out):
        raise SystemExit(f"non-finite field {label}: {value!r}")
    return out


def _summary_float(scenario: ScenarioData, key: str) -> float:
    return _as_float(scenario.summary.get(key), f"{scenario.name}.{key}")


def _request_float(request: Dict[str, Any], key: str, scenario: str) -> float:
    return _as_float(request.get(key), f"{scenario}.request[{request.get('request_id')}].{key}")


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise SystemExit("cannot compute percentile over empty values")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise SystemExit("cannot compute mean over empty values")
    return float(np.mean(np.asarray(values, dtype=float)))


def _observed_tpot_values(requests: Sequence[Dict[str, Any]], label: str) -> List[float]:
    values: List[float] = []
    for idx, request in enumerate(requests):
        if request.get("tpot_ms") is None:
            continue
        if request.get("tpot_observed") is False:
            continue
        values.append(_as_float(request.get("tpot_ms"), f"{label}.request[{idx}].tpot_ms"))
    if not values:
        raise SystemExit(f"{label}: no observed TPOT samples")
    return values


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise SystemExit(f"no rows to write: {path}")
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(
    path: Path,
    figure: str,
    round_dir: Path,
    output_path: Path,
    csv_path: Path,
    sources: Iterable[Path],
    *,
    output_key: str = "pdf",
    extra: Dict[str, Any] | None = None,
) -> None:
    payload = {
        "figure": figure,
        "round_dir": str(round_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        output_key: str(output_path),
        "csv": str(csv_path),
        "sources": [str(p) for p in sources],
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _style_xgrid_axes(ax: plt.Axes) -> None:
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _xlabel_with_panel(ax: plt.Axes, xlabel: str, panel_caption: str) -> None:
    ax.set_xlabel(f"{xlabel}\n{panel_caption}" if xlabel else panel_caption)


def _use_motivation_fonts(axes: Sequence[plt.Axes]) -> None:
    for ax in axes:
        ax.xaxis.label.set_size(MOTIVATION_LABEL_FONTSIZE)
        ax.yaxis.label.set_size(MOTIVATION_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=MOTIVATION_TICK_FONTSIZE)


def _annotate_value(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    xoffset: float = 8.0,
    yoffset: float = 0.0,
    ha: str | None = None,
    fontsize: float = ANNOTATION_FONTSIZE,
) -> None:
    align = ha or ("left" if xoffset >= 0 else "right")
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(xoffset, yoffset),
        textcoords="offset points",
        ha=align,
        va="center",
        fontsize=fontsize,
        bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        zorder=4,
    )


def _annotate_barh_value(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    fontsize: float = ANNOTATION_FONTSIZE,
) -> None:
    _annotate_value(
        ax,
        x,
        y,
        text,
        xoffset=7.0 if x >= 0 else -7.0,
        fontsize=fontsize,
    )


def _signed_pct_text(value: float, suffix: str = "%") -> str:
    if abs(value) < 0.05:
        return f"0.0{suffix}"
    return f"{value:+.1f}{suffix}"


def _improvement_pct(baseline: float, value: float, *, higher_is_better: bool) -> float:
    """Return percent improvement relative to a baseline; positive is better."""
    if baseline == 0:
        raise SystemExit("cannot compute relative improvement with zero baseline")
    if higher_is_better:
        return (value / baseline - 1.0) * 100.0
    if value == 0:
        raise SystemExit("cannot compute lower-is-better improvement with zero value")
    return (baseline / value - 1.0) * 100.0


def _plot_ecdf(ax: plt.Axes, values: Sequence[float], *, label: str, color: str) -> None:
    if not values:
        return
    arr = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(arr) + 1, dtype=float) / len(arr)
    ax.step(arr, y, where="post", label=label, color=color, linewidth=1.55)


def _plot_change_panel(
    ax: plt.Axes,
    metric_labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], str]],
    *,
    title: str,
    xlabel: str = "Change vs reference (%)",
    label_suffix: str = "%",
    min_span: float = 1.0,
) -> None:
    y = np.arange(len(metric_labels), dtype=float)
    offsets = np.linspace(-0.13, 0.13, len(series)) if len(series) > 1 else np.array([0.0])
    all_values: List[float] = [0.0]
    for _, values, _ in series:
        all_values.extend(float(v) for v in values)

    lo = min(all_values)
    hi = max(all_values)
    span = max(hi - lo, min_span)
    pad = span * 0.22
    ax.set_xlim(lo - pad, hi + pad)
    text_pad = span * 0.035

    for offset, (name, values, color) in zip(offsets, series):
        yy = y + offset
        vals = np.asarray(values, dtype=float)
        for yi, val in zip(yy, vals):
            ax.hlines(yi, 0, val, color=color, linewidth=1.5, alpha=0.82)
        ax.scatter(vals, yy, s=42, color=color, edgecolor="#333333", linewidth=0.4, label=name, zorder=3)
        for yi, val in zip(yy, vals):
            _annotate_value(
                ax,
                float(val),
                float(yi),
                _signed_pct_text(float(val), label_suffix),
                xoffset=8.0 if val >= 0 else -8.0,
            )

    ax.axvline(0, color="#4D4D4D", linewidth=0.8, linestyle="--")
    ax.set_yticks(y, metric_labels, fontsize=TICK_FONTSIZE)
    ax.invert_yaxis()
    _xlabel_with_panel(ax, xlabel, title)
    _style_xgrid_axes(ax)


def _plot_delta_bar_panel(
    ax: plt.Axes,
    metric_labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], str]],
    *,
    title: str,
    xlabel: str = "Change vs reference (%)",
    label_suffix: str = "%",
    min_span: float = 1.0,
) -> None:
    y = np.arange(len(metric_labels), dtype=float)
    nseries = max(len(series), 1)
    bar_height = min(0.34, 0.62 / nseries)
    offsets = np.linspace(-0.20, 0.20, nseries) if nseries > 1 else np.array([0.0])
    all_values: List[float] = [0.0]
    for _, values, _ in series:
        all_values.extend(float(v) for v in values)

    lo = min(all_values)
    hi = max(all_values)
    span = max(hi - lo, min_span)
    pad = span * 0.30
    ax.set_xlim(lo - pad, hi + pad)

    for offset, (name, values, color) in zip(offsets, series):
        yy = y + offset
        vals = np.asarray(values, dtype=float)
        ax.barh(
            yy,
            vals,
            height=bar_height,
            color=color,
            edgecolor="#333333",
            linewidth=0.35,
            alpha=0.86,
            label=name,
            zorder=2,
        )
        for yi, val in zip(yy, vals):
            label_x = 0.0 if abs(float(val)) < 0.05 else float(val)
            _annotate_barh_value(ax, label_x, float(yi), _signed_pct_text(float(val), label_suffix))

    ax.axvline(0, color="#4D4D4D", linewidth=0.8, linestyle="--")
    ax.set_yticks(y, metric_labels, fontsize=TICK_FONTSIZE)
    ax.invert_yaxis()
    _xlabel_with_panel(ax, xlabel, title)
    _style_xgrid_axes(ax)


def _improvement_shade(value: float | None) -> str:
    if value is None:
        return "#F2F2F2"
    magnitude = min(abs(value) / 30.0, 1.0)
    if abs(value) < 0.05:
        return "#F7F7F7"
    if value > 0:
        palette = ["#EAF4EA", "#D6EBD6", "#B8DDB9", "#8CC98E"]
    else:
        palette = ["#F8ECEA", "#F3D6D3", "#EBB6B2", "#DD8580"]
    return palette[min(int(magnitude * len(palette)), len(palette) - 1)]


def _draw_ablation_metric_matrix(
    ax: plt.Axes,
    rows: Sequence[Dict[str, Any]],
    reference: Dict[str, Any],
    metric_specs: Sequence[tuple[str, str, bool, float, str]],
    *,
    group_label: str | None = None,
) -> None:
    ax.set_xlim(0, len(metric_specs))
    ax.set_ylim(0, len(rows))
    main_font = 7.5 if len(metric_specs) >= 6 else 7.8
    delta_font = 6.7 if len(metric_specs) >= 6 else 7.0
    for yi, row in enumerate(rows):
        for xi, (label, key, higher_is_better, scale, fmt) in enumerate(metric_specs):
            is_reference = row["scenario"] == reference["scenario"]
            value = float(row[key]) * scale
            change = None if is_reference else _improvement_pct(reference[key], row[key], higher_is_better=higher_is_better)
            rect = plt.Rectangle(
                (xi, yi),
                1,
                1,
                facecolor=_improvement_shade(change),
                edgecolor="white",
                linewidth=0.95,
            )
            ax.add_patch(rect)
            main_text = fmt.format(value)
            change_text = "ref" if is_reference else _signed_pct_text(float(change))
            ax.text(xi + 0.5, yi + 0.38, main_text, ha="center", va="center", fontsize=main_font, color="#111111")
            ax.text(xi + 0.5, yi + 0.68, change_text, ha="center", va="center", fontsize=delta_font, color="#4A4A4A")

    ax.set_xticks(np.arange(len(metric_specs)) + 0.5, [spec[0] for spec in metric_specs])
    ax.set_yticks(np.arange(len(rows)) + 0.5, [row["label"] for row in rows])
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=6.7, length=0, pad=1.8, top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=7.7, length=0, pad=2.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if group_label:
        ax.text(0.0, 1.17, group_label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, fontweight="bold")


def _add_bar_labels(ax: plt.Axes, bars: Iterable[Any], *, fmt: str = "{:.0f}", padding_frac: float = 0.012) -> None:
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * padding_frac
    for bar in bars:
        height = float(bar.get_height())
        if not math.isfinite(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + pad,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONTSIZE,
        )


def _add_point_labels(ax: plt.Axes, xs: Sequence[float], ys: Sequence[float], *, fmt: str = "{:+.1f}%", dy_frac: float = 0.025) -> None:
    ymin, ymax = ax.get_ylim()
    dy = (ymax - ymin) * dy_frac
    for x, y in zip(xs, ys):
        ax.text(x, y + dy, fmt.format(y), ha="center", va="bottom", fontsize=ANNOTATION_FONTSIZE)


def _bar_with_labels(ax: plt.Axes, xs: Sequence[float], vals: Sequence[float], *, width: float, color: str, label: str | None = None, fmt: str = "{:.0f}") -> None:
    bars = ax.bar(xs, vals, width=width, color=color, label=label, edgecolor="#333333", linewidth=0.4)
    _add_bar_labels(ax, bars, fmt=fmt)


def _round_data(round_dir: Path) -> Dict[str, ScenarioData]:
    manifest = _load_json(round_dir / "MANIFEST.json")
    raw_dir = round_dir / "raw" / "faaslora"
    data: Dict[str, ScenarioData] = {}
    for scenario in SCENARIOS:
        result_path = raw_dir / f"{manifest['run_tag']}_{scenario}_result.json"
        if not result_path.exists():
            continue
        payload = _load_json(result_path)
        if payload.get("metric_schema_version") != "e2e_v3":
            raise SystemExit(f"{result_path}: metric_schema_version must be e2e_v3")
        summaries = payload.get("scenario_summaries") or {}
        details = payload.get("detailed_results") or {}
        if scenario not in summaries:
            raise SystemExit(f"{result_path}: missing summary for {scenario}")
        if scenario not in details:
            raise SystemExit(f"{result_path}: missing detailed results for {scenario}")
        summary = summaries[scenario]
        total = int(summary.get("total_requests", -1))
        completed = int(summary.get("completed_requests", -1))
        failed = int(summary.get("failed_requests", 0) or 0)
        if total <= 0 or completed != total or failed != 0:
            raise SystemExit(f"{result_path}: invalid completion total={total} completed={completed} failed={failed}")
        requests = details[scenario].get("requests") or []
        if len(requests) != total:
            raise SystemExit(f"{result_path}: request count mismatch total={total} requests={len(requests)}")
        data[scenario] = ScenarioData(scenario, result_path, summary, requests)
    return data


def _require_scenarios(data: Dict[str, ScenarioData], scenarios: Sequence[str]) -> List[ScenarioData]:
    missing = [scenario for scenario in scenarios if scenario not in data]
    if missing:
        raise SystemExit(f"missing required scenarios: {missing}")
    return [data[scenario] for scenario in scenarios]


def _system_key(raw_name: str) -> str:
    name = raw_name.lower()
    if "faaslora" in name or "primelora" in name:
        return "faaslora"
    if "sglang" in name:
        return "sglang"
    if "serverlessllm" in name:
        return "serverlessllm"
    if "vllm" in name:
        return "vllm"
    if "s-lora" in name or "slora" in name:
        return "slora"
    raise SystemExit(f"unknown system label: {raw_name}")


def _row_dict(headers: Sequence[str], row: Sequence[Any]) -> Dict[str, Any]:
    return {str(k): row[i] if i < len(row) else None for i, k in enumerate(headers)}


def _find_main_compare(round_dir: Path, manifest: Dict[str, Any]) -> Path:
    from_manifest = manifest.get("compare_json")
    if from_manifest:
        path = Path(str(from_manifest))
        if path.exists():
            return path
    matches = sorted((round_dir / "compare").glob("*five_system_compare.json"))
    if len(matches) != 1:
        raise SystemExit(f"expected one main compare JSON under {round_dir / 'compare'}, found {len(matches)}")
    return matches[0]


def _main_summary_path(round_dir: Path, run_tag: str, key: str) -> Path:
    if key == "faaslora":
        return round_dir / "raw" / "faaslora" / f"{run_tag}_faaslora_result.json"
    suffixes = {
        "sglang": "sglang_dp4_tp1_summary.json",
        "serverlessllm": "serverlessllm_summary.json",
        "vllm": "vllm_dp4_tp1_summary.json",
        "slora": "slora_dp4_tp1_summary.json",
    }
    if key not in suffixes:
        raise SystemExit(f"unknown main system key: {key}")
    default = round_dir / "raw" / "replay" / f"{run_tag}_{suffixes[key]}"
    if default.exists():
        return default
    matches = sorted((round_dir / "raw" / "replay").glob(f"{run_tag}_{key}_*_summary.json"))
    if len(matches) == 1:
        return matches[0]
    if key == "slora":
        matches = sorted((round_dir / "raw" / "replay").glob(f"{run_tag}_slora_*_summary.json"))
        if len(matches) == 1:
            return matches[0]
    raise SystemExit(f"expected one summary for {key} under {round_dir / 'raw' / 'replay'}, found {len(matches)}")


def _main_replay_path(round_dir: Path, run_tag: str, key: str) -> Path:
    if key == "faaslora":
        return _main_summary_path(round_dir, run_tag, key)
    suffixes = {
        "sglang": "sglang_dp4_tp1_replay.json",
        "serverlessllm": "serverlessllm_replay.json",
        "vllm": "vllm_dp4_tp1_replay.json",
        "slora": "slora_dp4_tp1_replay.json",
    }
    if key not in suffixes:
        raise SystemExit(f"unknown main system key: {key}")
    default = round_dir / "raw" / "replay" / f"{run_tag}_{suffixes[key]}"
    if default.exists():
        return default
    matches = sorted((round_dir / "raw" / "replay").glob(f"{run_tag}_{key}_*_replay.json"))
    if len(matches) == 1:
        return matches[0]
    if key == "slora":
        matches = sorted((round_dir / "raw" / "replay").glob(f"{run_tag}_slora_*_replay.json"))
        if len(matches) == 1:
            return matches[0]
    raise SystemExit(f"expected one replay for {key} under {round_dir / 'raw' / 'replay'}, found {len(matches)}")


def _main_row_from_summary(path: Path, key: str) -> Dict[str, Any]:
    payload = _load_json(path)
    if payload.get("metric_schema_version") != "e2e_v3":
        raise SystemExit(f"{path}: metric_schema_version must be e2e_v3")
    if key == "faaslora":
        summary = (payload.get("scenario_summaries") or {}).get("faaslora_full")
        if not summary:
            raise SystemExit(f"{path}: missing scenario_summaries.faaslora_full")
        requests = (((payload.get("detailed_results") or {}).get("faaslora_full") or {}).get("requests") or [])
        tpot_values = _observed_tpot_values(requests, "faaslora_full")
        return {
            "completed": summary.get("completed_requests"),
            "total": summary.get("total_requests"),
            "TTFT_avg_ms": summary.get("avg_overall_ttft_ms"),
            "TTFT_p95_ms": summary.get("p95_overall_ttft_ms"),
            "E2E_avg_ms": summary.get("avg_overall_e2e_ms"),
            "E2E_p95_ms": summary.get("p95_overall_e2e_ms"),
            "TPOT_avg_ms": summary.get("avg_tpot_ms"),
            "TPOT_p95_ms": summary.get("p95_tpot_ms") or _percentile(tpot_values, 95),
            "Tok_s": summary.get("throughput_tok_per_s"),
            "Cost_req_usd": summary.get("monetary_cost_per_request_usd"),
            "CE": summary.get("monetary_ce"),
            "cost_per_1m_total_tokens_usd": summary.get("cost_per_1m_total_tokens_usd"),
            "monetary_cost_total_usd": summary.get("monetary_cost_total_usd"),
            "monetary_active_charge_gpu_seconds": summary.get("monetary_active_charge_gpu_seconds"),
            "monetary_idle_charge_gpu_seconds": summary.get("monetary_idle_charge_gpu_seconds"),
            "infra_active_gpu_seconds": summary.get("infra_active_gpu_seconds"),
            "infra_idle_ready_gpu_seconds": summary.get("infra_idle_ready_gpu_seconds"),
            "infra_startup_gpu_seconds": summary.get("infra_startup_gpu_seconds"),
            "serverless_invocation_cost_per_request_usd": summary.get("serverless_invocation_cost_per_request_usd"),
            "monetary_pricing_runtime_class": summary.get("monetary_pricing_runtime_class"),
        }

    table = payload.get("comparison_table") or []
    if len(table) != 1:
        raise SystemExit(f"{path}: expected one comparison_table row, found {len(table)}")
    row = table[0]
    replay_path = path.with_name(path.name.replace("_summary.json", "_replay.json"))
    replay = _load_json(replay_path)
    if replay.get("metric_schema_version") != "e2e_v3":
        raise SystemExit(f"{replay_path}: metric_schema_version must be e2e_v3")
    tpot_values = _observed_tpot_values(replay.get("results") or [], key)
    return {
        "completed": row.get("completed"),
        "total": row.get("total"),
        "TTFT_avg_ms": row.get("TTFT_e2e_avg_ms"),
        "TTFT_p95_ms": row.get("TTFT_e2e_P95_ms"),
        "E2E_avg_ms": row.get("E2E_avg_ms"),
        "E2E_p95_ms": row.get("E2E_P95_ms"),
        "TPOT_avg_ms": row.get("TPOT_avg_ms"),
        "TPOT_p95_ms": row.get("TPOT_P95_ms") or _percentile(tpot_values, 95),
        "Tok_s": row.get("throughput_TOKPS"),
        "Cost_req_usd": row.get("monetary_cost_per_request_usd"),
        "CE": row.get("monetary_ce"),
        "cost_per_1m_total_tokens_usd": row.get("cost_per_1m_total_tokens_usd"),
        "monetary_cost_total_usd": row.get("monetary_cost_total_usd"),
        "monetary_active_charge_gpu_seconds": row.get("monetary_active_charge_gpu_seconds"),
        "monetary_idle_charge_gpu_seconds": row.get("monetary_idle_charge_gpu_seconds"),
        "infra_active_gpu_seconds": row.get("infra_active_gpu_seconds"),
        "infra_idle_ready_gpu_seconds": row.get("infra_idle_ready_gpu_seconds"),
        "infra_startup_gpu_seconds": row.get("infra_startup_gpu_seconds"),
        "serverless_invocation_cost_per_request_usd": row.get("serverless_invocation_cost_per_request_usd"),
        "monetary_pricing_runtime_class": row.get("monetary_pricing_runtime_class"),
    }


def _main_round_data(round_dir: Path) -> List[MainSystemData]:
    manifest = _load_json(round_dir / "MANIFEST.json")
    if manifest.get("metric_schema_version") != "e2e_v3":
        raise SystemExit(f"{round_dir / 'MANIFEST.json'}: metric_schema_version must be e2e_v3")
    run_tag = str(manifest.get("run_tag") or "")
    if not run_tag:
        raise SystemExit(f"{round_dir / 'MANIFEST.json'}: missing run_tag")

    compare_path = _find_main_compare(round_dir, manifest)
    compare = _load_json(compare_path)
    if compare.get("metric_schema_version") != "e2e_v3":
        raise SystemExit(f"{compare_path}: metric_schema_version must be e2e_v3")

    strict_rows = [_row_dict(compare.get("strict_headers") or [], row) for row in compare.get("strict_rows") or []]
    present_keys = {_system_key(str(row.get("System"))) for row in strict_rows}
    missing = [key for key in SYSTEM_ORDER if key not in present_keys]
    if missing:
        raise SystemExit(f"{compare_path}: missing systems in strict_rows: {missing}")

    systems: List[MainSystemData] = []
    for key in SYSTEM_ORDER:
        source = _main_summary_path(round_dir, run_tag, key)
        raw = _main_row_from_summary(source, key)
        completed = int(_as_float(raw.get("completed"), f"{key}.completed"))
        total = int(_as_float(raw.get("total"), f"{key}.total"))
        if completed <= 0 or completed != total:
            raise SystemExit(f"{source}: invalid completion completed={completed} total={total}")

        metrics = {
            "completed": float(completed),
            "ttft_avg_ms": _as_float(raw.get("TTFT_avg_ms"), f"{key}.TTFT_avg_ms"),
            "ttft_p95_ms": _as_float(raw.get("TTFT_p95_ms"), f"{key}.TTFT_p95_ms"),
            "e2e_avg_ms": _as_float(raw.get("E2E_avg_ms"), f"{key}.E2E_avg_ms"),
            "e2e_p95_ms": _as_float(raw.get("E2E_p95_ms"), f"{key}.E2E_p95_ms"),
            "tpot_avg_ms": _as_float(raw.get("TPOT_avg_ms"), f"{key}.TPOT_avg_ms"),
            "tpot_p95_ms": _as_float(raw.get("TPOT_p95_ms"), f"{key}.TPOT_p95_ms"),
            "tok_s": _as_float(raw.get("Tok_s"), f"{key}.Tok_s"),
            "cost_req_usd": _as_float(raw.get("Cost_req_usd"), f"{key}.Cost_req_usd"),
            "ce": _as_float(raw.get("CE"), f"{key}.CE"),
            "cost_1mtok_usd": _as_float(raw.get("cost_per_1m_total_tokens_usd"), f"{key}.cost_per_1m_total_tokens_usd"),
            "monetary_cost_total_usd": _as_float(raw.get("monetary_cost_total_usd"), f"{key}.monetary_cost_total_usd"),
            "monetary_active_charge_gpu_seconds": _as_float(raw.get("monetary_active_charge_gpu_seconds"), f"{key}.monetary_active_charge_gpu_seconds"),
            "monetary_idle_charge_gpu_seconds": _as_float(raw.get("monetary_idle_charge_gpu_seconds"), f"{key}.monetary_idle_charge_gpu_seconds"),
            "infra_active_gpu_seconds": _as_float(raw.get("infra_active_gpu_seconds"), f"{key}.infra_active_gpu_seconds"),
            "infra_idle_ready_gpu_seconds": _as_float(raw.get("infra_idle_ready_gpu_seconds"), f"{key}.infra_idle_ready_gpu_seconds"),
            "infra_startup_gpu_seconds": _as_float(raw.get("infra_startup_gpu_seconds"), f"{key}.infra_startup_gpu_seconds"),
            "serverless_invocation_cost_per_request_usd": _as_float(raw.get("serverless_invocation_cost_per_request_usd"), f"{key}.serverless_invocation_cost_per_request_usd"),
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
        systems.append(MainSystemData(key, SYSTEM_LABELS[key], source, metrics))
    return systems


def _main_faaslora_full(round_dir: Path) -> ScenarioData:
    manifest = _load_json(round_dir / "MANIFEST.json")
    run_tag = str(manifest.get("run_tag") or "")
    if not run_tag:
        raise SystemExit(f"{round_dir / 'MANIFEST.json'}: missing run_tag")
    source = _main_summary_path(round_dir, run_tag, "faaslora")
    payload = _load_json(source)
    summary = (payload.get("scenario_summaries") or {}).get("faaslora_full")
    details = (payload.get("detailed_results") or {}).get("faaslora_full") or {}
    requests = details.get("requests") or []
    if not isinstance(summary, dict):
        raise SystemExit(f"{source}: missing faaslora_full summary")
    if not requests:
        raise SystemExit(f"{source}: missing faaslora_full request details")
    return ScenarioData("faaslora_full", source, summary, requests)


def plot_fig2(round_dir: Path, out_dir: Path) -> None:
    manifest_payload = _load_json(round_dir / "MANIFEST.json")
    run_tag = str(manifest_payload.get("run_tag") or "")
    if not run_tag:
        raise SystemExit(f"{round_dir / 'MANIFEST.json'}: missing run_tag")
    source = _main_replay_path(round_dir, run_tag, "serverlessllm")
    replay = _load_json(source)
    if replay.get("metric_schema_version") != "e2e_v3":
        raise SystemExit(f"{source}: metric_schema_version must be e2e_v3")
    requests = [r for r in replay.get("results") or [] if str(r.get("status", "ok")).lower() == "ok"]
    if not requests:
        raise SystemExit(f"{source}: no completed ServerlessLLM requests")

    def field_values(reqs: Sequence[Dict[str, Any]], keys: Sequence[str], label: str) -> List[float]:
        values: List[float] = []
        for idx, request in enumerate(reqs):
            value = None
            used_key = keys[0]
            for key in keys:
                if request.get(key) is not None:
                    value = request.get(key)
                    used_key = key
                    break
            if value is None:
                continue
            values.append(_as_float(value, f"{label}.request[{idx}].{used_key}"))
        if not values:
            raise SystemExit(f"{label}: no samples for {keys}")
        return values

    def metric_row(panel: str, category: str, reqs: Sequence[Dict[str, Any]], field: str, values: Sequence[float]) -> Dict[str, Any]:
        return {
            "panel": panel,
            "system": "ServerlessLLM",
            "benchmark": "Llama-2-7B / 4000 requests / 500 adapters / Zipf 1.0 / hot set 48 / rotation 500 / time scale 8",
            "category": category,
            "field": field,
            "count": len(reqs),
            "avg_ms": _mean(values),
            "p95_ms": _percentile(values, 95),
        }

    rows: List[Dict[str, Any]] = []

    all_ttft = field_values(requests, ["overall_ttft_ms"], "serverlessllm.all")
    all_wait = field_values(
        requests,
        ["dispatch_admission_wait_ms", "replay_dispatch_wait_ms", "server_queue_wait_ms"],
        "serverlessllm.all",
    )
    all_runtime = field_values(requests, ["runtime_ttft_ms", "service_ttft_ms"], "serverlessllm.all")
    startup_requests = [r for r in requests if bool(r.get("scaleup_affected")) or bool(r.get("scaleup_first_service"))]
    if not startup_requests:
        raise SystemExit(f"{source}: no startup/scale-up affected ServerlessLLM requests")
    startup_first = [r for r in requests if bool(r.get("scaleup_first_service"))]
    startup_cold = field_values(startup_requests, ["cold_start_latency_ms"], "serverlessllm.startup")
    startup_wait = field_values(
        startup_requests,
        ["dispatch_admission_wait_ms", "replay_dispatch_wait_ms", "server_queue_wait_ms"],
        "serverlessllm.startup",
    )
    startup_runtime = field_values(startup_requests, ["runtime_ttft_ms", "service_ttft_ms"], "serverlessllm.startup")
    startup_ttft = field_values(startup_requests, ["overall_ttft_ms"], "serverlessllm.startup")

    rows.extend(
        [
            metric_row("all_request_ttft_path", "All requests", requests, "overall_ttft_ms", all_ttft),
            metric_row("all_request_ttft_path", "Admission/dispatch wait", requests, "dispatch_admission_wait_ms", all_wait),
            metric_row("all_request_ttft_path", "Runtime TTFT", requests, "runtime_ttft_ms", all_runtime),
            metric_row("startup_path", "Startup affected", startup_requests, "overall_ttft_ms", startup_ttft),
            metric_row("startup_path", "Cold-start latency", startup_requests, "cold_start_latency_ms", startup_cold),
            metric_row("startup_path", "Admission/dispatch wait", startup_requests, "dispatch_admission_wait_ms", startup_wait),
            metric_row("startup_path", "Runtime TTFT", startup_requests, "runtime_ttft_ms", startup_runtime),
            {
                "panel": "startup_counts",
                "system": "ServerlessLLM",
                "benchmark": "Llama-2-7B / 4000 requests / 500 adapters / Zipf 1.0 / hot set 48 / rotation 500 / time scale 8",
                "startup_affected_count": len(startup_requests),
                "first_service_count": len(startup_first),
                "total_requests": len(requests),
            },
        ]
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=SINGLE_COL_MOTIVATION_FIGSIZE,
        gridspec_kw={"height_ratios": [1.0, 1.05]},
        constrained_layout=True,
    )
    stat_names = ["Avg", "p95"]
    wait_vals = [_mean(all_wait) / 1000.0, _percentile(all_wait, 95) / 1000.0]
    runtime_vals = [_mean(all_runtime) / 1000.0, _percentile(all_runtime, 95) / 1000.0]
    total_vals = [_mean(all_ttft) / 1000.0, _percentile(all_ttft, 95) / 1000.0]
    yy = np.arange(len(stat_names), dtype=float)
    axes[0].barh(yy, wait_vals, height=0.34, color="#E88989", edgecolor="#333333", linewidth=0.35, label="Admission wait")
    axes[0].barh(
        yy,
        runtime_vals,
        left=wait_vals,
        height=0.34,
        color="#7FA7D9",
        edgecolor="#333333",
        linewidth=0.35,
        label="Runtime TTFT",
    )
    for yi, total, runtime in zip(yy, total_vals, runtime_vals):
        _annotate_barh_value(axes[0], total, yi, f"{total:.0f}s total", fontsize=MOTIVATION_ANNOTATION_FONTSIZE)
        axes[0].annotate(
            f"runtime {runtime:.1f}s",
            xy=(0, yi),
            xytext=(8.0, 0.0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=MOTIVATION_SMALL_TEXT_FONTSIZE,
            color="#2F4C66",
        )
    axes[0].set_yticks(yy, stat_names)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(total_vals) * 1.20)
    axes[0].legend(frameon=False, fontsize=MOTIVATION_LEGEND_FONTSIZE, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncols=2)
    _xlabel_with_panel(axes[0], "ServerlessLLM TTFT path (s)", "(a) Runtime ready is not service ready")
    _style_xgrid_axes(axes[0])

    component_labels = ["Cold\nstart", "Admission\nwait", "Runtime\nTTFT"]
    avg_components = [_mean(startup_cold) / 1000.0, _mean(startup_wait) / 1000.0, _mean(startup_runtime) / 1000.0]
    p95_components = [_percentile(startup_cold, 95) / 1000.0, _percentile(startup_wait, 95) / 1000.0, _percentile(startup_runtime, 95) / 1000.0]
    cx = np.arange(len(component_labels), dtype=float)
    width = 0.32
    axes[1].bar(
        cx - width / 2,
        avg_components,
        width=width,
        color="#A7D3A8",
        edgecolor="#333333",
        linewidth=0.35,
        label="Avg",
    )
    axes[1].bar(
        cx + width / 2,
        p95_components,
        width=width,
        color="#F2B36D",
        edgecolor="#333333",
        linewidth=0.35,
        label="p95",
    )
    ymax = max(p95_components) * 1.28
    axes[1].set_ylim(0, ymax)
    for x, avg, p95 in zip(cx, avg_components, p95_components):
        axes[1].text(x - width / 2, avg + ymax * 0.020, f"{avg:.1f}", ha="center", va="bottom", fontsize=MOTIVATION_SMALL_TEXT_FONTSIZE)
        axes[1].text(x + width / 2, p95 + ymax * 0.020, f"{p95:.1f}", ha="center", va="bottom", fontsize=MOTIVATION_SMALL_TEXT_FONTSIZE)
    axes[1].set_xticks(cx, component_labels, rotation=0)
    axes[1].set_ylabel("Latency (s)")
    axes[1].legend(frameon=False, fontsize=MOTIVATION_LEGEND_FONTSIZE, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncols=2)
    axes[1].text(
        0.98,
        0.92,
        f"startup n={len(startup_requests)}, first-service n={len(startup_first)}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=MOTIVATION_SMALL_TEXT_FONTSIZE,
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "#CFCFCF", "linewidth": 0.35},
    )
    _xlabel_with_panel(axes[1], "", "(b) Startup-affected requests")
    _style_axes(axes[1])
    _use_motivation_fonts(axes)

    pdf = out_dir / "fig2_mismatch.pdf"
    csv_path = out_dir / "fig2_mismatch_data.csv"
    manifest = out_dir / "fig2_mismatch_manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    _write_csv(csv_path, rows)
    _write_manifest(
        manifest,
        "fig2_mismatch",
        round_dir,
        pdf,
        csv_path,
        [source],
        extra={
            "system": "ServerlessLLM",
            "baseline_or_own_system": "external_baseline",
            "evidence_role": "external serverless baseline motivation",
            "figure_layout": "IEEE single-column PDF intended for includegraphics width=\\columnwidth",
            "model": "Llama-2-7B",
            "request_count": len(requests),
            "adapter_pool_size": 500,
            "trace_id": run_tag,
            "fields_used": [
                "overall_ttft_ms",
                "dispatch_admission_wait_ms",
                "runtime_ttft_ms",
                "cold_start_latency_ms",
                "scaleup_affected",
                "scaleup_first_service",
            ],
            "benchmark": "Llama-2-7B, 4000 requests, 500 adapters, Zipf=1.0, hot set=48, rotation=500, time scale=8",
            "category_definition": "Startup affected requests are those with scaleup_affected or scaleup_first_service flags in the ServerlessLLM replay.",
            "limitations": "This replay exposes serverless admission/startup readiness gaps but does not include per-request adapter-tier instrumentation.",
        },
    )


def plot_fig3(round_dir: Path, out_dir: Path) -> None:
    manifest_payload = _load_json(round_dir / "MANIFEST.json")
    run_tag = str(manifest_payload.get("run_tag") or "")
    if not run_tag:
        raise SystemExit(f"{round_dir / 'MANIFEST.json'}: missing run_tag")
    source = _main_replay_path(round_dir, run_tag, "slora")
    payload = _load_json(source)
    if payload.get("metric_schema_version") != "e2e_v3":
        raise SystemExit(f"{source}: metric_schema_version must be e2e_v3")
    requests = [r for r in payload.get("results") or [] if str(r.get("status", "ok")).lower() == "ok"]
    if not requests:
        raise SystemExit(f"{source}: no completed S-LoRA requests")
    requests = sorted(requests, key=lambda r: (_as_float(r.get("arrival_time_s"), "slora.arrival_time_s"), str(r.get("request_id") or "")))

    def classify_reuse(reqs: Sequence[Dict[str, Any]]) -> Dict[str, List[float]]:
        seen: Dict[str, int] = {}
        buckets = {
            "first_touch": [],
            "hot_reuse": [],
            "warm_reuse": [],
            "cold_reuse": [],
        }
        for idx, request in enumerate(reqs):
            adapter_id = request.get("adapter_id")
            if not adapter_id:
                raise SystemExit(f"slora.request[{idx}]: missing adapter_id")
            ttft = _as_float(request.get("overall_ttft_ms"), f"slora.request[{idx}].overall_ttft_ms")
            adapter = str(adapter_id)
            if adapter not in seen:
                buckets["first_touch"].append(ttft)
            else:
                distance = idx - seen[adapter]
                if distance <= 16:
                    buckets["hot_reuse"].append(ttft)
                elif distance <= 64:
                    buckets["warm_reuse"].append(ttft)
                else:
                    buckets["cold_reuse"].append(ttft)
            seen[adapter] = idx
        for key, values in buckets.items():
            if not values:
                raise SystemExit(f"S-LoRA: no samples for reuse bucket {key}")
        return buckets

    bucket_labels = {
        "first_touch": "First touch",
        "hot_reuse": "Hot reuse <=16",
        "warm_reuse": "Warm reuse 17-64",
        "cold_reuse": "Cold reuse >64",
    }
    bucket_colors = {
        "first_touch": "#E88989",
        "hot_reuse": "#78B87A",
        "warm_reuse": "#F2B36D",
        "cold_reuse": "#7FA7D9",
    }
    buckets = classify_reuse(requests)
    rows: List[Dict[str, Any]] = []
    hot_avg = _mean(buckets["hot_reuse"])
    hot_p95 = _percentile(buckets["hot_reuse"], 95)
    total = sum(len(values) for values in buckets.values())
    for bucket, values in buckets.items():
        avg = _mean(values)
        p95 = _percentile(values, 95)
        rows.append(
            {
                "panel": "adapter_reuse_ttft",
                "system_key": "slora",
                "system": "S-LoRA",
                "benchmark": "Llama-2-7B / 4000 requests / 500 adapters / Zipf 1.0 / hot set 48 / rotation 500 / time scale 8",
                "category": bucket_labels[bucket],
                "count": len(values),
                "fraction": len(values) / total,
                "ttft_avg_ms": avg,
                "ttft_p95_ms": p95,
                "avg_ratio_vs_hot_reuse": avg / hot_avg,
                "p95_ratio_vs_hot_reuse": p95 / hot_p95,
            }
        )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=SINGLE_COL_MOTIVATION_FIGSIZE,
        gridspec_kw={"height_ratios": [0.78, 1.32]},
        constrained_layout=True,
    )

    bar_order = ["first_touch", "hot_reuse", "warm_reuse", "cold_reuse"]
    left = 0.0
    for bucket in bar_order:
        frac = len(buckets[bucket]) / total
        axes[0].barh(
            [0],
            [frac * 100.0],
            left=left,
            height=0.38,
            color=bucket_colors[bucket],
            edgecolor="#333333",
            linewidth=0.35,
            label=bucket_labels[bucket],
        )
        if frac >= 0.08:
            axes[0].text(left + frac * 50.0, 0, f"{frac * 100:.0f}%", ha="center", va="center", fontsize=MOTIVATION_ANNOTATION_FONTSIZE)
        else:
            axes[0].text(left + frac * 100.0 + 1.4, 0.18, f"{frac * 100:.0f}%", ha="left", va="center", fontsize=MOTIVATION_SMALL_TEXT_FONTSIZE)
        rows.append(
            {
                "panel": "workload_reuse_mix",
                "category": bucket_labels[bucket],
                "count": len(buckets[bucket]),
                "fraction": frac,
            }
        )
        left += frac * 100.0
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(-0.38, 0.44)
    axes[0].set_yticks([])
    axes[0].set_xlabel("Share of requests (%)\n(a) Shared replay adapter churn")
    axes[0].legend(frameon=False, fontsize=MOTIVATION_LEGEND_FONTSIZE, loc="upper center", bbox_to_anchor=(0.5, 1.38), ncols=2)
    _style_xgrid_axes(axes[0])

    cdf_specs = [("first_touch", bucket_colors["first_touch"]), ("hot_reuse", bucket_colors["hot_reuse"]), ("cold_reuse", bucket_colors["cold_reuse"])]
    all_cdf_values: List[float] = []
    for bucket, color in cdf_specs:
        values = buckets[bucket]
        all_cdf_values.extend(values)
        row = next(item for item in rows if item.get("panel") == "adapter_reuse_ttft" and item["category"] == bucket_labels[bucket])
        label = f"{bucket_labels[bucket]} n={len(values)}, p95={row['ttft_p95_ms']:.0f}"
        _plot_ecdf(axes[1], values, label=label, color=color)

    axes[1].set_xlim(0, _percentile(all_cdf_values, 99.2) * 1.04)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("CDF")
    _xlabel_with_panel(axes[1], "TTFT (ms)", "(b) S-LoRA TTFT by reuse distance")
    axes[1].legend(frameon=False, fontsize=MOTIVATION_LEGEND_FONTSIZE, loc="lower right")
    _style_xgrid_axes(axes[1])
    _use_motivation_fonts(axes)

    pdf = out_dir / "fig3_tier.pdf"
    csv_path = out_dir / "fig3_tier_data.csv"
    manifest = out_dir / "fig3_tier_manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    _write_csv(csv_path, rows)
    _write_manifest(
        manifest,
        "fig3_tier",
        round_dir,
        pdf,
        csv_path,
        [source],
        extra={
            "system": "S-LoRA",
            "baseline_or_own_system": "external_baseline",
            "evidence_role": "external multi-LoRA/runtime motivation",
            "figure_layout": "IEEE single-column PDF intended for includegraphics width=\\columnwidth",
            "model": "Llama-2-7B",
            "request_count": len(requests),
            "adapter_pool_size": 500,
            "trace_id": run_tag,
            "fields_used": ["arrival_time_s", "request_id", "adapter_id", "overall_ttft_ms"],
            "benchmark": "Llama-2-7B, 4000 requests, 500 adapters, Zipf=1.0, hot set=48, rotation=500, time scale=8",
            "category_definition": "Adapter reuse buckets are derived only from the adapter_id sequence: first touch, hot reuse <=16 requests, warm reuse 17-64 requests, and cold reuse >64 requests.",
            "limitations": "The figure does not claim per-request cache tier or transfer latency; current baseline replays do not export those fields.",
        },
    )


def plot_fig4(round_dir: Path, out_dir: Path) -> None:
    data = _round_data(round_dir)
    scenarios = _require_scenarios(data, ["faaslora_no_coord", "faaslora_full"])
    rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        rows.append(
            {
                "scenario": scenario.name,
                "label": SCENARIO_LABELS[scenario.name],
                "ttft_avg_ms": _summary_float(scenario, "avg_overall_ttft_ms"),
                "ttft_p95_ms": _summary_float(scenario, "p95_overall_ttft_ms"),
                "e2e_avg_ms": _summary_float(scenario, "avg_overall_e2e_ms"),
                "e2e_p95_ms": _summary_float(scenario, "p95_overall_e2e_ms"),
                "tpot_avg_ms": _summary_float(scenario, "avg_tpot_ms"),
                "tpot_p95_ms": _percentile(_observed_tpot_values(scenario.requests, scenario.name), 95),
                "lora_io_ms": _summary_float(scenario, "avg_lora_io_ms"),
                "ce": _summary_float(scenario, "monetary_ce"),
                "cost_per_req_usd": _summary_float(scenario, "monetary_cost_per_request_usd"),
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL_FIGSIZE, constrained_layout=True)
    base = rows[0]
    full = rows[1]
    latency_metrics = [
        ("TTFT avg", "ttft_avg_ms", False),
        ("TTFT p95", "ttft_p95_ms", False),
        ("E2E avg", "e2e_avg_ms", False),
        ("E2E p95", "e2e_p95_ms", False),
        ("TPOT avg", "tpot_avg_ms", False),
        ("TPOT p95", "tpot_p95_ms", False),
    ]
    efficiency_metrics = [
        ("LoRA I/O", "lora_io_ms", False),
        ("Cost/req", "cost_per_req_usd", False),
        ("CE", "ce", True),
    ]
    rows.append(
        {
            "panel": "coordination_change",
            "reference": base["label"],
            "target": full["label"],
            **{
                f"{key}_improvement_pct": _improvement_pct(base[key], full[key], higher_is_better=higher)
                for _, key, higher in latency_metrics + efficiency_metrics
            },
        }
    )
    _plot_delta_bar_panel(
        axes[0],
        [label for label, _, _ in latency_metrics],
        [
            (
                "Full vs NoCoord",
                [_improvement_pct(base[key], full[key], higher_is_better=higher) for _, key, higher in latency_metrics],
                SYSTEM_COLORS["faaslora"],
            )
        ],
        title="(a) Latency change from coordination",
        xlabel="Improvement over NoCoord (%)",
        min_span=3.0,
    )
    _plot_delta_bar_panel(
        axes[1],
        [label for label, _, _ in efficiency_metrics],
        [
            (
                "Full vs NoCoord",
                [_improvement_pct(base[key], full[key], higher_is_better=higher) for _, key, higher in efficiency_metrics],
                SYSTEM_COLORS["faaslora"],
            )
        ],
        title="(b) Efficiency change from coordination",
        xlabel="Improvement over NoCoord (%)",
        min_span=2.0,
    )

    pdf = out_dir / "fig4_coordination.pdf"
    csv_path = out_dir / "fig4_coordination_data.csv"
    manifest = out_dir / "fig4_coordination_manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    _write_csv(csv_path, rows)
    _write_manifest(manifest, "fig4_coordination", round_dir, pdf, csv_path, [s.source for s in scenarios])


def plot_fig6(round_dir: Path, out_dir: Path) -> None:
    data = _round_data(round_dir)
    scenarios = _require_scenarios(data, list(SCENARIOS))
    rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        rows.append(
            {
                "scenario": scenario.name,
                "label": SCENARIO_LABELS[scenario.name],
                "ttft_avg_ms": _summary_float(scenario, "avg_overall_ttft_ms"),
                "ttft_p95_ms": _summary_float(scenario, "p95_overall_ttft_ms"),
                "e2e_avg_ms": _summary_float(scenario, "avg_overall_e2e_ms"),
                "e2e_p95_ms": _summary_float(scenario, "p95_overall_e2e_ms"),
                "tpot_avg_ms": _summary_float(scenario, "avg_tpot_ms"),
                "tpot_p95_ms": _percentile(_observed_tpot_values(scenario.requests, scenario.name), 95),
                "tok_s": _summary_float(scenario, "throughput_tok_per_s"),
                "gpu_hit_rate_pct": _summary_float(scenario, "gpu_hit_rate") * 100,
                "lora_io_ms": _summary_float(scenario, "avg_lora_io_ms"),
                "dispatch_wait_ms": _summary_float(scenario, "avg_dispatch_admission_wait_ms"),
                "cost_per_req_usd": _summary_float(scenario, "monetary_cost_per_request_usd"),
                "ce": _summary_float(scenario, "monetary_ce"),
            }
        )

    reference = rows[0]
    compared = rows[1:]

    for row in compared:
        row["reference"] = reference["label"]
        row["ttft_avg_improvement_pct"] = _improvement_pct(reference["ttft_avg_ms"], row["ttft_avg_ms"], higher_is_better=False)
        row["ttft_p95_improvement_pct"] = _improvement_pct(reference["ttft_p95_ms"], row["ttft_p95_ms"], higher_is_better=False)
        row["e2e_avg_improvement_pct"] = _improvement_pct(reference["e2e_avg_ms"], row["e2e_avg_ms"], higher_is_better=False)
        row["e2e_p95_improvement_pct"] = _improvement_pct(reference["e2e_p95_ms"], row["e2e_p95_ms"], higher_is_better=False)
        row["tpot_avg_improvement_pct"] = _improvement_pct(reference["tpot_avg_ms"], row["tpot_avg_ms"], higher_is_better=False)
        row["tpot_p95_improvement_pct"] = _improvement_pct(reference["tpot_p95_ms"], row["tpot_p95_ms"], higher_is_better=False)
        row["tok_s_improvement_pct"] = _improvement_pct(reference["tok_s"], row["tok_s"], higher_is_better=True)
        row["lora_io_improvement_pct"] = _improvement_pct(reference["lora_io_ms"], row["lora_io_ms"], higher_is_better=False)
        row["dispatch_wait_improvement_pct"] = _improvement_pct(reference["dispatch_wait_ms"], row["dispatch_wait_ms"], higher_is_better=False)
        row["cost_improvement_pct"] = _improvement_pct(reference["cost_per_req_usd"], row["cost_per_req_usd"], higher_is_better=False)
        row["ce_improvement_pct"] = _improvement_pct(reference["ce"], row["ce"], higher_is_better=True)

    latency_metric_specs = [
        ("TTFT avg\n(ms)", "ttft_avg_ms", False, 1.0, "{:.0f}"),
        ("TTFT p95\n(ms)", "ttft_p95_ms", False, 1.0, "{:.0f}"),
        ("E2E avg\n(ms)", "e2e_avg_ms", False, 1.0, "{:.0f}"),
        ("E2E p95\n(ms)", "e2e_p95_ms", False, 1.0, "{:.0f}"),
        ("TPOT avg\n(ms)", "tpot_avg_ms", False, 1.0, "{:.1f}"),
        ("TPOT p95\n(ms)", "tpot_p95_ms", False, 1.0, "{:.1f}"),
    ]
    serving_metric_specs = [
        ("Dispatch\nwait (ms)", "dispatch_wait_ms", False, 1.0, "{:.1f}"),
        ("LoRA I/O\n(ms)", "lora_io_ms", False, 1.0, "{:.2f}"),
        ("Cost/req\nmUSD", "cost_per_req_usd", False, 1000.0, "{:.3f}"),
        ("Throughput\n(tok/s)", "tok_s", True, 1.0, "{:.1f}"),
        ("CE\n$1/(\\bar{L}\\bar{C})$", "ce", True, 1.0, "{:.1f}"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.55), constrained_layout=False)
    _draw_ablation_metric_matrix(axes[0], rows, reference, latency_metric_specs)
    _draw_ablation_metric_matrix(axes[1], rows, reference, serving_metric_specs)
    axes[1].set_xlabel("cell: value; Δ vs NVMe", fontsize=7.4, labelpad=3.0)
    fig.subplots_adjust(left=0.185, right=0.995, top=0.87, bottom=0.135, hspace=0.46)

    pdf = out_dir / "fig6_ablation.pdf"
    csv_path = out_dir / "fig6_ablation_data.csv"
    manifest = out_dir / "fig6_ablation_manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf)
    plt.close(fig)
    _write_csv(csv_path, rows)
    _write_manifest(
        manifest,
        "fig6_ablation",
        round_dir,
        pdf,
        csv_path,
        [s.source for s in scenarios],
        extra={
            "model": "Llama-2-7B",
            "scenario_labels": ["NVMe", "NoCoord", "Full"],
            "reference": "NVMe",
            "layout": "single-column transposed matrix; rows are implemented variants and columns are metrics",
            "fields_used": [
                "avg_overall_ttft_ms",
                "p95_overall_ttft_ms",
                "avg_overall_e2e_ms",
                "p95_overall_e2e_ms",
                "avg_tpot_ms",
                "request_level_tpot_ms_p95",
                "throughput_tok_per_s",
                "avg_dispatch_wait_ms",
                "avg_lora_io_ms",
                "monetary_cost_per_request_usd",
                "monetary_ce",
            ],
            "cell_semantics": "Each cell reports absolute value plus signed change relative to NVMe; positive change means better.",
        },
    )


def _main_csv_rows(systems: Sequence[MainSystemData]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for system in systems:
        row: Dict[str, Any] = {"system_key": system.key, "system": system.label, "source": str(system.source)}
        row.update(system.metrics)
        rows.append(row)
    return rows


def _best_baseline(systems: Sequence[MainSystemData], metric: str, *, higher: bool = False) -> MainSystemData:
    baselines = [system for system in systems if system.key != "faaslora"]
    if not baselines:
        raise SystemExit("cannot compute baseline normalization without baselines")
    fn = max if higher else min
    return fn(baselines, key=lambda item: item.metrics[metric])


def plot_fig1(round_dir: Path, out_dir: Path) -> None:
    systems = _main_round_data(round_dir)
    rows = _main_csv_rows(systems)
    for system in systems:
        rows.append(
            {
                "panel": "cost_ce_opportunity",
                "system_key": system.key,
                "system": system.label,
                "benchmark": "Llama-2-7B / 4000 requests / 500 adapters / Zipf 1.0 / hot set 48 / rotation 500 / time scale 8",
                "runtime_class": "serverless-style" if system.key in {"faaslora", "serverlessllm"} else "serverful",
                "cost_req_musd": system.metrics["cost_req_usd"] * 1000.0,
                "ce": system.metrics["ce"],
                "e2e_avg_ms": system.metrics["e2e_avg_ms"],
            }
        )

    fig, ax = plt.subplots(figsize=(3.45, 1.55), constrained_layout=True)
    label_offsets = {
        "faaslora": (5.0, 6.0, "left"),
        "sglang": (-5.0, 6.0, "right"),
        "vllm": (-5.0, -1.0, "right"),
        "slora": (-5.0, -7.0, "right"),
        "serverlessllm": (5.0, 7.0, "left"),
    }
    for system in systems:
        x = system.metrics["cost_req_usd"] * 1000.0
        y = system.metrics["ce"]
        marker = "D" if system.key in {"faaslora", "serverlessllm"} else "o"
        size = 46 if system.key == "faaslora" else 34
        ax.scatter(
            x,
            y,
            s=size,
            marker=marker,
            color=SYSTEM_COLORS[system.key],
            edgecolor="#333333",
            linewidth=0.45,
            zorder=3,
        )
        xoff, yoff, ha = label_offsets[system.key]
        ax.annotate(
            system.label,
            xy=(x, y),
            xytext=(xoff, yoff),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=7.4,
            bbox={"boxstyle": "round,pad=0.06", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )

    ax.set_xlim(2.35, 3.92)
    ax.set_ylim(0, 136)
    ax.set_xlabel("Cost/req (mUSD)")
    ax.set_ylabel("CE")
    ax.grid(axis="both", color="#D9D9D9", linewidth=0.55, alpha=0.80)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", length=2.4, width=0.6)
    ax.annotate(
        "",
        xy=(1.02, 0.0),
        xytext=(0.0, 0.0),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "linewidth": 0.75, "color": "#333333"},
        annotation_clip=False,
    )
    ax.annotate(
        "",
        xy=(0.0, 1.04),
        xytext=(0.0, 0.0),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "linewidth": 0.75, "color": "#333333"},
        annotation_clip=False,
    )
    ax.text(0.03, 0.95, "higher\nbetter", transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color="#4A4A4A")

    pdf = out_dir / "fig1_intro_teaser.pdf"
    csv_path = out_dir / "fig1_intro_teaser_data.csv"
    manifest = out_dir / "fig1_intro_teaser_manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    _write_csv(csv_path, rows)
    _write_manifest(
        manifest,
        "fig1_intro_teaser",
        round_dir,
        pdf,
        csv_path,
        [s.source for s in systems],
        extra={
            "baseline_or_own_system": "five_system_main_round",
            "model": "Llama-2-7B",
            "request_count": 4000,
            "adapter_pool_size": 500,
            "trace_id": str((_load_json(round_dir / "MANIFEST.json")).get("run_tag") or ""),
            "fields_used": ["monetary_cost_per_request_usd", "avg_overall_e2e_ms", "monetary_ce"],
            "benchmark": "Representative Llama-2-7B main workload: 4000 requests, 500 adapters, Zipf=1.0, hot set=48, rotation=500, time scale=8.",
            "role": "Measured serverless-style cost/CE opportunity from the completed five-system main round.",
        },
    )


def _format_metric(value: float, metric: str) -> str:
    if metric == "cost_req_usd":
        return f"{value:.6f}"
    if metric == "ce":
        return f"{value:.1f}"
    if metric == "tok_s":
        return f"{value:.1f}"
    if metric in {"tpot_ms", "tpot_avg_ms", "tpot_p95_ms"}:
        return f"{value:.1f}"
    return f"{value:.0f}"


def _latex_best(value: float, best: float, text: str) -> str:
    if math.isclose(value, best, rel_tol=1e-9, abs_tol=1e-9):
        return f"\\textbf{{{text}}}"
    return text


def plot_table1(round_dir: Path, out_dir: Path) -> None:
    systems = _main_round_data(round_dir)
    rows = _main_csv_rows(systems)
    metric_specs = [
        ("ttft_avg_ms", "TTFT Avg", False),
        ("ttft_p95_ms", "TTFT p95", False),
        ("e2e_avg_ms", "E2E Avg", False),
        ("e2e_p95_ms", "E2E p95", False),
        ("tpot_avg_ms", "TPOT Avg", False),
        ("tpot_p95_ms", "TPOT p95", False),
        ("tok_s", "Tok/s", True),
        ("cost_req_usd", "Cost/req", False),
        ("ce", "CE", True),
    ]
    best_values: Dict[str, float] = {}
    for metric, _, higher in metric_specs:
        values = [system.metrics[metric] for system in systems]
        best_values[metric] = max(values) if higher else min(values)

    lines = [
        "% Auto-generated by scripts/plot_paper_figures.py. Verify caption wording before final submission.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{End-to-end performance under the representative Llama-2 7B main workload. Lower is better for latency and cost; higher is better for Tok/s and CE. TPOT is reported as both average and p95 over observed per-request decode samples.}",
        "\\label{tab:end_to_end}",
        "\\begin{tabular}{lrrrrrrrrr}",
        "\\hline",
        "System & TTFT Avg & TTFT p95 & E2E Avg & E2E p95 & TPOT Avg & TPOT p95 & Tok/s & Cost/req & CE \\\\",
        "\\hline",
    ]
    for system in systems:
        cells = [system.label]
        for metric, _, _ in metric_specs:
            text = _format_metric(system.metrics[metric], metric)
            cells.append(_latex_best(system.metrics[metric], best_values[metric], text))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    tex = out_dir / "table1_end_to_end.tex"
    csv_path = out_dir / "table1_end_to_end_data.csv"
    manifest = out_dir / "table1_end_to_end_manifest.json"
    tex.write_text("\n".join(lines), encoding="utf-8")
    _write_csv(csv_path, rows)
    _write_manifest(manifest, "table1_end_to_end", round_dir, tex, csv_path, [s.source for s in systems], output_key="tex")


def plot_fig5(round_dir: Path, out_dir: Path) -> None:
    systems = _main_round_data(round_dir)
    metric_specs = [
        ("TTFT\navg", "ttft_avg_ms", False),
        ("TTFT\np95", "ttft_p95_ms", False),
        ("E2E\navg", "e2e_avg_ms", False),
        ("E2E\np95", "e2e_p95_ms", False),
        ("TPOT\navg", "tpot_avg_ms", False),
        ("TPOT\np95", "tpot_p95_ms", False),
        ("Cost/\nreq", "cost_req_usd", False),
    ]
    rows: List[Dict[str, Any]] = []
    for system in systems:
        rows.append(
            {
                "panel": "ce_ranking",
                "system_key": system.key,
                "system": system.label,
                "ce": system.metrics["ce"],
                "e2e_avg_ms": system.metrics["e2e_avg_ms"],
                "cost_req_usd": system.metrics["cost_req_usd"],
            }
        )
    for label, metric, higher in metric_specs:
        best = _best_baseline(systems, metric, higher=higher)
        for system in systems:
            normalized = system.metrics[metric] / best.metrics[metric]
            rows.append(
                {
                    "panel": "latency_cost_matrix",
                    "metric": label,
                    "metric_key": metric,
                    "system_key": system.key,
                    "system": system.label,
                    "value": system.metrics[metric],
                    "best_baseline": best.label,
                    "best_baseline_value": best.metrics[metric],
                    "normalized": normalized,
                    "higher_is_better": higher,
                }
            )

    labels = [spec[0] for spec in metric_specs]
    matrix = np.asarray(
        [
            [
                next(row["normalized"] for row in rows if row.get("metric") == label and row["system_key"] == system.key)
                for label in labels
            ]
            for system in systems
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 3.18),
        gridspec_kw={"width_ratios": [0.82, 1.58]},
        constrained_layout=True,
    )

    ce_order = sorted(systems, key=lambda system: system.metrics["ce"], reverse=True)
    cy = np.arange(len(ce_order), dtype=float)
    ce_vals = [system.metrics["ce"] for system in ce_order]
    sglang_ce = next(system.metrics["ce"] for system in systems if system.key == "sglang")
    bars = axes[0].barh(
        cy,
        ce_vals,
        height=0.46,
        color=[SYSTEM_COLORS[system.key] for system in ce_order],
        edgecolor="#333333",
        linewidth=0.35,
        alpha=0.88,
    )
    for system, bar, value in zip(ce_order, bars, ce_vals):
        label = f"{value:.1f}"
        if system.key == "faaslora":
            label = f"{value:.1f} (+{(value / sglang_ce - 1.0) * 100:.0f}% vs SGLang)"
        axes[0].annotate(
            label,
            xy=(value, bar.get_y() + bar.get_height() / 2),
            xytext=(6.0, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.0,
            bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.80},
        )
    axes[0].set_yticks(cy, [AXIS_SYSTEM_LABELS[system.key].replace("\n", " ") for system in ce_order])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(ce_vals) * 1.46)
    _xlabel_with_panel(axes[0], "CE (higher is better)", "(a) Cost-effectiveness ranking")
    _style_xgrid_axes(axes[0])

    ax = axes[1]
    # Keep exact values in-cell. The light background is capped only to prevent
    # ServerlessLLM's large ratios from washing out the rest of the matrix.
    display = np.minimum(matrix, 4.0)
    ax.imshow(display, cmap="Blues", vmin=0.0, vmax=4.0, aspect="auto", alpha=0.55)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=0)
    ax.set_yticks(np.arange(len(systems)), [AXIS_SYSTEM_LABELS[system.key] for system in systems])
    _xlabel_with_panel(ax, "Normalized to best baseline (lower is better)", "(b) Latency and cost factors")
    for i, system in enumerate(systems):
        for j, value in enumerate(matrix[i]):
            text = f"{value:.2f}x" if value < 10 else f"{value:.0f}x"
            weight = "bold" if system.key == "faaslora" else "normal"
            ax.text(j, i, text, ha="center", va="center", fontsize=9.0, fontweight=weight, color="#1F1F1F")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(systems), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.9)

    pdf = out_dir / "fig5_main_normalized.pdf"
    csv_path = out_dir / "fig5_main_normalized_data.csv"
    manifest = out_dir / "fig5_main_normalized_manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    _write_csv(csv_path, rows)
    _write_manifest(manifest, "fig5_main_normalized", round_dir, pdf, csv_path, [s.source for s in systems])


def plot_fig7(round_dir: Path, out_dir: Path) -> None:
    systems = _main_round_data(round_dir)
    rows = _main_csv_rows(systems)
    labels = [AXIS_SYSTEM_LABELS[system.key] for system in systems]
    y = np.arange(len(systems))
    label_fontsize = 7.1
    tick_fontsize = 6.8
    panel_fontsize = 7.0
    legend_fontsize = 6.5
    components = [
        ("Startup", "cost_startup_usd", "#A9C4E8"),
        ("Active", "cost_active_usd", "#A7D3A8"),
        ("Idle-ready", "cost_idle_ready_usd", "#F4C58D"),
        ("Invocation", "cost_invocation_usd", "#D8B6D9"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(3.58, 2.10), constrained_layout=False)
    bottom = np.zeros(len(systems))
    legend_handles = []
    legend_labels = []
    for name, key, color in components:
        vals = np.asarray([system.metrics[key] * 1000.0 for system in systems])
        bars = axes[0].barh(
            y,
            vals,
            left=bottom,
            height=0.58,
            label=name,
            color=color,
            edgecolor="#555555",
            linewidth=0.25,
        )
        if np.any(vals > 0):
            legend_handles.append(bars[0])
            legend_labels.append(name)
        bottom += vals
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_ylim(len(systems) - 0.31, -0.295)
    _xlabel_with_panel(axes[0], "Cost/req (mUSD)", "(a) Cost")
    _style_axes(axes[0])
    axes[0].xaxis.label.set_size(panel_fontsize)
    axes[0].tick_params(axis="both", labelsize=tick_fontsize)

    gpu_components = [
        ("Startup", "infra_startup_gpu_seconds", "#A9C4E8"),
        ("Active serving", "infra_active_gpu_seconds", "#A7D3A8"),
        ("Idle-ready", "infra_idle_ready_gpu_seconds", "#F4C58D"),
    ]
    bottom = np.zeros(len(systems))
    for name, key, color in gpu_components:
        vals = np.asarray([system.metrics[key] / system.metrics["completed"] for system in systems])
        axes[1].barh(
            y,
            vals,
            left=bottom,
            height=0.58,
            label=name,
            color=color,
            edgecolor="#555555",
            linewidth=0.25,
        )
        bottom += vals
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_ylim(len(systems) - 0.31, -0.295)
    _xlabel_with_panel(axes[1], "GPU-s/req", "(b) GPU time")
    _style_axes(axes[1])
    axes[1].xaxis.label.set_size(panel_fontsize)
    axes[1].tick_params(axis="both", labelsize=tick_fontsize)
    fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        fontsize=legend_fontsize,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.882),
        columnspacing=0.68,
        handlelength=1.1,
    )
    for ax in axes:
        ax.xaxis.labelpad = 4.0
        ax.grid(axis="x", color="#E7E7E7", linewidth=0.45)
    fig.subplots_adjust(left=0.28, right=0.99, top=0.825, bottom=0.215, wspace=0.26)

    pdf = out_dir / "fig7_lifecycle_cost.pdf"
    csv_path = out_dir / "fig7_lifecycle_cost_data.csv"
    manifest = out_dir / "fig7_lifecycle_cost_manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    _write_csv(csv_path, rows)
    _write_manifest(manifest, "fig7_lifecycle_cost", round_dir, pdf, csv_path, [s.source for s in systems])


PLOTTERS: Dict[str, Callable[[Path, Path], None]] = {
    "fig1_intro": plot_fig1,
    "fig1": plot_fig1,
    "table1_main": plot_table1,
    "table1": plot_table1,
    "fig2_mismatch": plot_fig2,
    "fig3_tier": plot_fig3,
    "fig4_coordination": plot_fig4,
    "fig5_normalized": plot_fig5,
    "fig5": plot_fig5,
    "fig6_ablation": plot_fig6,
    "fig6": plot_fig6,
    "fig7_cost": plot_fig7,
    "fig7": plot_fig7,
}

MAIN_FIGURES = ("fig1_intro", "table1_main", "fig5_normalized", "fig7_cost")
MOTIVATION_FIGURES = ("fig2_mismatch", "fig3_tier")
ABLATION_FIGURES = ("fig4_coordination", "fig6_ablation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PrimeLoRA paper figures from result JSONs.")
    parser.add_argument("--round-dir", required=True, type=Path)
    parser.add_argument(
        "--figure",
        default="all",
        help=f"Figure name, main_all, motivation_all, ablation_all, or all. Choices: {', '.join(PLOTTERS)}",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("figs/paper/ablation"))
    args = parser.parse_args()

    round_dir = args.round_dir.resolve()
    out_dir = args.out_dir.resolve()
    _require_file(round_dir / "MANIFEST.json")

    if args.figure == "all":
        selected: List[str] = []
        manifest = _load_json(round_dir / "MANIFEST.json")
        try:
            _find_main_compare(round_dir, manifest)
            selected.extend(MAIN_FIGURES)
        except SystemExit:
            pass
        if manifest.get("scenarios"):
            selected.extend(ABLATION_FIGURES)
        if not selected:
            raise SystemExit(f"could not infer figure group for round dir: {round_dir}")
    elif args.figure == "main_all":
        selected = list(MAIN_FIGURES)
    elif args.figure == "motivation_all":
        selected = list(MOTIVATION_FIGURES)
    elif args.figure == "ablation_all":
        selected = list(ABLATION_FIGURES)
    else:
        if args.figure not in PLOTTERS:
            raise SystemExit(f"unknown figure {args.figure!r}; choose one of {sorted(PLOTTERS)}, main_all, motivation_all, ablation_all, or all")
        selected = [args.figure]

    for figure in selected:
        PLOTTERS[figure](round_dir, out_dir)
        print(f"generated {figure} -> {out_dir}")


if __name__ == "__main__":
    main()
