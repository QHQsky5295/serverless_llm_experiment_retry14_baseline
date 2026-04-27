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
    "faaslora_nvme": "NVMe-pre",
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
    "faaslora": "Prime\nLoRA",
    "sglang": "SGLang",
    "vllm": "vLLM",
    "slora": "S-LoRA",
    "serverlessllm": "Serverless\nLLM",
}
METRIC_COLORS = {
    "ttft": "#7FA7D9",
    "e2e": "#8FD0C7",
    "cost": "#F2B36D",
    "ce": "#78B87A",
}
DOUBLE_COL_FIGSIZE = (7.16, 3.15)
DOUBLE_COL_TALL_FIGSIZE = (7.16, 5.9)
PANEL_TITLE_FONTSIZE = 10.4
TICK_FONTSIZE = 9.2
LEGEND_FONTSIZE = 9.0
ANNOTATION_FONTSIZE = 9.2


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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
) -> None:
    payload = {
        "figure": figure,
        "round_dir": str(round_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        output_key: str(output_path),
        "csv": str(csv_path),
        "sources": [str(p) for p in sources],
    }
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
    offsets = np.linspace(-0.08, 0.08, len(series)) if len(series) > 1 else np.array([0.0])
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
            align = "left" if val >= 0 else "right"
            dx = text_pad if val >= 0 else -text_pad
            ax.text(val + dx, yi, f"{val:+.1f}{label_suffix}", ha=align, va="center", fontsize=ANNOTATION_FONTSIZE)

    ax.axvline(0, color="#4D4D4D", linewidth=0.8, linestyle="--")
    ax.set_yticks(y, metric_labels, fontsize=TICK_FONTSIZE)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    _style_xgrid_axes(ax)


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
    return round_dir / "raw" / "replay" / f"{run_tag}_{suffixes[key]}"


def _main_row_from_summary(path: Path, key: str) -> Dict[str, Any]:
    payload = _load_json(path)
    if payload.get("metric_schema_version") != "e2e_v3":
        raise SystemExit(f"{path}: metric_schema_version must be e2e_v3")
    if key == "faaslora":
        summary = (payload.get("scenario_summaries") or {}).get("faaslora_full")
        if not summary:
            raise SystemExit(f"{path}: missing scenario_summaries.faaslora_full")
        return {
            "completed": summary.get("completed_requests"),
            "total": summary.get("total_requests"),
            "TTFT_avg_ms": summary.get("avg_overall_ttft_ms"),
            "TTFT_p95_ms": summary.get("p95_overall_ttft_ms"),
            "E2E_avg_ms": summary.get("avg_overall_e2e_ms"),
            "E2E_p95_ms": summary.get("p95_overall_e2e_ms"),
            "TPOT_ms": summary.get("avg_tpot_ms"),
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
    return {
        "completed": row.get("completed"),
        "total": row.get("total"),
        "TTFT_avg_ms": row.get("TTFT_e2e_avg_ms"),
        "TTFT_p95_ms": row.get("TTFT_e2e_P95_ms"),
        "E2E_avg_ms": row.get("E2E_avg_ms"),
        "E2E_p95_ms": row.get("E2E_P95_ms"),
        "TPOT_ms": row.get("TPOT_avg_ms"),
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
            "tpot_ms": _as_float(raw.get("TPOT_ms"), f"{key}.TPOT_ms"),
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
                "cost_startup_usd": startup,
                "cost_active_usd": active,
                "cost_idle_ready_usd": idle,
                "cost_invocation_usd": invocation,
            }
        )
        systems.append(MainSystemData(key, SYSTEM_LABELS[key], source, metrics))
    return systems


def plot_fig2(round_dir: Path, out_dir: Path) -> None:
    data = _round_data(round_dir)
    full = _require_scenarios(data, ["faaslora_full"])[0]
    rows: List[Dict[str, Any]] = []

    gpu_ready = [r for r in full.requests if str(r.get("cache_tier", "")).lower() == "gpu"]
    scaleup = [r for r in full.requests if bool(r.get("scaleup_affected"))]
    first_service = [r for r in full.requests if bool(r.get("scaleup_first_service"))]
    categories = [
        ("GPU-ready", gpu_ready),
        ("Scale-up\naffected", scaleup),
        ("First\nservice", first_service),
    ]
    for name, reqs in categories:
        ttfts = [_request_float(r, "overall_ttft_ms", full.name) for r in reqs]
        rows.append(
            {
                "panel": "request_category_ttft",
                "category": name,
                "count": len(reqs),
                "ttft_avg_ms": _mean(ttfts),
                "ttft_p95_ms": _percentile(ttfts, 95),
            }
        )

    rate_rows = [
        ("Runtime GPU-hit", _summary_float(full, "scaleup_runtime_gpu_hit_rate")),
        ("First-service GPU-hit", _summary_float(full, "scaleup_first_service_gpu_hit_rate")),
        ("First-service planned match", _summary_float(full, "scaleup_first_service_planned_match_rate")),
    ]
    for name, rate in rate_rows:
        rows.append({"panel": "scaleup_readiness_rate", "metric": name, "rate": rate})

    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL_FIGSIZE, constrained_layout=True)
    category_rows = [row for row in rows if row.get("panel") == "request_category_ttft"]
    cy = np.arange(len(category_rows), dtype=float)
    avg_vals = [row["ttft_avg_ms"] for row in category_rows]
    p95_vals = [row["ttft_p95_ms"] for row in category_rows]
    for yi, avg, p95 in zip(cy, avg_vals, p95_vals):
        axes[0].hlines(yi, avg, p95, color="#D2D2D2", linewidth=2.2, zorder=1)
    axes[0].scatter(avg_vals, cy - 0.035, s=50, color=COLORS["avg"], edgecolor="#333333", linewidth=0.4, label="Avg", zorder=3)
    axes[0].scatter(p95_vals, cy + 0.035, s=50, color=COLORS["p95"], edgecolor="#333333", linewidth=0.4, label="p95", zorder=3)
    x_pad = max(p95_vals) * 0.018
    for yi, avg, p95 in zip(cy, avg_vals, p95_vals):
        axes[0].text(avg + x_pad, yi - 0.035, f"{avg:.0f}", ha="left", va="center", fontsize=ANNOTATION_FONTSIZE)
        axes[0].text(p95 + x_pad, yi + 0.035, f"{p95:.0f}", ha="left", va="center", fontsize=ANNOTATION_FONTSIZE)
    axes[0].set_yticks(
        cy,
        [f"{name.replace(chr(10), ' ')}\n(n={len(reqs)})" for name, reqs in categories],
        fontsize=TICK_FONTSIZE,
    )
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(p95_vals) * 1.18)
    axes[0].set_xlabel("TTFT (ms)")
    axes[0].set_title("(a) TTFT by request class", fontsize=PANEL_TITLE_FONTSIZE)
    axes[0].legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper right", ncols=2)
    _style_xgrid_axes(axes[0])

    rate_labels = [name for name, _ in rate_rows]
    rates = [rate * 100 for _, rate in rate_rows]
    ry = np.arange(len(rate_rows))
    axes[1].hlines(ry, 0, rates, color="#A7D3A8", linewidth=4.0, alpha=0.65)
    axes[1].scatter(rates, ry, s=52, color="#6E9F76", edgecolor="#333333", linewidth=0.45, zorder=3)
    for rate, yi in zip(rates, ry):
        axes[1].text(rate + 1.5, yi, f"{rate:.0f}%", ha="left", va="center", fontsize=ANNOTATION_FONTSIZE)
    axes[1].set_yticks(ry, rate_labels, fontsize=TICK_FONTSIZE)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Rate (%)")
    axes[1].set_xlim(0, 105)
    axes[1].set_title("(b) Scale-up adapter readiness", fontsize=PANEL_TITLE_FONTSIZE)
    _style_xgrid_axes(axes[1])

    pdf = out_dir / "fig2_mismatch.pdf"
    csv_path = out_dir / "fig2_mismatch_data.csv"
    manifest = out_dir / "fig2_mismatch_manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    _write_csv(csv_path, rows)
    _write_manifest(manifest, "fig2_mismatch", round_dir, pdf, csv_path, [full.source])


def plot_fig3(round_dir: Path, out_dir: Path) -> None:
    data = _round_data(round_dir)
    full = _require_scenarios(data, ["faaslora_full"])[0]
    tier_order = ["gpu", "host", "nvme", "remote"]
    rows: List[Dict[str, Any]] = []
    present_tiers: List[str] = []
    for tier in tier_order:
        reqs = [r for r in full.requests if str(r.get("cache_tier", "")).lower() == tier]
        if not reqs:
            continue
        present_tiers.append(tier)
        ttft = [_request_float(r, "overall_ttft_ms", full.name) for r in reqs]
        lora = [_request_float(r, "lora_io_ms", full.name) for r in reqs]
        rows.append(
            {
                "tier": tier.upper(),
                "count": len(reqs),
                "ttft_avg_ms": _mean(ttft),
                "ttft_p95_ms": _percentile(ttft, 95),
                "lora_io_avg_ms": _mean(lora),
                "lora_io_p95_ms": _percentile(lora, 95),
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL_FIGSIZE, constrained_layout=True)
    tier_colors = {"gpu": "#78B87A", "host": "#F2B36D", "nvme": "#7FA7D9", "remote": "#E88989"}
    tier_counts = {r["tier"].lower(): r["count"] for r in rows}
    all_ttft: List[float] = []
    all_lora: List[float] = []
    for tier in present_tiers:
        reqs = [r for r in full.requests if str(r.get("cache_tier", "")).lower() == tier]
        ttft = [_request_float(r, "overall_ttft_ms", full.name) for r in reqs]
        lora = [_request_float(r, "lora_io_ms", full.name) for r in reqs]
        all_ttft.extend(ttft)
        all_lora.extend(lora)
        label = f"{tier.upper()} (n={tier_counts[tier]})"
        _plot_ecdf(axes[0], ttft, label=label, color=tier_colors.get(tier, "#777777"))

    axes[0].set_xlim(0, max(_percentile(all_ttft, 99.5), max(r["ttft_p95_ms"] for r in rows) * 1.05))
    axes[0].set_ylim(0, 1.02)
    axes[0].set_xlabel("TTFT (ms)")
    axes[0].set_ylabel("CDF")
    axes[0].set_title("(a) TTFT distribution by adapter tier", fontsize=PANEL_TITLE_FONTSIZE)
    axes[0].legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="lower right")
    _style_xgrid_axes(axes[0])

    y = np.arange(len(rows), dtype=float)
    io_avg = [r["lora_io_avg_ms"] for r in rows]
    io_p95 = [r["lora_io_p95_ms"] for r in rows]
    for yi, avg, p95 in zip(y, io_avg, io_p95):
        axes[1].hlines(yi, min(avg, p95), max(avg, p95), color="#D2D2D2", linewidth=2.2, zorder=1)
    axes[1].scatter(io_avg, y - 0.035, s=50, color="#A9B8C8", edgecolor="#333333", linewidth=0.4, label="Avg", zorder=3)
    axes[1].scatter(io_p95, y + 0.035, s=50, color="#E2A07B", edgecolor="#333333", linewidth=0.4, label="p95", zorder=3)
    pad = max(max(io_p95), 1.0) * 0.02
    for yi, avg, p95 in zip(y, io_avg, io_p95):
        if abs(avg - p95) < 0.05:
            axes[1].text(max(avg, p95) + pad, yi, f"{avg:.1f}", ha="left", va="center", fontsize=ANNOTATION_FONTSIZE)
        else:
            axes[1].text(avg + pad, yi - 0.035, f"{avg:.1f}", ha="left", va="center", fontsize=ANNOTATION_FONTSIZE)
            axes[1].text(p95 + pad, yi + 0.035, f"{p95:.1f}", ha="left", va="center", fontsize=ANNOTATION_FONTSIZE)
    axes[1].set_yticks(y, [f"{r['tier']}\n(n={r['count']})" for r in rows], fontsize=TICK_FONTSIZE)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(io_p95) * 1.22 + 1.0)
    axes[1].set_xlabel("LoRA I/O (ms)")
    axes[1].set_title("(b) Adapter I/O by tier", fontsize=PANEL_TITLE_FONTSIZE)
    axes[1].legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper right", ncols=2)
    _style_xgrid_axes(axes[1])

    pdf = out_dir / "fig3_tier.pdf"
    csv_path = out_dir / "fig3_tier_data.csv"
    manifest = out_dir / "fig3_tier_manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    _write_csv(csv_path, rows)
    _write_manifest(manifest, "fig3_tier", round_dir, pdf, csv_path, [full.source])


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
                "tpot_ms": _summary_float(scenario, "avg_tpot_ms"),
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
        ("TPOT", "tpot_ms", False),
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
    _plot_change_panel(
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
    _plot_change_panel(
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
                "tpot_ms": _summary_float(scenario, "avg_tpot_ms"),
                "gpu_hit_rate_pct": _summary_float(scenario, "gpu_hit_rate") * 100,
                "lora_io_ms": _summary_float(scenario, "avg_lora_io_ms"),
                "dispatch_wait_ms": _summary_float(scenario, "avg_dispatch_admission_wait_ms"),
                "cost_per_req_usd": _summary_float(scenario, "monetary_cost_per_request_usd"),
                "ce": _summary_float(scenario, "monetary_ce"),
            }
        )

    fig, axes = plt.subplots(2, 2, figsize=DOUBLE_COL_TALL_FIGSIZE, constrained_layout=True)
    axes = axes.ravel()

    reference = rows[0]
    compared = rows[1:]

    def panel_values(metric_specs: Sequence[tuple[str, str, bool]]) -> List[tuple[str, List[float], str]]:
        out: List[tuple[str, List[float], str]] = []
        for row in compared:
            out.append(
                (
                    row["label"],
                    [_improvement_pct(reference[key], row[key], higher_is_better=higher) for _, key, higher in metric_specs],
                    COLORS[row["scenario"]],
                )
            )
        return out

    for row in compared:
        row["reference"] = reference["label"]
        row["ttft_avg_improvement_pct"] = _improvement_pct(reference["ttft_avg_ms"], row["ttft_avg_ms"], higher_is_better=False)
        row["ttft_p95_improvement_pct"] = _improvement_pct(reference["ttft_p95_ms"], row["ttft_p95_ms"], higher_is_better=False)
        row["e2e_avg_improvement_pct"] = _improvement_pct(reference["e2e_avg_ms"], row["e2e_avg_ms"], higher_is_better=False)
        row["e2e_p95_improvement_pct"] = _improvement_pct(reference["e2e_p95_ms"], row["e2e_p95_ms"], higher_is_better=False)
        row["tpot_improvement_pct"] = _improvement_pct(reference["tpot_ms"], row["tpot_ms"], higher_is_better=False) if "tpot_ms" in row else None
        row["lora_io_improvement_pct"] = _improvement_pct(reference["lora_io_ms"], row["lora_io_ms"], higher_is_better=False)
        row["dispatch_wait_improvement_pct"] = _improvement_pct(reference["dispatch_wait_ms"], row["dispatch_wait_ms"], higher_is_better=False)
        row["cost_improvement_pct"] = _improvement_pct(reference["cost_per_req_usd"], row["cost_per_req_usd"], higher_is_better=False)
        row["ce_improvement_pct"] = _improvement_pct(reference["ce"], row["ce"], higher_is_better=True)

    _plot_change_panel(
        axes[0],
        ["TTFT avg", "TTFT p95"],
        panel_values([("TTFT avg", "ttft_avg_ms", False), ("TTFT p95", "ttft_p95_ms", False)]),
        title="(a) First-token improvement",
        xlabel=f"Improvement vs {reference['label']} (%)",
        min_span=8.0,
    )
    _plot_change_panel(
        axes[1],
        ["E2E avg", "E2E p95"],
        panel_values([("E2E avg", "e2e_avg_ms", False), ("E2E p95", "e2e_p95_ms", False)]),
        title="(b) End-to-end impact",
        xlabel=f"Improvement vs {reference['label']} (%)",
        min_span=2.0,
    )
    _plot_change_panel(
        axes[2],
        ["Dispatch wait", "LoRA I/O"],
        panel_values([("Dispatch wait", "dispatch_wait_ms", False), ("LoRA I/O", "lora_io_ms", False)]),
        title="(c) Admission and I/O overhead",
        xlabel=f"Reduction vs {reference['label']} (%)",
        min_span=12.0,
    )
    _plot_change_panel(
        axes[3],
        ["Cost/req", "CE"],
        panel_values([("Cost/req", "cost_per_req_usd", False), ("CE", "ce", True)]),
        title="(d) Relative cost-efficiency",
        xlabel=f"Improvement vs {reference['label']} (%)",
        min_span=1.5,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=LEGEND_FONTSIZE, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.06))

    pdf = out_dir / "fig6_ablation.pdf"
    csv_path = out_dir / "fig6_ablation_data.csv"
    manifest = out_dir / "fig6_ablation_manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    _write_csv(csv_path, rows)
    _write_manifest(manifest, "fig6_ablation", round_dir, pdf, csv_path, [s.source for s in scenarios])


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
    prime = next(system for system in systems if system.key == "faaslora")
    latency = [
        ("TTFT p95", "ttft_p95_ms"),
        ("E2E p95", "e2e_p95_ms"),
    ]
    efficiency = [
        ("Cost/req", "cost_req_usd", False, "cost"),
        ("CE", "ce", True, "ce"),
    ]
    for label, metric in latency:
        best = _best_baseline(systems, metric, higher=False)
        rows.append(
            {
                "panel": "headline_latency",
                "metric": label,
                "prime_value": prime.metrics[metric],
                "best_baseline": best.label,
                "best_baseline_value": best.metrics[metric],
                "unit": "ms",
            }
        )
    for label, metric, higher, _ in efficiency:
        best = _best_baseline(systems, metric, higher=higher)
        rows.append(
            {
                "panel": "headline_efficiency_ratio",
                "metric": label,
                "prime_value": prime.metrics[metric],
                "best_baseline": best.label,
                "best_baseline_value": best.metrics[metric],
                "normalized_prime": prime.metrics[metric] / best.metrics[metric],
                "higher_is_better": higher,
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL_FIGSIZE, constrained_layout=True)
    y = np.arange(len(latency), dtype=float)
    prime_vals = [prime.metrics[metric] for _, metric in latency]
    best_vals = [_best_baseline(systems, metric, higher=False).metrics[metric] for _, metric in latency]
    best_names = [_best_baseline(systems, metric, higher=False).label for _, metric in latency]
    for yi, prime_val, best_val in zip(y, prime_vals, best_vals):
        axes[0].hlines(yi, min(prime_val, best_val), max(prime_val, best_val), color="#CFCFCF", linewidth=2.0, zorder=1)
    axes[0].scatter(best_vals, y - 0.035, s=56, label="Best baseline", color="#A9C4E8", edgecolor="#333333", linewidth=0.45, zorder=3)
    axes[0].scatter(prime_vals, y + 0.035, s=56, label="PrimeLoRA", color=SYSTEM_COLORS["faaslora"], edgecolor="#333333", linewidth=0.45, zorder=3)
    x_pad = max(prime_vals + best_vals) * 0.06
    for yi, val, name in zip(y, best_vals, best_names):
        axes[0].text(val + x_pad * 0.25, yi - 0.035, f"{name} {val:.0f}", ha="left", va="center", fontsize=ANNOTATION_FONTSIZE)
    for yi, val in zip(y, prime_vals):
        axes[0].text(val + x_pad * 0.25, yi + 0.035, f"PrimeLoRA {val:.0f}", ha="left", va="center", fontsize=ANNOTATION_FONTSIZE)
    axes[0].set_yticks(y, [label for label, _ in latency], fontsize=TICK_FONTSIZE)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(prime_vals + best_vals) * 1.18)
    axes[0].set_xlabel("p95 latency (ms)")
    axes[0].set_title("(a) Tail-latency headline")
    _style_xgrid_axes(axes[0])

    ratio_labels = []
    vals = []
    colors = []
    for label, metric, higher, color_key in efficiency:
        best = _best_baseline(systems, metric, higher=higher)
        vals.append(prime.metrics[metric] / best.metrics[metric])
        colors.append(METRIC_COLORS[color_key])
        ratio_labels.append(f"{label}\n{'higher' if higher else 'lower'}")
    ry = np.arange(len(efficiency), dtype=float)
    axes[1].axvline(1.0, color="#555555", linewidth=0.9, linestyle="--")
    for yi, value, color in zip(ry, vals, colors):
        axes[1].hlines(yi, 1.0, value, color=color, linewidth=2.0, alpha=0.82)
        axes[1].scatter(value, yi, s=58, color=color, edgecolor="#333333", linewidth=0.45, zorder=3)
        align = "left" if value >= 1.0 else "right"
        dx = 0.008 if value >= 1.0 else -0.008
        axes[1].text(value + dx, yi, f"{value:.2f}x", ha=align, va="center", fontsize=ANNOTATION_FONTSIZE)
    axes[1].set_yticks(ry, ratio_labels, fontsize=TICK_FONTSIZE)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.90, 1.11)
    axes[1].set_xlabel("PrimeLoRA / best baseline")
    axes[1].set_title("(b) Cost-efficiency headline")
    _style_xgrid_axes(axes[1])

    pdf = out_dir / "fig1_intro_teaser.pdf"
    csv_path = out_dir / "fig1_intro_teaser_data.csv"
    manifest = out_dir / "fig1_intro_teaser_manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    _write_csv(csv_path, rows)
    _write_manifest(manifest, "fig1_intro_teaser", round_dir, pdf, csv_path, [s.source for s in systems])


def _format_metric(value: float, metric: str) -> str:
    if metric == "cost_req_usd":
        return f"{value:.6f}"
    if metric == "ce":
        return f"{value:.1f}"
    if metric == "tok_s":
        return f"{value:.1f}"
    if metric in {"tpot_ms"}:
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
        ("tpot_ms", "TPOT", False),
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
        "\\caption{End-to-end performance under the main workload. Lower is better for latency and cost; higher is better for Tok/s and CE.}",
        "\\label{tab:end_to_end}",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\hline",
        "System & TTFT Avg & TTFT p95 & E2E Avg & E2E p95 & TPOT & Tok/s & Cost/req & CE \\\\",
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
        ("TTFT avg\nlower", "ttft_avg_ms", False),
        ("TTFT p95\nlower", "ttft_p95_ms", False),
        ("E2E avg\nlower", "e2e_avg_ms", False),
        ("E2E p95\nlower", "e2e_p95_ms", False),
        ("Cost/req\nlower", "cost_req_usd", False),
        ("CE\nhigher", "ce", True),
    ]
    rows: List[Dict[str, Any]] = []
    for label, metric, higher in metric_specs:
        best = _best_baseline(systems, metric, higher=higher)
        for system in systems:
            normalized = system.metrics[metric] / best.metrics[metric]
            rows.append(
                {
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
                next(row["normalized"] for row in rows if row["metric"] == label and row["system_key"] == system.key)
                for label in labels
            ]
            for system in systems
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7.16, 3.05), constrained_layout=True)
    # Keep exact values in-cell. The light background is capped only to prevent
    # ServerlessLLM's large ratios from washing out the rest of the matrix.
    display = np.minimum(matrix, 4.0)
    ax.imshow(display, cmap="Blues", vmin=0.0, vmax=4.0, aspect="auto", alpha=0.55)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=0)
    ax.set_yticks(np.arange(len(systems)), [AXIS_SYSTEM_LABELS[system.key] for system in systems])
    ax.set_title("Normalized main-workload comparison")
    ax.set_xlabel("Metric normalized to the best baseline")
    ax.set_ylabel("System")
    for i, system in enumerate(systems):
        for j, value in enumerate(matrix[i]):
            text = f"{value:.2f}x" if value < 10 else f"{value:.0f}x"
            weight = "bold" if system.key == "faaslora" else "normal"
            ax.text(j, i, text, ha="center", va="center", fontsize=10.1, fontweight=weight, color="#1F1F1F")
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
    x = np.arange(len(systems))
    components = [
        ("Startup", "cost_startup_usd", "#A9C4E8"),
        ("Active", "cost_active_usd", "#A7D3A8"),
        ("Idle-ready", "cost_idle_ready_usd", "#F4C58D"),
        ("Invocation", "cost_invocation_usd", "#D8B6D9"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), constrained_layout=True)
    bottom = np.zeros(len(systems))
    legend_handles = []
    legend_labels = []
    for name, key, color in components:
        vals = np.asarray([system.metrics[key] * 1000.0 for system in systems])
        bars = axes[0].bar(x, vals, bottom=bottom, width=0.58, label=name, color=color, edgecolor="#555555", linewidth=0.3)
        if np.any(vals > 0):
            legend_handles.append(bars[0])
            legend_labels.append(name)
        bottom += vals
    axes[0].set_xticks(x, labels, rotation=0)
    axes[0].set_ylabel("Cost/req (milli-USD)")
    axes[0].set_title("(a) Monetary lifecycle cost")
    _style_axes(axes[0])

    gpu_components = [
        ("Startup", "infra_startup_gpu_seconds", "#A9C4E8"),
        ("Active serving", "infra_active_gpu_seconds", "#A7D3A8"),
        ("Idle-ready", "infra_idle_ready_gpu_seconds", "#F4C58D"),
    ]
    bottom = np.zeros(len(systems))
    for name, key, color in gpu_components:
        vals = np.asarray([system.metrics[key] / system.metrics["completed"] for system in systems])
        axes[1].bar(x, vals, bottom=bottom, width=0.58, label=name, color=color, edgecolor="#555555", linewidth=0.3)
        bottom += vals
    axes[1].set_xticks(x, labels, rotation=0)
    axes[1].set_ylabel("GPU-seconds/req")
    axes[1].set_title("(b) Lifecycle GPU time")
    _style_axes(axes[1])
    fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        ncols=max(1, min(4, len(legend_labels))),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
    )

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
ABLATION_FIGURES = ("fig2_mismatch", "fig3_tier", "fig4_coordination", "fig6_ablation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PrimeLoRA paper figures from result JSONs.")
    parser.add_argument("--round-dir", required=True, type=Path)
    parser.add_argument("--figure", default="all", help=f"Figure name, main_all, ablation_all, or all. Choices: {', '.join(PLOTTERS)}")
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
        if (round_dir / "raw" / "faaslora").exists():
            selected.extend(ABLATION_FIGURES)
        if not selected:
            raise SystemExit(f"could not infer figure group for round dir: {round_dir}")
    elif args.figure == "main_all":
        selected = list(MAIN_FIGURES)
    elif args.figure == "ablation_all":
        selected = list(ABLATION_FIGURES)
    else:
        if args.figure not in PLOTTERS:
            raise SystemExit(f"unknown figure {args.figure!r}; choose one of {sorted(PLOTTERS)}, main_all, ablation_all, or all")
        selected = [args.figure]

    for figure in selected:
        PLOTTERS[figure](round_dir, out_dir)
        print(f"generated {figure} -> {out_dir}")


if __name__ == "__main__":
    main()
