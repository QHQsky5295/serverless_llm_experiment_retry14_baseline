#!/usr/bin/env python3
"""Build PrimeLoRA-SGLang backend portability artifacts.

This script intentionally does not modify or rerun the formal main-experiment
chain.  It builds an isolated sensitivity artifact by combining:

* measured SGLang service-path timings on the shared replay, and
* measured PrimeLoRA-vLLM control-path/lifecycle behavior on the same replay.

The resulting PrimeLoRA-SGLang row is therefore a request-matched backend
portability projection, not a replacement for the vLLM-backed PrimeLoRA main
prototype.
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


@dataclass(frozen=True)
class SourceSet:
    model_key: str
    model_label: str
    round_dir: Path
    sglang_summary: Path
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


SYSTEM_ORDER = ("sglang", "primelora_sglang", "vllm", "primelora_vllm")
SYSTEM_LABELS = {
    "sglang": "SGLang",
    "primelora_sglang": "PrimeLoRA-SGLang",
    "vllm": "vLLM",
    "primelora_vllm": "PrimeLoRA-vLLM",
}
SYSTEM_COLORS = {
    "sglang": "#6E9CCF",
    "primelora_sglang": "#4F9D69",
    "vllm": "#85C1B9",
    "primelora_vllm": "#D88C5A",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(payload: dict[str, Any], preferred: str | None = None) -> dict[str, Any]:
    summaries = payload.get("scenario_summaries") or {}
    if preferred and preferred in summaries:
        return summaries[preferred]
    if len(summaries) != 1:
        raise SystemExit(f"expected exactly one scenario summary, got {list(summaries)}")
    return next(iter(summaries.values()))


def _requests(payload: dict[str, Any], preferred: str | None = None) -> list[dict[str, Any]]:
    details = payload.get("detailed_results") or {}
    if preferred and preferred in details:
        return list(details[preferred].get("requests") or [])
    if len(details) != 1:
        raise SystemExit(f"expected exactly one detailed result, got {list(details)}")
    return list(next(iter(details.values())).get("requests") or [])


def _as_float(value: Any, field: str) -> float:
    if value is None:
        raise SystemExit(f"missing required numeric field: {field}")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{field}: expected numeric value, got {value!r}") from exc
    if not math.isfinite(out):
        raise SystemExit(f"{field}: non-finite value {out!r}")
    return out


def _percentile(values: Sequence[float], q: float) -> float:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        raise SystemExit(f"cannot compute p{q} over an empty value list")
    return float(np.percentile(np.asarray(clean, dtype=float), q))


def _mean(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        raise SystemExit("cannot compute mean over an empty value list")
    return float(np.mean(np.asarray(clean, dtype=float)))


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


def _source_set(model_key: str, model_label: str, round_dir: Path, prime_override: Path | None = None) -> SourceSet:
    run_tag = _round_run_tag(round_dir)
    sglang = ppf._main_summary_path(round_dir, run_tag, "sglang")
    vllm = ppf._main_summary_path(round_dir, run_tag, "vllm")
    prime = prime_override or ppf._main_summary_path(round_dir, run_tag, "faaslora")
    return SourceSet(
        model_key=model_key,
        model_label=model_label,
        round_dir=round_dir,
        sglang_summary=sglang,
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


def _token_count(req: dict[str, Any]) -> float:
    for key in ("completion_tokens", "output_tokens", "generated_tokens", "observed_output_tokens"):
        if req.get(key) is not None:
            return float(req[key])
    usage = req.get("usage") or {}
    if usage.get("completion_tokens") is not None:
        return float(usage["completion_tokens"])
    return 0.0


def _scheduled_offset(req: dict[str, Any], fallback: dict[str, Any]) -> float:
    for key in ("scheduled_arrival_offset_s", "arrival_time_s"):
        if req.get(key) is not None:
            return float(req[key])
    for key in ("arrival_time_s", "scheduled_arrival_offset_s"):
        if fallback.get(key) is not None:
            return float(fallback[key])
    return 0.0


def _validate_request_pair(idx: int, prime_req: dict[str, Any], backend_req: dict[str, Any]) -> None:
    if prime_req.get("request_id") != backend_req.get("request_id"):
        raise SystemExit(
            f"request_id mismatch at index {idx}: "
            f"{prime_req.get('request_id')!r} vs {backend_req.get('request_id')!r}"
        )
    if prime_req.get("adapter_id") != backend_req.get("adapter_id"):
        raise SystemExit(
            f"adapter_id mismatch at {prime_req.get('request_id')}: "
            f"{prime_req.get('adapter_id')!r} vs {backend_req.get('adapter_id')!r}"
        )
    if prime_req.get("success") is not True or backend_req.get("success") is not True:
        raise SystemExit(f"unsuccessful request cannot be used for portability projection: {prime_req.get('request_id')}")


def _copy_prime_cost_components(metrics: dict[str, float]) -> dict[str, float]:
    keys = [
        "completed",
        "cost_req_usd",
        "cost_startup_usd",
        "cost_active_usd",
        "cost_idle_ready_usd",
        "cost_invocation_usd",
        "monetary_cost_total_usd",
        "monetary_active_charge_gpu_seconds",
        "monetary_idle_charge_gpu_seconds",
        "infra_active_gpu_seconds",
        "infra_idle_ready_gpu_seconds",
        "infra_startup_gpu_seconds",
        "serverless_invocation_cost_per_request_usd",
        "cost_1mtok_usd",
    ]
    return {key: float(metrics[key]) for key in keys if key in metrics}


def _derive_primelora_sglang(model: SourceSet, out_results_dir: Path) -> PortabilityRow:
    backend_payload = _load_json(model.sglang_summary)
    prime_payload = _load_json(model.prime_vllm_summary)
    backend_summary = _summary(backend_payload)
    prime_summary = _summary(prime_payload, "faaslora_full")
    backend_reqs = _requests(backend_payload)
    prime_reqs = _requests(prime_payload, "faaslora_full")
    if len(backend_reqs) != len(prime_reqs):
        raise SystemExit(
            f"{model.model_key}: request-count mismatch for PrimeLoRA-SGLang projection: "
            f"backend={len(backend_reqs)} prime={len(prime_reqs)}"
        )

    derived_requests: list[dict[str, Any]] = []
    ttft_values: list[float] = []
    e2e_values: list[float] = []
    service_ttft_values: list[float] = []
    service_e2e_values: list[float] = []
    dispatch_values: list[float] = []
    tpot_values: list[float] = []
    output_tokens = 0.0
    latest_completion = 0.0

    for idx, (prime_req, backend_req) in enumerate(zip(prime_reqs, backend_reqs)):
        _validate_request_pair(idx, prime_req, backend_req)
        dispatch_ms = _as_float(
            prime_req.get("dispatch_admission_wait_ms"),
            f"{model.model_key}.{prime_req.get('request_id')}.dispatch_admission_wait_ms",
        )
        service_ttft_ms = _as_float(
            backend_req.get("service_ttft_ms"),
            f"{model.model_key}.{backend_req.get('request_id')}.sglang_service_ttft_ms",
        )
        service_e2e_ms = _as_float(
            backend_req.get("service_e2e_ms"),
            f"{model.model_key}.{backend_req.get('request_id')}.sglang_service_e2e_ms",
        )
        tpot_raw = backend_req.get("tpot_ms")
        tpot_ms = float(tpot_raw) if tpot_raw is not None else None
        overall_ttft_ms = dispatch_ms + service_ttft_ms
        overall_e2e_ms = dispatch_ms + service_e2e_ms
        tokens = _token_count(backend_req)
        scheduled_s = _scheduled_offset(prime_req, backend_req)
        completion_s = scheduled_s + overall_e2e_ms / 1000.0
        latest_completion = max(latest_completion, completion_s)
        output_tokens += tokens

        ttft_values.append(overall_ttft_ms)
        e2e_values.append(overall_e2e_ms)
        service_ttft_values.append(service_ttft_ms)
        service_e2e_values.append(service_e2e_ms)
        dispatch_values.append(dispatch_ms)
        if tpot_ms is not None and math.isfinite(tpot_ms):
            tpot_values.append(tpot_ms)
        derived_requests.append(
            {
                "request_id": prime_req.get("request_id"),
                "adapter_id": prime_req.get("adapter_id"),
                "success": True,
                "scheduled_arrival_offset_s": scheduled_s,
                "completion_offset_s": completion_s,
                "dispatch_admission_wait_ms": dispatch_ms,
                "service_ttft_ms": service_ttft_ms,
                "service_e2e_ms": service_e2e_ms,
                "overall_ttft_ms": overall_ttft_ms,
                "overall_e2e_ms": overall_e2e_ms,
                "ttft_ms": overall_ttft_ms,
                "e2e_ms": overall_e2e_ms,
                "tpot_ms": tpot_ms,
                "tpot_observed": tpot_ms is not None,
                "completion_tokens": tokens,
                "output_tokens": tokens,
                "prime_control_source": str(model.prime_vllm_summary),
                "sglang_backend_source": str(model.sglang_summary),
                "derived_backend_substitution": True,
            }
        )

    prime_metrics = main_artifacts._metrics_from_summary(model.prime_vllm_summary, "faaslora")
    copied_costs = _copy_prime_cost_components(prime_metrics)
    completed = float(len(derived_requests))
    elapsed_sec = latest_completion
    tok_s = output_tokens / max(elapsed_sec, 1e-12)
    cost_req = copied_costs["cost_req_usd"]
    e2e_avg = _mean(e2e_values)
    ce = 1.0 / ((e2e_avg / 1000.0) * cost_req)

    metrics = {
        **copied_costs,
        "completed": completed,
        "ttft_avg_ms": _mean(ttft_values),
        "ttft_p95_ms": _percentile(ttft_values, 95),
        "e2e_avg_ms": e2e_avg,
        "e2e_p95_ms": _percentile(e2e_values, 95),
        "tpot_avg_ms": _mean(tpot_values),
        "tpot_p95_ms": _percentile(tpot_values, 95),
        "tok_s": tok_s,
        "ce": ce,
        "total_output_tokens": output_tokens,
        "elapsed_sec": elapsed_sec,
        "cost_1mtok_usd": cost_req * completed / max(output_tokens, 1e-12) * 1_000_000.0,
    }
    diagnostics = {
        "service_ttft_ms": _mean(service_ttft_values),
        "dispatch_wait_ms": _mean(dispatch_values),
        "service_e2e_ms": _mean(service_e2e_values),
    }

    summary = {
        "scenario_name": "primelora_sglang_portability",
        "baseline_type": "primelora_sglang",
        "metric_schema_version": "e2e_v3",
        "completed_requests": int(completed),
        "total_requests": int(completed),
        "failed_requests": 0,
        "avg_overall_ttft_ms": metrics["ttft_avg_ms"],
        "p95_overall_ttft_ms": metrics["ttft_p95_ms"],
        "avg_overall_e2e_ms": metrics["e2e_avg_ms"],
        "p95_overall_e2e_ms": metrics["e2e_p95_ms"],
        "avg_service_ttft_ms": diagnostics["service_ttft_ms"],
        "p95_service_ttft_ms": _percentile(service_ttft_values, 95),
        "avg_service_e2e_ms": diagnostics["service_e2e_ms"],
        "p95_service_e2e_ms": _percentile(service_e2e_values, 95),
        "avg_dispatch_admission_wait_ms": diagnostics["dispatch_wait_ms"],
        "p95_dispatch_admission_wait_ms": _percentile(dispatch_values, 95),
        "avg_tpot_ms": metrics["tpot_avg_ms"],
        "p95_tpot_ms": metrics["tpot_p95_ms"],
        "throughput_tok_per_s": metrics["tok_s"],
        "total_output_tokens": output_tokens,
        "monetary_cost_per_request_usd": cost_req,
        "avg_cost_usd": cost_req,
        "monetary_cost_total_usd": copied_costs.get("monetary_cost_total_usd", cost_req * completed),
        "monetary_ce": ce,
        "ce": ce,
        "infra_active_gpu_seconds": copied_costs["infra_active_gpu_seconds"],
        "infra_idle_ready_gpu_seconds": copied_costs["infra_idle_ready_gpu_seconds"],
        "infra_startup_gpu_seconds": copied_costs["infra_startup_gpu_seconds"],
        "monetary_active_charge_gpu_seconds": copied_costs.get("monetary_active_charge_gpu_seconds"),
        "monetary_idle_charge_gpu_seconds": copied_costs.get("monetary_idle_charge_gpu_seconds"),
        "serverless_invocation_cost_per_request_usd": copied_costs.get("serverless_invocation_cost_per_request_usd", 0.0),
        "serverless_idle_gpu_cost_factor": prime_summary.get("serverless_idle_gpu_cost_factor"),
        "projection_note": (
            "Request-matched backend portability projection: PrimeLoRA control-path "
            "wait/lifecycle cost envelope plus measured SGLang service-path timings."
        ),
    }
    comparison_row = {
        "scenario": "primelora_sglang_portability",
        "completed": int(completed),
        "TTFT_avg_ms": round(metrics["ttft_avg_ms"], 4),
        "TTFT_P95_ms": round(metrics["ttft_p95_ms"], 4),
        "TTFT_e2e_avg_ms": round(metrics["ttft_avg_ms"], 4),
        "TTFT_e2e_P95_ms": round(metrics["ttft_p95_ms"], 4),
        "TTFT_service_avg_ms": round(diagnostics["service_ttft_ms"], 4),
        "Dispatch_admission_wait_avg_ms": round(diagnostics["dispatch_wait_ms"], 4),
        "E2E_avg_ms": round(metrics["e2e_avg_ms"], 4),
        "E2E_P95_ms": round(metrics["e2e_p95_ms"], 4),
        "E2E_service_avg_ms": round(diagnostics["service_e2e_ms"], 4),
        "TPOT_avg_ms": round(metrics["tpot_avg_ms"], 4),
        "TPOT_P95_ms": round(metrics["tpot_p95_ms"], 4),
        "throughput_TOKPS": round(metrics["tok_s"], 6),
        "avg_cost_USD": round(cost_req, 8),
        "CE": round(ce, 6),
        "monetary_ce": round(ce, 6),
        "method": "request_matched_portability_projection",
    }
    payload = {
        "schema_version": "1.0",
        "metric_schema_version": "e2e_v3",
        "metadata": {
            "artifact": "PrimeLoRA-SGLang backend portability projection",
            "model_key": model.model_key,
            "model_label": model.model_label,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "round_dir": str(model.round_dir),
            "prime_control_source": str(model.prime_vllm_summary),
            "sglang_backend_source": str(model.sglang_summary),
            "vllm_baseline_source": str(model.vllm_summary),
            "method": (
                "Per-request projection using PrimeLoRA-vLLM dispatch/admission wait "
                "and lifecycle cost with SGLang measured service_ttft/service_e2e/tpot "
                "for the same request_id and adapter_id."
            ),
            "not_formal_runtime_replacement": True,
        },
        "comparison_table": [comparison_row],
        "scenario_summaries": {"primelora_sglang_portability": summary},
        "detailed_results": {
            "primelora_sglang_portability": {
                "scenario_name": "primelora_sglang_portability",
                "baseline_type": "primelora_sglang",
                "total": int(completed),
                "completed": int(completed),
                "failed": 0,
                "elapsed_sec": elapsed_sec,
                "requests": derived_requests,
            }
        },
    }
    out_results_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_results_dir / f"{model.model_key}_primelora_sglang_portability.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return PortabilityRow(
        model_key=model.model_key,
        model_label=model.model_label,
        system_key="primelora_sglang",
        system_label=SYSTEM_LABELS["primelora_sglang"],
        source=out_path,
        method="request-matched projection",
        metrics=metrics,
        diagnostics=diagnostics,
    )


def _rows_for_models(models: Sequence[SourceSet], out_results_dir: Path) -> list[PortabilityRow]:
    rows: list[PortabilityRow] = []
    for model in models:
        rows.append(_measured_row(model, "sglang", model.sglang_summary, "sglang"))
        rows.append(_derive_primelora_sglang(model, out_results_dir))
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
        "\\caption{Backend portability sensitivity using SGLang and vLLM service paths. PrimeLoRA-SGLang is a request-matched projection that keeps PrimeLoRA's measured control/lifecycle envelope and substitutes measured SGLang service-path timings for the same replay.}",
        "\\label{tab:backend_portability}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{1.9pt}",
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
    for mi, model_key in enumerate(by_model):
        if mi:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{model_labels[model_key]}}}}} \\\\")
        best = {
            "ttft_avg_ms": _best_keys(rows, model_key, "ttft_avg_ms"),
            "ttft_p95_ms": _best_keys(rows, model_key, "ttft_p95_ms"),
            "e2e_avg_ms": _best_keys(rows, model_key, "e2e_avg_ms"),
            "e2e_p95_ms": _best_keys(rows, model_key, "e2e_p95_ms"),
            "tpot_avg_ms": _best_keys(rows, model_key, "tpot_avg_ms"),
            "tpot_p95_ms": _best_keys(rows, model_key, "tpot_p95_ms"),
            "tok_s": _best_keys(rows, model_key, "tok_s", higher=True),
            "cost_req_usd": _best_keys(rows, model_key, "cost_req_usd"),
            "ce": _best_keys(rows, model_key, "ce", higher=True),
        }
        ordered = sorted(by_model[model_key], key=lambda row: SYSTEM_ORDER.index(row.system_key))
        for row in ordered:
            m = row.metrics
            lines.append(
                " & ".join(
                    [
                        row.system_label,
                        _fmt_ms(m["ttft_avg_ms"], bold=row.system_key in best["ttft_avg_ms"]),
                        _fmt_ms(m["ttft_p95_ms"], bold=row.system_key in best["ttft_p95_ms"]),
                        _fmt_ms(m["e2e_avg_ms"], bold=row.system_key in best["e2e_avg_ms"]),
                        _fmt_ms(m["e2e_p95_ms"], bold=row.system_key in best["e2e_p95_ms"]),
                        _fmt(m["tpot_avg_ms"], "{:.1f}", bold=row.system_key in best["tpot_avg_ms"]),
                        _fmt(m["tpot_p95_ms"], "{:.1f}", bold=row.system_key in best["tpot_p95_ms"]),
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
        "\\caption{Backend-portability first-token decomposition. PrimeLoRA-SGLang keeps the measured PrimeLoRA dispatch wait and uses request-matched SGLang service TTFT.}",
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


def write_manifest(rows: Sequence[PortabilityRow], out_dir: Path, results_dir: Path) -> None:
    payload = {
        "artifact": "primelora_sglang_backend_portability",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Measured SGLang/vLLM baselines and PrimeLoRA-vLLM results are reused. "
            "PrimeLoRA-SGLang is generated by request-matched substitution of SGLang "
            "service timings into PrimeLoRA's measured control/lifecycle envelope."
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
            "derived_results_dir": str(results_dir),
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


def build(models: Sequence[SourceSet], out_dir: Path, results_dir: Path, copy_root_fig: bool = True) -> list[PortabilityRow]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows_for_models(models, results_dir)
    write_main_table(rows, out_dir)
    write_decomposition_table(rows, out_dir)
    write_lifecycle_figure(rows, out_dir)
    write_manifest(rows, out_dir, results_dir)
    if copy_root_fig:
        root_fig = Path("figs") / "fig_backend_portability_lifecycle_cost.pdf"
        root_fig.parent.mkdir(parents=True, exist_ok=True)
        root_fig.write_bytes((out_dir / "fig_backend_portability_lifecycle_cost.pdf").read_bytes())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PrimeLoRA-SGLang backend portability artifacts.")
    parser.add_argument("--round-7b", type=Path, default=DEFAULT_7B_ROUND)
    parser.add_argument("--round-3b", type=Path, default=DEFAULT_3B_ROUND)
    parser.add_argument("--prime-3b-summary", type=Path, default=DEFAULT_3B_PRIME)
    parser.add_argument("--out-dir", type=Path, default=Path("figs/paper/backend_portability"))
    parser.add_argument("--results-dir", type=Path, default=Path("figs/paper/backend_portability/derived_results"))
    args = parser.parse_args()

    models = [
        _source_set("llama2_7b", "Llama-2 7B", args.round_7b.resolve()),
        _source_set("llama32_3b", "Llama-3.2 3B", args.round_3b.resolve(), args.prime_3b_summary.resolve()),
    ]
    rows = build(models, args.out_dir.resolve(), args.results_dir.resolve())
    print(f"wrote backend portability artifacts to {args.out_dir.resolve()}")
    for row in rows:
        print(
            f"{row.model_label:14s} {row.system_label:18s} "
            f"E2E={row.metrics['e2e_avg_ms']:.1f}ms Cost={row.metrics['cost_req_usd']*1000.0:.3f}mUSD "
            f"CE={row.metrics['ce']:.2f} method={row.method}"
        )


if __name__ == "__main__":
    main()
