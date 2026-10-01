#!/usr/bin/env python3
"""Generate publication figures from PrimeLoRA paper experiment rounds.

The script intentionally reads only completed, audited JSON results. It writes
the PDF figure, a CSV data dump, and a small manifest for each generated figure.
Missing fields fail fast instead of being silently converted to zero.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:  # Direct ``python scripts/...`` execution.
    from eurosys27_v2_provenance import (
        FormalAnalysisIdentity,
        build_formal_provenance_index,
        validate_formal_analysis_sources,
    )
except ModuleNotFoundError:  # Package import used by the test suite.
    from scripts.eurosys27_v2_provenance import (
        FormalAnalysisIdentity,
        build_formal_provenance_index,
        validate_formal_analysis_sources,
    )


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
MAIN_SUMMARY_OVERRIDES: Dict[str, Path] = {}
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


V2_ABLATION_SCENARIOS = (
    "v2_elastic_only",
    "v2_hit_aware_preparation",
    "v2_hierarchical_no_coord",
    "v2_full",
)
V2_ABLATION_LABELS = {
    "v2_elastic_only": "ElasticOnly",
    "v2_hit_aware_preparation": "+Hit-aware prep.",
    "v2_hierarchical_no_coord": "+Hierarchy",
    "v2_full": "Full (+Admission)",
}
V2_ABLATION_COLORS = {
    "v2_elastic_only": "#A8A8A8",
    "v2_hit_aware_preparation": "#7FA7D9",
    "v2_hierarchical_no_coord": "#F2B36D",
    "v2_full": "#78B87A",
}
V2_ABLATION_FEATURE_GATES = {
    "v2_elastic_only": (False, False, False, False, False),
    "v2_hit_aware_preparation": (True, True, False, False, False),
    "v2_hierarchical_no_coord": (True, True, True, False, False),
    "v2_full": (True, True, True, True, True),
}
V2_ABLATION_METRICS = (
    ("p95_overall_ttft_ms", "P95 TTFT (ms)", False),
    ("avg_overall_e2e_ms", "Average E2E (ms)", False),
    ("monetary_cost_per_request_usd", "Cost/request (USD)", False),
    ("monetary_ce", "CE", True),
)
V2_ABLATION_AUXILIARY_METRICS = (
    ("avg_overall_ttft_ms", "Average TTFT (ms)", False),
    ("avg_tpot_ms", "Average TPOT (ms)", False),
    ("throughput_tok_per_s", "Throughput (token/s)", True),
)
V2_ABLATION_TABLE_METRICS = (
    *V2_ABLATION_METRICS,
    *V2_ABLATION_AUXILIARY_METRICS,
)
V2_ABLATION_TRIGGER_FIELDS = (
    "routing_decision_count",
    "readiness_aware_routing_decision_count",
    "load_only_routing_decision_count",
    "scale_up_event_count",
    "scale_up_events_with_planned_adapters",
    "scaleup_first_service_request_count",
    "scaleup_first_service_planned_match_count",
    "initial_or_current_nvme_adapter_count",
    "initial_or_current_host_adapter_count",
    "host_promotion_scheduled_count",
    "host_promotion_completed_count",
    "runtime_gpu_forward_attempt_count",
    "runtime_gpu_forward_success_count",
    "dispatch_to_service_transition_count",
    "gpu_admission_observed_request_count",
    "gpu_admission_decision_count",
    "gpu_admission_admit_count",
    "gpu_admission_defer_count",
    "gpu_admission_reject_count",
)
V2_ABLATION_ADJACENT_REFERENCES = {
    "v2_hit_aware_preparation": ("M1", "v2_elastic_only"),
    "v2_hierarchical_no_coord": ("M2", "v2_hit_aware_preparation"),
    "v2_full": ("M3", "v2_hierarchical_no_coord"),
}

V2_FORMAL_SEEDS = (43, 44, 45)


@dataclass(frozen=True)
class V2AblationResult:
    model: str
    scenario: str
    seed: int
    run_tag: str
    source: Path
    total: int
    completed: int
    shared_trace_sha256: str
    shared_adapter_subset_sha256: str
    generation_contract: str
    generation_contract_request_map_sha256: str
    non_feature_frozen_config_sha256: str
    formal_axes: Dict[str, Any]
    metrics: Dict[str, float]
    triggers: Dict[str, int]


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


def _optional_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


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
    return ((baseline - value) / baseline) * 100.0


def _v2_read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"cannot parse V2 result {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"V2 result must be a JSON object: {path}")
    return payload


def _manifest_json_references(manifest: Path) -> List[Path]:
    try:
        payload = _v2_read_json(manifest)
    except SystemExit:
        return []
    references: List[Path] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and (value.endswith(".json") or value.endswith(".json.gz")):
            raw = Path(value).expanduser()
            candidates = [raw] if raw.is_absolute() else [manifest.parent / raw, raw]
            for candidate in candidates:
                if candidate.is_file():
                    references.append(candidate.resolve())
                    break

    visit(payload)
    return references


def _v2_result_candidates(inputs: Sequence[Path]) -> List[Path]:
    candidates: set[Path] = set()
    for raw_input in inputs:
        input_path = raw_input.expanduser().resolve()
        if input_path.is_file():
            candidates.add(input_path)
            if input_path.name == "MANIFEST.json":
                candidates.update(_manifest_json_references(input_path))
            continue
        if not input_path.is_dir():
            raise SystemExit(f"V2 ablation input does not exist: {input_path}")
        candidates.update(path.resolve() for path in input_path.rglob("*_result.json"))
        candidates.update(path.resolve() for path in input_path.rglob("*_result.json.gz"))
        manifests = list(input_path.rglob("MANIFEST.json"))
        for manifest in manifests:
            candidates.update(_manifest_json_references(manifest))
    return sorted(candidates)


def _v2_model_identity(payload: Dict[str, Any], path: Path) -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    selection = metadata.get("profile_selection") if isinstance(metadata.get("profile_selection"), dict) else {}
    raw = selection.get("model") or metadata.get("model_profile") or metadata.get("model") or payload.get("model")
    if raw:
        text = str(raw).strip().rstrip("/")
        return Path(text).name or text
    for part in reversed(path.parts):
        if re.search(r"(?:llama|model).*(?:3b|7b|13b)", part, flags=re.IGNORECASE):
            return part
    raise SystemExit(f"cannot determine model identity for V2 result: {path}")


def _v2_seed(payload: Dict[str, Any], path: Path, run_tag: str) -> int:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    direct = (
        metadata.get("sampling_seed"),
        metadata.get("generation_seed"),
        metadata.get("workload_seed"),
        metadata.get("seed"),
        payload.get("generation_seed"),
        payload.get("seed"),
    )
    for raw in direct:
        if raw is not None and str(raw).strip() != "":
            try:
                return int(raw)
            except Exception:
                pass
    search_text = "_".join((run_tag, *path.parts))
    match = re.search(r"(?:^|[_/.-])seed[_-]?(\d+)(?:$|[_/.-])", search_text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    raise SystemExit(f"cannot determine seed for V2 result {path}; record metadata.seed/generation_seed")


def _v2_metric(detail: Dict[str, Any], summary: Dict[str, Any], key: str, label: str) -> float:
    aliases = {
        "p95_overall_ttft_ms": ("p95_overall_ttft_ms", "TTFT_e2e_P95_ms"),
        "avg_overall_ttft_ms": ("avg_overall_ttft_ms", "TTFT_e2e_avg_ms", "TTFT_avg_ms"),
        "avg_overall_e2e_ms": ("avg_overall_e2e_ms", "E2E_e2e_avg_ms", "E2E_avg_ms"),
        "avg_tpot_ms": ("avg_tpot_ms", "TPOT_avg_ms"),
        "throughput_tok_per_s": (
            "throughput_tok_per_s",
            "throughput_TOKPS",
            "Throughput_TOKPS",
        ),
        "monetary_cost_per_request_usd": ("monetary_cost_per_request_usd", "Monetary_cost_per_request_usd"),
        "monetary_ce": ("monetary_ce", "Monetary_CE", "CE"),
    }
    for alias in aliases[key]:
        if detail.get(alias) is not None:
            return _as_float(detail.get(alias), f"{label}.{alias}")
        if summary.get(alias) is not None:
            return _as_float(summary.get(alias), f"{label}.{alias}")
    raise SystemExit(f"{label}: missing V2 ablation metric {key}")


def _v2_sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise SystemExit(f"{label}: missing or invalid SHA-256 digest")
    return digest


def _v2_generation_contract_map_sha256(requests: Sequence[Dict[str, Any]]) -> str:
    rows = [
        {
            "request_id": request.get("request_id"),
            "adapter_id": request.get("adapter_id"),
            "arrival_time_s": request.get("scheduled_arrival_offset_s"),
            "source_expected_output_tokens": int(
                request.get("source_expected_output_tokens", 0) or 0
            ),
            "requested_completion_tokens": int(
                request.get("requested_completion_tokens", 0) or 0
            ),
            "canonical_prompt_sha256": str(
                request.get("canonical_prompt_sha256", "") or ""
            ),
            "canonical_prompt_tokens": int(
                request.get("canonical_prompt_tokens", 0) or 0
            ),
        }
        for request in requests
    ]
    encoded = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _v2_ablation_audit(
    *,
    metadata: Dict[str, Any],
    detail: Dict[str, Any],
    scenario: str,
    completed: int,
    label: str,
) -> tuple[str, str, str, str, Dict[str, int]]:
    """Fail closed on provenance, generation, dispatch, and mechanism evidence."""
    trace_sha = _v2_sha256(metadata.get("shared_trace_sha256"), f"{label}.shared_trace_sha256")
    subset_sha = _v2_sha256(
        metadata.get("shared_adapter_subset_sha256"),
        f"{label}.shared_adapter_subset_sha256",
    )

    contract = str(metadata.get("generation_contract") or "").strip().lower()
    if contract not in {"legacy", "fixed_length_greedy_v1"}:
        raise SystemExit(
            f"{label}: missing or unsupported generation_contract={contract!r}"
        )
    contract_maps = metadata.get("generation_contract_request_map_sha256")
    if not isinstance(contract_maps, dict):
        raise SystemExit(f"{label}: missing generation_contract_request_map_sha256 mapping")
    recorded_contract_map_sha = _v2_sha256(
        contract_maps.get(scenario),
        f"{label}.generation_contract_request_map_sha256[{scenario}]",
    )

    requests = detail.get("requests")
    if not isinstance(requests, list) or len(requests) != completed:
        observed = len(requests) if isinstance(requests, list) else "missing"
        raise SystemExit(
            f"{label}: request-level records must cover every completed request; "
            f"observed={observed}, completed={completed}"
        )
    legal_tiers = {"gpu", "host", "nvme", "remote"}
    actual_dispatch_counts = {tier: 0 for tier in sorted(legal_tiers)}
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise SystemExit(f"{label}: request[{index}] is not an object")
        if not bool(request.get("success")):
            raise SystemExit(f"{label}: request[{index}] is not successful")
        if not str(request.get("adapter_id") or "").strip():
            raise SystemExit(f"{label}: request[{index}] is missing adapter_id")
        request_contract = str(request.get("generation_contract") or "").strip().lower()
        if request_contract != contract:
            raise SystemExit(
                f"{label}: request[{index}] generation_contract={request_contract!r} "
                f"does not match metadata={contract!r}"
            )
        tier = str(request.get("readiness_tier_before_dispatch") or "").strip().lower()
        if tier not in legal_tiers:
            raise SystemExit(
                f"{label}: incomplete dispatch-time readiness tier at request[{index}]: {tier!r}"
            )
        actual_dispatch_counts[tier] += 1
        if contract == "fixed_length_greedy_v1":
            _v2_sha256(
                request.get("canonical_prompt_sha256"),
                f"{label}:request[{index}].canonical_prompt_sha256",
            )
            target = int(request.get("requested_completion_tokens", 0) or 0)
            actual = int(request.get("completion_tokens", 0) or 0)
            if target <= 0 or actual != target or request.get("output_contract_match") is not True:
                raise SystemExit(
                    f"{label}: fixed generation contract mismatch at request[{index}]: "
                    f"actual_tokens={actual}, target_tokens={target}"
                )
    recomputed_contract_map_sha = _v2_generation_contract_map_sha256(requests)
    if recorded_contract_map_sha != recomputed_contract_map_sha:
        raise SystemExit(
            f"{label}: generation contract request-map SHA mismatch: "
            f"recorded={recorded_contract_map_sha}, recomputed={recomputed_contract_map_sha}"
        )

    scenario_coordination = metadata.get("scenario_coordination")
    coordination = (
        scenario_coordination.get(scenario)
        if isinstance(scenario_coordination, dict)
        and isinstance(scenario_coordination.get(scenario), dict)
        else None
    )
    if not isinstance(coordination, dict):
        raise SystemExit(f"{label}: missing scenario_coordination audit block")
    gates = coordination.get("feature_gates")
    if not isinstance(gates, dict):
        raise SystemExit(f"{label}: missing feature_gates audit block")
    gate_names = (
        "readiness_routing_enabled",
        "scale_up_handoff_enabled",
        "hierarchical_residency_enabled",
        "coordination_enabled",
        "effective_capacity_admission_enabled",
    )
    actual_gates = tuple(bool(gates.get(name)) for name in gate_names)
    expected_gates = V2_ABLATION_FEATURE_GATES[scenario]
    if actual_gates != expected_gates:
        raise SystemExit(
            f"{label}: feature gate mismatch; expected={expected_gates}, actual={actual_gates}"
        )
    activation = coordination.get("feature_activation")
    if not isinstance(activation, dict):
        raise SystemExit(f"{label}: missing feature_activation audit block")

    def count(name: str) -> int:
        try:
            value = int(activation.get(name, 0) or 0)
        except Exception as exc:
            raise SystemExit(f"{label}: invalid activation counter {name}") from exc
        if value < 0:
            raise SystemExit(f"{label}: negative activation counter {name}={value}")
        return value

    if count("successful_request_count") != completed or count("routing_decision_count") != completed:
        raise SystemExit(f"{label}: routing/success activation counts do not cover completed requests")
    if count("routing_selection_attempt_count") < completed:
        raise SystemExit(f"{label}: routing selection attempts are incomplete")
    readiness_count = count("readiness_aware_routing_decision_count")
    load_only_count = count("load_only_routing_decision_count")
    if expected_gates[0]:
        if readiness_count != completed or load_only_count != 0:
            raise SystemExit(f"{label}: readiness-aware routing activation is inconsistent")
    elif load_only_count != completed or readiness_count != 0:
        raise SystemExit(f"{label}: ElasticOnly routing is not purely load-only")

    planned_handoffs = count("scale_up_events_with_planned_adapters")
    first_service_count = count("scaleup_first_service_request_count")
    planned_match_count = count("scaleup_first_service_planned_match_count")
    if expected_gates[1]:
        if (
            count("scale_up_event_count") <= 0
            or planned_handoffs <= 0
            or first_service_count <= 0
            or planned_match_count <= 0
        ):
            raise SystemExit(f"{label}: scale-out handoff enabled but never triggered")
        if count("initial_or_current_nvme_adapter_count") <= 0:
            raise SystemExit(f"{label}: hit-aware preparation enabled but NVMe was never populated")
    elif planned_handoffs != 0 or first_service_count != 0 or planned_match_count != 0:
        raise SystemExit(f"{label}: disabled scale-out handoff recorded planned/served activity")

    hierarchy_specific = (
        count("host_promotion_completed_count")
        + count("runtime_gpu_forward_success_count")
    )
    if expected_gates[2]:
        if count("initial_or_current_host_adapter_count") <= 0:
            raise SystemExit(f"{label}: hierarchy enabled but HOST tier was never populated")
        if count("initial_or_current_nvme_adapter_count") <= 0:
            raise SystemExit(f"{label}: hierarchy enabled but NVMe tier was never populated")
        if hierarchy_specific <= 0:
            raise SystemExit(f"{label}: hierarchy enabled but no online tier transition completed")
    elif any(
        count(name) != 0
        for name in (
            "initial_or_current_host_adapter_count",
            "host_promotion_scheduled_count",
            "host_promotion_completed_count",
            "runtime_gpu_forward_attempt_count",
            "runtime_gpu_forward_success_count",
        )
    ):
        raise SystemExit(f"{label}: disabled hierarchy recorded hierarchy-specific activation")

    admission_count = count("gpu_admission_decision_count")
    admission_requests = count("gpu_admission_observed_request_count")
    admission_outcomes = {
        "admit": count("gpu_admission_admit_count"),
        "defer": count("gpu_admission_defer_count"),
        "reject": count("gpu_admission_reject_count"),
    }
    admission_outcome_total = sum(admission_outcomes.values())
    if expected_gates[4]:
        if admission_count <= 0 or admission_requests <= 0:
            raise SystemExit(f"{label}: effective-capacity admission enabled but never triggered")
        if admission_outcome_total <= 0:
            raise SystemExit(f"{label}: admission outcome total must be positive")
        if admission_outcome_total != admission_count:
            raise SystemExit(
                f"{label}: admission outcome invariant violated; "
                f"admit+defer+reject={admission_outcome_total}, "
                f"decisions={admission_count}, outcomes={admission_outcomes}"
            )
    elif (
        admission_count != 0
        or admission_requests != 0
        or admission_outcome_total != 0
    ):
        raise SystemExit(f"{label}: disabled admission recorded admission decisions")

    recorded_dispatch = activation.get("dispatch_tier_counts")
    if not isinstance(recorded_dispatch, dict):
        raise SystemExit(f"{label}: missing dispatch_tier_counts")
    normalized_dispatch = {
        tier: int(recorded_dispatch.get(tier, 0) or 0) for tier in sorted(legal_tiers)
    }
    if normalized_dispatch != actual_dispatch_counts:
        raise SystemExit(
            f"{label}: dispatch tier counts disagree with request records; "
            f"recorded={normalized_dispatch}, actual={actual_dispatch_counts}"
        )
    return (
        trace_sha,
        subset_sha,
        contract,
        recorded_contract_map_sha,
        {name: count(name) for name in V2_ABLATION_TRIGGER_FIELDS},
    )


def _v2_dig(mapping: Any, *path: str) -> Any:
    current = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _v2_consistent_axis(
    label: str,
    candidates: Sequence[Any],
    normalize: Callable[[Any], Any],
) -> Any:
    values: List[Any] = []
    for candidate in candidates:
        if candidate is None or str(candidate).strip() == "":
            continue
        try:
            values.append(normalize(candidate))
        except Exception as exc:
            raise SystemExit(f"{label}: invalid formal-axis value {candidate!r}") from exc
    if not values:
        return None
    if any(value != values[0] for value in values[1:]):
        raise SystemExit(f"{label}: conflicting recorded values {values}")
    return values[0]


def _v2_ablation_formal_axes(metadata: Dict[str, Any], label: str) -> Dict[str, Any]:
    profiles = [
        candidate
        for candidate in (
            metadata.get("shared_trace_load_profile"),
            _v2_dig(metadata, "shared_trace_metadata", "load_profile"),
            _v2_dig(metadata, "sampling_stats", "shared_trace_metadata", "load_profile"),
        )
        if isinstance(candidate, dict)
    ]

    def profile_values(*names: str) -> List[Any]:
        return [
            profile.get(name)
            for profile in profiles
            for name in names
            if profile.get(name) is not None
        ]

    return {
        "selected_num_adapters": _v2_consistent_axis(
            f"{label}.selected_num_adapters",
            [
                metadata.get("num_adapters"),
                _v2_dig(metadata, "shared_trace_metadata", "selected_num_adapters"),
                _v2_dig(
                    metadata,
                    "sampling_stats",
                    "shared_trace_metadata",
                    "selected_num_adapters",
                ),
            ],
            int,
        ),
        "bandwidth_mib_s": _v2_consistent_axis(
            f"{label}.bandwidth_mib_s",
            [metadata.get("bandwidth_mib_s"), metadata.get("bandwidth_mbps")],
            float,
        ),
        "configured_time_scale_factor": _v2_consistent_axis(
            f"{label}.configured_time_scale_factor",
            [
                metadata.get("configured_time_scale_factor"),
                metadata.get("shared_trace_configured_time_scale_factor"),
                _v2_dig(metadata, "shared_trace_metadata", "configured_time_scale_factor"),
            ],
            float,
        ),
        "effective_time_scale_factor": _v2_consistent_axis(
            f"{label}.effective_time_scale_factor",
            [
                metadata.get("effective_time_scale_factor"),
                metadata.get("shared_trace_effective_time_scale_factor"),
                _v2_dig(metadata, "shared_trace_metadata", "effective_time_scale_factor"),
            ],
            float,
        ),
        "zipf_exponent": _v2_consistent_axis(
            f"{label}.zipf_exponent",
            [metadata.get("zipf_exponent"), *profile_values("zipf_exponent")],
            float,
        ),
        "active_adapter_cap": _v2_consistent_axis(
            f"{label}.active_adapter_cap",
            [metadata.get("active_adapter_cap"), *profile_values("active_adapter_cap")],
            int,
        ),
        "hotset_rotation_requests": _v2_consistent_axis(
            f"{label}.hotset_rotation_requests",
            [
                metadata.get("hotset_rotation_requests"),
                *profile_values("hotset_rotation_requests", "rotation_requests"),
            ],
            int,
        ),
        "hotset_rotation_mode": _v2_consistent_axis(
            f"{label}.hotset_rotation_mode",
            [
                metadata.get("hotset_rotation_mode"),
                metadata.get("rotation_mode"),
                *profile_values("rotation_mode", "hotset_rotation_mode"),
            ],
            lambda value: str(value).strip().lower(),
        ),
        "hotset_overlap_fraction": _v2_consistent_axis(
            f"{label}.hotset_overlap_fraction",
            [
                metadata.get("hotset_overlap_fraction"),
                *profile_values(
                    "hotset_overlap_fraction",
                    "rotation_overlap_fraction",
                ),
            ],
            float,
        ),
    }


def _v2_formal_model_key(model: str) -> str:
    """Map recorded model labels to the two model identities in the V2 protocol."""
    normalized = re.sub(r"[^a-z0-9]+", "", str(model).lower())
    if "llama323b" in normalized:
        return "llama32_3b"
    if "llama27b" in normalized:
        return "llama2_7b"
    raise SystemExit(
        f"formal matrix contains unsupported model identity {model!r}; expected "
        "Llama-2-7B or Llama-3.2-3B"
    )


def _format_v2_matrix_identity(identity: tuple[str, str, int]) -> str:
    model, scenario, seed = identity
    return f"model={model},scenario={scenario},seed={seed}"


def validate_v2_ablation_formal_matrix(
    results: Sequence[V2AblationResult],
) -> None:
    """Require the exact A2/A3 held-out matrix declared in the V2 protocol.

    The formal analysis is deliberately all-or-nothing.  A partial campaign is
    still useful during exploration, but it must be plotted without
    ``--formal-matrix`` and cannot accidentally be published as the formal
    Fig. 9 dataset.
    """
    expected: set[tuple[str, str, int]] = {
        ("llama2_7b", scenario, 43) for scenario in V2_ABLATION_SCENARIOS
    }
    expected.update(
        ("llama2_7b", scenario, seed)
        for scenario in (
            "v2_elastic_only",
            "v2_hierarchical_no_coord",
            "v2_full",
        )
        for seed in (44, 45)
    )
    expected.update(
        ("llama32_3b", scenario, seed)
        for scenario in ("v2_elastic_only", "v2_full")
        for seed in V2_FORMAL_SEEDS
    )

    observed_counts: Dict[tuple[str, str, int], int] = defaultdict(int)
    for result in results:
        identity = (
            _v2_formal_model_key(result.model),
            result.scenario,
            result.seed,
        )
        observed_counts[identity] += 1

    observed = set(observed_counts)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    duplicates = sorted(
        identity for identity, count in observed_counts.items() if count != 1
    )
    if missing or extra or duplicates:
        parts = ["formal A2/A3 matrix identity mismatch"]
        if missing:
            parts.append(
                "missing=["
                + "; ".join(_format_v2_matrix_identity(item) for item in missing)
                + "]"
            )
        if extra:
            parts.append(
                "extra=["
                + "; ".join(_format_v2_matrix_identity(item) for item in extra)
                + "]"
            )
        if duplicates:
            parts.append(
                "duplicate=["
                + "; ".join(
                    f"{_format_v2_matrix_identity(item)} x{observed_counts[item]}"
                    for item in duplicates
                )
                + "]"
            )
        raise SystemExit("; ".join(parts))

    expected_axes = {
        "selected_num_adapters": 500,
        "bandwidth_mib_s": 250.0,
        "configured_time_scale_factor": 8.0,
        "effective_time_scale_factor": 8.0,
        "zipf_exponent": 1.0,
        "active_adapter_cap": 48,
        "hotset_rotation_requests": 500,
        "hotset_rotation_mode": "legacy",
        "hotset_overlap_fraction": 0.75,
    }
    non_feature_by_model: Dict[str, set[str]] = defaultdict(set)
    for result in results:
        model_key = _v2_formal_model_key(result.model)
        if result.total != 4000 or result.completed != 4000:
            raise SystemExit(
                f"formal A2/A3 {result.source}: expected 4000/4000 requests, "
                f"observed {result.completed}/{result.total}"
            )
        if result.generation_contract != "legacy":
            raise SystemExit(
                f"formal A2/A3 {result.source}: generation_contract must be legacy"
            )
        for name, expected_value in expected_axes.items():
            observed_value = result.formal_axes.get(name)
            if isinstance(expected_value, float):
                matches = observed_value is not None and math.isclose(
                    float(observed_value),
                    expected_value,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            else:
                matches = observed_value == expected_value
            if not matches:
                raise SystemExit(
                    f"formal A2/A3 {result.source}: wrong {name}; "
                    f"expected={expected_value!r}, observed={observed_value!r}"
                )
        frozen_hash = _v2_sha256(
            result.non_feature_frozen_config_sha256,
            f"formal A2/A3 {result.source}.non_feature_frozen_config_sha256",
        )
        non_feature_by_model[model_key].add(frozen_hash)
        for metric, _, _ in V2_ABLATION_TABLE_METRICS:
            if metric not in result.metrics or not math.isfinite(result.metrics[metric]):
                raise SystemExit(
                    f"formal A2/A3 {result.source}: missing complete-table metric {metric}"
                )
    drift = {
        model: sorted(hashes)
        for model, hashes in non_feature_by_model.items()
        if len(hashes) != 1
    }
    if drift:
        raise SystemExit(
            "formal A2/A3 cross-scenario non-feature frozen configuration drift: "
            f"{drift}"
        )


def load_v2_ablation_results(
    inputs: Sequence[Path], *, formal_matrix: bool = False
) -> List[V2AblationResult]:
    results: Dict[tuple[str, str, int], V2AblationResult] = {}
    candidates = _v2_result_candidates(inputs)
    for path in candidates:
        payload = _v2_read_json(path)
        detailed = payload.get("detailed_results")
        if not isinstance(detailed, dict):
            continue
        if not any(isinstance(detailed.get(scenario), dict) for scenario in V2_ABLATION_SCENARIOS):
            continue
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        run_tag = str(metadata.get("results_tag") or metadata.get("run_tag") or payload.get("run_tag") or path.stem)
        model = _v2_model_identity(payload, path)
        seed = _v2_seed(payload, path, run_tag)
        scenario_summaries = payload.get("scenario_summaries") if isinstance(payload.get("scenario_summaries"), dict) else {}
        for scenario in V2_ABLATION_SCENARIOS:
            detail = detailed.get(scenario)
            if not isinstance(detail, dict):
                continue
            summary = scenario_summaries.get(scenario) if isinstance(scenario_summaries.get(scenario), dict) else {}
            total = int(_as_float(detail.get("total", summary.get("total_requests")), f"{path}:{scenario}.total"))
            completed = int(_as_float(detail.get("completed", summary.get("completed_requests")), f"{path}:{scenario}.completed"))
            if total <= 0 or completed != total:
                raise SystemExit(f"{path}:{scenario}: incomplete result completed={completed} total={total}")
            label = f"{path}:{scenario}"
            (
                trace_sha,
                subset_sha,
                contract,
                contract_map_sha,
                trigger_counts,
            ) = _v2_ablation_audit(
                metadata=metadata,
                detail=detail,
                scenario=scenario,
                completed=completed,
                label=label,
            )
            metrics = {
                key: _v2_metric(detail, summary, key, label)
                for key, _, _ in V2_ABLATION_TABLE_METRICS
            }
            identity = (model, scenario, seed)
            if identity in results:
                raise SystemExit(
                    "duplicate V2 ablation identity "
                    f"(model={model!r}, scenario={scenario!r}, seed={seed}): "
                    f"{results[identity].source} and {path}"
                )
            results[identity] = V2AblationResult(
                model=model,
                scenario=scenario,
                seed=seed,
                run_tag=run_tag,
                source=path,
                total=total,
                completed=completed,
                shared_trace_sha256=trace_sha,
                shared_adapter_subset_sha256=subset_sha,
                generation_contract=contract,
                generation_contract_request_map_sha256=contract_map_sha,
                non_feature_frozen_config_sha256=str(
                    metadata.get("non_feature_frozen_config_sha256") or ""
                ).strip().lower(),
                formal_axes=_v2_ablation_formal_axes(metadata, label),
                metrics=metrics,
                triggers=trigger_counts,
            )
    if not results:
        raise SystemExit(
            "no V2 ablation scenarios found; expected one of " + ", ".join(V2_ABLATION_SCENARIOS)
        )
    models = sorted({result.model for result in results.values()})
    for model in models:
        if not any(
            result.model == model and result.scenario == "v2_elastic_only"
            for result in results.values()
        ):
            raise SystemExit(f"{model}: missing v2_elastic_only reference")
    provenance_groups: Dict[tuple[str, int], List[V2AblationResult]] = defaultdict(list)
    for result in results.values():
        provenance_groups[(result.model, result.seed)].append(result)
    for (model, seed), group in sorted(provenance_groups.items()):
        trace_hashes = {result.shared_trace_sha256 for result in group}
        subset_hashes = {result.shared_adapter_subset_sha256 for result in group}
        contracts = {result.generation_contract for result in group}
        contract_map_hashes = {
            result.generation_contract_request_map_sha256 for result in group
        }
        completions = {result.completed for result in group}
        if (
            len(trace_hashes) != 1
            or len(subset_hashes) != 1
            or len(contracts) != 1
            or len(contract_map_hashes) != 1
            or len(completions) != 1
        ):
            raise SystemExit(
                f"V2 ablation comparability failed for model={model!r}, seed={seed}: "
                f"trace_hashes={trace_hashes}, subset_hashes={subset_hashes}, "
                f"generation_contracts={contracts}, contract_map_hashes={contract_map_hashes}, "
                f"completions={completions}"
            )
    ordered = sorted(
        results.values(),
        key=lambda item: (
            item.model,
            V2_ABLATION_SCENARIOS.index(item.scenario),
            item.seed,
        ),
    )
    if formal_matrix:
        validate_v2_ablation_formal_matrix(ordered)
    return ordered


def _v2_t95(run_count: int) -> float:
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    if run_count < 2:
        return float("nan")
    return critical.get(run_count, 1.96)


def _v2_mean_ci95(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise SystemExit("V2 CI requires non-empty finite seed-level values")
    avg = float(np.mean(array))
    if len(array) == 1:
        return avg, float("nan"), float("nan")
    std = float(np.std(array, ddof=1))
    return avg, std, _v2_t95(len(array)) * std / math.sqrt(len(array))


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "model"


def plot_v2_fig9_ablation(
    inputs: Sequence[Path], out_dir: Path, *, formal_matrix: bool = False
) -> None:
    provenance_index = (
        build_formal_provenance_index(inputs) if formal_matrix else None
    )
    results = load_v2_ablation_results(inputs, formal_matrix=formal_matrix)
    if provenance_index is not None:
        validate_formal_analysis_sources(
            provenance_index,
            (
                FormalAnalysisIdentity(
                    source=result.source,
                    model=_v2_formal_model_key(result.model),
                    variant=result.scenario,
                    seed=result.seed,
                )
                for result in results
            ),
            analysis_label="A2/A3 V2 Fig. 9",
        )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty V2 Fig. 9 output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    models = sorted({result.model for result in results})

    per_seed_rows: List[Dict[str, Any]] = []
    for result in results:
        per_seed_rows.append(
            {
                "model": result.model,
                "scenario": result.scenario,
                "scenario_label": V2_ABLATION_LABELS[result.scenario],
                "seed": result.seed,
                "run_tag": result.run_tag,
                "source": str(result.source),
                "total": result.total,
                "completed": result.completed,
                "shared_trace_sha256": result.shared_trace_sha256,
                "shared_adapter_subset_sha256": result.shared_adapter_subset_sha256,
                "generation_contract": result.generation_contract,
                "generation_contract_request_map_sha256": (
                    result.generation_contract_request_map_sha256
                ),
                "non_feature_frozen_config_sha256": (
                    result.non_feature_frozen_config_sha256
                ),
                **result.formal_axes,
                **result.metrics,
                **result.triggers,
            }
        )

    absolute_rows: List[Dict[str, Any]] = []
    relative_seed_rows: List[Dict[str, Any]] = []
    relative_rows: List[Dict[str, Any]] = []
    adjacent_seed_rows: List[Dict[str, Any]] = []
    adjacent_rows: List[Dict[str, Any]] = []
    complete_summary_rows: List[Dict[str, Any]] = []
    for model in models:
        model_results = [result for result in results if result.model == model]
        result_by_scenario_seed = {
            (result.scenario, result.seed): result for result in model_results
        }
        reference_by_seed = {
            result.seed: result
            for result in model_results
            if result.scenario == "v2_elastic_only"
        }
        for scenario in V2_ABLATION_SCENARIOS:
            scenario_results = [result for result in model_results if result.scenario == scenario]
            if not scenario_results:
                continue
            for metric, _, higher_is_better in V2_ABLATION_METRICS:
                values = [result.metrics[metric] for result in scenario_results]
                avg, std, half_width = _v2_mean_ci95(values)
                absolute_rows.append(
                    {
                        "model": model,
                        "scenario": scenario,
                        "scenario_label": V2_ABLATION_LABELS[scenario],
                        "metric": metric,
                        "higher_is_better": higher_is_better,
                        "seed_count": len(values),
                        "seeds": ";".join(str(result.seed) for result in scenario_results),
                        "mean": avg,
                        "std": std,
                        "ci95_half_width": half_width,
                    }
                )
                relative_values: List[float] = []
                paired_seeds: List[int] = []
                for result in scenario_results:
                    reference = reference_by_seed.get(result.seed)
                    if reference is None:
                        continue
                    relative = _improvement_pct(
                        reference.metrics[metric],
                        result.metrics[metric],
                        higher_is_better=higher_is_better,
                    )
                    relative_values.append(relative)
                    paired_seeds.append(result.seed)
                    relative_seed_rows.append(
                        {
                            "model": model,
                            "scenario": scenario,
                            "scenario_label": V2_ABLATION_LABELS[scenario],
                            "seed": result.seed,
                            "metric": metric,
                            "reference_scenario": "v2_elastic_only",
                            "reference_value": reference.metrics[metric],
                            "value": result.metrics[metric],
                            "improvement_pct": relative,
                            "higher_is_better": higher_is_better,
                        }
                    )
                if relative_values:
                    rel_avg, rel_std, rel_half = _v2_mean_ci95(relative_values)
                    relative_rows.append(
                        {
                            "model": model,
                            "scenario": scenario,
                            "scenario_label": V2_ABLATION_LABELS[scenario],
                            "metric": metric,
                            "reference_scenario": "v2_elastic_only",
                            "paired_seed_count": len(relative_values),
                            "paired_seeds": ";".join(map(str, paired_seeds)),
                            "improvement_pct_mean": rel_avg,
                            "improvement_pct_std": rel_std,
                            "improvement_pct_ci95_half_width": rel_half,
                            "higher_is_better": higher_is_better,
                        }
                    )

        for scenario in V2_ABLATION_SCENARIOS:
            scenario_results = [
                result for result in model_results if result.scenario == scenario
            ]
            if not scenario_results:
                continue
            for field, _, _ in V2_ABLATION_TABLE_METRICS:
                values = [result.metrics[field] for result in scenario_results]
                avg, std, half = _v2_mean_ci95(values)
                complete_summary_rows.append(
                    {
                        "model": model,
                        "scenario": scenario,
                        "scenario_label": V2_ABLATION_LABELS[scenario],
                        "field_kind": "metric",
                        "field": field,
                        "seed_count": len(values),
                        "seeds": ";".join(str(result.seed) for result in scenario_results),
                        "mean": avg,
                        "std": std,
                        "ci95_half_width": half,
                    }
                )
            for field in V2_ABLATION_TRIGGER_FIELDS:
                values = [float(result.triggers[field]) for result in scenario_results]
                avg, std, half = _v2_mean_ci95(values)
                complete_summary_rows.append(
                    {
                        "model": model,
                        "scenario": scenario,
                        "scenario_label": V2_ABLATION_LABELS[scenario],
                        "field_kind": "mechanism_trigger",
                        "field": field,
                        "seed_count": len(values),
                        "seeds": ";".join(str(result.seed) for result in scenario_results),
                        "mean": avg,
                        "std": std,
                        "ci95_half_width": half,
                    }
                )

        for scenario, (mechanism, reference_scenario) in V2_ABLATION_ADJACENT_REFERENCES.items():
            for metric, _, higher_is_better in V2_ABLATION_TABLE_METRICS:
                paired_values: List[float] = []
                paired_improvements: List[float] = []
                paired_seeds: List[int] = []
                for seed in sorted({result.seed for result in model_results}):
                    value_result = result_by_scenario_seed.get((scenario, seed))
                    reference_result = result_by_scenario_seed.get(
                        (reference_scenario, seed)
                    )
                    if value_result is None or reference_result is None:
                        continue
                    value = value_result.metrics[metric]
                    reference_value = reference_result.metrics[metric]
                    difference = value - reference_value
                    improvement = _improvement_pct(
                        reference_value,
                        value,
                        higher_is_better=higher_is_better,
                    )
                    paired_values.append(difference)
                    paired_improvements.append(improvement)
                    paired_seeds.append(seed)
                    adjacent_seed_rows.append(
                        {
                            "model": model,
                            "mechanism": mechanism,
                            "scenario": scenario,
                            "reference_scenario": reference_scenario,
                            "seed": seed,
                            "metric": metric,
                            "higher_is_better": higher_is_better,
                            "reference_value": reference_value,
                            "value": value,
                            "paired_difference_value_minus_reference": difference,
                            "improvement_pct": improvement,
                        }
                    )
                if paired_values:
                    diff_avg, diff_std, diff_half = _v2_mean_ci95(paired_values)
                    imp_avg, imp_std, imp_half = _v2_mean_ci95(paired_improvements)
                    adjacent_rows.append(
                        {
                            "model": model,
                            "mechanism": mechanism,
                            "scenario": scenario,
                            "reference_scenario": reference_scenario,
                            "metric": metric,
                            "higher_is_better": higher_is_better,
                            "paired_seed_count": len(paired_values),
                            "paired_seeds": ";".join(map(str, paired_seeds)),
                            "paired_difference_mean": diff_avg,
                            "paired_difference_std": diff_std,
                            "paired_difference_ci95_half_width": diff_half,
                            "improvement_pct_mean": imp_avg,
                            "improvement_pct_std": imp_std,
                            "improvement_pct_ci95_half_width": imp_half,
                        }
                    )

    per_seed_csv = out_dir / "fig9_v2_ablation_per_seed.csv"
    complete_summary_csv = out_dir / "fig9_v2_ablation_complete_summary.csv"
    absolute_csv = out_dir / "fig9_v2_ablation_absolute_summary.csv"
    relative_seed_csv = out_dir / "fig9_v2_ablation_relative_per_seed.csv"
    relative_csv = out_dir / "fig9_v2_ablation_relative_summary.csv"
    adjacent_seed_csv = out_dir / "fig9_v2_ablation_adjacent_increment_per_seed.csv"
    adjacent_csv = out_dir / "fig9_v2_ablation_adjacent_increment_summary.csv"
    _write_csv(per_seed_csv, per_seed_rows)
    _write_csv(complete_summary_csv, complete_summary_rows)
    _write_csv(absolute_csv, absolute_rows)
    _write_csv(relative_seed_csv, relative_seed_rows)
    _write_csv(relative_csv, relative_rows)
    _write_csv(adjacent_seed_csv, adjacent_seed_rows)
    _write_csv(adjacent_csv, adjacent_rows)

    generated_pdfs: List[str] = []
    for model in models:
        scenarios = [
            scenario
            for scenario in V2_ABLATION_SCENARIOS
            if any(row["model"] == model and row["scenario"] == scenario for row in absolute_rows)
        ]
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.2), constrained_layout=True)
        for ax, (metric, axis_label, _) in zip(axes.flat, V2_ABLATION_METRICS):
            rows = [
                next(
                    row
                    for row in absolute_rows
                    if row["model"] == model and row["scenario"] == scenario and row["metric"] == metric
                )
                for scenario in scenarios
            ]
            x = np.arange(len(rows))
            means = np.asarray([float(row["mean"]) for row in rows])
            errors = np.asarray([
                0.0 if not math.isfinite(float(row["ci95_half_width"])) else float(row["ci95_half_width"])
                for row in rows
            ])
            bars = ax.bar(
                x,
                means,
                yerr=errors,
                capsize=3.0,
                color=[V2_ABLATION_COLORS[scenario] for scenario in scenarios],
                edgecolor="#444444",
                linewidth=0.4,
            )
            ax.set_xticks(x, [V2_ABLATION_LABELS[scenario] for scenario in scenarios], rotation=18, ha="right")
            ax.set_ylabel(axis_label)
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            for bar, row in zip(bars, rows):
                value = float(row["mean"])
                text = f"{value:.4g}\n(n={int(row['seed_count'])})"
                ax.annotate(text, (bar.get_x() + bar.get_width() / 2.0, bar.get_height()), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=7.2)
        fig.suptitle(f"V2 elasticity and adapter-management ablation — {model}", fontsize=10.5)
        filename = "fig9_v2_ablation.pdf" if len(models) == 1 else f"fig9_v2_ablation_{_safe_slug(model)}.pdf"
        fig.savefig(out_dir / filename, bbox_inches="tight")
        plt.close(fig)
        generated_pdfs.append(filename)

    manifest = {
        "figure": "fig9_v2_ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(Path(path).expanduser().resolve()) for path in inputs],
        "sources": sorted({str(result.source) for result in results}),
        "models": models,
        "formal_matrix": formal_matrix,
        "scenario_order": list(V2_ABLATION_SCENARIOS),
        "seed_is_statistical_unit": True,
        "ci": "two-sided 95% Student-t over independent seeds; absent for n=1",
        "absolute_metrics": [metric for metric, _, _ in V2_ABLATION_METRICS],
        "complete_table_metrics": [
            metric for metric, _, _ in V2_ABLATION_TABLE_METRICS
        ],
        "complete_table_trigger_fields": list(V2_ABLATION_TRIGGER_FIELDS),
        "relative_reference": "v2_elastic_only matched on model and seed",
        "adjacent_increment_references": {
            scenario: {"mechanism": mechanism, "reference_scenario": reference}
            for scenario, (mechanism, reference) in V2_ABLATION_ADJACENT_REFERENCES.items()
        },
        "relative_formulas": {
            "lower_is_better": "(reference - value) / reference * 100",
            "higher_is_better": "(value - reference) / reference * 100",
        },
        "strict_checks": [
            "shared trace/subset SHA-256 values are present and identical within model/seed",
            "generation contract and recomputed request-map SHA agree for every scenario",
            "all completed LoRA requests carry a valid pre-dispatch readiness tier",
            "request-level dispatch tier counts equal the recorded activation counters",
            "feature gates exactly match the four cumulative paper-mechanism scenarios",
            "each enabled mechanism has positive activation evidence and disabled mechanisms do not leak",
            *(
                [
                    "formal A2/A3 identity set exactly matches 7B four scenarios on seed 43 plus ElasticOnly/Hierarchy/Full on seeds 44/45, and 3B ElasticOnly/Full on seeds 43/44/45",
                    "formal model identities are exactly Llama-2-7B and Llama-3.2-3B",
                    "formal axes are exactly 4000 requests, 500 adapters, 250 MiB/s, time-scale 8, legacy generation/rotation, Zipf 1, active cap 48, rotation 500, overlap 0.75",
                    "non_feature_frozen_config_sha256 is valid and invariant across all scenarios and seeds within each model",
                    "formal campaign manifests are complete, heldout, source-clean, and tied to non-empty commits",
                    "every analyzed raw JSON matches its manifest byte count and SHA-256 record",
                    "system_resolved_config_sha256 is valid and invariant across seeds within each model/scenario",
                ]
                if formal_matrix
                else []
            ),
        ],
        "pdfs": generated_pdfs,
        "csvs": [
            per_seed_csv.name,
            complete_summary_csv.name,
            absolute_csv.name,
            relative_seed_csv.name,
            relative_csv.name,
            adjacent_seed_csv.name,
            adjacent_csv.name,
        ],
    }
    (out_dir / "fig9_v2_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


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
            "cost_per_1m_output_tokens_usd": summary.get("cost_per_1m_output_tokens_usd"),
            "monetary_cost_total_usd": summary.get("monetary_cost_total_usd"),
            "monetary_active_charge_gpu_seconds": summary.get("monetary_active_charge_gpu_seconds"),
            "monetary_idle_charge_gpu_seconds": summary.get("monetary_idle_charge_gpu_seconds"),
            "infra_active_gpu_seconds": summary.get("infra_active_gpu_seconds"),
            "infra_idle_ready_gpu_seconds": summary.get("infra_idle_ready_gpu_seconds"),
            "infra_startup_gpu_seconds": summary.get("infra_startup_gpu_seconds"),
            "infra_gpu_seconds_total": summary.get("infra_gpu_seconds_total"),
            "infra_ce": summary.get("infra_ce"),
            "slo_attainment": summary.get("slo_attainment"),
            "slo_goodput_rps": summary.get("slo_goodput_rps"),
            "slo_goodput_tok_per_s": summary.get("slo_goodput_tok_per_s"),
            "goodput_requests_per_gpu_second": summary.get("goodput_requests_per_gpu_second"),
            "goodput_tokens_per_gpu_second": summary.get("goodput_tokens_per_gpu_second"),
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
        "cost_per_1m_output_tokens_usd": row.get("cost_per_1m_output_tokens_usd"),
        "monetary_cost_total_usd": row.get("monetary_cost_total_usd"),
        "monetary_active_charge_gpu_seconds": row.get("monetary_active_charge_gpu_seconds"),
        "monetary_idle_charge_gpu_seconds": row.get("monetary_idle_charge_gpu_seconds"),
        "infra_active_gpu_seconds": row.get("infra_active_gpu_seconds"),
        "infra_idle_ready_gpu_seconds": row.get("infra_idle_ready_gpu_seconds"),
        "infra_startup_gpu_seconds": row.get("infra_startup_gpu_seconds"),
        "infra_gpu_seconds_total": row.get("infra_gpu_seconds_total"),
        "infra_ce": row.get("Infra_CE") or row.get("infra_ce"),
        "slo_attainment": row.get("SLO_attainment") or row.get("slo_attainment"),
        "slo_goodput_rps": row.get("SLO_goodput_RPS") or row.get("slo_goodput_rps"),
        "slo_goodput_tok_per_s": row.get("SLO_goodput_TOKPS") or row.get("slo_goodput_tok_per_s"),
        "goodput_requests_per_gpu_second": row.get("goodput_requests_per_gpu_second"),
        "goodput_tokens_per_gpu_second": row.get("goodput_tokens_per_gpu_second"),
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
    present_keys.update(MAIN_SUMMARY_OVERRIDES.keys())
    missing = [key for key in SYSTEM_ORDER if key not in present_keys]
    if missing:
        raise SystemExit(f"{compare_path}: missing systems in strict_rows: {missing}")

    systems: List[MainSystemData] = []
    for key in SYSTEM_ORDER:
        source = MAIN_SUMMARY_OVERRIDES.get(key) or _main_summary_path(round_dir, run_tag, key)
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
            "cost_1m_output_tok_usd": _optional_float(raw.get("cost_per_1m_output_tokens_usd")),
            "monetary_cost_total_usd": _as_float(raw.get("monetary_cost_total_usd"), f"{key}.monetary_cost_total_usd"),
            "monetary_active_charge_gpu_seconds": _as_float(raw.get("monetary_active_charge_gpu_seconds"), f"{key}.monetary_active_charge_gpu_seconds"),
            "monetary_idle_charge_gpu_seconds": _as_float(raw.get("monetary_idle_charge_gpu_seconds"), f"{key}.monetary_idle_charge_gpu_seconds"),
            "infra_active_gpu_seconds": _as_float(raw.get("infra_active_gpu_seconds"), f"{key}.infra_active_gpu_seconds"),
            "infra_idle_ready_gpu_seconds": _as_float(raw.get("infra_idle_ready_gpu_seconds"), f"{key}.infra_idle_ready_gpu_seconds"),
            "infra_startup_gpu_seconds": _as_float(raw.get("infra_startup_gpu_seconds"), f"{key}.infra_startup_gpu_seconds"),
            "infra_gpu_seconds_total": _optional_float(raw.get("infra_gpu_seconds_total")),
            "infra_ce": _optional_float(raw.get("infra_ce")),
            "slo_attainment": _optional_float(raw.get("slo_attainment")),
            "slo_goodput_rps": _optional_float(raw.get("slo_goodput_rps")),
            "slo_goodput_tok_per_s": _optional_float(raw.get("slo_goodput_tok_per_s")),
            "goodput_requests_per_gpu_second": _optional_float(raw.get("goodput_requests_per_gpu_second")),
            "goodput_tokens_per_gpu_second": _optional_float(raw.get("goodput_tokens_per_gpu_second")),
            "serverless_invocation_cost_per_request_usd": _as_float(raw.get("serverless_invocation_cost_per_request_usd"), f"{key}.serverless_invocation_cost_per_request_usd"),
            "is_serverless": 1.0 if str(raw.get("monetary_pricing_runtime_class") or "").strip().lower() == "serverless" else 0.0,
        }
        invocation_total = metrics["serverless_invocation_cost_per_request_usd"] * metrics["completed"]
        active_rate = max(metrics["monetary_cost_total_usd"] - invocation_total, 0.0) / max(
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
                "gpu_cost_rate_usd_per_s": active_rate,
                "gpu_seconds_per_request": metrics["infra_gpu_seconds_total"] / metrics["completed"],
                "slo_goodput_requests_per_dollar": (
                    metrics["completed"] * metrics["slo_attainment"]
                    / metrics["monetary_cost_total_usd"]
                ),
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
    cost_musd_values = [system.metrics["cost_req_usd"] * 1000.0 for system in systems]
    ce_values = [system.metrics["ce"] for system in systems]
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

    x_span = max(cost_musd_values) - min(cost_musd_values)
    x_pad_left = max(0.08, x_span * 0.08)
    x_pad_right = max(0.18, x_span * 0.18)
    ax.set_xlim(max(0.0, min(cost_musd_values) - x_pad_left), max(cost_musd_values) + x_pad_right)
    ax.set_ylim(0, max(10.0, max(ce_values) * 1.18))
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


CE_SUPPLEMENT_REQUIRED_METRICS = (
    "completed",
    "e2e_avg_ms",
    "cost_req_usd",
    "ce",
    "infra_ce",
    "infra_gpu_seconds_total",
    "infra_startup_gpu_seconds",
    "infra_active_gpu_seconds",
    "infra_idle_ready_gpu_seconds",
    "slo_attainment",
    "slo_goodput_rps",
    "slo_goodput_tok_per_s",
    "goodput_requests_per_gpu_second",
    "goodput_tokens_per_gpu_second",
    "cost_1mtok_usd",
    "cost_1m_output_tok_usd",
    "gpu_cost_rate_usd_per_s",
)


def _require_ce_supplement_metrics(systems: Sequence[MainSystemData]) -> None:
    missing: List[str] = []
    for system in systems:
        for key in CE_SUPPLEMENT_REQUIRED_METRICS:
            value = system.metrics.get(key)
            if value is None or not math.isfinite(float(value)):
                missing.append(f"{system.key}.{key}")
    if missing:
        raise SystemExit(
            "CE supplementary analysis requires complete e2e_v3 lifecycle metrics; "
            "missing/non-finite: " + ", ".join(missing)
        )


def _cost_at_idle_factor(system: MainSystemData, idle_factor: float) -> float:
    metrics = system.metrics
    if metrics.get("is_serverless", 0.0) < 0.5:
        return metrics["cost_req_usd"]
    gpu_seconds = (
        metrics["infra_startup_gpu_seconds"]
        + metrics["infra_active_gpu_seconds"]
        + idle_factor * metrics["infra_idle_ready_gpu_seconds"]
    )
    return (
        gpu_seconds * metrics["gpu_cost_rate_usd_per_s"] / metrics["completed"]
        + metrics["cost_invocation_usd"]
    )


def _break_even_idle_factor(target: MainSystemData, reference: MainSystemData) -> float:
    """Solve L_target*C_target(f) == L_reference*C_reference(f)."""
    target_base = _cost_at_idle_factor(target, 0.0)
    reference_base = _cost_at_idle_factor(reference, 0.0)
    target_slope = _cost_at_idle_factor(target, 1.0) - target_base
    reference_slope = _cost_at_idle_factor(reference, 1.0) - reference_base
    target_latency = target.metrics["e2e_avg_ms"] / 1000.0
    reference_latency = reference.metrics["e2e_avg_ms"] / 1000.0
    denominator = target_latency * target_slope - reference_latency * reference_slope
    if abs(denominator) < 1e-15:
        return float("nan")
    return (
        reference_latency * reference_base - target_latency * target_base
    ) / denominator


def plot_ce_supplement(round_dir: Path, out_dir: Path) -> None:
    """Generate the reviewer-facing CE robustness package without changing headline CE."""
    systems = _main_round_data(round_dir)
    _require_ce_supplement_metrics(systems)
    prime = next(system for system in systems if system.key == "faaslora")

    metric_rows: List[Dict[str, Any]] = []
    for system in systems:
        metrics = system.metrics
        metric_rows.append(
            {
                "system_key": system.key,
                "system": system.label,
                "source": str(system.source),
                "avg_e2e_s": metrics["e2e_avg_ms"] / 1000.0,
                "cost_per_request_usd": metrics["cost_req_usd"],
                "monetary_ce": metrics["ce"],
                "infra_ce": metrics["infra_ce"],
                "gpu_seconds_per_request": metrics["gpu_seconds_per_request"],
                "startup_gpu_seconds": metrics["infra_startup_gpu_seconds"],
                "active_gpu_seconds": metrics["infra_active_gpu_seconds"],
                "idle_ready_gpu_seconds": metrics["infra_idle_ready_gpu_seconds"],
                "slo_attainment": metrics["slo_attainment"],
                "slo_goodput_rps": metrics["slo_goodput_rps"],
                "slo_goodput_tok_per_s": metrics["slo_goodput_tok_per_s"],
                "slo_goodput_requests_per_dollar": metrics["slo_goodput_requests_per_dollar"],
                "goodput_requests_per_gpu_second": metrics["goodput_requests_per_gpu_second"],
                "goodput_tokens_per_gpu_second": metrics["goodput_tokens_per_gpu_second"],
                "cost_per_1m_total_tokens_usd": metrics["cost_1mtok_usd"],
                "cost_per_1m_output_tokens_usd": metrics["cost_1m_output_tok_usd"],
            }
        )

    generalized_rows: List[Dict[str, Any]] = []
    for alpha in (0.5, 1.0, 2.0):
        for beta in (0.5, 1.0, 2.0):
            scored = []
            for system in systems:
                latency_s = system.metrics["e2e_avg_ms"] / 1000.0
                cost = system.metrics["cost_req_usd"]
                score = 1.0 / ((latency_s**alpha) * (cost**beta))
                scored.append((system, score))
            ranks = {
                system.key: rank
                for rank, (system, _) in enumerate(
                    sorted(scored, key=lambda item: item[1], reverse=True), start=1
                )
            }
            for system, score in scored:
                generalized_rows.append(
                    {
                        "alpha_latency": alpha,
                        "beta_cost": beta,
                        "system_key": system.key,
                        "system": system.label,
                        "generalized_ce": score,
                        "rank": ranks[system.key],
                    }
                )

    contribution_rows: List[Dict[str, Any]] = []
    for reference in systems:
        if reference.key == prime.key:
            continue
        latency_log = math.log(
            (reference.metrics["e2e_avg_ms"] / 1000.0)
            / (prime.metrics["e2e_avg_ms"] / 1000.0)
        )
        cost_log = math.log(reference.metrics["cost_req_usd"] / prime.metrics["cost_req_usd"])
        total_log = latency_log + cost_log
        denominator = abs(latency_log) + abs(cost_log)
        contribution_rows.append(
            {
                "target": prime.label,
                "reference_key": reference.key,
                "reference": reference.label,
                "latency_log_contribution": latency_log,
                "cost_log_contribution": cost_log,
                "total_log_ce_ratio": total_log,
                "observed_log_ce_ratio": math.log(prime.metrics["ce"] / reference.metrics["ce"]),
                "latency_absolute_share_pct": 100.0 * abs(latency_log) / denominator if denominator else 0.0,
                "cost_absolute_share_pct": 100.0 * abs(cost_log) / denominator if denominator else 0.0,
                "dominant_absolute_factor": "latency" if abs(latency_log) >= abs(cost_log) else "cost",
            }
        )

    idle_rows: List[Dict[str, Any]] = []
    for idle_factor in (0.0, 0.238095, 0.5, 0.75, 1.0):
        for system in systems:
            cost = _cost_at_idle_factor(system, idle_factor)
            latency_s = system.metrics["e2e_avg_ms"] / 1000.0
            idle_rows.append(
                {
                    "idle_billing_factor": idle_factor,
                    "system_key": system.key,
                    "system": system.label,
                    "runtime_class": "serverless" if system.metrics["is_serverless"] >= 0.5 else "serverful_fixed",
                    "cost_per_request_usd": cost,
                    "ce": 1.0 / (latency_s * cost),
                }
            )

    break_even_rows = []
    for reference_key in ("sglang", "serverlessllm"):
        reference = next(system for system in systems if system.key == reference_key)
        factor = _break_even_idle_factor(prime, reference)
        break_even_rows.append(
            {
                "target": prime.label,
                "reference_key": reference.key,
                "reference": reference.label,
                "break_even_idle_billing_factor": factor,
                "within_analyzed_0_to_1": math.isfinite(factor) and 0.0 <= factor <= 1.0,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "ce_supplementary_metrics.csv"
    generalized_csv = out_dir / "ce_generalized_sensitivity.csv"
    contribution_csv = out_dir / "ce_log_contribution.csv"
    idle_csv = out_dir / "ce_idle_factor_sensitivity.csv"
    break_even_csv = out_dir / "ce_idle_factor_break_even.csv"
    _write_csv(metrics_csv, metric_rows)
    _write_csv(generalized_csv, generalized_rows)
    _write_csv(contribution_csv, contribution_rows)
    _write_csv(idle_csv, idle_rows)
    _write_csv(break_even_csv, break_even_rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75), constrained_layout=True)
    for system in systems:
        axes[0].scatter(
            system.metrics["cost_req_usd"] * 1000.0,
            system.metrics["e2e_avg_ms"] / 1000.0,
            s=45 if system.key == "faaslora" else 34,
            color=SYSTEM_COLORS[system.key],
            edgecolor="#333333",
            linewidth=0.4,
            label=system.label,
        )
    axes[0].set_xlabel("Cost/req (mUSD; lower is better)")
    axes[0].set_ylabel("Average E2E (s; lower is better)")
    axes[0].set_title("(a) Cost--latency Pareto view")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7.5)

    y = np.arange(len(contribution_rows))
    latency_values = [float(row["latency_log_contribution"]) for row in contribution_rows]
    cost_values = [float(row["cost_log_contribution"]) for row in contribution_rows]
    axes[1].barh(y, latency_values, color=METRIC_COLORS["e2e"], label="Latency")
    axes[1].barh(y, cost_values, left=latency_values, color=METRIC_COLORS["cost"], label="Cost")
    axes[1].axvline(0.0, color="#555555", linewidth=0.7)
    axes[1].set_yticks(y, [str(row["reference"]) for row in contribution_rows])
    axes[1].invert_yaxis()
    axes[1].set_xlabel(r"Contribution to $\ln(CE_{Prime}/CE_{reference})$")
    axes[1].set_title("(b) CE log-contribution audit")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8.0)

    pdf = out_dir / "fig_ce_supplementary.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    manifest = out_dir / "ce_supplementary_manifest.json"
    _write_manifest(
        manifest,
        "ce_supplementary",
        round_dir,
        pdf,
        metrics_csv,
        [system.source for system in systems],
        extra={
            "supplementary_csvs": [
                str(generalized_csv),
                str(contribution_csv),
                str(idle_csv),
                str(break_even_csv),
            ],
            "headline_ce_unchanged": True,
            "generalized_ce_definition": "1 / (average_E2E_seconds^alpha * cost_per_request_USD^beta)",
            "log_contribution_identity": "ln(CE_Prime/CE_ref) = ln(L_ref/L_Prime) + ln(C_ref/C_Prime)",
            "idle_factor_policy": (
                "Only systems labeled serverless are recomputed; serverful system cost remains fixed."
            ),
        },
    )


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
    "ce_supplementary": plot_ce_supplement,
    "fig7_cost": plot_fig7,
    "fig7": plot_fig7,
}

MAIN_FIGURES = ("fig1_intro", "table1_main", "fig5_normalized", "fig7_cost")
MOTIVATION_FIGURES = ("fig2_mismatch", "fig3_tier")
ABLATION_FIGURES = ("fig4_coordination", "fig6_ablation")


def plot_tc_serverless_wait_audit(inputs: Sequence[Path], out_dir: Path,
                                  *, native_polling: bool = False) -> None:
    """Historical/native development diagnosis; preserve failures, never infer CI."""
    from matplotlib import font_manager
    import subprocess
    # The long-lived plotting env may cache its font list before installation.
    # Register installed TNR faces explicitly; never substitute another family.
    font_paths = subprocess.check_output(
        ['fc-list', '-f', '%{file}\n', ':family=Times New Roman'], text=True).splitlines()
    for path in font_paths:
        if font_manager.FontProperties(fname=path).get_name() == 'Times New Roman':
            font_manager.fontManager.addfont(path)
    font = font_manager.findfont('Times New Roman', fallback_to_default=False)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit('audit output must be new/empty; no figure overwrite')
    audits = [_load_json(p) for p in inputs]
    labels = {'llama2_7b': '7B', 'llama32_3b': '3B'}
    if native_polling:
        if (not 1 <= len(audits) <= 2 or len({a.get('model_profile') for a in audits}) != 1
                or audits[0].get('model_profile') not in labels
                or len({a.get('variant') for a in audits}) != len(audits)
                or not {a.get('variant') for a in audits} <= {'original', 'repaired'}):
            raise SystemExit('expected one model, one observation per polling variant')
        if (len({a.get('http_config_sha256') for a in audits}) != 1
                or len({a.get('deployment_sha256') for a in audits}) != 1):
            raise SystemExit('paired polling configuration differs')
        for a in audits:
            if (a.get('kind') != 'ieee_tc_serverless_polling_diagnostic'
                    or not a.get('measurement_complete') or a['counts']['N_plan'] != 1000):
                raise SystemExit('incomplete/wrong native polling diagnostic')
            for path, digest in (('http_config_path', 'http_config_sha256'),
                                 ('deployment_path', 'deployment_sha256')):
                if hashlib.sha256(Path(a[path]).read_bytes()).hexdigest() != a[digest]:
                    raise SystemExit('native diagnostic configuration SHA differs')
            # Adapt to the existing diagnostic renderer without dropping failed
            # offered rows from the source or the accompanying summary table.
            valid = [r for r in a['diagnostic_rows'] if r['status'] == 'http_response']
            if not valid:
                raise SystemExit('no valid timing population; publish a failure table instead')
            t0 = min(r['planned_arrival_s'] for r in a['diagnostic_rows'])
            a['diagnostic_rows'] = [dict(arrival_time_s=r['planned_arrival_s']-t0,
                dispatch_wait_s=(r['submit_lag_ms']+r['dispatch_wait_after_submit_ms'])/1000)
                for r in valid]
            a['backend_gaps_s'] = a['assignment_gaps_s']
            a['means'] = {k: v['mean'] for k,v in a['conditional_metrics'].items()}
            a['backend_gap_seconds'] = a['assignment_gap_stats']
            a['requests'] = a['counts']['N_plan']
        labels = {'original': 'Original', 'repaired': 'Repaired'}
    elif len(audits) != 2 or {a.get('model_profile') for a in audits} != set(labels):
        raise SystemExit('expected one clean historical audit per model')
    for a in audits:
        if not native_polling and (a['kind'] != 'historical_serverless_dispatch_audit' or a['requests'] != 4000):
            raise SystemExit('wrong audit type or incomplete historical run')
        if hashlib.sha256(Path(a['replay_path']).read_bytes()).hexdigest() != a['replay_sha256']:
            raise SystemExit('historical source SHA mismatch')
    out_dir.mkdir(parents=True, exist_ok=True)
    style = {'font.family': 'serif', 'font.serif': ['Times New Roman'],
             'axes.labelsize': 10.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
             'legend.fontsize': 9.5, 'savefig.bbox': None, 'pdf.fonttype': 42}
    qa = []
    with plt.rc_context(style):
        for kind, caption in [('queue', '(a) Serverless: accumulated wait'),
                              ('cadence', '(b) Serverless: backend cadence')]:
            if native_polling and kind == 'cadence':
                caption = '(b) Serverless: assignment cadence'
            height = 2.85 if native_polling else 2.65
            fig, ax = plt.subplots(figsize=(3.45, height))
            fig.subplots_adjust(left=.19, right=.975, bottom=.29,
                                top=.80 if native_polling else .84)
            for a, color, linestyle in zip(audits, ['#0072B2', '#D55E00'], ['-', '--']):
                label = labels[a['variant'] if native_polling else a['model_profile']]
                if kind == 'queue':
                    rows = sorted(a['diagnostic_rows'], key=lambda r:r['arrival_time_s'])
                    ax.plot([r['arrival_time_s'] for r in rows],
                            [r['dispatch_wait_s'] for r in rows],
                            color=color, linestyle=linestyle, lw=1.4, label=label)
                else:
                    _plot_ecdf(ax, a['backend_gaps_s'], color=color, label=label)
                    ax.lines[-1].set_linestyle(linestyle)
            _style_axes(ax)
            if kind == 'queue':
                ax.set_xlabel('Scheduled arrival (s)', labelpad=2)
                ax.set_ylabel('Dispatch wait (s)', labelpad=2)
                ax.set_ylim(bottom=0)
            else:
                ax.set_xscale('log')
                if native_polling and any(gap < .5 for a in audits for gap in a['backend_gaps_s']):
                    # Ready-first allocation may span milliseconds to seconds.
                    # Label every visible decade instead of leaving the entire
                    # subsecond region unlabeled by the historical fixed ticks.
                    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10))
                    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f'{x:g}'))
                else:
                    ax.set_xticks([.5, 1, 2, 5], ['0.5', '1', '2', '5'])
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax.axvline(1, color='#666666', linestyle=':', lw=1)
                ax.set_xlabel('Assignment gap (s, log)' if native_polling else 'Backend-start gap (s, log)', labelpad=2)
                ax.set_ylabel('Cumulative fraction', labelpad=2)
                ax.set_ylim(0, 1.02)
            ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2,
                      frameon=False, borderaxespad=.1, handlelength=1.6,
                      columnspacing=1.4)
            note = ('Development; 1 run/variant\n'
                    + '; '.join(f"{labels[a['variant']]}: {a['counts']['N_failed']} failed/1000" for a in audits)
                    if native_polling else 'Historical replay; 1 run/model')
            fig.text(.58, .975, note, ha='center', va='top', fontsize=9)
            fig.text(.19 + (.975-.19)/2, .04, caption, ha='center',
                     weight='bold', fontsize=10.5)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            text = [ax.xaxis.label, ax.yaxis.label, *fig.texts]
            boxes = [t.get_window_extent(renderer) for t in text]
            if any(not fig.bbox.contains(b.x0,b.y0) or not fig.bbox.contains(b.x1,b.y1) for b in boxes):
                raise SystemExit('clipped figure label')
            legend_box = ax.get_legend().get_window_extent(renderer)
            if any(t.get_window_extent(renderer).overlaps(legend_box) for t in fig.texts):
                raise SystemExit('diagnostic note/subtitle overlaps legend')
            qa.append({'figure': kind, 'width_inches': 3.45,
                       'height_inches': height, 'label_clipping': False,
                       'note_legend_overlap': False,
                       'font_path': font, 'manual_visual_review': 'required'})
            stem = out_dir/f'serverless_{"polling" if native_polling else "historical"}_{kind}'
            fig.savefig(stem.with_suffix('.pdf'), bbox_inches=None)
            fig.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches=None)
            plt.close(fig)
    table = [{'model':a['model_profile'] if native_polling else labels[a['model_profile']],
              **({'variant': labels[a['variant']]} if native_polling else {}),
              'requests':a['requests'], **(a['counts'] if native_polling else {}),
              **a['means'], **{'gap_'+k:v for k,v in a['backend_gap_seconds'].items()}}
             for a in audits]
    _write_csv(out_dir/f'serverless_{"polling" if native_polling else "historical"}_summary.csv', table)
    manifest = {'kind':'native_polling_development' if native_polling else 'historical_diagnostic_not_formal_comparison',
        **({'runs_per_variant':1} if native_polling else {'runs_per_model':1}),
        'ci':None, 'display_name':'Serverless',
        'interpretation':('Conditional valid-response timing; all failures retained in table. Not formal/remote/numerical LoRA evidence.'
                          if native_polling else 'Queue/cadence evidence only; no measured repaired-model latency yet'),
        'sources':[{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'qa':qa,
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out_dir.iterdir()) if p.is_file()}}
    (out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def tc_native_source_rows(payload: dict, *, expected_purpose: str =
        'representative_static_content_classes_development_profile_not_formal_S1') -> List[dict]:
    """Strict descriptive export; initialization observations are not run repeats."""
    if (payload.get('kind') != 'backend_native_native_source_matrix_qualification_v1'
            or payload.get('pass') is not True or payload.get('stage') != 'complete'
            or payload.get('shutdown_called') is not True
            or payload.get('profile_workspaces_removed') is not True
            or payload.get('artifact_mode') != 'prepublished_gzip_v1_real_remote_no_fallback'):
        raise ValueError('source profile is incomplete or has the wrong delivery contract')
    spec = payload['source_profile_spec']
    if not isinstance(expected_purpose, str) or not expected_purpose.strip() or spec['purpose'] != expected_purpose:
        raise ValueError('source plot purpose differs from the explicitly selected analysis identity')
    expected = {}
    for i, wave in enumerate(spec['waves']):
        if wave['role'] not in ('kernel_warmup_retained', 'representative_measurement'):
            raise ValueError('source wave has an unknown analysis role')
        for lane, selected in enumerate(wave['requests']):
            expected[f'source-profile/w{i}/l{lane}'] = (wave, selected)
    requests = payload['requests']
    if (len(requests) != len(expected)
            or {q['request_id'] for q in requests} != set(expected)
            or len(payload['profile_waves']) != len(spec['waves'])
            or not all(w['complete'] for w in payload['profile_waves'])):
        raise ValueError('source profile does not complete every predeclared wave/request')
    rows = []
    for q in requests:
        wave, selected = expected[q['request_id']]
        t, features = q['timing'], q['class_features']
        if (q.get('pass') is not True or q.get('reservation_released') is not True
                or q['requested_source'] != wave['source']
                or any(q[k] != selected[k] for k in ('source_request_id', 'adapter_id'))
                or q['target_tokens'] != q['actual_tokens']
                or t['native_output_tokens'] != q['actual_tokens']
                or t['native_terminal_observed'] is not True
                or len(q['native_events']) != 2
                or q['native_events'][-1]['token_count'] != q['actual_tokens']
                or q['admission_clock_id'] != t['native_clock_id']
                or q['native_clock_id'] != t['native_clock_id']
                or q['source_evidence']['state'] != 'released'):
            raise ValueError('source request contract or ownership failed')
        a, b, f, z = [q[k] for k in ('admitted_monotonic_s', 'acquired_monotonic_s',
                                     'first_token_monotonic_s', 'last_token_monotonic_s')]
        if not (all(math.isfinite(x) for x in (a,b,f,z)) and 0 < a <= b <= f <= z):
            raise ValueError('invalid measured D/T/O ordering')
        if (f != t['native_first_token_monotonic_s']
                or z != t['native_last_token_monotonic_s']):
            raise ValueError('native endpoints disagree with observed service intervals')
        d, pre, dec = (b-a)*1000, (f-b)*1000, (z-f)*1000
        n = q['actual_tokens']
        checked_times = [t[k] for k in ('native_decode_ms', 'native_dispatch_monotonic_s',
                         'worker_wall_e2e_ms', 'worker_completion_notification_ms')]
        if n > 1:
            checked_times.append(t['native_tpot_ms'])
        if (type(n) is not int or n < 1
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in checked_times)):
            raise ValueError('source native metrics must be finite and have positive integer output')
        tpot = dec/(n-1) if n > 1 else None
        if (abs(dec-t['native_decode_ms']) > 1
                or (n > 1 and abs(tpot-t['native_tpot_ms']) > 1)
                or abs(t['worker_wall_e2e_ms'] - ((f-t['native_dispatch_monotonic_s'])*1000
                       + dec + t['worker_completion_notification_ms'])) > 1):
            raise ValueError('native timing reconstruction exceeds 1 ms')
        rows.append(dict(request_id=q['request_id'], source=wave['source'], role=wave['role'],
            round=wave['round'], adapter_id=q['adapter_id'], source_request_id=q['source_request_id'],
            rank=features['adapter_rank'], footprint_bytes=features['footprint_bytes'],
            admitted_after_accept=features['admitted_after_accept'],
            actual_tokens=n, prompt_sha256=q['prompt_sha256'],
            native_prompt_sha256=t['native_prompt_token_ids_sha256'],
            output_sha256=t['completion_token_ids_sha256'],
            D_ms=d, T_ms=pre, O_ms=dec, TPOT_ms=tpot))
    return rows


def _tc_drawn_tick_labels(ax) -> list:
    """Matplotlib locators can retain labels outside the displayed view interval.

    Axis.draw omits those ticks. QA must check the labels that are actually drawn,
    not reject an invisible next tick above the data-dependent upper limit.
    """
    labels = []
    for positions, texts, limits in (
            (ax.get_xticks(), ax.get_xticklabels(), ax.get_xlim()),
            (ax.get_yticks(), ax.get_yticklabels(), ax.get_ylim())):
        low, high = sorted(limits)
        labels.extend(text for position, text in zip(positions, texts)
                      if low <= position <= high and text.get_visible())
    return labels


def plot_tc_native_source_profile(inputs: Sequence[Path], out_dir: Path, *, expected_purpose: str =
        'representative_static_content_classes_development_profile_not_formal_S1') -> None:
    """Single-run profile preview, not S1 or a causal system-ranking figure."""
    from matplotlib import font_manager
    import subprocess
    if len(inputs) != 1:
        raise ValueError('source preview requires exactly one completed model run')
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError('source preview output must be new/empty')
    path = inputs[0]
    raw = _load_json(path)
    rows = tc_native_source_rows(raw, expected_purpose=expected_purpose)
    launch_path = path.with_name(path.stem + '_launch.json')
    launch = _load_json(launch_path)
    if (launch.get('pass') is not True or launch['service_returncode'] != 0
            or launch['watchdog_returncode'] != 0
            or not launch['native_gpu_context_release_confirmed']
            or not launch['service_path_removed']):
        raise ValueError('source run did not finish actual service cleanup')
    for p in subprocess.check_output(
            ['fc-list', '-f', '%{file}\n', ':family=Times New Roman'], text=True).splitlines():
        if font_manager.FontProperties(fname=p).get_name() == 'Times New Roman':
            font_manager.fontManager.addfont(p)
    font = font_manager.findfont('Times New Roman', fallback_to_default=False)
    order = ['remote', 'nvme', 'file_host', 'native_host', 'gpu']
    labels = ['Remote', 'NVMe', 'HOST\nfile', 'HOST\ntensor', 'GPU']
    colors = ['#D55E00', '#0072B2', '#009E73', '#CC79A7', '#E69F00']
    selected = [r for r in rows if r['role'] == 'representative_measurement']
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir/'all_observations.csv', rows)
    summary = []
    for source in order:
        group = [r for r in selected if r['source'] == source]
        if not group:
            raise ValueError('representative source category missing')
        values = {key: [r[key] for r in group if r[key] is not None]
                  for key in ('D_ms', 'T_ms', 'O_ms', 'TPOT_ms')}
        summary.append(dict(source=source, requests=len(group),
            adapters=len({r['adapter_id'] for r in group}),
            rounds=len({r['round'] for r in group}),
            **{f'{k}_mean': math.fsum(v)/len(v) if v else None for k,v in values.items()},
            **{f'{k}_p95_type1': sorted(v)[math.ceil(.95*len(v))-1] if v else None
               for k,v in values.items()}))
    _write_csv(out_dir/'summary.csv', summary)
    style = {'font.family':'serif', 'font.serif':['Times New Roman'],
             'axes.labelsize':10.5, 'xtick.labelsize':9.5, 'ytick.labelsize':9.5,
             'pdf.fonttype':42, 'savefig.bbox':None}
    qa = []
    with plt.rc_context(style):
        for key, ylabel, caption in (
                ('D_ms', 'Preparation D (ms, symlog)', '(a) Preparation after admission'),
                ('T_ms', 'Acquired-to-first T (ms)', '(b) Acquired-to-first token'),
                ('TPOT_ms', 'TPOT (ms/token)', '(c) Native decode spacing')):
            fig, ax = plt.subplots(figsize=(3.45, 2.65))
            fig.subplots_adjust(left=.21, right=.975, bottom=.25, top=.85)
            for i, (source, color) in enumerate(zip(order, colors)):
                values = [r[key] for r in selected if r['source'] == source and r[key] is not None]
                if not values:
                    raise ValueError('source metric has no observed eligible values')
                # Deterministic horizontal separation is visual jitter only.
                x = i + np.linspace(-.15, .15, len(values))
                ax.scatter(x, values, s=9, color=color, alpha=.65, linewidths=0, zorder=3)
                ax.plot([i-.23,i+.23], [np.median(values)]*2, color='#202020', lw=1.4, zorder=4)
            ax.set_xticks(range(5), labels)
            ax.set_xlim(-.5,4.5)
            ax.set_ylabel(ylabel, labelpad=2)
            if key == 'D_ms':
                ax.set_yscale('symlog', linthresh=1)
                ticks=[0,1,10,100,1000,10000]
                ax.set_yticks([v for v in ticks if v <= ax.get_ylim()[1]])
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_:f'{v:g}'))
            ax.set_ylim(bottom=0)
            _style_axes(ax)
            fig.text(.59,.965,'Development profile; one run\nPoints: requests; bars: medians',
                     ha='center', va='top', fontsize=9)
            fig.text(.5925,.035,caption,ha='center',weight='bold',fontsize=10.5)
            fig.canvas.draw()
            renderer=fig.canvas.get_renderer()
            texts=[ax.yaxis.label,*_tc_drawn_tick_labels(ax),*fig.texts]
            boxes=[t.get_window_extent(renderer) for t in texts if t.get_text()]
            if any(not fig.bbox.contains(b.x0,b.y0) or not fig.bbox.contains(b.x1,b.y1) for b in boxes):
                raise ValueError('source preview text is clipped')
            if any(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:]):
                raise ValueError('source preview text overlaps')
            stem=out_dir/('source_'+key)
            fig.savefig(stem.with_suffix('.pdf'),bbox_inches=None)
            fig.savefig(stem.with_suffix('.png'),dpi=300,bbox_inches=None)
            plt.close(fig)
            qa.append(dict(metric=key,width_inches=3.45,height_inches=2.65,
                           text_clipping=False,text_overlap=False,font_path=font,
                           manual_visual_review='required'))
    manifest=dict(kind='development_native_source_profile_preview_v1',formal_S1=False,
        source_profile_purpose=raw['source_profile_spec']['purpose'],expected_purpose=expected_purpose,
        independent_runs=1,ci=None,warmup_retained_but_not_in_means=True,
        warmup_requests=len(rows)-len(selected),measured_requests=len(selected),
        limitations=['not Full qualification or a causal system comparison',
                     'within-run rounds are not independent experimental repeats',
                     'NVMe source may be page-cache resident',
                     'variable prompt/token/concurrency composition; see all_observations.csv'],
        sources=[dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in (path,launch_path)],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qa=qa,
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
               for p in out_dir.iterdir() if p.is_file()})
    (out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PrimeLoRA paper figures from result JSONs.")
    parser.add_argument("--round-dir", type=Path, help="Completed legacy round directory.")
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        default=[],
        help=(
            "V2 Fig. 9 result JSON, MANIFEST.json, or round/campaign root. "
            "Repeat for multiple seeds."
        ),
    )
    parser.add_argument(
        "--figure",
        default="all",
        help=(
            "Figure name, main_all, motivation_all, ablation_all, all, or "
            f"fig9_v2_ablation. Choices: {', '.join(PLOTTERS)}"
        ),
    )
    parser.add_argument(
        "--system-summary-override",
        action="append",
        default=[],
        help=(
            "Override one main-table system summary. Use SYSTEM_KEY:SUMMARY_JSON, "
            "or MODEL_KEY:SYSTEM_KEY:SUMMARY_JSON for compatibility with the combined main builder."
        ),
    )
    parser.add_argument(
        "--out-dir",
        "--output-dir",
        dest="out_dir",
        required=True,
        type=Path,
        help="Explicit output directory; no legacy figure directory is selected implicitly.",
    )
    parser.add_argument(
        "--formal-matrix",
        action="store_true",
        help=(
            "For V2 Fig. 9, require the exact held-out model/scenario/seed "
            "matrix before generating any artifact."
        ),
    )
    parser.add_argument('--source-profile-purpose',
        default='representative_static_content_classes_development_profile_not_formal_S1',
        help='Exact declared purpose for source-profile analysis; does not grant formal qualification.')
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    if args.figure == 'tc_native_source_profile':
        plot_tc_native_source_profile(args.input, out_dir, expected_purpose=args.source_profile_purpose)
        return
    if args.figure == 'tc_serverless_wait_audit':
        plot_tc_serverless_wait_audit(args.input, out_dir)
        return
    if args.figure == 'tc_serverless_polling_audit':
        plot_tc_serverless_wait_audit(args.input, out_dir, native_polling=True)
        return
    if args.figure in {"fig9_v2_ablation", "v2_fig9", "v2_ablation"}:
        inputs = list(args.input)
        if args.round_dir is not None:
            inputs.append(args.round_dir)
        if not inputs:
            raise SystemExit("fig9_v2_ablation requires at least one --input or --round-dir")
        if args.system_summary_override:
            raise SystemExit("--system-summary-override is not valid for fig9_v2_ablation")
        plot_v2_fig9_ablation(
            inputs,
            out_dir,
            formal_matrix=args.formal_matrix,
        )
        print(f"generated fig9_v2_ablation -> {out_dir}")
        return

    if args.input:
        raise SystemExit("--input is only valid with --figure fig9_v2_ablation")
    if args.formal_matrix:
        raise SystemExit("--formal-matrix is only valid with --figure fig9_v2_ablation")
    if args.round_dir is None:
        raise SystemExit("--round-dir is required for legacy paper figures")
    round_dir = args.round_dir.resolve()
    _require_file(round_dir / "MANIFEST.json")
    MAIN_SUMMARY_OVERRIDES.clear()
    for spec in args.system_summary_override:
        parts = spec.split(":", 2)
        if len(parts) == 2:
            system_key, path_text = parts
        elif len(parts) == 3:
            _, system_key, path_text = parts
        else:
            raise SystemExit(
                "--system-summary-override must use SYSTEM_KEY:SUMMARY_JSON "
                "or MODEL_KEY:SYSTEM_KEY:SUMMARY_JSON"
            )
        if system_key not in SYSTEM_ORDER:
            raise SystemExit(f"{spec}: unknown system key {system_key!r}")
        path = Path(path_text).expanduser().resolve()
        _require_file(path)
        MAIN_SUMMARY_OVERRIDES[system_key] = path

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
