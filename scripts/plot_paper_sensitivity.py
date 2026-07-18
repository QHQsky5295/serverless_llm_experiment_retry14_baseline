#!/usr/bin/env python3
"""Generate PrimeLoRA sensitivity figures from multiple completed rounds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_paper_figures import (
    LEGEND_FONTSIZE,
    MainSystemData,
    SYSTEM_COLORS,
    SYSTEM_LABELS,
    SYSTEM_ORDER,
    TICK_FONTSIZE,
    _as_float,
    _load_json,
    _main_round_data,
    _main_row_from_summary,
    _main_summary_path,
    _row_dict,
    _style_axes,
    _style_xgrid_axes,
    _system_key,
    _v2_mean_ci95,
    _v2_model_identity,
    _v2_read_json,
    _v2_result_candidates,
    _v2_seed,
    _write_csv,
    _xlabel_with_panel,
)

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

SENSITIVITY_SYSTEM_LABELS = {
    **SYSTEM_LABELS,
    "elastic_only": "ElasticOnly",
}
SENSITIVITY_SYSTEM_COLORS = {
    **SYSTEM_COLORS,
    "elastic_only": "#A8A8A8",
}
SENSITIVITY_SYSTEM_MARKERS = {
    **ADAPTER_POOL_MARKERS,
    "elastic_only": "P",
}

FORMAL_SEEDS = (43, 44, 45)
FORMAL_BANDWIDTH_KEYS = (
    "mib_s=11.920900",
    "mib_s=29.802300",
    "mib_s=59.604600",
    "mib_s=119.209300",
    "mib_s=250.000000",
    "no-delay",
)
FORMAL_BANDWIDTH_REPLICATED_KEYS = (
    "mib_s=11.920900",
    "mib_s=119.209300",
    "no-delay",
)
FORMAL_WORKLOAD_PROFILES = (
    "stationary_zipf1",
    "abrupt_rotation100_zipf1",
    "abrupt_rotation500_zipf1",
    "abrupt_rotation2000_zipf1",
    "abrupt_rotation500_zipf0.6",
    "abrupt_rotation500_zipf1.4",
    "gradual_rotation500_zipf1_overlap0.5",
)
FORMAL_WORKLOAD_REPLICATED_PROFILES = (
    "stationary_zipf1",
    "abrupt_rotation100_zipf1",
    "abrupt_rotation500_zipf1",
)


@dataclass
class V2SensitivityObservation:
    model: str
    profile_id: str
    profile_name: str
    profile: Dict[str, Any]
    seed: int
    system_key: str
    scenario: str
    run_tag: str
    source: Path
    completed: int
    total: int
    requests: List[Dict[str, Any]]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    manifest: Dict[str, Any]


@dataclass(frozen=True)
class BandwidthAudit:
    bandwidth_key: str
    bandwidth_label: str
    raw_limit_mode: str
    limit_mode: str
    configured_mib_s: float | None
    configured_gbit_s: float | None
    transfer_count: int
    total_bytes: int
    reservation_span_s: float
    total_injected_wait_s: float
    achieved_reserved_mib_s: float
    trace_sha256: str
    adapter_subset_sha256: str


def _formal_sensitivity_model_key(model: str) -> str:
    canonical = _canonical_model(model)
    if canonical == "llama2_7b":
        return "7b"
    if canonical == "llama32_3b":
        return "3b"
    raise SystemExit(
        f"formal sensitivity matrix contains unsupported model identity {model!r}; "
        "expected only Llama-2-7B and Llama-3.2-3B"
    )


def _validate_formal_system_scenario(
    observation: V2SensitivityObservation,
) -> None:
    if observation.system_key == "faaslora" and observation.scenario != "v2_full":
        raise SystemExit(
            f"formal matrix requires PrimeLoRA scenario='v2_full', observed "
            f"{observation.scenario!r} in {observation.source}"
        )
    if (
        observation.system_key == "elastic_only"
        and observation.scenario != "v2_elastic_only"
    ):
        raise SystemExit(
            f"formal matrix requires ElasticOnly scenario='v2_elastic_only', observed "
            f"{observation.scenario!r} in {observation.source}"
        )
    if observation.system_key == "serverlessllm" and "serverlessllm" not in (
        observation.scenario.lower().replace("_", "")
    ):
        raise SystemExit(
            f"formal matrix requires the ServerlessLLM-new path, observed scenario "
            f"{observation.scenario!r} in {observation.source}"
        )


def _format_formal_identity(identity: tuple[Any, ...]) -> str:
    return "(" + ",".join(str(item) for item in identity) + ")"


def _raise_formal_identity_mismatch(
    analysis: str,
    expected: set[tuple[Any, ...]],
    observed_counts: Mapping[tuple[Any, ...], int],
) -> None:
    observed = set(observed_counts)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    duplicates = sorted(
        identity for identity, count in observed_counts.items() if count != 1
    )
    if not (missing or extra or duplicates):
        return
    parts = [f"formal {analysis} matrix identity mismatch"]
    if missing:
        parts.append(
            "missing=["
            + "; ".join(_format_formal_identity(item) for item in missing)
            + "]"
        )
    if extra:
        parts.append(
            "extra=["
            + "; ".join(_format_formal_identity(item) for item in extra)
            + "]"
        )
    if duplicates:
        parts.append(
            "duplicate=["
            + "; ".join(
                f"{_format_formal_identity(item)} x{observed_counts[item]}"
                for item in duplicates
            )
            + "]"
        )
    raise SystemExit("; ".join(parts))


def _dig(mapping: Mapping[str, Any] | None, *path: str) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _canonical_model(raw: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", raw.lower())
    if "llama" in normalized and "32" in normalized and "3b" in normalized:
        return "llama32_3b"
    if "llama" in normalized and "2" in normalized and "7b" in normalized:
        return "llama2_7b"
    if "llama" in normalized and "2" in normalized and "13b" in normalized:
        return "llama2_13b"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_") or "model"


def _manifest_contexts(inputs: Sequence[Path]) -> tuple[List[Path], Dict[Path, Dict[str, Any]]]:
    candidates = set(_v2_result_candidates(inputs))
    contexts: Dict[Path, Dict[str, Any]] = {}
    for raw_input in inputs:
        root = raw_input.expanduser().resolve()
        manifests = [root] if root.is_file() and root.name == "MANIFEST.json" else []
        if root.is_dir():
            manifests.extend(root.rglob("MANIFEST.json"))
            candidates.update(path.resolve() for path in root.rglob("*_summary.json"))
            candidates.update(path.resolve() for path in root.rglob("*_summary.json.gz"))
        for manifest_path in manifests:
            try:
                payload = _v2_read_json(manifest_path)
            except SystemExit:
                continue
            contexts[manifest_path.parent.resolve()] = payload
    return sorted(candidates), contexts


def _context_for(path: Path, contexts: Mapping[Path, Dict[str, Any]]) -> Dict[str, Any]:
    matching = [root for root in contexts if root == path.parent or root in path.parents]
    if not matching:
        return {}
    root = max(matching, key=lambda item: len(item.parts))
    return contexts[root]


def _load_profile_from_trace(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    raw_path = _first_value(
        metadata.get("shared_trace_path"),
        metadata.get("trace_source"),
        _dig(metadata, "sampling_stats", "shared_trace_path"),
    )
    if not raw_path:
        return {}
    path = Path(str(raw_path)).expanduser()
    if not path.is_file():
        return {}
    try:
        payload = _v2_read_json(path)
    except SystemExit:
        return {}
    candidates = (
        payload.get("load_profile"),
        _dig(payload, "metadata", "load_profile"),
        _dig(payload, "metadata", "shared_trace_metadata", "load_profile"),
    )
    for candidate in candidates:
        if isinstance(candidate, dict):
            profile = dict(candidate)
            if payload.get("selected_num_adapters") is not None:
                profile.setdefault("num_adapters", payload.get("selected_num_adapters"))
            if payload.get("workload_profile"):
                profile.setdefault("profile_id", payload.get("workload_profile"))
            return profile
    return {}


def _workload_profile(metadata: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[str, str, Dict[str, Any]]:
    profile_candidates = (
        _dig(metadata, "sampling_stats", "shared_trace_metadata", "load_profile"),
        _dig(metadata, "shared_trace_metadata", "load_profile"),
        metadata.get("load_profile"),
        _dig(manifest, "shared_trace_metadata", "load_profile"),
        manifest.get("load_profile"),
        _load_profile_from_trace(metadata),
    )
    profile = next((dict(value) for value in profile_candidates if isinstance(value, dict) and value), {})
    rotation = int(
        _first_value(
            profile.get("hotset_rotation_requests"),
            profile.get("rotation_requests"),
            metadata.get("hotset_rotation_requests"),
            manifest.get("hotset_rotation_requests"),
            0,
        )
    )
    zipf = float(
        _first_value(
            profile.get("zipf_exponent"),
            metadata.get("zipf_exponent"),
            manifest.get("zipf_exponent"),
            1.0,
        )
    )
    overlap = float(
        _first_value(
            profile.get("hotset_overlap_fraction"),
            profile.get("rotation_overlap_fraction"),
            metadata.get("hotset_overlap_fraction"),
            manifest.get("hotset_overlap_fraction"),
            0.0,
        )
    )
    mode = str(
        _first_value(
            profile.get("rotation_mode"),
            profile.get("hotset_rotation_mode"),
            metadata.get("rotation_mode"),
            metadata.get("hotset_rotation_mode"),
            manifest.get("rotation_mode"),
            manifest.get("hotset_rotation_mode"),
            "stationary" if rotation <= 0 else "abrupt",
        )
    ).strip().lower()
    active_cap = int(
        _first_value(
            profile.get("active_adapter_cap"),
            metadata.get("active_adapter_cap"),
            manifest.get("active_adapter_cap"),
            0,
        )
    )
    nominal_pool = int(
        _first_value(
            profile.get("num_adapters"),
            profile.get("adapter_pool_size"),
            metadata.get("num_adapters"),
            manifest.get("num_adapters"),
            0,
        )
    )
    explicit = str(
        _first_value(
            profile.get("profile_id"),
            profile.get("name"),
            metadata.get("workload_profile_id"),
            metadata.get("workload_profile"),
            _dig(metadata, "profile_selection", "workload"),
            manifest.get("workload_profile_id"),
            manifest.get("workload_profile"),
            "workload",
        )
    )
    normalized = {
        **profile,
        "rotation_mode": mode,
        "hotset_rotation_requests": rotation,
        "zipf_exponent": zipf,
        "hotset_overlap_fraction": overlap,
        "active_adapter_cap": active_cap,
        "nominal_adapter_pool_size": nominal_pool,
    }
    profile_id = (
        f"mode={mode}|rot={rotation}|zipf={zipf:g}|overlap={overlap:g}|"
        f"active={active_cap}|pool={nominal_pool}"
    )
    display_name = explicit if explicit != "workload" else profile_id.replace("|", ", ")
    return profile_id, display_name, normalized


def _formal_workload_profile_key(observation: V2SensitivityObservation) -> str:
    profile = observation.profile
    mode = str(profile.get("rotation_mode") or "").strip().lower()
    rotation = int(profile.get("hotset_rotation_requests") or 0)
    zipf = float(profile.get("zipf_exponent") or 0.0)
    overlap = float(profile.get("hotset_overlap_fraction") or 0.0)
    nominal_pool = int(profile.get("nominal_adapter_pool_size") or 0)
    if nominal_pool != 500:
        raise SystemExit(
            f"formal C4 profile in {observation.source} must declare a 500-adapter "
            f"universe, observed {nominal_pool}"
        )

    def close(value: float, expected: float) -> bool:
        return math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-9)

    # ``stationary`` semantically disables rotation in the generator even if a
    # legacy interval remains recorded in the frozen profile.
    if mode == "stationary" and close(zipf, 1.0) and close(overlap, 0.0):
        return "stationary_zipf1"
    if mode == "abrupt" and close(overlap, 0.0):
        if rotation == 100 and close(zipf, 1.0):
            return "abrupt_rotation100_zipf1"
        if rotation == 500 and close(zipf, 1.0):
            return "abrupt_rotation500_zipf1"
        if rotation == 2000 and close(zipf, 1.0):
            return "abrupt_rotation2000_zipf1"
        if rotation == 500 and close(zipf, 0.6):
            return "abrupt_rotation500_zipf0.6"
        if rotation == 500 and close(zipf, 1.4):
            return "abrupt_rotation500_zipf1.4"
    if (
        mode == "gradual"
        and rotation == 500
        and close(zipf, 1.0)
        and close(overlap, 0.5)
    ):
        return "gradual_rotation500_zipf1_overlap0.5"
    raise SystemExit(
        f"formal C4 matrix contains an undeclared workload profile in {observation.source}: "
        f"mode={mode!r}, rotation={rotation}, zipf={zipf:g}, overlap={overlap:g}, "
        f"pool={nominal_pool}"
    )


def _sensitivity_system_key(payload: Mapping[str, Any], scenario: str) -> str | None:
    text = " ".join(
        str(value or "")
        for value in (
            scenario,
            _dig(payload, "metadata", "system"),
            _dig(payload, "metadata", "baseline_type"),
        )
    ).lower()
    if scenario == "v2_elastic_only":
        return "elastic_only"
    if scenario in {"v2_full", "faaslora_full"}:
        return "faaslora"
    if "serverlessllm" in text or "serverless_llm" in text:
        return "serverlessllm"
    if "sglang" in text:
        return "sglang"
    if "s-lora" in text or "slora" in text:
        return "slora"
    if "vllm" in text and "faaslora" not in text and "primelora" not in text:
        return "vllm"
    return None


def _scenario_summary(payload: Mapping[str, Any], scenario: str) -> Dict[str, Any]:
    candidate = _dig(payload, "scenario_summaries", scenario)
    if isinstance(candidate, dict):
        return dict(candidate)
    rows = payload.get("comparison_table") if isinstance(payload.get("comparison_table"), list) else []
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and str(row.get("scenario") or row.get("baseline_type") or "") == scenario
    ]
    if len(matches) == 1:
        return dict(matches[0])
    if len(rows) == 1 and isinstance(rows[0], dict):
        return dict(rows[0])
    return {}


def _sensitivity_metric(detail: Mapping[str, Any], summary: Mapping[str, Any], key: str, label: str) -> float:
    aliases = {
        "ttft_avg_ms": ("avg_overall_ttft_ms", "TTFT_e2e_avg_ms", "TTFT_avg_ms"),
        "ttft_p95_ms": ("p95_overall_ttft_ms", "TTFT_e2e_P95_ms", "TTFT_P95_ms"),
        "e2e_avg_ms": ("avg_overall_e2e_ms", "E2E_e2e_avg_ms", "E2E_avg_ms"),
        "e2e_p95_ms": ("p95_overall_e2e_ms", "E2E_e2e_P95_ms", "E2E_P95_ms"),
        "tpot_avg_ms": ("avg_tpot_ms", "TPOT_avg_ms"),
        "tok_s": ("throughput_tok_per_s", "throughput_TOKPS", "Throughput_TOKPS"),
        "cost_req_usd": ("monetary_cost_per_request_usd", "Monetary_cost_per_request_usd", "avg_cost_USD"),
        "ce": ("monetary_ce", "Monetary_CE", "CE"),
    }
    for alias in aliases[key]:
        value = detail.get(alias) if detail.get(alias) is not None else summary.get(alias)
        if value is not None:
            return _as_float(value, f"{label}.{alias}")
    raise SystemExit(f"{label}: missing required metric {key}")


def _successful_lora_requests(detail: Mapping[str, Any], label: str) -> List[Dict[str, Any]]:
    requests = detail.get("requests")
    if not isinstance(requests, list) or not requests:
        raise SystemExit(f"{label}: missing request-level records")
    output: List[Dict[str, Any]] = []
    failed = 0
    for index, raw in enumerate(requests):
        if not isinstance(raw, dict):
            raise SystemExit(f"{label}: request[{index}] is not an object")
        success = raw.get("success")
        if success is None:
            success = str(raw.get("status") or "ok").lower() in {"ok", "success"}
        if not bool(success):
            failed += 1
            continue
        if str(raw.get("adapter_id") or "").strip():
            output.append(dict(raw))
    if failed:
        raise SystemExit(f"{label}: contains {failed} failed requests")
    if not output:
        raise SystemExit(f"{label}: no successful LoRA requests")
    return output


def load_v2_sensitivity_observations(inputs: Sequence[Path]) -> List[V2SensitivityObservation]:
    candidates, contexts = _manifest_contexts(inputs)
    observations: Dict[tuple[str, str, int, str, str, str], V2SensitivityObservation] = {}
    for source in candidates:
        payload = _v2_read_json(source)
        detailed = payload.get("detailed_results")
        if not isinstance(detailed, dict):
            continue
        manifest = _context_for(source, contexts)
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        model = _canonical_model(_v2_model_identity(payload, source))
        run_tag = str(metadata.get("results_tag") or metadata.get("run_tag") or payload.get("run_tag") or source.stem)
        seed = _v2_seed(payload, source, run_tag)
        profile_id, profile_name, profile = _workload_profile(metadata, manifest)
        for scenario, raw_detail in detailed.items():
            if not isinstance(raw_detail, dict):
                continue
            system_key = _sensitivity_system_key(payload, str(scenario))
            if system_key is None:
                continue
            summary = _scenario_summary(payload, str(scenario))
            label = f"{source}:{scenario}"
            total = int(_as_float(_first_value(raw_detail.get("total"), summary.get("total_requests"), summary.get("total")), f"{label}.total"))
            completed = int(_as_float(_first_value(raw_detail.get("completed"), summary.get("completed_requests"), summary.get("completed")), f"{label}.completed"))
            if total <= 0 or completed != total:
                raise SystemExit(f"{label}: incomplete completed={completed} total={total}")
            requests = _successful_lora_requests(raw_detail, label)
            if len(requests) != completed:
                raise SystemExit(
                    f"{label}: successful LoRA request records={len(requests)} but completed={completed}"
                )
            metrics = {
                key: _sensitivity_metric(raw_detail, summary, key, label)
                for key in ("ttft_avg_ms", "ttft_p95_ms", "e2e_avg_ms", "e2e_p95_ms", "tpot_avg_ms", "tok_s", "cost_req_usd", "ce")
            }
            tpot_samples = [
                float(request["tpot_ms"])
                for request in requests
                if request.get("tpot_ms") is not None and request.get("tpot_observed") is not False
            ]
            metrics["tpot_p95_ms"] = float(np.percentile(tpot_samples, 95)) if tpot_samples else float("nan")
            identity = (model, profile_id, seed, system_key, str(scenario), str(source))
            if identity in observations:
                raise SystemExit(
                    "duplicate sensitivity observation "
                    f"{identity}: {observations[identity].source} and {source}"
                )
            observations[identity] = V2SensitivityObservation(
                model=model,
                profile_id=profile_id,
                profile_name=profile_name,
                profile=profile,
                seed=seed,
                system_key=system_key,
                scenario=str(scenario),
                run_tag=run_tag,
                source=source,
                completed=completed,
                total=total,
                requests=requests,
                metrics=metrics,
                metadata=dict(metadata),
                manifest=dict(manifest),
            )
    if not observations:
        raise SystemExit("no V2 workload/bandwidth observations found in inputs")
    return sorted(
        observations.values(),
        key=lambda item: (item.model, item.profile_id, item.seed, item.system_key, item.scenario),
    )


def validate_formal_workload_matrix(
    observations: Sequence[V2SensitivityObservation],
) -> None:
    """Require the exact seven-profile C4 model/system/seed matrix."""
    expected: set[tuple[Any, ...]] = {
        ("7b", profile, 43, system)
        for profile in FORMAL_WORKLOAD_PROFILES
        for system in ("faaslora", "serverlessllm")
    }
    expected.update(
        ("7b", profile, seed, system)
        for profile in FORMAL_WORKLOAD_REPLICATED_PROFILES
        for seed in (44, 45)
        for system in ("faaslora", "serverlessllm")
    )
    expected.update(
        ("7b", profile, seed, "elastic_only")
        for profile in FORMAL_WORKLOAD_REPLICATED_PROFILES
        for seed in FORMAL_SEEDS
    )

    observed_counts: DefaultDict[tuple[Any, ...], int] = defaultdict(int)
    for observation in observations:
        _validate_formal_system_scenario(observation)
        identity = (
            _formal_sensitivity_model_key(observation.model),
            _formal_workload_profile_key(observation),
            observation.seed,
            observation.system_key,
        )
        observed_counts[identity] += 1
    _raise_formal_identity_mismatch("C4", expected, observed_counts)


def _ordered_adapter_sequence(requests: Sequence[Dict[str, Any]]) -> List[str]:
    def order_key(item: tuple[int, Dict[str, Any]]) -> tuple[float, int]:
        index, request = item
        raw = _first_value(
            request.get("scheduled_arrival_offset_s"),
            request.get("arrival_time_s"),
            request.get("dispatch_offset_s"),
            index,
        )
        try:
            return float(raw), index
        except Exception:
            return float(index), index

    ordered = sorted(enumerate(requests), key=order_key)
    return [str(request.get("adapter_id") or "").strip() for _, request in ordered]


def _adapter_sequence_sha256(sequence: Sequence[str]) -> str:
    encoded = json.dumps(list(sequence), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _gini(values: Sequence[int]) -> float:
    array = np.sort(np.asarray(values, dtype=float))
    if len(array) == 0 or float(np.sum(array)) <= 0.0:
        return float("nan")
    index = np.arange(1, len(array) + 1, dtype=float)
    return float(np.sum((2.0 * index - len(array) - 1.0) * array) / (len(array) * np.sum(array)))


def _workload_characteristics(sequence: Sequence[str], profile: Mapping[str, Any]) -> Dict[str, Any]:
    if not sequence or any(not adapter for adapter in sequence):
        raise SystemExit("workload characteristics require a non-empty all-LoRA adapter sequence")
    counts = Counter(sequence)
    probabilities = np.asarray(list(counts.values()), dtype=float) / len(sequence)
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    last_seen: Dict[str, int] = {}
    reuse_distances: List[int] = []
    for index, adapter in enumerate(sequence):
        if adapter in last_seen:
            # This is request-index reuse distance, matching the existing Fig. 3
            # hot/warm/cold classification rather than LRU stack distance.
            reuse_distances.append(index - last_seen[adapter])
        last_seen[adapter] = index
    if reuse_distances:
        reuse_p50, reuse_p95, reuse_p99 = (
            float(np.percentile(reuse_distances, q)) for q in (50, 95, 99)
        )
    else:
        reuse_p50 = reuse_p95 = reuse_p99 = float("nan")

    rotation = int(profile.get("hotset_rotation_requests") or 0)
    if rotation > 0:
        windows = [set(sequence[start : start + rotation]) for start in range(0, len(sequence), rotation)]
        turnovers = []
        for previous, current in zip(windows, windows[1:]):
            union = previous | current
            turnovers.append(1.0 - (len(previous & current) / len(union)) if union else 0.0)
        turnover = float(np.mean(turnovers)) if turnovers else 0.0
    else:
        turnovers = []
        turnover = 0.0

    return {
        "request_count": len(sequence),
        "actual_unique_adapters": len(counts),
        "effective_adapter_count": math.exp(entropy),
        "entropy_nats": entropy,
        "adapter_frequency_gini": _gini(list(counts.values())),
        "first_touch_ratio": len(counts) / len(sequence),
        "reuse_distance_request_gap_p50": reuse_p50,
        "reuse_distance_request_gap_p95": reuse_p95,
        "reuse_distance_request_gap_p99": reuse_p99,
        "rotation_window_pairs": len(turnovers),
        "empirical_observed_set_turnover_mean": turnover,
        "adapter_sequence_sha256": _adapter_sequence_sha256(sequence),
    }


def _remote_miss_ratio(requests: Sequence[Dict[str, Any]]) -> tuple[float, str]:
    fields = (
        "adapter_remote_cold_before_dispatch",
        "remote_mismatch",
    )
    for field in fields:
        present = [field in request for request in requests]
        if any(present):
            if not all(present):
                raise SystemExit(f"partial request-level remote-miss field coverage: {field}")
            return float(np.mean([bool(request.get(field)) for request in requests])), field
    if all(request.get("readiness_tier_before_dispatch") is not None for request in requests):
        return (
            float(
                np.mean(
                    [
                        str(request.get("readiness_tier_before_dispatch") or "").lower() == "remote"
                        for request in requests
                    ]
                )
            ),
            "readiness_tier_before_dispatch",
        )
    if all(request.get("cache_tier") is not None for request in requests):
        return (
            float(np.mean([str(request.get("cache_tier") or "").lower() == "remote" for request in requests])),
            "cache_tier_service_time_proxy",
        )
    return float("nan"), "unavailable"


def _mean_ci_allow_nan(values: Sequence[float]) -> tuple[float, float, float, int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan"), float("nan"), float("nan"), 0
    mean, std, half_width = _v2_mean_ci95(finite)
    return mean, std, half_width, len(finite)


def _ensure_fresh_output_dir(out_dir: Path, analysis: str) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {analysis} output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)


def _profile_sort_key(profile: Mapping[str, Any]) -> tuple[Any, ...]:
    mode = str(profile.get("rotation_mode") or "")
    rotation = int(profile.get("hotset_rotation_requests") or 0)
    zipf = float(profile.get("zipf_exponent") or 0.0)
    overlap = float(profile.get("hotset_overlap_fraction") or 0.0)
    mode_rank = {"stationary": 0, "abrupt": 1, "gradual": 2}.get(mode, 3)
    return mode_rank, rotation, zipf, overlap


def _profile_short_label(profile: Mapping[str, Any]) -> str:
    mode = str(profile.get("rotation_mode") or "unknown")
    rotation = int(profile.get("hotset_rotation_requests") or 0)
    zipf = float(profile.get("zipf_exponent") or 0.0)
    overlap = float(profile.get("hotset_overlap_fraction") or 0.0)
    if mode == "stationary" or rotation <= 0:
        return f"stationary\nz={zipf:g}"
    suffix = f"\nz={zipf:g}"
    if overlap > 0:
        suffix += f", ov={overlap:g}"
    return f"{mode} r={rotation}{suffix}"


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _write_workload_performance_table(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "% Auto-generated V2 workload-sensitivity table.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Workload-shift sensitivity. Values are seed-level means; CI values remain in the accompanying CSV.}",
        "\\label{tab:v2_workload_sensitivity}",
        "\\begin{tabular}{llrrrrrrrr}",
        "\\hline",
        "Profile & System & TTFT Avg & TTFT p95 & E2E Avg & E2E p95 & TPOT Avg & Tok/s & Cost/req & CE \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(row['profile_name'])} & {_latex_escape(row['system'])} & "
            f"{row['ttft_avg_ms_mean']:.1f} & {row['ttft_p95_ms_mean']:.1f} & "
            f"{row['e2e_avg_ms_mean']:.1f} & {row['e2e_p95_ms_mean']:.1f} & "
            f"{row['tpot_avg_ms_mean']:.2f} & {row['tok_s_mean']:.2f} & "
            f"{row['cost_req_usd_mean']:.6f} & {row['ce_mean']:.2f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_workload_sensitivity(
    inputs: Sequence[Path], out_dir: Path, *, formal_matrix: bool = False
) -> None:
    provenance_index = (
        build_formal_provenance_index(inputs) if formal_matrix else None
    )
    observations = load_v2_sensitivity_observations(inputs)
    if formal_matrix:
        validate_formal_workload_matrix(observations)
        assert provenance_index is not None
        validate_formal_analysis_sources(
            provenance_index,
            (
                FormalAnalysisIdentity(
                    source=observation.source,
                    model=_formal_sensitivity_model_key(observation.model),
                    variant=observation.system_key,
                    seed=observation.seed,
                )
                for observation in observations
            ),
            analysis_label="C4 workload sensitivity",
        )
    _ensure_fresh_output_dir(out_dir, "workload sensitivity")
    by_trace: DefaultDict[tuple[str, str, int], List[V2SensitivityObservation]] = defaultdict(list)
    for observation in observations:
        by_trace[(observation.model, observation.profile_id, observation.seed)].append(observation)

    characteristic_rows: List[Dict[str, Any]] = []
    performance_rows: List[Dict[str, Any]] = []
    for (model, profile_id, seed), group in sorted(by_trace.items()):
        system_counts = Counter(observation.system_key for observation in group)
        duplicates = {key: count for key, count in system_counts.items() if count > 1}
        if duplicates:
            raise SystemExit(
                f"duplicate workload system observations for model={model} "
                f"profile={profile_id} seed={seed}: {duplicates}"
            )
        sequences = {
            observation.system_key: _ordered_adapter_sequence(observation.requests)
            for observation in group
        }
        hashes = {key: _adapter_sequence_sha256(sequence) for key, sequence in sequences.items()}
        if len(set(hashes.values())) != 1:
            raise SystemExit(
                f"adapter sequence mismatch for model={model} profile={profile_id} seed={seed}: {hashes}"
            )
        trace_hashes = {
            _sha256_field(
                observation.metadata,
                observation.manifest,
                "shared_trace_sha256",
            )
            for observation in group
        }
        subset_hashes = {
            _sha256_field(
                observation.metadata,
                observation.manifest,
                "shared_adapter_subset_sha256",
            )
            for observation in group
        }
        if len(trace_hashes) != 1 or len(subset_hashes) != 1:
            raise SystemExit(
                f"workload comparability failed for model={model} profile={profile_id} "
                f"seed={seed}: trace_hashes={trace_hashes}, subset_hashes={subset_hashes}"
            )
        trace_sha = next(iter(trace_hashes))
        subset_sha = next(iter(subset_hashes))
        canonical = next(
            (observation for observation in group if observation.system_key == "faaslora"),
            group[0],
        )
        characteristics = _workload_characteristics(
            sequences[canonical.system_key], canonical.profile
        )
        characteristic_rows.append(
            {
                "model": model,
                "profile_id": profile_id,
                "profile_name": canonical.profile_name,
                "seed": seed,
                "rotation_mode": canonical.profile["rotation_mode"],
                "hotset_rotation_requests": canonical.profile["hotset_rotation_requests"],
                "zipf_exponent": canonical.profile["zipf_exponent"],
                "hotset_overlap_fraction": canonical.profile["hotset_overlap_fraction"],
                "active_adapter_cap": canonical.profile["active_adapter_cap"],
                "nominal_adapter_pool_size": canonical.profile["nominal_adapter_pool_size"],
                "shared_trace_sha256": trace_sha,
                "shared_adapter_subset_sha256": subset_sha,
                **characteristics,
            }
        )
        for observation in group:
            remote_ratio, remote_source = _remote_miss_ratio(observation.requests)
            performance_rows.append(
                {
                    "model": model,
                    "profile_id": profile_id,
                    "profile_name": observation.profile_name,
                    "seed": seed,
                    "system_key": observation.system_key,
                    "system": SENSITIVITY_SYSTEM_LABELS[observation.system_key],
                    "scenario": observation.scenario,
                    "source": str(observation.source),
                    "completed": observation.completed,
                    "shared_trace_sha256": trace_sha,
                    "shared_adapter_subset_sha256": subset_sha,
                    "adapter_sequence_sha256": hashes[observation.system_key],
                    "remote_miss_ratio": remote_ratio,
                    "remote_miss_source": remote_source,
                    **observation.metrics,
                }
            )

    characteristic_metrics = (
        "request_count",
        "actual_unique_adapters",
        "effective_adapter_count",
        "entropy_nats",
        "adapter_frequency_gini",
        "first_touch_ratio",
        "reuse_distance_request_gap_p50",
        "reuse_distance_request_gap_p95",
        "reuse_distance_request_gap_p99",
        "empirical_observed_set_turnover_mean",
    )
    characteristic_summary: List[Dict[str, Any]] = []
    grouped_characteristics: DefaultDict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in characteristic_rows:
        grouped_characteristics[(str(row["model"]), str(row["profile_id"]))].append(row)
    for (model, profile_id), rows in sorted(grouped_characteristics.items()):
        output: Dict[str, Any] = {
            "model": model,
            "profile_id": profile_id,
            "profile_name": rows[0]["profile_name"],
            "seed_count": len(rows),
            "seeds": ";".join(str(row["seed"]) for row in rows),
            "rotation_mode": rows[0]["rotation_mode"],
            "hotset_rotation_requests": rows[0]["hotset_rotation_requests"],
            "zipf_exponent": rows[0]["zipf_exponent"],
            "hotset_overlap_fraction": rows[0]["hotset_overlap_fraction"],
            "nominal_adapter_pool_size": rows[0]["nominal_adapter_pool_size"],
            "shared_trace_sha256_by_seed": ";".join(
                f"{row['seed']}:{row['shared_trace_sha256']}" for row in rows
            ),
            "shared_adapter_subset_sha256_by_seed": ";".join(
                f"{row['seed']}:{row['shared_adapter_subset_sha256']}" for row in rows
            ),
        }
        for metric in characteristic_metrics:
            avg, std, half, n = _mean_ci_allow_nan([float(row[metric]) for row in rows])
            output[f"{metric}_mean"] = avg
            output[f"{metric}_std"] = std
            output[f"{metric}_ci95_half_width"] = half
            output[f"{metric}_seed_count"] = n
        characteristic_summary.append(output)

    performance_metrics = (
        "ttft_avg_ms",
        "ttft_p95_ms",
        "e2e_avg_ms",
        "e2e_p95_ms",
        "tpot_avg_ms",
        "tpot_p95_ms",
        "tok_s",
        "cost_req_usd",
        "ce",
        "remote_miss_ratio",
    )
    performance_summary: List[Dict[str, Any]] = []
    grouped_performance: DefaultDict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in performance_rows:
        grouped_performance[(str(row["model"]), str(row["profile_id"]), str(row["system_key"]))].append(row)
    for (model, profile_id, system_key), rows in sorted(grouped_performance.items()):
        output = {
            "model": model,
            "profile_id": profile_id,
            "profile_name": rows[0]["profile_name"],
            "system_key": system_key,
            "system": SENSITIVITY_SYSTEM_LABELS[system_key],
            "seed_count": len(rows),
            "seeds": ";".join(str(row["seed"]) for row in rows),
            "shared_trace_sha256_by_seed": ";".join(
                f"{row['seed']}:{row['shared_trace_sha256']}" for row in rows
            ),
            "shared_adapter_subset_sha256_by_seed": ";".join(
                f"{row['seed']}:{row['shared_adapter_subset_sha256']}" for row in rows
            ),
            "remote_miss_sources": ";".join(sorted({str(row["remote_miss_source"]) for row in rows})),
        }
        for metric in performance_metrics:
            avg, std, half, n = _mean_ci_allow_nan([float(row[metric]) for row in rows])
            output[f"{metric}_mean"] = avg
            output[f"{metric}_std"] = std
            output[f"{metric}_ci95_half_width"] = half
            output[f"{metric}_seed_count"] = n
        performance_summary.append(output)

    characteristics_seed_csv = out_dir / "workload_characteristics_per_seed.csv"
    characteristics_csv = out_dir / "workload_characteristics_summary.csv"
    performance_seed_csv = out_dir / "workload_performance_per_seed.csv"
    performance_csv = out_dir / "workload_performance_summary.csv"
    table_path = out_dir / "table_workload_sensitivity.tex"
    _write_csv(characteristics_seed_csv, characteristic_rows)
    _write_csv(characteristics_csv, characteristic_summary)
    _write_csv(performance_seed_csv, performance_rows)
    _write_csv(performance_csv, performance_summary)
    _write_workload_performance_table(table_path, performance_summary)

    models = sorted({observation.model for observation in observations})
    pdfs: List[str] = []
    for model in models:
        model_characteristics = [row for row in characteristic_rows if row["model"] == model]
        profile_lookup: Dict[str, Dict[str, Any]] = {}
        for observation in observations:
            if observation.model == model:
                profile_lookup[observation.profile_id] = observation.profile
        profiles = sorted(profile_lookup, key=lambda key: _profile_sort_key(profile_lookup[key]))
        systems = [
            key
            for key in ("faaslora", "elastic_only", "serverlessllm", "sglang", "vllm", "slora")
            if any(row["model"] == model and row["system_key"] == key for row in performance_summary)
        ]
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.0), constrained_layout=True)
        specs = (
            ("ttft_p95_ms", "P95 TTFT (ms)"),
            ("e2e_avg_ms", "Average E2E (ms)"),
            ("cost_req_usd", "Cost/request (mUSD)"),
            ("ce", "CE"),
        )
        for ax, (metric, ylabel) in zip(axes.flat, specs):
            for system_key in systems:
                means: List[float] = []
                errors: List[float] = []
                for profile_id in profiles:
                    row = next(
                        (
                            item
                            for item in performance_summary
                            if item["model"] == model
                            and item["profile_id"] == profile_id
                            and item["system_key"] == system_key
                        ),
                        None,
                    )
                    scale = 1000.0 if metric == "cost_req_usd" else 1.0
                    means.append(float(row[f"{metric}_mean"]) * scale if row else float("nan"))
                    raw_error = float(row[f"{metric}_ci95_half_width"]) if row else float("nan")
                    errors.append(0.0 if not math.isfinite(raw_error) else raw_error * scale)
                ax.errorbar(
                    np.arange(len(profiles)),
                    means,
                    yerr=errors,
                    marker=SENSITIVITY_SYSTEM_MARKERS[system_key],
                    color=SENSITIVITY_SYSTEM_COLORS[system_key],
                    linewidth=1.3,
                    capsize=2.5,
                    label=SENSITIVITY_SYSTEM_LABELS[system_key],
                )
            ax.set_xticks(
                np.arange(len(profiles)),
                [_profile_short_label(profile_lookup[profile]) for profile in profiles],
                rotation=18,
                ha="right",
            )
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncols=max(1, len(labels)), frameon=False)
        fig.suptitle(f"V2 workload representativeness — {model}", fontsize=10.5)
        filename = "fig_workload_sensitivity.pdf" if len(models) == 1 else f"fig_workload_sensitivity_{model}.pdf"
        fig.savefig(out_dir / filename, bbox_inches="tight")
        plt.close(fig)
        pdfs.append(filename)

    manifest = {
        "analysis": "v2_workload_sensitivity",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(Path(path).expanduser().resolve()) for path in inputs],
        "sources": sorted({str(observation.source) for observation in observations}),
        "formal_matrix": formal_matrix,
        "profile_identity_source": "trace load_profile and/or campaign MANIFEST metadata; paths are not used as profile identity",
        "statistical_unit": "independent seed",
        "ci": "two-sided 95% Student-t over seed-level values; absent for n=1",
        "entropy_log_base": "natural",
        "effective_adapter_count": "exp(entropy_nats)",
        "gini_scope": "request counts over adapters actually observed in the trace",
        "reuse_distance_definition": "request-index gap since previous request for the same adapter; first touches excluded",
        "empirical_observed_set_turnover_definition": (
            "mean 1-Jaccard between adapter sets empirically observed in adjacent "
            "request windows of the declared rotation length; it is not a direct "
            "measurement of the generator's latent hot sets; stationary/no-pair value is 0"
        ),
        "strict_comparability": (
            "within each model/profile/seed, shared trace and adapter-subset SHA-256 "
            "digests must be present and identical across systems, and the observed "
            "adapter sequence SHA-256 must also match"
        ),
        "formal_matrix_check": (
            "exact C4 seven-profile Full/ServerlessLLM seed-43 matrix; "
            "stationary/rotation-100/rotation-500 Full/ServerlessLLM seeds 44/45; "
            "and ElasticOnly seeds 43/44/45 for those three profiles"
            if formal_matrix
            else "not requested"
        ),
        "formal_provenance_check": (
            "complete heldout source-clean committed manifests; per-source byte/SHA-256 "
            "integrity; invariant system_resolved_config_sha256 across workload points "
            "within each model/system"
            if formal_matrix
            else "not requested"
        ),
        "remote_miss_policy": "dispatch-time remote field preferred; cache_tier is explicitly labeled service-time proxy",
        "pdfs": pdfs,
        "csvs": [
            characteristics_seed_csv.name,
            characteristics_csv.name,
            performance_seed_csv.name,
            performance_csv.name,
        ],
        "table_tex": table_path.name,
    }
    (out_dir / "workload_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _sha256_field(metadata: Mapping[str, Any], manifest: Mapping[str, Any], field: str) -> str:
    recorded = [
        (source, str(value).strip().lower())
        for source, value in (
            (f"metadata.{field}", metadata.get(field)),
            (f"manifest.{field}", manifest.get(field)),
        )
        if value is not None and str(value).strip()
    ]
    if not recorded:
        raise SystemExit(f"missing or invalid {field}; expected a recorded SHA-256 digest")
    for source, digest in recorded:
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise SystemExit(f"invalid {source}; expected a SHA-256 digest")
    unique = {digest for _, digest in recorded}
    if len(unique) != 1:
        raise SystemExit(f"conflicting {field} values: {recorded}")
    return next(iter(unique))


def _bandwidth_block(observation: V2SensitivityObservation) -> tuple[Dict[str, Any], str]:
    metadata = observation.metadata
    manifest = observation.manifest
    candidates = (
        (
            _dig(metadata, "scenario_coordination", observation.scenario, "aggregate_bandwidth"),
            f"metadata.scenario_coordination.{observation.scenario}.aggregate_bandwidth",
        ),
        (metadata.get("aggregate_bandwidth"), "metadata.aggregate_bandwidth"),
        (
            _dig(manifest, "scenario_coordination", observation.scenario, "aggregate_bandwidth"),
            f"manifest.scenario_coordination.{observation.scenario}.aggregate_bandwidth",
        ),
        (manifest.get("aggregate_bandwidth"), "manifest.aggregate_bandwidth"),
    )
    for candidate, label in candidates:
        if isinstance(candidate, dict):
            return dict(candidate), label
    raise SystemExit(
        f"{observation.source}:{observation.scenario}: missing aggregate_bandwidth audit metadata"
    )


def _bandwidth_audit(observation: V2SensitivityObservation) -> BandwidthAudit:
    metadata = observation.metadata
    manifest = observation.manifest
    block, block_source = _bandwidth_block(observation)
    top_mode = _first_value(metadata.get("bandwidth_limit_mode"), manifest.get("bandwidth_limit_mode"))
    raw_mode = str(_first_value(block.get("limit_mode"), top_mode) or "").strip()
    normalized_raw_mode = raw_mode.lower().replace("_", "-")
    if not normalized_raw_mode:
        raise SystemExit(f"{observation.source}: missing bandwidth limit_mode")
    if top_mode:
        normalized_top = str(top_mode).strip().lower().replace("_", "-")
        if ("aggregate" in normalized_raw_mode) != ("aggregate" in normalized_top):
            raise SystemExit(
                f"{observation.source}: top-level bandwidth_limit_mode={top_mode!r} "
                f"conflicts with {block_source}.limit_mode={block.get('limit_mode')!r}"
            )

    configured_raw = _first_value(block.get("configured_mib_s"), metadata.get("bandwidth_mib_s"), manifest.get("bandwidth_mib_s"))
    gbit_raw = _first_value(block.get("configured_gbit_s"), metadata.get("bandwidth_gbit_s"), manifest.get("bandwidth_gbit_s"))
    configured_numeric = None
    if configured_raw not in (None, ""):
        try:
            configured_numeric = float(configured_raw)
        except Exception:
            configured_numeric = None
    disabled = (
        normalized_raw_mode
        in {
            "disabled",
            "no-delay",
            "nodelay",
            "unlimited",
            "none",
            "local-sim-no-delay",
            "file-no-delay",
        }
        or configured_numeric == 0.0
    )
    if disabled:
        configured = None if configured_raw in (None, "", 0, 0.0, "0") else float(configured_raw)
        if configured is not None:
            raise SystemExit(f"{observation.source}: no-delay mode cannot record a positive configured_mib_s")
        configured_gbit = None if gbit_raw in (None, "", 0, 0.0, "0") else float(gbit_raw)
        if configured_gbit is not None:
            raise SystemExit(f"{observation.source}: no-delay mode cannot record a positive configured_gbit_s")
        bandwidth_key = "no-delay"
        bandwidth_label = "no-delay"
        semantic_mode = "no-delay"
    else:
        if "aggregate" not in normalized_raw_mode:
            raise SystemExit(
                f"{observation.source}: limited run must use aggregate limit mode, "
                f"observed {raw_mode!r}"
            )
        configured = _as_float(configured_raw, f"{observation.source}.configured_mib_s")
        if configured <= 0:
            raise SystemExit(f"{observation.source}: configured_mib_s must be positive")
        configured_gbit = _as_float(gbit_raw, f"{observation.source}.configured_gbit_s")
        expected_gbit = configured * 8.0 * 1024.0 * 1024.0 / 1_000_000_000.0
        if not math.isclose(configured_gbit, expected_gbit, rel_tol=2e-4, abs_tol=2e-5):
            raise SystemExit(
                f"{observation.source}: configured_gbit_s={configured_gbit} is inconsistent "
                f"with configured_mib_s={configured} (expected {expected_gbit})"
            )
        for field, top_value in (
            ("bandwidth_mib_s", metadata.get("bandwidth_mib_s")),
            ("bandwidth_gbit_s", metadata.get("bandwidth_gbit_s")),
        ):
            if top_value is None:
                continue
            expected = configured if field.endswith("mib_s") else configured_gbit
            if not math.isclose(float(top_value), float(expected), rel_tol=1e-6, abs_tol=1e-8):
                raise SystemExit(
                    f"{observation.source}: metadata.{field}={top_value} conflicts with aggregate block"
                )
        bandwidth_key = f"mib_s={configured:.6f}"
        bandwidth_label = f"{configured_gbit:.4g} Gbit/s"
        # Prime's process limiter and ServerlessLLM's file-backed reservation
        # limiter have different implementation labels but the same experiment
        # semantics: one application-level aggregate cap shared by concurrent
        # fetches. Keep the implementation label separately for audit.
        semantic_mode = "aggregate"

    audit_values: Dict[str, float] = {}
    for field in (
        "transfer_count",
        "total_bytes",
        "reservation_span_s",
        "total_injected_wait_s",
        "achieved_reserved_mib_s",
    ):
        audit_values[field] = _as_float(block.get(field), f"{observation.source}.{block_source}.{field}")
        if audit_values[field] < 0:
            raise SystemExit(f"{observation.source}: aggregate bandwidth field {field} must be non-negative")
    if not disabled and (
        int(audit_values["transfer_count"]) <= 0 or int(audit_values["total_bytes"]) <= 0
    ):
        raise SystemExit(f"{observation.source}: aggregate bandwidth audit must record transfers and bytes")
    trace_sha = _sha256_field(metadata, manifest, "shared_trace_sha256")
    subset_sha = _sha256_field(metadata, manifest, "shared_adapter_subset_sha256")
    return BandwidthAudit(
        bandwidth_key=bandwidth_key,
        bandwidth_label=bandwidth_label,
        raw_limit_mode=raw_mode,
        limit_mode=semantic_mode,
        configured_mib_s=configured,
        configured_gbit_s=configured_gbit,
        transfer_count=int(audit_values["transfer_count"]),
        total_bytes=int(audit_values["total_bytes"]),
        reservation_span_s=audit_values["reservation_span_s"],
        total_injected_wait_s=audit_values["total_injected_wait_s"],
        achieved_reserved_mib_s=audit_values["achieved_reserved_mib_s"],
        trace_sha256=trace_sha,
        adapter_subset_sha256=subset_sha,
    )


def validate_formal_bandwidth_matrix(
    observations: Sequence[V2SensitivityObservation],
) -> None:
    """Require the exact A6 model/bandwidth/system/seed matrix."""
    expected: set[tuple[Any, ...]] = {
        ("7b", bandwidth, 43, system)
        for bandwidth in FORMAL_BANDWIDTH_KEYS
        for system in ("faaslora", "serverlessllm")
    }
    expected.update(
        ("7b", bandwidth, seed, system)
        for bandwidth in FORMAL_BANDWIDTH_REPLICATED_KEYS
        for seed in (44, 45)
        for system in ("faaslora", "serverlessllm")
    )
    expected.update(
        ("3b", bandwidth, 43, system)
        for bandwidth in FORMAL_BANDWIDTH_REPLICATED_KEYS
        for system in ("faaslora", "serverlessllm")
    )

    observed_counts: DefaultDict[tuple[Any, ...], int] = defaultdict(int)
    profiles_by_model: DefaultDict[str, set[str]] = defaultdict(set)
    for observation in observations:
        _validate_formal_system_scenario(observation)
        model = _formal_sensitivity_model_key(observation.model)
        profiles_by_model[model].add(observation.profile_id)
        audit = _bandwidth_audit(observation)
        identity = (model, audit.bandwidth_key, observation.seed, observation.system_key)
        observed_counts[identity] += 1
    multiple_profiles = {
        model: sorted(profiles)
        for model, profiles in profiles_by_model.items()
        if len(profiles) != 1
    }
    if multiple_profiles:
        raise SystemExit(
            "formal A6 matrix must use exactly one frozen workload profile per model; "
            f"observed={multiple_profiles}"
        )
    _raise_formal_identity_mismatch("A6", expected, observed_counts)


def _validate_bandwidth_comparability(
    rows: Sequence[Dict[str, Any]],
) -> None:
    groups: DefaultDict[tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model"]), str(row["profile_id"]), int(row["seed"]))].append(row)
    for (model, profile_id, seed), group in groups.items():
        trace_hashes = {str(row["shared_trace_sha256"]) for row in group}
        subset_hashes = {str(row["shared_adapter_subset_sha256"]) for row in group}
        completions = {(int(row["completed"]), int(row["total"])) for row in group}
        if len(trace_hashes) != 1 or len(subset_hashes) != 1:
            raise SystemExit(
                f"bandwidth comparability failed for {model}/{profile_id}/seed={seed}: "
                f"trace_hashes={trace_hashes}, subset_hashes={subset_hashes}"
            )
        if len(completions) != 1:
            raise SystemExit(
                f"bandwidth completion mismatch for {model}/{profile_id}/seed={seed}: {completions}"
            )
        by_bandwidth: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_bandwidth[str(row["bandwidth_key"])].append(row)
        expected_systems: set[str] | None = None
        for bandwidth_key, point_rows in sorted(by_bandwidth.items()):
            systems = [str(row["system_key"]) for row in point_rows]
            if len(systems) != len(set(systems)):
                raise SystemExit(
                    f"duplicate system at bandwidth point {model}/{seed}/{bandwidth_key}: {systems}"
                )
            system_set = set(systems)
            if expected_systems is None:
                expected_systems = system_set
            elif system_set != expected_systems:
                raise SystemExit(
                    f"system-set mismatch across bandwidth points for {model}/seed={seed}: "
                    f"expected={sorted(expected_systems)}, {bandwidth_key}={sorted(system_set)}"
                )
            modes = {str(row["limit_mode"]) for row in point_rows}
            configured = {str(row["configured_mib_s"]) for row in point_rows}
            if len(modes) != 1 or len(configured) != 1:
                raise SystemExit(
                    f"limit metadata mismatch between systems at {model}/seed={seed}/{bandwidth_key}"
                )


def _write_bandwidth_table(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "% Auto-generated V2 bandwidth-sensitivity table.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Application-level aggregate storage-bandwidth sensitivity. Values are means over independent seeds.}",
        "\\label{tab:v2_bandwidth_sensitivity}",
        "\\begin{tabular}{llrrrrrrrr}",
        "\\hline",
        "Bandwidth & System & TTFT Avg & TTFT p95 & E2E Avg & E2E p95 & TPOT Avg & Tok/s & Cost/req & CE \\\\ ",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(row['bandwidth_label'])} & {_latex_escape(row['system'])} & "
            f"{row['ttft_avg_ms_mean']:.1f} & {row['ttft_p95_ms_mean']:.1f} & "
            f"{row['e2e_avg_ms_mean']:.1f} & {row['e2e_p95_ms_mean']:.1f} & "
            f"{row['tpot_avg_ms_mean']:.2f} & {row['tok_s_mean']:.2f} & "
            f"{row['cost_req_usd_mean']:.6f} & {row['ce_mean']:.2f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_bandwidth_sensitivity(
    inputs: Sequence[Path], out_dir: Path, *, formal_matrix: bool = False
) -> None:
    provenance_index = (
        build_formal_provenance_index(inputs) if formal_matrix else None
    )
    observations = load_v2_sensitivity_observations(inputs)
    if formal_matrix:
        validate_formal_bandwidth_matrix(observations)
        assert provenance_index is not None
        validate_formal_analysis_sources(
            provenance_index,
            (
                FormalAnalysisIdentity(
                    source=observation.source,
                    model=_formal_sensitivity_model_key(observation.model),
                    variant=observation.system_key,
                    seed=observation.seed,
                )
                for observation in observations
            ),
            analysis_label="A6 bandwidth sensitivity",
        )
    _ensure_fresh_output_dir(out_dir, "bandwidth sensitivity")
    per_seed_rows: List[Dict[str, Any]] = []
    for observation in observations:
        audit = _bandwidth_audit(observation)
        per_seed_rows.append(
            {
                "model": observation.model,
                "profile_id": observation.profile_id,
                "profile_name": observation.profile_name,
                "seed": observation.seed,
                "system_key": observation.system_key,
                "system": SENSITIVITY_SYSTEM_LABELS[observation.system_key],
                "scenario": observation.scenario,
                "run_tag": observation.run_tag,
                "source": str(observation.source),
                "completed": observation.completed,
                "total": observation.total,
                "bandwidth_key": audit.bandwidth_key,
                "bandwidth_label": audit.bandwidth_label,
                "limit_mode": audit.limit_mode,
                "raw_limit_mode": audit.raw_limit_mode,
                "configured_mib_s": audit.configured_mib_s,
                "configured_gbit_s": audit.configured_gbit_s,
                "transfer_count": audit.transfer_count,
                "total_bytes": audit.total_bytes,
                "reservation_span_s": audit.reservation_span_s,
                "total_injected_wait_s": audit.total_injected_wait_s,
                "achieved_reserved_mib_s": audit.achieved_reserved_mib_s,
                "shared_trace_sha256": audit.trace_sha256,
                "shared_adapter_subset_sha256": audit.adapter_subset_sha256,
                **observation.metrics,
            }
        )
    _validate_bandwidth_comparability(per_seed_rows)

    metrics = (
        "ttft_avg_ms",
        "ttft_p95_ms",
        "e2e_avg_ms",
        "e2e_p95_ms",
        "tpot_avg_ms",
        "tpot_p95_ms",
        "tok_s",
        "cost_req_usd",
        "ce",
        "achieved_reserved_mib_s",
        "total_injected_wait_s",
        "transfer_count",
        "total_bytes",
    )
    grouped: DefaultDict[tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in per_seed_rows:
        grouped[(str(row["model"]), str(row["profile_id"]), str(row["system_key"]), str(row["bandwidth_key"]))].append(row)
    summary_rows: List[Dict[str, Any]] = []
    for (model, profile_id, system_key, bandwidth_key), rows in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            float("inf")
            if item[1][0]["configured_mib_s"] is None
            else float(item[1][0]["configured_mib_s"]),
            item[0][2],
        ),
    ):
        output: Dict[str, Any] = {
            "model": model,
            "profile_id": profile_id,
            "profile_name": rows[0]["profile_name"],
            "system_key": system_key,
            "system": SENSITIVITY_SYSTEM_LABELS[system_key],
            "bandwidth_key": bandwidth_key,
            "bandwidth_label": rows[0]["bandwidth_label"],
            "limit_mode": rows[0]["limit_mode"],
            "raw_limit_modes": ";".join(
                sorted({str(row["raw_limit_mode"]) for row in rows})
            ),
            "configured_mib_s": rows[0]["configured_mib_s"],
            "configured_gbit_s": rows[0]["configured_gbit_s"],
            "seed_count": len(rows),
            "seeds": ";".join(str(row["seed"]) for row in rows),
            "shared_trace_sha256_by_seed": ";".join(f"{row['seed']}:{row['shared_trace_sha256']}" for row in rows),
            "shared_adapter_subset_sha256_by_seed": ";".join(f"{row['seed']}:{row['shared_adapter_subset_sha256']}" for row in rows),
        }
        for metric in metrics:
            avg, std, half, n = _mean_ci_allow_nan([float(row[metric]) for row in rows])
            output[f"{metric}_mean"] = avg
            output[f"{metric}_std"] = std
            output[f"{metric}_ci95_half_width"] = half
            output[f"{metric}_seed_count"] = n
        summary_rows.append(output)

    per_seed_csv = out_dir / "bandwidth_sensitivity_per_seed.csv"
    summary_csv = out_dir / "bandwidth_sensitivity_summary.csv"
    table_path = out_dir / "table_bandwidth_sensitivity.tex"
    _write_csv(per_seed_csv, per_seed_rows)
    _write_csv(summary_csv, summary_rows)
    _write_bandwidth_table(table_path, summary_rows)

    models = sorted({str(row["model"]) for row in summary_rows})
    pdfs: List[str] = []
    for model in models:
        model_rows = [row for row in summary_rows if row["model"] == model]
        bandwidths = sorted(
            {str(row["bandwidth_key"]) for row in model_rows},
            key=lambda key: (
                next(row["configured_mib_s"] for row in model_rows if row["bandwidth_key"] == key)
                is None,
                float(
                    next(row["configured_mib_s"] for row in model_rows if row["bandwidth_key"] == key)
                    or 0.0
                ),
            ),
        )
        systems = [
            key
            for key in ("faaslora", "elastic_only", "serverlessllm", "sglang", "vllm", "slora")
            if any(row["system_key"] == key for row in model_rows)
        ]
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.8), constrained_layout=True)
        specs = (
            ("ttft_avg_ms", "Average TTFT (ms)"),
            ("ttft_p95_ms", "P95 TTFT (ms)"),
            ("cost_req_usd", "Cost/request (mUSD)"),
            ("ce", "CE"),
        )
        labels = [
            str(next(row["bandwidth_label"] for row in model_rows if row["bandwidth_key"] == key))
            for key in bandwidths
        ]
        for ax, (metric, ylabel) in zip(axes.flat, specs):
            for system_key in systems:
                selected = {
                    str(row["bandwidth_key"]): row
                    for row in model_rows
                    if row["system_key"] == system_key
                }
                scale = 1000.0 if metric == "cost_req_usd" else 1.0
                means = [float(selected[key][f"{metric}_mean"]) * scale for key in bandwidths]
                errors = []
                for key in bandwidths:
                    value = float(selected[key][f"{metric}_ci95_half_width"])
                    errors.append(0.0 if not math.isfinite(value) else value * scale)
                ax.errorbar(
                    np.arange(len(bandwidths)),
                    means,
                    yerr=errors,
                    marker=SENSITIVITY_SYSTEM_MARKERS[system_key],
                    color=SENSITIVITY_SYSTEM_COLORS[system_key],
                    linewidth=1.3,
                    capsize=2.5,
                    label=SENSITIVITY_SYSTEM_LABELS[system_key],
                )
            ax.set_xticks(np.arange(len(bandwidths)), labels, rotation=18, ha="right")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", ncols=max(1, len(legend_labels)), frameon=False)
        fig.suptitle(f"Aggregate storage-bandwidth sensitivity — {model}", fontsize=10.5)
        filename = "fig_bandwidth_sensitivity.pdf" if len(models) == 1 else f"fig_bandwidth_sensitivity_{model}.pdf"
        fig.savefig(out_dir / filename, bbox_inches="tight")
        plt.close(fig)
        pdfs.append(filename)

    manifest_payload = {
        "analysis": "v2_bandwidth_sensitivity",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(Path(path).expanduser().resolve()) for path in inputs],
        "sources": sorted({str(observation.source) for observation in observations}),
        "formal_matrix": formal_matrix,
        "strict_checks": [
            "completed == total and equal completion count within model/profile/seed",
            "identical shared_trace_sha256 and shared_adapter_subset_sha256 within model/profile/seed",
            "raw limiter modes are retained while Prime and ServerlessLLM implementations normalize to semantic aggregate/no-delay modes",
            "semantic aggregate limiter mode and configured MiB/s agree between systems at each point",
            "all systems expose the same bandwidth points within each seed",
            "configured Gbit/s is consistent with MiB/s using binary-byte to decimal-bit conversion",
            *(
                [
                    "formal A6 identity set exactly matches the declared 7B six-point and 3B three-anchor model/system/seed matrix",
                    "formal campaign manifests are complete, heldout, source-clean, and tied to non-empty commits",
                    "every analyzed raw JSON matches its manifest byte count and SHA-256 record",
                    "system_resolved_config_sha256 is valid and invariant across bandwidth points within each model/system",
                ]
                if formal_matrix
                else []
            ),
        ],
        "statistical_unit": "independent seed",
        "ci": "two-sided 95% Student-t over seed-level values; absent for n=1",
        "no_delay_interpretation": "application-level no-delay optimistic upper bound; not a physical 100/400GbE claim",
        "pdfs": pdfs,
        "csvs": [per_seed_csv.name, summary_csv.name],
        "table_tex": table_path.name,
    }
    (out_dir / "bandwidth_sensitivity_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


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
    if rows:
        return _as_float(rows[0].get("RPS"), f"{round_dir}.fallback.RPS")
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


OverrideMap = Dict[tuple[str, str], Path]


def _parse_overrides(values: Sequence[str] | None) -> OverrideMap:
    overrides: OverrideMap = {}
    for value in values or []:
        parts = value.split(":", 2)
        if len(parts) != 3:
            raise SystemExit(
                "--system-summary-override must use '<round_dir>:<system_key>:<summary_json>'"
            )
        round_dir, system_key, summary_path = parts
        key = _system_key(system_key)
        if key not in SYSTEM_ORDER:
            raise SystemExit(f"unknown system in override: {system_key}")
        path = Path(summary_path).resolve()
        if not path.exists():
            raise SystemExit(f"override summary does not exist: {path}")
        overrides[(str(Path(round_dir).resolve()), key)] = path
    return overrides


def _main_round_data_with_overrides(round_dir: Path, overrides: OverrideMap) -> List[MainSystemData]:
    round_dir = round_dir.resolve()
    local_overrides = {
        system_key: path
        for (override_round, system_key), path in overrides.items()
        if override_round == str(round_dir)
    }
    if not local_overrides:
        return _main_round_data(round_dir)

    manifest = _load_json(round_dir / "MANIFEST.json")
    if manifest.get("metric_schema_version") != "e2e_v3":
        raise SystemExit(f"{round_dir / 'MANIFEST.json'}: metric_schema_version must be e2e_v3")
    run_tag = str(manifest.get("run_tag") or "")
    if not run_tag:
        raise SystemExit(f"{round_dir / 'MANIFEST.json'}: missing run_tag")

    systems: List[MainSystemData] = []
    for key in SYSTEM_ORDER:
        source = local_overrides.get(key) or _main_summary_path(round_dir, run_tag, key)
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
            metrics["monetary_active_charge_gpu_seconds"] + metrics["monetary_idle_charge_gpu_seconds"],
            1e-12,
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
        systems.append(MainSystemData(key=key, label=SYSTEM_LABELS[key], source=source, metrics=metrics))
    return systems


def _collect(round_dirs: Sequence[Path], overrides: OverrideMap) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for round_dir in round_dirs:
        systems = _main_round_data_with_overrides(round_dir, overrides)
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


def _collect_adapter_pool(round_dirs: Sequence[Path], overrides: OverrideMap) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for round_dir in round_dirs:
        systems = _main_round_data_with_overrides(round_dir, overrides)
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


def plot_adapter_pool_sensitivity(round_dirs: Sequence[Path], out_dir: Path, overrides: OverrideMap) -> None:
    rows = _collect_adapter_pool([Path(path).resolve() for path in round_dirs], overrides)
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


def plot_load_sensitivity(round_dirs: Sequence[Path], out_dir: Path, overrides: OverrideMap) -> None:
    rows = _collect([Path(path).resolve() for path in round_dirs], overrides)
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
    parser.add_argument("--round-dir", action="append", type=Path, help="Completed fair round directory; repeat as needed.")
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        default=[],
        help="V2 result JSON, MANIFEST.json, or campaign root; repeat as needed.",
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
        "--figure",
        choices=["load", "adapter_pool", "workload", "bandwidth"],
        default="load",
    )
    parser.add_argument(
        "--system-summary-override",
        action="append",
        help="Override one system summary for one round as '<round_dir>:<system_key>:<summary_json>'.",
    )
    parser.add_argument(
        "--formal-matrix",
        action="store_true",
        help=(
            "For workload or bandwidth analysis, require the exact formal V2 "
            "model/profile/point/system/seed identity matrix."
        ),
    )
    args = parser.parse_args()
    round_dirs = list(args.round_dir or [])
    if args.figure in {"workload", "bandwidth"}:
        inputs = [*args.input, *round_dirs]
        if not inputs:
            raise SystemExit(f"--figure {args.figure} requires at least one --input or --round-dir")
        if args.system_summary_override:
            raise SystemExit(f"--system-summary-override is not valid for --figure {args.figure}")
        if args.figure == "workload":
            plot_workload_sensitivity(
                inputs,
                args.out_dir.resolve(),
                formal_matrix=args.formal_matrix,
            )
        else:
            plot_bandwidth_sensitivity(
                inputs,
                args.out_dir.resolve(),
                formal_matrix=args.formal_matrix,
            )
        print(f"generated {args.figure} sensitivity -> {args.out_dir.resolve()}")
        return
    if args.input:
        raise SystemExit("--input is only valid with --figure workload or bandwidth")
    if args.formal_matrix:
        raise SystemExit("--formal-matrix is only valid with --figure workload or bandwidth")
    if not round_dirs:
        raise SystemExit(f"--figure {args.figure} requires at least one --round-dir")
    overrides = _parse_overrides(args.system_summary_override)
    if args.figure == "adapter_pool":
        plot_adapter_pool_sensitivity(round_dirs, args.out_dir.resolve(), overrides)
        print(f"generated fig9_adapter_pool_sensitivity -> {args.out_dir.resolve()}")
    else:
        plot_load_sensitivity(round_dirs, args.out_dir.resolve(), overrides)
        print(f"generated fig8_load_sensitivity -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
