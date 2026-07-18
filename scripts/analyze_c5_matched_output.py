#!/usr/bin/env python3
"""Validate and summarize the EuroSys V2 C5 matched-output experiment.

The statistical unit in this analysis is one frozen workload seed.  Request
records are used to validate the generation contract and to compute each run's
metrics, but they are never treated as independent experimental repetitions.

The command accepts either completed fair-round directories or explicit result
files::

    python scripts/analyze_c5_matched_output.py \
      --round-dir /path/to/seed43-round \
      --round-dir /path/to/seed44-round \
      --round-dir /path/to/seed45-round \
      --output-dir paper_results/eurosys27_v2/c5_slora/formal

    python scripts/analyze_c5_matched_output.py \
      --run prime llama2_7b 43 /path/to/faaslora_result.json \
      --run slora llama2_7b 43 /path/to/slora_summary.json \
      --output-dir /new/output/directory

``--output-dir`` must be absent or empty.  This deliberately prevents a V2
analysis from silently overwriting an earlier publication artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONTRACT = "fixed_length_greedy_v1"
SYSTEM_LABELS = {"prime": "PrimeLoRA", "slora": "S-LoRA"}
SHA256_LENGTH = 64

# Student-t 0.975 quantiles.  Formal C5 uses three seeds (df=2); the wider
# table keeps the analyzer useful for later repetitions without scipy.
T975 = {
    1: 12.706204736,
    2: 4.30265273,
    3: 3.182446305,
    4: 2.776445105,
    5: 2.570581836,
    6: 2.446911851,
    7: 2.364624252,
    8: 2.306004135,
    9: 2.262157163,
    10: 2.228138852,
    11: 2.20098516,
    12: 2.17881283,
    13: 2.160368656,
    14: 2.144786688,
    15: 2.131449546,
    16: 2.119905299,
    17: 2.109815578,
    18: 2.10092204,
    19: 2.093024054,
    20: 2.085963447,
    21: 2.079613845,
    22: 2.073873068,
    23: 2.06865761,
    24: 2.063898562,
    25: 2.059538553,
    26: 2.055529439,
    27: 2.051830516,
    28: 2.048407142,
    29: 2.045229642,
    30: 2.042272456,
}


class ValidationError(ValueError):
    """Raised when an input cannot support the formal C5 comparison."""


@dataclass(frozen=True)
class RunSpec:
    system: str
    model: str
    seed: int
    path: Path
    round_manifest: Path | None = None


@dataclass
class ValidatedRun:
    system: str
    model: str
    seed: int
    path: Path
    scenario: str
    requests: List[Dict[str, Any]]
    request_map: List[Dict[str, Any]]
    request_map_sha256: str
    source_sha256: str
    trace_sha256: str
    adapter_subset_sha256: str
    metrics: Dict[str, float]


RUN_METRIC_FIELDS = (
    "dispatch_admission_mean_ms",
    "dispatch_admission_p95_ms",
    "service_ttft_mean_ms",
    "service_ttft_p95_ms",
    "decode_mean_ms",
    "decode_p95_ms",
    "overall_ttft_mean_ms",
    "overall_ttft_p95_ms",
    "service_e2e_mean_ms",
    "service_e2e_p95_ms",
    "e2e_mean_ms",
    "e2e_p95_ms",
    "tpot_mean_ms",
    "tpot_p95_ms",
)

# These are the stage and headline metrics for which paired, seed-level
# differences and t confidence intervals are emitted.
PAIRED_METRIC_FIELDS = (
    "dispatch_admission_mean_ms",
    "service_ttft_mean_ms",
    "decode_mean_ms",
    "overall_ttft_mean_ms",
    "e2e_mean_ms",
    "tpot_mean_ms",
    "service_ttft_p95_ms",
    "e2e_p95_ms",
)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationError(f"result does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"result root must be an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == SHA256_LENGTH and all(char in "0123456789abcdef" for char in text)


def _finite_number(value: Any, context: str, *, nonnegative: bool = True) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{context} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{context} must be finite, got {result!r}")
    if nonnegative and result < 0.0:
        raise ValidationError(f"{context} must be non-negative, got {result!r}")
    return result


def _integer(value: Any, context: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise ValidationError(f"{context} must be an integer, got {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{context} must be an integer, got {value!r}") from exc
    try:
        if float(value) != float(result):
            raise ValidationError(f"{context} must be an integer, got {value!r}")
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{context} must be an integer, got {value!r}") from exc
    if positive and result <= 0:
        raise ValidationError(f"{context} must be positive, got {result}")
    return result


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValidationError("cannot compute a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (float(percentile) / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _mean_ci95(values: Sequence[float]) -> Dict[str, float | int | None]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValidationError("cannot compute a confidence interval without seeds")
    mean_value = statistics.fmean(samples)
    if len(samples) < 2:
        return {
            "n": len(samples),
            "mean": mean_value,
            "stddev": None,
            "ci95_half_width": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    stddev = statistics.stdev(samples)
    df = len(samples) - 1
    critical = T975.get(df, 1.959963985)
    half_width = critical * stddev / math.sqrt(len(samples))
    return {
        "n": len(samples),
        "mean": mean_value,
        "stddev": stddev,
        "ci95_half_width": half_width,
        "ci95_low": mean_value - half_width,
        "ci95_high": mean_value + half_width,
    }


def _normalize_system(system: str) -> str:
    key = str(system).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "prime": "prime",
        "primelora": "prime",
        "faaslora": "prime",
        "slora": "slora",
    }
    if key not in aliases:
        raise ValidationError(
            f"unsupported C5 system {system!r}; expected prime/faaslora or slora"
        )
    return aliases[key]


def _extract_result(
    payload: Mapping[str, Any], system: str, source: Path
) -> Tuple[str, Mapping[str, Any], List[Dict[str, Any]]]:
    detailed = payload.get("detailed_results")
    if isinstance(detailed, Mapping):
        if len(detailed) != 1:
            raise ValidationError(
                f"{source}: formal C5 input must contain exactly one scenario; "
                f"found {list(detailed)}"
            )
        scenario, raw = next(iter(detailed.items()))
        if not isinstance(raw, Mapping):
            raise ValidationError(f"{source}: detailed_results[{scenario!r}] is not an object")
        requests = raw.get("requests")
        if not isinstance(requests, list):
            raise ValidationError(f"{source}: scenario {scenario!r} has no request list")
        return str(scenario), raw, [dict(item) for item in requests if isinstance(item, Mapping)]

    # The S-LoRA replay file, before conversion to the shared summary schema,
    # is also accepted.  PrimeLoRA formal results always use detailed_results.
    raw_results = payload.get("results")
    if system == "slora" and isinstance(raw_results, list):
        if any(not isinstance(item, Mapping) for item in raw_results):
            raise ValidationError(f"{source}: replay results contain non-object records")
        ok = sum(1 for item in raw_results if bool(item.get("success")))
        raw = {
            "total": payload.get("expected_requests", len(raw_results)),
            "completed": ok,
            "failed": len(raw_results) - ok,
            "requests": raw_results,
        }
        return str(payload.get("label") or "slora_replay"), raw, [dict(item) for item in raw_results]

    raise ValidationError(
        f"{source}: expected one detailed_results scenario"
        + (" or a raw S-LoRA replay results list" if system == "slora" else "")
    )


def _declared_contract(payload: Mapping[str, Any], requests: Sequence[Mapping[str, Any]]) -> str:
    candidates = [
        payload.get("generation_contract"),
        (payload.get("metadata") or {}).get("generation_contract")
        if isinstance(payload.get("metadata"), Mapping)
        else None,
    ]
    candidates.extend(request.get("generation_contract") for request in requests[:1])
    return str(next((value for value in candidates if value not in (None, "")), ""))


def _declared_seed(payload: Mapping[str, Any]) -> int | None:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    candidates = (
        payload.get("generation_seed"),
        metadata.get("generation_seed"),
        metadata.get("sampling_seed"),
    )
    for value in candidates:
        if value not in (None, ""):
            return _integer(value, "declared generation/sampling seed")
    return None


def _canonical_model_identity(value: Any) -> str:
    text = str(value or "").strip().lower()
    compact = "".join(character for character in text if character.isalnum())
    if "llama2" in compact and "7b" in compact:
        return "llama2_7b"
    if ("llama32" in compact or "llama3" in compact) and "3b" in compact:
        return "llama32_3b"
    return ""


def _declared_model(payload: Mapping[str, Any]) -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    for value in (
        metadata.get("model_profile"),
        metadata.get("model_name"),
        metadata.get("model"),
        payload.get("model_profile"),
        payload.get("model_name"),
    ):
        identity = _canonical_model_identity(value)
        if identity:
            return identity
    return ""


def _metadata_hash(payload: Mapping[str, Any], key: str) -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    value = metadata.get(key) or payload.get(key)
    return str(value or "")


def _assert_no_fallback_counts(payload: Mapping[str, Any], source: Path) -> None:
    roots: List[Any] = [payload.get("metadata"), payload.get("scenario_summaries")]
    forbidden_labels = {
        "trace_expected",
        "local_generated_text",
        "local_generated_text_empty",
    }

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if str(key).endswith("_source_counts") and isinstance(child, Mapping):
                    for label in forbidden_labels:
                        count = child.get(label, 0)
                        if count not in (None, 0, 0.0, "0"):
                            raise ValidationError(
                                f"{source}: forbidden fallback count {child_path}[{label!r}]={count}"
                            )
                visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    for root in roots:
        visit(root, "")


def _request_map_row(request: Mapping[str, Any], system: str) -> Dict[str, Any]:
    arrival_field = "scheduled_arrival_offset_s" if system == "prime" else "arrival_time_s"
    prompt_tokens_field = "canonical_prompt_tokens" if system == "prime" else "guard_prompt_tokens"
    return {
        "request_id": str(request.get("request_id") or ""),
        "adapter_id": str(request.get("adapter_id") or ""),
        "arrival_time_s": _finite_number(
            request.get(arrival_field),
            f"request {request.get('request_id')!r} {arrival_field}",
        ),
        "source_expected_output_tokens": _integer(
            request.get("source_expected_output_tokens"),
            f"request {request.get('request_id')!r} source_expected_output_tokens",
            positive=True,
        ),
        "requested_completion_tokens": _integer(
            request.get("requested_completion_tokens"),
            f"request {request.get('request_id')!r} requested_completion_tokens",
            positive=True,
        ),
        "canonical_prompt_sha256": str(request.get("canonical_prompt_sha256") or ""),
        "canonical_prompt_tokens": _integer(
            request.get(prompt_tokens_field),
            f"request {request.get('request_id')!r} {prompt_tokens_field}",
            positive=True,
        ),
    }


def _request_actual_tokens(request: Mapping[str, Any], system: str) -> int:
    value = request.get("completion_tokens")
    if value is None and system == "prime":
        value = request.get("output_tokens")
    return _integer(
        value,
        f"request {request.get('request_id')!r} actual completion tokens",
        positive=True,
    )


def _validate_request(
    request: Mapping[str, Any],
    *,
    system: str,
    fixed_output_cap: int,
    fixed_prompt_cap: int,
    tolerance_ms: float,
) -> Tuple[Dict[str, Any], Dict[str, float | None]]:
    request_id = str(request.get("request_id") or "")
    prefix = f"{SYSTEM_LABELS[system]} request {request_id or '<missing>'}"
    if not request_id:
        raise ValidationError(f"{prefix}: request_id is missing")
    if not str(request.get("adapter_id") or ""):
        raise ValidationError(f"{prefix}: adapter_id is missing")
    if request.get("success") is not True:
        raise ValidationError(f"{prefix}: request is not successful")
    if request.get("error") not in (None, ""):
        raise ValidationError(f"{prefix}: error/fallback record is non-empty")
    if request.get("final_empty_success") is True:
        raise ValidationError(f"{prefix}: final_empty_success fallback was used")
    if request.get("output_contract_match") is not True:
        raise ValidationError(f"{prefix}: output_contract_match is not true")
    if str(request.get("generation_contract") or "") != CONTRACT:
        raise ValidationError(f"{prefix}: per-request generation contract is not {CONTRACT}")

    row = _request_map_row(request, system)
    if not _is_sha256(row["canonical_prompt_sha256"]):
        raise ValidationError(f"{prefix}: canonical_prompt_sha256 is invalid")
    if not _is_sha256(request.get("completion_token_ids_sha256")):
        raise ValidationError(f"{prefix}: completion_token_ids_sha256 is invalid")
    if int(row["canonical_prompt_tokens"]) > fixed_prompt_cap:
        raise ValidationError(
            f"{prefix}: canonical prompt has {row['canonical_prompt_tokens']} tokens, "
            f"exceeding fixed cap {fixed_prompt_cap}"
        )

    source_target = int(row["source_expected_output_tokens"])
    requested = int(row["requested_completion_tokens"])
    expected_target = min(source_target, fixed_output_cap)
    actual = _request_actual_tokens(request, system)
    if requested != expected_target:
        raise ValidationError(
            f"{prefix}: target is not min(source_expected_output_tokens, {fixed_output_cap}); "
            f"source={source_target}, requested={requested}, expected={expected_target}"
        )
    if actual != requested:
        raise ValidationError(
            f"{prefix}: actual_tokens != target_tokens ({actual} != {requested})"
        )

    completion_source = str(request.get("completion_token_source") or "")
    expected_source = "vllm_token_ids" if system == "prime" else "slora_native_sse_token_id"
    if completion_source != expected_source:
        raise ValidationError(
            f"{prefix}: completion_token_source={completion_source!r}, expected {expected_source!r}"
        )
    if str(request.get("prompt_token_source") or "") in {
        "trace_expected",
        "local_generated_text",
        "local_generated_text_empty",
    }:
        raise ValidationError(f"{prefix}: forbidden prompt-token fallback source")
    if system == "slora":
        invalid_ids = _integer(
            request.get("native_sse_invalid_token_id_count", 0),
            f"{prefix} native_sse_invalid_token_id_count",
        )
        integer_ids = _integer(
            request.get("native_sse_integer_token_id_count"),
            f"{prefix} native_sse_integer_token_id_count",
        )
        if invalid_ids != 0 or integer_ids != actual:
            raise ValidationError(
                f"{prefix}: native SSE token-id audit failed; "
                f"integer={integer_ids}, invalid={invalid_ids}, actual={actual}"
            )

    dispatch = _finite_number(request.get("dispatch_admission_wait_ms"), f"{prefix} dispatch")
    service_ttft = _finite_number(request.get("service_ttft_ms"), f"{prefix} service TTFT")
    service_e2e = _finite_number(request.get("service_e2e_ms"), f"{prefix} service E2E")
    e2e = _finite_number(
        request.get("overall_e2e_ms", request.get("e2e_ms")), f"{prefix} overall E2E"
    )
    overall_ttft = _finite_number(
        request.get("overall_ttft_ms", request.get("ttft_ms")), f"{prefix} overall TTFT"
    )
    tpot = _finite_number(request.get("tpot_ms"), f"{prefix} TPOT")
    if service_e2e + tolerance_ms < service_ttft:
        raise ValidationError(f"{prefix}: service E2E is smaller than service TTFT")
    decode = max(0.0, service_e2e - service_ttft)
    e2e_error = abs(e2e - (dispatch + service_e2e))
    expanded_error = abs(e2e - (dispatch + service_ttft + decode))
    ttft_error = abs(overall_ttft - (dispatch + service_ttft))
    if max(e2e_error, expanded_error, ttft_error) > tolerance_ms:
        raise ValidationError(
            f"{prefix}: latency decomposition error exceeds {tolerance_ms:g} ms "
            f"(E2E={e2e_error:.6f}, expanded={expanded_error:.6f}, TTFT={ttft_error:.6f})"
        )
    observed_tpot: float | None = None
    if actual > 1:
        recomputed_tpot = decode / (actual - 1)
        if abs(tpot - recomputed_tpot) > tolerance_ms:
            raise ValidationError(
                f"{prefix}: TPOT recomputation error exceeds {tolerance_ms:g} ms "
                f"(observed={tpot:.6f}, recomputed={recomputed_tpot:.6f})"
            )
        if request.get("tpot_observed") is False:
            raise ValidationError(f"{prefix}: TPOT is not marked observed")
        observed_tpot = tpot
    elif abs(tpot) > tolerance_ms:
        raise ValidationError(f"{prefix}: one-token request must not report a decode TPOT")

    return row, {
        "dispatch_admission_ms": dispatch,
        "service_ttft_ms": service_ttft,
        "decode_ms": decode,
        "overall_ttft_ms": overall_ttft,
        "service_e2e_ms": service_e2e,
        "e2e_ms": e2e,
        "tpot_ms": observed_tpot,
    }


def validate_run(
    spec: RunSpec,
    *,
    expected_requests: int = 4000,
    fixed_output_cap: int = 256,
    fixed_prompt_cap: int = 759,
    tolerance_ms: float = 1.0,
) -> ValidatedRun:
    system = _normalize_system(spec.system)
    source = spec.path.resolve()
    payload = _read_json(source)
    scenario, result, requests = _extract_result(payload, system, source)
    if len(requests) != len(result.get("requests") or []):
        raise ValidationError(f"{source}: one or more request records are not JSON objects")

    total = _integer(result.get("total", payload.get("expected_requests")), f"{source} total")
    completed = _integer(
        result.get("completed", payload.get("completed_records")), f"{source} completed"
    )
    failed = _integer(result.get("failed", total - completed), f"{source} failed")
    if total != expected_requests or completed != expected_requests or failed != 0:
        raise ValidationError(
            f"{source}: incomplete formal run; total={total}, completed={completed}, "
            f"failed={failed}, expected={expected_requests}"
        )
    if len(requests) != expected_requests:
        raise ValidationError(
            f"{source}: request-record count={len(requests)}, expected={expected_requests}"
        )
    contract = _declared_contract(payload, requests)
    if contract != CONTRACT:
        raise ValidationError(
            f"{source}: generation contract={contract!r}, expected {CONTRACT!r}"
        )
    declared_seed = _declared_seed(payload)
    if declared_seed is None:
        raise ValidationError(f"{source}: result does not declare a generation/sampling seed")
    if declared_seed != int(spec.seed):
        raise ValidationError(
            f"{source}: declared seed={declared_seed}, RunSpec seed={spec.seed}"
        )
    declared_model = _declared_model(payload)
    expected_model = _canonical_model_identity(spec.model)
    if not declared_model:
        raise ValidationError(f"{source}: result does not declare a recognized model identity")
    if not expected_model:
        raise ValidationError(f"RunSpec model is not a recognized C5 model: {spec.model!r}")
    if declared_model != expected_model:
        raise ValidationError(
            f"{source}: declared model={declared_model}, RunSpec model={expected_model}"
        )
    _assert_no_fallback_counts(payload, source)

    trace_sha = _metadata_hash(payload, "shared_trace_sha256")
    subset_sha = _metadata_hash(payload, "shared_adapter_subset_sha256")
    if not _is_sha256(trace_sha):
        raise ValidationError(f"{source}: missing/invalid shared_trace_sha256")
    if not _is_sha256(subset_sha):
        raise ValidationError(f"{source}: missing/invalid shared_adapter_subset_sha256")

    seen: set[str] = set()
    request_map: List[Dict[str, Any]] = []
    samples: Dict[str, List[float]] = {
        "dispatch_admission_ms": [],
        "service_ttft_ms": [],
        "decode_ms": [],
        "overall_ttft_ms": [],
        "service_e2e_ms": [],
        "e2e_ms": [],
        "tpot_ms": [],
    }
    for request in requests:
        row, stages = _validate_request(
            request,
            system=system,
            fixed_output_cap=fixed_output_cap,
            fixed_prompt_cap=fixed_prompt_cap,
            tolerance_ms=tolerance_ms,
        )
        request_id = str(row["request_id"])
        if request_id in seen:
            raise ValidationError(f"{source}: duplicate request_id={request_id!r}")
        seen.add(request_id)
        request_map.append(row)
        for key, value in stages.items():
            if value is not None:
                samples[key].append(value)

    metrics: Dict[str, float] = {}
    for source_key in samples:
        if not samples[source_key]:
            raise ValidationError(
                f"{source}: no observed samples are available for {source_key}"
            )
        metrics[f"{source_key.removesuffix('_ms')}_mean_ms"] = statistics.fmean(samples[source_key])
        metrics[f"{source_key.removesuffix('_ms')}_p95_ms"] = _percentile(samples[source_key], 95)

    # The generated names above intentionally match RUN_METRIC_FIELDS.
    missing = [field for field in RUN_METRIC_FIELDS if field not in metrics]
    if missing:
        raise AssertionError(f"internal metric-name mismatch: {missing}")

    return ValidatedRun(
        system=system,
        model=str(spec.model),
        seed=int(spec.seed),
        path=source,
        scenario=scenario,
        requests=requests,
        request_map=request_map,
        request_map_sha256=_canonical_sha256(request_map),
        source_sha256=_sha256_file(source),
        trace_sha256=trace_sha,
        adapter_subset_sha256=subset_sha,
        metrics=metrics,
    )


def _compare_request_maps(prime: ValidatedRun, slora: ValidatedRun) -> None:
    context = f"model={prime.model}, seed={prime.seed}"
    if prime.trace_sha256 != slora.trace_sha256:
        raise ValidationError(f"{context}: Prime/S-LoRA shared trace SHA-256 differs")
    if prime.adapter_subset_sha256 != slora.adapter_subset_sha256:
        raise ValidationError(f"{context}: Prime/S-LoRA adapter subset SHA-256 differs")
    if len(prime.request_map) != len(slora.request_map):
        raise ValidationError(f"{context}: Prime/S-LoRA request-map lengths differ")
    for index, (prime_row, slora_row) in enumerate(zip(prime.request_map, slora.request_map)):
        if prime_row != slora_row:
            differing = [
                key for key in prime_row if prime_row.get(key) != slora_row.get(key)
            ]
            raise ValidationError(
                f"{context}: request/adapter/arrival/target/prompt map differs at "
                f"index={index}, request_id={prime_row.get('request_id')!r}, fields={differing}"
            )
    if prime.request_map_sha256 != slora.request_map_sha256:
        raise AssertionError(f"{context}: equal maps unexpectedly have different hashes")


def validate_pairs(runs: Sequence[ValidatedRun]) -> Dict[Tuple[str, int], Dict[str, ValidatedRun]]:
    grouped: Dict[Tuple[str, int], Dict[str, ValidatedRun]] = {}
    for run in runs:
        key = (run.model, run.seed)
        systems = grouped.setdefault(key, {})
        if run.system in systems:
            raise ValidationError(
                f"duplicate {run.system} run for model={run.model}, seed={run.seed}"
            )
        systems[run.system] = run
    if not grouped:
        raise ValidationError("no C5 runs were supplied")
    for (model, seed), systems in sorted(grouped.items()):
        if set(systems) != {"prime", "slora"}:
            raise ValidationError(
                f"model={model}, seed={seed}: expected exactly Prime and S-LoRA, "
                f"found {sorted(systems)}"
            )
        _compare_request_maps(systems["prime"], systems["slora"])

    model_seeds: Dict[str, set[int]] = {}
    for model, seed in grouped:
        model_seeds.setdefault(model, set()).add(seed)
    # Seed sets need not be identical across 7B and 3B, but each model must use
    # unique, paired seeds.  The manifest records the exact set.
    return grouped


def _per_run_rows(runs: Sequence[ValidatedRun]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in sorted(runs, key=lambda item: (item.model, item.seed, item.system)):
        row: Dict[str, Any] = {
            "model": run.model,
            "seed": run.seed,
            "system": run.system,
            "system_label": SYSTEM_LABELS[run.system],
            "scenario": run.scenario,
            "completed": len(run.requests),
            "generation_contract": CONTRACT,
            "request_map_sha256": run.request_map_sha256,
            "shared_trace_sha256": run.trace_sha256,
            "shared_adapter_subset_sha256": run.adapter_subset_sha256,
            "source": str(run.path),
            "source_sha256": run.source_sha256,
        }
        row.update(run.metrics)
        rows.append(row)
    return rows


def _paired_rows(
    grouped: Mapping[Tuple[str, int], Mapping[str, ValidatedRun]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (model, seed), systems in sorted(grouped.items()):
        for metric in PAIRED_METRIC_FIELDS:
            prime = float(systems["prime"].metrics[metric])
            slora = float(systems["slora"].metrics[metric])
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "metric": metric,
                    "prime": prime,
                    "slora": slora,
                    "prime_minus_slora": prime - slora,
                    "slora_minus_prime_improvement": slora - prime,
                    "prime_improvement_pct": ((slora - prime) / slora * 100.0) if slora else None,
                    "ci_unit": "seed",
                }
            )
    return rows


def _aggregate_rows(paired_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in paired_rows:
        grouped.setdefault((str(row["model"]), str(row["metric"])), []).append(row)
    output: List[Dict[str, Any]] = []
    for (model, metric), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["seed"]))
        seeds = [int(row["seed"]) for row in rows]
        prime_ci = _mean_ci95([float(row["prime"]) for row in rows])
        slora_ci = _mean_ci95([float(row["slora"]) for row in rows])
        diff_ci = _mean_ci95([float(row["prime_minus_slora"]) for row in rows])
        improvement_ci = _mean_ci95(
            [float(row["slora_minus_prime_improvement"]) for row in rows]
        )
        slora_mean = float(slora_ci["mean"])
        output.append(
            {
                "model": model,
                "metric": metric,
                "n_seeds": len(seeds),
                "seeds": ",".join(str(seed) for seed in seeds),
                "ci_unit": "seed",
                "ci_method": "paired two-sided Student-t 95% CI",
                "prime_mean": prime_ci["mean"],
                "prime_ci95_low": prime_ci["ci95_low"],
                "prime_ci95_high": prime_ci["ci95_high"],
                "slora_mean": slora_ci["mean"],
                "slora_ci95_low": slora_ci["ci95_low"],
                "slora_ci95_high": slora_ci["ci95_high"],
                "prime_minus_slora_mean": diff_ci["mean"],
                "prime_minus_slora_stddev": diff_ci["stddev"],
                "prime_minus_slora_ci95_half_width": diff_ci["ci95_half_width"],
                "prime_minus_slora_ci95_low": diff_ci["ci95_low"],
                "prime_minus_slora_ci95_high": diff_ci["ci95_high"],
                "slora_minus_prime_improvement_mean": improvement_ci["mean"],
                "slora_minus_prime_improvement_ci95_low": improvement_ci["ci95_low"],
                "slora_minus_prime_improvement_ci95_high": improvement_ci["ci95_high"],
                "prime_improvement_pct_of_slora_mean": (
                    float(improvement_ci["mean"]) / slora_mean * 100.0 if slora_mean else None
                ),
            }
        )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValidationError(f"refusing to write empty CSV: {path}")
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _prepare_output_dir(output_dir: Path) -> Path:
    output = output_dir.resolve()
    if output.exists():
        if not output.is_dir():
            raise ValidationError(f"output path exists and is not a directory: {output}")
        existing = list(output.iterdir())
        if existing:
            raise ValidationError(
                f"output directory is not fresh/empty: {output}; existing entries include "
                f"{[item.name for item in existing[:5]]}"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
    return output


def _plot_stage_decomposition(
    runs: Sequence[ValidatedRun], output_dir: Path
) -> Tuple[Path, Path]:
    models = sorted({run.model for run in runs})
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(max(5.4, 4.8 * len(models)), 4.2),
        squeeze=False,
    )
    stage_fields = (
        ("dispatch_admission_mean_ms", "Dispatch/admission", "#999999"),
        ("service_ttft_mean_ms", "Service TTFT", "#E69F00"),
        ("decode_mean_ms", "Decode", "#56B4E9"),
    )
    for axis, model in zip(axes[0], models):
        model_runs = [run for run in runs if run.model == model]
        systems = ("prime", "slora")
        x_positions = list(range(len(systems)))
        bottoms = [0.0, 0.0]
        for metric, label, color in stage_fields:
            heights = [
                statistics.fmean(
                    run.metrics[metric] for run in model_runs if run.system == system
                )
                for system in systems
            ]
            axis.bar(
                x_positions,
                heights,
                bottom=bottoms,
                label=label,
                color=color,
                edgecolor="white",
                linewidth=0.6,
            )
            bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]
        e2e_cis = [
            _mean_ci95(
                [
                    run.metrics["e2e_mean_ms"]
                    for run in model_runs
                    if run.system == system
                ]
            )
            for system in systems
        ]
        axis.errorbar(
            x_positions,
            [float(item["mean"]) for item in e2e_cis],
            yerr=[
                float(item["ci95_half_width"] or 0.0)
                for item in e2e_cis
            ],
            fmt="none",
            ecolor="black",
            capsize=4,
            linewidth=1.1,
            label="E2E seed-level 95% t-CI",
        )
        axis.set_xticks(x_positions, [SYSTEM_LABELS[system] for system in systems])
        axis.set_title(model)
        axis.set_ylabel("Latency (ms)")
        axis.grid(axis="y", alpha=0.25, linewidth=0.7)
        axis.set_axisbelow(True)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
    fig.suptitle("Matched-output E2E stage decomposition", y=1.03, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    pdf = output_dir / "c5_matched_output_stage_decomposition.pdf"
    png = output_dir / "c5_matched_output_stage_decomposition.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def analyze(
    specs: Sequence[RunSpec],
    *,
    output_dir: Path,
    expected_requests: int = 4000,
    expected_seeds: Sequence[int] | None = (43, 44, 45),
    fixed_output_cap: int = 256,
    fixed_prompt_cap: int = 759,
    tolerance_ms: float = 1.0,
) -> Dict[str, Any]:
    if expected_requests <= 0:
        raise ValidationError("expected_requests must be positive")
    if fixed_output_cap <= 0:
        raise ValidationError("fixed_output_cap must be positive")
    if fixed_prompt_cap <= 0:
        raise ValidationError("fixed_prompt_cap must be positive")
    if tolerance_ms < 0.0:
        raise ValidationError("tolerance_ms must be non-negative")

    # All validation happens before the output directory is created.  An input
    # failure therefore cannot leave a misleading partial publication bundle.
    runs = [
        validate_run(
            spec,
            expected_requests=expected_requests,
            fixed_output_cap=fixed_output_cap,
            fixed_prompt_cap=fixed_prompt_cap,
            tolerance_ms=tolerance_ms,
        )
        for spec in specs
    ]
    grouped = validate_pairs(runs)
    if expected_seeds is not None:
        required_seeds = sorted({int(seed) for seed in expected_seeds})
        if len(required_seeds) < 2:
            raise ValidationError(
                "formal paired confidence intervals require at least two expected seeds"
            )
        for model in sorted({model for model, _seed in grouped}):
            observed_seeds = sorted(seed for candidate, seed in grouped if candidate == model)
            if observed_seeds != required_seeds:
                raise ValidationError(
                    f"model={model}: incomplete formal seed matrix; "
                    f"observed={observed_seeds}, expected={required_seeds}"
                )
    output = _prepare_output_dir(output_dir)

    per_run_rows = _per_run_rows(runs)
    paired_rows = _paired_rows(grouped)
    aggregate_rows = _aggregate_rows(paired_rows)
    per_run_csv = output / "c5_per_run_metrics.csv"
    paired_csv = output / "c5_paired_seed_differences.csv"
    aggregate_csv = output / "c5_seed_level_ci.csv"
    _write_csv(per_run_csv, per_run_rows)
    _write_csv(paired_csv, paired_rows)
    _write_csv(aggregate_csv, aggregate_rows)
    figure_pdf, figure_png = _plot_stage_decomposition(runs, output)

    model_seed_sets = {
        model: sorted(seed for candidate, seed in grouped if candidate == model)
        for model in sorted({model for model, _seed in grouped})
    }
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "analysis": "eurosys27_v2_c5_matched_output",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_contract": CONTRACT,
        "validation_gates": {
            "expected_requests_per_run": expected_requests,
            "expected_seeds_per_model": (
                sorted({int(seed) for seed in expected_seeds})
                if expected_seeds is not None
                else None
            ),
            "completed_requests_per_run": expected_requests,
            "failed_requests_per_run": 0,
            "fixed_output_cap": fixed_output_cap,
            "fixed_prompt_cap": fixed_prompt_cap,
            "actual_tokens_equal_target_required": True,
            "native_token_id_source_required": {
                "prime": "vllm_token_ids",
                "slora": "slora_native_sse_token_id",
            },
            "cross_system_request_adapter_arrival_target_prompt_map_exact": True,
            "shared_trace_and_adapter_subset_sha_match_required": True,
            "latency_identity_tolerance_ms": tolerance_ms,
            "tpot_recompute_tolerance_ms": tolerance_ms,
            "fallback_allowed": False,
        },
        "statistical_method": {
            "replication_unit": "frozen workload seed",
            "request_records_are_independent_repetitions": False,
            "paired_difference": "PrimeLoRA minus S-LoRA for the same model and seed",
            "confidence_interval": "two-sided 95% Student-t interval over paired seed differences",
            "model_seed_sets": model_seed_sets,
        },
        "runs": per_run_rows,
        "paired_seed_statistics": aggregate_rows,
        "artifacts": {
            "per_run_metrics_csv": per_run_csv.name,
            "paired_seed_differences_csv": paired_csv.name,
            "seed_level_ci_csv": aggregate_csv.name,
            "stage_decomposition_pdf": figure_pdf.name,
            "stage_decomposition_png": figure_png.name,
        },
    }
    manifest_path = output / "c5_matched_output_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _canonical_model_from_profile(profile: str) -> str:
    text = str(profile).strip().lower()
    aliases = (
        ("llama32_3b", "llama32_3b"),
        ("llama3.2_3b", "llama32_3b"),
        ("llama2_7b", "llama2_7b"),
    )
    for needle, label in aliases:
        if needle in text:
            return label
    return str(profile).strip()


def specs_from_round(round_dir: Path) -> List[RunSpec]:
    round_path = round_dir.resolve()
    manifest_path = round_path / "MANIFEST.json"
    manifest = _read_json(manifest_path)
    if str(manifest.get("status") or "") != "complete":
        raise ValidationError(
            f"round manifest is incomplete: {manifest_path} status={manifest.get('status')!r}"
        )
    if str(manifest.get("generation_contract") or "") != CONTRACT:
        raise ValidationError(
            f"round manifest generation contract is not {CONTRACT}: {manifest_path}"
        )
    model = _canonical_model_from_profile(str(manifest.get("model_profile") or ""))
    if not model:
        raise ValidationError(f"round manifest has no model_profile: {manifest_path}")
    seed = _integer(manifest.get("sampling_seed"), f"{manifest_path} sampling_seed")

    prime_candidates = sorted((round_path / "raw" / "faaslora").glob("*result.json"))
    slora_candidates = sorted((round_path / "raw" / "replay").glob("*slora*summary.json"))
    if len(prime_candidates) != 1:
        raise ValidationError(
            f"{round_path}: expected one PrimeLoRA result, found {prime_candidates}"
        )
    if len(slora_candidates) != 1:
        raise ValidationError(
            f"{round_path}: expected one S-LoRA summary, found {slora_candidates}"
        )
    return [
        RunSpec("prime", model, seed, prime_candidates[0], manifest_path),
        RunSpec("slora", model, seed, slora_candidates[0], manifest_path),
    ]


def _parse_run_specs(values: Iterable[Sequence[str]]) -> List[RunSpec]:
    specs: List[RunSpec] = []
    for system, model, seed_text, path_text in values:
        specs.append(
            RunSpec(
                system=_normalize_system(system),
                model=str(model),
                seed=_integer(seed_text, f"--run {model} seed"),
                path=Path(path_text),
            )
        )
    return specs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--round-dir",
        action="append",
        default=[],
        type=Path,
        help="Completed fair-round directory; repeat once per model/seed.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        nargs=4,
        metavar=("SYSTEM", "MODEL", "SEED", "PATH"),
        help="Explicit run specification; SYSTEM is prime/faaslora or slora.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-requests", type=int, default=4000)
    parser.add_argument(
        "--expected-seeds",
        default="43,44,45",
        help="Comma-separated formal seed set. Use 'any' only for non-publication diagnostics.",
    )
    parser.add_argument("--fixed-output-cap", type=int, default=256)
    parser.add_argument("--fixed-prompt-cap", type=int, default=759)
    parser.add_argument("--identity-tolerance-ms", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    specs = _parse_run_specs(args.run)
    for round_dir in args.round_dir:
        specs.extend(specs_from_round(round_dir))
    if not specs:
        raise SystemExit("at least one --round-dir or --run is required")
    if str(args.expected_seeds).strip().lower() == "any":
        expected_seeds = None
    else:
        try:
            expected_seeds = [
                int(item.strip())
                for item in str(args.expected_seeds).split(",")
                if item.strip()
            ]
        except ValueError as exc:
            raise SystemExit("--expected-seeds must be comma-separated integers or 'any'") from exc
    try:
        manifest = analyze(
            specs,
            output_dir=args.output_dir,
            expected_requests=args.expected_requests,
            expected_seeds=expected_seeds,
            fixed_output_cap=args.fixed_output_cap,
            fixed_prompt_cap=args.fixed_prompt_cap,
            tolerance_ms=args.identity_tolerance_ms,
        )
    except ValidationError as exc:
        raise SystemExit(f"C5 validation failed: {exc}") from exc
    print(
        f"C5 matched-output analysis complete: models={list(manifest['statistical_method']['model_seed_sets'])} "
        f"output={args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
