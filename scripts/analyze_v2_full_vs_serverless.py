#!/usr/bin/env python3
"""Formal PrimeLoRA Full versus ServerlessLLM-new publication analyzer.

Formal mode is intentionally fail-closed: inputs must be completed held-out
campaign manifests, the matrix and frozen workload axes must match exactly,
and all statistics use the seed-level paired difference as their unit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

try:
    from eurosys27_v2_provenance import (
        FormalAnalysisIdentity,
        build_formal_provenance_index,
        validate_formal_analysis_sources,
    )
    from plot_paper_figures import _v2_mean_ci95
    from plot_paper_sensitivity import (
        V2SensitivityObservation,
        _sha256_field,
        load_v2_sensitivity_observations,
    )
except ImportError:  # Package imports used by tests.
    from scripts.eurosys27_v2_provenance import (
        FormalAnalysisIdentity,
        build_formal_provenance_index,
        validate_formal_analysis_sources,
    )
    from scripts import plot_paper_figures as _plot_paper_figures

    sys.modules.setdefault("plot_paper_figures", _plot_paper_figures)
    from scripts.plot_paper_figures import _v2_mean_ci95
    from scripts.plot_paper_sensitivity import (
        V2SensitivityObservation,
        _sha256_field,
        load_v2_sensitivity_observations,
    )


FORMAL_MODELS = ("llama2_7b", "llama32_3b")
FORMAL_SEEDS = (43, 44, 45)
FORMAL_SYSTEMS = ("faaslora", "serverlessllm")
FORMAL_CAMPAIGN_KIND = "v2_full_vs_serverless"
FORMAL_EXECUTION_ORDERS = {
    ("llama2_7b", 43): ("faaslora", "serverlessllm"),
    ("llama2_7b", 44): ("serverlessllm", "faaslora"),
    ("llama2_7b", 45): ("faaslora", "serverlessllm"),
    ("llama32_3b", 43): ("serverlessllm", "faaslora"),
    ("llama32_3b", 44): ("faaslora", "serverlessllm"),
    ("llama32_3b", 45): ("serverlessllm", "faaslora"),
}
FORMAL_MODEL_PROFILES = {
    "llama2_7b": "llama2_7b_main_v2_publicmix",
    "llama32_3b": "llama32_3b_main_modelscope",
}
FORMAL_SCENARIOS = {
    "faaslora": "v2_full",
    # The official ServerlessLLM-new harness deliberately retains the
    # historical result-schema scenario name.  "new" identifies the pinned
    # implementation path/system, not a second scenario row.
    "serverlessllm": "serverlessllm_fair",
}
METRICS = (
    "ttft_avg_ms",
    "ttft_p95_ms",
    "e2e_avg_ms",
    "e2e_p95_ms",
    "tpot_avg_ms",
    "tpot_p95_ms",
    "tok_s",
    "cost_req_usd",
    "ce",
)
HIGHER_IS_BETTER = {"tok_s", "ce"}
CE_RELATIVE_TOLERANCE = 0.005


@dataclass(frozen=True)
class FormalRun:
    model: str
    seed: int
    system: str
    scenario: str
    source: Path
    metadata_sampling_seed: int
    total: int
    completed: int
    generation_contract: str
    selected_num_adapters: int | None
    bandwidth_mib_s: float | None
    configured_time_scale_factor: float | None
    effective_time_scale_factor: float | None
    zipf_exponent: float | None
    active_adapter_cap: int | None
    hotset_rotation_requests: int | None
    hotset_rotation_mode: str
    hotset_overlap_fraction: float | None
    trace_sha256: str
    adapter_subset_sha256: str
    system_resolved_config_sha256: str
    metrics: Mapping[str, float]


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None and str(value).strip() != "":
            return value
    return None


def _sha(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise SystemExit(f"formal Full-vs-Serverless {label}: invalid SHA-256")
    return digest


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(
            f"formal Full-vs-Serverless {label}: cannot read JSON {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SystemExit(
            f"formal Full-vs-Serverless {label}: JSON root must be an object: {path}"
        )
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _integer(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"formal Full-vs-Serverless {label}: expected an integer, "
            f"observed {value!r}"
        ) from exc


def _require_number(value: Any, expected: float, label: str) -> None:
    try:
        observed = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"formal Full-vs-Serverless {label}: expected {expected:g}, "
            f"observed {value!r}"
        ) from exc
    if not math.isfinite(observed) or not math.isclose(
        observed, float(expected), rel_tol=0.0, abs_tol=1e-9
    ):
        raise SystemExit(
            f"formal Full-vs-Serverless {label}: expected {expected:g}, "
            f"observed {observed!r}"
        )


def _require_text(value: Any, expected: str, label: str) -> None:
    observed = str(value or "").strip().lower()
    if observed != expected:
        raise SystemExit(
            f"formal Full-vs-Serverless {label}: expected {expected!r}, "
            f"observed {observed!r}"
        )


def _read_trace_axes(observation: V2SensitivityObservation) -> Dict[str, Any]:
    raw_path = _first(
        observation.metadata.get("shared_trace_path"),
        observation.metadata.get("trace_source"),
        observation.manifest.get("shared_trace_path"),
    )
    if raw_path is None:
        return {}
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = observation.source.parent / path
    if not path.is_file():
        raise SystemExit(
            f"formal Full-vs-Serverless {observation.source}: shared trace is "
            f"missing: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"cannot read shared trace axes from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"shared trace must be a JSON object: {path}")
    return payload


def _exact_model(model: str) -> str:
    if model in FORMAL_MODELS:
        return model
    raise SystemExit(
        "formal Full-vs-Serverless unsupported model "
        f"{model!r}; expected {FORMAL_MODELS}"
    )


def _run_from_observation(observation: V2SensitivityObservation) -> FormalRun:
    trace = _read_trace_axes(observation)
    metadata = observation.metadata
    manifest = observation.manifest
    profile = observation.profile
    contract_values = {
        str(value).strip().lower()
        for value in (
            metadata.get("generation_contract"),
            manifest.get("generation_contract"),
        )
        if value is not None and str(value).strip()
    }
    if len(contract_values) != 1:
        raise SystemExit(
            f"formal Full-vs-Serverless {observation.source}: missing/conflicting "
            f"generation contract {sorted(contract_values)}"
        )
    if (
        metadata.get("sampling_seed") is None
        or str(metadata.get("sampling_seed")).strip() == ""
    ):
        raise SystemExit(
            f"formal Full-vs-Serverless {observation.source}: raw result metadata "
            "must record sampling_seed"
        )
    metadata_sampling_seed = _integer(
        metadata.get("sampling_seed"),
        f"{observation.source}.metadata.sampling_seed",
    )
    for field in ("workload_seed", "generation_seed", "seed"):
        raw_seed = metadata.get(field)
        if raw_seed is None or str(raw_seed).strip() == "":
            continue
        if (
            _integer(raw_seed, f"{observation.source}.metadata.{field}")
            != metadata_sampling_seed
        ):
            raise SystemExit(
                f"formal Full-vs-Serverless {observation.source}: metadata.{field} "
                "conflicts with metadata.sampling_seed"
            )
    config_sha = _sha(
        _first(
            metadata.get("system_resolved_config_sha256"),
            manifest.get("system_resolved_config_sha256"),
        ),
        f"{observation.source}.system_resolved_config_sha256",
    )
    return FormalRun(
        model=_exact_model(observation.model),
        seed=int(observation.seed),
        system=observation.system_key,
        scenario=observation.scenario,
        source=observation.source.resolve(),
        metadata_sampling_seed=metadata_sampling_seed,
        total=observation.total,
        completed=observation.completed,
        generation_contract=next(iter(contract_values)),
        selected_num_adapters=(
            int(
                _first(
                    metadata.get("selected_num_adapters"),
                    metadata.get("num_adapters"),
                    manifest.get("selected_num_adapters"),
                    profile.get("nominal_adapter_pool_size"),
                )
            )
            if _first(
                metadata.get("selected_num_adapters"),
                metadata.get("num_adapters"),
                manifest.get("selected_num_adapters"),
                profile.get("nominal_adapter_pool_size"),
            )
            is not None
            else None
        ),
        bandwidth_mib_s=(
            float(
                _first(metadata.get("bandwidth_mib_s"), manifest.get("bandwidth_mib_s"))
            )
            if _first(metadata.get("bandwidth_mib_s"), manifest.get("bandwidth_mib_s"))
            is not None
            else None
        ),
        configured_time_scale_factor=(
            float(
                _first(
                    metadata.get("configured_time_scale_factor"),
                    metadata.get("shared_trace_configured_time_scale_factor"),
                    trace.get("configured_time_scale_factor"),
                )
            )
            if _first(
                metadata.get("configured_time_scale_factor"),
                metadata.get("shared_trace_configured_time_scale_factor"),
                trace.get("configured_time_scale_factor"),
            )
            is not None
            else None
        ),
        effective_time_scale_factor=(
            float(
                _first(
                    metadata.get("effective_time_scale_factor"),
                    metadata.get("shared_trace_effective_time_scale_factor"),
                    trace.get("effective_time_scale_factor"),
                )
            )
            if _first(
                metadata.get("effective_time_scale_factor"),
                metadata.get("shared_trace_effective_time_scale_factor"),
                trace.get("effective_time_scale_factor"),
            )
            is not None
            else None
        ),
        zipf_exponent=(
            float(profile.get("zipf_exponent"))
            if profile.get("zipf_exponent") is not None
            else None
        ),
        active_adapter_cap=(
            int(profile.get("active_adapter_cap"))
            if profile.get("active_adapter_cap") is not None
            else None
        ),
        hotset_rotation_requests=(
            int(profile.get("hotset_rotation_requests"))
            if profile.get("hotset_rotation_requests") is not None
            else None
        ),
        hotset_rotation_mode=str(profile.get("rotation_mode") or "").strip().lower(),
        hotset_overlap_fraction=(
            float(profile.get("hotset_overlap_fraction"))
            if profile.get("hotset_overlap_fraction") is not None
            else None
        ),
        trace_sha256=_sha256_field(metadata, manifest, "shared_trace_sha256"),
        adapter_subset_sha256=_sha256_field(
            metadata, manifest, "shared_adapter_subset_sha256"
        ),
        system_resolved_config_sha256=config_sha,
        metrics=dict(observation.metrics),
    )


def load_formal_runs(inputs: Sequence[Path]) -> List[FormalRun]:
    return [
        _run_from_observation(observation)
        for observation in load_v2_sensitivity_observations(inputs)
    ]


def _identity(run: FormalRun) -> tuple[str, int, str]:
    return run.model, run.seed, run.system


def validate_formal_matrix(runs: Sequence[FormalRun]) -> None:
    expected = {
        (model, seed, system)
        for model in FORMAL_MODELS
        for seed in FORMAL_SEEDS
        for system in FORMAL_SYSTEMS
    }
    counts: Dict[tuple[str, int, str], int] = defaultdict(int)
    for run in runs:
        counts[_identity(run)] += 1
    observed = set(counts)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    duplicate = sorted(identity for identity, count in counts.items() if count != 1)
    if missing or extra or duplicate:
        raise SystemExit(
            "formal Full-vs-Serverless matrix identity mismatch: "
            f"missing={missing}, extra={extra}, duplicate={duplicate}"
        )

    expected_axes: Mapping[str, Any] = {
        "total": 4000,
        "completed": 4000,
        "generation_contract": "legacy",
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
    config_hashes: Dict[str, set[str]] = defaultdict(set)
    pair_hashes: Dict[tuple[str, int], Dict[str, set[str]]] = defaultdict(
        lambda: {"trace": set(), "subset": set(), "config": set()}
    )
    for run in runs:
        expected_scenario = FORMAL_SCENARIOS.get(run.system)
        if run.scenario != expected_scenario:
            raise SystemExit(
                f"formal Full-vs-Serverless {run.source}: system={run.system} must use "
                f"scenario={expected_scenario}, observed={run.scenario}"
            )
        for name, expected_value in expected_axes.items():
            value = getattr(run, name)
            if isinstance(expected_value, float):
                matches = value is not None and math.isclose(
                    float(value), expected_value, rel_tol=0.0, abs_tol=1e-9
                )
            else:
                matches = value == expected_value
            if not matches:
                raise SystemExit(
                    f"formal Full-vs-Serverless {run.source}: wrong {name}; "
                    f"expected={expected_value!r}, observed={value!r}"
                )
        if run.metadata_sampling_seed != run.seed:
            raise SystemExit(
                f"formal Full-vs-Serverless {run.source}: result seed mismatch; "
                f"identity={run.seed}, "
                f"metadata.sampling_seed={run.metadata_sampling_seed}"
            )
        for metric in METRICS:
            value = run.metrics.get(metric)
            if value is None or not math.isfinite(float(value)):
                raise SystemExit(
                    f"formal Full-vs-Serverless {run.source}: missing metric {metric}"
                )
        e2e_ms = float(run.metrics["e2e_avg_ms"])
        cost = float(run.metrics["cost_req_usd"])
        reported_ce = float(run.metrics["ce"])
        if e2e_ms <= 0.0 or cost <= 0.0 or reported_ce <= 0.0:
            raise SystemExit(
                f"formal Full-vs-Serverless {run.source}: CE inputs must be positive; "
                f"e2e_avg_ms={e2e_ms}, cost_req_usd={cost}, ce={reported_ce}"
            )
        recomputed_ce = 1.0 / ((e2e_ms / 1000.0) * cost)
        if not math.isclose(
            reported_ce,
            recomputed_ce,
            rel_tol=CE_RELATIVE_TOLERANCE,
            abs_tol=1e-12,
        ):
            relative_error = abs(reported_ce - recomputed_ce) / recomputed_ce
            raise SystemExit(
                f"formal Full-vs-Serverless {run.source}: reported CE does not match "
                f"1/(E2E_seconds*cost/request); reported={reported_ce}, "
                f"recomputed={recomputed_ce}, relative_error={relative_error:.6%}, "
                f"tolerance={CE_RELATIVE_TOLERANCE:.3%}"
            )
        config_sha = _sha(run.system_resolved_config_sha256, str(run.source))
        # One fair-round sidecar hashes both systems.  It is frozen across all
        # three held-out seeds for one model, not independently per system.
        config_hashes[run.model].add(config_sha)
        pair_hashes[(run.model, run.seed)]["trace"].add(run.trace_sha256)
        pair_hashes[(run.model, run.seed)]["subset"].add(run.adapter_subset_sha256)
        pair_hashes[(run.model, run.seed)]["config"].add(config_sha)
    drift = {
        group: hashes for group, hashes in config_hashes.items() if len(hashes) != 1
    }
    if drift:
        raise SystemExit(
            f"formal Full-vs-Serverless frozen configuration drift: {drift}"
        )
    mismatched_pairs = {
        pair: values
        for pair, values in pair_hashes.items()
        if (
            len(values["trace"]) != 1
            or len(values["subset"]) != 1
            or len(values["config"]) != 1
        )
    }
    if mismatched_pairs:
        raise SystemExit(
            f"formal Full-vs-Serverless trace/subset mismatch: {mismatched_pairs}"
        )


def _manifest_paths_for_runs(
    provenance: Any,
    runs: Sequence[FormalRun],
) -> Dict[Path, Path]:
    manifest_by_source: Dict[Path, Path] = {}
    for run in runs:
        source = run.source.resolve()
        coverage = provenance.records_by_source.get(source, ())
        if len(coverage) != 1:
            manifests = [str(item.manifest_path) for item in coverage]
            raise SystemExit(
                f"formal Full-vs-Serverless {source}: expected unique manifest "
                f"coverage, observed={manifests}"
            )
        manifest_by_source[source] = Path(coverage[0].manifest_path).resolve()
    return manifest_by_source


def _resolve_manifest_sidecar(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> Path:
    raw_path = str(manifest.get("system_resolved_config_path") or "").strip()
    if not raw_path:
        raise SystemExit(
            f"formal Full-vs-Serverless {manifest_path}: missing "
            "system_resolved_config_path"
        )
    sidecar_path = Path(raw_path).expanduser()
    if not sidecar_path.is_absolute():
        sidecar_path = manifest_path.parent / sidecar_path
    sidecar_path = sidecar_path.resolve()
    expected_path = (
        manifest_path.parent / "protocol" / "system_resolved_config.json"
    ).resolve()
    if sidecar_path != expected_path:
        raise SystemExit(
            f"formal Full-vs-Serverless {manifest_path}: resolved-config sidecar "
            f"must be {expected_path}, observed={sidecar_path}"
        )
    if not sidecar_path.is_file():
        raise SystemExit(
            f"formal Full-vs-Serverless {manifest_path}: resolved-config sidecar "
            f"is missing: {sidecar_path}"
        )
    declared_sha = _sha(
        manifest.get("system_resolved_config_sidecar_sha256"),
        f"{manifest_path}.system_resolved_config_sidecar_sha256",
    )
    if _sha256_file(sidecar_path) != declared_sha:
        raise SystemExit(
            f"formal Full-vs-Serverless {manifest_path}: resolved-config sidecar "
            "SHA-256 does not match manifest"
        )
    declared_bytes = _integer(
        manifest.get("system_resolved_config_sidecar_bytes"),
        f"{manifest_path}.system_resolved_config_sidecar_bytes",
    )
    if sidecar_path.stat().st_size != declared_bytes:
        raise SystemExit(
            f"formal Full-vs-Serverless {manifest_path}: resolved-config sidecar "
            "byte count does not match manifest"
        )
    return sidecar_path


def _require_workload_axes(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise SystemExit(
            f"formal Full-vs-Serverless {label}: workload_overrides must be an object"
        )
    expected: Mapping[str, Any] = {
        "zipf_exponent": 1.0,
        "active_adapter_cap": 48,
        "hotset_rotation_requests": 500,
        "hotset_rotation_mode": "legacy",
        "hotset_overlap_fraction": 0.75,
    }
    for name, expected_value in expected.items():
        context = f"{label}.{name}"
        if isinstance(expected_value, str):
            _require_text(value.get(name), expected_value, context)
        else:
            _require_number(value.get(name), float(expected_value), context)


def _source_commits(value: Any, label: str) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        raise SystemExit(
            f"formal Full-vs-Serverless {label}: source_commits must be an object"
        )
    commits: Dict[str, str] = {}
    for key in ("baselines", "faaslora"):
        commit = str(value.get(key) or "").strip().lower()
        if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
            raise SystemExit(
                f"formal Full-vs-Serverless {label}.{key}: invalid Git commit"
            )
        commits[key] = commit
    return commits


def _validated_recorded_file(
    owner_path: Path,
    *,
    raw_path: Any,
    raw_bytes: Any,
    raw_sha256: Any,
    expected_path: Path | None,
    label: str,
) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise SystemExit(f"formal Full-vs-Serverless {label}: recorded path is empty")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = owner_path.parent / path
    path = path.resolve()
    if expected_path is not None and path != expected_path.resolve():
        raise SystemExit(
            f"formal Full-vs-Serverless {label}: expected path "
            f"{expected_path.resolve()}, "
            f"observed={path}"
        )
    if not path.is_file():
        raise SystemExit(
            f"formal Full-vs-Serverless {label}: recorded file is missing: {path}"
        )
    expected_bytes = _integer(raw_bytes, f"{label}.bytes")
    if path.stat().st_size != expected_bytes:
        raise SystemExit(
            f"formal Full-vs-Serverless {label}: byte count does not match current file"
        )
    expected_sha = _sha(raw_sha256, f"{label}.sha256")
    if _sha256_file(path) != expected_sha:
        raise SystemExit(
            f"formal Full-vs-Serverless {label}: SHA-256 does not match current file"
        )
    return path


def _validate_seed41_evidence(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    sidecar_path: Path,
    sidecar: Mapping[str, Any],
    model: str,
    heldout_seed: int,
    manifest_config: str,
    identity_sha: str,
) -> None:
    expected_evidence_path = (
        manifest_path.parent / "protocol" / "seed41_validation_evidence.json"
    ).resolve()
    evidence_path = _validated_recorded_file(
        manifest_path,
        raw_path=manifest.get("seed41_validation_evidence_path"),
        raw_bytes=manifest.get("seed41_validation_evidence_bytes"),
        raw_sha256=manifest.get("seed41_validation_evidence_sha256"),
        expected_path=expected_evidence_path,
        label=f"{manifest_path}.seed41_validation_evidence",
    )
    evidence = _read_json(evidence_path, "seed-41 validation evidence")
    if evidence.get("schema_version") != "eurosys27_v2_seed41_validation_evidence_v1":
        raise SystemExit(
            f"formal Full-vs-Serverless {evidence_path}: unsupported evidence schema"
        )

    family_id = _sha(
        sidecar.get("configuration_family_id"),
        f"{sidecar_path}.configuration_family_id",
    )
    heldout_commits = _source_commits(
        sidecar.get("source_commits"), f"{sidecar_path}.source_commits"
    )
    manifest_commits = {
        "baselines": str(
            (manifest.get("baseline_git") or {}).get("commit")
            if isinstance(manifest.get("baseline_git"), Mapping)
            else ""
        )
        .strip()
        .lower(),
        "faaslora": str(
            (manifest.get("faaslora_git") or {}).get("commit")
            if isinstance(manifest.get("faaslora_git"), Mapping)
            else ""
        )
        .strip()
        .lower(),
    }
    if manifest_commits != heldout_commits:
        raise SystemExit(
            f"formal Full-vs-Serverless {manifest_path}: manifest/sidecar source "
            "commit mismatch"
        )
    heldout = evidence.get("heldout")
    if not isinstance(heldout, Mapping):
        raise SystemExit(
            f"formal Full-vs-Serverless {evidence_path}: heldout must be an object"
        )
    heldout_expected: Mapping[str, Any] = {
        "sampling_seed": heldout_seed,
        "campaign_kind": FORMAL_CAMPAIGN_KIND,
        "model_profile": FORMAL_MODEL_PROFILES[model],
        "configuration_family_id": family_id,
        "system_resolved_config_sha256": manifest_config,
        "full_run_identity_sha256": identity_sha,
        "source_commits": heldout_commits,
    }
    for field, expected_value in heldout_expected.items():
        observed = heldout.get(field)
        if field == "sampling_seed":
            observed = _integer(observed, f"{evidence_path}.heldout.{field}")
        if observed != expected_value:
            raise SystemExit(
                f"formal Full-vs-Serverless {evidence_path}: heldout.{field} mismatch"
            )

    validation = evidence.get("seed41_validation")
    if not isinstance(validation, Mapping):
        raise SystemExit(
            f"formal Full-vs-Serverless {evidence_path}: "
            "seed41_validation must be an object"
        )
    if (
        _integer(
            validation.get("sampling_seed"),
            f"{evidence_path}.seed41_validation.sampling_seed",
        )
        != 41
        or _integer(
            validation.get("total_requests"),
            f"{evidence_path}.seed41_validation.total_requests",
        )
        != 1000
    ):
        raise SystemExit(
            f"formal Full-vs-Serverless {evidence_path}: validation must bind "
            "seed 41 and 1,000 requests"
        )
    expected_order = FORMAL_EXECUTION_ORDERS.get((model, 41))
    if expected_order is None:
        # The held-out table intentionally lists only 43--45.  Seed 41 follows
        # the same odd-seed alternation and 3B reversal used by the runner.
        expected_order = (
            ("faaslora", "serverlessllm")
            if model == "llama2_7b"
            else ("serverlessllm", "faaslora")
        )
    validation_expected: Mapping[str, Any] = {
        "campaign_kind": FORMAL_CAMPAIGN_KIND,
        "model_profile": FORMAL_MODEL_PROFILES[model],
        "configuration_family_id": family_id,
        "system_resolved_config_sha256": manifest_config,
        "systems": list(FORMAL_SYSTEMS),
        "execution_order": list(expected_order),
        "source_commits": heldout_commits,
    }
    for field, expected_value in validation_expected.items():
        observed = validation.get(field)
        if field == "systems" and isinstance(observed, list):
            if len(observed) == 2 and set(str(item) for item in observed) == set(
                FORMAL_SYSTEMS
            ):
                continue
        if observed != expected_value:
            raise SystemExit(
                f"formal Full-vs-Serverless {evidence_path}: "
                f"seed41_validation.{field} mismatch"
            )

    validation_sidecar_path = _validated_recorded_file(
        evidence_path,
        raw_path=validation.get("sidecar_path"),
        raw_bytes=validation.get("sidecar_bytes"),
        raw_sha256=validation.get("sidecar_sha256"),
        expected_path=None,
        label=f"{evidence_path}.seed41_validation.sidecar",
    )
    validation_manifest_path = _validated_recorded_file(
        evidence_path,
        raw_path=validation.get("manifest_path"),
        raw_bytes=validation.get("manifest_bytes"),
        raw_sha256=validation.get("manifest_sha256"),
        expected_path=validation_sidecar_path.parent.parent / "MANIFEST.json",
        label=f"{evidence_path}.seed41_validation.manifest",
    )
    if (
        validation_sidecar_path
        != (
            validation_manifest_path.parent / "protocol" / "system_resolved_config.json"
        ).resolve()
    ):
        raise SystemExit(
            f"formal Full-vs-Serverless {evidence_path}: validation sidecar is not "
            "the protocol sidecar of its recorded MANIFEST"
        )
    validation_sidecar = _read_json(
        validation_sidecar_path, "seed-41 resolved-config sidecar"
    )
    validation_manifest = _read_json(
        validation_manifest_path, "seed-41 campaign manifest"
    )
    if (
        validation_sidecar.get("formal_run") is not True
        or str(validation_sidecar.get("trace_role") or "").strip().lower()
        != "validation"
        or _integer(
            validation_sidecar.get("sampling_seed"),
            f"{validation_sidecar_path}.sampling_seed",
        )
        != 41
    ):
        raise SystemExit(
            f"formal Full-vs-Serverless {validation_sidecar_path}: invalid "
            "formal seed-41 validation identity"
        )
    validation_identity = validation_sidecar.get("full_run_identity")
    if (
        not isinstance(validation_identity, Mapping)
        or _integer(
            validation_identity.get("total_requests"),
            f"{validation_sidecar_path}.full_run_identity.total_requests",
        )
        != 1000
    ):
        raise SystemExit(
            f"formal Full-vs-Serverless {validation_sidecar_path}: validation "
            "full-run identity must record 1,000 requests"
        )
    for observed, expected_value, label in (
        (validation_sidecar.get("campaign_kind"), FORMAL_CAMPAIGN_KIND, "campaign"),
        (
            validation_sidecar.get("model_profile"),
            FORMAL_MODEL_PROFILES[model],
            "model",
        ),
        (validation_sidecar.get("configuration_family_id"), family_id, "family"),
        (
            validation_sidecar.get("system_resolved_config_sha256"),
            manifest_config,
            "config",
        ),
        (validation_sidecar.get("source_commits"), heldout_commits, "source commits"),
    ):
        if observed != expected_value:
            raise SystemExit(
                f"formal Full-vs-Serverless {validation_sidecar_path}: validation "
                f"{label} mismatch"
            )
    if len(validation_sidecar.get("systems") or []) != 2 or set(
        str(item) for item in (validation_sidecar.get("systems") or [])
    ) != set(FORMAL_SYSTEMS):
        raise SystemExit(
            f"formal Full-vs-Serverless {validation_sidecar_path}: validation "
            "systems mismatch"
        )
    if (
        tuple(str(item) for item in (validation_identity.get("execution_order") or []))
        != expected_order
    ):
        raise SystemExit(
            f"formal Full-vs-Serverless {validation_sidecar_path}: validation "
            "execution order mismatch"
        )

    if (
        validation_manifest.get("status") != "complete"
        or validation_manifest.get("formal_run") is not True
        or validation_manifest.get("source_clean_for_formal") is not True
        or str(validation_manifest.get("trace_role") or "").strip().lower()
        != "validation"
        or _integer(
            validation_manifest.get("sampling_seed"),
            f"{validation_manifest_path}.sampling_seed",
        )
        != 41
        or _integer(
            validation_manifest.get("total_requests"),
            f"{validation_manifest_path}.total_requests",
        )
        != 1000
    ):
        raise SystemExit(
            f"formal Full-vs-Serverless {validation_manifest_path}: invalid "
            "completed seed-41 validation manifest"
        )
    validation_manifest_expected: Mapping[str, Any] = {
        "campaign_kind": FORMAL_CAMPAIGN_KIND,
        "model_profile": FORMAL_MODEL_PROFILES[model],
        "system_resolved_config_family_id": family_id,
        "system_resolved_config_sha256": manifest_config,
        "execution_order": list(expected_order),
    }
    for field, expected_value in validation_manifest_expected.items():
        if validation_manifest.get(field) != expected_value:
            raise SystemExit(
                f"formal Full-vs-Serverless {validation_manifest_path}: "
                f"{field} mismatch"
            )
    if len(validation_manifest.get("systems") or []) != 2 or set(
        str(item) for item in (validation_manifest.get("systems") or [])
    ) != set(FORMAL_SYSTEMS):
        raise SystemExit(
            f"formal Full-vs-Serverless {validation_manifest_path}: systems mismatch"
        )
    if (
        Path(str(validation_manifest.get("system_resolved_config_path") or ""))
        .expanduser()
        .resolve()
        != validation_sidecar_path
    ):
        raise SystemExit(
            f"formal Full-vs-Serverless {validation_manifest_path}: "
            "sidecar path mismatch"
        )
    if _integer(
        validation_manifest.get("system_resolved_config_sidecar_bytes"),
        f"{validation_manifest_path}.system_resolved_config_sidecar_bytes",
    ) != validation_sidecar_path.stat().st_size or _sha(
        validation_manifest.get("system_resolved_config_sidecar_sha256"),
        f"{validation_manifest_path}.system_resolved_config_sidecar_sha256",
    ) != _sha256_file(
        validation_sidecar_path
    ):
        raise SystemExit(
            f"formal Full-vs-Serverless {validation_manifest_path}: current "
            "validation sidecar bytes are not preserved"
        )

    for path_field, sha_field, identity_field in (
        ("shared_trace_path", "shared_trace_sha256", "trace_sha256"),
        (
            "shared_adapter_subset_path",
            "shared_adapter_subset_sha256",
            "adapter_subset_sha256",
        ),
    ):
        artifact_path = (
            Path(str(validation.get(path_field) or "")).expanduser().resolve()
        )
        if not artifact_path.is_file():
            raise SystemExit(
                f"formal Full-vs-Serverless {evidence_path}: validation artifact "
                f"is missing: {artifact_path}"
            )
        artifact_sha = _sha256_file(artifact_path)
        if (
            _sha(validation.get(sha_field), f"{evidence_path}.{sha_field}")
            != artifact_sha
            or _sha(
                validation_manifest.get(sha_field),
                f"{validation_manifest_path}.{sha_field}",
            )
            != artifact_sha
            or _sha(
                validation_identity.get(identity_field),
                f"{validation_sidecar_path}.{identity_field}",
            )
            != artifact_sha
        ):
            raise SystemExit(
                f"formal Full-vs-Serverless {evidence_path}: validation artifact "
                f"identity mismatch for {path_field}"
            )
        manifest_artifact_path = (
            Path(str(validation_manifest.get(path_field) or "")).expanduser().resolve()
        )
        if manifest_artifact_path != artifact_path:
            raise SystemExit(
                f"formal Full-vs-Serverless {validation_manifest_path}: "
                f"{path_field} mismatch"
            )


def validate_formal_campaign_protocol(
    runs: Sequence[FormalRun],
    manifest_by_source: Mapping[Path, Path],
) -> None:
    """Bind every paired result to one immutable formal fair-round identity."""

    lookup = {_identity(run): run for run in runs}
    for model in FORMAL_MODELS:
        for seed in FORMAL_SEEDS:
            pair = [lookup[(model, seed, system)] for system in FORMAL_SYSTEMS]
            manifest_paths: set[Path] = set()
            for run in pair:
                source = run.source.resolve()
                raw_manifest = manifest_by_source.get(source)
                if raw_manifest is None:
                    raise SystemExit(
                        f"formal Full-vs-Serverless {source}: no campaign manifest "
                        "was associated with the raw result"
                    )
                manifest_paths.add(Path(raw_manifest).resolve())
            if len(manifest_paths) != 1:
                raise SystemExit(
                    f"formal Full-vs-Serverless model={model},seed={seed}: paired "
                    f"systems must share one fair-round manifest, observed="
                    f"{sorted(str(path) for path in manifest_paths)}"
                )
            manifest_path = next(iter(manifest_paths))
            manifest = _read_json(manifest_path, "campaign manifest")
            if str(manifest.get("campaign_kind") or "") != FORMAL_CAMPAIGN_KIND:
                raise SystemExit(
                    f"formal Full-vs-Serverless {manifest_path}: campaign_kind must "
                    f"be {FORMAL_CAMPAIGN_KIND!r}"
                )

            systems = [str(item) for item in (manifest.get("systems") or [])]
            supported = [
                str(item) for item in (manifest.get("supported_systems") or [])
            ]
            if len(systems) != 2 or set(systems) != set(FORMAL_SYSTEMS):
                raise SystemExit(
                    f"formal Full-vs-Serverless {manifest_path}: systems must be "
                    f"exactly {list(FORMAL_SYSTEMS)}, observed={systems}"
                )
            if len(supported) != 2 or set(supported) != set(FORMAL_SYSTEMS):
                raise SystemExit(
                    f"formal Full-vs-Serverless {manifest_path}: supported_systems "
                    f"must be exactly {list(FORMAL_SYSTEMS)}, observed={supported}"
                )
            expected_order = FORMAL_EXECUTION_ORDERS[(model, seed)]
            observed_order = tuple(
                str(item) for item in (manifest.get("execution_order") or [])
            )
            if observed_order != expected_order:
                raise SystemExit(
                    f"formal Full-vs-Serverless {manifest_path}: execution_order for "
                    f"model={model},seed={seed} must be {list(expected_order)}, "
                    f"observed={list(observed_order)}"
                )

            expected_profile = FORMAL_MODEL_PROFILES[model]
            if str(manifest.get("model_profile") or "") != expected_profile:
                raise SystemExit(
                    f"formal Full-vs-Serverless {manifest_path}: model_profile mismatch"
                )
            if (
                _integer(
                    manifest.get("sampling_seed"), f"{manifest_path}.sampling_seed"
                )
                != seed
            ):
                raise SystemExit(
                    f"formal Full-vs-Serverless {manifest_path}: manifest seed mismatch"
                )
            if (
                _integer(
                    manifest.get("total_requests"), f"{manifest_path}.total_requests"
                )
                != 4000
            ):
                raise SystemExit(
                    f"formal Full-vs-Serverless {manifest_path}: "
                    "total_requests must be 4000"
                )
            if (
                _integer(
                    manifest.get("selected_num_adapters"),
                    f"{manifest_path}.selected_num_adapters",
                )
                != 500
            ):
                raise SystemExit(
                    f"formal Full-vs-Serverless {manifest_path}: "
                    "selected_num_adapters must be 500"
                )
            _require_number(
                manifest.get("bandwidth_mib_s"),
                250.0,
                f"{manifest_path}.bandwidth_mib_s",
            )
            _require_text(
                manifest.get("generation_contract"),
                "legacy",
                f"{manifest_path}.generation_contract",
            )
            _require_text(
                manifest.get("faaslora_scenario"),
                "v2_full",
                f"{manifest_path}.faaslora_scenario",
            )
            _require_workload_axes(
                manifest.get("workload_overrides"),
                f"{manifest_path}.workload_overrides",
            )

            pair_trace = {run.trace_sha256 for run in pair}
            pair_subset = {run.adapter_subset_sha256 for run in pair}
            pair_config = {run.system_resolved_config_sha256 for run in pair}
            manifest_trace = _sha(
                manifest.get("shared_trace_sha256"),
                f"{manifest_path}.shared_trace_sha256",
            )
            manifest_subset = _sha(
                manifest.get("shared_adapter_subset_sha256"),
                f"{manifest_path}.shared_adapter_subset_sha256",
            )
            manifest_config = _sha(
                manifest.get("system_resolved_config_sha256"),
                f"{manifest_path}.system_resolved_config_sha256",
            )
            if pair_trace != {manifest_trace} or pair_subset != {manifest_subset}:
                raise SystemExit(
                    f"formal Full-vs-Serverless {manifest_path}: raw results and "
                    "manifest do not share trace/subset identity"
                )
            if pair_config != {manifest_config}:
                raise SystemExit(
                    f"formal Full-vs-Serverless {manifest_path}: raw results and "
                    "manifest do not share resolved configuration"
                )

            sidecar_path = _resolve_manifest_sidecar(manifest_path, manifest)
            sidecar = _read_json(sidecar_path, "resolved-config sidecar")
            if (
                sidecar.get("formal_run") is not True
                or str(sidecar.get("trace_role") or "").strip().lower() != "heldout"
            ):
                raise SystemExit(
                    f"formal Full-vs-Serverless {sidecar_path}: sidecar must declare "
                    "formal_run=true and trace_role='heldout'"
                )
            if str(sidecar.get("campaign_kind") or "") != FORMAL_CAMPAIGN_KIND:
                raise SystemExit(
                    f"formal Full-vs-Serverless {sidecar_path}: campaign_kind mismatch"
                )
            if str(sidecar.get("model_profile") or "") != expected_profile:
                raise SystemExit(
                    f"formal Full-vs-Serverless {sidecar_path}: model_profile mismatch"
                )
            if (
                _integer(sidecar.get("sampling_seed"), f"{sidecar_path}.sampling_seed")
                != seed
            ):
                raise SystemExit(
                    f"formal Full-vs-Serverless {sidecar_path}: sidecar seed mismatch"
                )
            if (
                _sha(
                    sidecar.get("system_resolved_config_sha256"),
                    f"{sidecar_path}.system_resolved_config_sha256",
                )
                != manifest_config
            ):
                raise SystemExit(
                    f"formal Full-vs-Serverless {sidecar_path}: sidecar/manifest "
                    "resolved configuration mismatch"
                )
            sidecar_family_id = _sha(
                sidecar.get("configuration_family_id"),
                f"{sidecar_path}.configuration_family_id",
            )
            if (
                _sha(
                    manifest.get("system_resolved_config_family_id"),
                    f"{manifest_path}.system_resolved_config_family_id",
                )
                != sidecar_family_id
            ):
                raise SystemExit(
                    f"formal Full-vs-Serverless {sidecar_path}: "
                    "configuration family differs from manifest"
                )

            identity = sidecar.get("full_run_identity")
            if not isinstance(identity, Mapping):
                raise SystemExit(
                    f"formal Full-vs-Serverless {sidecar_path}: "
                    "missing full_run_identity"
                )
            identity_sha = _canonical_sha256(identity)
            if (
                _sha(
                    sidecar.get("full_run_identity_sha256"),
                    f"{sidecar_path}.full_run_identity_sha256",
                )
                != identity_sha
                or _sha(
                    manifest.get("full_run_identity_sha256"),
                    f"{manifest_path}.full_run_identity_sha256",
                )
                != identity_sha
            ):
                raise SystemExit(
                    f"formal Full-vs-Serverless {sidecar_path}: full_run_identity "
                    "SHA-256 does not match sidecar/manifest"
                )

            text_identity_fields = {
                "model_profile": expected_profile,
                "generation_contract": "legacy",
                "faaslora_scenario": "v2_full",
                "campaign_kind": FORMAL_CAMPAIGN_KIND,
            }
            for field, expected_value in text_identity_fields.items():
                _require_text(
                    identity.get(field), expected_value, f"{sidecar_path}.{field}"
                )
            integer_identity_fields = {
                "total_requests": 4000,
                "selected_num_adapters": 500,
                "sampling_seed": seed,
            }
            for field, expected_value in integer_identity_fields.items():
                if (
                    _integer(identity.get(field), f"{sidecar_path}.{field}")
                    != expected_value
                ):
                    raise SystemExit(
                        f"formal Full-vs-Serverless {sidecar_path}: full-run {field} "
                        f"must be {expected_value}"
                    )
            _require_number(
                identity.get("storage_bandwidth_mib_s"),
                250.0,
                f"{sidecar_path}.storage_bandwidth_mib_s",
            )
            _require_number(
                identity.get("time_scale_factor"),
                8.0,
                f"{sidecar_path}.time_scale_factor",
            )
            if (
                tuple(str(item) for item in (identity.get("execution_order") or []))
                != expected_order
            ):
                raise SystemExit(
                    f"formal Full-vs-Serverless {sidecar_path}: full-run "
                    "execution_order mismatch"
                )
            if len(identity.get("systems") or []) != 2 or set(
                str(item) for item in (identity.get("systems") or [])
            ) != set(FORMAL_SYSTEMS):
                raise SystemExit(
                    f"formal Full-vs-Serverless {sidecar_path}: "
                    "full-run systems mismatch"
                )
            if (
                _sha(identity.get("trace_sha256"), f"{sidecar_path}.trace_sha256")
                != manifest_trace
                or _sha(
                    identity.get("adapter_subset_sha256"),
                    f"{sidecar_path}.adapter_subset_sha256",
                )
                != manifest_subset
            ):
                raise SystemExit(
                    f"formal Full-vs-Serverless {sidecar_path}: full-run "
                    "trace/subset identity mismatch"
                )
            if (
                _sha(
                    identity.get("system_resolved_config_sha256"),
                    f"{sidecar_path}.system_resolved_config_sha256",
                )
                != manifest_config
            ):
                raise SystemExit(
                    f"formal Full-vs-Serverless {sidecar_path}: full-run resolved "
                    "configuration mismatch"
                )
            _require_workload_axes(
                identity.get("workload_overrides"),
                f"{sidecar_path}.workload_overrides",
            )
            if _source_commits(
                identity.get("source_commits"),
                f"{sidecar_path}.full_run_identity.source_commits",
            ) != _source_commits(
                sidecar.get("source_commits"), f"{sidecar_path}.source_commits"
            ):
                raise SystemExit(
                    f"formal Full-vs-Serverless {sidecar_path}: full-run source "
                    "commits mismatch"
                )
            _validate_seed41_evidence(
                manifest_path=manifest_path,
                manifest=manifest,
                sidecar_path=sidecar_path,
                sidecar=sidecar,
                model=model,
                heldout_seed=seed,
                manifest_config=manifest_config,
                identity_sha=identity_sha,
            )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise SystemExit(f"refusing to write empty CSV: {path}")
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    runs: Sequence[FormalRun],
    out_dir: Path,
    *,
    inputs: Sequence[Path],
    manifest_paths: Sequence[Path],
) -> None:
    validate_formal_matrix(runs)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            "refusing to overwrite non-empty Full-vs-Serverless output "
            f"directory: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run: List[Dict[str, Any]] = []
    lookup = {_identity(run): run for run in runs}
    for run in sorted(runs, key=lambda item: (item.model, item.seed, item.system)):
        per_run.append(
            {
                "model": run.model,
                "seed": run.seed,
                "system": run.system,
                "scenario": run.scenario,
                "source": str(run.source),
                "total": run.total,
                "completed": run.completed,
                "generation_contract": run.generation_contract,
                "shared_trace_sha256": run.trace_sha256,
                "shared_adapter_subset_sha256": run.adapter_subset_sha256,
                "system_resolved_config_sha256": run.system_resolved_config_sha256,
                **run.metrics,
            }
        )

    paired: List[Dict[str, Any]] = []
    for model in FORMAL_MODELS:
        for seed in FORMAL_SEEDS:
            prime = lookup[(model, seed, "faaslora")]
            baseline = lookup[(model, seed, "serverlessllm")]
            for metric in METRICS:
                prime_value = float(prime.metrics[metric])
                baseline_value = float(baseline.metrics[metric])
                difference = prime_value - baseline_value
                if baseline_value == 0.0:
                    raise SystemExit(
                        f"cannot compute paired improvement for zero {metric}"
                    )
                improvement = (
                    (prime_value - baseline_value) / baseline_value * 100.0
                    if metric in HIGHER_IS_BETTER
                    else (baseline_value - prime_value) / baseline_value * 100.0
                )
                paired.append(
                    {
                        "model": model,
                        "seed": seed,
                        "metric": metric,
                        "higher_is_better": metric in HIGHER_IS_BETTER,
                        "prime_value": prime_value,
                        "serverlessllm_value": baseline_value,
                        "paired_difference_prime_minus_serverlessllm": difference,
                        "prime_improvement_pct": improvement,
                    }
                )

    summary: List[Dict[str, Any]] = []
    for model in FORMAL_MODELS:
        for metric in METRICS:
            rows = [
                row
                for row in paired
                if row["model"] == model and row["metric"] == metric
            ]
            differences = [
                float(row["paired_difference_prime_minus_serverlessllm"])
                for row in rows
            ]
            improvements = [float(row["prime_improvement_pct"]) for row in rows]
            diff_mean, diff_std, diff_half = _v2_mean_ci95(differences)
            imp_mean, imp_std, imp_half = _v2_mean_ci95(improvements)
            summary.append(
                {
                    "model": model,
                    "metric": metric,
                    "higher_is_better": metric in HIGHER_IS_BETTER,
                    "paired_seed_count": len(rows),
                    "paired_seeds": ";".join(str(row["seed"]) for row in rows),
                    "paired_difference_mean": diff_mean,
                    "paired_difference_std": diff_std,
                    "paired_difference_ci95_half_width": diff_half,
                    "prime_improvement_pct_mean": imp_mean,
                    "prime_improvement_pct_std": imp_std,
                    "prime_improvement_pct_ci95_half_width": imp_half,
                }
            )

    per_run_path = out_dir / "full_vs_serverless_per_run.csv"
    paired_path = out_dir / "full_vs_serverless_paired_per_seed.csv"
    summary_path = out_dir / "full_vs_serverless_paired_summary.csv"
    _write_csv(per_run_path, per_run)
    _write_csv(paired_path, paired)
    _write_csv(summary_path, summary)
    manifest = {
        "analysis": "v2_full_vs_serverlessllm_new_formal",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_matrix": True,
        "inputs": [str(Path(path).expanduser().resolve()) for path in inputs],
        "provenance_manifests": [str(path) for path in manifest_paths],
        "identity_count": len(runs),
        "statistical_unit": "independent held-out seed",
        "ci": "two-sided 95% Student-t over three paired seed-level differences",
        "frozen_workload": {
            "requests": 4000,
            "adapters": 500,
            "bandwidth_mib_s": 250.0,
            "time_scale": 8.0,
            "generation_contract": "legacy",
            "zipf_exponent": 1.0,
            "active_adapter_cap": 48,
            "hotset_rotation_requests": 500,
            "hotset_rotation_mode": "legacy",
            "hotset_overlap_fraction": 0.75,
        },
        "csvs": [per_run_path.name, paired_path.name, summary_path.name],
    }
    (out_dir / "full_vs_serverless_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def analyze(inputs: Sequence[Path], out_dir: Path) -> None:
    provenance = build_formal_provenance_index(inputs)
    runs = load_formal_runs(inputs)
    validate_formal_matrix(runs)
    validate_formal_campaign_protocol(
        runs,
        _manifest_paths_for_runs(provenance, runs),
    )
    validate_formal_analysis_sources(
        provenance,
        (
            FormalAnalysisIdentity(
                source=run.source,
                model=run.model,
                variant=run.system,
                seed=run.seed,
            )
            for run in runs
        ),
        analysis_label="V2 Full versus ServerlessLLM-new",
    )
    write_outputs(
        runs,
        out_dir,
        inputs=inputs,
        manifest_paths=provenance.manifest_paths,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Formal V2 PrimeLoRA Full versus ServerlessLLM-new analyzer"
    )
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.input, args.output_dir.resolve())
    print(
        f"generated formal Full-vs-Serverless analysis -> {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
