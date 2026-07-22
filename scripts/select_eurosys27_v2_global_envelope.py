#!/usr/bin/env python3
"""Select one seed-41 PrimeLoRA runtime envelope per V2 model.

Every candidate is supplied as one C5 fair round and one Full-vs-Serverless
fair round.  Inputs are validated from their completed MANIFEST.json files and
raw result records.  The output directory is create-once: an existing path is
never reused, including when no candidate is eligible.

Example::

    python scripts/select_eurosys27_v2_global_envelope.py \
      --candidate llama2_7b P0 /rounds/c5-7b-p0 /rounds/fvs-7b-p0 \
      --candidate llama2_7b P1 /rounds/c5-7b-p1 /rounds/fvs-7b-p1 \
      --candidate llama32_3b P0 /rounds/c5-3b-p0 /rounds/fvs-3b-p0 \
      --output-dir paper_results/eurosys27_v2/validation/selection-001
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


SCHEMA_VERSION = "eurosys27_v2_global_envelope_selection_result_v1"
DEFAULT_PROTOCOL = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "eurosys27_v2_validation_selection_protocol.json"
)
CE_RELATIVE_TOLERANCE = 0.005


class SelectionError(ValueError):
    """An input cannot support a seed-41 selection decision."""


@dataclass(frozen=True)
class CandidateInput:
    model: str
    label: str
    family_paths: Mapping[str, Path]


@dataclass(frozen=True)
class FamilyMetrics:
    prime_avg_ttft_ms: float
    baseline_avg_ttft_ms: float
    prime_avg_e2e_ms: float
    baseline_avg_e2e_ms: float
    prime_p95_ttft_ms: float
    baseline_p95_ttft_ms: float
    prime_ce: float
    baseline_ce: float

    @property
    def p95_ratio(self) -> float:
        return self.prime_p95_ttft_ms / self.baseline_p95_ttft_ms

    @property
    def ce_ratio(self) -> float:
        return self.prime_ce / self.baseline_ce


@dataclass
class CandidateAudit:
    model: str
    label: str
    expected_envelope: Mapping[str, int]
    observed_envelopes: Dict[str, Mapping[str, int]] = field(default_factory=dict)
    source_identities: Dict[str, Mapping[str, Mapping[str, str]]] = field(
        default_factory=dict
    )
    metrics: Dict[str, FamilyMetrics] = field(default_factory=dict)
    rejection_reasons: list[str] = field(default_factory=list)
    eligible: bool = False
    score: float | None = None
    selected: bool = False


class ProvenanceRecorder:
    def __init__(self) -> None:
        self._records: Dict[str, Dict[str, Any]] = {}

    def record(self, path: Path, role: str) -> Path:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise SelectionError(f"missing {role}: {resolved}")
        digest = hashlib.sha256()
        size = 0
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
        key = str(resolved)
        row = self._records.setdefault(
            key,
            {"path": key, "bytes": size, "sha256": digest.hexdigest(), "roles": []},
        )
        if row["bytes"] != size or row["sha256"] != digest.hexdigest():
            raise SelectionError(f"input changed while selecting: {resolved}")
        if role not in row["roles"]:
            row["roles"].append(role)
        return resolved

    def rows(self) -> list[Dict[str, Any]]:
        return [self._records[key] for key in sorted(self._records)]


def _read_json(path: Path, role: str, provenance: ProvenanceRecorder) -> Dict[str, Any]:
    source = provenance.record(path, role)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SelectionError(f"cannot read {role} JSON {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SelectionError(f"{role} JSON root must be an object: {source}")
    return payload


def _manifest_path(raw: Path) -> Path:
    candidate = raw.expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / "MANIFEST.json"
    if candidate.name != "MANIFEST.json":
        raise SelectionError(f"candidate input must be a round or MANIFEST.json: {raw}")
    return candidate


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise SelectionError(f"{label} must be an integer, observed boolean {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"{label} must be an integer, observed {value!r}") from exc
    try:
        if float(value) != parsed:
            raise ValueError
    except (TypeError, ValueError):
        raise SelectionError(f"{label} must be an integer, observed {value!r}")
    return parsed


def _finite_positive(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"{label} must be numeric, observed {value!r}") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise SelectionError(
            f"{label} must be finite and positive, observed {parsed!r}"
        )
    return parsed


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expected_envelope(
    protocol: Mapping[str, Any], model: str, label: str
) -> Dict[str, int]:
    fields = protocol["envelope_fields"]
    values = protocol["models"][model]["candidates"][label]
    if not isinstance(fields, list) or len(fields) != 6 or len(set(fields)) != 6:
        raise SelectionError("protocol envelope_fields must contain six unique names")
    if not isinstance(values, list) or len(values) != len(fields):
        raise SelectionError(
            f"protocol candidate {model}/{label} has wrong envelope width"
        )
    return {
        str(name): _integer(value, f"{model}/{label}.{name}")
        for name, value in zip(fields, values)
    }


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "trace_role",
        "sampling_seed",
        "total_requests",
        "families",
        "envelope_fields",
        "models",
        "candidate_validity",
        "selection",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise SelectionError(f"selection protocol lacks fields {missing}")
    if str(protocol["schema_version"]) != "eurosys27_v2_global_envelope_selection_v1":
        raise SelectionError("unsupported selection protocol schema_version")
    if str(protocol["trace_role"]) != "validation":
        raise SelectionError("selection protocol trace_role must be validation")
    if _integer(protocol["sampling_seed"], "protocol.sampling_seed") != 41:
        raise SelectionError("selection protocol sampling_seed must be 41")
    if _integer(protocol["total_requests"], "protocol.total_requests") != 1000:
        raise SelectionError("selection protocol total_requests must be 1000")
    if set(protocol["families"]) != {"c5", "fvs"}:
        raise SelectionError("selection protocol must define exactly c5 and fvs")
    validity = protocol["candidate_validity"]
    expected_validity = {
        "required_families": ["c5", "fvs"],
        "completed_requests": 1000,
        "failed_requests": 0,
        "allow_fallback": False,
        "constraints_per_family": {
            "prime_avg_ttft_lte_baseline": True,
            "prime_avg_e2e_lte_baseline": True,
            "prime_p95_ttft_ratio_max": 1.05,
        },
    }
    if validity != expected_validity:
        raise SelectionError(
            "candidate_validity differs from the implemented fail-closed gate: "
            f"expected={expected_validity!r}, observed={validity!r}"
        )
    selection = protocol["selection"]
    expected_selection = {
        "score": "min_over_families(CE_prime / CE_baseline)",
        "direction": "maximize",
        "tie_relative_tolerance": 0.01,
        "tie_break": "lowest_candidate_label",
        "no_eligible_candidate": "stop_before_heldout",
    }
    if selection != expected_selection:
        raise SelectionError(
            "selection policy differs from the implemented rule: "
            f"expected={expected_selection!r}, observed={selection!r}"
        )
    for model, model_spec in protocol["models"].items():
        candidates = (
            model_spec.get("candidates") if isinstance(model_spec, Mapping) else None
        )
        if not isinstance(candidates, Mapping) or not candidates:
            raise SelectionError(f"protocol model {model!r} has no candidates")
        for label in candidates:
            _expected_envelope(protocol, str(model), str(label))


def _sidecar_path(manifest_path: Path, manifest: Mapping[str, Any]) -> Path:
    raw = str(manifest.get("system_resolved_config_path") or "").strip()
    if not raw:
        raise SelectionError(f"{manifest_path}: missing system_resolved_config_path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    expected = (
        manifest_path.parent / "protocol" / "system_resolved_config.json"
    ).resolve()
    if path != expected:
        raise SelectionError(
            f"{manifest_path}: sidecar must be the round-local {expected}, observed {path}"
        )
    return path


def _source_identity(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> Dict[str, Dict[str, str]]:
    if manifest.get("source_clean_for_formal") is not True:
        raise SelectionError(f"{manifest_path}: source_clean_for_formal must be true")
    cleanliness = manifest.get("source_cleanliness")
    if (
        isinstance(cleanliness, Mapping)
        and cleanliness.get("source_clean_for_formal") is not True
    ):
        raise SelectionError(
            f"{manifest_path}: source_cleanliness.source_clean_for_formal must be true"
        )
    expected_branches = {
        "baseline_git": "main",
        "faaslora_git": "retry14_continuous_queue_v2",
    }
    identity: Dict[str, Dict[str, str]] = {}
    for key, expected_branch in expected_branches.items():
        row = manifest.get(key)
        if not isinstance(row, Mapping):
            raise SelectionError(f"{manifest_path}: missing {key} source identity")
        branch = str(row.get("branch") or "").strip()
        commit = str(row.get("commit") or "").strip().lower()
        if branch != expected_branch:
            raise SelectionError(
                f"{manifest_path}: {key}.branch must be {expected_branch!r}, observed {branch!r}"
            )
        if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
            raise SelectionError(
                f"{manifest_path}: {key}.commit is not a full Git SHA-1"
            )
        identity[key] = {"branch": branch, "commit": commit}
    return identity


def _read_candidate_source_identity(
    candidate: CandidateInput,
    family: str,
    provenance: ProvenanceRecorder,
) -> Dict[str, Dict[str, str]]:
    manifest_path = _manifest_path(candidate.family_paths[family])
    manifest = _read_json(
        manifest_path,
        f"{candidate.model}/{candidate.label}/{family} source identity",
        provenance,
    )
    return _source_identity(manifest_path, manifest)


_RESOLVED_ENVELOPE_PATHS = {
    "FAASLORA_MIN_INSTANCES": ("resource_coordination", "min_instances"),
    "FAASLORA_MAX_INSTANCES": ("resource_coordination", "max_instances"),
    "FAASLORA_RUNTIME_CONCURRENCY_CAP": ("model", "runtime_concurrency_cap"),
    "FAASLORA_MAX_NUM_SEQS": ("model", "max_num_seqs"),
    "FAASLORA_MAX_LORAS": ("model", "max_loras"),
    "FAASLORA_MAX_NUM_BATCHED_TOKENS": ("model", "max_num_batched_tokens"),
}


def _extract_envelope(
    sidecar: Mapping[str, Any], fields: Iterable[str], label: str
) -> Dict[str, int]:
    hashed = sidecar.get("hashed_config")
    systems = hashed.get("systems") if isinstance(hashed, Mapping) else None
    prime = systems.get("PrimeLoRA") if isinstance(systems, Mapping) else None
    if not isinstance(prime, Mapping):
        raise SelectionError(f"{label}: sidecar lacks hashed_config.systems.PrimeLoRA")
    overrides = prime.get("environment_overrides")
    if not isinstance(overrides, Mapping):
        overrides = {}
    profiles = prime.get("resolved_profiles")
    if not isinstance(profiles, Mapping):
        profiles = {}
    output: Dict[str, int] = {}
    for field_name in fields:
        values: list[tuple[str, int]] = []
        if field_name in overrides and str(overrides[field_name]).strip() != "":
            values.append(
                (
                    "environment_overrides",
                    _integer(overrides[field_name], f"{label}.{field_name}"),
                )
            )
        section_name, key = _RESOLVED_ENVELOPE_PATHS.get(str(field_name), ("", ""))
        section = profiles.get(section_name) if section_name else None
        if isinstance(section, Mapping) and section.get(key) is not None:
            values.append(
                (
                    f"resolved_profiles.{section_name}",
                    _integer(section[key], f"{label}.{field_name}"),
                )
            )
        if not values:
            raise SelectionError(
                f"{label}: missing resolved envelope field {field_name}"
            )
        distinct = {value for _, value in values}
        if len(distinct) != 1:
            raise SelectionError(f"{label}: conflicting {field_name} values {values}")
        output[str(field_name)] = values[0][1]
    return output


def _validate_manifest_and_sidecar(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    sidecar_path: Path,
    family: str,
    family_spec: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> Dict[str, int]:
    prefix = f"{manifest_path} ({family})"
    expected_systems = {
        str(family_spec["prime_system"]),
        str(family_spec["baseline_system"]),
    }
    model_key = (
        "7b"
        if str(model_spec["model_profile"]) == "llama2_7b_main_v2_publicmix"
        else "3b"
    )
    expected_orders = {
        ("c5", "7b"): ["slora", "faaslora"],
        ("c5", "3b"): ["faaslora", "slora"],
        ("fvs", "7b"): ["faaslora", "serverlessllm"],
        ("fvs", "3b"): ["serverlessllm", "faaslora"],
    }
    expected_order = expected_orders[(family, model_key)]
    checks = (
        (manifest.get("status") == "complete", "status must be complete"),
        (manifest.get("formal_run") is False, "formal_run must be false"),
        (
            str(manifest.get("trace_role") or "") == "validation",
            "trace_role must be validation",
        ),
        (
            _integer(manifest.get("sampling_seed"), f"{prefix}.sampling_seed") == 41,
            "sampling_seed must be 41",
        ),
        (
            _integer(manifest.get("total_requests"), f"{prefix}.total_requests")
            == 1000,
            "total_requests must be 1000",
        ),
        (
            str(manifest.get("campaign_kind") or "")
            == str(family_spec["campaign_kind"]),
            "campaign_kind mismatch",
        ),
        (
            str(manifest.get("generation_contract") or "")
            == str(family_spec["generation_contract"]),
            "generation_contract mismatch",
        ),
        (
            str(manifest.get("model_profile") or "")
            == str(model_spec["model_profile"]),
            "model_profile mismatch",
        ),
        (
            str(manifest.get("workload_profile") or "")
            == str(model_spec["workload_profile"]),
            "workload_profile mismatch",
        ),
        (
            set(map(str, manifest.get("systems") or [])) == expected_systems,
            "systems mismatch",
        ),
        (
            set(map(str, manifest.get("supported_systems") or [])) == expected_systems,
            "supported_systems mismatch",
        ),
        (
            list(map(str, manifest.get("execution_order") or [])) == expected_order,
            "seed-41 execution_order mismatch",
        ),
        (
            str(manifest.get("faaslora_scenario") or "") == "v2_full",
            "PrimeLoRA scenario must be v2_full",
        ),
        (
            manifest.get("source_clean_for_formal") is True,
            "source_clean_for_formal must be true",
        ),
        (sidecar.get("formal_run") is False, "sidecar formal_run must be false"),
        (
            str(sidecar.get("trace_role") or "") == "validation",
            "sidecar trace_role must be validation",
        ),
        (
            _integer(sidecar.get("sampling_seed"), f"{prefix}.sidecar.sampling_seed")
            == 41,
            "sidecar sampling_seed must be 41",
        ),
        (
            sidecar.get("source_clean_for_formal") is True,
            "sidecar source_clean_for_formal must be true",
        ),
    )
    errors = [message for passed, message in checks if not passed]
    sidecar_bytes = sidecar_path.stat().st_size
    sidecar_sha = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    if (
        _integer(
            manifest.get("system_resolved_config_sidecar_bytes"),
            f"{prefix}.sidecar_bytes",
        )
        != sidecar_bytes
    ):
        errors.append("sidecar byte count differs from MANIFEST")
    if str(manifest.get("system_resolved_config_sidecar_sha256") or "") != sidecar_sha:
        errors.append("sidecar SHA-256 differs from MANIFEST")
    hashed = sidecar.get("hashed_config")
    if not isinstance(hashed, Mapping):
        errors.append("sidecar hashed_config must be an object")
    else:
        hashed_systems = hashed.get("systems")
        required_hashed_names = {
            "PrimeLoRA",
            "S-LoRA" if family == "c5" else "ServerlessLLM-new",
        }
        if not isinstance(
            hashed_systems, Mapping
        ) or not required_hashed_names.issubset(hashed_systems):
            errors.append(
                f"sidecar lacks required resolved system semantics {sorted(required_hashed_names)}"
            )
        logical_sha = _canonical_sha256(hashed)
        declared = str(sidecar.get("system_resolved_config_sha256") or "")
        if logical_sha != declared:
            errors.append("sidecar logical config SHA-256 is invalid")
        if str(manifest.get("system_resolved_config_sha256") or "") != declared:
            errors.append("MANIFEST and sidecar logical config SHA-256 differ")
    if errors:
        raise SelectionError(prefix + ": " + "; ".join(errors))
    return _extract_envelope(sidecar, protocol["envelope_fields"], prefix)


def _fallback_used(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) != 0.0
    if isinstance(value, str):
        return value.strip().lower() not in {
            "",
            "0",
            "false",
            "none",
            "null",
            "disabled",
            "not_used",
            "unused",
        }
    return bool(value)


def _fallback_findings(value: Any, path: str = "root") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            normalized = str(key).lower().replace("-", "_")
            fallback_signal = (
                normalized
                in {"fallback", "fallback_used", "used_fallback", "fallback_applied"}
                or normalized.endswith("_fallback_count")
                or normalized.endswith("_fallback_counts")
                or normalized.endswith("_fallback_used")
            )
            if fallback_signal and isinstance(child, Mapping):
                used = {
                    name: count
                    for name, count in child.items()
                    if _fallback_used(count)
                }
                if used:
                    findings.append(f"{child_path}={used!r}")
            elif fallback_signal and _fallback_used(child):
                findings.append(f"{child_path}={child!r}")
            if isinstance(child, (Mapping, list)):
                findings.extend(_fallback_findings(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, (Mapping, list)):
                findings.extend(_fallback_findings(child, f"{path}[{index}]"))
    return findings


def _check_ce(metrics: FamilyMetrics, family: str) -> None:
    for system in ("prime", "baseline"):
        e2e = getattr(metrics, f"{system}_avg_e2e_ms")
        ce = getattr(metrics, f"{system}_ce")
        # Cost is algebraically recoverable from CE and E2E only if the raw
        # parser already verified the reported CE.  Family loaders below do
        # that check before constructing this common record.
        _finite_positive(e2e, f"{family}.{system}.avg_e2e_ms")
        _finite_positive(ce, f"{family}.{system}.ce")


def _validate_reported_ce(
    e2e_ms: float, cost_req: float, ce: float, label: str
) -> None:
    expected = 1.0 / (cost_req * (e2e_ms / 1000.0))
    if not math.isclose(ce, expected, rel_tol=CE_RELATIVE_TOLERANCE, abs_tol=1e-3):
        raise SelectionError(
            f"{label}: CE inconsistent with E2E and cost; reported={ce:.9g}, expected={expected:.9g}"
        )


def _load_c5_metrics(
    manifest_path: Path, provenance: ProvenanceRecorder
) -> FamilyMetrics:
    try:
        from analyze_c5_matched_output import (
            specs_from_round,
            validate_pairs,
            validate_run,
        )
    except ImportError:  # pragma: no cover - package import in tests
        from scripts.analyze_c5_matched_output import (
            specs_from_round,
            validate_pairs,
            validate_run,
        )

    specs = specs_from_round(manifest_path)
    runs = []
    for spec in specs:
        source = provenance.record(spec.path, f"c5 raw {spec.system}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise SelectionError(f"{source}: C5 raw metadata must be an object")
        if (
            _integer(metadata.get("sampling_seed"), f"{source}.metadata.sampling_seed")
            != 41
        ):
            raise SelectionError(f"{source}: C5 raw metadata.sampling_seed must be 41")
        runs.append(validate_run(spec, expected_requests=1000, formal_mode=False))
        provenance.record(spec.path, f"c5 raw {spec.system} post-validation")
    pairs = validate_pairs(runs)
    if len(pairs) != 1:
        raise SelectionError(
            f"{manifest_path}: C5 must contain exactly one model/seed pair"
        )
    pair = next(iter(pairs.values()))
    if set(pair) != {"prime", "slora"}:
        raise SelectionError(
            f"{manifest_path}: C5 raw systems are not PrimeLoRA/S-LoRA"
        )
    if pair["prime"].scenario != "v2_full" or pair["slora"].scenario != "slora_fair":
        raise SelectionError(f"{manifest_path}: C5 raw scenario semantics mismatch")
    prime = pair["prime"].metrics
    baseline = pair["slora"].metrics
    for label, row in (("prime", prime), ("baseline", baseline)):
        _validate_reported_ce(
            float(row["e2e_mean_ms"]),
            float(row["monetary_cost_per_request_usd"]),
            float(row["monetary_ce"]),
            f"{manifest_path}:C5:{label}",
        )
    return FamilyMetrics(
        prime_avg_ttft_ms=float(prime["overall_ttft_mean_ms"]),
        baseline_avg_ttft_ms=float(baseline["overall_ttft_mean_ms"]),
        prime_avg_e2e_ms=float(prime["e2e_mean_ms"]),
        baseline_avg_e2e_ms=float(baseline["e2e_mean_ms"]),
        prime_p95_ttft_ms=float(prime["overall_ttft_p95_ms"]),
        baseline_p95_ttft_ms=float(baseline["overall_ttft_p95_ms"]),
        prime_ce=float(prime["monetary_ce"]),
        baseline_ce=float(baseline["monetary_ce"]),
    )


def _load_fvs_metrics(
    manifest_path: Path,
    expected_model: str,
    provenance: ProvenanceRecorder,
) -> FamilyMetrics:
    try:
        from analyze_c5_matched_output import _assert_no_fallback_counts
        from analyze_v2_full_vs_serverless import _run_from_observation
        from plot_paper_sensitivity import load_v2_sensitivity_observations
    except ImportError:  # pragma: no cover - package imports in tests
        from scripts.analyze_c5_matched_output import _assert_no_fallback_counts
        from scripts.analyze_v2_full_vs_serverless import _run_from_observation
        from scripts.plot_paper_sensitivity import load_v2_sensitivity_observations

    observations = load_v2_sensitivity_observations([manifest_path])
    if len(observations) != 2:
        raise SelectionError(
            f"{manifest_path}: FVS must contain exactly two raw observations"
        )
    runs: Dict[str, Any] = {}
    for observation in observations:
        source = provenance.record(
            observation.source, f"fvs raw {observation.system_key}"
        )
        payload = json.loads(source.read_text(encoding="utf-8"))
        _assert_no_fallback_counts(payload, source)
        findings = _fallback_findings(payload)
        if findings:
            raise SelectionError(
                f"{source}: forbidden fallback evidence: {findings[:8]}"
            )
        run = _run_from_observation(observation)
        provenance.record(source, f"fvs raw {observation.system_key} post-validation")
        if run.model != expected_model:
            raise SelectionError(
                f"{source}: FVS raw model {run.model!r} differs from candidate {expected_model!r}"
            )
        if run.total != 1000 or run.completed != 1000:
            raise SelectionError(
                f"{source}: FVS requires 1000/1000, observed {run.completed}/{run.total}"
            )
        if run.metadata_sampling_seed != 41 or run.seed != 41:
            raise SelectionError(f"{source}: FVS requires sampling seed 41")
        if run.generation_contract != "legacy":
            raise SelectionError(f"{source}: FVS generation contract must be legacy")
        if run.system in runs:
            raise SelectionError(f"{manifest_path}: duplicate FVS system {run.system}")
        runs[run.system] = run
    if set(runs) != {"faaslora", "serverlessllm"}:
        raise SelectionError(
            f"{manifest_path}: FVS raw systems are not PrimeLoRA/ServerlessLLM-new"
        )
    if (
        runs["faaslora"].scenario != "v2_full"
        or runs["serverlessllm"].scenario != "serverlessllm_fair"
    ):
        raise SelectionError(f"{manifest_path}: FVS scenario semantics mismatch")
    if runs["faaslora"].trace_sha256 != runs["serverlessllm"].trace_sha256:
        raise SelectionError(
            f"{manifest_path}: FVS systems do not share one trace SHA-256"
        )
    if (
        runs["faaslora"].adapter_subset_sha256
        != runs["serverlessllm"].adapter_subset_sha256
    ):
        raise SelectionError(
            f"{manifest_path}: FVS systems do not share one adapter subset SHA-256"
        )
    if (
        runs["faaslora"].system_resolved_config_sha256
        != runs["serverlessllm"].system_resolved_config_sha256
    ):
        raise SelectionError(
            f"{manifest_path}: FVS systems do not share one resolved-config SHA-256"
        )
    prime = runs["faaslora"].metrics
    baseline = runs["serverlessllm"].metrics
    for label, row in (("prime", prime), ("baseline", baseline)):
        _validate_reported_ce(
            float(row["e2e_avg_ms"]),
            float(row["cost_req_usd"]),
            float(row["ce"]),
            f"{manifest_path}:FVS:{label}",
        )
    return FamilyMetrics(
        prime_avg_ttft_ms=float(prime["ttft_avg_ms"]),
        baseline_avg_ttft_ms=float(baseline["ttft_avg_ms"]),
        prime_avg_e2e_ms=float(prime["e2e_avg_ms"]),
        baseline_avg_e2e_ms=float(baseline["e2e_avg_ms"]),
        prime_p95_ttft_ms=float(prime["ttft_p95_ms"]),
        baseline_p95_ttft_ms=float(baseline["ttft_p95_ms"]),
        prime_ce=float(prime["ce"]),
        baseline_ce=float(baseline["ce"]),
    )


def load_family(
    candidate: CandidateInput,
    family: str,
    protocol: Mapping[str, Any],
    provenance: ProvenanceRecorder,
) -> tuple[Mapping[str, int], FamilyMetrics]:
    manifest_path = _manifest_path(candidate.family_paths[family])
    manifest = _read_json(
        manifest_path,
        f"{candidate.model}/{candidate.label}/{family} MANIFEST",
        provenance,
    )
    family_spec = protocol["families"][family]
    model_spec = protocol["models"][candidate.model]
    sidecar_path = _sidecar_path(manifest_path, manifest)
    sidecar = _read_json(
        sidecar_path,
        f"{candidate.model}/{candidate.label}/{family} sidecar",
        provenance,
    )
    envelope = _validate_manifest_and_sidecar(
        manifest_path=manifest_path,
        manifest=manifest,
        sidecar=sidecar,
        sidecar_path=sidecar_path,
        family=family,
        family_spec=family_spec,
        model_spec=model_spec,
        protocol=protocol,
    )
    findings = _fallback_findings(manifest) + _fallback_findings(sidecar)
    if findings:
        raise SelectionError(
            f"{manifest_path}: forbidden fallback evidence: {findings[:8]}"
        )
    metrics = (
        _load_c5_metrics(manifest_path, provenance)
        if family == "c5"
        else _load_fvs_metrics(manifest_path, candidate.model, provenance)
    )
    provenance.record(
        manifest_path,
        f"{candidate.model}/{candidate.label}/{family} MANIFEST post-validation",
    )
    provenance.record(
        sidecar_path,
        f"{candidate.model}/{candidate.label}/{family} sidecar post-validation",
    )
    _check_ce(metrics, family)
    return envelope, metrics


def evaluate_candidate(audit: CandidateAudit) -> None:
    if audit.rejection_reasons:
        return
    if set(audit.observed_envelopes) != {"c5", "fvs"} or set(audit.metrics) != {
        "c5",
        "fvs",
    }:
        audit.rejection_reasons.append("both C5 and FVS must validate")
        return
    for family, envelope in audit.observed_envelopes.items():
        if dict(envelope) != dict(audit.expected_envelope):
            audit.rejection_reasons.append(
                f"{family}: observed envelope {dict(envelope)} differs from predeclared {dict(audit.expected_envelope)}"
            )
    if audit.observed_envelopes["c5"] != audit.observed_envelopes["fvs"]:
        audit.rejection_reasons.append("C5 and FVS envelope fields differ")
    for family, metrics in audit.metrics.items():
        if metrics.prime_avg_ttft_ms > metrics.baseline_avg_ttft_ms:
            audit.rejection_reasons.append(
                f"{family}: Prime avg TTFT {metrics.prime_avg_ttft_ms:.9g} exceeds baseline {metrics.baseline_avg_ttft_ms:.9g}"
            )
        if metrics.prime_avg_e2e_ms > metrics.baseline_avg_e2e_ms:
            audit.rejection_reasons.append(
                f"{family}: Prime avg E2E {metrics.prime_avg_e2e_ms:.9g} exceeds baseline {metrics.baseline_avg_e2e_ms:.9g}"
            )
        if metrics.p95_ratio > 1.05:
            audit.rejection_reasons.append(
                f"{family}: Prime/baseline P95 TTFT ratio {metrics.p95_ratio:.9g} exceeds 1.05"
            )
    if not audit.rejection_reasons:
        audit.score = min(metrics.ce_ratio for metrics in audit.metrics.values())
        audit.eligible = True


def choose_per_model(
    audits: Sequence[CandidateAudit], tolerance: float
) -> Dict[str, str]:
    selected: Dict[str, str] = {}
    models = sorted({audit.model for audit in audits})
    for model in models:
        eligible = [
            audit for audit in audits if audit.model == model and audit.eligible
        ]
        if not eligible:
            continue
        best_score = max(
            float(audit.score) for audit in eligible if audit.score is not None
        )
        tied = [
            audit
            for audit in eligible
            if (best_score - float(audit.score)) / max(abs(best_score), 1e-12)
            <= tolerance
        ]
        winner = min(tied, key=lambda item: item.label)
        winner.selected = True
        selected[model] = winner.label
    return selected


def freeze_source_identity(
    audits: Sequence[CandidateAudit], expected_identity_count: int
) -> tuple[Dict[str, Dict[str, str]], str | None]:
    identity_rows = [
        (audit.model, audit.label, family, identity)
        for audit in audits
        for family, identity in audit.source_identities.items()
    ]
    if len(identity_rows) != expected_identity_count:
        return {}, (
            "source identity is unavailable for the complete six-round candidate cohort: "
            f"observed={len(identity_rows)}, expected={expected_identity_count}"
        )
    frozen: Dict[str, Dict[str, str]] = {}
    for repo in ("baseline_git", "faaslora_git"):
        distinct = {
            (identity[repo]["branch"], identity[repo]["commit"])
            for _, _, _, identity in identity_rows
        }
        if len(distinct) != 1:
            return (
                {},
                f"candidate rounds mix source versions for {repo}: {sorted(distinct)}",
            )
        branch, commit = next(iter(distinct))
        frozen[repo] = {"branch": branch, "commit": commit}
    return frozen, None


def _audit_json(audit: CandidateAudit) -> Dict[str, Any]:
    return {
        "model": audit.model,
        "candidate": audit.label,
        "eligible": audit.eligible,
        "selected": audit.selected,
        "score_min_family_ce_ratio": audit.score,
        "expected_envelope": dict(audit.expected_envelope),
        "observed_envelopes": {
            key: dict(value) for key, value in sorted(audit.observed_envelopes.items())
        },
        "source_identities": {
            family: {repo: dict(identity) for repo, identity in sorted(repos.items())}
            for family, repos in sorted(audit.source_identities.items())
        },
        "families": {
            family: {
                **metrics.__dict__,
                "prime_to_baseline_p95_ttft_ratio": metrics.p95_ratio,
                "prime_to_baseline_ce_ratio": metrics.ce_ratio,
            }
            for family, metrics in sorted(audit.metrics.items())
        },
        "rejection_reasons": list(audit.rejection_reasons),
    }


def _write_outputs(
    output_dir: Path, payload: Mapping[str, Any], audits: Sequence[CandidateAudit]
) -> None:
    target = output_dir.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=False)
    json_path = target / "global_envelope_selection.json"
    csv_path = target / "global_envelope_selection.csv"
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    columns = (
        "model",
        "candidate",
        "family",
        "eligible",
        "selected",
        "score_min_family_ce_ratio",
        "prime_avg_ttft_ms",
        "baseline_avg_ttft_ms",
        "prime_avg_e2e_ms",
        "baseline_avg_e2e_ms",
        "prime_p95_ttft_ms",
        "baseline_p95_ttft_ms",
        "prime_to_baseline_p95_ttft_ratio",
        "prime_ce",
        "baseline_ce",
        "prime_to_baseline_ce_ratio",
        "rejection_reasons",
    )
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for audit in sorted(audits, key=lambda item: (item.model, item.label)):
            families = sorted(audit.metrics) or [""]
            for family in families:
                metrics = audit.metrics.get(family)
                writer.writerow(
                    {
                        "model": audit.model,
                        "candidate": audit.label,
                        "family": family,
                        "eligible": str(audit.eligible).lower(),
                        "selected": str(audit.selected).lower(),
                        "score_min_family_ce_ratio": audit.score,
                        "prime_avg_ttft_ms": (
                            metrics.prime_avg_ttft_ms if metrics else None
                        ),
                        "baseline_avg_ttft_ms": (
                            metrics.baseline_avg_ttft_ms if metrics else None
                        ),
                        "prime_avg_e2e_ms": (
                            metrics.prime_avg_e2e_ms if metrics else None
                        ),
                        "baseline_avg_e2e_ms": (
                            metrics.baseline_avg_e2e_ms if metrics else None
                        ),
                        "prime_p95_ttft_ms": (
                            metrics.prime_p95_ttft_ms if metrics else None
                        ),
                        "baseline_p95_ttft_ms": (
                            metrics.baseline_p95_ttft_ms if metrics else None
                        ),
                        "prime_to_baseline_p95_ttft_ratio": (
                            metrics.p95_ratio if metrics else None
                        ),
                        "prime_ce": metrics.prime_ce if metrics else None,
                        "baseline_ce": metrics.baseline_ce if metrics else None,
                        "prime_to_baseline_ce_ratio": (
                            metrics.ce_ratio if metrics else None
                        ),
                        "rejection_reasons": " | ".join(audit.rejection_reasons),
                    }
                )


def run_selection(
    protocol_path: Path,
    candidates: Sequence[CandidateInput],
    output_dir: Path,
) -> tuple[Dict[str, Any], int]:
    provenance = ProvenanceRecorder()
    protocol = _read_json(protocol_path, "selection protocol", provenance)
    _validate_protocol(protocol)
    expected_keys = {
        (str(model), str(label))
        for model, model_spec in protocol["models"].items()
        for label in model_spec["candidates"]
    }
    observed_keys = [(candidate.model, candidate.label) for candidate in candidates]
    duplicates = sorted({key for key in observed_keys if observed_keys.count(key) > 1})
    missing = sorted(expected_keys - set(observed_keys))
    extra = sorted(set(observed_keys) - expected_keys)
    global_errors: list[str] = []
    if duplicates or missing or extra:
        global_errors.append(
            f"candidate matrix mismatch: missing={missing}, extra={extra}, duplicate={duplicates}"
        )

    audits: list[CandidateAudit] = []
    by_key = {(candidate.model, candidate.label): candidate for candidate in candidates}
    for model, label in sorted(expected_keys):
        audit = CandidateAudit(model, label, _expected_envelope(protocol, model, label))
        candidate = by_key.get((model, label))
        if candidate is None:
            audit.rejection_reasons.append("candidate was not explicitly supplied")
            audits.append(audit)
            continue
        for family in ("c5", "fvs"):
            try:
                audit.source_identities[family] = _read_candidate_source_identity(
                    candidate, family, provenance
                )
            except (SelectionError, ValueError, SystemExit) as exc:
                audit.rejection_reasons.append(f"{family}: {exc}")
                continue
            try:
                envelope, metrics = load_family(candidate, family, protocol, provenance)
                audit.observed_envelopes[family] = envelope
                audit.metrics[family] = metrics
            except (SelectionError, ValueError, SystemExit) as exc:
                audit.rejection_reasons.append(f"{family}: {exc}")
        audits.append(audit)

    frozen_source_identity, source_identity_error = freeze_source_identity(
        audits, len(expected_keys) * 2
    )
    if source_identity_error:
        global_errors.append(source_identity_error)
        for audit in audits:
            audit.rejection_reasons.append(source_identity_error)

    for audit in audits:
        evaluate_candidate(audit)

    tolerance = _finite_positive(
        protocol["selection"]["tie_relative_tolerance"], "tie_relative_tolerance"
    )
    selected = choose_per_model(audits, tolerance)
    missing_selection = sorted(set(protocol["models"]) - set(selected))
    if missing_selection:
        global_errors.append(
            f"no eligible candidate for models {missing_selection}; stop before held-out"
        )
    status = "selected" if not global_errors else "failed"
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "protocol": {
            "path": str(protocol_path.expanduser().resolve()),
            "schema_version": protocol["schema_version"],
            "sampling_seed": 41,
            "trace_role": "validation",
            "total_requests": 1000,
        },
        "selection_rule": dict(protocol["selection"]),
        "frozen_source_identity": frozen_source_identity,
        "selected_envelopes": {
            model: {
                "candidate": label,
                "envelope": dict(_expected_envelope(protocol, model, label)),
            }
            for model, label in sorted(selected.items())
        },
        "global_errors": global_errors,
        "candidates": [
            _audit_json(audit)
            for audit in sorted(audits, key=lambda item: (item.model, item.label))
        ],
        "input_files": provenance.rows(),
        "immutability": "output directory was required not to exist and files were created with exclusive mode",
    }
    _write_outputs(output_dir, payload, audits)
    return payload, 0 if status == "selected" else 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        nargs=4,
        metavar=("MODEL", "LABEL", "C5_ROUND", "FVS_ROUND"),
        help="Explicit candidate pair; repeat for every candidate declared by the protocol.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    candidates = [
        CandidateInput(
            model=str(model),
            label=str(label),
            family_paths={"c5": Path(c5), "fvs": Path(fvs)},
        )
        for model, label, c5, fvs in args.candidate
    ]
    try:
        payload, exit_code = run_selection(args.protocol, candidates, args.output_dir)
    except SelectionError as exc:
        raise SystemExit(
            f"global envelope selection failed before audit output: {exc}"
        ) from exc
    print(
        f"global envelope selection {payload['status']}: "
        f"selected={payload['selected_envelopes']} output={args.output_dir.resolve()}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
