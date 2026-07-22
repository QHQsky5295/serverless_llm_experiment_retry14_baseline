#!/usr/bin/env python3
"""Analyze selected-replica adapter readiness from completed FaaSLoRA rounds."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from eurosys27_v2_provenance import (
        FormalAnalysisIdentity,
        FormalProvenanceIndex,
        build_formal_provenance_index,
        validate_formal_analysis_sources,
    )
except ImportError:  # Package imports used by tests.
    from scripts.eurosys27_v2_provenance import (
        FormalAnalysisIdentity,
        FormalProvenanceIndex,
        build_formal_provenance_index,
        validate_formal_analysis_sources,
    )


SCENARIO_ORDER = (
    "v2_elastic_only",
    "v2_hierarchical_no_coord",
    "v2_full",
    "faaslora_nvme",
    "faaslora_no_coord",
    "faaslora_full",
)
SCENARIO_LABELS = {
    "v2_elastic_only": "ElasticOnly",
    "v2_hierarchical_no_coord": "+HierarchicalResidency",
    "v2_full": "PrimeLoRA",
    "faaslora_nvme": "PrimeLoRA-NVMe",
    "faaslora_no_coord": "PrimeLoRA-NoCoord",
    "faaslora_full": "PrimeLoRA",
}
SHORT_LABELS = {
    "v2_elastic_only": "ElasticOnly",
    "v2_hierarchical_no_coord": "+Hierarchy",
    "v2_full": "PrimeLoRA",
    "faaslora_nvme": "NVMe",
    "faaslora_no_coord": "NoCoord",
    "faaslora_full": "PrimeLoRA",
}
TIER_ORDER = ("gpu", "host", "nvme", "remote")
TIER_LABELS = {
    "gpu": "GPU-ready",
    "host": "HOST-prepared",
    "nvme": "NVMe-prepared",
    "remote": "Remote-cold",
}
TIER_COLORS = {
    "gpu": "#78B87A",
    "host": "#7FA7D9",
    "nvme": "#F2B36D",
    "remote": "#C84D4D",
}
METRIC_CMAP = "YlGnBu"

DIAGNOSTIC_EXPECTED_TOTAL = 4000
DIAGNOSTIC_EXPECTED_PHASES = 8
DIAGNOSTIC_MIN_SCALE_UP_EVENTS = 8
DIAGNOSTIC_MIN_FIRST_SERVICE = 20
DIAGNOSTIC_MAX_COLD_RUNS = 2

FORMAL_READINESS_CAMPAIGN_KIND = "v2_a2_a3_ablation"
FORMAL_READINESS_SEED = 43
FORMAL_READINESS_REQUESTS = 4000
FORMAL_READINESS_ADAPTERS = 500
FORMAL_READINESS_BANDWIDTH_MIB_S = 250.0
FORMAL_READINESS_GENERATION_CONTRACT = "legacy"
FORMAL_READINESS_DATASET_PROFILE = "azure_sharegpt_rep4000"
FORMAL_READINESS_MODELS: Mapping[str, Dict[str, Any]] = {
    "llama2_7b_main_v2_publicmix": {
        "model": "llama2_7b",
        "workload_profile": "llama2_7b_auto500_formal4000_s8",
        "scenarios": (
            "v2_elastic_only",
            "v2_hierarchical_no_coord",
            "v2_full",
        ),
    },
    "llama32_3b_main_modelscope": {
        "model": "llama32_3b",
        "workload_profile": "llama32_3b_auto500_formal4000_s8",
        "scenarios": ("v2_full",),
    },
}
FORMAL_READINESS_FEATURE_GATES: Mapping[str, Tuple[bool, bool, bool, bool, bool]] = {
    "v2_elastic_only": (False, False, False, False, False),
    "v2_hierarchical_no_coord": (True, True, True, False, False),
    "v2_full": (True, True, True, True, True),
}


@dataclass
class ScenarioRecords:
    name: str
    model: str
    run_tag: str
    source: Path
    records: List[Dict[str, Any]]
    dispatch_tier_records: int
    total: int | None = None
    completed: int | None = None
    failed: int | None = None
    phase_count: int | None = None
    scale_up_event_count: int | None = None
    tier_invariant_conflicts: int = 0
    phase_results: Tuple[Dict[str, Any], ...] = ()
    cold_cache_reset_before_run: bool | None = None
    configuration_fingerprint: str = ""
    diagnostic_schema_errors: Tuple[str, ...] = ()
    dispatch_validation_errors: Tuple[str, ...] = ()

    @property
    def used_dispatch_before_tier(self) -> bool:
        return self.dispatch_tier_records > 0

    @property
    def all_records_have_dispatch_tier(self) -> bool:
        return self.dispatch_tier_records == len(self.records)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 8.8,
            "axes.linewidth": 0.65,
            "grid.linewidth": 0.55,
            "lines.linewidth": 1.35,
            "patch.linewidth": 0.45,
        }
    )


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def percentile(values: Sequence[float], q: float) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.percentile(np.asarray(vals, dtype=float), q))


def mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.mean(np.asarray(vals, dtype=float)))


def readiness_tier(record: Dict[str, Any], *, allow_proxy: bool = True) -> str:
    tier = str(record.get("readiness_tier_before_dispatch") or "").strip().lower()
    if not tier and allow_proxy:
        tier = str(record.get("cache_tier") or "unknown").strip().lower()
    return tier


def service_tier(record: Dict[str, Any]) -> str:
    return str(record.get("cache_tier") or "unknown").strip().lower()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot parse JSON result {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"result payload must be a JSON object: {path}")
    return payload


def _model_identity(payload: Dict[str, Any], path: Path) -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    profile_selection = (
        metadata.get("profile_selection")
        if isinstance(metadata.get("profile_selection"), dict)
        else {}
    )
    raw = (
        profile_selection.get("model")
        or metadata.get("model_profile")
        or metadata.get("model")
        or payload.get("model")
    )
    if raw:
        text = str(raw).strip().rstrip("/")
        return Path(text).name or text
    return f"unknown_model:{path.stem}"


def _run_tag(payload: Dict[str, Any], path: Path) -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    raw = metadata.get("results_tag") or metadata.get("run_tag") or payload.get("run_tag")
    return str(raw or path.stem.removesuffix(".json")).strip()


def _formal_fail(message: str) -> None:
    raise SystemExit(f"formal readiness gate failed: {message}")


def _formal_sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        _formal_fail(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _formal_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        _formal_fail(f"{label} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError):
        _formal_fail(f"{label} must be an integer")
    try:
        exact = float(value) == float(number)
    except (TypeError, ValueError):
        exact = False
    if not exact:
        _formal_fail(f"{label} must be an integer")
    return number


def _formal_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        _formal_fail(f"{label} must be numeric")
    if not math.isfinite(number):
        _formal_fail(f"{label} must be finite")
    return number


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_manifest_path(manifest_path: Path, value: Any, label: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        _formal_fail(f"{manifest_path}:{label} is empty")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _validate_manifest_artifact(
    manifest_path: Path,
    block: Any,
    *,
    label: str,
) -> Tuple[Path, str, Dict[str, Any]]:
    if not isinstance(block, dict):
        _formal_fail(f"{manifest_path}:{label} must be an object")
    artifact_path = _resolve_manifest_path(manifest_path, block.get("path"), f"{label}.path")
    if not artifact_path.is_file():
        _formal_fail(f"{manifest_path}:{label} file is missing: {artifact_path}")
    expected_sha = _formal_sha256(block.get("sha256"), f"{manifest_path}:{label}.sha256")
    actual_sha = _file_sha256(artifact_path)
    if actual_sha != expected_sha:
        _formal_fail(
            f"{manifest_path}:{label} SHA-256 mismatch; "
            f"recorded={expected_sha}, actual={actual_sha}"
        )
    return artifact_path, expected_sha, _read_json(artifact_path)


def _current_source_commit(repo: Path) -> str:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo.resolve()), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:
        _formal_fail(f"cannot resolve current FaaSLoRA source revision: {exc}")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit.lower()):
        _formal_fail(f"current FaaSLoRA source revision is invalid: {commit!r}")
    return commit.lower()


def _load_scenarios_from_sources(
    sources: Sequence[Path],
    *,
    require_dispatch_tier: bool,
    include_empty: bool,
) -> List[ScenarioRecords]:
    scenarios: List[ScenarioRecords] = []
    seen: Dict[Tuple[str, str, str], Path] = {}
    for source in sorted({Path(path).resolve() for path in sources}):
        for scenario in load_scenarios(
            source,
            require_dispatch_tier=require_dispatch_tier,
            include_empty=include_empty,
        ):
            key = (scenario.model, scenario.name, scenario.run_tag)
            if key in seen:
                _formal_fail(
                    "duplicate result identity across inputs "
                    f"{key!r}: {seen[key]} and {scenario.source}"
                )
            seen[key] = scenario.source
            scenarios.append(scenario)
    if not scenarios:
        _formal_fail("no request-level readiness records were found")
    rank = {name: index for index, name in enumerate(SCENARIO_ORDER)}
    return sorted(
        scenarios,
        key=lambda item: (
            item.model,
            rank.get(item.name, len(rank)),
            item.name,
            item.run_tag,
        ),
    )


def load_formal_v2_readiness_campaign(
    inputs: Sequence[Path],
    *,
    repo: Path | None = None,
) -> Tuple[List[ScenarioRecords], FormalProvenanceIndex, Dict[str, Any]]:
    """Load and fail-closed validate the pre-registered dispatch-readiness matrix."""

    resolved_inputs = [Path(path).expanduser().resolve() for path in inputs]
    provenance = build_formal_provenance_index(resolved_inputs)
    repo_path = (repo or Path(__file__).resolve().parents[1]).resolve()
    current_commit = _current_source_commit(repo_path)

    bindings: Dict[Path, Dict[str, Any]] = {}
    manifest_reports: List[Dict[str, Any]] = []
    evidence_paths: List[str] = []
    trace_hashes_by_model: DefaultDict[str, set[str]] = defaultdict(set)
    subset_hashes_by_model: DefaultDict[str, set[str]] = defaultdict(set)

    for manifest_path in provenance.manifest_paths:
        manifest = _read_json(manifest_path)
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            _formal_fail(
                f"{manifest_path}: readiness formal mode accepts only FaaSLoRA "
                "ablation manifests with entries[]"
            )
        family = manifest.get("configuration_family")
        if not isinstance(family, dict):
            _formal_fail(f"{manifest_path}: configuration_family must be an object")
        if str(family.get("campaign_kind") or "") != FORMAL_READINESS_CAMPAIGN_KIND:
            _formal_fail(
                f"{manifest_path}: configuration_family.campaign_kind must be "
                f"{FORMAL_READINESS_CAMPAIGN_KIND!r}"
            )
        model_profile = str(manifest.get("model_profile") or family.get("model_profile") or "")
        model_spec = FORMAL_READINESS_MODELS.get(model_profile)
        if model_spec is None:
            _formal_fail(f"{manifest_path}: unsupported model_profile={model_profile!r}")
        model = str(model_spec["model"])
        expected_scenarios = tuple(str(item) for item in model_spec["scenarios"])
        expected_workload = str(model_spec["workload_profile"])
        expected_family = {
            "campaign_kind": FORMAL_READINESS_CAMPAIGN_KIND,
            "model_profile": model_profile,
            "dataset_profile": FORMAL_READINESS_DATASET_PROFILE,
            "workload_profile": expected_workload,
            "selected_num_adapters": FORMAL_READINESS_ADAPTERS,
            "generation_contract": FORMAL_READINESS_GENERATION_CONTRACT,
        }
        for field, expected in expected_family.items():
            if family.get(field) != expected:
                _formal_fail(
                    f"{manifest_path}: configuration_family.{field}="
                    f"{family.get(field)!r}, expected {expected!r}"
                )
        for field, expected in (
            ("dataset_profile", FORMAL_READINESS_DATASET_PROFILE),
            ("workload_profile", expected_workload),
        ):
            if str(manifest.get(field) or "") != expected:
                _formal_fail(
                    f"{manifest_path}: {field}={manifest.get(field)!r}, expected {expected!r}"
                )
        if _formal_int(
            manifest.get("multi_cycle_phases"), f"{manifest_path}:multi_cycle_phases"
        ) != DIAGNOSTIC_EXPECTED_PHASES:
            _formal_fail(
                f"{manifest_path}: multi_cycle_phases must be {DIAGNOSTIC_EXPECTED_PHASES}"
            )
        if not math.isclose(
            _formal_float(
                manifest.get("idle_between_phases_s"),
                f"{manifest_path}:idle_between_phases_s",
            ),
            2.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            _formal_fail(f"{manifest_path}: idle_between_phases_s must be 2.0")
        if manifest.get("mechanism_trigger_gate_required") is not True:
            _formal_fail(
                f"{manifest_path}: mechanism_trigger_gate_required must be true"
            )
        _formal_sha256(
            manifest.get("tuning_env_sha256"), f"{manifest_path}:tuning_env_sha256"
        )
        if not math.isclose(
            _formal_float(
                manifest.get("bandwidth_mib_s"), f"{manifest_path}:bandwidth_mib_s"
            ),
            FORMAL_READINESS_BANDWIDTH_MIB_S,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            _formal_fail(
                f"{manifest_path}: bandwidth_mib_s must be "
                f"{FORMAL_READINESS_BANDWIDTH_MIB_S:g}"
            )

        snapshot = manifest.get("code_snapshot")
        if not isinstance(snapshot, dict):
            _formal_fail(f"{manifest_path}: code_snapshot must be an object")
        recorded_commit = str(snapshot.get("git_commit") or "").strip().lower()
        if recorded_commit != current_commit:
            _formal_fail(
                f"{manifest_path}: source commit {recorded_commit!r} does not match "
                f"current FaaSLoRA HEAD {current_commit!r}"
            )

        trace_path, trace_sha, trace_payload = _validate_manifest_artifact(
            manifest_path, manifest.get("shared_trace"), label="shared_trace"
        )
        subset_path, subset_sha, subset_payload = _validate_manifest_artifact(
            manifest_path,
            manifest.get("shared_adapter_subset"),
            label="shared_adapter_subset",
        )
        trace = manifest["shared_trace"]
        subset = manifest["shared_adapter_subset"]
        for label, payload, block in (
            ("shared_trace", trace_payload, trace),
            ("shared_adapter_subset", subset_payload, subset),
        ):
            if _formal_int(
                block.get("sampling_seed"), f"{manifest_path}:{label}.sampling_seed"
            ) != FORMAL_READINESS_SEED:
                _formal_fail(
                    f"{manifest_path}:{label}.sampling_seed must be {FORMAL_READINESS_SEED}"
                )
            if _formal_int(
                payload.get("sampling_seed"),
                f"{manifest_path}:{label} artifact sampling_seed",
            ) != FORMAL_READINESS_SEED:
                _formal_fail(
                    f"{manifest_path}:{label} artifact seed must be {FORMAL_READINESS_SEED}"
                )
            for field, expected in (
                ("model_profile", model_profile),
                ("dataset_profile", FORMAL_READINESS_DATASET_PROFILE),
                ("workload_profile", expected_workload),
            ):
                if payload.get(field) != expected:
                    _formal_fail(
                        f"{manifest_path}:{label} artifact {field}="
                        f"{payload.get(field)!r}, expected {expected!r}"
                    )
        if _formal_int(trace.get("requests"), f"{manifest_path}:shared_trace.requests") != FORMAL_READINESS_REQUESTS:
            _formal_fail(
                f"{manifest_path}: shared_trace.requests must be {FORMAL_READINESS_REQUESTS}"
            )
        raw_requests = trace_payload.get("requests")
        if not isinstance(raw_requests, list) or len(raw_requests) != FORMAL_READINESS_REQUESTS:
            _formal_fail(
                f"{manifest_path}: shared trace artifact must contain exactly "
                f"{FORMAL_READINESS_REQUESTS} requests"
            )
        if _formal_int(
            trace.get("selected_num_adapters"),
            f"{manifest_path}:shared_trace.selected_num_adapters",
        ) != FORMAL_READINESS_ADAPTERS:
            _formal_fail(
                f"{manifest_path}: shared_trace.selected_num_adapters must be "
                f"{FORMAL_READINESS_ADAPTERS}"
            )
        if _formal_int(
            subset.get("selected_num_adapters"),
            f"{manifest_path}:shared_adapter_subset.selected_num_adapters",
        ) != FORMAL_READINESS_ADAPTERS:
            _formal_fail(
                f"{manifest_path}: shared_adapter_subset.selected_num_adapters must be "
                f"{FORMAL_READINESS_ADAPTERS}"
            )
        adapters = subset_payload.get("adapters")
        if not isinstance(adapters, list) or len(adapters) != FORMAL_READINESS_ADAPTERS:
            _formal_fail(
                f"{manifest_path}: shared adapter subset artifact must contain exactly "
                f"{FORMAL_READINESS_ADAPTERS} adapters"
            )
        trace_hashes_by_model[model].add(trace_sha)
        subset_hashes_by_model[model].add(subset_sha)

        manifest_scenarios = manifest.get("scenarios")
        if not isinstance(manifest_scenarios, list) or not manifest_scenarios:
            _formal_fail(f"{manifest_path}: scenarios must be a non-empty list")
        scenario_names = tuple(str(item) for item in manifest_scenarios)
        if len(set(scenario_names)) != len(scenario_names):
            _formal_fail(f"{manifest_path}: scenarios contains duplicates")
        unknown = sorted(set(scenario_names) - set(expected_scenarios))
        if unknown:
            _formal_fail(
                f"{manifest_path}: unexpected {model} readiness scenario(s): {unknown}"
            )
        if len(entries) != len(scenario_names):
            _formal_fail(
                f"{manifest_path}: entries/scenarios cardinality mismatch; "
                f"entries={len(entries)}, scenarios={len(scenario_names)}"
            )
        entry_scenarios = [
            str(entry.get("scenario") or "") if isinstance(entry, dict) else ""
            for entry in entries
        ]
        if sorted(entry_scenarios) != sorted(scenario_names):
            _formal_fail(f"{manifest_path}: entries do not exactly match scenarios")

        evidence = manifest.get("seed41_validation_evidence")
        if not isinstance(evidence, dict):
            _formal_fail(f"{manifest_path}: seed41_validation_evidence is missing")
        evidence_path = _resolve_manifest_path(
            manifest_path, evidence.get("path"), "seed41_validation_evidence.path"
        )
        evidence_paths.append(str(evidence_path))
        non_feature_hash = _formal_sha256(
            manifest.get("non_feature_frozen_config_sha256"),
            f"{manifest_path}:non_feature_frozen_config_sha256",
        )

        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                _formal_fail(f"{manifest_path}:entries[{index}] must be an object")
            if entry.get("exists") is not True:
                _formal_fail(
                    f"{manifest_path}:entries[{index}].exists must be true in a "
                    "completed readiness campaign"
                )
            scenario_name = str(entry.get("scenario") or "")
            source = _resolve_manifest_path(
                manifest_path,
                entry.get("result_json"),
                f"entries[{index}].result_json",
            )
            if source in bindings:
                _formal_fail(
                    f"raw readiness result is covered by multiple campaign entries: {source}"
                )
            if source not in provenance.records_by_source:
                _formal_fail(
                    f"{manifest_path}: result has no validated provenance record: {source}"
                )
            config_sha = _formal_sha256(
                entry.get("system_resolved_config_sha256"),
                f"{manifest_path}:entries[{index}].system_resolved_config_sha256",
            )
            if _formal_sha256(
                entry.get("non_feature_frozen_config_sha256"),
                f"{manifest_path}:entries[{index}].non_feature_frozen_config_sha256",
            ) != non_feature_hash:
                _formal_fail(
                    f"{manifest_path}:entries[{index}] non-feature config hash mismatch"
                )
            result = _read_json(source)
            metadata = result.get("metadata")
            if not isinstance(metadata, dict):
                _formal_fail(f"{source}: metadata must be an object")
            if metadata.get("formal_run") is not True:
                _formal_fail(f"{source}: metadata.formal_run must be true")
            if str(metadata.get("trace_role") or "").strip().lower() != "heldout":
                _formal_fail(f"{source}: metadata.trace_role must be 'heldout'")
            for field in ("sampling_seed", "workload_seed", "generation_seed"):
                if _formal_int(metadata.get(field), f"{source}:metadata.{field}") != FORMAL_READINESS_SEED:
                    _formal_fail(
                        f"{source}: metadata.{field} must be {FORMAL_READINESS_SEED}"
                    )
            if _formal_int(
                metadata.get("total_requests"), f"{source}:metadata.total_requests"
            ) != FORMAL_READINESS_REQUESTS:
                _formal_fail(
                    f"{source}: metadata.total_requests must be {FORMAL_READINESS_REQUESTS}"
                )
            if _formal_int(
                metadata.get("num_adapters"), f"{source}:metadata.num_adapters"
            ) != FORMAL_READINESS_ADAPTERS:
                _formal_fail(
                    f"{source}: metadata.num_adapters must be {FORMAL_READINESS_ADAPTERS}"
                )
            if str(metadata.get("generation_contract") or "").strip().lower() != FORMAL_READINESS_GENERATION_CONTRACT:
                _formal_fail(
                    f"{source}: metadata.generation_contract must be "
                    f"{FORMAL_READINESS_GENERATION_CONTRACT!r}"
                )
            if not math.isclose(
                _formal_float(metadata.get("bandwidth_mib_s"), f"{source}:metadata.bandwidth_mib_s"),
                FORMAL_READINESS_BANDWIDTH_MIB_S,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                _formal_fail(
                    f"{source}: metadata.bandwidth_mib_s must be "
                    f"{FORMAL_READINESS_BANDWIDTH_MIB_S:g}"
                )
            selection = metadata.get("profile_selection")
            if not isinstance(selection, dict):
                _formal_fail(f"{source}: metadata.profile_selection must be an object")
            for field, expected in (
                ("model", model_profile),
                ("dataset", FORMAL_READINESS_DATASET_PROFILE),
                ("workload", expected_workload),
            ):
                if selection.get(field) != expected:
                    _formal_fail(
                        f"{source}: metadata.profile_selection.{field}="
                        f"{selection.get(field)!r}, expected {expected!r}"
                    )
            if _formal_sha256(
                metadata.get("shared_trace_sha256"),
                f"{source}:metadata.shared_trace_sha256",
            ) != trace_sha:
                _formal_fail(f"{source}: result/manifest shared trace SHA mismatch")
            if _formal_sha256(
                metadata.get("shared_adapter_subset_sha256"),
                f"{source}:metadata.shared_adapter_subset_sha256",
            ) != subset_sha:
                _formal_fail(f"{source}: result/manifest adapter subset SHA mismatch")
            if _formal_sha256(
                metadata.get("system_resolved_config_sha256"),
                f"{source}:metadata.system_resolved_config_sha256",
            ) != config_sha:
                _formal_fail(f"{source}: result/manifest system config SHA mismatch")
            if _formal_sha256(
                metadata.get("non_feature_frozen_config_sha256"),
                f"{source}:metadata.non_feature_frozen_config_sha256",
            ) != non_feature_hash:
                _formal_fail(f"{source}: result/manifest non-feature config SHA mismatch")
            detailed = result.get("detailed_results")
            if not isinstance(detailed, dict) or sorted(detailed) != [scenario_name]:
                _formal_fail(
                    f"{source}: detailed_results must contain exactly {scenario_name!r}"
                )
            coordination = metadata.get("scenario_coordination")
            coordination = coordination.get(scenario_name) if isinstance(coordination, dict) else None
            if not isinstance(coordination, dict):
                _formal_fail(
                    f"{source}: metadata.scenario_coordination.{scenario_name} is missing"
                )
            if coordination.get("cold_cache_reset_before_run") is not True:
                _formal_fail(f"{source}: cold_cache_reset_before_run must be true")
            gates = coordination.get("feature_gates")
            if not isinstance(gates, dict):
                _formal_fail(f"{source}: feature_gates must be an object")
            observed_gates = (
                as_bool(gates.get("readiness_routing_enabled")),
                as_bool(gates.get("scale_up_handoff_enabled")),
                as_bool(gates.get("hierarchical_residency_enabled")),
                as_bool(gates.get("coordination_enabled")),
                as_bool(gates.get("effective_capacity_admission_enabled")),
            )
            if observed_gates != FORMAL_READINESS_FEATURE_GATES[scenario_name]:
                _formal_fail(
                    f"{source}: feature-gate tuple {observed_gates!r} does not match "
                    f"{scenario_name}"
                )
            bindings[source] = {
                "manifest": manifest_path,
                "model": model,
                "model_profile": model_profile,
                "scenario": scenario_name,
                "seed": FORMAL_READINESS_SEED,
                "trace_sha256": trace_sha,
                "subset_sha256": subset_sha,
                "system_resolved_config_sha256": config_sha,
                "trace_path": trace_path,
                "subset_path": subset_path,
            }

        manifest_reports.append(
            {
                "manifest": str(manifest_path),
                "model": model,
                "model_profile": model_profile,
                "seed": FORMAL_READINESS_SEED,
                "requests": FORMAL_READINESS_REQUESTS,
                "scenarios": list(scenario_names),
                "shared_trace_sha256": trace_sha,
                "shared_adapter_subset_sha256": subset_sha,
                "seed41_validation_evidence": str(evidence_path),
                "source_commit": current_commit,
            }
        )

    for model in ("llama2_7b", "llama32_3b"):
        if trace_hashes_by_model[model] and len(trace_hashes_by_model[model]) != 1:
            _formal_fail(
                f"model={model}: cold runs/variants do not share one trace SHA: "
                f"{sorted(trace_hashes_by_model[model])}"
            )
        if subset_hashes_by_model[model] and len(subset_hashes_by_model[model]) != 1:
            _formal_fail(
                f"model={model}: cold runs/variants do not share one adapter subset SHA: "
                f"{sorted(subset_hashes_by_model[model])}"
            )

    scenarios = _load_scenarios_from_sources(
        list(bindings), require_dispatch_tier=True, include_empty=True
    )
    by_source: Dict[Path, List[ScenarioRecords]] = defaultdict(list)
    for scenario in scenarios:
        by_source[scenario.source.resolve()].append(scenario)
    identities: List[FormalAnalysisIdentity] = []
    groups: DefaultDict[Tuple[str, str], List[ScenarioRecords]] = defaultdict(list)
    normalized: List[ScenarioRecords] = []
    for source, binding in sorted(bindings.items(), key=lambda item: str(item[0])):
        loaded = by_source.get(source, [])
        if len(loaded) != 1 or loaded[0].name != binding["scenario"]:
            _formal_fail(
                f"{source}: expected exactly one loaded scenario {binding['scenario']!r}, "
                f"observed {[item.name for item in loaded]!r}"
            )
        scenario = loaded[0]
        scenario.model = str(binding["model"])
        normalized.append(scenario)
        key = (scenario.model, scenario.name)
        groups[key].append(scenario)
        identities.append(
            FormalAnalysisIdentity(
                source=source,
                model=scenario.model,
                variant=scenario.name,
                seed=FORMAL_READINESS_SEED,
            )
        )

    expected_groups = {
        (str(spec["model"]), str(scenario))
        for spec in FORMAL_READINESS_MODELS.values()
        for scenario in spec["scenarios"]
    }
    actual_groups = set(groups)
    if actual_groups != expected_groups:
        _formal_fail(
            "readiness matrix mismatch; "
            f"missing={sorted(expected_groups - actual_groups)}, "
            f"unexpected={sorted(actual_groups - expected_groups)}"
        )
    for key, runs in sorted(groups.items()):
        if not 1 <= len(runs) <= DIAGNOSTIC_MAX_COLD_RUNS:
            _formal_fail(
                f"model={key[0]},variant={key[1]} has {len(runs)} cold runs; "
                f"expected one or at most {DIAGNOSTIC_MAX_COLD_RUNS}"
            )

    validate_formal_analysis_sources(
        provenance,
        identities,
        analysis_label="V2 dispatch-time readiness",
    )
    normalized.sort(
        key=lambda item: (
            item.model,
            SCENARIO_ORDER.index(item.name)
            if item.name in SCENARIO_ORDER
            else len(SCENARIO_ORDER),
            item.run_tag,
        )
    )
    report = {
        "analysis": "eurosys27_v2_formal_dispatch_readiness_protocol",
        "gate_version": "eurosys27_v2_readiness_formal_v1",
        "passed": True,
        "source_repo": str(repo_path),
        "source_commit": current_commit,
        "formal_seed": FORMAL_READINESS_SEED,
        "requests_per_run": FORMAL_READINESS_REQUESTS,
        "max_cold_runs_per_model_variant": DIAGNOSTIC_MAX_COLD_RUNS,
        "required_matrix": [
            {"model": model, "scenario": scenario}
            for model, scenario in sorted(expected_groups)
        ],
        "observed_run_counts": [
            {"model": model, "scenario": scenario, "run_count": len(groups[(model, scenario)])}
            for model, scenario in sorted(expected_groups)
        ],
        "manifests": manifest_reports,
        "seed41_validation_evidence": sorted(set(evidence_paths)),
    }
    return normalized, provenance, report


def _strict_dispatch_errors(
    *, source: Path, scenario: str, records: Sequence[Dict[str, Any]]
) -> List[str]:
    """Validate dispatch-time readiness fields for every successful LoRA request."""
    errors: List[str] = []
    flag_expectations = {
        "adapter_gpu_ready_before_dispatch": lambda tier: tier == "gpu",
        "adapter_local_ready_before_dispatch": lambda tier: tier in ("host", "nvme"),
        "adapter_remote_cold_before_dispatch": lambda tier: tier == "remote",
        "adapter_replica_mismatch": lambda tier: tier != "gpu",
        "remote_mismatch": lambda tier: tier == "remote",
    }
    for index, record in enumerate(records):
        request_id = str(record.get("request_id") or f"index={index}")
        tier = readiness_tier(record, allow_proxy=False)
        prefix = f"{source}:{scenario}:{request_id}"
        if tier not in TIER_ORDER:
            errors.append(f"{prefix}: invalid or missing readiness_tier_before_dispatch={tier!r}")
            continue
        for field, expected_fn in flag_expectations.items():
            if field not in record:
                errors.append(f"{prefix}: missing {field}")
                continue
            expected = bool(expected_fn(tier))
            actual = as_bool(record.get(field))
            if actual != expected:
                errors.append(
                    f"{prefix}: {field}={actual!r}, expected {expected!r} for tier={tier!r}"
                )
    return errors


def _tier_invariant_conflict_count(records: Sequence[Dict[str, Any]]) -> int:
    """Count contradictory tier/boolean observations, excluding missing fields."""
    expectations = {
        "adapter_gpu_ready_before_dispatch": lambda tier: tier == "gpu",
        "adapter_local_ready_before_dispatch": lambda tier: tier in ("host", "nvme"),
        "adapter_remote_cold_before_dispatch": lambda tier: tier == "remote",
        "adapter_replica_mismatch": lambda tier: tier != "gpu",
        "remote_mismatch": lambda tier: tier == "remote",
    }
    conflicts = 0
    for record in records:
        tier = readiness_tier(record, allow_proxy=False)
        if tier not in TIER_ORDER:
            continue
        for field, expected_fn in expectations.items():
            if field in record and as_bool(record.get(field)) != bool(expected_fn(tier)):
                conflicts += 1
    return conflicts


def _diagnostic_int(
    detail: Dict[str, Any], key: str, errors: List[str]
) -> int | None:
    value = detail.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"missing or non-numeric detailed_results.{key}")
        return None
    number = int(value)
    if float(value) != float(number) or number < 0:
        errors.append(f"invalid detailed_results.{key}={value!r}")
        return None
    return number


def _diagnostic_configuration_fingerprint(
    *, metadata: Dict[str, Any], scenario: str, model: str
) -> str:
    """Hash result-affecting configuration while excluding run/cache identities."""
    coordination_by_scenario = metadata.get("scenario_coordination")
    if not isinstance(coordination_by_scenario, dict):
        coordination_by_scenario = {}
    coordination = coordination_by_scenario.get(scenario)
    if not isinstance(coordination, dict):
        coordination = {}
    coordination_keys = (
        "instance_mode",
        "min_instances",
        "max_instances",
        "routing_policy",
        "max_concurrent_loads",
        "warm_pool_size",
        "feature_gates",
        "arrival_window_s",
        "scale_eval_interval_s",
        "scale_cooldown_s",
        "scale_decision_interval",
        "scale_up_alpha",
        "scale_up_t_min",
        "scale_down_beta",
        "scale_down_duration_s",
        "scale_down_duration_policy",
        "scale_down_cooldown_s",
        "scale_down_cooldown_policy",
        "ttft_slo_ms",
        "ttft_latency_scale_up_threshold_ms",
        "ttft_latency_scale_down_threshold_ms",
        "host_capacity_mb",
        "max_model_len",
        "max_input_len",
        "max_output_tokens_cap",
        "max_num_seqs",
        "requested_runtime_concurrency_cap",
        "runtime_concurrency_cap",
        "max_loras",
    )
    metadata_keys = (
        "backend",
        "device_id",
        "visible_device_ids",
        "visible_gpu_count",
        "tensor_parallel_size",
        "gpu_per_request",
        "runtime_gpu_count",
        "parallelism_topology",
        "max_model_len",
        "max_input_len",
        "max_output_tokens_cap",
        "max_num_seqs",
        "max_num_batched_tokens",
        "requested_runtime_concurrency_cap",
        "runtime_concurrency_cap",
        "max_loras",
        "max_cpu_loras",
        "generation_seed",
        "workload_seed",
        "generation_contract",
        "fixed_output_max_tokens",
        "fixed_prompt_max_tokens",
        "sampling_strategy",
        "configured_time_scale_factor",
        "effective_time_scale_factor",
        "workload_timing_mode",
        "num_adapters",
        "active_adapter_cap",
        "hotset_rotation_requests",
        "hotset_rotation_mode",
        "hotset_overlap_fraction",
        "bandwidth_mib_s",
        "bandwidth_gbit_s",
        "bandwidth_limit_mode",
        "total_requests",
        "shared_trace_sha256",
        "shared_adapter_subset_sha256",
        "shared_trace_load_profile",
        "profile_selection",
        "preset_name",
    )
    applied_overrides = metadata.get("applied_env_overrides")
    if not isinstance(applied_overrides, dict):
        applied_overrides = {}
    stable_overrides = {
        str(key): value
        for key, value in applied_overrides.items()
        if not str(key).endswith(("_DIR", "_PATH"))
        and str(key) not in {"FAASLORA_RESULTS_TAG"}
    }
    identity = {
        "model": model,
        "scenario": scenario,
        "metadata": {key: metadata.get(key) for key in metadata_keys},
        "scenario_coordination": {
            key: coordination.get(key) for key in coordination_keys
        },
        "applied_env_overrides": stable_overrides,
    }
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ttft_ms(record: Dict[str, Any]) -> float:
    return as_float(record.get("overall_ttft_ms"), as_float(record.get("ttft_ms"), 0.0))


def adapter_prep_ms(record: Dict[str, Any]) -> float:
    return as_float(record.get("lora_io_ms")) + as_float(record.get("defer_ms"))


def runtime_ttft_ms(record: Dict[str, Any]) -> float:
    return as_float(record.get("vllm_ttft_ms"), as_float(record.get("service_ttft_ms"), 0.0))


def dispatch_wait_ms(record: Dict[str, Any]) -> float:
    return sum(
        as_float(record.get(key))
        for key in (
            "ingress_queue_wait_ms",
            "dispatch_admission_wait_ms",
            "dispatch_window_wait_ms",
            "runtime_slot_wait_ms",
        )
    )


def load_scenarios(
    input_path: Path,
    *,
    require_dispatch_tier: bool = False,
    include_empty: bool = False,
) -> List[ScenarioRecords]:
    if input_path.is_file():
        candidates = [input_path]
    else:
        candidates = sorted(
            set(input_path.rglob("*_result.json"))
            | set(input_path.rglob("*_result.json.gz"))
        )
    scenarios: Dict[Tuple[str, str, str], ScenarioRecords] = {}
    strict_errors: List[str] = []
    for path in candidates:
        try:
            payload = _read_json(path)
        except Exception:
            continue
        detailed = payload.get("detailed_results")
        if not isinstance(detailed, dict):
            continue
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        model = _model_identity(payload, path)
        run_tag = _run_tag(payload, path)
        for name, detail in detailed.items():
            if not isinstance(detail, dict) or not isinstance(detail.get("requests"), list):
                continue
            records = [
                dict(item)
                for item in detail["requests"]
                if as_bool(item.get("success")) and str(item.get("adapter_id") or "").strip()
            ]
            if not records and not include_empty:
                continue
            dispatch_tier_records = sum(
                1
                for item in records
                if str(item.get("readiness_tier_before_dispatch") or "").strip()
            )
            dispatch_errors = _strict_dispatch_errors(
                source=path, scenario=name, records=records
            )
            if require_dispatch_tier:
                strict_errors.extend(dispatch_errors)

            schema_errors: List[str] = []
            total = _diagnostic_int(detail, "total", schema_errors)
            completed = _diagnostic_int(detail, "completed", schema_errors)
            failed = _diagnostic_int(detail, "failed", schema_errors)

            raw_phases = detail.get("multi_cycle_phase_results")
            if isinstance(raw_phases, list):
                phase_results = tuple(
                    dict(item) for item in raw_phases if isinstance(item, dict)
                )
                if len(phase_results) != len(raw_phases):
                    schema_errors.append(
                        "detailed_results.multi_cycle_phase_results contains non-object entries"
                    )
                phase_count: int | None = len(raw_phases)
            else:
                phase_results = ()
                phase_count = None
                schema_errors.append(
                    "missing detailed_results.multi_cycle_phase_results list"
                )

            raw_scale_events = detail.get("scale_up_events")
            detail_scale_count: int | None
            if isinstance(raw_scale_events, list):
                detail_scale_count = len(raw_scale_events)
            else:
                detail_scale_count = None

            coordination_by_scenario = metadata.get("scenario_coordination")
            if not isinstance(coordination_by_scenario, dict):
                coordination_by_scenario = {}
            coordination = coordination_by_scenario.get(name)
            if not isinstance(coordination, dict):
                coordination = {}
            activation = coordination.get("feature_activation")
            if not isinstance(activation, dict):
                activation = {}
            activation_scale_count_raw = activation.get("scale_up_event_count")
            activation_scale_count = (
                int(activation_scale_count_raw)
                if isinstance(activation_scale_count_raw, (int, float))
                and not isinstance(activation_scale_count_raw, bool)
                and float(activation_scale_count_raw).is_integer()
                and float(activation_scale_count_raw) >= 0.0
                else None
            )
            if detail_scale_count is None and activation_scale_count is None:
                scale_up_event_count = None
                schema_errors.append(
                    "missing detailed_results.scale_up_events and "
                    "metadata.scenario_coordination.feature_activation.scale_up_event_count"
                )
            else:
                scale_up_event_count = (
                    detail_scale_count
                    if detail_scale_count is not None
                    else activation_scale_count
                )
            if (
                detail_scale_count is not None
                and activation_scale_count is not None
                and detail_scale_count != activation_scale_count
            ):
                schema_errors.append(
                    "scale-up count mismatch between detailed_results.scale_up_events "
                    "and feature_activation.scale_up_event_count"
                )

            cold_cache_raw = coordination.get("cold_cache_reset_before_run")
            cold_cache_reset = (
                as_bool(cold_cache_raw) if cold_cache_raw is not None else None
            )
            key = (model, name, run_tag)
            if key in scenarios:
                raise SystemExit(
                    "duplicate readiness result identity "
                    f"(model={model!r}, scenario={name!r}, run_tag={run_tag!r}): "
                    f"{scenarios[key].source} and {path}"
                )
            scenarios[key] = ScenarioRecords(
                name=name,
                model=model,
                run_tag=run_tag,
                source=path,
                records=records,
                dispatch_tier_records=dispatch_tier_records,
                total=total,
                completed=completed,
                failed=failed,
                phase_count=phase_count,
                scale_up_event_count=scale_up_event_count,
                tier_invariant_conflicts=_tier_invariant_conflict_count(records),
                phase_results=phase_results,
                cold_cache_reset_before_run=cold_cache_reset,
                configuration_fingerprint=_diagnostic_configuration_fingerprint(
                    metadata=metadata, scenario=name, model=model
                ),
                diagnostic_schema_errors=tuple(schema_errors),
                dispatch_validation_errors=tuple(dispatch_errors),
            )
    if strict_errors:
        preview = "\n".join(f"  - {item}" for item in strict_errors[:25])
        suffix = "" if len(strict_errors) <= 25 else f"\n  ... {len(strict_errors) - 25} more"
        raise SystemExit(
            "strict dispatch-readiness validation failed for "
            f"{len(strict_errors)} field(s):\n{preview}{suffix}"
        )
    scenario_rank = {name: index for index, name in enumerate(SCENARIO_ORDER)}
    ordered = sorted(
        scenarios.values(),
        key=lambda item: (
            item.model,
            scenario_rank.get(item.name, len(SCENARIO_ORDER)),
            item.name,
            item.run_tag,
        ),
    )
    if not ordered:
        raise SystemExit(f"no FaaSLoRA request records found under {input_path}")
    return ordered


def aggregate_scenarios(scenarios: Sequence[ScenarioRecords]) -> List[ScenarioRecords]:
    """Pool records across run tags without losing the per-run source collection."""
    grouped: DefaultDict[Tuple[str, str], List[ScenarioRecords]] = defaultdict(list)
    for scenario in scenarios:
        grouped[(scenario.model, scenario.name)].append(scenario)
    aggregated: List[ScenarioRecords] = []
    for (model, name), runs in sorted(grouped.items()):
        records = [record for run in runs for record in run.records]
        aggregated.append(
            ScenarioRecords(
                name=name,
                model=model,
                run_tag=f"aggregate:{len(runs)}runs",
                source=runs[0].source,
                records=records,
                dispatch_tier_records=sum(run.dispatch_tier_records for run in runs),
                total=(
                    sum(int(run.total) for run in runs)
                    if all(run.total is not None for run in runs)
                    else None
                ),
                completed=(
                    sum(int(run.completed) for run in runs)
                    if all(run.completed is not None for run in runs)
                    else None
                ),
                failed=(
                    sum(int(run.failed) for run in runs)
                    if all(run.failed is not None for run in runs)
                    else None
                ),
                phase_count=(
                    sum(int(run.phase_count) for run in runs)
                    if all(run.phase_count is not None for run in runs)
                    else None
                ),
                scale_up_event_count=(
                    sum(int(run.scale_up_event_count) for run in runs)
                    if all(run.scale_up_event_count is not None for run in runs)
                    else None
                ),
                tier_invariant_conflicts=sum(
                    int(run.tier_invariant_conflicts) for run in runs
                ),
                phase_results=tuple(
                    phase for run in runs for phase in run.phase_results
                ),
                cold_cache_reset_before_run=(
                    True
                    if all(run.cold_cache_reset_before_run is True for run in runs)
                    else False
                ),
                configuration_fingerprint=(
                    runs[0].configuration_fingerprint
                    if len({run.configuration_fingerprint for run in runs}) == 1
                    else "mixed"
                ),
                diagnostic_schema_errors=tuple(
                    error for run in runs for error in run.diagnostic_schema_errors
                ),
                dispatch_validation_errors=tuple(
                    error for run in runs for error in run.dispatch_validation_errors
                ),
            )
        )
    scenario_rank = {name: index for index, name in enumerate(SCENARIO_ORDER)}
    model_rank = {"llama2_7b": 0, "llama32_3b": 1}
    aggregated.sort(
        key=lambda item: (
            model_rank.get(item.model, len(model_rank)),
            item.model,
            scenario_rank.get(item.name, len(scenario_rank)),
            item.name,
        )
    )
    return aggregated


def validate_diagnostic_evidence(
    scenarios: Sequence[ScenarioRecords],
    *,
    aggregate_runs: bool,
    expected_total: int = DIAGNOSTIC_EXPECTED_TOTAL,
    expected_phases: int = DIAGNOSTIC_EXPECTED_PHASES,
    min_scale_up_events: int = DIAGNOSTIC_MIN_SCALE_UP_EVENTS,
    min_first_service: int = DIAGNOSTIC_MIN_FIRST_SERVICE,
    max_cold_runs: int = DIAGNOSTIC_MAX_COLD_RUNS,
) -> Dict[str, Any]:
    """Apply the pre-registered V2 multi-cycle readiness evidence gates.

    Request-level observations may be pooled for the first-service sample-size
    gate only across at most two independent cold runs with identical frozen
    configuration.  All run-integrity gates remain per-run.
    """
    grouped: DefaultDict[Tuple[str, str], List[ScenarioRecords]] = defaultdict(list)
    for scenario in scenarios:
        grouped[(scenario.model, scenario.name)].append(scenario)

    integrity_errors: List[str] = []
    sample_shortfalls: List[str] = []
    group_reports: List[Dict[str, Any]] = []

    for (model, scenario_name), runs in sorted(grouped.items()):
        runs = sorted(runs, key=lambda item: item.run_tag)
        group_prefix = f"model={model!r}, variant={scenario_name!r}"
        group_errors: List[str] = []
        run_reports: List[Dict[str, Any]] = []

        if len(runs) > max_cold_runs:
            group_errors.append(
                f"{group_prefix}: {len(runs)} cold runs supplied; maximum is {max_cold_runs}"
            )
        if len(runs) > 1 and not aggregate_runs:
            group_errors.append(
                f"{group_prefix}: multiple cold runs require --aggregate-runs"
            )
        fingerprints = {run.configuration_fingerprint for run in runs}
        if len(runs) > 1 and ("" in fingerprints or len(fingerprints) != 1):
            group_errors.append(
                f"{group_prefix}: cold-run configurations are not identical"
            )

        for run in runs:
            prefix = f"{group_prefix}, run_tag={run.run_tag!r}"
            run_errors: List[str] = [
                f"{prefix}: {error}" for error in run.diagnostic_schema_errors
            ]
            if run.total != expected_total:
                run_errors.append(
                    f"{prefix}: total={run.total!r}, expected {expected_total}"
                )
            if run.completed != expected_total:
                run_errors.append(
                    f"{prefix}: completed={run.completed!r}, expected {expected_total}"
                )
            if run.failed != 0:
                run_errors.append(f"{prefix}: failed={run.failed!r}, expected 0")
            if len(run.records) != expected_total:
                run_errors.append(
                    f"{prefix}: successful LoRA request records={len(run.records)}, "
                    f"expected {expected_total}"
                )
            if run.phase_count != expected_phases:
                run_errors.append(
                    f"{prefix}: phase_count={run.phase_count!r}, expected {expected_phases}"
                )
            if run.scale_up_event_count is None or run.scale_up_event_count < min_scale_up_events:
                run_errors.append(
                    f"{prefix}: scale_up_events={run.scale_up_event_count!r}, "
                    f"expected >= {min_scale_up_events}"
                )
            if run.dispatch_tier_records != expected_total:
                run_errors.append(
                    f"{prefix}: dispatch tier coverage={run.dispatch_tier_records}/"
                    f"{expected_total}, expected 100%"
                )
            if run.tier_invariant_conflicts != 0:
                run_errors.append(
                    f"{prefix}: tier invariant conflicts={run.tier_invariant_conflicts}, expected 0"
                )
            if run.dispatch_validation_errors:
                preview = "; ".join(run.dispatch_validation_errors[:3])
                suffix = (
                    ""
                    if len(run.dispatch_validation_errors) <= 3
                    else f"; ... {len(run.dispatch_validation_errors) - 3} more"
                )
                run_errors.append(
                    f"{prefix}: strict dispatch validation has "
                    f"{len(run.dispatch_validation_errors)} error(s): {preview}{suffix}"
                )
            if run.cold_cache_reset_before_run is not True:
                run_errors.append(
                    f"{prefix}: cold_cache_reset_before_run must be true"
                )

            if len(run.phase_results) == expected_phases:
                phase_ids: List[int] = []
                phase_total_sum = 0
                for index, phase in enumerate(run.phase_results):
                    phase_id = phase.get("phase")
                    phase_total = phase.get("total")
                    phase_completed = phase.get("completed")
                    if (
                        isinstance(phase_id, bool)
                        or not isinstance(phase_id, (int, float))
                        or not float(phase_id).is_integer()
                    ):
                        run_errors.append(
                            f"{prefix}: phase[{index}].phase={phase_id!r} is invalid"
                        )
                    else:
                        phase_ids.append(int(phase_id))
                    if (
                        isinstance(phase_total, bool)
                        or not isinstance(phase_total, (int, float))
                        or not float(phase_total).is_integer()
                        or int(phase_total) <= 0
                    ):
                        run_errors.append(
                            f"{prefix}: phase[{index}].total={phase_total!r} is invalid"
                        )
                    else:
                        phase_total_int = int(phase_total)
                        phase_total_sum += phase_total_int
                        if phase_completed != phase_total_int:
                            run_errors.append(
                                f"{prefix}: phase[{index}] completed={phase_completed!r}, "
                                f"expected {phase_total_int}"
                            )
                        if expected_total % expected_phases == 0:
                            expected_phase_total = expected_total // expected_phases
                            if phase_total_int != expected_phase_total:
                                run_errors.append(
                                    f"{prefix}: phase[{index}].total={phase_total_int}, "
                                    f"expected {expected_phase_total}"
                                )
                if sorted(phase_ids) != list(range(expected_phases)):
                    run_errors.append(
                        f"{prefix}: phase ids={sorted(phase_ids)!r}, "
                        f"expected {list(range(expected_phases))!r}"
                    )
                if phase_total_sum != expected_total:
                    run_errors.append(
                        f"{prefix}: phase totals sum to {phase_total_sum}, "
                        f"expected {expected_total}"
                    )

            first_service_n = sum(
                1 for record in run.records if as_bool(record.get("scaleup_first_service"))
            )
            run_reports.append(
                {
                    "run_tag": run.run_tag,
                    "source": str(run.source),
                    "total": run.total,
                    "completed": run.completed,
                    "failed": run.failed,
                    "phase_count": run.phase_count,
                    "scale_up_event_count": run.scale_up_event_count,
                    "first_service_n": first_service_n,
                    "dispatch_tier_records": run.dispatch_tier_records,
                    "dispatch_tier_coverage_pct": (
                        100.0 * run.dispatch_tier_records / expected_total
                    ),
                    "tier_invariant_conflicts": run.tier_invariant_conflicts,
                    "cold_cache_reset_before_run": run.cold_cache_reset_before_run,
                    "configuration_fingerprint": run.configuration_fingerprint,
                    "passed_integrity": not run_errors,
                    "integrity_errors": run_errors,
                }
            )
            group_errors.extend(run_errors)

        first_service_total = sum(int(row["first_service_n"]) for row in run_reports)
        if (
            len(run_reports) > 1
            and int(run_reports[0]["first_service_n"]) >= min_first_service
        ):
            group_errors.append(
                f"{group_prefix}: a second cold run was supplied even though the "
                f"first run already had {run_reports[0]['first_service_n']} first-service "
                f"samples (threshold={min_first_service})"
            )
        group_shortfall = first_service_total < min_first_service
        if group_shortfall:
            sample_shortfalls.append(
                f"{group_prefix}: first-service samples={first_service_total}, "
                f"expected >= {min_first_service}"
            )
        integrity_errors.extend(group_errors)
        group_reports.append(
            {
                "model": model,
                "variant": scenario_name,
                "run_count": len(runs),
                "run_tags": [run.run_tag for run in runs],
                "configuration_fingerprints_identical": len(fingerprints) == 1,
                "first_service_n": first_service_total,
                "min_first_service": min_first_service,
                "first_service_gate_passed": not group_shortfall,
                "additional_cold_run_allowed": len(runs) < max_cold_runs,
                "passed_integrity": not group_errors,
                "runs": run_reports,
            }
        )

    return {
        "analysis": "readiness_diagnostic_gate",
        "gate_version": "eurosys27_v2_readiness_8cycle_v1",
        "thresholds": {
            "expected_total_per_run": expected_total,
            "expected_completed_per_run": expected_total,
            "expected_failed_per_run": 0,
            "expected_phases_per_run": expected_phases,
            "min_scale_up_events_per_run": min_scale_up_events,
            "min_first_service_per_model_variant_across_runs": min_first_service,
            "max_identical_cold_runs_per_model_variant": max_cold_runs,
            "dispatch_tier_coverage_pct": 100.0,
            "tier_invariant_conflicts": 0,
        },
        "aggregate_runs": aggregate_runs,
        "passed": not integrity_errors and not sample_shortfalls,
        "rerun_required": bool(sample_shortfalls),
        "integrity_errors": integrity_errors,
        "sample_shortfalls": sample_shortfalls,
        "groups": group_reports,
    }


def _t95_critical(run_count: int) -> float:
    # Two-sided 95% Student-t critical values. V2 formal evidence uses n=3;
    # values through n=10 keep the helper useful for larger rerun sets.
    values = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }
    if run_count < 2:
        return float("nan")
    return values.get(run_count, 1.96)


def build_across_run_summary(per_run_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate run-level metrics without treating requests as independent repeats."""
    grouped: DefaultDict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in per_run_rows:
        grouped[(str(row["model"]), str(row["scenario"]))].append(row)
    metrics = (
        "gpu_ready_pct",
        "host_nvme_pct",
        "remote_cold_pct",
        "mismatch_pct",
        "ttft_p95_gpu_ready_ms",
        "ttft_p95_mismatch_ms",
        "adapter_prep_p95_mismatch_ms",
        "dispatch_wait_p95_mismatch_ms",
        "runtime_ttft_p95_mismatch_ms",
        "scaleout_first_service_n",
        "scaleout_mismatch_n",
    )
    output: List[Dict[str, Any]] = []
    for (model, scenario), rows in sorted(grouped.items()):
        run_count = len(rows)
        out: Dict[str, Any] = {
            "model": model,
            "scenario": scenario,
            "run_count": run_count,
            "run_tags": ";".join(sorted(str(row["run_tag"]) for row in rows)),
            "total_successful_lora_requests": sum(int(row["n"]) for row in rows),
            "all_dispatch_tier_complete": all(as_bool(row["dispatch_tier_complete"]) for row in rows),
            "ci_unit": "independent run_tag",
            "ci_method": "two-sided 95% Student-t over run-level metrics",
        }
        critical = _t95_critical(run_count)
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows], dtype=float)
            finite = values[np.isfinite(values)]
            out[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
            if len(finite) >= 2 and len(finite) == run_count:
                std = float(np.std(finite, ddof=1))
                half_width = critical * std / math.sqrt(run_count)
            else:
                std = float("nan")
                half_width = float("nan")
            out[f"{metric}_std"] = std
            out[f"{metric}_ci95_half_width"] = half_width
        output.append(out)
    return output


def build_summary(scenarios: Sequence[ScenarioRecords]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        records = scenario.records
        n = len(records)
        tier_counts = {tier: 0 for tier in TIER_ORDER}
        for record in records:
            tier = readiness_tier(record)
            if tier in tier_counts:
                tier_counts[tier] += 1
        gpu_records = [r for r in records if readiness_tier(r) == "gpu"]
        mismatch_records = [r for r in records if readiness_tier(r) != "gpu"]
        scaleout_first = [r for r in records if as_bool(r.get("scaleup_first_service"))]
        scaleout_mismatch = [
            r
            for r in records
            if as_bool(r.get("scaleup_first_service"))
            and not as_bool(r.get("scaleup_planned_adapter_match"))
        ]
        rows.append(
            {
                "model": scenario.model,
                "scenario": scenario.name,
                "run_tag": scenario.run_tag,
                "label": SCENARIO_LABELS.get(scenario.name, scenario.name),
                "short_label": SHORT_LABELS.get(scenario.name, scenario.name),
                "source": str(scenario.source),
                "n": n,
                "used_dispatch_before_tier": scenario.used_dispatch_before_tier,
                "dispatch_tier_complete": scenario.all_records_have_dispatch_tier,
                "dispatch_tier_records": scenario.dispatch_tier_records,
                "dispatch_tier_coverage_pct": 100.0 * scenario.dispatch_tier_records / n,
                "unknown_tier_n": n - sum(tier_counts.values()),
                "gpu_ready_pct": 100.0 * tier_counts["gpu"] / n,
                "host_pct": 100.0 * tier_counts["host"] / n,
                "nvme_pct": 100.0 * tier_counts["nvme"] / n,
                "host_nvme_pct": 100.0 * (tier_counts["host"] + tier_counts["nvme"]) / n,
                "remote_cold_pct": 100.0 * tier_counts["remote"] / n,
                "mismatch_pct": 100.0 * len(mismatch_records) / n,
                "ttft_p50_gpu_ready_ms": percentile([ttft_ms(r) for r in gpu_records], 50),
                "ttft_p95_gpu_ready_ms": percentile([ttft_ms(r) for r in gpu_records], 95),
                "ttft_p50_mismatch_ms": percentile([ttft_ms(r) for r in mismatch_records], 50),
                "ttft_p95_mismatch_ms": percentile([ttft_ms(r) for r in mismatch_records], 95),
                "adapter_prep_p95_all_ms": percentile([adapter_prep_ms(r) for r in records], 95),
                "adapter_prep_p95_mismatch_ms": percentile(
                    [adapter_prep_ms(r) for r in mismatch_records], 95
                ),
                "dispatch_wait_p95_mismatch_ms": percentile(
                    [dispatch_wait_ms(r) for r in mismatch_records], 95
                ),
                "runtime_ttft_p95_mismatch_ms": percentile(
                    [runtime_ttft_ms(r) for r in mismatch_records], 95
                ),
                "scaleout_first_service_n": len(scaleout_first),
                "scaleout_mismatch_n": len(scaleout_mismatch),
            }
        )
    return rows


def build_by_tier(scenarios: Sequence[ScenarioRecords]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        records = scenario.records
        n = len(records)
        for tier in TIER_ORDER:
            group = [r for r in records if readiness_tier(r) == tier]
            if not group:
                continue
            rows.append(
                {
                    "model": scenario.model,
                    "scenario": scenario.name,
                    "run_tag": scenario.run_tag,
                    "label": SCENARIO_LABELS.get(scenario.name, scenario.name),
                    "tier": tier,
                    "tier_label": TIER_LABELS[tier],
                    "n": len(group),
                    "share_pct": 100.0 * len(group) / n,
                    "ttft_p50_ms": percentile([ttft_ms(r) for r in group], 50),
                    "ttft_p95_ms": percentile([ttft_ms(r) for r in group], 95),
                    "adapter_prep_p95_ms": percentile([adapter_prep_ms(r) for r in group], 95),
                    "dispatch_wait_p95_ms": percentile([dispatch_wait_ms(r) for r in group], 95),
                    "runtime_ttft_p95_ms": percentile([runtime_ttft_ms(r) for r in group], 95),
                    "ttft_mean_ms": mean([ttft_ms(r) for r in group]),
                    "adapter_prep_mean_ms": mean([adapter_prep_ms(r) for r in group]),
                    "dispatch_wait_mean_ms": mean([dispatch_wait_ms(r) for r in group]),
                    "runtime_ttft_mean_ms": mean([runtime_ttft_ms(r) for r in group]),
                }
            )
    return rows


def build_scaleout(scenarios: Sequence[ScenarioRecords]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        records = scenario.records
        first = [r for r in records if as_bool(r.get("scaleup_first_service"))]
        match = [r for r in first if as_bool(r.get("scaleup_planned_adapter_match"))]
        miss = [r for r in first if not as_bool(r.get("scaleup_planned_adapter_match"))]
        rows.append(
            {
                "model": scenario.model,
                "scenario": scenario.name,
                "run_tag": scenario.run_tag,
                "label": SCENARIO_LABELS.get(scenario.name, scenario.name),
                "first_service_n": len(first),
                "planned_match_n": len(match),
                "planned_miss_n": len(miss),
                "planned_match_rate_pct": 100.0 * len(match) / len(first) if first else float("nan"),
                "gpu_ready_rate_pct": 100.0
                * sum(1 for r in first if readiness_tier(r) == "gpu")
                / len(first)
                if first
                else float("nan"),
                "remote_cold_rate_pct": 100.0
                * sum(1 for r in first if readiness_tier(r) == "remote")
                / len(first)
                if first
                else float("nan"),
                "ttft_p50_match_ms": percentile([ttft_ms(r) for r in match], 50),
                "ttft_p95_match_ms": percentile([ttft_ms(r) for r in match], 95),
                "ttft_p50_miss_ms": percentile([ttft_ms(r) for r in miss], 50),
                "ttft_p95_miss_ms": percentile([ttft_ms(r) for r in miss], 95),
            }
        )
    return rows


def _transition_kind(dispatch_tier: str, observed_service_tier: str) -> str:
    if dispatch_tier not in TIER_ORDER or observed_service_tier not in TIER_ORDER:
        return "unknown"
    if dispatch_tier == observed_service_tier:
        return "stable"
    dispatch_rank = TIER_ORDER.index(dispatch_tier)
    service_rank = TIER_ORDER.index(observed_service_tier)
    return "promoted_before_service" if service_rank < dispatch_rank else "demoted_before_service"


def build_transition_outputs(
    scenarios: Sequence[ScenarioRecords],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build dispatch-snapshot to service-source transition summaries and audit rows.

    A tier change is only a *potentially stale signal*: it may also be a legitimate
    promotion or eviction between dispatch and adapter resolution.  The CSV keeps
    the observation separate from that interpretation.
    """
    summary_counts: DefaultDict[Tuple[str, str, str, str, str, str], int] = defaultdict(int)
    audit_rows: List[Dict[str, Any]] = []
    totals: DefaultDict[Tuple[str, str, str], int] = defaultdict(int)
    for scenario in scenarios:
        group_key = (scenario.model, scenario.name, scenario.run_tag)
        for record in scenario.records:
            dispatch = readiness_tier(record, allow_proxy=False) or "missing"
            observed = service_tier(record)
            transition = _transition_kind(dispatch, observed)
            summary_counts[group_key + (dispatch, observed, transition)] += 1
            totals[group_key] += 1
            if transition not in ("stable", "unknown"):
                audit_rows.append(
                    {
                        "model": scenario.model,
                        "scenario": scenario.name,
                        "run_tag": scenario.run_tag,
                        "source": str(scenario.source),
                        "request_id": record.get("request_id", ""),
                        "adapter_id": record.get("adapter_id", ""),
                        "instance_id": record.get("instance_id", ""),
                        "dispatch_tier": dispatch,
                        "service_tier": observed,
                        "transition": transition,
                        "potentially_stale_signal": True,
                        "selected_instance_age_s": as_float(
                            record.get("selected_instance_age_s"), float("nan")
                        ),
                        "overall_ttft_ms": ttft_ms(record),
                    }
                )
    summary_rows: List[Dict[str, Any]] = []
    for key, count in sorted(summary_counts.items()):
        model, scenario, run_tag, dispatch, observed, transition = key
        n = totals[(model, scenario, run_tag)]
        summary_rows.append(
            {
                "model": model,
                "scenario": scenario,
                "run_tag": run_tag,
                "dispatch_tier": dispatch,
                "service_tier": observed,
                "transition": transition,
                "n": count,
                "share_of_successful_lora_pct": 100.0 * count / n,
            }
        )
    return summary_rows, audit_rows


def write_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames:
            with path.open("w", encoding="utf-8", newline="") as handle:
                csv.DictWriter(
                    handle,
                    fieldnames=list(fieldnames),
                    lineterminator="\n",
                ).writeheader()
        else:
            path.write_text("", encoding="utf-8")
        return
    resolved_fieldnames = list(fieldnames or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=resolved_fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_num(value: Any, digits: int = 1) -> str:
    try:
        val = float(value)
    except Exception:
        return "--"
    if not math.isfinite(val):
        return "--"
    return f"{val:.{digits}f}"


def write_latex_table(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    include_remote = bool(
        rows and max(float(row.get("remote_cold_pct", 0.0) or 0.0) for row in rows) > 0.0
    )
    colspec = "lrrrrrrr" if include_remote else "lrrrrrr"
    header = (
        r"System & \shortstack{GPU\\(\%)} & \shortstack{HOST/NVMe\\(\%)} & "
    )
    if include_remote:
        header += r"\shortstack{Remote\\(\%)} & "
    header += (
        r"\shortstack{Mismatch\\(\%)} & \shortstack{TTFT p95\\GPU (ms)} & "
        r"\shortstack{TTFT p95\\Non-GPU (ms)} & \shortstack{Prep p95\\Non-GPU (ms)} \\"
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Service-readiness analysis on the representative Llama-2-7B multi-LoRA workload.}",
        r"\label{tab:service_readiness}",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\hline",
        header,
        r"\hline",
    ]
    for row in rows:
        cells = [
            str(row["label"]),
            fmt_num(row["gpu_ready_pct"], 2),
            fmt_num(row["host_nvme_pct"], 2),
        ]
        if include_remote:
            cells.append(fmt_num(row["remote_cold_pct"], 2))
        cells.extend(
            [
                fmt_num(row["mismatch_pct"], 2),
                fmt_num(row["ttft_p95_gpu_ready_ms"], 1),
                fmt_num(row["ttft_p95_mismatch_ms"], 1),
                fmt_num(row["adapter_prep_p95_mismatch_ms"], 1),
            ]
        )
        lines.append(
            " & ".join(cells)
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_summary(out_path: Path, summary_rows: Sequence[Dict[str, Any]]) -> None:
    labels = [str(row["short_label"]) for row in summary_rows]
    y = np.arange(len(summary_rows))
    fig = plt.figure(figsize=(3.45, 4.15))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.12],
        width_ratios=[1.04, 1.0],
        hspace=0.70,
        wspace=0.34,
    )

    ax_gpu = fig.add_subplot(grid[0, 0])
    ax_mix = fig.add_subplot(grid[0, 1], sharey=ax_gpu)
    ax_tail = fig.add_subplot(grid[1, :])

    gpu_vals = [float(row["gpu_ready_pct"]) for row in summary_rows]
    ax_gpu.barh(y, gpu_vals, height=0.55, color=TIER_COLORS["gpu"], edgecolor="white", linewidth=0.55)
    ax_gpu.set_yticks(y)
    ax_gpu.set_yticklabels(labels)
    ax_gpu.invert_yaxis()
    ax_gpu.set_xlim(0, 100)
    ax_gpu.set_xlabel("GPU-ready (%)", labelpad=2)
    ax_gpu.grid(axis="x", alpha=0.25)
    for yi, value in zip(y, gpu_vals):
        ax_gpu.text(value - 1.2, yi, f"{value:.2f}", ha="right", va="center", fontsize=8.6, color="white")

    bottoms = np.zeros(len(summary_rows))
    tier_values: Dict[str, List[float]] = {}
    for tier in TIER_ORDER:
        vals = []
        for row in summary_rows:
            if tier == "gpu":
                vals.append(float(row["gpu_ready_pct"]))
            elif tier == "host":
                vals.append(float(row["host_pct"]))
            elif tier == "nvme":
                vals.append(float(row["nvme_pct"]))
            else:
                vals.append(float(row["remote_cold_pct"]))
        tier_values[tier] = vals
    visible_tiers = [
        tier
        for tier in ("host", "nvme", "remote")
        if tier != "remote" or max(tier_values[tier]) > 0.0
    ]
    for tier in visible_tiers:
        vals = tier_values[tier]
        ax_mix.barh(
            y,
            vals,
            left=bottoms,
            height=0.55,
            color=TIER_COLORS[tier],
            edgecolor="white",
            linewidth=0.55,
            label=TIER_LABELS[tier],
        )
        for yi, left, value in zip(y, bottoms, vals):
            if value <= 0:
                continue
            x_mid = left + value / 2.0
            if value >= 0.55:
                ax_mix.text(
                    x_mid,
                    yi,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color="#1F2A35",
                )
            else:
                ax_mix.text(
                    left + value + 0.08,
                    yi,
                    f"{value:.2f}",
                    ha="left",
                    va="center",
                    fontsize=7.8,
                    color="#1F2A35",
                )
        bottoms += np.asarray(vals)
    xmax_mix = max(5.6, float(max(bottoms)) * 1.28)
    ax_mix.set_xlim(0, xmax_mix)
    ax_mix.set_xlabel("Non-GPU tier (%)", labelpad=2)
    ax_mix.grid(axis="x", alpha=0.25)
    ax_mix.tick_params(axis="y", left=False, labelleft=False)
    ax_mix.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    fig.text(0.5, 0.545, "(a) Selected-replica adapter-readiness distribution", ha="center", va="top", fontsize=10.0)

    ax = ax_tail
    gpu_s = [float(row["ttft_p95_gpu_ready_ms"]) / 1000.0 for row in summary_rows]
    mis_s = [float(row["ttft_p95_mismatch_ms"]) / 1000.0 for row in summary_rows]
    for yi, gv, mv in zip(y, gpu_s, mis_s):
        ax.plot([gv, mv], [yi, yi], color="#A0A0A0", lw=1.0, zorder=1)
    ax.scatter(gpu_s, y, marker="o", s=34, color=TIER_COLORS["gpu"], label="GPU-ready", zorder=2)
    ax.scatter(mis_s, y, marker="D", s=34, color="#C74343", label="Non-GPU", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("TTFT p95 (s)")
    ax.grid(axis="x", alpha=0.25)
    xmax = max(mis_s + gpu_s) * 1.65
    ax.set_xlim(0, xmax)
    ax.set_ylim(len(summary_rows) - 0.45, -0.45)
    ax.legend(
        loc="center right",
        bbox_to_anchor=(0.98, 0.50),
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#D0D0D0",
        fontsize=8.2,
        handlelength=1.25,
        handletextpad=0.45,
        borderpad=0.35,
    )
    for yi, gv, mv in zip(y, gpu_s, mis_s):
        ax.text(gv + xmax * 0.018, yi - 0.14, f"{gv:.2f}", fontsize=8.8, color="#2F6F3E")
        ax.text(mv + xmax * 0.018, yi - 0.14, f"{mv:.2f}", fontsize=8.8, color="#8A2E2E")
    ax.text(0.5, -0.36, "(b) Tail penalty when not GPU-ready", transform=ax.transAxes, ha="center", va="top", fontsize=10.0)

    fig.subplots_adjust(left=0.22, right=0.98, top=0.92, bottom=0.16)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_mechanism_matrix(out_path: Path, summary_rows: Sequence[Dict[str, Any]]) -> None:
    labels = [str(row["short_label"]) for row in summary_rows]
    metric_rows = [
        ("Non-GPU dispatch (%)", [float(row["mismatch_pct"]) for row in summary_rows], "{:.2f}"),
        ("Prep p95, mismatch (ms)", [float(row["adapter_prep_p95_mismatch_ms"]) for row in summary_rows], "{:.1f}"),
        ("Mismatch TTFT p95 (s)", [float(row["ttft_p95_mismatch_ms"]) / 1000.0 for row in summary_rows], "{:.2f}"),
    ]
    remote_values = [float(row["remote_cold_pct"]) for row in summary_rows]
    if max(remote_values) > 0.0:
        metric_rows.insert(1, ("Remote-cold (%)", remote_values, "{:.2f}"))
    raw = np.asarray([vals for _, vals, _ in metric_rows], dtype=float)
    norm = np.zeros_like(raw)
    for i in range(raw.shape[0]):
        row = raw[i]
        lo, hi = float(np.nanmin(row)), float(np.nanmax(row))
        norm[i] = 0.45 if abs(hi - lo) < 1e-12 else (row - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(3.45, 3.1))
    ax.imshow(norm, cmap=METRIC_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(metric_rows)))
    ax.set_yticklabels([item[0] for item in metric_rows])
    for i, (_, vals, fmt) in enumerate(metric_rows):
        for j, value in enumerate(vals):
            text_color = "white" if norm[i, j] > 0.62 else "#1F2A35"
            ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=9.4, color=text_color)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(metric_rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)
    ax.text(0.5, -0.20, "Lower is better for all cells", transform=ax.transAxes, ha="center", va="top", fontsize=9.4)
    fig.subplots_adjust(left=0.48, right=0.99, top=0.96, bottom=0.16)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_full_cdf(out_path: Path, scenarios: Sequence[ScenarioRecords]) -> bool:
    full = next(
        (
            scenario
            for scenario in scenarios
            if scenario.name in {"v2_full", "faaslora_full"}
            and scenario.model == "llama2_7b"
        ),
        None,
    )
    if full is None:
        full = next(
            (
                scenario
                for scenario in scenarios
                if scenario.name in {"v2_full", "faaslora_full"}
            ),
            None,
        )
    if full is None:
        return False
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    plotted = False
    for tier in TIER_ORDER:
        group = [ttft_ms(r) / 1000.0 for r in full.records if readiness_tier(r) == tier]
        if len(group) < 5:
            continue
        values = np.sort(np.asarray(group, dtype=float))
        cdf = np.arange(1, len(values) + 1) / len(values)
        p95 = percentile(values, 95)
        ax.plot(values, cdf, label=f"{TIER_LABELS[tier]} (n={len(values)}, p95={p95:.2f}s)", color=TIER_COLORS[tier], lw=1.35)
        ax.axvline(p95, color=TIER_COLORS[tier], lw=0.8, ls="--", alpha=0.55)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel("TTFT (s)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=True, framealpha=0.94, fontsize=7.6)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.98, bottom=0.22)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def write_manifest(
    out_path: Path,
    scenarios: Sequence[ScenarioRecords],
    generated: Sequence[str],
    skipped: Dict[str, str],
    *,
    require_dispatch_tier: bool,
    aggregate_runs: bool,
    diagnostic_gate: Dict[str, Any] | None = None,
    formal_campaign: Dict[str, Any] | None = None,
) -> None:
    all_dispatch = all(s.all_records_have_dispatch_tier for s in scenarios)
    no_dispatch = all(not s.used_dispatch_before_tier for s in scenarios)
    if all_dispatch:
        tier_field = "readiness_tier_before_dispatch"
        field_caveat = None
    elif no_dispatch:
        tier_field = "cache_tier proxy"
        field_caveat = (
            "No input record contains readiness_tier_before_dispatch. cache_tier is "
            "used only as a selected-replica service-time readiness proxy."
        )
    else:
        tier_field = "readiness_tier_before_dispatch with per-record cache_tier fallback"
        field_caveat = (
            "Dispatch-time readiness coverage is partial. Rows without that field use "
            "cache_tier as a service-time proxy; use --require-dispatch-tier for formal evidence."
        )
    payload = {
        "analysis": "service_readiness",
        "input_sources": [
            {
                "model": scenario.model,
                "scenario": scenario.name,
                "run_tag": scenario.run_tag,
                "source": str(scenario.source),
                "requests_used": len(scenario.records),
                "dispatch_tier_records": scenario.dispatch_tier_records,
            }
            for scenario in scenarios
        ],
        "filter": "success == True and adapter_id non-empty",
        "tier_field": tier_field,
        "dispatch_before_tier_available": all_dispatch,
        "field_caveat": field_caveat,
        "require_dispatch_tier": require_dispatch_tier,
        "aggregate_runs": aggregate_runs,
        "diagnostic_gate": (
            {
                "enabled": True,
                "report": "readiness_diagnostic_gate.json",
                "gate_version": diagnostic_gate.get("gate_version"),
                "passed": diagnostic_gate.get("passed"),
                "rerun_required": diagnostic_gate.get("rerun_required"),
            }
            if diagnostic_gate is not None
            else {"enabled": False}
        ),
        "formal_campaign_gate": (
            {
                "enabled": True,
                "report": "readiness_formal_v2_gate.json",
                "gate_version": formal_campaign.get("gate_version"),
                "passed": formal_campaign.get("passed"),
                "source_commit": formal_campaign.get("source_commit"),
            }
            if formal_campaign is not None
            else {"enabled": False}
        ),
        "run_level_aggregation": {
            "csv": "service_readiness_across_runs.csv",
            "unit": "independent (model, scenario, run_tag) result",
            "ci": "two-sided 95% Student-t over run-level metrics",
        },
        "transition_audit": {
            "summary": "readiness_tier_transitions.csv",
            "changed_records": "readiness_potentially_stale_signals.csv",
            "interpretation": (
                "A dispatch/service tier change is a potentially stale signal, not proof of "
                "staleness: a valid promotion or eviction may occur after dispatch."
            ),
        },
        "generated": list(generated),
        "skipped": skipped,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _prepare_output_dir(out_dir: Path) -> None:
    """Create a fresh analysis directory; never overwrite an earlier campaign."""
    if out_dir.exists() and not out_dir.is_dir():
        raise SystemExit(f"output path exists and is not a directory: {out_dir}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"refusing to overwrite non-empty output directory: {out_dir}; "
            "choose a fresh V2 campaign directory"
        )
    out_dir.mkdir(parents=True, exist_ok=True)


def _publish_figures(
    out_dir: Path,
    publish_dir: Path,
    generated: Sequence[str],
    *,
    require_fresh: bool = False,
) -> None:
    if publish_dir.exists() and not publish_dir.is_dir():
        raise SystemExit(f"publish path exists and is not a directory: {publish_dir}")
    if require_fresh and publish_dir.exists() and any(publish_dir.iterdir()):
        raise SystemExit(
            f"refusing to publish formal readiness figures into non-empty directory: "
            f"{publish_dir}"
        )
    publish_dir.mkdir(parents=True, exist_ok=True)
    collisions = [name for name in generated if (publish_dir / name).exists()]
    if collisions:
        raise SystemExit(
            "refusing to overwrite published figure(s): "
            + ", ".join(str(publish_dir / name) for name in collisions)
        )
    for fig_name in generated:
        src = out_dir / fig_name
        if src.exists():
            shutil.copy2(src, publish_dir / fig_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        action="append",
        type=Path,
        help=(
            "FaaSLoRA result/campaign path. Repeat for multiple formal cold-run "
            "campaigns; loose result JSON is forbidden with --formal-v2."
        ),
    )
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--require-dispatch-tier",
        action="store_true",
        help=(
            "Require a valid readiness_tier_before_dispatch and consistent readiness "
            "booleans on every successful LoRA request."
        ),
    )
    parser.add_argument(
        "--aggregate-runs",
        action="store_true",
        help=(
            "Pool request records by (model, scenario) after preserving per-run summaries; "
            "required when a diagnostic first-service sample uses a second cold run."
        ),
    )
    parser.add_argument(
        "--diagnostic-gates",
        action="store_true",
        help=(
            "Enforce the fixed V2 8-cycle diagnostic protocol: 4000/4000/0, eight "
            "complete 500-request phases, >=8 scale-ups, 100% dispatch-tier coverage, "
            "zero tier conflicts, and >=20 first-service samples across at most two "
            "identically configured cold runs."
        ),
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        help=(
            "Explicit figure publication directory. Optional for exploratory analysis; "
            "required and required to be fresh/empty with --formal-v2."
        ),
    )
    parser.add_argument(
        "--formal-v2",
        action="store_true",
        help=(
            "Fail-closed EuroSys'27 V2 dispatch-readiness mode. Inputs must be "
            "completed formal campaign manifests and exactly cover 7B "
            "ElasticOnly/+Hierarchy/Full plus 3B Full at held-out seed 43. This "
            "mode implies dispatch, diagnostic, and multi-run aggregation gates."
        ),
    )
    args = parser.parse_args()

    configure_matplotlib()
    out_dir = args.output.resolve()
    input_paths = [path.resolve() for path in args.input]
    if not args.formal_v2 and len(input_paths) != 1:
        parser.error("repeat --input only with --formal-v2; use one common root otherwise")
    publish_dir = args.publish_dir.resolve() if args.publish_dir is not None else None
    if args.formal_v2:
        if publish_dir is None:
            parser.error("--formal-v2 requires an explicit fresh --publish-dir")
        if publish_dir == out_dir:
            parser.error("--output and --publish-dir must be different directories")
        if publish_dir.exists() and not publish_dir.is_dir():
            raise SystemExit(f"publish path exists and is not a directory: {publish_dir}")
        if publish_dir.exists() and any(publish_dir.iterdir()):
            raise SystemExit(
                "refusing to use non-empty formal readiness publish directory: "
                f"{publish_dir}"
            )

    formal_campaign: Dict[str, Any] | None = None
    if args.formal_v2:
        input_scenarios, _provenance, formal_campaign = (
            load_formal_v2_readiness_campaign(input_paths)
        )
        effective_require_dispatch = True
        effective_diagnostic = True
        effective_aggregate = True
        _prepare_output_dir(out_dir)
        (out_dir / "readiness_formal_v2_gate.json").write_text(
            json.dumps(formal_campaign, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    else:
        effective_require_dispatch = args.require_dispatch_tier
        effective_diagnostic = args.diagnostic_gates
        effective_aggregate = args.aggregate_runs
        _prepare_output_dir(out_dir)
        input_scenarios = load_scenarios(
            input_paths[0],
            require_dispatch_tier=(
                effective_require_dispatch and not effective_diagnostic
            ),
            include_empty=effective_diagnostic,
        )
    table_dir = out_dir / "tables"

    diagnostic_gate: Dict[str, Any] | None = None
    if effective_diagnostic:
        diagnostic_gate = validate_diagnostic_evidence(
            input_scenarios, aggregate_runs=effective_aggregate
        )
        diagnostic_path = out_dir / "readiness_diagnostic_gate.json"
        diagnostic_path.write_text(
            json.dumps(diagnostic_gate, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if not diagnostic_gate["passed"]:
            raise SystemExit(
                "readiness diagnostic gates failed "
                f"(rerun_required={str(diagnostic_gate['rerun_required']).lower()}); "
                f"see {diagnostic_path}"
            )
    per_run_summary_rows = build_summary(input_scenarios)
    across_run_rows = build_across_run_summary(per_run_summary_rows)
    scenarios = (
        aggregate_scenarios(input_scenarios)
        if effective_aggregate
        else input_scenarios
    )
    summary_rows = build_summary(scenarios)
    tier_rows = build_by_tier(scenarios)
    scaleout_rows = build_scaleout(scenarios)
    transition_rows, stale_rows = build_transition_outputs(scenarios)

    write_csv(out_dir / "service_readiness_per_run_summary.csv", per_run_summary_rows)
    write_csv(out_dir / "service_readiness_across_runs.csv", across_run_rows)
    write_csv(out_dir / "service_readiness_summary.csv", summary_rows)
    write_csv(out_dir / "service_readiness_by_tier.csv", tier_rows)
    write_csv(out_dir / "scaleout_first_service_summary.csv", scaleout_rows)
    write_csv(out_dir / "readiness_tier_transitions.csv", transition_rows)
    write_csv(
        out_dir / "readiness_potentially_stale_signals.csv",
        stale_rows,
        fieldnames=(
            "model",
            "scenario",
            "run_tag",
            "source",
            "request_id",
            "adapter_id",
            "instance_id",
            "dispatch_tier",
            "service_tier",
            "transition",
            "potentially_stale_signal",
            "selected_instance_age_s",
            "overall_ttft_ms",
        ),
    )
    write_latex_table(table_dir / "table_service_readiness.tex", summary_rows)

    generated: List[str] = []
    plot_summary(out_dir / "fig_service_readiness_summary.pdf", summary_rows)
    generated.append("fig_service_readiness_summary.pdf")
    plot_mechanism_matrix(out_dir / "fig_mechanism_gap_ablation.pdf", summary_rows)
    generated.append("fig_mechanism_gap_ablation.pdf")
    if plot_full_cdf(out_dir / "fig_ttft_breakdown_readiness.pdf", scenarios):
        generated.append("fig_ttft_breakdown_readiness.pdf")

    skipped: Dict[str, str] = {}
    min_first_service = min((int(row["first_service_n"]) for row in scaleout_rows), default=0)
    if min_first_service < 20:
        skipped["fig_scaleout_first_service.pdf"] = (
            "Skipped as a paper figure because scaleup_first_service has fewer than 20 samples "
            f"per scenario (minimum observed n={min_first_service})."
        )
    skipped["fig_control_plane_overhead.pdf"] = (
        "Skipped because archived results do not contain routing_decision_us, "
        "gpu_admission_decision_us, or control_plane_total_us."
    )
    (out_dir / "service_readiness_warnings.txt").write_text(
        "\n".join(f"{name}: {reason}" for name, reason in skipped.items()) + "\n",
        encoding="utf-8",
    )
    write_manifest(
        out_dir / "service_readiness_manifest.json",
        input_scenarios,
        generated,
        skipped,
        require_dispatch_tier=effective_require_dispatch or effective_diagnostic,
        aggregate_runs=effective_aggregate,
        diagnostic_gate=diagnostic_gate,
        formal_campaign=formal_campaign,
    )

    if publish_dir is not None:
        _publish_figures(
            out_dir,
            publish_dir,
            generated,
            require_fresh=args.formal_v2,
        )

    print("Service-readiness summary")
    for row in summary_rows:
        print(
            f"  {row['label']}: gpu={row['gpu_ready_pct']:.2f}% "
            f"mismatch={row['mismatch_pct']:.2f}% "
            f"ttft_p95_gpu={row['ttft_p95_gpu_ready_ms']:.1f}ms "
            f"ttft_p95_mismatch={row['ttft_p95_mismatch_ms']:.1f}ms"
        )
    print(f"wrote outputs -> {out_dir}")


if __name__ == "__main__":
    main()
