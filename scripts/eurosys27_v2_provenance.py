#!/usr/bin/env python3
"""Fail-closed provenance checks for formal EuroSys'27 V2 analyses.

Exploratory plots intentionally remain permissive.  A formal plot, however,
must be rooted in completed campaign manifests rather than in a hand-picked
list of loose result files.  This module understands the two V2 campaign
manifest schemas currently emitted by the FaaSLoRA ablation runner and the
cross-system fair-round runner.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, NoReturn, Sequence


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ABLATION_EVIDENCE_SCHEMA = "eurosys27_v2_faaslora_ablation_validation_evidence_v1"


@dataclass(frozen=True)
class FormalAnalysisIdentity:
    """One raw result participating in a formal configuration-drift check."""

    source: Path
    model: str
    variant: str
    seed: int


@dataclass(frozen=True)
class _ManifestRecord:
    source: Path
    manifest_path: Path
    sha256: str
    size_bytes: int
    system_resolved_config_sha256: str


@dataclass(frozen=True)
class FormalProvenanceIndex:
    """Validated formal manifests and their indexed raw JSON records."""

    manifest_paths: tuple[Path, ...]
    records_by_source: Mapping[Path, tuple[_ManifestRecord, ...]]


def _fail(message: str) -> NoReturn:
    raise SystemExit(f"formal provenance gate failed: {message}")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _fail(f"cannot parse JSON {path}: {exc}")
    if not isinstance(payload, dict):
        _fail(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
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


def _digest(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        _fail(f"{label} must be a 64-character lowercase SHA-256 digest")
    return digest


def _size(value: Any, label: str) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError):
        _fail(f"{label} must be an integer byte count")
    if size < 0:
        _fail(f"{label} must be non-negative")
    return size


def _integer(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        _fail(f"{label} must be an integer")


def _resolve_record_path(manifest_path: Path, raw: Any, label: str) -> Path:
    text = str(raw or "").strip()
    if not text:
        _fail(f"{label} is empty")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return candidate.resolve()


def _manifest_source_clean(payload: Mapping[str, Any]) -> bool:
    if "source_clean_for_formal" in payload:
        return payload.get("source_clean_for_formal") is True
    snapshot = payload.get("code_snapshot")
    return isinstance(snapshot, dict) and snapshot.get("source_clean_for_formal") is True


def _manifest_commit(payload: Mapping[str, Any]) -> str:
    snapshot = payload.get("code_snapshot")
    if isinstance(snapshot, dict):
        commit = str(snapshot.get("git_commit") or "").strip()
        if commit:
            return commit
    return str(payload.get("git_commit") or "").strip()


def _validated_file_identity(
    *,
    owner_path: Path,
    raw_path: Any,
    raw_size: Any,
    raw_sha256: Any,
    label: str,
) -> Path:
    candidate = _resolve_record_path(owner_path, raw_path, f"{label}.path")
    if not candidate.is_file():
        _fail(f"{label}: recorded file is missing: {candidate}")
    expected_size = _size(raw_size, f"{label}.bytes")
    actual_size = candidate.stat().st_size
    if expected_size != actual_size:
        _fail(
            f"{label}: byte-count mismatch; recorded={expected_size}, actual={actual_size}"
        )
    expected_sha = _digest(raw_sha256, f"{label}.sha256")
    actual_sha = _sha256(candidate)
    if expected_sha != actual_sha:
        _fail(f"{label}: SHA-256 mismatch; recorded={expected_sha}, actual={actual_sha}")
    return candidate


def _validate_config_snapshot(
    owner_path: Path,
    raw_snapshot: Any,
    *,
    label: str,
) -> tuple[Path, str, int]:
    if not isinstance(raw_snapshot, dict):
        _fail(f"{label} must be an object")
    config_path = _validated_file_identity(
        owner_path=owner_path,
        raw_path=raw_snapshot.get("path"),
        raw_size=raw_snapshot.get("bytes"),
        raw_sha256=raw_snapshot.get("sha256"),
        label=label,
    )
    return (
        config_path,
        _digest(raw_snapshot.get("sha256"), f"{label}.sha256"),
        _size(raw_snapshot.get("bytes"), f"{label}.bytes"),
    )


def _validate_seed41_validation_manifest(
    validation_path: Path,
    validation: Mapping[str, Any],
    *,
    heldout_path: Path,
    selected_hash: str,
    family: Mapping[str, Any],
    heldout_commit: str,
    heldout_config: tuple[Path, str, int],
) -> None:
    label = f"{heldout_path}:seed41 validation {validation_path}"
    if validation.get("status") != "complete":
        _fail(f"{label}: status must be 'complete'")
    if validation.get("formal_run") is not True:
        _fail(f"{label}: formal_run must be true")
    if str(validation.get("trace_role") or "").strip().lower() != "validation":
        _fail(f"{label}: trace_role must be 'validation'")
    if not _manifest_source_clean(validation):
        _fail(f"{label}: source_clean_for_formal must be true")
    if list(validation.get("scenarios") or []) != ["v2_full"]:
        _fail(f"{label}: scenarios must be exactly ['v2_full']")
    shared_trace = validation.get("shared_trace")
    if not isinstance(shared_trace, dict):
        _fail(f"{label}: shared_trace must be an object")
    try:
        validation_seed = int(shared_trace.get("sampling_seed", -1))
        validation_requests = int(shared_trace.get("requests", -1))
    except (TypeError, ValueError):
        _fail(f"{label}: invalid validation seed/request count")
    if validation_seed != 41 or validation_requests != 1000:
        _fail(f"{label}: validation must bind seed 41 and exactly 1,000 requests")
    if _manifest_commit(validation) != heldout_commit:
        _fail(f"{label}: source commit differs from held-out source commit")
    if validation.get("configuration_family") != family:
        _fail(f"{label}: configuration family differs from held-out family")
    validation_config = _validate_config_snapshot(
        validation_path,
        validation.get("config_snapshot"),
        label=f"{label}.config_snapshot",
    )
    if validation_config != heldout_config:
        _fail(f"{label}: config snapshot differs from held-out config snapshot")
    if validation.get("non_feature_frozen_config_consistent") is not True:
        _fail(f"{label}: non-feature frozen hash is not internally consistent")
    validation_hash = _digest(
        validation.get("non_feature_frozen_config_sha256"),
        f"{label}.non_feature_frozen_config_sha256",
    )
    if validation_hash != selected_hash:
        _fail(f"{label}: frozen non-feature hash differs from held-out selection")

    entries = validation.get("entries")
    if not isinstance(entries, list) or len(entries) != 1:
        _fail(f"{label}: validation must preserve exactly one v2_full result entry")
    entry = entries[0]
    if not isinstance(entry, dict) or str(entry.get("scenario") or "") != "v2_full":
        _fail(f"{label}: validation result entry must be v2_full")
    entry_hash = _digest(
        entry.get("non_feature_frozen_config_sha256"),
        f"{label}.entries[0].non_feature_frozen_config_sha256",
    )
    if entry_hash != selected_hash:
        _fail(f"{label}: validation result entry frozen hash mismatch")
    result_path = _validated_file_identity(
        owner_path=validation_path,
        raw_path=entry.get("result_json"),
        raw_size=entry.get("bytes"),
        raw_sha256=entry.get("sha256"),
        label=f"{label}.entries[0].result_json",
    )
    result = _read_json(result_path)
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        _fail(f"{label}: validation result metadata is missing")
    if metadata.get("formal_run") is not True:
        _fail(f"{label}: validation result metadata.formal_run must be true")
    if str(metadata.get("trace_role") or "").strip().lower() != "validation":
        _fail(f"{label}: validation result metadata.trace_role must be validation")
    result_hash = _digest(
        metadata.get("non_feature_frozen_config_sha256"),
        f"{label}:validation result metadata.non_feature_frozen_config_sha256",
    )
    if result_hash != selected_hash:
        _fail(f"{label}: validation result frozen hash mismatch")


def _validate_ablation_seed41_evidence(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Re-validate the immutable seed-41 freeze chain for held-out ablations."""

    summary = payload.get("seed41_validation_evidence")
    if not isinstance(summary, dict):
        _fail(f"{path}: formal ablation is missing seed41_validation_evidence")
    evidence_path = _validated_file_identity(
        owner_path=path,
        raw_path=summary.get("path"),
        raw_size=summary.get("bytes"),
        raw_sha256=summary.get("sha256"),
        label=f"{path}:seed41_validation_evidence",
    )
    evidence = _read_json(evidence_path)
    if evidence.get("schema_version") != _ABLATION_EVIDENCE_SCHEMA:
        _fail(f"{evidence_path}: unsupported seed41 evidence schema")

    selected_hash = _digest(
        evidence.get("selected_non_feature_frozen_config_sha256"),
        f"{evidence_path}:selected_non_feature_frozen_config_sha256",
    )
    if _digest(
        summary.get("selected_non_feature_frozen_config_sha256"),
        f"{path}:seed41_validation_evidence.selected_non_feature_frozen_config_sha256",
    ) != selected_hash:
        _fail(f"{path}: held-out/evidence selected non-feature hash mismatch")
    if payload.get("non_feature_frozen_config_consistent") is not True:
        _fail(f"{path}: held-out non-feature frozen hash is not internally consistent")
    if _digest(
        payload.get("non_feature_frozen_config_sha256"),
        f"{path}:non_feature_frozen_config_sha256",
    ) != selected_hash:
        _fail(f"{path}: held-out result hash differs from seed41 selection")
    entries = payload.get("entries") or []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            _fail(f"{path}:entries[{index}] must be an object")
        if _digest(
            entry.get("non_feature_frozen_config_sha256"),
            f"{path}:entries[{index}].non_feature_frozen_config_sha256",
        ) != selected_hash:
            _fail(f"{path}:entries[{index}] frozen non-feature hash mismatch")

    family = payload.get("configuration_family")
    if not isinstance(family, dict) or not family:
        _fail(f"{path}: configuration_family must be a non-empty object")
    if evidence.get("configuration_family") != family:
        _fail(f"{path}: seed41 evidence configuration family mismatch")
    family_id = _digest(
        evidence.get("configuration_family_id"),
        f"{evidence_path}:configuration_family_id",
    )
    if family_id != _canonical_sha256(family):
        _fail(f"{evidence_path}: configuration_family_id does not match family bytes")
    summary_family_id = summary.get("configuration_family_id")
    if summary_family_id is not None and _digest(
        summary_family_id,
        f"{path}:seed41_validation_evidence.configuration_family_id",
    ) != family_id:
        _fail(f"{path}: evidence-summary configuration family ID mismatch")

    heldout_trace = payload.get("shared_trace")
    if not isinstance(heldout_trace, dict):
        _fail(f"{path}: shared_trace must be an object")
    try:
        heldout_seed = int(heldout_trace.get("sampling_seed", -1))
        heldout_requests = int(heldout_trace.get("requests", -1))
    except (TypeError, ValueError):
        _fail(f"{path}: invalid held-out seed/request count")
    if heldout_seed not in {43, 44, 45} or heldout_requests != 4000:
        _fail(f"{path}: held-out ablation must use seed 43/44/45 and 4,000 requests")
    if _integer(evidence.get("heldout_seed"), f"{evidence_path}:heldout_seed") != heldout_seed:
        _fail(f"{evidence_path}: heldout_seed mismatch")
    if _integer(
        evidence.get("heldout_requests"), f"{evidence_path}:heldout_requests"
    ) != heldout_requests:
        _fail(f"{evidence_path}: heldout_requests mismatch")
    manifest_round_dir = str(payload.get("round_dir") or "").strip()
    if manifest_round_dir and Path(manifest_round_dir).resolve() != Path(
        str(evidence.get("heldout_round_dir") or "")
    ).resolve():
        _fail(f"{evidence_path}: heldout_round_dir mismatch")

    heldout_commit = _manifest_commit(payload)
    if str(evidence.get("source_commit") or "").strip() != heldout_commit:
        _fail(f"{evidence_path}: source commit differs from held-out manifest")
    summary_commit = str(summary.get("source_commit") or "").strip()
    if summary_commit and summary_commit != heldout_commit:
        _fail(f"{path}: evidence-summary source commit mismatch")
    heldout_config = _validate_config_snapshot(
        path,
        payload.get("config_snapshot"),
        label=f"{path}:config_snapshot",
    )
    evidence_config_path = _resolve_record_path(
        evidence_path,
        evidence.get("config_path"),
        f"{evidence_path}:config_path",
    )
    evidence_config_sha = _digest(
        evidence.get("config_sha256"),
        f"{evidence_path}:config_sha256",
    )
    if (evidence_config_path, evidence_config_sha) != heldout_config[:2]:
        _fail(f"{evidence_path}: config snapshot differs from held-out manifest")
    summary_config_path = _resolve_record_path(
        path,
        summary.get("config_path"),
        f"{path}:seed41_validation_evidence.config_path",
    )
    summary_config_sha = _digest(
        summary.get("config_sha256"),
        f"{path}:seed41_validation_evidence.config_sha256",
    )
    if (summary_config_path, summary_config_sha) != heldout_config[:2]:
        _fail(f"{path}: evidence-summary config snapshot mismatch")
    evidence_registry_path = _resolve_record_path(
        evidence_path,
        evidence.get("registry_path"),
        f"{evidence_path}:registry_path",
    )
    summary_registry_path = _resolve_record_path(
        path,
        summary.get("registry_path"),
        f"{path}:seed41_validation_evidence.registry_path",
    )
    if evidence_registry_path != summary_registry_path:
        _fail(f"{path}: evidence-summary registry path mismatch")
    if _digest(
        evidence.get("registry_sha256_after_freeze"),
        f"{evidence_path}:registry_sha256_after_freeze",
    ) != _digest(
        summary.get("registry_sha256_after_freeze"),
        f"{path}:seed41_validation_evidence.registry_sha256_after_freeze",
    ):
        _fail(f"{path}: evidence-summary registry SHA mismatch")

    successful = evidence.get("successful_validation")
    if not isinstance(successful, dict):
        _fail(f"{evidence_path}: successful_validation must be an object")
    validation_path = _validated_file_identity(
        owner_path=evidence_path,
        raw_path=successful.get("manifest"),
        raw_size=successful.get("manifest_bytes"),
        raw_sha256=successful.get("manifest_sha256"),
        label=f"{evidence_path}:successful_validation.manifest",
    )
    summary_validation_path = _resolve_record_path(
        path,
        summary.get("successful_validation_manifest"),
        f"{path}:seed41_validation_evidence.successful_validation_manifest",
    )
    if summary_validation_path != validation_path:
        _fail(f"{path}: validation manifest path differs from evidence record")
    if _digest(
        summary.get("successful_validation_manifest_sha256"),
        f"{path}:seed41_validation_evidence.successful_validation_manifest_sha256",
    ) != _digest(
        successful.get("manifest_sha256"),
        f"{evidence_path}:successful_validation.manifest_sha256",
    ):
        _fail(f"{path}: validation manifest SHA differs from evidence record")
    if _size(
        summary.get("successful_validation_manifest_bytes"),
        f"{path}:seed41_validation_evidence.successful_validation_manifest_bytes",
    ) != _size(
        successful.get("manifest_bytes"),
        f"{evidence_path}:successful_validation.manifest_bytes",
    ):
        _fail(f"{path}: validation manifest byte count differs from evidence record")
    if _integer(successful.get("seed"), f"{evidence_path}:successful_validation.seed") != 41:
        _fail(f"{evidence_path}: successful validation seed must be 41")
    if _digest(
        successful.get("non_feature_frozen_config_sha256"),
        f"{evidence_path}:successful_validation.non_feature_frozen_config_sha256",
    ) != selected_hash:
        _fail(f"{evidence_path}: successful validation frozen hash mismatch")
    if str(successful.get("source_commit") or "").strip() != heldout_commit:
        _fail(f"{evidence_path}: successful validation source commit mismatch")
    if _digest(
        successful.get("config_sha256"),
        f"{evidence_path}:successful_validation.config_sha256",
    ) != heldout_config[1]:
        _fail(f"{evidence_path}: successful validation config SHA mismatch")
    successful_config_path = _resolve_record_path(
        evidence_path,
        successful.get("config_path"),
        f"{evidence_path}:successful_validation.config_path",
    )
    if successful_config_path != heldout_config[0]:
        _fail(f"{evidence_path}: successful validation config path mismatch")

    validation = _read_json(validation_path)
    _validate_seed41_validation_manifest(
        validation_path,
        validation,
        heldout_path=path,
        selected_hash=selected_hash,
        family=family,
        heldout_commit=heldout_commit,
        heldout_config=heldout_config,
    )


def _validate_manifest_header(path: Path, payload: Mapping[str, Any]) -> None:
    if payload.get("status") != "complete":
        _fail(f"{path}: status must be 'complete', observed {payload.get('status')!r}")
    if payload.get("formal_run") is not True:
        _fail(f"{path}: formal_run must be true")
    if str(payload.get("trace_role") or "").strip().lower() != "heldout":
        _fail(f"{path}: trace_role must be 'heldout'")
    if not _manifest_source_clean(payload):
        _fail(f"{path}: source_clean_for_formal must be true")

    # Fair-round manifests preserve both repositories; ablation manifests
    # preserve their single FaaSLoRA code snapshot.
    fair_commit_fields = []
    for name in ("baseline_git", "faaslora_git"):
        block = payload.get(name)
        if isinstance(block, dict):
            fair_commit_fields.append((name, str(block.get("commit") or "").strip()))
    if fair_commit_fields:
        for name, commit in fair_commit_fields:
            if not commit:
                _fail(f"{path}: {name}.commit is empty")
        if {name for name, _ in fair_commit_fields} != {"baseline_git", "faaslora_git"}:
            _fail(f"{path}: fair manifest must record both baseline_git and faaslora_git commits")
        return

    snapshot = payload.get("code_snapshot")
    snapshot_commit = (
        str(snapshot.get("git_commit") or "").strip()
        if isinstance(snapshot, dict)
        else ""
    )
    top_commit = str(payload.get("git_commit") or "").strip()
    if not (snapshot_commit or top_commit):
        _fail(f"{path}: committed source revision is empty")


def _manifest_top_config_hash(path: Path, payload: Mapping[str, Any]) -> str:
    raw = payload.get("system_resolved_config_sha256")
    if raw is None or str(raw).strip() == "":
        return ""
    return _digest(raw, f"{path}:system_resolved_config_sha256")


def _ablation_records(
    path: Path,
    payload: Mapping[str, Any],
    top_config_hash: str,
) -> list[_ManifestRecord]:
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return []
    records: list[_ManifestRecord] = []
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict) or not raw_entry.get("result_json"):
            continue
        label = f"{path}:entries[{index}]"
        config_hash = _digest(
            raw_entry.get("system_resolved_config_sha256") or top_config_hash,
            f"{label}.system_resolved_config_sha256",
        )
        records.append(
            _ManifestRecord(
                source=_resolve_record_path(
                    path,
                    raw_entry.get("result_json"),
                    f"{label}.result_json",
                ),
                manifest_path=path,
                sha256=_digest(raw_entry.get("sha256"), f"{label}.sha256"),
                size_bytes=_size(raw_entry.get("bytes"), f"{label}.bytes"),
                system_resolved_config_sha256=config_hash,
            )
        )
    return records


def _fair_records(
    path: Path,
    payload: Mapping[str, Any],
    top_config_hash: str,
) -> list[_ManifestRecord]:
    source_files = payload.get("source_files")
    if not isinstance(source_files, dict):
        return []
    if not top_config_hash:
        _fail(f"{path}: fair manifest is missing system_resolved_config_sha256")
    records: list[_ManifestRecord] = []
    for raw_name, raw_record in source_files.items():
        # Only JSON result material can be consumed by these analyzers.  Logs
        # and text reports are preserved by the manifest but are not analysis
        # inputs, so hashing them here would add cost without strengthening the
        # publication-data gate.
        name = str(raw_name)
        if not (name.endswith(".json") or name.endswith(".json.gz")):
            continue
        if not isinstance(raw_record, dict):
            _fail(f"{path}:source_files[{name!r}] must be an object")
        label = f"{path}:source_files[{name!r}]"
        records.append(
            _ManifestRecord(
                source=_resolve_record_path(path, name, label),
                manifest_path=path,
                sha256=_digest(raw_record.get("sha256"), f"{label}.sha256"),
                size_bytes=_size(raw_record.get("bytes"), f"{label}.bytes"),
                system_resolved_config_sha256=top_config_hash,
            )
        )
    return records


def build_formal_provenance_index(inputs: Sequence[Path]) -> FormalProvenanceIndex:
    """Validate formal manifest roots and index every recorded result JSON.

    A direct result-JSON argument is deliberately rejected even when its parent
    happens to contain a manifest.  Formal publication commands must name the
    campaign directory or its ``MANIFEST.json`` explicitly.
    """

    manifests: set[Path] = set()
    direct_manifests: set[Path] = set()
    for raw_input in inputs:
        input_path = Path(raw_input).expanduser().resolve()
        if input_path.is_file():
            if input_path.name != "MANIFEST.json":
                _fail(
                    f"loose raw JSON input is forbidden in formal mode: {input_path}; "
                    "pass its campaign MANIFEST.json or campaign directory"
                )
            manifests.add(input_path)
            direct_manifests.add(input_path)
        elif input_path.is_dir():
            manifests.update(path.resolve() for path in input_path.rglob("MANIFEST.json"))
        else:
            _fail(f"input does not exist: {input_path}")
    if not manifests:
        _fail("no MANIFEST.json was found in the formal inputs")

    by_source: Dict[Path, list[_ManifestRecord]] = {}
    recognized: list[Path] = []
    for manifest_path in sorted(manifests):
        payload = _read_json(manifest_path)
        has_ablation_schema = isinstance(payload.get("entries"), list)
        has_fair_schema = isinstance(payload.get("source_files"), dict)
        if not (has_ablation_schema or has_fair_schema):
            _fail(
                f"{manifest_path}: unsupported formal manifest schema; expected "
                "entries[] or source_files{}"
            )
        # A broad campaign-root input naturally contains the formal seed-41
        # validation round alongside held-out rounds.  It is not an analyzed
        # held-out identity; each held-out ablation independently revalidates
        # it through the immutable evidence chain below.
        if (
            has_ablation_schema
            and manifest_path not in direct_manifests
            and payload.get("formal_run") is True
            and str(payload.get("trace_role") or "").strip().lower() == "validation"
        ):
            continue
        _validate_manifest_header(manifest_path, payload)
        if has_ablation_schema:
            _validate_ablation_seed41_evidence(manifest_path, payload)
        top_hash = _manifest_top_config_hash(manifest_path, payload)
        records = (
            _ablation_records(manifest_path, payload, top_hash)
            if has_ablation_schema
            else _fair_records(manifest_path, payload, top_hash)
        )
        if not records:
            _fail(f"{manifest_path}: formal manifest records no result JSON files")
        recognized.append(manifest_path)
        for record in records:
            by_source.setdefault(record.source, []).append(record)

    if not recognized:
        _fail("no held-out formal MANIFEST.json was found in the formal inputs")

    return FormalProvenanceIndex(
        manifest_paths=tuple(recognized),
        records_by_source={key: tuple(value) for key, value in by_source.items()},
    )


def _validate_record_file(record: _ManifestRecord) -> None:
    source = record.source
    if not source.is_file():
        _fail(f"{record.manifest_path}: recorded source is missing: {source}")
    actual_size = source.stat().st_size
    if actual_size != record.size_bytes:
        _fail(
            f"{source}: byte-count mismatch; manifest={record.size_bytes}, actual={actual_size}"
        )
    actual_sha = _sha256(source)
    if actual_sha != record.sha256:
        _fail(f"{source}: SHA-256 mismatch; manifest={record.sha256}, actual={actual_sha}")

    payload = _read_json(source)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        raw_hash = metadata.get("system_resolved_config_sha256")
        if raw_hash is not None and str(raw_hash).strip():
            result_hash = _digest(
                raw_hash,
                f"{source}:metadata.system_resolved_config_sha256",
            )
            if result_hash != record.system_resolved_config_sha256:
                _fail(
                    f"{source}: result/manifest system_resolved_config_sha256 mismatch; "
                    f"manifest={record.system_resolved_config_sha256}, result={result_hash}"
                )
        raw_formal = metadata.get("formal_run")
        if raw_formal is not None and raw_formal is not True:
            _fail(f"{source}: metadata.formal_run contradicts formal manifest")
        raw_role = metadata.get("trace_role")
        if raw_role is not None and str(raw_role).strip().lower() != "heldout":
            _fail(f"{source}: metadata.trace_role contradicts heldout manifest")


def validate_formal_analysis_sources(
    index: FormalProvenanceIndex,
    identities: Iterable[FormalAnalysisIdentity],
    *,
    analysis_label: str,
) -> None:
    """Verify source coverage/checksums and frozen hashes for one analysis."""

    identity_rows = list(identities)
    if not identity_rows:
        _fail(f"{analysis_label}: no analyzed source identities were provided")

    checked: Dict[Path, _ManifestRecord] = {}
    hashes_by_group: Dict[tuple[str, str], set[str]] = {}
    seeds_by_group: Dict[tuple[str, str], set[int]] = {}
    for identity in identity_rows:
        source = Path(identity.source).expanduser().resolve()
        coverage = index.records_by_source.get(source, ())
        if not coverage:
            _fail(
                f"{analysis_label}: analyzed raw JSON has no MANIFEST.json record: {source}"
            )
        if len(coverage) != 1:
            manifests = ", ".join(str(item.manifest_path) for item in coverage)
            _fail(
                f"{analysis_label}: analyzed raw JSON is ambiguously covered by "
                f"{len(coverage)} manifests: {source} ({manifests})"
            )
        record = coverage[0]
        if source not in checked:
            _validate_record_file(record)
            checked[source] = record
        group = (str(identity.model), str(identity.variant))
        hashes_by_group.setdefault(group, set()).add(
            record.system_resolved_config_sha256
        )
        seeds_by_group.setdefault(group, set()).add(int(identity.seed))

    drift = {
        group: sorted(hashes)
        for group, hashes in hashes_by_group.items()
        if len(hashes) != 1
    }
    if drift:
        details = "; ".join(
            f"model={model},variant={variant},seeds={sorted(seeds_by_group[(model, variant)])},"
            f"hashes={hashes}"
            for (model, variant), hashes in sorted(drift.items())
        )
        _fail(f"{analysis_label}: frozen configuration drift detected: {details}")


def validate_formal_provenance(
    inputs: Sequence[Path],
    identities: Iterable[FormalAnalysisIdentity],
    *,
    analysis_label: str,
) -> FormalProvenanceIndex:
    """Convenience wrapper used by formal plotting entry points."""

    index = build_formal_provenance_index(inputs)
    validate_formal_analysis_sources(
        index,
        identities,
        analysis_label=analysis_label,
    )
    return index


def audit_legacy_full_pair(main: Path, ablation: Path, output: Path) -> dict:
    """Compare legacy Full sources without rewriting either historical result.

    Equality of input SHA is not equality of runs. This diagnostic deliberately
    does not call old runs formal TC evidence or attribute variation to a single
    observed state difference.
    """
    import csv
    if output.exists():
        raise ValueError('use a new audit directory; never overwrite provenance')
    paths = {'main_table': main.resolve(), 'ablation_full': ablation.resolve()}
    data = {k: _read_json(p) for k, p in paths.items()}
    def differences(a, b, prefix=''):
        if a == b:
            return []
        if isinstance(a, dict) and isinstance(b, dict):
            return [row for key in sorted(a.keys() | b.keys())
                    for row in differences(a.get(key), b.get(key), prefix+'.'+key)]
        return [{'field':prefix.lstrip('.'), 'main_table':a, 'ablation_full':b}]
    identities = {}
    for name, doc in data.items():
        meta = doc['metadata']
        inputs = {}
        for field in ('shared_trace_path', 'shared_adapter_subset_path'):
            path = Path(meta[field])
            inputs[field] = {'path':str(path), 'sha256':_sha256(path)}
        requests = doc['detailed_results']['faaslora_full']['requests']
        identities[name] = {'path':str(paths[name]), 'sha256':_sha256(paths[name]),
            'experiment_time':meta['experiment_time'], 'inputs':inputs,
            'request_count':len(requests),
            'generation_seed':meta.get('generation_seed'),
            'legacy_generation_contract':True}
    a, b = (data[k]['scenario_summaries']['faaslora_full'] for k in paths)
    rows = [{'metric':key, 'main_table':a.get(key), 'ablation_full':b.get(key),
             'equal':a.get(key)==b.get(key)} for key in sorted(a.keys() | b.keys())
            if not isinstance(a.get(key), (dict,list)) and not isinstance(b.get(key), (dict,list))]
    payload = {'kind':'legacy_full_provenance_audit', 'identities':identities,
        'same_trace': identities['main_table']['inputs']['shared_trace_path']['sha256'] == identities['ablation_full']['inputs']['shared_trace_path']['sha256'],
        'same_subset': identities['main_table']['inputs']['shared_adapter_subset_path']['sha256'] == identities['ablation_full']['inputs']['shared_adapter_subset_path']['sha256'],
        'same_result': _sha256(main)==_sha256(ablation),
        'metadata_differences':differences(data['main_table']['metadata'], data['ablation_full']['metadata']),
        'scalar_summary_rows':rows, 'reuse_for_tc_main':'R2',
        'conclusion':'Different executions of Full on identical recorded shared inputs; use one canonical run-set for the same main-result claim. No single-cause attribution from this comparison.'}
    output.mkdir(parents=True)
    with (output/'audit.json').open('x') as f:
        json.dump(payload,f,indent=2,ensure_ascii=False)
        f.write('\n')
    with (output/'all_scalar_metrics.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['metric','main_table','ablation_full','equal'],
                              lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    return payload
