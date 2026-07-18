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
    for raw_input in inputs:
        input_path = Path(raw_input).expanduser().resolve()
        if input_path.is_file():
            if input_path.name != "MANIFEST.json":
                _fail(
                    f"loose raw JSON input is forbidden in formal mode: {input_path}; "
                    "pass its campaign MANIFEST.json or campaign directory"
                )
            manifests.add(input_path)
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
        _validate_manifest_header(manifest_path, payload)
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
