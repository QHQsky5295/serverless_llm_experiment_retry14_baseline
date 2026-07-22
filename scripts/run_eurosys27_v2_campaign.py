#!/usr/bin/env python3
"""Fail-closed orchestration for the EuroSys'27 V2 experiment campaign.

This program is intentionally an outer coordinator.  It does not reproduce the
logic in ``run_full_fair_round.sh`` and it never invokes that runner's dry-run
mode.  ``plan`` and ``status`` are read-only; execution commands create unique
attempt directories, freeze their environment, append a locked JSONL ledger,
and call the configured runner normally.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "eurosys27_v2_campaign_v1"
RUN_KEY_SCHEMA_VERSION = "eurosys27_v2_campaign_run_key_v1"
LEDGER_SCHEMA_VERSION = "eurosys27_v2_campaign_ledger_v1"
ATTEMPT_SCHEMA_VERSION = "eurosys27_v2_campaign_attempt_v1"
EXPORT_SCHEMA_VERSION = "eurosys27_v2_campaign_analyzer_inputs_v1"
CACHE_PRUNE_SCHEMA_VERSION = "eurosys27_v2_cache_prune_receipt_v1"

SAFE_INHERITED_ENV = {
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
    "CPATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "HF_HOME",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "LOGNAME",
    "MODELSCOPE_CACHE",
    "PATH",
    "PYTHONPATH",
    "SHELL",
    "TORCH_HOME",
    "TRANSFORMERS_CACHE",
    "USER",
    "VIRTUAL_ENV",
    "XDG_CACHE_HOME",
}
RESERVED_RUN_ENV = {
    "FAIR_RESOLVED_CONFIG_REGISTRY",
    "FAIR_ROUND_DIR",
    "FAIR_ROUND_DRY_RUN",
    "FAIR_ROUND_FORCE",
    "FAIR_ROUND_LABEL",
    "FAIR_ROUND_ROOT",
    "FAIR_ROUND_TIMESTAMP",
    "PAPER_QUEUE_DRY_RUN",
    "SLLM_RUN_TAG",
    "V2_CAMPAIGN_ATTEMPT_ID",
    "V2_CAMPAIGN_CONFIG_SHA256",
    "V2_CAMPAIGN_RUN_ID",
    "V2_CAMPAIGN_RUN_KEY",
}
SENSITIVE_ENV_RE = re.compile(
    r"(?:^|_)(?:API_?KEY|ACCESS_?KEY|AUTH|BEARER|CREDENTIAL|PASSWORD|SECRET|TOKEN)(?:$|_)",
    re.IGNORECASE,
)
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CampaignError(RuntimeError):
    """A protocol or campaign-state violation."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _timestamp_token() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ").lower()


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignError(f"{context} must be a JSON object")
    return value


def _reject_unknown(
    value: Mapping[str, Any], allowed: Iterable[str], context: str
) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise CampaignError(f"unknown fields in {context}: {unknown}")


def _require_string(value: Any, context: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise CampaignError(
            f"{context} must be a {'non-empty ' if nonempty else ''}string"
        )
    return value


def _absolute_path(value: Any, context: str) -> Path:
    path = Path(_require_string(value, context)).expanduser()
    if not path.is_absolute():
        raise CampaignError(f"{context} must be an absolute path: {path}")
    return path.resolve()


def _string_map(value: Any, context: str) -> dict[str, str]:
    mapping = _require_mapping(value, context)
    result: dict[str, str] = {}
    for raw_key, raw_value in mapping.items():
        key = _require_string(raw_key, f"{context} key")
        if not isinstance(raw_value, (str, int, float, bool)):
            raise CampaignError(f"{context}.{key} must be a scalar")
        result[key] = (
            "1" if raw_value is True else "0" if raw_value is False else str(raw_value)
        )
    return result


def _command(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise CampaignError(f"{context} must be a non-empty argv array")
    command = tuple(_require_string(item, f"{context}[]") for item in value)
    if any("\x00" in item for item in command):
        raise CampaignError(f"{context} contains a NUL byte")
    return command


def _slug(value: str, limit: int = 80) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return (value or "run")[:limit].rstrip("_")


def _safe_relative(value: Any, context: str, *, allow_glob: bool = False) -> str:
    text = _require_string(value, context)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise CampaignError(f"{context} must stay below the attempt directory: {text}")
    if not allow_glob and any(ch in text for ch in "*?["):
        raise CampaignError(f"{context} may not contain glob metacharacters")
    return text


@dataclass(frozen=True)
class CleanupSpec:
    strict_gpu_idle: bool
    gpu_ids: tuple[int, ...]
    timeout_seconds: int
    poll_seconds: float
    prune_completed_cache: bool
    pre_commands: tuple[tuple[str, ...], ...]
    post_commands: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class SourceRepo:
    name: str
    path: Path
    required_branch: str | None
    allowed_dirty_paths: tuple[str, ...]


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    command: tuple[str, ...] | None
    working_directory: Path | None
    registry_name: str | None
    registry_path: Path | None
    environment: Mapping[str, str]
    depends_on: tuple[str, ...]
    reuse_of: str | None
    manifest_relative_path: str
    run_key: str


@dataclass(frozen=True)
class InputSelector:
    name: str
    pattern: str
    multiple: bool
    required: bool


@dataclass(frozen=True)
class AnalyzerExport:
    export_id: str
    run_ids: tuple[str, ...]
    output_directory: Path
    selectors: tuple[InputSelector, ...]


@dataclass(frozen=True)
class CampaignConfig:
    path: Path
    raw: Mapping[str, Any]
    config_sha256: str
    campaign_id: str
    campaign_root: Path
    inherit_environment: tuple[str, ...]
    base_environment: Mapping[str, str]
    registries: Mapping[str, Path]
    cleanup: CleanupSpec
    source_repositories: tuple[SourceRepo, ...]
    runs: tuple[RunSpec, ...]
    exports: tuple[AnalyzerExport, ...]

    @property
    def run_by_id(self) -> dict[str, RunSpec]:
        return {run.run_id: run for run in self.runs}


def _validate_environment_names(values: Mapping[str, str], context: str) -> None:
    for name in values:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise CampaignError(f"invalid environment name in {context}: {name!r}")
        if SENSITIVE_ENV_RE.search(name):
            raise CampaignError(
                f"{context} may not persist credential-like environment variable {name!r}"
            )


def _run_key_payload(
    *,
    command: Sequence[str],
    working_directory: Path,
    registry_path: Path,
    base_environment: Mapping[str, str],
    inherit_environment: Sequence[str],
    environment: Mapping[str, str],
    manifest_relative_path: str,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_KEY_SCHEMA_VERSION,
        "command": list(command),
        "working_directory": str(working_directory),
        "registry_path": str(registry_path),
        "base_environment": dict(base_environment),
        "inherited_environment_names": list(inherit_environment),
        "run_environment": dict(environment),
        "manifest_relative_path": manifest_relative_path,
    }


def _topological_order(runs: Sequence[RunSpec]) -> list[str]:
    dependencies = {
        run.run_id: set(run.depends_on) | ({run.reuse_of} if run.reuse_of else set())
        for run in runs
    }
    result: list[str] = []
    remaining = {key: set(value) for key, value in dependencies.items()}
    while remaining:
        ready = [
            run.run_id
            for run in runs
            if run.run_id in remaining and not remaining[run.run_id]
        ]
        if not ready:
            raise CampaignError(
                "run dependency/reuse graph contains a cycle: "
                + ", ".join(sorted(remaining))
            )
        for run_id in ready:
            result.append(run_id)
            remaining.pop(run_id)
            for needed in remaining.values():
                needed.discard(run_id)
    return result


def load_config(path: Path) -> CampaignConfig:
    path = path.expanduser().resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CampaignError(f"campaign config does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CampaignError(f"invalid JSON in {path}: {exc}") from exc
    raw = _require_mapping(raw, "campaign config")
    _reject_unknown(
        raw,
        {
            "$schema",
            "schema_version",
            "campaign_id",
            "campaign_root",
            "environment",
            "registries",
            "source_repositories",
            "cleanup",
            "runs",
            "analyzer_exports",
        },
        "campaign config",
    )
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise CampaignError(
            f"schema_version must be {SCHEMA_VERSION!r}; got {raw.get('schema_version')!r}"
        )
    campaign_id = _require_string(raw.get("campaign_id"), "campaign_id")
    if not ID_RE.fullmatch(campaign_id):
        raise CampaignError(f"invalid campaign_id: {campaign_id!r}")
    campaign_root = _absolute_path(raw.get("campaign_root"), "campaign_root")

    env_spec = _require_mapping(raw.get("environment", {}), "environment")
    _reject_unknown(env_spec, {"inherit", "values"}, "environment")
    inherit_raw = env_spec.get("inherit", [])
    if not isinstance(inherit_raw, list):
        raise CampaignError("environment.inherit must be an array")
    inherit = tuple(
        _require_string(item, "environment.inherit[]") for item in inherit_raw
    )
    if len(inherit) != len(set(inherit)):
        raise CampaignError("environment.inherit contains duplicates")
    forbidden_inherit = sorted(set(inherit) - SAFE_INHERITED_ENV)
    if forbidden_inherit:
        raise CampaignError(
            "environment.inherit contains non-auditable or credential-prone names: "
            + ", ".join(forbidden_inherit)
        )
    base_environment = _string_map(env_spec.get("values", {}), "environment.values")
    _validate_environment_names(base_environment, "environment.values")
    reserved_base = sorted(set(base_environment) & RESERVED_RUN_ENV)
    if reserved_base:
        raise CampaignError(
            "environment.values may not override orchestrator-owned variables: "
            f"{reserved_base}"
        )
    if set(base_environment) & set(inherit):
        raise CampaignError("environment values and inherit lists overlap")

    registries_raw = _require_mapping(raw.get("registries"), "registries")
    if not registries_raw:
        raise CampaignError(
            "registries must explicitly name at least one registry path"
        )
    registries = {
        _require_string(name, "registries key"): _absolute_path(
            value, f"registries.{name}"
        )
        for name, value in registries_raw.items()
    }

    cleanup_raw = _require_mapping(raw.get("cleanup", {}), "cleanup")
    _reject_unknown(
        cleanup_raw,
        {
            "strict_gpu_idle",
            "gpu_ids",
            "timeout_seconds",
            "poll_seconds",
            "prune_completed_cache",
            "pre_commands",
            "post_commands",
        },
        "cleanup",
    )
    strict_gpu_idle = cleanup_raw.get("strict_gpu_idle", True)
    if not isinstance(strict_gpu_idle, bool):
        raise CampaignError("cleanup.strict_gpu_idle must be boolean")
    gpu_ids_raw = cleanup_raw.get("gpu_ids", [])
    if not isinstance(gpu_ids_raw, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in gpu_ids_raw
    ):
        raise CampaignError("cleanup.gpu_ids must be an array of non-negative integers")
    gpu_ids = tuple(gpu_ids_raw)
    if len(gpu_ids) != len(set(gpu_ids)):
        raise CampaignError("cleanup.gpu_ids contains duplicates")
    if strict_gpu_idle and not gpu_ids:
        raise CampaignError("strict cleanup verification requires cleanup.gpu_ids")
    timeout_seconds = cleanup_raw.get("timeout_seconds", 180)
    poll_seconds = cleanup_raw.get("poll_seconds", 5)
    if (
        not isinstance(timeout_seconds, int)
        or isinstance(timeout_seconds, bool)
        or timeout_seconds < 0
    ):
        raise CampaignError("cleanup.timeout_seconds must be a non-negative integer")
    if (
        not isinstance(poll_seconds, (int, float))
        or isinstance(poll_seconds, bool)
        or poll_seconds <= 0
    ):
        raise CampaignError("cleanup.poll_seconds must be positive")
    prune_completed_cache = cleanup_raw.get("prune_completed_cache", False)
    if not isinstance(prune_completed_cache, bool):
        raise CampaignError("cleanup.prune_completed_cache must be boolean")
    command_groups: dict[str, tuple[tuple[str, ...], ...]] = {}
    for field in ("pre_commands", "post_commands"):
        commands_raw = cleanup_raw.get(field, [])
        if not isinstance(commands_raw, list):
            raise CampaignError(f"cleanup.{field} must be an array of argv arrays")
        command_groups[field] = tuple(
            _command(command, f"cleanup.{field}[]") for command in commands_raw
        )
    cleanup = CleanupSpec(
        strict_gpu_idle=strict_gpu_idle,
        gpu_ids=gpu_ids,
        timeout_seconds=timeout_seconds,
        poll_seconds=float(poll_seconds),
        prune_completed_cache=prune_completed_cache,
        pre_commands=command_groups["pre_commands"],
        post_commands=command_groups["post_commands"],
    )

    sources_raw = _require_mapping(
        raw.get("source_repositories", {}), "source_repositories"
    )
    source_repositories: list[SourceRepo] = []
    for name, source_value in sources_raw.items():
        source = _require_mapping(source_value, f"source_repositories.{name}")
        _reject_unknown(
            source,
            {"path", "required_branch", "allowed_tracked_dirty_paths"},
            f"source_repositories.{name}",
        )
        allowed = source.get("allowed_tracked_dirty_paths", [])
        if not isinstance(allowed, list):
            raise CampaignError(
                f"source_repositories.{name}.allowed_tracked_dirty_paths must be an array"
            )
        source_repositories.append(
            SourceRepo(
                name=_require_string(name, "source_repositories key"),
                path=_absolute_path(
                    source.get("path"), f"source_repositories.{name}.path"
                ),
                required_branch=(
                    _require_string(
                        source["required_branch"],
                        f"source_repositories.{name}.required_branch",
                    )
                    if source.get("required_branch") is not None
                    else None
                ),
                allowed_dirty_paths=tuple(
                    _safe_relative(
                        item,
                        f"source_repositories.{name}.allowed_tracked_dirty_paths[]",
                    )
                    for item in allowed
                ),
            )
        )

    runs_raw = raw.get("runs")
    if not isinstance(runs_raw, list) or not runs_raw:
        raise CampaignError("runs must be a non-empty array")
    partial_runs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_run_value in enumerate(runs_raw):
        raw_run = _require_mapping(raw_run_value, f"runs[{index}]")
        run_id = _require_string(raw_run.get("id"), f"runs[{index}].id")
        if not ID_RE.fullmatch(run_id):
            raise CampaignError(f"invalid run id: {run_id!r}")
        if run_id in seen_ids:
            raise CampaignError(f"duplicate run id: {run_id}")
        seen_ids.add(run_id)
        dependencies_raw = raw_run.get("depends_on", [])
        if not isinstance(dependencies_raw, list):
            raise CampaignError(f"runs[{index}].depends_on must be an array")
        dependencies = tuple(
            _require_string(item, f"runs[{index}].depends_on[]")
            for item in dependencies_raw
        )
        if len(dependencies) != len(set(dependencies)) or run_id in dependencies:
            raise CampaignError(f"invalid dependencies for run {run_id}")
        reuse_of = raw_run.get("reuse_of")
        if reuse_of is not None:
            _reject_unknown(
                raw_run,
                {"id", "depends_on", "reuse_of"},
                f"runs[{index}]",
            )
            reuse_of = _require_string(reuse_of, f"runs[{index}].reuse_of")
            forbidden = {"runner", "working_directory", "registry", "env"} & set(
                raw_run
            )
            if forbidden:
                raise CampaignError(
                    f"reuse run {run_id} may not define {sorted(forbidden)}"
                )
            partial_runs.append(
                {
                    "run_id": run_id,
                    "depends_on": dependencies,
                    "reuse_of": reuse_of,
                }
            )
            continue
        _reject_unknown(
            raw_run,
            {
                "id",
                "runner",
                "working_directory",
                "registry",
                "depends_on",
                "env",
                "manifest_relative_path",
            },
            f"runs[{index}]",
        )
        command = _command(raw_run.get("runner"), f"runs[{index}].runner")
        working_directory = _absolute_path(
            raw_run.get("working_directory"), f"runs[{index}].working_directory"
        )
        registry_name = _require_string(
            raw_run.get("registry"), f"runs[{index}].registry"
        )
        if registry_name not in registries:
            raise CampaignError(
                f"run {run_id} references unknown registry {registry_name!r}"
            )
        environment = _string_map(raw_run.get("env", {}), f"runs[{index}].env")
        _validate_environment_names(environment, f"runs[{index}].env")
        reserved = sorted(set(environment) & RESERVED_RUN_ENV)
        if reserved:
            raise CampaignError(
                f"run {run_id} may not override orchestrator-owned variables: {reserved}"
            )
        for dry_name in ("FAIR_ROUND_DRY_RUN", "PAPER_QUEUE_DRY_RUN"):
            if environment.get(dry_name, "0") not in {"", "0"}:
                raise CampaignError(
                    f"run {run_id} attempts to enable forbidden formal dry-run variable {dry_name}"
                )
        manifest_path = _safe_relative(
            raw_run.get("manifest_relative_path", "MANIFEST.json"),
            f"runs[{index}].manifest_relative_path",
        )
        payload = _run_key_payload(
            command=command,
            working_directory=working_directory,
            registry_path=registries[registry_name],
            base_environment=base_environment,
            inherit_environment=inherit,
            environment=environment,
            manifest_relative_path=manifest_path,
        )
        partial_runs.append(
            {
                "run_id": run_id,
                "command": command,
                "working_directory": working_directory,
                "registry_name": registry_name,
                "registry_path": registries[registry_name],
                "environment": environment,
                "depends_on": dependencies,
                "reuse_of": None,
                "manifest_relative_path": manifest_path,
                "run_key": _sha256_value(payload),
            }
        )

    partial_by_id = {item["run_id"]: item for item in partial_runs}
    for item in partial_runs:
        references = list(item["depends_on"])
        if item.get("reuse_of"):
            references.append(item["reuse_of"])
        missing = sorted(set(references) - set(partial_by_id))
        if missing:
            raise CampaignError(
                f"run {item['run_id']} references unknown runs: {missing}"
            )

    def resolve_reuse(run_id: str, trail: tuple[str, ...] = ()) -> dict[str, Any]:
        if run_id in trail:
            raise CampaignError("reuse cycle: " + " -> ".join((*trail, run_id)))
        item = partial_by_id[run_id]
        source_id = item.get("reuse_of")
        return resolve_reuse(source_id, (*trail, run_id)) if source_id else item

    runs: list[RunSpec] = []
    for item in partial_runs:
        if item.get("reuse_of"):
            source = resolve_reuse(item["run_id"])
            runs.append(
                RunSpec(
                    run_id=item["run_id"],
                    command=None,
                    working_directory=None,
                    registry_name=None,
                    registry_path=None,
                    environment={},
                    depends_on=item["depends_on"],
                    reuse_of=item["reuse_of"],
                    manifest_relative_path=source["manifest_relative_path"],
                    run_key=source["run_key"],
                )
            )
        else:
            runs.append(RunSpec(**item))
    _topological_order(runs)

    exports_raw = raw.get("analyzer_exports", [])
    if not isinstance(exports_raw, list):
        raise CampaignError("analyzer_exports must be an array")
    exports: list[AnalyzerExport] = []
    export_ids: set[str] = set()
    for index, raw_export_value in enumerate(exports_raw):
        raw_export = _require_mapping(raw_export_value, f"analyzer_exports[{index}]")
        _reject_unknown(
            raw_export,
            {"id", "run_ids", "output_directory", "inputs"},
            f"analyzer_exports[{index}]",
        )
        export_id = _require_string(
            raw_export.get("id"), f"analyzer_exports[{index}].id"
        )
        if not ID_RE.fullmatch(export_id) or export_id in export_ids:
            raise CampaignError(
                f"invalid or duplicate analyzer export id: {export_id!r}"
            )
        export_ids.add(export_id)
        export_runs_raw = raw_export.get("run_ids")
        if not isinstance(export_runs_raw, list) or not export_runs_raw:
            raise CampaignError(f"analyzer export {export_id} requires run_ids")
        export_runs = tuple(
            _require_string(item, f"analyzer_exports[{index}].run_ids[]")
            for item in export_runs_raw
        )
        missing = sorted(set(export_runs) - seen_ids)
        if missing:
            raise CampaignError(
                f"analyzer export {export_id} has unknown runs: {missing}"
            )
        selectors_raw = raw_export.get("inputs")
        if not isinstance(selectors_raw, list) or not selectors_raw:
            raise CampaignError(f"analyzer export {export_id} requires inputs")
        selectors: list[InputSelector] = []
        selector_names: set[str] = set()
        for selector_index, raw_selector_value in enumerate(selectors_raw):
            raw_selector = _require_mapping(
                raw_selector_value,
                f"analyzer_exports[{index}].inputs[{selector_index}]",
            )
            _reject_unknown(
                raw_selector,
                {"name", "glob", "multiple", "required"},
                f"analyzer_exports[{index}].inputs[{selector_index}]",
            )
            name = _require_string(
                raw_selector.get("name"),
                f"analyzer_exports[{index}].inputs[{selector_index}].name",
            )
            if not ID_RE.fullmatch(name) or name in selector_names:
                raise CampaignError(f"invalid or duplicate selector name: {name!r}")
            selector_names.add(name)
            multiple = raw_selector.get("multiple", False)
            required = raw_selector.get("required", True)
            if not isinstance(multiple, bool) or not isinstance(required, bool):
                raise CampaignError(
                    f"analyzer export {export_id} selector booleans must be true/false"
                )
            selectors.append(
                InputSelector(
                    name=name,
                    pattern=_safe_relative(
                        raw_selector.get("glob"),
                        f"analyzer_exports[{index}].inputs[{selector_index}].glob",
                        allow_glob=True,
                    ),
                    multiple=multiple,
                    required=required,
                )
            )
        exports.append(
            AnalyzerExport(
                export_id=export_id,
                run_ids=export_runs,
                output_directory=_absolute_path(
                    raw_export.get("output_directory"),
                    f"analyzer_exports[{index}].output_directory",
                ),
                selectors=tuple(selectors),
            )
        )

    config = CampaignConfig(
        path=path,
        raw=raw,
        config_sha256=_sha256_value(raw),
        campaign_id=campaign_id,
        campaign_root=campaign_root,
        inherit_environment=inherit,
        base_environment=base_environment,
        registries=registries,
        cleanup=cleanup,
        source_repositories=tuple(source_repositories),
        runs=tuple(runs),
        exports=tuple(exports),
    )
    _preflight(config)
    return config


def _tracked_dirty_paths(repo: Path) -> list[str]:
    output = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=no",
        ]
    )
    paths: list[str] = []
    records = output.split(b"\0")
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        text = record.decode("utf-8", errors="surrogateescape")
        path = text[3:]
        if text[:2] in {"R ", "C ", "RM", "CM"} and index < len(records):
            path = records[index].decode("utf-8", errors="surrogateescape")
            index += 1
        paths.append(path)
    return sorted(set(paths))


def _source_identity(source: SourceRepo) -> dict[str, Any]:
    if not (source.path / ".git").exists():
        raise CampaignError(f"source repository is not a git worktree: {source.path}")
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(source.path), "rev-parse", "HEAD"], text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "-C", str(source.path), "branch", "--show-current"], text=True
        ).strip()
        dirty_paths = _tracked_dirty_paths(source.path)
        binary_diff = subprocess.check_output(
            [
                "git",
                "-C",
                str(source.path),
                "diff",
                "--binary",
                "--full-index",
                "HEAD",
                "--",
            ]
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CampaignError(f"cannot inspect source repository: {source.path}") from exc
    if source.required_branch and branch != source.required_branch:
        raise CampaignError(
            f"source {source.name} branch mismatch: expected={source.required_branch} observed={branch}"
        )
    unexpected = sorted(set(dirty_paths) - set(source.allowed_dirty_paths))
    if unexpected:
        raise CampaignError(
            f"source {source.name} has forbidden tracked changes: {unexpected}"
        )
    return {
        "name": source.name,
        "path": str(source.path),
        "branch": branch,
        "commit": commit,
        "tracked_dirty_paths": dirty_paths,
        "head_to_worktree_binary_diff_sha256": hashlib.sha256(binary_diff).hexdigest(),
    }


def _resolve_executable(
    command: Sequence[str], environment: Mapping[str, str]
) -> str | None:
    executable = command[0]
    if "/" in executable:
        path = Path(executable)
        return str(path) if path.is_file() else None
    return shutil.which(executable, path=environment.get("PATH", os.defpath))


def _preview_base_environment(config: CampaignConfig) -> dict[str, str]:
    result = {
        name: os.environ[name]
        for name in config.inherit_environment
        if name in os.environ
    }
    result.update(config.base_environment)
    return result


def _preflight(config: CampaignConfig) -> None:
    base_environment = _preview_base_environment(config)
    for run in config.runs:
        if run.reuse_of:
            continue
        assert run.command is not None and run.working_directory is not None
        if not run.working_directory.is_dir():
            raise CampaignError(
                f"run {run.run_id} working directory does not exist: {run.working_directory}"
            )
        execution_environment = dict(base_environment)
        execution_environment.update(run.environment)
        if _resolve_executable(run.command, execution_environment) is None:
            raise CampaignError(
                f"run {run.run_id} executable is unavailable: {run.command[0]}"
            )
        if execution_environment.get("FAIR_ROUND_DRY_RUN", "0") not in {"", "0"}:
            raise CampaignError(
                "formal campaign orchestration never permits FAIR_ROUND_DRY_RUN"
            )
        if execution_environment.get("PAPER_QUEUE_DRY_RUN", "0") not in {"", "0"}:
            raise CampaignError(
                "formal campaign orchestration never permits PAPER_QUEUE_DRY_RUN"
            )
    for command in (*config.cleanup.pre_commands, *config.cleanup.post_commands):
        if _resolve_executable(command, base_environment) is None:
            raise CampaignError(f"cleanup hook executable is unavailable: {command[0]}")
    for source in config.source_repositories:
        _source_identity(source)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8")
        + b"\n",
    )


def _ledger_path(config: CampaignConfig) -> Path:
    return config.campaign_root / "campaign_ledger.jsonl"


def _parse_ledger(config: CampaignConfig) -> list[dict[str, Any]]:
    path = _ledger_path(config)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            raise CampaignError(
                f"blank line in append-only ledger at line {line_number}"
            )
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CampaignError(
                f"invalid ledger JSON at line {line_number}: {exc}"
            ) from exc
        if event.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise CampaignError(f"wrong ledger schema at line {line_number}")
        if int(event.get("sequence", -1)) != len(events) + 1:
            raise CampaignError(f"non-contiguous ledger sequence at line {line_number}")
        events.append(event)
    return events


def _read_ledger(
    config: CampaignConfig, *, writer_lock_already_held: bool = False
) -> list[dict[str, Any]]:
    path = _ledger_path(config)
    if not path.exists() or writer_lock_already_held:
        return _parse_ledger(config)
    lock_path = config.campaign_root / "campaign_ledger.lock"
    if not lock_path.is_file():
        raise CampaignError(f"ledger exists without its lock file: {lock_path}")
    with lock_path.open("r", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
        return _parse_ledger(config)


def _append_event(config: CampaignConfig, event: Mapping[str, Any]) -> dict[str, Any]:
    lock_path = config.campaign_root / "campaign_ledger.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        events = _read_ledger(config, writer_lock_already_held=True)
        payload = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "sequence": len(events) + 1,
            "timestamp_utc": _utc_now(),
            "campaign_id": config.campaign_id,
            "config_sha256": config.config_sha256,
            **event,
        }
        encoded = _canonical_bytes(payload) + b"\n"
        descriptor = os.open(
            _ledger_path(config), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644
        )
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return payload


class _ExecutionLock:
    def __init__(self, config: CampaignConfig):
        self.path = config.campaign_root / "campaign_execution.lock"
        self.handle: Any = None

    def __enter__(self) -> "_ExecutionLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            raise CampaignError(
                "another campaign mutation/execution is active"
            ) from exc
        return self

    def __exit__(self, *_: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def _ensure_initialized(config: CampaignConfig) -> dict[str, str]:
    config.campaign_root.mkdir(parents=True, exist_ok=True)
    protocol = config.campaign_root / "protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    events = _read_ledger(config)
    initialized = [
        event for event in events if event.get("event") == "campaign_initialized"
    ]
    if initialized:
        first = initialized[0]
        if first.get("config_sha256") != config.config_sha256:
            raise CampaignError(
                "campaign config changed after initialization; use a new campaign_root"
            )
        if len(initialized) != 1:
            raise CampaignError("ledger contains multiple campaign_initialized events")
        environment_path = protocol / "campaign_environment.json"
        source_path = protocol / "campaign_sources.json"
        snapshot_path = protocol / "campaign_config.json"
        for field, path in (
            ("environment_sha256", environment_path),
            ("sources_sha256", source_path),
            ("config_snapshot_sha256", snapshot_path),
        ):
            if not path.is_file() or _sha256_file(path) != first.get(field):
                raise CampaignError(
                    f"frozen campaign protocol file failed integrity: {path}"
                )
        payload = json.loads(environment_path.read_text(encoding="utf-8"))
        return {str(key): str(value) for key, value in payload["environment"].items()}

    frozen_environment = _preview_base_environment(config)
    source_payload = {
        "schema_version": SCHEMA_VERSION,
        "repositories": [
            _source_identity(source) for source in config.source_repositories
        ],
    }
    environment_payload = {
        "schema_version": SCHEMA_VERSION,
        "inherited_names": list(config.inherit_environment),
        "environment": frozen_environment,
    }
    snapshot_path = protocol / "campaign_config.json"
    environment_path = protocol / "campaign_environment.json"
    source_path = protocol / "campaign_sources.json"
    _atomic_json(snapshot_path, config.raw)
    _atomic_json(environment_path, environment_payload)
    _atomic_json(source_path, source_payload)
    _append_event(
        config,
        {
            "event": "campaign_initialized",
            "config_snapshot_path": str(snapshot_path),
            "config_snapshot_sha256": _sha256_file(snapshot_path),
            "environment_path": str(environment_path),
            "environment_sha256": _sha256_file(environment_path),
            "sources_path": str(source_path),
            "sources_sha256": _sha256_file(source_path),
        },
    )
    return frozen_environment


def _verify_frozen_sources(config: CampaignConfig) -> None:
    source_path = config.campaign_root / "protocol" / "campaign_sources.json"
    expected = json.loads(source_path.read_text(encoding="utf-8"))
    observed = {
        "schema_version": SCHEMA_VERSION,
        "repositories": [
            _source_identity(source) for source in config.source_repositories
        ],
    }
    if observed != expected:
        raise CampaignError(
            "source repositories changed after campaign initialization; start a new campaign"
        )


def _attempt_events(
    events: Sequence[Mapping[str, Any]]
) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        attempt_id = event.get("attempt_id")
        if isinstance(attempt_id, str):
            result.setdefault(attempt_id, []).append(event)
    return result


def _pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _real_run_state(
    config: CampaignConfig, run: RunSpec, events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    attempts: dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        if event.get("run_id") == run.run_id and isinstance(
            event.get("attempt_id"), str
        ):
            attempts.setdefault(str(event["attempt_id"]), []).append(event)
    if not attempts:
        return {"status": "pending", "attempt_id": None}
    ordered = sorted(
        attempts.items(), key=lambda pair: int(pair[1][0].get("attempt_number", 0))
    )
    attempt_id, attempt_log = ordered[-1]
    last = attempt_log[-1]
    event_name = str(last.get("event"))
    if event_name == "attempt_completed":
        manifest = Path(str(last.get("manifest_path") or ""))
        if not manifest.is_file() or _sha256_file(manifest) != last.get(
            "manifest_sha256"
        ):
            status = "invalid"
        else:
            try:
                status = (
                    "complete"
                    if json.loads(manifest.read_text(encoding="utf-8")).get("status")
                    == "complete"
                    else "invalid"
                )
            except (OSError, json.JSONDecodeError):
                status = "invalid"
    elif event_name == "attempt_failed":
        status = "failed"
    elif event_name in {"attempt_started", "attempt_resumed"}:
        status = "running" if _pid_alive(last.get("executor_pid")) else "interrupted"
    else:
        status = "invalid"
    return {
        "status": status,
        "attempt_id": attempt_id,
        "attempt_number": int(attempt_log[0].get("attempt_number", 0)),
        "attempt_dir": attempt_log[0].get("attempt_dir"),
        "run_tag": attempt_log[0].get("run_tag"),
        "last_event": event_name,
        "manifest_path": next(
            (
                event.get("manifest_path")
                for event in reversed(attempt_log)
                if event.get("manifest_path")
            ),
            None,
        ),
    }


def campaign_status(config: CampaignConfig) -> dict[str, Any]:
    events = _read_ledger(config)
    initialized = next(
        (event for event in events if event.get("event") == "campaign_initialized"),
        None,
    )
    config_matches = (
        initialized is None or initialized.get("config_sha256") == config.config_sha256
    )
    states: dict[str, dict[str, Any]] = {}
    for run in config.runs:
        if run.reuse_of:
            source_run = _source_real_run(config, run.run_id)
            source = states.get(source_run.run_id)
            if source is None:
                source = _real_run_state(config, source_run, events)
            source_status = source["status"]
            states[run.run_id] = {
                "status": (
                    "complete"
                    if source_status == "complete"
                    else f"reuse_{source_status}"
                ),
                "reuse_of": run.reuse_of,
                "attempt_id": source.get("attempt_id"),
                "attempt_dir": source.get("attempt_dir"),
                "manifest_path": source.get("manifest_path"),
            }
        else:
            states[run.run_id] = _real_run_state(config, run, events)
        states[run.run_id]["run_key"] = run.run_key
        states[run.run_id]["depends_on"] = list(run.depends_on)
    for run in config.runs:
        state = states[run.run_id]
        dependencies_complete = all(
            states[dependency]["status"] == "complete" for dependency in run.depends_on
        )
        if not dependencies_complete:
            if state["status"] == "pending":
                state["status"] = "blocked"
            elif run.reuse_of and state["status"] == "complete":
                state["status"] = "reuse_blocked"
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": config.campaign_id,
        "campaign_root": str(config.campaign_root),
        "config_sha256": config.config_sha256,
        "initialized": initialized is not None,
        "config_matches_initialized_campaign": config_matches,
        "ledger_path": str(_ledger_path(config)),
        "ledger_events": len(events),
        "runs": states,
    }


def _next_attempt_number(events: Sequence[Mapping[str, Any]], run_id: str) -> int:
    numbers = [
        int(event.get("attempt_number", 0))
        for event in events
        if event.get("run_id") == run_id and event.get("event") == "attempt_started"
    ]
    return max(numbers, default=0) + 1


def _new_attempt(config: CampaignConfig, run: RunSpec, mode: str) -> dict[str, Any]:
    events = _read_ledger(config)
    number = _next_attempt_number(events, run.run_id)
    unique = uuid.uuid4().hex[:10]
    token = _timestamp_token()
    attempt_id = f"{_slug(run.run_id, 45)}-a{number:03d}-{token}-{unique}"
    run_tag = f"{_slug(config.campaign_id, 28)}_{_slug(run.run_id, 38)}_a{number:03d}_{token}_{unique}"
    attempt_dir = config.campaign_root / "attempts" / run.run_id / attempt_id
    return {
        "attempt_id": attempt_id,
        "attempt_number": number,
        "attempt_dir": attempt_dir,
        "run_tag": run_tag,
        "mode": mode,
    }


def _build_attempt_environment(
    config: CampaignConfig,
    run: RunSpec,
    attempt: Mapping[str, Any],
    frozen_base: Mapping[str, str],
) -> dict[str, str]:
    assert run.registry_path is not None
    environment = dict(frozen_base)
    environment.update(run.environment)
    environment.update(
        {
            "FAIR_RESOLVED_CONFIG_REGISTRY": str(run.registry_path),
            "FAIR_ROUND_DIR": str(attempt["attempt_dir"]),
            "FAIR_ROUND_ROOT": str(config.campaign_root / "attempts"),
            "FAIR_ROUND_LABEL": str(attempt["attempt_id"]),
            "FAIR_ROUND_FORCE": "0",
            "FAIR_ROUND_DRY_RUN": "0",
            "PAPER_QUEUE_DRY_RUN": "0",
            "SLLM_RUN_TAG": str(attempt["run_tag"]),
            "V2_CAMPAIGN_ATTEMPT_ID": str(attempt["attempt_id"]),
            "V2_CAMPAIGN_CONFIG_SHA256": config.config_sha256,
            "V2_CAMPAIGN_RUN_ID": run.run_id,
            "V2_CAMPAIGN_RUN_KEY": run.run_key,
        }
    )
    if config.cleanup.gpu_ids and "FAIR_ROUND_GPU_IDS" not in environment:
        environment["FAIR_ROUND_GPU_IDS"] = ",".join(
            str(item) for item in config.cleanup.gpu_ids
        )
    return environment


def _write_attempt_protocol(
    config: CampaignConfig,
    run: RunSpec,
    attempt: Mapping[str, Any],
    environment: Mapping[str, str],
) -> None:
    attempt_dir = Path(attempt["attempt_dir"])
    attempt_dir.mkdir(parents=True, exist_ok=False)
    env_path = attempt_dir / "orchestrator_frozen_environment.json"
    _atomic_json(env_path, dict(environment))
    metadata = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "campaign_id": config.campaign_id,
        "config_sha256": config.config_sha256,
        "run_id": run.run_id,
        "run_key": run.run_key,
        "attempt_id": attempt["attempt_id"],
        "attempt_number": attempt["attempt_number"],
        "run_tag": attempt["run_tag"],
        "attempt_dir": str(attempt_dir),
        "mode": attempt["mode"],
        "command": list(run.command or ()),
        "working_directory": str(run.working_directory),
        "registry_name": run.registry_name,
        "registry_path": str(run.registry_path),
        "manifest_relative_path": run.manifest_relative_path,
        "environment_path": str(env_path),
        "environment_sha256": _sha256_file(env_path),
        "created_at_utc": _utc_now(),
    }
    _atomic_json(attempt_dir / "orchestrator_attempt.json", metadata)


def _load_attempt_for_resume(
    config: CampaignConfig, run: RunSpec, state: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    attempt_dir = Path(str(state.get("attempt_dir") or ""))
    metadata_path = attempt_dir / "orchestrator_attempt.json"
    env_path = attempt_dir / "orchestrator_frozen_environment.json"
    if not metadata_path.is_file() or not env_path.is_file():
        raise CampaignError(f"attempt protocol files are missing under {attempt_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != ATTEMPT_SCHEMA_VERSION:
        raise CampaignError("attempt metadata schema mismatch")
    for field, expected in (
        ("config_sha256", config.config_sha256),
        ("run_id", run.run_id),
        ("run_key", run.run_key),
        ("attempt_id", state.get("attempt_id")),
    ):
        if metadata.get(field) != expected:
            raise CampaignError(f"resume attempt metadata mismatch for {field}")
    if _sha256_file(env_path) != metadata.get("environment_sha256"):
        raise CampaignError("resume frozen environment failed SHA-256 validation")
    environment_raw = _require_mapping(
        json.loads(env_path.read_text(encoding="utf-8")), "frozen attempt environment"
    )
    environment = {str(key): str(value) for key, value in environment_raw.items()}
    attempt = {
        "attempt_id": metadata["attempt_id"],
        "attempt_number": int(metadata["attempt_number"]),
        "attempt_dir": attempt_dir,
        "run_tag": metadata["run_tag"],
        "mode": "resume",
    }
    return attempt, environment


def _run_hook_commands(
    commands: Sequence[Sequence[str]],
    *,
    environment: Mapping[str, str],
    working_directory: Path,
    log_handle: Any,
    phase: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, command in enumerate(commands):
        started = time.monotonic()
        log_handle.write(f"[orchestrator] {phase} hook {index}: {list(command)!r}\n")
        log_handle.flush()
        completed = subprocess.run(
            list(command),
            cwd=working_directory,
            env=dict(environment),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        result = {
            "phase": phase,
            "index": index,
            "command": list(command),
            "returncode": completed.returncode,
            "elapsed_seconds": time.monotonic() - started,
        }
        results.append(result)
        if completed.returncode != 0:
            raise CampaignError(
                f"{phase} cleanup hook {index} failed with status {completed.returncode}"
            )
    return results


def _gpu_compute_rows(
    gpu_ids: Sequence[int], environment: Mapping[str, str]
) -> list[str]:
    rows: list[str] = []
    for gpu_id in gpu_ids:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-compute-apps=pid,used_gpu_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            env=dict(environment),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise CampaignError(
                f"nvidia-smi cleanup verification failed for GPU {gpu_id}: "
                f"{completed.stderr.strip()}"
            )
        for raw_row in completed.stdout.splitlines():
            row = raw_row.strip()
            if row and "no running processes" not in row.lower():
                rows.append(f"gpu={gpu_id} {row}")
    return rows


def _verify_cleanup(
    config: CampaignConfig,
    *,
    environment: Mapping[str, str],
    phase: str,
    log_handle: Any,
) -> dict[str, Any]:
    if not config.cleanup.strict_gpu_idle:
        result = {"phase": phase, "strict": False, "status": "skipped"}
        log_handle.write(f"[orchestrator] {phase} GPU idle verification skipped\n")
        log_handle.flush()
        return result
    if shutil.which("nvidia-smi", path=environment.get("PATH", os.defpath)) is None:
        raise CampaignError("strict cleanup verification requires nvidia-smi")
    deadline = time.monotonic() + config.cleanup.timeout_seconds
    while True:
        rows = _gpu_compute_rows(config.cleanup.gpu_ids, environment)
        if not rows:
            result = {
                "phase": phase,
                "strict": True,
                "status": "idle",
                "gpu_ids": list(config.cleanup.gpu_ids),
            }
            log_handle.write(
                f"[orchestrator] {phase} GPU idle verified: {list(config.cleanup.gpu_ids)}\n"
            )
            log_handle.flush()
            return result
        if time.monotonic() >= deadline:
            raise CampaignError(
                f"{phase} GPU cleanup verification timed out: " + "; ".join(rows)
            )
        log_handle.write(
            f"[orchestrator] waiting for {phase} GPU idle: {'; '.join(rows)}\n"
        )
        log_handle.flush()
        time.sleep(config.cleanup.poll_seconds)


def _cache_inventory(cache_path: Path) -> dict[str, Any]:
    """Describe a cache tree without following symlinks or hashing model bytes."""
    entries: list[tuple[str, str, int]] = []
    file_count = 0
    directory_count = 1
    symlink_count = 0
    logical_bytes = 0
    allocated_bytes = 0
    for root, directories, filenames in os.walk(cache_path, followlinks=False):
        directories.sort()
        root_path = Path(root)
        for name in directories:
            path = root_path / name
            stat = path.lstat()
            relative = path.relative_to(cache_path).as_posix()
            if path.is_symlink():
                kind = "symlink"
                symlink_count += 1
            elif path.is_dir():
                kind = "directory"
                directory_count += 1
            else:
                raise CampaignError(f"unsupported cache entry type: {path}")
            entries.append((relative, kind, int(stat.st_size)))
        for name in sorted(filenames):
            path = root_path / name
            stat = path.lstat()
            relative = path.relative_to(cache_path).as_posix()
            if path.is_symlink():
                kind = "symlink"
                symlink_count += 1
            elif path.is_file():
                kind = "file"
                file_count += 1
                logical_bytes += int(stat.st_size)
                allocated_bytes += int(stat.st_blocks) * 512
            else:
                raise CampaignError(f"unsupported cache entry type: {path}")
            entries.append((relative, kind, int(stat.st_size)))
    return {
        "file_count": file_count,
        "directory_count": directory_count,
        "symlink_count": symlink_count,
        "logical_bytes": logical_bytes,
        "allocated_bytes": allocated_bytes,
        "inventory_sha256": _sha256_value(entries),
    }


def _prune_completed_cache(
    config: CampaignConfig,
    run: RunSpec,
    attempt: Mapping[str, Any],
    manifest_path: Path,
    invocation_number: int,
) -> tuple[Path, dict[str, Any]]:
    """Remove only an accepted attempt's disposable cache and leave evidence."""
    attempt_dir = Path(attempt["attempt_dir"])
    attempts_root = (config.campaign_root / "attempts").resolve()
    resolved_attempt = attempt_dir.resolve(strict=True)
    if resolved_attempt != attempt_dir or not resolved_attempt.is_relative_to(
        attempts_root
    ):
        raise CampaignError(
            f"refusing cache prune outside canonical attempts root: {attempt_dir}"
        )
    cache_path = attempt_dir / "cache"
    if os.path.lexists(cache_path) and cache_path.is_symlink():
        raise CampaignError(f"refusing to prune symlinked cache root: {cache_path}")

    suffix = f"invocation_{invocation_number:03d}.json"
    intent_path = attempt_dir / f"orchestrator_cache_prune_intent_{suffix}"
    receipt_path = attempt_dir / f"orchestrator_cache_prune_receipt_{suffix}"
    if intent_path.exists() or receipt_path.exists():
        raise CampaignError(
            f"cache-prune evidence already exists for invocation {invocation_number}"
        )

    present = os.path.lexists(cache_path)
    if present and not cache_path.is_dir():
        raise CampaignError(f"cache path is not a directory: {cache_path}")
    inventory = (
        _cache_inventory(cache_path)
        if present
        else {
            "file_count": 0,
            "directory_count": 0,
            "symlink_count": 0,
            "logical_bytes": 0,
            "allocated_bytes": 0,
            "inventory_sha256": _sha256_value([]),
        }
    )
    intent = {
        "schema_version": CACHE_PRUNE_SCHEMA_VERSION,
        "status": "intent",
        "campaign_id": config.campaign_id,
        "run_id": run.run_id,
        "run_key": run.run_key,
        "attempt_id": attempt["attempt_id"],
        "invocation_number": invocation_number,
        "cache_relative_path": "cache",
        "cache_present": present,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "inventory": inventory,
        "created_at_utc": _utc_now(),
    }
    _atomic_json(intent_path, intent)
    if present:
        shutil.rmtree(cache_path)
    if os.path.lexists(cache_path):
        raise CampaignError(f"cache path still exists after prune: {cache_path}")
    receipt = {
        **intent,
        "status": "pruned" if present else "already_absent",
        "intent_path": str(intent_path),
        "intent_sha256": _sha256_file(intent_path),
        "completed_at_utc": _utc_now(),
    }
    _atomic_json(receipt_path, receipt)
    return receipt_path, receipt


def _stream_command(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    working_directory: Path,
    log_handle: Any,
) -> int:
    process = subprocess.Popen(
        list(command),
        cwd=working_directory,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
            log_handle.flush()
        return process.wait()
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        raise
    finally:
        process.stdout.close()


def _validate_attempt_manifest(
    config: CampaignConfig, run: RunSpec, attempt: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(attempt["attempt_dir"]) / run.manifest_relative_path
    if not manifest_path.is_file():
        raise CampaignError(
            f"runner did not publish required manifest: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CampaignError(
            f"runner manifest is invalid JSON: {manifest_path}"
        ) from exc
    if manifest.get("status") != "complete":
        raise CampaignError(
            f"runner manifest is not complete: status={manifest.get('status')!r}"
        )
    if manifest.get("run_tag") not in {None, attempt["run_tag"]}:
        raise CampaignError("runner manifest run_tag does not match frozen attempt")
    if manifest.get("round_dir") not in {None, str(attempt["attempt_dir"])}:
        raise CampaignError("runner manifest round_dir does not match frozen attempt")
    if run.registry_path is not None and manifest.get(
        "system_resolved_config_registry_path"
    ) not in {
        None,
        str(run.registry_path),
    }:
        raise CampaignError("runner manifest resolved-config registry path mismatch")
    if (
        run.environment.get("FAIR_FORMAL_RUN") == "1"
        and manifest.get("formal_run") is not True
    ):
        raise CampaignError("formal run did not publish formal_run=true")
    return manifest_path, manifest


def _execute(
    config: CampaignConfig,
    run: RunSpec,
    attempt: Mapping[str, Any],
    environment: Mapping[str, str],
    *,
    resumed: bool,
) -> bool:
    assert run.command is not None and run.working_directory is not None
    event_name = "attempt_resumed" if resumed else "attempt_started"
    previous_invocations = sum(
        1
        for event in _read_ledger(config)
        if event.get("attempt_id") == attempt["attempt_id"]
        and event.get("event") in {"attempt_started", "attempt_resumed"}
    )
    invocation_number = previous_invocations + 1
    base_event = {
        "event": event_name,
        "run_id": run.run_id,
        "run_key": run.run_key,
        "attempt_id": attempt["attempt_id"],
        "attempt_number": attempt["attempt_number"],
        "attempt_dir": str(attempt["attempt_dir"]),
        "run_tag": attempt["run_tag"],
        "mode": attempt["mode"],
        "invocation_number": invocation_number,
        "executor_pid": os.getpid(),
    }
    _append_event(config, base_event)
    log_path = (
        Path(attempt["attempt_dir"])
        / f"orchestrator_runner_invocation_{invocation_number:03d}.log"
    )
    runner_returncode: int | None = None
    failure: str | None = None
    cleanup_evidence: list[dict[str, Any]] = []
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        try:
            cleanup_evidence.extend(
                _run_hook_commands(
                    config.cleanup.pre_commands,
                    environment=environment,
                    working_directory=run.working_directory,
                    log_handle=log_handle,
                    phase="pre",
                )
            )
            cleanup_evidence.append(
                _verify_cleanup(
                    config,
                    environment=environment,
                    phase="pre",
                    log_handle=log_handle,
                )
            )
            log_handle.write(f"[orchestrator] runner command: {list(run.command)!r}\n")
            runner_returncode = _stream_command(
                run.command,
                environment=environment,
                working_directory=run.working_directory,
                log_handle=log_handle,
            )
            if runner_returncode != 0:
                failure = f"runner exited with status {runner_returncode}"
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                cleanup_evidence.extend(
                    _run_hook_commands(
                        config.cleanup.post_commands,
                        environment=environment,
                        working_directory=run.working_directory,
                        log_handle=log_handle,
                        phase="post",
                    )
                )
                cleanup_evidence.append(
                    _verify_cleanup(
                        config,
                        environment=environment,
                        phase="post",
                        log_handle=log_handle,
                    )
                )
            except BaseException as exc:
                post_failure = f"post-cleanup {type(exc).__name__}: {exc}"
                failure = f"{failure}; {post_failure}" if failure else post_failure
    evidence_path = (
        Path(attempt["attempt_dir"])
        / f"orchestrator_cleanup_evidence_invocation_{invocation_number:03d}.json"
    )
    _atomic_json(evidence_path, cleanup_evidence)
    manifest_path: Path | None = None
    cache_prune_receipt_path: Path | None = None
    cache_prune_receipt: dict[str, Any] | None = None
    if failure is None:
        try:
            manifest_path, _ = _validate_attempt_manifest(config, run, attempt)
        except CampaignError as exc:
            failure = str(exc)
        if (
            failure is None
            and manifest_path is not None
            and config.cleanup.prune_completed_cache
        ):
            try:
                cache_prune_receipt_path, cache_prune_receipt = (
                    _prune_completed_cache(
                        config,
                        run,
                        attempt,
                        manifest_path,
                        invocation_number,
                    )
                )
            except (CampaignError, OSError) as exc:
                failure = f"post-validation cache cleanup failed: {exc}"
    common = {
        "run_id": run.run_id,
        "run_key": run.run_key,
        "attempt_id": attempt["attempt_id"],
        "attempt_number": attempt["attempt_number"],
        "attempt_dir": str(attempt["attempt_dir"]),
        "run_tag": attempt["run_tag"],
        "invocation_number": invocation_number,
        "executor_pid": os.getpid(),
        "runner_returncode": runner_returncode,
        "elapsed_seconds": time.monotonic() - started,
        "log_path": str(log_path),
        "log_sha256": _sha256_file(log_path),
        "cleanup_evidence_path": str(evidence_path),
        "cleanup_evidence_sha256": _sha256_file(evidence_path),
        "cache_prune_enabled": config.cleanup.prune_completed_cache,
        "cache_prune_receipt_path": (
            str(cache_prune_receipt_path)
            if cache_prune_receipt_path is not None
            else None
        ),
        "cache_prune_receipt_sha256": (
            _sha256_file(cache_prune_receipt_path)
            if cache_prune_receipt_path is not None
            else None
        ),
        "cache_prune_status": (
            cache_prune_receipt.get("status")
            if cache_prune_receipt is not None
            else None
        ),
    }
    if failure is not None:
        _append_event(config, {"event": "attempt_failed", "reason": failure, **common})
        print(f"[FAILED] {run.run_id}: {failure}", file=sys.stderr)
        return False
    assert manifest_path is not None
    _append_event(
        config,
        {
            "event": "attempt_completed",
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            **common,
        },
    )
    print(f"[COMPLETE] {run.run_id} -> {manifest_path}")
    return True


def _select_run(config: CampaignConfig, requested: str | None = None) -> RunSpec:
    status = campaign_status(config)
    if not status["config_matches_initialized_campaign"]:
        raise CampaignError("campaign config hash differs from initialized campaign")
    states = status["runs"]
    unresolved = [
        run_id
        for run_id, state in states.items()
        if state["status"]
        in {"failed", "interrupted", "invalid", "reuse_failed", "reuse_invalid"}
    ]
    if unresolved:
        raise CampaignError(
            "failure-stop is active; resolve with resume or retry before run-next: "
            + ", ".join(unresolved)
        )
    if requested:
        try:
            run = config.run_by_id[requested]
        except KeyError as exc:
            raise CampaignError(f"unknown run id: {requested}") from exc
        if states[requested]["status"] != "pending":
            raise CampaignError(
                f"requested run {requested} is not pending: {states[requested]['status']}"
            )
        if run.reuse_of:
            raise CampaignError("reuse aliases are never executed")
        if any(
            states[dependency]["status"] != "complete" for dependency in run.depends_on
        ):
            raise CampaignError(
                f"requested run {requested} has incomplete dependencies"
            )
        return run
    order = _topological_order(config.runs)
    for run_id in order:
        run = config.run_by_id[run_id]
        if run.reuse_of:
            continue
        if states[run_id]["status"] == "pending" and all(
            states[dependency]["status"] == "complete" for dependency in run.depends_on
        ):
            return run
    raise CampaignError("no runnable pending run remains")


def _prepare_new_execution(
    config: CampaignConfig, run: RunSpec, mode: str
) -> tuple[dict[str, Any], dict[str, str]]:
    frozen_base = _ensure_initialized(config)
    _verify_frozen_sources(config)
    attempt = _new_attempt(config, run, mode)
    environment = _build_attempt_environment(config, run, attempt, frozen_base)
    _write_attempt_protocol(config, run, attempt, environment)
    return attempt, environment


def command_run_next(config: CampaignConfig, run_id: str | None) -> int:
    with _ExecutionLock(config):
        _ensure_initialized(config)
        _verify_frozen_sources(config)
        run = _select_run(config, run_id)
        attempt, environment = _prepare_new_execution(config, run, "run-next")
        return 0 if _execute(config, run, attempt, environment, resumed=False) else 1


def _resume_or_retry(config: CampaignConfig, run_id: str | None, *, retry: bool) -> int:
    with _ExecutionLock(config):
        _ensure_initialized(config)
        _verify_frozen_sources(config)
        status = campaign_status(config)
        candidates = [
            candidate_id
            for candidate_id, state in status["runs"].items()
            if state["status"] in {"failed", "interrupted", "invalid"}
            and not config.run_by_id[candidate_id].reuse_of
        ]
        selected_id = run_id or (candidates[0] if len(candidates) == 1 else None)
        if selected_id is None:
            raise CampaignError(
                "specify --run-id; recoverable candidates=" + ",".join(candidates)
            )
        if selected_id not in config.run_by_id:
            raise CampaignError(f"unknown run id: {selected_id}")
        run = config.run_by_id[selected_id]
        state = status["runs"][selected_id]
        if run.reuse_of:
            raise CampaignError("reuse aliases cannot be resumed or retried")
        if state["status"] not in {"failed", "interrupted", "invalid"}:
            raise CampaignError(
                f"run {selected_id} is not recoverable: {state['status']}"
            )
        if retry:
            attempt, environment = _prepare_new_execution(config, run, "retry")
            resumed = False
        else:
            attempt, environment = _load_attempt_for_resume(config, run, state)
            resumed = True
        return 0 if _execute(config, run, attempt, environment, resumed=resumed) else 1


def _source_real_run(config: CampaignConfig, run_id: str) -> RunSpec:
    run = config.run_by_id[run_id]
    seen: set[str] = set()
    while run.reuse_of:
        if run.run_id in seen:
            raise CampaignError(f"reuse cycle while resolving {run_id}")
        seen.add(run.run_id)
        run = config.run_by_id[run.reuse_of]
    return run


def command_export_inputs(config: CampaignConfig, export_ids: Sequence[str]) -> int:
    with _ExecutionLock(config):
        _ensure_initialized(config)
        _verify_frozen_sources(config)
        status = campaign_status(config)
        selected = [
            export
            for export in config.exports
            if not export_ids or export.export_id in set(export_ids)
        ]
        missing_ids = sorted(set(export_ids) - {item.export_id for item in selected})
        if missing_ids:
            raise CampaignError(f"unknown analyzer exports: {missing_ids}")
        if not selected:
            raise CampaignError("no analyzer exports are configured")
        for export in selected:
            records: list[dict[str, Any]] = []
            lists: dict[str, list[str]] = {
                selector.name: [] for selector in export.selectors
            }
            for logical_run_id in export.run_ids:
                logical_state = status["runs"][logical_run_id]
                if logical_state["status"] != "complete":
                    raise CampaignError(
                        f"export {export.export_id} requires complete run {logical_run_id}; "
                        f"status={logical_state['status']}"
                    )
                source_run = _source_real_run(config, logical_run_id)
                source_state = status["runs"][source_run.run_id]
                attempt_dir = Path(str(source_state["attempt_dir"]))
                input_record: dict[str, list[str]] = {}
                for selector in export.selectors:
                    matches = sorted(
                        path.resolve()
                        for path in attempt_dir.glob(selector.pattern)
                        if path.is_file()
                    )
                    for match in matches:
                        try:
                            match.relative_to(attempt_dir.resolve())
                        except ValueError as exc:
                            raise CampaignError(
                                f"selector escaped attempt directory: {selector.pattern}"
                            ) from exc
                    if selector.required and not matches:
                        raise CampaignError(
                            f"export {export.export_id} selector {selector.name} matched no "
                            f"files for {logical_run_id}: {selector.pattern}"
                        )
                    if not selector.multiple and len(matches) > 1:
                        raise CampaignError(
                            f"export {export.export_id} selector {selector.name} matched "
                            f"{len(matches)} files but multiple=false"
                        )
                    paths = [str(path) for path in matches]
                    input_record[selector.name] = paths
                    lists[selector.name].extend(paths)
                records.append(
                    {
                        "logical_run_id": logical_run_id,
                        "source_run_id": source_run.run_id,
                        "run_key": source_run.run_key,
                        "attempt_id": source_state["attempt_id"],
                        "attempt_dir": str(attempt_dir),
                        "inputs": input_record,
                    }
                )
            payload = {
                "schema_version": EXPORT_SCHEMA_VERSION,
                "campaign_id": config.campaign_id,
                "config_sha256": config.config_sha256,
                "export_id": export.export_id,
                "generated_at_utc": _utc_now(),
                "records": records,
            }
            export_attempt_id = f"{_timestamp_token()}-{uuid.uuid4().hex[:10]}"
            invocation_directory = export.output_directory / export_attempt_id
            invocation_directory.mkdir(parents=True, exist_ok=False)
            json_path = invocation_directory / "inputs.json"
            _atomic_json(json_path, payload)
            list_paths: dict[str, str] = {}
            for name, paths in lists.items():
                list_path = invocation_directory / f"{name}.txt"
                _atomic_write(
                    list_path, ("\n".join(paths) + ("\n" if paths else "")).encode()
                )
                list_paths[name] = str(list_path)
            _append_event(
                config,
                {
                    "event": "analyzer_inputs_exported",
                    "export_id": export.export_id,
                    "export_attempt_id": export_attempt_id,
                    "inputs_path": str(json_path),
                    "inputs_sha256": _sha256_file(json_path),
                    "list_paths": list_paths,
                },
            )
            print(f"[EXPORTED] {export.export_id} -> {json_path}")
    return 0


def plan_payload(config: CampaignConfig) -> dict[str, Any]:
    status = campaign_status(config)
    source_preview = [_source_identity(source) for source in config.source_repositories]
    return {
        "schema_version": SCHEMA_VERSION,
        "read_only": True,
        "runner_dry_run_invoked": False,
        "campaign_id": config.campaign_id,
        "campaign_root": str(config.campaign_root),
        "config_path": str(config.path),
        "config_sha256": config.config_sha256,
        "registries": {name: str(path) for name, path in config.registries.items()},
        "source_preview": source_preview,
        "cleanup": {
            "strict_gpu_idle": config.cleanup.strict_gpu_idle,
            "gpu_ids": list(config.cleanup.gpu_ids),
            "timeout_seconds": config.cleanup.timeout_seconds,
            "prune_completed_cache": config.cleanup.prune_completed_cache,
            "pre_hook_count": len(config.cleanup.pre_commands),
            "post_hook_count": len(config.cleanup.post_commands),
        },
        "runs": [
            {
                "id": run.run_id,
                "run_key": run.run_key,
                "depends_on": list(run.depends_on),
                "reuse_of": run.reuse_of,
                "registry": run.registry_name,
                "status": status["runs"][run.run_id]["status"],
                "command": list(run.command) if run.command else None,
            }
            for run in config.runs
        ],
        "topological_order": _topological_order(config.runs),
        "analyzer_exports": [
            {
                "id": item.export_id,
                "run_ids": list(item.run_ids),
                "output_directory": str(item.output_directory),
            }
            for item in config.exports
        ],
    }


def _print_status(payload: Mapping[str, Any]) -> None:
    print(
        f"campaign={payload['campaign_id']} initialized={payload['initialized']} "
        f"config_match={payload['config_matches_initialized_campaign']}"
    )
    for run_id, state in payload["runs"].items():
        suffix = (
            f" attempt={state.get('attempt_id')}" if state.get("attempt_id") else ""
        )
        reuse = f" reuse_of={state.get('reuse_of')}" if state.get("reuse_of") else ""
        print(f"{run_id:40s} {state['status']:18s}{suffix}{reuse}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="read-only validation and execution plan")
    plan.add_argument("--json", action="store_true")
    status = subparsers.add_parser("status", help="read-only ledger/result status")
    status.add_argument("--json", action="store_true")
    run_next = subparsers.add_parser("run-next", help="execute the next runnable run")
    run_next.add_argument("--run-id")
    resume = subparsers.add_parser(
        "resume", help="resume the latest failed/interrupted attempt"
    )
    resume.add_argument("--run-id")
    retry = subparsers.add_parser(
        "retry", help="retry a failed/interrupted run in a fresh attempt"
    )
    retry.add_argument("--run-id")
    export = subparsers.add_parser(
        "export-inputs", help="export strict analyzer input JSON and path lists"
    )
    export.add_argument("--export", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "plan":
            payload = plan_payload(config)
            if args.json:
                print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
            else:
                print(
                    f"campaign={config.campaign_id} root={config.campaign_root} "
                    f"config_sha256={config.config_sha256}"
                )
                print("read_only=true runner_dry_run_invoked=false")
                for run in payload["runs"]:
                    print(
                        f"{run['id']:40s} {run['status']:18s} "
                        f"key={run['run_key'][:12]} depends={run['depends_on']} "
                        f"reuse_of={run['reuse_of']}"
                    )
            return 0
        if args.command == "status":
            payload = campaign_status(config)
            if args.json:
                print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
            else:
                _print_status(payload)
            return 0
        if args.command == "run-next":
            return command_run_next(config, args.run_id)
        if args.command == "resume":
            return _resume_or_retry(config, args.run_id, retry=False)
        if args.command == "retry":
            return _resume_or_retry(config, args.run_id, retry=True)
        if args.command == "export-inputs":
            return command_export_inputs(config, args.export)
        raise AssertionError(args.command)
    except CampaignError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
