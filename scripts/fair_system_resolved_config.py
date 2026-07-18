#!/usr/bin/env python3
"""Resolve and freeze the V2 Prime/S-LoRA/ServerlessLLM configuration.

The hash produced here is deliberately a *configuration* hash.  Request-trace
identity belongs to the round provenance and therefore is not part of the
hashed payload.  In particular, sampling seed, trace/subset paths or hashes,
execution order, and run tags are recorded only in the unhashed audit section.

A held-out launch is accepted only after a matching seed-41 formal validation
has completed and its manifest/sidecar bytes have been promoted.  The first
held-out seed freezes that successful hash; later held-out seeds must match.
Registry updates are locked and atomic so concurrent launchers cannot silently
establish different configurations.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import yaml


SCHEMA_VERSION = "eurosys27_v2_system_resolved_config_v1"
REGISTRY_SCHEMA_VERSION = "eurosys27_v2_system_resolved_config_registry_v1"
STRICT_V2_SCENARIOS = {
    "v2_elastic_only",
    "v2_hit_aware_preparation",
    "v2_hierarchical_no_coord",
    "v2_full",
}
SEED_TRACE_ROLES = {
    41: "validation",
    42: "smoke",
    43: "heldout",
    44: "heldout",
    45: "heldout",
}
V2_CAMPAIGN_KINDS = {
    "v2_full_vs_serverless",
    "v2_c5_matched_output",
}
FORBIDDEN_HASH_KEYS = {
    "seed",
    "sampling_seed",
    "generation_seed",
    "workload_seed",
    "artifact_pool_seed",
    "total_requests",
    "num_requests",
    "bandwidth_mbps",
    "bandwidth_mib_s",
    "storage_bandwidth_mbps",
    "storage_bandwidth_mib_s",
    "zipf_exponent",
    "active_adapter_cap",
    "hotset_rotation_requests",
    "hotset_rotation_mode",
    "hotset_overlap_fraction",
    "time_scale_factor",
    "trace",
    "trace_path",
    "trace_sha256",
    "shared_trace_path",
    "shared_trace_sha256",
    "subset",
    "subset_path",
    "subset_sha256",
    "adapter_subset_path",
    "adapter_subset_sha256",
    "shared_adapter_subset_path",
    "shared_adapter_subset_sha256",
    "execution_order",
    "run_tag",
    "results_tag",
}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _git_output(path: Path, arguments: Sequence[str]) -> bytes:
    path = path.resolve()
    try:
        return subprocess.check_output(
            [
                "git",
                "-c",
                "color.ui=false",
                "-c",
                "core.quotePath=true",
                "-C",
                str(path),
                *arguments,
            ],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"git command failed for worktree identity: repo={path} args={arguments}"
        ) from exc


def _nul_git_paths(data: bytes) -> list[str]:
    return sorted(
        {
            item.decode("utf-8", errors="surrogateescape")
            for item in data.split(b"\0")
            if item
        }
    )


def _dirty_file_content_identity(repo: Path, relative_path: str) -> Dict[str, Any]:
    candidate = repo / relative_path
    if not candidate.exists() and not candidate.is_symlink():
        return {"kind": "deleted", "bytes": 0, "content_sha256": None}
    if candidate.is_symlink():
        content = os.fsencode(os.readlink(candidate))
        return {
            "kind": "symlink",
            "bytes": len(content),
            "content_sha256": hashlib.sha256(content).hexdigest(),
        }
    if candidate.is_file():
        digest = hashlib.sha256()
        size = 0
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
        return {
            "kind": "regular",
            "bytes": size,
            "content_sha256": digest.hexdigest(),
        }
    if candidate.is_dir() and (candidate / ".git").exists():
        nested_payload = {
            "commit": _git_commit(candidate),
            "head_to_worktree_binary_diff_sha256": hashlib.sha256(
                _git_output(
                    candidate,
                    [
                        "diff",
                        "--no-color",
                        "--binary",
                        "--full-index",
                        "--no-ext-diff",
                        "--no-textconv",
                        "--no-renames",
                        "HEAD",
                        "--",
                    ],
                )
            ).hexdigest(),
        }
        content = _canonical_bytes(nested_payload)
        return {
            "kind": "git_directory",
            "bytes": len(content),
            "content_sha256": hashlib.sha256(content).hexdigest(),
            "git_identity": nested_payload,
        }
    raise RuntimeError(
        f"unsupported tracked worktree entry while hashing {repo}: {relative_path}"
    )


def _git_worktree_identity(repo: Path) -> Dict[str, Any]:
    """Bind a nested upstream checkout to its tracked local patch exactly.

    Untracked files are intentionally outside this identity.  Runtime patches in
    the tracked worktree are represented by the canonical HEAD-to-worktree
    binary diff and raw SHA-256 for every dirty tracked file.  Staged and
    unstaged diff hashes are retained separately for auditability.
    """

    repo = repo.resolve()
    commit = _git_commit(repo)
    if not commit:
        raise RuntimeError(f"cannot resolve git HEAD for nested repository: {repo}")

    unmerged = _git_output(repo, ["ls-files", "--unmerged", "-z"])
    if unmerged:
        entries = []
        for record in unmerged.split(b"\0"):
            if not record:
                continue
            path = record.split(b"\t", 1)[-1]
            entries.append(path.decode("utf-8", errors="surrogateescape"))
        raise RuntimeError(
            f"nested repository has unmerged tracked entries: repo={repo} "
            f"paths={sorted(set(entries))}"
        )

    common_diff_args = [
        "--no-color",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
    ]
    head_to_worktree = _git_output(
        repo, ["diff", *common_diff_args, "HEAD", "--"]
    )
    staged = _git_output(
        repo, ["diff", "--cached", *common_diff_args, "HEAD", "--"]
    )
    unstaged = _git_output(repo, ["diff", *common_diff_args, "--"])

    path_sets = (
        _nul_git_paths(
            _git_output(repo, ["diff", "--no-renames", "--name-only", "-z", "HEAD", "--"])
        ),
        _nul_git_paths(
            _git_output(
                repo,
                ["diff", "--cached", "--no-renames", "--name-only", "-z", "HEAD", "--"],
            )
        ),
        _nul_git_paths(
            _git_output(repo, ["diff", "--no-renames", "--name-only", "-z", "--"])
        ),
    )
    dirty_paths = sorted({path for paths in path_sets for path in paths})
    dirty_files = {
        path: _dirty_file_content_identity(repo, path) for path in dirty_paths
    }
    return {
        "commit": commit,
        "tracked_dirty_paths": dirty_paths,
        "has_tracked_patch": bool(dirty_paths),
        "head_to_worktree_binary_diff_sha256": hashlib.sha256(
            head_to_worktree
        ).hexdigest(),
        "head_to_worktree_binary_diff_bytes": len(head_to_worktree),
        "staged_binary_diff_sha256": hashlib.sha256(staged).hexdigest(),
        "staged_binary_diff_bytes": len(staged),
        "unstaged_binary_diff_sha256": hashlib.sha256(unstaged).hexdigest(),
        "unstaged_binary_diff_bytes": len(unstaged),
        "dirty_tracked_files": dirty_files,
        "unmerged_entries": [],
        "untracked_files_included": False,
    }


def _tracked_dirty_paths(path: Path) -> list[str]:
    """Return tracked changes only; deliberately ignore cache/untracked files."""
    path = path.resolve()
    changed: set[str] = set()
    for command in (
        ["git", "-C", str(path), "diff", "--name-only", "HEAD", "--"],
        ["git", "-C", str(path), "diff", "--cached", "--name-only", "HEAD", "--"],
    ):
        try:
            output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"cannot inspect tracked source state: {path}") from exc
        changed.update(line.strip() for line in output.splitlines() if line.strip())
    return sorted(changed)


def source_cleanliness(baselines_root: Path, main_repo: Path) -> Dict[str, Any]:
    baseline_dirty = _tracked_dirty_paths(baselines_root)
    faaslora_dirty = _tracked_dirty_paths(main_repo)
    faaslora_allowed = {"configs/generated/lora_manifest_1000.json"}
    faaslora_disallowed = sorted(set(faaslora_dirty) - faaslora_allowed)
    return {
        "baseline_tracked_dirty_paths": baseline_dirty,
        "baseline_tracked_clean": not baseline_dirty,
        "faaslora_tracked_dirty_paths": faaslora_dirty,
        "faaslora_allowed_tracked_dirty_paths": sorted(faaslora_allowed),
        "faaslora_disallowed_tracked_dirty_paths": faaslora_disallowed,
        "faaslora_tracked_clean_for_formal": not faaslora_disallowed,
        "source_clean_for_formal": not baseline_dirty and not faaslora_disallowed,
        "untracked_files_checked": False,
    }


def require_formal_source_cleanliness(baselines_root: Path, main_repo: Path) -> Dict[str, Any]:
    status = source_cleanliness(baselines_root, main_repo)
    if not status["source_clean_for_formal"]:
        raise ValueError(
            "formal source gate failed: "
            f"baseline_tracked_dirty={status['baseline_tracked_dirty_paths']} "
            f"faaslora_disallowed_tracked_dirty={status['faaslora_disallowed_tracked_dirty_paths']}"
        )
    return status


def _source_identity(paths: Iterable[Path]) -> Dict[str, str]:
    identity: Dict[str, str] = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"configuration source does not exist: {resolved}")
        identity[resolved.name] = _file_sha256(resolved)
    return identity


def _normalize_for_hash(value: Any) -> Any:
    """Return JSON-safe data while rejecting forbidden provenance fields."""
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if key.lower() in FORBIDDEN_HASH_KEYS:
                continue
            normalized[key] = _normalize_for_hash(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _load_config(config_path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"experiment config must contain a mapping: {config_path}")
    return data


def _profile_overlay(cfg: Mapping[str, Any], bucket: str, name: str) -> Dict[str, Any]:
    profiles = cfg.get(bucket, {}) or {}
    if not isinstance(profiles, Mapping) or name not in profiles:
        raise KeyError(f"unknown profile {name!r} in {bucket}")
    profile = profiles[name] or {}
    if not isinstance(profile, Mapping):
        raise ValueError(f"profile {bucket}.{name} must be a mapping")
    return dict(profile)


def _scenario_config(cfg: Mapping[str, Any], scenario: str) -> Dict[str, Any]:
    buckets: Sequence[str]
    if scenario.startswith("v2_"):
        buckets = ("revision_v2_scenarios",)
    else:
        buckets = ("scenarios", "revision_v2_scenarios")
    for bucket in buckets:
        entries = cfg.get(bucket, []) or []
        if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
            for entry in entries:
                if isinstance(entry, Mapping) and str(entry.get("name") or "") == scenario:
                    return dict(entry)
    raise KeyError(f"unknown scenario {scenario!r}")


def resolve_profiles(
    cfg: Mapping[str, Any],
    *,
    model_profile: str,
    dataset_profile: str,
    workload_profile: str,
    scenario: str,
) -> Dict[str, Any]:
    model_overlay = _profile_overlay(cfg, "model_profiles", model_profile)
    dataset_overlay = _profile_overlay(cfg, "dataset_profiles", dataset_profile)
    workload_overlay = _profile_overlay(cfg, "workload_profiles", workload_profile)
    scenario_overlay = _scenario_config(cfg, scenario)

    section_names = (
        "experiment",
        "hardware",
        "model",
        "lora_adapters",
        "datasets",
        "workload",
        "resource_coordination",
        "storage",
        "cost_model",
    )
    resolved: Dict[str, Any] = {
        name: dict(cfg.get(name, {}) or {})
        for name in section_names
        if isinstance(cfg.get(name, {}) or {}, Mapping)
    }
    for overlay in (model_overlay, dataset_overlay, workload_overlay, scenario_overlay):
        for section, value in overlay.items():
            if section in {"name", "description", "baseline_type"}:
                continue
            if isinstance(value, Mapping) and isinstance(resolved.get(section), Mapping):
                resolved[section] = _deep_merge(resolved[section], value)  # type: ignore[arg-type]
            elif section in section_names:
                resolved[section] = _normalize_for_hash(value)
    resolved["scenario"] = _normalize_for_hash(scenario_overlay)
    return _normalize_for_hash(resolved)


def _env_subset(
    env: Mapping[str, str],
    prefixes: Sequence[str],
    excluded: Iterable[str],
) -> Dict[str, str]:
    excluded_set = set(excluded)
    selected: Dict[str, str] = {}
    for key, value in env.items():
        if not any(key.startswith(prefix) for prefix in prefixes):
            continue
        if key in excluded_set or "SEED" in key:
            continue
        selected[key] = str(value)
    return dict(sorted(selected.items()))


COMMON_ENV_EXCLUSIONS = {
    "SLLM_RUN_TAG",
    "SLLM_SHARED_TRACE_PATH",
    "SLLM_SHARED_ADAPTER_SUBSET_PATH",
    "SLLM_RESULT_DIR",
    "SLLM_LOG_DIR",
    "SLLM_SHARED_INPUT_DIR",
    "SLLM_SHARED_ROUND_DIR",
    "SLLM_SAMPLING_SEED",
    "SLLM_GENERATION_SEED",
    "SLLM_TOTAL_REQUESTS",
    "SLLM_MODEL_PROFILE",
    "SLLM_DATASET_PROFILE",
    "SLLM_WORKLOAD_PROFILE",
    "SLLM_SELECTED_NUM_ADAPTERS",
    "SLLM_TIME_SCALE_FACTOR",
    "SLLM_ZIPF_EXPONENT",
    "SLLM_ACTIVE_ADAPTER_CAP",
    "SLLM_HOTSET_ROTATION_REQUESTS",
    "SLLM_HOTSET_ROTATION_MODE",
    "SLLM_HOTSET_OVERLAP_FRACTION",
    "SLLM_SLEEP_SCALE",
    "SLLM_REPLAY_MAX_REQUESTS",
    "FAIR_ROUND_DIR",
    "FAIR_ROUND_ROOT",
    "FAIR_ROUND_LABEL",
    "FAIR_ROUND_TIMESTAMP",
    "FAIR_ROUND_EXECUTION_ORDER",
    "FAIR_TRACE_ROLE",
    "FAIR_RESOLVED_CONFIG_REGISTRY",
    "FAIR_RESOLVED_CONFIG_FAMILY",
}
PRIME_ENV_EXCLUSIONS = COMMON_ENV_EXCLUSIONS | {
    "FAASLORA_SHARED_TRACE_PATH",
    "FAASLORA_SHARED_ADAPTER_SUBSET_PATH",
    "FAASLORA_RESULTS_TAG",
    "FAASLORA_RUN_FROZEN_SETTINGS_SHA256",
    "FAASLORA_SYSTEM_RESOLVED_CONFIG_SHA256",
    "FAASLORA_TRACE_ROLE",
    "FAASLORA_FORMAL_RUN",
    "FAASLORA_NVME_CACHE_DIR",
    "FAASLORA_HOST_CACHE_DIR",
    "FAASLORA_WORKER_LOG_ARCHIVE_DIR",
    "FAASLORA_PROFILE_MODEL",
    "FAASLORA_PROFILE_DATASET",
    "FAASLORA_PROFILE_WORKLOAD",
    "FAASLORA_SCENARIO",
    "FAASLORA_TOTAL_REQUESTS",
    "FAASLORA_TIME_SCALE_FACTOR",
    "FAASLORA_STORAGE_BANDWIDTH_MBPS",
    "FAASLORA_STORAGE_BANDWIDTH_MIB_S",
    "FAASLORA_ZIPF_EXPONENT",
    "FAASLORA_ACTIVE_ADAPTER_CAP",
    "FAASLORA_HOTSET_ROTATION_REQUESTS",
    "FAASLORA_HOTSET_ROTATION_MODE",
    "FAASLORA_HOTSET_OVERLAP_FRACTION",
}
SLORA_ENV_EXCLUSIONS = COMMON_ENV_EXCLUSIONS | {
    "SLORA_RESULT_TAG",
    "SLORA_REMOTE_ARTIFACT_STAGE_ENDPOINT",
    "SLORA_REMOTE_ARTIFACT_STAGE_CACHE_DIR",
    "SLORA_HOST",
    "SLORA_PORT",
    "SLORA_PORT_STRIDE",
    "SLORA_NCCL_PORT",
    "SLORA_SLEEP_SCALE",
}
SERVERLESS_ENV_EXCLUSIONS = COMMON_ENV_EXCLUSIONS | {
    "SLLM_REMOTE_ARTIFACT_STAGE_ENDPOINT",
    "SLLM_REMOTE_ARTIFACT_STAGE_CACHE_DIR",
    "SLLM_REMOTE_ARTIFACT_STAGE_BANDWIDTH_MBPS",
    "SLLM_REMOTE_ARTIFACT_STAGE_BANDWIDTH_MIB_S",
    "BASELINE_REMOTE_ARTIFACT_BANDWIDTH_MBPS",
    "BASELINE_REMOTE_ARTIFACT_BANDWIDTH_MIB_S",
    "BASELINE_REMOTE_ARTIFACT_STAGE_ENDPOINT",
    "BASELINE_REMOTE_ARTIFACT_STAGE_CACHE_DIR",
    "SLLM_REQUEST_REMOTE_ADAPTER_MAP",
    "SLLM_RESULT_TAG",
    "SLLM_STACK_SUFFIX",
    "SLLM_HEAD_SESSION",
    "SLLM_STORE_SESSION",
    "SLLM_SERVE_SESSION",
    "SLLM_WORKER_SESSION_PREFIX",
    "SLLM_SERVE_LOG_PATH",
}


def _int_env(env: Mapping[str, str], key: str, default: int) -> int:
    raw = str(env.get(key, "") or "").strip()
    return int(raw) if raw else int(default)


def _float_env(env: Mapping[str, str], key: str, default: float) -> float:
    raw = str(env.get(key, "") or "").strip()
    return float(raw) if raw else float(default)


def _coerce_axis_value(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _resolved_slora(
    profiles: Mapping[str, Any],
    env: Mapping[str, str],
    gpu_ids: Sequence[str],
    generation_contract: str,
    fixed_output_max_tokens: int,
    fixed_prompt_max_tokens: int,
    selected_num_adapters: int,
) -> Dict[str, Any]:
    model = dict(profiles.get("model", {}) or {})
    tp = _int_env(env, "SLORA_TENSOR_PARALLEL_SIZE", int(model.get("tensor_parallel_size", 1) or 1))
    dp_default = max(1, len(gpu_ids) // max(1, tp))
    dp = _int_env(env, "SLORA_DATA_PARALLEL_REPLICAS", dp_default)
    max_model_len = int(model.get("max_model_len", 1024) or 1024)
    output_cap = int(model.get("max_output_tokens_cap", 0) or 0)
    reserve = max(32, output_cap if output_cap > 0 else 32)
    input_default = int(model.get("max_input_len") or max(32, max_model_len - reserve - 8))
    total_default = max(max_model_len, input_default + max(1, reserve))
    max_input = _int_env(env, "SLORA_MAX_REQ_INPUT_LEN", input_default)
    max_total = _int_env(env, "SLORA_MAX_REQ_TOTAL_LEN", total_default)
    return {
        "topology": {"data_parallel_replicas": dp, "tensor_parallel_size": tp, "gpu_ids": list(gpu_ids)},
        "serving": {
            "max_total_token_num": _int_env(env, "SLORA_MAX_TOTAL_TOKEN_NUM", 14000),
            "max_req_input_len": max_input,
            "max_req_total_len": max_total,
            "batch_max_tokens": _int_env(env, "SLORA_BATCH_MAX_TOKENS", max_total),
            "use_bmm": str(env.get("SLORA_USE_BMM", "auto") or "auto"),
        },
        "generation": {
            "contract": generation_contract,
            "fixed_output_max_tokens": fixed_output_max_tokens,
            "fixed_prompt_max_tokens": fixed_prompt_max_tokens,
        },
        "selected_num_adapters": selected_num_adapters,
        "environment_overrides": _env_subset(env, ("SLORA_",), SLORA_ENV_EXCLUSIONS),
    }


def _resolved_serverless(
    profiles: Mapping[str, Any],
    env: Mapping[str, str],
    gpu_ids: Sequence[str],
    storage_bandwidth_mib_s: float,
    selected_num_adapters: int,
) -> Dict[str, Any]:
    model = dict(profiles.get("model", {}) or {})
    coordination = dict(profiles.get("resource_coordination", {}) or {})
    tp = int(model.get("tensor_parallel_size", 1) or 1)
    realizable_max = max(1, len(gpu_ids) // max(1, tp))
    requested_max = _int_env(env, "SLLM_DEPLOY_MAX_INSTANCES", int(coordination.get("max_instances", 1) or 1))
    backend = str(env.get("SLLM_BACKEND", "vllm") or "vllm").strip().lower()
    if backend == "auto":
        backend = "vllm"
    return {
        "backend": backend,
        "topology": {"tensor_parallel_size": tp, "gpu_ids": list(gpu_ids)},
        "autoscaling": {
            "metric": "concurrency",
            "target": _int_env(env, "SLLM_DEPLOY_TARGET", int(model.get("runtime_concurrency_cap", 1) or 1)),
            "min_instances": _int_env(env, "SLLM_DEPLOY_MIN_INSTANCES", int(coordination.get("min_instances", 0) or 0)),
            "max_instances": min(requested_max, realizable_max),
            "requested_max_instances": requested_max,
            "keep_alive_s": _int_env(env, "SLLM_DEPLOY_KEEP_ALIVE", int(coordination.get("idle_timeout_s", 0) or 0)),
        },
        "backend_config": {
            "model": model.get("name"),
            "dtype": model.get("dtype", "float16"),
            "max_model_len": int(model.get("max_model_len", 0) or 0),
            "max_input_len": int(model.get("max_input_len", 0) or 0),
            "max_output_tokens_cap": int(model.get("max_output_tokens_cap", 0) or 0),
            "gpu_memory_utilization": float(model.get("gpu_memory_utilization", 0.85) or 0.85),
            "max_num_seqs": int(model.get("max_num_seqs", 0) or 0),
            "max_num_batched_tokens": int(model.get("max_num_batched_tokens", 0) or 0),
            "max_loras": int(model.get("max_loras", 1) or 1),
            "max_lora_rank": int(model.get("max_lora_rank", 16) or 16),
            "enable_chunked_prefill": bool(model.get("enable_chunked_prefill", False)),
            "enable_prefix_caching": bool(model.get("enable_prefix_caching", False)),
            "enforce_eager": model.get("enforce_eager", True),
            "vllm_use_v1": model.get("vllm_use_v1"),
            "vllm_attention_backend": model.get("vllm_attention_backend"),
            "vllm_use_flashinfer_sampler": model.get("vllm_use_flashinfer_sampler"),
            "lora_runtime": "vllm_request",
        },
        "remote_artifact": {
            "mode": "dynamic",
            "workers": _int_env(env, "SLLM_REMOTE_ARTIFACT_STAGE_WORKERS", 1),
            "cold_round_cache": True,
        },
        "replay": {
            "post_deploy_wait_s": _float_env(env, "SLLM_POST_DEPLOY_WAIT_S", 0.0),
            "empty_success_retries": _int_env(env, "SLLM_EMPTY_SUCCESS_RETRIES", 2),
            "empty_success_retry_delay_s": _float_env(env, "SLLM_EMPTY_SUCCESS_RETRY_DELAY_S", 1.0),
        },
        "selected_num_adapters": selected_num_adapters,
        "environment_overrides": _env_subset(
            env,
            ("SLLM_", "VLLM_", "BASELINE_REMOTE_ARTIFACT_"),
            SERVERLESS_ENV_EXCLUSIONS,
        ),
    }


def build_hashed_config(
    *,
    baselines_root: Path,
    main_repo: Path,
    model_profile: str,
    dataset_profile: str,
    workload_profile: str,
    total_requests: int,
    selected_num_adapters: int,
    generation_contract: str,
    fixed_output_max_tokens: int,
    fixed_prompt_max_tokens: int,
    storage_bandwidth_mib_s: float,
    zipf_exponent: str,
    active_adapter_cap: str,
    hotset_rotation_requests: str,
    hotset_rotation_mode: str,
    hotset_overlap_fraction: str,
    faaslora_scenario: str,
    gpu_ids: str,
    env: Mapping[str, str],
) -> Dict[str, Any]:
    baselines_root = baselines_root.resolve()
    main_repo = main_repo.resolve()
    config_path = main_repo / "configs" / "experiments.yaml"
    cfg = _load_config(config_path)
    profiles = resolve_profiles(
        cfg,
        model_profile=model_profile,
        dataset_profile=dataset_profile,
        workload_profile=workload_profile,
        scenario=faaslora_scenario,
    )
    gpu_list = [item.strip() for item in gpu_ids.split(",") if item.strip()]
    if not gpu_list:
        raise ValueError("at least one GPU id is required")

    shared = {
        "model_profile": model_profile,
        "selected_num_adapters": int(selected_num_adapters),
        "generation_contract": generation_contract,
        "fixed_output_max_tokens": int(fixed_output_max_tokens),
        "fixed_prompt_max_tokens": int(fixed_prompt_max_tokens),
        "gpu_ids": gpu_list,
    }
    system_profiles = {
        key: value
        for key, value in profiles.items()
        if key not in {"datasets", "workload"}
    }
    prime = {
        "scenario": faaslora_scenario,
        "resolved_profiles": system_profiles,
        "environment_overrides": _env_subset(env, ("FAASLORA_",), PRIME_ENV_EXCLUSIONS),
        "source_identity": _source_identity(
            (
                main_repo / "scripts" / "run_all_experiments.py",
                main_repo / "scripts" / "run_faaslora_shared_artifact_experiment.sh",
            )
        ),
        "source_commit": _git_commit(main_repo),
    }
    slora = _resolved_slora(
        profiles,
        env,
        gpu_list,
        generation_contract,
        fixed_output_max_tokens,
        fixed_prompt_max_tokens,
        selected_num_adapters,
    )
    slora["source_identity"] = _source_identity(
        (
            baselines_root / "scripts" / "run_slora_fair_experiment.sh",
            baselines_root / "scripts" / "replay_openai_trace.py",
            baselines_root / "scripts" / "summarize_serverlessllm_replay.py",
        )
    )
    slora_worktree = _git_worktree_identity(baselines_root / "repos" / "S-LoRA")
    slora["upstream_commit"] = slora_worktree["commit"]
    slora["upstream_worktree_identity"] = slora_worktree
    serverless = _resolved_serverless(
        profiles,
        env,
        gpu_list,
        storage_bandwidth_mib_s,
        selected_num_adapters,
    )
    serverless["source_identity"] = _source_identity(
        (
            baselines_root / "scripts" / "run_serverlessllm_fair_experiment.sh",
            baselines_root / "scripts" / "generate_serverlessllm_deploy_config.py",
            baselines_root / "scripts" / "replay_openai_trace.py",
            baselines_root / "scripts" / "start_serverlessllm_stack.sh",
            baselines_root / "scripts" / "deploy_serverlessllm_model.sh",
        )
    )
    serverless_worktree = _git_worktree_identity(
        baselines_root / "repos" / "ServerlessLLM"
    )
    serverless["upstream_commit"] = serverless_worktree["commit"]
    serverless["upstream_worktree_identity"] = serverless_worktree

    hashed = _normalize_for_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "shared_execution_envelope": shared,
            "systems": {
                "PrimeLoRA": prime,
                "S-LoRA": slora,
                "ServerlessLLM-new": serverless,
            },
        }
    )
    _assert_no_forbidden_hash_keys(hashed)
    return hashed


def _assert_no_forbidden_hash_keys(value: Any, path: str = "hashed_config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_HASH_KEYS:
                raise ValueError(f"forbidden provenance key entered hash at {path}.{key}")
            _assert_no_forbidden_hash_keys(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_forbidden_hash_keys(item, f"{path}[{index}]")


def expected_trace_role(seed: int, scenario: str) -> str:
    if scenario in STRICT_V2_SCENARIOS:
        try:
            return SEED_TRACE_ROLES[int(seed)]
        except KeyError as exc:
            raise ValueError(
                f"V2 protocol permits only seeds 41, 42, 43, 44, 45; got {seed}"
            ) from exc
    return "legacy"


def validate_trace_role(seed: int, trace_role: str, scenario: str, total_requests: int) -> str:
    expected = expected_trace_role(seed, scenario)
    observed = str(trace_role or expected).strip().lower()
    if observed != expected:
        raise ValueError(
            f"trace_role mismatch for seed {seed}: expected={expected!r} observed={observed!r}"
        )
    if scenario in STRICT_V2_SCENARIOS:
        if expected == "validation" and int(total_requests) != 1000:
            raise ValueError("V2 validation seed 41 must use exactly 1,000 requests")
        if expected == "heldout" and int(total_requests) != 4000:
            raise ValueError("V2 held-out seeds 43/44/45 must use exactly 4,000 requests")
        if expected == "smoke" and not (1 <= int(total_requests) <= 100):
            raise ValueError("V2 smoke seed 42 must use between 1 and 100 requests")
    return observed


def validate_campaign_protocol(
    *,
    campaign_kind: str,
    formal_run: bool,
    trace_role: str,
    model_profile: str,
    sampling_seed: int,
    faaslora_scenario: str,
    systems: Sequence[str],
    execution_order: Sequence[str],
    generation_contract: str,
) -> Dict[str, Any]:
    """Validate exact system/contract/order semantics for V2 paired campaigns.

    Exploratory non-formal rounds may omit a campaign kind.  Every formal V2
    round must opt into a publication protocol, and an opted-in protocol is
    checked even for smoke runs.
    """

    kind = str(campaign_kind or "").strip()
    system_list = [str(item) for item in systems]
    order_list = [str(item) for item in execution_order]
    if len(system_list) != len(set(system_list)):
        raise ValueError(f"campaign systems contain duplicates: {system_list}")
    if len(order_list) != len(set(order_list)):
        raise ValueError(f"campaign execution order contains duplicates: {order_list}")

    is_v2 = faaslora_scenario in STRICT_V2_SCENARIOS
    if formal_run and is_v2 and kind not in V2_CAMPAIGN_KINDS:
        raise ValueError(
            "formal V2 round requires FAIR_CAMPAIGN_KIND in "
            f"{sorted(V2_CAMPAIGN_KINDS)}; got {kind!r}"
        )
    if not kind:
        return {
            "campaign_kind": None,
            "publication_protocol_enforced": False,
            "expected_execution_order": None,
        }
    if set(order_list) != set(system_list):
        raise ValueError(
            "campaign execution order must be an exact permutation of selected systems: "
            f"systems={system_list} execution_order={order_list}"
        )
    if kind not in V2_CAMPAIGN_KINDS:
        raise ValueError(f"unsupported FAIR_CAMPAIGN_KIND={kind!r}")
    if faaslora_scenario != "v2_full":
        raise ValueError(
            f"{kind} requires FAIR_FAASLORA_SCENARIO=v2_full; "
            f"got {faaslora_scenario!r}"
        )

    model_key_by_profile = {
        "llama2_7b_main_v2_publicmix": "7b",
        "llama32_3b_main_modelscope": "3b",
    }
    try:
        model_key = model_key_by_profile[model_profile]
    except KeyError as exc:
        raise ValueError(
            f"{kind} permits only the frozen 7B/3B model profiles; got {model_profile!r}"
        ) from exc

    seed = int(sampling_seed)
    if seed not in SEED_TRACE_ROLES:
        raise ValueError(f"{kind} has no predeclared execution order for seed {seed}")
    expected_role = SEED_TRACE_ROLES[seed]
    if str(trace_role) != expected_role:
        raise ValueError(
            f"{kind} trace role mismatch for seed {seed}: "
            f"expected={expected_role!r} observed={trace_role!r}"
        )

    if kind == "v2_full_vs_serverless":
        expected_systems = {"faaslora", "serverlessllm"}
        if generation_contract != "legacy":
            raise ValueError(
                "v2_full_vs_serverless requires generation_contract=legacy"
            )
        seven_b_first = "faaslora" if seed % 2 else "serverlessllm"
        if model_key == "3b":
            seven_b_first = (
                "serverlessllm" if seven_b_first == "faaslora" else "faaslora"
            )
        expected_order = [
            seven_b_first,
            "serverlessllm" if seven_b_first == "faaslora" else "faaslora",
        ]
    else:
        expected_systems = {"faaslora", "slora"}
        if generation_contract != "fixed_length_greedy_v1":
            raise ValueError(
                "v2_c5_matched_output requires "
                "generation_contract=fixed_length_greedy_v1"
            )
        # The seed-42 smoke order was predeclared with S-LoRA first for 7B and
        # Prime first for 3B.  Held-out seeds alternate by seed, with the 3B
        # schedule reversed, so each system is balanced across model/seed.
        if seed == 42:
            first = "slora" if model_key == "7b" else "faaslora"
        else:
            first = "slora" if seed % 2 else "faaslora"
            if model_key == "3b":
                first = "faaslora" if first == "slora" else "slora"
        expected_order = [first, "faaslora" if first == "slora" else "slora"]

    if set(system_list) != expected_systems or len(system_list) != 2:
        raise ValueError(
            f"{kind} requires exact systems={sorted(expected_systems)}; "
            f"got {system_list}"
        )
    if order_list != expected_order:
        raise ValueError(
            f"{kind} execution order mismatch for model={model_key} seed={seed}: "
            f"expected={expected_order} observed={order_list}"
        )
    return {
        "campaign_kind": kind,
        "publication_protocol_enforced": True,
        "expected_execution_order": expected_order,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def register_sidecar(sidecar: Mapping[str, Any], registry_path: Path) -> Dict[str, Any]:
    registry_path = registry_path.resolve()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if registry_path.exists():
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        else:
            registry = {"schema_version": REGISTRY_SCHEMA_VERSION, "families": {}}
        if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
            raise ValueError(f"unsupported config registry schema: {registry_path}")
        families = registry.setdefault("families", {})
        family_id = str(sidecar["configuration_family_id"])
        family = families.setdefault(
            family_id,
            {
                "configuration_family": sidecar["configuration_family"],
                "model_profile": sidecar["model_profile"],
                "validation": [],
                "successful_validation": [],
                "smoke": [],
                "heldout": {
                    "system_resolved_config_sha256": None,
                    "seeds": [],
                    "sidecars": [],
                },
            },
        )
        if family.get("configuration_family") != sidecar["configuration_family"]:
            raise ValueError(f"configuration-family hash collision: {family_id}")
        # Backward-compatible extension of registries written before successful
        # completion became a separate publication gate.
        family.setdefault("successful_validation", [])
        role = str(sidecar["trace_role"])
        record = {
            "seed": int(sidecar["sampling_seed"]),
            "system_resolved_config_sha256": sidecar["system_resolved_config_sha256"],
            "sidecar": sidecar["sidecar_path"],
        }
        if role == "heldout":
            heldout = family["heldout"]
            frozen_hash = heldout.get("system_resolved_config_sha256")
            observed_hash = str(sidecar["system_resolved_config_sha256"])
            validation_hashes = {
                str(item.get("system_resolved_config_sha256") or "")
                for item in family.get("successful_validation", [])
                if int(item.get("seed", -1)) == 41
            }
            if observed_hash not in validation_hashes:
                raise ValueError(
                    "held-out configuration was not completed successfully on seed 41 "
                    "validation: "
                    f"family={family_id} observed={observed_hash} "
                    f"successful_validation_hashes={sorted(validation_hashes)}"
                )
            if frozen_hash not in (None, observed_hash):
                raise ValueError(
                    "held-out resolved configuration changed within one family: "
                    f"family={family_id} frozen={frozen_hash} observed={observed_hash} "
                    f"seed={sidecar['sampling_seed']}"
                )
            heldout["system_resolved_config_sha256"] = observed_hash
            seed = int(sidecar["sampling_seed"])
            previous_by_seed = {
                int(item["seed"]): str(item["system_resolved_config_sha256"])
                for item in heldout.get("sidecars", [])
            }
            if seed in previous_by_seed and previous_by_seed[seed] != observed_hash:
                raise ValueError(
                    f"held-out seed {seed} was already registered with a different configuration"
                )
            if seed not in heldout["seeds"]:
                heldout["seeds"].append(seed)
                heldout["seeds"].sort()
            if not any(
                int(item.get("seed", -1)) == seed
                and str(item.get("sidecar")) == str(record["sidecar"])
                for item in heldout["sidecars"]
            ):
                heldout["sidecars"].append(record)
        elif role in {"validation", "smoke"}:
            records = family[role]
            if record not in records:
                records.append(record)
        _atomic_write_json(registry_path, registry)
        return registry


def mark_successful_validation(
    *,
    sidecar_path: Path,
    registry_path: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    """Promote a registered seed-41 sidecar only after a complete formal round."""

    sidecar_path = sidecar_path.resolve()
    registry_path = registry_path.resolve()
    manifest_path = manifest_path.resolve()
    if not sidecar_path.is_file():
        raise ValueError(f"missing validation sidecar: {sidecar_path}")
    if not manifest_path.is_file():
        raise ValueError(f"missing validation manifest: {manifest_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if str(sidecar.get("trace_role") or "") != "validation":
        raise ValueError("successful validation marker requires trace_role=validation")
    if int(sidecar.get("sampling_seed", -1)) != 41:
        raise ValueError("successful validation marker requires seed 41")
    if sidecar.get("formal_run") is not True:
        raise ValueError("successful validation marker requires formal_run=true")
    if sidecar.get("source_clean_for_formal") is not True:
        raise ValueError("successful validation sidecar must pass the formal source gate")
    recorded_sidecar = Path(str(sidecar.get("sidecar_path") or "")).resolve()
    if recorded_sidecar != sidecar_path:
        raise ValueError(
            f"validation sidecar path mismatch: recorded={recorded_sidecar} "
            f"actual={sidecar_path}"
        )

    if str(manifest.get("status") or "") != "complete":
        raise ValueError("successful validation requires MANIFEST status=complete")
    if manifest.get("formal_run") is not True:
        raise ValueError("successful validation manifest requires formal_run=true")
    if manifest.get("source_clean_for_formal") is not True:
        raise ValueError("successful validation manifest failed the formal source gate")
    if str(manifest.get("trace_role") or "") != "validation":
        raise ValueError("successful validation manifest requires trace_role=validation")
    if int(manifest.get("sampling_seed", -1)) != 41:
        raise ValueError("successful validation manifest requires seed 41")

    sidecar_sha256 = _file_sha256(sidecar_path)
    manifest_sidecar_path = Path(
        str(manifest.get("system_resolved_config_path") or "")
    ).resolve()
    if manifest_sidecar_path != sidecar_path:
        raise ValueError(
            "manifest references a different resolved-config sidecar: "
            f"manifest={manifest_sidecar_path} actual={sidecar_path}"
        )
    if str(manifest.get("system_resolved_config_sidecar_sha256") or "") != sidecar_sha256:
        raise ValueError("manifest resolved-config sidecar SHA does not match current bytes")

    family_id = str(sidecar.get("configuration_family_id") or "")
    config_sha256 = str(sidecar.get("system_resolved_config_sha256") or "")
    if str(manifest.get("system_resolved_config_family_id") or "") != family_id:
        raise ValueError("manifest/sidecar configuration-family mismatch")
    if str(manifest.get("system_resolved_config_sha256") or "") != config_sha256:
        raise ValueError("manifest/sidecar resolved-config hash mismatch")
    if str(manifest.get("campaign_kind") or "") != str(
        sidecar.get("campaign_kind") or ""
    ):
        raise ValueError("manifest/sidecar campaign-kind mismatch")

    manifest_sha256 = _file_sha256(manifest_path)
    success_record = {
        "seed": 41,
        "system_resolved_config_sha256": config_sha256,
        "sidecar": str(sidecar_path),
        "sidecar_sha256": sidecar_sha256,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "campaign_kind": str(sidecar.get("campaign_kind") or ""),
    }

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if not registry_path.is_file():
            raise ValueError(
                f"validation sidecar was not registered before completion: {registry_path}"
            )
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
            raise ValueError(f"unsupported config registry schema: {registry_path}")
        family = (registry.get("families") or {}).get(family_id)
        if not isinstance(family, MutableMapping):
            raise ValueError(f"validation configuration family is absent: {family_id}")
        registered = any(
            int(item.get("seed", -1)) == 41
            and str(item.get("system_resolved_config_sha256") or "") == config_sha256
            and Path(str(item.get("sidecar") or "")).resolve() == sidecar_path
            for item in family.get("validation", [])
        )
        if not registered:
            raise ValueError(
                "successful validation sidecar has no matching pre-run registry record"
            )
        successes = family.setdefault("successful_validation", [])
        previous_same_sidecar = [
            item
            for item in successes
            if Path(str(item.get("sidecar") or "")).resolve() == sidecar_path
        ]
        if previous_same_sidecar and success_record not in previous_same_sidecar:
            raise ValueError(
                "validation sidecar was already promoted with different manifest bytes"
            )
        if success_record not in successes:
            successes.append(success_record)
        _atomic_write_json(registry_path, registry)
        return registry


def build_sidecar(args: argparse.Namespace, env: Mapping[str, str]) -> Dict[str, Any]:
    role = validate_trace_role(
        args.sampling_seed,
        args.trace_role,
        args.faaslora_scenario,
        args.total_requests,
    )
    hashed_config = build_hashed_config(
        baselines_root=args.baselines_root,
        main_repo=args.main_repo,
        model_profile=args.model_profile,
        dataset_profile=args.dataset_profile,
        workload_profile=args.workload_profile,
        total_requests=args.total_requests,
        selected_num_adapters=args.selected_num_adapters,
        generation_contract=args.generation_contract,
        fixed_output_max_tokens=args.fixed_output_max_tokens,
        fixed_prompt_max_tokens=args.fixed_prompt_max_tokens,
        storage_bandwidth_mib_s=args.storage_bandwidth_mib_s,
        zipf_exponent=args.zipf_exponent,
        active_adapter_cap=args.active_adapter_cap,
        hotset_rotation_requests=args.hotset_rotation_requests,
        hotset_rotation_mode=args.hotset_rotation_mode,
        hotset_overlap_fraction=args.hotset_overlap_fraction,
        faaslora_scenario=args.faaslora_scenario,
        gpu_ids=args.gpu_ids,
        env=env,
    )
    family = {
        "campaign_kind": args.campaign_kind or None,
        "model_profile": args.model_profile,
        "dataset_profile": args.dataset_profile,
        "workload_profile": args.workload_profile,
        "selected_num_adapters": int(args.selected_num_adapters),
        "generation_contract": args.generation_contract,
        "fixed_output_max_tokens": int(args.fixed_output_max_tokens),
        "fixed_prompt_max_tokens": int(args.fixed_prompt_max_tokens),
        "faaslora_scenario": args.faaslora_scenario,
    }
    config_hash = _canonical_sha256(hashed_config)
    trace_candidate = Path(args.trace_path).expanduser() if args.trace_path else None
    subset_candidate = (
        Path(args.adapter_subset_path).expanduser() if args.adapter_subset_path else None
    )
    trace_payload: Dict[str, Any] = {}
    if trace_candidate is not None and trace_candidate.is_file():
        loaded_trace = json.loads(trace_candidate.read_text(encoding="utf-8"))
        if isinstance(loaded_trace, Mapping):
            trace_payload = dict(loaded_trace)
    trace_load_profile = dict(trace_payload.get("load_profile", {}) or {})

    def effective_axis(explicit: Any, *trace_keys: str) -> Any:
        explicit_value = _coerce_axis_value(explicit)
        if explicit_value is not None:
            return explicit_value
        for key in trace_keys:
            if key in trace_payload and trace_payload.get(key) is not None:
                return _coerce_axis_value(trace_payload.get(key))
            if key in trace_load_profile and trace_load_profile.get(key) is not None:
                return _coerce_axis_value(trace_load_profile.get(key))
        return None

    effective_time_scale = effective_axis(
        args.time_scale_factor,
        "effective_time_scale_factor",
        "configured_time_scale_factor",
        "time_scale_factor",
    )
    effective_workload_axes = {
        "zipf_exponent": effective_axis(args.zipf_exponent, "zipf_exponent"),
        "active_adapter_cap": effective_axis(args.active_adapter_cap, "active_adapter_cap"),
        "hotset_rotation_requests": effective_axis(
            args.hotset_rotation_requests, "hotset_rotation_requests"
        ),
        "hotset_rotation_mode": effective_axis(
            args.hotset_rotation_mode, "hotset_rotation_mode"
        ),
        "hotset_overlap_fraction": effective_axis(
            args.hotset_overlap_fraction, "hotset_overlap_fraction"
        ),
    }
    full_run_identity = {
        "system_resolved_config_sha256": config_hash,
        "model_profile": args.model_profile,
        "dataset_profile": args.dataset_profile,
        "workload_profile": args.workload_profile,
        "total_requests": int(args.total_requests),
        "time_scale_factor": effective_time_scale,
        "selected_num_adapters": int(args.selected_num_adapters),
        "sampling_seed": int(args.sampling_seed),
        "trace_sha256": (
            _file_sha256(trace_candidate)
            if trace_candidate is not None and trace_candidate.is_file()
            else None
        ),
        "adapter_subset_sha256": (
            _file_sha256(subset_candidate)
            if subset_candidate is not None and subset_candidate.is_file()
            else None
        ),
        "run_tag": args.run_tag,
        "execution_order": args.execution_order.split(),
        "generation_contract": args.generation_contract,
        "fixed_output_max_tokens": int(args.fixed_output_max_tokens),
        "fixed_prompt_max_tokens": int(args.fixed_prompt_max_tokens),
        "storage_bandwidth_mib_s": float(args.storage_bandwidth_mib_s),
        "workload_overrides": effective_workload_axes,
        "faaslora_scenario": args.faaslora_scenario,
        "campaign_kind": args.campaign_kind or None,
    }
    source_status = source_cleanliness(args.baselines_root, args.main_repo)
    formal_run = bool(int(args.formal_run))
    if formal_run and not source_status["source_clean_for_formal"]:
        raise ValueError(
            "formal source gate failed while writing resolved-config sidecar: "
            f"baseline_tracked_dirty={source_status['baseline_tracked_dirty_paths']} "
            f"faaslora_disallowed_tracked_dirty={source_status['faaslora_disallowed_tracked_dirty_paths']}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_kind": args.campaign_kind or None,
        "formal_run": formal_run,
        "trace_role": role,
        "sampling_seed": int(args.sampling_seed),
        "model_profile": args.model_profile,
        "configuration_family": family,
        "configuration_family_label": args.configuration_family or None,
        "configuration_family_id": _canonical_sha256(family),
        "system_resolved_config_sha256": config_hash,
        "full_run_identity": full_run_identity,
        "full_run_identity_sha256": _canonical_sha256(full_run_identity),
        "source_clean_for_formal": bool(source_status["source_clean_for_formal"]),
        "source_cleanliness": source_status,
        "hashed_config": hashed_config,
        "excluded_from_hash_audit": {
            "run_tag": args.run_tag,
            "total_requests": int(args.total_requests),
            "time_scale_factor": effective_time_scale,
            "trace_path": args.trace_path,
            "adapter_subset_path": args.adapter_subset_path,
            "execution_order": args.execution_order.split(),
            "storage_bandwidth_mib_s": float(args.storage_bandwidth_mib_s),
            "workload_overrides": effective_workload_axes,
            "requested_axis_overrides": {
                "time_scale_factor": _coerce_axis_value(args.time_scale_factor),
                "zipf_exponent": _coerce_axis_value(args.zipf_exponent),
                "active_adapter_cap": _coerce_axis_value(args.active_adapter_cap),
                "hotset_rotation_requests": _coerce_axis_value(
                    args.hotset_rotation_requests
                ),
                "hotset_rotation_mode": _coerce_axis_value(args.hotset_rotation_mode),
                "hotset_overlap_fraction": _coerce_axis_value(
                    args.hotset_overlap_fraction
                ),
            },
        },
        "sidecar_path": str(args.output.resolve()),
        "registry_path": str(args.registry.resolve()),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines-root", type=Path, required=True)
    parser.add_argument("--main-repo", type=Path, required=True)
    parser.add_argument("--model-profile", required=True)
    parser.add_argument("--dataset-profile", required=True)
    parser.add_argument("--workload-profile", required=True)
    parser.add_argument("--total-requests", type=int, required=True)
    parser.add_argument("--selected-num-adapters", type=int, required=True)
    parser.add_argument("--sampling-seed", type=int, required=True)
    parser.add_argument("--time-scale-factor", default="")
    parser.add_argument("--formal-run", choices=("0", "1"), default="0")
    parser.add_argument("--trace-role", default="")
    parser.add_argument("--generation-contract", required=True)
    parser.add_argument("--fixed-output-max-tokens", type=int, required=True)
    parser.add_argument("--fixed-prompt-max-tokens", type=int, required=True)
    parser.add_argument("--storage-bandwidth-mib-s", type=float, required=True)
    parser.add_argument("--zipf-exponent", default="")
    parser.add_argument("--active-adapter-cap", default="")
    parser.add_argument("--hotset-rotation-requests", default="")
    parser.add_argument("--hotset-rotation-mode", default="")
    parser.add_argument("--hotset-overlap-fraction", default="")
    parser.add_argument("--faaslora-scenario", required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--configuration-family", default="")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--trace-path", default="")
    parser.add_argument("--adapter-subset-path", default="")
    parser.add_argument("--execution-order", default="")
    parser.add_argument("--campaign-kind", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if arguments and arguments[0] == "mark-validation-complete":
        parser = argparse.ArgumentParser(
            description="Promote a complete formal seed-41 validation round"
        )
        parser.add_argument("--sidecar", type=Path, required=True)
        parser.add_argument("--registry", type=Path, required=True)
        parser.add_argument("--manifest", type=Path, required=True)
        completion_args = parser.parse_args(arguments[1:])
        registry = mark_successful_validation(
            sidecar_path=completion_args.sidecar,
            registry_path=completion_args.registry,
            manifest_path=completion_args.manifest,
        )
        print(
            "[resolved-config] successful validation registered "
            f"sidecar={completion_args.sidecar.resolve()} "
            f"manifest={completion_args.manifest.resolve()} "
            f"families={len(registry.get('families', {}))}"
        )
        return 0

    args = parse_args(arguments)
    sidecar = build_sidecar(args, os.environ)
    _atomic_write_json(args.output.resolve(), sidecar)
    try:
        register_sidecar(sidecar, args.registry)
    except Exception:
        # Keep the sidecar as auditable evidence of the rejected launch.
        raise
    print(
        "[resolved-config] "
        f"role={sidecar['trace_role']} family={sidecar['configuration_family_id'][:12]} "
        f"sha256={sidecar['system_resolved_config_sha256']} sidecar={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
