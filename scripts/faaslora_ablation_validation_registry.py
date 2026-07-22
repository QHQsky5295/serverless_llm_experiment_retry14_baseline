#!/usr/bin/env python3
"""Immutable seed-41 validation registry for formal V2 FaaSLoRA ablations."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence


SCHEMA_VERSION = "eurosys27_v2_faaslora_ablation_validation_registry_v1"
EVIDENCE_SCHEMA_VERSION = "eurosys27_v2_faaslora_ablation_validation_evidence_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
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


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo.resolve()), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot resolve source commit for {repo}") from exc


def _digest(value: Any, label: str) -> str:
    result = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(result):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return result


def configuration_family(
    *,
    model_profile: str,
    dataset_profile: str,
    workload_profile: str,
    selected_num_adapters: int,
    gpu_ids: str,
    generation_contract: str = "legacy",
) -> Dict[str, Any]:
    normalized_gpu_ids = [item.strip() for item in str(gpu_ids).split(",") if item.strip()]
    if not normalized_gpu_ids:
        raise ValueError("at least one GPU id is required")
    return {
        "campaign_kind": "v2_a2_a3_ablation",
        "model_profile": str(model_profile),
        "dataset_profile": str(dataset_profile),
        "workload_profile": str(workload_profile),
        "selected_num_adapters": int(selected_num_adapters),
        "gpu_ids": normalized_gpu_ids,
        "generation_contract": str(generation_contract),
    }


def _load_object(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object: {path}")
    return payload


def _validation_record(
    *,
    manifest_path: Path,
    repo: Path,
    config_path: Path,
    family: Mapping[str, Any],
) -> Dict[str, Any]:
    manifest_path = manifest_path.resolve()
    repo = repo.resolve()
    config_path = config_path.resolve()
    manifest = _load_object(manifest_path, "validation manifest")
    if manifest.get("status") != "complete":
        raise ValueError("seed-41 validation manifest must have status=complete")
    if manifest.get("formal_run") is not True:
        raise ValueError("seed-41 validation manifest must have formal_run=true")
    if str(manifest.get("trace_role") or "") != "validation":
        raise ValueError("seed-41 validation manifest must have trace_role=validation")
    if list(manifest.get("scenarios") or []) != ["v2_full"]:
        raise ValueError("seed-41 validation must run exactly the v2_full scenario")
    trace = manifest.get("shared_trace") or {}
    if int(trace.get("sampling_seed", -1)) != 41:
        raise ValueError("validation manifest must bind sampling seed 41")
    if int(trace.get("requests", -1)) != 1000:
        raise ValueError("validation manifest must contain exactly 1,000 requests")
    snapshot = manifest.get("code_snapshot") or {}
    if snapshot.get("source_clean_for_formal") is not True:
        raise ValueError("validation manifest failed the formal source-clean gate")
    source_commit = str(snapshot.get("git_commit") or "")
    current_commit = _git_commit(repo)
    if not source_commit or source_commit != current_commit:
        raise ValueError(
            "validation manifest source commit does not match the current FaaSLoRA commit"
        )
    config_snapshot = manifest.get("config_snapshot") or {}
    current_config_sha = _file_sha256(config_path)
    if Path(str(config_snapshot.get("path") or "")).resolve() != config_path:
        raise ValueError("validation manifest config path mismatch")
    if str(config_snapshot.get("sha256") or "") != current_config_sha:
        raise ValueError("validation manifest config SHA mismatch")
    non_feature_hash = _digest(
        manifest.get("non_feature_frozen_config_sha256"),
        "validation non_feature_frozen_config_sha256",
    )
    if manifest.get("non_feature_frozen_config_consistent") is not True:
        raise ValueError("validation manifest non-feature hash is not internally consistent")
    manifest_family = manifest.get("configuration_family")
    if manifest_family != family:
        raise ValueError("validation manifest configuration family mismatch")
    return {
        "seed": 41,
        "non_feature_frozen_config_sha256": non_feature_hash,
        "manifest": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "manifest_bytes": manifest_path.stat().st_size,
        "source_commit": source_commit,
        "config_path": str(config_path),
        "config_sha256": current_config_sha,
    }


def register_successful_validation(
    *,
    registry_path: Path,
    manifest_path: Path,
    repo: Path,
    config_path: Path,
    family: Mapping[str, Any],
) -> Dict[str, Any]:
    registry_path = registry_path.resolve()
    record = _validation_record(
        manifest_path=manifest_path,
        repo=repo,
        config_path=config_path,
        family=family,
    )
    family_id = _canonical_sha256(family)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        registry = (
            _load_object(registry_path, "validation registry")
            if registry_path.is_file()
            else {"schema_version": SCHEMA_VERSION, "families": {}}
        )
        if registry.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported validation registry schema: {registry_path}")
        families = registry.setdefault("families", {})
        entry = families.setdefault(
            family_id,
            {
                "configuration_family": dict(family),
                "successful_validation": [],
                "heldout_frozen_non_feature_sha256": None,
                "heldout_launches": [],
            },
        )
        if entry.get("configuration_family") != family:
            raise ValueError(f"configuration-family hash collision: {family_id}")
        successes = entry.setdefault("successful_validation", [])
        if successes:
            if record in successes:
                return registry
            frozen_hashes = {
                str(item.get("non_feature_frozen_config_sha256") or "")
                for item in successes
            }
            observed_hash = str(record["non_feature_frozen_config_sha256"])
            if observed_hash not in frozen_hashes:
                raise ValueError(
                    "formal seed-41 validation is already frozen to a different "
                    "non-feature configuration; tune candidates only in non-formal "
                    "scratch rounds"
                )
            raise ValueError(
                "this configuration family already has a selected formal seed-41 "
                "validation manifest; reuse its immutable record"
            )
        successes.append(record)
        _atomic_write_json(registry_path, registry)
        return registry


def _verify_record_bytes(record: Mapping[str, Any]) -> None:
    manifest_path = Path(str(record.get("manifest") or "")).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"registered validation manifest is missing: {manifest_path}")
    if manifest_path.stat().st_size != int(record.get("manifest_bytes", -1)):
        raise ValueError("registered validation manifest byte count changed")
    if _file_sha256(manifest_path) != str(record.get("manifest_sha256") or ""):
        raise ValueError("registered validation manifest SHA changed")


def resolve_heldout_validation(
    *,
    registry_path: Path,
    evidence_path: Path,
    repo: Path,
    config_path: Path,
    family: Mapping[str, Any],
    sampling_seed: int,
    total_requests: int,
    round_dir: Path,
    expected_non_feature_sha256: str = "",
) -> Dict[str, Any]:
    if int(sampling_seed) not in {43, 44, 45} or int(total_requests) != 4000:
        raise ValueError("held-out ablation requires seeds 43/44/45 and 4,000 requests")
    registry_path = registry_path.resolve()
    evidence_path = evidence_path.resolve()
    repo = repo.resolve()
    config_path = config_path.resolve()
    current_commit = _git_commit(repo)
    current_config_sha = _file_sha256(config_path)
    requested_hash = str(expected_non_feature_sha256 or "").strip().lower()
    if requested_hash:
        requested_hash = _digest(requested_hash, "expected non-feature hash")
    family_id = _canonical_sha256(family)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        registry = _load_object(registry_path, "validation registry")
        if registry.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported validation registry schema: {registry_path}")
        entry = (registry.get("families") or {}).get(family_id)
        if not isinstance(entry, MutableMapping):
            raise ValueError("no successful seed-41 validation exists for this family")
        if entry.get("configuration_family") != family:
            raise ValueError("held-out configuration family mismatch")
        compatible = []
        for record in entry.get("successful_validation", []):
            _verify_record_bytes(record)
            if (
                str(record.get("source_commit") or "") == current_commit
                and str(record.get("config_sha256") or "") == current_config_sha
            ):
                compatible.append(record)
        if not compatible:
            raise ValueError(
                "no successful seed-41 validation matches the current source commit/config SHA"
            )
        if len(compatible) != 1:
            raise ValueError(
                "held-out launch requires exactly one selected formal seed-41 "
                f"validation record; found {len(compatible)}"
            )
        frozen = str(entry.get("heldout_frozen_non_feature_sha256") or "")
        if frozen:
            selected_hash = _digest(frozen, "frozen held-out non-feature hash")
            if requested_hash and requested_hash != selected_hash:
                raise ValueError("requested held-out hash differs from the already frozen hash")
        elif requested_hash:
            selected_hash = requested_hash
        else:
            selected_hash = str(
                compatible[0].get("non_feature_frozen_config_sha256") or ""
            )
        matching = [
            item
            for item in compatible
            if str(item.get("non_feature_frozen_config_sha256") or "") == selected_hash
        ]
        if not matching:
            raise ValueError("selected held-out hash has no compatible successful validation")
        selected_record = matching[0]
        if not frozen:
            entry["heldout_frozen_non_feature_sha256"] = selected_hash
            entry["heldout_frozen_from_validation_manifest"] = selected_record["manifest"]
            entry["heldout_frozen_from_validation_manifest_sha256"] = selected_record[
                "manifest_sha256"
            ]
        launch = {
            "seed": int(sampling_seed),
            "round_dir": str(round_dir.resolve()),
            "non_feature_frozen_config_sha256": selected_hash,
        }
        launches = entry.setdefault("heldout_launches", [])
        if launch not in launches:
            launches.append(launch)
        _atomic_write_json(registry_path, registry)

    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "configuration_family": dict(family),
        "configuration_family_id": family_id,
        "selected_non_feature_frozen_config_sha256": selected_hash,
        "successful_validation": selected_record,
        "heldout_seed": int(sampling_seed),
        "heldout_requests": int(total_requests),
        "heldout_round_dir": str(round_dir.resolve()),
        "source_commit": current_commit,
        "config_path": str(config_path),
        "config_sha256": current_config_sha,
        "registry_path": str(registry_path),
        "registry_sha256_after_freeze": _file_sha256(registry_path),
    }
    if evidence_path.is_file():
        existing = _load_object(evidence_path, "held-out validation evidence")
        if existing.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported held-out evidence schema: {evidence_path}")
        _digest(
            existing.get("registry_sha256_after_freeze"),
            "existing evidence registry_sha256_after_freeze",
        )
        stable_existing = dict(existing)
        stable_candidate = dict(evidence)
        # The registry legitimately grows as later held-out seeds are launched.
        # Its SHA records the original freeze point but must not make an already
        # published per-round evidence file mutable on resume.
        stable_existing.pop("registry_sha256_after_freeze", None)
        stable_candidate.pop("registry_sha256_after_freeze", None)
        if stable_existing != stable_candidate:
            raise ValueError(
                "existing held-out validation evidence differs from the requested "
                "run; use a new unique round directory"
            )
        return existing
    _atomic_write_json(evidence_path, evidence)
    return evidence


def _family_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return configuration_family(
        model_profile=args.model_profile,
        dataset_profile=args.dataset_profile,
        workload_profile=args.workload_profile,
        selected_num_adapters=args.selected_num_adapters,
        gpu_ids=args.gpu_ids,
        generation_contract=args.generation_contract,
    )


def _add_family_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-profile", required=True)
    parser.add_argument("--dataset-profile", required=True)
    parser.add_argument("--workload-profile", required=True)
    parser.add_argument("--selected-num-adapters", type=int, required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--generation-contract", default="legacy")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register-validation")
    register.add_argument("--registry", type=Path, required=True)
    register.add_argument("--manifest", type=Path, required=True)
    register.add_argument("--repo", type=Path, required=True)
    register.add_argument("--config", type=Path, required=True)
    _add_family_args(register)
    resolve = subparsers.add_parser("resolve-heldout")
    resolve.add_argument("--registry", type=Path, required=True)
    resolve.add_argument("--evidence", type=Path, required=True)
    resolve.add_argument("--repo", type=Path, required=True)
    resolve.add_argument("--config", type=Path, required=True)
    resolve.add_argument("--sampling-seed", type=int, required=True)
    resolve.add_argument("--total-requests", type=int, required=True)
    resolve.add_argument("--round-dir", type=Path, required=True)
    resolve.add_argument("--expected-non-feature-sha256", default="")
    _add_family_args(resolve)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    family = _family_from_args(args)
    if args.command == "register-validation":
        register_successful_validation(
            registry_path=args.registry,
            manifest_path=args.manifest,
            repo=args.repo,
            config_path=args.config,
            family=family,
        )
        print(f"registered successful seed-41 validation: {args.manifest.resolve()}")
        return 0
    evidence = resolve_heldout_validation(
        registry_path=args.registry,
        evidence_path=args.evidence,
        repo=args.repo,
        config_path=args.config,
        family=family,
        sampling_seed=args.sampling_seed,
        total_requests=args.total_requests,
        round_dir=args.round_dir,
        expected_non_feature_sha256=args.expected_non_feature_sha256,
    )
    print(evidence["selected_non_feature_frozen_config_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
