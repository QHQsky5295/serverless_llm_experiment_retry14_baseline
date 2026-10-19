#!/usr/bin/env python3
"""Strict, append-only export of the D701 3B native source profile.

This is a qualification artifact, not a performance result.  It binds the
measured native service/preparation intervals to the exact D701 source run,
input trace, adapter content identities, runtime receipt, and resource
envelope.  Missing classes, source/configuration drift, request-time packing,
or incomplete release evidence are fatal; no interpolation, pooled class, or
zero-cost default is permitted.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from faaslora.datasets.workload_generator import FrozenReplayPlan
from faaslora.experiment.instance_pool import FrozenServiceProfiles
from faaslora.preloading.preloading_planner import FrozenPreparationProfiles
from scripts import ieee_tc_preflight as p


ROOT = p.ROOT
RAW_DIR = ROOT / "results/ieee_tc/p2_backend_qualification/d701_20261019"
RAW_PATH = RAW_DIR / "source1.json"
LAUNCH_PATH = RAW_DIR / "source1_launch.json"
CONFIG_PATH = ROOT / "configs/ieee_tc/20261017_d641_3b_native_source.yaml"
PREP_DIR = ROOT / "paper_results/ieee_tc/p2_backend/d701_3b_same_spec_profile"
SPEC_PATH = PREP_DIR / "source_spec.json"
PREP_PATH = PREP_DIR / "prepare.json"
OUT = PREP_DIR / "export"
CONTRACT_PATH = ROOT / "paper_results/ieee_tc/p2_backend/d643_3b_initializer/development_contract.json"


def read(path: Path):
    return json.loads(path.read_text())


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, allow_nan=False,
        separators=(",", ":")).encode()).hexdigest()


def ref(path: Path) -> dict:
    return {"path": str(path), "sha256": p.digest(path)}


def require_digest(value: str, label: str) -> None:
    if not (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value)):
        raise ValueError(f"{label} is not a SHA-256 digest")


def require_completed_run(raw: dict, launch: dict, spec: dict) -> None:
    if raw.get("kind") != "backend_native_native_source_matrix_qualification_v1":
        raise ValueError("D701 raw kind is not the native source qualification contract")
    if raw.get("pass") is not True or raw.get("stage") != "complete":
        raise ValueError("D701 native source run is not complete")
    if raw.get("source_profile_spec") != spec:
        raise ValueError("D701 raw source specification differs from the frozen specification")
    if raw.get("source_profile_spec_sha256") != p.digest(SPEC_PATH):
        raise ValueError("D701 source specification SHA mismatch")
    for key in ("runtime_receipt_sha256", "config_sha256", "source_profile_spec_sha256"):
        require_digest(raw.get(key), f"raw.{key}")
    if (launch.get("pass") is not True
            or launch.get("service_returncode") != 0
            or launch.get("watchdog_returncode") != 0
            or launch.get("service_path_removed") is not True
            or launch.get("native_gpu_context_release_confirmed") is not True):
        raise ValueError("D701 gated launch did not close cleanly")
    if (raw.get("shutdown_called") is not True
            or raw.get("profile_workspaces_removed") is not True
            or raw.get("physical_allocation", {}).get("released") is not True):
        raise ValueError("D701 native allocation/workspace release is incomplete")
    if raw.get("artifact_mode") != "prepublished_gzip_v1_real_remote_no_fallback":
        raise ValueError("D701 did not use the frozen published-source delivery mode")
    requests = raw.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("D701 has no request observations")
    if any(q.get("pass") is not True or q.get("reservation_released") is not True
           or q.get("actual_tokens") != q.get("target_tokens") for q in requests):
        raise ValueError("D701 contains an incomplete request or generation contract")
    transfers = raw.get("remote_transfers")
    if not isinstance(transfers, list) or not transfers:
        raise ValueError("D701 has no remote transfer evidence")
    ids = [r.get("transfer_id") for r in transfers]
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("D701 transfer identities are not unique")
    for transfer in transfers:
        if (transfer.get("state") != "published"
                or transfer.get("content_verified") is not True
                or transfer.get("published_archive_verified") is not True
                or transfer.get("remote_delivery_mode") != "prepublished_gzip_v1"
                or transfer.get("remote_pack_performed") is not False):
            raise ValueError("D701 contains an unverified or request-time-packed transfer")
    policy = launch.get("preflight", {}).get("policy")
    if not isinstance(policy, dict) or not isinstance(policy.get("service_cpus"), list):
        raise ValueError("D701 launch lacks the observed service CPU policy")
    if raw.get("service", {}).get("affinity") != policy["service_cpus"]:
        raise ValueError("D701 service affinity differs from its launch policy")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"append-only export already exists: {OUT}")
    raw = read(RAW_PATH)
    launch = read(LAUNCH_PATH)
    spec = read(SPEC_PATH)
    prep = read(PREP_PATH)
    contract = read(CONTRACT_PATH)

    if prep.get("kind") != "d701_3b_same_spec_profile_preparation_v1":
        raise ValueError("D701 preparation manifest kind mismatch")
    if prep.get("profile_export_authorized") is not False or prep.get("gpu_runs") != 0:
        raise ValueError("D701 preparation manifest has been changed from its static identity")
    if prep.get("source_spec", {}).get("sha256") != p.digest(SPEC_PATH):
        raise ValueError("D701 preparation manifest source SHA mismatch")
    if p.digest(CONFIG_PATH) != raw.get("config_sha256"):
        raise ValueError("D701 configuration SHA differs from the gated run")
    require_completed_run(raw, launch, spec)
    if raw.get("trace", {}).get("source_sha256") != spec.get("trace_sha256"):
        raise ValueError("D701 trace SHA differs from its source specification")
    trace_path = Path(prep["source_plan"]["path"])
    if not trace_path.is_absolute():
        trace_path = ROOT / trace_path
    if p.digest(trace_path) != prep["source_plan"]["sha256"]:
        raise ValueError("D701 source trace has changed")

    config = yaml.safe_load(CONFIG_PATH.read_text())
    cfg = copy.deepcopy(config["model_profiles"]["llama32_3b_main_modelscope"]["model"])
    cfg.update(
        generation_contract="fixed_length_greedy_v1",
        canonical_prompt_renderer="role_lines_v1",
        max_input_len=759,
        max_output_tokens_cap=256,
        ieee_physical_allocation=True,
        timing_contract="ieee_tc_native_v1",
        ieee_worker_observation=True,
        ieee_gpu_references=True,
        ieee_scheduler_observation=True,
        ieee_input_upper_bounds=[759],
    )
    cfg.update(copy.deepcopy(spec["model_overrides"]))
    pool = Path(config["model_profiles"]["llama32_3b_main_modelscope"]["storage"]["remote_dir"])
    pool = pool.resolve(strict=True)
    plan = FrozenReplayPlan.load(trace_path, count=1000, profile="W0")
    resolved_spec, _, identities, _, _ = p.source_profile_inputs(SPEC_PATH, plan, cfg, pool)
    if resolved_spec != spec or len(identities) != 24:
        raise ValueError("D701 identity resolver did not recover the 24 frozen representatives")
    if any(identity.get("adapter_id") != adapter_id
           for adapter_id, identity in identities.items()):
        raise ValueError("D701 adapter identity is not self-consistent")

    # Recompute static coverage before materializing any profile.  This is the
    # guard that blocks the historical 30-class gap from being silently filled.
    static_coverage = p.source_profile_class_coverage(raw, identities)
    if (static_coverage.get("measured_service_classes") != 160
            or static_coverage.get("required_service_classes") != 160
            or static_coverage.get("missing_service_classes")
            or static_coverage.get("missing_content_sources")
            or static_coverage.get("class_domain_covered") is not True):
        raise ValueError("D701 does not cover the complete measured service class domain")

    input_contract = dict(
        source_trace_sha256=spec["trace_sha256"],
        content_index_sha256=spec["content_index"]["sha256"],
        generation_contract=raw["model_config"]["generation_contract"],
        canonical_prompt_renderer=raw["model_config"]["canonical_prompt_renderer"],
        prompt_content_cap=raw["model_config"]["max_input_len"],
        output_cap=raw["model_config"]["max_output_tokens_cap"],
        artifact_mode=raw["artifact_mode"],
        declared_source_profile_sha256=raw["source_profile_spec_sha256"],
    )
    observed_policy = launch["preflight"]["policy"]
    envelope = dict(
        inference_policy=copy.deepcopy(observed_policy),
        artifact_service=dict(
            memory_high_bytes=2 * p.GIB,
            memory_max_bytes=4 * p.GIB,
            memory_swap_max_bytes=0,
            cpus=list(observed_policy["service_cpus"]),
            published_server_sha256="a31fa9212143914f5a7fb21dea126a23eaafedffe199ff9513de9ce03eb62a2c",
            delivery_mode="prepublished_gzip_v1",
            link_mbps=1000,
            rate_limit=None,
        ),
    )
    # The published server fingerprint is part of the historical export
    # contract; reject accidental typos through the explicit length check.
    require_digest(envelope["artifact_service"]["published_server_sha256"],
                   "published server SHA")
    context = dict(
        backend_environment_sha256=raw["runtime_receipt_sha256"],
        resource_envelope_sha256=object_hash(envelope),
        input_contract_sha256=object_hash(input_contract),
    )
    require_digest(context["resource_envelope_sha256"], "resource envelope SHA")
    require_digest(context["input_contract_sha256"], "input contract SHA")
    if FrozenServiceProfiles.model_identity(raw["model_config"]) != FrozenServiceProfiles.model_identity(contract["model_config"]):
        raise ValueError("D701 runtime identity differs from the frozen development contract")

    source_run_sha = p.digest(RAW_PATH)
    service, preparation, measured_coverage = p.measured_source_profile_payloads(
        raw, source_run_sha256=source_run_sha, identities=identities,
        context=context, model_config=raw["model_config"])
    if (len(service["samples"]) != sum(len(w["requests"]) for w in spec["waves"]
                                        if w["role"] == "representative_measurement")
            or not preparation["samples"]):
        raise ValueError("D701 measured profile sample counts do not match representative waves")

    # Build the files only after all raw evidence and domain checks pass.  The
    # exclusive directory/file creation is intentional: an existing export is
    # never silently replaced.
    OUT.mkdir(parents=False, exist_ok=False)

    def save(name: str, value) -> Path:
        path = OUT / name
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False, sort_keys=True)
            stream.write("\n")
        return path

    service_path = save("service.json", service)
    preparation_path = save("preparation.json", preparation)
    coverage_path = save("coverage.json", measured_coverage)
    loaded_service = FrozenServiceProfiles.load(
        service_path, expected_sha256=p.digest(service_path),
        model_config=raw["model_config"], expected_context=context,
        beta=contract["service_ewma_beta"])
    loaded_preparation = FrozenPreparationProfiles.load(
        preparation_path, expected_sha256=p.digest(preparation_path),
        model_config=raw["model_config"], expected_context=context,
        beta=contract["preparation_ewma_beta"])
    if (len(loaded_service.profiles) != 160
            or len(loaded_preparation.profiles) < 1):
        raise ValueError("D701 profile loader did not recover measured classes")

    manifest = dict(
        kind="d701_3b_same_spec_profile_export_v1",
        campaign="D701_20261019",
        passed=True,
        development_only=True,
        production_profile_frozen=False,
        full_qualified=False,
        numerical_adapter_correctness_qualified=False,
        no_figures=True,
        source_run_sha256=source_run_sha,
        source_run=ref(RAW_PATH),
        launch=ref(LAUNCH_PATH),
        source_spec=ref(SPEC_PATH),
        preparation_manifest=ref(PREP_PATH),
        config=ref(CONFIG_PATH),
        trace=ref(trace_path),
        development_contract=ref(CONTRACT_PATH),
        raw_runtime_receipt_sha256=raw["runtime_receipt_sha256"],
        input_contract=input_contract,
        resource_envelope=envelope,
        context=context,
        structured_hash="SHA256 UTF-8 JSON sort_keys=True separators=(comma,colon) allow_nan=False",
        service_profile=loaded_service.identity(),
        preparation_profile=loaded_preparation.identity(),
        coverage=dict(
            static_class_domain=static_coverage,
            measured_class_domain=measured_coverage,
            service_class_domain_covered=True,
            measured_service_classes=len(loaded_service.profiles),
            required_service_classes=160,
            missing_service_classes=[],
            missing_content_sources=[],
            measured_preparation_classes=len(loaded_preparation.profiles),
            preparation_samples=len(preparation["samples"]),
            service_samples=len(service["samples"]),
            timing_samples_exported=True,
        ),
        remote_delivery=dict(
            artifact_mode=raw["artifact_mode"],
            delivery_mode="prepublished_gzip_v1",
            remote_pack_performed=False,
            transfer_count=len(raw["remote_transfers"]),
        ),
        qualification_boundary=(
            "development-only measured native profile; this artifact does not establish "
            "numerical adapter correctness, Full workload qualification, SLO attainment, "
            "or a performance ranking"),
    )
    save("manifest.json", manifest)
    print(json.dumps({
        "passed": True,
        "service_classes": len(loaded_service.profiles),
        "required_service_classes": 160,
        "preparation_classes": len(loaded_preparation.profiles),
        "service_samples": len(service["samples"]),
        "preparation_samples": len(preparation["samples"]),
        "remote_transfers": len(raw["remote_transfers"]),
        "development_only": True,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
