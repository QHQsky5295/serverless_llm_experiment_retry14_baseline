#!/usr/bin/env python3
"""Strict append-only audit for the D703 3B full W0 replay.

D703 is deliberately a full native-contract/SLO qualification, not a numerical
adapter-correctness oracle.  The audit preserves that boundary: it accepts the
complete fixed-length request and lifecycle evidence, but keeps ``n_correct``
unknown and therefore does not manufacture a G1 ranking result.
"""

from __future__ import annotations

import collections
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from faaslora.experiment.instance_pool import FrozenServiceProfiles, ServiceObservationClass
from faaslora.preloading.preloading_planner import FrozenPreparationProfiles
from scripts import ieee_tc_preflight as p


ROOT = p.ROOT
RAW = Path("/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d703_20261019/3b_full_w0")
RESULT = next(RAW.glob("experiment_results_full*.json"))
LAUNCH = RAW / "launch.json"
OUTCOME = RAW / "launch.launch/physical_deployment/main_outcome.json"
SUMMARY = RAW / "launch.launch/physical_deployment/summary.json"
PHYSICAL = RAW / "launch.launch/physical_deployment/physical_resource_summary.json"
WATCHDOG = RAW / "launch.launch/watchdog.jsonl"
CONFIG = ROOT / "configs/ieee_tc/20261019_d703_3b_full_w0_d701profile.yaml"
SERVICE_PROFILE = ROOT / "paper_results/ieee_tc/p2_backend/d701_3b_same_spec_profile/export/service.json"
PREPARATION_PROFILE = ROOT / "paper_results/ieee_tc/p2_backend/d701_3b_same_spec_profile/export/preparation.json"
PROFILE_MANIFEST = ROOT / "paper_results/ieee_tc/p2_backend/d701_3b_same_spec_profile/export/manifest.json"
OUT = ROOT / "paper_results/ieee_tc/p2_backend/d703_3b_full_w0_d701profile"


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ref(path: Path) -> dict:
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def remote_state() -> dict:
    """Read-only post-run witness; this function never mutates the remote node."""
    state = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "primelora-artifact-174",
         "systemctl --user show primelora-artifact-tc-3b.service "
         "-p ActiveState -p SubState -p MainPID -p Result"],
        check=True, capture_output=True, text=True,
    )
    fields = dict(line.split("=", 1) for line in state.stdout.splitlines() if "=" in line)
    if fields.get("ActiveState") != "inactive" or fields.get("MainPID") != "0":
        raise ValueError(f"D703 remote service is not closed: {fields}")
    link = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "primelora-artifact-174",
         "cat /sys/class/net/eno1/speed /sys/class/net/eno1/duplex /sys/class/net/eno1/carrier"],
        check=True, capture_output=True, text=True,
    )
    values = link.stdout.split()
    if len(values) != 3:
        raise ValueError(f"unexpected remote link witness: {link.stdout!r}")
    return {
        "service_unit": "primelora-artifact-tc-3b.service",
        "service": fields,
        "link": {"interface": "eno1", "speed_mbps": int(values[0]),
                  "duplex": values[1], "carrier": int(values[2])},
        "command_contract": "read_only_post_run_state_v1",
    }


def calibrate_control_lower(output: Path) -> None:
    """Reuse sealed D703 evidence for the D711 preregistered calibration.

    This mode never contacts the remote machine and never changes the original
    audit. Stream the large outcome with the existing ijson projection pattern.
    """
    import ijson
    import yaml
    from faaslora.datasets.workload_generator import FrozenReplayPlan
    if output.exists():
        raise FileExistsError(output)
    audit = read(OUT / 'audit.json')
    for key in ('config', 'launch', 'outcome'):
        source = audit[key]
        if digest(Path(source['path'])) != source['sha256']:
            raise ValueError(f'sealed D703 {key} changed')
    launch = read(LAUNCH)
    plan = FrozenReplayPlan.load(launch['external_replay']['plan']['source_path'])
    if plan.identity() != launch['external_replay']['plan'] or len(plan.entries) != 4000:
        raise ValueError('D703 source trace identity changed')
    start = launch['external_replay']['replay_t0_s']
    stop = start + plan.entries[1000].offset_s
    cfg = yaml.safe_load(CONFIG.read_text())['resource_coordination']
    with OUTCOME.open('rb') as stream:
        events = ijson.items(stream, 'mechanism_events._ieee_control_events.item', use_float=True)
        result = p.derive_ieee_low_load_control(events, cfg['ieee_scaling'],
            start_s=start, stop_s=stop, min_instances=cfg['min_instances'])
    result.update(source_campaign='d703_3b_full_w0_d701profile',
        prefix_count=1000, development_only=True, held_out=False,
        plan_sha256=p.check_plan(),
        source_refs=[ref(CONFIG), ref(LAUNCH), ref(OUTCOME), ref(OUT/'audit.json'),
                     ref(Path(__file__)), ref(Path(p.__file__))],
        source_trace_sha256=plan.source_sha256,
        scope=p.scope_identity(p.cg_path().name))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('selected','source_refs','scope')}, indent=2))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"append-only audit already exists: {OUT}")
    required = (RESULT, LAUNCH, OUTCOME, SUMMARY, PHYSICAL, WATCHDOG, CONFIG,
                SERVICE_PROFILE, PREPARATION_PROFILE, PROFILE_MANIFEST)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    plan_sha = p.check_plan()
    frozen_v1 = ROOT / "docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md"
    single_run_v2 = ROOT / "docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md"
    if digest(frozen_v1) != "5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22":
        raise ValueError("frozen V1 metric protocol changed")

    result = read(RESULT)
    launch = read(LAUNCH)
    outcome = read(OUTCOME)
    summary = read(SUMMARY)
    physical = read(PHYSICAL)
    config = __import__("yaml").safe_load(CONFIG.read_text())
    service_payload = read(SERVICE_PROFILE)
    preparation_payload = read(PREPARATION_PROFILE)
    profile_manifest = read(PROFILE_MANIFEST)
    scenario = result["detailed_results"]["tc_ieee_full"]
    rows = scenario["requests"]
    if not (len(rows) == 4000 and result["metadata"].get("total_requests") == 4000
            and scenario.get("total") == 4000 and scenario.get("completed") == 4000
            and scenario.get("failed") == 0):
        raise ValueError("D703 is not a complete 4,000-request replay")

    service_sha = digest(SERVICE_PROFILE)
    preparation_sha = digest(PREPARATION_PROFILE)
    if config["resource_coordination"]["ieee_service_profile"]["sha256"] != service_sha:
        raise ValueError("D703 config/service profile SHA mismatch")
    if config["resource_coordination"]["ieee_preparation_profile"]["sha256"] != preparation_sha:
        raise ValueError("D703 config/preparation profile SHA mismatch")
    if profile_manifest.get("development_only") is not True:
        raise ValueError("D701 profile boundary changed unexpectedly")

    loaded_service = FrozenServiceProfiles.load(
        SERVICE_PROFILE, expected_sha256=service_sha,
        model_config=service_payload["model_config"],
        expected_context=service_payload["context"], beta=0.5,
    )
    loaded_preparation = FrozenPreparationProfiles.load(
        PREPARATION_PROFILE, expected_sha256=preparation_sha,
        model_config=preparation_payload["model_config"],
        expected_context=preparation_payload["context"], beta=0.5,
    )
    if len(loaded_service.profiles) != 160 or len(loaded_preparation.profiles) != 96:
        raise ValueError("D701 profile cardinality changed")

    request_ids, adapters = set(), set()
    tiers = collections.Counter()
    service_classes, preparation_classes = set(), set()
    remote_rows = 0
    output_tokens = 0
    for row in rows:
        rid, aid = row.get("request_id"), row.get("adapter_id")
        if not isinstance(rid, str) or not rid or rid in request_ids:
            raise ValueError(f"invalid/duplicate request identity: {rid!r}")
        if not isinstance(aid, str) or not aid:
            raise ValueError(f"invalid adapter identity for {rid!r}")
        request_ids.add(rid)
        adapters.add(aid)
        if row.get("success") is not True or row.get("error") is not None:
            raise ValueError(f"request {rid} failed")
        target = min(int(row["source_expected_output_tokens"]), 256)
        if any(row.get(k) != target for k in
               ("requested_completion_tokens", "completion_tokens", "output_tokens")):
            raise ValueError(f"request {rid} violates fixed output contract")
        timing = row.get("native_token_timing") or {}
        if (int(timing.get("native_output_tokens", -1)) != target
                or timing.get("native_terminal_observed") is not True
                or row.get("completion_token_source") != "vllm_token_ids"
                or row.get("output_contract_match") is not True
                or row.get("generation_contract") != "fixed_length_greedy_v1"
                or not isinstance(row.get("completion_token_ids_sha256"), str)):
            raise ValueError(f"request {rid} lacks native generation evidence")
        output_tokens += target

        evidence = row.get("gpu_reference_evidence") or {}
        intent = evidence.get("intent") or {}
        receipt = evidence.get("receipt") or {}
        source_admission = evidence.get("source_admission") or {}
        source = source_admission.get("source") or {}
        release = evidence.get("release_receipt") or {}
        if (intent.get("lora_name") != aid or receipt.get("lora_name") != aid
                or evidence.get("state") != "released"
                or release.get("released") is not True
                or release.get("live_staged_host_leases") != 0
                or not isinstance(receipt.get("expected_content_sha256"), str)
                or source.get("content_sha256", source.get("expected_content_sha256"))
                != receipt["expected_content_sha256"]):
            raise ValueError(f"request {rid} lacks adapter/content/release binding")
        if evidence.get("confirmed_dispatch_snapshot") is not True:
            raise ValueError(f"request {rid} lacks confirmed dispatch snapshot")

        service_class = ServiceObservationClass(**source_admission["service_class"])
        if service_class not in loaded_service.profiles:
            raise ValueError(f"request {rid} used unsupported service class")
        service_classes.add(service_class)
        tier = source.get("tier")
        tiers[tier] += 1
        if tier != "gpu":
            prep_class = FrozenPreparationProfiles.source_class(
                source, loaded_preparation.size_edges_bytes)
            if prep_class not in loaded_preparation.profiles:
                raise ValueError(f"request {rid} used unsupported preparation class")
            preparation_classes.add(prep_class)
        remote = evidence.get("remote_preparation")
        if remote is not None:
            remote_rows += 1
            if (remote.get("remote_delivery_mode") != "prepublished_gzip_v1"
                    or remote.get("remote_pack_performed") is not False
                    or remote.get("content_verified") is not True
                    or remote.get("published_archive_verified") is not True):
                raise ValueError(f"request {rid} has invalid remote delivery evidence")

    contract = outcome.get("launch_contract") or {}
    if (contract.get("contract") != "ieee_full_integrated_replay_v1"
            or contract.get("replay_scope") != "full_source_v1"
            or contract.get("service_profile_id") != service_sha
            or contract.get("preparation_profile_id") != preparation_sha):
        raise ValueError("D703 launch contract did not bind full source and D701 profiles")
    transfers = outcome.get("remote_transfers") or []
    if len(transfers) != result["metadata"].get("remote_artifact_transfer_count"):
        raise ValueError("remote transfer count mismatch")
    for transfer in transfers:
        if (transfer.get("state") != "published"
                or transfer.get("remote_delivery_mode") != "prepublished_gzip_v1"
                or transfer.get("remote_pack_performed") is not False
                or transfer.get("content_verified") is not True
                or transfer.get("published_archive_verified") is not True):
            raise ValueError("unverified/request-time-packed transfer")
    if result["metadata"].get("request_time_pack_count") != 0:
        raise ValueError("request-time packing was observed")

    if not (launch.get("pass") is True and launch.get("service_returncode") == 0
            and launch.get("watchdog_returncode") == 0
            and launch.get("replay_returncode") == 0
            and launch.get("service_path_removed") is True
            and launch.get("native_gpu_context_release_confirmed") is True):
        raise ValueError("gated launch did not close cleanly")
    if (outcome.get("run_completed") is not True or outcome.get("error") is not None
            or outcome.get("cleanup_errors") or outcome.get("measurement_errors")
            or outcome.get("cleanup", {}).get("complete") is not True):
        raise ValueError("D703 outcome cleanup/measurement contract is incomplete")
    if (summary.get("measurement_complete") is not True
            or summary.get("n_plan") != 4000 or summary.get("n_terminal") != 4000
            or summary.get("n_native_contract_complete") != 4000
            or summary.get("n_interrupted") != 0 or summary.get("open_lease_ids")
            or any(not j.get("released") for j in summary.get("allocation_journals", []))):
        raise ValueError("D703 physical lifecycle summary is incomplete")

    watchdog = [json.loads(line) for line in WATCHDOG.read_text().splitlines() if line.strip()]
    resources = [row for row in watchdog if row.get("event") == "resource_sample"]
    if not resources or any(row.get("decision", {}).get("abort_reasons") for row in resources):
        raise ValueError("watchdog has no clean resource samples")
    event_max = {key: max(row["service"]["memory.events"][key] for row in resources)
                 for key in ("high", "max", "oom", "oom_kill")}
    max_swap = max(row["service"]["memory.swap.current"] for row in resources)
    if any(event_max.values()) or max_swap != 0:
        raise ValueError(f"resource gate failed: {event_max}, swap={max_swap}")
    remote = remote_state()

    # D703 is a full native/SLO qualification, but numerical correctness is
    # intentionally not inferred from token-termination evidence.
    slo_count = round(float(scenario["slo_attainment"]) * 4000)
    audit = {
        "kind": "d703_3b_full_w0_d701profile_audit_v1",
        "campaign_id": "d703_3b_full_w0_d701profile",
        "passed": True,
        "full_replay_complete": True,
        "native_service_qualification": True,
        "numerical_adapter_correctness_qualified": False,
        "ranking_eligible": False,
        "n_plan": 4000,
        "n_terminal": 4000,
        "n_native_contract_complete": 4000,
        "n_correct": None,
        "joint_slo_count_observed": slo_count,
        "joint_slo_fraction_observed": scenario.get("slo_attainment"),
        "mean_ttft_ms": scenario.get("avg_ttft_ms"),
        "p95_ttft_ms": scenario.get("p95_ttft_ms"),
        "mean_tpot_ms": scenario.get("avg_tpot_ms"),
        "mean_e2e_ms": scenario.get("avg_e2e_ms"),
        "physical_gpu_seconds_observed": summary.get("gpu_seconds_observed"),
        "physical_gpu_seconds_per_offered_request": summary.get("gpu_seconds_per_offered_observed"),
        "request_tier_counts": dict(tiers),
        "service_classes_observed": len(service_classes),
        "service_classes_available": len(loaded_service.profiles),
        "preparation_classes_observed": len(preparation_classes),
        "preparation_classes_available": len(loaded_preparation.profiles),
        "unique_adapter_identities": len(adapters),
        "native_output_tokens": output_tokens,
        "remote_transfers": len(transfers),
        "remote_request_preparations": remote_rows,
        "request_time_pack_count": 0,
        "watchdog": {"samples": len(resources), "memory_events_max": event_max,
                     "max_swap_current": max_swap,
                     "service_memory_peak_bytes": max(row["service"]["memory.peak"] for row in resources)},
        "remote_post_run": remote,
        "source_trace_sha256": result["metadata"]["sampling_stats"]["shared_trace_metadata"]["replay_plan"]["source_sha256"],
        "source_result": ref(RESULT), "launch": ref(LAUNCH), "outcome": ref(OUTCOME),
        "summary": ref(SUMMARY), "physical_resource_summary": ref(PHYSICAL),
        "watchdog_raw": ref(WATCHDOG), "config": ref(CONFIG),
        "service_profile": ref(SERVICE_PROFILE), "preparation_profile": ref(PREPARATION_PROFILE),
        "profile_manifest": ref(PROFILE_MANIFEST), "plan_sha256": plan_sha,
        "metric_protocol_v1_sha256": digest(frozen_v1),
        "metric_protocol_v2_sha256": digest(single_run_v2),
        "interpretation": (
            "D703 closes a complete 4,000-request native service replay with the D701 "
            "exact measured profile and observed joint SLO, and improves the prior D682 "
            "diagnostic on physical GPU seconds and SLO. It does not infer numerical "
            "adapter correctness from fixed-length token termination; therefore it is not "
            "a formal G1 ranking point and must not be plotted as one until correctness "
            "is independently qualified."
        ),
    }
    OUT.mkdir(parents=False, exist_ok=False)
    with (OUT / "audit.json").open("x") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with (OUT / "manifest.json").open("x") as stream:
        json.dump({"kind": "d703_3b_full_w0_audit_manifest_v1", "passed": True,
                   "audit": ref(OUT / "audit.json"),
                   "raw_evidence": [ref(path) for path in required]},
                  stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"passed": True, "full_replay_complete": True,
                      "numerical_adapter_correctness_qualified": False,
                      "ranking_eligible": False, "requests": 4000,
                      "joint_slo_count_observed": slo_count,
                      "physical_gpu_seconds": summary.get("gpu_seconds_observed"),
                      "service_classes": len(service_classes),
                      "preparation_classes": len(preparation_classes),
                      "remote_transfers": len(transfers)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--calibrate-control-lower', type=Path)
    args = parser.parse_args()
    if args.calibrate_control_lower is not None:
        calibrate_control_lower(args.calibrate_control_lower)
    else:
        main()
