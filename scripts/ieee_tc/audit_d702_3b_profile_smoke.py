#!/usr/bin/env python3
"""Strict, append-only audit for the D702 3B profile qualification smoke.

This audit deliberately does not infer numerical adapter correctness or a
performance rank from a 100-request prefix.  It verifies the narrower claim
that the D701 measured profile can be loaded and used by the native PrimeLoRA
request path without a profile fallback, while the request/token/adapter,
remote-delivery, and resource-lifecycle contracts close.
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
RAW = Path("/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d702_20261019/3b_profile_smoke100")
RESULT = next(RAW.glob("experiment_results_full*.json"))
LAUNCH = RAW / "launch.json"
OUTCOME = RAW / "launch.launch/physical_deployment/main_outcome.json"
SUMMARY = RAW / "launch.launch/physical_deployment/summary.json"
PHYSICAL = RAW / "launch.launch/physical_deployment/physical_resource_summary.json"
WATCHDOG = RAW / "launch.launch/watchdog.jsonl"
CONFIG = ROOT / "configs/ieee_tc/20261019_d702_3b_profile_smoke100.yaml"
SERVICE_PROFILE = ROOT / "paper_results/ieee_tc/p2_backend/d701_3b_same_spec_profile/export/service.json"
PREPARATION_PROFILE = ROOT / "paper_results/ieee_tc/p2_backend/d701_3b_same_spec_profile/export/preparation.json"
PROFILE_MANIFEST = ROOT / "paper_results/ieee_tc/p2_backend/d701_3b_same_spec_profile/export/manifest.json"
OUT = ROOT / "paper_results/ieee_tc/p2_backend/d702_3b_profile_smoke100"
CAMPAIGN = "d702_3b_profile_smoke100"


def configure_campaign(campaign: str) -> None:
    """Reuse the sealed 100-request qualification rules for a registered run."""
    global RAW, RESULT, LAUNCH, OUTCOME, SUMMARY, PHYSICAL, WATCHDOG, CONFIG, OUT, CAMPAIGN
    if campaign == 'd702':
        return
    if campaign != 'd718':
        raise ValueError('only registered 3B smoke campaigns are supported')
    CAMPAIGN = 'd718_3b_identity_smoke100'
    RAW = Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d718_20261022/3b_identity_smoke100')
    results = list(RAW.glob('experiment_results_full*.json'))
    if len(results) != 1:
        raise ValueError('exactly one completed source result required')
    RESULT = results[0]
    LAUNCH = RAW / 'launch.json'
    OUTCOME = RAW / 'launch.launch/physical_deployment/main_outcome.json'
    SUMMARY = RAW / 'launch.launch/physical_deployment/summary.json'
    PHYSICAL = RAW / 'launch.launch/physical_deployment/physical_resource_summary.json'
    WATCHDOG = RAW / 'launch.launch/watchdog.jsonl'
    CONFIG = ROOT / 'configs/ieee_tc/20261022_d718_3b_identity_smoke100.yaml'
    OUT = ROOT / 'paper_results/ieee_tc/p2_backend/d718_3b_identity_smoke100'


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
    """Read-only post-run state; no service/configuration mutation is allowed."""
    cmd = [
        "ssh", "-o", "BatchMode=yes", "primelora-artifact-174",
        "systemctl --user show primelora-artifact-tc-3b.service "
        "-p ActiveState -p SubState -p MainPID -p Result",
    ]
    state = subprocess.run(cmd, check=True, capture_output=True, text=True)
    lines = dict(line.split("=", 1) for line in state.stdout.splitlines() if "=" in line)
    if lines.get("ActiveState") != "inactive" or lines.get("MainPID") != "0":
        raise ValueError(f"D702 remote 3B service is not closed: {lines}")
    link = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "primelora-artifact-174",
         "cat /sys/class/net/eno1/speed /sys/class/net/eno1/duplex /sys/class/net/eno1/carrier"],
        check=True, capture_output=True, text=True,
    )
    values = link.stdout.split()
    if len(values) != 3:
        raise ValueError(f"unexpected remote link witness: {link.stdout!r}")
    return dict(
        service_unit="primelora-artifact-tc-3b.service",
        service=lines,
        link=dict(interface="eno1", speed_mbps=int(values[0]), duplex=values[1], carrier=int(values[2])),
        command_contract="read_only_post_run_state_v1",
    )


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"append-only audit already exists: {OUT}")
    for path in (RESULT, LAUNCH, OUTCOME, SUMMARY, PHYSICAL, WATCHDOG,
                 CONFIG, SERVICE_PROFILE, PREPARATION_PROFILE, PROFILE_MANIFEST):
        if not path.exists():
            raise FileNotFoundError(path)

    # The current plan and frozen metric protocol are recorded, never replaced
    # by an audit-local definition.
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
    if len(rows) != 100:
        raise ValueError(f"D702 request count is {len(rows)}, expected 100")
    if (result["metadata"].get("total_requests") != 100
            or scenario.get("total") != 100
            or scenario.get("completed") != 100
            or scenario.get("failed") != 0):
        raise ValueError("D702 summary is not 100/100 complete")

    expected_service_sha = digest(SERVICE_PROFILE)
    expected_preparation_sha = digest(PREPARATION_PROFILE)
    if config["resource_coordination"]["ieee_service_profile"]["sha256"] != expected_service_sha:
        raise ValueError("D702 config/service-profile SHA mismatch")
    if config["resource_coordination"]["ieee_preparation_profile"]["sha256"] != expected_preparation_sha:
        raise ValueError("D702 config/preparation-profile SHA mismatch")
    if profile_manifest.get("development_only") is not True or profile_manifest.get("full_qualified") is not False:
        raise ValueError("D701 profile boundary was changed")

    # Re-run the exact frozen loader checks.  This rejects interpolation,
    # pooled classes and defaults before any request class is inspected.
    loaded_service = FrozenServiceProfiles.load(
        SERVICE_PROFILE, expected_sha256=expected_service_sha,
        model_config=service_payload["model_config"],
        expected_context=service_payload["context"], beta=0.5,
    )
    loaded_preparation = FrozenPreparationProfiles.load(
        PREPARATION_PROFILE, expected_sha256=expected_preparation_sha,
        model_config=preparation_payload["model_config"],
        expected_context=preparation_payload["context"], beta=0.5,
    )
    if len(loaded_service.profiles) != 160 or len(loaded_preparation.profiles) != 96:
        raise ValueError("D701 profile cardinality changed")

    tier_counts = collections.Counter()
    adapters = set()
    request_ids = set()
    remote_request_count = 0
    profile_class_hits = collections.Counter()
    for row in rows:
        rid = row.get("request_id")
        aid = row.get("adapter_id")
        evidence = row.get("gpu_reference_evidence") or {}
        intent = evidence.get("intent") or {}
        receipt = evidence.get("receipt") or {}
        source_admission = evidence.get("source_admission") or {}
        source = source_admission.get("source") or {}
        release = evidence.get("release_receipt") or {}
        timing = row.get("native_token_timing") or {}
        if (not isinstance(rid, str) or not rid or rid in request_ids
                or not isinstance(aid, str) or not aid):
            raise ValueError(f"invalid or duplicate request identity: {rid!r}, {aid!r}")
        request_ids.add(rid)
        adapters.add(aid)
        if not (row.get("success") is True and row.get("error") is None):
            raise ValueError(f"request {rid} did not succeed")
        target = min(int(row["source_expected_output_tokens"]), 256)
        if any((row.get(k) != target) for k in ("requested_completion_tokens", "completion_tokens", "output_tokens")):
            raise ValueError(f"request {rid} violates fixed output contract")
        if (int(timing.get("native_output_tokens", -1)) != target
                or timing.get("native_terminal_observed") is not True
                or row.get("completion_token_source") != "vllm_token_ids"
                or row.get("output_contract_match") is not True
                or row.get("generation_contract") != "fixed_length_greedy_v1"
                or not isinstance(row.get("completion_token_ids_sha256"), str)):
            raise ValueError(f"request {rid} lacks native fixed-length token evidence")
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
            raise ValueError(f"request {rid} used an unsupported service class")
        profile_class_hits["service"] += 1
        tier = source.get("tier")
        tier_counts[tier] += 1
        if tier != "gpu":
            prep_class = FrozenPreparationProfiles.source_class(
                source, loaded_preparation.size_edges_bytes)
            if prep_class not in loaded_preparation.profiles:
                raise ValueError(f"request {rid} used an unsupported preparation class")
            profile_class_hits["preparation"] += 1

        remote = evidence.get("remote_preparation")
        if remote is not None:
            remote_request_count += 1
            if (remote.get("remote_delivery_mode") != "prepublished_gzip_v1"
                    or remote.get("remote_pack_performed") is not False
                    or remote.get("content_verified") is not True
                    or remote.get("published_archive_verified") is not True):
                raise ValueError(f"request {rid} has invalid remote delivery evidence")

    if len(adapters) < 1:
        raise ValueError("no adapter identities observed")
    launch_contract = outcome.get("launch_contract") or {}
    if (launch_contract.get("service_profile_id") != expected_service_sha
            or launch_contract.get("preparation_profile_id") != expected_preparation_sha):
        raise ValueError("runtime launch did not bind the D701 profile identities")
    if launch_contract.get("replay_scope") != "diagnostic_prefix_v1":
        raise ValueError("D702 is not the registered diagnostic prefix")

    transfers = outcome.get("remote_transfers") or []
    if len(transfers) != result["metadata"].get("remote_artifact_transfer_count"):
        raise ValueError("remote transfer count mismatch")
    for transfer in transfers:
        if (transfer.get("state") != "published"
                or transfer.get("remote_delivery_mode") != "prepublished_gzip_v1"
                or transfer.get("remote_pack_performed") is not False
                or transfer.get("content_verified") is not True
                or transfer.get("published_archive_verified") is not True):
            raise ValueError("unverified or request-time-packed remote transfer")
    if result["metadata"].get("request_time_pack_count") != 0:
        raise ValueError("request-time packing was observed")

    if not (launch.get("pass") is True and launch.get("service_returncode") == 0
            and launch.get("watchdog_returncode") == 0
            and launch.get("replay_returncode") == 0
            and launch.get("service_path_removed") is True
            and launch.get("native_gpu_context_release_confirmed") is True):
        raise ValueError("D702 gated launch did not close cleanly")
    if (not outcome.get("run_completed") or outcome.get("error") is not None
            or outcome.get("cleanup_errors") or outcome.get("measurement_errors")
            or outcome.get("cleanup", {}).get("complete") is not True):
        raise ValueError("D702 outcome cleanup/measurement contract is incomplete")
    if (summary.get("measurement_complete") is not True
            or summary.get("n_plan") != 100 or summary.get("n_terminal") != 100
            or summary.get("n_native_contract_complete") != 100
            or summary.get("n_interrupted") != 0 or summary.get("open_lease_ids")
            or summary.get("eligible_correctness") is not False
            or summary.get("n_correct") is not None):
        raise ValueError("D702 physical lifecycle summary is not qualification-only")
    if any(not row.get("released") for row in summary.get("allocation_journals", [])):
        raise ValueError("a physical allocation journal remained open")

    watchdog = [json.loads(line) for line in WATCHDOG.read_text().splitlines() if line.strip()]
    resources = [row for row in watchdog if row.get("event") == "resource_sample"]
    if not resources or any(row.get("decision", {}).get("abort_reasons") for row in resources):
        raise ValueError("watchdog has no clean resource samples")
    event_max = {key: max(row["service"]["memory.events"][key] for row in resources)
                 for key in ("high", "max", "oom", "oom_kill")}
    max_swap = max(row["service"]["memory.swap.current"] for row in resources)
    if any(event_max.values()) or max_swap != 0:
        raise ValueError(f"D702 resource gate failed: {event_max}, swap={max_swap}")

    remote = remote_state()
    # Same retained-receipt checks used for the full D716/D717 diagnosis. A zero
    # trigger count is evidence of no coverage, never a claimed mechanism gain.
    from scripts.ieee_tc.analyze_d297_full import retained_preparation_row
    from scripts.ieee_tc.audit_d703_3b_full import retained_routing_attempts
    retries = []
    for row in rows:
        retained_preparation_row(row)
        retries.extend(retained_routing_attempts(row, clock_id=summary['clock_id']))
    OUT.mkdir(parents=False, exist_ok=False)
    audit = {
        "kind": f"{CAMPAIGN}_qualification_v1",
        "campaign_id": CAMPAIGN,
        "passed": True,
        "qualification_only": True,
        "ranking_eligible": False,
        "performance_4000_complete": False,
        "numerical_adapter_correctness_qualified": False,
        "profile_runtime_qualified": True,
        "profile_fallback_used": False,
        "n_plan": 100,
        "n_terminal": 100,
        "n_native_contract_complete": 100,
        "native_output_tokens": sum(int(row["completion_tokens"]) for row in rows),
        "request_tier_counts": dict(tier_counts),
        "routing_retries": {
            "attempts": len(retries),
            "requests": len({r['request_id'] for r in retries}),
            "reasons": dict(collections.Counter(r['reason'] for r in retries)),
            "repeated_absence_views": sum(r['repeated_absence_view'] for r in retries),
            "causal_speedup_qualified": False,
        },
        "profile_class_hits": dict(profile_class_hits),
        "unique_adapter_identities": len(adapters),
        "remote_transfers": len(transfers),
        "remote_request_preparations": remote_request_count,
        "request_time_pack_count": 0,
        "mean_ttft_ms": scenario.get("avg_ttft_ms"),
        "p95_ttft_ms": scenario.get("p95_ttft_ms"),
        "mean_tpot_ms": scenario.get("avg_tpot_ms"),
        "physical_gpu_seconds_observed": summary.get("gpu_seconds_observed"),
        "watchdog": {
            "samples": len(resources), "memory_events_max": event_max,
            "max_swap_current": max_swap,
            "service_memory_peak_bytes": max(row["service"]["memory.peak"] for row in resources),
        },
        "remote_post_run": remote,
        "source_trace_sha256": result["metadata"]["sampling_stats"]["shared_trace_metadata"]["replay_plan"]["source_sha256"],
        "source_result": ref(RESULT),
        "launch": ref(LAUNCH),
        "outcome": ref(OUTCOME),
        "summary": ref(SUMMARY),
        "physical_resource_summary": ref(PHYSICAL),
        "watchdog_raw": ref(WATCHDOG),
        "config": ref(CONFIG),
        "service_profile": ref(SERVICE_PROFILE),
        "preparation_profile": ref(PREPARATION_PROFILE),
        "profile_manifest": ref(PROFILE_MANIFEST),
        "plan_sha256": plan_sha,
        "metric_protocol_v1_sha256": digest(frozen_v1),
        "metric_protocol_v2_sha256": digest(single_run_v2),
        "interpretation": (
            f"{CAMPAIGN} establishes only that the D701 exact measured profiles are loadable and "
            "used for a clean 100-request PrimeLoRA prefix. It does not establish numerical "
            "adapter correctness, 4,000-request qualification, G1/G2, SLO attainment, or "
            "a comparison/ranking. The observed latency and GPU seconds are retained as a "
            "qualification observation and must not be plotted as a performance result."
        ),
    }
    with (OUT / "qualification.json").open("x") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with (OUT / "manifest.json").open("x") as stream:
        json.dump({"kind": f"{CAMPAIGN}_audit_manifest_v1",
                   "passed": True, "qualification": ref(OUT / "qualification.json"),
                   "raw_evidence": [ref(path) for path in
                                    (RESULT, LAUNCH, OUTCOME, SUMMARY, PHYSICAL, WATCHDOG,
                                     CONFIG, SERVICE_PROFILE, PREPARATION_PROFILE, PROFILE_MANIFEST)]},
                  stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"passed": True, "qualification_only": True,
                      "requests": 100, "service_classes": len(loaded_service.profiles),
                      "preparation_classes": len(loaded_preparation.profiles),
                      "remote_transfers": len(transfers), "remote_pack": False,
                      "physical_gpu_seconds": summary.get("gpu_seconds_observed")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--campaign', choices=['d702', 'd718'], default='d702')
    configure_campaign(parser.parse_args().campaign)
    main()
