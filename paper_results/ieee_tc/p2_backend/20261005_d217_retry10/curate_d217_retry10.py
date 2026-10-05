"""Curate the D217 confirmed-snapshot diagnostic without promoting it to a formal run.

This deliberately reports request-level invariants and mechanism counters as
observed.  It does not convert cumulative counters into event totals and does
not infer a causal comparison from this 100-request prefix.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path


root, result_json, output_dir, config, trace, subset, remote_journal = map(Path, sys.argv[1:])
raw = root / "launch.launch"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


result = load(result_json)
scenario = result["detailed_results"]["tc_ieee_full"]
requests = scenario["requests"]
launch = load(root / "launch.json")
main = load(raw / "physical_deployment/main_outcome.json")
physical = load(raw / "physical_deployment/summary.json")
route_audit = load(
    Path("paper_results/ieee_tc/p2_backend/20261005_d217_retry10/route_evidence/route_evidence_audit.json")
)


def count(predicate):
    return sum(1 for row in requests if predicate(row))


def evidence_rows(row):
    routing = row.get("gpu_reference_evidence", {}).get("routing_snapshot", {}) or {}
    return list((routing.get("sources") or {}).values())


all_sources = [source for row in requests for source in evidence_rows(row)]
source_collections = [source.get("source_collection", {}) for source in all_sources]
collection_ids = sorted({int(x["collection_id"]) for x in source_collections if "collection_id" in x})

remote_all = [json.loads(line) for line in remote_journal.read_text().splitlines() if line.strip()]
main_transfers = main.get("remote_transfers", [])
main_ids = {x.get("http_transfer_id") for x in main_transfers}
remote_joined = [x for x in remote_all if x.get("transfer_id") in main_ids]
if len(remote_joined) != len(main_transfers) or len({x.get("transfer_id") for x in remote_joined}) != len(remote_joined):
    raise RuntimeError("remote transfer journal did not join one-to-one with main outcome")


mechanism = scenario.get("feature_activation", {})
invariants = {
    "request_count": len(requests),
    "unique_request_ids": len({x.get("request_id") for x in requests}),
    "successful": count(lambda x: x.get("success") is True),
    "output_contract_match": count(lambda x: x.get("output_contract_match") is True),
    "confirmed_dispatch_snapshot": count(
        lambda x: x.get("gpu_reference_evidence", {}).get("confirmed_dispatch_snapshot") is True
    ),
    "selected_intent_receipt_owner_match": count(
        lambda x: (
            x.get("gpu_reference_evidence", {}).get("intent", {}).get("expected_owner_id")
            == x.get("gpu_reference_evidence", {}).get("receipt", {}).get("owner_id")
        )
    ),
    "selected_receipt_epoch_not_older": count(
        lambda x: (
            isinstance(x.get("gpu_reference_evidence", {}).get("intent", {}).get("expected_epoch"), int)
            and isinstance(x.get("gpu_reference_evidence", {}).get("receipt", {}).get("epoch"), int)
            and x["gpu_reference_evidence"]["receipt"]["epoch"]
            >= x["gpu_reference_evidence"]["intent"]["expected_epoch"]
        )
    ),
    "dispatch_tier_present": count(lambda x: isinstance(x.get("readiness_tier_before_dispatch"), str)),
    "source_tier_equals_dispatch_tier": count(
        lambda x: x.get("readiness_tier_before_dispatch")
        == x.get("gpu_reference_evidence", {}).get("source_admission", {}).get("source", {}).get("tier")
    ),
    "stale_recheck_requests": count(
        lambda x: int(x.get("gpu_reference_evidence", {}).get("stale_rechecks", 0) or 0) > 0
    ),
    "stale_recheck_sum": sum(
        int(x.get("gpu_reference_evidence", {}).get("stale_rechecks", 0) or 0) for x in requests
    ),
}

source_counter_summary = {
    "source_rows": len(source_collections),
    "collection_ids": collection_ids,
    "collection_count": len(collection_ids),
    "modes": sorted({x.get("mode") for x in source_collections}),
    "full_authoritative_read_replica_count_sum": sum(
        int(x.get("full_authoritative_read_replica_count", 0) or 0) for x in source_collections
    ),
    "full_fallback_replica_count_sum": sum(
        int(x.get("full_fallback_replica_count", 0) or 0) for x in source_collections
    ),
    "confirmed_snapshot_cache_hit_replica_count_sum": sum(
        int(x.get("confirmed_snapshot_cache_hit_replica_count", 0) or 0) for x in source_collections
    ),
    "cache_hits_cumulative_max": max(
        (int(x.get("cache_hits", 0) or 0) for x in source_collections), default=0
    ),
    "identity_only_sum": sum(int(x.get("identity_only", 0) or 0) for x in source_collections),
    "interpretation": (
        "cache_hits is a cumulative scope-local counter; the per-collection fields are emitted snapshots. "
        "All are retained as mechanism evidence and are not reported as per-request hit rates or event totals."
    ),
}

descriptive_keys = (
    "avg_ttft_ms", "p95_ttft_ms", "p99_ttft_ms", "avg_service_ttft_ms", "p95_service_ttft_ms",
    "avg_e2e_ms", "p95_e2e_ms", "p99_e2e_ms", "avg_tpot_ms", "throughput_rps",
    "slo_attainment", "avg_control_path_total_us", "p95_control_path_total_us",
    "avg_dispatch_admission_wait_ms", "p95_dispatch_admission_wait_ms", "cache_hit_rate",
    "gpu_hit_rate", "completed", "failed", "completed_requests", "failed_requests", "infra_gpu_seconds_total",
    "infra_cost_total_usd", "infra_ce", "monetary_ce",
)

curated = {
    "kind": "d217_confirmed_snapshot_cache_diagnostic_curated_v1",
    "run_id": "d217_7b_confirmed_snapshot_cache_validation_retry10",
    "qualification": {
        "launch_pass": launch.get("pass"),
        "service_returncode": launch.get("service_returncode"),
        "replay_returncode": launch.get("replay_returncode"),
        "watchdog_returncode": launch.get("watchdog_returncode"),
        "native_gpu_context_release_confirmed": launch.get("native_gpu_context_release_confirmed"),
        "service_path_removed": launch.get("service_path_removed"),
        "physical_measurement_complete": physical.get("measurement_complete"),
        "physical_open_lease_ids": physical.get("open_lease_ids"),
        "formal_comparison_qualified": False,
        "reason": "diagnostic prefix (100 requests), one seed, no paired formal comparison",
    },
    "population_and_invariants": invariants,
    "mechanism_feature_activation": mechanism,
    "route_snapshot_evidence": source_counter_summary,
    "route_audit": {
        "control_path": route_audit.get("control_path"),
        "conditional_request_groups": route_audit.get("conditional_request_groups"),
        "observed_collection_labels": route_audit.get("observed_collection_labels"),
        "source_snapshot_count": route_audit.get("source_snapshot_count"),
        "formal_comparison_qualified": route_audit.get("formal_comparison_qualified"),
        "caveat": route_audit.get("caveat"),
    },
    "descriptive_metrics": {key: scenario.get(key) for key in descriptive_keys},
    "physical_lifecycle": {
        "gpu_seconds_observed": physical.get("gpu_seconds_observed"),
        "gpu_seconds_per_offered_request": physical.get("gpu_seconds_per_offered_observed"),
        "n_plan": physical.get("n_plan"),
        "n_terminal": physical.get("n_terminal"),
        "n_correct": physical.get("n_correct"),
    },
    "delivery": {
        "main_outcome_transfer_count": len(main_transfers),
        "remote_journal_line_count": len(remote_all),
        "joined_transfer_count": len(remote_joined),
        "joined_payload_bytes": sum(int(x.get("bytes_written", 0) or 0) for x in remote_joined),
        "delivery_modes": sorted({x.get("delivery_mode") for x in remote_joined}),
        "pack_performed": sorted({bool(x.get("pack_performed")) for x in remote_joined}),
    },
    "source_sha256": {
        "launch": sha(root / "launch.json"),
        "main_outcome": sha(raw / "physical_deployment/main_outcome.json"),
        "physical_summary": sha(raw / "physical_deployment/summary.json"),
        "request_result_json": sha(result_json),
        "route_audit": sha(Path("paper_results/ieee_tc/p2_backend/20261005_d217_retry10/route_evidence/route_evidence_audit.json")),
        "remote_transfer_journal": sha(remote_journal),
        "config": sha(config),
        "trace": sha(trace),
        "adapter_subset": sha(subset),
    },
    "caveat": (
        "The run demonstrates native output-contract completion and selected owner/epoch checks for the "
        "100-request diagnostic prefix. It does not establish a 4,000-request performance result, a "
        "paired improvement over D215/D216, or G1/G2 qualification."
    ),
}

output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "d217_retry10_curated.json").write_text(json.dumps(curated, indent=2, sort_keys=True) + "\n")
with (output_dir / "d217_request_invariants.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=sorted(invariants))
    writer.writeheader()
    writer.writerow(invariants)
with (output_dir / "d217_control_path_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
    writer.writeheader()
    for key, value in route_audit.get("control_path", {}).items():
        writer.writerow({"metric": key, "value": value})
