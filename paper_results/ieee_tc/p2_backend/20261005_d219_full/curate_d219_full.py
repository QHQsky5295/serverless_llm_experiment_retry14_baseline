"""Curate the D219 full replay with request-level and physical gates."""
import csv
import hashlib
import json
import sys
from pathlib import Path


root, result_json, output_dir, config, trace, subset, remote_journal = map(Path, sys.argv[1:])
raw = root / "launch.launch"


def load(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


result = load(result_json)
scenario = result["detailed_results"]["tc_ieee_full"]
requests = scenario["requests"]
metadata = result["metadata"]["scenario_coordination"]["tc_ieee_full"]
feature_activation = metadata.get("feature_activation", {})
launch = load(root / "launch.json")
main = load(raw / "physical_deployment/main_outcome.json")
physical = load(raw / "physical_deployment/summary.json")


def count(predicate):
    return sum(1 for request in requests if predicate(request))


def sources(request):
    snapshot = request.get("gpu_reference_evidence", {}).get("routing_snapshot", {}) or {}
    return list((snapshot.get("sources") or {}).values())


all_sources = [source for request in requests for source in sources(request)]
collections = [source.get("source_collection", {}) for source in all_sources]
dispatch_tiers = {}
for request in requests:
    tier = request.get("readiness_tier_before_dispatch")
    dispatch_tiers[tier] = dispatch_tiers.get(tier, 0) + 1

invariants = {
    "request_count": len(requests),
    "unique_request_ids": len({request.get("request_id") for request in requests}),
    "successful": count(lambda x: x.get("success") is True),
    "output_contract_match": count(lambda x: x.get("output_contract_match") is True),
    "confirmed_dispatch_snapshot": count(
        lambda x: x.get("gpu_reference_evidence", {}).get("confirmed_dispatch_snapshot") is True
    ),
    "dispatch_tier_present": count(lambda x: isinstance(x.get("readiness_tier_before_dispatch"), str)),
    "source_tier_equals_dispatch_tier": count(
        lambda x: x.get("readiness_tier_before_dispatch")
        == x.get("gpu_reference_evidence", {}).get("source_admission", {}).get("source", {}).get("tier")
    ),
    "selected_owner_matches_intent": count(
        lambda x: x.get("gpu_reference_evidence", {}).get("intent", {}).get("expected_owner_id")
        == x.get("gpu_reference_evidence", {}).get("receipt", {}).get("owner_id")
    ),
    "selected_epoch_not_older": count(
        lambda x: (
            isinstance(x.get("gpu_reference_evidence", {}).get("intent", {}).get("expected_epoch"), int)
            and isinstance(x.get("gpu_reference_evidence", {}).get("receipt", {}).get("epoch"), int)
            and x["gpu_reference_evidence"]["receipt"]["epoch"]
            >= x["gpu_reference_evidence"]["intent"]["expected_epoch"]
        )
    ),
    "stale_recheck_requests": count(
        lambda x: int(x.get("gpu_reference_evidence", {}).get("stale_rechecks", 0) or 0) > 0
    ),
    "stale_recheck_sum": sum(
        int(x.get("gpu_reference_evidence", {}).get("stale_rechecks", 0) or 0) for x in requests
    ),
    "prompt_hash_present": count(lambda x: isinstance(x.get("canonical_prompt_sha256"), str) and bool(x.get("canonical_prompt_sha256"))),
    "timing_contract_present": count(lambda x: x.get("timing_contract") == "ieee_tc_native_v1"),
}

source_evidence = {
    "source_rows": len(all_sources),
    "collection_ids": sorted({int(row["collection_id"]) for row in collections if "collection_id" in row}),
    "modes": sorted({row.get("mode") for row in collections}),
    "full_authoritative_read_sum": sum(int(row.get("full_authoritative_read_replica_count", 0) or 0) for row in collections),
    "full_fallback_sum": sum(int(row.get("full_fallback_replica_count", 0) or 0) for row in collections),
    "confirmed_snapshot_cache_hit_sum": sum(int(row.get("confirmed_snapshot_cache_hit_replica_count", 0) or 0) for row in collections),
    "cache_hits_cumulative_max": max((int(row.get("cache_hits", 0) or 0) for row in collections), default=0),
    "identity_only_sum": sum(int(row.get("identity_only", 0) or 0) for row in collections),
    "interpretation": "Per-collection fields are observed snapshots; cache_hits is cumulative and is not a hit rate or event total.",
}

remote_all = [json.loads(line) for line in remote_journal.read_text().splitlines() if line.strip()]
main_transfers = main.get("remote_transfers", [])
transfer_ids = {row.get("http_transfer_id") for row in main_transfers}
remote_joined = [row for row in remote_all if row.get("transfer_id") in transfer_ids]
if len(remote_joined) != len(main_transfers) or len({row.get("transfer_id") for row in remote_joined}) != len(remote_joined):
    raise RuntimeError("remote transfer journal did not join one-to-one")

journals = []
expected_events = ["acquire", "worker_spawn", "native_workers", "native_workers_exited", "release"]
for path in sorted((raw / "physical_allocations").glob("*.jsonl")):
    events = [json.loads(line)["event"] for line in path.read_text().splitlines() if line.strip()]
    journals.append({"path": str(path), "events": events, "event_order_valid": events == expected_events, "sha256": sha(path)})

metric_keys = (
    "avg_ttft_ms", "p95_ttft_ms", "p99_ttft_ms", "avg_overall_ttft_ms", "p95_overall_ttft_ms",
    "avg_service_ttft_ms", "p95_service_ttft_ms", "avg_e2e_ms", "p95_e2e_ms", "p99_e2e_ms",
    "avg_overall_e2e_ms", "p95_overall_e2e_ms", "avg_tpot_ms", "throughput_rps", "slo_attainment",
    "infra_gpu_seconds_total", "infra_cost_total_usd", "infra_ce", "monetary_ce", "completed", "failed",
)

curated = {
    "kind": "d219_confirmed_snapshot_cache_full_replay_curated_v1",
    "run_id": "d219_7b_confirmed_snapshot_cache_full_replay",
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
        "formal_comparison_reason": "single exploratory full replay; no paired held-out seed set",
    },
    "population_and_invariants": invariants,
    "dispatch_tier_counts_request_level": dispatch_tiers,
    "feature_activation": feature_activation,
    "source_snapshot_evidence": source_evidence,
    "descriptive_metrics": {key: scenario.get(key) for key in metric_keys},
    "physical_lifecycle": {
        "n_plan": physical.get("n_plan"), "n_terminal": physical.get("n_terminal"),
        "n_native_contract_complete": physical.get("n_native_contract_complete"),
        "n_correct": physical.get("n_correct"), "gpu_seconds_observed": physical.get("gpu_seconds_observed"),
        "gpu_seconds_per_offered_observed": physical.get("gpu_seconds_per_offered_observed"),
        "journals": journals,
    },
    "delivery": {
        "main_outcome_transfer_count": len(main_transfers), "remote_journal_line_count": len(remote_all),
        "joined_transfer_count": len(remote_joined),
        "joined_payload_bytes": sum(int(row.get("bytes_written", 0) or 0) for row in remote_joined),
        "delivery_modes": sorted({row.get("delivery_mode") for row in remote_joined}),
        "pack_performed": sorted({bool(row.get("pack_performed")) for row in remote_joined}),
    },
    "source_sha256": {
        "launch": sha(root / "launch.json"),
        "main_outcome": sha(raw / "physical_deployment/main_outcome.json"),
        "physical_summary": sha(raw / "physical_deployment/summary.json"),
        "request_result_json": sha(result_json),
        "remote_transfer_journal": sha(remote_journal),
        "config": sha(config), "trace": sha(trace), "adapter_subset": sha(subset),
    },
    "caveat": "D219 is a complete single-seed mechanism/full-replay diagnostic. It is not a formal cross-system or held-out multi-seed comparison.",
}

output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "d219_full_curated.json").write_text(json.dumps(curated, indent=2, sort_keys=True) + "\n")
with (output_dir / "d219_request_invariants.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=sorted(invariants)); writer.writeheader(); writer.writerow(invariants)
with (output_dir / "d219_dispatch_tiers.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["tier", "requests"]); writer.writeheader()
    writer.writerows({"tier": tier, "requests": count} for tier, count in sorted(dispatch_tiers.items(), key=lambda item: str(item[0])))
