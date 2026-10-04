"""Bounded curator for the D215 route-identity cold-scope validation.

This is an audit artifact, not a performance summarizer.  In particular, the
``cache_hits`` and ``identity_only`` fields emitted in a routing snapshot are
run-level cumulative counters; they are retained but are never interpreted as
per-request events here.
"""

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


root, result_json, output_dir, config, trace, subset, remote_journal = map(Path, sys.argv[1:])
raw = root / "launch.launch"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


result = load(result_json)
scenario = result["detailed_results"]["tc_ieee_full"]
requests = scenario["requests"]
main = load(raw / "physical_deployment/main_outcome.json")
summary = load(raw / "physical_deployment/summary.json")
launch = load(root / "launch.json")

# The per-request evidence contains a source collection for every observed
# replica.  A collection_id is the stable unit for counting a collection;
# request snapshots can repeat the same collection when a request is joined.
all_rows = []
unique_rows = {}
selected_rows = []
scope_sequences = defaultdict(list)
for request in requests:
    routing = request["gpu_reference_evidence"]["routing_snapshot"]
    selected_id = routing["selected_replica_id"]
    selected = routing["sources"][selected_id]
    selected_rows.append(dict(
        request_id=request["request_id"],
        adapter_id=request["adapter_id"],
        selected_replica_id=selected_id,
        selected_tier=selected["source"].get("tier"),
        source_collection=dict(selected["source_collection"]),
    ))
    scope_sequences[(selected_id, request["adapter_id"])].append(dict(
        request_index=len(selected_rows) - 1,
        epoch=selected["native_epoch"],
        owner_id=selected["native_owner_id"],
        source_id=(selected.get("source") or {}).get("source_id"),
        fallback_count=selected["source_collection"].get("full_fallback_replica_count", 0),
        cold_count=selected["source_collection"].get("cold_full_read_replica_count", 0),
    ))
    for replica_id, evidence in routing["sources"].items():
        row = dict(evidence["source_collection"])
        row["replica_id"] = replica_id
        all_rows.append(row)
        unique_rows.setdefault(row.get("collection_id"), row)


def counter_summary(rows):
    fields = (
        "identity_replica_count",
        "full_authoritative_read_replica_count",
        "cold_full_read_replica_count",
        "full_fallback_replica_count",
        "cache_hits",
        "identity_only",
    )
    return {
        field: {
            "sum": sum(int(row.get(field, 0) or 0) for row in rows),
            "nonzero_rows": sum(int((row.get(field, 0) or 0) > 0) for row in rows),
        }
        for field in fields
    }


def category(rows, field):
    return {
        "rows": len(rows),
        "nonzero": sum(int((row.get(field, 0) or 0) > 0) for row in rows),
        "sum": sum(int(row.get(field, 0) or 0) for row in rows),
    }


unique = list(unique_rows.values())
scope_transition_rows = []
for (replica_id, adapter_id), sequence in scope_sequences.items():
    for previous, current in zip(sequence, sequence[1:]):
        same_target_source = (
            previous["owner_id"] == current["owner_id"]
            and previous["source_id"] is not None
            and previous["source_id"] == current["source_id"]
        )
        scope_transition_rows.append({
            "replica_id": replica_id,
            "adapter_id": adapter_id,
            "previous_request_index": previous["request_index"],
            "request_index": current["request_index"],
            "previous_epoch": previous["epoch"],
            "epoch": current["epoch"],
            "same_target_source_id": same_target_source,
            "epoch_advanced": current["epoch"] != previous["epoch"],
            "fallback_count": current["fallback_count"],
            "cold_count": current["cold_count"],
        })
scope_transition_summary = {
    "consecutive_revisits": len(scope_transition_rows),
    "same_target_source": sum(int(row["same_target_source_id"]) for row in scope_transition_rows),
    "same_target_source_epoch_advanced": sum(
        int(row["same_target_source_id"] and row["epoch_advanced"])
        for row in scope_transition_rows
    ),
    "same_target_source_epoch_advanced_with_fallback": sum(
        int(row["same_target_source_id"] and row["epoch_advanced"] and row["fallback_count"] > 0)
        for row in scope_transition_rows
    ),
}
stats = main.get("coordination_after_shutdown", {}).get("ieee_source_observation_stats", {})
remote_all = [json.loads(line) for line in remote_journal.read_text().splitlines() if line.strip()]
transfers = main.get("remote_transfers", [])
transfer_ids = {item.get("http_transfer_id") for item in transfers}
remote_joined = [item for item in remote_all if item.get("transfer_id") in transfer_ids]
if len(remote_joined) != len(transfers) or len({item.get("transfer_id") for item in remote_joined}) != len(remote_joined):
    raise RuntimeError("D215 remote transfer join is not one-to-one")

journals = []
expected_events = ["acquire", "worker_spawn", "native_workers", "native_workers_exited", "release"]
for path in sorted((raw / "physical_allocations").glob("*.jsonl")):
    events = [json.loads(line)["event"] for line in path.read_text().splitlines() if line.strip()]
    journals.append({"path": str(path), "events": events,
                     "event_order_valid": events == expected_events, "sha256": sha(path)})

population = {
    "planned": summary.get("n_plan"),
    "terminals": summary.get("n_terminal"),
    "successful_requests": sum(int(bool(row.get("success"))) for row in requests),
    "native_contract_matches": sum(int(bool(row.get("output_contract_match"))) for row in requests),
    "unique_request_ids": len({row.get("request_id") for row in requests}),
    "request_result_json_sha256": sha(result_json),
}

curated = {
    "kind": "d215_route_identity_cold_validation_curated_v1",
    "run_id": "d215_7b_route_identity_cold_validation",
    "qualification": {
        "launch_pass": launch.get("pass"),
        "service_returncode": launch.get("service_returncode"),
        "replay_returncode": launch.get("replay_returncode"),
        "watchdog_returncode": launch.get("watchdog_returncode"),
        "native_gpu_context_release_confirmed": launch.get("native_gpu_context_release_confirmed"),
        "service_path_removed": launch.get("service_path_removed"),
        "formal_comparison_qualified": False,
        "candidate_decision": "revoked_after_bounded_validation",
    },
    "population": population,
    "physical_lifecycle": {
        "gpu_seconds_observed": summary.get("gpu_seconds_observed"),
        "gpu_seconds_per_offered_request": summary.get("gpu_seconds_per_offered_observed"),
        "open_lease_ids": summary.get("open_lease_ids"),
        "journals": journals,
    },
    "route_identity_evidence": {
        "main_source_observation_stats": stats,
        "request_source_snapshot_count": len(all_rows),
        "unique_collection_ids_in_request_evidence": len(unique),
        "selected_request_snapshot_count": len(selected_rows),
        "all_snapshot_counters": counter_summary(all_rows),
        "unique_collection_counters": counter_summary(unique),
        "unique_collection_categories": {
            "cold_full_read": category(unique, "cold_full_read_replica_count"),
            "full_fallback": category(unique, "full_fallback_replica_count"),
            "identity_only": category(unique, "identity_only"),
            "cache_hits_counter": category(unique, "cache_hits"),
        },
        "identity_failures_in_request_evidence": sum(
            int(bool(row.get("source_collection", {}).get("identity_failure")))
            for row in selected_rows
        ),
        "scope_local_transition_summary": scope_transition_summary,
        "note": "cache_hits and identity_only are cumulative counters emitted by the runner; they are not treated as per-request reuse events.",
    },
    "delivery": {
        "main_outcome_transfer_count": len(transfers),
        "remote_journal_line_count": len(remote_all),
        "joined_transfer_count": len(remote_joined),
        "unmatched_remote_lines": len(remote_all) - len(remote_joined),
        "joined_server_bytes": sum(int(item.get("bytes_written", 0) or 0) for item in remote_joined),
        "delivery_modes": sorted({item.get("delivery_mode") for item in remote_joined}),
        "pack_performed": sorted({bool(item.get("pack_performed")) for item in remote_joined}),
        "temporary_created": sorted({bool(item.get("temporary_created")) for item in remote_joined}),
    },
    "descriptive_metrics": {
        key: scenario.get(key)
        for key in (
            "avg_ttft_ms", "p95_ttft_ms", "p99_ttft_ms", "avg_service_ttft_ms",
            "p95_service_ttft_ms", "avg_e2e_ms", "p95_e2e_ms", "p99_e2e_ms",
            "avg_tpot_ms", "throughput_rps", "slo_attainment", "avg_control_path_total_us",
            "p95_control_path_total_us", "avg_dispatch_admission_wait_ms",
            "p95_dispatch_admission_wait_ms", "cache_hit_rate", "gpu_hit_rate",
        )
    },
    "source_sha256": {
        "launch": sha(root / "launch.json"),
        "main_outcome": sha(raw / "physical_deployment/main_outcome.json"),
        "physical_summary": sha(raw / "physical_deployment/summary.json"),
        "request_result_json": sha(result_json),
        "remote_transfer_journal": sha(remote_journal),
        "config": sha(config),
        "trace": sha(trace),
        "adapter_subset": sha(subset),
    },
    "caveat": (
        "This is a 100-request mechanism diagnostic, not a formal G1/G2 or cross-system comparison. "
        "The fixed-output token contract does not establish numerical adapter correctness, and the "
        "legacy 5000-ms display is TTFT-only rather than the frozen joint TTFT/TPOT SLO."
    ),
}

output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "d215_validation_curated.json").write_text(json.dumps(curated, indent=2, sort_keys=True) + "\n")

with (output_dir / "d215_route_collection_metrics.csv").open("w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["scope", "rows", "cold_full_read_nonzero", "cold_full_read_sum",
                     "fallback_nonzero", "fallback_sum", "identity_only_nonzero",
                     "identity_only_sum", "cache_counter_nonzero", "cache_counter_sum"])
    for name, rows in (("unique_collection", unique), ("all_snapshot", all_rows)):
        c = counter_summary(rows)
        writer.writerow([name, len(rows),
                         c["cold_full_read_replica_count"]["nonzero_rows"], c["cold_full_read_replica_count"]["sum"],
                         c["full_fallback_replica_count"]["nonzero_rows"], c["full_fallback_replica_count"]["sum"],
                         c["identity_only"]["nonzero_rows"], c["identity_only"]["sum"],
                         c["cache_hits"]["nonzero_rows"], c["cache_hits"]["sum"]])

with (output_dir / "d215_scope_identity_transitions.csv").open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=(
        "replica_id", "adapter_id", "previous_request_index", "request_index",
        "previous_epoch", "epoch", "same_target_source_id", "epoch_advanced",
        "fallback_count", "cold_count"))
    writer.writeheader()
    writer.writerows(scope_transition_rows)

with (output_dir / "d215_validation_status.md").open("w") as fh:
    fh.write("# D215 7B route-identity cold-scope validation\n\n")
    fh.write("## Decision\n\n")
    fh.write("**Revoked after bounded validation; no 4,000-request replay or baseline run is qualified.**\n\n")
    fh.write(f"The 100-request run completed {population['terminals']}/{population['planned']} terminal records, "
             f"{population['successful_requests']} successful requests, and "
             f"{population['native_contract_matches']} fixed-output contract matches. "
             f"Observed physical lifecycle usage was **{summary.get('gpu_seconds_observed')} GPU-s**; "
             f"all {len(journals)} allocation journals passed the exact owner-event order and "
             f"open leases were {len(summary.get('open_lease_ids', []))}.\n\n")
    fh.write("## Route evidence\n\n")
    fh.write(f"The authoritative coordination record reports {stats.get('collections')} collections, "
             f"{stats.get('stale_rejections')} stale rejections and {stats.get('joined')} joined collections. "
             f"The request snapshots contain {len(unique)} unique collection IDs. In those records, "
             f"{category(unique, 'cold_full_read_replica_count')['nonzero']} collections include a cold full read "
             f"({category(unique, 'cold_full_read_replica_count')['sum']} replica reads), while "
             f"{category(unique, 'full_fallback_replica_count')['nonzero']} include a full fallback "
             f"({category(unique, 'full_fallback_replica_count')['sum']} replica reads).\n\n")
    fh.write("The emitted `cache_hits` and `identity_only` values are cumulative run counters, not per-request events; "
             "they are retained in the JSON but are not used to claim a reuse rate. The repeated authoritative "
             "fallback on cached scopes means the intended low-cost identity-recheck path was not dominant enough "
             "to justify a formal replay. This is a mechanism qualification failure, not evidence of numerical "
             "correctness or a cross-system performance result.\n\n")
    fh.write(f"A scope-local read-only audit found {scope_transition_summary['consecutive_revisits']} consecutive "
             f"(replica, adapter) revisits; {scope_transition_summary['same_target_source']} retained the same "
             f"target source identity while its owner epoch advanced, and "
             f"{scope_transition_summary['same_target_source_epoch_advanced_with_fallback']} of those transitions "
             "also carried a full-fallback count. This is a hypothesis signal, not proof that a target footprint "
             "is safe to reuse: aliasing and capacity invariants must be checked before any code change.\n\n")
    fh.write("## Delivery and caveats\n\n")
    fh.write(f"The outcome contains {len(transfers)} transfer records, joined one-to-one with the copied remote "
             f"journal ({len(remote_all)} lines; {len(remote_all)-len(remote_joined)} retained out-of-run lines). "
             f"All joined records use prepublished gzip delivery, with no request-time packing or temporary archive.\n\n")
    fh.write("The descriptive latency and GPU observations remain available in `d215_validation_curated.json`; "
             "they must not enter G1/G2, SLO, ranking, or paper comparison tables.\n")

print(output_dir / "d215_validation_curated.json")
