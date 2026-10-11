#!/usr/bin/env python3
"""Create a paper-facing diagnostic figure for the D290 bounded PrimeLoRA run.

This is deliberately a diagnostic, not a comparison plot: D290 is a valid
native-contract replay, but it does not qualify the frozen G1/G2 gates.  The
script derives all values from the immutable result JSON and records its SHA in
the manifest so the figure cannot silently drift from the experiment.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def percentile(values: list[float], q: float) -> float:
    values = sorted(float(v) for v in values)
    if not values:
        raise ValueError("cannot summarize an empty request population")
    # Frozen V1 Type-1 percentile; legacy runtime summaries use interpolation.
    idx = max(0, min(len(values) - 1, int(math.ceil(q * len(values))) - 1))
    return values[idx]


def load(path: Path) -> tuple[dict, dict]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if len(result["detailed_results"]) != 1:
        raise ValueError("select a single scenario explicitly before plotting")
    scenario = next(iter(result["detailed_results"].values()))
    requests = scenario["requests"]
    if (not requests or scenario.get("failed") != 0
            or len(requests) != scenario.get("completed")
            or len(requests) != result["metadata"]["total_requests"]
            or not all(r.get("success") is True for r in requests)):
        raise ValueError("incomplete or failed replay: report a status table instead")
    request_vectors(requests)
    return result, scenario


def request_vectors(requests: list[dict]) -> tuple[list, list, list]:
    """Validate the saved timing identity; dispatch wait already includes lateness."""
    vectors = ([], [], [])
    seen = set()
    for r in requests:
        request_id = r["request_id"]
        if not request_id or request_id in seen:
            raise ValueError("missing or duplicate request identity")
        seen.add(request_id)
        if (r.get("generation_contract") != "fixed_length_greedy_v1"
                or r.get("output_contract_match") is not True
                or r.get("completion_token_source") != "vllm_token_ids"
                or r.get("completion_tokens") != r.get("requested_completion_tokens")
                or not r.get("completion_tokens")):
            raise ValueError("native output contract is not complete")
        tier = r.get("readiness_tier_before_dispatch")
        if not isinstance(tier, str) or tier.lower() not in {"gpu", "host", "nvme", "remote"}:
            raise ValueError("missing/unknown dispatch tier")
        values = []
        for key in ("dispatch_admission_wait_ms", "service_ttft_ms", "overall_ttft_ms"):
            v = r[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"invalid {key}")
            values.append(float(v))
        if abs(values[0] + values[1] - values[2]) > 1.0:
            raise ValueError("TTFT decomposition exceeds the frozen 1-ms tolerance")
        for vector, value in zip(vectors, values):
            vector.append(value)
    if not requests:
        raise ValueError("empty request population")
    return vectors


def audit_elastic_locality_mechanisms(outcome: dict, requests: list[dict]) -> dict:
    """Require actual negative/positive mechanism evidence, never just a label.

    This augments the common timing/source/lifecycle audit; it does not by itself
    certify output arithmetic, a complete workload, SLO or system superiority.
    """
    if outcome['scenario'] != 'tc_ieee_elastic_locality' or not requests:
        raise ValueError('complete ElasticLocality identity and request population required')
    metrics = outcome['coordination_after_shutdown']
    contract = metrics['ieee_mechanism_contract']
    if (contract != outcome['launch_contract']['system_mechanism_contract']
            or contract['system'] != 'ElasticLocality'
            or contract['variant'] != 'elastic_locality_v1'
            or contract['proactive_preparation'] is not False
            or contract['native_demand_loading'] is not True
            or contract['physical_admission'] is not True
            or contract['selection_strategy'] != 'gpu_locality'
            or contract['file_cache_policy'] != 'demand_lru'
            or contract['handoff_policy'] != 'no_handoff'):
        raise ValueError('ElasticLocality mechanism identity mismatch')
    selection = dict(schema='owned_routing_selection_v1',owner_policy='ieee_confirmed',
        strategy='gpu_locality',locality_load_slack=contract['locality_load_slack'])
    if type(selection['locality_load_slack']) is not int or selection['locality_load_slack'] < 0:
        raise ValueError('invalid locality load slack')
    summary_selection = metrics['ieee_routing_selection']
    if any(summary_selection[key] != value for key,value in selection.items()):
        raise ValueError('actual routing policy differs from launch')
    reasons = summary_selection['locality_attempt_reasons']
    if (not reasons or set(reasons) - {'gpu_hit','load_slack_exceeded',
            'no_feasible_gpu_hit','empty_feasible_set'}
            or any(type(value) is not int or value < 0 for value in reasons.values())
            or sum(reasons.values()) < len(requests)):
        raise ValueError('missing or invalid actual locality attempts')
    for row in requests:
        if row['gpu_reference_evidence']['routing_snapshot']['selection_contract'] != selection:
            raise ValueError('request uses a different routing policy')
    events = outcome['mechanism_events']
    for key in ('_ieee_residency_epochs','_ieee_file_preparation_plans',
                '_ieee_gpu_preparation_plans','_ieee_native_host_preparations',
                '_ieee_proactive_diagnostic'):
        if events[key] is not None and events[key] != []:
            raise ValueError('unexpected proactive mechanism: ' + key)
    if outcome['planning_cpu_events'] != []:
        raise ValueError('proactive planning CPU work occurred')
    activations = events['_ieee_activations']
    if not activations or any(row['policy'] != 'no_handoff'
            or row['plan_sha256'] is not None or row['preparation_state'] != 'disabled'
            for row in activations):
        raise ValueError('activation preparation was not disabled')
    cache = metrics['ieee_file_cache']
    if cache['file_cache_policy'] != 'demand_lru' or cache['plans'] != []:
        raise ValueError('file cache is not demand-only LRU')
    if any(row['policy'] != 'demand_lru' or row['state'] != 'incoming_allocated'
           for row in cache['replacements']):
        raise ValueError('unexpected or failed file replacement')
    return dict(kind='elastic_locality_mechanism_audit_v1',passed=True,
        requests=len(requests),selection_attempt_reasons=dict(reasons),
        activations=len(activations),file_replacements=len(cache['replacements']),
        proactive_plans=0,numerical_correctness_qualified=False,
        performance_qualified=False)


def audit_run(path: Path) -> tuple[dict, dict, list[dict]]:
    result, scenario = load(path)
    requests = scenario["requests"]
    queue, service, overall = request_vectors(requests)
    ttft_errors, tpot_errors = [], []
    for row in requests:
        timing = row["native_token_timing"]
        first, last = (timing["native_first_token_monotonic_s"],
                       timing["native_last_token_monotonic_s"])
        arrival = row["external_arrival_timing"]["planned_arrival_s"]
        ttft_errors.append(abs((first - arrival) * 1000 - row["overall_ttft_ms"]))
        if row["completion_tokens"] >= 2:
            tpot_errors.append(abs((last - first) * 1000 /
                                  (row["completion_tokens"] - 1) - row["tpot_ms"]))
    if max(ttft_errors) > 1 or not tpot_errors or max(tpot_errors) > 1:
        raise ValueError("native TTFT/TPOT clock audit failed")
    physical_path = path.parent / "launch.launch/physical_deployment/physical_resource_summary.json"
    physical = json.loads(physical_path.read_text())
    if not physical["measurement_complete"] or physical["open_lease_ids"]:
        raise ValueError("physical lifecycle is incomplete")
    if physical["result_file"]["sha256"] != sha256(path):
        raise ValueError("physical sidecar belongs to a different result")
    stats = {
        "completed": len(requests), "mean_ttft_ms": mean(overall),
        "p95_ttft_ms": percentile(overall, .95),
        "mean_queue_ms": mean(queue), "p95_queue_ms": percentile(queue, .95),
        "mean_service_ms": mean(service), "p95_service_ms": percentile(service, .95),
        "mean_tpot_ms": mean([r["tpot_ms"] for r in requests if r["completion_tokens"] >= 2]),
        "p95_tpot_ms": percentile([r["tpot_ms"] for r in requests if r["completion_tokens"] >= 2], .95),
        "initial_preload_wall_s": scenario["initial_preload_wall_s"],
        "physical_gpu_seconds": physical["physical_gpu_seconds_total"],
        "n_correct": physical["n_correct"],
    }
    evidence = {
        "source": str(path), "source_sha256": sha256(path),
        "physical_source": str(physical_path), "physical_sha256": sha256(physical_path),
        "max_native_ttft_error_ms": max(ttft_errors),
        "max_native_tpot_error_ms": max(tpot_errors),
        "shared_trace_sha256": result["metadata"]["shared_trace_sha256"],
        "shared_adapter_subset_sha256": result["metadata"]["shared_adapter_subset_sha256"],
    }
    if scenario.get('scenario_name') == 'tc_ieee_elastic_locality':
        outcome_path = path.parent/'launch.launch/physical_deployment/main_outcome.json'
        outcome = json.loads(outcome_path.read_text())
        evidence['elastic_locality'] = audit_elastic_locality_mechanisms(outcome,requests)
        evidence['mechanism_source_sha256'] = sha256(outcome_path)
    return stats, evidence, requests


def compare_diagnostics(reference: Path, candidate: Path, out_dir: Path) -> None:
    left, le, lrows = audit_run(reference)
    right, re, rrows = audit_run(candidate)
    same_fields = ("request_id", "adapter_id", "scheduled_arrival_offset_s",
                   "source_expected_output_tokens", "requested_completion_tokens",
                   "canonical_prompt_sha256", "generation_contract", "completion_token_source")
    same_inputs = len(lrows) == len(rrows) and all(
        all(a[k] == b[k] for k in same_fields) for a, b in zip(lrows, rrows))
    same_inputs = same_inputs and all(le[k] == re[k] for k in (
        "shared_trace_sha256", "shared_adapter_subset_sha256"))
    if not same_inputs:
        raise ValueError("diagnostic comparison input identities differ")
    with (out_dir / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "reference", "candidate", "lower_better_change_percent"], lineterminator="\n")
        writer.writeheader()
        for key in left:
            a, b = left[key], right[key]
            delta = ((a - b) / a * 100 if a is not None and b is not None
                     and a != 0 and key not in {"completed", "n_correct"} else None)
            writer.writerow(dict(metric=key, reference=a, candidate=b, lower_better_change_percent=delta))
    changes = [{"request_id": a["request_id"], "adapter_id": a["adapter_id"],
                "target_tokens": a["requested_completion_tokens"],
                "reference_sha256": a["completion_token_ids_sha256"],
                "candidate_sha256": b["completion_token_ids_sha256"]}
               for a, b in zip(lrows, rrows)
               if a["completion_token_ids_sha256"] != b["completion_token_ids_sha256"]]
    (out_dir / "comparison_manifest.json").write_text(json.dumps({
        "reference": le, "candidate": re, "input_identity_equal": same_inputs,
        "input_fields_checked": same_fields, "output_hash_changes": changes,
        "formal_qualification": False, "causal_speedup_claim": False,
        "replicates_per_configuration": 1, "confidence_interval": None,
        "limitations": ["100-request prefix only", "file cache not randomized",
                        "output hash differences remain unaudited", "joint SLO/reference budget not frozen"],
        "script_sha256": sha256(Path(__file__)),
    }, indent=2) + "\n")


def binding_checks(row: dict, expected_content: str) -> dict[str, bool]:
    """Audit saved control-plane witnesses, never claim kernel arithmetic proof."""
    e, n = row["gpu_reference_evidence"], row["native_token_timing"]
    intent, receipt, release = e["intent"], e["receipt"], e["release_receipt"]
    post = receipt["post_load_source_snapshot"]
    aid, slot = receipt["adapter_int_id"], receipt["slot"]
    sources = [s for s in post["sources"] if s["adapter_int_id"] == aid]
    selected = sources[0] if len(sources) == 1 else {}
    return {
        "request_adapter_binding": row["adapter_id"] == intent["lora_name"] == receipt["lora_name"] == selected.get("adapter_id"),
        "owner_binding": intent["expected_owner_id"] == receipt["owner_id"] == post["owner_id"] == n["gpu_reference_owner_id"] == release["owner_id"],
        "lease_binding": intent["lease_id"] == receipt["lease_id"] == n["gpu_reference_lease_id"],
        "native_adapter_binding": intent["adapter_int_id"] == aid == n["gpu_reference_adapter_int_id"],
        "slot_binding": (type(slot) is int and 0 <= slot < len(post["slot_adapter_ids"])
                         and post["slot_adapter_ids"][slot] == aid and selected.get("gpu_slot") == slot),
        "frozen_content_binding": (expected_content == intent["expected_content_sha256"]
                                   == receipt["expected_content_sha256"] == selected.get("content_sha256")),
        "acquired_and_released": receipt["acquired"] is True and release["released"] is True and e["state"] == "released",
        "common_native_clock": receipt["clock_id"] == post["clock_id"] == n["native_clock_id"],
        "acquired_before_dispatch": receipt["acquired_monotonic_s"] <= n["native_dispatch_monotonic_s"],
        "token_digest_transport": row["completion_token_ids_sha256"] == n["completion_token_ids_sha256"],
    }


def write_binding_audit(reference: Path, candidate: Path, index_path: Path, out_dir: Path) -> None:
    # Independent canonicalization of the frozen file-list format; no live
    # filesystem/remote read, and no reliance on the receipt's expected hash.
    index = json.loads(index_path.read_text())
    expected = {a["id"]: hashlib.sha256(json.dumps(
        sorted(a["files"], key=lambda f: f["path"]), sort_keys=True,
        separators=(",", ":")).encode()).hexdigest() for a in index["artifacts"]}
    if len(expected) != len(index["artifacts"]):
        raise ValueError("duplicate frozen adapter ID")
    audits, by_run = {}, {}
    for label, path in (("reference", reference), ("candidate", candidate)):
        _, scenario = load(path)
        rows = scenario["requests"]
        by_run[label] = {r["request_id"]: r for r in rows}
        checks = [dict(request_id=r["request_id"], adapter_id=r["adapter_id"],
                       checks=binding_checks(r, expected[r["adapter_id"]])) for r in rows]
        audits[label] = dict(source=str(path), sha256=sha256(path), rows=checks,
                             control_binding_pass_count=sum(all(r["checks"].values()) for r in checks))
    if by_run["reference"].keys() != by_run["candidate"].keys():
        raise ValueError("unmatched request populations")
    changes = []
    for rid, left in by_run["reference"].items():
        right = by_run["candidate"][rid]
        if left["completion_token_ids_sha256"] != right["completion_token_ids_sha256"]:
            changes.append(dict(request_id=rid, adapter_id=left["adapter_id"],
                native_prompt_hash_equal=left["native_token_timing"]["native_prompt_token_ids_sha256"]
                == right["native_token_timing"]["native_prompt_token_ids_sha256"],
                reference_tier=left["readiness_tier_before_dispatch"], candidate_tier=right["readiness_tier_before_dispatch"],
                first_token_divergence_position=None, native_logits_available=False,
                batch_membership_available=False, numerical_cause_proven=False))
    artifact = dict(kind="saved_native_binding_audit_v1", runs=audits, changed_requests=changes,
        content_index=str(index_path), content_index_sha256=sha256(index_path),
        script_sha256=sha256(Path(__file__)), numerical_qualification=False,
        formal_qualification=False, interpretation="control_witness_consistency_only_not_actual_kernel_arithmetic")
    (out_dir / "binding_audit.json").write_text(json.dumps(artifact, indent=2) + "\n")
    with (out_dir / "binding_status.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["run", "requests", "control_binding_pass", "numerical_qualified"])
        for label, audit in audits.items():
            writer.writerow([label, len(audit["rows"]), audit["control_binding_pass_count"], False])
    print(json.dumps({k: v["control_binding_pass_count"] for k, v in audits.items()}))


def write_numerical_repeat_audit(paths: list[Path], out_dir: Path) -> dict:
    """Bounded same-backend correctness table, never a performance figure."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.ieee_tc_preflight import (compare_native_sequences,
        numerical_repeat_indices, validate_native_execution_journal)
    if not 1 <= len(paths) <= 3 or len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("one to three distinct preregistered numerical blocks required")
    sources, rows, retained, comparisons = [], [], [], []
    config_sha = None
    selected = numerical_repeat_indices(100)
    for path in paths:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8') as stream:
            result = json.load(stream)
        uncompressed_hash = hashlib.sha256()
        with opener(path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                uncompressed_hash.update(chunk)
        launch_path = path.parent / "launch.json"
        launch = json.loads(launch_path.read_text())
        if (result.get("kind") != "backend_native_native_numerical_repeat_qualification_v1"
                or result.get("pass") is not True or result.get("stage") != "complete"
                or result.get("performance_sample") is not False
                or result.get("n_correct") is not None
                or result.get("selected_source_indices") != selected
                or result.get("shutdown_called") is not True
                or launch.get("pass") is not True or launch.get("service_returncode") != 0
                or launch.get("watchdog_returncode") != 0
                or launch.get("native_gpu_context_release_confirmed") is not True
                or launch.get("service_path_removed") is not True):
            raise ValueError("incomplete numerical measurement/cleanup; publish failure status separately")
        if config_sha is None:
            config_sha = result["config_sha256"]
        elif config_sha != result["config_sha256"]:
            raise ValueError("numerical blocks use different configurations")
        label = path.parent.name
        cases = {}
        for case in result["requests"]:
            key = case["arm"], case["source_index"]
            if (key in cases or case.get("pass") is not True
                    or case["actual_tokens"] != case["target_tokens"]
                    or case["release"].get("released") is not True
                    or case["timing"].get("native_terminal_observed") is not True):
                raise ValueError("incomplete or duplicate numerical case")
            compare_native_sequences(case, case)
            cases[key] = case
        if set(cases) != {(arm, i) for arm in ("single", "concurrent4") for i in selected}:
            raise ValueError("missing numerical cases")
        validations = []
        if [a["name"] for a in result["numerical_arms"]] != ["single", "concurrent4"]:
            raise ValueError("numerical arms differ from registration")
        for arm in result["numerical_arms"]:
            checked = validate_native_execution_journal(arm["journal"], arm["expected_leases"],
                result["workers_before"]["workers"][0], arm["scheduler_before"], arm["scheduler_after"])
            if checked != arm["validation"]:
                raise ValueError("retained execution validation differs from recomputation")
            sizes = sorted({len(e["observation"]["requests"]) for s in arm["journal"]
                            for e in s["events"] if e["kind"] == "forward_before"})
            if (sizes != arm["observed_batch_sizes"] or not sizes
                    or (arm["name"] == "single" and sizes != [1])
                    or (arm["name"] == "concurrent4" and max(sizes) <= 1)):
                raise ValueError("real batch coverage does not match diagnostic arm")
            slot_count = 0
            for wave in arm["waves"]:
                for boundary in ("slot_before", "slot_after"):
                    if not wave[boundary] or any(x.get("pass") is not True for x in wave[boundary]):
                        raise ValueError("slot-content observation is incomplete")
                    slot_count += len(wave[boundary])
            validations.append(dict(arm=arm["name"], observed_batch_sizes=sizes,
                execution=checked, slot_content_observations=slot_count))
        within = [dict(source_index=i, **compare_native_sequences(cases['single', i],
                    cases['concurrent4', i])) for i in selected]
        if within != result["single_vs_concurrent"]:
            raise ValueError("stored numerical comparison differs from recomputation")
        comparisons.append(dict(left=label + '/single', right=label + '/concurrent4', rows=within))
        rows.append(dict(block=label, completed=54, selected_requests=27,
            single_concurrent_exact=sum(x['exact_match'] for x in within),
            different=sum(not x['exact_match'] for x in within), performance_sample=False,
            formal_qualification=False))
        sources.append(dict(path=str(path.resolve()), sha256=sha256(path),
            uncompressed_sha256=uncompressed_hash.hexdigest(),
            launch_sha256=sha256(launch_path), config_sha256=config_sha,
            trace=result['trace'], execution_validations=validations))
        retained.append((label, cases))
        # Keep only the small sequences across blocks, not hundreds of MiB of
        # already-validated per-iteration journals while decoding the next run.
        del result, arm, wave
    for left_idx in range(len(retained)):
        for right_idx in range(left_idx + 1, len(retained)):
            left_label, left = retained[left_idx]
            right_label, right = retained[right_idx]
            for arm in ('single', 'concurrent4'):
                comparisons.append(dict(left=left_label + '/' + arm, right=right_label + '/' + arm,
                    rows=[dict(source_index=i, **compare_native_sequences(left[arm, i], right[arm, i]))
                          for i in selected]))
    artifact = dict(kind='controlled_native_numerical_audit_v1', blocks=len(paths),
        planned_blocks=3, provisional=len(paths) < 3, sources=sources, status=rows,
        comparisons=comparisons, n_correct=None, performance_sample=False,
        formal_qualification=False, script_sha256=sha256(Path(__file__)),
        metric_protocol_sha256=sha256(Path(__file__).resolve().parents[2] /
                                     'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md'))
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / 'numerical_audit.json').write_text(json.dumps(artifact, indent=2) + '\n')
    with (out_dir / 'status.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="d290")
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--binding-audit-only", action="store_true")
    parser.add_argument("--content-index", type=Path)
    parser.add_argument("--numerical-repeat-only", action="store_true")
    parser.add_argument("--additional-numerical-result", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.numerical_repeat_only:
        if args.binding_audit_only or args.reference_json or args.content_index or args.data_dir:
            parser.error("numerical audit must be a separate explicit analysis")
        artifact = write_numerical_repeat_audit([args.result_json] + args.additional_numerical_result,
                                               args.out_dir)
        print(json.dumps(artifact['status'], indent=2))
        return
    if args.additional_numerical_result:
        parser.error("additional numerical results require --numerical-repeat-only")
    if args.binding_audit_only:
        if not args.reference_json or not args.content_index:
            parser.error("binding audit requires --reference-json and --content-index")
        if args.out_dir.exists() and any(args.out_dir.iterdir()):
            raise ValueError("refusing to overwrite an existing audit directory")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_binding_audit(args.reference_json.resolve(), args.result_json.resolve(),
                            args.content_index.resolve(), args.out_dir.resolve())
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np

    # fontconfig knows the installed TNR files even if this environment's
    # matplotlib cache predates installation. Register, never substitute.
    for pattern in ("Times New Roman:style=Regular", "Times New Roman:style=Bold"):
        installed = subprocess.check_output(
            ["fc-match", "-f", "%{file}", pattern], text=True).strip()
        if font_manager.FontProperties(fname=installed).get_name() != "Times New Roman":
            raise ValueError("Times New Roman is not installed")
        font_manager.fontManager.addfont(installed)
    font_path = font_manager.findfont("Times New Roman", fallback_to_default=False)

    result_path = args.result_json.resolve()
    out_dir = args.out_dir.resolve()
    if not args.label.replace("_", "").isalnum():
        raise ValueError("label must contain only letters, numbers or underscores")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("refusing to overwrite an existing analysis directory")
    data_dir = args.data_dir.resolve() if args.data_dir else out_dir
    if data_dir != out_dir and data_dir.exists() and any(data_dir.iterdir()):
        raise ValueError("refusing to overwrite an existing data directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    result, scenario = load(result_path)
    requests = [r for r in scenario["requests"] if r.get("success")]

    # Dispatch/admission is d-a, NOT d-e: adding e-a again double counts it.
    queue, service, overall = request_vectors(requests)
    startup_affected = [
        float(r.get("overall_ttft_ms", 0.0) or 0.0)
        for r in requests
        if r.get("scaleup_affected") or r.get("on_scaleup_runtime")
    ]
    tiers = {}
    for r in requests:
        tier = str(r.get("readiness_tier_before_dispatch") or "unknown")
        tiers[tier] = tiers.get(tier, 0) + 1
    frontier_rows = 0
    cold_reads = 0
    for r in requests:
        evidence = r.get("gpu_reference_evidence") or {}
        routing = evidence.get("routing_snapshot") or {}
        sources = routing.get("sources") if isinstance(routing, dict) else None
        if isinstance(sources, dict):
            row_frontier = False
            for source in sources.values():
                collection = (source or {}).get("source_collection") if isinstance(source, dict) else None
                if not isinstance(collection, dict):
                    continue
                row_frontier = row_frontier or bool(collection.get("frontier_absence_replica_indexes"))
                cold_reads += int(collection.get("cold_full_read_replica_count", 0) or 0)
            frontier_rows += int(row_frontier)

    stage_rows = []
    for metric, values in (
        ("queue/admission", queue),
        ("service", service),
        ("overall TTFT", overall),
    ):
        stage_rows.append({
            "metric": metric,
            "mean_ms": mean(values) if values else 0.0,
            "p95_ms": percentile(values, 0.95),
            "n": len(values),
        })
    with (data_dir / "stage_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean_ms", "p95_ms", "n"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(stage_rows)

    # Fail closed on font substitution; retain the final 3.45-inch page width.
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "axes.titlesize": 10.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    colors = {"queue": "#D55E00", "service": "#0072B2", "overall": "#009E73"}
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.7))
    ax = axes[0]
    labels = ["Queue/\nadmission", "Service\nTTFT", "Overall\nTTFT"]
    means = [row["mean_ms"] / 1000 for row in stage_rows]
    p95s = [row["p95_ms"] / 1000 for row in stage_rows]
    x = np.arange(3)
    width = 0.34
    ax.bar(x - width / 2, means, width, color=[colors["queue"], colors["service"], colors["overall"]], label="mean")
    ax.bar(x + width / 2, p95s, width, facecolor="white", edgecolor="#333333", linewidth=0.85, hatch="//", label="P95")
    ax.set_xticks(x, labels)
    ax.set_ylabel("TTFT (s; lower is better)")
    ax.text(0.5, -0.34, "(a) Latency components", transform=ax.transAxes,
            ha="center", va="top", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper left", borderaxespad=0.2, handlelength=1.2, columnspacing=0.8)
    ymax = max(p95s) * 1.16 if p95s else 1.0
    ax.set_ylim(0, ymax)
    for xx, value in zip(x - width / 2, means):
        ax.text(xx, value + ymax * 0.018, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    counts = [len(startup_affected), frontier_rows, cold_reads]
    labels = ["scale-up\naffected", "frontier\nrows", "cold-read\nobservations"]
    bar_colors = ["#E69F00", "#56B4E9", "#CC79A7"]
    x = np.arange(3)
    bars = ax.bar(x, counts, color=bar_colors, width=0.62)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Count")
    ax.text(0.5, -0.34, "(b) Control-path evidence", transform=ax.transAxes,
            ha="center", va="top", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(counts + [1]) * 1.25)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(counts + [1]) * 0.035, str(value), ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.15, top=0.94, hspace=0.9)
    fig.text(0.58, 0.99, f"{args.label.upper()}: diagnostic, n={len(requests)}", ha="center", va="top", fontsize=9)
    figure_name = f"{args.label}_latency_control_diagnostic"
    fig.savefig(out_dir / f"{figure_name}.pdf")
    fig.savefig(out_dir / f"{figure_name}.png", dpi=300)
    plt.close(fig)

    manifest = {
        "figure": figure_name,
        "analysis_version": "dispatch_wait_no_double_count_v2",
        "script_sha256": sha256(Path(__file__)),
        "metric_protocol_sha256": sha256(Path(__file__).resolve().parents[2] / "docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md"),
        "font_file": font_path,
        "figure_size_inches": [3.45, 4.7],
        "source_result": str(result_path),
        "source_sha256": sha256(result_path),
        "scenario": scenario.get("scenario_name"),
        "completed": scenario.get("completed"),
        "failed": scenario.get("failed"),
        "mean_overall_ttft_ms": mean(overall),
        "p95_overall_ttft_ms": percentile(overall, 0.95),
        "mean_service_ttft_ms": mean(service),
        "p95_service_ttft_ms": percentile(service, 0.95),
        "quantile_definition": "type_1_ordered_ceil_p_times_n",
        "legacy_interpolated_p95_overall_ttft_ms": scenario.get("p95_overall_ttft_ms"),
        "legacy_interpolated_p95_service_ttft_ms": scenario.get("p95_service_ttft_ms"),
        "mean_queue_admission_ms": mean(queue) if queue else 0.0,
        "p95_queue_admission_ms": percentile(queue, 0.95),
        "scaleup_affected_count": len(startup_affected),
        "frontier_rows": frontier_rows,
        "cold_read_observations": cold_reads,
        "readiness_tier_counts": tiers,
        "formal_qualification": False,
        "numerical_correctness_qualified": False,
        "interpretation": "single_prefix_diagnostic_no_causal_or_G1_G2_claim",
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.reference_json:
        compare_diagnostics(args.reference_json.resolve(), result_path, data_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
