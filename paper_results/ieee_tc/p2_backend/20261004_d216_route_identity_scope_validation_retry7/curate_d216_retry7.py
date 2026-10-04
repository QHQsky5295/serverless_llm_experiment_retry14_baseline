"""Bounded curator for D216 retry7; never a formal comparison summarizer."""
import csv, hashlib, json, sys
from collections import Counter
from pathlib import Path

root, result_json, output_dir, config, trace, subset, remote_journal = map(Path, sys.argv[1:])
raw = root / "launch.launch"

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def load(path):
    return json.loads(path.read_text())

result = load(result_json)
scenario = result['detailed_results']['tc_ieee_full']
requests = scenario['requests']
main = load(raw / 'physical_deployment/main_outcome.json')
summary = load(raw / 'physical_deployment/summary.json')
launch = load(root / 'launch.json')
remote_all = [json.loads(x) for x in remote_journal.read_text().splitlines() if x.strip()]
main_transfers = main.get('remote_transfers', [])
transfer_ids = {x.get('http_transfer_id') for x in main_transfers}
remote_joined = [x for x in remote_all if x.get('transfer_id') in transfer_ids]
if len(remote_joined) != len(main_transfers) or len({x.get('transfer_id') for x in remote_joined}) != len(remote_joined):
    raise RuntimeError('remote transfer join is not one-to-one')

all_rows, unique_rows = [], {}
for request in requests:
    routing = request['gpu_reference_evidence']['routing_snapshot']
    for replica_id, evidence in routing['sources'].items():
        row = dict(evidence['source_collection'])
        row['replica_id'] = replica_id
        all_rows.append(row)
        unique_rows.setdefault(row.get('collection_id'), row)
unique = list(unique_rows.values())

def count_rows(rows, field):
    return {'rows': len(rows), 'nonzero': sum(int(bool(x.get(field, 0))) for x in rows),
            'sum': sum(int(x.get(field, 0) or 0) for x in rows)}

fields = ('identity_replica_count', 'full_authoritative_read_replica_count',
          'cold_full_read_replica_count', 'full_fallback_replica_count',
          'cache_hits', 'identity_only', 'joined')
all_counters = {f: count_rows(all_rows, f) for f in fields}
unique_counters = {f: count_rows(unique, f) for f in fields}
last_collection = max(all_rows, key=lambda x: int(x.get('collection_id', -1))) if all_rows else {}
scope_local_counter = {
    'source_field': 'cache_hits',
    'value_in_last_collection': int(last_collection.get('cache_hits', 0) or 0),
    'maximum_cumulative_value': max((int(x.get('cache_hits', 0) or 0) for x in all_rows), default=0),
    'interpretation': 'The runner increments cache_hits only at the scope-local reuse predicate; emitted values are cumulative, not per-request events.'
}

journals = []
expected = ['acquire', 'worker_spawn', 'native_workers', 'native_workers_exited', 'release']
for path in sorted((raw / 'physical_allocations').glob('*.jsonl')):
    events = [json.loads(x)['event'] for x in path.read_text().splitlines() if x.strip()]
    journals.append({'path': str(path), 'events': events,
                     'event_order_valid': events == expected, 'sha256': sha(path)})

comparisons = [
    {'run': 'D214', 'control_mean_us': 462790.392475, 'control_p95_us': None, 'source': 'D214 retry4 stage table'},
    {'run': 'D215', 'control_mean_us': 236003.72014, 'control_p95_us': 947055.56495, 'source': 'D215 curated validation'},
    {'run': 'D216 retry7', 'control_mean_us': scenario.get('avg_control_path_total_us'), 'control_p95_us': scenario.get('p95_control_path_total_us'), 'source': 'this run'},
]
for row in comparisons:
    ref = 236003.72014
    row['mean_change_vs_D215_lower_better'] = None if row['control_mean_us'] is None else (ref - row['control_mean_us']) / ref
    if row['control_p95_us'] is not None:
        row['p95_change_vs_D215_lower_better'] = (947055.56495 - row['control_p95_us']) / 947055.56495
    else:
        row['p95_change_vs_D215_lower_better'] = None

population = {
    'planned': summary.get('n_plan'), 'terminals': summary.get('n_terminal'),
    'successful_requests': sum(int(bool(x.get('success'))) for x in requests),
    'native_contract_matches': sum(int(bool(x.get('output_contract_match'))) for x in requests),
    'unique_request_ids': len({x.get('request_id') for x in requests}),
    'prompt_hashes': sum(int(bool(x.get('canonical_prompt_sha256'))) for x in requests),
    'dispatch_readiness_fields': sum(int('readiness_tier_before_dispatch' in x) for x in requests),
    'actual_equals_target': sum(int(x.get('output_tokens') == x.get('requested_completion_tokens')) for x in requests),
}
curated = {
    'kind': 'd216_route_identity_scope_validation_curated_v1',
    'run_id': 'd216_7b_route_identity_scope_validation_retry7',
    'qualification': {
        'launch_pass': launch.get('pass'), 'service_returncode': launch.get('service_returncode'),
        'replay_returncode': launch.get('replay_returncode'), 'watchdog_returncode': launch.get('watchdog_returncode'),
        'native_gpu_context_release_confirmed': launch.get('native_gpu_context_release_confirmed'),
        'service_path_removed': launch.get('service_path_removed'),
        'formal_comparison_qualified': False,
        'candidate_decision': 'mechanism_path_observed_mean_control_improvement_unresolved',
    },
    'population': population,
    'physical_lifecycle': {
        'gpu_seconds_observed': summary.get('gpu_seconds_observed'),
        'gpu_seconds_per_offered_request': summary.get('gpu_seconds_per_offered_observed'),
        'n_correct': summary.get('n_correct'), 'open_lease_ids': summary.get('open_lease_ids'),
        'journals': journals,
    },
    'route_identity_evidence': {
        'request_source_snapshot_count': len(all_rows),
        'unique_collection_ids_in_request_evidence': len(unique),
        'all_snapshot_counters': all_counters,
        'unique_collection_counters': unique_counters,
        'scope_local_counter_evidence': scope_local_counter,
        'stale_rechecks': sum(int((x.get('gpu_reference_evidence') or {}).get('stale_rechecks', 0) or 0) for x in requests),
        'note': 'cache_hits and identity_only are cumulative runner counters; they are not converted into a per-request hit rate. The scope-local predicate is identified from the source-level counter update and retained as mechanism evidence only.',
    },
    'delivery': {
        'main_outcome_transfer_count': len(main_transfers), 'remote_journal_line_count': len(remote_all),
        'joined_transfer_count': len(remote_joined), 'unmatched_remote_lines': len(remote_all) - len(remote_joined),
        'joined_server_bytes': sum(int(x.get('bytes_written', 0) or 0) for x in remote_joined),
        'delivery_modes': sorted({x.get('delivery_mode') for x in remote_joined}),
        'outcomes': sorted({x.get('outcome') for x in remote_joined}),
        'pack_performed': sorted({bool(x.get('pack_performed')) for x in remote_joined}),
        'temporary_created': sorted({bool(x.get('temporary_created')) for x in remote_joined}),
    },
    'descriptive_metrics': {k: scenario.get(k) for k in (
        'avg_ttft_ms', 'p95_ttft_ms', 'p99_ttft_ms', 'avg_service_ttft_ms', 'p95_service_ttft_ms',
        'avg_e2e_ms', 'p95_e2e_ms', 'p99_e2e_ms', 'avg_tpot_ms', 'throughput_rps', 'slo_attainment',
        'avg_control_path_total_us', 'p95_control_path_total_us', 'avg_dispatch_admission_wait_ms',
        'p95_dispatch_admission_wait_ms', 'cache_hit_rate', 'gpu_hit_rate', 'completed_requests', 'failed_requests')},
    'control_path_comparison': comparisons,
    'source_sha256': {
        'launch': sha(root / 'launch.json'), 'main_outcome': sha(raw / 'physical_deployment/main_outcome.json'),
        'physical_summary': sha(raw / 'physical_deployment/summary.json'), 'request_result_json': sha(result_json),
        'remote_transfer_journal': sha(remote_journal), 'config': sha(config), 'trace': sha(trace), 'adapter_subset': sha(subset),
    },
    'caveat': 'This is a 100-request mechanism diagnostic, not a formal G1/G2 or cross-system comparison. Mean control-path cost is not lower than D215 in this single run, although the P95 is lower; no 4,000-request replay is authorized by this artifact alone.',
}
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / 'd216_retry7_curated.json').write_text(json.dumps(curated, indent=2, sort_keys=True) + '\n')
with (output_dir / 'd216_route_collection_metrics.csv').open('w', newline='') as f:
    w=csv.writer(f); w.writerow(['scope','rows'] + list(fields))
    for name, rows in (('all_snapshot', all_rows), ('unique_collection', unique)):
        w.writerow([name, len(rows)] + [json.dumps(count_rows(rows, field), sort_keys=True) for field in fields])
with (output_dir / 'd216_control_path_comparison.csv').open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=sorted(comparisons[0])); w.writeheader(); w.writerows(comparisons)
