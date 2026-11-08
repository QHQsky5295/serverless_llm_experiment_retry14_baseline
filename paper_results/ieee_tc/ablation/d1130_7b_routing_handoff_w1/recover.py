"""Recover D1130 observations from the retained failed outcome, not a clean run.

Uses D1113's D299 timeline/binding/SLO and closed-physical reducers. No synthetic
successful export, modified launch status, inferred correctness, or GPU replay.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import ijson

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
BASE = Path(__file__).resolve().parent
RAW = Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1130_20261108/7b_routing_handoff_w1')
DEPLOY = RAW/'launch.launch/physical_deployment'
SOURCE = DEPLOY/'main_outcome.json'
OUT = BASE/'recovery1'
sys.path.insert(0, str(ROOT))
from scripts import ieee_tc_preflight as p
from scripts.ieee_tc.check_d299_full_context import (
    closed_physical_integral, matched_controller_rows, common_slo_flags)
from scripts.ieee_tc.audit_d402_serverlessllm_path import timing_stats
from faaslora.metrics.metrics_collector import NativeV1TokenTimeline
from faaslora.datasets.workload_generator import FrozenReplayPlan

REFS = {}


def witness(path, expected=None):
    path = Path(path)
    digest = p.digest(path)
    if expected is not None:
        assert digest == expected, str(path)
    REFS[str(path)] = digest
    return path


def read(path, expected=None):
    return json.loads(witness(path, expected).read_text())


def item(prefix):
    with SOURCE.open('rb') as stream:
        return next(ijson.items(stream, prefix, use_float=True))


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    assert not OUT.exists()
    assert os.sched_getaffinity(0) == {2, 3, 26, 27}
    group = p.cg_path()
    assert (group/'memory.max').read_text().strip() == str(4*p.GIB)
    assert (group/'memory.swap.max').read_text().strip() == '0'
    pre = read(BASE/'prelaunch.json')
    launch = read(RAW/'launch.json')
    assert launch['pass'] is False and launch['classification'] == 'qualification_command_failure'
    assert launch['service_returncode'] == 1
    assert launch['replay_returncode'] == launch['watchdog_returncode'] == 0
    assert launch['native_gpu_context_release_confirmed'] and launch['service_path_removed']
    assert not Path(read(DEPLOY/'physical_resource_summary.json')['result_file']['path']).exists()
    witness(SOURCE)
    errors = item('cleanup_errors')
    assert len(errors) == 1 and errors[0]['stage'] == 'runtime_shutdown'
    assert 'os.pidfd_open' in errors[0]['traceback']
    assert item('error') is None and item('measurement_errors') == []
    assert item('interrupted_replays') == [] and item('run_completed') is False
    contract = item('launch_contract')
    ext = launch['external_replay']
    assert contract['input_plan'] == ext['plan']
    assert abs(ext['replay_t0_s']-ext['deployment_notice_s']-60) < 1e-6
    plan = FrozenReplayPlan.load(ext['plan']['source_path'], profile='W1', count=4000,
        rate_scale=ext['plan']['rate_scale'], adapter_map=ext['plan'].get('adapter_map'),
        adapter_map_sha256=ext['plan'].get('adapter_map_sha256'))
    assert plan.identity() == ext['plan']
    witness(plan.identity()['source_path'], plan.identity()['source_sha256'])
    index = read(ROOT/'paper_results/ieee_tc/inputs/20260927_7b_materialized_content_index.json')
    content = {a['id']: hashlib.sha256(json.dumps(sorted(a['files'], key=lambda f:f['path']),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest() for a in index['artifacts']}
    rows = matched_controller_rows(SOURCE, 'tc_ieee_routing_handoff', content, retained_outcome=True)
    assert set(rows) == {entry.request_id for entry in plan.entries}
    physical = read(DEPLOY/'physical_resource_summary.json')
    summary = read(DEPLOY/'summary.json', physical['summary_sha256'])
    gpu = closed_physical_integral(summary)
    assert gpu == physical['physical_gpu_seconds_total']
    for lease in summary['allocation_journals']:
        assert lease['released']
        witness(lease['path'], lease['sha256'])
    terminals_path = witness(summary['request_terminals_path'], summary['request_terminals_sha256'])
    terminals = [json.loads(line) for line in terminals_path.read_text().splitlines()]
    assert len(terminals) == 4000
    freeze = read(ROOT/'paper_results/ieee_tc/protocols/20261010_7b_slo_d394_frozen.json',
        '2d906f8aaf3fa1bf8cf30d116990db9970fb7be7bf1374fdbd1d8995090104f3')
    witness(ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md', freeze['metric_protocol_sha256'])
    witness(ROOT/'docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md', freeze['single_run_amendment_sha256'])
    witness(ROOT/freeze['calibration']['path'], freeze['calibration']['sha256'])
    counts, max_errors = Counter(), Counter()
    metrics = {k: [] for k in ('ttft_ms', 'tpot_ms', 'e2e_ms')}
    for entry in plan.entries:
        r, original = rows[entry.request_id], json.loads(entry.source_json)
        assert r['source_item_sha256'] == entry.source_sha256
        assert r['adapter_id'] == original['adapter_id']
        assert r['target_tokens'] == min(original['expected_output_tokens'], 256)
        assert abs(r['arrival']-ext['replay_t0_s']-entry.offset_s) < 1e-6
        n = r['native']
        NativeV1TokenTimeline.service_breakdown(n,
            admitted_at=n['controller_admitted_monotonic_s'], completed_at=r['completed'],
            clock_id=summary['clock_id'])
        first, last = n['native_first_token_monotonic_s'], n['native_last_token_monotonic_s']
        assert r['arrival'] <= r['submit'] <= r['native_entry_s'] <= first <= last <= r['completed']
        assert n['native_output_tokens'] == r['target_tokens'] and n['native_terminal_observed']
        values = dict(ttft_ms=1000*(first-r['arrival']),
            tpot_ms=1000*(last-first)/(r['target_tokens']-1) if r['target_tokens']>1 else None,
            e2e_ms=1000*(r['completed']-r['arrival']))
        for key, value in values.items():
            assert (r['stored'][key] is None if value is None else abs(value-r['stored'][key]) <= 1)
            if value is not None:
                metrics[key].append(value)
                max_errors[key] = max(max_errors[key], abs(value-r['stored'][key]))
        counts.update(common_slo_flags(freeze['groups'], prompt_tokens=r['native_prompt_tokens'],
            output_tokens=r['target_tokens'], ttft_ms=values['ttft_ms'], tpot_ms=values['tpot_ms']))
    health = read(RAW/'remote_pre_health.json')
    remote_path = witness(RAW/'remote_transfers.jsonl',
        'dca30d217864e9f5fb8081f660b8547d9a3d0be2e3b815cdb0451ba3b27876ec')
    remote_rows = [json.loads(line) for line in remote_path.read_text().splitlines()]
    remote = {r['transfer_id']: r for r in remote_rows}
    transfers = item('remote_transfers')
    assert len(remote) == len(remote_rows) == len(transfers)
    assert set(remote) == {r['http_transfer_id'] for r in transfers}
    for r in transfers:
        s = remote[r['http_transfer_id']]
        assert r['state'] == 'published' and r['content_verified'] and r['published_archive_verified']
        assert r['artifact_id'] == s['artifact_id'] and r['remote_archive_sha256'] == s['archive_sha256']
        assert r['transferred_bytes'] == s['archive_bytes'] == s['bytes_read'] == s['bytes_written']
        assert s['clock_id'] == health['clock_id'] and s['outcome'] == 'sent'
        assert not r['remote_pack_performed'] and not s['pack_performed'] and not s['temporary_created']
    samples, events = [], Counter()
    watchdog = witness(RAW/'launch.launch/watchdog.jsonl')
    for line in watchdog.read_text().splitlines():
        r = json.loads(line)
        events[r['event']] += 1
        if r['event'] == 'resource_sample':
            assert not r['decision']['abort_reasons']
            samples.append(r)
    assert events['gpu_terminal_census'] == events['service_domain_gone'] == 1
    mechanism = item('coordination_after_shutdown.ieee_mechanism_contract')
    assert mechanism['variant'] == 'routing_handoff_v1'
    assert mechanism['file_preparation_scope'] == 'handoff_only'
    assert mechanism['file_cache_policy'] == 'demand_lru'
    for path in (Path(__file__), ROOT/'scripts/ieee_tc/check_d299_full_context.py',
                 ROOT/'faaslora/metrics/metrics_collector.py', ROOT/'faaslora/linux_pidfd.py'):
        witness(path)
    report = dict(kind='d1130_failed_cleanup_recovered_observation_v1',
        run='D1130', model='Llama-2-7B', variant='RoutingHandoff', workload='W1',
        run_success=False, final_ranking_eligible=False, reuse_level_run='R3',
        source=str(SOURCE), source_sha256=REFS[str(SOURCE)], counts=dict(counts),
        requests=len(rows), n_correct=summary['n_correct'], independent_numerical_proof=False,
        metrics={k:timing_stats(v) for k,v in metrics.items()}, max_stored_error_ms=dict(max_errors),
        physical_gpu_seconds=gpu, gpu_seconds_per_offered_request=gpu/4000,
        physical_measurement_complete=True, window_gpu_seconds=summary['window_gpu_seconds'],
        closed_allocations=len(summary['allocation_journals']), cleanup_errors=errors,
        remote=dict(server_transfers=len(remote), wire_bytes=sum(r['bytes_written'] for r in remote_rows),
            clock=health['clock_id'], packed_in_request=False), actual_mechanism_contract=mechanism,
        resource=dict(samples=len(samples), events=dict(events),
            memory_events={k:max(r['service']['memory.events'][k] for r in samples)
                for k in ('high','max','oom','oom_kill')},
            max_swap_bytes=max(r['service']['memory.swap.current'] for r in samples)),
        source_refs=REFS, no_gpu_replay=True, confidence_interval=None,
        caveats=['Replay complete but service CPU cleanup failed; not a clean execution.',
            'All recovered latency and physical-resource observations are retained, not substituted for a successful export.',
            'Native binding/counts are not independent arithmetic proof; n_correct remains unknown.',
            'Only the affected execution may need a repaired attempt; completed Full/baselines are not rerun.'])
    OUT.mkdir()
    save(OUT/'recovery.json', report)
    save(OUT/'manifest.json', dict(sources=REFS, outputs={'recovery.json':p.digest(OUT/'recovery.json')}))
    print(json.dumps({k:report[k] for k in ('requests','counts','physical_gpu_seconds','metrics','run_success')}))


if __name__ == '__main__':
    main()
