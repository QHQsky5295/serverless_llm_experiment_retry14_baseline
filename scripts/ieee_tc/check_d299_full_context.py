"""Thin post-release full-run check reusing native audits and seal helpers.

No inference, figures, new metric definitions, or overwriting existing results.
The large outcome is projected through jq streaming, never loaded as a tree.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

from scripts import ieee_tc_preflight as p
from scripts.ieee_tc.plot_d290_diagnostic import audit_run, binding_checks, request_vectors, sha256
from faaslora.metrics.metrics_collector import NativeV1TokenTimeline

ROOT = p.ROOT
RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d299_20261007'
RUN = RAW/'7b_full_w0'
OUT = ROOT/'paper_results/ieee_tc/p2_backend/d299_full_w0/context1'
CURATED = ROOT/'paper_results/ieee_tc/p2_backend'
FIELDS = ('request_id','adapter_id','scheduled_arrival_offset_s',
          'source_expected_output_tokens','requested_completion_tokens',
          'canonical_prompt_sha256','generation_contract','completion_token_source')
META = ('run_completed','error','cleanup_errors','measurement_errors')
QUERY = ('select(length==2 and (((.[0]|length)==1 and '
         '(.[0][0]|IN("run_completed","error","cleanup_errors","measurement_errors"))) '
         'or ((.[0]|length)==3 and .[0][0]=="remote_transfers")))')


def read(path):
    return json.loads(path.read_text())


def project(outcome):
    raw = subprocess.check_output(['jq','-c','--stream',QUERY,str(outcome)],text=True)
    meta, indexed = {}, {}
    for line in raw.splitlines():
        path, value = json.loads(line)
        if len(path)==1:
            assert path[0] not in meta
            meta[path[0]]=value
        else:
            index, key = path[1:]
            assert type(index) is int and isinstance(key,str)
            row=indexed.setdefault(index,{})
            assert key not in row
            row[key]=value
    assert set(meta)==set(META)
    assert meta['run_completed'] is True and not meta['error']
    assert meta['cleanup_errors']==meta['measurement_errors']==[]
    assert indexed and sorted(indexed)==list(range(len(indexed)))
    return meta, [indexed[i] for i in range(len(indexed))]


def failed_run_partition(result):
    """Failure-only evidence path; never relax the all-success audit_run gate."""
    if len(result['detailed_results']) != 1:
        raise ValueError('one explicit scenario required')
    scenario = next(iter(result['detailed_results'].values()))
    rows = scenario['requests']
    n = result['metadata']['total_requests']
    if (type(n) is not int or n <= 0 or len(rows) != n
            or len({r['request_id'] for r in rows}) != n
            or any(not r['request_id'] or type(r['success']) is not bool for r in rows)):
        raise ValueError('complete unique offered terminal population required')
    good = [r for r in rows if r['success']]
    failed = [r for r in rows if not r['success']]
    if not failed or scenario['completed'] != len(good) or scenario['failed'] != len(failed):
        raise ValueError('failure-only accounting requires matching success/failure counts')
    return rows, good, failed


def complete_failure_output_hashes(directory):
    """Hash the two evidence payloads, never the manifest under construction."""
    return {name: sha256(directory/name) for name in ('context.json', 'failed_requests.json')}


def project_incomplete(outcome):
    """Bounded leaf projection; retain nested errors, never load request trees."""
    query = ('reduce (inputs | select(length==2 and (.[0][0] | '
             'IN("run_completed","error","cleanup_errors","measurement_errors",'
             '"planned_request_count","remote_transfers")))) as [$p,$v] '
             '({}; setpath($p;$v))')
    value = json.loads(subprocess.check_output(
        ['jq','-n','-c','--stream',query,str(outcome)],text=True))
    if value.get('run_completed') is not False or not isinstance(value.get('error'),dict):
        raise ValueError('explicit incomplete outcome with retained error required')
    return value


def audit_d311_incomplete():
    """Failure receipt only. No full-success, per-correct or latency ranking."""
    raw = ROOT/'results/ieee_tc/p2_backend_qualification/d311_20261007'
    run = raw/'7b_full_w0'
    physical_dir = run/'launch.launch/physical_deployment'
    out = CURATED/'d311_full_w0/failure_audit1'
    assert not out.exists()
    plan_sha = p.check_plan()
    assert plan_sha == '0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    protocol = ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md'
    assert sha256(protocol) == '5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    launch = read(run/'launch.json')
    assert sha256(run/'launch.json') == '0573b52eb763b157f0742cbdf5d1fccbd1541e88d77f0f8ee58fb268087e66af'
    assert launch['pass'] is False and launch['native_gpu_context_release_confirmed']
    assert launch['service_path_removed'] and launch['watchdog_returncode'] == 0
    physical = read(physical_dir/'physical_resource_summary.json')
    summary = read(physical_dir/'summary.json')
    assert physical['measurement_complete'] is False and not physical['open_lease_ids']
    assert physical['n_plan'] == 4000 and physical['n_terminal'] == 1640
    assert physical['n_native_contract_complete'] == 1640 and physical['n_correct'] is None
    assert physical['physical_gpu_seconds_total'] is None
    assert physical['summary_sha256'] == sha256(physical_dir/'summary.json')
    for entry in summary['allocation_journals']:
        assert entry['released'] and sha256(Path(entry['path'])) == entry['sha256']
    assert sha256(Path(summary['request_terminals_path'])) == summary['request_terminals_sha256']
    with (run/'launch.launch/replay.jsonl').open() as handle:
        replay = json.loads(next(reversed(handle.readlines())))
    assert replay['event'] == 'replay_incomplete' and replay['N_plan'] == 4000
    assert replay['N_arrived'] == 1738 and replay['N_submitted'] == 1732
    outcome = physical_dir/'main_outcome.json'
    projected = project_incomplete(outcome)
    assert projected['planned_request_count'] == 4000
    transfers = projected.pop('remote_transfers')
    journal = raw/'remote_transfers.jsonl'
    with journal.open() as handle:
        remote_rows = [json.loads(line) for line in handle]
    remote = {r['transfer_id']:r for r in remote_rows}
    assert len(remote) == len(remote_rows)
    matched = []
    for r in transfers:
        s = remote.get(r.get('http_transfer_id'))
        if s is not None:
            assert r['artifact_id'] == s['artifact_id']
            matched.append(s['transfer_id'])
        if r.get('state') == 'published':
            assert s and r['content_verified'] and r['published_archive_verified']
            assert r['remote_archive_sha256'] == s['archive_sha256']
            assert r['transferred_bytes'] == s['archive_bytes'] == s['bytes_read'] == s['bytes_written']
            assert s['outcome'] == 'sent'
        assert r.get('remote_pack_performed') is False
    assert all(not s['pack_performed'] and not s['temporary_created'] for s in remote_rows)
    samples, events = [], Counter()
    watchdog = run/'launch.launch/watchdog.jsonl'
    with watchdog.open() as handle:
        for line in handle:
            row = json.loads(line)
            events[row['event']] += 1
            if row['event'] == 'resource_sample':
                samples.append(row)
    assert samples and events['gpu_terminal_census'] == events['service_domain_gone'] == 1
    seal_path = ROOT/'paper_results/ieee_tc/safety/20260925_execution_start_protected.json'
    seal = p.verify_seal(seal_path)
    known = '/home/qhq/serverless_llm_baselines/scripts/replay_openai_trace.py'
    assert seal['changed'] == [known] and seal['entry_count'] == 147
    assert sha256(Path(known)) == '04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22'
    result = dict(kind='d311_incomplete_evidence_v1',plan_sha256=plan_sha,
        metric_protocol_sha256=sha256(protocol),replay_terminal=replay,
        physical=physical,terminal_meta=projected,launch_classification=launch['classification'],
        underlying_failure='native HOST file source acquisition: confirmed owner/epoch conflict',
        resources=dict(samples=len(samples),event_counts=dict(events),
            peak_bytes=max(r['service']['memory.peak'] for r in samples),
            min_host_available_bytes=min(r['host']['available_bytes'] for r in samples),
            max_swap_bytes=max(r['service']['memory.swap.current'] for r in samples),
            abort_reasons=[r['decision']['abort_reasons'] for r in samples if r['decision']['abort_reasons']],
            memory_events={k:max(r['service']['memory.events'][k] for r in samples)
                           for k in ('high','max','oom','oom_kill')}),
        remote=dict(client_records=len(transfers),server_records=len(remote_rows),
            matched_records=len(matched),client_states=dict(Counter(r['state'] for r in transfers)),
            unmatched_server_ids=sorted(set(remote)-set(matched)),
            published_wire_bytes=sum(r['transferred_bytes'] for r in transfers if r['state']=='published'),
            pack_performed=False),protection_raw=seal,n_correct=None,n_good=None,
        formal_qualification=False,ranking_eligible=False,no_figures=True,
        caveats=['Truncated observation: unarrived requests are not imputed as timeouts.',
                 'Native contract count is from the physical journal, not numerical qualification.',
                 'No whole-workload latency or per-correct GPU cost is computed.',
                 'The precise competing thread was not recorded; CPU reproduction is separate.'])
    paths = [outcome,run/'launch.json',watchdog,run/'launch.launch/replay.jsonl',
        run/'launch.launch/service.log',physical_dir/'physical_resource_summary.json',
        physical_dir/'summary.json',Path(summary['request_terminals_path']),journal,
        raw/'remote_monitor.log',raw/'remote_health.json',raw/'launch_identity.json',
        protocol,seal_path,Path(__file__)]
    out.mkdir(parents=True,exist_ok=False)
    with (out/'context.json').open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    with (out/'manifest.json').open('x') as handle:
        json.dump(dict(source_refs=[dict(path=str(f),sha256=sha256(f)) for f in paths],
            outputs={'context.json':sha256(out/'context.json')}),handle,indent=2)
    print(json.dumps(dict(output=str(out),resources=result['resources'],remote=result['remote'],
                          terminal_meta=projected),indent=2),flush=True)


def audit_d309_failure(campaign='d309'):
    """Preserve failed offered requests; no ranking, SLO qualification or figures."""
    cases = {
        'd309': ('811aadf1cc61d14f58b7ebf59a54f1406d05c8e161003f8fd6da947da662257b', 4040),
        'd313': ('091c7ffe720b413610a2a366ed4e18315ec9896c68f6f1080d279e913d229f7c', 4036),
    }
    expected_source_sha, expected_samples = cases[campaign]
    raw = ROOT/f'results/ieee_tc/p2_backend_qualification/{campaign}_20261007'
    run = raw/'7b_full_w0'
    revision = 'failure_audit2' if campaign == 'd313' else 'failure_audit1'
    out = CURATED/f'{campaign}_full_w0'/revision
    source = run/f'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_{campaign}_7b_full_w0.json'
    assert not out.exists()
    assert p.check_plan() == '0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    protocol = ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md'
    assert sha256(protocol) == '5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    receipt = read(raw/'completion_receipt.json')
    assert sha256(source) == receipt['source_sha256'] == expected_source_sha
    assert sha256(run/'launch.json') == receipt['launch_sha256']
    launch = read(run/'launch.json')
    assert launch['pass'] and launch['native_gpu_context_release_confirmed'] and launch['service_path_removed']
    assert launch['service_returncode'] == launch['watchdog_returncode'] == launch['replay_returncode'] == 0
    # Sequential reading bounds memory. D303 supplies only the common input map,
    # not a performance comparator or a source of replacement successful rows.
    prior = read(CURATED/'d303_full_w0/validated1/audit.json')['evidence']
    reference = Path(prior['source'])
    assert sha256(reference) == prior['source_sha256']
    data = read(reference)
    inputs = {r['request_id']: {k:r[k] for k in FIELDS[:-1]}
              for r in data['detailed_results']['tc_ieee_full']['requests']}
    del data
    gc.collect()
    data = read(source)
    rows, successful, failed = failed_run_partition(data)
    assert len(rows) == len(inputs) == 4000 and len(successful) == 3999 and len(failed) == 1
    assert all(n == data['metadata'][k] for k,n in prior.items()
               if k in ('shared_trace_sha256','shared_adapter_subset_sha256'))
    assert set(inputs) == {r['request_id'] for r in rows}
    for row in rows:
        assert all(row[k] == v for k,v in inputs[row['request_id']].items())
    for key, value in receipt['failure'].items():
        assert failed[0][key] == value
    request_vectors(successful)
    index_path = ROOT/'paper_results/ieee_tc/inputs/20260927_7b_materialized_content_index.json'
    index = read(index_path)
    expected = {a['id']: hashlib.sha256(json.dumps(sorted(a['files'],key=lambda f:f['path']),
        sort_keys=True,separators=(',',':')).encode()).hexdigest() for a in index['artifacts']}
    assert len(expected) == len(index['artifacts']) == 500
    deployment = read(run/'launch.launch/physical_deployment/deployment.json')
    counts, max_errors = Counter(), dict(ttft_ms=0.,tpot_ms=0.,stage_ms=0.)
    for r in successful:
        checks = binding_checks(r, expected[r['adapter_id']])
        assert all(checks.values()), r['request_id']
        counts.update(k for k,v in checks.items() if v)
        t = r['native_token_timing']
        spans = NativeV1TokenTimeline.service_breakdown(t,
            admitted_at=t['controller_admitted_monotonic_s'],
            completed_at=t['controller_completed_monotonic_s'],clock_id=deployment['clock_id'])
        arrival = r['external_arrival_timing']['planned_arrival_s']
        assert abs(arrival-deployment['arrival_start_s']-r['scheduled_arrival_offset_s']) < 1e-6
        first, last = t['native_first_token_monotonic_s'], t['native_last_token_monotonic_s']
        stage = (t['controller_admitted_monotonic_s']-arrival)*1000 + sum(spans[k] for k in (
            'admission_to_engine_dispatch_ms','engine_entry_to_queue_ms','native_engine_queue_ms','native_prefill_ms'))
        errors = dict(ttft_ms=abs((first-arrival)*1000-r['overall_ttft_ms']),
            stage_ms=abs(stage-r['overall_ttft_ms']))
        if r['completion_tokens'] >= 2:
            errors['tpot_ms'] = abs((last-first)*1000/(r['completion_tokens']-1)-r['tpot_ms'])
        assert all(math.isfinite(v) and v <= 1 for v in errors.values())
        for k,v in errors.items():
            max_errors[k] = max(max_errors[k],v)
    tiers = dict(Counter(r['readiness_tier_before_dispatch'] for r in successful))
    failures = [dict(r) for r in failed]  # Keep the actual failure, including missing tier/timing.
    del rows, successful, failed, data, inputs
    gc.collect()
    physical_path = run/'launch.launch/physical_deployment/physical_resource_summary.json'
    physical = read(physical_path)
    summary_path = run/'launch.launch/physical_deployment/summary.json'
    summary = read(summary_path)
    assert physical['measurement_complete'] and not physical['open_lease_ids']
    assert physical['result_file']['sha256'] == receipt['source_sha256']
    assert physical['summary_sha256'] == sha256(summary_path)
    assert physical['n_plan'] == physical['n_terminal'] == 4000
    assert physical['n_native_contract_complete'] == 3999 and physical['n_correct'] is None
    assert abs(sum(summary['window_gpu_seconds'].values())-physical['physical_gpu_seconds_total']) < 1e-6
    for entry in summary['allocation_journals']:
        assert entry['released'] and sha256(Path(entry['path'])) == entry['sha256']
    assert sha256(Path(summary['request_terminals_path'])) == summary['request_terminals_sha256']
    outcome = run/'launch.launch/physical_deployment/main_outcome.json'
    meta, transfers = project(outcome)
    journal = raw/'remote_transfers.jsonl'
    assert sha256(journal) == receipt['remote_transfers_sha256']
    assert sha256(raw/'remote_monitor.log') == receipt['remote_monitor_sha256']
    remote_rows = [json.loads(line) for line in journal.read_text().splitlines()]
    remote = {r['transfer_id']:r for r in remote_rows}
    assert len(remote) == len(remote_rows) == len(transfers)
    assert set(remote) == {r['http_transfer_id'] for r in transfers}
    clock = read(raw/'remote_health.json')['health']['clock_id']
    for r in transfers:
        s = remote[r['http_transfer_id']]
        assert r['state'] == 'published' and r['content_verified'] and r['published_archive_verified']
        assert r['artifact_id'] == s['artifact_id'] and r['remote_archive_sha256'] == s['archive_sha256']
        assert r['transferred_bytes'] == s['archive_bytes'] == s['bytes_read'] == s['bytes_written']
        assert s['clock_id'] == clock and s['outcome'] == 'sent'
        assert not r['remote_pack_performed'] and not s['pack_performed'] and not s['temporary_created']
    samples, events = [], Counter()
    watchdog = run/'launch.launch/watchdog.jsonl'
    with watchdog.open() as handle:
        for line in handle:
            r = json.loads(line)
            events[r['event']] += 1
            if r['event'] == 'resource_sample':
                assert not r['decision']['abort_reasons']
                samples.append(r)
    assert len(samples) == expected_samples and events['gpu_terminal_census'] == events['service_domain_gone'] == 1
    seal_path = ROOT/'paper_results/ieee_tc/safety/20260925_execution_start_protected.json'
    seal = p.verify_seal(seal_path)
    known = '/home/qhq/serverless_llm_baselines/scripts/replay_openai_trace.py'
    assert seal['changed'] == [known] and seal['entry_count'] == 147
    assert sha256(Path(known)) == '04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22'
    result = dict(kind=f'{campaign}_failure_aware_context_v1',planned=4000,terminal=4000,
        successful_returns=3999,failed=1,n_correct=None,n_good=None,
        all_correct_gate_pass=False,formal_qualification=False,ranking_eligible=False,
        input_identity_equal=True,input_fields_checked=FIELDS[:-1],control_binding_pass_counts=dict(counts),
        successful_dispatch_tier_counts=tiers,max_successful_timing_errors_ms=max_errors,
        physical_gpu_seconds_observed=physical['physical_gpu_seconds_observed'],
        window_gpu_seconds=summary['window_gpu_seconds'],terminal_meta=meta,
        remote=dict(transfers=len(transfers),wire_bytes=sum(r['transferred_bytes'] for r in transfers),
                    pack_performed=False,clock=clock),remote_stop=receipt['remote_stop'],
        resources=dict(samples=len(samples),peak_bytes=max(r['service']['memory.peak'] for r in samples),
            min_host_available_bytes=min(r['host']['available_bytes'] for r in samples),
            max_swap_bytes=max(r['service']['memory.swap.current'] for r in samples),
            memory_events={k:max(r['service']['memory.events'][k] for r in samples)
                           for k in ('high','max','oom','oom_kill')},event_counts=dict(events)),
        protection_raw=seal,no_figures=True,confidence_interval=None,
        caveats=['3999 successful returns are a conditional audit population, not 4000 correct requests.',
                 'The failed offered request is preserved verbatim; its tier/timing are not imputed.',
                 'Control binding and token count do not qualify kernel arithmetic.',
                 'Cleanup success is not workload correctness. No performance ranking or SLO acceptance.'])
    out.mkdir(parents=True,exist_ok=False)
    for name,value in [('context.json',result),('failed_requests.json',failures)]:
        with (out/name).open('x') as handle:
            json.dump(value,handle,indent=2,allow_nan=False)
    paths = [source,reference,outcome,journal,raw/'remote_monitor.log',raw/'remote_health.json',
        raw/'completion_receipt.json',run/'launch.json',watchdog,physical_path,summary_path,
        index_path,protocol,seal_path,Path(__file__),ROOT/'scripts/ieee_tc/plot_d290_diagnostic.py']
    manifest = dict(source_refs=[dict(path=str(p),sha256=sha256(p)) for p in paths],
                    outputs=complete_failure_output_hashes(out))
    with (out/'manifest.json').open('x') as handle:
        json.dump(manifest,handle,indent=2)
    print(json.dumps(result,indent=2),flush=True)


def main(campaign='d299', reference_override=None):
    global RAW, RUN, OUT
    if reference_override is not None and (campaign != 'd315' or reference_override not in ('d303', 'd314')):
        raise ValueError('explicit reference is restricted to the registered D315 contrasts')
    reference = 'd297'
    health_name = 'remote_health2.json'
    change_caveat = 'Capacity, slot, rank and memory package changed together; not single-factor ablation.'
    if campaign == 'd300':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d300_20261007'
        RUN = RAW/'7b_full_w0'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d300_full_w0/context1'
        reference = 'd299'
        health_name = 'remote_health.json'
        change_caveat = 'Only the request-path scale-in latency reference changes; closed-loop trajectories may all change.'
    elif campaign == 'd303':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d303_20261007'
        RUN = RAW/'7b_full_w0'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d303_full_w0/context1'
        reference = 'd300'
        health_name = 'remote_health.json'
        change_caveat = 'Same controls; only D301 confirmed-scope publication fix changes. Closed-loop trajectories may all change.'
    elif campaign == 'd314':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d314_20261007'
        RUN = RAW/'7b_full_w0'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d314_full_w0/context1'
        reference = 'd303'
        health_name = 'remote_health.json'
        change_caveat = ('D307 token budget/profile plus subsequent ownership/copy-attribution fixes; '
                         'not a pure token-budget ablation or an independent held-out comparison.')
    elif campaign == 'd315':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d315_20261007'
        RUN = RAW/'7b_full_w0'
        reference = reference_override or 'd303'
        suffix = 'context1' if reference == 'd303' else 'context_d314'
        OUT = ROOT/f'paper_results/ieee_tc/p2_backend/d315_full_w0/{suffix}'
        health_name = 'remote_health.json'
        change_caveat = (
            'D303 measured 1024-token settings restored with later ownership/copy-attribution fixes; '
            'closed-loop trajectories and run variation remain.' if reference == 'd303' else
            'Same serving implementation; token budget and its matching measured initialization both change; '
            'not a pure budget-only ablation or an independent held-out comparison.')
    elif campaign == 'd318':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d318_20261007'
        RUN = RAW/'7b_full_w0'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d318_full_w0/context1'
        reference = 'd315'
        health_name = 'remote_health.json'
        change_caveat = (
            'Same configuration and measured initialization; D316 event-confirmed GPU-copy reuse changes. '
            'Closed-loop trajectories and single-run variation remain; no statistical superiority claim.')
    elif campaign == 'd323':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d323_20261007'
        RUN = RAW/'7b_full_w0'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d323_full_w0/context1'
        reference = 'd318'
        health_name = 'remote_health.json'
        change_caveat = (
            'D318 model/control/preparation settings retained; D321 measured declared-output '
            'service classes replace the previous initialization. Independent calibration input '
            'provenance is verified by the corrected joint loader; shared runtime/resource/model '
            'binding remains strict. Single complete development replay, not statistical superiority.')
    elif campaign == 'd326':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d326_20261008'
        RUN = RAW/'7b_full_w0'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d326_full_w0/context1'
        reference = 'd323'
        health_name = 'remote_health.json'
        change_caveat = (
            'D323 model, controls and profiles retained; only D325 owned scale-in '
            'withdrawal, drain and physical release changes. Closed-loop trajectories '
            'and single-run variation remain; no statistical superiority claim.')
    elif campaign == 'd332':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d332_20261008'
        RUN = RAW/'7b_full_w0'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d332_full_w0/context1'
        reference = 'd326'
        health_name = 'remote_health.json'
        change_caveat = (
            'D326 model, controls, measured profiles and ownership lifecycle retained; '
            'D330 publishes already measured planner snapshots into confirmed route scopes. '
            'Counters prove trigger, not avoided RPC count or causal latency savings. '
            'Single development run per configuration; closed-loop variation remains.')
    elif campaign == 'd335':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d335_20261008'
        RUN = RAW/'7b_full_w0'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d335_full_w0/context1'
        reference = 'd332'
        health_name = 'remote_health.json'
        change_caveat = (
            'D332 model, controls, measured profiles and ownership lifecycle retained; '
            'D333 combines selected-GPU pending registration and protection in one native '
            'core turn. HOST ordering is unchanged. Trigger is not causal latency saving. '
            'Single development run per configuration; closed-loop variation remains.')
    elif campaign == 'd336':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d336_20261008'
        RUN = RAW/'7b_full_w0'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d336_full_w0/context1'
        reference = 'd335'
        health_name = 'remote_health.json'
        change_caveat = (
            'D335 serving code, model, measured profiles, hard capacity and ownership '
            'lifecycle retained. Only the D88 capacity-derived queue/active soft targets '
            'are applied to qualified C8; delay guards and min1/max4 remain unchanged. '
            'Single development run per configuration; closed-loop variation remains.')
    elif campaign == 'd341':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d341_20261008'
        RUN = RAW/'7b_full_w0'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d341_full_w0/context1'
        reference = 'd336'
        health_name = 'remote_health.json'
        change_caveat = (
            'D336 model, controls, measured ready-state profiles and lifecycle retained. '
            'Only optional unquantized Humming preparation becomes on-demand; native spawn '
            'is retained. Full development replay, not a held-out block or statistical '
            'superiority claim; closed-loop trajectories can change.')
    elif campaign != 'd299':
        raise ValueError('unknown frozen full-run identity')
    assert not OUT.exists()
    assert p.check_plan()=='0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    assert sha256(ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md')=='5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    audits, sources = {}, {}
    for name in (reference,campaign):
        directory=CURATED/f'{name}_full_w0/validated1'
        manifest=read(directory/'manifest.json')
        for filename,digest in manifest['outputs'].items():
            assert sha256(directory/filename)==digest
        audits[name]=read(directory/'audit.json')
        source=Path(audits[name]['evidence']['source'])
        assert sha256(source)==audits[name]['evidence']['source_sha256']
        sources[name]=source
        assert all(n==4000 for n in audits[name]['control_binding_pass_counts'].values())
    left, le, lr = audit_run(sources[reference])
    right, re, rr = audit_run(sources[campaign])
    assert len(lr)==len(rr)==4000
    assert all(all(a[k]==b[k] for k in FIELDS) for a,b in zip(lr,rr))
    assert all(le[k]==re[k] for k in ('shared_trace_sha256','shared_adapter_subset_sha256'))
    differences=[dict(request_id=a['request_id'],adapter_id=a['adapter_id'],
        reference_sha256=a['completion_token_ids_sha256'],candidate_sha256=b['completion_token_ids_sha256'])
        for a,b in zip(lr,rr) if a['completion_token_ids_sha256']!=b['completion_token_ids_sha256']]
    tiers=dict(Counter(r['readiness_tier_before_dispatch'] for r in rr))
    del lr,rr
    gc.collect()
    outcome=RUN/'launch.launch/physical_deployment/main_outcome.json'
    meta,transfers=project(outcome)
    journal=RAW/'remote_transfers.jsonl'
    remote_rows=[json.loads(line) for line in journal.read_text().splitlines()]
    remote={r['transfer_id']:r for r in remote_rows}
    assert len(remote)==len(remote_rows)==len(transfers)
    assert set(remote)=={r['http_transfer_id'] for r in transfers}
    health=read(RAW/health_name)
    clock=(health['7b']['clock_id'] if campaign in ('d332', 'd335', 'd336', 'd341') else
           health['clock_id'] if campaign in ('d323', 'd326') else health['health']['clock_id'])
    for r in transfers:
        s=remote[r['http_transfer_id']]
        assert r['state']=='published' and r['content_verified'] and r['published_archive_verified']
        assert r['artifact_id']==s['artifact_id'] and r['remote_archive_sha256']==s['archive_sha256']
        assert r['transferred_bytes']==s['archive_bytes']==s['bytes_read']==s['bytes_written']
        assert s['clock_id']==clock and s['outcome']=='sent'
        assert not r['remote_pack_performed'] and not s['pack_performed'] and not s['temporary_created']
    samples=[]
    with (RUN/'launch.launch/watchdog.jsonl').open() as handle:
        for line in handle:
            r=json.loads(line)
            if r['event']=='resource_sample':
                assert not r['decision']['abort_reasons']
                samples.append(r)
    assert samples
    seal=p.verify_seal(ROOT/'paper_results/ieee_tc/safety/20260925_execution_start_protected.json')
    known='/home/qhq/serverless_llm_baselines/scripts/replay_openai_trace.py'
    assert seal['changed']==[known] and seal['entry_count']==147
    assert sha256(Path(known))=='04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22'
    result=dict(kind=f'{campaign}_full_context_audit_v1',reference=le,candidate=re,
        input_identity_equal=True,input_fields_checked=FIELDS,output_hash_changes=differences,
        tier_counts=tiers,terminal_meta=meta,
        remote=dict(transfers=len(transfers),wire_bytes=sum(r['transferred_bytes'] for r in transfers),
                    pack_performed=False,clock=clock),
        resources=dict(samples=len(samples),peak_bytes=max(r['service']['memory.peak'] for r in samples),
            min_host_available_bytes=min(r['host']['available_bytes'] for r in samples),
            max_swap_bytes=max(r['service']['memory.swap.current'] for r in samples),
            memory_events={k:max(r['service']['memory.events'][k] for r in samples)
                           for k in ('high','max','oom','oom_kill')}),
        protection_raw=seal,n_correct=None,formal_qualification=False,no_figures=True,
        replicates_per_configuration=1,confidence_interval=None,
        caveats=['Full development W0, not held-out; no statistical superiority claim.',
                 change_caveat,
                 'Binding and native counts do not by themselves qualify kernel arithmetic.',
                 'Warm reference remains provisional; shared final references remain open.'])
    if campaign == 'd303':
        receipt = read(CURATED/'d303_full_w0/completion_receipt.json')
        assert receipt['source_result_sha256'] == re['source_sha256']
        assert receipt['remote_transfers_sha256'] == sha256(journal)
        assert receipt['remote_monitor_sha256'] == sha256(RAW/'remote_monitor.log')
        result['remote_shutdown'] = receipt['remote_shutdown']
        result['caveats'].append('Remote monitor exit73 is retained as a post-measurement shutdown-order guard; not all remote return codes are zero.')
    if campaign == 'd314':
        receipt = read(RAW/'completion_receipt.json')
        assert receipt['source_sha256'] == re['source_sha256']
        assert receipt['launch_sha256'] == sha256(RUN/'launch.json')
        assert receipt['remote_transfers_sha256'] == sha256(journal)
        assert receipt['remote_monitor_sha256'] == sha256(RAW/'remote_monitor.log')
        result['remote_shutdown'] = receipt['remote_stop']
    if campaign in ('d315', 'd318', 'd323', 'd326', 'd332', 'd335', 'd336', 'd341'):
        receipt = read(RAW/'completion_receipt.json')
        entries = receipt['evidence']
        assert entries[sources[campaign].name]['sha256'] == re['source_sha256']
        assert entries['launch.json']['sha256'] == sha256(RUN/'launch.json')
        assert entries['remote_transfers.jsonl']['sha256'] == sha256(journal)
        assert entries['remote_monitor.log']['sha256'] == sha256(RAW/'remote_monitor.log')
        assert receipt['physical_gpu_seconds_total'] == right['physical_gpu_seconds']
        assert receipt['n_plan'] == receipt['n_terminal'] == receipt['n_native_contract_complete'] == 4000
        assert receipt['n_failed_from_terminal_journal'] == 0 and receipt['n_correct'] is None
        assert receipt['formal_qualification'] is False
        result['remote_shutdown'] = receipt['remote_cleanup']
    OUT.mkdir(parents=True,exist_ok=False)
    with (OUT/'context.json').open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    with (OUT/'comparison.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['metric','reference','candidate','lower_better_change_percent'],
                              lineterminator='\n')
        writer.writeheader()
        for key,a in audits[reference]['stats'].items():
            b=audits[campaign]['stats'][key]
            change=(a-b)/a*100 if a not in (0,None) and b is not None and key not in ('completed','n_correct') else None
            writer.writerow(dict(metric=key,reference=a,candidate=b,lower_better_change_percent=change))
    paths=[*sources.values(),outcome,journal,RAW/'remote_monitor.log',RAW/health_name,
           RUN/'launch.json',RUN/'launch.launch/watchdog.jsonl',Path(__file__),
           ROOT/'scripts/ieee_tc/plot_d290_diagnostic.py',ROOT/'scripts/ieee_tc_preflight.py']
    if campaign == 'd303':
        paths.append(CURATED/'d303_full_w0/completion_receipt.json')
    if campaign in ('d314', 'd315', 'd318', 'd323', 'd326', 'd332', 'd335', 'd336', 'd341'):
        paths.append(RAW/'completion_receipt.json')
    with (OUT/'manifest.json').open('x') as handle:
        json.dump(dict(source_refs=[dict(path=str(p),sha256=sha256(p)) for p in paths],
            outputs={p.name:sha256(p) for p in sorted(OUT.iterdir()) if p.name!='manifest.json'}),handle,indent=2)
    print(json.dumps({k:result[k] for k in ('remote','resources','tier_counts')},indent=2),flush=True)
    print(json.dumps(dict(input_identity_equal=True,output_hash_change_count=len(differences))),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', choices=('d299', 'd300', 'd303', 'd314', 'd315', 'd318', 'd323', 'd326', 'd332', 'd335', 'd336', 'd341', 'd309-failure', 'd311-incomplete', 'd313-failure'), default='d299')
    parser.add_argument('--reference', choices=('d303', 'd314'),
        help='D315 only: registered complete-run reference, with separate non-overwriting output')
    args = parser.parse_args()
    campaign = args.campaign
    if args.reference is not None and campaign != 'd315':
        parser.error('--reference is only valid for D315')
    if campaign == 'd311-incomplete':
        audit_d311_incomplete()
    elif campaign == 'd309-failure':
        audit_d309_failure()
    elif campaign == 'd313-failure':
        audit_d309_failure('d313')
    else:
        main(campaign, args.reference)
