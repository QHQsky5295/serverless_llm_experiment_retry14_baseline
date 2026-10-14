"""Thin post-release full-run check reusing native audits and seal helpers.

No inference, figures, new metric definitions, or overwriting existing results.
The large outcome is projected through jq streaming, never loaded as a tree.
"""
import argparse
import copy
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
        'd364': ('b90a4f6874d0f5f8a9c0a1ad8ddf749430e11c46b32c35c9eaa71c3edb6762aa', 4029),
    }
    expected_source_sha, expected_samples = cases[campaign]
    date = '20261009' if campaign == 'd364' else '20261007'
    raw = ROOT/f'results/ieee_tc/p2_backend_qualification/{campaign}_{date}'
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
    health = read(raw/'remote_health.json')
    clock = health['7b']['clock_id'] if campaign == 'd364' else health['health']['clock_id']
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


REGISTERED_REFERENCES = {'d315': ('d303', 'd314'), 'd353': ('d349', 'd336'),
                         'd358': ('d353', 'd336', 'd349'),
                         'd364': ('d358', 'd353', 'd336'),
                         'd367': ('d358', 'd353', 'd336')}


def main(campaign='d299', reference_override=None):
    global RAW, RUN, OUT
    if reference_override is not None and reference_override not in REGISTERED_REFERENCES.get(campaign, ()):
        raise ValueError('explicit reference is restricted to the registered campaign contrasts')
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
    elif campaign == 'd346':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d346_20261008'
        RUN = RAW/'7b_full_w0'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d346_full_w0/context1'
        reference = 'd336'
        health_name = 'remote_health.json'
        change_caveat = (
            'Qualified native capacity changes from C8 to C14, with matching D344 '
            'measured service/preparation profiles and capacity-derived soft targets. '
            'D336 control rule, model/precision, lifecycle and common evaluation ruler '
            'are retained; this is not an isolated single-gate ablation. '
            'One complete development replay per configuration, not held-out or '
            'statistical superiority evidence. D341 prebuild change is not applied.')
    elif campaign == 'd349':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d349_20261008'
        RUN = RAW/'7b_full_w0'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d349_full_w0/context1'
        reference = 'd346'
        health_name = 'remote_health.json'
        change_caveat = (
            'D346 native C14 capacity, measured D344 profiles and IEEE control law '
            'are unchanged. Only queue/active soft targets restore the D336 '
            'absolute working point S8, without adding a hard admission cap8. '
            'This is one complete development replay per configuration, not '
            'held-out or statistical superiority evidence. D341 is not applied.')
    elif campaign == 'd353':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d353_20261008'
        RUN = RAW/'7b_full_w0'
        reference = reference_override or 'd349'
        if reference not in ('d349', 'd336'):
            raise ValueError('D353 compares only its registered D349/D336 full references')
        suffix = 'context1' if reference == 'd349' else 'context_d336'
        OUT = ROOT/f'paper_results/ieee_tc/p2_backend/d353_full_w0/{suffix}'
        health_name = 'remote_health.json'
        change_caveat = (
            'D349 configuration/profiles are unchanged; only D352 transaction-local '
            'HOST postcheck evidence on confirmed GPU acquisition changes. '
            'Fresh allocator occupancy and all load-path checks remain. ' if reference == 'd349' else
            'D336 supplies earlier service-quality context, not a single-factor '
            'contrast: C14 capacity, D344 profiles, D349 soft targets and the '
            'D352 HOST postcheck candidate differ. ')
        change_caveat += (
            'Ordinary Full4000 with no D351 detailed observer. One development '
            'replay per configuration; closed-loop variation remains. '
            'No statistical superiority or formal G1/G2 qualification.')
    elif campaign == 'd358':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d358_20261008'
        RUN = RAW/'7b_full_w0'
        reference = reference_override or 'd353'
        suffix = 'context1' if reference == 'd353' else f'context_{reference}'
        OUT = ROOT/f'paper_results/ieee_tc/p2_backend/d358_full_w0/{suffix}'
        health_name = 'remote_health.json'
        change_caveat = (
            'Relative to D353: atomic native control interleaving, corresponding '
            'D356 measured initialization, and copied registration counter metadata; '
            'controller, native capacity and workload are unchanged. '
            'D336 and D349 are historical context with additional configuration '
            'or implementation differences, not single-factor contrasts. '
            'Ordinary Full4000 with no intrusive timeline observer. One development '
            'replay per configuration; no statistical superiority or formal G1/G2 qualification.')
    elif campaign == 'd364':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d364_20261009'
        RUN = RAW/'7b_full_w0'
        reference = reference_override or 'd358'
        suffix = 'context1' if reference == 'd358' else f'context_{reference}'
        OUT = ROOT/f'paper_results/ieee_tc/p2_backend/d364_full_w0/{suffix}'
        health_name = 'remote_health.json'
        change_caveat = (
            'Relative to D358: protected held-file owner-turn validation and its '
            'corresponding D362 measured initialization; controller, native capacity, '
            'generation contract and workload are unchanged. D353 and D336 are '
            'historical context with additional implementation/configuration differences, '
            'not single-factor contrasts. Candidate exposure does not establish avoided '
            'RPC count or causal speedup. Ordinary Full4000 with no intrusive observer. '
            'One development replay per configuration; no statistical superiority or '
            'formal G1/G2 qualification. D363 prefix regressions remain evidence.')
    elif campaign == 'd367':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d367_20261009'
        RUN = RAW/'7b_full_w0'
        reference = reference_override or 'd358'
        suffix = 'context1' if reference == 'd358' else f'context_{reference}'
        OUT = ROOT/f'paper_results/ieee_tc/p2_backend/d367_full_w0/{suffix}'
        health_name = 'remote_health.json'
        change_caveat = (
            'Relative to D358: the D365 immutable file-demand identity repair and '
            'the unchanged D362 measured initialization are evaluated on the complete '
            'W0 replay. D364 remains a failed offered population; D366 is a prefix '
            'diagnostic. This is one development replay, not a held-out block or '
            'formal G1/G2 qualification.')
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
    clock=(health['7b']['clock_id'] if campaign in ('d332', 'd335', 'd336', 'd341', 'd346', 'd349', 'd353', 'd358', 'd364', 'd367') else
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
    if campaign in ('d315', 'd318', 'd323', 'd326', 'd332', 'd335', 'd336', 'd341', 'd346', 'd349', 'd353', 'd358', 'd364', 'd367'):
        receipt = read(RAW/'completion_receipt.json')
        # D367 uses the compact closure receipt emitted by its resumable
        # collector; older campaigns retain the per-file ``evidence`` map.
        entries = receipt.get('evidence')
        if entries is not None:
            assert entries[sources[campaign].name]['sha256'] == re['source_sha256']
            assert entries['launch.json']['sha256'] == sha256(RUN/'launch.json')
            assert entries['remote_transfers.jsonl']['sha256'] == sha256(journal)
            assert entries['remote_monitor.log']['sha256'] == sha256(RAW/'remote_monitor.log')
        else:
            assert campaign == 'd367'
            assert receipt['source_sha256'] == re['source_sha256']
            assert receipt['launch_sha256'] == sha256(RUN/'launch.json')
            assert receipt['remote_transfers_sha256'] == sha256(journal)
            assert receipt['remote_monitor_sha256'] == sha256(RAW/'remote_monitor.log')
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
    if campaign in ('d314', 'd315', 'd318', 'd323', 'd326', 'd332', 'd335', 'd336', 'd341', 'd346', 'd349', 'd353', 'd358', 'd364', 'd367'):
        paths.append(RAW/'completion_receipt.json')
    with (OUT/'manifest.json').open('x') as handle:
        json.dump(dict(source_refs=[dict(path=str(p),sha256=sha256(p)) for p in paths],
            outputs={p.name:sha256(p) for p in sorted(OUT.iterdir()) if p.name!='manifest.json'}),handle,indent=2)
    print(json.dumps({k:result[k] for k in ('remote','resources','tier_counts')},indent=2),flush=True)
    print(json.dumps(dict(input_identity_equal=True,output_hash_change_count=len(differences))),flush=True)


def review_single_run(source, warm_path, output):
    """Stream a closed historical Full run under the unchanged V1 metric ruler.

    Reuse existing binding/timeline checks. Never relabel a development run as
    an unseen formal run, infer arithmetic correctness, or overwrite evidence.
    The output is an evidence review; its eligibility must be interpreted with
    the single-run V2 amendment and the independently frozen warm manifest.
    """
    import ijson
    from statistics import fmean
    from scripts.ieee_tc.plot_d290_diagnostic import percentile

    if output.exists():
        raise ValueError('refusing to overwrite a single-run review')
    run = source.parent
    launch_path = run/'launch.json'
    launch = read(launch_path)
    if not all(launch.get(k) is True for k in (
            'pass', 'service_path_removed', 'native_gpu_context_release_confirmed')):
        raise ValueError('closed and released successful launch required')
    physical_path = run/'launch.launch/physical_deployment/physical_resource_summary.json'
    physical, summary = read(physical_path), read(run/'launch.launch/physical_deployment/summary.json')
    source_sha = sha256(source)
    if (physical['result_file']['sha256'] != source_sha or not summary['measurement_complete']
            or summary['open_lease_ids'] or summary['n_plan'] != 4000):
        raise ValueError('complete physical ledger bound to this result required')
    paths = [source, launch_path, physical_path, run/'launch.launch/physical_deployment/summary.json',
             warm_path, Path(__file__), ROOT/'scripts/ieee_tc/plot_d290_diagnostic.py']
    for entry in summary['allocation_journals']:
        path = Path(entry['path'])
        if not entry['released'] or sha256(path) != entry['sha256']:
            raise ValueError('allocation journal hash or release mismatch')
        paths.append(path)
    terminal_path = Path(summary['request_terminals_path'])
    if sha256(terminal_path) != summary['request_terminals_sha256']:
        raise ValueError('terminal ledger hash mismatch')
    paths.append(terminal_path)
    gpu_s = 0.0
    for intervals in summary['device_intervals'].values():
        end = -math.inf
        for a, b in sorted(intervals):
            if not all(math.isfinite(v) for v in (a, b)) or b < a or a < end:
                raise ValueError('ordered disjoint physical ownership intervals required')
            gpu_s += b-a
            end = b
    if (abs(gpu_s-summary['gpu_seconds_observed']) > 1e-6
            or abs(gpu_s-sum(summary['window_gpu_seconds'].values())) > 1e-6):
        raise ValueError('physical integral/window partition mismatch')
    with source.open('rb') as stream:
        metadata = next(ijson.items(stream, 'metadata', use_float=True))
    external = launch['external_replay']
    trace_path = Path(external['plan']['source_path'])
    if sha256(trace_path) != external['plan']['source_sha256']:
        raise ValueError('source trace hash mismatch')
    if abs(external['replay_t0_s']-external['deployment_notice_s']-60) > 1e-6:
        raise ValueError('common deployment notice mismatch')
    from faaslora.datasets.workload_generator import FrozenReplayPlan
    replay_plan = FrozenReplayPlan.load(trace_path, profile=external['plan']['profile'],
                                       rate_scale=external['plan']['rate_scale'])
    if replay_plan.identity() != external['plan']:
        raise ValueError('frozen replay view differs from the launch receipt')
    source_entries = {entry.request_id: entry for entry in replay_plan.entries}
    paths.append(trace_path)
    warm = read(warm_path)
    if (warm['source_trace_sha256'] != external['plan']['source_sha256']
            or warm['contract']['native_pass'] != warm['measurement_requests']):
        raise ValueError('warm reference source/native contract mismatch')
    if (warm['model'] != 'llama2_7b_main_v2_publicmix'
            or [g['native_input_boundary'] for g in warm['groups']] != ['<=616', '>616']):
        raise ValueError('this registered review supports only the two D394 7B strata')
    index_path = ROOT/'paper_results/ieee_tc/inputs/20260927_7b_materialized_content_index.json'
    index = read(index_path)
    content = {a['id']: hashlib.sha256(json.dumps(sorted(a['files'], key=lambda f: f['path']),
               sort_keys=True, separators=(',', ':')).encode()).hexdigest() for a in index['artifacts']}
    paths.append(index_path)
    remote_path = run.parent/'remote_transfers.jsonl'
    with remote_path.open() as stream:
        transfers = [json.loads(line) for line in stream]
    server = {x['transfer_id']: x for x in transfers}
    if len(server) != len(transfers) or any(
            x['pack_performed'] or x['temporary_created'] or x['outcome'] != 'sent' for x in transfers):
        raise ValueError('published remote-transfer journal contract failed')
    paths.append(remote_path)
    seen, counts, binding, tiers, matched = set(), Counter(), Counter(), Counter(), set()
    values = {k: [] for k in ('ttft_ms', 'tpot_ms', 'e2e_ms')}
    max_error = {k: 0.0 for k in ('ttft_ms', 'tpot_ms', 'e2e_ms')}
    vectors = []
    with source.open('rb') as stream:
        for r in ijson.items(stream, 'detailed_results.tc_ieee_full.requests.item', use_float=True):
            rid = r['request_id']
            if rid in seen or not rid or r['success'] is not True:
                raise ValueError('unique successful complete request population required')
            seen.add(rid)
            entry = source_entries[rid]
            original = json.loads(entry.source_json)
            if (r['external_arrival_timing']['source_item_sha256'] != entry.source_sha256
                    or r['adapter_id'] != original['adapter_id']
                    or r['source_expected_output_tokens'] != original['expected_output_tokens']
                    or abs(r['scheduled_arrival_offset_s']-entry.offset_s) > 1e-6
                    or abs(r['external_arrival_timing']['planned_arrival_s']
                           - external['replay_t0_s']-entry.offset_s) > 1e-6):
                raise ValueError('request identity/arrival/adapter/target differs from frozen source')
            request_vectors([r])
            t, e = r['native_token_timing'], r['gpu_reference_evidence']
            tests = binding_checks(r, content[r['adapter_id']])
            for name, passed in tests.items():
                binding[name] += passed
            if not all(tests.values()):
                raise ValueError(f'native binding evidence failed: {rid}: {tests}')
            if (t['native_clock_id'] != summary['clock_id'] or not t['native_terminal_observed']
                    or r['completion_tokens'] != min(r['source_expected_output_tokens'], 256)
                    or t['native_output_tokens'] != r['completion_tokens']
                    or e['confirmed_dispatch_snapshot'] is not True):
                raise ValueError('native output/clock/dispatch contract mismatch')
            arrival = r['external_arrival_timing']['planned_arrival_s']
            first, last, completed = (t['native_first_token_monotonic_s'],
                                     t['native_last_token_monotonic_s'], t['controller_completed_monotonic_s'])
            if not arrival <= first <= last <= completed:
                raise ValueError('unordered user/native timing')
            n = r['completion_tokens']
            metrics = dict(ttft_ms=(first-arrival)*1000, e2e_ms=(completed-arrival)*1000)
            if n >= 2:
                metrics['tpot_ms'] = (last-first)*1000/(n-1)
            stored = dict(ttft_ms=r['overall_ttft_ms'], e2e_ms=r['overall_e2e_ms'], tpot_ms=r['tpot_ms'])
            for name, v in metrics.items():
                if not math.isfinite(v) or v < 0 or abs(v-stored[name]) > 1:
                    raise ValueError('nonfinite/negative timing or reconstruction error over 1 ms')
                max_error[name] = max(max_error[name], abs(v-stored[name]))
                values[name].append(v)
            group = warm['groups'][0 if t['actual_prompt_tokens'] <= 616 else 1]
            tp = n == 1 or metrics['tpot_ms'] <= group['candidate_threshold_ms']['tpot']
            tt = metrics['ttft_ms'] <= group['candidate_threshold_ms']['ttft']
            counts.update(N=1, native_contract=1, ttft_pass=int(tt), tpot_pass=int(tp),
                          joint_pass=int(tt and tp), only_ttft_fail=int(not tt and tp),
                          only_tpot_fail=int(tt and not tp), both_fail=int(not tt and not tp))
            tiers[r['readiness_tier_before_dispatch']] += 1
            remote = e.get('remote_preparation')
            if remote is not None and remote.get('http_transfer_id'):
                s = server[remote['http_transfer_id']]
                if (s['artifact_id'] != r['adapter_id'] or remote['content_verified'] is not True
                        or remote['remote_pack_performed'] is not False
                        or remote['remote_archive_sha256'] != s['archive_sha256']
                        or remote['transferred_bytes'] != s['archive_bytes']):
                    raise ValueError('client/server delivery identity mismatch')
                matched.add(s['transfer_id'])
            vectors.append({k:r[k] for k in FIELDS})
    if len(seen) != 4000 or seen != set(source_entries) or counts['N'] != metadata['total_requests']:
        raise ValueError('full 4000-request population required')
    samples, events = [], Counter()
    watchdog = run/'launch.launch/watchdog.jsonl'
    with watchdog.open() as stream:
        for line in stream:
            row = json.loads(line)
            events[row['event']] += 1
            if row['event'] == 'resource_sample':
                samples.append(row)
    paths.append(watchdog)
    if not samples or any(x['decision']['abort_reasons'] for x in samples):
        raise ValueError('missing or aborted independent resource monitoring')
    report = dict(kind='single_run_evidence_review_v2', source=str(source), source_sha256=source_sha,
        original_formal_run=metadata['formal_run'], original_trace_role=metadata['trace_role'],
        reuse_identity='reused_development_observation', independent_holdout=False,
        metric_protocol_sha256=sha256(ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md'),
        single_run_amendment_sha256=sha256(ROOT/'docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md'),
        source_plan_sha256=launch['plan_sha256'], current_plan_sha256=p.check_plan(),
        counts=dict(counts), native_binding_checks=dict(binding), tier_counts=dict(tiers),
        timing_joint_rate=counts['joint_pass']/counts['N'], timing_feasible=counts['joint_pass']>=3800,
        metrics={k:dict(mean=fmean(v), p95_type1=percentile(v,.95), n=len(v)) for k,v in values.items()},
        native_reconstruction_max_error_ms=max_error, physical_gpu_seconds=gpu_s,
        gpu_seconds_per_offered=gpu_s/4000, measurement_complete=True, open_lease_ids=[],
        source_trace_sha256=external['plan']['source_sha256'], rate_scale=external['plan']['rate_scale'],
        workload_profile=external['plan']['profile'], preparation_lead_s=60,
        request_vectors_sha256=hashlib.sha256(json.dumps(vectors, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        remote=dict(server_transfers=len(server), matched_request_transfers=len(matched),
                    unmatched_server_transfers=len(set(server)-matched), wire_bytes=sum(x['bytes_written'] for x in transfers),
                    delivery='prepublished_gzip_v1', request_time_packing=False),
        resource=dict(samples=len(samples), events=dict(events),
                      host_min_available_bytes=min(x['host']['available_bytes'] for x in samples),
                      service_peak_bytes=max(x['service']['memory.peak'] for x in samples),
                      max_swap_bytes=max(x['service']['memory.swap.current'] for x in samples),
                      memory_events={k:max(x['service']['memory.events'][k] for x in samples)
                                     for k in ('high','max','oom','oom_kill')}),
        caveats=['Control-plane/source binding is not independent kernel arithmetic proof.',
                 'Legacy local_sim_no_delay label describes inactive limiter, not the HTTP transfer journal.',
                 'Legacy time_scale=8 belongs to the retained source trace; replay rate_scale is recorded separately.',
                 'Single W0 observation cannot establish W1/W2 feasibility, G2 or a baseline ranking.'],
        confidence_interval=None, formal_g1_g2_qualified=False,
        source_refs=[dict(path=str(path), sha256=sha256(path)) for path in dict.fromkeys(paths)])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('source','counts','timing_joint_rate','physical_gpu_seconds','remote')}))


def matched_request_contract(left, right):
    """Exact work equality, with only a 1-us clock-subtraction allowance.

    Output digests are deliberately excluded: the protocol requires an audit of
    changed outputs, not cross-backend bitwise equality or winner selection.
    """
    fields = ('request_id', 'source_item_sha256', 'adapter_id', 'target_tokens',
              'canonical_prompt_sha256', 'native_prompt_token_ids_sha256',
              'native_prompt_tokens', 'artifact_content_sha256', 'generation_contract')
    for name in fields:
        if name not in left or name not in right or left[name] != right[name]:
            raise ValueError(f'matched request contract differs: {name}')
    for row in (left, right):
        if (type(row.get('arrival_offset_s')) not in (int, float)
                or not math.isfinite(row['arrival_offset_s'])):
            raise ValueError('finite arrival offset required')
    if abs(left['arrival_offset_s']-right['arrival_offset_s']) > 1e-6:
        raise ValueError('matched request contract differs: arrival_offset_s')


def closed_physical_integral(summary):
    """Check the retained physical union, not a logical replica-time estimate."""
    if (summary['measurement_complete'] is not True or summary['open_lease_ids']
            or summary['n_plan'] != 4000 or summary['n_terminal'] != 4000
            or summary['n_native_contract_complete'] != 4000):
        raise ValueError('complete closed 4000-request physical ledger required')
    total = 0.
    if not summary['device_intervals']:
        raise ValueError('physical device intervals required')
    for intervals in summary['device_intervals'].values():
        end = -math.inf
        for a, b in sorted(intervals):
            if not all(math.isfinite(v) for v in (a, b)) or b < a or a < end:
                raise ValueError('ordered disjoint physical intervals required')
            total += b-a
            end = b
    if (total <= 0 or abs(total-summary['gpu_seconds_observed']) > 1e-6
            or abs(total-sum(summary['window_gpu_seconds'].values())) > 1e-6):
        raise ValueError('physical integral/window partition mismatch')
    return total


def matched_native_entry(name, contract, native):
    """Read each backend's actual boundary without fabricating vLLM fields.

    This is a join check, not a replacement for the SHA-bound full registry,
    process and content audit. Cross-backend phase boundaries remain distinct.
    """
    if name == 'S-LoRA':
        path = native.get('native_adapter_path')
        if (contract.get('native_backend') != 'slora'
                or native.get('timing_contract') != 'ieee_tc_slora_native_v1'
                or not isinstance(path, str) or not Path(path).is_absolute()
                or Path(path).name != contract['adapter_id']
                or type(native.get('native_adapter_index')) is not int
                or native['native_adapter_index'] < 0
                or native.get('native_adapter_binding_source') != 'slora_model_rpc_actual_batch_adapter'):
            raise ValueError('S-LoRA native path/index binding required')
        value = native.get('native_rpc_registered_monotonic_s')
    elif name == 'Loquetier':
        path = native.get('native_adapter_path')
        count = native.get('native_output_tokens')
        if (contract.get('native_backend') != 'loquetier'
                or native.get('timing_contract') != 'ieee_tc_loquetier_native_v1'
                or native.get('native_adapter_id') != contract['adapter_id']
                or native.get('native_request_id') != contract['request_id']
                or not isinstance(path, str) or not Path(path).is_absolute()
                or Path(path).name != contract['adapter_id']
                or type(native.get('native_adapter_index')) is not int
                or native['native_adapter_index'] < 0
                or native.get('native_adapter_binding_source') != 'loquetier_actual_forward_segments_and_startup_pointer_tables'
                or type(count) is not int or count <= 0 or count != contract['target_tokens']
                or type(native.get('native_forward_count')) is not int
                or native['native_forward_count'] != count
                or native.get('native_terminal_observed') is not True):
            raise ValueError('Loquetier native path/index/forward binding required')
        value = native.get('native_registered_monotonic_s')
        scheduled = native.get('native_scheduled_monotonic_s')
        first = native.get('native_first_token_monotonic_s')
        if (any(type(t) not in (int, float) or not math.isfinite(t)
                for t in (value, scheduled, first)) or not value <= scheduled <= first):
            raise ValueError('ordered Loquetier registration/schedule/token boundaries required')
    else:
        if native.get('native_lora_name') != contract['adapter_id']:
            raise ValueError('vLLM native adapter binding required')
        value = native.get('native_dispatch_monotonic_s')
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('finite backend-specific native entry required')
    return value


def matched_relative_observation(prime, baseline):
    """Keep raw differences even if the baseline is outside the G1 feasible set."""
    return dict(
        gpu_seconds_relative_reduction=(baseline['physical_gpu_seconds']-prime['physical_gpu_seconds'])/baseline['physical_gpu_seconds'],
        joint_slo_difference_percentage_points=100*(prime['joint_slo']-baseline['joint_slo']),
        p95_ttft_relative_reduction=(baseline['metrics']['ttft_ms']['p95_type1']-prime['metrics']['ttft_ms']['p95_type1'])/baseline['metrics']['ttft_ms']['p95_type1'],
        both_observed_protocol_feasible=(prime['observed_protocol_feasible'] and baseline['observed_protocol_feasible']),
        scope='Raw single-run differences; constrained G1 comparison requires both rows feasible.')


def require_unchanged_reference(recomputed, frozen):
    if recomputed != frozen:
        raise ValueError('existing frozen Resident reference differs; do not refreeze')
    return frozen


def matched_controller_rows(source, scenario, content):
    """Stream native controller requests with the same checks for Full and controls."""
    import ijson
    result = {}
    with source.open('rb') as handle:
        for r in ijson.items(handle, f'detailed_results.{scenario}.requests.item', use_float=True):
            rid, n, ext = r['request_id'], r['native_token_timing'], r['external_arrival_timing']
            if rid in result or not r['success'] or not all(binding_checks(r, content[r['adapter_id']]).values()):
                raise ValueError('controller source/native binding failed')
            request_vectors([r])
            result[rid] = dict(request_id=rid,
                source_item_sha256=ext['source_item_sha256'], adapter_id=r['adapter_id'],
                target_tokens=r['requested_completion_tokens'],
                canonical_prompt_sha256=r['canonical_prompt_sha256'],
                native_prompt_token_ids_sha256=n['native_prompt_token_ids_sha256'],
                native_prompt_tokens=int(n['actual_prompt_tokens']),
                artifact_content_sha256=content[r['adapter_id']],
                generation_contract=r['generation_contract'], native=n,
                native_entry_s=n['native_dispatch_monotonic_s'],
                arrival=ext['planned_arrival_s'], submit=ext['client_submit_s'],
                completed=n['controller_completed_monotonic_s'],
                stored=dict(ttft_ms=r['overall_ttft_ms'], tpot_ms=r['tpot_ms'], e2e_ms=r['overall_e2e_ms']))
    if len(result) != 4000:
        raise ValueError('full controller population required')
    return result


def matched_output_digest_audit(left, right, order):
    """Retain every same-backend output difference; missing digests cannot be equal."""
    if set(left) != set(order) or set(right) != set(order) or len(set(order)) != len(order):
        raise ValueError('complete unique digest audit population required')
    changed, by_adapter = [], {}
    for rid in order:
        a, b = left[rid], right[rid]
        matched_request_contract(a, b)
        digests = []
        for row in (a, b):
            value = row['native'].get('completion_token_ids_sha256')
            if (not isinstance(value, str) or len(value) != 64
                    or any(c not in '0123456789abcdef' for c in value)
                    or row['native']['native_output_tokens'] != row['target_tokens']):
                raise ValueError('complete native output count and SHA256 required')
            digests.append(value)
        differs = digests[0] != digests[1]
        counts = by_adapter.setdefault(a['adapter_id'], dict(total=0, changed=0))
        counts['total'] += 1
        counts['changed'] += int(differs)
        if differs:
            changed.append(rid)
    return dict(equal=len(order)-len(changed), changed=len(changed),
        changed_request_ids=changed, by_adapter=by_adapter,
        exact_output_equality_required=False, independent_numerical_proof=False,
        first_divergence_and_logits_available=False,
        explanation='Native sequence digests and bindings audited; no claim that differences are rounding or caused solely by batching.')


def compare_d424_sealed_w0(output, include_slora=False, include_loquetier=False,
                          include_elastic_locality=False):
    """Offline join of specified sealed observations; no new inference.

    Reuses the full native/content audits bound by SHA. It does not redownload
    already-audited and reclaimed objects, mutate raw unknown correctness flags,
    or require current mutable analysis code to match an old execution source.
    """
    import ijson
    from itertools import combinations
    from statistics import fmean
    from faaslora.datasets.workload_generator import FrozenReplayPlan
    from scripts.ieee_tc.audit_d402_serverlessllm_path import timing_stats

    if output.exists():
        raise ValueError('refusing to overwrite matched W0 analysis')
    if include_loquetier and not include_slora:
        raise ValueError('Loquetier extension must retain the sealed S-LoRA row')
    if include_elastic_locality and not (include_slora and include_loquetier):
        raise ValueError('ElasticLocality extension must retain all five sealed rows')
    refs = {}
    def checked(path, expected=None):
        path = Path(path)
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f'sealed source changed: {path}')
        refs[str(path)] = actual
        return path
    def checked_json(path, expected=None):
        return read(checked(path, expected))
    def canonical(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                        allow_nan=False).encode()).hexdigest()

    protocol = checked(ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md',
        '5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22')
    amendment = checked(ROOT/'docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md',
        '193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d')
    plan_sha = p.check_plan()
    freeze_path = ROOT/'paper_results/ieee_tc/protocols/20261010_7b_slo_d394_frozen.json'
    freeze = checked_json(freeze_path,
        '2d906f8aaf3fa1bf8cf30d116990db9970fb7be7bf1374fdbd1d8995090104f3')
    index_path = ROOT/'paper_results/ieee_tc/inputs/20260927_7b_materialized_content_index.json'
    index = checked_json(index_path,
        'e85cce3c3611da15530f9e663549231d3493282c85913344fe41c2a67044260c')
    content = {a['id']: canonical(sorted(a['files'], key=lambda f: f['path']))
               for a in index['artifacts']}
    if len(content) != 500:
        raise ValueError('frozen full 500-ID pool required')
    prime_path = ROOT/'paper_results/ieee_tc/p2_backend/20261010_d397_d371_single_run_review_v2.json'
    prime = checked_json(prime_path,
        'f48a0943810fb4519d7fa8fc3a284061efb1b05ff96cebad56c1cf6d8d40498e')
    source = checked(prime['source'], prime['source_sha256'])
    sources = {'PrimeLoRA': {x['path']: x['sha256'] for x in prime['source_refs']}}
    seals = {'PrimeLoRA': prime}
    run = source.parent
    locations = {'PrimeLoRA': (run/'launch.json',
        run/'launch.launch/physical_deployment/summary.json')}
    rows = {'PrimeLoRA': {}}
    baseline_specs = [
        ('ServerlessLLM', 'd414_7b_full_w0',
         '714a2fdb01b2d5a0e2e6f3a9b0faf7cb23b29632efbf5b343822b7dfe4b7dd74',
         '8912ab5259d7b10e990ab3b6233b789ce4432c0af2c1544498b7587b700904fd',
         'model7b_repaired', 'launch7b_repaired'),
        ('vLLM', 'd423_7b_full_w0',
         '7d9ac56d0db8cf3af0f87c7a5314eb272771ff6408c419bba616e8f7eceeb310',
         'b3796a4ea857e0a5cc8bb105397bd511f6c8548065b1c20a73ca9242975be516',
         'model7b', 'launch7b')]
    if include_slora:
        baseline_specs.append(('S-LoRA', 'd435_7b_slora_full_w0',
            '1c9361f2d995c27e0672611728697f4de48bd0ff3c4e9f4fd778523565b6e4c0',
            'd0f6b8b4aca2ac8b34f13230e9ebd559982ed7bce684441c7c5a998e8b0886c2',
            'model7b', 'launch7b'))
    if include_loquetier:
        baseline_specs.append(('Loquetier', 'd445_7b_loquetier_full_w0',
            '1fc16d0b15a201a8063142feb0ddc1c430cefed1fc69843927f8c8c170b27e31',
            '7057e39dddf78e370de620ef12cfb047c0923ac852db3dabd8a2eab593a067b8',
            'model7b', 'launch7b'))
    baseline_names = [spec[0] for spec in baseline_specs]
    for name, tag, audit_sha, manifest_sha, model, launch_name in baseline_specs:
        directory = ROOT/f'paper_results/ieee_tc/baseline_audit/{tag}'
        seal = checked_json(directory/'audit.json', audit_sha)
        manifest = checked_json(directory/'evidence_manifest.json', manifest_sha)
        sources[name] = {x['path']: x['sha256'] for x in manifest['files']}
        seals[name] = seal
        replay = checked(seal['replay_path'], seal['replay_sha256'])
        raw = replay.parent.parent
        locations[name] = (raw/(launch_name+'.json'), raw/model/'physical/summary.json')
        cfgpath = raw/'http_7b_4000.json'
        cfg = checked_json(cfgpath, sources[name][str(cfgpath)])
        if (cfg['generation_contract'] != 'fixed_length_greedy_v1'
                or cfg['pool_index_sha256'] != refs[str(index_path)]
                or cfg['artifact_source'] != 'published_gzip_ondemand_peft_v1'
                or cfg['ingress_mode'] != 'service_pre_ready_v1'
                or cfg['formal_run'] is not False):
            raise ValueError('baseline generation/pool/remote/initial-state identity differs')
        if name == 'vLLM' and (cfg['experiment_role'] != 'full_w0_resident_reference_candidate'
                              or seal['scheduler_contract'] != 'resident_rr_v1'):
            raise ValueError('preregistered resident reference identity required')
        if name == 'S-LoRA' and (cfg['experiment_role'] != 'full_w0_baseline_observation'
                                or seal['scheduler_contract'] != 'slora_resident_rr_v1'):
            raise ValueError('sealed S-LoRA native full-W0 identity required')
        if name == 'Loquetier' and (cfg['experiment_role'] != 'full_w0_baseline_observation'
                                   or seal['scheduler_contract'] != 'loquetier_static_shards_v1'):
            raise ValueError('sealed Loquetier native full-W0 identity required')
        byevent = {k: {} for k in ('request_contract', 'http_raw_response', 'http_response')}
        with replay.open() as handle:
            for line in handle:
                event = json.loads(line)
                if event['event'] in byevent:
                    dest = byevent[event['event']]
                    if event['request_id'] in dest:
                        raise ValueError('duplicate request/event')
                    dest[event['request_id']] = event
        if any(len(v) != 4000 for v in byevent.values()):
            raise ValueError('full three-way native/request/response join required')
        rows[name] = {}
        for rid, c in byevent['request_contract'].items():
            rawrow, response = byevent['http_raw_response'][rid], byevent['http_response'][rid]
            n = rawrow['body']['metrics']['ieee_tc']
            native_entry = matched_native_entry(name, c, n)
            if (rawrow['status'] != 200 or response['response']['protocol_valid'] is not True
                    or n['native_prompt_token_ids_sha256'] != c['native_prompt_token_ids_sha256']
                    or n['native_output_tokens'] != c['target_tokens']
                    or c['artifact_content_sha256'] != content[c['adapter_id']]):
                raise ValueError('native/content binding differs from sealed contract')
            rows[name][rid] = dict(c, native_prompt_tokens=len(n['native_prompt_token_ids']),
                generation_contract=cfg['generation_contract'], native=n,
                native_entry_s=native_entry,
                arrival=response['planned_arrival_s'], submit=response['client_submit_s'],
                completed=rawrow['client_completed_s'], stored=response['response'])
        del byevent

    with source.open('rb') as handle:
        metadata = next(ijson.items(handle, 'metadata', use_float=True))
    subset_path = checked(metadata['shared_adapter_subset_path'], metadata['shared_adapter_subset_sha256'])
    subset = read(subset_path)
    subset_ids = {a['id'] for a in subset['adapters']}
    if subset_ids != set(content):
        raise ValueError('PrimeLoRA subset and baseline frozen pool differ')
    rows['PrimeLoRA'] = matched_controller_rows(source, 'tc_ieee_full', content)
    if include_elastic_locality:
        locality = checked_json(ROOT/'paper_results/ieee_tc/elastic_locality/d453_actual_full_w0.json',
            '3cb2e9254941125f4f4d53465c682829fc0cac2127c1cf71c6b059b5e48be9de')
        if (locality['passed'] is not True or locality['protocol_bound_count'] != 4000
                or locality['fixed_workload_complete4000'] is not True
                or locality['input_witness']['passed'] is not True
                or locality['evidence']['elastic_locality']['passed'] is not True):
            raise ValueError('sealed full ElasticLocality qualification required')
        local_refs = locality['refs'] + locality['input_witness']['refs'] + locality['raw_analysis_refs']
        for ref in local_refs:
            checked(ref['path'], ref['sha256'])
        local_source = checked(locality['evidence']['source'], locality['evidence']['source_sha256'])
        physical_receipt = checked_json(locality['evidence']['physical_source'],
            locality['evidence']['physical_sha256'])
        checked(physical_receipt['summary_path'], physical_receipt['summary_sha256'])
        local_run = local_source.parent
        local_launch = local_run/'launch.json'
        sources['ElasticLocality'] = {**{r['path']:r['sha256'] for r in local_refs},
            physical_receipt['summary_path']: physical_receipt['summary_sha256']}
        locations['ElasticLocality'] = (local_launch, Path(physical_receipt['summary_path']))
        seals['ElasticLocality'] = locality
        rows['ElasticLocality'] = matched_controller_rows(local_source, 'tc_ieee_elastic_locality', content)
        baseline_names.append('ElasticLocality')

    summaries, launches, report_rows = {}, {}, {}
    common_plan = common_policy = None
    order = None
    for name, (launch_path, physical_path) in locations.items():
        launch = checked_json(launch_path, sources[name][str(launch_path)])
        physical = checked_json(physical_path, sources[name][str(physical_path)])
        launches[name], summaries[name] = launch, physical
        if not all(launch.get(k) is True for k in ('pass', 'service_path_removed', 'native_gpu_context_release_confirmed')):
            raise ValueError('successful actual closure required')
        ext, policy = launch['external_replay'], launch['preflight']['policy']
        if abs(ext['replay_t0_s']-ext['deployment_notice_s']-60) > 1e-6:
            raise ValueError('common deployment notice differs')
        trace = checked(ext['plan']['source_path'], ext['plan']['source_sha256'])
        replay_plan = FrozenReplayPlan.load(trace, profile='W0', rate_scale=1.)
        if replay_plan.identity() != ext['plan'] or len(replay_plan.entries) != 4000:
            raise ValueError('complete original W0 required')
        identity = {k:v for k,v in ext['plan'].items() if k != 'source_path'}
        if common_plan is None:
            common_plan, common_policy = identity, policy
            order = [x.request_id for x in replay_plan.entries]
        elif identity != common_plan or policy != common_policy:
            raise ValueError('common arrival or resource envelope differs')
        if set(rows[name]) != set(order):
            raise ValueError('offered population differs')
        for entry in physical['allocation_journals']:
            if entry['released'] is not True:
                raise ValueError('open physical allocation')
            checked(entry['path'], entry['sha256'])
        checked(physical['request_terminals_path'], physical['request_terminals_sha256'])
        gpu_s = closed_physical_integral(physical)
        sealed_gpu_s = (seals[name]['stats']['physical_gpu_seconds'] if name == 'ElasticLocality'
                        else seals[name]['physical_gpu_seconds'])
        if abs(gpu_s-sealed_gpu_s) > 1e-6:
            raise ValueError('resource total differs from sealed audit')
        metrics = {k:[] for k in ('ttft_ms','tpot_ms','e2e_ms','arrival_to_last_token_ms',
            'submit_lag_ms','submit_to_native_entry_ms','native_entry_to_first_ms',
            'decode_ms','completion_tail_ms')}
        good = Counter()
        phases = [Counter() for _ in range(8)]
        for ordinal, entry in enumerate(replay_plan.entries):
            r, original = rows[name][entry.request_id], json.loads(entry.source_json)
            r['arrival_offset_s'] = r['arrival']-ext['replay_t0_s']
            if (r['source_item_sha256'] != entry.source_sha256
                    or r['adapter_id'] != original['adapter_id']
                    or r['target_tokens'] != min(original['expected_output_tokens'],256)
                    or abs(r['arrival_offset_s']-entry.offset_s) > 1e-6):
                raise ValueError('source/adapter/target/arrival contract mismatch')
            n = r['native']
            a,e,d,f,l,c = (r['arrival'],r['submit'],r['native_entry_s'],
                n['native_first_token_monotonic_s'],n['native_last_token_monotonic_s'],r['completed'])
            if (n['native_clock_id'] != physical['clock_id'] or not n['native_terminal_observed']
                    or n['native_output_tokens'] != r['target_tokens']
                    or not all(math.isfinite(x) for x in (a,e,d,f,l,c))
                    or not a <= e <= d <= f <= l <= c):
                raise ValueError('native time/count contract differs')
            nt = r['target_tokens']
            values = dict(ttft_ms=(f-a)*1000, tpot_ms=(l-f)*1000/(nt-1) if nt>1 else None,
                e2e_ms=(c-a)*1000, arrival_to_last_token_ms=(l-a)*1000,
                submit_lag_ms=(e-a)*1000, submit_to_native_entry_ms=(d-e)*1000,
                native_entry_to_first_ms=(f-d)*1000, decode_ms=(l-f)*1000,
                completion_tail_ms=(c-l)*1000)
            for key in ('ttft_ms','tpot_ms','e2e_ms'):
                if values[key] is None:
                    if r['stored'][key] is not None:
                        raise ValueError('single-token TPOT must remain N/A')
                elif abs(values[key]-r['stored'][key]) > 1:
                    raise ValueError('native metric recomputation exceeds 1 ms')
            groups = [g for g in freeze['groups'] if
                (g['lower_exclusive'] is None or r['native_prompt_tokens'] > g['lower_exclusive']) and
                (g['upper_inclusive'] is None or r['native_prompt_tokens'] <= g['upper_inclusive'])]
            if len(groups) != 1:
                raise ValueError('one frozen SLO stratum required')
            tt = values['ttft_ms'] <= groups[0]['ttft_ms']
            tp = nt == 1 or values['tpot_ms'] <= groups[0]['tpot_ms']
            cts = dict(n=1, tokens=nt, ttft_pass=int(tt), tpot_pass=int(tp), joint_pass=int(tt and tp))
            good.update(cts); phases[ordinal//500].update(cts)
            for key,value in values.items():
                if value is not None:
                    metrics[key].append(value)
        expected = (seals[name]['common_threshold_joint_count'] if name == 'ElasticLocality' else
                    seals[name]['counts']['joint_pass'] if name == 'PrimeLoRA' else
                    seals[name]['common_slo']['joint_good'])
        if good['joint_pass'] != expected:
            raise ValueError('SLO differs from sealed audit')
        stats = {k:timing_stats(v) for k,v in metrics.items()}
        for key in ('ttft_ms','tpot_ms','e2e_ms'):
            if name == 'ElasticLocality':
                # The D453 seal stores TTFT/TPOT; E2E is recomputed above from
                # every native timeline and checked against every stored request.
                if key != 'e2e_ms' and any(abs(stats[key][stat]-seals[name]['stats'][prefix+key]) > 1e-6
                        for stat, prefix in (('mean', 'mean_'), ('p95_type1', 'p95_'))):
                    raise ValueError('ElasticLocality aggregate differs from sealed audit')
                continue
            prior = seals[name]['metrics' if name == 'PrimeLoRA' else 'conditional_metrics'][key]
            if any(abs(stats[key][x]-prior[x]) > 1e-6 for x in ('mean','p95_type1')):
                raise ValueError('latency aggregate differs from sealed audit')
        report_rows[name] = dict(counts=dict(good), observed_protocol_feasible=good['joint_pass']>=3800,
            joint_slo=good['joint_pass']/4000, physical_gpu_seconds=gpu_s,
            gpu_seconds_per_protocol_valid_request=gpu_s/4000,
            window_gpu_seconds=physical['window_gpu_seconds'], metrics=stats,
            source_order_500_request_phases=[dict(x) for x in phases],
            completion_boundary='controller completion' if name in ('PrimeLoRA','ElasticLocality') else 'external HTTP completion',
            raw_n_correct=physical['n_correct'], independent_numerical_proof=False)
        if include_slora:
            report_rows[name].update(
                native_entry_boundary=({'S-LoRA': 'S-LoRA model RPC registration',
                    'Loquetier': 'Loquetier native batch registration'}.get(name, 'vLLM native engine entry')),
                remote_transfers=(prime['remote']['server_transfers'] if name=='PrimeLoRA' else seals[name]['remote_transfers']),
                remote_wire_bytes=(prime['remote']['wire_bytes'] if name=='PrimeLoRA' else seals[name]['remote_wire_bytes']))

    output_changes, joined = {}, []
    contract_keys = ('request_id','source_item_sha256','adapter_id','target_tokens',
        'canonical_prompt_sha256','native_prompt_token_ids_sha256','native_prompt_tokens',
        'artifact_content_sha256','generation_contract')
    for rid in order:
        ref = rows['PrimeLoRA'][rid]
        for name in baseline_names:
            matched_request_contract(ref, rows[name][rid])
        joined.append({k:ref[k] for k in contract_keys})
    for left,right in combinations(rows,2):
        changed = [rid for rid in order if rows[left][rid]['native']['completion_token_ids_sha256']
                   != rows[right][rid]['native']['completion_token_ids_sha256']]
        output_changes[f'{left} / {right}'] = dict(equal=4000-len(changed), changed=len(changed),
            changed_request_ids=changed, exact_output_equality_required=False,
            explanation='Observed digests only; no assumption that all differences are rounding.')
    diffs = {}
    pr = report_rows['PrimeLoRA']
    for name in baseline_names:
        br = report_rows[name]
        diffs[name] = dict(gpu_seconds_relative_reduction=(br['physical_gpu_seconds']-pr['physical_gpu_seconds'])/br['physical_gpu_seconds'],
            joint_slo_difference_percentage_points=100*(pr['joint_slo']-br['joint_slo']),
            p95_ttft_relative_reduction=(br['metrics']['ttft_ms']['p95_type1']-pr['metrics']['ttft_ms']['p95_type1'])/br['metrics']['ttft_ms']['p95_type1'])
        if include_slora:
            diffs[name] = matched_relative_observation(pr, br)
    vr = report_rows['vLLM']
    if not vr['observed_protocol_feasible'] or len(summaries['vLLM']['device_intervals']) != 4:
        raise ValueError('eligible four-card complete Resident reference required')
    reference = dict(kind='single_run_resident_reference_v2', reference_id='7b_W0_vllm0300_D423',
        model=freeze['model'], workload='W0', reference_frozen=True,
        U_ref_gpu_seconds=vr['physical_gpu_seconds'], Q_ref_p95_ttft_ms=vr['metrics']['ttft_ms']['p95_type1'],
        budget_fraction=.75, B_gpu_seconds=.75*vr['physical_gpu_seconds'],
        n_protocol_valid=4000, raw_n_correct=None, independent_numerical_proof=False,
        eligibility='Sealed source/native/content/worker/physical audits plus full matched input join; not HTTP 200 alone.',
        warm_reference_sha256=freeze['calibration']['sha256'], slo_eval_sha256=refs[str(freeze_path)],
        metric_protocol_sha256=refs[str(protocol)], single_run_amendment_sha256=refs[str(amendment)],
        reused_development_observation=True, independent_holdout=False,
        scope='W0 only; W1/W2 references and G2 open; Resident is not resource-optimal vLLM.',
        source_audit_sha256=refs[str(ROOT/'paper_results/ieee_tc/baseline_audit/d423_7b_full_w0/audit.json')])
    if include_slora:
        reference_path = ROOT/'paper_results/ieee_tc/main_comparison/d424_7b_w0/resident_reference.json'
        reference = require_unchanged_reference(reference, checked_json(reference_path,
            '737c84804023c9f98677fb5f19c8240b4b05cabdb52f6ecdbbfd4527964c3269'))
        checked(ROOT/'docs/ieee_tc/BASELINE_EXECUTION_RULES_20261011.md')
        checked(ROOT/'tests/test_tc_serverless_qualification_population.py')
    checked(Path(__file__))
    checked(ROOT/'scripts/ieee_tc/audit_d402_serverlessllm_path.py')
    report = dict(kind='matched_sealed_w0_comparison_v1', current_plan_sha256=plan_sha,
        metric_protocol_sha256=refs[str(protocol)], single_run_amendment_sha256=refs[str(amendment)],
        common_slo_sha256=refs[str(freeze_path)], common_input=common_plan,
        common_resource_policy=common_policy, preparation_lead_s=60,
        matched_request_count=4000, matched_contract_sha256=canonical(joined),
        actual_unique_adapters=len({r['adapter_id'] for r in joined}),
        native_tokens=sum(r['target_tokens'] for r in joined), pool_index_sha256=refs[str(index_path)],
        results=report_rows, PrimeLoRA_relative_observations=diffs, output_digest_audit=output_changes,
        reference=reference, confidence_interval=None, formal_m1_m2_complete=False,
        caveats=['Known W0 development observations, one per configuration; no independent holdout or statistical claim.',
            'Different native batching/cache capacities and ServerlessLLM backend version; whole-system points, not mechanism isolation.',
            'PrimeLoRA controller completion differs from baseline HTTP completion; arrival-to-last-token is a common supplemental endpoint, not renamed E2E.',
            'Published remote protocol shared; all three made 132 HTTP transfers/131480060 bytes. No lower remote traffic benefit in this W0.',
            'Mostly zero/repeated LoRA weights; binding plus bounded independent evidence is not full arithmetic proof.',
            'PrimeLoRA remote request-level phase join covers 131/132 deliveries; total server bytes remain known.',
            'Native compiler disk caches differ; no prestarted GPU, no OS-wide page-cache flush.',
            'W1/W2, optimized baseline resource points, remaining systems, G2 and mechanism evidence are open.'],
        source_refs=[dict(path=k,sha256=v) for k,v in refs.items()])
    if include_slora:
        report['kind'] = 'matched_sealed_w0_comparison_v2'
        report['resident_reference_reused_unchanged'] = True
        report['caveats'][3] = ('Published remote protocol shared. PrimeLoRA/ServerlessLLM/vLLM: '
            '132 transfers/131480060 wire bytes each; S-LoRA full static preparation: '
            '500 transfers/420740410 bytes. Different preparation strategies, not isolated hierarchy evidence.')
        report['caveats'][-1] = ('W1/W2, remaining systems, G2 and mechanism evidence remain open. '
            'No baseline performance search is authorized; measured baseline points are not asserted optimal.')
        report['caveats'].extend([
            'S-LoRA remains in the complete table despite joint SLO below 95%; raw differences are not a two-feasible-system G1 comparison.',
            'S-LoRA native entry is model RPC registration; vLLM-based rows use engine entry. Phase columns are descriptive, not identical causal boundaries.',
            'S-LoRA first-500/later-request slices are not independent repeats and never replace the full W0 denominator.'])
    if include_loquetier:
        previous = checked_json(ROOT/'paper_results/ieee_tc/main_comparison/d436_7b_w0/comparison.json',
            '59a6dfa6b3a512ac561c65fb334de4843895c5c7f120e709302d50afd6704427')
        if (any(report_rows[name] != row for name, row in previous['results'].items())
                or any(diffs[name] != row for name, row in previous['PrimeLoRA_relative_observations'].items())
                or report['matched_contract_sha256'] != previous['matched_contract_sha256']):
            raise ValueError('previous four-system matched observations changed')
        report['kind'] = 'matched_sealed_w0_comparison_v3'
        report['previous_four_system_rows_unchanged'] = True
        report['caveats'][3] = ('Published remote protocol shared. PrimeLoRA/ServerlessLLM/vLLM: '
            '132 transfers/131480060 wire bytes each; S-LoRA and Loquetier full static preparation: '
            '500 transfers/420740410 bytes each. Different preparation strategies, not isolated hierarchy evidence.')
        report['caveats'].extend([
            'Loquetier retains native SMLM with disclosed static four-way 125-ID shards and batch2; not an official adaptive cluster scheduler or an optimized baseline.',
            'Loquetier remains in the table with 4.35% joint SLO; it is outside the observed G1 feasible set.',
            'Loquetier native entry is batch registration, not vLLM engine entry or S-LoRA model RPC registration; native scheduling is separately observed, not substituted.',
            'Queue-stage observations do not isolate a causal sharding, hierarchy or confirmed-state effect.'])
        report['source_refs'] = [dict(path=k, sha256=v) for k,v in refs.items()]
    if include_elastic_locality:
        previous = checked_json(ROOT/'paper_results/ieee_tc/main_comparison/d446_7b_w0/comparison.json',
            '95a1037c494b87d1c52c151bc01850b5a96c9cf578f6af3d117a40d73e4b6edf')
        if (any(report_rows[name] != row for name, row in previous['results'].items())
                or any(diffs[name] != row for name, row in previous['PrimeLoRA_relative_observations'].items())
                or report['matched_contract_sha256'] != previous['matched_contract_sha256']):
            raise ValueError('previous five-system matched observations changed')
        report['kind'] = 'matched_sealed_w0_comparison_v4'
        report['previous_five_system_rows_unchanged'] = True
        report['same_backend_output_audit'] = matched_output_digest_audit(
            rows['PrimeLoRA'], rows['ElasticLocality'], order)
        report['caveats'][3] = ('Published remote protocol shared. PrimeLoRA/ElasticLocality/ServerlessLLM/vLLM: '
            '132 transfers/131480060 wire bytes each; S-LoRA/Loquetier:500/420740410. '
            'No isolated remote-transfer reduction from Full against ElasticLocality in W0.')
        report['caveats'].extend([
            'ElasticLocality is a composite internal control, not a single-factor ablation or an external-paper baseline.',
            'Same-backend digest changes are retained with per-adapter counts; no logits/first-divergence proof and no automatic rounding explanation.',
            'Neither W0 timing feasibility nor lower GPU time establishes W1/W2 feasibility or G2.'])
        report['source_refs'] = [dict(path=k, sha256=v) for k,v in refs.items()]
    report['analysis_key'] = canonical(dict(source_refs=report['source_refs'], protocol=refs[str(protocol)],
                                          amendment=refs[str(amendment)]))
    output.mkdir(parents=True, exist_ok=False)
    for filename,value in (('comparison.json',report),('resident_reference.json',reference)):
        with (output/filename).open('x') as handle:
            json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n')
    with (output/'metrics.csv').open('x',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['system','run','n_protocol_valid','joint_slo','gpu_seconds','gpu_seconds_per_request',
                         'mean_ttft_ms','p95_ttft_ms','mean_tpot_ms','p95_tpot_ms','mean_e2e_ms','completion_boundary'])
        tags = dict(PrimeLoRA='D371/D397', ServerlessLLM='D414', vLLM='D423',
                    Loquetier='D445', ElasticLocality='D453', **{'S-LoRA':'D435'})
        for name in report_rows:
            r=report_rows[name];m=r['metrics']
            writer.writerow([name,tags[name],4000,r['joint_slo'],r['physical_gpu_seconds'],r['gpu_seconds_per_protocol_valid_request'],
                m['ttft_ms']['mean'],m['ttft_ms']['p95_type1'],m['tpot_ms']['mean'],m['tpot_ms']['p95_type1'],
                m['e2e_ms']['mean'],r['completion_boundary']])
    print(json.dumps(dict(output=str(output),matched=4000,reference=reference,differences=diffs)))


def published_w0_view(report, registry):
    """Offline, lossless row classification; never change execution or SLOs.

    The thresholds below are the already approved sensitivity grid, NOT new
    main-point selection or additional experiments. W0 cannot certify W1/W2 G2.
    """
    if (report['matched_request_count'] != 4000 or report['native_tokens'] != 458224
            or report['common_input']['profile'] != 'W0'
            or report['formal_m1_m2_complete'] is not False
            or report['confidence_interval'] is not None):
        raise ValueError('complete known W0 single-run observation required')
    catalog = registry['systems']
    rows, variants = {}, {}
    for name, row in report['results'].items():
        fact = catalog[name]  # Missing publication identity must not silently drop a row.
        if fact['role'] == 'subject':
            if name != 'PrimeLoRA':
                raise ValueError('unexpected subject')
        elif fact['role'] == 'external_baseline':
            if (fact['published'] is not True or not fact['venue']
                    or type(fact['year']) is not int or not fact['source'].startswith('https://')):
                raise ValueError('verified publication metadata required')
        elif fact['role'] == 'composite_ablation':
            variants[name] = copy.deepcopy(row)
            continue
        else:
            raise ValueError('unknown comparison role')
        counts = row['counts']
        if ('N' in counts and counts['N'] != 4000) or ('n' in counts and counts['n'] != 4000):
            raise ValueError('complete 4000-row population required')
        if not {'N', 'n'} & counts.keys():
            raise ValueError('explicit population required')
        joint = counts['joint_pass']
        if (type(joint) is not int or not 0 <= joint <= 4000
                or row['joint_slo'] != joint / 4000
                or row['observed_protocol_feasible'] is not (joint >= 3800)):
            raise ValueError('joint count/ratio/frozen feasibility mismatch')
        u, r = row['physical_gpu_seconds'], row['gpu_seconds_per_protocol_valid_request']
        q = row['metrics']['ttft_ms']['p95_type1']
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in (u, r, q)):
            raise ValueError('finite positive observed resource and tail required')
        if not math.isclose(r * 4000, u, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError('resource denominator mismatch')
        # Preserve unknown proof rather than inferring correctness from completion.
        if 'raw_n_correct' not in row or 'independent_numerical_proof' not in row:
            raise ValueError('explicit numerical evidence boundary required')
        rows[name] = copy.deepcopy(row)
    if 'PrimeLoRA' not in rows:
        raise ValueError('subject missing')
    reference = report['reference']
    uref = reference['U_ref_gpu_seconds']
    if (reference['reference_frozen'] is not True or reference['workload'] != 'W0'
            or type(uref) not in (int, float) or not math.isfinite(uref) or uref <= 0):
        raise ValueError('finite frozen W0 reference required')
    sensitivity = []
    for attainment in (.90, .95, .99):
        feasible = [name for name, row in rows.items()
                    if row['counts']['joint_pass'] >= math.ceil(attainment * 4000)]
        sensitivity.append(dict(joint_attainment_requirement=attainment,
            observed_feasible_resource_order=sorted(feasible, key=lambda n: (rows[n]['physical_gpu_seconds'], n)),
            budget_slices=[dict(fraction=fraction, budget_gpu_seconds=fraction * uref,
                observed_feasible_names=[name for name in feasible
                    if rows[name]['physical_gpu_seconds'] <= fraction * uref])
                for fraction in (.5, .75, 1.)]))
    shared = ('matched_request_count', 'matched_contract_sha256', 'native_tokens',
        'actual_unique_adapters', 'common_input', 'common_resource_policy', 'common_slo_sha256',
        'metric_protocol_sha256', 'single_run_amendment_sha256', 'reference',
        'formal_m1_m2_complete', 'confidence_interval', 'independent_numerical_qualification')
    view = {key: copy.deepcopy(report[key]) for key in shared}
    view.update(kind='published_baseline_known_w0_view_v1', publication_registry_id=registry['id'],
        results=rows, publication_facts={name: copy.deepcopy(catalog[name]) for name in rows},
        pending_published_baselines={name: copy.deepcopy(fact) for name, fact in catalog.items()
            if fact['role'] == 'external_baseline' and name not in rows},
        PrimeLoRA_relative_observations={name: matched_relative_observation(rows['PrimeLoRA'], row)
            for name, row in rows.items() if name != 'PrimeLoRA'},
        observed_feasible_resource_order=sensitivity[1]['observed_feasible_resource_order'],
        fixed_configuration_sensitivity=sensitivity,
        caveats=['Post-result publication eligibility revision requested by user, not a measured improvement.',
            'Rows retained byte-equivalent as JSON values; source report and all excluded variant evidence remain available.',
            'No new inference, retuning, SLO threshold change, CI or independent holdout claim.',
            'Publication is not proof of mechanism coverage or numerical correctness; retain source run limitations.',
            'Sensitivity only classifies existing fixed configurations, not optimal achievable frontiers.',
            'Budget slices are W0 diagnostics only; W1/W2 and formal G2 remain incomplete.'])
    mechanism = dict(kind='retained_composite_ablation_evidence_v1', results=variants,
        external_baseline=False, formal_g1_ranking=False,
        caveat='Historical observations retained unchanged; not a single-factor causal experiment.')
    return view, mechanism


def export_published_w0(source, expected_sha, registry_path, output):
    if sha256(source) != expected_sha:
        raise ValueError('sealed source report SHA mismatch')
    report, registry = read(source), read(registry_path)
    view, mechanism = published_w0_view(report, registry)
    refs = [dict(path=str(source.resolve()), sha256=expected_sha),
            dict(path=str(registry_path.resolve()), sha256=sha256(registry_path)),
            dict(path=str(Path(__file__).resolve()), sha256=sha256(Path(__file__)))]
    for value in (view, mechanism):
        value['source_refs'] = refs
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (('comparison.json', view), ('mechanism_evidence.json', mechanism)):
        with (output/name).open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(output=str(output), systems=list(view['results']),
        observed_feasible_resource_order=view['observed_feasible_resource_order'])))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', choices=('d299', 'd300', 'd303', 'd314', 'd315', 'd318', 'd323', 'd326', 'd332', 'd335', 'd336', 'd341', 'd346', 'd349', 'd353', 'd358', 'd364', 'd367', 'd309-failure', 'd311-incomplete', 'd313-failure', 'd364-failure'), default='d299')
    parser.add_argument('--reference', choices=('d303', 'd314', 'd349', 'd336', 'd353', 'd358'),
        help='Registered full reference; separate non-overwriting output')
    parser.add_argument('--single-run-source', type=Path)
    parser.add_argument('--warm-reference', type=Path)
    parser.add_argument('--review-output', type=Path)
    parser.add_argument('--published-w0-source', type=Path)
    parser.add_argument('--published-w0-source-sha256')
    parser.add_argument('--publication-registry', type=Path)
    parser.add_argument('--published-w0-output', type=Path)
    parser.add_argument('--matched-w0-output', type=Path,
        help='D424 offline join of sealed D371/D397, D414 and D423; exclusive output directory')
    parser.add_argument('--include-sealed-slora', action='store_true',
        help='D436: also join sealed D435, retaining nonfeasibility and the existing D424 reference')
    parser.add_argument('--include-sealed-loquetier', action='store_true',
        help='D446: add sealed D445; requires the full four-system join and unchanged reference')
    parser.add_argument('--include-sealed-elastic-locality', action='store_true',
        help='D454: add sealed D453, preserving all five prior rows and native-output audit')
    args = parser.parse_args()
    publication_args = (args.published_w0_source, args.published_w0_source_sha256,
                        args.publication_registry, args.published_w0_output)
    if any(value is not None for value in publication_args):
        if not all(value is not None for value in publication_args):
            parser.error('publication view requires source, explicit source SHA, registry and new output')
        if args.matched_w0_output is not None or args.single_run_source is not None:
            parser.error('publication view cannot be combined with another analysis mode')
        export_published_w0(*publication_args)
        raise SystemExit(0)
    if args.include_sealed_elastic_locality and not (args.include_sealed_loquetier and args.include_sealed_slora):
        parser.error('--include-sealed-elastic-locality requires both sealed baseline extensions')
    if args.include_sealed_loquetier and not args.include_sealed_slora:
        parser.error('--include-sealed-loquetier requires --include-sealed-slora')
    if args.include_sealed_slora and args.matched_w0_output is None:
        parser.error('--include-sealed-slora requires --matched-w0-output')
    if args.matched_w0_output is not None:
        compare_d424_sealed_w0(args.matched_w0_output, include_slora=args.include_sealed_slora,
                              include_loquetier=args.include_sealed_loquetier,
                              include_elastic_locality=args.include_sealed_elastic_locality)
        raise SystemExit(0)
    if args.single_run_source is not None:
        if args.warm_reference is None or args.review_output is None:
            parser.error('single-run review requires explicit warm reference and output')
        review_single_run(args.single_run_source, args.warm_reference, args.review_output)
        raise SystemExit(0)
    campaign = args.campaign
    if args.reference is not None and args.reference not in REGISTERED_REFERENCES.get(campaign, ()):
        parser.error('--reference is not registered for this campaign')
    if campaign == 'd311-incomplete':
        audit_d311_incomplete()
    elif campaign == 'd309-failure':
        audit_d309_failure()
    elif campaign == 'd313-failure':
        audit_d309_failure('d313')
    elif campaign == 'd364-failure':
        audit_d309_failure('d364')
    else:
        main(campaign, args.reference)
