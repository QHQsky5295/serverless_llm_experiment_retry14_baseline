#!/usr/bin/env python3
"""D297/D299/D300/D303/D314/D315 data-only audit using native and timing checks.

This is an offline driver, not a serving change, new benchmark, or qualification.
Use a bounded auxiliary scope after all inference processes have exited.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ieee_tc.plot_d290_diagnostic import audit_run, binding_checks, percentile, sha256
from scripts.analyze_control_path_overhead import (provisional_timing_bound,
    summarize_admission_waits, summarize_route_copy_revisits)
from faaslora.metrics.metrics_collector import NativeV1TokenTimeline

RAW = ROOT / 'results/ieee_tc/p2_backend_qualification/d297_20261007'
RUN = RAW / '7b_full_w0'
OUT = ROOT / 'paper_results/ieee_tc/p2_backend/d297_full_w0/validated1'
SOURCE = RUN / 'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d297_7b_full_w0.json'
WARM = ROOT / 'paper_results/ieee_tc/p2_backend/20261002_d170_warm_measurement.json'
INDEX = ROOT / 'paper_results/ieee_tc/inputs/20260927_7b_materialized_content_index.json'
PLAN = Path('/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md')
PROTOCOL = ROOT / 'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md'


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    with (OUT / name).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def table(name, rows):
    with (OUT / name).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def audit_preparation_attribution(requests, *, clock_id):
    """Recompute retained copy/stage evidence; not an independent arithmetic test."""
    from faaslora.preloading.preloading_planner import observed_preparation_interval
    reasons, tiers, changed, seen = Counter(), Counter(), [], set()
    eligible = 0
    for row in requests:
        rid = row['request_id']
        if not rid or rid in seen or row['success'] is not True:
            raise ValueError('complete unique successful request observations required')
        seen.add(rid)
        evidence = row['gpu_reference_evidence']
        admission = evidence['source_admission']
        tier = admission['source']['tier']
        tiers[tier] += 1
        if tier == 'gpu':
            if 'preparation_interval' in evidence:
                raise ValueError('GPU-ready request must not invent a load sample')
            continue
        expected = observed_preparation_interval(adapter_id=row['adapter_id'],
            request_id=rid, admission=admission, native=evidence['receipt'],
            remote=evidence.get('remote_preparation'),
            file_reference=evidence.get('local_source_reference'), expected_clock_id=clock_id)
        actual = evidence['preparation_interval']
        extras = {'preparation_class', 'preparation_profile_id', 'cost_model_updated'}
        if (set(actual) != set(expected) | extras
                or any(actual[k] != v for k, v in expected.items())
                or actual['preparation_class'] != admission['preparation_class']
                or actual['preparation_profile_id'] != admission['preparation_profile_id']
                or actual['cost_model_updated'] is not expected['profile_eligible']):
            raise ValueError('stored preparation attribution differs from its witnessed stages')
        reasons[actual['reason']] += 1
        eligible += actual['profile_eligible']
        if actual['reason'] == 'confirmed_file_source_changed_before_loading':
            changed.append(dict(request_id=rid, adapter_id=row['adapter_id'],
                remote_transfer_id=actual['remote_transfer_id'],
                remote_target_path=actual['remote_target_path'], native_lora_path=actual['native_lora_path']))
    if not seen:
        raise ValueError('empty request population')
    return dict(requests=len(seen), source_tiers=dict(tiers), reasons=dict(reasons),
        eligible_samples=eligible, ineligible_samples=sum(reasons.values())-eligible,
        changed_copy_cases=changed, numerical_qualification=False,
        caveat='Consistency of recorded ownership/stages only; no subtraction from service or user latency.')


def decode_overlap_summary(requests, activations, *, clock_id, arrival_start, thresholds):
    """D304 descriptive intervals, NOT kernel stalls or a causal contrast."""
    def finite(v):
        return type(v) in (int, float) and math.isfinite(v)

    def union_seconds(spans):
        total, end = 0.0, -math.inf
        for a, b in sorted(spans):
            total += max(0.0, b-max(a, end))
            end = max(end, b)
        return total

    if not finite(arrival_start) or not requests or not activations:
        raise ValueError('complete request and activation observations required')
    starts, ids, records = [], set(), []
    for a in activations:
        identity = a['activation_id']
        begin, end = a['trigger_at'], a['engine_ready_at']
        if (identity in ids or a['target_replica'] != identity or a['clock_id'] != clock_id
                or a['category'] not in ('initial', 'natural_scaleout', 'controlled')
                or not finite(begin) or not finite(end) or end < begin):
            raise ValueError('invalid or mixed-clock activation interval')
        ids.add(identity)
        starts.append(dict(replica=identity, start=begin, end=end, category=a['category']))
    for r in requests:
        t = r['native_token_timing']
        admitted, dispatched, first, last = [t[k] for k in (
            'controller_admitted_monotonic_s', 'native_dispatch_monotonic_s',
            'native_first_token_monotonic_s', 'native_last_token_monotonic_s')]
        n, prompt = r['completion_tokens'], t['actual_prompt_tokens']
        if (r['success'] is not True or r['instance_id'] not in ids
                or t['native_clock_id'] != clock_id or type(n) is not int or not 2 <= n <= 256
                or not all(finite(x) for x in (admitted, dispatched, first, last, prompt, r['tpot_ms']))
                or not admitted <= dispatched <= first < last or int(prompt) != prompt):
            raise ValueError('complete ordered native decode intervals required')
        if dispatched < next(s['end'] for s in starts if s['replica']==r['instance_id']):
            raise ValueError('native dispatch precedes replica readiness')
        tpot = (last-first)*1000/(n-1)
        if abs(tpot-r['tpot_ms']) > 1:
            raise ValueError('native TPOT identity mismatch')
        group = [g for g in thresholds if (g['lower_exclusive'] is None or prompt > g['lower_exclusive'])
                 and (g['upper_inclusive'] is None or prompt <= g['upper_inclusive'])]
        if len(group) != 1:
            raise ValueError('prompt does not map to one frozen group')
        tier = r['readiness_tier_before_dispatch']
        if tier not in ('gpu', 'host', 'nvme', 'remote'):
            raise ValueError('missing selected tier')
        records.append(dict(request_id=r['request_id'], replica=r['instance_id'],
            prompt_tokens=int(prompt), tokens=n, output_group='2-32' if n<=32 else '33-128' if n<=128 else '129-256',
            tier=tier, admitted=admitted, dispatched=dispatched, first=first, last=last,
            first_offset_s=first-arrival_start, decode_s=last-first, tpot_ms=tpot,
            timing_tpot_miss=tpot>group[0]['tpot_s']*1000))
    if len({r['request_id'] for r in records}) != len(records):
        raise ValueError('duplicate request identity')
    for r in records:
        a, b = r['first'], r['last']
        startup = [(max(a, s['start']), min(b, s['end'])) for s in starts
                   if s['replica'] != r['replica'] and max(a,s['start']) < min(b,s['end'])]
        peers = [q for q in records if q['replica']==r['replica'] and q['request_id']!=r['request_id']]
        preparation = [(max(a,q['admitted']),min(b,q['dispatched'])) for q in peers
                       if q['tier']!='gpu' and max(a,q['admitted']) < min(b,q['dispatched'])]
        occupancy = [(max(a,q['dispatched']),min(b,q['last'])) for q in peers
                     if max(a,q['dispatched']) < min(b,q['last'])]
        r.update(other_startup_overlap_s=union_seconds(startup),
            same_replica_preparation_envelope_s=union_seconds(preparation),
            other_native_request_seconds=sum(y-x for x,y in occupancy))
        r['mean_other_native_requests']=r['other_native_request_seconds']/r['decode_s']
        r['startup_overlap']=bool(startup)
        r['preparation_overlap']=bool(preparation)
    groups=[]
    for name, labels in [('all',['all']), ('startup_overlap',[False,True]),
                         ('preparation_overlap',[False,True]),
                         ('output_group',['2-32','33-128','129-256'])]:
        for label in labels:
            rs=[r for r in records if name=='all' or r[name]==label]
            groups.append(dict(group=name,label=label,count=len(rs),
                timing_tpot_misses=sum(r['timing_tpot_miss'] for r in rs),
                mean_tpot_ms=mean(r['tpot_ms'] for r in rs) if rs else None,
                p95_tpot_ms=percentile([r['tpot_ms'] for r in rs],.95) if rs else None,
                mean_other_native_requests=mean(r['mean_other_native_requests'] for r in rs) if rs else None))
    # Cross-tab retains length strata; no claim that conditioning removes confounding.
    for length in ('2-32','33-128','129-256'):
        for overlap in (False,True):
            rs=[r for r in records if r['output_group']==length and r['startup_overlap']==overlap]
            groups.append(dict(group='output_x_startup',label=f'{length}/{overlap}',count=len(rs),
                timing_tpot_misses=sum(r['timing_tpot_miss'] for r in rs),
                mean_tpot_ms=mean(r['tpot_ms'] for r in rs) if rs else None,
                p95_tpot_ms=percentile([r['tpot_ms'] for r in rs],.95) if rs else None,
                mean_other_native_requests=mean(r['mean_other_native_requests'] for r in rs) if rs else None))
    return dict(requests=records,groups=groups,activations=starts,
        startup_union_s=union_seconds([(max(arrival_start,s['start']),s['end']) for s in starts if s['end']>arrival_start]),
        n_correct=None,formal_qualification=False,confidence_interval=None,
        caveats=['Same-host native monotonic intervals only.',
                 'Startup overlap is an exposure, not the duration or cause of a decode stall.',
                 'Preparation envelope includes waiting/materialization; not isolated device-copy time.',
                 'Overlap depends on observed decode duration; long requests have more opportunity for exposure.',
                 'Conditional groups differ; no causal effect, independent samples, or counterfactual subtraction.',
                 'Provisional TPOT thresholds unchanged; no formal SLO qualification.'])


def inspect_retained_decode(output):
    """Reuse D300/D303 complete runs; no serving, workload or figure changes."""
    global OUT
    OUT=Path(output).resolve()
    if OUT.exists():
        raise FileExistsError(OUT)
    assert sha256(PLAN)=='0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    assert sha256(PROTOCOL)=='5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    summaries, allgroups, paths={}, [], [PLAN, PROTOCOL, Path(__file__)]
    OUT.mkdir(parents=True,exist_ok=False)
    for name in ('d300','d303'):
        curated=ROOT/f'paper_results/ieee_tc/p2_backend/{name}_full_w0/validated1'
        manifest=read(curated/'manifest.json')
        for filename,digest in manifest['outputs'].items():
            assert sha256(curated/filename)==digest
        audit=read(curated/'audit.json')
        source=Path(audit['evidence']['source'])
        assert sha256(source)==audit['evidence']['source_sha256']
        _,evidence,requests=audit_run(source)
        assert len(requests)==4000
        deployment_path=source.parent/'launch.launch/physical_deployment/deployment.json'
        deployment=read(deployment_path)
        outcome=source.parent/'launch.launch/physical_deployment/main_outcome.json'
        query='fromstream(2|truncate_stream(inputs | select(.[0][:2] == ["mechanism_events","_ieee_activations"])))'
        activations=json.loads(subprocess.check_output(['jq','-cn','--stream',query,str(outcome)]))
        result=decode_overlap_summary(requests,activations,clock_id=deployment['clock_id'],
            arrival_start=deployment['arrival_start_s'],thresholds=audit['thresholds'])
        table(f'{name}_request_intervals.csv',result.pop('requests'))
        allgroups.extend(dict(run=name,**r) for r in result['groups'])
        result['source']=evidence
        summaries[name]=result
        paths.extend([curated/'audit.json',curated/'manifest.json',source,deployment_path,outcome])
        del requests
        gc.collect()
        print(json.dumps(dict(run=name,groups=result['groups']),indent=2),flush=True)
    write('summary.json',dict(kind='d304_retained_decode_overlap_v1',runs=summaries,new_performance_runs=0,no_figures=True))
    table('groups.csv',allgroups)
    write('manifest.json',dict(source_refs=[dict(path=str(p),sha256=sha256(p)) for p in paths],
        outputs={p.name:sha256(p) for p in sorted(OUT.iterdir())},no_figures=True))


def native_load_overlap_summary(requests, decode_records, *, clock_id):
    """D305: intersect witnessed owner transactions, not the broad preparation span.

    The load interval includes acquire's completion fence; its components must
    not be added again. A decode-window overlap is NOT a measured token stall.
    """
    def finite(v):
        return type(v) in (int, float) and math.isfinite(v)

    def union_overlap(a, b, peers, start_key, end_key):
        spans = sorted((max(a, q[start_key]), min(b, q[end_key])) for q in peers
            if q[start_key] is not None and max(a, q[start_key]) < min(b, q[end_key]))
        total, end = 0.0, -math.inf
        for x, y in spans:
            total += max(0., y-max(x, end))
            end = max(end, y)
        return total

    records = [dict(r) for r in decode_records]
    by_id = {r['request_id']: r for r in records}
    if (not requests or len(by_id) != len(records) or len(requests) != len(records)
            or {r['request_id'] for r in requests} != set(by_id)):
        raise ValueError('complete unique request and retained decode identities required')
    owners, replicas, leases, owner_rows = {}, {}, set(), {}
    for q in requests:
        r = by_id[q['request_id']]
        n, e = q['native_token_timing'], q['gpu_reference_evidence']['receipt']
        owner, replica = e['owner_id'], q['instance_id']
        lease, pid = e['lease_id'], e['worker_pid']
        for key, native_key in (
            ('admitted', 'controller_admitted_monotonic_s'), ('dispatched', 'native_dispatch_monotonic_s'),
            ('first', 'native_first_token_monotonic_s'), ('last', 'native_last_token_monotonic_s')):
            if not finite(n[native_key]) or n[native_key] != r[key]:
                raise ValueError('retained decode and native interval mismatch')
        acquired, fence = e['acquired_monotonic_s'], e['completion_fence_ms']
        start, end = e['native_load_started_monotonic_s'], e['native_load_completed_monotonic_s']
        invoked, gpu_hit = e['native_load_invoked'], e['gpu_resident_before_load']
        if (q['success'] is not True or e['acquired'] is not True
                or not owner or not lease or lease in leases or type(pid) is not int or pid <= 0
                or e['worker_rank'] != 0 or e['acquisition_operation'] != 'demand_load_and_acquire'
                or e['completion_fence_scope'] != 'current_worker_cuda_stream'
                or not (e['clock_id'] == n['native_clock_id'] == clock_id)
                or n['gpu_reference_owner_id'] != owner or n['gpu_reference_lease_id'] != lease
                or r['replica'] != replica or q['completion_tokens'] != r['tokens']
                or not finite(acquired) or not finite(fence) or fence < 0
                # D184's selected-GPU protection precedes controller admission.
                # Admission is not a lower bound on the witnessed transaction.
                or not 0 <= acquired-fence/1000 <= acquired <= r['dispatched'] <= r['first'] < r['last']
                or type(invoked) is not bool or type(gpu_hit) is not bool or invoked == gpu_hit
                or e['source_tier_before_acquisition'] not in ('gpu', 'host', 'staging', 'file')
                or not finite(e['load_and_acquire_ms']) or e['load_and_acquire_ms'] < fence):
            raise ValueError(f"invalid native owner/load/acquire witness {q['request_id']}: "
                f"owner={owner}, pid={pid}, rank={e['worker_rank']}, "
                f"operation={e['acquisition_operation']}, scope={e['completion_fence_scope']}, "
                f"admitted={r['admitted']}, acquired={acquired}, dispatch={r['dispatched']}, "
                f"fence_ms={fence}, load_and_acquire_ms={e['load_and_acquire_ms']}, "
                f"invoked={invoked}, gpu_hit={gpu_hit}, source={e['source_tier_before_acquisition']}")
        if invoked:
            if (not finite(start) or not finite(end) or end != acquired
                    or not 0 <= start <= acquired-fence/1000
                    or (end-start)*1000 > e['load_and_acquire_ms'] + 1e-6):
                raise ValueError('invalid witnessed load interval')
        elif start is not None or end is not None:
            raise ValueError('GPU-hit load timestamps must remain null')
        if owner in owners and owners[owner] != (replica, pid):
            raise ValueError('one owner maps to multiple worker incarnations')
        if replica in replicas and replicas[replica] != owner:
            raise ValueError('one activation maps to multiple owners')
        owners[owner], replicas[replica] = (replica, pid), owner
        leases.add(lease)
        r.update(owner=owner, worker_pid=pid, native_load_invoked=invoked,
            native_source=e['source_tier_before_acquisition'], load_start=start, load_end=end,
            load_body_end=acquired-fence/1000 if invoked else None,
            fence_start=acquired-fence/1000, acquired=acquired, fence_ms=fence,
            native_load_ms=(end-start)*1000 if invoked else None,
            load_and_acquire_ms=e['load_and_acquire_ms'])
        owner_rows.setdefault(owner, []).append(r)
    for r in records:
        peers = [q for q in owner_rows[r['owner']] if q['request_id'] != r['request_id']]
        for name, start, end in (
            ('load', 'load_start', 'load_end'), ('load_body', 'load_start', 'load_body_end'),
            ('fence', 'fence_start', 'acquired')):
            overlap = union_overlap(r['first'], r['last'], peers, start, end)
            r[f'{name}_overlap_s'] = overlap
            r[f'{name}_overlap'] = overlap > 0
            r[f'{name}_decode_fraction'] = overlap/r['decode_s']
    groups = []

    def add_group(name, label, rs):
        groups.append(dict(group=name, label=label, count=len(rs),
            timing_tpot_misses=sum(r['timing_tpot_miss'] for r in rs),
            mean_tpot_ms=mean(r['tpot_ms'] for r in rs) if rs else None,
            p95_tpot_ms=percentile([r['tpot_ms'] for r in rs], .95) if rs else None,
            mean_other_native_requests=mean(r['mean_other_native_requests'] for r in rs) if rs else None,
            mean_load_decode_fraction=mean(r['load_decode_fraction'] for r in rs) if rs else None,
            mean_fence_decode_fraction=mean(r['fence_decode_fraction'] for r in rs) if rs else None))

    add_group('all', 'all', records)
    for name in ('load_overlap', 'load_body_overlap', 'fence_overlap'):
        for flag in (False, True):
            add_group(name, flag, [r for r in records if r[name] == flag])
    for length in ('2-32', '33-128', '129-256'):
        for flag in (False, True):
            add_group('output_x_load', f'{length}/{flag}',
                [r for r in records if r['output_group'] == length and r['load_overlap'] == flag])
    for startup in (False, True):
        for loading in (False, True):
            add_group('startup_x_load', f'{startup}/{loading}',
                [r for r in records if r['startup_overlap'] == startup and r['load_overlap'] == loading])
    loads = [r for r in records if r['native_load_invoked']]
    transactions = []
    for source in ('all', 'gpu', 'host', 'staging', 'file'):
        rs = records if source == 'all' else [r for r in records if r['native_source'] == source]
        ls = [r['native_load_ms'] for r in rs if r['native_load_invoked']]
        fs = [r['fence_ms'] for r in rs]
        transactions.append(dict(source=source, count=len(rs), native_loads=len(ls),
            total_load_ms=sum(ls), mean_load_ms=mean(ls) if ls else None,
            p95_load_ms=percentile(ls, .95) if ls else None, max_load_ms=max(ls) if ls else None,
            total_acquire_fence_ms=sum(fs), mean_acquire_fence_ms=mean(fs) if fs else None,
            p95_acquire_fence_ms=percentile(fs, .95) if fs else None))
    return dict(requests=records, groups=groups, transactions=transactions,
        request_count=len(records), native_load_count=len(loads), native_owner_count=len(owners),
        native_load_fields_complete=True, n_correct=None, confidence_interval=None,
        formal_qualification=False, caveats=[
            'Same-clock/same-owner request-associated load+acquire receipts; not every proactive operation.',
            'Load includes acquire fence; body excludes only that fence, not every possible synchronization.',
            'Fence covers acquire only, not release, inventory checks or all utility methods.',
            'Overlap is a duration-dependent exposure, not causal token stall time or removable latency.',
            'Conditional workloads/concurrency differ; single retained development run per version, no CI.',
            'GPU-hit load timestamps are null, not zero-duration measured loads.',
            'Selected-GPU acquire may precede controller admission (D184); no artificial lower bound.',
            'D170 TPOT timing threshold remains provisional; no formal SLO or numerical qualification.'])


def inspect_retained_native_load(output):
    """D305 narrows D304 with existing receipts; no new GPU or broad startup scan."""
    global OUT
    OUT = Path(output).resolve()
    if OUT.exists():
        raise FileExistsError(OUT)
    assert sha256(PLAN) == '0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    assert sha256(PROTOCOL) == '5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    prior = ROOT/'paper_results/ieee_tc/p2_backend/d304_decode/analysis1'
    seal = read(prior/'manifest.json')
    for name, digest in seal['outputs'].items():
        assert sha256(prior/name) == digest
    summaries, groups, transactions = {}, [], []
    paths = [PLAN, PROTOCOL, Path(__file__), ROOT/'scripts/ieee_tc/plot_d290_diagnostic.py',
        prior/'manifest.json', prior/'summary.json']
    OUT.mkdir(parents=True, exist_ok=False)
    for name in ('d300', 'd303'):
        previous = read(prior/'summary.json')['runs'][name]
        source = Path(previous['source']['source'])
        assert sha256(source) == previous['source']['source_sha256']
        _, evidence, requests = audit_run(source)
        projection = prior/f'{name}_request_intervals.csv'
        with projection.open(newline='') as handle:
            records = list(csv.DictReader(handle))
        for r in records:
            for k in ('prompt_tokens', 'tokens'):
                r[k] = int(r[k])
            for k in ('timing_tpot_miss', 'startup_overlap', 'preparation_overlap'):
                if r[k] not in ('True', 'False'):
                    raise ValueError('invalid retained boolean')
                r[k] = r[k] == 'True'
            for k in ('admitted', 'dispatched', 'first', 'last', 'first_offset_s', 'decode_s',
                      'tpot_ms', 'other_startup_overlap_s', 'same_replica_preparation_envelope_s',
                      'other_native_request_seconds', 'mean_other_native_requests'):
                r[k] = float(r[k])
        deployment_path = source.parent/'launch.launch/physical_deployment/deployment.json'
        result = native_load_overlap_summary(requests, records, clock_id=read(deployment_path)['clock_id'])
        assert result['request_count'] == 4000
        table(f'{name}_native_intervals.csv', result.pop('requests'))
        groups.extend(dict(run=name, **r) for r in result['groups'])
        transactions.extend(dict(run=name, **r) for r in result['transactions'])
        result['source'] = evidence
        summaries[name] = result
        paths.extend([source, projection, deployment_path, Path(evidence['physical_source'])])
        del requests
        gc.collect()
        print(json.dumps(dict(run=name, **result), indent=2), flush=True)
    write('summary.json', dict(kind='d305_retained_native_load_overlap_v1', runs=summaries,
        new_performance_runs=0, no_figures=True))
    table('groups.csv', groups)
    table('transactions.csv', transactions)
    write('manifest.json', dict(source_refs=[dict(path=str(p), sha256=sha256(p)) for p in paths],
        outputs={p.name:sha256(p) for p in sorted(OUT.iterdir())}, no_figures=True))


def inspect_retained_waits(output):
    """D301: reuse complete D299/D300; fresh explicit output, no GPU or plot."""
    global OUT
    OUT = Path(output).resolve()
    if OUT.exists():
        raise FileExistsError(OUT)
    assert sha256(PLAN) == '0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    assert sha256(PROTOCOL) == '5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    identities = {
        'd299': 'df066b365fa13ecc25b225a793c362706157224965b8a7b360981d8cf45887ab',
        'd300': '0964d7a58feeebadf607e72f6f6ea7a09b949d6df44da684bb86d0c27e8be974',
    }
    OUT.mkdir(parents=True, exist_ok=False)
    summaries, metrics, paths = {}, [], [PLAN, PROTOCOL, Path(__file__),
        ROOT/'scripts/analyze_control_path_overhead.py',
        ROOT/'scripts/ieee_tc/plot_d290_diagnostic.py']
    for campaign, expected_sha in identities.items():
        run = ROOT / f'results/ieee_tc/p2_backend_qualification/{campaign}_20261007/7b_full_w0'
        source = run / f'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_{campaign}_7b_full_w0.json'
        assert sha256(source) == expected_sha
        launch_path = run/'launch.json'
        launch = read(launch_path)
        assert launch['pass'] and launch['native_gpu_context_release_confirmed']
        assert launch['service_path_removed']
        assert launch['service_returncode'] == launch['replay_returncode'] == launch['watchdog_returncode'] == 0
        stats, evidence, requests = audit_run(source)
        assert len(requests) == 4000
        summary = summarize_admission_waits(requests)
        summary.update(source=evidence, stats=stats)
        summaries[campaign] = summary
        metrics.extend(dict(run=campaign, **row) for row in summary['metrics'])
        paths.extend([source, launch_path])
        del requests
        gc.collect()
    write('summary.json', dict(kind='d301_offline_wait_diagnosis_v1', runs=summaries,
        new_performance_runs=0, numerical_qualification=False, no_figures=True))
    table('wait_intervals.csv', metrics)
    write('manifest.json', dict(source_refs=[dict(path=str(p), sha256=sha256(p)) for p in paths],
        outputs={p.name:sha256(p) for p in sorted(OUT.iterdir())}, no_figures=True))
    print(json.dumps(dict(output=str(OUT), complete_sources=list(summaries),
        new_performance_runs=0), indent=2), flush=True)


def prefill_overlap_summary(requests, retained, *, clock_id):
    """First-schedule to first-token envelope; never call this kernel time."""
    prior = {r['request_id']: r for r in retained}
    if (not requests or len(prior) != len(retained) or len(requests) != len(retained)
            or len({r['request_id'] for r in requests}) != len(requests)
            or {r['request_id'] for r in requests} != set(prior)):
        raise ValueError('complete unique retained and native request identities required')
    records, owners = [], {}
    for q in requests:
        r, n = prior[q['request_id']], q['native_token_timing']
        e = q['gpu_reference_evidence']['receipt']
        dispatch, queued, scheduled, first, last = [n[k] for k in (
            'native_dispatch_monotonic_s', 'native_queued_monotonic_s',
            'native_scheduled_monotonic_s', 'native_first_token_monotonic_s',
            'native_last_token_monotonic_s')]
        if (not all(type(x) in (int, float) and math.isfinite(x)
                    for x in (dispatch, queued, scheduled, first, last))
                or not dispatch <= queued <= scheduled <= first < last
                or q['success'] is not True or type(q['completion_tokens']) is not int
                or not 2 <= q['completion_tokens'] <= 256
                or e['clock_id'] != clock_id or n['native_clock_id'] != clock_id
                or e['owner_id'] != r['owner'] or n['gpu_reference_owner_id'] != r['owner']
                or e['worker_pid'] != r['worker_pid'] or q['instance_id'] != r['replica']
                or first != r['first'] or last != r['last'] or dispatch != r['dispatched']
                or q['completion_tokens'] != r['tokens']
                or abs((last-first)*1000/(q['completion_tokens']-1)-r['tpot_ms']) > 1e-6
                or any(type(r[k]) is not bool for k in ('load_overlap', 'startup_overlap', 'timing_tpot_miss'))):
            raise ValueError('native prefill interval/owner differs from sealed projection')
        row = {k:r[k] for k in ('request_id', 'owner', 'replica', 'worker_pid', 'tokens', 'output_group',
            'prompt_tokens', 'first', 'last', 'tpot_ms', 'timing_tpot_miss', 'load_overlap',
            'startup_overlap', 'mean_other_native_requests')}
        row.update(scheduled=scheduled, prefill_envelope_ms=(first-scheduled)*1000,
            decode_s=last-first)
        records.append(row)
        owners.setdefault(row['owner'], []).append(row)
    for r in records:
        peers = [q for q in owners[r['owner']] if q['request_id'] != r['request_id']
            and max(r['first'],q['scheduled']) < min(r['last'],q['first'])]
        end, total = -math.inf, 0.
        for a, b in sorted((max(r['first'],q['scheduled']),min(r['last'],q['first'])) for q in peers):
            total += max(0., b-max(a,end))
            end = max(end,b)
        r.update(peer_prefill_count=len(peers), prefill_overlap=bool(peers),
            prefill_overlap_s=total, prefill_decode_fraction=total/r['decode_s'])
    groups=[]

    def add(name, label, rs):
        groups.append(dict(group=name,label=label,count=len(rs),
            timing_tpot_misses=sum(r['timing_tpot_miss'] for r in rs),
            mean_tpot_ms=mean(r['tpot_ms'] for r in rs) if rs else None,
            p95_tpot_ms=percentile([r['tpot_ms'] for r in rs],.95) if rs else None,
            mean_prefill_decode_fraction=mean(r['prefill_decode_fraction'] for r in rs) if rs else None,
            mean_other_native_requests=mean(r['mean_other_native_requests'] for r in rs) if rs else None))

    add('all','all',records)
    for flag in (False,True):
        add('prefill_overlap',flag,[r for r in records if r['prefill_overlap']==flag])
        add('no_startup_no_load_x_prefill',flag,[r for r in records
            if not r['startup_overlap'] and not r['load_overlap'] and r['prefill_overlap']==flag])
        for length in ('2-32','33-128','129-256'):
            add('output_x_prefill',f'{length}/{flag}',[r for r in records
                if r['output_group']==length and r['prefill_overlap']==flag])
    for load in (False,True):
        for prefill in (False,True):
            add('load_x_prefill',f'{load}/{prefill}',[r for r in records
                if r['load_overlap']==load and r['prefill_overlap']==prefill])
    return dict(requests=records,groups=groups,n_correct=None,formal_qualification=False,
        confidence_interval=None, request_count=len(records),
        mean_prefill_envelope_ms=mean(r['prefill_envelope_ms'] for r in records),
        p95_prefill_envelope_ms=percentile([r['prefill_envelope_ms'] for r in records],.95),
        caveats=['First-schedule to first-token is a service envelope, not kernel prefill time.',
            'This is not a joint-batch or token-iteration witness; utility work can coexist in the envelope.',
            'Same-clock same-owner exposure only; slower requests have more opportunity to overlap.',
            'Concurrency and conditional populations differ; no causal subtraction or independent replicates.',
            'D170 timing threshold remains provisional; no formal SLO or numerical qualification.'])


def inspect_retained_prefill(output):
    global OUT
    OUT=Path(output).resolve()
    if OUT.exists():
        raise FileExistsError(OUT)
    assert sha256(PLAN)=='0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    assert sha256(PROTOCOL)=='5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    prior=ROOT/'paper_results/ieee_tc/p2_backend/d305_native_overlap/analysis3'
    seal=read(prior/'manifest.json')
    for name,digest in seal['outputs'].items():
        assert sha256(prior/name)==digest
    summaries,groups={},[]
    paths=[PLAN,PROTOCOL,Path(__file__),ROOT/'scripts/ieee_tc/plot_d290_diagnostic.py',
        prior/'manifest.json',prior/'summary.json']
    OUT.mkdir(parents=True,exist_ok=False)
    for name in ('d300','d303'):
        source_ref=read(prior/'summary.json')['runs'][name]['source']
        source=Path(source_ref['source'])
        assert sha256(source)==source_ref['source_sha256']
        _,evidence,requests=audit_run(source)
        projection=prior/f'{name}_native_intervals.csv'
        with projection.open(newline='') as handle:
            records=list(csv.DictReader(handle))
        for r in records:
            for k in ('prompt_tokens','tokens','worker_pid'):
                r[k]=int(r[k])
            for k in ('load_overlap','startup_overlap','timing_tpot_miss'):
                if r[k] not in ('True','False'):
                    raise ValueError('invalid retained boolean')
                r[k]=r[k]=='True'
            for k in ('dispatched','first','last','tpot_ms','mean_other_native_requests'):
                r[k]=float(r[k])
        deployment=source.parent/'launch.launch/physical_deployment/deployment.json'
        result=prefill_overlap_summary(requests,records,clock_id=read(deployment)['clock_id'])
        assert result['request_count']==4000
        table(f'{name}_prefill_intervals.csv',result.pop('requests'))
        groups.extend(dict(run=name,**g) for g in result['groups'])
        result['source']=evidence
        summaries[name]=result
        paths.extend([source,projection,deployment,Path(evidence['physical_source'])])
        del requests
        gc.collect()
        print(json.dumps(dict(run=name,**result),indent=2),flush=True)
    write('summary.json',dict(kind='d306_retained_prefill_overlap_v1',runs=summaries,
        new_performance_runs=0,no_figures=True))
    table('groups.csv',groups)
    write('manifest.json',dict(source_refs=[dict(path=str(p),sha256=sha256(p)) for p in paths],
        outputs={p.name:sha256(p) for p in sorted(OUT.iterdir())},no_figures=True))


def main(campaign='d297'):
    global RAW, RUN, OUT, SOURCE
    source_sha = '7f171e1d58b061e7dd7392e1651d087c300ce1d6c2223c3ed16ef2fdce3aa07b'
    if campaign == 'd299':
        RAW = ROOT / 'results/ieee_tc/p2_backend_qualification/d299_20261007'
        RUN = RAW / '7b_full_w0'
        OUT = ROOT / 'paper_results/ieee_tc/p2_backend/d299_full_w0/validated1'
        SOURCE = RUN / 'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d299_7b_full_w0.json'
        source_sha = 'df066b365fa13ecc25b225a793c362706157224965b8a7b360981d8cf45887ab'
    elif campaign == 'd300':
        RAW = ROOT / 'results/ieee_tc/p2_backend_qualification/d300_20261007'
        RUN = RAW / '7b_full_w0'
        OUT = ROOT / 'paper_results/ieee_tc/p2_backend/d300_full_w0/validated1'
        SOURCE = RUN / 'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d300_7b_full_w0.json'
        source_sha = '0964d7a58feeebadf607e72f6f6ea7a09b949d6df44da684bb86d0c27e8be974'
    elif campaign == 'd303':
        RAW = ROOT / 'results/ieee_tc/p2_backend_qualification/d303_20261007'
        RUN = RAW / '7b_full_w0'
        OUT = ROOT / 'paper_results/ieee_tc/p2_backend/d303_full_w0/validated1'
        SOURCE = RUN / 'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d303_7b_full_w0.json'
        source_sha = '9bd86f5647a2723fa0aed67a98b92a72b477f725ef91b98a0182f348eaa7598e'
    elif campaign == 'd314':
        RAW = ROOT / 'results/ieee_tc/p2_backend_qualification/d314_20261007'
        RUN = RAW / '7b_full_w0'
        OUT = ROOT / 'paper_results/ieee_tc/p2_backend/d314_full_w0/validated1'
        SOURCE = RUN / 'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d314_7b_full_w0.json'
        source_sha = '5cbc1b213ab54d282ee2b41a3ff22c0fde09d8615d900005c2676e2bd9ff4c9e'
    elif campaign == 'd315':
        RAW = ROOT / 'results/ieee_tc/p2_backend_qualification/d315_20261007'
        RUN = RAW / '7b_full_w0'
        OUT = ROOT / 'paper_results/ieee_tc/p2_backend/d315_full_w0/validated1'
        SOURCE = RUN / 'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d315_7b_full_w0.json'
        source_sha = '6e5007630c9780c417e65109ef3005daedba3dafa87a843ff75c4885f91c362f'
    elif campaign != 'd297':
        raise ValueError('unknown frozen full-run identity')
    assert sha256(PLAN) == '0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    assert sha256(PROTOCOL) == '5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    assert sha256(WARM) == '0ebc7587effe293dcb986d33fb932c100cbcb3384c56ab2c0b0bec73b2bdd88c'
    assert sha256(SOURCE) == source_sha
    launch = read(RUN / 'launch.json')
    assert launch['pass'] and launch['native_gpu_context_release_confirmed']
    assert launch['service_path_removed']
    assert launch['service_returncode'] == launch['watchdog_returncode'] == launch['replay_returncode'] == 0
    OUT.mkdir(parents=True, exist_ok=False)
    stats, evidence, requests = audit_run(SOURCE)
    assert len(requests) == 4000
    deployment = read(RUN / 'launch.launch/physical_deployment/deployment.json')
    physical = read(RUN / 'launch.launch/physical_deployment/summary.json')
    assert abs(sum(physical['window_gpu_seconds'].values()) - stats['physical_gpu_seconds']) < 1e-6
    index = read(INDEX)
    expected = {a['id']: hashlib.sha256(json.dumps(sorted(a['files'], key=lambda f:f['path']),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest() for a in index['artifacts']}
    assert len(expected) == len(index['artifacts']) == 500
    warm = read(WARM)
    thresholds = [dict(group_id=t['group_id'], lower_exclusive=t['lower_exclusive'],
        upper_inclusive=t['upper_inclusive'], ttft_s=t['candidate_ttft_threshold_ms']/1000,
        tpot_s=t['candidate_tpot_threshold_ms']/1000) for t in warm['warm_threshold_candidates']]
    stages = ('arrival_to_admission_s', 'admission_to_engine_dispatch_s',
              'engine_entry_to_queue_s', 'native_engine_queue_s', 'native_prefill_s')
    normalized, failures, errors = [], [], []
    integral_float_prompt_counts = 0
    binding_counts = Counter()
    for row in requests:
        checks = binding_checks(row, expected[row['adapter_id']])
        binding_counts.update(k for k,v in checks.items() if v)
        if not all(checks.values()):
            failures.append(dict(request_id=row['request_id'], checks=checks))
        t = row['native_token_timing']
        spans = NativeV1TokenTimeline.service_breakdown(t,
            admitted_at=t['controller_admitted_monotonic_s'],
            completed_at=t['controller_completed_monotonic_s'], clock_id=deployment['clock_id'])
        arrival = row['external_arrival_timing']['planned_arrival_s']
        assert abs(arrival - deployment['arrival_start_s'] - row['scheduled_arrival_offset_s']) < 1e-6
        sv = dict(arrival_to_admission_s=t['controller_admitted_monotonic_s']-arrival)
        sv.update({k:spans[k[:-2]+'_ms']/1000 for k in stages[1:]})
        assert all(math.isfinite(v) and v >= 0 for v in sv.values())
        error = abs(sum(sv.values()) - row['overall_ttft_ms']/1000)
        assert error <= .001
        errors.append(error)
        prompt_count = t['actual_prompt_tokens']
        assert type(prompt_count) in (int, float) and math.isfinite(prompt_count)
        assert prompt_count >= 1 and int(prompt_count) == prompt_count
        integral_float_prompt_counts += type(prompt_count) is float
        normalized.append(dict(request_id=row['request_id'], adapter_id=row['adapter_id'], success=True,
            prompt_tokens=int(prompt_count), token_count=row['completion_tokens'],
            ttft_s=row['overall_ttft_ms']/1000, tpot_s=row['tpot_ms']/1000,
            e2e_s=row['overall_e2e_ms']/1000, tier=row['readiness_tier_before_dispatch'],
            canonical_prompt_sha256=row['canonical_prompt_sha256'],
            completion_token_ids_sha256=row['completion_token_ids_sha256'], **sv))
    assert not failures
    timing = provisional_timing_bound(normalized, thresholds, offered_count=4000)
    for row, decision in zip(normalized, timing.pop('decisions')):
        row.update(decision)
    stats.update(p99_ttft_ms=percentile([r['ttft_s']*1000 for r in normalized], .99),
        mean_e2e_s=mean(r['e2e_s'] for r in normalized),
        gpu_s_per_offered=stats['physical_gpu_seconds']/4000)
    groups = []
    for key, labels in [('all',['all']),('tier',sorted({r['tier'] for r in normalized})),
                        ('outcome',list(timing['outcome_counts']))]:
        for label in labels:
            rs = [r for r in normalized if key == 'all' or r[key] == label]
            if not rs:
                continue
            groups.append(dict(group=key, label=label, count=len(rs),
                mean_ttft_s=mean(r['ttft_s'] for r in rs), p95_ttft_s=percentile([r['ttft_s'] for r in rs], .95),
                mean_tpot_s=mean(r['tpot_s'] for r in rs),
                **{'mean_'+k:mean(r[k] for r in rs) for k in stages}))
    table('requests.csv', normalized)
    table('timing_groups.csv', groups)
    write('audit.json', dict(kind=f'{campaign}_full_development_audit_v1', stats=stats, evidence=evidence,
        timing=timing, thresholds=thresholds, window_gpu_seconds=physical['window_gpu_seconds'],
        control_binding_pass_counts=dict(binding_counts), max_stage_error_s=max(errors),
        integral_float_prompt_counts=integral_float_prompt_counts,
        n_correct=None, formal_qualification=False, confidence_interval=None,
        caveats=['One complete W0 development replay, not a held-out block.',
                 'Control binding consistency is not kernel arithmetic qualification.',
                 'Provisional warm thresholds are not globally frozen common SLO.',
                 'Conditional tier groups are not causal comparisons.']))
    print(json.dumps(dict(stats=stats, timing=timing), indent=2), flush=True)
    if campaign in ('d303', 'd314', 'd315'):
        write('admission_waits.json', summarize_admission_waits(requests))
    if campaign in ('d314', 'd315'):
        write('preparation_attribution.json', audit_preparation_attribution(
            requests, clock_id=deployment['clock_id']))
    del requests, normalized
    gc.collect()

    # jq's streaming parser avoids materializing the 629-MiB outcome tree.
    outcome_path = RUN / 'launch.launch/physical_deployment/main_outcome.json'
    query = 'fromstream(2|truncate_stream(inputs | select(.[0][:2] == ["mechanism_events","_ieee_control_events"])))'
    controls = json.loads(subprocess.check_output(['jq','-cn','--stream',query,str(outcome_path)]))
    import yaml
    config_path = ROOT / f'configs/ieee_tc/20261007_{campaign}_7b_full_w0.yaml'
    cfg = yaml.safe_load(config_path.read_text())['resource_coordination']['ieee_scaling']
    counters = Counter()
    longest = current_start = 0.0
    previous = -math.inf
    control_rows = []
    for e in controls:
        assert previous <= e['observed_at']
        previous = e['observed_at']
        q = e['queue_depth'] < cfg['queue_lower']
        a = e['active_saturation'] < cfg['active_lower']
        known = e['p95_ttft_ms'] is not None
        t = known and e['p95_ttft_ms'] < cfg['ttft_lower_ms']
        assert e['all_low'] == (q and a and t)
        flags = dict(queue_low=q, active_low=a, ttft_known=known, ttft_low=t,
            queue_active_low=q and a, ttft_only_blocks=q and a and known and not t,
            all_low=e['all_low'], pending=e['pending_instances']>0)
        counters.update(k for k,v in flags.items() if v)
        counters['action:'+e['action']] += 1
        counters['outcome:'+e['outcome']] += 1
        if e['all_low'] and not e['pending_instances']:
            current_start = current_start or e['observed_at']
            longest = max(longest, e['observed_at']-current_start)
        else:
            current_start = 0.0
        control_rows.append(dict(offset_s=e['observed_at']-deployment['arrival_start_s'],
            queue=e['queue_depth'], active=e['active_requests'], capacity=e['ready_capacity'],
            pending=e['pending_instances'], p95_ttft_ms=e['p95_ttft_ms'],
            action=e['action'], outcome=e['outcome'], **{k:v for k,v in flags.items() if k != 'pending'}))
    assert controls
    table('control_samples.csv', control_rows)
    control_audit = dict(samples=len(controls), counts=dict(counters),
        longest_sampled_all_low_span_s=longest, cooldown_s=cfg['scale_down_cooldown_s'],
        min_known_p95_ttft_ms=min(e['p95_ttft_ms'] for e in controls if e['p95_ttft_ms'] is not None),
        config=cfg, counterfactual_gpu_s=None, causal_performance_gain=None,
        caveat='Sampled predicates only; changing one threshold may change the entire closed loop.')
    write('control_audit.json', control_audit)
    print(json.dumps(control_audit, indent=2), flush=True)
    paths = [SOURCE, WARM, INDEX, PLAN, PROTOCOL, config_path, outcome_path, Path(__file__),
        ROOT/'scripts/ieee_tc/plot_d290_diagnostic.py', ROOT/'scripts/analyze_control_path_overhead.py']
    write('manifest.json', dict(source_refs=[dict(path=str(p),sha256=sha256(p)) for p in paths],
        outputs={p.name:sha256(p) for p in sorted(OUT.iterdir())}, no_figures=True))


def inspect_retained_route_copies(output):
    """D316: inspect the sealed complete D315 target views, without inference."""
    global OUT
    OUT = Path(output).resolve()
    if OUT.exists():
        raise FileExistsError(OUT)
    assert sha256(PLAN) == '0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    assert sha256(PROTOCOL) == '5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    source = ROOT / ('results/ieee_tc/p2_backend_qualification/d315_20261007/7b_full_w0/'
        'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d315_7b_full_w0.json')
    assert sha256(source) == '6e5007630c9780c417e65109ef3005daedba3dafa87a843ff75c4885f91c362f'
    _, evidence, requests = audit_run(source)
    assert len(requests) == 4000
    result = summarize_route_copy_revisits(requests)
    opportunities = result.pop('opportunities')
    result.update(source=evidence, new_performance_runs=0, no_figures=True)
    OUT.mkdir(parents=True, exist_ok=False)
    write('summary.json', result)
    write('opportunities.json', opportunities)
    paths = [PLAN, PROTOCOL, source, Path(__file__),
        ROOT/'scripts/analyze_control_path_overhead.py',
        ROOT/'scripts/ieee_tc/plot_d290_diagnostic.py']
    write('manifest.json', dict(source_refs=[dict(path=str(p), sha256=sha256(p)) for p in paths],
        outputs={p.name:sha256(p) for p in sorted(OUT.iterdir())}, no_figures=True))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', choices=('d297', 'd299', 'd300', 'd303', 'd314', 'd315'), default='d297')
    parser.add_argument('--admission-wait-output', type=Path,
        help='Data-only D299/D300 retained wait partition; explicit new output directory')
    parser.add_argument('--decode-tail-output', type=Path,
        help='D304 data-only D300/D303 decode interval overlap; explicit new output')
    parser.add_argument('--native-load-output', type=Path,
        help='D305 narrow native load/acquire overlap using D304 and retained receipts')
    parser.add_argument('--prefill-output', type=Path,
        help='D306 retained same-owner prefill envelope using D305 and native timing')
    parser.add_argument('--route-copy-output', type=Path,
        help='D316 retained D315 target-copy revisits; explicit fresh output, no inference')
    args = parser.parse_args()
    if sum(x is not None for x in (args.admission_wait_output, args.decode_tail_output,
                                  args.native_load_output, args.prefill_output, args.route_copy_output)) > 1:
        parser.error('choose one retained diagnostic')
    if args.route_copy_output is not None:
        inspect_retained_route_copies(args.route_copy_output)
    elif args.prefill_output is not None:
        inspect_retained_prefill(args.prefill_output)
    elif args.native_load_output is not None:
        inspect_retained_native_load(args.native_load_output)
    elif args.decode_tail_output is not None:
        inspect_retained_decode(args.decode_tail_output)
    elif args.admission_wait_output is not None:
        inspect_retained_waits(args.admission_wait_output)
    else:
        main(args.campaign)
