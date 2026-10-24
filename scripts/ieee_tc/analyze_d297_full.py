#!/usr/bin/env python3
"""D297–D358 registered full native and timing audits.

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
from bisect import bisect_left
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ieee_tc.plot_d290_diagnostic import audit_run, binding_checks, percentile, sha256, load
from scripts.analyze_control_path_overhead import (provisional_timing_bound,
    summarize_admission_waits, summarize_route_copy_revisits, summarize_dispatch_gate_intervals,
    summarize_dispatch_capacity_context)
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


def proactive_load_overlap_summary(records, plans, host_preparations, activations, *, clock_id):
    """Intersect observed proactive operations on the same native owner.

    A deferred receipt without a native interval is unmeasured, not a zero-time
    operation. These unions are exposure, not token stalls or causal savings.
    """
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    owners, replicas = {}, {}
    for a in activations:
        owner = a.get('native_owner_id')
        if owner is not None:
            if owner in replicas or a['clock_id'] != clock_id:
                raise ValueError('duplicate or mixed-clock activation owner')
            replicas[owner] = a['target_replica']
    for r in records:
        identity = (r['replica'], r['worker_pid'])
        if (r['owner'] not in replicas or replicas[r['owner']] != r['replica']
                or r['owner'] in owners and owners[r['owner']] != identity
                or not finite(r['first']) or not finite(r['last']) or r['first'] >= r['last']):
            raise ValueError('request owner/incarnation or interval mismatch')
        owners[r['owner']] = identity
    if not records or len({r['request_id'] for r in records}) != len(records):
        raise ValueError('nonempty unique request projection required')
    counts, receipts, intervals, leases, attempts = Counter(), [], [], set(), set()
    for kind, collection in (('gpu', plans), ('host', host_preparations)):
        counts[kind+'_plans'] = len(collection)
        for plan in collection:
            for attempt in plan['attempts']:
                aid = attempt['attempt_id']
                if not aid or aid in attempts:
                    raise ValueError('duplicate preparation attempt')
                attempts.add(aid)
                counts[kind+'_attempts'] += 1
                counts[kind+'_state_'+attempt['state']] += 1
                if 'receipt' not in attempt:
                    counts[kind+'_without_receipt'] += 1
                    continue
                e = attempt['receipt']
                owner, pid = e['owner_id'], e['worker_pid']
                identity = (plan['target_replica'], pid)
                if (e['clock_id'] != clock_id or e['worker_rank'] != 0
                        or type(pid) is not int or pid <= 0
                        or replicas.get(owner) != plan['target_replica']
                        or kind == 'gpu' and plan['owner_id'] != owner
                        or owner in owners and owners[owner] != identity):
                    raise ValueError('proactive owner/clock/worker differs from activation')
                owners[owner] = identity
                accepted = e['acquired' if kind == 'gpu' else 'held']
                if type(accepted) is not bool:
                    raise ValueError('explicit proactive outcome required')
                counts[kind+'_receipts'] += 1
                if not accepted:
                    counts[kind+'_deferred_interval_unmeasured'] += 1
                    continue
                lease = e['lease_id']
                if not lease or lease in leases or type(e['native_load_invoked']) is not bool:
                    raise ValueError('invalid/duplicate native preparation lease')
                leases.add(lease)
                counts[kind+'_accepted'] += 1
                invoked = e['native_load_invoked']
                counts[kind+'_loads' if invoked else kind+'_reused'] += 1
                projection = dict(kind=kind, owner=owner, worker_pid=pid,
                    replica=plan['target_replica'], plan_id=plan['plan_id'],
                    trigger_reason=plan['trigger_reason'], attempt_id=aid,
                    lease_id=lease, native_load_invoked=invoked)
                spans = []
                if kind == 'gpu':
                    end, fence = e['acquired_monotonic_s'], e['completion_fence_ms']
                    if not finite(end) or not finite(fence) or fence < 0:
                        raise ValueError('invalid preparation fence')
                    spans.append(('gpu_fence', end-fence/1000, end))
                    if invoked:
                        start, loaded = e['native_load_started_monotonic_s'], e['native_load_completed_monotonic_s']
                        if not finite(start) or loaded != end or start > end-fence/1000:
                            raise ValueError('GPU load does not contain acquisition fence')
                        spans.append(('gpu_load', start, loaded))
                    elif any(e.get(k) is not None for k in ('native_load_started_monotonic_s',
                                                           'native_load_completed_monotonic_s')):
                        raise ValueError('reused GPU source must not claim a load interval')
                else:
                    start, end = e['load_started_monotonic_s'], e['load_completed_monotonic_s']
                    spans.append(('host_transaction', start, end))
                    if invoked:
                        spans.append(('host_load', start, end))
                for category, start, end in spans:
                    if not finite(start) or not finite(end) or not 0 <= start <= end:
                        raise ValueError('invalid proactive native interval')
                    intervals.append(dict(category=category,start=start,end=end,**projection))
                receipts.append(projection)
    output, groups = [], []
    categories = ('gpu_load', 'gpu_fence', 'host_load', 'host_transaction', 'any_load')
    for r in records:
        row = dict(r)
        for category in categories:
            matching = [e for e in intervals if e['owner'] == r['owner']
                and (e['category'] == category or category == 'any_load'
                     and e['category'] in ('gpu_load', 'host_load'))
                and max(r['first'], e['start']) < min(r['last'], e['end'])]
            end, total = -math.inf, 0.
            for a, b in sorted((max(r['first'], e['start']), min(r['last'], e['end'])) for e in matching):
                total += max(0., b-max(a,end)); end=max(end,b)
            row['proactive_'+category+'_overlap_s'] = total
            row['proactive_'+category+'_overlap'] = bool(matching)
            row['proactive_'+category+'_decode_fraction'] = total/(r['last']-r['first'])
        output.append(row)
    for category in categories:
        for flag in (False, True):
            rs = [r for r in output if r['proactive_'+category+'_overlap'] == flag]
            groups.append(dict(group=category,label=flag,count=len(rs),
                timing_tpot_misses=sum(r['timing_tpot_miss'] for r in rs),
                mean_tpot_ms=mean(r['tpot_ms'] for r in rs) if rs else None,
                mean_other_native_requests=mean(r['mean_other_native_requests'] for r in rs) if rs else None,
                mean_decode_fraction=mean(r['proactive_'+category+'_decode_fraction'] for r in rs) if rs else None))
    return dict(requests=output,groups=groups,counts=dict(counts),intervals=intervals,
        native_owners={o:dict(replica=v[0],worker_pid=v[1]) for o,v in owners.items()},
        observed_interval_ms={c:sum((e['end']-e['start'])*1000 for e in intervals if e['category']==c)
                              for c in categories if c != 'any_load'},
        causal_attribution_established=False,
        caveats=['Same native owner/incarnation and monotonic clock; not a kernel stall measurement.',
            'GPU load includes the acquisition fence; HOST transaction includes reuse. Do not add nested spans.',
            'Rejected/missing-receipt operations have no interval here; not assumed to have zero overhead.',
            'Source snapshots, release and other utility calls are not fully covered.',
            'Exposure depends on decode duration; conditional groups and native concurrency are not controlled.'])


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


def lifecycle_gaps(physical, deployment):
    """Observed between-lease gaps, not hypothetical savings from no retirement."""
    if (physical['clock_id'] != deployment['clock_id']
            or physical['measurement_complete'] is not True or physical['open_lease_ids']):
        raise ValueError('complete same-clock physical measurement required')
    origin, total, gaps = deployment['arrival_start_s'], 0., []
    for gpu, spans in physical['device_intervals'].items():
        previous = None
        for start, end in spans:
            if (not all(type(x) in (int, float) and math.isfinite(x) for x in (start, end))
                    or end <= start or (previous is not None and start < previous)):
                raise ValueError('ordered disjoint physical intervals required')
            total += end-start
            if previous is not None and start > previous:
                gaps.append(dict(gpu=gpu, released_offset_s=previous-origin,
                    reallocated_offset_s=start-origin, duration_s=start-previous))
            previous = end
    if not physical['device_intervals'] or abs(total-physical['gpu_seconds']) > 1e-6:
        raise ValueError('physical interval integral mismatch')
    return dict(gaps=gaps, released_gap_gpu_s=sum(g['duration_s'] for g in gaps),
        observed_gpu_s=total, causal_savings=None)


LIFECYCLE_CONTRASTS = {
    ('d315', 'd318'): (
        'd319_retained_lifecycle_decode_v1',
        ('6e5007630c9780c417e65109ef3005daedba3dafa87a843ff75c4885f91c362f',
         '4c781c7b10890293130b425e7cc94dc80c96f3b3dd8037f89db53e1fb08ffb1b')),
    ('d323', 'd326'): (
        'd327_retained_drain_lifecycle_v1',
        ('a790cd59968a0d8d33bc017938467c1f818c5e5c617fb9628ab25a4843b5e991',
         '9e1057f50d1cc8dafacfd4f232216cd5979ac6c7e9f2a479dcd26f654295150a')),
    ('d335', 'd336'): (
        'd337_retained_capacity_lifecycle_v1',
        ('a3b6ee044db0ba46915e94ad1de8e67f638b3b0f188fed7eaac1e69078ffaa97',
         'b8452f4f577446128b4a77a7117485660f3b9841caf831ad867a58d9a9587a05')),
    ('d336', 'd346'): (
        'd347_retained_native_capacity_lifecycle_v1',
        ('b8452f4f577446128b4a77a7117485660f3b9841caf831ad867a58d9a9587a05',
         '52e7d3fddde1f4bd8ae05831dd4ce0d69e291f9c2b9aba3a7ef5b659a987151e')),
    ('d336', 'd349'): (
        'd350_retained_decoupled_control_lifecycle_v1',
        ('b8452f4f577446128b4a77a7117485660f3b9841caf831ad867a58d9a9587a05',
         'c289b97bd146ae8ec355f59ea5abd373b271e9848814ea3b60ddff58cfdd97e7')),
}


def partition_activation_readiness(activations, requests, *, clock_id):
    """Preserve non-ready epochs without inventing a ready/cancel timestamp.

    Only measured trigger-to-ready intervals enter closed startup overlaps.
    All requests must belong to ready epochs. Resource accounting is separate
    and still includes every physical lease, including unused/cancelled startup.
    """
    ready, unready, seen = [], [], set()
    for a in activations:
        identity, begin = a['activation_id'], a['trigger_at']
        if (not identity or identity in seen or a['target_replica'] != identity
                or a['clock_id'] != clock_id
                or a['category'] not in ('initial', 'natural_scaleout', 'controlled')
                or type(begin) not in (int, float) or not math.isfinite(begin)):
            raise ValueError('unique same-clock activation identity required')
        seen.add(identity)
        if 'engine_ready_at' in a:
            end = a['engine_ready_at']
            if type(end) not in (int, float) or not math.isfinite(end) or end < begin:
                raise ValueError('finite ordered actual engine-ready timestamp required')
            ready.append(a)
        elif a.get('state') in ('cancelled', 'failed', 'startup_ownership_unresolved'):
            unready.append(a)
        else:
            raise ValueError('missing readiness without a recorded non-ready terminal state')
    if not requests or not {r['instance_id'] for r in requests} <= {a['activation_id'] for a in ready}:
        raise ValueError('all served requests must bind to an actually ready epoch')
    return ready, unready


def capacity_phase_groups(records):
    """All eight original 500-request phases; native overlap is not kernel batch."""
    if len(records) != 4000 or len({r['request_id'] for r in records}) != 4000:
        raise ValueError('complete unique 4000-request population required')
    ordered = sorted(records, key=lambda r: (r['arrival_offset_s'], r['request_id']))
    for r in ordered:
        for k in ('arrival_offset_s', 'ttft_ms', 'tpot_ms', 'mean_other_native_requests'):
            if type(r[k]) not in (int, float) or not math.isfinite(r[k]) or r[k] < 0:
                raise ValueError('finite nonnegative phase measurement required')
        for k in ('ttft_pass', 'timing_tpot_miss', 'timing_joint_pass', 'startup_overlap'):
            if type(r[k]) is not bool:
                raise ValueError('explicit timing/overlap booleans required')
    groups = []
    for phase in range(8):
        rows = ordered[phase*500:(phase+1)*500]
        groups.append(dict(phase=phase, count=len(rows),
            first_arrival_s=rows[0]['arrival_offset_s'], last_arrival_s=rows[-1]['arrival_offset_s'],
            timing_ttft_misses=sum(not r['ttft_pass'] for r in rows),
            timing_tpot_misses=sum(r['timing_tpot_miss'] for r in rows),
            timing_joint_misses=sum(not r['timing_joint_pass'] for r in rows),
            startup_overlap_requests=sum(r['startup_overlap'] for r in rows),
            mean_ttft_ms=mean(r['ttft_ms'] for r in rows),
            p95_ttft_ms=percentile([r['ttft_ms'] for r in rows], .95),
            mean_tpot_ms=mean(r['tpot_ms'] for r in rows),
            p95_tpot_ms=percentile([r['tpot_ms'] for r in rows], .95),
            mean_other_native_requests=mean(r['mean_other_native_requests'] for r in rows)))
    return groups


def paired_timeline_groups(runs, *, contrast=('d315', 'd318')):
    """Registered common arrival-time cohorts, descriptive and post hoc, no CI."""
    if contrast not in LIFECYCLE_CONTRASTS or set(runs) != set(contrast):
        raise ValueError('registered complete pair required')
    reference, candidate = contrast
    maps = {name: {r['request_id']: r for r in run['requests']} for name, run in runs.items()}
    if (not maps[reference] or set(maps[reference]) != set(maps[candidate])
            or any(len(maps[k]) != len(runs[k]['requests']) for k in runs)):
        raise ValueError('complete unique paired request population required')
    gap_spans = [(g['released_offset_s'], g['reallocated_offset_s'])
        for run in runs.values() for g in run['lifecycle']['gaps']]
    startup_spans = [(s['start_offset_s'], s['end_offset_s'])
        for run in runs.values() for s in run['activation_offsets']]
    if not startup_spans or any(not math.isfinite(x) for s in startup_spans+gap_spans for x in s):
        raise ValueError('finite observed event spans required')
    last_ready = max(b for a, b in startup_spans)
    pairs, grouped = [], []
    for rid, old in maps[reference].items():
        new = maps[candidate][rid]
        for key in ('adapter_id', 'prompt_tokens', 'tokens', 'canonical_prompt_sha256', 'output_group'):
            if old[key] != new[key]:
                raise ValueError('paired input mismatch')
        if (not math.isfinite(old['arrival_offset_s']) or not math.isfinite(new['arrival_offset_s'])
                or abs(old['arrival_offset_s']-new['arrival_offset_s']) > 1e-6):
            raise ValueError('paired planned arrival mismatch')
        offset = old['arrival_offset_s']
        pairs.append(dict(request_id=rid, arrival_offset_s=offset, output_group=old['output_group'],
            in_either_release_gap=any(a <= offset < b for a,b in gap_spans),
            in_either_startup=any(a <= offset < b for a,b in startup_spans),
            after_both_last_ready=offset >= last_ready,
            output_hash_changed=old['completion_token_ids_sha256'] != new['completion_token_ids_sha256']))
    for dimension, labels in (('all', ['all']), ('in_either_release_gap', [False, True]),
            ('in_either_startup', [False, True]), ('after_both_last_ready', [False, True]),
            ('output_group', ['2-32', '33-128', '129-256'])):
        for label in labels:
            cohort = [r['request_id'] for r in pairs if dimension == 'all' or r[dimension] == label]
            for name in runs:
                rows = [maps[name][rid] for rid in cohort]
                group = dict(run=name, group=dimension, label=label, count=len(rows),
                    timing_ttft_misses=sum(not r['ttft_pass'] for r in rows),
                    timing_tpot_misses=sum(r['timing_tpot_miss'] for r in rows),
                    timing_joint_misses=sum(not r['timing_joint_pass'] for r in rows))
                for field in ('ttft_ms', 'admission_ms', 'service_ms', 'tpot_ms', 'e2e_s'):
                    values = [r[field] for r in rows]
                    group['mean_'+field] = mean(values) if values else None
                    group['p95_'+field] = percentile(values, .95) if values else None
                grouped.append(group)
    return dict(pairs=pairs, groups=grouped, common_last_ready_offset_s=last_ready,
        output_hash_differences=sum(r['output_hash_changed'] for r in pairs),
        caveat='Same request cohorts from union of observed relative event times; post hoc, not causal or independent repeats.')


def inspect_retained_lifecycle(output, *, contrast=('d315', 'd318')):
    """Reuse D304-D306 interval methods on an explicitly registered sealed pair."""
    global OUT
    if contrast not in LIFECYCLE_CONTRASTS:
        raise ValueError('registered complete pair required')
    kind, digests = LIFECYCLE_CONTRASTS[contrast]
    OUT = Path(output).resolve()
    if OUT.exists():
        raise FileExistsError(OUT)
    assert sha256(PLAN) == '0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    assert sha256(PROTOCOL) == '5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    paths = [PLAN, PROTOCOL, Path(__file__), ROOT/'scripts/ieee_tc/plot_d290_diagnostic.py']
    summaries, retained, groups, transactions = {}, {}, [], []
    OUT.mkdir(parents=True, exist_ok=False)
    identities = dict(zip(contrast, digests))
    for name, digest in identities.items():
        if contrast in (('d336', 'd346'), ('d336', 'd349')) and name == 'd336':
            # Reuse the already sealed full reference projection; do not rescan
            # its large raw outcome merely to reproduce identical intervals.
            cache = ROOT/'paper_results/ieee_tc/p2_backend/d337_capacity_lifecycle/analysis1'
            seal = read(cache/'manifest.json')
            for filename in ('summary.json', 'd336_intervals.csv', 'transactions.csv'):
                assert sha256(cache/filename) == seal['outputs'][filename]
                paths.append(cache/filename)
            cached = read(cache/'summary.json')['runs']['d336']
            assert cached['source']['source_sha256'] == digest
            assert sha256(Path(cached['source']['source'])) == digest
            with (cache/'d336_intervals.csv').open(newline='') as f:
                records = list(csv.DictReader(f))
            for r in records:
                for k in ('prompt_tokens', 'tokens'):
                    r[k] = int(r[k])
                for k in ('arrival_offset_s', 'ttft_ms', 'admission_ms', 'service_ms',
                          'tpot_ms', 'e2e_s', 'mean_other_native_requests'):
                    r[k] = float(r[k])
                for k in ('ttft_pass', 'timing_tpot_miss', 'timing_joint_pass', 'startup_overlap'):
                    if r[k] not in ('True', 'False'):
                        raise ValueError('invalid cached timing decision')
                    r[k] = r[k] == 'True'
            assert len(records) == 4000
            summaries[name] = cached
            retained[name] = dict(requests=records, lifecycle=cached['lifecycle'],
                activation_offsets=cached['activation_offsets'])
            for family in ('decode', 'native', 'prefill'):
                groups.extend(dict(run=name, family='load' if family=='native' else family, **g)
                              for g in cached[family]['groups'])
            with (cache/'transactions.csv').open(newline='') as f:
                transactions.extend(r for r in csv.DictReader(f) if r['run'] == name)
            paths.extend([cache/'manifest.json', Path(cached['source']['source'])])
            continue
        prior = ROOT/f'paper_results/ieee_tc/p2_backend/{name}_full_w0/validated1'
        seal = read(prior/'manifest.json')
        for filename, expected in seal['outputs'].items():
            assert sha256(prior/filename) == expected
        audit = read(prior/'audit.json')
        source = Path(audit['evidence']['source'])
        assert sha256(source) == digest == audit['evidence']['source_sha256']
        outcome = source.parent/'launch.launch/physical_deployment/main_outcome.json'
        refs = {r['path']: r['sha256'] for r in seal['source_refs']}
        assert sha256(outcome) == refs[str(outcome)]
        stats, evidence, requests = audit_run(source)
        assert len(requests) == 4000
        deployment_path = source.parent/'launch.launch/physical_deployment/deployment.json'
        physical_path = source.parent/'launch.launch/physical_deployment/summary.json'
        deployment, physical = read(deployment_path), read(physical_path)
        query = 'fromstream(2|truncate_stream(inputs | select(.[0][:2] == ["mechanism_events","_ieee_activations"])))'
        activations = json.loads(subprocess.check_output(['jq','-cn','--stream',query,str(outcome)]))
        unready = []
        if contrast in (('d336', 'd346'), ('d336', 'd349')):
            write(name+'_all_activations.json', activations)
            activations, unready = partition_activation_readiness(
                activations, requests, clock_id=deployment['clock_id'])
        decode = decode_overlap_summary(requests, activations, clock_id=deployment['clock_id'],
            arrival_start=deployment['arrival_start_s'], thresholds=audit['thresholds'])
        native = native_load_overlap_summary(requests, decode.pop('requests'), clock_id=deployment['clock_id'])
        records = native.pop('requests')
        prefill = prefill_overlap_summary(requests, records, clock_id=deployment['clock_id'])
        pf = {r['request_id']: r for r in prefill.pop('requests')}
        with (prior/'requests.csv').open(newline='') as handle:
            normalized = {r['request_id']:r for r in csv.DictReader(handle)}
        raw = {r['request_id']:r for r in requests}
        assert len(normalized) == len(raw) == 4000 and set(normalized) == set(raw)
        for r in records:
            rid = r['request_id']
            q, n = raw[rid], normalized[rid]
            for key, value in pf[rid].items():
                if key in r and r[key] != value:
                    raise ValueError('interval projections disagree')
                r[key] = value
            for key in ('ttft_pass', 'timing_joint_pass', 'tpot_pass'):
                if n[key] not in ('True', 'False'):
                    raise ValueError('invalid retained timing decision')
            assert r['timing_tpot_miss'] == (n['tpot_pass'] == 'False')
            r.update(adapter_id=q['adapter_id'], arrival_offset_s=q['scheduled_arrival_offset_s'],
                canonical_prompt_sha256=q['canonical_prompt_sha256'],
                completion_token_ids_sha256=q['completion_token_ids_sha256'],
                ttft_ms=float(n['ttft_s'])*1000, admission_ms=float(n['arrival_to_admission_s'])*1000,
                service_ms=(float(n['ttft_s'])-float(n['arrival_to_admission_s']))*1000,
                e2e_s=float(n['e2e_s']), ttft_pass=n['ttft_pass']=='True',
                timing_joint_pass=n['timing_joint_pass']=='True')
        life = lifecycle_gaps(physical, deployment)
        offsets = [dict(replica=s['replica'], category=s['category'],
            start_offset_s=s['start']-deployment['arrival_start_s'],
            end_offset_s=s['end']-deployment['arrival_start_s']) for s in decode['activations']]
        table(f'{name}_intervals.csv', records)
        table(f'{name}_activations.csv', offsets)
        for family, result in (('decode', decode), ('load', native), ('prefill', prefill)):
            groups.extend(dict(run=name, family=family, **g) for g in result['groups'])
        transactions.extend(dict(run=name, **g) for g in native['transactions'])
        summaries[name] = dict(source=evidence, stats=stats, lifecycle=life, activation_offsets=offsets,
            decode=decode, native=native, prefill=prefill)
        if contrast in (('d336', 'd346'), ('d336', 'd349')):
            summaries[name].update(unready_activations=unready,
                startup_overlap_definition='Measured trigger-to-ready intervals only. Non-ready epochs retained separately; no ready/cancel time inferred. All physical leases remain in U.',
                all_startup_exposure_complete=not unready)
        retained[name] = dict(requests=records, lifecycle=life, activation_offsets=offsets)
        paths.extend([source, outcome, deployment_path, physical_path, Path(evidence['physical_source']),
            prior/'manifest.json', prior/'audit.json', prior/'requests.csv', prior/'control_samples.csv'])
        del raw, normalized, requests
        gc.collect()
        print(json.dumps(dict(run=name, complete_requests=len(records), lifecycle=life)), flush=True)
    paired = paired_timeline_groups(retained, contrast=contrast)
    table('paired_arrival_cohorts.csv', paired.pop('pairs'))
    table('paired_groups.csv', paired.pop('groups'))
    if contrast in (('d336', 'd346'), ('d336', 'd349')):
        table('phases.csv', [dict(run=name, **row) for name in contrast
            for row in capacity_phase_groups(retained[name]['requests'])])
    # Family summaries have differing columns; retain their complete fields in JSON.
    write('overlap_groups.json', groups)
    table('transactions.csv', transactions)
    write('summary.json', dict(kind=kind, runs=summaries, paired=paired,
        new_performance_runs=0, no_figures=True, n_correct=None, formal_qualification=False,
        confidence_interval=None, causal_savings=None))
    write('manifest.json', dict(source_refs=[dict(path=str(p), sha256=sha256(p)) for p in dict.fromkeys(paths)],
        outputs={p.name:sha256(p) for p in sorted(OUT.iterdir())}, no_figures=True))


def routing_load_consistency(requests, *, clock_id, admitted_edges, delta_ms):
    """D320: conservative witnessed ownership, not exact reserve/release replay.

    Controller admission is after reservation; a request is certainly still
    owned before its native last token. Gaps before admission/after last token
    cannot be reconstructed exactly from this historical schema. File capture
    and Q reads occur in one no-await controller segment, after native awaits.
    """
    from faaslora.experiment.instance_pool import (
        ReplicaRoutingSnapshot, ServiceObservationClass, ServiceComponents)
    if (not requests or not clock_id or not math.isfinite(delta_ms) or delta_ms <= 0
            or any(type(x) is not int or x < 0 for x in admitted_edges)
            or any(a >= b for a, b in zip(admitted_edges, admitted_edges[1:]))):
        raise ValueError('valid requests, clock and frozen class/routing bins required')
    records, owners, seen = [], {}, set()
    for r in requests:
        rid, timing, evidence = r['request_id'], r['native_token_timing'], r['gpu_reference_evidence']
        if (not rid or rid in seen or r['success'] is not True
                or timing['native_clock_id'] != clock_id
                or timing['native_terminal_observed'] is not True):
            raise ValueError('unique completed same-clock native observations required')
        seen.add(rid)
        a, d, last, acquired = [timing[k] for k in (
            'controller_admitted_monotonic_s', 'native_dispatch_monotonic_s',
            'native_last_token_monotonic_s', 'gpu_reference_acquired_monotonic_s')]
        if (not all(type(x) in (float, int) and math.isfinite(x) for x in (a, d, last, acquired))
                or not 0 < a <= d <= last or not 0 < acquired <= d):
            raise ValueError('ordered admission/acquisition/native bounds required')
        snap = evidence['routing_snapshot']
        if (snap['kind'] != 'ieee_predecision_received_view_v1'
                or snap['selected_replica_id'] != r['instance_id']):
            raise ValueError('final request-local routing snapshot required')
        selected = [c for c in snap['candidates'] if c['replica_id'] == r['instance_id']]
        if len(selected) != 1 or selected[0]['service_class'] is None:
            raise ValueError('one selected feasible candidate required')
        owners.setdefault(r['instance_id'], []).append(dict(request_id=rid,
            adapter_id=r['adapter_id'], admitted=a, dispatched=d, last=last,
            acquired=acquired, pending=selected[0]['service_class']['tier'] != 'gpu'))
    for r in requests:
        rid, snap = r['request_id'], r['gpu_reference_evidence']['routing_snapshot']
        raw_rows, sources = snap['candidates'], snap['sources']
        rows = []
        for raw in raw_rows:
            value = dict(raw, active_adapters=frozenset(raw['active_adapters']))
            value['service_class'] = ServiceObservationClass(**raw['service_class']) if raw['service_class'] else None
            value['service'] = ServiceComponents(**raw['service']) if raw['service'] else None
            rows.append(ReplicaRoutingSnapshot(**value))
        if (not rows or len({s.replica_id for s in rows}) != len(rows)
                or len({s.epoch for s in rows}) != 1
                or set(sources) != {s.replica_id for s in rows}
                or any(s.request_id != rid or s.adapter_id != r['adapter_id'] for s in rows)):
            raise ValueError('complete same-request same-epoch candidate population required')
        feasible = [s for s in rows if s.feasible]
        if not feasible:
            raise ValueError('selected request has no feasible row')
        winner = min(feasible, key=lambda s: s.routing_key(delta_ms))
        selected = next(s for s in rows if s.replica_id == snap['selected_replica_id'])
        min_q = min(s.admitted_requests for s in feasible)
        for s in rows:
            at = sources[s.replica_id]['file_captured_monotonic_s']
            if (type(at) not in (float, int) or not math.isfinite(at) or at <= 0
                    or at > r['native_token_timing']['controller_admitted_monotonic_s']):
                raise ValueError('pre-admission same-host file observation required')
            peers = [p for p in owners.get(s.replica_id, []) if p['request_id'] != rid
                     and p['admitted'] <= at < p['last']]
            native = sum(p['dispatched'] <= at for p in peers)
            pending = sum(p['pending'] and at < p['acquired'] for p in peers)
            active = {p['adapter_id'] for p in peers}
            is_selected = s.replica_id == selected.replica_id
            records.append(dict(request_id=rid, replica_id=s.replica_id, selected=is_selected,
                epoch=s.epoch, observation_monotonic_s=at, feasible=s.feasible,
                reported_q=s.admitted_requests, witnessed_q_lower_bound=len(peers),
                witnessed_native_lower_bound=native, witnessed_predispatch_lower_bound=len(peers)-native,
                q_below_witnessed_lower_bound=s.admitted_requests < len(peers),
                reported_pending=s.pending_loads, witnessed_pending_lower_bound=pending,
                pending_below_witnessed_lower_bound=s.pending_loads < pending,
                missing_witnessed_active_adapters=len(active-set(s.active_adapters)),
                admitted_bin=s.service_class.admitted_bin if s.feasible else None,
                admitted_bin_mismatch=(s.service_class.admitted_bin != bisect_left(admitted_edges, s.admitted_requests+1)) if s.feasible else None,
                selection_key_mismatch=winner.replica_id != selected.replica_id if is_selected else None,
                min_feasible_q=min_q, selected_above_min_q=is_selected and s.admitted_requests > min_q,
                tier=s.service_class.tier if s.feasible else None,
                service_bin=math.floor(s.service.total_ms/delta_ms) if s.feasible else None,
                predicted_d_ms=s.service.d_ms if s.feasible else None,
                predicted_t_ms=s.service.t_ms if s.feasible else None,
                predicted_o_ms=s.service.o_ms if s.feasible else None))
    selected_rows = [r for r in records if r['selected']]
    return dict(requests=len(requests), candidate_views=len(records),
        feasible_views=sum(r['feasible'] for r in records),
        violations={k:sum(bool(r[k]) for r in records) for k in (
            'q_below_witnessed_lower_bound', 'pending_below_witnessed_lower_bound',
            'missing_witnessed_active_adapters', 'admitted_bin_mismatch', 'selection_key_mismatch')},
        views_with_witnessed_native=sum(r['witnessed_native_lower_bound']>0 for r in records),
        views_with_witnessed_predispatch=sum(r['witnessed_predispatch_lower_bound']>0 for r in records),
        views_with_witnessed_pending=sum(r['witnessed_pending_lower_bound']>0 for r in records),
        selected_above_min_feasible_q=sum(r['selected_above_min_q'] for r in selected_rows),
        selected_q_histogram=dict(sorted(Counter(r['reported_q'] for r in selected_rows).items())),
        exact_controller_ownership_reconstruction=False, causal_performance_gain=None,
        caveat='Lower bounds exclude pre-admission reservations and post-last-token cleanup; zero contradictions is not proof of exact equality. Selection replay is an implementation consistency check, not counterfactual performance.',
        candidates=records)


def inspect_retained_routing_load(output):
    """Reuse two sealed full runs; never start inference or overwrite outputs."""
    global OUT
    OUT = Path(output).resolve()
    if OUT.exists():
        raise FileExistsError(OUT)
    assert sha256(PLAN) == '0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    assert sha256(PROTOCOL) == '5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    import yaml
    paths = [PLAN, PROTOCOL, Path(__file__), ROOT/'scripts/ieee_tc/plot_d290_diagnostic.py',
        ROOT/'scripts/run_all_experiments.py', ROOT/'faaslora/experiment/instance_pool.py']
    identities = {'d315': '6e5007630c9780c417e65109ef3005daedba3dafa87a843ff75c4885f91c362f',
                  'd318': '4c781c7b10890293130b425e7cc94dc80c96f3b3dd8037f89db53e1fb08ffb1b'}
    OUT.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for name, digest in identities.items():
        prior = ROOT/f'paper_results/ieee_tc/p2_backend/{name}_full_w0/validated1'
        seal = read(prior/'manifest.json')
        assert sha256(prior/'audit.json') == seal['outputs']['audit.json']
        audit = read(prior/'audit.json')
        source = Path(audit['evidence']['source'])
        assert sha256(source) == digest == audit['evidence']['source_sha256']
        cfg_path = ROOT/f'configs/ieee_tc/20261007_{name}_7b_full_w0.yaml'
        refs = {r['path']: r['sha256'] for r in seal['source_refs']}
        assert sha256(cfg_path) == refs[str(cfg_path)]
        config = yaml.safe_load(cfg_path.read_text())['resource_coordination']
        profile_path = ROOT/config['ieee_service_profile']['path']
        assert sha256(profile_path) == config['ieee_service_profile']['sha256']
        stats, evidence, requests = audit_run(source)
        assert len(requests) == 4000
        clocks = {r['native_token_timing']['native_clock_id'] for r in requests}
        assert len(clocks) == 1
        result = routing_load_consistency(requests, clock_id=clocks.pop(),
            admitted_edges=read(profile_path)['bins']['admitted_requests'],
            delta_ms=config['service_bin_ms'])
        table(f'{name}_candidates.csv', result.pop('candidates'))
        result.update(source=evidence, stats=stats)
        summaries[name] = result
        paths.extend([source, cfg_path, profile_path, prior/'audit.json', prior/'manifest.json',
            Path(evidence['physical_source'])])
        del requests
        gc.collect()
        print(json.dumps(dict(run=name, **result), allow_nan=False), flush=True)
    write('summary.json', dict(kind='d320_retained_routing_load_consistency_v1', runs=summaries,
        new_performance_runs=0, no_figures=True, n_correct=None, formal_qualification=False))
    write('manifest.json', dict(source_refs=[dict(path=str(p), sha256=sha256(p)) for p in dict.fromkeys(paths)],
        outputs={p.name:sha256(p) for p in sorted(OUT.iterdir())}, no_figures=True))


def routing_length_diagnosis(candidates, requests):
    """Observed selected-cost calibration only; alternative latency is unknown."""
    by_request, records = {}, []
    for row in candidates:
        by_request.setdefault(row['request_id'], []).append(row)
    if len({r['request_id'] for r in requests}) != len(requests) or set(by_request) != {r['request_id'] for r in requests}:
        raise ValueError('matched unique complete request population required')
    for r in requests:
        feasible = [s for s in by_request[r['request_id']] if s['feasible']]
        chosen = [s for s in feasible if s['selected']]
        if len(chosen) != 1:
            raise ValueError('one selected feasible candidate required')
        s = chosen[0]
        low_q = min(c['reported_q'] for c in feasible)
        alternative = min((c for c in feasible if c['reported_q'] == low_q),
            key=lambda c: (c['service_bin'], sum(c[k] for k in ('predicted_d_ms', 'predicted_t_ms', 'predicted_o_ms')), c['replica_id']))
        observed_o = 1000*(r['last']-r['first'])
        if not math.isfinite(observed_o) or observed_o < 0 or r['tokens'] < 1:
            raise ValueError('valid observed native decode interval required')
        records.append(dict(request_id=r['request_id'], output_group=r['output_group'], tokens=r['tokens'],
            selected_q=s['reported_q'], minimum_q=low_q, selected_tier=s['tier'],
            alternative_tier=alternative['tier'], predicted_o_ms=s['predicted_o_ms'],
            observed_o_ms=observed_o, decode_prediction_error_ms=s['predicted_o_ms']-observed_o,
            selected_above_min_q=s['reported_q']>low_q,
            predicted_d_advantage_ms=alternative['predicted_d_ms']-s['predicted_d_ms'],
            predicted_t_advantage_ms=alternative['predicted_t_ms']-s['predicted_t_ms'],
            predicted_o_advantage_ms=alternative['predicted_o_ms']-s['predicted_o_ms'],
            alternative_performance_known=False))
    groups = []
    for name in ['all'] + sorted({r['output_group'] for r in records}):
        rows = [r for r in records if name == 'all' or r['output_group'] == name]
        groups.append(dict(group=name, requests=len(rows),
            mean_predicted_o_ms=mean(r['predicted_o_ms'] for r in rows),
            mean_observed_o_ms=mean(r['observed_o_ms'] for r in rows),
            mean_decode_prediction_error_ms=mean(r['decode_prediction_error_ms'] for r in rows),
            mean_absolute_decode_prediction_error_ms=mean(abs(r['decode_prediction_error_ms']) for r in rows),
            selected_above_min_q=sum(r['selected_above_min_q'] for r in rows),
            above_min_q_without_predicted_d_advantage=sum(r['selected_above_min_q'] and
                r['predicted_d_advantage_ms'] <= 0 for r in rows)))
    return dict(groups=groups, requests=records, causal_performance_gain=None,
        caveat='Selected estimates versus their own observed outcome; unselected outcomes unknown. Completed token count equals the previously validated fixed target, not a new online future-length feature.')


def lifecycle_pair_decomposition(runs, contrast=('d318', 'd323')):
    """Exact observed envelope-minus-gaps identity, not a policy counterfactual."""
    if len(contrast) != 2 or contrast[0] == contrast[1] or set(runs) != set(contrast):
        raise ValueError('complete explicitly identified reference/candidate pair required')
    reference, candidate = contrast
    summaries, devices = {}, {}
    for name, run in runs.items():
        p, d = run['physical'], run['deployment']
        life = lifecycle_gaps(p, d)
        origin = d['arrival_start_s']
        if type(origin) not in (int, float) or not math.isfinite(origin):
            raise ValueError('finite arrival origin required')
        devices[name] = {}
        for gpu, spans in p['device_intervals'].items():
            if not spans:
                raise ValueError('empty physical device history')
            devices[name][gpu] = dict(first_offset_s=spans[0][0]-origin,
                final_offset_s=spans[-1][1]-origin,
                envelope_gpu_s=spans[-1][1]-spans[0][0])
        envelope = sum(v['envelope_gpu_s'] for v in devices[name].values())
        if abs(envelope-life['released_gap_gpu_s']-life['observed_gpu_s']) > 1e-6:
            raise ValueError('envelope-minus-gap identity failed')
        summaries[name] = dict(**life, envelope_gpu_s=envelope, devices=devices[name])
    if set(devices[reference]) != set(devices[candidate]):
        raise ValueError('same physical device population required')
    old, new = summaries[reference], summaries[candidate]
    boundary = new['envelope_gpu_s']-old['envelope_gpu_s']
    gaps = old['released_gap_gpu_s']-new['released_gap_gpu_s']
    delta = new['observed_gpu_s']-old['observed_gpu_s']
    if abs(boundary+gaps-delta) > 1e-6:
        raise ValueError('paired resource decomposition failed')
    earlier_allocation = sum(old['devices'][gpu]['first_offset_s']-
        new['devices'][gpu]['first_offset_s'] for gpu in old['devices'])
    later_release = sum(new['devices'][gpu]['final_offset_s']-
        old['devices'][gpu]['final_offset_s'] for gpu in old['devices'])
    if abs(earlier_allocation+later_release-boundary) > 1e-6:
        raise ValueError('allocation/release boundary identity failed')
    return dict(runs=summaries, candidate_minus_reference_gpu_s=delta,
        envelope_boundary_delta_gpu_s=boundary, reduced_release_gaps_gpu_s=gaps,
        earlier_allocation_gpu_s=earlier_allocation, later_release_gpu_s=later_release,
        causal_savings=None, confidence_interval=None)


def inspect_retained_service_lifecycle(output):
    """D324: one bounded retained-data check after the complete D323 replay."""
    global OUT
    OUT = Path(output).resolve()
    if OUT.exists():
        raise FileExistsError(OUT)
    assert sha256(PLAN) == '0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    assert sha256(PROTOCOL) == '5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    import yaml
    identities = {'d318': '4c781c7b10890293130b425e7cc94dc80c96f3b3dd8037f89db53e1fb08ffb1b',
                  'd323': 'a790cd59968a0d8d33bc017938467c1f818c5e5c617fb9628ab25a4843b5e991'}
    paths = [PLAN, PROTOCOL, Path(__file__), ROOT/'scripts/ieee_tc/plot_d290_diagnostic.py',
        ROOT/'faaslora/experiment/instance_pool.py']
    runs, lifecycle, inputs, outputs, groups, actions = {}, {}, {}, {}, [], []
    OUT.mkdir(parents=True, exist_ok=False)
    for name, digest in identities.items():
        prior = ROOT/f'paper_results/ieee_tc/p2_backend/{name}_full_w0/validated1'
        seal = read(prior/'manifest.json')
        for filename in ('audit.json', 'control_samples.csv'):
            assert sha256(prior/filename) == seal['outputs'][filename]
        audit = read(prior/'audit.json')
        source = Path(audit['evidence']['source'])
        assert sha256(source) == digest == audit['evidence']['source_sha256']
        cfg_path = ROOT/f'configs/ieee_tc/20261007_{name}_7b_full_w0.yaml'
        refs = {r['path']:r['sha256'] for r in seal['source_refs']}
        assert sha256(cfg_path) == refs[str(cfg_path)]
        config = yaml.safe_load(cfg_path.read_text())['resource_coordination']
        profile = ROOT/config['ieee_service_profile']['path']
        assert sha256(profile) == config['ieee_service_profile']['sha256']
        stats, evidence, requests = audit_run(source)
        assert len(requests) == 4000
        assert evidence == audit['evidence']
        physical_path = source.parent/'launch.launch/physical_deployment/summary.json'
        deployment_path = source.parent/'launch.launch/physical_deployment/deployment.json'
        physical, deployment = read(physical_path), read(deployment_path)
        assert abs(physical['gpu_seconds']-stats['physical_gpu_seconds']) < 1e-6
        lifecycle[name] = dict(physical=physical, deployment=deployment)
        result = routing_load_consistency(requests, clock_id=deployment['clock_id'],
            admitted_edges=read(profile)['bins']['admitted_requests'], delta_ms=config['service_bin_ms'])
        candidates = result.pop('candidates')
        projection = [dict(request_id=r['request_id'], tokens=r['completion_tokens'],
            output_group='2-32' if r['completion_tokens']<=32 else '33-128' if r['completion_tokens']<=128 else '129-256',
            first=r['native_token_timing']['native_first_token_monotonic_s'],
            last=r['native_token_timing']['native_last_token_monotonic_s']) for r in requests]
        lengths = routing_length_diagnosis(candidates, projection)
        table(f'{name}_requests.csv', lengths.pop('requests'))
        groups.extend(dict(run=name, **g) for g in lengths.pop('groups'))
        runs[name] = dict(stats=stats, evidence=evidence, routing=result, length_caveat=lengths,
            declared_output_edges=read(profile)['bins']['declared_output_tokens'])
        inputs[name] = {r['request_id']:tuple(r[k] for k in ('adapter_id', 'scheduled_arrival_offset_s',
            'canonical_prompt_sha256', 'requested_completion_tokens', 'generation_contract')) for r in requests}
        outputs[name] = {r['request_id']:r['completion_token_ids_sha256'] for r in requests}
        with (prior/'control_samples.csv').open(newline='') as handle:
            actions.extend(dict(run=name, **r) for r in csv.DictReader(handle) if r['action'] != 'no_action')
        paths.extend([source, cfg_path, profile, physical_path, deployment_path,
            Path(evidence['physical_source']), prior/'manifest.json', prior/'audit.json', prior/'control_samples.csv'])
        print(json.dumps(dict(run=name, requests=len(requests), violations=result['violations'],
            length_groups=[g for g in groups if g['run']==name]), allow_nan=False), flush=True)
        del requests, candidates, projection
        gc.collect()
    assert inputs['d318'] == inputs['d323']
    differences = [dict(request_id=rid, reference_sha=outputs['d318'][rid], candidate_sha=outputs['d323'][rid])
        for rid in sorted(inputs['d318']) if outputs['d318'][rid] != outputs['d323'][rid]]
    life = lifecycle_pair_decomposition(lifecycle)
    table('length_groups.csv', groups)
    table('control_actions.csv', actions)
    write('output_hash_differences.json', differences)
    write('summary.json', dict(kind='d324_retained_service_lifecycle_v1', runs=runs, lifecycle=life,
        output_hash_differences=len(differences), same_input_vectors=True, new_performance_runs=0,
        no_figures=True, n_correct=None, formal_qualification=False, confidence_interval=None))
    write('manifest.json', dict(source_refs=[dict(path=str(p), sha256=sha256(p)) for p in dict.fromkeys(paths)],
        outputs={p.name:sha256(p) for p in sorted(OUT.iterdir())}, no_figures=True))
    print(json.dumps(life, indent=2, allow_nan=False), flush=True)


def inspect_retained_routing_lengths(output):
    global OUT
    OUT = Path(output).resolve()
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = [Path(__file__), PLAN, PROTOCOL]
    all_groups = []
    OUT.mkdir(parents=True, exist_ok=False)
    for name in ('d315', 'd318'):
        rows = []
        for family, filename in [('d320_routing_load', f'{name}_candidates.csv'),
                                 ('d319_lifecycle', f'{name}_intervals.csv')]:
            prior = ROOT/f'paper_results/ieee_tc/p2_backend/{family}/analysis1'
            manifest = read(prior/'manifest.json')
            path = prior/filename
            assert sha256(path) == manifest['outputs'][filename]
            refs = {r['path']:r['sha256'] for r in manifest['source_refs']}
            assert sha256(PLAN) == refs[str(PLAN)] and sha256(PROTOCOL) == refs[str(PROTOCOL)]
            with path.open(newline='') as handle:
                data = list(csv.DictReader(handle))
            for r in data:
                for k in ('selected', 'feasible'):
                    if k in r:
                        if r[k] not in ('True', 'False'): raise ValueError('invalid boolean')
                        r[k] = r[k] == 'True'
                for k in ('reported_q', 'tokens'):
                    if k in r: r[k] = int(r[k])
                for k in ('service_bin', 'predicted_d_ms', 'predicted_t_ms', 'predicted_o_ms', 'last', 'first'):
                    if k in r: r[k] = float(r[k]) if r[k] else None
            rows.append(data)
            paths.extend([path, prior/'manifest.json'])
        result = routing_length_diagnosis(*rows)
        table(f'{name}_requests.csv', result.pop('requests'))
        all_groups.extend(dict(run=name, **g) for g in result.pop('groups'))
    write('summary.json', dict(groups=all_groups, new_performance_runs=0, no_figures=True, **result))
    write('manifest.json', dict(source_refs=[dict(path=str(p), sha256=sha256(p)) for p in dict.fromkeys(paths)],
        outputs={p.name:sha256(p) for p in sorted(OUT.iterdir())}, no_figures=True))
    print(json.dumps(all_groups, indent=2, allow_nan=False), flush=True)


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
    elif campaign == 'd318':
        RAW = ROOT / 'results/ieee_tc/p2_backend_qualification/d318_20261007'
        RUN = RAW / '7b_full_w0'
        OUT = ROOT / 'paper_results/ieee_tc/p2_backend/d318_full_w0/validated1'
        SOURCE = RUN / 'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d318_7b_full_w0.json'
        source_sha = '4c781c7b10890293130b425e7cc94dc80c96f3b3dd8037f89db53e1fb08ffb1b'
    elif campaign == 'd323':
        RAW = ROOT / 'results/ieee_tc/p2_backend_qualification/d323_20261007'
        RUN = RAW / '7b_full_w0'
        OUT = ROOT / 'paper_results/ieee_tc/p2_backend/d323_full_w0/validated1'
        SOURCE = RUN / 'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d323_7b_full_w0.json'
        source_sha = 'a790cd59968a0d8d33bc017938467c1f818c5e5c617fb9628ab25a4843b5e991'
    elif campaign == 'd326':
        RAW = ROOT / 'results/ieee_tc/p2_backend_qualification/d326_20261008'
        RUN = RAW / '7b_full_w0'
        OUT = ROOT / 'paper_results/ieee_tc/p2_backend/d326_full_w0/validated1'
        SOURCE = RUN / 'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d326_7b_full_w0.json'
        source_sha = '9e1057f50d1cc8dafacfd4f232216cd5979ac6c7e9f2a479dcd26f654295150a'
    elif campaign == 'd332':
        RAW = ROOT / 'results/ieee_tc/p2_backend_qualification/d332_20261008'
        RUN = RAW / '7b_full_w0'
        OUT = ROOT / 'paper_results/ieee_tc/p2_backend/d332_full_w0/validated1'
        SOURCE = RUN / 'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d332_7b_full_w0.json'
        source_sha = '8d82df4014aac5b5476d994ea5255b89d87e28ea9ba739bec1816e1fdf14cef7'
    elif campaign == 'd335':
        RAW = ROOT / 'results/ieee_tc/p2_backend_qualification/d335_20261008'
        RUN = RAW / '7b_full_w0'
        OUT = ROOT / 'paper_results/ieee_tc/p2_backend/d335_full_w0/validated1'
        SOURCE = RUN / 'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d335_7b_full_w0.json'
        source_sha = 'a3b6ee044db0ba46915e94ad1de8e67f638b3b0f188fed7eaac1e69078ffaa97'
    elif campaign == 'd336':
        RAW = ROOT / 'results/ieee_tc/p2_backend_qualification/d336_20261008'
        RUN = RAW / '7b_full_w0'
        OUT = ROOT / 'paper_results/ieee_tc/p2_backend/d336_full_w0/validated1'
        SOURCE = RUN / 'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d336_7b_full_w0.json'
        source_sha = 'b8452f4f577446128b4a77a7117485660f3b9841caf831ad867a58d9a9587a05'
    elif campaign == 'd341':
        RAW = ROOT / 'results/ieee_tc/p2_backend_qualification/d341_20261008'
        RUN = RAW / '7b_full_w0'
        OUT = ROOT / 'paper_results/ieee_tc/p2_backend/d341_full_w0/validated1'
        SOURCE = RUN / 'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d341_7b_full_w0.json'
        source_sha = '9f87d2e10ac42d08717d40a7fbafe725c16916eb4e7d34aa70cee177fd39b664'
    elif campaign == 'd346':
        RAW = ROOT / 'results/ieee_tc/p2_backend_qualification/d346_20261008'
        RUN = RAW / '7b_full_w0'
        OUT = ROOT / 'paper_results/ieee_tc/p2_backend/d346_full_w0/validated1'
        SOURCE = RUN / 'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d346_7b_full_w0.json'
        source_sha = '52e7d3fddde1f4bd8ae05831dd4ce0d69e291f9c2b9aba3a7ef5b659a987151e'
    elif campaign == 'd349':
        RAW = ROOT / 'results/ieee_tc/p2_backend_qualification/d349_20261008'
        RUN = RAW / '7b_full_w0'
        OUT = ROOT / 'paper_results/ieee_tc/p2_backend/d349_full_w0/validated1'
        SOURCE = RUN / 'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d349_7b_full_w0.json'
        source_sha = 'c289b97bd146ae8ec355f59ea5abd373b271e9848814ea3b60ddff58cfdd97e7'
    elif campaign in ('d353', 'd358', 'd364', 'd367'):
        date = '20261009' if campaign in ('d364', 'd367') else '20261008'
        RAW = ROOT / f'results/ieee_tc/p2_backend_qualification/{campaign}_{date}'
        RUN = RAW / '7b_full_w0'
        OUT = ROOT / f'paper_results/ieee_tc/p2_backend/{campaign}_full_w0/validated1'
        SOURCE = RUN / f'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_{campaign}_7b_full_w0.json'
        # The collection receipt is created only after physical release and
        # exact-owned local/remote closure; never analyze a live prefix here.
        collected = read(RAW / 'collection_receipt.json')
        assert Path(collected['source_path']).resolve() == SOURCE.resolve()
        assert collected['source_bytes'] == SOURCE.stat().st_size
        assert read(RAW / 'local_cleanup.json')['native_gpu_release_confirmed'] is True
        assert read(RAW / 'remote_stop.json')['all_owned_units_inactive'] is True
        source_sha = collected['source_sha256']
    elif campaign != 'd297':
        raise ValueError('unknown frozen full-run identity')
    if OUT.exists():
        raise FileExistsError(OUT)
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
    if campaign in ('d303', 'd314', 'd315', 'd318', 'd323', 'd326', 'd332', 'd335', 'd336', 'd341', 'd346', 'd349', 'd353', 'd358', 'd364', 'd367'):
        write('admission_waits.json', summarize_admission_waits(requests))
    if campaign in ('d314', 'd315', 'd318', 'd323', 'd326', 'd332', 'd335', 'd336', 'd341', 'd346', 'd349', 'd353', 'd358', 'd364', 'd367'):
        write('preparation_attribution.json', audit_preparation_attribution(
            requests, clock_id=deployment['clock_id']))
    if campaign in ('d318', 'd323', 'd326', 'd332', 'd335', 'd336', 'd341', 'd346', 'd349', 'd353', 'd358', 'd364', 'd367'):
        from scripts.ieee_tc.analyze_d298_smoke import copy_retention_trigger
        trigger = copy_retention_trigger(requests)
        assert trigger['observed_cumulative_retained_scopes_max'] > 0
        write('copy_retention_trigger.json', trigger)
    if campaign in ('d335', 'd336', 'd341', 'd346', 'd349', 'd353', 'd358', 'd364', 'd367'):
        from scripts.ieee_tc.analyze_d298_smoke import native_pending_turn_trigger
        write('native_pending_turn_trigger.json', native_pending_turn_trigger(requests))
    if campaign in ('d353', 'd358', 'd364', 'd367'):
        from scripts.ieee_tc.analyze_d298_smoke import hit_host_transaction_trigger
        write('hit_host_transaction_trigger.json', hit_host_transaction_trigger(requests))
    if campaign in ('d358', 'd364', 'd367'):
        from scripts.ieee_tc.analyze_d298_smoke import control_interleave_trigger
        write('control_interleave_trigger.json', control_interleave_trigger(requests))
    if campaign in ('d364', 'd367'):
        from scripts.ieee_tc.analyze_d298_smoke import file_demand_transaction_trigger
        write('file_demand_transaction_trigger.json', file_demand_transaction_trigger(requests))
    del requests, normalized
    gc.collect()

    # jq's streaming parser avoids materializing the 629-MiB outcome tree.
    outcome_path = RUN / 'launch.launch/physical_deployment/main_outcome.json'
    if campaign in ('d332', 'd335', 'd336', 'd341', 'd346', 'd349', 'd353', 'd358', 'd364', 'd367'):
        from scripts.ieee_tc.analyze_d298_smoke import planning_publication_trigger
        query = 'fromstream(2|truncate_stream(inputs | select(.[0][:2] == ["mechanism_events","_ieee_route_cache_stats"])))'
        stats = json.loads(subprocess.check_output(['jq','-cn','--stream',query,str(outcome_path)]))
        write('planning_publication_trigger.json', planning_publication_trigger(
            {'_ieee_route_cache_stats': stats}))
    query = 'fromstream(2|truncate_stream(inputs | select(.[0][:2] == ["mechanism_events","_ieee_control_events"])))'
    controls = json.loads(subprocess.check_output(['jq','-cn','--stream',query,str(outcome_path)]))
    import yaml
    config_date = ('20261009' if campaign in ('d364', 'd367') else '20261008' if campaign in
        ('d326', 'd332', 'd335', 'd336', 'd341', 'd346', 'd349', 'd353', 'd358') else '20261007')
    config_path = ROOT / f'configs/ieee_tc/{config_date}_{campaign}_7b_full_w0.yaml'
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
    if campaign in ('d318', 'd323', 'd326', 'd332', 'd335', 'd336', 'd341', 'd346', 'd349', 'd353', 'd358', 'd364', 'd367'):
        paths.append(ROOT/'scripts/ieee_tc/analyze_d298_smoke.py')
    if campaign in ('d353', 'd358', 'd364', 'd367'):
        paths.extend(RAW/name for name in (
            'collection_receipt.json', 'local_cleanup.json', 'remote_stop.json'))
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


TTFT_STAGE_MS = ('arrival_release_lateness_ms', 'pre_acquire_ms',
    'acquire_to_resume_ms', 'runtime_slot_wait_ms', 'admission_to_engine_dispatch_ms',
    'engine_entry_to_queue_ms', 'native_engine_queue_ms', 'native_prefill_ms')


def retained_ttft_stages(row, request, timing):
    """Join existing same-request clocks; routing is nested, not an extra stage."""
    if not row['request_id'] == request['request_id'] == timing['request_id']:
        raise ValueError('same-request stage identity required')
    values = {k: row[k] for k in TTFT_STAGE_MS[1:4]}
    values['arrival_release_lateness_ms'] = request['arrival_release_lateness_ms']
    for key in TTFT_STAGE_MS[4:]:
        values[key] = float(timing[key[:-3]+'_s'])*1000
    values['ttft_ms'] = float(timing['ttft_s'])*1000
    values['tpot_ms'] = float(timing['tpot_s'])*1000
    values['routing_nested_ms'] = request['control_path_total_us']/1000
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
           for v in values.values()):
        raise ValueError('complete finite nonnegative stages required')
    error = abs(math.fsum(values[k] for k in TTFT_STAGE_MS)-values['ttft_ms'])
    admission = math.fsum(values[k] for k in TTFT_STAGE_MS[:4])
    if max(error, abs(admission-float(timing['arrival_to_admission_s'])*1000),
           abs(values['ttft_ms']-request['overall_ttft_ms'])) > 1:
        raise ValueError('retained TTFT stage identity exceeds 1ms')
    values['stage_identity_error_ms'] = error
    values['token_count'] = int(timing['token_count'])
    if values['token_count'] < 2:
        raise ValueError('this registered diagnostic requires measured multi-token requests')
    values['tier'] = timing['tier']
    return values


def retained_preparation_row(request):
    """Describe admission/acquisition/load receipts without filling missing spans.

    A GPU reference may be acquired before source admission. A non-GPU selected
    source may legally become resident while waiting; no load is then invented.
    The final conflict is not a journal of every intermediate conflict.
    """
    e, timing = request['gpu_reference_evidence'], request['native_token_timing']
    admission, receipt = e['source_admission'], e['receipt']
    tier = admission['source']['tier']
    if (request['success'] is not True or tier != request['readiness_tier_before_dispatch']
            or tier not in ('gpu','host','nvme','remote') or receipt['acquired'] is not True
            or receipt['clock_id'] != timing['native_clock_id']
            or admission['clock_id'] != receipt['clock_id']
            or receipt['owner_id'] != timing['gpu_reference_owner_id']
            or receipt['lease_id'] != timing['gpu_reference_lease_id']):
        raise ValueError('matched successful preparation receipts required')
    admitted = admission['admitted_monotonic_s']
    acquired, dispatch = receipt['acquired_monotonic_s'], timing['native_dispatch_monotonic_s']
    if (any(type(x) not in (int,float) or not math.isfinite(x) or x <= 0
            for x in (admitted,acquired,dispatch))
            or admitted != timing['controller_admitted_monotonic_s']
            or max(admitted,acquired)>dispatch or (tier!='gpu' and acquired<admitted)):
        raise ValueError('ordered same-clock preparation boundaries required')
    retries = e['stale_rechecks']
    invoked = receipt['native_load_invoked']
    if type(retries) is not int or retries < 0 or type(invoked) is not bool:
        raise ValueError('explicit stale counter and native-load flag required')
    start, end = receipt['native_load_started_monotonic_s'], receipt['native_load_completed_monotonic_s']
    before, native = None, None
    if invoked:
        if (any(type(x) not in (int,float) or not math.isfinite(x) for x in (start,end))
                or not admitted <= start <= end == acquired):
            raise ValueError('ordered final native-load boundaries required')
        before, native = (start-admitted)*1000, (end-start)*1000
    elif start is not None or end is not None:
        raise ValueError('non-loading acquisition must not claim a load span')
    interval = e.get('preparation_interval')
    if tier!='gpu' and (not isinstance(interval,dict)
            or interval['request_id']!=request['request_id']
            or interval['native_lease_id']!=receipt['lease_id']):
        raise ValueError('selected non-GPU source requires its retained attribution')
    conflict = e.get('last_conflict')
    reason = conflict['reason'] if conflict is not None else None
    return dict(request_id=request['request_id'],adapter_id=request['adapter_id'],tier=tier,
        native_source=receipt['source_tier_before_acquisition'],native_load_invoked=invoked,
        preparation_reason=interval['reason'] if interval else 'gpu_reference_before_admission',
        owner_id=receipt['owner_id'],stale_rechecks=retries,last_conflict_reason=reason,
        admission_to_acquired_ms=(max(admitted,acquired)-admitted)*1000,
        acquired_before_admission_ms=max(0.,admitted-acquired)*1000,
        executable_to_engine_ms=(dispatch-max(admitted,acquired))*1000,
        admission_to_engine_ms=(dispatch-admitted)*1000,
        before_final_load_ms=before,final_native_load_ms=native,
        final_rpc_timing=receipt.get('timing'))


def inspect_retained_preparation(output):
    """D360: reuse D353/D358 receipts to test cold-path stale-retry amplification."""
    global OUT
    OUT = Path(output).resolve()
    if OUT.exists():
        raise FileExistsError(OUT)
    assert sha256(PLAN)=='0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    assert sha256(PROTOCOL)=='5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    old = ROOT/'paper_results/ieee_tc/p2_backend/d359_ttft_partition/analysis1'
    old_seal = read(old/'manifest.json')
    refs = {x['path']:x['sha256'] for x in old_seal['source_refs']}
    paths=[PLAN,PROTOCOL,Path(__file__),old/'manifest.json']
    summaries, groups = {}, []
    OUT.mkdir(parents=True,exist_ok=False)
    for name in ('d353','d358'):
        prior=ROOT/f'paper_results/ieee_tc/p2_backend/{name}_full_w0/validated1'
        audit=read(prior/'audit.json')
        assert sha256(prior/'audit.json')==refs[str(prior/'audit.json')]
        source=Path(audit['evidence']['source'])
        assert sha256(source)==refs[str(source)]==audit['evidence']['source_sha256']
        result,scenario=load(source)
        requests=scenario['requests']
        assert len(requests)==4000
        rows=[]
        for i,r in enumerate(requests):
            row=retained_preparation_row(r)
            row.update(trace_index=i,phase=i//500+1)
            assert abs(row['admission_to_engine_ms']-r['native_token_timing']['admission_to_engine_dispatch_ms'])<.001
            rows.append(row)
        cohorts=[('all','all',rows)]
        cohorts += [('phase',str(i),[r for r in rows if r['phase']==i]) for i in range(1,9)]
        cohorts += [('tier',t,[r for r in rows if r['tier']==t]) for t in ('gpu','host','nvme','remote')]
        cohorts += [('second_phase_tier',t,[r for r in rows if r['tier']==t and r['phase']==2])
                    for t in ('gpu','host','nvme','remote')]
        cohorts += [('stale_rechecks',label,[r for r in rows if (r['stale_rechecks']>0)==positive])
                    for label,positive in (('none',False),('positive',True))]
        for kind,label,rs in cohorts:
            loaded=[r for r in rs if r['native_load_invoked']]
            groups.append(dict(run=name,group=kind,label=label,n=len(rs),
                stale_affected=sum(r['stale_rechecks']>0 for r in rs),
                stale_rechecks=sum(r['stale_rechecks'] for r in rs),
                max_stale_rechecks=max((r['stale_rechecks'] for r in rs),default=None),
                loaded_count=len(loaded),
                **{k:mean(r[k] for r in rs) if rs else None for k in
                   ('admission_to_acquired_ms','executable_to_engine_ms','admission_to_engine_ms')},
                loaded_before_final_load_ms=mean(r['before_final_load_ms'] for r in loaded) if loaded else None,
                loaded_final_native_load_ms=mean(r['final_native_load_ms'] for r in loaded) if loaded else None))
        summaries[name]=dict(requests=len(rows),last_conflict_counts=dict(Counter(
            r['last_conflict_reason'] or 'none' for r in rows)),
            preparation_counts=dict(Counter(r['preparation_reason'] for r in rows)),
            max_recheck_row=max(rows,key=lambda r:r['stale_rechecks']))
        write(name+'_requests.json',rows)
        paths.extend([source,prior/'audit.json'])
        del result,scenario,requests,rows
        gc.collect()
    table('groups.csv',groups)
    write('summary.json',dict(runs=summaries,new_gpu_runs=0,formal_qualification=False,
        n_correct=None,confidence_interval=None,no_figures=True,caveats=[
            'Final successful native load excludes earlier RPCs and conflicts; it is not total resource use.',
            'Before-final-load includes all earlier preparation, transfer, contention and control waiting.',
            'Only stale_rechecks is a cumulative counter; last_conflict is not a full conflict journal.',
            'GPU acquisition precedes admission; no negative stage or invented load is imputed.',
            'Conditional groups and one development run each cannot establish counterfactual gains.']))
    write('manifest.json',dict(source_refs=[dict(path=str(p),sha256=sha256(p)) for p in paths],
        outputs={p.name:sha256(p) for p in OUT.iterdir()},no_figures=True))
    print(json.dumps(summaries,indent=2,allow_nan=False),flush=True)


def inspect_retained_dispatch_gate(output, contrast=('d336', 'd341')):
    """Reuse registered complete native data, not another performance replay."""
    global OUT
    OUT = Path(output).resolve()
    if OUT.exists():
        raise FileExistsError(OUT)
    assert sha256(PLAN) == '0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    assert sha256(PROTOCOL) == '5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    if contrast not in (('d336', 'd341'), ('d353', 'd358')):
        raise ValueError('unregistered retained dispatch contrast')
    identities = dict(d336='b8452f4f577446128b4a77a7117485660f3b9841caf831ad867a58d9a9587a05',
        d341='9f87d2e10ac42d08717d40a7fbafe725c16916eb4e7d34aa70cee177fd39b664',
        d353='0fe856cf44923ec66fb91aed5ed9d088891bedb7c2b603d416e78b04f0751c9d',
        d358='0d6c188c302d83f57d3f581181d639c2dd37ed486b5817ce2027127ba055fe51')
    paths = [PLAN, PROTOCOL, Path(__file__), ROOT/'scripts/analyze_control_path_overhead.py',
             ROOT/'scripts/ieee_tc/plot_d290_diagnostic.py']
    summaries, input_maps, phases = {}, [], []
    OUT.mkdir(parents=True, exist_ok=False)
    for name in contrast:
        digest = identities[name]
        prior = ROOT/f'paper_results/ieee_tc/p2_backend/{name}_full_w0/validated1'
        audit, seal = read(prior/'audit.json'), read(prior/'manifest.json')
        for filename in ('audit.json', 'requests.csv', 'control_samples.csv'):
            assert sha256(prior/filename) == seal['outputs'][filename]
            paths.append(prior/filename)
        source = Path(audit['evidence']['source'])
        assert sha256(source) == digest == audit['evidence']['source_sha256']
        physical = source.parent/'launch.launch/physical_deployment'
        outcome = physical/'main_outcome.json'
        refs = {r['path']: r['sha256'] for r in seal['source_refs']}
        assert sha256(outcome) == refs[str(outcome)]
        result, scenario = load(source)
        requests = scenario['requests']
        assert len(requests) == 4000
        query = 'fromstream(2|truncate_stream(inputs | select(.[0][:2] == ["mechanism_events","_ieee_activations"])))'
        activations = json.loads(subprocess.check_output(['jq','-cn','--stream',query,str(outcome)]))
        deployment = read(physical/'deployment.json')
        with (physical/'request_terminals.jsonl').open() as f:
            terminals = [json.loads(line) for line in f]
        split = summarize_dispatch_gate_intervals(requests, deployment, terminals, activations)
        rows = split.pop('rows')
        with (prior/'control_samples.csv').open(newline='') as f:
            controls = list(csv.DictReader(f))
        times = [float(c['offset_s']) for c in controls]
        assert times == sorted(times) and len(times) == len(set(times))
        with (prior/'requests.csv').open(newline='') as f:
            timing = {r['request_id']:r for r in csv.DictReader(f)}
        assert set(timing) == {r['request_id'] for r in rows}
        for i, (row, request) in enumerate(zip(rows, requests)):
            assert row['request_id'] == request['request_id']
            row.update(trace_index=i, phase=i//500, adapter_id=request['adapter_id'],
                ttft_ms=request['overall_ttft_ms'], timing_outcome=timing[row['request_id']]['outcome'])
            if contrast == ('d353', 'd358'):
                row.update(retained_ttft_stages(row, request, timing[row['request_id']]))
            for boundary in ('acquire', 'gate_resumed'):
                at = row[boundary+'_offset_s']
                index = bisect_left(times, at)-1
                control = controls[index] if index >= 0 else None
                for field in ('offset_s', 'queue', 'active', 'capacity', 'pending', 'action', 'outcome'):
                    row[boundary+'_prior_control_'+field] = control[field] if control else None
        same_fields = ('request_id', 'adapter_id', 'scheduled_arrival_offset_s',
                       'source_expected_output_tokens', 'requested_completion_tokens',
                       'canonical_prompt_sha256', 'generation_contract', 'completion_token_source')
        input_maps.append((deployment['plan'], result['metadata']['shared_adapter_subset_sha256'],
                           [tuple(r[k] for k in same_fields) for r in requests]))
        split['source_sha256'] = digest
        split['prior_audit_sha256'] = seal['outputs']['audit.json']
        split['top_one_percent_window_request_ids'] = [r['request_id'] for r in
            sorted(rows, key=lambda r: (-r['dispatch_window_wait_ms'], r['request_id']))[:40]]
        for phase in range(8):
            group = [r for r in rows if r['phase'] == phase]
            phases.append(dict(run=name, phase=phase, count=len(group),
                **{k:mean(r[k] for r in group) for k in
                   ('pre_acquire_ms', 'acquire_to_resume_ms', 'dispatch_window_wait_ms', 'ttft_ms')}))
            if contrast == ('d353', 'd358'):
                phases[-1].update({k:mean(r[k] for r in group) for k in TTFT_STAGE_MS +
                    ('tpot_ms', 'routing_nested_ms', 'pre_acquire_startup_overlap_ms',
                     'acquire_to_resume_startup_overlap_ms')})
                phases[-1]['ttft_misses'] = sum(r['timing_outcome'] in
                    ('ttft_only_miss', 'both_miss') for r in group)
                phases[-1]['tpot_misses'] = sum(r['timing_outcome'] in
                    ('tpot_only_miss', 'both_miss') for r in group)
        if contrast == ('d353', 'd358'):
            split['ttft_stage_means_ms'] = {k:mean(r[k] for r in rows) for k in TTFT_STAGE_MS}
            split['max_ttft_stage_error_ms'] = max(r['stage_identity_error_ms'] for r in rows)
            split['routing_nested_mean_ms'] = mean(r['routing_nested_ms'] for r in rows)
        summaries[name] = split
        table(name+'_requests.csv', rows)
        write(name+'_activations.json', activations)
        paths.extend([source, outcome, prior/'manifest.json', physical/'deployment.json',
                      physical/'request_terminals.jsonl'])
        del result, scenario, requests, rows
        gc.collect()
    assert input_maps[0] == input_maps[1], 'comparison input identities differ'
    table('phases.csv', phases)
    write('summary.json', dict(kind=('d342_retained_dispatch_split_v1' if contrast == ('d336', 'd341')
        else 'd359_retained_dispatch_split_v1'), contrast=contrast, runs=summaries,
        matched_inputs=True, new_performance_runs=0, no_figures=True,
        control_caveat='Prior sampled control values are context, not exact live dispatch gate state.'))
    write('manifest.json', dict(source_refs=[dict(path=str(p),sha256=sha256(p)) for p in paths],
        outputs={p.name:sha256(p) for p in sorted(OUT.iterdir())}, no_figures=True))
    print(json.dumps(summaries, indent=2, allow_nan=False), flush=True)


def inspect_retained_dispatch_context(output, prior=None, contrast=('d336', 'd341')):
    """Follow the retained split with conservative sampled ownership bounds."""
    global OUT
    OUT = Path(output).resolve()
    if OUT.exists():
        raise FileExistsError(OUT)
    if contrast not in (('d336', 'd341'), ('d353', 'd358')):
        raise ValueError('unregistered retained dispatch context')
    if prior is None:
        prior = ROOT/'paper_results/ieee_tc/p2_backend/d342_dispatch_wait/analysis1'
    seal = read(prior/'manifest.json')
    paths = [prior/'manifest.json', Path(__file__), ROOT/'scripts/analyze_control_path_overhead.py']
    summaries = {}
    OUT.mkdir(parents=True, exist_ok=False)
    for name in contrast:
        request_path = prior/(name+'_requests.csv')
        control_path = ROOT/f'paper_results/ieee_tc/p2_backend/{name}_full_w0/validated1/control_samples.csv'
        assert sha256(request_path) == seal['outputs'][request_path.name]
        refs = {r['path']:r['sha256'] for r in seal['source_refs']}
        assert sha256(control_path) == refs[str(control_path)]
        with request_path.open(newline='') as f:
            requests = list(csv.DictReader(f))
        for r in requests:
            for k in ('acquire_offset_s','gate_resumed_offset_s','native_completed_offset_s','terminal_offset_s'):
                r[k] = float(r[k])
        with control_path.open(newline='') as f:
            controls = list(csv.DictReader(f))
        for c in controls:
            c['offset_s'] = float(c['offset_s'])
            for k in ('queue','active','capacity','pending'):
                c[k] = int(c[k])
        result = summarize_dispatch_capacity_context(requests, controls)
        rows = result.pop('rows')
        result['peak_wait_sample'] = max(rows, key=lambda r:r['acquire_not_yet_resumed'])
        summaries[name] = result
        table(name+'_capacity_samples.csv', rows)
        paths.extend([request_path,control_path])
    write('summary.json', dict(kind='d342_sampled_capacity_context_v1', runs=summaries,
        new_performance_runs=0, formal_qualification=False, no_figures=True))
    write('manifest.json', dict(source_refs=[dict(path=str(p),sha256=sha256(p)) for p in paths],
        outputs={p.name:sha256(p) for p in sorted(OUT.iterdir())}, no_figures=True))
    print(json.dumps(summaries, indent=2, allow_nan=False), flush=True)


def calibration_exposure(rows):
    """Describe whole-service exposure, not a steady-concurrency service curve.

    A post-admission count is a snapshot. Subsequent admissions can change the
    measured TPOT. Use only same-run, same-wave admission/first/last intervals;
    these envelopes are neither kernel batch sizes nor prefill kernel timings.
    """
    if not rows:
        raise ValueError('nonempty complete calibration required')
    seen = set()
    for r in rows:
        key = (r['source_sha256'], r['request_id'])
        if key in seen or not all(isinstance(x, str) and x for x in key):
            raise ValueError('unique source-bound calibration identities required')
        seen.add(key)
        if (not isinstance(r['clock_id'], str) or not r['clock_id']
                or type(r['wave']) is not int or r['wave'] < 0
                or type(r['admitted_count']) is not int or r['admitted_count'] < 1
                or type(r['tokens']) is not int or r['tokens'] < 2
                or type(r['prompt_tokens']) is not int or r['prompt_tokens'] < 1):
            raise ValueError('explicit native counts and clock required')
        a, f, z = (r[k] for k in ('admitted', 'first', 'last'))
        if (any(type(v) not in (int, float) or not math.isfinite(v)
                for v in (a, f, z)) or not 0 < a <= f < z):
            raise ValueError('ordered finite calibration intervals required')
    groups = {}
    for r in rows:
        groups.setdefault((r['source_sha256'], r['wave']), []).append(r)
    output = []
    for rs in groups.values():
        if len({r['clock_id'] for r in rs}) != 1:
            raise ValueError('mixed-clock calibration wave')
        for r in rs:
            f, z = r['first'], r['last']
            peers = [q for q in rs if q['request_id'] != r['request_id']]
            exposure = sum(max(0., min(z, q['last'])-max(f, q['admitted'])) for q in peers)
            spans = sorted((max(f, q['admitted']), min(z, q['first'])) for q in peers
                           if max(f, q['admitted']) < min(z, q['first']))
            union, end = 0., -math.inf
            for a, b in spans:
                union += max(0., b-max(a, end))
                end = max(end, b)
            output.append(dict(r, tpot_ms=(z-f)*1000/(r['tokens']-1),
                admission_to_first_ms=(f-r['admitted'])*1000,
                later_admissions_during_decode=sum(f <= q['admitted'] < z for q in peers),
                mean_other_admitted_requests=exposure/(z-f),
                peer_admission_to_first_envelope_s=union,
                wave_population=len(rs)))
    return output


def inspect_calibration_targets(output):
    """D348: reuse sealed C14 observations; never infer unsupported soft targets."""
    global OUT
    OUT = output.resolve()
    if OUT.exists():
        raise FileExistsError('calibration diagnosis is append-only')
    base = ROOT/'paper_results/ieee_tc/p2_backend'
    sources = {
        base/'d344_capacity_profiles/initialization/service.json':
            '794d00bac794da9c076b34da38cc310c90d70330b9414ef02de09faab9c7cd32',
        base/'d344_capacity_profiles/native_requests1.csv':
            '25c0e9591b30a4df4d8ccc3efce49f9d6aa4d6be27bc164d1c5d029e838c60bd',
        base/'d344_capacity_profiles/gaps1/native_requests1.csv':
            '6370581dc071e1df38745db79842ac697a4d94b110933e4af15ac5bf98224aeb',
        base/'d343_capacity_probe/cap14/analysis1/requests.csv':
            'eafe2ae05e5c2d9616b2276aa6c74b73bfdc8ea1b9f45255fc187d390f1082dc',
        ROOT/'results/ieee_tc/p2_backend_qualification/d344_20261008/gaps1/source1.json':
            '764509162dd0889b469e6150ae42aed8ca018f1899271d7b1789406e04508e4d',
        WARM:'0ebc7587effe293dcb986d33fb932c100cbcb3384c56ab2c0b0bec73b2bdd88c',
        PLAN:'0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7',
        PROTOCOL:'5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22',
    }
    for path, expected in sources.items():
        if sha256(path) != expected:
            raise ValueError('calibration source changed: '+str(path))
    profile = read(base/'d344_capacity_profiles/initialization/service.json')
    gap = read(ROOT/'results/ieee_tc/p2_backend_qualification/d344_20261008/gaps1/source1.json')
    gap_rows = {r['request_id']:r for r in gap['requests']}
    if len(gap_rows) != 40 or gap['pass'] is not True:
        raise ValueError('complete sealed gap run required')
    if len(profile['samples']) != 1545 or profile['model_config']['runtime_concurrency_cap'] != 14:
        raise ValueError('complete C14 profile required')
    csv_rows = {}
    for rel, source, count in (
            ('native_requests1.csv', '09d2f2b688c7cbce1591bb8b5ab4af8f28d3df1f26377b109b231ec2440c924e',1529),
            ('gaps1/native_requests1.csv', '764509162dd0889b469e6150ae42aed8ca018f1899271d7b1789406e04508e4d',40)):
        with (base/'d344_capacity_profiles'/rel).open(newline='') as f:
            rs = list(csv.DictReader(f))
        if len(rs) != count or len({r['request_id'] for r in rs}) != count:
            raise ValueError('complete sealed calibration CSV required')
        for r in rs:
            if r['role'] == 'representative_measurement':
                csv_rows[source,r['request_id']] = r
            elif r['role'] != 'kernel_warmup_retained':
                raise ValueError('unknown calibration role')
    if set(csv_rows) != {(s['source_run_sha256'],s['request_id']) for s in profile['samples']}:
        raise ValueError('profile and CSV measurement populations differ')
    rows = []
    for s in profile['samples']:
        c = csv_rows[s['source_run_sha256'],s['request_id']]
        if s['class_features']['tier'] != 'gpu':
            continue
        if (c['source'] != 'gpu' or s['correct'] is not True
                or s['admission_clock_id'] != s['native_clock_id']
                or s['admitted_monotonic_s'] != s['acquired_monotonic_s']
                or s['protected_at_admission'] is not True):
            raise ValueError('protected same-clock GPU measurements required')
        features = s['class_features']
        # The gap CSV lacks native prompt count: recover its own raw measurement,
        # not canonical prompt length or a neighboring request's observation.
        native_prompt = c.get('native_prompt_tokens')
        if native_prompt is None:
            original = gap_rows[s['request_id']]
            if (s['source_run_sha256'] != sources[ROOT/'results/ieee_tc/p2_backend_qualification/d344_20261008/gaps1/source1.json']
                    or original['source_request_id'] != c['source_request_id']
                    or original['class_features'] != features
                    or original['last_token_monotonic_s'] != s['last_token_monotonic_s']):
                raise ValueError('gap native prompt provenance differs')
            native_prompt = original['timing']['actual_prompt_tokens']
        prompt = float(native_prompt)
        if not math.isfinite(prompt) or int(prompt) != prompt or prompt < 1:
            raise ValueError('invalid measured native prompt count')
        row = dict(source_sha256=s['source_run_sha256'], request_id=s['request_id'],
            source_request_id=c['source_request_id'], clock_id=s['native_clock_id'],
            wave=int(c['wave']), round=int(c['round']), rank=features['adapter_rank'],
            admitted_count=features['admitted_after_accept'], prompt_tokens=int(prompt),
            tokens=s['native_output_tokens'], admitted=s['admitted_monotonic_s'],
            first=s['first_token_monotonic_s'], last=s['last_token_monotonic_s'])
        if (row['admitted_count'] != int(c['admitted_after_accept'])
                or abs((row['last']-row['first'])*1000/(row['tokens']-1)-float(c['native_tpot_ms'])) > 1):
            raise ValueError('profile native timing/count differs from sealed CSV')
        rows.append(row)
    if len(rows) != 333:
        raise ValueError('all333 GPU profile observations required')
    records = calibration_exposure(rows)
    thresholds = read(WARM)['warm_threshold_candidates']
    for r in records:
        group = [g for g in thresholds if
                 (g['lower_exclusive'] is None or r['prompt_tokens'] > g['lower_exclusive']) and
                 (g['upper_inclusive'] is None or r['prompt_tokens'] <= g['upper_inclusive'])]
        if len(group) != 1:
            raise ValueError('native prompt lacks unique common timing group')
        r['timing_tpot_miss'] = r['tpot_ms'] > group[0]['candidate_tpot_threshold_ms']
    groups = []
    for population in ('all','short_output_le24','other_output_gt24'):
        for admitted in range(1,15):
            rs = [r for r in records if r['admitted_count']==admitted and
                  (population=='all' or (r['tokens']<=24)==(population=='short_output_le24'))]
            groups.append(dict(population=population,admitted_count=admitted,count=len(rs),
                timing_tpot_misses=sum(r['timing_tpot_miss'] for r in rs) if rs else None,
                mean_tpot_ms=mean(r['tpot_ms'] for r in rs) if rs else None,
                mean_later_admissions=mean(r['later_admissions_during_decode'] for r in rs) if rs else None,
                mean_other_admitted_requests=mean(r['mean_other_admitted_requests'] for r in rs) if rs else None,
                mean_peer_admission_to_first_envelope_s=mean(r['peer_admission_to_first_envelope_s'] for r in rs) if rs else None))
    with (base/'d343_capacity_probe/cap14/analysis1/requests.csv').open(newline='') as f:
        stress = [r for r in csv.DictReader(f) if r['role']=='representative_measurement']
    if len(stress) != 66:
        raise ValueError('complete measured8/14stress contrast required')
    summary = dict(kind='d348_retained_C14_soft_target_identifiability_v1',
        gpu_profile_samples=len(records),matched_stress_samples=len(stress),
        stress_output_token_values=sorted({int(r['output_tokens']) for r in stress}),
        stress_native_prompt_values=sorted({float(r['native_prompt_tokens']) for r in stress}),
        profile_actual_admission_coverage=sorted({r['admitted_count'] for r in records}),
        proposed_latency_optimal_soft_target=None,service_curve_identified=False,
        reason='Admission count is not held constant through decode; stress contrast has only long outputs; no matched short-output steady-load curve.',
        next_contrast='Hold C14 runtime/profiles fixed; recover prior measured C8-control absolute soft targets as an explicit causal working-point contrast, not an inferred latency optimum.',
        n_correct=None,formal_SLO_qualified=False,new_GPU_runs=0,no_figures=True)
    OUT.mkdir(parents=True,exist_ok=False)
    table('gpu_profile_exposure.csv',records)
    table('admission_groups.csv',groups)
    write('summary.json',summary)
    write('manifest.json',dict(source_refs=[dict(path=str(p),sha256=s) for p,s in sources.items()],
        analysis_script_sha256=sha256(Path(__file__)),
        outputs={p.name:sha256(p) for p in OUT.iterdir()},no_figures=True))
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', choices=('d297', 'd299', 'd300', 'd303', 'd314', 'd315', 'd318', 'd323', 'd326', 'd332', 'd335', 'd336', 'd341', 'd346', 'd349', 'd353', 'd358', 'd364', 'd367'), default='d297')
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
    parser.add_argument('--lifecycle-output', type=Path,
        help='D319 retained D315/D318 lifecycle and decode analysis, no inference')
    parser.add_argument('--drain-lifecycle-output', type=Path,
        help='D327 retained D323/D326 lifecycle cohorts; same D319 method, no inference')
    parser.add_argument('--capacity-lifecycle-output', type=Path,
        help='D337 retained D335/D336 capacity/lifecycle cohorts; same D319 method, no inference')
    parser.add_argument('--c14-lifecycle-output', type=Path,
        help='D347 D336/D346 native-capacity comparison; reuse sealed D336 projection, no inference')
    parser.add_argument('--decoupled-lifecycle-output', type=Path,
        help='D350 D336/D349 measured load/prefill/startup overlap; no inference')
    parser.add_argument('--routing-load-output', type=Path,
        help='D320 retained D315/D318 routing load consistency, no inference')
    parser.add_argument('--routing-length-output', type=Path,
        help='D320 selected-cost length diagnosis using the sealed compact projections')
    parser.add_argument('--service-lifecycle-output', type=Path,
        help='D324 retained D318/D323 lifecycle and selected-service estimation check')
    parser.add_argument('--dispatch-gate-output', type=Path,
        help='D342 retained D336/D341 acquire-entry split, no inference')
    parser.add_argument('--dispatch-context-output', type=Path,
        help='D342 compact split data and conservative sampled permit bounds')
    parser.add_argument('--interleave-wait-output', type=Path,
        help='D359 retained D353/D358 stage and dispatch split; no inference')
    parser.add_argument('--preparation-wait-output',type=Path,
        help='D360 retained D353/D358 preparation/acquisition receipts; no inference')
    parser.add_argument('--calibration-target-output', type=Path,
        help='D348 retained C14 profile exposure; no steady-load inference or new GPU run')
    args = parser.parse_args()
    if sum(x is not None for x in (args.admission_wait_output, args.decode_tail_output,
                                  args.native_load_output, args.prefill_output, args.route_copy_output,
                                  args.lifecycle_output, args.routing_load_output, args.routing_length_output,
                                  args.service_lifecycle_output, args.drain_lifecycle_output,
                                  args.capacity_lifecycle_output, args.dispatch_gate_output,
                                  args.dispatch_context_output, args.c14_lifecycle_output,
                                  args.calibration_target_output, args.decoupled_lifecycle_output,
                                  args.interleave_wait_output,args.preparation_wait_output)) > 1:
        parser.error('choose one retained diagnostic')
    if args.preparation_wait_output is not None:
        inspect_retained_preparation(args.preparation_wait_output)
    elif args.interleave_wait_output is not None:
        inspect_retained_dispatch_gate(args.interleave_wait_output, contrast=('d353', 'd358'))
        inspect_retained_dispatch_context(args.interleave_wait_output/'sampled_capacity',
            prior=args.interleave_wait_output.resolve(), contrast=('d353', 'd358'))
    elif args.calibration_target_output is not None:
        inspect_calibration_targets(args.calibration_target_output)
    elif args.dispatch_context_output is not None:
        inspect_retained_dispatch_context(args.dispatch_context_output)
    elif args.dispatch_gate_output is not None:
        inspect_retained_dispatch_gate(args.dispatch_gate_output)
    elif args.capacity_lifecycle_output is not None:
        inspect_retained_lifecycle(args.capacity_lifecycle_output, contrast=('d335', 'd336'))
    elif args.c14_lifecycle_output is not None:
        inspect_retained_lifecycle(args.c14_lifecycle_output, contrast=('d336', 'd346'))
    elif args.decoupled_lifecycle_output is not None:
        inspect_retained_lifecycle(args.decoupled_lifecycle_output, contrast=('d336', 'd349'))
    elif args.drain_lifecycle_output is not None:
        inspect_retained_lifecycle(args.drain_lifecycle_output, contrast=('d323', 'd326'))
    elif args.service_lifecycle_output is not None:
        inspect_retained_service_lifecycle(args.service_lifecycle_output)
    elif args.routing_length_output is not None:
        inspect_retained_routing_lengths(args.routing_length_output)
    elif args.routing_load_output is not None:
        inspect_retained_routing_load(args.routing_load_output)
    elif args.lifecycle_output is not None:
        inspect_retained_lifecycle(args.lifecycle_output)
    elif args.route_copy_output is not None:
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
