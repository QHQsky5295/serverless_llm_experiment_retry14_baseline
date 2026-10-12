"""D455: reuse sealed D454 native requests to locate Full's pre-engine overhead."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
import ijson

from scripts.analyze_control_path_overhead import partition_retained_request_stages
from scripts.ieee_tc.check_d299_full_context import ROOT, read, sha256
from scripts.ieee_tc.audit_d402_serverlessllm_path import timing_stats
from scripts import ieee_tc_preflight as p
from faaslora.datasets.workload_generator import FrozenReplayPlan


def with_d458_candidate(comparison, candidate):
    """Join a sealed candidate, without replacing any of the six prior rows."""
    if candidate['previous_six_system_results'] != comparison['results']:
        raise ValueError('candidate changed a prior result')
    if candidate['reference'] != comparison['reference']:
        raise ValueError('candidate changed the common reference')
    if candidate['matched_contract_sha256'] != comparison['matched_contract_sha256']:
        raise ValueError('candidate input contract differs')
    refs = {r['path']:r['sha256'] for r in comparison['source_refs']}
    for row in candidate['refs']:
        if row['path'] in refs and refs[row['path']] != row['sha256']:
            raise ValueError('conflicting source identity')
        refs[row['path']] = row['sha256']
    sources = [r for r in candidate['refs'] if 'd458_20261012' in r['path']
               and 'experiment_results_full' in Path(r['path']).name]
    if len(sources) != 1:
        raise ValueError('exactly one D458 request source required')
    return dict(comparison, results=dict(comparison['results'],
        PrimeLoRA_D458=candidate['candidate']),
        source_refs=[dict(path=path, sha256=digest) for path,digest in refs.items()])


def analyze(output, *, include_d458=False):
    output = Path(output)
    if output.exists():
        raise ValueError('refusing to overwrite D455 stage evidence')
    comparison_path = ROOT/'paper_results/ieee_tc/main_comparison/d454_7b_w0/comparison.json'
    control_path = ROOT/'paper_results/ieee_tc/main_comparison/d454_7b_w0/control_diagnosis.json'
    assert sha256(comparison_path) == 'e8246c0203202f8d1e50adeac85a6569d134ea95729627da7a6a1df12b6c38ce'
    assert sha256(control_path) == '26003b8f4fbcb676a59b167f8f585e558631ec33444d7517796807f7e749b387'
    comparison, control = read(comparison_path), read(control_path)
    candidate_path = ROOT/'paper_results/ieee_tc/p2_backend/d458_full_w0/matched_comparison.json'
    if include_d458:
        assert sha256(candidate_path) == 'd8188014ac448205e33198f81e9bb12a793dea664a1de90e5c504535ea500a42'
        comparison = with_d458_candidate(comparison, read(candidate_path))
    pinned = {r['path']:r['sha256'] for r in comparison['source_refs']}
    refs = [dict(path=str(path), sha256=sha256(path)) for path in (comparison_path, control_path)]
    if include_d458:
        refs.append(dict(path=str(candidate_path), sha256=sha256(candidate_path)))

    def checked(path):
        assert sha256(path) == pinned[str(path)], path
        refs.append(dict(path=str(path), sha256=pinned[str(path)]))
        return path

    reports, all_rows = {}, {}
    for ref in comparison['source_refs']:
        source = Path(ref['path'])
        if 'experiment_results_full' not in source.name:
            continue
        if 'd371_20261010' in str(source):
            name, scenario = 'PrimeLoRA', 'tc_ieee_full'
        elif 'd453_20261012' in str(source):
            name, scenario = 'ElasticLocality', 'tc_ieee_elastic_locality'
        elif include_d458 and 'd458_20261012' in str(source):
            name, scenario = 'PrimeLoRA_D458', 'tc_ieee_full'
        else:
            raise ValueError('unexpected controller source')
        checked(source)
        launch = read(checked(source.parent/'launch.json'))
        start = launch['external_replay']['replay_t0_s']
        clock = launch['external_replay']['clock_id']
        plan = FrozenReplayPlan.load(launch['external_replay']['plan']['source_path'])
        assert plan.identity() == launch['external_replay']['plan']
        order = {entry.request_id:i for i,entry in enumerate(plan.entries)}
        threshold = control['controller_settings']['ttft_lower_ms']
        rows = []
        with source.open('rb') as stream:
            for r in ijson.items(stream, f'detailed_results.{scenario}.requests.item', use_float=True):
                row = partition_retained_request_stages(r, arrival_start_s=start, clock_id=clock)
                e = r['gpu_reference_evidence']
                receipt = e['receipt']
                extra = {key:r[key] for key in ('lora_io_ms', 'dispatch_admission_wait_ms',
                                               'service_ttft_ms', 'parent_rpc_overhead_ms')}
                extra['native_load_and_acquire_ms'] = receipt['load_and_acquire_ms']
                extra['routing_ms'] = row['nested_routing_ms']
                row.update(system=name, adapter_id=r['adapter_id'], instance_id=r['instance_id'],
                    source_order=order[r['request_id']], phase=order[r['request_id']]//500,
                    dispatch_tier=r['readiness_tier_before_dispatch'],
                    ttft_not_low=row['ttft_ms'] >= threshold,
                    native_load_invoked=receipt['native_load_invoked'],
                    prior_routing_attempts=len(e['prior_routing_attempts']), nested_ms=extra)
                rows.append(row)
        assert len(rows) == len({r['request_id'] for r in rows}) == len(order) == 4000
        rows.sort(key=lambda r:r['source_order'])
        assert [r['request_id'] for r in rows] == [entry.request_id for entry in plan.entries]

        def summary(group):
            if not group:
                return dict(count=0, stages_ms={}, nested_ms={})
            return dict(count=len(group),
                stages_ms={key:timing_stats([r['stages_ms'][key] for r in group])
                           for key in group[0]['stages_ms']},
                nested_ms={key:timing_stats([r['nested_ms'][key] for r in group])
                           for key in group[0]['nested_ms']},
                ttft_ms=timing_stats([r['ttft_ms'] for r in group]),
                e2e_ms=timing_stats([r['e2e_ms'] for r in group]),
                dispatch_tiers=dict(Counter(r['dispatch_tier'] for r in group)),
                native_load_invoked=sum(r['native_load_invoked'] is True for r in group),
                prior_routing_attempts=sum(r['prior_routing_attempts'] for r in group))

        all_summary = summary(rows)
        for key in ('ttft_ms', 'e2e_ms'):
            for statistic in ('mean', 'p95_type1'):
                assert abs(all_summary[key][statistic]-comparison['results'][name]['metrics'][key][statistic]) < 1e-6
        assert abs(math.fsum(r['mean'] for r in all_summary['stages_ms'].values())-
                   all_summary['e2e_ms']['mean']) < 1e-6
        reports[name] = dict(all=all_summary,
            phases={str(i):summary([r for r in rows if r['phase']==i]) for i in range(8)},
            by_tier={tier:summary([r for r in rows if r['dispatch_tier']==tier])
                     for tier in ('gpu','host','nvme','remote')},
            ttft_not_low=summary([r for r in rows if r['ttft_not_low']]),
            max_identity_error_ms=max(r['max_identity_error_ms'] for r in rows))
        all_rows[name] = rows
    assert set(reports) == ({'PrimeLoRA', 'ElasticLocality', 'PrimeLoRA_D458'}
                            if include_d458 else {'PrimeLoRA', 'ElasticLocality'})
    delta = {key:reports['PrimeLoRA']['all']['stages_ms'][key]['mean']-
                 reports['ElasticLocality']['all']['stages_ms'][key]['mean']
             for key in reports['PrimeLoRA']['all']['stages_ms']}
    output.mkdir(parents=True, exist_ok=False)
    with (output/'requests.jsonl').open('x') as handle:
        for rows in all_rows.values():
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False)+'\n')
    result = dict(kind='d455_matched_retained_request_stage_partition_v1', reports=reports,
        PrimeLoRA_minus_ElasticLocality_mean_stage_ms=delta,
        controller_ttft_lower_ms=threshold, source_refs=refs + [
            dict(path=str(Path(__file__).resolve()), sha256=sha256(Path(__file__))),
            dict(path=str(ROOT/'scripts/analyze_control_path_overhead.py'),
                 sha256=sha256(ROOT/'scripts/analyze_control_path_overhead.py'))],
        request_projection=dict(path=str((output/'requests.jsonl').resolve()),
                                sha256=sha256(output/'requests.jsonl')),
        scope=p.scope_identity(p.cg_path().name), causal_attribution_established=False,
        caveats=['All means use complete4000; stage quantiles are not additive.',
                 'Routing/LoRA load are nested observations, not extra E2E phases or pure CPU time.',
                 'TTFT-not-low and tier groups differ between systems; they are diagnostic, not controlled populations.',
                 'Controller samples observe completed requests later; per-request threshold membership does not reconstruct the exact feedback window.',
                 'No counterfactual latency, independent numerical proof, new performance run or confidence interval.'])
    if include_d458:
        result['kind'] = 'd459_d458_matched_retained_request_stage_partition_v1'
        result['candidate_minus_reference_mean_stage_ms'] = {
            name:{key:reports['PrimeLoRA_D458']['all']['stages_ms'][key]['mean']-
                      reports[name]['all']['stages_ms'][key]['mean']
                  for key in reports[name]['all']['stages_ms']}
            for name in ('PrimeLoRA', 'ElasticLocality')}
    with (output/'summary.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(output), delta=delta,
        all={name:r['all'] for name,r in reports.items()}, scope=result['scope']), indent=2))
    return result


def analyze_owner_decode(output):
    """D460 reuse the validated native interval analyzers on three sealed runs."""
    import gzip
    from collections import defaultdict
    from scripts.ieee_tc.analyze_d297_full import (partition_activation_readiness,
        decode_overlap_summary, native_load_overlap_summary, prefill_overlap_summary,
        proactive_load_overlap_summary)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    previous = ROOT/'paper_results/ieee_tc/main_comparison/d459_d458_diagnosis/stages.json'
    assert sha256(previous)=='a1c7f0a7b84ac928a6e0466f0b506d98477ebfd45cbaaea138c3882da757cf04'
    seal = read(previous)
    pinned = {r['path']:r['sha256'] for r in seal['source_refs']}
    # Outcome/archive identities were independently pinned in D459 planning.
    planning = ROOT/'paper_results/ieee_tc/main_comparison/d459_d458_diagnosis/planning.json'
    assert sha256(planning)=='45dbd1e3b973ad758aa0fb186ba06a3e677aae16be84373bb05e782d5ba13997'
    for ref in read(planning)['source_refs']:
        if ref['path'] in pinned and pinned[ref['path']] != ref['sha256']:
            raise ValueError('conflicting retained source identity')
        pinned[ref['path']] = ref['sha256']
    comparison_path=ROOT/'paper_results/ieee_tc/main_comparison/d454_7b_w0/comparison.json'
    candidate_path=ROOT/'paper_results/ieee_tc/p2_backend/d458_full_w0/matched_comparison.json'
    assert sha256(comparison_path)=='e8246c0203202f8d1e50adeac85a6569d134ea95729627da7a6a1df12b6c38ce'
    assert sha256(candidate_path)=='d8188014ac448205e33198f81e9bb12a793dea664a1de90e5c504535ea500a42'
    comparison=with_d458_candidate(read(comparison_path),read(candidate_path))
    for ref in comparison['source_refs']:
        if ref['path'] in pinned and pinned[ref['path']] != ref['sha256']:
            raise ValueError('conflicting matched source identity')
        pinned[ref['path']] = ref['sha256']
    refs = []
    def checked(path, digest=None):
        path = Path(path)
        actual = sha256(path)
        if actual != (digest if digest is not None else pinned[str(path)]):
            raise ValueError('source hash mismatch: '+str(path))
        refs.append(dict(path=str(path),sha256=actual))
        return path
    for path in (previous,planning,comparison_path,candidate_path):
        refs.append(dict(path=str(path),sha256=sha256(path)))
    threshold_path=ROOT/'paper_results/ieee_tc/protocols/20261010_7b_slo_d394_frozen.json'
    threshold=read(checked(threshold_path,'2d906f8aaf3fa1bf8cf30d116990db9970fb7be7bf1374fdbd1d8995090104f3'))
    thresholds=[dict(g,tpot_s=g['tpot_ms']/1000) for g in threshold['groups']]
    summaries, all_rows = {}, {}
    for source_text in pinned:
        source=Path(source_text)
        if 'experiment_results_full' not in source.name:
            continue
        if 'd371_20261010' in source_text:
            name,scenario='PrimeLoRA','tc_ieee_full'
        elif 'd453_20261012' in source_text:
            name,scenario='ElasticLocality','tc_ieee_elastic_locality'
        elif 'd458_20261012' in source_text:
            name,scenario='PrimeLoRA_D458','tc_ieee_full'
        else:
            raise ValueError('unexpected retained request source')
        checked(source)
        launch=read(checked(source.parent/'launch.json'))
        start=launch['external_replay']['replay_t0_s']; clock=launch['external_replay']['clock_id']
        requests=[]
        with source.open('rb') as stream:
            for r in ijson.items(stream,f'detailed_results.{scenario}.requests.item',use_float=True):
                row={k:r[k] for k in ('request_id','instance_id','success','completion_tokens',
                    'readiness_tier_before_dispatch','tpot_ms','native_token_timing','adapter_id')}
                row['gpu_reference_evidence']=dict(receipt=r['gpu_reference_evidence']['receipt'])
                requests.append(row)
        assert len(requests)==4000
        outcome=source.parent/'launch.launch/physical_deployment/main_outcome.json'
        if name=='PrimeLoRA':
            receipt=read(checked(source.parent/'compression_receipt.json'))
            archive=checked(receipt['archive'])
            opener=lambda:gzip.open(archive,'rb')
        else:
            checked(outcome);opener=lambda:outcome.open('rb')
        events={}
        for key in ('_ieee_activations','_ieee_gpu_preparation_plans','_ieee_native_host_preparations'):
            events[key]=[]
            with opener() as stream:
                for row in ijson.items(stream,'mechanism_events.'+key+'.item',use_float=True):
                    if key != '_ieee_activations':
                        row={k:row[k] for k in ('plan_id','target_replica','trigger_reason','attempts','owner_id') if k in row}
                        projected=[]
                        for attempt in row['attempts']:
                            item={k:attempt[k] for k in ('attempt_id','state')}
                            if 'receipt' in attempt:
                                fields=('owner_id','worker_pid','worker_rank','clock_id','acquired','held',
                                    'lease_id','native_load_invoked','acquired_monotonic_s','completion_fence_ms',
                                    'native_load_started_monotonic_s','native_load_completed_monotonic_s',
                                    'load_started_monotonic_s','load_completed_monotonic_s')
                                item['receipt']={k:attempt['receipt'][k] for k in fields if k in attempt['receipt']}
                            projected.append(item)
                        row['attempts']=projected
                    events[key].append(row)
        ready,unready=partition_activation_readiness(events['_ieee_activations'],requests,clock_id=clock)
        decode=decode_overlap_summary(requests,ready,clock_id=clock,arrival_start=start,thresholds=thresholds)
        native=native_load_overlap_summary(requests,decode.pop('requests'),clock_id=clock)
        prefill=prefill_overlap_summary(requests,native['requests'],clock_id=clock)
        proactive=proactive_load_overlap_summary(native.pop('requests'),
            events['_ieee_gpu_preparation_plans'],events['_ieee_native_host_preparations'],
            events['_ieee_activations'],clock_id=clock)
        rows=proactive.pop('requests'); pmap={r['request_id']:r for r in prefill.pop('requests')}
        for r in rows:
            for k,v in pmap[r['request_id']].items():
                if k in r and r[k]!=v:raise ValueError('request projections differ')
                r[k]=v
        # Native PID-to-physical GPU census from the already closed allocation journals.
        physical_path=source.parent/'launch.launch/physical_deployment/summary.json'
        physical=read(checked(physical_path))
        pid_gpus=defaultdict(set); pid_starts=defaultdict(set)
        for entry in physical['allocation_journals']:
            assert entry['released'] is True
            journal=checked(entry['path'],entry['sha256'])
            for line in journal.read_text().splitlines():
                event=json.loads(line)
                assert event['clock_id']==clock
                for device in event.get('census',{}).get('devices',[]):
                    for proc in device['processes']:
                        if proc['kind']=='compute' and proc['service_member']:
                            pid_gpus[proc['pid']].add(device['gpu_uuid'])
                            pid_starts[proc['pid']].add(proc['identity']['start_ticks'])
        for owner,record in proactive['native_owners'].items():
            pid=record['worker_pid']
            if len(pid_gpus[pid])!=1 or len(pid_starts[pid])!=1:
                raise ValueError('native owner requires unambiguous TP1 physical census')
            record.update(gpu_uuid=next(iter(pid_gpus[pid])),start_ticks=next(iter(pid_starts[pid])))
        for r in rows:
            r['gpu_uuid']=proactive['native_owners'][r['owner']]['gpu_uuid']
        def stats(rs):
            return dict(count=len(rs),tpot_ms=timing_stats([r['tpot_ms'] for r in rs]) if rs else None,
                mean_other_native_requests=sum(r['mean_other_native_requests'] for r in rs)/len(rs) if rs else None,
                tpot_misses=sum(r['timing_tpot_miss'] for r in rs),
                mean_decode_ms=sum(r['decode_s']*1000 for r in rs)/len(rs) if rs else None)
        counts={k:sum(r[k] for r in rows) for k in ('startup_overlap','load_overlap','prefill_overlap',
            'proactive_any_load_overlap','timing_tpot_miss')}
        flagkeys=('startup_overlap','load_overlap','prefill_overlap','proactive_any_load_overlap')
        cross={str(mask):stats([r for r in rows if sum(int(r[k])<<i for i,k in enumerate(flagkeys))==mask])
               for mask in range(16)}
        all_stats=stats(rows)
        assert abs(all_stats['mean_decode_ms']-seal['reports'][name]['all']['stages_ms']['native_decode_ms']['mean'])<1e-6
        summaries[name]=dict(all=all_stats,counts=counts,decode=decode,native=native,
            prefill=prefill,proactive=proactive,cross_flags=list(flagkeys),cross_groups=cross,
            ready_activations=len(ready),unready_activations=unready)
        all_rows[name]=rows
        with (output/(name+'_requests.jsonl')).open('x') as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(dict(system=name,all=all_stats,counts=counts,proactive_counts=proactive['counts'])),flush=True)
    assert set(summaries)=={'PrimeLoRA','ElasticLocality','PrimeLoRA_D458'}
    projections=[dict(path=str(x),sha256=sha256(x)) for x in sorted(output.glob('*requests.jsonl'))]
    result=dict(kind='d460_same_owner_retained_decode_diagnosis_v1',reports=summaries,
        source_refs=refs+[dict(path=str(Path(__file__)),sha256=sha256(Path(__file__))),
            dict(path=str(ROOT/'scripts/ieee_tc/analyze_d297_full.py'),sha256=sha256(ROOT/'scripts/ieee_tc/analyze_d297_full.py'))],
        request_projections=projections,scope=p.scope_identity(p.cg_path().name),
        threshold_manifest_sha256=sha256(threshold_path),causal_attribution_established=False,
        new_performance_runs=0,no_figures=True,
        threshold_note='D394 frozen common TPOT thresholds replace legacy D170 provisional caveat in reused helper output; not a new joint-SLO/correctness qualification.')
    with (output/'summary.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--include-d458', action='store_true')
    parser.add_argument('--owner-decode', action='store_true')
    args = parser.parse_args()
    if args.owner_decode:
        if not args.include_d458:parser.error('--owner-decode requires --include-d458')
        analyze_owner_decode(args.output)
    else:
        analyze(args.output, include_d458=args.include_d458)
