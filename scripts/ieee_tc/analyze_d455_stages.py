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


def retained_control_reference(model):
    """Select an existing frozen reference, never infer thresholds from results."""
    references = {
        '7b': ('20261010_7b_slo_d394_frozen.json',
               '2d906f8aaf3fa1bf8cf30d116990db9970fb7be7bf1374fdbd1d8995090104f3',
               '20260927_7b_materialized_content_index.json'),
        '3b': ('20261017_3b_slo_d648_frozen.json',
               'd15123570c4a8d830515ca942d1db553c6717ffc1a40ade5c1546d35e2bed9d8',
               '20260927_3b_remote_content_index.json'),
    }
    if model not in references:
        raise ValueError('retained control model must be 7b or 3b')
    filename, digest, content = references[model]
    threshold_path = ROOT/'paper_results/ieee_tc/protocols'/filename
    if sha256(threshold_path) != digest:
        raise ValueError('frozen threshold identity differs')
    return threshold_path, ROOT/'paper_results/ieee_tc/inputs'/content


def retained_threshold_group(groups, prompt_tokens):
    if not math.isfinite(prompt_tokens) or prompt_tokens <= 0:
        raise ValueError('positive finite native prompt count required')
    matches = [g for g in groups
               if (g['lower_exclusive'] is None or prompt_tokens > g['lower_exclusive'])
               and (g['upper_inclusive'] is None or prompt_tokens <= g['upper_inclusive'])]
    if len(matches) != 1:
        raise ValueError('native prompt must match exactly one frozen threshold group')
    return matches[0]


def retained_control_paths(source, launch_path=None):
    """Select an explicit successful attempt, not the newest/best attempt."""
    launch_path = Path(launch_path) if launch_path is not None else source.parent/'launch.json'
    if launch_path.suffix != '.json':
        raise ValueError('explicit launch receipt must be a JSON path')
    physical_path = launch_path.with_suffix('.launch')/'physical_deployment/summary.json'
    return launch_path, physical_path, physical_path.parent/'main_outcome.json'


def analyze_retained_controls(output, sources, *, model='7b', launches=None, scenarios=None):
    """Reuse native/binding/stage helpers for explicit completed control runs.

    This is a descriptive audit, not a correctness certificate or a controlled
    causal comparison. Different compiler caches/configurations remain possible.
    """
    import hashlib
    from scripts.ieee_tc.plot_d290_diagnostic import binding_checks

    output = Path(output)
    if output.exists() or not sources or len(set(map(str, sources))) != len(sources):
        raise ValueError('new output and unique explicit retained sources required')
    if launches is not None and len(launches) != len(sources):
        raise ValueError('explicit launch receipts must match source count and order')
    if scenarios is not None and len(scenarios) != len(sources):
        raise ValueError('explicit scenarios must match source count and order')
    protocol_paths = [ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md',
                      ROOT/'docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md']
    if [sha256(x) for x in protocol_paths] != [
        '5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22',
        '193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d']:
        raise ValueError('frozen metric protocol identity differs')
    threshold_path, index_path = retained_control_reference(model)
    thresholds = read(threshold_path)['groups']
    content = {a['id']:hashlib.sha256(json.dumps(sorted(a['files'], key=lambda f:f['path']),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        for a in read(index_path)['artifacts']}
    refs = [dict(path=str(x), sha256=sha256(x)) for x in
            [*protocol_paths, threshold_path, index_path, Path(__file__).resolve()]]
    reports = {}
    for source_index, source in enumerate(map(Path, sources)):
        name = source.parent.parent.name
        if name in reports:
            raise ValueError('run directory identifiers must be distinct')
        launch_path, physical_path, outcome_path = retained_control_paths(
            source, None if launches is None else launches[source_index])
        launch, physical = read(launch_path), read(physical_path)
        if (not launch['pass'] or not physical['measurement_complete'] or physical['open_lease_ids']
                or any(physical[k] != 4000 for k in ('n_plan','n_terminal','n_native_contract_complete'))):
            raise ValueError('complete retained replay and resource lifecycle required')
        with outcome_path.open('rb') as stream:
            if next(ijson.items(stream, 'run_completed')) is not True:
                raise ValueError('main service outcome is incomplete')
        gpu_s = 0.0
        for intervals in physical['device_intervals'].values():
            previous_end = -math.inf
            for a, b in sorted(intervals):
                if not all(math.isfinite(x) for x in (a,b)) or not previous_end <= a <= b:
                    raise ValueError('invalid/disjoint physical ownership intervals')
                gpu_s += b-a
                previous_end = b
        if (abs(gpu_s-physical['gpu_seconds_observed']) > 1e-6
                or abs(gpu_s-sum(physical['window_gpu_seconds'].values())) > 1e-6):
            raise ValueError('physical lifecycle/window integral mismatch')
        for journal in physical['allocation_journals']:
            if not journal['released'] or sha256(Path(journal['path'])) != journal['sha256']:
                raise ValueError('allocation journal identity/release mismatch')
        if sha256(Path(physical['request_terminals_path'])) != physical['request_terminals_sha256']:
            raise ValueError('terminal journal identity mismatch')
        with source.open('rb') as stream:
            metadata = next(ijson.items(stream, 'metadata', use_float=True))
        from scripts.ieee_tc.check_d299_full_context import require_controller_scenario
        scenario = 'tc_ieee_full' if scenarios is None else scenarios[source_index]
        require_controller_scenario(metadata, scenario)
        rows, identities, counts, bindings = [], [], Counter(), Counter()
        seen, tier_counts = set(), Counter()
        external = launch['external_replay']
        with source.open('rb') as stream:
            for r in ijson.items(stream, f'detailed_results.{scenario}.requests.item', use_float=True):
                rid, n = r['request_id'], r['native_token_timing']
                if rid in seen or r['success'] is not True:
                    raise ValueError('unique successful requests required')
                seen.add(rid)
                row = partition_retained_request_stages(r,
                    arrival_start_s=external['replay_t0_s'], clock_id=external['clock_id'])
                checks = binding_checks(r, content[r['adapter_id']])
                if not all(checks.values()):
                    raise ValueError(f'control-plane binding failed: {rid}: {checks}')
                bindings.update({k:int(v) for k,v in checks.items()})
                count = r['completion_tokens']
                if (count != min(r['source_expected_output_tokens'],256)
                        or count != n['native_output_tokens'] or not n['native_terminal_observed']
                        or n['native_clock_id'] != physical['clock_id']
                        or r['generation_contract'] != 'fixed_length_greedy_v1'
                        or r['gpu_reference_evidence']['confirmed_dispatch_snapshot'] is not True):
                    raise ValueError('native token/generation/dispatch contract differs')
                tpot = ((n['native_last_token_monotonic_s']-n['native_first_token_monotonic_s'])*1000/(count-1)
                        if count >= 2 else None)
                if tpot is not None and (not math.isfinite(tpot) or tpot < 0 or abs(tpot-r['tpot_ms']) > 1):
                    raise ValueError('native TPOT reconstruction failed')
                group = retained_threshold_group(thresholds, n['actual_prompt_tokens'])
                tt = row['ttft_ms'] <= group['ttft_ms']
                tp = tpot is None or tpot <= group['tpot_ms']
                counts.update(N=1, ttft_pass=int(tt), tpot_pass_including_single_token=int(tp),
                    joint_timing_pass=int(tt and tp), single_token=int(count==1),
                    tpot_conditional_pass=int(tp and count>=2), tpot_conditional_N=int(count>=2))
                tier_counts[r['readiness_tier_before_dispatch']] += 1
                identities.append([rid, r['adapter_id'], r['scheduled_arrival_offset_s'],
                    r['canonical_prompt_sha256'], n['native_prompt_token_ids_sha256'],
                    n['actual_prompt_tokens'], count, r['generation_contract']])
                row.update(tpot_ms=tpot, source_offset_s=r['scheduled_arrival_offset_s'])
                rows.append(row)
        if len(rows) != 4000 or set(tier_counts)-{'gpu','host','nvme','remote'}:
            raise ValueError('complete 4000 dispatch tiers required')
        rows.sort(key=lambda r:(r['source_offset_s'],r['request_id']))
        identities.sort(key=lambda r:r[0])
        with outcome_path.open('rb') as stream:
            controls = list(ijson.items(stream, 'mechanism_events._ieee_control_events.item', use_float=True))
        if not controls:
            raise ValueError('retained control journal is absent')
        metrics = {key:timing_stats([r[key] for r in rows if r[key] is not None])
                   for key in ('ttft_ms','tpot_ms','e2e_ms')}
        for key in metrics:
            ordered = sorted(r[key] for r in rows if r[key] is not None)
            metrics[key]['p99_type1'] = ordered[math.ceil(.99*len(ordered))-1] if ordered else None
        reports[name] = dict(counts=dict(counts), metrics=metrics, scenario=scenario,
            input_contract_sha256=hashlib.sha256(json.dumps(identities,separators=(',',':')).encode()).hexdigest(),
            source_trace_sha256=metadata['shared_trace_sha256'],
            source_subset_sha256=metadata['shared_adapter_subset_sha256'],
            original_trace_role=metadata['trace_role'], original_formal_run=metadata['formal_run'],
            n_correct=physical['n_correct'], gpu_seconds_per_correct_request=None,
            physical_gpu_seconds=gpu_s, gpu_seconds_per_offered=gpu_s/4000,
            physical_windows=physical['window_gpu_seconds'], native_binding_checks=dict(bindings),
            dispatch_tier_counts=dict(tier_counts),
            max_stage_identity_error_ms=max(r['max_identity_error_ms'] for r in rows),
            mean_stages_ms={k:timing_stats([r['stages_ms'][k] for r in rows])['mean'] for k in rows[0]['stages_ms']},
            first100_ttft_ms=timing_stats([r['ttft_ms'] for r in rows[:100]]),
            first3_ttft_ms=[r['ttft_ms'] for r in rows[:3]],
            control_outcomes=dict(Counter(c['outcome'] for c in controls)),
            control_action_reasons={action:dict(Counter(c['reason'] for c in controls
                if c['action']==action and c['outcome'] in
                ('activation_scheduled','drained_replica_retired')))
                for action in ('scale_up','scale_down')},
            first_control_actions=[c for c in controls if c['outcome'] != 'no_action'][:12],
            suppressed_control_actions=[c for c in controls if c['outcome']=='no_action_current_demand_covered'],
            measurement_complete=True, independent_holdout=False, g1_g2_qualified=False)
        refs.extend(dict(path=str(x), sha256=sha256(x)) for x in
                    (source, launch_path, physical_path, outcome_path))
    report = dict(kind=f'retained_{model}_control_observation_v1', model=model,
        reports=reports, source_refs=refs,
        matched_request_inputs=len({r['input_contract_sha256'] for r in reports.values()})==1,
        causal_attribution_established=False, independent_numerical_qualification=False,
        new_performance_runs=0, confidence_interval=None,
        caveats=['Binding witnesses do not certify numerical LoRA arithmetic.',
            'GPU-s/offered is diagnostic; unknown N_correct is not replaced by 4000.',
            'Matched input does not establish identical compiler caches or sole-feature execution differences.',
            'Timing attainment is not correctness-qualified joint SLO.',
            'No single-configuration cross-scenario optimum or G2 budget qualification.'])
    output.mkdir(parents=True,exist_ok=False)
    with (output/'summary.json').open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(output), matched_request_inputs=report['matched_request_inputs'],
        reports={k:{f:r[f] for f in ('counts','metrics','physical_gpu_seconds','control_outcomes')}
                 for k,r in reports.items()}),indent=2))
    return report


def control_busy_time_bounds(controls, requests):
    """Retrospective lower bound, not an exact occupancy or alternative policy.

    In the TP=1 retained runs, confirmed admission through the native last token
    is inside the owned runtime reservation. Merge by native owner and clip at
    routing withdrawal: draining owners are excluded by the sampled controller.
    Pre-admission reservation and post-token cleanup are intentionally absent.
    """
    from collections import defaultdict
    if not controls or not requests:
        raise ValueError('nonempty controls and requests required')
    previous = -math.inf
    withdrawals = {}
    for c in controls:
        now = c['observed_at']
        if not math.isfinite(now) or now <= previous:
            raise ValueError('control observations must be finite and increasing')
        previous = now
        if c['action'] == 'scale_down':
            replica = c['target_replica']
            if c['outcome'] != 'drained_replica_retired' or replica in withdrawals:
                raise ValueError('complete unique withdrawal required')
            withdrawals[replica] = now
    intervals, seen = defaultdict(list), set()
    for r in requests:
        a, b = float(r['admitted']), float(r['last'])
        if (r['request_id'] in seen or not r['owner'] or not r['replica']
                or not all(math.isfinite(x) for x in (a, b)) or b < a):
            raise ValueError('invalid or duplicate admitted request interval')
        seen.add(r['request_id'])
        b = min(b, withdrawals.get(r['replica'], math.inf))
        if b > a:
            intervals[r['owner']].append((a, b))
    merged = []
    for spans in intervals.values():
        union = []
        for a, b in sorted(spans):
            if union and a <= union[-1][1]:
                union[-1] = (union[-1][0], max(b, union[-1][1]))
            else:
                union.append((a, b))
        merged.extend(union)
    windows = []
    for c in controls:
        observation = c['predictive_observation']
        if observation is None:
            continue
        elapsed, sampled = (observation['history_elapsed_s'],
                            observation['busy_instance_seconds'])
        if (not all(math.isfinite(x) for x in (elapsed, sampled))
                or elapsed <= 0 or sampled < 0):
            raise ValueError('invalid sampled work window')
        now = c['observed_at']
        start = now - elapsed
        lower = sum(max(0., min(b, now) - max(a, start)) for a, b in merged)
        windows.append(dict(observed_at=now, history_elapsed_s=elapsed,
            sampled_busy_instance_seconds=sampled, owned_busy_lower_bound_s=lower,
            sampled_to_lower_bound=sampled/lower if lower > 0 else None,
            action=c['action'], reason=c['reason'],
            predictive_target_instances=c['predictive_target_instances']))
    ratios = sorted(w['sampled_to_lower_bound'] for w in windows
                    if w['sampled_to_lower_bound'] is not None)
    releases = []
    for row in windows:
        if row['action'] != 'scale_down':
            continue
        following = [c['observed_at'] for c in controls
                     if c['action'] == 'scale_up' and c['observed_at'] > row['observed_at']]
        releases.append(dict(row, next_scale_up_after_s=(
            min(following)-row['observed_at'] if following else None)))
    return dict(request_count=len(requests), control_count=len(controls),
        window_count=len(windows), owner_union_intervals=len(merged),
        windows_below_owned_lower_bound=sum(w['sampled_busy_instance_seconds'] + 1e-6
            < w['owned_busy_lower_bound_s'] for w in windows),
        windows_below_90pct_of_lower_bound=sum(x < .9 for x in ratios),
        ratio_quantiles={str(p): ratios[math.ceil(p*len(ratios))-1] if ratios else None
                         for p in (.05, .5, .95)},
        scale_down_with_score_above_one=sum(c['action'] == 'scale_down' and c['score'] > 1
                                           for c in controls),
        overload_release_events=[c for c in controls
                                 if c['action'] == 'scale_down' and c['score'] > 1],
        releases=releases)


def checked_retained_source(source, expected_sha256=None):
    """Read a sealed source, or its explicitly receipted byte-identical gzip.

    No restoration or source substitution; changed plain files do not fall back
    to an archive. Preserve both original content and current storage identities.
    """
    import gzip
    from scripts.ieee_tc.compress_closed_main_outcomes import decompressed_digest
    source = Path(source)
    if source.exists():
        digest = sha256(source)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError('sealed source identity changed')
        return (lambda: source.open('rb')), [dict(path=str(source), sha256=digest)]
    receipt_path = (source.parents[2]/'compression_receipt.json'
                    if source.name == 'main_outcome.json'
                    else source.with_name(source.name+'.compression_receipt.json'))
    receipt = read(receipt_path)
    archive = source.with_name(source.name+'.gz')
    if (receipt['source'] != str(source) or receipt['archive'] != str(archive)
            or receipt['removed_uncompressed_duplicate'] is not True
            or sha256(archive) != receipt['archive_sha256']
            or decompressed_digest(archive) != receipt['source_sha256']
            or (expected_sha256 is not None and receipt['source_sha256'] != expected_sha256)):
        raise ValueError('compressed sealed source identity changed')
    return (lambda: gzip.open(archive, 'rb')), [
        dict(path=str(archive), sha256=receipt['archive_sha256'],
             original_path=str(source), original_sha256=receipt['source_sha256']),
        dict(path=str(receipt_path), sha256=sha256(receipt_path))]


def ownership_dwell(device_intervals, start, end):
    """Exact physical-card count dwell, reusing D454's interval-edge method."""
    if not all(math.isfinite(x) for x in (start, end)) or end <= start:
        raise ValueError('positive finite observation window required')
    edges = Counter({start: 0, end: 0})
    for intervals in device_intervals.values():
        previous = -math.inf
        for a, b in sorted(intervals):
            if not all(math.isfinite(x) for x in (a, b)) or not previous <= a <= b:
                raise ValueError('disjoint finite device intervals required')
            previous = b
            a, b = max(a, start), min(b, end)
            if a < b:
                edges[a] += 1
                edges[b] -= 1
    dwell, active = Counter(), 0
    stamps = sorted(edges)
    for i, stamp in enumerate(stamps[:-1]):
        active += edges[stamp]
        if not 0 <= active <= len(device_intervals):
            raise ValueError('physical ownership count invalid')
        dwell[active] += stamps[i+1]-stamp
    return dict(sorted(dwell.items()))


def owner_request_partition(start, end, requests):
    """Partition a TP=1 lease around admitted-to-last-token interval unions.

    The complement includes preparation and other real work: it is NOT idle
    GPU time or a counterfactual resource-saving estimate.
    """
    if not all(math.isfinite(x) for x in (start, end)) or end < start:
        raise ValueError('finite closed lease required')
    merged = []
    for a, b in sorted(requests):
        if not all(math.isfinite(x) for x in (a, b)) or not start <= a <= b <= end:
            raise ValueError('request interval lies outside its physical lease')
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    if not merged:
        return dict(before_first_admission_s=0., admitted_request_union_s=0.,
            between_request_gaps_s=0., after_last_token_s=0., unused_owner_s=end-start)
    busy = sum(b-a for a, b in merged)
    return dict(before_first_admission_s=merged[0][0]-start,
        admitted_request_union_s=busy,
        between_request_gaps_s=merged[-1][1]-merged[0][0]-busy,
        after_last_token_s=end-merged[-1][1], unused_owner_s=0.)


def audit_retained_budget(output, comparison_path):
    """Describe a sealed 7B W1/W2 result; no controller simulation or new run."""
    from collections import defaultdict
    from scripts.ieee_tc.check_d299_full_context import closed_physical_integral
    output, comparison_path = Path(output), Path(comparison_path)
    if output.exists():
        raise ValueError('refusing to overwrite retained budget evidence')
    comparison = read(comparison_path)
    refs = dict(comparison['source_refs'])
    reviews = [Path(path) for path in refs if Path(path).name == 'evidence_review.json']
    if len(reviews) != 1 or sha256(reviews[0]) != refs[str(reviews[0])]:
        raise ValueError('one SHA-bound existing evidence review required')
    for ref in read(reviews[0])['source_refs']:
        if ref['path'] in refs and refs[ref['path']] != ref['sha256']:
            raise ValueError('conflicting retained source identity')
        refs[ref['path']] = ref['sha256']
    sources = [Path(path) for path in refs if Path(path).name.startswith('experiment_results_')
               and comparison['prime']['run'].lower() in str(path).lower()]
    if len(sources) != 1:
        raise ValueError('one sealed PrimeLoRA request source required')
    source = sources[0]
    opener, source_refs = checked_retained_source(source, refs[str(source)])
    launch_path, physical_path, outcome_path = retained_control_paths(source)
    launch, physical = read(launch_path), read(physical_path)
    for path in (launch_path, physical_path):
        if sha256(path) != refs[str(path)]:
            raise ValueError('sealed launch or physical source changed')
    total = closed_physical_integral(physical)
    if not launch['pass'] or abs(total-comparison['prime']['physical_gpu_seconds']) > 1e-6:
        raise ValueError('completed comparison/lifecycle mismatch')
    external = launch['external_replay']
    start = external['replay_t0_s']
    owners, native_owners, offsets, seen = defaultdict(list), defaultdict(set), [], set()
    with opener() as stream:
        for r in ijson.items(stream, 'detailed_results.tc_ieee_full.requests.item', use_float=True):
            n = r['native_token_timing']
            if (r['request_id'] in seen or not r['success'] or
                    n['native_clock_id'] != physical['clock_id']):
                raise ValueError('duplicate/failed request or clock mismatch')
            seen.add(r['request_id'])
            # Adapter-reference owner IDs and physical lease IDs are different
            # namespaces. Join the actual worker PID, checked against its closed
            # native-worker lease and request timestamps below.
            pid = r['gpu_reference_evidence']['receipt']['worker_pid']
            native_owners[pid].add(n['gpu_reference_owner_id'])
            owners[pid].append((
                n['controller_admitted_monotonic_s'], n['native_last_token_monotonic_s']))
            offsets.append(r['scheduled_arrival_offset_s'])
    if len(seen) != 4000:
        raise ValueError('complete retained request set required')
    end = start + max(offsets)
    dwell = ownership_dwell(physical['device_intervals'], start, end)
    if abs(sum(k*v for k,v in dwell.items())-physical['window_gpu_seconds']['arrival']) > 1e-6:
        raise ValueError('arrival dwell integral mismatch')
    leases, partition, worker_pids = [], Counter(), set()
    for journal in physical['allocation_journals']:
        path = Path(journal['path'])
        if sha256(path) != journal['sha256'] or not journal['released']:
            raise ValueError('allocation journal changed or unclosed')
        events = [json.loads(line) for line in path.read_text().splitlines()]
        acquire = [e for e in events if e['event'] == 'acquire']
        release = [e for e in events if e['event'] == 'release']
        if len(acquire) != 1 or len(release) != 1 or len(acquire[0]['gpu_uuids']) != 1:
            raise ValueError('one closed TP=1 lease per journal required')
        a, b, owner = acquire[0]['at'], release[0]['at'], acquire[0]['owner_id']
        native = [e for e in events if e['event'] == 'native_workers']
        if len(native) != 1 or len(native[0]['workers']) != 1:
            raise ValueError('one native TP=1 worker binding required')
        worker = native[0]['workers'][0]
        pid = worker['pid']
        if pid in worker_pids or worker['device_uuid'] != acquire[0]['gpu_uuids'][0]:
            raise ValueError('ambiguous worker PID or physical device binding')
        worker_pids.add(pid)
        part = owner_request_partition(a, b, owners.pop(pid, []))
        partition.update(part)
        leases.append(dict(owner=owner, worker_pid=pid, native_reference_owners=sorted(native_owners[pid]),
            start_offset_s=a-start, end_offset_s=b-start,
            physical_gpu_seconds=b-a, request_partition=part))
        source_refs.append(dict(path=str(path), sha256=journal['sha256']))
    if owners or abs(sum(partition.values())-total) > 1e-6:
        raise ValueError('unbound request owner or overlapping owner leases')
    outcome_opener, outcome_refs = checked_retained_source(outcome_path)
    with outcome_opener() as stream:
        controls = list(ijson.items(stream, 'mechanism_events._ieee_control_events.item', use_float=True))
    if not controls or any(a['observed_at'] >= b['observed_at'] for a,b in zip(controls, controls[1:])):
        raise ValueError('ordered retained control records required')
    actions = [c for c in controls if c['outcome'] in ('activation_scheduled','drained_replica_retired')]
    releases = []
    for i, c in enumerate(actions):
        if c['action'] != 'scale_down':
            continue
        nxt = next((r for r in actions[i+1:] if r['action']=='scale_up'), None)
        releases.append(dict(offset_s=c['observed_at']-start, reason=c['reason'],
            ready_instances=c['ready_instances'], active_requests=c['active_requests'],
            ready_capacity=c['ready_capacity'], predictive_target_instances=c['predictive_target_instances'],
            queue_depth=c['queue_depth'], score=c['score'],
            next_scale_up_after_s=nxt['observed_at']-c['observed_at'] if nxt else None,
            next_scale_up_reason=nxt['reason'] if nxt else None))
    reference_path = ROOT/'paper_results/ieee_tc/main_comparison/d594_reference_and_ttft/resident_references.json'
    if sha256(reference_path) != 'a9b6b25fe4ec8a17921dd5968c4fb494b7a6d795c0d4a11c7b7d306ac06bc60c':
        raise ValueError('frozen 7B resource reference changed')
    reference = read(reference_path)[external['plan']['profile']]
    source_refs.extend(outcome_refs)
    source_refs.extend(dict(path=str(path), sha256=sha256(path)) for path in
        (launch_path, physical_path, comparison_path, reviews[0], reference_path, Path(__file__).resolve()))
    report = dict(kind='retained_budget_ownership_description_v1', run=comparison['prime']['run'],
        new_performance_runs=0, physical_gpu_seconds=total, budget_gpu_seconds=reference['B_gpu_seconds'],
        budget_excess_gpu_seconds=total-reference['B_gpu_seconds'],
        physical_windows=physical['window_gpu_seconds'], arrival_duration_s=end-start,
        arrival_ownership_dwell_s=dwell, arrival_mean_physical_gpus=physical['window_gpu_seconds']['arrival']/(end-start),
        owner_request_partition_s=dict(partition), leases=sorted(leases,key=lambda l:l['start_offset_s']),
        control_outcomes=dict(Counter(c['outcome'] for c in controls)), releases=releases,
        source_refs=source_refs, independent_numerical_proof=False, causal_attribution_established=False,
        limitations=['Admitted-to-last-token union is not GPU utilization or a compute lower bound.',
            'Its complement includes preparation, planned spare capacity, drain and cleanup; not all avoidable.',
            'Rapid release/reactivation is observed, not proof of a cheaper feasible counterfactual.',
            'Four-card dwell cannot be subtracted to predict a three-replica execution.'])
    output.mkdir(parents=True, exist_ok=False)
    with (output/'summary.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('leases','source_refs')},indent=2))
    return report


def audit_control_busy_windows(output, stages):
    """Reuse sealed D716 request tables; never rewrite the performance results."""
    import csv
    output = Path(output)
    if output.exists() or not stages or len(set(map(str, stages))) != len(stages):
        raise ValueError('new output and unique stage reports required')
    refs, reports = [], {}
    for path in map(Path, stages):
        stage = read(path)
        if stage['passed'] is not True or stage['counts']['n'] != 4000:
            raise ValueError('complete passed stage report required')
        table = Path(stage['request_table']['path'])
        if sha256(table) != stage['request_table']['sha256']:
            raise ValueError('sealed request table changed')
        raw = Path(stage['source_run'])
        name = raw.parent.name
        if name in reports:
            raise ValueError('ambiguous run identity')
        source_refs = [r for r in stage['source_refs']
                       if Path(r['path']).name.startswith('experiment_results_')]
        if len(source_refs) != 1:
            raise ValueError('stage request source identity changed')
        _, request_refs = checked_retained_source(
            source_refs[0]['path'], source_refs[0]['sha256'])
        outcome = raw/'launch.launch/physical_deployment/main_outcome.json'
        opener, outcome_refs = checked_retained_source(outcome)
        with opener() as stream:
            if next(ijson.items(stream, 'run_completed')) is not True:
                raise ValueError('control outcome is incomplete')
        with opener() as stream:
            controls = list(ijson.items(stream, 'mechanism_events._ieee_control_events.item', use_float=True))
        with table.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 4000:
            raise ValueError('full request table required')
        reports[name] = control_busy_time_bounds(controls, rows)
        refs.extend(dict(path=str(p), sha256=sha256(p)) for p in (path, table))
        refs.extend(outcome_refs + request_refs)
    refs.append(dict(path=str(Path(__file__).resolve()), sha256=sha256(Path(__file__))))
    report = dict(kind='retained_control_busy_lower_bound_v1', reports=reports,
        source_refs=refs, new_performance_runs=0, causal_attribution_established=False,
        limitations=['Retrospective TP=1 native-owner union; not exact active-slot accounting.',
            'Reservation before confirmed admission and cleanup after last token are excluded.',
            'Window overlap is not independent replication; no CI or counterfactual SLO.',
            '90% is descriptive error reporting, not an online threshold or acceptance gate.',
            'No replay of another controller policy and no changed performance metric.'])
    output.mkdir(parents=True, exist_ok=False)
    with (output/'summary.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(output), reports=reports), indent=2))
    return report


def retained_nested_overhead(request):
    """Read existing nested spans; do not turn them into additive CPU costs.

    The historical routing timer omits early-continue snapshot attempts. RPC
    pickup is a same-host wall-clock estimate, not an exclusive scheduling span.
    Missing observations are errors rather than zero-cost operations.
    """
    native = request['native_token_timing']
    if native['parent_rpc_transport'] != 'native_async_socket_v1':
        raise ValueError('retained overhead requires the measured native transport')
    result = {key.removesuffix('_us') + '_ms': request[key] / 1000
              for key in ('routing_decision_us', 'control_path_total_us')
              if type(request[key]) in (int, float)}
    if len(result) != 2:
        raise ValueError('complete numeric routing observations required')
    result.update({key: request[key] for key in
                   ('lora_io_ms', 'dispatch_admission_wait_ms', 'dispatch_window_wait_ms')})
    result.update({key: native[key] for key in (
        'parent_rpc_overhead_ms', 'parent_rpc_channel_acquire_ms',
        'parent_rpc_response_pickup_delay_ms', 'parent_rpc_send_flush_ms',
        'parent_rpc_thread_resume_delay_ms', 'worker_rpc_queue_ms')})
    result['native_load_and_acquire_ms'] = request['gpu_reference_evidence']['receipt']['load_and_acquire_ms']
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for value in result.values()):
        raise ValueError('complete finite nonnegative nested observations required')
    return result


def analyze_retained_overhead(output, comparisons):
    """Stream the sources of sealed comparisons, reusing the stage partition.

    Pure R1 analysis: no replay, no new workloads, no source/result edits. This
    locates observed waiting; composite ablations do not isolate a cause.
    """
    import csv
    import hashlib

    output = Path(output)
    if output.exists() or not comparisons:
        raise ValueError('explicit sealed comparisons and a new output required')
    pinned, refs = {}, []
    for path in map(Path, comparisons):
        comparison = read(path)
        refs.append(dict(path=str(path), sha256=sha256(path)))
        for ref in comparison['source_refs']:
            if ref['path'] in pinned and pinned[ref['path']] != ref['sha256']:
                raise ValueError('conflicting sealed source identities')
            pinned[ref['path']] = ref['sha256']

    def checked(path):
        path = Path(path)
        if sha256(path) != pinned[str(path)]:
            raise ValueError('sealed comparison source changed')
        refs.append(dict(path=str(path), sha256=pinned[str(path)]))
        return path

    summaries = sorted(Path(path) for path in pinned if Path(path).name == 'full4000.json')
    reports, rows_csv, input_hashes, plans = {}, [], set(), []
    for summary_path in summaries:
        summary = read(checked(summary_path))
        if not summary['passed'] or summary['counts']['n'] != 4000 or not summary['full_replay']:
            raise ValueError('complete previously qualified replay required')
        run = Path(summary['source_run'])
        name = run.name
        if name in reports:
            raise ValueError('duplicate retained run')
        scenario = summary.get('scenario', 'tc_ieee_full')
        sources = [Path(path) for path in pinned
                   if Path(path).parent == run and Path(path).name.startswith('experiment_results_full')]
        if len(sources) != 1:
            raise ValueError('one explicit retained request source per run required')
        source = checked(sources[0])
        launch = read(checked(run/'launch.json'))
        if launch['pass'] is not True:
            raise ValueError('successful resource closure required')
        external = launch['external_replay']
        rows, identities = [], []
        with source.open('rb') as stream:
            for r in ijson.items(stream, f'detailed_results.{scenario}.requests.item', use_float=True):
                row = partition_retained_request_stages(r,
                    arrival_start_s=external['replay_t0_s'], clock_id=external['clock_id'])
                row.update(nested_ms=retained_nested_overhead(r),
                    output_tokens=r['completion_tokens'], tier=r['readiness_tier_before_dispatch'],
                    arrival_offset_s=r['scheduled_arrival_offset_s'])
                if row['output_tokens'] != r['requested_completion_tokens']:
                    raise ValueError('retained output contract changed')
                identities.append([r['request_id'], r['adapter_id'], r['scheduled_arrival_offset_s'],
                    r['canonical_prompt_sha256'], r['native_token_timing']['native_prompt_token_ids_sha256'],
                    r['requested_completion_tokens'], r['completion_tokens'], r['generation_contract']])
                rows.append(row)
        if len(rows) != len({r['request_id'] for r in rows}) or len(rows) != 4000:
            raise ValueError('4000 unique request observations required')
        rows.sort(key=lambda r: (r['arrival_offset_s'], r['request_id']))
        identities.sort(key=lambda r: r[0])
        input_hash = hashlib.sha256(json.dumps(identities, separators=(',', ':')).encode()).hexdigest()
        input_hashes.add(input_hash)
        for metric in ('ttft_ms', 'e2e_ms'):
            measured = timing_stats([r[metric] for r in rows])
            if any(abs(measured[k]-summary['metrics'][metric][k]) > 1e-6
                   for k in ('mean', 'p95_type1')):
                raise ValueError('retained metrics differ from sealed summary')
        groups = [('all', rows)] + [(f'phase_{i+1}', rows[i*500:(i+1)*500]) for i in range(8)]
        groups += [('output_le32', [r for r in rows if r['output_tokens'] <= 32]),
                   ('output_gt32', [r for r in rows if r['output_tokens'] > 32])]
        report_groups = {}
        for group, subset in groups:
            group_summary = dict(count=len(subset))
            for family in ('stages_ms', 'nested_ms'):
                group_summary[family] = {}
                for field in rows[0][family]:
                    values = [r[family][field] for r in subset]
                    stats = timing_stats(values) if values else dict(count=0, mean=None, p50_type1=None, p95_type1=None)
                    group_summary[family][field] = stats
                    rows_csv.append(dict(run=name, group=group, family=family, field=field, **stats))
            report_groups[group] = group_summary
        outcome = run/'launch.launch/physical_deployment/main_outcome.json'
        opener, outcome_refs = checked_retained_source(outcome, pinned[str(outcome)])
        refs.extend(outcome_refs)
        with opener() as stream:
            events = list(ijson.items(stream, 'planning_cpu_events.item', use_float=True))
        with opener() as stream:
            if next(ijson.items(stream, 'run_completed')) is not True:
                raise ValueError('retained service outcome is incomplete')
        planning = {}
        for label, begin, end in (
            ('capture_serialization_ms', 'captured_monotonic_s', 'detached_monotonic_s'),
            ('serialized_owner_wait_ms', 'detached_monotonic_s', 'submitted_monotonic_s'),
            ('worker_dispatch_ms', 'submitted_monotonic_s', 'worker_started_monotonic_s'),
            ('worker_wall_ms', 'worker_started_monotonic_s', 'worker_finished_monotonic_s'),
            ('worker_result_pickup_ms', 'worker_finished_monotonic_s', 'joined_monotonic_s')):
            values = [(event[end]-event[begin])*1000 for event in events]
            if any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError('invalid retained planning interval')
            planning[label] = dict(timing_stats(values), sum=math.fsum(values)) if values else None
        values = [event['worker_cpu_seconds']*1000 for event in events]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('invalid retained planning CPU duration')
        planning['worker_cpu_ms'] = dict(timing_stats(values), sum=math.fsum(values)) if values else None
        reports[name] = dict(input_contract_sha256=input_hash, scenario=scenario,
            sealed_metrics=summary['metrics'], observed_joint_fraction=summary['joint_slo_fraction'],
            physical_gpu_seconds=summary['physical_gpu_seconds'], n_correct=summary['n_correct'],
            max_partition_error_ms=max(r['max_identity_error_ms'] for r in rows),
            groups=report_groups, planning_event_count=len(events), planning=planning,
            planning_states=dict(Counter(e['state'] for e in events)),
            planning_operations=dict(Counter(e['operation'] for e in events)))
        plans.append(summary['replay_plan'])
    if len(reports) < 2 or len(input_hashes) != 1 or any(plan != plans[0] for plan in plans):
        raise ValueError('at least two complete matched-input retained runs required')
    report = dict(kind='retained_nested_overhead_d1064_v1', reports=reports,
        matched_request_inputs=True, new_gpu_runs=0, formal_ranking_eligible=False,
        causal_attribution_established=False, confidence_interval=None, source_refs=refs,
        scope=p.scope_identity(p.cg_path().name),
        script=dict(path=str(Path(__file__).resolve()), sha256=sha256(Path(__file__).resolve())),
        caveats=[
            'Nested spans are not additive E2E components or exclusive CPU durations.',
            'Historical routing timers omit early-continue attempts; smaller values do not prove cheaper routing.',
            'Composite ablations change decisions and execution work; matched inputs do not fix closed-loop paths.',
            'Planning worker CPU and wall times overlap requests and cannot be subtracted from their latency.',
            'RPC pickup uses same-host wall clocks and producer clamping; it is not an exclusive event-loop span.',
            'Unknown numerical correctness is preserved; output length and binding alone do not fill N_correct.',
            'One known trace per configuration, no independent holdout or run-level uncertainty.'])
    output.mkdir(parents=True, exist_ok=False)
    with (output/'summary.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    with (output/'stages_and_nested.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_csv[0]))
        writer.writeheader()
        writer.writerows(rows_csv)
    print(json.dumps(dict(output=str(output), runs=list(reports), new_gpu_runs=0)))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--include-d458', action='store_true')
    parser.add_argument('--owner-decode', action='store_true')
    parser.add_argument('--retained-control-source', action='append', type=Path)
    parser.add_argument('--retained-control-launch', action='append', type=Path,
                        help='explicit successful receipts in the same order as sources')
    parser.add_argument('--retained-control-model', choices=('7b', '3b'), default='7b')
    parser.add_argument('--retained-control-scenario', action='append',
                        help='explicit recorded scenario for each retained source, in order')
    parser.add_argument('--control-busy-stage', action='append', type=Path)
    parser.add_argument('--overhead-comparison', action='append', type=Path)
    parser.add_argument('--budget-comparison', type=Path)
    args = parser.parse_args()
    if args.retained_control_scenario and not args.retained_control_source:
        parser.error('--retained-control-scenario requires explicit retained sources')
    if args.budget_comparison:
        if (args.overhead_comparison or args.control_busy_stage or args.retained_control_source
                or args.retained_control_launch or args.include_d458 or args.owner_decode):
            parser.error('budget analysis requires one sealed comparison only')
        audit_retained_budget(args.output, args.budget_comparison)
    elif args.overhead_comparison:
        if (args.control_busy_stage or args.retained_control_source or args.retained_control_launch
                or args.include_d458 or args.owner_decode):
            parser.error('nested overhead analysis requires its explicit comparison sources only')
        analyze_retained_overhead(args.output, args.overhead_comparison)
    elif args.control_busy_stage:
        if args.retained_control_source or args.retained_control_launch or args.include_d458 or args.owner_decode:
            parser.error('busy-stage analysis cannot be mixed with performance comparison modes')
        audit_control_busy_windows(args.output, args.control_busy_stage)
    elif args.retained_control_source:
        if args.include_d458 or args.owner_decode:
            parser.error('explicit retained sources cannot be mixed with registered legacy modes')
        analyze_retained_controls(args.output,args.retained_control_source,
            model=args.retained_control_model, launches=args.retained_control_launch,
            scenarios=args.retained_control_scenario)
    elif args.owner_decode:
        if not args.include_d458:parser.error('--owner-decode requires --include-d458')
        analyze_owner_decode(args.output)
    else:
        analyze(args.output, include_d458=args.include_d458)
