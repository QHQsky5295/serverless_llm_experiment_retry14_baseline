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


def analyze(output):
    output = Path(output)
    if output.exists():
        raise ValueError('refusing to overwrite D455 stage evidence')
    comparison_path = ROOT/'paper_results/ieee_tc/main_comparison/d454_7b_w0/comparison.json'
    control_path = ROOT/'paper_results/ieee_tc/main_comparison/d454_7b_w0/control_diagnosis.json'
    assert sha256(comparison_path) == 'e8246c0203202f8d1e50adeac85a6569d134ea95729627da7a6a1df12b6c38ce'
    assert sha256(control_path) == '26003b8f4fbcb676a59b167f8f585e558631ec33444d7517796807f7e749b387'
    comparison, control = read(comparison_path), read(control_path)
    pinned = {r['path']:r['sha256'] for r in comparison['source_refs']}
    refs = [dict(path=str(path), sha256=sha256(path)) for path in (comparison_path, control_path)]

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
    assert set(reports) == {'PrimeLoRA', 'ElasticLocality'}
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
    with (output/'summary.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(output), delta=delta,
        all={name:r['all'] for name,r in reports.items()}, scope=result['scope']), indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', required=True, type=Path)
    analyze(parser.parse_args().output)
