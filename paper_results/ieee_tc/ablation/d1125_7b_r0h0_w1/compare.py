"""D1125: reuse D1124 paired audit plus D1056 four-cell contrast formulas.

Reuses D1123/D1050 without modifying their sealed scripts/results.
No serving or new framework: D299 supplies native-contract checks, D389 supplies
first-service semantics, and the actual Router recomputes recorded decisions.
Full is the existing D1115 main-table source, not a new favorable Full replay.
"""
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import ijson
import yaml

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
BASE = Path(__file__).resolve().parent
OUT = BASE/'comparison'
sys.path.insert(0, str(ROOT))
from scripts import ieee_tc_preflight as p
from scripts.ieee_tc.check_d299_full_context import (
    matched_controller_rows, matched_request_contract, require_controller_scenario)
from scripts.ieee_tc.audit_d389_scaleup_handoff_labels import activation_first_services
from faaslora.experiment.instance_pool import (
    Router, ReplicaRoutingSnapshot, ServiceObservationClass, ServiceComponents)

REFS = {}


def witness(path, expected=None):
    path = Path(path)
    actual = p.digest(path)
    if expected is not None and actual != expected:
        raise ValueError(f'changed source: {path}')
    REFS[str(path)] = actual
    return path


def read(path, expected=None):
    return json.loads(witness(path, expected).read_text())


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')



def four_cell_observation(current):
    """Reuse D1056 formulas, binding only the sealed current 7B observations."""
    cells = {'Y11': current['results']['Full'], 'Y00': current['results']['R0H0']}
    prior_sources = {}
    for cell, name, variant, closure_sha in (
        ('Y10', 'd1124_7b_no_handoff_w1', 'NoHandoff',
         '1fef63e63ee495b539d1f56e83b16edaadbab425447bd91c4beb05c12d5e451f'),
        ('Y01', 'd1123_7b_no_routing_w1', 'NoRouting',
         'e07ab8b53e8d4478da839d218eb3b3c13ae7c947299dfb35c4596b7cd0817d91')):
        directory = ROOT/'paper_results/ieee_tc/ablation'/name
        closure_path = directory/'closure_manifest.json'
        closure = read(closure_path, closure_sha)
        source = directory/'comparison/comparison.json'
        previous = read(source, closure['outputs']['comparison/comparison.json'])
        assert previous['same_non_target_configuration'] is True
        assert previous['matched_inputs'] == 4000
        assert previous['plan_sha256'] == current['plan_sha256']
        for key in ('run', 'workload', 'replay_plan', 'counts', 'physical_gpu_seconds',
                    'metrics', 'joint_slo', 'common_slo_sha256'):
            assert previous['results']['Full'][key] == cells['Y11'][key], key
        for protocol in ('METRIC_PROTOCOL_FROZEN_V1.md', 'METRIC_PROTOCOL_SINGLE_RUN_V2.md'):
            path = str(ROOT/'docs/ieee_tc'/protocol)
            assert previous['source_refs'][path] == current['source_refs'][path]
        previous_raw = {name: sha for name, sha in previous['source_refs'].items()
                        if Path(name).name.startswith('experiment_results_full')}
        current_raw = {name: sha for name, sha in current['source_refs'].items()
                       if Path(name).name.startswith('experiment_results_full')}
        common = previous_raw.keys() & current_raw.keys()
        assert len(common) == 1
        assert all(previous_raw[name] == current_raw[name] for name in common)
        assert next(iter(common)).endswith('_d1115_7b_same_version_w1.json')
        cells[cell] = previous['results'][variant]
        assert cells[cell]['replay_plan'] == cells['Y11']['replay_plan']
        assert cells[cell]['common_slo_sha256'] == cells['Y11']['common_slo_sha256']
        prior_sources[str(source)] = REFS[str(source)]
        prior_sources[str(closure_path)] = REFS[str(closure_path)]
    budget_path = ROOT/'paper_results/ieee_tc/main_comparison/d594_reference_and_ttft/resident_references.json'
    budget = read(budget_path,
        'a9b6b25fe4ec8a17921dd5968c4fb494b7a6d795c0d4a11c7b7d306ac06bc60c')['W1']['B_gpu_seconds']
    definitions = (
        ('physical_gpu_seconds', 'GPU-s', 'lower', lambda r: r['physical_gpu_seconds']),
        ('gpu_seconds_per_offered_request', 'GPU-s/offered-request', 'lower',
         lambda r: r['gpu_seconds_per_offered_request']),
        ('joint_slo_percent', 'percentage points', 'higher', lambda r: 100*r['joint_slo']),
        ('mean_ttft_ms', 'ms', 'lower', lambda r: r['metrics']['ttft_ms']['mean']),
        ('p95_ttft_ms', 'ms', 'lower', lambda r: r['metrics']['ttft_ms']['p95_type1']),
        ('mean_tpot_ms', 'ms', 'lower', lambda r: r['metrics']['tpot_ms']['mean']),
        ('p95_tpot_ms', 'ms', 'lower', lambda r: r['metrics']['tpot_ms']['p95_type1']),
        ('mean_e2e_ms', 'ms', 'lower', lambda r: r['metrics']['e2e_ms']['mean']),
    )
    rows = []
    for name, unit, direction, getter in definitions:
        values = {cell: getter(row) for cell, row in cells.items()}
        sign = 1 if direction == 'lower' else -1
        routing_without = sign*(values['Y00']-values['Y10'])
        routing_with = sign*(values['Y01']-values['Y11'])
        handoff_without = sign*(values['Y00']-values['Y01'])
        handoff_with = sign*(values['Y10']-values['Y11'])
        gamma = sign*(values['Y10']+values['Y01']-values['Y00']-values['Y11'])
        assert abs((routing_with-routing_without)-gamma) < 1e-8
        assert abs((handoff_with-handoff_without)-gamma) < 1e-8
        rows.append(dict(metric=name, unit=unit, better=direction, **values,
            routing_benefit_without_handoff=routing_without,
            routing_benefit_with_handoff=routing_with,
            handoff_benefit_without_routing=handoff_without,
            handoff_benefit_with_routing=handoff_with, gamma=gamma))
    return dict(kind='d1125_7b_routing_handoff_four_cell_descriptive_v1',
        cell_runs={key: dict(run=row['run'], variant=row['variant'],
            timing_feasible=row['timing_feasible'],
            within_budget=row['physical_gpu_seconds'] <= budget,
            physical_gpu_seconds=row['physical_gpu_seconds'],
            n_correct=row['n_correct']) for key, row in cells.items()},
        budget_gpu_seconds=budget, metrics=rows, prior_sources=prior_sources,
        current_pair_file='comparison.json', matched_input_requests_per_pair=4000,
        confidence_interval=None, observed_runs_per_cell=1,
        independent_holdout=False, independent_numerical_proof=False,
        formal_ranking_eligible=False, n_correct=None,
        formula_lower='Y10 + Y01 - Y00 - Y11',
        formula_higher='Y00 + Y11 - Y10 - Y01',
        interpretation='Positive Gamma is a descriptive complementary contrast, not proof of significant or causal synergy; signed conditional effects must be shown.',
        caveats=[
            'Natural closed-loop cache, resource and activation trajectories are not fixed.',
            'One known source42 observation per cell; no request-level pseudo-replication or CI.',
            'Quantile contrasts are not additive request-stage decompositions.',
            'Same-backend output digest changes and unverified independent numerical correctness remain unresolved.',
            'All four cells retain max4; do not borrow cap3 resource costs for a G2 claim.',
            'Controlled same-plan A2/A4 and information-view A3/S2 remain separate experiments.'])

def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    group = p.cg_path()
    assert os.sched_getaffinity(0) == {2, 3, 26, 27}
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert (group/'memory.max').read_text().strip() == str(4*p.GIB)
    assert (group/'memory.swap.max').read_text().strip() == '0'
    assert p.check_plan() == 'd7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83'
    configs = [yaml.safe_load(witness(ROOT/'configs/ieee_tc'/name, sha).read_text())
        for name, sha in (
            ('20261107_d1100_7b_busy_service_rate_w2.yaml',
             '386a34c84ec70e5d0d366f6d3bcaf0891536be503ba1359d583a93bd5ecf303f'),
            ('20261108_d1125_7b_r0h0_w1.yaml',
             '8213f9e60b61335d38ac85410ab819c8430a1cdae68df9051ea2bbf701e99bac'))]
    bin_ms = configs[0]['resource_coordination']['service_bin_ms']
    for cfg in configs:
        del cfg['experiment']['output_dir']
        del cfg['storage']['nvme_cache_dir']
        del cfg['storage']['host_cache_dir']
    assert configs[0]['resource_coordination']['min_instances'] == 1
    assert configs[1]['resource_coordination']['min_instances'] == 1
    assert all(c['resource_coordination']['max_instances'] == 4 for c in configs)
    assert configs[0]['resource_coordination']['ieee_handoff_policy'] == 'full'
    assert configs[1]['resource_coordination']['ieee_handoff_policy'] == 'no_handoff'
    configs[1]['resource_coordination']['ieee_handoff_policy'] = 'full'
    assert configs[1]['resource_coordination'].pop('ieee_selection_strategy') == 'load_only'
    assert configs[0] == configs[1]
    index = read(ROOT/'paper_results/ieee_tc/inputs/20260927_7b_materialized_content_index.json')
    content = {a['id']: hashlib.sha256(json.dumps(sorted(a['files'], key=lambda f:f['path']),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest() for a in index['artifacts']}
    results, contracts, mechanisms, launches, output_differences = {}, {}, {}, {}, []
    for variant, run_id, scenario, directory in (
        ('Full', 'D1115', 'tc_ieee_full', ROOT/'paper_results/ieee_tc/main_comparison/d1115_same_version_w1/result1'),
        ('R0H0', 'D1125', 'tc_ieee_full', BASE/'result1')):
        metric = read(directory/'reconciliation.json')['prime']
        review = read(directory/'evidence_review.json')
        assert metric['run'] == run_id and metric['workload'] == review['workload_profile'] == 'W1'
        assert metric['counts']['joint_pass'] == review['counts']['joint_pass']
        assert metric['physical_gpu_seconds'] == review['physical_gpu_seconds']
        source = witness(review['source'], review['source_sha256'])
        launch = read(source.parent/'launch.json')
        assert launch['pass'] and launch['native_gpu_context_release_confirmed']
        launches[variant] = launch
        rows = matched_controller_rows(source, scenario, content)
        for row in rows.values():
            row['arrival_offset_s'] = row['arrival']-launch['external_replay']['replay_t0_s']
        contracts[variant] = rows
        with source.open('rb') as f:
            metadata = next(ijson.items(f, 'metadata', use_float=True))
        coordination = require_controller_scenario(metadata, scenario)
        result = dict(metric, variant=variant, scenario=scenario,
            joint_slo=review['timing_joint_rate'], timing_feasible=review['timing_feasible'],
            remote=review['remote'], resource=review['resource'], tier_counts=review['tier_counts'])
        results[variant] = result
        dispatches, routing_rows = [], []
        expected_strategy = 'service_bins' if variant == 'Full' else 'load_only'
        with source.open('rb') as f:
            for r in ijson.items(f, f'detailed_results.{scenario}.requests.item', use_float=True):
                routing = r['gpu_reference_evidence']['routing_snapshot']
                contract = routing['selection_contract']
                assert contract['owner_policy'] == 'ieee_confirmed'
                assert contract['strategy'] == expected_strategy
                assert contract['locality_load_slack'] is None
                candidates = tuple(ReplicaRoutingSnapshot(**(v | {
                    'active_adapters': frozenset(v['active_adapters']),
                    'service_class': ServiceObservationClass(**v['service_class']) if v['service_class'] else None,
                    'service': ServiceComponents(**v['service']) if v['service'] else None,
                })) for v in routing['candidates'])
                assert candidates and all(v.request_id == r['request_id'] and v.adapter_id == r['adapter_id'] for v in candidates)
                full_choice = Router.select_ieee_snapshot(candidates, bin_ms)
                load_choice = Router.select_load_snapshot(candidates)
                actual = full_choice if variant == 'Full' else load_choice
                assert actual is not None and full_choice is not None and load_choice is not None
                assert actual.replica_id == routing['selected_replica_id'] == r['instance_id']
                routing_rows.append(dict(request_id=r['request_id'], candidate_count=len(candidates),
                    feasible_count=sum(v.feasible for v in candidates), actual=actual.replica_id,
                    full_choice=full_choice.replica_id, load_choice=load_choice.replica_id,
                    differs=full_choice.replica_id != load_choice.replica_id,
                    actual_tier=actual.service_class.tier, full_tier=full_choice.service_class.tier,
                    load_tier=load_choice.service_class.tier,
                    full_estimated_service_ms=full_choice.service.total_ms,
                    load_estimated_service_ms=load_choice.service.total_ms))
                n, e = r['native_token_timing'], r['external_arrival_timing']
                dispatches.append(dict(request_id=r['request_id'], instance_id=r['instance_id'],
                    adapter_id=r['adapter_id'], readiness_tier_before_dispatch=r['readiness_tier_before_dispatch'],
                    clock_id=n['native_clock_id'], dispatch_monotonic_s=n['native_dispatch_monotonic_s'],
                    first_token_monotonic_s=n['native_first_token_monotonic_s'],
                    arrival_monotonic_s=e['planned_arrival_s'], success=r['success']))
        assert len(routing_rows) == 4000
        outcome = witness(source.parent/'launch.launch/physical_deployment/main_outcome.json')
        with outcome.open('rb') as f:
            activations = list(ijson.items(f, 'mechanism_events._ieee_activations.item', use_float=True))
        mechanism = dict(activations=activations,
            first_services=activation_first_services(activations, dispatches, clock_id=launch['external_replay']['clock_id']),
            routing=dict(strategy=expected_strategy, validated_requests=4000,
                same_state_choice_disagreements=sum(x['differs'] for x in routing_rows),
                multiple_feasible_requests=sum(x['feasible_count'] > 1 for x in routing_rows),
                rows=routing_rows, counterfactual_latency_generated=False), plans={})
        for key in ('_ieee_file_preparation_plans', '_ieee_gpu_preparation_plans',
                    '_ieee_native_host_preparations', '_ieee_residency_epochs'):
            count, reasons, states, decision_reasons, evaluated = 0, Counter(), Counter(), Counter(), 0
            handoff_records = []
            with outcome.open('rb') as f:
                for plan in ijson.items(f, 'mechanism_events.'+key+'.item', use_float=True):
                    count += 1
                    reasons[plan.get('trigger_reason', 'not_recorded')] += 1
                    states[plan.get('state', 'not_recorded')] += 1
                    if plan.get('trigger_reason') == 'handoff':
                        handoff_records.append({k:plan[k] for k in (
                            'plan_id','target_replica','activation_id','trigger_reason',
                            'objective_sha256','planning_sha256','state','started_at','finished_at',
                            'targets','registration','results','attempts') if k in plan})
                    if key == '_ieee_gpu_preparation_plans':
                        assert plan['capacity_only'] is False
                        for attempt in plan['attempts']:
                            receipt = attempt.get('receipt')
                            if receipt and receipt.get('proactive_admission_evaluated') is True:
                                evaluated += 1
                                decision_reasons[receipt['admission']['reason']] += 1
            mechanism['plans'][key] = dict(count=count, reasons=dict(reasons), states=dict(states),
                evaluated_admission_attempts=evaluated, admission_decision_reasons=dict(decision_reasons),
                handoff_records=handoff_records)
        assert activations
        if variant == 'R0H0':
            assert all(a['policy'] == 'no_handoff' and a['plan_sha256'] is None
                and a['preparation_state'] == 'disabled' for a in activations)
            assert all(not entry['handoff_records'] for entry in mechanism['plans'].values())
        else:
            assert all(a['policy'] == 'full' for a in activations)
        physical = read(source.parent/'launch.launch/physical_deployment/summary.json')
        with source.open('rb') as f:
            scale_up = list(ijson.items(f, f'detailed_results.{scenario}.scale_up_events.item', use_float=True))
        with source.open('rb') as f:
            scale_down = next(ijson.items(f, f'detailed_results.{scenario}.scale_down_events', use_float=True))
        with source.open('rb') as f:
            down_log = list(ijson.items(f, f'detailed_results.{scenario}.scale_down_event_log.item', use_float=True))
        assert scale_down == len(down_log)
        mechanism['control'] = dict(scale_up_events=scale_up, scale_down_events=scale_down,
            scale_down_event_log=down_log, physical_allocations=len(physical['allocation_journals']))
        mechanisms[variant] = mechanism
        # Actual recorded configuration, not the live legacy TTFT-only scoreboard.
        mechanism['feature_gates'] = coordination['feature_gates']
    assert launches['Full']['external_replay']['plan'] == launches['R0H0']['external_replay']['plan']
    assert launches['Full']['preflight']['policy'] == launches['R0H0']['preflight']['policy']
    assert contracts['Full'].keys() == contracts['R0H0'].keys()
    for rid, a in contracts['Full'].items():
        b = contracts['R0H0'][rid]
        matched_request_contract(a, b)
        if a['native']['completion_token_ids_sha256'] != b['native']['completion_token_ids_sha256']:
            output_differences.append(dict(request_id=rid,
                Full=a['native']['completion_token_ids_sha256'], R0H0=b['native']['completion_token_ids_sha256']))
    full, elastic = results['Full'], results['R0H0']
    relative = dict(gpu_seconds_reduction=(elastic['physical_gpu_seconds']-full['physical_gpu_seconds'])/elastic['physical_gpu_seconds'],
        joint_slo_difference_percentage_points=100*(full['joint_slo']-elastic['joint_slo']),
        p95_ttft_reduction=(elastic['metrics']['ttft_ms']['p95_type1']-full['metrics']['ttft_ms']['p95_type1'])/elastic['metrics']['ttft_ms']['p95_type1'])
    for path in (Path(__file__), ROOT/'scripts/ieee_tc/check_d299_full_context.py',
                 ROOT/'scripts/ieee_tc/audit_d389_scaleup_handoff_labels.py',
                 ROOT/'faaslora/experiment/instance_pool.py',
                 ROOT/'paper_results/ieee_tc/ablation/d1050_3b_no_handoff_w1/comparison/compare.py.snapshot',
                 ROOT/'paper_results/ieee_tc/ablation/d1123_7b_no_routing_w1/compare.py',
                 ROOT/'paper_results/ieee_tc/ablation/d1124_7b_no_handoff_w1/compare.py',
                 ROOT/'paper_results/ieee_tc/ablation/d1056_3b_r0h0_w1/comparison/compare.py.snapshot',
                 ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md',
                 ROOT/'docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md'):
        witness(path)
    report = dict(kind='d1125_7b_r0h0_w1_observation_v1', results=results,
        Full_relative_to_R0H0=relative,
        matched_inputs=4000, output_digest_differences=output_differences,
        same_non_target_configuration=True, confidence_interval=None, independent_numerical_proof=False,
        final_ranking_eligible=False, source_refs=REFS, plan_sha256=p.check_plan(),
        caveats=['Known seed42 development observations; not independent holdout or statistical inference.',
                 'Raw n_correct remains unknown; native binding and token contract are separate evidence.',
                 'Same-backend output changes retained, without an assumed numerical explanation.',
                 'Only routing ranking and activating-replica handoff are ablated. Ready-residency, admission, demand loading and physical safety are shared; closed-loop plans and activations may differ.',
                 'Actual routing checks use service_bins/load_only respectively, not counterfactual latency. Reuse the sealed D1124 Full physical-handoff audit; plans alone do not prove early physical work.',
                 'All resource and timing trade-offs are retained; no winner is preconditioned by this analyzer.'])
    factorial = four_cell_observation(report)
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT/'comparison.json', report)
    save(OUT/'mechanisms.json', mechanisms)
    save(OUT/'four_cell.json', factorial)
    with (OUT/'four_cell.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(factorial['metrics'][0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(factorial['metrics'])
    with (OUT/'metrics.csv').open('x', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['variant', 'native_contract_requests', 'joint_good', 'joint_slo', 'gpu_seconds',
            'gpu_seconds_per_offered', 'mean_ttft_ms', 'p95_ttft_ms', 'mean_tpot_ms', 'p95_tpot_ms',
            'mean_e2e_ms', 'server_transfers', 'wire_bytes'])
        for v, r in results.items():
            m = r['metrics']
            w.writerow([v, 4000, r['counts']['joint_pass'], r['joint_slo'], r['physical_gpu_seconds'],
                r['gpu_seconds_per_offered_request'], m['ttft_ms']['mean'], m['ttft_ms']['p95_type1'],
                m['tpot_ms']['mean'], m['tpot_ms']['p95_type1'], m['e2e_ms']['mean'],
                r['remote']['server_transfers'], r['remote']['wire_bytes']])
    save(OUT/'manifest.json', dict(sources=REFS,
        outputs={x.name:p.digest(x) for x in OUT.iterdir() if x.is_file()}))
    print(json.dumps(dict(relative=relative, matched_inputs=4000, output_differences=len(output_differences),
        mechanisms={v:dict(activations=len(m['activations']),
            routing_disagreements=m['routing']['same_state_choice_disagreements'],
            plans={k:p['count'] for k,p in m['plans'].items()}) for v,m in mechanisms.items()}), indent=2))


if __name__ == '__main__':
    main()
