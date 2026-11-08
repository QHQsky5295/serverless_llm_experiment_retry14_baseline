"""D1129: reuse D1120/D1119 for the RoutingOnly cumulative node.

Prepared during replay; execute only after actual release and metric audit.
Reuses D1120/D1119 without modifying their sealed scripts/results.
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


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    group = p.cg_path()
    assert os.sched_getaffinity(0) == {2, 3, 26, 27}
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert (group/'memory.max').read_text().strip() == str(4*p.GIB)
    assert (group/'memory.swap.max').read_text().strip() == '0'
    assert p.check_plan() == 'b5170643aa5d332efc67407af10ab7022a2d68fd802e2389d253968bd92dcf8f'
    configs = [yaml.safe_load(witness(ROOT/'configs/ieee_tc'/name, sha).read_text())
        for name, sha in (
            ('20261107_d1100_7b_busy_service_rate_w2.yaml',
             '386a34c84ec70e5d0d366f6d3bcaf0891536be503ba1359d583a93bd5ecf303f'),
            ('20261108_d1129_7b_routing_only_w1.yaml',
             '279e9df7382afa5d4cb73e3795eb3b1f5ac6690db87e73154ef6c1f19b825e8c'))]
    # Establish the direct cumulative contrast before normalizing to Full.
    elastic_cfg = yaml.safe_load(witness(ROOT/'configs/ieee_tc/20261107_d1119_7b_elastic_only_w1.yaml',
        '2796e6d03e257ccfb69bd76be36058c966ef2ac6c978fc7ff6061413da000a74').read_text())
    candidate_cfg = json.loads(json.dumps(configs[1]))
    for cfg in (elastic_cfg, candidate_cfg):
        del cfg['experiment']['output_dir']
        del cfg['storage']['nvme_cache_dir']
        del cfg['storage']['host_cache_dir']
    assert elastic_cfg['resource_coordination']['ieee_system_variant'] == 'elastic_only_v1'
    assert elastic_cfg['resource_coordination']['ieee_selection_strategy'] == 'load_only'
    assert elastic_cfg['scenarios'][0]['name'] == 'tc_ieee_elastic_only'
    elastic_cfg['resource_coordination']['ieee_system_variant'] = 'routing_only_v1'
    elastic_cfg['resource_coordination']['ieee_selection_strategy'] = 'service_bins'
    elastic_cfg['scenarios'][0]['name'] = 'tc_ieee_routing_only'
    assert elastic_cfg == candidate_cfg
    bin_ms = configs[0]['resource_coordination']['service_bin_ms']
    for cfg in configs:
        del cfg['experiment']['output_dir']
        del cfg['storage']['nvme_cache_dir']
        del cfg['storage']['host_cache_dir']
    assert configs[0]['resource_coordination']['ieee_handoff_policy'] == 'full'
    assert configs[1]['resource_coordination']['ieee_handoff_policy'] == 'no_handoff'
    configs[1]['resource_coordination']['ieee_handoff_policy'] = 'full'
    for key, value in dict(ieee_system_variant='routing_only_v1', ieee_selection_strategy='service_bins',
            coordination_enabled=False, scale_up_handoff_enabled=False,
            effective_capacity_admission_enabled=False).items():
        assert key not in configs[0]['resource_coordination']
        assert configs[1]['resource_coordination'].pop(key) == value
    assert configs[1]['scenarios'][0]['name'] == 'tc_ieee_routing_only'
    configs[1]['scenarios'][0]['name'] = 'tc_ieee_full'
    for key, value in dict(file_cache_policy='demand_lru', enabled=False,
            hierarchical_residency_enabled=False, dynamic_forwarding_enabled=False,
            gpu_dynamic_forwarding_enabled=False, host_promotion_on_nvme_hit_enabled=False).items():
        assert key not in configs[0]['scenarios'][0]['preloading']
        assert configs[1]['scenarios'][0]['preloading'].pop(key) == value
    assert configs[0] == configs[1]
    index = read(ROOT/'paper_results/ieee_tc/inputs/20260927_7b_materialized_content_index.json')
    content = {a['id']: hashlib.sha256(json.dumps(sorted(a['files'], key=lambda f:f['path']),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest() for a in index['artifacts']}
    results, contracts, mechanisms, launches, output_differences = {}, {}, {}, {}, []
    for variant, run_id, scenario, directory in (
        ('Full', 'D1115', 'tc_ieee_full', ROOT/'paper_results/ieee_tc/main_comparison/d1115_same_version_w1/result1'),
        ('RoutingOnly', 'D1129', 'tc_ieee_routing_only', BASE/'result1')):
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
        expected_strategy = 'service_bins'
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
                actual = full_choice
                assert actual is not None and full_choice is not None and load_choice is not None
                assert actual.replica_id == routing['selected_replica_id'] == r['instance_id']
                routing_rows.append(dict(request_id=r['request_id'], candidate_count=len(candidates),
                    feasible_count=sum(v.feasible for v in candidates), actual=actual.replica_id,
                    full_choice=full_choice.replica_id, load_choice=load_choice.replica_id,
                    differs=full_choice.replica_id != load_choice.replica_id,
                    actual_tier=actual.service_class.tier,
                    load_tier=load_choice.service_class.tier,
                    actual_admitted=actual.admitted_requests,
                    load_admitted=load_choice.admitted_requests,
                    actual_pending_loads=actual.pending_loads,
                    load_pending_loads=load_choice.pending_loads,
                    actual_estimated_service_ms=actual.service.total_ms,
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
                selected_tiers=dict(Counter(x['actual_tier'] for x in routing_rows)),
                same_state_load_choice_tiers=dict(Counter(x['load_tier'] for x in routing_rows)),
                selected_admitted_histogram=dict(Counter(x['actual_admitted'] for x in routing_rows)),
                same_state_load_admitted_histogram=dict(Counter(x['load_admitted'] for x in routing_rows)),
                rows=routing_rows, counterfactual_latency_generated=False), plans={})
        for key in ('_ieee_file_preparation_plans', '_ieee_gpu_preparation_plans',
                    '_ieee_native_host_preparations', '_ieee_residency_epochs'):
            count, reasons, states, decision_reasons, evaluated = 0, Counter(), Counter(), Counter(), 0
            with outcome.open('rb') as f:
                for plan in ijson.items(f, 'mechanism_events.'+key+'.item', use_float=True):
                    count += 1
                    reasons[plan.get('trigger_reason', 'not_recorded')] += 1
                    states[plan.get('state', 'not_recorded')] += 1
                    if key == '_ieee_gpu_preparation_plans':
                        assert variant == 'Full' and plan['capacity_only'] is False
                        for attempt in plan['attempts']:
                            receipt = attempt.get('receipt')
                            if receipt and receipt.get('proactive_admission_evaluated') is True:
                                evaluated += 1
                                decision_reasons[receipt['admission']['reason']] += 1
            mechanism['plans'][key] = dict(count=count, reasons=dict(reasons), states=dict(states),
                evaluated_admission_attempts=evaluated, admission_decision_reasons=dict(decision_reasons))
        if variant == 'RoutingOnly':
            assert activations and all(a['preparation_state'] == 'disabled' and a['plan_sha256'] is None for a in activations)
            assert all(x['count'] == 0 for x in mechanism['plans'].values())
        assert all(a['policy'] == ('full' if variant == 'Full' else 'no_handoff') for a in activations)
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
    assert launches['Full']['external_replay']['plan'] == launches['RoutingOnly']['external_replay']['plan']
    assert launches['Full']['preflight']['policy'] == launches['RoutingOnly']['preflight']['policy']
    assert contracts['Full'].keys() == contracts['RoutingOnly'].keys()
    for rid, a in contracts['Full'].items():
        b = contracts['RoutingOnly'][rid]
        matched_request_contract(a, b)
        if a['native']['completion_token_ids_sha256'] != b['native']['completion_token_ids_sha256']:
            output_differences.append(dict(request_id=rid,
                Full=a['native']['completion_token_ids_sha256'], RoutingOnly=b['native']['completion_token_ids_sha256']))
    full, elastic = results['Full'], results['RoutingOnly']
    relative = dict(gpu_seconds_reduction=(elastic['physical_gpu_seconds']-full['physical_gpu_seconds'])/elastic['physical_gpu_seconds'],
        joint_slo_difference_percentage_points=100*(full['joint_slo']-elastic['joint_slo']),
        p95_ttft_reduction=(elastic['metrics']['ttft_ms']['p95_type1']-full['metrics']['ttft_ms']['p95_type1'])/elastic['metrics']['ttft_ms']['p95_type1'])
    # Reuse the sealed ElasticOnly observation; its identical Full input anchor
    # makes input equality transitive. No ElasticOnly performance replay or
    # unobserved routing-vs-load counterfactual latency is generated.
    previous_dir = ROOT/'paper_results/ieee_tc/ablation/d1119_7b_elastic_only_w1/comparison'
    previous_manifest = read(previous_dir/'manifest.json')
    previous = read(previous_dir/'comparison.json',
                    previous_manifest['outputs']['comparison.json'])
    previous_mechanisms = read(previous_dir/'mechanisms.json',
                              previous_manifest['outputs']['mechanisms.json'])
    assert previous['matched_inputs'] == 4000 and previous['same_non_target_configuration'] is True
    assert previous['results']['Full'] == results['Full']
    for path, digest in previous['source_refs'].items():
        witness(path, digest)
    results = dict(Full=results['Full'],
        ElasticOnly=previous['results']['ElasticOnly'],
        RoutingOnly=results['RoutingOnly'])
    mechanisms['ElasticOnly'] = previous_mechanisms['ElasticOnly']
    routing_vs_elastic_only = dict(
        gpu_seconds_reduction=(results['ElasticOnly']['physical_gpu_seconds']-elastic['physical_gpu_seconds'])/results['ElasticOnly']['physical_gpu_seconds'],
        joint_slo_difference_percentage_points=100*(elastic['joint_slo']-results['ElasticOnly']['joint_slo']),
        p95_ttft_reduction=(results['ElasticOnly']['metrics']['ttft_ms']['p95_type1']-elastic['metrics']['ttft_ms']['p95_type1'])/results['ElasticOnly']['metrics']['ttft_ms']['p95_type1'])
    reference = read(ROOT/'paper_results/ieee_tc/main_comparison/d594_reference_and_ttft/resident_references.json',
        'a9b6b25fe4ec8a17921dd5968c4fb494b7a6d795c0d4a11c7b7d306ac06bc60c')['W1']
    assert reference['reference_frozen'] and reference['budget_fraction'] == 0.75
    budget = reference['B_gpu_seconds']
    assert budget == reference['budget_fraction'] * reference['U_ref_gpu_seconds']
    for result in results.values():
        result['budget_gpu_seconds'] = budget
        result['within_budget'] = result['physical_gpu_seconds'] <= budget
    for path in (Path(__file__), ROOT/'scripts/ieee_tc/check_d299_full_context.py',
                 ROOT/'scripts/ieee_tc/audit_d389_scaleup_handoff_labels.py',
                 ROOT/'faaslora/experiment/instance_pool.py',
                 ROOT/'paper_results/ieee_tc/ablation/d1057_3b_elastic_only_w1/comparison/compare.py.snapshot',
                 ROOT/'paper_results/ieee_tc/ablation/d1120_7b_elastic_locality_w1/compare.py',
                 ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md',
                 ROOT/'docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md'):
        witness(path)
    report = dict(kind='d1129_7b_routing_only_w1_observation_v1', results=results,
        Full_relative_to_RoutingOnly=relative,
        RoutingOnly_relative_to_ElasticOnly=routing_vs_elastic_only,
        reused_ElasticOnly=dict(source=str(previous_dir/'comparison.json'),
            input_match_via_same_Full_reference=4000,
            output_differences_vs_Full=previous['output_digest_differences'],
            direct_ElasticOnly_vs_RoutingOnly_output_comparison_performed=False),
        matched_inputs=4000, output_digest_differences=output_differences,
        cumulative_contrast=dict(reference='ElasticOnly', candidate='RoutingOnly',
            only_target_routing_and_identity_changed=True),
        budget_comparison=dict(common_budget_gpu_seconds=budget,
            all_within_budget=all(r['within_budget'] for r in results.values()),
            jointly_within_budget_RoutingOnly_ElasticOnly=all(results[v]['within_budget'] for v in ('RoutingOnly','ElasticOnly')),
            joint_not_lower_RoutingOnly_than_ElasticOnly=elastic['joint_slo'] >= results['ElasticOnly']['joint_slo'],
            p95_lower_RoutingOnly_than_ElasticOnly=elastic['metrics']['ttft_ms']['p95_type1'] < results['ElasticOnly']['metrics']['ttft_ms']['p95_type1']),
        same_non_target_configuration=True, confidence_interval=None, independent_numerical_proof=False,
        final_ranking_eligible=False, source_refs=REFS, plan_sha256=p.check_plan(),
        caveats=['Known seed42 development observations; not independent holdout or statistical inference.',
                 'Raw n_correct remains unknown; native binding and token contract are separate evidence.',
                 'Same-backend output changes retained, without an assumed numerical explanation.',
                 'RoutingOnly versus ElasticOnly isolates routing within demand-only caching; Full comparison combines multiple mechanisms.',
                 'Natural feedback may change residency and activations; this is not a last-known versus confirmed-state experiment.',
                 'Budget feasibility is reported per complete workpoint; no cost/quality splicing.',
                 'Snapshot shadow decisions do not generate counterfactual latency.',
                 'All resource and timing trade-offs are retained; no winner is preconditioned by this analyzer.'])
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT/'comparison.json', report)
    save(OUT/'mechanisms.json', mechanisms)
    with (OUT/'metrics.csv').open('x', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['variant', 'native_contract_requests', 'joint_good', 'joint_slo', 'gpu_seconds',
            'gpu_seconds_per_offered', 'mean_ttft_ms', 'p95_ttft_ms', 'mean_tpot_ms', 'p95_tpot_ms',
            'mean_e2e_ms', 'server_transfers', 'wire_bytes', 'budget_gpu_seconds', 'within_budget'])
        for v, r in results.items():
            m = r['metrics']
            w.writerow([v, 4000, r['counts']['joint_pass'], r['joint_slo'], r['physical_gpu_seconds'],
                r['gpu_seconds_per_offered_request'], m['ttft_ms']['mean'], m['ttft_ms']['p95_type1'],
                m['tpot_ms']['mean'], m['tpot_ms']['p95_type1'], m['e2e_ms']['mean'],
                r['remote']['server_transfers'], r['remote']['wire_bytes'], budget, r['within_budget']])
    save(OUT/'manifest.json', dict(sources=REFS,
        outputs={x.name:p.digest(x) for x in OUT.iterdir() if x.is_file()}))
    print(json.dumps(dict(relative=relative, matched_inputs=4000, output_differences=len(output_differences),
        mechanisms={v:dict(activations=len(m['activations']),
            routing_disagreements=m['routing']['same_state_choice_disagreements'],
            plans={k:p['count'] for k,p in m['plans'].items()}) for v,m in mechanisms.items()}), indent=2))


if __name__ == '__main__':
    main()
