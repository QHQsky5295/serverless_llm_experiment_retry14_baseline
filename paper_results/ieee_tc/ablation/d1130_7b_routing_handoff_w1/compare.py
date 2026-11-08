"""D1130 cumulative handoff comparison, reusing the sealed D1129 analysis.

Prepared during replay; do not execute until release and D1113/D299/D455.
Only the new run is projected. Previous observations and their Full anchor are
reused by immutable output SHA, not by rerunning historical analysis against
changed serving sources. Physical handoff timing is a separate retained-receipt
audit following D1124/D1126; plan counts alone do not establish useful overlap.
"""
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess

import ijson
import yaml

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
BASE = Path(__file__).resolve().parent
OUT = BASE/'comparison'
REUSED = ROOT/'paper_results/ieee_tc/ablation/d1129_7b_routing_only_w1/compare.py'
assert hashlib.sha256(REUSED.read_bytes()).hexdigest() == 'fda9c39c6e23b9dc564062f31a6ebdb9c47506140ff6002c06543a693d42b398'
# Import only existing helpers; the sealed script's main() is not called.
old = runpy.run_path(str(REUSED), run_name='d1129_analysis_helpers')
p, witness, read, save = (old[k] for k in ('p', 'witness', 'read', 'save'))
REFS = old['REFS']
matched_controller_rows = old['matched_controller_rows']
matched_request_contract = old['matched_request_contract']
require_controller_scenario = old['require_controller_scenario']
activation_first_services = old['activation_first_services']
Router = old['Router']
ReplicaRoutingSnapshot = old['ReplicaRoutingSnapshot']
ServiceObservationClass = old['ServiceObservationClass']
ServiceComponents = old['ServiceComponents']


def items(path, prefix):
    with path.open('rb') as f:
        yield from ijson.items(f, prefix, use_float=True)


def relative(candidate, reference):
    return dict(
        gpu_seconds_reduction=(reference['physical_gpu_seconds']-candidate['physical_gpu_seconds'])/reference['physical_gpu_seconds'],
        joint_slo_difference_percentage_points=100*(candidate['joint_slo']-reference['joint_slo']),
        p95_ttft_reduction=(reference['metrics']['ttft_ms']['p95_type1']-candidate['metrics']['ttft_ms']['p95_type1'])/reference['metrics']['ttft_ms']['p95_type1'])


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    group = p.cg_path()
    assert os.sched_getaffinity(0) == {2, 3, 26, 27}
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    for name, value in [('memory.high', 3*p.GIB), ('memory.max', 4*p.GIB), ('memory.swap.max', 0)]:
        assert (group/name).read_text().strip() == str(value)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert p.check_plan() == 'b5170643aa5d332efc67407af10ab7022a2d68fd802e2389d253968bd92dcf8f'

    previous = read(ROOT/'paper_results/ieee_tc/ablation/d1129_7b_routing_only_w1/comparison/comparison.json',
        '0d4e05e895377b7226f87b7129aa09812e23b63e83c87cbaf96bc0746ac5da0d')
    previous_mechanisms = read(ROOT/'paper_results/ieee_tc/ablation/d1129_7b_routing_only_w1/comparison/mechanisms.json',
        'b2e194430e0dce30cf785efc31268984ae963921b79d993dc35442803d184612')
    capacity = read(ROOT/'paper_results/ieee_tc/ablation/d1128_7b_capacity_only_w1/comparison/comparison.json',
        'bae8c73863997c79318d683c40e656c7389d7ae6bfb662aeb95b0c07232db6cc')
    for retained in (previous, capacity):
        assert retained['matched_inputs'] == 4000 and retained['same_non_target_configuration'] is True
    assert previous['results']['Full'] == capacity['results']['Full']
    prelaunch = read(BASE/'prelaunch.json')
    configs = [yaml.safe_load(witness(ROOT/'configs/ieee_tc'/name, sha).read_text()) for name, sha in (
        ('20261108_d1129_7b_routing_only_w1.yaml', '279e9df7382afa5d4cb73e3795eb3b1f5ac6690db87e73154ef6c1f19b825e8c'),
        ('20261108_d1130_7b_routing_handoff_w1.yaml', '517c2367ddd55b4d8221d37d36cb7ed08efb0475b836dd098510be7aa86e2572'))]
    bin_ms = configs[1]['resource_coordination']['service_bin_ms']
    for cfg in configs:
        del cfg['experiment']['output_dir']
        del cfg['storage']['nvme_cache_dir']
        del cfg['storage']['host_cache_dir']
    before, after = configs
    for key, before_value, after_value in (
        ('ieee_system_variant', 'routing_only_v1', 'routing_handoff_v1'),
        ('ieee_handoff_policy', 'no_handoff', 'full'),
        ('scale_up_handoff_enabled', False, True)):
        assert before['resource_coordination'][key] == before_value
        assert after['resource_coordination'][key] == after_value
        before['resource_coordination'][key] = after_value
    assert before['scenarios'][0]['name'] == 'tc_ieee_routing_only'
    before['scenarios'][0]['name'] = 'tc_ieee_routing_handoff'
    assert 'file_preparation_scope' not in before['scenarios'][0]['preloading']
    before['scenarios'][0]['preloading']['file_preparation_scope'] = 'handoff_only'
    assert before == after

    metric = read(BASE/'result1/reconciliation.json')['prime']
    review = read(BASE/'result1/evidence_review.json')
    assert metric['run'] == 'D1130' and metric['workload'] == review['workload_profile'] == 'W1'
    assert metric['counts']['joint_pass'] == review['counts']['joint_pass']
    assert metric['physical_gpu_seconds'] == review['physical_gpu_seconds']
    source = witness(review['source'], review['source_sha256'])
    launch = read(source.parent/'launch.json')
    assert launch['pass'] and launch['native_gpu_context_release_confirmed']
    reference_source = witness(prelaunch['routing_only_result'], prelaunch['routing_only_result_sha256'])
    reference_launch = read(reference_source.parent/'launch.json')
    assert launch['external_replay']['plan'] == reference_launch['external_replay']['plan']
    assert launch['preflight']['policy'] == reference_launch['preflight']['policy']
    index = read(ROOT/'paper_results/ieee_tc/inputs/20260927_7b_materialized_content_index.json')
    content = {a['id']: hashlib.sha256(json.dumps(sorted(a['files'], key=lambda f: f['path']),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest() for a in index['artifacts']}
    contracts = {}
    for variant, path, scenario, run_launch in (
        ('RoutingOnly', reference_source, 'tc_ieee_routing_only', reference_launch),
        ('RoutingHandoff', source, 'tc_ieee_routing_handoff', launch)):
        rows = matched_controller_rows(path, scenario, content)
        for row in rows.values():
            row['arrival_offset_s'] = row['arrival']-run_launch['external_replay']['replay_t0_s']
        contracts[variant] = rows
    assert contracts['RoutingOnly'].keys() == contracts['RoutingHandoff'].keys()
    differences = []
    for rid, a in contracts['RoutingOnly'].items():
        b = contracts['RoutingHandoff'][rid]
        matched_request_contract(a, b)
        if a['native']['completion_token_ids_sha256'] != b['native']['completion_token_ids_sha256']:
            differences.append(dict(request_id=rid, RoutingOnly=a['native']['completion_token_ids_sha256'],
                RoutingHandoff=b['native']['completion_token_ids_sha256']))

    metadata = next(items(source, 'metadata'))
    coordination = require_controller_scenario(metadata, 'tc_ieee_routing_handoff')
    gates = coordination['feature_gates']
    assert gates['scale_up_handoff_enabled'] is True
    assert gates['hierarchical_residency_enabled'] is False and gates['coordination_enabled'] is False
    result = dict(metric, variant='RoutingHandoff', scenario='tc_ieee_routing_handoff',
        joint_slo=review['timing_joint_rate'], timing_feasible=review['timing_feasible'],
        remote=review['remote'], resource=review['resource'], tier_counts=review['tier_counts'])
    dispatches, routing_rows = [], []
    for r in items(source, 'detailed_results.tc_ieee_routing_handoff.requests.item'):
        routing = r['gpu_reference_evidence']['routing_snapshot']
        contract = routing['selection_contract']
        assert contract['owner_policy'] == 'ieee_confirmed' and contract['strategy'] == 'service_bins'
        assert contract['locality_load_slack'] is None
        candidates = tuple(ReplicaRoutingSnapshot(**(v | {
            'active_adapters': frozenset(v['active_adapters']),
            'service_class': ServiceObservationClass(**v['service_class']) if v['service_class'] else None,
            'service': ServiceComponents(**v['service']) if v['service'] else None,
        })) for v in routing['candidates'])
        assert candidates and all(v.request_id == r['request_id'] and v.adapter_id == r['adapter_id'] for v in candidates)
        actual, load = Router.select_ieee_snapshot(candidates, bin_ms), Router.select_load_snapshot(candidates)
        assert actual is not None and load is not None
        assert actual.replica_id == routing['selected_replica_id'] == r['instance_id']
        routing_rows.append(dict(request_id=r['request_id'], candidate_count=len(candidates),
            feasible_count=sum(v.feasible for v in candidates), actual=actual.replica_id,
            load_choice=load.replica_id, differs=actual.replica_id != load.replica_id,
            actual_tier=actual.service_class.tier, load_tier=load.service_class.tier,
            actual_admitted=actual.admitted_requests, load_admitted=load.admitted_requests))
        n, e = r['native_token_timing'], r['external_arrival_timing']
        dispatches.append(dict(request_id=r['request_id'], instance_id=r['instance_id'],
            adapter_id=r['adapter_id'], readiness_tier_before_dispatch=r['readiness_tier_before_dispatch'],
            clock_id=n['native_clock_id'], dispatch_monotonic_s=n['native_dispatch_monotonic_s'],
            first_token_monotonic_s=n['native_first_token_monotonic_s'],
            arrival_monotonic_s=e['planned_arrival_s'], success=r['success']))
    assert len(routing_rows) == 4000
    outcome = witness(source.parent/'launch.launch/physical_deployment/main_outcome.json')
    actual_contract = next(items(outcome, 'coordination_after_shutdown.ieee_mechanism_contract'))
    file_owner = next(items(outcome, 'coordination_after_shutdown.ieee_file_pending_targets'))
    assert actual_contract['variant'] == 'routing_handoff_v1'
    assert actual_contract['proactive_preparation'] is True
    assert actual_contract['file_preparation_scope'] == file_owner['file_preparation_scope'] == 'handoff_only'
    assert actual_contract['file_cache_policy'] == file_owner['file_cache_policy'] == 'demand_lru'
    assert all(row['mode'] == 'handoff' and row['activation_id'] for row in file_owner['plans'])
    activations = list(items(outcome, 'mechanism_events._ieee_activations.item'))
    assert activations and all(a['policy'] == 'full' for a in activations)
    activation_ids = {a['activation_id'] for a in activations}
    mechanism = dict(activations=activations,
        first_services=activation_first_services(activations, dispatches, clock_id=launch['external_replay']['clock_id']),
        feature_gates=gates, actual_contract=actual_contract, file_owner=file_owner,
        plans={}, routing=dict(strategy='service_bins', validated_requests=4000,
            same_state_choice_disagreements=sum(x['differs'] for x in routing_rows),
            multiple_feasible_requests=sum(x['feasible_count'] > 1 for x in routing_rows),
            selected_tiers=dict(Counter(x['actual_tier'] for x in routing_rows)),
            same_state_load_choice_tiers=dict(Counter(x['load_tier'] for x in routing_rows)),
            rows=routing_rows, counterfactual_latency_generated=False))
    for key in ('_ieee_file_preparation_plans', '_ieee_gpu_preparation_plans',
                '_ieee_native_host_preparations', '_ieee_residency_epochs'):
        plans = list(items(outcome, 'mechanism_events.'+key+'.item'))
        if key == '_ieee_residency_epochs':
            assert not plans
        handoff_records, decisions = [], Counter()
        for plan in plans:
            assert plan['trigger_reason'] == 'handoff'
            assert plan['activation_id'] in activation_ids and plan['target_replica'] == plan['activation_id']
            handoff_records.append({k: plan[k] for k in (
                'plan_id', 'target_replica', 'activation_id', 'trigger_reason', 'objective_sha256',
                'planning_sha256', 'state', 'started_at', 'finished_at', 'targets', 'registration',
                'results', 'attempts') if k in plan})
            if key == '_ieee_gpu_preparation_plans':
                assert plan['capacity_only'] is True
                for attempt in plan['attempts']:
                    receipt = attempt.get('receipt')
                    if receipt and receipt.get('proactive_admission_evaluated') is True:
                        decisions[receipt['admission']['reason']] += 1
        mechanism['plans'][key] = dict(count=len(plans),
            reasons=dict(Counter(x.get('trigger_reason', 'not_recorded') for x in plans)),
            states=dict(Counter(x.get('state', 'not_recorded') for x in plans)),
            handoff_records=handoff_records, admission_decision_reasons=dict(decisions))
    physical = read(source.parent/'launch.launch/physical_deployment/summary.json')
    scale_up = list(items(source, 'detailed_results.tc_ieee_routing_handoff.scale_up_events.item'))
    scale_down = next(items(source, 'detailed_results.tc_ieee_routing_handoff.scale_down_events'))
    down_log = list(items(source, 'detailed_results.tc_ieee_routing_handoff.scale_down_event_log.item'))
    assert scale_down == len(down_log)
    mechanism['control'] = dict(scale_up_events=scale_up, scale_down_events=scale_down,
        scale_down_event_log=down_log, physical_allocations=len(physical['allocation_journals']))

    results = dict(ElasticOnly=previous['results']['ElasticOnly'], RoutingOnly=previous['results']['RoutingOnly'],
        RoutingHandoff=result, CapacityOnly=capacity['results']['CapacityOnly'], Full=previous['results']['Full'])
    reference = read(ROOT/'paper_results/ieee_tc/main_comparison/d594_reference_and_ttft/resident_references.json',
        'a9b6b25fe4ec8a17921dd5968c4fb494b7a6d795c0d4a11c7b7d306ac06bc60c')['W1']
    budget = reference['B_gpu_seconds']
    assert reference['reference_frozen'] and reference['budget_fraction'] == 0.75
    assert budget == reference['budget_fraction'] * reference['U_ref_gpu_seconds']
    for r in results.values():
        r['budget_gpu_seconds'] = budget
        r['within_budget'] = r['physical_gpu_seconds'] <= budget
    chain = list(results)
    contrasts = {f'{candidate}_relative_to_{ref}': relative(results[candidate], results[ref])
        for ref, candidate in zip(chain, chain[1:])}
    for path in (Path(__file__), REUSED, ROOT/'scripts/ieee_tc/check_d299_full_context.py',
                 ROOT/'scripts/ieee_tc/audit_d389_scaleup_handoff_labels.py',
                 ROOT/'faaslora/experiment/instance_pool.py',
                 ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md',
                 ROOT/'docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md'):
        witness(path)
    report = dict(kind='d1130_7b_routing_handoff_w1_observation_v1', results=results,
        adjacent_contrasts=contrasts, Full_relative_to_RoutingHandoff=relative(results['Full'], result),
        matched_inputs=4000, output_digest_differences=differences,
        output_difference_reference='RoutingOnly D1129',
        same_non_target_configuration=True,
        reused_observations=dict(Full='D1115', ElasticOnly='D1119', RoutingOnly='D1129', CapacityOnly='D1128',
            basis='sealed comparison output SHA and common Full input anchor; no historical analysis rerun'),
        cumulative_contrast=dict(reference='RoutingOnly', candidate='RoutingHandoff',
            only_activation_handoff_scope_and_identity_changed=True),
        source_refs=REFS, plan_sha256=p.check_plan(), confidence_interval=None,
        independent_numerical_proof=False, final_ranking_eligible=False,
        caveats=['Known seed42 development observations, not independent holdout or statistical inference.',
            'Native binding/count agreement is distinct from independent numerical correctness; n_correct stays unknown.',
            'Same-backend digest differences retained without an assumed numerical cause.',
            'Closed-loop activations and prepared sets may differ; not a controlled identical-plan timing experiment.',
            'Cumulative increments depend on the order of mechanisms.',
            'Plan counts are not physical completion, use, latency savings or independent causal proof.',
            'Each full workpoint retains its own resource, SLO and tail values; no splicing or preconditioned winner.',
            'Legacy readiness_routing_enabled metadata is not the IEEE selector; actual snapshots are recomputed.'])
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT/'comparison.json', report)
    save(OUT/'mechanisms.json', dict(RoutingHandoff=mechanism,
        RoutingOnly=previous_mechanisms['RoutingOnly']))
    with (OUT/'metrics.csv').open('x', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['variant', 'native_contract_requests', 'joint_good', 'joint_slo', 'gpu_seconds',
            'gpu_seconds_per_offered', 'mean_ttft_ms', 'p95_ttft_ms', 'mean_tpot_ms', 'p95_tpot_ms',
            'mean_e2e_ms', 'server_transfers', 'wire_bytes', 'budget_gpu_seconds', 'within_budget'])
        for variant, r in results.items():
            m = r['metrics']
            writer.writerow([variant, 4000, r['counts']['joint_pass'], r['joint_slo'], r['physical_gpu_seconds'],
                r['gpu_seconds_per_offered_request'], m['ttft_ms']['mean'], m['ttft_ms']['p95_type1'],
                m['tpot_ms']['mean'], m['tpot_ms']['p95_type1'], m['e2e_ms']['mean'],
                r['remote']['server_transfers'], r['remote']['wire_bytes'], budget, r['within_budget']])
    save(OUT/'manifest.json', dict(sources=REFS,
        outputs={x.name: p.digest(x) for x in OUT.iterdir() if x.is_file()}))
    print(json.dumps(dict(adjacent_contrasts=contrasts, output_differences=len(differences),
        new_plans={k: v['count'] for k, v in mechanism['plans'].items()}), indent=2))


if __name__ == '__main__':
    main()
