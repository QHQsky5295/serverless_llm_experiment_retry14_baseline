"""D1128: reuse D1127 comparison and D1052 typed-admission extraction.

Prepared while the replay runs; execute only after real release and metric audit.
Reuses existing analyzers without modifying their sealed scripts/results.
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


def typed_admission_attempts(plan, activations, expected):
    """D1052 projection without hardcoded counts or imputing missing receipts."""
    activation_id = plan.get('activation_id')
    activation_kind = None
    if activation_id is not None:
        activation_kind = next(a['category'] for a in activations
            if a['activation_id'] == activation_id)
    rows = []
    for attempt in plan['attempts']:
        receipt = attempt.get('receipt')
        entry = dict(plan_id=plan['plan_id'], attempt_id=attempt['attempt_id'],
            trigger_reason=plan['trigger_reason'], activation_kind=activation_kind,
            target_replica=plan['target_replica'], capacity_only=expected,
            attempt_state=attempt['state'], receipt_present=receipt is not None,
            receipt_reason=None, resource_admission_evaluated=False,
            native_load_invoked=None, actual_admit=None,
            observed_soft_headroom_deficit=None, decision=None,
            plan_state=plan.get('state'), plan_closed=plan.get('closed'),
            plan_error_type=plan.get('error_type'))
        if receipt is not None:
            if 'capacity_only' in receipt:
                assert receipt['capacity_only'] is expected
            entry['receipt_reason'] = receipt.get('reason')
            entry['native_load_invoked'] = receipt.get('native_load_invoked')
            evaluated = receipt.get('proactive_admission_evaluated')
            assert evaluated is None or type(evaluated) is bool
            entry['resource_admission_evaluated'] = evaluated is True
            if evaluated is True:
                decision = receipt['admission']
                assert type(decision['admit']) is bool
                entry['actual_admit'] = decision['admit']
                entry['decision'] = {k:decision[k] for k in (
                    'reason','effective_capacity_bytes','physical_increment_bytes',
                    'predicted_kv_bytes','batch_pressure','load_pressure')}
                entry['proposal'] = decision['proposal']
                entry['observed_soft_headroom_deficit'] = (
                    decision['proposal']['footprint_bytes'] > decision['effective_capacity_bytes'])
                if expected:
                    assert decision['reason'] != 'defer_effective_capacity'
        if not entry['receipt_present']:
            category = 'receipt_absent'
            entry['unresolved_attempt'] = attempt
            entry['plan_context'] = {k:v for k,v in plan.items()
                if k not in ('attempts','registration','results','targets')}
        elif entry['resource_admission_evaluated']:
            category = 'typed_admit' if entry['actual_admit'] else 'typed_defer'
        elif entry['receipt_reason'] == 'stale_snapshot':
            category = 'stale_snapshot_before_admission'
        elif entry['native_load_invoked'] is False and entry['attempt_state'] == 'completed':
            category = 'completed_reuse_without_native_load'
        else:
            category = 'other_recorded_receipt_not_typed'
            entry['unclassified_receipt'] = receipt
        entry['category'] = category
        rows.append(entry)
    return rows


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
            ('20261108_d1128_7b_capacity_only_w1.yaml',
             'e136c37406075476fc1f14b5b0b37d49df3f835277aeb8bb8d21623361fa44a2'))]
    bin_ms = configs[0]['resource_coordination']['service_bin_ms']
    for cfg in configs:
        del cfg['experiment']['output_dir']
        del cfg['storage']['nvme_cache_dir']
        del cfg['storage']['host_cache_dir']
    assert configs[0]['resource_coordination']['min_instances'] == 1
    assert configs[1]['resource_coordination']['min_instances'] == 1
    assert all(c['resource_coordination']['max_instances'] == 4 for c in configs)
    assert configs[0]['resource_coordination']['ieee_handoff_policy'] == 'full'
    assert configs[1]['resource_coordination']['ieee_handoff_policy'] == 'full'
    assert 'coordination_enabled' not in configs[0]['resource_coordination']
    assert configs[1]['resource_coordination'].pop('coordination_enabled') is False
    assert configs[0] == configs[1]
    reference = read(ROOT/'paper_results/ieee_tc/main_comparison/d594_reference_and_ttft/resident_references.json',
        'a9b6b25fe4ec8a17921dd5968c4fb494b7a6d795c0d4a11c7b7d306ac06bc60c')['W1']
    assert reference['reference_frozen'] and reference['budget_fraction'] == 0.75
    budget = reference['B_gpu_seconds']
    assert budget == reference['budget_fraction'] * reference['U_ref_gpu_seconds']
    index = read(ROOT/'paper_results/ieee_tc/inputs/20260927_7b_materialized_content_index.json')
    content = {a['id']: hashlib.sha256(json.dumps(sorted(a['files'], key=lambda f:f['path']),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest() for a in index['artifacts']}
    results, contracts, mechanisms, launches, output_differences = {}, {}, {}, {}, []
    for variant, run_id, scenario, directory in (
        ('Full', 'D1115', 'tc_ieee_full', ROOT/'paper_results/ieee_tc/main_comparison/d1115_same_version_w1/result1'),
        ('CapacityOnly', 'D1128', 'tc_ieee_full', BASE/'result1')):
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
            remote=review['remote'], resource=review['resource'], tier_counts=review['tier_counts'],
            budget_gpu_seconds=budget, within_budget=metric['physical_gpu_seconds'] <= budget)
        results[variant] = result
        dispatches, routing_rows = [], []
        native_sources, dispatch_sources = Counter(), Counter()
        staged_joins = 0
        expected_strategy = 'service_bins'
        with source.open('rb') as f:
            for r in ijson.items(f, f'detailed_results.{scenario}.requests.item', use_float=True):
                native = r['gpu_reference_evidence']['receipt']
                native_sources[native['source_tier_before_acquisition']] += 1
                dispatch_sources[r['readiness_tier_before_dispatch']] += 1
                staged_joins += native.get('native_staged_source_reused', False) is True
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
            native_acquisition_sources=dict(native_sources), dispatch_sources=dict(dispatch_sources),
            staged_demand_joins=staged_joins,
            first_services=activation_first_services(activations, dispatches, clock_id=launch['external_replay']['clock_id']),
            routing=dict(strategy=expected_strategy, validated_requests=4000,
                same_state_choice_disagreements=sum(x['differs'] for x in routing_rows),
                multiple_feasible_requests=sum(x['feasible_count'] > 1 for x in routing_rows),
                rows=routing_rows, counterfactual_latency_generated=False), plans={})
        for key in ('_ieee_file_preparation_plans', '_ieee_gpu_preparation_plans',
                    '_ieee_native_host_preparations', '_ieee_residency_epochs'):
            count, reasons, states, decision_reasons, evaluated = 0, Counter(), Counter(), Counter(), 0
            handoff_records = []
            file_result_targets = Counter()
            admission_records = []
            with outcome.open('rb') as f:
                for plan in ijson.items(f, 'mechanism_events.'+key+'.item', use_float=True):
                    count += 1
                    if key == '_ieee_file_preparation_plans':
                        for target in plan['results']:
                            file_result_targets[target['target_tier']] += 1
                    reasons[plan.get('trigger_reason', 'not_recorded')] += 1
                    states[plan.get('state', 'not_recorded')] += 1
                    if plan.get('trigger_reason') == 'handoff':
                        handoff_records.append({k:plan[k] for k in (
                            'plan_id','target_replica','activation_id','trigger_reason',
                            'objective_sha256','planning_sha256','state','started_at','finished_at',
                            'targets','registration','results','attempts') if k in plan})
                    if key == '_ieee_gpu_preparation_plans':
                        expected = variant == 'CapacityOnly'
                        assert type(plan['capacity_only']) is bool and plan['capacity_only'] is expected
                        records = typed_admission_attempts(plan, activations, expected)
                        admission_records.extend(records)
                        for entry in records:
                            if entry['resource_admission_evaluated']:
                                evaluated += 1
                                decision_reasons[entry['decision']['reason']] += 1
            mechanism['plans'][key] = dict(count=count, reasons=dict(reasons), states=dict(states),
                evaluated_admission_attempts=evaluated, admission_decision_reasons=dict(decision_reasons),
                handoff_records=handoff_records,
                reported_file_results_by_target=dict(file_result_targets),
                registered_target_tiers_recorded=False if key == '_ieee_file_preparation_plans' else None)
            if key == '_ieee_gpu_preparation_plans':
                mechanism['proactive_admission'] = dict(
                    policy_flag=variant == 'CapacityOnly', plan_count=count,
                    attempt_count=len(admission_records), attempts=admission_records,
                    categories=dict(Counter(r['category'] for r in admission_records)),
                    trigger_counts=dict(Counter(r['trigger_reason'] for r in admission_records)),
                    typed_decisions=evaluated, decision_reasons=dict(decision_reasons),
                    observed_soft_headroom_deficits=sum(r['observed_soft_headroom_deficit'] is True
                        for r in admission_records),
                    missing_receipts=[r for r in admission_records if not r['receipt_present']],
                    caveat='Attempts are not unique physical loads; missing receipts and non-typed outcomes remain explicit. Same-state headroom observations do not generate counterfactual service latency.')
        assert activations
        expected_policy = 'full'
        assert all(a['policy'] == expected_policy for a in activations)
        # Keep all native preparation/closed-loop evidence; no target-count or
        # admission count is treated as proof of saved latency or physical loads.
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
    full_gates = dict(mechanisms['Full']['feature_gates'])
    capacity_gates = dict(mechanisms['CapacityOnly']['feature_gates'])
    assert full_gates.pop('coordination_enabled') is True
    assert capacity_gates.pop('coordination_enabled') is False
    assert full_gates == capacity_gates
    assert launches['Full']['external_replay']['plan'] == launches['CapacityOnly']['external_replay']['plan']
    assert launches['Full']['preflight']['policy'] == launches['CapacityOnly']['preflight']['policy']
    assert contracts['Full'].keys() == contracts['CapacityOnly'].keys()
    for rid, a in contracts['Full'].items():
        b = contracts['CapacityOnly'][rid]
        matched_request_contract(a, b)
        if a['native']['completion_token_ids_sha256'] != b['native']['completion_token_ids_sha256']:
            output_differences.append(dict(request_id=rid,
                Full=a['native']['completion_token_ids_sha256'], CapacityOnly=b['native']['completion_token_ids_sha256']))
    full, ablated = results['Full'], results['CapacityOnly']
    relative = dict(gpu_seconds_reduction=(ablated['physical_gpu_seconds']-full['physical_gpu_seconds'])/ablated['physical_gpu_seconds'],
        joint_slo_difference_percentage_points=100*(full['joint_slo']-ablated['joint_slo']),
        p95_ttft_reduction=(ablated['metrics']['ttft_ms']['p95_type1']-full['metrics']['ttft_ms']['p95_type1'])/ablated['metrics']['ttft_ms']['p95_type1'])
    for path in (Path(__file__), ROOT/'scripts/ieee_tc/check_d299_full_context.py',
                 ROOT/'scripts/ieee_tc/audit_d389_scaleup_handoff_labels.py',
                 ROOT/'faaslora/experiment/instance_pool.py',
                 ROOT/'paper_results/ieee_tc/ablation/d1052_3b_capacity_only_w1/comparison/compare.py.snapshot',
                 ROOT/'paper_results/ieee_tc/ablation/d1052_3b_capacity_only_w1/verification/verify_admission.py.snapshot',
                 ROOT/'paper_results/ieee_tc/ablation/d1127_7b_nohost_w1/compare.py',
                 ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md',
                 ROOT/'docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md'):
        witness(path)
    report = dict(kind='d1128_7b_capacity_only_w1_observation_v1', results=results,
        Full_relative_to_CapacityOnly=relative,
        primary_observations=dict(
            both_timing_feasible=all(r['timing_feasible'] for r in results.values()),
            both_within_budget=all(r['within_budget'] for r in results.values()),
            full_uses_less_gpu_time=relative['gpu_seconds_reduction'] > 0,
            full_p95_lower=relative['p95_ttft_reduction'] > 0,
            full_joint_not_lower=relative['joint_slo_difference_percentage_points'] >= 0,
            independent_correctness_qualified=False),
        matched_inputs=4000, output_digest_differences=output_differences,
        same_non_target_configuration=True, confidence_interval=None, independent_numerical_proof=False,
        final_ranking_eligible=False, source_refs=REFS, plan_sha256=p.check_plan(),
        execution_plan_sha256={v:x['plan_sha256'] for v,x in launches.items()},
        caveats=['Known seed42 development observations; not independent holdout or statistical inference.',
                 'Raw n_correct remains unknown; native binding and token contract are separate evidence.',
                 'Same-backend output changes retained, without an assumed numerical explanation.',
                 'CapacityOnly disables only proactive soft admission. Native victim/replacement, physical feasibility, demand loading, routing, handoff and residency remain enabled; natural closed-loop plans and activations may differ.',
                 'Routing checks confirm shared service-bin semantics, not counterfactual latency. Typed decisions, physical rejection, stale snapshots, reuse and absent receipts are separated. No soft-deficit trigger means no causal E(t) benefit attribution; same-candidate controlled S3 remains separate.',
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

