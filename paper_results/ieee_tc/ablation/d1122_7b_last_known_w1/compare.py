"""D1122: adapt D1121 7B comparator with D1062 LastKnown receipt auditing.

Reuses sealed source analyzers; no serving code, workload or historical result
is changed. Full remains canonical D1115. Only ranking-source sample-and-hold
changes; actual selected-source protection and all non-targets remain live.
"""
from collections import Counter
import csv
import hashlib
import json
import os
import subprocess
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
from scripts.ieee_tc.analyze_d297_full import last_known_routing_receipt
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


def event_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def stats(values):
    values = sorted(values)
    if not values:
        return dict(n=0, mean=None, p95=None, max=None)
    return dict(n=len(values), mean=sum(values)/len(values),
        p95=values[(95*len(values)+99)//100-1], max=values[-1])


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    group = p.cg_path()
    assert os.sched_getaffinity(0) == {2, 3, 26, 27}
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert (group/'memory.high').read_text().strip() == str(3*p.GIB)
    assert (group/'memory.max').read_text().strip() == str(4*p.GIB)
    assert (group/'memory.swap.max').read_text().strip() == '0'
    assert p.check_plan() == 'd7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83'
    configs = [yaml.safe_load(witness(ROOT/'configs/ieee_tc'/name, sha).read_text())
        for name, sha in (
            ('20261107_d1100_7b_busy_service_rate_w2.yaml',
             '386a34c84ec70e5d0d366f6d3bcaf0891536be503ba1359d583a93bd5ecf303f'),
            ('20261108_d1122_7b_last_known_w1.yaml',
             '41db0c10b351aad74e8b4d4cebf36c6d6bc2c6ee707b3d7bf7c2521e0d9dcba6'))]
    bin_ms = configs[0]['resource_coordination']['service_bin_ms']
    for cfg in configs:
        del cfg['experiment']['output_dir']
        del cfg['storage']['nvme_cache_dir']
        del cfg['storage']['host_cache_dir']
    assert all(c['resource_coordination']['min_instances'] == 1 for c in configs)
    assert all(c['resource_coordination']['max_instances'] == 4 for c in configs)
    assert all(c['resource_coordination']['ieee_handoff_policy'] == 'full' for c in configs)
    assert configs[1]['resource_coordination'].pop('ieee_routing_information_view') == 'last_known_source_sample_hold_v1'
    assert configs[1]['resource_coordination'].pop('ieee_routing_source_hold_s') == .5
    assert configs[0] == configs[1]
    budget = read(ROOT/'paper_results/ieee_tc/main_comparison/d594_reference_and_ttft/resident_references.json')['W1']['B_gpu_seconds']
    index = read(ROOT/'paper_results/ieee_tc/inputs/20260927_7b_materialized_content_index.json')
    content = {a['id']: hashlib.sha256(json.dumps(sorted(a['files'], key=lambda f:f['path']),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest() for a in index['artifacts']}
    results, contracts, mechanisms, launches, output_differences = {}, {}, {}, {}, []
    for variant, run_id, scenario, directory in (
        ('Full', 'D1115', 'tc_ieee_full', ROOT/'paper_results/ieee_tc/main_comparison/d1115_same_version_w1/result1'),
        ('LastKnown', 'D1122', 'tc_ieee_full', BASE/'result1')):
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
        dispatches, routing_rows, held_rows = [], [], []
        final_events = set()
        expected_strategy = 'service_bins'
        with source.open('rb') as f:
            for r in ijson.items(f, f'detailed_results.{scenario}.requests.item', use_float=True):
                routing = r['gpu_reference_evidence']['routing_snapshot']
                contract = routing['selection_contract']
                assert contract['owner_policy'] == 'ieee_confirmed'
                assert contract['strategy'] == expected_strategy
                assert contract['locality_load_slack'] is None
                if variant == 'LastKnown':
                    receipt = last_known_routing_receipt(r, delta_ms=bin_ms, hold_interval_s=.5)
                    held_rows.append(receipt)
                    final_events.add(event_sha(routing['ranking_information_view']))
                    routing_rows.append(dict(request_id=r['request_id'],
                        feasible_count=receipt['feasible_count'], actual=receipt['selected_replica_id'],
                        received_choice=receipt['received_view_choice'],
                        differs=receipt['choice_disagreement']))
                else:
                    candidates = tuple(ReplicaRoutingSnapshot(**(v | {
                        'active_adapters': frozenset(v['active_adapters']),
                        'service_class': ServiceObservationClass(**v['service_class']) if v['service_class'] else None,
                        'service': ServiceComponents(**v['service']) if v['service'] else None,
                    })) for v in routing['candidates'])
                    assert candidates and all(v.request_id == r['request_id'] and v.adapter_id == r['adapter_id'] for v in candidates)
                    full_choice = Router.select_ieee_snapshot(candidates, bin_ms)
                    locality_choice = Router.select_locality_snapshot(candidates, 1)
                    actual = full_choice
                    assert actual is not None and full_choice is not None and locality_choice is not None
                    assert actual.replica_id == routing['selected_replica_id'] == r['instance_id']
                    routing_rows.append(dict(request_id=r['request_id'], candidate_count=len(candidates),
                        feasible_count=sum(v.feasible for v in candidates), actual=actual.replica_id,
                        full_choice=full_choice.replica_id, locality_choice=locality_choice.replica_id,
                        differs=full_choice.replica_id != locality_choice.replica_id))
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
            with outcome.open('rb') as f:
                for plan in ijson.items(f, 'mechanism_events.'+key+'.item', use_float=True):
                    count += 1
                    reasons[plan.get('trigger_reason', 'not_recorded')] += 1
                    states[plan.get('state', 'not_recorded')] += 1
                    if key == '_ieee_gpu_preparation_plans':
                        assert plan['capacity_only'] is False
                        for attempt in plan['attempts']:
                            receipt = attempt.get('receipt')
                            if receipt and receipt.get('proactive_admission_evaluated') is True:
                                evaluated += 1
                                decision_reasons[receipt['admission']['reason']] += 1
            mechanism['plans'][key] = dict(count=count, reasons=dict(reasons), states=dict(states),
                evaluated_admission_attempts=evaluated, admission_decision_reasons=dict(decision_reasons))
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
        if variant == 'LastKnown':
            event_count, disagreement, previous = 0, 0, 0.
            phases, reasons, matched_events = Counter(), Counter(), Counter()
            with outcome.open('rb') as f:
                for event in ijson.items(f, 'mechanism_events._ieee_routing_information_events.item', use_float=True):
                    assert event['kind'] == 'last_known_source_sample_hold_v1' and event['hold_interval_s'] == .5
                    assert event['observed_monotonic_s'] >= previous
                    previous = event['observed_monotonic_s']
                    assert event['counterfactual_latency_generated'] is False
                    assert event['executable_reference_granted'] is False
                    event_count += 1
                    phases[event['phase']] += 1
                    disagreement += event['choice_disagreement']
                    reasons.update(r['reason'] for r in event['candidates'])
                    digest = event_sha(event)
                    if digest in final_events:
                        matched_events[digest] += 1
            assert len(final_events) == 4000 and set(matched_events) == final_events
            assert all(v == 1 for v in matched_events.values())
            mechanism['all_routing_event_journal'] = dict(count=event_count,
                phase_counts=dict(phases), reason_counts=dict(reasons),
                choice_disagreements=disagreement, matched_final_request_events=4000,
                all_event_history_EWMA_replayed=False)
            mechanism['final_request_routing'] = dict(requests=4000,
                choice_disagreements=sum(r['choice_disagreement'] for r in held_rows),
                selected_gpu_false_ready_at_acquisition=sum(r['selected_gpu_false_ready_at_acquisition'] for r in held_rows),
                selected_gpu_missed_ready_at_acquisition=sum(r['selected_gpu_missed_ready_at_acquisition'] for r in held_rows),
                prior_attempts=sum(r['prior_attempts'] for r in held_rows),
                feasible_candidates=sum(r['feasible_count'] for r in held_rows),
                source_class_disagreements=sum(c['source_class_disagreement'] for r in held_rows for c in r['candidates']),
                age_statistics={k:stats([c[k] for r in held_rows for c in r['candidates']]) for k in (
                    'controller_hold_age_s', 'native_observation_age_s', 'file_observation_age_s',
                    'source_observation_age_s', 'oldest_input_age_s')},
                unselected_physical_truth_known=False, counterfactual_latency_generated=False,
                caveat='Selected acquisition is later than routing; mismatch is not simultaneous physical truth or necessarily caused by sample-and-hold.')
            mechanism['last_known_receipts'] = held_rows
        mechanism['routing']['comparison'] = (
            'received Full versus held LastKnown on same state' if variant == 'LastKnown'
            else 'Full versus locality shadow on same state; different question, not comparable disagreement rates')
        mechanisms[variant] = mechanism
        # Actual recorded configuration, not the live legacy TTFT-only scoreboard.
        mechanism['feature_gates'] = coordination['feature_gates']
    assert launches['Full']['external_replay']['plan'] == launches['LastKnown']['external_replay']['plan']
    assert launches['Full']['preflight']['policy'] == launches['LastKnown']['preflight']['policy']
    assert contracts['Full'].keys() == contracts['LastKnown'].keys()
    for rid, a in contracts['Full'].items():
        b = contracts['LastKnown'][rid]
        matched_request_contract(a, b)
        if a['native']['completion_token_ids_sha256'] != b['native']['completion_token_ids_sha256']:
            output_differences.append(dict(request_id=rid,
                Full=a['native']['completion_token_ids_sha256'], LastKnown=b['native']['completion_token_ids_sha256']))
    full, elastic = results['Full'], results['LastKnown']
    relative = dict(gpu_seconds_reduction=(elastic['physical_gpu_seconds']-full['physical_gpu_seconds'])/elastic['physical_gpu_seconds'],
        joint_slo_difference_percentage_points=100*(full['joint_slo']-elastic['joint_slo']),
        p95_ttft_reduction=(elastic['metrics']['ttft_ms']['p95_type1']-full['metrics']['ttft_ms']['p95_type1'])/elastic['metrics']['ttft_ms']['p95_type1'])
    for path in (Path(__file__), ROOT/'scripts/ieee_tc/check_d299_full_context.py',
                 ROOT/'scripts/ieee_tc/audit_d389_scaleup_handoff_labels.py',
                 ROOT/'faaslora/experiment/instance_pool.py',
                 ROOT/'paper_results/ieee_tc/ablation/d1062_3b_last_known_w1/comparison/compare.py.snapshot',
                 ROOT/'paper_results/ieee_tc/ablation/d1121_7b_fixed_full_w1/compare.py',
                 ROOT/'scripts/ieee_tc/analyze_d297_full.py',
                 ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md',
                 ROOT/'docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md'):
        witness(path)
    report = dict(kind='d1122_7b_last_known_w1_observation_v1', results=results,
        Full_relative_to_LastKnown=relative,
        matched_inputs=4000, output_digest_differences=output_differences,
        same_non_target_configuration=True, hold_interval_s=.5, confidence_interval=None, independent_numerical_proof=False,
        final_ranking_eligible=False, source_refs=REFS, plan_sha256=p.check_plan(),
        caveats=['Known seed42 development observations; not independent holdout or statistical inference.',
                 'Raw n_correct remains unknown; native binding and token contract are separate evidence.',
                 'Same-backend output changes retained, without an assumed numerical explanation.',
                 'Only ranking source classes are held for 0.5 s; all actual authority, feasibility and non-target signals remain live.',
                 'Full is cached historical confirmed provenance, not globally instantaneous physical truth.',
                 'The request-triggered hold interval is not a bound on underlying source age.',
                 'Snapshot shadow decisions do not generate counterfactual latency.',
                 'All resource and timing trade-offs are retained; no winner is preconditioned by this analyzer.'])
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT/'comparison.json', report)
    save(OUT/'mechanisms.json', mechanisms)
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
            final_request_routing=m.get('final_request_routing'),
            plans={k:p['count'] for k,p in m['plans'].items()}) for v,m in mechanisms.items()}), indent=2))


if __name__ == '__main__':
    main()
