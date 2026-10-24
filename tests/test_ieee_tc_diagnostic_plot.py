"""Offline timing/plot guards; no GPU or scientific-effect qualification."""
import copy
import json
from pathlib import Path

import pytest


def d823_pair():
    from scripts.ieee_tc.audit_d703_3b_full import D716_TTFT_STAGES
    rows = [dict(request_id=f'r{i}', ordinal=i, phase=0, adapter_id=f'a{i}',
        prompt_tokens=24, output_tokens=64, output_bin='2-64', tier='gpu',
        ttft_pass=True, tpot_pass=True, joint_pass=True,
        ttft_ms=60., tpot_ms=5., e2e_ms=375., routing_nested_ms=3.,
        final_native_load_ms=None, peer_prefill_fraction=0.,
        peer_demand_load_fraction=0., peer_decode_fraction=.2,
        **{key:10. for key in D716_TTFT_STAGES}) for i in range(4)]
    rows[0].update(tpot_pass=False, joint_pass=False)
    newer = copy.deepcopy(rows)
    newer[0].update(tpot_pass=True, joint_pass=True)
    newer[1].update(tpot_pass=False, joint_pass=False, tpot_ms=6.)
    newer[2].update(tpot_pass=False, joint_pass=False, tpot_ms=8., tier='host')
    newer[2]['admission_to_acquired_ms'] += 4.
    newer[2]['ttft_ms'] += 4.
    return rows, newer


def test_d823_pair_retains_population_and_reconciles_changes_without_mutation():
    from scripts.ieee_tc.audit_d703_3b_full import paired_stage_groups, D716_TTFT_STAGES
    pair = d823_pair()
    before = copy.deepcopy(pair)
    result = paired_stage_groups(*pair)
    assert pair == before
    assert result['transitions']['joint_pass'] == {'fail->pass':1, 'pass->fail':2, 'pass->pass':1}
    all_group = result['groups'][0]
    assert all_group['n'] == 4
    assert all_group['mean_paired_delta']['ttft_ms'] == 1.
    assert 'final_native_load_ms' not in all_group['mean_paired_delta']
    assert sum(all_group['mean_paired_delta'][k] for k in D716_TTFT_STAGES) == 1.
    assert all_group['reference']['counts']['joint_pass'] == 3
    assert all_group['candidate']['counts']['joint_pass'] == 2
    assert result['confidence_interval'] is None and result['causal_performance_gain'] is None
    assert all(g['grouping'] == 'post_treatment' for g in result['groups'] if 'transition' in g['dimension'])
    assert {r['request_id'] for r in result['requests']} == {'r0','r1','r2','r3'}


@pytest.mark.parametrize('fault', ['missing','duplicate','adapter','tokens','boolean','nan','stage','joint'])
def test_d823_rejects_misaligned_or_invalid_pair(fault):
    from scripts.ieee_tc.audit_d703_3b_full import paired_stage_groups
    old, new = d823_pair()
    if fault == 'missing': new.pop()
    elif fault == 'duplicate': new.append(copy.deepcopy(new[0]))
    elif fault == 'adapter': new[0]['adapter_id'] = 'other'
    elif fault == 'tokens': new[0]['output_tokens'] += 1
    elif fault == 'boolean': new[0]['tpot_pass'] = 'True'
    elif fault == 'nan': new[0]['tpot_ms'] = float('nan')
    elif fault == 'stage': new[0]['ttft_ms'] += 2.
    elif fault == 'joint': new[1]['joint_pass'] = True
    with pytest.raises(ValueError):
        paired_stage_groups(old, new)


def test_d823_existing_output_cannot_be_overwritten(tmp_path):
    from scripts.ieee_tc.audit_d703_3b_full import compare_stage_audits
    with pytest.raises(FileExistsError):
        compare_stage_audits(tmp_path/'old', tmp_path/'new', tmp_path)


def d721_rows():
    from scripts.ieee_tc.audit_d703_3b_full import D716_TTFT_STAGES
    def row(rid, owner, start, first, last, tokens, load=False):
        return dict(request_id=rid, owner=owner, worker_pid=1 if owner == 'a' else 2,
            replica=owner, scheduled=start, first=first, last=last, output_tokens=tokens,
            native_load_invoked=load, load_start=.9 if load else None, load_end=1.15 if load else None,
            ttft_pass=True, tpot_pass=rid != 'r0', joint_pass=rid != 'r0',
            ttft_ms=60., tpot_ms=(last-first)*1000/(tokens-1),
            peer_prefill_count=0, peer_prefill_overlap_ms=0.,
            peer_demand_load_count=0, peer_demand_load_overlap_ms=0.,
            **{key:10. for key in D716_TTFT_STAGES})
    a = row('r0', 'a', .5, 1., 2., 10)
    b = row('r1', 'a', 1.1, 1.2, 1.8, 100, True)
    c = row('r2', 'b', 1.1, 1.2, 1.8, 10, True)
    a.update(peer_prefill_count=1, peer_prefill_overlap_ms=100.,
             peer_demand_load_count=1, peer_demand_load_overlap_ms=150.)
    return [a, b, c]


def test_d721_disjoint_exposure_counts_and_cross_owner_exclusion():
    from scripts.ieee_tc.audit_d703_3b_full import decode_overlap_groups
    groups = decode_overlap_groups(d721_rows())
    assert len(groups) == 45
    index = {(g['output_length'], g['tpot_status'], g['peer_exposure']):g for g in groups}
    assert index['all','all','all']['n'] == 3
    assert index['all','all','both']['n'] == 1
    assert index['all','all','neither']['n'] == 2
    assert index['2-64','tpot_fail','both']['tpot_violations'] == 1
    empty = index['65+','tpot_fail','all']
    assert empty['n'] == 0 and empty['metrics']['tpot_ms']['mean'] is None
    for length in ('all', '2-64', '65+'):
        for status in ('all', 'tpot_pass', 'tpot_fail'):
            assert sum(index[length,status,e]['n'] for e in (
                'neither','prefill_only','demand_load_only','both')) == index[length,status,'all']['n']


@pytest.mark.parametrize('fault', ['empty','duplicate','missing','boolean','joint','tokens',
    'nan','order','tpot','ttft_identity','owner','count','duration','missing_load','missing_span','partial_span'])
def test_d721_rejects_missing_or_contradictory_overlap_evidence(fault):
    from scripts.ieee_tc.audit_d703_3b_full import decode_overlap_groups
    rows = d721_rows()
    if fault == 'empty': rows = []
    elif fault == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    elif fault == 'missing': del rows[0]['last']
    elif fault == 'boolean': rows[0]['tpot_pass'] = 'False'
    elif fault == 'joint': rows[0]['joint_pass'] = True
    elif fault == 'tokens': rows[0]['output_tokens'] = 1
    elif fault == 'nan': rows[0]['first'] = float('nan')
    elif fault == 'order': rows[0]['scheduled'] = 3.
    elif fault == 'tpot': rows[0]['tpot_ms'] += 1.
    elif fault == 'ttft_identity': rows[0]['ttft_ms'] += 2.
    elif fault == 'owner': rows[1]['worker_pid'] = 99
    elif fault == 'count': rows[0]['peer_prefill_count'] = 2
    elif fault == 'duration': rows[0]['peer_prefill_overlap_ms'] += 2.
    elif fault == 'missing_load': rows[1]['native_load_invoked'] = False
    elif fault == 'missing_span': rows[1].update(load_start=None, load_end=None)
    elif fault == 'partial_span': rows[1]['load_end'] = None
    with pytest.raises((ValueError, KeyError)):
        decode_overlap_groups(rows)


def d717_request():
    opened = dict(intent_id='p',state='pending',clock_id='clock',scheduler_owner_id='scheduler',observed_at=1.)
    final = dict(opened,state='withdrawn',observed_at=1.1)
    attempt = dict(native_source_conflict='faster native source appeared before admission',
        routing_snapshot=dict(selected_reference_acquired=False,selected_replica_id='replica',sources={
            'replica':dict(source=dict(tier='nvme',native=False),native_owner_id='owner',native_epoch=1,
                native_captured_monotonic_s=.5,source_collection=dict(mode='confirmed_snapshot_cache_v1',
                    frontier_absence_replica_indexes=[0]))}),
        pending_kv_admission=dict(intent_id='p',state='closed',receipt=opened,
            close_receipt=dict(intent_id='p',closed=True,receipt=final)))
    return dict(request_id='r',gpu_reference_evidence=dict(prior_routing_attempts=[attempt]))


def test_d717_preserves_zero_attempts_and_counts_repeated_negative_views():
    from scripts.ieee_tc.audit_d703_3b_full import retained_routing_attempts
    assert retained_routing_attempts(dict(gpu_reference_evidence={}),clock_id='clock') == []
    request = d717_request()
    request['gpu_reference_evidence']['prior_routing_attempts'] *= 2
    rows = retained_routing_attempts(request,clock_id='clock')
    assert [r['repeated_absence_view'] for r in rows] == [False,True]
    assert rows[0]['pending_residence_ms'] == pytest.approx(100.)


@pytest.mark.parametrize('fault',['clock','pending','indexes','native','acquired','negative','type'])
def test_d717_rejects_incomplete_or_contradictory_attempt_evidence(fault):
    from scripts.ieee_tc.audit_d703_3b_full import retained_routing_attempts
    request = d717_request()
    a = request['gpu_reference_evidence']['prior_routing_attempts'][0]
    if fault == 'clock': a['pending_kv_admission']['receipt']['clock_id'] = 'other'
    elif fault == 'pending': a['pending_kv_admission']['state'] = 'pending'
    elif fault == 'indexes': a['routing_snapshot']['sources']['replica']['source_collection']['frontier_absence_replica_indexes'] = [1]
    elif fault == 'native': a['routing_snapshot']['sources']['replica']['source']['native'] = True
    elif fault == 'acquired': a['routing_snapshot']['selected_reference_acquired'] = True
    elif fault == 'negative': a['pending_kv_admission']['close_receipt']['receipt']['observed_at'] = .1
    elif fault == 'type': request['gpu_reference_evidence']['prior_routing_attempts'] = None
    with pytest.raises(ValueError):
        retained_routing_attempts(request,clock_id='clock')


def d719_file_retry(tier='nvme'):
    request = d717_request()
    attempt = request['gpu_reference_evidence']['prior_routing_attempts'][0]
    del attempt['native_source_conflict']
    attempt['file_source_conflict'] = ('Remote source view changed before admission'
        if tier == 'remote' else 'confirmed file source owner/epoch changed')
    selected = attempt['routing_snapshot']['sources']['replica']
    selected.update(file_owner_id='file-owner', file_epoch=7)
    selected['source'].update(tier=tier, owner_id='file-owner', epoch=7)
    return request


@pytest.mark.parametrize('tier', ['host', 'nvme', 'remote'])
def test_d719_counts_file_rejection_without_inventing_native_host_receipt(tier):
    from scripts.ieee_tc.audit_d703_3b_full import retained_routing_attempts
    rows = retained_routing_attempts(d719_file_retry(tier), clock_id='clock')
    assert len(rows) == 1 and rows[0]['reason'].startswith('file:')
    assert rows[0]['pending_residence_ms'] == pytest.approx(100.)


@pytest.mark.parametrize('fault', ['missing', 'unknown', 'native', 'tier', 'owner',
    'epoch', 'epoch_bool', 'lease', 'host_receipt', 'clock', 'pending'])
def test_d719_file_rejection_requires_matching_source_and_closed_pending(fault):
    from scripts.ieee_tc.audit_d703_3b_full import retained_routing_attempts
    request = d719_file_retry()
    a = request['gpu_reference_evidence']['prior_routing_attempts'][0]
    s = a['routing_snapshot']['sources']['replica']['source']
    if fault == 'missing': del a['file_source_conflict']
    elif fault == 'unknown': a['file_source_conflict'] = 'unknown'
    elif fault == 'native': s['native'] = True
    elif fault == 'tier': s['tier'] = 'remote'
    elif fault == 'owner': s['owner_id'] = 'different'
    elif fault == 'epoch': s['epoch'] += 1
    elif fault == 'epoch_bool': s['epoch'] = True
    elif fault == 'lease': a['local_source_reference'] = {}
    elif fault == 'host_receipt': a['native_host_source'] = {}
    elif fault == 'clock': a['pending_kv_admission']['receipt']['clock_id'] = 'foreign'
    elif fault == 'pending': a['pending_kv_admission']['state'] = 'pending'
    with pytest.raises(ValueError):
        retained_routing_attempts(request, clock_id='clock')


@pytest.mark.parametrize('fault', ['count', 'run', 'passed', 'hash', 'output_exists'])
def test_d719_profile_audit_rejects_unqualified_or_overwritten_sources(tmp_path, monkeypatch, fault):
    from scripts.ieee_tc import audit_d703_3b_full as audit
    raw, output = tmp_path/'run', tmp_path/'audit.json'
    metric = dict(source_run=str(raw), passed=True, counts={'n':4000},
        source_refs=[dict(path=str(tmp_path/'experiment_results_full.json'), sha256='sealed')])
    if fault == 'count': metric['counts']['n'] = 100
    elif fault == 'run': metric['source_run'] = 'other'
    elif fault == 'passed': metric['passed'] = False
    elif fault == 'output_exists': output.write_text('protected')
    monkeypatch.setattr(audit, 'read', lambda _: metric)
    monkeypatch.setattr(audit, 'digest', lambda _: 'changed')
    with pytest.raises(FileExistsError if fault == 'output_exists' else ValueError):
        audit.audit_profile_bindings(raw, output, tmp_path/'metric.json', tmp_path/'config.yaml')
    if fault == 'output_exists': assert output.read_text() == 'protected'
    else: assert not output.exists()


def d716_request():
    n = dict(timing_contract='ieee_tc_native_v1', native_clock_id='clock',
        controller_admitted_monotonic_s=1., native_dispatch_monotonic_s=2.,
        native_queued_monotonic_s=2.1, native_scheduled_monotonic_s=2.2,
        native_first_token_monotonic_s=3., native_last_token_monotonic_s=4.,
        worker_completed_monotonic_s=4.1, controller_completed_monotonic_s=4.2,
        native_output_tokens=3, native_tpot_ms=500., actual_prompt_tokens=20,
        gpu_reference_owner_id='owner', gpu_reference_lease_id='lease')
    receipt = dict(acquired=True, clock_id='clock', owner_id='owner', lease_id='lease',
        acquired_monotonic_s=1.5, native_load_invoked=True,
        native_load_started_monotonic_s=1.2, native_load_completed_monotonic_s=1.5,
        source_tier_before_acquisition='host', worker_pid=123)
    evidence = dict(source_admission=dict(source={'tier':'host'}, clock_id='clock',
        admitted_monotonic_s=1.), receipt=receipt, stale_rechecks=0,
        preparation_interval=dict(request_id='r0', native_lease_id='lease', reason='demand'))
    return dict(request_id='r0', adapter_id='a', instance_id='replica', success=True,
        readiness_tier_before_dispatch='host', native_token_timing=n,
        external_arrival_timing=dict(planned_arrival_s=.5, index=0), gpu_reference_evidence=evidence,
        completion_tokens=3, requested_completion_tokens=3, overall_ttft_ms=2500.,
        tpot_ms=500., overall_e2e_ms=3700., arrival_release_lateness_ms=100.,
        dispatch_window_wait_ms=300., runtime_slot_wait_ms=100., control_path_total_us=1234.)


@pytest.mark.parametrize('tt, tp, label', [(3000.,600.,'pass'),(2000.,600.,'ttft_only'),
                                         (3000.,400.,'tpot_only'),(2000.,400.,'both')])
def test_d716_partition_retains_all_joint_outcomes(tt, tp, label):
    from scripts.ieee_tc.audit_d703_3b_full import partition_common_slo_request, D716_TTFT_STAGES
    groups = [dict(lower_exclusive=None,upper_inclusive=None,ttft_ms=tt,tpot_ms=tp)]
    row = partition_common_slo_request(d716_request(), groups, clock_id='clock')
    assert row['slo_group'] == label
    assert sum(row[k] for k in D716_TTFT_STAGES) == pytest.approx(2500.)
    assert row['before_final_load_ms'] == pytest.approx(200.)
    assert row['final_native_load_ms'] == pytest.approx(300.)
    assert row['routing_nested_ms'] == 1.234


@pytest.mark.parametrize('fault', ['clock','missing','ingress','lease','count','metric'])
def test_d716_partition_rejects_incomplete_or_mismatched_evidence(fault):
    from scripts.ieee_tc.audit_d703_3b_full import partition_common_slo_request
    row = d716_request()
    if fault == 'clock': row['native_token_timing']['native_clock_id'] = 'foreign'
    elif fault == 'missing': del row['gpu_reference_evidence']['receipt']['native_load_started_monotonic_s']
    elif fault == 'ingress': row['runtime_slot_wait_ms'] += 2
    elif fault == 'lease': row['gpu_reference_evidence']['receipt']['lease_id'] = 'wrong'
    elif fault == 'count': row['requested_completion_tokens'] = 4
    elif fault == 'metric': row['overall_ttft_ms'] += 2
    with pytest.raises((ValueError,KeyError)):
        partition_common_slo_request(row, [dict(lower_exclusive=None,upper_inclusive=None,
            ttft_ms=3000., tpot_ms=600.)], clock_id='clock')


def test_d716_overlap_is_union_and_not_double_counted():
    from scripts.ieee_tc.audit_d703_3b_full import clipped_interval_union
    assert clipped_interval_union(1.,4.,[(0.,2.),(1.5,3.),(3.5,5.),(4.,5.)]) == (2.5,3)
    assert clipped_interval_union(1.,4.,[]) == (0.,0)
    assert clipped_interval_union(1.,1.,[(0.,2.)]) == (0.,0)


@pytest.mark.parametrize('span', [(2.,1.),(float('nan'),2.),(0.,float('inf'))])
def test_d716_overlap_rejects_unknown_clock_spans(span):
    from scripts.ieee_tc.audit_d703_3b_full import clipped_interval_union
    with pytest.raises(ValueError):
        clipped_interval_union(1.,4.,[span])

from scripts.ieee_tc.plot_d290_diagnostic import (binding_checks, load, percentile,
    request_vectors, write_numerical_repeat_audit, audit_elastic_locality_mechanisms)


def locality_evidence():
    contract = dict(system='ElasticLocality',variant='elastic_locality_v1',
        proactive_preparation=False,native_demand_loading=True,physical_admission=True,
        selection_strategy='gpu_locality',file_cache_policy='demand_lru',
        handoff_policy='no_handoff',locality_load_slack=1)
    selection = dict(schema='owned_routing_selection_v1',owner_policy='ieee_confirmed',
        strategy='gpu_locality',locality_load_slack=1)
    metrics = dict(ieee_mechanism_contract=contract,
        ieee_routing_selection=selection | {'locality_attempt_reasons':{'gpu_hit':1}},
        ieee_file_cache=dict(file_cache_policy='demand_lru',plans=[],replacements=[]))
    events = {key:None for key in ('_ieee_residency_epochs','_ieee_file_preparation_plans',
        '_ieee_gpu_preparation_plans','_ieee_native_host_preparations','_ieee_proactive_diagnostic')}
    events['_ieee_activations'] = [dict(policy='no_handoff',plan_sha256=None,
        preparation_state='disabled')]
    outcome = dict(scenario='tc_ieee_elastic_locality',coordination_after_shutdown=metrics,
        launch_contract={'system_mechanism_contract':copy.deepcopy(contract)},
        mechanism_events=events,planning_cpu_events=[])
    rows = [dict(gpu_reference_evidence={'routing_snapshot':{'selection_contract':selection}})]
    return outcome,rows


def test_locality_audit_requires_actual_mechanisms_not_label():
    outcome, rows = locality_evidence()
    result = audit_elastic_locality_mechanisms(outcome,rows)
    assert result['passed'] and result['requests'] == 1
    assert not result['numerical_correctness_qualified']
    assert not result['performance_qualified']


@pytest.mark.parametrize('key', ['_ieee_residency_epochs','_ieee_file_preparation_plans',
    '_ieee_gpu_preparation_plans','_ieee_native_host_preparations','_ieee_proactive_diagnostic'])
def test_locality_audit_rejects_proactive_work_or_missing_journal(key):
    outcome,rows = locality_evidence()
    outcome['mechanism_events'][key] = [{'state':'completed'}]
    with pytest.raises(ValueError):
        audit_elastic_locality_mechanisms(outcome,rows)
    del outcome['mechanism_events'][key]
    with pytest.raises(KeyError):
        audit_elastic_locality_mechanisms(outcome,rows)


@pytest.mark.parametrize('fault', ['full_label','no_activation','handoff','planner_cpu',
    'file_plan','failed_replacement','wrong_router','no_attempt','request_strategy', 'slack_bool'])
def test_locality_audit_rejects_partial_or_mixed_variant(fault):
    outcome, rows = locality_evidence()
    metrics = outcome['coordination_after_shutdown']
    if fault == 'full_label': outcome['scenario'] = 'tc_ieee_full'
    elif fault == 'no_activation': outcome['mechanism_events']['_ieee_activations'] = []
    elif fault == 'handoff': outcome['mechanism_events']['_ieee_activations'][0]['plan_sha256'] = 'plan'
    elif fault == 'planner_cpu': outcome['planning_cpu_events'] = [{'op':'plan'}]
    elif fault == 'file_plan': metrics['ieee_file_cache']['plans'] = [{'plan_id':'p'}]
    elif fault == 'failed_replacement':
        metrics['ieee_file_cache']['replacements'] = [{'policy':'demand_lru','state':'reclamation_failed'}]
    elif fault == 'wrong_router': metrics['ieee_routing_selection']['strategy'] = 'service_bins'
    elif fault == 'no_attempt': metrics['ieee_routing_selection']['locality_attempt_reasons'] = {}
    elif fault == 'request_strategy':
        rows[0]['gpu_reference_evidence']['routing_snapshot']['selection_contract']['strategy'] = 'service_bins'
    elif fault == 'slack_bool':
        metrics['ieee_mechanism_contract']['locality_load_slack'] = True
        outcome['launch_contract']['system_mechanism_contract']['locality_load_slack'] = True
    with pytest.raises(ValueError):
        audit_elastic_locality_mechanisms(outcome,rows)


def request():
    return {
        "request_id": "r0", "success": True,
        "generation_contract": "fixed_length_greedy_v1",
        "output_contract_match": True, "completion_token_source": "vllm_token_ids",
        "completion_tokens": 2, "requested_completion_tokens": 2,
        "readiness_tier_before_dispatch": "HOST",
        "arrival_release_lateness_ms": 5.0,
        "dispatch_admission_wait_ms": 20.0,
        "service_ttft_ms": 10.0, "overall_ttft_ms": 30.0,
    }


def test_lateness_is_not_added_twice():
    assert request_vectors([request()]) == ([20.0], [10.0], [30.0])


@pytest.mark.parametrize("field,value", [
    ("overall_ttft_ms", 35.0), ("service_ttft_ms", float("nan")),
    ("dispatch_admission_wait_ms", -1.0),
    ("completion_token_source", "expected"), ("completion_tokens", 1),
    ("readiness_tier_before_dispatch", ""), ("output_contract_match", False),
])
def test_missing_or_inconsistent_data_is_not_filled(field, value):
    row = request()
    row[field] = value
    with pytest.raises(ValueError):
        request_vectors([row])


def test_missing_timing_rejected():
    row = request()
    del row["service_ttft_ms"]
    with pytest.raises(KeyError):
        request_vectors([row])


def test_duplicates_and_empty_rejected():
    with pytest.raises(ValueError):
        request_vectors([request(), request()])
    with pytest.raises(ValueError):
        request_vectors([])


def test_type_one_quantile():
    assert percentile([10, 20, 30, 40], 0.5) == 20
    assert percentile([10, 20, 30, 40], 0.95) == 40
    with pytest.raises(ValueError):
        percentile([], 0.95)


def test_common_slo_is_joint_and_not_legacy_five_seconds():
    from scripts.ieee_tc.check_d299_full_context import common_slo_flags
    groups = [dict(lower_exclusive=None, upper_inclusive=506, ttft_ms=1466., tpot_ms=30.),
              dict(lower_exclusive=506, upper_inclusive=None, ttft_ms=2540., tpot_ms=36.)]
    # Both would pass the legacy 5000-ms aggregate; neither passes joint SLO.
    a = common_slo_flags(groups, prompt_tokens=506, output_tokens=2, ttft_ms=2000., tpot_ms=20.)
    b = common_slo_flags(groups, prompt_tokens=507, output_tokens=2, ttft_ms=1000., tpot_ms=40.)
    assert a == dict(ttft_pass=False, tpot_pass=True, joint_pass=False)
    assert b == dict(ttft_pass=True, tpot_pass=False, joint_pass=False)
    assert common_slo_flags(groups, prompt_tokens=507, output_tokens=2,
                            ttft_ms=2540., tpot_ms=36.)['joint_pass']


def test_common_slo_single_token_is_not_zero_tpot():
    from scripts.ieee_tc.check_d299_full_context import common_slo_flags
    groups = [dict(lower_exclusive=None, upper_inclusive=None, ttft_ms=10., tpot_ms=1.)]
    assert common_slo_flags(groups, prompt_tokens=1, output_tokens=1, ttft_ms=10., tpot_ms=None) == {
        'ttft_pass': True, 'tpot_pass': None, 'joint_pass': True}
    with pytest.raises(ValueError, match='N/A'):
        common_slo_flags(groups, prompt_tokens=1, output_tokens=1, ttft_ms=10., tpot_ms=0.)


@pytest.mark.parametrize('override', [dict(prompt_tokens=0), dict(prompt_tokens=True),
    dict(output_tokens=0), dict(output_tokens=True), dict(ttft_ms=float('nan')),
    dict(ttft_ms=-1.), dict(tpot_ms=None), dict(tpot_ms=float('inf'))])
def test_common_slo_rejects_missing_or_nonfinite_native_values(override):
    from scripts.ieee_tc.check_d299_full_context import common_slo_flags
    groups = [dict(lower_exclusive=None, upper_inclusive=None, ttft_ms=10., tpot_ms=1.)]
    values = dict(prompt_tokens=1, output_tokens=2, ttft_ms=10., tpot_ms=.5)
    with pytest.raises(ValueError):
        common_slo_flags(groups, **(values | override))


def test_common_slo_rejects_missing_or_overlapping_strata():
    from scripts.ieee_tc.check_d299_full_context import common_slo_flags
    g = dict(lower_exclusive=None, upper_inclusive=None, ttft_ms=10., tpot_ms=1.)
    values = dict(prompt_tokens=1, output_tokens=2, ttft_ms=10., tpot_ms=.5)
    for groups in ([], [g,g], [g | {'ttft_ms': float('nan')}], [g | {'tpot_ms': 0.}]):
        with pytest.raises(ValueError):
            common_slo_flags(groups, **values)


@pytest.mark.parametrize("count", [100, 1000, 4000])
def test_registered_common_slo_population_still_requires_completed_source(tmp_path, count):
    from scripts.ieee_tc.audit_d703_3b_full import reevaluate_common_slo
    with pytest.raises(ValueError, match="exactly one completed source result required"):
        reevaluate_common_slo(tmp_path, tmp_path / "audit.json", expected_count=count)
    assert not (tmp_path / "audit.json").exists()


@pytest.fixture
def d813_arrival_source(tmp_path):
    path = tmp_path/'source.json'
    path.write_text(json.dumps({'requests': [dict(request_id=f'r{i:04d}',
        arrival_time_s=i*2., adapter_id='adapter', expected_output_tokens=8)
        for i in range(501)]}))
    return path


@pytest.mark.parametrize('profile', ['W0', 'W1'])
def test_d813_reconstructs_recorded_arrival_view(d813_arrival_source, profile):
    from faaslora.datasets.workload_generator import FrozenReplayPlan
    from scripts.ieee_tc.audit_d703_3b_full import registered_arrival_plan
    source = FrozenReplayPlan.load(d813_arrival_source, profile=profile)
    restored = registered_arrival_plan(source.identity(), expected_count=501)
    assert restored.identity() == source.identity()
    assert restored.entries[-1].offset_s == (529. if profile == 'W1' else 1000.)


@pytest.mark.parametrize('field,value', [('profile','W0'), ('profile','W2'),
    ('profile','unknown'), ('view_sha256','0'*64), ('source_sha256','0'*64),
    ('count',500), ('rate_scale',2.), ('source_count',500)])
def test_d813_rejects_changed_arrival_identity(d813_arrival_source, field, value):
    from faaslora.datasets.workload_generator import FrozenReplayPlan
    from scripts.ieee_tc.audit_d703_3b_full import registered_arrival_plan
    identity = FrozenReplayPlan.load(d813_arrival_source, profile='W1').identity()
    identity[field] = value
    with pytest.raises(ValueError):
        registered_arrival_plan(identity, expected_count=501)


@pytest.fixture
def d814_w2_plan():
    from faaslora.datasets.workload_generator import FrozenReplayPlan
    root = Path(__file__).resolve().parents[1]
    mapping = root/'paper_results/ieee_tc/inputs/20261018_3b_w2_adapter_map_v1.json'
    source = json.loads(mapping.read_text())['source_path']
    return FrozenReplayPlan.load(source, profile='W2', adapter_map=mapping,
        adapter_map_sha256='28c97dc7c1ffcbad49801ec19bec2a9c4cf9c4915c88a7e5163e223ece315e23')


def test_d814_reconstructs_content_bound_w2(d814_w2_plan):
    from scripts.ieee_tc.audit_d703_3b_full import registered_arrival_plan
    restored = registered_arrival_plan(d814_w2_plan.identity(), expected_count=4000)
    assert restored.entries == d814_w2_plan.entries
    assert restored.identity() == d814_w2_plan.identity()


@pytest.mark.parametrize('field,value', [('profile','W0'), ('source_sha256','0'*64),
    ('view_sha256','0'*64), ('adapter_map_sha256','0'*64), ('rate_scale',2.), ('count',3999)])
def test_d814_rejects_changed_w2_identity(d814_w2_plan, field, value):
    from scripts.ieee_tc.audit_d703_3b_full import registered_arrival_plan
    identity = d814_w2_plan.identity()
    identity[field] = value
    with pytest.raises(ValueError):
        registered_arrival_plan(identity, expected_count=4000)


def test_d814_rejects_missing_w2_map(d814_w2_plan):
    from scripts.ieee_tc.audit_d703_3b_full import registered_arrival_plan
    identity = d814_w2_plan.identity()
    del identity['adapter_map']
    with pytest.raises(ValueError):
        registered_arrival_plan(identity, expected_count=4000)


@pytest.mark.parametrize("count", [0, 99, 101, 3999])
def test_unregistered_common_slo_population_rejected(tmp_path, count):
    from scripts.ieee_tc.audit_d703_3b_full import reevaluate_common_slo
    with pytest.raises(ValueError, match="registered prefix100/prefix1000 or full4000 only"):
        reevaluate_common_slo(tmp_path, tmp_path / "audit.json", expected_count=count)


def test_incomplete_replay_rejected(tmp_path):
    result = {"metadata": {"total_requests": 1}, "detailed_results": {
        "one": {"completed": 1, "failed": 0, "requests": [request()]}}}
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    assert load(path)[1]["completed"] == 1
    incomplete = copy.deepcopy(result)
    incomplete["metadata"]["total_requests"] = 2
    path.write_text(json.dumps(incomplete))
    with pytest.raises(ValueError):
        load(path)
    result["detailed_results"]["two"] = result["detailed_results"]["one"]
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        load(path)


def binding_row():
    return {
        "adapter_id": "a", "completion_token_ids_sha256": "tokens",
        "native_token_timing": {
            "gpu_reference_owner_id": "o", "gpu_reference_lease_id": "l",
            "gpu_reference_adapter_int_id": 1, "native_clock_id": "clock",
            "native_dispatch_monotonic_s": 3, "completion_token_ids_sha256": "tokens"},
        "gpu_reference_evidence": {
            "state": "released",
            "intent": {"lora_name": "a", "expected_owner_id": "o", "lease_id": "l",
                       "adapter_int_id": 1, "expected_content_sha256": "content"},
            "receipt": {"lora_name": "a", "owner_id": "o", "lease_id": "l",
                        "adapter_int_id": 1, "slot": 0, "expected_content_sha256": "content",
                        "acquired": True, "clock_id": "clock", "acquired_monotonic_s": 2,
                        "post_load_source_snapshot": {"owner_id": "o", "clock_id": "clock",
                            "slot_adapter_ids": [1], "sources": [{"adapter_id": "a",
                            "adapter_int_id": 1, "gpu_slot": 0, "content_sha256": "content"}]}},
            "release_receipt": {"released": True, "owner_id": "o"}},
    }


def test_binding_matches_but_is_not_arithmetic_qualification():
    assert all(binding_checks(binding_row(), "content").values())


@pytest.mark.parametrize("field,bad_value,failed_check", [
    ("gpu_reference_owner_id", "wrong", "owner_binding"),
    ("gpu_reference_lease_id", "wrong", "lease_binding"),
    ("gpu_reference_adapter_int_id", 0, "native_adapter_binding"),
    ("native_clock_id", "other", "common_native_clock"),
    ("native_dispatch_monotonic_s", 1, "acquired_before_dispatch"),
    ("completion_token_ids_sha256", "wrong", "token_digest_transport"),
])
def test_binding_negatives(field, bad_value, failed_check):
    row = binding_row()
    row["native_token_timing"][field] = bad_value
    assert not binding_checks(row, "content")[failed_check]


def test_expected_content_is_independent_and_slot_is_checked():
    row = binding_row()
    assert not binding_checks(row, "independent_wrong")["frozen_content_binding"]
    row["gpu_reference_evidence"]["receipt"]["post_load_source_snapshot"]["slot_adapter_ids"] = [7]
    assert not binding_checks(row, "content")["slot_binding"]


def test_numerical_audit_rejects_duplicate_or_empty_blocks(tmp_path):
    path = tmp_path / 'not_read.json'
    for paths in ([], [path, path], [path] * 4):
        with pytest.raises(ValueError, match='distinct'):
            write_numerical_repeat_audit(paths, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_numerical_audit_rejects_incomplete_before_publishing(tmp_path):
    path = tmp_path / 'partial.json'
    path.write_text(json.dumps({'kind': 'backend_native_native_numerical_repeat_qualification_v1',
                                'pass': False, 'stage': 'failed'}))
    (tmp_path / 'launch.json').write_text(json.dumps({'pass': False}))
    with pytest.raises(ValueError, match='incomplete'):
        write_numerical_repeat_audit([path], tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('start,end,expected', [
    (0., 10., [1000., 3000., 6000.]),
    (2., 8., [0., 2000., 4000.]),
    (5., 8., [0., 0., 3000.]),
    (0., .5, [500., 0., 0.]),
])
def test_d818_clipped_sequential_partition(start, end, expected):
    from scripts.ieee_tc.audit_d703_3b_full import clipped_sequential_stages
    result = clipped_sequential_stages(start, end, [('before_ms', 1.), ('work_ms', 4.)])
    assert list(result.values()) == expected
    assert sum(result.values()) == pytest.approx((end-start)*1000)


@pytest.mark.parametrize('start,end,boundaries', [
    (1., 0., [('a', 1.)]),
    (0., 5., [('a', 3.), ('b', 2.)]),
    (0., 5., [('a', 2.), ('a', 3.)]),
    (0., 5., [('a', float('nan'))]),
    (0., 5., []),
])
def test_d818_rejects_invalid_local_boundaries(start, end, boundaries):
    from scripts.ieee_tc.audit_d703_3b_full import clipped_sequential_stages
    with pytest.raises(ValueError):
        clipped_sequential_stages(start, end, boundaries)
