"""D298/D302/D308 data-only Full100 audit; existing native/binding/SLO checks."""
import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

from scripts import ieee_tc_preflight as p
from scripts.ieee_tc.plot_d290_diagnostic import (
    audit_run, compare_diagnostics, write_binding_audit, sha256)
from scripts.analyze_control_path_overhead import provisional_timing_bound, summarize_admission_waits

ROOT = p.ROOT
RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d298_20261007'
RUN = RAW/'7b_capacity_validation_100'
SOURCE = RUN/'experiment_results_full_vllm_dedicated_a500_r100_c2_tc_ieee_prefetch_validation_100_d298_7b_capacity_validation_100.json'
REFERENCE = ROOT/'results/ieee_tc/p2_backend_qualification/d296_20261007/7b_lifecycle_validation_100/experiment_results_full_vllm_dedicated_a500_r100_c2_tc_ieee_prefetch_validation_100_d296_7b_lifecycle_validation_100.json'
OUT = ROOT/'paper_results/ieee_tc/p2_backend/d298_capacity/smoke1'
INDEX = ROOT/'paper_results/ieee_tc/inputs/20260927_7b_materialized_content_index.json'
WARM = ROOT/'paper_results/ieee_tc/p2_backend/20261002_d170_warm_measurement.json'


def read(path):
    return json.loads(path.read_text())


def copy_retention_trigger(requests):
    """Use the actually exported per-request cumulative field, not event totals."""
    values = []
    if not requests:
        raise ValueError('complete request population required')
    for request in requests:
        sources = request['gpu_reference_evidence']['routing_snapshot']['sources']
        if not isinstance(sources, dict) or not sources:
            raise ValueError('complete source snapshots required')
        for source in sources.values():
            value = source['source_collection']['identity_frontier_retained_gpu_scopes']
            if type(value) is not int or value < 0:
                raise ValueError('invalid native event reuse counter')
            values.append(value)
    return dict(observed_cumulative_retained_scopes_max=max(values),
        source_snapshot_count=len(values), exact_rpc_savings=None, causal_latency_savings=None,
        caveat='Retained per-request observations only; repeated publication may count a scope again. Not a final event total or avoided RPC count.')


def planning_publication_trigger(events):
    """Positive measured publication/reuse, not a counterfactual RPC saving."""
    stats = events['_ieee_route_cache_stats']
    fields = ('planning_snapshot_publications', 'planning_snapshot_seeded_scopes',
              'planning_snapshot_hits')
    values = {key: stats[key] for key in fields}
    if any(type(value) is not int or value <= 0 for value in values.values()):
        raise ValueError('planning publication and request reuse must actually trigger')
    return dict(counters=values, exact_rpc_savings=None, causal_latency_savings=None)


def native_pending_turn_trigger(requests):
    if not requests:
        raise ValueError('complete request population required')
    counts = Counter()
    for request in requests:
        tier = request['readiness_tier_before_dispatch']
        pending = request['gpu_reference_evidence']['pending_kv_admission']
        if pending['state'] != 'closed':
            raise ValueError('pending demand did not retire')
        scope = pending.get('execution_scope')
        if (tier == 'gpu') != (scope == 'single_native_core_turn_v1'):
            raise ValueError('native combined scope must match the selected GPU path')
        if tier != 'gpu' and scope is not None:
            raise ValueError('lower-tier path has an unexpected combined scope')
        counts[tier] += 1
    if counts['gpu'] <= 0:
        raise ValueError('native pending/GPU core-turn candidate did not trigger')
    return dict(final_selected_tier_counts=dict(counts),
        combined_successful_requests=counts['gpu'],causal_latency_savings=None)


def prebuild_policy_witness(body):
    """Require the initialized native-model witness, not a configured flag."""
    records = re.findall(r'^\[quantization-prebuild\] (.+)$', body, re.MULTILINE)
    if len(records) != 1:
        raise ValueError('exactly one actual initialized prebuild witness required')
    row = json.loads(records[0])
    if (row.get('policy') != 'on_demand_v1' or 'quantization' not in row
            or row['quantization'] is not None
            or type(row.get('background_build_threads')) is not int
            or row['background_build_threads'] != 0
            or row.get('native_core_methods') != ['spawn']):
        raise ValueError('actual prebuild/native process contract failed')
    return row


def hit_host_transaction_trigger(requests):
    """Count only audited final acquisitions, not duplicated nested receipts."""
    counts = Counter()
    if not requests:
        raise ValueError('complete request population required')
    for request in requests:
        receipt = request['gpu_reference_evidence']['receipt']
        allocation = receipt['host_allocation']
        kind = allocation['after_storage_observation']
        eligible = receipt['gpu_confirmed_before_acquisition'] is True
        if kind not in ('fresh_inventory', 'same_transaction_confirmed_gpu_hit_v1'):
            raise ValueError('unknown HOST storage observation')
        reused = kind == 'same_transaction_confirmed_gpu_hit_v1'
        if reused != eligible or (reused and receipt['native_load_invoked'] is not False):
            raise ValueError('transaction-local HOST evidence used outside confirmed GPU hit')
        for side in ('before', 'after'):
            check = allocation[side]
            observed = check['before']
            if (check['admitted'] is not True or observed['available'] is not True
                    or observed['accounted_tensor_bytes'] > check['tensor_budget_bytes']):
                raise ValueError('HOST transaction did not pass actual occupancy checks')
        counts[kind] += 1
    if counts['same_transaction_confirmed_gpu_hit_v1'] <= 0:
        raise ValueError('transaction-local HOST evidence did not trigger')
    return dict(final_acquisitions=len(requests), counts=dict(counts),
                misclassified=0, causal_latency_savings=None)


def control_interleave_trigger(requests):
    """Audit actual final registration receipts, separately for each native owner.

    This qualified TP=1 path uses the same process for scheduler and worker.
    Counters precede the current utility's completion. They are not token/model
    step counts and cannot establish GPU idle time or causal speedup.
    """
    if not requests:
        raise ValueError('complete request population required')
    keys = ('input_turns', 'handled_controls', 'yields_to_native_loop',
            'yields_with_queued_input')
    owners, request_ids, intent_ids, pid_owners = {}, set(), set(), {}
    for request in requests:
        rid = request['request_id']
        evidence, timing = request['gpu_reference_evidence'], request['native_token_timing']
        pending = evidence['pending_kv_admission']
        receipt, retired = pending['receipt'], pending['close_receipt']['receipt']
        owner, pid, clock = (receipt['scheduler_owner_id'], receipt['scheduler_pid'],
                             receipt['clock_id'])
        intent = receipt['intent_id']
        stamp, dispatch = receipt['observed_at'], timing['native_dispatch_monotonic_s']
        state = receipt['control_interleave']
        if (request['success'] is not True or not isinstance(rid, str) or not rid
                or rid in request_ids or not isinstance(intent, str) or not intent
                or intent in intent_ids or pending['intent_id'] != intent
                or timing['backend_request_id'] != intent or pending['state'] != 'closed'
                or receipt['kind'] != 'ieee_pending_admission_v1'
                or receipt['state'] != 'pending' or receipt['native_request_id'] is not None
                or receipt['physical_kv_reservation'] is not False
                or not isinstance(owner, str) or not owner
                or type(pid) is not int or pid <= 0
                or type(evidence['receipt']['worker_pid']) is not int
                or evidence['receipt']['worker_pid'] != pid
                or not isinstance(clock, str) or not clock
                or clock != timing['native_clock_id'] or clock != evidence['receipt']['clock_id']
                or retired['scheduler_owner_id'] != owner or retired['clock_id'] != clock
                or retired['external_request_id'] != intent or retired['retired'] is not True
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                       for v in (stamp, dispatch)) or stamp >= dispatch
                or state.get('kind') != 'ieee_native_control_interleave_v1'
                or state.get('policy') != 'atomic_control_yield_v1'
                or any(type(state.get(k)) is not int or state[k] < 0 for k in keys)):
            raise ValueError('complete same-request/owner native control receipt required')
        values = {k: state[k] for k in keys}
        if (values['yields_with_queued_input'] > values['yields_to_native_loop']
                or values['yields_to_native_loop'] > values['handled_controls']
                or values['yields_to_native_loop'] > values['input_turns']):
            raise ValueError('interleave counter subset invariant violated')
        request_ids.add(rid); intent_ids.add(intent)
        identity = (pid, clock)
        if identity in pid_owners and pid_owners[identity] != owner:
            raise ValueError('native process has conflicting scheduler owners')
        pid_owners[identity] = owner
        group = owners.setdefault(owner, dict(scheduler_pid=pid, clock_id=clock, observations=[]))
        if (group['scheduler_pid'], group['clock_id']) != identity:
            raise ValueError('scheduler owner changed process or clock')
        group['observations'].append(dict(request_id=rid, intent_id=intent,
                                         observed_at=stamp, **values))
    for group in owners.values():
        rows = sorted(group['observations'], key=lambda r: r['observed_at'])
        for before, after in zip(rows, rows[1:]):
            if (before['observed_at'] >= after['observed_at']
                    or any(after[k] < before[k] for k in keys)):
                raise ValueError('same-owner snapshot timestamp or counter regressed')
        delta = {k: rows[-1][k] - rows[0][k] for k in keys} if len(rows) > 1 else None
        group.update(observations=rows, delta=delta,
                     triggered=delta is not None and delta['yields_to_native_loop'] > 0,
                     queued_input_triggered=delta is not None and delta['yields_with_queued_input'] > 0)
    return dict(kind='full_native_control_receipt_audit_v1', requests=len(requests),
        policy='atomic_control_yield_v1', owners=owners,
        triggered=any(g['triggered'] for g in owners.values()),
        queued_input_triggered=any(g['queued_input_triggered'] for g in owners.values()),
        counter_scope='per_owner_between_final_registration_receipts',
        single_snapshot_owners=sum(g['delta'] is None for g in owners.values()),
        model_step_count=None, gpu_idle_seconds=None, causal_speedup=None,
        caveat='Retains zero-trigger and single-snapshot owners. Initial cumulative values include earlier work; no cross-owner subtraction or summation of repeated snapshots.')


def file_demand_transaction_trigger(requests):
    """Audit final held-file transactions without fabricating counterfactual time."""
    from faaslora.memory.residency_manager import (
        IEEE_FILE_DEMAND_TRANSACTION, validate_file_demand_witness)
    if not requests:
        raise ValueError('complete request population required')
    seen, counts, advances = set(), Counter(), []
    for row in requests:
        rid, aid = row['request_id'], row['adapter_id']
        if rid in seen or row.get('success') is not True:
            raise ValueError('complete unique successful request receipts required')
        seen.add(rid)
        e = row['gpu_reference_evidence']
        intent, native, source = e['intent'], e['receipt'], e['source_admission']['source']
        tier = row['readiness_tier_before_dispatch']
        if (source['tier'] != tier or type(source['native']) is not bool
                or e.get('state') != 'released' or native.get('acquired') is not True
                or e['release_receipt'].get('released') is not True
                or native['lease_id'] != intent['lease_id']
                or native['owner_id'] != intent['expected_owner_id']
                or native['lora_name'] != aid):
            raise ValueError('selected source and native ownership receipts differ')
        if source['native']:
            if tier not in ('gpu','host') or any(key in obj for obj in (intent,native)
                    for key in ('file_source_witness','file_demand_transaction')):
                raise ValueError('file policy leaked onto native cached-source path')
            counts['native_'+tier] += 1
            continue
        if tier not in ('remote','nvme','host'):
            raise ValueError('unknown file-source tier')
        witness = native.get('file_source_witness')
        validate_file_demand_witness(witness,lora_name=aid,lora_path=intent['lora_path'],
                                    content_sha256=source['content_sha256'])
        held = e['local_source_reference']
        fence = held.get('protected_until_native_acquisition',{})
        if (witness != intent.get('file_source_witness')
                or any(witness[k] != held.get(k) for k in
                       ('owner_id','lease_id','adapter_id','path','tier','device','inode'))
                or held.get('content_verified') is not True or held.get('state') != 'released'
                or witness['tier'] != ('host' if tier=='host' else 'nvme')
                or native.get('expected_content_sha256') != witness['content_sha256']
                or intent.get('expected_content_sha256') != witness['content_sha256']
                or any(fence.get(k) != native[k] for k in
                       ('owner_id','lease_id','lora_path','clock_id','acquired_monotonic_s'))):
            raise ValueError('held file/content/lifetime binding differs')
        txn = native.get('file_demand_transaction',{})
        observed, validated = txn.get('observed_epoch'), txn.get('validated_epoch')
        if (txn.get('kind') != IEEE_FILE_DEMAND_TRANSACTION
                or any(type(x) is not int or x<0 for x in (observed,validated,native.get('epoch')))
                or not observed <= validated <= native['epoch']
                or observed != intent['expected_epoch']
                or type(txn.get('owner_epoch_advance')) is not int
                or txn['owner_epoch_advance'] != validated-observed
                or txn.get('live_native_constraints_checked') is not True
                or txn.get('controller_capacity_snapshot_reused') is not False):
            raise ValueError('file-demand live transaction receipt differs')
        counts['file_'+tier] += 1
        advances.append(validated-observed)
    return dict(kind='full_file_demand_transaction_audit_v1',requests=len(seen),
        policy=IEEE_FILE_DEMAND_TRANSACTION,source_counts=dict(counts),
        file_transactions=len(advances),triggered=bool(advances),
        advanced_owner_epoch_transactions=sum(x>0 for x in advances),
        max_owner_epoch_advance=max(advances) if advances else None,
        old_capacity_snapshot_reused=False,n_correct=None,causal_speedup=None)


def main(campaign='d298'):
    global RAW, RUN, SOURCE, REFERENCE, OUT
    health_name = 'smoke_remote_health2.json'
    comparison_caveat = 'D298 changes the measured capacity package versus D296, not a single-factor ablation.'
    if campaign == 'd302':
        REFERENCE = SOURCE
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d302_20261007'
        RUN = RAW/'7b_frontier_validation_100'
        SOURCE = RUN/'experiment_results_full_vllm_dedicated_a500_r100_c2_tc_ieee_frontier_validation_100_d302_7b_frontier_validation_100.json'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d302_frontier/smoke1'
        health_name = 'remote_health.json'
        comparison_caveat = ('D298 prefix is reused for descriptive context only: D302 also inherits D300 scale-in lower reference. '
            'Full W0 versus D300 is still required to isolate the publication fix; prefix GPU-s is not extrapolated.')
    elif campaign == 'd308':
        REFERENCE = ROOT/'results/ieee_tc/p2_backend_qualification/d302_20261007/7b_frontier_validation_100/experiment_results_full_vllm_dedicated_a500_r100_c2_tc_ieee_frontier_validation_100_d302_7b_frontier_validation_100.json'
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d308_20261007'
        RUN = RAW/'7b_prefill_validation_100'
        SOURCE = RUN/'experiment_results_full_vllm_dedicated_a500_r100_c2_tc_ieee_prefill_validation_100_d308_7b_prefill_validation_100.json'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d308_prefill/smoke1'
        health_name = 'remote_health.json'
        comparison_caveat = ('D308 uses D303 controls with a 512-token iteration budget, newly measured D307 initialization timings '
            'and the necessary atomic source-pin correctness repair. Not a pure token-budget causal ablation; '
            'Full W0 is required; prefix GPU-s is not extrapolated.')
    elif campaign == 'd310':
        REFERENCE = ROOT/'results/ieee_tc/p2_backend_qualification/d308_20261007/7b_prefill_validation_100/experiment_results_full_vllm_dedicated_a500_r100_c2_tc_ieee_prefill_validation_100_d308_7b_prefill_validation_100.json'
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d310_20261007'
        RUN = RAW/'7b_prefill_validation_100'
        SOURCE = RUN/'experiment_results_full_vllm_dedicated_a500_r100_c2_tc_ieee_prefill_validation_100_d310_7b_prefill_validation_100.json'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d310_read_pin_validation/smoke1'
        health_name = 'remote_health.json'
        comparison_caveat = ('D308 settings and measured profiles unchanged; source read-pin repair only. '
            'One prefix run, not complete W0, numerical or formal SLO qualification.')
    elif campaign == 'd312':
        REFERENCE = ROOT/'results/ieee_tc/p2_backend_qualification/d310_20261007/7b_prefill_validation_100/experiment_results_full_vllm_dedicated_a500_r100_c2_tc_ieee_prefill_validation_100_d310_7b_prefill_validation_100.json'
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d312_20261007'
        RUN = RAW/'7b_prefill_validation_100'
        SOURCE = RUN/'experiment_results_full_vllm_dedicated_a500_r100_c2_tc_ieee_prefill_validation_100_d312_7b_prefill_validation_100.json'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d312_host_validation/smoke1'
        health_name = 'remote_health.json'
        comparison_caveat = ('D310 settings and measured profiles unchanged; native HOST file observation/pin transaction repair only. '
            'One prefix run, not complete W0, numerical or formal SLO qualification.')
    elif campaign == 'd317':
        REFERENCE = ROOT/'results/ieee_tc/p2_backend_qualification/d302_20261007/7b_frontier_validation_100/experiment_results_full_vllm_dedicated_a500_r100_c2_tc_ieee_frontier_validation_100_d302_7b_frontier_validation_100.json'
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d317_20261007'
        RUN = RAW/'7b_copy_validation_100'
        # Filename retains the frozen source-config count; the guarded external
        # replay and saved metadata both explicitly declare the 100-prefix.
        SOURCE = RUN/'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_copy_validation_100_d317_7b_copy_validation_100.json'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d317_copy_validation/smoke2'
        health_name = 'remote_health.json'
        comparison_caveat = ('D302 is descriptive same-1024/control/profile prefix context, not a causal single-change contrast: '
            'intervening native-source correctness repairs remain. D317 tests D316 scope retention with D315 settings. '
            'Unchanged-candidate full4000 versus D315 is required; prefix GPU-s is not extrapolated.')
    elif campaign in ('d322', 'd322-attempt2'):
        REFERENCE = ROOT/'results/ieee_tc/p2_backend_qualification/d317_20261007/7b_copy_validation_100/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_copy_validation_100_d317_7b_copy_validation_100.json'
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d322_20261007'
        RUN = RAW/'7b_length_validation_100'
        SOURCE = RUN/'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_length_validation_100_d322_7b_length_validation_100.json'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d322_length_candidate/smoke1'
        health_name = 'remote_health.json'
        comparison_caveat = ('D317 settings and serving code retained; only D321 measured declared-output classes/initialization changed. '
            'One prefix diagnostic; unchanged candidate requires complete W0. Not a formal SLO or numerical qualification.')
        if campaign == 'd322-attempt2':
            RUN = RAW/'7b_length_validation_100_attempt2'
            SOURCE = RUN/'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_length_validation_100_d322_7b_length_validation_100_attempt2.json'
            OUT = ROOT/'paper_results/ieee_tc/p2_backend/d322_length_candidate/smoke2'
            health_name = 'attempt2/remote_health.json'
            comparison_caveat += ' Joint profile binding permits independently verified input provenance; common runtime/environment/resource validation remains strict.'
    elif campaign in ('d331', 'd331-attempt2'):
        REFERENCE = ROOT/'results/ieee_tc/p2_backend_qualification/d322_20261007/7b_length_validation_100_attempt2/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_length_validation_100_d322_7b_length_validation_100_attempt2.json'
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d331_20261008'
        RUN = RAW/'7b_planning_validation_100'
        SOURCE = RUN/'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d331_7b_planning_validation_100.json'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d331_planning_validation/smoke2'
        health_name = 'remote_health.json'
        comparison_caveat = ('D322 attempt2 is descriptive same-input prefix context, not a single-change contrast: '
            'D325 physical retirement correction is also retained. D331 changes only D330 planning-source publication '
            'from D326 settings/code. Same-candidate complete4000 versus D326 is required; do not extrapolate prefix GPU-s.')
        if campaign == 'd331-attempt2':
            RUN = RAW/'7b_planning_validation_100_attempt2'
            SOURCE = RUN/'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d331_7b_planning_validation_100_attempt2.json'
            OUT = ROOT/'paper_results/ieee_tc/p2_backend/d331_planning_validation/attempt2'
            health_name = 'attempt2/remote_health.json'
            comparison_caveat += ' Only final post-shutdown route-counter export repaired; serving policy unchanged.'
    elif campaign == 'd334':
        REFERENCE = ROOT/'results/ieee_tc/p2_backend_qualification/d331_20261008/7b_planning_validation_100_attempt2/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d331_7b_planning_validation_100_attempt2.json'
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d334_20261008'
        RUN = RAW/'7b_native_pending_validation_100'
        SOURCE = RUN/'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d334_7b_native_pending_validation_100.json'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d334_native_pending_validation/smoke1'
        health_name = 'remote_health.json'
        comparison_caveat = ('D331 attempt2 reused as the same-input/policy prefix context. D334 changes only '
            'selected GPU native registration/protection sequencing. One prefix is not a performance selection '
            'or formal qualification; complete4000 versus D332 is required, without extrapolating prefix GPU-s.')
    elif campaign == 'd340':
        REFERENCE = ROOT/'results/ieee_tc/p2_backend_qualification/d334_20261008/7b_native_pending_validation_100/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d334_7b_native_pending_validation_100.json'
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d340_20261008'
        RUN = RAW/'7b_prebuild_validation_100'
        SOURCE = RUN/'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d340_7b_prebuild_validation_100.json'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d340_prebuild_candidate/smoke1'
        health_name = 'remote_health.json'
        comparison_caveat = ('D334 is descriptive same-input prefix context only: D340 retains D336 capacity controls, '
            'not D334 controls. D340 changes only the unused quantization-helper prebuild policy from D336. '
            'Complete4000 versus D336 is required; do not attribute prefix differences or extrapolate GPU-s.')
    elif campaign == 'd345':
        REFERENCE = ROOT/'results/ieee_tc/p2_backend_qualification/d334_20261008/7b_native_pending_validation_100/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d334_7b_native_pending_validation_100.json'
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d345_20261008'
        RUN = RAW/'7b_capacity_validation_100'
        SOURCE = RUN/'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d345_7b_capacity_validation_100.json'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d345_capacity_validation/smoke1'
        health_name = 'remote_health.json'
        comparison_caveat = ('D334 is descriptive same-input prefix context, not a single-change comparison. '
            'D345 uses the predeclared C14 runtime and fresh D344 profiles with D336 capacity-derived control rules. '
            'The unchanged batch8 warm thresholds are provisional timing-only bounds; no numerical or G1/G2 '
            'qualification and no prefix GPU-s extrapolation. Complete4000 versus D336 is required.')
    elif campaign == 'd352':
        REFERENCE = ROOT/'results/ieee_tc/p2_backend_qualification/d345_20261008/7b_capacity_validation_100/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d345_7b_capacity_validation_100.json'
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d352_20261008'
        RUN = RAW/'7b_hit_validation_100'
        SOURCE = RUN/'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d352_7b_hit_validation_100.json'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d352_hit_host_transaction/smoke1'
        health_name = 'remote_health.json'
        comparison_caveat = ('D345 is descriptive same-C14/backend/profile/input prefix context only: '
            'D349 soft scaling targets are retained in D352, so this is not a single-change performance contrast. '
            'Candidate changes only the same-transaction HOST postcheck from D349. Full4000 versus '
            'D349 and D336 is required; no numerical/formal G1/G2 qualification or prefix extrapolation.')
    elif campaign == 'd357':
        REFERENCE = ROOT/'results/ieee_tc/p2_backend_qualification/d352_20261008/7b_hit_validation_100/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d352_7b_hit_validation_100.json'
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d357_20261008'
        RUN = RAW/'7b_interleave_validation_100'
        SOURCE = RUN/'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d357_7b_interleave_validation_100.json'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d357_control_interleave/smoke1'
        health_name = 'remote_health.json'
        comparison_caveat = ('D352 is descriptive same-input prefix context only. D357 retains D353 controls, '
            'enables atomic_control_yield_v1 and uses the corresponding complete D356 measured profiles. '
            'Trigger counters do not prove latency savings. Unchanged full4000 versus D353/D336 is required; '
            'no numerical/formal qualification or prefix GPU-s extrapolation.')
    elif campaign == 'd363':
        REFERENCE = ROOT/'results/ieee_tc/p2_backend_qualification/d357_20261008/7b_interleave_validation_100/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d357_7b_interleave_validation_100.json'
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d363_20261009'
        RUN = RAW/'7b_file_demand_validation_100'
        SOURCE = RUN/'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d363_7b_file_demand_validation_100.json'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d363_file_demand_validation/smoke1'
        health_name = 'remote_health.json'
        comparison_caveat = ('D357 is descriptive same-input prefix context only. D363 retains D358 controls, '
            'enables protected_file_owner_turn_v1 and uses the complete D362 measured profiles. '
            'Live owner-epoch advances verify exposure, not avoided RPCs or causal speedup. '
            'Complete4000 versus D358/D353/D336 is required; no numerical/formal qualification or extrapolation.')
    elif campaign != 'd298':
        raise ValueError('unknown frozen smoke identity')
    assert not OUT.exists(), 'never overwrite a prior audit'
    assert p.check_plan() == '0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    assert sha256(ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md') == '5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    launch = read(RUN/'launch.json')
    assert launch['pass'] and launch['service_returncode'] == launch['watchdog_returncode'] == 0
    assert launch['service_path_removed'] and launch['native_gpu_context_release_confirmed']
    stats, evidence, requests = audit_run(SOURCE)
    assert len(requests) == 100
    outcome_path = RUN/'launch.launch/physical_deployment/main_outcome.json'
    outcome = read(outcome_path)
    assert outcome['run_completed'] and not outcome['error']
    assert not outcome['cleanup_errors'] and not outcome['measurement_errors']
    journal = RAW/('attempt2/smoke_remote_transfers.jsonl' if campaign in ('d322-attempt2', 'd331-attempt2') else 'smoke_remote_transfers.jsonl')
    if campaign in ('d345', 'd352', 'd357', 'd363'):
        journal = RAW/'remote_transfers.jsonl'
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    remote = {r['transfer_id']:r for r in rows}
    transfers = outcome['remote_transfers']
    assert len(remote) == len(rows) == len(transfers)
    assert set(remote) == {r['http_transfer_id'] for r in transfers}
    health = read(RAW/health_name)
    if campaign in ('d331', 'd331-attempt2', 'd334', 'd340', 'd345', 'd352', 'd357', 'd363'):
        clock = health['7b']['clock_id']
    else:
        clock = health['clock_id'] if campaign in ('d322', 'd322-attempt2') else health['health']['clock_id']
    for r in transfers:
        s = remote[r['http_transfer_id']]
        assert r['state'] == 'published' and r['content_verified'] and r['published_archive_verified']
        assert r['artifact_id'] == s['artifact_id'] and r['remote_archive_sha256'] == s['archive_sha256']
        assert r['transferred_bytes'] == s['archive_bytes'] == s['bytes_read'] == s['bytes_written']
        assert s['clock_id'] == clock and s['outcome'] == 'sent'
        assert not r['remote_pack_performed'] and not s['pack_performed'] and not s['temporary_created']
    physical = read(RUN/'launch.launch/physical_deployment/summary.json')
    assert abs(sum(physical['window_gpu_seconds'].values())-stats['physical_gpu_seconds']) < 1e-6
    with (RUN/'launch.launch/watchdog.jsonl').open() as stream:
        resources = [r for line in stream if (r:=json.loads(line))['event']=='resource_sample']
    assert resources and not any(r['decision']['abort_reasons'] for r in resources)
    assert sha256(WARM) == '0ebc7587effe293dcb986d33fb932c100cbcb3384c56ab2c0b0bec73b2bdd88c'
    thresholds = [dict(group_id=t['group_id'],lower_exclusive=t['lower_exclusive'],
        upper_inclusive=t['upper_inclusive'],ttft_s=t['candidate_ttft_threshold_ms']/1000,
        tpot_s=t['candidate_tpot_threshold_ms']/1000) for t in read(WARM)['warm_threshold_candidates']]
    normalized = []
    for r in requests:
        n = r['native_token_timing']['actual_prompt_tokens']
        assert type(n) in (int,float) and math.isfinite(n) and n >= 1 and int(n)==n
        normalized.append(dict(request_id=r['request_id'],success=True,prompt_tokens=int(n),
            token_count=r['completion_tokens'],ttft_s=r['overall_ttft_ms']/1000,tpot_s=r['tpot_ms']/1000))
    timing = provisional_timing_bound(normalized,thresholds,offered_count=100)
    seal = p.verify_seal(ROOT/'paper_results/ieee_tc/safety/20260925_execution_start_protected.json')
    known = '/home/qhq/serverless_llm_baselines/scripts/replay_openai_trace.py'
    assert seal['changed']==[known] and seal['entry_count']==147
    assert p.digest(Path(known))=='04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22'
    OUT.mkdir(parents=True, exist_ok=False)
    compare_diagnostics(REFERENCE,SOURCE,OUT)
    write_binding_audit(REFERENCE,SOURCE,INDEX,OUT)
    binding = read(OUT/'binding_audit.json')
    assert all(r['control_binding_pass_count']==100 for r in binding['runs'].values())
    result = dict(kind=f'{campaign}_smoke_audit_v1',stats=stats,evidence=evidence,
        timing=timing,window_gpu_seconds=physical['window_gpu_seconds'],
        tier_counts=dict(Counter(r['readiness_tier_before_dispatch'] for r in requests)),
        remote=dict(transfers=len(transfers),wire_bytes=sum(r['transferred_bytes'] for r in transfers),
            pack_performed=False,clock=clock),
        resources=dict(samples=len(resources),peak_bytes=max(r['service']['memory.peak'] for r in resources),
            min_host_available_bytes=min(r['host']['available_bytes'] for r in resources),
            max_swap_bytes=max(r['service']['memory.swap.current'] for r in resources),
            memory_events={k:max(r['service']['memory.events'][k] for r in resources)
                for k in ('high','max','oom','oom_kill')}),
        protection_raw=seal,full_qualified=False,n_correct=None,common_slo_qualified=False,
        no_figures=True,replicates_per_configuration=1,confidence_interval=None,
        caveat='Complete100-request diagnostic only; not complete4000W0, numerical correctness or formal G1/G2.',
        comparison_caveat=comparison_caveat,
        raw_evidence=[dict(path=str(path),sha256=sha256(path)) for path in
            (SOURCE,REFERENCE,outcome_path,journal,WARM,INDEX,
             RAW/('remote_monitor.log' if campaign in ('d345', 'd352', 'd357', 'd363') else
                  'attempt2/smoke_remote_monitor.log' if campaign in ('d322-attempt2', 'd331-attempt2') else 'smoke_remote_monitor.log'),
             ROOT/'scripts/run_all_experiments.py', Path(__file__))])
    if campaign in ('d302', 'd308', 'd310', 'd312', 'd317', 'd322', 'd322-attempt2', 'd331', 'd331-attempt2', 'd334', 'd340', 'd345', 'd352', 'd357', 'd363'):
        result['admission_waits'] = summarize_admission_waits(requests)
        result['mechanism_events'] = {k:v for k,v in outcome['mechanism_events'].items()
            if k in ('_ieee_source_observation_stats', '_ieee_route_cache_stats')}
    if campaign in ('d317', 'd322', 'd322-attempt2'):
        result['copy_retention_trigger'] = copy_retention_trigger(requests)
        assert result['copy_retention_trigger']['observed_cumulative_retained_scopes_max'] > 0, 'new event-confirmed branch did not trigger'
    if campaign == 'd331':
        # Exact known original run: counters were computed but never serialized.
        # Preserve other valid measurements, but do NOT pass the mechanism gate.
        assert '_ieee_route_cache_stats' not in outcome['mechanism_events']
        result['planning_publication_trigger'] = dict(qualification_pass=False,
            status='unobserved_export_missing', counters=None, exact_rpc_savings=None,
            causal_latency_savings=None, requires_new_observation=True)
    if campaign == 'd331-attempt2':
        result['planning_publication_trigger'] = planning_publication_trigger(outcome['mechanism_events'])
    if campaign in ('d334', 'd340', 'd345', 'd352', 'd357', 'd363'):
        result['native_pending_turn_trigger'] = native_pending_turn_trigger(requests)
    if campaign in ('d352', 'd357', 'd363'):
        result['hit_host_transaction_trigger'] = hit_host_transaction_trigger(requests)
    if campaign in ('d357', 'd363'):
        result['control_interleave_trigger'] = control_interleave_trigger(requests)
        result['candidate_trigger_gate_passed'] = result['control_interleave_trigger']['triggered']
    if campaign == 'd363':
        result['file_demand_transaction_trigger'] = file_demand_transaction_trigger(requests)
        result['candidate_trigger_gate_passed'] = (result['candidate_trigger_gate_passed']
            and result['file_demand_transaction_trigger']['triggered'])
    if campaign == 'd340':
        service_log = RUN/'launch.launch/service.log'
        archives = list(dict.fromkeys(re.findall(
            r'preserved dedicated worker workdir: (.+)', service_log.read_text())))
        assert archives, 'preserved initialized worker evidence required'
        witnesses = {}
        for directory in archives:
            log = Path(directory)/'worker.log'
            body = log.read_text()
            cores = re.findall(r'EngineCore pid=(\d+)\) INFO .* Initializing a V1 LLM engine', body)
            assert len(cores) == 1
            witness = prebuild_policy_witness(body)
            if cores[0] in witnesses:
                assert witnesses[cores[0]]['witness'] == witness
            else:
                witnesses[cores[0]] = dict(witness=witness, log=str(log), sha256=sha256(log))
        result['prebuild_witnesses'] = witnesses
        result['raw_evidence'].append(dict(path=str(service_log), sha256=sha256(service_log)))
    with (OUT/'audit.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(stats=stats,timing={k:v for k,v in timing.items() if k!='decisions'},
        resources=result['resources'],remote=result['remote']),indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', choices=('d298','d302','d308','d310','d312','d317','d322','d322-attempt2','d331','d331-attempt2','d334','d340','d345','d352','d357','d363'), default='d298')
    main(parser.parse_args().campaign)
