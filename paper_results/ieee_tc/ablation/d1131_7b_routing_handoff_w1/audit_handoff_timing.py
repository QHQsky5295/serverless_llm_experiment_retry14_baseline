"""D1131 RoutingHandoff physical timing audit, reused from D1126 retained receipts.

Run only after the sealed comparison and actual GPU release. No new workload,
inference, old reference replay, counterfactual latency, or result selection.
"""
import collections
import json
import os
from pathlib import Path
import subprocess
import sys
import ijson

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
RAW = Path(__file__).resolve().parent
BASE = RAW
OUT = BASE/'timing'
sys.path.insert(0, str(ROOT))
from scripts import ieee_tc_preflight as p


def read(path):
    return json.loads(path.read_text())


def ref(path):
    return dict(path=str(path), sha256=p.digest(path), bytes=path.stat().st_size)


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def items(path, prefix):
    with path.open('rb') as f:
        yield from ijson.items(f, prefix, use_float=True)


group = p.cg_path()
# Same bounded transient-service identity as the completed D1124 offline audit.
identity = dict(unit=group.name, path=str(group), pid=os.getpid(), uid=os.getuid(),
    role='cpu_only_post_run_analysis_service')
assert os.sched_getaffinity(0) == {2, 3, 26, 27}
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
for name, value in [('memory.high', 3*p.GIB), ('memory.max', 4*p.GIB), ('memory.swap.max', 0)]:
    assert (group/name).read_text().strip() == str(value)
assert not subprocess.check_output(
    ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
assert p.check_plan() == 'b5170643aa5d332efc67407af10ab7022a2d68fd802e2389d253968bd92dcf8f'
comparison = read(BASE/'comparison/comparison.json')
mechanisms = read(BASE/'comparison/mechanisms.json')
for name, expected_sha in read(BASE/'comparison/manifest.json')['outputs'].items():
    assert p.digest(BASE/'comparison'/name) == expected_sha
sources = [ref(BASE/'comparison'/name) for name in ('comparison.json', 'mechanisms.json', 'manifest.json')]
reports, violations, unknown = {}, [], []
# Reuse the already completed Full physical audit, rather than rereading or
# reinterpreting the same Full handoff work in every ablation.
full_path = ROOT/'paper_results/ieee_tc/ablation/d1124_7b_no_handoff_w1/handoff/handoff.json'
assert p.digest(full_path) == 'ff3e52e0787b9cbdfb6b1e535a847257e93081bd7d2ff3f349c901985c0ce2d0'
full_reference = dict(source=ref(full_path), summary=read(full_path)['summary'],
    reused_without_reanalysis=True, run='D1115')
sources.append(ref(full_path))

for variant, stem in [('RoutingHandoff', 'd1131_')]:
    paths = [Path(path) for path in comparison['source_refs']
        if Path(path).name == 'main_outcome.json' and stem in path]
    assert len(paths) == 1
    outcome = paths[0]
    assert p.digest(outcome) == comparison['source_refs'][str(outcome)]
    sources.append(ref(outcome))
    acts = {a['activation_id']: a for a in mechanisms[variant]['activations']}
    firsts = {a['activation_id']: a for a in mechanisms[variant]['first_services']['records']}
    assert acts.keys() == firsts.keys()
    epochs = {}
    for aid, a in acts.items():
        assert not firsts[aid]['first_dispatch_ambiguous']
        first = firsts[aid]['first_request']
        epochs[aid] = dict(activation_id=aid, category=a['category'],
            engine_ready_at=a['engine_ready_at'], first_request=first,
            native_attempts=[], file_plan_records=[], movement_job_ids=[])

    # Read original attempts, not the comparison's intentionally small plan projection.
    for family, tier, start_key, end_key in [
        ('_ieee_gpu_preparation_plans', 'gpu', 'native_load_started_monotonic_s', 'native_load_completed_monotonic_s'),
        ('_ieee_native_host_preparations', 'host', 'load_started_monotonic_s', 'load_completed_monotonic_s')]:
        for plan in items(outcome, 'mechanism_events.'+family+'.item'):
            if plan.get('trigger_reason') != 'handoff':
                continue
            aid = plan['activation_id']
            assert plan['target_replica'] == aid
            a, epoch = acts[aid], epochs[aid]
            for attempt in plan['attempts']:
                receipt = attempt.get('receipt')
                row = dict(tier=tier, plan_id=plan['plan_id'], attempt_id=attempt['attempt_id'],
                    state=attempt['state'], receipt_present=receipt is not None,
                    native_load_invoked=None if receipt is None else receipt.get('native_load_invoked'))
                if receipt is not None:
                    assert receipt['owner_id'] == a['native_owner_id'] and receipt['clock_id'] == a['clock_id']
                    row.update(owner_id=receipt['owner_id'], clock_id=receipt['clock_id'],
                        adapter_id=receipt.get('lora_name'))
                if type(row['native_load_invoked']) is not bool:
                    # A missing invocation flag is not evidence of no physical
                    # work. It may be a pre-RPC cancellation or a refusal whose
                    # receipt omits the flag. Preserve it without guessing from
                    # state/reason labels, and do not invalidate performance.
                    unknown.append(dict(variant=variant, activation_id=aid,
                        reason='native_invocation_not_explicit', row=row))
                if row['native_load_invoked'] is True:
                    start, end, ready = receipt[start_key], receipt[end_key], a['engine_ready_at']
                    assert a['trigger_at'] <= start <= end
                    fr = epoch['first_request']
                    row.update(start=start, end=end, duration_ms=(end-start)*1000,
                        starts_before_engine_ready=None if ready is None else start < ready,
                        completes_before_engine_ready=None if ready is None else end <= ready,
                        completes_before_first_dispatch=None if fr is None else end <= fr['dispatch_monotonic_s'],
                        first_request_adapter_match=fr is not None and fr['adapter_id'] == row['adapter_id'])
                    if ready is None:
                        unknown.append(dict(variant=variant, activation_id=aid, reason='native_load_without_ready', row=row))
                    elif variant == 'DelayedHandoff' and start < ready:
                        violations.append(dict(variant=variant, activation_id=aid, reason='native_load_before_ready', row=row))
                epoch['native_attempts'].append(row)
    for plan in items(outcome, 'mechanism_events._ieee_file_preparation_plans.item'):
        if plan.get('trigger_reason') == 'handoff':
            # Registration time is explicitly NOT interpreted as physical I/O.
            epochs[plan['activation_id']]['file_plan_records'].append({k: plan[k] for k in (
                'plan_id', 'target_replica', 'activation_id', 'state', 'started_at', 'finished_at')})

    remote = list(items(outcome, 'remote_transfers.item'))
    jobs = []
    for job in items(outcome, 'coordination_after_shutdown.ieee_movements.item'):
        subs = job['subscriptions']
        handoffs = {i:s for i,s in subs.items() if s['metadata']['trigger_reason'] == 'handoff'}
        if not handoffs:
            continue
        earliest = min(s['attached_at'] for s in subs.values())
        creators = [i for i,s in subs.items() if s['attached_at'] == earliest]
        creator_id = creators[0] if len(creators) == 1 else None
        creator = None if creator_id is None else subs[creator_id]['metadata']
        # Use timestamp order, not JSON key order; exporters may sort object keys.
        row = dict(job_id=job['job_id'], key=job['key'], state=job['state'],
            created_at=job['created_at'], finished_at=job['finished_at'],
            subscriptions=subs, attempts=job['attempts'], creator_intent_id=creator_id,
            creator_metadata=creator, handoff_subscriptions=[], remote_receipts=[])
        if creator_id is None:
            unknown.append(dict(variant=variant, job_id=job['job_id'], reason='tied_creator_timestamps'))
        for intent, subscription in handoffs.items():
            aid = subscription['metadata']['activation_id']
            assert subscription['metadata']['target_replica'] == aid
            a, epoch = acts[aid], epochs[aid]
            ready = a['engine_ready_at']
            record = dict(intent_id=intent, activation_id=aid, attached_at=subscription['attached_at'],
                engine_ready_at=ready, creates_action=intent == creator_id if creator_id is not None else None,
                attached_before_engine_ready=None if ready is None else subscription['attached_at'] < ready,
                storage_domain='native' if job['key'][0] == a['native_owner_id'] else 'shared_file')
            row['handoff_subscriptions'].append(record)
            epoch['movement_job_ids'].append(job['job_id'])
            if ready is None:
                unknown.append(dict(variant=variant, job_id=job['job_id'], reason='subscription_without_ready', record=record))
            elif variant == 'DelayedHandoff':
                if record['attached_before_engine_ready']:
                    violations.append(dict(variant=variant, reason='subscription_before_ready', record=record))
                if intent == creator_id:
                    for attempt in job['attempts']:
                        if attempt['started_at'] < ready:
                            violations.append(dict(variant=variant, reason='owned_executor_entry_before_ready',
                                job_id=job['job_id'], attempt=attempt, record=record))
        for transfer in remote:
            if transfer['artifact_id'] != job['key'][2] or transfer['local_source_owner_id'] != job['key'][0]:
                continue
            if (job['finished_at'] is None or transfer['loading_started_monotonic_s'] < job['created_at']
                    or transfer['published_monotonic_s'] > job['finished_at']):
                continue
            assert transfer['content_verified'] is True
            # The manifest SHA names the entire frozen pool, not one adapter.
            publication = transfer['confirmed_file_publication']
            assert publication['adapter_id'] == job['key'][2]
            assert publication['content_sha256'] == job['key'][3]
            assert all(transfer['loading_clock_id'] == acts[s['activation_id']]['clock_id']
                for s in row['handoff_subscriptions'])
            row['remote_receipts'].append({k:transfer[k] for k in (
                'transfer_id', 'artifact_id', 'loading_started_monotonic_s', 'published_monotonic_s',
                'transferred_bytes', 'content_verified', 'remote_pack_performed', 'content_manifest_sha256',
                'confirmed_file_publication')})
            if variant == 'DelayedHandoff' and creator is not None and creator['trigger_reason'] == 'handoff':
                ready = acts[creator['activation_id']]['engine_ready_at']
                if ready is not None and transfer['loading_started_monotonic_s'] < ready:
                    violations.append(dict(variant=variant, reason='owned_remote_load_before_ready',
                        job_id=job['job_id'], transfer_id=transfer['transfer_id']))
        jobs.append(row)

    attempts = [r for e in epochs.values() for r in e['native_attempts']]
    invoked = [r for r in attempts if r['native_load_invoked'] is True]
    # Shared native receipts may occur in more than one plan; count physical spans once.
    unique_loads = {(r['tier'], r['owner_id'], r['clock_id'], r['adapter_id'], r['start'], r['end']):r for r in invoked}
    counts = dict(activations=len(epochs), categories=dict(collections.Counter(e['category'] for e in epochs.values())),
        native_attempt_records=len(attempts), native_invoked_records=len(invoked),
        native_receipt_absent_records=sum(not r['receipt_present'] for r in attempts),
        native_explicit_no_load_records=sum(r['native_load_invoked'] is False for r in attempts),
        native_invocation_unknown_records=sum(type(r['native_load_invoked']) is not bool for r in attempts),
        unique_physical_native_loads=len(unique_loads),
        unique_native_loads_by_tier=dict(collections.Counter(r['tier'] for r in unique_loads.values())),
        native_loads_before_engine_ready=sum(r['starts_before_engine_ready'] is True for r in unique_loads.values()),
        native_loads_completed_before_first_dispatch=sum(r['completes_before_first_dispatch'] is True for r in unique_loads.values()),
        movement_jobs=len(jobs), handoff_subscriptions=sum(len(j['handoff_subscriptions']) for j in jobs),
        joined_actions=sum(s['creates_action'] is False for j in jobs for s in j['handoff_subscriptions']),
        handoff_subscriptions_before_ready=sum(s['attached_before_engine_ready'] is True for j in jobs for s in j['handoff_subscriptions']),
        matched_remote_transfer_ids=sorted({r['transfer_id'] for j in jobs for r in j['remote_receipts']}),
        movement_states=dict(collections.Counter(j['state'] for j in jobs)))
    reports[variant] = dict(summary=counts, epochs=list(epochs.values()), movement_jobs=jobs)
    print(json.dumps(dict(variant=variant, summary=counts)), flush=True)

OUT.mkdir(parents=True, exist_ok=False)
save(OUT/'handoff_timing.json', dict(kind='d1131_routing_handoff_timing_retained_receipts_v1',
    variants=reports, full_reference=full_reference, violations=violations, unknown=unknown,
    observed_timing_checks_passed=not violations,
    timing_evidence_complete=not unknown,
    timing_contract_passed=not violations and not unknown,
    limitations=[
        'A plan registration or queue executor entry is not itself a physical transfer.',
        'RoutingHandoff is not DelayedHandoff: there is no requirement to wait for engine-ready. Preserved Delayed-only checks do not apply to this variant.',
        'Shared jobs are attributed to their earliest unambiguous subscriber, not every later handoff.',
        'Native loads use invoked=true receipts in the recorded owner/clock; absent receipts stay unknown.',
        'An omitted invocation flag is not treated as false. Unknown timing coverage does not invalidate the complete performance observation or trigger a rerun.',
        'Natural activations and chosen plans can differ. This does not replace controlled same-plan A2/A4.',
        'No continuous residency or causal attribution is inferred from a later matching adapter hit.',
        'No independent numerical correctness or formal performance ranking is established by this audit.']))
with (OUT/'audit_handoff_timing.py.snapshot').open('x') as f:
    f.write(Path(__file__).read_text())
save(OUT/'manifest.json', dict(passed=not violations and not unknown,
    observed_timing_checks_passed=not violations, timing_evidence_complete=not unknown,
    source_refs=sources,
    analysis_script=ref(Path(__file__)), reused_analysis=ref(ROOT/'paper_results/ieee_tc/ablation/d1126_7b_delayed_handoff_w1/audit_delayed_timing.py'),
    implementation_sources=[ref(ROOT/n) for n in ('scripts/run_all_experiments.py', 'faaslora/preloading/preloading_manager.py')],
    identity=identity, resources=p.cgroup_snapshot(group), plan_sha256=p.check_plan(),
    metric_protocol_sha256=p.digest(ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md'),
    single_run_sha256=p.digest(ROOT/'docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md'),
    outputs=[ref(x) for x in sorted(OUT.iterdir()) if x.is_file()]))
print(json.dumps(dict(timing_contract_passed=not violations and not unknown,
    observed_timing_checks_passed=not violations, timing_evidence_complete=not unknown,
    violations=len(violations), unknown=len(unknown))), flush=True)
