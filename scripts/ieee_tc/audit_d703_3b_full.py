#!/usr/bin/env python3
"""Strict append-only audit for the D703 3B full W0 replay.

D703 is deliberately a full native-contract/SLO qualification, not a numerical
adapter-correctness oracle.  The audit preserves that boundary: it accepts the
complete fixed-length request and lifecycle evidence, but keeps ``n_correct``
unknown and therefore does not manufacture a G1 ranking result.
"""

from __future__ import annotations

import collections
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from faaslora.experiment.instance_pool import FrozenServiceProfiles, ServiceObservationClass
from faaslora.preloading.preloading_planner import FrozenPreparationProfiles
from scripts import ieee_tc_preflight as p


ROOT = p.ROOT
RAW = Path("/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d703_20261019/3b_full_w0")
RESULT = next(RAW.glob("experiment_results_full*.json"))
LAUNCH = RAW / "launch.json"
OUTCOME = RAW / "launch.launch/physical_deployment/main_outcome.json"
SUMMARY = RAW / "launch.launch/physical_deployment/summary.json"
PHYSICAL = RAW / "launch.launch/physical_deployment/physical_resource_summary.json"
WATCHDOG = RAW / "launch.launch/watchdog.jsonl"
CONFIG = ROOT / "configs/ieee_tc/20261019_d703_3b_full_w0_d701profile.yaml"
SERVICE_PROFILE = ROOT / "paper_results/ieee_tc/p2_backend/d701_3b_same_spec_profile/export/service.json"
PREPARATION_PROFILE = ROOT / "paper_results/ieee_tc/p2_backend/d701_3b_same_spec_profile/export/preparation.json"
PROFILE_MANIFEST = ROOT / "paper_results/ieee_tc/p2_backend/d701_3b_same_spec_profile/export/manifest.json"
OUT = ROOT / "paper_results/ieee_tc/p2_backend/d703_3b_full_w0_d701profile"


def clipped_interval_union(start, end, spans):
    """Witnessed temporal overlap, not attributable delay or a token stall."""
    if not all(type(v) in (int, float) and math.isfinite(v) for v in (start, end)) or start > end:
        raise ValueError('finite ordered observation interval required')
    clipped = []
    for a, b in spans:
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (a, b)) or a > b:
            raise ValueError('finite ordered peer interval required')
        if max(start, a) < min(end, b):
            clipped.append((max(start, a), min(end, b)))
    total, previous = 0., start
    for a, b in sorted(clipped):
        total += max(0., b-max(a, previous))
        previous = max(previous, b)
    return total, len(clipped)


D716_TTFT_STAGES = ('arrival_to_admission_ms', 'admission_to_acquired_ms',
    'executable_to_engine_ms', 'engine_entry_to_queue_ms',
    'native_engine_queue_ms', 'native_prefill_ms')


def partition_common_slo_request(request, groups, *, clock_id):
    """Reuse native boundaries and preparation receipts; no inferred spans."""
    from faaslora.metrics.metrics_collector import NativeV1TokenTimeline
    from scripts.ieee_tc.check_d299_full_context import common_slo_flags
    from scripts.ieee_tc.analyze_d297_full import retained_preparation_row
    n, ext = request['native_token_timing'], request['external_arrival_timing']
    b = NativeV1TokenTimeline.service_breakdown(n,
        admitted_at=n['controller_admitted_monotonic_s'],
        completed_at=n['controller_completed_monotonic_s'], clock_id=clock_id)
    prep = retained_preparation_row(request)
    arrival, first, last = (ext['planned_arrival_s'], n['native_first_token_monotonic_s'],
                           n['native_last_token_monotonic_s'])
    tokens, prompt = request['completion_tokens'], int(n['actual_prompt_tokens'])
    if (type(tokens) is not int or tokens < 2 or n['native_output_tokens'] != tokens
            or n['actual_prompt_tokens'] != prompt or tokens != request['requested_completion_tokens']):
        raise ValueError('complete matched multi-token native counts required')
    values = dict(ttft_ms=(first-arrival)*1000, tpot_ms=(last-first)*1000/(tokens-1),
        e2e_ms=(n['controller_completed_monotonic_s']-arrival)*1000,
        arrival_to_admission_ms=(n['controller_admitted_monotonic_s']-arrival)*1000,
        admission_to_acquired_ms=prep['admission_to_acquired_ms'],
        executable_to_engine_ms=prep['executable_to_engine_ms'],
        **{k:b[k] for k in D716_TTFT_STAGES[3:]})
    if any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError('finite nonnegative native stages required')
    error = abs(math.fsum(values[k] for k in D716_TTFT_STAGES)-values['ttft_ms'])
    for key, stored in (('ttft_ms','overall_ttft_ms'), ('tpot_ms','tpot_ms'), ('e2e_ms','overall_e2e_ms')):
        error = max(error, abs(values[key]-request[stored]))
    ingress = {k:request[k] for k in ('arrival_release_lateness_ms',
        'dispatch_window_wait_ms', 'runtime_slot_wait_ms')}
    if any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in ingress.values()):
        raise ValueError('complete nonnegative ingress stages required')
    error = max(error, abs(math.fsum(ingress.values())-values['arrival_to_admission_ms']))
    if error > 1:
        raise ValueError('native timing partition exceeds 1 ms')
    flags = common_slo_flags(groups, prompt_tokens=prompt, output_tokens=tokens,
                            ttft_ms=values['ttft_ms'], tpot_ms=values['tpot_ms'])
    label = ('pass' if flags['joint_pass'] else 'tpot_only' if flags['ttft_pass']
             else 'ttft_only' if flags['tpot_pass'] else 'both')
    receipt = request['gpu_reference_evidence']['receipt']
    return dict(request_id=request['request_id'], ordinal=ext['index'],
        phase=ext['index']//500, adapter_id=request['adapter_id'], owner=prep['owner_id'],
        replica=request['instance_id'], worker_pid=receipt['worker_pid'],
        tier=prep['tier'], native_source=prep['native_source'],
        output_tokens=tokens, prompt_tokens=prompt, output_bin=f'{(tokens-1)//64*64+1}-{((tokens-1)//64+1)*64}',
        slo_group=label, **flags, **values, **ingress, stage_identity_error_ms=error,
        dominant_ttft_stage=max(D716_TTFT_STAGES, key=values.__getitem__),
        routing_nested_ms=request['control_path_total_us']/1000,
        stale_rechecks=prep['stale_rechecks'], native_load_invoked=prep['native_load_invoked'],
        before_final_load_ms=prep['before_final_load_ms'], final_native_load_ms=prep['final_native_load_ms'],
        admitted=n['controller_admitted_monotonic_s'], scheduled=n['native_scheduled_monotonic_s'],
        first=first, last=last, load_start=receipt['native_load_started_monotonic_s'],
        load_end=receipt['native_load_completed_monotonic_s'])


def diagnose_common_slo(raw: Path, output: Path, metric_audit: Path) -> None:
    """D716: all-request, source-bound, CPU-only descriptive stage diagnosis."""
    import ijson
    from scripts.ieee_tc.audit_d402_serverlessllm_path import timing_stats
    if output.exists():
        raise FileExistsError(output)
    metric = read(metric_audit)
    if (metric['kind'] != 'native_common_slo_reanalysis_d714_v1' or not metric['passed']
            or metric['source_run'] != str(raw) or metric['counts']['n'] != 4000):
        raise ValueError('complete same-run D714 common-SLO audit required')
    refs = {Path(x['path']):x['sha256'] for x in metric['source_refs']}
    result = next(x for x in refs if x.name.startswith('experiment_results_full'))
    freeze_path = ROOT/'paper_results/ieee_tc/protocols/20261017_3b_slo_d648_frozen.json'
    summary_path = raw/'launch.launch/physical_deployment/summary.json'
    content_path = ROOT/'paper_results/ieee_tc/inputs/20260927_3b_remote_content_index.json'
    v1, v2 = (ROOT/'docs/ieee_tc'/x for x in
              ('METRIC_PROTOCOL_FROZEN_V1.md','METRIC_PROTOCOL_SINGLE_RUN_V2.md'))
    sources = [result, freeze_path, summary_path, content_path, v1, v2]
    for path in sources:
        if digest(path) != refs[path]:
            raise ValueError(f'sealed source changed: {path}')
    freeze, summary = read(freeze_path), read(summary_path)
    rows = []
    with result.open('rb') as handle:
        for request in ijson.items(handle, 'detailed_results.tc_ieee_full.requests.item', use_float=True):
            rows.append(partition_common_slo_request(request, freeze['groups'], clock_id=summary['clock_id']))
    rows.sort(key=lambda r:r['ordinal'])
    if (len(rows) != 4000 or len({r['request_id'] for r in rows}) != 4000
            or [r['ordinal'] for r in rows] != list(range(4000))):
        raise ValueError('all offered requests required; no slow-request subset')
    counts = dict(n=len(rows), ttft_pass=sum(r['ttft_pass'] for r in rows),
        tpot_eligible=len(rows), tpot_pass=sum(r['tpot_pass'] for r in rows),
        joint_pass=sum(r['joint_pass'] for r in rows))
    if counts != metric['counts']:
        raise ValueError('D714 population/count mismatch')
    violations = {r['request_id']:r for r in metric['slo_violations']}
    if {r['request_id'] for r in rows if not r['joint_pass']} != set(violations):
        raise ValueError('D714 violation identities differ')
    for r in rows:
        if r['request_id'] in violations:
            old = violations[r['request_id']]
            if any(r[k] != old[k] for k in ('ordinal','ttft_ms','tpot_ms','e2e_ms','ttft_pass','tpot_pass','joint_pass')):
                raise ValueError('D714 per-request metrics/flags differ')
    owners = collections.defaultdict(list)
    for r in rows:
        owners[r['owner']].append(r)
    for owner_rows in owners.values():
        if len({(r['worker_pid'],r['replica']) for r in owner_rows}) != 1:
            raise ValueError('owner/worker/replica mapping is not unique')
        for r in owner_rows:
            peers = [q for q in owner_rows if q['request_id'] != r['request_id']]
            for name, spans in (
                ('prefill', ((q['scheduled'],q['first']) for q in peers)),
                ('demand_load', ((q['load_start'],q['load_end']) for q in peers if q['native_load_invoked'])),
                ('decode', ((q['first'],q['last']) for q in peers))):
                duration, count = clipped_interval_union(r['first'], r['last'], spans)
                r[f'peer_{name}_overlap_ms'] = duration*1000
                r[f'peer_{name}_count'] = count
                r[f'peer_{name}_fraction'] = duration/(r['last']-r['first'])
    stats_keys = (*D716_TTFT_STAGES, 'ttft_ms','tpot_ms','e2e_ms',
        'arrival_release_lateness_ms','dispatch_window_wait_ms','runtime_slot_wait_ms',
        'routing_nested_ms','before_final_load_ms','final_native_load_ms','stale_rechecks',
        'peer_prefill_fraction','peer_demand_load_fraction','peer_decode_fraction')
    def summarize(selected):
        return dict(n=len(selected), joint_pass=sum(r['joint_pass'] for r in selected),
            ttft_pass=sum(r['ttft_pass'] for r in selected), tpot_pass=sum(r['tpot_pass'] for r in selected),
            dominant_ttft_stages=dict(collections.Counter(r['dominant_ttft_stage'] for r in selected)),
            peer_overlap_counts={name:sum(r[f'peer_{name}_count']>0 for r in selected)
                                 for name in ('prefill','demand_load','decode')},
            metrics={k:timing_stats([r[k] for r in selected if r[k] is not None]) for k in stats_keys})
    groups = [dict(dimension='all',label='all',**summarize(rows))]
    for key in ('slo_group','tier','owner','phase','output_bin','native_load_invoked'):
        for value in sorted({r[key] for r in rows}, key=str):
            groups.append(dict(dimension=key,label=value,**summarize([r for r in rows if r[key]==value])))
    for field in ('tier','output_bin'):
        for label in ('pass','ttft_only','tpot_only','both'):
            for value in sorted({r[field] for r in rows}):
                groups.append(dict(dimension=f'slo_group/{field}',label=f'{label}/{value}',
                    **summarize([r for r in rows if r['slo_group']==label and r[field]==value])))
    for key in ('ttft_ms','tpot_ms','e2e_ms'):
        for stat, value in metric['metrics'][key].items():
            if abs(groups[0]['metrics'][key][stat]-value)>1e-8:
                raise ValueError('whole-population metric differs from D714')
    sources += [metric_audit, Path(__file__), ROOT/'scripts/ieee_tc/analyze_d297_full.py',
                ROOT/'scripts/ieee_tc/check_d299_full_context.py',
                Path('/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md')]
    report = dict(kind='d716_all_request_stage_diagnosis_v1', passed=True,
        source_run=str(raw), clock_id=summary['clock_id'], counts=counts, groups=groups,
        max_stage_identity_error_ms=max(r['stage_identity_error_ms'] for r in rows),
        n_correct=metric['n_correct'], numerical_adapter_correctness_qualified=False,
        ranking_eligible=False, physical_gpu_seconds=metric['physical_gpu_seconds'],
        source_refs=[ref(x) for x in sources], confidence_interval=None,
        limitations=['Single-run conditional association, not a causal speedup estimate.',
            'Overlap uses same-owner/same-clock observed envelopes, not GPU kernel time.',
            'Demand-load spans only; proactive preparation is not covered by this overlap count.',
            'Ingress and routing fields are nested; never add them to the six TTFT stages again.',
            'All 4000 requests retained; no new performance execution or protocol change.'])
    output.mkdir(parents=True, exist_ok=False)
    with (output/'requests.csv').open('x') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report['request_table'] = ref(output/'requests.csv')
    with (output/'diagnosis.json').open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps(dict(passed=True, counts=counts, output=str(output),
                         max_stage_identity_error_ms=report['max_stage_identity_error_ms'])))


def retained_routing_attempts(request, *, clock_id):
    """D717: rejected attempts, not final-acquisition rechecks or saved time."""
    attempts = request['gpu_reference_evidence'].get('prior_routing_attempts', [])
    if not isinstance(attempts, list):
        raise ValueError('prior routing attempts must be an explicit list when present')
    rows, seen = [], set()
    for index, attempt in enumerate(attempts):
        snapshot = attempt['routing_snapshot']
        if snapshot['selected_reference_acquired'] is not False:
            raise ValueError('prior attempt must not assert an acquired selected source')
        replica = snapshot['selected_replica_id']
        sources = snapshot['sources']
        selected = sources[replica]
        source, collection = selected['source'], selected['source_collection']
        if collection['mode'] != 'confirmed_snapshot_cache_v1':
            raise ValueError('D717 requires the retained confirmed-cache contract')
        replicas = list(sources)
        absence_indexes = collection['frontier_absence_replica_indexes']
        if (not isinstance(absence_indexes, list) or len(set(absence_indexes)) != len(absence_indexes)
                or any(type(i) is not int or not 0 <= i < len(replicas) for i in absence_indexes)):
            raise ValueError('invalid retained frontier-absence membership')
        absent = replicas.index(replica) in absence_indexes
        if absent and source['native']:
            raise ValueError('absence cannot classify a native source')
        reason = attempt.get('native_source_conflict')
        if reason is None:
            conflict = attempt.get('last_conflict')
            if conflict is not None:
                if conflict['acquired'] is not False or conflict['clock_id'] != clock_id:
                    raise ValueError('GPU rejection receipt differs from the attempt')
                reason = 'gpu:' + conflict['reason']
            else:
                conflict = attempt['native_host_source']['conflict']
                if conflict['held'] is not False or conflict['clock_id'] != clock_id:
                    raise ValueError('HOST rejection receipt differs from the attempt')
                reason = 'host:' + conflict['reason']
        pending = attempt['pending_kv_admission']
        opened, closed = pending['receipt'], pending['close_receipt']
        final = closed['receipt']
        if (pending['state'] != 'closed' or closed['closed'] is not True
                or opened['state'] != 'pending' or final['state'] != 'withdrawn'
                or any(x['intent_id'] != pending['intent_id'] for x in (opened, closed, final))
                or any(x['clock_id'] != clock_id for x in (opened, final))
                or opened['scheduler_owner_id'] != final['scheduler_owner_id']):
            raise ValueError('rejected attempt lacks its same-owner closed pending receipt')
        start, end = opened['observed_at'], final['observed_at']
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (start,end)) or start > end:
            raise ValueError('invalid pending interval')
        view = (replica, selected['native_owner_id'], selected['native_epoch'],
                selected['native_captured_monotonic_s'])
        repeated = absent and view in seen
        if absent:
            seen.add(view)
        rows.append(dict(request_id=request['request_id'], attempt_index=index,
            reason=reason, selected_replica=replica, selected_tier=source['tier'],
            selected_native=source['native'], frontier_absence=absent,
            repeated_absence_view=repeated, native_owner=view[1], native_epoch=view[2],
            native_captured=view[3], pending_start=start, pending_end=end,
            pending_residence_ms=(end-start)*1000))
    return rows


def diagnose_routing_retries(raw: Path, output: Path, stage_audit: Path) -> None:
    """Join ALL requests to the sealed D716 population; no GPU execution."""
    import ijson
    from scripts.ieee_tc.audit_d402_serverlessllm_path import timing_stats
    if output.exists():
        raise FileExistsError(output)
    stage = read(stage_audit)
    if (stage['kind'] != 'd716_all_request_stage_diagnosis_v1' or not stage['passed']
            or stage['source_run'] != str(raw) or stage['counts']['n'] != 4000):
        raise ValueError('complete same-run D716 diagnosis required')
    refs = {Path(x['path']):x['sha256'] for x in stage['source_refs']}
    result = next(x for x in refs if x.name.startswith('experiment_results_full'))
    table = Path(stage['request_table']['path'])
    if digest(result) != refs[result] or digest(table) != stage['request_table']['sha256']:
        raise ValueError('sealed source changed')
    with table.open() as handle:
        previous = list(csv.DictReader(handle))
    by_id = {r['request_id']:r for r in previous}
    if len(previous) != 4000 or len(by_id) != 4000:
        raise ValueError('complete unique stage table required')
    rows, attempts = [], []
    with result.open('rb') as handle:
        for request in ijson.items(handle, 'detailed_results.tc_ieee_full.requests.item', use_float=True):
            old = by_id[request['request_id']]
            if int(old['ordinal']) != request['external_arrival_timing']['index']:
                raise ValueError('D716 request ordinal differs')
            entries = retained_routing_attempts(request, clock_id=stage['clock_id'])
            attempts.extend(entries)
            fields = ('ttft_ms','tpot_ms','runtime_slot_wait_ms','arrival_to_admission_ms')
            faster = [a for a in entries if a['reason'] == 'faster native source appeared before admission']
            rows.append(dict(request_id=request['request_id'], ordinal=int(old['ordinal']),
                slo_group=old['slo_group'], tier=old['tier'], attempts=len(entries),
                faster_source_attempts=len(faster),
                repeated_absence_attempts=sum(a['repeated_absence_view'] for a in entries),
                pending_residence_ms=math.fsum(a['pending_residence_ms'] for a in entries),
                **{k:float(old[k]) for k in fields}))
    rows.sort(key=lambda r:r['ordinal'])
    if [r['ordinal'] for r in rows] != list(range(4000)) or len({r['request_id'] for r in rows}) != 4000:
        raise ValueError('all offered IDs required')
    def summarize(group):
        return dict(n=len(group), slo_groups=dict(collections.Counter(r['slo_group'] for r in group)),
            attempt_count=sum(r['attempts'] for r in group),
            faster_source_attempt_count=sum(r['faster_source_attempts'] for r in group),
            repeated_absence_count=sum(r['repeated_absence_attempts'] for r in group),
            metrics={k:timing_stats([r[k] for r in group]) for k in
                     (*fields,'pending_residence_ms','attempts')})
    groups = {name:summarize(group) for name, group in (
        ('all',rows), ('no_retry',[r for r in rows if not r['attempts']]),
        ('with_retry',[r for r in rows if r['attempts']]),
        ('faster_source',[r for r in rows if r['faster_source_attempts']]),
        ('repeated_absence',[r for r in rows if r['repeated_absence_attempts']]))}
    for label in ('pass','ttft_only','tpot_only','both'):
        groups[label] = summarize([r for r in rows if r['slo_group']==label])
    report = dict(kind='d717_all_request_routing_retries_v1', passed=True,
        source_run=str(raw), groups=groups,
        attempt_reasons=dict(collections.Counter(a['reason'] for a in attempts)),
        attempt_histogram=dict(sorted(collections.Counter(r['attempts'] for r in rows).items())),
        reason_frontier_counts=dict(collections.Counter(a['reason'] for a in attempts if a['frontier_absence'])),
        source_refs=[ref(x) for x in (result,table,stage_audit,Path(__file__))],
        n_correct=None, ranking_eligible=False, performance_execution=False,
        limitations=['Rejected routing attempts differ from final native-acquisition stale_rechecks.',
            'Pending residence is a witnessed nested interval, not full retry time or GPU holding time.',
            'Conditional groups are not causal counterfactuals; do not subtract delays from old results.',
            'All offered requests retained, including zero-retry and passing controls.'])
    output.mkdir(parents=True, exist_ok=False)
    report['tables'] = []
    for name, values in (('requests',rows), ('attempts',attempts)):
        path = output/(name+'.csv')
        with path.open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(values)
        report['tables'].append(ref(path))
    with (output/'diagnosis.json').open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps(dict(output=str(output),attempt_reasons=report['attempt_reasons'],groups=groups)))


def audit_closed_outcome(raw: Path, output: Path, metric_audit: Path) -> None:
    """Join closed remote, control and watchdog evidence after the SLO audit."""
    import ijson
    from scripts.ieee_tc.check_d299_full_context import project
    if output.exists():
        raise FileExistsError(output)
    metric = read(metric_audit)
    if metric['source_run'] != str(raw) or metric['passed'] is not True:
        raise ValueError('same-run completed native metric audit required')
    source = next(x for x in metric['source_refs'] if Path(x['path']).name.startswith('experiment_results_full'))
    result = Path(source['path'])
    if digest(result) != source['sha256']:
        raise ValueError('metric source changed')
    outcome = raw/'launch.launch/physical_deployment/main_outcome.json'
    meta, transfers = project(outcome)
    journal = raw.parent/'remote_transfers.jsonl'
    server_rows = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    server = {r['transfer_id']:r for r in server_rows}
    if len(server) != len(server_rows) or len(server) != len(transfers) or set(server) != {r['http_transfer_id'] for r in transfers}:
        raise ValueError('client/server remote transfer identities differ')
    health = read(raw.parent/'remote_health.json')
    clock = health['3b']['clock_id']
    for r in transfers:
        s = server[r['http_transfer_id']]
        if (r['state'] != 'published' or r['content_verified'] is not True
                or r['published_archive_verified'] is not True
                or r['artifact_id'] != s['artifact_id'] or r['remote_archive_sha256'] != s['archive_sha256']
                or not r['transferred_bytes'] == s['archive_bytes'] == s['bytes_read'] == s['bytes_written']
                or s['clock_id'] != clock or s['outcome'] != 'sent'
                or r['remote_pack_performed'] is not False or s['pack_performed'] is not False
                or s['temporary_created'] is not False):
            raise ValueError('remote published delivery/bytes/no-pack join failed')
    watchdog = raw/'launch.launch/watchdog.jsonl'
    events, memory = collections.Counter(), collections.Counter()
    peak = swap = 0
    available, free = math.inf, math.inf
    with watchdog.open() as stream:
        for line in stream:
            r = json.loads(line); events[r['event']] += 1
            if r['event'] != 'resource_sample':
                continue
            if r['decision']['abort_reasons']:
                raise ValueError('watchdog aborted this run')
            for k, value in r['service']['memory.events'].items():
                memory[k] = max(memory[k], value)
            peak = max(peak, r['service']['memory.peak'])
            swap = max(swap, r['service']['memory.swap.current'])
            available = min(available, r['host']['available_bytes'])
            free = min(free, *(x['free_bytes'] for x in r['filesystems']))
    if not events['resource_sample'] or events['gpu_terminal_census'] != 1 or events['service_domain_gone'] != 1:
        raise ValueError('complete watchdog closure required')
    control = {}
    with result.open('rb') as stream:
        for prefix, event, value in ijson.parse(stream, use_float=True):
            if prefix == 'detailed_results.tc_ieee_full.scale_down_events':
                if event != 'number':
                    raise ValueError('scalar observed event count required')
                control[prefix.rsplit('.',1)[1]] = value
    if 'scale_down_events' not in control:
        raise ValueError('scale-in count must be recorded')
    with result.open('rb') as stream:
        control['scale_up_event_log'] = list(ijson.items(stream,
            'detailed_results.tc_ieee_full.scale_up_events.item', use_float=True))
    control['completed_scale_up_events'] = len(control['scale_up_event_log'])
    with result.open('rb') as stream:
        control['scale_down_event_log'] = list(ijson.items(stream,
            'detailed_results.tc_ieee_full.scale_down_event_log.item', use_float=True))
    if control['scale_down_events'] != len(control['scale_down_event_log']):
        raise ValueError('scale-in log/count mismatch')
    for event in control['scale_down_event_log']:
        times = [event[k] for k in ('observed_monotonic_s', 'withdrawn_monotonic_s',
                                    'drained_monotonic_s', 'released_monotonic_s')]
        if (event['reason'] != 'ieee_all_low_through_cooldown'
                or not all(math.isfinite(x) for x in times)
                or any(a>b for a,b in zip(times,times[1:]))):
            raise ValueError('ordered natural scale-in lifecycle required')
    summary = read(raw/'launch.launch/physical_deployment/summary.json')
    audit = dict(kind='d713_closed_delivery_and_control_context_v1', passed=True,
        native_metric_audit=ref(metric_audit), terminal_meta=meta, control=control,
        physical_allocation_count=len(summary['allocation_journals']),
        all_allocations_released=all(j['released'] for j in summary['allocation_journals']),
        remote=dict(transfers=len(transfers), wire_bytes=sum(r['transferred_bytes'] for r in transfers),
                    all_server_client_ids_bytes_hashes_match=True, request_time_packing=False),
        resources=dict(events=dict(events), memory_events_max=dict(memory), service_peak_bytes=peak,
            max_swap_bytes=swap, min_host_available_bytes=available, min_free_disk_bytes=free),
        source_refs=[ref(x) for x in (result,outcome,journal,watchdog,raw.parent/'remote_health.json',
            raw.parent/'remote_monitor.log',raw.parent/'remote_stop.log',Path(__file__))])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(audit, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k:v for k,v in audit.items() if k != 'source_refs'}, indent=2))


def reevaluate_common_slo(raw: Path, output: Path, *, expected_count: int) -> None:
    """R1: append-only native SLO correction, with bounded streaming memory.

    No remote call, GPU launch, threshold change or modification of v1 evidence.
    Both D703 full W0 and D713's registered prefix use the SAME D648 thresholds.
    """
    import ijson
    from faaslora.datasets.workload_generator import FrozenReplayPlan
    from faaslora.metrics.metrics_collector import NativeV1TokenTimeline
    from scripts.ieee_tc.check_d299_full_context import (
        closed_physical_integral, common_slo_flags, matched_controller_rows)
    from scripts.ieee_tc.audit_d402_serverlessllm_path import timing_stats
    if output.exists():
        raise FileExistsError(output)
    if expected_count not in (100, 1000, 4000):
        raise ValueError('registered prefix100/prefix1000 or full4000 only')
    results = list(raw.glob('experiment_results_full*.json'))
    if len(results) != 1:
        raise ValueError('exactly one completed source result required')
    result = results[0]
    launch_path = raw/'launch.json'
    summary_path = raw/'launch.launch/physical_deployment/summary.json'
    physical_path = raw/'launch.launch/physical_deployment/physical_resource_summary.json'
    launch, summary, physical = map(read, (launch_path, summary_path, physical_path))
    if (not all(launch.get(k) is True for k in
                ('pass', 'service_path_removed', 'native_gpu_context_release_confirmed'))
            or any(launch.get(k) != 0 for k in
                   ('service_returncode', 'watchdog_returncode', 'replay_returncode'))):
        raise ValueError('actual gated release required before offline analysis')
    result_sha = digest(result)
    if (physical['result_file']['sha256'] != result_sha
            or physical['summary_sha256'] != digest(summary_path)
            or physical['measurement_complete'] is not True or physical['open_lease_ids']):
        raise ValueError('source-bound complete physical sidecar required')
    gpu_s = closed_physical_integral(summary, expected_count=expected_count)
    for j in summary['allocation_journals']:
        if j['released'] is not True or digest(Path(j['path'])) != j['sha256']:
            raise ValueError('closed allocation journal SHA mismatch')
    if digest(Path(summary['request_terminals_path'])) != summary['request_terminals_sha256']:
        raise ValueError('request terminal journal changed')
    ext = launch['external_replay']
    plan = FrozenReplayPlan.load(ext['plan']['source_path'], count=expected_count)
    if plan.identity() != ext['plan'] or abs(ext['replay_t0_s']-ext['deployment_notice_s']-60) > 1e-6:
        raise ValueError('frozen source/prefix/arrival contract mismatch')
    freeze_path = ROOT/'paper_results/ieee_tc/protocols/20261017_3b_slo_d648_frozen.json'
    if digest(freeze_path) != 'd15123570c4a8d830515ca942d1db553c6717ffc1a40ade5c1546d35e2bed9d8':
        raise ValueError('frozen D648 evaluation changed')
    freeze = read(freeze_path)
    v1, v2 = (ROOT/'docs/ieee_tc'/name for name in
              ('METRIC_PROTOCOL_FROZEN_V1.md', 'METRIC_PROTOCOL_SINGLE_RUN_V2.md'))
    if (digest(v1) != freeze['metric_protocol_sha256']
            or digest(v2) != freeze['single_run_amendment_sha256']):
        raise ValueError('metric protocol identity mismatch')
    warm = ROOT/freeze['calibration']['path']
    if digest(warm) != freeze['calibration']['sha256']:
        raise ValueError('warm reference identity mismatch')
    content_path = ROOT/'paper_results/ieee_tc/inputs/20260927_3b_remote_content_index.json'
    index = read(content_path)
    content = {a['id']: hashlib.sha256(json.dumps(sorted(a['files'], key=lambda f:f['path']),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest() for a in index['artifacts']}
    if len(content) != len(index['artifacts']) or len(content) != 500:
        raise ValueError('existing full content index required')
    rows = matched_controller_rows(result, 'tc_ieee_full', content, expected_count=expected_count)
    if set(rows) != {e.request_id for e in plan.entries}:
        raise ValueError('offered population mismatch')
    counts = collections.Counter(n=0, ttft_pass=0, tpot_eligible=0, tpot_pass=0, joint_pass=0)
    phases, failed_slo = [collections.Counter() for _ in range(math.ceil(expected_count/500))], []
    metrics = {k:[] for k in ('ttft_ms', 'tpot_ms', 'e2e_ms')}
    max_error = {k:0. for k in metrics}
    for ordinal, entry in enumerate(plan.entries):
        r, source = rows[entry.request_id], json.loads(entry.source_json)
        n, a, e, d, c = r['native'], r['arrival'], r['submit'], r['native_entry_s'], r['completed']
        if (r['source_item_sha256'] != entry.source_sha256 or r['adapter_id'] != source['adapter_id']
                or r['target_tokens'] != min(source['expected_output_tokens'], 256)
                or abs(a-ext['replay_t0_s']-entry.offset_s) > 1e-6):
            raise ValueError('source/adapter/target/arrival mismatch')
        NativeV1TokenTimeline.service_breakdown(n, admitted_at=n['controller_admitted_monotonic_s'],
            completed_at=c, clock_id=summary['clock_id'])
        f, l, nt = n['native_first_token_monotonic_s'], n['native_last_token_monotonic_s'], r['target_tokens']
        if (n['native_terminal_observed'] is not True or n['native_output_tokens'] != nt
                or not all(math.isfinite(t) for t in (a,e,d,f,l,c)) or not a <= e <= d <= f <= l <= c):
            raise ValueError('native terminal/count/time mismatch')
        values = dict(ttft_ms=(f-a)*1000, tpot_ms=(l-f)*1000/(nt-1) if nt>1 else None,
                      e2e_ms=(c-a)*1000)
        for key, value in values.items():
            saved = r['stored'][key]
            if value is None:
                if saved is not None:
                    raise ValueError('single-token TPOT must remain N/A')
            else:
                error = abs(value-saved)
                if not math.isfinite(error) or error > 1:
                    raise ValueError('native metric recomputation exceeds 1 ms')
                max_error[key] = max(max_error[key], error)
                metrics[key].append(value)
        flags = common_slo_flags(freeze['groups'], prompt_tokens=r['native_prompt_tokens'],
            output_tokens=nt, ttft_ms=values['ttft_ms'], tpot_ms=values['tpot_ms'])
        cts = dict(n=1, ttft_pass=int(flags['ttft_pass']), tpot_eligible=int(nt>1),
                   tpot_pass=int(flags['tpot_pass'] is True), joint_pass=int(flags['joint_pass']))
        counts.update(cts); phases[ordinal//500].update(cts)
        if not flags['joint_pass']:
            failed_slo.append(dict(request_id=entry.request_id, ordinal=ordinal,
                native_prompt_tokens=r['native_prompt_tokens'], **values, **flags))
    with result.open('rb') as stream:
        legacy = next(ijson.items(stream, 'detailed_results.tc_ieee_full.slo_attainment', use_float=True))
    sources = [result, launch_path, summary_path, physical_path, freeze_path, warm, content_path,
               v1, v2, Path(__file__), ROOT/'scripts/ieee_tc/check_d299_full_context.py']
    if raw == RAW:
        seal = read(OUT/'audit.json')
        if result_sha != seal['source_result']['sha256'] or digest(launch_path) != seal['launch']['sha256']:
            raise ValueError('D703 original sealed source changed')
        sources.append(OUT/'audit.json')
    audit = dict(kind='native_common_slo_reanalysis_d714_v1', source_run=str(raw),
        reuse_level='R1', passed=True, counts=dict(counts),
        joint_slo_fraction=counts['joint_pass']/expected_count,
        ttft_marginal_fraction=counts['ttft_pass']/expected_count,
        tpot_conditional_fraction=counts['tpot_pass']/counts['tpot_eligible'] if counts['tpot_eligible'] else None,
        observed_latency_feasible=counts['joint_pass']*100 >= expected_count*95,
        numerical_adapter_correctness_qualified=False, n_correct=summary['n_correct'], ranking_eligible=False,
        full_w0=expected_count==4000, development_only=True, confidence_interval=None,
        physical_gpu_seconds=gpu_s, gpu_seconds_per_offered_request=gpu_s/expected_count,
        metrics={k:timing_stats(v) for k,v in metrics.items()}, max_recompute_error_ms=max_error,
        phases_500=[dict(x) for x in phases], slo_violations=failed_slo,
        legacy_runtime_ttft_only_fraction=legacy, legacy_is_common_joint_slo=False,
        common_slo_sha256=digest(freeze_path), source_refs=[ref(x) for x in sources],
        note='No simulation, rerun, threshold change or inference of independent numerical correctness.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(audit, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('source_refs','slo_violations')}, indent=2))


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ref(path: Path) -> dict:
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def remote_state() -> dict:
    """Read-only post-run witness; this function never mutates the remote node."""
    state = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "primelora-artifact-174",
         "systemctl --user show primelora-artifact-tc-3b.service "
         "-p ActiveState -p SubState -p MainPID -p Result"],
        check=True, capture_output=True, text=True,
    )
    fields = dict(line.split("=", 1) for line in state.stdout.splitlines() if "=" in line)
    if fields.get("ActiveState") != "inactive" or fields.get("MainPID") != "0":
        raise ValueError(f"D703 remote service is not closed: {fields}")
    link = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "primelora-artifact-174",
         "cat /sys/class/net/eno1/speed /sys/class/net/eno1/duplex /sys/class/net/eno1/carrier"],
        check=True, capture_output=True, text=True,
    )
    values = link.stdout.split()
    if len(values) != 3:
        raise ValueError(f"unexpected remote link witness: {link.stdout!r}")
    return {
        "service_unit": "primelora-artifact-tc-3b.service",
        "service": fields,
        "link": {"interface": "eno1", "speed_mbps": int(values[0]),
                  "duplex": values[1], "carrier": int(values[2])},
        "command_contract": "read_only_post_run_state_v1",
    }


def calibrate_control_lower(output: Path) -> None:
    """Reuse sealed D703 evidence for the D711 preregistered calibration.

    This mode never contacts the remote machine and never changes the original
    audit. Stream the large outcome with the existing ijson projection pattern.
    """
    import ijson
    import yaml
    from faaslora.datasets.workload_generator import FrozenReplayPlan
    if output.exists():
        raise FileExistsError(output)
    audit = read(OUT / 'audit.json')
    for key in ('config', 'launch', 'outcome'):
        source = audit[key]
        if digest(Path(source['path'])) != source['sha256']:
            raise ValueError(f'sealed D703 {key} changed')
    launch = read(LAUNCH)
    plan = FrozenReplayPlan.load(launch['external_replay']['plan']['source_path'])
    if plan.identity() != launch['external_replay']['plan'] or len(plan.entries) != 4000:
        raise ValueError('D703 source trace identity changed')
    start = launch['external_replay']['replay_t0_s']
    stop = start + plan.entries[1000].offset_s
    cfg = yaml.safe_load(CONFIG.read_text())['resource_coordination']
    with OUTCOME.open('rb') as stream:
        events = ijson.items(stream, 'mechanism_events._ieee_control_events.item', use_float=True)
        result = p.derive_ieee_low_load_control(events, cfg['ieee_scaling'],
            start_s=start, stop_s=stop, min_instances=cfg['min_instances'])
    result.update(source_campaign='d703_3b_full_w0_d701profile',
        prefix_count=1000, development_only=True, held_out=False,
        plan_sha256=p.check_plan(),
        source_refs=[ref(CONFIG), ref(LAUNCH), ref(OUTCOME), ref(OUT/'audit.json'),
                     ref(Path(__file__)), ref(Path(p.__file__))],
        source_trace_sha256=plan.source_sha256,
        scope=p.scope_identity(p.cg_path().name))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('selected','source_refs','scope')}, indent=2))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"append-only audit already exists: {OUT}")
    required = (RESULT, LAUNCH, OUTCOME, SUMMARY, PHYSICAL, WATCHDOG, CONFIG,
                SERVICE_PROFILE, PREPARATION_PROFILE, PROFILE_MANIFEST)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    plan_sha = p.check_plan()
    frozen_v1 = ROOT / "docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md"
    single_run_v2 = ROOT / "docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md"
    if digest(frozen_v1) != "5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22":
        raise ValueError("frozen V1 metric protocol changed")

    result = read(RESULT)
    launch = read(LAUNCH)
    outcome = read(OUTCOME)
    summary = read(SUMMARY)
    physical = read(PHYSICAL)
    config = __import__("yaml").safe_load(CONFIG.read_text())
    service_payload = read(SERVICE_PROFILE)
    preparation_payload = read(PREPARATION_PROFILE)
    profile_manifest = read(PROFILE_MANIFEST)
    scenario = result["detailed_results"]["tc_ieee_full"]
    rows = scenario["requests"]
    if not (len(rows) == 4000 and result["metadata"].get("total_requests") == 4000
            and scenario.get("total") == 4000 and scenario.get("completed") == 4000
            and scenario.get("failed") == 0):
        raise ValueError("D703 is not a complete 4,000-request replay")

    service_sha = digest(SERVICE_PROFILE)
    preparation_sha = digest(PREPARATION_PROFILE)
    if config["resource_coordination"]["ieee_service_profile"]["sha256"] != service_sha:
        raise ValueError("D703 config/service profile SHA mismatch")
    if config["resource_coordination"]["ieee_preparation_profile"]["sha256"] != preparation_sha:
        raise ValueError("D703 config/preparation profile SHA mismatch")
    if profile_manifest.get("development_only") is not True:
        raise ValueError("D701 profile boundary changed unexpectedly")

    loaded_service = FrozenServiceProfiles.load(
        SERVICE_PROFILE, expected_sha256=service_sha,
        model_config=service_payload["model_config"],
        expected_context=service_payload["context"], beta=0.5,
    )
    loaded_preparation = FrozenPreparationProfiles.load(
        PREPARATION_PROFILE, expected_sha256=preparation_sha,
        model_config=preparation_payload["model_config"],
        expected_context=preparation_payload["context"], beta=0.5,
    )
    if len(loaded_service.profiles) != 160 or len(loaded_preparation.profiles) != 96:
        raise ValueError("D701 profile cardinality changed")

    request_ids, adapters = set(), set()
    tiers = collections.Counter()
    service_classes, preparation_classes = set(), set()
    remote_rows = 0
    output_tokens = 0
    for row in rows:
        rid, aid = row.get("request_id"), row.get("adapter_id")
        if not isinstance(rid, str) or not rid or rid in request_ids:
            raise ValueError(f"invalid/duplicate request identity: {rid!r}")
        if not isinstance(aid, str) or not aid:
            raise ValueError(f"invalid adapter identity for {rid!r}")
        request_ids.add(rid)
        adapters.add(aid)
        if row.get("success") is not True or row.get("error") is not None:
            raise ValueError(f"request {rid} failed")
        target = min(int(row["source_expected_output_tokens"]), 256)
        if any(row.get(k) != target for k in
               ("requested_completion_tokens", "completion_tokens", "output_tokens")):
            raise ValueError(f"request {rid} violates fixed output contract")
        timing = row.get("native_token_timing") or {}
        if (int(timing.get("native_output_tokens", -1)) != target
                or timing.get("native_terminal_observed") is not True
                or row.get("completion_token_source") != "vllm_token_ids"
                or row.get("output_contract_match") is not True
                or row.get("generation_contract") != "fixed_length_greedy_v1"
                or not isinstance(row.get("completion_token_ids_sha256"), str)):
            raise ValueError(f"request {rid} lacks native generation evidence")
        output_tokens += target

        evidence = row.get("gpu_reference_evidence") or {}
        intent = evidence.get("intent") or {}
        receipt = evidence.get("receipt") or {}
        source_admission = evidence.get("source_admission") or {}
        source = source_admission.get("source") or {}
        release = evidence.get("release_receipt") or {}
        if (intent.get("lora_name") != aid or receipt.get("lora_name") != aid
                or evidence.get("state") != "released"
                or release.get("released") is not True
                or release.get("live_staged_host_leases") != 0
                or not isinstance(receipt.get("expected_content_sha256"), str)
                or source.get("content_sha256", source.get("expected_content_sha256"))
                != receipt["expected_content_sha256"]):
            raise ValueError(f"request {rid} lacks adapter/content/release binding")
        if evidence.get("confirmed_dispatch_snapshot") is not True:
            raise ValueError(f"request {rid} lacks confirmed dispatch snapshot")

        service_class = ServiceObservationClass(**source_admission["service_class"])
        if service_class not in loaded_service.profiles:
            raise ValueError(f"request {rid} used unsupported service class")
        service_classes.add(service_class)
        tier = source.get("tier")
        tiers[tier] += 1
        if tier != "gpu":
            prep_class = FrozenPreparationProfiles.source_class(
                source, loaded_preparation.size_edges_bytes)
            if prep_class not in loaded_preparation.profiles:
                raise ValueError(f"request {rid} used unsupported preparation class")
            preparation_classes.add(prep_class)
        remote = evidence.get("remote_preparation")
        if remote is not None:
            remote_rows += 1
            if (remote.get("remote_delivery_mode") != "prepublished_gzip_v1"
                    or remote.get("remote_pack_performed") is not False
                    or remote.get("content_verified") is not True
                    or remote.get("published_archive_verified") is not True):
                raise ValueError(f"request {rid} has invalid remote delivery evidence")

    contract = outcome.get("launch_contract") or {}
    if (contract.get("contract") != "ieee_full_integrated_replay_v1"
            or contract.get("replay_scope") != "full_source_v1"
            or contract.get("service_profile_id") != service_sha
            or contract.get("preparation_profile_id") != preparation_sha):
        raise ValueError("D703 launch contract did not bind full source and D701 profiles")
    transfers = outcome.get("remote_transfers") or []
    if len(transfers) != result["metadata"].get("remote_artifact_transfer_count"):
        raise ValueError("remote transfer count mismatch")
    for transfer in transfers:
        if (transfer.get("state") != "published"
                or transfer.get("remote_delivery_mode") != "prepublished_gzip_v1"
                or transfer.get("remote_pack_performed") is not False
                or transfer.get("content_verified") is not True
                or transfer.get("published_archive_verified") is not True):
            raise ValueError("unverified/request-time-packed transfer")
    if result["metadata"].get("request_time_pack_count") != 0:
        raise ValueError("request-time packing was observed")

    if not (launch.get("pass") is True and launch.get("service_returncode") == 0
            and launch.get("watchdog_returncode") == 0
            and launch.get("replay_returncode") == 0
            and launch.get("service_path_removed") is True
            and launch.get("native_gpu_context_release_confirmed") is True):
        raise ValueError("gated launch did not close cleanly")
    if (outcome.get("run_completed") is not True or outcome.get("error") is not None
            or outcome.get("cleanup_errors") or outcome.get("measurement_errors")
            or outcome.get("cleanup", {}).get("complete") is not True):
        raise ValueError("D703 outcome cleanup/measurement contract is incomplete")
    if (summary.get("measurement_complete") is not True
            or summary.get("n_plan") != 4000 or summary.get("n_terminal") != 4000
            or summary.get("n_native_contract_complete") != 4000
            or summary.get("n_interrupted") != 0 or summary.get("open_lease_ids")
            or any(not j.get("released") for j in summary.get("allocation_journals", []))):
        raise ValueError("D703 physical lifecycle summary is incomplete")

    watchdog = [json.loads(line) for line in WATCHDOG.read_text().splitlines() if line.strip()]
    resources = [row for row in watchdog if row.get("event") == "resource_sample"]
    if not resources or any(row.get("decision", {}).get("abort_reasons") for row in resources):
        raise ValueError("watchdog has no clean resource samples")
    event_max = {key: max(row["service"]["memory.events"][key] for row in resources)
                 for key in ("high", "max", "oom", "oom_kill")}
    max_swap = max(row["service"]["memory.swap.current"] for row in resources)
    if any(event_max.values()) or max_swap != 0:
        raise ValueError(f"resource gate failed: {event_max}, swap={max_swap}")
    remote = remote_state()

    # D703 is a full native/SLO qualification, but numerical correctness is
    # intentionally not inferred from token-termination evidence.
    slo_count = round(float(scenario["slo_attainment"]) * 4000)
    audit = {
        "kind": "d703_3b_full_w0_d701profile_audit_v1",
        "campaign_id": "d703_3b_full_w0_d701profile",
        "passed": True,
        "full_replay_complete": True,
        "native_service_qualification": True,
        "numerical_adapter_correctness_qualified": False,
        "ranking_eligible": False,
        "n_plan": 4000,
        "n_terminal": 4000,
        "n_native_contract_complete": 4000,
        "n_correct": None,
        "joint_slo_count_observed": slo_count,
        "joint_slo_fraction_observed": scenario.get("slo_attainment"),
        "mean_ttft_ms": scenario.get("avg_ttft_ms"),
        "p95_ttft_ms": scenario.get("p95_ttft_ms"),
        "mean_tpot_ms": scenario.get("avg_tpot_ms"),
        "mean_e2e_ms": scenario.get("avg_e2e_ms"),
        "physical_gpu_seconds_observed": summary.get("gpu_seconds_observed"),
        "physical_gpu_seconds_per_offered_request": summary.get("gpu_seconds_per_offered_observed"),
        "request_tier_counts": dict(tiers),
        "service_classes_observed": len(service_classes),
        "service_classes_available": len(loaded_service.profiles),
        "preparation_classes_observed": len(preparation_classes),
        "preparation_classes_available": len(loaded_preparation.profiles),
        "unique_adapter_identities": len(adapters),
        "native_output_tokens": output_tokens,
        "remote_transfers": len(transfers),
        "remote_request_preparations": remote_rows,
        "request_time_pack_count": 0,
        "watchdog": {"samples": len(resources), "memory_events_max": event_max,
                     "max_swap_current": max_swap,
                     "service_memory_peak_bytes": max(row["service"]["memory.peak"] for row in resources)},
        "remote_post_run": remote,
        "source_trace_sha256": result["metadata"]["sampling_stats"]["shared_trace_metadata"]["replay_plan"]["source_sha256"],
        "source_result": ref(RESULT), "launch": ref(LAUNCH), "outcome": ref(OUTCOME),
        "summary": ref(SUMMARY), "physical_resource_summary": ref(PHYSICAL),
        "watchdog_raw": ref(WATCHDOG), "config": ref(CONFIG),
        "service_profile": ref(SERVICE_PROFILE), "preparation_profile": ref(PREPARATION_PROFILE),
        "profile_manifest": ref(PROFILE_MANIFEST), "plan_sha256": plan_sha,
        "metric_protocol_v1_sha256": digest(frozen_v1),
        "metric_protocol_v2_sha256": digest(single_run_v2),
        "interpretation": (
            "D703 closes a complete 4,000-request native service replay with the D701 "
            "exact measured profile and observed joint SLO, and improves the prior D682 "
            "diagnostic on physical GPU seconds and SLO. It does not infer numerical "
            "adapter correctness from fixed-length token termination; therefore it is not "
            "a formal G1 ranking point and must not be plotted as one until correctness "
            "is independently qualified."
        ),
    }
    OUT.mkdir(parents=False, exist_ok=False)
    with (OUT / "audit.json").open("x") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with (OUT / "manifest.json").open("x") as stream:
        json.dump({"kind": "d703_3b_full_w0_audit_manifest_v1", "passed": True,
                   "audit": ref(OUT / "audit.json"),
                   "raw_evidence": [ref(path) for path in required]},
                  stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"passed": True, "full_replay_complete": True,
                      "numerical_adapter_correctness_qualified": False,
                      "ranking_eligible": False, "requests": 4000,
                      "joint_slo_count_observed": slo_count,
                      "physical_gpu_seconds": summary.get("gpu_seconds_observed"),
                      "service_classes": len(service_classes),
                      "preparation_classes": len(preparation_classes),
                      "remote_transfers": len(transfers)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--calibrate-control-lower', type=Path)
    parser.add_argument('--reevaluate-common-slo', type=Path)
    parser.add_argument('--audit-closed-outcome', type=Path)
    parser.add_argument('--diagnose-common-slo', type=Path)
    parser.add_argument('--diagnose-routing-retries', type=Path)
    parser.add_argument('--stage-audit', type=Path)
    parser.add_argument('--metric-audit', type=Path)
    parser.add_argument('--raw', type=Path, default=RAW)
    parser.add_argument('--expected-count', type=int, default=4000)
    args = parser.parse_args()
    if sum(x is not None for x in (args.calibrate_control_lower, args.reevaluate_common_slo, args.audit_closed_outcome, args.diagnose_common_slo, args.diagnose_routing_retries)) != 1:
        parser.error('select exactly one offline analysis mode')
    if args.diagnose_routing_retries is not None:
        if args.stage_audit is None:
            parser.error('--stage-audit is required for --diagnose-routing-retries')
        diagnose_routing_retries(args.raw, args.diagnose_routing_retries, args.stage_audit)
    elif args.diagnose_common_slo is not None:
        if args.metric_audit is None:
            parser.error('--metric-audit is required for --diagnose-common-slo')
        diagnose_common_slo(args.raw, args.diagnose_common_slo, args.metric_audit)
    elif args.audit_closed_outcome is not None:
        if args.metric_audit is None:
            parser.error('--metric-audit is required for --audit-closed-outcome')
        audit_closed_outcome(args.raw, args.audit_closed_outcome, args.metric_audit)
    elif args.reevaluate_common_slo is not None:
        reevaluate_common_slo(args.raw, args.reevaluate_common_slo, expected_count=args.expected_count)
    elif args.calibrate_control_lower is not None:
        calibrate_control_lower(args.calibrate_control_lower)
    else:
        parser.error('v1 audit is sealed and its SLO alias is withdrawn; use --reevaluate-common-slo OUTPUT')
