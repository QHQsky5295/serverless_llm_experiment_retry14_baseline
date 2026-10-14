"""Data-only D157/D89 export adaptation; existing validators remain authoritative.

No figures, inference, remote mutation, new profile estimators or default fills.
Run once in the bounded auxiliary domain, after actual source-run release.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
from dataclasses import replace
from bisect import bisect_left
import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import yaml
from scripts import ieee_tc_preflight as p
from faaslora.storage.http_artifact_store import HttpArtifactStoreClient
from faaslora.experiment.instance_pool import (FrozenServiceProfiles, ServiceClassBins,
    ServiceIntervalObservation, NativeServiceIntervalObserver)
from faaslora.preloading.preloading_planner import FrozenPreparationProfiles

ROOT = p.ROOT
RAW = ROOT / 'results/ieee_tc/p2_backend_qualification/d298_20261007'
CURATED = ROOT / 'paper_results/ieee_tc/p2_backend/d298_capacity'


def read(path):
    return json.loads(path.read_text())


def ref(path):
    return dict(path=str(path), sha256=p.digest(path))


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
        separators=(',', ':')).encode()).hexdigest()


def audit_metadata_source_run(raw):
    """Actual installation/cumulative use, not full-model numerical proof."""
    from faaslora.memory.lora_metadata_publication import staging_bytes
    cfg = raw['model_config']
    receipt = raw['metadata_host_installation']
    initial = receipt['metadata_publication']
    snapshots = [initial]
    workers = raw['workers_after']['workers']
    if (cfg.get('ieee_lora_metadata_policy') != 'cpu_live_mapping_v1'
            or len(workers) != 1 or receipt.get('configured') is not True
            or receipt.get('tensor_budget_bytes') != cfg['ieee_native_host_tensor_budget_bytes']
            or receipt.get('worker_pid') != workers[0]['pid']
            or receipt.get('clock_id') != workers[0]['clock_id']):
        raise ValueError('metadata installation lacks exact native HOST/worker binding')
    snapshots.append(workers[0]['metadata_publication'])
    for name in ('sources_after', 'sources_after_eviction'):
        source = raw[name]
        if (source.get('owner_id') != receipt['owner_id']
                or source.get('worker_pid') != receipt['worker_pid']
                or source.get('clock_id') != receipt['clock_id']):
            raise ValueError('metadata source owner changed')
        snapshots.append(source['metadata_publication'])
    expected_bytes = staging_bytes(cfg['max_num_batched_tokens'], cfg['max_loras'], 2)
    counters = ('updates', 'reuse_waits', 'copied_bytes')
    for index, row in enumerate(snapshots):
        if (row.get('policy') != 'cpu_live_mapping_v1' or row.get('installed') is not True
                or row.get('poisoned') is not False or row.get('gpu_buffers_replaced') is not False
                or row.get('native_mapping_invalidation_preserved') is not True
                or row.get('banks') != 2 or row.get('rows') != cfg['max_num_batched_tokens']
                or row.get('slots') != cfg['max_loras'] or row.get('staging_bytes') != expected_bytes
                or any(type(row.get(key)) is not int or row[key] < 0 for key in counters)
                or row['reuse_waits'] > row['updates']
                or (index and any(row[key] < snapshots[index-1][key] for key in counters))):
            raise ValueError('metadata publisher contract/cumulative counter mismatch')
    before, after = initial['accounted_before_bytes'], initial['accounted_after_bytes']
    partition = receipt['workspace_partition']
    workspace = partition['registered_upper_bytes'] + 2 * partition['demand_load_peak_bytes']
    if (any(type(x) is not int or x < 0 for x in (before, after, workspace))
            or after - before < expected_bytes
            or initial['required_minimum_with_staging_bytes'] != before + expected_bytes + workspace
            or after + workspace > receipt['tensor_budget_bytes']
            or receipt['allocation'].get('admitted') is not True
            or receipt['allocation']['before']['accounted_tensor_bytes'] != after
            or initial['updates'] != 0 or initial['copied_bytes'] != 0
            or snapshots[-1]['updates'] <= 0 or snapshots[-1]['copied_bytes'] <= 0):
        raise ValueError('metadata pinned accounting/actual use not established')
    return dict(policy='cpu_live_mapping_v1', installed=True, triggered=True,
        pinned_staging_bytes=expected_bytes, actual_pinned_increment=after-before,
        tensor_budget_bytes=receipt['tensor_budget_bytes'], snapshots=snapshots,
        global_host_budget=raw['metadata_global_host_budget'],
        timing_samples_relabelled=False, numerical_correctness_qualified=False,
        causal_speedup=None)


def audit_control_interleave_snapshots(observations, *, expected_pid, expected_clock):
    """Validate cumulative native counters without inventing latency attribution.

    Observation occurs *inside* the current utility, before that utility's own
    handled/yield counters advance. Differences therefore describe the recorded
    snapshot interval, not an exact number of requests or model steps.
    """
    keys = ('input_turns', 'handled_controls', 'yields_to_native_loop',
            'yields_with_queued_input')
    if (type(expected_pid) is not int or expected_pid <= 0
            or not isinstance(expected_clock, str) or not expected_clock
            or len(observations) < 2):
        raise ValueError('one native owner and at least two observations required')
    rows = []
    for label, observation in observations:
        state = observation.get('control_interleave', {})
        stamp = observation.get('captured_at')
        if (observation.get('kind') != 'ieee_native_scheduler_observation_v1'
                or observation.get('scheduler_pid') != expected_pid
                or type(observation.get('scheduler_pid')) is not int
                or observation.get('clock_id') != expected_clock
                or type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp <= 0
                or state.get('kind') != 'ieee_native_control_interleave_v1'
                or state.get('policy') != 'atomic_control_yield_v1'
                or any(type(state.get(k)) is not int or state[k] < 0 for k in keys)):
            raise ValueError('complete, same-owner native interleave snapshot required')
        values = {k:state[k] for k in keys}
        if (values['yields_with_queued_input'] > values['yields_to_native_loop']
                or values['yields_to_native_loop'] > values['handled_controls']
                or values['yields_to_native_loop'] > values['input_turns']):
            raise ValueError('interleave counter subset invariant violated')
        row = dict(label=label, captured_at=stamp, **values)
        if rows and (stamp <= rows[-1]['captured_at']
                     or any(row[k] < rows[-1][k] for k in keys)):
            raise ValueError('snapshot order or cumulative counters regressed')
        rows.append(row)
    delta = {k:rows[-1][k]-rows[0][k] for k in keys}
    return dict(kind='native_control_interleave_snapshot_audit_v1',
        policy='atomic_control_yield_v1', scheduler_pid=expected_pid, clock_id=expected_clock,
        observations=rows, delta=delta,
        triggered=delta['yields_to_native_loop'] > 0,
        queued_input_triggered=delta['yields_with_queued_input'] > 0,
        counter_scope='between_recorded_native_scheduler_snapshots',
        model_step_count=None, gpu_idle_seconds=None, causal_speedup=None)


def audit_control_source_run(raw):
    """All waves including warmups and zero-trigger intervals; no winner filter."""
    if raw['model_config'].get('ieee_control_interleave') != 'atomic_control_yield_v1':
        raise ValueError('source run does not declare the qualified control policy')
    workers = raw['workers_before']['workers']
    if len(workers) != 1:
        raise ValueError('control calibration requires its single native worker')
    waves, spec = raw['profile_waves'], raw['source_profile_spec']['waves']
    if len(waves) != len(spec) or not waves:
        raise ValueError('complete planned wave observations required')
    observations = [('scheduler_before', raw['scheduler_before'])]
    for index, wave in enumerate(waves):
        if wave['wave'] != index or wave.get('complete') is not True:
            raise ValueError('complete, ordered source waves required')
        observations.extend((f'wave{index}:{side}', wave['capacity_'+side])
                            for side in ('before', 'after'))
    observations.append(('scheduler_after', raw['scheduler_after']))
    result = audit_control_interleave_snapshots(observations,
        expected_pid=workers[0]['pid'], expected_clock=workers[0]['clock_id'])
    keys = tuple(result['delta'])
    result['waves'] = [dict(wave=index, role=spec[index]['role'],
        delta={k:wave['capacity_after']['control_interleave'][k]
                 -wave['capacity_before']['control_interleave'][k] for k in keys})
        for index,wave in enumerate(waves)]
    result['zero_yield_waves'] = [r['wave'] for r in result['waves']
                                if r['delta']['yields_to_native_loop'] == 0]
    return result


def audit_file_demand_source_run(raw):
    """Bind every measured file acquisition to its held owner and native ACK.

    This checks the changed path, including warmups, not numerical model output
    or speedup. Native HOST/GPU acquisitions must retain their original path.
    """
    from faaslora.memory.residency_manager import (
        IEEE_FILE_DEMAND_TRANSACTION, validate_file_demand_witness)
    if raw['model_config'].get('ieee_file_demand_transaction') != IEEE_FILE_DEMAND_TRANSACTION:
        raise ValueError('source run lacks its file-demand policy')
    expected = {f'source-profile/w{i}/l{lane}':wave['source']
                for i,wave in enumerate(raw['source_profile_spec']['waves'])
                for lane,_ in enumerate(wave['requests'])}
    counts, advances, seen = Counter(), [], set()
    for row in raw['requests']:
        rid = row['request_id']
        if rid in seen or rid not in expected or row.get('pass') is not True:
            raise ValueError('complete unique file-policy measurements required')
        seen.add(rid)
        source = expected[rid]
        if row.get('requested_source') != source:
            raise ValueError('file-policy case source differs')
        e = row['source_evidence']
        native, intent = e['receipt'], e['intent']
        if (e.get('state') != 'released' or native.get('acquired') is not True
                or e['release_receipt'].get('released') is not True
                or native['lease_id'] != intent['lease_id']
                or native['owner_id'] != intent['expected_owner_id']
                or native['lora_name'] != row['adapter_id']):
            raise ValueError('file-policy native ownership incomplete')
        if source in ('gpu','native_host'):
            if any(key in obj for obj in (native,intent)
                   for key in ('file_source_witness','file_demand_transaction')):
                raise ValueError('file policy leaked onto native cached-source path')
            counts[source]+=1
            continue
        if source not in ('remote','nvme','file_host'):
            raise ValueError('unknown file-policy source')
        witness = native.get('file_source_witness')
        validate_file_demand_witness(witness,lora_name=row['adapter_id'],
            lora_path=intent['lora_path'],content_sha256=intent.get('expected_content_sha256'))
        held = e['local_source_reference']
        if (witness != intent.get('file_source_witness')
                or any(witness[k] != held.get(k) for k in
                       ('owner_id','lease_id','adapter_id','path','tier','device','inode'))
                or held.get('content_verified') is not True
                or witness['tier'] != ('host' if source=='file_host' else 'nvme')
                or native.get('expected_content_sha256') != witness['content_sha256']):
            raise ValueError('file-policy held source/content binding differs')
        transaction = native.get('file_demand_transaction',{})
        observed,validated = transaction.get('observed_epoch'),transaction.get('validated_epoch')
        if (transaction.get('kind') != IEEE_FILE_DEMAND_TRANSACTION
                or any(type(x) is not int or x<0 for x in (observed,validated,native.get('epoch')))
                or not observed <= validated <= native['epoch']
                or observed != intent['expected_epoch']
                or transaction.get('owner_epoch_advance') != validated-observed
                or type(transaction.get('owner_epoch_advance')) is not int
                or transaction.get('live_native_constraints_checked') is not True
                or transaction.get('controller_capacity_snapshot_reused') is not False):
            raise ValueError('file-policy live validation receipt differs')
        counts[source]+=1
        advances.append(validated-observed)
    if seen != set(expected) or not seen:
        raise ValueError('file-policy source coverage incomplete')
    return dict(kind='protected_file_demand_source_audit_v1',policy=IEEE_FILE_DEMAND_TRANSACTION,
        request_count=len(seen),source_counts=dict(counts),file_transactions=len(advances),
        advanced_owner_epoch_transactions=sum(x>0 for x in advances),
        max_owner_epoch_advance=max(advances,default=0),
        old_capacity_snapshot_reused=False,n_correct=None,causal_speedup=None)


def historical_service_samples(requests, *, source_sha, identities, previous_bins):
    """Reuse completed Full native intervals, not estimates or pooled fallbacks.

    This is development timing evidence only. Native identity/count consistency
    does not establish numerical adapter qualification. The caller verifies the
    entire run, physical release, model/configuration and content SHA chain.
    """
    if not re.fullmatch('[0-9a-f]{64}', source_sha) or not requests:
        raise ValueError('nonempty SHA-bound history required')
    samples, seen = [], set()
    for r in requests:
        rid, aid = r['request_id'], r['adapter_id']
        if not rid or rid in seen or aid not in identities:
            raise ValueError('unique requests and known artifact identity required')
        seen.add(rid)
        e, t = r['gpu_reference_evidence'], r['native_token_timing']
        admission, native, intervals = e['source_admission'], e['receipt'], e['service_intervals']
        source, identity = admission['source'], identities[aid]
        n, target, prompt = r['completion_tokens'], r['requested_completion_tokens'], r['canonical_prompt_tokens']
        native_prompt = t['actual_prompt_tokens']
        if (r['success'] is not True or r['generation_contract'] != 'fixed_length_greedy_v1'
                or r['completion_token_source'] != 'vllm_token_ids' or r['output_contract_match'] is not True
                or type(n) is not int or n < 1 or n != target or n != t['native_output_tokens']
                or t['native_terminal_observed'] is not True or not isinstance(prompt, (int, float))
                or not math.isfinite(prompt) or int(prompt) != prompt or prompt < 0
                or not isinstance(native_prompt, (int, float)) or not math.isfinite(native_prompt)
                or int(native_prompt) != native_prompt or native_prompt < 1
                or e['state'] != 'released' or e['release_receipt'].get('released') is not True
                or e['pending_kv_admission']['state'] != 'closed'
                or e['pending_kv_admission']['close_receipt'].get('closed') is not True
                or native.get('acquired') is not True or native.get('lora_name') != aid):
            raise ValueError('completed native generation and released ownership required')
        if (admission['clock_id'] != t['native_clock_id'] or native['clock_id'] != t['native_clock_id']
                or native['owner_id'] != t['gpu_reference_owner_id']
                or native['lease_id'] != t['gpu_reference_lease_id']):
            raise ValueError('mixed native clock or reference identity')
        content = source.get('expected_content_sha256') if source['native'] else source.get('content_sha256')
        if content != identity['content_sha256']:
            raise ValueError('source content identity differs from immutable artifact')
        features = dict(tier=source['tier'], representation=source['representation'],
            footprint_bytes=source['footprint_bytes'], adapter_rank=identity['rank'],
            prompt_tokens=int(prompt), declared_output_tokens=target,
            admitted_after_accept=admission['admitted_after_accept'])
        old_key = previous_bins.classify(**features)
        if asdict(old_key) != admission['service_class'] or intervals['service_class'] != admission['service_class']:
            raise ValueError('reconstructed features disagree with admission-time class: '
                + json.dumps(dict(request_id=rid, reconstructed=asdict(old_key),
                    admission=admission['service_class'], interval=intervals['service_class'],
                    features=features), sort_keys=True))
        a, b, f, last = [intervals[k] for k in ('admitted_monotonic_s', 'acquired_monotonic_s',
                                              'first_monotonic_s', 'last_monotonic_s')]
        if (any(type(x) not in (int, float) or not math.isfinite(x) for x in (a, b, f, last))
                or not 0 < a <= b <= f <= last or a != admission['admitted_monotonic_s']
                or a != t['controller_admitted_monotonic_s']
                or f != t['native_first_token_monotonic_s'] or last != t['native_last_token_monotonic_s']
                or (source['tier'] == 'gpu' and a != b)
                or (source['tier'] != 'gpu' and b != t['gpu_reference_acquired_monotonic_s'])):
            raise ValueError('completed native D/T/O boundaries required')
        events = e['service_events']
        if len(events) != 2:
            raise ValueError('complete first/last native event pair required')
        observation = ServiceIntervalObservation.for_profiling(old_key, a)
        if observation.acquired_at is None:
            observation.acquire(b)
        observer = NativeServiceIntervalObserver(observation, clock_id=t['native_clock_id'],
            adapter_id=aid, gpu_reference=native)
        for event in events:
            observer(event)
        # The frontend may deliver several already-timed native tokens together;
        # first-event count need not be one. Reuse the production observer and
        # native first/last boundaries, never chunk receipt time or text counts.
        if observation.first_at != f or observation.last_at != last or events[-1]['token_count'] != n:
            raise ValueError('native event does not bind this completed interval')
        samples.append(dict(correct=True, source_run_sha256=source_sha, request_id=rid,
            attempt_id=native['lease_id'], source_measurement='native_admission_acquisition_token_events_v1',
            admission_clock_id=admission['clock_id'], native_clock_id=t['native_clock_id'],
            class_features=features, native_output_tokens=n, protected_at_admission=True,
            native_prompt_tokens=int(native_prompt),
            prompt_feature_source='canonical_prompt_tokens_at_admission',
            admitted_monotonic_s=a, acquired_monotonic_s=b,
            first_token_monotonic_s=f, last_token_monotonic_s=last))
    return samples


def historical_output_bins(samples, *, output_cap):
    """One predetermined quartile rule; never use latency/rankings to pick edges."""
    values = sorted(s['class_features']['declared_output_tokens'] for s in samples)
    if (not values or type(output_cap) is not int or output_cap < 1
            or any(type(x) is not int or not 1 <= x <= output_cap for x in values)):
        raise ValueError('completed declared-output history within the fixed cap required')
    return sorted({values[math.ceil(len(values)*q)-1] for q in (.25, .5, .75)} | {output_cap})


def service_rebin_coverage(base, historical, *, output_cap, additional=()):
    """Report the full reachable domain; missing classes remain missing."""
    bins = copy.deepcopy(base['bins'])
    bins['declared_output_tokens'] = historical_output_bins(historical, output_cap=output_cap)
    old = ServiceClassBins(**{k:tuple(v) for k,v in base['bins'].items()})
    new = ServiceClassBins(**{k:tuple(v) for k,v in bins.items()})
    old_keys = {old.classify(**s['class_features']) for s in base['samples']}
    output_bins = {bisect_left(new.declared_output_tokens, x) for x in range(1, output_cap+1)}
    required = {replace(key, output_limit_bin=o) for key in old_keys for o in output_bins}
    samples = copy.deepcopy(base['samples']) + copy.deepcopy(historical) + copy.deepcopy(list(additional))
    if len({(s['source_run_sha256'], s['request_id'], s['attempt_id']) for s in samples}) != len(samples):
        raise ValueError('duplicate profile observations')
    counts = Counter(new.classify(**s['class_features']) for s in samples)
    if not set(counts) <= required:
        raise ValueError('historical sample escapes the qualified non-output class domain')
    encode = lambda k: json.dumps(asdict(k), sort_keys=True)
    return dict(bins=bins, samples=samples, rule='historical_declared_output_type1_quartiles_cap_v1'), dict(
        required_classes=len(required), observed_classes=len(counts),
        missing_classes=[asdict(k) for k in sorted(required-set(counts), key=encode)],
        counts=[dict(service_class=asdict(k), n=counts[k]) for k in sorted(counts, key=encode)],
        class_domain_covered=required <= set(counts), output_edges=bins['declared_output_tokens'],
        historical_samples=len(historical), retained_source_samples=len(base['samples']),
        additional_measured_samples=len(additional),
        no_default_fills=True, production_profile_frozen=False, numerical_qualification=False)


def completed_gap_samples(raw, *, source_sha, identities):
    """Normalize source-collector observations; never call a partial profile full.

    The existing domain validator checks every planned row. Final completeness
    is checked after combining with retained, SHA-bound history/source samples.
    """
    coverage = p.source_profile_class_coverage(raw, identities)
    if (raw['runtime_boundary'] != 'dedicated_subprocess'
            or raw['physical_allocation']['released'] is not True):
        raise ValueError('gap collector runtime/release evidence required')
    bins = ServiceClassBins(**{k:tuple(v) for k,v in raw['source_profile_spec']['bins'].items()})
    roles = {f'source-profile/w{i}/l{lane}':w['role'] for i,w in enumerate(raw['source_profile_spec']['waves'])
             for lane in range(len(w['requests']))}
    normalized = []
    for q in raw['requests']:
        e,t = copy.deepcopy(q['source_evidence']),copy.deepcopy(q['timing'])
        a,b,f,z = [q[k] for k in ('admitted_monotonic_s','acquired_monotonic_s',
                                'first_token_monotonic_s','last_token_monotonic_s')]
        e['service_intervals'] = dict(service_class=q['service_class'],
            admitted_monotonic_s=a,acquired_monotonic_s=b,first_monotonic_s=f,last_monotonic_s=z)
        e['service_events'] = q['native_events']
        # These are the collector's actually measured admission/acquisition
        # boundaries, not generated or estimated missing timestamps.
        t.update(controller_admitted_monotonic_s=a,gpu_reference_acquired_monotonic_s=b,
            gpu_reference_owner_id=e['receipt']['owner_id'],gpu_reference_lease_id=e['receipt']['lease_id'])
        normalized.append(dict(request_id=q['request_id'],adapter_id=q['adapter_id'],success=q['pass'],
            generation_contract=raw['model_config']['generation_contract'],
            completion_token_source='vllm_token_ids',output_contract_match=q['actual_tokens']==q['target_tokens'],
            completion_tokens=q['actual_tokens'],requested_completion_tokens=q['target_tokens'],
            canonical_prompt_tokens=q['class_features']['prompt_tokens'],gpu_reference_evidence=e,
            native_token_timing=t))
    checked = historical_service_samples(normalized,source_sha=source_sha,identities=identities,previous_bins=bins)
    return [s for s in checked if roles[s['request_id']]=='representative_measurement'],coverage


def export_completed_length_candidate():
    """D321 complete initialization; no Full performance/qualification claim."""
    folder = ROOT/'paper_results/ieee_tc/p2_backend/d321_output_classes'
    out,history,gap = folder/'initialization1',folder/'history1',folder/'gaps1'
    if out.exists(): raise FileExistsError(out)
    for name,sha in read(history/'manifest.json')['outputs'].items():
        assert p.digest(history/name)==sha
    prepared = read(gap/'prepare.json')
    spec_path = gap/'source_spec.json'
    assert p.digest(spec_path)==prepared['spec_sha256']
    spec = read(spec_path)
    rawdir = ROOT/'results/ieee_tc/p2_backend_qualification/d321_20261007/gaps1'
    source,launch_path = rawdir/'source1.json',rawdir/'source1_launch.json'
    raw,launch = read(source),read(launch_path)
    assert raw['pass'] and raw['stage']=='complete' and raw['shutdown_called'] and raw['profile_workspaces_removed']
    assert raw['source_profile_spec_sha256']==p.digest(spec_path) and raw['source_profile_spec']==spec
    assert launch['pass'] and launch['service_returncode']==launch['watchdog_returncode']==0
    assert launch['service_path_removed'] and launch['native_gpu_context_release_confirmed']
    assert launch['preflight']['policy']==p.POLICY
    definition = read(history/'candidate_definition.json')
    cfg = definition['model_config']
    assert FrozenServiceProfiles.model_identity(raw['model_config'])==FrozenServiceProfiles.model_identity(cfg)
    source_cfg=ROOT/'configs/ieee_tc/20261007_d298_7b_capacity_source.yaml'
    assert p.digest(source_cfg)==raw['config_sha256']=='6b96df57e5f05a9b4a0b597ef988b464d37af6b61e4dac33b3e4444778d1487f'
    index_path=ROOT/spec['content_index']['path']
    assert p.digest(index_path)==spec['content_index']['sha256']
    index=read(index_path)
    client=HttpArtifactStoreClient(endpoint='http://127.0.0.1:1')
    client.configure_content_manifest(index)
    pool=(ROOT/yaml.safe_load(source_cfg.read_text())['model_profiles']['llama2_7b_main_v2_publicmix']['storage']['remote_dir']).resolve(strict=True)
    identities={}
    for row in index['artifacts']:
        path=(pool/row['id']/'adapter_config.json').resolve(strict=True)
        assert path.parent.parent==pool
        assert p.digest(path)==next(f['sha256'] for f in row['files'] if f['path']=='adapter_config.json')
        identities[row['id']]=client.routing_identity(row['id'],path.read_bytes())
    journal=rawdir/'remote_transfers.jsonl'
    remote_rows=[json.loads(line) for line in journal.read_text().splitlines()]
    remote={r['transfer_id']:r for r in remote_rows}
    transfers=raw['remote_transfers']
    assert len(remote)==len(remote_rows)==len(transfers)
    assert {r['http_transfer_id'] for r in transfers}==set(remote)
    clock=read(rawdir/'remote_health2.json')['clock_id']
    for r in transfers:
        s=remote[r['http_transfer_id']]
        assert s['clock_id']==clock and r['state']=='published' and r['content_verified'] and r['published_archive_verified']
        assert r['artifact_id']==s['artifact_id'] and r['remote_archive_sha256']==s['archive_sha256']
        assert r['transferred_bytes']==s['archive_bytes']==s['bytes_read']==s['bytes_written']
        assert not r['remote_pack_performed'] and not s['pack_performed'] and not s['temporary_created'] and s['outcome']=='sent'
    assert all(not Path(path).exists() for path in raw['owned_profile_workspaces'].values())
    samples,standalone=completed_gap_samples(raw,source_sha=p.digest(source),identities=identities)
    assert len(samples)==81 and len(raw['requests'])==86
    base_path=Path(definition['base_profile']['path'])
    assert p.digest(base_path)==definition['base_profile']['sha256']
    base=read(base_path)
    retained=read(history/'historical_samples.json')['samples']
    draft,coverage=service_rebin_coverage(base,retained,output_cap=256,additional=samples)
    assert draft['bins']==definition['bins'] and coverage['required_classes']==160
    assert coverage['class_domain_covered'],coverage['missing_classes']
    context=copy.deepcopy(definition['context'])
    context['input_contract_sha256']=object_hash(dict(previous_context=definition['context'],
        history=p.digest(history/'manifest.json'),gap_spec=p.digest(spec_path),gap_measurement=p.digest(source),
        bins=draft['bins'],rule=draft['rule']))
    service=copy.deepcopy(base)
    service.update(context=context,bins=draft['bins'],samples=draft['samples'],
        development_only=True,full_qualified=False,numerical_adapter_correctness_qualified=False,
        calibration_rule=draft['rule'],history_initialization=True)
    out.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        with (out/name).open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False)
    save('service.json',service)
    profile=FrozenServiceProfiles.load(out/'service.json',expected_sha256=p.digest(out/'service.json'),
        model_config=cfg,expected_context=context,beta=definition['ewma_beta'])
    assert len(profile.profiles)==160
    save('coverage.json',dict(combined=coverage,standalone_gap_measurement=standalone,
        historical_definition_unchanged=True,new_preparation_profile=False))
    save('binding.json',dict(ieee_service_profile=dict(path=str(out/'service.json'),
        sha256=p.digest(out/'service.json'),context=context,ewma_beta=definition['ewma_beta'])))
    paths=[source,launch_path,spec_path,gap/'prepare.json',history/'manifest.json',
        history/'historical_samples.json',history/'candidate_definition.json',base_path,index_path,source_cfg,
        journal,rawdir/'remote_health2.json',rawdir/'remote_start.log',rawdir/'remote_stop.log',
        Path(__file__),folder/'provenance/export_at_history1.py',ROOT/'scripts/ieee_tc_preflight.py',
        ROOT/'faaslora/experiment/instance_pool.py',Path('/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md'),
        ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md']
    save('manifest.json',dict(kind='d321_complete_development_initialization_v1',
        source_refs=[ref(path) for path in paths],outputs={path.name:p.digest(path) for path in sorted(out.iterdir())},
        profile_samples=len(service['samples']),observed_classes=len(profile.profiles),
        source_execution_commit=(rawdir/'source_commit.txt').read_text().strip(),
        native_calls=86,measurements=81,transfers=len(transfers),wire_bytes=sum(r['transferred_bytes'] for r in transfers),
        no_remote_pack=True,new_gpu_runs=1,full_candidate_launched=False,n_correct=None,no_figures=True))
    print(json.dumps(dict(profile_classes=len(profile.profiles),profile_samples=len(service['samples']),
        output_edges=draft['bins']['declared_output_tokens'],binding=ref(out/'binding.json'))),flush=True)


def export_history_length_coverage():
    """D321: retain reusable historical samples and exact missing-class inventory."""
    from scripts.ieee_tc.plot_d290_diagnostic import audit_run, binding_checks
    from scripts.run_all_experiments import (
        _prepare_scenario_runtime_model_config, _prepare_dedicated_subprocess_model_cfg)
    out = ROOT/'paper_results/ieee_tc/p2_backend/d321_output_classes/history1'
    if out.exists(): raise FileExistsError(out)
    plan = Path('/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md')
    protocol = ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md'
    assert p.digest(plan) == '0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    assert p.digest(protocol) == '5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    prior = ROOT/'paper_results/ieee_tc/p2_backend/d315_full_w0/validated1'
    seal = read(prior/'manifest.json')
    assert p.digest(prior/'audit.json') == seal['outputs']['audit.json']
    audit = read(prior/'audit.json')
    source = Path(audit['evidence']['source'])
    assert p.digest(source) == audit['evidence']['source_sha256'] == '6e5007630c9780c417e65109ef3005daedba3dafa87a843ff75c4885f91c362f'
    cfg_path = ROOT/'configs/ieee_tc/20261007_d315_7b_full_w0.yaml'
    refs = {r['path']:r['sha256'] for r in seal['source_refs']}
    assert p.digest(cfg_path) == refs[str(cfg_path)]
    cfg = yaml.safe_load(cfg_path.read_text())
    binding = cfg['resource_coordination']['ieee_service_profile']
    base_path = ROOT/binding['path']
    configured = _prepare_scenario_runtime_model_config(cfg['model'], cfg['workload'], cfg['resource_coordination'])
    child, _ = _prepare_dedicated_subprocess_model_cfg(configured, device_id=0, runtime_gpu_ids=[0])
    loaded = FrozenServiceProfiles.load(base_path, expected_sha256=binding['sha256'],
        model_config=child, expected_context=binding['context'], beta=binding['ewma_beta'])
    index_path = Path(cfg['model']['artifact_content_manifest_path'])
    index = read(index_path)
    assert p.digest(index_path) == refs[str(index_path)]
    client = HttpArtifactStoreClient(endpoint='http://127.0.0.1:1')
    client.configure_content_manifest(index)
    pool = Path(cfg['storage']['remote_dir']).resolve(strict=True)
    identities = {}
    for entry in index['artifacts']:
        config_file = (pool/entry['id']/'adapter_config.json').resolve(strict=True)
        expected = next(f['sha256'] for f in entry['files'] if f['path']=='adapter_config.json')
        assert config_file.parent.parent == pool and p.digest(config_file) == expected
        identities[entry['id']] = client.routing_identity(entry['id'], config_file.read_bytes())
    _, evidence, requests = audit_run(source)
    assert len(requests) == 4000 and len(identities) == 500
    for row in requests:
        assert all(binding_checks(row, identities[row['adapter_id']]['content_sha256']).values())
    samples = historical_service_samples(requests, source_sha=evidence['source_sha256'],
        identities=identities, previous_bins=loaded.bins)
    draft, coverage = service_rebin_coverage(read(base_path), samples, output_cap=child['max_output_tokens_cap'])
    paths = [source, cfg_path, base_path, index_path, prior/'manifest.json', prior/'audit.json',
        Path(evidence['physical_source']), plan, protocol, Path(__file__),
        ROOT/'scripts/ieee_tc/plot_d290_diagnostic.py', ROOT/'faaslora/experiment/instance_pool.py']
    out.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        with (out/name).open('x') as stream: json.dump(value, stream, indent=2, allow_nan=False)
    # Not a runtime profile identity: incomplete coverage may not be launched.
    save('historical_samples.json', dict(kind='historical_full_service_samples_v1',
        source=evidence, samples=samples, numerical_qualification=False))
    save('coverage.json', coverage)
    save('candidate_definition.json', dict(kind='declared_output_rebin_candidate_v1',
        rule=draft['rule'], bins=draft['bins'], base_profile=ref(base_path),
        historical_samples=ref(out/'historical_samples.json'),
        launchable=False, stage='coverage_only', model_config=child,
        context=binding['context'], ewma_beta=binding['ewma_beta'],
        numerical_qualification=False, required_next='measure_missing_classes_or_validate_complete_export'))
    save('manifest.json', dict(source_refs=[ref(path) for path in paths],
        outputs={path.name:p.digest(path) for path in sorted(out.iterdir())},
        no_figures=True, new_gpu_runs=0, n_correct=None))
    print(json.dumps(coverage, indent=2), flush=True)


def audited_source_run(raw_dir, curated_dir, config_path, *, health_model=None,
                       require_measurement_seal=False):
    """Verify each real run independently before combining measured samples."""
    source, launch_path = raw_dir/'source1.json', raw_dir/'source1_launch.json'
    raw, launch = read(source), read(launch_path)
    assert raw['pass'] and raw['stage'] == 'complete'
    assert raw['shutdown_called'] and raw['profile_workspaces_removed']
    assert launch['pass'] and launch['service_returncode'] == launch['watchdog_returncode'] == 0
    assert launch['service_path_removed'] and launch['native_gpu_context_release_confirmed']
    assert launch['preflight']['policy'] == p.POLICY
    assert raw['service']['affinity'] == p.POLICY['service_cpus']
    spec_path = curated_dir/'source_spec.json'
    spec = read(spec_path)
    assert p.digest(spec_path) == raw['source_profile_spec_sha256']
    assert raw['source_profile_spec'] == spec
    contract_path = ROOT/spec['full_development_contract']['path']
    assert p.digest(contract_path) == spec['full_development_contract']['sha256']
    contract = read(contract_path)
    runtime = ROOT/'results/ieee_tc/p2_backend_qualification/runtime_20260926/cuda_import_attempt3.json'
    assert p.digest(runtime) == raw['runtime_receipt_sha256'] == contract['runtime_receipt_sha256']
    assert p.digest(config_path) == raw['config_sha256'] == read(curated_dir/'launch_config_binding.json')['source_config']['sha256']
    assert p.digest(Path(raw['trace']['source_path'])) == spec['trace_sha256'] == raw['trace']['source_sha256']
    journal = raw_dir/'remote_transfers.jsonl'
    remote_rows = [json.loads(line) for line in journal.read_text().splitlines()]
    remote = {r['transfer_id']:r for r in remote_rows}
    transfers = raw['remote_transfers']
    assert len(remote) == len(remote_rows) == len(transfers)
    assert {r['http_transfer_id'] for r in transfers} == set(remote)
    health = read(raw_dir/'remote_health.json')
    remote_clock = health[health_model or 'health']['clock_id']
    for r in transfers:
        s = remote[r['http_transfer_id']]
        assert s['clock_id'] == remote_clock
        assert r['state'] == 'published' and r['content_verified'] and r['published_archive_verified']
        assert r['artifact_id'] == s['artifact_id'] and r['remote_archive_sha256'] == s['archive_sha256']
        assert r['transferred_bytes'] == s['archive_bytes'] == s['bytes_read'] == s['bytes_written']
        assert not r['remote_pack_performed'] and not s['pack_performed'] and not s['temporary_created']
        assert s['outcome'] == 'sent'
    lease = raw['physical_allocation']
    acquire = [e['at'] for e in lease['events'] if e['event'] == 'acquire']
    release = [e['at'] for e in lease['events'] if e['event'] == 'release']
    assert len(acquire) == len(release) == len(lease['gpu_uuids']) == 1
    assert lease['released'] and release[0] > acquire[0]
    assert all(not Path(path).exists() for path in raw['owned_profile_workspaces'].values())
    watchdog = raw_dir/'source1_launch.launch/watchdog.jsonl'
    with watchdog.open() as stream:
        samples = [r for line in stream if (r := json.loads(line))['event'] == 'resource_sample']
    assert samples and not any(r['decision']['abort_reasons'] for r in samples)
    paths = [source, launch_path, spec_path, contract_path, journal, watchdog,
        raw_dir/'remote_monitor.log', raw_dir/'remote_stop.log', raw_dir/'remote_health.json']
    if require_measurement_seal:
        seal_path = curated_dir/'measurement_outputs.sha256'
        for line in seal_path.read_text().splitlines():
            sha, name = line.split(maxsplit=1)
            path = curated_dir/name
            assert path.parent == curated_dir and p.digest(path) == sha
        checkpoint_path = curated_dir/'measurement_checkpoint.json'
        checkpoint = read(checkpoint_path)
        assert checkpoint['passed'] and checkpoint['measurement_complete']
        assert Path(checkpoint['source_run']['path']).resolve() == source.resolve()
        assert checkpoint['source_run']['sha256'] == p.digest(source)
        for entry in checkpoint['source_refs']:
            assert p.digest(Path(entry['path'])) == entry['sha256']
        commit = (raw_dir/'source_commit.txt').read_text().strip()
        assert re.fullmatch('[0-9a-f]{40}', commit)
        assert commit == checkpoint['source_execution_commit']
        paths.extend([seal_path, checkpoint_path, raw_dir/'source_commit.txt'])
    measured = [w for w in spec['waves'] if w['role'] == 'representative_measurement']
    roles = {f'source-profile/w{i}/l{lane}':w['role'] for i,w in enumerate(spec['waves'])
             for lane in range(len(w['requests']))}
    summary = dict(requests=len(raw['requests']),
        representative_requests=sum(len(w['requests']) for w in measured),
        warmup_requests=sum(len(w['requests']) for w in spec['waves'] if w['role'] == 'kernel_warmup_retained'),
        preparation_requests=sum(len(w['requests']) for w in measured if w['source'] != 'gpu'),
        measured_source_counts=dict(Counter(q['requested_source'] for q in raw['requests']
            if roles[q['request_id']] == 'representative_measurement')),
        remote_uuid_pairs=len(transfers),wire_bytes=sum(r['transferred_bytes'] for r in transfers),
        remote_pack_performed=False,physical_gpu_seconds=release[0]-acquire[0],
        resources=dict(samples=len(samples),peak_bytes=max(r['service']['memory.peak'] for r in samples),
            min_host_available_bytes=min(r['host']['available_bytes'] for r in samples),
            max_swap_bytes=max(r['service']['memory.swap.current'] for r in samples),
            memory_events={k:max(r['service']['memory.events'][k] for r in samples)
                for k in ('high','max','oom','oom_kill')}))
    return dict(payload=raw, source_run_sha256=p.digest(source), spec=spec,
        contract=contract, launch=launch, roles=roles, summary=summary,
        source_run=ref(source), raw_evidence=[ref(path) for path in paths])


def main(campaign='d298'):
    global RAW, CURATED
    if campaign == 'd298':
        config_path = ROOT/'configs/ieee_tc/20261007_d298_7b_capacity_source.yaml'
        source_commit = '783fa42e6b3ba1cd0e034d8e0cddcd3d72c9b42e'
    elif campaign in ('d307', 'd307-attempt2'):
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d307_20261007'
        if campaign == 'd307-attempt2':
            RAW = RAW/'attempt2'
        CURATED = ROOT/'paper_results/ieee_tc/p2_backend/d307_prefill'
        config_path = ROOT/'configs/ieee_tc/20261007_d307_7b_prefill_source.yaml'
        source_commit = (RAW/'source_commit.txt').read_text().strip()
    elif campaign in ('d344', 'd344-merged'):
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d344_20261008'
        CURATED = ROOT/'paper_results/ieee_tc/p2_backend/d344_capacity_profiles'
        config_path = ROOT/'configs/ieee_tc/20261008_d343_7b_capacity_probe14.yaml'
        source_commit = (RAW/'source_commit.txt').read_text().strip()
    elif campaign in ('d356', 'd356-merged'):
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d356_20261008'
        CURATED = ROOT/'paper_results/ieee_tc/p2_backend/d356_control_profiles'
        config_path = ROOT/'configs/ieee_tc/20261008_d356_7b_control_source.yaml'
        source_commit = (RAW/'source_commit.txt').read_text().strip()
    elif campaign == 'd524':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d524_20261014'
        CURATED = ROOT/'paper_results/ieee_tc/p2_backend/d524_metadata_profiles'
        config_path = ROOT/'configs/ieee_tc/20261014_d524_7b_metadata_source.yaml'
        source_commit = (RAW/'source_commit.txt').read_text().strip()
    elif campaign == 'd362':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d362_20261009'
        CURATED = ROOT/'paper_results/ieee_tc/p2_backend/d362_file_demand_profiles'
        config_path = ROOT/'configs/ieee_tc/20261009_d362_7b_file_demand_source.yaml'
        source_commit = (RAW/'source_commit.txt').read_text().strip()
    else:
        raise ValueError(campaign)
    assert re.fullmatch('[0-9a-f]{40}', source_commit), 'missing exact launch commit'
    out = CURATED / 'initialization'
    assert not out.exists(), 'never overwrite a prior export'
    directories = [(RAW,CURATED)]
    if campaign in ('d344-merged', 'd356-merged'):
        directories.append((RAW/'gaps1', CURATED/'gaps1'))
    runs = [audited_source_run(r,c,config_path,
        health_model='7b' if campaign.startswith(('d344','d356','d362','d524')) else None,
        require_measurement_seal=campaign.startswith(('d344','d356','d362','d524'))) for r,c in directories]
    first = runs[0]
    raw, spec, contract, launch = [first[k] for k in ('payload','spec','contract','launch')]
    metadata_audit = audit_metadata_source_run(raw) if campaign == 'd524' else None
    control_audit = None
    if campaign.startswith(('d356','d362','d524')):
        audits = []
        for run,(raw_dir,_) in zip(runs,directories):
            audit = audit_control_source_run(run['payload'])
            assert audit['triggered'], 'enabled mechanism never triggered in calibration'
            assert read(raw_dir/'remote_stop.json')['all_owned_units_inactive']
            assert read(raw_dir/'local_cleanup.json')['native_gpu_release_confirmed']
            audits.append(dict(source_run=run['source_run'],audit=audit))
        control_audit = (audits[0]['audit'] if len(audits)==1 else
            dict(per_source=audits,clock_domains_joined=False,
                 causal_speedup=None,gpu_idle_time=None,native_model_steps=None))
    assert all(r['contract'] == contract for r in runs)
    profile = yaml.safe_load(config_path.read_text())['model_profiles']['llama2_7b_main_v2_publicmix']
    pool = (ROOT/profile['storage']['remote_dir']).resolve(strict=True)
    index_path = ROOT/spec['content_index']['path']
    assert p.digest(index_path) == spec['content_index']['sha256']
    index = read(index_path)
    client = HttpArtifactStoreClient(endpoint='http://127.0.0.1:1')
    client.configure_content_manifest(index)
    identities = {}
    for row in index['artifacts']:
        path = (pool/row['id']).resolve(strict=True)
        assert path.parent == pool
        identities[row['id']] = client.routing_identity(row['id'], (path/'adapter_config.json').read_bytes())
    assert len(identities) == 500
    input_contract = dict(source_trace_sha256=spec['trace_sha256'],
        content_index_sha256=spec['content_index']['sha256'],
        generation_contract=raw['model_config']['generation_contract'],
        canonical_prompt_renderer=raw['model_config']['canonical_prompt_renderer'],
        prompt_content_cap=raw['model_config']['max_input_len'],
        output_cap=raw['model_config']['max_output_tokens_cap'],
        artifact_mode=raw['artifact_mode'],declared_source_profile_sha256=raw['source_profile_spec_sha256'])
    if len(runs) > 1:
        del input_contract['declared_source_profile_sha256']
        input_contract['declared_source_profile_sha256s'] = [r['payload']['source_profile_spec_sha256'] for r in runs]
        assert all(r['payload']['artifact_mode'] == raw['artifact_mode'] for r in runs)
    envelope = dict(inference_policy=launch['preflight']['policy'],
        artifact_service=dict(memory_high_bytes=2*p.GIB,memory_max_bytes=4*p.GIB,
            memory_swap_max_bytes=0,cpus=list(range(2,20))+list(range(22,40)),
            published_server_sha256='a31fa9212143914f5a7fb21dea126a23eaafedffe199ff9513de9ce03eb62a2c',
            delivery_mode='prepublished_gzip_v1',link_mbps=1000,rate_limit=None))
    context = dict(backend_environment_sha256=raw['runtime_receipt_sha256'],
        resource_envelope_sha256=object_hash(envelope),input_contract_sha256=object_hash(input_contract))
    if len(runs) > 1:
        service, prep, coverage = p.measured_source_profile_collection(runs,
            identities=identities, context=context, model_config=contract['model_config'])
    else:
        service, prep, coverage = p.measured_source_profile_payloads(raw,
            source_run_sha256=first['source_run_sha256'],identities=identities,
            context=context,model_config=contract['model_config'])
    assert len(service['samples']) == sum(r['summary']['representative_requests'] for r in runs)
    assert len(prep['samples']) == sum(r['summary']['preparation_requests'] for r in runs)
    seal = p.verify_seal(ROOT/'paper_results/ieee_tc/safety/20260925_execution_start_protected.json')
    known = '/home/qhq/serverless_llm_baselines/scripts/replay_openai_trace.py'
    assert seal['changed'] == [known] and seal['entry_count'] == 147
    assert p.digest(Path(known)) == '04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22'
    out.mkdir(exist_ok=False)
    def save(name, value):
        with (out/name).open('x') as stream:
            json.dump(value,stream,indent=2,allow_nan=False)
    loaded, bindings = {}, {}
    for name,payload,cls,beta in (
        ('service',service,FrozenServiceProfiles,contract['service_ewma_beta']),
        ('preparation',prep,FrozenPreparationProfiles,contract['preparation_ewma_beta'])):
        save(name+'.json',payload)
        path = out/(name+'.json')
        sha = p.digest(path)
        loaded[name] = cls.load(path,expected_sha256=sha,model_config=contract['model_config'],
            expected_context=context,beta=beta)
        bindings['ieee_'+name+'_profile'] = dict(path=str(path),sha256=sha,context=context,ewma_beta=beta)
    estimates = dict(service=[dict(service_class=asdict(k),**asdict(v),n=loaded['service'].sample_counts[k])
        for k,v in loaded['service'].profiles.items()],preparation=[dict(preparation_class=asdict(k),
        d_ms=v,n=loaded['preparation'].sample_counts[k]) for k,v in loaded['preparation'].profiles.items()])
    outputs = defaultdict(set)
    source_counts = Counter()
    for run in runs:
        source_counts.update(run['summary']['measured_source_counts'])
        for q in run['payload']['requests']:
            if run['roles'][q['request_id']] == 'representative_measurement':
                outputs[q['source_request_id'],q['adapter_id'],q['prompt_sha256'],q['target_tokens']].add(
                    q['timing']['completion_token_ids_sha256'])
    summary = {k:sum(r['summary'][k] for r in runs) for k in ('requests','representative_requests',
        'warmup_requests','preparation_requests','remote_uuid_pairs','wire_bytes','physical_gpu_seconds')}
    summary.update(measured_source_counts=dict(source_counts),remote_pack_performed=False,
        varied_output_groups=sum(len(v)>1 for v in outputs.values()),total_output_groups=len(outputs),
        resources=dict(samples=sum(r['summary']['resources']['samples'] for r in runs),
            peak_bytes=max(r['summary']['resources']['peak_bytes'] for r in runs),
            min_host_available_bytes=min(r['summary']['resources']['min_host_available_bytes'] for r in runs),
            max_swap_bytes=max(r['summary']['resources']['max_swap_bytes'] for r in runs),
            memory_events={k:max(r['summary']['resources']['memory_events'][k] for r in runs)
                for k in ('high','max','oom','oom_kill')}),
        per_source=[dict(source_run=r['source_run'],**r['summary']) for r in runs])
    result = dict(kind='measured_development_profile_export_v1',model='7b',passed=True,
        source_execution_commit=source_commit,source_run=first['source_run'],
        contract=spec['full_development_contract'],context=context,
        context_definition=dict(backend_environment='SHA256 of qualified native runtime receipt',
            resource_envelope=envelope,input_contract=input_contract,
            structured_hash='SHA256 UTF-8 json sort_keys=True separators=(comma,colon) allow_nan=False'),
        bindings=bindings,profiles={k:v.identity() for k,v in loaded.items()},coverage=coverage,
        estimates=estimates,activation_geometry=loaded['preparation'].activation_layout(),summary=summary,
        full_qualified=False,production_profile_frozen=False,numerical_adapter_correctness_qualified=False,
        n_correct=None,common_slo_qualified=False,independent_runs=1 if len(runs)==1 else None,ci=None,
        protection_raw=seal,offline_only=True,new_gpu_runs=0,no_figures=True,
        raw_evidence=[entry for r in runs for entry in r['raw_evidence']] +
            [ref(path) for path in (config_path,ROOT/'results/ieee_tc/p2_backend_qualification/runtime_20260926/cuda_import_attempt3.json',
                                   Path(__file__),ROOT/'scripts/ieee_tc_preflight.py')])
    if len(runs)>1:
        del result['source_run'], result['source_execution_commit']
        result.update(source_runs=[r['source_run'] for r in runs],
            source_execution_commits=[(d/'source_commit.txt').read_text().strip() for d,_ in directories],
            calibration_executions=len(runs),
            repetition_scope='complementary_class_measurements_not_independent_full_workload_repeats',
            clock_domains_joined=False)
    elif campaign in ('d307', 'd307-attempt2', 'd344', 'd356', 'd362', 'd524'):
        result['raw_evidence'].append(ref(RAW/'source_commit.txt'))
    if control_audit is not None:
        result['control_interleave_audit'] = control_audit
    if metadata_audit is not None:
        result['metadata_publication_audit'] = metadata_audit
    if campaign in ('d362', 'd524'):
        result['file_demand_transaction_audit'] = audit_file_demand_source_run(raw)
    save('manifest.json',result)
    print(json.dumps(dict(passed=True,summary=summary,profiles=result['profiles'],coverage=coverage),indent=2),flush=True)


if __name__ == '__main__':
    if not sys.argv[1:]:
        main()
    elif sys.argv[1:] in (['d307'], ['d307-attempt2'], ['d344'], ['d344-merged'], ['d356'], ['d356-merged'], ['d362'], ['d524']):
        main(sys.argv[1])
    elif sys.argv[1:] == ['--history-length-coverage']:
        export_history_length_coverage()
    elif sys.argv[1:] == ['--completed-length-candidate']:
        export_completed_length_candidate()
    else:
        raise SystemExit('Expected no argument, d307, d307-attempt2, d344, d344-merged, d356, d356-merged, --history-length-coverage or --completed-length-candidate')
