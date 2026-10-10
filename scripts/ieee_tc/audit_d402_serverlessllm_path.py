#!/usr/bin/env python3
"""ServerlessLLM evidence recipe; reuses existing baseline/native validators.

Read-only with respect to experiment inputs. Output is exclusive, small, and
never upgrades a prefix qualification to a main performance comparison.
"""
import argparse
import ast
import asyncio
import hashlib
import json
import logging
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace as NS


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


async def reproduce_queue(source):
    """Execute the actual control-loop AST with a three-item mocked queue.

    No Ray, GPU, network or altered scheduling body. This proves a source-level
    failure, not the unlogged exception type in the historical live process.
    """
    tree = ast.parse(source.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
               and x.name == 'StorageAwareScheduler')
    fn = next(x for x in cls.body if isinstance(x, ast.AsyncFunctionDef)
              and x.name == '_control_loop')
    namespace = {'asyncio': asyncio, 'logger': logging.getLogger('queue-reproduction')}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 str(source), 'exec'), namespace)
    futures = [asyncio.get_running_loop().create_future() for _ in range(3)]
    async def empty(*args):
        return {}
    store = NS(**{k: NS(remote=empty) for k in
                  ('get_model_info', 'get_store_info', 'get_hardware_info', 'load_to_host')})
    nodes = {'0': {'free_gpu': 3}}
    async def get_nodes():
        return nodes
    async def schedule(*args):
        return [NS(node_id='0', latency=1, migration_plans=None)]
    updates = []
    async def update(value):
        updates.append(value)
    obj = NS(running=True, queue_lock=asyncio.Lock(), store_manager=store,
             model_loading_queues={'model': [(i, 1, f) for i, f in enumerate(futures)]},
             _get_worker_nodes=get_nodes, schedule=schedule, _update_worker_nodes=update)
    try:
        await asyncio.wait_for(namespace['_control_loop'](obj), timeout=2)
    except IndexError as error:
        result = dict(exception=type(error).__name__, message=str(error),
                      resolved=[f.done() for f in futures],
                      remaining_queue_future_indices=[futures.index(r[2]) for r in obj.model_loading_queues['model']],
                      metadata_updates=len(updates), free_gpu_local=nodes['0']['free_gpu'])
        assert result['resolved'] == [True, True, False]
        assert result['remaining_queue_future_indices'] == [1]
        return result
    raise AssertionError('expected original scheduler queue failure did not reproduce')


def validate_population(config, ready, contracts, native, population):
    """Reconstruct the offered view; do not infer full coverage from a label."""
    counts = {'prefix100': 100, 'prefix1000': 1000, 'adapter_coverage_v1': 500,
              'full4000': 4000}
    if population not in counts:
        raise ValueError('unknown qualification population')
    count = counts[population]
    if config['request_count'] != count or len(contracts) != count:
        raise ValueError('qualification population count differs')
    plan = native.load_http_replay_plan(config)
    if plan.identity() != ready['plan'] or set(contracts) != {e.request_id for e in plan.entries}:
        raise ValueError('offered view differs from frozen input derivation')
    if population == 'full4000':
        identity = plan.identity()
        if (identity.get('source_count') != count or identity.get('count') != count
                or len(plan.entries) != count or identity.get('rate_scale') != 1.
                or identity.get('profile') != 'W0'):
            raise ValueError('full4000 requires the complete unscaled W0 source, not a prefix or transformed view')
    for entry in plan.entries:
        source, contract = json.loads(entry.source_json), contracts[entry.request_id]
        if (contract['source_item_sha256'] != entry.source_sha256
                or contract['adapter_id'] != source['adapter_id']
                or contract['target_tokens'] != min(source['expected_output_tokens'], 256)):
            raise ValueError('offered adapter/target/derived row differs')
        if population == 'adapter_coverage_v1':
            if (plan.profile != 'ADAPTER_COVERAGE_V1'
                    or contract.get('tc_adapter_coverage') != source.get('tc_adapter_coverage')
                    or contract.get('tc_adapter_coverage') is None):
                raise ValueError('whole-pool functional derivation evidence missing')
        elif plan.profile != 'W0' or 'tc_adapter_coverage' in contract:
            raise ValueError('ordinary prefix cannot silently become a coverage view')
    if population == 'adapter_coverage_v1' and len({r['adapter_id'] for r in contracts.values()}) != 500:
        raise ValueError('functional view has incomplete unique adapter coverage')
    return count


def compare_reference_row(observation, contract, source, reference):
    """Compare retained hashes; never turn a mismatch into numerical success."""
    def ids_sha(ids):
        return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    timing = source['native_token_timing']
    ids = observation['completion_token_ids']
    if (not ids or any(type(i) is not int or i < 0 for i in ids)
            or ids_sha(ids) != observation['completion_token_ids_sha256']
            or ids_sha(timing['completion_token_ids']) != reference['native_ids_sha256']
            or timing['completion_token_ids_sha256'] != reference['native_ids_sha256']
            or not source['success']
            or source['request_id'] != reference['request_id']
            or contract['request_id'] != source['request_id']
            or contract['adapter_id'] != source['adapter_id']
            or reference['adapter_id'] != source['adapter_id']
            or observation['native_lora_name'] != source['adapter_id']
            or contract['canonical_prompt_sha256'] != source['canonical_prompt_sha256']
            or observation['native_prompt_token_ids_sha256'] != timing['native_prompt_token_ids_sha256']
            or len(ids) != contract['target_tokens']
            or contract['target_tokens'] != source['requested_completion_tokens']
            or contract['target_tokens'] != reference['reference_tokens']
            or reference['target_tokens'] != contract['target_tokens']):
        raise ValueError('reference input, adapter, target or native output identity differs')
    exact = observation['completion_token_ids_sha256'] == reference['reference_ids_sha256']
    known_reference = (timing['completion_token_ids'] if reference['exact_sequence_match']
                       else reference['reference_prefix'])
    divergence = next((i for i, (a, b) in enumerate(zip(ids, known_reference)) if a != b), None)
    # D385 stores logits only for its own first divergence, not arbitrary later ones.
    reusable_margin = (not exact and divergence is not None
        and divergence == reference['first_divergent_position']
        and ids[divergence] == reference['native_prefix'][divergence])
    return dict(request_id=source['request_id'], adapter_id=source['adapter_id'],
        target_tokens=len(ids), exact_sequence_match=exact,
        native_ids_sha256=observation['completion_token_ids_sha256'],
        reference_ids_sha256=reference['reference_ids_sha256'],
        known_first_divergent_position=None if exact else divergence,
        divergence_position_known=exact or divergence is not None,
        independently_observed_margin_reusable=bool(reusable_margin),
        hf_top1_minus_top2=reference['hf_top1_minus_top2'] if reusable_margin else None,
        native_token_in_hf_top2=reference['native_token_in_hf_top2'] if reusable_margin else None)


def reuse_numerical_reference(root, reference_path, main):
    """D407 bounded independent-reference reuse; no GPU or new reference data."""
    reference = json.loads(reference_path.read_text())
    source_path = Path(reference['source_native_path'])
    if digest(source_path) != reference['source_native_sha256'] or not reference['measurement_complete']:
        raise ValueError('independent reference provenance incomplete')
    source_result = json.loads(source_path.read_text())
    sources = next(iter(source_result['detailed_results'].values()))['requests']
    source_index = {r['request_id']: r for r in sources}
    reference_index = {r['request_id']: r for r in reference['requests']}
    events = [json.loads(l) for l in (root/'launch7b_repaired.launch/replay.jsonl').read_text().splitlines()]
    contracts = {r['request_id']: r for r in events if r['event'] == 'request_contract'}
    native = {r['request_id']: r['body']['metrics']['ieee_tc'] for r in events if r['event'] == 'http_raw_response'}
    cfg = json.loads((root/'http_7b_100.json').read_text())
    if (len(reference_index) != len(reference['requests']) or len(reference_index) != 100
            or set(contracts) != set(reference_index) or set(native) != set(reference_index)
            or set(source_index) != set(reference_index)
            or cfg['trace_sha256'] != reference['trace_sha256']
            or digest(cfg['trace']) != reference['trace_sha256']):
        raise ValueError('bounded witness populations or source trace differ')
    pool_path = Path(cfg['pool_index'])
    if digest(pool_path) != cfg['pool_index_sha256']:
        raise ValueError('pool identity differs')
    pool = {a['id']: a for a in json.loads(pool_path.read_text())['artifacts']}
    classes = {}
    rows = []
    for rid in sorted(native):
        row = compare_reference_row(native[rid], contracts[rid], source_index[rid], reference_index[rid])
        aid = row['adapter_id']
        selected = [f for f in pool[aid]['files'] if f['path'] in ('adapter_model.safetensors', 'adapter_config.json')]
        for file in selected:
            if digest(Path(reference['adapter_root'])/aid/file['path']) != file['sha256']:
                raise ValueError('reference adapter bytes differ from delivered pool')
        weight = next(f['sha256'] for f in selected if f['path'] == 'adapter_model.safetensors')
        row['weight_sha256'] = weight
        group = classes.setdefault(weight, dict(requests=0, exact_sequences=0, adapter_ids=set()))
        group['requests'] += 1
        group['exact_sequences'] += row['exact_sequence_match']
        group['adapter_ids'].add(aid)
        rows.append(row)
    for group in classes.values():
        group['adapter_ids'] = sorted(group['adapter_ids'])
    return dict(schema='ieee_tc_serverless_reference_reuse_v1', display_name='ServerlessLLM',
        count=len(rows), exact_sequences=sum(r['exact_sequence_match'] for r in rows),
        source_adapter_input_identity_complete=True, weight_classes=classes, requests=rows,
        full_semantic_correctness_proof=False, full_performance_qualification=False,
        limitation='Bounded HF/PEFT witness only; mismatches retained, not blanket-classified as rounding. V1 does not require cross-backend exact sequences.',
        sources=[dict(path=str(p), sha256=digest(p)) for p in
            (reference_path, source_path, pool_path, root/'launch7b_repaired.launch/replay.jsonl',
             main/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md',
             main/'docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md', Path(__file__))])


def frozen_slo_summary(rows, observations, groups):
    """Evaluate unchanged native timing by common prompt group, not live CE.

    Caller must first validate terminal/source/content/native identity. This
    pure calculation cannot turn a failed request into a correctness proof.
    """
    counts = dict(offered=len(rows), binding_valid=0, ttft_good=0, tpot_eligible=0,
                  tpot_good=0, joint_good=0)
    if not rows or len({r['request_id'] for r in rows}) != len(rows):
        raise ValueError('unique nonempty offered population required')
    for row in rows:
        if row['status'] != 'http_response':
            continue
        native = observations[row['request_id']]
        prompt = len(native['native_prompt_token_ids'])
        selected = [g for g in groups if
            (g['lower_exclusive'] is None or prompt > g['lower_exclusive']) and
            (g['upper_inclusive'] is None or prompt <= g['upper_inclusive'])]
        if prompt == 0 or len(selected) != 1:
            raise ValueError('native prompt requires exactly one frozen SLO group')
        g, n = selected[0], row['actual_tokens']
        if (type(n) is not int or n < 1 or n != row['target_tokens']
                or native['native_output_tokens'] != n
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                       for v in (row['ttft_ms'], g['ttft_ms'], g['tpot_ms']))
                or g['ttft_ms'] <= 0 or g['tpot_ms'] <= 0):
            raise ValueError('invalid native count, timing or threshold')
        if n == 1:
            if row['tpot_ms'] is not None:
                raise ValueError('single-token TPOT must be N/A')
            p_good = True
        else:
            if (type(row['tpot_ms']) not in (int, float)
                    or not math.isfinite(row['tpot_ms']) or row['tpot_ms'] < 0):
                raise ValueError('missing native TPOT')
            p_good = row['tpot_ms'] <= g['tpot_ms']
            counts['tpot_eligible'] += 1
            counts['tpot_good'] += p_good
        t_good = row['ttft_ms'] <= g['ttft_ms']
        counts['binding_valid'] += 1
        counts['ttft_good'] += t_good
        counts['joint_good'] += t_good and p_good
    return dict(counts, ttft_attainment=counts['ttft_good']/len(rows),
                joint_attainment=counts['joint_good']/len(rows),
                conditional_tpot_attainment=(counts['tpot_good']/counts['tpot_eligible']
                                            if counts['tpot_eligible'] else None),
                observed_feasible=(counts['binding_valid'] == len(rows)
                                   and counts['joint_good']*100 >= 95*len(rows)),
                independent_numerical_proof=False)


def timing_stats(values):
    ordered = sorted(values)
    return dict(count=len(ordered), mean=statistics.mean(ordered) if ordered else None,
                p50_type1=ordered[math.ceil(.5*len(ordered))-1] if ordered else None,
                p95_type1=ordered[math.ceil(.95*len(ordered))-1] if ordered else None)


def audit(root, baseline, main, scheduler_contract='original', population='prefix100', slo_freeze=None):
    sys.path[:0] = [str(baseline/'scripts'), str(main)]
    import summarize_serverlessllm_replay as analyzer
    import prepare_ieee_tc_serverless_stack as native
    from faaslora.storage.http_artifact_store import preparation_content_sha256
    replay = root/'launch7b_repaired.launch/replay.jsonl'
    summary = analyzer.audit_tc_http_journal(replay)
    events = [json.loads(line) for line in replay.read_text().splitlines()]
    kinds = ('request_contract', 'request_created', 'http_headers_sent', 'http_raw_response', 'http_response')
    index = {k: {e['request_id']: e for e in events if e['event'] == k} for k in kinds}
    ready = summary['ready_identity']
    assert digest(ready['plan']['source_path']) == ready['plan']['source_sha256']
    config_path = root/{'adapter_coverage_v1': 'http_7b_500.json',
                       'prefix1000': 'http_7b_1000.json', 'prefix100': 'http_7b_100.json',
                       'full4000': 'http_7b_4000.json'}[population]
    assert digest(config_path) == ready['config_sha256']
    count = validate_population(json.loads(config_path.read_text()), ready,
                                index['request_contract'], native, population)
    assert summary['counts']['N_response'] == summary['counts']['N_plan'] == count
    cfg = json.loads((root/'model7b_repaired/configuration.json').read_text())
    qualifier = json.loads((root/'model7b_repaired/model_qualification.json').read_text())
    assert qualifier['passed'] and qualifier['physical_lifecycle']['lifecycle_qualified']
    native.audit_backend_receipts(qualifier, cfg['backend_config']['tc_worker_audit'], ready['clock_id'])
    view = json.loads((root/'source_repaired/measurement_manifest.json').read_text())
    assert view.get('scheduler_queue', 'original') == scheduler_contract
    scheduler_source = root/'storage_aware_scheduler_executed.py'
    if scheduler_contract == 'request_identity_v1':
        observed = qualifier['scheduler_readback']
        assert digest(scheduler_source) == observed['source_sha256']
        assert observed['source_sha256'] == view['measured_sha256']['sllm/schedulers/storage_aware_scheduler.py']
        native.validate_scheduler_readback(observed,
            source=root/'source_repaired/sllm/schedulers/storage_aware_scheduler.py',
            service_cgroup=cfg['backend_config']['tc_worker_audit']['service_cgroup'],
            service_cpus=cfg['backend_config']['tc_worker_audit']['service_cpus'])
        if population == 'prefix100':
            assert len(qualifier['backend_ready_receipts']) == 4  # D404 affected scale-out gate.
        else:
            assert qualifier['backend_ready_receipts']  # All served epochs already checked above.
        assert all(q['total'] == q['done'] == 0 for q in observed['queues'].values())
        queue_evidence = dict(contract=scheduler_contract, readback=observed,
            conclusion='actual served backends verified; live native loop and no pending allocations')
    elif scheduler_contract == 'original':
        queue_evidence = asyncio.run(reproduce_queue(scheduler_source))
    else:
        raise ValueError('unknown scheduler contract')
    frozen = cfg['backend_config']['tc_remote_artifacts']
    assert digest(frozen['content_manifest']) == frozen['content_manifest_file_sha256']
    pool = {a['id']: a for a in json.loads(Path(frozen['content_manifest']).read_text())['artifacts']}
    if population == 'adapter_coverage_v1':
        assert set(pool) == {r['adapter_id'] for r in index['request_contract'].values()}
    remote_rows = [json.loads(line) for line in (root/'remote_transfers.jsonl').read_text().splitlines()]
    remote = {row['transfer_id']: row for row in remote_rows}
    assert len(remote) == len(remote_rows)
    receipts = {p.stem: (p, json.loads(p.read_text())) for p in (root/'remote_resolver_evidence/receipts').glob('*.json')}
    used_transfers, hits, transferred, native_error = set(), 0, 0, 0.0
    health = json.loads((root/'remote_health.json').read_text())
    for rid, terminal in index['http_response'].items():
        contract, raw = index['request_contract'][rid], index['http_raw_response'][rid]
        observation = raw['body']['metrics']['ieee_tc']
        ids = observation['native_prompt_token_ids']
        assert hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() == contract['native_prompt_token_ids_sha256']
        recalculated = native.validate_http_observation(dict(contract, input_token_ids=ids),
            raw['body'], index['http_headers_sent'][rid], raw['client_completed_s'], ready['clock_id'], ingress_required=True)
        for key, value in recalculated.items():
            previous = terminal['response'][key]
            if key.endswith('_ms') and value is not None:
                error = abs(value-previous)
                assert math.isfinite(error) and error <= 1
                native_error = max(native_error, error)
            else:
                assert value == previous
        c = observation['control_observation']
        path, receipt = receipts[contract['adapter_id']]
        assert digest(path) == c['tc_artifact_receipt_sha256']
        files = {f['path']: (f['size_bytes'], f['sha256']) for f in pool[contract['adapter_id']]['files']}
        assert preparation_content_sha256(files) == receipt['content_sha256'] == c['tc_artifact_content_sha256']
        assert receipt['content_manifest_sha256'] == c['tc_artifact_manifest_sha256']
        evidence = receipt['evidence']
        server = remote[c['tc_artifact_transfer_id']]
        assert server['artifact_id'] == receipt['adapter_id'] == contract['adapter_id']
        assert server['outcome'] == 'sent' and server['pack_performed'] is False and server['temporary_created'] is False
        assert server['delivery_mode'] == evidence['remote_delivery_mode'] == 'prepublished_gzip_v1'
        assert server['clock_id'] == health['7b']['clock_id']
        assert server['transfer_id'] == evidence['http_transfer_id']
        assert server['archive_sha256'] == evidence['remote_archive_sha256']
        assert server['archive_bytes'] == server['bytes_read'] == server['bytes_written'] == evidence['transferred_bytes']
        assert evidence['published_archive_verified'] and evidence['content_verified'] and not evidence['remote_pack_performed']
        assert evidence['loading_clock_id'] == ready['clock_id']
        used_transfers.add(server['transfer_id'])
        hits += c['tc_artifact_cache_hit']
        transferred += c['tc_artifact_transferred_bytes']
    assert used_transfers == set(remote) and len(receipts) == len(remote)
    assert transferred == sum(row['bytes_written'] for row in remote_rows)
    # Verify actual materialized bytes before reclaiming this rebuildable cache.
    payload = 0
    for aid, (_, receipt) in receipts.items():
        for file in pool[aid]['files']:
            path = Path(frozen['cache_dir'])/'objects'/aid/file['path']
            assert path.stat().st_size == file['size_bytes'] and digest(path) == file['sha256']
            payload += file['size_bytes']
    resource_path = root/'model7b_repaired/physical/physical_resource_summary.json'
    resource = json.loads(resource_path.read_text())
    physical = json.loads(Path(resource['summary_path']).read_text())
    assert digest(resource['summary_path']) == resource['summary_sha256']
    assert digest(resource['result_file']['path']) == resource['result_file']['sha256']
    assert digest(resource['request_terminals_path']) == resource['request_terminals_sha256']
    for row in resource['allocation_journals']:
        assert digest(row['path']) == row['sha256'] and row['released']
    assert physical['measurement_complete'] and not physical['open_lease_ids']
    integral = sum(b-a for intervals in physical['device_intervals'].values() for a, b in intervals)
    assert abs(integral-physical['gpu_seconds']) < 1e-9
    assert abs(sum(physical['window_gpu_seconds'].values())-integral) < 1e-9
    assert physical['n_native_contract_complete'] == physical['n_terminal'] == physical['n_plan'] == count
    samples = [e for e in map(json.loads, (root/'launch7b_repaired.launch/watchdog.jsonl').read_text().splitlines()) if e['event'] == 'resource_sample']
    assert samples
    service_events = {k: max(s['service'].get('memory.events', {}).get(k, 0) for s in samples) for k in ('high', 'max', 'oom', 'oom_kill')}
    source_paths = [replay, resource_path, Path(resource['summary_path']), Path(resource['result_file']['path']),
                    config_path, root/'remote_transfers.jsonl', root/'remote_monitor.log',
                    root/'source_repaired/measurement_manifest.json', root/'storage_aware_scheduler_executed.py',
                    Path(analyzer.__file__), Path(native.__file__), Path(__file__)]
    result = {k: v for k, v in summary.items() if k not in ('ready_identity', 'diagnostic_rows', 'assignment_gaps_s', 'kind', 'display_name')}
    result.update(schema=('ieee_tc_d402_affected_path_audit_v1' if scheduler_contract == 'original'
                          else 'ieee_tc_affected_path_audit_v2'), display_name='ServerlessLLM',
        remote_affected_path_qualified=True, complete_system_qualified=False,
        source_plan=ready['plan'], native_revalidated_requests=count, max_native_recompute_error_ms=native_error,
        physical_gpu_seconds=integral, physical_window_gpu_seconds=physical['window_gpu_seconds'],
        n_correct=None, independent_lora_numerical_correctness=False,
        remote_transfers=len(remote), remote_wire_bytes=transferred, verified_materialized_bytes=payload,
        cache_hit_requests=hits, request_time_pack_count=0, backend_ready_receipts=len(qualifier['backend_ready_receipts']),
        remote_clock_ids=sorted({r['clock_id'] for r in remote_rows}),
        service_memory_event_maxima=service_events,
        service_memory_peak_bytes=max(s['service'].get('memory.peak', 0) for s in samples),
        service_swap_max_bytes=max(s['service'].get('memory.swap.current', 0) for s in samples),
        host_min_available_bytes=min(s['host']['available_bytes'] for s in samples),
        disk_min_free_bytes=min(f['free_bytes'] for s in samples for f in s.get('filesystems', [])),
        queue_source_reproduction=queue_evidence if scheduler_contract == 'original' else None,
        scheduler_contract=scheduler_contract,
        scheduler_qualification=queue_evidence if scheduler_contract != 'original' else None,
        raw_sources=[dict(path=str(p), sha256=digest(p)) for p in source_paths],
        limitations=[('500-ID functional mapping of existing prompts/arrivals/targets; not a W0 performance observation.'
                      if population == 'adapter_coverage_v1' else
                      'Complete 4000-request development W0 observation; not unseen evaluation or a cross-scenario configuration optimum.'
                      if population == 'full4000' else
                      f'{count}-request development prefix, not a full W0 comparison or configuration optimum.'),
            'Raw unknown numerical correctness and old qualification flags are preserved.',
            ('Four logical instances include starting instances; three served engines. Scheduler queue bug independently reproduced; live exception was not logged.'
             if scheduler_contract == 'original' else
             'Actual backend-ready receipts and live loop qualify the affected path; qualification does not quantify a causal latency benefit.'),
            'Fixed native-store four-GPU pool is charged until actual release; not a claim about every ServerlessLLM deployment.',
            ('Cross-system comparison requires a separate full input, generation, initial-state and resource-contract join.'
             if population == 'full4000' else
             'No relative performance comparison with the 4000-request PrimeLoRA observation.')])
    if slo_freeze is not None:
        frozen_slo = json.loads(slo_freeze.read_text())
        if (frozen_slo['model'] != 'llama2_7b_main_v2_publicmix'
                or frozen_slo['slo_reference_frozen'] is not True
                or frozen_slo['thresholds_common_to_all_systems'] is not True
                or frozen_slo['metric_protocol_sha256'] != digest(main/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md')
                or frozen_slo['single_run_amendment_sha256'] != digest(main/'docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md')):
            raise ValueError('common frozen SLO protocol identity differs')
        for key in ('path', 'audit_path'):
            sha_key = 'sha256' if key == 'path' else 'audit_sha256'
            if digest(main/frozen_slo['calibration'][key]) != frozen_slo['calibration'][sha_key]:
                raise ValueError('frozen common calibration content differs')
        observations = {rid: r['body']['metrics']['ieee_tc'] for rid, r in index['http_raw_response'].items()}
        rows = summary['diagnostic_rows']
        result['common_slo'] = frozen_slo_summary(rows, observations, frozen_slo['groups'])
        result['common_slo'].update(freeze_path=str(slo_freeze), freeze_sha256=digest(slo_freeze),
            scope=('complete W0 native-binding observation; not a cross-scenario optimum or independent numerical proof'
                   if population == 'full4000' else
                   'complete qualified population only; prefix or coverage is not full W0 qualification'),
            physical_gpu_seconds_per_binding_valid_request=integral/count)
        result['conditional_metrics']['tpot_ms'] = timing_stats(
            [r['tpot_ms'] for r in rows if r['status'] == 'http_response' and r['actual_tokens'] > 1])
        if population in ('prefix1000', 'full4000'):
            # Exhaustive, nonoverlapping source-order slices; diagnostic only.
            # Never drop the startup portion from the primary SLO denominator.
            ordered = sorted(rows, key=lambda r: r['planned_arrival_s'])
            result['source_order_diagnostics'] = []
            slices = ([(lo, lo+500) for lo in range(0, 4000, 500)]
                      if population == 'full4000' else [(0, 100), (100, 500), (500, 1000)])
            for lo, hi in slices:
                subset = ordered[lo:hi]
                result['source_order_diagnostics'].append(dict(
                    source_indices_inclusive=[lo, hi-1],
                    common_slo=frozen_slo_summary(subset, observations, frozen_slo['groups']),
                    metrics={k: timing_stats([r[k] for r in subset if r[k] is not None])
                             for k in ('ttft_ms', 'tpot_ms', 'router_queue_ms', 'service_ttft_ms')},
                    interpretation='Post-run descriptive slice; not an independent run or startup-free primary result.'))
        result['raw_sources'].append(dict(path=str(slo_freeze), sha256=digest(slo_freeze)))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--main', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scheduler-contract', choices=('original', 'request_identity_v1'), default='original')
    parser.add_argument('--population', choices=('prefix100', 'prefix1000', 'adapter_coverage_v1', 'full4000'), default='prefix100')
    parser.add_argument('--reference-result', type=Path)
    parser.add_argument('--slo-freeze', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args.root, args.baseline, args.main, args.scheduler_contract, args.population, args.slo_freeze)
    if args.reference_result:
        if args.population != 'prefix100':
            raise ValueError('independent reference is the unchanged first 100 requests only')
        report['bounded_numerical_reference'] = reuse_numerical_reference(args.root, args.reference_result, args.main)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in
        ('raw_sources', 'conditional_metrics', 'source_plan', 'bounded_numerical_reference')}, indent=2))
