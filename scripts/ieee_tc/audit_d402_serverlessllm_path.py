#!/usr/bin/env python3
"""ServerlessLLM/vLLM/S-LoRA/Loquetier evidence recipe; reuses native validators.

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
              'full4000': 4000, 'full4000_w1': 4000}
    if population not in counts:
        raise ValueError('unknown qualification population')
    count = counts[population]
    if config['request_count'] != count or len(contracts) != count:
        raise ValueError('qualification population count differs')
    plan = native.load_http_replay_plan(config)
    if plan.identity() != ready['plan'] or set(contracts) != {e.request_id for e in plan.entries}:
        raise ValueError('offered view differs from frozen input derivation')
    expected_profile = 'W1' if population == 'full4000_w1' else 'W0'
    if population in ('full4000', 'full4000_w1'):
        identity = plan.identity()
        if (identity.get('source_count') != count or identity.get('count') != count
                or len(plan.entries) != count or identity.get('rate_scale') != 1.
                or identity.get('profile') != expected_profile):
            raise ValueError(f'{population} requires the complete unscaled {expected_profile} view')
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
        elif plan.profile != expected_profile or 'tc_adapter_coverage' in contract:
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
            or (Path(observation['native_adapter_path']).name
                if observation.get('timing_contract') in ('ieee_tc_slora_native_v1', 'ieee_tc_loquetier_native_v1')
                else observation['native_lora_name']) != source['adapter_id']
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


def reuse_numerical_reference(root, reference_path, main, scheduler_contract='original'):
    """D407 bounded independent-reference reuse; no GPU or new reference data."""
    reference = json.loads(reference_path.read_text())
    source_path = Path(reference['source_native_path'])
    if digest(source_path) != reference['source_native_sha256'] or not reference['measurement_complete']:
        raise ValueError('independent reference provenance incomplete')
    source_result = json.loads(source_path.read_text())
    sources = next(iter(source_result['detailed_results'].values()))['requests']
    source_index = {r['request_id']: r for r in sources}
    reference_index = {r['request_id']: r for r in reference['requests']}
    replay = root/('launch7b.launch/replay.jsonl' if scheduler_contract in ('resident_rr_v1', 'slora_resident_rr_v1', 'loquetier_static_shards_v1')
                  else 'launch7b_repaired.launch/replay.jsonl')
    events = [json.loads(l) for l in replay.read_text().splitlines()]
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
    return dict(schema='ieee_tc_serverless_reference_reuse_v1',
        display_name=({'resident_rr_v1': 'vLLM', 'slora_resident_rr_v1': 'S-LoRA', 'loquetier_static_shards_v1': 'Loquetier'}
                      .get(scheduler_contract, 'ServerlessLLM')),
        count=len(rows), exact_sequences=sum(r['exact_sequence_match'] for r in rows),
        source_adapter_input_identity_complete=True, weight_classes=classes, requests=rows,
        full_semantic_correctness_proof=False, full_performance_qualification=False,
        limitation='Bounded HF/PEFT witness only; mismatches retained, not blanket-classified as rounding. V1 does not require cross-backend exact sequences.',
        sources=[dict(path=str(p), sha256=digest(p)) for p in
            (reference_path, source_path, pool_path, replay,
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


def validate_terminal_outcome(summary, qualifier, count, outcome,
                              scheduler_contract='original', physical_summary=None):
    """Separate complete failure diagnostics from the unchanged success gate.

    This only qualifies the observed population, never adapter correctness or
    performance. Native, content, worker and physical checks remain mandatory.
    """
    if outcome not in ('success', 'complete_with_failures'):
        raise ValueError('unknown terminal outcome contract')
    counts = summary['counts']
    keys = ('N_plan', 'N_arrived', 'N_submitted', 'N_terminal', 'N_response', 'N_failed')
    if (any(type(counts.get(k)) is not int or counts[k] < 0 for k in keys)
            or any(counts[k] != count for k in keys[:4])
            or counts['N_response'] + counts['N_failed'] != count
            or summary.get('measurement_complete') is not True
            or summary.get('workload_passed') is not (counts['N_failed'] == 0)):
        raise ValueError('complete offered/submitted/terminal population required')
    if outcome == 'success':
        if counts['N_failed'] or qualifier.get('passed') is not True:
            raise ValueError('success outcome requires all responses and passed qualification')
    else:
        if (counts['N_failed'] == 0 or qualifier.get('passed') is not False
                or any(qualifier.get(k) is not None for k in
                       ('physical_confirmation_error', 'worker_audit_error', 'delete_error',
                        'physical_release_error', 'physical_measurement_error'))):
            raise ValueError('explicit complete request-failure diagnostic required; other errors remain unqualified')
        resident_contracts = {
            'resident_rr_v1': ('vllm', 'vllm_resident_pool_v1'),
            'slora_resident_rr_v1': ('slora', 'slora_resident_pool_v1'),
            'loquetier_static_shards_v1': ('loquetier', 'loquetier_static_shards_v1'),
        }
        if scheduler_contract in resident_contracts:
            backend, lifecycle = resident_contracts[scheduler_contract]
            physical = physical_summary or {}
            completion = qualifier.get('http_completion', {})
            # The resident runner's terminal check deliberately rejects ANY
            # failed request. Its second error conflates missing native success
            # with missing resource measurement. Accept neither by message alone:
            # independently require the closed physical/population evidence.
            if (qualifier.get('schema') != f'ieee_tc_{backend}_native_model_qualification_v1'
                    or qualifier.get('error') != 'ValueError: incomplete/failed native HTTP workload; retained for diagnosis'
                    or qualifier.get('physical_terminal_error') != 'physical or native workload measurement incomplete'
                    or qualifier.get('physical_release_errors') is not None
                    or qualifier.get('physical_lifecycle', {}).get('mode') != lifecycle
                    or qualifier.get('physical_lifecycle', {}).get('measurement_complete') is not True
                    or completion.get('event') != 'http_replay_complete'
                    or any(completion.get(k) != counts[k] for k in
                           ('N_plan', 'N_arrived', 'N_terminal', 'N_response', 'N_failed'))
                    or physical.get('contract') != 'physical_gpu_resource_summary_v1'
                    or physical.get('measurement_complete') is not True
                    or physical.get('eligible_physical_measurement') is not True
                    or physical.get('open_lease_ids') != []
                    or physical.get('n_plan') != count or physical.get('n_terminal') != count
                    or physical.get('n_native_contract_complete') != counts['N_response']
                    or physical.get('n_interrupted') != 0):
                raise ValueError('resident failure diagnostic requires exact terminal and closed physical evidence')
        elif (scheduler_contract not in ('original', 'request_identity_v1')
              or qualifier.get('error') != 'ValueError: external HTTP qualification failed; do not label binding as correctness'):
            raise ValueError('explicit complete request-failure diagnostic required; other errors remain unqualified')
    return counts['N_response']


def validate_vllm_ready_row(row, cfg, control, ordinal, service, clock_id):
    """Exact static RR/ready/source contract, separate from ServerlessLLM actors."""
    index = ordinal % len(cfg['gpu_ids'])
    identity = row.get('identity', {})
    at, entry = row.get('at'), control.get('tc_backend_entry_s')
    if (row.get('schema') != 'ieee_tc_vllm_backend_ready_v1'
            or row.get('index') != index or control.get('tc_route_ordinal') != ordinal
            or control.get('instance_id') != f'vllm-{index}'
            or row.get('cuda_visible') != str(cfg['gpu_ids'][index])
            or row.get('clock_id') != clock_id or control.get('tc_clock_id') != clock_id
            or identity.get('cgroup') != service['cgroup']
            or identity.get('affinity') != service['affinity']
            or any(type(identity.get(k)) is not int or identity[k] <= 0 for k in ('pid', 'start_ticks'))
            or row.get('engine_args') != cfg['engine_args']
            or row.get('native_toolchain') != cfg['native_toolchain']
            or any(type(t) not in (int, float) or not math.isfinite(t) for t in (at, entry))
            or at > entry):
        raise ValueError('vLLM static route/ready/containment/toolchain differs')
    return index


def audit_vllm_receipts(root, cfg, qualifier, ready, config_path, native):
    from replay_vllm_engine_trace import validate_native_service_config
    http = json.loads(config_path.read_text())
    validate_native_service_config(cfg, http)
    if (qualifier['schema'] != 'ieee_tc_vllm_native_model_qualification_v1'
            or qualifier['physical_lifecycle']['mode'] != 'vllm_resident_pool_v1'
            or qualifier['physical_lifecycle']['measurement_complete'] is not True
            or cfg['source_sha256'] != qualifier['source_sha256']
            or any(cfg.get(k) != v for k, v in qualifier['configuration'].items())):
        raise ValueError('explicit native vLLM service/lifecycle identity differs')
    for path, sha in cfg['source_sha256'].items():
        if digest(path) != sha:
            raise ValueError('executed native adapter source changed')
    plan = native.load_http_replay_plan(http)
    ordinals = {e.request_id: i for i, e in enumerate(plan.entries)}
    receipts, seen_indices = {}, set()
    worker_path = root/'model7b/worker_config.json'
    for request in qualifier['requests']:
        observation = request['response']['metrics']['ieee_tc']
        c = observation['control_observation']
        rid = c['tc_worker_receipt_id']
        if len(rid) != 32 or any(ch not in '0123456789abcdef' for ch in rid):
            raise ValueError('invalid native receipt ID')
        path = root/'model7b/worker_receipts'/(rid+'.json')
        row = json.loads(path.read_text())
        if (path.is_symlink() or digest(path) != c['tc_worker_receipt_sha256']
                or row['receipt_id'] != rid or row['worker_config_sha256'] != digest(worker_path)
                or row['runner_sha256'] != cfg['source_sha256'][str(Path(sys.modules['replay_vllm_engine_trace'].__file__).resolve())]):
            raise ValueError('vLLM receipt/worker-config source differs')
        i = validate_vllm_ready_row(row, cfg, c, ordinals[request['request_id']],
                                   qualifier['service'], ready['clock_id'])
        aid = observation['native_lora_name']
        # D415 checks the native output's path at runtime but retains only
        # name/int-ID in JSON. Do not invent a missing persisted path field.
        if (observation['native_lora_int_id'] != cfg['adapters'][aid][0]
                or cfg['adapters'][aid][1] != str(Path(cfg['remote_config']['cache_dir'])/'objects'/aid)):
            raise ValueError('native LoRA object differs from frozen adapter mapping')
        snapshot = row['snapshot']
        if snapshot['version'] != '0.30.0' or snapshot['schema'] != 'ieee_tc_vllm030_native_snapshot_v1':
            raise ValueError('native observation version differs')
        for path0, sha in row['native_api_sha256'].items():
            if digest(path0) != sha:
                raise ValueError('native API source changed')
        for relative, sha in snapshot['source_sha256'].items():
            if digest(Path(snapshot['package_root'])/relative) != sha:
                raise ValueError('native observation producer source changed')
        allocation = qualifier['physical_allocations'][i]
        spawns = [e for e in allocation['events'] if e['event'] == 'worker_spawn']
        if not allocation['released'] or len(spawns) != 1 or spawns[0]['worker'] != row['identity']:
            raise ValueError('ready API process differs from physical owner spawn')
        receipts[rid] = dict(path=str(path), sha256=digest(path))
        seen_indices.add(i)
    if seen_indices != set(range(len(cfg['gpu_ids']))):
        raise ValueError('resident qualification did not serve every declared worker')
    return list(receipts.values())


def validate_slora_ready_row(row, cfg, control, ordinal, service, clock_id, observation):
    """Bind actual native GPU child, not its HTTP parent, to static RR and pool."""
    index = ordinal % len(cfg['gpu_ids'])
    identity, parent = row.get('identity', {}), row.get('api_parent_identity', {})
    process = {k: identity.get(k) for k in ('pid', 'start_ticks')}
    process['clock_id'] = clock_id
    registry = {path: i for i, path in cfg['adapters'].values()}
    at, notice, entry = row.get('at'), cfg.get('deployment_notice_s'), control.get('tc_backend_entry_s')
    if (row.get('schema') != 'ieee_tc_slora_backend_ready_v1'
            or row.get('index') != index or control.get('tc_route_ordinal') != ordinal
            or control.get('instance_id') != f'slora-{index}'
            or row.get('clock_id') != clock_id or control.get('tc_clock_id') != clock_id
            or row.get('cuda_visible') != str(cfg['gpu_ids'][index])
            or row.get('cuda_device_count') != 1 or row.get('tp') != 1
            or any(p.get('cgroup') != service['cgroup'] or p.get('affinity') != service['affinity']
                   or any(type(p.get(k)) is not int or p[k] <= 0 for k in ('pid', 'start_ticks'))
                   for p in (identity, parent))
            or identity.get('pid') == parent.get('pid')
            or row.get('native_input_params') != cfg['native_input_params']
            or row.get('native_environment') != cfg['native_environment']
            or row.get('host_layout') != cfg['host_layout']
            or row.get('source_view_manifest_sha256') != cfg['source_view_manifest_sha256']
            or row.get('model_dir') != cfg['native_args']['model_dir']
            or row.get('ready_boundary') != 'native_full_pool_and_model_initialized'
            or len(registry) != 500 or set(registry.values()) != set(range(500))
            or row.get('adapter_count') != 500 or row.get('adapter_ids') != registry
            or observation.get('timing_contract') != 'ieee_tc_slora_native_v1'
            or observation.get('native_process') != process
            or control.get('tc_worker_native_process') != process
            or observation.get('native_adapter_path') not in registry
            or observation.get('native_adapter_index') != registry.get(observation.get('native_adapter_path'))
            or control.get('tc_worker_ready_s') != at
            or control.get('tc_deployment_notice_s') != notice
            or any(type(t) not in (int, float) or not math.isfinite(t) for t in (notice, at, entry))
            or not notice <= at <= entry):
        raise ValueError('S-LoRA native child/ready/full-pool/route/configuration differs')
    return index


def validate_slora_allocation(allocation, row, cfg, index, clock_id, completed, *, backend='slora'):
    """Shared single-GPU physical evidence; S-LoRA has a distinct API parent."""
    if backend not in ('slora', 'loquetier'):
        raise ValueError('unknown physical native process contract')
    events = allocation['events']
    by_kind = {kind: [e for e in events if e['event'] == kind] for kind in
               ('acquire', 'worker_spawn', 'native_workers', 'native_workers_exited', 'release')}
    if (allocation.get('released') is not True or len(allocation.get('gpu_uuids', [])) != 1
            or any(len(es) != 1 for es in by_kind.values())
            or any(e.get('clock_id') != clock_id or e.get('lease_id') != allocation['lease_id']
                   or e.get('owner_id') != allocation['lease_id']
                   or e.get('gpu_uuids') != allocation['gpu_uuids'] for e in events)):
        raise ValueError('S-LoRA complete TP1 physical lifetime required')
    acquire, spawn, native, exited, release = [by_kind[k][0] for k in by_kind]
    gpu = allocation['gpu_uuids'][0]
    devices = [d for d in native['census']['devices'] if d['gpu_uuid'] == gpu]
    workers = [p for d in devices for p in d['processes'] if p['kind'] == 'compute']
    times = [cfg['deployment_notice_s'], acquire['at'], spawn['at'], row['at'],
             native['at'], completed, exited['at'], release['at']]
    if (spawn['worker'] != row['api_parent_identity' if backend == 'slora' else 'identity']
            or native['workers'] != [dict(device_uuid=gpu, pid=row['identity']['pid'], worker_rank=index)]
            or len(devices) != 1 or devices[0]['index'] != cfg['gpu_ids'][index]
            or len(workers) != 1 or workers[0]['identity'] != row['identity']
            or workers[0]['pid'] != row['identity']['pid'] or workers[0]['service_member'] is not True
            or exited['worker_pids'] != [row['identity']['pid']]
            or any(type(t) not in (int, float) or not math.isfinite(t) for t in times)
            or times != sorted(times)):
        raise ValueError('S-LoRA actual worker differs from physical lease/census/exit')
    released_devices = [d for d in release['census']['devices'] if d['gpu_uuid'] == gpu]
    if (any(e['census'].get('source') != 'nvml_v3_compute_and_graphics'
            or e['census'].get('clock_id') != clock_id for e in (acquire, native, release))
            or len(released_devices) != 1
            or any(p['identity'] is None or p['kind'] == 'compute'
                   or p['service_member'] or p['previously_owned'] for p in released_devices[0]['processes'])):
        raise ValueError('S-LoRA physical release lacks empty native census')


def audit_slora_receipts(root, cfg, qualifier, ready, config_path, native):
    from replay_vllm_engine_trace import validate_slora_service_config, verify_slora_source_view, SLORA_INPUT_KEYS
    http = json.loads(config_path.read_text())
    validate_slora_service_config(cfg, http)
    verify_slora_source_view(cfg)
    if (qualifier['schema'] != 'ieee_tc_slora_native_model_qualification_v1'
            or qualifier['physical_lifecycle']['mode'] != 'slora_resident_pool_v1'
            or qualifier['physical_lifecycle']['measurement_complete'] is not True
            or cfg['source_sha256'] != qualifier['source_sha256']
            or any(cfg.get(k) != v for k, v in qualifier['configuration'].items())
            or len(qualifier['physical_allocations']) != len(cfg['gpu_ids'])
            or len({a['lease_id'] for a in qualifier['physical_allocations']}) != len(cfg['gpu_ids'])
            or len({g for a in qualifier['physical_allocations'] for g in a['gpu_uuids']}) != len(cfg['gpu_ids'])
            or cfg['native_input_params'] != {k: cfg['native_args'][k] for k in SLORA_INPUT_KEYS}
            or any(p != str(Path(cfg['remote_config']['cache_dir'])/'objects'/aid)
                   for aid, (i, p) in cfg['adapters'].items())):
        raise ValueError('explicit S-LoRA configuration/lifecycle identity differs')
    for path, sha in cfg['source_sha256'].items():
        if digest(path) != sha:
            raise ValueError('executed S-LoRA adapter source changed')
    ordinals = {e.request_id: i for i, e in enumerate(native.load_http_replay_plan(http).entries)}
    receipts, seen_indices = {}, set()
    worker_path = root/'model7b/worker_config.json'
    for request in qualifier['requests']:
        o = request['response']['metrics']['ieee_tc']
        c, rid = o['control_observation'], o['control_observation']['tc_worker_receipt_id']
        if len(rid) != 32 or set(rid)-set('0123456789abcdef'):
            raise ValueError('invalid S-LoRA receipt ID')
        path = root/'model7b/worker_receipts'/(rid+'.json')
        row = json.loads(path.read_text())
        rpc = Path(cfg['source_view'])/'slora/server/router/model_infer/model_rpc.py'
        runner = str(Path(sys.modules['replay_vllm_engine_trace'].__file__).resolve())
        if (path.is_symlink() or digest(path) != c['tc_worker_receipt_sha256']
                or row['receipt_id'] != rid or row['worker_config_sha256'] != digest(worker_path)
                or row['runner_sha256'] != cfg['source_sha256'][runner]
                or row['rpc_path'] != str(rpc.resolve()) or row['rpc_sha256'] != digest(rpc)):
            raise ValueError('S-LoRA receipt/worker/source differs')
        i = validate_slora_ready_row(row, cfg, c, ordinals[request['request_id']],
                                    qualifier['service'], ready['clock_id'], o)
        allocation = qualifier['physical_allocations'][i]
        records = [json.loads(line) for line in Path(allocation['journal']).read_text().splitlines()]
        if records != allocation['events']:
            raise ValueError('S-LoRA retained allocation journal differs from qualification')
        validate_slora_allocation(allocation, row, cfg, i, ready['clock_id'], o['worker_completed_monotonic_s'])
        receipts[rid] = dict(path=str(path), sha256=digest(path))
        seen_indices.add(i)
    if seen_indices != set(range(len(cfg['gpu_ids']))) or len(receipts) != len(cfg['gpu_ids']):
        raise ValueError('S-LoRA qualification did not bind every declared worker exactly once')
    return list(receipts.values())


def validate_loquetier_ready_row(row, cfg, control, service, clock_id, observation, plan):
    """Actual static destination and native forward binding, not ordinal RR."""
    aid = observation.get('native_adapter_id')
    shards = [s for s in plan['shards'] if aid in s['adapter_ids']]
    if len(shards) != 1:
        raise ValueError('Loquetier native adapter has no unique static shard')
    index = shards[0]['index']
    identity = row.get('identity', {})
    process = dict(pid=identity.get('pid'), start_ticks=identity.get('start_ticks'), clock_id=clock_id)
    at, notice, entry = row.get('at'), cfg.get('deployment_notice_s'), control.get('tc_backend_entry_s')
    if (row.get('schema') != 'ieee_tc_loquetier_backend_ready_v1'
            or row.get('index') != index or type(control.get('tc_static_shard_index')) is not int
            or control.get('tc_static_shard_index') != index
            or 'tc_route_ordinal' in control or control.get('instance_id') != f'loquetier-{index}'
            or row.get('clock_id') != clock_id or control.get('tc_clock_id') != clock_id
            or identity.get('cgroup') != service['cgroup'] or identity.get('affinity') != service['affinity']
            or any(type(identity.get(k)) is not int or identity[k] <= 0 for k in ('pid', 'start_ticks'))
            or observation.get('native_process') != process or control.get('tc_worker_native_process') != process
            or observation.get('timing_contract') != 'ieee_tc_loquetier_native_v1'
            or observation.get('native_adapter_path') != cfg['adapters'][aid][1]
            or type(observation.get('native_adapter_index')) is not int
            or observation['native_adapter_index'] != shards[0]['adapter_ids'].index(aid)
            or observation.get('native_adapter_binding_source') !=
                'loquetier_actual_forward_segments_and_startup_pointer_tables'
            or observation.get('native_terminal_observed') is not True
            or type(observation.get('native_forward_count')) is not int
            or observation['native_forward_count'] != observation.get('native_output_tokens')
            or control.get('tc_worker_ready_s') != at or control.get('tc_deployment_notice_s') != notice
            or any(type(t) not in (int, float) or not math.isfinite(t) for t in (notice, at, entry))
            or not notice <= at <= entry):
        raise ValueError('Loquetier static destination/native forward/process/notice differs')
    return index


def audit_loquetier_receipts(root, cfg, qualifier, ready, config_path, native):
    from replay_vllm_engine_trace import (validate_loquetier_service_config,
        verify_loquetier_source_view, load_loquetier_static_plan, validate_loquetier_ready_receipt)
    validate_loquetier_service_config(cfg, json.loads(config_path.read_text()))
    verify_loquetier_source_view(cfg)
    plan = load_loquetier_static_plan(cfg, cfg['adapters'])
    if (qualifier['schema'] != 'ieee_tc_loquetier_native_model_qualification_v1'
            or qualifier['physical_lifecycle']['mode'] != 'loquetier_static_shards_v1'
            or qualifier['physical_lifecycle']['measurement_complete'] is not True
            or cfg['source_sha256'] != qualifier['source_sha256']
            or any(cfg.get(k) != v for k, v in qualifier['configuration'].items())
            or len(qualifier['physical_allocations']) != len(cfg['gpu_ids'])
            or len({a['lease_id'] for a in qualifier['physical_allocations']}) != len(cfg['gpu_ids'])
            or len({g for a in qualifier['physical_allocations'] for g in a['gpu_uuids']}) != len(cfg['gpu_ids'])
            or any(p != str(Path(cfg['remote_config']['cache_dir'])/'objects'/aid)
                   for aid, (_, p) in cfg['adapters'].items())):
        raise ValueError('explicit Loquetier source/static pool/physical lifecycle differs')
    for path, sha in cfg['source_sha256'].items():
        if digest(path) != sha:
            raise ValueError('executed Loquetier adapter source changed')
    worker_path = root/'model7b/worker_config.json'
    preparation_sha = digest(root/'model7b/loquetier_full_pool_preparation.json')
    receipts, seen_indices = {}, set()
    for request in qualifier['requests']:
        o = request['response']['metrics']['ieee_tc']
        c, rid = o['control_observation'], o['control_observation']['tc_worker_receipt_id']
        if not isinstance(rid, str) or len(rid) != 32 or set(rid)-set('0123456789abcdef'):
            raise ValueError('invalid Loquetier receipt ID')
        path = root/'model7b/worker_receipts'/(rid+'.json')
        row = json.loads(path.read_text())
        if (path.is_symlink() or digest(path) != c['tc_worker_receipt_sha256']
                or row['receipt_id'] != rid or row['worker_config_sha256'] != digest(worker_path)
                or o.get('native_request_id') != request['request_id']):
            raise ValueError('Loquetier receipt/worker configuration/request differs')
        i = validate_loquetier_ready_row(row, cfg, c, qualifier['service'], ready['clock_id'], o, plan)
        validate_loquetier_ready_receipt(row, cfg=cfg, index=i, plan=plan, identity=row['identity'],
                                        preparation_sha256=preparation_sha)
        allocation = qualifier['physical_allocations'][i]
        records = [json.loads(line) for line in Path(allocation['journal']).read_text().splitlines()]
        if records != allocation['events']:
            raise ValueError('Loquetier physical journal differs from native qualification')
        validate_slora_allocation(allocation, row, cfg, i, ready['clock_id'],
                                  o['worker_completed_monotonic_s'], backend='loquetier')
        receipts[rid] = dict(path=str(path), sha256=digest(path))
        seen_indices.add(i)
    if seen_indices != set(range(len(cfg['gpu_ids']))) or len(receipts) != len(cfg['gpu_ids']):
        raise ValueError('Loquetier qualification did not bind every declared worker exactly once')
    return list(receipts.values())


def validate_artifact_transfer(aid, control, receipt, receipt_sha, files, server, remote_clock, local_clock):
    """Shared request/preparation transfer join; bytes counted by transfer, not references."""
    from faaslora.storage.http_artifact_store import preparation_content_sha256
    evidence = receipt['evidence']
    if (receipt_sha != control['tc_artifact_receipt_sha256']
            or preparation_content_sha256(files) != receipt['content_sha256']
            or receipt['content_sha256'] != control['tc_artifact_content_sha256']
            or receipt['content_manifest_sha256'] != control['tc_artifact_manifest_sha256']
            or server['artifact_id'] != receipt['adapter_id'] or receipt['adapter_id'] != aid
            or server['outcome'] != 'sent' or server['pack_performed'] is not False
            or server['temporary_created'] is not False
            or server['delivery_mode'] != 'prepublished_gzip_v1'
            or evidence['remote_delivery_mode'] != server['delivery_mode']
            or server['clock_id'] != remote_clock
            or server['transfer_id'] != control['tc_artifact_transfer_id']
            or server['transfer_id'] != evidence['http_transfer_id']
            or server['archive_sha256'] != evidence['remote_archive_sha256']
            or not server['archive_bytes'] == server['bytes_read'] == server['bytes_written'] == evidence['transferred_bytes']
            or evidence['published_archive_verified'] is not True or evidence['content_verified'] is not True
            or evidence['remote_pack_performed'] is not False or evidence['loading_clock_id'] != local_clock):
        raise ValueError('published artifact transfer/receipt/content identity differs')


def validate_slora_preparation(preparation, pool_ids, notice, first_acquire, remote, *, backend='slora'):
    """All 500 static IDs after notice, before workers, sequential and one delivery each."""
    if backend not in ('slora', 'loquetier'):
        raise ValueError('unknown full-pool preparation contract')
    if len(pool_ids) != 500 or set(preparation) != set(pool_ids) or len(remote) != 500:
        raise ValueError('S-LoRA full-static-pool preparation is incomplete')
    previous, transfers, total = notice, set(), 0
    for aid in sorted(pool_ids):
        row = preparation[aid]
        start, end = row['tc_artifact_resolve_started_s'], row['tc_artifact_resolve_completed_s']
        transfer, size = row['tc_artifact_transfer_id'], row['tc_artifact_transferred_bytes']
        if (row['tc_artifact_id'] != aid or row['tc_artifact_request_id'] != backend+'-full-pool:'+aid
                or row['tc_artifact_cache_hit'] is not False or row['tc_artifact_delivery_mode'] != 'prepublished_gzip_v1'
                or transfer in transfers or transfer not in remote
                or remote[transfer]['artifact_id'] != aid
                or type(size) is not int or size <= 0 or size != remote[transfer]['bytes_written']
                or any(type(t) not in (int, float) or not math.isfinite(t) for t in (previous, start, end, first_acquire))
                or not previous <= start <= end <= first_acquire):
            raise ValueError('S-LoRA preparation order/notice/transfer/worker-allocation differs')
        previous = end
        transfers.add(transfer)
        total += size
    if transfers != set(remote):
        raise ValueError('S-LoRA preparation has unjoined remote transfers')
    return total


def validate_slora_preparation_reference(control, prep, recalculated):
    if (any(control.get(k) != v for k, v in prep.items())
            or recalculated['artifact_attribution'] != 'deployment_preparation_reference_not_request_transfer'
            or recalculated['artifact_transferred_bytes'] != 0
            or recalculated['preparation_transferred_bytes'] != prep['tc_artifact_transferred_bytes']):
        raise ValueError('S-LoRA request changed or double-counted deployment preparation')


def validate_slora_notice(launch, cfg, ready, entries, submitted):
    if (launch['deployment_notice_s'] != cfg['deployment_notice_s']
            or launch['replay_t0_s'] != launch['deployment_notice_s']+60.
            or launch['clock_id'] != ready['clock_id'] or launch['plan'] != ready['plan']
            or launch['config_sha256'] != ready['config_sha256']):
        raise ValueError('S-LoRA preparation and publisher have different notice/input clocks')
    for entry in entries:
        arrival = submitted[entry.request_id]['planned_arrival_s']
        if (type(arrival) not in (int, float) or not math.isfinite(arrival)
                or abs(arrival - (launch['replay_t0_s']+entry.offset_s)) > 1e-6):
            raise ValueError('S-LoRA business arrivals were shifted from common notice')


def validate_slora_resolver_journal(rows, preparation):
    completed = [r for r in rows if r['event'] == 'resolve_completed']
    if (len(completed) != 500 or any(r['event'] == 'resolve_failed' for r in rows)
            or {r['tc_artifact_id'] for r in completed} != set(preparation)
            or any(any(r.get(k) != v for k, v in preparation[r['tc_artifact_id']].items()) for r in completed)):
        raise ValueError('S-LoRA preparation differs from actual resolver journal')


def audit(root, baseline, main, scheduler_contract='original', population='prefix100', slo_freeze=None,
          outcome='success'):
    sys.path[:0] = [str(baseline/'scripts'), str(main)]
    import summarize_serverlessllm_replay as analyzer
    import prepare_ieee_tc_serverless_stack as native
    from faaslora.storage.http_artifact_store import preparation_content_sha256
    is_vllm = scheduler_contract == 'resident_rr_v1'
    is_slora = scheduler_contract == 'slora_resident_rr_v1'
    is_loquetier = scheduler_contract == 'loquetier_static_shards_v1'
    static_native = is_slora or is_loquetier
    resident = is_vllm or static_native
    backend = 'loquetier' if is_loquetier else 'slora' if is_slora else 'vllm'
    launch_root = root/('launch7b.launch' if resident else 'launch7b_repaired.launch')
    model_root = root/('model7b' if resident else 'model7b_repaired')
    replay = launch_root/'replay.jsonl'
    summary = analyzer.audit_tc_http_journal(replay)
    events = [json.loads(line) for line in replay.read_text().splitlines()]
    kinds = ('request_contract', 'request_created', 'http_headers_sent', 'http_raw_response', 'http_response')
    index = {k: {e['request_id']: e for e in events if e['event'] == k} for k in kinds}
    ready = summary['ready_identity']
    assert digest(ready['plan']['source_path']) == ready['plan']['source_sha256']
    config_path = root/{'adapter_coverage_v1': 'http_7b_500.json',
                       'prefix1000': 'http_7b_1000.json', 'prefix100': 'http_7b_100.json',
                       'full4000': 'http_7b_4000.json',
                       'full4000_w1': 'http_7b_4000.json'}[population]
    assert digest(config_path) == ready['config_sha256']
    count = validate_population(json.loads(config_path.read_text()), ready,
                                index['request_contract'], native, population)
    cfg = json.loads((model_root/('worker_config.json' if resident else 'configuration.json')).read_text())
    qualifier = json.loads((model_root/'model_qualification.json').read_text())
    terminal_physical = (json.loads((model_root/'physical/physical_resource_summary.json').read_text())
                         if resident and outcome == 'complete_with_failures' else None)
    response_count = validate_terminal_outcome(summary, qualifier, count, outcome,
                                               scheduler_contract, terminal_physical)
    if resident:
        request_index = {r['request_id']: r for r in qualifier['requests']}
        if (len(request_index) != len(qualifier['requests'])
                or set(request_index) != set(index['http_raw_response'])
                or any(request_index[rid]['response'] != raw['body']
                       for rid, raw in index['http_raw_response'].items())):
            raise ValueError('resident service and independent publisher responses differ')
        receipt_audit = audit_loquetier_receipts if is_loquetier else audit_slora_receipts if is_slora else audit_vllm_receipts
        qualifier['backend_ready_receipts'] = receipt_audit(root, cfg, qualifier, ready, config_path, native)
    else:
        assert qualifier['physical_lifecycle']['lifecycle_qualified']
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
    elif resident:
        queue_evidence = dict(contract=scheduler_contract, native_worker_receipts=qualifier['backend_ready_receipts'],
            conclusion=('Frozen input-only static shard mapping; every selected native forward/worker/ready/physical owner verified.'
                        if is_loquetier else
                        'Original request-ordinal round robin; every selected native worker/source/ready/owner binding verified.'))
    else:
        raise ValueError('unknown scheduler contract')
    frozen = cfg['remote_config'] if resident else cfg['backend_config']['tc_remote_artifacts']
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
    preparation = None
    if static_native:
        preparation = json.loads((model_root/f'{backend}_full_pool_preparation.json').read_text())
        if set(cfg['adapters']) != set(pool):
            raise ValueError('S-LoRA native registry differs from frozen full pool')
        launch = json.loads((launch_root/'exec_receipt.json').read_text())['external_replay']
        validate_slora_notice(launch, cfg, ready,
            native.load_http_replay_plan(json.loads(config_path.read_text())).entries, index['http_headers_sent'])
        first_acquire = min(e['at'] for a in qualifier['physical_allocations']
                            for e in a['events'] if e['event'] == 'acquire')
        transferred = validate_slora_preparation(preparation, set(pool), cfg['deployment_notice_s'], first_acquire, remote,
                                                backend=backend)
        if set(receipts) != set(pool):
            raise ValueError('S-LoRA preparation receipt population differs from full pool')
        for aid, prep in preparation.items():
            path, receipt = receipts[aid]
            files = {f['path']: (f['size_bytes'], f['sha256']) for f in pool[aid]['files']}
            validate_artifact_transfer(aid, prep, receipt, digest(path), files,
                remote[prep['tc_artifact_transfer_id']], health['7b']['clock_id'], ready['clock_id'])
            used_transfers.add(prep['tc_artifact_transfer_id'])
        resolver_rows = [json.loads(line) for path in (root/'remote_resolver_evidence/journals').glob('*.jsonl')
                         for line in path.read_text().splitlines()]
        validate_slora_resolver_journal(resolver_rows, preparation)
    for rid, terminal in index['http_response'].items():
        contract, raw = index['request_contract'][rid], index['http_raw_response'][rid]
        observation = raw['body']['metrics']['ieee_tc']
        ids = observation['native_prompt_token_ids']
        assert hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() == contract['native_prompt_token_ids_sha256']
        prepared_contract = dict(contract, input_token_ids=ids)
        if static_native:
            prepared_contract['deployment_notice_s'] = cfg['deployment_notice_s']
        recalculated = native.validate_http_observation(prepared_contract,
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
        if static_native:
            prep = preparation[contract['adapter_id']]
            validate_slora_preparation_reference(c, prep, recalculated)
            continue
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
    resource_path = model_root/'physical/physical_resource_summary.json'
    resource = json.loads(resource_path.read_text())
    physical = json.loads(Path(resource['summary_path']).read_text())
    assert digest(resource['summary_path']) == resource['summary_sha256']
    assert digest(resource['result_file']['path']) == resource['result_file']['sha256']
    assert digest(resource['request_terminals_path']) == resource['request_terminals_sha256']
    for row in resource['allocation_journals']:
        assert digest(row['path']) == row['sha256'] and row['released']
    if static_native and ({r['path'] for r in resource['allocation_journals']}
                     != {a['journal'] for a in qualifier['physical_allocations']}):
        raise ValueError('S-LoRA physical summary and native owners have different journals')
    assert physical['measurement_complete'] and not physical['open_lease_ids']
    integral = sum(b-a for intervals in physical['device_intervals'].values() for a, b in intervals)
    assert abs(integral-physical['gpu_seconds']) < 1e-9
    assert abs(sum(physical['window_gpu_seconds'].values())-integral) < 1e-9
    assert physical['n_native_contract_complete'] == response_count
    assert physical['n_terminal'] == physical['n_plan'] == count
    samples = [e for e in map(json.loads, (launch_root/'watchdog.jsonl').read_text().splitlines()) if e['event'] == 'resource_sample']
    assert samples
    service_events = {k: max(s['service'].get('memory.events', {}).get(k, 0) for s in samples) for k in ('high', 'max', 'oom', 'oom_kill')}
    source_paths = [replay, resource_path, Path(resource['summary_path']), Path(resource['result_file']['path']),
                    config_path, root/'remote_transfers.jsonl', root/'remote_monitor.log',
                    Path(analyzer.__file__), Path(native.__file__), Path(__file__)]
    source_paths += ([model_root/'worker_config.json', model_root/'native_toolchain.json'] +
                     [Path(r['path']) for r in qualifier['backend_ready_receipts']] if resident else
                    [root/'source_repaired/measurement_manifest.json', root/'storage_aware_scheduler_executed.py'])
    if static_native:
        source_paths += [model_root/f'{backend}_full_pool_preparation.json', Path(cfg['source_view'])/'measurement_manifest.json',
                         launch_root/'exec_receipt.json']
        source_paths += sorted((root/'remote_resolver_evidence/journals').glob('*.jsonl'))
    if is_loquetier:
        source_paths += [Path(cfg['static_plan']), Path(cfg['native_kernel'])]
        source_paths += sorted(model_root.glob('worker_*.native_events.jsonl'))
    result = {k: v for k, v in summary.items() if k not in ('ready_identity', 'diagnostic_rows', 'assignment_gaps_s', 'kind', 'display_name')}
    result.update(schema=('ieee_tc_complete_request_failure_audit_v1' if outcome == 'complete_with_failures' else
                          'ieee_tc_d402_affected_path_audit_v1' if scheduler_contract == 'original'
                          else 'ieee_tc_affected_path_audit_v2'), display_name='Loquetier' if is_loquetier else 'S-LoRA' if is_slora else 'vLLM' if is_vllm else 'ServerlessLLM',
        remote_affected_path_qualified=outcome == 'success', complete_system_qualified=False,
        source_plan=ready['plan'], native_revalidated_requests=response_count, max_native_recompute_error_ms=native_error,
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
                      f"Complete 4000-request development {ready['plan']['profile']} observation; not unseen evaluation or a cross-scenario configuration optimum."
                      if population in ('full4000', 'full4000_w1') else
                      f'{count}-request development prefix, not a full W0 comparison or configuration optimum.'),
            'Raw unknown numerical correctness and old qualification flags are preserved.',
            ('Four logical instances include starting instances; three served engines. Scheduler queue bug independently reproduced; live exception was not logged.'
             if scheduler_contract == 'original' else
             'Actual backend-ready receipts and live loop qualify the affected path; qualification does not quantify a causal latency benefit.'),
            ('Loquetier uses disclosed four-way static-sharded deployment, not an official adaptive cluster scheduler. Full500 preparation follows common notice; native SMLM and static125-ID workers retained. No baseline optimization or PrimeLoRA policy added.' if is_loquetier else
             'S-LoRA native full-static-pool preparation follows common notice; actual resident GPU leases include HOST load and end only after native exit. No request-time fetch or PrimeLoRA policy added.' if is_slora else
             'Resident native vLLM four-GPU pool is charged until actual release; no PrimeLoRA policy added.' if is_vllm else
             'Fixed native-store four-GPU pool is charged until actual release; not a claim about every ServerlessLLM deployment.'),
            ('Cross-system comparison requires a separate full input, generation, initial-state and resource-contract join.'
             if population in ('full4000', 'full4000_w1') else
             'No relative performance comparison with the 4000-request PrimeLoRA observation.')])
    if static_native:
        result.update(artifact_attribution='deployment_preparation_reference_not_request_transfer',
            deployment_prepared_adapters=len(preparation), deployment_remote_wire_bytes=transferred,
            request_remote_wire_bytes=0, cache_hit_requests=None,
            preparation_references=response_count,
            served_adapter_ids=len({index['request_contract'][rid]['adapter_id'] for rid in index['http_response']}))
    if outcome == 'complete_with_failures':
        result.update(outcome_contract=outcome, performance_ranking_eligible=False,
            remote_successful_response_paths_qualified=True,
            terminal_failures=[r for r in summary['diagnostic_rows'] if r['status'] != 'http_response'])
        result['limitations'].append('All offered requests reached a terminal, but request failures remain; conditional latency and total resources do not constitute a feasible G1/G2 point. Failure attribution is separate.')
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
        observations = {rid: index['http_raw_response'][rid]['body']['metrics']['ieee_tc']
                        for rid in index['http_response']}
        rows = summary['diagnostic_rows']
        result['common_slo'] = frozen_slo_summary(rows, observations, frozen_slo['groups'])
        result['common_slo'].update(freeze_path=str(slo_freeze), freeze_sha256=digest(slo_freeze),
            scope=(f"complete {ready['plan']['profile']} native-binding observation; not a cross-scenario optimum or independent numerical proof"
                   if population in ('full4000', 'full4000_w1') else
                   'complete qualified population only; prefix or coverage is not full W0 qualification'),
            physical_gpu_seconds_per_binding_valid_request=(integral/response_count if response_count else None))
        if outcome == 'complete_with_failures':
            result['common_slo']['scope'] = f"Complete offered {ready['plan']['profile']} failure diagnostic; failures retained, not an eligible G1/G2 point or independent numerical proof"
            result['common_slo']['physical_gpu_seconds_per_offered_request'] = integral/count
        result['conditional_metrics']['tpot_ms'] = timing_stats(
            [r['tpot_ms'] for r in rows if r['status'] == 'http_response' and r['actual_tokens'] > 1])
        if population in ('prefix1000', 'full4000', 'full4000_w1'):
            # Exhaustive, nonoverlapping source-order slices; diagnostic only.
            # Never drop the startup portion from the primary SLO denominator.
            ordered = sorted(rows, key=lambda r: r['planned_arrival_s'])
            result['source_order_diagnostics'] = []
            slices = ([(lo, lo+500) for lo in range(0, 4000, 500)]
                      if population in ('full4000', 'full4000_w1') else [(0, 100), (100, 500), (500, 1000)])
            for lo, hi in slices:
                subset = ordered[lo:hi]
                result['source_order_diagnostics'].append(dict(
                    source_indices_inclusive=[lo, hi-1],
                    common_slo=frozen_slo_summary(subset, observations, frozen_slo['groups']),
                    metrics={k: timing_stats([r[k] for r in subset if r.get(k) is not None])
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
    parser.add_argument('--scheduler-contract', choices=('original', 'request_identity_v1', 'resident_rr_v1', 'slora_resident_rr_v1', 'loquetier_static_shards_v1'), default='original')
    parser.add_argument('--population', choices=('prefix100', 'prefix1000', 'adapter_coverage_v1', 'full4000', 'full4000_w1'), default='prefix100')
    parser.add_argument('--reference-result', type=Path)
    parser.add_argument('--slo-freeze', type=Path)
    parser.add_argument('--outcome', choices=('success', 'complete_with_failures'), default='success',
                        help='Explicit diagnostic only; never bypass success qualification or hide failed offered rows.')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args.root, args.baseline, args.main, args.scheduler_contract, args.population, args.slo_freeze,
                   args.outcome)
    if args.reference_result:
        if args.population != 'prefix100':
            raise ValueError('independent reference is the unchanged first 100 requests only')
        report['bounded_numerical_reference'] = reuse_numerical_reference(args.root, args.reference_result, args.main,
                                                                          args.scheduler_contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in
        ('raw_sources', 'conditional_metrics', 'source_plan', 'bounded_numerical_reference')}, indent=2))
