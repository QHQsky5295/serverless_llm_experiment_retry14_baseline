#!/usr/bin/env python3
"""D402 evidence recipe; reuses the existing baseline/native validators.

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


def audit(root, baseline, main):
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
    assert digest(root/'http_7b_100.json') == ready['config_sha256']
    assert summary['counts']['N_response'] == summary['counts']['N_plan'] == 100
    cfg = json.loads((root/'model7b_repaired/configuration.json').read_text())
    qualifier = json.loads((root/'model7b_repaired/model_qualification.json').read_text())
    assert qualifier['passed'] and qualifier['physical_lifecycle']['lifecycle_qualified']
    native.audit_backend_receipts(qualifier, cfg['backend_config']['tc_worker_audit'], ready['clock_id'])
    frozen = cfg['backend_config']['tc_remote_artifacts']
    assert digest(frozen['content_manifest']) == frozen['content_manifest_file_sha256']
    pool = {a['id']: a for a in json.loads(Path(frozen['content_manifest']).read_text())['artifacts']}
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
    assert physical['n_native_contract_complete'] == physical['n_terminal'] == physical['n_plan'] == 100
    samples = [e for e in map(json.loads, (root/'launch7b_repaired.launch/watchdog.jsonl').read_text().splitlines()) if e['event'] == 'resource_sample']
    assert samples
    service_events = {k: max(s['service'].get('memory.events', {}).get(k, 0) for s in samples) for k in ('high', 'max', 'oom', 'oom_kill')}
    source_paths = [replay, resource_path, Path(resource['summary_path']), Path(resource['result_file']['path']),
                    root/'http_7b_100.json', root/'remote_transfers.jsonl', root/'remote_monitor.log',
                    root/'source_repaired/measurement_manifest.json', root/'storage_aware_scheduler_executed.py',
                    Path(analyzer.__file__), Path(native.__file__), Path(__file__)]
    result = {k: v for k, v in summary.items() if k not in ('ready_identity', 'diagnostic_rows', 'assignment_gaps_s', 'kind', 'display_name')}
    result.update(schema='ieee_tc_d402_affected_path_audit_v1', display_name='ServerlessLLM',
        remote_affected_path_qualified=True, complete_system_qualified=False,
        source_plan=ready['plan'], native_revalidated_requests=100, max_native_recompute_error_ms=native_error,
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
        queue_source_reproduction=asyncio.run(reproduce_queue(root/'storage_aware_scheduler_executed.py')),
        raw_sources=[dict(path=str(p), sha256=digest(p)) for p in source_paths],
        limitations=['100-request development prefix, not a full W0 comparison or configuration optimum.',
            'Raw unknown numerical correctness and old qualification flags are preserved.',
            'Four logical instances include starting instances; three served engines. Scheduler queue bug independently reproduced; live exception was not logged.',
            'Fixed native-store four-GPU pool is charged until actual release; not a claim about every ServerlessLLM deployment.',
            'No relative performance comparison with the 4000-request PrimeLoRA observation.'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--main', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args.root, args.baseline, args.main)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('raw_sources', 'conditional_metrics', 'source_plan')}, indent=2))
