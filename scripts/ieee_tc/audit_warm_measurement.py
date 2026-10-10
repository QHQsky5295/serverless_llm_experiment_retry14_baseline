"""Reuse D170's streaming projection and native validators; never launch GPU work."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import ieee_tc_preflight as p


def audit(raw_dir, projection_filter, output):
    """D170 curation checks, with explicit input/output and no old-file overwrite."""
    if output.exists():
        raise FileExistsError(output)
    archive = raw_dir / '7b_measure1.json.gz'
    projection = raw_dir / 'measurement_projection1.json'
    # The D170 filter removes repeated batch tensor inventories only.
    if not projection.exists():
        with projection.open('xb') as target:
            inflate = subprocess.Popen(['gzip', '-dc', str(archive)], stdout=subprocess.PIPE)
            try:
                project = subprocess.run(['jq', '--stream', '-n', '-f', str(projection_filter)],
                                         stdin=inflate.stdout, stdout=target, check=False)
                inflate.stdout.close()
                inflate_status = inflate.wait()
                if project.returncode or inflate_status:
                    raise RuntimeError('stream projection failed; partial evidence retained')
            finally:
                if inflate.poll() is None:
                    inflate.terminate()
                    inflate.wait()
    if projection.stat().st_size > 128 * 1024**2:
        raise ValueError('projection exceeds existing D170 analysis bound')
    data = json.loads(projection.read_text())
    launch = json.loads((raw_dir / '7b_measure1_launch.json').read_text())
    index_path = ROOT / 'paper_results/ieee_tc/p2_backend/20261002_d168_warm_input_index.json'
    index = json.loads(index_path.read_text())
    assert p.digest(index_path) == data['warm_reference_index_sha256']
    assert data['warm_phase'] == 'measure' and data['batch_size'] == 8
    assert launch['pass'] and data['pass'] and data['stage'] == 'complete'
    assert launch['service_returncode'] == launch['watchdog_returncode'] == 0
    assert launch['cleanup']['released'] and data['shutdown_called']
    samples = p.completed_warm_length_samples(data, index)
    assert len(samples) == 512 and len(data['requests']) == 1552
    expected = list(p.warm_reference_batches(index, 'measure'))
    assert len(expected) == len(data['warm_batches']) == 194
    assert sorted(i for b in data['warm_batches'] for i in b['request_indices']) == list(range(1552))
    for serial, (batch, spec) in enumerate(zip(data['warm_batches'], expected)):
        cases = [data['requests'][i] for i in batch['request_indices']]
        assert len(cases) == len(spec['selected']) == 8
        assert all(c['batch_serial'] == serial for c in cases)
        assert batch['sources_before']['complete_for_native_caches']
        assert all(c['reference']['adapter_int_id'] in batch['sources_before']['slot_adapter_ids']
                   for c in cases)
        before, after = batch['scheduler_before'], batch['scheduler_after']
        assert all(not s[k] for s in (before, after) for k in
                   ('admitted', 'unretired_iterations', 'native_deferred_free_batches'))
        needed = sum(math.ceil((c['native_prompt_tokens'] + c['target_tokens']) /
                               before['kv_tokens_per_block']) for c in cases)
        assert needed == batch['full_declared_kv_blocks'] <= before['kv_unreserved_free_blocks']
    assert not data['sources_after_eviction']['sources']
    assert all(v is None for v in data['sources_after_eviction']['slot_adapter_ids'])
    rounds = []
    for gid in (0, 1):
        for rid in range(3):
            cases = [c for c in data['requests'] if c['phase'] == 'measurement'
                     and c['group_id'] == gid and c['round'] == rid]
            assert len(cases) == 256
            rounds.append(dict(group_id=gid, round=rid, count=len(cases),
                               ttft_ms=statistics.fmean(c['ttft_ms'] for c in cases),
                               tpot_ms=statistics.fmean(c['tpot_ms'] for c in cases)))
    watch = raw_dir / '7b_measure1_launch.launch/watchdog.jsonl'
    event_counts, resource = {}, []
    for line in watch.open():
        event = json.loads(line)
        event_counts[event['event']] = event_counts.get(event['event'], 0) + 1
        if event['event'] == 'resource_sample':
            resource.append(event)
    physical = data['physical_allocation']
    acquire = [e for e in physical['events'] if e['event'] == 'acquire']
    release = [e for e in physical['events'] if e['event'] == 'release']
    assert physical['released'] and len(acquire) == len(release) == len(physical['gpu_uuids']) == 1
    assert release[0]['worker_returncode'] == 0
    census = release[0]['census']
    assert census['service_native_contexts_clear']
    assert not any(census[k] for k in ('service_held_gpu_uuids', 'unresolved_processes',
                                      'escaped_owned_processes', 'foreign_compute_processes'))
    assert acquire[0]['clock_id'] == release[0]['clock_id']
    seal = p.verify_seal(ROOT / 'paper_results/ieee_tc/safety/20260925_execution_start_protected.json')
    known_change = Path('/home/qhq/serverless_llm_baselines/scripts/replay_openai_trace.py')
    known_change_sha = '04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22'
    protected_scope_ok = (seal['entry_count'] == 147
                          and seal['changed'] == [str(known_change)]
                          and p.digest(known_change) == known_change_sha)
    result = dict(kind='warm_reference_independent_audit_v1', passed=True,
        projection_sha256=p.digest(projection), projection_filter_sha256=p.digest(projection_filter),
        source_archive_sha256=p.digest(archive), launch_sha256=p.digest(raw_dir/'7b_measure1_launch.json'),
        source_plan_sha256=data['plan_sha256'], metric_protocol_sha256=p.digest(
            ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md'),
        native_checked=1552, measured=1536, warmup=16, unique_measured_source_requests=len(samples),
        batch_checked=194, round_means=rounds, watchdog_events=event_counts,
        physical_gpu_s=release[0]['at']-acquire[0]['at'],
        census_alone_proves_lease_release=census['proves_physical_lease_release'],
        release_journal_verified=True,
        resource=dict(samples=len(resource),
            peak_bytes=max(e['service']['memory.peak'] for e in resource),
            host_min_available_bytes=min(e['host']['available_bytes'] for e in resource),
            memory_events={k:max(e['service']['memory.events'][k] for e in resource)
                           for k in resource[0]['service']['memory.events']},
            max_swap_bytes=max(e['service']['memory.swap.current'] for e in resource),
            warnings=sum(bool(e['decision']['warning']) for e in resource),
            aborts=[e['decision']['abort_reasons'] for e in resource if e['decision']['abort_reasons']]),
        protected_manifest=seal, protected_scope_ok=protected_scope_ok,
        known_protected_change={"path": str(known_change), "sha256": known_change_sha},
        common_reference_frozen=False, formal_g1_g2_qualified=False)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--projection-filter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.raw_dir, args.projection_filter, args.output)
