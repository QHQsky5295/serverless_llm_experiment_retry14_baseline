"""Close D1112 using the existing D703/D701 analyzers, without replaying serving.

Adapted from D1081/D1084 collection. The remote process has already stopped:
derive its clock from the retained journal, never manufacture a health response.
"""
import collections
import json
from pathlib import Path
import sys

import ijson
import yaml

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
RUN = Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1112_20261107/3b_same_version_w2')
DONE = Path(__file__).resolve().parent / 'completion'
REF = ROOT / 'paper_results/ieee_tc/p2_backend/d1084_same_version_w2/completion'
CONFIG = ROOT / 'configs/ieee_tc/20261104_d1047_3b_event_busy_time_w1.yaml'
sys.path.insert(0, str(ROOT))
from scripts.ieee_tc.audit_d703_3b_full import (  # noqa: E402
    audit_profile_bindings, compare_stage_audits, digest, read, ref)
from scripts.ieee_tc.export_d701_3b_profile import matched_remote_transfers  # noqa: E402


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    metric = read(DONE / 'full4000.json')
    assert metric['passed'] and metric['source_run'] == str(RUN)
    assert metric['counts']['n'] == 4000
    for source in metric['source_refs']:
        assert digest(Path(source['path'])) == source['sha256']
    launch = read(RUN / 'launch.json')
    assert all(launch[k] is True for k in (
        'pass', 'service_path_removed', 'native_gpu_context_release_confirmed'))
    assert all(launch[k] == 0 for k in (
        'service_returncode', 'watchdog_returncode', 'replay_returncode'))
    assert digest(CONFIG) == '263bdcd6ff255108ddaffdd9ce46898d7f12be15948d7fe3d7ac27d5ea8828d2'
    binding = DONE / 'analysis_output_binding.json'
    if not binding.exists():
        config = yaml.safe_load(CONFIG.read_text())
        config['experiment']['output_dir'] = str(RUN)
        save(binding, config)
        save(DONE / 'binding_provenance.json', dict(
            original_config=ref(CONFIG), launch=ref(RUN / 'launch.json'),
            changed_fields={'experiment.output_dir': str(RUN)},
            purpose='Analysis-only binding of the recorded output override.'))
    if not (DONE / 'profile_bindings.json').exists():
        audit_profile_bindings(RUN, DONE / 'profile_bindings.json',
            DONE / 'full4000.json', binding,
            ROOT / 'paper_results/ieee_tc/p2_backend/d1031_3b_startup_memory_profile/export/manifest.json')

    outcome = RUN / 'launch.launch/physical_deployment/main_outcome.json'
    journal = RUN / 'remote_transfers_remote.jsonl'
    with outcome.open('rb') as stream:
        transfers = next(ijson.items(stream, 'remote_transfers', use_float=True))
    server = [json.loads(line) for line in journal.read_text().splitlines()]
    clocks = {row['clock_id'] for row in server}
    assert len(clocks) == 1
    clock = next(iter(clocks))
    matched = matched_remote_transfers(
        {'remote_transfers': transfers}, journal.read_text(), clock)
    assert len(matched) == len(server) == len(transfers) == 503
    assert all(r['state'] == 'published' and r['content_verified'] is True
               and r['published_archive_verified'] is True
               and r['remote_pack_performed'] is False for r in transfers)
    watchdog = RUN / 'launch.launch/watchdog.jsonl'
    events, memory = collections.Counter(), collections.Counter()
    peak = swap = foreign = escaped = warnings = 0
    available, free = float('inf'), float('inf')
    with watchdog.open() as stream:
        for line in stream:
            row = json.loads(line)
            events[row['event']] += 1
            if row['event'] != 'resource_sample':
                continue
            assert not row['decision']['abort_reasons']
            warnings += bool(row['decision']['warning'])
            for key, value in row['service']['memory.events'].items():
                memory[key] = max(memory[key], value)
            peak = max(peak, row['service']['memory.peak'])
            swap = max(swap, row['service']['memory.swap.current'])
            foreign += len(row['gpu']['foreign_compute_processes'])
            escaped += len(row['gpu']['escaped_owned_processes'])
            available = min(available, row['host']['available_bytes'])
            free = min(free, *(f['free_bytes'] for f in row['filesystems']))
    assert events['resource_sample'] > 0
    assert events['gpu_terminal_census'] == events['service_domain_gone'] == 1
    physical_path = RUN / 'launch.launch/physical_deployment/physical_resource_summary.json'
    physical = read(physical_path)
    assert physical['measurement_complete'] is True
    assert physical['physical_gpu_seconds_observed'] == metric['physical_gpu_seconds']
    closure_path = DONE / 'delivery_and_resources.json'
    if not closure_path.exists():
        save(closure_path, dict(
            kind='d1112_retained_journal_resource_closure_v1', passed=True,
            remote=dict(transfers=len(matched),
                wire_bytes=sum(r['transferred_bytes'] for r in transfers),
                clock_id=clock, clock_source='retained_remote_journal',
                independent_health_response_saved=False,
                all_client_server_ids_bytes_hashes_match=True,
                request_time_packing=False,
                continuous_remote_interference_monitor=False),
            resources=dict(events=dict(events), memory_events_max=dict(memory),
                service_peak_bytes=peak, max_swap_bytes=swap,
                min_host_available_bytes=available, min_free_disk_bytes=free,
                foreign_compute_samples=foreign, escaped_owned_samples=escaped,
                warning_samples=warnings),
            physical_gpu_seconds=metric['physical_gpu_seconds'],
            n_correct=None, ranking_eligible=False,
            source_refs=[ref(p) for p in (outcome, journal, watchdog, physical_path,
                RUN / 'launch.json', DONE / 'full4000.json', Path(__file__))]))
    if not (DONE / 'versus_d1084').exists():
        compare_stage_audits(REF / 'stages', DONE / 'stages', DONE / 'versus_d1084')
    if not (DONE / 'manifest.json').exists():
        save(DONE / 'manifest.json', dict(
            files=[ref(p) for p in sorted(DONE.rglob('*')) if p.is_file()],
            collector=ref(Path(__file__)),
            note='Valid measurement is not SLO feasibility or independent numerical correctness.'))
    print(json.dumps(dict(joint_pass=metric['counts']['joint_pass'],
        joint_fraction=metric['joint_slo_fraction'], physical_gpu_seconds=metric['physical_gpu_seconds'],
        remote_transfers=len(matched))))


if __name__ == '__main__':
    main()
