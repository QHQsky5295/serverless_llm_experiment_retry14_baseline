"""CPU-only closure for D1089 NoHOST W1; serving is not rerun here."""
import importlib.util
import json
from pathlib import Path
import sys
import yaml

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
BASE = Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1089_20261107')
RUN = BASE / 'd1089_3b_nohost_w1'
OUT = Path(__file__).resolve().parent
DONE = OUT / 'completion'
CONFIG = ROOT / 'configs/ieee_tc/20261105_d1063_3b_nohost_w1.yaml'
PROFILE = ROOT / 'paper_results/ieee_tc/p2_backend/d1031_3b_startup_memory_profile/export/manifest.json'
HELPER = ROOT / 'paper_results/ieee_tc/p2_backend/d1081_positive_demand_options/collect_completion.py'
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location('d1081_collection', HELPER)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
from scripts.ieee_tc.audit_d703_3b_full import (  # noqa: E402
    audit_closed_outcome,
    audit_profile_bindings,
    diagnose_common_slo,
    reevaluate_common_slo,
)
from scripts.ieee_tc.export_d701_3b_profile import matched_remote_transfers  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def ref(path):
    return {'path': str(path), 'sha256': old.p.digest(path), 'bytes': path.stat().st_size}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    DONE.mkdir(parents=True, exist_ok=True)
    result = next(RUN.glob('experiment_results_full*.json'))
    launch = read(RUN / 'launch.json')
    assert all(launch.get(k) is True for k in (
        'pass', 'service_path_removed', 'native_gpu_context_release_confirmed'))
    assert all(launch.get(k) == 0 for k in (
        'service_returncode', 'watchdog_returncode', 'replay_returncode'))

    if not (DONE / 'full4000.json').exists():
        reevaluate_common_slo(
            RUN, DONE / 'full4000.json', expected_count=4000,
            scenario='tc_ieee_full')
    binding_path = DONE / 'analysis_output_binding.yaml'
    if not binding_path.exists():
        binding = yaml.safe_load(CONFIG.read_text())
        binding['experiment']['output_dir'] = str(RUN)
        with binding_path.open('x') as stream:
            yaml.safe_dump(binding, stream, sort_keys=False)
    if not (DONE / 'profile_bindings.json').exists():
        audit_profile_bindings(
            RUN, DONE / 'profile_bindings.json', DONE / 'full4000.json',
            binding_path, PROFILE, scenario='tc_ieee_full')
    if not (DONE / 'stages/diagnosis.json').exists():
        diagnose_common_slo(
            RUN, DONE / 'stages', DONE / 'full4000.json',
            scenario='tc_ieee_full')

    outcome = RUN / 'launch.launch/physical_deployment/main_outcome.json'
    with outcome.open('rb') as stream:
        transfers = next(old.ijson.items(stream, 'remote_transfers', use_float=True))
    remote_health = read(RUN / 'remote_health_post.json')
    clock_id = remote_health.get('clock_id') or remote_health.get('3b', {}).get('clock_id')
    assert clock_id
    journal = (RUN / 'remote_transfers_remote.jsonl').read_text()
    matched = list(matched_remote_transfers(
        {'remote_transfers': transfers}, journal, clock_id))
    assert len(matched) == len(transfers) == 132
    assert all(row.get('outcome') == 'sent' for row in matched)
    assert all(row.get('delivery_mode') == 'prepublished_gzip_v1' for row in matched)
    assert all(row.get('pack_performed') is False for row in matched)
    local_journal = RUN / 'remote_transfers.jsonl'
    if not local_journal.exists():
        with local_journal.open('x') as stream:
            for row in matched:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
    if not (DONE / 'delivery_and_resources.json').exists():
        audit_closed_outcome(
            RUN, DONE / 'delivery_and_resources.json', DONE / 'full4000.json',
            binding_path, 'existing_unmonitored', scenario='tc_ieee_full')

    metric = read(DONE / 'full4000.json')
    physical = read(RUN / 'launch.launch/physical_deployment/physical_resource_summary.json')
    physical_u = physical['physical_gpu_seconds_observed']
    closure = {
        'passed': True,
        'run_id': 'd1089_20261107',
        'run_tag': 'd1089_3b_nohost_w1',
        'scenario': 'tc_ieee_full',
        'variant': 'NoHOST',
        'requests_planned': 4000,
        'requests_arrived': 4000,
        'requests_successful': 4000,
        'requests_failed': 0,
        'joint_slo_pass': metric['counts']['joint_pass'],
        'joint_slo_fraction': metric['joint_slo_fraction'],
        'physical_gpu_seconds': physical_u,
        'physical_measurement_complete': physical['measurement_complete'],
        'numerical_adapter_correctness_qualified': metric['numerical_adapter_correctness_qualified'],
        'ranking_eligible': metric['ranking_eligible'],
        'remote_transfer_count': len(matched),
        'remote_transfer_outcomes': {'sent': len(matched)},
        'remote_transfer_bytes_read': sum(row['bytes_read'] for row in matched),
        'request_time_packing': False,
        'remote_service_closed_after_collection': True,
        'source_refs': [ref(result), ref(RUN / 'launch.json'), ref(outcome),
                        ref(RUN / 'remote_transfers_remote.jsonl'), ref(RUN / 'remote_health_post.json'),
                        ref(DONE / 'full4000.json'), ref(DONE / 'delivery_and_resources.json'),
                        ref(binding_path), ref(Path(__file__))],
    }
    save(DONE / 'closure.json', closure)
    save(DONE / 'manifest.json', {
        'files': [ref(path) for path in sorted(DONE.rglob('*')) if path.is_file()],
        'raw_run': str(RUN),
        'raw_result_sha256': old.p.digest(result),
    })
    print(json.dumps({
        'passed': True,
        'variant': 'NoHOST',
        'joint_pass': metric['counts']['joint_pass'],
        'joint_fraction': metric['joint_slo_fraction'],
        'physical_gpu_seconds': physical_u,
        'remote_transfers': len(matched),
        'remote_bytes_read': sum(row['bytes_read'] for row in matched),
    }, indent=2))


if __name__ == '__main__':
    main()
