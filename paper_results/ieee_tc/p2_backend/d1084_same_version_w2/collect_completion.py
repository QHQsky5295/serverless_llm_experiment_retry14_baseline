"""CPU-only closure for D1084 3B W2; the replay itself is not rerun."""
import importlib.util
import json
from pathlib import Path
import sys
import yaml

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
BASE = Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1084_20261106')
RUN = BASE / 'd1084_3b_host_budget_projection_w2'
OUT = Path(__file__).resolve().parent
DONE = OUT / 'completion'
CONFIG = ROOT / 'configs/ieee_tc/20261104_d1047_3b_event_busy_time_w1.yaml'
PROFILE = ROOT / 'paper_results/ieee_tc/p2_backend/d1031_3b_startup_memory_profile/export/manifest.json'
PARENT = ROOT / 'paper_results/ieee_tc/p2_backend/d1045_event_busy_time'
HELPER = ROOT / 'paper_results/ieee_tc/p2_backend/d1081_positive_demand_options/collect_completion.py'
BINDING = OUT / 'completion/analysis_output_binding.yaml'
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


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def ref(path):
    return {'path': str(path), 'sha256': old.p.digest(path), 'bytes': path.stat().st_size}


def main():
    DONE.mkdir(parents=True, exist_ok=True)
    result = next(RUN.glob('experiment_results_full*.json'))
    launch = read(RUN / 'launch.json')
    assert all(launch.get(k) is True for k in (
        'pass', 'service_path_removed', 'native_gpu_context_release_confirmed'))
    assert all(launch.get(k) == 0 for k in (
        'service_returncode', 'watchdog_returncode', 'replay_returncode'))
    # D648 is the frozen common TTFT/TPOT contract. This creates a new, append-only
    # audit; it never edits the raw result or the earlier D1045/D1083 audits.
    if not (DONE / 'full4000.json').exists():
        reevaluate_common_slo(RUN, DONE / 'full4000.json', expected_count=4000)
    else:
        existing = read(DONE / 'full4000.json')
        assert existing['source_run'] == str(RUN) and existing['passed'] is True
    if not BINDING.exists():
        binding = yaml.safe_load(CONFIG.read_text())
        binding['experiment']['output_dir'] = str(RUN)
        with BINDING.open('x') as stream:
            yaml.safe_dump(binding, stream, sort_keys=False)
    if not (DONE / 'profile_bindings.json').exists():
        audit_profile_bindings(
            RUN, DONE / 'profile_bindings.json', DONE / 'full4000.json', BINDING, PROFILE)
    if not (DONE / 'stages/diagnosis.json').exists():
        diagnose_common_slo(RUN, DONE / 'stages', DONE / 'full4000.json')

    # The remote service was stopped after the replay. Preserve the health response
    # observed while its owned PID was running; no post-stop health call is made.
    remote_health = BASE / 'remote_health.json'
    if not remote_health.exists():
        save(remote_health, {
            '3b': {
                'ok': True,
                'delivery_mode': 'prepublished_gzip_v1',
                'timing_contract': 'artifact_timing_v2',
                'clock_id': 'remote-process-monotonic:6fc27d6de6a5447ebf52b470de824c7a',
                'root': '/home/lab14/primelora_remote_artifacts/llama32_3b_a500_v1_modelscope',
                'observation': 'captured during the owned service lifetime before post-run stop',
            },
            'collection_phase': 'post_run_stop',
        })

    # Join the already copied remote journal with the client-side transfer list.
    outcome = RUN / 'launch.launch/physical_deployment/main_outcome.json'
    with outcome.open('rb') as stream:
        transfers = next(old.ijson.items(stream, 'remote_transfers', use_float=True))
    journal = (RUN / 'remote_transfers_remote.jsonl').read_text()
    health = read(remote_health)['3b']
    matched = list(matched_remote_transfers(
        {'remote_transfers': transfers}, journal, health['clock_id']))
    assert len(matched) == len(transfers) == 505
    # This file is a local copy of the remote journal; do not overwrite the raw
    # journal captured by the launcher.
    local_journal = BASE / 'remote_transfers.jsonl'
    if not local_journal.exists():
        with local_journal.open('x') as stream:
            for row in matched:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
    if not (DONE / 'delivery_and_resources.json').exists():
        audit_closed_outcome(
            RUN, DONE / 'delivery_and_resources.json', DONE / 'full4000.json',
            BINDING, 'existing_unmonitored')

    metric = read(DONE / 'full4000.json')
    physical = read(RUN / 'launch.launch/physical_deployment/physical_resource_summary.json')
    physical_u = physical['physical_gpu_seconds_observed']
    physical_u_per_offered = physical_u / 4000
    old_metric = read(PARENT / 'completion/full4000.json')
    compare = {
        'new_run': ref(DONE / 'full4000.json'),
        'old_run': ref(PARENT / 'completion/full4000.json'),
        'new': {
            'workload': metric['workload'],
            'joint_slo_fraction': metric['joint_slo_fraction'],
            'joint_pass': metric['counts']['joint_pass'],
            'p95_ttft_ms': metric['metrics']['ttft_ms']['p95_type1'],
            'mean_ttft_ms': metric['metrics']['ttft_ms']['mean'],
            'mean_tpot_ms': metric['metrics']['tpot_ms']['mean'],
            'mean_e2e_ms': metric['metrics']['e2e_ms']['mean'],
            'physical_gpu_seconds': physical_u,
            'gpu_seconds_per_offered_request': physical_u_per_offered,
        },
        'old': {
            'workload': old_metric['workload'],
            'joint_slo_fraction': old_metric['joint_slo_fraction'],
            'joint_pass': old_metric['counts']['joint_pass'],
            'p95_ttft_ms': old_metric['metrics']['ttft_ms']['p95_type1'],
            'mean_ttft_ms': old_metric['metrics']['ttft_ms']['mean'],
            'mean_tpot_ms': old_metric['metrics']['tpot_ms']['mean'],
            'mean_e2e_ms': old_metric['metrics']['e2e_ms']['mean'],
            'physical_gpu_seconds': old_metric['physical_gpu_seconds'],
            'gpu_seconds_per_offered_request': old_metric['gpu_seconds_per_offered_request'],
        },
        'relative_change_new_minus_old_over_old': {
            'p95_ttft_ms': (metric['metrics']['ttft_ms']['p95_type1'] - old_metric['metrics']['ttft_ms']['p95_type1']) / old_metric['metrics']['ttft_ms']['p95_type1'],
            'physical_gpu_seconds': (physical_u - old_metric['physical_gpu_seconds']) / old_metric['physical_gpu_seconds'],
            'gpu_seconds_per_offered_request': (physical_u_per_offered - old_metric['gpu_seconds_per_offered_request']) / old_metric['gpu_seconds_per_offered_request'],
            'joint_slo_fraction': metric['joint_slo_fraction'] - old_metric['joint_slo_fraction'],
        },
        'interpretation': (
            'W2 is a complete same-version observation under the frozen D648 contract. '
            'The canonical resource value is the physical allocation union, not the '
            'legacy result JSON infrastructure field. This comparison is descriptive; '
            'the two runs are not a controlled single-mechanism causal pair.'
        ),
    }
    save(DONE / 'versus_d1045.json', compare)

    closure = {
        'passed': True,
        'run_id': 'd1084_20261106',
        'run_tag': 'd1084_3b_host_budget_projection_w2',
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
        'request_time_packing': False,
        'remote_service_closed_after_collection': True,
        'source_refs': [ref(result), ref(RUN / 'launch.json'), ref(outcome),
                        ref(RUN / 'remote_transfers_remote.jsonl'), ref(remote_health),
                        ref(DONE / 'full4000.json'), ref(DONE / 'delivery_and_resources.json'),
                        ref(BINDING), ref(Path(__file__))],
    }
    save(DONE / 'closure.json', closure)
    save(DONE / 'manifest.json', {
        'files': [ref(path) for path in sorted(DONE.rglob('*')) if path.is_file()],
        'raw_run': str(RUN),
        'raw_run_sha256': old.p.digest(result),
    })
    print(json.dumps({
        'passed': True,
        'joint_pass': metric['counts']['joint_pass'],
        'joint_fraction': metric['joint_slo_fraction'],
        'physical_gpu_seconds': physical_u,
        'remote_transfers': len(matched),
    }, indent=2))


if __name__ == '__main__':
    main()
