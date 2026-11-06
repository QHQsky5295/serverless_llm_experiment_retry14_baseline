"""D1081 closed evidence: reuse D703/D299 and D1064, never rerun serving."""
import collections
import json
import os
from pathlib import Path
import subprocess
import sys

import ijson
import yaml

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
BASE = Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1081_20261106')
RUN = BASE/'d1081_3b_positive_demand_options_w1'
OUT = Path(__file__).resolve().parent
DONE = OUT/'completion'
sys.path.insert(0, str(ROOT))
from scripts import ieee_tc_preflight as p
from scripts.ieee_tc.audit_d703_3b_full import (
    audit_closed_outcome, audit_profile_bindings, diagnose_common_slo)
from scripts.ieee_tc.export_d701_3b_profile import matched_remote_transfers
from scripts.ieee_tc.analyze_d455_stages import analyze_retained_overhead
from faaslora.storage.http_artifact_store import HttpArtifactStoreClient


def read(path):
    return json.loads(path.read_text())


def ref(path):
    return dict(path=str(path), sha256=p.digest(path), bytes=path.stat().st_size)


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    group = p.cg_path()
    identity = p.scope_identity(group.name)
    assert os.sched_getaffinity(0) == {2, 3, 26, 27}
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert (group/'memory.high').read_text().strip() == str(3*p.GIB)
    assert (group/'memory.max').read_text().strip() == str(4*p.GIB)
    assert (group/'memory.swap.max').read_text().strip() == '0'
    assert not subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    launch = read(RUN/'launch.json')
    assert all(launch[k] is True for k in
               ('pass', 'service_path_removed', 'native_gpu_context_release_confirmed'))
    metric = read(DONE/'full4000.json')
    assert metric['passed'] and metric['counts']['n'] == 4000
    assert metric['source_run'] == str(RUN) and metric['n_correct'] is None
    for source in metric['source_refs']:
        assert p.digest(Path(source['path'])) == source['sha256']
    result, = RUN.glob('experiment_results_full*.json')
    with result.open('rb') as stream:
        metadata = next(ijson.items(stream, 'metadata', use_float=True))
    assert metadata['applied_env_overrides']['FAASLORA_OUTPUT_DIR'] == str(RUN)
    config = ROOT/'configs/ieee_tc/20261104_d1047_3b_event_busy_time_w1.yaml'
    assert p.digest(config) == '263bdcd6ff255108ddaffdd9ce46898d7f12be15948d7fe3d7ac27d5ea8828d2'
    assert launch['command'][3] == str(config.relative_to(ROOT))
    projection = yaml.safe_load(config.read_text())
    projection['experiment']['output_dir'] = str(RUN)
    binding = DONE/'analysis_output_binding.json'
    save(binding, projection)
    save(DONE/'analysis_output_binding_provenance.json', dict(
        original_config=ref(config), launch=ref(RUN/'launch.json'), result=ref(result),
        changed_fields={'experiment.output_dir': str(RUN)},
        reason='Bind recorded output override only; no serving or metric change.'))
    audit_profile_bindings(RUN, DONE/'profile_bindings.json', DONE/'full4000.json', binding,
        ROOT/'paper_results/ieee_tc/p2_backend/d1031_3b_startup_memory_profile/export/manifest.json')
    diagnose_common_slo(RUN, DONE/'stages', DONE/'full4000.json')

    outcome = RUN/'launch.launch/physical_deployment/main_outcome.json'
    with outcome.open('rb') as stream:
        transfers = next(ijson.items(stream, 'remote_transfers', use_float=True))
    remote = '/home/lab14/primelora_remote/tc/d80_20260927/3b/transfers-cb5b7717a5b54506bcf591e4d67a0b5d.jsonl'
    # Verified open descriptor of the unchanged service PID3725723 before collection.
    journal = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
        'primelora-artifact-174', 'cat', remote], text=True, timeout=30)
    client = HttpArtifactStoreClient(endpoint='http://192.168.4.174:18080',
        token=Path('/home/qhq/.config/primelora-tc-d75/artifact.token').read_text().strip(),
        timeout_s=5, required_delivery_mode='prepublished_gzip_v1')
    health = client.health()
    assert health['ok'] and health['timing_contract'] == 'artifact_timing_v2'
    matched = matched_remote_transfers(dict(remote_transfers=transfers), journal, health['clock_id'])
    with (BASE/'remote_transfers.jsonl').open('x') as stream:
        for row in matched:
            stream.write(json.dumps(row, sort_keys=True)+'\n')
    save(BASE/'remote_health.json', {'3b': health, 'collection_phase': 'post_run_existing_service'})
    audit_closed_outcome(RUN, DONE/'delivery_and_resources.json', DONE/'full4000.json',
                         binding, 'existing_unmonitored')

    parent = ROOT/'paper_results/ieee_tc/p2_backend/d1047_same_version_w1/completion'
    sources = []
    for directory in (parent, DONE):
        for name in ('full4000.json', 'delivery_and_resources.json'):
            path = directory/name
            sources.append(ref(path))
            sources += [r for r in read(path)['source_refs']
                if Path(r['path']).name == 'main_outcome.json'
                or Path(r['path']).name.startswith('experiment_results_full')]
        sources.append(ref(Path(read(directory/'full4000.json')['source_run'])/'launch.json'))
    source_set = OUT/'completion_sources.json'
    save(source_set, dict(source_refs=sources))
    comparison = analyze_retained_overhead(OUT/'overhead_comparison', [source_set])
    with outcome.open('rb') as stream:
        events = list(ijson.items(stream, 'planning_cpu_events.item', use_float=True))
    sparse = [e for e in events if e.get('option_scope') == 'positive_demand_v1']
    noops = [e for e in events if e.get('confirmed_gpu_no_preparation') is True]
    missing = [e for e in events if e['operation'] == 'owned_execution_epoch'
               and e.get('option_scope') != 'positive_demand_v1'
               and e.get('confirmed_gpu_no_preparation') is not True]
    assert sparse and not missing
    assert all(e['frozen_execution_validated'] is True
               and e['excluded_zero_demand_options'] >= 0
               and e['materialized_positive_demand_options'] >= 0 for e in sparse)
    save(DONE/'option_representation.json', dict(
        passed=True, planning_events=len(events), sparse_events=len(sparse),
        confirmed_gpu_no_preparation_events=len(noops),
        states=dict(collections.Counter(e['state'] for e in events)),
        excluded_zero_demand_options=sum(e['excluded_zero_demand_options'] for e in sparse),
        materialized_positive_demand_options=sum(e['materialized_positive_demand_options'] for e in sparse),
        counting_rule='Canonical planning_cpu_events only; no duplicate public/private aggregation.',
        source=ref(outcome)))

    seal_path = ROOT/'paper_results/ieee_tc/safety/20260925_execution_start_protected.json'
    protected = {path: value for path, value in read(seal_path)['entries'].items()
        if '/paper_results/final_v2/' in path or '/figs/paper/' in path}
    assert len(protected) == 144
    assert all(p.digest(Path(path)) == value['sha256'] for path, value in protected.items())
    assert p.digest(ROOT/'configs/generated/lora_manifest_1000.json') == '3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df'
    assert p.digest(ROOT/'docs/ieee_tc/EXECUTION_STATUS.md') == '41a4db5bc464dfb885b20d2b5ac321a3631a1a02291d64ac823735505957e374'
    caches = [BASE/(RUN.name+'_nvme'), Path('/dev/shm')/('tc-'+RUN.name)]
    assert all(not path.exists() or not any(path.iterdir()) for path in caches)
    samples = foreign = escaped = 0
    with (RUN/'launch.launch/watchdog.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['event'] == 'resource_sample':
                samples += 1
                foreign += len(row['gpu']['foreign_compute_processes'])
                escaped += len(row['gpu']['escaped_owned_processes'])
    assert samples and foreign == escaped == 0
    save(DONE/'closure.json', dict(passed=True, requests=4000,
        source_commit='8cdc8f815aa35952c8329bcba402f092db7be5ad',
        n_correct=None, ranking_eligible=False, development_only=True,
        resource_samples=samples, foreign_compute_samples=foreign, escaped_owned_samples=escaped,
        protected_files_unchanged=144, work_caches_empty=True, cache_paths=list(map(str, caches)),
        auxiliary_identity=identity, auxiliary_resources=p.cgroup_snapshot(group),
        first_metric_scope_peak_bytes=1869828096, first_metric_scope_memory_events=0,
        matched_request_inputs=comparison['matched_request_inputs'],
        source_refs=[ref(x) for x in (result, RUN/'launch.json', config, binding, seal_path, Path(__file__))]))
    save(DONE/'manifest.json', dict(files=[ref(path) for path in sorted(DONE.rglob('*')) if path.is_file()]))
    print(json.dumps(dict(passed=True, requests=4000, remote_transfers=len(matched),
                         sparse_planning_events=len(sparse), confirmed_gpu_noops=len(noops))))


if __name__ == '__main__':
    main()
