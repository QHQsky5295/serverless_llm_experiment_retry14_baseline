"""D1083 W0 collection using D1081/D703/D299/D1064; no serving execution."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
BASE = Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1083_20261106')
RUN = BASE/'d1083_3b_host_budget_projection_w0'
OUT = Path(__file__).resolve().parent
DONE = OUT/'completion'
PARENT = OUT.parent/'d1046_same_version_w0'
HELPER = OUT.parent/'d1081_positive_demand_options/collect_completion.py'
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('d1081_collection', HELPER)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
p, read, ref, save = old.p, old.read, old.ref, old.save


def main():
    group = p.cg_path()
    identity = p.scope_identity(group.name)
    assert os.sched_getaffinity(0) == {2, 3, 26, 27}
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    for name, expected in [('memory.high', 3*p.GIB), ('memory.max', 4*p.GIB), ('memory.swap.max', 0)]:
        assert (group/name).read_text().strip() == str(expected)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    launch = read(RUN/'launch.json')
    assert all(launch[k] is True for k in ('pass', 'service_path_removed', 'native_gpu_context_release_confirmed'))
    metric = read(DONE/'full4000.json')
    assert metric['passed'] and metric['counts']['n'] == 4000 and metric['workload'] == 'W0'
    assert metric['source_run'] == str(RUN) and metric['n_correct'] is None
    for source in metric['source_refs']:
        assert p.digest(Path(source['path'])) == source['sha256']
    result, = RUN.glob('experiment_results_full*.json')
    with result.open('rb') as stream:
        metadata = next(old.ijson.items(stream, 'metadata', use_float=True))
    assert metadata['applied_env_overrides']['FAASLORA_OUTPUT_DIR'] == str(RUN)
    config = ROOT/'configs/ieee_tc/20261104_d1047_3b_event_busy_time_w1.yaml'
    assert p.digest(config) == '263bdcd6ff255108ddaffdd9ce46898d7f12be15948d7fe3d7ac27d5ea8828d2'
    assert launch['command'][3] == str(config.relative_to(ROOT))
    projection = old.yaml.safe_load(config.read_text())
    projection['experiment']['output_dir'] = str(RUN)
    binding = DONE/'analysis_output_binding.json'
    save(binding, projection)
    save(DONE/'analysis_output_binding_provenance.json', dict(
        original_config=ref(config), launch=ref(RUN/'launch.json'), result=ref(result),
        changed_fields={'experiment.output_dir': str(RUN)},
        reason='Bind the recorded output override, not serving or evaluation. W0 is the actual replay view.'))
    old.audit_profile_bindings(RUN, DONE/'profile_bindings.json', DONE/'full4000.json', binding,
        ROOT/'paper_results/ieee_tc/p2_backend/d1031_3b_startup_memory_profile/export/manifest.json')
    old.diagnose_common_slo(RUN, DONE/'stages', DONE/'full4000.json')

    outcome = RUN/'launch.launch/physical_deployment/main_outcome.json'
    with outcome.open('rb') as stream:
        transfers = next(old.ijson.items(stream, 'remote_transfers', use_float=True))
    # Observed through the unchanged owned PID's open fd after full GPU closure.
    remote = '/home/lab14/primelora_remote/tc/d80_20260927/3b/transfers-d340510e653447e9aef73f2bbce29e0a.jsonl'
    state = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', 'primelora-artifact-174',
        'systemctl --user show primelora-artifact-tc-3b.service -p MainPID -p InvocationID'], text=True)
    assert 'MainPID=3815006' in state and 'InvocationID=4028149ca9974aabbaaafee381867b00' in state
    journal = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
        'primelora-artifact-174', 'cat', remote], text=True, timeout=30)
    client = old.HttpArtifactStoreClient(endpoint='http://192.168.4.174:18080',
        token=Path('/home/qhq/.config/primelora-tc-d75/artifact.token').read_text().strip(),
        timeout_s=5, required_delivery_mode='prepublished_gzip_v1')
    health = client.health()
    assert health['ok'] and health['timing_contract'] == 'artifact_timing_v2'
    assert health['clock_id'] == 'remote-process-monotonic:021a783af0934f0284674bb5af981b8f'
    matched = old.matched_remote_transfers(dict(remote_transfers=transfers), journal, health['clock_id'])
    with (BASE/'remote_transfers.jsonl').open('x') as stream:
        for row in matched:
            stream.write(json.dumps(row, sort_keys=True)+'\n')
    save(BASE/'remote_health.json', {'3b': health, 'collection_phase': 'post_run_existing_service'})
    old.audit_closed_outcome(RUN, DONE/'delivery_and_resources.json', DONE/'full4000.json', binding, 'existing_unmonitored')

    sources = []
    for directory in (PARENT/'completion', DONE):
        for name in ('full4000.json', 'delivery_and_resources.json'):
            path = directory/name
            sources.append(ref(path))
            sources += [r for r in read(path)['source_refs'] if Path(r['path']).name == 'main_outcome.json'
                        or Path(r['path']).name.startswith('experiment_results_full')]
        sources.append(ref(Path(read(directory/'full4000.json')['source_run'])/'launch.json'))
    source_set = OUT/'completion_sources.json'
    save(source_set, dict(source_refs=sources))
    comparison = old.analyze_retained_overhead(OUT/'overhead_comparison', [source_set])
    seal = ROOT/'paper_results/ieee_tc/safety/20260925_execution_start_protected.json'
    protected = {path: value for path, value in read(seal)['entries'].items()
                 if '/paper_results/final_v2/' in path or '/figs/paper/' in path}
    assert len(protected) == 144
    assert all(p.digest(Path(path)) == value['sha256'] for path, value in protected.items())
    for name, digest in [
        ('configs/generated/lora_manifest_1000.json', '3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df'),
        ('docs/ieee_tc/EXECUTION_STATUS.md', '41a4db5bc464dfb885b20d2b5ac321a3631a1a02291d64ac823735505957e374')]:
        assert p.digest(ROOT/name) == digest
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
        source_commit='78219424ddf135f24f5e69b3c3468c217f5bd21a',
        same_serving_implementation_as='462f9987944d68370be7e411abb9252980a3e2b1',
        n_correct=None, ranking_eligible=False, development_only=True,
        resource_samples=samples, foreign_compute_samples=foreign, escaped_owned_samples=escaped,
        protected_files_unchanged=144, work_caches_empty=True, cache_paths=list(map(str, caches)),
        auxiliary_identity=identity, auxiliary_resources=p.cgroup_snapshot(group),
        first_metric_scope='primelora-tc-aux-7dca3c1f223f49cc17066b39562f9127.scope',
        first_metric_scope_peak_bytes=898539520, first_metric_scope_memory_events=0,
        matched_request_inputs=comparison['matched_request_inputs'],
        source_refs=[ref(x) for x in (result, RUN/'launch.json', config, binding, seal, Path(__file__), HELPER)]))
    save(DONE/'manifest.json', dict(files=[ref(path) for path in sorted(DONE.rglob('*')) if path.is_file()]))
    print(json.dumps(dict(passed=True, requests=4000, remote_transfers=len(matched))))


if __name__ == '__main__':
    main()
