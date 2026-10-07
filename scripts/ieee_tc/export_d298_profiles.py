"""Data-only D157/D89 export adaptation; existing validators remain authoritative.

No figures, inference, remote mutation, new profile estimators or default fills.
Run once in the bounded auxiliary domain, after actual source-run release.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml
from scripts import ieee_tc_preflight as p
from faaslora.storage.http_artifact_store import HttpArtifactStoreClient
from faaslora.experiment.instance_pool import FrozenServiceProfiles
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


def main(campaign='d298'):
    global RAW, CURATED
    if campaign == 'd298':
        config_path = ROOT/'configs/ieee_tc/20261007_d298_7b_capacity_source.yaml'
        source_commit = '783fa42e6b3ba1cd0e034d8e0cddcd3d72c9b42e'
    elif campaign == 'd307':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d307_20261007'
        CURATED = ROOT/'paper_results/ieee_tc/p2_backend/d307_prefill'
        config_path = ROOT/'configs/ieee_tc/20261007_d307_7b_prefill_source.yaml'
        source_commit = (RAW/'source_commit.txt').read_text().strip()
        assert re.fullmatch('[0-9a-f]{40}', source_commit), 'missing exact launch commit'
    else:
        raise ValueError(campaign)
    out = CURATED / 'initialization'
    assert not out.exists(), 'never overwrite a prior export'
    source, launch_path = RAW/'source1.json', RAW/'source1_launch.json'
    raw, launch = read(source), read(launch_path)
    assert raw['pass'] and raw['stage'] == 'complete'
    assert raw['shutdown_called'] and raw['profile_workspaces_removed']
    assert launch['pass'] and launch['service_returncode'] == launch['watchdog_returncode'] == 0
    assert launch['service_path_removed'] and launch['native_gpu_context_release_confirmed']
    assert launch['preflight']['policy'] == p.POLICY
    assert raw['service']['affinity'] == p.POLICY['service_cpus']
    spec_path = CURATED/'source_spec.json'
    spec = read(spec_path)
    assert p.digest(spec_path) == raw['source_profile_spec_sha256']
    assert raw['source_profile_spec'] == spec
    contract_path = ROOT/spec['full_development_contract']['path']
    assert p.digest(contract_path) == spec['full_development_contract']['sha256']
    contract = read(contract_path)
    runtime = ROOT/'results/ieee_tc/p2_backend_qualification/runtime_20260926/cuda_import_attempt3.json'
    assert p.digest(runtime) == raw['runtime_receipt_sha256'] == contract['runtime_receipt_sha256']
    assert p.digest(config_path) == raw['config_sha256'] == read(CURATED/'launch_config_binding.json')['source_config']['sha256']
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
    assert p.digest(Path(raw['trace']['source_path'])) == spec['trace_sha256'] == raw['trace']['source_sha256']
    journal = RAW/'remote_transfers.jsonl'
    remote_rows = [json.loads(line) for line in journal.read_text().splitlines()]
    remote = {r['transfer_id']:r for r in remote_rows}
    transfers = raw['remote_transfers']
    assert len(remote) == len(remote_rows) == len(transfers)
    assert {r['http_transfer_id'] for r in transfers} == set(remote)
    remote_clock = read(RAW/'remote_health.json')['health']['clock_id']
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
    watchdog = RAW/'source1_launch.launch/watchdog.jsonl'
    with watchdog.open() as stream:
        samples = [r for line in stream if (r := json.loads(line))['event'] == 'resource_sample']
    assert samples and not any(r['decision']['abort_reasons'] for r in samples)
    input_contract = dict(source_trace_sha256=spec['trace_sha256'],
        content_index_sha256=spec['content_index']['sha256'],
        generation_contract=raw['model_config']['generation_contract'],
        canonical_prompt_renderer=raw['model_config']['canonical_prompt_renderer'],
        prompt_content_cap=raw['model_config']['max_input_len'],
        output_cap=raw['model_config']['max_output_tokens_cap'],
        artifact_mode=raw['artifact_mode'],declared_source_profile_sha256=raw['source_profile_spec_sha256'])
    envelope = dict(inference_policy=launch['preflight']['policy'],
        artifact_service=dict(memory_high_bytes=2*p.GIB,memory_max_bytes=4*p.GIB,
            memory_swap_max_bytes=0,cpus=list(range(2,20))+list(range(22,40)),
            published_server_sha256='a31fa9212143914f5a7fb21dea126a23eaafedffe199ff9513de9ce03eb62a2c',
            delivery_mode='prepublished_gzip_v1',link_mbps=1000,rate_limit=None))
    context = dict(backend_environment_sha256=raw['runtime_receipt_sha256'],
        resource_envelope_sha256=object_hash(envelope),input_contract_sha256=object_hash(input_contract))
    service, prep, coverage = p.measured_source_profile_payloads(raw,
        source_run_sha256=p.digest(source),identities=identities,
        context=context,model_config=contract['model_config'])
    assert len(service['samples']) == 240 and len(prep['samples']) == 192
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
    roles = {f'source-profile/w{i}/l{lane}':w['role'] for i,w in enumerate(spec['waves'])
             for lane in range(len(w['requests']))}
    outputs = defaultdict(set)
    for q in raw['requests']:
        if roles[q['request_id']] == 'representative_measurement':
            outputs[q['source_request_id'],q['adapter_id'],q['prompt_sha256'],q['target_tokens']].add(
                q['timing']['completion_token_ids_sha256'])
    summary = dict(requests=len(raw['requests']),representative_requests=240,warmup_requests=8,
        measured_source_counts=dict(Counter(q['requested_source'] for q in raw['requests']
            if roles[q['request_id']] == 'representative_measurement')),
        remote_uuid_pairs=len(transfers),wire_bytes=sum(r['transferred_bytes'] for r in transfers),
        remote_pack_performed=False,physical_gpu_seconds=release[0]-acquire[0],
        varied_output_groups=sum(len(v)>1 for v in outputs.values()),total_output_groups=len(outputs),
        resources=dict(samples=len(samples),peak_bytes=max(r['service']['memory.peak'] for r in samples),
            min_host_available_bytes=min(r['host']['available_bytes'] for r in samples),
            max_swap_bytes=max(r['service']['memory.swap.current'] for r in samples),
            memory_events={k:max(r['service']['memory.events'][k] for r in samples)
                for k in ('high','max','oom','oom_kill')}))
    result = dict(kind='measured_development_profile_export_v1',model='7b',passed=True,
        source_execution_commit=source_commit,source_run=ref(source),
        contract=spec['full_development_contract'],context=context,
        context_definition=dict(backend_environment='SHA256 of qualified native runtime receipt',
            resource_envelope=envelope,input_contract=input_contract,
            structured_hash='SHA256 UTF-8 json sort_keys=True separators=(comma,colon) allow_nan=False'),
        bindings=bindings,profiles={k:v.identity() for k,v in loaded.items()},coverage=coverage,
        estimates=estimates,activation_geometry=loaded['preparation'].activation_layout(),summary=summary,
        full_qualified=False,production_profile_frozen=False,numerical_adapter_correctness_qualified=False,
        n_correct=None,common_slo_qualified=False,independent_runs=1,ci=None,
        protection_raw=seal,offline_only=True,new_gpu_runs=0,no_figures=True,
        raw_evidence=[ref(path) for path in (source,launch_path,journal,watchdog,RAW/'remote_monitor.log',
            RAW/'remote_stop.log',config_path,runtime,Path(__file__),ROOT/'scripts/ieee_tc_preflight.py')])
    if campaign == 'd307':
        result['raw_evidence'].append(ref(RAW/'source_commit.txt'))
    save('manifest.json',result)
    print(json.dumps(dict(passed=True,summary=summary,profiles=result['profiles'],coverage=coverage),indent=2),flush=True)


if __name__ == '__main__':
    if not sys.argv[1:]:
        main()
    elif sys.argv[1:] == ['d307']:
        main('d307')
    else:
        raise SystemExit('Expected no argument or d307')
