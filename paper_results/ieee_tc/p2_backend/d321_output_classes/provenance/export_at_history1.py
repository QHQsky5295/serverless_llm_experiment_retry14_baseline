"""Data-only D157/D89 export adaptation; existing validators remain authoritative.

No figures, inference, remote mutation, new profile estimators or default fills.
Run once in the bounded auxiliary domain, after actual source-run release.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
from dataclasses import replace
from bisect import bisect_left
import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import yaml
from scripts import ieee_tc_preflight as p
from faaslora.storage.http_artifact_store import HttpArtifactStoreClient
from faaslora.experiment.instance_pool import (FrozenServiceProfiles, ServiceClassBins,
    ServiceIntervalObservation, NativeServiceIntervalObserver)
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


def historical_service_samples(requests, *, source_sha, identities, previous_bins):
    """Reuse completed Full native intervals, not estimates or pooled fallbacks.

    This is development timing evidence only. Native identity/count consistency
    does not establish numerical adapter qualification. The caller verifies the
    entire run, physical release, model/configuration and content SHA chain.
    """
    if not re.fullmatch('[0-9a-f]{64}', source_sha) or not requests:
        raise ValueError('nonempty SHA-bound history required')
    samples, seen = [], set()
    for r in requests:
        rid, aid = r['request_id'], r['adapter_id']
        if not rid or rid in seen or aid not in identities:
            raise ValueError('unique requests and known artifact identity required')
        seen.add(rid)
        e, t = r['gpu_reference_evidence'], r['native_token_timing']
        admission, native, intervals = e['source_admission'], e['receipt'], e['service_intervals']
        source, identity = admission['source'], identities[aid]
        n, target, prompt = r['completion_tokens'], r['requested_completion_tokens'], r['canonical_prompt_tokens']
        native_prompt = t['actual_prompt_tokens']
        if (r['success'] is not True or r['generation_contract'] != 'fixed_length_greedy_v1'
                or r['completion_token_source'] != 'vllm_token_ids' or r['output_contract_match'] is not True
                or type(n) is not int or n < 1 or n != target or n != t['native_output_tokens']
                or t['native_terminal_observed'] is not True or not isinstance(prompt, (int, float))
                or not math.isfinite(prompt) or int(prompt) != prompt or prompt < 0
                or not isinstance(native_prompt, (int, float)) or not math.isfinite(native_prompt)
                or int(native_prompt) != native_prompt or native_prompt < 1
                or e['state'] != 'released' or e['release_receipt'].get('released') is not True
                or e['pending_kv_admission']['state'] != 'closed'
                or e['pending_kv_admission']['close_receipt'].get('closed') is not True
                or native.get('acquired') is not True or native.get('lora_name') != aid):
            raise ValueError('completed native generation and released ownership required')
        if (admission['clock_id'] != t['native_clock_id'] or native['clock_id'] != t['native_clock_id']
                or native['owner_id'] != t['gpu_reference_owner_id']
                or native['lease_id'] != t['gpu_reference_lease_id']):
            raise ValueError('mixed native clock or reference identity')
        content = source.get('expected_content_sha256') if source['native'] else source.get('content_sha256')
        if content != identity['content_sha256']:
            raise ValueError('source content identity differs from immutable artifact')
        features = dict(tier=source['tier'], representation=source['representation'],
            footprint_bytes=source['footprint_bytes'], adapter_rank=identity['rank'],
            prompt_tokens=int(prompt), declared_output_tokens=target,
            admitted_after_accept=admission['admitted_after_accept'])
        old_key = previous_bins.classify(**features)
        if asdict(old_key) != admission['service_class'] or intervals['service_class'] != admission['service_class']:
            raise ValueError('reconstructed features disagree with admission-time class: '
                + json.dumps(dict(request_id=rid, reconstructed=asdict(old_key),
                    admission=admission['service_class'], interval=intervals['service_class'],
                    features=features), sort_keys=True))
        a, b, f, last = [intervals[k] for k in ('admitted_monotonic_s', 'acquired_monotonic_s',
                                              'first_monotonic_s', 'last_monotonic_s')]
        if (any(type(x) not in (int, float) or not math.isfinite(x) for x in (a, b, f, last))
                or not 0 < a <= b <= f <= last or a != admission['admitted_monotonic_s']
                or a != t['controller_admitted_monotonic_s']
                or f != t['native_first_token_monotonic_s'] or last != t['native_last_token_monotonic_s']
                or (source['tier'] == 'gpu' and a != b)
                or (source['tier'] != 'gpu' and b != t['gpu_reference_acquired_monotonic_s'])):
            raise ValueError('completed native D/T/O boundaries required')
        events = e['service_events']
        if len(events) != 2:
            raise ValueError('complete first/last native event pair required')
        observation = ServiceIntervalObservation.for_profiling(old_key, a)
        if observation.acquired_at is None:
            observation.acquire(b)
        observer = NativeServiceIntervalObserver(observation, clock_id=t['native_clock_id'],
            adapter_id=aid, gpu_reference=native)
        for event in events:
            observer(event)
        # The frontend may deliver several already-timed native tokens together;
        # first-event count need not be one. Reuse the production observer and
        # native first/last boundaries, never chunk receipt time or text counts.
        if observation.first_at != f or observation.last_at != last or events[-1]['token_count'] != n:
            raise ValueError('native event does not bind this completed interval')
        samples.append(dict(correct=True, source_run_sha256=source_sha, request_id=rid,
            attempt_id=native['lease_id'], source_measurement='native_admission_acquisition_token_events_v1',
            admission_clock_id=admission['clock_id'], native_clock_id=t['native_clock_id'],
            class_features=features, native_output_tokens=n, protected_at_admission=True,
            native_prompt_tokens=int(native_prompt),
            prompt_feature_source='canonical_prompt_tokens_at_admission',
            admitted_monotonic_s=a, acquired_monotonic_s=b,
            first_token_monotonic_s=f, last_token_monotonic_s=last))
    return samples


def historical_output_bins(samples, *, output_cap):
    """One predetermined quartile rule; never use latency/rankings to pick edges."""
    values = sorted(s['class_features']['declared_output_tokens'] for s in samples)
    if (not values or type(output_cap) is not int or output_cap < 1
            or any(type(x) is not int or not 1 <= x <= output_cap for x in values)):
        raise ValueError('completed declared-output history within the fixed cap required')
    return sorted({values[math.ceil(len(values)*q)-1] for q in (.25, .5, .75)} | {output_cap})


def service_rebin_coverage(base, historical, *, output_cap):
    """Report the full reachable domain; missing classes remain missing."""
    bins = copy.deepcopy(base['bins'])
    bins['declared_output_tokens'] = historical_output_bins(historical, output_cap=output_cap)
    old = ServiceClassBins(**{k:tuple(v) for k,v in base['bins'].items()})
    new = ServiceClassBins(**{k:tuple(v) for k,v in bins.items()})
    old_keys = {old.classify(**s['class_features']) for s in base['samples']}
    output_bins = {bisect_left(new.declared_output_tokens, x) for x in range(1, output_cap+1)}
    required = {replace(key, output_limit_bin=o) for key in old_keys for o in output_bins}
    samples = copy.deepcopy(base['samples']) + copy.deepcopy(historical)
    if len({(s['source_run_sha256'], s['request_id'], s['attempt_id']) for s in samples}) != len(samples):
        raise ValueError('duplicate profile observations')
    counts = Counter(new.classify(**s['class_features']) for s in samples)
    if not set(counts) <= required:
        raise ValueError('historical sample escapes the qualified non-output class domain')
    encode = lambda k: json.dumps(asdict(k), sort_keys=True)
    return dict(bins=bins, samples=samples, rule='historical_declared_output_type1_quartiles_cap_v1'), dict(
        required_classes=len(required), observed_classes=len(counts),
        missing_classes=[asdict(k) for k in sorted(required-set(counts), key=encode)],
        counts=[dict(service_class=asdict(k), n=counts[k]) for k in sorted(counts, key=encode)],
        class_domain_covered=required <= set(counts), output_edges=bins['declared_output_tokens'],
        historical_samples=len(historical), retained_source_samples=len(base['samples']),
        no_default_fills=True, production_profile_frozen=False, numerical_qualification=False)


def export_history_length_coverage():
    """D321: retain reusable historical samples and exact missing-class inventory."""
    from scripts.ieee_tc.plot_d290_diagnostic import audit_run, binding_checks
    from scripts.run_all_experiments import (
        _prepare_scenario_runtime_model_config, _prepare_dedicated_subprocess_model_cfg)
    out = ROOT/'paper_results/ieee_tc/p2_backend/d321_output_classes/history1'
    if out.exists(): raise FileExistsError(out)
    plan = Path('/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md')
    protocol = ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md'
    assert p.digest(plan) == '0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    assert p.digest(protocol) == '5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    prior = ROOT/'paper_results/ieee_tc/p2_backend/d315_full_w0/validated1'
    seal = read(prior/'manifest.json')
    assert p.digest(prior/'audit.json') == seal['outputs']['audit.json']
    audit = read(prior/'audit.json')
    source = Path(audit['evidence']['source'])
    assert p.digest(source) == audit['evidence']['source_sha256'] == '6e5007630c9780c417e65109ef3005daedba3dafa87a843ff75c4885f91c362f'
    cfg_path = ROOT/'configs/ieee_tc/20261007_d315_7b_full_w0.yaml'
    refs = {r['path']:r['sha256'] for r in seal['source_refs']}
    assert p.digest(cfg_path) == refs[str(cfg_path)]
    cfg = yaml.safe_load(cfg_path.read_text())
    binding = cfg['resource_coordination']['ieee_service_profile']
    base_path = ROOT/binding['path']
    configured = _prepare_scenario_runtime_model_config(cfg['model'], cfg['workload'], cfg['resource_coordination'])
    child, _ = _prepare_dedicated_subprocess_model_cfg(configured, device_id=0, runtime_gpu_ids=[0])
    loaded = FrozenServiceProfiles.load(base_path, expected_sha256=binding['sha256'],
        model_config=child, expected_context=binding['context'], beta=binding['ewma_beta'])
    index_path = Path(cfg['model']['artifact_content_manifest_path'])
    index = read(index_path)
    assert p.digest(index_path) == refs[str(index_path)]
    client = HttpArtifactStoreClient(endpoint='http://127.0.0.1:1')
    client.configure_content_manifest(index)
    pool = Path(cfg['storage']['remote_dir']).resolve(strict=True)
    identities = {}
    for entry in index['artifacts']:
        config_file = (pool/entry['id']/'adapter_config.json').resolve(strict=True)
        expected = next(f['sha256'] for f in entry['files'] if f['path']=='adapter_config.json')
        assert config_file.parent.parent == pool and p.digest(config_file) == expected
        identities[entry['id']] = client.routing_identity(entry['id'], config_file.read_bytes())
    _, evidence, requests = audit_run(source)
    assert len(requests) == 4000 and len(identities) == 500
    for row in requests:
        assert all(binding_checks(row, identities[row['adapter_id']]['content_sha256']).values())
    samples = historical_service_samples(requests, source_sha=evidence['source_sha256'],
        identities=identities, previous_bins=loaded.bins)
    draft, coverage = service_rebin_coverage(read(base_path), samples, output_cap=child['max_output_tokens_cap'])
    paths = [source, cfg_path, base_path, index_path, prior/'manifest.json', prior/'audit.json',
        Path(evidence['physical_source']), plan, protocol, Path(__file__),
        ROOT/'scripts/ieee_tc/plot_d290_diagnostic.py', ROOT/'faaslora/experiment/instance_pool.py']
    out.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        with (out/name).open('x') as stream: json.dump(value, stream, indent=2, allow_nan=False)
    # Not a runtime profile identity: incomplete coverage may not be launched.
    save('historical_samples.json', dict(kind='historical_full_service_samples_v1',
        source=evidence, samples=samples, numerical_qualification=False))
    save('coverage.json', coverage)
    save('candidate_definition.json', dict(kind='declared_output_rebin_candidate_v1',
        rule=draft['rule'], bins=draft['bins'], base_profile=ref(base_path),
        historical_samples=ref(out/'historical_samples.json'),
        launchable=False, stage='coverage_only', model_config=child,
        context=binding['context'], ewma_beta=binding['ewma_beta'],
        numerical_qualification=False, required_next='measure_missing_classes_or_validate_complete_export'))
    save('manifest.json', dict(source_refs=[ref(path) for path in paths],
        outputs={path.name:p.digest(path) for path in sorted(out.iterdir())},
        no_figures=True, new_gpu_runs=0, n_correct=None))
    print(json.dumps(coverage, indent=2), flush=True)


def main(campaign='d298'):
    global RAW, CURATED
    if campaign == 'd298':
        config_path = ROOT/'configs/ieee_tc/20261007_d298_7b_capacity_source.yaml'
        source_commit = '783fa42e6b3ba1cd0e034d8e0cddcd3d72c9b42e'
    elif campaign in ('d307', 'd307-attempt2'):
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d307_20261007'
        if campaign == 'd307-attempt2':
            RAW = RAW/'attempt2'
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
    if campaign in ('d307', 'd307-attempt2'):
        result['raw_evidence'].append(ref(RAW/'source_commit.txt'))
    save('manifest.json',result)
    print(json.dumps(dict(passed=True,summary=summary,profiles=result['profiles'],coverage=coverage),indent=2),flush=True)


if __name__ == '__main__':
    if not sys.argv[1:]:
        main()
    elif sys.argv[1:] in (['d307'], ['d307-attempt2']):
        main(sys.argv[1])
    elif sys.argv[1:] == ['--history-length-coverage']:
        export_history_length_coverage()
    else:
        raise SystemExit('Expected no argument, d307, d307-attempt2 or --history-length-coverage')
