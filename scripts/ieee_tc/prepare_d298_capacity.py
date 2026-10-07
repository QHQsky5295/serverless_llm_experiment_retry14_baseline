"""D298: assemble existing capacity/length evidence with the D157 collector.

No inference, weights or trace generation. Every output is exclusive/new.
The service/preparation timing profiles must be newly measured, not relabelled.
"""
import copy
import json
import sys
from bisect import bisect_left
from pathlib import Path

import yaml

from scripts import ieee_tc_preflight as p
from faaslora.datasets.workload_generator import FrozenReplayPlan
from faaslora.storage.http_artifact_store import HttpArtifactStoreClient


def main():
    root = p.ROOT
    rawdir = root / 'results/ieee_tc/p2_backend_qualification/d298_20261007'
    out = root / 'paper_results/ieee_tc/p2_backend/d298_capacity'
    seal = p.verify_seal(root/'paper_results/ieee_tc/safety/20260925_execution_start_protected.json')
    known_change = '/home/qhq/serverless_llm_baselines/scripts/replay_openai_trace.py'
    assert seal['changed'] == [known_change] and seal['entry_count'] == 147
    assert p.digest(Path(known_change)) == '04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22'
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Refusing to overwrite existing D298 preparation evidence')
    rawdir.mkdir(parents=True, exist_ok=True)

    def read(path):
        return json.loads((root / path).read_text())

    def save(name, value):
        path = out / name
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
        return dict(path=str(path.relative_to(root)), sha256=p.digest(path))

    summary_path = Path('paper_results/ieee_tc/p2_backend/20261002_d170_warm_measurement.json')
    summary = read(summary_path)
    projection = root / 'results/ieee_tc/p2_backend_qualification/d170_20261002/measurement_projection1.json'
    assert p.digest(projection) == summary['sources'][str(projection.resolve(strict=True))]
    warm = json.loads(projection.read_text())
    index_path = root / 'paper_results/ieee_tc/p2_backend/20261002_d168_warm_input_index.json'
    assert p.digest(index_path) == warm['warm_reference_index_sha256'] == summary['sources'][str(index_path)]
    plan = FrozenReplayPlan.load(Path(warm['trace']['source_path']), count=4000)
    assert plan.identity() == warm['trace']
    index = p.validate_warm_reference_index(json.loads(index_path.read_text()), plan)
    samples = p.completed_warm_length_samples(warm, index)
    assert len(samples) == 512
    bounds = warm['model_config']['ieee_input_upper_bounds']
    buckets = [[] for _ in range(len(bounds)+1)]
    for sample in samples.values():
        buckets[bisect_left(bounds, sample['native_prompt_tokens'])].append(sample['completed_output_tokens'])
    assert all(buckets)
    audit = dict(kind='native_completed_output_length_initialization_audit_v1',
        development_only=True, production_profile_frozen=False,
        aggregation='deduplicate_original_request_identity_not_warm_round_repetitions',
        source_projection_provenance=dict(path=str(summary_path), sha256=p.digest(root/summary_path)),
        models={'7b': dict(source_run=dict(path=str(projection.relative_to(root)), sha256=p.digest(projection)),
            source_warm_index=dict(path=str(index_path.relative_to(root)), sha256=p.digest(index_path)),
            model_config=warm['model_config'], same_backend='0.30.0', input_upper_bounds=bounds,
            samples=list(samples.values()), unique_original_requests=len(samples), unobserved_bucket_ids=[],
            observed_buckets=[dict(bucket=i, unique_requests=len(values),
                completed_output_mean=sum(values)/len(values)) for i, values in enumerate(buckets)])})
    audit_ref = save('completed_length_audit.json', audit)

    # Reuse the actual D169 backend configuration, not a guessed new search point.
    config_path = root/'configs/ieee_tc/20261002_d169_7b_warm_reference.yaml'
    config = yaml.safe_load(config_path.read_text())['model_profiles']['llama2_7b_main_v2_publicmix']
    cfg = dict(config['model'])
    cfg.update(visible_device_ids=[0], device_id=0, timing_contract='ieee_tc_native_v1',
        ieee_worker_observation=True, ieee_gpu_references=True, ieee_scheduler_observation=True,
        ieee_input_upper_bounds=[759], generation_contract='fixed_length_greedy_v1',
        canonical_prompt_renderer='role_lines_v1', max_input_len=759, max_output_tokens_cap=256,
        ieee_physical_allocation=True, safetensors_load_strategy='auto')
    spec = read('paper_results/ieee_tc/p2_backend/20261002_d157_7b_admission_source_spec.json')
    spec.pop('full_development_contract')
    spec.update(purpose='D298_7b_cap8_admission_enabled_source_calibration_development_only',
        nvme_parent=str(rawdir.resolve()), admission_initialization=dict(
            kind='native_completed_length_binding_v1', audit=audit_ref, model='7b', window_s=5),
        derivation='D169 qualified capacity; D170 completed lengths reused, never latency relabelled; D297 controller retained.',
        order_rule='Six original content-class representatives plus first two distinct existing adapters; eight lanes, offsets 0/4, rotate by round; three rounds and five sources.')
    spec['bins']['admitted_requests'] = [1,2,4,8]
    content_path = root/spec['content_index']['path']
    assert p.digest(content_path) == spec['content_index']['sha256']
    content = json.loads(content_path.read_text())
    client = HttpArtifactStoreClient(endpoint='http://127.0.0.1:1')
    client.configure_content_manifest(content)
    pool = (root/config['storage']['remote_dir']).resolve(strict=True)
    reps = list(spec['representatives'])
    # Extra lanes come from the first1000 existing requests, not generated data.
    for entry in plan.entries[:1000]:
        source = json.loads(entry.source_json)
        aid = source['adapter_id']
        if aid in {r['adapter_id'] for r in reps}:
            continue
        item = client.routing_identity(aid,(pool/aid/'adapter_config.json').read_bytes())
        reps.append(dict(adapter_id=aid, source_request_id=entry.request_id,
            rank=item['rank'], file_tree_sha256=item['content_sha256'],
            logical_bytes=item['remote_payload_bytes'], selection='first_development_request_for_additional_distinct_lane'))
        if len(reps) == 8:
            break
    assert len(reps) == 8 and max(r['rank'] for r in reps) <= 16
    spec['representatives'] = reps
    refs = [dict(source_request_id=r['source_request_id'], adapter_id=r['adapter_id']) for r in reps]
    spec['waves'] = [dict(role='kernel_warmup_retained', round=-1, source='gpu', requests=refs)]
    sources = ['remote','nvme','file_host','native_host','gpu']
    for round_id in range(3):
        for group, offset in enumerate((0,4)):
            shift = (round_id+offset) % 8
            selected = refs[shift:] + refs[:shift]
            for position in range(5):
                spec['waves'].append(dict(role='representative_measurement', round=round_id,
                    group=group, source=sources[(position+round_id+group)%5], requests=selected))
    cfg.update(spec['model_overrides'])
    runtime = root/'results/ieee_tc/p2_backend_qualification/runtime_20260926/cuda_import_attempt3.json'
    prior = json.loads(runtime.read_text())
    configured, evidence, assembly = p.prepare_admission_source_runtime(cfg,spec,
        backend_version=prior['modules']['vllm']['version'], runtime_receipt_sha256=p.digest(runtime),
        source_trace_sha256=plan.source_sha256)
    old = read('paper_results/ieee_tc/p2_backend/20261002_d157_7b_development_contract.json')
    child = assembly['planned_child_model_config']
    contract = dict(kind='ieee_development_control_runtime_contract_v1', model='7b', development_only=True,
        model_config=child, runtime_receipt_sha256=p.digest(runtime), coordination=copy.deepcopy(old['coordination']),
        service_ewma_beta=old['service_ewma_beta'], preparation_ewma_beta=old['preparation_ewma_beta'],
        derivation=dict(kind='qualified_capacity_package_controller_retention_v1',
            source_configuration=dict(path=str(config_path.relative_to(root)),sha256=p.digest(config_path)),
            source_completed_length_summary=dict(path=str(summary_path),sha256=p.digest(root/summary_path)),
            old_timings_relabelled=False), source_raw=evidence['source_run'],
        completed_length_binding=spec['admission_initialization'], base_config_sha256=p.digest(config_path),
        source_workload_profile=old['source_workload_profile'], production_profile_frozen=False,
        full_qualified=False, plan_sha256=p.check_plan())
    contract_ref = save('development_contract.json',contract)
    spec['full_development_contract'] = contract_ref
    spec_ref = save('source_spec.json',spec)
    again = p.prepare_admission_source_runtime(cfg,spec,
        backend_version=prior['modules']['vllm']['version'], runtime_receipt_sha256=p.digest(runtime),
        source_trace_sha256=plan.source_sha256)
    assert again[2]['full_development_contract'] == contract_ref
    small_plan = FrozenReplayPlan.load(Path(warm['trace']['source_path']),count=1000)
    _, bins, identities, waves, static = p.source_profile_inputs(root/spec_ref['path'],small_plan,cfg,pool)
    assert len(waves)==31 and sum(map(len,waves))==248 and len(identities)==8
    # Preserve the raw false result; do not erase the known unrelated user edit.
    save('prepare.json',dict(kind='d298_capacity_preparation_v1', passed=True,
        length_initialization=evidence, assembly=again[2], contract=contract_ref, spec=spec_ref,
        cases=248, representative_cases=240, waves=31, static_adapters=len(static['artifacts']),
        protection_raw=seal, serving_runner_sha256=p.digest(root/'scripts/run_all_experiments.py'),
        metric_protocol_sha256=p.digest(root/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md'),
        plan_sha256=p.check_plan(), gpu_runs=0, full_qualified=False))
    print(json.dumps(dict(passed=True, spec=spec_ref, profile_means=child['ieee_admission_profile']['profile_means'],
        cases=248, protection_raw=seal)),flush=True)


def prepare_prefill_candidate():
    """D307: reuse D298's exact source cases, changing only iteration granularity."""
    root = p.ROOT
    out = root/'paper_results/ieee_tc/p2_backend/d307_prefill'
    rawdir = root/'results/ieee_tc/p2_backend_qualification/d307_20261007'
    if out.exists():
        raise FileExistsError(out)
    refs = {
        'configs/ieee_tc/20261007_d298_7b_capacity_source.yaml': '6b96df57e5f05a9b4a0b597ef988b464d37af6b61e4dac33b3e4444778d1487f',
        'configs/ieee_tc/20261007_d303_7b_full_w0.yaml': '18a5d633fac9a5cf13d4be260d323bb7796ad8698c4d7fd2e0493de141fcb3e5',
        'paper_results/ieee_tc/p2_backend/d298_capacity/source_spec.json': '0f9d8ccb886a81315634ba925e2cbc25153ce7b2d2d6d306008088125eb03649',
        'paper_results/ieee_tc/p2_backend/d298_capacity/development_contract.json': '98bc4292efcb2f9c5c9a1068fccfaeacc460bb35f4b1fbc579b8ef34f32a8afb',
    }
    for path, expected in refs.items():
        assert p.digest(root/path) == expected
    source_path = root/'configs/ieee_tc/20261007_d307_7b_prefill_source.yaml'
    source = yaml.safe_load(source_path.read_text())
    original = yaml.safe_load((root/'configs/ieee_tc/20261007_d298_7b_capacity_source.yaml').read_text())
    key = 'llama2_7b_main_v2_publicmix'
    assert original['model_profiles'][key]['model']['max_num_batched_tokens'] == 1024
    original['model_profiles'][key]['model']['max_num_batched_tokens'] = 512
    assert source == original
    spec = json.loads((root/'paper_results/ieee_tc/p2_backend/d298_capacity/source_spec.json').read_text())
    old = json.loads((root/'paper_results/ieee_tc/p2_backend/d298_capacity/development_contract.json').read_text())
    full = yaml.safe_load((root/'configs/ieee_tc/20261007_d303_7b_full_w0.yaml').read_text())
    spec.pop('full_development_contract')
    spec.update(purpose='D307_7b_iteration_budget512_source_calibration_development_only',
        nvme_parent=str(rawdir.resolve()),
        derivation='D298 identical representative requests/waves; fixed-output completed lengths explicitly transferred; all timings newly measured for budget512.')
    spec['admission_initialization'].update(kind='native_fixed_output_length_transfer_v1',
        model_config_changes={'max_num_batched_tokens': dict(source=1024, target=512)})
    cfg = dict(source['model_profiles'][key]['model'])
    cfg.update(visible_device_ids=[0], device_id=0, timing_contract='ieee_tc_native_v1',
        ieee_worker_observation=True, ieee_gpu_references=True, ieee_scheduler_observation=True,
        ieee_input_upper_bounds=[759], generation_contract='fixed_length_greedy_v1',
        canonical_prompt_renderer='role_lines_v1', max_input_len=759, max_output_tokens_cap=256,
        ieee_physical_allocation=True)
    cfg.update(spec['model_overrides'])
    runtime = root/'results/ieee_tc/p2_backend_qualification/runtime_20260926/cuda_import_attempt3.json'
    prior = json.loads(runtime.read_text())
    args = dict(backend_version=prior['modules']['vllm']['version'],
        runtime_receipt_sha256=p.digest(runtime), source_trace_sha256=spec['trace_sha256'])
    _, evidence, assembly = p.prepare_admission_source_runtime(cfg, spec, **args)
    contract = copy.deepcopy(old)
    coord = copy.deepcopy(full['resource_coordination'])
    for name in ('ieee_service_profile', 'ieee_preparation_profile'):
        coord.pop(name)
    contract.update(model_config=assembly['planned_child_model_config'], coordination=coord,
        completed_length_binding=spec['admission_initialization'], source_raw=evidence['source_run'],
        base_config_sha256=p.digest(source_path), plan_sha256=p.check_plan(),
        derivation=dict(kind='single_native_iteration_budget_candidate_v1',
            source_references=refs, source_configuration=dict(path=str(source_path.relative_to(root)),
                sha256=p.digest(source_path)), old_timings_relabelled=False,
            only_native_change={'max_num_batched_tokens':dict(source=1024, target=512)},
            full_control_source='D303 unchanged', length_transfer=evidence['length_transfer']))
    seal = p.verify_seal(root/'paper_results/ieee_tc/safety/20260925_execution_start_protected.json')
    known = '/home/qhq/serverless_llm_baselines/scripts/replay_openai_trace.py'
    assert seal['changed'] == [known] and seal['entry_count'] == 147
    assert p.digest(Path(known)) == '04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22'
    out.mkdir(parents=True, exist_ok=False)
    rawdir.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        path = out/name
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
        return dict(path=str(path.relative_to(root)), sha256=p.digest(path))
    spec['full_development_contract'] = save('development_contract.json', contract)
    spec_ref = save('source_spec.json', spec)
    _, _, again = p.prepare_admission_source_runtime(cfg, spec, **args)
    trace = root/old['source_raw']['path']
    trace = Path(json.loads(trace.read_text())['trace']['source_path'])
    plan = FrozenReplayPlan.load(trace, count=1000)
    pool = (root/source['model_profiles'][key]['storage']['remote_dir']).resolve(strict=True)
    _, _, identities, waves, static = p.source_profile_inputs(root/spec_ref['path'], plan, cfg, pool)
    assert len(waves) == 31 and sum(map(len,waves)) == 248 and len(identities) == 8
    save('prepare.json', dict(kind='d307_prefill_candidate_preparation_v1', passed=True,
        assembly=again, length_initialization=evidence, source_references=refs,
        spec=spec_ref, cases=248, representative_cases=240, waves=31,
        static_adapters=len(static['artifacts']), protection_raw=seal,
        metric_protocol_sha256=p.digest(root/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md'),
        plan_sha256=p.check_plan(), gpu_runs=0, full_qualified=False))
    print(json.dumps(dict(passed=True, spec=spec_ref, cases=248,
        profile_means=contract['model_config']['ieee_admission_profile']['profile_means'])), flush=True)


def verify_existing_launch(campaign='d298'):
    root = p.ROOT
    if campaign == 'd298':
        out = root/'paper_results/ieee_tc/p2_backend/d298_capacity'
        source_path = root/'configs/ieee_tc/20261007_d298_7b_capacity_source.yaml'
        base_path = root/'configs/ieee_tc/20261002_d169_7b_warm_reference.yaml'
        differences = {'safetensors_load_strategy':'auto'}
    elif campaign == 'd307':
        out = root/'paper_results/ieee_tc/p2_backend/d307_prefill'
        source_path = root/'configs/ieee_tc/20261007_d307_7b_prefill_source.yaml'
        base_path = root/'configs/ieee_tc/20261007_d298_7b_capacity_source.yaml'
        differences = {'max_num_batched_tokens':512}
    else:
        raise ValueError(campaign)
    original = yaml.safe_load(base_path.read_text())
    source = yaml.safe_load(source_path.read_text())
    key = 'llama2_7b_main_v2_publicmix'
    expected = copy.deepcopy(original)
    expected['model_profiles'][key]['model'].update(differences)
    assert source == expected
    cfg = dict(source['model_profiles'][key]['model'])
    cfg.update(visible_device_ids=[0],device_id=0,timing_contract='ieee_tc_native_v1',
        ieee_worker_observation=True,ieee_gpu_references=True,ieee_scheduler_observation=True,
        ieee_input_upper_bounds=[759],generation_contract='fixed_length_greedy_v1',
        canonical_prompt_renderer='role_lines_v1',max_input_len=759,max_output_tokens_cap=256,
        ieee_physical_allocation=True)
    spec = json.loads((out/'source_spec.json').read_text())
    cfg.update(spec['model_overrides'])
    runtime = root/'results/ieee_tc/p2_backend_qualification/runtime_20260926/cuda_import_attempt3.json'
    prior = json.loads(runtime.read_text())
    _,_,assembly = p.prepare_admission_source_runtime(cfg,spec,
        backend_version=prior['modules']['vllm']['version'],runtime_receipt_sha256=p.digest(runtime),
        source_trace_sha256=spec['trace_sha256'])
    pool = (root/source['model_profiles'][key]['storage']['remote_dir']).resolve(strict=True)
    content = json.loads((root/spec['content_index']['path']).read_text())
    client = HttpArtifactStoreClient(endpoint='http://127.0.0.1:1')
    client.configure_content_manifest(content)
    ranks = [client.routing_identity(a['id'],(pool/a['id']/'adapter_config.json').read_bytes())['rank']
             for a in content['artifacts']]
    assert len(ranks)==500 and max(ranks)==16
    receipt = dict(kind=campaign+'_actual_source_launch_config_binding_v1',passed=True,
        source_config=dict(path=str(source_path.relative_to(root)),sha256=p.digest(source_path)),
        original_capacity_config=dict(path=str(base_path.relative_to(root)),sha256=p.digest(base_path)),
        sole_yaml_difference=differences,assembly=assembly,
        pool_count=len(ranks),pool_max_rank=max(ranks),gpu_runs=0)
    with (out/'launch_config_binding.json').open('x') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=True,pool_max_rank=16,source_config=receipt['source_config'])),flush=True)


def verify_prefill_smoke(campaign='d308'):
    """Bind the existing Full runner to the actually measured D307 child."""
    from scripts.run_all_experiments import (
        _prepare_scenario_runtime_model_config, _prepare_dedicated_subprocess_model_cfg)
    from faaslora.experiment.instance_pool import FrozenServiceProfiles
    from faaslora.preloading.preloading_planner import FrozenPreparationProfiles
    root = p.ROOT
    out = root/'paper_results/ieee_tc/p2_backend/d308_prefill'
    config = root/'configs/ieee_tc/20261007_d308_7b_prefill_validation_100.yaml'
    previous = root/'configs/ieee_tc/20261007_d302_7b_frontier_validation_100.yaml'
    smoke_ref = None
    if campaign == 'd309':
        out = root/'paper_results/ieee_tc/p2_backend/d309_full_w0'
        assert not out.exists(), 'never overwrite preparation'
        config = root/'configs/ieee_tc/20261007_d309_7b_full_w0.yaml'
        previous = root/'configs/ieee_tc/20261007_d303_7b_full_w0.yaml'
        smoke_path = root/'paper_results/ieee_tc/p2_backend/d308_prefill/smoke1/audit.json'
        smoke = json.loads(smoke_path.read_text())
        assert smoke['stats']['completed'] == 100
        for reference in smoke['raw_evidence']:
            assert p.digest(Path(reference['path'])) == reference['sha256']
        smoke_ref = dict(path=str(smoke_path.relative_to(root)),sha256=p.digest(smoke_path))
    elif campaign == 'd310':
        out = root/'paper_results/ieee_tc/p2_backend/d310_read_pin_validation'
        config = root/'configs/ieee_tc/20261007_d310_7b_prefill_validation_100.yaml'
        previous = root/'configs/ieee_tc/20261007_d308_7b_prefill_validation_100.yaml'
    elif campaign == 'd312':
        out = root/'paper_results/ieee_tc/p2_backend/d312_host_validation'
        config = root/'configs/ieee_tc/20261007_d312_7b_prefill_validation_100.yaml'
        previous = root/'configs/ieee_tc/20261007_d310_7b_prefill_validation_100.yaml'
    elif campaign in ('d311', 'd313'):
        out = root/f'paper_results/ieee_tc/p2_backend/{campaign}_full_w0'
        config = root/f'configs/ieee_tc/20261007_{campaign}_7b_full_w0.yaml'
        source_campaign = 'd309' if campaign == 'd311' else 'd311'
        smoke_directory = 'd310_read_pin_validation' if campaign == 'd311' else 'd312_host_validation'
        previous = root/f'configs/ieee_tc/20261007_{source_campaign}_7b_full_w0.yaml'
        smoke_path = root/f'paper_results/ieee_tc/p2_backend/{smoke_directory}/smoke1/audit.json'
        smoke = json.loads(smoke_path.read_text())
        assert smoke['stats']['completed'] == 100
        for reference in smoke['raw_evidence']:
            assert p.digest(Path(reference['path'])) == reference['sha256']
        smoke_ref = dict(path=str(smoke_path.relative_to(root)),sha256=p.digest(smoke_path))
    elif campaign != 'd308':
        raise ValueError('unknown frozen prefill validation identity')
    assert not out.exists(), 'never overwrite preparation'
    contract_path = root/'paper_results/ieee_tc/p2_backend/d307_prefill/development_contract.json'
    measured_path = root/'paper_results/ieee_tc/p2_backend/d307_prefill/initialization/manifest.json'
    c = yaml.safe_load(config.read_text())
    old = yaml.safe_load(previous.read_text())
    contract = json.loads(contract_path.read_text())
    measured = json.loads(measured_path.read_text())
    assert p.check_plan() == '0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    assert measured['passed'] and measured['contract']['sha256'] == p.digest(contract_path)
    assert measured['source_run']['sha256'] == p.digest(Path(measured['source_run']['path']))
    assert measured['source_execution_commit'] == 'f0813771f35ffe30463ae19d1abec38f89658f14'
    expected = copy.deepcopy(old)
    expected['model']['max_num_batched_tokens'] = 512
    expected['model']['ieee_admission_profile'] = contract['model_config']['ieee_admission_profile']
    expected['experiment']['output_dir'] = old['experiment']['output_dir'].replace('d302_', 'd308_').replace('frontier', 'prefill')
    expected['storage']['nvme_cache_dir'] = old['storage']['nvme_cache_dir'].replace('d302_', 'd308_').replace('frontier', 'prefill')
    expected['storage']['host_cache_dir'] = old['storage']['host_cache_dir'].replace('tc-d302', 'tc-d308').replace('frontier', 'prefill')
    expected['scenarios'][0]['name'] = 'tc_ieee_prefill_validation_100'
    if campaign == 'd309':
        expected['experiment']['output_dir'] = old['experiment']['output_dir'].replace('d303_', 'd309_')
        expected['storage']['nvme_cache_dir'] = old['storage']['nvme_cache_dir'].replace('d303_', 'd309_')
        expected['storage']['host_cache_dir'] = old['storage']['host_cache_dir'].replace('tc-d303', 'tc-d309')
        expected['scenarios'][0]['name'] = 'tc_ieee_full'
        smoke_config = yaml.safe_load((root/'configs/ieee_tc/20261007_d308_7b_prefill_validation_100.yaml').read_text())
        assert c['model'] == smoke_config['model'] and c['resource_coordination'] == smoke_config['resource_coordination']
        assert c['workload']['total_requests'] == 4000
    elif campaign in ('d310', 'd312'):
        # The read-pin repair changes no control/profile/model setting. Only
        # output and cold-cache identities differ from its existing source smoke.
        expected = copy.deepcopy(old)
        source_campaign = 'd308' if campaign == 'd310' else 'd310'
        expected['experiment']['output_dir'] = old['experiment']['output_dir'].replace(source_campaign+'_', campaign+'_')
        expected['storage']['nvme_cache_dir'] = old['storage']['nvme_cache_dir'].replace(source_campaign+'_', campaign+'_')
        expected['storage']['host_cache_dir'] = old['storage']['host_cache_dir'].replace('tc-'+source_campaign, 'tc-'+campaign)
    elif campaign in ('d311', 'd313'):
        expected = copy.deepcopy(old)
        source_campaign = 'd309' if campaign == 'd311' else 'd311'
        smoke_campaign = 'd310' if campaign == 'd311' else 'd312'
        expected['experiment']['output_dir'] = old['experiment']['output_dir'].replace(source_campaign+'_', campaign+'_')
        expected['storage']['nvme_cache_dir'] = old['storage']['nvme_cache_dir'].replace(source_campaign+'_', campaign+'_')
        expected['storage']['host_cache_dir'] = old['storage']['host_cache_dir'].replace('tc-'+source_campaign, 'tc-'+campaign)
        smoke_config = yaml.safe_load((root/f'configs/ieee_tc/20261007_{smoke_campaign}_7b_prefill_validation_100.yaml').read_text())
        assert c['model'] == smoke_config['model'] and c['resource_coordination'] == smoke_config['resource_coordination']
        assert c['workload']['total_requests'] == 4000
    profile_receipts = {}
    for key, cls in (('ieee_service_profile', FrozenServiceProfiles), ('ieee_preparation_profile', FrozenPreparationProfiles)):
        binding = copy.deepcopy(measured['bindings'][key])
        binding['path'] = str(Path(binding['path']).relative_to(root))
        expected['resource_coordination'][key] = binding
    assert c == expected, 'undeclared Full configuration difference'
    configured = _prepare_scenario_runtime_model_config(c['model'], c['workload'], c['resource_coordination'])
    child, assembly = _prepare_dedicated_subprocess_model_cfg(configured, device_id=0, runtime_gpu_ids=[0])
    assert child == contract['model_config'], 'Full child differs from measured child'
    coordination = {k:v for k,v in c['resource_coordination'].items()
                    if k not in ('ieee_service_profile','ieee_preparation_profile')}
    assert coordination == contract['coordination'], 'D303 controls changed'
    for key, cls in (('ieee_service_profile', FrozenServiceProfiles), ('ieee_preparation_profile', FrozenPreparationProfiles)):
        binding = c['resource_coordination'][key]
        profile = cls.load(root/binding['path'], expected_sha256=binding['sha256'],
            model_config=child, expected_context=binding['context'], beta=binding['ewma_beta'])
        profile_receipts[key] = dict(profile_id=profile.profile_id, classes=len(profile.profiles), binding=binding)
    receipt = dict(kind=campaign+'_full_validation_binding_v1', passed=True, gpu_runs=0, smoke_reference=smoke_ref,
        source_references=[dict(path=str(x.relative_to(root)),sha256=p.digest(x))
            for x in (config, previous, contract_path, measured_path)],
        model_config=child, assembly=assembly, profiles=profile_receipts,
        controls_unchanged=True, workload_unchanged=c['workload']==old['workload'],
        plan_sha256=p.check_plan(), metric_protocol_sha256=p.digest(root/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md'),
        serving_runner_sha256=p.digest(root/'scripts/run_all_experiments.py'),
        common_slo_qualified=False, full_qualified=False, no_figures=True)
    out.mkdir(parents=True, exist_ok=False)
    with (out/'prelaunch_binding.json').open('x') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=True,config_sha256=p.digest(config),classes={k:v['classes'] for k,v in profile_receipts.items()})),flush=True)


def verify_restored_full(campaign='d315'):
    """Bind restored 1024 Full or its D317 smoke to the measured child."""
    from scripts.run_all_experiments import (
        _prepare_scenario_runtime_model_config, _prepare_dedicated_subprocess_model_cfg)
    from faaslora.experiment.instance_pool import FrozenServiceProfiles
    from faaslora.preloading.preloading_planner import FrozenPreparationProfiles
    root = p.ROOT
    out = root/'paper_results/ieee_tc/p2_backend/d315_restored_full'
    config = root/'configs/ieee_tc/20261007_d315_7b_full_w0.yaml'
    previous = root/'configs/ieee_tc/20261007_d303_7b_full_w0.yaml'
    if campaign == 'd317':
        out = root/'paper_results/ieee_tc/p2_backend/d317_copy_validation'
        config = root/'configs/ieee_tc/20261007_d317_7b_copy_validation_100.yaml'
        previous = root/'configs/ieee_tc/20261007_d315_7b_full_w0.yaml'
    elif campaign == 'd318':
        out = root/'paper_results/ieee_tc/p2_backend/d318_copy_full'
        config = root/'configs/ieee_tc/20261007_d318_7b_full_w0.yaml'
        previous = root/'configs/ieee_tc/20261007_d315_7b_full_w0.yaml'
    elif campaign != 'd315':
        raise ValueError('unknown restored-1024 validation identity')
    assert not out.exists(), 'never overwrite preparation'
    contract_path = root/'paper_results/ieee_tc/p2_backend/d298_capacity/development_contract.json'
    measured_path = root/'paper_results/ieee_tc/p2_backend/d298_capacity/initialization/manifest.json'
    c, old = (yaml.safe_load(x.read_text()) for x in (config, previous))
    expected = copy.deepcopy(old)
    expected['experiment']['output_dir'] = old['experiment']['output_dir'].replace('d303_', 'd315_')
    expected['storage']['nvme_cache_dir'] = old['storage']['nvme_cache_dir'].replace('d303_', 'd315_')
    expected['storage']['host_cache_dir'] = old['storage']['host_cache_dir'].replace('tc-d303-', 'tc-d315-')
    if campaign == 'd317':
        expected['experiment']['output_dir'] = old['experiment']['output_dir'].replace('d315_', 'd317_').replace('7b_full_w0', '7b_copy_validation_100')
        expected['storage']['nvme_cache_dir'] = old['storage']['nvme_cache_dir'].replace('d315_', 'd317_').replace('7b_nvme_full_w0', '7b_nvme_copy_validation_100')
        expected['storage']['host_cache_dir'] = old['storage']['host_cache_dir'].replace('tc-d315-7b-full-w0', 'tc-d317-7b-copy-validation-100')
        expected['scenarios'][0]['name'] = 'tc_ieee_copy_validation_100'
    elif campaign == 'd318':
        expected['experiment']['output_dir'] = old['experiment']['output_dir'].replace('d315_', 'd318_')
        expected['storage']['nvme_cache_dir'] = old['storage']['nvme_cache_dir'].replace('d315_', 'd318_')
        expected['storage']['host_cache_dir'] = old['storage']['host_cache_dir'].replace('tc-d315-', 'tc-d318-')
    assert c == expected, 'undeclared difference from measured D303 settings'
    for path in (c['experiment']['output_dir'], c['storage']['nvme_cache_dir'], c['storage']['host_cache_dir']):
        assert not Path(path).exists(), 'run/cache identity already exists'
    assert c['model']['max_num_batched_tokens'] == 1024
    assert c['workload']['total_requests'] == 4000
    contract, measured = (json.loads(x.read_text()) for x in (contract_path, measured_path))
    assert measured['passed'] and measured['contract']['sha256'] == p.digest(contract_path)
    assert measured['source_run']['sha256'] == p.digest(Path(measured['source_run']['path']))
    assert measured['source_execution_commit'] == '783fa42e6b3ba1cd0e034d8e0cddcd3d72c9b42e'
    configured = _prepare_scenario_runtime_model_config(c['model'], c['workload'], c['resource_coordination'])
    child, assembly = _prepare_dedicated_subprocess_model_cfg(configured, device_id=0, runtime_gpu_ids=[0])
    assert child == contract['model_config'], 'restored Full child differs from measured D298 child'
    profiles = {}
    for key, cls in (('ieee_service_profile', FrozenServiceProfiles), ('ieee_preparation_profile', FrozenPreparationProfiles)):
        binding = c['resource_coordination'][key]
        recorded = copy.deepcopy(measured['bindings'][key])
        recorded['path'] = str(Path(recorded['path']).relative_to(root))
        assert binding == recorded, 'profile binding changed'
        profile = cls.load(root/binding['path'], expected_sha256=binding['sha256'],
            model_config=child, expected_context=binding['context'], beta=binding['ewma_beta'])
        profiles[key] = dict(profile_id=profile.profile_id, classes=len(profile.profiles), binding=binding)
    seal = p.verify_seal(root/'paper_results/ieee_tc/safety/20260925_execution_start_protected.json')
    known = '/home/qhq/serverless_llm_baselines/scripts/replay_openai_trace.py'
    assert seal['changed'] == [known] and seal['entry_count'] == 147
    assert p.digest(Path(known)) == '04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22'
    files = (config, previous, contract_path, measured_path,
        root/'scripts/run_all_experiments.py', root/'faaslora/preloading/preloading_planner.py',
        root/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh', Path(__file__))
    if campaign == 'd318':
        smoke_dir = root/'paper_results/ieee_tc/p2_backend/d317_copy_validation'
        checkpoint_path = smoke_dir/'validation_checkpoint.json'
        checkpoint = json.loads(checkpoint_path.read_text())
        for name, sha in checkpoint['curated_output_sha256'].items():
            assert p.digest(smoke_dir/name) == sha, 'D317 evidence changed'
        smoke_path = smoke_dir/'smoke2/audit.json'
        smoke = json.loads(smoke_path.read_text())
        assert smoke['stats']['completed'] == 100
        assert smoke['copy_retention_trigger']['observed_cumulative_retained_scopes_max'] > 0
        for row in smoke['raw_evidence']:
            assert p.digest(Path(row['path'])) == row['sha256'], 'D317 raw source changed'
        assert p.digest(root/'scripts/run_all_experiments.py') == checkpoint['source_sha256']['scripts/run_all_experiments.py'], 'candidate changed after smoke'
        smoke_config = yaml.safe_load((root/'configs/ieee_tc/20261007_d317_7b_copy_validation_100.yaml').read_text())
        assert c['model'] == smoke_config['model'] and c['resource_coordination'] == smoke_config['resource_coordination']
        files += (checkpoint_path, smoke_path)
    receipt = dict(kind=campaign+'_restored_full_binding_v1', passed=True,
        source_references=[dict(path=str(x.relative_to(root)),sha256=p.digest(x)) for x in files],
        model_config=child, assembly=assembly, profiles=profiles, protection_raw=seal,
        same_measured_settings_except_paths_and_scenario=True, restored_measured_budget=1024,
        diagnostic_prefix_count=100 if campaign == 'd317' else None,
        reused_source=measured['source_run'], current_correctness_fixes_retained=True,
        plan_sha256=p.check_plan(), metric_protocol_sha256=p.digest(root/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md'),
        user_manifest_sha256=p.digest(root/'configs/generated/lora_manifest_1000.json'),
        gpu_runs=0, common_slo_qualified=False, full_qualified=False, n_correct=None, no_figures=True)
    assert receipt['plan_sha256'] == '0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    assert receipt['metric_protocol_sha256'] == '5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    assert receipt['user_manifest_sha256'] == '3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df'
    out.mkdir(parents=True, exist_ok=False)
    with (out/'prelaunch_binding.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=True, config_sha256=p.digest(config),
        classes={k:v['classes'] for k,v in profiles.items()})), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--prefill-candidate']:
        prepare_prefill_candidate()
    elif sys.argv[1:] == ['--verify-prefill-launch']:
        verify_existing_launch('d307')
    elif sys.argv[1:] == ['--verify-launch']:
        verify_existing_launch()
    elif sys.argv[1:] == ['--verify-prefill-smoke']:
        verify_prefill_smoke()
    elif sys.argv[1:] == ['--verify-prefill-full']:
        verify_prefill_smoke('d309')
    elif sys.argv[1:] == ['--verify-read-pin-smoke']:
        verify_prefill_smoke('d310')
    elif sys.argv[1:] == ['--verify-read-pin-full']:
        verify_prefill_smoke('d311')
    elif sys.argv[1:] == ['--verify-host-transaction-smoke']:
        verify_prefill_smoke('d312')
    elif sys.argv[1:] == ['--verify-host-transaction-full']:
        verify_prefill_smoke('d313')
    elif sys.argv[1:] == ['--verify-restored-full']:
        verify_restored_full()
    elif sys.argv[1:] == ['--verify-copy-smoke']:
        verify_restored_full('d317')
    elif sys.argv[1:] == ['--verify-copy-full']:
        verify_restored_full('d318')
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('Expected no argument, --verify-launch, --prefill-candidate, --verify-prefill-launch, --verify-prefill-smoke, --verify-prefill-full, --verify-read-pin-smoke, --verify-read-pin-full, --verify-host-transaction-smoke, --verify-host-transaction-full, --verify-restored-full, --verify-copy-smoke or --verify-copy-full')
