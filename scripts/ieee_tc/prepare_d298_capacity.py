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


def verify_existing_launch():
    root = p.ROOT
    out = root/'paper_results/ieee_tc/p2_backend/d298_capacity'
    source_path = root/'configs/ieee_tc/20261007_d298_7b_capacity_source.yaml'
    base_path = root/'configs/ieee_tc/20261002_d169_7b_warm_reference.yaml'
    original = yaml.safe_load(base_path.read_text())
    source = yaml.safe_load(source_path.read_text())
    key = 'llama2_7b_main_v2_publicmix'
    expected = copy.deepcopy(original)
    expected['model_profiles'][key]['model']['safetensors_load_strategy'] = 'auto'
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
    receipt = dict(kind='d298_actual_source_launch_config_binding_v1',passed=True,
        source_config=dict(path=str(source_path.relative_to(root)),sha256=p.digest(source_path)),
        original_capacity_config=dict(path=str(base_path.relative_to(root)),sha256=p.digest(base_path)),
        sole_yaml_difference={'safetensors_load_strategy':'auto'},assembly=assembly,
        pool_count=len(ranks),pool_max_rank=max(ranks),gpu_runs=0)
    with (out/'launch_config_binding.json').open('x') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=True,pool_max_rank=16,source_config=receipt['source_config'])),flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--verify-launch']:
        verify_existing_launch()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('Expected no argument or --verify-launch')
