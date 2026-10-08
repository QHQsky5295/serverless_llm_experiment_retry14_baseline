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


def capacity_probe_waves(index, *, source_count, candidate_count, prefix=1000):
    """Existing longest declared contexts, distinct adapters, stable index ties.

    These are capacity diagnostics, not a representative workload or SLO warm
    recalibration. Three interleaved8/candidate waves expose the TPOT trade-off.
    """
    if not (type(source_count) is int and type(candidate_count) is int
            and 0 < source_count < candidate_count and type(prefix) is int and prefix > 0):
        raise ValueError('explicit increasing diagnostic capacities required')
    population = [r for g in index['groups'] for r in g['selected'] if r['source_index'] < prefix]
    population.sort(key=lambda r: (-(r['native_prompt_tokens'] + r['target_tokens']), r['source_index']))
    selected, seen = [], set()
    for row in population:
        if row['adapter_id'] in seen:
            continue
        selected.append(dict(row))
        seen.add(row['adapter_id'])
        if len(selected) == candidate_count:
            break
    if len(selected) != candidate_count:
        raise ValueError('insufficient existing distinct adapter requests')
    refs = [dict(source_request_id=r['source_request_id'], adapter_id=r['adapter_id']) for r in selected]
    waves = [dict(role='kernel_warmup_retained', round=-1, source='gpu', requests=refs)]
    for r in range(3):
        for n in ((source_count, candidate_count) if r % 2 == 0 else (candidate_count, source_count)):
            waves.append(dict(role='representative_measurement', round=r, source='gpu',
                              diagnostic_wave_size=n, requests=refs[:n]))
    return selected, waves


def prepare_capacity_probe(*, followup=False):
    """D343 bounded capacity candidate, same native collector; no GPU here."""
    root = p.ROOT
    out = root/'paper_results/ieee_tc/p2_backend/d343_capacity_probe'
    raw = root/'results/ieee_tc/p2_backend_qualification/d343_20261008'
    prior_raw = raw
    if followup:
        out, raw = out/'cap14', raw/'cap14'
    if out.exists() or raw.exists():
        raise FileExistsError('D343 preparation is append-only')
    read = lambda path: json.loads((root/path).read_text())
    probe_path = 'paper_results/ieee_tc/p2_backend/20261002_d169_warm_probe.json'
    old_probe = read(probe_path)
    assert old_probe['passed'] and old_probe['native_completed'] == 32
    index_path = 'paper_results/ieee_tc/p2_backend/20261002_d168_warm_input_index.json'
    index = read(index_path)
    assert p.digest(root/index_path) == old_probe['sources'][str(root/index_path)]
    oldpath = root/'configs/ieee_tc/20261007_d298_7b_capacity_source.yaml'
    cfgpath = root/('configs/ieee_tc/20261008_d343_7b_capacity_probe14.yaml' if followup
                   else 'configs/ieee_tc/20261008_d343_7b_capacity_probe.yaml')
    old = yaml.safe_load(oldpath.read_text())
    actual = yaml.safe_load(cfgpath.read_text())
    key = 'llama2_7b_main_v2_publicmix'
    base = old['model_profiles'][key]['model']
    blocks = min(b['kv_free_before'] for b in old_probe['batches'])
    block_sizes = {b['kv_tokens_per_block'] for b in old_probe['batches']}
    assert len(block_sizes) == 1
    block = block_sizes.pop()
    followup_evidence = None
    if followup:
        first = json.loads((prior_raw/'source1.json').read_text())
        launch = json.loads((prior_raw/'source1_launch.json').read_text())
        assert p.digest(prior_raw/'source1.json') == 'fe7c440cd0b064c6094dfc16888c84c4ae06dfff7b7a35a8f44939f91fb37073'
        assert not first['pass'] and not first['requests'] and first['physical_allocation']['released']
        assert launch['native_gpu_context_release_confirmed'] and launch['service_path_removed']
        assert len(first['profile_waves']) == 1
        obs = first['profile_waves'][0]['capacity_before']
        assert obs['admitted'] == [] and obs['unretired_iterations'] == 0 and obs['native_deferred_free_batches'] == 0
        assert obs['kv_tokens_per_block'] == block and obs['kv_unreserved_free_blocks'] == 942
        blocks = obs['kv_unreserved_free_blocks']
        followup_evidence = dict(path=str((prior_raw/'source1.json').relative_to(root)),
            sha256=p.digest(prior_raw/'source1.json'), source_capacity=15,
            fresh_free_blocks=blocks, first_attempt_retained=True,
            diagnostic_schema_error='integer_zero_unretired_iterations_was_checked_as_empty_list',
            independent_capacity_reason='942free < 15*64; corrected guard still rejects15',
            same_run_fallback=False, maximum_minimal_checks=2)
    per_request = (base['max_model_len'] + block - 1)//block
    capacity = blocks//per_request
    assert capacity == (14 if followup else 15) and base['runtime_concurrency_cap'] == 8
    fields = ('max_num_seqs', 'runtime_concurrency_cap', 'max_loras')
    changed = {k:dict(source=base[k], target=capacity) for k in fields}
    expected = copy.deepcopy(old)
    expected['model_profiles'][key]['model'].update({k:capacity for k in fields})
    assert expected == actual
    selected, waves = capacity_probe_waves(index, source_count=8, candidate_count=capacity)
    spec = read('paper_results/ieee_tc/p2_backend/d298_capacity/source_spec.json')
    spec.pop('full_development_contract')
    spec.update(purpose='D343_bounded_native_capacity_probe_not_SLO_reference',
        nvme_parent=str(raw.resolve()), waves=waves, representatives=selected,
        capacity_probe_contract=dict(kind='native_capacity_probe_v1', candidate_concurrency=capacity,
            source_concurrency=8, max_model_len=base['max_model_len']),
        derivation=('floor(D343_C15_fresh_free_KV_blocks/ceil(max_model_len/block_size)); separate bounded followup; remeasure again' if followup else
                    'floor(min_observed_D169_free_KV_blocks/ceil(max_model_len/block_size)); fresh larger-runtime KV guard required; no full-run acceptance.'),
        order_rule='Existing D168 index within first1000, descending declared full context, unique adapters; alternating8/candidate waves three rounds; no new inputs.')
    if followup_evidence is not None:
        spec['bounded_followup_evidence'] = followup_evidence
    spec['bins']['admitted_requests'] = [1,2,4,8,capacity]
    spec['admission_initialization'].update(kind='native_fixed_output_capacity_length_transfer_v1',
        model_config_changes=changed)
    cfg = dict(actual['model_profiles'][key]['model'])
    cfg.update(visible_device_ids=[0], device_id=0, timing_contract='ieee_tc_native_v1',
        ieee_worker_observation=True, ieee_gpu_references=True, ieee_scheduler_observation=True,
        ieee_input_upper_bounds=[759], generation_contract='fixed_length_greedy_v1',
        canonical_prompt_renderer='role_lines_v1', max_input_len=759, max_output_tokens_cap=256,
        ieee_physical_allocation=True)
    cfg.update(spec['model_overrides'])
    runtime = root/'results/ieee_tc/p2_backend_qualification/runtime_20260926/cuda_import_attempt3.json'
    prior = json.loads(runtime.read_text())
    _, evidence, assembly = p.prepare_admission_source_runtime(cfg,spec,
        backend_version=prior['modules']['vllm']['version'], runtime_receipt_sha256=p.digest(runtime),
        source_trace_sha256=spec['trace_sha256'])
    seal = p.verify_seal(root/'paper_results/ieee_tc/safety/20260925_execution_start_protected.json')
    known = Path('/home/qhq/serverless_llm_baselines/scripts/replay_openai_trace.py')
    assert seal['entry_count'] == 147 and seal['changed'] == [str(known)]
    assert p.digest(known) == '04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22'
    out.mkdir(parents=True, exist_ok=False)
    raw.mkdir(parents=True, exist_ok=False)
    def save(name, data):
        path = out/name
        with path.open('x') as f: json.dump(data,f,indent=2,allow_nan=False)
        return dict(path=str(path.relative_to(root)),sha256=p.digest(path))
    spec_ref = save('source_spec.json',spec)
    plan = FrozenReplayPlan.load(Path(index['source']['source_path']), count=1000)
    pool = (root/actual['model_profiles'][key]['storage']['remote_dir']).resolve(strict=True)
    _,_,identities,resolved,_ = p.source_profile_inputs(root/spec_ref['path'],plan,cfg,pool)
    case_count = capacity + 3*(8+capacity)
    assert len(resolved) == 7 and sum(map(len,resolved)) == case_count and len(identities) == capacity
    # Existing full native startup logs, not a new GPU measurement.
    startup = read('paper_results/ieee_tc/p2_backend/d341_full_w0/startup_manifest.json')
    import re
    log_evidence = []
    for ref in startup['sources']:
        path = Path(ref['path'])
        if path.name != 'worker.log': continue
        assert p.digest(path) == ref['sha256']
        values = re.findall(r'GPU KV cache size: ([0-9,]+) tokens',path.read_text())
        log_evidence.append(dict(source=ref, kv_pool_tokens=[int(v.replace(',','')) for v in values]))
    receipt = dict(kind='d343_capacity_probe_preparation_v1', passed=True, development_only=True,
        full_qualified=False, n_correct=None, gpu_runs=0, cases=case_count,
        measurement_cases=3*(8+capacity), bounded_followup_evidence=followup_evidence,
        candidate_derivation=dict(free_blocks=blocks, tokens_per_block=block,
            maximum_context_blocks=per_request,candidate_capacity=capacity,
            limitation='old capacity is only a candidate bound; fresh native guard required'),
        changed_model_fields=changed, runtime_configuration_assembly=assembly,
        completed_length_transfer=evidence, spec=spec_ref, protected=seal,
        d341_native_startup_capacity=log_evidence,
        retained_request_limitation='D341 request projection has no raw KV scheduler snapshots; startup log capacity is not a live free-block series',
        source_references=[dict(path=path,sha256=p.digest(root/path)) for path in (
            probe_path,index_path,str(oldpath.relative_to(root)),str(cfgpath.relative_to(root)),
            'docs/ieee_tc/D342_DISPATCH_WAIT_DIAGNOSIS.md')],
        plan_sha256=p.check_plan(),metric_protocol_sha256=p.digest(root/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md'))
    save('prepare.json',receipt)
    print(json.dumps(dict(passed=True,candidate_capacity=capacity,cases=case_count,spec=spec_ref)),flush=True)


def missing_length_waves(population, bins, missing, *, rounds=3):
    """Index existing requests only; collect genuine gaps without a new matrix.

    Lowest count in each missing admitted bin is sufficient to attempt coverage.
    Actual admission classes, not this intended order, decide final coverage.
    """
    from collections import defaultdict
    from faaslora.experiment.instance_pool import ServiceClassBins
    if type(rounds) is not int or rounds < 1 or not population or not missing:
        raise ValueError('positive rounds and nonempty existing population/gaps required')
    classes = ServiceClassBins(**{k:tuple(v) for k,v in bins.items()})
    groups = defaultdict(list)
    fields = ('prompt_bin', 'output_limit_bin', 'rank_bin', 'footprint_bin')
    for key in missing:
        source = {('host','verified_regular_file_tree_v1'):'file_host',
                  ('remote','tar_gzip_verified_file_tree_v1'):'remote'}.get(
                      (key['tier'],key['representation']))
        if source is None:
            raise ValueError('this measured gap set needs an explicitly supported source')
        groups[(source,)+tuple(key[f] for f in fields)].append(key)
    templates = []
    for group, keys in sorted(groups.items()):
        admitted = max(k['admitted_bin'] for k in keys)
        size = 1 if admitted == 0 else classes.admitted_requests[admitted-1]+1
        candidates, seen = [], set()
        for row in population:
            observed = classes.classify(**row['class_features'])
            if tuple(getattr(observed,f) for f in fields) != group[1:]:
                continue
            if row['adapter_id'] in seen:
                continue
            candidates.append(dict(source_request_id=row['source_request_id'], adapter_id=row['adapter_id']))
            seen.add(row['adapter_id'])
            if len(candidates) == size:
                break
        if len(candidates) != size:
            raise ValueError(f'insufficient existing distinct adapters for gap {group}: {len(candidates)}/{size}')
        templates.append(dict(source=group[0], requests=candidates, intended_missing_classes=keys))
    # Retain one native warmup using existing selected inputs, not measured fill.
    warm, seen = [], set()
    for wave in templates:
        for selected in wave['requests']:
            if selected['adapter_id'] not in seen:
                warm.append(selected)
                seen.add(selected['adapter_id'])
    warm = warm[:max(len(w['requests']) for w in templates)]
    waves = [dict(role='kernel_warmup_retained', round=-1, source='gpu', requests=warm)]
    for r in range(rounds):
        for wave in templates[r % len(templates):]+templates[:r % len(templates)]:
            shift = r % len(wave['requests'])
            selected = wave['requests'][shift:]+wave['requests'][:shift]
            waves.append(dict(wave, role='representative_measurement', round=r, requests=selected))
    return waves


def capacity_profile_waves(population, bins, *, capacity, rounds=3):
    """Remeasure changed-runtime classes using indexes, never synthetic inputs.

    Homogeneous length/rank waves expose every actual admission bin. Within a
    class, longer declared outputs keep the measured lanes alive; ties retain
    original trace order. Cover each available exact content before extra lanes.
    This is controlled calibration, not a representative workload or SLO ruler.
    Actual observations, not the intended lane numbers, determine coverage.
    """
    from collections import defaultdict
    from faaslora.experiment.instance_pool import ServiceClassBins
    if (type(capacity) is not int or capacity < 1 or type(rounds) is not int
            or rounds < 1 or not population):
        raise ValueError('positive capacity/rounds and existing population required')
    classes = ServiceClassBins(**{k:tuple(v) for k,v in bins.items()})
    if classes.admitted_requests[-1] != capacity:
        raise ValueError('profile bins must end at the actual capacity')
    groups = defaultdict(list)
    for row in population:
        k = classes.classify(**row['class_features'])
        groups[(k.prompt_bin,k.output_limit_bin,k.rank_bin,k.footprint_bin)].append(row)
    templates = []
    minimum = 1 if len(classes.admitted_requests) == 1 else classes.admitted_requests[-2]+1
    for key, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda r:(-r['class_features']['declared_output_tokens'],r['source_index']))
        chosen, adapters, contents = [], set(), set()
        for content_pass in (True,False):
            for row in ordered:
                if (row['adapter_id'] in adapters or
                        (content_pass and row['content_sha256'] in contents)):
                    continue
                chosen.append(row)
                adapters.add(row['adapter_id']); contents.add(row['content_sha256'])
                if len(chosen) == capacity:
                    break
            if len(chosen) == capacity:
                break
        if len(chosen) < minimum:
            raise ValueError(f'insufficient existing distinct adapters for capacity class {key}')
        templates.append(dict(intended_length_rank_class=list(key),
            requests=[dict(source_request_id=r['source_request_id'],adapter_id=r['adapter_id']) for r in chosen]))
    selected_ids = {r['adapter_id'] for t in templates for r in t['requests']}
    if {r['content_sha256'] for r in population if r['adapter_id'] in selected_ids} != \
            {r['content_sha256'] for r in population}:
        raise ValueError('existing wave selection misses an exact content class')
    warm, seen = [], set()
    for template in templates:
        for row in template['requests']:
            if row['adapter_id'] not in seen:
                warm.append(row); seen.add(row['adapter_id'])
    waves = [dict(role='kernel_warmup_retained',round=-1,source='gpu',requests=warm[:capacity])]
    sources = ('remote','nvme','file_host','native_host','gpu')
    for r in range(rounds):
        for group, template in enumerate(templates):
            requests = template['requests']
            shift = r % len(requests)
            for pos in range(len(sources)):
                waves.append(dict(template,role='representative_measurement',round=r,
                    source=sources[(pos+r+group)%len(sources)],requests=requests[shift:]+requests[:shift]))
    return waves


def gpu_short_gap_waves(population, bins, missing, *, rounds=3):
    """Existing long requests precede short targets; actual counts remain measured.

    The background count is the missing admission bin's lower boundary, not a
    new runtime parameter. This does not hold completed requests open or impose
    a barrier. Fail if the existing population cannot supply distinct adapters.
    """
    from dataclasses import asdict
    from faaslora.experiment.instance_pool import ServiceClassBins
    if not population or not missing or type(rounds) is not int or rounds < 1:
        raise ValueError('existing population, missing classes and positive rounds required')
    classes = ServiceClassBins(**{k:tuple(v) for k,v in bins.items()})
    identities = [json.dumps(k,sort_keys=True) for k in missing]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate missing class')
    admission_bins = {k['admitted_bin'] for k in missing}
    if (len(admission_bins) != 1 or any(k['tier'] != 'gpu' or k['output_limit_bin'] != 0
            or k['admitted_bin'] <= 0 or k['admitted_bin'] >= len(classes.admitted_requests)
            for k in missing)):
        raise ValueError('only measured GPU short-output admission gaps supported')
    admitted_bin = next(iter(admission_bins))
    background_count = classes.admitted_requests[admitted_bin-1]
    if background_count+len(missing) > classes.admitted_requests[admitted_bin]:
        raise ValueError('targets exceed missing admission bin capacity')
    ordered = sorted(population,key=lambda r:(-r['class_features']['declared_output_tokens'],r['source_index']))
    targets, seen = [], set()
    for wanted in sorted(missing,key=lambda k:json.dumps(k,sort_keys=True)):
        choices = []
        for row in ordered:
            if row['adapter_id'] in seen:
                continue
            features = dict(row['class_features'],tier='gpu',representation=wanted['representation'],
                            admitted_after_accept=background_count+1)
            if asdict(classes.classify(**features)) == wanted:
                choices.append(row)
        if not choices:
            raise ValueError('missing existing distinct short target')
        targets.append(choices[0]); seen.add(choices[0]['adapter_id'])
    background = []
    for row in ordered:
        if (row['adapter_id'] in seen
                or row['class_features']['declared_output_tokens'] <= classes.declared_output_tokens[0]):
            continue
        background.append(row); seen.add(row['adapter_id'])
        if len(background) == background_count:
            break
    if len(background) != background_count:
        raise ValueError('insufficient existing distinct long background requests')
    ref = lambda r:dict(source_request_id=r['source_request_id'],adapter_id=r['adapter_id'])
    background_refs, target_refs = list(map(ref,background)),list(map(ref,targets))
    waves = [dict(role='kernel_warmup_retained',round=-1,source='gpu',
                  requests=background_refs+target_refs)]
    for r in range(rounds):
        shift = r % len(target_refs)
        order = target_refs[shift:]+target_refs[:shift]
        waves.append(dict(role='representative_measurement',round=r,source='gpu',
            requests=background_refs+order,background_request_count=background_count,
            target_source_request_ids=[q['source_request_id'] for q in order],
            intended_missing_classes=copy.deepcopy(missing),
            actual_admission_coverage_required=True))
    return waves


def prepare_capacity_profile_gaps():
    """D344 supplement only two actual short-GPU classes; no runtime change."""
    root = p.ROOT
    base = root/'paper_results/ieee_tc/p2_backend/d344_capacity_profiles'
    out = base/'gaps1'
    raw = root/'results/ieee_tc/p2_backend_qualification/d344_20261008/gaps1'
    if out.exists() or raw.exists():
        raise FileExistsError('D344 gap preparation is append-only')
    read = lambda path:json.loads(Path(path).read_text())
    for line in (base/'measurement_outputs.sha256').read_text().splitlines():
        sha,name = line.split('  ',1)
        assert p.digest(base/name) == sha
    coverage, checkpoint = read(base/'coverage1.json'),read(base/'measurement_checkpoint.json')
    assert checkpoint['passed'] and checkpoint['measurement_complete'] and not checkpoint['profile_complete']
    assert coverage['measured_service_classes'] == 198 and coverage['required_service_classes'] == 200
    assert not coverage['missing_content_sources'] and len(coverage['missing_service_classes']) == 2
    source_spec = base/'source_spec.json'
    assert p.digest(source_spec) == '2e8be28b5031075dec4560a68452b7aaee9887ee56db7f0f689d005f401e28b5'
    spec = read(source_spec)
    waves = gpu_short_gap_waves(spec['representatives'],spec['bins'],coverage['missing_service_classes'])
    trace = Path('/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/'
        '20260424_131528_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1/shared_artifacts/'
        'llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_trace.json')
    plan = FrozenReplayPlan.load(trace,count=1000)
    assert plan.source_sha256 == spec['trace_sha256']
    entries = {e.request_id:json.loads(e.source_json) for e in plan.entries}
    known = {r['source_request_id']:r for r in spec['representatives']}
    for wave in waves:
        for ref in wave['requests']:
            row = entries[ref['source_request_id']]
            assert row['adapter_id'] == ref['adapter_id']
            assert min(row['expected_output_tokens'],256) == known[ref['source_request_id']]['class_features']['declared_output_tokens']
    spec.update(purpose='D344_only_two_missing_GPU_short_high_admission_classes',
        nvme_parent=str(raw.resolve()),waves=waves,
        derivation='Retain198/200measured C14classes; lower admitted-bin edge long backgrounds then two existing short targets. No artificial wait; native events determine coverage.',
        order_rule='Existing first1000/D344representatives only; descending declared length/index; alternate short-target order after long backgrounds for3rounds.',
        supplementation=dict(source_run=checkpoint['source_run'],
            coverage=dict(path=str(base/'coverage1.json'),sha256=p.digest(base/'coverage1.json')),
            missing_classes=coverage['missing_service_classes'],capacity_changed=False))
    config_path = root/'configs/ieee_tc/20261008_d343_7b_capacity_probe14.yaml'
    source = yaml.safe_load(config_path.read_text())['model_profiles']['llama2_7b_main_v2_publicmix']
    cfg = dict(source['model'])
    cfg.update(visible_device_ids=[0],device_id=0,timing_contract='ieee_tc_native_v1',
        ieee_worker_observation=True,ieee_gpu_references=True,ieee_scheduler_observation=True,
        ieee_input_upper_bounds=[759],generation_contract='fixed_length_greedy_v1',
        canonical_prompt_renderer='role_lines_v1',max_input_len=759,max_output_tokens_cap=256,
        ieee_physical_allocation=True)
    cfg.update(spec['model_overrides'])
    runtime = root/'results/ieee_tc/p2_backend_qualification/runtime_20260926/cuda_import_attempt3.json'
    _,length_evidence,assembly = p.prepare_admission_source_runtime(cfg,spec,
        backend_version=read(runtime)['modules']['vllm']['version'],runtime_receipt_sha256=p.digest(runtime),
        source_trace_sha256=plan.source_sha256)
    contract = read(root/spec['full_development_contract']['path'])
    assert assembly['planned_child_model_config'] == contract['model_config']
    assert p.digest(config_path) == contract['source_config']['sha256']
    out.mkdir(parents=True,exist_ok=False); raw.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        with (out/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    save('source_spec.json',spec)
    _,_,identities,actual,static = p.source_profile_inputs(out/'source_spec.json',plan,cfg,
        (root/source['storage']['remote_dir']).resolve(strict=True))
    save('launch_config_binding.json',dict(source_config=contract['source_config']))
    result = dict(kind='d344_missing_gpu_class_preparation_v1',passed=True,
        spec_sha256=p.digest(out/'source_spec.json'),waves=len(actual),calls=sum(map(len,actual)),
        measurement_calls=sum(len(w['requests']) for w in waves if w['role']=='representative_measurement'),
        selected_adapters=len(identities),static_adapters=len(static['artifacts']),
        assembly=assembly,length_initialization=length_evidence,missing_classes=coverage['missing_service_classes'],
        source_refs=[dict(path=str(path),sha256=p.digest(path)) for path in
            (source_spec,base/'measurement_outputs.sha256',base/'measurement_checkpoint.json',
             base/'coverage1.json',config_path,runtime,trace,Path(__file__))],
        actual_coverage_pending=True,full_qualified=False,n_correct=None,new_gpu_runs=0,
        plan_sha256=p.check_plan(),metric_protocol_sha256=p.digest(root/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md'))
    save('prepare.json',result)
    print(json.dumps({k:result[k] for k in ('passed','waves','calls','measurement_calls','selected_adapters')}),flush=True)


def prepare_capacity_profiles():
    """D344: same qualified C14 runtime, newly measured timing initialization."""
    root = p.ROOT
    out = root/'paper_results/ieee_tc/p2_backend/d344_capacity_profiles'
    raw = root/'results/ieee_tc/p2_backend_qualification/d344_20261008'
    if out.exists() or raw.exists():
        raise FileExistsError('D344 preparation is append-only')
    read = lambda path:json.loads(Path(path).read_text())
    previous = root/'paper_results/ieee_tc/p2_backend/d343_capacity_probe/cap14'
    for line in (previous/'completed_outputs.sha256').read_text().splitlines():
        sha,name = line.split('  ',1)
        assert p.digest(previous/name) == sha
    spec = read(previous/'source_spec.json')
    config_path = root/'configs/ieee_tc/20261008_d343_7b_capacity_probe14.yaml'
    config = yaml.safe_load(config_path.read_text())['model_profiles']['llama2_7b_main_v2_publicmix']
    history = root/'paper_results/ieee_tc/p2_backend/d321_output_classes/history1'
    manifest = read(history/'manifest.json')
    for name,sha in manifest['outputs'].items():
        assert p.digest(history/name) == sha
    definition = read(history/'candidate_definition.json')
    bins = copy.deepcopy(definition['bins'])
    bins['admitted_requests'] = [1,2,4,8,14]
    history_rows = read(history/'historical_samples.json')['samples']
    known = {r['request_id']:r for r in history_rows}
    assert len(known) == len(history_rows) == 4000
    old_raw = root/'results/ieee_tc/p2_backend_qualification/d343_20261008/cap14/source1.json'
    old = read(old_raw)
    assert old['pass'] and old['physical_allocation']['released']
    assert len(old['requests']) == 80 and all(r['pass'] for r in old['requests'])
    plan = FrozenReplayPlan.load(Path(old['trace']['source_path']), count=1000)
    assert plan.source_sha256 == spec['trace_sha256']
    content_path = root/spec['content_index']['path']
    assert p.digest(content_path) == spec['content_index']['sha256']
    content = read(content_path)
    client = HttpArtifactStoreClient(endpoint='http://127.0.0.1:1')
    client.configure_content_manifest(content)
    pool = (root/config['storage']['remote_dir']).resolve(strict=True)
    identities = {r['id']:client.routing_identity(r['id'],(pool/r['id']/'adapter_config.json').read_bytes())
                  for r in content['artifacts']}
    population = []
    for i,entry in enumerate(plan.entries):
        row = json.loads(entry.source_json)
        features = known[entry.request_id]['class_features']
        identity = identities[row['adapter_id']]
        assert features['declared_output_tokens'] == min(row['expected_output_tokens'],256)
        assert features['adapter_rank'] == identity['rank']
        population.append(dict(source_index=i,source_request_id=entry.request_id,adapter_id=row['adapter_id'],
            content_sha256=identity['content_sha256'],class_features=features))
    waves = capacity_profile_waves(population,bins,capacity=14)
    selected = {r['adapter_id'] for w in waves for r in w['requests']}
    assert {identities[a]['content_sha256'] for a in selected} == {v['content_sha256'] for v in identities.values()}
    spec.update(purpose='D344_C14_affected_service_preparation_calibration_not_warm_SLO',
        nvme_parent=str(raw.resolve()),bins=bins,waves=waves,
        representatives=[r for r in population if r['adapter_id'] in selected],
        derivation='Qualified D343 C14; retain D321 output edges and fixed-output completed lengths only. All timing samples freshly measured under C14; no C8 timing transfer.',
        order_rule='Existing first1000 only; homogeneous output/rank classes, content coverage first, descending declared target then original index, up to14 distinct lanes; three rotated rounds across five sources.')
    spec.pop('bounded_followup_evidence',None)
    cfg = dict(config['model'])
    cfg.update(visible_device_ids=[0],device_id=0,timing_contract='ieee_tc_native_v1',
        ieee_worker_observation=True,ieee_gpu_references=True,ieee_scheduler_observation=True,
        ieee_input_upper_bounds=[759],generation_contract='fixed_length_greedy_v1',
        canonical_prompt_renderer='role_lines_v1',max_input_len=759,max_output_tokens_cap=256,
        ieee_physical_allocation=True)
    cfg.update(spec['model_overrides'])
    runtime = root/'results/ieee_tc/p2_backend_qualification/runtime_20260926/cuda_import_attempt3.json'
    _,length_evidence,assembly = p.prepare_admission_source_runtime(cfg,spec,
        backend_version=read(runtime)['modules']['vllm']['version'],runtime_receipt_sha256=p.digest(runtime),
        source_trace_sha256=plan.source_sha256)
    from faaslora.experiment.instance_pool import FrozenServiceProfiles
    assert FrozenServiceProfiles.model_identity(assembly['planned_child_model_config']) == \
        FrozenServiceProfiles.model_identity(old['model_config'])
    full_path = root/'configs/ieee_tc/20261008_d336_7b_full_w0.yaml'
    full = yaml.safe_load(full_path.read_text())
    coord = copy.deepcopy(full['resource_coordination'])
    for key in ('ieee_service_profile','ieee_preparation_profile'):
        coord.pop(key)
    coord['ieee_scaling'].update(p.derive_ieee_capacity_targets(14,
        min_instances=coord['min_instances'],max_instances=coord['max_instances']))
    contract = dict(kind='ieee_development_control_runtime_contract_v1',model='7b',development_only=True,
        model_config=assembly['planned_child_model_config'],runtime_receipt_sha256=p.digest(runtime),
        coordination=coord,service_ewma_beta=.5,preparation_ewma_beta=.5,
        completed_length_binding=spec['admission_initialization'],full_qualified=False,
        production_profile_frozen=False,plan_sha256=p.check_plan(),
        source_config=dict(path=str(config_path.relative_to(root)),sha256=p.digest(config_path)),
        controller_rule_source=dict(path=str(full_path.relative_to(root)),sha256=p.digest(full_path)),
        old_timings_relabelled=False,common_warm_reference_changed=False)
    seal = p.verify_seal(root/'paper_results/ieee_tc/safety/20260925_execution_start_protected.json')
    changed = Path('/home/qhq/serverless_llm_baselines/scripts/replay_openai_trace.py')
    assert seal['entry_count'] == 147 and seal['changed'] == [str(changed)]
    assert p.digest(changed) == '04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22'
    out.mkdir(parents=True,exist_ok=False); raw.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        with (out/name).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)
        return dict(path=str((out/name).relative_to(root)),sha256=p.digest(out/name))
    contract_ref = save('development_contract.json',contract)
    spec['full_development_contract'] = contract_ref
    spec_ref = save('source_spec.json',spec)
    _,_,selected_identities,actual,static = p.source_profile_inputs(out/'source_spec.json',plan,cfg,pool)
    final_assembly = p.prepare_admission_source_runtime(cfg,spec,
        backend_version=read(runtime)['modules']['vllm']['version'],runtime_receipt_sha256=p.digest(runtime),
        source_trace_sha256=plan.source_sha256)[2]
    assert final_assembly['full_development_contract'] == contract_ref
    save('launch_config_binding.json',dict(source_config=contract['source_config']))
    receipt = dict(kind='d344_capacity_profile_preparation_v1',passed=True,development_only=True,
        spec=spec_ref,contract=contract_ref,assembly=final_assembly,length_initialization=length_evidence,
        waves=len(actual),cases=sum(map(len,actual)),
        representative_cases=sum(len(w['requests']) for w in waves if w['role']=='representative_measurement'),
        selected_adapters=len(selected_identities),static_adapters=len(static['artifacts']),
        output_edges=bins['declared_output_tokens'],admitted_edges=bins['admitted_requests'],
        actual_coverage_pending=True,n_correct=None,gpu_runs=0,full_qualified=False,
        production_profile_frozen=False,new_trace=False,new_weights=False,protection_raw=seal,
        source_refs=[dict(path=str(path),sha256=p.digest(path)) for path in (old_raw,
            previous/'completed_outputs.sha256',history/'manifest.json',history/'historical_samples.json',
            history/'candidate_definition.json',config_path,full_path,runtime,content_path)],
        plan_sha256=p.check_plan(),metric_protocol_sha256=p.digest(root/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md'))
    save('prepare.json',receipt)
    print(json.dumps(dict(passed=True,waves=receipt['waves'],cases=receipt['cases'],
        selected_adapters=receipt['selected_adapters'],spec=spec_ref)),flush=True)


def prepare_control_profiles():
    """D356: same existing cases, fresh control-policy timing measurements."""
    root = p.ROOT
    out = root/'paper_results/ieee_tc/p2_backend/d356_control_profiles'
    raw = root/'results/ieee_tc/p2_backend_qualification/d356_20261008'
    if out.exists() or raw.exists():
        raise FileExistsError('D356 preparation is append-only')
    base = root/'paper_results/ieee_tc/p2_backend/d344_capacity_profiles'
    refs = [(base/'source_spec.json','2e8be28b5031075dec4560a68452b7aaee9887ee56db7f0f689d005f401e28b5'),
            (base/'gaps1/source_spec.json','a9483c569d52605c10aa6b7bac9cd21949e7758ac87ffafd40b0a7f8ca4088a9')]
    for path,sha in refs:
        assert p.digest(path) == sha
    spec, supplement = [json.loads(path.read_text()) for path,_ in refs]
    spec.pop('full_development_contract')
    spec['waves'].extend(supplement['waves'])
    spec.update(purpose='D356_affected_control_interleave_profiles_not_warm_SLO',
        nvme_parent=str(raw.resolve()),
        derivation='Unchanged C14 plus atomic_control_yield_v1; fresh latency, old completed fixed-output lengths only.',
        order_rule='Reuse exact D344 source and known short-GPU coverage waves including retained warmups; no new inputs.')
    binding = spec['admission_initialization']
    binding['kind'] = 'native_fixed_output_control_length_transfer_v1'
    binding['model_config_changes']['ieee_control_interleave'] = dict(source=None,target='atomic_control_yield_v1')
    config_path = root/'configs/ieee_tc/20261008_d356_7b_control_source.yaml'
    old_path = root/'configs/ieee_tc/20261008_d343_7b_capacity_probe14.yaml'
    candidate,old = [yaml.safe_load(path.read_text()) for path in (config_path,old_path)]
    key = 'llama2_7b_main_v2_publicmix'
    expected = copy.deepcopy(old)
    expected['model_profiles'][key]['model']['ieee_control_interleave'] = 'atomic_control_yield_v1'
    assert candidate == expected
    source = candidate['model_profiles'][key]
    cfg = dict(source['model'])
    cfg.update(visible_device_ids=[0],device_id=0,timing_contract='ieee_tc_native_v1',
        ieee_worker_observation=True,ieee_gpu_references=True,ieee_scheduler_observation=True,
        ieee_input_upper_bounds=[759],generation_contract='fixed_length_greedy_v1',
        canonical_prompt_renderer='role_lines_v1',max_input_len=759,max_output_tokens_cap=256,
        ieee_physical_allocation=True)
    cfg.update(spec['model_overrides'])
    runtime = root/'results/ieee_tc/p2_backend_qualification/runtime_20260926/cuda_import_attempt3.json'
    args = dict(backend_version=json.loads(runtime.read_text())['modules']['vllm']['version'],
                runtime_receipt_sha256=p.digest(runtime),source_trace_sha256=spec['trace_sha256'])
    _,evidence,assembly = p.prepare_admission_source_runtime(cfg,spec,**args)
    previous = json.loads((base/'development_contract.json').read_text())
    before,after = copy.deepcopy(previous['model_config']),copy.deepcopy(assembly['planned_child_model_config'])
    assert after.pop('ieee_control_interleave') == 'atomic_control_yield_v1'
    after_init,before_init = after.pop('ieee_admission_profile'),before.pop('ieee_admission_profile')
    assert after == before
    assert {k:v for k,v in after_init.items() if k not in ('profile_id','model_backend_id')} == \
           {k:v for k,v in before_init.items() if k not in ('profile_id','model_backend_id')}
    assert after_init['profile_id'] != before_init['profile_id']
    assert not evidence['length_transfer']['timing_samples_transferred']
    full_path = root/'configs/ieee_tc/20261008_d353_7b_full_w0.yaml'
    full = yaml.safe_load(full_path.read_text())
    coord = copy.deepcopy(full['resource_coordination'])
    for name in ('ieee_service_profile','ieee_preparation_profile'):
        coord.pop(name)
    contract = dict(previous,model_config=assembly['planned_child_model_config'],coordination=coord,
        completed_length_binding=binding,source_config=dict(path=str(config_path.relative_to(root)),sha256=p.digest(config_path)),
        controller_rule_source=dict(path=str(full_path.relative_to(root)),sha256=p.digest(full_path)),
        old_timings_relabelled=False,common_warm_reference_changed=False)
    trace = Path('/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/'
        '20260424_131528_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1/shared_artifacts/'
        'llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_trace.json')
    plan = FrozenReplayPlan.load(trace,count=1000)
    assert plan.source_sha256 == spec['trace_sha256']
    pool = (root/source['storage']['remote_dir']).resolve(strict=True)
    out.mkdir(parents=True,exist_ok=False);raw.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        with (out/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
        return dict(path=str((out/name).relative_to(root)),sha256=p.digest(out/name))
    spec['full_development_contract'] = save('development_contract.json',contract)
    spec_ref = save('source_spec.json',spec)
    _,_,identities,waves,static = p.source_profile_inputs(out/'source_spec.json',plan,cfg,pool)
    final_assembly = p.prepare_admission_source_runtime(cfg,spec,**args)[2]
    assert len(waves) == 125 and sum(map(len,waves)) == 1569
    save('launch_config_binding.json',dict(source_config=contract['source_config']))
    save('prepare.json',dict(kind='d356_control_profile_preparation_v1',passed=True,
        development_only=True,n_correct=None,gpu_runs=0,full_qualified=False,
        spec=spec_ref,assembly=final_assembly,length_initialization=evidence,
        waves=len(waves),cases=sum(map(len,waves)),representative_cases=1545,warmups=24,
        selected_adapters=len(identities),static_adapters=len(static['artifacts']),
        actual_coverage_pending=True,old_timings_relabelled=False,
        source_refs=[dict(path=str(path),sha256=p.digest(path)) for path in
                     ([x for x,_ in refs]+[config_path,old_path,full_path,runtime])],
        plan_sha256=p.check_plan(),metric_protocol_sha256=p.digest(root/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md')))
    print(json.dumps(dict(passed=True,waves=len(waves),cases=sum(map(len,waves)),spec=spec_ref)))


def prepare_output_class_gaps():
    """D321: reuse D298 collector/model and only fill audited output-class gaps."""
    root = p.ROOT
    history = root/'paper_results/ieee_tc/p2_backend/d321_output_classes/history1'
    out = history.parent/'gaps1'
    raw = root/'results/ieee_tc/p2_backend_qualification/d321_20261007/gaps1'
    if out.exists() or raw.exists():
        raise FileExistsError('gap preparation is append-only')
    read = lambda path:json.loads(Path(path).read_text())
    manifest = read(history/'manifest.json')
    for name, sha in manifest['outputs'].items():
        assert p.digest(history/name) == sha
    definition, coverage = read(history/'candidate_definition.json'), read(history/'coverage.json')
    assert definition['launchable'] is False and coverage['no_default_fills']
    assert coverage['required_classes'] == 160 and coverage['observed_classes'] == 141
    records = read(history/'historical_samples.json')['samples']
    known = {r['request_id']:r for r in records}
    assert len(known) == len(records) == 4000
    source_path = root/'configs/ieee_tc/20261007_d298_7b_capacity_source.yaml'
    spec_path = root/'paper_results/ieee_tc/p2_backend/d298_capacity/source_spec.json'
    source = yaml.safe_load(source_path.read_text())['model_profiles']['llama2_7b_main_v2_publicmix']
    old_spec = read(spec_path)
    old_run = read(root/'paper_results/ieee_tc/p2_backend/d298_capacity/initialization/manifest.json')
    # Bind the existing D298 spec/config through its already published evidence.
    assert p.digest(spec_path) == '0f9d8ccb886a81315634ba925e2cbc25153ce7b2d2d6d306008088125eb03649'
    assert p.digest(source_path) == '6b96df57e5f05a9b4a0b597ef988b464d37af6b61e4dac33b3e4444778d1487f'
    old_source = Path(old_run['source_run']['path'])
    assert p.digest(old_source) == old_run['source_run']['sha256']
    trace = Path(read(old_source)['trace']['source_path'])
    plan = FrozenReplayPlan.load(trace, count=1000)
    assert plan.source_sha256 == old_spec['trace_sha256']
    population = []
    for entry in plan.entries:
        row = json.loads(entry.source_json)
        prior = known[entry.request_id]
        assert prior['class_features']['declared_output_tokens'] == min(row['expected_output_tokens'],256)
        population.append(dict(source_request_id=entry.request_id, adapter_id=row['adapter_id'],
            class_features=prior['class_features']))
    waves = missing_length_waves(population, definition['bins'], coverage['missing_classes'])
    spec = copy.deepcopy(old_spec)
    spec.update(purpose='D321_declared_output_missing_class_measurements_only',
        bins=definition['bins'], waves=waves, nvme_parent=str(raw.resolve()),
        derivation='Retain D298+all4000 D315 completed development samples. Measure19missing classes only; no default fill.',
        order_rule='First distinct existing request/adapter per same length/rank class in original first1000; minimum wave size for largest missing admitted bin;3rounds with deterministic rotation.')
    cfg = dict(source['model'])
    cfg.update(visible_device_ids=[0],device_id=0,timing_contract='ieee_tc_native_v1',
        ieee_worker_observation=True,ieee_gpu_references=True,ieee_scheduler_observation=True,
        ieee_input_upper_bounds=[759],generation_contract='fixed_length_greedy_v1',
        canonical_prompt_renderer='role_lines_v1',max_input_len=759,max_output_tokens_cap=256,
        ieee_physical_allocation=True)
    cfg.update(spec['model_overrides'])
    runtime = root/'results/ieee_tc/p2_backend_qualification/runtime_20260926/cuda_import_attempt3.json'
    _, evidence, assembly = p.prepare_admission_source_runtime(cfg,spec,
        backend_version=read(runtime)['modules']['vllm']['version'],runtime_receipt_sha256=p.digest(runtime),
        source_trace_sha256=plan.source_sha256)
    from faaslora.experiment.instance_pool import FrozenServiceProfiles
    assert FrozenServiceProfiles.model_identity(assembly['planned_child_model_config']) == \
        FrozenServiceProfiles.model_identity(definition['model_config'])
    seal = p.verify_seal(root/'paper_results/ieee_tc/safety/20260925_execution_start_protected.json')
    changed = '/home/qhq/serverless_llm_baselines/scripts/replay_openai_trace.py'
    assert seal['changed'] == [changed] and seal['entry_count'] == 147
    assert p.digest(Path(changed)) == '04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22'
    out.mkdir(parents=True,exist_ok=False)
    raw.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        with (out/name).open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False)
    save('source_spec.json',spec)
    pool = (root/source['storage']['remote_dir']).resolve(strict=True)
    _, _, identities, actual, static = p.source_profile_inputs(out/'source_spec.json',plan,cfg,pool)
    sources = [history/name for name in manifest['outputs']] + [history/'manifest.json', spec_path,
        source_path, old_source, trace, runtime, Path(__file__), root/'scripts/ieee_tc_preflight.py']
    save('prepare.json',dict(kind='d321_missing_class_preparation_v1',pass_inputs=True,
        source_refs=[dict(path=str(path),sha256=p.digest(path)) for path in sources],
        spec_sha256=p.digest(out/'source_spec.json'),waves=len(actual),
        requests=sum(map(len,actual)),measurement_requests=sum(len(w['requests']) for w in waves if w['role']=='representative_measurement'),
        selected_adapters=len(identities),static_adapters=len(static['artifacts']),
        missing_classes=coverage['missing_classes'],assembly=assembly,length_initialization=evidence,
        protection_raw=seal,plan_sha256=p.check_plan(),
        metric_protocol_sha256=p.digest(root/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md'),
        new_trace=False,new_weights=False,new_gpu_runs=0,actual_class_coverage_pending=True))
    print(json.dumps(dict(pass_inputs=True,waves=len(actual),requests=sum(map(len,actual)),
        missing_classes=len(coverage['missing_classes']),selected_adapters=len(identities))),flush=True)


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


def length_candidate_config(previous, service_binding, *, smoke, attempt=1):
    """Only the measured service classes and exclusive storage identities differ."""
    if attempt not in (1, 2) or (attempt == 2 and not smoke):
        raise ValueError('unknown length candidate attempt')
    candidate = copy.deepcopy(previous)
    campaign = 'd322' if smoke else 'd323'
    raw = '/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/' + campaign + '_20261007'
    stem = 'length_validation_100' if smoke else 'full_w0'
    if attempt == 2:
        stem += '_attempt2'
    candidate['experiment']['output_dir'] = raw + '/7b_' + stem
    candidate['storage']['nvme_cache_dir'] = raw + '/7b_nvme_' + stem
    candidate['storage']['host_cache_dir'] = '/dev/shm/tc-' + campaign + '-7b-' + stem.replace('_', '-')
    candidate['scenarios'][0]['name'] = 'tc_ieee_length_validation_100' if smoke else 'tc_ieee_full'
    candidate['resource_coordination']['ieee_service_profile'] = copy.deepcopy(service_binding)
    return candidate


def verify_length_candidate(*, smoke=True, attempt=1):
    from scripts.run_all_experiments import (_prepare_scenario_runtime_model_config,
        _prepare_dedicated_subprocess_model_cfg)
    from faaslora.experiment.instance_pool import FrozenServiceProfiles
    from faaslora.preloading.preloading_planner import FrozenPreparationProfiles
    root = p.ROOT
    campaign = 'd322' if smoke else 'd323'
    stem = 'length_validation_100' if smoke else 'full_w0'
    out = root / ('paper_results/ieee_tc/p2_backend/' + campaign + '_length_candidate')
    if attempt == 2:
        stem += '_attempt2'
        out = out/'attempt2'
    assert not out.exists(), 'append-only preparation'
    config = root / ('configs/ieee_tc/20261007_' + campaign + '_7b_' + stem + '.yaml')
    previous = root / 'configs/ieee_tc/20261007_d318_7b_full_w0.yaml'
    initialization = root / 'paper_results/ieee_tc/p2_backend/d321_output_classes/initialization1'
    manifest_path = initialization / 'manifest.json'
    assert p.digest(manifest_path) == '895bf0da6f9698333555149b344165cec4e9a309b179edc87ad468a9cfaf492d'
    read = lambda path: json.loads(path.read_text())
    manifest = read(manifest_path)
    for name, sha in manifest['outputs'].items():
        assert p.digest(initialization/name) == sha, 'D321 output changed'
    for row in manifest['source_refs']:
        assert p.digest(Path(row['path'])) == row['sha256'], 'D321 evidence changed'
    coverage = read(initialization/'coverage.json')['combined']
    assert coverage['required_classes'] == coverage['observed_classes'] == 160
    binding = read(initialization/'binding.json')['ieee_service_profile']
    binding['path'] = str(Path(binding['path']).relative_to(root))
    c, old = (yaml.safe_load(x.read_text()) for x in (config, previous))
    assert c == length_candidate_config(old, binding, smoke=smoke, attempt=attempt), 'undeclared candidate difference'
    for path in (c['experiment']['output_dir'], c['storage']['nvme_cache_dir'], c['storage']['host_cache_dir']):
        assert not Path(path).exists(), 'run/cache identity already exists'
    contract_path = root/'paper_results/ieee_tc/p2_backend/d298_capacity/development_contract.json'
    configured = _prepare_scenario_runtime_model_config(c['model'], c['workload'], c['resource_coordination'])
    child, assembly = _prepare_dedicated_subprocess_model_cfg(configured, device_id=0, runtime_gpu_ids=[0])
    assert child == read(contract_path)['model_config']
    assert c['workload']['total_requests'] == 4000 and c['model']['max_num_batched_tokens'] == 1024
    profiles = {}
    for key, cls, count in [('ieee_service_profile', FrozenServiceProfiles, 160),
                          ('ieee_preparation_profile', FrozenPreparationProfiles, 24)]:
        b = c['resource_coordination'][key]
        profile = cls.load(root/b['path'], expected_sha256=b['sha256'], model_config=child,
            expected_context=b['context'], beta=b['ewma_beta'])
        assert len(profile.profiles) == count
        profiles[key] = dict(profile_id=profile.profile_id, classes=count, binding=b)
    # Exercise the actual joint runtime entry, not only the two standalone loaders.
    from types import SimpleNamespace
    from scripts.run_all_experiments import ScenarioRunner
    bound = ScenarioRunner.__new__(ScenarioRunner)
    bound.model_cfg, bound.engine = child, SimpleNamespace(model_cfg=child)
    bound.coord_cfg = c['resource_coordination']
    bound._service_profiles = bound._load_ieee_service_profiles()
    assert len(bound._load_ieee_preparation_profiles().profiles) == 24
    refs = [config, previous, manifest_path, contract_path, Path(__file__),
        root/'scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh', root/'scripts/run_all_experiments.py']
    if not smoke:
        smoke_path = root/'paper_results/ieee_tc/p2_backend/d322_length_candidate/smoke2/audit.json'
        checked = read(smoke_path)
        assert checked['stats']['completed'] == 100
        assert checked['copy_retention_trigger']['observed_cumulative_retained_scopes_max'] > 0
        for row in checked['raw_evidence']:
            assert p.digest(Path(row['path'])) == row['sha256'], 'smoke evidence changed'
        smoke_config = yaml.safe_load((root/'configs/ieee_tc/20261007_d322_7b_length_validation_100_attempt2.yaml').read_text())
        assert c['model'] == smoke_config['model'] and c['resource_coordination'] == smoke_config['resource_coordination']
        refs.append(smoke_path)
    seal = p.verify_seal(root/'paper_results/ieee_tc/safety/20260925_execution_start_protected.json')
    known = Path('/home/qhq/serverless_llm_baselines/scripts/replay_openai_trace.py')
    assert seal['changed'] == [str(known)] and seal['entry_count'] == 147
    assert p.digest(known) == '04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22'
    assert p.check_plan() == '0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    assert p.digest(root/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md') == '5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    assert p.digest(root/'configs/generated/lora_manifest_1000.json') == '3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df'
    receipt = dict(kind=campaign+'_length_candidate_binding_v1', passed=True,
        profiles=profiles, model_config=child, assembly=assembly, protection_raw=seal,
        source_references=[dict(path=str(x), sha256=p.digest(x)) for x in refs],
        diagnostic_prefix_count=100 if smoke else None, attempt=attempt, formal_qualification=False,
        n_correct=None, no_figures=True, only_service_initialization_changed=True)
    out.mkdir(parents=True, exist_ok=False)
    with (out/'prelaunch_binding.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=True, campaign=campaign, classes={k:v['classes'] for k,v in profiles.items()})))


if __name__ == '__main__':
    if sys.argv[1:] == ['--control-profiles']:
        prepare_control_profiles()
    elif sys.argv[1:] == ['--capacity-profile-gaps']:
        prepare_capacity_profile_gaps()
    elif sys.argv[1:] == ['--capacity-profiles']:
        prepare_capacity_profiles()
    elif sys.argv[1:] == ['--capacity-probe']:
        prepare_capacity_probe()
    elif sys.argv[1:] == ['--capacity-probe-followup']:
        prepare_capacity_probe(followup=True)
    elif sys.argv[1:] == ['--verify-length-smoke']:
        verify_length_candidate()
    elif sys.argv[1:] == ['--verify-length-smoke-attempt2']:
        verify_length_candidate(attempt=2)
    elif sys.argv[1:] == ['--verify-length-full']:
        verify_length_candidate(smoke=False)
    elif sys.argv[1:] == ['--output-class-gaps']:
        prepare_output_class_gaps()
    elif sys.argv[1:] == ['--prefill-candidate']:
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
