"""Append-only D1113 reconciliation; reuses existing native/physical analyzers.

CPU-only, no serving, no new workload, no historical result mutation.  Legacy
replays remain diagnostics even when their prompt/target maps match exactly.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
RAW = Path('/home/qhq/serverless_llm_experiment/results/ieee_tc')
BASE = Path('/home/qhq/serverless_llm_baselines')
OUT = Path(__file__).resolve().parent / 'result1'
sys.path.insert(0, str(ROOT))
from scripts.ieee_tc.plot_d290_diagnostic import sha256
from scripts.ieee_tc.check_d299_full_context import (
    closed_physical_integral, common_slo_flags, matched_controller_rows,
    matched_request_contract)
from scripts.ieee_tc.audit_d402_serverlessllm_path import timing_stats
from faaslora.metrics.metrics_collector import NativeV1TokenTimeline

REFS = {}


def read(path, expected=None):
    path = Path(path)
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError(f'changed source: {path}')
    REFS[str(path)] = digest
    return json.loads(path.read_text())


def witness(path, expected=None):
    path = Path(path)
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError(f'changed source: {path}')
    REFS[str(path)] = digest
    return path


def one(paths):
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f'explicit unique source required: {paths}')
    return paths[0]


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    run = RAW/'p2_backend_qualification/d1100_20261107/7b_busy_service_rate_w2'
    launch = read(run/'launch.json')
    physical_dir = run/'launch.launch/physical_deployment'
    physical = read(physical_dir/'physical_resource_summary.json')
    summary = read(physical_dir/'summary.json', physical['summary_sha256'])
    source = witness(one(run.glob('experiment_results_full*.json')),
                     physical['result_file']['sha256'])
    for key in ('pass', 'service_path_removed', 'native_gpu_context_release_confirmed'):
        assert launch[key] is True
    for record in summary['allocation_journals']:
        assert record['released'] is True
        witness(record['path'], record['sha256'])
    witness(summary['request_terminals_path'], summary['request_terminals_sha256'])
    gpu = closed_physical_integral(summary)
    ext = launch['external_replay']
    assert abs(ext['replay_t0_s']-ext['deployment_notice_s']-60) < 1e-6
    index = read(ROOT/'paper_results/ieee_tc/inputs/20260927_7b_materialized_content_index.json')
    content = {a['id']: hashlib.sha256(json.dumps(sorted(a['files'], key=lambda f:f['path']),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest() for a in index['artifacts']}
    assert len(content) == 500
    rows = matched_controller_rows(source, 'tc_ieee_full', content)
    freeze = read(ROOT/'paper_results/ieee_tc/protocols/20261010_7b_slo_d394_frozen.json',
                  '2d906f8aaf3fa1bf8cf30d116990db9970fb7be7bf1374fdbd1d8995090104f3')
    witness(ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md', freeze['metric_protocol_sha256'])
    witness(ROOT/'docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md', freeze['single_run_amendment_sha256'])
    witness(ROOT/freeze['calibration']['path'], freeze['calibration']['sha256'])
    counts = Counter()
    metrics = {k: [] for k in ('ttft_ms', 'tpot_ms', 'e2e_ms')}
    for r in rows.values():
        n = r['native']
        NativeV1TokenTimeline.service_breakdown(n,
            admitted_at=n['controller_admitted_monotonic_s'], completed_at=r['completed'],
            clock_id=summary['clock_id'])
        r['arrival_offset_s'] = r['arrival']-ext['replay_t0_s']
        f, last = n['native_first_token_monotonic_s'], n['native_last_token_monotonic_s']
        assert r['arrival'] <= r['submit'] <= r['native_entry_s'] <= f <= last <= r['completed']
        assert n['native_output_tokens'] == r['target_tokens'] and n['native_terminal_observed']
        values = dict(ttft_ms=1000*(f-r['arrival']),
            tpot_ms=1000*(last-f)/(r['target_tokens']-1) if r['target_tokens']>1 else None,
            e2e_ms=1000*(r['completed']-r['arrival']))
        for k, v in values.items():
            assert (r['stored'][k] is None if v is None else abs(v-r['stored'][k]) <= 1)
            if v is not None:
                metrics[k].append(v)
        counts.update(common_slo_flags(freeze['groups'], prompt_tokens=r['native_prompt_tokens'],
            output_tokens=r['target_tokens'], ttft_ms=values['ttft_ms'], tpot_ms=values['tpot_ms']))
    prime = dict(run='D1100', system='PrimeLoRA', requests=len(rows), counts=dict(counts),
        metrics={k:timing_stats(v) for k,v in metrics.items()}, physical_gpu_seconds=gpu,
        gpu_seconds_per_offered_request=gpu/4000, n_correct=summary['n_correct'])
    registration = read(ROOT/'paper_results/ieee_tc/main_comparison/d593_complete_7b/source_registration.json')
    old = []
    for key in ('D555', 'D557', 'D591'):
        item = one(x for x in registration if x['run'] == key)
        audit = read(item['path'], item['sha256'])
        source_refs = {x['path']:x['sha256'] for x in audit['raw_sources']}
        replay = witness(audit['replay_path'], audit['replay_sha256'])
        cfg_path = one(Path(p) for p in source_refs if p.endswith('/http_7b_4000.json'))
        cfg = read(cfg_path, source_refs[str(cfg_path)])
        p = one(Path(p) for p in source_refs if p.endswith('/physical/summary.json'))
        ledger = read(p, source_refs[str(p)])
        witness(ledger['request_terminals_path'], ledger['request_terminals_sha256'])
        receipt = read(replay.parent/'exec_receipt.json')
        launch_path = replay.parent.with_suffix('.json')
        old_launch = read(launch_path)
        oldext = receipt['external_replay']
        assert oldext['plan'] == ext['plan'] == audit['source_plan']
        assert old_launch['preflight']['policy'] == launch['preflight']['policy']
        assert abs(oldext['replay_t0_s']-oldext['deployment_notice_s']-60) < 1e-6
        assert cfg['generation_contract'] == 'fixed_length_greedy_v1'
        assert cfg['ingress_mode'] == 'service_pre_ready_v1' and cfg['formal_run'] is False
        assert cfg['artifact_source'] == 'published_gzip_ondemand_peft_v1'
        contracts, natives, arrivals = {}, {}, {}
        with replay.open() as stream:
            for line in stream:
                record = json.loads(line)
                rid = record.get('request_id')
                if record['event'] == 'request_contract':
                    assert rid not in contracts
                    contracts[rid] = record
                elif record['event'] == 'http_raw_response':
                    n = record['body']['metrics']['ieee_tc']
                    assert rid not in natives
                    natives[rid] = (len(n['native_prompt_token_ids']), n['completion_token_ids_sha256'])
                elif record['event'] == 'http_response':
                    assert rid not in arrivals
                    arrivals[rid] = record['planned_arrival_s']-oldext['replay_t0_s']
        assert set(contracts) == set(natives) == set(arrivals) == set(rows)
        for rid, c in contracts.items():
            c.update(native_prompt_tokens=natives[rid][0], generation_contract=cfg['generation_contract'],
                     arrival_offset_s=arrivals[rid])
            matched_request_contract(rows[rid], c)
        old_gpu = closed_physical_integral(ledger)
        assert abs(old_gpu-audit['physical_gpu_seconds']) < 1e-6
        for j in ledger['allocation_journals']:
            assert j['released']
            witness(j['path'], j['sha256'])
        old.append(dict(run=key, system=item['system'], reuse_level='R0',
            reuse_scope='retained development native-binding timing/resource observation',
            request_contract_matches=4000, source_plan_matches=True, resource_policy_matches=True,
            notice_to_arrival_seconds=60, common_slo=audit['common_slo'],
            physical_gpu_seconds=old_gpu, gpu_seconds_per_offered_request=old_gpu/4000,
            metrics=audit['conditional_metrics'], n_correct=ledger['n_correct'],
            output_digest_differences=sum(natives[r][1] != rows[r]['native']['completion_token_ids_sha256'] for r in rows),
            limitations=audit['limitations']))
    legacy = []
    specs = [('D1103','16_serverlessllm_new_remote_w2_v1/d1103*'),
             ('D1106','19_vllm_w2_v3/d1106_20261107_vllm_w2'),
             ('D1108','20_slora_w2_v1/d1108_20261107_slora_w2_mem100')]
    for key, glob in specs:
        directory = one((BASE/'results/paper_experiments').glob(glob))
        replay = read(one(directory.glob('raw/replay/*_replay.json')))
        reported = read(one(directory.glob('raw/replay/*_summary.json')))
        values = replay['results']
        assert len(values) == len({r['request_id'] for r in values}) == 4000
        mismatch = {k:sum(r[k] != rows[r['request_id']][v] for r in values) for k,v in
            [('canonical_prompt_sha256','canonical_prompt_sha256'),
             ('requested_completion_tokens','target_tokens'), ('adapter_id','adapter_id')]}
        reasons = ['arrival clock is process_start, not the common deployment-notice + 60 s clock',
                   'TPOT definition ends at completion, not the last native token',
                   'reported cost is simulated lifecycle with a 300 s idle tail, not physical owner-union',
                   '5,000 ms TTFT-only attainment is not the frozen joint TTFT/TPOT event']
        if key == 'D1106':
            reasons.append('vLLM 0.10.2 is not the common 0.30.0 reference backend')
        if key == 'D1108':
            a = read(directory/'D1108_AUDIT.json')
            reasons.append('recorded 95/100 GiB high/max differs from common 72/80 GiB')
        legacy.append(dict(run=key, reuse_level='R2', main_comparison_eligible=False,
            reasons=reasons, input_map_mismatches=mismatch, requests=4000,
            replay_time_origin=replay['replay_time_origin'],
            excluded_client_prewarm_seconds=replay['client_prewarm_sec_excluded_from_workload_clock'],
            metric_definitions=replay['metric_definitions'],
            reported_cost_model=reported['metadata']['infra_cost_model'],
            resource_envelope_reported=a['resource_envelope'] if key=='D1108' else None,
            native_last_timestamp_in_server_metrics=sum('last_token_at' in r.get('server_metrics',{}) for r in values)))
    for name in ('run_vllm_fair_experiment.sh','run_slora_fair_experiment.sh','run_serverlessllm_fair_experiment.sh'):
        witness(BASE/'scripts'/name)
    witness(Path(__file__))
    witness(ROOT/'scripts/ieee_tc/check_d299_full_context.py')
    witness(ROOT/'scripts/ieee_tc/plot_d290_diagnostic.py')
    result = dict(kind='d1113_contract_reconciliation_v1', prime=prime,
        reused_baselines=old, legacy_diagnostics=legacy, no_gpu_run=True,
        no_independent_review=True, final_ranking_eligible=False,
        limitations=['Known seed42 development observations, not unseen holdout or statistical inference.',
                     'No post-hoc startup subtraction, clock shift or synthetic physical GPU accounting.',
                     'Unknown independent numerical adapter correctness is not imputed.',
                     'Input match does not prove equal OS cache state or absence of remote interference.',
                     'Same-backend changed output digests are recorded, not dismissed as rounding.'])
    OUT.mkdir(parents=True, exist_ok=False)
    with (OUT/'reconciliation.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    with (OUT/'manifest.json').open('x') as stream:
        json.dump(dict(sources=REFS, outputs={'reconciliation.json':sha256(OUT/'reconciliation.json')}),
                  stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(output=str(OUT), prime=prime,
        reused=[dict(run=r['run'], matched=r['request_contract_matches'],
                     output_differences=r['output_digest_differences']) for r in old],
        diagnostics=[dict(run=r['run'], mismatches=r['input_map_mismatches']) for r in legacy]), indent=2))


if __name__ == '__main__':
    main()
