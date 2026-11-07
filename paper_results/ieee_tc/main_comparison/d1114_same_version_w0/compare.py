"""Reuse D476's contract join and D593's sealed rows; no inference or tuning."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
OUT = Path(__file__).resolve().parent / 'result1'
sys.path.insert(0, str(ROOT))
from scripts.ieee_tc.check_d299_full_context import (
    matched_controller_rows, matched_request_contract, matched_relative_observation,
    sealed_matrix_row)
from scripts.ieee_tc.plot_d290_diagnostic import sha256

refs = {}


def read(path, expected=None):
    path = Path(path)
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError(f'changed source: {path}')
    refs[str(path)] = digest
    return json.loads(path.read_text())


def main():
    output = OUT / 'comparison.json'
    if output.exists():
        raise FileExistsError(output)
    review = read(OUT / 'evidence_review.json')
    metric = read(OUT / 'reconciliation.json')['prime']
    assert review['counts']['joint_pass'] == metric['counts']['joint_pass']
    assert review['physical_gpu_seconds'] == metric['physical_gpu_seconds']
    source = Path(review['source'])
    refs[str(source)] = sha256(source)
    assert refs[str(source)] == review['source_sha256']
    launch = read(source.parent / 'launch.json')
    old = read(ROOT / 'paper_results/ieee_tc/main_comparison/d513_published_7b_w0/comparison.json',
               '044c04d6bb5b894761628146a65de248a53436cdd4ba23346837eccb3473b17b')
    index = read(ROOT / 'paper_results/ieee_tc/inputs/20260927_7b_materialized_content_index.json')
    content = {a['id']: hashlib.sha256(json.dumps(sorted(a['files'], key=lambda f: f['path']),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest() for a in index['artifacts']}
    rows = matched_controller_rows(source, 'tc_ieee_full', content)
    for r in rows.values():
        r['arrival_offset_s'] = r['arrival'] - launch['external_replay']['replay_t0_s']
    keys = ('request_id', 'source_item_sha256', 'adapter_id', 'target_tokens',
            'canonical_prompt_sha256', 'native_prompt_token_ids_sha256', 'native_prompt_tokens',
            'artifact_content_sha256', 'generation_contract')
    joined = [{k: rows[rid][k] for k in keys} for rid in sorted(rows)]
    digest = hashlib.sha256(json.dumps(joined, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert digest == old['matched_contract_sha256']
    assert launch['preflight']['policy'] == old['common_resource_policy']
    assert all(metric['replay_plan'][k] == v for k, v in old['common_input'].items())
    registration = read(ROOT / 'paper_results/ieee_tc/main_comparison/d593_complete_7b/source_registration.json')
    baselines = []
    for record in registration:
        if record['workload'] != 'W0' or record['system'] == 'PrimeLoRA':
            continue
        payload = read(record['path'], record['sha256'])
        baselines.append(sealed_matrix_row(payload, source_kind=record['source_kind'],
            system=record['system'], run=record['run'], workload='W0', slo_sha=metric['common_slo_sha256']))
    # Preserve same-backend output differences rather than assuming rounding.
    audit = read(ROOT / 'paper_results/ieee_tc/baseline_audit/d423_7b_full_w0/audit.json')
    replay = Path(audit['replay_path'])
    refs[str(replay)] = sha256(replay)
    assert refs[str(replay)] == audit['replay_sha256']
    receipt = read(replay.parent / 'exec_receipt.json')
    ext = receipt['external_replay']
    assert abs(ext['replay_t0_s'] - ext['deployment_notice_s'] - 60) < 1e-6
    assert all(ext['plan'][k] == v for k, v in old['common_input'].items())
    configs = [r for r in audit['raw_sources'] if r['path'].endswith('/http_7b_4000.json')]
    assert len(configs) == 1
    cfg = read(configs[0]['path'], configs[0]['sha256'])
    assert cfg['generation_contract'] == 'fixed_length_greedy_v1'
    contracts, natives, arrivals = {}, {}, {}
    with replay.open() as stream:
        for line in stream:
            x = json.loads(line)
            if x['event'] == 'request_contract':
                contracts[x['request_id']] = x
            elif x['event'] == 'http_raw_response':
                natives[x['request_id']] = x['body']['metrics']['ieee_tc']
            elif x['event'] == 'http_response':
                arrivals[x['request_id']] = x['planned_arrival_s'] - ext['replay_t0_s']
    assert set(contracts) == set(natives) == set(arrivals) == set(rows)
    changed = []
    for rid, c in contracts.items():
        c.update(native_prompt_tokens=len(natives[rid]['native_prompt_token_ids']),
                 generation_contract=cfg['generation_contract'], arrival_offset_s=arrivals[rid])
        matched_request_contract(rows[rid], c)
        if natives[rid]['completion_token_ids_sha256'] != rows[rid]['native']['completion_token_ids_sha256']:
            changed.append(rid)
    prime = dict(system='PrimeLoRA', run='D1114', workload='W0', counts=review['counts'],
        joint_slo=review['timing_joint_rate'], observed_protocol_feasible=review['timing_feasible'],
        physical_gpu_seconds=review['physical_gpu_seconds'], metrics=review['metrics'],
        gpu_seconds_per_offered_request=review['gpu_seconds_per_offered'],
        raw_n_correct=None, independent_numerical_proof=False)
    result = dict(kind='d1114_reused_w0_observation_v1', prime=prime, baselines=baselines,
        matched_contract_sha256=digest, matched_request_count=4000,
        differences={b['system']: matched_relative_observation(prime, b) for b in baselines},
        same_backend_output_audit=dict(reference='D423', changed_request_ids=sorted(changed),
            changed_count=len(changed), independent_numerical_proof=False),
        source_refs=refs, no_new_baseline_run=True, formal_m1_m2_complete=False,
        caveats=['Known fixed-workload single observations, not independent holdout or significance.',
                 'Baseline mechanism/numerical limits remain those of the sealed source audits.',
                 'Output-digest differences have no independently established numerical cause.',
                 'W0 is not the required W1/W2 budget-constrained G2 result.'])
    refs[str(Path(__file__))] = sha256(Path(__file__))
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(output), matched=4000, output_changes=len(changed),
                          differences=result['differences']), indent=2))


if __name__ == '__main__':
    main()
