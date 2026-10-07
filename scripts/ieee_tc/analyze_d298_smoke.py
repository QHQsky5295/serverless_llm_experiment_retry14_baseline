"""D298/D302/D308 data-only Full100 audit; existing native/binding/SLO checks."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from scripts import ieee_tc_preflight as p
from scripts.ieee_tc.plot_d290_diagnostic import (
    audit_run, compare_diagnostics, write_binding_audit, sha256)
from scripts.analyze_control_path_overhead import provisional_timing_bound, summarize_admission_waits

ROOT = p.ROOT
RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d298_20261007'
RUN = RAW/'7b_capacity_validation_100'
SOURCE = RUN/'experiment_results_full_vllm_dedicated_a500_r100_c2_tc_ieee_prefetch_validation_100_d298_7b_capacity_validation_100.json'
REFERENCE = ROOT/'results/ieee_tc/p2_backend_qualification/d296_20261007/7b_lifecycle_validation_100/experiment_results_full_vllm_dedicated_a500_r100_c2_tc_ieee_prefetch_validation_100_d296_7b_lifecycle_validation_100.json'
OUT = ROOT/'paper_results/ieee_tc/p2_backend/d298_capacity/smoke1'
INDEX = ROOT/'paper_results/ieee_tc/inputs/20260927_7b_materialized_content_index.json'
WARM = ROOT/'paper_results/ieee_tc/p2_backend/20261002_d170_warm_measurement.json'


def read(path):
    return json.loads(path.read_text())


def main(campaign='d298'):
    global RAW, RUN, SOURCE, REFERENCE, OUT
    health_name = 'smoke_remote_health2.json'
    comparison_caveat = 'D298 changes the measured capacity package versus D296, not a single-factor ablation.'
    if campaign == 'd302':
        REFERENCE = SOURCE
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d302_20261007'
        RUN = RAW/'7b_frontier_validation_100'
        SOURCE = RUN/'experiment_results_full_vllm_dedicated_a500_r100_c2_tc_ieee_frontier_validation_100_d302_7b_frontier_validation_100.json'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d302_frontier/smoke1'
        health_name = 'remote_health.json'
        comparison_caveat = ('D298 prefix is reused for descriptive context only: D302 also inherits D300 scale-in lower reference. '
            'Full W0 versus D300 is still required to isolate the publication fix; prefix GPU-s is not extrapolated.')
    elif campaign == 'd308':
        REFERENCE = ROOT/'results/ieee_tc/p2_backend_qualification/d302_20261007/7b_frontier_validation_100/experiment_results_full_vllm_dedicated_a500_r100_c2_tc_ieee_frontier_validation_100_d302_7b_frontier_validation_100.json'
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d308_20261007'
        RUN = RAW/'7b_prefill_validation_100'
        SOURCE = RUN/'experiment_results_full_vllm_dedicated_a500_r100_c2_tc_ieee_prefill_validation_100_d308_7b_prefill_validation_100.json'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d308_prefill/smoke1'
        health_name = 'remote_health.json'
        comparison_caveat = ('D308 uses D303 controls with a 512-token iteration budget, newly measured D307 initialization timings '
            'and the necessary atomic source-pin correctness repair. Not a pure token-budget causal ablation; '
            'Full W0 is required; prefix GPU-s is not extrapolated.')
    elif campaign == 'd310':
        REFERENCE = ROOT/'results/ieee_tc/p2_backend_qualification/d308_20261007/7b_prefill_validation_100/experiment_results_full_vllm_dedicated_a500_r100_c2_tc_ieee_prefill_validation_100_d308_7b_prefill_validation_100.json'
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d310_20261007'
        RUN = RAW/'7b_prefill_validation_100'
        SOURCE = RUN/'experiment_results_full_vllm_dedicated_a500_r100_c2_tc_ieee_prefill_validation_100_d310_7b_prefill_validation_100.json'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d310_read_pin_validation/smoke1'
        health_name = 'remote_health.json'
        comparison_caveat = ('D308 settings and measured profiles unchanged; source read-pin repair only. '
            'One prefix run, not complete W0, numerical or formal SLO qualification.')
    elif campaign == 'd312':
        REFERENCE = ROOT/'results/ieee_tc/p2_backend_qualification/d310_20261007/7b_prefill_validation_100/experiment_results_full_vllm_dedicated_a500_r100_c2_tc_ieee_prefill_validation_100_d310_7b_prefill_validation_100.json'
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d312_20261007'
        RUN = RAW/'7b_prefill_validation_100'
        SOURCE = RUN/'experiment_results_full_vllm_dedicated_a500_r100_c2_tc_ieee_prefill_validation_100_d312_7b_prefill_validation_100.json'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d312_host_validation/smoke1'
        health_name = 'remote_health.json'
        comparison_caveat = ('D310 settings and measured profiles unchanged; native HOST file observation/pin transaction repair only. '
            'One prefix run, not complete W0, numerical or formal SLO qualification.')
    elif campaign != 'd298':
        raise ValueError('unknown frozen smoke identity')
    assert not OUT.exists(), 'never overwrite a prior audit'
    assert p.check_plan() == '0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    assert sha256(ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md') == '5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    launch = read(RUN/'launch.json')
    assert launch['pass'] and launch['service_returncode'] == launch['watchdog_returncode'] == 0
    assert launch['service_path_removed'] and launch['native_gpu_context_release_confirmed']
    stats, evidence, requests = audit_run(SOURCE)
    assert len(requests) == 100
    outcome_path = RUN/'launch.launch/physical_deployment/main_outcome.json'
    outcome = read(outcome_path)
    assert outcome['run_completed'] and not outcome['error']
    assert not outcome['cleanup_errors'] and not outcome['measurement_errors']
    journal = RAW/'smoke_remote_transfers.jsonl'
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    remote = {r['transfer_id']:r for r in rows}
    transfers = outcome['remote_transfers']
    assert len(remote) == len(rows) == len(transfers)
    assert set(remote) == {r['http_transfer_id'] for r in transfers}
    clock = read(RAW/health_name)['health']['clock_id']
    for r in transfers:
        s = remote[r['http_transfer_id']]
        assert r['state'] == 'published' and r['content_verified'] and r['published_archive_verified']
        assert r['artifact_id'] == s['artifact_id'] and r['remote_archive_sha256'] == s['archive_sha256']
        assert r['transferred_bytes'] == s['archive_bytes'] == s['bytes_read'] == s['bytes_written']
        assert s['clock_id'] == clock and s['outcome'] == 'sent'
        assert not r['remote_pack_performed'] and not s['pack_performed'] and not s['temporary_created']
    physical = read(RUN/'launch.launch/physical_deployment/summary.json')
    assert abs(sum(physical['window_gpu_seconds'].values())-stats['physical_gpu_seconds']) < 1e-6
    with (RUN/'launch.launch/watchdog.jsonl').open() as stream:
        resources = [r for line in stream if (r:=json.loads(line))['event']=='resource_sample']
    assert resources and not any(r['decision']['abort_reasons'] for r in resources)
    assert sha256(WARM) == '0ebc7587effe293dcb986d33fb932c100cbcb3384c56ab2c0b0bec73b2bdd88c'
    thresholds = [dict(group_id=t['group_id'],lower_exclusive=t['lower_exclusive'],
        upper_inclusive=t['upper_inclusive'],ttft_s=t['candidate_ttft_threshold_ms']/1000,
        tpot_s=t['candidate_tpot_threshold_ms']/1000) for t in read(WARM)['warm_threshold_candidates']]
    normalized = []
    for r in requests:
        n = r['native_token_timing']['actual_prompt_tokens']
        assert type(n) in (int,float) and math.isfinite(n) and n >= 1 and int(n)==n
        normalized.append(dict(request_id=r['request_id'],success=True,prompt_tokens=int(n),
            token_count=r['completion_tokens'],ttft_s=r['overall_ttft_ms']/1000,tpot_s=r['tpot_ms']/1000))
    timing = provisional_timing_bound(normalized,thresholds,offered_count=100)
    seal = p.verify_seal(ROOT/'paper_results/ieee_tc/safety/20260925_execution_start_protected.json')
    known = '/home/qhq/serverless_llm_baselines/scripts/replay_openai_trace.py'
    assert seal['changed']==[known] and seal['entry_count']==147
    assert p.digest(Path(known))=='04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22'
    OUT.mkdir(parents=True, exist_ok=False)
    compare_diagnostics(REFERENCE,SOURCE,OUT)
    write_binding_audit(REFERENCE,SOURCE,INDEX,OUT)
    binding = read(OUT/'binding_audit.json')
    assert all(r['control_binding_pass_count']==100 for r in binding['runs'].values())
    result = dict(kind=f'{campaign}_smoke_audit_v1',stats=stats,evidence=evidence,
        timing=timing,window_gpu_seconds=physical['window_gpu_seconds'],
        tier_counts=dict(Counter(r['readiness_tier_before_dispatch'] for r in requests)),
        remote=dict(transfers=len(transfers),wire_bytes=sum(r['transferred_bytes'] for r in transfers),
            pack_performed=False,clock=clock),
        resources=dict(samples=len(resources),peak_bytes=max(r['service']['memory.peak'] for r in resources),
            min_host_available_bytes=min(r['host']['available_bytes'] for r in resources),
            max_swap_bytes=max(r['service']['memory.swap.current'] for r in resources),
            memory_events={k:max(r['service']['memory.events'][k] for r in resources)
                for k in ('high','max','oom','oom_kill')}),
        protection_raw=seal,full_qualified=False,n_correct=None,common_slo_qualified=False,
        no_figures=True,replicates_per_configuration=1,confidence_interval=None,
        caveat='Complete100-request diagnostic only; not complete4000W0, numerical correctness or formal G1/G2.',
        comparison_caveat=comparison_caveat,
        raw_evidence=[dict(path=str(path),sha256=sha256(path)) for path in
            (SOURCE,REFERENCE,outcome_path,journal,WARM,INDEX,RAW/'smoke_remote_monitor.log',Path(__file__))])
    if campaign in ('d302', 'd308', 'd310', 'd312'):
        result['admission_waits'] = summarize_admission_waits(requests)
        result['mechanism_events'] = {k:v for k,v in outcome['mechanism_events'].items()
            if k in ('_ieee_source_observation_stats', '_ieee_route_cache_stats')}
    with (OUT/'audit.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(stats=stats,timing={k:v for k,v in timing.items() if k!='decisions'},
        resources=result['resources'],remote=result['remote']),indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', choices=('d298','d302','d308','d310','d312'), default='d298')
    main(parser.parse_args().campaign)
