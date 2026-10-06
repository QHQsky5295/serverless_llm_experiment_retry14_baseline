#!/usr/bin/env python3
"""D297/D299 data-only audit using existing native and provisional timing checks.

This is an offline driver, not a serving change, new benchmark, or qualification.
Use a bounded auxiliary scope after all inference processes have exited.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ieee_tc.plot_d290_diagnostic import audit_run, binding_checks, percentile, sha256
from scripts.analyze_control_path_overhead import provisional_timing_bound
from faaslora.metrics.metrics_collector import NativeV1TokenTimeline

RAW = ROOT / 'results/ieee_tc/p2_backend_qualification/d297_20261007'
RUN = RAW / '7b_full_w0'
OUT = ROOT / 'paper_results/ieee_tc/p2_backend/d297_full_w0/validated1'
SOURCE = RUN / 'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d297_7b_full_w0.json'
WARM = ROOT / 'paper_results/ieee_tc/p2_backend/20261002_d170_warm_measurement.json'
INDEX = ROOT / 'paper_results/ieee_tc/inputs/20260927_7b_materialized_content_index.json'
PLAN = Path('/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md')
PROTOCOL = ROOT / 'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md'


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    with (OUT / name).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def table(name, rows):
    with (OUT / name).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main(campaign='d297'):
    global RAW, RUN, OUT, SOURCE
    source_sha = '7f171e1d58b061e7dd7392e1651d087c300ce1d6c2223c3ed16ef2fdce3aa07b'
    if campaign == 'd299':
        RAW = ROOT / 'results/ieee_tc/p2_backend_qualification/d299_20261007'
        RUN = RAW / '7b_full_w0'
        OUT = ROOT / 'paper_results/ieee_tc/p2_backend/d299_full_w0/validated1'
        SOURCE = RUN / 'experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d299_7b_full_w0.json'
        source_sha = 'df066b365fa13ecc25b225a793c362706157224965b8a7b360981d8cf45887ab'
    elif campaign != 'd297':
        raise ValueError('unknown frozen full-run identity')
    assert sha256(PLAN) == '0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    assert sha256(PROTOCOL) == '5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    assert sha256(WARM) == '0ebc7587effe293dcb986d33fb932c100cbcb3384c56ab2c0b0bec73b2bdd88c'
    assert sha256(SOURCE) == source_sha
    launch = read(RUN / 'launch.json')
    assert launch['pass'] and launch['native_gpu_context_release_confirmed']
    assert launch['service_path_removed']
    assert launch['service_returncode'] == launch['watchdog_returncode'] == launch['replay_returncode'] == 0
    OUT.mkdir(parents=True, exist_ok=False)
    stats, evidence, requests = audit_run(SOURCE)
    assert len(requests) == 4000
    deployment = read(RUN / 'launch.launch/physical_deployment/deployment.json')
    physical = read(RUN / 'launch.launch/physical_deployment/summary.json')
    assert abs(sum(physical['window_gpu_seconds'].values()) - stats['physical_gpu_seconds']) < 1e-6
    index = read(INDEX)
    expected = {a['id']: hashlib.sha256(json.dumps(sorted(a['files'], key=lambda f:f['path']),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest() for a in index['artifacts']}
    assert len(expected) == len(index['artifacts']) == 500
    warm = read(WARM)
    thresholds = [dict(group_id=t['group_id'], lower_exclusive=t['lower_exclusive'],
        upper_inclusive=t['upper_inclusive'], ttft_s=t['candidate_ttft_threshold_ms']/1000,
        tpot_s=t['candidate_tpot_threshold_ms']/1000) for t in warm['warm_threshold_candidates']]
    stages = ('arrival_to_admission_s', 'admission_to_engine_dispatch_s',
              'engine_entry_to_queue_s', 'native_engine_queue_s', 'native_prefill_s')
    normalized, failures, errors = [], [], []
    integral_float_prompt_counts = 0
    binding_counts = Counter()
    for row in requests:
        checks = binding_checks(row, expected[row['adapter_id']])
        binding_counts.update(k for k,v in checks.items() if v)
        if not all(checks.values()):
            failures.append(dict(request_id=row['request_id'], checks=checks))
        t = row['native_token_timing']
        spans = NativeV1TokenTimeline.service_breakdown(t,
            admitted_at=t['controller_admitted_monotonic_s'],
            completed_at=t['controller_completed_monotonic_s'], clock_id=deployment['clock_id'])
        arrival = row['external_arrival_timing']['planned_arrival_s']
        assert abs(arrival - deployment['arrival_start_s'] - row['scheduled_arrival_offset_s']) < 1e-6
        sv = dict(arrival_to_admission_s=t['controller_admitted_monotonic_s']-arrival)
        sv.update({k:spans[k[:-2]+'_ms']/1000 for k in stages[1:]})
        assert all(math.isfinite(v) and v >= 0 for v in sv.values())
        error = abs(sum(sv.values()) - row['overall_ttft_ms']/1000)
        assert error <= .001
        errors.append(error)
        prompt_count = t['actual_prompt_tokens']
        assert type(prompt_count) in (int, float) and math.isfinite(prompt_count)
        assert prompt_count >= 1 and int(prompt_count) == prompt_count
        integral_float_prompt_counts += type(prompt_count) is float
        normalized.append(dict(request_id=row['request_id'], adapter_id=row['adapter_id'], success=True,
            prompt_tokens=int(prompt_count), token_count=row['completion_tokens'],
            ttft_s=row['overall_ttft_ms']/1000, tpot_s=row['tpot_ms']/1000,
            e2e_s=row['overall_e2e_ms']/1000, tier=row['readiness_tier_before_dispatch'],
            canonical_prompt_sha256=row['canonical_prompt_sha256'],
            completion_token_ids_sha256=row['completion_token_ids_sha256'], **sv))
    assert not failures
    timing = provisional_timing_bound(normalized, thresholds, offered_count=4000)
    for row, decision in zip(normalized, timing.pop('decisions')):
        row.update(decision)
    stats.update(p99_ttft_ms=percentile([r['ttft_s']*1000 for r in normalized], .99),
        mean_e2e_s=mean(r['e2e_s'] for r in normalized),
        gpu_s_per_offered=stats['physical_gpu_seconds']/4000)
    groups = []
    for key, labels in [('all',['all']),('tier',sorted({r['tier'] for r in normalized})),
                        ('outcome',list(timing['outcome_counts']))]:
        for label in labels:
            rs = [r for r in normalized if key == 'all' or r[key] == label]
            if not rs:
                continue
            groups.append(dict(group=key, label=label, count=len(rs),
                mean_ttft_s=mean(r['ttft_s'] for r in rs), p95_ttft_s=percentile([r['ttft_s'] for r in rs], .95),
                mean_tpot_s=mean(r['tpot_s'] for r in rs),
                **{'mean_'+k:mean(r[k] for r in rs) for k in stages}))
    table('requests.csv', normalized)
    table('timing_groups.csv', groups)
    write('audit.json', dict(kind=f'{campaign}_full_development_audit_v1', stats=stats, evidence=evidence,
        timing=timing, thresholds=thresholds, window_gpu_seconds=physical['window_gpu_seconds'],
        control_binding_pass_counts=dict(binding_counts), max_stage_error_s=max(errors),
        integral_float_prompt_counts=integral_float_prompt_counts,
        n_correct=None, formal_qualification=False, confidence_interval=None,
        caveats=['One complete W0 development replay, not a held-out block.',
                 'Control binding consistency is not kernel arithmetic qualification.',
                 'Provisional warm thresholds are not globally frozen common SLO.',
                 'Conditional tier groups are not causal comparisons.']))
    print(json.dumps(dict(stats=stats, timing=timing), indent=2), flush=True)
    del requests, normalized
    gc.collect()

    # jq's streaming parser avoids materializing the 629-MiB outcome tree.
    outcome_path = RUN / 'launch.launch/physical_deployment/main_outcome.json'
    query = 'fromstream(2|truncate_stream(inputs | select(.[0][:2] == ["mechanism_events","_ieee_control_events"])))'
    controls = json.loads(subprocess.check_output(['jq','-cn','--stream',query,str(outcome_path)]))
    import yaml
    config_path = ROOT / f'configs/ieee_tc/20261007_{campaign}_7b_full_w0.yaml'
    cfg = yaml.safe_load(config_path.read_text())['resource_coordination']['ieee_scaling']
    counters = Counter()
    longest = current_start = 0.0
    previous = -math.inf
    control_rows = []
    for e in controls:
        assert previous <= e['observed_at']
        previous = e['observed_at']
        q = e['queue_depth'] < cfg['queue_lower']
        a = e['active_saturation'] < cfg['active_lower']
        known = e['p95_ttft_ms'] is not None
        t = known and e['p95_ttft_ms'] < cfg['ttft_lower_ms']
        assert e['all_low'] == (q and a and t)
        flags = dict(queue_low=q, active_low=a, ttft_known=known, ttft_low=t,
            queue_active_low=q and a, ttft_only_blocks=q and a and known and not t,
            all_low=e['all_low'], pending=e['pending_instances']>0)
        counters.update(k for k,v in flags.items() if v)
        counters['action:'+e['action']] += 1
        counters['outcome:'+e['outcome']] += 1
        if e['all_low'] and not e['pending_instances']:
            current_start = current_start or e['observed_at']
            longest = max(longest, e['observed_at']-current_start)
        else:
            current_start = 0.0
        control_rows.append(dict(offset_s=e['observed_at']-deployment['arrival_start_s'],
            queue=e['queue_depth'], active=e['active_requests'], capacity=e['ready_capacity'],
            pending=e['pending_instances'], p95_ttft_ms=e['p95_ttft_ms'],
            action=e['action'], outcome=e['outcome'], **{k:v for k,v in flags.items() if k != 'pending'}))
    assert controls
    table('control_samples.csv', control_rows)
    control_audit = dict(samples=len(controls), counts=dict(counters),
        longest_sampled_all_low_span_s=longest, cooldown_s=cfg['scale_down_cooldown_s'],
        min_known_p95_ttft_ms=min(e['p95_ttft_ms'] for e in controls if e['p95_ttft_ms'] is not None),
        config=cfg, counterfactual_gpu_s=None, causal_performance_gain=None,
        caveat='Sampled predicates only; changing one threshold may change the entire closed loop.')
    write('control_audit.json', control_audit)
    print(json.dumps(control_audit, indent=2), flush=True)
    paths = [SOURCE, WARM, INDEX, PLAN, PROTOCOL, config_path, outcome_path, Path(__file__),
        ROOT/'scripts/ieee_tc/plot_d290_diagnostic.py', ROOT/'scripts/analyze_control_path_overhead.py']
    write('manifest.json', dict(source_refs=[dict(path=str(p),sha256=sha256(p)) for p in paths],
        outputs={p.name:sha256(p) for p in sorted(OUT.iterdir())}, no_figures=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', choices=('d297', 'd299'), default='d297')
    main(parser.parse_args().campaign)
