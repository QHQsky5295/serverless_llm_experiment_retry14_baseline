"""R1 D1066 analysis: reuse sealed metrics, partitioner and recorded mechanism events."""
import bisect
import json
import math
import os
from collections import Counter
from pathlib import Path
import sys

import ijson

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
sys.path.insert(0, str(ROOT))
from scripts import ieee_tc_preflight as p
from scripts.ieee_tc.analyze_d455_stages import analyze_retained_overhead, checked_retained_source
from scripts.ieee_tc.audit_d402_serverlessllm_path import timing_stats

OUT = Path(__file__).parent

def read(path):
    return json.loads(path.read_text())

def ref(path):
    return dict(path=str(path), sha256=p.digest(path))

def save(path, payload):
    with path.open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')

def main():
    group = p.cg_path()
    assert os.sched_getaffinity(0) == {2, 3, 26, 27}
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert (group/'memory.max').read_text().strip() == str(4*p.GIB)
    completion = OUT/'completion'
    assert read(completion/'closure.json')['passed']
    parent = ROOT/'paper_results/ieee_tc/p2_backend/d1047_same_version_w1/completion'
    refs = []
    for base in (parent, completion):
        for name in ('full4000.json', 'delivery_and_resources.json'):
            path = base/name
            refs.append(ref(path))
            # The shared analyzer may evolve between two runs. Preserve each
            # original manifest, but pin only the actual run data consumed by
            # this existing overhead analyzer; do not conflate code revisions.
            refs += [r for r in read(path)['source_refs']
                     if Path(r['path']).name == 'main_outcome.json'
                     or Path(r['path']).name.startswith('experiment_results_full')]
        refs.append(ref(Path(read(base/'full4000.json')['source_run'])/'launch.json'))
    source_set = OUT/'comparison_sources_v2.json'
    save(source_set, dict(source_refs=refs))
    report = analyze_retained_overhead(OUT/'overhead_comparison', [source_set])
    metric = read(completion/'full4000.json')
    old = read(parent/'full4000.json')
    run = Path(metric['source_run'])
    source = next(r for r in read(completion/'delivery_and_resources.json')['source_refs']
                  if Path(r['path']).name == 'main_outcome.json')
    opener, source_refs = checked_retained_source(Path(source['path']), source['sha256'])
    with opener() as stream:
        noops = list(ijson.items(stream, 'mechanism_events._ieee_residency_noop_rounds.item', use_float=True))
    assert noops
    for row in noops:
        assert row['kind'] == 'ieee_residency_zero_demand_v1'
        assert row['total_arrivals'] == 0 and row['arrival_counts'] == {}
        assert row['window_seconds'] == 5.0 and math.isfinite(row['demand_observed_at'])
        assert row['physical_resources_reserved'] is False
        assert row['target_replicas'] and len(set(row['target_replicas'])) == len(row['target_replicas'])
    ingress = run/'launch.launch/service_ingress.jsonl'
    with ingress.open() as stream:
        arrivals = [r for line in stream if (r := json.loads(line))['event'] == 'request_received']
    assert len(arrivals) == len({r['request_id'] for r in arrivals}) == 4000
    times = sorted(r['server_received_s'] for r in arrivals)
    ingress_checks = Counter()
    for row in noops:
        t = row['demand_observed_at']
        count = bisect.bisect_right(times, t)-bisect.bisect_right(times, t-row['window_seconds'])
        ingress_checks[str(count)] += 1
    rounds, counts, states = {}, Counter(), Counter()
    with opener() as stream:
        for epoch in ijson.items(stream, 'mechanism_events._ieee_residency_epochs.item', use_float=True):
            states[epoch['state']] += 1
            observation = epoch['shared_file_observation']
            rid = observation['round_id']
            if rid in rounds:
                assert observation == rounds[rid]
            rounds[rid] = observation
            counts[rid] += 1
    assert all(counts[k] == row['replicas'] for k, row in rounds.items())
    file_times = [r['elapsed_ms'] for r in rounds.values()]
    assert all(math.isfinite(x) and x >= 0 for x in file_times)
    files, file_targets, file_results = Counter(), 0, 0
    with opener() as stream:
        for plan in ijson.items(stream, 'mechanism_events._ieee_file_preparation_plans.item', use_float=True):
            files[plan['state']] += 1
            file_targets += plan['registration']['targets']
            file_results += len(plan['results'])
    lower = {'physical_gpu_seconds': (old['physical_gpu_seconds'], metric['physical_gpu_seconds'])}
    for key in ('ttft_ms', 'tpot_ms', 'e2e_ms'):
        for stat in ('mean', 'p95_type1'):
            lower[key+'_'+stat] = (old['metrics'][key][stat], metric['metrics'][key][stat])
    differences = {k: dict(parent=a, candidate=b, absolute_improvement=a-b,
                           relative_improvement=(a-b)/a) for k,(a,b) in lower.items()}
    output = dict(kind='zero_demand_completion_analysis_d1066_v1', passed=True,
        noop_rounds=len(noops), noop_replica_targets=sum(len(r['target_replicas']) for r in noops),
        noop_window_ingress_counts=dict(ingress_checks),
        residency_epochs=sum(states.values()), residency_states=dict(states),
        unique_file_observations=len(rounds),
        file_observation_wall_ms=dict(timing_stats(file_times), sum=math.fsum(file_times)),
        file_plan_states=dict(files), file_plan_targets=file_targets, file_plan_results=file_results,
        differences=differences, joint_slo_difference_pp=100*(metric['joint_slo_fraction']-old['joint_slo_fraction']),
        matched_request_inputs=report['matched_request_inputs'], numerical_correctness_qualified=False,
        n_correct=None, ranking_eligible=False, confidence_interval=None, new_gpu_runs=0,
        source_refs=[*source_refs, ref(ingress), ref(source_set), ref(completion/'closure.json'),
                     ref(OUT/'overhead_comparison/summary.json')], script=ref(Path(__file__)),
        resources=p.cgroup_snapshot(group), caveats=[
            'No-op rounds and per-replica avoided opportunities are distinct counts.',
            'Historical 260 was only a conservative bound, not the exact number removed here.',
            'Live request ingress is an independent observation, not a replacement for the authoritative demand journal.',
            'Closed-loop histories differ; less planning CPU is not an additive subtraction from TTFT/E2E.',
            'One fixed known trace; neither numerical adapter correctness nor general performance dominance is established.'])
    save(OUT/'completion_analysis.json', output)
    print(json.dumps({k: output[k] for k in ('noop_rounds','noop_replica_targets','noop_window_ingress_counts',
        'residency_epochs','unique_file_observations','file_plan_states','differences','joint_slo_difference_pp')}))

if __name__ == '__main__':
    main()
