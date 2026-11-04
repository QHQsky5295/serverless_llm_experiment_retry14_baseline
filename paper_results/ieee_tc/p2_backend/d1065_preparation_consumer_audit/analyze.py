"""Reuse D1064 source bindings to count actual preparation consumers, not timing."""
import json
import sys
from collections import Counter
from pathlib import Path

import ijson

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
sys.path.insert(0, str(ROOT))
from scripts.ieee_tc.analyze_d455_stages import checked_retained_source
from scripts.ieee_tc.check_d299_full_context import read, sha256
from scripts import ieee_tc_preflight as p


def main():
    source_report = ROOT/'paper_results/ieee_tc/p2_backend/d1064_retained_a1_overhead/summary.json'
    assert sha256(source_report) == '304fe266e7481fa873338f1b7f561dbb167f28509831c25628efdc5bac0af798'
    sources = [r for r in read(source_report)['source_refs']
               if '/d1047_20261104/' in r['path'] and Path(r['path']).name == 'main_outcome.json']
    assert len(sources) == 1
    source = sources[0]
    opener, refs = checked_retained_source(Path(source['path']), source['sha256'])
    states, holds, results, targets, groups = (Counter() for _ in range(5))
    identities = set()
    with opener() as stream:
        for row in ijson.items(stream, 'mechanism_events._ieee_file_preparation_plans.item', use_float=True):
            identity = row['plan_id']
            assert identity not in identities
            identities.add(identity)
            assert row['registration']['plan_id'] == identity
            assert row['registration']['registered'] is True
            assert row['registration']['physical_resources_reserved'] is False
            n_targets = row['registration']['targets']
            assert type(n_targets) is int and n_targets >= 0
            assert isinstance(row['native_fallbacks'], list) and isinstance(row['results'], list)
            assert row['state'] in ('completed', 'superseded', 'cancelled')
            states[row['state']] += 1
            holds[len(row['native_fallbacks'])] += 1
            results[len(row['results'])] += 1
            targets[n_targets] += 1
            groups[(row['state'], len(row['results']), n_targets)] += 1
    assert len(identities) == 2397
    with opener() as stream:
        assert next(ijson.items(stream, 'run_completed')) is True
    group = p.cg_path()
    report = dict(kind='retained_preparation_consumers_d1065_v1', run='D1047',
        plans=len(identities), states=dict(states), native_fallback_count_histogram=dict(holds),
        native_fallback_holds=sum(k*v for k, v in holds.items()),
        multi_fallback_plans=sum(v for k, v in holds.items() if k > 1),
        result_count_histogram=dict(results), registered_file_target_histogram=dict(targets),
        state_result_target_counts=[dict(state=s, results=r, registered_file_targets=t, count=n)
            for (s, r, t), n in sorted(groups.items())],
        source_refs=[dict(path=str(source_report), sha256=sha256(source_report)), *refs],
        script=dict(path=str(Path(__file__)), sha256=sha256(Path(__file__))),
        scope=p.scope_identity(group.name), resource_snapshot={name:(group/name).read_text().strip()
            for name in ('memory.high', 'memory.max', 'memory.peak', 'memory.events',
                         'memory.swap.current', 'memory.swap.max')},
        new_gpu_runs=0, formal_ranking_eligible=False, causal_attribution_established=False,
        caveats=['Zero native fallback holds rules out this path as observed D1047 overhead, not all workloads.',
                 'An empty results list is not proof of an empty profitable candidate set.',
                 'Do not bypass fresh capacity or source checks using these aggregate counts.'])
    with Path(__file__).with_name('summary.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in ('plans', 'states', 'native_fallback_holds',
        'multi_fallback_plans', 'result_count_histogram', 'registered_file_target_histogram')}))


if __name__ == '__main__':
    main()
