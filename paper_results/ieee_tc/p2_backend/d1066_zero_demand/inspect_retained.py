"""Bound the demand sampling instant using retained epoch/CPU/ingress records."""
import bisect
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
    prior = ROOT/'paper_results/ieee_tc/p2_backend/d1065_preparation_consumer_audit/summary.json'
    assert sha256(prior) == '6047ed4d45dd1af54638bdd9e3374de87bd38acfda0a42c457d402c926f5682d'
    source = next(r for r in read(prior)['source_refs'] if Path(r['path']).name == 'main_outcome.json')
    opener, refs = checked_retained_source(Path(source['path']), source['sha256'])
    ingress = Path(source['path']).parent.parent/'service_ingress.jsonl'
    with ingress.open() as stream:
        arrivals = [row for line in stream if (row := json.loads(line))['event'] == 'request_received']
    assert len(arrivals) == len({r['request_id'] for r in arrivals}) == 4000
    times = sorted(r['server_received_s'] for r in arrivals)
    with opener() as stream:
        events = [e for e in ijson.items(stream, 'planning_cpu_events.item', use_float=True)
                  if e['operation'] == 'owned_execution_epoch']
    by_hash = {e['plan_sha256']: e for e in events}
    assert len(by_hash) == len(events)
    with opener() as stream:
        plans = list(ijson.items(stream, 'mechanism_events._ieee_file_preparation_plans.item', use_float=True))
    files = {r['objective_sha256']: r for r in plans}
    assert len(files) == len(plans)
    config = ROOT/'configs/ieee_tc/20261104_d1047_3b_event_busy_time_w1.yaml'
    import yaml
    cfg = yaml.safe_load(config.read_text())
    # Exact frozen configuration, not an analysis-selected window.
    window = cfg['resource_coordination']['online_hotness_window_s']
    assert window == 5.0
    rows, states = [], Counter()
    with opener() as stream:
        for epoch in ijson.items(stream, 'mechanism_events._ieee_residency_epochs.item', use_float=True):
            states[epoch['state']] += 1
            key = epoch.get('plan_sha256')
            if key is None:
                continue
            event = by_hash[key]
            start, end = epoch['started_at'], event['captured_monotonic_s']
            assert start <= end
            count = bisect.bisect_right(times, end) - bisect.bisect_right(times, start-window)
            row = dict(plan_sha256=key, earliest_sampling_s=start, latest_sampling_s=end,
                state=epoch['state'], arrivals_in_union_of_possible_windows=count,
                certified_zero_demand=count == 0)
            if key in files:
                row.update(file_plan_state=files[key]['state'], results=len(files[key]['results']),
                           registered_targets=files[key]['registration']['targets'])
            rows.append(row)
    zero = [r for r in rows if r['certified_zero_demand']]
    assert all(r['file_plan_state'] == 'completed' and r['results'] == r['registered_targets'] == 0 for r in zero)
    group = p.cg_path()
    result = dict(kind='retained_zero_demand_bound_d1066_v1', window_seconds=window,
        ingress_requests=len(times), residency_states=dict(states), linked_epochs=len(rows),
        certified_zero_demand=len(zero), zero_demand_records=zero,
        inconclusive_linked_epochs=len(rows)-len(zero),
        empty_completed_file_plans=sum(r['state']=='completed' and not r['results'] for r in plans),
        source_refs=[dict(path=str(prior), sha256=sha256(prior)), *refs,
                     dict(path=str(ingress), sha256=sha256(ingress)), dict(path=str(config), sha256=sha256(config))],
        script=dict(path=str(Path(__file__)), sha256=sha256(Path(__file__))),
        scope=p.scope_identity(group.name), resources=p.cgroup_snapshot(group), new_gpu_runs=0,
        caveat='Only absence over the union of all possible demand windows is certified; other epochs remain unknown.')
    with Path(__file__).with_name('retained.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k:result[k] for k in ('linked_epochs','certified_zero_demand',
        'inconclusive_linked_epochs','empty_completed_file_plans')}))


if __name__ == '__main__':
    main()
