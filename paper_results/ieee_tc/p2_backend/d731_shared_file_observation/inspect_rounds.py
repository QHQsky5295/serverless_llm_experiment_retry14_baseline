"""Read closed D729 planning journals; no replay or performance prediction."""
import bisect
import collections
import json
import os
import sys
from pathlib import Path

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
sys.path.insert(0, str(ROOT))
import ijson
from scripts import ieee_tc_preflight as p

RAW = Path(__file__).resolve().parent
SOURCE = ROOT/'results/ieee_tc/p2_backend_qualification/d729_20261022/3b_bulk_loader_full_w0/launch.launch/physical_deployment/main_outcome.json'

def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

assert os.sched_getaffinity(0) == {2, 3, 26, 27}
save(RAW/'inspect_identity.json', p.scope_identity(p.cg_path().name))
# Read only the three small event lists, not the one-GiB outcome as a dict.
groups = {}
for name, prefix in (
    ('control', 'mechanism_events._ieee_control_events.item'),
    ('residency', 'mechanism_events._ieee_residency_epochs.item'),
    ('planning', 'planning_cpu_events.item')):
    with SOURCE.open('rb') as f:
        groups[name] = list(ijson.items(f, prefix, use_float=True))
times = [e['observed_at'] for e in groups['control']]
assert times == sorted(times)
rounds = collections.defaultdict(list)
for event in groups['residency']:
    rounds[bisect.bisect_right(times, event['started_at'])-1].append(event)
summary = dict(
    source=dict(path=str(SOURCE), sha256=p.digest(SOURCE)),
    control_events=len(times), residency_epochs=len(groups['residency']),
    residency_states=dict(collections.Counter(e['state'] for e in groups['residency'])),
    planning_operations=dict(collections.Counter(e['operation'] for e in groups['planning'])),
    epochs_per_control_interval=dict(collections.Counter(len(es) for es in rounds.values())),
    unique_target_counts=dict(collections.Counter(len({e['target_replica'] for e in es}) for es in rounds.values())),
    max_start_spread_ms=max((max(e['started_at'] for e in es)-min(e['started_at'] for e in es))*1000 for es in rounds.values()),
    current_file_observation_calls_inferred_from_code=len(groups['residency']),
    one_per_eligible_control_interval_calls=len(rounds),
    limitation='Intervals identify likely scheduling batches, not an explicitly logged batch ID. File scan duration is not logged. Counts are not CPU time or predicted SLO.',
    sample_residency=groups['residency'][:4], sample_planning=groups['planning'][:2],
    resources=p.cgroup_snapshot(p.cg_path()))
save(RAW/'rounds.json', summary)
print(json.dumps(summary, indent=2))
