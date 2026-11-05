"""Join existing planning hashes, never infer all-hot-GPU from an empty plan."""
import collections
import json
import os
from pathlib import Path
import sys
ROOT=Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import ijson
from scripts import ieee_tc_preflight as p
from scripts.ieee_tc.audit_d402_serverlessllm_path import timing_stats

assert os.sched_getaffinity(0)=={2,3,26,27}
assert os.environ['CUDA_VISIBLE_DEVICES']==''
context=HERE/'analysis/planner.json'
source=json.loads(context.read_text())['source']
path=Path(source['path'])
assert p.digest(path)==source['sha256']
def read(prefix):
    with path.open('rb') as stream:
        return list(ijson.items(stream,prefix,use_float=True))
cpu=read('planning_cpu_events.item')
files=read('mechanism_events._ieee_file_preparation_plans.item')
gpus=read('mechanism_events._ieee_gpu_preparation_plans.item')
epochs=read('mechanism_events._ieee_residency_epochs.item')
by_hash={e['plan_sha256']:e for e in cpu if e['operation']=='owned_execution_epoch'}
assert len(by_hash)==sum(e['operation']=='owned_execution_epoch' for e in cpu)
gpu_hashes={r['planning_sha256'] for r in gpus}
residency_hashes={r['plan_sha256'] for r in epochs if r['state']=='completed'}
counts=collections.Counter()
empty=[]
for record in files:
    count=record['registration']['targets']
    assert type(count) is int and count>=0
    key=record['objective_sha256']
    assert key in by_hash
    classification=('completed_no_file_targets_no_gpu_plan' if count==0 and key not in gpu_hashes
        and record['state']=='completed' else 'other')
    counts[classification]+=1
    if classification!='other':
        row=by_hash[key]
        empty.append(dict(plan_sha256=key,trigger_reason=record['trigger_reason'],
            completed_residency_epoch=key in residency_hashes,
            worker_cpu_ms=row['worker_cpu_seconds']*1000))
assert len({r['plan_sha256'] for r in empty})==len(empty)
result=dict(kind='d1067_completed_no_target_planning_v1',source=source,
    script_sha256=p.digest(Path(__file__)),context_sha256=p.digest(context),
    file_plans=len(files),gpu_plans=len(gpus),classifications=dict(counts),
    empty_trigger_counts=dict(collections.Counter(r['trigger_reason'] for r in empty)),
    completed_residency_empty=sum(r['completed_residency_epoch'] for r in empty),
    empty_cpu_ms=dict(timing_stats([r['worker_cpu_ms'] for r in empty]),
        total=sum(r['worker_cpu_ms'] for r in empty)),empty_plan_refs=empty,
    limitation='No file target and no GPU plan after completed execution establishes no selected movement, NOT its reason. Does not prove all positive-demand adapters were GPU-ready, zero demand, capacity infeasibility, or that safe early elision was available.',
    production_changes=False,new_gpu_runs=0)
with (HERE/'analysis/no_actions.json').open('x') as f:
    json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
cg=p.cg_path()
with (HERE/'analysis/no_actions_resources.json').open('x') as f:
    json.dump(dict(scope=p.scope_identity(cg.name),peak_bytes=int((cg/'memory.peak').read_text()),
        events=(cg/'memory.events').read_text(),swap_bytes=int((cg/'memory.swap.current').read_text())),f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='empty_plan_refs'}))
