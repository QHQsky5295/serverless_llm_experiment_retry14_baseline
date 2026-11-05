"""Seal only the new D1067 analysis; reuse project checks, no runtime mutation."""
import collections
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts import ieee_tc_preflight as p

assert os.sched_getaffinity(0)=={2,3,26,27}
assert os.environ['CUDA_VISIBLE_DEVICES']==''
load=lambda path:json.loads(path.read_text())
prior=load(ROOT/'paper_results/ieee_tc/p2_backend/d1066_zero_demand/completion_checkpoint.json')
for name,sha in prior['protected'].items():
    assert p.digest(ROOT/name)==sha,name
for ref in prior['tested_sources']:
    assert p.digest(Path(ref['path']))==ref['sha256'],ref['path']
seal=p.verify_seal(ROOT/'results/ieee_tc/slora_qualification/d435_20261011/protected_before.json')
assert seal['pass'] and seal['entry_count']==147
assert seal['plan_sha256']==prior['plan_sha256']
component=load(HERE/'component/result.json')
checked={}
for ref in [component['source'],*component['source_refs'],load(HERE/'analysis/planner.json')['source']]:
    path=Path(ref['path'])
    assert p.digest(path)==ref['sha256']
    checked[str(path)]=ref['sha256']
stats=[]
for index in range(4):
    means={}
    for decoder in ('stdlib','msgspec'):
        rows=[r for r in component['measurements'] if r['snapshot']==index and r['decoder']==decoder]
        assert len(rows)==3
        assert all(len(r['wall_ms'])==len(r['cpu_ms'])==5 for r in rows)
        means[decoder]=sum(sum(r['wall_ms']) for r in rows)/15
    stats.append(dict(snapshot=index,registered=component['cases'][index]['registered'],
        mean_wall_ms=means,relative_improvement=(means['stdlib']-means['msgspec'])/means['stdlib']))
assert len(component['fixtures'])==4 and all(r['complete_bundle_and_all_hashes_equal'] for r in component['fixtures'])
resources=[]
for relative in ('analysis/resources.json','component/resources.json','analysis/no_actions_resources.json'):
    record=load(HERE/relative)
    assert not Path(record['scope']['path']).exists()
    assert all(int(line.split()[1])==0 for line in record['events'].splitlines())
    assert record['swap_bytes']==0
    resources.append(record)
profile_path=ROOT/'paper_results/ieee_tc/p2_backend/d843_3b_controller_full_w2/completion/business_callers.json'
profile=load(profile_path)
planner,=[r for r in profile['details'] if r['role']=='planner']
profile_summary=dict(path=str(profile_path),sha256=p.digest(profile_path),
    window_samples=planner['window_samples'],inclusive=planner['inclusive'],
    freeze_callers=[r for r in planner['callers'] if r['function']=='freeze_preparation_source_view'],
    caveat='Historical inclusive stack occurrences, not current CPU-time percentages.')
noactions=load(HERE/'analysis/no_actions.json')
owned,=[r for r in load(HERE/'analysis/planner.json')['groups'] if r['operation']=='owned_execution_epoch']
share=noactions['empty_cpu_ms']['total']/owned['metrics']['worker_cpu_ms']['total']
doc=ROOT/'docs/ieee_tc/D1067_PLANNER_REPRESENTATION_20261105.md'
paths=[doc,*sorted(q for q in HERE.rglob('*') if q.is_file())]
assert sum(q.stat().st_size for q in paths)<1024**2
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
    capture_output=True,text=True,check=True).stdout.strip()
assert not gpu,gpu
result=dict(kind='d1067_retained_analysis_checkpoint_v1',
    parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    decision='no_decoder_production_integration_or_gpu_replay; investigate exact_zero_benefit_witness',
    old_submission_seal=seal,protected=prior['protected'],checked_sources=checked,
    historical_profile=profile_summary,component_means=stats,
    no_selected_movement_owned_cpu_share=share,resources=resources,
    gpu_apps=[],production_changes=False,new_gpu_replays=0,
    n_correct=None,formal_ranking_eligible=False,
    files=[dict(path=str(q),sha256=p.digest(q),bytes=q.stat().st_size) for q in paths])
with (HERE/'checkpoint.json').open('x') as f:
    json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(protected=seal,component_means=stats,
    no_selected_movement_cpu_share=share,files=len(paths),gpu_apps=[])))
