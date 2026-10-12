"""Bounded, read-only D454 control/physical-window projection; no simulation."""
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean

import ijson
import yaml

ROOT=Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
RAW=Path(__file__).resolve().parent
OUT=ROOT/'paper_results/ieee_tc/main_comparison/d454_7b_w0'
sys.path.insert(0,str(ROOT))
from scripts.ieee_tc.check_d299_full_context import read, sha256, closed_physical_integral
from scripts import ieee_tc_preflight as p
from faaslora.datasets.workload_generator import FrozenReplayPlan

comparison=read(OUT/'comparison.json')
sealed={x['path']:x['sha256'] for x in comparison['source_refs']}
refs=[]
def checked(path, expected=None):
    path=Path(path)
    digest=sha256(path)
    if expected is not None:
        assert digest==expected,path
    refs.append(dict(path=str(path),sha256=digest))
    return path

reports={}
settings=None
for name, date, config, parent, runname in [
    ('PrimeLoRA','d371_20261010','20261010_d371_7b_full_w0.yaml','p2_backend_qualification','7b_full_w0'),
    ('ElasticLocality','d453_20261012','20261012_d453_7b_elastic_locality_w0.yaml','elastic_locality_qualification','7b_elastic_locality_w0')]:
    run=Path('/home/qhq/serverless_llm_experiment/results/ieee_tc')/parent/date/runname
    launch_path=run/'launch.json'
    launch=read(checked(launch_path,sealed[str(launch_path)]))
    physical_path=run/'launch.launch/physical_deployment/summary.json'
    physical=read(checked(physical_path,sealed[str(physical_path)]))
    total=closed_physical_integral(physical)
    config_path=checked(ROOT/'configs/ieee_tc'/config)
    cfg=yaml.safe_load(config_path.read_text())['resource_coordination']['ieee_scaling']
    if settings is None:
        settings=cfg
    assert cfg==settings,'controller configuration changed'
    outcome=run/'launch.launch/physical_deployment/main_outcome.json'
    if name=='PrimeLoRA':
        receipt=read(checked(run/'compression_receipt.json'))
        archive=checked(receipt['archive'],receipt['archive_sha256'])
        digest=hashlib.sha256()
        with gzip.open(archive,'rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                digest.update(block)
        assert digest.hexdigest()==receipt['source_sha256']
        opener=lambda:gzip.open(archive,'rb')
    else:
        audit=read(ROOT/'paper_results/ieee_tc/elastic_locality/d453_actual_full_w0.json')
        ref=next(x for x in audit['refs'] if x['path']==str(outcome))
        checked(outcome,ref['sha256'])
        opener=lambda:outcome.open('rb')
    with opener() as stream:
        events=list(ijson.items(stream,'mechanism_events._ieee_control_events.item',use_float=True))
    assert events
    start=launch['external_replay']['replay_t0_s']
    plan=FrozenReplayPlan.load(launch['external_replay']['plan']['source_path'])
    assert plan.identity()==launch['external_replay']['plan']
    duration=plan.entries[-1].offset_s
    end=start+duration
    events=[r for r in events if start<=r['observed_at']<=end]
    assert events and all(a['observed_at']<b['observed_at'] for a,b in zip(events,events[1:]))
    reasons=Counter()
    for event in events:
        flags=dict(queue_not_low=event['queue_depth']>=cfg['queue_lower'],
            active_not_low=event['active_saturation']>=cfg['active_lower'],
            ttft_not_low=event['p95_ttft_ms'] is None or event['p95_ttft_ms']>=cfg['ttft_lower_ms'])
        assert event['all_low']==(not any(flags.values()))
        if event['ready_instances']>1:
            reasons.update(dict(eligible_ready_gt1=1, all_low=int(event['all_low']),
                pending=int(event['pending_instances']>0),**{k:int(v) for k,v in flags.items()}))
    edges=Counter({start:0,end:0})
    for intervals in physical['device_intervals'].values():
        for a,b in intervals:
            a,b=max(a,start),min(b,end)
            if a<b:
                edges[a]+=1
                edges[b]-=1
    occupancy=Counter()
    active=0
    stamps=sorted(edges)
    for i,t in enumerate(stamps[:-1]):
        active+=edges[t]
        assert 0<=active<=4
        occupancy[active]+=stamps[i+1]-t
    assert abs(sum(occupancy.values())-duration)<1e-6
    assert abs(sum(k*v for k,v in occupancy.items())-physical['window_gpu_seconds']['arrival'])<1e-6
    reports[name]=dict(control_samples=len(events),
        control_actions=dict(Counter(r['action'] for r in events)),
        control_outcomes=dict(Counter(r['outcome'] for r in events)),
        scalein_blocker_sample_counts=dict(reasons),
        sample_means={k:fmean(r[k] for r in events) for k in
            ('queue_depth','active_requests','active_saturation','ready_instances','pending_instances')},
        physical_gpu_seconds=total,window_gpu_seconds=physical['window_gpu_seconds'],
        arrival_duration_s=duration,arrival_occupancy_wall_seconds=dict(sorted(occupancy.items())),
        arrival_mean_physical_gpus=physical['window_gpu_seconds']['arrival']/duration)

first,second=reports['PrimeLoRA'],reports['ElasticLocality']
delta={k:first['window_gpu_seconds'][k]-second['window_gpu_seconds'][k] for k in first['window_gpu_seconds']}
assert abs(sum(delta.values())-(first['physical_gpu_seconds']-second['physical_gpu_seconds']))<1e-6
result=dict(kind='d454_control_and_physical_window_observation_v1',reports=reports,
    controller_settings=settings,PrimeLoRA_minus_ElasticLocality_gpu_seconds_by_window=delta,
    causal_attribution_established=False,
    notes=['Controller blockers overlap and are sample counts, not additive elapsed time.',
           'Physical occupancy is exact interval-union within the common arrival window, not utilization.',
           'Different routing/preparation/admission may change feedback; this projection does not isolate causality.'],
    source_refs=refs+[dict(path=str(checked(OUT/'comparison.json')),sha256=sha256(OUT/'comparison.json')),
                      dict(path=str(Path(__file__)),sha256=sha256(Path(__file__)))],
    scope=p.scope_identity(p.cg_path().name))
with (OUT/'control_diagnosis.json').open('x') as stream:
    json.dump(result,stream,indent=2,allow_nan=False)
print(json.dumps(result,indent=2))
