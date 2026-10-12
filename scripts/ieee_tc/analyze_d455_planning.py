"""D455 retained planning/transport evidence; CPU only, immutable outputs."""
import gzip
import json
import math
import sys
from pathlib import Path
from collections import Counter
import ijson

ROOT=Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
RAW=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts.ieee_tc.check_d299_full_context import read,sha256
from scripts.ieee_tc.audit_d402_serverlessllm_path import timing_stats
from scripts import ieee_tc_preflight as p

comparison=read(ROOT/'paper_results/ieee_tc/main_comparison/d454_7b_w0/comparison.json')
control=read(ROOT/'paper_results/ieee_tc/main_comparison/d454_7b_w0/control_diagnosis.json')
pinned={r['path']:r['sha256'] for r in comparison['source_refs']+control['source_refs']}
refs=[]
def checked(path):
    path=Path(path)
    assert sha256(path)==pinned[str(path)],path
    refs.append(dict(path=str(path),sha256=pinned[str(path)]))
    return path

reports={}
for ref in comparison['source_refs']:
    source=Path(ref['path'])
    if 'experiment_results_full' not in source.name:
        continue
    checked(source)
    name,scenario=('PrimeLoRA','tc_ieee_full') if 'd371_' in str(source) else ('ElasticLocality','tc_ieee_elastic_locality')
    path=source.parent/'launch.launch/physical_deployment/main_outcome.json'
    if name=='PrimeLoRA':
        receipt=read(checked(source.parent/'compression_receipt.json'))
        archive=checked(receipt['archive'])
        opener=lambda:gzip.open(archive,'rb')
    else:
        checked(path)
        opener=lambda:path.open('rb')
    with opener() as stream:
        events=list(ijson.items(stream,'planning_cpu_events.item',use_float=True))
    cpu={}
    if events:
        for key,a,b in [('capture_serialization_ms','captured_monotonic_s','detached_monotonic_s'),
                        ('serialized_owner_wait_ms','detached_monotonic_s','submitted_monotonic_s'),
                        ('worker_dispatch_ms','submitted_monotonic_s','worker_started_monotonic_s'),
                        ('worker_wall_ms','worker_started_monotonic_s','worker_finished_monotonic_s'),
                        ('worker_result_pickup_ms','worker_finished_monotonic_s','joined_monotonic_s')]:
            values=[(e[b]-e[a])*1000 for e in events]
            assert all(math.isfinite(v) and v>=0 for v in values)
            cpu[key]=dict(timing_stats(values),sum=math.fsum(values))
        for key in ['worker_cpu_seconds','input_bytes','output_bytes']:
            values=[e[key] for e in events]
            cpu[key]=dict(timing_stats(values),sum=math.fsum(values))
    fields=['parent_rpc_wall_ms','parent_rpc_overhead_ms','parent_rpc_channel_acquire_ms',
            'parent_rpc_response_pickup_delay_ms','parent_rpc_thread_resume_delay_ms',
            'parent_rpc_send_flush_ms','parent_rpc_wait_response_ms','parent_rpc_request_bytes',
            'parent_rpc_response_bytes','worker_rpc_handler_wall_ms','worker_rpc_queue_ms']
    observations={k:[] for k in fields}
    external_first=None
    with source.open('rb') as stream:
        for r in ijson.items(stream,f'detailed_results.{scenario}.requests.item',use_float=True):
            timing=r['native_token_timing']
            assert timing['parent_rpc_transport']=='native_async_socket_v1'
            for k in fields:
                v=timing[k]
                assert type(v) in (int,float) and math.isfinite(v) and v>=0
                observations[k].append(v)
            if external_first is None:
                external_first=r['external_arrival_timing']
    assert all(len(v)==4000 for v in observations.values())
    reports[name]=dict(planning_event_count=len(events),planning_states=dict(Counter(e['state'] for e in events)),
        planning_operations=dict(Counter(e['operation'] for e in events)),planning=cpu,
        first_planning_event=events[0] if events else None,
        rpc={k:timing_stats(v) for k,v in observations.items()},external_schema_example=external_first)
result=dict(reports=reports,source_refs=refs,script=dict(path=str(Path(__file__)),sha256=sha256(Path(__file__))),
    scope=p.scope_identity(p.cg_path().name),causal_attribution_established=False,
    caveat='RPC pickup uses same-host wall times and producer clamping; planning spans overlap requests and are not additive E2E.')
with (RAW/'cpu_inspection.json').open('x') as f:
    json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result,indent=2))
