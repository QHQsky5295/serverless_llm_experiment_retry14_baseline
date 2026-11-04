"""Bounded native NoHOST smoke audit; reuse the existing metric interpreter."""
import collections
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
RAW = Path(__file__).resolve().parent
RUN = RAW/'d1063_3b_nohost_w1_smoke100_a2'
OUT = RAW/'smoke100_a2_analysis1'
sys.path.insert(0, str(ROOT))
from scripts import ieee_tc_preflight as p
from scripts.ieee_tc.audit_d703_3b_full import reevaluate_common_slo
from scripts.ieee_tc.export_d701_3b_profile import matched_remote_transfers
from faaslora.storage.http_artifact_store import HttpArtifactStoreClient


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


group = p.cg_path()
assert os.sched_getaffinity(0) == {2,3,26,27} and os.environ['CUDA_VISIBLE_DEVICES'] == ''
for name, value in [('memory.high',3*p.GIB), ('memory.max',4*p.GIB), ('memory.swap.max',0)]:
    assert (group/name).read_text().strip() == str(value)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
entry = json.loads((RAW/'d1063_3b_nohost_w1_smoke100_a2_fresh_launch_check.json').read_text())
for name, sha in entry['source_hashes'].items():
    assert p.digest(ROOT/name) == sha, name
OUT.mkdir(exist_ok=False)
reevaluate_common_slo(RUN, OUT/'native100.json', expected_count=100)
metric = json.loads((OUT/'native100.json').read_text())
result = next(RUN.glob('experiment_results_full*.json'))
data = json.loads(result.read_text())
requests = data['detailed_results']['tc_ieee_full']['requests']
assert len(requests) == 100 and metric['n_correct'] is None
dispatch, native, modes = collections.Counter(), collections.Counter(), collections.Counter()
retired = joined = 0
for request in requests:
    assert request['success'] and request['completion_tokens'] == request['requested_completion_tokens']
    evidence = request['gpu_reference_evidence']
    receipt = evidence['receipt']
    assert receipt['independent_host_reuse'] is False
    assert evidence['confirmed_dispatch_snapshot'] is True
    assert request['readiness_tier_before_dispatch'] in ('remote','nvme','gpu')
    assert receipt['source_tier_before_acquisition'] in ('file','gpu','staging')
    modes[str(receipt['independent_host_reuse'])] += 1
    dispatch[request['readiness_tier_before_dispatch']] += 1
    native[receipt['source_tier_before_acquisition']] += 1
    retired += len(receipt.get('nohost_retired_adapter_ids', []))
    joined += receipt.get('native_staged_source_reused', False) is True
outcome_path = RUN/'launch.launch/physical_deployment/main_outcome.json'
outcome = json.loads(outcome_path.read_text())
mechanisms = {}
for key in ('_ieee_file_preparation_plans','_ieee_gpu_preparation_plans','_ieee_native_host_preparations'):
    records = outcome['mechanism_events'][key]
    mechanisms[key] = dict(count=len(records),states=dict(collections.Counter(r.get('state','not_recorded') for r in records)),
        attempts=sum(len(r.get('attempts',[])) for r in records))
transfers = outcome['remote_transfers']
client = HttpArtifactStoreClient(endpoint='http://192.168.4.174:18080',
    token=Path('/home/qhq/.config/primelora-tc-d75/artifact.token').read_text().strip(),
    timeout_s=8, required_delivery_mode='prepublished_gzip_v1')
health = client.health()
assert health['ok']
journal = subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',
    'primelora-artifact-174','cat',
    '/home/lab14/primelora_remote/tc/d80_20260927/3b/transfers-026842a3bec74f1b9e012bb813642067.jsonl'], text=True, timeout=60)
matched = matched_remote_transfers(dict(remote_transfers=transfers),journal,health['clock_id'])
save(OUT/'remote_matched.json',matched)
samples = foreign = escaped = 0
max_memory = 0
for line in (RUN/'launch.launch/watchdog.jsonl').read_text().splitlines():
    row = json.loads(line)
    if row['event'] != 'resource_sample':
        continue
    samples += 1
    foreign += len(row['gpu']['foreign_compute_processes'])
    escaped += len(row['gpu']['escaped_owned_processes'])
    max_memory = max(max_memory, row['service']['memory.peak'])
assert samples and foreign == escaped == 0
seal = p.verify_seal(Path('/home/qhq/serverless_llm_experiment/results/ieee_tc/slora_qualification/d435_20261011/protected_before.json'))
assert seal['pass'] and seal['entry_count'] == 147
save(OUT/'summary.json', dict(passed=True,scope='100-request real native entry, not a performance comparison',
    n=100,n_correct=None,formal_ranking_eligible=False, independent_host_reuse_receipts=dict(modes),
    dispatch_tiers=dict(dispatch),native_sources=dict(native),nohost_retired_ids_in_demand_receipts=retired,
    native_staged_demand_joins=joined,mechanisms=mechanisms,remote_transfers=len(matched),
    remote_monitoring='existing_unmonitored',external_remote_interference_unknown=True,
    watchdog_samples=samples,foreign_gpu_samples=foreign,escaped_gpu_samples=escaped,
    service_memory_peak_bytes=max_memory,protected=seal,sources=entry['source_hashes'],
    result=dict(path=str(result),sha256=p.digest(result)),resources=p.cgroup_snapshot(group)))
print(json.dumps(json.loads((OUT/'summary.json').read_text())),flush=True)
