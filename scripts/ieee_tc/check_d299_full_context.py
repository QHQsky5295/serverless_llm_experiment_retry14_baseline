"""Thin post-release D299/D300/D303 check reusing native audits and seal helpers.

No inference, figures, new metric definitions, or overwriting existing results.
The large outcome is projected through jq streaming, never loaded as a tree.
"""
import argparse
import csv
import gc
import json
import subprocess
from collections import Counter
from pathlib import Path

from scripts import ieee_tc_preflight as p
from scripts.ieee_tc.plot_d290_diagnostic import audit_run, sha256

ROOT = p.ROOT
RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d299_20261007'
RUN = RAW/'7b_full_w0'
OUT = ROOT/'paper_results/ieee_tc/p2_backend/d299_full_w0/context1'
CURATED = ROOT/'paper_results/ieee_tc/p2_backend'
FIELDS = ('request_id','adapter_id','scheduled_arrival_offset_s',
          'source_expected_output_tokens','requested_completion_tokens',
          'canonical_prompt_sha256','generation_contract','completion_token_source')
META = ('run_completed','error','cleanup_errors','measurement_errors')
QUERY = ('select(length==2 and (((.[0]|length)==1 and '
         '(.[0][0]|IN("run_completed","error","cleanup_errors","measurement_errors"))) '
         'or ((.[0]|length)==3 and .[0][0]=="remote_transfers")))')


def read(path):
    return json.loads(path.read_text())


def project(outcome):
    raw = subprocess.check_output(['jq','-c','--stream',QUERY,str(outcome)],text=True)
    meta, indexed = {}, {}
    for line in raw.splitlines():
        path, value = json.loads(line)
        if len(path)==1:
            assert path[0] not in meta
            meta[path[0]]=value
        else:
            index, key = path[1:]
            assert type(index) is int and isinstance(key,str)
            row=indexed.setdefault(index,{})
            assert key not in row
            row[key]=value
    assert set(meta)==set(META)
    assert meta['run_completed'] is True and not meta['error']
    assert meta['cleanup_errors']==meta['measurement_errors']==[]
    assert indexed and sorted(indexed)==list(range(len(indexed)))
    return meta, [indexed[i] for i in range(len(indexed))]


def main(campaign='d299'):
    global RAW, RUN, OUT
    reference = 'd297'
    health_name = 'remote_health2.json'
    change_caveat = 'Capacity, slot, rank and memory package changed together; not single-factor ablation.'
    if campaign == 'd300':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d300_20261007'
        RUN = RAW/'7b_full_w0'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d300_full_w0/context1'
        reference = 'd299'
        health_name = 'remote_health.json'
        change_caveat = 'Only the request-path scale-in latency reference changes; closed-loop trajectories may all change.'
    elif campaign == 'd303':
        RAW = ROOT/'results/ieee_tc/p2_backend_qualification/d303_20261007'
        RUN = RAW/'7b_full_w0'
        OUT = ROOT/'paper_results/ieee_tc/p2_backend/d303_full_w0/context1'
        reference = 'd300'
        health_name = 'remote_health.json'
        change_caveat = 'Same controls; only D301 confirmed-scope publication fix changes. Closed-loop trajectories may all change.'
    elif campaign != 'd299':
        raise ValueError('unknown frozen full-run identity')
    assert not OUT.exists()
    assert p.check_plan()=='0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7'
    assert sha256(ROOT/'docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md')=='5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22'
    audits, sources = {}, {}
    for name in (reference,campaign):
        directory=CURATED/f'{name}_full_w0/validated1'
        manifest=read(directory/'manifest.json')
        for filename,digest in manifest['outputs'].items():
            assert sha256(directory/filename)==digest
        audits[name]=read(directory/'audit.json')
        source=Path(audits[name]['evidence']['source'])
        assert sha256(source)==audits[name]['evidence']['source_sha256']
        sources[name]=source
        assert all(n==4000 for n in audits[name]['control_binding_pass_counts'].values())
    left, le, lr = audit_run(sources[reference])
    right, re, rr = audit_run(sources[campaign])
    assert len(lr)==len(rr)==4000
    assert all(all(a[k]==b[k] for k in FIELDS) for a,b in zip(lr,rr))
    assert all(le[k]==re[k] for k in ('shared_trace_sha256','shared_adapter_subset_sha256'))
    differences=[dict(request_id=a['request_id'],adapter_id=a['adapter_id'],
        reference_sha256=a['completion_token_ids_sha256'],candidate_sha256=b['completion_token_ids_sha256'])
        for a,b in zip(lr,rr) if a['completion_token_ids_sha256']!=b['completion_token_ids_sha256']]
    tiers=dict(Counter(r['readiness_tier_before_dispatch'] for r in rr))
    del lr,rr
    gc.collect()
    outcome=RUN/'launch.launch/physical_deployment/main_outcome.json'
    meta,transfers=project(outcome)
    journal=RAW/'remote_transfers.jsonl'
    remote_rows=[json.loads(line) for line in journal.read_text().splitlines()]
    remote={r['transfer_id']:r for r in remote_rows}
    assert len(remote)==len(remote_rows)==len(transfers)
    assert set(remote)=={r['http_transfer_id'] for r in transfers}
    clock=read(RAW/health_name)['health']['clock_id']
    for r in transfers:
        s=remote[r['http_transfer_id']]
        assert r['state']=='published' and r['content_verified'] and r['published_archive_verified']
        assert r['artifact_id']==s['artifact_id'] and r['remote_archive_sha256']==s['archive_sha256']
        assert r['transferred_bytes']==s['archive_bytes']==s['bytes_read']==s['bytes_written']
        assert s['clock_id']==clock and s['outcome']=='sent'
        assert not r['remote_pack_performed'] and not s['pack_performed'] and not s['temporary_created']
    samples=[]
    with (RUN/'launch.launch/watchdog.jsonl').open() as handle:
        for line in handle:
            r=json.loads(line)
            if r['event']=='resource_sample':
                assert not r['decision']['abort_reasons']
                samples.append(r)
    assert samples
    seal=p.verify_seal(ROOT/'paper_results/ieee_tc/safety/20260925_execution_start_protected.json')
    known='/home/qhq/serverless_llm_baselines/scripts/replay_openai_trace.py'
    assert seal['changed']==[known] and seal['entry_count']==147
    assert sha256(Path(known))=='04f3eb52c64c6d684dec3ecef1e6a450d687e0922430c2d77932917e4a38fc22'
    result=dict(kind=f'{campaign}_full_context_audit_v1',reference=le,candidate=re,
        input_identity_equal=True,input_fields_checked=FIELDS,output_hash_changes=differences,
        tier_counts=tiers,terminal_meta=meta,
        remote=dict(transfers=len(transfers),wire_bytes=sum(r['transferred_bytes'] for r in transfers),
                    pack_performed=False,clock=clock),
        resources=dict(samples=len(samples),peak_bytes=max(r['service']['memory.peak'] for r in samples),
            min_host_available_bytes=min(r['host']['available_bytes'] for r in samples),
            max_swap_bytes=max(r['service']['memory.swap.current'] for r in samples),
            memory_events={k:max(r['service']['memory.events'][k] for r in samples)
                           for k in ('high','max','oom','oom_kill')}),
        protection_raw=seal,n_correct=None,formal_qualification=False,no_figures=True,
        replicates_per_configuration=1,confidence_interval=None,
        caveats=['Full development W0, not held-out; no statistical superiority claim.',
                 change_caveat,
                 'Binding and native counts do not by themselves qualify kernel arithmetic.',
                 'Warm reference remains provisional; shared final references remain open.'])
    if campaign == 'd303':
        receipt = read(CURATED/'d303_full_w0/completion_receipt.json')
        assert receipt['source_result_sha256'] == re['source_sha256']
        assert receipt['remote_transfers_sha256'] == sha256(journal)
        assert receipt['remote_monitor_sha256'] == sha256(RAW/'remote_monitor.log')
        result['remote_shutdown'] = receipt['remote_shutdown']
        result['caveats'].append('Remote monitor exit73 is retained as a post-measurement shutdown-order guard; not all remote return codes are zero.')
    OUT.mkdir(parents=True,exist_ok=False)
    with (OUT/'context.json').open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    with (OUT/'comparison.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['metric','reference','candidate','lower_better_change_percent'])
        writer.writeheader()
        for key,a in audits[reference]['stats'].items():
            b=audits[campaign]['stats'][key]
            change=(a-b)/a*100 if a not in (0,None) and b is not None and key not in ('completed','n_correct') else None
            writer.writerow(dict(metric=key,reference=a,candidate=b,lower_better_change_percent=change))
    paths=[*sources.values(),outcome,journal,RAW/'remote_monitor.log',RAW/health_name,
           RUN/'launch.json',RUN/'launch.launch/watchdog.jsonl',Path(__file__),
           ROOT/'scripts/ieee_tc/plot_d290_diagnostic.py',ROOT/'scripts/ieee_tc_preflight.py']
    if campaign == 'd303':
        paths.append(CURATED/'d303_full_w0/completion_receipt.json')
    with (OUT/'manifest.json').open('x') as handle:
        json.dump(dict(source_refs=[dict(path=str(p),sha256=sha256(p)) for p in paths],
            outputs={p.name:sha256(p) for p in sorted(OUT.iterdir()) if p.name!='manifest.json'}),handle,indent=2)
    print(json.dumps({k:result[k] for k in ('remote','resources','tier_counts')},indent=2),flush=True)
    print(json.dumps(dict(input_identity_equal=True,output_hash_change_count=len(differences))),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', choices=('d299', 'd300', 'd303'), default='d299')
    main(parser.parse_args().campaign)
