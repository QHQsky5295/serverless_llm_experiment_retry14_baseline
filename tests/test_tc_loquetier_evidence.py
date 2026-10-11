"""CPU-only negative evidence tests; fixtures are not GPU/performance proof."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from scripts.ieee_tc.audit_d402_serverlessllm_path import (
    audit_loquetier_receipts, validate_loquetier_ready_row, validate_slora_allocation,
    validate_slora_preparation, compare_reference_row, digest)
from tests import test_tc_serverless_qualification_population as fixtures


class LoquetierEvidenceTests(unittest.TestCase):
    def fixture(self):
        cfg = dict(gpu_ids=[2,3], deployment_notice_s=1.,
            remote_config={'cache_dir':'/cache'},
            adapters={'a':[1,'/cache/objects/a'],'b':[2,'/cache/objects/b']})
        svc = dict(cgroup='/owned',affinity=[4,5])
        identity = dict(svc,pid=123,start_ticks=9)
        row = dict(schema='ieee_tc_loquetier_backend_ready_v1',index=1,clock_id='local',
            identity=identity,at=10.)
        process = dict(pid=123,start_ticks=9,clock_id='local')
        control = dict(tc_static_shard_index=1,instance_id='loquetier-1',tc_clock_id='local',
            tc_backend_entry_s=11.,tc_worker_ready_s=10.,tc_deployment_notice_s=1.,tc_worker_native_process=process)
        observation = dict(timing_contract='ieee_tc_loquetier_native_v1',native_process=process,
            native_adapter_id='b',native_adapter_path='/cache/objects/b',native_adapter_index=0,
            native_terminal_observed=True,native_forward_count=2,native_output_tokens=2,
            native_adapter_binding_source='loquetier_actual_forward_segments_and_startup_pointer_tables')
        plan = dict(shards=[dict(index=0,adapter_ids=['a']),dict(index=1,adapter_ids=['b'])])
        return row,cfg,control,svc,observation,plan

    def test_route_is_static_shard_not_request_ordinal_or_global_adapter_index(self):
        row,cfg,c,svc,o,plan=self.fixture()
        self.assertEqual(validate_loquetier_ready_row(row,cfg,c,svc,'local',o,plan),1)
        for k,v in [('tc_static_shard_index',0),('tc_static_shard_index',True),('tc_route_ordinal',1),('instance_id','loquetier-0')]:
            with self.subTest(k=k),self.assertRaises(ValueError):
                validate_loquetier_ready_row(row,cfg,dict(c,**{k:v}),svc,'local',o,plan)
        with self.assertRaises(ValueError):
            validate_loquetier_ready_row(row,cfg,c,svc,'local',dict(o,native_adapter_index=2),plan)

    def test_actual_native_process_forward_and_path_required(self):
        row,cfg,c,svc,o,plan=self.fixture()
        for k,v in [('native_adapter_id','unknown'),('native_adapter_path','/foreign/b'),
            ('native_adapter_index',True),('native_process',{}),('timing_contract','ieee_tc_native_v1'),
            ('native_forward_count',1),('native_forward_count',True),('native_terminal_observed',False),
            ('native_adapter_binding_source','slora_model_rpc_actual_batch_adapter')]:
            with self.subTest(k=k),self.assertRaises(ValueError):
                validate_loquetier_ready_row(row,cfg,c,svc,'local',dict(o,**{k:v}),plan)

    def test_ready_cannot_shift_notice_clock_or_resource_identity(self):
        row,cfg,c,svc,o,plan=self.fixture()
        for k,v in [('clock_id','foreign'),('at',12.),('at',float('nan')),
            ('index',0),('schema','ieee_tc_slora_backend_ready_v1')]:
            with self.subTest(k=k),self.assertRaises(ValueError):
                validate_loquetier_ready_row(dict(row,**{k:v}),cfg,c,svc,'local',o,plan)
        for k,v in [('pid',0),('start_ticks',None),('affinity',[0]),('cgroup','/foreign')]:
            with self.subTest(k=k),self.assertRaises(ValueError):
                validate_loquetier_ready_row(dict(row,identity=dict(row['identity'],**{k:v})),cfg,c,svc,'local',o,plan)
        for k,v in [('tc_worker_ready_s',9.),('tc_deployment_notice_s',0.),('tc_clock_id','foreign')]:
            with self.subTest(k=k),self.assertRaises(ValueError):
                validate_loquetier_ready_row(row,cfg,dict(c,**{k:v}),svc,'local',o,plan)

    def allocation(self,row):
        # Reuse the common physical fixture, but Loquetier's spawned process IS
        # its native worker. No invented separate S-LoRA API parent in evidence.
        a=fixtures.SloraEvidenceTests().allocation(dict(row,api_parent_identity=row['identity']))
        return a

    def test_native_worker_is_physical_owner_and_release_requires_actual_exit(self):
        row,cfg,*_=self.fixture()
        a=self.allocation(row)
        self.assertNotIn('api_parent_identity',row)
        validate_slora_allocation(a,row,cfg,1,'local',14.,backend='loquetier')
        for idx,key,value in [(1,'worker',{}),(3,'worker_pids',[]),(4,'at',13.)]:
            bad=copy.deepcopy(a);bad['events'][idx][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_slora_allocation(bad,row,cfg,1,'local',14.,backend='loquetier')
        a['events'][-1]['census']['devices'][0]['processes'][0]['kind']='compute'
        with self.assertRaises(ValueError):
            validate_slora_allocation(a,row,cfg,1,'local',14.,backend='loquetier')

    def test_full500_preparation_has_explicit_loquetier_identity(self):
        prep,remote=fixtures.SloraEvidenceTests().preparation()
        with self.assertRaises(ValueError):
            validate_slora_preparation(prep,set(prep),1.,501.,remote,backend='loquetier')
        for aid,row in prep.items():row['tc_artifact_request_id']='loquetier-full-pool:'+aid
        self.assertEqual(validate_slora_preparation(prep,set(prep),1.,501.,remote,backend='loquetier'),5000)
        with self.assertRaises(ValueError):
            validate_slora_preparation(prep,set(prep),1.,501.,remote)

    def test_reference_reuse_retains_cross_backend_divergence(self):
        o,c,s,r=fixtures.ReferenceReuseTests().fixture()
        o.pop('native_lora_name')
        o.update(timing_contract='ieee_tc_loquetier_native_v1',native_adapter_path='/objects/a')
        self.assertTrue(compare_reference_row(o,c,s,r)['exact_sequence_match'])
        o['completion_token_ids']=[1,3]
        o['completion_token_ids_sha256']=hashlib.sha256(b'[1,3]').hexdigest()
        self.assertFalse(compare_reference_row(o,c,s,r)['exact_sequence_match'])

    def test_retained_receipts_and_journals_join_with_native_validators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model7b';(model/'worker_receipts').mkdir(parents=True)
            row,cfg,c,svc,o,plan=self.fixture()
            cfg['source_sha256']={}
            worker=model/'worker_config.json';worker.write_text(json.dumps(cfg))
            prep=model/'loquetier_full_pool_preparation.json';prep.write_text('{}')
            http=root/'http.json';http.write_text('{}')
            requests,allocations=[],[]
            for i,aid in enumerate(('a','b')):
                r,control,observation=copy.deepcopy(row),copy.deepcopy(c),copy.deepcopy(o)
                rid=f'{i:032x}'
                r.update(index=i,receipt_id=rid,worker_config_sha256=digest(worker))
                r['identity']['pid']+=i
                process=dict(pid=r['identity']['pid'],start_ticks=9,clock_id='local')
                control.update(tc_static_shard_index=i,instance_id=f'loquetier-{i}',
                    tc_worker_receipt_id=rid,tc_worker_native_process=process)
                observation.update(native_adapter_id=aid,native_adapter_path=f'/cache/objects/{aid}',
                    native_process=process,native_request_id=f'r{i}',worker_completed_monotonic_s=14.)
                path=model/'worker_receipts'/(rid+'.json');path.write_text(json.dumps(r))
                control['tc_worker_receipt_sha256']=digest(path)
                observation['control_observation']=control
                requests.append(dict(request_id=f'r{i}',response={'metrics':{'ieee_tc':observation}}))
                a=self.allocation(r);a.update(lease_id=f'lease{i}',gpu_uuids=[f'GPU{i}'])
                for event in a['events']:
                    event.update(lease_id=f'lease{i}',owner_id=f'lease{i}',gpu_uuids=[f'GPU{i}'])
                    if 'census' in event:
                        event['census']['devices'][0].update(index=cfg['gpu_ids'][i],gpu_uuid=f'GPU{i}')
                        for p in event['census']['devices'][0]['processes']:
                            if p['kind']=='compute':p.update(pid=r['identity']['pid'],identity=r['identity'])
                    if event['event']=='native_workers':
                        event['workers'][0].update(pid=r['identity']['pid'],worker_rank=i,device_uuid=f'GPU{i}')
                    if event['event']=='native_workers_exited':event['worker_pids']=[r['identity']['pid']]
                journal=root/f'lease{i}.jsonl';journal.write_text(''.join(json.dumps(e)+'\n' for e in a['events']))
                a['journal']=str(journal);allocations.append(a)
            q=dict(schema='ieee_tc_loquetier_native_model_qualification_v1',configuration=copy.deepcopy(cfg),
                physical_lifecycle=dict(mode='loquetier_static_shards_v1',measurement_complete=True),
                source_sha256={},physical_allocations=allocations,requests=requests,service=svc)
            verified=[]
            def validate_receipt(r,**kw):
                self.assertEqual(kw['preparation_sha256'],digest(prep));verified.append(kw['index'])
            module=NS(validate_loquetier_service_config=lambda *a:None,verify_loquetier_source_view=lambda *a:None,
                load_loquetier_static_plan=lambda *a:plan,validate_loquetier_ready_receipt=validate_receipt)
            with patch.dict(sys.modules,replay_vllm_engine_trace=module):
                self.assertEqual(len(audit_loquetier_receipts(root,cfg,q,{'clock_id':'local'},http,None)),2)
                self.assertEqual(verified,[0,1])
                Path(allocations[0]['journal']).write_text('{}\n')
                with self.assertRaisesRegex(ValueError,'journal differs'):
                    audit_loquetier_receipts(root,cfg,q,{'clock_id':'local'},http,None)


if __name__=='__main__':unittest.main()
