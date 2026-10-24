"""Guard/config/native worker lifecycle tests. Synthetic bridge is not GPU inference."""
import asyncio
import ast
import copy
import inspect
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
MAIN=Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(MAIN))
import replay_vllm_engine_trace as service
import prepare_ieee_tc_serverless_stack as shared
import ieee_tc_serverless_measurement as measurement
from tests.test_ieee_tc_loquetier_http import preparation

PLAN=ROOT/'paper_results/ieee_tc/loquetier/d437_7b_static_capacity.json'
NATIVE=ROOT/'vendor_new_baselines/Loquetier_main_20260520'
PYTHON=Path('/home/qhq/anaconda3/envs/loquetier_20260520/bin/python')
KERNEL=PYTHON.parent.parent/'lib/python3.9/site-packages/loquetier/ops/_kernels.cpython-39-x86_64-linux-gnu.so'


def configs():
    plan=json.loads(PLAN.read_text())
    cfg=dict(schema='ieee_tc_loquetier_native_service_v1',python=sys.executable,
        native_python=str(PYTHON),source_view='/new/view',source_view_manifest_sha256='a'*64,
        static_plan=str(PLAN),static_plan_sha256=service._sha(PLAN),
        native_kernel=str(KERNEL),native_kernel_sha256=service._sha(KERNEL),
        routing='static_shards_v1',worker_startup='sequential_v1',
        generation_contract='fixed_length_greedy_v1',gpu_ids=[0,1,2,3],
        router_port=18091,worker_ports=[18092,18093,18094,18095],
        native_args=dict(model_dir=plan['base_model'],dtype='float16',batch_size=2,
                         context_tokens=1024,apply_scaling='A'))
    http=dict(schema='ieee_tc_loquetier_http_replay_v1',backbone=plan['base_model'],
        url='http://127.0.0.1:18090/v1/chat/completions',
        pool_index=plan['pool_index'],pool_index_sha256=plan['pool_index_sha256'],
        physical_lifecycle='loquetier_static_shards_v1',ingress_mode='service_pre_ready_v1',
        artifact_preparation='native_full_pool_before_service_v1',remote_artifacts={'real':True})
    return cfg,http,plan


class ConfigurationAndGuard(unittest.TestCase):
    def test_fixed_model_batch_comes_from_frozen_static_plan_not_7b_default(self):
        cfg, http, plan = configs()
        with tempfile.TemporaryDirectory() as tmp:
            plan['batch_size'] = 4
            path = Path(tmp)/'static.json'
            path.write_text(json.dumps(plan))
            cfg.update(static_plan=str(path), static_plan_sha256=service._sha(path))
            cfg['native_args']['batch_size'] = 4
            self.assertEqual(service.validate_loquetier_service_config(cfg, http), 18090)
            cfg['native_args']['batch_size'] = 2
            with self.assertRaisesRegex(ValueError, 'batch differs'):
                service.validate_loquetier_service_config(cfg, http)

    def test_model_kernel_binding_checks_backbone_receipt_and_actual_import_path(self):
        cfg, _, _ = configs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'python/loquetier/ops').mkdir(parents=True)
            (root/'python/loquetier/__init__.py').touch()
            kernel = root/'python/loquetier/ops/_kernels.so'
            kernel.write_bytes(b'synthetic CPU fixture, never imported')
            (root/'model').mkdir()
            (root/'model/config.json').write_text('{"model":"synthetic"}')
            (root/'kernel_src').mkdir()
            contract = root/'kernel_src/tc_model_contract.json'
            contract.write_text(json.dumps(dict(model_config_sha256=service._sha(root/'model/config.json'))))
            receipt = root/'receipt.json'
            data = dict(compiled=True, original_environment_modified=False,
                kernel_path=str(kernel), kernel_sha256=service._sha(kernel),
                model_contract_sha256=service._sha(contract))
            receipt.write_text(json.dumps(data))
            cfg.update(native_kernel=str(kernel), native_kernel_sha256=service._sha(kernel),
                model_kernel=dict(receipt=str(receipt), receipt_sha256=service._sha(receipt),
                                  python_root=str(root/'python')))
            cfg['native_args']['model_dir'] = str(root/'model')
            self.assertEqual(service.loquetier_model_kernel(cfg), str(root/'python'))
            env = service.loquetier_worker_environment(cfg, 0, {})
            self.assertEqual(env['PYTHONPATH'], str(root/'python')+':'+cfg['source_view'])
            for key, value in [('python_root', str(root/'other')), ('receipt_sha256', '0'*64)]:
                bad = dict(cfg, model_kernel=dict(cfg['model_kernel'], **{key:value}))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    service.loquetier_model_kernel(bad)
            (root/'model/config.json').write_text('{"model":"different"}')
            with self.assertRaisesRegex(ValueError, 'backbone/import identity'):
                service.loquetier_model_kernel(cfg)

    def capacity_config(self):
        cfg = json.loads((ROOT/'configs/ieee_tc/loquetier_7b_prepared_capacity_v1.json').read_text())
        _, http, plan = configs()
        cfg.update(router_port=18091, worker_ports=[18092,18093,18094,18095])
        return cfg, http, plan

    def test_new_capacity_config_binds_real_evidence_without_changing_static_placement(self):
        cfg, http, plan = self.capacity_config()
        self.assertEqual(service.validate_native_service_config(cfg, http), 18090)
        self.assertEqual(service.load_loquetier_static_plan(cfg), plan)
        self.assertEqual(plan['batch_size'], 2)
        model = service.loquetier_capacity_evidence(cfg, plan)
        self.assertEqual(model['candidate_batch_size'], 8)
        self.assertEqual(model['prompt_tokens'], 760)
        self.assertFalse(model['runtime_qualified'])
        with self.assertRaises(ValueError):
            service.validate_native_service_config(dict(cfg, inference_capacity=None), http)
        for key, value in [('policy', 'automatic_retry'), ('evidence_sha256', '0'*64)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                service.validate_native_service_config(dict(cfg, inference_capacity=dict(
                    cfg['inference_capacity'], **{key:value})), http)

    def test_capacity_rejects_incomplete_or_rebound_diagnostic_even_with_new_sha(self):
        cfg, http, _ = self.capacity_config()
        proof = json.loads(Path(cfg['inference_capacity']['evidence']).read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'evidence.json'
            for kind in ('partial', 'kernel', 'view', 'plan', 'release', 'batch', 'geometry',
                         'missing_mixed', 'kv_not_drained', 'one_output_missing', 'no_actual_mixed'):
                bad = copy.deepcopy(proof)
                if kind == 'partial': bad['passed'] = False
                elif kind in ('kernel','view','plan'):
                    key = dict(kernel='native_kernel_sha256',view='measurement_manifest_sha256',
                               plan='static_plan_sha256')[kind]
                    bad[key] = '0'*64
                elif kind == 'release': bad['forward_capacity']['release']['remaining_owners'] = 1
                elif kind == 'batch': bad['capacity_model']['candidate_batch_size'] = 9
                elif kind == 'geometry': bad['forward_capacity']['input']['cases'][0]['prompt_ids'].pop()
                elif kind == 'missing_mixed': bad['forward_capacity']['cases'].pop()
                elif kind == 'kv_not_drained': bad['forward_capacity']['cases'][-1]['samples'][-1]['live_kv_pages'] = 1
                elif kind == 'one_output_missing': bad['forward_capacity']['cases'][-1]['outputs'].pop()
                else:
                    for row in bad['forward_capacity']['cases'][-1]['samples']:
                        row['native_prefill_requests'] = 0
                path.write_text(json.dumps(bad))
                changed = dict(cfg, inference_capacity=dict(cfg['inference_capacity'],
                    evidence=str(path), evidence_sha256=service._sha(path)))
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    service.validate_native_service_config(changed, http)

    def test_actual_capacity_guard_accounts_cached_reservation_not_nvml_free_alone(self):
        cfg, _, plan = self.capacity_config()
        model = service.loquetier_capacity_evidence(cfg, plan)
        args = dict(allocated=model['warm_live_bytes'], reserved=23754440704,
            free=model['device_total_bytes']-23754440704-model['observed_nonallocator_bytes'],
            total=model['device_total_bytes'],kv_page_bytes=model['kv_page_bytes'],
            kv_page_tokens=model['kv_page_tokens'])
        receipt = service.loquetier_capacity_device_guard(model, **args)
        self.assertEqual(receipt['nonallocator_bytes'], model['observed_nonallocator_bytes'])
        self.assertLess(args['free'], 8*model['per_request_kv_bytes'])
        for key,value in [('allocated',model['warm_live_bytes']+1),('free',0),
                          ('total',model['device_total_bytes']-1),('kv_page_tokens',32)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                service.loquetier_capacity_device_guard(model, **dict(args, **{key:value}))

    def test_ready_receipt_requires_actual_owner_and_device_evidence(self):
        cfg, _, plan = self.capacity_config()
        cfg['adapters'] = {a:[i,'/remote/'+a] for s in plan['shards'] for i,a in enumerate(s['adapter_ids'])}
        cfg['native_environment'] = {'verified':'synthetic'}
        aids = plan['shards'][0]['adapter_ids']
        model = service.loquetier_capacity_evidence(cfg, plan)
        guard = service.loquetier_capacity_device_guard(model,
            allocated=model['warm_live_bytes'], reserved=23754440704,
            free=model['device_total_bytes']-23754440704-model['observed_nonallocator_bytes'],
            total=model['device_total_bytes'],kv_page_bytes=model['kv_page_bytes'],
            kv_page_tokens=model['kv_page_tokens'])
        raw = plan['shards'][0]['peft_original_fp32_bytes']
        capacity = dict(binding=cfg['inference_capacity'],model=model,device_guard=guard,
            release=dict(policy=cfg['inference_capacity']['policy'],remaining_owners=0,
                remaining_parameters=0,retained_layout_unchanged=True,raw_storage_bytes=raw,
                releasable_storage_bytes=raw,allocated_reduction_bytes=raw,aliased_storage_bytes=0))
        row = dict(schema='ieee_tc_loquetier_backend_ready_v1',index=0,identity={'pid':1},
            adapter_ids={a:i for i,a in enumerate(aids)},adapter_paths={a:cfg['adapters'][a][1] for a in aids},
            source_view_manifest_sha256=cfg['source_view_manifest_sha256'],
            static_plan_sha256=cfg['static_plan_sha256'],preparation_sha256='f'*64,
            native_environment=cfg['native_environment'],native_args=cfg['native_args'],
            native_kernel_sha256=cfg['native_kernel_sha256'],model_dir=cfg['native_args']['model_dir'],
            cuda_visible='0',cuda_device_count=1,runner_sha256=service._sha(service.__file__),
            ready_boundary='native_static_shard_and_model_initialized',
            registry=dict(adapter_count=125,pointer_tables_verified=True),
            allocated_bytes=guard['allocated_bytes'],reserved_bytes=guard['reserved_bytes'],
            inference_capacity=capacity)
        def validate(value):
            service.validate_loquetier_ready_receipt(value,cfg=cfg,index=0,plan=plan,
                identity={'pid':1},preparation_sha256='f'*64)
        validate(row)
        for kind in ('missing','owner','layout','freed','allocated','bytes'):
            bad = copy.deepcopy(row)
            if kind == 'missing': del bad['inference_capacity']
            elif kind == 'allocated': bad['allocated_bytes'] += 1
            elif kind == 'bytes': bad['inference_capacity']['device_guard']['nonallocator_bytes'] += 1
            else:
                key = dict(owner='remaining_owners',layout='retained_layout_unchanged',
                           freed='allocated_reduction_bytes')[kind]
                bad['inference_capacity']['release'][key] = 1 if kind=='owner' else 0
            with self.subTest(kind=kind),self.assertRaises(ValueError): validate(bad)

    def test_existing_full_500_static_plan_reused_without_trace(self):
        cfg,http,plan=configs()
        self.assertEqual(service.validate_native_service_config(cfg,http),18090)
        ids=[a for s in plan['shards'] for a in s['adapter_ids']]
        self.assertEqual(service.load_loquetier_static_plan(cfg,ids),plan)
        self.assertEqual(len(set(ids)),500)
        self.assertEqual([len(s['adapter_ids']) for s in plan['shards']],[125]*4)
        self.assertFalse(plan['runtime_qualified'])  # Metadata does not authorize GPU execution.

    def test_no_partial_pool_new_policy_or_changed_native_point(self):
        cfg,http,plan=configs()
        for key,value in [('routing','round_robin'),('worker_startup','parallel'),('gpu_ids',[0,0,2,3]),
                ('router_port',18090),('static_plan_sha256','0'*64),('generation_contract','natural')]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                service.validate_native_service_config(dict(cfg,**{key:value}),http)
        for key,value in [('batch_size',3),('context_tokens',2048),('dtype','bfloat16'),('apply_scaling',None)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                service.validate_native_service_config(dict(cfg,native_args=dict(cfg['native_args'],**{key:value})),http)
        with self.assertRaises(ValueError):
            service.load_loquetier_static_plan(cfg,plan['shards'][0]['adapter_ids'])
        for key,value in [('remote_artifacts',None),('pool_index_sha256','0'*64),
                          ('physical_lifecycle','vllm_resident_pool_v1')]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                service.validate_native_service_config(cfg,dict(http,**{key:value}))

    def test_placement_duplicates_and_reordered_shard_indices_rejected(self):
        cfg,_,plan=configs()
        with tempfile.TemporaryDirectory() as tmp:
            for key in ('duplicate','index','trace'):
                bad=copy.deepcopy(plan)
                if key=='duplicate':bad['shards'][1]['adapter_ids'][0]=bad['shards'][0]['adapter_ids'][0]
                elif key=='index':bad['shards'][0]['index']=1
                else:bad['no_trace_read']=False
                path=Path(tmp)/key;path.write_text(json.dumps(bad))
                with self.subTest(key=key),self.assertRaises(ValueError):
                    service.load_loquetier_static_plan(dict(cfg,static_plan=str(path),static_plan_sha256=service._sha(path)))

    def test_source_view_and_native_kernel_drift_rejected(self):
        cfg,_,_=configs()
        with tempfile.TemporaryDirectory() as tmp:
            view=Path(tmp)/'view'
            shared.prepare_loquetier_measurement_view(NATIVE,view,MAIN)
            cfg.update(main_repo=str(MAIN),source_view=str(view),
                       source_view_manifest_sha256=service._sha(view/'measurement_manifest.json'))
            service.verify_loquetier_source_view(cfg)
            with self.assertRaises(ValueError):
                service.verify_loquetier_source_view(dict(cfg,native_kernel_sha256='0'*64))
            (view/'loquetier_src/model_generation.py').write_text('# changed view only\n')
            with self.assertRaises(ValueError):
                service.verify_loquetier_source_view(cfg)

    def test_existing_environment_is_explicit_offline_without_foreign_pythonpath(self):
        cfg,_,_=configs()
        source=dict(PATH='/usr/bin',PYTHONHOME='/foreign',PYTHONPATH='/foreign',KEEP='unchanged')
        env=service.loquetier_worker_environment(cfg,2,source)
        self.assertEqual(env['PYTHONPATH'],cfg['source_view'])
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'2')
        self.assertEqual(env['TRANSFORMERS_OFFLINE'],'1')
        self.assertEqual(env['HF_DATASETS_OFFLINE'],'1')
        self.assertNotIn('PYTHONHOME',env)
        self.assertEqual(source['PYTHONHOME'],'/foreign')
        self.assertTrue(env['PATH'].startswith(str(PYTHON.parent)+':'))

    def test_native_context_rejects_unfrozen_source_before_guard_or_backend(self):
        cfg,_,_=configs()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'worker.json';p.write_text(json.dumps(cfg))
            with patch.object(shared,'load_guard') as guard,self.assertRaises(ValueError):
                service.loquetier_worker_context(p,0)
            guard.assert_not_called()

    def test_native_context_accepts_only_current_remote_pool_and_after_notice_preparation(self):
        cfg,_,plan=configs()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);view=root/'view'
            shared.prepare_loquetier_measurement_view(NATIVE,view,MAIN)
            cfg.update(main_repo=str(MAIN),source_view=str(view),output=str(root),
                native_python=sys.executable,source_view_manifest_sha256=service._sha(view/'measurement_manifest.json'),
                deployment_notice_s=time.perf_counter(),source_sha256={str(ROOT/'scripts'/n):service._sha(ROOT/'scripts'/n)
                    for n in ('replay_vllm_engine_trace.py','ieee_tc_serverless_measurement.py','prepare_ieee_tc_serverless_stack.py')})
            ids=[a for s in plan['shards'] for a in s['adapter_ids']]
            cfg.update(adapters={a:[i,str(root/'remote/objects'/a)] for i,a in enumerate(ids)},
                remote_config=dict(cache_dir=str(root/'remote'),content_manifest=plan['pool_index'],
                                   content_manifest_file_sha256=plan['pool_index_sha256']))
            env=service.loquetier_worker_environment(dict(cfg,native_python=str(PYTHON)),0,{})
            cfg['native_environment']={k:env[k] for k in service.LOQUETIER_ENV_KEYS}
            path=root/'worker.json';path.write_text(json.dumps(cfg))
            prep={a:preparation(a) for a in ids}
            pp=root/'loquetier_full_pool_preparation.json';pp.write_text(json.dumps(prep))
            env['FAASLORA_TC_LOQUETIER_WORKER_CONFIG_SHA256']=service._sha(path)
            guard=NS(verify_current_service=Mock())
            with patch.dict(os.environ,env),patch.object(service.sys,'version_info',(3,9)), \
                    patch.object(shared,'load_guard',return_value=guard):
                got=service.loquetier_worker_context(path,0)
                self.assertEqual(got[2],plan);self.assertEqual(got[3],prep)
                guard.verify_current_service.assert_called_once_with()
                guard.verify_current_service.side_effect=RuntimeError('outside admitted service')
                with self.assertRaisesRegex(RuntimeError,'outside admitted service'):
                    service.loquetier_worker_context(path,0)
                guard.verify_current_service.side_effect=None
                prep[ids[0]]['tc_artifact_resolve_started_s']=cfg['deployment_notice_s']-1
                pp.write_text(json.dumps(prep))
                with self.assertRaisesRegex(ValueError,'predates'):
                    service.loquetier_worker_context(path,0)
                cfg['adapters'][ids[0]][1]='/original/frozen/'+ids[0]
                path.write_text(json.dumps(cfg))
                os.environ['FAASLORA_TC_LOQUETIER_WORKER_CONFIG_SHA256']=service._sha(path)
                with self.assertRaisesRegex(ValueError,'common remote objects'):
                    service.loquetier_worker_context(path,0)

    def test_native_factory_guard_is_before_torch_import(self):
        cfg,_,plan=configs()
        guard=NS(verify_current_service=Mock(side_effect=RuntimeError('uncontained')))
        with patch.object(shared,'load_guard',return_value=guard),self.assertRaisesRegex(RuntimeError,'uncontained'):
            service._load_loquetier_native_bridge(dict(cfg,main_repo=str(MAIN)),0,plan,lambda row:None)
        tree=ast.parse(inspect.getsource(service._load_loquetier_native_bridge))
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                  and n.func.attr=='verify_current_service')
        imp=next(n for n in ast.walk(tree) if isinstance(n,ast.Import) and n.names[0].name=='torch')
        self.assertLess(call.lineno,imp.lineno)

    def test_cli_has_explicit_native_mode_without_legacy_replay_import(self):
        with patch.object(service,'_native_loquetier_worker') as worker:
            async def empty(*args):pass
            worker.side_effect=empty
            self.assertEqual(service.native_http_main(['native-loquetier-worker','--config','/frozen','--index','0']),0)
            worker.assert_called_once_with(Path('/frozen'),0)


class NativeWorkerLifecycle(unittest.IsolatedAsyncioTestCase):
    async def exercise(self,mode):
        import aiohttp
        loop=asyncio.get_running_loop()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'worker_receipts').mkdir()
            sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
            cfg,_,_=configs();cfg.update(output=str(root),model='base',gpu_ids=[0],worker_ports=[port],
                adapters={'a':[0,'/owned/a']},native_environment={},deployment_notice_s=time.perf_counter())
            plan=dict(shards=[dict(index=0,adapter_ids=['a'])])
            prepared={'a':preparation()}
            shared_path=root/'loquetier_full_pool_preparation.json';service._exclusive_json(shared_path,prepared)
            config_path=root/'worker.json';service._exclusive_json(config_path,cfg)
            native_process=measurement._slora_process_identity(os.getpid())
            identity=dict(native_process,uid=os.getuid(),cgroup='synthetic')
            guard=NS(gpu_process_identity=lambda pid:identity)
            closed=[]
            class Bridge:
                adapters=frozenset(('a',));batch_size=2;process=native_process
                startup=dict(registry=dict(adapter_count=1,pointer_tables_verified=True),
                    adapter_ids={'a':0},adapter_paths={'a':'/owned/a'},cuda_device_count=1,
                    model_dir=cfg['native_args']['model_dir'],allocated_bytes=1,reserved_bytes=1,
                    native_kernel_sha256=cfg['native_kernel_sha256'])
                def add(self,*args):return 0
                def step(self):raise RuntimeError('synthetic native forward failure')
                def close(self):closed.append(True);return []
            def factory(*args):
                if mode=='init_failure':raise RuntimeError('synthetic initialization failure')
                return Bridge()
            with patch.object(service,'loquetier_worker_context',return_value=(cfg,guard,plan,prepared)), \
                    patch.object(service,'_load_loquetier_native_bridge',side_effect=factory), \
                    patch.object(loop,'add_signal_handler') as signal_handler, \
                    patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0'):
                task=asyncio.create_task(service._native_loquetier_worker(config_path,0))
                try:
                    if mode=='init_failure':
                        with self.assertRaisesRegex(RuntimeError,'initialization failure'):
                            await asyncio.wait_for(task,5.)
                        self.assertFalse((root/'worker_0.ready.json').exists())
                    else:
                        for _ in range(1000):
                            if task.done():await task
                            if (root/'worker_0.ready.json').exists():break
                            await asyncio.sleep(.001)
                        ready=json.loads((root/'worker_0.ready.json').read_text())
                        receipt=json.loads((root/'worker_receipts'/(ready['tc_worker_receipt_id']+'.json')).read_text())
                        service.validate_loquetier_ready_receipt(receipt,cfg=cfg,index=0,plan=plan,
                            identity=identity,preparation_sha256=service._sha(shared_path))
                        for key,value in [('adapter_ids',{'a':1}),('identity',{}),('preparation_sha256','0'*64),
                                ('native_kernel_sha256','0'*64),('cuda_visible','1')]:
                            with self.subTest(key=key),self.assertRaises(ValueError):
                                service.validate_loquetier_ready_receipt(dict(receipt,**{key:value}),cfg=cfg,
                                    index=0,plan=plan,identity=identity,preparation_sha256=service._sha(shared_path))
                        async with aiohttp.ClientSession(trust_env=False) as session:
                            async with session.get(f'http://127.0.0.1:{port}/health') as response:
                                health=await response.json();self.assertEqual(response.status,200)
                                self.assertEqual(health['process'],native_process)
                            if mode=='forward_failure':
                                body=dict(model='base',request_id='r',canonical_prompt='prompt',input_tokens=[1,2],
                                    lora_adapter_name='a',max_tokens=2,temperature=0,top_p=1,ignore_eos=True,
                                    stop=[],stop_token_ids=[],stream=False,_sllm_internal_metrics=dict(
                                        tc_clock_id=native_process['clock_id'],instance_id='loquetier-0',tc_static_shard_index=0))
                                try:
                                    async with session.post(f'http://127.0.0.1:{port}/v1/chat/completions',json=body) as response:
                                        self.assertGreaterEqual(response.status,500)
                                except aiohttp.ClientConnectionError:
                                    pass  # The worker is allowed to close the connection on fatal failure.
                                with self.assertRaisesRegex(RuntimeError,'queue stopped'):
                                    await asyncio.wait_for(task,5.)
                            else:
                                signal_handler.call_args.args[1]()
                                await asyncio.wait_for(task,5.)
                        self.assertEqual(closed,[True])
                    events=[json.loads(x) for x in (root/'worker_0.native_events.jsonl').read_text().splitlines()]
                    self.assertEqual(events[-1]['event'],'loquetier_worker_exiting')
                    self.assertFalse(events[-1]['gpu_release_proven'])
                    if mode!='normal':self.assertIn('loquetier_worker_failed',[r['event'] for r in events])
                finally:
                    if not task.done():
                        signal_handler.call_args.args[1]()
                        await asyncio.wait_for(task,5.)

    async def test_worker_ready_health_and_owned_normal_exit(self):await self.exercise('normal')
    async def test_initialization_failure_publishes_no_ready(self):await self.exercise('init_failure')
    async def test_native_forward_failure_stops_worker_without_retry(self):await self.exercise('forward_failure')


class ParentServiceAssembly(unittest.IsolatedAsyncioTestCase):
    """Real loopback parent path, synthetic workers/census (not GPU evidence)."""

    async def exercise(self, corrupt_census=False):
        import aiohttp
        from aiohttp import web
        from faaslora.clock import local_monotonic_clock_id
        from faaslora.datasets.workload_generator import FrozenReplayPlan
        import faaslora.metrics.metrics_collector as physical
        clock=local_monotonic_clock_id();notice=time.perf_counter()
        order=[];spawned={};receipts={};calls=[];runners=[];offered=[]
        preparation_started=asyncio.Event();allow_preparation=asyncio.Event()
        def identity(pid):return dict(pid=pid,start_ticks=pid+100,uid=os.getuid(),cgroup='synthetic')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'output';cfg,http,placement=configs()
            try:
                ports=[]
                for i in range(4):
                    async def health(request,index=i):
                        row=receipts[index]
                        return web.json_response(dict(ready=True,receipt_id=row['receipt_id'],
                            process=dict(pid=row['identity']['pid'],start_ticks=row['identity']['start_ticks'],clock_id=clock)))
                    async def generate(request,index=i):
                        body=await request.json();calls.append((index,body))
                        self.assertIn(('confirmed',index),order)
                        self.assertIn(body['lora_adapter_name'],placement['shards'][index]['adapter_ids'])
                        return web.json_response(dict(id=body['request_id'],synthetic_cpu_only=True))
                    app=web.Application();app.router.add_get('/health',health)
                    app.router.add_post('/v1/chat/completions',generate)
                    runner=web.AppRunner(app,access_log=None);await runner.setup();runners.append(runner)
                    site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
                    ports.append(site._server.sockets[0].getsockname()[1])
                sockets=[socket.socket(),socket.socket()]
                for sock in sockets:sock.bind(('127.0.0.1',0))
                ingress_port,router_port=[s.getsockname()[1] for s in sockets]
                for sock in sockets:sock.close()
                view=root/'view';shared.prepare_loquetier_measurement_view(NATIVE,view,MAIN)
                cfg.update(source_view=str(view),source_view_manifest_sha256=service._sha(view/'measurement_manifest.json'),
                           router_port=router_port,worker_ports=ports)
                http.update(main_repo=str(MAIN),model='base',url=f'http://127.0.0.1:{ingress_port}/v1/chat/completions')
                config_path,http_path=root/'config.json',root/'http.json'
                service._exclusive_json(config_path,cfg);service._exclusive_json(http_path,http)
                paths={a:str(root/'private/objects'/a) for s in placement['shards'] for a in s['adapter_ids']}
                # Reverse the first two target shards to detect accidental ordinal RR.
                requested=[placement['shards'][i]['adapter_ids'][0] for i in (3,0)]
                trace=root/'trace.json';trace.write_text(json.dumps(dict(requests=[
                    dict(request_id=f'r{i}',arrival_time_s=float(i),adapter_id=a,expected_output_tokens=2)
                    for i,a in enumerate(requested)])))
                plan=FrozenReplayPlan.load(trace);replay=root/'replay.jsonl';replay.touch()
                context=dict(transport='http',config_sha256=service._sha(http_path),result_path=str(replay),
                    deployment_notice_s=notice,replay_t0_s=notice+60.,clock_id=clock,plan=plan.identity())
                launch=root/'launch.json';service._exclusive_json(launch,dict(external_replay=context))
                async def prepare(resolver,received_paths,received_notice,*,backend):
                    self.assertEqual(received_paths,paths);self.assertEqual(received_notice,notice)
                    self.assertEqual(backend,'loquetier')
                    order.append(('prepare',500));preparation_started.set();await allow_preparation.wait()
                    order.append(('prepared',500));return {a:preparation(a) for a in paths}
                class Census:
                    def __init__(self,api):pass
                    def sample(self,path):return dict(devices=[dict(gpu_uuid=f'GPU-{i}',processes=[dict(
                        pid=p.pid+(1 if corrupt_census else 0),service_member=True,kind='compute')])
                        for i,p in spawned.items() if p.poll() is None])
                    def close(self):pass
                class Owner:
                    def __init__(self,**kw):
                        self.index=kw['device_indices'][0];self.census=kw['census']
                        self.gpu_uuids=[f'GPU-{self.index}'];self.events=[];self.process=None
                        order.append(('allocate',self.index))
                    def bind_process(self,p,ident):self.process=p
                    def confirm_workers(self,workers):
                        self.events.append(dict(event='native_workers'));order.append(('confirmed',self.index))
                    async def wait_workers(self,timeout_s):
                        assert self.process.poll() is not None;order.append(('exited',self.index))
                    def release(self):
                        assert self.process.poll() is not None;order.append(('released',self.index))
                    def evidence(self):return dict(synthetic_cpu_only=True)
                class Deployment:
                    def __init__(self,root,**kw):
                        self.root=root;root.mkdir();self.allocations=root/'allocations';self.allocations.mkdir()
                    def bind_result_file(self,path):pass
                    def summarize(self,**kw):return dict(measurement_complete=True,n_native_contract_complete=2)
                    def finalize(self):order.append(('finalized',None))
                class Process:
                    returncode=None
                    def __init__(proc,argv,*,env,**kw):
                        self.assertEqual(argv[3],'native-loquetier-worker')
                        index=int(argv[-1]);proc.pid=900000+index;proc.index=index
                        order.append(('spawned',index));spawned[index]=proc
                        worker_path=Path(argv[5]);worker=json.loads(worker_path.read_text())
                        self.assertEqual(env['FAASLORA_TC_LOQUETIER_WORKER_CONFIG_SHA256'],service._sha(worker_path))
                        aids=placement['shards'][index]['adapter_ids'];rid=f'{index+1:032x}'
                        row=dict(schema='ieee_tc_loquetier_backend_ready_v1',receipt_id=rid,index=index,
                            at=time.perf_counter(),clock_id=clock,identity=identity(proc.pid),
                            worker_config_sha256=service._sha(worker_path),runner_sha256=service._sha(service.__file__),
                            adapter_ids={a:i for i,a in enumerate(aids)},adapter_paths={a:paths[a] for a in aids},
                            source_view_manifest_sha256=cfg['source_view_manifest_sha256'],
                            static_plan_sha256=cfg['static_plan_sha256'],native_kernel_sha256=cfg['native_kernel_sha256'],
                            native_args=worker['native_args'],native_environment=worker['native_environment'],
                            model_dir=worker['native_args']['model_dir'],cuda_visible=str(index),cuda_device_count=1,
                            registry=dict(adapter_count=125,pointer_tables_verified=True),allocated_bytes=1,reserved_bytes=1,
                            preparation_sha256=service._sha(output/'loquetier_full_pool_preparation.json'),
                            ready_boundary='native_static_shard_and_model_initialized')
                        receipts[index]=row;path=output/'worker_receipts'/(rid+'.json');service._exclusive_json(path,row)
                        service._exclusive_json(output/f'worker_{index}.ready.json',dict(
                            tc_worker_receipt_id=rid,tc_worker_receipt_sha256=service._sha(path)))
                    def poll(self):return self.returncode
                    def terminate(self):order.append(('terminated',self.index));self.returncode=0
                    def wait(self,timeout):return self.returncode
                guard=NS(verify_current_service=lambda:dict(service_identity=dict(path='/sys/fs/cgroup/synthetic')),
                    gpu_process_identity=identity,load_nvml_binding=lambda *a:None,NativeGPUCensus=Census)
                env=dict(FAASLORA_TC_LAUNCH_RECEIPT=str(launch),FAASLORA_TC_NVML_BINDING='/synthetic',
                         FAASLORA_TC_NVML_SHA256='f'*64)
                with patch.dict(os.environ,env),patch.object(shared,'load_guard',return_value=guard), \
                        patch.object(shared,'load_http_replay_plan',return_value=plan), \
                        patch.object(shared,'prepare_remote_artifacts',return_value=({},paths)), \
                        patch.object(shared,'prepare_slora_full_pool',side_effect=prepare), \
                        patch.object(shared,'record_http_physical_terminals'), \
                        patch.object(measurement,'PublishedArtifactResolver',return_value=object()), \
                        patch.object(service.subprocess,'Popen',Process), \
                        patch.object(asyncio.get_running_loop(),'add_signal_handler'), \
                        patch.object(physical,'PhysicalGPUDeployment',Deployment),patch.object(physical,'PhysicalGPUAllocation',Owner):
                    running=asyncio.create_task(service._resident_http_service(config_path,http_path,output,root/'private'))
                    try:
                        await asyncio.wait_for(preparation_started.wait(),5.)
                        self.assertEqual(order,[('prepare',500)])
                        async with aiohttp.ClientSession(trust_env=False) as session:
                            async def request(i,aid):
                                async with session.post(http['url'],json=dict(model='base',request_id=f'r{i}',
                                        lora_adapter_name=aid,stream=False)) as response:
                                    self.assertEqual(response.status,200);return await response.json()
                            if not corrupt_census:
                                offered=[asyncio.create_task(request(i,a)) for i,a in enumerate(requested)]
                                for _ in range(1000):
                                    if (output/'ingress.jsonl').read_text().count('ingress_received')==2:break
                                    await asyncio.sleep(.001)
                                self.assertEqual((output/'ingress.jsonl').read_text().count('ingress_received'),2)
                                self.assertFalse(any(t.done() for t in offered));self.assertEqual(len(spawned),0)
                            allow_preparation.set()
                            if corrupt_census:
                                with self.assertRaisesRegex(RuntimeError,'qualification failed'):
                                    await asyncio.wait_for(running,5.)
                                failed=json.loads((output/'model_qualification.json').read_text())
                                self.assertFalse(failed['passed']);self.assertIn('census',failed['error'])
                                self.assertEqual(len(spawned),1);self.assertFalse(calls)
                            else:
                                responses=await asyncio.wait_for(asyncio.gather(*offered),5.)
                                with replay.open('a') as handle:
                                    for i,body in enumerate(responses):handle.write(json.dumps(dict(
                                        event='http_raw_response',request_id=f'r{i}',body=body,status=200))+'\n')
                                    handle.write(json.dumps(dict(event='http_replay_complete',N_failed=0,N_response=2))+'\n')
                                result=await asyncio.wait_for(running,5.)
                                self.assertTrue(result['passed'])
                                self.assertEqual(result['physical_lifecycle']['mode'],'loquetier_static_shards_v1')
                                self.assertEqual({body['request_id']:i for i,body in calls},{'r0':3,'r1':0})
                                for i in range(3):self.assertLess(order.index(('confirmed',i)),order.index(('allocate',i+1)))
                            for i in spawned:self.assertLess(order.index(('terminated',i)),order.index(('released',i)))
                    finally:
                        allow_preparation.set()
                        for task in offered:
                            if not task.done():task.cancel()
                        await asyncio.gather(*offered,return_exceptions=True)
                        if not running.done():running.cancel()
                        await asyncio.gather(running,return_exceptions=True)
            finally:
                for runner in runners:await runner.cleanup()

    async def test_full_pool_preready_sequential_static_route_and_release(self):await self.exercise()
    async def test_wrong_gpu_process_rejects_ready_without_dispatch(self):await self.exercise(True)


if __name__=='__main__':unittest.main()
