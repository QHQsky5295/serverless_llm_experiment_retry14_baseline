"""CPU transport integration, not a native model/performance qualification."""
import asyncio
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(MAIN))
import replay_vllm_engine_trace as service
import prepare_ieee_tc_serverless_stack as shared
import ieee_tc_serverless_measurement as measurement
from faaslora.clock import local_monotonic_clock_id
from faaslora.datasets.workload_generator import FrozenReplayPlan
from faaslora.metrics.metrics_collector import PhysicalGPUDeployment


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def preparation(aid='a', start=None):
    start = time.perf_counter() if start is None else start
    return dict(tc_artifact_id=aid, tc_artifact_request_id='slora-full-pool:'+aid,
        tc_artifact_cache_hit=False, tc_artifact_content_sha256='a'*64,
        tc_artifact_manifest_sha256='b'*64, tc_artifact_receipt_sha256='c'*64,
        tc_artifact_transfer_id='d'*32, tc_artifact_transferred_bytes=1024,
        tc_artifact_delivery_mode='prepublished_gzip_v1',
        tc_artifact_resolve_started_s=start, tc_artifact_resolve_completed_s=start)


def native_configs():
    http = dict(schema='ieee_tc_slora_http_replay_v1', backbone='/model',
        url='http://127.0.0.1:18090/v1/chat/completions',
        physical_lifecycle='slora_resident_pool_v1', ingress_mode='service_pre_ready_v1',
        artifact_preparation='native_full_pool_before_service_v1', remote_artifacts={'real': True})
    args = dict(model_dir='/model', tokenizer_mode='auto', tp=1, max_req_total_len=1024,
        swap=True, scheduler='slora', dummy=False, bmm=False, no_lora=False,
        no_kernel=False, no_lora_compute=False, no_lora_swap=False, no_mem_pool=False,
        profile=False, enable_abort=False, prefetch=False, prefetch_size=0, pool_size_lora=0,
        batch_num_adapters=None, fair_weights=[], trust_remote_code=True, disable_log_stats=True,
        eos_id=2, log_stats_interval=10, max_req_input_len=768, max_total_token_num=14000,
        batch_max_tokens=1024, running_max_req_size=1000)
    cfg = dict(schema='ieee_tc_slora_native_service_v1', python=sys.executable,
        native_python=sys.executable, source_view='/source/view', native_args=args,
        host_layout='packed_views_v1', routing='round_robin', worker_startup='sequential_v1',
        generation_contract='fixed_length_greedy_v1', gpu_ids=[0, 1], router_port=18091,
        worker_ports=[18092, 18093], nccl_ports=[29630, 29631])
    return cfg, http


class NativeConfiguration(unittest.TestCase):
    def test_explicit_source_and_native_modes_not_new_policy(self):
        cfg, http = native_configs()
        self.assertEqual(service.validate_native_service_config(cfg, http), 18090)
        for key, value in [('worker_startup', 'parallel'), ('routing', 'least_loaded'),
                           ('nccl_ports', [18092, 29631]), ('gpu_ids', [0, 0]), ('host_layout', 'auto')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                service.validate_native_service_config(dict(cfg, **{key: value}), http)
        for key, value in [('dummy', True), ('no_lora', True), ('tp', 2), ('prefetch', True),
                           ('max_req_total_len', 2048), ('scheduler', 'peft'), ('batch_max_tokens', 999)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                service.validate_native_service_config(dict(cfg, native_args=dict(cfg['native_args'], **{key: value})), http)

    def test_environment_reuses_native_library_without_dirty_pythonpath(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'bin').mkdir()
            (root/'bin/python').symlink_to(sys.executable)
            (root/'lib/python3.9/site-packages/torch/lib').mkdir(parents=True)
            cfg = dict(native_python=str(root/'bin/python'), source_view='/explicit/view')
            source = dict(PATH='/usr/bin', PYTHONPATH='/dirty', PYTHONHOME='/dirty',
                          LD_LIBRARY_PATH='/old', KEEP='kept')
            env = service.slora_worker_environment(cfg, 2, source)
            self.assertEqual(env['PYTHONPATH'], '/explicit/view')
            self.assertEqual(env['CUDA_HOME'], str(root))
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '2')
            self.assertTrue(env['PATH'].startswith(str(root/'bin')+':'))
            self.assertTrue(env['LD_LIBRARY_PATH'].endswith(':'+str(root/'lib')+':/old'))
            self.assertNotIn('PYTHONHOME', env)
            self.assertEqual(source['PYTHONPATH'], '/dirty')

    def test_native_context_checks_complete_source_before_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            view = root/'view'
            shared.prepare_slora_measurement_view(ROOT/'repos/S-LoRA', view, MAIN,
                host_layout='packed_views_v1', worker_lifecycle=True)
            cfg, _ = native_configs()
            cfg.update(main_repo=str(MAIN), source_view=str(view),
                native_environment={k:str(view) if k == 'PYTHONPATH' else os.environ.get(k,'') for k in service.SLORA_ENV_KEYS},
                source_view_manifest_sha256=service._sha(view/'measurement_manifest.json'),
                source_sha256={str(ROOT/'scripts'/p): service._sha(ROOT/'scripts'/p) for p in (
                    'replay_vllm_engine_trace.py','ieee_tc_serverless_measurement.py','prepare_ieee_tc_serverless_stack.py')},
                adapters={f'a{i:03}': [i, f'/owned/a{i:03}'] for i in range(500)},
                native_input_params={k: cfg['native_args'][k] for k in service.SLORA_INPUT_KEYS})
            path = root/'worker.json'
            service._exclusive_json(path, cfg)
            env = dict(cfg['native_environment'], FAASLORA_TC_SLORA_WORKER_CONFIG_SHA256=service._sha(path),
                       CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(view))
            guard = NS(verify_current_service=Mock())
            with patch.dict(os.environ, env), patch.object(service.sys, 'version_info', (3, 9)), \
                    patch.object(shared, 'load_guard', return_value=guard):
                got, returned = service.slora_worker_context(path, 0)
                self.assertEqual(got, cfg)
                self.assertIs(returned, guard)
                guard.verify_current_service.assert_called_once()
                guard.verify_current_service.reset_mock()
                with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='1'), self.assertRaises(ValueError):
                    service.slora_worker_context(path, 0)
                guard.verify_current_service.assert_not_called()
                (view/'slora/server/api_server.py').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'guarded source'):
                    service.slora_worker_context(path, 0)
                guard.verify_current_service.assert_not_called()

    def test_bootstrap_refuses_before_backend_import_without_worker_identity(self):
        with patch.object(service, 'slora_worker_context', side_effect=ValueError('guarded')):
            with self.assertRaisesRegex(ValueError, 'guarded'):
                service._native_slora_worker(Path('/not/a/worker'), 0)

    def test_bootstrap_arguments_match_actual_native_parser(self):
        cfg, _ = native_configs()
        cfg['adapters'] = {f'a{i:03}':[i,f'/owned/a{i:03}'] for i in range(500)}
        api_path = ROOT/'repos/S-LoRA/slora/server/api_server.py'
        native_main = next(n for n in ast.parse(api_path.read_text()).body
                          if isinstance(n, ast.FunctionDef) and n.name == 'main')
        parser_body = []
        for node in native_main.body:
            parser_body.append(node)
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'args' for t in node.targets):
                break
        parsed = []
        def main():
            ns = dict(argparse=argparse)
            exec(compile(ast.Module(body=parser_body,type_ignores=[]),'<native argparse>','exec'),ns)
            parsed.append(ns['args'])
        api = NS(__file__=str(api_path), main=main)
        torch = NS(multiprocessing=NS(set_start_method=Mock()))
        with patch.object(service,'slora_worker_context',return_value=(cfg,object())), \
                patch.object(service.signal,'signal'), patch.object(sys,'argv',list(sys.argv)), \
                patch.dict(sys.modules,{'torch':torch,'slora.server':NS(api_server=api)}):
            service._native_slora_worker(Path('/fixture/config'),0)
        args = vars(parsed[0])
        for key,value in cfg['native_args'].items():
            self.assertEqual(args[key],value,key)
        self.assertEqual(args['lora_dirs'],[p for _,p in cfg['adapters'].values()])
        self.assertEqual(args['port'],cfg['worker_ports'][0])
        self.assertEqual(args['nccl_port'],cfg['nccl_ports'][0])
        torch.multiprocessing.set_start_method.assert_called_once_with('spawn')


class NativeLifecycleSource(unittest.TestCase):
    def source(self):
        return shared.slora_measurement_sources(ROOT/'repos/S-LoRA',
            host_layout='packed_views_v1', worker_lifecycle=True)

    def test_opt_in_boundaries_and_original_api_statements(self):
        source = self.source()
        rpc = source['slora/server/router/model_infer/model_rpc.py']
        self.assertLess(rpc.index('slora_worker_launch_guard(world_size)'), rpc.index("dist.init_process_group('nccl'"))
        self.assertGreater(rpc.index('slora_record_worker_ready(self'), rpc.index('set_random_seed(2147483647)'))
        original = ast.parse((ROOT/'repos/S-LoRA/slora/server/api_server.py').read_text())
        measured = ast.parse(source['slora/server/api_server.py'])
        old = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        new = next(n for n in measured.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        wrapped = new.body[-1]
        self.assertIsInstance(wrapped, ast.Try)
        self.assertEqual(ast.dump(ast.Module(body=new.body[:-2]+wrapped.body, type_ignores=[])),
                         ast.dump(ast.Module(body=old.body, type_ignores=[])))
        self.assertEqual(len(wrapped.finalbody), 2)

    def execute_native_main(self, *, fail_start=False, fail_serve=False):
        source = self.source()['slora/server/api_server.py']
        main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        children = []
        class Process:
            pid = None
            alive = False
            def __init__(self, **kwargs):
                children.append(self)
            def start(self):
                self.pid, self.alive = len(children)+100, True
                if fail_start:
                    raise RuntimeError('start failure')
            def is_alive(self):
                return self.alive
            def terminate(self):
                self.alive = False
            def join(self, timeout):
                pass
            def kill(self):
                self.alive = False
        def serve(*a, **kw):
            if fail_serve:
                raise RuntimeError('serve failure')
        module = NS(slora_join_owned_children=measurement.slora_join_owned_children)
        namespace = dict(argparse=argparse, setting={},
            alloc_can_use_network_port=lambda **kw: [1,2,3,4], HttpServerManager=Mock(),
            mp=NS(Process=Process, Pipe=lambda **kw:(NS(recv=lambda:'init ok'),object())),
            start_router_process=object(), start_detokenization_process=object(),
            print_mem_stats=Mock(), uvicorn=NS(run=serve), app=object(), TIMEOUT_KEEP_ALIVE=5, sys=sys)
        exec(compile(ast.Module(body=[main], type_ignores=[]), '<actual native API main>', 'exec'), namespace)
        with patch.dict(sys.modules, {'slora.server.tc_measurement':module}), \
                patch.object(sys, 'argv', ['native', '--model_dir', '/model', '--lora-dirs','/owned/a']):
            try:
                namespace['main']()
            finally:
                self.assertTrue(children)
                self.assertTrue(all(not p.is_alive() for p in children))
        return children

    def test_actual_api_main_closes_both_handles_after_normal_return(self):
        self.assertEqual(len(self.execute_native_main()), 2)

    def test_actual_api_main_closes_children_on_start_and_serving_error(self):
        with self.assertRaisesRegex(RuntimeError, 'start failure'):
            self.execute_native_main(fail_start=True)
        with self.assertRaisesRegex(RuntimeError, 'serve failure'):
            self.execute_native_main(fail_serve=True)

    def test_real_owned_cpu_children_terminate_and_join(self):
        import multiprocessing
        ctx = multiprocessing.get_context('fork')
        children = [ctx.Process(target=time.sleep, args=(30,)) for _ in range(2)]
        try:
            for child in children:
                child.start()
            measurement.slora_join_owned_children(children, grace_s=1.)
            self.assertTrue(all(p.exitcode is not None and not p.is_alive() for p in children))
        finally:
            for child in children:
                if child.is_alive():
                    child.kill()
                child.join(timeout=2.)


class ReadyFixture:
    pass


class NativeReadyReceipt(unittest.TestCase):
    def setup_ready(self, root):
        cfg, _ = native_configs()
        paths = {f'/owned/a{i:03}': i for i in range(500)}
        cfg.update(output=str(root), deployment_notice_s=time.perf_counter()-1.,
            native_environment={'synthetic': 'no native process'},
            adapters={Path(p).name:[i,p] for p,i in paths.items()},
            native_input_params={k:cfg['native_args'][k] for k in service.SLORA_INPUT_KEYS},
            source_view_manifest_sha256='a'*64)
        path = root/'config.json'
        service._exclusive_json(path, cfg)
        (root/'worker_receipts').mkdir()
        server = ReadyFixture()
        server.__dict__.update(world_size=1, tp_rank=0, adapter_id=paths|{None:500},
            adapters=[NS(lora_dir=p) for p in paths]+[None], input_params=NS(**cfg['native_input_params']),
            model=object(), infer_adapter=object())
        guard = NS(gpu_process_identity=lambda pid:dict(pid=pid,start_ticks=5,uid=os.getuid(),
            cgroup='/synthetic-service',affinity=[2]))
        torch = NS(cuda=NS(device_count=lambda:1,current_device=lambda:0), __version__='synthetic-no-CUDA')
        return cfg,path,server,guard,torch

    def test_receipt_binds_actual_registry_process_and_after_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,path,server,guard,torch = self.setup_ready(Path(tmp))
            with patch.dict(os.environ, FAASLORA_TC_SLORA_WORKER_CONFIG=str(path),
                    FAASLORA_TC_SLORA_WORKER_INDEX='0', CUDA_VISIBLE_DEVICES='0'), \
                    patch.dict(sys.modules, torch=torch), \
                    patch.object(service, 'slora_worker_context', new=lambda *args:(cfg,guard)):
                row = measurement.slora_record_worker_ready(server,'/model',list(server.adapter_id)[:-1],14000,'HF',[])
                self.assertEqual(row['identity']['pid'],os.getpid())
                self.assertEqual(row['adapter_count'],500)
                self.assertEqual(row['ready_boundary'],'native_full_pool_and_model_initialized')
                self.assertGreaterEqual(row['at'],cfg['deployment_notice_s'])
                ready=json.loads((Path(tmp)/'worker_0.native_ready.json').read_text())
                receipt=Path(tmp)/'worker_receipts'/(ready['tc_worker_receipt_id']+'.json')
                self.assertEqual(service._sha(receipt),ready['tc_worker_receipt_sha256'])

    def test_wrong_registry_or_uninitialized_pool_publishes_no_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,path,server,guard,torch = self.setup_ready(Path(tmp))
            paths=list(server.adapter_id)[:-1]
            with patch.dict(os.environ, FAASLORA_TC_SLORA_WORKER_CONFIG=str(path),
                    FAASLORA_TC_SLORA_WORKER_INDEX='0'), \
                    patch.object(service, 'slora_worker_context', return_value=(cfg,guard)):
                for field,wrong in [('adapter_id',{}),('infer_adapter',None),('world_size',2)]:
                    with patch.object(server,field,wrong),self.subTest(field=field),self.assertRaises(ValueError):
                        measurement.slora_record_worker_ready(server,'/model',paths,14000,'HF',[])
                self.assertEqual(list((Path(tmp)/'worker_receipts').iterdir()),[])


class StartupContracts(unittest.IsolatedAsyncioTestCase):
    def test_explicit_lifecycle_not_vllm_or_store_alias(self):
        cfg = dict(schema='ieee_tc_slora_http_replay_v1', physical_lifecycle='slora_resident_pool_v1',
            ingress_mode='service_pre_ready_v1', artifact_preparation='native_full_pool_before_service_v1')
        shared.validate_http_lifecycle_contract(cfg)
        for field, bad in [('physical_lifecycle', 'native_store_pool_v1'),
                           ('physical_lifecycle', 'vllm_resident_pool_v1'),
                           ('ingress_mode', 'native_direct_v1'), ('artifact_preparation', 'on_demand')]:
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                shared.validate_http_lifecycle_contract(dict(cfg, **{field: bad}))

    async def test_all_500_static_ids_prepared_serially_after_notice(self):
        notice = time.perf_counter()
        paths = {f'a{i:03}': f'/owned/a{i:03}' for i in reversed(range(500))}
        calls, active = [], 0
        async def resolve(aid, path, rid):
            nonlocal active
            active += 1
            self.assertEqual(active, 1)
            self.assertEqual(path, paths[aid])
            self.assertEqual(rid, 'slora-full-pool:'+aid)
            await asyncio.sleep(0)
            calls.append(aid)
            active -= 1
            return preparation(aid)
        result = await shared.prepare_slora_full_pool(NS(resolve=resolve), paths, notice)
        self.assertEqual(calls, sorted(paths))
        self.assertEqual(set(result), set(paths))

    async def test_no_partial_pool_reused_or_pre_notice_preparation(self):
        notice = time.perf_counter()
        resolver = NS(resolve=AsyncMock())
        with self.assertRaises(ValueError):
            await shared.prepare_slora_full_pool(resolver, {'a': '/owned/a'}, notice)
        resolver.resolve.assert_not_called()
        paths = {f'a{i:03}': f'/owned/a{i:03}' for i in range(500)}
        for wrong in (dict(tc_artifact_cache_hit=True), dict(tc_artifact_resolve_started_s=notice-1)):
            resolver.resolve = AsyncMock(return_value=dict(preparation('a000'), **wrong))
            with self.assertRaises(ValueError):
                await shared.prepare_slora_full_pool(resolver, paths, notice)
            self.assertEqual(resolver.resolve.call_count, 1)

    async def test_same_round_robin_blocks_selected_worker_not_every_worker(self):
        clock = local_monotonic_clock_id()
        router = service.ResidentRoundRobin(['r0', 'r1'], 2, clock, instance_prefix='slora')
        def body(rid):
            return dict(request_id=rid, _sllm_internal_metrics=dict(tc_clock_id=clock))
        waiting = asyncio.create_task(router.route(body('r0'), time.perf_counter()))
        await asyncio.sleep(0)
        router.ready[1].set()
        index, second = await router.route(body('r1'), time.perf_counter())
        self.assertEqual((index, second['_sllm_internal_metrics']['instance_id']), (1, 'slora-1'))
        self.assertFalse(waiting.done())
        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting
        router.ready[0].set()
        with self.assertRaises(ValueError):
            await router.route(body('r0'), time.perf_counter())


class NativeHTTPBridge(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import aiohttp
        from aiohttp import web
        self.web = web
        self.clock = local_monotonic_clock_id()
        self.notice = time.perf_counter()
        self.prep = preparation()
        self.process = measurement._slora_process_identity(os.getpid())
        self.receipt = dict(tc_worker_receipt_id='e'*32, tc_worker_receipt_sha256='f'*64,
                            tc_worker_ready_s=time.perf_counter())
        self.corrupt, self.truncate, self.failure, self.calls = None, False, False, []
        self.gate = None
        app = web.Application()
        app.router.add_post('/generate_stream', self.native)
        async def health(request):
            return web.json_response('OK')
        app.router.add_get('/health', health)
        self.runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0, force_close=True),
            timeout=aiohttp.ClientTimeout(total=None), trust_env=False)
        self.handler = service.NativeSloraRequestHandler(session=self.session,
            endpoint=f'http://127.0.0.1:{port}/generate_stream', adapters={'a': (0, '/owned/a')},
            preparation={'a': self.prep}, clock_id=self.clock, instance_id='slora-0',
            receipt=self.receipt, worker_process=self.process, model='base', deployment_notice_s=self.notice)

    async def test_common_service_assembly_prepares_after_ingress_and_owns_shutdown(self):
        """Run the real outer service with CPU transport/synthetic GPU ownership.

        This checks assembly/order, not NVML, cgroup or native model qualification.
        """
        import socket
        import faaslora.metrics.metrics_collector as physical
        sockets=[]
        for _ in range(3):
            sock=socket.socket();sock.bind(('127.0.0.1',0));sockets.append(sock)
        ingress_port,router_port,nccl_port=[s.getsockname()[1] for s in sockets]
        for sock in sockets:
            sock.close()
        order=[]
        prepared_started,release_preparation=asyncio.Event(),asyncio.Event()
        native_pid=self.process['pid']
        identities={native_pid:dict(pid=native_pid,start_ticks=self.process['start_ticks'],
            uid=os.getuid(),cgroup='/sys/fs/cgroup/synthetic',affinity=[2]),
            987654:dict(pid=987654,start_ticks=1,uid=os.getuid(),cgroup='/sys/fs/cgroup/synthetic',affinity=[2])}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'output'
            cfg,http=native_configs()
            cfg.update(gpu_ids=[0],worker_ports=[int(self.handler.endpoint.split(':')[2].split('/')[0])],
                router_port=router_port,nccl_ports=[nccl_port],source_view=str(root/'view'))
            http.update(main_repo=str(MAIN),model='base',url=f'http://127.0.0.1:{ingress_port}/v1/chat/completions')
            shared.prepare_slora_measurement_view(ROOT/'repos/S-LoRA',root/'view',MAIN,
                host_layout='packed_views_v1',worker_lifecycle=True)
            cfg['source_view_manifest_sha256']=service._sha(root/'view/measurement_manifest.json')
            config_path,http_path=root/'config.json',root/'http.json'
            service._exclusive_json(config_path,cfg);service._exclusive_json(http_path,http)
            trace=root/'trace.json'
            trace.write_text(json.dumps(dict(requests=[dict(request_id='r',arrival_time_s=0.,
                adapter_id='a000',expected_output_tokens=3)])))
            plan=FrozenReplayPlan.load(trace)
            replay=root/'replay.jsonl';replay.touch()
            context=dict(transport='http',config_sha256=service._sha(http_path),result_path=str(replay),
                deployment_notice_s=self.notice,replay_t0_s=self.notice+60.,clock_id=self.clock,plan=plan.identity())
            launch_receipt=root/'launch.json';service._exclusive_json(launch_receipt,dict(external_replay=context))
            paths={f'a{i:03}':f'/owned/a{i:03}' for i in range(500)}
            async def prepare(resolver,received_paths,notice):
                self.assertEqual(received_paths,paths);self.assertEqual(notice,self.notice)
                order.append('prepare_started');prepared_started.set()
                await release_preparation.wait()
                order.append('prepare_finished')
                return {aid:preparation(aid) for aid in paths}
            class Census:
                def __init__(self,api):pass
                def sample(self,path):
                    return dict(devices=[dict(gpu_uuid='GPU-synthetic',processes=[dict(
                        pid=native_pid,service_member=True,kind='compute')])])
                def close(self):pass
            class Owner:
                def __init__(self,**kw):
                    order.append('allocate');self.census=kw['census'];self.gpu_uuids=['GPU-synthetic']
                    self.events=[];self.process=None
                def bind_process(self,p,identity):self.process=p
                def confirm_workers(self,workers):
                    order.append('confirm_native_worker');self.events.append(dict(event='native_workers'))
                async def wait_workers(self,timeout_s):
                    assert self.process.poll() is not None;order.append('native_exited')
                def release(self):
                    assert self.process.poll() is not None;order.append('release')
                def evidence(self):return dict(synthetic_cpu_only=True)
            class Deployment:
                def __init__(self,root,**kw):
                    self.root=root;root.mkdir();self.allocations=root/'allocations';self.allocations.mkdir()
                def bind_result_file(self,path):pass
                def summarize(self,**kw):return dict(measurement_complete=True,n_native_contract_complete=1)
                def finalize(self):order.append('finalize')
            class Process:
                pid=987654;returncode=None
                def __init__(proc,argv,*,env,**kw):
                    order.append('spawn')
                    worker_path=Path(env['FAASLORA_TC_SLORA_WORKER_CONFIG'])
                    worker=json.loads(worker_path.read_text())
                    assert argv[3]=='native-slora-worker'
                    rpc=root/'view/slora/server/router/model_infer/model_rpc.py'
                    receipt=dict(schema='ieee_tc_slora_backend_ready_v1',receipt_id='e'*32,index=0,
                        at=time.perf_counter(),clock_id=self.clock,identity=identities[native_pid],
                        api_parent_identity=identities[proc.pid],worker_config_sha256=service._sha(worker_path),
                        adapter_ids={p:i for i,p in worker['adapters'].values()},adapter_count=500,
                        source_view_manifest_sha256=cfg['source_view_manifest_sha256'],
                        native_input_params=worker['native_input_params'],native_environment=worker['native_environment'],
                        host_layout=cfg['host_layout'],cuda_visible='0',cuda_device_count=1,tp=1,model_dir='/model',
                        runner_sha256=service._sha(service.__file__),rpc_path=str(rpc),rpc_sha256=service._sha(rpc),
                        ready_boundary='native_full_pool_and_model_initialized')
                    path=output/'worker_receipts'/('e'*32+'.json');service._exclusive_json(path,receipt)
                    service._exclusive_json(output/'worker_0.native_ready.json',dict(
                        tc_worker_receipt_id='e'*32,tc_worker_receipt_sha256=service._sha(path)))
                def poll(self):return self.returncode
                def terminate(self):order.append('terminate_api');self.returncode=0
                def wait(self,timeout):return self.returncode
            guard=NS(verify_current_service=lambda:dict(service_identity=dict(path='/sys/fs/cgroup/synthetic')),
                gpu_process_identity=lambda pid:identities[pid],load_nvml_binding=lambda *a:None,NativeGPUCensus=Census)
            env=dict(FAASLORA_TC_LAUNCH_RECEIPT=str(launch_receipt),FAASLORA_TC_NVML_BINDING='/synthetic',
                     FAASLORA_TC_NVML_SHA256='f'*64)
            loop=asyncio.get_running_loop()
            with patch.dict(os.environ,env),patch.object(shared,'load_guard',return_value=guard), \
                    patch.object(shared,'load_http_replay_plan',return_value=plan), \
                    patch.object(shared,'prepare_remote_artifacts',return_value=({},paths)), \
                    patch.object(shared,'prepare_slora_full_pool',side_effect=prepare), \
                    patch.object(shared,'record_http_physical_terminals'), \
                    patch.object(measurement,'PublishedArtifactResolver',return_value=object()), \
                    patch.object(service,'slora_worker_environment',return_value={k:'synthetic' for k in service.SLORA_ENV_KEYS}), \
                    patch.object(service.subprocess,'Popen',Process),patch.object(loop,'add_signal_handler'), \
                    patch.object(physical,'PhysicalGPUDeployment',Deployment),patch.object(physical,'PhysicalGPUAllocation',Owner):
                running=asyncio.create_task(service._resident_http_service(config_path,http_path,output,root/'private'))
                try:
                    await asyncio.wait_for(prepared_started.wait(),5.)
                    self.assertEqual(order,['prepare_started'])
                    body=self.inputs()[0];body['lora_adapter_name']='a000'
                    async def request():
                        async with self.session.post(http['url'],json=body) as response:
                            self.assertEqual(response.status,200)
                            return await response.json()
                    offered=asyncio.create_task(request())
                    # Wait for real pre-ready receipt, not a sleep-based ordering assumption.
                    for _ in range(500):
                        if 'ingress_received' in (output/'ingress.jsonl').read_text():break
                        await asyncio.sleep(.001)
                    self.assertIn('ingress_received',(output/'ingress.jsonl').read_text())
                    self.assertNotIn('allocate',order);self.assertFalse(offered.done())
                    release_preparation.set()
                    response=await asyncio.wait_for(offered,5.)
                    with replay.open('a') as handle:
                        handle.write(json.dumps(dict(event='http_raw_response',request_id='r',body=response,status=200))+'\n')
                        handle.write(json.dumps(dict(event='http_replay_complete',N_failed=0,N_response=1))+'\n')
                    result=await asyncio.wait_for(running,5.)
                    self.assertTrue(result['passed'])
                    self.assertEqual(result['physical_lifecycle']['mode'],'slora_resident_pool_v1')
                    self.assertEqual(order,['prepare_started','prepare_finished','allocate','spawn',
                        'confirm_native_worker','terminate_api','native_exited','release','finalize'])
                finally:
                    release_preparation.set()
                    if not running.done():running.cancel()
                    await asyncio.gather(running,return_exceptions=True)

    async def asyncTearDown(self):
        await self.session.close()
        await self.runner.cleanup()

    async def native(self, request):
        body = await request.json()
        self.calls.append(body)
        if self.failure:
            return self.web.Response(status=500)
        if self.gate is not None:
            await self.gate.wait()
        now = time.perf_counter()
        target = body['parameters']['max_new_tokens']
        first = dict(native_prompt_token_ids=[1, 2], native_prompt_token_ids_sha256=digest([1, 2]),
            target_tokens=target, registered_monotonic_s=now, scheduled_monotonic_s=now,
            engine_class='slora.models.peft.lora_unordered_batch_infer.LoraUnorderedBatchInfer',
            process=dict(self.process))
        payload = b''
        for i in range(target):
            event = dict(schema='ieee_tc_slora_token_event_v1', request_id=body['req_id'],
                adapter_path=body['lora_dir'], adapter_index=0, output_index=i+1,
                native_token_id=100+i, produced_monotonic_s=time.perf_counter())
            if i == 0:
                event['first_event'] = first
            row = dict(token=dict(id=100+i, text='文字'), finished=i == target-1,
                       tc_slora_native_token=event)
            if self.corrupt:
                self.corrupt(row)
            payload += ('data:'+json.dumps(row, ensure_ascii=False)+'\n\n').encode()
        response = self.web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        if self.truncate:
            payload = payload[:-1]
        # Deliberately fragment UTF-8 and frame boundaries; receive chunks do not
        # define native token production times. EOF is deliberately later.
        for start in range(0, len(payload), 7):
            await response.write(payload[start:start+7])
        await asyncio.sleep(.01)
        await response.write_eof()
        return response

    def inputs(self, target=3):
        now = time.perf_counter()
        body = dict(model='base', request_id='r', canonical_prompt='canonical prompt', input_tokens=[1, 2],
            lora_adapter_name='a', max_tokens=target, temperature=0, top_p=1, ignore_eos=True,
            stop=[], stop_token_ids=[], stream=False,
            _sllm_internal_metrics=dict(tc_clock_id=self.clock, instance_id='slora-0',
                tc_ingress_received_s=now, tc_ingress_forwarded_s=now, tc_http_received_s=now,
                tc_router_enqueued_s=now, tc_instance_assigned_s=now, ready_instances_at_enqueue=1))
        prepared = dict(request_id='r', native_backend='slora', adapter_id='a', input_token_ids=[1, 2],
            target_tokens=target, native_prompt_token_ids_sha256=digest([1, 2]),
            canonical_prompt_sha256=hashlib.sha256(body['canonical_prompt'].encode()).hexdigest(),
            artifact_content_sha256='a'*64, artifact_manifest_sha256='b'*64,
            worker_lifetime_required=True, deployment_notice_s=self.notice)
        event = dict(planned_arrival_s=now, task_created_s=now, client_submit_s=now)
        return body, prepared, event

    async def valid(self, target=3):
        body, prepared, event = self.inputs(target)
        result = await self.handler.generate(body)
        completed = time.perf_counter()
        audit = shared.validate_http_observation(prepared, result, event, completed, self.clock,
                                                  ingress_required=True)
        return prepared, event, result, completed, audit

    async def test_real_loopback_native_ids_timing_prompt_and_preparation_reference(self):
        prepared, event, result, completed, audit = await self.valid()
        native = result['metrics']['ieee_tc']
        sent = self.calls[0]
        self.assertEqual(sent['inputs'], 'canonical prompt')
        self.assertEqual(sent['lora_dir'], '/owned/a')
        self.assertFalse(sent['parameters']['do_sample'])
        self.assertTrue(sent['parameters']['ignore_eos'])
        self.assertEqual(sent['parameters']['stop_sequences'], [])
        self.assertEqual(native['completion_token_ids'], [100, 101, 102])
        self.assertNotIn('native_queued_monotonic_s', native)
        self.assertNotIn('native_lora_name', native)
        self.assertEqual(native['native_adapter_index'], 0)
        self.assertGreater(native['worker_completion_notification_ms'], 0)
        self.assertEqual(audit['artifact_transferred_bytes'], 0)
        self.assertEqual(audit['preparation_transferred_bytes'], 1024)
        self.assertFalse(audit['lora_numerical_correctness_qualified'])
        self.assertAlmostEqual(audit['e2e_ms'], sum(audit[k] for k in (
            'submit_lag_ms', 'dispatch_wait_after_submit_ms', 'service_ttft_ms',
            'decode_ms', 'completion_notification_ms')))
        self.assertEqual(audit['tpot_ms'], native['native_tpot_ms'])

    async def test_single_token_has_no_tpot(self):
        *_, audit = await self.valid(1)
        self.assertIsNone(audit['tpot_ms'])

    async def test_truncated_sse_no_synthetic_completion(self):
        self.truncate = True
        with self.assertRaisesRegex(ValueError, 'incomplete native SSE'):
            await self.handler.generate(self.inputs()[0])
        self.assertEqual(len(self.calls), 1)

    async def test_error_is_visible_not_retried(self):
        self.failure = True
        with self.assertRaises(ValueError):
            await self.handler.generate(self.inputs()[0])
        self.assertEqual(len(self.calls), 1)

    async def test_changed_native_worker_prompt_or_adapter_rejected(self):
        def foreign_worker(row):
            first = row['tc_slora_native_token'].get('first_event')
            if first:
                first['process']['pid'] += 1
        def wrong_prompt(row):
            first = row['tc_slora_native_token'].get('first_event')
            if first:
                first['native_prompt_token_ids'] = [4]
        def wrong_index(row):
            row['tc_slora_native_token']['adapter_index'] = 1
        for change in (foreign_worker, wrong_prompt, wrong_index):
            self.corrupt = change
            with self.subTest(change=change.__name__), self.assertRaises(ValueError):
                await self.handler.generate(self.inputs()[0])

    async def test_cancel_has_no_retry_and_closes_request(self):
        self.gate = asyncio.Event()
        task = asyncio.create_task(self.handler.generate(self.inputs()[0]))
        for _ in range(100):
            if self.calls:
                break
            await asyncio.sleep(.001)
        self.assertEqual(len(self.calls), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.gate.set()
        self.assertEqual(len(self.calls), 1)

    async def test_outer_validator_rejects_mislabelled_or_missing_evidence(self):
        prepared, event, result, completed, _ = await self.valid()
        bad_controls = dict(tc_deployment_notice_s=self.notice-1, tc_worker_ready_s=0,
            tc_artifact_request_id='r', tc_artifact_cache_hit=True,
            tc_artifact_preparation_mode='on_demand', tc_canonical_prompt_sha256='0'*64,
            tc_worker_native_process={}, tc_worker_receipt_sha256=None,
            tc_native_http_submitted_s=float('nan'))
        for field, value in bad_controls.items():
            bad = copy.deepcopy(result)
            bad['metrics']['ieee_tc']['control_observation'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                shared.validate_http_observation(prepared, bad, event, completed, self.clock)
        for field, value in [('timing_contract', 'ieee_tc_native_v1'),
                             ('native_adapter_index', True), ('native_timing_source', 'client_sse'),
                             ('native_rpc_registered_monotonic_s', None), ('native_adapter_path', '/wrong/b')]:
            bad = copy.deepcopy(result)
            bad['metrics']['ieee_tc'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                shared.validate_http_observation(prepared, bad, event, completed, self.clock)

    async def test_actual_bridge_response_enters_physical_union_with_native_identity(self):
        prepared, event, body, completed, _ = await self.valid()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root/'trace.json'
            trace.write_text(json.dumps(dict(requests=[dict(request_id='r', arrival_time_s=0.,
                adapter_id='a', expected_output_tokens=3)])))
            plan = FrozenReplayPlan.load(trace)
            context = dict(deployment_notice_s=self.notice, replay_t0_s=event['planned_arrival_s'],
                           clock_id=self.clock, plan=plan.identity())
            deployment = PhysicalGPUDeployment(root=root/'physical', plan=plan, context=context)
            lease = dict(lease_id='pool', owner_id='pool', gpu_uuids=['GPU-0'], clock_id=self.clock)
            # Explicit synthetic physical events: not proof of a real GPU lease.
            events = [('acquire', self.notice), ('worker_spawn', self.notice),
                ('native_workers', self.notice), ('native_workers_exited', completed), ('release', completed)]
            (deployment.allocations/'pool.jsonl').write_text(''.join(
                json.dumps(dict(lease, event=e, at=t))+'\n' for e, t in events))
            rows = [dict(prepared, event='request_contract', source_item_sha256=plan.entries[0].source_sha256),
                dict(event='http_raw_response', request_id='r', status=200, body=body),
                dict(event='http_response', request_id='r', client_completed_s=completed,
                     response=dict(protocol_valid=True))]
            replay = root/'replay.jsonl'
            replay.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            shared.record_http_physical_terminals(deployment, replay)
            report = deployment.summarize(observed_until_s=completed+.1)
            self.assertTrue(report['measurement_complete'])
            self.assertEqual(report['n_native_contract_complete'], 1)
            self.assertIsNone(report['n_correct'])
            self.assertEqual(deployment.terminals['r']['completion_token_source'], 'slora_native_sse_token_id')


if __name__ == '__main__':
    unittest.main()
