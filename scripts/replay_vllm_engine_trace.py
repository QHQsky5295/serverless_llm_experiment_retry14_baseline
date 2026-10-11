#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import os
import sys
import time
import signal
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

LIVE_PRINT_INTERVAL_S = 2.0


def _exclusive_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    try:
        with temporary.open('x') as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic publication without replacing even a concurrently created file.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_worker_environment(cfg, device, environ=None):
    """Bind subprocess tools to the declared interpreter before acquiring GPUs.

    Calling a venv Python does not activate its bin directory for native JIT
    subprocesses. Do not resolve the Python symlink before deriving that bin.
    No installation, backend fallback or serving-policy change occurs here.
    """
    python = Path(cfg['python'])
    jobs = cfg.get('build_max_jobs')
    if (not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK)
            or type(jobs) is not int or not 1 <= jobs <= 2):
        raise ValueError('absolute executable interpreter and bounded build_max_jobs required')
    env = dict(os.environ if environ is None else environ)
    env.update(PATH=str(python.parent)+os.pathsep+env.get('PATH', ''),
        MAX_JOBS=str(jobs), CUDA_VISIBLE_DEVICES=str(device),
        VLLM_WORKER_MULTIPROC_METHOD='spawn', PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    expected = python.parent/'ninja'
    ninja = shutil.which('ninja', path=env['PATH'])
    if not expected.is_file() or not ninja or Path(ninja).resolve() != expected.resolve():
        raise ValueError('declared environment has no executable ninja; no system fallback')
    version = subprocess.run([ninja, '--version'], env=env, check=True, timeout=10,
        capture_output=True, text=True).stdout.strip()
    if not version:
        raise ValueError('native ninja version probe returned no identity')
    identity = dict(python=str(python), python_resolved=str(python.resolve()),
        ninja=str(expected), ninja_sha256=_sha(expected), ninja_version=version,
        build_max_jobs=jobs, path_sha256=hashlib.sha256(env['PATH'].encode()).hexdigest())
    return env, identity


def validate_native_service_config(cfg, http):
    """Fixed-version resident transport adapter, not a second scheduling policy."""
    from prepare_ieee_tc_serverless_stack import validate_http_lifecycle_contract
    validate_http_lifecycle_contract(http)
    if (cfg.get('schema') != 'ieee_tc_vllm_native_service_v1'
            or http.get('schema') != 'ieee_tc_vllm_http_replay_v1'
            or cfg.get('routing') != 'round_robin'
            or cfg.get('generation_contract') != 'fixed_length_greedy_v1'
            or cfg.get('version') != '0.30.0'
            or type(cfg.get('build_max_jobs')) is not int or not 1 <= cfg['build_max_jobs'] <= 2
            or not isinstance(cfg.get('gpu_ids'), list) or not cfg['gpu_ids']
            or any(type(i) is not int or i not in range(4) for i in cfg['gpu_ids'])
            or len(set(cfg['gpu_ids'])) != len(cfg['gpu_ids'])
            or len(cfg.get('worker_ports', [])) != len(cfg['gpu_ids'])):
        raise ValueError('explicit resident vLLM/round-robin/generation identity required')
    from urllib.parse import urlsplit
    endpoint = urlsplit(http['url'])
    ports = [endpoint.port, cfg.get('router_port'), *cfg['worker_ports']]
    if (endpoint.scheme != 'http' or endpoint.hostname != '127.0.0.1'
            or endpoint.path != '/v1/chat/completions' or endpoint.query or endpoint.fragment
            or endpoint.username or endpoint.password
            or any(type(p) is not int or not 0 < p < 65536 for p in ports)
            or len(set(ports)) != len(ports)):
        raise ValueError('distinct explicit loopback ingress/router/worker ports required')
    engine = cfg.get('engine_args', {})
    required = dict(model=http['backbone'], tokenizer=http['backbone'],
        tensor_parallel_size=1, dtype='float16', enable_lora=True,
        disable_log_stats=False, max_model_len=1024, enable_prefix_caching=False)
    if any(engine.get(k) != v for k, v in required.items()):
        raise ValueError('native engine identity or required timing configuration differs')
    if not http.get('remote_artifacts'):
        raise ValueError('resident main path requires real published artifacts; no local fallback')
    return endpoint.port


class NativeVLLMRequestHandler:
    """One native AsyncLLM: request-driven loading and the native LRU only.

    No preload, planner, admission ranking, adapter-affinity or load-balancing
    logic. Remote resolution, registration/loading and generation belong to the
    same guarded service. Dependencies are injectable for CPU contract tests.
    """

    def __init__(self, *, engine, resolver, adapters, lora_factory, sampling_factory,
                 observation_factory, clock_id, instance_id, receipt, model):
        self.engine, self.resolver, self.adapters = engine, resolver, adapters
        self.lora_factory, self.sampling_factory = lora_factory, sampling_factory
        self.observation_factory, self.clock_id = observation_factory, clock_id
        self.instance_id, self.receipt, self.model = instance_id, receipt, model

    async def generate(self, body):
        entry = time.perf_counter()
        ids, target, aid = body.get('input_tokens'), body.get('max_tokens'), body.get('lora_adapter_name')
        if (body.get('model') != self.model or not isinstance(body.get('request_id'), str)
                or not body['request_id'] or type(ids) is not list or not ids
                or any(type(t) is not int or t < 0 for t in ids)
                or type(target) is not int or not 0 < target <= 256 or len(ids)+target > 1024
                or body.get('temperature') != 0 or body.get('top_p') != 1
                or body.get('ignore_eos') is not True or body.get('stream') is not False
                or body.get('stop') != [] or body.get('stop_token_ids') != []
                or aid not in self.adapters):
            raise ValueError('request differs from native fixed-output/adapter contract')
        control = dict(body.get('_sllm_internal_metrics', {}))
        if control.get('tc_clock_id') != self.clock_id or control.get('instance_id') != self.instance_id:
            raise ValueError('native worker routing/clock identity differs')
        control.update(tc_backend_entry_s=entry, **self.receipt)
        int_id, path = self.adapters[aid]
        control.update(await self.resolver.resolve(aid, path, body['request_id']))
        lora = self.lora_factory(lora_name=aid, lora_int_id=int_id, lora_path=path)
        # CUMULATIVE is explicit in the concrete factory; do not infer tokens
        # from text or accept DELTA/FINAL_ONLY in the native timing observer.
        params = self.sampling_factory(temperature=0., top_p=1., max_tokens=target,
            ignore_eos=True, stop=[], stop_token_ids=[])
        observation = self.observation_factory(lora)
        generator = self.engine.generate({'prompt_token_ids': ids}, params,
            body['request_id'], lora_request=lora)
        final = None
        try:
            async for output in generator:
                observation.observe(output)
                final = output
        finally:
            # AsyncLLM owns abort on cancellation/GeneratorExit. Explicitly
            # close a suspended generator if observation validation raised.
            await generator.aclose()
        measured = observation.finish(control)
        if (final is None or not final.finished or measured['native_output_tokens'] != target
                or measured['native_prompt_token_ids'] != ids):
            raise ValueError('native engine did not finish the exact requested work')
        return dict(id=body['request_id'], model=self.model,
            choices=[dict(index=0, finish_reason='length',
                          message=dict(role='assistant', content=final.outputs[0].text))],
            usage=dict(prompt_tokens=len(ids), completion_tokens=target, total_tokens=len(ids)+target),
            metrics=dict(ieee_tc=measured))


class ResidentRoundRobin:
    """Stable request-ordinal round robin, as in the existing fair runner.

    Readiness only blocks the already-selected worker. It never changes the
    target, looks at adapter tier/queue lengths or waits for all workers.
    """

    def __init__(self, request_ids, worker_count, clock_id, *, instance_prefix='vllm'):
        if (worker_count < 1 or len(set(request_ids)) != len(request_ids)
                or instance_prefix not in ('vllm', 'slora')):
            raise ValueError('unique offered request IDs and workers required')
        self.ordinals = {rid: i for i, rid in enumerate(request_ids)}
        self.ready = [asyncio.Event() for _ in range(worker_count)]
        self.seen = set()
        self.clock_id = clock_id
        self.instance_prefix = instance_prefix

    async def route(self, body, received):
        rid = body.get('request_id')
        if rid not in self.ordinals or rid in self.seen:
            raise ValueError('unknown/duplicate offered request; no hidden retries')
        self.seen.add(rid)
        control = dict(body.get('_sllm_internal_metrics', {}))
        if control.get('tc_clock_id') != self.clock_id:
            raise ValueError('router clock differs from startup ingress')
        ordinal = self.ordinals[rid]
        index = ordinal % len(self.ready)
        control.update(tc_http_received_s=received, tc_router_enqueued_s=time.perf_counter(),
            ready_instances_at_enqueue=sum(r.is_set() for r in self.ready),
            tc_route_ordinal=ordinal)
        await self.ready[index].wait()
        control.update(tc_instance_assigned_s=time.perf_counter(),
                       instance_id=f'{self.instance_prefix}-{index}')
        return index, dict(body, _sllm_internal_metrics=control)


class NativeSloraRequestHandler:
    """Shared HTTP contract -> native generate_stream, not a new scheduler.

    All-pool preparation is a startup action after deployment notice, not an
    artificial request-time fetch. Native token events and the real ready
    worker identity must agree. HTTP receive/EOF are diagnostics, not TTFT/TPOT.
    The owning service supplies its already-verified preparation/ready records.
    """
    def __init__(self, *, session, endpoint, adapters, preparation, clock_id,
                 instance_id, receipt, worker_process, model, deployment_notice_s):
        from urllib.parse import urlsplit
        url = urlsplit(endpoint)
        if (url.scheme != 'http' or url.hostname != '127.0.0.1'
                or not url.port or url.path != '/generate_stream'
                or url.username or url.password or url.query or url.fragment
                or not instance_id.startswith('slora-')
                or set(adapters) != set(preparation)
                or worker_process.get('clock_id') != clock_id
                or any(type(worker_process.get(k)) is not int or worker_process[k] <= 0
                       for k in ('pid', 'start_ticks'))
                or any(not isinstance(receipt.get(k), str) or len(receipt[k]) != n
                       or set(receipt[k]) - set('0123456789abcdef') for k, n in (
                           ('tc_worker_receipt_id', 32), ('tc_worker_receipt_sha256', 64)))
                or type(receipt.get('tc_worker_ready_s')) not in (float, int)
                or not math.isfinite(receipt['tc_worker_ready_s'])):
            raise ValueError('explicit native S-LoRA endpoint/preparation/worker identity required')
        self.session, self.endpoint, self.adapters = session, endpoint, adapters
        self.preparation, self.clock_id, self.instance_id = preparation, clock_id, instance_id
        self.receipt, self.worker_process = dict(receipt), dict(worker_process)
        self.model, self.notice = model, deployment_notice_s

    async def generate(self, body):
        from ieee_tc_serverless_measurement import SloraNativeRequestObservation
        from native_stream_timing import NativeStreamTiming
        entry = time.perf_counter()
        ids, target, aid = body.get('input_tokens'), body.get('max_tokens'), body.get('lora_adapter_name')
        prompt, rid = body.get('canonical_prompt'), body.get('request_id')
        if (body.get('model') != self.model or not isinstance(rid, str) or not rid
                or not isinstance(prompt, str) or not prompt
                or type(ids) is not list or not ids or any(type(t) is not int or t < 0 for t in ids)
                or type(target) is not int or not 1 <= target <= 256 or len(ids)+target > 1024
                or body.get('temperature') != 0 or body.get('top_p') != 1
                or body.get('ignore_eos') is not True or body.get('stream') is not False
                or body.get('stop') != [] or body.get('stop_token_ids') != []
                or aid not in self.adapters):
            raise ValueError('request differs from S-LoRA fixed-output/prompt/adapter contract')
        control = dict(body.get('_sllm_internal_metrics', {}))
        if control.get('tc_clock_id') != self.clock_id or control.get('instance_id') != self.instance_id:
            raise ValueError('S-LoRA routing/clock identity differs')
        index, path = self.adapters[aid]
        prep = self.preparation[aid]
        start, end = prep.get('tc_artifact_resolve_started_s'), prep.get('tc_artifact_resolve_completed_s')
        if (prep.get('tc_artifact_id') != aid or prep.get('tc_artifact_cache_hit') is not False
                or any(type(t) not in (int, float) or not math.isfinite(t)
                       for t in (self.notice, start, end))
                or not self.notice <= start <= end <= self.receipt['tc_worker_ready_s'] <= entry):
            raise ValueError('native full-pool preparation must occur after notice and before serving')
        control.update(prep, **self.receipt, tc_backend_entry_s=entry,
            tc_artifact_preparation_mode='native_full_pool_before_service_v1',
            tc_deployment_notice_s=self.notice,
            tc_worker_native_process=dict(self.worker_process),
            tc_canonical_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        observation = SloraNativeRequestObservation(request_id=rid, adapter_path=path,
            prompt_ids=ids, target=target, clock_id=self.clock_id)
        frames = NativeStreamTiming(event_callback=observation.observe)
        native_body = dict(req_id=rid, inputs=prompt, lora_dir=path,
            parameters=dict(do_sample=False, temperature=0., top_p=1., top_k=1,
                presence_penalty=0., frequency_penalty=0., ignore_eos=True,
                max_new_tokens=target, stop_sequences=[]))
        control['tc_native_http_submitted_s'] = time.perf_counter()
        async with self.session.post(self.endpoint, json=native_body, allow_redirects=False) as response:
            if response.status != 200 or response.content_type != 'text/event-stream':
                raise ValueError('native S-LoRA did not return its SSE contract')
            async for chunk in response.content.iter_any():
                frames.feed(chunk, time.perf_counter())
            frames.finish()
        completed = time.perf_counter()
        measured = observation.finish(completed)
        if (measured['native_process'] != self.worker_process
                or measured['native_adapter_index'] != index
                or frames.token_count != target or frames.token_events != target):
            raise ValueError('native S-LoRA output differs from ready worker/adapter binding')
        measured['control_observation'] = control
        measured['client_sse_diagnostics'] = frames.metrics(completed)
        return dict(id=rid, model=self.model,
            choices=[dict(index=0, finish_reason='length')],
            usage=dict(prompt_tokens=len(ids), completion_tokens=target, total_tokens=len(ids)+target),
            metrics=dict(ieee_tc=measured))


async def _native_worker(config_path, index):
    # No backend import until this exact process has passed containment.
    from prepare_ieee_tc_serverless_stack import load_guard
    cfg = _load_json(config_path)
    expected_sources = {str(SCRIPT_DIR/name) for name in
        ('replay_vllm_engine_trace.py', 'ieee_tc_serverless_measurement.py',
         'prepare_ieee_tc_serverless_stack.py')}
    if (type(index) is not int or not 0 <= index < len(cfg['gpu_ids'])
            or set(cfg.get('source_sha256', {})) != expected_sources
            or any(_sha(path) != digest for path, digest in cfg['source_sha256'].items())):
        raise ValueError('native worker index or frozen adapter source differs')
    sys.path.insert(0, cfg['main_repo'])
    guard = load_guard(Path(cfg['main_repo']))
    guard.verify_current_service()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(cfg['gpu_ids'][index]):
        raise ValueError('worker visible device differs from physical allocation')
    # Read back the actual tool inherited by this worker, not only the parent
    # launch intention. Avoid prepending PATH twice in this check.
    toolchain = cfg['native_toolchain']
    if (hashlib.sha256(os.environ.get('PATH', '').encode()).hexdigest() != toolchain['path_sha256']
            or os.environ.get('MAX_JOBS') != str(toolchain['build_max_jobs'])
            or shutil.which('ninja') != toolchain['ninja']
            or _sha(toolchain['ninja']) != toolchain['ninja_sha256']):
        raise ValueError('native worker toolchain differs from launch qualification')
    from ieee_tc_serverless_measurement import (PublishedArtifactResolver,
        install_v030_snapshot, V030NativeRequestObservation)
    from faaslora.clock import local_monotonic_clock_id
    snapshot = install_v030_snapshot()
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import SamplingParams, RequestOutputKind
    from aiohttp import web
    engine = None
    runner = None
    stopped = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stopped.set)
    try:
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**cfg['engine_args']))
        receipt_id = uuid.uuid4().hex
        row = dict(schema='ieee_tc_vllm_backend_ready_v1', receipt_id=receipt_id,
            at=time.perf_counter(), clock_id=local_monotonic_clock_id(),
            identity=guard.gpu_process_identity(os.getpid()),
            cuda_visible=os.environ['CUDA_VISIBLE_DEVICES'], index=index,
            snapshot=snapshot, engine_args=cfg['engine_args'],
            native_toolchain=toolchain,
            native_api_sha256={str(Path(inspect.getfile(cls)).resolve()):
                               _sha(inspect.getfile(cls)) for cls in (AsyncLLM, AsyncEngineArgs)},
            runner_sha256=_sha(__file__), worker_config_sha256=_sha(config_path))
        receipt_path = Path(cfg['output'])/'worker_receipts'/(receipt_id+'.json')
        _exclusive_json(receipt_path, row)
        receipt = dict(tc_worker_receipt_id=receipt_id, tc_worker_receipt_sha256=_sha(receipt_path))
        handler = NativeVLLMRequestHandler(engine=engine,
            resolver=PublishedArtifactResolver(cfg['remote_config']), adapters=cfg['adapters'],
            lora_factory=LoRARequest,
            sampling_factory=lambda **kw: SamplingParams(**kw, output_kind=RequestOutputKind.CUMULATIVE),
            observation_factory=V030NativeRequestObservation, clock_id=row['clock_id'],
            instance_id=f'vllm-{index}', receipt=receipt, model=cfg['model'])

        async def generate(request):
            return web.json_response(await handler.generate(await request.json()))

        app = web.Application()
        app.router.add_post('/v1/chat/completions', generate)
        runner = web.AppRunner(app, access_log=None, handler_cancellation=True, shutdown_timeout=1.)
        await runner.setup()
        await web.TCPSite(runner, '127.0.0.1', cfg['worker_ports'][index]).start()
        _exclusive_json(Path(cfg['output'])/f'worker_{index}.ready.json',
                        dict(**receipt, port=cfg['worker_ports'][index]))
        await stopped.wait()
    finally:
        if runner is not None:
            await runner.cleanup()
        if engine is not None:
            engine.shutdown(timeout=60.)


async def _resident_http_service(config_path, http_path, output, private_root):
    from prepare_ieee_tc_serverless_stack import (load_guard, load_http_replay_plan,
        prepare_remote_artifacts, record_http_physical_terminals)
    from ieee_tc_serverless_measurement import PreReadyHTTPIngress
    import aiohttp
    from aiohttp import web
    cfg, http = _load_json(config_path), _load_json(http_path)
    ingress_port = validate_native_service_config(cfg, http)
    sys.path.insert(0, http['main_repo'])
    guard = load_guard(Path(http['main_repo']))
    admission = guard.verify_current_service()
    context = _load_json(Path(os.environ['FAASLORA_TC_LAUNCH_RECEIPT']))['external_replay']
    if (context.get('transport') != 'http' or context.get('config_sha256') != _sha(http_path)
            or Path(sys.executable).resolve() != Path(cfg['python']).resolve()):
        raise ValueError('service interpreter/publisher contract differs')
    from faaslora.metrics.metrics_collector import PhysicalGPUAllocation, PhysicalGPUDeployment
    from faaslora.clock import local_monotonic_clock_id
    plan = load_http_replay_plan(http)
    # All subprocess prerequisites are qualified before the first physical
    # allocation. The executable is already installed; this does not compile.
    worker_environments = [native_worker_environment(cfg, device) for device in cfg['gpu_ids']]
    toolchain = worker_environments[0][1]
    if any(identity != toolchain for _, identity in worker_environments):
        raise ValueError('resident workers have inconsistent native toolchains')
    output.mkdir(mode=0o700, exist_ok=False)
    private_root.mkdir(mode=0o700, exist_ok=False)
    (output/'worker_receipts').mkdir()
    _exclusive_json(output/'native_toolchain.json', toolchain)
    deployment = PhysicalGPUDeployment(root=output/'physical', plan=plan, context=context)
    deployment.bind_result_file(output/'model_qualification.json')
    service_path = Path(admission['service_identity']['path'])
    remote, paths = prepare_remote_artifacts(http, private_root,
        '/'+str(service_path.relative_to('/sys/fs/cgroup')))
    sources = {str(SCRIPT_DIR/name): _sha(SCRIPT_DIR/name) for name in
        ('replay_vllm_engine_trace.py', 'ieee_tc_serverless_measurement.py',
         'prepare_ieee_tc_serverless_stack.py')}
    worker_cfg = dict(cfg, model=http['model'], main_repo=http['main_repo'], output=str(output),
        source_sha256=sources, native_toolchain=toolchain,
        remote_config=remote, adapters={aid: [i+1, paths[aid]] for i, aid in enumerate(sorted(paths))})
    worker_path = output/'worker_config.json'
    _exclusive_json(worker_path, worker_cfg)
    router = ResidentRoundRobin([e.request_id for e in plan.entries], len(cfg['gpu_ids']),
                                local_monotonic_clock_id())
    result = dict(schema='ieee_tc_vllm_native_model_qualification_v1', passed=False,
        qualification_only=True, numerical_correctness_qualified=False,
        configuration=cfg, configuration_sha256=_sha(config_path),
        http_config_sha256=_sha(http_path), source_sha256=sources,
        service=admission, requests=[])
    owners, processes, logs, starts = [], [], [], []
    app_runner = None
    failure = None
    stop = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)

    async def await_ready(index, owner, process):
        path = output/f'worker_{index}.ready.json'
        deadline = context['deployment_notice_s']+1800.
        while not path.exists():
            if process.poll() is not None:
                raise RuntimeError(f'vLLM worker {index} exited before ready: {process.returncode}')
            if stop.is_set() or time.perf_counter() >= deadline:
                raise TimeoutError('native startup interrupted or qualification protection elapsed')
            await asyncio.sleep(.05)
        # A file is notification, not GPU proof. Bind live contexts to this
        # exact allocation before releasing any queued request to the worker.
        sample = owner.census.sample(service_path)
        workers = [dict(device_uuid=d['gpu_uuid'], pid=p['pid'], worker_rank=index)
            for d in sample['devices'] if d['gpu_uuid'] in owner.gpu_uuids
            for p in d['processes'] if p['service_member'] and p['kind'] == 'compute']
        owner.confirm_workers(workers)
        ready = _load_json(path)
        receipt = _load_json(output/'worker_receipts'/(ready['tc_worker_receipt_id']+'.json'))
        if (ready['tc_worker_receipt_sha256'] != _sha(output/'worker_receipts'/(ready['tc_worker_receipt_id']+'.json'))
                or receipt['identity'] != guard.gpu_process_identity(process.pid) or receipt['index'] != index
                or receipt['worker_config_sha256'] != _sha(worker_path)
                or receipt['clock_id'] != router.clock_id):
            raise ValueError('native ready receipt/source/process identity changed')
        router.ready[index].set()

    async def wait_for_replay():
        deadline = context['replay_t0_s']+max(e.offset_s for e in plan.entries)+1800.+60.
        with Path(context['result_path']).open() as handle:
            while True:
                if stop.is_set():
                    raise RuntimeError('resident service interrupted')
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError('resident worker exited during offered workload')
                for task in starts:
                    if task.done():
                        task.result()
                pos = handle.tell()
                line = handle.readline()
                if not line.endswith('\n'):
                    handle.seek(pos)
                    if time.perf_counter() >= deadline:
                        raise TimeoutError('external publisher terminal record absent')
                    await asyncio.sleep(.1)
                    continue
                row = json.loads(line)
                if row['event'] == 'http_raw_response':
                    result['requests'].append(dict(request_id=row['request_id'], response=row['body'],
                                                   http_status=row['status']))
                if row['event'] in ('http_replay_complete', 'http_replay_incomplete'):
                    result['http_completion'] = row
                    if (row['event'] != 'http_replay_complete' or row.get('N_failed') != 0
                            or row.get('N_response') != len(plan.entries)):
                        raise ValueError('incomplete/failed native HTTP workload; retained for diagnosis')
                    return

    try:
        with PreReadyHTTPIngress(port=ingress_port,
                upstream_url=f'http://127.0.0.1:{cfg["router_port"]}/v1/chat/completions',
                model=http['model'], request_count=len(plan.entries),
                journal_path=output/'ingress.jsonl') as ingress:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0, force_close=True),
                    timeout=aiohttp.ClientTimeout(total=None), trust_env=False) as session:
                async def route(request):
                    received = time.perf_counter()
                    index, body = await router.route(await request.json(), received)
                    url = f'http://127.0.0.1:{cfg["worker_ports"][index]}/v1/chat/completions'
                    async with session.post(url, json=body, allow_redirects=False) as response:
                        return web.Response(body=await response.read(), status=response.status,
                                            content_type='application/json')

                app = web.Application()
                app.router.add_post('/v1/chat/completions', route)
                app_runner = web.AppRunner(app, access_log=None, handler_cancellation=True, shutdown_timeout=1.)
                await app_runner.setup()
                await web.TCPSite(app_runner, '127.0.0.1', cfg['router_port']).start()
                ingress.router_ready()
                for index, device in enumerate(cfg['gpu_ids']):
                    api = guard.load_nvml_binding(Path(os.environ['FAASLORA_TC_NVML_BINDING']),
                                                  os.environ['FAASLORA_TC_NVML_SHA256'])
                    census = guard.NativeGPUCensus(api)
                    try:
                        owner = PhysicalGPUAllocation(root=deployment.allocations, census=census,
                            service_path=service_path, device_indices=(device,),
                            owner_identity=guard.gpu_process_identity(os.getpid()))
                    except BaseException:
                        census.close()
                        raise
                    owners.append(owner)
                    log = (output/f'worker_{index}.log').open('xb')
                    logs.append(log)
                    # Backend imports occur only in this separate interpreter;
                    # no dirty replay or repository startup hook is imported.
                    env = worker_environments[index][0]
                    process = subprocess.Popen([cfg['python'], '-B', str(Path(__file__).resolve()),
                        'native-http-worker', '--config', str(worker_path), '--index', str(index)],
                        stdout=log, stderr=subprocess.STDOUT, env=env)
                    processes.append(process)
                    owner.bind_process(process, guard.gpu_process_identity(process.pid))
                    starts.append(asyncio.create_task(await_ready(index, owner, process)))
                await wait_for_replay()
                result['passed'] = True
    except Exception as exc:
        failure = exc
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        for task in starts:
            task.cancel()
        await asyncio.gather(*starts, return_exceptions=True)
        if app_runner is not None:
            await app_runner.cleanup()
        for process in processes:
            if process.poll() is None:
                process.terminate()  # Exact owned Popen; no name/group guessing.
        waits = await asyncio.gather(*(asyncio.to_thread(p.wait, timeout=60) for p in processes),
                                     return_exceptions=True)
        for index, owner in enumerate(owners):
            try:
                if index < len(waits) and isinstance(waits[index], BaseException):
                    raise waits[index]
                if owner.process is not None and any(e['event'] == 'native_workers' for e in owner.events):
                    await owner.wait_workers(timeout_s=60.)
                owner.release()  # pidfd exit AND native empty census.
            except Exception as exc:
                result.setdefault('physical_release_errors', []).append(str(exc))
                result['passed'] = False
                failure = failure or exc
        for log in logs:
            log.close()
        result['physical_allocations'] = [owner.evidence() for owner in owners]
        try:
            record_http_physical_terminals(deployment, context['result_path'])
            physical = deployment.summarize(observed_until_s=time.monotonic())
            result['physical_lifecycle'] = dict(mode='vllm_resident_pool_v1',
                summary_path=str(deployment.root/'physical_resource_summary.json'),
                measurement_complete=physical['measurement_complete'])
            if (not physical['measurement_complete']
                    or physical['n_native_contract_complete'] != len(plan.entries)):
                raise ValueError('physical or native workload measurement incomplete')
        except Exception as exc:
            result['physical_terminal_error'] = str(exc)
            result['passed'] = False
            failure = failure or exc
        _exclusive_json(output/'model_qualification.json', result)
        deployment.finalize()
    if failure:
        raise RuntimeError('vLLM qualification failed; preserve original evidence') from failure
    return result


def native_http_main(argv):
    parser = argparse.ArgumentParser(description='Opt-in shared TC native HTTP measurement')
    parser.add_argument('mode', choices=('native-http-service', 'native-http-worker'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--index', type=int)
    parser.add_argument('--http-replay-config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--private-root', type=Path)
    args = parser.parse_args(argv)
    if args.mode == 'native-http-worker':
        if args.index is None:
            parser.error('worker requires --index')
        asyncio.run(_native_worker(args.config, args.index))
    else:
        if any(p is None or not p.is_absolute() for p in
               (args.http_replay_config, args.output, args.private_root)):
            parser.error('service requires absolute publisher/output/private paths')
        asyncio.run(_resident_http_service(args.config, args.http_replay_config, args.output, args.private_root))
    return 0


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_messages_prompt(messages: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{str(message.get('role') or 'user').capitalize()}: "
        f"{'' if message.get('content') is None else str(message.get('content'))}"
        for message in messages
    )


def _as_bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def _filtered_dataclass(cls: Any, kwargs: Dict[str, Any]) -> Any:
    sig = inspect.signature(cls)
    filtered = {key: value for key, value in kwargs.items() if key in sig.parameters and value is not None}
    return cls(**filtered)


def _make_lora_request(cls: Any, name: str, int_id: int, path: str, base_model: str) -> Any:
    sig = inspect.signature(cls)
    kwargs = {
        "lora_name": name,
        "lora_int_id": int_id,
        "lora_path": path,
        "base_model_name": base_model,
    }
    try:
        return cls(**{key: value for key, value in kwargs.items() if key in sig.parameters})
    except TypeError:
        return cls(name, int_id, path)


def _make_sampling_params(cls: Any, body: Dict[str, Any], max_tokens: int) -> Any:
    kwargs = {
        "temperature": float(body.get("temperature", 0.0) or 0.0),
        "top_p": float(body.get("top_p", 1.0) or 1.0),
        "max_tokens": max(1, int(max_tokens)),
    }
    if body.get("ignore_eos") is not None:
        kwargs["ignore_eos"] = bool(body.get("ignore_eos"))
    sig = inspect.signature(cls)
    return cls(**{key: value for key, value in kwargs.items() if key in sig.parameters})


def _prepare_prompt(tokenizer: Any, request: Dict[str, Any], max_model_len: int, max_tokens: int) -> tuple[str, int]:
    body = dict(request.get("body") or {})
    prompt = body.get("prompt")
    if prompt is None:
        prompt = _render_messages_prompt(list(body.get("messages") or []))
    prompt = str(prompt or "")
    budget = max(8, int(max_model_len) - int(max_tokens) - 8)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) > budget:
        token_ids = token_ids[-budget:]
        prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
    return prompt, len(token_ids)


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(1.0, q / 100.0)) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _fmt_ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def _print_live(label: str, start_perf: float, total: int, results: List[Dict[str, Any]], active: int) -> None:
    done = len(results)
    ok = sum(1 for item in results if item.get("success"))
    fail = done - ok
    elapsed = max(1e-9, time.perf_counter() - start_perf)
    ttfts = [float(item["overall_ttft_ms"]) for item in results if item.get("success") and item.get("overall_ttft_ms") is not None]
    e2es = [float(item["overall_e2e_ms"]) for item in results if item.get("success") and item.get("overall_e2e_ms") is not None]
    tpots = [float(item["tpot_ms"]) for item in results if item.get("success") and item.get("tpot_ms") is not None]
    toks = sum(int(item.get("completion_tokens") or 0) for item in results if item.get("success"))
    eta = (total - done) / max(done / elapsed, 1e-9) if done else 0.0
    width = 28
    filled = int(round(min(1.0, done / max(1, total)) * width))
    bar = "[" + "#" * filled + "-" * (width - filled) + "]"
    print(
        f"[live:{label}] {bar} done={done}/{total} ok={ok} fail={fail} "
        f"active={active} elapsed={elapsed:.0f}s eta={eta:.0f}s req/s={done/elapsed:.2f} tok/s={toks/elapsed:.2f}",
        flush=True,
    )
    print(
        f"[live:{label}] ttft(avg/p95/p99)="
        f"{_fmt_ms((sum(ttfts) / len(ttfts)) if ttfts else None)}/"
        f"{_fmt_ms(_percentile(ttfts, 95))}/{_fmt_ms(_percentile(ttfts, 99))}ms "
        f"e2e(avg/p95/p99)="
        f"{_fmt_ms((sum(e2es) / len(e2es)) if e2es else None)}/"
        f"{_fmt_ms(_percentile(e2es, 95))}/{_fmt_ms(_percentile(e2es, 99))}ms "
        f"tpot={_fmt_ms((sum(tpots) / len(tpots)) if tpots else None)}ms",
        flush=True,
    )


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in ('native-http-service', 'native-http-worker'):
        return native_http_main(sys.argv[1:])
    # Legacy path unchanged; the new native service does not import this
    # user-owned dirty replay module or inherit its external loader behavior.
    from replay_openai_trace import RemoteArtifactFetcher
    ap = argparse.ArgumentParser(description="Replay a shared trace directly through a vLLM-compatible LLMEngine.")
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--adapter-subset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tokenizer-path")
    ap.add_argument("--label", default="vllm_engine")
    ap.add_argument("--generation-seed", type=int, default=42)
    ap.add_argument("--max-requests", type=int, default=0)
    ap.add_argument("--sleep-scale", type=float, default=1.0)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    ap.add_argument("--max-loras", type=int, default=4)
    ap.add_argument("--max-cpu-loras", type=int, default=500)
    ap.add_argument("--max-lora-rank", type=int, default=64)
    ap.add_argument("--max-num-seqs", type=int, default=2)
    ap.add_argument("--max-num-batched-tokens", type=int, default=1024)
    ap.add_argument("--enforce-eager", type=_as_bool, default=False)
    ap.add_argument("--trust-remote-code", type=_as_bool, default=True)
    ap.add_argument("--remote-artifact-endpoint", default="")
    ap.add_argument("--remote-artifact-cache-dir", type=Path)
    ap.add_argument("--remote-artifact-timeout-s", type=float, default=3600.0)
    ap.add_argument("--remote-artifact-bandwidth-mbps", type=float, default=250.0)
    args = ap.parse_args()

    trace_path = args.trace.expanduser().resolve()
    subset_path = args.adapter_subset.expanduser().resolve()
    trace = _load_json(trace_path)
    subset = _load_json(subset_path)
    requests = list(trace.get("requests") or [])
    if args.max_requests > 0:
        requests = requests[: args.max_requests]
    total = len(requests)
    if total <= 0:
        raise SystemExit("trace has no requests to replay")

    remote_dir = Path(subset["remote_dir"]).expanduser().resolve()
    adapters = {str(item["id"]): item for item in subset.get("adapters") or []}
    adapter_int_ids = {adapter_id: idx + 1 for idx, adapter_id in enumerate(sorted(adapters))}
    remote_cache_dir = (
        args.remote_artifact_cache_dir.expanduser().resolve()
        if args.remote_artifact_cache_dir
        else Path("results/remote_artifact_cache/vllm_engine") / args.label
    )
    fetcher = None
    if args.remote_artifact_endpoint:
        fetcher = RemoteArtifactFetcher(
            endpoint=args.remote_artifact_endpoint,
            timeout_s=args.remote_artifact_timeout_s,
            token_env="PRIME_REMOTE_TOKEN",
            bandwidth_mbps=args.remote_artifact_bandwidth_mbps,
        )

    from transformers import AutoTokenizer
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.lora.request import LoRARequest

    model_path = str(Path(args.model_path).expanduser().resolve())
    tokenizer_path = str(Path(args.tokenizer_path or args.model_path).expanduser().resolve())
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)

    engine_kwargs = {
        "model": model_path,
        "tokenizer": tokenizer_path,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_lora": True,
        "max_loras": args.max_loras,
        "max_cpu_loras": args.max_cpu_loras,
        "max_lora_rank": args.max_lora_rank,
        "enforce_eager": args.enforce_eager,
        "disable_log_stats": True,
    }
    engine = LLMEngine.from_engine_args(_filtered_dataclass(EngineArgs, engine_kwargs))
    if hasattr(engine, "init_phase2"):
        engine.init_phase2()

    pending = sorted(requests, key=lambda item: float(item.get("arrival_time_s", 0.0) or 0.0))
    in_flight: Dict[str, Dict[str, Any]] = {}
    results: List[Dict[str, Any]] = []
    start_perf = time.perf_counter()
    next_live = start_perf + LIVE_PRINT_INTERVAL_S

    while pending or engine.has_unfinished_requests():
        now = time.perf_counter()
        elapsed = now - start_perf
        added = False
        while pending and elapsed + 1e-9 >= float(pending[0].get("arrival_time_s", 0.0) or 0.0) * max(0.0, args.sleep_scale):
            request = pending.pop(0)
            request_id = str(request.get("request_id") or f"req_{len(in_flight) + len(results):05d}")
            body = dict(request.get("body") or {})
            adapter_id = str(body.get("lora_adapter_name") or request.get("adapter_id") or "")
            max_tokens = int(body.get("max_tokens") or request.get("expected_output_tokens") or 1)
            scheduled_offset_s = float(request.get("arrival_time_s", 0.0) or 0.0) * max(0.0, args.sleep_scale)
            dispatch_start_perf = time.perf_counter()
            remote_metrics: Dict[str, Any] = {}
            adapter_path = remote_dir / adapter_id
            if fetcher is not None:
                adapter_path = remote_cache_dir / adapter_id
                remote_metrics = fetcher.ensure(adapter_id, str(adapter_path))
            elif not adapter_path.exists():
                raise RuntimeError(f"missing local adapter path for {adapter_id}: {adapter_path}")
            prompt, guard_prompt_tokens = _prepare_prompt(tokenizer, request, args.max_model_len, max_tokens)
            sampling_params = _make_sampling_params(SamplingParams, body, max_tokens)
            lora_request = _make_lora_request(
                LoRARequest,
                adapter_id,
                adapter_int_ids.get(adapter_id, len(adapter_int_ids) + 1),
                str(adapter_path),
                model_path,
            )
            engine.add_request(request_id, prompt, sampling_params, lora_request=lora_request)
            add_perf = time.perf_counter()
            in_flight[request_id] = {
                "request": request,
                "body": body,
                "adapter_id": adapter_id,
                "scheduled_offset_s": scheduled_offset_s,
                "dispatch_start_perf": dispatch_start_perf,
                "add_perf": add_perf,
                "first_perf": None,
                "last_seen_tokens": 0,
                "guard_prompt_tokens": guard_prompt_tokens,
                "max_tokens": max_tokens,
                "remote_metrics": remote_metrics,
            }
            added = True
            elapsed = time.perf_counter() - start_perf

        outputs = engine.step() if (added or engine.has_unfinished_requests()) else []
        step_perf = time.perf_counter()
        for output in outputs:
            rid = str(output.request_id)
            meta = in_flight.get(rid)
            if meta is None:
                continue
            completion = output.outputs[0] if output.outputs else None
            token_ids = list(getattr(completion, "token_ids", []) or []) if completion is not None else []
            if token_ids and meta.get("first_perf") is None:
                meta["first_perf"] = step_perf
            meta["last_seen_tokens"] = max(int(meta.get("last_seen_tokens") or 0), len(token_ids))
            if not getattr(output, "finished", False):
                continue
            first_perf = float(meta.get("first_perf") or step_perf)
            add_perf = float(meta["add_perf"])
            scheduled_perf = start_perf + float(meta["scheduled_offset_s"])
            prompt_token_ids = list(getattr(output, "prompt_token_ids", []) or [])
            prompt_tokens = len(prompt_token_ids) or int(meta["guard_prompt_tokens"])
            completion_tokens = len(token_ids)
            overall_ttft_ms = max(0.0, (first_perf - scheduled_perf) * 1000.0)
            overall_e2e_ms = max(0.0, (step_perf - scheduled_perf) * 1000.0)
            service_ttft_ms = max(0.0, (first_perf - add_perf) * 1000.0)
            service_e2e_ms = max(0.0, (step_perf - add_perf) * 1000.0)
            tpot_ms = None
            if completion_tokens > 1:
                tpot_ms = max(0.0, (service_e2e_ms - service_ttft_ms) / (completion_tokens - 1))
            result = {
                "request_id": rid,
                "generation_seed": args.generation_seed,
                "arrival_time_s": float(meta["scheduled_offset_s"]),
                "dispatch_offset_s": max(0.0, add_perf - start_perf),
                "completion_offset_s": max(0.0, step_perf - start_perf),
                "adapter_id": meta["adapter_id"],
                "ttft_ms": overall_ttft_ms,
                "e2e_ms": overall_e2e_ms,
                "overall_ttft_ms": overall_ttft_ms,
                "overall_e2e_ms": overall_e2e_ms,
                "service_ttft_ms": service_ttft_ms,
                "service_e2e_ms": service_e2e_ms,
                "dispatch_admission_wait_ms": max(0.0, (add_perf - scheduled_perf) * 1000.0),
                "replay_dispatch_wait_ms": max(0.0, (add_perf - scheduled_perf) * 1000.0),
                "server_queue_wait_ms": 0.0,
                "tpot_ms": tpot_ms,
                "tpot_observed": tpot_ms is not None,
                "runtime_ttft_ms": service_ttft_ms,
                "service_overhead_ms": 0.0,
                "lora_load_ms": None,
                "metrics_source": "vllm_engine",
                "status_code": 200,
                "success": True,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_token_source": "engine",
                "completion_token_source": "engine",
                "guard_prompt_tokens": int(meta["guard_prompt_tokens"]),
                "guard_max_tokens": int(meta["max_tokens"]),
                "generated_text_chars": len(str(getattr(completion, "text", "") or "")) if completion is not None else 0,
                "raw_text_preview": str(getattr(completion, "text", "") or "")[:240] if completion is not None else "",
                "error": None,
                **dict(meta.get("remote_metrics") or {}),
            }
            results.append(result)
            in_flight.pop(rid, None)

        if time.perf_counter() >= next_live:
            _print_live(args.label, start_perf, total, results, len(in_flight))
            next_live = time.perf_counter() + LIVE_PRINT_INTERVAL_S

        if not engine.has_unfinished_requests() and pending:
            target = float(pending[0].get("arrival_time_s", 0.0) or 0.0) * max(0.0, args.sleep_scale)
            delay = max(0.0, target - (time.perf_counter() - start_perf))
            time.sleep(min(delay, 0.05))

    payload = {
        "metric_schema_version": "e2e_v3",
        "metric_definitions": {
            "primary_ttft": "scheduled trace arrival to first generated output token/chunk",
            "primary_e2e": "scheduled trace arrival to response completion",
            "service_ttft": "engine add_request to first generated output token/chunk",
            "service_e2e": "engine add_request to response completion",
        },
        "trace_source": str(trace_path),
        "adapter_subset_path": str(subset_path),
        "base_url": "inprocess://vllm-engine",
        "label": args.label,
        "generation_seed": args.generation_seed,
        "sleep_scale": args.sleep_scale,
        "model_path": model_path,
        "engine_config": engine_kwargs,
        "remote_artifact_endpoint": args.remote_artifact_endpoint or None,
        "remote_artifact_cache_dir": str(remote_cache_dir) if args.remote_artifact_endpoint else None,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_live(args.label, start_perf, total, results, len(in_flight))
    ok = sum(1 for item in results if item.get("success"))
    print(f"wrote replay -> {args.output} ok={ok}/{total}", flush=True)
    if ok != total:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
