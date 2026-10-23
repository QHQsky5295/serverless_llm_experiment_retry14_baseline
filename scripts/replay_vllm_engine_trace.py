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
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
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
    compiler_receipt = None
    if cfg.get('native_compiler_contract') is None and any(env.get(key) for key in (
            'FAASLORA_TC_NATIVE_TOOLCHAIN_CONTRACT', 'FAASLORA_TC_NATIVE_TOOLCHAIN_SHA256',
            'FAASLORA_TC_NATIVE_TOOLCHAIN_BOUND_SHA256')):
        raise ValueError('compiler contract must be explicit in native service configuration')
    if cfg.get('native_compiler_contract') is not None:
        from scripts.ieee_tc_native_toolchain import bind_native_toolchain
        binding = cfg['native_compiler_contract']
        if not isinstance(binding, dict) or set(binding) != {'path', 'sha256'}:
            raise ValueError('explicit native compiler contract path/SHA required')
        env, compiler_receipt = bind_native_toolchain(env, python=str(python),
            contract=binding['path'], expected_sha=binding['sha256'])
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
    if compiler_receipt is not None:
        if env['MAX_JOBS'] != str(jobs):
            raise ValueError('build_max_jobs differs from native compiler contract')
        identity['compiler_contract'] = compiler_receipt
    return env, identity


def validate_native_service_config(cfg, http):
    """Fixed-version resident transport adapter, not a second scheduling policy."""
    from prepare_ieee_tc_serverless_stack import validate_http_lifecycle_contract
    validate_http_lifecycle_contract(http)
    if cfg.get('schema') == 'ieee_tc_slora_native_service_v1':
        return validate_slora_service_config(cfg, http)
    if cfg.get('schema') == 'ieee_tc_loquetier_native_service_v1':
        return validate_loquetier_service_config(cfg, http)
    if cfg.get('schema') == 'ieee_tc_dlora_native_service_v1':
        return validate_dlora_service_config(cfg, http)
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


SLORA_INPUT_KEYS = ('max_req_total_len', 'max_total_token_num', 'pool_size_lora',
    'batch_max_tokens', 'running_max_req_size', 'swap', 'prefetch', 'prefetch_size',
    'scheduler', 'profile', 'batch_num_adapters', 'enable_abort', 'dummy',
    'no_lora_compute', 'no_lora_swap', 'no_kernel', 'no_mem_pool', 'bmm', 'no_lora', 'fair_weights')
SLORA_ENV_KEYS = ('PATH', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'PYTHONPATH', 'PYTHONNOUSERSITE', 'MAX_JOBS')


def validate_slora_service_config(cfg, http):
    from urllib.parse import urlsplit
    from prepare_ieee_tc_serverless_stack import validate_http_lifecycle_contract
    validate_http_lifecycle_contract(http)
    devices = cfg.get('gpu_ids', [])
    endpoint = urlsplit(http['url'])
    ports = [endpoint.port, cfg.get('router_port'), *cfg.get('worker_ports', []), *cfg.get('nccl_ports', [])]
    if (cfg.get('schema') != 'ieee_tc_slora_native_service_v1'
            or http.get('schema') != 'ieee_tc_slora_http_replay_v1'
            or cfg.get('routing') != 'round_robin' or cfg.get('worker_startup') != 'sequential_v1'
            or cfg.get('generation_contract') != 'fixed_length_greedy_v1'
            or not devices or type(devices) is not list or len(set(devices)) != len(devices)
            or any(type(d) is not int or d not in range(4) for d in devices)
            or len(cfg.get('worker_ports', [])) != len(devices)
            or len(cfg.get('nccl_ports', [])) != len(devices)
            or any(type(p) is not int or not 0 < p < 65536 for p in ports)
            or len(set(ports)) != len(ports)
            or endpoint.scheme != 'http' or endpoint.hostname != '127.0.0.1'
            or endpoint.path != '/v1/chat/completions' or endpoint.query or endpoint.fragment
            or endpoint.username or endpoint.password or not http.get('remote_artifacts')
            or cfg.get('host_layout') not in ('native_v1', 'packed_views_v1')):
        raise ValueError('explicit S-LoRA full-pool/resident/sequential/loopback contract required')
    if cfg.get('input_mode', 'native_text_v1') not in ('native_text_v1', 'canonical_token_ids_v1'):
        raise ValueError('unknown explicit S-LoRA input mode')
    args = cfg.get('native_args', {})
    # Preserve the existing explicit native BMM path used for GQA models by
    # run_slora_fair_experiment.sh. Packed HOST views are an audited non-GQA,
    # non-BMM representation only; never silently select or fall back to it.
    bmm = args.get('bmm')
    if type(bmm) is not bool or (bmm and cfg['host_layout'] != 'native_v1'):
        raise ValueError('native S-LoRA BMM requires explicit native HOST layout')
    fixed = dict(model_dir=http['backbone'], tokenizer_mode='auto', tp=1,
        max_req_total_len=1024, swap=True, scheduler='slora', dummy=False, bmm=bmm,
        no_lora=False, no_kernel=False, no_lora_compute=False, no_lora_swap=False,
        no_mem_pool=False, profile=False, enable_abort=False, prefetch=False,
        prefetch_size=0, pool_size_lora=0, batch_num_adapters=None, fair_weights=[],
        trust_remote_code=True, disable_log_stats=True, eos_id=2, log_stats_interval=10)
    keys = set(fixed) | {'max_req_input_len', 'max_total_token_num', 'batch_max_tokens', 'running_max_req_size'}
    if (set(args) != keys or any(args.get(k) != v or type(args.get(k)) is not type(v)
                               for k, v in fixed.items())
            or any(type(args[k]) is not int or args[k] <= 0 for k in keys-set(fixed))
            or not 759 <= args['max_req_input_len'] < 1024
            or not 1024 <= args['batch_max_tokens'] <= args['max_total_token_num']):
        raise ValueError('explicit native S-LoRA args differ from supported original serving path')
    for key in ('python', 'native_python', 'source_view'):
        if not isinstance(cfg.get(key), str) or not Path(cfg[key]).is_absolute():
            raise ValueError('explicit interpreter/source paths required')
    return endpoint.port


def verify_slora_source_view(cfg):
    root = Path(cfg['source_view'])
    path = root/'measurement_manifest.json'
    if _sha(path) != cfg['source_view_manifest_sha256']:
        raise ValueError('S-LoRA source view manifest changed')
    manifest = _load_json(path)
    if (manifest.get('schema') != 'ieee_tc_slora_measurement_view_v1'
            or manifest.get('source_view') != str(root.resolve())
            or manifest.get('worker_lifecycle') is not True
            or manifest.get('worker_lifecycle_scope') != 'TP1_owned_router_detokenizer'
            or manifest.get('host_layout') != cfg['host_layout']
            or manifest.get('input_mode', 'native_text_v1') != cfg.get('input_mode', 'native_text_v1')
            or manifest.get('helper_path') != str(SCRIPT_DIR/'ieee_tc_serverless_measurement.py')
            or manifest.get('helper_sha256') != _sha(manifest['helper_path'])
            or manifest.get('source_adapter_sha256') != _sha(SCRIPT_DIR/'prepare_ieee_tc_serverless_stack.py')
            or manifest.get('shared_package') != str((Path(cfg['main_repo'])/'faaslora').resolve())):
        raise ValueError('S-LoRA source view helper/layout/lifecycle identity differs')
    for key, base in (('original_sha256', Path(manifest['native_source'])),
                      ('measured_sha256', root), ('shared_source_sha256', Path(cfg['main_repo']))):
        for name, digest in manifest[key].items():
            if _sha(base/name) != digest:
                raise ValueError('S-LoRA guarded source changed: '+name)
    return manifest


def slora_worker_environment(cfg, device, environ=None):
    python = Path(cfg['native_python'])
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError('declared native S-LoRA interpreter unavailable')
    root = python.parent.parent
    torchlib = root/'lib/python3.9/site-packages/torch/lib'
    if not torchlib.is_dir():
        raise ValueError('historical native Python3.9 torch library path missing')
    env = dict(os.environ if environ is None else environ)
    env.pop('PYTHONHOME', None)
    env.update(PATH=str(python.parent)+os.pathsep+env.get('PATH', ''),
        CUDA_HOME=str(root), LD_LIBRARY_PATH=str(torchlib)+os.pathsep+str(root/'lib')
            +os.pathsep+env.get('LD_LIBRARY_PATH', ''),
        PYTHONPATH=cfg['source_view'], PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
        PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES=str(device), MAX_JOBS='2')
    return env


def slora_worker_context(config_path, index):
    """Before backend import and again in the real native GPU process."""
    from prepare_ieee_tc_serverless_stack import load_guard
    cfg = _load_json(config_path)
    expected_sources = {str(SCRIPT_DIR/name) for name in (
        'replay_vllm_engine_trace.py', 'ieee_tc_serverless_measurement.py', 'prepare_ieee_tc_serverless_stack.py')}
    if (cfg.get('schema') != 'ieee_tc_slora_native_service_v1'
            or type(index) is not int or not 0 <= index < len(cfg['gpu_ids'])
            or set(cfg.get('source_sha256', {})) != expected_sources
            or any(_sha(p) != h for p, h in cfg['source_sha256'].items())
            or os.environ.get('FAASLORA_TC_SLORA_WORKER_CONFIG_SHA256') != _sha(config_path)
            or Path(sys.executable).resolve() != Path(cfg['native_python']).resolve()
            or sys.version_info[:2] != (3, 9)
            or os.environ.get('CUDA_VISIBLE_DEVICES') != str(cfg['gpu_ids'][index])
            or os.environ.get('PYTHONPATH') != cfg['source_view']
            or set(cfg.get('native_environment', {})) != set(SLORA_ENV_KEYS)
            or any(os.environ.get(k) != v for k, v in cfg['native_environment'].items())
            or len(cfg.get('adapters', {})) != 500
            or {i for i, path in cfg['adapters'].values()} != set(range(500))
            or any(not Path(path).is_absolute() or Path(path).name != aid
                   for aid, (i, path) in cfg['adapters'].items())
            or cfg.get('native_input_params') != {k: cfg['native_args'][k] for k in SLORA_INPUT_KEYS}):
        raise ValueError('S-LoRA native process/source/config identity differs')
    verify_slora_source_view(cfg)
    sys.path.insert(0, cfg['main_repo'])
    guard = load_guard(Path(cfg['main_repo']))
    guard.verify_current_service()
    return cfg, guard


def _native_slora_worker(config_path, index):
    cfg, _ = slora_worker_context(config_path, index)
    # The original API owns router/detokenizer handles; the opt-in source view
    # adds finally-cleanup. A startup SIGTERM must unwind that same scope.
    def stop(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stop)
    import torch
    from slora.server import api_server
    torch.multiprocessing.set_start_method('spawn')
    argv = [str(Path(api_server.__file__)), '--host', '127.0.0.1',
            '--port', str(cfg['worker_ports'][index]), '--nccl_port', str(cfg['nccl_ports'][index])]
    for key, value in cfg['native_args'].items():
        flag = '--'+(key.replace('_', '-') if key in ('pool_size_lora', 'prefetch_size',
            'batch_num_adapters', 'enable_abort', 'no_lora_swap', 'no_lora_compute',
            'no_kernel', 'no_mem_pool', 'no_lora', 'fair_weights') else key)
        if type(value) is bool:
            if value:
                argv.append(flag)
        elif value is not None and value != []:
            argv.extend([flag, str(value)])
    for i, path in cfg['adapters'].values():
        argv.extend(['--lora-dirs', path])
    sys.argv = argv
    api_server.main()


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


def validate_dlora_service_config(cfg, http):
    """Retained DP2/TP1 work point; transport must not replace native routing."""
    from urllib.parse import urlsplit
    from prepare_ieee_tc_serverless_stack import validate_http_lifecycle_contract
    validate_http_lifecycle_contract(http)
    endpoint = urlsplit(http['url'])
    ports = [endpoint.port, cfg.get('router_port'), *cfg.get('worker_ports', [])]
    if (cfg.get('schema') != 'ieee_tc_dlora_native_service_v1'
            or http.get('schema') != 'ieee_tc_dlora_http_replay_v1'
            or cfg.get('routing') != 'native_engine_manager'
            or cfg.get('worker_startup') != 'native_dp2_manager_v1'
            or cfg.get('generation_contract') != 'fixed_length_greedy_v1'
            or cfg.get('migration_type') != 3 or cfg.get('gpu_ids') != [0, 1]
            or len(cfg.get('worker_ports', [])) != 1
            or any(type(p) is not int or not 0 < p < 65536 for p in ports)
            or len(set(ports)) != 3 or endpoint.scheme != 'http' or endpoint.hostname != '127.0.0.1'
            or endpoint.path != '/v1/chat/completions' or endpoint.query or endpoint.fragment
            or endpoint.username or endpoint.password or not http.get('remote_artifacts')):
        raise ValueError('dLoRA requires native DP2 manager, one API and shared remote/pre-ready input')
    args = cfg.get('engine_args', {})
    fixed = dict(model=http['backbone'], tokenizer=http['backbone'], use_dummy_weights=False,
        use_np_weights=False, dtype='half', worker_use_ray=True, engine_use_ray=True,
        tensor_parallel_size=1, pipeline_parallel_size=1, num_groups=2, num_models=500,
        exec_type=3, gpu_memory_utilization=0.92, max_num_seqs=4, max_num_batched_tokens=1024,
        gpu_capacity=8, swap_space=8)
    if (any(args.get(k) != value or type(args.get(k)) is not type(value) for k, value in fixed.items())
            or set(args)-set(fixed) or cfg.get('request_max_model_len') != 1024):
        raise ValueError('retained dLoRA native arguments differ; no implicit performance search')
    for key in ('python', 'native_python', 'source_view', 'ray_root'):
        if not isinstance(cfg.get(key), str) or not Path(cfg[key]).is_absolute():
            raise ValueError('explicit dLoRA interpreter/source/private paths required')
    dlora_private_ray_options(cfg)
    return endpoint.port


def dlora_worker_environment(cfg, environ=None):
    env = dict(os.environ if environ is None else environ)
    native_bin = str(Path(cfg['native_python']).parent)
    for key in ('RAY_ADDRESS', 'PYTHONHOME'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES='0,1', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
        PYTHONUNBUFFERED='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        PATH=native_bin+os.pathsep+env.get('PATH', ''),
        PYTHONPATH=os.pathsep.join((cfg['source_view'], cfg['main_repo'], str(SCRIPT_DIR))))
    return env


class NativeManagerIngress:
    """Wait for one native manager; do not choose an engine or inspect locality."""
    def __init__(self, request_ids, clock_id):
        if not request_ids or len(set(request_ids)) != len(request_ids):
            raise ValueError('unique offered requests required')
        self.requests, self.seen = set(request_ids), set()
        self.clock_id, self.ready = clock_id, [asyncio.Event()]

    async def route(self, body, received):
        rid = body.get('request_id')
        control = dict(body.get('_sllm_internal_metrics', {}))
        if rid not in self.requests or rid in self.seen or control.get('tc_clock_id') != self.clock_id:
            raise ValueError('native-manager ingress received an unknown/duplicate request or clock')
        self.seen.add(rid)
        control.update(tc_http_received_s=received, tc_router_enqueued_s=received,
            ready_instances_at_enqueue=int(self.ready[0].is_set()))
        await self.ready[0].wait()
        return 0, dict(body, _sllm_internal_metrics=control)


def dlora_private_ray_options(cfg):
    """One new local Ray node, never discovery/auto or a shared cluster.

    The 8 GiB object store is inside the whole-service cgroup. CPU/GPU counts
    describe Ray resources; actual isolation is verified separately by the TC
    launcher and native worker guard, not inferred from these logical counts.
    """
    root = Path(cfg['ray_root'])
    if (not root.is_absolute() or root.is_symlink() or '..' in root.parts
            or cfg.get('gpu_ids') != [0, 1]
            or cfg.get('ray_object_store_bytes') != 8*1024**3
            or cfg.get('ray_num_cpus') != 40):
        raise ValueError('explicit private Ray root and retained DP2 resource contract required')
    return dict(address='local', namespace='primelora-tc-dlora',
        num_cpus=cfg['ray_num_cpus'], num_gpus=2,
        object_store_memory=cfg['ray_object_store_bytes'], include_dashboard=False,
        log_to_driver=False, _temp_dir=str(root),
        object_spilling_directory=str(root/'spill'), _node_ip_address='127.0.0.1')


def dlora_worker_context(config_path, *, actor=False):
    """Current process must belong to the admitted service and allocated pool."""
    from prepare_ieee_tc_serverless_stack import load_guard
    cfg = _load_json(config_path)
    if (cfg.get('schema') != 'ieee_tc_dlora_native_worker_v1'
            or _sha(config_path) != os.environ.get('FAASLORA_TC_DLORA_WORKER_CONFIG_SHA256')
            or Path(sys.executable).resolve() != Path(cfg['native_python']).resolve()
            or cfg.get('generation_contract') != 'fixed_length_greedy_v1'
            or cfg.get('migration_type') != 3
            or any(_sha(path) != digest for path, digest in cfg['source_sha256'].items())):
        raise ValueError('dLoRA frozen native interpreter/config/source differs')
    dlora_private_ray_options(cfg)
    sys.path.insert(0, cfg['main_repo'])
    guard = load_guard(Path(cfg['main_repo']))
    admission = guard.verify_current_service()
    path = Path(cfg['allocation_journal'])
    records = [json.loads(line) for line in path.read_text().splitlines()]
    acquire = records[0]
    service = Path(admission['service_identity']['path'])
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (acquire['event'] != 'acquire' or acquire['lease_id'] != cfg['allocation_lease_id']
            or any(r['event'] == 'release' for r in records)
            or not Path(acquire['owner_identity']['cgroup']).is_relative_to(service)
            or guard.gpu_process_identity(acquire['owner_identity']['pid']) != acquire['owner_identity']
            or (visible not in ('0', '1') if actor else visible != '0,1')
            or len(acquire['gpu_uuids']) != 2):
        raise ValueError('native dLoRA process has no live pre-initialization physical pool')
    return cfg, guard


def verify_dlora_source_view(cfg):
    root = Path(cfg['source_view'])
    path = root/'measurement_manifest.json'
    if _sha(path) != cfg['source_view_manifest_sha256']:
        raise ValueError('dLoRA source view manifest changed')
    manifest = _load_json(path)
    if (manifest.get('schema') != 'ieee_tc_dlora_measurement_view_v1'
            or manifest.get('input_contract') != 'canonical_token_ids_v1'
            or manifest.get('worker_lifecycle_scope') != 'DP2_TP1_private_Ray_pool_v1'
            or manifest.get('worker_lifecycle') is not True):
        raise ValueError('dLoRA source view lacks input or native lifecycle adaptation')
    for relative, digest in manifest['original_sha256'].items():
        if _sha(Path(manifest['native_source'])/relative) != digest:
            raise ValueError('retained dLoRA source changed: '+relative)
    for relative, digest in manifest['measured_sha256'].items():
        if _sha(root/relative) != digest:
            raise ValueError('measured dLoRA source changed: '+relative)
    if _sha(manifest['helper_path']) != manifest['helper_sha256']:
        raise ValueError('dLoRA measurement helper changed')
    return manifest


@asynccontextmanager
async def dlora_native_runtime(config_path):
    """Own the existing native EngineManager and its new private Ray lifetime.

    Called only by an already-bound GPU-pool child. Startup includes native
    NCCL and original LoRA placement; teardown closes that same local Ray node,
    never `ray stop`. The external owner still verifies actual exits/NVML before
    releasing its allocation, including when construction fails.
    """
    cfg, guard = dlora_worker_context(config_path)
    verify_dlora_source_view(cfg)
    root = Path(cfg['ray_root'])
    root.mkdir(mode=0o700, exist_ok=False)
    (root/'spill').mkdir()
    import ray
    if ray.__version__ != '2.51.2' or ray.is_initialized():
        raise ValueError('dLoRA requires the retained native Ray version and no inherited connection')
    manager = None
    try:
        ray.init(**dlora_private_ray_options(cfg))
        nodes = [node for node in ray.nodes() if node['Alive']]
        if (len(nodes) != 1 or nodes[0]['Resources'].get('GPU') != 2
                or nodes[0]['Resources'].get('CPU') != 40
                or nodes[0]['Resources'].get('object_store_memory') != 8*1024**3):
            raise ValueError('actual private Ray resources differ from the shared envelope')
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.engine_manager import EngineManager, MigrationType
        from vllm.config import ExecType
        from faaslora.clock import local_monotonic_clock_id
        if not Path(inspect.getfile(EngineManager)).absolute().is_relative_to(Path(cfg['source_view'])):
            raise ValueError('dLoRA imported a different native source view')
        args = AsyncEngineArgs(**cfg['engine_args'])
        if (args.num_groups != 2 or args.tensor_parallel_size != 1 or args.pipeline_parallel_size != 1
                or args.use_dummy_weights or not args.worker_use_ray or not args.engine_use_ray
                or args.exec_type != ExecType.LORA.value or args.num_models != len(cfg['adapters'])):
            raise ValueError('native dLoRA DP2/TP1 real-weight arguments differ')
        manager = EngineManager(ExecType.LORA, MigrationType.PERIOD_MIG, 2, args.num_models, args)
        from ieee_tc_serverless_measurement import dlora_native_adapter_bindings
        if {a:list(v) for a,v in dlora_native_adapter_bindings(manager).items()} != cfg['adapters']:
            raise ValueError('native manager pool differs from published static adapter mapping')
        workers = []
        for engine in manager.engines:
            if len(engine.workers) != 1:
                raise ValueError('retained dLoRA TP1 engine does not have exactly one native worker')
            row = ray.get(engine.workers[0].tc_worker_receipt.remote())
            if (row['identity'] != guard.gpu_process_identity(row['identity']['pid'])
                    or row['clock_id'] != local_monotonic_clock_id()):
                raise ValueError('native worker birth or local clock changed before ready')
            workers.append(dict(row, engine_id=engine.engine_id))
        receipt = dict(schema='ieee_tc_dlora_backend_ready_v1', receipt_id=uuid.uuid4().hex,
            at=time.perf_counter(), clock_id=local_monotonic_clock_id(),
            identity=guard.gpu_process_identity(os.getpid()), workers=workers,
            worker_config_sha256=_sha(config_path), ray_version=ray.__version__,
            ray_nodes=nodes, ray_root=str(root),
            source_view_manifest_sha256=cfg['source_view_manifest_sha256'],
            ready_boundary='native_EngineManager_model_LoRA_placement_and_KV_initialized',
            allocation_lease_id=cfg['allocation_lease_id'],
            native_input_parameters=vars(args),
            numerical_correctness_qualified=False)
        yield manager, receipt
    finally:
        if manager is not None:
            tasks = [task for task in [manager.background_loop,
                *(engine.background_loop for engine in manager.engines)] if task is not None]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        # ray.init(address='local') creates a head owned by this driver. The
        # retained Ray shutdown kills its own processes even after partial init.
        ray.shutdown()


class NativeDloraRequestHandler:
    """Common input into native EngineManager, not an alternative router.

    The service launcher must supply owned Ray/lifecycle evidence separately.
    This request adapter alone is not a qualified service or a performance run.
    """
    def __init__(self, *, manager, sampling_factory, model, vocab_size, max_model_len, clock_id,
                 service_context=None):
        from ieee_tc_serverless_measurement import dlora_native_adapter_bindings
        self.manager, self.sampling_factory = manager, sampling_factory
        self.model, self.vocab_size, self.max_model_len = model, vocab_size, max_model_len
        self.clock_id = clock_id
        self.adapters = dlora_native_adapter_bindings(manager)
        self.seen = set()
        self.service_context = service_context
        if service_context is not None:
            if (set(service_context['preparation']) != set(self.adapters)
                    or set(service_context['workers']) != {engine.engine_id for engine in manager.engines}):
                raise ValueError('dLoRA service context does not cover actual pool/engines')

    async def generate(self, body):
        from ieee_tc_serverless_measurement import (dlora_prepare_canonical_request,
            dlora_validate_final_output, dlora_abort_current_owner)
        entry = time.perf_counter()
        request = dlora_prepare_canonical_request(body, model=self.model,
            adapters=self.adapters, vocab_size=self.vocab_size, max_model_len=self.max_model_len)
        rid = request['request_id']
        control = dict(body.get('_sllm_internal_metrics', {}))
        if control.get('tc_clock_id') != self.clock_id or rid in self.seen:
            raise ValueError('dLoRA request clock differs or request ID was already offered')
        params = self.sampling_factory(**request['sampling_kwargs'])
        self.seen.add(rid)
        async with self.manager.select_lock:
            selection_started = time.perf_counter()
            engine, generator = await self.manager.select_engine(rid, request['prompt'], params,
                request['model_id'], prompt_token_ids=list(request['prompt_token_ids']))
            dispatched = time.perf_counter()
        final, valid = None, False
        try:
            async for output in generator:
                final = output
            completed = time.perf_counter()
            if final is None:
                raise ValueError('dLoRA returned no native terminal output')
            measured = dlora_validate_final_output(final, request_id=rid,
                model_id=request['model_id'], prompt_ids=request['prompt_token_ids'],
                target=request['target_tokens'], dispatched_at=dispatched,
                completed_at=completed, clock_id=self.clock_id)
            control.update(tc_manager_entry_s=entry, tc_backend_entry_s=dispatched,
                tc_instance_assigned_s=dispatched, instance_id=f'dlora-{engine.engine_id}',
                tc_native_selection_started_s=selection_started,
                tc_native_submitted_s=dispatched, tc_native_selected_engine_id=engine.engine_id,
                tc_canonical_prompt_sha256=request['canonical_prompt_sha256'])
            if self.service_context is not None:
                context = self.service_context
                prep = context['preparation'][request['adapter_id']]
                if (prep['tc_artifact_request_id'] != 'dlora-full-pool:'+request['adapter_id']
                        or prep['tc_artifact_cache_hit'] is not False
                        or not context['deployment_notice_s'] <= prep['tc_artifact_resolve_started_s']
                        <= prep['tc_artifact_resolve_completed_s'] <= context['receipt']['tc_worker_ready_s'] <= entry):
                    raise ValueError('dLoRA preparation/ready does not precede native selection')
                # A request can migrate. Keep every observed engine's actual GPU
                # worker binding; a single final worker would erase this scope.
                workers = [context['workers'][i] for i in dict.fromkeys(measured['native_engine_path'])]
                measured['native_engine_workers'] = workers
                control.update(prep, **context['receipt'], tc_native_engine_workers=workers,
                    tc_artifact_preparation_mode='native_full_pool_before_service_v1',
                    tc_deployment_notice_s=context['deployment_notice_s'])
            measured.update(native_request_id=rid, native_adapter_id=request['adapter_id'],
                native_adapter_path=request['adapter_path'], native_adapter_index=request['model_id'],
                control_observation=control)
            valid = True
            return dict(id=rid, model=self.model, choices=[dict(index=0, finish_reason='length',
                message=dict(role='assistant', content=final.outputs[0].text))],
                usage=dict(prompt_tokens=len(request['prompt_token_ids']),
                    completion_tokens=request['target_tokens'],
                    total_tokens=len(request['prompt_token_ids'])+request['target_tokens']),
                metrics=dict(ieee_tc=measured))
        finally:
            # The original generate catches Exception, not task cancellation;
            # its captured engine can also be stale after a native migration.
            try:
                if not valid:
                    await dlora_abort_current_owner(self.manager, rid)
            finally:
                await generator.aclose()


async def _native_dlora_worker(config_path, index):
    """Existing TC HTTP endpoint over native EngineManager and its private Ray."""
    if index != 0:
        raise ValueError('one dLoRA manager owns both native engines')
    cfg, _ = dlora_worker_context(config_path)
    from aiohttp import web
    stopped, runner = asyncio.Event(), None
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stopped.set)
    async with dlora_native_runtime(config_path) as (manager, row):
        row.update(index=index, preparation_sha256=_sha(Path(cfg['output'])/'dlora_full_pool_preparation.json'))
        receipt_path = Path(cfg['output'])/'worker_receipts'/(row['receipt_id']+'.json')
        _exclusive_json(receipt_path, row)
        ready = dict(tc_worker_receipt_id=row['receipt_id'], tc_worker_receipt_sha256=_sha(receipt_path),
            tc_worker_ready_s=row['at'], port=cfg['worker_ports'][0])
        from vllm.sampling_params import SamplingParams
        workers = {w['engine_id']: dict(engine_id=w['engine_id'], pid=w['identity']['pid'],
            start_ticks=w['identity']['start_ticks'], clock_id=w['clock_id']) for w in row['workers']}
        handler = NativeDloraRequestHandler(manager=manager, sampling_factory=SamplingParams,
            model=cfg['model'], vocab_size=row['workers'][0]['vocab_size'],
            max_model_len=cfg['request_max_model_len'], clock_id=row['clock_id'],
            service_context=dict(preparation=_load_json(Path(cfg['output'])/'dlora_full_pool_preparation.json'),
                workers=workers, receipt=ready, deployment_notice_s=cfg['deployment_notice_s']))
        async def generate(request):
            return web.json_response(await handler.generate(await request.json()))
        async def health(request):
            return web.json_response(dict(ready=True, receipt_id=row['receipt_id'], identity=row['identity']))
        try:
            app = web.Application()
            app.router.add_post('/v1/chat/completions', generate)
            app.router.add_get('/health', health)
            runner = web.AppRunner(app, access_log=None, handler_cancellation=True, shutdown_timeout=1.)
            await runner.setup()
            await web.TCPSite(runner, '127.0.0.1', cfg['worker_ports'][0]).start()
            _exclusive_json(Path(cfg['output'])/'worker_0.ready.json', ready)
            await stopped.wait()
        finally:
            if runner is not None:
                await runner.cleanup()


def validate_dlora_ready_receipt(receipt, *, cfg, identity, worker_identity, allocation, clock_id,
                                 preparation_sha256):
    if (receipt.get('schema') != 'ieee_tc_dlora_backend_ready_v1'
            or receipt.get('identity') != identity or receipt.get('index') != 0
            or receipt.get('clock_id') != clock_id
            or receipt.get('allocation_lease_id') != allocation.owner_id
            or receipt.get('source_view_manifest_sha256') != cfg['source_view_manifest_sha256']
            or receipt.get('preparation_sha256') != preparation_sha256
            or receipt.get('ready_boundary') != 'native_EngineManager_model_LoRA_placement_and_KV_initialized'
            or len(receipt.get('workers', [])) != 2):
        raise ValueError('dLoRA actual manager/pool preparation differs from ready receipt')
    expected = {i: allocation.gpu_uuids[i] for i in range(2)}
    workers = receipt['workers']
    if {w['engine_id'] for w in workers} != {0, 1} or len({w['identity']['pid'] for w in workers}) != 2:
        raise ValueError('dLoRA must have two distinct native TP1 workers')
    bindings = []
    for w in workers:
        visible = w.get('cuda_visible')
        if (visible not in ('0', '1') or w['identity'] != worker_identity(w['identity']['pid'])
                or w['identity']['pid'] == identity['pid'] or w['clock_id'] != clock_id
                or w.get('adapter_bindings') != cfg['adapters']
                or w.get('model') != cfg['engine_args']['model']
                or w.get('tensor_parallel_size') != 1 or w.get('pipeline_parallel_size') != 1
                or w.get('cuda_device_count') != 1 or w.get('num_gpu_blocks', 0) <= 0
                or w.get('actual_gpu_capacity') != len(w.get('gpu_lora_models', []))):
            raise ValueError('dLoRA ready worker differs from its native binding/resource evidence')
        bindings.append(dict(device_uuid=expected[int(visible)], pid=w['identity']['pid'], worker_rank=w['engine_id']))
    if {w['device_uuid'] for w in bindings} != set(allocation.gpu_uuids):
        raise ValueError('dLoRA ready workers do not cover both allocated GPUs')
    return bindings


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


class LoquetierStaticRouter:
    """Explicit deployment mapping, not an adaptive/native cluster scheduler."""

    def __init__(self, adapter_to_worker, worker_count, clock_id):
        if (type(worker_count) is not int or worker_count < 1 or not adapter_to_worker
                or any(not isinstance(a, str) or not a or type(i) is not int
                       or not 0 <= i < worker_count for a, i in adapter_to_worker.items())
                or set(adapter_to_worker.values()) != set(range(worker_count))):
            raise ValueError('complete nonempty static shards required')
        self.mapping = dict(adapter_to_worker)
        self.ready = [asyncio.Event() for _ in range(worker_count)]
        self.clock_id, self.seen = clock_id, set()

    async def route(self, body, received):
        rid, aid = body.get('request_id'), body.get('lora_adapter_name')
        control = dict(body.get('_sllm_internal_metrics', {}))
        if (not isinstance(rid, str) or not rid or rid in self.seen or aid not in self.mapping
                or control.get('tc_clock_id') != self.clock_id):
            raise ValueError('unknown/duplicate request or static adapter/clock mismatch')
        self.seen.add(rid)
        index = self.mapping[aid]
        control.update(tc_http_received_s=received, tc_router_enqueued_s=time.perf_counter(),
            ready_instances_at_enqueue=sum(r.is_set() for r in self.ready),
            tc_static_shard_index=index)
        await self.ready[index].wait()  # No rerouting to a ready but wrong shard.
        control.update(tc_instance_assigned_s=time.perf_counter(), instance_id=f'loquetier-{index}')
        return index, dict(body, _sllm_internal_metrics=control)


def validate_loquetier_input(request_id, adapter_id, prompt_ids, target, adapter_ids,
                             max_prompt_tokens=None):
    if (not isinstance(request_id, str) or not request_id or adapter_id not in adapter_ids
            or type(prompt_ids) is not list or not prompt_ids
            or any(type(i) is not int or i < 0 for i in prompt_ids)
            or type(target) is not int or not 1 <= target <= 256
            or len(prompt_ids)+target > 1024
            or (max_prompt_tokens is not None and len(prompt_ids) > max_prompt_tokens)):
        raise ValueError('Loquetier fixed native input/static adapter differs')


class LoquetierNativeBatchBridge:
    """Single-thread owner of original ModelGeneration; no replacement kernels.

    The factory supplies an already guarded native generator and CPU request
    construction. The approved historical batch bound stays explicit. No future
    trace, queue priority, dynamic placement or performance configuration search.
    """
    def __init__(self, generator, request_factory, batch_size, *, max_prompt_tokens=None):
        from ieee_tc_serverless_measurement import _slora_process_identity
        if (type(batch_size) is not int or batch_size <= 0
                or not getattr(generator, '_tc_loquetier_registry', None)
                or generator.requests_len != 0
                or (max_prompt_tokens is not None and
                    (type(max_prompt_tokens) is not int or not 1 <= max_prompt_tokens < 1024))):
            raise ValueError('empty verified native generator and explicit batch bound required')
        self.generator, self.factory, self.batch_size = generator, request_factory, batch_size
        self.max_prompt_tokens = max_prompt_tokens
        self.adapters = frozenset(generator._tc_loquetier_registry)
        self.process = _slora_process_identity(os.getpid())
        self.thread = threading.get_ident()
        self.objects, self.next_mark = {}, 0

    def _owner(self):
        if threading.get_ident() != self.thread:
            raise ValueError('native Loquetier state accessed outside its serial owner')

    def add(self, rid, aid, ids, target):
        self._owner()
        validate_loquetier_input(rid, aid, ids, target, self.adapters, self.max_prompt_tokens)
        if self.generator.requests_len >= self.batch_size:
            raise ValueError('native batch capacity exceeded')
        mark = self.next_mark
        self.next_mark += 1
        request = self.factory(mark, rid, aid, list(ids), target)
        self.generator.add_requests([request])
        self.objects[mark] = request
        return mark

    def step(self):
        self._owner()
        from ieee_tc_serverless_measurement import loquetier_request_result, loquetier_clear_released_decode_cache
        self.generator.generate()
        results = {}
        for mark in sorted(self.generator.finished_mark_ids):
            if mark not in self.objects:
                raise ValueError('native terminal belongs to an unknown online request')
            results[mark] = loquetier_request_result(self.generator, mark,
                time.perf_counter(), self.process)
            del self.objects[mark]
        loquetier_clear_released_decode_cache(self.generator)
        return results

    def cancel(self, mark):
        self._owner()
        from ieee_tc_serverless_measurement import loquetier_cancel_request
        if mark not in self.objects:
            raise ValueError('cannot cancel an unknown native request')
        result = loquetier_cancel_request(self.generator, mark)
        del self.objects[mark]
        return result

    def close(self):
        self._owner()
        return [self.cancel(mark) for mark in list(self.objects)]


class LoquetierOnlineRequests:
    """FIFO arrival bridge, with all native mutations on one owned thread.

    This adapts the official offline workflow to online arrivals. Each iteration
    admits waiting requests up to the existing batch bound and invokes native
    generate once. HTTP cancellation is processed between native iterations;
    there is no concurrent cache mutation or hidden replacement request.
    """
    def __init__(self, bridge_factory, emit):
        self.factory, self.emit = bridge_factory, emit
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='loquetier-native')
        self.pending, self.active, self.seen = deque(), {}, set()
        self.wake = asyncio.Event()
        self.bridge = self.task = None
        self.closing = False
        self.failure = None

    async def _native(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)

    async def start(self):
        if self.bridge is not None or self.closing:
            raise ValueError('online native bridge already started/closed')
        try:
            self.bridge = await self._native(self.factory)
        except BaseException:
            self.closing = True
            self.executor.shutdown(wait=True)
            raise
        self.task = asyncio.create_task(self._pump())
        return self.bridge

    async def submit(self, rid, aid, ids, target):
        if self.bridge is None or self.closing or self.failure is not None:
            raise RuntimeError('native online service is not available')
        validate_loquetier_input(rid, aid, ids, target, self.bridge.adapters,
                                getattr(self.bridge, 'max_prompt_tokens', None))
        if rid in self.seen:
            raise ValueError('duplicate online request; no hidden retry')
        self.seen.add(rid)
        future = asyncio.get_running_loop().create_future()
        self.pending.append((rid, aid, list(ids), target, future))
        self.emit(dict(event='loquetier_online_enqueued', request_id=rid, adapter_id=aid,
                       at=time.perf_counter()))
        self.wake.set()
        try:
            return await future
        finally:
            self.wake.set()  # Also wakes an idle owner after cancellation.

    async def _pump(self):
        try:
            while not self.closing:
                while self.pending and len(self.active) < self.bridge.batch_size:
                    item = self.pending.popleft()
                    rid, aid, ids, target, future = item
                    if future.cancelled():
                        self.emit(dict(event='loquetier_request_cancelled', request_id=rid,
                            adapter_id=aid, native_registered=False, native_output_tokens=0,
                            native_terminal_observed=False, at=time.perf_counter()))
                        continue
                    # Keep the item visible to exception cleanup even if native
                    # registration fails (e.g. a real allocation failure).
                    try:
                        mark = await self._native(self.bridge.add, rid, aid, ids, target)
                    except BaseException as exc:
                        self.emit(dict(event='loquetier_registration_failed', request_id=rid,
                            error=f'{type(exc).__name__}: {exc}', at=time.perf_counter()))
                        if not future.done():
                            future.set_exception(exc)
                        raise
                    self.active[mark] = item
                for mark, item in list(self.active.items()):
                    if item[-1].cancelled():
                        self.emit(await self._native(self.bridge.cancel, mark))
                        del self.active[mark]
                if self.active:
                    results = await self._native(self.bridge.step)
                    for mark, result in results.items():
                        rid, _, _, _, future = self.active.pop(mark)
                        self.emit(dict(event='loquetier_native_completed', request_id=rid,
                            client_cancelled=future.cancelled(), observation=result))
                        if not future.cancelled():
                            future.set_result(result)
                elif not self.pending:
                    self.wake.clear()
                    await self.wake.wait()
        except Exception as exc:
            self.failure = exc
            self.emit(dict(event='loquetier_native_failure', error=f'{type(exc).__name__}: {exc}',
                           at=time.perf_counter()))
        finally:
            try:
                for record in await self._native(self.bridge.close):
                    self.emit(record)
            except Exception as exc:
                self.failure = self.failure or exc
                self.emit(dict(event='loquetier_cleanup_failure', error=f'{type(exc).__name__}: {exc}',
                               at=time.perf_counter()))
            error = RuntimeError('native online service stopped; no fallback or retry')
            for item in list(self.active.values())+list(self.pending):
                if not item[-1].done():
                    item[-1].set_exception(error)
            self.active.clear()
            self.pending.clear()

    async def close(self):
        self.closing = True
        self.wake.set()
        if self.task is not None:
            await self.task  # Finish a running iteration; do not mutate concurrently.
        self.executor.shutdown(wait=True)


class NativeLoquetierRequestHandler:
    """Common HTTP body to the real native online queue; no new routing policy."""
    def __init__(self, *, queue, adapters, preparation, clock_id, shard_index,
                 receipt, worker_process, model, deployment_notice_s):
        if (type(shard_index) is not int or shard_index < 0 or not adapters
                or set(adapters) != set(preparation)
                or set(adapters) != set(queue.bridge.adapters)
                or {i for i, _ in adapters.values()} != set(range(len(adapters)))
                or any(type(i) is not int or not Path(p).is_absolute() or Path(p).name != aid
                       for aid, (i, p) in adapters.items())
                or worker_process.get('clock_id') != clock_id
                or any(type(worker_process.get(k)) is not int or worker_process[k] <= 0
                       for k in ('pid', 'start_ticks'))
                or any(not isinstance(receipt.get(k), str) or len(receipt[k]) != n
                       or set(receipt[k])-set('0123456789abcdef') for k, n in (
                           ('tc_worker_receipt_id', 32), ('tc_worker_receipt_sha256', 64)))
                or type(receipt.get('tc_worker_ready_s')) not in (float, int)
                or not math.isfinite(receipt['tc_worker_ready_s'])):
            raise ValueError('explicit Loquetier shard/preparation/native worker identity required')
        self.queue, self.adapters, self.preparation = queue, dict(adapters), preparation
        self.clock_id, self.index = clock_id, shard_index
        self.receipt, self.process = dict(receipt), dict(worker_process)
        self.model, self.notice = model, deployment_notice_s

    async def generate(self, body):
        entry = time.perf_counter()
        ids, target, aid = body.get('input_tokens'), body.get('max_tokens'), body.get('lora_adapter_name')
        prompt, rid = body.get('canonical_prompt'), body.get('request_id')
        validate_loquetier_input(rid, aid, ids, target, self.adapters)
        if (body.get('model') != self.model or not isinstance(prompt, str) or not prompt
                or body.get('temperature') != 0 or body.get('top_p') != 1
                or body.get('ignore_eos') is not True or body.get('stream') is not False
                or body.get('stop') != [] or body.get('stop_token_ids') != []):
            raise ValueError('request differs from Loquetier fixed-output/prompt contract')
        control = dict(body.get('_sllm_internal_metrics', {}))
        if (control.get('tc_clock_id') != self.clock_id
                or control.get('instance_id') != f'loquetier-{self.index}'
                or type(control.get('tc_static_shard_index')) is not int
                or control['tc_static_shard_index'] != self.index):
            raise ValueError('Loquetier selected static shard/clock differs')
        index, path = self.adapters[aid]
        prep = self.preparation[aid]
        start, end = prep.get('tc_artifact_resolve_started_s'), prep.get('tc_artifact_resolve_completed_s')
        if (prep.get('tc_artifact_id') != aid or prep.get('tc_artifact_cache_hit') is not False
                or prep.get('tc_artifact_request_id') != 'loquetier-full-pool:'+aid
                or any(type(t) not in (int, float) or not math.isfinite(t)
                       for t in (self.notice, start, end))
                or not self.notice <= start <= end <= self.receipt['tc_worker_ready_s'] <= entry):
            raise ValueError('Loquetier full-pool preparation must follow notice and precede ready')
        control.update(prep, **self.receipt, tc_backend_entry_s=entry,
            tc_artifact_preparation_mode='native_full_pool_before_service_v1',
            tc_deployment_notice_s=self.notice, tc_worker_native_process=dict(self.process),
            tc_canonical_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        control['tc_native_submitted_s'] = time.perf_counter()
        measured = await self.queue.submit(rid, aid, ids, target)
        if (measured.get('native_process') != self.process
                or measured.get('native_request_id') != rid or measured.get('native_adapter_id') != aid
                or measured.get('native_adapter_path') != path or measured.get('native_adapter_index') != index
                or type(measured.get('native_adapter_index')) is not int
                or measured.get('native_prompt_token_ids') != ids
                or type(measured.get('native_output_tokens')) is not int
                or measured.get('native_output_tokens') != target
                or type(measured.get('native_forward_count')) is not int
                or measured.get('native_forward_count') != target
                or measured.get('native_terminal_observed') is not True
                or measured.get('timing_contract') != 'ieee_tc_loquetier_native_v1'):
            raise ValueError('native Loquetier result differs from ready worker/input/adapter binding')
        measured = dict(measured, control_observation=control)
        return dict(id=rid, model=self.model, choices=[dict(index=0, finish_reason='length')],
            usage=dict(prompt_tokens=len(ids), completion_tokens=target, total_tokens=len(ids)+target),
            metrics=dict(ieee_tc=measured))


LOQUETIER_ENV_KEYS = ('PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'PYTHONNOUSERSITE',
    'MAX_JOBS', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE')


def loquetier_capacity_evidence(cfg, plan):
    """Opt-in measured envelope; legacy placement/configuration stays unchanged.

    A largest-shard diagnostic is a capacity witness, not an all-ID numerical
    oracle or performance qualification. No batch sweep or OOM fallback here.
    """
    policy = cfg.get('inference_capacity')
    if policy is None:
        if cfg['native_args']['batch_size'] != plan['batch_size']:
            raise ValueError('legacy Loquetier batch differs from static plan')
        return None
    if (not isinstance(policy, dict) or set(policy) != {'policy', 'evidence', 'evidence_sha256'}
            or policy['policy'] != 'prepared_only_observed_capacity_v1'
            or not isinstance(policy['evidence'], str) or not Path(policy['evidence']).is_absolute()
            or _sha(policy['evidence']) != policy['evidence_sha256']):
        raise ValueError('explicit SHA-bound Loquetier inference capacity policy required')
    proof = _load_json(Path(policy['evidence']))
    forward, model = proof.get('forward_capacity', {}), proof.get('capacity_model', {})
    largest = max(plan['shards'], key=lambda s: (s['weights_and_kv_accounted_bytes'], -s['index']))
    if (proof.get('schema') != 'ieee_tc_loquetier_native_capacity_probe_v1'
            or proof.get('passed') is not True or proof.get('phase') != 'weights_and_kv_ready'
            or proof.get('performance_run') is not False or forward.get('passed') is not True
            or proof.get('static_plan_sha256') != cfg['static_plan_sha256']
            or proof.get('measurement_manifest_sha256') != cfg['source_view_manifest_sha256']
            or proof.get('native_kernel_sha256') != cfg['native_kernel_sha256']
            or plan.get('base_config_sha256') != _sha(Path(plan['base_model'])/'config.json')
            or proof.get('shard') != largest or proof.get('loaded_adapter_count') != largest['adapter_count']
            or proof.get('verified_adapter_count') != largest['adapter_count']
            or model.get('schema') != 'loquetier_observed_capacity_bound_v1'
            or forward.get('capacity_model') != model
            or model.get('candidate_batch_size') != cfg['native_args']['batch_size']
            or forward.get('batch_size') != model.get('candidate_batch_size')
            or model.get('context_tokens') != cfg['native_args']['context_tokens']):
        raise ValueError('Loquetier measured capacity/source/placement identity differs')
    integer_keys = ('candidate_batch_size', 'context_tokens', 'prompt_tokens', 'kv_page_bytes',
        'kv_page_tokens', 'device_total_bytes', 'warm_live_bytes', 'per_request_kv_bytes',
        'per_request_scratch_bytes', 'per_request_largest_allocation_headroom_bytes',
        'per_request_accounted_bytes', 'available_budget_bytes', 'predicted_total_bytes')
    if (any(type(model.get(k)) is not int or model[k] <= 0 for k in integer_keys)
            or any(type(model.get(k)) is not int or model[k] < 0 for k in
                   ('observed_nonallocator_bytes', 'observed_fragmentation_bytes'))):
        raise ValueError('invalid measured capacity byte/geometry accounting')
    b = model['candidate_batch_size']
    per = sum(model[k] for k in ('per_request_kv_bytes', 'per_request_scratch_bytes',
                                'per_request_largest_allocation_headroom_bytes'))
    budget = model['device_total_bytes'] - sum(model[k] for k in
        ('warm_live_bytes', 'observed_nonallocator_bytes', 'observed_fragmentation_bytes'))
    if (per != model['per_request_accounted_bytes'] or budget != model['available_budget_bytes']
            or b != budget//per or b <= plan['batch_size']
            or model['predicted_total_bytes'] != model['device_total_bytes']-budget+b*per
            or model['prompt_tokens']+256 > model['context_tokens']
            or any(s[k] > largest[k] for s in plan['shards'] for k in
                   ('adapter_count', 'native_max_rank', 'peft_original_fp32_bytes',
                    'prepared_fp16_bytes', 'pointer_bytes', 'backbone_fp16_bytes'))):
        raise ValueError('Loquetier batch is not the measured bounded capacity')
    release = forward.get('release', {})
    if (release.get('remaining_owners') != 0 or release.get('remaining_parameters') != 0
            or release.get('allocated_reduction_bytes') != largest['peft_original_fp32_bytes']
            or release.get('raw_storage_bytes') != largest['peft_original_fp32_bytes']):
        raise ValueError('measured raw-owner release missing')
    inputs = forward.get('input', {}).get('cases', [])
    if (len(inputs) != b or len({c['source_request_id'] for c in inputs}) != b
            or any(len(c.get('prompt_ids', [])) != model['prompt_tokens']
                   or c.get('target_tokens') != 256 for c in inputs)):
        raise ValueError('measured capacity input geometry incomplete')
    for phase in ('all_prefill', 'mixed_prefill_decode'):
        cases = [c for c in forward.get('cases', []) if c.get('selection') == phase]
        if len(cases) != 1:
            raise ValueError('complete native all/mixed capacity cases required')
        case = cases[0]
        samples, outputs = case.get('samples', []), case.get('outputs', [])
        if (case.get('batch_size') != b or len(outputs) != b or not samples
                or samples[-1].get('live_kv_pages') != 0
                or len({r['native_adapter_id'] for r in outputs}) != b
                or any(r.get('native_output_tokens') != 256 or len(r.get('completion_token_ids', [])) != 256
                       for r in outputs)
                or not any(s.get('native_prefill_requests') == (b if phase == 'all_prefill' else b//2)
                           and s.get('native_decode_requests') == (0 if phase == 'all_prefill' else b-b//2)
                           for s in samples)):
            raise ValueError('native capacity execution/geometry incomplete')
    return model


def loquetier_capacity_device_guard(model, *, allocated, reserved, free, total,
                                    kv_page_bytes, kv_page_tokens):
    """Reuse the diagnostic's byte budget on the actual worker, before ready."""
    values = (allocated, reserved, free, total, kv_page_bytes, kv_page_tokens)
    outside = total-free-reserved
    slack = model['available_budget_bytes'] - model['candidate_batch_size']*model['per_request_accounted_bytes']
    if (any(type(v) is not int or v < 0 for v in values) or allocated > reserved
            or free > total or outside < 0 or total < model['device_total_bytes']
            or allocated > model['warm_live_bytes']
            or outside > model['observed_nonallocator_bytes']+slack
            or kv_page_bytes != model['kv_page_bytes'] or kv_page_tokens != model['kv_page_tokens']):
        raise ValueError('actual worker exceeds measured Loquetier capacity envelope')
    return dict(allocated_bytes=allocated, reserved_bytes=reserved, device_free_bytes=free,
        device_total_bytes=total, nonallocator_bytes=outside,
        kv_page_bytes=kv_page_bytes, kv_page_tokens=kv_page_tokens)


def load_loquetier_static_plan(cfg, pool_ids=None):
    """Reuse the frozen input-only placement, never reconstruct from a trace."""
    path = Path(cfg['static_plan'])
    if _sha(path) != cfg['static_plan_sha256']:
        raise ValueError('Loquetier static placement identity changed')
    plan = _load_json(path)
    shards = plan.get('shards', [])
    ids = [aid for shard in shards for aid in shard.get('adapter_ids', [])]
    if (plan.get('schema') != 'ieee_tc_loquetier_static_capacity_v1'
            or plan.get('no_trace_read') is not True or plan.get('adapter_count') != 500
            or plan.get('sharding_rule') != 'descending_static_bytes_then_ID_least_bytes_count_index_v1'
            or plan.get('context_tokens') != cfg['native_args']['context_tokens']
            or plan.get('base_model') != cfg['native_args']['model_dir']
            or len(shards) != len(cfg['gpu_ids']) or len(ids) != 500 or len(set(ids)) != 500
            or any(type(s.get('index')) is not int or s['index'] != i
                   or not s.get('adapter_ids') for i, s in enumerate(shards))
            or any(not isinstance(a, str) or not a or Path(a).name != a for a in ids)
            or (pool_ids is not None and set(ids) != set(pool_ids))):
        raise ValueError('Loquetier complete static shards differ from frozen input-only plan')
    loquetier_capacity_evidence(cfg, plan)
    return plan


def validate_loquetier_service_config(cfg, http):
    from urllib.parse import urlsplit
    from prepare_ieee_tc_serverless_stack import validate_http_lifecycle_contract
    validate_http_lifecycle_contract(http)
    endpoint = urlsplit(http['url'])
    devices = cfg.get('gpu_ids', [])
    ports = [endpoint.port, cfg.get('router_port'), *cfg.get('worker_ports', [])]
    plan = load_loquetier_static_plan(cfg)
    capacity = loquetier_capacity_evidence(cfg, plan)
    required = dict(model_dir=http['backbone'], dtype='float16',
                    batch_size=capacity['candidate_batch_size'] if capacity else 2,
                    context_tokens=1024, apply_scaling='A')
    if (cfg.get('schema') != 'ieee_tc_loquetier_native_service_v1'
            or http.get('schema') != 'ieee_tc_loquetier_http_replay_v1'
            or cfg.get('routing') != 'static_shards_v1'
            or cfg.get('worker_startup') != 'sequential_v1'
            or cfg.get('generation_contract') != 'fixed_length_greedy_v1'
            or type(devices) is not list or not devices or len(set(devices)) != len(devices)
            or any(type(d) is not int or d not in range(4) for d in devices)
            or len(cfg.get('worker_ports', [])) != len(devices)
            or any(type(p) is not int or not 0 < p < 65536 for p in ports)
            or len(set(ports)) != len(ports)
            or endpoint.scheme != 'http' or endpoint.hostname != '127.0.0.1'
            or endpoint.path != '/v1/chat/completions' or endpoint.query or endpoint.fragment
            or endpoint.username or endpoint.password or not http.get('remote_artifacts')
            or cfg.get('native_args') != required
            or any(type(cfg['native_args'].get(k)) is not type(v) for k,v in required.items())):
        raise ValueError('explicit Loquetier native/static-shard/sequential/loopback contract required')
    for key in ('python', 'native_python', 'source_view', 'static_plan', 'native_kernel'):
        if not isinstance(cfg.get(key), str) or not Path(cfg[key]).is_absolute():
            raise ValueError('explicit Loquetier interpreter/source/placement/kernel paths required')
    if (plan['pool_index'] != http['pool_index']
            or plan['pool_index_sha256'] != http['pool_index_sha256']):
        raise ValueError('Loquetier placement and common remote pool differ')
    return endpoint.port


def verify_loquetier_source_view(cfg):
    root = Path(cfg['source_view'])
    path = root/'measurement_manifest.json'
    if _sha(path) != cfg['source_view_manifest_sha256']:
        raise ValueError('Loquetier source view manifest changed')
    manifest = _load_json(path)
    if (manifest.get('schema') != 'ieee_tc_loquetier_measurement_view_v1'
            or manifest.get('source_view') != str(root.resolve())
            or manifest.get('supported_observation') != 'native_loquetier_inference_fixed_length_greedy_v1'
            or manifest.get('helper_path') != str(SCRIPT_DIR/'ieee_tc_serverless_measurement.py')
            or manifest.get('helper_sha256') != _sha(manifest['helper_path'])
            or manifest.get('source_adapter_sha256') != _sha(SCRIPT_DIR/'prepare_ieee_tc_serverless_stack.py')
            or manifest.get('shared_package') != str((Path(cfg['main_repo'])/'faaslora').resolve())
            or _sha(cfg['native_kernel']) != cfg['native_kernel_sha256']):
        raise ValueError('Loquetier source view/helper/native kernel identity differs')
    for key, base in (('original_sha256', Path(manifest['native_source'])/'src/loquetier_src'),
            ('measured_sha256', root/'loquetier_src'), ('shared_source_sha256', Path(cfg['main_repo']))):
        for name,digest in manifest[key].items():
            if _sha(base/name) != digest:
                raise ValueError('Loquetier guarded source changed: '+name)
    return manifest


def loquetier_worker_environment(cfg, device, environ=None):
    python = Path(cfg['native_python'])
    root = python.parent.parent
    torchlib = root/'lib/python3.9/site-packages/torch/lib'
    if (not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK)
            or not torchlib.is_dir()):
        raise ValueError('existing Loquetier native Python3.9/torch environment required')
    env = dict(os.environ if environ is None else environ)
    env.pop('PYTHONHOME', None)
    env.update(PATH=str(python.parent)+os.pathsep+env.get('PATH', ''),
        LD_LIBRARY_PATH=str(torchlib)+os.pathsep+str(root/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH', ''),
        PYTHONPATH=cfg['source_view'], PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
        PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES=str(device), MAX_JOBS='2',
        TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    return env


def loquetier_worker_context(config_path, index):
    """No torch/native imports before source, process and containment checks."""
    from prepare_ieee_tc_serverless_stack import load_guard
    cfg = _load_json(config_path)
    expected = {str(SCRIPT_DIR/n) for n in ('replay_vllm_engine_trace.py',
        'ieee_tc_serverless_measurement.py', 'prepare_ieee_tc_serverless_stack.py')}
    if (cfg.get('schema') != 'ieee_tc_loquetier_native_service_v1'
            or type(index) is not int or not 0 <= index < len(cfg['gpu_ids'])
            or set(cfg.get('source_sha256', {})) != expected
            or any(_sha(p) != h for p,h in cfg['source_sha256'].items())
            or os.environ.get('FAASLORA_TC_LOQUETIER_WORKER_CONFIG_SHA256') != _sha(config_path)
            or Path(sys.executable).resolve() != Path(cfg['native_python']).resolve()
            or sys.version_info[:2] != (3, 9)
            or os.environ.get('CUDA_VISIBLE_DEVICES') != str(cfg['gpu_ids'][index])
            or set(cfg.get('native_environment', {})) != set(LOQUETIER_ENV_KEYS)
            or any(os.environ.get(k) != v for k,v in cfg['native_environment'].items())):
        raise ValueError('Loquetier native process/source/config identity differs')
    verify_loquetier_source_view(cfg)
    plan = load_loquetier_static_plan(cfg, cfg['adapters'])
    root = Path(cfg['remote_config']['cache_dir'])/'objects'
    if (len(cfg['adapters']) != 500 or not root.is_absolute()
            or any(Path(p) != root/aid for aid,(_,p) in cfg['adapters'].items())
            or cfg['remote_config']['content_manifest'] != plan['pool_index']
            or cfg['remote_config']['content_manifest_file_sha256'] != plan['pool_index_sha256']):
        raise ValueError('Loquetier workers must use this run common remote objects')
    sys.path.insert(0, cfg['main_repo'])
    guard = load_guard(Path(cfg['main_repo']))
    guard.verify_current_service()
    prep_path = Path(cfg['output'])/'loquetier_full_pool_preparation.json'
    prepared = _load_json(prep_path)
    notice = cfg['deployment_notice_s']
    if (set(prepared) != set(cfg['adapters']) or type(notice) not in (int,float)
            or not math.isfinite(notice)):
        raise ValueError('Loquetier complete after-notice preparation required')
    for aid,row in prepared.items():
        start,end = row.get('tc_artifact_resolve_started_s'), row.get('tc_artifact_resolve_completed_s')
        if (row.get('tc_artifact_id') != aid or row.get('tc_artifact_request_id') != 'loquetier-full-pool:'+aid
                or row.get('tc_artifact_cache_hit') is not False
                or any(type(t) not in (float,int) or not math.isfinite(t) for t in (start,end))
                or not notice <= start <= end <= time.perf_counter()):
            raise ValueError('Loquetier preparation predates common notice or has wrong identity')
    return cfg, guard, plan, prepared


def _loquetier_layout(base, mixed):
    """Primitive-only live storage/pointer inventory; no weight copies or owners."""
    def prepared():
        for name, adapters in sorted(mixed.target_module_saves.items()):
            layer = mixed.model.get_submodule(name)
            if (layer.wa_ptrs.detach().cpu().tolist() != [a.wa.data_ptr() for a in adapters]
                    or layer.wb_ptrs.detach().cpu().tolist() != [a.wb.data_ptr() for a in adapters]):
                raise ValueError('Loquetier prepared SMLM pointer table differs')
            yield name+'.wa_ptrs', layer.wa_ptrs
            yield name+'.wb_ptrs', layer.wb_ptrs
            for adapter in adapters:
                yield name+'.'+adapter.adapter_name+'.A', adapter.wa
                yield name+'.'+adapter.adapter_name+'.B', adapter.wb
    def inventory(items):
        rows, storages = [], {}
        for name, tensor in items:
            storage = tensor.untyped_storage()
            pointer = storage.data_ptr()
            storages[pointer] = storage.nbytes()
            rows.append((name, str(tensor.dtype), str(tensor.device), tuple(tensor.shape),
                         tuple(tensor.stride()), tensor.data_ptr(), pointer, storage.nbytes()))
        if not rows:
            raise ValueError('Loquetier retained tensor inventory missing')
        return rows, storages
    return inventory(base.named_parameters()), inventory(prepared()), dict(mixed.lora_name_to_id)


def release_loquetier_inference_owners(models, base, mixed):
    """Drop only obsolete PEFT wrappers after native preparation, once at deploy.

    No cpu()/to(), parameter deletion, precision change, or per-request GC.
    Aliased prepared storage remains owned by native SMLM; weak Parameter death
    alone is not a freed-byte claim. Actual CUDA accounting is checked by caller.
    """
    import gc
    import weakref
    if type(models) is not list or not models or mixed.training_loras:
        raise ValueError('only prepared pure-inference Loquetier owners may be released')
    before = _loquetier_layout(base, mixed)
    def inventory():
        owners, parameters, storages = [], [], {}
        for model in models:
            owners.extend(weakref.ref(x) for x in (model, model.base_model, model.base_model.model))
            for name, tensor in model.named_parameters():
                if '.lora_' in name:
                    parameters.append(weakref.ref(tensor))
                    storage = tensor.untyped_storage()
                    storages[storage.data_ptr()] = storage.nbytes()
        return owners, parameters, storages
    owners, parameters, raw = inventory()
    if not owners or not parameters or not raw:
        raise ValueError('raw Loquetier owner/parameter inventory missing')
    retained = set(before[0][1]) | set(before[1][1])
    releasable = sum(size for ptr, size in raw.items() if ptr not in retained)
    started = time.perf_counter()
    models.clear()
    collected = gc.collect()  # Original virtual wrappers contain bound-method cycles.
    remaining_owners = sum(ref() is not None for ref in owners)
    remaining_parameters = sum(ref() is not None for ref in parameters)
    if remaining_owners or remaining_parameters or before != _loquetier_layout(base, mixed):
        raise ValueError('Loquetier raw owners remain or retained layout/registry changed')
    return dict(policy='prepared_only_observed_capacity_v1', collected_objects=collected,
        seconds=time.perf_counter()-started, raw_storage_bytes=sum(raw.values()),
        releasable_storage_bytes=releasable, aliased_storage_bytes=sum(raw.values())-releasable,
        owner_count=len(owners), parameter_count=len(parameters), remaining_owners=remaining_owners,
        remaining_parameters=remaining_parameters, retained_layout_unchanged=True,
        runtime_content_hash_checked=False)


def _load_loquetier_native_bridge(cfg, index, plan, emit):
    """Exact previously capacity-tested native load sequence; single owner thread."""
    from prepare_ieee_tc_serverless_stack import load_guard
    load_guard(Path(cfg['main_repo'])).verify_current_service()
    capacity = loquetier_capacity_evidence(cfg, plan)
    import torch
    from transformers import LlamaConfig, GenerationConfig, LogitsProcessorList
    from loquetier_src import LoquetierFramework, create_virtual_model
    from loquetier_src.models.mixed_llama import MixedLlamaForCausalLM
    from loquetier_src.models.mixed_lora import MixedLoraModel
    from loquetier_src.model_generation import ModelGeneration, InputRequest, InferStatus
    from loquetier_src.tc_measurement import loquetier_bind_static_registry
    from loquetier.ops import _kernels
    if (torch.cuda.device_count() != 1 or str(Path(_kernels.__file__).resolve()) !=
            str(Path(cfg['native_kernel']).resolve()) or _sha(_kernels.__file__) != cfg['native_kernel_sha256']
            or Path(inspect.getfile(ModelGeneration)).resolve() !=
                Path(cfg['source_view'])/'loquetier_src/model_generation.py'):
        raise ValueError('actual Loquetier native imports or visible device differ')
    torch.cuda.set_device(0)
    device, dtype = torch.device('cuda:0'), torch.float16
    aids = plan['shards'][index]['adapter_ids']
    paths = {aid: cfg['adapters'][aid][1] for aid in aids}
    basepath = cfg['native_args']['model_dir']
    emit(dict(event='loquetier_backbone_loading', index=index, at=time.perf_counter()))
    base = MixedLlamaForCausalLM.from_pretrained(basepath,
        low_cpu_mem_usage=True, torch_dtype=dtype).to(device)
    LoquetierFramework.BaseModelList[basepath] = base
    models = [LoquetierFramework.load_model(paths[aid], use_fast=False, torch_dtype=dtype) for aid in aids]
    mixed = MixedLoraModel(create_virtual_model(base),
        {aid: model.base_model for aid,model in zip(aids,models)}, dtype=dtype, device=device, apply_scaling='A')
    generation = GenerationConfig.from_pretrained(basepath)
    generation.do_sample, generation.num_beams = False, 1
    generation.eos_token_id, generation.max_length = None, cfg['native_args']['context_tokens']
    generator = ModelGeneration(mixed, LlamaConfig.from_pretrained(basepath), generation, device, dtype=dtype)
    registry = loquetier_bind_static_registry(generator, paths)
    if generator._tc_loquetier_registry != {aid:i for i,aid in enumerate(aids)}:
        raise ValueError('native Loquetier adapter index differs from frozen shard order')
    def request_factory(mark, rid, aid, ids, target):
        request = InputRequest(status=InferStatus.Prefill, input_ids=torch.tensor(ids, dtype=torch.long),
            next_input_ids=list(ids), mark_id=mark, lora_name=aid,
            logits_processor=LogitsProcessorList(), max_length=len(ids)+target)
        request._tc_request_id = rid
        return request
    capacity_receipt = None
    if capacity is not None:
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated(0)
        release = release_loquetier_inference_owners(models, base, mixed)
        torch.cuda.synchronize()
        release['allocated_reduction_bytes'] = allocated_before-torch.cuda.memory_allocated(0)
        if (release['allocated_reduction_bytes'] < release['releasable_storage_bytes']
                or release['raw_storage_bytes'] != plan['shards'][index]['peft_original_fp32_bytes']):
            raise ValueError('actual Loquetier raw-owner allocation release differs')
        free, total = torch.cuda.mem_get_info(0)
        device_guard = loquetier_capacity_device_guard(capacity,
            allocated=torch.cuda.memory_allocated(0), reserved=torch.cuda.memory_reserved(0),
            free=free, total=total, kv_page_bytes=generator.kvpool.page_nbytes,
            kv_page_tokens=generator.kvpool.page_len)
        capacity_receipt = dict(binding=cfg['inference_capacity'], model=capacity,
            release=release, device_guard=device_guard)
        emit(dict(event='loquetier_inference_owner_release', index=index,
                  at=time.perf_counter(), capacity=capacity_receipt))
    bridge = LoquetierNativeBatchBridge(generator, request_factory, cfg['native_args']['batch_size'],
        max_prompt_tokens=capacity['prompt_tokens'] if capacity else None)
    if capacity is None:
        bridge.native_models = models  # Legacy behavior and execution identity preserved.
    bridge.startup = dict(registry=registry, adapter_ids=dict(generator._tc_loquetier_registry),
        adapter_paths=paths, cuda_device_count=torch.cuda.device_count(),
        model_dir=basepath, allocated_bytes=torch.cuda.memory_allocated(0),
        reserved_bytes=torch.cuda.memory_reserved(0), native_kernel_sha256=_sha(_kernels.__file__))
    if capacity_receipt is not None:
        bridge.startup['inference_capacity'] = capacity_receipt
    return bridge


async def _native_loquetier_worker(config_path, index):
    cfg, guard, plan, prepared = loquetier_worker_context(config_path, index)
    from aiohttp import web
    from faaslora.clock import local_monotonic_clock_id
    runner = queue = waiter = None
    stopped = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stopped.set)
    # Journal retained even for initialization/forward failure; no hidden retry.
    with (Path(cfg['output'])/f'worker_{index}.native_events.jsonl').open('x') as journal:
        def emit(row):
            journal.write(json.dumps(row, allow_nan=False)+'\n')
            journal.flush()
        try:
            queue = LoquetierOnlineRequests(lambda: _load_loquetier_native_bridge(cfg,index,plan,emit), emit)
            bridge = await queue.start()
            receipt_id = uuid.uuid4().hex
            row = dict(schema='ieee_tc_loquetier_backend_ready_v1', receipt_id=receipt_id,
                at=time.perf_counter(), clock_id=local_monotonic_clock_id(), index=index,
                identity=guard.gpu_process_identity(os.getpid()), cuda_visible=os.environ['CUDA_VISIBLE_DEVICES'],
                native_environment=cfg['native_environment'], native_args=cfg['native_args'],
                worker_config_sha256=_sha(config_path), runner_sha256=_sha(__file__),
                source_view_manifest_sha256=cfg['source_view_manifest_sha256'],
                static_plan_sha256=cfg['static_plan_sha256'],
                preparation_sha256=_sha(Path(cfg['output'])/'loquetier_full_pool_preparation.json'),
                ready_boundary='native_static_shard_and_model_initialized', **bridge.startup)
            receipt_path = Path(cfg['output'])/'worker_receipts'/(receipt_id+'.json')
            _exclusive_json(receipt_path,row)
            ready = dict(tc_worker_receipt_id=receipt_id, tc_worker_receipt_sha256=_sha(receipt_path),
                         tc_worker_ready_s=row['at'], port=cfg['worker_ports'][index])
            adapters = {aid:(i,cfg['adapters'][aid][1]) for i,aid in enumerate(plan['shards'][index]['adapter_ids'])}
            handler = NativeLoquetierRequestHandler(queue=queue, adapters=adapters,
                preparation={aid:prepared[aid] for aid in adapters}, clock_id=row['clock_id'], shard_index=index,
                receipt=ready, worker_process=bridge.process, model=cfg['model'],
                deployment_notice_s=cfg['deployment_notice_s'])
            async def generate(request):
                return web.json_response(await handler.generate(await request.json()))
            async def health(request):
                return web.json_response(dict(ready=queue.failure is None and not queue.closing,
                    receipt_id=receipt_id, process=bridge.process), status=200 if queue.failure is None else 503)
            app = web.Application()
            app.router.add_post('/v1/chat/completions',generate)
            app.router.add_get('/health',health)
            runner = web.AppRunner(app,access_log=None,handler_cancellation=True,shutdown_timeout=1.)
            await runner.setup()
            await web.TCPSite(runner,'127.0.0.1',cfg['worker_ports'][index]).start()
            _exclusive_json(Path(cfg['output'])/f'worker_{index}.ready.json',ready)
            waiter = asyncio.create_task(stopped.wait())
            await asyncio.wait((waiter,queue.task),return_when=asyncio.FIRST_COMPLETED)
            if queue.task.done():
                raise RuntimeError('native Loquetier queue stopped; worker must exit') from queue.failure
        except BaseException as exc:
            emit(dict(event='loquetier_worker_failed',error=f'{type(exc).__name__}: {exc}',at=time.perf_counter()))
            raise
        finally:
            if waiter is not None:
                waiter.cancel()
                await asyncio.gather(waiter,return_exceptions=True)
            if runner is not None:
                await runner.cleanup()
            if queue is not None:
                await queue.close()
            emit(dict(event='loquetier_worker_exiting',at=time.perf_counter(),
                      gpu_release_proven=False))  # Parent proves process exit + native GPU census.


def validate_loquetier_ready_receipt(receipt, *, cfg, index, plan, identity,
                                     preparation_sha256):
    aids = plan['shards'][index]['adapter_ids']
    if (receipt.get('schema') != 'ieee_tc_loquetier_backend_ready_v1'
            or receipt.get('index') != index or receipt.get('identity') != identity
            or receipt.get('adapter_ids') != {aid:i for i,aid in enumerate(aids)}
            or receipt.get('adapter_paths') != {aid:cfg['adapters'][aid][1] for aid in aids}
            or receipt.get('source_view_manifest_sha256') != cfg['source_view_manifest_sha256']
            or receipt.get('static_plan_sha256') != cfg['static_plan_sha256']
            or receipt.get('preparation_sha256') != preparation_sha256
            or receipt.get('native_environment') != cfg['native_environment']
            or receipt.get('native_args') != cfg['native_args']
            or receipt.get('native_kernel_sha256') != cfg['native_kernel_sha256']
            or receipt.get('model_dir') != cfg['native_args']['model_dir']
            or receipt.get('cuda_visible') != str(cfg['gpu_ids'][index])
            or receipt.get('cuda_device_count') != 1
            or receipt.get('runner_sha256') != _sha(__file__)
            or receipt.get('ready_boundary') != 'native_static_shard_and_model_initialized'
            or receipt.get('registry', {}).get('adapter_count') != len(aids)
            or receipt.get('registry', {}).get('pointer_tables_verified') is not True
            or any(type(receipt.get(k)) is not int or receipt[k] <= 0
                   for k in ('allocated_bytes','reserved_bytes'))):
        raise ValueError('actual Loquetier worker/static registry/preparation differs from ready receipt')
    model = loquetier_capacity_evidence(cfg, plan) if cfg.get('inference_capacity') is not None else None
    if model is not None:
        capacity = receipt.get('inference_capacity', {})
        release = capacity.get('release', {})
        guard = capacity.get('device_guard', {})
        if (capacity.get('binding') != cfg['inference_capacity'] or capacity.get('model') != model
                or release.get('policy') != cfg['inference_capacity']['policy']
                or release.get('remaining_owners') != 0 or release.get('remaining_parameters') != 0
                or release.get('retained_layout_unchanged') is not True
                or release.get('raw_storage_bytes') != plan['shards'][index]['peft_original_fp32_bytes']
                or type(release.get('releasable_storage_bytes')) is not int
                or type(release.get('allocated_reduction_bytes')) is not int
                or not 0 < release['releasable_storage_bytes'] <= release['raw_storage_bytes']
                or release.get('aliased_storage_bytes') != release['raw_storage_bytes']-release['releasable_storage_bytes']
                or release['allocated_reduction_bytes'] < release['releasable_storage_bytes']
                or guard.get('allocated_bytes') != receipt['allocated_bytes']
                or guard.get('reserved_bytes') != receipt['reserved_bytes']):
            raise ValueError('Loquetier actual owner release/capacity receipt missing or inconsistent')
        checked = loquetier_capacity_device_guard(model, allocated=guard['allocated_bytes'],
            reserved=guard['reserved_bytes'], free=guard['device_free_bytes'], total=guard['device_total_bytes'],
            kv_page_bytes=guard['kv_page_bytes'], kv_page_tokens=guard['kv_page_tokens'])
        if checked != guard:
            raise ValueError('Loquetier capacity receipt byte accounting differs')
    elif 'inference_capacity' in receipt:
        raise ValueError('legacy Loquetier receipt unexpectedly changed ownership policy')


class NativeSloraRequestHandler:
    """Shared HTTP contract -> native generate_stream, not a new scheduler.

    All-pool preparation is a startup action after deployment notice, not an
    artificial request-time fetch. Native token events and the real ready
    worker identity must agree. HTTP receive/EOF are diagnostics, not TTFT/TPOT.
    The owning service supplies its already-verified preparation/ready records.
    """
    def __init__(self, *, session, endpoint, adapters, preparation, clock_id,
                 instance_id, receipt, worker_process, model, deployment_notice_s,
                 input_mode='native_text_v1'):
        from urllib.parse import urlsplit
        url = urlsplit(endpoint)
        if (input_mode not in ('native_text_v1', 'canonical_token_ids_v1')
                or url.scheme != 'http' or url.hostname != '127.0.0.1'
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
        self.input_mode = input_mode

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
        if self.input_mode == 'canonical_token_ids_v1':
            native_body['tc_input_contract'] = dict(mode=self.input_mode, token_ids=list(ids),
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                token_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest())
            control['tc_slora_input_mode'] = self.input_mode
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
    if cfg.get('native_compiler_contract') is not None:
        from scripts.ieee_tc_native_toolchain import bind_native_toolchain
        binding = cfg['native_compiler_contract']
        if os.environ.get('FAASLORA_TC_NATIVE_TOOLCHAIN_BOUND_SHA256') != binding['sha256']:
            raise ValueError('native worker must inherit its qualified compiler environment')
        _, actual = bind_native_toolchain(os.environ, python=sys.executable,
            contract=binding['path'], expected_sha=binding['sha256'])
        if actual != toolchain.get('compiler_contract'):
            raise ValueError('native worker compiler receipt differs')
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
        prepare_remote_artifacts, record_http_physical_terminals, prepare_slora_full_pool)
    from ieee_tc_serverless_measurement import PreReadyHTTPIngress, PublishedArtifactResolver
    import aiohttp
    from aiohttp import web
    cfg, http = _load_json(config_path), _load_json(http_path)
    ingress_port = validate_native_service_config(cfg, http)
    slora = cfg['schema'] == 'ieee_tc_slora_native_service_v1'
    loquetier = cfg['schema'] == 'ieee_tc_loquetier_native_service_v1'
    dlora = cfg['schema'] == 'ieee_tc_dlora_native_service_v1'
    static_native = slora or loquetier or dlora
    backend = 'dlora' if dlora else 'loquetier' if loquetier else 'slora' if slora else 'vllm'
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
    if static_native:
        verify_view = verify_dlora_source_view if dlora else verify_loquetier_source_view if loquetier else verify_slora_source_view
        verify_view(dict(cfg, main_repo=http['main_repo']))
        toolchain = dict(native_python=cfg['native_python'],
            source_view_manifest_sha256=cfg['source_view_manifest_sha256'], startup=cfg['worker_startup'])
        if dlora:
            worker_environments = [(dlora_worker_environment(dict(cfg, main_repo=http['main_repo'])), toolchain)]
        else:
            make_env = loquetier_worker_environment if loquetier else slora_worker_environment
            worker_environments = [(make_env(cfg, device), toolchain) for device in cfg['gpu_ids']]
    else:
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
        remote_config=remote, adapters={aid: [i if slora or dlora else i+1, paths[aid]] for i, aid in enumerate(sorted(paths))})
    if dlora:
        # This is a small index over the original published pool, not new weights
        # or a second dataset. All IDs use the same native subset loader/order.
        subset = output/'native_adapter_subset.json'
        _exclusive_json(subset, dict(adapters=[dict(id=aid) for aid in sorted(paths)]))
        worker_cfg.update(schema='ieee_tc_dlora_native_worker_v1',
            deployment_notice_s=context['deployment_notice_s'],
            engine_args=dict(cfg['engine_args'], lora_adapter_subset=str(subset),
                lora_adapter_root=str(Path(next(iter(paths.values()))).parent), lora_adapter_limit=500))
    if loquetier:
        shard_plan = load_loquetier_static_plan(cfg, paths)
        worker_cfg.update(deployment_notice_s=context['deployment_notice_s'],
            native_environment={k:worker_environments[0][0][k] for k in LOQUETIER_ENV_KEYS})
    if slora:
        worker_cfg.update(deployment_notice_s=context['deployment_notice_s'],
            native_environment={k: worker_environments[0][0][k] for k in SLORA_ENV_KEYS},
            native_input_params={k: cfg['native_args'][k] for k in SLORA_INPUT_KEYS})
    worker_path = output/'worker_config.json'
    if not dlora:  # dLoRA config also binds the acquire record written before spawn.
        _exclusive_json(worker_path, worker_cfg)
    if dlora:
        router = NativeManagerIngress([e.request_id for e in plan.entries], local_monotonic_clock_id())
    elif loquetier:
        router = LoquetierStaticRouter({aid:s['index'] for s in shard_plan['shards'] for aid in s['adapter_ids']},
                                      len(cfg['gpu_ids']),local_monotonic_clock_id())
    else:
        router = ResidentRoundRobin([e.request_id for e in plan.entries], len(cfg['gpu_ids']),
                                    local_monotonic_clock_id(), instance_prefix=backend)
    result = dict(schema=f'ieee_tc_{backend}_native_model_qualification_v1', passed=False,
        qualification_only=True, numerical_correctness_qualified=False,
        configuration=cfg, configuration_sha256=_sha(config_path),
        http_config_sha256=_sha(http_path), source_sha256=sources,
        service=admission, requests=[])
    owners, processes, logs, starts = [], [], [], []
    slora_handlers, slora_preparation = {}, {}
    app_runner = None
    failure = None
    stop = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)

    async def await_ready(index, owner, process):
        path = output/(f'worker_{index}.native_ready.json' if slora else f'worker_{index}.ready.json')
        deadline = context['deployment_notice_s']+1800.
        while not path.exists():
            if process.poll() is not None:
                raise RuntimeError(f'{backend} worker {index} exited before ready: {process.returncode}')
            if stop.is_set() or time.perf_counter() >= deadline:
                raise TimeoutError('native startup interrupted or qualification protection elapsed')
            await asyncio.sleep(.05)
        ready = _load_json(path)
        receipt = _load_json(output/'worker_receipts'/(ready['tc_worker_receipt_id']+'.json'))
        if (ready['tc_worker_receipt_sha256'] != _sha(output/'worker_receipts'/(ready['tc_worker_receipt_id']+'.json'))
                or receipt['index'] != index
                or receipt['worker_config_sha256'] != _sha(worker_path)
                or receipt['clock_id'] != router.clock_id):
            raise ValueError('native ready receipt/source/process identity changed')
        if slora:
            if (receipt.get('schema') != 'ieee_tc_slora_backend_ready_v1'
                    or receipt.get('api_parent_identity') != guard.gpu_process_identity(process.pid)
                    or receipt['identity'] != guard.gpu_process_identity(receipt['identity']['pid'])
                    or receipt['identity']['pid'] == process.pid
                    or receipt['adapter_ids'] != {p: i for i, p in worker_cfg['adapters'].values()}
                    or receipt.get('adapter_count') != 500
                    or receipt.get('source_view_manifest_sha256') != cfg['source_view_manifest_sha256']
                    or receipt.get('native_input_params') != worker_cfg['native_input_params']
                    or receipt.get('native_environment') != worker_cfg['native_environment']
                    or receipt.get('host_layout') != cfg['host_layout']
                    or receipt.get('cuda_visible') != str(cfg['gpu_ids'][index])
                    or receipt.get('cuda_device_count') != 1 or receipt.get('tp') != 1
                    or receipt.get('model_dir') != cfg['native_args']['model_dir']
                    or receipt.get('runner_sha256') != sources[str(Path(__file__).resolve())]
                    or receipt.get('rpc_path') != str((Path(cfg['source_view'])/'slora/server/router/model_infer/model_rpc.py').resolve())
                    or receipt.get('rpc_sha256') != _sha(receipt['rpc_path'])
                    or receipt.get('ready_boundary') != 'native_full_pool_and_model_initialized'):
                raise ValueError('actual S-LoRA GPU child/preparation differs from ready receipt')
            while True:
                if process.poll() is not None or stop.is_set() or time.perf_counter() >= deadline:
                    raise RuntimeError('S-LoRA API did not become ready with its native worker')
                try:
                    async with session.get(f'http://127.0.0.1:{cfg["worker_ports"][index]}/health',
                            timeout=aiohttp.ClientTimeout(total=1.), allow_redirects=False) as response:
                        if response.status != 200:
                            raise ValueError('native S-LoRA health returned an error')
                        await response.read()
                        break
                except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
                    await asyncio.sleep(.05)
        elif dlora:
            dlora_bindings = validate_dlora_ready_receipt(receipt, cfg=worker_cfg,
                identity=guard.gpu_process_identity(process.pid), worker_identity=guard.gpu_process_identity,
                allocation=owner, clock_id=router.clock_id,
                preparation_sha256=_sha(output/'dlora_full_pool_preparation.json'))
            async with session.get(f'http://127.0.0.1:{cfg["worker_ports"][0]}/health',
                    timeout=aiohttp.ClientTimeout(total=1.), allow_redirects=False) as response:
                health = await response.json()
                if (response.status != 200 or health.get('ready') is not True
                        or health.get('receipt_id') != receipt['receipt_id']
                        or health.get('identity') != receipt['identity']):
                    raise ValueError('dLoRA native HTTP endpoint does not match its live manager')
        elif loquetier:
            validate_loquetier_ready_receipt(receipt,cfg=worker_cfg,index=index,plan=shard_plan,
                identity=guard.gpu_process_identity(process.pid),
                preparation_sha256=_sha(output/'loquetier_full_pool_preparation.json'))
            async with session.get(f'http://127.0.0.1:{cfg["worker_ports"][index]}/health',
                    timeout=aiohttp.ClientTimeout(total=1.),allow_redirects=False) as response:
                health = await response.json()
                expected_process = dict(pid=receipt['identity']['pid'],
                    start_ticks=receipt['identity']['start_ticks'],clock_id=receipt['clock_id'])
                if (response.status != 200 or health.get('ready') is not True
                        or health.get('receipt_id') != ready['tc_worker_receipt_id']
                        or health.get('process') != expected_process):
                    raise ValueError('Loquetier HTTP health does not match the native ready worker')
        elif receipt['identity'] != guard.gpu_process_identity(process.pid):
            raise ValueError('native vLLM API process differs from receipt')
        # Notification and HTTP health alone are not GPU proof. Bind live
        # contexts to this exact allocation before releasing queued requests.
        sample = owner.census.sample(service_path)
        workers = [dict(device_uuid=d['gpu_uuid'], pid=p['pid'], worker_rank=index)
            for d in sample['devices'] if d['gpu_uuid'] in owner.gpu_uuids
            for p in d['processes'] if p['service_member'] and p['kind'] == 'compute']
        if dlora:
            if {(w['device_uuid'], w['pid']) for w in workers} != {(w['device_uuid'], w['pid']) for w in dlora_bindings}:
                raise ValueError('dLoRA native NVML census differs from the two actual worker receipts')
            workers = dlora_bindings
        elif static_native and (len(workers) != 1 or workers[0]['pid'] != receipt['identity']['pid']):
            raise ValueError('static native census is not the receipted single-GPU process')
        owner.confirm_workers(workers)
        if slora:
            native_process = dict(pid=receipt['identity']['pid'],
                start_ticks=receipt['identity']['start_ticks'], clock_id=receipt['clock_id'])
            slora_handlers[index] = NativeSloraRequestHandler(session=session,
                endpoint=f'http://127.0.0.1:{cfg["worker_ports"][index]}/generate_stream',
                adapters=worker_cfg['adapters'], preparation=slora_preparation,
                clock_id=router.clock_id, instance_id=f'slora-{index}',
                receipt=dict(ready, tc_worker_ready_s=receipt['at']), worker_process=native_process,
                model=http['model'], deployment_notice_s=context['deployment_notice_s'],
                input_mode=cfg.get('input_mode', 'native_text_v1'))
            _exclusive_json(output/f'worker_{index}.ready.json',
                dict(ready, port=cfg['worker_ports'][index], api_health_verified_s=time.perf_counter()))
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
                    if slora:
                        return web.json_response(await slora_handlers[index].generate(body))
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
                async def start_workers():
                    if static_native:
                        slora_preparation.update(await prepare_slora_full_pool(
                            PublishedArtifactResolver(remote), paths, context['deployment_notice_s'],
                            **({'backend':backend} if dlora or loquetier else {})))
                        _exclusive_json(output/f'{backend}_full_pool_preparation.json', slora_preparation)
                    allocation_groups = [cfg['gpu_ids']] if dlora else [[d] for d in cfg['gpu_ids']]
                    for index, devices in enumerate(allocation_groups):
                        if stop.is_set():
                            raise RuntimeError('resident startup interrupted before next GPU allocation')
                        api = guard.load_nvml_binding(Path(os.environ['FAASLORA_TC_NVML_BINDING']),
                                                      os.environ['FAASLORA_TC_NVML_SHA256'])
                        census = guard.NativeGPUCensus(api)
                        try:
                            owner = PhysicalGPUAllocation(root=deployment.allocations, census=census,
                                service_path=service_path, device_indices=devices,
                                owner_identity=guard.gpu_process_identity(os.getpid()))
                        except BaseException:
                            census.close()
                            raise
                        owners.append(owner)
                        log = (output/f'worker_{index}.log').open('xb')
                        logs.append(log)
                        env = worker_environments[index][0]
                        if dlora:
                            worker_cfg.update(allocation_journal=str(owner.journal), allocation_lease_id=owner.owner_id)
                            _exclusive_json(worker_path, worker_cfg)
                            env.update(FAASLORA_TC_DLORA_WORKER_CONFIG=str(worker_path),
                                FAASLORA_TC_DLORA_WORKER_CONFIG_SHA256=_sha(worker_path))
                        elif slora:
                            env.update(FAASLORA_TC_SLORA_WORKER_CONFIG=str(worker_path),
                                FAASLORA_TC_SLORA_WORKER_CONFIG_SHA256=_sha(worker_path),
                                FAASLORA_TC_SLORA_WORKER_INDEX=str(index))
                        elif loquetier:
                            env['FAASLORA_TC_LOQUETIER_WORKER_CONFIG_SHA256'] = _sha(worker_path)
                        process = subprocess.Popen([cfg['native_python'] if static_native else cfg['python'],
                            '-B', str(Path(__file__).resolve()),
                            'native-dlora-worker' if dlora else 'native-loquetier-worker' if loquetier else 'native-slora-worker' if slora else 'native-http-worker',
                            '--config', str(worker_path), '--index', str(index)],
                            stdout=log, stderr=subprocess.STDOUT, env=env)
                        processes.append(process)
                        owner.bind_process(process, guard.gpu_process_identity(process.pid))
                        task = asyncio.create_task(await_ready(index, owner, process))
                        starts.append(task)
                        if static_native:
                            await task  # Existing sequential initialization bounds peak preparation memory.
                starts.append(asyncio.create_task(start_workers()))
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
            result['physical_lifecycle'] = dict(mode=('dlora_native_pool_v1' if dlora else 'loquetier_static_shards_v1' if loquetier
                                                    else f'{backend}_resident_pool_v1'),
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
        raise RuntimeError(f'{backend} qualification failed; preserve original evidence') from failure
    return result


def native_http_main(argv):
    parser = argparse.ArgumentParser(description='Opt-in shared TC native HTTP measurement')
    parser.add_argument('mode', choices=('native-http-service', 'native-http-worker', 'native-slora-worker',
                                         'native-loquetier-worker', 'native-dlora-worker'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--index', type=int)
    parser.add_argument('--http-replay-config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--private-root', type=Path)
    args = parser.parse_args(argv)
    if args.mode in ('native-http-worker', 'native-slora-worker', 'native-loquetier-worker', 'native-dlora-worker'):
        if args.index is None:
            parser.error('worker requires --index')
        if args.mode == 'native-dlora-worker':
            asyncio.run(_native_dlora_worker(args.config, args.index))
        elif args.mode == 'native-slora-worker':
            _native_slora_worker(args.config, args.index)
        elif args.mode == 'native-loquetier-worker':
            asyncio.run(_native_loquetier_worker(args.config,args.index))
        else:
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
    if len(sys.argv) > 1 and sys.argv[1] in ('native-http-service', 'native-http-worker', 'native-slora-worker',
                                          'native-loquetier-worker', 'native-dlora-worker'):
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
