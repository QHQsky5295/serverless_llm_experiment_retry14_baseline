#!/usr/bin/env python3
"""Materialize an owned TC view of the existing native launch scripts.

This is a source adapter, not a second service supervisor. The existing TC
gated launcher/watchdog owns process-tree teardown. `prepare` starts nothing;
the generated stack refuses launch outside that admitted service domain.
"""
from __future__ import annotations

import argparse
import asyncio
import ast
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
TMUX_CONFIG = "set-window-option -g remain-on-exit on\n"
SOURCE_SHA = {
    "start_serverlessllm_stack.sh": "9ded784fd6c1990dc9c0994d3ec7744b799cfbff830554ff22b85b8f9ddbb95d",
    "run_serverlessllm_head.sh": "8bb2cbb3e257085183f3cd586e86f0073770d98289f515850c7a84bb488c93ac",
    "run_serverlessllm_worker.sh": "d55ddb27716683e9866a5bd72c06e9f0302b35f5f776872b37fc63da0c720f09",
    "run_serverlessllm_serve.sh": "4b3a232f587f1c00986469f59ab7641b1e69835c45115b70b557e450545a27ec",
    "run_serverlessllm_store.sh": "8f7d23d2023d0e856dd2398265891417c3a8c475af904ac7d096c51fa8d2e07d",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scheduler_queue_source(source: str) -> str:
    """Correct pending-request identity, without changing placement policy.

    The upstream FcfsScheduler already removes the request tuple. Its storage-
    aware sibling instead pops indices captured before earlier removals. Keep
    its sort/score/migration/loop cadence, but commit only the still-live future.
    """
    if sha(source.encode()) != 'b7779e41f451a3328eb73337533d6d9b117aafe6ee6b78f9b28e9b5f5e2b0567':
        raise ValueError('storage-aware scheduler source drift')
    source = replace_once(source,
        '                    logger.info(f"Processing request for model {model_name}")\n',
        '                    loading_request = (request_time, num_gpus, allocation_result)\n'
        '                    async with self.queue_lock:\n'
        '                        queue = self.model_loading_queues.get(model_name, [])\n'
        '                        if loading_request not in queue:\n'
        '                            continue\n'
        '                        if allocation_result.done():\n'
        '                            queue.remove(loading_request)\n'
        '                            continue\n'
        '                    logger.info(f"Processing request for model {model_name}")\n')
    return replace_once(source,
        '                            self.model_loading_queues[model_name].pop(idx)\n',
        '                            queue = self.model_loading_queues.get(model_name, [])\n'
        '                            if loading_request not in queue:\n'
        '                                continue\n'
        '                            queue.remove(loading_request)\n'
        '                            if allocation_result.done():\n'
        '                                continue\n')


def measurement_sources(native_source: Path, variant: str,
                        scheduler_queue: str = 'original') -> dict[str, str]:
    """Instrument identical control/engine boundaries; only router loop differs.

    Returns source text for an exclusive view; the dirty upstream checkout and
    installed environment are never overwritten. The historical staged router
    is accepted only if its already-audited pre-TC hash matches exactly.
    """
    paths = ['sllm/app_lib.py', 'sllm/backends/vllm_backend.py',
             'sllm/routers/roundrobin_router.py']
    original = {p: (native_source / p).read_text() for p in paths}
    if sha(original[paths[1]].encode()) != '994c80ae9c6106a6469f9d8d5889132e5c2032608c4e497141c00127f91cc777':
        raise ValueError('native backend source drift')
    if sha(original[paths[2]].encode()) != '2bcbba0d8a3fdf46354f75c1d8ca5bcd79e505f059fcb296bf30dbff176e8f6e':
        raise ValueError('repaired router source drift')
    if variant == 'original':
        original[paths[2]] = subprocess.check_output(
            ['git', '-C', str(native_source), 'show', ':'+paths[2]], text=True)
        if sha(original[paths[2]].encode()) != '0182bd4862c1c0e1c4bf5d00507704f05d3d5092c267b4670ff31a4c79b18843':
            raise ValueError('historical pre-TC router identity differs')
    elif variant != 'repaired':
        raise ValueError('explicit original/repaired router variant required')
    app, backend, router = (original[p] for p in paths)
    import_line = 'from sllm.backends.tc_measurement import stamp\n'
    app = replace_once(app, 'import time\n', 'import time\n'+import_line)
    app = replace_once(app, '        internal_metrics.setdefault("request_received_at", time.time())',
        '        stamp(internal_metrics, "tc_http_received_s")\n'
        '        internal_metrics.setdefault("request_received_at", time.time())')
    router = replace_once(router, 'import time\n', 'import time\n'+import_line)
    router = replace_once(router, '        enqueue_at = time.time()',
        '        stamp(internal_metrics, "tc_router_enqueued_s")\n        enqueue_at = time.time()')
    router = replace_once(router, '            assigned_at = time.time()',
        '            stamp(internal_metrics, "tc_instance_assigned_s")\n            assigned_at = time.time()')
    backend = replace_once(backend, 'import time\n', 'import time\n'
        'from sllm.backends.tc_measurement import (stamp, install_v1_snapshot, '
        'NativeRequestObservation, PublishedArtifactResolver, record_backend_ready)\n')
    backend = replace_once(backend, '        self.backend_config = backend_config\n',
        '        self.backend_config = backend_config\n'
        '        self._tc_artifact_resolver = (PublishedArtifactResolver(backend_config["tc_remote_artifacts"])\n'
        '            if backend_config.get("tc_remote_artifacts") else None)\n')
    backend = replace_once(backend, '            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)',
        '            if self.backend_config.get("tc_native_measurement", False):\n'
        '                install_v1_snapshot()\n'
        '            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)')
    backend = replace_once(backend, '        internal_metrics["backend_started_at"] = time.time()',
        '        stamp(internal_metrics, "tc_backend_entry_s")\n'
        '        if self.backend_config.get("tc_worker_audit"):\n'
        '            internal_metrics.update(self._tc_ready_receipt)\n'
        '        internal_metrics["backend_started_at"] = time.time()')
    backend = replace_once(backend, '            self.status = BackendStatus.RUNNING\n',
        '            self._tc_ready_receipt = record_backend_ready(self, __file__)\n'
        '            self.status = BackendStatus.RUNNING\n')
    backend = replace_once(backend, '        lora_request = self._build_lora_request(\n',
        '        if self._tc_artifact_resolver is not None:\n'
        '            internal_metrics.update(await self._tc_artifact_resolver.resolve(\n'
        '                lora_adapter_name, lora_adapter_path, request_id))\n'
        '        lora_request = self._build_lora_request(\n')
    backend = replace_once(backend, '        results_generator = self.engine.generate(',
        '        tc_observation = (NativeRequestObservation(lora_request)\n'
        '            if self.backend_config.get("tc_native_measurement", False) else None)\n'
        '        results_generator = self.engine.generate(')
    backend = replace_once(backend, '        async for response_output in results_generator:\n',
        '        async for response_output in results_generator:\n'
        '            if tc_observation is not None:\n'
        '                tc_observation.observe(response_output)\n')
    backend = replace_once(backend, '        response["metrics"] = metrics\n',
        '        if tc_observation is not None:\n'
        '            metrics["ieee_tc"] = tc_observation.finish(internal_metrics)\n'
        '        response["metrics"] = metrics\n')
    result = dict(zip(paths, (app, backend, router)))
    if scheduler_queue == 'request_identity_v1':
        path = 'sllm/schedulers/storage_aware_scheduler.py'
        result[path] = scheduler_queue_source((native_source/path).read_text())
    elif scheduler_queue != 'original':
        raise ValueError('unknown scheduler queue contract')
    return result


def prepare_measurement_view(native_source: Path, output: Path, variant: str,
                             main_repo: Path, scheduler_queue: str = 'original') -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError('preserve previous source view')
    replacements = measurement_sources(native_source, variant, scheduler_queue)
    support = ROOT / 'scripts/ieee_tc_serverless_measurement.py'
    output.mkdir(mode=0o700)
    # Copy only instrumented/repaired source files; unchanged files are symlinks.
    # No model, environment, artifact pool or workload copy.
    for parent, dirs, files in os.walk(native_source / 'sllm'):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        relative = Path(parent).relative_to(native_source)
        dest = output / relative
        dest.mkdir(exist_ok=True)
        for name in files:
            key = str(relative / name)
            if key in replacements:
                (dest / name).write_text(replacements[key])
            elif not name.endswith('.pyc'):
                (dest / name).symlink_to(Path(parent) / name)
    (output / 'sllm/backends/tc_measurement.py').symlink_to(support)
    # Expose the shared package, not its repository-level sitecustomize. That
    # legacy startup hook modifies torch.load and imports the serving stack.
    shared_package = (main_repo / 'faaslora').resolve(strict=True)
    (output / 'faaslora').symlink_to(shared_package, target_is_directory=True)
    result = dict(schema='ieee_tc_serverless_measurement_view_v1', variant=variant,
        native_source=str(native_source), source_view=str(output.resolve()),
        shared_package=str(shared_package), repository_startup_hooks=False,
        original_sha256={p: sha((native_source/p).read_bytes()) for p in replacements},
        measured_sha256={p: sha(text.encode()) for p, text in replacements.items()},
        helper_sha256=sha(support.read_bytes()), helper_path=str(support),
        shared_source_sha256={p: sha((main_repo/p).read_bytes()) for p in
            ('faaslora/storage/http_artifact_store.py', 'faaslora/clock.py')},
        loader_policy_changed=False, routing_policy_changed=variant == 'repaired',
        scheduler_queue=scheduler_queue, scheduler_placement_policy_changed=False,
        source_adapter_sha256=sha(Path(__file__).read_bytes()),
        performance_run_authorized=False)
    with (output / 'measurement_manifest.json').open('x') as handle:
        json.dump(result, handle, indent=2)
    return result


def validate_http_observation(prepared, response, event, completed, clock_id, *, ingress_required=False):
    """Native binding/timing validity, NOT a numerical adapter correctness proof."""
    observed = response.get('metrics', {}).get('ieee_tc', {})
    control = observed.get('control_observation', {})
    ids = observed.get('completion_token_ids')
    target = prepared['target_tokens']
    digest = lambda value: sha(json.dumps(value, separators=(',', ':')).encode())
    if (response.get('error') or response.get('id') != prepared['request_id']
            or response.get('usage', {}).get('completion_tokens') != target
            or response.get('usage', {}).get('prompt_tokens') != len(prepared['input_token_ids'])
            or type(ids) is not list or len(ids) != target
            or any(type(t) is not int or t < 0 for t in ids)
            or observed.get('native_output_tokens') != target
            or observed.get('completion_token_ids_sha256') != digest(ids)
            or observed.get('native_prompt_token_ids') != prepared['input_token_ids']
            or observed.get('native_prompt_token_ids_sha256') != prepared['native_prompt_token_ids_sha256']
            or observed.get('native_lora_name') != prepared['adapter_id']
            or type(observed.get('native_lora_int_id')) is not int or observed['native_lora_int_id'] <= 0
            or observed.get('native_clock_id') != clock_id or control.get('tc_clock_id') != clock_id
            or observed.get('timing_contract') != 'ieee_tc_native_v1'
            or observed.get('native_terminal_observed') is not True
            or not control.get('instance_id')):
        raise ValueError('native generation/adapter/clock observation differs from frozen request')
    times = [event['planned_arrival_s'], event['task_created_s'], event['client_submit_s'],
             control.get('tc_http_received_s'), control.get('tc_router_enqueued_s'),
             control.get('tc_instance_assigned_s'), control.get('tc_backend_entry_s'),
             observed.get('native_dispatch_monotonic_s'), observed.get('native_queued_monotonic_s'),
             observed.get('native_scheduled_monotonic_s'), observed.get('native_first_token_monotonic_s'),
             observed.get('native_last_token_monotonic_s'), observed.get('worker_completed_monotonic_s'), completed]
    if (any(type(t) not in (int, float) or not math.isfinite(t) for t in times)
            or times != sorted(times)):
        raise ValueError('planned/HTTP/router/native/completion times are not ordered')
    ingress = [control.get('tc_ingress_received_s'), control.get('tc_ingress_forwarded_s')]
    ingress_present = any(t is not None for t in ingress)
    if ingress_required or ingress_present:
        if (any(type(t) not in (int, float) or not math.isfinite(t) for t in ingress)
                or not times[2] <= ingress[0] <= ingress[1] <= times[3]):
            raise ValueError('service ingress receive/forward times absent or not ordered')
    if 'artifact_content_sha256' in prepared:
        artifact_times = [control.get('tc_artifact_resolve_started_s'),
                          control.get('tc_artifact_resolve_completed_s')]
        hit, transferred = control.get('tc_artifact_cache_hit'), control.get('tc_artifact_transferred_bytes')
        if (control.get('tc_artifact_request_id') != prepared['request_id']
                or control.get('tc_artifact_id') != prepared['adapter_id']
                or control.get('tc_artifact_content_sha256') != prepared['artifact_content_sha256']
                or control.get('tc_artifact_manifest_sha256') != prepared['artifact_manifest_sha256']
                or control.get('tc_artifact_delivery_mode') != 'prepublished_gzip_v1'
                or not re.fullmatch('[0-9a-f]{64}', str(control.get('tc_artifact_receipt_sha256')))
                or not re.fullmatch('[0-9a-f]{32}', str(control.get('tc_artifact_transfer_id')))
                or type(hit) is not bool or type(transferred) is not int
                or (transferred != 0 if hit else transferred <= 0)
                or any(type(t) not in (int, float) or not math.isfinite(t) for t in artifact_times)
                or not times[6] <= artifact_times[0] <= artifact_times[1] <= times[7]):
            raise ValueError('service remote artifact identity/timing/receipt differs')
    if prepared.get('worker_lifetime_required'):
        if (not re.fullmatch('[0-9a-f]{32}', str(control.get('tc_worker_receipt_id')))
                or not re.fullmatch('[0-9a-f]{64}', str(control.get('tc_worker_receipt_sha256')))):
            raise ValueError('native backend lifetime receipt absent')
    a, e, d, f, last = times[0], times[2], times[5], times[10], times[11]
    components = [e-a, d-e, f-d, last-f, completed-last]
    tpot_ms = (last-f)*1000/(target-1) if target > 1 else None
    if abs(sum(components)-(completed-a))*1000 > 1:
        raise ValueError('E2E decomposition identity failed')
    native_tpot = observed.get('native_tpot_ms')
    if ((target == 1 and native_tpot is not None)
            or (target > 1 and (not isinstance(native_tpot, (int, float))
                               or not math.isfinite(native_tpot) or abs(native_tpot-tpot_ms) > 1))):
        raise ValueError('native TPOT recomputation failed')
    result = dict(protocol_valid=True, lora_numerical_correctness_qualified=False,
                ttft_ms=(f-a)*1000, tpot_ms=tpot_ms, e2e_ms=(completed-a)*1000,
                submit_lag_ms=components[0]*1000, dispatch_wait_after_submit_ms=components[1]*1000,
                service_ttft_ms=components[2]*1000, decode_ms=components[3]*1000,
                completion_notification_ms=components[4]*1000,
                router_queue_ms=(times[5]-times[4])*1000,
                instance_assigned_s=d, ready_instances_at_enqueue=control.get('ready_instances_at_enqueue'))
    if ingress_present:
        # Included in d-e already; this diagnostic subspan is NOT another term
        # in E2E, and arrival/submit times are never reset after startup.
        result['ingress_wait_ms'] = (ingress[1]-ingress[0])*1000
    if 'artifact_content_sha256' in prepared:
        result.update(artifact_resolve_ms=(artifact_times[1]-artifact_times[0])*1000,
                      artifact_cache_hit=hit, artifact_transferred_bytes=transferred)
    return result


def remote_artifact_contract(http_cfg):
    """Shared static identity, no request-time downloads in the replay client."""
    remote = http_cfg.get('remote_artifacts')
    if remote is None:
        return None, None
    if (http_cfg.get('ingress_mode') != 'service_pre_ready_v1'
            or set(remote) != {'mode', 'endpoint', 'token_env', 'timeout_s'}):
        raise ValueError('published remote mode requires explicit startup admission and transport keys')
    config = dict(remote, content_manifest=http_cfg['pool_index'],
                  content_manifest_file_sha256=http_cfg['pool_index_sha256'])
    helper = ROOT/'scripts/ieee_tc_serverless_measurement.py'
    spec = importlib.util.spec_from_file_location('tc_published_artifact_support', helper)
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    return config, support.published_artifact_client(config)


def prepare_remote_artifacts(http_cfg, private_root, service_cgroup):
    config, client = remote_artifact_contract(http_cfg)
    if config is None:
        return None, None
    root = private_root/'remote_artifacts'
    config.update(cache_dir=str(root), service_cgroup=service_cgroup)
    # Fail on reuse, including a stale symlink. Download only touched IDs later.
    root.mkdir(mode=0o700)
    for name in ('objects', 'receipts', 'locks', 'journals'):
        (root/name).mkdir(mode=0o700)
    with (root/'run_contract.json').open('x') as handle:
        json.dump(config, handle, sort_keys=True)
    ids = [row['id'] for row in json.loads(Path(http_cfg['pool_index']).read_text())['artifacts']]
    descriptions = client.preparation_manifests(ids)
    return config, {aid: str(root/'objects'/aid) for aid in descriptions}


async def replay_http_session(plan, origin, prepared, url, emit, *, ingress_required=False):
    import aiohttp
    from faaslora.datasets.workload_generator import replay_frozen_http
    trace = aiohttp.TraceConfig()

    async def headers_sent(session, context, params):
        event = context.trace_request_ctx
        event['client_submit_s'] = time.perf_counter()
        emit(dict(event='http_headers_sent', **event))

    async def queued(session, context, params):
        emit(dict(event='http_connection_queued', request_id=context.trace_request_ctx['request_id'],
                  timestamp_s=time.perf_counter()))

    trace.on_request_headers_sent.append(headers_sent)
    trace.on_connection_queued_start.append(queued)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
            timeout=aiohttp.ClientTimeout(total=None), trace_configs=[trace], trust_env=False) as session:
        async def send(row, event):
            event['http_task_started_s'] = time.perf_counter()
            async with session.post(url, json=row['body'], trace_request_ctx=event,
                                    allow_redirects=False) as response:
                raw = await response.read()
                completed = time.perf_counter()
                try:
                    body = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    body = dict(non_json_response=raw[:4096].decode(errors='replace'))
                # Preserve rejected responses too, before any validator raises.
                emit(dict(event='http_raw_response', request_id=row['request_id'],
                          status=response.status, body=body, client_completed_s=completed))
                if response.status != 200:
                    raise ValueError(f'HTTP status {response.status}')
                return validate_http_observation(row, body, event, completed, origin['clock_id'],
                                                 ingress_required=ingress_required)
        return await replay_frozen_http(plan, origin, prepared, send, emit)


def qualified_http_population(cfg):
    """Coverage is a named functional view, never an ordinary W0 prefix."""
    count = cfg.get('request_count')
    view = cfg.get('request_view', 'trace_prefix_v1')
    if type(count) is not int:
        return False
    if view == 'trace_prefix_v1':
        return count in (100, 1000, 4000)
    return (view == 'adapter_coverage_v1' and count == 500
            and cfg.get('qualification_only') is True
            and cfg.get('formal_run') is False
            and cfg.get('experiment_role') == 'adapter_pool_functional_coverage')


def load_http_replay_plan(cfg, *, trace=None):
    """Index existing prompts/arrivals/targets; never write a replacement trace.

    The opt-in coverage view binds sorted static pool IDs to the first 500
    existing requests. Derived rows and their hashes are explicitly different
    from source rows. Publisher, physical ledger and deadline use this SAME
    view; no serving policy or performance-workload mapping is changed.
    """
    from faaslora.datasets.workload_generator import FrozenReplayPlan
    path = cfg.get('trace', trace)
    if trace is not None and Path(path).resolve() != Path(trace).resolve():
        raise ValueError('service and publisher trace differ')
    plan = FrozenReplayPlan.load(path, count=cfg['request_count'])
    if 'trace_sha256' in cfg and plan.source_sha256 != cfg['trace_sha256']:
        raise ValueError('existing trace changed')
    view = cfg.get('request_view', 'trace_prefix_v1')
    if view == 'trace_prefix_v1':
        if cfg['request_count'] == 500:
            raise ValueError('500-ID coverage requires an explicit functional view')
        return plan
    if not qualified_http_population(cfg):
        raise ValueError('unqualified functional replay view')
    raw = Path(cfg['pool_index']).read_bytes()
    if sha(raw) != cfg['pool_index_sha256']:
        raise ValueError('existing pool index changed')
    ids = [row['id'] for row in json.loads(raw)['artifacts']]
    if (len(ids) != 500 or any(not isinstance(aid, str) or not aid for aid in ids)
            or len(set(ids)) != 500):
        raise ValueError('coverage requires 500 distinct existing adapter IDs')
    entries = []
    for ordinal, (entry, aid) in enumerate(zip(plan.entries, sorted(ids))):
        row = json.loads(entry.source_json)
        if 'tc_adapter_coverage' in row or not row.get('adapter_id'):
            raise ValueError('coverage source is not an original adapter-bound request')
        row['tc_adapter_coverage'] = dict(contract=view, ordinal=ordinal,
            original_source_item_sha256=entry.source_sha256,
            original_adapter_id=row['adapter_id'], pool_index_sha256=sha(raw))
        row['adapter_id'] = aid
        canonical = json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        entries.append(replace(entry, source_json=canonical, source_sha256=sha(canonical.encode())))
    return replace(plan, entries=tuple(entries), profile='ADAPTER_COVERAGE_V1')


def http_replay(args):
    """Auxiliary child of the existing guarded launcher, not a new supervisor."""
    cfg = json.loads(args.config.read_text())
    guard = load_guard(Path(cfg['main_repo']))
    if (guard.cgroup_snapshot(guard.cg_path())['memory.max'] != guard.POLICY['aux_max_bytes']
            or set(os.sched_getaffinity(0)) != set(guard.POLICY['aux_cpus'])):
        raise ValueError('HTTP publisher must share the bounded auxiliary domain')
    sys.path.insert(0, cfg['main_repo'])
    from faaslora.clock import local_monotonic_clock_id
    from faaslora.datasets.workload_generator import prepare_frozen_http_request
    from transformers import AutoTokenizer
    if (cfg.get('schema') != 'ieee_tc_serverless_http_replay_v1'
            or cfg.get('generation_contract') != 'fixed_length_greedy_v1'
            or cfg.get('renderer') != 'role_lines_v1'
            or cfg.get('ingress_mode', 'native_direct_v1') not in
                ('native_direct_v1', 'service_pre_ready_v1')
            or not qualified_http_population(cfg)
            or not re.fullmatch(r'http://127\.0\.0\.1:[0-9]+/v1/chat/completions', cfg['url'])):
        raise ValueError('unqualified HTTP replay contract')
    plan = load_http_replay_plan(cfg)
    if plan.source_sha256 != cfg['trace_sha256']:
        raise ValueError('existing trace changed')
    tokenizer = AutoTokenizer.from_pretrained(cfg['backbone'], local_files_only=True)
    prepared = {e.request_id: prepare_frozen_http_request(e, tokenizer, cfg['model']) for e in plan.entries}
    if plan.profile == 'ADAPTER_COVERAGE_V1':
        for entry in plan.entries:
            prepared[entry.request_id]['tc_adapter_coverage'] = json.loads(entry.source_json)['tc_adapter_coverage']
    if cfg.get('physical_lifecycle') not in (None, 'native_store_pool_v1'):
        raise ValueError('unknown physical lifecycle contract')
    if cfg.get('physical_lifecycle') == 'native_store_pool_v1':
        for row in prepared.values():
            row['worker_lifetime_required'] = True
    _, artifact_client = remote_artifact_contract(cfg)
    if artifact_client is not None:
        from faaslora.storage.http_artifact_store import preparation_content_sha256
        descriptions = artifact_client.preparation_manifests(sorted({r['adapter_id'] for r in prepared.values()}))
        for row in prepared.values():
            row.update(artifact_content_sha256=preparation_content_sha256(descriptions[row['adapter_id']]),
                       artifact_manifest_sha256=artifact_client.content_manifest_sha256)
    imported_backends = sorted(name for name in ('torch', 'vllm', 'ray', 'sllm') if name in sys.modules)
    if imported_backends:
        raise RuntimeError(f'external tokenizer publisher imported serving modules: {imported_backends}')
    with args.output.open('x') as log:
        def emit(event):
            log.write(json.dumps(event, separators=(',', ':'), ensure_ascii=False)+'\n')
            log.flush()
        ready = dict(event='replay_ready', plan=plan.identity(), clock_id=local_monotonic_clock_id(),
                     frame_limit=0, transport='http', config_sha256=sha(args.config.read_bytes()),
                     pid=os.getpid(), imported_backend_modules=imported_backends,
                     helper_sha256=sha(Path(__file__).read_bytes()),
                     shared_source_sha256={p: sha((Path(cfg['main_repo'])/p).read_bytes()) for p in
                         ('faaslora/datasets/workload_generator.py', 'faaslora/clock.py',
                          'faaslora/metrics/metrics_collector.py',
                          'faaslora/storage/http_artifact_store.py')})
        emit(ready)
        for row in prepared.values():
            emit(dict(event='request_contract', **{k: v for k, v in row.items()
                                                   if k not in ('body', 'prompt', 'input_token_ids')}))
        print(json.dumps(ready), flush=True)
        origin = json.loads(sys.stdin.readline())

        async def run():
            import signal
            task = asyncio.current_task()
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
            return await replay_http_session(plan, origin, prepared, cfg['url'], emit,
                ingress_required=cfg.get('ingress_mode') == 'service_pre_ready_v1')

        counts = asyncio.run(run())
        if counts['N_failed'] or counts['N_response'] != counts['N_plan']:
            raise RuntimeError('HTTP protocol qualification incomplete; failures preserved')


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("native launcher anchor differs; inspect before adapting")
    return source.replace(old, new, 1)


def render(sources: dict[str, str], *, script_dir: Path, private_root: Path,
           main_repo: Path, gpu_ids: tuple[int, ...], ray_only: bool = False) -> tuple[dict[str, str], dict]:
    if (not gpu_ids or len(set(gpu_ids)) != len(gpu_ids)
            or any(type(i) is not int or i not in range(4) for i in gpu_ids)):
        raise ValueError("explicit unique local GPU IDs from 0..3 required")
    if set(sources) != set(SOURCE_SHA):
        raise ValueError("incomplete native source set")
    for name, source in sources.items():
        if sha(source.encode()) != SOURCE_SHA[name]:
            raise ValueError(f"native source drift: {name}")
    # Only the established single-host, one worker-raylet deployment is adapted.
    # The head has its own object store too: the qualification total is 8 GiB,
    # NOT 8 GiB per Ray node. This equal partition is not a selected M1 optimum.
    allocation = {"head": 4 * GIB, "worker_0": 4 * GIB}
    q = shlex.quote
    for path in (script_dir, private_root, main_repo):
        if not path.is_absolute() or not re.fullmatch(r"[A-Za-z0-9_./-]+", str(path)):
            raise ValueError("native shell adapter requires explicit safe absolute paths")
    if len(str(private_root / "tmux.sock").encode()) >= 100:
        raise ValueError("private socket path is too long")
    helper = Path(__file__).resolve()
    guard = f'''
# TC ownership is verified before any native launch side effect.
/usr/bin/python3 {q(str(helper))} verify --manifest {q(str(script_dir / 'launch_manifest.json'))}
unset TMUX TMUX_PANE
export RAY_TMPDIR={q(str(private_root / 'ray_tmp'))}
export SLLM_TC_RAY_TEMP={q(str(private_root / 'ray_tmp' / 'ray'))}
export SLLM_TC_SPILL={q(str(script_dir / 'spill'))}
export SLLM_TC_TMUX_SOCKET={q(str(private_root / 'tmux.sock'))}
export SLLM_TC_TMUX_CONFIG={q(str(script_dir / 'tmux.conf'))}
export SLLM_SINGLE_HOST_MULTI_GPU=1
export SLLM_WORKER_GPUS={q(','.join(map(str, gpu_ids)))}
export SLLM_SERVE_LOG_PATH={q(str(script_dir / 'serve.log'))}
export SLLM_STORE_LOG={q(str(script_dir / 'store.log'))}
# The fixed native scripts use env prefixes; require explicit environment/source
# identity instead of accidentally selecting the old default 2.48 environment.
: "${{SLLM_HEAD_ENV_PREFIX:?explicit native head environment required}}"
: "${{SLLM_WORKER_ENV_PREFIX:?explicit native worker environment required}}"
: "${{SLLM_STORE_ENV_PREFIX:?explicit native store environment required}}"
: "${{SLLM_REPO_ROOT:?explicit audited native source required}}"
: "${{SLLM_RAY_HEAD_HOST:?explicit cluster host required}}"
: "${{SLLM_RAY_PORT:?explicit private cluster port required}}"
: "${{SLLM_PORT:?explicit service port required}}"
: "${{SLLM_STORE_PATH:?explicit existing checkpoint namespace required}}"
[[ -d "${{SLLM_STORE_PATH}}" ]] || exit 1
[[ "${{SLLM_DIRECT_PATH_MODE:-0}}" == 0 ]] || {{ echo "TC native loader may not fall back to direct path" >&2; exit 1; }}
[[ ! -e "${{SLLM_SERVE_LOG_PATH}}" && ! -e "${{SLLM_STORE_LOG}}" ]] || exit 1
tmux() {{ command tmux -f "${{SLLM_TC_TMUX_CONFIG}}" -S "${{SLLM_TC_TMUX_SOCKET}}" "$@"; }}
[[ ! -e "${{SLLM_TC_TMUX_SOCKET}}" ]] || {{ echo "private socket already exists" >&2; exit 1; }}
'''
    result = dict(sources)
    stack = result["start_serverlessllm_stack.sh"]
    stack = replace_once(stack, "set -euo pipefail\n", "set -euo pipefail\n" + guard)
    stack = replace_once(stack, 'SCRIPTS_DIR="${ROOT_DIR}/scripts"',
                         f'SCRIPTS_DIR={q(str(script_dir))}')
    begin = stack.index('bash "${SCRIPTS_DIR}/sync_serverlessllm_runtime_sources.sh"')
    end = stack.index('\ntmux new-session -d -s "${HEAD_SESSION}"', begin)
    stack = stack[:begin] + "# No installed-source mutation or pre-existing process cleanup.\n" + stack[end:]
    stack = replace_once(stack, 'CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} bash ${SCRIPTS_DIR}/run_serverlessllm_store.sh',
                         'CUDA_VISIBLE_DEVICES=${WORKER_GPUS} bash ${SCRIPTS_DIR}/run_serverlessllm_store.sh')
    stack = replace_once(stack, 'store  : ${STORE_SESSION} (gpu=${GPU_LIST[0]})',
                         'store  : ${STORE_SESSION} (gpus=${WORKER_GPUS})')
    stack = replace_once(stack, 'rm -f "${SERVE_LOG_PATH}"', '# New log only; checked before launch.')
    stack = replace_once(stack, '"bash -lc \'env${VLLM_ENV_PREFIX}', '"bash -c \'env${VLLM_ENV_PREFIX}')
    if ray_only:
        # Explicit infrastructure qualification, not a direct-path fallback.
        # Execute the identical guarded head/worker prefix; do not start store,
        # API, load a model or submit inference requests in this mode.
        marker = 'wait_for_workers "${EXPECTED_WORKERS}"\n'
        if stack.count(marker) != 1:
            raise ValueError("native worker-start boundary differs")
        stack = stack[:stack.index(marker) + len(marker)] + '\necho "TC ray-only infrastructure ready"\n'
    else:
        # Native CheckpointStore construction has no Ray dependency. Start it
        # alongside the head/worker chain, but retain BOTH readiness barriers
        # before starting the controller. All processes still start after the
        # common deployment notice inside the same guarded service domain.
        # This changes launcher dependencies, not model registration, RR,
        # scaling, native loading, queueing, or the replay's fixed t=0.
        store_begin = stack.index('if [[ "${DIRECT_PATH_MODE}" != "1" ]]; then\n  tmux new-session')
        store_end = stack.index('\nfi', store_begin) + len('\nfi')
        store_block = stack[store_begin:store_end]
        if store_block.count('  wait_for_store\n') != 1:
            raise ValueError('native store readiness boundary differs')
        store_start = replace_once(store_block, '  wait_for_store\n', '')
        stack = stack[:store_begin] + 'wait_for_store' + stack[store_end:]
        head_start = '\ntmux new-session -d -s "${HEAD_SESSION}"'
        stack = replace_once(stack, head_start, '\n' + store_start + '\n' + head_start)
    result["start_serverlessllm_stack.sh"] = stack
    for role in ("head", "worker"):
        name = f"run_serverlessllm_{role}.sh"
        source = result[name]
        # A JSON closing brace inside ${var:-...} otherwise terminates the
        # expansion and appends another brace when an explicit value is set.
        resource_var = f"SLLM_{role.upper()}_RESOURCES"
        resource_line = next(line for line in source.splitlines()
                             if line.startswith(resource_var + '='))
        default = ('\'{"control_node": 1}\'' if role == "head" else
                   '"{\\"worker_node\\": 1, \\"worker_id_${WORKER_ID}\\": 1}"')
        source = replace_once(source, resource_line,
            f'{resource_var}="${{{resource_var}:-}}"\n'
            f'if [[ -z "${{{resource_var}}}" ]]; then\n'
            f'  {resource_var}={default}\nfi')
        source = replace_once(source, 'RAY_OBJECT_STORE_MEMORY_BYTES="${SLLM_RAY_OBJECT_STORE_MEMORY_BYTES:-}"',
                              f'RAY_OBJECT_STORE_MEMORY_BYTES={allocation["head" if role == "head" else "worker_0"]}')
        extra = ['  --object-spilling-directory="${SLLM_TC_SPILL}/' + role + '"']
        if role == "head":
            extra.append('  --temp-dir="${SLLM_TC_RAY_TEMP}"')
        source = replace_once(source, "  --block\n)", "  --block\n" + "\n".join(extra) + "\n)")
        result[name] = source
    name = "run_serverlessllm_serve.sh"
    result[name] = replace_once(result[name], '-m sllm.cli.clic start --host "${HOST}" --port "${PORT}"',
                               '-m sllm.cli.clic start --host "${HOST}" --port "${PORT}" --enable-storage-aware')
    return result, allocation


def prepare(output: Path, private_root: Path, main_repo: Path, gpu_ids: tuple[int, ...],
            *, ray_only: bool = False) -> dict:
    for path in (output, private_root):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refuse to reuse or overwrite {path}")
    # Results are intentionally mounted through an existing repository symlink.
    # Freeze the physical destination once, instead of generating a logical
    # path then comparing it against a resolved path at launch time.
    requested_output = str(output)
    output, private_root, main_repo = (p.resolve() for p in (output, private_root, main_repo))
    sources = {name: (ROOT / "scripts" / name).read_text() for name in SOURCE_SHA}
    rendered, allocation = render(sources, script_dir=output, private_root=private_root,
                                 main_repo=main_repo, gpu_ids=gpu_ids, ray_only=ray_only)
    manifest = {
        "schema": "ieee_tc_serverless_native_launcher_view_v1", "display_name": "ServerlessLLM",
        "main_repo": str(main_repo), "private_root": str(private_root),
        "script_dir": str(output), "source_sha256": SOURCE_SHA,
        "requested_script_dir": requested_output,
        "tmux_config_sha256": sha(TMUX_CONFIG.encode()),
        "rendered_sha256": {name: sha(data.encode()) for name, data in rendered.items()},
        "helper_sha256": sha(Path(__file__).read_bytes()),
        "object_store_bytes_by_raylet": allocation,
        "object_store_bytes_total": sum(allocation.values()), "worker_gpu_ids": list(gpu_ids),
        "ray_only_infrastructure": ray_only,
        "bootstrap_dependencies": ("head_then_worker" if ray_only else
                                   "store_parallel_head_worker_then_controller_v1"),
        "qualification_only": True, "actual_workers_verified": False,
        "native_loader_qualified": False, "performance_run_authorized": False,
        "cleanup_owner": "existing external TC gated launcher and watchdog",
    }
    output.mkdir(mode=0o700)
    private_root.mkdir(mode=0o700)
    for sub in (private_root / "ray_tmp", output / "spill", output / "spill" / "head", output / "spill" / "worker"):
        sub.mkdir(mode=0o700)
    for name, data in rendered.items():
        with (output / name).open("x") as handle:
            handle.write(data)
    with (output / 'tmux.conf').open('x') as handle:
        handle.write(TMUX_CONFIG)
    with (output / "launch_manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def load_guard(main_repo: Path):
    checker = main_repo / "scripts" / "ieee_tc_preflight.py"
    spec = importlib.util.spec_from_file_location("tc_native_launch_guard", checker)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent.resolve(strict=True)
    if manifest.get("schema") != "ieee_tc_serverless_native_launcher_view_v1" or str(root) != manifest["script_dir"]:
        raise ValueError("launcher identity differs")
    if sha(Path(__file__).read_bytes()) != manifest["helper_sha256"]:
        raise ValueError("launcher adapter changed after preparation")
    if (manifest.get('tmux_config_sha256') != sha(TMUX_CONFIG.encode())
            or sha((root / 'tmux.conf').read_bytes()) != manifest['tmux_config_sha256']):
        raise ValueError('private tmux configuration changed')
    if (manifest.get("source_sha256") != SOURCE_SHA
            or set(manifest["rendered_sha256"]) != set(SOURCE_SHA)
            or manifest["object_store_bytes_by_raylet"] != {"head": 4 * GIB, "worker_0": 4 * GIB}
            or manifest["object_store_bytes_total"] != 8 * GIB):
        raise ValueError("native source or aggregate object-store contract differs")
    for name, expected in manifest["rendered_sha256"].items():
        if Path(name).name != name or sha((root / name).read_bytes()) != expected:
            raise ValueError("generated launch script changed")
    private = Path(manifest["private_root"])
    if private.is_symlink() or private.stat().st_uid != os.getuid() or private.stat().st_mode & 0o077:
        raise ValueError("private runtime directory ownership differs")
    module = load_guard(Path(manifest["main_repo"]))
    admission = module.verify_current_service()
    if os.environ.get("SLLM_RAY_OBJECT_STORE_MEMORY_BYTES"):
        raise ValueError("legacy per-node object-store override conflicts with aggregate TC contract")
    # Binding is only a pre-launch collision check, not proof that a later HTTP
    # listener belongs to us. Actual Ray workers/API/store still need ownership
    # and native-source readback in the qualification experiment.
    endpoints = [(os.environ["SLLM_RAY_HEAD_HOST"], int(os.environ["SLLM_RAY_PORT"])),
                 (os.environ.get("SLLM_HOST", "127.0.0.1"), int(os.environ["SLLM_PORT"])),
                 ("0.0.0.0", 8073)]
    if len({port for _, port in endpoints}) != len(endpoints):
        raise ValueError("service/control/store ports collide")
    for host, port in endpoints:
        if not 1024 <= port <= 65535:
            raise ValueError("invalid unprivileged service port")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, port))
    return admission


def validate_ray_nodes(nodes: list[dict], gpu_count: int) -> dict:
    alive = [node for node in nodes if node["Alive"]]
    if len(alive) != 2 or len({n['NodeID'] for n in alive}) != 2:
        raise ValueError("expected exactly two live native raylets")
    result = {}
    for node in alive:
        resources = node["Resources"]
        is_head, is_worker = resources.get("control_node", 0) == 1, resources.get("worker_node", 0) == 1
        if is_head == is_worker:
            raise ValueError("native raylet role is ambiguous")
        role = "head" if is_head else "worker_0"
        if role in result or resources.get("GPU", 0) != (0 if is_head else gpu_count):
            raise ValueError("native role or GPU inventory differs")
        if resources.get("object_store_memory") != 4 * GIB:
            raise ValueError("live raylet object-store capacity differs")
        result[role] = node
    if set(result) != {"head", "worker_0"}:
        raise ValueError("native head/worker roles incomplete")
    return result


def checkpoint_tensor_sources(source_keys: set[str]) -> dict[str, list[str]]:
    """The native TP=1 Llama packing; reject incomplete fused projections.

    vLLM 0.10.2 LlamaForCausalLM packs q/k/v and gate/up in this order.
    This is an audit mapping, not a checkpoint converter or a model loader.
    """
    grouped: dict[str, list[str]] = {}
    for name in sorted(source_keys):
        target, parts = name, [name]
        for fused, separate in (("qkv_proj", ("q_proj", "k_proj", "v_proj")),
                                ("gate_up_proj", ("gate_proj", "up_proj"))):
            for part in separate:
                token = f".{part}."
                if token in name:
                    target = name.replace(token, f".{fused}.")
                    parts = [name.replace(token, f".{item}.") for item in separate]
                    break
        if not set(parts) <= source_keys:
            raise ValueError(f"incomplete native projection source: {name}")
        if target in grouped and grouped[target] != parts:
            raise ValueError(f"ambiguous native packing: {target}")
        grouped[target] = parts
    if sum(map(len, grouped.values())) != len(source_keys):
        raise ValueError("source tensors are duplicated or omitted")
    return grouped


def validate_checkpoint_index(index: dict, sources: dict, size: int) -> list[tuple]:
    if set(index) != set(sources):
        raise ValueError("native checkpoint tensor keys differ from the complete source")
    ordered = sorted(index.items(), key=lambda item: item[1][0])
    end = 0
    for name, record in ordered:
        if len(record) != 5:
            raise ValueError("unexpected native tensor record")
        offset, length, shape, stride, dtype = record
        if (type(offset) is not int or type(length) is not int or offset != end
                or dtype != 'torch.float16' or not shape
                or any(type(n) is not int or n <= 0 for n in shape)
                or length != 2 * math.prod(shape)
                or stride != [math.prod(shape[i + 1:]) for i in range(len(shape))]):
            raise ValueError(f"unsupported or non-contiguous TP1 FP16 tensor: {name}")
        end += length
    if end != size:
        raise ValueError("native checkpoint has missing or extra bytes")
    return ordered


def stream_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


class NativeTensorStream:
    """Bounded sequential reads over native store partitions in numeric order."""
    def __init__(self, handles):
        self.handles = handles
        self.partition = 0
        self.position = 0
        self.digests = [hashlib.sha256() for _ in handles]

    def read(self, size: int) -> bytes:
        if type(size) is not int or size < 0 or size > 4 * 1024**2:
            raise ValueError('bounded nonnegative native tensor read required')
        chunks, remaining = [], size
        while remaining and self.partition < len(self.handles):
            chunk = self.handles[self.partition].read(remaining)
            if not chunk:
                self.partition += 1
                continue
            self.digests[self.partition].update(chunk)
            chunks.append(chunk)
            remaining -= len(chunk)
            self.position += len(chunk)
        return b''.join(chunks)

    def tell(self) -> int:
        return self.position


def checkpoint_parts(rank: Path) -> list[Path]:
    parts = list(rank.glob('tensor.data_*'))
    if not parts or any(not re.fullmatch(r'tensor\.data_[0-9]+', p.name) for p in parts):
        raise ValueError('missing or malformed native partitions')
    parts.sort(key=lambda path: int(path.name.split('_')[-1]))
    if ([p.name for p in parts] != [f'tensor.data_{i}' for i in range(len(parts))]
            or any(not p.is_file() or p.is_symlink() or p.stat().st_size <= 0 for p in parts)
            or set(p.name for p in rank.iterdir()) != {'tensor_index.json', *(p.name for p in parts)}):
        raise ValueError('native partitions must be complete numbered ordinary files')
    return parts


def derived_rope_sources(source_keys: set[str], config: dict) -> set[str]:
    observed = {name for name in source_keys if 'rotary_emb.inv_freq' in name}
    if not observed:
        return set()
    expected = {f'model.layers.{i}.self_attn.rotary_emb.inv_freq'
                for i in range(config['num_hidden_layers'])}
    if (observed != expected or config.get('rope_scaling') is not None
            or config.get('partial_rotary_factor', 1) != 1):
        raise ValueError('unsupported or incomplete derived RoPE source buffers')
    return observed


def validate_stored_rope(actual, expected) -> str:
    """No tolerance: accept exact FP32 or its exact FP16 serialization roundtrip."""
    import torch
    if actual.shape != expected.shape or actual.dtype not in (torch.float16, torch.float32):
        raise ValueError('stored RoPE shape/dtype differs')
    if torch.equal(actual, expected.to(actual.dtype)):
        return 'exact_config_formula_at_stored_dtype'
    if torch.equal(actual, expected.half().to(actual.dtype)):
        return 'exact_fp16_serialization_roundtrip'
    raise ValueError('stored RoPE values do not match the declared configuration')


def audit_derived_rope(names, config, readers, source_map) -> dict:
    """Account for buffers the official vLLM loader deliberately recomputes."""
    if not names:
        return dict(buffers=[], scope='no stored derived RoPE buffers')
    import torch
    from types import SimpleNamespace
    root = Path(importlib.util.find_spec('vllm').origin).parent
    identities = {
        'model_executor/models/llama.py': '231282afc57850184704a0ab064a83828701ab8662c9547042fbe4f987a4ea93',
        'model_executor/layers/rotary_embedding/base.py': '69986aca0500b2f170d8567ace04b54cbc3d2dc0f5e8ab9de0524244945ed108',
    }
    for name, expected in identities.items():
        if stream_sha(root / name) != expected:
            raise ValueError('inspected native Llama/RoPE implementation changed')
    rope_path = root / 'model_executor/layers/rotary_embedding/base.py'
    tree = ast.parse(rope_path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RotaryEmbedding')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_compute_inv_freq')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(rope_path), 'exec'), namespace)
    dimension = config.get('head_dim') or config['hidden_size'] // config['num_attention_heads']
    base = config.get('rope_theta', 10000)
    expected = namespace['_compute_inv_freq'](SimpleNamespace(rotary_dim=dimension), base)
    rows = []
    for name in sorted(names):
        actual = readers[source_map[name]].get_tensor(name)
        interpretation = validate_stored_rope(actual, expected)
        rows.append(dict(tensor=name, dtype=str(actual.dtype), shape=list(actual.shape),
                         source_sha256=sha(actual.contiguous().numpy().tobytes()),
                         interpretation=interpretation,
                         max_abs_difference_from_recomputed_fp32=float((actual.float() - expected).abs().max())))
    return dict(buffers=rows, backend_source_sha256=identities, base=base, rotary_dim=dimension,
                scope='Official vLLM ignores stored inv_freq and recomputes nonpersistent RoPE; not bitwise HF inference equivalence')


def audit_checkpoint(checkpoint: Path, backbone: Path) -> dict:
    """Read existing TP1 Llama FP16 checkpoint bytes; never load a CUDA model.

    Compare every native element with the original source cast to FP16,
    preserving exact packed projection order. Bounded row slices avoid loading
    or duplicating either complete 6GB checkpoint. Current hashes establish
    current identity, not retrospective identity for an old performance run.
    """
    from contextlib import ExitStack
    import torch
    from safetensors import safe_open
    if torch.cuda.is_initialized():
        raise ValueError('checkpoint audit requires an uninitialized CUDA context')
    torch.set_num_threads(2)
    checkpoint, backbone = checkpoint.resolve(strict=True), backbone.resolve(strict=True)
    rank = checkpoint / 'rank_0'
    index_path = rank / 'tensor_index.json'
    data_paths = checkpoint_parts(rank)
    data_bytes = sum(path.stat().st_size for path in data_paths)
    native_index = json.loads(index_path.read_text())
    source_index = json.loads((backbone / 'model.safetensors.index.json').read_text())
    source_map = source_index['weight_map']
    config = json.loads((backbone / 'config.json').read_text())
    if (config.get('model_type') != 'llama' or config.get('architectures') != ['LlamaForCausalLM']
            or config.get('quantization_config')):
        raise ValueError('auditor requires an unquantized Llama TP1 checkpoint')
    # Llama-2 has an independent lm_head; Llama-3.2 ties it to embeddings.
    # Do not silently discard either source: exact key/byte comparison below
    # must cover the independent head when present.
    if not config.get('tie_word_embeddings', False) and 'lm_head.weight' not in source_map:
        raise ValueError('untied Llama checkpoint is missing its independent output head')
    if set(p.name for p in checkpoint.glob('rank_*')) != {'rank_0'}:
        raise ValueError('audit requires exactly one native tensor-parallel rank')
    inputs = [index_path, *data_paths, backbone / 'model.safetensors.index.json']
    small = {}
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json',
                 'generation_config.json', 'special_tokens_map.json'):
        current = stream_sha(backbone / name)
        if current != stream_sha(checkpoint / name):
            raise ValueError(f'checkpoint and source differ: {name}')
        small[name] = current
        inputs.extend((backbone / name, checkpoint / name))
    derived = derived_rope_sources(set(source_map), config)
    mapping = checkpoint_tensor_sources(set(source_map) - derived)
    ordered = validate_checkpoint_index(native_index, mapping, data_bytes)
    shards = {}
    for name in set(source_map.values()):
        path = backbone / name
        if Path(name).name != name or path.resolve().parent != backbone:
            raise ValueError('source shard escapes backbone directory')
        shards[name] = path
    inputs.extend(shards.values())
    identity = lambda p: (p.stat().st_dev, p.stat().st_ino, p.stat().st_size,
                          p.stat().st_mtime_ns, p.stat().st_ctime_ns)
    before = {str(p): identity(p) for p in inputs}
    rows, all_native = [], hashlib.sha256()
    started = time.monotonic()
    with ExitStack() as stack:
        readers = {name: stack.enter_context(safe_open(path, framework='pt', device='cpu'))
                   for name, path in shards.items()}
        actual_keys = {key: name for name, reader in readers.items() for key in reader.keys()}
        if actual_keys != source_map or sum(len(r.keys()) for r in readers.values()) != len(source_map):
            raise ValueError('source index does not describe every actual tensor exactly once')
        derived_audit = audit_derived_rope(derived, config, readers, source_map)
        raw = NativeTensorStream([stack.enter_context(path.open('rb')) for path in data_paths])
        for name, (offset, length, shape, stride, dtype) in ordered:
            native_digest, reference_digest = hashlib.sha256(), hashlib.sha256()
            consumed, source_rows = 0, 0
            for source in mapping[name]:
                part = readers[source_map[source]].get_slice(source)
                source_shape = part.get_shape()
                if source_shape[1:] != shape[1:]:
                    raise ValueError(f'packed tensor shape mismatch: {source}')
                source_rows += source_shape[0]
                step = max(1, 4 * 1024**2 // (2 * math.prod(source_shape[1:])))
                for start in range(0, source_shape[0], step):
                    tensor = part[start:min(start + step, source_shape[0])]
                    expected = tensor.to(dtype=torch.float16).contiguous().numpy().tobytes()
                    actual = raw.read(len(expected))
                    if actual != expected:
                        raise ValueError(f'exact FP16 checkpoint mismatch: {name}, source={source}, row={start}')
                    native_digest.update(actual)
                    reference_digest.update(expected)
                    all_native.update(actual)
                    consumed += len(actual)
                    del tensor, expected, actual
            if consumed != length or source_rows != shape[0] or raw.tell() != offset + length:
                raise ValueError(f'packed tensor extent differs: {name}')
            rows.append(dict(tensor=name, source_tensors=mapping[name], offset=offset,
                             bytes=length, shape=shape, native_sha256=native_digest.hexdigest(),
                             reference_fp16_sha256=reference_digest.hexdigest(), exact=True))
    source_sha = {name: stream_sha(path) for name, path in shards.items()}
    if before != {str(p): identity(p) for p in inputs} or torch.cuda.is_initialized():
        raise ValueError('input changed or CUDA initialized during read-only checkpoint audit')
    return dict(schema='ieee_tc_serverless_native_checkpoint_identity_v1', passed=True,
                checkpoint=str(checkpoint), backbone=str(backbone), tensor_parallel_size=1,
                dtype='float16', native_tensor_count=len(rows), source_tensor_count=len(source_map),
                serialized_source_tensor_count=len(source_map) - len(derived),
                derived_source_buffer_audit=derived_audit,
                native_data_bytes=data_bytes, native_data_sha256=all_native.hexdigest(),
                native_parts=[dict(name=path.name, bytes=path.stat().st_size,
                                   sha256=digest.hexdigest())
                              for path, digest in zip(data_paths, raw.digests)],
                native_index_sha256=stream_sha(index_path), small_file_sha256=small,
                source_index_sha256=stream_sha(backbone / 'model.safetensors.index.json'),
                source_shard_sha256=source_sha, tensors=rows,
                elapsed_s=time.monotonic() - started, cuda_initialized=False,
                native_loader_qualified=False, performance_run_authorized=False,
                limitation='Current serialized tensor identity only; no historical, runtime loading or LoRA correctness claim')


def export_checkpoint(args) -> dict:
    """Call the existing native exporter once for a missing local format.

    This is representation conversion of existing backbone weights, not a new
    model, a serving run or a replacement for the later exact-byte audit. The
    official exporter is used unchanged; output is exclusive and remains local.
    """
    guard = load_guard(args.main_repo)
    admission = guard.verify_current_service()
    if Path(sys.executable).resolve() != (args.environment / 'bin/python').resolve():
        raise ValueError('wrong native exporter interpreter')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.model_name):
        raise ValueError('invalid exclusive native checkpoint name')
    destination = args.checkpoint_root.resolve(strict=True) / 'vllm' / args.model_name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'refuse to overwrite existing checkpoint {destination}')
    source = args.backbone.resolve(strict=True)
    source_index = json.loads((source / 'model.safetensors.index.json').read_text())
    shard_names = set(source_index['weight_map'].values())
    if not shard_names or any(Path(name).name != name or not (source / name).is_file()
                              for name in shard_names):
        raise ValueError('complete existing local backbone is required')
    package = args.store_package / 'site-packages'
    if (os.environ.get('PYTHONPATH') != f'{args.native_source}:{package}'
            or os.environ.get('LD_LIBRARY_PATH') != str(package / 'sllm_store')
            or os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu_id)):
        raise ValueError('explicit native source/library and single-GPU composition required')
    receipt = json.loads(args.overlay_receipt.read_text())
    expected_root = args.environment / 'lib/python3.12/site-packages/vllm'
    if (receipt.get('status') != 'INSTALLED' or len(receipt['members']) != 7
            or Path(receipt['vllm_root']).resolve() != expected_root.resolve()):
        raise ValueError('installed loader-only overlay receipt required')
    for row in receipt['members']:
        if sha((expected_root / row['path']).read_bytes()) != row['after_sha256']:
            raise ValueError('installed loader-only source changed')
    import sllm.model_downloader as downloader
    if Path(downloader.__file__).resolve() != args.native_source / 'sllm/model_downloader.py':
        raise ValueError('selected native exporter source differs')
    # Offline inputs only; no whole-pool regeneration, download or Ray cluster.
    os.environ.update(STORAGE_PATH=str(args.checkpoint_root), HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', VLLM_USE_V1='1')
    result = dict(schema='ieee_tc_serverless_native_checkpoint_export_v1', passed=False,
                  representation_conversion_only=True, native_loader_qualified=False,
                  performance_run_authorized=False, service=admission,
                  backbone=str(source), checkpoint=str(destination), gpu_id=args.gpu_id,
                  tensor_parallel_size=1, dtype='float16', source_shards=sorted(shard_names),
                  source_index_sha256=stream_sha(source / 'model.safetensors.index.json'),
                  exporter_sha256=stream_sha(Path(downloader.__file__)),
                  overlay_receipt_sha256=stream_sha(args.overlay_receipt),
                  started_monotonic=time.monotonic())
    # Reserve the result before the first model allocation. Failure evidence is
    # retained even though the official exporter removes its own partial model.
    with args.output.open('x') as handle:
        try:
            downloader.VllmModelDownloader().download_vllm_model(
                model_name=args.model_name, pretrained_model_name_or_path=str(source),
                torch_dtype='float16', tensor_parallel_size=1)
            if not (destination / 'rank_0/tensor.data_0').is_file():
                raise RuntimeError('native exporter did not produce a rank-0 checkpoint')
            result['files'] = {str(path.relative_to(destination)): dict(bytes=path.stat().st_size,
                              sha256=stream_sha(path)) for path in sorted(destination.rglob('*'))
                              if path.is_file()}
            result['passed'] = True
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            result['finished_monotonic'] = time.monotonic()
            result['external_cleanup_required'] = True
            json.dump(result, handle, indent=2)
    return result


def qualify_ray(args) -> dict:
    """Observe the actual two-raylet launch inside the existing external gate."""
    guard = load_guard(args.main_repo)
    admission = guard.verify_current_service()
    if Path(sys.executable).resolve() != (args.environment / 'bin/python').resolve():
        raise ValueError("use the explicitly selected native Ray interpreter")
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    if ray.__version__ != '2.54.0' or ray.__commit__ != '48bd1f8fa43d0e8222b0f57357b99b48c7437ed3':
        raise ValueError("native Ray identity differs from the inspected environment")
    result = dict(schema="ieee_tc_serverless_two_raylet_witness_v1", passed=False,
                  inference_requests=0, model_loaded=False, actual_model_workers_qualified=False,
                  performance_run_authorized=False, service=admission,
                  ray_version=ray.__version__, ray_commit=ray.__commit__, ray_file=ray.__file__,
                  ray_init_sha256=sha(Path(ray.__file__).read_bytes()))
    manifest = prepare(args.output, args.private_root, args.main_repo, args.gpu_ids, ray_only=True)
    result['launcher_manifest_sha256'] = sha((args.output / 'launch_manifest.json').read_bytes())
    env = dict(os.environ)
    env.pop('TMUX', None)
    env.pop('TMUX_PANE', None)
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env[key] = ''
    env.update(SLLM_HEAD_ENV_PREFIX=str(args.environment), SLLM_WORKER_ENV_PREFIX=str(args.environment),
               SLLM_STORE_ENV_PREFIX=str(args.environment), SLLM_REPO_ROOT=str(args.native_source),
               SLLM_RAY_HEAD_HOST=args.host, SLLM_RAY_PORT=str(args.ray_port), SLLM_PORT=str(args.api_port),
               SLLM_STORE_PATH=str(args.checkpoint_root), SLLM_DIRECT_PATH_MODE='0',
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', RAY_USAGE_STATS_ENABLED='0',
               RAY_TMPDIR=str(args.private_root / 'ray_tmp'),
               NO_PROXY=f"{args.host},127.0.0.1,localhost", no_proxy=f"{args.host},127.0.0.1,localhost")
    env.update(SLLM_HEAD_SESSION='sllm_head_formal', SLLM_WORKER_SESSION_PREFIX='sllm_worker_formal',
               SLLM_RAY_HEAD_ADDRESS=f'{args.host}:{args.ray_port}', RAY_ADDRESS=f'{args.host}:{args.ray_port}',
               SLLM_HEAD_RESOURCES='{"control_node": 1}',
               SLLM_WORKER_RESOURCES='{"worker_node": 1, "worker_id_0": 1}')
    # Native shell prefixes must not inherit alternate binary selections.
    for key in ('SLLM_HEAD_RAY_BIN', 'SLLM_WORKER_RAY_BIN', 'SLLM_HEAD_PYTHON_BIN'):
        env.pop(key, None)
    os.environ.update({key: env[key] for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                       'http_proxy', 'https_proxy', 'all_proxy', 'NO_PROXY', 'no_proxy', 'RAY_TMPDIR')})
    tmux = ['tmux', '-f', str(args.output / 'tmux.conf'), '-S', str(args.private_root / 'tmux.sock')]
    actors, failure = [], None
    try:
        with (args.output / 'startup.log').open('x') as log:
            startup = subprocess.run(['bash', str(args.output / 'start_serverlessllm_stack.sh')],
                                     env=env, stdout=log, stderr=subprocess.STDOUT, timeout=210)
        result['startup_returncode'] = startup.returncode
        if startup.returncode:
            raise RuntimeError('native Ray startup failed; preserve startup and private Ray logs')
        ray.init(address=f'{args.host}:{args.ray_port}', namespace='tc-launch-qualification',
                 log_to_driver=False)
        nodes = ray.nodes()
        result['nodes'] = nodes
        by_role = validate_ray_nodes(nodes, len(args.gpu_ids))

        @ray.remote(max_restarts=0)
        class Witness:
            def report(self):
                import os
                import pathlib
                import subprocess
                import sys
                import json
                import ray
                child = subprocess.check_output([sys.executable, '-c',
                    'import os,json,pathlib; print(json.dumps(dict(pid=os.getpid(),'
                    'cgroup=pathlib.Path("/proc/self/cgroup").read_text(),'
                    'affinity=sorted(os.sched_getaffinity(0)))))'], text=True, timeout=10)
                return dict(pid=os.getpid(), cgroup=pathlib.Path('/proc/self/cgroup').read_text(),
                            affinity=sorted(os.sched_getaffinity(0)), child=json.loads(child),
                            node_id=ray.get_runtime_context().get_node_id(),
                            gpu_ids=ray.get_gpu_ids(), cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'))

        for role, node in by_role.items():
            for _ in range(1 if role == 'head' else len(args.gpu_ids)):
                actors.append(Witness.options(num_cpus=1, num_gpus=0 if role == 'head' else 1,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node['NodeID'], soft=False)).remote())
        reports = ray.get([actor.report.remote() for actor in actors], timeout=60)
        result['worker_reports'] = reports
        expected_group = str(Path(admission['service_identity']['path']).relative_to('/sys/fs/cgroup'))
        if len({row['pid'] for row in reports}) != len(actors):
            raise RuntimeError('expected distinct native worker processes')
        for row in reports:
            for observed in (row, row['child']):
                if observed['cgroup'].strip() != '0::/' + expected_group or observed['affinity'] != sorted(guard.SERVICE_CPUS):
                    raise RuntimeError('actual Ray worker or its child escaped the common service envelope')
        worker_reports = [r for r in reports if r['node_id'] == by_role['worker_0']['NodeID']]
        if (len(worker_reports) != len(args.gpu_ids)
                or sorted(int(g) for r in worker_reports for g in r['gpu_ids']) != list(range(len(args.gpu_ids)))):
            raise RuntimeError('native witness GPU scheduling coverage differs')
        group = Path(admission['service_identity']['path'])
        result['owned_processes'] = guard.owned_pids(group)
        result['resource_snapshot'] = guard.cgroup_snapshot(group)
        raylets = []
        for proc in result['owned_processes']:
            try:
                command = Path(f"/proc/{proc['pid']}/cmdline").read_bytes().split(b'\0')
            except FileNotFoundError:
                continue
            if command and Path(command[0].decode()).name == 'raylet':
                args_readback = [part.decode() for part in command if part]
                if f'--object_store_memory={4 * GIB}' not in args_readback:
                    raise RuntimeError('actual raylet process capacity differs')
                raylets.append(dict(identity=proc, command=args_readback))
        result['raylet_processes'] = raylets
        if len(raylets) != 2:
            raise RuntimeError('two live owned raylet processes were not observed')
        result['passed'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        failure = exc
    finally:
        cleanup_errors = []
        if ray.is_initialized():
            for actor in actors:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception as exc:
                    cleanup_errors.append(str(exc))
            try:
                ray.shutdown()
            except Exception as exc:
                cleanup_errors.append(str(exc))
        # Only this fresh private tmux server is addressed. Its descendants
        # remain covered by the original gated launcher's final whole-tree check.
        for session in ('sllm_head_formal', 'sllm_worker_formal_0'):
            capture = subprocess.run(tmux + ['capture-pane', '-pJt', session, '-S', '-'],
                                     text=True, capture_output=True, timeout=5)
            with (args.output / f'{session}.log').open('x') as handle:
                handle.write(capture.stdout + capture.stderr)
        killed = subprocess.run(tmux + ['kill-server'], capture_output=True, text=True, timeout=5)
        result['private_tmux_stop_returncode'] = killed.returncode
        result['cleanup_errors'] = cleanup_errors
        if cleanup_errors:
            result['passed'] = False
            failure = failure or RuntimeError('native actor cleanup failed')
        result['external_cleanup_required'] = True
        with (args.output / 'ray_witness.json').open('x') as handle:
            json.dump(result, handle, indent=2)
    if failure:
        raise RuntimeError('ServerlessLLM Ray qualification failed; see preserved witness') from failure
    return result


def native_model_config(checkpoint: Path, backbone: Path, *, min_instances=1,
                        max_instances=1, target=1, keep_alive=10, enforce_eager=True) -> dict:
    """Explicit native scaling settings; defaults retain the loader-only witness.

    Development polling pairs can reuse historical scaling without changing
    the native controller or treating a one-inflight witness as a tuned point.
    Eager remains the historical default. Opting out exposes the upstream
    graph/eager hybrid; it does not qualify graphs or silently retry in eager.
    """
    if (any(type(v) is not int for v in (min_instances, max_instances, target, keep_alive))
            or not 0 <= min_instances <= max_instances <= 4
            or max_instances < 1 or target < 1 or keep_alive < 0):
        raise ValueError('invalid native scaling configuration')
    if type(enforce_eager) is not bool:
        raise ValueError('enforce_eager must be an explicit boolean')
    return dict(model=checkpoint.name, backend='vllm', num_gpus=1,
                auto_scaling_config=dict(metric='concurrency', target=target,
                    min_instances=min_instances, max_instances=max_instances,
                    keep_alive=keep_alive), router_config={},
                backend_config=dict(pretrained_model_name_or_path=str(backbone),
                    tensor_parallel_size=1, torch_dtype='float16',
                    gpu_memory_utilization=0.72, max_model_len=1024, max_num_seqs=4,
                    max_num_batched_tokens=1024, enable_chunked_prefill=True,
                    enable_prefix_caching=True, enforce_eager=enforce_eager, task='generate',
                    vllm_use_v1=True, skip_store_model_registration=False,
                    skip_store_lora_registration=False))


def validate_native_response(body: dict, request_id: str, target: int, input_count: int) -> None:
    if (body.get('error') or body.get('id') != request_id
            or body.get('usage', {}).get('completion_tokens') != target
            or body.get('usage', {}).get('prompt_tokens') != input_count
            or not body.get('metrics', {}).get('instance_id')
            or not body.get('choices')):
        raise ValueError('native fixed-output backbone response failed; not a LoRA correctness test')


def existing_pool_embedding_policy(adapter_map: dict[str, str]) -> dict:
    """Reuse the historical content-based deployment selector, never guess.

    A native backbone checkpoint has no extra-vocabulary rows. The existing
    environment supports linear-only adapters without modifying that layout.
    Reject embedding deltas instead of truncating, zero-filling or ignoring them.
    """
    helper = ROOT / 'scripts/generate_serverlessllm_deploy_config.py'
    if sha(helper.read_bytes()) != '45ab8b151a9fb3b53e2b1a8fb86d0c180604eb3f14ef999792e2a413ad9d53f4':
        raise ValueError('historical deployment selector changed; inspect before reuse')
    spec = importlib.util.spec_from_file_location('tc_existing_deploy_selector', helper)
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    if selector.safe_open is None or not adapter_map:
        raise ValueError('complete safetensors header inspection required')
    config_shas = {}
    for aid, path in sorted(adapter_map.items()):
        directory = Path(path)
        config_path = directory/'adapter_config.json'
        config = json.loads(config_path.read_text())
        targets = config.get('target_modules')
        if (not isinstance(targets, list) or not targets
                or any(not isinstance(t, str) for t in targets)
                or config.get('modules_to_save')
                or any('embed' in t.lower() or 'lm_head' in t.lower() for t in targets)
                or not (directory/'adapter_model.safetensors').is_file()
                or selector._adapter_has_embedding_delta(directory)):
            raise ValueError(f'{aid}: native backbone layout cannot discard embedding/saved-module deltas')
        config_shas[aid] = sha(config_path.read_bytes())
    return dict(disable_lora_embeddings=True, inspected_adapters=len(adapter_map),
                selector_path=str(helper), selector_sha256=sha(helper.read_bytes()),
                config_sha256=config_shas, basis='complete_current_pool_configs_and_tensor_headers',
                independent_numerical_correctness=False)


def pre_ready_ingress_options(http_cfg, native_port, model, output):
    """Fail closed on opt-in deployment identity; old native path is unchanged."""
    if http_cfg is None or http_cfg.get('ingress_mode', 'native_direct_v1') == 'native_direct_v1':
        return None
    if http_cfg.get('ingress_mode') != 'service_pre_ready_v1':
        raise ValueError('unknown service ingress mode')
    endpoint = urlsplit(http_cfg['url'])
    if (endpoint.scheme != 'http' or endpoint.hostname != '127.0.0.1'
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
            or endpoint.path != '/v1/chat/completions' or not endpoint.port
            or endpoint.port == native_port or http_cfg.get('model') != model
            or type(native_port) is not int or not 0 < native_port < 65536
            or not qualified_http_population(http_cfg)):
        raise ValueError('frozen public ingress/private native endpoint contract differs')
    return dict(port=endpoint.port, upstream_url=f'http://127.0.0.1:{native_port}/v1/chat/completions',
                model=model, request_count=http_cfg['request_count'],
                journal_path=output.with_name(output.name+'.ingress.jsonl'), timeout_s=1800.)


def prepare_physical_deployment(args, guard, admission, http_cfg, context, result):
    if http_cfg is None or http_cfg.get('physical_lifecycle') is None:
        return None
    if (http_cfg['physical_lifecycle'] != 'native_store_pool_v1'
            or http_cfg.get('ingress_mode') != 'service_pre_ready_v1'):
        raise ValueError('native store-pool lifecycle requires service-owned startup admission')
    from faaslora.metrics.metrics_collector import PhysicalGPUDeployment
    plan = load_http_replay_plan(http_cfg, trace=args.trace)
    deployment = PhysicalGPUDeployment(root=args.output/'physical', plan=plan, context=context)
    deployment.bind_result_file(args.output/'model_qualification.json')
    root = args.output/'worker_receipts'
    root.mkdir(mode=0o700)
    backend_path = args.native_source/'sllm/backends/vllm_backend.py'
    store_path = args.store_package/'site-packages/sllm_store/torch.py'
    result['configuration']['backend_config']['tc_worker_audit'] = dict(root=str(root),
        service_cgroup='/'+str(Path(admission['service_identity']['path']).relative_to('/sys/fs/cgroup')),
        service_cpus=sorted(guard.SERVICE_CPUS), backend_path=str(backend_path.resolve()),
        backend_sha256=sha(backend_path.read_bytes()), store_path=str(store_path.resolve()),
        store_sha256=sha(store_path.read_bytes()),
        checkpoint_path=str((args.checkpoint_root/'vllm'/args.model_name).resolve()))
    result['physical_lifecycle'] = dict(mode='native_store_pool_v1',
        ownership='one exclusive native store/engine GPU pool; not logical instance count',
        summary_path=str(deployment.root/'physical_resource_summary.json'),
        lifecycle_qualified=False)
    return deployment


def allocate_native_pool(args, guard, admission, deployment):
    """Allocate immediately before native stack spawn, after CPU-only setup."""
    from faaslora.metrics.metrics_collector import PhysicalGPUAllocation
    api = guard.load_nvml_binding(Path(os.environ['FAASLORA_TC_NVML_BINDING']),
                                  os.environ['FAASLORA_TC_NVML_SHA256'])
    census = guard.NativeGPUCensus(api)
    try:
        return PhysicalGPUAllocation(root=deployment.allocations, census=census,
            service_path=Path(admission['service_identity']['path']), device_indices=args.gpu_ids,
            owner_identity=guard.gpu_process_identity(os.getpid()))
    except BaseException:
        census.close()
        raise


def confirm_native_pool(allocation):
    """Bind actual service contexts, including the persistent native store."""
    sample = allocation.census.sample(allocation.service_path)
    workers = []
    for device in sample['devices']:
        for process in device['processes']:
            if process['service_member'] and process['kind'] == 'compute':
                if device['gpu_uuid'] not in allocation.gpu_uuids:
                    raise ValueError('native service used an unallocated physical GPU')
                workers.append(dict(device_uuid=device['gpu_uuid'], pid=process['pid'],
                                    worker_rank=device['index']))
    allocation.confirm_workers(workers)


def release_native_pool(allocation):
    # A failed startup can have a durable allocation/spawn but no confirmed
    # workers. Do not manufacture an empty native-exit event in that state:
    # release() preserves its unqualified lease with release_deferred instead.
    if any(event['event'] == 'native_workers' for event in allocation.events):
        asyncio.run(allocation.wait_workers(timeout_s=60))
    return allocation.release()


def audit_backend_receipts(result, config, clock_id):
    """Positive live-at-ready evidence survives legitimate actor retirement."""
    sources, bindings = {}, {}
    for request in result['requests']:
        body = request.get('response')
        metrics = body.get('metrics') if isinstance(body, dict) else None
        if not isinstance(metrics, dict) or 'ieee_tc' not in metrics:
            continue  # Failed responses remain failures, not invented workers.
        control = metrics['ieee_tc']['control_observation']
        rid, expected = control.get('tc_worker_receipt_id'), control.get('tc_worker_receipt_sha256')
        if not re.fullmatch('[0-9a-f]{32}', str(rid)):
            raise ValueError('served backend has no ready-time receipt')
        path = Path(config['root'])/(rid+'.json')
        if path.is_symlink():
            raise ValueError('backend ready receipt must be an owned regular file')
        raw = path.read_bytes()
        row = json.loads(raw)
        entry = control.get('tc_backend_entry_s')
        if (sha(raw) != expected or row.get('receipt_id') != rid
                or row.get('schema') != 'ieee_tc_native_backend_ready_v1'
                or row.get('clock_id') != clock_id or control.get('tc_clock_id') != clock_id
                or row.get('cgroup') != '0::'+config['service_cgroup']
                or row.get('affinity') != config['service_cpus']
                or any(row.get(k) != config[k] for k in
                       ('backend_path', 'backend_sha256', 'store_path', 'store_sha256', 'checkpoint_path'))
                or row.get('engine_present') is not True or row.get('enable_lora') is not True
                or row.get('load_format') != 'serverless_llm'
                or type(row.get('pid')) is not int or row['pid'] <= 0
                or type(row.get('start_ticks')) is not int or row['start_ticks'] <= 0
                or type(row.get('at')) not in (int, float) or not math.isfinite(row['at'])
                or type(entry) not in (int, float) or not math.isfinite(entry)
                or not row['at'] <= entry
                or not isinstance(control.get('instance_id'), str) or not control['instance_id']):
            raise ValueError('served backend ready-time identity/source/clock differs')
        if rid in bindings and bindings[rid] != control['instance_id']:
            raise ValueError('one backend receipt bound to multiple native instances')
        bindings[rid] = control['instance_id']
        sources[rid] = dict(path=str(path), sha256=sha(raw), observation=row)
    if not sources:
        raise ValueError('no served backend has a verified ready-time receipt')
    result['backend_ready_receipts'] = list(sources.values())
    result['served_instance_ids'] = sorted(set(bindings.values()))
    result['backend_receipt_instances'] = bindings
    result['worker_evidence_scope'] = 'native ready-time source/containment; physical pool ledger proves GPU release'


def record_http_physical_terminals(deployment, replay_path):
    """Reuse exact external terminals, including failures; never infer success."""
    contracts, raw_responses = {}, {}
    with Path(replay_path).open() as handle:
        for line in handle:
            if not line.endswith('\n'):
                break  # Interrupted writer: keep measurement incomplete.
            event = json.loads(line)
            kind, rid = event['event'], event.get('request_id')
            if kind == 'request_contract':
                if rid in contracts or event['source_item_sha256'] != deployment.entries[rid].source_sha256:
                    raise ValueError('physical request contract identity differs')
                contracts[rid] = event
            elif kind == 'http_raw_response':
                if rid in raw_responses:
                    raise ValueError('duplicate raw HTTP response')
                raw_responses[rid] = event
            elif kind in ('http_response', 'http_request_failed', 'http_request_cancelled'):
                value = None
                if kind == 'http_response':
                    raw = raw_responses[rid]
                    body, prepared = raw['body'], contracts[rid]
                    native = body['metrics']['ieee_tc']
                    if (raw['status'] != 200 or event['response'].get('protocol_valid') is not True
                            or body['id'] != rid or native['native_lora_name'] != prepared['adapter_id']):
                        raise ValueError('physical terminal lacks a validated native response')
                    value = dict(success=True, request_id=rid,
                        generation_contract='fixed_length_greedy_v1', timing_contract=native['timing_contract'],
                        output_contract_match=True, output_tokens=native['native_output_tokens'],
                        completion_tokens=body['usage']['completion_tokens'],
                        requested_completion_tokens=prepared['target_tokens'], completion_token_source='vllm_token_ids',
                        adapter_id=native['native_lora_name'], instance_id=native['control_observation']['instance_id'],
                        completion_token_ids_sha256=native['completion_token_ids_sha256'],
                        canonical_prompt_sha256=prepared['canonical_prompt_sha256'])
                deployment.terminal(rid, at=event['client_completed_s'], result=value,
                    error_type=None if kind == 'http_response' else event.get('error', kind),
                    interrupted=kind == 'http_request_cancelled')


def validate_scheduler_readback(observed, *, source, service_cgroup, service_cpus):
    if (observed.get('class_name') != 'StorageAwareScheduler'
            or Path(observed.get('source_file', '')).resolve() != source.resolve()
            or observed.get('source_sha256') != sha(source.read_bytes())
            or observed.get('cgroup', '').strip() != '0::'+service_cgroup
            or observed.get('affinity') != sorted(service_cpus)
            or observed.get('running') is not True
            or observed.get('loop_present') is not True
            or observed.get('loop_done') is not False
            or observed.get('loop_cancelled') is not False
            or observed.get('loop_exception') is not None):
        raise ValueError('actual native scheduler source/containment/liveness differs')


def capture_native_scheduler(ray, args, guard, admission, result):
    measured = result.get('source_view_manifest') or {}
    if measured.get('scheduler_queue', 'original') != 'request_identity_v1':
        return  # Historical views keep their recorded scope, not invented evidence.
    names = [r for r in ray.util.list_named_actors(all_namespaces=True)
             if r['name'] == 'model_loading_scheduler']
    if len(names) != 1:
        raise ValueError('exactly one native loading scheduler required')
    actor = ray.get_actor(names[0]['name'], namespace=names[0]['namespace'])

    def observe(scheduler):
        import hashlib, os, pathlib
        import sllm.schedulers.storage_aware_scheduler as module
        task = scheduler.loop_task
        return dict(class_name=type(scheduler).__name__, pid=os.getpid(),
            source_file=module.__file__,
            source_sha256=hashlib.sha256(pathlib.Path(module.__file__).read_bytes()).hexdigest(),
            cgroup=pathlib.Path('/proc/self/cgroup').read_text(),
            affinity=sorted(os.sched_getaffinity(0)), running=scheduler.running,
            loop_present=task is not None, loop_done=None if task is None else task.done(),
            loop_cancelled=None if task is None else task.cancelled(),
            loop_exception=(repr(task.exception()) if task is not None and task.done()
                            and not task.cancelled() and task.exception() is not None else None),
            queues={name: dict(total=len(queue), done=sum(row[2].done() for row in queue))
                    for name, queue in scheduler.model_loading_queues.items()})

    observed = ray.get(actor.__ray_call__.remote(observe), timeout=30)
    result['scheduler_readback'] = dict(observed, **names[0])
    group = Path(admission['service_identity']['path'])
    validate_scheduler_readback(observed,
        source=args.native_source/'sllm/schedulers/storage_aware_scheduler.py',
        service_cgroup='/'+str(group.relative_to('/sys/fs/cgroup')),
        service_cpus=guard.SERVICE_CPUS)


def capture_native_model_workers(ray, args, guard, admission, checkpoint, package, result):
    """Retain worker/source/resource readback on failed HTTP qualification too.

    Opt-in ready receipts survive actor retirement; legacy actor lookups remain
    end-of-run-only. Neither substitutes for the separate physical pool ledger.
    """
    group = Path(admission['service_identity']['path'])
    result['owned_processes'] = guard.owned_pids(group)
    result['resource_snapshot'] = guard.cgroup_snapshot(group)
    capture_native_scheduler(ray, args, guard, admission, result)
    worker_config = result.get('configuration', {}).get('backend_config', {}).get('tc_worker_audit')
    if worker_config is not None:
        from faaslora.clock import local_monotonic_clock_id
        audit_backend_receipts(result, worker_config, local_monotonic_clock_id())
        store_log = (args.output/'store.log').read_text(errors='replace')
        result['store_confirmations'] = re.findall(r'Confirm model (\S+) replica (\S+) success', store_log)
        if not any(path == f'vllm/{args.model_name}/rank_0' for path, _ in result['store_confirmations']):
            raise ValueError('actual native GPU-load confirmation missing')
        return
    names = ray.util.list_named_actors(all_namespaces=True)
    instance_ids = set()
    for row in result['requests']:
        body = row.get('response')
        if not isinstance(body, dict):
            continue
        metrics = body.get('metrics')
        identity = metrics.get('instance_id') if isinstance(metrics, dict) else None
        if isinstance(identity, str) and identity:
            instance_ids.add(identity)
    result['served_instance_ids'] = sorted(instance_ids)
    result['model_workers'] = []
    for name in names:
        if name['name'] not in instance_ids:
            continue
        actor = ray.get_actor(name['name'], namespace=name['namespace'])

        def observe(backend):
            import hashlib, os, pathlib, ray
            import sllm.backends.vllm_backend as module
            import sllm_store.torch as store
            return dict(pid=os.getpid(), cgroup=pathlib.Path('/proc/self/cgroup').read_text(),
                affinity=sorted(os.sched_getaffinity(0)), gpu_ids=ray.get_gpu_ids(),
                cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                backend_file=module.__file__, backend_sha256=hashlib.sha256(pathlib.Path(module.__file__).read_bytes()).hexdigest(),
                store_file=store.__file__, load_format=backend.engine_args.load_format,
                model_path=backend.engine_args.model, engine_present=backend.engine is not None,
                enable_lora=backend.enable_lora)

        observed = ray.get(actor.__ray_call__.remote(observe), timeout=30)
        observed.update(name)
        result['model_workers'].append(observed)
        if (observed['cgroup'].strip() != '0::/' + str(group.relative_to('/sys/fs/cgroup'))
                or observed['affinity'] != sorted(guard.SERVICE_CPUS)
                or observed['load_format'] != 'serverless_llm'
                or Path(observed['model_path']).resolve() != checkpoint.resolve()
                or Path(observed['backend_file']).resolve() != args.native_source / 'sllm/backends/vllm_backend.py'
                or Path(observed['store_file']).resolve() != package / 'sllm_store/torch.py'
                or not observed['engine_present']):
            raise ValueError('actual native model worker identity/containment differs')
    result['missing_served_instance_ids'] = sorted(instance_ids - {row['name'] for row in result['model_workers']})
    if result['missing_served_instance_ids'] or not instance_ids:
        raise ValueError('actual model worker missing; end-of-run readback cannot qualify retired workers')
    store_log = (args.output / 'store.log').read_text(errors='replace')
    result['store_confirmations'] = re.findall(r'Confirm model (\S+) replica (\S+) success', store_log)
    if not any(path == f'vllm/{args.model_name}/rank_0' for path, _ in result['store_confirmations']):
        raise ValueError('actual native GPU-load confirmation missing')


def qualify_model(args) -> dict:
    """Exercise the real store -> native loader -> model -> HTTP path.

    Reuses the contained native script view and the existing external supervisor.
    No routing replacement, ordinary-load fallback, artifact conversion or
    performance claim. The loader-only overlay is restored by its existing
    installer AFTER the external supervisor confirms complete process exit.
    """
    guard = load_guard(args.main_repo)
    admission = guard.verify_current_service()
    http_cfg = None
    replay_context = None
    measured = None
    if getattr(args, 'http_replay_config', None) is not None:
        http_cfg = json.loads(args.http_replay_config.read_text())
        replay_context = json.loads(Path(os.environ['FAASLORA_TC_LAUNCH_RECEIPT']).read_text()).get('external_replay', {})
        if (replay_context.get('transport') != 'http'
                or replay_context.get('config_sha256') != sha(args.http_replay_config.read_bytes())
                or http_cfg['model'] != args.model_name or Path(http_cfg['trace']) != args.trace
                or Path(http_cfg['backbone']) != args.backbone):
            raise ValueError('service and external HTTP publisher contracts differ')
        measured = json.loads((args.native_source / 'measurement_manifest.json').read_text())
        for path, expected in measured['measured_sha256'].items():
            if sha((args.native_source/path).read_bytes()) != expected:
                raise ValueError('measured source view changed')
        if sha(Path(measured['helper_path']).read_bytes()) != measured['helper_sha256']:
            raise ValueError('native measurement helper changed')
        for path, expected in measured.get('shared_source_sha256', {}).items():
            if sha((args.main_repo/path).read_bytes()) != expected:
                raise ValueError('shared measurement/transport source changed')
        if http_cfg.get('remote_artifacts') is not None and set(measured.get('shared_source_sha256', {})) != {
                'faaslora/storage/http_artifact_store.py', 'faaslora/clock.py'}:
            raise ValueError('remote qualification requires frozen common transport source')
        if ((args.native_source/'faaslora').resolve(strict=True) !=
                (args.main_repo/'faaslora').resolve(strict=True)
                or measured.get('repository_startup_hooks') is not False
                or (args.native_source/'sitecustomize.py').exists()):
            raise ValueError('shared package view must exclude repository startup hooks')
    options = pre_ready_ingress_options(http_cfg, args.api_port, args.model_name, args.output)
    if options is None:
        ingress_context = nullcontext(None)
    else:
        # Loading this dependency-light module does not import Ray, vLLM or
        # transformers. Open service admission before expensive native setup.
        helper = ROOT/'scripts/ieee_tc_serverless_measurement.py'
        if helper.resolve() != Path(measured['helper_path']).resolve():
            raise ValueError('ingress and measured native helper identities differ')
        spec = importlib.util.spec_from_file_location('tc_service_ingress', helper)
        support = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(support)
        ingress_context = support.PreReadyHTTPIngress(**options)
    with ingress_context as ingress:
        return _qualify_model(args, guard, admission, http_cfg, replay_context, measured, ingress)


def _qualify_model(args, guard, admission, http_cfg, replay_context, measured, ingress) -> dict:
    """Native model lifecycle inside the already-admitted service domain."""
    if Path(sys.executable).resolve() != (args.environment / 'bin/python').resolve():
        raise ValueError('wrong native model interpreter')
    receipt = json.loads(args.overlay_receipt.read_text())
    if (receipt.get('status') != 'INSTALLED' or len(receipt['members']) != 7
            or Path(receipt['vllm_root']).resolve() !=
            (args.environment / 'lib/python3.12/site-packages/vllm').resolve()):
        raise ValueError('explicit installed loader-only receipt required')
    for row in receipt['members']:
        if sha((Path(receipt['vllm_root']) / row['path']).read_bytes()) != row['after_sha256']:
            raise ValueError('installed loader-only source changed')
    checkpoint = args.checkpoint_root / 'vllm' / args.model_name
    if not (checkpoint / 'rank_0/tensor.data_0').is_file():
        raise ValueError('existing native checkpoint required; no conversion permitted')
    # Ray reconstructs named-actor method metadata in this diagnostic driver.
    # Its imports must select the SAME source/library composition as workers;
    # subprocess-only PYTHONPATH is insufficient for that reconstruction.
    package = args.store_package / 'site-packages'
    extra_path = str(package)
    source_path = f'{args.native_source}:{extra_path}'
    if (os.environ.get('PYTHONPATH') != source_path
            or os.environ.get('LD_LIBRARY_PATH') != str(package / 'sllm_store')):
        raise ValueError('native composition must be selected before diagnostic interpreter startup')
    import sllm.backends.vllm_backend as selected_backend
    import sllm_store.torch as selected_store
    if (Path(selected_backend.__file__).resolve() != args.native_source / 'sllm/backends/vllm_backend.py'
            or Path(selected_store.__file__).resolve() != package / 'sllm_store/torch.py'):
        raise ValueError('diagnostic driver native source composition differs')
    manifest = prepare(args.output, args.private_root, args.main_repo, args.gpu_ids)
    result = dict(schema='ieee_tc_serverless_native_model_qualification_v1', passed=False,
                  service=admission, qualification_only=True, lora_correctness_qualified=False,
                  performance_run_authorized=False, overlay_receipt_sha256=sha(args.overlay_receipt.read_bytes()),
                  requests=[], configuration=native_model_config(checkpoint, args.backbone,
                      min_instances=args.min_instances, max_instances=args.max_instances,
                      target=args.target, keep_alive=args.keep_alive,
                      enforce_eager=args.enforce_eager),
                  trace_path=str(args.trace), trace_sha256=sha(args.trace.read_bytes()),
                  launcher_manifest_sha256=sha((args.output / 'launch_manifest.json').read_bytes()))
    if http_cfg:
        index = json.loads(Path(http_cfg['pool_index']).read_text())
        if sha(Path(http_cfg['pool_index']).read_bytes()) != http_cfg['pool_index_sha256']:
            raise ValueError('existing pool index changed')
        pool = Path(index['provenance']['pool_root'])
        adapter_map = {row['id']: str(pool/row['id']) for row in index['artifacts']}
        if len(adapter_map) != 500 or any(not Path(p).is_dir() for p in adapter_map.values()):
            raise ValueError('full existing pool required; no replacement or download')
        embedding_policy = existing_pool_embedding_policy(adapter_map)
        result['embedding_layout_qualification'] = embedding_policy
        remote_config, remote_map = prepare_remote_artifacts(http_cfg, args.private_root,
            '/'+str(Path(admission['service_identity']['path']).relative_to('/sys/fs/cgroup')))
        if remote_config is not None:
            if set(remote_map) != set(adapter_map):
                raise ValueError('local metadata audit and remote content identity differ')
            adapter_map = remote_map
            result['configuration']['backend_config'].update(tc_remote_artifacts=remote_config,
                skip_store_lora_registration=True)
            # This PEFT transport adaptation does not disable native backbone
            # store registration/fast loading or change RR/scaler policies.
            if result['configuration']['backend_config']['skip_store_model_registration']:
                raise ValueError('remote LoRA adapter must preserve native backbone loading')
        result['configuration']['backend_config'].update(
            tc_native_measurement=True, enable_lora=True, require_lora_for_inference=True,
            lora_adapters=adapter_map, max_loras=4, max_cpu_loras=4, max_lora_rank=64,
            disable_log_stats=False,
            disable_lora_embeddings=embedding_policy['disable_lora_embeddings'])
        result.update(external_http_replay=replay_context, source_view_manifest=measured,
                      artifact_source=(remote_config['mode'] if remote_config else
                          'existing_local_pool_mechanical_qualification_only'),
                      remote_qualified=False, polling_comparison_completed=False)
        result['ingress'] = dict(mode=http_cfg.get('ingress_mode', 'native_direct_v1'),
            service_owned=True, business_t0_shifted=False, hidden_retries=False,
            release_condition='model_router_constructed_not_engine_warm' if ingress else None,
            journal_path=str(ingress.journal_path) if ingress else None)
    env = dict(os.environ)
    for key in ('TMUX', 'TMUX_PANE', 'SLLM_HEAD_RAY_BIN', 'SLLM_WORKER_RAY_BIN', 'SLLM_HEAD_PYTHON_BIN'):
        env.pop(key, None)
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env[key] = ''
    env.update(SLLM_HEAD_ENV_PREFIX=str(args.environment), SLLM_WORKER_ENV_PREFIX=str(args.environment),
               SLLM_STORE_ENV_PREFIX=str(args.environment), SLLM_REPO_ROOT=str(args.native_source),
               SLLM_EXTRA_PYTHONPATH=extra_path, SLLM_STORE_BIN=str(package / 'bin/sllm-store'),
               LD_LIBRARY_PATH=str(package / 'sllm_store'), PYTHONPATH=source_path,
               SLLM_SKIP_CONFIRM_MODEL_LOADED='0', SLLM_DIRECT_PATH_MODE='0',
               SLLM_RAY_HEAD_HOST=args.host, SLLM_RAY_PORT=str(args.ray_port), SLLM_PORT=str(args.api_port),
               SLLM_HOST='127.0.0.1', SLLM_STORE_PATH=str(args.checkpoint_root),
               SLLM_RAY_HEAD_ADDRESS=f'{args.host}:{args.ray_port}', RAY_ADDRESS=f'{args.host}:{args.ray_port}',
               SLLM_HEAD_SESSION='sllm_head_formal', SLLM_WORKER_SESSION_PREFIX='sllm_worker_formal',
               SLLM_HEAD_RESOURCES='{"control_node": 1}', SLLM_WORKER_RESOURCES='{"worker_node": 1, "worker_id_0": 1}',
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', RAY_USAGE_STATS_ENABLED='0',
               VLLM_USE_V1='1', VLLM_NO_USAGE_STATS='1', VLLM_USE_FLASHINFER_SAMPLER='0',
               RAY_TMPDIR=str(args.private_root / 'ray_tmp'), TMPDIR=str(args.private_root),
               NO_PROXY=f'{args.host},127.0.0.1,localhost', no_proxy=f'{args.host},127.0.0.1,localhost')
    # Freeze explicit native defaults rather than inheriting unrelated sessions.
    env.update(SLLM_STORE_MEM_POOL_SIZE='32GB', SLLM_STORE_NUM_THREAD='4', SLLM_STORE_CHUNK_SIZE='32MB')
    if http_cfg:
        env['SLLM_TC_MEASUREMENT'] = '1'
        env['VLLM_DISABLE_LORA_EMBEDDINGS'] = '1'
        result['embedding_layout_qualification']['native_llama_sha256'] = sha(
            (args.environment/'lib/python3.12/site-packages/vllm/model_executor/models/llama.py').read_bytes())
    result['selected_environment'] = {k: env[k] for k in ('PYTHONPATH', 'LD_LIBRARY_PATH',
        'SLLM_STORE_BIN', 'SLLM_STORE_MEM_POOL_SIZE', 'SLLM_SKIP_CONFIRM_MODEL_LOADED', 'TMPDIR')}
    os.environ.update({k: env[k] for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy',
        'https_proxy', 'all_proxy', 'NO_PROXY', 'no_proxy', 'RAY_TMPDIR')})
    tmux = ['tmux', '-f', str(args.output / 'tmux.conf'), '-S', str(args.private_root / 'tmux.sock')]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def post(endpoint, body, timeout=1800):
        request = urllib.request.Request(f'http://127.0.0.1:{args.api_port}{endpoint}',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)

    failure, registered = None, False
    deployment = allocation = native_startup = None
    import ray
    # These spans share the request publisher's host monotonic clock. Pair each
    # with a wall reading so coarse native wall logs are not silently treated
    # as precise monotonic observations. Preserve the journal even on failure.
    def startup_event(event):
        observation = dict(event=event, monotonic_s=time.perf_counter(), wall_time_s=time.time())
        with (args.output / 'startup_events.jsonl').open('a') as handle:
            handle.write(json.dumps(observation) + '\n')

    try:
        deployment = prepare_physical_deployment(args, guard, admission, http_cfg, replay_context, result)
        with (args.output / 'configuration.json').open('x') as handle:
            json.dump(result['configuration'], handle, indent=2)
        with (args.output / 'startup.log').open('x') as log:
            startup_event('native_stack_start')
            command = ['bash', str(args.output / 'start_serverlessllm_stack.sh')]
            if deployment is None:
                startup = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            else:
                allocation = allocate_native_pool(args, guard, admission, deployment)
                native_startup = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                allocation.bind_process(native_startup, guard.gpu_process_identity(native_startup.pid))
                native_startup.wait(timeout=600)
                startup = native_startup
            startup_event('native_stack_return')
        result['startup_returncode'] = startup.returncode
        if startup.returncode:
            raise RuntimeError('native stack startup failed; inspect preserved logs')
        if allocation is not None:
            # Store readiness precedes controller/model registration. It is
            # actual multi-card CUDA ownership, not a count of ready replicas.
            confirm_native_pool(allocation)
        ray.init(address=f'{args.host}:{args.ray_port}', log_to_driver=False)
        result['nodes'] = ray.nodes()
        validate_ray_nodes(result['nodes'], len(args.gpu_ids))
        startup_event('model_registration_start')
        result['registration'] = post('/register', result['configuration'])
        startup_event('model_registration_return')
        registered = True
        # Registration enqueues router construction; wait for its actual startup
        # notification, not a sacrificial inference/prewarm request.
        deadline = time.monotonic() + 600
        while f'Started handler for model {args.model_name}' not in (args.output / 'serve.log').read_text(errors='replace'):
            if time.monotonic() >= deadline:
                raise TimeoutError('native router construction did not complete')
            time.sleep(1)
        startup_event('native_router_start_observed')
        if ingress is not None:
            ingress.router_ready()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.backbone, local_files_only=True) if not http_cfg else None
        source_requests = json.loads(args.trace.read_text())['requests'][:4] if not http_cfg else []
        for row in source_requests:
            prompt = '\n'.join(f"{m['role']}: {m['content']}" for m in row['body']['messages'])
            tokens = tokenizer.encode(prompt, add_special_tokens=True)[:759]
            target = min(row['expected_output_tokens'], 256)
            request = dict(model=args.model_name, request_id=row['request_id'], input_tokens=tokens,
                           max_tokens=target, temperature=0, top_p=1, ignore_eos=True, stream=False)
            observation = dict(request_id=row['request_id'], source_adapter_id=row['adapter_id'],
                adapter_applied=False, target_tokens=target, input_tokens=tokens,
                input_sha256=sha(json.dumps(tokens).encode()), started_monotonic=time.monotonic())
            result['requests'].append(observation)
            observation['response'] = post('/v1/chat/completions', request)
            observation['finished_monotonic'] = time.monotonic()
            validate_native_response(observation['response'], row['request_id'], target, len(tokens))
        if http_cfg:
            # Arrivals started at the supervisor's fixed notice+60, independent
            # of readiness. This service never starts or paces the HTTP client.
            frozen = load_http_replay_plan(http_cfg, trace=args.trace)
            deadline = replay_context['replay_t0_s']+frozen.entries[-1].offset_s+1800+5
            completed = None
            with Path(replay_context['result_path']).open() as log:
                while completed is None:
                    position = log.tell()
                    line = log.readline()
                    if not line.endswith('\n'):
                        log.seek(position)
                        if time.perf_counter() >= deadline:
                            raise TimeoutError('external HTTP replay terminal record absent')
                        time.sleep(.1)
                        continue
                    event = json.loads(line)
                    if event['event'] == 'http_raw_response':
                        result['requests'].append(dict(request_id=event['request_id'], response=event['body'],
                                                       http_status=event['status']))
                    if event['event'] in ('http_replay_complete', 'http_replay_incomplete'):
                        completed = event
            result['http_completion'] = completed
            if (completed['event'] != 'http_replay_complete' or completed.get('N_failed') != 0
                    or completed.get('N_response') != http_cfg['request_count']):
                raise ValueError('external HTTP qualification failed; do not label binding as correctness')
        result['passed'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        failure = exc
    finally:
        if allocation is not None and allocation.process is not None:
            try:
                confirm_native_pool(allocation)
            except Exception as exc:
                result['physical_confirmation_error'] = f'{type(exc).__name__}: {exc}'
                result['passed'] = False
                failure = failure or exc
        if ray.is_initialized():
            try:
                capture_native_model_workers(ray, args, guard, admission, checkpoint, package, result)
            except Exception as exc:
                result['worker_audit_error'] = f'{type(exc).__name__}: {exc}'
                result['passed'] = False
                failure = failure or exc
        if registered:
            try:
                result['delete_response'] = post('/delete', {'model': args.model_name}, timeout=60)
            except Exception as exc:
                result['delete_error'] = f'{type(exc).__name__}: {exc}'
                result['passed'] = False
                failure = failure or exc
        if ray.is_initialized():
            ray.shutdown()
        for session in ('sllm_head_formal', 'sllm_worker_formal_0', 'sllm_store_formal', 'sllm_serve_formal'):
            captured = subprocess.run(tmux + ['capture-pane', '-pJt', session, '-S', '-'],
                                     capture_output=True, text=True, timeout=5)
            with (args.output / f'{session}.log').open('x') as handle:
                handle.write(captured.stdout + captured.stderr)
        result['private_tmux_stop_returncode'] = subprocess.run(tmux + ['kill-server'], capture_output=True,
                                                               text=True, timeout=5).returncode
        if native_startup is not None and native_startup.poll() is None:
            native_startup.terminate()  # Exact Popen, never a name-based kill.
            try:
                native_startup.wait(timeout=5)
            except subprocess.TimeoutExpired:
                native_startup.kill()
                native_startup.wait(timeout=5)
        if allocation is not None:
            try:
                release_native_pool(allocation)  # Worker exit AND fresh native census.
            except Exception as exc:
                result['physical_release_error'] = f'{type(exc).__name__}: {exc}'
                result['passed'] = False
                failure = failure or exc
            result['physical_allocation'] = allocation.evidence()
        if deployment is not None:
            try:
                record_http_physical_terminals(deployment, replay_context['result_path'])
                physical = deployment.summarize(observed_until_s=time.monotonic())
                result['physical_lifecycle']['lifecycle_qualified'] = physical['measurement_complete'] is True
                if not physical['measurement_complete']:
                    raise ValueError('physical lifecycle remains incomplete; no finite full-run resource score')
            except Exception as exc:
                result['physical_measurement_error'] = f'{type(exc).__name__}: {exc}'
                result['passed'] = False
                failure = failure or exc
        result['external_cleanup_required'] = True
        with (args.output / 'model_qualification.json').open('x') as handle:
            json.dump(result, handle, indent=2)
        if deployment is not None:
            deployment.finalize()  # Binds the immutable result SHA, including failures.
    if failure:
        raise RuntimeError('native model qualification failed; original evidence preserved') from failure
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    source = sub.add_parser('prepare-measurement-view')
    source.add_argument('--native-source', type=Path, required=True)
    source.add_argument('--output', type=Path, required=True)
    source.add_argument('--variant', choices=('original', 'repaired'), required=True)
    source.add_argument('--scheduler-queue', choices=('original', 'request_identity_v1'),
                        default='original')
    source.add_argument('--main-repo', type=Path, required=True)
    replay = sub.add_parser('http-replay')
    replay.add_argument('--config', type=Path, required=True)
    replay.add_argument('--output', type=Path, required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--private-root", type=Path, required=True)
    prep.add_argument("--main-repo", type=Path, required=True)
    prep.add_argument("--gpu-ids", required=True)
    check = sub.add_parser("verify")
    check.add_argument("--manifest", type=Path, required=True)
    checkpoint = sub.add_parser('audit-checkpoint')
    checkpoint.add_argument('--checkpoint', type=Path, required=True)
    checkpoint.add_argument('--backbone', type=Path, required=True)
    checkpoint.add_argument('--output', type=Path, required=True)
    export = sub.add_parser('export-checkpoint')
    for name in ('output', 'main-repo', 'environment', 'native-source', 'checkpoint-root',
                 'backbone', 'store-package', 'overlay-receipt'):
        export.add_argument('--' + name, type=Path, required=True)
    export.add_argument('--model-name', required=True)
    export.add_argument('--gpu-id', type=int, choices=range(4), required=True)
    witness = sub.add_parser('qualify-ray')
    for name in ('output', 'private-root', 'main-repo', 'environment', 'native-source', 'checkpoint-root'):
        witness.add_argument('--' + name, type=Path, required=True)
    witness.add_argument('--gpu-ids', required=True)
    witness.add_argument('--host', required=True)
    witness.add_argument('--ray-port', required=True, type=int)
    witness.add_argument('--api-port', required=True, type=int)
    model = sub.add_parser('qualify-model')
    for name in ('output', 'private-root', 'main-repo', 'environment', 'native-source',
                 'checkpoint-root', 'backbone', 'trace', 'store-package', 'overlay-receipt'):
        model.add_argument('--' + name, type=Path, required=True)
    for name in ('gpu-ids', 'host', 'model-name'):
        model.add_argument('--' + name, required=True)
    for name in ('ray-port', 'api-port'):
        model.add_argument('--' + name, required=True, type=int)
    model.add_argument('--http-replay-config', type=Path)
    model.add_argument('--min-instances', type=int, default=1)
    model.add_argument('--max-instances', type=int, default=1)
    model.add_argument('--target', type=int, default=1)
    model.add_argument('--keep-alive', type=int, default=10)
    model.add_argument('--enforce-eager', action=argparse.BooleanOptionalAction, default=True,
                       help='Historical default is eager; --no-enforce-eager explicitly tests native CUDA graphs.')
    args = parser.parse_args()
    if args.action == 'http-replay':
        http_replay(args)
        return
    if args.action == 'prepare-measurement-view':
        print(json.dumps(prepare_measurement_view(args.native_source, args.output, args.variant,
                                                 args.main_repo, args.scheduler_queue), indent=2))
        return
    if args.action == 'export-checkpoint':
        print(json.dumps(export_checkpoint(args), indent=2))
        return
    if args.action == 'audit-checkpoint':
        # Reserve output before the read; a failed attempt remains visible.
        with args.output.open('x') as handle:
            try:
                result = audit_checkpoint(args.checkpoint, args.backbone)
            except Exception as exc:
                json.dump(dict(passed=False, error=f'{type(exc).__name__}: {exc}',
                               performance_run_authorized=False), handle, indent=2)
                raise
            json.dump(result, handle, indent=2)
        print(json.dumps({k: v for k, v in result.items() if k != 'tensors'}, indent=2))
        return
    if args.action in ('qualify-ray', 'qualify-model'):
        args.gpu_ids = tuple(int(i) for i in args.gpu_ids.split(','))
        print(json.dumps((qualify_ray if args.action == 'qualify-ray' else qualify_model)(args), indent=2))
        return
    result = (prepare(args.output, args.private_root, args.main_repo,
                      tuple(int(i) for i in args.gpu_ids.split(','))) if args.action == "prepare"
              else verify(args.manifest))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
