"""Native observations and an explicitly opt-in service startup HTTP ingress.

The upstream 0.10.2 V1 frontend computes engine-core timestamps but omits them
from RequestOutput. Snapshot those existing scalar stats before the asynchronous
consumer can see later mutations. No scheduling, loading or sampling change.
The ingress is a disclosed deployment adapter, never installed in legacy runs.
"""
import copy
import asyncio
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path
import threading
import time
import uuid
from urllib.parse import urlsplit

OUTPUT_PROCESSOR_SHA256 = '50f5e0aa5d0b7ece086632de6e37376dbeba0a1c0d2f22f95b6759189e76e04d'
V030_NATIVE_SOURCES = {
    'v1/engine/output_processor.py': 'e47b86cd69ce1c1e3f53dac655eaa75c32a4057d10f4a1db16bb83a397f2a329',
    'outputs.py': '346d1f9204a441867efc3af7c5a99d8a70ef0f69fe607dfa8a4a2557a82ca425',
    'v1/metrics/stats.py': 'af1983ddc32fa4d3284e72212c6525daf961dd84429023a02d612b68b3112cf3',
}


def configure_device_compile_cache(backend_config):
    """Give independent TP1 engines distinct, persistent native cache roots.

    vLLM 0.10.2 includes rank but not the assigned physical device in its cache
    path. PyTorch 2.8 unpacked saves replace that directory non-atomically.
    Native model/config/code hashing remains below this root. Never clear the
    shared cache, disable compilation, or make a fresh cache for each run.
    Assumes the native scheduler's exclusive one-engine-per-full-GPU allocation.
    """
    root = Path(backend_config['tc_device_compile_cache_root'])
    device = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (not root.is_absolute() or '..' in root.parts
            or backend_config.get('tensor_parallel_size') != 1
            or not re.fullmatch(r'(?:0|[1-9][0-9]*|GPU-[0-9a-fA-F-]+)', device)):
        raise ValueError('device compile cache requires an absolute root and one assigned TP1 GPU')
    hostname = os.uname().nodename
    host_key = hashlib.sha256(hostname.encode()).hexdigest()[:16]
    selected = root/host_key/('cuda-'+device)
    os.environ['VLLM_CACHE_ROOT'] = str(selected)
    return dict(contract='physical_device_compile_cache_v1', root=str(root),
        hostname=hostname, cuda_visible=device, vllm_cache_root=str(selected),
        native_config_code_hash_preserved=True, run_specific=False,
        shared_cache_cleared=False, compilation_disabled=False)


def configure_verified_shared_tokenizer(backend):
    """Intern byte-identical published tokenizers before native V1 serving.

    vLLM 0.10.2's TokenizerGroup flag controls only tokenizer lookup, not
    LoRARequest/weight execution. Both input and output use the same group.
    This opt-in contract supports the audited JSON-tokenizer artifact layout;
    heterogeneous/custom tokenizers fail rather than falling back to base.
    The normal resolver still downloads/verifies every requested artifact.
    No artifact prefetch, weight mutation, cache eviction, or periodic GC.
    """
    config = backend.backend_config
    if config.get('tc_tokenizer_contract') != 'verified_published_base_v1':
        raise ValueError('explicit verified published tokenizer contract required')
    engine, resolver = backend.engine, backend._tc_artifact_resolver
    group = engine.tokenizer
    if (resolver is None or not backend.enable_lora or not backend.engine_args.enable_lora
            or engine.vllm_config.lora_config is None or group is None
            or not group.enable_lora or len(group.lora_tokenizers)
            or engine.processor.tokenizer is not group
            or engine.processor.input_preprocessor.tokenizer is not group
            or engine.output_processor.tokenizer is not group):
        raise ValueError('shared tokenizer requires an unused native LoRA-enabled V1 frontend')
    names = ('config.json', 'generation_config.json', 'special_tokens_map.json',
             'tokenizer.json', 'tokenizer_config.json')
    payload_names = {'adapter_config.json', 'adapter_model.safetensors', 'adapter_data.bin'}
    root = Path(group.tokenizer_id)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError('actual engine tokenizer must have a local immutable source')
    # Unsupported alternative assets could change AutoTokenizer selection.
    extra = {'tokenizer.model', 'added_tokens.json', 'vocab.json', 'vocab.txt', 'merges.txt'}
    if any((root/name).exists() for name in extra):
        raise ValueError('tokenizer source has assets outside the audited JSON layout')
    canonical = {}
    for name in names:
        raw = (root/name).read_bytes()
        canonical[name] = (len(raw), hashlib.sha256(raw).hexdigest())
    adapter_map = config.get('lora_adapters', {})
    if not adapter_map:
        raise ValueError('explicit published adapter mapping required')
    manifests = resolver.client.preparation_manifests(tuple(sorted(adapter_map)))
    for adapter_id, files in manifests.items():
        if (set(files) != set(names) | payload_names
                or any(files[name] != canonical[name] for name in names)
                or adapter_map[adapter_id] != str(resolver.root/'objects'/adapter_id)):
            raise ValueError('published tokenizer differs from actual engine tokenizer: '+adapter_id)
    # All validation precedes the only mutation. Do NOT change backend.enable_lora,
    # engine_args.enable_lora, vllm_config.lora_config, or native LoRA requests.
    group.enable_lora = False
    return dict(contract='verified_published_base_v1', tokenizer_id=str(root),
        adapter_count=len(manifests),
        content_manifest_sha256=resolver.client.content_manifest_sha256,
        content_manifest_file_sha256=resolver.config['content_manifest_file_sha256'],
        tokenizer_files={name: dict(size_bytes=size, sha256=digest)
                         for name, (size, digest) in canonical.items()},
        tokenizer_group_shared_by_input_output=True, lora_weights_enabled=True,
        remote_resolver_preserved=True, per_adapter_tokenizer_construction=False)


def record_backend_ready(backend, module_file):
    """Durable live identity before serving, not an actor-exit/GPU-release claim."""
    config = backend.backend_config.get('tc_worker_audit')
    if config is None:
        return None
    import sllm_store.torch as store
    from faaslora.clock import local_monotonic_clock_id
    identity = dict(pid=os.getpid(),
        start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
        cgroup=Path('/proc/self/cgroup').read_text().strip(),
        affinity=sorted(os.sched_getaffinity(0)))
    backend_path, store_path = Path(module_file).resolve(), Path(store.__file__).resolve()
    backend_sha = hashlib.sha256(backend_path.read_bytes()).hexdigest()
    store_sha = hashlib.sha256(store_path.read_bytes()).hexdigest()
    if (identity['cgroup'] != '0::'+config['service_cgroup']
            or identity['affinity'] != config['service_cpus']
            or str(backend_path) != config['backend_path'] or backend_sha != config['backend_sha256']
            or str(store_path) != config['store_path'] or store_sha != config['store_sha256']
            or backend.engine_args.load_format != 'serverless_llm'
            or str(Path(backend.engine_args.model).resolve()) != config['checkpoint_path']
            or backend.engine is None):
        raise ValueError('actual backend ready identity differs from the frozen native service')
    receipt_id = uuid.uuid4().hex
    row = dict(schema='ieee_tc_native_backend_ready_v1', receipt_id=receipt_id,
        at=time.perf_counter(), clock_id=local_monotonic_clock_id(), **identity,
        backend_path=str(backend_path), backend_sha256=backend_sha,
        store_path=str(store_path), store_sha256=store_sha,
        checkpoint_path=config['checkpoint_path'], load_format=backend.engine_args.load_format,
        cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'), engine_present=True,
        enable_lora=backend.enable_lora)
    if hasattr(backend, '_tc_compile_cache'):
        row['compile_cache'] = backend._tc_compile_cache
    if hasattr(backend, '_tc_shared_tokenizer'):
        row['shared_tokenizer'] = backend._tc_shared_tokenizer
    raw = json.dumps(row, sort_keys=True, separators=(',', ':')).encode()
    with (Path(config['root'])/(receipt_id+'.json')).open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return dict(tc_worker_receipt_id=receipt_id,
                tc_worker_receipt_sha256=hashlib.sha256(raw).hexdigest())


def published_artifact_client(config):
    """Validate static delivery metadata only; never fetch from the publisher."""
    from faaslora.storage.http_artifact_store import HttpArtifactStoreClient
    endpoint = urlsplit(config['endpoint'])
    if (config.get('mode') != 'published_gzip_ondemand_peft_v1'
            or endpoint.scheme not in ('http', 'https') or not endpoint.hostname
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
            or not 0 < config['timeout_s'] < float('inf')):
        raise ValueError('explicit published artifact delivery contract required')
    raw = Path(config['content_manifest']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config['content_manifest_file_sha256']:
        raise ValueError('frozen artifact index changed')
    client = HttpArtifactStoreClient(endpoint=config['endpoint'],
        token_env=config['token_env'], timeout_s=config['timeout_s'],
        use_env_proxy=False, required_delivery_mode='prepublished_gzip_v1')
    client.configure_content_manifest(json.loads(raw))
    return client


class PublishedArtifactResolver:
    """Generic on-demand PEFT transport in the service, not a cache planner.

    Reuses the common strict downloader; no local-pool fallback, prediction,
    prefetch, eviction or scheduler changes. A per-object interprocess lock
    coalesces simultaneous misses across native actors. Unrelated objects do
    not share a lock. Receipts prove verified publication, not numerical LoRA
    correctness. The launcher creates a fresh run-local cache before bootstrap.
    """

    def __init__(self, config):
        self.config = dict(config)
        self._check_service()
        self.client = published_artifact_client(config)
        self.root = Path(config['cache_dir'])
        if (not self.root.is_absolute() or self.root.is_symlink()
                or any(not (self.root/p).is_dir() or (self.root/p).is_symlink()
                       for p in ('objects', 'receipts', 'locks', 'journals'))
                or json.loads((self.root/'run_contract.json').read_text()) != config):
            raise ValueError('fresh service-owned artifact cache contract missing')
        self.journal = self.root/'journals'/f'{os.getpid()}-{uuid.uuid4().hex}.jsonl'
        self.journal.touch(exist_ok=False)
        self._journal_lock = threading.Lock()
        self._emit('resolver_opened', cgroup=Path('/proc/self/cgroup').read_text(),
                   affinity=sorted(os.sched_getaffinity(0)))

    def _check_service(self):
        expected = self.config['service_cgroup']
        actual = Path('/proc/self/cgroup').read_text().strip()
        if (not isinstance(expected, str) or not expected.startswith('/')
                or expected == '/' or '..' in Path(expected).parts
                or not (actual == '0::'+expected or actual.startswith('0::'+expected+'/'))):
            raise ValueError('artifact resolver must execute within its admitted service cgroup')

    def _emit(self, event, **fields):
        with self._journal_lock, self.journal.open('a') as handle:
            handle.write(json.dumps(dict(event=event, pid=os.getpid(),
                monotonic_s=time.perf_counter(), **fields), separators=(',', ':'))+'\n')

    @staticmethod
    def _file_identity(target, files):
        # Verified immutable files are stat-checked on reuse, not rehashed for
        # every request. This is an integrity guard in an owned run directory,
        # not protection against a malicious actor who can forge stat metadata.
        if target.is_symlink() or not target.is_dir():
            raise ValueError('unverified artifact directory')
        found = set()
        for path in target.rglob('*'):
            if path.is_symlink():
                raise ValueError('artifact cache contains a symlink')
            if path.is_file():
                found.add(path.relative_to(target).as_posix())
        if found != set(files):
            raise ValueError('artifact cache file set changed')
        identities = {}
        for name, (size, _) in files.items():
            stat = (target/name).stat()
            if stat.st_size != size:
                raise ValueError('artifact cache file size changed')
            identities[name] = [stat.st_dev, stat.st_ino, stat.st_size,
                                stat.st_mtime_ns, stat.st_ctime_ns]
        return identities

    def _resolve(self, adapter_id, path, request_id, cancelled):
        from faaslora.storage.http_artifact_store import preparation_content_sha256
        self._check_service()
        # FrozenPreparationDescriptions validates each ID before path use.
        files = self.client.preparation_manifests((adapter_id,))[adapter_id]
        target = self.root/'objects'/adapter_id
        if str(target) != path or not isinstance(request_id, str) or not request_id:
            raise ValueError('native request/path differs from frozen artifact mapping')
        start = time.perf_counter()
        receipt_path = self.root/'receipts'/f'{adapter_id}.json'
        content_sha = preparation_content_sha256(files)
        self._emit('resolve_started', request_id=request_id, adapter_id=adapter_id)
        evidence = {}
        try:
            with (self.root/'locks'/f'{adapter_id}.lock').open('a+b') as lock:
                # Do not occupy the native async event loop while another actor
                # downloads. 50 ms is only cancellation/lock wakeup granularity,
                # not a service policy, transfer throttle or request retry.
                while True:
                    if cancelled.is_set():
                        raise InterruptedError('artifact resolution cancelled')
                    if time.perf_counter()-start >= self.config['timeout_s']:
                        raise TimeoutError('artifact publication lock deadline')
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        cancelled.wait(.05)
                cache_hit = receipt_path.exists()
                if cache_hit:
                    receipt_raw = receipt_path.read_bytes()
                    receipt = json.loads(receipt_raw)
                    if (receipt.get('adapter_id') != adapter_id
                            or receipt.get('content_sha256') != content_sha
                            or receipt.get('content_manifest_sha256') != self.client.content_manifest_sha256
                            or receipt.get('files') != self._file_identity(target, files)
                            or receipt.get('evidence', {}).get('content_verified') is not True
                            or receipt['evidence'].get('remote_pack_performed') is not False
                            or receipt['evidence'].get('remote_delivery_mode') != 'prepublished_gzip_v1'):
                        raise ValueError('artifact publication receipt or cached content changed')
                else:
                    if target.exists() or target.is_symlink():
                        raise ValueError('bare artifact path has no verified remote receipt')
                    ok, _, _ = self.client.download_artifact(adapter_id, str(target),
                        require_content_manifest=True, require_remote_timing=True,
                        evidence=evidence, cancel_event=cancelled)
                    if (not ok or evidence.get('content_verified') is not True
                            or evidence.get('remote_pack_performed') is not False
                            or evidence.get('state') != 'published'):
                        raise ValueError('unverified remote artifact publication')
                    receipt = dict(adapter_id=adapter_id, content_sha256=content_sha,
                        content_manifest_sha256=self.client.content_manifest_sha256,
                        files=self._file_identity(target, files), evidence=evidence)
                    receipt_raw = json.dumps(receipt, sort_keys=True, separators=(',', ':')).encode()
                    with receipt_path.open('xb') as handle:
                        handle.write(receipt_raw)
                # Receipt and object are published under the same lock. A
                # process crash can leave an untrusted path, never a fake hit.
            result = dict(tc_artifact_request_id=request_id, tc_artifact_id=adapter_id,
                tc_artifact_content_sha256=content_sha,
                tc_artifact_manifest_sha256=self.client.content_manifest_sha256,
                tc_artifact_receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
                tc_artifact_transfer_id=receipt['evidence']['http_transfer_id'],
                tc_artifact_cache_hit=cache_hit,
                tc_artifact_transferred_bytes=0 if cache_hit else evidence['transferred_bytes'],
                tc_artifact_delivery_mode='prepublished_gzip_v1',
                tc_artifact_resolve_started_s=start,
                tc_artifact_resolve_completed_s=time.perf_counter())
            self._emit('resolve_completed', **result)
            return result
        except BaseException as exc:
            self._emit('resolve_failed', request_id=request_id, adapter_id=adapter_id,
                       error_type=type(exc).__name__, evidence=evidence)
            raise

    async def resolve(self, adapter_id, path, request_id):
        cancelled = threading.Event()
        task = asyncio.create_task(asyncio.to_thread(self._resolve, adapter_id, path, request_id, cancelled))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            # Drain the bounded I/O worker so cancellation cannot leave an
            # invisible writer behind when service cleanup begins.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()  # retrieve a failed transfer, preserve cancellation
            raise


class PreReadyHTTPIngress:
    """Service-owned startup buffer, not a replacement scheduler or retry layer.

    Opens before native bootstrap and forwards once the model router exists,
    NOT once engines are warm. After that all capacity/scaling decisions remain
    in the native router. One async I/O thread inherits the service's cgroup
    and CPU set. No model/artifact/tokenizer work occurs in the replay client.
    """

    upstream_connection_policy = 'single_request_loopback_no_retry_v1'

    def __init__(self, *, port, upstream_url, model, request_count, journal_path,
                 timeout_s=1800.):
        upstream = urlsplit(upstream_url)
        if (type(port) is not int or not 0 <= port <= 65535
                or upstream.scheme != 'http' or upstream.hostname != '127.0.0.1'
                or upstream.username or upstream.password or not upstream.port
                or upstream.path != '/v1/chat/completions' or upstream.query or upstream.fragment
                or upstream.port == port or not isinstance(model, str) or not model
                or type(request_count) is not int or request_count < 1
                or not 0 < timeout_s < float('inf')):
            raise ValueError('explicit loopback ingress and finite workload required')
        self.port, self.upstream_url, self.model = port, upstream_url, model
        self.request_count, self.timeout_s = request_count, timeout_s
        self.journal_path = Path(journal_path)
        self._started = concurrent.futures.Future()
        self._thread = None
        self._failure = None
        self._fatal = None
        self._seen = set()
        self._router_ready = False
        self._sequence = 0

    def _emit(self, event, **fields):
        self._log.write(json.dumps(dict(event=event, monotonic_s=time.perf_counter(),
                                       **fields), separators=(',', ':'))+'\n')
        self._log.flush()

    async def _handle(self, request):
        from aiohttp import web
        from faaslora.clock import local_monotonic_clock_id
        received = time.perf_counter()
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            return web.json_response({'error': 'invalid JSON'}, status=400)
        if (not isinstance(body, dict) or body.get('model') != self.model
                or not isinstance(body.get('request_id'), str) or not body['request_id']
                or body.get('stream') is not False):
            return web.json_response({'error': 'invalid frozen ingress request'}, status=400)
        rid = body['request_id']
        if rid in self._seen:
            self._emit('duplicate_rejected', request_id=rid)
            return web.json_response({'error': 'duplicate request ID; retries not hidden'}, status=409)
        if len(self._seen) >= self.request_count:
            return web.json_response({'error': 'offered workload bound exceeded'}, status=429)
        self._seen.add(rid)
        self._sequence += 1
        self._emit('ingress_received', request_id=rid, sequence=self._sequence,
                   router_ready=self._router_ready, received_s=received,
                   request_sha256=hashlib.sha256(json.dumps(body, sort_keys=True,
                       separators=(',', ':')).encode()).hexdigest())
        forwarded = False
        try:
            async with asyncio.timeout(self.timeout_s):
                await self._ready.wait()
                if self._failure is not None:
                    self._emit('ingress_failed', request_id=rid, reason=self._failure, forwarded=False)
                    return web.json_response({'error': self._failure}, status=503)
                metrics = dict(body.get('_sllm_internal_metrics', {}) or {})
                metrics.update(tc_clock_id=local_monotonic_clock_id(),
                               tc_ingress_received_s=received,
                               tc_ingress_forwarded_s=time.perf_counter())
                body['_sllm_internal_metrics'] = metrics
                forwarded = True
                self._emit('ingress_forwarded', request_id=rid,
                           received_s=received, forwarded_s=metrics['tc_ingress_forwarded_s'])
                async with self._session.post(self.upstream_url, json=body,
                                               allow_redirects=False) as response:
                    payload = await response.read()
                    self._emit('ingress_response', request_id=rid, status=response.status)
                    return web.Response(body=payload, status=response.status,
                                        headers={'Content-Type': response.headers.get(
                                            'Content-Type', 'application/json')})
        except asyncio.CancelledError:
            self._emit('ingress_cancelled', request_id=rid, forwarded=forwarded)
            raise
        except TimeoutError:
            self._emit('ingress_timeout', request_id=rid, forwarded=forwarded)
            return web.json_response({'error': 'request protection timeout'}, status=504)
        except Exception as exc:
            self._emit('ingress_transport_failed', request_id=rid, forwarded=forwarded,
                       error_type=type(exc).__name__,
                       error_errno=(exc.errno if isinstance(exc, OSError)
                                    and type(exc.errno) is int else None))
            return web.json_response({'error': 'native transport failed; not retried'}, status=502)

    async def _serve(self):
        import aiohttp
        from aiohttp import web
        self._loop = asyncio.get_running_loop()
        self._ready, self._stop = asyncio.Event(), asyncio.Event()
        app = web.Application()
        app.router.add_post('/v1/chat/completions', self._handle)
        runner = web.AppRunner(app, access_log=None, handler_cancellation=True,
                               shutdown_timeout=1.)
        # This extra measurement ingress owns one loopback connection per POST.
        # Native idle-close and client pool reuse are otherwise unsynchronized;
        # retrying an ambiguous POST could execute inference twice. Avoid shared
        # idle connection state, not failures via retries or a tuned timeout.
        # Connection establishment remains inside measured service time/cgroup.
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0, force_close=True),
                timeout=aiohttp.ClientTimeout(total=None), trust_env=False) as self._session:
            try:
                await runner.setup()
                site = web.TCPSite(runner, '127.0.0.1', self.port)
                await site.start()
                self.port = site._server.sockets[0].getsockname()[1]
                self._emit('ingress_listening', port=self.port, pid=os.getpid(),
                           upstream_connection_policy=self.upstream_connection_policy,
                           cgroup=Path('/proc/self/cgroup').read_text(),
                           affinity=sorted(os.sched_getaffinity(0)))
                self._started.set_result(self.port)
                await self._stop.wait()
            finally:
                self._failure = self._failure or 'service shutting down'
                self._ready.set()
                await runner.cleanup()
                self._emit('ingress_closed', received_requests=len(self._seen))

    def __enter__(self):
        self._log = self.journal_path.open('x')

        def serve():
            try:
                asyncio.run(self._serve())
            except BaseException as exc:
                self._fatal = exc
                if not self._started.done():
                    self._started.set_exception(exc)
                else:
                    self._failure = f'ingress loop failed: {type(exc).__name__}'

        self._thread = threading.Thread(target=serve, name='tc-native-startup-ingress', daemon=True)
        self._thread.start()
        try:
            self._started.result(timeout=10.)
        except BaseException:
            if getattr(self, '_loop', None) is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._stop.set)
            self._thread.join(timeout=10.)
            if not self._thread.is_alive():
                self._log.close()
            raise
        return self

    def router_ready(self):
        acknowledged = concurrent.futures.Future()
        def mark():
            if self._failure is not None:
                acknowledged.set_exception(RuntimeError('failed bootstrap cannot become ready'))
                return
            self._router_ready = True
            self._emit('native_router_available')
            self._ready.set()
            acknowledged.set_result(None)
        self._loop.call_soon_threadsafe(mark)
        acknowledged.result(timeout=10.)

    def __exit__(self, exc_type, exc, traceback):
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=10.)
        if self._thread.is_alive():
            raise RuntimeError('ingress did not stop; outer service cleanup remains required')
        self._log.close()
        if self._fatal is not None and exc_type is None:
            raise RuntimeError('service ingress failed; evidence preserved') from self._fatal


def stamp(metrics, name):
    if os.environ.get('SLLM_TC_MEASUREMENT') == '1':
        from faaslora.clock import local_monotonic_clock_id
        clock_id = local_monotonic_clock_id()
        if metrics.get('tc_clock_id', clock_id) != clock_id:
            raise ValueError('cross-host or time-namespace boundary requires clock calibration')
        metrics['tc_clock_id'] = clock_id
        metrics[name] = time.perf_counter()


def snapshot_output(state, output):
    if output is not None:
        if state.stats is None:
            raise ValueError('TC native observation requires log_stats; no timestamp fallback')
        output.metrics = copy.copy(state.stats)
        output.tc_native_lora_name = state.lora_name
    return output


def snapshot_collector(collector, output):
    # RequestOutput.add merges/replaces tokens but does not propagate metrics in
    # 0.10.2. If the producer outruns its consumer, the retained object therefore
    # needs the SAME latest snapshot as its now-updated cumulative token IDs.
    if hasattr(output, 'tc_native_lora_name') and collector.output is not None:
        if collector.output.request_id != output.request_id:
            raise ValueError('native collector mixed request identities')
        collector.output.metrics = copy.copy(output.metrics)
        collector.output.tc_native_lora_name = output.tc_native_lora_name
        if hasattr(output, 'tc_native_lora_identity'):
            collector.output.tc_native_lora_identity = output.tc_native_lora_identity


def install_v1_snapshot():
    import vllm.v1.engine.output_processor as module
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != OUTPUT_PROCESSOR_SHA256:
        raise ValueError('unqualified vLLM output-processor version')
    _install_snapshot_hooks(module, snapshot_output, '_ieee_tc_snapshot_v1')


def _install_snapshot_hooks(module, snapshot, marker):
    cls = module.RequestState
    old = cls._new_request_output
    if getattr(old, marker, False):
        return

    def observed(state, *args, **kwargs):
        return snapshot(state, old(state, *args, **kwargs))

    setattr(observed, marker, True)
    cls._new_request_output = observed
    old_put = module.RequestOutputCollector.put

    def put(collector, output):
        old_put(collector, output)
        snapshot_collector(collector, output)

    module.RequestOutputCollector.put = put


def validate_v030_sources(root, version):
    """Exact installed observation boundary; no version/hash fallback."""
    if version != '0.30.0':
        raise ValueError('vLLM 0.30.0 native observation requires its qualified version')
    actual = {name: hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()
              for name in V030_NATIVE_SOURCES}
    if actual != V030_NATIVE_SOURCES:
        raise ValueError('unqualified vLLM 0.30.0 native observation sources')
    return actual


def snapshot_v030_output(state, output):
    """Copy only observations, before mutable stats/token lists advance.

    Scope is single-sequence cumulative text generation, as frozen for TC.
    Retain native integer IDs even when detokenization emits no text. This
    hook never samples, schedules, loads, reroutes or reconstructs timestamps.
    """
    if output is None:
        return None
    if (len(output.outputs) != 1 or state.output_kind.name != 'CUMULATIVE'
            or state.parent_req is not None or state.streaming_input):
        raise ValueError('TC vLLM observation requires single cumulative text output')
    output = snapshot_output(state, output)
    completion = copy.copy(output.outputs[0])
    completion.token_ids = list(completion.token_ids)
    output.outputs = [completion]
    count = len(completion.token_ids)
    if output.metrics.num_generation_tokens != count:
        raise ValueError('native metric/token count mismatch at output production')
    previous = getattr(state, '_tc_v030_token_snapshot', None)
    if previous is not None and count == previous.num_generation_tokens:
        # A terminal notification without a new token must not move the token
        # boundary, including when the consumer sees only the merged terminal.
        for name in ('queued_ts', 'scheduled_ts', 'first_token_ts', 'last_token_ts'):
            setattr(output.metrics, name, getattr(previous, name))
    if count:
        state._tc_v030_token_snapshot = copy.copy(output.metrics)
    lora = state.lora_request
    output.tc_native_lora_identity = (None if lora is None else
        (lora.lora_name, lora.lora_int_id, lora.lora_path))
    return output


def install_v030_snapshot():
    """Opt-in in-memory adapter; does not edit the installed vLLM package."""
    from importlib.metadata import version
    import vllm.v1.engine.output_processor as module
    root = Path(module.__file__).resolve().parents[2]
    sources = validate_v030_sources(root, version('vllm'))
    _install_snapshot_hooks(module, snapshot_v030_output, '_ieee_tc_snapshot_v030')
    return dict(schema='ieee_tc_vllm030_native_snapshot_v1', version='0.30.0',
                source_sha256=sources, package_root=str(root))


def dlora_worker_launch_guard():
    """Check the already-owned pool before Ray's NCCL/model initialization."""
    from replay_vllm_engine_trace import dlora_worker_context
    return dlora_worker_context(Path(os.environ['FAASLORA_TC_DLORA_WORKER_CONFIG']), actor=True)


def dlora_native_worker_receipt(worker):
    """Read actual TP1 worker state after native model, LoRA and KV allocation.

    No warm inference, CUDA synchronization, new placement, or inferred weight
    correctness. The parent binds this PID to its live NVML pool before serving.
    """
    cfg, guard = dlora_worker_launch_guard()
    from faaslora.clock import local_monotonic_clock_id
    import torch
    lora, cache = worker.lora_config, worker.cache_config
    expected = cfg['adapters']
    actual = {aid: [i, lora.adapter_paths[i]] for i, aid in enumerate(lora.adapter_ids)}
    if (worker.exec_type.name != 'LORA' or worker.model_config.use_dummy_weights
            or worker.parallel_config.tensor_parallel_size != 1
            or worker.parallel_config.pipeline_parallel_size != 1
            or worker.model_config.model != cfg['engine_args']['model']
            or actual != expected or lora.num_models != len(expected)
            or len(worker.lora_engine.cpu_lora_weights) != worker.lora_engine.num_hidden_layers
            or any(len(layer) != worker.lora_engine.lora_cnt
                   or any(weights.lora_As.shape[0] != len(expected)
                          or weights.lora_Bs.shape[0] != len(expected) for weights in layer)
                   for layer in worker.lora_engine.cpu_lora_weights)
            or lora.gpu_capacity != len(worker.lora_engine.gpu_lora_models)
            or worker.cache_engine is None or cache.num_gpu_blocks <= 0
            or torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0):
        raise ValueError('dLoRA actual native worker differs from owned full-pool contract')
    return dict(schema='ieee_tc_dlora_native_worker_v1', at=time.perf_counter(),
        clock_id=local_monotonic_clock_id(), identity=guard.gpu_process_identity(os.getpid()),
        cuda_visible=os.environ['CUDA_VISIBLE_DEVICES'], cuda_device_count=1,
        global_rank=worker.rank, tensor_parallel_size=1, pipeline_parallel_size=1,
        model=worker.model_config.model, adapter_bindings=actual,
        vocab_size=worker.model_config.hf_config.vocab_size,
        cpu_lora_layers=len(worker.lora_engine.cpu_lora_weights),
        cpu_adapter_capacity=len(expected),
        gpu_lora_models=list(worker.lora_engine.gpu_lora_models),
        actual_gpu_capacity=lora.gpu_capacity, num_gpu_blocks=cache.num_gpu_blocks,
        num_cpu_blocks=cache.num_cpu_blocks, native_torch_version=torch.__version__,
        numerical_correctness_qualified=False)


def dlora_native_adapter_bindings(manager):
    """Read the exact zero-based order passed to native PEFT loaders.

    This is a static identity map, not replica placement. EngineManager retains
    authority over routing and migration. Loaded bytes/arithmetic need their
    own evidence; an integer model ID alone does not prove either.
    """
    if (getattr(manager.exec_type, 'name', None) != 'LORA' or not manager.engines
            or type(manager.num_models) is not int or manager.num_models <= 0):
        raise ValueError('dLoRA binding requires native LORA engines and an explicit pool')
    bindings = None
    for ordinal, engine in enumerate(manager.engines):
        cfg = engine.lora_config
        ids, paths = cfg.adapter_ids, cfg.adapter_paths
        if (engine.engine_id != ordinal or cfg.num_models != manager.num_models
                or len(ids) != manager.num_models or len(paths) != len(ids)
                or any(not isinstance(a, str) or not a for a in ids)
                or len(set(ids)) != len(ids)
                or any(not isinstance(p, str) or not Path(p).is_absolute() for p in paths)):
            raise ValueError('dLoRA native loader adapter identity is incomplete')
        current = {aid: (i, paths[i]) for i, aid in enumerate(ids)}
        if bindings is not None and bindings != current:
            raise ValueError('dLoRA engines disagree on native model-ID/adapter order')
        bindings = current
    return bindings


def dlora_prepare_canonical_request(body, *, model, adapters, vocab_size, max_model_len):
    """Common frozen body -> existing native sampling/input arguments, no tokenize.

    Preserve the supplied canonical ID list instead of re-encoding its text
    using dLoRA's historical tokenizer. Return copies, so transport buffers
    cannot rewrite an enqueued request. A later native output must match them.
    """
    ids, target = body.get('input_tokens'), body.get('max_tokens')
    aid, rid = body.get('lora_adapter_name'), body.get('request_id')
    prompt = body.get('canonical_prompt')
    if (body.get('model') != model or not isinstance(rid, str) or not rid
            or not isinstance(aid, str) or aid not in adapters
            or not isinstance(prompt, str) or not prompt
            or type(vocab_size) is not int or vocab_size <= 0
            or type(max_model_len) is not int or max_model_len <= 0
            or type(ids) is not list or not ids
            or any(type(t) is not int or not 0 <= t < vocab_size for t in ids)
            or type(target) is not int or not 1 <= target <= 256
            or len(ids)+target > max_model_len
            or body.get('temperature') != 0 or body.get('top_p') != 1
            or body.get('ignore_eos') is not True or body.get('stream') is not False
            or body.get('stop') != [] or body.get('stop_token_ids') != []
            or body.get('n', 1) != 1 or body.get('best_of', 1) != 1
            or body.get('use_beam_search', False) is not False):
        raise ValueError('request differs from dLoRA canonical fixed-output contract')
    index, path = adapters[aid]
    return dict(request_id=rid, adapter_id=aid, model_id=index, adapter_path=path,
        prompt=prompt, prompt_token_ids=list(ids), target_tokens=target,
        canonical_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        prompt_token_ids_sha256=_native_ids_digest(ids),
        sampling_kwargs=dict(n=1, best_of=1, temperature=0., top_p=1.,
            max_tokens=target, ignore_eos=True, stop=[], use_beam_search=False))


async def dlora_abort_current_owner(manager, request_id):
    """Abort the current native owner, including after request migration.

    The original API captures the initially selected engine. Its tracker may
    no longer own the stream after migration. Use the same native select_lock
    as migration; do not broadcast an abort or change its placement policy.
    No owner means not submitted yet or already terminal. No global Ray stop.
    """
    async with manager.select_lock:
        owners = [engine for engine in manager.engines
                  if request_id in engine.request_tracker]
        if len(owners) > 1:
            raise ValueError('dLoRA request has multiple native stream owners')
        if owners:
            # Native _abort is synchronous and schedules cleanup in engine_step.
            # No await between ownership lookup and this mutation.
            owners[0]._abort(request_id)
            return owners[0].engine_id
        return None


def dlora_register_sequence_group(group):
    """Observe native enqueue; the existing migrated SequenceGroup owns state.

    Restricted to the shared fixed-length greedy contract. No requested token
    is manufactured, no scheduler decision is made, and no CUDA call is added.
    The native migration deepcopy carries these scalar fields and ID snapshots.
    """
    from faaslora.clock import local_monotonic_clock_id
    params, seqs = group.sampling_params, group.get_seqs()
    if (hasattr(group, '_tc_dlora') or not group.request_id
            or type(group.model_id) is not int or group.model_id < 0
            or len(seqs) != 1 or params.n != 1 or params.best_of != 1
            or params.use_beam_search or params.temperature != 0 or params.top_p != 1
            or not params.ignore_eos or params.stop
            or type(params.max_tokens) is not int or not 1 <= params.max_tokens <= 256):
        raise ValueError('dLoRA observation requires a fresh fixed-length greedy request')
    prompt_ids = list(seqs[0].data.prompt_token_ids)
    if (not prompt_ids or any(type(t) is not int or t < 0 for t in prompt_ids)
            or seqs[0].get_output_token_ids()):
        raise ValueError('dLoRA enqueue requires native prompt IDs and no output yet')
    group._tc_dlora = dict(schema='ieee_tc_dlora_native_sequence_v1',
        request_id=group.request_id, model_id=group.model_id,
        native_clock_id=local_monotonic_clock_id(), native_prompt_token_ids=prompt_ids,
        target_tokens=params.max_tokens, queued_at=time.perf_counter(),
        scheduled_at=None, first_at=None, last_at=None, token_ids=[], engine_path=[])


def dlora_scheduled(groups, engine_id):
    """Record actual native scheduling, preserving the first boundary on resume."""
    from faaslora.clock import local_monotonic_clock_id
    if type(engine_id) is not int or engine_id < 0:
        raise ValueError('native dLoRA engine ID required')
    at, clock = time.perf_counter(), local_monotonic_clock_id()
    for group in groups:
        state = group._tc_dlora
        if (state['request_id'] != group.request_id or state['model_id'] != group.model_id
                or state['native_clock_id'] != clock or at < state['queued_at']
                or (state['last_at'] is not None and at < state['last_at'])):
            raise ValueError('dLoRA scheduling identity/clock changed')
        if state['scheduled_at'] is None:
            state['scheduled_at'] = at
        path = state['engine_path']
        if not path or path[-1] != engine_id:
            path.append(engine_id)


def dlora_tokens_available(groups, engine_id, produced_at):
    """Timestamp the existing sampled-ID result before native text decoding.

    `produced_at` is taken on entry to _process_worker_outputs, after the
    existing worker result has arrived. It is not a CUDA kernel timestamp.
    One returned batch shares this boundary. No extra synchronization or HTTP
    streaming change; migration does not reset the first/last token history.
    """
    from faaslora.clock import local_monotonic_clock_id
    clock = local_monotonic_clock_id()
    for group in groups:
        state, seqs = group._tc_dlora, group.get_seqs()
        if len(seqs) != 1:
            raise ValueError('dLoRA measured generation must have exactly one sequence')
        ids = list(seqs[0].get_output_token_ids())
        if (state['request_id'] != group.request_id or state['model_id'] != group.model_id
                or state['native_clock_id'] != clock
                or not state['engine_path'] or state['engine_path'][-1] != engine_id
                or list(seqs[0].data.prompt_token_ids) != state['native_prompt_token_ids']
                or any(type(t) is not int or t < 0 for t in ids)
                or len(ids) != len(state['token_ids'])+1 or ids[:-1] != state['token_ids']
                or len(ids) > state['target_tokens'] or state['scheduled_at'] is None
                or type(produced_at) not in (int, float) or not math.isfinite(produced_at)
                or produced_at < state['scheduled_at']
                or (state['last_at'] is not None and produced_at < state['last_at'])):
            raise ValueError('dLoRA native output identity/count/time is inconsistent')
        if state['first_at'] is None:
            state['first_at'] = produced_at
        state['last_at'], state['token_ids'] = produced_at, ids


def dlora_output_snapshot(group):
    """Immutable-by-copy metadata on an existing native RequestOutput."""
    return copy.deepcopy(group._tc_dlora)


def dlora_validate_final_output(output, *, request_id, model_id, prompt_ids, target,
                                dispatched_at, completed_at, clock_id):
    """Validate final transport against the recorded native events, not text.

    The native API intentionally returns only a terminal output; first/last
    times below come from earlier engine events carried through migration.
    This proves sequence/model-ID binding, NOT independent LoRA arithmetic.
    """
    state = output.tc_dlora_native
    times = [dispatched_at, state['queued_at'], state['scheduled_at'],
             state['first_at'], state['last_at'], completed_at]
    ids = state['token_ids']
    if (state['schema'] != 'ieee_tc_dlora_native_sequence_v1'
            or state['native_clock_id'] != clock_id
            or output.request_id != request_id or state['request_id'] != request_id
            or type(model_id) is not int or state['model_id'] != model_id
            or type(target) is not int or not 1 <= target <= 256
            or state['target_tokens'] != target or len(ids) != target
            or any(type(t) is not int or t < 0 for t in ids)
            or list(output.prompt_token_ids) != prompt_ids
            or state['native_prompt_token_ids'] != prompt_ids
            or not output.finished or len(output.outputs) != 1
            or output.outputs[0].finish_reason != 'length'
            or list(output.outputs[0].token_ids) != ids
            or not state['engine_path']
            or any(type(i) is not int or i < 0 for i in state['engine_path'])
            or any(type(t) not in (int, float) or not math.isfinite(t) for t in times)
            or times != sorted(times) or (target == 1 and state['first_at'] != state['last_at'])):
        raise ValueError('dLoRA final native evidence does not match the submitted request')
    decode = (state['last_at']-state['first_at'])*1000.
    return dict(timing_contract='ieee_tc_dlora_native_v1', native_terminal_observed=True,
        native_timing_source='dlora_worker_result_available_before_native_text_decode',
        native_clock_id=clock_id, native_dispatch_monotonic_s=dispatched_at,
        native_queued_monotonic_s=state['queued_at'],
        native_scheduled_monotonic_s=state['scheduled_at'],
        native_first_token_monotonic_s=state['first_at'], native_last_token_monotonic_s=state['last_at'],
        worker_completed_monotonic_s=completed_at, native_output_tokens=len(ids),
        native_decode_ms=decode, native_tpot_ms=decode/(len(ids)-1) if len(ids)>1 else None,
        worker_completion_notification_ms=(completed_at-state['last_at'])*1000.,
        native_prompt_token_ids=list(prompt_ids), native_prompt_token_ids_sha256=_native_ids_digest(prompt_ids),
        completion_token_ids=list(ids), completion_token_ids_sha256=_native_ids_digest(ids),
        native_model_id=model_id, native_engine_path=list(state['engine_path']),
        native_adapter_binding_source='dlora_native_sequence_group_model_id',
        lora_numerical_correctness_qualified=False)


class NativeRequestObservation:
    def __init__(self, lora_request):
        from faaslora.clock import local_monotonic_clock_id
        from faaslora.metrics.metrics_collector import NativeV1TokenTimeline
        self.timeline = NativeV1TokenTimeline(time.perf_counter(), local_monotonic_clock_id())
        self.lora_name = None if lora_request is None else lora_request.lora_name
        self.lora_int_id = None if lora_request is None else lora_request.lora_int_id
        self.prompt_ids = None

    def observe(self, output):
        if (len(output.outputs) != 1
                or getattr(output, 'tc_native_lora_name', object()) != self.lora_name):
            raise ValueError('native output sequence/adapter binding differs')
        ids = output.prompt_token_ids
        if not ids or any(type(t) is not int or t < 0 for t in ids):
            raise ValueError('native prompt token IDs missing')
        if self.prompt_ids is not None and list(ids) != self.prompt_ids:
            raise ValueError('native prompt IDs changed during request')
        self.prompt_ids = list(ids)
        if output.finished and output.outputs[0].finish_reason != 'length':
            raise ValueError('fixed-output request did not finish by length')
        self.timeline.observe(output.metrics, output.outputs[0].token_ids, finished=output.finished)

    def finish(self, internal_metrics):
        observed = self.timeline.finalize(time.perf_counter())
        digest = lambda x: hashlib.sha256(json.dumps(x, separators=(',', ':')).encode()).hexdigest()
        return dict(**observed, native_prompt_token_ids=list(self.prompt_ids),
                    native_prompt_token_ids_sha256=digest(self.prompt_ids),
                    completion_token_ids=list(self.timeline.token_ids),
                    completion_token_ids_sha256=digest(self.timeline.token_ids),
                    native_lora_name=self.lora_name, native_lora_int_id=self.lora_int_id,
                    native_adapter_binding_source='vllm_v1_engine_request_state',
                    lora_numerical_correctness_qualified=False,
                    control_observation={k: v for k, v in internal_metrics.items()
                                         if k.startswith('tc_') or k in
                                         ('ready_instances_at_enqueue', 'instance_id')})


class V030NativeRequestObservation(NativeRequestObservation):
    """Use the shared timeline with output-event adapter identity checks."""

    def __init__(self, lora_request):
        super().__init__(lora_request)
        self.lora_identity = (None if lora_request is None else
            (lora_request.lora_name, lora_request.lora_int_id, lora_request.lora_path))

    def observe(self, output):
        if getattr(output, 'tc_native_lora_identity', object()) != self.lora_identity:
            raise ValueError('native output adapter ID/path differs from the submitted request')
        super().observe(output)


def loquetier_bind_static_registry(generator, adapter_paths):
    """One startup readback of native name/module/pointer tables, before traffic.

    Does not load, move, deduplicate or alter weights. This proves structural
    binding only; remote content and numerical qualification are separate.
    """
    model, cfg = generator.model, generator.generation_config
    native_class = type(model).__module__+'.'+type(model).__name__
    registry = dict(model.lora_name_to_id)
    if (native_class != 'loquetier_src.models.mixed_lora.MixedLoraModel'
            or hasattr(generator, '_tc_loquetier_registry')
            or not adapter_paths or set(adapter_paths) != set(registry)
            or sorted(registry.values()) != list(range(len(registry)))
            or any(type(i) is not int for i in registry.values())
            or any(not isinstance(p, str) or not Path(p).is_absolute() for p in adapter_paths.values())
            or cfg.do_sample is not False or cfg.num_beams != 1 or cfg.eos_token_id is not None
            or generator.mode.name != 'GREEDY_SEARCH' or generator.eos_token_ids != [None]
            or model.training_loras or str(model.dtype) != 'torch.float16'):
        raise ValueError('native static Loquetier fixed-output registry/config required')
    names = sorted(registry, key=registry.get)
    if set(model.adapter_lists) != set(names) or not model.target_module_saves:
        raise ValueError('native Loquetier module inventory differs')
    for module_name, adapters in model.target_module_saves.items():
        layer = model.model.get_submodule(module_name)
        if (type(layer).__module__+'.'+type(layer).__name__ !=
                'loquetier_src.models.mixed_lora.MixedLoraLinear'
                or layer.root_context is not model.root_context
                or layer.adapters is not adapters or layer.rank != model.rank
                or [a.adapter_name for a in adapters] != names):
            raise ValueError('native per-module LoRA index/layout differs')
        expected_a, expected_b = [], []
        for name, adapter in zip(names, adapters):
            if (sum(a is adapter for a in model.adapter_lists[name]) != 1
                    or str(adapter.wa.dtype) != 'torch.float16'
                    or str(adapter.wb.dtype) != 'torch.float16'
                    or adapter.wa.device != layer.device or adapter.wb.device != layer.device
                    or adapter.wa.ndim != 2 or adapter.wb.ndim != 2
                    or adapter.wa.shape[0] != model.rank or adapter.wb.shape[1] != model.rank
                    or not adapter.wa.is_contiguous() or not adapter.wb.is_contiguous()):
                raise ValueError('native prepared weight storage differs')
            expected_a.append(adapter.wa.data_ptr())
            expected_b.append(adapter.wb.data_ptr())
        # Explicit startup-only readback. Never called in per-token hooks.
        if (layer.wa_ptrs.detach().cpu().tolist() != expected_a
                or layer.wb_ptrs.detach().cpu().tolist() != expected_b):
            raise ValueError('native SMLM pointer table differs from adapter storage')
    if any(len(model.adapter_lists[n]) != len(model.target_module_saves) for n in names):
        raise ValueError('native adapter is missing a target module')
    generator._tc_loquetier_registry = registry
    generator._tc_loquetier_paths = dict(adapter_paths)
    generator._tc_loquetier_active = {}
    generator._tc_loquetier_seen = set()
    generator._tc_loquetier_request_ids = set()
    return dict(native_class=native_class, adapter_count=len(names),
        target_modules=len(model.target_module_saves), rank=model.rank,
        pointer_tables_verified=True, numerical_correctness_qualified=False)


def loquetier_register_requests(generator, requests):
    """Reject unknown IDs before native index-0 fallback or any KV allocation."""
    registry = generator._tc_loquetier_registry
    at, pending, seen, request_ids = time.perf_counter(), [], set(), set()
    for request in requests:
        rid, aid, mark = getattr(request, '_tc_request_id', None), request.lora_name, request.mark_id
        ids = request.next_input_ids
        if (not isinstance(rid, str) or not rid or type(mark) is not int or mark < 0
                or rid in request_ids or rid in generator._tc_loquetier_request_ids
                or mark in seen or mark in generator._tc_loquetier_seen
                or hasattr(request, '_tc_loquetier') or aid not in registry
                or generator.model.lora_name_to_id.get(aid) != registry[aid]
                or request.status.name != 'Prefill' or request.labels is not None
                or request.input_ids.device.type != 'cpu' or request.input_ids.ndim != 1
                or str(request.input_ids.dtype) != 'torch.int64'
                or type(ids) is not list or not ids or any(type(t) is not int or t < 0 for t in ids)
                or request.input_ids.tolist() != ids
                or type(request.max_length) is not int
                or not 1 <= request.max_length-len(ids) <= 256 or request.max_length > 1024
                or request.logits_processor is None or len(request.logits_processor) != 0):
            raise ValueError('native Loquetier request differs from fixed input/static binding contract')
        seen.add(mark)
        request_ids.add(rid)
        state = dict(request_id=rid, mark_id=mark, adapter_id=aid, adapter_index=registry[aid],
            adapter_path=generator._tc_loquetier_paths[aid], native_prompt_token_ids=list(ids),
            native_prompt_token_ids_sha256=_native_ids_digest(ids),
            target_tokens=request.max_length-len(ids), registered_monotonic_s=at,
            scheduled_monotonic_s=None, first_token_monotonic_s=None,
            last_token_monotonic_s=None, output_ids=[], forward_count=0, finished=False)
        pending.append((request, state))
    # All validation precedes attachment. No partial batch on validation failure.
    for request, state in pending:
        request._tc_loquetier = state
        generator._tc_loquetier_active[request.mark_id] = state
        generator._tc_loquetier_seen.add(request.mark_id)
        generator._tc_loquetier_request_ids.add(state['request_id'])


def loquetier_forward_started(generator, lora_ids, lora_lens):
    """Observe actual native grouped input IDs and root context before forward."""
    at = time.perf_counter()
    expected_ids, expected_lens, states = [], [], []
    for request in generator.requests:
        state = request._tc_loquetier
        aid, index = state['adapter_id'], state['adapter_index']
        if (state['finished'] or state['request_id'] != request._tc_request_id
                or state['mark_id'] != request.mark_id or request.lora_name != aid
                or request.lora_id != index or generator.model.lora_name_to_id.get(aid) != index
                or state['forward_count'] != len(state['output_ids'])
                or at < state['registered_monotonic_s']
                or (state['last_token_monotonic_s'] is not None and at < state['last_token_monotonic_s'])):
            raise ValueError('actual Loquetier forward identity/order differs')
        length = len(request.next_input_ids)
        if not expected_ids or expected_ids[-1] != index:
            expected_ids.append(index)
            expected_lens.append(length)
        else:
            expected_lens[-1] += length
        states.append(state)
    context = generator.model.root_context
    if (not states or lora_ids != expected_ids or lora_lens != expected_lens
            or context.lora_ids != lora_ids or context.train_idx):
        raise ValueError('actual Loquetier forward segments/context differ')
    for state in states:
        if state['scheduled_monotonic_s'] is None:
            state['scheduled_monotonic_s'] = at
        state['forward_count'] += 1


def loquetier_token_event(generator, request, token_id, produced_at):
    """Immediately after the ORIGINAL scalar .item(); no added CUDA operation."""
    state = request._tc_loquetier
    count = len(state['output_ids'])
    previous = (state['scheduled_monotonic_s'] if state['last_token_monotonic_s'] is None
                else state['last_token_monotonic_s'])
    if (generator._tc_loquetier_active.get(request.mark_id) is not state
            or state['adapter_id'] != request.lora_name or state['adapter_index'] != request.lora_id
            or state['finished'] or state['forward_count'] != count+1
            or count >= state['target_tokens'] or type(token_id) is not int or token_id < 0
            or type(produced_at) not in (int, float) or not math.isfinite(produced_at)
            or previous is None or produced_at < previous
            or request.next_input_ids != [token_id]
            or request.input_ids.shape[0] != len(state['native_prompt_token_ids'])+count):
        raise ValueError('native Loquetier produced token identity/time/count differs')
    if count == 0:
        state['first_token_monotonic_s'] = produced_at
    state['last_token_monotonic_s'] = produced_at
    state['output_ids'].append(token_id)


def loquetier_finished(generator):
    """Check native terminal markers without inventing completion or token IDs."""
    for mark in generator.finished_mark_ids:
        state = generator._tc_loquetier_active[mark]
        if (state['finished'] or len(state['output_ids']) != state['target_tokens']
                or state['forward_count'] != state['target_tokens']):
            raise ValueError('native Loquetier fixed-length terminal differs')
        state['finished'] = True


def loquetier_request_result(generator, mark, completed_at, process):
    """Consume a real native terminal once; no HTTP/text-derived token counts."""
    state = generator._tc_loquetier_active[mark]
    ids, prompt = state['output_ids'], state['native_prompt_token_ids']
    times = [state['registered_monotonic_s'], state['scheduled_monotonic_s'],
             state['first_token_monotonic_s'], state['last_token_monotonic_s'], completed_at]
    if (not state['finished'] or mark not in generator.finished_mark_ids
            or len(ids) != state['target_tokens'] or state['forward_count'] != len(ids)
            or any(type(t) not in (int, float) or not math.isfinite(t) for t in times)
            or times != sorted(times) or process != _slora_process_identity(os.getpid())):
        raise ValueError('incomplete native Loquetier terminal/process/timing')
    result = dict(timing_contract='ieee_tc_loquetier_native_v1', native_terminal_observed=True,
        native_timing_source='loquetier_native_cpu_integer_after_existing_item',
        native_clock_id=process['clock_id'], native_registered_monotonic_s=times[0],
        native_scheduled_monotonic_s=times[1], native_first_token_monotonic_s=times[2],
        native_last_token_monotonic_s=times[3], worker_completed_monotonic_s=times[4],
        native_output_tokens=len(ids), native_decode_ms=(times[3]-times[2])*1000.,
        native_tpot_ms=(times[3]-times[2])*1000./(len(ids)-1) if len(ids)>1 else None,
        worker_completion_notification_ms=(times[4]-times[3])*1000.,
        native_prompt_token_ids=list(prompt), native_prompt_token_ids_sha256=_native_ids_digest(prompt),
        completion_token_ids=list(ids), completion_token_ids_sha256=_native_ids_digest(ids),
        native_adapter_id=state['adapter_id'], native_adapter_path=state['adapter_path'],
        native_adapter_index=state['adapter_index'], native_request_id=state['request_id'],
        native_adapter_binding_source='loquetier_actual_forward_segments_and_startup_pointer_tables',
        native_process=dict(process), native_engine_class='loquetier_src.models.mixed_lora.MixedLoraModel',
        native_forward_count=state['forward_count'], lora_numerical_correctness_qualified=False)
    del generator._tc_loquetier_active[mark]
    return result


def loquetier_cancel_request(generator, mark):
    """Inference-only cancellation at a serialized native iteration boundary.

    Only the online adapter calls this, never concurrently with generate().
    The native cache release and decode-list invalidation preserve surviving
    request order. Partial tokens remain failure evidence, not successful work.
    """
    if generator.train_requests or generator.eval_requests:
        raise ValueError('online cancellation does not support training/evaluation')
    state = generator._tc_loquetier_active[mark]
    requests = [r for r in generator.requests+generator.prefill_requests if r.mark_id == mark]
    if state['finished'] or len(requests) != 1 or requests[0]._tc_loquetier is not state:
        raise ValueError('native cancellation requires exactly one active request')
    request = requests[0]
    if request.kvcache is None:
        raise ValueError('native registered request has no KV cache')
    request.kvcache.release()
    generator.requests = [r for r in generator.requests if r is not request]
    generator.prefill_requests = [r for r in generator.prefill_requests if r is not request]
    # The next native pre_process must not grow a just-released cache.
    generator.cached_decode_kv = [r.kvcache for r in generator.requests]
    generator.decode_request_changed = bool(generator.requests)
    if not generator.requests:
        generator.new_decode_request = False
    del generator._tc_loquetier_active[mark]
    return dict(event='loquetier_request_cancelled', request_id=state['request_id'],
        mark_id=mark, adapter_id=state['adapter_id'], at=time.perf_counter(),
        native_registered=True, native_output_tokens=len(state['output_ids']),
        completion_token_ids=list(state['output_ids']),
        native_first_token_monotonic_s=state['first_token_monotonic_s'],
        native_last_token_monotonic_s=state['last_token_monotonic_s'],
        native_forward_count=state['forward_count'], native_terminal_observed=False)


def loquetier_clear_released_decode_cache(generator):
    """Drop stale references only after the last native request has finished.

    Native post_process releases pages but leaves cached_decode_kv populated
    when requests becomes empty. A subsequent pre_process would acquire a page
    for those dead sequences. This is a lifecycle correctness adaptation, not
    a new KV/cache policy; never free live pages or modify an active batch.
    """
    if generator.requests_len:
        return 0
    caches = generator.cached_decode_kv
    if any(c.num_pages != 0 or c.seqlen != 0 for c in caches):
        raise ValueError('empty native batch retains live decode pages; cannot discard silently')
    count = len(caches)
    generator.cached_decode_kv = []
    return count


def slora_join_owned_children(children, grace_s=10.):
    """Close exact native multiprocessing handles, never a process-name search.

    TP1's router is the GPU worker; its death still is NOT a physical-release
    receipt. The outer owner separately verifies pidfd exit and native census.
    """
    owned = [child for child in children if child is not None and child.pid is not None]
    for child in owned:
        if child.is_alive():
            child.terminate()
    deadline = time.monotonic()+grace_s
    for child in owned:
        child.join(timeout=max(0., deadline-time.monotonic()))
    for child in owned:
        if child.is_alive():
            child.kill()
    for child in owned:
        child.join(timeout=2.)
        if child.is_alive():
            raise RuntimeError('owned native S-LoRA child did not exit')


def slora_worker_launch_guard(world_size):
    from replay_vllm_engine_trace import slora_worker_context
    if type(world_size) is not int or world_size != 1:
        raise ValueError('native S-LoRA worker lifecycle requires TP1')
    return slora_worker_context(Path(os.environ['FAASLORA_TC_SLORA_WORKER_CONFIG']),
                               int(os.environ['FAASLORA_TC_SLORA_WORKER_INDEX']))


def slora_record_worker_ready(server, weight_dir, adapter_dirs, max_total_token_num, load_way, mode):
    """Called AFTER native model/all-adapter/pool initialization; no warm inference."""
    from replay_vllm_engine_trace import slora_worker_context, _exclusive_json, _sha
    config_path = Path(os.environ['FAASLORA_TC_SLORA_WORKER_CONFIG'])
    index = int(os.environ['FAASLORA_TC_SLORA_WORKER_INDEX'])
    cfg, guard = slora_worker_context(config_path, index)
    expected = {path: i for i, path in cfg['adapters'].values()}
    params = vars(server.input_params)
    if (server.world_size != 1 or server.tp_rank != 0 or load_way != 'HF' or mode != []
            or str(Path(weight_dir).resolve()) != cfg['native_args']['model_dir']
            or list(adapter_dirs) != list(expected)
            or server.adapter_id != (expected | {None: len(expected)})
            or len(server.adapters) != len(expected)+1 or server.adapters[-1] is not None
            or any(server.adapters[i].lora_dir != path for path, i in expected.items())
            or params != cfg['native_input_params']
            or max_total_token_num != cfg['native_args']['max_total_token_num']
            or server.model is None or server.infer_adapter is None):
        raise ValueError('actual S-LoRA initialization differs from frozen full-pool worker')
    import inspect
    import torch
    now = time.perf_counter()
    from faaslora.clock import local_monotonic_clock_id
    identity = guard.gpu_process_identity(os.getpid())
    if (identity is None or torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0
            or now < cfg['deployment_notice_s']):
        raise ValueError('native S-LoRA device/process/notice identity differs')
    receipt_id = uuid.uuid4().hex
    rpc_path = Path(inspect.getfile(type(server))).resolve()
    row = dict(schema='ieee_tc_slora_backend_ready_v1', receipt_id=receipt_id,
        at=now, clock_id=local_monotonic_clock_id(), identity=identity, index=index,
        api_parent_identity=guard.gpu_process_identity(os.getppid()),
        worker_config_sha256=_sha(config_path), runner_sha256=_sha(inspect.getfile(slora_worker_context)),
        rpc_path=str(rpc_path), rpc_sha256=_sha(rpc_path),
        source_view_manifest_sha256=cfg['source_view_manifest_sha256'],
        cuda_visible=os.environ['CUDA_VISIBLE_DEVICES'], cuda_device_count=1, tp=1,
        native_python=__import__('sys').version, torch_version=torch.__version__,
        native_environment=dict(cfg['native_environment']),
        model_dir=str(Path(weight_dir).resolve()), native_input_params=params,
        adapter_ids=expected, adapter_count=len(expected),
        host_layout=cfg['host_layout'], ready_boundary='native_full_pool_and_model_initialized')
    path = Path(cfg['output'])/'worker_receipts'/(receipt_id+'.json')
    _exclusive_json(path, row)
    _exclusive_json(Path(cfg['output'])/f'worker_{index}.native_ready.json',
        dict(tc_worker_receipt_id=receipt_id, tc_worker_receipt_sha256=_sha(path)))
    return row


def slora_require_packed_host_mode(world_size, params):
    """Opt-in representation adapter, never a fallback from unsupported modes."""
    if (type(world_size) is not int or world_size != 1
            or params.swap is not True or params.scheduler != 'slora'
            or any(getattr(params, flag) is not False for flag in (
                'dummy', 'bmm', 'no_lora_swap', 'no_lora', 'no_kernel',
                'no_lora_compute', 'no_mem_pool'))):
        raise ValueError('packed HOST views require real TP1 unified-pool swap execution')


def slora_load_packed_host_weights(layer, weights):
    """One pinned packed allocation; module fields are value-equivalent views.

    Only the explicitly selected TP1 non-GQA packed engine is supported. Its
    unchanged loader/paging consumes w_combined_home, not the module-view strides.
    No cross-ID deduplication, quantization, cache policy or loading-time change.
    The original separate+packed representation remains the default source view.
    """
    import torch
    cfg = layer.network_config
    hidden, heads, kv_heads = (cfg.get(k) for k in (
        'hidden_size', 'num_attention_heads', 'num_key_value_heads'))
    rank = layer.lora_config.get('r')
    if (layer.world_size_ != 1 or layer.tp_rank_ != 0 or layer.no_lora_swap
            or layer.data_type_ != torch.float16
            or any(type(v) is not int or v <= 0 for v in (hidden, heads, kv_heads, rank))
            or heads != kv_heads or hidden % heads or not weights
            or hasattr(layer, 'tc_host_storage')):
        raise ValueError('unsupported or repeated packed HOST initialization')
    prefix = next(iter(weights))
    if '.layers.' not in prefix:
        raise ValueError('native attention tensor prefix required')
    prefix = prefix[:prefix.find('layers')] + f'layers.{layer.layer_num_}.self_attn'
    values = []
    for side in ('A', 'B'):
        for module in ('q', 'k', 'v', 'o'):
            key = f'{prefix}.{module}_proj.lora_{side}.weight'
            value = weights.get(key)
            shape = (rank, hidden) if side == 'A' else (hidden, rank)
            if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                    or value.dtype != torch.float16 or tuple(value.shape) != shape):
                raise ValueError('complete native FP16 CPU QKVO tensors required: '+key)
            values.append((side, module, value))
    # Validate all tensors first. Native order is A:q/k/v/o then B:q/k/v/o.
    packed = torch.empty((2, 4*rank, heads, hidden//heads), dtype=torch.float16,
                         device='cpu', pin_memory=True)
    views = {}
    for index, (side, module, value) in enumerate(values):
        part = packed[index//4, (index%4)*rank:(index%4+1)*rank]
        part.copy_(value.reshape(rank, heads, hidden//heads))
        views[f'{module}_lora_{side}_home'] = part.view(tuple(value.shape)).T
    # Commit only after complete initialization. Packed bytes/order/contiguity
    # are unchanged; module views deliberately need not be contiguous.
    layer.w_combined_home, layer.w_combined = packed, None
    for name, view in views.items():
        setattr(layer, name, view)
        setattr(layer, name.removesuffix('_home'), None)
    layer.tc_host_storage = dict(layout='packed_views_v1',
        unique_storage_bytes=packed.untyped_storage().nbytes(),
        layer_num=layer.layer_num_, rank=rank, module_fields_are_views=True)


@lru_cache(maxsize=4)
def _slora_process_identity(pid):
    # PID is a cache key: a fork must not inherit its parent's identity.
    from faaslora.clock import local_monotonic_clock_id
    if pid != os.getpid():
        raise ValueError('native observation process identity differs')
    return dict(pid=pid, start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
                clock_id=local_monotonic_clock_id())


def _native_ids_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def slora_register_requests(server, requests):
    """Observe original RPC inputs before InferBatch aliases/appends to them.

    Only the qualified TP=1 LoRA path is supported. Metadata lives on the
    existing request objects, so native filter/merge/free own its lifetime.
    No global request cache, input mutation, scheduling or sampling change.
    """
    params = server.input_params
    if (server.world_size != 1 or server.tp_rank != 0
            or any(getattr(params, flag) for flag in ('dummy', 'no_lora', 'no_kernel', 'no_lora_compute'))
            or params.scheduler != 'slora'):
        raise ValueError('S-LoRA native observation requires real TP1 slora execution')
    at = time.perf_counter()
    seen = set()
    prepared = []
    for request in requests:
        rid, path = request['request_id'], request['adapter_dir']
        ids, target = request['input_id'], request['output_len']
        sample = request['sampling_param']
        if (not isinstance(rid, str) or not rid or rid in seen or '_tc_slora' in request
                or not isinstance(path, str) or not Path(path).is_absolute()
                or type(ids) is not list or not ids or any(type(t) is not int or t < 0 for t in ids)
                or type(target) is not int or not 1 <= target <= 256
                or sample.get('do_sample') is not False or sample.get('temperature') != 1.
                or sample.get('top_p') != 1. or sample.get('top_k') != 1
                or sample.get('presence_penalty') != 0. or sample.get('frequency_penalty') != 0.):
            raise ValueError('native RPC input differs from fixed greedy LoRA contract')
        index = server.adapter_id[path]
        if type(index) is not int or index < 0 or server.adapters[index].lora_dir != path:
            raise ValueError('native registered adapter object/path differs')
        seen.add(rid)
        prepared.append((request, dict(request_id=rid, adapter_path=path, adapter_index=index,
            native_prompt_token_ids=list(ids), native_prompt_token_ids_sha256=_native_ids_digest(ids),
            target_tokens=target, registered_monotonic_s=at, scheduled_monotonic_s=None,
            count=0, process=dict(_slora_process_identity(os.getpid())))))
    # All validation precedes metadata attachment; failed validation is not a
    # partially registered native batch. No native fields above are modified.
    for request, state in prepared:
        request['_tc_slora'] = state


def slora_forward_started(server, batch, engine):
    """Record first actual forward boundary and verify its selected LoRA path."""
    expected = ('slora.models.bmm.lora_bmm_infer.LoraBmmInfer' if server.input_params.bmm else
                'slora.models.peft.lora_unordered_batch_infer.LoraUnorderedBatchInfer')
    actual = type(engine).__module__+'.'+type(engine).__name__
    if actual != expected or engine.base_model is not server.model:
        raise ValueError('native forward is not the declared LoRA engine')
    if len(batch.adapter_dirs) != len(batch.requests):
        raise ValueError('native batch request/adapter dimensions differ')
    at = time.perf_counter()
    for request, path in zip(batch.requests, batch.adapter_dirs):
        state = request['_tc_slora']
        index = server.adapter_id[path]
        adapter = server.adapters[index]
        if (state['request_id'] != request['request_id'] or state['adapter_path'] != path
                or state['adapter_index'] != index or adapter.lora_dir != path
                or (server.input_params.bmm and adapter.is_on_gpu() is not True)
                or (not server.input_params.bmm and
                    (engine.infer_adapter is not server.infer_adapter
                     or path not in server.infer_adapter.adapter_dirs))):
            raise ValueError('actual native forward adapter differs or is not resident')
        if state['scheduled_monotonic_s'] is None:
            state['scheduled_monotonic_s'] = at
            state['engine_class'] = actual


def slora_token_event(request, adapter_path, token_id, produced_at):
    """Snapshot after the existing CPU ID transfer, before logprob/text/HTTP.

    No new CUDA synchronization. One batch timestamp applies to its already
    returned ID vector, not an invented per-kernel timestamp. Original prompt
    IDs are emitted once, so streaming does not duplicate them per token.
    """
    state = request['_tc_slora']
    scheduled = state['scheduled_monotonic_s']
    if (adapter_path != state['adapter_path'] or request['request_id'] != state['request_id']
            or type(token_id) is not int or token_id < 0
            or type(produced_at) not in (float, int) or not math.isfinite(produced_at)
            or scheduled is None or produced_at < scheduled
            or produced_at < state.get('last_at', scheduled)
            or state['count'] >= state['target_tokens']
            or len(request['input_id']) != len(state['native_prompt_token_ids'])+state['count']+1
            or request['input_id'][-1] != token_id):
        raise ValueError('native produced token/boundary differs from observed request')
    state['count'] += 1
    state['last_at'] = produced_at
    event = dict(schema='ieee_tc_slora_token_event_v1', request_id=state['request_id'],
        adapter_path=adapter_path, adapter_index=state['adapter_index'],
        output_index=state['count'], native_token_id=token_id, produced_monotonic_s=produced_at)
    if state['count'] == 1:
        event['first_event'] = {key: copy.deepcopy(state[key]) for key in
            ('native_prompt_token_ids', 'native_prompt_token_ids_sha256', 'target_tokens',
             'registered_monotonic_s', 'scheduled_monotonic_s', 'engine_class', 'process')}
    return event


def slora_canonical_prompt_ids(prompt, contract, vocab_size):
    """Explicit pretokenized input boundary; no old-tokenizer fallback.

    The native router still consumes a plain ID list, with its original context
    and scheduling constraints. The returned copy cannot alias caller state.
    """
    if (not isinstance(prompt, str) or not prompt or type(contract) is not dict
            or set(contract) != {'mode', 'token_ids', 'prompt_sha256', 'token_ids_sha256'}
            or contract['mode'] != 'canonical_token_ids_v1'
            or type(vocab_size) is not int or vocab_size <= 0):
        raise ValueError('explicit canonical S-LoRA input contract required')
    ids = contract['token_ids']
    if (type(ids) is not list or not ids
            or any(type(token) is not int or not 0 <= token < vocab_size for token in ids)
            or contract['prompt_sha256'] != hashlib.sha256(prompt.encode()).hexdigest()
            or contract['token_ids_sha256'] != _native_ids_digest(ids)):
        raise ValueError('canonical S-LoRA prompt/ID identity differs')
    return list(ids)


class SloraNativeRequestObservation:
    """Validate native SSE IDs/events without guessing missing timestamps.

    The RPC registration boundary is NOT relabeled as vLLM engine queue entry.
    The downstream common service adapter must separately record its true
    ingress/dispatch boundaries, resource owner and artifact provenance.
    """
    def __init__(self, *, request_id, adapter_path, prompt_ids, target, clock_id):
        if (not isinstance(request_id, str) or not request_id
                or not isinstance(adapter_path, str) or not Path(adapter_path).is_absolute()
                or type(prompt_ids) is not list or not prompt_ids
                or any(type(t) is not int or t < 0 for t in prompt_ids)
                or type(target) is not int or not 1 <= target <= 256 or not clock_id):
            raise ValueError('explicit S-LoRA native request contract required')
        self.request_id, self.adapter_path = request_id, adapter_path
        self.prompt_ids, self.target, self.clock_id = list(prompt_ids), target, clock_id
        self.tokens, self.first, self.last, self.binding = [], None, None, None
        self.finished = False

    def observe(self, payload):
        event = payload.get('tc_slora_native_token', {})
        token = payload.get('token', {}).get('id')
        finished = payload.get('finished')
        at = event.get('produced_monotonic_s')
        index = event.get('output_index')
        if (self.finished or event.get('schema') != 'ieee_tc_slora_token_event_v1'
                or event.get('request_id') != self.request_id or event.get('adapter_path') != self.adapter_path
                or type(event.get('adapter_index')) is not int or event['adapter_index'] < 0
                or type(index) is not int or index != len(self.tokens)+1 or index > self.target
                or type(token) is not int or token < 0 or type(finished) is not bool
                or type(event.get('native_token_id')) is not int or event['native_token_id'] != token
                or finished != (index == self.target)
                or type(at) not in (float, int) or not math.isfinite(at)
                or (self.last is not None and at < self.last)):
            raise ValueError('missing, duplicate, truncated or mismatched S-LoRA native event')
        if self.binding is None:
            binding = event.get('first_event', {})
            process = binding.get('process', {})
            times = [binding.get('registered_monotonic_s'), binding.get('scheduled_monotonic_s'), at]
            if (binding.get('native_prompt_token_ids') != self.prompt_ids
                    or binding.get('native_prompt_token_ids_sha256') != _native_ids_digest(self.prompt_ids)
                    or binding.get('target_tokens') != self.target
                    or process.get('clock_id') != self.clock_id
                    or type(process.get('pid')) is not int or process['pid'] <= 0
                    or type(process.get('start_ticks')) is not int or process['start_ticks'] <= 0
                    or binding.get('engine_class') not in (
                        'slora.models.peft.lora_unordered_batch_infer.LoraUnorderedBatchInfer',
                        'slora.models.bmm.lora_bmm_infer.LoraBmmInfer')
                    or any(type(t) not in (float, int) or not math.isfinite(t) for t in times)
                    or times != sorted(times)):
                raise ValueError('native initial prompt/process/adapter/timing binding differs')
            self.binding = copy.deepcopy(binding)
            self.adapter_index, self.first = event['adapter_index'], at
        elif 'first_event' in event or event['adapter_index'] != self.adapter_index:
            raise ValueError('native initial binding was repeated or changed')
        self.tokens.append(token)
        self.last, self.finished = at, finished

    def finish(self, completed_at):
        if (not self.finished or len(self.tokens) != self.target
                or type(completed_at) not in (int, float) or not math.isfinite(completed_at)
                or completed_at < self.last):
            raise ValueError('incomplete native S-LoRA output/completion boundary')
        decode = (self.last-self.first)*1000.
        return dict(timing_contract='ieee_tc_slora_native_v1', native_terminal_observed=True,
            native_timing_source='slora_model_rpc_cpu_token_vector_available',
            native_clock_id=self.clock_id, native_rpc_registered_monotonic_s=self.binding['registered_monotonic_s'],
            native_scheduled_monotonic_s=self.binding['scheduled_monotonic_s'],
            native_first_token_monotonic_s=self.first, native_last_token_monotonic_s=self.last,
            worker_completed_monotonic_s=completed_at, native_output_tokens=len(self.tokens),
            native_decode_ms=decode, native_tpot_ms=decode/(len(self.tokens)-1) if len(self.tokens)>1 else None,
            worker_completion_notification_ms=(completed_at-self.last)*1000.,
            native_prompt_token_ids=list(self.prompt_ids), native_prompt_token_ids_sha256=_native_ids_digest(self.prompt_ids),
            completion_token_ids=list(self.tokens), completion_token_ids_sha256=_native_ids_digest(self.tokens),
            native_adapter_path=self.adapter_path, native_adapter_index=self.adapter_index,
            native_adapter_binding_source='slora_model_rpc_actual_batch_adapter',
            native_process=dict(self.binding['process']), native_engine_class=self.binding['engine_class'],
            lora_numerical_correctness_qualified=False)
