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
import os
from pathlib import Path
import threading
import time
import uuid
from urllib.parse import urlsplit

OUTPUT_PROCESSOR_SHA256 = '50f5e0aa5d0b7ece086632de6e37376dbeba0a1c0d2f22f95b6759189e76e04d'


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
                       error_type=type(exc).__name__)
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
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                timeout=aiohttp.ClientTimeout(total=None), trust_env=False) as self._session:
            try:
                await runner.setup()
                site = web.TCPSite(runner, '127.0.0.1', self.port)
                await site.start()
                self.port = site._server.sockets[0].getsockname()[1]
                self._emit('ingress_listening', port=self.port, pid=os.getpid(),
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


def install_v1_snapshot():
    import vllm.v1.engine.output_processor as module
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != OUTPUT_PROCESSOR_SHA256:
        raise ValueError('unqualified vLLM output-processor version')
    cls = module.RequestState
    old = cls._new_request_output
    if getattr(old, '_ieee_tc_snapshot_v1', False):
        return

    def observed(state, *args, **kwargs):
        return snapshot_output(state, old(state, *args, **kwargs))

    observed._ieee_tc_snapshot_v1 = True
    cls._new_request_output = observed
    old_put = module.RequestOutputCollector.put

    def put(collector, output):
        old_put(collector, output)
        snapshot_collector(collector, output)

    module.RequestOutputCollector.put = put


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
