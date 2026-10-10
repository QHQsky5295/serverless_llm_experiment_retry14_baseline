"""Native observations and an explicitly opt-in service startup HTTP ingress.

The upstream 0.10.2 V1 frontend computes engine-core timestamps but omits them
from RequestOutput. Snapshot those existing scalar stats before the asynchronous
consumer can see later mutations. No scheduling, loading or sampling change.
The ingress is a disclosed deployment adapter, never installed in legacy runs.
"""
import copy
import asyncio
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

OUTPUT_PROCESSOR_SHA256 = '50f5e0aa5d0b7ece086632de6e37376dbeba0a1c0d2f22f95b6759189e76e04d'


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
