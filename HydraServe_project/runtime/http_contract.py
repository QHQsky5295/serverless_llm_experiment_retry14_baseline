"""Common HTTP measurement adapter around the original HydraServe scheduler.

The official main.py calls process_request(..., stream=True), then consumes the
synchronous generator off the event loop. Keep that path and its idle loop;
only translate frozen input/output and expose a pre-bootstrap service ingress.
No placement, cache, retry, prefetch, new arrival or alternative engine here.
"""
import asyncio
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import time

from .request_contract import BoundRequest, FrozenRequest, matches_worker_process


def digest_ids(ids):
    return sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


async def consume_native_stream(stream):
    """One owner serializes native next/close, even after repeated cancellation.

    Keep the original per-chunk AnyIO off-loop iterator. Cancelling an asyncio
    waiter cannot stop a running synchronous next(). Signal the owner to stop at
    its next yield, then JOIN it before propagating cancellation. No extra chunk
    is requested after that yield, no completion is fabricated, and native
    request counters remain held until the generator actually closes. A pending
    first read may still finish (or hit its existing transport timeout); this is
    not instantaneous backend preemption and does not change any timeout.
    """
    from starlette.concurrency import iterate_in_threadpool
    stop = asyncio.Event()

    async def own():
        try:
            async for _ in iterate_in_threadpool(stream):
                if stop.is_set():
                    break
        finally:
            stream.close()

    owner = asyncio.create_task(own())
    try:
        await asyncio.shield(owner)
    except asyncio.CancelledError:
        stop.set()
        while not owner.done():
            try:
                await asyncio.shield(owner)
            except asyncio.CancelledError:
                # Repeated cancellation of the waiter must not abandon next().
                # An independently cancelled owner is detected by result().
                continue
        owner.result()  # Preserve an actual stream/close error, not a fake success.
        raise


def decode_request(body, registry, public_model):
    """Use supplied canonical native IDs, never tokenize or apply a template."""
    fixed = dict(model=public_model, temperature=0, top_p=1, ignore_eos=True,
                 stop=[], stop_token_ids=[], stream=False)
    if (not isinstance(body, dict) or any(body.get(k) != v for k, v in fixed.items())
            or type(body.get('input_tokens')) is not list
            or body.get('lora_adapter_name') not in registry):
        raise ValueError('invalid common HydraServe HTTP contract')
    aid = body['lora_adapter_name']
    binding = registry[aid]
    return FrozenRequest(body['request_id'], aid, binding['adapter_index'],
        binding['adapter_path'], body['canonical_prompt'], tuple(body['input_tokens']),
        body['max_tokens'])


def observe_worker(bound, platform, clock_id):
    """Bind container-native process to an actual live host process and clock."""
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    owners = [w for w in platform.workers.values()
        if not w.removed and w.owner and matches_worker_process(bound.native_process,
            w.owner, json.loads((w.output/'entry.json').read_text()), boot)]
    if len(owners) != 1:
        raise ValueError('native output has no unique live owned worker')
    worker = owners[0]
    current = platform.identity(worker.owner['pid'])
    if (current != worker.owner or not Path(current['cgroup']).is_relative_to(platform.parent)
            or os.readlink('/proc/self/ns/time') != os.readlink(f"/proc/{current['pid']}/ns/time")):
        raise ValueError('native output process/containment/clock changed')
    if platform.lifecycle_observer is not None:
        platform.lifecycle_observer.confirm(worker)
    return dict(worker_name=worker.name, container_id=worker.cid,
        host_process=current, native_process=bound.native_process,
        clock_id=clock_id, verified_at_s=time.perf_counter())


def response_from_bound(bound, proof, completed):
    c, events = bound.contract, bound.events
    if not bound.terminal or not events:
        raise ValueError('native stream did not terminate')
    ids = [token for event in events for token in event['delta_token_ids']]
    token_events = [event for event in events if event['delta_token_ids']]
    first, last = (token_events[i]['output_monotonic_s'] for i in (0, -1))
    native = dict(timing_contract='ieee_tc_hydraserve_native_v1',
        native_output_tokens=len(ids), completion_token_ids=ids,
        completion_token_ids_sha256=digest_ids(ids),
        native_prompt_token_ids=list(c.prompt_token_ids),
        native_prompt_token_ids_sha256=digest_ids(list(c.prompt_token_ids)),
        native_request_id=bound.native_request_id, native_adapter_id=c.adapter_id,
        native_adapter_index=c.adapter_index, native_adapter_path=c.adapter_path,
        native_first_token_monotonic_s=first, native_last_token_monotonic_s=last,
        native_tpot_ms=(last-first)*1000/(len(ids)-1) if len(ids)>1 else None,
        worker_completed_monotonic_s=completed, native_terminal_observed=True,
        native_process=bound.native_process, native_clock_id=proof['clock_id'],
        token_source='native_RequestOutput.token_ids',
        timing_source='native_RequestOutput_construction_monotonic',
        worker_proof=proof, control_observation=dict(bound.control),
        artifact_observation=bound.artifact_evidence,
        evidence_scope='native_sequence_binding_not_numerical_correctness')
    return dict(id=c.request_id, usage=dict(completion_tokens=len(ids),
        prompt_tokens=len(c.prompt_token_ids)), metrics=dict(ieee_tc=native))


def validate_observation(prepared, response, event, completed, clock_id, *, ingress_required=False):
    """Require only actual HydraServe events; do not invent vLLM V1 stages."""
    native = response.get('metrics', {}).get('ieee_tc', {})
    control, proof = native.get('control_observation', {}), native.get('worker_proof', {})
    ids, target = native.get('completion_token_ids'), prepared['target_tokens']
    process = native.get('native_process', {})
    path, index = native.get('native_adapter_path'), native.get('native_adapter_index')
    if (response.get('error') or response.get('id') != prepared['request_id']
            or native.get('timing_contract') != 'ieee_tc_hydraserve_native_v1'
            or native.get('native_clock_id') != clock_id or proof.get('clock_id') != clock_id
            or control.get('clock_id') != clock_id or proof.get('native_process') != process
            or not proof.get('worker_name') or not proof.get('container_id')
            or not isinstance(proof.get('host_process'), dict)
            or type(process.get('pid')) is not int or process['pid'] <= 0
            or type(process.get('start_ticks')) is not int or process['start_ticks'] <= 0
            or process.get('start_ticks') != proof.get('host_process', {}).get('start_ticks')
            or not isinstance(process.get('boot_id'), str)
            or not clock_id.startswith('linux-monotonic:'+process['boot_id']+':')
            or native.get('native_adapter_id') != prepared['adapter_id']
            or not isinstance(path, str) or not Path(path).is_absolute()
            or Path(path).name != prepared['adapter_id'] or type(index) is not int or index<=0
            or not isinstance(native.get('native_request_id'), str) or not native['native_request_id']
            or type(ids) is not list or len(ids) != target
            or any(type(token) is not int or token < 0 for token in ids)
            or native.get('completion_token_ids_sha256') != digest_ids(ids)
            or native.get('native_output_tokens') != target
            or response.get('usage') != dict(completion_tokens=target, prompt_tokens=len(prepared['input_token_ids']))
            or native.get('native_prompt_token_ids') != prepared['input_token_ids']
            or native.get('native_prompt_token_ids_sha256') != prepared['native_prompt_token_ids_sha256']
            or control.get('canonical_prompt_sha256') != prepared['canonical_prompt_sha256']
            or native.get('native_terminal_observed') is not True
            or native.get('token_source') != 'native_RequestOutput.token_ids'
            or native.get('timing_source') != 'native_RequestOutput_construction_monotonic'
            or not isinstance(control.get('instance_id'), str) or not control['instance_id']):
        raise ValueError('HydraServe native identity/token/clock differs')
    names = ('http_received_s', 'controller_forwarded_s', 'router_enqueued_s',
             'instance_assigned_s', 'backend_ready_s', 'native_http_submitted_s')
    times = [event['planned_arrival_s'], event['task_created_s'], event['client_submit_s'],
        *[control.get(k) for k in names], native.get('native_first_token_monotonic_s'),
        native.get('native_last_token_monotonic_s'), native.get('worker_completed_monotonic_s'),
        proof.get('verified_at_s'), completed]
    if any(type(t) not in (int,float) or not math.isfinite(t) for t in times) or times != sorted(times):
        raise ValueError('HydraServe actual timeline missing or unordered')
    artifact = native.get('artifact_observation', {})
    if 'artifact_content_sha256' in prepared:
        a_times = [artifact.get(k) for k in ('tc_artifact_resolve_started_s',
            'tc_artifact_resolve_completed_s','native_view_completed_s')]
        hit, size = artifact.get('tc_artifact_cache_hit'), artifact.get('tc_artifact_transferred_bytes')
        if (artifact.get('tc_artifact_request_id') != prepared['request_id']
                or artifact.get('tc_artifact_id') != prepared['adapter_id']
                or artifact.get('tc_artifact_content_sha256') != prepared['artifact_content_sha256']
                or artifact.get('tc_artifact_manifest_sha256') != prepared['artifact_manifest_sha256']
                or artifact.get('native_adapter_path') != path
                or artifact.get('tc_artifact_delivery_mode') != 'prepublished_gzip_v1'
                or not re.fullmatch('[0-9a-f]{64}', str(artifact.get('tc_artifact_receipt_sha256')))
                or not re.fullmatch('[0-9a-f]{32}', str(artifact.get('tc_artifact_transfer_id')))
                or type(hit) is not bool or type(size) is not int or (size != 0 if hit else size<=0)
                or any(type(t) not in (int,float) or not math.isfinite(t) for t in a_times)
                or a_times != sorted(a_times) or not times[7]<=a_times[0]<=a_times[-1]<=times[8]):
            raise ValueError('HydraServe request-driven artifact proof differs')
    a,e,d,f,last = times[0],times[2],times[6],times[9],times[10]
    components = [e-a,d-e,f-d,last-f,completed-last]
    tpot = (last-f)*1000/(target-1) if target>1 else None
    if (abs(sum(components)-(completed-a))*1000>1
            or (target==1 and native.get('native_tpot_ms') is not None)
            or (target>1 and (type(native.get('native_tpot_ms')) not in (int,float)
                or not math.isfinite(native['native_tpot_ms']) or abs(native['native_tpot_ms']-tpot)>1))):
        raise ValueError('HydraServe E2E/TPOT identity differs')
    result = dict(protocol_valid=True, lora_numerical_correctness_qualified=False,
        ttft_ms=(f-a)*1000,tpot_ms=tpot,e2e_ms=(completed-a)*1000,
        submit_lag_ms=components[0]*1000,dispatch_wait_after_submit_ms=components[1]*1000,
        service_ttft_ms=components[2]*1000,decode_ms=components[3]*1000,
        completion_notification_ms=components[4]*1000,router_queue_ms=(times[6]-times[5])*1000,
        ingress_wait_ms=(times[4]-times[3])*1000,instance_assigned_s=d)
    if 'artifact_content_sha256' in prepared:
        result.update(artifact_resolve_ms=(a_times[1]-a_times[0])*1000,
            artifact_cache_hit=hit, artifact_transferred_bytes=size)
    return result


class NativeHTTPAdapter:
    """Service-owned startup wait and input adapter, not a replacement scheduler."""
    def __init__(self, *, public_model, native_model, registry, views, platform, clock_id, journal):
        self.public_model,self.native_model = public_model,native_model
        self.registry,self.views,self.platform = registry,views,platform
        self.clock_id,self.journal = clock_id,Path(journal)
        self.scheduler = None
        self.ready = asyncio.Event()
        self.idle = None
        self.seen = set()

    def emit(self, event, **fields):
        self.log.write(json.dumps(dict(event=event,at=time.perf_counter(),**fields),separators=(',',':'))+'\n')
        self.log.flush()

    def set_scheduler(self, scheduler):
        if self.scheduler is not None or self.native_model not in scheduler.model_list:
            raise ValueError('invalid native controller binding')
        self.scheduler = scheduler
        self.ready.set()  # Controller exists; engines need NOT be ready.

    async def handle(self, request):
        from aiohttp import web
        received=time.perf_counter()
        rid=None
        bound=None
        try:
            body=await request.json()
            c=decode_request(body,self.registry,self.public_model)
            rid=c.request_id
            if rid in self.seen:
                return web.json_response(dict(error='duplicate request not retried'),status=409)
            self.seen.add(rid)
            bound=BoundRequest(c,artifact_preparer=self.views)
            bound.control.update(http_received_s=received,clock_id=self.clock_id,
                                 canonical_prompt_sha256=c.prompt_sha256)
            self.emit('received',request_id=rid)
            await self.ready.wait()
            bound.control['controller_forwarded_s']=time.perf_counter()
            stream=await self.scheduler.process_request(rid,self.native_model,bound,stream=True)
            if self.idle is None:
                self.idle=asyncio.create_task(self.scheduler.check_instance_loop(30))
            await consume_native_stream(stream)
            completed=time.perf_counter()
            proof=observe_worker(bound,self.platform,self.clock_id)
            result=response_from_bound(bound,proof,completed)
            self.emit('completed',request_id=rid,native_request_id=bound.native_request_id)
            return web.json_response(result)
        except asyncio.CancelledError:
            self.emit('cancelled',request_id=rid)
            raise
        except Exception as exc:
            self.emit('failed',request_id=rid,error=type(exc).__name__+': '+str(exc),
                native_events=[] if bound is None else bound.events,
                control=None if bound is None else bound.control,
                artifact=None if bound is None else bound.artifact_evidence)
            return web.json_response(dict(error=type(exc).__name__+': '+str(exc)),status=500)

    async def start(self, port):
        from aiohttp import web
        self.log=self.journal.open('x')
        app=web.Application()
        app.router.add_post('/v1/chat/completions',self.handle)
        self.runner=web.AppRunner(app,access_log=None,handler_cancellation=True,shutdown_timeout=10.)
        await self.runner.setup()
        site=web.TCPSite(self.runner,'127.0.0.1',port)
        await site.start()
        self.emit('listening',port=site._server.sockets[0].getsockname()[1])
        return site._server.sockets[0].getsockname()[1]

    async def close(self):
        await self.runner.cleanup()
        if self.idle is not None:
            self.idle.cancel()
            await asyncio.gather(self.idle,return_exceptions=True)
        self.log.close()
