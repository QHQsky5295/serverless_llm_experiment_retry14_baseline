"""Opt-in common workload plumbing through the ORIGINAL HydraServe controller.

No scheduler, placement, scaling, loading, LoRA eviction or batch policy lives
here. FrozenRequest travels in the existing Request.prompt slot. Only ChatEngine
recognizes it and calls the native completion API without a second chat template.
Native request binding is evidence of propagation, NOT numerical correctness.
"""
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace

from . import platform_port
from .native_delivery import replace_one

ENGINE = platform_port.SOURCE_DIR + '/engine.py'
COMPLETION = 'vllm/entrypoints/openai/serving_completion.py'
OUTPUTS = 'vllm/outputs.py'
SOURCE_SHA = {
    ENGINE: 'c0722e5512fd7ef88a19146f85850d9715fe8002baf49de4fe33aed377174337',
    COMPLETION: '4be1f3160c82d79c7fdedee3b5b9171db82596dd3a7ddb14b2ffbeeb46b28b29',
    OUTPUTS: 'ba851a17c65be15524cb54ffc331af2ad032c87b4cb58a99168ef774c59a0553',
}
PREFIX = 'ieee-tc-hydra:'


@dataclass(frozen=True)
class FrozenRequest:
    request_id: str
    adapter_id: str
    adapter_index: int
    adapter_path: str
    canonical_prompt: str
    prompt_token_ids: tuple
    target_tokens: int

    def __post_init__(self):
        if (not isinstance(self.request_id, str)
                or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', self.request_id)
                or not isinstance(self.adapter_id, str)
                or not re.fullmatch(r'[A-Za-z0-9_-]+', self.adapter_id)
                or type(self.adapter_index) is not int or self.adapter_index <= 0
                or not isinstance(self.adapter_path, str)
                or not Path(self.adapter_path).is_absolute()
                or '..' in Path(self.adapter_path).parts
                or Path(self.adapter_path).name != self.adapter_id
                or not isinstance(self.canonical_prompt, str) or not self.canonical_prompt
                or type(self.prompt_token_ids) is not tuple
                or not 1 <= len(self.prompt_token_ids) <= 760
                or any(type(t) is not int or t < 0 for t in self.prompt_token_ids)
                or type(self.target_tokens) is not int or not 1 <= self.target_tokens <= 256):
            raise ValueError('invalid frozen HydraServe request contract')

    @property
    def prompt_sha256(self):
        return sha256(self.canonical_prompt.encode()).hexdigest()

    def completion_arguments(self):
        return dict(model=self.adapter_id, prompt=list(self.prompt_token_ids),
            max_tokens=self.target_tokens, temperature=0, top_p=1, stop=[],
            n=1, stream=True, user=PREFIX+self.request_id,
            extra_body=dict(ignore_eos=True, stop_token_ids=[]))


@dataclass
class BoundRequest:
    contract: FrozenRequest
    events: list = field(default_factory=list)
    terminal: bool = False
    native_request_id: str = None
    native_process: dict = None
    artifact_preparer: object = None
    artifact_evidence: dict = None
    control: dict = field(default_factory=dict)


def mark_bound(prompt, event, **fields):
    """Observe an actual native boundary once, without making a decision."""
    if not isinstance(prompt, BoundRequest):
        return
    if event in prompt.control or set(fields) & prompt.control.keys():
        raise ValueError('duplicate HydraServe control observation')
    prompt.control.update(fields)
    prompt.control[event] = time.perf_counter()


async def wait_for_native_assignment(request, model):
    """Cancel exactly the native queue reservation whose owner has not started.

    Native assignment and this coroutine run on the same event loop. If the
    assignment event was set just before cancellation, Instance.chat has been
    created but not awaited: close that unstarted coroutine and release its
    reservation here. Once the await returns, Instance.chat owns cancellation
    and its existing exception path releases the counters. Do not reset demand
    history, change FIFO order, or cancel other requests.
    """
    import asyncio
    import inspect
    try:
        await request.resp_event.wait()
    except asyncio.CancelledError:
        if request.resp_func is None:
            model.waiting_queue.remove(request)
        else:
            if inspect.getcoroutinestate(request.resp_func) != inspect.CORO_CREATED:
                raise RuntimeError('native assignment cancellation has no unstarted owner')
            request.resp_func.close()
            request.assigned_instance.num_reqs -= 1
            request.assigned_instance.last_chat_time = time.time()
        raise


async def prepare_bound_artifact(prompt):
    """Optional on-demand delivery after native assignment/init, before submit.

    This is a data-path prerequisite, not a routing signal or a prefetch. The
    original five-second backend submission timeout remains unchanged; bounded
    remote I/O has its own common transport deadline and cancellation contract.
    """
    if not isinstance(prompt, BoundRequest) or prompt.artifact_preparer is None:
        return
    if prompt.artifact_evidence is not None or prompt.native_request_id is not None:
        raise ValueError('artifact preparation cannot reuse a bound request')
    evidence = await prompt.artifact_preparer.prepare(prompt.contract)
    c = prompt.contract
    if (not isinstance(evidence, dict)
            or evidence.get('tc_artifact_request_id') != c.request_id
            or evidence.get('tc_artifact_id') != c.adapter_id
            or evidence.get('native_adapter_path') != c.adapter_path):
        raise ValueError('prepared artifact differs from native request')
    prompt.artifact_evidence = evidence


class CompletionAsChatStream:
    """Preserve original synchronous stream consumption and profiling formulas.

    Translate only response shape. Token IDs/timestamps are validated separately
    from text; empty or multi-token chunks must not alter work or token counting.
    No retries, tokenization, padding, base-model fallback or fabricated events.
    """
    def __init__(self, stream, bound):
        if bound.events or bound.terminal or bound.native_request_id is not None:
            stream.close()
            raise ValueError('HydraServe request binding cannot be reused')
        self.stream, self.bound = stream, bound
        self.count = 0

    def __iter__(self):
        c, b = self.bound.contract, self.bound
        try:
            for response in self.stream:
                data = response.model_dump()
                event = data.get('hydraserve_native')
                if not isinstance(event, dict) or len(data.get('choices', [])) != 1:
                    raise ValueError('native HydraServe event missing')
                choice = data['choices'][0]
                ids = event.get('delta_token_ids')
                stamp = event.get('output_monotonic_s')
                process = event.get('process')
                rid = event.get('native_request_id')
                if (b.terminal or event.get('request_id') != c.request_id
                        or event.get('adapter_id') != c.adapter_id
                        or type(event.get('adapter_index')) is not int
                        or event['adapter_index'] != c.adapter_index
                        or event.get('adapter_path') != c.adapter_path
                        or event.get('prompt_token_ids') != list(c.prompt_token_ids)
                        or type(ids) is not list
                        or any(type(t) is not int or t < 0 for t in ids)
                        or type(stamp) not in (int, float) or not math.isfinite(stamp)
                        or stamp <= 0 or (b.events and stamp < b.events[-1]['output_monotonic_s'])
                        or not isinstance(process, dict)
                        or any(type(process.get(k)) is not int or process[k] <= 0
                               for k in ('pid', 'start_ticks'))
                        or not isinstance(process.get('boot_id'), str) or not process['boot_id']
                        or not isinstance(rid, str) or not rid
                        or (b.native_request_id is not None and b.native_request_id != rid)
                        or (b.native_process is not None and b.native_process != process)
                        or choice.get('index') != 0
                        or not isinstance(choice.get('text'), str)):
                    raise ValueError('native HydraServe binding/token/clock mismatch')
                b.native_request_id, b.native_process = rid, dict(process)
                self.count += len(ids)
                if (self.count > c.target_tokens
                        or event.get('cumulative_tokens') != self.count
                        or type(event.get('cumulative_tokens')) is not int):
                    raise ValueError('native HydraServe output count mismatch')
                finish = choice.get('finish_reason')
                if finish is not None:
                    usage = data.get('usage')
                    if (finish != 'length' or self.count != c.target_tokens
                            or not isinstance(usage, dict)
                            or type(usage.get('completion_tokens')) is not int
                            or usage['completion_tokens'] != self.count
                            or usage.get('prompt_tokens') != len(c.prompt_token_ids)):
                        raise ValueError('native HydraServe terminal contract mismatch')
                    b.terminal = True
                elif not ids or data.get('usage') is not None:
                    raise ValueError('empty/nonterminal native HydraServe output')
                b.events.append(event)
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=choice['text']))],
                    usage=SimpleNamespace(**data['usage']) if data.get('usage') else None)
            if not b.terminal:
                raise ValueError('native HydraServe stream ended without terminal event')
        finally:
            self.stream.close()


_PROCESS = None


def native_process():
    global _PROCESS
    pid = os.getpid()
    if _PROCESS is None or _PROCESS['pid'] != pid:
        stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
        _PROCESS = dict(pid=pid, start_ticks=int(stat[19]),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    return dict(_PROCESS)


def matches_worker_process(process, owner, entry, boot_id):
    """Match container PID to its entry receipt, host birth to actual owner.

    Container PID need not equal host PID. This pure receipt check does not
    replace live ownership/cgroup/namespace validation by the platform.
    """
    return (isinstance(process, dict) and isinstance(owner, dict) and isinstance(entry, dict)
        and type(process.get('pid')) is int and process['pid'] > 0
        and type(owner.get('pid')) is int and owner['pid'] > 0
        and type(process.get('start_ticks')) is int and process['start_ticks'] > 0
        and process['pid'] == entry.get('pid')
        and process['start_ticks'] == owner.get('start_ticks') == entry.get('start_ticks')
        and process.get('boot_id') == boot_id and isinstance(boot_id, str) and bool(boot_id))


def frozen_lora_modules():
    """Register static identity/path metadata only; never pre-load adapter bytes.

    Paths must be materialized by a separately qualified request-driven artifact
    owner before native loading. This does NOT implement remote fetching or
    assert that files exist, are verified, resident or backend-executable.
    """
    from vllm.entrypoints.openai.serving_engine import LoRAModulePath
    raw = Path(os.environ['HYDRASERVE_LORA_REGISTRY']).read_bytes()
    if sha256(raw).hexdigest() != os.environ['HYDRASERVE_LORA_REGISTRY_SHA256']:
        raise ValueError('HydraServe LoRA registry identity drift')
    rows = json.loads(raw)
    if (type(rows) is not list or not rows
            or any(type(r) is not dict or set(r) != {'adapter_id', 'adapter_index', 'adapter_path'}
                   for r in rows)):
        raise ValueError('invalid static HydraServe LoRA registry')
    for index, row in enumerate(rows, 1):
        # Reuse the identity/type validator, without reading weights or trace.
        FrozenRequest('registry', row['adapter_id'], row['adapter_index'],
                      row['adapter_path'], 'metadata only', (1,), 1)
        if row['adapter_index'] != index:
            raise ValueError('registry must match native enumeration identity')
    if len({r['adapter_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate static HydraServe adapter ID')
    return [LoRAModulePath(name=r['adapter_id'], local_path=r['adapter_path']) for r in rows]


def enable_app_registration(source):
    """Called only on already source-pinned D495 app; affects both native stages."""
    old = 'OpenAIServingCompletion(engine, served_model_names)'
    if source.count(old) != 2:
        raise ValueError('native completion registration sites changed')
    source = replace_one(source, 'from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion\n',
        'from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion\n'
        'from HydraServe_project.runtime.request_contract import frozen_lora_modules\n')
    return source.replace(old,
        'OpenAIServingCompletion(engine, served_model_names, lora_modules=frozen_lora_modules())')


def annotate_native_event(response_json, request, res, output, delta_token_ids):
    """Use immutable native output-construction time, not HTTP receive time."""
    if not (isinstance(request.user, str) and request.user.startswith(PREFIX)):
        return response_json
    if (request.n != 1 or request.echo or not request.ignore_eos
            or request.temperature != 0 or request.top_p != 1
            or request.stop or request.stop_token_ids or not res.lora_request
            or res.lora_request.lora_name != request.model
            or not hasattr(res, 'tc_output_monotonic_s')):
        raise ValueError('invalid native fixed-output adapter request')
    lora = res.lora_request
    payload = json.loads(response_json)
    payload['hydraserve_native'] = dict(
        request_id=request.user[len(PREFIX):], native_request_id=res.request_id,
        adapter_id=lora.lora_name, adapter_index=lora.lora_int_id,
        adapter_path=lora.lora_local_path, prompt_token_ids=list(res.prompt_token_ids),
        delta_token_ids=list(delta_token_ids), cumulative_tokens=len(output.token_ids),
        output_monotonic_s=res.tc_output_monotonic_s, process=native_process(),
        token_source='native_RequestOutput.token_ids',
        timing_source='native_RequestOutput_construction_monotonic',
        evidence_scope='native_sequence_binding_not_numerical_correctness')
    return json.dumps(payload, separators=(',', ':'))


def render_overlays(native_root, *, control_observations=False, cancellation_ownership=False):
    """Compose D480 lifecycle port; exact source identities before each splice."""
    root = Path(native_root)
    originals = {}
    for name, expected in SOURCE_SHA.items():
        raw = (root/name).read_bytes()
        if sha256(raw).hexdigest() != expected:
            raise ValueError('unqualified HydraServe request source: '+name)
        originals[name] = raw.decode()
    result = platform_port.render_overlays(root)
    source = originals[ENGINE]
    source = replace_one(source,
        '        if stream:\n',
        '        from HydraServe_project.runtime.request_contract import BoundRequest, CompletionAsChatStream\n'
        '        if isinstance(prompt, BoundRequest):\n'
        '            if not stream:\n'
        '                raise ValueError("native fixed-output requires streaming")\n'
        '            reply = self.client.completions.create(**prompt.contract.completion_arguments())\n'
        '            return CompletionAsChatStream(reply, prompt)\n'
        '        if stream:\n')
    result[ENGINE] = source.encode()
    # Native output fields otherwise alias mutable metrics. Snapshot at creation
    # before AsyncStream/HTTP buffering, not when the client consumes a chunk.
    source = replace_one(originals[OUTPUTS],
        '        self.metrics = metrics\n',
        '        self.metrics = metrics\n'
        '        self.tc_output_monotonic_s = time.perf_counter()\n')
    source = replace_one(source, '        self.token_ids = token_ids\n',
        '        self.token_ids = list(token_ids)  # snapshot before async buffering\n')
    result[OUTPUTS] = source.encode()
    source = replace_one(originals[COMPLETION],
        '                    yield f"data: {response_json}\\n\\n"\n',
        '                    from HydraServe_project.runtime.request_contract import annotate_native_event\n'
        '                    response_json = annotate_native_event(response_json, request, res, output, delta_token_ids)\n'
        '                    yield f"data: {response_json}\\n\\n"\n')
    result[COMPLETION] = source.encode()
    # Correct terminal accounting on malformed/error/disconnected streams. The
    # original successful stream, profiling arithmetic and counters are retained.
    name = platform_port.SOURCE_DIR+'/scheduler.py'
    source = result[name].decode()
    start = source.index('        stime = time.time()\n', source.index('    def resp_generator('))
    end = source.index('\n    async def chat(', start)
    block = source[start:end]
    split = block.index('        self.num_reqs -= 1\n')
    body, release = block[:split], block[split:].rstrip()+'\n'
    wrapped = ('        try:\n'+''.join('    '+line if line.strip() else line
               for line in body.splitlines(keepends=True))+'        finally:\n'
               +''.join('    '+line for line in release.splitlines(keepends=True)))
    source = source[:start]+wrapped+source[end:]
    # Propagate actual submission failure instead of using an unbound reply and
    # leaving an instance permanently busy. Preserve the native five-second
    # submission timeout and synchronous stream semantics (not tuning).
    marker = '            print("".join(traceback.format_exception(*exc_info)))\n'
    source = replace_one(source, marker, marker+'            raise\n')
    start = source.index('        # Count num_coldstart_reqs', source.index('    async def chat('))
    end = source.index('\nclass Request:', start)
    body = source[start:end].rstrip()+'\n'
    body = replace_one(body, '        self.num_coldstart_reqs += 1\n',
                       '        self.num_coldstart_reqs += 1\n        cold_counted = True\n')
    body = replace_one(body,
        '        if not is_coldstart_req:\n            self.num_coldstart_reqs -= 1\n',
        '        if not is_coldstart_req:\n            self.num_coldstart_reqs -= 1\n'
        '            cold_counted = False\n')
    body = replace_one(body,
        '        try:\n            reply = await asyncio.wait_for(self.chat_engine.chat(prompt, stream), timeout=5)\n',
        '        from HydraServe_project.runtime.request_contract import prepare_bound_artifact\n'
        '        await prepare_bound_artifact(prompt)\n'
        '        try:\n            reply = await asyncio.wait_for(self.chat_engine.chat(prompt, stream), timeout=5)\n')
    source = (source[:start]+'        cold_counted = False\n        try:\n'
        +''.join('    '+line if line.strip() else line for line in body.splitlines(keepends=True))
        +'        except BaseException:\n'
        +'            self.num_reqs -= 1\n'
        +'            if cold_counted:\n                self.num_coldstart_reqs -= 1\n'
        +'            self.last_chat_time = time.time()\n            raise\n'+source[end:])
    result[name] = source.encode()
    if cancellation_ownership:
        source = replace_one(source,
            '        instance.num_reqs += 1\n        self.resp_func = instance.chat',
            '        instance.num_reqs += 1\n        self.assigned_instance = instance\n'
            '        self.resp_func = instance.chat')
        source = replace_one(source,
            '        await request.resp_event.wait()\n',
            '        from HydraServe_project.runtime.request_contract import wait_for_native_assignment\n'
            '        await wait_for_native_assignment(request, model)\n')
        result[name] = source.encode()
    if control_observations:
        # Opt-in instrumentation only: older qualification source remains exact.
        source = replace_one(source, 'KVCacheRequiredSize = 5\n',
            'from HydraServe_project.runtime.request_contract import mark_bound\n\n'
            'KVCacheRequiredSize = 5\n')
        source = replace_one(source,
            '    def add_request(self, request: Request):\n',
            '    def add_request(self, request: Request):\n'
            '        mark_bound(request.prompt, "router_enqueued_s")\n')
        source = replace_one(source,
            '    def allocate_instance(self, instance: Instance):\n',
            '    def allocate_instance(self, instance: Instance):\n'
            '        mark_bound(self.prompt, "instance_assigned_s", instance_id=str(instance.id))\n')
        source = replace_one(source,
            '            is_coldstart_req = True if self.pp_size > 1 else False\n',
            '            mark_bound(prompt, "backend_ready_s")\n'
            '            is_coldstart_req = True if self.pp_size > 1 else False\n')
        result[name] = source.encode()
        source = result[ENGINE].decode()
        source = replace_one(source,
            '            reply = self.client.completions.create(**prompt.contract.completion_arguments())\n',
            '            from HydraServe_project.runtime.request_contract import mark_bound\n'
            '            mark_bound(prompt, "native_http_submitted_s")\n'
            '            reply = self.client.completions.create(**prompt.contract.completion_arguments())\n')
        result[ENGINE] = source.encode()
    for name, source in result.items():
        compile(source, name, 'exec')
    return result
