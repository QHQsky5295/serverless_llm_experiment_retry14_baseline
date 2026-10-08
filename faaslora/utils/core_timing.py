"""Opt-in host-span observations of the qualified native core, not a policy.

No CUDA calls, GPU readbacks, tensors, prompt text, request IDs or RPC payloads.
All wrappers invoke the original exactly once and preserve its result/exception.
Detailed observation adds overhead: these runs are NOT performance comparisons.
"""
import atexit
from functools import wraps
import json
import os
from pathlib import Path
import threading
import time

from faaslora.clock import local_monotonic_clock_id
from .logger import diagnostic_service_receipt


_UTILITIES = frozenset(('collective_rpc', 'ieee_scheduler_observation',
    'ieee_request_retirement', 'ieee_prepare_host', 'ieee_admit_and_demand_load',
    'ieee_pending_admission', 'ieee_pending_gpu_reference', 'ieee_transfer_event',
    'add_lora', 'remove_lora', 'pin_lora', 'list_loras', 'check_health'))
_OPERATIONS = frozenset(('acquire', 'release', 'completion_fence', 'demand_load_and_acquire',
    'proactive_host_prepare_and_acquire', 'admit_and_demand_load', 'request_source_snapshot',
    'routing_source_snapshot', 'source_identity_snapshot', 'source_snapshot',
    'snapshot', 'register', 'bind', 'withdraw'))


def _client_label(args):
    # Only constants are emitted; never stringify a request or unknown method.
    kind = getattr(args[0], 'name', None) if args else None
    kind = kind if kind in ('ADD', 'ABORT', 'UTILITY', 'WAKEUP', 'EXECUTOR_FAILED') else 'OTHER'
    label = {'request_type': kind}
    if kind == 'UTILITY' and len(args) == 2:
        request = args[1]
        if isinstance(request, (tuple, list)) and len(request) == 4:
            method, params = request[2:]
            label['utility'] = method if type(method) is str and method in _UTILITIES else 'other'
            if method == 'collective_rpc' and isinstance(params, (list, tuple)) and params:
                label['worker_method'] = 'ieee_gpu_reference' if params[0] == 'ieee_gpu_reference' else 'other'
                if len(params) == 4 and type(params[3]) is dict:
                    operation = params[3].get('operation')
                    label['operation'] = operation if type(operation) is str and operation in _OPERATIONS else 'other'
            elif method == 'ieee_pending_admission' and isinstance(params, (list, tuple)) and params:
                label['operation'] = params[0] if type(params[0]) is str and params[0] in _OPERATIONS else 'other'
    return label


class CoreHostTiming:
    """Single-owner, capped append-only timeline; nested spans are not additive."""
    MAX_BYTES = 64 * 1024 * 1024
    MAX_SPANS = 200000
    RESERVE_BYTES = 4096

    def __init__(self, stream, scheduler, *, max_bytes=MAX_BYTES, max_spans=MAX_SPANS):
        self.stream, self.scheduler = stream, scheduler
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.max_bytes, self.max_spans = max_bytes, max_spans
        self.byte_count = self.span_count = self.sequence = 0
        self.observer_ns = self.write_ns = 0
        self.stack = []
        self.closed = self.truncated = False
        self.core = None

    def _write(self, row):
        start = time.monotonic_ns()
        payload = (json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n').encode()
        if self.stream.write(payload) != len(payload):
            raise OSError('incomplete native core timing write')
        self.byte_count += len(payload)
        self.write_ns += time.monotonic_ns()-start

    def _owner(self):
        if os.getpid() != self.pid or threading.get_ident() != self.thread:
            raise ValueError('native core timing owner changed')

    def call(self, name, original, args, kwargs):
        self._owner()
        if self.closed or self.truncated:
            return original(*args, **kwargs)
        observe_start = time.monotonic_ns()
        if len(self.stack) >= 16:
            raise ValueError('unexpected native core timing nesting')
        self.sequence += 1
        sequence, parent = self.sequence, self.stack[-1] if self.stack else None
        details = _client_label(args) if name == 'client' else {}
        if name in ('input_queue', 'step'):
            details.update(unfinished_before=len(self.scheduler.requests),
                           running_before=len(self.scheduler.running),
                           waiting_before=len(self.scheduler.waiting))
        self.stack.append(sequence)
        begin = time.monotonic_ns()
        cpu_begin = time.thread_time_ns()
        outcome, result = 'raised', None
        try:
            result = original(*args, **kwargs)
            outcome = 'returned'  # Native utility may itself encode a failure or Future.
            return result
        finally:
            cpu_end, end = time.thread_time_ns(), time.monotonic_ns()
            self.stack.pop()
            if outcome == 'returned' and name == 'schedule':
                counts = result.num_scheduled_tokens
                details.update(scheduled_requests=len(counts),
                    scheduled_tokens=result.total_num_scheduled_tokens,
                    multi_token_requests=sum(n > 1 for n in counts.values()))
                # >1 token allocations are not labelled pure GPU prefill.
            elif outcome == 'returned' and name == 'step':
                details['model_executed'] = result if type(result) is bool else None
            row = dict(event='native_core_host_span_v1', id=sequence, parent=parent,
                boundary=name, begin_ns=begin, end_ns=end,
                thread_cpu_begin_ns=cpu_begin, thread_cpu_end_ns=cpu_end,
                outcome=outcome, details=details,
                preceding_observer_ns=self.observer_ns, preceding_write_ns=self.write_ns)
            if not self.truncated:
                encoded_size = len(json.dumps(row, separators=(',', ':'), allow_nan=False).encode())+1
                if (self.span_count >= self.max_spans
                        or self.byte_count+encoded_size > self.max_bytes-self.RESERVE_BYTES):
                    self.truncated = True
                    self._write(dict(event='observation_limit', monotonic_ns=end,
                                     completed_spans=self.span_count, first_unwritten_span=sequence))
                else:
                    self._write(row)
                    self.span_count += 1
            self.observer_ns += begin-observe_start + time.monotonic_ns()-end

    def close(self):
        if self.closed:
            return
        self._owner()
        self._write(dict(event='observation_close', monotonic_ns=time.monotonic_ns(),
            completed_spans=self.span_count, truncated=self.truncated,
            unfinished_span_ids=list(self.stack), observer_ns=self.observer_ns,
            write_ns=self.write_ns, bytes_before_close=self.byte_count))
        self.closed = True
        self.stream.close()

    def attach(self, core_class):
        existing = getattr(core_class, '_ieee_core_host_timing', None)
        if existing is not None:
            if existing is not self:
                raise ValueError('native core timing already has another owner')
            return
        # Validate every boundary before changing any method.
        names = (('_process_input_queue', 'input_queue'), ('_handle_client_request', 'client'),
                 ('_process_engine_step', 'step'))
        originals = [(attribute, label, getattr(core_class, attribute)) for attribute, label in names]
        shutdown = getattr(core_class, 'shutdown')
        scheduled = self.scheduler.schedule

        def wrapper(original, label):
            @wraps(original)
            def observed(core, *args, **kwargs):
                if core.scheduler is not self.scheduler or (self.core is not None and core is not self.core):
                    raise ValueError('native timing invoked by another core')
                self.core = core
                return self.call(label, lambda *a, **k: original(core, *a, **k), args, kwargs)
            return observed

        @wraps(shutdown)
        def observed_shutdown(core, *args, **kwargs):
            if core.scheduler is not self.scheduler or (self.core is not None and core is not self.core):
                raise ValueError('native timing shutdown by another core')
            try:
                return shutdown(core, *args, **kwargs)
            finally:
                self.close()

        @wraps(scheduled)
        def observed_schedule(*args, **kwargs):
            return self.call('schedule', scheduled, args, kwargs)

        for attribute, label, original in originals:
            setattr(core_class, attribute, wrapper(original, label))
        core_class.shutdown = observed_shutdown
        self.scheduler.schedule = observed_schedule
        core_class._ieee_core_host_timing = self


def install_core_host_timing(core_class, scheduler):
    flag = os.environ.get('FAASLORA_TC_CORE_TIMING')
    if flag is None or flag == '0':
        return None
    if flag != 'host_spans_v1':
        raise ValueError('unsupported native core timing diagnostic')
    receipt_path, actual, replay = diagnostic_service_receipt('native core timing')
    existing = getattr(core_class, '_ieee_core_host_timing', None)
    if existing is not None:
        existing._owner()
        if existing.scheduler is not scheduler:
            raise ValueError('native core timing already has another scheduler')
        return existing
    directory = receipt_path.parent/'diagnostic_core'
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.is_symlink():
        raise ValueError('native core timing output must not be a symlink')
    ticks = int(Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19])
    path = directory/f'{os.getpid()}-{ticks}.jsonl'
    stream = path.open('xb', buffering=0)
    observer = CoreHostTiming(stream, scheduler)
    try:
        observer._write(dict(event='native_core_host_timing_v1', pid=observer.pid,
            start_ticks=ticks, thread_ident=observer.thread, cgroup=str(actual),
            clock_id=local_monotonic_clock_id(),
            replay_scope=replay['replay_scope'], diagnostic_prefix_count=replay['diagnostic_prefix_count'],
            max_bytes=observer.max_bytes, max_spans=observer.max_spans,
            formal_performance_result=False, gpu_kernel_timing=False,
            includes_queue_idle_wait=True, nested_spans_not_additive=True,
            observer_overhead_included=True, abrupt_exit_may_lose_active_spans=True))
        observer.attach(core_class)
    except BaseException:
        stream.close()
        raise
    atexit.register(observer.close)
    return observer
