"""Opt-in host-span observations of the qualified native core, not a policy.

No CUDA calls, GPU readbacks, tensors, prompt text, request IDs or RPC payloads.
All wrappers invoke the original exactly once and preserve its result/exception.
Detailed observation adds overhead: these runs are NOT performance comparisons.
"""
import atexit
from functools import wraps
import json
import math
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

    def __init__(self, stream, scheduler, *, max_bytes=MAX_BYTES, max_spans=MAX_SPANS,
                 step_components=False, host_inventory=False):
        self.stream, self.scheduler = stream, scheduler
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.max_bytes, self.max_spans = max_bytes, max_spans
        self.byte_count = self.span_count = self.sequence = 0
        self.observer_ns = self.write_ns = 0
        self.stack = []
        self.closed = self.truncated = False
        self.core = None
        self.step_components = step_components
        self.component_rows = None
        self.component_targets = []
        self.host_stats = None
        self.host_inventory = host_inventory
        self.inventory_targets = []

    def attach_host_inventory(self, module):
        """Time two whole functions, never each tensor or an RPC wait interval.

        These module-level functions may also be used outside the native core.
        Observe only this owner's active spans; other calls execute untouched.
        No storage, model, pointer, request payload or returned tensor is retained.
        """
        if not self.host_inventory or self.inventory_targets:
            raise ValueError('HOST inventory diagnostic absent or already installed')
        names = (('_ieee_lora_host_inventory', 'host_inventory'),
                 ('_ieee_pinned_host_allocator_stats', 'host_allocator_stats'))
        originals = [(name, label, getattr(module, name)) for name, label in names]
        if any(not callable(method) or hasattr(method, '_ieee_inventory_timing_owner')
               for _, _, method in originals):
            raise ValueError('invalid or already observed HOST inventory boundary')

        def wrapper(original, label):
            @wraps(original)
            def observed(*args, **kwargs):
                if (self.closed or self.truncated or not self.stack
                        or os.getpid() != self.pid or threading.get_ident() != self.thread):
                    return original(*args, **kwargs)
                return self.call(label, original, args, kwargs)
            observed._ieee_inventory_timing_owner = self
            return observed

        for name, label, original in originals:
            observed = wrapper(original, label)
            setattr(module, name, observed)
            self.inventory_targets.append((module, name, original, observed))

    def restore_host_inventory(self):
        # Never overwrite an unrelated later change, including on failed install.
        for module, name, original, observed in self.inventory_targets:
            if getattr(module, name) is observed:
                setattr(module, name, original)
        self.inventory_targets.clear()

    def attach_step_components(self, core):
        """Observe the actual V2 worker only, after native engine initialization.

        Instance methods are timed without CUDA events/readback or replacing
        native futures. No timers run outside a step. Inclusive nested times
        must not be summed; process allocator counters are not causal timings.
        """
        if self.component_targets:
            return
        import vllm
        import torch
        from vllm.v1.executor.uniproc_executor import UniProcExecutor
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        executor = core.model_executor
        if vllm.__version__ != '0.30.0' or type(executor) is not UniProcExecutor:
            raise ValueError('step components require native vLLM 0.30 UniProc')
        runner = executor.driver_worker.worker.model_runner
        parallel = runner.vllm_config.parallel_config
        if (type(runner) is not GPUModelRunner
                or not core.async_scheduling or core.batch_queue_size != 2
                or any(getattr(parallel, key) != 1 for key in
                    ('tensor_parallel_size', 'pipeline_parallel_size', 'data_parallel_size'))
                or runner.speculative_config is not None or runner.is_pooling_model
                or runner.pcp_manager is not None or runner.batch_sharder is not None
                or runner.uses_inputs_embeds or runner.lora_config is None):
            raise ValueError('step components require qualified dense V2 async TP/PP/DP=1 LoRA')
        targets = [(runner, name, name) for name in (
            'execute_model', 'add_requests', 'update_requests', 'prepare_inputs',
            'prepare_attn', '_set_active_loras', 'sample_tokens')]
        targets += [(runner.block_tables, 'apply_staged_writes', 'block_table_writes'),
                    (runner.cudagraph_manager, 'run_fullgraph', 'fullgraph'),
                    (runner.cudagraph_manager, 'run_pw_graph', 'piecewise_graph'),
                    (self.scheduler, 'update_from_output', 'update_from_output')]
        self._attach_component_targets(targets, torch.cuda.host_memory_stats)

    def _attach_component_targets(self, targets, host_stats):
        # Validate ALL methods before modifying any, including collision checks.
        originals = [(obj, name, label, getattr(obj, name)) for obj, name, label in targets]
        if (not callable(host_stats) or len({label for _, _, label, _ in originals}) != len(originals)
                or any(not callable(method) for _, _, _, method in originals)):
            raise ValueError('invalid step component boundaries')
        if self.component_targets:
            raise ValueError('step components already installed')
        def wrapper(method, label):
            @wraps(method)
            def observed(*args, **kwargs):
                if self.component_rows is None or self.closed or self.truncated:
                    return method(*args, **kwargs)
                self._owner()
                begin, cpu_begin = time.monotonic_ns(), time.thread_time_ns()
                raised = 1
                try:
                    result = method(*args, **kwargs)
                    raised = 0
                    return result
                finally:
                    cpu_end, end = time.thread_time_ns(), time.monotonic_ns()
                    values = self.component_rows.setdefault(label, [0, 0, 0, 0])
                    values[0] += 1
                    values[1] += end - begin
                    values[2] += cpu_end - cpu_begin
                    values[3] += raised
            return observed
        for obj, name, label, method in originals:
            observed = wrapper(method, label)
            setattr(obj, name, observed)
            self.component_targets.append((obj, name, method, observed))
        self.host_stats = host_stats

    def _allocator_counts(self):
        stats = self.host_stats()
        result = {key: stats.get(key) for key in ('num_host_alloc', 'num_host_free')}
        if any(type(value) is not int or value < 0 for value in result.values()):
            raise ValueError('native host allocation counters unavailable')
        for key in ('host_alloc_time.total', 'host_free_time.total'):
            value = stats.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('native host allocation time counters unavailable')
            result[key] = value  # Native microseconds, process-wide, not GPU time.
        return result

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
        components = name == 'step' and self.step_components
        if components:
            if self.component_rows is not None or not self.component_targets:
                raise ValueError('step component scope missing or reentrant')
            alloc_before = self._allocator_counts()
            self.component_rows = {}
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
            if components:
                values, self.component_rows = self.component_rows, None
                alloc_after = self._allocator_counts()
                if any(alloc_after[key] < alloc_before[key] for key in alloc_before):
                    raise ValueError('host allocator counters reset during observation')
                details['step_components_v1'] = dict(
                    columns=['calls', 'inclusive_wall_ns', 'inclusive_thread_cpu_ns', 'raised'],
                    values=values, nested_not_additive=True,
                    allocator_before=alloc_before, allocator_after=alloc_after,
                    allocator_scope='process_counters_background_frees_possible',
                    allocator_time_unit='microseconds',
                    gpu_kernel_timing=False)
            if outcome == 'returned' and name == 'schedule':
                counts = result.num_scheduled_tokens
                details.update(scheduled_requests=len(counts),
                    scheduled_tokens=result.total_num_scheduled_tokens,
                    multi_token_requests=sum(n > 1 for n in counts.values()))
                # >1 token allocations are not labelled pure GPU prefill.
            elif outcome == 'returned' and name == 'step':
                details['model_executed'] = result if type(result) is bool else None
            elif outcome == 'returned' and name == 'host_inventory' and type(result) is dict:
                # Scalar projection only; do not traverse or serialize tensors.
                scopes = ('native_registered_tensor_storage_capacity',
                          'native_registered_and_staged_tensor_storage_capacity',
                          'native_requested_tensor_storage_capacity')
                scope = result.get('host_footprint_scope')
                details['scope'] = scope if scope in scopes else 'other'
                for field, label in (('host_allocations', 'allocation_count'),
                                     ('host_adapter_footprints', 'adapter_count')):
                    value = result.get(field)
                    if type(value) is list:
                        details[label] = len(value)
                for field in ('host_tensor_storage_bytes', 'requested_host_storage_bytes'):
                    value = result.get(field)
                    if type(value) is int and value >= 0:
                        details[field] = value
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
        self.restore_host_inventory()

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
                if label == 'step' and self.step_components:
                    self.attach_step_components(core)
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
    component_flag = os.environ.get('FAASLORA_TC_CORE_STEP_TIMING')
    inventory_flag = os.environ.get('FAASLORA_TC_HOST_INVENTORY_TIMING')
    if inventory_flag not in (None, '0', 'fresh_inventory_v1'):
        raise ValueError('unsupported HOST inventory diagnostic')
    inventory = inventory_flag == 'fresh_inventory_v1'
    if component_flag not in (None, '0', 'v2_components_v1'):
        raise ValueError('unsupported native step component diagnostic')
    components = component_flag == 'v2_components_v1'
    if flag is None or flag == '0':
        if components or inventory:
            raise ValueError('component observations require host_spans_v1 diagnostic')
        return None
    if flag != 'host_spans_v1':
        raise ValueError('unsupported native core timing diagnostic')
    receipt_path, actual, replay = diagnostic_service_receipt('native core timing')
    existing = getattr(core_class, '_ieee_core_host_timing', None)
    if existing is not None:
        existing._owner()
        if existing.scheduler is not scheduler:
            raise ValueError('native core timing already has another scheduler')
        if existing.step_components != components:
            raise ValueError('native step diagnostic identity changed')
        if existing.host_inventory != inventory:
            raise ValueError('HOST inventory diagnostic identity changed')
        return existing
    directory = receipt_path.parent/'diagnostic_core'
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.is_symlink():
        raise ValueError('native core timing output must not be a symlink')
    ticks = int(Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19])
    path = directory/f'{os.getpid()}-{ticks}.jsonl'
    stream = path.open('xb', buffering=0)
    observer = CoreHostTiming(stream, scheduler, step_components=components,
                              host_inventory=inventory)
    try:
        observer._write(dict(event='native_core_host_timing_v1', pid=observer.pid,
            start_ticks=ticks, thread_ident=observer.thread, cgroup=str(actual),
            clock_id=local_monotonic_clock_id(),
            replay_scope=replay['replay_scope'], diagnostic_prefix_count=replay['diagnostic_prefix_count'],
            max_bytes=observer.max_bytes, max_spans=observer.max_spans,
            formal_performance_result=False, gpu_kernel_timing=False,
            includes_queue_idle_wait=True, nested_spans_not_additive=True,
            observer_overhead_included=True, abrupt_exit_may_lose_active_spans=True,
            **({'step_components': 'v2_components_v1',
                'allocator_counter_units': {'count': 'blocks', 'time': 'microseconds'},
                'component_times_nested_not_additive': True} if components else {}),
            **({'host_inventory_timing': 'fresh_inventory_v1',
                'inventory_scope': 'owner_thread_calls_inside_observed_core_spans',
                'inventory_timing': 'whole_function_no_per_tensor_instrumentation'}
               if inventory else {})))
        # The original header/schema stays unchanged for the default diagnostic.
        # Optional component payloads have their own explicit versioned key.
        if inventory:
            from faaslora.memory import gpu_monitor
            observer.attach_host_inventory(gpu_monitor)
        observer.attach(core_class)
    except BaseException:
        observer.restore_host_inventory()
        stream.close()
        raise
    atexit.register(observer.close)
    return observer
