"""Opt-in fairness between atomic Prime control transactions and native steps.

No thread, cache, extra model step, time budget, or change to admission math.
Adapted input polling is pinned to vLLM 0.30.0; dispatch/step/abort stay native.
"""
import ast
import hashlib
import inspect
import os
import queue
import textwrap
import threading

POLICY = 'atomic_control_yield_v1'
_NATIVE_INPUT_AST_SHA = '5d1a82d46b2a673a0f8e32db251d1251613c6f63794758cea9eae6dd5845a845'
_PRIME_METHODS = frozenset(('ieee_scheduler_observation', 'ieee_request_retirement',
    'ieee_prepare_host', 'ieee_admit_and_demand_load', 'ieee_pending_admission',
    'ieee_pending_gpu_reference', 'ieee_transfer_event'))


def native_input_fingerprint(method):
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('native input method source is not a single function')
    # Python 3.12 adds empty type_params fields to the same nongeneric source.
    # Normalize only that empty representation, not executable AST content.
    for node in ast.walk(tree):
        if hasattr(node, 'type_params') and not node.type_params:
            del node.type_params
    return hashlib.sha256(ast.dump(tree.body[0], include_attributes=False).encode()).hexdigest()


class AtomicControlInterleave:
    def __init__(self, core_class, scheduler, executor_type, request_types):
        self.core_class, self.scheduler = core_class, scheduler
        self.executor_type, self.request_types = executor_type, request_types
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.core = None
        self.input_turns = self.controls = self.yields = self.backlog_yields = 0

    def _owner(self, core):
        if (os.getpid() != self.pid or threading.get_ident() != self.thread
                or type(core) is not self.core_class or core.scheduler is not self.scheduler
                or type(core.model_executor) is not self.executor_type
                or (self.core is not None and self.core is not core)):
            raise RuntimeError('control interleave requires its qualified single native owner')
        self.core = core

    def _prime_control(self, request):
        kind, payload = request
        if kind != self.request_types.UTILITY:
            return False
        # Classification only. Native dispatch validates and executes the payload;
        # never echo private arguments or infer success from a control reply.
        if not isinstance(payload, (tuple, list)) or len(payload) != 4:
            return False
        method, args = payload[2:]
        return (type(method) is str and (method in _PRIME_METHODS or
            (method == 'collective_rpc' and isinstance(args, (tuple, list))
             and len(args) > 0 and args[0] == 'ieee_gpu_reference')))

    def _handle(self, core, request):
        prime = self._prime_control(request)
        core._handle_client_request(*request)  # Original dispatch, exactly once.
        if prime:
            self.controls += 1
        # A complete transaction has returned. A Future-returning retirement
        # utility may still be pending; its original callback owns completion.
        # No claim that returning from dispatch means the whole request is done.
        if prime and core.is_running() and core.has_work():
            self.yields += 1
            self.backlog_yields += int(not core.input_queue.empty())
            return True
        return False

    def process_input_queue(self, core):
        self._owner(core)
        self.input_turns += 1
        # Same native idle wait, idle callbacks and duplicate-abort draining.
        # FIFO queue ownership remains with the native engine core.
        while not core.has_work() and core.is_running():
            core._notify_idle_state_callbacks()
            if core.input_queue.empty():
                with core.aborts_queue.mutex:
                    core.aborts_queue.queue.clear()
            block = core.process_input_queue_block
            try:
                request = core.input_queue.get(block=block)
                if self._handle(core, request):
                    return
            except queue.Empty:
                break
            if not block:
                break
        while not core.input_queue.empty():
            request = core.input_queue.get_nowait()
            if self._handle(core, request):
                return
        # The original busy loop alone advances the model and publishes counts.
        # There is no reentrant step, and no split inside an admission transaction.

    def snapshot(self):
        return dict(kind='ieee_native_control_interleave_v1', policy=POLICY,
            input_turns=self.input_turns, handled_controls=self.controls,
            yields_to_native_loop=self.yields, yields_with_queued_input=self.backlog_yields)


def install_core_control_interleave(core_class, scheduler, *, executor_type,
                                   request_types, native_version):
    config = scheduler.vllm_config
    policy = config.additional_config.get('ieee_tc_control_interleave')
    if policy is None:
        return None
    if policy != POLICY or native_version != '0.30.0':
        raise ValueError('control interleave requires atomic_control_yield_v1 on vLLM 0.30.0')
    parallel = scheduler.parallel_config
    if (parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1 or config.model_config.is_moe):
        raise ValueError('control interleave requires dense TP/PP/DP=1')
    existing = getattr(core_class, '_ieee_control_interleave', None)
    if existing is not None:
        if existing.scheduler is not scheduler:
            raise ValueError('control interleave already has another scheduler owner')
        return existing
    if (getattr(core_class, '_ieee_core_host_timing', None) is not None
            or native_input_fingerprint(core_class._process_input_queue) != _NATIVE_INPUT_AST_SHA):
        raise ValueError('control interleave native input source changed or was already wrapped')
    state = AtomicControlInterleave(core_class, scheduler, executor_type, request_types)
    def process_input_queue(core):
        return state.process_input_queue(core)
    core_class._process_input_queue = process_input_queue
    core_class._ieee_control_interleave = state
    return state
