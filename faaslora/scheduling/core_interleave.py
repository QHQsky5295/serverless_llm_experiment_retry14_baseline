"""Opt-in fairness between atomic PrimeLoRA control transactions and native steps.

The optional private CPU builder completes through this same core queue; GPU
admission remains atomic. No extra model step or time-budget policy is added.
"""
import ast
import hashlib
import inspect
import os
import queue
import textwrap
import threading
from concurrent.futures import Future
from dataclasses import dataclass

POLICY = 'atomic_control_yield_v1'
HOST_CONSTRUCTION_POLICY = 'private_cpu_core_commit_v1'
_NATIVE_INPUT_AST_SHA = '5d1a82d46b2a673a0f8e32db251d1251613c6f63794758cea9eae6dd5845a845'
_PRIME_METHODS = frozenset(('ieee_scheduler_observation', 'ieee_request_retirement',
    'ieee_prepare_host', 'ieee_admit_and_demand_load', 'ieee_pending_admission',
    'ieee_pending_gpu_reference', 'ieee_transfer_event', 'ieee_prepare_file_host'))


@dataclass(frozen=True)
class _HostConstructionWake:
    bridge: object
    build: object


class NativeHostConstructionBridge:
    """Native utility Future resolves only after a core-thread commit.

    The single UniProc worker is called synchronously for begin/settlement.
    Only its private constructor runs in a thread. Wake messages are local
    Python objects, never decoded client commands or arbitrary callbacks.
    """

    def __init__(self, interleave):
        self.interleave = interleave
        self.active = None
        self.closed = False
        self.started = self.settled = self.discarded = self.failed = 0

    @staticmethod
    def _one_worker(core, method, *args):
        values = core.collective_rpc(method, args=args)
        if not isinstance(values, list) or len(values) != 1:
            raise RuntimeError('private HOST construction requires one local worker result')
        return values[0]

    def prepare(self, core, command):
        from faaslora.memory.residency_manager import NativeHostConstruction
        self.interleave._owner(core)
        if self.closed or not core.is_running():
            raise RuntimeError('private HOST construction owner is shutting down')
        if not isinstance(command, dict):
            raise ValueError('private HOST construction requires an exact command')
        if self.active is not None and command.get('lease_id') == self.active[0].command['lease_id']:
            if command != dict(self.active[0].command):
                raise ValueError('in-flight native HOST command identity changed')
            return self.active[1]
        result = self._one_worker(core, 'ieee_begin_file_host_construction', command)
        if isinstance(result, dict):
            return result  # Existing hit or definite non-mutating deferral.
        if not isinstance(result, NativeHostConstruction) or self.active is not None:
            raise RuntimeError('native HOST worker returned an unowned construction')
        future = Future()
        future.set_running_or_notify_cancel()  # A waiter cannot release live work.
        self.active = (result, future)
        self.started += 1

        def wake(build):
            core.input_queue.put_nowait((self.interleave.request_types.WAKEUP,
                                        _HostConstructionWake(self, build)))

        result.start(wake)
        return future

    def handle_wakeup(self, core, request):
        kind, payload = request
        if kind != self.interleave.request_types.WAKEUP or not isinstance(payload, _HostConstructionWake):
            return False
        self.interleave._owner(core)
        if payload.bridge is not self:
            raise RuntimeError('native HOST wakeup belongs to another core')
        if self.active is None or self.active[0] is not payload.build:
            # Shutdown may already have joined/settled the queued completion.
            if payload.build.settled:
                return True
            raise RuntimeError('native HOST wakeup has no live construction')
        self._settle(core, cancelled=not core.is_running())
        return True

    def _settle(self, core, *, cancelled):
        build, future = self.active
        try:
            receipt = self._one_worker(core, 'ieee_finish_file_host_construction', build, cancelled)
            if not isinstance(receipt, dict) or not build.settled:
                raise RuntimeError('native HOST settlement lacks a terminal receipt')
        except Exception as error:
            self.failed += 1
            self.closed = True
            # Do not retain a traceback with private tensors in the Future.
            future.set_exception(RuntimeError(f'native HOST settlement failed: {error}'))
        else:
            self.settled += 1
            self.discarded += int(receipt.get('private_construction', {}).get('discarded', False))
            future.set_result(receipt)
        finally:
            self.active = None

    def shutdown(self, core):
        self.interleave._owner(core)
        self.closed = True
        if self.active is not None:
            # Native shutdown is already outside measured request execution.
            # Its elapsed GPU lifetime still counts. The existing whole-service
            # watchdog owns a wedged process; we never fake a joined thread.
            self.active[0].join_for_shutdown()
            self._settle(core, cancelled=True)

    def snapshot(self):
        return dict(policy=HOST_CONSTRUCTION_POLICY, started=self.started,
                    settled=self.settled, discarded=self.discarded, failed=self.failed,
                    in_flight=int(self.active is not None), closed=self.closed)


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
        self.host_construction = None

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
        completion = (self.host_construction is not None
                      and self.host_construction.handle_wakeup(core, request))
        prime = completion or self._prime_control(request)
        if not completion:
            core._handle_client_request(*request)  # Original client dispatch, exactly once.
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
            yields_to_native_loop=self.yields, yields_with_queued_input=self.backlog_yields,
            **({'private_host_construction': self.host_construction.snapshot()}
               if self.host_construction is not None else {}))


def install_core_control_interleave(core_class, scheduler, *, executor_type,
                                   request_types, native_version):
    config = scheduler.vllm_config
    policy = config.additional_config.get('ieee_tc_control_interleave')
    host_policy = config.additional_config.get('ieee_tc_host_construction')
    if policy is None:
        if host_policy is not None:
            raise ValueError('private HOST construction requires native control interleave')
        return None
    if host_policy not in (None, HOST_CONSTRUCTION_POLICY):
        raise ValueError('unknown native HOST construction policy')
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
        if (existing.host_construction is not None) != (host_policy is not None):
            raise ValueError('native HOST construction policy changed after installation')
        return existing
    if (getattr(core_class, '_ieee_core_host_timing', None) is not None
            or native_input_fingerprint(core_class._process_input_queue) != _NATIVE_INPUT_AST_SHA):
        raise ValueError('control interleave native input source changed or was already wrapped')
    state = AtomicControlInterleave(core_class, scheduler, executor_type, request_types)
    if host_policy is not None:
        if hasattr(core_class, 'ieee_prepare_file_host'):
            raise ValueError('native HOST construction utility is already installed')
        original_shutdown = core_class.shutdown
        bridge = state.host_construction = NativeHostConstructionBridge(state)
        def prepare_file_host(core, command):
            return bridge.prepare(core, command)
        def shutdown(core):
            try:
                bridge.shutdown(core)
            finally:
                original_shutdown(core)
        core_class.ieee_prepare_file_host = prepare_file_host
        core_class.shutdown = shutdown
    def process_input_queue(core):
        return state.process_input_queue(core)
    core_class._process_input_queue = process_input_queue
    core_class._ieee_control_interleave = state
    return state
