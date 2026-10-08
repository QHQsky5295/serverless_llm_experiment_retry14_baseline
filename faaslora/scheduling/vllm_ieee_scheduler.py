"""Opt-in, version-scoped native scheduler observation (not a scheduler rewrite).

Imported only when selected via vLLM's scheduler_cls. Policy, preemption, block
allocation and async execution remain the official AsyncScheduler's methods.
The EngineCore utility bridge is a documented local integration of its private
utility protocol, not a claim that vLLM publishes a stable scheduler metrics API.
"""
import vllm
import time
import os
from bisect import bisect_left

if vllm.__version__ != '0.30.0':
    raise RuntimeError('IEEE native scheduler adapter requires qualified vLLM 0.30.0')

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import FullAttentionSpec

from .resource_coordinator import (NativeIterationObservation, NativeRequestRetirement,
                                   NativePendingAdmissions, NativeTransferObservation, CompletedLengthWindow,
                                   capture_native_kv_observation)


def _core_pending_admission(core, operation, arguments):
    if not isinstance(core.scheduler, IEEENativeAsyncScheduler):
        raise RuntimeError('pending KV utility requires the native scheduler owner')
    if type(operation) is not str or operation not in ('register', 'bind', 'withdraw'):
        raise ValueError('unsupported pending KV ownership transition')
    # vLLM 0.30's utility converter counts declared signature entries; a
    # variadic method cannot carry register/bind's differing positional arity.
    # One decoded list preserves the command payload without changing ownership.
    arity = {'register': 2, 'bind': 3, 'withdraw': 1}[operation]
    if type(arguments) is not list or len(arguments) != arity:
        raise ValueError('pending KV ownership command payload mismatch')
    receipt = getattr(core.scheduler._ieee_pending_admissions, operation)(*arguments)
    interleave = getattr(core.scheduler, '_ieee_control_interleave', None)
    if operation == 'register' and interleave is not None:
        # Reuse this native owner/clock-stamped acknowledgement. The current
        # utility has not yet advanced the cumulative control counters. This
        # is trigger evidence, not a model-step or per-request latency metric.
        # Do not mutate the canonical pending entry or add a sampling RPC.
        receipt = dict(receipt, scheduler_pid=os.getpid(),
                       control_interleave=interleave.snapshot())
    return receipt


def _core_pending_gpu_reference(core, intent_id, descriptor, reference):
    """Register then protect an exact GPU copy in one serialized core turn.

    No scheduling step interleaves the two operations on the qualified UniProc
    executor. This is NOT rollback-atomic: any error/lost reply may leave pending
    demand and/or a GPU reference owned. Existing retirement is still required.
    HOST has an intervening controller update and must not use this operation.
    """
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    from faaslora.clock import local_monotonic_clock_id
    from .resource_coordinator import validate_pending_gpu_reference
    if (not isinstance(core.scheduler, IEEENativeAsyncScheduler)
            or type(core.model_executor) is not UniProcExecutor):
        raise RuntimeError('combined pending/GPU protection requires the native UniProc owner')
    validate_pending_gpu_reference(reference,
        descriptor.get('adapter_int_id') if isinstance(descriptor, dict) else None)
    pending = _core_pending_admission(core, 'register', [intent_id, descriptor])
    if (not isinstance(pending, dict) or pending.get('kind') != 'ieee_pending_admission_v1'
            or pending.get('intent_id') != intent_id or pending.get('state') != 'pending'
            or pending.get('clock_id') != local_monotonic_clock_id()
            or pending.get('physical_kv_reservation') is not False):
        raise ValueError('GPU protection requires native pending acknowledgement')
    result = core.collective_rpc('ieee_gpu_reference', kwargs={
        'operation': 'demand_load_and_acquire', **reference})
    if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], dict):
        raise RuntimeError('combined protection lacks single-worker acknowledgement')
    return dict(kind='ieee_pending_gpu_reference_v1', pending=pending, reference=result[0],
                execution_scope='single_native_core_turn_v1')


def _core_observation(core):
    if not isinstance(core.scheduler, IEEENativeAsyncScheduler):
        raise RuntimeError('IEEE observation is not enabled for this native scheduler')
    return core.scheduler.ieee_scheduler_observation()


def _core_transfer_event(core, command):
    if (not isinstance(core.scheduler, IEEENativeAsyncScheduler)
            or core.scheduler._ieee_transfers is None):
        raise RuntimeError('transfer events require the configured native admission owner')
    return core.scheduler._ieee_transfers.event(**command)


def _core_retirement(core, request_id, abort):
    if not isinstance(core.scheduler, IEEENativeAsyncScheduler) or type(abort) is not bool:
        raise RuntimeError('IEEE native retirement utility contract mismatch')
    scheduler = core.scheduler
    future = scheduler._ieee_retirement.wait(request_id)
    if abort:
        core.abort_requests([request_id])  # Original native cancellation, not a queue rewrite.
    scheduler._ieee_retirement.advance(scheduler.requests, scheduler.processed_step_seq)
    return future


def _core_prepare_host(core, command):
    """One synchronous core/worker transaction; not two controller samples.

    UniProc calls the worker on this same engine-core thread. Native async GPU
    execution is preserved, but no new scheduling/allocation is interleaved
    between the KV observation and the fenced slot commit. Other executors
    require their own reservation protocol and must not silently use this one.
    """
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    if (not isinstance(core.scheduler, IEEENativeAsyncScheduler)
            or type(core.model_executor) is not UniProcExecutor):
        raise RuntimeError('proactive HOST preparation requires the native UniProc owner')
    scheduler = core.scheduler
    if scheduler._ieee_lengths is None:
        raise RuntimeError('proactive admission has no frozen completion-length profile')
    legacy_allowed = {'lease_id', 'adapter_int_id', 'lora_name', 'lora_path',
                      'expected_owner_id', 'expected_epoch', 'capacity_only'}
    allowed = legacy_allowed | {'expected_content_sha256'}
    accepted = (legacy_allowed, legacy_allowed | {'replacement_epoch'},
                legacy_allowed | {'replacement_epoch', 'preparation_plan_id'},
                legacy_allowed | {'replacement_epoch', 'preparation_plan_id', 'host_file_fallbacks'},
                allowed, allowed | {'replacement_epoch'},
                allowed | {'replacement_epoch', 'preparation_plan_id'},
                allowed | {'replacement_epoch', 'preparation_plan_id', 'host_file_fallbacks'})
    if not isinstance(command, dict) or set(command) not in accepted:
        raise ValueError('proactive preparation command fields differ from the contract')
    observation = scheduler.ieee_scheduler_observation()
    lengths = scheduler._ieee_lengths.snapshot(now=observation['captured_at'])
    result = core.collective_rpc('ieee_gpu_reference', kwargs={
        'operation': 'proactive_host_prepare_and_acquire', **command,
        'scheduler_observation': observation, 'lengths': lengths})
    if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], dict):
        raise RuntimeError('proactive preparation lacks single-worker acknowledgement')
    return result[0]


def _core_admit_and_demand_load(core, command):
    """Observe IEEE capacity state and issue one serialized native demand load.

    The scheduler owner captures a pre-load KV/length observation and holds its
    utility call while the single UniProc worker evaluates the same proposal
    and issues the adapter load.  This serialized call is not an atomic KV
    admission transaction: vLLM remains the owner of actual KV blocks and the
    backend may advance its owner epoch while completing the load.  The
    returned receipt is an observational witness, not a KV reservation or
    proactive E(t) decision.
    """
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    if (not isinstance(core.scheduler, IEEENativeAsyncScheduler)
            or type(core.model_executor) is not UniProcExecutor):
        raise RuntimeError('native admission requires the native UniProc owner')
    scheduler = core.scheduler
    if scheduler._ieee_lengths is None:
        raise RuntimeError('native admission has no frozen completion-length profile')
    legacy_allowed = {'lease_id', 'adapter_int_id', 'lora_name', 'lora_path',
                      'expected_owner_id', 'expected_epoch', 'required_source_tier',
                      'expected_source_id', 'capacity_only'}
    allowed = legacy_allowed | {'expected_content_sha256'}
    if not isinstance(command, dict) or set(command) not in (
            legacy_allowed, allowed, allowed | {'file_source_witness'}):
        raise ValueError('native admission command fields differ from the contract')
    if command.get('capacity_only') is not True:
        raise ValueError('request capacity witness requires capacity_only=True')
    observation = scheduler.ieee_scheduler_observation()
    lengths = scheduler._ieee_lengths.snapshot(now=observation['captured_at'])
    result = core.collective_rpc('ieee_gpu_reference', kwargs={
        'operation': 'admit_and_demand_load', **command,
        'scheduler_observation': observation, 'lengths': lengths})
    if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], dict):
        raise RuntimeError('native admission lacks a single-worker acknowledgement')
    return result[0]


class IEEENativeAsyncScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.vllm_config
        parallel = self.parallel_config
        if (parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1
                or self.dcp_world_size != 1 or self.pcp_world_size != 1
                or config.speculative_config is not None or self.connector is not None
                or self.kv_cache_manager.watermark_blocks != 0):
            raise ValueError('IEEE scheduler observation requires TP/PP/CP=1, no speculative/KV transfer/watermark')
        groups = self.kv_cache_config.kv_cache_groups
        if len(groups) != 1 or type(groups[0].kv_cache_spec) is not FullAttentionSpec:
            raise ValueError('IEEE scheduler observation requires exact full-attention single-group spec')
        self._ieee_input_upper_bounds = tuple(config.additional_config[
            'ieee_tc_scheduler_observation']['input_upper_bounds'])
        self._ieee_iterations = NativeIterationObservation()
        self._ieee_retirement = NativeRequestRetirement(self._ieee_iterations)
        self._ieee_pending_admissions = NativePendingAdmissions(
            self._ieee_iterations, self._ieee_input_upper_bounds)
        profile = config.additional_config.get('ieee_tc_admission_profile')
        self._ieee_lengths = None
        self._ieee_transfers = None
        if profile is not None:
            means = profile['profile_means']
            if not isinstance(means, list) or len(means) != len(self._ieee_input_upper_bounds) + 1:
                raise ValueError('completion profile must cover every frozen input bucket')
            self._ieee_lengths = CompletedLengthWindow(
                window_s=profile['window_s'], model_backend_id=profile['model_backend_id'],
                profile_id=profile['profile_id'], profile_means=dict(enumerate(means)))
            self._ieee_transfers = NativeTransferObservation(self._ieee_iterations, profile['transfer_limit'])
        # Validate actual layout immediately, before claiming this hook is ready.
        self.ieee_scheduler_observation()
        existing = getattr(EngineCore, 'ieee_scheduler_observation', None)
        if existing is not None and existing is not _core_observation:
            raise RuntimeError('native EngineCore utility bridge name collision')
        EngineCore.ieee_scheduler_observation = _core_observation
        existing = getattr(EngineCore, 'ieee_request_retirement', None)
        if existing is not None and existing is not _core_retirement:
            raise RuntimeError('native EngineCore retirement bridge name collision')
        EngineCore.ieee_request_retirement = _core_retirement
        existing = getattr(EngineCore, 'ieee_prepare_host', None)
        if existing is not None and existing is not _core_prepare_host:
            raise RuntimeError('native EngineCore proactive preparation bridge name collision')
        EngineCore.ieee_prepare_host = _core_prepare_host
        existing = getattr(EngineCore, 'ieee_admit_and_demand_load', None)
        if existing is not None and existing is not _core_admit_and_demand_load:
            raise RuntimeError('native EngineCore admission bridge name collision')
        EngineCore.ieee_admit_and_demand_load = _core_admit_and_demand_load
        existing = getattr(EngineCore, 'ieee_pending_admission', None)
        if existing is not None and existing is not _core_pending_admission:
            raise RuntimeError('native EngineCore pending admission bridge name collision')
        EngineCore.ieee_pending_admission = _core_pending_admission
        existing = getattr(EngineCore, 'ieee_pending_gpu_reference', None)
        if existing is not None and existing is not _core_pending_gpu_reference:
            raise RuntimeError('native EngineCore pending GPU bridge name collision')
        EngineCore.ieee_pending_gpu_reference = _core_pending_gpu_reference
        existing = getattr(EngineCore, 'ieee_transfer_event', None)
        if existing is not None and existing is not _core_transfer_event:
            raise RuntimeError('native EngineCore transfer bridge name collision')
        EngineCore.ieee_transfer_event = _core_transfer_event
        self._ieee_control_interleave = None
        if config.additional_config.get('ieee_tc_control_interleave') is not None:
            from vllm.v1.engine.core import EngineCoreProc
            from vllm.v1.engine import EngineCoreRequestType
            from vllm.v1.executor.uniproc_executor import UniProcExecutor
            from .core_interleave import install_core_control_interleave
            self._ieee_control_interleave = install_core_control_interleave(
                EngineCoreProc, self, executor_type=UniProcExecutor,
                request_types=EngineCoreRequestType, native_version=vllm.__version__)
        # Diagnostic-only host spans. Default leaves every native method intact.
        if os.environ.get('FAASLORA_TC_CORE_TIMING') not in (None, '0'):
            from vllm.v1.engine.core import EngineCoreProc
            from faaslora.utils.core_timing import install_core_host_timing
            install_core_host_timing(EngineCoreProc, self)

    def add_request(self, request):
        self._ieee_pending_admissions.validate_add(request)
        result = super().add_request(request)
        self._ieee_pending_admissions.added(request.request_id)
        self._ieee_retirement.added(request.request_id)
        return result

    def _free_request(self, request, *args, **kwargs):
        if self._ieee_lengths is not None:
            from vllm.v1.request import RequestStatus
            if request.status in (RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED):
                self._ieee_lengths.record_completed(request.request_id,
                    bisect_left(self._ieee_input_upper_bounds, request.num_prompt_tokens),
                    request.num_output_tokens, completed_at=time.monotonic())
        result = super()._free_request(request, *args, **kwargs)
        # Native _free_request_blocks uses sched_step_seq for deferred frees.
        self._ieee_retirement.removed(request.request_id, self.sched_step_seq)
        return result

    def schedule(self, *args, **kwargs):
        self._ieee_iterations.check_thread()
        output = super().schedule(*args, **kwargs)
        self._ieee_iterations.scheduled(output)
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        self._ieee_iterations.check_thread()
        output = super().update_from_output(scheduler_output, model_runner_output)
        self._ieee_iterations.completed(scheduler_output)
        self._ieee_retirement.advance(self.requests, self.processed_step_seq)
        return output

    def ieee_scheduler_observation(self):
        result = capture_native_kv_observation(self, self._ieee_iterations,
                                              input_upper_bounds=self._ieee_input_upper_bounds)
        interleave = getattr(self, '_ieee_control_interleave', None)
        if interleave is not None:
            result['control_interleave'] = interleave.snapshot()
        return result
