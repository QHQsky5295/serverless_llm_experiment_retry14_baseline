"""Opt-in, version-scoped native scheduler observation (not a scheduler rewrite).

Imported only when selected via vLLM's scheduler_cls. Policy, preemption, block
allocation and async execution remain the official AsyncScheduler's methods.
The EngineCore utility bridge is a documented local integration of its private
utility protocol, not a claim that vLLM publishes a stable scheduler metrics API.
"""
import vllm

if vllm.__version__ != '0.30.0':
    raise RuntimeError('IEEE native scheduler adapter requires qualified vLLM 0.30.0')

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import FullAttentionSpec

from .resource_coordinator import (NativeIterationObservation, NativeRequestRetirement,
                                   capture_native_kv_observation)


def _core_observation(core):
    if not isinstance(core.scheduler, IEEENativeAsyncScheduler):
        raise RuntimeError('IEEE observation is not enabled for this native scheduler')
    return core.scheduler.ieee_scheduler_observation()


def _core_retirement(core, request_id, abort):
    if not isinstance(core.scheduler, IEEENativeAsyncScheduler) or type(abort) is not bool:
        raise RuntimeError('IEEE native retirement utility contract mismatch')
    scheduler = core.scheduler
    future = scheduler._ieee_retirement.wait(request_id)
    if abort:
        core.abort_requests([request_id])  # Original native cancellation, not a queue rewrite.
    scheduler._ieee_retirement.advance(scheduler.requests, scheduler.processed_step_seq)
    return future


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

    def add_request(self, request):
        result = super().add_request(request)
        self._ieee_retirement.added(request.request_id)
        return result

    def _free_request(self, request, *args, **kwargs):
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
        return capture_native_kv_observation(self, self._ieee_iterations,
                                            input_upper_bounds=self._ieee_input_upper_bounds)
