"""Opt-in native worker: narrow graph cases before model loading/capture.

Uses vLLM's public worker_cls interface; does not patch the installed backend,
dispatcher or kernels. Optional exact memory reuse preserves native compilation
and service capture while avoiding a repeated temporary-memory estimate.
"""
import vllm
from vllm.logger import init_logger
from vllm.v1.worker.gpu_worker import Worker

from .reachable_lora_graphs import POLICY, restrict_runner_capture_cases

logger = init_logger(__name__)


class IEEEReachableGraphWorker(Worker):
    def init_device(self):
        config = self.vllm_config
        parallel = config.parallel_config
        if (vllm.__version__ != '0.30.0' or not self.use_v2_model_runner
                or config.additional_config.get('ieee_tc_lora_capture_policy') != POLICY
                or config.speculative_config is not None
                or parallel.tensor_parallel_size != 1
                or parallel.pipeline_parallel_size != 1
                or parallel.data_parallel_size != 1):
            raise RuntimeError('reachable LoRA graphs require the qualified vLLM V2 single-worker path')
        super().init_device()
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.v1.worker.gpu.lora_utils import get_lora_capture_cases
        runner = self.model_runner
        if (type(runner) is not GPUModelRunner
                or runner.max_num_reqs != config.scheduler_config.max_num_seqs
                or runner.lora_capture_cases != get_lora_capture_cases(
                    runner.lora_config, runner.compilation_config)):
            raise RuntimeError('native LoRA capture initialization differs from the qualified path')
        receipt = restrict_runner_capture_cases(runner)
        logger.info('PrimeLoRA reachable active-LoRA capture: %s', receipt)

    def determine_available_memory(self):
        contract = self.vllm_config.additional_config.get('ieee_tc_startup_memory')
        self._ieee_memory_plan = None
        if contract is not None:
            import vllm.envs as envs
            from vllm.v1.worker.startup_plan import _applicable_kv_cache_memory_bytes
            from .startup_memory_plan import (
                POLICY, ExactStartupMemoryPlan, runtime_memory_identity)
            if (contract.get('policy') != POLICY or envs.VLLM_ENABLE_STARTUP_PLAN
                    or self.cache_config.kv_cache_memory_bytes is not None):
                raise ValueError('exact memory reuse cannot combine with another startup/KV policy')
            self._ieee_memory_plan = ExactStartupMemoryPlan(
                envs.VLLM_CACHE_ROOT, runtime_memory_identity(self, contract),
                self.init_snapshot.free_memory, self.requested_memory)
            self._ieee_memory_plan.apply(self, _applicable_kv_cache_memory_bytes)
            logger.info('PrimeLoRA startup memory lookup: %s', self._ieee_memory_plan.receipt)
        # Even on hit the native path runs profile_run (compilation), then the
        # engine initializes KV and captures actual service graphs normally.
        result = super().determine_available_memory()
        self._ieee_actual_kv_bytes = result
        return result

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        plan = getattr(self, '_ieee_memory_plan', None)
        if plan is not None:
            plan.publish(self._ieee_actual_kv_bytes)
            logger.info('PrimeLoRA startup memory completed: %s', plan.receipt)
        return result
