"""Opt-in native worker: narrow graph cases before model loading/capture.

Uses vLLM's public worker_cls interface; does not patch the installed backend,
dispatcher, kernels, resident capacity or temporary-memory profiling lifecycle.
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
