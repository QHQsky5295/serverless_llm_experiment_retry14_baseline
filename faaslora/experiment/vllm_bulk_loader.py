"""vLLM public loader extension using fastsafetensors' bounded pipeline.

Retains vLLM 0.30's file/index selection, parameter assignment, dtype conversion
and post-load checks. The sole iterator change is the upstream ParallelLoader
with public set_numa=False: the application owns the CPU resource envelope.
nogds=True deliberately uses the portable pread/bounce-buffer path, avoiding
cuFile's device-wide context initialization. No retries or alternate weights.
"""
import copy
import os
import time

import torch
from fastsafetensors import SingleGroup
from fastsafetensors.parallel_loader import ParallelLoader
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import _natural_sort_key, enable_tqdm
from vllm.platforms import current_platform

from .backbone_loader import POLICY
from ..memory.gpu_monitor import IEEEWorkerObservationExtension

logger = init_logger(__name__)


@register_model_loader(POLICY)
class AffinityPreservingFastLoader(DefaultModelLoader):
    def __init__(self, load_config):
        native = copy.copy(load_config)
        # Reuse native explicit safetensors selection: no .pt fallback or new
        # checkpoint format, and never mutate the recorded execution config.
        native.load_format = 'safetensors'
        super().__init__(native)

    def _get_weights_iterator(self, source):
        _, files, is_safetensors = self._prepare_weights(
            source.model_or_path, source.subfolder, source.revision,
            source.fall_back_to_pt, source.allow_patterns_overrides)
        if not is_safetensors or self.local_expert_ids is not None:
            raise ValueError('lossless bulk loader supports dense safetensors only')
        pg = torch.distributed.group.WORLD if torch.distributed.is_initialized() else SingleGroup()
        if pg.size() != 1:
            raise ValueError('lossless bulk loader is qualified only for TP1')
        device = f'cuda:{current_platform.current_device()}'
        affinity = os.sched_getaffinity(0)
        loader = None
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        try:
            loader = ParallelLoader(
                pg=pg, hf_weights_files=sorted(files, key=_natural_sort_key),
                device=device, nogds=True, set_numa=False,
                queue_size=envs.VLLM_FASTSAFETENSORS_QUEUE_SIZE,
                use_tqdm_on_load=enable_tqdm(self.load_config.use_tqdm_on_load))
            logger.info('Backbone loader %s: nogds=True set_numa=False affinity=%s',
                        POLICY, sorted(affinity))
            for name, tensor in loader.iterate_weights():
                yield source.prefix + name, tensor
        finally:
            if loader is not None:
                loader.close()
            if os.sched_getaffinity(0) != affinity:
                raise RuntimeError('backbone loader changed the owned CPU affinity')


class IEEEFastLoaderWorkerExtension(IEEEWorkerObservationExtension):
    """Resolving this extension registers the loader in each spawned worker."""
