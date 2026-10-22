"""Explicit lossless startup policy; no inference or adapter policy changes."""

POLICY = 'fastsafetensors_affinity_preserving_v1'


def validate_backbone_loader(config):
    policy = config.get('ieee_backbone_loader')
    if policy is None:
        return None
    if (policy != POLICY or config.get('backend') != 'vllm'
            or config.get('dtype') != 'float16'
            or config.get('quantization') is not None
            or config.get('tensor_parallel_size') != 1
            or config.get('ieee_worker_observation') is not True
            or config.get('ieee_gpu_references') is not True):
        raise ValueError('bulk backbone loader requires the native unquantized FP16 TP1 owner')
    if (config.get('safetensors_load_strategy') not in (None, 'auto')
            or config.get('safetensors_prefetch_num_threads') is not None
            or config.get('safetensors_prefetch_block_size') is not None):
        raise ValueError('bulk backbone loader cannot silently ignore a safetensors strategy')
    return policy
