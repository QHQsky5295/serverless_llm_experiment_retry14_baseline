"""Restrict active-LoRA graph cases to the scheduler's reachable request count.

The resident slot pool is unchanged. Keep native upward rounding for every
reachable count and retain max_loras+1 for native dummy/profile execution.
This module imports no CUDA/backend code; installation is worker-local.
"""

POLICY = 'scheduler_reachable_active_lora_v1'


def validate_capture_policy(config):
    policy = config.get('ieee_lora_capture_policy')
    if policy is None:
        return None
    if (policy != POLICY or config.get('backend') != 'vllm'
            or config.get('specialize_active_lora') is not True
            or config.get('enable_lora') is not True
            or config.get('tensor_parallel_size') != 1
            or config.get('dtype') != 'float16'
            or config.get('enforce_eager') not in (False, 'auto')
            or not config.get('ieee_scheduler_observation')
            or not config.get('ieee_gpu_references')):
        raise ValueError('reachable LoRA graphs require the qualified specialized native owner')
    for key in ('max_num_seqs', 'max_loras'):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f'reachable LoRA graphs require explicit positive {key}')
    return policy


def reachable_capture_cases(native_cases, *, max_loras, max_num_seqs):
    """Return native cases needed by any legal batch, plus the dummy sentinel."""
    if (type(max_loras) is not int or max_loras <= 0
            or type(max_num_seqs) is not int or max_num_seqs <= 0
            or not isinstance(native_cases, list) or not native_cases
            or any(type(n) is not int or n < 0 for n in native_cases)
            or native_cases != sorted(set(native_cases))
            or native_cases[0] != 0 or native_cases[-1] != max_loras + 1):
        raise ValueError('invalid native active-LoRA capture cases or scheduler bound')
    bound = min(max_loras, max_num_seqs)
    rounded_bound = next(n for n in native_cases if n >= bound)
    return [n for n in native_cases if n <= rounded_bound or n == max_loras + 1]


def restrict_runner_capture_cases(runner):
    """Apply once, after runner construction and before any model/KV capture."""
    lora = runner.lora_config
    if (lora is None or lora.specialize_active_lora is not True
            or runner.compilation_config.cudagraph_specialize_lora is not True
            or runner.cudagraph_manager is not None
            or hasattr(runner, '_ieee_lora_capture_receipt')):
        raise RuntimeError('reachable LoRA capture policy requires an uncaptured specialized runner')
    before = list(runner.lora_capture_cases)
    after = reachable_capture_cases(before, max_loras=lora.max_loras,
                                    max_num_seqs=runner.max_num_reqs)
    receipt = dict(policy=POLICY, native_cases=before, capture_cases=after,
        removed_cases=[n for n in before if n not in after],
        max_loras=lora.max_loras, max_num_seqs=runner.max_num_reqs,
        dummy_sentinel=lora.max_loras + 1,
        token_capture_sizes=list(runner.compilation_config.cudagraph_capture_sizes))
    runner.lora_capture_cases = after
    runner._ieee_lora_capture_receipt = receipt
    return receipt
