"""Pure resolution of existing serving-facade defaults; no backend imports.

These are Prime's historical facade defaults, not vLLM's upstream defaults.
Keeping this operation shared lets planned and initialized configurations agree
without removing capacity fields from profile identity.
"""
import copy


def validate_quantization_prebuild_policy(model_config, environment):
    """Bind the upstream on-demand Humming build switch to an FP16 experiment.

    This changes only optional import-time background preparation. The native
    kernels still build on demand, and multiprocessing remains spawn.
    """
    policy = model_config.get('ieee_quantization_prebuild_policy')
    marker = environment.get('FAASLORA_IEEE_QUANTIZATION_PREBUILD_POLICY')
    if policy is None and not marker:
        if environment.get('HUMMING_DISABLE_PARALLEL_BUILD') == '1':
            raise ValueError('on-demand quantization prebuild must have frozen configuration')
        return False
    if (policy != 'on_demand_v1' or marker != policy
            or model_config.get('backend') != 'vllm'
            or model_config.get('dtype') != 'float16'
            or model_config.get('quantization') is not None
            or environment.get('HUMMING_DISABLE_PARALLEL_BUILD') != '1'
            or environment.get('VLLM_WORKER_MULTIPROC_METHOD', 'spawn') != 'spawn'):
        raise ValueError('quantization prebuild candidate requires pre-import FP16/spawn contract')
    return True


# These values are derived inside the fresh worker from the local checkpoint
# and current host memory.  They are receipts for startup observability, not
# requested serving configuration, so they must not invalidate the parent/
# worker configuration handshake when the available-memory reading changes by
# a small amount during launch.
_EPHEMERAL_STARTUP_FIELDS = frozenset({
    'resolved_safetensors_load_strategy', 'safetensors_load_decision',
    'safetensors_model_bytes', 'safetensors_mem_available_bytes',
})


def resolve_facade_lora_capacity(model_config):
    config = copy.deepcopy(model_config)
    if (str(config.get('backend', 'vllm')).lower() == 'vllm'
            and config.get('max_cpu_loras') is None):
        config['max_cpu_loras'] = max(int(config.get('max_loras', 4)), 24)
    return config


def initialized_worker_configuration(ready, requested):
    """Validate the worker's post-initialize receipt, never trust parent intent."""
    if (ready.get('configuration_contract') != 'initialized_model_config_v1'
            or not isinstance(ready.get('model_config'), dict)):
        raise ValueError('dedicated worker lacks initialized model configuration receipt')
    actual = ready['model_config']
    expected = resolve_facade_lora_capacity(requested)
    comparable_actual = {
        key: value for key, value in actual.items()
        if key not in _EPHEMERAL_STARTUP_FIELDS
    }
    comparable_expected = {
        key: value for key, value in expected.items()
        if key not in _EPHEMERAL_STARTUP_FIELDS
    }
    if comparable_actual != comparable_expected:
        differing = sorted(key for key in set(comparable_actual) | set(comparable_expected)
                           if key not in comparable_actual or key not in comparable_expected
                           or comparable_actual[key] != comparable_expected[key])
        raise ValueError('initialized worker model configuration differs: ' + ','.join(differing))
    return copy.deepcopy(actual)
