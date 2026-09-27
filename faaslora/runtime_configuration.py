"""Pure resolution of existing serving-facade defaults; no backend imports.

These are Prime's historical facade defaults, not vLLM's upstream defaults.
Keeping this operation shared lets planned and initialized configurations agree
without removing capacity fields from profile identity.
"""
import copy


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
    if actual != expected:
        differing = sorted(key for key in set(actual) | set(expected)
                           if key not in actual or key not in expected or actual[key] != expected[key])
        raise ValueError('initialized worker model configuration differs: ' + ','.join(differing))
    return copy.deepcopy(actual)
