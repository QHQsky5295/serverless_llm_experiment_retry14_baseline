"""Lossless startup-loader integration, CPU fixtures (no CUDA context)."""
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
from vllm.config.load import LoadConfig
from faaslora.experiment.backbone_loader import POLICY, validate_backbone_loader
from faaslora.experiment.instance_pool import FrozenServiceProfiles
from faaslora.experiment import vllm_bulk_loader as bulk
from scripts import run_all_experiments as runner


def config(**changes):
    return dict(backend='vllm', dtype='float16', tensor_parallel_size=1,
                ieee_worker_observation=True, ieee_gpu_references=True,
                ieee_scheduler_observation=True, ieee_input_upper_bounds=[759],
                timing_contract='ieee_tc_native_v1', ieee_backbone_loader=POLICY,
                safetensors_load_strategy='auto') | changes


def test_startup_identity_reuse_only_for_explicit_lossless_policy():
    cfg = config()
    baseline = {k: v for k, v in cfg.items() if k != 'ieee_backbone_loader'}
    assert FrozenServiceProfiles.model_identity(cfg) == FrozenServiceProfiles.model_identity(baseline)
    assert runner._resolve_vllm_safetensors_load(cfg)['resolved'] is None
    assert runner._resolve_vllm_safetensors_load(cfg)['decision'] == POLICY
    changed = cfg | {'max_num_seqs': 16}
    assert FrozenServiceProfiles.model_identity(changed) != FrozenServiceProfiles.model_identity(cfg)


@pytest.mark.parametrize('change', [dict(ieee_backbone_loader='dummy'), dict(dtype='bfloat16'),
    dict(tensor_parallel_size=2), dict(quantization='awq'), dict(backend='mock'),
    dict(ieee_gpu_references=False), dict(safetensors_load_strategy='prefetch'),
    dict(safetensors_prefetch_num_threads=8)])
def test_incompatible_policy_rejected(change):
    with pytest.raises(ValueError):
        validate_backbone_loader(config(**change))


def test_default_and_native_constructor_contract():
    assert validate_backbone_loader({}) is None
    native = object()
    engine = runner.InferenceEngine(config(), {})
    engine._resolve_vllm_visible_devices = Mock(return_value='0')
    engine._resolve_vllm_executor_backend = Mock(return_value=None)
    args = Mock(side_effect=lambda **kw: NS(**kw))
    with patch.object(runner, '_lazy_import_vllm', return_value=True), \
         patch.object(runner, 'AsyncEngineArgs', args), \
         patch('faaslora.scheduling.vllm_ieee_frontend.IEEENativeAsyncLLM.from_engine_args',
               return_value=native):
        actual = asyncio.run(engine._try_create_engine('local', 1, .72, 1024, False, True, 8, 16))
    assert actual is native
    assert args.call_args.kwargs['load_format'] == POLICY
    assert args.call_args.kwargs['worker_extension_cls'].endswith('.IEEEFastLoaderWorkerExtension')
    assert 'safetensors_load_strategy' not in args.call_args.kwargs
    assert args.call_count == 1


def make_loader():
    original = LoadConfig(load_format=POLICY)
    loader = bulk.AffinityPreservingFastLoader(original)
    assert original.load_format == POLICY
    assert loader.load_config.load_format == 'safetensors'
    loader._prepare_weights = Mock(return_value=('dir', ['model-10.safetensors', 'model-2.safetensors'], True))
    source = NS(model_or_path='dir', subfolder=None, revision=None,
                fall_back_to_pt=True, allow_patterns_overrides=None, prefix='prefix.')
    return loader, source


def test_native_iteration_preserves_selection_prefix_affinity_and_closes():
    loader, source = make_loader()
    tensor = object()
    pipeline = NS(iterate_weights=lambda: iter([('a', tensor)]), close=Mock())
    with patch.object(bulk, 'ParallelLoader', return_value=pipeline) as factory, \
         patch.object(bulk.torch.distributed, 'is_initialized', return_value=False), \
         patch.object(bulk.current_platform, 'current_device', return_value=0):
        assert list(loader._get_weights_iterator(source)) == [('prefix.a', tensor)]
    kwargs = factory.call_args.kwargs
    assert kwargs['set_numa'] is False and kwargs['nogds'] is True
    assert kwargs['hf_weights_files'] == ['model-2.safetensors', 'model-10.safetensors']
    assert loader.counter_before_loading_weights > 0
    pipeline.close.assert_called_once()


def test_loading_failure_is_not_retried_or_silently_replaced():
    loader, source = make_loader()
    pipeline = NS(iterate_weights=Mock(side_effect=RuntimeError('read failed')), close=Mock())
    with patch.object(bulk, 'ParallelLoader', return_value=pipeline) as factory, \
         patch.object(bulk.torch.distributed, 'is_initialized', return_value=False), \
         patch.object(bulk.current_platform, 'current_device', return_value=0):
        with pytest.raises(RuntimeError, match='read failed'):
            list(loader._get_weights_iterator(source))
        factory.assert_called_once()
    pipeline.close.assert_called_once()


def test_consumer_cancellation_closes_pipeline():
    loader, source = make_loader()
    pipeline = NS(iterate_weights=lambda: iter([('a', 1), ('b', 2)]), close=Mock())
    with patch.object(bulk, 'ParallelLoader', return_value=pipeline), \
         patch.object(bulk.torch.distributed, 'is_initialized', return_value=False), \
         patch.object(bulk.current_platform, 'current_device', return_value=0):
        stream = loader._get_weights_iterator(source)
        assert next(stream) == ('prefix.a', 1)
        stream.close()
    pipeline.close.assert_called_once()
