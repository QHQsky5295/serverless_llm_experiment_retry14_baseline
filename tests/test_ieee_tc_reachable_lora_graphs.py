"""Reachability, native rounding, sentinel and public worker integration (CPU)."""
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.lora_utils import get_lora_capture_cases
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from faaslora.experiment.reachable_lora_graphs import (
    POLICY, reachable_capture_cases, restrict_runner_capture_cases, validate_capture_policy)
from faaslora.experiment.vllm_reachable_graph_worker import IEEEReachableGraphWorker, Worker
from faaslora.experiment.instance_pool import FrozenServiceProfiles
from scripts import run_all_experiments as runner_module


def model_config(**changes):
    return dict(backend='vllm', dtype='float16', tensor_parallel_size=1,
        ieee_scheduler_observation=True, ieee_gpu_references=True,
        ieee_worker_observation=True, ieee_input_upper_bounds=[759],
        timing_contract='ieee_tc_native_v1', enable_lora=True, enforce_eager='auto',
        specialize_active_lora=True, max_num_seqs=8, max_loras=32,
        ieee_lora_capture_policy=POLICY) | changes


def native_runner(seq=8, slots=32):
    obj = GPUModelRunner.__new__(GPUModelRunner)
    obj.lora_config = NS(max_loras=slots, specialize_active_lora=True)
    obj.compilation_config = NS(cudagraph_specialize_lora=True,
        cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    obj.lora_capture_cases = get_lora_capture_cases(obj.lora_config, obj.compilation_config)
    obj.max_num_reqs = seq
    obj.cudagraph_manager = None
    return obj


@pytest.mark.parametrize('seq', [1, 3, 8, 17, 64])
def test_every_reachable_native_mapping_and_dummy_stay_identical(seq):
    runner = native_runner(seq)
    before = list(runner.lora_capture_cases)
    receipt = restrict_runner_capture_cases(runner)
    maps = []
    for cases in (before, runner.lora_capture_cases):
        manager = CudaGraphManager.__new__(CudaGraphManager)
        manager.lora_capture_cases = cases
        manager._lora_dispatch_map, manager._max_lora_case = manager._build_lora_dispatch_map()
        maps.append([manager._resolve_effective_loras(n)
                     for n in [*range(min(seq, 32) + 1), 33]])
    assert maps[0] == maps[1]
    assert receipt['dummy_sentinel'] == 33
    assert runner.lora_config.max_loras == 32
    assert receipt['token_capture_sizes'] == runner.compilation_config.cudagraph_capture_sizes
    if seq == 8:
        assert runner.lora_capture_cases == [0, 1, 2, 4, 8, 33]
        assert receipt['removed_cases'] == [16, 32]
    if seq >= 32:
        assert runner.lora_capture_cases == before


@pytest.mark.parametrize('cases', [[1, 2, 33], [0, 8, 32], [0, 2, 1, 33], [0, True, 33]])
def test_invalid_case_contract_fails(cases):
    with pytest.raises(ValueError):
        reachable_capture_cases(cases, max_loras=32, max_num_seqs=8)


def test_install_is_before_capture_and_not_repeated():
    runner = native_runner()
    runner.cudagraph_manager = object()
    with pytest.raises(RuntimeError):
        restrict_runner_capture_cases(runner)
    runner.cudagraph_manager = None
    restrict_runner_capture_cases(runner)
    with pytest.raises(RuntimeError):
        restrict_runner_capture_cases(runner)


@pytest.mark.parametrize('change', [dict(ieee_lora_capture_policy='unknown'),
    dict(specialize_active_lora=False), dict(enable_lora=False), dict(max_num_seqs=True),
    dict(max_loras=0), dict(tensor_parallel_size=2), dict(enforce_eager=True)])
def test_invalid_policy_is_not_silently_defaulted(change):
    with pytest.raises(ValueError):
        validate_capture_policy(model_config(**change))


def test_profile_identity_requires_new_measurement_and_default_has_no_policy():
    cfg = model_config()
    old = {k: v for k, v in cfg.items() if k != 'ieee_lora_capture_policy'}
    assert validate_capture_policy(old) is None
    assert FrozenServiceProfiles.model_identity(cfg) != FrozenServiceProfiles.model_identity(old)


def test_native_worker_uses_super_once_then_changes_only_runner_cases():
    worker = IEEEReachableGraphWorker.__new__(IEEEReachableGraphWorker)
    worker.use_v2_model_runner = True
    worker.model_runner = native_runner()
    worker.vllm_config = NS(additional_config={'ieee_tc_lora_capture_policy': POLICY},
        speculative_config=None, scheduler_config=NS(max_num_seqs=8),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1))
    with patch.object(Worker, 'init_device') as parent:
        worker.init_device()
    parent.assert_called_once_with()
    assert worker.model_runner.lora_capture_cases == [0, 1, 2, 4, 8, 33]
    worker.use_v2_model_runner = False
    with patch.object(Worker, 'init_device') as parent, pytest.raises(RuntimeError):
        worker.init_device()
    parent.assert_not_called()


@pytest.mark.parametrize('memory', [False, True])
def test_engine_passes_public_worker_without_replacing_loader_extension(memory):
    cfg = model_config(ieee_backbone_loader='fastsafetensors_affinity_preserving_v1',
                       safetensors_load_strategy='auto')
    if memory:
        cfg['ieee_startup_memory_policy'] = 'exact_native_startup_memory_v1'
    engine = runner_module.InferenceEngine(cfg, {})
    args = Mock(side_effect=lambda **kw: NS(**kw))
    native = object()
    with patch.object(runner_module, '_lazy_import_vllm', return_value=True), \
         patch.object(runner_module, 'AsyncEngineArgs', args), \
         patch('faaslora.scheduling.vllm_ieee_frontend.IEEENativeAsyncLLM.from_engine_args', return_value=native):
        assert asyncio.run(engine._try_create_engine('local', 1, .72, 1024, False, True, 32, 16)) is native
    kw = args.call_args.kwargs
    assert kw['worker_cls'].endswith('.IEEEReachableGraphWorker')
    assert kw['worker_extension_cls'].endswith('.IEEEFastLoaderWorkerExtension')
    assert kw['additional_config']['ieee_tc_lora_capture_policy'] == POLICY
    assert kw['max_loras'] == 32 and kw['max_num_seqs'] == 8
    assert args.call_count == 1
    if memory:
        contract = kw['additional_config']['ieee_tc_startup_memory']
        assert contract['engine_arguments']['load_format'] == kw['load_format']
        assert 'ieee_tc_startup_memory' not in contract['engine_arguments']['additional_config']
