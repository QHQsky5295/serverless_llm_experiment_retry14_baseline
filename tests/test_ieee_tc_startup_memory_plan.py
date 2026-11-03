"""Exact allocation caching must not change native compile/capture semantics."""
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
from vllm.v1.worker.startup_plan import _applicable_kv_cache_memory_bytes

from faaslora.experiment.startup_memory_plan import (
    POLICY, ExactStartupMemoryPlan, digest_json, engine_memory_contract, runtime_memory_identity)
from faaslora.experiment.vllm_reachable_graph_worker import IEEEReachableGraphWorker, Worker


def kwargs():
    return dict(model='/sealed/model', tensor_parallel_size=1, dtype='float16',
        enforce_eager=False, enable_lora=True, gpu_memory_utilization=.72,
        max_loras=32, max_num_seqs=8, load_format='fastsafetensors_affinity_preserving_v1',
        worker_cls='faaslora.experiment.vllm_reachable_graph_worker.IEEEReachableGraphWorker',
        additional_config={'profile': 'one'})


def worker():
    return NS(cache_config=NS(kv_cache_memory_bytes=None))


def plan(tmp_path, identity=None, free=10000):
    return ExactStartupMemoryPlan(tmp_path, identity or dict(model='sealed', driver='one'),
                                  free, 8000)


def test_opt_in_and_no_mutable_alias():
    kw = kwargs()
    assert engine_memory_contract({}, kw) is None
    contract = engine_memory_contract(dict(ieee_startup_memory_policy=POLICY), kw)
    kw['additional_config']['ieee_tc_startup_memory'] = contract
    assert 'ieee_tc_startup_memory' not in contract['engine_arguments']['additional_config']
    for change in (dict(gpu_memory_utilization=.71), dict(load_format='auto'),
                   dict(max_loras=16), dict(max_num_seqs=4)):
        other = engine_memory_contract(dict(ieee_startup_memory_policy=POLICY), kwargs() | change)
        assert digest_json(other) != digest_json(contract)


@pytest.mark.parametrize('change', [dict(tensor_parallel_size=2), dict(dtype='float32'),
    dict(enforce_eager=True), dict(enable_lora=False), dict(worker_cls='auto'),
    dict(kv_cache_memory_bytes=4000)])
def test_invalid_runtime_rejected(change):
    with pytest.raises(ValueError):
        engine_memory_contract(dict(ieee_startup_memory_policy=POLICY), kwargs() | change)


def test_only_successfully_published_actual_capacity_is_reused(tmp_path):
    cold = plan(tmp_path)
    w = worker()
    cold.apply(w, _applicable_kv_cache_memory_bytes)
    assert cold.receipt['status'] == 'miss' and w.cache_config.kv_cache_memory_bytes is None
    assert not cold.path.exists()
    cold.publish(4123)  # actual native result, not a suggested capacity
    warm = plan(tmp_path)
    warm.apply(w, _applicable_kv_cache_memory_bytes)
    assert warm.receipt['status'] == 'hit' and w.cache_config.kv_cache_memory_bytes == 4123
    warm.publish(4123)
    with pytest.raises(RuntimeError):
        warm.publish(4124)


def test_native_free_memory_gate_and_first_success_is_immutable(tmp_path):
    cold = plan(tmp_path)
    cold.apply(worker(), _applicable_kv_cache_memory_bytes)
    cold.publish(4123)
    original = cold.path.read_bytes()
    lower = plan(tmp_path, free=9999)
    w = worker()
    lower.apply(w, _applicable_kv_cache_memory_bytes)
    assert lower.receipt['status'] == 'insufficient_free_memory'
    assert w.cache_config.kv_cache_memory_bytes is None
    lower.publish(4000)
    assert lower.path.read_bytes() == original
    assert lower.receipt['publication'] == 'existing_preserved'
    changed = plan(tmp_path, identity={'model': 'sealed', 'driver': 'two'})
    changed.apply(worker(), _applicable_kv_cache_memory_bytes)
    assert changed.receipt['status'] == 'miss'


@pytest.mark.parametrize('field,value', [('kv_cache_memory_bytes', True),
    ('kv_cache_memory_bytes', 8001), ('free_memory_baseline', -1),
    ('identity', {}), ('schema', 'unknown'), ('producer', None)])
def test_invalid_record_never_sets_capacity(tmp_path, field, value):
    cold = plan(tmp_path)
    cold.publish(4123)
    payload = json.loads(cold.path.read_text())
    payload[field] = value
    cold.path.write_text(json.dumps(payload))
    w = worker()
    cold.apply(w, _applicable_kv_cache_memory_bytes)
    assert cold.receipt['status'] == 'invalid' and w.cache_config.kv_cache_memory_bytes is None


def test_native_calls_once_and_no_publication_on_failed_capture(tmp_path):
    w = IEEEReachableGraphWorker.__new__(IEEEReachableGraphWorker)
    w.vllm_config = NS(additional_config={})
    with patch.object(Worker, 'determine_available_memory', return_value=4123) as determine:
        assert w.determine_available_memory() == 4123
    determine.assert_called_once_with()
    w._ieee_memory_plan = plan(tmp_path)
    with patch.object(Worker, 'compile_or_warm_up_model', side_effect=RuntimeError('capture failed')):
        with pytest.raises(RuntimeError):
            w.compile_or_warm_up_model()
    assert not w._ieee_memory_plan.path.exists()
    result = object()
    with patch.object(Worker, 'compile_or_warm_up_model', return_value=result) as capture:
        assert w.compile_or_warm_up_model() is result
    capture.assert_called_once_with()
    assert json.loads(w._ieee_memory_plan.path.read_text())['kv_cache_memory_bytes'] == 4123


def test_native_hit_still_profiles_compile_without_estimating_graph_memory(tmp_path):
    identity = {'policy': POLICY, 'driver': 'one'}
    cold = plan(tmp_path, identity=identity)
    cold.publish(4123)
    w = IEEEReachableGraphWorker.__new__(IEEEReachableGraphWorker)
    w.vllm_config = NS(additional_config={'ieee_tc_startup_memory': {'policy': POLICY}})
    w.cache_config = NS(kv_cache_memory_bytes=None)
    w.init_snapshot = NS(free_memory=10000)
    w.requested_memory = 8000
    w.model_config = NS(multimodal_config=None)
    w.parallel_config = NS(_api_process_count=1)
    w.model_runner = NS(profile_run=Mock(), profile_cudagraph_memory=Mock())
    with patch('vllm.envs.VLLM_ENABLE_STARTUP_PLAN', False), \
         patch('vllm.envs.VLLM_CACHE_ROOT', str(tmp_path)), \
         patch('faaslora.experiment.startup_memory_plan.runtime_memory_identity', return_value=identity):
        assert w.determine_available_memory() == 4123
    w.model_runner.profile_run.assert_called_once_with()
    w.model_runner.profile_cudagraph_memory.assert_not_called()
    assert w._ieee_memory_plan.receipt['status'] == 'hit'


def test_runtime_identity_binds_driver_files_and_installed_code_without_cuda(tmp_path):
    (tmp_path / 'model.safetensors').write_bytes(b'fixture only')
    (tmp_path / 'config.json').write_text('{}')
    w = NS(model_config=NS(model=str(tmp_path), is_multimodal_model=False),
           vllm_config=NS(), rank=0, parallel_config=NS(world_size=1))
    nvml = NS(nvmlInit=Mock(), nvmlShutdown=Mock(), nvmlSystemGetDriverVersion=lambda: b'test-driver')
    with patch('vllm.utils.import_utils.import_pynvml', return_value=nvml), \
         patch('vllm.v1.worker.startup_plan.compute_plan_fingerprint', return_value='native-test'), \
         patch('vllm.compilation.compiler_interface.get_inductor_factors', return_value=[{'device': 'test'}, b'\x00\xff']):
        identity = runtime_memory_identity(w, {'policy': POLICY})
    assert identity['driver'] == 'test-driver'
    assert identity['model_files']['model.safetensors']['size'] == 12
    assert len(identity['sources']) == 5
    assert len(identity['compiler_component_hashes']) == 2
    # The complete receipt is JSON, while native compiler inputs need not be.
    assert len(digest_json(identity)) == 64
    nvml.nvmlShutdown.assert_called_once_with()


def test_actual_native_compiler_components_use_native_string_encoding():
    from vllm.utils.hashing import safe_hash
    from vllm.compilation.compiler_interface import get_inductor_factors, InductorStandaloneAdaptor
    factors = get_inductor_factors()
    assert isinstance(factors[1], bytes)
    digest = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
    native = InductorStandaloneAdaptor.__new__(InductorStandaloneAdaptor)
    assert native.compute_hash(None) == digest[:10]
