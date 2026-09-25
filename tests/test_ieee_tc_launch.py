"""TC launcher integration, with no model/driver operation in these unit tests."""
import asyncio
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from scripts import run_all_experiments as runner


class ManagedEngineLaunch(unittest.TestCase):
    def test_managed_cleanup_only_revalidates_owned_service(self):
        engine = runner.InferenceEngine({}, {})
        with patch.dict(os.environ, {'FAASLORA_TC_LAUNCH_RECEIPT':'/private/receipt'}), \
             patch('scripts.ieee_tc_preflight.verify_current_service') as verify, \
             patch.object(runner, '_kill_stale_gpu_processes') as global_kill:
            engine._maybe_kill_stale_gpu_processes()
            verify.assert_called_once_with()
            global_kill.assert_not_called()

    def test_global_cleanup_is_forbidden_inside_managed_launch(self):
        with patch.dict(os.environ, {'FAASLORA_TC_LAUNCH_RECEIPT':'/private/receipt'}):
            with self.assertRaisesRegex(RuntimeError, 'owned service scope'):
                runner._kill_stale_gpu_processes()

    def test_native_mode_cannot_use_historical_unbounded_cleanup(self):
        engine = runner.InferenceEngine({'timing_contract':'ieee_tc_native_v1'}, {})
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(runner, '_kill_stale_gpu_processes') as global_kill:
            with self.assertRaisesRegex(RuntimeError, 'guarded qualification launcher'):
                engine._maybe_kill_stale_gpu_processes()
            global_kill.assert_not_called()

    def test_native_initialize_tries_exactly_one_configuration(self):
        engine = runner.InferenceEngine({'timing_contract':'ieee_tc_native_v1',
                                         'name':'existing-local-model'}, {})
        engine._resolve_vllm_visible_devices = Mock(return_value='0')
        engine._resolve_vllm_executor_backend = Mock(return_value=None)
        engine._resolve_vllm_runtime_settings = Mock(return_value={
            'env_updates':{}, 'enable_chunked_prefill':True,
            'enable_prefix_caching':True, 'tokenizer_mode':'auto'})
        engine._maybe_kill_stale_gpu_processes = Mock()
        engine._try_create_engine = AsyncMock(return_value=None)
        with patch('scripts.ieee_tc_preflight.verify_current_service', return_value={}), \
             patch.object(runner, 'CUDA_AVAILABLE', True), \
             patch.object(runner, '_lazy_import_vllm', return_value=True), \
             patch.object(runner, '_check_shm_for_vllm'):
            with self.assertRaisesRegex(RuntimeError, 'engine creation failed'):
                asyncio.run(engine.initialize())
        self.assertEqual(engine._try_create_engine.await_count, 1)
        self.assertTrue(engine._try_create_engine.call_args.kwargs['enable_chunked_prefill'])
        self.assertTrue(engine._try_create_engine.call_args.kwargs['enable_prefix_caching'])

    def test_construction_failure_does_not_clean_unrelated_workers_or_hide_error(self):
        engine = runner.InferenceEngine({'timing_contract':'ieee_tc_native_v1'}, {})
        engine._resolve_vllm_visible_devices = Mock(return_value='0')
        engine._resolve_vllm_executor_backend = Mock(return_value=None)
        with patch.object(runner, '_lazy_import_vllm', return_value=True), \
             patch.object(runner, 'AsyncEngineArgs', side_effect=lambda **kw: SimpleNamespace(**kw)), \
             patch.object(runner, 'AsyncLLMEngine', SimpleNamespace(
                 from_engine_args=Mock(side_effect=RuntimeError('native operator failed')))), \
             patch.object(runner, '_kill_stale_gpu_processes') as global_kill:
            with self.assertRaisesRegex(RuntimeError, 'without config retry.*native operator failed'):
                asyncio.run(engine._try_create_engine('model', tp=1, gpu_util=.8, max_len=1024,
                    eager=True, enable_lora=True, max_loras=2, max_lora_rank=16))
            global_kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
