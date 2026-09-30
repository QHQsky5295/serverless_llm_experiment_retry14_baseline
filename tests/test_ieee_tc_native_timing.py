"""Native clock/token contracts using deterministic events, never GPU timing claims."""
import asyncio
import hashlib
import json
import math
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from faaslora.metrics.metrics_collector import NativeV1TokenTimeline
from scripts.run_all_experiments import (
    InferenceEngine, RequestExecutionPlan, RequestResult, ScenarioResult, ScenarioRunner,
    SubprocessInferenceEngineProxy,
    _attach_parent_rpc_breakdown,
)


def metrics(**changes):
    fields = dict(arrival_time=1_800_000_000., queued_ts=100.1, scheduled_ts=100.2,
                  first_token_ts=100.5, last_token_ts=101.1, num_generation_tokens=3)
    fields.update(changes)
    return SimpleNamespace(**fields)


class NativeTokenContract(unittest.TestCase):
    def test_wall_clock_arrival_and_completion_are_not_decode(self):
        timeline = NativeV1TokenTimeline(100., 'clock-a')
        timeline.observe(metrics(), [11, 12, 13], finished=True)
        result = timeline.finalize(102.)
        self.assertAlmostEqual(result['native_ttft_ms'], 500.)
        self.assertAlmostEqual(result['native_decode_ms'], 600.)
        self.assertAlmostEqual(result['native_tpot_ms'], 300.)
        self.assertAlmostEqual(result['worker_completion_notification_ms'], 900.)

    def test_terminal_without_new_token_does_not_move_last_boundary(self):
        timeline = NativeV1TokenTimeline(100., 'clock-a')
        timeline.observe(metrics(num_generation_tokens=1, last_token_ts=100.5), [11], finished=False)
        timeline.observe(metrics(), [11, 12, 13], finished=False)
        timeline.observe(metrics(last_token_ts=102.), [11, 12, 13], finished=True)
        self.assertEqual(timeline.finalize(102.)['native_last_token_monotonic_s'], 101.1)

    def test_single_token_tpot_is_null_in_native_payload(self):
        timeline = NativeV1TokenTimeline(100., 'c')
        timeline.observe(metrics(num_generation_tokens=1, last_token_ts=100.5), [11], finished=True)
        result = timeline.finalize(101.)
        self.assertIsNone(result['native_tpot_ms'])
        transported = _attach_parent_rpc_breakdown(result)
        self.assertIsNone(transported['native_tpot_ms'])
        self.assertEqual(transported['native_clock_id'], 'c')
        self.assertIs(transported['native_terminal_observed'], True)

    def test_missing_native_fields_never_fall_back_to_text_or_legacy(self):
        for stats in (None, SimpleNamespace(arrival_time=100., first_token_time=100.5),
                      metrics(last_token_ts=math.nan), metrics(num_generation_tokens=4),
                      metrics(first_token_ts=99.), metrics(queued_ts=0.)):
            with self.assertRaises(ValueError):
                NativeV1TokenTimeline(100., 'c').observe(stats, [11, 12, 13], finished=True)

    def test_cumulative_count_identity_and_first_event_are_immutable(self):
        for ids, stats in [([12, 13], metrics(num_generation_tokens=2)),
                           ([], metrics(num_generation_tokens=0)),
                           ([11, 12], metrics(num_generation_tokens=2, first_token_ts=100.6)),
                           ([11, '12'], metrics(num_generation_tokens=2))]:
            timeline = NativeV1TokenTimeline(100., 'c')
            timeline.observe(metrics(num_generation_tokens=1, last_token_ts=100.5), [11], finished=False)
            with self.assertRaises(ValueError):
                timeline.observe(stats, ids, finished=True)

    def test_incomplete_or_duplicate_terminal_is_not_success(self):
        timeline = NativeV1TokenTimeline(100., 'c')
        with self.assertRaises(ValueError):
            timeline.finalize(102.)
        timeline.observe(metrics(), [11, 12, 13], finished=True)
        with self.assertRaises(ValueError):
            timeline.observe(metrics(), [11, 12, 13], finished=True)
        with self.assertRaises(ValueError):
            timeline.finalize(101.)

    def test_service_intervals_sum_without_inventing_arrival_or_acquisition(self):
        timeline = NativeV1TokenTimeline(100., 'clock-a')
        timeline.observe(metrics(), [11, 12, 13], finished=True)
        result = NativeV1TokenTimeline.service_breakdown(
            timeline.finalize(102.), admitted_at=99., completed_at=102.5, clock_id='clock-a')
        keys = ['admission_to_engine_dispatch_ms', 'engine_entry_to_queue_ms',
                'native_engine_queue_ms', 'native_prefill_ms', 'native_decode_ms',
                'worker_completion_notification_ms', 'worker_to_controller_completion_ms']
        self.assertAlmostEqual(sum(result[key] for key in keys), 3500., places=6)
        self.assertAlmostEqual(result['admitted_service_ttft_ms'], 1500.)
        self.assertAlmostEqual(result['admitted_service_e2e_ms'], 3500.)
        self.assertNotIn('arrival_monotonic_s', result)
        self.assertNotIn('executable_acquired_monotonic_s', result)
        for changes in (dict(clock_id='clock-b'), dict(admitted_at=100.6),
                        dict(completed_at=101.5)):
            args = dict(admitted_at=99., completed_at=102.5, clock_id='clock-a')
            args.update(changes)
            with self.assertRaises(ValueError):
                NativeV1TokenTimeline.service_breakdown(timeline.finalize(102.), **args)


class EngineTimingIntegration(unittest.TestCase):
    def engine(self, *, native=True, ids=(11, 12, 13), stats=None, finished=True, finish_reason='length'):
        class FakeVllmEngine:
            async def generate(self, **kwargs):
                yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=list(ids), finish_reason=finish_reason)],
                                      metrics=stats if stats is not None else metrics(),
                                      prompt_token_ids=[1, 2], finished=finished)
            async def ieee_retire_request(self, external, *, abort):
                return {'retired': True, 'external_request_id': external}
        engine = InferenceEngine({"backend": "vllm", "generation_contract": "fixed_length_greedy_v1",
                                  "timing_contract": "ieee_tc_native_v1" if native else "legacy"}, {})
        engine.engine = FakeVllmEngine()
        engine._lora_in_engine = False
        return engine

    def generate(self, engine, *, target=3):
        with patch('scripts.run_all_experiments.SamplingParams',
                   side_effect=lambda **kw: SimpleNamespace(**kw)), \
             patch('scripts.run_all_experiments.time.perf_counter', side_effect=[100., 102.]), \
             patch('faaslora.metrics.metrics_collector.local_monotonic_clock_id', return_value='clock-a'):
            return asyncio.run(engine.generate_prepared(
                request_plan=RequestExecutionPlan('hello', 2, target), lora_path=None,
                adapter_id=None, return_timing=True))

    def test_actual_generate_prepared_returns_native_timing(self):
        ttft, tpot, count, timing = self.generate(self.engine())
        self.assertAlmostEqual(ttft, 500.)
        self.assertAlmostEqual(tpot, 300.)
        self.assertEqual(count, 3)
        self.assertEqual(timing['timing_contract'], 'ieee_tc_native_v1')
        self.assertEqual(timing['native_output_tokens'], 3)
        self.assertEqual(timing['backend_request_id'], 'req_1')
        self.assertAlmostEqual(timing['worker_wall_e2e_ms'], 2000.)
        self.assertAlmostEqual(timing['runtime_estimated_e2e_ms'], 1100.)
        self.assertEqual(timing['native_prompt_token_ids_sha256'],
                         hashlib.sha256(b'[1,2]').hexdigest())

    def test_actual_engine_rejects_wrong_fixed_output_and_missing_terminal(self):
        with self.assertRaisesRegex(RuntimeError, 'output length violates'):
            self.generate(self.engine(), target=4)
        with self.assertRaisesRegex(RuntimeError, 'incomplete native'):
            self.generate(self.engine(finished=False))

    def test_frontend_abort_is_not_success_even_when_token_count_matches(self):
        for reason in ('abort', 'error', None):
            with self.subTest(reason=reason), self.assertRaisesRegex(RuntimeError, 'non-success native'):
                self.generate(self.engine(finish_reason=reason))
        with self.assertRaisesRegex(RuntimeError, 'finish by length'):
            self.generate(self.engine(finish_reason='stop'))

    def test_actual_engine_does_not_retry_hidden_attempts(self):
        engine = self.engine()
        class FailedEngine:
            async def generate(self, **kw):
                raise RuntimeError('engine died')
                yield  # native asynchronous generator interface
        engine.engine = FailedEngine()
        engine.reinitialize = AsyncMock()
        with self.assertRaisesRegex(RuntimeError, 'engine died'):
            self.generate(engine)
        engine.reinitialize.assert_not_awaited()

    def test_rpc_preserves_native_null_and_identity(self):
        proxy = SubprocessInferenceEngineProxy.__new__(SubprocessInferenceEngineProxy)
        proxy._rpc = AsyncMock(return_value={
            'ttft_ms': 500., 'tpot_ms': 0., 'output_tokens': 1,
            'timing': {'native_tpot_ms': None, 'native_clock_id': 'clock-a',
                       'timing_contract': 'ieee_tc_native_v1'}})
        result = asyncio.run(proxy.generate('p', None, None, 1, 1, return_timing=True))
        self.assertIsNone(result[3]['native_tpot_ms'])
        self.assertEqual(result[3]['native_clock_id'], 'clock-a')

    def test_rpc_preserves_typed_terminal_acknowledgement(self):
        for flag in (True, False, 1.0, None):
            with self.subTest(flag=flag):
                payload = json.loads(json.dumps({
                    'ttft_ms': 500., 'tpot_ms': 0., 'output_tokens': 1,
                    'timing': {'native_terminal_observed': flag}}))
                # Both actual parent timing normalization sites must preserve
                # the Boolean; numbers/null must not become acknowledgements.
                payload['timing'] = _attach_parent_rpc_breakdown(payload['timing'])
                proxy = SubprocessInferenceEngineProxy.__new__(SubprocessInferenceEngineProxy)
                proxy._rpc = AsyncMock(return_value=payload)
                result = asyncio.run(proxy.generate('p', None, None, 1, 1, return_timing=True))
                observed = result[3]['native_terminal_observed']
                self.assertEqual(type(observed), type(flag))
                self.assertEqual(observed, flag)

    def test_native_zero_is_observed_but_single_token_is_not(self):
        result = ScenarioResult('native', 'faaslora_full', total=3)
        result.requests = [RequestResult(
            request_id=str(i), adapter_id=None, is_burst=False, burst_phase='normal',
            cache_hit=False, cache_tier='backbone', lora_io_ms=0., vllm_ttft_ms=10.,
            ttft_ms=10., contention_ms=0., defer_ms=0., tpot_ms=tpot,
            e2e_ms=20., input_tokens=2, output_tokens=count, cost_usd=0., success=True,
            timing_contract='ieee_tc_native_v1', tpot_observed=count > 1)
            for i, (tpot, count) in enumerate(((0., 3), (10., 3), (None, 1)))]
        result.aggregate(1.)
        self.assertEqual(result.avg_tpot_ms, 5.)

    def test_actual_controller_uses_native_decode_not_completion_tail(self):
        class FakeEngine:
            backend = 'vllm'
            def prepare_request(self, **kwargs):
                return RequestExecutionPlan('p', 2, 2)
            async def generate_prepared(self, **kwargs):
                base = time.perf_counter()
                timeline = NativeV1TokenTimeline(base, 'clock-a')
                stats = metrics(queued_ts=base+.001, scheduled_ts=base+.002,
                                first_token_ts=base+.003, last_token_ts=base+.004,
                                num_generation_tokens=2)
                timeline.observe(stats, [11, 12], finished=True)
                timing = timeline.finalize(base+.008)
                return timing['native_ttft_ms'], timing['native_tpot_ms'], 2, timing
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner.model_cfg = {'timing_contract': 'ieee_tc_native_v1'}
        runner._unsettled_runtime_reservations = {}
        runner.router = None
        runner.engine = FakeEngine()
        runner.coordinator = None
        runner._stack = None
        runner._refresh_all_slot_runtime_hints = lambda: None
        runner._refresh_slot_runtime_hints = lambda _slot: None
        runner.adapter_info = {}
        runner.baseline_type = 'faaslora_full'
        runner._access_count = {}
        runner.cost_model = {}
        runner._generation_contract = 'fixed_length_greedy_v1'
        runner._resolve_lora = AsyncMock(return_value=(None, None, 0., 'backbone', 0., 0.))
        trace = SimpleNamespace(request_id='r', adapter_id=None, is_burst=False,
                                expected_input_tokens=2, prompt_input_tokens=2,
                                expected_output_tokens=2, prompt_output_tokens=2, prompt='p')
        ticks = iter(100. + i*.01 for i in range(100))
        with patch('scripts.run_all_experiments.time.perf_counter', side_effect=lambda: next(ticks)), \
             patch('faaslora.metrics.metrics_collector.local_monotonic_clock_id', return_value='clock-a'):
            # The outer admission ended at 99.5, before wrapper/slot setup at
            # 100+. That real interval must not disappear between two timers.
            result = asyncio.run(runner._exec_request(trace, max_tokens=2, temperature=0.,
                dispatch_admitted_at=99.5, dispatch_admission_wait_ms=1500.,
                dispatch_window_wait_ms=1250., arrival_release_lateness_ms=250.,
                admitted_offset_s=1.5, scheduled_arrival_offset_s=0.,
                arrival_released_offset_s=.25))
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.timing_contract, 'ieee_tc_native_v1')
        self.assertAlmostEqual(result.tpot_ms, 1.)
        self.assertGreater(result.service_e2e_ms-result.service_ttft_ms, result.tpot_ms)
        self.assertAlmostEqual(result.native_token_timing['worker_completion_notification_ms'], 4.)
        native = result.native_token_timing
        admitted = native['controller_admitted_monotonic_s']
        self.assertAlmostEqual(result.runtime_slot_wait_ms, (admitted-99.5)*1000.)
        self.assertGreaterEqual(result.runtime_slot_wait_ms, 500.)
        self.assertAlmostEqual(result.dispatch_admission_wait_ms, (admitted-98.)*1000.)
        self.assertAlmostEqual(result.admitted_offset_s, admitted-98.)
        self.assertAlmostEqual(result.dispatch_admission_wait_ms,
            result.arrival_release_lateness_ms + result.dispatch_window_wait_ms
            + result.runtime_slot_wait_ms)
        self.assertAlmostEqual(result.overall_ttft_ms,
            (native['native_first_token_monotonic_s']-98.)*1000.)
        self.assertAlmostEqual(result.overall_e2e_ms,
            (native['controller_completed_monotonic_s']-98.)*1000.)


class FixedWorkContract(unittest.TestCase):
    def engine(self, **overrides):
        config = {'generation_contract': 'fixed_length_greedy_v1', 'max_model_len': 1024,
                  'max_input_len': 759, 'max_output_tokens_cap': 256}
        config.update(overrides)
        engine = InferenceEngine(config, {})
        engine._prompt_guard_tokenizer = SimpleNamespace(
            encode=lambda text, add_special_tokens=False: ([1] if add_special_tokens else []) +
                [ord(char) for char in text],
            decode=lambda ids, skip_special_tokens=False: ''.join(chr(token) for token in ids))
        return engine

    def test_tokenizer_failure_cannot_become_character_estimate(self):
        engine = self.engine()
        engine._get_prompt_guard_tokenizer = Mock(side_effect=ValueError('tokenizer unavailable'))
        with self.assertRaisesRegex(RuntimeError, 'fallback forbidden'):
            engine.prepare_request('text', 3, 999)

    def test_common_output_target_cannot_shrink_at_model_or_context_guard(self):
        with self.assertRaisesRegex(ValueError, 'common fixed-output target'):
            self.engine(max_output_tokens_cap=2).prepare_request('text', 3, 4)
        with self.assertRaisesRegex(RuntimeError, 'fallback forbidden'):
            self.engine(max_model_len=34, max_input_len=32).prepare_request('x'*32, 30, 32)
        plan = self.engine().prepare_request('x'*900, 256, 999)
        self.assertEqual((len(plan.prompt), plan.input_tokens, plan.max_tokens), (759, 759, 256))

    def test_normalization_is_explicit_and_degenerate_boundary_does_not_guess(self):
        engine = self.engine(max_input_len=4)
        engine._prompt_guard_tokenizer.decode = lambda ids, **kw: ''.join(chr(token) for token in ids).upper()
        self.assertEqual(engine.prepare_request('abc', 3, 3).prompt, 'ABC')
        engine._prompt_guard_tokenizer.decode = lambda ids, **kw: 'x'*9
        with self.assertRaisesRegex(RuntimeError, 'fallback forbidden'):
            engine.prepare_request('abc', 3, 3)

    def test_source_target_is_required_not_filled_from_other_fields(self):
        scenario = ScenarioRunner.__new__(ScenarioRunner)
        scenario.wl_cfg = {'generation_contract': 'fixed_length_greedy_v1'}
        for source in (None, 0, -1, True, 1.5, '5'):
            trace = SimpleNamespace(expected_output_tokens=source, prompt_output_tokens=20)
            with self.subTest(source=source), self.assertRaises(ValueError):
                scenario._trace_requested_output_tokens(trace, 100)
        self.assertEqual(scenario._trace_requested_output_tokens(
            SimpleNamespace(expected_output_tokens=300), 100), 256)

    def test_missing_request_preparer_is_not_an_implicit_legacy_path(self):
        scenario = ScenarioRunner.__new__(ScenarioRunner)
        scenario._generation_contract = 'fixed_length_greedy_v1'
        scenario.wl_cfg = {'generation_contract': 'fixed_length_greedy_v1'}
        trace = SimpleNamespace(expected_output_tokens=3, prompt='text', expected_input_tokens=4)
        with self.assertRaisesRegex(RuntimeError, 'real prompt/token request preparer'):
            scenario._prepare_request_execution_plan(SimpleNamespace(), trace, 3)

    def test_native_chat_renderer_is_frozen_not_selected_by_exception(self):
        messages = [{'role': 'user', 'content': 'hello'}]
        engine = self.engine(timing_contract='ieee_tc_native_v1')
        with self.assertRaisesRegex(ValueError, 'frozen canonical_prompt_renderer'):
            engine.prepare_request('unused', 3, 4, chat_messages=messages)
        engine.model_cfg['canonical_prompt_renderer'] = 'role_lines_v1'
        self.assertEqual(engine.prepare_request('unused', 3, 4, chat_messages=messages).prompt, 'User: hello')
        engine.model_cfg['canonical_prompt_renderer'] = 'tokenizer_chat_template'
        engine._prompt_guard_tokenizer.apply_chat_template = Mock(side_effect=ValueError('bad template'))
        with self.assertRaisesRegex(ValueError, 'bad template'):
            engine.prepare_request('unused', 3, 4, chat_messages=messages)

    def test_prepared_prompt_survives_actual_proxy_and_worker_decoder_once(self):
        from scripts.dedicated_engine_worker import _decode_prepared_request
        proxy = SubprocessInferenceEngineProxy.__new__(SubprocessInferenceEngineProxy)
        proxy._rpc = AsyncMock(return_value={'ttft_ms': 1., 'tpot_ms': 1., 'output_tokens': 3,
                                            'timing': {'worker_wall_e2e_ms': 3.}})
        plan = RequestExecutionPlan('canonical once', 2, 3)
        asyncio.run(proxy.generate_prepared(request_plan=plan, lora_path=None, adapter_id=None))
        payload = json.loads(json.dumps(proxy._rpc.call_args.kwargs))
        decoded = _decode_prepared_request(payload)
        self.assertEqual(decoded['_prepared_request'], plan)
        payload['prompt'] = 'changed across boundary'
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            _decode_prepared_request(payload)

    def test_missing_native_prompt_and_base_model_substitution_reject(self):
        helper = EngineTimingIntegration()
        engine = helper.engine()
        class MissingPromptEngine:
            async def generate(self, **kwargs):
                yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[11, 12, 13], finish_reason='length')],
                                      metrics=metrics(), prompt_token_ids=None, finished=True)
        engine.engine = MissingPromptEngine()
        with self.assertRaisesRegex(RuntimeError, 'native prompt token IDs required'):
            helper.generate(engine)
        with patch('scripts.run_all_experiments.SamplingParams', side_effect=lambda **kw: SimpleNamespace(**kw)):
            with self.assertRaisesRegex(RuntimeError, 'cannot fall back to the base model'):
                asyncio.run(engine.generate_prepared(request_plan=RequestExecutionPlan('p', 1, 3),
                    lora_path='/existing/adapter', adapter_id='a'))


if __name__ == '__main__':
    unittest.main()
