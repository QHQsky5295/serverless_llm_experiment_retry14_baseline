"""Native D/T/O bridge tests: deterministic events and real loopback RPC, no GPU."""
import asyncio
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from faaslora.experiment.instance_pool import (
    ServiceComponents, ServiceCostModel, ServiceIntervalObservation,
    ServiceObservationClass, NativeServiceIntervalObserver,
)
from scripts import dedicated_engine_worker as worker
from scripts.run_all_experiments import (
    InferenceEngine, RequestExecutionPlan, SubprocessInferenceEngineProxy,
)


def observer(*, adapter=False):
    key = ServiceObservationClass('nvme', 0, 0, 0, 0, 'fixture-fp16', 0)
    costs = ServiceCostModel({key: ServiceComponents(100., 200., 300.)},
                             beta=.5, profile_id='deterministic-test-only')
    observation = ServiceIntervalObservation(costs, key, 99.)
    observation.acquire(100.)
    return NativeServiceIntervalObserver(observation, clock_id='clock-a',
        adapter_id='a' if adapter else None,
        gpu_reference=dict(owner_id='o', lease_id='l', adapter_int_id=1) if adapter else None)


def event(index=1, *, adapter=False, **changes):
    value = dict(contract='native_service_events_v1', kind='first_token' if index == 1 else 'last_token',
                 sequence=index, backend_request_id='req_1', native_clock_id='clock-a',
                 timestamp_monotonic_s=100.5 if index == 1 else 101.1,
                 token_count=1 if index == 1 else 3, adapter_id='a' if adapter else None,
                 gpu_reference_owner_id='o' if adapter else None,
                 gpu_reference_lease_id='l' if adapter else None,
                 gpu_reference_adapter_int_id=1 if adapter else None)
    return value | changes


class NativeServiceObservationTests(unittest.TestCase):
    def test_initial_profile_collection_has_no_synthetic_estimate(self):
        key = ServiceObservationClass('host', 0, 0, 0, 0, 'fixture-fp16', 0)
        obs = ServiceIntervalObservation.for_profiling(key, 99.)
        self.assertIsNone(obs.model)
        obs.acquire(100.)
        receive = NativeServiceIntervalObserver(obs, clock_id='clock-a', adapter_id=None)
        receive(event())
        receive(event(2))
        self.assertEqual((obs.admitted_at, obs.acquired_at, obs.first_at, obs.last_at),
                         (99., 100., 100.5, 101.1))
        self.assertTrue(obs.closed)
        with self.assertRaises(AttributeError):
            ServiceIntervalObservation(None, key, 99.)

    def test_profile_collection_keeps_gpu_zero_and_cancel_invariants(self):
        key = ServiceObservationClass('gpu', 0, 0, 0, 0, 'fixture-fp16', 0)
        obs = ServiceIntervalObservation.for_profiling(key, 100.)
        self.assertEqual(obs.acquired_at, 100.)
        with self.assertRaises(ValueError):
            obs.acquire(100.1)
        obs.first_token(100.5)
        obs.cancel()
        with self.assertRaises(ValueError):
            obs.last_token(101.)
        self.assertIsNone(obs.last_at)

    def test_admission_class_and_each_completed_interval_updated_once(self):
        receive = observer(adapter=True)
        receive(event(adapter=True))
        obs = receive.observation
        self.assertEqual(obs.model.sample_counts(obs.key), dict(d_ms=1, t_ms=1, o_ms=0))
        self.assertEqual(obs.model.estimate(obs.key), ServiceComponents(550., 350., 300.))
        receive(event(2, adapter=True))
        self.assertAlmostEqual(obs.model.estimate(obs.key).o_ms, 450.)
        self.assertTrue(obs.closed)
        with self.assertRaises(ValueError):
            receive(event(2, adapter=True))

    def test_bad_identity_clock_order_and_values_do_not_update(self):
        cases = [dict(native_clock_id='elsewhere'), dict(adapter_id='b'),
                 dict(gpu_reference_lease_id='wrong'), dict(gpu_reference_adapter_int_id=True),
                 dict(backend_request_id=''),
                 dict(sequence=True), dict(kind='last_token'), dict(token_count=0),
                 dict(timestamp_monotonic_s=99.), dict(timestamp_monotonic_s=float('nan'))]
        for changes in cases:
            receive = observer(adapter=True)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                receive(event(adapter=True, **changes))
            self.assertEqual(receive.observation.model.sample_counts(receive.observation.key),
                             dict(d_ms=1, t_ms=0, o_ms=0))

    def test_cancel_retains_completed_prefix_and_rejects_late_event(self):
        receive = observer()
        receive(event())
        receive.observation.cancel()
        with self.assertRaises(ValueError):
            receive(event(2))
        self.assertEqual(receive.observation.model.sample_counts(receive.observation.key),
                         dict(d_ms=1, t_ms=1, o_ms=0))

    def test_last_request_mismatch_and_single_token_contract(self):
        receive = observer()
        receive(event())
        with self.assertRaisesRegex(ValueError, 'identity'):
            receive(event(2, backend_request_id='req_other'))
        with self.assertRaisesRegex(ValueError, 'single-token'):
            receive(event(2, token_count=1))
        receive(event(2, token_count=1, timestamp_monotonic_s=100.5))
        self.assertAlmostEqual(receive.observation.model.estimate(receive.observation.key).o_ms, 150.)


class NativeEngineEventTests(unittest.IsolatedAsyncioTestCase):
    async def exercise_stream(self, *, cancel=False):
        receive = observer()
        first_received = asyncio.Event()
        release_decode = asyncio.Event()
        def accept(value):
            receive(value)
            if value['sequence'] == 1:
                first_received.set()
        class FakeNative:
            async def generate(self, **kwargs):
                for ids, finished, last in (([11], False, 100.5), ([11, 12, 13], True, 101.1)):
                    yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=ids, finish_reason='length')],
                        metrics=SimpleNamespace(queued_ts=100.1, scheduled_ts=100.2,
                            first_token_ts=100.5, last_token_ts=last, num_generation_tokens=len(ids)),
                        prompt_token_ids=[1, 2], finished=finished)
                    if not finished:
                        await release_decode.wait()
        engine = InferenceEngine(dict(backend='vllm', generation_contract='fixed_length_greedy_v1',
                                      timing_contract='ieee_tc_native_v1'), {})
        engine.engine = FakeNative()
        engine._lora_in_engine = False
        with patch('scripts.run_all_experiments.SamplingParams', side_effect=lambda **kw: SimpleNamespace(**kw)), \
             patch('scripts.run_all_experiments.time.perf_counter', side_effect=[100., 102.]), \
             patch('faaslora.metrics.metrics_collector.local_monotonic_clock_id', return_value='clock-a'):
            task = asyncio.create_task(engine.generate_prepared(request_plan=RequestExecutionPlan('p', 2, 3),
                adapter_id=None, lora_path=None, return_timing=True, native_event_observer=accept))
            try:
                await asyncio.wait_for(first_received.wait(), 1.)
                self.assertFalse(task.done())
                self.assertEqual(receive.observation.model.sample_counts(receive.observation.key)['o_ms'], 0)
                if cancel:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    receive.observation.cancel()
                    self.assertEqual(len(receive.events), 1)
                else:
                    release_decode.set()
                    result = await task
                    self.assertEqual(result[2], 3)
                    self.assertAlmostEqual(receive.observation.last_at, 101.1)
            finally:
                release_decode.set()
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    async def test_real_engine_method_publishes_before_stream_finishes(self):
        await self.exercise_stream()

    async def test_real_engine_cancellation_does_not_emit_decode_completion(self):
        await self.exercise_stream(cancel=True)

    async def test_legacy_contract_cannot_emit_native_events(self):
        engine = InferenceEngine(dict(backend='vllm', timing_contract='legacy'), {})
        with self.assertRaisesRegex(ValueError, 'native timing'):
            await engine.generate('p', None, None, 3, 2, native_event_observer=observer())


class NativeWorkerRPCEvents(unittest.IsolatedAsyncioTestCase):
    async def roundtrip(self, *, cancel=False, wrong_terminal=False, fail_after_first=False,
                        duplicate_first=False, large_control_payload=False):
        receive = observer()
        seen = asyncio.Event()
        proceed = asyncio.Event()
        finished = asyncio.Event()
        ready = asyncio.get_running_loop().create_future()
        class FakeEngine:
            def __init__(self, *args):
                self.model_cfg = dict(args[0])
            async def initialize(self):
                self.model_cfg['max_cpu_loras'] = 24
            async def shutdown(self):
                pass
            async def ieee_scheduler_observation(self):
                return {'admitted': [], 'kind': 'loopback-test-only'}
            async def ieee_prepare_host(self, **command):
                return {'acquired': False, 'reason': 'defer_effective_capacity', 'command': command}
            async def ieee_transfer_event(self, **command):
                return {'kind': 'ieee_adapter_transfer_event_v1', 'transfer_id': command['transfer_id'],
                        'state': {'start': 'active', 'finish': 'finished'}[command['operation']]}
            async def ieee_register_pending(self, **command):
                return {'intent_id': command['intent_id'], 'state': 'pending', 'command': command}
            async def ieee_close_pending(self, **command):
                return {'intent_id': command['intent_id'], 'closed': True}
            async def ieee_gpu_reference(self, **command):
                if command in ({'operation': 'source_snapshot'}, {'operation': 'routing_source_snapshot'},
                               {'operation': 'source_identity_snapshot'}):
                    from faaslora.clock import local_monotonic_clock_id
                    from tests.test_ieee_tc_service_routing import measured_source_payload
                    value = measured_source_payload() | dict(clock_id=local_monotonic_clock_id(),
                        device_uuid='GPU-loopback-test')
                    if command['operation'] == 'source_identity_snapshot':
                        value.pop('native_footprints')
                    return value
                return {'command': command, 'registered': True}
            async def ieee_routing_sources(self):
                return await InferenceEngine.ieee_routing_sources(self)
            async def ieee_source_identities(self):
                return await InferenceEngine.ieee_source_identities(self)
            async def generate(self, **kwargs):
                if kwargs.get('pending_admission_id') != 'pending-test':
                    raise ValueError('dedicated generation lost its pending identity')
                emit = kwargs['native_event_observer']
                emit(event())
                await proceed.wait()
                try:
                    if fail_after_first:
                        raise RuntimeError('deliberate failed generation')
                    emit(event() if duplicate_first else event(2))
                    return 500., 300., 3, dict(backend_request_id='wrong' if wrong_terminal else 'req_1',
                        native_clock_id='clock-a', native_first_token_monotonic_s=100.5,
                        native_last_token_monotonic_s=101.1)
                finally:
                    finished.set()
        def accept(value):
            receive(value)
            seen.set()
        with tempfile.TemporaryDirectory() as temp:
            payload = Path(temp) / 'payload.json'
            payload.write_text(json.dumps(dict(repo_root=str(Path.cwd()), model_cfg={}, cost_model={})))
            with patch('scripts.run_all_experiments.InferenceEngine', FakeEngine), \
                 patch.object(worker, '_write_ready', side_effect=lambda path, value: ready.set_result(value)):
                server_task = asyncio.create_task(worker._run_worker(payload, Path(temp) / 'ready.json'))
                address = await asyncio.wait_for(ready, 2.)
                self.assertEqual(address['configuration_contract'], 'initialized_model_config_v1')
                self.assertEqual(address['model_config'], {'max_cpu_loras':24})
                proxy = SubprocessInferenceEngineProxy.__new__(SubprocessInferenceEngineProxy)
                proxy.model_cfg = dict(timing_contract='ieee_tc_native_v1')
                proxy._engine_dead = False
                proxy._native_rpc_uncertain = {}
                proxy._process = SimpleNamespace(poll=Mock(return_value=None))
                proxy._host, proxy._port = address['host'], address['port']
                proxy._rpc_pool_size = 1
                proxy._rpc_channel_init_lock = asyncio.Lock()
                proxy._rpc_channel_queue = None
                proxy._rpc_channels = []
                proxy._with_worker_log_context = lambda value: value
                self.assertEqual((await proxy.ieee_scheduler_observation())['admitted'], [])
                from faaslora.experiment.instance_pool import NativeSourceSnapshot
                from faaslora.clock import local_monotonic_clock_id
                routing = await proxy.ieee_routing_sources()
                compact = NativeSourceSnapshot.from_routing_wire(routing,
                    expected_clock_id=local_monotonic_clock_id(), received_monotonic_s=time.monotonic())
                full = await proxy.ieee_gpu_reference(operation='source_snapshot')
                self.assertEqual(compact, NativeSourceSnapshot.from_native(full,
                    expected_clock_id=local_monotonic_clock_id(), received_monotonic_s=time.monotonic()))
                self.assertNotIn('native_footprints', routing)
                self.assertIn('native_footprints', full)
                identities = await proxy.ieee_source_identities()
                observed_identity = NativeSourceSnapshot.from_routing_wire(identities,
                    expected_clock_id=local_monotonic_clock_id(), received_monotonic_s=time.monotonic())
                self.assertEqual(observed_identity, compact.identity_view())
                self.assertIsNone(identities['routing_footprints'])
                self.assertFalse(proxy._native_rpc_uncertain)
                if large_control_payload:
                    objective = {'candidates': [dict(adapter_id='adapter_'+str(i),
                        metadata='x'*2048, benefit_ms=float(i)) for i in range(500)]}
                    self.assertGreater(len(json.dumps(objective)), 65536)
                    # Actual production reader/writer, no StreamReader stand-in.
                    reply = await proxy.ieee_gpu_reference(operation='register_preparation_plan',
                        plan_id='large', objective=objective, target_adapter_ids=[1],
                        expected_owner_id='worker')
                    self.assertEqual(reply['command']['objective'], objective)
                    self.assertGreater(reply['timing']['parent_rpc_request_bytes'], 65536)
                    self.assertGreater(reply['timing']['parent_rpc_response_bytes'], 65536)
                    # A following frame on the same pool must remain aligned.
                    self.assertEqual((await proxy.ieee_scheduler_observation())['admitted'], [])
                prepared = await proxy.ieee_prepare_host(lease_id='prepare-test', expected_owner_id='worker')
                self.assertFalse(prepared['acquired'])
                self.assertEqual(prepared['command']['lease_id'], 'prepare-test')
                started = await proxy.ieee_transfer_event(operation='start', transfer_id='transfer-test')
                self.assertEqual(started['state'], 'active')
                ended = await proxy.ieee_transfer_event(operation='finish', transfer_id='transfer-test')
                self.assertEqual(ended['state'], 'finished')
                registered = await proxy.ieee_register_pending(intent_id='pending-test', prompt='p',
                    max_tokens=3, adapter_id=None)
                self.assertEqual(registered['command']['prompt'], 'p')
                task = asyncio.create_task(proxy.generate_prepared(request_plan=RequestExecutionPlan('p', 2, 3),
                    adapter_id=None, lora_path=None, return_timing=True, native_event_observer=accept,
                    pending_admission_id='pending-test'))
                try:
                    await asyncio.wait_for(seen.wait(), 2.)
                    self.assertFalse(task.done())
                    self.assertEqual(receive.observation.model.sample_counts(receive.observation.key)['t_ms'], 1)
                    if cancel:
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                        receive.observation.cancel()
                        self.assertTrue(proxy._native_rpc_uncertain)
                        self.assertEqual(receive.observation.model.sample_counts(receive.observation.key)['o_ms'], 0)
                    proceed.set()
                    if not cancel:
                        if wrong_terminal or fail_after_first or duplicate_first:
                            with self.assertRaisesRegex(RuntimeError, 'native terminal|deliberate failed|progress sequence'):
                                await asyncio.wait_for(task, 2.)
                            self.assertTrue(proxy._native_rpc_uncertain)
                        else:
                            self.assertEqual((await asyncio.wait_for(task, 2.))[2], 3)
                            self.assertFalse(proxy._native_rpc_uncertain)
                    await asyncio.wait_for(finished.wait(), 2.)
                    closed = await proxy.ieee_close_pending(intent_id='pending-test')
                    self.assertTrue(closed['closed'])
                    if fail_after_first or duplicate_first:
                        self.assertEqual(len(receive.events), 1)
                    elif not cancel:
                        self.assertEqual(len(receive.events), 2)
                finally:
                    proceed.set()
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    for channel in list(proxy._rpc_channels):
                        await proxy._drop_rpc_channel(channel)
                    await proxy._rpc('shutdown')
                    await asyncio.wait_for(server_task, 2.)

    async def test_actual_worker_and_proxy_stream_completed_intervals(self):
        await self.roundtrip()

    async def test_full_pool_objective_above_default_reader_limit_roundtrips_exactly(self):
        await self.roundtrip(large_control_payload=True)

    async def test_cancelled_rpc_retains_first_interval_and_drops_late_events(self):
        await self.roundtrip(cancel=True)

    async def test_progress_cannot_replace_wrong_terminal_identity(self):
        await self.roundtrip(wrong_terminal=True)

    async def test_generation_failure_preserves_only_completed_prefix(self):
        await self.roundtrip(fail_after_first=True)

    async def test_repeated_progress_is_rejected_before_second_update(self):
        await self.roundtrip(duplicate_first=True)


class BoundedDedicatedRPC(unittest.TestCase):
    def test_readonly_routing_cancellation_does_not_create_native_ownership(self):
        async def check(method):
            proxy = SubprocessInferenceEngineProxy.__new__(SubprocessInferenceEngineProxy)
            proxy.model_cfg = {'timing_contract': 'ieee_tc_native_v1'}
            proxy._engine_dead = False
            proxy._process = SimpleNamespace(poll=lambda: None)
            proxy._native_rpc_uncertain = {}
            proxy._open_rpc_channel = AsyncMock(return_value=object())
            proxy._drop_rpc_channel = AsyncMock()
            proxy._native_rpc_roundtrip = AsyncMock(side_effect=asyncio.CancelledError)
            with self.assertRaises(asyncio.CancelledError):
                await getattr(proxy, method)()
            self.assertFalse(proxy._native_rpc_uncertain)
            proxy._native_rpc_roundtrip.assert_awaited_once()
            proxy._drop_rpc_channel.assert_awaited_once()
        for method in ('ieee_routing_sources', 'ieee_source_identities'):
            with self.subTest(method=method):
                asyncio.run(check(method))

    def test_encoding_limit_is_wire_bytes_and_includes_all_fields(self):
        from scripts import run_all_experiments as r
        with patch.object(r, '_RPC_FRAME_MAX_BYTES', 16):
            self.assertEqual(r._encode_rpc_frame({'v':'1234567'}), b'{"v": "1234567"}\n')
            with self.assertRaisesRegex(ValueError, 'protocol byte limit'):
                r._encode_rpc_frame({'v':'12345678'})
            # ensure_ascii expands Unicode; character count is not wire size.
            with self.assertRaisesRegex(ValueError, 'protocol byte limit'):
                r._encode_rpc_frame({'v':'两个'})

    def test_receiver_rejects_oversize_with_or_without_delimiter(self):
        from scripts import run_all_experiments as r
        for suffix in [b'', b'\n']:
            channel = r._BlockingRPCChannel(sock=Mock(), recv_buffer=bytearray(b'x'*17+suffix))
            with patch.object(r, '_RPC_FRAME_MAX_BYTES', 16), self.assertRaisesRegex(ValueError, 'protocol byte limit'):
                r.SubprocessInferenceEngineProxy._blocking_rpc_roundtrip(None, channel, b'{}\n')
            channel.sock.recv.assert_not_called()

    def test_fragmented_and_coalesced_frames_preserve_boundaries(self):
        from scripts import run_all_experiments as r
        sock = Mock()
        sock.recv.side_effect = [b'{"o', b'k":true}\n{"ok":false}\n']
        channel = r._BlockingRPCChannel(sock=sock)
        with patch.object(r, '_RPC_FRAME_MAX_BYTES', 16):
            a = r.SubprocessInferenceEngineProxy._blocking_rpc_roundtrip(None, channel, b'{}\n')[0]
            b = r.SubprocessInferenceEngineProxy._blocking_rpc_roundtrip(None, channel, b'{}\n')[0]
        self.assertEqual(json.loads(a), {'ok':True})
        self.assertEqual(json.loads(b), {'ok':False})
        self.assertEqual(sock.recv.call_count, 2)
