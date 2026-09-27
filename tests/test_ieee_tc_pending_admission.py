"""Pending/native ownership handoff: real bridge methods, synthetic owners.

No CUDA or performance qualification. Exact-token descriptors include native
special tokens; transport gaps/cancellation cannot remove predicted demand.
"""
import asyncio
import inspect
import json
from dataclasses import fields
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

from faaslora.clock import local_monotonic_clock_id
from faaslora.scheduling.resource_coordinator import (
    NativeIterationObservation, NativePendingAdmissions, native_prompt_identity,
    AdmittedKVRequest,
)
from scripts.run_all_experiments import (
    InferenceEngine, RuntimeRequestReservation, RequestExecutionPlan,
    SubprocessInferenceEngineProxy,
)
from tests.test_ieee_tc_scheduler_observation import scheduler, request, observe
from tests import test_ieee_tc_scheduler_observation as native_hook_tests
from tests import test_ieee_tc_request_lifecycle as lifecycle_tests
from tests import test_ieee_tc_admission as admission_tests
from faaslora.scheduling.resource_coordinator import evaluate_ieee_admission
from tests.test_ieee_tc_native_retirement import frontend_type
from tests.test_ieee_tc_request_lifecycle import native_reference_fixture


def descriptor(tokens=(1, 20, 30), limit=64, adapter=4):
    return dict(prompt_tokens=len(tokens), prompt_sha256=native_prompt_identity(tokens),
                output_limit=limit, adapter_int_id=adapter)


def native_request(rid='native-random', tokens=(1, 20, 30), limit=64, adapter=4):
    return request(rid, num_prompt_tokens=len(tokens), prompt_token_ids=list(tokens),
        max_tokens=limit, lora_request=NS(lora_int_id=adapter) if adapter is not None else None,
        num_computed_tokens=0, num_in_flight_tokens=0)


def utility_call(method, *args):
    # vLLM 0.30 EngineCoreProc._convert_msgspec_args counts signature entries,
    # not the arity accepted by Python *args. These commands use plain payloads,
    # not typed msgspec structs. Also exercise the decoded wire container shape.
    wire_args = json.loads(json.dumps(args))
    assert len(wire_args) <= len(inspect.signature(method).parameters)
    return method(*wire_args)


class PendingOwnership(unittest.TestCase):
    def make(self):
        steps = NativeIterationObservation()
        return NativePendingAdmissions(steps, (3, 20))

    def test_bound_transport_gap_stays_counted_then_unique_handoff(self):
        journal = self.make()
        journal.register('intent', descriptor())
        before = journal.snapshot()[0]
        self.assertEqual((before['unprocessed_prompt_tokens'], before['generated_tokens'],
                          before['reserved_unused_token_positions']), (3, 0, 0))
        self.assertEqual(before['input_bucket'], 0)
        journal.bind('intent', 'native-random', descriptor())
        self.assertEqual(len(journal.snapshot()), 1)
        native = native_request()
        journal.validate_add(native)
        journal.added(native.request_id)
        self.assertEqual(journal.snapshot(), [])
        with self.assertRaisesRegex(ValueError, 'changed or reused'):
            journal.validate_add(native)

    def test_withdraw_before_late_registration_is_durable(self):
        journal = self.make()
        self.assertEqual(journal.withdraw('cancelled')['state'], 'withdrawn')
        with self.assertRaisesRegex(ValueError, 'already used'):
            journal.register('cancelled', descriptor())
        journal.register('other', descriptor())
        self.assertEqual(journal.withdraw('other')['state'], 'withdrawn')
        self.assertEqual(journal.withdraw('other')['state'], 'withdrawn')
        self.assertEqual(journal.snapshot(), [])

    def test_bound_cannot_be_cancelled_by_assuming_native_absence(self):
        journal = self.make()
        journal.register('i', descriptor())
        journal.bind('i', 'native-random', descriptor())
        with self.assertRaisesRegex(ValueError, 'retirement'):
            journal.withdraw('i')
        self.assertEqual(len(journal.snapshot()), 1)

    def test_prompt_output_and_adapter_are_all_checked_before_native_add(self):
        for changed in (native_request(tokens=(1, 20, 31)), native_request(limit=65),
                        native_request(adapter=5), native_request(tokens=(20, 30))):
            journal = self.make()
            journal.register('i', descriptor())
            journal.bind('i', 'native-random', descriptor())
            with self.assertRaisesRegex(ValueError, 'changed'):
                journal.validate_add(changed)
            self.assertEqual(len(journal.snapshot()), 1)

    def test_rebind_and_wrong_thread_cannot_create_second_owner(self):
        journal = self.make()
        journal.register('i', descriptor())
        journal.bind('i', 'native-random', descriptor())
        with self.assertRaises(ValueError):
            journal.bind('i', 'second', descriptor())
        with patch('faaslora.scheduling.resource_coordinator.threading.get_ident', return_value=-1):
            with self.assertRaisesRegex(RuntimeError, 'owner thread'):
                journal.snapshot()

    def test_owner_snapshot_includes_pending_and_native_without_block_double_count(self):
        s = scheduler()
        steps = NativeIterationObservation()
        s._ieee_pending_admissions = NativePendingAdmissions(steps, (20, 100))
        s._ieee_pending_admissions.register('i', descriptor())
        result = observe(s, steps)
        self.assertEqual(result['admitted_scope'], 'controller_pending_and_native_unfinished')
        self.assertEqual([r['demand_owner'] for r in result['admitted']], ['native', 'controller_pending'])
        self.assertEqual(result['kv_unreserved_free_blocks'], 8)
        keys = tuple(f.name for f in fields(AdmittedKVRequest))
        self.assertEqual(len([AdmittedKVRequest(**{k: r[k] for k in keys})
                              for r in result['admitted']]), 2)

    def test_actual_core_utility_and_native_add_hook_own_the_handoff(self):
        module, native, core = native_hook_tests.NativeHookWiring().load_adapter()
        native.add_request = lambda self, req: self.requests.__setitem__(req.request_id, req)
        hook = module.IEEENativeAsyncScheduler(scheduler())
        engine_core = core()
        engine_core.scheduler = hook
        utility_call(engine_core.ieee_pending_admission, 'register', ['i', descriptor()])
        utility_call(engine_core.ieee_pending_admission, 'bind', ['i', 'native-random', descriptor()])
        self.assertEqual(len(hook.ieee_scheduler_observation()['admitted']), 2)
        hook.add_request(native_request())
        rows = hook.ieee_scheduler_observation()['admitted']
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r['demand_owner'] == 'native' for r in rows))
        self.assertIn('native-random', hook._ieee_retirement.registered)

    def test_native_converter_rejects_legacy_variadic_transport(self):
        reached = []
        def old_bound_bridge(operation, *args):
            reached.append((operation, args))
        with self.assertRaises(AssertionError):
            utility_call(old_bound_bridge, 'register', 'i', descriptor())
        self.assertEqual(reached, [])

    def test_packet_shape_rejects_before_mutating_pending_owner(self):
        module, _, core = native_hook_tests.NativeHookWiring().load_adapter()
        hook = module.IEEENativeAsyncScheduler(scheduler())
        engine_core = core()
        engine_core.scheduler = hook
        for operation, packet in (('register', ['i']), ('bind', ['i', 'n']),
                                  ('withdraw', []), ('withdraw', ['i', 'extra']),
                                  ('register', {'i': descriptor()}),
                                  ('unknown', ['i']), (None, ['i'])):
            with self.subTest(operation=operation, packet=packet):
                with self.assertRaises(ValueError):
                    utility_call(engine_core.ieee_pending_admission, operation, packet)
                self.assertEqual(hook._ieee_pending_admissions.snapshot(), [])
        result = utility_call(engine_core.ieee_pending_admission, 'withdraw', ['i'])
        self.assertEqual(result['state'], 'withdrawn')
        with self.assertRaisesRegex(ValueError, 'already used'):
            utility_call(engine_core.ieee_pending_admission, 'register', ['i', descriptor()])

    def test_pending_demand_changes_existing_equation_not_physical_allocator(self):
        journal = self.make()
        journal.register('i', descriptor())
        row = journal.snapshot()[0]
        keys = [f.name for f in fields(AdmittedKVRequest)]
        demand = AdmittedKVRequest(**{k: row[k] for k in keys})
        proposal = admission_tests.proposal(footprint_bytes=400, pool_reuse_bytes=300,
                                           additional_storage_bytes=100)
        empty = evaluate_ieee_admission(admission_tests.snapshot(), admission_tests.lengths(), proposal)
        pending = evaluate_ieee_admission(admission_tests.snapshot(admitted=(demand,)),
                                         admission_tests.lengths(), proposal)
        self.assertEqual((empty.predicted_kv_bytes, pending.predicted_kv_bytes), (0, 500))
        self.assertTrue(empty.admit)
        self.assertEqual(pending.reason, 'defer_effective_capacity')
        self.assertEqual(empty.physical_increment_bytes, pending.physical_increment_bytes)


class FrontendHandoff(unittest.TestCase):
    async def make(self):
        frontend = frontend_type()()
        journal = NativePendingAdmissions(NativeIterationObservation(), (3, 20))
        def call(method, operation, arguments):
            if method != 'ieee_pending_admission':
                raise AssertionError(method)
            self.assertIs(type(arguments), list)
            return getattr(journal, operation)(*arguments)
        frontend.engine_core.call_utility_async = AsyncMock(side_effect=call)
        frontend.get_supported_tasks = AsyncMock(return_value=('generate',))
        frontend.input_processor = NS(process_inputs_async=AsyncMock(return_value=NS(
            prompt_token_ids=[1, 20, 30], sampling_params=NS(max_tokens=64))))
        return frontend, journal

    def test_native_tokenization_then_bind_before_original_send(self):
        async def check():
            front, journal = await self.make()
            await front.ieee_register_pending('i', 'text', NS(max_tokens=64), 4)
            self.assertEqual(journal.snapshot()[0]['unprocessed_prompt_tokens'], 3)
            front.ieee_pending['i']['state'] = 'generating'
            req = NS(external_req_id='i', request_id='native-random', prompt_token_ids=[1, 20, 30],
                     sampling_params=NS(max_tokens=64), lora_request=NS(lora_int_id=4))
            task = asyncio.create_task(front._add_request(req, 'text', None, 0, object()))
            await front.entered.wait()
            self.assertEqual(journal.snapshot()[0]['handoff_state'], 'bound')
            self.assertEqual(front.sent, [])
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            front.release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(front.sent, ['native-random'])
            self.assertEqual(len(journal.snapshot()), 1)  # send is not actual ADD
        asyncio.run(check())

    def test_cancel_registration_still_requires_owner_withdrawal(self):
        async def check():
            front, journal = await self.make()
            entered, proceed = asyncio.Event(), asyncio.Event()
            async def process(*args, **kwargs):
                entered.set()
                await proceed.wait()
                return NS(prompt_token_ids=[1, 20, 30], sampling_params=NS(max_tokens=64))
            front.input_processor.process_inputs_async.side_effect = process
            task = asyncio.create_task(front.ieee_register_pending('i', 'text', NS(max_tokens=64), 4))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            close = asyncio.create_task(front.ieee_close_pending('i'))
            await asyncio.sleep(0)
            self.assertFalse(close.done())
            proceed.set()
            self.assertEqual((await close)['state'], 'withdrawn')
            self.assertEqual(journal.snapshot(), [])
            with self.assertRaises(ValueError):
                await front.ieee_register_pending('i', 'text', NS(max_tokens=64), 4)
        asyncio.run(check())

    def test_close_before_registration_and_unclosed_generation(self):
        async def check():
            front, journal = await self.make()
            await front.ieee_close_pending('late')
            with self.assertRaises(ValueError):
                await front.ieee_register_pending('late', 'text', NS(max_tokens=64), 4)
            await front.ieee_register_pending('i', 'text', NS(max_tokens=64), 4)
            front.ieee_pending['i']['state'] = 'generating'
            with self.assertRaisesRegex(RuntimeError, 'possible native ADD'):
                await front.ieee_close_pending('i')
            self.assertEqual(len(journal.snapshot()), 1)
        asyncio.run(check())

    def test_changed_native_prompt_never_reaches_send(self):
        async def check():
            front, journal = await self.make()
            await front.ieee_register_pending('i', 'text', NS(max_tokens=64), 4)
            front.ieee_pending['i']['state'] = 'generating'
            req = NS(external_req_id='i', request_id='native-random', prompt_token_ids=[20, 30],
                     sampling_params=NS(max_tokens=64), lora_request=NS(lora_int_id=4))
            with self.assertRaisesRegex(ValueError, 'admitted prompt'):
                await front._add_request(req, 'text', None, 0, None)
            self.assertEqual(front.sent, [])
            self.assertEqual(len(journal.snapshot()), 1)
        asyncio.run(check())


class ControllerAndRPC(unittest.TestCase):
    def test_engine_bridge_uses_native_preprocessing_and_rejects_unproven_close(self):
        async def check():
            engine = InferenceEngine(dict(backend='vllm', generation_contract='fixed_length_greedy_v1',
                ieee_admission_profile={'test_only': True}), {})
            engine.engine = NS(ieee_register_pending=AsyncMock(return_value={'registered': True}),
                ieee_close_pending=AsyncMock(return_value=dict(kind='ieee_pending_admission_v1',
                    intent_id='wrong', state='withdrawn', clock_id=local_monotonic_clock_id())))
            with patch('scripts.run_all_experiments.SamplingParams', side_effect=lambda **kw: NS(**kw)):
                await engine.ieee_register_pending(intent_id='i', prompt='canonical', max_tokens=64, adapter_id='a')
            args = engine.engine.ieee_register_pending.await_args.args
            self.assertEqual(args[:2], ('i', 'canonical'))
            self.assertEqual((args[2].max_tokens, args[2].temperature, args[2].ignore_eos), (64, 0., True))
            self.assertEqual(args[3], InferenceEngine._lora_int_id('a'))
            with self.assertRaisesRegex(ValueError, 'proof'):
                await engine.ieee_close_pending(intent_id='i')
            engine.engine.ieee_close_pending.return_value['intent_id'] = 'i'
            self.assertTrue((await engine.ieee_close_pending(intent_id='i'))['closed'])
        asyncio.run(check())

    def test_proactive_generation_cannot_skip_pending_registration(self):
        engine = InferenceEngine(dict(backend='vllm', timing_contract='ieee_tc_native_v1',
                                     ieee_admission_profile={'test_only': True}), {})
        with self.assertRaisesRegex(ValueError, 'registered controller demand'):
            asyncio.run(engine.generate('p', None, None, 4, 2))

    def test_actual_request_path_registers_before_host_prepare_and_preserves_generation_id(self):
        runner, slot, trace, plan, owner, observed = lifecycle_tests.SelectedSourceAdmissionIntegration.build(self, 'host')
        runner.model_cfg['ieee_admission_profile'] = {'test_only': True}
        events = []
        ids = []
        async def register(**kw):
            events.append('register')
            ids.append(kw['intent_id'])
            return dict(kind='ieee_pending_admission_v1', intent_id=kw['intent_id'], state='pending',
                        clock_id=local_monotonic_clock_id(), physical_kv_reservation=False)
        async def source(**kw):
            if kw['operation'] == 'hold_host_source':
                self.assertEqual(events, ['register'])
                events.append('protect')
            return await observed(**kw)
        original_generate = slot.engine.generate_prepared.side_effect
        async def generate(**kw):
            self.assertEqual(kw['pending_admission_id'], ids[0])
            events.append('generate')
            return await original_generate(**kw)
        async def close(**kw):
            events.append('close')
            return dict(intent_id=kw['intent_id'], closed=True)
        slot.engine.ieee_register_pending = AsyncMock(side_effect=register)
        slot.engine.ieee_gpu_reference.side_effect = source
        slot.engine.generate_prepared.side_effect = generate
        slot.engine.ieee_close_pending = AsyncMock(side_effect=close)
        result = asyncio.run(runner._exec_request(trace, 4, 0., request_plan=plan))
        self.assertTrue(result.success, result.error)
        self.assertEqual(events, ['register', 'protect', 'generate', 'close'])
        self.assertEqual(owner.snapshot()['live_leases'], 0)
        self.assertEqual(slot.active_requests, 0)
        self.assertEqual(result.gpu_reference_evidence['pending_kv_admission']['state'], 'closed')

    def test_controller_registers_actual_plan_and_closes_before_capacity_return(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        runner.model_cfg['ieee_admission_profile'] = {'test_only': True}
        async def register(**kw):
            self.assertEqual((kw['prompt'], kw['max_tokens'], kw['adapter_id']),
                             (plan.prompt, plan.max_tokens, trace.adapter_id))
            return dict(kind='ieee_pending_admission_v1', intent_id=kw['intent_id'], state='pending',
                        clock_id=local_monotonic_clock_id(), physical_kv_reservation=False)
        async def close(**kw):
            self.assertEqual(slot.active_requests, 1)
            return dict(intent_id=kw['intent_id'], closed=True)
        slot.engine.ieee_register_pending = AsyncMock(side_effect=register)
        slot.engine.ieee_close_pending = AsyncMock(side_effect=close)
        async def check():
            reservation = RuntimeRequestReservation('controller')
            reservation.bind(slot, trace.adapter_id, False)
            slot.active_requests = 1
            await runner._register_ieee_pending_admission(reservation, plan)
            await runner._finish_runtime_request_reservation(reservation)
            self.assertTrue(reservation.released)
            self.assertEqual(reservation.ieee_pending_admission['state'], 'closed')
        asyncio.run(check())
        self.assertEqual(slot.active_requests, 0)

    def test_lost_register_reply_keeps_intent_until_close(self):
        runner, slot, trace, plan, owner, rpc = native_reference_fixture()
        runner.model_cfg['ieee_admission_profile'] = {'test_only': True}
        slot.engine.ieee_register_pending = AsyncMock(side_effect=RuntimeError('lost reply'))
        slot.engine.ieee_close_pending = AsyncMock(side_effect=RuntimeError('unknown owner'))
        async def check():
            r = RuntimeRequestReservation('controller')
            r.bind(slot, trace.adapter_id, False)
            slot.active_requests = 1
            with self.assertRaisesRegex(RuntimeError, 'lost reply'):
                await runner._register_ieee_pending_admission(r, plan)
            with self.assertRaisesRegex(RuntimeError, 'unknown owner'):
                await runner._finish_runtime_request_reservation(r)
            self.assertFalse(r.released)
            self.assertEqual(slot.active_requests, 1)
        asyncio.run(check())

    def test_proxy_only_clears_the_matching_pending_transport(self):
        proxy = SubprocessInferenceEngineProxy.__new__(SubprocessInferenceEngineProxy)
        proxy._rpc = AsyncMock(return_value=dict(intent_id='i', closed=True))
        proxy._native_rpc_uncertain = {
            'one': dict(cmd='ieee_register_pending', intent_id='i'),
            'two': dict(cmd='ieee_register_pending', intent_id='other'),
            'three': dict(cmd='generate', intent_id=None)}
        asyncio.run(proxy.ieee_close_pending(intent_id='i'))
        self.assertEqual(set(proxy._native_rpc_uncertain), {'two', 'three'})


if __name__ == '__main__':
    unittest.main()
