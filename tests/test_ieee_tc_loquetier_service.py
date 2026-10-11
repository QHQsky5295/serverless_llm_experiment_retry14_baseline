"""CPU transport/ownership tests; not model arithmetic or performance evidence."""
import asyncio
import os
from pathlib import Path
import sys
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import replay_vllm_engine_trace as service


class StaticRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_wait_for_exact_static_shard_without_ready_reroute(self):
        router = service.LoquetierStaticRouter({'a': 0, 'b': 1}, 2, 'clock')
        def body(rid, aid):
            return dict(request_id=rid, lora_adapter_name=aid,
                        _sllm_internal_metrics=dict(tc_clock_id='clock'))
        router.ready[1].set()
        waiting = asyncio.create_task(router.route(body('first', 'a'), time.perf_counter()))
        await asyncio.sleep(0)
        index, result = await router.route(body('second', 'b'), time.perf_counter())
        self.assertEqual(index, 1)
        self.assertEqual(result['_sllm_internal_metrics']['instance_id'], 'loquetier-1')
        self.assertFalse(waiting.done())
        router.ready[0].set()
        self.assertEqual((await waiting)[0], 0)
        self.assertEqual((await router.route(body('third', 'a'), time.perf_counter()))[0], 0)
        with self.assertRaises(ValueError):
            await router.route(body('first', 'a'), time.perf_counter())
        with self.assertRaises(ValueError):
            await router.route(body('missing', 'unknown'), time.perf_counter())
        self.assertNotIn('missing', router.seen)

    async def test_cancelled_wait_does_not_create_replacement_or_retry(self):
        router = service.LoquetierStaticRouter({'a': 0}, 1, 'clock')
        body = dict(request_id='r', lora_adapter_name='a',
                    _sllm_internal_metrics=dict(tc_clock_id='clock'))
        task = asyncio.create_task(router.route(body, time.perf_counter()))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        router.ready[0].set()
        with self.assertRaises(ValueError):
            await router.route(body, time.perf_counter())

    def test_invalid_static_mapping_and_fixed_input_rejected(self):
        for mapping, count in (({}, 1), ({'a': 0}, 2), ({'a': True}, 1), ({'a': 2}, 2)):
            with self.assertRaises(ValueError):
                service.LoquetierStaticRouter(mapping, count, 'clock')
        for rid, aid, ids, target in (('', 'a', [1], 1), ('r', 'x', [1], 1),
                ('r', 'a', [], 1), ('r', 'a', [True], 1), ('r', 'a', [1], True),
                ('r', 'a', [1], 257), ('r', 'a', [1]*1000, 25)):
            with self.assertRaises(ValueError):
                service.validate_loquetier_input(rid, aid, ids, target, {'a'})


class FakeBridge:
    """Explicit synthetic bridge, with a blocked iteration for race tests."""
    def __init__(self, test, batch=2):
        self.test, self.batch_size = test, batch
        self.thread, self.adapters = threading.get_ident(), frozenset(('a', 'b'))
        self.active, self.admitted, self.cancelled = {}, [], []
        self.next_mark, self.peak = 0, 0
        self.step_started, self.allow_step = threading.Event(), threading.Event()
        self.closed, self.raise_step = False, False

    def owner(self):
        self.test.assertEqual(threading.get_ident(), self.thread)

    def add(self, rid, aid, ids, target):
        self.owner()
        self.test.assertLess(len(self.active), self.batch_size)
        mark = self.next_mark
        self.next_mark += 1
        self.admitted.append(rid)
        self.active[mark] = [rid, target, 0]
        self.peak = max(self.peak, len(self.active))
        return mark

    def step(self):
        self.owner()
        self.step_started.set()
        if not self.allow_step.wait(3):
            raise RuntimeError('test did not release synthetic iteration')
        if self.raise_step:
            raise RuntimeError('synthetic native failure')
        results = {}
        for mark, row in list(self.active.items()):
            row[2] += 1
            if row[2] == row[1]:
                results[mark] = dict(request_id=row[0], native_output_tokens=row[2])
                del self.active[mark]
        return results

    def cancel(self, mark):
        self.owner()
        rid, target, count = self.active.pop(mark)
        self.cancelled.append(rid)
        return dict(event='loquetier_request_cancelled', request_id=rid,
            native_registered=True, native_output_tokens=count, native_terminal_observed=False)

    def close(self):
        self.owner()
        self.closed = True
        return [self.cancel(mark) for mark in list(self.active)]


class OnlineQueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.events = []
        self.queue = service.LoquetierOnlineRequests(lambda: FakeBridge(self), self.events.append)
        self.bridge = await self.queue.start()

    async def asyncTearDown(self):
        self.bridge.allow_step.set()
        await asyncio.wait_for(self.queue.close(), 4)
        self.assertTrue(self.bridge.closed)
        self.assertEqual(self.bridge.active, {})

    async def entered_iteration(self):
        deadline = time.monotonic()+2
        while not self.bridge.step_started.is_set():
            if time.monotonic() >= deadline:
                self.fail('synthetic iteration did not start')
            await asyncio.sleep(.001)

    def submit(self, rid, target=3):
        return asyncio.create_task(self.queue.submit(rid, 'a', [1, 2], target))

    async def test_online_arrival_continues_during_native_iteration_and_fifo_bound(self):
        first = self.submit('r0')
        await self.entered_iteration()
        rest = [self.submit('r'+str(i), target) for i, target in ((1, 1), (2, 2), (3, 256))]
        await asyncio.sleep(0)
        self.assertEqual(len(self.queue.pending), 3)  # Loop not blocked by native work.
        self.assertNotEqual(self.bridge.thread, threading.get_ident())
        self.bridge.allow_step.set()
        results = await asyncio.wait_for(asyncio.gather(first, *rest), 4)
        self.assertEqual([r['native_output_tokens'] for r in results], [3, 1, 2, 256])
        self.assertEqual(self.bridge.admitted, ['r0', 'r1', 'r2', 'r3'])
        self.assertLessEqual(self.bridge.peak, 2)
        self.assertEqual(len([e for e in self.events if e['event']=='loquetier_native_completed']), 4)

    async def test_pending_cancel_never_registers_native_and_duplicate_rejected(self):
        first = self.submit('first', 3)
        await self.entered_iteration()
        cancelled = self.submit('cancelled')
        await asyncio.sleep(0)
        cancelled.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled
        with self.assertRaises(ValueError):
            await self.queue.submit('cancelled', 'a', [1], 2)
        self.bridge.allow_step.set()
        await first
        self.assertEqual(self.bridge.admitted, ['first'])
        rows = [e for e in self.events if e['event']=='loquetier_request_cancelled']
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]['native_registered'])

    async def test_active_cancel_at_iteration_boundary_does_not_cancel_other_requests(self):
        cancelled = self.submit('cancelled', 5)
        await self.entered_iteration()
        survivor = self.submit('survivor', 3)
        cancelled.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled
        self.bridge.allow_step.set()
        result = await survivor
        self.assertEqual(result['native_output_tokens'], 3)
        self.assertEqual(self.bridge.cancelled, ['cancelled'])
        rows = [e for e in self.events if e['event']=='loquetier_request_cancelled']
        self.assertEqual(rows[0]['native_output_tokens'], 1)

    async def test_cancel_racing_native_completion_keeps_completed_evidence_not_success(self):
        task = self.submit('r', 1)
        await self.entered_iteration()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.bridge.allow_step.set()
        await self.queue.close()
        rows = [e for e in self.events if e['event']=='loquetier_native_completed']
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]['client_cancelled'])
        self.assertEqual(self.bridge.cancelled, [])

    async def test_native_failure_terminates_all_offered_work_without_fallback(self):
        self.bridge.raise_step = True
        first = self.submit('first')
        await self.entered_iteration()
        second = self.submit('second')
        await asyncio.sleep(0)
        self.bridge.allow_step.set()
        results = await asyncio.gather(first, second, return_exceptions=True)
        self.assertTrue(all(isinstance(r, RuntimeError) for r in results))
        self.assertIsNotNone(self.queue.failure)
        with self.assertRaises(RuntimeError):
            await self.queue.submit('third', 'a', [1], 2)
        self.assertEqual(self.bridge.admitted, ['first'])
        self.assertEqual(self.bridge.cancelled, ['first'])
        self.assertFalse(any(e['event']=='loquetier_native_completed' for e in self.events))

    async def test_close_waits_for_native_boundary_then_fails_remaining_work(self):
        first = self.submit('first', 3)
        await self.entered_iteration()
        closing = asyncio.create_task(self.queue.close())
        await asyncio.sleep(0)
        self.assertFalse(closing.done())
        self.assertFalse(self.bridge.closed)
        self.bridge.allow_step.set()
        await closing
        with self.assertRaises(RuntimeError):
            await first
        self.assertTrue(self.bridge.closed)
        self.assertEqual(self.bridge.cancelled, ['first'])

    async def test_native_registration_failure_is_terminal_and_journaled(self):
        def fail_add(*args):
            self.bridge.owner()
            raise RuntimeError('synthetic allocation failure')
        self.bridge.add = fail_add
        first, second = self.submit('first'), self.submit('second')
        results = await asyncio.gather(first, second, return_exceptions=True)
        self.assertTrue(all(isinstance(result, RuntimeError) for result in results))
        await self.queue.close()
        self.assertIsNotNone(self.queue.failure)
        failures = [e for e in self.events if e['event']=='loquetier_registration_failed']
        self.assertEqual([e['request_id'] for e in failures], ['first'])
        self.assertFalse(any(e['event']=='loquetier_native_completed' for e in self.events))
        self.assertEqual(self.bridge.admitted, [])

    async def test_factory_failure_cannot_leave_available_or_restartable_service(self):
        def fail_factory():
            raise RuntimeError('synthetic initialization failure')
        queue = service.LoquetierOnlineRequests(fail_factory, self.events.append)
        with self.assertRaisesRegex(RuntimeError, 'initialization failure'):
            await queue.start()
        self.assertTrue(queue.closing)
        with self.assertRaises(RuntimeError):
            await queue.submit('r', 'a', [1], 1)
        with self.assertRaises(ValueError):
            await queue.start()
        await queue.close()


if __name__ == '__main__':
    unittest.main()
