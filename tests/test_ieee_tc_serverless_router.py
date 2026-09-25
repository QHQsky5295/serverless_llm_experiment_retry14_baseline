"""Execute the real router loop AST without loading Ray, models or GPUs.

The virtual clock counts mandatory waits, not wall-clock performance. Official
source and the patched local loop are both exercised; exact sources are hashed
by the audit artifact, and a full model replay is still required for attribution.
"""
import ast
import asyncio
import contextlib
import logging
from pathlib import Path
import subprocess
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'vendor_new_baselines/ServerlessLLM_new_main_20260518'
ROUTER = 'sllm/routers/roundrobin_router.py'
SHA = '9f50241baa5386e06a9321c51f19a9ef5f964c2b'


class Instance:
    def __init__(self):
        self.concurrency = 0
        self.capacity = 100
        self.on_add = None

    async def check_request_queue(self):
        return self.concurrency < self.capacity

    async def add_requests(self, n):
        self.concurrency += n
        if n > 0 and self.on_add:
            await self.on_add()
        return True


class RouterContract(unittest.IsolatedAsyncioTestCase):
    def make_router(self, source=None, on_sleep=None):
        source = source or (SOURCE/ROUTER).read_text()
        tree = ast.parse(source)
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'RoundRobinRouter')
        fn = next(x for x in cls.body if isinstance(x, ast.AsyncFunctionDef) and x.name == '_load_balancer_loop')
        module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
        sleeps = []
        async def sleep(seconds):
            sleeps.append(seconds)
            if on_sleep:
                on_sleep()
            await asyncio.sleep(0)
        ns = {'asyncio': types.SimpleNamespace(sleep=sleep), 'logger': logging.getLogger('test')}
        exec(compile(module, str(SOURCE/ROUTER), 'exec'), ns)
        obj = types.SimpleNamespace(request_queue=asyncio.Queue(), model_name='test',
            instance_management_lock=asyncio.Lock(), ready_inference_instances={}, loop_interval=1)
        task = asyncio.create_task(ns['_load_balancer_loop'](obj))
        self.tasks.append(task)
        return obj, sleeps, task

    async def asyncSetUp(self):
        self.tasks = []

    async def asyncTearDown(self):
        for task in self.tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def submit(self, obj, n):
        futures = [asyncio.get_running_loop().create_future() for _ in range(n)]
        for future in futures:
            obj.request_queue.put_nowait(future)
        return await asyncio.wait_for(asyncio.gather(*futures), timeout=1)

    async def test_official_ready_path_waits_once_per_request(self):
        source = subprocess.check_output(['git', '-C', str(SOURCE), 'show', f'{SHA}:{ROUTER}'], text=True)
        obj, sleeps, _ = self.make_router(source)
        obj.ready_inference_instances = {'a': Instance(), 'b': Instance()}
        self.assertEqual(await self.submit(obj, 4), ['a', 'b', 'a', 'b'])
        self.assertEqual(sleeps, [1, 1, 1, 1])

    async def test_ready_path_no_mandatory_wait_and_same_rr(self):
        obj, sleeps, _ = self.make_router()
        obj.ready_inference_instances = {'a': Instance(), 'b': Instance()}
        self.assertEqual(await self.submit(obj, 4), ['a', 'b', 'a', 'b'])
        self.assertEqual(sleeps, [])

    async def test_empty_waits_and_rechecks(self):
        obj, sleeps, _ = self.make_router(on_sleep=lambda: obj.ready_inference_instances.update(a=Instance()))
        self.assertEqual(await self.submit(obj, 1), ['a'])
        self.assertEqual(sleeps, [1])

    async def test_full_waits_and_keeps_original_rr_progression(self):
        obj, sleeps, _ = self.make_router()
        a, b = Instance(), Instance()
        a.capacity = 0
        obj.ready_inference_instances = {'a': a, 'b': b}
        self.assertEqual(await self.submit(obj, 1), ['b'])
        self.assertEqual(sleeps, [1])
        self.assertEqual(a.concurrency, 0)
        self.assertEqual(b.concurrency, 1)

    async def test_cancelled_queue_item_does_not_kill_router(self):
        obj, sleeps, task = self.make_router()
        obj.ready_inference_instances = {'a': Instance()}
        cancelled = asyncio.get_running_loop().create_future()
        cancelled.cancel()
        obj.request_queue.put_nowait(cancelled)
        self.assertEqual(await self.submit(obj, 1), ['a'])
        self.assertFalse(task.done())
        self.assertEqual(obj.ready_inference_instances['a'].concurrency, 1)

    async def test_cancel_during_capacity_reservation_rolls_back(self):
        obj, sleeps, task = self.make_router()
        inst = Instance()
        obj.ready_inference_instances = {'a': inst}
        future = asyncio.get_running_loop().create_future()
        async def cancel():
            future.cancel()
            inst.on_add = None
            await asyncio.sleep(0)
        inst.on_add = cancel
        obj.request_queue.put_nowait(future)
        self.assertEqual(await self.submit(obj, 1), ['a'])
        self.assertEqual(inst.concurrency, 1)
        self.assertFalse(task.done())


if __name__ == '__main__':
    unittest.main()
