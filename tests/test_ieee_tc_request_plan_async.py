import asyncio
import time
import unittest
from types import SimpleNamespace

from scripts.run_all_experiments import RequestExecutionPlan, ScenarioRunner


class RequestPlanPreparationAsyncTests(unittest.IsolatedAsyncioTestCase):
    def _runner(self, *, external=True):
        runner = ScenarioRunner.__new__(ScenarioRunner)
        runner._external_replay = object() if external else None
        runner.model_cfg = {"timing_contract": "ieee_tc_native_v1"}
        runner._request_plan_executor = None
        runner._request_plan_preparation_metrics = {
            "schema": "request_plan_preparation_v1",
            "mode": "pre_replay_sync_v1",
            "count": 0,
            "total_wait_s": 0.0,
            "max_wait_s": 0.0,
            "first_start_monotonic_s": None,
            "last_end_monotonic_s": None,
            "executor_workers": 0,
        }
        return runner

    async def test_native_external_preparation_does_not_block_event_loop(self):
        runner = self._runner()

        def prepare(_engine, _trace, _max_tokens):
            time.sleep(0.08)
            return RequestExecutionPlan(prompt="p", input_tokens=1, max_tokens=1)

        runner._prepare_request_execution_plan = prepare
        trace = SimpleNamespace(request_id="req_1")
        ticks = []

        async def heartbeat():
            started = time.perf_counter()
            while True:
                await asyncio.sleep(0.01)
                ticks.append(time.perf_counter() - started)

        task = asyncio.create_task(heartbeat())
        plan = await runner._prepare_request_execution_plan_after_arrival(
            object(), trace, 1
        )
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(plan.prompt, "p")
        self.assertGreaterEqual(len(ticks), 3)
        self.assertEqual(runner._request_plan_preparation_metrics["count"], 1)
        self.assertEqual(runner._request_plan_preparation_metrics["executor_workers"], 1)
        runner._shutdown_request_plan_executor()

    async def test_preparation_failure_is_not_counted_as_success(self):
        runner = self._runner()

        def prepare(_engine, _trace, _max_tokens):
            raise RuntimeError("fixed contract failure")

        runner._prepare_request_execution_plan = prepare
        with self.assertRaisesRegex(RuntimeError, "fixed contract failure"):
            await runner._prepare_request_execution_plan_after_arrival(
                object(), SimpleNamespace(request_id="req_1"), 1
            )
        self.assertEqual(runner._request_plan_preparation_metrics["count"], 0)
        runner._shutdown_request_plan_executor()

    def test_only_native_external_replay_uses_lazy_mode(self):
        self.assertTrue(self._runner()._request_plan_preparation_is_lazy())
        self.assertFalse(self._runner(external=False)._request_plan_preparation_is_lazy())
        runner = self._runner()
        runner.model_cfg["timing_contract"] = "legacy"
        self.assertFalse(runner._request_plan_preparation_is_lazy())


if __name__ == "__main__":
    unittest.main()
