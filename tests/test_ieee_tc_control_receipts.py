"""Pending-receipt metadata retention; synthetic fixtures, no GPU qualification."""
import asyncio
import os
import unittest

from faaslora.scheduling.core_interleave import AtomicControlInterleave
from tests.test_ieee_tc_core_interleave import Kind
from tests.test_ieee_tc_pending_admission import descriptor, utility_call
from tests.test_ieee_tc_pending_gpu_pipeline import (
    NativeCoreTransaction, NativeFrontendTransaction, ControllerSequence)


def install_state(core):
    state=AtomicControlInterleave(type(core),core.scheduler,type(core.model_executor),Kind)
    state.input_turns=7;state.controls=10;state.yields=3;state.backlog_yields=1
    core.scheduler._ieee_control_interleave=state
    return state


class NativeControlReceiptRetention(unittest.TestCase):
    def test_default_off_does_not_add_fields(self):
        fixture=NativeCoreTransaction();core,modules=fixture.make()
        with modules:reply=fixture.invoke(core)
        self.assertNotIn('control_interleave',reply['pending'])
        self.assertNotIn('scheduler_pid',reply['pending'])

    def test_gpu_transaction_retains_same_owner_copy_without_extra_rpc(self):
        fixture=NativeCoreTransaction();core,modules=fixture.make();state=install_state(core)
        expected=state.snapshot()
        with modules:reply=fixture.invoke(core)
        pending=reply['pending']
        self.assertEqual(pending['control_interleave'],expected)
        self.assertEqual(pending['scheduler_pid'],os.getpid())
        self.assertEqual(pending['scheduler_owner_id'],core.scheduler._ieee_iterations.owner_id)
        core.collective_rpc.assert_called_once()
        self.assertEqual(state.snapshot(),expected)
        self.assertNotIn('control_interleave',core.scheduler._ieee_pending_admissions.entries['pending-test'])
        state.controls+=1
        self.assertEqual(pending['control_interleave'],expected)

    def test_non_gpu_registration_no_gpu_call_and_no_withdraw_record(self):
        fixture=NativeCoreTransaction();core,modules=fixture.make();state=install_state(core)
        with modules:
            pending=utility_call(core.ieee_pending_admission,'register',['host-intent',descriptor()])
            withdrawn=utility_call(core.ieee_pending_admission,'withdraw',['host-intent'])
        self.assertEqual(pending['control_interleave'],state.snapshot())
        self.assertNotIn('control_interleave',withdrawn)
        self.assertFalse(core.scheduler._ieee_pending_admissions.snapshot())
        core.collective_rpc.assert_not_called()


class FrontendControlReceiptRetention(unittest.IsolatedAsyncioTestCase):
    async def test_existing_native_frontend_keeps_registration_metadata(self):
        fixture=NativeFrontendTransaction();front,core=fixture.make();state=install_state(core)
        result=await fixture.register(front)
        self.assertEqual(result['pending']['control_interleave'],state.snapshot())
        front.engine_core.call_utility_async.assert_awaited_once()


class ControllerControlReceiptRetention(unittest.TestCase):
    def test_gpu_and_host_results_keep_metadata_without_new_registration(self):
        for tier in ('gpu','host'):
            with self.subTest(tier=tier):
                fixture=ControllerSequence()
                runner,slot,trace,plan,owner,observed=fixture.build(tier)
                original=slot.engine.ieee_register_pending.side_effect
                expected=dict(kind='ieee_native_control_interleave_v1',policy='atomic_control_yield_v1',
                    input_turns=7,handled_controls=10,yields_to_native_loop=3,yields_with_queued_input=1)
                async def register(**kw):
                    value=await original(**kw)
                    pending=value['pending'] if value['kind']=='ieee_pending_gpu_reference_v1' else value
                    pending.update(control_interleave=dict(expected),scheduler_pid=123)
                    return value
                slot.engine.ieee_register_pending.side_effect=register
                result=asyncio.run(runner._exec_request(trace,4,0.,request_plan=plan))
                self.assertTrue(result.success,result.error)
                receipt=result.gpu_reference_evidence['pending_kv_admission']['receipt']
                self.assertEqual(receipt['control_interleave'],expected)
                self.assertEqual(receipt['scheduler_pid'],123)
                slot.engine.ieee_register_pending.assert_awaited_once()
                self.assertEqual(owner.snapshot()['live_leases'],0)


if __name__=='__main__':
    unittest.main()
