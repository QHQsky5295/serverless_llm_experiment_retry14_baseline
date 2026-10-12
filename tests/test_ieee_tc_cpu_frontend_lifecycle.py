"""D468 CPU state/IPC fixtures, NOT native CUDA or performance evidence."""
import asyncio
import copy
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

from faaslora.cpu_frontend import (
    CONTRACT, MAX_ACTIVATION_BYTES, CPUFrontendPool, PreparedCPUFrontend,
    activation_message, configuration_sha256, receive_activation, validate_preparation)

UUID = 'GPU-01234567-0123-4567-89ab-0123456789ab'
BIRTH = dict(pid=20, start_ticks=2, cgroup='/service/worker')
OWNER = dict(pid=os.getpid(), start_ticks=1, cgroup='/service')


def evidence():
    return dict(contract=CONTRACT, preparation_id='a'*32, gpu_uuid=UUID,
        configuration_sha256='b'*64, deployment_notice_s=0., process_started_s=.1,
        birth=dict(BIRTH), owner=dict(OWNER))


def allocation(root, process=None):
    lease = 'c'*32
    prep = evidence()
    common = dict(lease_id=lease, owner_id=lease, clock_id='clock', gpu_uuids=[UUID])
    events = [dict(common, event='acquire', at=1., owner_identity=OWNER),
              dict(common, event='prepared_worker_bind', at=1.1, worker=BIRTH, preparation=prep)]
    journal = root/'physical_allocations'/f'{lease}.jsonl'
    journal.parent.mkdir(exist_ok=True)
    journal.write_text(''.join(json.dumps(r)+'\n' for r in events))
    return NS(events=events, owner_id=lease, journal=journal, gpu_uuids=[UUID],
              owner_identity=OWNER, process=process or NS(pid=20), released=False)


class PreparationIdentity(unittest.TestCase):
    def check(self, value):
        return validate_preparation(value, birth=BIRTH, owner=OWNER,
                                    gpu_uuids=[UUID], acquire_at=1.)

    def test_configuration_identity_is_order_independent_but_not_value_independent(self):
        first = configuration_sha256({'a': 1, 'b': 2}, {'p': 3})
        self.assertEqual(first, configuration_sha256({'b': 2, 'a': 1}, {'p': 3}))
        self.assertNotEqual(first, configuration_sha256({'a': 2, 'b': 2}, {'p': 3}))
        with self.assertRaises(ValueError):
            configuration_sha256({'bad': float('nan')}, {})

    def test_preparation_before_acquisition_preserves_two_separate_times(self):
        self.assertEqual(self.check(evidence())['process_started_s'], .1)

    def test_foreign_identity_device_and_schema_fail(self):
        for key, value in [('contract', 'ready'), ('gpu_uuid', 'GPU-9'),
                ('preparation_id', 'wrong'), ('configuration_sha256', 1),
                ('birth', dict(BIRTH, start_ticks=3)), ('owner', dict(OWNER, pid=99))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(dict(evidence(), **{key: value}))
        with self.assertRaises(ValueError):
            self.check(dict(evidence(), gpu_ready=True))

    def test_no_pre_notice_preparation_future_birth_or_nonfinite_time(self):
        for updates in [dict(deployment_notice_s=.2), dict(process_started_s=2.),
                        dict(process_started_s=float('nan')), dict(deployment_notice_s=True)]:
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.check(dict(evidence(), **updates))

    def test_device_visibility_and_native_imports_absent_from_pool(self):
        from faaslora import cpu_frontend
        source = inspect.getsource(cpu_frontend)
        for forbidden in ('import torch', 'import vllm', 'CUDA_VISIBLE_DEVICES=',
                          'engine.initialize(', 'engine.shutdown(', 'os.fork('):
            self.assertNotIn(forbidden, source)

    def test_public_import_does_not_pull_in_the_experiment_serving_stack(self):
        root=str(Path(__file__).resolve().parents[1])
        command=('import sys; sys.path.insert(0, '+repr(root)+'); '
                 'import faaslora.cpu_frontend; '
                 'assert not ({"torch", "vllm", "numpy", "faaslora.experiment"} & set(sys.modules))')
        result=subprocess.run(['/usr/bin/python3','-I','-B','-c',command],
                              capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)


class ActivationIPC(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.owner = allocation(self.root)
        self.env = patch.dict(os.environ, FAASLORA_TC_LAUNCH_RECEIPT=str(self.root/'launch.json'))
        self.env.start()
        self.addCleanup(self.env.stop)

    async def receive(self, frame):
        parent, child = socket.socketpair()
        try:
            if frame:
                parent.sendall(frame)
            parent.close()
            return await receive_activation(child, expected=evidence(), birth=BIRTH, clock_id='clock')
        finally:
            parent.close()
            child.close()

    def rewrite(self, rows):
        self.owner.journal.write_text(''.join(json.dumps(row)+'\n' for row in rows))

    async def test_exact_binding_unlocks_activation(self):
        frame = activation_message(evidence(), self.owner)
        result = await self.receive(frame)
        self.assertEqual(result['lease_id'], self.owner.owner_id)

    async def test_eof_without_command_cancels(self):
        self.assertIsNone(await self.receive(b''))

    async def test_partial_oversized_duplicate_command_fail(self):
        frame = activation_message(evidence(), self.owner)
        for invalid in (b'{', b'x'*(MAX_ACTIVATION_BYTES+1), frame+frame):
            with self.subTest(length=len(invalid)), self.assertRaises(ValueError):
                await self.receive(invalid)

    async def test_foreign_packet_binding_uuid_config_birth_rejected(self):
        frame = json.loads(activation_message(evidence(), self.owner))
        for key, value in [('preparation_id','f'*32), ('gpu_uuid','GPU-8'),
                ('configuration_sha256','f'*64), ('birth',dict(BIRTH,start_ticks=7)),
                ('lease_id', 7), ('journal', str(self.root/'foreign.jsonl'))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                await self.receive(json.dumps(dict(frame, **{key:value})).encode())

    async def test_released_missing_or_foreign_journal_is_not_permission(self):
        frame = activation_message(evidence(), self.owner)
        original = copy.deepcopy(self.owner.events)
        rows = original+[dict(original[-1], event='release')]
        for invalid in (rows, original[:1], [dict(original[0], clock_id='foreign'), original[1]],
                        [original[0],dict(original[1],at=.5)]):
            self.rewrite(invalid)
            with self.assertRaises(ValueError):
                await self.receive(frame)

    async def test_socket_wait_is_cancellable_without_activation(self):
        parent, child = socket.socketpair()
        try:
            task = asyncio.create_task(receive_activation(child, expected=evidence(),
                                                         birth=BIRTH, clock_id='clock'))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(child.fileno(), -1)
        finally:
            parent.close()
            child.close()

    def test_send_requires_already_acquired_bound_unreleased_lease(self):
        for field, value in [('released',True), ('process',None), ('events',self.owner.events[:1])]:
            old = getattr(self.owner, field)
            setattr(self.owner, field, value)
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                activation_message(evidence(), self.owner)
            setattr(self.owner, field, old)


class TicketOwnership(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.process = NS(pid=20, poll=Mock(return_value=None), wait=Mock())
        self.channel = Mock()
        self.identity = Mock(return_value=dict(BIRTH))
        self.ticket = PreparedCPUFrontend(process=self.process, channel=self.channel,
            evidence=evidence(), identity=self.identity)
        self.owner = allocation(Path(self.tmp.name), self.process)

    def test_activation_once_before_cpu_ready_wait(self):
        self.ticket.activate(self.owner)
        self.channel.sendall.assert_called_once()
        self.channel.close.assert_called_once()
        self.assertEqual(self.ticket.state, 'activation_sent')
        with self.assertRaises(RuntimeError):
            self.ticket.activate(self.owner)

    def test_dead_or_reused_process_cannot_activate(self):
        self.process.poll.return_value = 1
        with self.assertRaises(RuntimeError):
            self.ticket.activate(self.owner)
        self.process.poll.return_value = None
        self.identity.return_value = dict(BIRTH, start_ticks=99)
        with self.assertRaises(RuntimeError):
            self.ticket.activate(self.owner)
        self.channel.sendall.assert_not_called()

    def test_partial_send_never_allows_a_retry(self):
        self.channel.sendall.side_effect = BrokenPipeError('child exited')
        with self.assertRaises(BrokenPipeError):
            self.ticket.activate(self.owner)
        with self.assertRaises(RuntimeError):
            self.ticket.activate(self.owner)
        self.assertEqual(self.ticket.state, 'activation_sent')

    async def test_abandon_before_activation_closes_permission_then_joins(self):
        await self.ticket.abandon()
        self.channel.close.assert_called_once()
        self.process.wait.assert_called_once_with(5)
        self.channel.sendall.assert_not_called()
        self.assertEqual(self.ticket.state,'abandoned')

    async def test_abandon_after_activation_must_use_physical_owner(self):
        self.ticket.activate(self.owner)
        with self.assertRaises(RuntimeError):
            await self.ticket.abandon()
        self.process.wait.assert_not_called()

    async def test_changed_birth_never_signalled(self):
        self.identity.return_value = None
        with patch('os.killpg') as kill:
            with self.assertRaises(RuntimeError):
                await self.ticket.abandon()
            kill.assert_not_called()


class PoolOwnership(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.created=[]
        def create(device, notice):
            ticket = NS(device=device, notice=notice, abandon=AsyncMock(),
                        process=NS(poll=Mock(return_value=0)))
            self.created.append(ticket)
            return ticket
        self.create = Mock(side_effect=create)
        self.pool = CPUFrontendPool(devices=[0,1], create=self.create)

    async def test_bound_number_duplicate_claim_and_exact_replacement(self):
        self.pool.start(deployment_notice_s=0.)
        first = self.pool.claim(0)
        self.assertEqual(len(self.pool.prepared)+len(self.pool.claimed),2)
        with self.assertRaises(RuntimeError):self.pool.claim(0)
        with self.assertRaises(RuntimeError):self.pool.claim(2)
        self.pool.returned(0,first,allocation=NS(released=True,process=first.process))
        self.assertEqual(len(self.created),3)
        self.assertEqual(len(self.pool.prepared)+len(self.pool.claimed),2)
        await self.pool.close()

    async def test_replacement_requires_confirmed_release_and_exit(self):
        self.pool.start(deployment_notice_s=0.)
        ticket=self.pool.claim(0)
        for allocation_ in (NS(released=False,process=ticket.process),NS(released=True,process=object())):
            with self.assertRaises(RuntimeError):self.pool.returned(0,ticket,allocation=allocation_)
        ticket.process.poll.return_value=None
        with self.assertRaises(RuntimeError):
            self.pool.returned(0,ticket,allocation=NS(released=True,process=ticket.process))
        self.assertEqual(len(self.created),2)
        await self.pool.close()

    async def test_shutdown_suppresses_replacement_and_never_retires_claimed_owner(self):
        self.pool.start(deployment_notice_s=0.)
        ticket=self.pool.claim(0)
        await self.pool.close()
        ticket.abandon.assert_not_awaited()
        self.pool.returned(0,ticket,allocation=NS(released=True,process=ticket.process))
        self.assertEqual(len(self.created),2)
        self.assertFalse(self.pool.claimed)
        with self.assertRaises(RuntimeError):self.pool.claim(1)

    async def test_partial_start_failure_retains_then_closes_created_objects(self):
        self.create.side_effect=[NS(abandon=AsyncMock()),RuntimeError('launch failed')]
        with self.assertRaises(RuntimeError):self.pool.start(deployment_notice_s=0.)
        survivor=self.pool.prepared[0]
        with self.assertRaises(RuntimeError):self.pool.claim(0)
        await self.pool.close()
        survivor.abandon.assert_awaited_once()

    async def test_cancelled_close_joins_owned_cleanup_before_raising(self):
        self.pool.start(deployment_notice_s=0.)
        reached, finish = asyncio.Event(),asyncio.Event()
        async def abandon():
            reached.set()
            await finish.wait()
        self.created[0].abandon.side_effect=abandon
        close=asyncio.create_task(self.pool.close())
        await reached.wait()
        close.cancel()
        await asyncio.sleep(0)
        self.assertFalse(close.done())
        finish.set()
        with self.assertRaises(asyncio.CancelledError):await close
        self.assertFalse(self.pool.prepared)
        await self.pool.close()  # Idempotent; no second abandon.
        self.created[0].abandon.assert_awaited_once()

    async def test_failed_close_keeps_unresolved_ticket(self):
        self.pool.start(deployment_notice_s=0.)
        self.created[0].abandon.side_effect=RuntimeError('identity lost')
        with self.assertRaises(RuntimeError):await self.pool.close()
        self.assertEqual(list(self.pool.prepared),[0])

    def test_invalid_devices_pre_notice_and_restarts_rejected(self):
        for devices in ([],[0,0],[-1],[True],['0']):
            with self.assertRaises(ValueError):CPUFrontendPool(devices=devices,create=self.create)
        with self.assertRaises(RuntimeError):self.pool.start(deployment_notice_s=time.monotonic()+60)
        with self.assertRaises(RuntimeError):self.pool.claim(0)


class DedicatedWorkerBoundary(unittest.IsolatedAsyncioTestCase):
    """Actual worker control flow with a mocked engine, never a native result."""
    def setUp(self):
        from scripts import dedicated_engine_worker as worker
        from scripts import run_all_experiments as runner
        self.worker, self.runner = worker,runner
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.payload=self.root/'payload.json'
        self.ready=self.root/'ready.json'
        self.engine=NS(model_cfg=dict(backend='vllm'),engine=None,backend='vllm',
                       initialize=AsyncMock(),shutdown=AsyncMock())
        self.cfg=dict(repo_root=str(self.root),model_cfg=dict(backend='vllm',tensor_parallel_size=1),
                      cost_model={},cpu_preparation=evidence())
        self.payload.write_text(json.dumps(self.cfg))

    async def exercise(self, permission=None, gate_error=None, init_error=None):
        self.engine.initialize.side_effect=init_error
        with patch.object(self.runner,'InferenceEngine',return_value=self.engine), \
                patch.object(self.runner,'_build_local_tp_runtime_env_updates',return_value={}), \
                patch.object(self.worker,'_await_cpu_frontend_activation',
                             new=AsyncMock(return_value=permission,side_effect=gate_error)) as gate:
            await self.worker._run_worker(self.payload,self.ready,activation_channel=object())
            gate.assert_awaited_once()

    async def test_cancel_without_permission_never_initializes_or_shuts_down_cuda(self):
        await self.exercise()
        self.engine.initialize.assert_not_awaited()
        self.engine.shutdown.assert_not_awaited()
        self.assertFalse(self.ready.exists())

    async def test_rejected_gate_never_initializes_or_shuts_down_cuda(self):
        with self.assertRaisesRegex(ValueError,'foreign binding'):
            await self.exercise(gate_error=ValueError('foreign binding'))
        self.engine.initialize.assert_not_awaited()
        self.engine.shutdown.assert_not_awaited()

    async def test_permitted_initialize_failure_uses_existing_shutdown(self):
        with self.assertRaisesRegex(RuntimeError,'native init error'):
            await self.exercise(permission={'lease_id':'a'*32},init_error=RuntimeError('native init error'))
        self.engine.initialize.assert_awaited_once()
        self.engine.shutdown.assert_awaited_once()

    async def test_payload_without_descriptor_capability_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'inherited activation'):
            await self.worker._run_worker(self.payload,self.ready)
        del self.cfg['cpu_preparation']
        self.payload.write_text(json.dumps(self.cfg))
        with self.assertRaisesRegex(ValueError,'inherited activation'):
            await self.worker._run_worker(self.payload,self.ready,activation_channel=object())


if __name__ == '__main__':unittest.main()
