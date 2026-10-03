"""Read-only source RPC boundary qualification; CPU fixtures, not performance."""
import asyncio
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import uuid

from faaslora.clock import local_monotonic_clock_id
from faaslora.utils import logger
from faaslora.memory import gpu_monitor
from scripts import dedicated_engine_worker as worker
from scripts.run_all_experiments import InferenceEngine, SubprocessInferenceEngineProxy
from tests import test_ieee_tc_gpu_references as references
from tests import test_ieee_tc_stack_diagnostic as stacks
from tests.test_ieee_tc_request_footprints import scoped_payload


def observer(output, pid=None):
    # Unit fixtures only. Real launch qualification is separately exercised below.
    return patch.object(logger, '_diagnostic_stack_state',
                        (os.getpid() if pid is None else pid, None, None, None, output))


def events(output):
    return [json.loads(row) for row in output.getvalue().splitlines()]


class ControlEventContract(unittest.TestCase):
    def test_absent_id_has_no_output_and_nonqualified_id_rejects(self):
        with patch.object(logger, '_diagnostic_stack_state', None):
            self.assertFalse(logger.diagnostic_control_enabled())
            logger.diagnostic_control_event(None, 'parent_begin')
            with self.assertRaisesRegex(ValueError, 'qualified'):
                logger.diagnostic_control_event(uuid.uuid4().hex, 'parent_begin')

    def test_fork_inherited_observer_is_not_qualified(self):
        with observer(io.BytesIO(), pid=os.getpid()+1):
            self.assertFalse(logger.diagnostic_control_enabled())
            with self.assertRaisesRegex(ValueError, 'qualified'):
                logger.diagnostic_control_event(uuid.uuid4().hex, 'parent_begin')

    def test_bad_identity_boundary_bytes_outcome_do_not_write(self):
        out, aid = io.BytesIO(), uuid.uuid4().hex
        with observer(out):
            for args, kwargs in [((True, 'parent_begin'), {}), ((aid.upper(), 'parent_begin'), {}),
                    ((aid, 'unknown'), {}), ((aid, 'parent_send'), {'byte_count': True}),
                    ((aid, 'parent_send'), {'byte_count': -1}),
                    ((aid, 'native_ready'), {'byte_count': 1}),
                    ((aid, 'parent_terminal'), {}), ((aid, 'native_ready'), {'outcome': 'success'})]:
                with self.subTest(args=args, kwargs=kwargs), self.assertRaises(ValueError):
                    logger.diagnostic_control_event(*args, **kwargs)
        self.assertEqual(out.getvalue(), b'')

    def test_short_write_is_failure_not_silent_success(self):
        with observer(NS(write=lambda _: 0)):
            with self.assertRaisesRegex(OSError, 'incomplete'):
                logger.diagnostic_control_event(uuid.uuid4().hex, 'parent_begin')

    def test_actual_gated_processes_write_same_clock_and_survive_abrupt_exit(self):
        aid = uuid.uuid4().hex
        code = f'''import faaslora, os
from faaslora.utils.logger import diagnostic_control_event, diagnostic_control_enabled
assert diagnostic_control_enabled()
diagnostic_control_event({aid!r}, 'native_begin')
diagnostic_control_event({aid!r}, 'native_ready')
os._exit(23)
'''
        with tempfile.TemporaryDirectory() as root:
            for _ in range(2):
                r = stacks.StackDiagnostic().invoke(root, code)
                self.assertEqual(r.returncode, 23, r.stderr)
            paths = sorted((Path(root)/'diagnostic_stacks').glob('*.control.jsonl'))
            self.assertEqual(len(paths), 2)
            rows = [json.loads(line) for p in paths for line in p.read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(len({r['pid'] for r in rows}), 2)
            self.assertEqual({r['clock_id'] for r in rows}, {local_monotonic_clock_id()})
            self.assertEqual({r['attempt_id'] for r in rows}, {aid})
            self.assertTrue(all(r['monotonic_s'] > 0 and r['thread_cpu_s'] >= 0 for r in rows))
            self.assertTrue(all(set(r) == {'event', 'attempt_id', 'boundary', 'pid', 'thread_ident',
                'clock_id', 'monotonic_s', 'thread_cpu_s', 'byte_count', 'outcome'} for r in rows))

    def test_native_owner_observation_unchanged_and_failures_remain_incomplete(self):
        case = references.NativeDemandTransactions()
        case.setUp()
        case.demand()
        native = gpu_monitor.IEEEWorkerObservationExtension()
        native.device, native.rank = NS(type='cuda'), 0
        native.model_runner = NS(lora_manager=NS(_adapter_manager=case.manager))
        native._ieee_gpu_reference_owner = case.owner
        native._ieee_host_allocator_policy = {'verified': False}
        torch = NS(cuda=NS(get_device_properties=lambda _: NS(uuid=NS(bytes=list(range(16))))))
        out = io.BytesIO()
        with observer(out), patch.object(gpu_monitor, 'torch', torch), \
             patch.object(gpu_monitor, '_ieee_lora_host_inventory', return_value={'requested_host_storage_bytes':16}) as host, \
             patch.object(gpu_monitor, '_ieee_lora_pool_inventory', return_value={'slot_capacity_bytes':32}) as pool:
            original = native.ieee_gpu_reference(operation='request_source_snapshot', requested_adapter_ids=[4])
            measured = native.ieee_gpu_reference(operation='request_source_snapshot', requested_adapter_ids=[4],
                _diagnostic_control_id=uuid.uuid4().hex)
            self.assertEqual({k:v for k,v in original.items() if k != 'captured_monotonic_s'},
                             {k:v for k,v in measured.items() if k != 'captured_monotonic_s'})
            self.assertEqual(host.call_count, 2)
            self.assertEqual(pool.call_count, 2)
            self.assertEqual([r['boundary'] for r in events(out)], ['native_begin', 'native_ready'])
            case.manager._registered_adapters[4] = object()
            with self.assertRaisesRegex(RuntimeError, 'source object'):
                native.ieee_gpu_reference(operation='request_source_snapshot', requested_adapter_ids=[4],
                    _diagnostic_control_id=uuid.uuid4().hex)
            self.assertEqual([r['boundary'] for r in events(out)][-1], 'native_begin')
            self.assertEqual(host.call_count, 2)  # Existing failure still precedes inventory.
            with self.assertRaisesRegex(ValueError, 'read-only'):
                native.ieee_gpu_reference(operation='release', _diagnostic_control_id=uuid.uuid4().hex)


class ControlTransport(unittest.IsolatedAsyncioTestCase):
    async def roundtrip(self, mode='success', enabled=True):
        ready = asyncio.get_running_loop().create_future()
        entered, proceed, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls, out = [], io.BytesIO()
        payload = scoped_payload()
        payload['clock_id'] = local_monotonic_clock_id()
        class FakeEngine:
            ieee_gpu_reference = InferenceEngine.ieee_gpu_reference
            ieee_request_sources = InferenceEngine.ieee_request_sources
            def __init__(self, cfg, *args):
                self.model_cfg = dict(cfg, ieee_gpu_references=True)
                self.backend, self._engine_dead = 'vllm', False
                self.engine = NS(collective_rpc=self.collective)
            async def collective(self, method, *, kwargs):
                calls.append((method, kwargs))
                entered.set()
                if mode == 'cancelled': await proceed.wait()
                # Only native compute is stubbed here; separate test uses actual owner.
                aid = kwargs.get('_diagnostic_control_id')
                logger.diagnostic_control_event(aid, 'native_begin')
                if mode == 'error': raise ValueError('fixture source failure')
                logger.diagnostic_control_event(aid, 'native_ready')
                completed.set()
                return [copy.deepcopy(payload)]
            async def initialize(self): pass
            async def shutdown(self): pass
        with tempfile.TemporaryDirectory() as root, \
             patch.object(logger, '_diagnostic_stack_state', (os.getpid(), None, None, None, out) if enabled else None):
            path = Path(root)/'payload.json'
            path.write_text(json.dumps(dict(repo_root=str(Path.cwd()), model_cfg={}, cost_model={})))
            with patch('scripts.run_all_experiments.InferenceEngine', FakeEngine), \
                 patch.object(worker, '_write_ready', side_effect=lambda p,v: ready.set_result(v)):
                serving = asyncio.create_task(worker._run_worker(path, Path(root)/'ready.json'))
                address = await asyncio.wait_for(ready, 2.)
                proxy = SubprocessInferenceEngineProxy(process=NS(poll=Mock(return_value=None)),
                    host=address['host'], port=address['port'], model_cfg={'timing_contract':'ieee_tc_native_v1'},
                    cost_model={}, device_id=0, workdir=Path(root), log_path=Path(root)/'none')
                task = asyncio.create_task(proxy.ieee_request_sources(requested_adapter_ids=[4]))
                try:
                    if mode == 'cancelled':
                        await asyncio.wait_for(entered.wait(), 2.)
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError): await task
                        self.assertFalse(proxy._native_rpc_uncertain)
                        proceed.set()
                        await asyncio.wait_for(completed.wait(), 2.)
                        result = await proxy.ieee_request_sources(requested_adapter_ids=[4])
                        self.assertEqual(result['kind'], 'native_lora_request_sources_v1')
                    elif mode == 'error':
                        with self.assertRaisesRegex(RuntimeError, 'fixture source failure'): await task
                        self.assertFalse(proxy._native_rpc_uncertain)
                    else:
                        result = await asyncio.wait_for(task, 2.)
                        self.assertEqual(result['kind'], 'native_lora_request_sources_v1')
                        self.assertNotIn('_diagnostic_control_id', result)
                finally:
                    proceed.set()
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    for channel in list(proxy._rpc_channels): await proxy._drop_rpc_channel(channel)
                    await proxy._rpc('shutdown')
                    await asyncio.wait_for(serving, 2.)
        return calls, events(out)

    async def test_default_wire_and_result_have_no_diagnostic_fields(self):
        calls, rows = await self.roundtrip(enabled=False)
        self.assertEqual(calls, [('ieee_gpu_reference', {'operation':'request_source_snapshot', 'requested_adapter_ids':[4]})])
        self.assertEqual(rows, [])

    async def test_one_identity_ordered_boundaries_and_common_clock(self):
        calls, rows = await self.roundtrip()
        self.assertEqual(len(calls), 1)
        self.assertEqual([r['boundary'] for r in rows], ['parent_begin', 'parent_send', 'worker_received',
            'frontend_begin', 'frontend_native_send', 'native_begin', 'native_ready',
            'frontend_native_received', 'frontend_ready', 'parent_received', 'parent_terminal'])
        self.assertEqual({r['attempt_id'] for r in rows}, {calls[0][1]['_diagnostic_control_id']})
        self.assertEqual({r['clock_id'] for r in rows}, {local_monotonic_clock_id()})
        times = [r['monotonic_s'] for r in rows]
        self.assertEqual(times, sorted(times))
        self.assertEqual(rows[-1]['outcome'], 'success')
        self.assertTrue(all(r['byte_count'] > 0 for r in rows if r['boundary'] in ('parent_send', 'parent_received')))

    async def test_read_failure_has_terminal_error_no_fake_native_ready(self):
        calls, rows = await self.roundtrip(mode='error')
        self.assertEqual(len(calls), 1)
        self.assertEqual(rows[-1]['outcome'], 'error')
        self.assertNotIn('native_ready', [r['boundary'] for r in rows])

    async def test_cancelled_read_and_subsequent_fresh_read_have_distinct_ids(self):
        calls, rows = await self.roundtrip(mode='cancelled')
        self.assertEqual(len(calls), 2)
        self.assertEqual(len({r['attempt_id'] for r in rows}), 2)
        terminal = [r for r in rows if r['boundary'] == 'parent_terminal']
        self.assertEqual([r['outcome'] for r in terminal], ['cancelled', 'success'])


if __name__ == '__main__':
    unittest.main()
