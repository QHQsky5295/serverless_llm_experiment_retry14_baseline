"""CPU-only observations: preserve ordering/return/error and expose truncation."""
import io
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from faaslora.utils.core_timing import CoreHostTiming, _client_label, install_core_host_timing


def fixture():
    calls = []
    result = NS(num_scheduled_tokens={'private-request': 10, 'private-request2': 1},
                total_num_scheduled_tokens=11)
    class Scheduler:
        requests = {'private-request': object()}
        running, waiting = [object()], []
        def schedule(self):
            calls.append('schedule')
            return result
    class Core:
        def __init__(self, scheduler): self.scheduler = scheduler
        def _process_input_queue(self):
            calls.append('queue')
            self._handle_client_request(NS(name='UTILITY'),
                [0, 19, 'collective_rpc', ['ieee_gpu_reference', None, [],
                    {'operation':'completion_fence', 'secret':'do-not-record'}]])
            return 'queue-return'
        def _handle_client_request(self, kind, value): calls.append('client')
        def _process_engine_step(self):
            calls.append('step')
            assert self.scheduler.schedule() is result
            return True
        def shutdown(self): calls.append('shutdown')
    return Core, Scheduler(), calls


class CoreTiming(unittest.TestCase):
    def rows(self, out): return [json.loads(s) for s in out.getvalue().splitlines()]

    def test_nested_spans_preserve_methods_results_order_and_private_payload(self):
        cls, scheduler, calls = fixture()
        out = io.BytesIO()
        observer = CoreHostTiming(out, scheduler)
        observer.attach(cls)
        observer.attach(cls)
        core = cls(scheduler)
        self.assertEqual(core._process_input_queue(), 'queue-return')
        self.assertIs(core._process_engine_step(), True)
        self.assertEqual(calls, ['queue', 'client', 'step', 'schedule'])
        rows = self.rows(out)
        self.assertEqual([r['boundary'] for r in rows], ['client', 'input_queue', 'schedule', 'step'])
        for child, parent in [(rows[0], rows[1]), (rows[2], rows[3])]:
            self.assertEqual(child['parent'], parent['id'])
            self.assertLessEqual(parent['begin_ns'], child['begin_ns'])
            self.assertLessEqual(child['end_ns'], parent['end_ns'])
        self.assertEqual(rows[0]['details'], dict(request_type='UTILITY', utility='collective_rpc',
            worker_method='ieee_gpu_reference', operation='completion_fence'))
        self.assertEqual(rows[2]['details'], dict(scheduled_requests=2, scheduled_tokens=11, multi_token_requests=1))
        self.assertEqual(rows[1]['details']['unfinished_before'], 1)
        self.assertGreater(observer.observer_ns, 0)
        self.assertGreater(observer.write_ns, 0)
        self.assertTrue(all(r['end_ns'] >= r['begin_ns'] for r in rows))
        self.assertNotIn(b'private-request', out.getvalue())
        self.assertNotIn(b'do-not-record', out.getvalue())

    def test_default_does_not_patch_native_methods_or_scheduler(self):
        cls, scheduler, _ = fixture()
        before = dict(vars(cls))
        for flag in (None, '0'):
            with patch.dict(os.environ, {}, clear=True):
                if flag is not None: os.environ['FAASLORA_TC_CORE_TIMING'] = flag
                self.assertIsNone(install_core_host_timing(cls, scheduler))
            self.assertEqual(dict(vars(cls)), before)
            self.assertNotIn('schedule', vars(scheduler))

    def test_unknown_labels_never_echo_payload(self):
        label = _client_label((NS(name='UTILITY'), [0, 1, 'secret-unknown', ['secret']]))
        self.assertEqual(label, dict(request_type='UTILITY', utility='other'))
        self.assertEqual(_client_label((NS(name='secret'), 'secret')), dict(request_type='OTHER'))

    def test_original_exception_identity_is_preserved(self):
        _, scheduler, _ = fixture()
        out, failure = io.BytesIO(), ValueError('private-error')
        observer = CoreHostTiming(out, scheduler)
        def fail(): raise failure
        with self.assertRaises(ValueError) as ctx:
            observer.call('step', fail, (), {})
        self.assertIs(ctx.exception, failure)
        self.assertEqual(self.rows(out)[0]['outcome'], 'raised')
        self.assertEqual(observer.stack, [])
        self.assertNotIn(b'private-error', out.getvalue())

    def test_cap_stops_recording_not_native_work(self):
        cls, scheduler, calls = fixture()
        out = io.BytesIO()
        observer = CoreHostTiming(out, scheduler, max_spans=1)
        observer.attach(cls)
        core = cls(scheduler)
        for _ in range(3): self.assertTrue(core._process_engine_step())
        self.assertEqual(calls, ['step', 'schedule']*3)
        self.assertEqual([r['event'] for r in self.rows(out)], ['native_core_host_span_v1', 'observation_limit'])
        self.assertTrue(observer.truncated)

    def test_byte_cap_and_close_footer_are_bounded(self):
        _, scheduler, _ = fixture()
        class KeepBytes(io.BytesIO):
            def close(self): pass
        out = KeepBytes()
        observer = CoreHostTiming(out, scheduler, max_bytes=8192)
        for _ in range(100): observer.call('step', lambda: False, (), {})
        observer.close()
        observer.close()
        self.assertLessEqual(len(out.getvalue()), 8192)
        self.assertTrue(self.rows(out)[-1]['truncated'])
        self.assertEqual(self.rows(out)[-1]['event'], 'observation_close')
        self.assertIs(observer.call('step', lambda: True, (), {}), True)

    def test_foreign_pid_thread_core_and_collision_reject(self):
        cls, scheduler, _ = fixture()
        observer = CoreHostTiming(io.BytesIO(), scheduler)
        observer.attach(cls)
        core = cls(scheduler)
        core._process_engine_step()
        with self.assertRaisesRegex(ValueError, 'another core'): cls(scheduler)._process_engine_step()
        with self.assertRaisesRegex(ValueError, 'another owner'):
            CoreHostTiming(io.BytesIO(), scheduler).attach(cls)
        with patch('faaslora.utils.core_timing.os.getpid', return_value=observer.pid+1):
            with self.assertRaisesRegex(ValueError, 'owner changed'): core._process_engine_step()
        errors = []
        def other():
            try: core._process_engine_step()
            except ValueError as error: errors.append(str(error))
        thread = threading.Thread(target=other)
        thread.start(); thread.join()
        self.assertEqual(errors, ['native core timing owner changed'])

    def test_short_write_is_not_silently_accepted(self):
        _, scheduler, _ = fixture()
        observer = CoreHostTiming(NS(write=lambda data: 0), scheduler)
        with self.assertRaisesRegex(OSError, 'incomplete'):
            observer.call('step', lambda: True, (), {})

    def test_install_gate_actual_output_and_shutdown_close(self):
        group = next(r[3:] for r in Path('/proc/self/cgroup').read_text().splitlines() if r.startswith('0::'))
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp)/'receipt.json'
            good = dict(allow_exec=True, service_identity=dict(path='/sys/fs/cgroup'+group),
                external_replay=dict(replay_scope='diagnostic_prefix_v1', diagnostic_prefix_count=1000))
            receipt.write_text(json.dumps(good))
            env = dict(FAASLORA_FORMAL_RUN='0', FAASLORA_TC_CORE_TIMING='host_spans_v1',
                       FAASLORA_TC_LAUNCH_RECEIPT=str(receipt))
            for change in (dict(FAASLORA_FORMAL_RUN='1'), dict(FAASLORA_TC_CORE_TIMING='yes')):
                cls, scheduler, _ = fixture()
                with patch.dict(os.environ, dict(env, **change)), self.assertRaises(ValueError):
                    install_core_host_timing(cls, scheduler)
            self.assertFalse((Path(tmp)/'diagnostic_core').exists())
            good['service_identity']['path'] = '/sys/fs/cgroup/not-our-service'
            receipt.write_text(json.dumps(good))
            cls, scheduler, _ = fixture()
            with patch.dict(os.environ, env), self.assertRaisesRegex(ValueError, 'outside'):
                install_core_host_timing(cls, scheduler)
            good['service_identity']['path'] = '/sys/fs/cgroup'+group
            receipt.write_text(json.dumps(good))
            cls, scheduler, calls = fixture()
            with patch.dict(os.environ, env):
                observer = install_core_host_timing(cls, scheduler)
                self.assertIs(install_core_host_timing(cls, scheduler), observer)
                core = cls(scheduler)
                core._process_engine_step()
                core.shutdown()
                self.assertTrue(observer.closed)
                self.assertEqual(calls, ['step', 'schedule', 'shutdown'])
                cls2, scheduler2, _ = fixture()
                with self.assertRaises(FileExistsError): install_core_host_timing(cls2, scheduler2)
            path, = (Path(tmp)/'diagnostic_core').glob('*.jsonl')
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            self.assertFalse(rows[0]['gpu_kernel_timing'])
            self.assertFalse(rows[0]['formal_performance_result'])
            self.assertEqual(rows[-1]['event'], 'observation_close')


if __name__ == '__main__':
    unittest.main()
