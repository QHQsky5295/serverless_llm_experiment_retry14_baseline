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


class StepComponents(unittest.TestCase):
    def counts(self, alloc=0, free=0):
        return dict(num_host_alloc=alloc, num_host_free=free,
                    **{'host_alloc_time.total': 0.0, 'host_free_time.total': 0.0})

    def setup_observer(self):
        _, scheduler, _ = fixture()
        output = io.BytesIO()
        observer = CoreHostTiming(output, scheduler, step_components=True)
        return observer, output

    def test_nested_components_and_allocator_deltas_do_not_add_spans(self):
        observer, output = self.setup_observer()
        calls, result = [], object()
        counts = self.counts(4, 2)
        class Runner:
            def prepare_inputs(self, *args, **kwargs):
                calls.append((args, kwargs)); counts['num_host_alloc'] += 2
                return result
            def execute_model(self):
                assert self.prepare_inputs('private', _method='private-kw') is result
                return result
        runner = Runner()
        observer._attach_component_targets([
            (runner, 'prepare_inputs', 'prepare_inputs'),
            (runner, 'execute_model', 'execute_model')], lambda: counts)
        self.assertIs(runner.execute_model(), result)  # Outside step: no timing.
        self.assertEqual(output.getvalue(), b'')
        def step():
            self.assertIs(runner.execute_model(), result)
            return True
        self.assertIs(observer.call('step', step, (), {}), True)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(rows), 1)
        data = rows[0]['details']['step_components_v1']
        self.assertTrue(data['nested_not_additive'])
        self.assertFalse(data['gpu_kernel_timing'])
        self.assertEqual(data['allocator_before']['num_host_alloc'], 6)
        self.assertEqual(data['allocator_after']['num_host_alloc'], 8)
        inner, outer = [data['values'][key] for key in ('prepare_inputs', 'execute_model')]
        self.assertEqual((inner[0], outer[0]), (1, 1))
        self.assertEqual((inner[3], outer[3]), (0, 0))
        self.assertLessEqual(inner[1], outer[1])
        self.assertLessEqual(outer[1], rows[0]['end_ns'] - rows[0]['begin_ns'])
        self.assertEqual(calls, [(('private',), {'_method': 'private-kw'})] * 2)
        self.assertNotIn(b'private', output.getvalue())
        self.assertIsNone(observer.component_rows)

    def test_exception_identity_and_second_step_reset(self):
        observer, output = self.setup_observer()
        failure, count = ValueError('private failure'), []
        def original():
            count.append(1)
            raise failure
        obj = NS(method=original)
        observer._attach_component_targets([(obj, 'method', 'sample_tokens')],
            self.counts)
        with self.assertRaises(ValueError) as caught:
            observer.call('step', obj.method, (), {})
        self.assertIs(caught.exception, failure)
        self.assertIsNone(observer.component_rows)
        observer.call('step', lambda: False, (), {})
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(rows[0]['outcome'], 'raised')
        self.assertEqual(rows[0]['details']['step_components_v1']['values']['sample_tokens'][3], 1)
        self.assertEqual(rows[1]['details']['step_components_v1']['values'], {})
        self.assertEqual(count, [1])
        self.assertNotIn(b'private failure', output.getvalue())

    def test_cap_keeps_original_work_and_stops_component_observation(self):
        observer, output = self.setup_observer()
        observer.max_spans = 0
        calls = []
        obj = NS(method=lambda: calls.append(1))
        observer._attach_component_targets([(obj, 'method', 'prepare_inputs')],
            self.counts)
        for _ in range(3): observer.call('step', obj.method, (), {})
        self.assertTrue(observer.truncated)
        self.assertEqual(calls, [1, 1, 1])
        self.assertIsNone(observer.component_rows)
        self.assertEqual(len(output.getvalue().splitlines()), 1)

    def test_all_targets_validated_before_patch(self):
        observer, _ = self.setup_observer()
        original = lambda: None
        obj = NS(good=original, bad=None)
        with self.assertRaises(ValueError):
            observer._attach_component_targets([(obj, 'good', 'good'), (obj, 'bad', 'bad')], lambda: {})
        self.assertIs(obj.good, original)
        self.assertEqual(observer.component_targets, [])

    def test_explicit_diagnostic_flag_and_dependency(self):
        for main, component in ((None, 'v2_components_v1'), ('0', 'v2_components_v1'),
                                ('host_spans_v1', 'typo')):
            cls, scheduler, _ = fixture()
            before = dict(vars(cls))
            with patch.dict(os.environ, {}, clear=True):
                if main is not None: os.environ['FAASLORA_TC_CORE_TIMING'] = main
                os.environ['FAASLORA_TC_CORE_STEP_TIMING'] = component
                with self.assertRaises(ValueError): install_core_host_timing(cls, scheduler)
            self.assertEqual(dict(vars(cls)), before)

    def test_unknown_or_reset_allocator_counts_reject(self):
        observer, _ = self.setup_observer()
        obj = NS(method=lambda: True)
        observer._attach_component_targets([(obj, 'method', 'prepare_inputs')], lambda: {})
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            observer.call('step', obj.method, (), {})
        self.assertIsNone(observer.component_rows)
        counts = iter([self.counts(2, 1), self.counts(1, 1)])
        observer.host_stats = lambda: next(counts)
        with self.assertRaisesRegex(ValueError, 'reset'):
            observer.call('step', obj.method, (), {})
        self.assertIsNone(observer.component_rows)


class HostInventoryTiming(unittest.TestCase):
    def fixture(self, **options):
        _, scheduler, _ = fixture()
        output = io.BytesIO()
        observer = CoreHostTiming(output, scheduler, host_inventory=True, **options)
        calls = []
        result = dict(host_footprint_scope='native_registered_tensor_storage_capacity',
                      host_allocations=[{'private_pointer': object()}],
                      host_adapter_footprints=[{'private_id': 'secret-adapter'}],
                      host_tensor_storage_bytes=4096)
        def inventory(*args, **kwargs):
            calls.append(('inventory', args, kwargs))
            return result
        def allocator(*args, **kwargs):
            calls.append(('allocator', args, kwargs))
            return (1, 2, 3, 4)
        module = NS(_ieee_lora_host_inventory=inventory,
                    _ieee_pinned_host_allocator_stats=allocator)
        return observer, output, module, calls, result

    def test_whole_function_timings_nested_under_client_with_scalar_projection(self):
        observer, output, module, calls, result = self.fixture()
        observer.attach_host_inventory(module)
        def client():
            self.assertIs(module._ieee_lora_host_inventory('private', token='secret'), result)
            self.assertEqual(module._ieee_pinned_host_allocator_stats(), (1, 2, 3, 4))
            return result
        self.assertIs(observer.call('client', client, (), {}), result)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([r['boundary'] for r in rows],
                         ['host_inventory', 'host_allocator_stats', 'client'])
        for row in rows[:2]:
            self.assertEqual(row['parent'], rows[2]['id'])
            self.assertGreaterEqual(row['begin_ns'], rows[2]['begin_ns'])
            self.assertLessEqual(row['end_ns'], rows[2]['end_ns'])
        self.assertEqual(rows[0]['details'], dict(
            scope='native_registered_tensor_storage_capacity', allocation_count=1,
            adapter_count=1, host_tensor_storage_bytes=4096))
        self.assertEqual(len(calls), 2)
        self.assertNotIn(b'private', output.getvalue())
        self.assertNotIn(b'secret', output.getvalue())

    def test_outside_owner_span_and_foreign_thread_are_untouched(self):
        observer, output, module, calls, result = self.fixture()
        observer.attach_host_inventory(module)
        self.assertIs(module._ieee_lora_host_inventory(), result)
        observer.stack.append(123)
        returned = []
        thread = threading.Thread(target=lambda: returned.append(module._ieee_lora_host_inventory()))
        thread.start(); thread.join()
        with patch('faaslora.utils.core_timing.os.getpid', return_value=observer.pid+1):
            self.assertIs(module._ieee_lora_host_inventory(), result)
        observer.stack.pop()
        self.assertEqual(returned, [result])
        self.assertEqual(len(calls), 3)
        self.assertEqual(output.getvalue(), b'')

    def test_exception_identity_result_and_originals_restored(self):
        observer, output, module, calls, _ = self.fixture()
        failure = RuntimeError('secret-error')
        def failed():
            calls.append('failed')
            raise failure
        module._ieee_lora_host_inventory = failed
        original_stats = module._ieee_pinned_host_allocator_stats
        observer.attach_host_inventory(module)
        with self.assertRaises(RuntimeError) as caught:
            observer.call('client', module._ieee_lora_host_inventory, (), {})
        self.assertIs(caught.exception, failure)
        self.assertEqual(calls, ['failed'])
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([row['outcome'] for row in rows], ['raised', 'raised'])
        self.assertNotIn(b'secret', output.getvalue())
        observer.close()
        self.assertIs(module._ieee_lora_host_inventory, failed)
        self.assertIs(module._ieee_pinned_host_allocator_stats, original_stats)

    def test_invalid_target_or_collision_cannot_partially_patch(self):
        observer, _, module, _, _ = self.fixture()
        original = module._ieee_lora_host_inventory
        stats = module._ieee_pinned_host_allocator_stats
        module._ieee_pinned_host_allocator_stats = None
        with self.assertRaises(ValueError): observer.attach_host_inventory(module)
        self.assertIs(module._ieee_lora_host_inventory, original)
        module._ieee_pinned_host_allocator_stats = stats
        observer.attach_host_inventory(module)
        second, _, _, _, _ = self.fixture()
        with self.assertRaises(ValueError): second.attach_host_inventory(module)
        with self.assertRaises(ValueError): observer.attach_host_inventory(module)
        later = lambda: 'later'
        module._ieee_lora_host_inventory = later
        observer.restore_host_inventory()
        self.assertIs(module._ieee_lora_host_inventory, later)
        self.assertIs(module._ieee_pinned_host_allocator_stats, stats)

    def test_cap_preserves_all_native_calls(self):
        observer, output, module, calls, result = self.fixture(max_spans=0)
        observer.attach_host_inventory(module)
        for _ in range(3):
            self.assertIs(observer.call('client', module._ieee_lora_host_inventory, (), {}), result)
        self.assertEqual(len(calls), 3)
        self.assertTrue(observer.truncated)
        self.assertEqual(len(output.getvalue().splitlines()), 1)

    def test_flag_rejects_unregistered_and_missing_main_diagnostic(self):
        for flag, main in [('fresh_inventory_v1', None), ('fresh_inventory_v1', '0'),
                           ('typo', 'host_spans_v1')]:
            cls, scheduler, _ = fixture()
            before = dict(vars(cls))
            with patch.dict(os.environ, {}, clear=True):
                os.environ['FAASLORA_TC_HOST_INVENTORY_TIMING'] = flag
                if main is not None: os.environ['FAASLORA_TC_CORE_TIMING'] = main
                with self.assertRaises(ValueError): install_core_host_timing(cls, scheduler)
            self.assertEqual(dict(vars(cls)), before)

    def test_install_requires_existing_diagnostic_resource_gate_and_closes(self):
        from faaslora.memory import gpu_monitor
        original = gpu_monitor._ieee_lora_host_inventory
        group = next(r[3:] for r in Path('/proc/self/cgroup').read_text().splitlines() if r.startswith('0::'))
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp)/'receipt.json'
            good = dict(allow_exec=True, service_identity=dict(path='/sys/fs/cgroup'+group),
                external_replay=dict(replay_scope='diagnostic_prefix_v1', diagnostic_prefix_count=1000))
            receipt.write_text(json.dumps(good))
            env = dict(FAASLORA_FORMAL_RUN='0', FAASLORA_TC_CORE_TIMING='host_spans_v1',
                       FAASLORA_TC_HOST_INVENTORY_TIMING='fresh_inventory_v1',
                       FAASLORA_TC_LAUNCH_RECEIPT=str(receipt))
            cls, scheduler, _ = fixture()
            with patch.dict(os.environ, dict(env, FAASLORA_FORMAL_RUN='1')):
                with self.assertRaises(ValueError): install_core_host_timing(cls, scheduler)
            self.assertIs(gpu_monitor._ieee_lora_host_inventory, original)
            with patch.dict(os.environ, env):
                observer = install_core_host_timing(cls, scheduler)
                self.assertIs(install_core_host_timing(cls, scheduler), observer)
                self.assertIsNot(gpu_monitor._ieee_lora_host_inventory, original)
                with patch.dict(os.environ, FAASLORA_TC_HOST_INVENTORY_TIMING='0'):
                    with self.assertRaisesRegex(ValueError, 'identity changed'):
                        install_core_host_timing(cls, scheduler)
                cls(scheduler).shutdown()
            self.assertTrue(observer.closed)
            self.assertIs(gpu_monitor._ieee_lora_host_inventory, original)
            path, = (Path(tmp)/'diagnostic_core').glob('*.jsonl')
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[0]['host_inventory_timing'], 'fresh_inventory_v1')
            self.assertFalse(rows[0]['formal_performance_result'])


if __name__ == '__main__':
    unittest.main()
