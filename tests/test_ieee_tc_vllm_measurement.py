"""CPU-only execution of pinned native output methods; no engine or weights.

The upstream method bodies are compiled unchanged from the installed 0.30.0
sources. This tests the actual output/collector semantics without importing
CUDA or pretending that a synthetic token event is a performance measurement.
"""
import ast
import asyncio
from collections.abc import MutableSequence
from dataclasses import dataclass
from enum import Enum
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
from typing import cast
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/qhq/serverless_llm_experiment_retry14_baseline')
PACKAGE = Path('/home/qhq/.venvs/primelora_vllm0300_tc_20260925/lib/python3.12/site-packages/vllm')
sys.path.insert(0, str(MAIN))
spec = importlib.util.spec_from_file_location('v030_measurement',
    ROOT/'scripts/ieee_tc_serverless_measurement.py')
measurement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measurement)


class OutputKind(Enum):
    CUMULATIVE = 0
    DELTA = 1
    FINAL_ONLY = 2


def native_fixture():
    measurement.validate_v030_sources(PACKAGE, '0.30.0')
    module = ModuleType('tc_v030_native_output_fixture')
    sys.modules[module.__name__] = module
    module.__dict__.update(asyncio=asyncio, dataclass=dataclass, cast=cast,
        MutableSequence=MutableSequence, RequestOutputKind=OutputKind,
        PoolingOutput=type('PoolingOutput', (), {}),
        PoolingRequestOutput=type('PoolingRequestOutput', (), {}))
    nodes = []
    for source, names in (
        ('outputs.py', ('CompletionOutput', 'RequestOutput')),
        ('v1/metrics/stats.py', ('RequestStateStats',)),
        ('v1/engine/output_processor.py', ('RequestState', 'RequestOutputCollector')),
    ):
        tree = ast.parse((PACKAGE/source).read_text())
        for name in names:
            node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
            if name == 'RequestState':
                # No tokenizer/engine initialization; the tested producer body
                # itself is not rewritten or replaced by a mocked algorithm.
                node.body = [n for n in node.body if getattr(n, 'name', '') == '_new_request_output']
            nodes.append(node)
    tree = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), '<pinned-vllm030-output-methods>', 'exec'),
         module.__dict__)
    return module


class NativeOutputTests(unittest.TestCase):
    def setUp(self):
        self.native = native_fixture()
        self.state = self.native.RequestState()
        self.state.__dict__.update(prompt_token_ids=[1, 2], prompt_embeds=None,
            logprobs_processor=NS(prompt_logprobs=None), output_kind=OutputKind.CUMULATIVE,
            lora_request=NS(lora_name='a', lora_int_id=7, lora_path='/owned/objects/a'),
            lora_name='a', prompt='fixture', num_cached_tokens=0,
            num_cache_creation_tokens=0, parent_req=None, streaming_input=False,
            stats=self.native.RequestStateStats(num_generation_tokens=0, queued_ts=1.1,
                scheduled_ts=1.2))
        self.ids = []

    def install(self):
        measurement._install_snapshot_hooks(self.native, measurement.snapshot_v030_output,
                                             '_ieee_tc_snapshot_v030')

    def emit(self, tokens, at, finished=False):
        self.ids.extend(tokens)
        stats = self.state.stats
        stats.num_generation_tokens = len(self.ids)
        if not stats.first_token_ts and tokens:
            stats.first_token_ts = at
        stats.last_token_ts = at
        completion = self.native.CompletionOutput(index=0, text='', token_ids=self.ids,
            cumulative_logprob=None, logprobs=None,
            finish_reason='length' if finished else None, stop_reason=None)
        return self.state._new_request_output('r', [completion], finished)

    def observer(self):
        with patch.object(measurement.time, 'perf_counter', return_value=1.):
            return measurement.V030NativeRequestObservation(self.state.lora_request)

    def test_uninstrumented_native_stats_and_tokens_alias_later_output(self):
        early = self.emit([10], 2.)
        self.emit([20], 3., True)
        self.assertIs(early.metrics, self.state.stats)
        self.assertEqual(early.metrics.num_generation_tokens, 2)
        self.assertEqual(early.outputs[0].token_ids, [10, 20])

    def test_uninstrumented_native_merge_does_not_propagate_metrics(self):
        self.install()
        early, late = self.emit([10], 2.), self.emit([20], 3., True)
        early.add(late, aggregate=False)  # exact upstream method without collector hook
        self.assertEqual(early.outputs[0].token_ids, [10, 20])
        self.assertEqual(early.metrics.num_generation_tokens, 1)

    def test_snapshot_retains_native_token_even_with_empty_decoded_text(self):
        self.install()
        observed = self.observer()
        early = self.emit([10], 2.)
        late = self.emit([20], 3., True)
        self.assertEqual(early.outputs[0].token_ids, [10])
        self.assertEqual(early.metrics.num_generation_tokens, 1)
        observed.observe(early)
        observed.observe(late)
        with patch.object(measurement.time, 'perf_counter', return_value=7.):
            result = observed.finish({})
        self.assertEqual(result['native_tpot_ms'], 1000.)
        self.assertEqual(result['worker_completion_notification_ms'], 4000.)
        self.assertEqual(result['completion_token_ids'], [10, 20])

    def test_slow_consumer_merges_tokens_and_snapshots_at_same_event(self):
        self.install()
        observed = self.observer()
        collector = self.native.RequestOutputCollector(OutputKind.CUMULATIVE, 'r')
        collector.put(self.emit([10], 2.))
        collector.put(self.emit([20], 3.))
        collector.put(self.emit([], 8., True))
        output = collector.get_nowait()
        observed.observe(output)
        self.assertIsNone(collector.get_nowait())
        with patch.object(measurement.time, 'perf_counter', return_value=9.):
            result = observed.finish({})
        self.assertEqual(result['native_last_token_monotonic_s'], 3.)
        self.assertEqual(result['native_tpot_ms'], 1000.)
        self.assertEqual(result['worker_completion_notification_ms'], 6000.)
        self.assertEqual(self.state.stats.last_token_ts, 8.)  # native stats never rewritten

    def test_one_token_and_separate_terminal_do_not_invent_tpot(self):
        self.install()
        observed = self.observer()
        observed.observe(self.emit([10], 2.))
        observed.observe(self.emit([], 5., True))
        with patch.object(measurement.time, 'perf_counter', return_value=6.):
            result = observed.finish({})
        self.assertIsNone(result['native_tpot_ms'])
        self.assertEqual(result['native_last_token_monotonic_s'], 2.)

    def test_native_exception_is_not_converted_to_a_successful_output(self):
        self.install()
        collector = self.native.RequestOutputCollector(OutputKind.CUMULATIVE, 'r')
        collector.put(self.emit([10], 2.))
        error = RuntimeError('native failure')
        collector.put(error)
        with self.assertRaises(RuntimeError) as caught:
            collector.get_nowait()
        self.assertIs(caught.exception, error)

    def test_wrong_terminal_and_clock_order_fail_without_fallback(self):
        self.install()
        observed = self.observer()
        output = self.emit([10], 2., True)
        output.outputs[0].finish_reason = 'abort'
        with self.assertRaisesRegex(ValueError, 'finish by length'):
            observed.observe(output)
        output.outputs[0].finish_reason = 'length'
        output.metrics.queued_ts = 3.
        with self.assertRaisesRegex(ValueError, 'timestamps violate'):
            observed.observe(output)

    def test_mixed_request_collector_fails_instead_of_merging_evidence(self):
        self.install()
        early, late = self.emit([10], 2.), self.emit([20], 3., True)
        late.request_id = 'another-request'
        collector = self.native.RequestOutputCollector(OutputKind.CUMULATIVE, 'r')
        collector.put(early)
        with self.assertRaisesRegex(ValueError, 'mixed request'):
            collector.put(late)

    def test_wrong_adapter_identity_rejected_even_after_output_merge(self):
        self.install()
        for field, value in (('lora_name', 'wrong'), ('lora_int_id', 8),
                             ('lora_path', '/different/path')):
            with self.subTest(field=field):
                observed = self.observer()
                early = self.emit([10], 2.)
                output = self.emit([20], 3., True)
                identity = list(output.tc_native_lora_identity)
                identity[('lora_name', 'lora_int_id', 'lora_path').index(field)] = value
                output.tc_native_lora_identity = tuple(identity)
                collector = self.native.RequestOutputCollector(OutputKind.CUMULATIVE, 'r')
                collector.put(early)
                collector.put(output)
                with self.assertRaisesRegex(ValueError, 'adapter ID/path'):
                    observed.observe(collector.get_nowait())

    def test_missing_stats_or_non_cumulative_mode_has_no_fallback(self):
        self.install()
        for kind in (OutputKind.DELTA, OutputKind.FINAL_ONLY):
            self.state.output_kind = kind
            # Bypass DELTA prompt-logprob API only in this guard test: the
            # instrumentation must reject the mode before any observation.
            with self.assertRaisesRegex(ValueError, 'cumulative'):
                measurement.snapshot_v030_output(self.state, NS(outputs=[NS(token_ids=[])]))
        self.state.output_kind = OutputKind.CUMULATIVE
        self.state.stats = None
        with self.assertRaisesRegex(ValueError, 'log_stats'):
            measurement.snapshot_v030_output(self.state, NS(outputs=[NS(token_ids=[])]))

    def test_producer_count_mismatch_fails(self):
        self.install()
        output = self.emit([10], 2.)
        self.state.stats.num_generation_tokens = 2
        with self.assertRaisesRegex(ValueError, 'count mismatch'):
            measurement.snapshot_v030_output(self.state, output)

    def test_installation_is_idempotent_and_other_methods_unchanged(self):
        original_add = self.native.RequestOutput.add
        self.install()
        producer, put = self.native.RequestState._new_request_output, self.native.RequestOutputCollector.put
        self.install()
        self.assertIs(self.native.RequestState._new_request_output, producer)
        self.assertIs(self.native.RequestOutputCollector.put, put)
        self.assertIs(self.native.RequestOutput.add, original_add)

    def test_version_and_each_source_identity_fail_closed(self):
        self.assertEqual(measurement.validate_v030_sources(PACKAGE, '0.30.0'),
                         measurement.V030_NATIVE_SOURCES)
        with self.assertRaisesRegex(ValueError, 'version'):
            measurement.validate_v030_sources(PACKAGE, '0.10.2')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in measurement.V030_NATIVE_SOURCES:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((PACKAGE/name).read_bytes())
            for name in measurement.V030_NATIVE_SOURCES:
                path = root/name
                original = path.read_bytes()
                path.write_bytes(original+b'\n# different installation\n')
                with self.subTest(source=name), self.assertRaisesRegex(ValueError, 'sources'):
                    measurement.validate_v030_sources(root, '0.30.0')
                path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
