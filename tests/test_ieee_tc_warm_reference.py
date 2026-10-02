"""No-GPU V1 warm-reference indexing tests; never generate experiment inputs."""
import copy
import asyncio
import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.ieee_tc_preflight import (warm_reference_input_index, validate_warm_reference_index,
    warm_reference_batches, validate_warm_native_sample, qualify_warm_reference)


def fixture(lengths):
    entries, rows = [], []
    for i, length in enumerate(lengths):
        rid, aid = f'req_{i:05d}', f'adapter_{i % 8}'
        entries.append(SimpleNamespace(request_id=rid, source_sha256=f'{i:064x}',
            source_json=json.dumps(dict(adapter_id=aid, expected_output_tokens=64))))
        rows.append(dict(request_id=rid, adapter_id=aid, input_tokens=length,
            source_expected_output_tokens=64, requested_completion_tokens=64,
            generation_contract='fixed_length_greedy_v1', canonical_prompt_sha256='a'*64,
            native_token_timing=dict(actual_prompt_tokens=length,
                                    native_prompt_token_ids_sha256='b'*64)))
    plan = SimpleNamespace(entries=entries, source_count=len(entries), profile='W0',
                           rate_scale=1., identity=lambda: dict(count=len(entries)))
    return plan, rows


class WarmReferenceIndex(unittest.TestCase):
    def test_four_quartiles_complete_input_identity_and_no_latency_dependency(self):
        plan, rows = fixture([100]*300+[200]*300+[300]*300+[400]*300)
        original = copy.deepcopy(rows)
        index = warm_reference_input_index(plan, rows)
        self.assertEqual(index['raw_quartiles'], [100, 200, 300])
        self.assertEqual(index['finite_upper_bounds'], [100, 200, 300])
        self.assertEqual(len(index['groups']), 4)
        for group in index['groups']:
            self.assertEqual(group['source_count'], 300)
            self.assertEqual(group['selected_unique_requests'], 256)
            self.assertEqual(len(group['selected']), 256)
            self.assertEqual(group['selected'][0]['source_index'], group['group_id']*300)
        self.assertEqual(rows, original)
        for row in rows:
            row.update(ttft_ms=float('nan'), tpot_ms=-99, success=False)
        self.assertEqual(warm_reference_input_index(plan, rows[::-1]), index)
        self.assertFalse(index['measured_warm_reference'])
        self.assertFalse(index['formal_g1_g2_qualified'])
        self.assertIsNone(index['batch_size'])
        self.assertIsNone(index['thresholds'])

    def test_duplicate_boundaries_merge_and_never_make_empty_last_group(self):
        plan, rows = fixture([759]*1024)
        index = warm_reference_input_index(plan, rows)
        self.assertEqual(index['raw_quartiles'], [759]*3)
        self.assertEqual(index['finite_upper_bounds'], [])
        self.assertEqual(len(index['groups']), 1)
        self.assertEqual([r['source_index'] for r in index['groups'][0]['selected']],
                         list(range(0, 1024, 4)))

    def test_equal_middle_boundaries_are_merged(self):
        plan, rows = fixture([100]*512+[200]*512)
        index = warm_reference_input_index(plan, rows)
        self.assertEqual(index['raw_quartiles'], [100, 100, 200])
        self.assertEqual(index['finite_upper_bounds'], [100])
        self.assertEqual([g['source_count'] for g in index['groups']], [512, 512])

    def test_prefix_or_transformed_trace_is_rejected(self):
        for name, value in [('source_count',1025), ('profile','W1'), ('rate_scale',2.)]:
            with self.subTest(name=name):
                plan, rows = fixture([100]*1024)
                setattr(plan, name, value)
                with self.assertRaisesRegex(ValueError, 'entire unchanged'):
                    warm_reference_input_index(plan, rows)

    def test_missing_duplicate_or_foreign_measured_requests_rejected(self):
        plan, rows = fixture([100]*1024)
        for bad in (rows[:-1], rows+[rows[0]], rows[:-1]+[dict(rows[-1], request_id='other')]):
            with self.assertRaises(ValueError):
                warm_reference_input_index(plan, bad)

    def test_wrong_adapter_contract_hash_or_length_rejected(self):
        variants = [('adapter_id','wrong'),('input_tokens',101),
                    ('generation_contract','text_tokens'),('source_expected_output_tokens',65),
                    ('requested_completion_tokens',63),('canonical_prompt_sha256','missing')]
        for field, value in variants:
            with self.subTest(field=field):
                plan, rows = fixture([100]*1024)
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    warm_reference_input_index(plan, rows)
        for value in (0, True, 961, 1025):
            plan, rows = fixture([100]*1024)
            rows[0]['native_token_timing']['actual_prompt_tokens'] = value
            rows[0]['input_tokens'] = value
            with self.assertRaises(ValueError):
                warm_reference_input_index(plan, rows)

    def test_single_token_exclusion_is_explicit_and_does_not_fill_with_duplicates(self):
        plan, rows = fixture([100]*1024)
        for entry, row in zip(plan.entries[:10],rows[:10]):
            source = json.loads(entry.source_json)
            source['expected_output_tokens'] = 1
            entry.source_json = json.dumps(source)
            row.update(source_expected_output_tokens=1, requested_completion_tokens=1)
        group = warm_reference_input_index(plan, rows)['groups'][0]
        self.assertEqual(group['excluded_single_token_count'], 10)
        self.assertEqual(group['selected'][0]['source_index'], 10)
        plan, rows = fixture([100]*255)
        with self.assertRaisesRegex(ValueError, 'lacks 256 distinct'):
            warm_reference_input_index(plan, rows)


class WarmReferenceRuntimeInputs(unittest.TestCase):
    def test_bind_index_and_reject_changed_source_or_repeated_request(self):
        plan, rows = fixture([100]*512+[200]*512)
        index = warm_reference_input_index(plan, rows)
        self.assertIs(validate_warm_reference_index(index, plan), index)
        for field, value in [('source', {}), ('rounds', 2), ('thresholds', [999])]:
            bad = copy.deepcopy(index)
            bad[field] = value
            with self.assertRaises(ValueError):
                validate_warm_reference_index(bad, plan)
        for field, value in [('source_index', 2), ('adapter_id', 'wrong'),
                             ('native_prompt_tokens', 201), ('target_tokens', 63),
                             ('source_row_sha256', 'f'*64)]:
            bad = copy.deepcopy(index)
            bad['groups'][0]['selected'][0][field] = value
            with self.assertRaises(ValueError):
                validate_warm_reference_index(bad, plan)

    def test_measure_exact_256_per_group_three_rounds_and_separate_warmup(self):
        plan, rows = fixture([100]*512+[200]*512)
        index = warm_reference_input_index(plan, rows)
        batches = list(warm_reference_batches(index, 'measure'))
        self.assertEqual(len(batches), 194)
        self.assertEqual(sum(len(b['selected']) for b in batches if b['phase']=='warmup'), 16)
        for r in range(3):
            for g in range(2):
                selected = [item for b in batches if b['round']==r and b['group_id']==g
                            for item in b['selected']]
                self.assertEqual(selected, index['groups'][g]['selected'])
        self.assertTrue(all(len(b['selected'])==8 for b in batches))

    def test_probe_maximum_kv_and_distinct_adapter_batches_not_latency(self):
        plan, rows = fixture([100]*1024)
        index = warm_reference_input_index(plan, rows)
        selected = index['groups'][0]['selected']
        for i, item in enumerate(selected):
            item['adapter_id'] = str(i % 2)
        for i in range(8,16):
            selected[i]['target_tokens'] = 256
        for i in range(16,24):
            selected[i]['adapter_id'] = str(i)
        batches = list(warm_reference_batches(index, 'probe'))
        self.assertEqual([b['batch_index'] for b in batches], [1,2])
        self.assertTrue(all(b['round'] is None for b in batches))

    def test_native_contract_recomputed_not_proxy_wall(self):
        selected = dict(target_tokens=5, native_prompt_tokens=100,
                        native_prompt_token_ids_sha256='a'*64)
        timing = dict(native_dispatch_monotonic_s=10., native_first_token_monotonic_s=10.1,
                      native_last_token_monotonic_s=10.3, native_ttft_ms=100., native_tpot_ms=50.,
                      native_terminal_observed=True, actual_prompt_tokens=100,
                      native_prompt_token_ids_sha256='a'*64, gpu_reference_adapter_int_id=1,
                      parent_rpc_wall_ms=999999.)
        measured = validate_warm_native_sample(selected, timing, 5, 1)
        self.assertAlmostEqual(measured['ttft_ms'], 100.)
        self.assertAlmostEqual(measured['tpot_ms'], 50.)
        for field, value in [('native_terminal_observed',False),('actual_prompt_tokens',99),
                             ('native_prompt_token_ids_sha256','b'*64),
                             ('gpu_reference_adapter_int_id',2),('native_tpot_ms',52.),
                             ('native_first_token_monotonic_s',float('nan'))]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_warm_native_sample(selected, dict(timing, **{field:value}), 5, 1)
        with self.assertRaises(ValueError):
            validate_warm_native_sample(selected, timing, 4, 1)


class WarmReferenceExecution(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, free_blocks):
        plan, rows = fixture([100]*1024)
        for entry in plan.entries:
            source = json.loads(entry.source_json)
            source.update(expected_input_tokens=100, body=dict(messages=[dict(role='user',content='x')]))
            entry.source_json = json.dumps(source)
        for row in rows:
            row['canonical_prompt_sha256'] = hashlib.sha256(b'x').hexdigest()
        index = warm_reference_input_index(plan, rows)
        events, held = [], {}

        class Engine:
            def _lora_int_id(self, aid):
                return int(aid.split('_')[1])+1

            def prepare_request(self, *args, **kwargs):
                return SimpleNamespace(prompt='x', max_tokens=64)

            async def ieee_gpu_reference(self, operation, **kwargs):
                events.append(operation)
                if operation == 'snapshot':
                    return dict(owner_id='owner', epoch=0)
                if operation == 'demand_load_and_acquire':
                    ref = dict(acquired=True, lease_id=kwargs['lease_id'], owner_id='owner',
                               adapter_int_id=kwargs['adapter_int_id'])
                    held[ref['lease_id']] = ref
                    return ref
                if operation == 'source_snapshot':
                    return dict(complete_for_native_caches=True,
                                slot_adapter_ids=[r['adapter_int_id'] for r in held.values()])
                if operation == 'release':
                    held.pop(kwargs['lease_id'])
                    return dict(released=True)
                raise AssertionError(operation)

            async def ieee_scheduler_observation(self):
                return dict(admitted=[], unretired_iterations=0, native_deferred_free_batches=0,
                            kv_unreserved_free_blocks=free_blocks, kv_tokens_per_block=16)

            async def generate_prepared(self, **kwargs):
                self_test.assertNotIn('pending_admission_id', kwargs)
                self_test.assertEqual(len(held), 8)
                events.append('generate')
                await asyncio.sleep(0)
                return (100., 50., 64, dict(native_dispatch_monotonic_s=10.,
                    native_first_token_monotonic_s=10.1, native_last_token_monotonic_s=13.25,
                    native_ttft_ms=100., native_tpot_ms=50., native_terminal_observed=True,
                    actual_prompt_tokens=100, native_prompt_token_ids_sha256='b'*64,
                    gpu_reference_adapter_int_id=kwargs['gpu_reference']['adapter_int_id']))

        self_test = self
        result = dict(requests=[])
        adapters = {r['adapter_id']:dict(path='/existing') for r in rows}
        if free_blocks == 0:
            with self.assertRaisesRegex(RuntimeError, 'capacity screen failed'):
                await qualify_warm_reference(Engine(), plan, adapters, index, 'probe', result)
            self.assertNotIn('generate', events)
            self.assertFalse(result['measured_warm_reference'])
        else:
            await qualify_warm_reference(Engine(), plan, adapters, index, 'probe', result)
            self.assertEqual(len(result['requests']), 8)
            self.assertTrue(all(r['pass_native'] for r in result['requests']))
            self.assertFalse(held)
            self.assertGreater(result['warm_batches'][0]['all_decode_intersection_s'], 0.)
            self.assertLess(events.index('source_snapshot'), events.index('generate'))
            self.assertEqual(events.count('generate'), 8)
            self.assertFalse(result['common_reference_frozen'])
            self.assertFalse(result['formal_g1_g2_qualified'])

    async def test_preload_all_before_native_batch_then_drain_and_release(self):
        await self.exercise(512)

    async def test_capacity_failure_is_not_hidden_batch_fallback(self):
        await self.exercise(0)


if __name__ == '__main__':
    unittest.main()
