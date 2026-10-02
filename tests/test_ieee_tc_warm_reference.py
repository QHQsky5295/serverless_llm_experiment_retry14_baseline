"""No-GPU V1 warm-reference indexing tests; never generate experiment inputs."""
import copy
import json
import unittest
from types import SimpleNamespace

from scripts.ieee_tc_preflight import warm_reference_input_index


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


if __name__ == '__main__':
    unittest.main()
