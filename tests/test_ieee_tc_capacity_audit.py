"""Retained-event audit only; no fabricated native or continuous observations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_control_path_overhead import (
    analyze_admission_capacity, summarize_admission_capacity,
)


class AdmissionCapacityAuditTest(unittest.TestCase):
    def setUp(self):
        self.snapshot = dict(replica_id='replica1', epoch=1, captured_at=10.,
            admitted=[dict(request_id='r')], kv_tokens_per_block=16,
            kv_bytes_per_block=8388608, kv_unreserved_free_blocks=100,
            scheduled_tokens=2, iteration_token_budget=1024)

    def payload(self, *snapshots):
        return dict(mechanism_events=dict(events=[dict(snapshot=s) for s in snapshots]))

    def test_duplicates_and_type1_median(self):
        other = dict(self.snapshot, captured_at=11., kv_unreserved_free_blocks=80)
        r = summarize_admission_capacity(self.payload(self.snapshot, self.snapshot, other))
        self.assertEqual((r['source_occurrences'], r['distinct_snapshots']), (3, 2))
        self.assertEqual(r['replicas'][0]['free_blocks_p50'], 80)
        self.assertIsNone(r['preemption_total'])
        self.assertIsNone(r['qualified_runtime_capacity'])

    def test_missing_field_rejected(self):
        del self.snapshot['iteration_token_budget']
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            summarize_admission_capacity(self.payload(self.snapshot))

    def test_conflicting_identity_rejected(self):
        other = dict(self.snapshot, kv_unreserved_free_blocks=20)
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            summarize_admission_capacity(self.payload(self.snapshot, other))

    def test_no_samples_is_unknown_not_zero(self):
        with self.assertRaisesRegex(ValueError, 'unknown'):
            summarize_admission_capacity(self.payload())

    def test_invalid_counts_times_and_budget(self):
        for key, value in [('kv_bytes_per_block', 0), ('kv_tokens_per_block', True),
                           ('captured_at', float('nan')), ('scheduled_tokens', 1025),
                           ('kv_unreserved_free_blocks', -1)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                summarize_admission_capacity(self.payload(dict(self.snapshot, **{key: value})))

    def test_population_identity(self):
        other = copy.deepcopy(self.snapshot)
        other['admitted'] *= 2
        with self.assertRaisesRegex(ValueError, 'duplicate admitted'):
            summarize_admission_capacity(self.payload(other))

    def test_geometry_must_remain_fixed(self):
        other = dict(self.snapshot, captured_at=11., kv_tokens_per_block=32)
        with self.assertRaisesRegex(ValueError, 'geometry'):
            summarize_admission_capacity(self.payload(self.snapshot, other))

    def test_explicit_new_output_and_source_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.json'
            source.write_text(json.dumps(self.payload(self.snapshot)))
            result = analyze_admission_capacity(source, root/'out')
            self.assertEqual(len(result['source_ref']['sha256']), 64)
            self.assertTrue((root/'out/capacity_by_replica.csv').exists())
            with self.assertRaises(FileExistsError):
                analyze_admission_capacity(source, root/'out')


if __name__ == '__main__':
    unittest.main()
