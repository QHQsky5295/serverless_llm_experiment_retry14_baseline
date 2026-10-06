import unittest
import copy
import hashlib
import json

from scripts.run_all_experiments import (
    _maybe_attach_opt_in_completion_token_ids,
    _preserve_native_timing_value,
    _validate_diagnostic_logprobs,
)
from scripts.ieee_tc_preflight import compare_native_sequences, numerical_repeat_indices


class OptInCompletionTokenCaptureTests(unittest.TestCase):
    def test_probability_transport_rejects_missing_nonfinite_and_noncanonical_ids(self):
        value = [{'1': -0.1, '2': -2.0}]
        self.assertEqual(_preserve_native_timing_value('completion_top_logprobs', value), value)
        for bad in ([], [None], [{'1': 0.}], [{'1': 0., '2': float('nan')}],
                    [{'1': True, '2': -1.}], [{'01': -.1, '2': -1.}]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _validate_diagnostic_logprobs(bad)

    @staticmethod
    def case(ids=(1, 2)):
        return dict(source_row_sha256='row', adapter_id='a', target_tokens=2,
            prompt_sha256='prompt', timing=dict(completion_token_ids=list(ids),
            completion_token_ids_sha256=hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest(),
            completion_token_ids_capture_contract='fixed_length_greedy_v1_native_ids_opt_in_v1',
            completion_top_logprobs_contract='native_top2_plus_sampled_v1',
            completion_top_logprobs=[{'1': -.1, '2': -.2}, {'1': -.2, '2': -.1}],
            native_prompt_token_ids_sha256='native_prompt'))

    def test_exact_sequence_is_not_correctness_or_causal_proof(self):
        result = compare_native_sequences(self.case(), self.case())
        self.assertTrue(result['exact_match'])
        self.assertIsNone(result['first_divergence_zero_based'])
        self.assertIsNone(result['n_correct'])
        self.assertFalse(result['cause_established'])

    def test_first_divergence_retains_original_probabilities(self):
        left, right = self.case(), self.case((1, 1))
        result = compare_native_sequences(left, right)
        self.assertEqual(result['first_divergence_zero_based'], 1)
        self.assertEqual(result['shared_prefix_tokens'], 1)
        self.assertEqual(result['at_first_divergence'][1]['sampled_token_id'], 1)
        self.assertAlmostEqual(result['at_first_divergence'][0]['top_two_logprob_margin'], .1)

    def test_comparison_rejects_contract_identity_and_capture_damage(self):
        for mutate in (
            lambda c: c.update(adapter_id='wrong'),
            lambda c: c['timing'].update(native_prompt_token_ids_sha256='wrong'),
            lambda c: c['timing'].update(completion_token_ids_sha256='wrong'),
            lambda c: c['timing'].update(completion_token_ids=[True, 2]),
            lambda c: c['timing'].update(completion_top_logprobs=[{'1': -.1, '2': -.2}]),
            lambda c: c['timing'].update(completion_top_logprobs=[{'3': -.1, '4': -.2}]*2),
        ):
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                damaged = copy.deepcopy(self.case())
                mutate(damaged)
                compare_native_sequences(self.case(), damaged)

    def test_registered_existing_indices_cover_all_six_difference_rows(self):
        selected = numerical_repeat_indices(100)
        self.assertEqual(len(selected), 27)
        self.assertEqual(len(set(selected)), 27)
        self.assertTrue({5, 59, 74, 77, 84, 91}.issubset(selected))
        for count in (True, 27, 99, 101):
            with self.assertRaises(ValueError):
                numerical_repeat_indices(count)

    def test_default_path_keeps_digest_only_schema(self):
        timing = {"completion_token_ids_sha256": "a" * 64}
        _maybe_attach_opt_in_completion_token_ids(timing, {}, [1, 2, 3])
        self.assertEqual(timing, {"completion_token_ids_sha256": "a" * 64})

    def test_opt_in_records_native_integer_ids_and_contract(self):
        timing = {"completion_token_ids_sha256": "b" * 64}
        _maybe_attach_opt_in_completion_token_ids(
            timing, {"ieee_token_id_capture": True}, [0, 17, 32000]
        )
        self.assertEqual(timing["completion_token_ids"], [0, 17, 32000])
        self.assertEqual(
            timing["completion_token_ids_capture_contract"],
            "fixed_length_greedy_v1_native_ids_opt_in_v1",
        )

    def test_opt_in_rejects_text_or_negative_substitutes(self):
        for ids in ([1, -1], [1, "2"]):
            with self.subTest(ids=ids):
                with self.assertRaisesRegex(ValueError, "token capture"):
                    _maybe_attach_opt_in_completion_token_ids(
                        {}, {"ieee_token_id_capture": True}, ids
                    )

    def test_dedicated_rpc_preserves_only_valid_native_id_list(self):
        self.assertEqual(
            _preserve_native_timing_value("completion_token_ids", [0, 17, 32000]),
            [0, 17, 32000],
        )
        self.assertEqual(_preserve_native_timing_value("native_ttft_ms", 12.5), 12.5)
        with self.assertRaisesRegex(ValueError, "token capture"):
            _preserve_native_timing_value("completion_token_ids", 0.0)
        with self.assertRaisesRegex(ValueError, "token capture"):
            _preserve_native_timing_value("completion_token_ids", [1, -2])


if __name__ == "__main__":
    unittest.main()
