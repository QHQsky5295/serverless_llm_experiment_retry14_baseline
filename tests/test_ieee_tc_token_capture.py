import unittest

from scripts.run_all_experiments import (
    _maybe_attach_opt_in_completion_token_ids,
    _preserve_native_timing_value,
)


class OptInCompletionTokenCaptureTests(unittest.TestCase):
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
