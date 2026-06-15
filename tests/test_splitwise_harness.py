import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.prepare_splitwise_trace import convert_requests
from scripts.summarize_splitwise_sim import main as summarize_main


ROOT = Path(__file__).resolve().parents[1]


class SplitwiseHarnessTests(unittest.TestCase):
    def test_trace_conversion_preserves_arrival_and_token_shapes(self):
        rows = convert_requests(
            [
                {
                    "request_id": "a",
                    "arrival_time_s": 0.0,
                    "expected_input_tokens": 18,
                    "expected_output_tokens": 14,
                },
                {
                    "request_id": "b",
                    "arrival_time_s": 1.25,
                    "expected_input_tokens": 31,
                    "expected_output_tokens": 9,
                },
            ]
        )
        self.assertEqual(rows[0]["request_type"], 2)
        self.assertEqual(rows[0]["arrival_timestamp"], 1.0)
        self.assertEqual(rows[1]["arrival_timestamp"], 2.25)
        self.assertEqual(rows[0]["prompt_size"], 18)
        self.assertEqual(rows[0]["token_size"], 14)

    def test_trace_conversion_rejects_reordered_arrivals(self):
        with self.assertRaisesRegex(ValueError, "arrival order regressed"):
            convert_requests(
                [
                    {
                        "arrival_time_s": 2.0,
                        "expected_input_tokens": 1,
                        "expected_output_tokens": 1,
                    },
                    {
                        "arrival_time_s": 1.0,
                        "expected_input_tokens": 1,
                        "expected_output_tokens": 1,
                    },
                ],
                arrival_offset_s=1.0,
            )

    def test_summary_is_never_promoted_to_measured_main_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            detailed = root / "0.csv"
            manifest = root / "manifest.json"
            output = root / "summary.json"
            with detailed.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "ttft_times",
                        "tbt_times",
                        "response_times",
                        "queue_times",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "ttft_times": 0.1,
                        "tbt_times": 0.01,
                        "response_times": 0.2,
                        "queue_times": 0.0,
                    }
                )
            manifest.write_text(
                json.dumps({"trace_role": "smoke"}),
                encoding="utf-8",
            )
            argv = [
                "summarize_splitwise_sim.py",
                "--detailed",
                str(detailed),
                "--manifest",
                str(manifest),
                "--output",
                str(output),
                "--model-key",
                "3b",
                "--expected-requests",
                "1",
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(summarize_main(), 0)
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(summary["formal_main_comparison_eligible"])
            self.assertEqual(
                summary["runtime_class"],
                "official_discrete_event_simulator",
            )
            self.assertIsNone(summary["ce"])

    def test_runner_freezes_official_profile_and_refuses_overwrite(self):
        runner = (
            ROOT / "scripts/run_splitwise_official_sim.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("llama2-70b-fp16", runner)
        self.assertIn("cluster=half_half", runner)
        self.assertIn("start_state=splitwise", runner)
        self.assertIn("applications.0.scheduler=kv_token_jsq", runner)
        self.assertIn("formal_main_comparison_eligible", runner)
        self.assertIn("refusing to overwrite", runner)


if __name__ == "__main__":
    unittest.main()
