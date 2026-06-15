import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.replay_llumnix_trace import parse_token_latencies, percentile
from scripts.summarize_llumnix_replay import main as summarize_main

ROOT = Path(__file__).resolve().parents[1]


class LlumnixHarnessTests(unittest.TestCase):
    def test_parse_token_latencies_matches_official_benchmark_semantics(self):
        ttft, tpot, e2e = parse_token_latencies(
            {
                "per_token_latency": [
                    [10.0, 120.0],
                    [10.01, 10.0],
                    [10.02, 14.0],
                ]
            }
        )
        self.assertEqual(ttft, 120.0)
        self.assertEqual(tpot, 12.0)
        self.assertEqual(e2e, 144.0)

    def test_parse_single_token_response_has_zero_decode_tpot(self):
        ttft, tpot, e2e = parse_token_latencies(
            {"per_token_latency": [[10.0, 90.0]]}
        )
        self.assertEqual(ttft, 90.0)
        self.assertEqual(tpot, 0.0)
        self.assertEqual(e2e, 90.0)

    def test_parse_token_latencies_rejects_missing_samples(self):
        with self.assertRaisesRegex(ValueError, "missing per_token_latency"):
            parse_token_latencies({})

    def test_percentile_interpolates(self):
        self.assertTrue(math.isclose(percentile([1.0, 2.0, 3.0], 0.95), 2.9))

    def test_summary_uses_static_four_gpu_lifecycle_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay_path = root / "replay.json"
            output_path = root / "summary.json"
            record = {
                "success": True,
                "ttft_ms": 100.0,
                "service_ttft_ms": 90.0,
                "tpot_ms": 10.0,
                "e2e_ms": 150.0,
                "service_e2e_ms": 140.0,
                "dispatch_wait_ms": 10.0,
                "prompt_tokens": 32,
                "completion_tokens": 8,
            }
            replay_path.write_text(
                json.dumps(
                    {
                        "expected_requests": 4000,
                        "elapsed_sec": 9.0,
                        "ttft_slo_ms": 180.0,
                        "tpot_slo_ms": 14.0,
                        "results": [dict(record) for _ in range(4000)],
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                "summarize_llumnix_replay.py",
                "--replay",
                str(replay_path),
                "--output",
                str(output_path),
                "--model",
                "test-3b",
                "--model-key",
                "3b",
                "--gpu-budget",
                "4",
                "--startup-sec",
                "1",
                "--gpu-cost-per-second-usd",
                "0.001",
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(summarize_main(), 0)

            summary = json.loads(output_path.read_text(encoding="utf-8"))
            cost = summary["lifecycle_cost"]
            self.assertEqual(cost["infra_gpu_seconds_total"], 40.0)
            self.assertTrue(
                math.isclose(cost["monetary_cost_per_request_usd"], 0.00001)
            )
            self.assertTrue(
                math.isclose(cost["monetary_ce"], 666666.6666666666)
            )
            self.assertTrue(summary["strict_zero_failure_pass"])
            self.assertTrue(summary["formal_main_comparison_eligible"])
            self.assertTrue(summary["paper_slo_gate_pass"])

    def test_runner_freezes_official_migration_and_resource_guards(self):
        runner = (
            ROOT / "scripts/run_llumnix_relayserve_continuation.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--initial-instances \"${INITIAL_INSTANCES}\"", runner)
        self.assertIn("--migration-backend rayrpc", runner)
        self.assertIn("--enable-routine-migration", runner)
        self.assertIn("MIN_AVAILABLE_MEMORY_GB", runner)
        self.assertIn("MAX_GPU_TEMPERATURE_C", runner)
        self.assertIn("ray\" stop --force", runner)
        self.assertIn("refusing to overwrite existing Llumnix run directory", runner)


if __name__ == "__main__":
    unittest.main()
