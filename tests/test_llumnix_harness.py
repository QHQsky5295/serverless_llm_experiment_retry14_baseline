import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.replay_llumnix_trace import parse_token_latencies, percentile
from scripts.summarize_llumnix_replay import main as summarize_main
from scripts.validate_llumnix_service_health import build_report, classify

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
                        "client_prewarm_sec_excluded_from_workload_clock": 2.5,
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
            self.assertEqual(cost["infra_gpu_seconds_total"], 50.0)
            self.assertTrue(
                math.isclose(cost["monetary_cost_per_request_usd"], 0.0000125)
            )
            self.assertTrue(
                math.isclose(cost["monetary_ce"], 533333.3333333334)
            )
            self.assertEqual(
                summary["client_prewarm_sec_excluded_from_workload_clock"],
                2.5,
            )
            self.assertEqual(
                summary["scenario_summaries"][0]["deployment_duration_sec"],
                12.5,
            )
            self.assertEqual(
                summary["scenario_summaries"][0]["client_prewarm_sec"],
                2.5,
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
        self.assertIn('PATH="${ENV_DIR}/bin:${PATH}"', runner)
        self.assertIn("LLUMNIX_INIT_INSTANCES_TIMEOUT", runner)
        self.assertIn("LLUMNIX_INIT_WORKER_RPC_TIMEOUT", runner)
        self.assertIn("LLUMNIX_SCALE_UP_RPC_TIMEOUT", runner)
        self.assertIn("LLUMNIX_INSTANCE_READY_TIMEOUT", runner)
        self.assertIn("LLUMNIX_UTILITY_CALL_TIMEOUT", runner)
        self.assertIn("LLUMNIX_SERVICE_STABILIZATION_S", runner)
        self.assertIn("LLUMNIX_FULL_PATH_PROBE_TIMEOUT_S", runner)
        self.assertIn("LLUMNIX_FULL_PATH_PROBE_ATTEMPTS", runner)
        self.assertIn("apply_llumnix_relayserve_patch.sh", runner)
        self.assertIn("probe_llumnix_service.py", runner)
        self.assertIn("validate_llumnix_service_health.py", runner)
        self.assertIn('"observed_ready_instances"', runner)
        self.assertIn("--enable-routine-migration", runner)
        self.assertIn("MIN_AVAILABLE_MEMORY_GB", runner)
        self.assertIn("MAX_GPU_TEMPERATURE_C", runner)
        self.assertIn("ray\" stop --force", runner)
        self.assertIn("refusing to overwrite existing Llumnix run directory", runner)

    def test_compat_patch_propagates_declared_llumnix_env_vars(self):
        patch = (
            ROOT / "patches/llumnix_relayserve_compat.patch"
        ).read_text(encoding="utf-8")
        self.assertIn('key.removeprefix("LLUMNIX_")', patch)
        self.assertIn("normalized_key in llumnix_env_vars_keys", patch)
        self.assertIn('or key.startswith("LLUMNIX_")', patch)
        self.assertEqual(
            patch.count('runtime_env={"env_vars": get_llumnix_env_vars()}'),
            2,
        )

    def test_service_health_rejects_degraded_startup(self):
        text = "\n".join(
            [
                "Init Llumnix components done, 4 instances are ready",
                "global_scheduler.py:191] num_instances: 1",
                "Failed to scale up instance deadbeef",
                "Client received request request-1",
            ]
        )
        report = build_report(text, expected=4, phase="final")
        exit_code, _ = classify(report)
        self.assertEqual(exit_code, 20)

    def test_service_health_accepts_four_instance_run(self):
        text = "\n".join(
            [
                "Init Llumnix components done, 4 instances are ready",
                "global_scheduler.py:191] num_instances: 4",
                "Client received request request-1",
            ]
        )
        report = build_report(text, expected=4, phase="final")
        exit_code, _ = classify(report)
        self.assertEqual(exit_code, 0)

    def test_service_health_rejects_output_forwarder_failure(self):
        text = "\n".join(
            [
                "Init Llumnix components done, 4 instances are ready",
                "global_scheduler.py:191] num_instances: 4",
                "Client received request request-1",
                "Failed to send one way rpc request",
                "Unable to put items into queue",
                "output_forwarder.py:87] Server deadbeef is dead",
            ]
        )
        report = build_report(text, expected=4, phase="final")
        exit_code, _ = classify(report)
        self.assertEqual(exit_code, 23)

    def test_service_health_scopes_failures_to_measured_runtime(self):
        startup = "\n".join(
            [
                "Init Llumnix components done, 4 instances are ready",
                "global_scheduler.py:191] num_instances: 4",
                "Client received request readiness-probe",
                "Failed to send one way rpc request",
            ]
        ) + "\n"
        runtime = "\n".join(
            [
                "Client received request formal-request",
                "Engine finished request formal-request",
            ]
        )
        text = startup + runtime
        report = build_report(
            text,
            expected=4,
            phase="final",
            runtime_offset=len(startup.encode("utf-8")),
        )
        exit_code, _ = classify(report)
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["runtime_failures"], [])


if __name__ == "__main__":
    unittest.main()
