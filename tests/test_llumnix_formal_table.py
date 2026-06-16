import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_llumnix_formal_table import build_row
from scripts.verify_llumnix_formal_table import verify_row


class LlumnixFormalTableTests(unittest.TestCase):
    def test_build_and_verify_clean_formal_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "formal3b"
            run_dir.mkdir()
            trace_path = root / "trace.json"
            trace_path.write_text('{"requests":[]}\n', encoding="utf-8")
            trace_hash = hashlib.sha256(trace_path.read_bytes()).hexdigest()
            records = [
                {
                    "success": True,
                    "ttft_ms": 100.0,
                    "tpot_ms": 10.0,
                    "e2e_ms": 200.0,
                }
                for _ in range(4000)
            ]
            (run_dir / "raw_records.json").write_text(
                json.dumps(
                    {
                        "elapsed_sec": 90.0,
                        "client_prewarm_sec_excluded_from_workload_clock": 5.0,
                        "results": records,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "source_summary.json").write_text(
                json.dumps(
                    {
                        "formal_main_comparison_eligible": True,
                        "paper_slo_gate_pass": True,
                        "client_prewarm_sec_excluded_from_workload_clock": 5.0,
                        "lifecycle_cost": {
                            "monetary_ce": 100.0,
                            "monetary_cost_per_request_usd": 0.0001,
                            "monetary_cost_total_usd": 0.4,
                            "infra_gpu_seconds_total": 500.0,
                        },
                        "scenario_summaries": [
                            {
                                "ttft_target_ms": 180.0,
                                "tpot_target_ms": 14.0,
                                "ttft_ms": {"avg": 100.0, "p95": 100.0},
                                "tpot_ms": {"avg": 10.0, "p95": 10.0},
                                "e2e_ms": {"avg": 200.0, "p95": 200.0},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "system": "Llumnix",
                        "trace_role": "formal4000_v1",
                        "max_requests": 4000,
                        "strict_zero_failure_pass": True,
                        "initial_runtime_startup_sec": 30.0,
                        "trace_path": str(trace_path),
                        "trace_sha256": trace_hash,
                        "llumnix_git_commit": "a" * 40,
                        "baseline_harness_git_commit": "b" * 40,
                        "relayserve_git_commit": "c" * 40,
                        "launch_profile": {
                            "initial_instances": 4,
                            "migration_backend": "rayrpc",
                            "enable_routine_migration": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            row = build_row("3b", run_dir)
            verify_row({key: str(value) for key, value in row.items()})
            self.assertEqual(row["ok_requests"], 4000)
            self.assertEqual(row["failed_requests"], 0)
            self.assertEqual(row["system_key"], "llumnix")
            self.assertEqual(row["trace_role"], "formal4000")
            self.assertEqual(row["manifest_trace_role"], "formal4000_v1")
            self.assertEqual(row["llumnix_variant"], "official_routine_migration")
            self.assertEqual(row["routine_migration_enabled"], "true")

    def test_build_and_verify_no_routine_migration_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "formal7b"
            run_dir.mkdir()
            trace_path = root / "trace.json"
            trace_path.write_text('{"requests":[]}\n', encoding="utf-8")
            trace_hash = hashlib.sha256(trace_path.read_bytes()).hexdigest()
            records = [
                {
                    "success": True,
                    "ttft_ms": 300.0,
                    "tpot_ms": 20.0,
                    "e2e_ms": 500.0,
                }
                for _ in range(4000)
            ]
            (run_dir / "raw_records.json").write_text(
                json.dumps(
                    {
                        "elapsed_sec": 100.0,
                        "client_prewarm_sec_excluded_from_workload_clock": 10.0,
                        "results": records,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "source_summary.json").write_text(
                json.dumps(
                    {
                        "formal_main_comparison_eligible": True,
                        "paper_slo_gate_pass": True,
                        "client_prewarm_sec_excluded_from_workload_clock": 10.0,
                        "lifecycle_cost": {
                            "monetary_ce": 50.0,
                            "monetary_cost_per_request_usd": 0.0002,
                            "monetary_cost_total_usd": 0.8,
                            "infra_gpu_seconds_total": 520.0,
                        },
                        "scenario_summaries": [
                            {
                                "ttft_target_ms": 440.0,
                                "tpot_target_ms": 32.0,
                                "ttft_ms": {"avg": 300.0, "p95": 300.0},
                                "tpot_ms": {"avg": 20.0, "p95": 20.0},
                                "e2e_ms": {"avg": 500.0, "p95": 500.0},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "system": "Llumnix",
                        "trace_role": "formal4000_v1",
                        "max_requests": 4000,
                        "strict_zero_failure_pass": True,
                        "initial_runtime_startup_sec": 20.0,
                        "trace_path": str(trace_path),
                        "trace_sha256": trace_hash,
                        "llumnix_git_commit": "a" * 40,
                        "baseline_harness_git_commit": "b" * 40,
                        "relayserve_git_commit": "c" * 40,
                        "launch_profile": {
                            "initial_instances": 4,
                            "migration_backend": "rayrpc",
                            "enable_routine_migration": False,
                        },
                    }
                ),
                encoding="utf-8",
            )

            row = build_row("7b", run_dir)
            verify_row({key: str(value) for key, value in row.items()})
            self.assertEqual(row["trace_role"], "formal4000")
            self.assertEqual(row["manifest_trace_role"], "formal4000_v1")
            self.assertEqual(row["llumnix_variant"], "official_no_routine_migration")
            self.assertEqual(row["routine_migration_enabled"], "false")


if __name__ == "__main__":
    unittest.main()
