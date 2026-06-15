import json
import math
import subprocess
import sys
import tempfile
import threading
import unittest
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scripts.build_mooncake_formal_table import build_row
from scripts.verify_mooncake_formal_table import verify_row


ROOT = Path(__file__).resolve().parents[1]


class StreamingHandler(BaseHTTPRequestHandler):
    observed_stream = None

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length))
        type(self).observed_stream = body.get("stream")
        chunks = [
            {
                "choices": [
                    {"delta": {"content": "a"}, "index": 0, "finish_reason": None}
                ]
            },
            {
                "choices": [
                    {"delta": {"content": "b"}, "index": 0, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "total_tokens": 6,
                },
            },
        ]
        payload = "".join(
            f"data: {json.dumps(chunk)}\n\n" for chunk in chunks
        ) + "data: [DONE]\n\n"
        encoded = payload.encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args):
        return


class MooncakeHarnessTests(unittest.TestCase):
    def test_force_stream_replay_exposes_tpot_and_elapsed(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), StreamingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                trace = root / "trace.json"
                output = root / "raw.json"
                trace.write_text(
                    json.dumps(
                        {
                            "requests": [
                                {
                                    "request_id": "req-1",
                                    "arrival_time_s": 0.0,
                                    "expected_input_tokens": 4,
                                    "expected_output_tokens": 2,
                                    "body": {
                                        "model": "fixture",
                                        "prompt": "hello",
                                        "max_tokens": 2,
                                    },
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/replay_openai_trace.py"),
                        "--trace",
                        str(trace),
                        "--base-url",
                        f"http://127.0.0.1:{server.server_port}",
                        "--endpoint-path",
                        "/v1/completions",
                        "--output",
                        str(output),
                        "--force-stream",
                        "--include-stream-usage",
                        "--min-output-tokens",
                        "2",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                payload = json.loads(output.read_text(encoding="utf-8"))
                record = payload["results"][0]
                self.assertTrue(StreamingHandler.observed_stream)
                self.assertTrue(payload["force_stream"])
                self.assertEqual(payload["expected_requests"], 1)
                self.assertEqual(payload["completed_records"], 1)
                self.assertGreaterEqual(payload["elapsed_sec"], 0.0)
                self.assertTrue(record["success"])
                self.assertTrue(record["tpot_observed"])
                self.assertIsNotNone(record["tpot_ms"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_static_lifecycle_cost_includes_client_prewarm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.json"
            summary = root / "summary.json"
            record = {
                "success": True,
                "ttft_ms": 100.0,
                "service_ttft_ms": 90.0,
                "tpot_ms": 10.0,
                "tpot_observed": True,
                "e2e_ms": 200.0,
                "service_e2e_ms": 190.0,
                "dispatch_admission_wait_ms": 10.0,
                "prompt_tokens": 10,
                "completion_tokens": 2,
            }
            raw.write_text(
                json.dumps(
                    {
                        "force_stream": True,
                        "expected_requests": 4000,
                        "elapsed_sec": 90.0,
                        "client_prewarm_sec_excluded_from_workload_clock": 5.0,
                        "ttft_slo_ms": 180.0,
                        "results": [dict(record) for _ in range(4000)],
                    }
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/summarize_mooncake_replay.py"),
                    "--replay",
                    str(raw),
                    "--output",
                    str(summary),
                    "--model",
                    "fixture",
                    "--model-key",
                    "3b",
                    "--gpu-budget",
                    "4",
                    "--startup-sec",
                    "30",
                    "--gpu-cost-per-second-usd",
                    "0.0008",
                    "--tpot-slo-ms",
                    "14",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(summary.read_text(encoding="utf-8"))
            lifecycle = payload["lifecycle_cost"]
            self.assertEqual(lifecycle["infra_gpu_seconds_total"], 500.0)
            self.assertTrue(payload["strict_zero_failure_pass"])
            self.assertTrue(payload["formal_main_comparison_eligible"])
            self.assertTrue(payload["paper_slo_gate_pass"])
            self.assertTrue(
                math.isclose(
                    lifecycle["monetary_cost_per_request_usd"],
                    0.0001,
                )
            )

    def test_build_and_verify_clean_formal_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "formal3b"
            run_dir.mkdir()
            trace = root / "trace.json"
            trace.write_text('{"requests":[]}\n', encoding="utf-8")
            trace_hash = hashlib.sha256(trace.read_bytes()).hexdigest()
            records = [
                {
                    "success": True,
                    "ttft_ms": 100.0,
                    "tpot_ms": 10.0,
                    "tpot_observed": True,
                    "e2e_ms": 200.0,
                }
                for _ in range(4000)
            ]
            (run_dir / "raw_records.json").write_text(
                json.dumps(
                    {
                        "force_stream": True,
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
                        "system": "Mooncake",
                        "trace_role": "formal4000",
                        "max_requests": 4000,
                        "strict_zero_failure_pass": True,
                        "initial_runtime_startup_sec": 30.0,
                        "trace_path": str(trace),
                        "trace_sha256": trace_hash,
                        "mooncake_git_commit": "a" * 40,
                        "vllm_git_commit": "b" * 40,
                        "baseline_harness_git_commit": "c" * 40,
                        "relayserve_git_commit": "d" * 40,
                        "launch_profile": {
                            "topology": "2P+2D",
                            "protocol": "tcp",
                        },
                    }
                ),
                encoding="utf-8",
            )

            row = build_row("3b", run_dir)
            verify_row({key: str(value) for key, value in row.items()})
            self.assertEqual(row["ok_requests"], 4000)
            self.assertEqual(row["observable_tpot_requests"], 4000)
            self.assertEqual(row["system_key"], "mooncake")


if __name__ == "__main__":
    unittest.main()
