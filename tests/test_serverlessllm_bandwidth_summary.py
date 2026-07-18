from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import summarize_serverlessllm_replay as summary  # noqa: E402


class ServerlessLLMBandwidthSummaryTest(unittest.TestCase):
    def test_aggregate_reservation_fields(self) -> None:
        replay = {
            "remote_artifact_bandwidth": {
                "request_path": {
                    "limit_mode": "file_aggregate_reservation",
                    "configured_bandwidth_mib_s": 10.0,
                }
            }
        }
        results = [
            {
                "remote_lora_fetched": True,
                "remote_lora_bandwidth_limit_mode": "file_aggregate_reservation",
                "remote_lora_bandwidth_configured_mib_s": 10.0,
                "remote_lora_bandwidth_bytes": 1024 * 1024,
                "remote_lora_bandwidth_reserved_transfer_ms": 100.0,
                "remote_lora_bandwidth_wait_ms": wait_ms,
            }
            for wait_ms in (100.0, 200.0)
        ]
        observed = summary._summarize_aggregate_bandwidth(replay, results)
        self.assertEqual(observed["limit_mode"], "file_aggregate_reservation")
        self.assertEqual(observed["configured_mib_s"], 10.0)
        self.assertAlmostEqual(observed["configured_gbit_s"], 0.08388608)
        self.assertEqual(observed["transfer_count"], 2)
        self.assertEqual(observed["total_bytes"], 2 * 1024 * 1024)
        self.assertAlmostEqual(observed["reservation_span_s"], 0.2)
        self.assertAlmostEqual(observed["total_injected_wait_s"], 0.3)
        self.assertAlmostEqual(observed["achieved_reserved_mib_s"], 10.0)

    def test_no_delay_and_http_modes_are_explicit(self) -> None:
        no_delay = summary._summarize_aggregate_bandwidth(
            {
                "remote_artifact_bandwidth": {
                    "request_path": {
                        "limit_mode": "file_no_delay",
                        "configured_bandwidth_mib_s": 0.0,
                    }
                }
            },
            [],
        )
        self.assertEqual(no_delay["limit_mode"], "file_no_delay")
        self.assertEqual(no_delay["configured_mib_s"], 0.0)
        self.assertEqual(no_delay["achieved_reserved_mib_s"], 0.0)

        http = summary._summarize_aggregate_bandwidth(
            {
                "remote_artifact_bandwidth": {
                    "request_path": {
                        "limit_mode": "http_unthrottled",
                        "requested_bandwidth_mib_s": 119.2093,
                        "configured_bandwidth_mib_s": 0.0,
                    }
                }
            },
            [],
        )
        self.assertEqual(http["limit_mode"], "http_unthrottled")
        self.assertEqual(http["configured_mib_s"], 0.0)

    def test_summary_writes_bandwidth_and_real_file_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.yaml"
            trace = root / "trace.json"
            subset = root / "subset.json"
            runtime_subset = root / "runtime_subset.json"
            replay = root / "replay.json"
            deploy = root / "deploy.json"
            output = root / "summary.json"
            config.write_text(
                """
cost_model:
  gpu_cost_per_second_usd: 0.001
model_profiles:
  test_model:
    model:
      name: test/model
      tensor_parallel_size: 1
dataset_profiles:
  test_dataset:
    datasets:
      source: synthetic
workload_profiles:
  test_workload:
    workload:
      ttft_slo_ms: 5000
""".strip()
                + "\n",
                encoding="utf-8",
            )
            trace.write_text(
                json.dumps({"total_requests": 1, "selected_num_adapters": 1}),
                encoding="utf-8",
            )
            subset.write_text(json.dumps({"adapters": [{"id": "a0"}]}), encoding="utf-8")
            runtime_subset.write_text(
                json.dumps(
                    {
                        "adapters": [{"id": "a0"}],
                        "remote_dir": str(root / "runtime_cache"),
                        "remote_materialization": {"mode": "request_path_dynamic"},
                    }
                ),
                encoding="utf-8",
            )
            deploy.write_text(
                json.dumps(
                    {
                        "base_urls": ["http://127.0.0.1:1"],
                        "tensor_parallel_size": 1,
                        "data_parallel_replicas": 1,
                    }
                ),
                encoding="utf-8",
            )
            request = {
                "request_id": "req_0",
                "success": True,
                "ttft_ms": 10.0,
                "e2e_ms": 20.0,
                "service_ttft_ms": 10.0,
                "service_e2e_ms": 20.0,
                "dispatch_admission_wait_ms": 0.0,
                "tpot_ms": 5.0,
                "tpot_observed": True,
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
                "cost_usd": 0.001,
                "completion_offset_s": 0.02,
                "target_base_url": "http://127.0.0.1:1",
                "remote_lora_fetched": True,
                "remote_lora_bandwidth_limit_mode": "file_aggregate_reservation",
                "remote_lora_bandwidth_configured_mib_s": 10.0,
                "remote_lora_bandwidth_bytes": 1024 * 1024,
                "remote_lora_bandwidth_reserved_transfer_ms": 100.0,
                "remote_lora_bandwidth_wait_ms": 100.0,
            }
            replay.write_text(
                json.dumps(
                    {
                        "elapsed_sec": 0.02,
                        "remote_artifact_bandwidth": {
                            "request_path": {
                                "limit_mode": "file_aggregate_reservation",
                                "configured_bandwidth_mib_s": 10.0,
                            }
                        },
                        "results": [request],
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "summarize_serverlessllm_replay.py",
                "--main-repo",
                str(root),
                "--config",
                str(config),
                "--replay",
                str(replay),
                "--trace",
                str(trace),
                "--adapter-subset",
                str(subset),
                "--runtime-adapter-subset",
                str(runtime_subset),
                "--deploy",
                str(deploy),
                "--model-profile",
                "test_model",
                "--dataset-profile",
                "test_dataset",
                "--workload-profile",
                "test_workload",
                "--output",
                str(output),
                "--baseline-type",
                "vllm",
                "--instance-mode",
                "static_runtime",
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(summary.main(), 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            expected_trace_sha = hashlib.sha256(trace.read_bytes()).hexdigest()
            expected_subset_sha = hashlib.sha256(subset.read_bytes()).hexdigest()
            expected_runtime_subset_sha = hashlib.sha256(runtime_subset.read_bytes()).hexdigest()
            expected_subset_path = str(subset.resolve())
            expected_runtime_subset_path = str(runtime_subset.resolve())

        metadata = payload["metadata"]
        self.assertEqual(metadata["bandwidth_mib_s"], 10.0)
        self.assertAlmostEqual(metadata["bandwidth_gbit_s"], 0.08388608)
        self.assertEqual(metadata["bandwidth_limit_mode"], "file_aggregate_reservation")
        self.assertEqual(metadata["aggregate_bandwidth"]["transfer_count"], 1)
        self.assertEqual(
            metadata["shared_trace_sha256"],
            expected_trace_sha,
        )
        self.assertEqual(
            metadata["shared_adapter_subset_sha256"],
            expected_subset_sha,
        )
        self.assertEqual(metadata["adapter_subset_path"], expected_subset_path)
        self.assertEqual(metadata["shared_adapter_subset_path"], expected_subset_path)
        self.assertEqual(
            metadata["runtime_adapter_subset_path"], expected_runtime_subset_path
        )
        self.assertEqual(
            metadata["runtime_adapter_subset_sha256"],
            expected_runtime_subset_sha,
        )
        self.assertTrue(metadata["runtime_adapter_subset_rewritten"])


if __name__ == "__main__":
    unittest.main()
