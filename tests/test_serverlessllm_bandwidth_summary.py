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
    def test_tc_native_audit_retains_failures_and_rejects_live_duplicate_counts(self):
        contract = dict(event='request_contract', request_id='r', adapter_id='a',
                        target_tokens=2, native_prompt_token_ids_sha256='input',
                        canonical_prompt_sha256='prompt', source_item_sha256='source')
        arrival = dict(event='request_created', request_id='r', planned_arrival_s=10.,
                       source_item_sha256='source')
        failed = dict(event='http_request_failed', request_id='r', error='HTTP status 500')
        final = dict(event='http_replay_complete', N_plan=1, N_arrived=1,
                     N_terminal=1, N_response=0, N_failed=1)
        events = [dict(event='replay_ready'), contract, arrival, failed, final]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'journal.jsonl'

            def audit(rows):
                path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                return summary.audit_tc_http_journal(path)

            actual = audit(events)
            self.assertFalse(actual['workload_passed'])
            self.assertTrue(actual['measurement_complete'])
            self.assertEqual(actual['counts']['N_failed'], 1)
            self.assertIsNone(actual['conditional_metrics']['ttft_ms']['mean'])
            self.assertEqual(actual['diagnostic_rows'][0]['error'], 'HTTP status 500')
            for corrupt in (events[:-1], events[:-1]+[failed, final],
                            events[:-1]+[dict(final, N_response=1)],
                            events[:2]+[final], events+[arrival]):
                with self.assertRaises(ValueError):
                    audit(corrupt)

    def test_tc_native_audit_recomputes_timing_and_never_claims_numeric_correctness(self):
        contract = dict(event='request_contract', request_id='r', adapter_id='a',
                        target_tokens=2, native_prompt_token_ids_sha256='input',
                        canonical_prompt_sha256='prompt', source_item_sha256='source')
        arrival = dict(event='request_created', request_id='r', planned_arrival_s=10.,
                       source_item_sha256='source')
        timing = dict(protocol_valid=True, ttft_ms=6., e2e_ms=15., submit_lag_ms=1.,
                      dispatch_wait_after_submit_ms=2., service_ttft_ms=3.,
                      decode_ms=4., completion_notification_ms=5., router_queue_ms=2.,
                      tpot_ms=4., instance_assigned_s=10.003, ready_instances_at_enqueue=1)
        observed = dict(native_output_tokens=2, native_lora_name='a',
                        native_prompt_token_ids_sha256='input',
                        control_observation=dict(instance_id='native'))
        raw = dict(event='http_raw_response', request_id='r', status=200,
                   body=dict(metrics=dict(ieee_tc=observed)))
        events = [dict(event='replay_ready'), contract, arrival,
                  dict(event='http_headers_sent', request_id='r'), raw,
                  dict(event='http_response', request_id='r', response=timing),
                  dict(event='http_replay_complete', N_plan=1, N_arrived=1,
                       N_terminal=1, N_response=1, N_failed=0)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'journal.jsonl'

            def audit():
                path.write_text(''.join(json.dumps(row)+'\n' for row in events))
                return summary.audit_tc_http_journal(path)

            actual = audit()
            self.assertTrue(actual['workload_passed'])
            self.assertFalse(actual['formal_performance_qualified'])
            self.assertFalse(actual['independent_lora_numerical_correctness'])
            self.assertEqual(actual['actual_native_output_tokens'], 2)
            for field, value in [('e2e_ms', 17.), ('tpot_ms', 7.), ('ttft_ms', float('nan'))]:
                saved = timing[field]
                timing[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    audit()
                timing[field] = saved
            observed['native_lora_name'] = 'wrong'
            with self.assertRaises(ValueError):
                audit()

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
