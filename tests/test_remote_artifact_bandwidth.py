from __future__ import annotations

import concurrent.futures
import io
import json
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from replay_openai_trace import RemoteArtifactFetcher  # noqa: E402
from run_remote_artifact_bandwidth_microtest import (  # noqa: E402
    _create_adapter,
    run_case,
    select_actual_adapter_groups,
)


class RemoteArtifactBandwidthTest(unittest.TestCase):
    def test_actual_adapter_groups_are_size_ordered_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(12):
                _create_adapter(root, f"adapter_{index:02d}", (index + 1) * 1024)
            groups = select_actual_adapter_groups(root, concurrency=4)
            selected = groups["small"] + groups["medium"] + groups["large"]
            self.assertEqual(len(selected), len(set(selected)))
            sizes = {
                label: [
                    (root / adapter_id / "adapter_model.bin").stat().st_size
                    for adapter_id in adapter_ids
                ]
                for label, adapter_ids in groups.items()
            }
            self.assertLess(max(sizes["small"]), min(sizes["medium"]))
            self.assertLess(max(sizes["medium"]), min(sizes["large"]))

    def test_four_concurrent_fetches_share_one_aggregate_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_case(
                label="small",
                size_mib=0.0625,
                bandwidth_mib_s=4.0,
                concurrency=4,
                repeat=1,
                workspace=Path(tmp),
            )
        self.assertTrue(result["gate_pass"])
        self.assertEqual(result["limit_modes"], ["file_aggregate_reservation"])
        self.assertGreaterEqual(
            result["wall_s"] + 0.005,
            result["expected_min_wall_s"],
        )
        self.assertLessEqual(result["aggregate_achieved_mib_s"], 4.5)
        self.assertGreater(result["total_injected_wait_ms"], 0.0)

    def test_zero_bandwidth_means_file_no_delay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_case(
                label="small",
                size_mib=0.015625,
                bandwidth_mib_s=0.0,
                concurrency=4,
                repeat=1,
                workspace=Path(tmp),
            )
        self.assertTrue(result["gate_pass"])
        self.assertEqual(result["expected_min_wall_s"], 0.0)
        self.assertEqual(result["limit_modes"], ["file_no_delay"])
        self.assertEqual(result["total_injected_wait_ms"], 0.0)

    def test_separate_fetcher_instances_share_process_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            remote = workspace / "remote"
            target = workspace / "target"
            remote.mkdir()
            target.mkdir()
            adapter_ids = [f"adapter_{index}" for index in range(4)]
            for adapter_id in adapter_ids:
                _create_adapter(remote, adapter_id, 64 * 1024)
            fetchers = [
                RemoteArtifactFetcher(
                    endpoint=remote.resolve().as_uri(),
                    timeout_s=10.0,
                    token_env="PRIME_REMOTE_TOKEN",
                    bandwidth_mib_s=4.0,
                )
                for _ in adapter_ids
            ]
            barrier = threading.Barrier(4)

            def fetch(index: int):
                barrier.wait(timeout=5.0)
                return fetchers[index].ensure(
                    adapter_ids[index],
                    str(target / adapter_ids[index]),
                )

            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                metrics = list(pool.map(fetch, range(4)))
            wall_s = time.perf_counter() - started

        total_bytes = sum(int(item["remote_lora_bytes"]) for item in metrics)
        expected_s = (total_bytes / (1024 * 1024)) / 4.0
        self.assertGreaterEqual(wall_s + 0.005, expected_s)
        self.assertTrue(
            all(
                item["remote_lora_bandwidth_limit_mode"]
                == "file_aggregate_reservation"
                for item in metrics
            )
        )

    def test_legacy_mbps_name_remains_mib_per_second(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            endpoint = Path(tmp).resolve().as_uri()
            fetcher = RemoteArtifactFetcher(
                endpoint=endpoint,
                timeout_s=10.0,
                token_env="PRIME_REMOTE_TOKEN",
                bandwidth_mbps=12.5,
            )
        metadata = fetcher.configuration_metadata()
        self.assertEqual(metadata["requested_bandwidth_mib_s"], 12.5)
        self.assertEqual(metadata["configured_bandwidth_mib_s"], 12.5)
        self.assertEqual(metadata["limit_mode"], "file_aggregate_reservation")

    def test_http_metadata_is_explicitly_unthrottled(self) -> None:
        fetcher = RemoteArtifactFetcher(
            endpoint="https://artifact.example.test",
            timeout_s=10.0,
            token_env="PRIME_REMOTE_TOKEN",
            bandwidth_mib_s=12.5,
        )
        metadata = fetcher.configuration_metadata()
        self.assertEqual(metadata["requested_bandwidth_mib_s"], 12.5)
        self.assertEqual(metadata["configured_bandwidth_mib_s"], 0.0)
        self.assertEqual(metadata["limit_mode"], "http_unthrottled")

    def test_http_fetch_records_wire_rate_without_injected_wait(self) -> None:
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as tar:
            config = json.dumps({"r": 8}).encode("utf-8")
            config_info = tarfile.TarInfo("adapter_config.json")
            config_info.size = len(config)
            tar.addfile(config_info, io.BytesIO(config))
            weights = b"x" * 4096
            weights_info = tarfile.TarInfo("adapter_model.bin")
            weights_info.size = len(weights)
            tar.addfile(weights_info, io.BytesIO(weights))
        archive_bytes = archive.getvalue()

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def raise_for_status(self):
                return None

            def iter_content(self, **_kwargs):
                yield archive_bytes

        with tempfile.TemporaryDirectory() as tmp:
            fetcher = RemoteArtifactFetcher(
                endpoint="https://artifact.example.test",
                timeout_s=10.0,
                token_env="PRIME_REMOTE_TOKEN",
                bandwidth_mib_s=0.001,
            )
            fetcher._session.get = lambda *_args, **_kwargs: FakeResponse()
            metrics = fetcher.ensure("adapter_0", str(Path(tmp) / "adapter_0"))

        self.assertTrue(metrics["remote_lora_fetched"])
        self.assertEqual(metrics["remote_lora_bandwidth_limit_mode"], "http_unthrottled")
        self.assertEqual(metrics["remote_lora_bandwidth_requested_mib_s"], 0.001)
        self.assertEqual(metrics["remote_lora_bandwidth_configured_mib_s"], 0.0)
        self.assertEqual(metrics["remote_lora_bandwidth_bytes"], len(archive_bytes))
        self.assertEqual(metrics["remote_lora_bandwidth_wait_ms"], 0.0)
        self.assertEqual(
            metrics["remote_lora_bandwidth_achieved_basis"],
            "wire_bytes_over_fetch_wall",
        )
        self.assertGreater(metrics["remote_lora_bandwidth_achieved_mib_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
