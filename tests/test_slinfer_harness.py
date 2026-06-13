from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SlinferHarnessTest(unittest.TestCase):
    def test_gpu_lifecycle_classifies_startup_active_idle(self) -> None:
        module = _load_script("summarize_slinfer_replay.py")
        logs = {
            "node_usage": {"gpu": [0, 1, 1, 1, 0]},
            "node_density": {
                "gpu": [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            },
            "batch": {
                "gpu": [
                    [[], [], [], []],
                    [[0], [0], [0], [0]],
                    [[1], [0], [0], [0]],
                    [[0], [0], [0], [0]],
                    [[], [], [], []],
                ]
            },
        }
        lifecycle = module._gpu_lifecycle(logs)
        self.assertEqual(lifecycle["startup_gpu_seconds"], 1.0)
        self.assertEqual(lifecycle["active_gpu_seconds"], 1.0)
        self.assertEqual(lifecycle["idle_gpu_seconds"], 1.0)
        self.assertEqual(lifecycle["allocated_gpu_seconds"], 3.0)
        self.assertEqual(lifecycle["peak_allocated_gpus"], 1.0)

    def test_pool_config_preserves_official_worker_counts(self) -> None:
        module = _load_script("prepare_slinfer_relayserve.py")
        config_3b = module._pool_config("llama-3.2-3b", 8, 23.0)
        config_7b = module._pool_config("llama-2-7b", 4, 23.0)
        self.assertEqual(config_3b.count("models_info['llama-3.2-3b']"), 32)
        self.assertEqual(config_7b.count("models_info['llama-2-7b']"), 16)
        self.assertEqual(config_3b.count("'node_memory_capacity_GB': 23.0"), 4)
        self.assertEqual(config_7b.count("'node_memory_capacity_GB': 23.0"), 4)

    def test_gpu_lifecycle_handles_non_contiguous_allocations(self) -> None:
        module = _load_script("summarize_slinfer_replay.py")
        logs = {
            "node_usage": {"gpu": [1]},
            "node_density": {"gpu": [[0, 0, 1, 0]]},
            "batch": {"gpu": [[[], [], [1], []]]},
        }
        lifecycle = module._gpu_lifecycle(logs)
        self.assertEqual(lifecycle["startup_gpu_seconds"], 0.0)
        self.assertEqual(lifecycle["active_gpu_seconds"], 1.0)
        self.assertEqual(lifecycle["idle_gpu_seconds"], 0.0)
        self.assertEqual(lifecycle["allocated_gpu_seconds"], 1.0)


if __name__ == "__main__":
    unittest.main()
