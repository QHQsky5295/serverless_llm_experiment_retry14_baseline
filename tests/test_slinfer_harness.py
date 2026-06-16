from __future__ import annotations

import importlib.util
import subprocess
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
    def test_host_memory_defaults_keep_a_safety_reserve(self) -> None:
        start_script = (ROOT / "scripts/start_slinfer_stack.sh").read_text()
        run_script = (
            ROOT / "scripts/run_slinfer_relayserve_continuation.sh"
        ).read_text()
        self.assertIn("SLINFER_STORE_MEM_POOL_SIZE_GB:-20", start_script)
        self.assertIn("SLINFER_MIN_AVAILABLE_MEMORY_GB:-32", start_script)
        self.assertIn(
            'REQUESTED_WORKER_NUM="${SLINFER_WORKERS_PER_GPU:-0}"',
            start_script,
        )
        self.assertIn('if [[ "${REQUESTED_WORKER_NUM}" == "0" ]]', start_script)
        self.assertIn("WORKER_NUM > MAX_WORKER_NUM", start_script)
        self.assertIn(
            'export SLINFER_WORKERS_PER_GPU="${WORKERS_PER_GPU}"',
            run_script,
        )
        self.assertIn("/health", start_script)
        compat_patch = (
            ROOT / "patches/slinfer_relayserve_compat.patch"
        ).read_text()
        self.assertIn("--swap-space 1", compat_patch)
        self.assertIn("'failure_reason': 'deadline_violation'", compat_patch)
        self.assertIn("memory_guard.csv", run_script)
        self.assertIn("SLINFER_ALLOW_FAILED_REQUESTS", run_script)
        self.assertIn('REPLAY_EXIT_CODE="${PIPESTATUS[0]}"', run_script)
        self.assertIn(
            'if [[ ! -s "${RAW_PATH}" ]]',
            run_script,
        )
        self.assertIn("VALIDATION_EXIT_CODE", run_script)
        self.assertIn("finalize_slinfer_run_artifacts.py", run_script)
        self.assertIn(
            "SLINFER_SCHEDULER_TTFT_MAX_THRESHOLD_S:-7.6",
            run_script,
        )
        self.assertIn("SLINFER_SCHEDULER_TPOT_S:-0.2375", run_script)
        self.assertIn("SLINFER_ENABLE_DEFRAGMENTATION:-1", run_script)
        self.assertIn("tuned_slinfer_scheduler_runtime_config", run_script)
        self.assertIn("--disable-defragmentation", run_script)
        subprocess.run(
            ["bash", "-n", str(ROOT / "scripts/start_slinfer_stack.sh")],
            check=True,
        )
        subprocess.run(
            [
                "bash",
                "-n",
                str(ROOT / "scripts/run_slinfer_relayserve_continuation.sh"),
            ],
            check=True,
        )
        subprocess.run(
            [
                "bash",
                "-n",
                str(ROOT / "scripts/run_slinfer_calibration_suite.sh"),
            ],
            check=True,
        )

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

    def test_pool_config_uses_capacity_bounded_worker_counts(self) -> None:
        module = _load_script("prepare_slinfer_relayserve.py")
        config_3b = module._pool_config("llama-3.2-3b", 2, 23.0)
        config_7b = module._pool_config("llama-2-7b", 1, 23.0)
        self.assertEqual(config_3b.count("models_info['llama-3.2-3b']"), 8)
        self.assertEqual(config_7b.count("models_info['llama-2-7b']"), 4)
        self.assertEqual(config_3b.count("'node_memory_capacity_GB': 23.0"), 4)
        self.assertEqual(config_7b.count("'node_memory_capacity_GB': 23.0"), 4)
        self.assertIn(
            "from .models_info_template import models_info_template",
            config_3b,
        )

    def test_rate_matrix_run_parser_preserves_model_rate_and_path(self) -> None:
        module = _load_script("build_slinfer_rate_matrix.py")
        model_key, rate_scale, run_dir = module.parse_run(
            "3b:0.67=/tmp/slinfer-rate"
        )
        self.assertEqual(model_key, "3b")
        self.assertEqual(rate_scale, 0.67)
        self.assertEqual(run_dir, Path("/tmp/slinfer-rate"))
        self.assertEqual(module.EXPECTED_RATES, {0.67, 1.0, 1.3})

    def test_official_scheduler_deadline_is_separate_from_paper_slo(self) -> None:
        replay_script = (ROOT / "scripts/replay_slinfer_trace.py").read_text()
        self.assertNotIn(
            "args.ttft_slo_ms / 1000.0",
            replay_script,
        )
        self.assertIn(
            '"TTFT_max_threshold": args.scheduler_ttft_max_threshold_s',
            replay_script,
        )
        self.assertIn("separate_from_external_paper_slo", replay_script)
        self.assertIn('"enable_defragmentation": args.enable_defragmentation', replay_script)
        self.assertIn("--disable-defragmentation", replay_script)

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

    def test_calibration_policy_excludes_failed_candidates(self) -> None:
        module = _load_script("build_slinfer_calibration_table.py")
        failed = {
            "worst_normalized_p95_violation": 0.5,
            "joint_slo_attainment": 0.99,
            "ce": 999.0,
            "keep_alive_s": 60,
            "eligible_zero_failure": False,
        }
        clean = {
            "worst_normalized_p95_violation": 2.0,
            "joint_slo_attainment": 0.5,
            "ce": 1.0,
            "keep_alive_s": 1,
            "eligible_zero_failure": True,
        }
        eligible = [row for row in (failed, clean) if row["eligible_zero_failure"]]
        self.assertIs(min(eligible, key=module.ranking_key), clean)

    def test_calibration_policy_allows_audited_no_selection(self) -> None:
        module = _load_script("build_slinfer_calibration_table.py")
        failed = [
            {
                "worst_normalized_p95_violation": ratio,
                "joint_slo_attainment": 0.9,
                "ce": 80.0,
                "keep_alive_s": keep_alive,
                "eligible_zero_failure": False,
            }
            for ratio, keep_alive in ((1.0, 1), (0.9, 10))
        ]
        eligible = [row for row in failed if row["eligible_zero_failure"]]
        winner = min(eligible, key=module.ranking_key) if eligible else None
        self.assertIsNone(winner)

    def test_formal_finalizer_preserves_failure_counts(self) -> None:
        module = _load_script("finalize_slinfer_run_artifacts.py")
        raw = {
            "results": [
                {"success": True},
                {"success": False, "failure_reason": "deadline_violation"},
                {"success": False, "failure_reason": "deadline_violation"},
            ]
        }
        self.assertEqual(
            module.analyze(raw),
            {
                "total_requests": 3,
                "ok_requests": 1,
                "failed_requests": 2,
                "failure_reasons": {"deadline_violation": 2},
            },
        )

    def test_formal_gate_eligibility_is_independent_of_paper_slo(self) -> None:
        module = _load_script("build_slinfer_formal_gate_table.py")
        manifest = {"formal_main_comparison_eligible": True}
        self.assertTrue(module.comparison_eligible(manifest, 0))
        self.assertFalse(module.comparison_eligible(manifest, 1))


if __name__ == "__main__":
    unittest.main()
