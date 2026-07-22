from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from scripts import analyze_v2_full_vs_serverless as analyzer


def _run(model: str, seed: int, system: str) -> analyzer.FormalRun:
    multiplier = 0.8 if system == "faaslora" else 1.0
    metrics = {
        "ttft_avg_ms": 100.0 * multiplier + seed,
        "ttft_p95_ms": 150.0 * multiplier + seed,
        "e2e_avg_ms": 500.0 * multiplier + seed,
        "e2e_p95_ms": 650.0 * multiplier + seed,
        "tpot_avg_ms": 12.0 * multiplier,
        "tpot_p95_ms": 18.0 * multiplier,
        "tok_s": 120.0 / multiplier,
        "cost_req_usd": 0.01 * multiplier,
        "ce": 200.0 / multiplier,
    }
    model_char = "a" if model == "llama2_7b" else "b"
    system_char = "c" if system == "faaslora" else "d"
    return analyzer.FormalRun(
        model=model,
        seed=seed,
        system=system,
        scenario=analyzer.FORMAL_SCENARIOS[system],
        source=Path(f"/{model}/seed{seed}/{system}.json"),
        total=4000,
        completed=4000,
        generation_contract="legacy",
        selected_num_adapters=500,
        bandwidth_mib_s=250.0,
        configured_time_scale_factor=8.0,
        effective_time_scale_factor=8.0,
        zipf_exponent=1.0,
        active_adapter_cap=48,
        hotset_rotation_requests=500,
        hotset_rotation_mode="legacy",
        hotset_overlap_fraction=0.75,
        trace_sha256=(model_char * 62) + f"{seed % 100:02d}",
        adapter_subset_sha256=(model_char * 63) + "e",
        system_resolved_config_sha256=(model_char + system_char) * 32,
        metrics=metrics,
    )


def _matrix() -> list[analyzer.FormalRun]:
    return [
        _run(model, seed, system)
        for model in analyzer.FORMAL_MODELS
        for seed in analyzer.FORMAL_SEEDS
        for system in analyzer.FORMAL_SYSTEMS
    ]


class FullVsServerlessFormalTests(unittest.TestCase):
    def test_exact_matrix_and_output(self) -> None:
        runs = _matrix()
        self.assertEqual(
            analyzer.FORMAL_SCENARIOS["serverlessllm"],
            "serverlessllm_fair",
        )
        analyzer.validate_formal_matrix(runs)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new-output"
            analyzer.write_outputs(
                runs,
                output,
                inputs=[Path("campaign")],
                manifest_paths=[Path("campaign/MANIFEST.json")],
            )
            with (output / "full_vs_serverless_per_run.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                per_run = list(csv.DictReader(handle))
            with (output / "full_vs_serverless_paired_per_seed.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                paired = list(csv.DictReader(handle))
            with (output / "full_vs_serverless_paired_summary.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(len(per_run), 12)
            self.assertEqual(len(paired), 54)
            self.assertEqual(len(summary), 18)
            self.assertEqual({row["paired_seed_count"] for row in summary}, {"3"})
            for field in (
                "ttft_avg_ms",
                "e2e_avg_ms",
                "tpot_avg_ms",
                "tok_s",
                "cost_req_usd",
                "ce",
            ):
                self.assertIn(field, per_run[0])
            with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                analyzer.write_outputs(
                    runs,
                    output,
                    inputs=[],
                    manifest_paths=[],
                )

    def test_missing_extra_and_duplicate_identities_are_rejected(self) -> None:
        runs = _matrix()
        with self.assertRaisesRegex(SystemExit, "missing="):
            analyzer.validate_formal_matrix(runs[:-1])
        with self.assertRaisesRegex(SystemExit, "extra="):
            analyzer.validate_formal_matrix(
                [*runs, replace(runs[0], seed=42, source=Path("/extra.json"))]
            )
        with self.assertRaisesRegex(SystemExit, "duplicate="):
            analyzer.validate_formal_matrix([*runs, runs[0]])

    def test_wrong_axes_scenario_and_generation_contract_are_rejected(self) -> None:
        runs = _matrix()
        for replacement, pattern in (
            ({"bandwidth_mib_s": 119.2093}, "wrong bandwidth_mib_s"),
            ({"hotset_rotation_mode": "abrupt"}, "wrong hotset_rotation_mode"),
            ({"hotset_overlap_fraction": 0.0}, "wrong hotset_overlap_fraction"),
            ({"selected_num_adapters": 100}, "wrong selected_num_adapters"),
            ({"generation_contract": "fixed_length_greedy_v1"}, "wrong generation_contract"),
            ({"scenario": "faaslora_full"}, "must use scenario=v2_full"),
        ):
            with self.subTest(replacement=replacement):
                with self.assertRaisesRegex(SystemExit, pattern):
                    analyzer.validate_formal_matrix(
                        [replace(runs[0], **replacement), *runs[1:]]
                    )

    def test_config_drift_and_pair_hash_mismatch_are_rejected(self) -> None:
        runs = _matrix()
        with self.assertRaisesRegex(SystemExit, "frozen configuration drift"):
            analyzer.validate_formal_matrix(
                [
                    replace(runs[0], system_resolved_config_sha256="f" * 64),
                    *runs[1:],
                ]
            )
        with self.assertRaisesRegex(SystemExit, "trace/subset mismatch"):
            analyzer.validate_formal_matrix(
                [replace(runs[0], trace_sha256="f" * 64), *runs[1:]]
            )

    def test_formal_analyze_rejects_loose_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loose = root / "loose.json"
            loose.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "loose raw JSON input is forbidden"):
                analyzer.analyze([loose], root / "output")


if __name__ == "__main__":
    unittest.main()
