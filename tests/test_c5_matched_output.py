from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts import analyze_c5_matched_output as c5


TRACE_SHA = "a" * 64
SUBSET_SHA = "b" * 64


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _request(
    system: str,
    index: int,
    *,
    dispatch_ms: float,
    service_ttft_ms: float,
    tpot_ms: float,
) -> dict:
    completion_tokens = 5
    decode_ms = tpot_ms * (completion_tokens - 1)
    service_e2e_ms = service_ttft_ms + decode_ms
    result = {
        "request_id": f"req-{index}",
        "adapter_id": f"adapter-{index % 2}",
        "success": True,
        "error": None,
        "generation_contract": c5.CONTRACT,
        "source_expected_output_tokens": completion_tokens,
        "requested_completion_tokens": completion_tokens,
        "completion_tokens": completion_tokens,
        "completion_token_ids_sha256": _digest(f"tokens-{system}-{index}"),
        "canonical_prompt_sha256": _digest(f"prompt-{index}"),
        "output_contract_match": True,
        "dispatch_admission_wait_ms": dispatch_ms,
        "service_ttft_ms": service_ttft_ms,
        "service_e2e_ms": service_e2e_ms,
        "overall_ttft_ms": dispatch_ms + service_ttft_ms,
        "ttft_ms": dispatch_ms + service_ttft_ms,
        "overall_e2e_ms": dispatch_ms + service_e2e_ms,
        "e2e_ms": dispatch_ms + service_e2e_ms,
        "tpot_ms": tpot_ms,
        "tpot_observed": True,
    }
    if system == "prime":
        result.update(
            {
                "scheduled_arrival_offset_s": float(index),
                "canonical_prompt_tokens": 20 + index,
                "completion_token_source": "vllm_token_ids",
            }
        )
    else:
        result.update(
            {
                "arrival_time_s": float(index),
                "guard_prompt_tokens": 20 + index,
                "completion_token_source": "slora_native_sse_token_id",
                "prompt_token_source": "local_tokenizer_guard",
                "native_sse_integer_token_id_count": completion_tokens,
                "native_sse_invalid_token_id_count": 0,
                "final_empty_success": False,
            }
        )
    return result


def _write_result(
    path: Path,
    *,
    system: str,
    seed: int,
    expected_requests: int = 3,
    dispatch_ms: float = 2.0,
    service_ttft_ms: float = 10.0,
    tpot_ms: float = 2.0,
) -> None:
    requests = [
        _request(
            system,
            index,
            dispatch_ms=dispatch_ms + index * 0.1,
            service_ttft_ms=service_ttft_ms + index * 0.2,
            tpot_ms=tpot_ms + index * 0.05,
        )
        for index in range(expected_requests)
    ]
    metadata = {
        "shared_trace_sha256": TRACE_SHA,
        "shared_adapter_subset_sha256": SUBSET_SHA,
        "model_profile": "llama2_7b_main_v2_publicmix",
    }
    if system == "prime":
        metadata.update({"generation_contract": c5.CONTRACT, "generation_seed": seed})
        scenario = "v2_full"
    else:
        metadata["sampling_seed"] = seed
        scenario = "slora_fair"
    payload = {
        "metadata": metadata,
        "scenario_summaries": {
            scenario: {
                "completion_token_source_counts": {
                    (
                        "vllm_token_ids"
                        if system == "prime"
                        else "slora_native_sse_token_id"
                    ): expected_requests
                }
            }
        },
        "detailed_results": {
            scenario: {
                "total": expected_requests,
                "completed": expected_requests,
                "failed": 0,
                "requests": requests,
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class C5MatchedOutputTests(unittest.TestCase):
    def test_three_seed_analysis_uses_seed_level_paired_ci(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = []
            for seed_index, seed in enumerate((43, 44, 45)):
                prime = root / f"prime-{seed}.json"
                slora = root / f"slora-{seed}.json"
                _write_result(
                    prime,
                    system="prime",
                    seed=seed,
                    dispatch_ms=2.0 + seed_index,
                    service_ttft_ms=9.0 + seed_index,
                    tpot_ms=1.5 + seed_index * 0.1,
                )
                _write_result(
                    slora,
                    system="slora",
                    seed=seed,
                    dispatch_ms=4.0 + seed_index,
                    service_ttft_ms=12.0 + seed_index,
                    tpot_ms=2.0 + seed_index * 0.1,
                )
                specs.extend(
                    [
                        c5.RunSpec("prime", "llama2_7b", seed, prime),
                        c5.RunSpec("slora", "llama2_7b", seed, slora),
                    ]
                )

            output = root / "publication"
            manifest = c5.analyze(specs, output_dir=output, expected_requests=3)

            self.assertFalse(
                manifest["statistical_method"]["request_records_are_independent_repetitions"]
            )
            self.assertEqual(
                manifest["statistical_method"]["model_seed_sets"]["llama2_7b"],
                [43, 44, 45],
            )
            expected_files = {
                "c5_per_run_metrics.csv",
                "c5_paired_seed_differences.csv",
                "c5_seed_level_ci.csv",
                "c5_matched_output_manifest.json",
                "c5_matched_output_stage_decomposition.pdf",
                "c5_matched_output_stage_decomposition.png",
            }
            self.assertTrue(expected_files.issubset({path.name for path in output.iterdir()}))
            with (output / "c5_seed_level_ci.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            e2e = next(
                row
                for row in rows
                if row["model"] == "llama2_7b" and row["metric"] == "e2e_mean_ms"
            )
            self.assertEqual(e2e["n_seeds"], "3")
            self.assertEqual(e2e["ci_unit"], "seed")
            self.assertAlmostEqual(float(e2e["prime_minus_slora_mean"]), -7.0)
            self.assertGreaterEqual(float(e2e["slora_minus_prime_improvement_mean"]), 7.0)

    def test_cross_system_request_map_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prime = root / "prime.json"
            slora = root / "slora.json"
            _write_result(prime, system="prime", seed=43)
            _write_result(slora, system="slora", seed=43)
            payload = json.loads(slora.read_text(encoding="utf-8"))
            payload["detailed_results"]["slora_fair"]["requests"][1][
                "canonical_prompt_sha256"
            ] = _digest("different prompt")
            slora.write_text(json.dumps(payload), encoding="utf-8")

            specs = [
                c5.RunSpec("prime", "llama2_7b", 43, prime),
                c5.RunSpec("slora", "llama2_7b", 43, slora),
            ]
            with self.assertRaisesRegex(c5.ValidationError, "request/adapter/arrival/target/prompt"):
                c5.analyze(specs, output_dir=root / "out", expected_requests=3)
            self.assertFalse((root / "out").exists())

    def test_missing_formal_seeds_are_rejected_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prime = root / "prime.json"
            slora = root / "slora.json"
            _write_result(prime, system="prime", seed=43)
            _write_result(slora, system="slora", seed=43)
            specs = [
                c5.RunSpec("prime", "llama2_7b", 43, prime),
                c5.RunSpec("slora", "llama2_7b", 43, slora),
            ]
            with self.assertRaisesRegex(c5.ValidationError, "incomplete formal seed matrix"):
                c5.analyze(specs, output_dir=root / "out", expected_requests=3)
            self.assertFalse((root / "out").exists())

    def test_incomplete_and_token_contract_runs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prime = root / "prime.json"
            _write_result(prime, system="prime", seed=43)
            payload = json.loads(prime.read_text(encoding="utf-8"))
            payload["detailed_results"]["v2_full"]["completed"] = 2
            payload["detailed_results"]["v2_full"]["failed"] = 1
            prime.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(c5.ValidationError, "incomplete formal run"):
                c5.validate_run(c5.RunSpec("prime", "llama2_7b", 43, prime), expected_requests=3)

            _write_result(prime, system="prime", seed=43)
            payload = json.loads(prime.read_text(encoding="utf-8"))
            payload["detailed_results"]["v2_full"]["requests"][0]["completion_tokens"] = 4
            prime.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(c5.ValidationError, "actual_tokens != target_tokens"):
                c5.validate_run(c5.RunSpec("prime", "llama2_7b", 43, prime), expected_requests=3)

    def test_latency_identity_tpot_and_fallback_gates_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slora = root / "slora.json"
            _write_result(slora, system="slora", seed=43)
            payload = json.loads(slora.read_text(encoding="utf-8"))
            request = payload["detailed_results"]["slora_fair"]["requests"][0]
            request["overall_e2e_ms"] += 1.01
            request["e2e_ms"] += 1.01
            slora.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(c5.ValidationError, "decomposition error"):
                c5.validate_run(c5.RunSpec("slora", "llama2_7b", 43, slora), expected_requests=3)

            _write_result(slora, system="slora", seed=43)
            payload = json.loads(slora.read_text(encoding="utf-8"))
            payload["detailed_results"]["slora_fair"]["requests"][0]["tpot_ms"] += 1.01
            slora.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(c5.ValidationError, "TPOT recomputation error"):
                c5.validate_run(c5.RunSpec("slora", "llama2_7b", 43, slora), expected_requests=3)

            _write_result(slora, system="slora", seed=43)
            payload = json.loads(slora.read_text(encoding="utf-8"))
            payload["scenario_summaries"]["slora_fair"]["completion_token_source_counts"][
                "trace_expected"
            ] = 1
            slora.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(c5.ValidationError, "forbidden fallback count"):
                c5.validate_run(c5.RunSpec("slora", "llama2_7b", 43, slora), expected_requests=3)

    def test_nonfresh_output_and_incomplete_round_manifest_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out"
            output.mkdir()
            (output / "old.csv").write_text("old", encoding="utf-8")
            with self.assertRaisesRegex(c5.ValidationError, "not fresh/empty"):
                c5._prepare_output_dir(output)

            round_dir = root / "round"
            round_dir.mkdir()
            (round_dir / "MANIFEST.json").write_text(
                json.dumps(
                    {
                        "status": "incomplete",
                        "generation_contract": c5.CONTRACT,
                        "model_profile": "llama2_7b_main_v2_publicmix",
                        "sampling_seed": 43,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(c5.ValidationError, "manifest is incomplete"):
                c5.specs_from_round(round_dir)


if __name__ == "__main__":
    unittest.main()
