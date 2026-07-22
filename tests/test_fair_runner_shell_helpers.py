from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_full_fair_round.sh"


def _result_locator_functions() -> str:
    text = RUNNER.read_text(encoding="utf-8")
    start = text.index("sanitize_faaslora_result_tag() {")
    end = text.index("\ncollect_faaslora_result() {", start)
    return text[start:end]


def _run_faaslora_function() -> str:
    text = RUNNER.read_text(encoding="utf-8")
    start = text.index("run_faaslora() {")
    end = text.index("\nvalidate_cross_system_generation_contract() {", start)
    return text[start:end]


def _faaslora_generation_contract_function() -> str:
    text = RUNNER.read_text(encoding="utf-8")
    start = text.index("validate_faaslora_generation_contract() {")
    end = text.index("\nvalidate_faaslora_identity() {", start)
    return text[start:end]


class FairRunnerShellHelperTests(unittest.TestCase):
    def test_result_locator_matches_faaslora_lowercase_sanitized_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main_repo = root / "main"
            real_results = root / "real_results"
            nested = real_results / "nested"
            main_repo.mkdir()
            nested.mkdir(parents=True)
            (main_repo / "results").symlink_to(real_results, target_is_directory=True)

            older = real_results / "experiment_hello_world_t123.json"
            newer = nested / "experiment_hello_world_t123.json"
            older.write_text("{}", encoding="utf-8")
            newer.write_text("{}", encoding="utf-8")
            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))

            script = (
                "set -euo pipefail\n"
                + _result_locator_functions()
                + '\nMAIN_REPO="$1"\nfind_latest_faaslora_result "$2"\n'
            )
            observed = subprocess.check_output(
                ["bash", "-c", script, "bash", str(main_repo), "Hello.World-T123"],
                text=True,
            ).strip()
            self.assertEqual(Path(observed).resolve(), newer.resolve())

    def test_result_tag_sanitizer_matches_faaslora_contract(self) -> None:
        script = (
            "set -euo pipefail\n"
            + _result_locator_functions()
            + '\nsanitize_faaslora_result_tag "$1"\n'
        )
        observed = subprocess.check_output(
            ["bash", "-c", script, "bash", "__Mixed.Case--Tag__"],
            text=True,
        ).strip()
        self.assertEqual(observed, "mixed_case_tag")

    def test_faaslora_stage_forwards_heldout_seed_to_both_seed_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture_path = Path(tmp) / "run_logged_args.txt"
            script = (
                "set -euo pipefail\n"
                + _run_faaslora_function()
                + r'''
CAPTURE_PATH="$1"
MODEL_PROFILE=llama2_7b_main_v2_publicmix
DATASET_PROFILE=azure_sharegpt_rep4000
WORKLOAD_PROFILE=llama2_7b_auto500_formal4000_s8
TOTAL_REQUESTS=4000
SELECTED_NUM_ADAPTERS=500
SAMPLING_SEED=43
GENERATION_CONTRACT=fixed_length_greedy_v1
FIXED_OUTPUT_MAX_TOKENS=256
FIXED_PROMPT_MAX_TOKENS=759
STORAGE_BANDWIDTH_MIB_S=250
ZIPF_EXPONENT=1.0
ACTIVE_ADAPTER_CAP=48
HOTSET_ROTATION_REQUESTS=500
HOTSET_ROTATION_MODE=legacy
HOTSET_OVERLAP_FRACTION=0.75
FAASLORA_SCENARIO=v2_full
TRACE_ROLE=heldout
FORMAL_RUN=1
RUN_TAG=seed43_forwarding_test
TRACE_PATH=/tmp/trace.json
ADAPTER_SUBSET_PATH=/tmp/subset.json
ROUND_DIR=/tmp/round
MAIN_REPO=/tmp/main

faaslora_frozen_settings_sha256() { printf '%064d\n' 0; }
summary_path_for_system() { printf '/tmp/faaslora-summary.json\n'; }
is_done() { return 1; }
collect_calls=0
collect_faaslora_result() {
  collect_calls=$((collect_calls + 1))
  [[ "${collect_calls}" -ge 2 ]]
}
pre_system_clean_check() { :; }
resolved_config_sha256() { printf '%064d\n' 1; }
run_logged() {
  shift
  printf '%s\n' "$@" >"${CAPTURE_PATH}"
}
log() { :; }

run_faaslora
'''
            )
            subprocess.run(
                ["bash", "-c", script, "bash", str(capture_path)],
                check=True,
            )
            observed = capture_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(observed.count("FAASLORA_WORKLOAD_SEED=43"), 1)
            self.assertEqual(observed.count("FAASLORA_GENERATION_SEED=43"), 1)
            self.assertNotIn("FAASLORA_WORKLOAD_SEED=42", observed)
            self.assertNotIn("FAASLORA_GENERATION_SEED=42", observed)

    def test_fixed_contract_rejects_faaslora_seed_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.json"
            request_map_sha = hashlib.sha256(
                json.dumps(
                    [], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            payload = {
                "metadata": {
                    "generation_contract": "fixed_length_greedy_v1",
                    "fixed_output_max_tokens": 256,
                    "fixed_prompt_max_tokens": 759,
                    "workload_seed": 43,
                    "generation_seed": 43,
                    "generation_contract_request_map_sha256": {
                        "v2_full": request_map_sha
                    },
                },
                "detailed_results": {"v2_full": {"requests": []}},
            }
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            script = (
                "set -euo pipefail\n"
                + _faaslora_generation_contract_function()
                + r'''
GENERATION_CONTRACT=fixed_length_greedy_v1
TOTAL_REQUESTS=0
FIXED_OUTPUT_MAX_TOKENS=256
FIXED_PROMPT_MAX_TOKENS=759
SAMPLING_SEED=43
validate_faaslora_generation_contract "$1"
'''
            )
            subprocess.run(
                ["bash", "-c", script, "bash", str(result_path)],
                check=True,
                capture_output=True,
                text=True,
            )

            payload["metadata"]["generation_seed"] = 42
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            rejected = subprocess.run(
                ["bash", "-c", script, "bash", str(result_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("generation seed mismatch", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
