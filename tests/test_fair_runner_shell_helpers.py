from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
