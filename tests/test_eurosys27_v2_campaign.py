from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from scripts import run_eurosys27_v2_campaign as campaign


COMPLETE_RUNNER = r"""
import json
import os
from pathlib import Path

root = Path(os.environ["FAIR_ROUND_DIR"])
(root / "raw" / "faaslora").mkdir(parents=True, exist_ok=True)
(root / "raw" / "replay").mkdir(parents=True, exist_ok=True)
(root / "raw" / "faaslora" / "test_faaslora_result.json").write_text("{}")
(root / "raw" / "replay" / "test_slora_dp4_summary.json").write_text("{}")
payload = {
    "status": "complete",
    "run_tag": os.environ["SLLM_RUN_TAG"],
    "round_dir": str(root),
    "system_resolved_config_registry_path": os.environ["FAIR_RESOLVED_CONFIG_REGISTRY"],
    "formal_run": os.environ.get("FAIR_FORMAL_RUN") == "1",
}
(root / "MANIFEST.json").write_text(json.dumps(payload))
"""


FAIL_ONCE_RUNNER = r"""
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["FAIR_ROUND_DIR"])
marker = root / "failed_once.marker"
if not marker.exists():
    marker.write_text("failed")
    raise SystemExit(7)
payload = {
    "status": "complete",
    "run_tag": os.environ["SLLM_RUN_TAG"],
    "round_dir": str(root),
    "system_resolved_config_registry_path": os.environ["FAIR_RESOLVED_CONFIG_REGISTRY"],
    "formal_run": False,
}
(root / "MANIFEST.json").write_text(json.dumps(payload))
"""


ALWAYS_FAIL_RUNNER = "raise SystemExit(9)"


class CampaignTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.campaign_root = self.root / "campaign"
        self.registry = self.root / "protocol" / "registry.json"
        self.config_path = self.root / "campaign.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def config_payload(
        self,
        *,
        runner: str = COMPLETE_RUNNER,
        second_run: bool = False,
        reuse: bool = False,
        export: bool = False,
        run_env: dict[str, str] | None = None,
    ) -> dict[str, object]:
        runs: list[dict[str, object]] = [
            {
                "id": "run_one",
                "runner": [sys.executable, "-c", runner],
                "working_directory": str(self.root),
                "registry": "formal",
                "depends_on": [],
                "env": dict(run_env or {}),
            }
        ]
        if second_run:
            runs.append(
                {
                    "id": "run_two",
                    "runner": [sys.executable, "-c", COMPLETE_RUNNER],
                    "working_directory": str(self.root),
                    "registry": "formal",
                    "depends_on": ["run_one"],
                    "env": {"EXPERIMENT_AXIS": "second"},
                }
            )
        if reuse:
            runs.append(
                {
                    "id": "run_one_reused",
                    "reuse_of": "run_one",
                    "depends_on": ["run_one"],
                }
            )
        payload: dict[str, object] = {
            "schema_version": campaign.SCHEMA_VERSION,
            "campaign_id": "test_campaign",
            "campaign_root": str(self.campaign_root),
            "environment": {"inherit": [], "values": {}},
            "registries": {"formal": str(self.registry)},
            "source_repositories": {},
            "cleanup": {
                "strict_gpu_idle": False,
                "gpu_ids": [],
                "timeout_seconds": 0,
                "poll_seconds": 0.01,
                "pre_commands": [],
                "post_commands": [],
            },
            "runs": runs,
            "analyzer_exports": [],
        }
        if export:
            payload["analyzer_exports"] = [
                {
                    "id": "analysis",
                    "run_ids": ["run_one_reused" if reuse else "run_one"],
                    "output_directory": str(self.campaign_root / "exports"),
                    "inputs": [
                        {"name": "manifest", "glob": "MANIFEST.json"},
                        {
                            "name": "prime",
                            "glob": "raw/faaslora/*_faaslora_result.json",
                        },
                    ],
                }
            ]
        return payload

    def write_config(self, payload: dict[str, object]) -> None:
        self.config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def test_plan_is_read_only_and_never_invokes_runner(self) -> None:
        marker = self.root / "runner_invoked"
        runner = f"from pathlib import Path; Path({str(marker)!r}).write_text('bad')"
        self.write_config(self.config_payload(runner=runner))

        self.assertEqual(
            campaign.main(["--config", str(self.config_path), "plan", "--json"]),
            0,
        )
        self.assertFalse(marker.exists())
        self.assertFalse(self.campaign_root.exists())

        self.assertEqual(
            campaign.main(["--config", str(self.config_path), "status", "--json"]),
            0,
        )
        self.assertFalse(self.campaign_root.exists())

    def test_run_key_is_canonical_and_changes_with_semantics(self) -> None:
        payload = self.config_payload(run_env={"B": "2", "A": "1"})
        self.write_config(payload)
        first = campaign.load_config(self.config_path)
        first_key = first.runs[0].run_key

        payload["runs"][0]["env"] = {"A": "1", "B": "2"}  # type: ignore[index]
        self.write_config(payload)
        second = campaign.load_config(self.config_path)
        self.assertEqual(first_key, second.runs[0].run_key)

        payload["runs"][0]["env"] = {"A": "different", "B": "2"}  # type: ignore[index]
        self.write_config(payload)
        third = campaign.load_config(self.config_path)
        self.assertNotEqual(first_key, third.runs[0].run_key)

    def test_successful_run_freezes_protocol_and_appends_ledger(self) -> None:
        self.write_config(self.config_payload())
        self.assertEqual(
            campaign.main(["--config", str(self.config_path), "run-next"]), 0
        )
        config = campaign.load_config(self.config_path)
        status = campaign.campaign_status(config)
        state = status["runs"]["run_one"]
        self.assertEqual(state["status"], "complete")
        self.assertTrue(Path(state["manifest_path"]).is_file())
        self.assertTrue(
            (self.campaign_root / "protocol" / "campaign_environment.json").is_file()
        )
        events = campaign._read_ledger(config)
        self.assertEqual(
            [event["sequence"] for event in events], list(range(1, len(events) + 1))
        )
        self.assertEqual(events[0]["event"], "campaign_initialized")
        self.assertEqual(events[-1]["event"], "attempt_completed")
        attempt_dir = Path(state["attempt_dir"])
        frozen = json.loads(
            (attempt_dir / "orchestrator_frozen_environment.json").read_text()
        )
        self.assertEqual(frozen["FAIR_ROUND_DRY_RUN"], "0")
        self.assertEqual(frozen["PAPER_QUEUE_DRY_RUN"], "0")
        self.assertEqual(frozen["FAIR_RESOLVED_CONFIG_REGISTRY"], str(self.registry))

    def test_failure_stop_then_resume_same_attempt(self) -> None:
        self.write_config(self.config_payload(runner=FAIL_ONCE_RUNNER, second_run=True))
        self.assertEqual(
            campaign.main(["--config", str(self.config_path), "run-next"]), 1
        )
        config = campaign.load_config(self.config_path)
        failed = campaign.campaign_status(config)["runs"]["run_one"]
        self.assertEqual(failed["status"], "failed")
        original_attempt = failed["attempt_id"]
        self.assertEqual(
            campaign.main(["--config", str(self.config_path), "run-next"]), 2
        )

        self.assertEqual(
            campaign.main(
                ["--config", str(self.config_path), "resume", "--run-id", "run_one"]
            ),
            0,
        )
        resumed = campaign.campaign_status(config)["runs"]["run_one"]
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(resumed["attempt_id"], original_attempt)
        attempt_dir = Path(resumed["attempt_dir"])
        self.assertEqual(
            len(list(attempt_dir.glob("orchestrator_runner_invocation_*.log"))), 2
        )
        self.assertEqual(
            len(
                list(
                    attempt_dir.glob("orchestrator_cleanup_evidence_invocation_*.json")
                )
            ),
            2,
        )

        self.assertEqual(
            campaign.main(["--config", str(self.config_path), "run-next"]), 0
        )
        self.assertEqual(
            campaign.campaign_status(config)["runs"]["run_two"]["status"],
            "complete",
        )

    def test_retry_creates_new_attempt_and_preserves_failed_attempt(self) -> None:
        self.write_config(self.config_payload(runner=ALWAYS_FAIL_RUNNER))
        self.assertEqual(
            campaign.main(["--config", str(self.config_path), "run-next"]), 1
        )
        config = campaign.load_config(self.config_path)
        first = campaign.campaign_status(config)["runs"]["run_one"]
        first_dir = Path(first["attempt_dir"])

        self.assertEqual(
            campaign.main(
                ["--config", str(self.config_path), "retry", "--run-id", "run_one"]
            ),
            1,
        )
        second = campaign.campaign_status(config)["runs"]["run_one"]
        self.assertNotEqual(first["attempt_id"], second["attempt_id"])
        self.assertTrue(first_dir.is_dir())
        self.assertTrue(Path(second["attempt_dir"]).is_dir())

    def test_post_cleanup_hook_runs_even_when_runner_fails(self) -> None:
        payload = self.config_payload(runner=ALWAYS_FAIL_RUNNER)
        hook_log = self.root / "hook.log"
        payload["cleanup"]["pre_commands"] = [  # type: ignore[index]
            [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(hook_log)!r}).write_text('pre\\n')",
            ]
        ]
        payload["cleanup"]["post_commands"] = [  # type: ignore[index]
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"p=Path({str(hook_log)!r}); "
                    "p.write_text(p.read_text() + 'post\\n')"
                ),
            ]
        ]
        self.write_config(payload)
        self.assertEqual(
            campaign.main(["--config", str(self.config_path), "run-next"]), 1
        )
        self.assertEqual(hook_log.read_text(), "pre\npost\n")

    def test_initialized_config_is_immutable(self) -> None:
        payload = self.config_payload()
        self.write_config(payload)
        self.assertEqual(
            campaign.main(["--config", str(self.config_path), "run-next"]), 0
        )
        payload["runs"][0]["env"] = {"RETUNED_AFTER_RESULT": "1"}  # type: ignore[index]
        self.write_config(payload)
        self.assertEqual(
            campaign.main(["--config", str(self.config_path), "status"]), 0
        )
        self.assertEqual(
            campaign.main(["--config", str(self.config_path), "run-next"]), 2
        )

    def test_reuse_alias_and_analyzer_export_do_not_rerun(self) -> None:
        self.write_config(self.config_payload(reuse=True, export=True))
        self.assertEqual(
            campaign.main(["--config", str(self.config_path), "run-next"]), 0
        )
        config = campaign.load_config(self.config_path)
        status = campaign.campaign_status(config)
        self.assertEqual(status["runs"]["run_one_reused"]["status"], "complete")
        self.assertEqual(
            status["runs"]["run_one_reused"]["attempt_id"],
            status["runs"]["run_one"]["attempt_id"],
        )

        self.assertEqual(
            campaign.main(
                [
                    "--config",
                    str(self.config_path),
                    "export-inputs",
                    "--export",
                    "analysis",
                ]
            ),
            0,
        )
        exports = sorted((self.campaign_root / "exports").glob("*/inputs.json"))
        self.assertEqual(len(exports), 1)
        payload = json.loads(exports[0].read_text())
        self.assertEqual(payload["records"][0]["logical_run_id"], "run_one_reused")
        self.assertEqual(payload["records"][0]["source_run_id"], "run_one")
        prime_list = exports[0].parent / "prime.txt"
        self.assertIn("test_faaslora_result.json", prime_list.read_text())

        self.assertEqual(
            campaign.main(
                [
                    "--config",
                    str(self.config_path),
                    "export-inputs",
                    "--export",
                    "analysis",
                ]
            ),
            0,
        )
        self.assertEqual(
            len(list((self.campaign_root / "exports").glob("*/inputs.json"))), 2
        )

    def test_completed_manifest_mutation_is_detected(self) -> None:
        self.write_config(self.config_payload())
        self.assertEqual(
            campaign.main(["--config", str(self.config_path), "run-next"]), 0
        )
        config = campaign.load_config(self.config_path)
        state = campaign.campaign_status(config)["runs"]["run_one"]
        Path(state["manifest_path"]).write_text('{"status":"incomplete"}')
        self.assertEqual(
            campaign.campaign_status(config)["runs"]["run_one"]["status"],
            "invalid",
        )

    def test_forbids_implicit_registry_and_orchestrator_owned_environment(self) -> None:
        payload = self.config_payload()
        del payload["registries"]
        self.write_config(payload)
        with self.assertRaisesRegex(campaign.CampaignError, "registries"):
            campaign.load_config(self.config_path)

        payload = self.config_payload(run_env={"FAIR_ROUND_DRY_RUN": "1"})
        self.write_config(payload)
        with self.assertRaisesRegex(campaign.CampaignError, "orchestrator-owned"):
            campaign.load_config(self.config_path)

    def test_dependency_and_reuse_cycles_are_rejected(self) -> None:
        payload = self.config_payload(second_run=True)
        payload["runs"][0]["depends_on"] = ["run_two"]  # type: ignore[index]
        self.write_config(payload)
        with self.assertRaisesRegex(campaign.CampaignError, "cycle"):
            campaign.load_config(self.config_path)

        payload = self.config_payload(reuse=True)
        payload["runs"][-1]["reuse_of"] = "run_one_reused"  # type: ignore[index]
        self.write_config(payload)
        with self.assertRaisesRegex(campaign.CampaignError, "reuse cycle"):
            campaign.load_config(self.config_path)


if __name__ == "__main__":
    unittest.main()
