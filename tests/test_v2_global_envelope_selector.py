from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "select_eurosys27_v2_global_envelope.py"
SPEC = importlib.util.spec_from_file_location("v2_global_envelope_selector", SCRIPT)
assert SPEC and SPEC.loader
selector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = selector
SPEC.loader.exec_module(selector)


FIELDS = (
    "FAASLORA_MIN_INSTANCES",
    "FAASLORA_MAX_INSTANCES",
    "FAASLORA_RUNTIME_CONCURRENCY_CAP",
    "FAASLORA_MAX_NUM_SEQS",
    "FAASLORA_MAX_LORAS",
    "FAASLORA_MAX_NUM_BATCHED_TOKENS",
)


def metrics(
    *, ttft: float = 90.0, e2e: float = 900.0, p95: float = 100.0, ce: float = 12.0
):
    return selector.FamilyMetrics(
        prime_avg_ttft_ms=ttft,
        baseline_avg_ttft_ms=100.0,
        prime_avg_e2e_ms=e2e,
        baseline_avg_e2e_ms=1000.0,
        prime_p95_ttft_ms=p95,
        baseline_p95_ttft_ms=100.0,
        prime_ce=ce,
        baseline_ce=10.0,
    )


def audit(label: str, envelope: dict[str, int], c5, fvs):
    value = selector.CandidateAudit("llama2_7b", label, envelope)
    value.observed_envelopes = {"c5": dict(envelope), "fvs": dict(envelope)}
    value.metrics = {"c5": c5, "fvs": fvs}
    selector.evaluate_candidate(value)
    return value


def test_constraints_report_every_rejection_reason():
    envelope = dict(zip(FIELDS, (1, 4, 2, 2, 4, 1024)))
    rejected = audit(
        "P0",
        envelope,
        metrics(ttft=101, e2e=1001, p95=106),
        metrics(ttft=102, e2e=1002, p95=107),
    )
    assert not rejected.eligible
    assert len(rejected.rejection_reasons) == 6
    assert sum("avg TTFT" in reason for reason in rejected.rejection_reasons) == 2
    assert sum("avg E2E" in reason for reason in rejected.rejection_reasons) == 2
    assert sum("P95 TTFT ratio" in reason for reason in rejected.rejection_reasons) == 2


def test_score_is_worst_family_and_one_percent_tie_selects_p0():
    envelope = dict(zip(FIELDS, (1, 4, 2, 2, 4, 1024)))
    p0 = audit("P0", envelope, metrics(ce=12.0), metrics(ce=11.95))
    p1 = audit("P1", envelope, metrics(ce=12.1), metrics(ce=12.05))
    assert p0.score == pytest.approx(1.195)
    assert p1.score == pytest.approx(1.205)
    selected = selector.choose_per_model([p0, p1], 0.01)
    assert selected == {"llama2_7b": "P0"}
    assert p0.selected and not p1.selected


def test_no_eligible_candidate_has_no_selection():
    envelope = dict(zip(FIELDS, (1, 4, 2, 2, 4, 1024)))
    rejected = audit("P0", envelope, metrics(ttft=101), metrics())
    assert selector.choose_per_model([rejected], 0.01) == {}


def test_input_provenance_records_bytes_sha_and_rejects_mutation(tmp_path: Path):
    source = tmp_path / "input.json"
    source.write_bytes(b'{"a":1}\n')
    recorder = selector.ProvenanceRecorder()
    recorder.record(source, "first")
    row = recorder.rows()[0]
    assert row["bytes"] == len(b'{"a":1}\n')
    assert row["sha256"] == hashlib.sha256(b'{"a":1}\n').hexdigest()
    source.write_bytes(b'{"a":2}\n')
    with pytest.raises(selector.SelectionError, match="changed while selecting"):
        recorder.record(source, "second")


def test_source_identity_must_match_across_complete_six_round_cohort():
    envelope = dict(zip(FIELDS, (1, 4, 2, 2, 4, 1024)))
    audits = [
        selector.CandidateAudit("llama2_7b", label, envelope) for label in ("P0", "P1")
    ]
    audits.append(selector.CandidateAudit("llama32_3b", "P0", envelope))
    identity = {
        "baseline_git": {"branch": "main", "commit": "a" * 40},
        "faaslora_git": {
            "branch": "retry14_continuous_queue_v2",
            "commit": "b" * 40,
        },
    }
    for candidate in audits:
        candidate.source_identities = {"c5": identity, "fvs": identity}
    frozen, error = selector.freeze_source_identity(audits, 6)
    assert error is None
    assert frozen == identity

    audits[-1].source_identities["fvs"] = {
        **identity,
        "faaslora_git": {
            "branch": "retry14_continuous_queue_v2",
            "commit": "c" * 40,
        },
    }
    frozen, error = selector.freeze_source_identity(audits, 6)
    assert frozen == {}
    assert "mix source versions for faaslora_git" in str(error)


def test_protocol_gate_rejects_policy_drift():
    protocol = json.loads(
        (ROOT / "configs/eurosys27_v2_validation_selection_protocol.json").read_text()
    )
    selector._validate_protocol(protocol)
    protocol["candidate_validity"]["constraints_per_family"][
        "prime_p95_ttft_ratio_max"
    ] = 1.06
    with pytest.raises(selector.SelectionError, match="candidate_validity differs"):
        selector._validate_protocol(protocol)


def _synthetic_sidecar(envelope: tuple[int, ...], path: Path):
    resolved = {
        "resource_coordination": {
            "min_instances": envelope[0],
            "max_instances": envelope[1],
        },
        "model": {
            "runtime_concurrency_cap": envelope[2],
            "max_num_seqs": envelope[3],
            "max_loras": envelope[4],
            "max_num_batched_tokens": envelope[5],
        },
    }
    hashed = {
        "systems": {
            "PrimeLoRA": {"environment_overrides": {}, "resolved_profiles": resolved},
            "S-LoRA": {},
        }
    }
    payload = {
        "formal_run": False,
        "trace_role": "validation",
        "sampling_seed": 41,
        "source_clean_for_formal": True,
        "hashed_config": hashed,
        "system_resolved_config_sha256": selector._canonical_sha256(hashed),
    }
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_synthetic_manifest_sidecar_validates_and_detects_conflicting_override(
    tmp_path: Path,
):
    protocol = json.loads(
        (ROOT / "configs/eurosys27_v2_validation_selection_protocol.json").read_text()
    )
    round_dir = tmp_path / "round"
    sidecar_path = round_dir / "protocol" / "system_resolved_config.json"
    sidecar = _synthetic_sidecar((1, 4, 2, 2, 4, 1024), sidecar_path)
    sidecar_bytes = sidecar_path.read_bytes()
    manifest = {
        "status": "complete",
        "formal_run": False,
        "trace_role": "validation",
        "sampling_seed": 41,
        "total_requests": 1000,
        "campaign_kind": "v2_c5_matched_output",
        "generation_contract": "fixed_length_greedy_v1",
        "model_profile": "llama2_7b_main_v2_publicmix",
        "workload_profile": "llama2_7b_auto500_formal4000_s8",
        "systems": ["faaslora", "slora"],
        "supported_systems": ["faaslora", "slora"],
        "execution_order": ["slora", "faaslora"],
        "faaslora_scenario": "v2_full",
        "source_clean_for_formal": True,
        "system_resolved_config_path": str(sidecar_path),
        "system_resolved_config_sidecar_bytes": len(sidecar_bytes),
        "system_resolved_config_sidecar_sha256": hashlib.sha256(
            sidecar_bytes
        ).hexdigest(),
        "system_resolved_config_sha256": sidecar["system_resolved_config_sha256"],
    }
    observed = selector._validate_manifest_and_sidecar(
        manifest_path=round_dir / "MANIFEST.json",
        manifest=manifest,
        sidecar=sidecar,
        sidecar_path=sidecar_path,
        family="c5",
        family_spec=protocol["families"]["c5"],
        model_spec=protocol["models"]["llama2_7b"],
        protocol=protocol,
    )
    assert observed == dict(zip(FIELDS, (1, 4, 2, 2, 4, 1024)))

    sidecar["hashed_config"]["systems"]["PrimeLoRA"]["environment_overrides"] = {
        "FAASLORA_MAX_NUM_SEQS": "8"
    }
    with pytest.raises(
        selector.SelectionError, match="conflicting FAASLORA_MAX_NUM_SEQS"
    ):
        selector._extract_envelope(sidecar, FIELDS, "synthetic")


def test_output_directory_is_create_once_and_records_rejections(tmp_path: Path):
    envelope = dict(zip(FIELDS, (1, 4, 2, 2, 4, 1024)))
    rejected = audit("P0", envelope, metrics(ttft=101), metrics())
    payload = {
        "schema_version": selector.SCHEMA_VERSION,
        "status": "failed",
        "candidates": [selector._audit_json(rejected)],
    }
    output = tmp_path / "selection"
    selector._write_outputs(output, payload, [rejected])
    written = json.loads((output / "global_envelope_selection.json").read_text())
    assert written["status"] == "failed"
    assert (output / "global_envelope_selection.csv").read_text().count("avg TTFT") == 2
    with pytest.raises(FileExistsError):
        selector._write_outputs(output, payload, [rejected])
