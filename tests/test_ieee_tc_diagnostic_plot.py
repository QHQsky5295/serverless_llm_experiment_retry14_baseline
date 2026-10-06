"""Offline timing/plot guards; no GPU or scientific-effect qualification."""
import copy
import json

import pytest

from scripts.ieee_tc.plot_d290_diagnostic import (binding_checks, load, percentile,
    request_vectors, write_numerical_repeat_audit)


def request():
    return {
        "request_id": "r0", "success": True,
        "generation_contract": "fixed_length_greedy_v1",
        "output_contract_match": True, "completion_token_source": "vllm_token_ids",
        "completion_tokens": 2, "requested_completion_tokens": 2,
        "readiness_tier_before_dispatch": "HOST",
        "arrival_release_lateness_ms": 5.0,
        "dispatch_admission_wait_ms": 20.0,
        "service_ttft_ms": 10.0, "overall_ttft_ms": 30.0,
    }


def test_lateness_is_not_added_twice():
    assert request_vectors([request()]) == ([20.0], [10.0], [30.0])


@pytest.mark.parametrize("field,value", [
    ("overall_ttft_ms", 35.0), ("service_ttft_ms", float("nan")),
    ("dispatch_admission_wait_ms", -1.0),
    ("completion_token_source", "expected"), ("completion_tokens", 1),
    ("readiness_tier_before_dispatch", ""), ("output_contract_match", False),
])
def test_missing_or_inconsistent_data_is_not_filled(field, value):
    row = request()
    row[field] = value
    with pytest.raises(ValueError):
        request_vectors([row])


def test_missing_timing_rejected():
    row = request()
    del row["service_ttft_ms"]
    with pytest.raises(KeyError):
        request_vectors([row])


def test_duplicates_and_empty_rejected():
    with pytest.raises(ValueError):
        request_vectors([request(), request()])
    with pytest.raises(ValueError):
        request_vectors([])


def test_type_one_quantile():
    assert percentile([10, 20, 30, 40], 0.5) == 20
    assert percentile([10, 20, 30, 40], 0.95) == 40
    with pytest.raises(ValueError):
        percentile([], 0.95)


def test_incomplete_replay_rejected(tmp_path):
    result = {"metadata": {"total_requests": 1}, "detailed_results": {
        "one": {"completed": 1, "failed": 0, "requests": [request()]}}}
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    assert load(path)[1]["completed"] == 1
    incomplete = copy.deepcopy(result)
    incomplete["metadata"]["total_requests"] = 2
    path.write_text(json.dumps(incomplete))
    with pytest.raises(ValueError):
        load(path)
    result["detailed_results"]["two"] = result["detailed_results"]["one"]
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        load(path)


def binding_row():
    return {
        "adapter_id": "a", "completion_token_ids_sha256": "tokens",
        "native_token_timing": {
            "gpu_reference_owner_id": "o", "gpu_reference_lease_id": "l",
            "gpu_reference_adapter_int_id": 1, "native_clock_id": "clock",
            "native_dispatch_monotonic_s": 3, "completion_token_ids_sha256": "tokens"},
        "gpu_reference_evidence": {
            "state": "released",
            "intent": {"lora_name": "a", "expected_owner_id": "o", "lease_id": "l",
                       "adapter_int_id": 1, "expected_content_sha256": "content"},
            "receipt": {"lora_name": "a", "owner_id": "o", "lease_id": "l",
                        "adapter_int_id": 1, "slot": 0, "expected_content_sha256": "content",
                        "acquired": True, "clock_id": "clock", "acquired_monotonic_s": 2,
                        "post_load_source_snapshot": {"owner_id": "o", "clock_id": "clock",
                            "slot_adapter_ids": [1], "sources": [{"adapter_id": "a",
                            "adapter_int_id": 1, "gpu_slot": 0, "content_sha256": "content"}]}},
            "release_receipt": {"released": True, "owner_id": "o"}},
    }


def test_binding_matches_but_is_not_arithmetic_qualification():
    assert all(binding_checks(binding_row(), "content").values())


@pytest.mark.parametrize("field,bad_value,failed_check", [
    ("gpu_reference_owner_id", "wrong", "owner_binding"),
    ("gpu_reference_lease_id", "wrong", "lease_binding"),
    ("gpu_reference_adapter_int_id", 0, "native_adapter_binding"),
    ("native_clock_id", "other", "common_native_clock"),
    ("native_dispatch_monotonic_s", 1, "acquired_before_dispatch"),
    ("completion_token_ids_sha256", "wrong", "token_digest_transport"),
])
def test_binding_negatives(field, bad_value, failed_check):
    row = binding_row()
    row["native_token_timing"][field] = bad_value
    assert not binding_checks(row, "content")[failed_check]


def test_expected_content_is_independent_and_slot_is_checked():
    row = binding_row()
    assert not binding_checks(row, "independent_wrong")["frozen_content_binding"]
    row["gpu_reference_evidence"]["receipt"]["post_load_source_snapshot"]["slot_adapter_ids"] = [7]
    assert not binding_checks(row, "content")["slot_binding"]


def test_numerical_audit_rejects_duplicate_or_empty_blocks(tmp_path):
    path = tmp_path / 'not_read.json'
    for paths in ([], [path, path], [path] * 4):
        with pytest.raises(ValueError, match='distinct'):
            write_numerical_repeat_audit(paths, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_numerical_audit_rejects_incomplete_before_publishing(tmp_path):
    path = tmp_path / 'partial.json'
    path.write_text(json.dumps({'kind': 'backend_native_native_numerical_repeat_qualification_v1',
                                'pass': False, 'stage': 'failed'}))
    (tmp_path / 'launch.json').write_text(json.dumps({'pass': False}))
    with pytest.raises(ValueError, match='incomplete'):
        write_numerical_repeat_audit([path], tmp_path / 'out')
    assert not (tmp_path / 'out').exists()
