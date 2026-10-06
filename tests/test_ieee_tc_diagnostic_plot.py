"""Offline timing/plot guards; no GPU or scientific-effect qualification."""
import copy
import json

import pytest

from scripts.ieee_tc.plot_d290_diagnostic import load, percentile, request_vectors


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
