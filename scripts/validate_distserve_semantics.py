#!/usr/bin/env python3
"""Generate and verify frozen token references for DistServe."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.replay_distserve_trace import prepare_request
except ModuleNotFoundError:
    from replay_distserve_trace import prepare_request


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def select_prompt_quantiles(
    tokenizer: Any,
    trace_path: Path,
    *,
    sample_count: int,
    max_model_len: int,
    new_tokens: int,
) -> list[dict[str, Any]]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    candidates = [
        prepare_request(
            tokenizer,
            item,
            index=index,
            max_model_len=max_model_len,
            max_output_tokens_cap=new_tokens,
        )
        for index, item in enumerate(trace.get("requests") or [])
    ]
    if not candidates:
        raise ValueError(f"trace contains no requests: {trace_path}")
    candidates.sort(key=lambda item: (item["prompt_tokens"], item["index"]))

    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for position in range(sample_count):
        quantile = (
            0.5 if sample_count == 1 else position / (sample_count - 1)
        )
        candidate_index = round(quantile * (len(candidates) - 1))
        while candidate_index in seen and candidate_index + 1 < len(candidates):
            candidate_index += 1
        while candidate_index in seen and candidate_index > 0:
            candidate_index -= 1
        if candidate_index in seen:
            break
        seen.add(candidate_index)
        item = dict(candidates[candidate_index])
        item["semantic_output_tokens"] = new_tokens
        selected.append(item)
    if len(selected) != min(sample_count, len(candidates)):
        raise RuntimeError(
            f"failed to select semantic samples: {len(selected)}"
        )
    return selected


def generate_reference(args: argparse.Namespace) -> int:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )
    selected = select_prompt_quantiles(
        tokenizer,
        args.trace,
        sample_count=args.sample_count,
        max_model_len=args.max_model_len,
        new_tokens=args.new_tokens,
    )
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.to(args.device)
    model.eval()

    samples: list[dict[str, Any]] = []
    try:
        for sample in selected:
            encoded = tokenizer(
                sample["prompt"],
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"].to(args.device)
            attention_mask = encoded["attention_mask"].to(args.device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=args.new_tokens,
                    min_new_tokens=args.new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            output_ids = (
                generated[0, input_ids.shape[1] :]
                .detach()
                .to("cpu")
                .tolist()
            )
            if len(output_ids) != args.new_tokens:
                raise RuntimeError(
                    "reference generation returned the wrong token count: "
                    f"index={sample['index']} expected={args.new_tokens} "
                    f"observed={len(output_ids)}"
                )
            samples.append(
                {
                    "index": sample["index"],
                    "request_id": sample["request_id"],
                    "prompt": sample["prompt"],
                    "prompt_tokens": sample["prompt_tokens"],
                    "output_tokens": output_ids,
                    "generation_elapsed_sec": time.perf_counter() - started,
                }
            )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    payload = {
        "schema": "relayserve_distserve_semantic_reference_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace_path": str(args.trace.resolve()),
        "trace_sha256": sha256(args.trace),
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256(args.model / "config.json"),
        "tokenizer_path": str(args.tokenizer.resolve()),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "reference_backend": "transformers.AutoModelForCausalLM.generate",
        "generation_policy": {
            "do_sample": False,
            "new_tokens": args.new_tokens,
            "dtype": args.dtype,
            "attn_implementation": "eager",
        },
        "sample_selection": {
            "method": "prompt_token_length_quantiles",
            "requested_samples": args.sample_count,
            "selected_samples": len(samples),
            "max_model_len": args.max_model_len,
        },
        "samples": samples,
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": len(samples),
                "prompt_token_min": min(x["prompt_tokens"] for x in samples),
                "prompt_token_max": max(x["prompt_tokens"] for x in samples),
            },
            sort_keys=True,
        )
    )
    return 0


def request_distserve(
    base_url: str,
    prompt: str,
    new_tokens: int,
    timeout_s: float,
) -> tuple[int | None, dict[str, Any], str | None, float]:
    endpoint = f"{base_url.rstrip('/')}/generate"
    body = {
        "prompt": prompt,
        "n": 1,
        "best_of": 1,
        "use_beam_search": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": new_tokens,
        "ignore_eos": True,
        "stream": False,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return (
                int(response.status),
                payload,
                None,
                time.perf_counter() - started,
            )
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return (
            None,
            {},
            f"{type(exc).__name__}: {exc}",
            time.perf_counter() - started,
        )


def common_prefix_length(expected: list[int], observed: list[int]) -> int:
    length = 0
    for expected_token, observed_token in zip(expected, observed):
        if expected_token != observed_token:
            break
        length += 1
    return length


def compare_server(args: argparse.Namespace) -> int:
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    samples = list(reference.get("samples") or [])
    if not samples:
        raise ValueError("semantic reference contains no samples")

    reports: list[dict[str, Any]] = []
    matching_tokens = 0
    compared_tokens = 0
    for sample in samples:
        expected = [int(token) for token in sample["output_tokens"]]
        status, payload, error, elapsed = request_distserve(
            args.base_url,
            sample["prompt"],
            len(expected),
            args.timeout_s,
        )
        observed = [int(token) for token in payload.get("token_ids") or []]
        positional_matches = sum(
            expected_token == observed_token
            for expected_token, observed_token in zip(expected, observed)
        )
        compared_tokens += len(expected)
        matching_tokens += positional_matches
        exact = expected == observed
        reports.append(
            {
                "index": sample["index"],
                "request_id": sample["request_id"],
                "prompt_tokens": sample["prompt_tokens"],
                "status_code": status,
                "elapsed_sec": elapsed,
                "expected_token_ids": expected,
                "observed_token_ids": observed,
                "exact_match": exact,
                "first_token_match": bool(
                    expected and observed and expected[0] == observed[0]
                ),
                "common_prefix_tokens": common_prefix_length(
                    expected, observed
                ),
                "positional_token_matches": positional_matches,
                "error": error,
            }
        )

    exact_matches = sum(report["exact_match"] for report in reports)
    first_token_matches = sum(
        report["first_token_match"] for report in reports
    )
    exact_fraction = exact_matches / len(reports)
    first_token_fraction = first_token_matches / len(reports)
    token_fraction = (
        matching_tokens / compared_tokens if compared_tokens else 0.0
    )
    passed = (
        all(report["status_code"] == 200 and not report["error"] for report in reports)
        and exact_fraction >= args.required_exact_fraction
        and first_token_fraction >= args.required_first_token_fraction
        and token_fraction >= args.required_token_fraction
    )
    payload = {
        "schema": "relayserve_distserve_semantic_validation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_path": str(args.reference.resolve()),
        "reference_sha256": sha256(args.reference),
        "base_url": args.base_url,
        "sample_count": len(reports),
        "exact_matches": exact_matches,
        "exact_match_fraction": exact_fraction,
        "first_token_matches": first_token_matches,
        "first_token_match_fraction": first_token_fraction,
        "positional_token_matches": matching_tokens,
        "compared_tokens": compared_tokens,
        "positional_token_match_fraction": token_fraction,
        "thresholds": {
            "required_exact_fraction": args.required_exact_fraction,
            "required_first_token_fraction": (
                args.required_first_token_fraction
            ),
            "required_token_fraction": args.required_token_fraction,
        },
        "pass": passed,
        "samples": reports,
    }
    write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pass": passed,
                "exact_match_fraction": exact_fraction,
                "first_token_match_fraction": first_token_fraction,
                "positional_token_match_fraction": token_fraction,
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    reference = subparsers.add_parser("reference")
    reference.add_argument("--trace", type=Path, required=True)
    reference.add_argument("--model", type=Path, required=True)
    reference.add_argument("--tokenizer", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument("--sample-count", type=int, default=16)
    reference.add_argument("--new-tokens", type=int, default=8)
    reference.add_argument("--max-model-len", type=int, default=3072)
    reference.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    reference.add_argument("--device", default="cuda:0")
    reference.set_defaults(handler=generate_reference)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--base-url", required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--timeout-s", type=float, default=180.0)
    compare.add_argument("--required-exact-fraction", type=float, default=1.0)
    compare.add_argument(
        "--required-first-token-fraction", type=float, default=1.0
    )
    compare.add_argument("--required-token-fraction", type=float, default=1.0)
    compare.set_defaults(handler=compare_server)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for field in (
        "required_exact_fraction",
        "required_first_token_fraction",
        "required_token_fraction",
    ):
        if hasattr(args, field):
            value = float(getattr(args, field))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be in [0, 1]")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
