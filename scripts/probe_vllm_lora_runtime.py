#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_probe_request(trace_payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    requests = list(trace_payload.get("requests", []) or [])
    if not requests:
        raise RuntimeError("trace artifact contains no requests")
    candidate = None
    best_tokens = -1
    for item in requests:
        body = dict(item.get("body") or {})
        adapter_name = body.get("lora_adapter_name") or item.get("adapter_id")
        if not adapter_name:
            continue
        prompt_tokens = int(item.get("prompt_input_tokens", 0) or 0)
        if prompt_tokens > best_tokens:
            candidate = item
            best_tokens = prompt_tokens
    if candidate is None:
        raise RuntimeError("trace artifact contains no LoRA-bound requests for probing")
    return candidate, dict(candidate.get("body") or {})


def _render_messages_prompt(messages: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{str(message.get('role') or 'user').capitalize()}: "
        f"{'' if message.get('content') is None else str(message.get('content'))}"
        for message in messages
    )


def _prepare_prompt(tokenizer, prompt: str, max_model_len: int, max_tokens: int) -> Tuple[str, int]:
    prompt_budget = max(8, int(max_model_len) - int(max_tokens) - 8)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) > prompt_budget:
        token_ids = token_ids[-prompt_budget:]
        prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
    return prompt, len(token_ids)


def _fail(message: str, **payload: Any) -> int:
    result = {"ok": False, "reason": message, **payload}
    print(json.dumps(result, ensure_ascii=False))
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe whether vLLM can serve the selected many-LoRA workload correctly.")
    ap.add_argument("--deploy", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--probe-max-tokens", type=int, default=16)
    args = ap.parse_args()

    deploy_payload = _load_json(args.deploy.resolve())
    trace_payload = _load_json(args.trace.resolve())
    backend_config = dict(deploy_payload.get("backend_config", {}) or {})
    if not backend_config.get("enable_lora", False):
        return _fail("deploy config does not enable LoRA")

    model_path = str(backend_config.get("pretrained_model_name_or_path") or "")
    if not model_path:
        return _fail("missing pretrained_model_name_or_path")

    if backend_config.get("disable_lora_embeddings", False):
        os.environ["VLLM_DISABLE_LORA_EMBEDDINGS"] = "1"
    else:
        os.environ.pop("VLLM_DISABLE_LORA_EMBEDDINGS", None)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    req_meta, body = _pick_probe_request(trace_payload)
    lora_name = str(body.get("lora_adapter_name") or req_meta.get("adapter_id") or "")
    lora_path = str((backend_config.get("lora_adapters", {}) or {}).get(lora_name) or "")
    if not lora_path:
        return _fail("probe adapter not found in deploy config", adapter=lora_name)
    if not os.path.isabs(lora_path):
        storage_path = (
            os.getenv("STORAGE_PATH")
            or os.getenv("SLLM_STORAGE_PATH")
            or "/home/qhq/serverless_llm_baselines/models"
        )
        lora_path = os.path.join(storage_path, lora_path)
    if not os.path.exists(lora_path):
        return _fail("probe adapter path does not exist", adapter=lora_name, lora_path=lora_path)

    messages = list(body.get("messages", []) or [])
    prompt = body.get("prompt")
    if prompt is None:
        if not messages:
            return _fail("probe request body contains neither prompt nor messages")
        prompt = _render_messages_prompt(messages)

    requested_max_tokens = int(body.get("max_tokens", args.probe_max_tokens) or args.probe_max_tokens)
    requested_max_tokens = max(1, min(requested_max_tokens, int(args.probe_max_tokens)))

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt, prompt_tokens = _prepare_prompt(
        tokenizer,
        str(prompt),
        int(backend_config.get("max_model_len", 2048) or 2048),
        requested_max_tokens,
    )

    try:
        llm = LLM(
            model=model_path,
            tokenizer=model_path,
            dtype=str(backend_config.get("torch_dtype", "float16")),
            tensor_parallel_size=int(backend_config.get("tensor_parallel_size", 1) or 1),
            gpu_memory_utilization=float(backend_config.get("gpu_memory_utilization", 0.7) or 0.7),
            max_model_len=int(backend_config.get("max_model_len", 2048) or 2048),
            max_num_seqs=int(backend_config.get("max_num_seqs", 1) or 1),
            max_num_batched_tokens=int(backend_config.get("max_num_batched_tokens", 512) or 512),
            enable_lora=True,
            max_loras=int(backend_config.get("max_loras", 1) or 1),
            max_lora_rank=int(backend_config.get("max_lora_rank", 16) or 16),
            enable_chunked_prefill=bool(backend_config.get("enable_chunked_prefill", False)),
            enable_prefix_caching=bool(backend_config.get("enable_prefix_caching", False)),
            enforce_eager=bool(backend_config.get("enforce_eager", False)),
            disable_log_stats=True,
        )
        outputs = llm.generate(
            prompts=[prompt],
            sampling_params=SamplingParams(
                temperature=float(body.get("temperature", 0.0) or 0.0),
                top_p=float(body.get("top_p", 1.0) or 1.0),
                max_tokens=requested_max_tokens,
            ),
            lora_request=LoRARequest(
                lora_name=lora_name,
                lora_int_id=1,
                lora_path=lora_path,
                base_model_name=model_path,
            ),
        )
    except Exception as exc:
        return _fail(
            "vllm probe raised exception",
            adapter=lora_name,
            prompt_tokens=prompt_tokens,
            error=str(exc),
        )

    if not outputs or not outputs[0].outputs:
        return _fail("vllm probe returned no outputs", adapter=lora_name, prompt_tokens=prompt_tokens)

    output = outputs[0].outputs[0]
    token_ids = list(output.token_ids or [])
    text = str(output.text or "")

    if not token_ids:
        return _fail("vllm probe returned zero completion tokens", adapter=lora_name, prompt_tokens=prompt_tokens)
    if all(int(token_id) == 0 for token_id in token_ids):
        return _fail(
            "vllm probe produced all-zero token ids",
            adapter=lora_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(token_ids),
        )
    if not text.strip():
        return _fail(
            "vllm probe produced empty decoded text",
            adapter=lora_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(token_ids),
            sample_token_ids=token_ids[:16],
        )

    result = {
        "ok": True,
        "adapter": lora_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": len(token_ids),
        "sample_text": text[:200],
        "sample_token_ids": token_ids[:16],
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
