#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
import transformers
import yaml

from punica import (
    BatchedKvCache,
    BatchedLlamaLoraWeight,
    BatchLenInfo,
    KvCache,
    KvPool,
    LlamaForCausalLMWithLora,
    LlamaLoraWeight,
)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_profiles(
    cfg: Dict[str, Any],
    model_profile: str,
    dataset_profile: str,
    workload_profile: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    model_cfg = dict(cfg.get("model", {}) or {})
    adapters_cfg = dict(cfg.get("lora_adapters", {}) or {})
    datasets_cfg = dict(cfg.get("datasets", {}) or {})
    workload_cfg = dict(cfg.get("workload", {}) or {})
    coord_cfg = dict(cfg.get("resource_coordination", {}) or {})
    for bucket_name, selected in [
        ("model_profiles", model_profile),
        ("dataset_profiles", dataset_profile),
        ("workload_profiles", workload_profile),
    ]:
        profile = dict((cfg.get(bucket_name, {}) or {}).get(selected) or {})
        if not profile:
            raise KeyError(f"unknown profile '{selected}' in {bucket_name}")
        model_cfg = _deep_merge(model_cfg, profile.get("model", {}) or {})
        adapters_cfg = _deep_merge(adapters_cfg, profile.get("lora_adapters", {}) or {})
        datasets_cfg = _deep_merge(datasets_cfg, profile.get("datasets", {}) or {})
        workload_cfg = _deep_merge(workload_cfg, profile.get("workload", {}) or {})
        coord_cfg = _deep_merge(coord_cfg, profile.get("resource_coordination", {}) or {})
    return model_cfg, adapters_cfg, datasets_cfg, workload_cfg, coord_cfg


def _calc_cost(cost_model: Dict[str, Any], in_tok: int, out_tok: int) -> float:
    base = float(cost_model.get("base_cost_usd", 0.001) or 0.0)
    in_c = float(cost_model.get("input_token_cost_usd", 0.0000015) or 0.0) * int(in_tok)
    out_c = float(cost_model.get("output_token_cost_usd", 0.000002) or 0.0) * int(out_tok)
    return base + in_c + out_c


def _render_chat_messages_prompt(
    tokenizer: transformers.PreTrainedTokenizerBase,
    messages: Optional[List[Dict[str, Any]]],
    fallback_prompt: str,
) -> str:
    if not messages:
        return fallback_prompt
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        if isinstance(rendered, str) and rendered.strip():
            return rendered
    except Exception:
        pass
    lines: List[str] = []
    for message in messages:
        item = dict(message or {})
        role = str(item.get("role") or "user").strip().capitalize() or "User"
        content = item.get("content")
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        lines.append(f"{role}: {content}")
    fallback = "\n".join(lines).strip()
    return fallback or fallback_prompt


def _prepare_prompt(
    *,
    tokenizer: transformers.PreTrainedTokenizerBase,
    model_cfg: Dict[str, Any],
    messages: Optional[List[Dict[str, Any]]],
    fallback_prompt: str,
    max_tokens: int,
    input_tokens_hint: int,
) -> Tuple[str, List[int], int]:
    prompt = _render_chat_messages_prompt(tokenizer, messages, fallback_prompt)
    max_tokens = max(1, int(max_tokens or 1))
    cap = int(model_cfg.get("max_output_tokens_cap", 0) or 0)
    if cap > 0:
        max_tokens = min(max_tokens, cap)
    max_len = max(32, int(model_cfg.get("max_model_len", 2048) or 2048))
    max_input_len = max(0, int(model_cfg.get("max_input_len", 0) or 0))
    prompt_budget = max(8, max_len - max_tokens - 8)
    if max_input_len > 0:
        prompt_budget = min(prompt_budget, max_input_len)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) > prompt_budget:
        token_ids = token_ids[-prompt_budget:]
        prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
    actual_input_tokens = max(1, len(token_ids) if token_ids else int(input_tokens_hint or 1))
    safe_max_tokens = min(max_tokens, max(1, max_len - actual_input_tokens - 8))
    return prompt, token_ids, safe_max_tokens


class TextGeneration:
    def __init__(
        self,
        input_ids: List[int],
        *,
        temperature: float,
        repetition_penalty: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        stop_token_id: int,
    ) -> None:
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.stop_token_id = stop_token_id

        self.logits_processor = transformers.LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            self.logits_processor.append(
                transformers.TemperatureLogitsWarper(temperature)
            )
        if repetition_penalty > 1.0:
            self.logits_processor.append(
                transformers.RepetitionPenaltyLogitsProcessor(repetition_penalty)
            )
        if 0 < top_p < 1.0:
            self.logits_processor.append(transformers.TopPLogitsWarper(top_p))
        if top_k > 0:
            self.logits_processor.append(transformers.TopKLogitsWarper(top_k))

        self.output_ids = [int(x) for x in input_ids]
        self.prompt_len = len(self.output_ids)

    def get_next_token_id(self, logits_row: torch.Tensor) -> int:
        if self.logits_processor:
            if self.repetition_penalty > 1.0:
                t = torch.as_tensor([self.output_ids], device=logits_row.device)
            else:
                t = None
            last_token_logits = self.logits_processor(t, logits_row.unsqueeze(0))[0]
        else:
            last_token_logits = logits_row

        if self.temperature <= 0 or self.top_p <= 0:
            _, indices = torch.topk(last_token_logits, 2)
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
        return int(indices.tolist()[0])

    def append_token(self, token_id: int) -> None:
        self.output_ids.append(token_id)

    def is_stop(self) -> bool:
        if len(self.output_ids) - self.prompt_len >= self.max_new_tokens:
            return True
        if self.output_ids[-1] == self.stop_token_id:
            return True
        return False


@dataclass
class RequestPlan:
    request_id: str
    adapter_id: str
    adapter_domain: Optional[str]
    arrival_time_s: float
    prompt_text: str
    prompt_ids: List[int]
    prompt_tokens: int
    max_new_tokens: int
    temperature: float
    top_p: float


@dataclass
class RequestState:
    plan: RequestPlan
    textgen: TextGeneration
    kvcache: KvCache
    queued_at_s: float
    scheduled_at_s: Optional[float] = None
    lora_load_ms: float = 0.0
    gpu_cache_hit: Optional[bool] = None
    first_token_at_s: Optional[float] = None
    last_token_at_s: Optional[float] = None
    completion_at_s: Optional[float] = None
    success: bool = True
    error: Optional[str] = None


class PunicaLoraStore:
    def __init__(
        self,
        *,
        manifest_path: Path,
        llama_config: transformers.LlamaConfig,
        dtype: torch.dtype,
        device: torch.device,
        cpu_cache_size: int,
        gpu_cache_size: int,
    ) -> None:
        manifest = _load_json(manifest_path)
        self.entries = {
            str(item["id"]): dict(item)
            for item in manifest.get("adapters", []) or []
        }
        self.llama_config = llama_config
        self.dtype = dtype
        self.device = device
        self.cpu_cache_size = max(1, int(cpu_cache_size))
        self.gpu_cache_size = max(1, int(gpu_cache_size))
        self.cpu_cache: OrderedDict[str, Dict[str, torch.Tensor]] = OrderedDict()
        self.gpu_cache: OrderedDict[str, LlamaLoraWeight] = OrderedDict()

    def _load_cpu_tensors(self, adapter_id: str) -> Dict[str, torch.Tensor]:
        cached = self.cpu_cache.get(adapter_id)
        if cached is not None:
            self.cpu_cache.move_to_end(adapter_id)
            return cached
        entry = self.entries[adapter_id]
        tensors = torch.load(entry["punica_weight"], map_location="cpu", weights_only=True)
        self.cpu_cache[adapter_id] = tensors
        while len(self.cpu_cache) > self.cpu_cache_size:
            self.cpu_cache.popitem(last=False)
        return tensors

    def ensure_gpu(self, adapter_id: str) -> Tuple[LlamaLoraWeight, float, bool]:
        cached = self.gpu_cache.get(adapter_id)
        if cached is not None:
            self.gpu_cache.move_to_end(adapter_id)
            return cached, 0.0, True
        entry = self.entries[adapter_id]
        lora_rank = int(entry.get("punica_rank") or entry.get("lora_rank"))
        cpu_tensors = self._load_cpu_tensors(adapter_id)
        start = time.perf_counter()
        weight = LlamaLoraWeight(self.llama_config, lora_rank, self.dtype, self.device)
        weight.copy_from_tensors(cpu_tensors)
        load_ms = (time.perf_counter() - start) * 1000.0
        self.gpu_cache[adapter_id] = weight
        while len(self.gpu_cache) > self.gpu_cache_size:
            old_adapter_id, old_weight = self.gpu_cache.popitem(last=False)
            del old_weight
            if old_adapter_id != adapter_id:
                torch.cuda.empty_cache()
        return weight, load_ms, False


def _build_batched_lora(
    ordered_requests: List[RequestState],
    decode_count: int,
    weight_map: Dict[str, LlamaLoraWeight],
) -> BatchedLlamaLoraWeight:
    segments: List[Tuple[str, int]] = []
    for req in ordered_requests:
        seg_len = len(req.plan.prompt_ids) if req.first_token_at_s is None else 1
        adapter_id = req.plan.adapter_id
        if segments and segments[-1][0] == adapter_id:
            segments[-1] = (adapter_id, segments[-1][1] + seg_len)
        else:
            segments.append((adapter_id, seg_len))
    return BatchedLlamaLoraWeight(
        [weight_map[adapter_id] for adapter_id, _ in segments],
        [seg_len for _adapter_id, seg_len in segments],
    )


def _maybe_print_live(
    *,
    run_tag: str,
    total_requests: int,
    start_time: float,
    finished: List[Dict[str, Any]],
    workset: List[RequestState],
    wait_queue: Deque[RequestState],
    last_print_at: float,
    live_interval_s: float,
) -> float:
    now = time.perf_counter()
    if now - last_print_at < live_interval_s:
        return last_print_at
    done = len(finished)
    ok = sum(1 for item in finished if bool(item.get("success")))
    fail = done - ok
    elapsed = now - start_time
    reqps = ok / elapsed if elapsed > 0 else 0.0
    tokps = (
        sum(int(item.get("completion_tokens", 0) or 0) for item in finished if bool(item.get("success"))) / elapsed
        if elapsed > 0
        else 0.0
    )
    width = 30
    filled = min(width, int(width * done / max(total_requests, 1)))
    bar = "#" * filled + "-" * (width - filled)
    remaining = max(total_requests - done, 0)
    eta = remaining / reqps if reqps > 1e-6 else math.inf
    eta_s = "--:--" if not math.isfinite(eta) else f"{int(eta // 60):02d}:{int(eta % 60):02d}"
    elapsed_s = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
    print(
        f"[live:{run_tag}] [{bar}] done={done}/{total_requests} ok={ok} fail={fail} "
        f"active={len(workset)} queued={len(wait_queue)} elapsed={elapsed_s} eta={eta_s} "
        f"req/s={reqps:.2f} tok/s={tokps:.2f}",
        flush=True,
    )
    return now


def _finalize_request(
    *,
    req: RequestState,
    wall_start: float,
    cost_model: Dict[str, Any],
) -> Dict[str, Any]:
    prompt_tokens = int(req.plan.prompt_tokens)
    completion_tokens = max(0, len(req.textgen.output_ids) - req.textgen.prompt_len)
    total_tokens = prompt_tokens + completion_tokens
    ttft_ms = None
    tpot_ms = None
    e2e_ms = None
    if req.first_token_at_s is not None:
        ttft_ms = max(0.0, (req.first_token_at_s - req.plan.arrival_time_s) * 1000.0)
    if req.completion_at_s is not None:
        e2e_ms = max(0.0, (req.completion_at_s - req.plan.arrival_time_s) * 1000.0)
    if (
        req.first_token_at_s is not None
        and req.last_token_at_s is not None
        and completion_tokens > 1
    ):
        tpot_ms = ((req.last_token_at_s - req.first_token_at_s) * 1000.0) / float(completion_tokens - 1)
    queue_wait_ms = None
    if req.scheduled_at_s is not None:
        queue_wait_ms = max(0.0, (req.scheduled_at_s - req.plan.arrival_time_s) * 1000.0)
    return {
        "request_id": req.plan.request_id,
        "adapter_id": req.plan.adapter_id,
        "adapter_domain": req.plan.adapter_domain,
        "success": bool(req.success),
        "error": req.error,
        "arrival_offset_s": round(float(req.plan.arrival_time_s), 6),
        "scheduled_offset_s": round(float(req.scheduled_at_s), 6) if req.scheduled_at_s is not None else None,
        "first_token_offset_s": round(float(req.first_token_at_s), 6) if req.first_token_at_s is not None else None,
        "completion_offset_s": round(float(req.completion_at_s), 6) if req.completion_at_s is not None else None,
        "ttft_ms": round(float(ttft_ms), 4) if ttft_ms is not None else None,
        "tpot_ms": round(float(tpot_ms), 4) if tpot_ms is not None else None,
        "tpot_observed": tpot_ms is not None,
        "e2e_ms": round(float(e2e_ms), 4) if e2e_ms is not None else None,
        "queue_wait_ms": round(float(queue_wait_ms), 4) if queue_wait_ms is not None else None,
        "lora_load_ms": round(float(req.lora_load_ms), 4),
        "gpu_cache_hit": req.gpu_cache_hit,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(_calc_cost(cost_model, prompt_tokens, completion_tokens), 8) if req.success else 0.0,
        "runtime_ttft_ms": None,
        "serverless_overhead_ms": None,
        "cold_start_latency_ms": None,
        "comparable_request": None,
        "warm_standard_request": None,
        "gpu_ready_request": None,
        "scaleup_affected": None,
        "scaleup_first_service": None,
        "cache_hit": req.gpu_cache_hit,
    }


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser(description="Replay a shared fair-round trace on Punica.")
    ap.add_argument("--main-repo", type=Path, default=Path("/home/qhq/serverless_llm_experiment_retry14_baseline"))
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--punica-lora-manifest", type=Path, required=True)
    ap.add_argument("--model-profile", required=True)
    ap.add_argument("--dataset-profile", required=True)
    ap.add_argument("--workload-profile", required=True)
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--cpu-lora-cache-size", type=int, default=128)
    ap.add_argument("--gpu-lora-cache-size", type=int, default=8)
    ap.add_argument("--live-interval-s", type=float, default=2.0)
    args = ap.parse_args()

    torch.manual_seed(0xABCDABCD987)

    main_repo = args.main_repo.resolve()
    cfg_path = args.config.resolve() if args.config else (main_repo / "configs/experiments.yaml")
    cfg = _load_yaml(cfg_path)
    model_cfg, _adapters_cfg, _datasets_cfg, _workload_cfg, _coord_cfg = _resolve_profiles(
        cfg, args.model_profile, args.dataset_profile, args.workload_profile
    )
    cost_model = dict(cfg.get("cost_model", {}) or {})

    model_name = str(model_cfg.get("name") or "")
    if "llama" not in model_name.lower():
        raise RuntimeError("Punica reproduction currently supports only Llama-family backbones.")
    if int(model_cfg.get("tensor_parallel_size", 1) or 1) != 1:
        raise RuntimeError("Punica reproduction currently supports only tensor_parallel_size=1 profiles.")

    trace_payload = _load_json(args.trace.resolve())
    requests = list(trace_payload.get("requests", []) or [])
    total_requests = int(trace_payload.get("total_requests", len(requests)) or len(requests))
    if total_requests != len(requests):
        raise RuntimeError(f"trace total_requests mismatch: header={total_requests}, actual={len(requests)}")

    device = torch.device("cuda:0")
    dtype = torch.float16
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    llama_config = transformers.LlamaConfig.from_pretrained(model_name)
    model = LlamaForCausalLMWithLora.from_pretrained(
        model_name, low_cpu_mem_usage=True, torch_dtype=dtype
    ).to(device)
    kvpool = KvPool(
        num_layers=llama_config.num_hidden_layers,
        num_heads=llama_config.num_attention_heads,
        head_dim=llama_config.hidden_size // llama_config.num_attention_heads,
        page_len=16,
        dtype=dtype,
        device=device,
    )
    lora_store = PunicaLoraStore(
        manifest_path=args.punica_lora_manifest.resolve(),
        llama_config=llama_config,
        dtype=dtype,
        device=device,
        cpu_cache_size=args.cpu_lora_cache_size,
        gpu_cache_size=args.gpu_lora_cache_size,
    )

    plans: List[RequestPlan] = []
    for raw in sorted(requests, key=lambda item: float(item.get("arrival_time_s", 0.0) or 0.0)):
        body = dict(raw.get("body") or {})
        adapter_id = str(raw.get("adapter_id") or body.get("lora_adapter_name") or "")
        if not adapter_id:
            raise RuntimeError(f"trace request missing adapter_id: {raw.get('request_id')}")
        if adapter_id not in lora_store.entries:
            raise RuntimeError(f"Punica LoRA manifest missing adapter {adapter_id}")
        fallback_prompt = str(raw.get("prompt") or "")
        prompt_text, prompt_ids, safe_max_tokens = _prepare_prompt(
            tokenizer=tokenizer,
            model_cfg=model_cfg,
            messages=body.get("messages"),
            fallback_prompt=fallback_prompt,
            max_tokens=int(body.get("max_tokens", raw.get("expected_output_tokens", 1)) or 1),
            input_tokens_hint=int(raw.get("prompt_input_tokens", raw.get("expected_input_tokens", 1)) or 1),
        )
        plans.append(
            RequestPlan(
                request_id=str(raw.get("request_id") or f"req_{len(plans)}"),
                adapter_id=adapter_id,
                adapter_domain=raw.get("adapter_domain"),
                arrival_time_s=float(raw.get("arrival_time_s", 0.0) or 0.0),
                prompt_text=prompt_text,
                prompt_ids=prompt_ids,
                prompt_tokens=max(1, len(prompt_ids)),
                max_new_tokens=max(1, int(safe_max_tokens)),
                temperature=float(body.get("temperature", 0.7) or 0.0),
                top_p=float(body.get("top_p", 0.9) or 0.0),
            )
        )

    pending: Deque[RequestPlan] = deque(plans)
    wait_queue: Deque[RequestState] = deque()
    workset: List[RequestState] = []
    finished: List[Dict[str, Any]] = []

    wall_start = time.perf_counter()
    last_live = 0.0
    stop_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

    while len(finished) < total_requests:
        elapsed = time.perf_counter() - wall_start
        while pending and pending[0].arrival_time_s <= elapsed:
            plan = pending.popleft()
            wait_queue.append(
                RequestState(
                    plan=plan,
                    textgen=TextGeneration(
                        plan.prompt_ids,
                        temperature=plan.temperature,
                        repetition_penalty=1.0,
                        top_p=plan.top_p,
                        top_k=-1,
                        max_new_tokens=plan.max_new_tokens,
                        stop_token_id=stop_token_id,
                    ),
                    kvcache=KvCache(kvpool, len(plan.prompt_ids)),
                    queued_at_s=elapsed,
                )
            )

        newreqs: List[RequestState] = []
        while wait_queue and len(workset) + len(newreqs) < int(args.batch_size):
            req = wait_queue.popleft()
            req.scheduled_at_s = time.perf_counter() - wall_start
            _weight, load_ms, hit = lora_store.ensure_gpu(req.plan.adapter_id)
            req.lora_load_ms += load_ms
            req.gpu_cache_hit = hit
            newreqs.append(req)

        if not workset and not newreqs:
            if pending:
                sleep_s = max(0.0, pending[0].arrival_time_s - (time.perf_counter() - wall_start))
                if sleep_s > 0:
                    time.sleep(min(sleep_s, 0.01))
                last_live = _maybe_print_live(
                    run_tag=args.run_tag,
                    total_requests=total_requests,
                    start_time=wall_start,
                    finished=finished,
                    workset=workset,
                    wait_queue=wait_queue,
                    last_print_at=last_live,
                    live_interval_s=args.live_interval_s,
                )
                continue
            break

        ordered_batch: List[RequestState] = [*newreqs, *workset]
        weight_map: Dict[str, LlamaLoraWeight] = {}
        for req in ordered_batch:
            weight, _load_ms, _hit = lora_store.ensure_gpu(req.plan.adapter_id)
            weight_map[req.plan.adapter_id] = weight

        input_ids: List[int] = []
        prefills: List[int] = []
        for req in newreqs:
            input_ids.extend(req.plan.prompt_ids)
            prefills.append(len(req.plan.prompt_ids))
        for req in workset:
            input_ids.append(req.textgen.output_ids[-1])

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
        blen = BatchLenInfo(prefills=prefills, decode=len(workset), indptr_device=device)
        prefill_kv = BatchedKvCache([req.kvcache for req in newreqs]) if newreqs else None
        decode_kv = BatchedKvCache([req.kvcache for req in workset]) if workset else None
        lora = _build_batched_lora(ordered_batch, len(workset), weight_map)

        t1 = time.perf_counter()
        logits, _ = model(
            input_ids=input_ids_tensor,
            blen=blen,
            prefill_kv=prefill_kv,
            decode_kv=decode_kv,
            lora=lora,
        )
        t2 = time.perf_counter()
        offset_s = t2 - wall_start

        next_workset: List[RequestState] = []
        if newreqs:
            last_token_indices = (blen.indptr[1:] - 1).tolist()
            for batch_idx, req in enumerate(newreqs):
                next_token = req.textgen.get_next_token_id(logits[last_token_indices[batch_idx]])
                req.textgen.append_token(next_token)
                req.first_token_at_s = offset_s
                req.last_token_at_s = offset_s
                if req.textgen.is_stop():
                    req.completion_at_s = offset_s
                    req.kvcache.release()
                    finished.append(_finalize_request(req=req, wall_start=wall_start, cost_model=cost_model))
                else:
                    req.kvcache.acquire_one()
                    next_workset.append(req)

        if workset:
            decode_logits = logits[blen.doff :]
            for batch_idx, req in enumerate(workset):
                next_token = req.textgen.get_next_token_id(decode_logits[batch_idx])
                req.textgen.append_token(next_token)
                req.last_token_at_s = offset_s
                if req.textgen.is_stop():
                    req.completion_at_s = offset_s
                    req.kvcache.release()
                    finished.append(_finalize_request(req=req, wall_start=wall_start, cost_model=cost_model))
                else:
                    req.kvcache.acquire_one()
                    next_workset.append(req)

        workset = next_workset
        last_live = _maybe_print_live(
            run_tag=args.run_tag,
            total_requests=total_requests,
            start_time=wall_start,
            finished=finished,
            workset=workset,
            wait_queue=wait_queue,
            last_print_at=last_live,
            live_interval_s=args.live_interval_s,
        )

    elapsed_sec = time.perf_counter() - wall_start
    payload = {
        "metadata": {
            "system": "punica",
            "run_tag": args.run_tag,
            "model_profile": args.model_profile,
            "dataset_profile": args.dataset_profile,
            "workload_profile": args.workload_profile,
            "base_model": model_name,
            "batch_size": int(args.batch_size),
            "cpu_lora_cache_size": int(args.cpu_lora_cache_size),
            "gpu_lora_cache_size": int(args.gpu_lora_cache_size),
            "device": str(device),
            "elapsed_sec": round(elapsed_sec, 6),
        },
        "results": finished,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"punica replay -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
