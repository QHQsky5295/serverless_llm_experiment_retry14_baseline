#!/usr/bin/env python3
"""
backbone_only 场景推理核心：支持独立子进程调用（CLI）和内联直接调用（run_backbone_inline）。
Usage: python run_transformers_subprocess.py <model_path> <input_json> <output_json>
  input_json: [{"prompt": "...", "max_tokens": 128, "temperature": 0.7, "top_p": 0.9}, ...]
  output_json: [{"ttft_ms": float, "tpot_ms": float, "output_tokens": int}, ...]
"""
import gc
import copy
import json
import os
import sys
import time


def _load_model_to_cuda(model_path: str, device: str = "cuda:0"):
    """Load model to GPU with eager attention to avoid Triton JIT compilation."""
    from transformers import AutoModelForCausalLM
    import torch

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    device_id = int(device.split(":")[-1]) if ":" in device else 0
    load_kwargs = {
        "trust_remote_code": True,
        "device_map": {"": device_id},
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            **load_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            **load_kwargs,
        )
    model.eval()
    return model


def _warm_page_cache(model_path: str) -> None:
    """Pre-read model safetensors into page cache to prevent PSI spikes during model load."""
    import glob
    buf = bytearray(2 * 1024 * 1024)
    t0 = time.perf_counter()
    for sf_file in sorted(glob.glob(f"{model_path}/*.safetensors")):
        try:
            with open(sf_file, "rb") as fh:
                while fh.readinto(buf):
                    pass
        except Exception:
            pass
    if time.perf_counter() - t0 > 2.0:
        time.sleep(3)


def _build_generate_kwargs(model, tokenizer, max_new_tokens: int, *, temperature: float, top_p: float) -> dict:
    """Build warning-free HF generate kwargs for either sampling or greedy decoding."""
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0.0:
        kwargs.update(
            do_sample=True,
            temperature=max(float(temperature), 1e-5),
            top_p=min(max(float(top_p), 1e-5), 1.0),
        )
        return kwargs

    generation_config = copy.deepcopy(getattr(model, "generation_config", None))
    if generation_config is not None:
        generation_config.do_sample = False
        if hasattr(generation_config, "temperature"):
            generation_config.temperature = 1.0
        if hasattr(generation_config, "top_p"):
            generation_config.top_p = 1.0
        if hasattr(generation_config, "top_k"):
            generation_config.top_k = 50
        kwargs["generation_config"] = generation_config
    else:
        kwargs["do_sample"] = False
    return kwargs


def _tokenize_prompt(tokenizer, prompt: str, max_input_len: int):
    kwargs = {"return_tensors": "pt"}
    if max_input_len > 0:
        kwargs.update(truncation=True, max_length=max_input_len)
    return tokenizer(prompt, **kwargs)


def run_backbone_inline(
    model_path: str,
    requests: list,
    *,
    max_input_len: int = 0,
    max_output_cap: int = 0,
    return_stats: bool = False,
):
    """
    Core inference logic for backbone_only scenario.
    Can be called directly (no subprocess) or from main() via CLI.

    Args:
        model_path: path to the base language model
        requests: list of request dicts [{prompt, max_tokens, temperature, top_p}, ...]

    Returns:
        List of result dicts [{ttft_ms, tpot_ms, output_tokens}, ...]
    """
    total_t0 = time.perf_counter()
    _warm_page_cache(model_path)

    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model_to_cuda(model_path, device="cuda:0")
    model.eval()

    max_input_len = max(0, int(max_input_len or 0))
    max_output_cap = max(0, int(max_output_cap or 0))
    results = []
    serving_t0 = time.perf_counter()
    for i, req in enumerate(requests):
        prompt = req.get("prompt", "")
        max_tokens = int(req.get("max_tokens", 128))
        if max_output_cap > 0:
            max_tokens = min(max_tokens, max_output_cap)
        temperature = float(req.get("temperature", 0.7))
        top_p = float(req.get("top_p", 0.9))
        inputs = _tokenize_prompt(tokenizer, prompt, max_input_len)
        input_ids = inputs["input_ids"].to("cuda:0")
        attention_mask = inputs["attention_mask"].to("cuda:0")
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **_build_generate_kwargs(
                    model,
                    tokenizer,
                    max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                ),
            )
        t1 = time.perf_counter()
        output_tokens = max(int(out.shape[-1] - input_ids.shape[-1]), 1)
        total_ms = (t1 - t0) * 1000.0
        per_token_ms = total_ms / output_tokens
        del out, input_ids, attention_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results.append({"ttft_ms": per_token_ms, "tpot_ms": per_token_ms, "output_tokens": output_tokens})
        if (i + 1) % 5 == 0:
            print(f"  progress {i + 1}/{len(requests)}", flush=True)

    serving_elapsed_sec = max(time.perf_counter() - serving_t0, 0.0)
    total_elapsed_sec = max(time.perf_counter() - total_t0, 0.0)

    # Free GPU memory after inference
    try:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    stats = {
        "setup_elapsed_sec": max(serving_t0 - total_t0, 0.0),
        "serving_elapsed_sec": serving_elapsed_sec,
        "total_elapsed_sec": total_elapsed_sec,
    }
    if return_stats:
        return results, stats
    return results


def main():
    """CLI entry point: still usable as subprocess if needed."""
    if len(sys.argv) != 4:
        print("Usage: run_transformers_subprocess.py <model_path> <input_json> <output_json>", file=sys.stderr)
        sys.exit(1)
    model_path = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    with open(input_path) as f:
        requests = json.load(f)

    max_input_len = int(os.environ.get("FAASLORA_MAX_INPUT_LEN", "0") or 0)
    max_output_cap = int(os.environ.get("FAASLORA_MAX_OUTPUT_TOKENS_CAP", "0") or 0)
    results = run_backbone_inline(
        model_path,
        requests,
        max_input_len=max_input_len,
        max_output_cap=max_output_cap,
    )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=0)
    return 0


if __name__ == "__main__":
    # Only set oom_score_adj when running as a subprocess (CLI)
    try:
        with open("/proc/self/oom_score_adj", "w") as _f:
            _f.write("0")
    except Exception:
        pass

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    sys.exit(main())
