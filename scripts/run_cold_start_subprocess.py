#!/usr/bin/env python3
"""
cold_start 场景推理核心：支持独立子进程调用（CLI）和内联直接调用（run_cold_start_inline）。
用法: python run_cold_start_subprocess.py <model> <traces_json> <remote_dir> <nvme_dir> <bw_mbps> <output_json>
"""
import gc
import copy
import json
import os
import sys
import time
from pathlib import Path


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


def copy_with_timing(src: Path, dst: Path) -> tuple[bool, float]:
    t0 = time.perf_counter()
    try:
        if dst.exists():
            import shutil
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if src.is_dir():
            import shutil
            shutil.copytree(src, dst)
        else:
            import shutil
            shutil.copy2(src, dst)
        return True, (time.perf_counter() - t0) * 1000.0
    except Exception:
        return False, 0.0


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


def run_cold_start_inline(
    model_path: str,
    traces: list,
    remote_dir,
    nvme_dir,
    bw_mbps: float = 250.0,
    return_stats: bool = False,
):
    """
    Core inference logic for cold_start scenario.
    Can be called directly (no subprocess) or from main() via CLI.

    Args:
        model_path: path to the base language model
        traces: list of trace dicts [{prompt, adapter_id, max_tokens, temperature, top_p, size_mb}, ...]
        remote_dir: Path or str, remote adapter storage directory
        nvme_dir: Path or str, NVMe adapter cache directory
        bw_mbps: simulated bandwidth in Mbps for cold-start latency

    Returns:
        List of result dicts [{ttft_ms, tpot_ms, output_tokens, lora_io_ms}, ...]
    """
    total_t0 = time.perf_counter()
    _warm_page_cache(model_path)

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    remote_dir = Path(remote_dir)
    nvme_dir = Path(nvme_dir)
    nvme_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = _load_model_to_cuda(model_path, device="cuda:0")
    base_model.eval()

    max_input_len = 256
    max_tokens_default = 128
    peft_model = None
    loaded_adapter = None
    results = []
    serving_t0 = time.perf_counter()

    for i, tr in enumerate(traces):
        prompt = (tr.get("prompt") or "")[:1500]
        adapter_id = tr.get("adapter_id")
        max_tokens = int(tr.get("max_tokens", max_tokens_default))
        temperature = float(tr.get("temperature", 0.7))
        top_p = float(tr.get("top_p", 0.9))

        lora_path = None
        lora_io_ms = 0.0
        if adapter_id:
            src = remote_dir / adapter_id
            dst = nvme_dir / adapter_id
            t0 = time.perf_counter()
            ok, copy_ms = copy_with_timing(src, dst)
            if not ok:
                results.append({"ttft_ms": 0, "tpot_ms": 0, "output_tokens": 0, "lora_io_ms": 0, "error": "copy_failed"})
                continue
            if bw_mbps > 0:
                size_mb = tr.get("size_mb", 30)
                sleep_s = size_mb / bw_mbps
                if sleep_s > 0.001:
                    time.sleep(sleep_s)
                    copy_ms += sleep_s * 1000
            lora_io_ms = (time.perf_counter() - t0) * 1000.0
            lora_path = str(dst)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_len)
        input_ids = inputs["input_ids"].to("cuda:0")
        attention_mask = inputs["attention_mask"].to("cuda:0")

        model_to_use = base_model
        if lora_path and adapter_id:
            if peft_model is not None:
                try:
                    peft_model.delete_adapter(loaded_adapter)
                    gc.collect()
                    torch.cuda.empty_cache()
                    base_model = peft_model.unload()
                except Exception:
                    pass
                peft_model = None
                loaded_adapter = None
            peft_model = PeftModel.from_pretrained(
                base_model, lora_path, adapter_name=adapter_id, is_trainable=False
            )
            loaded_adapter = adapter_id
            model_to_use = peft_model

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model_to_use.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **_build_generate_kwargs(
                    model_to_use,
                    tokenizer,
                    max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                ),
            )
        t1 = time.perf_counter()
        output_tokens = max(int(out.shape[-1] - input_ids.shape[-1]), 1)
        infer_ms = (t1 - t0) * 1000.0
        per_token_ms = infer_ms / output_tokens
        del out, input_ids, attention_mask
        gc.collect()
        torch.cuda.empty_cache()

        ttft_ms = lora_io_ms + per_token_ms
        results.append({
            "ttft_ms": ttft_ms,
            "tpot_ms": per_token_ms,
            "output_tokens": output_tokens,
            "lora_io_ms": lora_io_ms,
        })
        if (i + 1) % 5 == 0:
            print(f"  progress {i + 1}/{len(traces)}", flush=True)

    serving_elapsed_sec = max(time.perf_counter() - serving_t0, 0.0)
    total_elapsed_sec = max(time.perf_counter() - total_t0, 0.0)

    # Free GPU memory after inference
    try:
        del peft_model, base_model, tokenizer
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
    if len(sys.argv) != 7:
        print(
            "Usage: run_cold_start_subprocess.py <model> <traces_json> <remote_dir> <nvme_dir> <bw_mbps> <output_json>",
            file=sys.stderr,
        )
        sys.exit(1)

    model_path = sys.argv[1]
    traces_path = sys.argv[2]
    remote_dir = sys.argv[3]
    nvme_dir = sys.argv[4]
    bw_mbps = float(sys.argv[5])
    output_path = sys.argv[6]

    with open(traces_path) as f:
        traces = json.load(f)

    results = run_cold_start_inline(model_path, traces, remote_dir, nvme_dir, bw_mbps)

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
