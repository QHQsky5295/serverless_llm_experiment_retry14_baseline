#!/usr/bin/env python3
"""
faaslora_full 场景推理核心：支持独立子进程调用（CLI）和内联直接调用（run_inference_inline）。
用法: python run_faaslora_subprocess.py <model> <traces_json> <remote_dir> <nvme_dir> <adapter_info_json> <output_json>
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
        # Backward-compat for older transformers that still use torch_dtype.
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
    """Pre-read model safetensors files into the OS page cache.

    WHY THIS IS NECESSARY:
    systemd-oomd monitors the user's session cgroup (ManagedOOMMemoryPressure=kill).
    When the model files are not in page cache, reading them triggers major page faults,
    which stall the process and accumulate in the cgroup's memory.pressure PSI metric.
    If PSI avg10 > 60% for 30 consecutive seconds, oomd sends SIGKILL.

    By pre-reading the files into page cache BEFORE launching heavy Python/CUDA operations,
    we separate the disk I/O (high PSI) from the CUDA loading (low PSI).
    The gap between pre-read completion and model loading lets PSI avg10 drop below 60%,
    resetting oomd's 30-second timer.  The model then loads from page cache with ~0 PSI.
    """
    import glob
    total_bytes = 0
    t0 = time.perf_counter()
    buf = bytearray(2 * 1024 * 1024)  # 2 MB read buffer (minimal heap footprint)
    for sf_file in sorted(glob.glob(f"{model_path}/*.safetensors")):
        try:
            with open(sf_file, "rb") as fh:
                while True:
                    n = fh.readinto(buf)
                    if not n:
                        break
                    total_bytes += n
        except Exception:
            pass
    elapsed = time.perf_counter() - t0
    cold_read = elapsed > 2.0
    print(f"[DBG] page-cache pre-warm: {total_bytes / 1e9:.2f} GB in {elapsed:.1f}s "
          f"({'cold' if cold_read else 'warm'})", flush=True)
    if cold_read:
        # Brief pause: lets PSI avg10 decay below 60% before model loading begins.
        # 3 s at 0% PSI: avg10 decays from ~63% → ~46% (< 60%), resetting oomd timer.
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


def run_inference_inline(
    model_path: str,
    traces: list,
    remote_dir,
    nvme_dir,
    adapter_info: dict,
    *,
    min_hotness: float = 0.3,
    gpu_warmup_hotness: float = 0.9,
    max_adapters: int = 2,
    max_input_len: int = 256,
    max_output_cap: int = 64,
    skip_page_cache_warm: bool = False,
    return_stats: bool = False,
):
    """
    Core inference logic for faaslora_full.
    Can be called directly (no subprocess) or from main() via CLI.

    Args:
        model_path: path to the base language model
        traces: list of trace dicts [{prompt, adapter_id, max_tokens, temperature, top_p}, ...]
        remote_dir: Path or str, remote adapter storage directory
        nvme_dir: Path or str, NVMe adapter cache directory
        adapter_info: dict of {adapter_id: {hotness, size_mb, ...}}

    Returns:
        List of result dicts [{ttft_ms, tpot_ms, output_tokens, lora_io_ms}, ...]
    """
    # Step 1: Pre-read model files into OS page cache BEFORE importing torch/CUDA.
    # This separates disk I/O (high PSI) from CUDA init (low PSI), preventing
    # systemd-oomd from accumulating 30s of continuous PSI > 60%.
    total_t0 = time.perf_counter()
    if skip_page_cache_warm:
        print("[DBG] page-cache pre-warm: skipped (already warmed by caller)", flush=True)
    else:
        _warm_page_cache(model_path)

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    remote_dir = Path(remote_dir)
    nvme_dir = Path(nvme_dir)

    print("[DBG] tokenizer load...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[DBG] model load...", flush=True)
    base_model = _load_model_to_cuda(model_path, device="cuda:0")
    base_model.eval()
    print("[DBG] model ready on GPU", flush=True)

    nvme_dir.mkdir(parents=True, exist_ok=True)
    max_adapters = max(int(max_adapters), 1)
    max_input_len = max(int(max_input_len), 8)
    max_output_cap = max(int(max_output_cap), 1)
    print(
        f"[DBG] runtime cfg: min_hotness={min_hotness} "
        f"gpu_warmup_hotness={gpu_warmup_hotness} "
        f"max_adapters={max_adapters} max_input_len={max_input_len} "
        f"max_output_cap={max_output_cap}",
        flush=True,
    )

    def _valid_adapter(d: Path) -> bool:
        for name in ("adapter_model.safetensors", "adapter_model.bin"):
            p = d / name
            if p.exists() and p.stat().st_size > 10_000:
                return True
        return False

    nvme_cache = {}
    print(f"[DBG] preload adapters (hotness>={min_hotness})...", flush=True)
    for aid, info in adapter_info.items():
        if info.get("hotness", 0) >= min_hotness:
            src = remote_dir / aid
            dst = nvme_dir / aid
            if src.exists():
                if not dst.exists() or not _valid_adapter(dst):
                    if dst.exists():
                        import shutil as _shutil
                        _shutil.rmtree(dst)
                    ok, _ = copy_with_timing(src, dst)
                    if ok:
                        nvme_cache[aid] = str(dst)
                else:
                    nvme_cache[aid] = str(dst)
    print(f"[DBG] nvme_cache={list(nvme_cache.keys())}", flush=True)

    peft_model = None
    loaded_adapters = set()
    adapter_lru = []
    hot_for_gpu = [(aid, path) for aid, path in nvme_cache.items()
                   if adapter_info.get(aid, {}).get("hotness", 0) >= gpu_warmup_hotness]
    print(f"[DBG] hot_for_gpu={[a for a, _ in hot_for_gpu]}", flush=True)
    if hot_for_gpu:
        aid, path = hot_for_gpu[0]
        print(f"[DBG] GPU warmup: loading {aid}...", flush=True)
        peft_model = PeftModel.from_pretrained(base_model, path, adapter_name=aid, is_trainable=False)
        peft_model.set_adapter(aid)
        loaded_adapters.add(aid)
        adapter_lru.append(aid)
        print("[DBG] GPU warmup done", flush=True)

    print("[DBG] CUDA pre-warm...", flush=True)
    _model_for_warm = peft_model if peft_model is not None else base_model
    try:
        warm_inputs = tokenizer(
            "Warmup",
            return_tensors="pt",
            truncation=True,
            max_length=min(max_input_len, 16),
        )
        warm_input_ids = warm_inputs["input_ids"].to("cuda:0")
        warm_attention_mask = warm_inputs["attention_mask"].to("cuda:0")
        with torch.inference_mode():
            warm_out = _model_for_warm.generate(
                input_ids=warm_input_ids,
                attention_mask=warm_attention_mask,
                **_build_generate_kwargs(
                    _model_for_warm,
                    tokenizer,
                    1,
                    temperature=0.0,
                    top_p=1.0,
                ),
            )
        torch.cuda.synchronize()
        del warm_out, warm_input_ids, warm_attention_mask
        print("[DBG] CUDA pre-warm done", flush=True)
    except Exception as exc:
        print(f"[WARN] CUDA pre-warm skipped: {exc}", flush=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    max_tokens_default = 128
    results = []
    serving_t0 = time.perf_counter()
    print(f"[DBG] starting inference loop ({len(traces)} traces)", flush=True)

    for i, tr in enumerate(traces):
        prompt = (tr.get("prompt") or "")[:1500]
        adapter_id = tr.get("adapter_id")
        max_tokens = (
            min(int(tr.get("max_tokens", max_tokens_default)), max_output_cap)
            if max_output_cap > 0
            else int(tr.get("max_tokens", max_tokens_default))
        )
        temperature = float(tr.get("temperature", 0.7))
        top_p = float(tr.get("top_p", 0.9))

        local_path = nvme_cache.get(adapter_id) if adapter_id else None
        if adapter_id and not local_path:
            src = remote_dir / adapter_id
            dst = nvme_dir / adapter_id
            if src.exists():
                ok, _ = copy_with_timing(src, dst)
                if ok:
                    local_path = str(dst)
                    nvme_cache[adapter_id] = local_path

        model_to_use = base_model
        if local_path and adapter_id and peft_model is not None:
            if adapter_id in loaded_adapters:
                if adapter_id in adapter_lru:
                    adapter_lru.remove(adapter_id)
                adapter_lru.append(adapter_id)
            else:
                to_evict = []
                while len(loaded_adapters) >= max_adapters and adapter_lru:
                    to_evict.append(adapter_lru.pop(0))
                for eid in to_evict:
                    if eid in loaded_adapters:
                        peft_model.delete_adapter(eid)
                        loaded_adapters.discard(eid)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                if len(loaded_adapters) == 0:
                    clean_base = peft_model.unload()
                    peft_model = None
                    base_model = clean_base
                    peft_model = PeftModel.from_pretrained(
                        base_model, local_path, adapter_name=adapter_id, is_trainable=False
                    )
                    loaded_adapters = {adapter_id}
                    adapter_lru = [adapter_id]
                else:
                    peft_model.load_adapter(local_path, adapter_name=adapter_id)
                    loaded_adapters.add(adapter_id)
                    adapter_lru.append(adapter_id)
            peft_model.set_adapter(adapter_id)
            model_to_use = peft_model
        elif local_path and adapter_id and peft_model is None:
            peft_model = PeftModel.from_pretrained(base_model, local_path, adapter_name=adapter_id, is_trainable=False)
            peft_model.set_adapter(adapter_id)
            loaded_adapters.add(adapter_id)
            adapter_lru.append(adapter_id)
            model_to_use = peft_model

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_len)
        input_ids = inputs["input_ids"].to("cuda:0")
        attention_mask = inputs["attention_mask"].to("cuda:0")

        if i == 0:
            print(f"[DBG] first generate: model={type(model_to_use).__name__} "
                  f"input_len={input_ids.shape[1]} max_new={max_tokens}", flush=True)
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
        if i == 0:
            print(f"[DBG] first generate done in {(t1-t0)*1000:.0f}ms", flush=True)
        output_tokens = max(int(out.shape[-1] - input_ids.shape[-1]), 1)
        infer_ms = (t1 - t0) * 1000.0
        per_token_ms = infer_ms / output_tokens
        del out, input_ids, attention_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results.append({
            "ttft_ms": per_token_ms,
            "tpot_ms": per_token_ms,
            "output_tokens": output_tokens,
            "lora_io_ms": 0.0,
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
            "Usage: run_faaslora_subprocess.py <model> <traces_json> <remote_dir> "
            "<nvme_dir> <adapter_info_json> <output_json>",
            file=sys.stderr,
        )
        sys.exit(1)

    model_path = sys.argv[1]
    traces_path = sys.argv[2]
    remote_dir = sys.argv[3]
    nvme_dir = sys.argv[4]
    adapter_info_path = sys.argv[5]
    output_path = sys.argv[6]

    with open(traces_path) as f:
        traces = json.load(f)
    with open(adapter_info_path) as f:
        adapter_info = json.load(f)

    # Patch torch.load for legacy .bin files when running as subprocess
    import torch
    import torch.serialization as _ts
    _orig = getattr(_ts, "load", None)
    if callable(_orig):
        def _load(f, *a, **k):
            if k.get("weights_only", True) is True:
                k["weights_only"] = False
            return _orig(f, *a, **k)
        _ts.load = _load
        torch.load = _load

    results = run_inference_inline(model_path, traces, remote_dir, nvme_dir, adapter_info)

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
