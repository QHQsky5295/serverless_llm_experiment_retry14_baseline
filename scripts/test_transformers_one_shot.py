#!/usr/bin/env python3
"""
Minimal test: load 3B and run ONE generate() in the main thread.
- If this completes: problem is likely in the experiment's asyncio/threading.
- If this gets killed: problem is model/GPU/driver.
Run: CUDA_VISIBLE_DEVICES=0 python scripts/test_transformers_one_shot.py
"""
import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

def main():
    model_path = os.path.join(REPO_ROOT, "models", "Qwen--Qwen2.5-3B-Instruct")
    if not os.path.isdir(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print("Loading tokenizer ...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (this may take ~10s) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()

    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda:0")

    print("Running one generate() in main thread ...")
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    n = out.shape[-1] - inputs["input_ids"].shape[-1]
    print(f"OK: generated {n} tokens. First inference succeeded.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
