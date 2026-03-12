"""Minimal vLLM + LoRA test using synchronous LLM API."""
import os

MODEL = "/home/qhq/serverless_llm_experiment/models/Qwen--Qwen2.5-7B-Instruct"
LORA_PATH = "/home/qhq/serverless_llm_experiment/artifacts/nvme_cache/faaslora_full/finance_lora"

def main():
    print("=" * 60)
    print("Minimal vLLM + LoRA crash test (synchronous API)")
    print("=" * 60, flush=True)

    # Step 1: Import
    print("\n[1] Importing vLLM ...", flush=True)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    print("    OK", flush=True)

    # Step 2: Create engine
    print("\n[2] Creating LLM engine ...", flush=True)
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        tensor_parallel_size=1,
        enforce_eager=True,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=64,
    )
    print("    OK: engine created", flush=True)

    # Step 3: generation WITHOUT LoRA
    print("\n[3] Test generation WITHOUT LoRA ...", flush=True)
    sp = SamplingParams(max_tokens=20, temperature=0.7)
    outputs = llm.generate(["Hello, how are you?"], sp)
    text = outputs[0].outputs[0].text
    print(f"    OK: got {len(text)} chars: {text[:50]!r}", flush=True)

    # Step 4: generation WITH LoRA
    print("\n[4] Test generation WITH LoRA ...", flush=True)
    lora_req = LoRARequest("finance", 1, LORA_PATH)
    outputs = llm.generate(["What is compound interest?"], sp, lora_request=lora_req)
    text = outputs[0].outputs[0].text
    print(f"    OK: got {len(text)} chars: {text[:50]!r}", flush=True)

    # Step 5: generation with DIFFERENT LoRA
    print("\n[5] Test generation with second LoRA ...", flush=True)
    lora2_path = LORA_PATH.replace("finance_lora", "medical_lora")
    if os.path.isdir(lora2_path):
        lora_req2 = LoRARequest("medical", 2, lora2_path)
        outputs = llm.generate(["What are the symptoms of flu?"], sp, lora_request=lora_req2)
        text = outputs[0].outputs[0].text
        print(f"    OK: got {len(text)} chars: {text[:50]!r}", flush=True)
    else:
        print(f"    SKIP: {lora2_path} not found", flush=True)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    main()
