# ServerlessLLM Environment

ServerlessLLM is the active general serverless baseline. It runs in isolated
head/worker environments and is launched through the shared baseline harness.

## Current Environments

- Head env: `sllm_head_official`
- Worker env: `sllm_worker_official`
- vLLM runtime env preference:
  - `sllm_vllm0102_official` when available
  - fallback source env: `LLM_vllm0102`
- Upstream repo: `/home/qhq/serverless_llm_baselines/repos/ServerlessLLM`
- Project entry: `/home/qhq/serverless_llm_baselines/ServerlessLLM_project`

## Current Runtime Rules

1. Prefer the ServerlessLLM vLLM backend for formal many-LoRA comparison.
2. Generate deploy config from the authoritative FaaSLoRA experiment profile.
3. Use the exact shared trace and shared adapter subset.
4. `enforce_eager: auto` must resolve to CUDA graph for stable single-GPU
   Llama-2 scale-out LoRA paths, while preserving conservative eager mode for
   TP or known-risk paths.
5. Runtime env such as `VLLM_USE_V1`, `VLLM_ATTENTION_BACKEND`, and
   `VLLM_USE_FLASHINFER_SAMPLER` must be written into deploy config and passed
   into tmux-launched head/worker/serve processes.
6. Do not silently fallback to transformers. If vLLM backend fails correctness
   probing, record the root cause and stop the formal run.

## Maintenance

Local source changes required for reproduction are recorded in:

```text
/home/qhq/serverless_llm_baselines/patches/ServerlessLLM_local_changes.patch
```

The stack sync script copies the patched runtime files into the active head and
worker site-packages before launch.
