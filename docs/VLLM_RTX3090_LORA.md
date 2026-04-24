# vLLM + LoRA Notes On RTX 3090

This file records current runtime lessons that still matter. Older
model-specific rollback notes were removed.

## Current Role

vLLM appears in two places:

1. FaaSLoRA uses a vLLM-based serving path internally.
2. The baseline workspace runs standalone vLLM as a separate paper baseline.

The standalone vLLM baseline is documented under:

```text
/home/qhq/serverless_llm_baselines/vLLM_project
```

## Current Stable Rules

- Llama-2 7B single-GPU LoRA replica paths can use CUDA graph when
  `enforce_eager: auto` resolves to `false`.
- TP or known-risk model paths may still require conservative eager execution.
- `VLLM_USE_V1`, `VLLM_ATTENTION_BACKEND`, and
  `VLLM_USE_FLASHINFER_SAMPLER` must be explicit in formal harnesses.
- Token accounting must come from actual response usage or local tokenizer
  counting of the generated text. It must not fall back to raw trace expected
  tokens in formal results.

## Current Known Risk

If TPOT suddenly returns to the 60ms range for Llama-2 7B on this host, first
check whether a path accidentally enabled eager mode or failed to propagate the
vLLM runtime env into tmux-launched processes.
