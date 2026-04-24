# FaaSLoRA Environment

This file records the current environment assumptions for the FaaSLoRA system.
Old Qwen-only rollback notes have been removed to avoid misleading future runs.

## Hardware

- Host: single-node local server.
- GPUs: 4 x RTX 3090 24GB.
- Formal fair comparison assumes all systems run sequentially under the same
  four-GPU budget.

## Main Runtime

- Conda env: `LLM_vllm0102`
- vLLM family: vLLM 0.10.x path used by the current FaaSLoRA runner.
- Long experiments should be launched from stable tmux/user sessions; avoid
  starting long runs inside an SSH/systemd session that is already closing.

## Baseline Runtime Boundary

Baseline systems use isolated environments under `/home/qhq/serverless_llm_baselines`.
Do not install or upgrade their dependencies inside `LLM_vllm0102` unless the
change is explicitly part of the FaaSLoRA runtime.

Important baseline envs:

- `sllm_head_official`
- `sllm_worker_official`
- `sllm_vllm0102_official` or `LLM_vllm0102` for ServerlessLLM-vLLM runtime
- `/home/qhq/.venvs/sglang_py310`
- `slora_official_cu118`

## Current Formal Workload

```text
Llama-2 7B / 4000 requests / 500 adapters /
Zipf exponent 1.0 / hot set cap 48 / hotset rotation 500 / time_scale 8
```

500-request rounds are debug/bring-up only.

## Current Metric And Cost Assumptions

- Metric schema: `e2e_v3`.
- Main cost metric: `Cost/req`.
- Main CE: `1 / (avg_E2E_e2e_seconds * Cost/req)`.
- Serverless monetary model uses active/idle differential billing.
- Current idle GPU cost factor: `0.2380952381`.
