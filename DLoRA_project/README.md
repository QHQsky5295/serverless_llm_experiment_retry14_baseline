# dLoRA Project

This project entry tracks the PrimeLoRA/FaaSLoRA reproduction gate for dLoRA.

- Upstream: `https://github.com/LLMServe/dLoRA-artifact`
- Local source: `../vendor_new_baselines/dLoRA_artifact_main_20260519`
- Upstream commit under test: `75f1c439446fe194b1df8a24982ef9067841fab5`
- Gate script: `../scripts/run_dlora_gate.sh`
- Gate evidence: `../results/paper_experiments/15_new_serverless_baselines_remote_v1/gates/dlora/`

The reproduction target is the already closed true-remote PrimeLoRA workload:
Llama-2 7B and Llama-3.2 3B, 4000 requests, 500 adapters, shared trace,
shared adapter subset, remote artifacts, and `e2e_v3` metrics.

Current result: dLoRA is highly relevant as a LoRA orchestration system. The
first 2026-05-19 gate only proved local build/import. The 2026-05-20 local
adaptation adds a narrow real-PEFT adapter loader and replay compatibility
layer, and now passes a real-weight Llama-3.2 3B smoke gate using the closed
true-remote trace and real adapter files. It has also passed 16-adapter and
64-adapter filtered Llama-3.2 3B replay gates without rewriting dLoRA's core
scheduling or migration logic, plus a real-weight Llama-2 7B filtered replay
gate at 2 adapters. It still cannot enter the formal comparison table until the
same path is scaled to the full 4000-request, 500-adapter 3B and 7B runs.

Tracked evidence:

- gate summary: `evidence/gate_2026-05-19.json`
- local compatibility patch: `patches/modern_ray_import_compat.patch`
- real-adapter smoke summary: `evidence/real_adapt_2026-05-20.json`
- real-adapter compatibility patch:
  `patches/real_peft_llama32_e2e_compat_20260520.patch`
