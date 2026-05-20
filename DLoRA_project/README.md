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
true-remote trace and real adapter files. It has also passed 16-adapter,
64-adapter, and 128-adapter filtered Llama-3.2 3B replay gates without
rewriting dLoRA's core scheduling or migration logic, plus a real-weight
Llama-2 7B filtered replay gate at 2 adapters.

The first full Llama-3.2 3B formal replay completed `4000/4000` requests with
the 500-adapter true-remote workload, but it used upstream `migration_type=1`
(`dlora_dispatch_only`). Keep that result as closed appendix/ablation evidence
for dispatch-only behavior, not as the official dLoRA row.

The first official strategy gate using upstream `migration_type=3`
(`dlora_period_mig`) also completed on the same true-remote trace subset:
Llama-3.2 3B, 500 adapters, first 128 scheduled requests, `ok=128/128`,
`fail=0`, and no `trace_expected` token fallback. It is not an OOM. The
post-replay Raylet/AsyncEngineDeadError messages happen during runtime stop.
However, performance is weak under the current 2-GPU, `max_num_seqs=1`
envelope (`TTFT_e2e` avg 29.5s, p95 116.4s), so a fair main-table dLoRA
candidate still requires configuration gates and a full upstream
`migration_type=3` replay, plus the Llama-2 7B full replay.

Tracked evidence:

- gate summary: `evidence/gate_2026-05-19.json`
- local compatibility patch: `patches/modern_ray_import_compat.patch`
- real-adapter smoke summary: `evidence/real_adapt_2026-05-20.json`
- formal 500-adapter preflight: `evidence/formal_preflight_2026-05-20.json`
- formal dispatch-only 3B replay:
  `evidence/formal_dispatch_only_3b_2026-05-20.json`
- official period-migration 3B 128-request gate:
  `evidence/formal_period_mig_gate128_3b_2026-05-21.json`
- real-adapter compatibility patch:
  `patches/real_peft_llama32_e2e_compat_20260520.patch`
- formal 500-adapter runtime compatibility patch:
  `patches/formal_500_adapter_runtime_compat_20260520.patch`
- formal replay wrapper:
  `scripts/run_dlora_remote_formal.sh`
