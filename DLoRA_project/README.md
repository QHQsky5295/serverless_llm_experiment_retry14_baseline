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

The official strategy gates using upstream `migration_type=3`
(`dlora_period_mig`) also complete on the same true-remote trace subset:
Llama-3.2 3B, 500 adapters, first 128 scheduled requests, `ok=128/128`,
`fail=0`, and no `trace_expected` token fallback. They are not OOMs. The
post-replay Raylet/AsyncEngineDeadError messages happen during runtime stop.
The first 2-GPU `max_num_seqs=1` envelope was weak (`TTFT_e2e` avg 29.5s,
p95 116.4s). `max_num_seqs=2` improved the tail (`TTFT_e2e` avg 24.7s,
p95 59.1s, 81.9 tok/s), and `max_num_seqs=4` is currently the best 2-GPU
envelope (`TTFT_e2e` avg 14.5s, p95 26.5s, 92.1 tok/s). The remaining
latency is still service-side engine wait, so a fair main-table dLoRA
candidate still requires a stable 4-GPU topology gate and a full upstream
`migration_type=3` replay, plus the Llama-2 7B full replay. The first 4-GPU
startup gate did not reach replay: Ray killed a worker at 124.16/125.38GB host
memory while using `num_groups=4`, `max_num_seqs=4`, and the wrapper default
`swap_space_gb=8`. This is a host-memory envelope failure, not a CUDA OOM or a
remote-materialization failure. The fair next attempt is the same 4-GPU gate
with a lower dLoRA/vLLM CPU KV swap envelope before considering more invasive
runtime changes. The `swap_space_gb=2` attempt also failed before replay after
heavy host-memory/swap pressure; Ray's default object store was about 38.6GB.
The bounded-object-store rerun connected to the pre-started Ray cluster, so the
object-store control itself was active, but the DP4/TP1 topology still failed
before HTTP readiness because four dLoRA/vLLM engines duplicated startup state
and Ray killed a worker under host-memory pressure. The wrapper-only DP2/TP2
four-GPU topology gate then reached HTTP readiness and completed `128/128`
requests without token fallback or OOM, but it was slower and more expensive
than the best 2-GPU `max_num_seqs=4` gate (`TTFT_e2e` avg 18.6s vs 14.5s,
p95 33.1s vs 26.5s, CE 1.97 vs 5.34). The fair full 3B formal replay should
therefore use the best measured wrapper-only dLoRA envelope: DP2/TP1,
`max_num_seqs=4`, upstream `migration_type=3`, 500 real remote adapters, and
no core dLoRA code changes. That full Llama-3.2 3B replay is now closed:
`ok=4000/4000`, `fail=0`, no token fallback, `TTFT_e2e` avg 11.16s, p95
27.28s, 115.67 tok/s, CE 45.52. This is the selected dLoRA 3B formal
candidate. The full dLoRA paper row still requires the matching Llama-2 7B
true-remote full replay before table/figure inclusion.

Tracked evidence:

- gate summary: `evidence/gate_2026-05-19.json`
- local compatibility patch: `patches/modern_ray_import_compat.patch`
- real-adapter smoke summary: `evidence/real_adapt_2026-05-20.json`
- formal 500-adapter preflight: `evidence/formal_preflight_2026-05-20.json`
- formal dispatch-only 3B replay:
  `evidence/formal_dispatch_only_3b_2026-05-20.json`
- official period-migration 3B 128-request gate:
  `evidence/formal_period_mig_gate128_3b_2026-05-21.json`
- official period-migration 3B 128-request `max_num_seqs=2` gate:
  `evidence/formal_period_mig_gate128_s2_3b_2026-05-21.json`
- official period-migration 3B 128-request `max_num_seqs=4` gate:
  `evidence/formal_period_mig_gate128_s4_3b_2026-05-21.json`
- official period-migration 3B 4-GPU startup memory gate:
  `evidence/formal_period_mig_gate128_g4_s4_hostoom_3b_2026-05-21.json`
- official period-migration 3B 4-GPU `swap_space_gb=2` startup memory gate:
  `evidence/formal_period_mig_gate128_g4_s4_swap2_hostoom_3b_2026-05-21.json`
- official period-migration 3B 4-GPU bounded-Ray object-store startup memory
  gate:
  `evidence/formal_period_mig_gate128_g4_s4_swap2_obj8_hostoom_3b_2026-05-21.json`
- official period-migration 3B 4-GPU DP2/TP2 topology gate:
  `evidence/formal_period_mig_gate128_g2tp2_g4_s4_3b_2026-05-21.json`
- official period-migration 3B full 4000-request replay:
  `evidence/formal_period_mig_full4000_3b_2026-05-21.json`
- real-adapter compatibility patch:
  `patches/real_peft_llama32_e2e_compat_20260520.patch`
- formal 500-adapter runtime compatibility patch:
  `patches/formal_500_adapter_runtime_compat_20260520.patch`
- formal replay wrapper:
  `scripts/run_dlora_remote_formal.sh`
