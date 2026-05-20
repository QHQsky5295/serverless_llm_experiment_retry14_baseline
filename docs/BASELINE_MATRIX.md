# Baseline Reproduction Matrix

This file is the current status matrix for systems used in the PrimeLoRA paper
comparison harness. Older three-system plans have been removed to avoid
confusing the formal experiment path.

## Current Formal Baselines

| System | Role in Paper | Local Entry | Upstream Handling | Current Status |
|---|---|---|---|---|
| `SGLang` | Serverful many-LoRA serving engine | `/home/qhq/serverless_llm_baselines/SGLang_project` | Upstream source under `repos/SGLang`; no core algorithm changes | Active main baseline; `e2e_v3`, lifecycle cost, shared trace/subset harness connected |
| `vLLM` | Standalone general serving baseline | `/home/qhq/serverless_llm_baselines/vLLM_project` | Installed vLLM runtime; no source modification | Active main baseline; standalone OpenAI server harness connected |
| `ServerlessLLM` | General serverless LLM baseline | `/home/qhq/serverless_llm_baselines/ServerlessLLM_project` | Upstream source under `repos/ServerlessLLM`; local reproducibility patch recorded in `patches/` | Active main baseline; vLLM backend, shared LoRA, metrics, runtime env alignment connected |
| `S-LoRA` | Serverful multi-LoRA paper baseline | `/home/qhq/serverless_llm_baselines/S-LoRA_project` | Upstream source under `repos/S-LoRA`; wrapper-level adaptation only | Active baseline after CUDA 11.8 environment and native prompt-budget guard |
| `Punica` | Scoped auxiliary many-LoRA baseline | `/home/qhq/serverless_llm_baselines/Punica_project` | Upstream source under `repos/Punica`; wrapper-level adaptation only | Limited Llama-2 7B auxiliary result only, not full main-table coverage |

## New Formal Candidates

| System | Role | Local Entry | Upstream Handling | Current Status |
|---|---|---|---|---|
| `ServerlessLLM-new` | Current upstream ServerlessLLM serverless LLM baseline | `/home/qhq/serverless_llm_baselines/ServerlessLLM_new_project` | Upstream source kept in ignored `vendor_new_baselines/`; no old ServerlessLLM code or data overwritten | True-remote 7B/3B LoRA workload closed and validated as a candidate row; not yet merged into default paper tables/figures |
| `dLoRA` | Multi-LoRA orchestration candidate | `/home/qhq/serverless_llm_baselines/DLoRA_project` | Upstream source kept in ignored `vendor_new_baselines/`; compatibility patches tracked under `DLoRA_project/patches/` without replacing scheduling/migration | 2026-05-20 real-adapter gates pass for Llama-3.2 3B through 128 adapters / 512 filtered requests and for Llama-2 7B at 2 adapters / 16 requests. A full Llama-3.2 3B 4000-request / 500-adapter true-remote replay also passes as `migration_type=1` / `dlora_dispatch_only` (`ok=4000/4000`, CE 0.2465), but this is appendix/ablation evidence. The first official `migration_type=3` / `dlora_period_mig` 3B gate passes for 128/128 requests with 500 adapters and no token fallback, but the 2-GPU `max_num_seqs=1` envelope is slow (`TTFT_e2e` avg 29.5s, p95 116.4s). Tune wrapper/runtime parameters before full official 3B and 7B replays. |
| `Loquetier` | Virtualized multi-LoRA serving candidate | `/home/qhq/serverless_llm_baselines/Loquetier_project` | Upstream source kept in ignored `vendor_new_baselines/`; compatibility patch tracks build/API/rank-pattern fixes without replacing SMLM kernels or mixed-LoRA forward path | 2026-05-20 real-adapter gates pass for Llama-3.2 3B through 256 adapters / 1024 filtered requests and for Llama-2 7B through 128 adapters / 256 filtered requests. The 3B/500-adapter preflight OOMs on single RTX 3090, so Loquetier is appendix scale-gate evidence, not a formal-table row. |
| `AIBrix` | Kubernetes-native vLLM control-plane candidate with LoRA lifecycle management | `/home/qhq/serverless_llm_baselines/AIBrix_project` | Upstream source kept in ignored `vendor_new_baselines/`; local work stays at build/deployment/runtime-sidecar compatibility and does not replace controller/gateway/CRD logic | 2026-05-20 controller-manager, gateway-plugins, Python runtime, and runtime-to-vLLM real LoRA component gate pass. Full formal true-remote replay is blocked because this machine has no user-accessible Kubernetes/Docker/GPU-pod runtime; appendix/gate evidence only. |
| `HydraServe` | Serverless LLM cold-start candidate | `/home/qhq/serverless_llm_baselines/HydraServe_project` | Upstream source kept in ignored `vendor_new_baselines/`; local work is limited to control-plane import, platform gate, and LoRA interface audit | 2026-05-20 control modules import and embedded vLLM 0.4.2 static LoRA argument parsing passes. Full formal replay is blocked by inaccessible Docker/Kubernetes runtime, and the current scheduler path does not preserve per-request adapter identity without semantic changes. Appendix/gate evidence only. |
| `Sarathi-Serve` | Chunked-prefill LLM scheduler candidate | `/home/qhq/serverless_llm_baselines/SarathiServe_project` | Upstream source kept in ignored `vendor_new_baselines/`; both OSDI artifact and main branches audited without adding LoRA features | 2026-05-20 package metadata resolves for OSDI/main branches, but OSDI has no LoRA/adapter/PEFT path and main has only an unused LoRA dataclass. Per-request LoRA workload support would require implementing new scheduler/model/API semantics, so appendix/gate evidence only. |

## Not Active In The Current Main Harness

| System | Reason |
|---|---|
| `SkyServe` | Useful related work for serving orchestration, but not currently reproduced as a same-trace many-LoRA baseline. |
| `Llumnix` | Not currently connected to the shared trace / shared adapter / `e2e_v3` harness. |
| `ServerlessLoRA` / `P-LoRA` | No complete local official-code reproduction path in the current workspace. Do not claim formal reproduction. |
| `Medusa` | 2026-05-19 gate closed as not formally reproducible on this machine: local build/import can be adapted, but required SPDK/NVMe/hugepage/GDRCopy runtime stack is absent. |
| `FaaScale` / `LambdaScale` | 2026-05-19 gate closed as appendix/gate evidence only: local import/IPC/RDMA binding can be adapted, but runtime finds zero usable IB devices and there is no ready Llama-3.2/LoRA workload path. |

## Formal Main Round

The active formal Llama-2 7B round is:

```text
llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1
```

Expected systems in the full round:

```text
sglang -> serverlessllm -> vllm -> slora -> faaslora -> compare
```

The round must be run through:

```text
/home/qhq/serverless_llm_baselines/scripts/run_full_fair_round.sh
```

Interrupted rounds should be resumed through:

```text
/home/qhq/serverless_llm_baselines/scripts/resume_fair_round_tmux.sh
```

## Inclusion Rule

A system can enter the formal main table only if it satisfies all of the
following:

1. Uses the exact shared trace artifact.
2. Uses the exact shared adapter subset artifact.
3. Runs under the same GPU budget envelope.
4. Emits `metric_schema_version=e2e_v3`.
5. Emits the shared headline fields used by the comparison table.
6. Fails fast instead of producing all-zero, partially missing, or mixed-schema
   results.
