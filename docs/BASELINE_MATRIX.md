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
| `dLoRA` | Multi-LoRA orchestration candidate | `/home/qhq/serverless_llm_baselines/DLoRA_project` | Upstream source kept in ignored `vendor_new_baselines/`; compatibility patches tracked under `DLoRA_project/patches/` without replacing scheduling/migration | Real-adapter gates pass for both closed backbones. Dispatch-only 3B full replay (`migration_type=1`) completed `4000/4000` but is appendix/ablation evidence. Official `migration_type=3` 3B gates selected DP2/TP1 `max_num_seqs=4`; the full 3B true-remote replay completed `4000/4000` with no token fallback, TTFT avg 11.16s, p95 27.28s, 115.67 tok/s, CE 45.52. The matching 7B formal path was driven through DP2/TP2 and G1/TP4 wrapper-only envelopes; the final `gpu_capacity=1`, `gpu_memory_utilization=0.99` gate still produced `# GPU blocks: 0` before HTTP readiness after all 500 remote adapters loaded. dLoRA is limited 3B/appendix evidence, not a full 3B+7B main row on this 4x3090 machine. |
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
| `Preble` | 2026-05-21 triage: open ICLR 2025 prefix-cache routing system, but not native LoRA/serverless adapter lifecycle. Possible future appendix gate only after adapter-identity audit. |
| `MuxServe` | 2026-05-21 triage: open multiple-full-LLM multiplexing system. It is not a LoRA adapter workload; treating adapters as full models would change the experiment. |
| `PowerInfer` | 2026-05-21 triage: open SOSP 2024 consumer-GPU hybrid sparse-inference engine, but not a serverless or dynamic LoRA workload fit for the closed 3B/7B PEFT traces. |
| `LoRAX` | 2026-05-21 triage: best remaining practical new gate candidate. It is open-source multi-LoRA serving with dynamic adapter loading and an OpenAI-compatible API, but it is not a serverless cold-start system. Gate only before any formal claim. |
| `TGI Multi-LoRA` | 2026-05-21 triage: possible secondary adapter-serving gate if LoRAX fails, but likely operationally heavier and not serverless. |
| `ServerlessLoRA` / `P-LoRA` / `Toppings` / `LoRAServe` / `InfiniLoRA` | 2026-05-21 triage: highly relevant papers, but no public official code located during this search. Related work only until code appears. |
| `llm-d` / `NVIDIA Dynamo` | 2026-05-21 triage: open platform stacks with LoRA-related routing/loading features, but Kubernetes/container/datacenter deployment requirements make them future appendix gates only on the current machine. |
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
