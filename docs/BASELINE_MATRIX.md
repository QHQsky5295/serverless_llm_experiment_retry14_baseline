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

## Not Active In The Current Main Harness

| System | Reason |
|---|---|
| `SkyServe` | Useful related work for serving orchestration, but not currently reproduced as a same-trace many-LoRA baseline. |
| `Llumnix` | Not currently connected to the shared trace / shared adapter / `e2e_v3` harness. |
| `dLoRA` | Related to adapter orchestration, but currently left for related work / discussion unless a complete reproducible path is added. |
| `ServerlessLoRA` / `P-LoRA` | No complete local official-code reproduction path in the current workspace. Do not claim formal reproduction. |

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
