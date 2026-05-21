# Documentation Index

This index lists the active baseline-harness documents after the April 24
cleanup. If a document is not listed here, do not treat it as a current
execution source.

## Canonical Documents

- `FAIR_COMPARISON_EXECUTION_PLAN.md`: current cross-system execution rules.
- `CURRENT_QUEUE_HANDOFF_2026-04-27.md`: current tmux queue status, monitor
  command, and resume command for the active operating-load sensitivity run.
- `SYSTEM_REPRODUCTION_RULES.md`: rules for adding or maintaining baselines.
- `BASELINE_MATRIX.md`: 当前基线状态和纳入边界。
- `UPSTREAM_REPO_STATE.md`: upstream commit and patch tracking.
- `CODEX_INTERACTION_RULES.md`: collaboration and root-cause analysis rules.
- `NEW_SERVERLESS_BASELINES_2026-05-18.md`: ServerlessLLM-new、Medusa、
  FaaScale 按顺序复现的 campaign 记录。
- `BASELINE_ADAPTATION_LIMITS_AND_NEW_SURVEY_2026-05-21.md`: 中文版基线适配
  边界和新增候选调查，区分严格论文系统、工程系统基线、adapter-serving
  基线和只能进入附录/相关工作的系统。
- `SERVERLESS_LLM_PAPER_BASELINE_REPRODUCIBILITY_2020_2026.md`: 中文版
  2020-2026 无服务器大模型推理论文系统调查，列出论文/代码网址，并说明每个
  候选为什么能或不能在当前环境下成为正式 3B+7B true-remote
  PrimeLoRA/FaaSLoRA 基线。
- `SERVERLESSLLM_NEW_OPTIMIZATION_ANALYSIS_2026-05-21.md`: 中文版
  ServerlessLLM-new true-remote LoRA 负载性能诊断和不改核心代码的
  warm-min4 优化验证记录；包含 3B/7B 正式 4000 请求结果、7B 外部
  GPU 显存占用说明、公平性裁决和其他 serverless 语义内优化方向检查。

## Current True-Remote Result Scope

`FAIR_COMPARISON_EXECUTION_PLAN.md` and `SYSTEM_REPRODUCTION_RULES.md` now include
the completed 2026-05-14 true-remote remote-fair run:

- section: `12_remote_fair_main_real_remote_v1`
- models: Llama-2 7B, Llama-2 13B, Llama-3.2 3B
- systems: SGLang, ServerlessLLM, vLLM, S-LoRA
- comparison outputs:
  `results/paper_experiments/12_remote_fair_main_real_remote_v1/_comparisons/`

The FaaSLoRA repository owns the merged paper tables, figures, PrimeLoRA source
summaries, and curated snapshot for this true-remote validation.

## New Serverless Baseline Candidate Scope

The 2026-05-18 ServerlessLLM-new true-remote candidate is isolated from the old
ServerlessLLM baseline:

- project entry: `../ServerlessLLM_new_project/`
- upstream clone: `../vendor_new_baselines/ServerlessLLM_new_main_20260518/`
  at commit `9f50241baa5386e06a9321c51f19a9ef5f964c2b`
- result section: `results/paper_experiments/15_new_serverless_baselines_remote_v1/`
- status: closed and validated for 7B/3B, pending paper-table integration
  policy.

The old `ServerlessLLM_project/` and all default paper data remain unchanged.

Medusa was also gated in this campaign:

- project entry: `../Medusa_project/`
- upstream clone: `../vendor_new_baselines/Medusa_main_20260518/` at commit
  `6581d2e5ec8fa4ecdabcdb50560982a78ea3ca89`
- status: local adaptation build/import gate closed; `vllm._C` builds/imports
  after `patches/Medusa_localadapt_20260519.patch`, but no formal true-remote
  LoRA replay was launched because this machine lacks Medusa's SPDK runtime
  prerequisites.

FaaScale/LambdaScale was gated after Medusa:

- project entry: `../FaaScale_project/`
- LambdaScale upstream clone:
  `../vendor_new_baselines/lambda-scale_main_20260518/` at commit
  `9db210fcb6979f7c1f73f9819a77e0edb6c5e343`
- RDMA-P2P upstream clone:
  `../vendor_new_baselines/rdma-p2p_main_20260518/` at commit
  `ed83237439d2103141fbc7c9b97f348055b6cb53`
- status: local import, IPC extension, and targeted RDMA-P2P Python binding
  build/import gate closed; no formal true-remote LoRA replay was launched
  because the machine exposes no usable InfiniBand device and the source lacks
  ready Llama-3.2 3B plus LoRA/PEFT workload support.

## Active Environment Notes

- `../environments/ServerlessLLM.md`
- `../environments/S-LoRA.md`
- `../environments/Punica.md`

## Removed As Obsolete

- `REPRO_PLAN.md`: superseded by the full fair-round harness and reproduction
  rules.
- `ServerlessLLM_REPRO_SCOPE.md`: superseded by `ServerlessLLM_project/README.md`
  and the current fair-comparison plan.
- `../environments/SkyServe.md`: SkyServe is not in the active main harness.
