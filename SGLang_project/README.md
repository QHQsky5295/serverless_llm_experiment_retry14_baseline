# SGLang 基线项目

本目录用于在 `/home/qhq/serverless_llm_baselines` 工作区内，以隔离方式复现
`SGLang`，并将其接入当前 many-LoRA 公平对比实验链。

## 目标

1. 不修改 `/home/qhq/serverless_llm_experiment` 主项目代码与环境。
2. 不污染 `/home/qhq/serverless_llm_experiment_retry14_baseline` 的正式实验环境。
3. 在独立项目目录中完成：
   - 官方仓库复现
   - 独立环境准备
   - shared trace / shared LoRA subset 接入
   - 指标口径对齐

## 目录说明

- [docs](/home/qhq/serverless_llm_baselines/SGLang_project/docs)：中文复现说明与实验记录
- [scripts](/home/qhq/serverless_llm_baselines/SGLang_project/scripts)：统一脚本目录
- [configs](/home/qhq/serverless_llm_baselines/SGLang_project/configs)：基线配置目录

## 当前状态

- `SGLang_project` 项目目录已建立
- 已完成：
  - 上游仓库拉取
  - 隔离虚拟环境 `/home/qhq/.venvs/sglang_py310`
  - official multi-LoRA 能力核查
  - many-LoRA 公平 replay 脚本接入
  - `Llama-2 7B + sanitized shared subset` 最小真实 GPU smoke（`4/4` 成功）
- 后续正式对比将直接复用：
  - `/home/qhq/serverless_llm_baselines/scripts/prepare_sanitized_shared_round.sh`
  - `/home/qhq/serverless_llm_baselines/scripts/run_sglang_fair_experiment.sh`
