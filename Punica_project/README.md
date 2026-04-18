# Punica 基线项目

本目录用于在 `/home/qhq/serverless_llm_baselines` 工作区内，以隔离方式复现
`Punica`，并将其接入当前 many-LoRA 公平对比实验链。

## 目标

1. 不修改 `/home/qhq/serverless_llm_experiment` 主项目代码与环境。
2. 不污染 `/home/qhq/serverless_llm_experiment_retry14_baseline` 的正式实验环境。
3. 在独立项目目录中完成：
   - 官方仓库复现
   - 独立环境准备
   - shared trace / shared LoRA subset 接入
   - 指标口径对齐

## 目录说明

- [repo](/home/qhq/serverless_llm_baselines/Punica_project/repo)：官方仓库映射目录
- [docs](/home/qhq/serverless_llm_baselines/Punica_project/docs)：中文复现说明与实验记录
- [scripts](/home/qhq/serverless_llm_baselines/Punica_project/scripts)：统一脚本目录
- [results](/home/qhq/serverless_llm_baselines/Punica_project/results)：结果输出目录
- [logs](/home/qhq/serverless_llm_baselines/Punica_project/logs)：运行日志目录
- [environments](/home/qhq/serverless_llm_baselines/Punica_project/environments)：环境说明目录

## 当前状态

- `Punica_project` 项目目录已建立
- 已完成：
  - 官方仓库 clone：`/home/qhq/serverless_llm_baselines/repos/Punica`
  - 项目目录映射：`repo/scripts/results/logs/environments/models`
  - 初步 fair-run 适配链路：`scripts/run_punica_fair_experiment.sh`
- 当前定位：
  - `Punica` 只作为次要备选 baseline。
  - 原论文和开源实现主要围绕 `Llama-2 7B` 范围验证。
  - 若后续强行覆盖更多 backbone，必须明确为工程适配，不得声称是官方原生能力。

## 复现边界

当前 Punica 入口通过项目目录下的 symlink 访问共享脚本：

- `scripts/run_punica_fair_experiment.sh`
- `scripts/materialize_punica_loras.py`
- `scripts/replay_punica_trace.py`
- `scripts/summarize_punica_replay.py`

后续如果继续推进 Punica，仍必须遵守：

- 不改 FaaSLoRA 主系统和运行环境；
- 不改变官方 Punica 的核心 batching/LoRA serving 语义；
- 只在输入工件、adapter materialization、replay 和 metric summary 层做公平对齐；
- 不能用 Punica 单模型限制污染四 backbone 主实验结论。
