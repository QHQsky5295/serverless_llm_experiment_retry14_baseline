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
- 后续将继续完成：
  - 官方仓库 clone
  - 上游版本记录
  - 隔离环境说明
  - fair-run 适配链路
