# Baseline Reproduction Matrix

## 目标

为与 `serverless_llm_experiment` 做论文对比，优先准备 3 个官方系统的隔离式复现：

1. `ServerlessLLM`
2. `S-LoRA`
3. `SkyServe`

## 当前状态

| System | Repo | Local Path | Current Commit | Status | Isolation Strategy |
|---|---|---|---|---|
| ServerlessLLM | `ServerlessLLM/ServerlessLLM` | `/home/qhq/serverless_llm_baselines/repos/ServerlessLLM` | `9f50241` | cloned | 独立环境 + 不接入主项目代码 |
| S-LoRA | `S-LoRA/S-LoRA` | `/home/qhq/serverless_llm_baselines/repos/S-LoRA` | `c1ddf48` | cloned | 独立环境 + 老版本依赖隔离 |
| SkyServe | `skypilot-org/skypilot` | `/home/qhq/serverless_llm_baselines/repos/skypilot` | `ce5970ae4` | cloned | 独立环境 + 单独 CLI/控制平面 |

## 推荐环境名

- `sllm_official`
- `slora_official`
- `skyserve_official`

## 共享与隔离边界

允许共享：

- prompt / trace / 结果分析脚本
- 统一论文口径中的 workload 描述

禁止共享：

- serving runtime
- CUDA / PyTorch 版本
- 官方系统源码目录
- 主项目 conda 环境

## 与主项目的关系

主项目路径：

- `/home/qhq/serverless_llm_experiment`

baseline 官方复现路径：

- `/home/qhq/serverless_llm_baselines`

两者现在已经物理隔离。
