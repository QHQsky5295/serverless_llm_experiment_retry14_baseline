# ServerlessLLM Reproduction Scope

## 复现目标

在不影响主项目 `serverless_llm_experiment` 的前提下，尽可能贴近
`ServerlessLLM` 论文与官方实现的核心机制完成独立复现。

## 不追求的内容

- 不在当前阶段追求完整多节点/完整 Docker 集群一比一复现
- 不把 ServerlessLLM 代码并入主项目
- 不把 ServerlessLLM 依赖安装进主项目环境

## 当前阶段的复现范围

优先复现下面三部分：

1. 官方仓库与依赖环境隔离准备
2. 本机单节点路径的最小可运行 serving / loading 入口
3. 论文中最核心的 checkpoint loading / multi-model sharing 思想验证

## 当前机器上的建议路径

### P0

- 隔离环境安装 `ServerlessLLM`
- 跑通官方 README 中的最小启动路径
- 确认 CLI / API / store 子系统能正常运行

### P1

- 选择一个当前机器可承受的小模型做本机单节点验证
- 记录启动时间、加载路径和基本吞吐

### P2

- 再决定是否进一步接入你的统一 trace / prompt / 结果分析脚本

## 与主项目的边界

- 主项目用于“统一框架下的机制级对照”
- ServerlessLLM 官方复现用于“官方系统独立 sanity check / artifact-level 支撑”

两者不是二选一，而是互补。
