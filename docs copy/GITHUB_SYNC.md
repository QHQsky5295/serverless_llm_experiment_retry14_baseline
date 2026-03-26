# GitHub 同步说明（clean-tree 回退快照）

本文件说明当前 `serverless_llm_experiment_retry14_baseline` 如何同步到 GitHub，以及哪些内容应当、哪些内容不应当进入回退快照。

## 当前远端与分支

- 远端仓库：`https://github.com/QHQsky5295/FaaSLoRA.git`
- 当前 clean-tree 分支：`retry14_rebuild`

当前建议：

- 先把当前 clean-tree 推到当前分支，作为可回退快照
- 是否合并到 `main` 另行决策

## 本次同步应提交的内容

应提交：

- `faaslora/`
- `scripts/`
- `configs/`
- `docs/`
- `tests/`
- 根目录文档：
  - `README.md`
  - `EXPERIMENT_GUIDE.md`
  - `PROJECT_STRUCTURE.md`
- 顶层文档镜像：
  - `docs copy/*.md`
- 当前刻意 curated 的 manifest：
  - `configs/generated/lora_manifest_1000.json`

说明：

- `configs/generated/lora_manifest_1000.json` 在大多数情况下属于 generated file。
- 但对当前 Qwen 7B V2 publicmix clean-tree rollback 快照，它是有意保留的基线文件，应随本次快照一起提交。

## 本次同步不应提交的内容

不提交：

- `results/`
- `artifacts/`
- `data/`
- `models/` 下的权重文件
- `/tmp` 下的 launch log
- 本机缓存、虚拟环境和用户级 shell 定制

## 文档同步规则

当前仓库同时维护两套同内容文档：

1. `docs/`：仓库主文档
2. `docs copy/`：IDE 常用镜像文档

同步 GitHub 前，应保证：

- `docs/` 与顶层 `docs copy/` 的同名文件内容一致
- 不提交误生成的嵌套镜像目录，如：
  - `docs copy/docs/`
  - `docs copy/docs copy/`

## 推荐同步流程

1. 先确认当前实验与代码状态已经分析完，知道本次快照代表什么。
2. 更新：
   - `README.md`
   - `EXPERIMENT_GUIDE.md`
   - `PROJECT_STRUCTURE.md`
   - `docs/*.md`
   - `docs copy/*.md`
3. 只提交 source/config/docs/tests 和当前刻意 curated 的 manifest。
4. 在当前工作分支形成一次带说明的提交。
5. 推送到 GitHub，作为后续可回退基线。

## 本次快照的目的

本次 GitHub 同步的目的不是：

- 声称系统已经完全收口
- 产出最终论文版本

而是：

- 固定当前 clean-tree 状态
- 保留最近这批有效修复
- 方便后续出现回归时直接回退

如果以后需要共享实验结果，应单独导出、单独整理，而不是直接提交本地运行输出目录。
