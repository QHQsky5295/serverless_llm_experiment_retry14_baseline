# GitHub 同步说明（clean-tree 回退快照）

本文件说明当前 `serverless_llm_experiment_retry14_baseline` 如何同步到 GitHub，以及哪些内容应当、哪些内容不应当进入回退快照。

## 当前远端与分支

- 远端仓库：`https://github.com/QHQsky5295/FaaSLoRA.git`
- 当前 clean-tree 分支：`retry14_rebuild`
- 当前已推送基线提交：`9147eb0`

当前建议：

- 先把当前 clean-tree 推到当前分支，作为可回退快照
- 是否合并到 `main` 另行决策

## 2026-03-27 晚更新：本次同步的目标

本次同步不是为了宣布 7B 性能完全收口，而是为了固定下面这批**已验证结构修复 + 已完成但待实验验证的新观测/路由修复**：

1. `retry30_baseline` 已验证：
   - `GPU0 resident≈0` 的主异常消失
   - `scale-up warmup` 真实生效
   - `Cold_start_latency` 变成可信真值
2. 当前 clean-tree 新增：
   - `Runtime_TTFT = vllm_ttft_ms` 接入 live / summary / JSON
   - runtime-aware routing 的最小修复，用实例最近真实 runtime 代价辅助 backbone / 浅路由
3. 当前还未正式跑下一轮验证，因此这次快照的价值是：
   - 锁定当前正确主线
   - 方便下一轮 `retry31` 前后快速回退

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

本次尤其应确认同步：

- `faaslora/experiment/instance_pool.py`
- `faaslora/experiment/experiment_stack.py`
- `faaslora/scheduling/resource_coordinator.py`
- `scripts/run_all_experiments.py`
- `tests/test_basic_smoke.py`
- `docs/*.md`
- `docs copy/*.md`

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
- 保留当前交互逻辑、第一性原则和 handoff 入口
- 方便后续出现回归时直接回退

## 当前同步后默认判断

同步完成后，后续主线应继续坚持：

1. 不偏离主线，不把系统改坏
2. 所有修改先服从论文两条第一性原则
3. 下一轮优先跑 `retry31_baseline`，验证 `Runtime_TTFT + runtime-aware routing` 是否真的改善 headline TTFT

如果以后需要共享实验结果，应单独导出、单独整理，而不是直接提交本地运行输出目录。
