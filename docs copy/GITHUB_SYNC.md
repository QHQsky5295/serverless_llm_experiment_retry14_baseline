# GitHub 同步说明（clean-tree 回退快照）

本文件说明当前 `serverless_llm_experiment_retry14_baseline` 如何同步到 GitHub，以及哪些内容应当、哪些内容不应当进入回退快照。

## 当前远端与分支

- 远端仓库：`https://github.com/QHQsky5295/FaaSLoRA.git`
- 当前 clean-tree 分支：`retry14_rebuild`
- 上一已推送基线提交：`70c90fe`

当前建议：

- 先把当前 clean-tree 推到当前分支，作为可回退快照
- 是否合并到 `main` 另行决策

## 2026-03-29 晚更新：本次同步已完成

- 已推送提交：`70c90fe`
- 推送分支：`retry14_rebuild`
- 本次同步纳入：
  - TODO `#1` 的 live scale-up 主线修复
  - scale-up 事件口径修复与实时动态 RPS 阈值刷新
  - `retry40_baseline` 的正式分析结论与 handoff 文档更新
- 本次同步后，当前最新已验证结果固定为 `retry40_baseline @ 500`
- 本次同步后，当前主线判断固定为：
  - TODO `#1` 已在当前 clean-tree 主线上收口
  - 下一高优先级 TODO 切换为 `#2`：清理残留 `device 0` 拓扑硬编码

## 本次同步必须写入的最高原则

以下原则优先级高于单轮实验表现、局部现象和临时调参判断：

1. 不能把系统改坏，不能偏离当前 clean-tree 的系统优化主线。
2. 所有修改都必须先服务于已敲定的论文主指标，而不是为局部现象救场。
3. 所有修改都必须对齐论文三项贡献，不能通过绕开贡献路径去“刷数字”。
4. 策略层不允许引入面向单实例、单轮实验、单 adapter 的不合理硬编码。
5. 尽量优先复用系统已经产生的可观测值做优化，避免拍脑袋 heuristics。
6. 不引入无必要的额外计算开销；若必须增加开销，必须证明它直接服务主指标且风险可控。
7. 公式、排序逻辑和成本模型都必须具备系统语义上的可解释性，能和真实运行路径对上。
8. 坚持第一性原则，不接受“先救场再说”的补丁式修复作为正式方案。

## 2026-03-28 凌晨更新：本次同步的目标

本次同步不是为了宣布 7B 性能完全收口，而是为了固定下面这批**已验证结构修复 + 已完成但待实验验证的新观测/路由修复**：

1. `retry30_baseline` 已验证：
   - `GPU0 resident≈0` 的主异常消失
   - `scale-up warmup` 真实生效
   - `Cold_start_latency` 变成可信真值
2. `retry31_baseline` 已正式分析：
   - 当前不是新的结构性 bug
   - backbone-only 路径变快了
   - LoRA runtime path 更慢，headline TTFT 继续变差
   - 轻量 runtime-aware routing 不能作为最终解
3. 当前 clean-tree 新增：
   - `Runtime_TTFT = vllm_ttft_ms` 接入 live / summary / JSON
   - 观测驱动 routing：按 `cache_tier + lora_io_ms + vllm_ttft_ms` 的已观测总成本做 LoRA 路由
4. 当前 `retry32_baseline` 已启动但尚未分析，因此这次快照的价值是：
   - 锁定当前正确主线
   - 方便 `retry32` 前后快速回退
   - 让以后回退时能立刻看懂：`retry31` 为什么失败、当前代码为什么这样改

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

建议提交信息明确写出：

- `retry31` 已正式分析，结论是 router/runtime path 仍是主矛盾
- 当前 clean-tree 已切到观测驱动 routing
- `retry32_baseline` 正在运行，待实验结束后统一读日志分析

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
2. 所有修改先服从本文档顶部列出的最高原则
3. 当前优先等待 `retry32_baseline` 跑完，再正式比较 `retry32 vs retry31 vs retry30`
4. 在 `retry32` 出结论前，不进入下一条高优先级 TODO

如果以后需要共享实验结果，应单独导出、单独整理，而不是直接提交本地运行输出目录。
