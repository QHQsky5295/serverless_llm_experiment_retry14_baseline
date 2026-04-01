# GitHub 同步说明（clean-tree 回退快照）

本文件说明当前 `serverless_llm_experiment_retry14_baseline` 如何同步到 GitHub，以及哪些内容应当、哪些内容不应当进入回退快照。

## 当前远端与分支

- 远端仓库：`https://github.com/QHQsky5295/FaaSLoRA.git`
- 当前 clean-tree 分支：`retry14_rebuild`
- 当前最新已验证 TODO `#2` 收口代码基线提交：`b314262`
- 本轮调研与规划文档首次同步提交：`34881fb`

当前建议：

- 本次同步应把 `retry44_fix15` / `retry44_fix12` 的正式分析结论与当前 TODO `#3` 在研代码状态一起推到当前分支，作为新的可回退快照
- 是否合并到 `main` 另行决策

## 2026-04-01 更新：本次同步应纳入 `retry44_fix15` 正式分析状态、`retry44_fix12` 局部最优结论与当前 TODO `#3` 最终方向

- 当前最新已正式分析结果：`retry44_fix15_baseline @ 500`
- 当前最近局部最优正式结果：`retry44_fix12_baseline @ 500`
- 当前最新已验证运行正常、并已完成正式分析的结果链：
  - `retry43_baseline`
  - `retry44_fix6_baseline`
  - `retry44_fix7_cleanrun2_baseline`
  - `retry44_fix8_baseline`
  - `retry44_fix9_baseline`
  - `retry44_fix10_baseline`
  - `retry44_fix11_baseline`
  - `retry44_fix12_baseline`
  - `retry44_fix15_baseline`
- 当前正式判断：
  - TODO `#2` 仍以 `retry42_fix4 / b314262` 为收口点
  - TODO `#3` 仍是当前唯一 next active 主线
  - 当前还不能进入 TODO `#4/#5`
  - `retry44_fix15` 已明确证明 submitted-window / wider frontier 扩张不是正确方向
  - 当前唯一正确主线是把 TODO `#3` 收敛到 `readiness-aware exact scale-up handoff plan`
- 本次同步应明确纳入：
  - `retry44_fix15` 与 `retry44_fix12` 的 handoff / progress / sync / survey / technical-route 文档结论
  - 当前重新收紧后的最高原则
  - 当前 TODO `#3` 在研代码：
    - `faaslora/experiment/experiment_stack.py`
    - `scripts/run_all_experiments.py`
    - `tests/test_basic_smoke.py`
  - 当前文档镜像：
    - `docs/*.md`
    - `docs copy/*.md`
- 这次同步必须明确区分：
  - `b314262` 是最新已验证 TODO `#2` 收口代码基线
  - 本次新推送快照包含 TODO `#3` 的阶段性结论与当前在研代码状态，同步目的不是宣布 TODO `#3` 已收口
- 当前最新本地测试状态：
  - `tests.test_basic_smoke = 135/135 OK`

本次同步后的默认回退点应能表达：

1. 系统运行形态仍在当前 clean-tree 主线允许的边界内：
   - `primary runtime` 继续保持 in-process
   - scale-up dedicated child 的生命周期修复被保留
2. 当前没有新的 crash 型结构性 bug，TODO `#3` 的主瓶颈仍是 scale-up cold path / preload coverage。
3. `retry44_fix15` 已明确证明：
   - frontier 继续扩张不是正确方向
   - 把 `warmed_adapters` 做大可以消掉局部 `scaleup_affected`，但会显著放大 `Cold_start_latency`
4. 当前下一步应在本次快照基础上继续实现：
   - `ready-time queue horizon`
   - `exact ordered prefix`
   - `prefix-bytes budget under headroom`
   - `plan-only execution`

## 2026-03-31 更新：下一次同步应纳入 `retry44_fix7_cleanrun2` 正式分析状态与当前 TODO `#3` 收代码

- 当前最新已正式分析结果：`retry44_fix7_cleanrun2_baseline @ 500`
- 当前最新已验证运行正常的结果链：
  - `retry43_baseline`
  - `retry44_fix5_baseline`
  - `retry44_fix6_baseline`
  - `retry44_fix7_cleanrun2_baseline`
- 当前正式判断：
  - TODO `#2` 仍以 `retry42_fix4` 为收口点
  - TODO `#3` 是当前唯一 next active 主线
  - 当前还不能进入 TODO `#4`
- 当前下一次同步应明确纳入：
  - `retry44_fix7_cleanrun2` 的 handoff / progress / sync / survey 文档结论
  - 当前重新收紧后的最高原则
  - TODO `#3` 在研代码：
    - `scripts/run_all_experiments.py`
    - `tests/test_basic_smoke.py`
- 这次同步必须明确区分：
  - `b314262` 是最新已验证 TODO `#2` 收口代码基线
  - 当前新推送快照包含 TODO `#3` 的收代码状态与最新 docs，同步目的不是宣布 TODO `#3` 已收口
- 当前最新本地测试状态：
  - `tests.test_basic_smoke = 126/126 OK`

本次同步后的默认回退点应能表达：

1. 运行形态已经回到当前 clean-tree 主线允许的边界内：
   - `primary runtime` 不再被改成 subprocess
   - dedicated child 的生命周期修复被保留
2. 指标层已经具备：
   - `TTFT_warm_standard`
   - `Cost_effectiveness_e2e`
   - `SLO_goodput`
   - schema v3 分层结果导出
3. 当前最新正式结论不是“系统有新的结构性 bug”，而是：
   - TODO `#3` 仍未收口
   - 剩余主瓶颈是 cold-path / preload coverage
   - `retry43` 之后那条更激进的 working-set 扩张链没有改善 headline 指标，应回收至 live-hotset 语义再重新验证

## 2026-03-31 同步特别说明：最近一度偏离主线的改动已经被收回

- 最近围绕 OOM / runtime death 的一段代码，一度把 `primary runtime` 也改成了 subprocess。
- 这条改动改变了运行形态和生命周期语义，不直接服务当前论文主指标，属于偏离 clean-tree 当前优化主线的越界修改。
- 当前这条越界改动已经收回；本次同步不应再把它描述成主线方向。
- 本次同步应明确写入：
  - 可以保留 dedicated child 真正需要的最小生命周期修复
  - 不应再为了局部现象继续扩改运行形态 / 生命周期层
  - 当前所有在研代码都必须继续服从“只服务主指标、只走三项贡献允许路径”的原则

## 2026-03-30 更新：下一次同步应纳入 TODO `#2` 收口代码与 `retry42_fix4` 文档状态

- 当前最新本地已验证结果：`retry42_fix4_baseline @ 500`
- 当前最新正式判断：
  - TODO `#2` 已在 `retry42_fix4` 上实质收口
  - TODO `#1` 仍保持收口，不应回头继续叠控制面
  - 当前 next active TODO 已切换到 `#3`：`scale_up_preload_mb=1024` 的 headroom-aware 动态预算
- 当前下一次同步应纳入：
  - TODO `#2` 收口代码
  - `TTFT_warm_standard / Cost_effectiveness_e2e / SLO_goodput` 指标层更新
  - `retry42_fix4` 的 handoff / progress / sync 文档结论
- 当前下一次同步后，新的默认回退点应能表达：
  - runtime-local topology accounting 已收口
  - stale sibling GPU residency 不再驱动 background forward
  - 评价体系已经具备 paper-facing 的双层指标结构

## 2026-03-29 深夜更新：上一次同步只纳入文档与规划，不纳入当时本地 TODO `#2` 代码修改

- 本次 docs-only 同步已完成并推送：
  - 提交：`34881fb`
  - 分支：`retry14_rebuild`

- 本次同步目的：
  - 固化 related work / 同类论文调研
  - 固化基于调研形成的正式 TODO 排序
  - 更新 handoff / progress / sync 文档，保证以后回退或换会话时可无缝续接
- 本次同步**不纳入**当前本地未提交代码：
  - `faaslora/memory/memory_coordinator.py`
  - `faaslora/memory/residency_manager.py`
  - `faaslora/serving/vllm_wrapper.py`
  - `tests/test_basic_smoke.py`
- 原因：
  - 上述代码属于正在本地验证的 TODO `#2` 主线修改
  - 本次用户明确要求“只改文档和规划，不改代码，只推文档”

## 2026-03-29 晚更新：上一次代码主线同步已完成

- 已推送代码基线：`1544de2`
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
- `faaslora/memory/memory_coordinator.py`
- `faaslora/memory/residency_manager.py`
- `faaslora/scheduling/resource_coordinator.py`
- `faaslora/serving/vllm_wrapper.py`
- `scripts/run_all_experiments.py`
- `tests/test_basic_smoke.py`
- `docs/*.md`
- `docs copy/*.md`
- 新增调研文档：
  - `docs/RELATED_WORK_AND_OPTIMIZATION_SURVEY_2026-03-29.md`
  - `docs copy/RELATED_WORK_AND_OPTIMIZATION_SURVEY_2026-03-29.md`

## 本次同步不应提交的内容

不提交：

- `results/`
- `artifacts/`
- `data/`
- `models/` 下的权重文件
- `/tmp` 下的 launch log
- 本机缓存、虚拟环境和用户级 shell 定制
- 当前本地未完成验证的 TODO `#2` 代码修改（若本次是 docs-only sync）

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
   - `docs/*.md`
   - `docs copy/*.md`
   - 新的调研文档 `RELATED_WORK_AND_OPTIMIZATION_SURVEY_2026-03-29.md`
3. 若当前本地代码已经在正式实验上收口，则本次应把代码与文档一起提交。
4. 在当前工作分支形成一次明确写明 `retry42_fix4 / TODO #2 closure` 的提交。
5. 推送到 GitHub，作为后续可回退的代码与文档入口。

建议提交信息明确写出：

- 本次纳入 `retry42_fix4`
- TODO `#2` 收口
- 指标层已补 `TTFT_warm_standard / Cost_effectiveness_e2e / SLO_goodput`
- 正式 TODO 顺序当前固定为 `#3 -> #4 -> #5`

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
