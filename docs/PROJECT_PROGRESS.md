# 项目进度记录

## 文档用途

本文档记录当前 clean-tree 的真实状态，用于：

- 形成可回退的 GitHub 快照
- 说明当前主线做到哪里
- 统一代码、文档、实验口径
- 为下一轮实验和下一次会话续接提供基线

若与旧实验记录冲突，以本文件和当前代码实现为准。

## 当前仓库标识

- 项目名称：`FaaSLoRA：面向多 LoRA 大模型推理的扩缩容感知 Serverless 系统`
- 当前干净树：`/home/qhq/serverless_llm_experiment_retry14_baseline`
- 历史脏树：`/home/qhq/serverless_llm_experiment`
- 当前工作分支：`retry14_rebuild`
- 当前已推送基线提交：`9147eb0`
- 远端仓库：`https://github.com/QHQsky5295/FaaSLoRA.git`

当前约定：

- 后续研究与回退统一以 `serverless_llm_experiment_retry14_baseline` 为准。
- `retry21` 及其对应脏树状态视为废案，不再作为正式对比对象。
- 本次 GitHub 同步的目的不是发布最终结论，而是把当前 clean-tree 形成一个稳定回退点。

## 2026-03-27 晚更新快照

### 当前最新已验证实验状态

- 当前最新**已验证干净结果**：`retry30_baseline @ 500`
- 当前最新**结构性结论**：
  - `GPU0 resident≈0` 的主异常已消失
  - `scale-up warmup` 已真实生效，`warmed_adapters = 14`
  - `Cold_start_latency` 已成为可信真值
  - headline TTFT 仍未显著收口，主矛盾已从 GPU tier 主链转向 `router/runtime path`
- 当前最新**代码状态**：
  - `Runtime_TTFT = vllm_ttft_ms` 已接入 live / summary / JSON
  - router 已补“实例最近真实 runtime 代价”信号，优先修 backbone / 浅路由场景
  - 上述最新 routing/Runtime_TTFT 修复**尚未经过新一轮实验验证**

### 当前论文第一性原则

后续所有代码修改与实验分析都必须同时满足：

1. 最优化已敲定的论文主指标
2. 对齐论文三项贡献，不允许把贡献抹掉或绕开

### 当前必须继续盯的三条高优先级问题

1. `scale_decision_interval=25`
   当前仍是“每处理 N 个请求才评估一次扩容”的请求数硬门槛，会直接拖 `TTFT_scaleup_affected / TTFT_overall`

2. 残留的 `device 0` 拓扑硬编码
   仍散落在 `memory_coordinator / vllm_wrapper / residency_manager` 等路径中，未来在 `14B` 或更复杂运行时仍可能再次放大多 GPU 偏置问题

3. `scale_up_preload_mb=1024`
   当前是固定预算，不是真实的 headroom-aware 动态预加载预算，会直接影响 `scale-up cold path`

## 本次同步纳入的真实变更

本次快照纳入的是当前 clean-tree 已完成、已验证的这批更新：

1. `HOST` 在线形成路径修复
   显式 `NVMe-hit -> HOST promotion` 不再被过度保守的 opportunistic gate 二次挡掉，`retry23+` 已验证 host 层可以真实出现。

2. 多 GPU 口径与 runtime 负载感知修复
   当前代码不再只盯 `device 0` 做部分核心决策，已经补上：
   - autoscaler 多卡利用率取最大值
   - residency shared GPU tier 支持 tracked device ids
   - router/实例提示侧接入 runtime 物理 GPU 利用率

3. 论文主指标口径落地
   终端 live 和 run summary 已统一围绕以下指标输出：
   - `TTFT_overall`
   - `TTFT_comparable`
   - `TTFT_scaleup_affected`
   - `TTFT_gpu_ready`
   - `TPOT`
   - `Throughput_req/s`
   - `Throughput_tok/s`
   - `E2E_latency`
   - `SLO_attainment`

4. `scaleup_affected` 升级成请求级真标签
   不再使用“固定 50 请求窗口”启发式近似；多周期 live 的索引错位也已一并修掉。

5. 暗箱 heuristics 显式化但不改行为
   当前仓库已经把以下旧隐式常数显式配置化，并保持原有效值不变：
   - `serverlessllm_overhead_ratio = 0.6`
   - `host/cpu/nvme/remote locality factors`
   - `scale_up_preload_mb = 1024`

6. 当前 Qwen 7B V2 publicmix 工件清单已更新为 curated baseline
   `configs/generated/lora_manifest_1000.json` 当前刻意随仓库快照一起提交，用于后续回退和复现实验入口。

7. Runtime 观测与 runtime-aware routing 的最小修复
   当前 clean-tree 已补：
   - `Runtime_TTFT = vllm_ttft_ms`，并接入 live / summary / JSON
   - router 侧基于实例最近真实 runtime 代价的轻量信号

   说明：
   - 这批改动是为了继续解决 `retry30` 中“headline TTFT 变差但结构性 bug 已消失”的问题
   - 当前已经过本地测试，但还没有新的正式实验轮次验证

## 当前论文主指标

后续系统优化和论文主表统一围绕以下指标：

1. `TTFT_overall`
2. `TTFT_comparable`
3. `TTFT_scaleup_affected`
4. `TTFT_gpu_ready`
5. `TPOT`
6. `Throughput_req/s`
7. `Throughput_tok/s`
8. `E2E_latency`
9. `SLO_attainment`

实验结束 summary / JSON 还补充输出：

10. `Cold_start_latency`
11. `Monetary_cost`

其中：

- `TTFT_comparable` 当前定义为：`非 scale-up 受影响` 且 `cache_tier ∈ {gpu, host, nvme}` 的请求 TTFT。
- `TTFT_scaleup_affected` 当前定义为：真实路由到 scale-up 新增 runtime，且当时不是 `GPU-ready` 的请求 TTFT。
- `Cold_start_latency` 当前定义为：dedicated scale-up 从开始创建新实例到新实例 warmup 完成的时延。
- `Monetary_cost` 当前输出为：`avg_cost_usd` 与 `total_cost_usd`。

其余指标继续完整记录在日志和 JSON 中，用于归因和调优，但不作为论文 headline。

## 当前主线实验状态

### 当前主线对象

当前 rollback 主线聚焦于：

- 模型：`Qwen2.5-7B-Instruct`
- profile：`qwen_7b_main_v2_publicmix`
- workload：`qwen_7b_auto500_main`
- dataset：`azure_sharegpt_rep1000`
- 正式回归入口：`500 requests`
- mode：`auto`
- max instances：`2`
- GPU：`2 × RTX 3090 24GB`

说明：

- `configs/experiments.yaml` 顶层默认 `profile_selection` 仍保留 `Qwen 14B TP=2` 入口，便于总配置浏览。
- 当前 7B rollback 实验统一通过环境变量覆盖 profile 选择与 `FAASLORA_TOTAL_REQUESTS=500` 进入。

### 最近已正式分析的轮次

#### `retry22_baseline`

- `500/500`，`fail=0`
- `avg_ttft_ms = 8843.08`
- `p95_ttft_ms = 18541.37`
- `throughput_rps = 0.1144`
- `avg_lora_io_ms = 406.10`
- tier 分布：`gpu=334 / host=0 / nvme=84 / remote=82`

结论：

- 假 GPU hit / 重复加载问题已被清掉
- 但 host 层完全没形成，成为后续主修复点

#### `retry23_baseline`

- host 恢复成非零
- tier 分布：`gpu=270 / host=64 / nvme=84 / remote=82`
- `avg_ttft_ms = 8596`
- `avg_lora_io_ms = 771`

结论：

- `NVMe -> HOST` 在线 promotion 修复确实生效
- 但 GPU0-only pressure 异常开始显性暴露

#### `retry24_baseline`

- tier 分布：`gpu=329 / host=5 / nvme=84 / remote=82`
- `avg_ttft_ms = 8919`
- `p95_ttft_ms = 18738`
- `p99_ttft_ms = 22691`
- `avg_lora_io_ms = 468`

结论：

- host 又几乎塌回去
- 但从分解指标看，主瓶颈已不再只是 LoRA I/O，而是 runtime / queue / serving path

#### `retry26_baseline`

- tier 分布：`gpu=293 / host=41 / nvme=84 / remote=82`
- `avg_ttft_ms = 9264.3`
- `p95_ttft_ms = 20856.1`
- `p99_ttft_ms = 27715.9`
- `avg_comparable_ttft_ms = 10272.7`
- `avg_scaleup_affected_ttft_ms = 6598.7`

结论：

- host 层比 `retry24` 更健康
- 但后半段 `GPU0 100% + inst_1 resident≈0` 的异常仍然存在
- 这轮直接推动了后续多 GPU 口径和请求级 `scaleup_affected` 修复

#### `retry27_baseline`

- `500/500`，`fail=0`
- tier 分布：`gpu=221 / host=113 / nvme=84 / remote=82`
- `avg_ttft_ms = 9097.6`
- `p95_ttft_ms = 18627.4`
- `p99_ttft_ms = 25686.4`
- `avg_e2e_ms = 12031.3`
- `throughput_rps = 0.1138`
- `avg_lora_io_ms = 1113.8`
- `avg_comparable_ttft_ms = 9525.5`
- `avg_gpu_ready_ttft_ms = 8107.4`
- `avg_scaleup_affected_ttft_ms = 6137.2`

结论：

- router 侧物理 GPU util 修复带来了 `p95/p99/throughput` 改善
- host 形成明显增强
- 但 GPU resident stickiness 仍弱，热点 adapter 反复从 host 走 `host->gpu`
- 当前主问题已收敛到：在 host 形成后，如何稳定提高 GPU-ready 命中并压低重复加载

#### `retry29_baseline`

- `500/500`，`fail=0`
- `TTFT_overall avg/p95/p99 = 9245 / 19518 / 27138 ms`
- `TTFT_comparable avg = 11102 ms`
- `TPOT = 46.7 ms`
- `Throughput = 0.1086 req/s, 13.88 tok/s`
- `avg_lora_io_ms = 440.0 ms`
- `gpu_hit_rate = 78.47%`

结论：

- 结构性运行健康
- 但后续下钻发现：headline TTFT 变差的主要来源不是 LoRA 请求，而是 backbone-only 请求在 `inst_1` 上明显更慢

#### `retry30_baseline`（当前最新干净验证结果）

- `500/500`，`fail=0`
- `TTFT_overall avg/p95/p99 = 9338 / 20589 / 27659 ms`
- `TTFT_comparable avg/p95/p99 = 11037 / 21640 / 27964 ms`
- `TTFT_scaleup_affected avg/p95 = 8000 / 8880 ms`
- `TTFT_gpu_ready avg = 10185 ms`
- `TPOT = 43.1 ms`
- `E2E avg/p95/p99 = 12421 / 22475 / 27964 ms`
- `Throughput = 0.1007 req/s, 12.856 tok/s`
- `SLO@5000ms = 17%`
- `avg_lora_io_ms = 530.3 ms`
- `avg_cold_start_latency_ms = 72983 ms`
- tier 分布：`gpu=323 / host=11 / nvme=84 / backbone=82`

结论：

- `GPU0 anomaly` 已被打掉
- `scale-up warmup` 已真实落地
- 7B 主线的结构性问题基本收口
- 但 headline TTFT 仍偏高；当前主矛盾已收敛到：
  - `router/runtime path`
  - 尤其是 `inst_1` 上的 backbone/runtime 请求明显慢于 `inst_2`

### 当前判断

当前 clean-tree 还没有“性能完全收口”，但已经满足：

- 指标口径比前几轮更硬
- 当前主线问题比前几轮更清楚
- 可以作为下一轮实验和以后回退的稳定代码快照

## 当前保留的关键 TODO

### 本次同步后立即继续的 TODO

1. 跑一轮新的 baseline（当前建议 `retry31_baseline`）
2. 验证 `Runtime_TTFT` 和 runtime-aware routing 是否真的改善 headline TTFT
3. 继续围绕：
   - `TTFT_overall`
   - `TTFT_comparable`
   - `TTFT_scaleup_affected`
   - `TPOT`
   - `Throughput_req/s`
   - `Throughput_tok/s`
   - `E2E_latency`
   - `SLO_attainment`
4. 若 `retry31` 没有新结构性 bug，再判断 7B 是否收尾并转 14B

### 已明确但暂缓的 TODO

1. `REMOTE` 真实性升级
   当前 `REMOTE` 仍是“本地目录 + 带宽仿真”的物理半模拟实现。后续计划接第二台非 GPU 节点做真实远端存储；本次同步只记录，不动代码。

2. 把不合理的暗箱 heuristics / 硬门槛继续逐步替换成：
   - 校准后的测量值
   - 或真实系统指标驱动
   当前已经明确的对象包括：
   - `scale_decision_interval=25`
   - 残留 `device 0` 拓扑硬编码
   - `scale_up_preload_mb=1024`

### 当前工作原则

后续所有修改统一服从两条宗旨：

1. 最优化已敲定的论文指标
2. 对齐论文贡献，而不是绕开贡献路径去“刷数字”

## GitHub 同步边界

本次快照应提交：

- source code
- configs
- docs
- tests
- 当前刻意 curated 的 `configs/generated/lora_manifest_1000.json`

本次快照不提交：

- `results/`
- `artifacts/`
- `data/`
- 本地模型权重
- `/tmp` 日志

本文件在每次 GitHub 同步前后都应更新。
