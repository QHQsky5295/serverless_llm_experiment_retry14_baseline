# 项目进度记录

## 文档用途

本文档记录当前 clean-tree 的真实状态，用于：

- 形成可回退的 GitHub 快照
- 说明当前主线做到哪里
- 统一代码、文档、实验口径
- 为下一轮实验和下一次会话续接提供基线

若与旧实验记录冲突，以本文件和当前代码实现为准。

## 项目迭代最高原则

以下原则优先级高于单轮实验表现、局部现象和临时调参判断：

1. 不能把系统改坏，不能偏离当前 clean-tree 的系统优化主线。
2. 所有修改都必须先服务于已敲定的论文主指标，而不是为局部现象救场。
3. 所有修改都必须对齐论文三项贡献，不能通过绕开贡献路径去“刷数字”。
4. 策略层不允许引入面向单实例、单轮实验、单 adapter 的不合理硬编码。
5. 尽量优先复用系统已经产生的可观测值做优化，避免拍脑袋 heuristics。
6. 不引入无必要的额外计算开销；若必须增加开销，必须证明它直接服务主指标且风险可控。
7. 公式、排序逻辑和成本模型都必须具备系统语义上的可解释性，能和真实运行路径对上。
8. 坚持第一性原则，不接受“先救场再说”的补丁式修复作为正式方案。

## 当前仓库标识

- 项目名称：`FaaSLoRA：面向多 LoRA 大模型推理的扩缩容感知 Serverless 系统`
- 当前干净树：`/home/qhq/serverless_llm_experiment_retry14_baseline`
- 历史脏树：`/home/qhq/serverless_llm_experiment`
- 当前工作分支：`retry14_rebuild`
- 上一已推送基线提交：`70c90fe`
- 远端仓库：`https://github.com/QHQsky5295/FaaSLoRA.git`

当前约定：

- 后续研究与回退统一以 `serverless_llm_experiment_retry14_baseline` 为准。
- `retry21` 及其对应脏树状态视为废案，不再作为正式对比对象。
- 本次 GitHub 同步的目的不是发布最终结论，而是把当前 clean-tree 形成一个稳定回退点。

## 2026-03-29 晚更新快照

### 当前最新已验证实验状态

- 当前最新已验证干净结果：`retry40_baseline @ 500`
- 当前最新已正式分析结果：`retry40 vs retry39 vs retry38`
- 当前最新结构性结论：
  - `retry39` 已正式证明 TODO `#1` 的第一阶段有效，live scale-up 已经不再被 `scale_decision_interval=25` 的请求数硬门槛卡死
  - `retry40` 进一步证明 live 动态 RPS 阈值刷新和事件口径修复后，TODO `#1` 的主瓶颈已经在当前 clean-tree 主线实质收口
  - 当前没有新的结构性 bug，主问题不再是 TODO `#1`
- 当前最新 headline 指标结论：
  - `TTFT_overall = 7196.9 ms`
  - `TTFT_comparable = 8122.7 ms`
  - `TTFT_scaleup_affected = 8722.8 ms`
  - `TTFT_gpu_ready = 8069.4 ms`
  - `Runtime_TTFT = 7042.5 ms`
  - `TPOT = 50.5 ms`
  - `E2E_latency = 10393.8 ms`
  - `Throughput_req/s = 0.13455`
  - `Throughput_tok/s = 17.205`
  - `SLO_attainment = 17.8%`
  - `Cold_start_latency = 49529.5 ms`
  - `Monetary_cost = $0.0034435 / req`
- 相对 `retry39` 的正式改进：
  - `TTFT_overall: 7571.1 -> 7196.9 ms`
  - `TTFT_comparable: 8375.6 -> 8122.7 ms`
  - `TTFT_scaleup_affected: 9146.3 -> 8722.8 ms`
  - `Runtime_TTFT: 7392.2 -> 7042.5 ms`
  - `E2E_latency: 11522.0 -> 10393.8 ms`
  - `Throughput_req/s: 0.10797 -> 0.13455`
  - `Throughput_tok/s: 13.811 -> 17.205`
  - `SLO_attainment: 16.8% -> 17.8%`
  - `Cold_start_latency: 61928.7 -> 49529.5 ms`
  - `P95/P99_TTFT: 10464.1/17195.9 -> 9724.9/15349.4 ms`
  - `P95/P99_E2E: 18706.8/25471.6 -> 14926.9/20427.2 ms`
  - `avg_lora_io_ms: 179.0 -> 154.4 ms`
  - `TPOT` 小幅回退 `48.8 -> 50.5 ms`，但仍优于 `retry38` 的 `52.4 ms`

### 当前最新代码状态

- TODO `#1` 当前主线相关代码已经包含并通过本地测试：
  - live scale-up 评估从“批次尾触发”改为“按真实时间 / 真实压力秒级触发”
  - scale-up 事件输出恢复可比 `request_index=submitted_request_count`，并补充 `submitted/completed/arrived_request_count`
  - live 决策前用实时 `arrival_rps` 刷新动态 RPS 阈值，不再残留 batch-end 滞后
  - scale-down 低负载计时统一使用 monotonic clock
- 当前测试状态：
  - `tests.test_basic_smoke = 98/98 OK`
  - `RuntimeAccountingAndMetricsSmokeTests = 24/24 OK`

### 当前高优先级 TODO 顺序

1. TODO `#1`：按真实时间 / 真实压力评估 scale-up
   - 当前状态：在 `retry40` 上已收口，除非后续出现新回归，否则不应继续为 TODO `#1` 叠加控制面改动

2. TODO `#2`：清理残留 `device 0` 拓扑硬编码
   - 当前成为 next active TODO
   - 只能在不破坏 `retry40` 已收口指标的前提下推进

3. TODO `#3`：`scale_up_preload_mb=1024` 改成 headroom-aware 动态预算
   - 只有在 TODO `#2` 收口后才允许进入

### 当前结论约束

- 当前不应再为了 TODO `#1` 继续改控制逻辑，因为 `retry40` 已经给出足够强的收口证据
- 当前若继续优化，必须优先选择：
  - 不破坏 `TTFT_overall / TTFT_comparable / TTFT_scaleup_affected / E2E / Throughput` 已收口部分
  - 不引入实例级、单轮级、单 adapter 级硬编码
  - 保持公式和事件口径可解释、可回溯到真实运行路径

## 2026-03-28 凌晨更新快照

### 当前最新已验证实验状态

- 当前最新**已验证干净结果**：`retry30_baseline @ 500`
- 当前最新**已正式分析但未通过 headline 目标的结果**：`retry31_baseline @ 500`
- 当前最新**结构性结论**：
  - `GPU0 resident≈0` 的主异常已消失
  - `scale-up warmup` 已真实生效，`warmed_adapters = 14`
  - `Cold_start_latency` 已成为可信真值
  - headline TTFT 仍未显著收口，主矛盾已从 GPU tier 主链转向 `router/runtime path`
- 当前最新**代码状态**：
  - `Runtime_TTFT = vllm_ttft_ms` 已接入 live / summary / JSON
  - `retry31` 已证明“轻量 runtime-aware routing”只改善了 backbone-only 路径，没有收口 headline TTFT
  - 当前 clean-tree 已进一步改成“按 `cache_tier + lora_io_ms + vllm_ttft_ms` 的观测总成本做 LoRA 路由”
  - 上述当前正式修复已通过本地测试，但**仍等待 `retry32_baseline` 的正式实验验证**
- 当前最新**实验推进状态**：
  - `retry32_baseline` 已启动
  - 当前约定是不持续观测，等实验结束后统一读取完整日志和结果做正式分析

### 当前论文与系统迭代最高原则

后续所有代码修改与实验分析都必须同时满足：

1. 最优化已敲定的论文主指标
2. 对齐论文三项贡献，不允许把贡献抹掉或绕开
3. 不把系统改坏，不偏离当前 clean-tree 主线
4. 不引入不合理硬编码，不接受救场式启发式补丁作为正式方案
5. 尽量使用已有观测值，不引入无必要的额外计算开销

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

7. Runtime 观测与观测驱动 routing 修复
   当前 clean-tree 已补：
   - `Runtime_TTFT = vllm_ttft_ms`，并接入 live / summary / JSON
   - router 侧不再先做 affinity 硬筛，而是按 slot 已观测的 `cache_tier + lora_io_ms + runtime_ttft_ms` 预测请求服务成本
   - 在无 exact bucket 观测时，只回落到已有 tier load cost 公式和 slot 已观测 runtime，不新增对象级硬编码

   说明：
   - `retry31` 已确认轻量 runtime-aware routing 没有收口 headline TTFT，且把主矛盾进一步暴露到 LoRA runtime path
   - 当前这批新改动是面向该主矛盾的正式修复
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

#### `retry31_baseline`（已正式分析，未通过 headline 目标）

- `500/500`，`fail=0`
- `TTFT_overall avg = 10050 ms`，相对 `retry30` 变差
- `TTFT_comparable avg = 12073 ms`
- `TTFT_scaleup_affected avg = 8716 ms`
- `TTFT_gpu_ready avg = 11421 ms`
- `TPOT = 49.0 ms`
- `E2E avg = 12984 ms`
- `Throughput = 0.0977 req/s, 12.419 tok/s`
- `SLO@5000ms = 18.4%`
- `avg_cold_start_latency_ms = 58478 ms`
- `avg_lora_io_ms = 485.7 ms`

结论：

- 当前不是新的结构性 bug
- `retry31` 证明 backbone-only 路径确实变快了
- 但 LoRA 请求在两台实例上的 runtime path 更慢，且更多 LoRA 落到更慢实例，导致 headline TTFT 继续变差
- 因此 `retry31` 只说明“轻量 runtime-aware routing”不是最终解，不能作为正式收口方案

### 当前判断

当前 clean-tree 还没有“性能完全收口”，但已经满足：

- 指标口径比前几轮更硬
- 当前主线问题比前几轮更清楚
- 可以作为下一轮实验和以后回退的稳定代码快照

当前更具体的判断是：

- 当前问题不是结构性 bug，而是 router/runtime path 上仍未收口的性能瓶颈
- 当前问题与论文主指标直接相关
- 当前 clean-tree 的正式修复方向已经从“轻量 runtime-aware routing”升级为“观测驱动总成本路由”
- 在 `retry32_baseline` 出结果前，不能进入下一条高优先级 TODO

## 当前保留的关键 TODO

### 本次同步后立即继续的 TODO

1. 等 `retry32_baseline` 结束后读取完整日志和 JSON
2. 正式比较 `retry32 vs retry31 vs retry30`
3. 判断当前观测驱动 routing 是否真的改善 headline TTFT，并确认 backbone-only 改善是否被保住
4. 继续围绕：
   - `TTFT_overall`
   - `TTFT_comparable`
   - `TTFT_scaleup_affected`
   - `TTFT_gpu_ready`
   - `TPOT`
   - `Throughput_req/s`
   - `Throughput_tok/s`
   - `E2E_latency`
   - `SLO_attainment`
5. 只有当 router/runtime 主矛盾验证收口后，才进入 `scale_decision_interval=25`

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

后续所有修改统一服从以下宗旨：

1. 最优化已敲定的论文指标
2. 对齐论文贡献，而不是绕开贡献路径去“刷数字”
3. 不把系统改坏，不偏离当前主线
4. 尽量用已有观测值，不引入不合理硬编码和无必要额外开销

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
