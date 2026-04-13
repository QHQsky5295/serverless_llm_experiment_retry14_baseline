# FaaSLoRA 技术路线与实现说明

本文档是当前仓库的**权威技术说明文档**。内容对齐当前实现，而不是历史设计稿。若 README、实验指南与本文冲突，以本文为准。

相关工作、同类论文实验方式、可比性边界以及基于调研形成的后续优化建议，统一见：

- [RELATED_WORK_AND_OPTIMIZATION_SURVEY_2026-03-29.md](RELATED_WORK_AND_OPTIMIZATION_SURVEY_2026-03-29.md)

> 2026-04-13 当前主线补充：
>
> - 当前正式论文模型家族已固定为 `Qwen + Llama-2` 两个家族、共 4 个模型：
>   - `Qwen 7B`
>   - `Qwen 14B TP=2`
>   - `Llama-2 7B`
>   - `Llama-2 13B TP=2`
> - `Mistral 7B / Mistral Nemo` 当前保留为历史迁移验证与 serving envelope 诊断结果，不再作为论文主线正式家族。
> - 当前正式综合主指标已固定为：
>   - `CE = 1 / (avg_e2e_sec * avg_cost_usd)`
> - 当前技术路线的正式收口判断是：
>   - `C1` 命中感知工件就绪机制已在四模型上形成可信主链结果
>   - `C2/C3` 仍有提升空间，但当前正式 workload 下尚未形成“继续动系统主链就一定更优”的强证据
>   - 因此当前技术路线应先进入跨论文系统对比阶段，再决定是否重开下一轮系统优化
> - 当前不再允许的做法：
>   - 为了单模型单指标局部变好，继续打补丁式修复
>   - 将模型 serving envelope 差异误判为系统主链漏洞

> 2026-04-12 当前主线补充：
>
> - 当前最新已正式分析、且属于 `continuous_queue_v2` 主线的 14B **最可信 checkpoint** 是 `retry14_continuous_queue_v2_qwen14b_r500_a500_main_baseline45_poststartup_elapsed @ 500`。
> - 当前最新已正式分析、且属于 `continuous_queue_v2` 主线的另一模型家族 7B **最可信 checkpoint** 是 `retry14_continuous_queue_v2_mistral7b_r500_a500_main_baseline3_maxloras4 @ 500`。
> - 当前正式结论必须进一步收紧为：
>   - 7B 继续以 `baseline44` 作为可信 checkpoint
>   - 14B 上第一项贡献对应的 handoff/control 语义链已经软收口
>   - `Mistral 7B baseline3` 已证明当前系统主线不是 Qwen 家族特化，而可以迁移到另一模型家族
>   - 当前最关键证据是：
>     - `scaleup_first_service_planned_match_rate = 1.0`
>     - `scaleup_first_service_gpu_hit_rate = 1.0`
>     - `TTFT_scaleup_first_service_avg_ms = 546.3`
>     - `Mistral 7B baseline2 -> baseline3`：
>       - `TTFT_overall = 5857.0 -> 1397.9 ms`
>       - `TTFT_comparable = 5152.0 -> 1160.7 ms`
>       - `avg_lora_io_ms = 1624.4 -> 303.6 ms`
>       - `QPR = 1839.4 -> 9121.1`
> - 当前 14B 尚未收口的问题已经从 C1 转到 C2/C3：
>   - `TTFT_scaleup_affected` 仍偏高
>   - `Cold_start_latency` 仍接近 `68s`
>   - `warm_pool_hits = 0`
> - 因此当前技术路线不应再理解成“继续围绕 14B 的 first-service handoff 打转”，而应理解成：
>   - 保持当前 7B 与 14B checkpoint
>   - 保持 `Mistral 7B baseline3` 这个跨家族 7B checkpoint
>   - 切到 `Mistral Nemo TP=2`
>   - 如果另一模型家族复现同样的后续接管冷路径问题，再回到跨模型共通的 C2/C3 主线继续优化

> 2026-04-11 当前主线补充：
>
> - `continuous_queue_v2` 当前最新已正式分析、且最可信的 7B checkpoint 是 `retry14_continuous_queue_v2_qwen7b_r500_baseline44_startup_budget @ 500`。
> - 当前正式结论必须进一步收紧为：
>   - TODO `#2R`：不需要重开
>   - 7B 上的 `handoff exactness / async scale control / startup budget` 主链已达到 soft-close checkpoint
>   - `baseline35` 不再被当成真实上界，因为其部分优势来自更乐观的 `bootstrap / ready-time` 语义
>   - 下一步：切到 `Qwen 14B TP=2`，再切到另一模型家族，验证当前三项贡献的可迁移性
> - `baseline44` 相比最近两轮可信主线的真实信号：
>   - 相比 `baseline43`，`TTFT_overall / TTFT_comparable / TPOT / E2E / QPR / ColdStart` 全部明显回正
>   - 相比 `baseline41`，`TTFT_overall / TTFT_comparable / TTFT_warm / TPOT / P95_TTFT / QPR` 进一步改善
> - 当前技术路线已经确认并保留的修复链包括：
>   - readiness-aware exact handoff prefix
>   - route-aware empty prefix 不伪预热
>   - runtime 启动后的 handoff refresh
>   - async scale control
>   - pending runtime sequence-aware slicing
>   - event request-order sorting
>   - startup fan-out budget 复用 `max_concurrent_loads`
> - 当前还没有发挥作用、但属于下一阶段 C3 主线的机制是：
>   - `warm pool retention`
>   - `受控资源保留`
>   - 当前结果信号是 `warm_pool_hits = 0`
> - 因此当前技术路线不应再理解成“继续在 7B 上围绕 handoff 细节无限返工”，而应理解成：
>   - 先冻结当前 7B checkpoint
>   - 再在更大模型和另一模型家族上验证同一套请求放置、三层驻留和资源协同控制机制
>   - 若跨模型验证重现同一条 C3 瓶颈，再回到 7B 与 14B 共通主链继续收口

> 2026-04-09 当前主线补充：
>
> - `continuous_queue_v2` 当前最新已正式分析的 7B 有效结果是 `retry14_continuous_queue_v2_qwen7b_r500_baseline34_multiruntime_routeaware @ 500`。
> - 当前正式结论必须收紧为：
>   - TODO `#2R`：不需要重开
>   - 7B：达到可冻结的 soft-close checkpoint，但未证明 TODO `#3` 在所有 headline 指标上完全收口
>   - 下一步：切到 `Qwen 14B TP=2` 的 `continuous_queue_v2` bring-up，验证当前三项贡献的可迁移性
> - `baseline34` 相比 `baseline33` 已改善：
>   - `TTFT_scaleup_affected`
>   - `TTFT_scaleup_first_service`
>   - `TPOT`
>   - `E2E_latency`
>   - `Throughput_tok/s`
> - 但 `baseline34` 相比 `baseline30` 仍没有完全拿回：
>   - `TTFT_overall`
>   - `TTFT_comparable`
>   - `GPU_hit_rate`
> - 因此当前技术路线不应再理解成“继续在 7B 上无限细抠局部指标”，而应理解成：
>   - 先冻结当前 7B checkpoint
>   - 再在更大模型上检验现有请求放置、三层驻留与资源协同控制三条贡献链是否仍成立
> - 当前待同步代码已经把最近几轮主要语义错位统一到了同一条因果链上：
>   - lane-aware instance routing/state guard
>   - multi-runtime route-aware scale-up handoff plan
>   - 4GPU 运行语义保持不变
>
> 2026-04-03 当前主线补充：
>
> - `continuous_queue_v2` 当前最新已正式分析结果是 `retry14_continuous_queue_v2_qwen7b_r500_baseline4_cadencefix @ 500`。
> - 当前正式结论已经收紧为：
>   - TODO `#2R`：当前主线可冻结，不再是 next active TODO
>   - TODO `#3`：重新成为唯一 next active 主线，而且必须在 `continuous_queue_v2` 上继续推进
> - 当前 `continuous_queue_v2` 的 official workload 语义已固定为：
>   - `Azure real trace arrivals`
>   - `Azure token distribution`
>   - `ShareGPT prompts`
>   - `time_scale_factor = 1.0`
>   - 若真实 Azure / ShareGPT 数据缺失则 fail-fast，不允许退回 `synthetic_poisson` 或 embedded prompts
> - `baseline4_cadencefix` 相对 `baseline3_realtiming` 已明显改善 `TTFT_overall / TTFT_comparable / TTFT_scaleup_affected / Cold_start_latency / throughput`，说明当前 substrate 已足够稳定，可以作为后续 TODO `#3` 的正式优化基线。
> - 但 `scaleup_affected` 请求仍全部落在 `host/nvme`，因此下一步仍必须围绕 scale-up cold path / preload coverage / handoff plan，而不是再去扩改 substrate 语义。
>
> 2026-03-31 当前主线补充：
>
> - 当前 clean-tree 的 active mainline 已经从 TODO `#2` 切到 TODO `#3`：`scale_up_preload_mb=1024` 的 headroom-aware 动态预算。
> - 当前继续允许进入主线的修改，必须直接服务：
>   - `TTFT_scaleup_affected`
>   - `Cold_start_latency`
>   - `TTFT_overall`
>   - `TTFT_comparable`
>   - `GPU_hit_rate`
>   - `avg_lora_io_ms`
> - 2026-03-30~31 一度出现过把 `primary runtime` 也改成 subprocess 的越界改动；这类“改运行形态而不是改当前因果链”的做法已经被判定为偏离当前主线，并已收回。
> - 当前技术路线重新收紧为：
>   - `primary runtime` 保持 in-process
>   - scale-up dedicated runtime 允许是 subprocess
>   - 所有后续优化继续围绕 cold-path / preload coverage 因果链，不再扩改运行形态
> - 当前最新正式分析结果是 `retry44_fix7_cleanrun2_baseline @ 500`；它说明 `retry43` 之后更激进的 TODO `#3` working-set 扩张链没有正确收口。
> - 当前本地最新未验证代码，已经把 TODO `#3` 的 warmup preferred set 与 dynamic preload budget target 收回到 `live hotset` 语义，同时保留 foreign GPU consumer fail-fast guard；下一轮正式验证目标是 `retry44_fix8_baseline`。
>
> 2026-04-01 当前主线补充：
>
> - 当前最新已正式分析结果已推进到 `retry44_fix15_baseline @ 500`；当前最近局部最优正式结果是 `retry44_fix12_baseline @ 500`。
> - `retry44_fix8 -> retry44_fix12` 说明，从 `retry43 -> fix6 -> fix7_cleanrun2` 的激进扩张链回撤之后，headline `TTFT / throughput / avg_lora_io_ms` 已明显回正。
> - 但 `retry44_fix8 -> retry44_fix15` 都没有把 `scaleup_affected` 请求真正推向 `GPU-ready`；`retry44_fix15` 还进一步证明 frontier 放大到 `18` 个 adapter 会把 `Cold_start_latency` 推到 `93145.9 ms`。
> - 因此，当前 TODO `#3` 的最终正确方向已经收敛为：
>   - 不再继续做 `live hotset vs working set` 或 `submitted-window frontier` 的补丁式扩张
>   - 直接实现 `readiness-aware exact scale-up handoff plan`
>   - 也就是：`ready-time queue horizon -> exact ordered prefix -> prefix-bytes budget -> plan-only execution`
> - 当前必须保持冻结、不在 TODO `#3` 中混改的实验口径：
>   - `time_scale_factor = 0.02`
>   - `active_adapter_cap = 48`
>   - `hotset_rotation_requests = 100`
>   - `scale_decision_interval = 25`
>
> 2026-04-02 当前主线补充：
>
> - 当前最新已正式分析、且仍属于 `substrate_v1` 历史基线的结果已推进到 `retry44_fix16_baseline @ 500`。
> - `retry44_fix16` 相对 `retry44_fix15 / retry44_fix12` 进一步改善了 `TTFT_overall / TTFT_comparable / TTFT_scaleup_affected / Cold_start_latency / avg_lora_io_ms`，但 `scaleup_affected` 请求仍未真正进入 `GPU-ready`。
> - 当前已经重新确认：旧 runner 不是 production-correct 的连续在线队列，而是 `arrival/backlog` 在线、`submission/decision` batch 的 `substrate_v1`。
> - 因此当前技术路线必须拆成两层：
>   - `retry14_rebuild`：冻结 `substrate_v1` 历史状态
>   - `retry14_continuous_queue_v2`：开启 `TODO #2R = continuous online queue substrate v2`
> - 新分支不会回退 4GPU 语义。当前代码已经完成 `4 × RTX 3090 24GB` 的设备分组、dedicated subprocess 隔离和 `max_instances` 适配；`retry44_fix17_4gpu_baseline` 已人工中止，作废，不作为正式结果。

## 1. 系统定位

FaaSLoRA 是一个面向多 LoRA 大模型推理的单节点 Serverless 研究原型。系统当前运行在单节点多 GPU 服务器上，本地最新代码基线已适配 `4 × RTX 3090 24GB`。系统通过共享 backbone、工件感知请求放置、分层驻留和资源协同控制，研究以下问题：

1. 请求被放置到现有函数实例时，目标实例是否已经命中所需 LoRA 工件。
2. 系统扩展新的物理 GPU 执行实例时，新实例是否能够带着热点工件快速进入服务。
3. 在 LoRA 工件加载、KV 缓存和批推理共享 GPU 资源的条件下，如何控制争用带来的 TTFT 和尾延迟代价。

当前系统不是生产级云平台，而是一个**真实可运行的单节点多 GPU Serverless LoRA 原型**。它已经可以稳定运行 `shared / auto / dedicated` 三种模式，并支持 `100 / 300 / 500 / 1000` adapters 的实验矩阵。

## 1.1 当前论文评测口径

当前 clean-tree 已把论文主指标收口为：

1. `TTFT_overall`
2. `TTFT_comparable`
3. `TTFT_scaleup_affected`
4. `TTFT_gpu_ready`
5. `TPOT`
6. `Throughput_req/s`
7. `Throughput_tok/s`
8. `E2E_latency`
9. `SLO_attainment`

另外补充两条论文辅助指标：

10. `Cold_start_latency`
11. `Monetary_cost`

其中最关键的两条口径是：

- `TTFT_comparable`：从请求到达到首 token，只统计 `非 scale-up 受影响` 且 `cache_tier ∈ {gpu, host, nvme}` 的请求。这个定义刻意保留本地 tier 差异，用于体现“热点工件动态保活”对真实本地路径的收益。
- `TTFT_scaleup_affected`：只统计真实路由到 scale-up 新增 runtime，且当时不是 `GPU-ready` 的请求。当前代码已经使用请求级真标签，不再依赖“固定 50 请求窗口”启发式近似。
- `Cold_start_latency`：只统计 dedicated scale-up，从开始创建新实例到新实例 warmup 完成的真实时延。它解释 `TTFT_scaleup_affected` 的冷路径前半段，不替代 TTFT。
- `Monetary_cost`：当前输出 `avg_cost_usd` 与 `total_cost_usd`，用于论文中的成本补充比较，但不是当前 live 优化第一优先级。

其余诊断指标仍会完整记录到日志和 JSON 中，用于归因与系统优化，但不再作为论文 headline。

## 1.2 当前实现边界

当前 `REMOTE` 的系统语义是真实存在的，但物理实现仍是：

- 本地源目录
- 加配置化带宽模型

因此它目前属于“逻辑真实、物理半模拟”的路径。双机真实 remote 存储升级已列入后续 TODO，但不属于当前 clean-tree 代码快照的改动范围。

## 2. 术语约定与形式化记号

文档统一使用以下术语：

- **函数实例 / function instance**：请求调度与隔离的逻辑单位，对应 `InstancePool` 中的一个 slot。
- **物理 GPU 执行实例 / physical GPU execution instance**：绑定某个 GPU 的独立推理 runtime。
- **worker node**：物理宿主机 / 计算节点。当前实现固定为单节点。
- **shared execution slot**：仅用于解释 `shared` 模式，表示多个逻辑 slot 共享同一物理 runtime。

当前文档中的主要记号如下：

| 记号 | 含义 |
|---|---|
| `r_k = (a_k, \ell_k, x_k, y_k)` | 第 `k` 个请求；`a_k` 为到达时间，`\ell_k` 为目标 LoRA，`x_k / y_k` 为输入/输出 token 数 |
| `I_t` | 时刻 `t` 的函数实例集合 |
| `G_t` | 时刻 `t` 的物理 runtime 集合；在 `shared` 模式下可有 `|I_t| > |G_t|` |
| `A_i(\ell)` | 请求对实例 `i` 的工件亲和度分数，取值 `{0,1,2,3}` |
| `q_i` | 实例 `i` 的 `load_queue_depth` |
| `n_i` | 实例 `i` 的 `active_requests` |
| `u_i` | 实例 `i` 的 `gpu_utilization_pct` |
| `\lambda_t` | 当前观测到的 arrival RPS |
| `\bar{\lambda}_t` | arrival RPS 的 EWMA 基线 |
| `b_t` | 当前 backlog 深度 |
| `\rho_t` | busy runtime ratio，即有请求在执行的 runtime 占比 |
| `L_t` | 当前批次的平均 E2E 延迟观测 |
| `w_t` | 本次 scale-down 使用的 warm pool 大小 |

本文中的符号含义与当前代码实现一一对应；若与历史设计稿冲突，以代码为准。

## 3. 当前系统架构

```text
                 +--------------------------------------+
                 | Azure Trace + ShareGPT Prompt Pool   |
                 | arrivals / tokens / prompt sampling  |
                 +------------------+-------------------+
                                    |
                                    v
                      +-------------+-------------+
                      | ScenarioRunner            |
                      | scripts/run_all_experiments.py |
                      +-------------+-------------+
                                    |
                +-------------------+-------------------+
                |                                       |
                v                                       v
      +---------+---------+                  +----------+----------+
      | AutoScaler        |                  | Router / InstancePool|
      | arrival/backlog   |                  | shared / auto / dedicated |
      | busy/latency      |                  | adapter_affinity routing   |
      +---------+---------+                  +----------+----------+
                |                                       |
                +-------------------+-------------------+
                                    |
                                    v
                         +----------+-----------+
                         | ExperimentStack      |
                         | - PreloadingPlanner  |
                         | - PreloadingManager  |
                         | - ResidencyManager   |
                         | - ResourceCoordinator|
                         +----------+-----------+
                                    |
                     +--------------+---------------+
                     |                              |
                     v                              v
            +--------+--------+             +-------+--------+
            | GPU0 runtime    |             | GPU1 runtime   |
            | vLLM/Transformers|            | vLLM runtime   |
            +--------+--------+             +-------+--------+
                     \                              /
                      \                            /
                       +-----------+--------------+
                                   |
                                   v
                  REMOTE source -> NVME -> HOST -> GPU VRAM
```

## 4. 模块划分

### 4.1 工作负载生成

职责：

- 读取 Azure LLM trace
- 加载 ShareGPT prompt pool
- 构造带真实到达间隔和 token 分布的 replay workload

关键实现：

- `faaslora/datasets/dataset_loader.py`
- `faaslora/datasets/workload_generator.py`

当前方法学：

- Azure trace 提供 `a_k` 与 token 规模骨架。
- ShareGPT 提供 prompt 文本池，而不是逐条完整对话回放。
- `representative` 采样保持以下统计特征与 full trace 近似一致：
  - inter-arrival CDF
  - context / generated token CDF
  - burst ratio
- 采样统计会写入结果文件中的 `sampling_stats`。

当前验证通过并仍在使用的论文主线配置是：

- `auto`
- `500` adapters
- `500` representative requests
- `4 x RTX 3090 24GB`
- `time_scale_factor = 1.0`
- `active_adapter_cap = 48`
- `hotset_rotation_requests = 100`
- `max_model_len = 1024`
- `max_num_seqs = 2`
- `max_loras = 6`
- `runtime_concurrency_cap = 2`
- `max_num_batched_tokens = 1024`
- `scale_eval_interval_s = 5.0`
- `arrival_window_s = 5.0`

### 4.2 路由与实例池

职责：

- 管理 `shared / auto / dedicated` 三种实例模式
- 在请求到达时选择目标实例
- 维护逻辑实例与物理 runtime 的对应关系

关键实现：

- `faaslora/experiment/instance_pool.py`
- `scripts/run_all_experiments.py`

当前默认路由策略名仍是 `adapter_affinity`，但其当前真实实现已经不是“纯缓存亲和 + 简单 tie-break”，而是：

1. 先基于 slot 当前可见的 `gpu / host / nvme / remote` tier 预测请求落点。
2. 再按该 tier 的已观测 bucket 成本估计请求的 `service_cost`：
   - LoRA 请求使用 `avg_lora_io_ms + avg_runtime_ttft_ms`
   - backbone 请求使用 `avg_runtime_ttft_ms`
   - 当 bucket 观测不足时，退回到按 `cache_tier + adapter_size` 估计的 fallback cost
3. 同时按 slot 当前 `active_requests / runtime_concurrency_cap` 估计 `occupancy_cost`。
4. 最终按以下元组取最小值：
   - `(service_cost + occupancy_cost, service_cost, occupancy_cost, load_queue_depth, active_requests, gpu_utilization_pct, last_selected_at, created_at)`

也就是说，当前路由器的系统语义是：**优先选择“当前请求预测服务代价更低、且不会把 decode / occupancy 压力推得更高”的实例**，而不是只按静态 tier 亲和度做派单。

## 5. 实例模式

### 5.1 `shared`

- 所有逻辑 slot 共享同一个物理 runtime。
- scale-up 增加的是逻辑 slot，因此事件类型是 `logical_scale_up`。
- 不会新建第二个物理 vLLM runtime。

### 5.2 `auto`

- 优先尝试新增独立物理 runtime。
- 若物理扩容失败且模式不是 `dedicated`，再回退到 `shared` 路径。
- 在当前 `4 × RTX 3090` 代码基线下：
  - `TP=1` 主线运行时最多扩到 `4` 个单卡 runtime
  - `TP=2` 主线运行时最多扩到 `2` 个双卡 runtime

### 5.3 `dedicated`

- 仅允许新增独立物理 runtime。
- 不回退到 `shared` slot。
- 配置接口仍保留，但当前不作为主线推进对象。

## 6. 扩缩容机制

### 6.1 决策时机

`retry14_rebuild` 当前冻结的 `substrate_v1` 主实验路径在 `ScenarioRunner.run()` 中采用**按批观测、批后决策**：

- 每批大小等于 `scale_decision_interval`
- 当前批完成后，系统观测该批期间的峰值 backlog、峰值 active requests 和峰值 busy ratio
- 然后用这组观测构造 `ScalingMetrics`

因此，`substrate_v1` 的扩缩容不是“每个请求到达时立即决策”，而是**批级离散决策**。

这已经被正式认定为：

- 可作为历史实验 substrate 保留
- 但不应再被描述成最终 production-correct 运行形态
- 后续 `retry14_continuous_queue_v2` 需要把这里升级为连续在线队列语义

### 6.2 动态阈值

当前主线默认开启 `dynamic_scaling`。批级状态更新为：

- `\bar{\lambda}_t = (1 - \beta) \bar{\lambda}_{t-1} + \beta \lambda_t`
- `T_up(t) = max(T_min, \bar{\lambda}_t (1 + \alpha))`
- `T_down(t) = \beta_down \bar{\lambda}_t`

其中：

- `\beta = baseline_rps_ewma_beta`
- `\alpha = scale_up_alpha`
- `T_min = scale_up_t_min`
- `\beta_down = scale_down_beta`

若 `\lambda_t < T_down(t)`，系统记录一次低负载开始时刻；只有持续时间超过 `scale_down_duration_s`，warm-pool 触发条件才成立。

### 6.3 AutoScaler 规则

当前 `AutoScaler` 已实现如下规则：

| 规则 | scale-up 阈值 | scale-down 阈值 | 类别 |
|---|---:|---:|---|
| `gpu_utilization` | `80%` | `48%` | `service_saturation` |
| `memory_utilization` | `85%` | `50%` | `service_saturation` |
| `response_time` | `2000 ms` | `500 ms` | `latency_degradation` |
| `instance_busy_ratio` | `0.75` | `0.10` | `service_saturation` |
| `queue_length` | `10` | `2` | `arrival_pressure` |
| `requests_per_second` | `T_up(t)` 或静态阈值 | `T_down(t)` 或静态阈值的 `0.3x` | `arrival_pressure` |

当前实验 runner 注入的关键观测包括：

- `requests_per_second = \lambda_t`
- `request_queue_length = b_t`
- `avg_response_time_ms = L_t`
- `gpu_utilization`
- `active_requests`
- `instance_busy_ratio = \rho_t`

当前主线并未向在线决策路径注入 CPU 利用率，因此 CPU 规则不会参与当前主实验决策。

### 6.4 投票与执行语义

AutoScaler 的规则先按类别汇总；同一类别内，正向负载证据优先于“空闲”证据。记：

- `S_up` 为归一化后的 scale-up score
- `S_down` 为归一化后的 scale-down score

当前决策条件是：

- 若 `S_up > 0.5`，满足 cooldown，且 `|I_t| < max_instances`，则触发 scale-up
- 若 `S_down > 0.5`，满足 cooldown，且 `|I_t| > min_instances`，并且没有任何正向 scale-up 类别证据，则触发 scale-down

执行层语义如下：

- `shared`：新增逻辑 slot，记录 `logical_scale_up`
- `auto`：优先新增独立 runtime，记录 `physical_scale_up`
- `dedicated`：只新增独立 runtime，失败则不回退

在当前单周期主实验中：

- 批级 scale-up 由上述决策逻辑驱动
- warm-pool 触发受持续低负载条件约束
- 额外实例的回收发生在场景收尾阶段，因此结果中通常会看到末尾的 `physical_scale_down`

## 7. 预加载与分层驻留

### 7.1 三层本地驻留

当前本地层级为：

- `GPU`
- `HOST`
- `NVME`

`REMOTE` 是源仓库，不记为本地驻留层。

### 7.2 启动时预加载

当前完整栈支持三阶段预加载：

1. `REMOTE -> NVME`
2. `NVME -> HOST`
3. `HOST/NVME -> GPU warmup`

其目标是把“远端冷加载”尽量转换为“本地层间迁移”。

### 7.3 scale-up 预加载与新实例 warmup

当系统决定 scale-up 时，会先调用 `trigger_scaling_preload()` 为新实例准备 instance-scoped GPU warmup plan。

候选集合满足以下条件：

- 最近 `3600s` 内被访问过
- `hotness_score >= min_hotness_threshold`
- 工件已在 `HOST` 或 `NVME` 中存在本地路径
- 大小不超过本次 warmup 预算

当前实现的候选打分为：

- `score(a) = 10 * last_accessed_at(a) + 1000 * hotness(a) + value_per_byte(a)`

新实例创建后，warmup 顺序为：

- 先按 instance-scoped plan 中的顺序
- 再补齐所有 `hotness >= gpu_warmup_hotness` 的候选

这部分逻辑位于：

- `faaslora/experiment/experiment_stack.py`
- `scripts/run_all_experiments.py`

## 8. 资源协同

### 8.1 主线默认路径

当前主线启用的是 `ResourceCoordinator` 的稳定 admission 路径；其中 `effective_capacity_admission_enabled`（P2.5 风格有效容量准入）已经并入该主路径，并在当前默认配置中保持开启。该开关继续保留，便于做 on/off 对照实验。

LoRA load 请求的核心流程是：

1. 若目标 LoRA 已在 GPU resident 集合中，直接返回
2. 否则根据 tier 计算本地加载代价
3. 若当前 GPU 可用空间足够，立即 admission
4. 否则：
   - 在 `coordination_enabled = true` 时进入 defer/queue 路径
   - 在 `coordination_enabled = false` 时走强制驱逐后的直接尝试路径

当前 `available_mb` 的计算为：

- `available_mb = gpu_budget_mb - model_weights_mb - kv_mb - lora_mb - reserve`

其中：

- `kv_mb = active_tokens / 1000 * kv_per_1k_tokens_mb`
- `lora_mb` 为当前 GPU resident LoRA 总大小
- `reserve = gpu_budget_mb * lora_load_reserve_ratio`

当接入 `ResidencyManager` 时，可用空间直接读取 GPU tier 的真实剩余容量。

### 8.2 加载延迟模型

当前 runner 在结果中分别记录：

- `lora_io_ms`
- `contention_ms`
- `defer_ms`
- `vllm_ttft_ms`

总 TTFT 为：

- `ttft_ms = lora_io_ms + contention_ms + defer_ms + vllm_ttft_ms`

历史 `Qwen2.5-3B seq8_lora8` 基线首先证明了 serving 参数是关键瓶颈；在修复 GPU 全局显存观测、ResidencyManager 监控与 contention/defer 记账之后，`Qwen2.5-7B r300` 的 `P2.5 off` 控制组则暴露出了非零的 `contention_ms` 与 `defer_ms`。这说明这两部分指标不能再沿用早期“长期为 0”的解释，而应以修复后的口径为准。

### 8.3 warm pool

当触发 scale-down 时，协调器会在最近访问窗口内为每个 resident LoRA 计算：

- `score(a) = freq(a) / log1p(recency(a))`

然后：

- 驱逐最冷的 `n_to_evict`
- 保留其余 `w_t` 个 LoRA 作为 warm pool

其中 `w_t` 的当前动态规则为：

- `w_t = clip(round(\hat{h}_t (1 + \gamma)), w_min, w_max)`

`\hat{h}_t` 是批级 active adapter 数的 EWMA。

### 8.4 P2.5 有效容量准入

代码中已经实现了 `effective_capacity_admission_enabled` 分支，其核心量为：

- `pressure = max(mem_pressure, kv_pressure, load_pressure)`
- `effective_capacity_mb = available_mb * (1 - pressure)`
- `utility = recent_hotness * locality_factor`

只有当：

- `utility > pressure`
- 且 `size_mb <= effective_capacity_mb`

时才允许 admission。

这条路径的代码一直存在；在修复观测口径后的 `Qwen2.5-7B r300` A/B 中，`P2.5 on` 相对 `P2.5 off` 的关键变化为：

- `contention_events: 19 -> 0`
- `avg_defer_ms: 6552 -> 0`
- `avg_ttft_ms: 7053 -> 1747`
- `throughput_rps: 0.171 -> 0.228`

因此，当前仓库默认配置已将 `faaslora_full` 的 `effective_capacity_admission_enabled` 切到开启状态。需要注意的是：现有 `Qwen2.5-3B` 冻结结果文件仍来自这一切换前的配置，后续需要补一轮 `3B + P2.5 on` 复验来统一口径。

该 `3B + P2.5 on` 复验现已完成。相对旧的 `Qwen2.5-3B seq8_lora8` frozen baseline，新结果为：

- `avg_ttft_ms: 1494 -> 1563`
- `p95_ttft_ms: 4138 -> 4442`
- `p99_ttft_ms: 6670 -> 6839`
- `throughput_rps: 0.359 -> 0.363`
- `contention_events: 0 -> 0`
- `avg_defer_ms: 0 -> 0`

这说明：对当前 3B 路线，P2.5 主要是为了统一 3B/7B 默认口径，而不是性能增益来源。

在进一步放大到 `Qwen2.5-7B auto + 100 adapters + 1000 requests` 后，`P2.5 on` 的长跑结果也保持了同一结论：

- `avg_ttft_ms = 2381`
- `p95_ttft_ms = 14274`
- `p99_ttft_ms = 15968`
- `throughput_rps = 0.228`
- `avg_lora_io_ms = 119.6`
- `contention_events = 0`
- `avg_defer_ms = 0`

因此，对当前 7B 路线而言，P2.5 已经不再只是实验接口，而是当前默认验证配置的一部分。

## 9. 推理后端

### 9.1 默认后端：`vllm`

当前主线默认且已验证通过的后端是 `vllm`。其原因不是抽象偏好，而是当前实现已经把以下路径打通：

- 多 LoRA serving
- `shared / auto / dedicated` 三种实例模式
- 真正的物理 runtime 扩展
- 实验结果的可重复落盘与 live 监控

### 9.2 回退后端：`transformers`

`transformers` 仍作为兼容性回退路径保留，但当前定位仅限于：

- 小规模验证
- 环境兼容
- 非主线回退

## 10. 当前实验状态

### 10.1 已验证通过的代表性历史基线结果

当前最重要的一组历史冻结基线结果为：

- 文件：`results/experiment_results_full_vllm_auto_a500_r1000_c8_faaslora_full_seq8_lora8.json`
- 复现实验：`results/experiment_results_full_vllm_auto_a500_r1000_c8_faaslora_full_seq8_lora8_r3.json`
- 配置：`auto + 500 adapters + representative 1000 requests`
- serving 参数：`max_num_seqs = 8`、`max_loras = 8`、`runtime_concurrency_cap = 8`

该结果的关键指标为：

- `TTFT avg = 1409 ms`
- `TTFT P95 = 4068 ms`
- `TTFT P99 = 6023 ms`
- `E2E P99 = 22550 ms`
- `RPS = 0.364`
- `Hit = 94.6%`
- `scale_up_events = 1`
- `scale_down_events = 1`
- `cold_starts_after_scale_up = [2]`

对应的 `r3` 复现实验结果为：

- `TTFT avg = 1494 ms`
- `TTFT P95 = 4138 ms`
- `TTFT P99 = 6670 ms`
- `E2E P99 = 22642 ms`
- `RPS = 0.359`
- `Hit = 94.6%`

这说明这组 `3B` 历史冻结基线已经通过一次稳定性复验，现阶段没有必要继续追加更多相同配置的重复实验。

### 10.2 当前已得出的工程结论

- 当前 Qwen 家族主线已经从“能否跑通”进入“主配置固化与工程闭环”阶段，`14B r4000@0.85` 也已完成。
- `Qwen2.5-3B` 历史主线首先表明了 vLLM 有效并发参数是首要瓶颈。
- 在修复显存观测与 contention/defer 记账后，`Qwen2.5-7B` 的高压阶段又表明 P2.5 有效容量准入可以显著改善 admission / defer。
- `Qwen2.5-3B + P2.5 on` 复验表明，3B 本身并不存在显著的 contention/defer，因此 P2.5 在 3B 上不是核心收益项。
- `Qwen2.5-7B r1000 + P2.5 on` 长跑进一步表明，这一收益不是短测偶然波动，而能稳定延续到更长工作负载。
- 将 `max_num_seqs / max_loras` 从保守 preset 提升到 `8 / 8` 后，TTFT 与 tail latency 显著改善。
- `shared / dedicated / full-trace` 相关接口均保留，但不作为当前主线推进对象。
- `effective_capacity_admission_enabled` 的 on/off 开关继续保留，但当前仓库默认值已经切到 `on`。
- 当前扩展主线已从 `Mistral-7B-Instruct-v0.3` 推进到 `Mistral-Nemo-Instruct-2407`，论文主线口径统一为 `PEFT+finetune + 500 adapters`。
- `Mistral-7B-Instruct-v0.3 + PEFT+finetune + 500 adapters + representative r1000` 已稳定完成；下一步应直接推进 `Mistral-Nemo-Instruct-2407 + TP=2 + PEFT+finetune + 500 adapters + representative r1000`。
- `scripts/generate_lora_adapters.py` 已进一步收敛到当前主线配置：
  - 默认值跟随 `profile_selection + model_profiles + workload_profiles`
  - `PEFT+finetune` 生成路径改为单次加载 base model 后循环生成多个 adapters
  - 顶层 `model / hardware / workload` 继续保留，但主要作为兼容回退层
- 基础测试也已同步更新：当前 `smoke + integration` 会校验 active profile 解析、生成器默认值和 batch PEFT 路径，而不再只盯顶层 fallback 配置

## 11. 当前主线 TODO

### A. 主配置固化

1. 已完成：将 `auto500 + representative1000 + seq8_lora8` 固化为 `3B` 历史冻结基线，并完成稳定性复验。
2. 已完成：用默认入口再做一次复验，确认后续复现不依赖长串环境变量覆盖。
3. 已完成：补一轮 `Qwen2.5-3B auto500 + representative1000 + P2.5 on` 复验，统一默认配置与结果口径。

### B. 工程闭环

4. 修 CLI / packaging 断裂。
5. 已完成：补齐稳定环境下可执行的基础测试，并将入口改为真正可被 `unittest discover` 执行的 `smoke + integration`。
6. 已完成：同步 README / GUIDE / docs 与当前实现和结果。

### C. 扩展主线

7. 已完成：`Qwen2.5-7B-Instruct` 主线与 `TP=2` 对照。
8. 已完成：`Qwen2.5-14B-Instruct` 的 `r1000` 收敛与 `r4000@0.85` 长跑冻结。
9. 已完成：第二家族小档位 `Mistral-7B-Instruct-v0.3 + PEFT+finetune + 500 adapters + representative r1000`。
10. 当前优先：推进 `Mistral-Nemo-Instruct-2407 + TP=2 + PEFT+finetune + 500 adapters + representative r1000`。
11. 已完成：将 `model / dataset / workload` 三类主线切换入口收敛到 `experiments.yaml` 的 `profile_selection`，减少手改散落字段。
12. 已确认：`mistral_nemo_12b_tp2_main` 固定为论文主线 `500 adapters`；`mistral_nemo_12b_tp2_bringup100_main` 仅保留给显式 bring-up / 排障。

## 12. 已知边界

- 当前系统是单节点多 GPU 原型，不是完整多节点云平台。
- ShareGPT 当前作为 prompt pool，而不是 full conversation replay。
- 当前主线统一使用 representative trace replay。
- 当前扩展主线已从 Qwen 切到 Mistral；当前第二家族 7B 档已完成，下一步固定为 `mistralai/Mistral-Nemo-Instruct-2407`。
- Gemma 暂不进入当前配置与实验轮次，但继续保留在计划列表中。
- 本机现有 `Qwen2.5-3B-Instruct` 目录继续保留，不删除。
- `shared` / `dedicated` / `28185 full trace` 的接口继续保留，但不作为当前主线默认配置。
- `effective_capacity_admission_enabled` 的 on/off 接口继续保留，但当前默认配置已切到 `on`。
- `shared` 模式不是强隔离函数进程模型，而是共享 runtime + shared execution slot 的实现方式。
