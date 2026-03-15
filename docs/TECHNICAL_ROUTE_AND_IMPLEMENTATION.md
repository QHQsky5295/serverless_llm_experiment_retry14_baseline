# FaaSLoRA 技术路线与实现说明

本文档是当前仓库的**权威技术说明文档**。内容对齐当前实现，而不是历史设计稿。若 README、实验指南与本文冲突，以本文为准。

## 1. 系统定位

FaaSLoRA 是一个面向多 LoRA 大模型推理的单节点 Serverless 研究原型。系统运行在单台双 GPU 服务器上，通过共享 backbone、工件感知请求放置、分层驻留和资源协同控制，研究以下问题：

1. 请求被放置到现有函数实例时，目标实例是否已经命中所需 LoRA 工件。
2. 系统扩展新的物理 GPU 执行实例时，新实例是否能够带着热点工件快速进入服务。
3. 在 LoRA 工件加载、KV 缓存和批推理共享 GPU 资源的条件下，如何控制争用带来的 TTFT 和尾延迟代价。

当前系统不是生产级云平台，而是一个**真实可运行的单节点双 GPU Serverless LoRA 原型**。它已经可以稳定运行 `shared / auto / dedicated` 三种模式，并支持 `100 / 300 / 500 / 1000` adapters 的实验矩阵。

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

当前验证通过的主线配置是：

- `auto`
- `500` adapters
- `1000` representative requests
- `time_scale_factor = 0.02`
- `max_model_len = 2048`
- `max_num_seqs = 8`
- `max_loras = 8`
- `runtime_concurrency_cap = 8`
- `max_num_batched_tokens = 4096`

### 4.2 路由与实例池

职责：

- 管理 `shared / auto / dedicated` 三种实例模式
- 在请求到达时选择目标实例
- 维护逻辑实例与物理 runtime 的对应关系

关键实现：

- `faaslora/experiment/instance_pool.py`
- `scripts/run_all_experiments.py`

当前默认路由策略是 `adapter_affinity`。其亲和度函数为：

- `A_i(\ell) = 3`，若 `\ell` 已在实例 `i` 的 GPU resident 集合中
- `A_i(\ell) = 2`，若 `\ell` 在 HOST cache 集合中
- `A_i(\ell) = 1`，若 `\ell` 在 NVMe cache 集合中
- `A_i(\ell) = 0`，否则

路由器先找出最大亲和度：

- `A^*(\ell) = max_{j in I_t} A_j(\ell)`

然后仅在 `A_i(\ell) = A^*(\ell)` 的候选集中按以下元组取最小值：

- `(q_i, n_i, u_i, last_selected_at_i, created_at_i)`

也就是说，当前实现不是全局最优放置器，而是**缓存亲和优先、再按负载做稳定 tie-break** 的启发式。

## 5. 实例模式

### 5.1 `shared`

- 所有逻辑 slot 共享同一个物理 runtime。
- scale-up 增加的是逻辑 slot，因此事件类型是 `logical_scale_up`。
- 不会新建第二个物理 vLLM runtime。

### 5.2 `auto`

- 优先尝试新增独立物理 runtime。
- 若物理扩容失败且模式不是 `dedicated`，再回退到 `shared` 路径。
- 在当前双 3090 环境下，主线运行时最多扩到两个物理 runtime。

### 5.3 `dedicated`

- 仅允许新增独立物理 runtime。
- 不回退到 `shared` slot。
- 配置接口仍保留，但当前不作为主线推进对象。

## 6. 扩缩容机制

### 6.1 决策时机

主实验路径在 `ScenarioRunner.run()` 中采用**按批观测、批后决策**：

- 每批大小等于 `scale_decision_interval`
- 当前批完成后，系统观测该批期间的峰值 backlog、峰值 active requests 和峰值 busy ratio
- 然后用这组观测构造 `ScalingMetrics`

因此，当前扩缩容不是“每个请求到达时立即决策”，而是**批级离散决策**。

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
- 当前扩展主线已切到 `Mistral-7B-Instruct-v0.3`，论文主线口径统一为 `PEFT+finetune + 500 adapters + representative r1000`。

## 11. 当前主线 TODO

### A. 主配置固化

1. 已完成：将 `auto500 + representative1000 + seq8_lora8` 固化为 `3B` 历史冻结基线，并完成稳定性复验。
2. 已完成：用默认入口再做一次复验，确认后续复现不依赖长串环境变量覆盖。
3. 已完成：补一轮 `Qwen2.5-3B auto500 + representative1000 + P2.5 on` 复验，统一默认配置与结果口径。

### B. 工程闭环

4. 修 CLI / packaging 断裂。
5. 补稳定环境下可执行的基础测试。
6. 同步 README / GUIDE / docs 与当前实现和结果。

### C. 扩展主线

7. 已开始：推进 `Qwen2.5-7B-Instruct`。
8. 已完成：`Qwen2.5-7B auto + 100 adapters + 1000 requests + P2.5 on`。
9. 当前优先：继续 Qwen 家族，推进 `Qwen2.5-14B-Instruct`（13B+ 级）bring-up，并把工作负载扩到 `representative 4000 requests`；必要时再判断是否需要新的 P2.5 A/B。
10. Qwen `14B` 稳定后，下一个家族切到 Mistral；`OPT` 已确认在当前 `vLLM 0.10.2 + LoRA` 环境下不支持。
11. 已完成：将 `model / dataset / workload` 三类主线切换入口收敛到 `experiments.yaml` 的 `profile_selection`，减少手改散落字段。

## 12. 已知边界

- 当前系统是单节点双 GPU 原型，不是完整多节点云平台。
- ShareGPT 当前作为 prompt pool，而不是 full conversation replay。
- 当前主线统一使用 representative trace replay。
- 当前扩展主线先完成 `Qwen2.5-14B-Instruct`；其后下一个家族固定为 Mistral，并按 `mistralai/Mistral-7B-Instruct-v0.3 -> mistralai/Mistral-Nemo-Instruct-2407` 推进。
- Gemma 暂不进入当前配置与实验轮次，但继续保留在计划列表中。
- 本机现有 `Qwen2.5-3B-Instruct` 目录继续保留，不删除。
- `shared` / `dedicated` / `28185 full trace` 的接口继续保留，但不作为当前主线默认配置。
- `effective_capacity_admission_enabled` 的 on/off 接口继续保留，但当前默认配置已切到 `on`。
- `shared` 模式不是强隔离函数进程模型，而是共享 runtime + shared execution slot 的实现方式。
