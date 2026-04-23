# PrimeLoRA 论文实验 TODO 与图表字段清单

本文档用于把 PrimeLoRA/FaaSLoRA 的正式论文实验从“临时跑数”整理成可复现、可取数、可画图、可审计的工程 checklist。后续所有正式实验图、表、消融和超参数实验，都应先对照本文档确认：

- 实验是否服务论文问题，而不是为了某个 baseline 临时改题。
- 指标是否能在结果 JSON 中真实观测。
- 横向图是否只使用所有系统都能输出的统一字段。
- FaaSLoRA 机制图是否只使用 FaaSLoRA 及其消融/超参变体都能输出的字段。
- run tag、结果目录和 manifest 是否能让未来无需重跑即可定位原始产物。

## 0. 硬性约束

### 0.1 不使用不可观测机制指标

正式论文 TODO、正式图表和主实验 checklist 只允许使用结果 JSON 中真实存在的字段。不能观测、不能跨系统统一、或者执行时只能填 `null` 的机制指标，不进入正式论文图。

允许的例外只有一种：调试审计文档可以记录 `null` 来说明某个系统缺少某项观测，但论文图、论文表和本文档中的正式 TODO 不使用这些字段。

因此：

- 横向对比图只用 FaaSLoRA、SGLang、ServerlessLLM、vLLM 都能输出的字段。
- FaaSLoRA 内部机制图只用 FaaSLoRA full、消融变体、超参变体都能输出的字段。
- 如果某个指标只能由 FaaSLoRA 输出，而 baseline 无法真实输出，它只能用于 FaaSLoRA 机制分析，不能用于横向公平对比图。
- 如果某个指标当前不存在，但理论上可通过所有系统的统一 wrapper 真实补齐，则先列入“工程补齐项”；补齐并通过 smoke test 后才进入正式图。
- 如果某个指标需要 baseline 内部事件、无法通过统一 replay wrapper 真实观测，则直接排除，不用替代值、估计值或硬编码值。

### 0.2 主指标口径

正式实验使用 `metric_schema_version=e2e_v3`。主延迟、成本和 CE 口径固定如下。

```text
TTFT_e2e = request scheduled arrival / replay-observed arrival
           -> first output token/chunk observed by client/replay

E2E_e2e  = request scheduled arrival / replay-observed arrival
           -> full response completion observed by client/replay

Cost/req = total monetary cost / completed requests

CE       = 1 / (avg E2E_e2e in seconds * Cost/req)
```

`Cost/1M tokens` 只作为补充审计指标，用来说明 token workload 没有异常偏移；主成本指标仍是 `Cost/req`，主性价比指标仍是 `CE`。

### 0.3 成本模型口径

正式 monetary cost 使用 cloud-style active/idle differential billing：

```text
Cost_serverful =
  P_gpu * GPUSeconds_lifecycle

Cost_serverless =
  P_gpu * (GPUSeconds_startup + GPUSeconds_active
           + alpha_idle * GPUSeconds_idle_ready)
  + C_invocation * N_completed
```

当前默认 `alpha_idle = 0.2380952381`，对应 Alibaba Function Compute Tesla GPU idle/active CU factor `0.5 / 2.1`。`InfraCost` 和 `InfraCE` 保留为 flat GPU-second 审计，不作为主 CE 结论。

## 1. 正式 workload 设置

### 1.1 主横向对比负载

正式主表建议采用：

- `requests = 4000`
- `adapter pool size = 500`
- `100% LoRA requests`
- `Zipf exponent = 1.0`
- `active hot set cap = 48`
- `hotset rotation interval = 500 requests`
- `time scale factor = 8.0`
- `seed = 42`

500 请求只作为 bring-up/debug，不写主结论。它用于快速检查脚本、环境、指标 schema、端口、tmux/session、GPU 是否健康；如果 500 请求结果趋势和 4000 请求不同，应以 4000 请求正式结果为准。

### 1.2 为什么主负载用 4000 请求

PrimeLoRA 的核心机制是热度驱动的 adapter readiness：路由、扩容、预加载、GPU/host/NVMe residency 都需要在一段足够长的 trace 中体现“学习、迁移、稳定、再迁移”的过程。500 请求可以发现系统 bug，但太短，容易出现以下偏差：

- 热点阶段太少，预加载策略尚未进入稳定区间。
- scale-out 和 scale-down 事件数量不足，不能充分体现 serverless lifecycle cost。
- adapter pool 为 500 时，500 请求几乎是“一请求一 adapter universe”的短跑，不适合作为正式主结论。
- P95/P99 更容易被少量冷启动、进程抖动、session closing 等偶然事件支配。

4000 请求能支持 8 个热点阶段，每个阶段 500 请求；既能保留真实动态性，又不会像每 100 请求换热点那样让热度学习一直处于急转弯状态。

### 1.3 Adapter 采样与热度

正式主表 adapter 设置：

- `adapter pool size = 500`。
- `active hot set cap = 48`。
- `Zipf exponent = 1.0`。
- 每个请求绑定一个 LoRA。
- 同一轮横向对比共用同一 shared trace 和 adapter subset。

设计理由：

- 500 adapters 对应 many-LoRA、多租户、长尾访问场景，能体现 adapter--replica mismatch。
- active hot set 约 48，与当前系统 `active_adapter_cap: 48` 对齐，避免让 GPU cache 承担不现实的全量 adapter 常驻压力。
- Zipf 1.0 能表达“少量热点 + 大量长尾”的多租户访问模式；均匀采样不适合作为主实验，因为它会削弱热度学习和预加载机制的意义。
- 所有 baseline 必须使用同一 adapter subset，不能让某个系统看到更简单或更难的 adapter universe。

### 1.4 热点迁移

正式主表使用：

```text
hotset_rotation_requests = 500
num_hotset_phases        = 4000 / 500 = 8
```

可选附加稳健性实验：

```text
hotset_rotation_requests in {250, 500, 1000}
```

默认不用每 100 请求换热点。每 100 请求换热点适合作为压力测试或最坏情况章节，因为它会让系统频繁面对尚未学稳的热点集合，不适合作为主横向对比。

### 1.5 LoRA pool size 消融

LoRA pool size 消融用于回答“adapter universe 变大时，PrimeLoRA 是否仍能维持 readiness 和 cost efficiency”。

建议设置：

| Adapter pool | Active hot cap | Requests | Zipf | Rotation | Time scale |
|---:|---:|---:|---:|---:|---:|
| 100 | 16 | 4000 | 1.0 | 500 | 8.0 |
| 200 | 24 | 4000 | 1.0 | 500 | 8.0 |
| 300 | 32 | 4000 | 1.0 | 500 | 8.0 |
| 400 | 40 | 4000 | 1.0 | 500 | 8.0 |
| 500 | 48 | 4000 | 1.0 | 500 | 8.0 |

注意：消融时保持同一 trace seed、同一 prompt source、同一 Zipf 规律，只改变 adapter universe 和 active hot cap。不要同时改 time scale、prompt length 或 GPU budget。

### 1.6 负载强度分层

主横向对比使用 `time scale factor = 8.0`。压力测试单独成章，建议：

```text
time_scale_factor in {8.0, 6.0, 4.0, 2.0, 1.0}
```

解释：

- `s8`：主论文对比，避免把所有系统推入不可区分的拥塞区。
- `s6/s4`：中高压，观察 FaaSLoRA 机制收益是否随压力增强。
- `s2/s1`：压力测试，不作为主表 headline；用于展示系统极限、SLO goodput 和 effective service rate。

## 2. 字段可用性总表

### 2.1 跨系统正式字段

以下字段已经由 FaaSLoRA 结果 JSON 和 baseline summary JSON 统一输出。横向对比图优先使用这些字段。

| 指标 | `scenario_summaries` 字段 | `comparison_table` 别名 | 用途 |
|---|---|---|---|
| completed requests | `completed_requests` | `Done` / `completed` | 成功率、过滤失败轮 |
| total requests | `total_requests` | `total` | 成功率、manifest |
| Avg TTFT_e2e | `avg_overall_ttft_ms` | `TTFT_e2e_avg_ms` | 主延迟 |
| P95 TTFT_e2e | `p95_overall_ttft_ms` | `TTFT_e2e_P95_ms` | 尾延迟 |
| Avg TTFT_service | `avg_service_ttft_ms` | `TTFT_service_avg_ms` | 服务路径延迟 |
| P95 TTFT_service | `p95_service_ttft_ms` | `TTFT_service_P95_ms` | 服务路径尾延迟 |
| Avg E2E_e2e | `avg_overall_e2e_ms` | `E2E_e2e_avg_ms` | 主完成延迟 |
| P95 E2E_e2e | `p95_overall_e2e_ms` | `E2E_P95_ms` | 主完成尾延迟 |
| TPOT | `avg_tpot_ms` | `TPOT_avg_ms` | decode 速度 |
| Throughput RPS | `throughput_rps` | `Throughput_RPS` | 请求吞吐 |
| Throughput TOK/s | `throughput_tok_per_s` | `Throughput_TOKPS` | token 吞吐 |
| SLO attainment | `slo_attainment` | `SLO_attainment` | SLO 达成率 |
| SLO goodput RPS | `slo_goodput_rps` | `SLO_goodput_RPS` | 有效请求吞吐 |
| SLO goodput TOK/s | `slo_goodput_tok_per_s` | `SLO_goodput_TOKPS` | 有效 token 吞吐 |
| Cost/req | `monetary_cost_per_request_usd` | `Monetary_cost_per_request_usd` / `avg_cost_USD` | 主成本 |
| Total monetary cost | `monetary_cost_total_usd` | `Monetary_cost_total_usd` | 生命周期成本 |
| CE | `monetary_ce` / `ce` | `Monetary_CE` / `CE` | 主性价比 |
| Cost/1M total tokens | `cost_per_1m_total_tokens_usd` | `Monetary_cost_per_1M_total_tokens_usd` | token 成本审计 |
| Total tokens | `total_tokens` | `total_tokens` | workload 审计 |
| Input tokens | `total_input_tokens` | `total_input_tokens` | workload 审计 |
| Output tokens | `total_output_tokens` | `total_output_tokens` | workload 审计 |
| GPU seconds total | `infra_gpu_seconds_total` | `Infra_GPU_seconds_total` | 资源占用 |
| Startup GPU seconds | `infra_startup_gpu_seconds` | `Infra_GPU_seconds_startup` | serverless 启动成本 |
| Active GPU seconds | `infra_active_gpu_seconds` | `Infra_GPU_seconds_active` | 执行成本 |
| Idle-ready GPU seconds | `infra_idle_ready_gpu_seconds` | `Infra_GPU_seconds_idle_ready` | 保活成本 |
| Active GPU ratio | `infra_active_gpu_ratio` | `Infra_active_GPU_ratio` | 资源效率 |
| Avg allocated GPUs | `infra_avg_allocated_gpus` | `Infra_avg_allocated_GPUs` | 资源占用 |
| Max allocated GPUs | `infra_max_allocated_gpus` | `Infra_max_allocated_GPUs` | GPU budget 审计 |
| Completed req/GPU-s | `completed_requests_per_gpu_second` | `Completed_requests_per_GPU_second` | 资源效率 |
| Goodput tok/GPU-s | `goodput_tokens_per_gpu_second` | `Goodput_tokens_per_GPU_second` | 有效资源效率 |
| Dispatch wait avg | `avg_dispatch_admission_wait_ms` | `Dispatch_admission_wait_avg_ms` | replay/调度等待 |
| Dispatch wait P95 | `p95_dispatch_admission_wait_ms` | `Dispatch_admission_wait_P95_ms` | replay/调度尾等待 |

横向正式图默认不使用 baseline 不能输出的内部 LoRA readiness、RPC、runtime shell、GPU SM utilization 字段。

### 2.2 FaaSLoRA 机制字段

以下字段可用于 FaaSLoRA full 与消融/超参变体之间的内部机制图。不要拿这些字段直接和无法观测的 baseline 机制字段硬比。

| 指标 | JSON 字段 | 支撑的贡献 |
|---|---|---|
| GPU hit rate | `gpu_hit_rate` / `GPU_hit_rate` | 贡献 1、贡献 2 |
| Cache hit rate | `cache_hit_rate` | 贡献 2 |
| LoRA I/O avg | `avg_lora_io_ms` / `LoRA_IO_avg_ms` | 贡献 1、贡献 2 |
| GPU-ready TTFT avg/P95 | `avg_gpu_ready_ttft_ms`, `p95_gpu_ready_ttft_ms` | 预加载后首 token 路径 |
| Scale-up affected TTFT avg/P95 | `avg_scaleup_affected_ttft_ms`, `p95_scaleup_affected_ttft_ms` | 扩容阶段 readiness |
| Scale-up first-service GPU hit | `scaleup_first_service_gpu_hit_rate` | 扩容前预加载 |
| Scale-up plan match | `scaleup_first_service_planned_match_rate` | 命中感知扩容准备 |
| Runtime TTFT avg/P95 | `avg_runtime_ttft_ms`, `p95_runtime_ttft_ms` | 后端执行路径 |
| Pre-runtime shell avg/P95 | `avg_pre_runtime_service_shell_ms`, `p95_pre_runtime_service_shell_ms` | admission 后前置壳成本 |
| Service residual avg/P95 | `avg_service_path_residual_ms`, `p95_service_path_residual_ms` | service path 闭合审计 |
| RPC overhead avg/P95 | `avg_parent_rpc_overhead_ms`, `p95_parent_rpc_overhead_ms` | subprocess RPC 审计 |
| Contention penalty | `avg_contention_ms` | 贡献 3 |
| Defer delay | `avg_defer_ms` | 贡献 3 |
| Cold start avg/P95 | `avg_cold_start_latency_ms`, `p95_cold_start_latency_ms` | serverless 扩容成本 |
| Scale events | `scale_up_events`, `scale_down_events` | 扩缩容行为 |

### 2.3 不进入正式 TODO 的字段

以下字段当前不作为正式论文图指标：

- `adapter_replica_mismatch_rate`：当前没有跨系统统一输出。
- `GPU_SM_util_avg`、`GPU_mem_bandwidth_avg`：当前没有统一 sampler 和 JSON schema。
- baseline 内部 adapter prepare phase：SGLang/vLLM/OpenAI server 不暴露同等事件，不能用推测值替代。
- baseline 内部 LoRA cache eviction/migration：不同系统内部语义不同，无法统一定义。
- first non-empty token：当前口径已统一为 first output token/chunk，不再改成 non-empty，以避免三系统解析不一致。

如果未来需要资源利用率图，应先实现一个独立的统一 GPU sampler，把 `nvidia-smi` 采样写入所有系统同格式 JSON，并通过 smoke test 后再加入正式 TODO。

## 3. 论文图与实验 checklist

### 3.1 引言图：Cost-Latency-CE teaser

目标：开篇说明 PrimeLoRA 位于更优 cost-latency tradeoff：E2E 可以接近 serverful，成本显著低于 serverful，CE 更高。

图形建议：

- 一张 Pareto scatter。
- X 轴：`Cost/req`。
- Y 轴：`Avg E2E_e2e`。
- 点大小或标签：`CE`。
- 误差棒：`P95 E2E_e2e`。

系统：

- PrimeLoRA/FaaSLoRA full。
- SGLang DP4/TP1。
- vLLM DP4/TP1。
- ServerlessLLM。

字段：

| 图中元素 | 必需字段 | 状态 |
|---|---|---|
| X: Cost/req | `monetary_cost_per_request_usd` | 已有 |
| Y: Avg E2E | `avg_overall_e2e_ms` | 已有 |
| Error: P95 E2E | `p95_overall_e2e_ms` | 已有 |
| Label: CE | `monetary_ce` | 已有 |
| Sanity | `completed_requests`, `total_requests`, `total_tokens` | 已有 |

run family：

```text
intro_teaser
```

run tag 示例：

```text
llama2_7b_r4000_a500_z1p0_hot48_rot500_s8_seed42_intro_teaser_faaslora_full_v1
llama2_7b_r4000_a500_z1p0_hot48_rot500_s8_seed42_intro_teaser_sglang_dp4_tp1_v1
llama2_7b_r4000_a500_z1p0_hot48_rot500_s8_seed42_intro_teaser_vllm_dp4_tp1_v1
llama2_7b_r4000_a500_z1p0_hot48_rot500_s8_seed42_intro_teaser_sllm_vllm_v1
```

### 3.2 Workload characterization

目标：证明实验负载确实是动态 many-LoRA serverless workload，而不是普通单模型 serving。

图形建议：

- Trace arrival rate over time。
- Adapter popularity distribution。
- Hotset phase timeline。
- Prompt/input/output token distribution。

字段来源：

- shared trace artifact。
- adapter subset artifact。
- result JSON metadata。

必需字段：

| 内容 | 字段或来源 | 状态 |
|---|---|---|
| requests | `metadata.total_requests` / shared trace length | 已有 |
| adapter pool | `metadata.selected_num_adapters` / subset length | 已有 |
| time scale | `metadata.effective_time_scale_factor` 或 run tag | 已有 |
| token totals | `total_input_tokens`, `total_output_tokens`, `total_tokens` | 已有 |
| completed sanity | `completed_requests`, `total_requests` | 已有 |

工程要求：

- 每个正式 shared artifact 放入 `results/paper_experiments/00_shared_artifacts/<base_run_tag>/`。
- 保存 trace、adapter subset、生成命令和 seed。
- 如果 trace artifact 没有显式记录 `zipf_exponent`、`hotset_rotation_requests`，在 run manifest 中补记录；不要在论文中从结果反推。

### 3.3 Motivation 1：Adapter-replica mismatch 的存在性

目标：证明 dynamic many-LoRA serverless 场景中，adapter readiness 会进入首 token 前关键路径。

建议只用 FaaSLoRA diagnostic/variant，不强行要求 baseline 输出内部 mismatch 事件。

图形建议：

- 分组柱：`Avg TTFT_e2e`、`P95 TTFT_e2e`、`Avg LoRA I/O`、`GPU hit rate`。
- 对比对象：无预加载/弱预加载 variant vs full PrimeLoRA。

字段：

| 内容 | 字段 | 状态 |
|---|---|---|
| Avg TTFT | `avg_overall_ttft_ms` | 已有 |
| P95 TTFT | `p95_overall_ttft_ms` | 已有 |
| LoRA I/O | `avg_lora_io_ms` | 已有 |
| GPU hit | `gpu_hit_rate` | 已有 |
| Scale-up affected TTFT | `avg_scaleup_affected_ttft_ms`, `p95_scaleup_affected_ttft_ms` | 已有 |

run family：

```text
motivation_mismatch
```

可运行变体：

- `faaslora_full`
- `faaslora_no_preload`
- `faaslora_no_hit_aware_placement`

### 3.4 Motivation 2：Serverful low latency but high lifecycle cost

目标：说明 serverful 系统在 latency 上通常强，但由于常驻 GPU 生命周期成本高，Cost/req 和 CE 不一定优。

图形建议：

- 双轴或两个子图。
- 子图 A：Avg/P95 E2E。
- 子图 B：Cost/req、CE。

系统：

- SGLang DP4/TP1。
- vLLM DP4/TP1。
- PrimeLoRA full。

字段：

| 内容 | 字段 | 状态 |
|---|---|---|
| Avg E2E | `avg_overall_e2e_ms` | 已有 |
| P95 E2E | `p95_overall_e2e_ms` | 已有 |
| Cost/req | `monetary_cost_per_request_usd` | 已有 |
| CE | `monetary_ce` | 已有 |
| GPU lifecycle | `infra_gpu_seconds_total`, `monetary_equivalent_gpu_seconds` | 已有 |
| Active/idle split | `infra_active_gpu_seconds`, `infra_idle_ready_gpu_seconds` | 已有 |

### 3.5 Motivation 3：General serverless LLM 不等于 many-LoRA serverless LLM

目标：证明 ServerlessLLM 擅长模型级启动/迁移，但在 many-LoRA adapter readiness 上不是专门优化。

图形建议：

- Grouped bar：Avg/P95 TTFT_e2e、Avg/P95 E2E_e2e、Cost/req、CE。
- 附加表：SLO attainment、SLO goodput。

系统：

- ServerlessLLM。
- PrimeLoRA full。

字段：

| 内容 | 字段 | 状态 |
|---|---|---|
| Avg TTFT | `avg_overall_ttft_ms` | 已有 |
| P95 TTFT | `p95_overall_ttft_ms` | 已有 |
| Avg E2E | `avg_overall_e2e_ms` | 已有 |
| P95 E2E | `p95_overall_e2e_ms` | 已有 |
| Cost/req | `monetary_cost_per_request_usd` | 已有 |
| CE | `monetary_ce` | 已有 |
| SLO | `slo_attainment` | 已有 |
| SLO goodput | `slo_goodput_rps`, `slo_goodput_tok_per_s` | 已有 |

### 3.6 Main comparison

目标：正式横向对比 PrimeLoRA、SGLang、vLLM、ServerlessLLM。

主表字段：

| 列 | 字段 | 状态 |
|---|---|---|
| Avg TTFT | `avg_overall_ttft_ms` | 已有 |
| P95 TTFT | `p95_overall_ttft_ms` | 已有 |
| Avg E2E | `avg_overall_e2e_ms` | 已有 |
| P95 E2E | `p95_overall_e2e_ms` | 已有 |
| TPOT | `avg_tpot_ms` | 已有 |
| Throughput TOK/s | `throughput_tok_per_s` | 已有 |
| Cost/req | `monetary_cost_per_request_usd` | 已有 |
| CE | `monetary_ce` | 已有 |
| SLO | `slo_attainment` | 已有 |

主图建议：

- 图 A：Avg/P95 TTFT grouped bar。
- 图 B：Avg/P95 E2E grouped bar。
- 图 C：Cost/req grouped bar。
- 图 D：CE grouped bar。

注意：延迟图每次都同时放 Avg 和 P95，方便同时看平均体验和尾延迟。CE 不放 P95，因为当前主 CE 定义是基于平均 E2E 和 Cost/req 的 workload-level 指标。

run family：

```text
main_comparison
```

### 3.7 Contribution ablation

目标：分别消融三大贡献，证明收益来自系统机制，而不是 workload 或成本公式。

变体：

- `full`：完整 PrimeLoRA。
- `no_c1`：关闭 LoRA Hit-Aware Request Placement and Scaling。
- `no_c2`：关闭 Hierarchical LoRA Adapter Residency and Dynamic Migration。
- `no_c3`：关闭 Coordinated Resource Control for LoRA Loading and Inference。

图形建议：

- 图 A：Avg/P95 TTFT_e2e。
- 图 B：Avg/P95 E2E_e2e。
- 图 C：GPU hit rate、LoRA I/O、Scale-up first-service GPU hit。
- 图 D：Cost/req、CE。

字段：

| 内容 | 字段 | 状态 |
|---|---|---|
| Avg/P95 TTFT | `avg_overall_ttft_ms`, `p95_overall_ttft_ms` | 已有 |
| Avg/P95 E2E | `avg_overall_e2e_ms`, `p95_overall_e2e_ms` | 已有 |
| GPU hit | `gpu_hit_rate` | 已有 |
| LoRA I/O | `avg_lora_io_ms` | 已有 |
| Scale-up first-service hit | `scaleup_first_service_gpu_hit_rate` | 已有 |
| Cost/req | `monetary_cost_per_request_usd` | 已有 |
| CE | `monetary_ce` | 已有 |

### 3.8 Hyperparameter experiments

只选两个最关键、最能对应系统贡献的超参数，避免超参章节变成调参流水账。

#### H1. Active hot set cap

目的：验证 GPU memory budget 下，hot adapter coverage 与 KV/cache/batching 之间的平衡。

设置：

```text
active_adapter_cap in {16, 24, 32, 40, 48, 64}
requests = 4000
adapter pool = 500
Zipf = 1.0
rotation = 500
time scale = 8.0
```

字段：

| 内容 | 字段 | 状态 |
|---|---|---|
| Avg/P95 TTFT | `avg_overall_ttft_ms`, `p95_overall_ttft_ms` | 已有 |
| Avg/P95 E2E | `avg_overall_e2e_ms`, `p95_overall_e2e_ms` | 已有 |
| GPU hit | `gpu_hit_rate` | 已有 |
| LoRA I/O | `avg_lora_io_ms` | 已有 |
| Active GPU ratio | `infra_active_gpu_ratio` | 已有 |
| CE | `monetary_ce` | 已有 |

#### H2. Idle retention / scale-down policy

目的：验证 serverless 按需计费和 warm readiness 的折中。这个参数必须来自自适应策略或明确配置，不允许写死成只适合 Llama-2 7B 的秒数。

设置建议：

```text
serverless_idle_retention_s in {60, 120, 300, 600, adaptive}
```

字段：

| 内容 | 字段 | 状态 |
|---|---|---|
| Avg/P95 TTFT | `avg_overall_ttft_ms`, `p95_overall_ttft_ms` | 已有 |
| Avg/P95 E2E | `avg_overall_e2e_ms`, `p95_overall_e2e_ms` | 已有 |
| Cost/req | `monetary_cost_per_request_usd` | 已有 |
| CE | `monetary_ce` | 已有 |
| Idle-ready GPU seconds | `infra_idle_ready_gpu_seconds` | 已有 |
| Startup GPU seconds | `infra_startup_gpu_seconds` | 已有 |
| Scale events | `scale_up_events`, `scale_down_events` | 已有 |

### 3.9 Scalability experiments

#### S1. Adapter pool size

使用第 1.5 节设置。图形：

- Avg/P95 TTFT vs adapter pool。
- Cost/req vs adapter pool。
- CE vs adapter pool。
- GPU hit rate vs adapter pool。

#### S2. Load intensity

使用第 1.6 节设置。图形：

- SLO attainment vs time scale。
- SLO goodput vs time scale。
- Avg/P95 E2E vs time scale。
- Cost/req and CE vs time scale。

### 3.10 Resource and cost efficiency

目标：证明 PrimeLoRA 的优势不只来自快，也来自更合理的 lifecycle GPU usage。

图形建议：

- Stacked bar：startup GPU-seconds、active GPU-seconds、idle-ready GPU-seconds。
- Grouped bar：completed req/GPU-s、goodput tok/GPU-s。
- Grouped bar：Cost/req、CE。

字段：

| 内容 | 字段 | 状态 |
|---|---|---|
| Startup GPU-s | `infra_startup_gpu_seconds` | 已有 |
| Active GPU-s | `infra_active_gpu_seconds` | 已有 |
| Idle-ready GPU-s | `infra_idle_ready_gpu_seconds` | 已有 |
| Total GPU-s | `infra_gpu_seconds_total` | 已有 |
| Completed req/GPU-s | `completed_requests_per_gpu_second` | 已有 |
| Goodput tok/GPU-s | `goodput_tokens_per_gpu_second` | 已有 |
| Cost/req | `monetary_cost_per_request_usd` | 已有 |
| CE | `monetary_ce` | 已有 |

### 3.11 Latency phase breakdown

目标：用首 token 前分段时间轴解释 PrimeLoRA 的机制，不做跨系统不可观测内部对比。

PrimeLoRA 内部分段：

```text
Arrival/replay wait
  -> dispatch admission wait
  -> pre-runtime service shell
  -> runtime TTFT
  -> first token
```

可观测字段：

| 阶段 | 字段 | 状态 |
|---|---|---|
| Dispatch/admission | `avg_dispatch_admission_wait_ms` | 已有 |
| Pre-runtime shell | `avg_pre_runtime_service_shell_ms` | 已有 |
| Runtime TTFT | `avg_runtime_ttft_ms` | 已有 |
| LoRA I/O side evidence | `avg_lora_io_ms` | 已有 |
| GPU-ready request TTFT | `avg_gpu_ready_ttft_ms` | 已有 |

不要把 baseline 内部 adapter loading phase 填入这张图，除非未来每个 baseline 都能真实输出同语义字段。

## 4. Run tag 命名规则

统一格式：

```text
{model}_r{requests}_a{adapters}_z{zipf}_hot{hotcap}_rot{rotation}_s{timescale}_seed{seed}_{section}_{system_or_variant}_v{version}
```

推荐 model slug：

- `llama2_7b`
- `llama2_13b`
- `qwen_7b`
- `qwen_14b`

推荐 section：

- `debug`
- `intro`
- `main`
- `motivation_mismatch`
- `motivation_cost`
- `motivation_serverless`
- `ablation`
- `hparam_hotcap`
- `hparam_retention`
- `scale_adapters`
- `scale_load`
- `resource_cost`
- `sanity`

系统后缀：

- `faaslora_full`
- `faaslora_no_c1`
- `faaslora_no_c2`
- `faaslora_no_c3`
- `sglang_dp4_tp1`
- `vllm_dp4_tp1`
- `serverlessllm_vllm`
- `punica_llama7b`，仅限 Punica 支持的模型范围。

示例：

```text
llama2_7b_r4000_a500_z1p0_hot48_rot500_s8_seed42_main_faaslora_full_v1
llama2_7b_r4000_a500_z1p0_hot48_rot500_s8_seed42_main_sglang_dp4_tp1_v1
llama2_7b_r4000_a500_z1p0_hot48_rot500_s8_seed42_ablation_no_c1_v1
llama2_7b_r4000_a300_z1p0_hot32_rot500_s8_seed42_scale_adapters_faaslora_full_v1
```

## 5. 结果目录组织

正式结果统一放在：

```text
results/paper_experiments/
```

目录结构：

```text
results/paper_experiments/
  00_shared_artifacts/
  01_intro_teaser/
  02_workload_characterization/
  03_main_comparison/
  04_motivation/
  05_ablation/
  06_scalability/
  07_hyperparams/
  08_resource_cost/
  09_sanity/
  _manifests/
```

每个 run 建议保存：

```text
<section>/<run_tag>/
  shared_trace.json
  adapter_subset.json
  raw/
  summaries/
  compare/
  logs/
  figures_input/
  MANIFEST.json
  NOTES.md
```

`MANIFEST.json` 必须包含：

- `run_tag`
- `section`
- `system`
- `variant`
- `git_commit`
- `git_branch`
- `metric_schema_version`
- `model_profile`
- `dataset_profile`
- `workload_profile`
- `total_requests`
- `selected_num_adapters`
- `zipf_exponent`
- `active_adapter_cap`
- `hotset_rotation_requests`
- `time_scale_factor`
- `seed`
- `shared_trace_path`
- `shared_adapter_subset_path`
- `result_json_path`
- `raw_replay_path`
- `summary_json_path`
- `tmux_session`
- `command_file`
- `started_at`
- `finished_at`
- `status`

这样未来画图只读 `results/paper_experiments/**/MANIFEST.json` 和 summary/result JSON，不需要靠聊天记录回忆。

## 6. 执行顺序

正式执行建议：

1. 先生成并冻结 shared trace 与 adapter subset。
2. 运行 500-request debug，确保四系统都输出 `e2e_v3` 且 completed rate 合格。
3. 运行 Llama-2 7B 4000-request 主横向对比。
4. 审计 served-token、completed、metric schema、GPU budget、Cost/req、CE。
5. 运行 vLLM/SGLang/ServerlessLLM/FaaSLoRA 对比图输入导出。
6. 运行 motivation 三组实验。
7. 运行 ablation。
8. 运行 hyperparameter。
9. 运行 scalability。
10. 最后再扩展到 13B/Qwen；不要在 7B 主链未闭合前并行扩展。

## 7. 文献和系统设置对齐备注

本文档的实验设计遵循以下原则：

- ServerlessLLM 类论文通常强调 cold start、model loading、effective service ratio 和 deadline；PrimeLoRA 保留这些指标，但主问题更聚焦 many-LoRA adapter readiness。
- S-LoRA/Punica 类 multi-LoRA serving 工作通常强调多 adapter serving、throughput、batching 和 memory efficiency；PrimeLoRA 继承 many-LoRA 长尾热度设置，但额外加入 serverless lifecycle cost。
- SGLang/vLLM 类 serverful serving 作为 strong serverful baselines；正式对比必须计入完整 lifecycle GPU cost，而不是只计 active request execution time。
- 论文中不要声称 baseline 没有的内部机制指标；只能从统一可观测结果解释系统差异。

相关参考记录已整理在：

- `docs/对比实验日志.md`
- `docs/CODEX_INTERACTION_RULES.md`
- baseline 工作区的 `docs/SYSTEM_REPRODUCTION_RULES.md`
- baseline 工作区的 `docs/FAIR_COMPARISON_EXECUTION_PLAN.md`

## 8. 当前工程补齐项

这些不是新论文指标，而是为了让实验结果更容易复用。

| 项目 | 目的 | 状态 |
|---|---|---|
| Paper manifest writer | 自动生成 `MANIFEST.json`，避免手工找结果 | 待补 |
| Figure input exporter | 从 result/summary JSON 导出统一 CSV/JSONL | 待补 |
| Shared artifact registry | 记录每个 run tag 对应 trace/subset | 待补 |
| Cross-system field validator | 检查正式图所需字段都非空 | 待补 |

只有这些工程项补齐后，才新增自动画图脚本。不要先画图再回头补字段。
