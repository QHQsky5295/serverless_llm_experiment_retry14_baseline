# Service-Readiness Gap 机制证据链计划

本文档记录 PrimeLoRA/FaaSLoRA 论文新增 service-readiness gap 分析的
取数口径、章节边界、已有结果可支持的结论，以及后续若需要更强证据时的最小
instrumentation。当前工作基于 `QHQsky5295/FaaSLoRA` 仓库
`retry14_continuous_queue_v2` 分支；不要切到 `main` 分支生成这些图或脚本。

## 1. 章节边界

这组分析的目标不是“多画几张图”，而是补上从 per-request adapter readiness
到 CE/cost-latency tradeoff 的机制证据链。但证据必须放在正确章节：

- Motivation 不能使用 `faaslora_full`、`PrimeLoRA-NVMe`、
  `PrimeLoRA-NoCoord` 或任何 PrimeLoRA 内部 instrumentation 来证明
  “已有系统存在问题”。否则读者会认为我们在证明 proposed system 自己仍有
  问题，和后续 Evaluation/Ablation 的职责冲突。
- Motivation 继续使用外部或代表性 baseline 的问题证据：
  - `figs/paper/motivation/fig2_mismatch.pdf`：来自 ServerlessLLM，
    证明 serverless runtime/admission readiness gap；
  - `figs/paper/motivation/fig3_tier.pdf`：来自 S-LoRA/shared replay，
    证明 adapter churn/reuse gap 会放大 TTFT tail。
- PrimeLoRA per-request readiness 图表只能放在 Evaluation/Ablation，
  用来解释机制是否减少 service-readiness gap，以及 CE 提升是否不仅来自
  cost model。

## 2. 外部 baseline 字段审计

已检查 Llama-2 7B main round 的五系统 replay：

```text
/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260424_104050_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1
```

审计结论：

- ServerlessLLM replay 有 `scaleup_affected`、`scaleup_first_service`、
  `cold_start_latency_ms`、`dispatch_admission_wait_ms` 和
  `runtime_ttft_ms`，但没有 adapter tier / selected-replica LoRA readiness。
- S-LoRA、vLLM、SGLang replay 有请求级 TTFT/E2E/TPOT 和 adapter id，
  但没有 `cache_tier`、`lora_load_ms` 或 dispatch 前 readiness tier。
- 因此，当前已有外部 baseline 不能严谨生成
  `selected-replica adapter tier distribution` 类型的 Motivation 图。
  若一定要在 Motivation 中展示 selected-replica adapter tier，需要重新补外部
  baseline instrumentation 或新增独立诊断实验；当前不硬画，避免埋雷。

## 3. Evaluation/Ablation 输入数据

当前可直接用于 service-readiness 机制审计的是已闭合的 Llama-2 7B
ablation round：

```text
/home/qhq/serverless_llm_baselines/results/paper_experiments/04_ablation/20260426_131203_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_ablation_v1
```

使用的原始结果：

```text
raw/faaslora/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_ablation_v1_faaslora_nvme_result.json
raw/faaslora/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_ablation_v1_faaslora_no_coord_result.json
raw/faaslora/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_ablation_v1_faaslora_full_result.json
```

三个文件均包含 `4000` 条 successful LoRA-bearing request，且有：

```text
adapter_id
success
cache_tier
lora_io_ms
defer_ms
overall_ttft_ms
overall_e2e_ms
vllm_ttft_ms
tpot_ms
ingress_queue_wait_ms
dispatch_admission_wait_ms
dispatch_window_wait_ms
runtime_slot_wait_ms
scaleup_first_service
scaleup_planned_adapter_match
```

当前归档结果没有 `readiness_tier_before_dispatch`。因此本轮审计使用
`cache_tier` 作为 selected-replica service-time adapter-readiness proxy。
这个口径可以支持 Evaluation 中的机制审计，但不能写成严格的 dispatch-before
selected-replica tier。若投稿版需要更强 claim，应按第 9 节补最小
instrumentation 后重跑相关 round。

## 4. 定义

默认统计条件：

```text
success == True
adapter_id 非空
```

派生定义：

- GPU-ready：`cache_tier == "gpu"`，目标 adapter 已经可在 selected replica 上
  GPU 执行。
- Local-prepared：`cache_tier in {"host", "nvme"}`，目标 adapter 在本地层级，
  但仍可能经历 GPU admission/loading。
- Remote-cold：`cache_tier == "remote"`，selected replica 的 GPU/HOST/NVMe
  都没有目标 adapter，是最严重 mismatch。
- Strict mismatch：`cache_tier != "gpu"`，目标 adapter 并非 GPU-ready。
- Scale-out mismatch：`scaleup_first_service == True and
  scaleup_planned_adapter_match == False`。

## 5. 已生成图表

生成命令：

```bash
cd /home/qhq/serverless_llm_experiment_retry14_baseline
python3 scripts/analyze_service_readiness.py \
  --input /home/qhq/serverless_llm_baselines/results/paper_experiments/04_ablation/20260426_131203_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_ablation_v1 \
  --output /home/qhq/serverless_llm_experiment_retry14_baseline/figs/paper/readiness
```

输出：

```text
figs/paper/readiness/service_readiness_summary.csv
figs/paper/readiness/service_readiness_by_tier.csv
figs/paper/readiness/scaleout_first_service_summary.csv
figs/paper/readiness/tables/table_service_readiness.tex
figs/paper/readiness/fig_service_readiness_summary.pdf
figs/paper/readiness/fig_mechanism_gap_ablation.pdf
figs/paper/readiness/fig_ttft_breakdown_readiness.pdf
figs/paper/readiness/service_readiness_manifest.json
figs/paper/readiness/service_readiness_warnings.txt
```

图表用途：

- `table_service_readiness.tex`：建议放 Evaluation/Ablation，作为主审计表。
- `fig_service_readiness_summary.pdf`：建议放 Evaluation/Ablation，左图同时展示
  GPU-ready rate 和 non-GPU-ready 请求的 HOST/NVMe 放大组成，右图展示
  non-GPU-ready tail penalty。
- `fig_mechanism_gap_ablation.pdf`：可作为 Fig. 6 后的补充机制矩阵或 appendix。
- `fig_ttft_breakdown_readiness.pdf`：可作为 appendix/补充图，用 CDF 展示
  PrimeLoRA 中 GPU/HOST/NVMe 请求的 TTFT 分布。

这些图均不放 Motivation。

## 6. 当前结果

| System | GPU-ready (%) | HOST/NVMe (%) | Remote-cold (%) | Mismatch (%) | TTFT p95 GPU (ms) | TTFT p95 mismatch (ms) | Prep p95 mismatch (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PrimeLoRA-NVMe | 95.62 | 4.38 | 0.00 | 4.38 | 1047.2 | 6042.7 | 51.2 |
| PrimeLoRA-NoCoord | 95.55 | 4.45 | 0.00 | 4.45 | 1024.3 | 4975.9 | 51.2 |
| PrimeLoRA | 95.80 | 4.20 | 0.00 | 4.20 | 906.5 | 5581.1 | 51.2 |

解释边界：

- Non-GPU-ready 请求占比约 `4.2--4.45%`，不是大多数请求，但它们形成了清晰
  的 tail group。
- GPU-ready 请求的 p95 TTFT 为 `0.91--1.05 s`；non-GPU-ready 请求的 p95 TTFT
  为 `4.98--6.04 s`。这支持“adapter readiness 会进入 first-token critical
  path”的机制链。
- PrimeLoRA 的 GPU-ready 占比最高，GPU-ready p95 TTFT 最低；但 mismatch-only
  p95 并非严格单调，因为 mismatch group 很小且组成敏感。因此不能声称所有
  readiness 子指标都单调最好。
- Remote-cold 在该 ablation round 中均为 `0`。这不是绘图错误，而是说明当前
  preparation/residency path 已经把 remote miss 转化为 local-prepared 或
  GPU-ready 状态。表格保留 Remote-cold=0 作为审计；图中若所有系统均为 0，
  不再显示 remote-cold 图例或全零矩阵行。
- 早期 100% stacked bar 中 HOST 只有约 `0.9--1.15%`，夹在大面积 GPU-ready
  和 NVMe 段之间，单栏下会像缺了一块。当前图改成混合审计视图：保留
  GPU-ready rate，同时放大 non-GPU-ready 组成，避免小比例被视觉压扁。

## 7. 论文插入建议

Evaluation/Ablation 中可增加一小节：

```text
Service-Readiness Audit
```

建议正文逻辑：

1. 先说明这不是新的横向 baseline 比较，而是 PrimeLoRA ablation round 的
   request-level 机制审计。
2. 说明 `cache_tier` 是 service-time adapter readiness proxy；dispatch-before
   tier 需要后续更严格 instrumentation。
3. 用表格展示 GPU-ready、HOST/NVMe、Remote-cold、Mismatch 占比，以及
   GPU-ready 与 mismatch 的 TTFT p95。
4. 用图展示 non-GPU-ready 请求虽然占比小，但 p95 TTFT 显著更高。
5. 与 Fig. 6 消融相连：这证明 CE/cost-latency tradeoff 的背后有 readiness
   tail reduction 机制证据，但不替代 Table 1 和 Fig. 7。

不建议把这组图写进 Motivation，因为数据来自 PrimeLoRA variants。

## 8. 暂不生成的图

- `fig_scaleout_first_service.pdf`：每个 scenario 只有 `6` 个
  `scaleup_first_service` 样本，其中 planned miss 只有 `2` 个。样本不足，
  不能作为稳定论文图。CSV 与 warning 已保留。
- cross-system control-plane overhead 对比：不做。现有 FaaSLoRA 结果中有
  `parent_rpc_overhead_ms`、`dispatch_admission_wait_ms` 和
  `service_overhead_ms`，但这些字段分别包含 RPC 包装、排队/准入等待或
  service-path 残差；vLLM、SGLang、S-LoRA 和 ServerlessLLM 也没有同一口径的
  routing/admission decision timing。把这些 wrapper/runtime/internal 字段横向
  对比会变成非等价指标比较。
- Motivation 版 selected-replica adapter tier 图：外部 baseline 当前没有
  adapter tier 字段，不硬画。

## 9. 后续最小 instrumentation

只有当需要严格证明 dispatch-before selected-replica readiness 时再补：

```text
readiness_tier_before_dispatch
adapter_gpu_ready_before_dispatch
adapter_local_ready_before_dispatch
adapter_remote_cold_before_dispatch
adapter_replica_mismatch
remote_mismatch
scaleout_mismatch
selected_instance_age_s
routing_decision_us
gpu_admission_decision_us
control_plane_total_us
```

2026-05-01 已补充第一版 control-path instrumentation，且不改变 workload、
adapter 采样、routing policy 或 admission policy：

- `RequestResult` 新增 dispatch-before readiness 标记：
  `readiness_tier_before_dispatch`、`adapter_gpu_ready_before_dispatch`、
  `adapter_local_ready_before_dispatch`、`adapter_remote_cold_before_dispatch`、
  `adapter_replica_mismatch`、`remote_mismatch`、`scaleout_mismatch`。
- `RequestResult` 新增在线控制路径耗时：
  `routing_decision_us`、`adapter_path_resolution_us`、
  `gpu_admission_decision_us`、`control_path_total_us`。
- `ScenarioResult` 新增聚合字段：
  `avg/p95_routing_decision_us`、`avg/p95_adapter_path_resolution_us`、
  `avg/p95_gpu_admission_decision_us`、`avg/p95_control_path_total_us`、
  `avg/p95_background_planning_us` 和
  `background_planning_event_count`。
- `ResourceCoordinator.evaluate_gpu_admission()` 仅增加计时包装，原有
  effective-capacity admission 逻辑不变。
- `_predict_scale_up_handoff_plan()` 记录 background handoff planning 时间，
  用于说明异步/准在线规划成本，不把它混入每请求 online total。

新增脚本：

```bash
python3 scripts/analyze_control_path_overhead.py \
  --input <new_faaslora_result_json_or_dir> \
  --output figs/paper/control_path
```

输出：

```text
figs/paper/control_path/control_path_overhead_summary.csv
figs/paper/control_path/tables/table_control_path_overhead.tex
figs/paper/control_path/fig_control_path_overhead.pdf
figs/fig_control_path_overhead.pdf
```

2026-05-01 已完成一次 Llama-2-7B/4000-request PrimeLoRA/FaaSLoRA 诊断
round：

```text
input:
/home/qhq/serverless_llm_experiment/results/experiment_results_full_vllm_auto_a500_r4000_c4_faaslora_full_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_controlpath_v1.json

Routing + tier lookup:       avg 0.108 ms, p95 0.195 ms, events 4000
Adapter-path resolution:     avg 4.524 ms, p95 5.957 ms, events 4000
GPU-admission check:         avg 3.354 ms, p95 23.016 ms, events 168
Online control total:        avg 4.773 ms, p95 6.119 ms, events 4000
Background handoff planning: avg 2.017 ms, p95 4.034 ms, events 6
```

`Online control total` 按每个请求统计；`GPU-admission check` 和
`Background handoff plan` 按实际触发事件统计。该图定位为 PrimeLoRA-only
audit，不作为跨系统 superiority claim。

关键要求：

- 只记录 selected replica 在 dispatch 前对目标 adapter 的 predicted tier；
- 不改变 routing policy、routing key、admission decision、workload generation
  或 adapter movement；
- instrumentation 后只需重跑相关 ablation/diagnostic round，不需要先重跑所有
  backbone robustness。
