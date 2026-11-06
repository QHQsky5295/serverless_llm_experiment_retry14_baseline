# D1100：7B W2 busy-service-rate outcome

## 身份与审计范围

- 执行键：`d1100_20261107 / 7b_busy_service_rate_w2`
- 配置：`configs/ieee_tc/20261107_d1100_7b_busy_service_rate_w2.yaml`
- 原始结果目录：`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1100_20261107/7b_busy_service_rate_w2`
- 共同合同：Llama-2 7B、vLLM 0.30.0、W2、4,000 请求、500-adapter、TP=1、fixed-length greedy v1、同一 W2 trace/adapter map、真实 HTTP `prepublished_gzip_v1` 交付。
- Plan SHA：`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`。

`launch.json` 给出 `pass=true`、service/replay/watchdog return code 均为 0、native GPU context release 已确认、service path 已移除；回放日志给出 `N_plan=N_arrived=N_submitted=4000` 和 `replay_complete=1`。远端停止回执为 `inactive`，结束后本机四张 GPU 均回到约 15 MiB、0% utilization。请求时 `remote_artifact_transfer_count=497`，`request_time_pack_count=0`；因此本运行没有把远端请求时打包混入服务路径。

## 冻结联合指标

按照 `20261010_7b_slo_d394_frozen.json` 的两档共同阈值逐请求重算，而不是使用运行时 legacy 的 TTFT-only 聚合：

- `TTFT <= 2995.3487 ms` 且 `TPOT <= 59.7441 ms`（prompt ≤ 616）；
- `TTFT <= 4633.3454 ms` 且 `TPOT <= 85.2470 ms`（prompt > 616）。

结果为：4,000/4,000 请求成功，TTFT 达标 3,941/4,000，TPOT 达标 3,974/4,000，联合达标 3,922/4,000，即 **98.05%**。平均 TTFT 为 1,046.1 ms，P95 TTFT 为 2,815.1 ms，平均 TPOT 为 30.76 ms，P95 TPOT 为 49.32 ms。`n_correct=null` 仍表示当前零权重 adapter 池没有提供数值正确性判定；本结果只能作为 timing/resource-qualified PrimeLoRA 候选，不能把 token contract 成功误写为数值正确性证明。

## 物理资源结果

物理 GPU allocation-union 摘要为：

- `physical_gpu_seconds_total = 10,515.9963 s`；
- `N_terminal = N_plan = 4,000`，`open_lease_ids=[]`，`measurement_complete=true`；
- 物理生命周期 GPU-s/request（按 offered terminal request 归一）为 **2.6290**。

这里采用 `physical_resource_summary.json` 的物理 owner-union，不采用结果 JSON 中旧的逻辑实例 billing 字段（后者为 15,566.05 GPU-s，属于兼容诊断视图，不是本计划的 canonical physical measurement）。

## 与此前 PrimeLoRA 控制结果的同合同对照

| 运行 | 联合 SLO | P95 TTFT | 平均 TPOT | 物理 GPU-s/request |
|---|---:|---:|---:|---:|
| D931 current-control W2 | 97.025% | 2,997.7 ms | 31.54 ms | 2.9290 |
| D1095 remote-handshake W2 | 96.750% | 3,192.6 ms | 31.06 ms | 3.0535 |
| **D1100 busy-service-rate W2** | **98.050%** | **2,815.1 ms** | **30.76 ms** | **2.6290** |

相对 D931，D1100 的物理 GPU-s/request 降低约 10.24%，联合 SLO 提高 1.025 个百分点，P95 TTFT 降低约 6.10%；相对 D1095，物理 GPU-s/request 降低约 13.90%，联合 SLO 提高 1.30 个百分点，P95 TTFT 降低约 11.82%。这些是单轮同合同观测差异，不声称统计显著性。

控制器实际发生 24 次扩容、23 次回收，18 个请求被标记为 scale-up first-service。GPU/remote dispatch tier 计数分别为 2,318/503；远端获取次数保持为 497。`ieee_predictive_scaling` 因而确实触发了控制路径，但其收益归因仍应表述为“忙时服务率与预测回收的候选改进”，不能把单轮差异拆成未经实验隔离的单机制因果结论。

## 裁决与后续边界

D1100 **接受为 7B PrimeLoRA 的当前优化候选**：它满足共同 timing SLO（>95%），并在 canonical physical GPU-s/request 上同时优于 D931 和 D1095；因此满足“先把 PrimeLoRA 优化得比此前同合同运行更好，再进入外部 baseline”这一阶段条件。

它不是最终数值正确性封存点，因为当前 500-adapter 固定池的 `n_correct` 仍为未知。后续比较必须保留这一限制，不能选择性把 D1100 宣称为全部正确。下一步按计划进入 7B ServerlessLLM baseline 的历史日志审计和同合同复现；不再为 D1100 盲目重复运行，也不生成图表，直到 PrimeLoRA 与外部系统完成同合同对照。
