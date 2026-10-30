# D946：ServerlessLLM D941 物理资源证据字段审计

日期：2026-10-30  
计划 SHA：`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`

## 审计对象

审计对象是 ServerlessLLM D941 的原始回放和汇总文件，不修改原始结果：

- `serverless_llm_baselines/results/paper_experiments/15_new_serverless_baselines_remote_v1/20261030_formal_serverlessllm_new_7b_*/raw/replay/*_replay.json`
- 同目录 `*_summary.json`

## 已存在的时间和身份字段

逐请求记录包含 `server_metrics.instance_id`、`instance_created_at`、`instance_ready_at`、`backend_started_at`、`request_received_at`、`finished_at`，以及请求级 `success`、`overall_ttft_ms`、`e2e_ms` 和生成合同字段。该运行共观察到 5 个不同实例 ID，汇总字段报告每个实例为 1 GPU，且 `infra_gpu_seconds_total=16899.378489`。

这些字段足以支持：

- 请求级 TTFT/E2E、联合 SLO 和 goodput；
- cold-start、dispatch/admission 与服务阶段分解；
- 实例创建/ready 时间的部署行为诊断；
- 现有 `simulated_gpu_second_deployment_lifecycle_v3_capped` 估计的来源追踪。

## 缺失的物理计量证据

在 D941 原始回放、汇总和 MANIFEST 中未发现以下同一物理合同所需字段：

| 证据 | 当前状态 | 不能做的推断 |
|---|---|---|
| 物理 GPU device identity | 未记录 | 不能确认多个实例是否对应同一物理设备、是否发生复用或重叠 |
| allocation/lease acquire 时间 | 未记录 | 不能从实例创建时间替代真实 GPU 分配起点 |
| release/lease close 时间 | 未记录 | 不能用最后一个请求完成时间冒充资源释放终点 |
| 运行结束 lease census | 未提供 | 不能证明所有物理资源已关闭且没有尾部占用 |
| 与 `physical_gpu_resource_summary_v1` 相同格式的摘要 | 未提供 | 不能与 PrimeLoRA D931 的物理摘要直接相除 |

已知的 `deployment_idle_tail_s=300` 和 `serverless_idle_retention_s=300` 是生命周期模型参数，不是观测到的每实例释放事件。将“最后请求完成 + 300 秒”直接作为物理 release，只能生成模型重建值，不能称为原始物理测量。

## 裁决

1. D941 的 TTFT、P95、E2E、联合 SLO、goodput 和实例行为诊断继续有效。
2. D941 的 `infra_gpu_seconds_total` 保留为部署生命周期估计，不能与 D931 的 `physical_gpu_seconds_total=11716.06410016818` 直接计算 G1 资源优势。
3. 不重跑 D931/D941，仅因字段不一致而伪造物理摘要；若后续 G1 主张必须覆盖 ServerlessLLM，则增加一次同合同物理资源侧证运行，或将 ServerlessLLM 的 G1 资源列标为 unavailable。

## 与后续实验的关系

该审计不阻塞已经可比的请求级服务分析，也不要求所有 baseline 达到预设 SLO。它只限定“物理 GPU-s 资源排名”必须使用同一证据合同；SLO 达成率本身仍作为结果维度报告。PrimeLoRA 的下一轮优化应在完成同合同 baseline 对照后，针对已观测到的瓶颈提出单一可证伪假设，不因 D941 的资源字段缺失而盲目重跑或改变 PrimeLoRA 的研究边界。

