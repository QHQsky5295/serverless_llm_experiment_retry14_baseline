# D944：D931/D941 GPU-s 计量修正

日期：2026-10-30  
计划 SHA：`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`

## 修正原因

D942 的合同审计正确确认了输入和生成合同，但其中把两个不同来源的 GPU-s 直接放在一起比较，这超出了冻结指标协议的证据强度。该问题现在明确更正，D942 原始记录保留，不覆盖历史文件。

## 两个结果的真实字段

### PrimeLoRA D931

- `comparison_table.infra_gpu_seconds_total = 4309.743187`：PrimeLoRA 汇总器的逻辑实例生命周期字段，不是本协议的最终物理资源字段。
- `physical_deployment/physical_resource_summary.json`：
  - `contract = physical_gpu_resource_summary_v1`；
  - `measurement_complete = true`；
  - `physical_gpu_seconds_total = 11716.06410016818`；
  - 4,000/4,000 终态，未关闭的物理 lease 为 0。
- 因此，若讨论 G1 的物理生命周期资源，D931 只能使用 11,716.0641 GPU-s；不能使用 4,309.7432 GPU-s。

### ServerlessLLM D941

- `infra_gpu_seconds_total = 16899.378489` 来自 `simulated_gpu_second_deployment_lifecycle_v3_capped`，其字段包括 5 个一 GPU 实例的生命周期估计。
- 该结果目前没有与 D931 同格式的 `physical_gpu_deployment_v1` / `physical_gpu_resource_summary_v1` 物理分配、释放和 lease census 证据。
- 因此 16,899.3785 GPU-s 可以作为 ServerlessLLM 的部署生命周期观测和成本诊断，但不能直接与 D931 的 11,716.0641 物理 GPU-s 计算 G1 节省比例。

## 撤回的结论

D942 中“PrimeLoRA 相对 ServerlessLLM 的 GPU-s/request 低约 30.7%”以及由此派生的最终资源排名，现全部降级为**不发布结论**。该算术本身没有意义，因为分子和分母不是同一个物理计量合同。

TTFT、P95 TTFT、联合 SLO 和请求合同的一致性仍然有效；D941 的平均 dispatch/admission wait、service TTFT 和生命周期字段继续作为机制诊断数据保存。

## 当前比较裁决

| 目标 | 当前状态 |
|---|---|
| 输入、adapter 集合、生成合同 | 可比较 |
| 请求级 TTFT/E2E | 可作诊断性比较，保留后端和 limiter 差异 |
| 物理 GPU-s/request 的 G1 排名 | 尚未闭合 |
| CE/美元成本 | 仅作二级诊断，不能替代 G1 |
| 数值 LoRA 正确性 | 两边均未获得独立 oracle，保持 unknown |

## 后续处理

1. 不重跑 D931 或 D941，不修改其原始结果。
2. 先检查 ServerlessLLM 已有实例生命周期日志能否通过真实 GPU allocation/release、实例身份和结束时刻重建 `physical_gpu_resource_summary_v1`。只有证据完整时才离线重算；不允许把模拟生命周期字段改名冒充物理字段。
3. 若无法重建，则 ServerlessLLM 保留 TTFT/SLO baseline 行，G1 资源列标记为 unavailable；再按计划选择是否为 ServerlessLLM 增加一次同合同物理资源侧证运行。
4. 在物理 GPU-s 口径闭合前，不生成 G1 资源排名图，也不据此继续调 PrimeLoRA。

## 证据位置

- PrimeLoRA D931 物理摘要：`results/ieee_tc/p2_backend_qualification/d931_20261029/7b_current_guard_off_w2/launch.launch/physical_deployment/physical_resource_summary.json`
- PrimeLoRA D931 逻辑汇总：对应实验目录下的 `experiment_results_*.json`
- ServerlessLLM D941：`serverless_llm_baselines/results/paper_experiments/15_new_serverless_baselines_remote_v1/20261030_formal_serverlessllm_new_7b_*/raw/replay/*_summary.json`
- 原合同审计：`docs/ieee_tc/D942_PRIME_SERVERLESSL_COMPARISON_CONTRACT_AUDIT_20261030.md`
- 指标冻结：`docs/ieee_tc/METRIC_PROTOCOL_EXECUTION_FREEZE_20261030.md`
