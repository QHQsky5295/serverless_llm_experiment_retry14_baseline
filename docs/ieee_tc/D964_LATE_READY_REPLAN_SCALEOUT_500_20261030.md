# D964：就绪边界需求重规划的自然扩容诊断（7B、500 请求）

日期：2026-10-30  
身份：诊断前缀，不是正式 G1/G2 性能点  
代码提交：`c10d636e`  
输入：既有 W2 shared trace、既有 500-adapter subset、同一固定长度/native token 合同；未生成新工件或新负载

## 目的与边界

D963 的 100 请求前缀没有触发扩容，无法覆盖本轮改动的目标路径。D964 将同一冻结输入扩展到计划中允许的 500 请求诊断前缀，只为观察自然 scale-out 是否在真实 GPU 运行中完成“触发 epoch → worker 就绪边界 epoch → 发布可路由副本”的顺序。它不替代 4,000 请求正式留出回放，所有汇总指标均不得外推。

## 运行与合同

- 500/500 完成，0 失败；固定长度/native token 合同通过。
- 3 次自然 scale-out、2 次 scale-down；4 个 activation（1 initial + 3 natural scale-out）。
- 所有 3 个扩容 activation 都记录了 `trigger_plan_sha256` 与不同的最终 `plan_sha256`，并记录 `preparation_replan.kind=observed_ready_boundary_v1`、`future_requests_used=false`。
- ready-boundary demand 快照的 `total_arrivals` 分别为 1、7、9，反映的是就绪时刻最近 5 秒观察窗，而不是未来 trace 或静态热点先验。
- 3 个 fresh epoch 均为 `state=completed`，副本在准备完成后才进入 `running`/可路由状态。

## 机制观察

扩容首服务样本为 3 个：

| 首服务请求 | 就绪 tier | GPU 命中 | 首服务 TTFT |
|---|---:|---:|---:|
| `req_00113` / `finance_lora_0037` | NVMe | 否 | 577.7 ms |
| `req_00330` / `medical_lora_0038` | GPU | 是 | 362.3 ms |
| `req_00344` / `finance_lora_0049` | NVMe | 否 | 422.7 ms |

因此 fresh epoch 在真实扩容路径上确实生效：没有把新副本直接暴露给一个仍在执行的旧计划，也没有把未来请求加入当前规划。`scaleup_planned_adapter_match` 仍为旧 legacy 标签，在 confirmed IEEE 路径中不解释为规划失败。

## 诊断数值（不作正式比较）

- 平均 overall TTFT：746.25 ms；P95：1,768.53 ms；TTFT SLO：100%。
- 首服务平均 TTFT：454.23 ms；首服务 GPU 命中率：33.3%。
- 这些数字与 D931 的完整 4,000 请求结果不构成同合同因果对照：请求数、扩容次数和运行区间不同，不能据此宣称性能提升或 GPU-s 最优。

结果文件：

`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d964_20261030/7b_route_overlap_validation_100/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d964_7b_late_replan_validation_500.json`

## 下一步裁决

本诊断验证了改动的物理执行语义，但没有完成 7B G1/G2。下一步应在不改变输入、SLO 或资源包络的前提下运行一个完整 4,000 请求开发/留出点，检查 fresh epoch 带来的首服务等待、远程 miss、GPU-s/request 和联合 SLO 是否相对 D931 改善。若完整点退化，应保留 D964 作为机制证据，并撤回该优化候选，不继续堆叠补丁。
