# D930：因果感知扩容门控的留出结果与否决

日期：2026-10-29  
模型：Llama-2-7B；W2；4,000 requests；500 adapters；固定 `fixed_length_greedy_v1` 生成契约。  
运行键：`d930_20261028/7b_cause_aware_scale_guard_full_w2`。

## 结论

D930 是一次完整、可复核的 PrimeLoRA 留出回放，但不是当前 7B 主结果。它比 D927 的需求门控候选明显改善，却仍然劣于已存在的 D917 当前控制结果。因此本轮门控不冻结，也不进入外部 baseline 比较或论文图表。

这不是“为了保留较好结果而删除一次运行”：D930 的失败原因具有可观测的控制含义。门控只用当前可见 backlog/active capacity 与 ready-time queue 判断是否可以抑制 TTFT 触发的扩容；它没有把启动窗口内的服务压力和启动期间的到达增长作为必须满足的安全条件。于是最初的扩容从 D917 的 request 3 推迟到 D930 的 request 110，随后在 request 586--592 和 986--998 形成补偿性扩容。该行为增加了早期 adapter/replica mismatch 的等待，并不能代表“同样服务约束下的最小 GPU 生命周期”。

## 完整性与资格边界

- 终端完成：4,000/4,000；失败终端：0。
- 固定生成契约：4,000/4,000；原生 token source：4,000/4,000；`timing_contract=ieee_tc_native_v1`。
- `n_correct=null`，因此本文件不把 HTTP 成功、token 数量或 timing contract 当作数值 LoRA 正确性的证明。
- 物理生命周期窗口完整，观察到的 GPU-s/request 为 3.0955876595；这是 G1 的候选指标，而不是比较表中按常驻实例计费的 `infra_gpu_seconds_total/requests`。

## 冻结 D394 阈值下的逐请求结果

按输入长度分层使用 D394 的共同 TTFT/TPOT 阈值，逐请求重新计算，不使用旧的 5,000-ms 单一 TTFT 显示：

| 运行 | TTFT pass | TPOT pass | joint pass | joint rate | physical GPU-s/request |
|---|---:|---:|---:|---:|---:|
| D917 当前控制 | 3,898 | 3,974 | 3,883 | 97.075% | 2.876960 |
| D930 因果门控 | 3,875 | 3,976 | 3,856 | 96.400% | 3.095588 |

D930 满足 3,800/4,000 的联合时延门槛，但在联合通过数和物理生命周期 GPU-s/request 两方面均不及 D917；因此不能声称门控同时改善 G1/G2。

## 端到端与机制对照

| 运行 | mean TTFT (ms) | P95 TTFT (ms) | mean TPOT (ms) | mean E2E (ms) | scale-up events | first-service requests |
|---|---:|---:|---:|---:|---:|---:|
| D917 当前控制 | 1124.8 | 2672.3 | 31.75 | 4717.6 | 19 | 19 |
| D930 因果门控 | 1230.8 | 3257.1 | 30.97 | 4802.3 | 10 | 30 |

D930 相对于 D927（physical GPU-s/request 3.637718、mean/P95 TTFT 1380.6/3695.7 ms）确实降低了 GPU 生命周期与 TTFT；但相对于 D917，GPU 生命周期增加约 7.6%，平均 TTFT 增加约 9.4%，P95 TTFT 增加约 21.9%。因此 D930 的“相对 D927 改善”不能被误写成“相对当前最佳结果改善”。

## 控制解释与下一步

下一候选必须把扩容安全条件写成“启动完成时的可服务需求”而不是“当前可见请求数足够”。至少需要同时检查：

1. 已确认的 ready-time queue 与 projected arrivals；
2. 启动延迟覆盖窗口内仍会到达的请求及其服务前缀；
3. 首次服务前缀的 adapter/replica 兼容性；
4. 当前压力样本是否来自可操作的 tier/admission/startup 路径。

在该安全条件没有被现有观测完整证明之前，不应继续增加人工阈值或用结果驱动的常数修补。下一轮应先以 D917 的当前控制作为可接受 7B 参考，再针对上述可观测缺口设计一个单独、可证伪的候选；D930 的所有原始文件和负面结论保留。

## 原始证据

- 结果 JSON：`results/ieee_tc/p2_backend_qualification/d930_20261028/7b_cause_aware_scale_guard_full_w2/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d930_7b_cause_aware_scale_guard_full_w2.json`
- 物理窗口：同目录 `launch.launch/physical_deployment/summary.json`
- D917 对照：`results/ieee_tc/p2_backend_qualification/d917_20261027/7b_current_full_w2_remote7b/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d917_7b_current_full_w2_remote7b.json`
- 本记录不修改既有 `final_v2/`、旧 `figs/paper/` 或历史结果。
