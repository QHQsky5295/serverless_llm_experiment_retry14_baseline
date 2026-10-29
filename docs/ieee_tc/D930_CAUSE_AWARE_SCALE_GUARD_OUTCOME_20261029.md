# D930：压力分类扩容守卫的开发结果与归因更正

日期：2026-10-29  
模型：Llama-2-7B；W2；4,000 requests；500 adapters；固定 `fixed_length_greedy_v1` 生成契约。  
运行键：`d930_20261028/7b_cause_aware_scale_guard_full_w2`。

## 结论

D930 是一次完整、可复核的 PrimeLoRA 已知开发轨迹回放，不是独立留出评估或当前 7B 主结果。它比 D927 的需求门控候选改善，却仍然劣于已存在的 D917 当前控制结果。因此本轮门控不冻结，也不进入外部 baseline 比较或论文图表。

2026-10-29 离线重审发现，本文件初版把扩容时序差异直接归因于守卫，是证据不足的解释，应撤回。完整控制日志中 D930 的 `no_action_current_demand_covered` 次数为 **0**：没有一次扩容因该守卫被抑制。D917 第一条请求 TTFT 为 13,669.073 ms，随后三次扩容都由该 TTFT 压力触发；D930 第一条为 1,291.387 ms，最初扩容由 active 压力触发。两者面对的反馈初态不同。守卫开启时仍可能执行额外投影计算，但其开销及影响尚未隔离，不能改为另一个未经检验的因果结论。原始运行、旧文档版本和阴性结果全部保留。

## 完整性与资格边界

- 终端完成：4,000/4,000；失败终端：0。
- 固定生成契约：4,000/4,000；原生 token source：4,000/4,000；`timing_contract=ieee_tc_native_v1`。
- `n_correct=null`，因此本文件不把 HTTP 成功、token 数量或 timing contract 当作数值 LoRA 正确性的证明。
- 物理生命周期窗口完整，观察到的 GPU-s/offered-request 为 3.0955876595；这是诊断量，不是 G1 的 GPU-s/correct-request。`n_correct` 未知时后者保持未知，不能用旧 `infra_gpu_seconds_total/requests` 代替物理计量。

## 冻结 D394 阈值下的逐请求结果

按输入长度分层使用 D394 的共同 TTFT/TPOT 阈值，逐请求重新计算，不使用旧的 5,000-ms 单一 TTFT 显示：

| 运行 | TTFT pass | TPOT pass | joint timing pass | timing rate | physical GPU-s/offered |
|---|---:|---:|---:|---:|---:|
| D917 当前控制 | 3,898 | 3,974 | 3,883 | 97.075% | 2.876962 |
| D930 因果门控 | 3,875 | 3,976 | 3,856 | 96.400% | 3.095588 |

D930 满足 3,800/4,000 的联合时延门槛，但在联合时延通过数和物理生命周期 GPU-s 两方面均不及 D917；因此不能声称门控同时改善 G1/G2。此处是 native/binding 证据下的时延统计，不补造独立数值正确性或 G2 预算资格。

## 端到端与机制对照

| 运行 | mean TTFT (ms) | Type-1 P95 TTFT (ms) | mean TPOT (ms) | mean E2E (ms) | scale-up events | legacy first-service labels |
|---|---:|---:|---:|---:|---:|---:|
| D917 当前控制 | 1124.8 | 2672.0 | 31.75 | 4717.6 | 19 | 19 |
| D930 因果门控 | 1230.8 | 3256.4 | 30.97 | 4802.3 | 10 | 30 |

D930 相对于 D927（physical GPU-s/offered 3.637717、mean/旧汇总 P95 TTFT 1380.6/3695.7 ms）降低了 GPU 生命周期与 TTFT；但相对于 D917，GPU 生命周期增加约 7.6%，平均 TTFT 增加约 9.4%，Type-1 P95 TTFT 增加约 21.9%。因此 D930 的“相对 D927 改善”不能被误写成“相对当前最佳结果改善”。first-service 标签计数沿用旧摘要，仅作标签诊断，不代替 A4 的每 epoch 第一实际 dispatch 定义。

## 控制解释与下一步

不基于已经被反证的“守卫推迟扩容”解释继续加规则。D931 已关闭该守卫并完成单次回放；结果为 3,881/4,000 时延通过、11,716.064 GPU-s、Type-1 P95 TTFT 2,997.200 ms，仍未超过 D917。下一项是复用三次运行的原生阶段与代码历史，定位请求释放、控制器处理和首请求反馈差异；没有新机制证据前不重跑同键、不继续修补守卫、不把差异自动归因于 JIT。

统一离线证据：`paper_results/ieee_tc/p2_backend/d931_control_audit/summary.json`。三轮 request/adapter/arrival/prompt/native prompt/target 合同摘要相同，阶段恒等式最大误差均为 0 ms；这不证明全部执行初态及源码相同，也不构成独立留出或因果试验。

## 原始证据

- 结果 JSON：`results/ieee_tc/p2_backend_qualification/d930_20261028/7b_cause_aware_scale_guard_full_w2/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d930_7b_cause_aware_scale_guard_full_w2.json`
- 物理窗口：同目录 `launch.launch/physical_deployment/summary.json`
- D917 对照：`results/ieee_tc/p2_backend_qualification/d917_20261027/7b_current_full_w2_remote7b/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d917_7b_current_full_w2_remote7b.json`
- 本记录不修改既有 `final_v2/`、旧 `figs/paper/` 或历史结果。
