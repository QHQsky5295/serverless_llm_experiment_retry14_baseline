# D965：就绪边界需求重规划的完整 7B 回放结果

日期：2026-10-31  
身份：PrimeLoRA 开发诊断点；不是正式 G1/G2 封存点  
代码提交：`c10d636e`（就绪边界重规划）  
输入：既有 W2 shared trace、既有 500-adapter subset、同一固定长度/native token 合同；未生成新工件或新负载

## 运行合同与完成性

- 7B、4,000 请求、500 adapters、W2 shared trace、no-delay 本地模拟远程交付。
- 4,000/4,000 请求完成，0 失败；原生 token 输出合同和固定长度生成合同通过。
- `formal_comparison_qualified=false`，`FAASLORA_FORMAL_RUN=0`；该点只用于验证候选机制和寻找性能瓶颈，不冒充正式留出结果。
- 运行结束后 4 个专用 runtime 均释放，GPU 显存回到空闲状态。

结果文件：

`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d965b_20261031/7b_route_overlap_validation_100/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d965b_7b_late_replan_full.json`

机制证据文件：

`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d965b_20261031/7b_route_overlap_validation_100/launch.launch/physical_deployment/main_outcome.json`

## 机制结果：重规划确实在自然扩容路径执行

完整回放产生 13 次自然扩容、10 次缩容。机制事件中的 13 个自然扩容 activation 均满足：

1. 旧的触发时准备任务被记录为 `trigger_preparation_state=cancelled`；
2. `preparation_replan.kind=observed_ready_boundary_v1`；
3. `future_requests_used=false`；
4. 需求来源为 worker 就绪时的 hotness tracker 快照，native/file 来源为就绪后的 owner snapshot；
5. 新计划状态为 `completed`，副本随后才进入可路由状态；
6. 触发计划 SHA 与最终计划 SHA 不同，说明不是把旧计划重新贴标签。

就绪边界快照的 `total_arrivals` 为 4、1、3、7、10、4、2、3、7、3、3、1、5。它们是就绪边界前的在线观察值，不是未来 trace 读取，也不是静态热点先验。

因此，D965 支持“确认后的 tier 状态可以在实际扩容 worker 就绪边界重新计算并传播”的执行语义；它不能单独证明该机制改善最终性能。

## 完整回放指标

| 指标 | D965 |
|---|---:|
| 平均 TTFT | 1,644.5 ms |
| P95 TTFT | 4,294.3 ms |
| P99 TTFT | 19,419.3 ms |
| 平均 E2E | 5,240.5 ms |
| P95 E2E | 11,057.2 ms |
| 平均 TPOT | 31.21 ms |
| 联合 TTFT 达成率（5,000 ms） | 95.75% |
| scale-up 受影响请求数 | 710 |
| scale-up 首服务样本数 | 13 |
| 首服务平均 TTFT | 3,088.0 ms |
| 首服务 GPU 命中率 | 23.08% |
| 平均 LoRA I/O | 256.2 ms |
| cache 命中率 | 87.55% |
| GPU 命中率 | 56.73% |
| 远程传输/本地物化 | 计入请求路径；远端发布管理不计入 |

物理 GPU-s 字段在该诊断点中不能作为 G1 结论：结果汇总同时报告 `infra_gpu_seconds_total=4313.20`、`infra_active_gpu_seconds=12800.58`，而 `infra_avg_replicas≈1` 与 13 次扩容不一致，说明当前生命周期聚合仍只完整覆盖了主实例。依据冻结指标协议，缺失或不一致的物理 allocation/release 不能推算或补填；D965 不用于 GPU-s/request 排名。

## 与 D931 的客观裁决

D931 是相同 7B W2 输入和资源包络下的前一候选：平均 TTFT 1,215.95 ms、P95 约 2,997.7 ms、联合达成率 97.025%。D965 的平均 TTFT、P95 和联合达成率分别退化约 35.3%、43.3% 和 1.275 个百分点；因此“就绪边界重规划”候选不能封存为 PrimeLoRA 的性能最优版本。

这不是正确性失败：D965 完成合同通过，且机制顺序真实执行。当前可证伪的解释是，重规划本身未消除 worker 启动期间形成的排队/扩容尾延迟；D965 还触发了更多扩容事件（13 对 11），并把大量请求暴露在扩容受影响窗口。下一轮必须先分解这些可观测阶段，再提出一个新的、单一的性能假设；不能在没有新证据时继续堆叠重规划补丁。

## 下一步

1. 保留 D965 原始 JSON、`main_outcome.json`、watchdog 与 worker 证据，不覆盖 D931 或 D964。
2. 离线核对 D931/D965 的扩容事件、dispatch/admission wait、GPU-ready、远程 miss 和请求级阶段分布；先确认退化是否来自扩容控制、请求排队或生命周期计量，而不是把它归因于 tier planner。
3. 若提出新的优化，只运行新的 execution key；通过同一 W2 合同完整回放后，只有在 G1/G2 指标和物理生命周期计量均一致时才可封存。
4. 在 7B 候选未达到冻结目标前，不推进外部 baseline 或 3B 主比较。

