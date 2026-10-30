# D963：就绪边界需求重规划验证（7B、100 请求诊断前缀）

日期：2026-10-30  
代码基线：`5c20c505` 加入本轮未封存的就绪边界重规划改动  
输入：既有 W2 trace、既有 500-adapter subset、`fixed_length_greedy_v1`，未生成新工件或新负载

## 目的

D931/D961 的请求级证据显示，扩容触发时冻结的准备前缀与运行时真正就绪后的首个请求之间存在可观测需求漂移。D963 只验证新代码能否在不读取未来 trace 的情况下运行；它不是正式性能点，也不用于 G1/G2 或与历史结果比较。

实现语义是：扩容触发时仍先冻结一次 Eq. (4)--(7) 计划，以便文件阶段与模型启动重叠；在真实回放且新副本的 worker 已启动后，取消并结清尚未发布的 pre-init epoch，重新读取当前 hotness、native source 和 shared-file owner，执行一个新的 handoff epoch，完成后才把副本置入可路由的 `running` 状态。旧 epoch 的 hash/state 保留在 activation record 中。该重规划不使用未来请求、不改变论文公式、不调用 legacy forecast/handoff prefix。

## 运行结果

- 诊断前缀：100/4,000 请求；仅一个初始副本；0 次 scale-out，因此本轮没有触发就绪边界重规划。
- 请求合同：100/100 完成，0 失败，固定长度/native token 合同通过。
- 观测值：平均 overall TTFT 612.19 ms，P95 overall TTFT 1,336.22 ms；TTFT SLO 100%；P95 comparable TTFT 1,336 ms；生命周期 GPU-s 535.13。
- 这些数值只描述该诊断前缀，不能外推到 4,000 请求，也不能与 D931 的完整回放作优劣结论。

结果文件：

`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d963_20261030/7b_route_overlap_validation_100/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d963_7b_late_replan_validation_100.json`

## 代码验证

针对 `ScenarioRunner._add_ieee_dedicated_instance` 的 targeted launch tests 中，既有 full/initial/no-handoff 路径保持原行为；在 synthetic owner fixture 中，`natural_scaleout + active replay` 会产生两次准备调用（trigger epoch 与 ready-boundary epoch），activation event 使用第二个 plan hash，副本在 fresh epoch 完成后才进入 `running`。本地完整 launch test 文件仍有两个与本改动无关的环境/历史问题：一个既有测试使用同步 `Mock` 替代异步 host-retirement 方法，另一个测试依赖本机可用 NVML；它们在父提交 `5c20c505` 上同样失败。

## 下一步

由于 100 请求前缀没有自然扩容，本轮不能验证首服务命中率变化。只有在同一 W2 输入和同一配置下触发自然扩容的诊断前缀或完整回放中，才能检验：

1. 首服务请求是否在 fresh ready-boundary epoch 的已确认 tier 上执行；
2. 首服务 GPU/LOCAL 命中率和首服务 TTFT 是否改善；
3. 额外等待是否被降低的 remote miss 与扩容尾延迟抵消；
4. 物理 GPU-s 是否仍满足 G1 的比较口径。

在上述证据出现前，不封存该候选、不画论文图，也不启动 baseline 主矩阵。
