# D1132：7B 受控证据边界与下一执行键登记

## 目的

本记录只追加当前 7B 证据状态，不覆盖历史结果，也不把自然触发、诊断运行或
信息受限运行包装为严格因果实验。所有数值仍遵循单轮观测协议；`n_correct` 未
被重新证明时，不形成正式质量—资源排名。

## 已封存的 A1/M2 结果

| 证据 | 当前判定 | 可支持的论文语义 |
|---|---|---|
| D1117/D1118 | 完整的 W1/W2 `CapacityOnly` 预算工作点 | 在已测预算点，容量门控有明确资源—尾延迟—联合 SLO 权衡；不是全局最优证明 |
| D1127 | `NoHOST` W1 完整运行 | 可用于区分 HOST 层级开关的自然闭环影响 |
| D1128 | `CapacityOnly` W1 完整运行 | 与 D1127、D1129、D1131、D1115 组成同一 W1 累积消融链 |
| D1129 | `RoutingOnly` W1 完整运行 | 可用于 routing 开关的增量比较 |
| D1131 | `RoutingHandoff` W1 完整运行 | handoff 事件和真实层级加载已观测，但单项未显示净尾延迟收益 |
| D1115 | `Full` W1 锚点 | 仅作为同协议 Full 参照，不因单次结果宣称全局最优 |

D1131 的累积链已固定为：

`ElasticOnly → RoutingOnly → RoutingHandoff → CapacityOnly → Full`。

该链显示：`RoutingOnly` 相对 `ElasticOnly` 减少了 15.98% GPU-s，但联合 SLO
下降 2.05 个百分点且 P95 TTFT 更高；`RoutingHandoff` 相对 `RoutingOnly`
增加 3.09% GPU-s、P95 TTFT 增加 12.48%；`Full` 相对 `CapacityOnly` 减少
4.61% GPU-s，联合 SLO 增加 0.30 个百分点，而 P95 TTFT 基本持平。因此，当前
证据支持“完整组合在该工作点改善资源效率且不牺牲联合 SLO”，不支持“每个机制
单调改善”或“handoff 单项必然改善尾延迟”。

## A4 readiness 证据边界

- D278+D279 是同一 7B 协议下的 8-phase readiness 诊断，合计记录 20 个
  first-service 观察；它们是自然 scale-out/first-service 证据，不是新的 Full
  性能排名点，且 `n_correct` 仍未知。
- D550 是 100 请求的初始部署、一次 controlled activation 和原始连续前缀的
  控制路径验证；它证明字段和生命周期边界可记录，但明确不是主性能比较。
- 因此，当前可写成“dispatch 前快照和 first-service 绑定在既有诊断中可追溯”，
  不能写成跨重复运行的 dispatch-time 因果收益。

## S2 confirmed-tier 证据边界

D1122 的 `Full` 与 `LastKnown` 保持同一 workload 和后端，显示受限信息视图会
改变少量最终选择（29/4000），并带来资源、联合 SLO 与 P95 的权衡；它可以支持
“confirmed tier state 具有可观测决策影响”。但是，该运行没有把 cache 位置、候选
集合和所有 admission 输入冻结成完全相同的受控对照，所以不能单独支持“tier
state propagation 是唯一因果来源”。论文应保留这一限定，并将严格同缓存/同位置
对照列为尚未封闭的证据。

## S3 proactive-admission 证据边界

现有 native capacity-wait、D238 语义复核和 D600 并行压力诊断证明了：容量等待、
候选准备时延和完整 W2 结果可以分离观测；但尚未形成“同一候选、同一初始状态、
同一触发条件，仅 Full admission 与 CapacityOnly admission 不同”的性能成对运行。
因此，不能把这些诊断写成 admission 的净收益或排序证据。若继续执行，必须使用
真实 in-flight demand 触发 typed admit/defer/reject，记录 candidate/admitted/
deferred/completed 及 downstream delay；没有合法候选时应报告 not exercised，
不能人工注入 sleep、压力或合成收益。

## 下一唯一执行键

当前不重复 D1131、Full、A1 或任何外部 baseline。下一执行只能在以下两项中选择一
项并登记唯一 key：

1. `7b_s3_proactive_admission_diagnostic_w1`：小规模、非排名、真实 in-flight
   demand 的 admission 诊断；或者
2. `7b_s2_same_cache_position_diagnostic_w1`：冻结 cache 位置、候选集合和
   admission 输入，仅切换 confirmed/last-known tier 视图。

在缺少满足上述冻结条件的 runner 之前，不启动大型 4000-request 性能运行；先做
入口检查和资源预算登记。该限制用于避免用“看起来更像受控”的自然运行替代真正
的因果实验。

## 追溯

- D1131 结果：`docs/ieee_tc/D1131_7B_ROUTING_HANDOFF_W1_RESULT_20261108.md`
- D1131 比较：`paper_results/ieee_tc/ablation/d1131_7b_routing_handoff_w1/comparison/comparison.json`
- D278/D279 readiness：`docs/ieee_tc/D278_7B_READINESS_POST_D277.md`、
  `docs/ieee_tc/D279_7B_READINESS_REPLICATE_20261006.md`
- D1122 LastKnown：`docs/ieee_tc/D1122_7B_LASTKNOWN_W1_RESULT_20261107.md`
- D238 admission 复核：`docs/ieee_tc/D238_ADMISSION_SEMANTIC_REASSESSMENT.md`
- D600 并行诊断：`docs/ieee_tc/D600_PARALLEL_PRESSURE_DIAGNOSIS.md`

