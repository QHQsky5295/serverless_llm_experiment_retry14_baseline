# D1165：A3 confirmed-tier state propagation 的证据裁决

日期：2026-11-09。本文将 D1122 的 LastKnown 单因素回放纳入最新 7B 闭口台账。
不启动新的 GPU 运行，不修改 Full、LastKnown、路由器、驻留或准入实现。

## 1. 实验是否真正隔离了状态传播

D1122 与 Full D1115 共享 Llama-2-7B、4,000 请求、W1 trace、500-adapter 集合、
`fixed_length_greedy_v1`、vLLM 0.30.0、GPU/CPU 资源边界、handoff/residency/admission
路径和共同到达时基。唯一目标变量是路由器排名所见的信息视图：

* Full 使用带确认来源的历史快照；
* LastKnown 对 tier/representation/footprint 使用 0.5 s request-triggered
  sample-and-hold；
* 可行性、pending、utilization、EWMA、owner、registry、planner、admission、
  reservation 和选中后的来源重新验证保持 live；
* LastKnown 不保存可执行授权，也不改变实际 dispatch tier。

因此它确实比“有没有分层缓存”更接近审稿人要求的对照：两侧都有相同层级缓存，
只改变路由器使用的状态新鲜度。它不是全局瞬时物理真值的证明，也不是所有系统
路由语义的反事实模拟。

## 2. 观测

| 指标 | Full D1115 | LastKnown D1122 | Full − LastKnown |
|---|---:|---:|---:|
| 完成且合同匹配 | 4,000 | 4,000 | 0 |
| 联合 SLO | 98.550% | 98.225% | +0.325 pp |
| 生命周期物理 GPU-s | 7,603.875 | 7,898.096 | −3.725% |
| 平均 TTFT (ms) | 988.900 | 949.563 | +39.337 |
| P95 TTFT (ms) | 2,331.330 | 2,308.963 | +22.367 |
| 平均 E2E (ms) | 5,141.061 | 4,989.729 | +151.332 |
| 平均 TPOT (ms) | 36.319 | 35.695 | +0.624 |
| 最终路由选择分歧 | — | 29/4,000 = 0.725% | — |

LastKnown 的视图确实改变了少量选择：4,107 次路由事件中有 29 次最终选择
与同状态 received-view shadow 不同；可行候选的来源类也发生了变化。与此同时，
Full 少用 3.725% GPU-s、联合达成率高 0.325 个百分点，但 LastKnown 的平均和
P95 TTFT 略低。两者均超过 95%，且都超过 W1 的 G2 预算，因此该结果是资源—质量
取舍，而不是 Full 在所有指标上的支配。

视图年龄也被实际记录：controller hold age 的 P95 约 0.208 s，而底层来源
观测年龄 P95 约 13.723 s。故 0.5 s 是本次替代视图的采样保持参数，不是物理
tier 状态年龄上界，也不能写成系统的最优刷新周期。

## 3. 能支持的论文结论

该回放支持一个有限的机制结论：**即使驻留、准入和来源重新验证保持不变，路由器
使用的 tier 状态新鲜度也会改变实际候选选择、资源持有和服务质量。**这把“具有
分层缓存”与“传播已确认层级状态”区分开来，并回应了审稿人要求的隔离方向。

但证据不支持以下更强表述：

* 不能声称 confirmed-state propagation 单独带来普遍的 TTFT 或 GPU-s 改善；
* 不能声称其他系统都没有类似状态传播；
* 不能把 29 次选择差异外推为全部 4,000 请求的主导原因；
* 不能把 Full 与 LastKnown 的单轮差异写成统计显著或全局最优。

因此 A3 的执行和“信息视图确实改变决策”子结论可以闭口；“该机制解释
PrimeLoRA 全部 headline 收益”的因果主张仍不成立，应在正文中收窄为有限机制证据。

## 4. 与 A2/A4 的关系和下一动作

D1159 已表明 handoff 固定计划能够被完整重放，但不产生净性能收益。D1122 则
直接隔离了状态视图变化，却显示资源和尾延迟方向相反。两者共同说明不能把计划
可执行性或状态传播的存在，自动改写为全面性能优势。

下一项应沿计划进入受控 A4：在实际 activation/first-service 时间线上绑定每个
epoch 的第一条 dispatch，分别报告 trigger→ready、arrival→first token 和
activation→token；若继续比较 handoff，必须冻结同一 plan hash/目标/预算。不得
为了使 Full 与 LastKnown 拉开差距而增加等待、清缓存或修改消融语义。

来源：
[`D1122 result`](D1122_7B_LASTKNOWN_W1_RESULT_20261108.md)、
[`D1159 A2 comparison`](D1159_A2_MECHANISM_COMPARISON_20261109.md)、
[`D1164 baseline review`](D1164_7B_BASELINE_CONTRACT_REVIEW_20261109.md)。

