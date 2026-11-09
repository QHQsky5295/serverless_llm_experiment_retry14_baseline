# D1159：A2 受控回放的机制对照与论文使用边界

日期：2026-11-09  
分支：`retry14_continuous_queue_v2`  
实验对象：Llama-2-7B、4,000 请求、500-adapter 池、W1 trace、
`fixed_length_greedy_v1` 生成契约  
用途：记录 d1150 Full capture 与 d1158 fixed-schedule replay 的同负载机制对照；不产生新运行，不生成图表。

## 1. 比较问题

A2 要回答的是一个控制问题：在请求 trace、adapter 子集、生成契约和规划输入保持一致时，
“同一组 handoff 计划和扩缩容提交前沿”能否在另一次真实执行中被完整重放，而不被回放器
重新观察到的需求窗口或串行化规则改写。

因此，d1150 是 Full 的计划捕获运行，d1158 是使用 d1150 的 handoff bundle 和固定
activation schedule 的受控回放运行。该对照用于验证机制路径和实验可审计性，不把两次运行
之间的性能差异解释为 handoff 机制的净收益。

## 2. 冻结条件与资格

两次运行共享：

- 相同的 4,000-request trace，SHA-256：
  `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`；
- 相同的 500-adapter subset，SHA-256：
  `aa94b21e1295aefde664e5b29c030a9e42e02af946369bfe9ae15336b89b3016`；
- 相同的 `fixed_length_greedy_v1`、prompt/token 上限和 vLLM 后端配置；
- 相同的 handoff bundle（20 个 activation plan，`complete=true`）和
  d1152 fixed activation schedule（36 个事件）；
- 相同的物理 GPU 资源边界和生命周期计量口径。

d1150 与 d1158 的 launcher 均返回成功。两次运行均完成 4,000/4,000 请求、失败数为 0，
且没有中断请求。d1158 消费了全部 36/36 个固定 schedule 事件和 20/20 个 handoff
计划，其中 19 个 scale-up、17 个 scale-down；所有物理 allocation journal 均释放，
没有遗留 open lease。

当前 launcher 将这一路径标记为 exploratory（`formal_run=0`），且部署摘要不会仅凭
token contract 推断独立的数值正确性。因此 d1150/d1158 都不是 held-out 主排名点；它们
是可审计的控制输入和机制执行证据。

## 3. 提交前沿是否被保持

捕获运行最早的三个并发 scale-up 在请求 18、20、24 提交，提交时已有 activation pending；
这正是旧回放器在 d1156 中错误串行化、导致计划前沿漂移的原因。修正后的 d1158 在相同
请求前沿提交了这三个事件，并继续按 d1152 的事件顺序执行后续 scale-up/scale-down。

固定 schedule 冻结的是动作类型、请求索引前沿和目标实例数；每次实际 activation 仍重新经过
当前 owner、内容、容量、reservation 和资源释放检查。它没有注入等待、伪造 GPU lease，
也没有把物理失败转换成成功。个别日志中的请求索引若落在轮询边界相邻位置，只作为采样
时间的记录差异，不被当作计划重规划，也不用于构造性能优势。

这排除了 d1156 暴露的回放器串行化错误：d1158 的完整执行不再把捕获的 18/20/24
提交前沿推迟到前一个 worker ready 之后。

## 4. 观测结果

| 指标 | d1150 Full capture | d1158 fixed replay | d1158 - d1150 |
|---|---:|---:|---:|
| 完成请求 | 4,000 | 4,000 | 0 |
| 失败请求 | 0 | 0 | 0 |
| 平均 TTFT (ms) | 1,015.46 | 1,096.33 | +80.87 |
| P95 TTFT (ms) | 2,793.71 | 2,785.17 | -8.54 |
| 平均 E2E (ms) | 5,128.32 | 5,377.39 | +249.07 |
| P95 E2E (ms) | 11,000.42 | 11,394.47 | +394.05 |
| 平均 TPOT (ms) | 35.86 | 37.36 | +1.50 |
| 联合 SLO 达成率 | 99.20% | 99.375% | +0.175 pp |
| 生命周期 GPU-s | 9,673.98 | 9,674.79 | +0.81（约 +0.008%） |
| active GPU-s | 7,907.61 | 7,943.93 | +36.31 |
| idle-ready GPU-s | 1,602.49 | 1,571.68 | -30.81 |
| cache hit rate | 96.575% | 96.650% | +0.075 pp |
| GPU hit rate | 64.025% | 64.525% | +0.500 pp |
| 平均 LoRA I/O (ms) | 105.57 | 116.23 | +10.66 |
| 平均 dispatch admission wait (ms) | 383.10 | 415.02 | +31.93 |
| P95 dispatch admission wait (ms) | 1,331.81 | 1,254.07 | -77.74 |
| scale-up 影响请求平均 TTFT (ms) | 1,214.12 | 1,368.20 | +154.08 |
| scale-up 影响请求 P95 TTFT (ms) | 3,040.20 | 3,854.78 | +814.58 |

请求级配对分析与上表一致：d1158 相对 d1150 的总体 TTFT 平均增加 80.87 ms，
E2E 平均增加 249.07 ms；两次运行的生命周期 GPU-s 几乎相同，且 d1158 并未呈现
稳定的延迟改善。服务 TTFT 平均增加 48.95 ms，dispatch admission wait 平均增加
31.93 ms，说明这次受控回放没有把“相同计划”误写成“更快执行”。

## 5. 可以写入论文的结论

这组实验支持以下有限结论：

1. PrimeLoRA 的 handoff 计划能够在真实 GPU 资源检查仍有效的条件下跨运行保存、重新绑定
   当前 owner，并完整消费固定的扩缩容动作序列。
2. 在存在并发 pending activation 时，按提交前沿重放是必要的；把 scale-up 强制串行化会
   改变扩缩容事件顺序，并造成错误的 A2 对照。
3. 计划捕获、固定 schedule、物理 reservation 和资源释放均有可审计记录，d1158 因此可
   作为“机制可执行且回放不发生时序漂移”的控制证据。

正文或附录中的推荐表述是“controlled mechanism replay”或“cross-run execution
control”，而不是“handoff lowers latency/cost”。若需要一句更强的边界说明：
“The controlled replay validates execution fidelity of the handoff path; it is not a
performance claim.”

## 6. 不能从这组实验推出的结论

- 不能声称 handoff 本身降低了 TTFT、E2E 或生命周期 GPU-s；d1158 的平均 TTFT/E2E
  实际略高于 d1150，GPU-s 基本不变。
- 不能将 d1150/d1158 当作 held-out 主比较结果，也不能用它们替代 Full、ServerlessLLM
  或其他 baseline 的主工作点。
- 不能据此声称全局最优性；固定 schedule 只冻结了一个已观测的提交序列，未证明规划器
  或 tier 选择是全局最优。
- 不能把略高的 SLO 达成率解释成系统质量改善，因为这是单轮控制回放，且延迟分布的
  其它部分并未同步改善。

## 7. A2 的后续使用决定

本记录完成 A2 的跨运行控制证据登记。后续不再因为 d1150/d1158 的轻微性能差异盲目
重跑或绘图；A2 若进入论文，应放在机制正确性/附录位置，并与主比较中的 G1（共同服务
要求下生命周期 GPU-s/request）和 G2（共同预算下 P95 TTFT 及联合 SLO）分开。下一步回到
7B PrimeLoRA 主线，按计划完成主结果与 ServerlessLLM 基线复核；本记录不改变旧结果、不
覆盖历史目录。

