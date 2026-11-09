# D1160：7B 主线结果的论文级证据整理

日期：2026-11-09  
用途：把已经完成的 7B Full、机制/消融和 A2 受控回放区分开，建立后续主比较应使用的结果身份；不启动新 GPU 运行。

## 1. 当前判断

7B 的运行工作并不是没有完成：Full、budget、routing、handoff、residency、HOST
和 capacity 变体均已有完整 4,000-request 观测，A2 的 capture/replay 也已完成。
但“已有完整运行”不等于“最终论文主结果已经闭口”。原因有两个：

1. 7B 结果同时存在请求级原始结果和物理 allocation union 的计量字段，不能把
   legacy lifecycle 字段与物理 GPU-s 交叉使用；
2. 许多变体达到共同 SLO，但 W1 预算约束和 P95 TTFT 的组合仍表现为资源—尾延迟
   取舍，不能仅凭某一个指标宣布 Full 全面胜出。

因此，当前阶段应是“7B 主表的 canonical source 整理与外部比较”，而不是继续
重复 A2 capture，也不是无限制地只对 PrimeLoRA 自身调参。

## 2. 可用于论文叙述的 7B 工作点

下表使用已封存的 curated 结果或对应结果报告中的物理 allocation union；所有点
均来自同一 7B source42/W1 工作负载，4000 个请求均完成。它们是描述性单轮观测，
不是运行间统计推断。

| 工作点 | 联合 TTFT/TPOT SLO | 生命周期物理 GPU-s | P95 TTFT (ms) | 论文用途 |
|---|---:|---:|---:|---|
| Full（D1115） | 98.550% | 7603.875 | 2331.330 | 主系统参考点 |
| NoHandoff（D1124） | 97.575% | 7931.749 | 2647.273 | handoff 消融 |
| R0H0（D1125） | 99.275% | 8577.319 | 1717.326 | 无驻留/无 handoff 对照 |
| DelayedHandoff（D1126） | 97.800% | 7864.276 | 2454.768 | 准备时机机制对照 |
| RoutingOnly（D1129） | 97.625% | 7204.901 | 1868.048 | 路由单机制对照 |
| RoutingHandoff（D1131） | 97.500% | 7427.712 | 2101.167 | 累积机制对照 |

这些结果支持的是分维度结论：Full 相对 NoHandoff/DelayedHandoff 改善了尾部或
资源的某些组合；RoutingOnly 的 GPU-s 更低但 Full 的联合达成率更高；R0H0 的
P95 TTFT 更低但资源更多。它们不支持把所有指标压缩成一个“全面最优”结论。

## 3. 计量字段不能混用

D1115 的论文结果报告给出的物理 allocation union 为 7603.874678 GPU-s，
而同目录原始 JSON 的 legacy `infra_gpu_seconds_total` 字段为约 9636 GPU-s。
这两个数并非同一物理量：前者是去重后的实际 GPU 持有区间并集，后者包含旧版
lifecycle 计量语义。D1126 的 Full 对照、D1131 的累积比较以及后续 7B 主表均应
沿用 D1115 curated 结果中的物理 union，不能把 9636 字段混入主表。

同理，A2 capture/replay 的 9674 GPU-s 结果不能回填 Full 主表。它的固定 schedule
和 handoff bundle 是为了验证跨运行计划控制，而不是为了替代主 Full 工作点。

## 4. A2 capture 与主线的关系

- d1150：在一次 Full 运行中捕获 20 个 activation handoff plan；
- d1152：把捕获到的 scale-up/scale-down 动作转换为 36-event fixed schedule；
- d1158：在真实 GPU 执行中完整重放 36/36 个事件和 20/20 个计划。

这组实验排除了自然回放因 autoscaler 时间分叉而改变提交前沿的问题，证明 handoff
路径可执行且可审计。它没有改变主 workload、没有增加新的 adapter 集合，也没有
证明 handoff 带来 TTFT 或 GPU-s 改善，因此应放在机制证据或附录，不作为 7B 主表
的 Full 新版本。

## 5. 7B 主线的论文状态

### 已完成

- 7B Full、主要消融和机制变体的完整请求回放；
- W1 下的联合 SLO、P95 TTFT 和物理 GPU-s 观测；
- A2 受控 handoff capture/replay；
- 结果之间的资源—质量取舍已经可直接写入实验分析。

### 当前阶段

将上述结果统一绑定到 canonical curated 来源，随后与已完成的 vLLM、ServerlessLLM、
S-LoRA、HydraServe 和 Loquetier 工作点进行同协议比较。该步骤的目标是确定 PrimeLoRA
在 G1（满足共同服务要求时的生命周期 GPU-s）和 G2（共同资源预算下的 P95 TTFT及
联合达成率）上的真实位置，而不是继续重复相同的 Full 或 A2 运行。

### 尚未完成

- 7B 全部外部 baseline 的最终 G1/G2 统一主表；
- W1 budget 约束下的正式 G2 闭口；当前 Full 的 7603.875 GPU-s 高于约 6756.571
  GPU-s 参考预算，因此已有 W1 点不能直接宣称 G2 胜出；
- 7B 其余 workload/带宽/规模敏感性；
- 3B 最终统一比较、限定 dLoRA 工作点、A1–A5 和 S1–S13 的完整论文交付；
- 最终图表和论文正文修改。

## 6. 下一步决定

不再运行 capture/replay，也不以 A2 结果替代 Full。下一项工作是完成 7B canonical
主比较表；只有该表显示 PrimeLoRA 在 G1 或 G2 存在明确缺口时，才提出一个与日志和
已有系统论文机制相对应的单一优化假设。这样可以在已有实验基础上推进论文主线，
避免把“完成了很多运行”误写成“已经证明全面最优”。

