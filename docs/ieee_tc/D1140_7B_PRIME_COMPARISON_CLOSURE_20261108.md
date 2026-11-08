# D1140：7B PrimeLoRA 主比较与自优化阶段闭口

2026-11-08。本文基于已封存的 7B Full、预算工作点和 published baseline
记录，按最新的 G1/G2 闭口规范进行离线复核。不启动新的 GPU 回放，不修改
PrimeLoRA、baseline、输入、阈值或历史结果。

## 1. 判定规则

- G1：在联合 TTFT/TPOT 达成率至少 95% 的完整工作点中，比较物理生命周期
  GPU-s/request；同时披露实际 joint 和 P95 TTFT。
- G2：在同一预算上限内，比较 P95 TTFT，并单独检查 joint 是否不低于对照。
- CE、平均延迟和美元成本不替代 G1/G2。
- 本文只讨论已测 7B 配置域，不宣称理论全局最优，也不把 ServerlessLLM
  D1138 的 R2 诊断结果混入主比较。

## 2. PrimeLoRA 已封存工作点

| 工作点 | Run | Joint SLO | GPU-s/request | P95 TTFT (ms) | 75% budget |
|---|---|---:|---:|---:|---|
| W0 Full | D1114 | 98.850% | 2.635 | 1,868.3 | 是 |
| W1 Full | D1115 | 98.550% | 1.901 | 2,331.3 | 否 |
| W2 Full | D1100 | 98.050% | 2.629 | 2,815.1 | 是 |
| W1 Cap3 | D1117 | 97.575% | 1.636 | 2,469.5 | 是 |
| W2 Cap3 | D1118 | 97.500% | 2.468 | 3,053.0 | 是 |

这些工作点均满足共同 95% 服务要求；D1117/D1118 是预算约束下的独立
工作点，不能与 max4 Full 拼接成不存在的配置。

## 3. 与已测外部点的逐场景结论

### W0

PrimeLoRA D1114 相对 vLLM D423 少用约 34.59% GPU 时间，但 P95 TTFT 高于
vLLM，joint 也低于 vLLM。该点支持“共同服务要求下的资源效率收益”，不支持
全面时延支配。ServerlessLLM、S-LoRA、Loquetier 和 HydraServe 的 W0 点均不
同时满足共同 95% 要求与完整资源合同，不能用它们的条件成本替代 G2 对照。

### W1

PrimeLoRA D1115 相对已登记的 vLLM、ServerlessLLM、S-LoRA、Loquetier 和
HydraServe W1 点，同时具有更低的 GPU-s/request、更低的 P95 TTFT 和更高的
joint。该结论限定在这五个已测点和 W1 trace，不外推到未测硬件或其他 workload。

PrimeLoRA D1117 是 W1 的预算工作点，joint=97.575%，GPU-s/request=1.636，
P95 TTFT=2,469.5 ms。现有外部 W1 点没有同时满足共同 95% 要求且位于同一
75% 预算上限内的可行对照，因此不能把空的外部可行集合写成“G2 已证明全面
优于所有系统”。

### W2

PrimeLoRA D1100 相对 vLLM D557 少用约 34.75% GPU 时间，但 P95 TTFT 高
约 60.74%，joint 低约 0.175 个百分点。vLLM 的物理 GPU 时间高于 W2 的
75% budget 上限；因此 PrimeLoRA 的 D1100 支持预算前提下的资源效率，不能
据此声称它在同一 budget 下已经击败 vLLM 的尾延迟。D1118 是预算内工作点，
其实际 P95 为 3,053.0 ms、joint=97.5%。

## 4. 对自优化阶段的闭口决定

**决定：7B PrimeLoRA 自优化执行阶段闭口，转入机制证据和剩余计划。**

理由是：

1. W0、W1、W2 已有完整 Full 工作点，且均达到共同 95% 服务要求；
2. W1 已出现对已测外部点的三主指标共同优势；
3. W0/W2 已出现共同服务要求下的物理资源优势；
4. W1/W2 已有预算工作点，能够报告预算内服务质量和实际 joint；
5. 当前剩余问题是 W0/W2 的资源—尾延迟取舍及外部预算可行点不足，不是
   一个已经定位、可以靠重复 Full 解决的实现错误。

因此不再为把 vLLM 的低尾延迟“调平”而盲目重跑 PrimeLoRA，也不修改评价
阈值、削弱 baseline 或从不同工作点拼接一条虚构 Pareto 曲线。若后续出现
新的、可证伪的瓶颈证据，只允许登记一个有明确假设的优化候选，并保留本
闭口结果作为原始版本。

## 5. 尚未关闭的主张与唯一下一动作

本闭口不等于 PrimeLoRA 已在所有指标、所有模型和所有场景全面最优。以下
问题仍开放：

- confirmed tier propagation 的受控因果隔离；
- handoff/first-service 的同计划证据；
- admission 对加载争用及延期成本的完整证据；
- S1–S13 中尚未完成的层级、churn、规模、带宽和预算敏感性；
- 3B 的对应主比较与机制证据。

下一项沿计划推进 7B 受控机制证据（A2/A4、S2/S3），然后再进入已登记的
预算/规模/敏感性缺口；不切换到 3B，也不重复已封存的 7B Full 或外部 baseline。

## 6. 来源与限制

- 输入、点值和资格字段来自
  `paper_results/ieee_tc/metric_review/closure_20261108/result1/points.csv`；
- 7B 主线闭口与消融边界见 `D1133_7B_CLOSURE_LEDGER_CURRENT_20261108.md`；
- G1/G2 解释见 `CLOSURE_CRITERIA_REASSESSMENT_20261108.md`；
- `n_correct=null` 的历史点仍只表示原生合同和服务计量完成，不能改写成独立
  数值正确性证明；
- 所有结果为单轮封存观测，不计算运行间置信区间或统计显著性。
