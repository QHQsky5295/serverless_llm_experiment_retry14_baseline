# D1164：7B baseline 合同复核与主比较边界

日期：2026-11-09。本文更新 D1133 的 7B 闭口台账，纳入随后完成的
ServerlessLLM D1139 full replay。只做来源和合同的离线归类，不启动 GPU、不修改
baseline、不修改 PrimeLoRA、不重命名历史结果。

## 1. 统一判断标准

主比较需要同时具备：

1. 4,000 请求完整终态、共同 `fixed_length_greedy_v1` native token 合同；
2. 与 PrimeLoRA 相同的请求/adapter/target/prompt/arrival 身份；
3. 共同 deployment notice + 60 s 到达时基；
4. 可映射到同一物理 GPU 生命周期账本；
5. 联合 TTFT/TPOT SLO，而不是单独 TTFT 计数。

缺少其中任一项时，结果仍保留，但只能进入相应的服务质量或诊断层，不能把
它包装成 G1/G2 资格点。联合 SLO 低于 95% 的系统也不删除，而是保留为真实
观测并标明“当前工作点不可行”。

## 2. 当前 7B baseline 分类

| 系统 | 运行 | 当前用途 | 关键边界 |
|---|---|---|---|
| ServerlessLLM | D1139 | 完整服务质量 R2 观测 | 4,000/4,000、native contract 完整；replay 以 stack ready 后的 process start 为时基，且没有 `physical_gpu_resource_summary_v1` |
| vLLM | D1106 | R2 诊断观测 | vLLM 0.10.2、ready 后 replay、生命周期字段与共同物理账本不一致 |
| S-LoRA | D1108 | R2 诊断观测 | 95/100 GiB 外部内存包络、ready 后 replay、TPOT/资源口径不属于当前共同合同 |
| HydraServe | D579/D582 | 已适配 published baseline 观测 | 物理 GPU-s 可审计，但 W1/W2 联合 SLO 分别为 49.425%/66.000%，不进入达标资源排名 |
| Loquetier | D587 | 已适配 published baseline 观测 | W1 联合 SLO 45.050%，保留其长尾和静态分片边界，不删除或人为惩罚 |

完整字段及来源见
[`baseline_contract_matrix.csv`](../../paper_results/ieee_tc/main_comparison/d1164_baseline_contract_review_20261109/baseline_contract_matrix.csv)。

## 3. ServerlessLLM D1139 的具体裁决

D1139 完成了 4,000/4,000 请求、native vLLM token-ID 计数和 W2 输入身份核对；
平均 TTFT 为 1,695.67 ms，P95 TTFT 为 5,794.45 ms，服务阶段平均 TTFT 为
412.04 ms，平均 dispatch/admission wait 为 1,283.63 ms。该结果清楚表明其
端到端长尾主要来自启动和 dispatch 队列，服务阶段本身不能替代端到端 TTFT。

但 D1139 的 replay origin 是 `process_start`，没有绑定 PrimeLoRA 的共同
deployment notice + 60 s 绝对到达时基；其资源为 reconstructed lifecycle GPU-s，
没有真实 allocation/release census。因此它可以回答：

* ServerlessLLM 在相同请求内容和生成合同下的端到端服务质量如何；
* ServerlessLLM 的 dispatch/cold-start 阶段如何贡献尾延迟。

它不能回答：

* ServerlessLLM 与 PrimeLoRA 在 G1 物理生命周期 GPU-s 上谁更省；
* 在严格共同到达时基下的 G2 尾延迟是否支配 PrimeLoRA。

不能通过事后平移 arrival、扣除 cold-start，或把 reconstructed GPU-s 改名为
physical GPU-s 来修补这些差异。D1139 因而保留为完整服务质量 R2 观测，而不是
伪造的 G1/G2 主表行。

## 4. 与 PrimeLoRA 主工作点的关系

PrimeLoRA D1115、D1117、D1118 均有共同到达时基、物理 GPU 生命周期账本和联合
SLO；D1117/D1118 还分别提供 W1/W2 的预算内工作点。它们是当前 7B 主比较的
资源参考，但不意味着外部 baseline 的不完整合同可以被忽略。

因此论文表格应分层呈现：

1. **主 G1/G2 表**：只放合同和物理计量均匹配的工作点；不可行 baseline 用
   “未观测到联合 95% 可行点”表示，不删除。
2. **服务质量诊断表**：放 D1139/D1106/D1108 的完整 TTFT、TPOT、E2E、SLO 和
   阶段分解，并明确其 arrival/resource 限制。
3. **published-baseline 适配表**：放 HydraServe、Loquetier 的完整回放、适配
   条件和低 SLO 结果，说明其机制边界，不把低 SLO 改写成失败复现或人为调低。

这比把所有系统强行压进一张混合分数表更可复核，也符合 G1/G2 的分维度评价。

## 5. 下一步

本复核关闭“7B 已有 baseline 结果是否被误标为共同主比较”的文档缺口。它不要求
重跑 D1139、D1106 或 D1108，也不要求重新部署共享 Kubernetes。下一重型任务应
按计划进入已登记的 7B 机制/敏感性缺口；若确实需要 G1 的 ServerlessLLM 物理
资源对照，必须单独登记一个带 physical allocation/release 证据的新执行键，
而不能从 D1139 的 reconstructed 数值推导。

PrimeLoRA 的后续优化只有在新对照证据揭示明确、可证伪的 G1/G2 瓶颈时才启动；
不会为了让 CE、平均延迟或某一辅助指标排名更好而重跑已封存点。

