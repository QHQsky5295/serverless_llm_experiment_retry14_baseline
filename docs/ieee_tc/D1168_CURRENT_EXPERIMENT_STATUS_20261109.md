# D1168：PrimeLoRA → IEEE TC 当前实验进度（论文证据状态）

日期：2026-11-09。本文档是当前阶段的追加式进度记录，不覆盖历史台账、
结果、图表、用户 dirty 文件或任何实验协议。

## 1. 当前主线判定

PrimeLoRA 已完成“可以进入同合同 baseline 对比”的阶段，但尚未完成“在已测
范围内全面优于对照”的阶段。这里的两个阶段必须分开：联合 TTFT/TPOT SLO
达到 95% 只是使工作点具备 G1/G2 比较资格，不是系统优越性的结论。

当前不再无证据地重复 PrimeLoRA Full，也不因平均 TTFT、CE 或单个 TPOT 指标
变化而另立目标。后续优化必须由 baseline 对照或受控机制证据指出具体瓶颈，
并且每个新执行键只运行一次、保留退化结果。

## 2. 已完成并可用于论文的证据

### 7B PrimeLoRA

- D1114/D1115/D1100：W0/W1/W2 Full 均达到共同联合 SLO；
- D1117/D1118：W1/W2 的 cap3 预算工作点达到联合 SLO，并记录了物理
  生命周期 GPU-s 与 P95 TTFT 的资源—尾延迟关系；
- D1127–D1131：NoHOST、CapacityOnly、RoutingOnly、RoutingHandoff 和
  DelayedHandoff 的当前 A1 累积消融已经完整封存；负向取舍不被删除；
- D1122/D1165：confirmed 与 LastKnown 信息视图的当前 7B 观测差异已封存，
  但严格同 cache 位置/同候选的 S2 因果对照仍未完成；
- D278+D279/D1166：dispatch-time readiness 证据完整，两个相同协议运行合计
  20 个真实 first-service 观察；该证据不是新的性能排名点；
- D1163：当前 PrimeLoRA CE/idle-factor 敏感性已经离线整理，CE 只作补充指标。

### 3B PrimeLoRA

- D1083/D1082/D1084：当前实现的 W0/W1/W2 Full 均达到共同联合 SLO；
- D1085/D1087/D1088/D1089：当前实现的 ElasticOnly、ElasticLocality、
  LastKnown、NoHOST 机制对照已经有完整 W1 观测；
- D1167：3B 当前 Full 与已封存 vLLM、ServerlessLLM、S-LoRA、HydraServe、
  Loquetier、dLoRA 观测已统一整理。PrimeLoRA 在已达标工作点的 GPU-s 低于
  vLLM/ServerlessLLM，但 P95 TTFT 并非最低；HydraServe/Loquetier/dLoRA 的
  低资源行若未满足共同服务要求，不作为达标资源胜出证据。

### 远端交付与统一指标

`docs/ieee_tc/REMOTE_DELIVERY_METRIC_POLICY_20261029.md` 已冻结以下口径：
请求触发的远端读取、网络发送/接收、本地物化和缓存/层级竞争计入 TTFT/E2E；
研究开始前一次性发布/压缩/哈希不计入请求时间，但单独记录；远端无关维护、
备份、扫描和额外打包不得安排在测量期。所有系统共享发布集合、输入、链路
和请求合同，系统自身合法的 miss、传输次数和驻留差异保留。

## 3. 当前可写的主结论

在已测 7B/3B 工作负载中，PrimeLoRA 已显示“满足共同服务要求时的较低物理
GPU 生命周期占用”这一方向性证据；在部分工作点还具有较好的联合达成率。
但当前证据不能写成所有指标均第一，也不能把较低 GPU-s 与较高 P95 TTFT
折算为一个 CE 分数后宣称全面优越。G2 仍需在共同预算内比较实际 P95 TTFT
和联合达成率，且 baseline 的可比物理资源合同尚未全部闭合。

## 4. 正在推进的主问题

1. **7B 受控机制证据**：S2 需要在同 cache 位置、候选和 admission 输入下
   只切换 confirmed/LastKnown 信息视图；S3 需要真实 proactive candidate
   下比较 Full 与 CapacityOnly 的 admission 和延期成本。不能用自然闭环或
   人工 sleep 代替这两个实验。
2. **baseline 同合同闭合**：ServerlessLLM、vLLM、S-LoRA、HydraServe、
   Loquetier 的已有观测保留；需要把缺少共同物理生命周期或输入时间基线的
   运行标为 R2/诊断，而不是把重建的 GPU-s 混进 G1。
3. **G2/规模敏感性**：在不改变主协议的前提下，补有限的共同预算、GPU 规模、
   到达强度、带宽和 churn 工作点；不按每个点重新调参，也不从单点插值制造
   资源—质量前沿。

## 5. 尚未完成

- 7B 严格 S2；
- 7B 真实 proactive-admission 的受控 S3；
- baseline 的统一物理 GPU 生命周期合同和所有必要预算点；
- 3B 的 G2/有限规模敏感性及剩余机制证据；
- S1–S13 的剩余敏感性、CE/Pareto 最终汇总、两轮审稿证据整合；
- 论文最终图表和 IEEE TC 正文修改。

## 6. 数据保护与当前执行状态

本阶段没有新的 GPU 回放，没有重新生成 LoRA 工件、trace 或 adapter subset，
没有操作共享 Kubernetes，也没有修改远端发布缓存。D1166–D1168 只追加文档，
旧 `final_v2/`、旧图表和用户 dirty 文件保持不变。下一次 GPU 运行前应重新
检查推理机磁盘/内存和远端 owned service 状态，但不增加与科学问题无关的门禁。

