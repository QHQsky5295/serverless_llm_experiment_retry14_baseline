# Baseline 发表核验与“最优”的准确含义

2026-10-14。用户最新问题的原始来源核查；不修改冻结指标 V1、单轮 V2、
任何历史运行身份或比较集合。阅读时与完整 PLAN、EXECUTION_STATUS 合用。

## 1. 当前比较对象及正式发表位置

| 系统 | 正式发表位置、年份 | 本项目的角色与边界 |
|---|---|---|
| vLLM / PagedAttention | ACM SOSP 2023 | 同后端常驻参照；实际使用 vLLM 0.30.0，不冒称复现了 2023 年全部原始实现 |
| S-LoRA | MLSys 2024 | 外部 Multi-LoRA 系统，原生专用后端 |
| ServerlessLLM | USENIX OSDI 2024 | 外部 serverless 系统，必要适配及历史轮询修复另行披露 |
| HydraServe | USENIX NSDI 2026 | 外部 serverless 系统；本机测得的是明确适配范围，不等于所有论文机制都已触发 |
| Loquetier | NeurIPS 2025 主会 | 外部 Multi-LoRA 系统；本机静态分片部署不是原生集群调度贡献 |
| dLoRA | USENIX OSDI 2024 | 已批准的 3B W1/W2 有限最近邻对照，不是已完成的 7B W0 行 |
| ElasticLocality | 不适用 | 自建强内部对照，非外部论文，保留以隔离弹性和普通 locality 收益 |

PrimeLoRA 是被测系统，不列为已发表的外部 baseline。SGLang 不恢复进矩阵；
Chameleon、ELORA、Toppings 不因本次核查自动成为新增性能实验。

上述六个外部系统均有正式会议论文依据，非仅 arXiv。SOSP、OSDI、NSDI、
NeurIPS 是对应领域公认旗舰会议；MLSys 是机器学习系统领域的重要顶级专门
会议。但“顶会”没有跨学科唯一官方定义，不能据此声称六项具有相同的 CCF
分级。若投稿需要某一具体目录，须另外按目录版本核查，而不是编造等级。

原始来源（已直接核验，不以搜索摘要代替发表记录）：

- [vLLM 0.30.0 官方引用](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/README.md)：
  Efficient Memory Management for Large Language Model Serving with PagedAttention，
  SOSP 2023；[ACM DOI](https://doi.org/10.1145/3600006.3613165)。ACM 正文入口本次返回 403，
  年份与 venue 同时由作者官方版本 README 的 BibTeX 核实。
- [S-LoRA，MLSys 2024 论文集](https://proceedings.mlsys.org/paper_files/paper/2024/hash/906419cd502575b617cc489a1a696a67-Abstract-Conference.html)。
  网页题名为 SLoRA: Scalable Serving of Thousands of LoRA Adapters，所链 PDF
  为 S-LoRA: Serving Thousands of Concurrent LoRA Adapters；保留该题名差异。
- [ServerlessLLM，OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/fu)，135–153 页。
- [HydraServe，NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/lou)，415–430 页。
- [Loquetier，NeurIPS 2025 主会论文集](https://proceedings.nips.cc/paper_files/paper/2025/hash/5ebc22174f6dff6fa7c2447d0dfa31ea-Abstract-Conference.html)，
  DOI 10.52202/085713-2199；不能因为同时存在 arXiv 而误称未发表。
- [dLoRA，OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang)，911–927 页。

发表地位不证明本机复现正确，也不证明任何系统在新场景一定表现好。特别是
Loquetier 等明显差结果，仍须保留已做适配审计和未覆盖机制说明；发现实际
执行缺陷就修正受影响运行，不能用“原论文已发表”或“基线不调优”代替核查。
不能为了符合“全部外部系统已发表”而删除更强的内部 ElasticLocality 对照。

## 2. 相关论文实际评价什么

| 原始论文 | 已核验的评价重点 | 可借鉴之处与不可混同之处 |
|---|---|---|
| HydraServe，NSDI 2026 §8.2–8.4、Fig.9/13 | 冷启动 TTFT、TTFT SLO 达成、TPOT、GPU memory-time 成本及取舍 | 原文明确承认 TPOT 代价；不是要求每个指标都第一，也不是本项目的整卡 GPU-s 定义 |
| ServerlessLLM，OSDI 2024 §7.1 | 主要报告模型启动延迟，迁移/抢占时加上暂停延迟 | 指标对应快速加载与迁移贡献；不把它误引为主要按联合 SLO 排名的论文 |
| dLoRA，OSDI 2024 §7.1 | 总请求 E2E 之和除以总输出 token 的延迟指标 | 衡量服务效率，不等于我们的逐请求 TPOT 均值 |
| AlpaServe，OSDI 2023 | 在服务时延约束下支持更大请求率/突发性 | 是约束优化的先例，不是新的本机 baseline 或我们的 95% 协议来源 |

来源：[HydraServe 原文](https://www.usenix.org/system/files/nsdi26-lou.pdf)、
[ServerlessLLM 原文](https://www.usenix.org/system/files/osdi24-fu.pdf)、
[dLoRA 原文](https://www.usenix.org/system/files/osdi24-wu-bingyang.pdf)、
[AlpaServe 官方论文页](https://www.usenix.org/conference/osdi23/presentation/li-zhouhan)。
本次核查这些相关章节，不声称系统性穷尽全部文献。research-lookup 的 Parallel
CLI 0.7.1 无认证，使用可用网页工具核验官方论文/论文集，未新增凭据或环境。

## 3. 保持既定 G1/G2，不改成新的综合冠军分数

首先必须完成同一全部工作负载，原生 token、adapter、时间、物理资源证据有效。
独立数值正确性缺失仍记未知，不能将 HTTP 成功或数量正确当作数值证明。

\[
A_{\rm joint}=N^{-1}\sum_r z_r\mathbf1[TTFT_r\le\theta_r^T]
\mathbf1[TPOT_r\le\theta_r^P],\qquad
U=\sum_d\int\mathbf1\{\text{服务持有 GPU }d\}\,dt.
\]

单 token 按冻结协议只检查 TTFT；失败为零，仍留在 offered 分母中。

- **G1：**在全部正确且共同联合 SLO ≥95% 的可行集合中，最小化
  `R=U/N_correct`。跨 W0/W1/W2 按冻结协议等权，单 W0 排名只代表 W0。
  95.8% 和 99.5% 都满足门槛，并不意味着质量相同；全部原始达成率和尾部
  继续报告，但不能临时将“达成率最高”变为 G1 的首要排序。
- **G2：**全部正确、联合 SLO ≥95%、`U≤0.75 U_ref` 且共同资源包络内，
  比较 W1/W2 P95 TTFT（跨场景选择用固定参考归一化）。另外逐对报告联合
  SLO 差值；若达成率较低，就不能声称相对该对手“服务质量全面不退化”。
  单轮只作观测比较，不作统计非劣或显著性结论。
- CE、成本、goodput、good requests/GPU-s 和 Pareto 是补充，不替换门槛和目标。

**GPU-s 最低不是 GPU utilization 最低。**前者是减少物理卡被占用的总时间，
后者可能恰好意味着更多空转；利用率用于定位瓶颈，不作为待最小化的主目标。

## 4. SLO 最高与 GPU-s 最低能否同时达到

可能，但没有保证。消除重复搬运、控制阻塞或无效准备可能同时减少持卡与排队；
更早准备、保留更多热副本也可能以更多持卡换取更高 SLO。若两点一项好一项差，
它们是取舍关系，不存在不依赖目标的唯一冠军。已冻结 G1/G2 正是给取舍提供
事先约束，而非看到结果后选择有利指标。

只有在相同合同/资源条件下，PrimeLoRA 的 SLO 不低、U 不高（至少一项严格
改善），才可在这两个维度称观测支配；加上尾延迟主张时还必须检验尾延迟。
共同预算是共同上限，不能把消耗远超预算的系统包装为同预算胜者，也不能
省略它的原始数据。不同 `resource_opt` / `budget_latency_opt` 不拼成虚构配置。

### 与行业评估实践的关系：原则一致，不冒称标准认证

[MLCommons 的 MLPerf Inference 5.1 小模型说明](https://mlcommons.org/2025/09/small-llm-inference-5-1/)
采用 token/s，并对在线场景设置 TTFT/TPOT 限制；同一 Llama3.1-8B 的 server
与 interactive 场景就使用不同的时延要求。这说明约束取决于模型、场景和
目标应用，不存在可直接套到任意 serverless LoRA 系统的唯一阈值。
[DistServe，OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
以同时满足 TTFT/TPOT 要求的 goodput 为核心，比较每 GPU 的有效服务能力。
[vLLM 0.30 的 bench serve](https://docs.vllm.ai/en/v0.30.0/cli/bench/serve/)
也允许配置 TTFT/TPOT/E2E goodput 约束。它们支持“服务质量约束＋有效服务
或资源效率”的方向，不支持用无约束的低 GPU 时间奖励慢服务或漏请求。

因此最终评估体系继续是：

1. 相同输入、实际输出量、adapter 正确性、完整 offered 分母和失败报告；
2. 同一绝对服务要求，分别报告 TTFT、TPOT 和联合达成率；
3. G1/G2 分别回答达标服务的生命周期资源代价、共同预算内的尾延迟；
4. 并列报告 P50/P95/P99 时延、token/s、`N_good/T_obs`、`N_good/U`；
5. 在既定 S4–S13/成本分析中展示负载、SLO 倍率、90/95/99%要求、预算与
   Pareto 敏感性，检验排序是否只由一个约束点决定。纯离线评价复用已有日志，
   不据结果更改在线配置或回写主阈值，不新增重复运行和 CI。

第4项不把 `goodput / 总成本` 当作 requests/$；按冻结文件使用一致窗口。
主测量取原生 token 边界，而部分工业工具取客户端流式事件，定义差异须保留。
不能称本实验已按 MLPerf 的模型、数据、LoadGen、准确性和全部规则提交，
也不能称95%/75%/5×/2×是通用行业规范。满足行业常见可比性原则与取得某一
标准认证是不同声明。即使所有点达标，也只称“已测场景和工作点的观测最优”，
不称所有模型/硬件/负载的全局最优。

### 对应 PrimeLoRA 创新点的预期收益

confirmed-state routing 应减少选错副本后的准备等待；handoff 应改善扩容后
首次服务；层级驻留应减少真实远端获取和重复加载；admission 应减少加载
与 decode 的争用。若这些机制有效，它们可以改善联合 SLO / P95 TTFT，并
通过减少服务驻留时间或所需副本减少 U。因此借鉴 HydraServe 的服务要求、
尾延迟与资源取舍展示，并用整卡生命周期 U 对齐本机资源模型，是合理迁移。

这些是可检验的预期，不是让指标预先保证 PrimeLoRA 获胜。管理开销、额外
预取和预留也可能增加 U；D512 正是在检验这个反方向。对齐创新点应体现在
A1–A5/S1–S3 因果对照和机制事件，而非只选有利请求、删除内部强对照或改变
冻结阈值。继续以领先为优化目标，保留当前未领先事实与每次改进的副作用。

## 5. 当前数据给出的结论和主线

已封存 D510 七系统比较（来源 `paper_results/ieee_tc/main_comparison/d510_7b_w0/comparison.json`）：
PrimeLoRA 97.425% / 2.694612 GPU-s/绑定请求，ElasticLocality 95.775% /
2.473179，ServerlessLLM 98.125% / 4.022441，vLLM 99.475% / 4.028844。
S-LoRA、Loquetier、HydraServe 在该点未满足共同联合 95% 要求，不进入达标
资源排名，但保留全部结果。此处为完整原生合同的开发观测，数值正确性边界
见封存 manifest；不是已完成所有正式资格、所有场景和全局最优的声明。

PrimeLoRA 比 ElasticLocality 的 SLO 高1.65个百分点、P95 TTFT低约4.35%，
但 GPU 时间多约8.95%。因此在当前 W0 可比观测中 **G1 尚未第一**，不能因
内部对照没有顶会论文而将它删掉。对常驻参照的低资源优势也不能写成 SLO
最高。W1/W2 及预算参考尚待完成，不能借 W0 宣告 G2。

HydraServe 4000 请求已完成并纳入上述比较，当前转回 PrimeLoRA 符合用户的
先完成 baseline 再优化次序。D512 是该完整比较后的当前控制开销诊断，不是
跳过 HydraServe。接下来针对已观测的冗余控制成本选择一项语义保持优化，
最小验证后完整回放；继续保留 7B W1/W2/G2、3B、dLoRA B6、A1–A5、S1–S13、
成本分析、两轮审稿文档和最终图表。不恢复 CI、重跑择优或改评价尺子。
