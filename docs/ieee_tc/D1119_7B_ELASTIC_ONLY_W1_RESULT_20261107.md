# D1119：7B ElasticOnly 与同版本 Full 的资源—服务质量取舍

2026-11-07。一次完整 W1 观测已经结束并释放资源；没有重复 Full、重新生成
负载、重跑基线或修改服务算法。本文使用 analyze-results 的“观测—解释—
主张边界”结构；全部反向差异保留，不作单轮显著性或全局最优主张。

## 1. 对照身份和一致性

Full 直接复用主表 D1115，配置 D1100/max4；ElasticOnly 为预登记 D1119。
不是以 cap3 的另一工作点拼接 Full 成本。两者非目标配置相等：同 vLLM
0.30.0、FP16/TP1、14 slots/seqs、1024 batch tokens、1–4 replicas 和
同一伸缩规律。只改变已批准的主动 adapter 管理组合及独占输出/缓存路径。
真实远端已发布 gzip、60秒通知、W1到达、生成与共同资源包络不变。

两组4,000条输入逐一匹配：源请求、adapter内容、canonical prompt、原生
prompt token摘要、目标长度、生成合同、实际计划到达。输出长度均合约完成；
**179条输出摘要不同**，request ID及两侧摘要保存在 comparison.json。
不未经验证将其归因为舍入，也不因长度匹配就宣称独立数值正确性已证明。
原始 `n_correct=null` 保留，下面资源分母为 offered request。

原始D1119：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1119_20261107/7b_elastic_only_w1/`

结果SHA：`9d06cf7d69ff99085fa6d3f6f9902d4b5ccbb5b22edfe216871cd6576c542ca3`。
Full来源SHA：`99574e407543545937b93d0277c7c70a4d3c28eb8107290d79db65aa1777e577`。
分析产物：`paper_results/ieee_tc/ablation/d1119_7b_elastic_only_w1/`。
`result1/` 为原生指标/交付/资源核查，`comparison/` 为输入和机制比较，
`stages/` 为复用既有阶段分析器的离线分解。没有第二次GPU性能执行。

## 2. 冻结口径下的完整数值

使用冻结V1＋单轮V2及7B D394共同长度分组SLO；不使用终端5000ms指标。

| 指标 | Full（D1115） | ElasticOnly（D1119） |
|---|---:|---:|
| 成功且原生合同完整 | 4,000 | 4,000 |
| TTFT达标 | 3,963 | 3,999 |
| TPOT达标 | 3,978 | 3,988 |
| 联合达标 | 3,942（98.550%） | 3,987（99.675%） |
| 物理GPU·秒 | 7,603.874678 | 8,575.626704 |
| GPU·秒/offered request | 1.900969 | 2.143907 |
| 平均TTFT（ms） | 988.900 | 557.648 |
| P95 TTFT（ms） | 2,331.330 | 1,035.577 |
| 平均TPOT（ms） | 36.319 | 27.603 |
| P95 TPOT（ms） | 56.037 | 46.536 |
| 平均E2E（ms） | 5,141.061 | 3,532.721 |
| P95 E2E（ms） | 10,955.810 | 7,469.208 |

观测：双方满足95% timing门槛时，Full少占971.752026 GPU·秒，即降低
**11.3316%**。代价是联合达标低1.125个百分点，P95 TTFT高125.1237%。
因此本点支持G1方向的“达标条件下减少物理资源”，不支持“所有延迟/SLO
都更好”、统计非劣或各机制单独有效。严格正确性限定仍按上一节披露。

冻结W1预算是6,756.571050 GPU·秒；这两个max4点都不满足G2预算，不能
把它们的尾延迟比较写成同预算G2胜负。已完成cap3工作点D1117/D1118保持
独立身份，不替换本次Full。

## 3. 机制实际执行，而不只是开关存在

| 证据 | Full | ElasticOnly |
|---|---:|---:|
| 按实际selector重算一致的请求 | 4,000 | 4,000 |
| 实际路由排序 | service bins | load-only |
| file准备计划 | 1,292 | 0 |
| GPU准备计划 | 1,287 | 0 |
| HOST准备记录 | 1,018 | 0 |
| residency epochs | 1,529 | 0 |
| initial activation | 1 | 1 |
| natural scale-out | 19 | 10 |
| 自然缩容完成 | 17 | 8 |
| 远端传输次数 | 132 | 132 |
| 线上字节 | 131,480,060 | 131,480,060 |
| dispatch时GPU tier | 2,590 | 2,351 |

ElasticOnly的11个activation均为no_handoff、准备disabled且无plan SHA；
保留普通按需加载、物理可行性和缓存。没有发现主动管理泄漏，不因它延迟
更低而人为禁用正常缓存或额外加等待。Full有1,328次有类型的主动admission
判断，其中1,327 admit、1次defer_effective_capacity；不能把计划/attempt
数量冒充实际独立加载次数或节省时间。

远端获取次数和字节完全相同，故**本场景不能把收益归因于更少远端下载**。
Full选中GPU-tier更多，但这不自动转化为更低用户时延。其到达窗占用为
7,539.095342 GPU·秒，ElasticOnly为8,500.048547；资源差主要落在业务窗，
不是靠忽略准备或清理。Full有更多伸缩，但平均持卡更少，符合资源与延迟
取舍；这不是对路由、handoff、hierarchy、admission各自因果贡献的分离。

复用D1057同状态selector重算：Full已观测快照上两种selector有2,867次
选择分歧，ElasticOnly快照上为2,537次。这些是各自实际状态下的shadow
选择，不生成另一条路径的反事实TTFT，更不代替LastKnown状态传播实验。

## 4. 阶段解释与首服务证据

既有阶段分析器在两组均验证E2E加和及原生计时误差0ms。平均TTFT差
431.253ms分布在多个阶段：runtime-slot等待差134.608ms、admission至
engine-entry差96.422ms、engine-entry至native队列差112.792ms；其余
差异见完整stages文件。阶段包含等待/执行重叠的背景影响，不把上述任何
一项称为某单机制的独占CPU开销，不直接扣掉它来构造“优化后性能”。

first-service复用D389定义：每activation的第一条实际native dispatch，
不按成功筛样本。Full19个、ElasticOnly10个natural epochs均有dispatch
和首token；initial各1个，controlled均0。没有歧义或unused epoch。
样本不足20如实记录，不按旧规则增加重复凑样本；受控A4仍是独立未完成项。

## 5. 收尾、验证和下一主线

D1119服务/回放/watchdog返回码均0，11个lease全部释放，实际GPU计算进程
为空；只停止本次拥有的远端服务，复查inactive、MainPID=0。2,245个资源
样本无high/max/OOM/kill、无服务swap；服务峰值19,766,685,696 bytes，
主机可用最低102,548,250,624 bytes。远端132条请求级传输全匹配，无未关联
传输、请求期打包或临时归档。matching-clock journal双端SHA为
`93f4ec19a36fd8b7a8a305f344e94d119200bf39102841f2735a4c201e7381ff`。

30项离线目标测试通过。现有D1113、D299、D455只扩展显式scenario，默认
Full与冻结公式不变；D1057机制分析调整复用。两次CPU启动问题保留记录：
首次直接执行D299脚本缺模块搜索路径，改为`python -m`；首次机制分析
`AllowedCPUs`未使进程affinity符合检查，改为显式taskset。均在分析输出
创建前终止，不涉及性能重跑。早期CPU统计虽受内存限制，不能把所请求的
AllowedCPUs冒称已核验affinity；最终机制/阶段/测试使用显式CPU2,3,26,27。

43个独立来源SHA重新核验一致，D1115原封存的4个汇总文件未改变；144个
旧投稿保护文件与用户dirty manifest均未改变。提交仅限本次明确分析脚本、
测试、文档及小型curated结果，不stage用户manifest或历史dirty状态全文。

本消融键按首次有效观测封存，不重跑求更有利结果。下一项按原A1推进
**同版本7B ElasticLocality**，复用既有slack=1定义及正常按需缓存，与
D1115/D1119核对非目标一致性；随后其他单因素/累积节点和状态传播机制。
此处有明确G1资源差，尚无理由为追求全指标同时领先而改Full或削弱消融。
剩余A1–A5、S1–S13、成本/CE/Pareto、两轮审稿文档和最终图表均未删除。
不重启已完成基线复现、不切回3B、不宣告全目标完成。
