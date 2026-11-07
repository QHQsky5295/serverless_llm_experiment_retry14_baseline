# D1120：7B 普通 locality 与完整 adapter 管理的资源—时延取舍

2026-11-08。W1 的一次完整运行和离线分析已完成。Full 直接复用主比较
D1115，ElasticOnly 复用 D1119；本次只新增已登记的 ElasticLocality，
没有重跑参照、改变负载、修改生产算法或搜索 slack。采用 analyze-results
的观测—解释—主张边界结构，负向结果与未证明事项均保留。

## 1. 比较合同

三者均为当前 D1100/max4 非目标配置：Llama-2-7B、vLLM 0.30.0、FP16、
TP=1、14 slots/seqs、1024 batch tokens、同一 1–4 实例伸缩规律。
W1 使用既有 4000 请求、500 个逻辑 adapter、seed42 的参数化变换，
共同业务前60秒通知、固定输出生成合同和真实远端已发布 gzip 交付。
不在请求中打包；传输和必要物化时间保留。全程未改网络或共享 Kubernetes。

ElasticLocality 是 PrimeLoRA 的复合消融，不是外部论文 baseline。
它保留原生按需缓存、普通文件 LRU、物理可行性/预留/引用；关闭主动
handoff、hierarchy 和软 admission。路由使用预先登记的 GPU-hit 优先和
slack=1 负载回退。其 admitted-only 负载排序不同于 ElasticOnly 的
(admitted, pending, GPU utilization)，不声称二者严格只差一个 locality bit。

Full 与本次4000条输入逐一匹配；ElasticOnly 通过同一 Full 来源和已封存
输入检查复用，三者输入一致。Full 与本次有 **173条输出摘要不同**；
Full 与 ElasticOnly 的179条差异沿用原记录。未做 ElasticOnly 与本次的
直接输出摘要比较。差异请求及摘要全部保留，不未经验证归因于浮点舍入。
原始 `n_correct=null` 不补写：原生 token、adapter/内容/lease 绑定检查
均通过，但不冒充独立数值正确性证明。下面资源分母为 offered request。

## 2. 完整数值与判定

冻结 V1＋单轮 V2、D394 共同长度分组 SLO；单轮已知工作负载观测，
不是独立留出测试，不计算 CI 或声称统计显著性。

| 指标 | Full D1115 | ElasticOnly D1119 | ElasticLocality D1120 |
|---|---:|---:|---:|
| 成功且原生合同完整 | 4000 | 4000 | 4000 |
| TTFT 达标 | 3963 | 3999 | 3991 |
| TPOT 达标 | 3978 | 3988 | 3982 |
| 联合达标 | 3942（98.550%） | 3987（99.675%） | 3977（99.425%） |
| 生命周期 GPU·秒 | 7603.874678 | 8575.626704 | 8497.621705 |
| GPU·秒/offered request | 1.900969 | 2.143907 | 2.124405 |
| 平均 TTFT（ms） | 988.900 | 557.648 | 531.908 |
| P95 TTFT（ms） | 2331.330 | 1035.577 | 1092.998 |
| 平均 TPOT（ms） | 36.319 | 27.603 | 27.437 |
| P95 TPOT（ms） | 56.037 | 46.536 | 45.543 |
| 平均 E2E（ms） | 5141.061 | 3532.721 | 3511.346 |
| P95 E2E（ms） | 10955.810 | 7469.208 | 7456.818 |

1. Full 比 ElasticLocality 少用893.747027 GPU·秒，即 **10.5176%**，
   但联合达标低0.875个百分点，P95 TTFT 高113.2967%。在共同 timing
   门槛成立及上述正确性限定下，这是 G1 方向的资源优势，不是全指标领先。
2. ElasticLocality 比 ElasticOnly 少用0.9096% GPU 时间，但联合达标低
   0.25个百分点，P95 TTFT 高5.5449%。这一小差异只有单轮描述意义；
   普通 locality 没有在本次把 Full 的约10.5%资源差解释掉。
3. 三个 max4 工作点均超过冻结 W1 预算6756.571050 GPU·秒，故此表
   不能支持 G2 预算内尾延迟优越性。cap3 的 D1117/D1118 是另一工作点，
   不拼接它们的成本和本表时延。

## 3. 机制证据及不能作出的归因

| 机制证据 | Full | ElasticOnly | ElasticLocality |
|---|---:|---:|---:|
| 实际 selector 重算一致 | 4000 | 4000 | 4000 |
| file 准备计划 | 1292 | 0 | 0 |
| GPU 准备计划 | 1287 | 0 | 0 |
| HOST 准备记录 | 1018 | 0 | 0 |
| residency epochs | 1529 | 0 | 0 |
| initial activation | 1 | 1 | 1 |
| natural scale-out | 19 | 10 | 11 |
| 自然缩容完成 | 17 | 8 | 8 |
| dispatch GPU tier | 2590 | 2351 | 3183 |
| dispatch HOST tier | 435 | 539 | 144 |
| dispatch NVMe tier | 840 | 978 | 540 |
| dispatch Remote tier | 135 | 132 | 133 |
| 实际远端传输 | 132 | 132 | 132 |
| 线上字节 | 131480060 | 131480060 | 131480060 |

本次12个 activation 全部为 no_handoff、准备 disabled、无 plan SHA，
主动准备/驻留计划为零。4000次实际选择均由原始快照重算验证，不能只
凭配置开关就认定消融生效。Full 的主动机制计数不等于独立加载次数。

普通 locality 的 GPU-tier 命中明显增加，但不自动转化为尾延迟或资源
全面改善。三者传输次数和字节相同，**本点不支持“更少远端下载”归因**。
dispatch Remote tier 是选择时状态，不等于最终下载次数，不能把133
个 Remote dispatch 写成133次传输。

Full/ElasticLocality 各自实际快照中的 Full-versus-locality shadow
选择分歧分别2741/1965。沿用的 ElasticOnly 2537是 Full-versus-load-only
分歧，比较对象不同，不把三项混作同一个量。shadow 仅比较选择，未生成
反事实时延。这一复合消融不替代 LastKnown 的 confirmed-state 隔离实验。

## 4. 阶段和 first-service

既有 D455 分析器验证两组4000条原生绑定，E2E 分解最大误差均为0ms。
本次 GPU 时间分解为业务前48.209884、业务窗8409.620861、drain
2.268568、清理37.522393 GPU·秒；Full 业务窗为7539.095342。
资源差主要在业务期间，不是忽略准备或延迟释放造成的虚假节省。

Full/ElasticLocality 平均 runtime-slot 等待为274.531/136.173ms，
admission 至 engine-entry 为239.654/105.812ms，engine-entry 至
native 队列为179.651/73.403ms；平均 native decode 为3955.958/
2906.540ms。完整阶段见 stages/summary.json。它们是闭环运行下的
观测阶段，不能当作某个控制机制的独占 CPU 开销，也不能事后相减造出
“优化后的实验”。后续单因素对照用于分离来源。

Full 的19个和 ElasticLocality 的11个 natural epochs 均有真实第一条
dispatch 及首 token；无歧义或 unused epoch，initial 各1、controlled
均0。按原定义报告，不额外重复同配置凑20样本；受控 A4 仍待完成。

## 5. 来源、收尾与下一步

原始目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1120_20261108/7b_elastic_locality_w1`。
结果 SHA `1584c087c2d7748f6dde9a8b22166ae617c9b6d71fb043146ba791baffed4e2e`。
配置 SHA `af32ca772e43fc20546be23f52b744180710257887cf7730c6c2976732939cf1`。
运行前备份 commit `7efef8de9bed03d96c02a3fdcab64a9377d833b7`。
curated 目录：`paper_results/ieee_tc/ablation/d1120_7b_elastic_locality_w1/`。

00:59:26运行终止，service/replay/watchdog均0，12个物理 lease 全释放，
service cgroup 已移除，GPU计算进程为空。只停止本次拥有的远端服务，
重新读取为 inactive/MainPID=0。2245个资源样本无 high/max/OOM/kill，
服务 swap 为0，服务内存峰值19760107520B，主机可用最低104014622720B。
远端132条传输全匹配，无未关联传输、请求期打包或本地模拟 fallback。
matching-clock journal 双端 SHA
`8ade8b056daf8f585349bcc2b907c49cc441052dce192ba534c0157c589a3a18`。

分析复用 D1113、D299、D455 和 D1119/D1057，只调整本次 comparison
脚本的变体、实际 locality selector 和两个封存参照。执行在显式
CPU2,3,26,27、3/4GiB、swap0 的 CPU scope；没有新 GPU 性能执行。
预登记23项目标测试通过。最终监测投影遇到终止记录不含 sample 的
KeyError；它只是只读观察程序，真实运行已成功，随后以 launch/lease/
census 证据确认释放。封存哈希核对首次遇到 list/dict 来源格式差异，
修正投影后重查；未修改或重跑任何分析结果或性能数据。

来源、旧投稿144个文件和用户 dirty manifest 的复核回执见 closure_manifest。
不 stage 用户 manifest、既有 dirty 状态全文或大体积原始日志。

本键首次有效完整观测封存。下一项为 **同版本7B FixedFull W1**：保留
Full 管理机制，只固定四实例常驻，核对共同准备和完整持卡计量，复用
D1115 主表 Full。之后继续 LastKnown/其他单因素、累积节点及机制证据。
不因本次尾延迟差而削弱消融，不重复 Full/基线，不切回3B。A1–A5、
S1–S13、成本/CE/Pareto、审稿文档和最终图表仍未全部完成。
