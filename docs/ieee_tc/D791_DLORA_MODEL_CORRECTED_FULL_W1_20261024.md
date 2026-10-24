# D791：模型兼容修正后的 dLoRA 3B 完整 W1

2026-10-24。唯一完整性能运行已结束，保留真实远端、原4000请求/500-ID池、
DP2/TP1/PP1、EngineManager/PERIOD_MIG及既定性能参数。仅使用D788模型语义
视图和独立运行目录；HTTP配置与D786字节相同，view SHA为
`8a7fa829c4902a761efcf58c765580cbdb5e0f367e3c2245ea43417409e9494f`。
实际offered ID为132、目标输出458224 tokens。没有生成新工件/负载，未调参。

## 为什么必须重新测量

D786/W1及D787/W2后来确认存在Llama-3.2 tied-head/llama3 RoPE模型兼容错误，
旧结果保留为R3，不用于证明PrimeLoRA优势，也不以其为合法性能基准计算提升。
D788修正遵循模型配置，D790的两个既有prompt/adapter共6个prefill/decode点
与参考greedy token全部一致；这是有界数值证据，不是完整4000请求的独立
数值证明。具体来源与误差见D788_DLORA_MODEL_COMPATIBILITY_20261024.md。

D791沿用该视图及原生后端，不将参考PEFT overlay放入性能环境。实际执行
MAIN为`c833cde62cfae38edf8f3b9debc0d0a1f0d6cddb`，BASE为
`cf421793e7b71342cf75ff063f8d64afe26d8169`；二者在启动前已备份核验。
`formal_run=false`及原数值未知字段不追改；这是已知工作负载上的开发期单轮。

## 完整结果与资格

| 观测量 | 数值 |
|---|---:|
| 计划/到达/提交/终态请求 | 均4000 |
| 原生计量与绑定可核对的响应 | 3402（85.05%） |
| 超时失败 | 598（14.95%），全部TimeoutError |
| 有效响应输出tokens | 382057，逐请求与目标一致 |
| 成功服务adapter ID | 124；offered132；部署准备500 |
| TTFT达标 | 30/4000 |
| 联合TTFT/TPOT达标 | 19/4000＝0.475% |
| 条件TPOT达标 | 2413/3402＝70.928865% |
| 生命周期GPU时间 | 6262.210177884437 GPU-s |
| 条件平均/P95 TTFT | 447545.943830 / 1481872.366592 ms |
| 条件平均/P95 TPOT | 279.106589 / 158.701799 ms |
| 条件平均/P95 E2E | 474759.875128 / 1551328.249658 ms |
| 原生时间、E2E、TPOT最大重算误差 | 均0 ms |

时延分布仅针对3402条有效响应；失败保留在4000的offered分母中。
`n_correct=null`及独立全工作负载数值证明未知保持；不能将HTTP成功或数量相符
升级为已证明的完整数值正确性。G1/G2均不可行，不进入达标资源排名。
GPU-s/offered为1.5655525444711094，GPU-s/绑定有效响应为1.840743732476319，
都不是数值正确请求的资源最优结论。单轮无CI，不称显著、稳健或全局最优。

互斥资源窗：pre-arrival=0、arrival=4181.426366370171、
drain=2062.1499672103673、cleanup=18.633844303898513 GPU-s，合计等于U。
CPU-only准备不虚构持卡；终态后的真实释放仍计入。实际service/replay退出码1，
watchdog0，GPU context release和service path移除已核验，不把收尾成功当运行成功。

## 解释范围：后端等待，而非简单归因远端或纯计算

成功响应的平均E2E分解为：提交4.152664ms、提交后dispatch等待23452.783646ms、
service TTFT424089.007520ms、decode27198.516660ms、完成通知15.414638ms。
这里dispatch是既定后端边界；service TTFT还包含原生内部排队、调度、加载与
prefill，不是纯GPU执行时间。后半负载同样明显等待：序号3000–3499全部500条
响应，router queue均值60.445431ms、service TTFT均值467063.049951ms。
该切片是事后描述，不作为无启动独立主实验，也不证明单一机制的因果贡献。

原生日志严格匹配得到43次ILP求解，总37.51206040382385s，最大3.330630302429199s；
43条迁移决策打印、364次请求提及。打印发生在实际搬移之前，不冒称364次
迁移成功，累计求解span不与请求时间相加减，不能单独解释全部长等待。

严格正常Finished记录3403个ID；其中req_01000为客户端超时，日志先Aborted
（7122行）后Finished（7123行），没有补记为成功。另11个missing-stream ID
全部属于已超时请求，且各自均有更早Aborted记录。当前证据不支持将它们
描述为“丢失正常成功响应导致超时”；取消之后的实际资源仍包含在U中。
输入、绑定和计量核查通过不等于对所有内部正确性的全称证明。

## 远端、资源及保存

复用D80通知前只读gzip发布；500次部署准备获取、线上1160474851bytes、
物化19482573296bytes，请求中动态打包0次。真实传输及本地准备保留在时间线，
未减去累计span构造结果。链路启动核验1000Mb/s/full，不写成物理100G。

服务内存峰值77313208320bytes、high3310、max/OOM/kill均0，swap峰值
1896730624bytes；主机available最低35989266432bytes，磁盘最低
142706237440bytes。软限制回收/换页属于实测条件，不因此作废或扣除时延。
本地和远端实验服务已关闭，54项Ray日志、完整请求/失败与远端证据保存。

离线统计首次误用缺NumPy的/usr/bin/python3，未改运行数据；原错误和回执保留。
第二次复用D786已合格的primelora_vllm0300_tc_20260925环境及同一analyzer，
`full4000_w1 / complete_with_failures`通过。只有一次GPU性能运行，两次离线统计。
分析环境选择错误不作为系统性能失败，也不触发GPU重跑。

RAW：`/home/qhq/serverless_llm_experiment/results/ieee_tc/dlora_qualification/d791_20261024`。
主来源为audit_full4000_w1.json、launch3b.json、worker_0.log及其来源SHA；
缓存回收与Git备份是否完成以storage_receipt/backup_receipt为准。

实际存储回执：仅删除本轮4000个重复objects文件，回收19497250816bytes
（约18.16GiB），之后可用162065883136bytes；147项旧投稿/用户修改未变。
唯一原池、原始测量和失败证据保留。不可读同UID进程2628、3360275、3361057
如实记录，不宣称全系统FD可见性证明。storage域已实际关闭；备份待回执核验。

下一完成存储/检查点收尾后，采用同一修正模型配置运行既定W2，再3B
Loquetier/HydraServe、PrimeLoRA全比较优化与同版本Full；A1–A5、S1–S13、
CE/Pareto、审稿文档、图表和V2交付全部保留。低SLO不单独触发基线参数搜索。
