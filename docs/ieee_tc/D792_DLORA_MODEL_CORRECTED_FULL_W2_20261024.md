# D792：模型兼容修正后的 dLoRA 3B 完整 W2

2026-10-24。一轮完整4000请求已结束并真实释放GPU；原始结果与失败全部保留。
本次不是达标资源点，且收尾发现迁移/取消生命周期异常，不能将全部退化直接
解释为dLoRA的正常机制局限。先保存证据，再有界检查必要正确性问题，不搜索
baseline性能参数、不屏蔽断言、不延长保护超时。

## 配置与复用边界

保留既定DP2/TP1/PP1、EngineManager/PERIOD_MIG、500-ID部署池、max-seqs4、
max-batched-tokens1024、GPU利用配置0.92、GPU LoRA容量8及swap8。与D791
使用相同D788模型兼容视图，只换W2输入索引和独立运行路径；没有生成工件或
负载。W2仍为原arrival/prompt/token，rotation100，实际offered ID为497，
目标总输出458224 tokens。

- MAIN执行HEAD：`072633f04cab013e4682bd602ba54b2ea12cfe75`。
- BASE执行HEAD：`cf421793e7b71342cf75ff063f8d64afe26d8169`。
- view SHA：`5558de43ede0e2cf49c7478dcfb94640e7a8516406f53766fabfdf524bf91174`。
- 模型源码视图SHA：`2d06f71560515a355a625880a221cbb6e8be3447c0afd9d9b080f7cf1df27e24`。
- 共用3B SLO冻结SHA：`d15123570c4a8d830515ca942d1db553c6717ffc1a40ade5c1546d35e2bed9d8`。

D790对两个既有prompt/adapter、prefill及两步decode的6点greedy一致性是有界
模型证据，不是完整4000请求的独立数值证明。`formal_run=false`、
`n_correct=null`和原数值未知字段不追改；这是已知回放上的单轮开发期观测。
旧D787因模型语义问题中止，继续保留为R3，不用于构造本次性能提升。

## 完整测量

| 观测量 | 数值 |
|---|---:|
| 计划/到达/提交/终态请求 | 均4000 |
| 原生计量和绑定可核对的响应 | 3222（80.55%） |
| 失败 | 778（19.45%），全部TimeoutError |
| 响应原生输出tokens | 361860，逐请求与目标一致 |
| 服务adapter ID | 413；offered497；部署准备500 |
| TTFT达标 | 44/4000＝1.1% |
| 联合TTFT/TPOT达标 | 34/4000＝0.85% |
| 条件TPOT达标 | 2508/3222＝77.839851% |
| 物理生命周期GPU占用 | 9904.341262000613 GPU-s |
| 条件平均/P95 TTFT | 302099.139254 / 1353856.556690 ms |
| 条件平均/P95 TPOT | 301.550469 / 236.961140 ms |
| 条件平均/P95 E2E | 340854.484843 / 1472460.216095 ms |
| 原生计时、E2E和TPOT最大重算误差 | 均0 ms |

时延仅针对3222条协议有效响应；失败仍在4000的offered分母中。
GPU-s/offered为2.4760853155001534，GPU-s/绑定有效响应为3.0739730794539457；
两者都不是已证明数值正确请求的资源效率。G1/G2不可行，不进入达标排名，
不因为有界模型检查通过而掩盖请求失败。只有一次GPU运行、一次离线分析；
没有重复均值/CI或显著性结论。

资源互斥窗：pre-arrival0、arrival7733.583980429918、
drain2151.5410938244313、cleanup19.21618774626404 GPU-s，合计等于U。
最后请求终态之后实际GPU释放仍计量。service/replay退出码均1、watchdog0，
`native_gpu_context_release_confirmed=true`、服务资源域实际消失。运行结束
不能称为运行成功。

## 阶段解释及新发现的异常

成功响应的平均E2E分项（ms）：提交3.606922、提交后dispatch等待10612.127430、
service TTFT291483.404902、decode38734.923534、完成通知20.422056。
service TTFT包含原生排队、调度、加载和prefill，不是纯GPU计算。后段序号
3000–3499仍有497响应，router queue均值101.305204ms，service TTFT均值
462459.610573ms；这是事后描述切片，不是无启动独立主实验。

严格匹配原生日志得到126次求解，累计89.03275299072266s，最大8.211064100265503s；
142条迁移决策打印、723次请求提及。这些不是723次成功迁移回执，不把累计
求解时间从用户延迟中相减。

**必须继续核查的具体问题**：worker_0.log的6386、6506、7600、8327行出现
迁移任务的`AssertionError`，对应EngineManager.migration_schedule中
`assert len(request_streams) == len(req_ids)`。四次之前分别出现
req_00727、req_00643、req_00990、req_01793的missing-stream警告；这些ID
全部是已超时请求，而且警告前有原生Aborted记录。另req_02000也有同类
警告及更早Aborted。D791 W1严格检索未见上述断言或未回收task异常。

已读执行源码：迁移先fetch_seq_groups，再取出源request streams，然后检查
数量并添加到目标。若取消后的旧sequence与stream集合不同，断言可能发生在
已经修改部分所有权之后；这是真实代码/日志线索，不是已证明全部778超时的
单一原因，也不是已完成的修复。不能以删除断言或吞异常代替生命周期一致性。
下一需对照官方源码和既有取消适配历史，做有界的所有权/终态核查，决定最小
必要修复及受影响键；保持调度目标、迁移策略和性能参数不变。

原生正常Finished恰3222个ID，与响应数一致，未发现Finished但客户端失败ID。
最后滞留的req_03436、req_03493、req_03517有原生Received，无Finished/Aborted
日志，最终均按1800秒保护超时；不能将它们补记成功，也不能仅凭现有日志把
它们逐条归因为上述四次断言。尾部等待完整保留在drain及U中。

## 远端、资源与证据

复用D80部署通知前只读gzip交付，500次部署准备获取、线上1160474851bytes、
物化19482573296bytes，请求中动态打包0。必要传输及本地准备仍在真实时间线，
没有事后减去span。服务期间没有远端维护/改参、并行构建或另一GPU实验。

服务内存峰值77406666752bytes、high3884、max/OOM/kill均0，swap峰值
2147479552bytes；主机available最低36091551744bytes、磁盘最低
142428389376bytes。软限制回收与swap属于性能条件，不声称没有内存影响。
分析资源域单独发生high21454，无OOM；这是运行后的离线分析，不能混入被测
服务压力或GPU时间。分析使用已合格NumPy环境，没有重复D791的环境选择错误。

本地与远端本实验服务均实际关闭，54项Ray日志及远端传输/监控、resolver
回执已保存；没有复制权重。RAW：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/dlora_qualification/d792_20261024`。
主证据为audit_full4000_w2.json、launch3b.json、worker_0.log及SHA清单。
缓存回收及备份状态以storage_receipt/backup_receipt为准，未完成不提前宣称。

09:57实际回收完成：仅删除本轮4000个重复objects文件，回收19497328640bytes
（约18.16GiB），之后磁盘可用161754988544bytes（约150.65GiB）。147项旧投稿
及用户修改保持原SHA。唯一原池、原始测量和失败证据均保留；不可读同UID
进程2628、3360275、3361057如实列于回执，不冒称全系统FD可见性证明。
存储资源域实际关闭，备份待独立回执核验。

封存后先处理本轮具体迁移/取消异常，再完成3B Loquetier/HydraServe必要适配，
然后PrimeLoRA全比较优化及同版本Full、A1–A5、S1–S13、CE/Pareto、两轮审稿
文档、图表和最终V2交付。未完成范围不省略，不因基线SLO低无界调参。
