# D1126：7B 延后扩容准备的单轮观测

2026-11-08。4000请求回放完整结束，服务/回放/watchdog均退出0，实际GPU
context释放，服务域移除，专属远端inactive/MainPID0。不要重跑本执行键。
Full固定复用D1115，与主表同源；没有为消融另选Full。

## 主要结果

| 指标 | Full D1115 | DelayedHandoff D1126 |
|---|---:|---:|
| 原生合同完成/offered | 4000/4000 | 4000/4000 |
| 联合TTFT/TPOT达成率 | 98.550% | 97.800% |
| 生命周期物理GPU-s | 7603.874678 | 7864.276309 |
| GPU-s/offered-request | 1.900969 | 1.966069 |
| P95 TTFT/ms | 2331.329520 | 2454.767521 |
| 平均TTFT/ms | 988.900322 | 995.753509 |
| P95 TPOT/ms | 56.037299 | 57.465137 |
| 平均TPOT/ms | 36.318603 | 35.803255 |
| 平均E2E/ms | 5141.060583 | 5048.701057 |
| 远端传输次数/线上字节 | 132/131480060 | 132/131480060 |

Full的GPU时间少3.311196%、P95 TTFT低5.028501%、联合达成率高0.75个百分点。
本自然闭环观测的这三项主要结果同时改善，不是资源少但尾延迟更高的取舍。
平均TPOT/E2E反向，完整保留；无CI或重复运行稳健性主张。

两点均满足共同95%时延要求，但均超过W1预算6756.571050GPU-s，不是同预算
G2胜出证据。Full的cap3工作点不能拼入这一行。n_correct仍为null，191条
同后端输出digest差异保留为未解释，原生数量/绑定检查不冒称独立数值证明。
这些是已知source42开发回放下的描述性观测，formal_ranking_eligible=false。

## 机制证据与归因边界

- 4000个请求的arrival/adapter/prompt/target/native计数合同逐条匹配；共用
  service-bin路由、hierarchy、admission和非目标配置。Full路由重算全部匹配。
- 本轮15次activation全为delayed策略：initial1、natural_scaleout14；各有
  第一条实际dispatch及首token，无歧义/unused。Full对应20次activation，
  不是跨运行冻结同计划、同扩容时刻的受控A2/A4。
- 实际记录到GPU/HOST加载各134次，共268个独立物理span，均在engine-ready
  后开始；103个span在对应first-dispatch前完成。303个handoff订阅均在ready
  后，无已观测时序违例。这些是时序证据，计划登记不当作物理I/O。
- 532条native attempt均有回执，其中268显式加载、12显式未加载，另252
  条deferred回执没有明确invoked标志（GPU20/HOST232）。保持未知，不以缺字段
  推成无物理工作或补造耗时；observed_timing_checks_passed=true，而完整
  invocation覆盖未证实。该分析限制不否决完整性能观测，也不触发重跑。
- Full物理审计直接复用D1124：GPU/HOST各115次、ready前均0。不能将本轮
  效果写成已证明native加载与初始化重叠；受控同计划A2/A4仍待完成。
- 本轮1582个稳态residency epoch；1371次软admission评估，1367admit/4defer。
  HOST/GPU/文件准备均保留，未将延后handoff变成关闭整个adapter管理。
- 远端次数和字节相同，不用“减少下载”解释该case的差距。

## 有效性、来源与收尾

原始结果SHA：`0b432e0b76c368affd475a8e6bf95c00fe6dc7a8ee4941ecdb044491431b6720`。
执行checkpoint：`4828565398acec5cd2e98a562a05150845bca00a`。
运行及本轮分析PLAN SHA：`d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83`。
65个唯一来源SHA复核、144个旧投稿保护文件和用户dirty manifest SHA均通过。
原始目录：`results/ieee_tc/p2_backend_qualification/d1126_20261108/7b_delayed_handoff_w1/`；
curated：`paper_results/ieee_tc/ablation/d1126_7b_delayed_handoff_w1/`。

2274次资源采样：high/max/OOM/kill/swap全0，主机最低可用101391585280B，
服务峰值21732651008B，无外来或逃逸GPU进程。15个物理allocation均关闭。
远端132行journal时钟与本轮health匹配，双边SHA一致：
`1ff5d23f5371fb9fde384deaad7f30f5533de8e759ae69706e765eaa1db369a6`。
日志文件名UUID不等于clock UUID，来源路径和最初无文件的查询记录保留。

离线分析使用CPU2/3/26/27、3/4GiB、swap0、隐藏GPU。u395/u396主进程0、
wrapper收尾timeout/1，实际域为空后移除，journal显示既有inotify ENOSPC；
不是测量运行失败，不重跑分析隐藏异常。u397阶段、u398比较、u399时序、
u400来源核验主进程及wrapper均0。分析尝试见curated/analysis_attempts.md。

## 本实验闭口及下一动作

执行结果封存；本次三项主指标的观测优势可作为有限范围子结论。不能据此
宣布全部模型/场景的G1/G2、同计划重叠机制或整个项目已闭口。

先保存结果来源；按用户最新确认修订完整PLAN、批准快照及执行绑定，保留
旧SHA。然后在下一重型实验前，对全部已完成实验组做一次来源绑定的闭口
台账，区分已支持主张、条件收益/取舍、直接证据不足。复用有效点，不重跑
完整矩阵找赢家。之后继续当前7B NoHOST、CapacityOnly、累积节点及既定
受控A2–A5/S1–S13、成本/Pareto、审稿文档、IEEE图表和V2交付。
