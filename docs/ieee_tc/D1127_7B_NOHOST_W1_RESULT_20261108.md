# D1127：7B 独立 HOST 复用消融的单轮观测

2026-11-08。完整回放已结束，4000请求终态、实际GPU释放、服务域移除及
专属远端inactive/MainPID0均核实。Full固定复用主表D1115，不另选Full，
不重跑本执行键。当前7B A1已有10/11配置，尚缺CapacityOnly。

## 主要结果

| 指标 | Full D1115 | NoHOST D1127 |
|---|---:|---:|
| 原生合同完成/offered | 4000/4000 | 4000/4000 |
| 联合TTFT/TPOT达成率 | 98.550% | 99.175% |
| 生命周期物理GPU-s | 7603.874678 | 7820.952236 |
| GPU-s/offered-request | 1.900969 | 1.955238 |
| P95 TTFT/ms | 2331.329520 | 2011.332495 |
| 平均TTFT/ms | 988.900322 | 830.675725 |
| P95 TPOT/ms | 56.037299 | 53.110615 |
| 平均TPOT/ms | 36.318603 | 34.845319 |
| 平均E2E/ms | 5141.060583 | 4778.557866 |
| 远端传输次数/线上字节 | 132/131480060 | 132/131480060 |

Full少用2.775590% GPU时间，但P95 TTFT高15.909703%，联合达成率低0.625
个百分点；平均TTFT、TPOT和E2E也保留NoHOST更好的观测值。**这是资源—质量
取舍，不是Full全面优越或HOST机制已改善尾延迟的证据。**

两点均达到共同95%时延要求，但都超过W1预算6756.571050 GPU-s，不能称
G2同预算胜出。两点实际达成率不同，不能写成服务质量相同。Full的cap3
工作点保持独立，不拼入这一行。原始n_correct=null，191条同后端输出digest
差异保留为原因未知；原生绑定/数量检查不是独立数值证明。这里只报告已知
source42开发回放的单轮描述，未做CI、显著性或独立留出推断。

## 机制确实改变了什么

配置只改变两处同义`independent_host_reuse: false`和三个独占输出/缓存路径。
GPU/NVMe/HOST物理预算、autoscaler、backend、routing、handoff、hierarchy和
admission的非目标配置一致；NoHOST不把HOST预算转给其他tier。

| 原生请求获取来源 | Full | NoHOST |
|---|---:|---:|
| GPU | 2605 | 2510 |
| 独立HOST复用 | 411 | 0 |
| file | 980 | 1479 |
| transient staging | 4 | 11 |

- 4000个NoHOST请求的原生owner回执均明确关闭独立HOST复用；dispatch前
  快照为GPU2498/NVMe1368/remote134，HOST为0。Full相应为GPU2590/HOST435/
  NVMe840/remote135。两时刻口径不混同，不由不同时间的tier变化推断错误。
- 两组各4000个实际service-bin路由选择均可按记录快照重算；保持原可行性
  和reservation。NoHOST与load-only影子选择有2879次分歧，仅表示决策差异，
  不生成反事实时延。
- NoHOST保留1461个稳态residency epoch、1383次软admission评估（1382 admit、
  1 defer_effective_capacity），21次activation的policy均为full。不是关闭
  整个adapter管理或只保留原生按需加载。
- 已报告文件准备结果包含1443个GPU目标、1个NVMe目标、0个HOST目标；Full
  对应1323 GPU、3 HOST。registration只有target数量，没有完整pending目标
  tier，因此只声明已报告结果覆盖，不把它冒充全部待执行目标的证明。
- NoHOST仍有1494条native HOST preparation记录；必要的GPU CPU backing、
  transient staging和OS page cache保留。记录数量不等于成功物理加载次数，
  也不代表独立HOST复用仍开启，不以“CPU内存必须为零”检验本消融。
- NoHOST为initial1、natural_scaleout20；自然epoch中19个有first-dispatch及
  首token，1个ready前取消，无首dispatch歧义。21个物理allocation均关闭。
  Full为initial1、natural19、全部有首服务。自然闭环的计划及扩容时刻会
  随反馈不同，不能据此声称两次运行共享固定准备计划或相同资源时间线。
- 132次远端交付全部与请求关联，线上字节与Full相同，无请求期打包。
  此case不能用减少远程下载解释Full资源收益。

该观测说明HOST复用路径确实存在、消融确实移除它；并不证明其在本场景
消除了主要尾部瓶颈。全局P95受自然路由、准备和扩缩容反馈共同影响，不能
用411次HOST获取直接等同于411次关键路径收益。其有限资源收益与质量代价
完整保留，后续按既定受控机制/争用及预算敏感性回答原因，不削弱NoHOST、
不增加人为等待、不为预期排序重复同键。

## 有效性、来源与收尾

- 原始结果SHA：`b5b95c0ff540cbd3524471b8a1991eb0a9839945268b7a46674a3f793fe8490d`。
- 启动前checkpoint：`bc7e665be1e213fe132ccc37e4d085078a424b73`，已推送V2。
- 本轮执行/分析PLAN：`b5170643aa5d332efc67407af10ab7022a2d68fd802e2389d253968bd92dcf8f`。
  Full保留其原执行PLAN，不追改旧身份。
- V1/V2冻结阈值、4000逐请求arrival/adapter/prompt/target合同及原生计量核验
  通过；TTFT/TPOT/E2E重算误差均0 ms。原始开发身份及正确性限制不改写。
- 2308次资源采样：high/max/OOM/kill/swap为0，主机最低可用101533585408B，
  服务峰值20923133952B，无外来/逃逸GPU进程。
- 远端journal双边SHA一致：
  `df4ab920c995647a7eee7b084633ac4ca2ea316c34640cf4d123829568272ab6`。
  文件名UUID与内容clock分开核验；来源回执在raw/remote_journal_source.json。
- 69个唯一来源SHA、144个旧投稿保护文件及用户dirty manifest核验通过。
  没有覆盖旧结果、修改服务代码或重新生成工件/负载。

原始目录：`results/ieee_tc/p2_backend_qualification/d1127_20261108/7b_nohost_w1/`。
curated：`paper_results/ieee_tc/ablation/d1127_7b_nohost_w1/`。
分析复用D1113/D299/D455、D1126比较与D1063 NoHOST核查；CPU2/3/26/27、
3/4GiB、swap0、隐藏GPU。管理wrapper的既有inotify收尾异常另列于
`analysis_attempts.md`，不当作性能回放失败，也不重跑成功分析隐藏异常。

## 闭口及下一主线

本执行完成并保留全部结果；只能支持共同最低时延要求下的条件资源收益，
HOST带来整体更优服务的主张尚未闭口。不能由本单轮推断所有负载下HOST均
有害或无用，也不能以命中计数掩盖实测质量代价。

下一先完成来源封存和结果备份，再当前7B CapacityOnly→RoutingOnly→
RoutingHandoff。随后继续受控A2/A4、S2位置/信息、S3争用、M2/S5必要点、
当前3B真正同版本缺项、完整S1–S13、成本/CE/Pareto、审稿文档和IEEE图表。
不重复Full/基线或已完成HydraServe适配，不改变共享k3s，整份目标继续。
