# D1063：NoHOST 完整结果与 3B A1 十一组汇总

2026-11-05。唯一 NoHOST 4,000 请求回放已结束；collector 与 comparison 均正常
退出，实际 GPU 已释放。3B 自然闭环 A1 现在完成 11/11 个配置，不代表 Full
已达到优化目标，也不替代受控机制实验。未重跑 Full、未新增工件或负载。

## 1. Full 与 NoHOST 的原始结果

两组共享 source42/W1 的到达、adapter、prompt、target 和固定输出合同，
同一 D648 阈值、D657 预算、V1/单轮 V2。Full 直接复用 D1047；NoHOST 仅关闭
独立受管 HOST 复用，GPU/NVMe 预算及其他控制配置不变。必要 GPU backing、
暂存与操作系统 page cache 不被虚构为不存在。

| 指标 | Full：D1047 | NoHOST：D1063 |
|---|---:|---:|
| 成功终态 / offered | 4000 / 4000 | 4000 / 4000 |
| 原生输出 token | 458224 | 458224 |
| TTFT 达标 | 3972 | 3986 |
| TPOT 达标 | 3858 | 3864 |
| 联合 TTFT/TPOT 达标 | 3836 | 3852 |
| 联合达成率 | 95.900% | 96.300% |
| 生命周期物理 GPU-s | 6182.715971 | 6213.232770 |
| GPU-s / offered-request | 1.545679 | 1.553308 |
| 平均 TTFT（ms） | 454.156048 | 447.272077 |
| P95 TTFT（ms） | 1114.091001 | 1073.016274 |
| 平均 TPOT（ms） | 17.417030 | 17.572065 |
| P95 TPOT（ms） | 31.774061 | 31.312108 |
| 平均 E2E（ms） | 2323.526111 | 2333.587627 |
| 远端获取次数 / 线上字节 | 132 / 306360162 | 132 / 306360162 |

Full 的 GPU 时间低 0.4912%，但 P95 TTFT 高 3.8280%、联合 SLO 低 0.40
个百分点；平均 TPOT/E2E略好。两点观测时延均超过95%，资源均低于
6754.783565 GPU-s的共同预算。这是单轮取舍，不能写成HOST复用的明确
端到端收益、统计显著性或正式G1/G2最优性证明。

原始 `n_correct=null`、`formal_ranking_eligible=false` 保留。既有3B池为零更新
结构负载，原生数量、绑定和引用路径检查不等于独立非零LoRA算术证明。
表中分母明确为offered，不把未知的正确请求数补为4000。没有运行间CI，
没有独立留出身份，也不按胜负追加同配置运行。

## 2. 消融确实改变了什么

4000条原生回执均明确 `independent_host_reuse=false`；dispatch HOST为0，
native acquisition HOST为0，非目标handoff/residency准备仍执行。

| 机制观测 | Full | NoHOST |
|---|---:|---:|
| dispatch GPU / HOST / NVMe / Remote | 3327 / 2 / 538 / 133 | 3303 / 0 / 564 / 133 |
| native acquisition GPU / HOST / file / staging | 3331 / 2 / 664 / 3 | 3306 / 0 / 689 / 5 |
| demand staging join | 3 | 5 |
| file preparation plans | 2397 | 2361 |
| GPU preparation plans | 1267 | 1204 |
| native CPU preparation records | 730 | 766 |
| residency epochs | 2389 | 2352 |
| initial / natural-scaleout / controlled | 1 / 9 / 0 | 1 / 9 / 0 |

NoHOST 仍有766条CPU准备记录并不等于独立HOST复用仍开启：GPU加载所需
backing和与请求/计划绑定的暂存按定义保留。记录数量也不是独立物理加载
次数或字节数。demand回执中的退役ID计数为0，不声称本轮覆盖全部退役冲突。
NoHOST文件计划含handoff10、residency2351，GPU计划含handoff9、residency1195；
各activation仍为full policy，没有顺带关掉层级规划或handoff。

Full仅2条直接HOST dispatch/acquisition，不能将其误称为所有CPU准备的
使用次数；GPU命中背后的准备仍可能经过HOST。另一方面，两组远端次数、
线上字节相同，本case不能用减少远端传输解释性能差距。

两组initial1/natural9均有first-dispatch及首token，无歧义、无unused epoch；
但触发时间、目标和后续请求轨迹不同，不是同plan的受控A2/A4对照。
不能因epoch计数相同就称闭环过程相同；不为凑20个样本追加运行。

Full平均TTFT高6.883971ms，阶段差（Full减NoHOST）分别为：
arrival→admission +4.595752、admission→acquired −0.304812、
executable→engine −0.808757、engine-entry→queue +1.940954、
native queue −0.103389、prefill +1.564222ms。这里只定位等待分布，
不是把闭环差值强行拆成HOST的因果收益。

## 3. 完整 3B A1：所有反例保留，同一个 Full

以下全部是相同输入与冻结评价口径下的一次自然闭环观测；ElasticOnly和
ElasticLocality是PrimeLoRA消融，不是外部论文基线。既有九份comparison
及其manifest逐一校验，Full的run、输入计划、全部指标及GPU时间必须与
D1047完全一致；不会为不同图表挑选不同Full。

| 配置 | 联合SLO（%） | 物理GPU-s | P95 TTFT（ms） | P95 TPOT（ms） |
|---|---:|---:|---:|---:|
| Full | 95.900 | 6182.716 | 1114.091 | 31.774 |
| ElasticOnly | 98.200 | 6032.623 | 695.140 | 27.148 |
| ElasticLocality | 98.850 | 6087.727 | 658.337 | 24.071 |
| FixedFull | 97.300 | 8933.754 | 1060.059 | 27.562 |
| LastKnown | 96.675 | 6021.305 | 1008.016 | 31.977 |
| NoRouting | 97.350 | 6581.707 | 980.708 | 29.381 |
| NoHandoff | 95.850 | 5990.135 | 1052.182 | 31.900 |
| R0H0 | 97.275 | 6423.550 | 910.913 | 28.836 |
| DelayedHandoff | 96.275 | 6190.254 | 1080.770 | 31.869 |
| CapacityOnly | 96.200 | 6371.761 | 1042.205 | 31.489 |
| NoHOST | 96.300 | 6213.233 | 1073.016 | 31.312 |

这些数据支持与不支持的结论必须分开：

1. Full比四实例常驻FixedFull少30.7938% GPU时间，但联合SLO低1.40百分点、
   P95 TTFT高5.0971%；支持此工作点存在弹性资源节省，不是全指标支配。
   FixedFull超出共同75%预算，不能用它证明同预算G2优势。
2. Full比ElasticOnly/ElasticLocality分别多2.4880%/1.5603% GPU时间，
   P95 TTFT高60.2686%/69.2281%，联合SLO低2.30/2.95百分点。
   目前未证明主动adapter管理在该自然W1工作点带来净收益，不能把复合
   消融差距全部归到某一个机制；全部十一组远端次数/字节相同。
3. LastKnown最终仅11/4000次纯视图选择分歧，见D1062；该观测未证明
   confirmed-state传播的端到端优势。S2的缓存存在/位置/信息视图区分仍必要。
4. CapacityOnly与Full未观测到有效软余量不足的defer，见D1052；不能把
   GPU-s差异归因于软admission收益。受控S3仍未完成。
5. NoHOST不是压倒性退化，不能用它强撑HOST收益。保留完整表与阶段证据，
   不改变SLO、预算、负载或对照策略来制造Full胜出。

## 4. 完整性、资源及来源

- 全4000条request/adapter/source-item/arrival-plan/prompt/native-prompt/target/
  generation/actual-token逐一匹配；原生输出量458224。输出hash相同3771、
  不同229，全部差异保留，不默认其是浮点误差或错误adapter，不丢弃请求。
  TTFT、TPOT、E2E的原生时间重算最大误差均为0ms。
- 132次实际远端传输的ID/字节/hash双边匹配，复用D80预发布gzip，无请求内
  打包。`existing_unmonitored`没有连续远端CPU/外部干扰监测，该限制保留。
- 10次物理allocation全部release；scale-up9、scale-down8。service/replay/
  watchdog退出0，实际服务域消失，GPU原生context已释放。
- 2266次资源采样无foreign compute或escaped worker。服务峰值21754093568B，
  high/max/OOM/kill/swap全零；主机最小可用93349130240B，磁盘最小
  157691170816B。两个离线分析域均inactive/dead且退出0，峰值分别
  720289792B、112668672B，保护事件/swap全零。
- 工作HOST/NVMe缓存正常清空，147个旧投稿文件和两项用户dirty SHA未变。
  01:12复查可用磁盘162075549696B（约150.94GiB），无GPU compute残留。
  不为凑清理动作删除任何唯一证据；下一重型任务仍需实时复查余量。

执行提交：`0ad0f12cb1c90084c8f37f8070ad662e535b7852`。
配置SHA：`bcbcad704a5993f08d9e4e63a746fefda52fbfa0565ddd55778ef01ad976357b`。
原始结果SHA：`58eb4e5b14ad62160228833e053669b3ef85794a8d03df09e1602a3b7aa6c2c6`。
comparison SHA：`2b106e20a2c9e0202d2366d22f48cb9d4332d9070cf137febd53e6fc9e6e9b7c`。
A1 matrix SHA：`f5819806ff88d4e1a32e1503bd547875d18889600d8e12b48aaac2831a2a1f39`。
RAW：`results/ieee_tc/p2_backend_qualification/d1063_20261105/`。
curated：`paper_results/ieee_tc/ablation/d1063_3b_nohost_w1/`。
相同现有analyzer完成计量，未修改服务代码，也未进行新GPU实验或绘图。

## 5. 下一项与完整范围

先备份本次闭合数据、十一点表和说明；以RAW `backup_completion.json`及
远端commit核验为实际推送证据，不预写成功。origin V2与FaaSLoRA专用backup
分别核验，后者不冒称已整合分叉的实际FaaSLoRA V2。

随后回到Full优化主线：先使用已保存的同版本请求阶段/控制事件，核对既往
失败方向，区分主动管理的实际执行开销与路由决策差异；结合官方源码/原始
文献，只提出一个有新增证据的、九公式保持的候选。不从本表直接跳到调参，
不重跑当前NoHOST或Full找赢家，不把关闭核心机制改名为Full优化。

仍待完成：7B十一组A1及两个累积点、受控A2–A5、M2工作点、S1–S13、
CE/价格/Pareto、B1/B2/C1/C2/C3和两轮审稿证据文档、IEEE图表、最终同版本
主结果来源统一与实际V2集成。HydraServe既有隔离适配/3B结果继续复用，
共享k3s不操作。本轮完成一个实验组的观测，不宣告整体目标完成。
