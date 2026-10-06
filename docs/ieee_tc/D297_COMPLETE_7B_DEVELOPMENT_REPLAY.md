# D297：退出依赖优化后的完整 7B W0 开发回放

2026-10-07。延续D296的同一候选，不添加算法修改、不调整参数、不重新抽样。
本项补足100请求前缀不能提供的持续负载/热点轮换证据，不是正式留出或G1/G2
结果。执行前完整读取源计划与冻结指标V1，达标前不画图。

## 依据与固定配置

- D296的100请求完成、实际GPU退出和CPU验证已封存于提交300c789；总GPU-s
  727.912111，收尾36.264719。P95用户TTFT10.771356秒，并未改善尾延迟。
- 本次保留其全部model、backend、resource_coordination、preloading配置，
  仅恢复4000请求、Full场景名和新的输出/缓存目录。既有W0 trace和500逻辑
  adapter清单直接复用，无新权重、负载或工件池。
- vLLM0.30/FP16/FLASH_ATTN、TP1、4slots/4seqs、1024context、759/256上限、
  fixed_length_greedy_v1、prefix off、真实预发布远端均不变。
- 原open-loop到达、共同部署通知60秒、请求保护1800秒、自然闭环、完整
  allocation/release计量不变；保留开发期5000ms在线控制，不能当正式共同SLO。
- 之前D219完整回放与当前候选在源码/主动准备资格等方面不同，只是历史参照，
  不能把本轮与它的差异归因于单一D296改动；100请求不能线性外推完整GPU-s。

## 启动前机制语义复核

当前原生主动准备由`_coordination_enabled`选择`capacity_only`，并通过
`_run_ieee_owned_preparation_plan`→`ieee_prepare_host`执行E(t)。旧汇总
`effective_capacity_admission_enabled=false`是旧coordinator/可选请求容量
见证的开关，不等于当前原生主动准备关闭。D226的旧路径解释不能直接套用。
本次不为消除旧标签而改变普通请求的加载优先语义。

D296原始`main_outcome.json`实测：4次activation均policy=full；167个residency
epoch；171个file plan；101个GPU plan全部capacity_only=false；144次GPU
attempt中115完成、29延后。109份receipt明确proactive_admission_evaluated=true，
13份为false（包括复用已在GPU的工件），22份无该标志，不补算成主动决策。
这些是实际触发的证据，不是完整数值正确性或性能收益的证明。

## 验收与安全边界

1. 原始4000 offered集合不删请求；失败、缺失、取消和真实token数全部保存。
2. 核验trace/subset、prompt/target、native token、dispatch tier/slot/owner绑定。
3. 正确性未知仍保持未知；不以HTTP成功、数量相等或近并列解释赋值N_correct。
4. 按V1重算平均/Type-1 P95/P99 TTFT、TPOT、各阶段、物理GPU-s和互斥窗口；
   记录actual mechanism events，不用旧汇总零计数替代原生路径证据。
5. 同一72/80GiB服务、3/4GiB外置域、swap和CPU限制、150/100GiB磁盘门；
   新增磁盘峰值沿用完整回放32GiB保守预算。已封存D219 run约828MiB，缓存
   预算和日志上界另保留，不能只用短回放磁盘量放宽门槛。
6. 必须先通过直连认证health及远端配置核查，再启动本机回放。沿用现有
   paired remote monitor，3B artifact unit只是空闲监控对象，无3B推理。
   测量期间不重启、改配置、打包、清理、压缩或另起压力测试。
7. 只运行此重任务；不启动暖参考、Resident或其他分析重任务并发争用。
8. 完成后释放→验证→无图数据/解释→备份。单次完整开发回放无CI、不宣称最优。

启动：`bash scripts/ieee_tc/run_d293_7b_prefetch_validation_100.sh d297`，
通过既有受限runner执行；取消diagnostic prefix，不绕过guard。目录已存在即拒绝。
原始目录：`results/ieee_tc/p2_backend_qualification/d297_20261007/7b_full_w0/`。
所有历史结果、用户manifest和论文源文件不动。

启动前核查：结构化比较确认只改变请求数、场景名、输出/缓存路径；运行代码
SHA仍为`cc821e089b952b66bc10be13e3b8ab11700c9d77dc542c20ca7edb582b1113f8`。
新配置SHA`ecf75d4d6a27ac9c54e3347dfa1089aafa8c159a84676039de86c2216fd012be`。
启动器bash语法通过；系统Python的open-loop与资源保护94项测试通过。
本轮没有改serving代码，D296的634项CPU、34项native环境专项和22项分析器
验证继续作为同一实现证据。尚未开始GPU回放，启动后另记真实状态。

下一步依据本轮完整证据继续7B并完成共同warm/Resident及旧Prime新口径对照；
数值资格与G1/G2未成立前不进入3B、外部基线、消融或正式绘图。

## 实际启动与运行中状态（2026-10-07 01:20+08）

已在提交`a40df0d307c31fa38f11f0b77b72dd16bba10907`推送并核对远端后，
于01:16启动唯一tmux任务`primelora-d297-7b`。上述“尚未开始”描述只对应
预登记检查时刻，不是当前状态。不得因尚无最终结果文件而重复启动。

- 直连认证health结束`814229.57178896`，部署通知`814247.332460961`，
  回放原点`814307.332460961`（均本机monotonic）。此次health先于本机启动。
- 回执确认W0/count4000/source_count4000/rate_scale1，无prefix；trace SHA
  `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`，
  view SHA`a5331be2e2204483f18206825d9aaa18cbaa259fe0babfc303f989055819430d`。
- 服务scope`primelora-tc-svc-72524b343c5c45a699736635a24184bd.scope`，
  invocation`3d9973309a41446baa8df1e8ac763a91`；外置watchdog PID1851665，
  独立aux scope`primelora-tc-aux-7f9be371ba2644b5a885484a29b73f98.scope`。
- 远端7B PID727107/invocation`2ceae044ea1b4290b738ef29f16f179d`；
  3B PID727105/invocation`cee328bfa4654233813c435c85789b96`（无3B推理）；
  `primelora-artifact-monitor-d297.service` PID727110/invocation
  `bec592d608814aa7a9a4512c7d364bff`。按身份停止monitor后，由原有退出逻辑
  停止本次配对服务；不能停止后来另一个invocation。
- 远端监控位置`/home/lab14/primelora_remote/tc/d297_20261007/remote_monitor.log`；
  7B clock`remote-process-monotonic:b7998243a505498db48f39cde67db697`，
  后续仅收取与该clock及本次transfer ID匹配的交付证据。
- 预发布交付方式`prepublished_gzip_v1`和`artifact_timing_v2`不变；NIC1000Mbps/full。
  既有D78/D80一次性交付缓存直接复用，用户本轮再次批准不意味着重新生成。
- 启动前空闲磁盘161,199,501,312bytes（约150.129GiB）、MemAvailable
  110,615,760KiB（约105.49GiB），满足原门槛。01:20监控sample220：
  服务peak18.240692GiB、主机available90.266891GiB、high/max/OOM/swap均0，
  无安全告警；磁盘149.509441GiB，只禁止下一重任务启动，不触发100GiB停止门槛。
- 此刻79请求到达、74正常完成、0报告失败；仍是部分运行，不能称4000完成、
  数值正确、联合SLO达标或优于历史。最终审计和真实释放后另补完整结果。

本轮只作轻量监控和持久记录，不并发运行分析、哈希扫描、压缩、清理或另一个
GPU任务。真实读取/传输/竞争/本地准备继续计入服务时间，不减去实际获取时延。

## 运行中核查与后处理复用（01:28+08）

同一tmux和服务身份继续运行，518到达、516正常完成、0报告失败；watchdog
sample696记录服务peak18.240692GiB、主机available89.911167GiB、high/max/OOM0，
无安全告警。匹配的远端monitor仍active/running，无重启或配置变化。
本段是经实时进程与递增日志证实的等待，不是完整结果或正式正确性结论。

本轮只读历史核查明确后处理应复用：

1. D170已完成1536个warm测量和16个warmup，文档及curated数据已封存；
   “共同参考仍开放”表示最终共同合同/数值身份尚未验收，不表示该测量未做。
   不重跑D169/D170，不重新解析其1.76GB原始JSON。
2. 对本轮可复用`paper_results/ieee_tc/p2_backend/20261002_d170_warm_measurement.json`
   及`analyze_control_path_overhead.py::provisional_timing_bound`进行候选门槛诊断。
   逐请求输入组、原生时间线和完整offered分母必须核验；正确性未知保持未知，
   时延两项通过比例仅是候选门槛下联合达成的上限，不是正式A_joint。
3. D234的旧GPU量被D240更正：D219物理总量15911.7125GPU-s，旧4317.05只是
   primary-runtime计费投影。以其封存`physical_deployment/summary.json`及
   allocation journals为依据，不能把旧逻辑账单与本轮物理量混比。
4. `plot_d290_diagnostic.py::audit_run`可复用当前记录的原生时间/新物理sidecar
   验证，但`compare_diagnostics`固定写有100-request限制，且D219没有同名
   `physical_resource_summary.json`。完整历史比较需显式记录不同schema与
   serving版本，不直接套用短实验标签，不关闭缺字段/来源检查。
5. 待本轮GPU真正释放后才作有界离线解析；避免同时在内存保留两个大型结果。
   先核验并保留紧凑请求投影，再分别统计；原始证据和旧结果不动。

## 待完整数据检验的资源回收假设（只读核查，未改配置）

2026-10-07 01:33+08。本轮846/4000正常完成时，在线记录仍为3次扩容、
0次缩容、4个runtime。部分运行不代表最终轨迹；本节只登记完成后的分析问题。

历史与源码证据：

- D88首次开发控制推导使用7B容量C=2，导出queue upper2、active upper/lower
  0.5/0.25；TTFT lower281.278192ms来自D84 GPU-ready的acquisition→first
  Type-1 P95。D88明确称这些为开发启发式，并预先列出下界过严的风险。
- D156（提交26e44a2）只把max_num_seqs/runtime cap从2提高到4，用于隔离
  容量假设。后续更新了容量相关service/preparation profile，但本轮配置仍
  保留上述控制阈值。保留阈值本身不违反公式，也不自动意味着错误。
- `IEEEReplicaControl.evaluate`要求queue、active saturation、最近完成请求
  的overall TTFT P95三者严格全低，并连续满足cooldown，才能缩容。
  `_evaluate_ieee_live_control`实际传入ready容量=ready实例数×有效cap4；
  TTFT观测来自`overall_ttft_ms`，不是D84的acquisition→first区间。
- 有效采样已经保存在`_ieee_control_events`，包含queue、active、capacity、
  P95、sample count、all_low、low_since、action和outcome；无需增加在线监控。

**可证伪问题：** 在低队列/低饱和度时，旧短路径TTFT下界是否持续阻止全低
条件，从而使早期扩出的GPU持续持有？容量变化后沿用旧软目标是否也使扩容
更积极？它们目前是待检验解释，不是已证明的资源瓶颈。

完成后只复用此次完整记录：逐项报告低队列/低active/已知低P95及其交集、
连续全低时长、pending导致的重置、实际回收及物理持卡区间。如果主要原因
是queue/active持续偏高、无已排空候选、或未知样本，则不能归因于TTFT下界。
即使观察到某一条件阻止回收，也不能从同一轨迹反推改变控制后的反事实GPU-s
或SLO收益。先完成完整结果审计，再决定是否形成下一单一候选；不现在改参数。

本轮联网复核：

- [Knative concurrency](https://knative.dev/docs/serving/autoscaling/concurrency/)
  区分硬容量和软目标，并以目标利用比例控制提前扩容；这支持分开审计两个量，
  不支持按容量比例机械调整所有阈值，也不采用其默认数值。
- [Kubernetes HPA](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/)
  对多指标采用最大的副本建议，缺失指标时对缩容保守，并单独处理稳定窗口。
  本项目继续遵守IEEE自己的max-score/全低语义，不替换为HPA算法，不把缺样本
  填零来强行产生缩容。

以上为轻量源码/历史/文献核查；没有运行重分析、增加GPU实验、改变正在测量的
系统或把正在运行的样本用于正式配置筛选。

## 完整结果与决定（02:36+08，替代上面的运行中状态）

4000/4000请求正常终态，原生输出数量、dispatch tier及10项保存的身份绑定
检查全部通过。4份物理allocation日志均已释放；远端仅按捕获的身份停止。
这些不等价于内核数值资格；`N_correct=null`保持不变。

| 描述性指标 | 历史D219完整7B | 当前D297完整7B |
|---|---:|---:|
| 平均TTFT / s | 1.508362 | 1.527145 |
| Type-1 P95 TTFT / s | 3.619799 | 3.359722 |
| 平均TPOT / ms | 37.565953 | 37.883698 |
| Type-1 P95 TPOT / ms | 60.478714 | 62.065918 |
| 平均E2E / s | 5.848015 | 5.872782 |
| 物理GPU-s | 15911.712503 | 15860.686503 |
| D170候选阈值下联合时延上限 | 3759/4000=93.975% | 3786/4000=94.65% |

源trace、subset、部署plan和逐请求adapter/canonical prompt/输出数量核对一致。
两个serving版本不同且各只有一次开发回放，不能作单因素因果归因或显著性
结论。126条输出hash变化保留；不因为D295揭示近并列就自动通过数值正确性。
新指标已有部分改善，但平均TTFT/TPOT退化、联合时延上限仍不足95%，资源
下降仅约0.32%。不能宣称优于历史系统的全部目标，不能进入3B或基线阶段。

D297资源互斥分项：业务前36.792974、到达期15773.837634、drain10.689912、
收尾39.365983GPU-s，合计15860.686503。没有用旧逻辑计费或空闲折扣替换。
GPU-s/offered为3.965172；per-correct和正式预算排名仍未知。

### 缩容假设的反证约束

1861次控制观察，1859次TTFT已知，其中最小P95仍为1525.370ms；没有一次
满足281.278ms低门槛。308次低queue/低active均受此门槛限制。
**但是**，在同一已观察轨迹上，只看低queue/低active且无pending的连续区间，
最长也仅16.882秒，小于38秒cooldown，没有任何区间达到38秒。
所以“只改TTFT下界便可回收”缺少支持。控制窗口/容量余量/实际到达共同决定
机会，不能将全部持卡成本归咎于一个参数，更不能从这些样本推算反事实节省。
本轮不选择新参数，不改九式、控制语义或评价阈值。

### 服务路径、远端与保护

平均TTFT分解：到达→admission0.817574s、admission→engine0.421192s、
engine入口→队列0.080526s、native queue0.009212s、prefill0.198641s。
这是可加和观察分解；前两段不是纯网络，也不能直接相减预测优化后的TTFT。
137条TTFT违约、80条TPOT违约，其中3条同时违约；联合通过还差14条才达到
候选阈值下的95%，不能添加舍入容差或放宽倍率。

132次交付全部按UUID与本地获取记录一一匹配，artifact/归档SHA/字节/时序
检查各132通过；线上131480060bytes，0次请求内打包，无未匹配记录。真实
传输/读取/本地准备保留计时。远端缓存和1Gbps/full链路不变，无远端调整。
本机4037次监测：服务peak19.031GiB、主机最低available87.140GiB，high/max/
OOM/swap均0。远端三个服务已inactive/dead、PID0。没有新增图或更改论文。

分析首轮因`actual_prompt_tokens=759.0`这类表示被严格整数接口拒绝，尚未
产生输出；保留`analysis_attempt1.json`。显式检查有限、正值、严格整数值后
转为int，所有4000条通过，未改变数量/时间/阈值。第二轮输出在`validated1/`，
原生时序/身份检查和8项既有时延分析测试通过；分析scope峰值约716MiB，无
回收/硬限/OOM事件。主结果SHA为
`7f171e1d58b061e7dd7392e1651d087c300ce1d6c2223c3ed16ef2fdce3aa07b`。

原始结果及远端证据仍在D297唯一目录；无图汇总在
`paper_results/ieee_tc/p2_backend/d297_full_w0/validated1/`。保护清单146/147
保持，唯一差异仍为预先存在的baseline replay脚本，不声称全清单通过。
当前磁盘约149.14GiB，下一重任务前先按保护规则恢复150GiB门槛，再结合
这些完整证据和官方实现提出下一项7B单一优化假设；不重复D170/D295补样本。
