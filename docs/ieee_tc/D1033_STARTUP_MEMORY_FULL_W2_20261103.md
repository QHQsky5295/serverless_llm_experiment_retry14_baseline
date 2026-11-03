# D1033：启动显存画像复用的完整 3B W2 验证

2026-11-03，承接 D1031 冷画像与 D1032 命中验证，当前完整目标 active。
本文件是当前交接，不改用户 dirty EXECUTION_STATUS.md，不重开共享 k3s。

当前状态：**完整运行已于14:04前闭合，3772/4000联合达标（94.30%），未达95%。禁止重复启动。**
结果与解释见本文末尾；当前没有运行中的D1033性能实验，不重新跑暖缓存版本。
历史启动记录：12:55:37 启动 tmux `tc-d1033-full`。
源检查点94df4b01c86c81d1486630cc71031c8739e3be2f已推送并ls-remote核验
origin/retry14_continuous_queue_v2与faaslora_origin/
backup/d1033-startup-memory-w2-20261103。后者仅备份，未force历史分歧V2。
配置SHA992e1244e98093ba47a038fdb000939ef356b29300babd8181bbe251ad2e2f34。
测前147旧保护项和D1032七处来源SHA核验通过，用户manifest/状态SHA未变。
可用磁盘161507696640 B，MemAvailable111755149312 B，无GPU compute。
新run目录由原runner独占创建；显式unset三个TC详细timing、prefix诊断及
VLLM_ENABLE_STARTUP_PLAN，保留exact policy，非启用第二套建议容量缓存。
本段是运行中交接，不是4000请求完成或达标声明；不在回放中追加备份。

12:58左右检查：154个资源采样时已38条终态，仍在运行。service scope
primelora-tc-svc-6ca4277f1a5648d6bfbec79c2e8f04bb.scope，aux scope
primelora-tc-aux-8b3f4b8b762340f38d1db24564df0448.scope；独立watchdog
PID4123960、replay PID4123942。初始native PID4125618，worker父4124528，
日志/tmp/faaslora_worker_esrgsl8f/worker.log。初始原生init77.14s、compile
32.66s，临时17s+真实12s capture，确实是Full自己的冷路径，不宣称命中。
当前host可用96309325824 B、service peak18267656192 B，high/max/OOM/
kill均0，watchdog无告警/中止。只代表截至当前采样，不代表完整安全验收。
后续继续监测同一tmux和raw，不再启动任何profile、short Full或第二个W2。

13:03左右继续核验：同一replay/watchdog PID仍live，样本442时310条终态，
不是阻塞或重启。当前host可用92.16GiB，service peak17.01GiB，high/max/
OOM/kill、foreign compute及逃逸均0。磁盘148.13GiB低于下轮启动150GiB，
仍高于当前运行底线100GiB；不要测量中压缩，结束释放后再核验下一任务。
后继native PID4129029（父4124533），日志/tmp/faaslora_worker_ju4sav_9/
worker.log明确skipped memory profiling，真实capture11秒/0.69GiB，
init16.79s、compile0.27s。此处只确认原生分支，最终需与startup_memory
回执、producer及activation类别绑定，不凭时延把它直接分类为natural。
本次goal turn属于verified wait：同一活进程、递增终态和资源采样均已核验。
没有新代码/配置、额外实验、远端操作、压缩或图表；全范围目标继续active。

13:04:52–13:15:24：继续同一运行的15次只读定时核查，每次根据/proc
start_ticks确认replay/watchdog仍为原进程，日志采样年龄均≤1秒，终态从
411增至1051，launch.json尚未形成。最后sample1168，host可用88.93GiB、
service peak21.91GiB、磁盘144.89GiB，所查high/max/OOM/kill、foreign/
escape均0，无保护告警或中止。服务仍在进行，不停止/重启或并行新任务。
此轮仍为verified wait，不能将四分之一负载当完整性能结论；下一动作继续
同一回放，终态后资源释放、完整指标/机制审计、保存和安全备份。

## 唯一假设与前置证据

D1029 的完整 W2 为 3774/4000 联合达标（94.35%）、8167.260452414863
GPU-s、P95 TTFT 1890.5496881343424 ms，离共同门槛仍差26条。
其编译已热的自然副本仍约28秒原生初始化，重复显存估计是可观测开销。
D1031/D1032 的唯一候选复用实际成功 KV 预算，不采用原生建议容量，
不改模型、32槽、并发8、token轴、GPU预算、九公式、扩缩容或 admission。
官方 gpu_worker/startup_plan 路径及历史失败方向见 D1031/D1030。

D1031 source2 已完成423请求、160服务类/96准备类及真实远端审计。
D1032 仅选原前两波16请求，实际 worker 回执 status=hit，指纹与原记录
相同、producer PID4043841；原生预算8092388312 B、block池8092385280 B、
32槽与LoRA池1825046528 B未变。真实服务图仍捕获一次，原生 init16.26s，
compile0.26s。不是全部冷/热差异的单因素因果证明，数值正确性仍未闭合。
不再追加短 Full 或重新收集423请求。

## 本次冻结配置与判据

配置为 configs/ieee_tc/20261103_d1033_3b_startup_memory_w2.yaml。
相对 D1029 仅14处结构变化：三个独占输出/缓存路径、新启动政策、两个
completed-length ID、两份新画像的路径/SHA/输入及资源上下文身份。
使用既有 strict loader 实际 dedicated 配置验证160/96类通过。
启动仍用 fastsafetensors_affinity_preserving_v1，和画像的loader不同会产生
自己的冷记录；不复制记录、预跑Full、抹去编译因子或降低free-memory条件。
自然后继副本是否命中必须用实际回执，不以较快初始化推断命中。

源输入仍seed42已知开发回放，4000请求，500逻辑adapter，W2 rotation100。
trace SHA 4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d；
W2 map SHA 28c97dc7c1ffcbad49801ec19bec2a9c4cf9c4915c88a7e5163e223ece315e23。
固定输出greedy、相同prompt/target/arrival、共同提前60秒部署全部保留。
原生remote读取、传输、竞争、物化计入；使用既有只读压缩交付对象，不打包。

PLAN SHA d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83；
V1 SHA 5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22；
V2 SHA 193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d。
D648 SLO：prompt≤506/>506，TTFT1466.050801102/2539.648668191 ms，
TPOT29.921872367/36.215515193 ms。至少3800联合达标；G1先要求全部正确，
再比GPU-s；G2另要求U≤12083.063870，才比较P95与本轮SLO差。
不计算CI、不重复同键、不挑请求阶段、不改分母；n_correct未知不填4000。

## 执行、保护和终态

复用 scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh，
虽然旧文件名含W1，实际明确W2。RUN_ID=d1033_20261103，
RUN_TAG=d1033_3b_startup_memory_w2_full；CONFIG为上述新配置。
保持D747 descriptor、stdlib gzip、相同已冻结本地/远端资源环境。
显式清除prefix与详细profiling开关；唯一完整回放，不做暖启动重跑。

推理service72/80GiB、swap2、CPU4–23/28–47；aux3/4GiB、swap0、
CPU2/3/26/27，独立watchdog和native-worker归属核验继续生效。
测前MemAvailable102GiB、磁盘150GiB且覆盖预测增长；运行底线16/100GiB。
不在回放期间进行压缩、构建、远端重启、备份或其他重型任务。
本轮为一个新完整执行键，旧D1029及D1031失败/成功证据全部保留。

结束先验证真实GPU释放，再按D703分析4000请求、冻结SLO、GPU-s、
阶段/每个worker冷命中及KV容量、画像类和双端transfer证据。
D703只对原生日志明确 skipped memory profiling 时允许一次真实capture，
其他仍要求两次；该分支不单独宣称本政策命中，须绑定worker回执。
47项相关CPU测试已通过，原既有警告6个；未做多余基础资格矩阵。

D810外部baseline18点与已完成适配保留，不重跑来改变对手结果。
W2的vLLM联合100%、U16110.752、P95441.635ms；ServerlessLLM98.125%、
U16084.201、P95480.724ms。它们超75%预算但原始延迟更低，不隐藏。
达标后推进同一最终配置W0/W1和完整外部对比，不继续局部启动调参。
未达标保留反例，回到完整请求阶段证据；不强行将微测通过写成G1/G2。

全部剩余范围不删减：两模型同最终版本W0/W1/W2/G1/G2、独立数值正确性、
最终基线比较、A1–A5、S1–S13、CE/Pareto、B1/B2/C1/C2/C3和两轮审稿文档、
接近目标后图表、最终V2集成。备份分支不冒充历史分歧V2已集成。

## 14:13完整结果：启动子阶段缩短，整体目标仍未达成

同一完整回放4000条全部终态；service/watchdog/replay退出码均0，实际GPU
context释放、service资源域消失，独立nvidia-smi无compute进程。没有重跑、
修改测量中的配置/源码、进行压缩、推送或远端管理。分析始终使用冻结V1/V2
及D648阈值，不使用日志99.325%的旧5000ms TTFT显示判定联合达标。

| 指标 | D1029 | D1033 |
|---|---:|---:|
| 原生完成/失败 | 4000/0 | 4000/0 |
| 联合TTFT/TPOT达标 | 3774/4000 (94.35%) | 3772/4000 (94.30%) |
| TTFT达标 | 3867 | 3850 |
| TPOT达标 | 3901 | 3913 |
| 物理生命周期GPU-s | 8167.260452414863 | 8155.283735964913 |
| GPU-s/offered-request | 2.041815113103716 | 2.038820933991228 |
| 平均TTFT/ms | 763.3946093007689 | 765.1991312011378 |
| P95 TTFT/ms | 1890.5496881343424 | 1956.042105332017 |
| 平均TPOT/ms | 15.710652143074844 | 15.610976187487047 |
| P95 TPOT/ms | 27.332940477539193 | 27.056324062868953 |
| 平均E2E/ms | 2493.134081077995 | 2481.9043227854418 |
| P95 E2E/ms | 4941.7237606830895 | 4820.602268911898 |

联合净少2条（−0.05百分点），离3800仍差28条；GPU-s减少11.97671644994989
（0.146643%），P95 TTFT增加3.4642%。P95 TPOT降低1.0120%、E2E降低2.4510%，
是取舍，不是整体最优。U虽小于12083.063870预算，联合约束未满足，不能进入
G1/G2达标排名。独立数值正确性仍未知，n_correct=null、ranking_eligible=false；
不能把原生token/绑定检查等价为数值正确。仅固定已知负载的单轮描述，不作CI、
显著性、独立留出或纯启动单因素因果主张，新初始化画像仍是执行条件的一部分。

### 完整请求、阶段与对应变化

- 同请求配对：联合fail→pass116条、pass→fail118条；TTFT净少17、TPOT净多12。
- TTFT-only141、TPOT-only78、两者9。150条TTFT违约的最大互斥阶段为
  arrival→admission64、admission→acquired84、executable→engine2；不能都归给GPU计算。
- 八个预定500请求phase的联合数为465/479/484/476/477/466/468/457。
  后3500请求3307条联合通过，D1029为3306；首phase为465对468。保留全部
  4000分母，不删除冷启动或宣称扣掉启动后即可达标。
- 首次实际admission前到达26条，全TTFT违约；这是观测子集，不是改动请求
  时间线的理由。完整平均TTFT约765ms，不能用首25条约29秒替代全局均值。
- 4000条dispatch tier完整：GPU2998/HOST12/NVMe500/Remote490；实际调用
  native load995次。98个service类、52个preparation类与D1031画像严格匹配。
- 原生token/adapter绑定/到达/时间线检查通过，TTFT/TPOT/E2E重算误差均0；
  4000条D747 descriptor的库SHA匹配，但仍不补造数值正确性。

### 全部worker的实际启动证据与边界

| 原生PID | activation类别 | 请求数 | engine activation/ms | native init/s | compile/s | 跳过显存估计 |
|---|---|---:|---:|---:|---:|---|
| 4125618 | initial | 2015 | 112125.341595 | 77.14 | 32.66 | 否 |
| 4129029 | natural_scaleout | 1955 | 37835.326664 | 16.79 | 0.27 | 是 |
| 4128955 | natural_scaleout | 7 | 50263.676083 | 28.22 | 0.28 | 否 |
| 4173524 | natural_scaleout | 23 | 41859.053635 | 17.53 | 0.28 | 是 |

类别通过真实activation记录、native_owner、请求replica及物理lease绑定，
不是根据启动时刻猜测。native init是engine activation内部子阶段，不能相加。
两个skip worker仍各捕获一次真实服务图（66/24）；另外两个含临时估计图。
全部4个worker日志及Full生产者的小型缓存记录已归档，5成员逐一SHA核验，
原日志和唯一缓存记录没有删除；archive SHA
706638a93be8b98b32c4e0f69f4ecf9cc983ed9a2ee189eeaca91fdf37a19b20。

重要限制：D1032短验证确实保存了startup_memory命中回执；Full虽然调用worker
observation确认物理归属，却未将该结构化回执写入最终结果。D1033结果完整
扫描无startup_memory字段，因此不能逐worker断言status/producer/未命中原因。
本次只接受原生日志明确的skip分支与实际类别绑定。保留的Full缓存fingerprint
为5ad8d20703b33f81bed212492a108f561ebbaf773f8e4eeca9e04b9069f1c94f，
producer PID4125618，实际成功预算8096582616 B；这是生产者事后不可变记录，
不是两个消费者的命中回执。不能用D1031的8092388312 B冒充Full实际预算。
此观察缺口不改变不达标结论；不为补回执重跑本性能点，也不放宽身份校验。

### 真实远端、资源与保存

503次真实传输、1167457984线上字节，双端ID/bytes/archive SHA匹配，无请求期
打包，服务未重启。没有连续远端CPU/外部任务监测的既有限制继续披露。
全局传输与selected-Remote请求不是同一集合：490条Remote中488条可按共享
预取及native文件见证关联486个不同传输。4000条route-prefetch记录完整，
493 completed、3507 skipped；不能把shared等待或包含多个操作的时间段
称为纯网络时间、文件锁时间或可加和因果损失。

4047个资源采样；service峰值30419263488 B，主机最低可用93137031168 B，
最低磁盘144775487488 B。high/max/OOM/kill/swap、foreign compute和escape均0。
3次natural scale-up、2次scale-down，全部4个allocation释放；运行HOST/NVMe
工作缓存清空。147项旧投稿保护SHA和两个用户dirty文件SHA未变。

分析产物：paper_results/ieee_tc/p2_backend/d1033_3b_startup_memory_w2/。
completion含full4000、stages、paired_d1029、profile_bindings、native_descriptor、
delivery_and_resources、closure及原collector快照；另有native_startup、
startup_binding及remote_preparation_shared。completion manifest不追改。
全部分析在CUDA不可见、CPU2/3/26/27、3/4GiB、swap0辅助域内串行执行。

保留离线调用问题，不混作GPU失败：首次paired调用误传diagnosis.json而非
目录，在输出前拒绝；改正路径后完成，不重跑已成功stages。首次collector
使用了31位而非32位辅助域标识，在输出前拒绝；随后原代码使用合法UUID完成。
最先remote_preparation采用direct-only模式，只关联8条；该有限分析保留，
另以原有include-route-prefetch模式生成remote_preparation_shared，未覆盖前者。
experiment-audit技能引用的配套规则不存在，本次是项目确定性检查，未伪称
fresh-agent独立审计通过；数值正确性仍未闭合。

### 决定与下一主线

本候选**不接受为G1/G2达标结果，也不宣称整体改进**；保存为有效取舍/负结果。
停止这一轮局部启动优化，不重跑暖缓存，不通过删前26条、改阈值或筛seed过线。
后续先从已有完整等待/准备证据及历史已否决方向定位一个新的可证伪瓶颈，
再参考对应官方实现；尚未登记或启动下一GPU候选。本次并未修改生产代码。
若以后继续采用该可选启动机制，应在正常新版本里持久保存实际启动回执，
不能用该观测缺口另开一轮没有新性能假设的重复实验。

D810外部18点仍复用，不恢复HydraServe/Loquetier基础适配。3B当前尚未达到
共同联合SLO，不能直接切换模型或宣布跨场景达标；后续同最终配置W0/W1/W2、
独立数值正确性、完整外部对比、A1–A5、S1–S13、CE/Pareto、审稿/设计文档、
图表及最终V2集成全部保留，完整目标active。先封存备份，再在测量外恢复必要
磁盘余量；旧结果、工件、负载、环境和用户文件不覆盖。
