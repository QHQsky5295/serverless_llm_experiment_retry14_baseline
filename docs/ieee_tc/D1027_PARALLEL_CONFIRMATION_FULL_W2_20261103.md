# D1027：独立文件并行确认的完整3B W2验证

2026-11-03。唯一完整回放已完成，离线核验通过；联合SLO为94.60%，未达标，
全部GPU已释放。当前交接以下方“完成结果”节为准，不再启动本执行键。
接续D1026的94项相关回归和三个
真实工件的组件观察；仅组件发布wall下降10.43–18.93%，累计CPU反而增加，
故必须以完整回放判断净收益，不能由局部值推造SLO。

## 完成结果、归因限制与当前交接

本回合为progress：确认原运行终态，完成共同SLO、4000请求阶段、画像绑定、
双端交付/资源闭合，以及启动和远端准备的离线分析。没有重复GPU运行。
使用analyze-results与vLLM技能核查实际原生证据，不采用技能中的通用吞吐
示例或500ms目标替代冻结V1/单轮V2；无图表、论文修改、阈值修改或CI。

| 3B W2单轮观测 | D1022串行确认 | D1027并行确认 |
|---|---:|---:|
| 原生合同完成/失败 | 4000/0 | 4000/0 |
| 联合TTFT/TPOT达标 | 3770（94.25%） | 3784（94.60%） |
| 物理生命周期GPU-s | 8062.415716 | 8316.260704 |
| GPU-s/offered-request | 2.015603929 | 2.079065176 |
| 平均TTFT ms | 825.647517 | 623.772085 |
| P95 TTFT ms | 1997.601528 | 1970.868099 |
| 平均TPOT ms | 15.491261 | 15.610764 |
| P95 TPOT ms | 26.119779 | 27.179632 |
| 平均E2E ms | 2542.603823 | 2342.806522 |
| P95 E2E ms | 4899.350524 | 4835.258111 |
| 原生按需load数 | 1040 | 999 |
| 自然scale-up / scale-down | 2 / 1 | 5 / 4 |

联合净增加14条（0.35个百分点），物理GPU-s增加253.844988（约3.15%）。
P95 TTFT降低26.733429ms，但P95 TPOT增加1.059853ms。仍差16条达到3800，
不能以U低于12083.063870预算代替服务要求。完整请求/原生生成/adapter绑定
核验通过；独立数值正确性仍未完成，n_correct=null、ranking_eligible=false。

**决定：保留完整开发候选及生产实现，不宣称文件并行校验已证明端到端净收益，
不冻结为最终工作点。** 当前实现可继续作下一项有证据候选的开发基点，但
D1022低资源点及所有旧候选保留，不因总SLO高14条而丢掉退化或来源差别。
不重复同key找赢家；同版本W0/W1仍待完成，不能与历史版本拼接。

### 1. 必须保留的启动混杂与阶段变化

六个实际worker均以allocation journal的PID绑定原生日志，覆盖全生命周期。
D1027初始init为32.76秒，其中编译0.27秒；D1022初始为82.56秒，其中编译
31.00秒。D1027日志明确直接加载既有AOT编译缓存，未人为预热或清空缓存。
因此总体平均TTFT降低201.875431ms不等于并行SHA的因果收益；两次启动
编译状态不同必须披露。不能把本轮改名为严格单因素性能消融。

D1027其余五个worker的init依次为36.07/36.74/34.10/35.29/33.98秒，
编译均0.26–0.31秒，内存估计及实际服务的捕获日志分别为14+14、14+15、
12+15、12+16、12+14整数秒。初始为11+14秒。实际服务捕获集合仍为88个
PIECEWISE和32个FULL描述；未改变token轴、LoRA集合或dummy分支。
这是原生日志的阶段证据，不是每个kernel的因果时间。

八组500请求的联合数为495/437/485/480/481/479/467/460，相比D1022为
+36/-39/+2/+4/+4/+12/-3/-2。首500请求平均TTFT减少1870.006873ms，
第二组反而增加306.763066ms；后3500联合为3289，低于D1022的3311。
这些分组解释差别，不改变完整4000分母、不删除首段或第二组。

同请求配对：联合fail→pass136、pass→fail122；TTFT为91/87，TPOT为58/63。
六段平均TTFT差（D1027−D1022）为arrival→admission −195.019653ms、
admission→acquired −5.413240、executable→engine −1.174514、
engine-entry→queue +0.638442、native queue −0.243333、prefill −0.663133。
恒等式最大误差0；嵌套routing不重复计入。TTFT-only136、TPOT-only59、both21。
后3500的TTFT违约主导段：arrival→admission80、admission→acquired73。
这不是“只剩启动问题”，也不证明再次调扩缩容阈值合理。

### 2. 文件准备证据：局部改善，不推造整体收益

全部505次传输完整保留。实际发布包络均值266.560826ms，D1022为318.542ms；
本地完整传输包络777.536906ms，D1022为864.340ms。不同运行传输集合和
闭环状态不同，这不是逐transfer配对效果或可直接从旧TTFT扣除的量。

502条dispatch-Remote中497条可按真实回执分解（直接8、共享预取489），
关联496个不同transfer。其平均admission→acquired969.935725ms：发布
266.177054、发布至原生load252.886374、最终load39.500501ms。86条TTFT
违约的相同条件分组平均准备1778.267196ms，仍有明显准备和交接长尾。
另5条保留为unjoined，其中4条原生来源复用/变化、1条共享文件复用，不能
猜测关联、补0或删请求。分解误差0，完整分母与原SLO保持。

预取记录覆盖4000条：completed500、skipped3500；495个I/O owner，5个
共享消费者，5条在admission前完成。发布后任务结算均值356.339139ms。
这些重叠跨度不再叠加到用户TTFT，也不称为纯网络/纯锁等待。

### 3. 资源、来源、失败记录与保护

- 测量源码6099467a39f0f2a2c4d2db6b776cb0c442000d3d，配置身份仍为下节SHA。
- 原始请求结果288289134B，SHA为
  5afcd08a24212d658c0b7839a5b271a81a0963e5509e43c6c930717dbe235841。
  main_outcome476053009B，SHA为
  4d0a853b99d226f327f6b05a1ba43f9ef9745b61e6a063c462bd6d2b794377ee。
- 数据目录paper_results/ieee_tc/p2_backend/d1027_3b_parallel_confirmation_w2/：
  completion含full4000、stages、paired_d1022、profile_bindings、descriptor、
  delivery/resources、closure和manifest；其外另有native_startup.json与
  remote_preparation，不覆盖已封存completion manifest。
- 4000条D747 fresh descriptor SHA一致；dispatch tier为GPU2994、HOST3、
  NVMe501、Remote502，95个service类和51个preparation类绑定通过。
- 505次远端交付、1172100920线上字节，两端ID/字节/归档SHA/无请求打包通过。
  未重启既有远端服务；仍无连续远端CPU/干扰监测，不声称排除全程干扰。
- 6个物理allocation全部释放，service/watchdog/replay返回0，服务域已消失，
  当前GPU compute为空。4049个资源样本foreign/escaped均0；high/max/OOM/
  kill/swap为0，服务峰值30512488448B、主机最低available94495055872B、
  磁盘最低144598511616B，无安全中止。
- 最后离线收集的辅助快照峰值532144128B，high/max/OOM/swap为0，scope
  已结束。此前阶段与画像绑定两个离线分析曾短时并行，各约0.53GB；没有
  GPU测量或维护与其重叠，不将它们误写为全程串行或完整零事件证明。
- 首次离线collector使用非TC UUID命名的scope，被既有身份检查在读取结果
  前拒绝（ValueError: not an owned TC service/test UUID scope）。随后以合法
  aux UUID运行通过，没有放松检查、覆盖输出或重新回放GPU。
- 147项旧投稿文件和两项用户dirty SHA不变，不修改/暂存EXECUTION_STATUS。
  六份原生worker日志另存raw/native_startup_logs.tar.gz，SHA为
  823a7ef8e8717eacaf36429d325c4b4710029d521c137586db05649d6d779dda。
  原/tmp文件未删除；归档逐项内容核验后可恢复来源，不依赖/tmp永久存在。
- 仅在GPU/分析结束后复用既有工具无损压缩D1020 main_outcome及D1017原始
  result，两次解压SHA核验后移除未压缩重复件，分别回收407138304和254586880B。
  新回执在paper_results/ieee_tc/storage/d1027_closed_{d1020,d1017_result}_20261103.json。
  可用磁盘161160937472B，约150.09GiB；下一重任务仍即时检查，不重复全盘扫描。

### 4. 下一主线，不缩减完整目标

结束本次文件并行微优化的评估，不继续搜索线程数或重复CPU/GPU测量。
下一项先检查D1023已经识别的执行图可达性：max_num_seqs=8却捕获16/32
活跃LoRA分支；保留32驻留槽、token轴及dummy=33，不能粗删哨兵或静默回退。
本轮五次实际扩容、所有worker在编译热态仍有25–29秒捕获，是新的完整运行
证据；但尚未实现或证明裁减可减少违约。须先结合历史关闭方向和官方实际
V2实现验证可达集合，再决定一个独立候选，不叠加扩缩容参数修改。
文件准备长尾同时保留，不以容易修改的启动问题宣称解释全部216条违约。

外部比较继续复用D810全部观测，不重开HydraServe/Loquetier基础适配或共享
k3s。vLLM W2为100%联合、16110.752 GPU-s、P95 TTFT441.635ms；
ServerlessLLM为98.125%、16084.201、480.724ms。它们超75%预算但服务质量
更好；D1027仍不能宣布同预算最优或G1达标。其余低分/失败点不隐去。
达到可用工作点后立即推进同版本W0/W1与完整比较；独立数值正确性、两模型
G1/G2、A1–A5、S1–S13、CE/Pareto、B1/B2/C1/C2/C3、两轮审稿文档、后期
图表及最终V2集成全部保留。完整目标继续active。

后置备份：21文件检查点7f5da3f82c888bf3be90ebc3805a4bba1924deb8已推送，
ls-remote核验origin/retry14_continuous_queue_v2及faaslora_origin的
backup/d1027-parallel-confirmation-w2-20261103均为该SHA。后者是回退备份，
不是已有分歧V2的最终集成；未强推。10项completion manifest、6份归档
worker日志SHA、collector语法、147保护项、diff及暂存凭据模式检查通过。
用户manifest/EXECUTION_STATUS未暂存、SHA不变。未改生产算法，沿用D1026
相关回归及本轮严格原生/闭合分析，不新增无目的GPU smoke。
最后可用磁盘161150361600B、主机available约105GiB；GPU compute为空，
本轮性能和离线分析均已结束。此后置说明随下一正常检查点提交。

## 唯一改变与主线（运行前登记，历史保留）

相对D1022，只改变发布事务内独立文件SHA校验的执行并行性，保留owner锁、
全部内容/签名/容量检查、join后发布和失败清理。D628解锁hash等已否决路线
不重开，不叠加capture或控制策略；九公式与核心机制不变。具体历史、原始
资料与CPU证据见D1026，不新增短GPU诊断或后端画像。

配置`configs/ieee_tc/20261103_d1027_3b_parallel_confirmation_w2.yaml` SHA
a5898c34f9abdb33cd3c82e4f3ef944a0c70865598beafe130aa6c4a0e58faa3。
与D1022结构比较后，除三个输出/缓存路径外所有字段逐项相等。32 GPU/CPU
LoRA slots、并发8、native active-LoRA specialization、原D1021服务/准备
画像、原生vLLM0.30、D747 descriptor、stdlib gzip、1–3实例、全部控制参数
保持不变。画像对应的原生后端未变；准备初值保留同一冻结观测，在线仍按原
EWMA更新，不按场景或正式结果另调参数。

输入复用既有4000请求、500-adapter universe、seed42、W2 rotation100、
同arrival/prompt/target与fixed_length_greedy_v1；原source trace已用于开发，
不是独立留出。60秒部署通知、真实已发布远端工件、完整生命周期计量不变。
原W2 map SHA为28c97dc7c1ffcbad49801ec19bec2a9c4cf9c4915c88a7e5163e223ece315e23。
不覆盖D1022，不重跑同key找胜出。

PLAN、V1、单轮V2的SHA与D1026一致；D648共同SLO不改：prompt≤506/>506的
TTFT分别1466.050801102/2539.648668191ms，TPOT分别29.921872367/
36.215515193ms。G1要求全部正确且至少3800联合达标，再比较物理GPU-s；
G2另需U≤12083.063870GPU-s，再比较W2 P95及观测SLO差。独立数值正确性
仍须按真实证据判断，不能用binding/token通过填造n_correct。

参照D1022：3770联合达标（94.25%）、8062.415716 GPU-s、P95 TTFT
1997.601528ms。并列报告TTFT/TPOT/E2E、失败、来源、资源与CPU代价，
不只看比旧候选提升。外部D810基线完整观测保留；达标后立即推进同版本
W0/W1及全基线比较，不无限停留在局部校验微测。

## 入口与保护

```bash
env -u FAASLORA_TC_CORE_TIMING -u FAASLORA_TC_CORE_STEP_TIMING -u FAASLORA_TC_HOST_INVENTORY_TIMING \
  FAASLORA_IEEE_HOST_INVENTORY_MODE=descriptor \
  FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d747_20261023/_ieee_native_host_descriptor.cpython-312-x86_64-linux-gnu.so \
  FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256=b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044 \
  FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER=stdlib \
  FAASLORA_D1000_RUN_ID=d1027_20261103 \
  FAASLORA_D1000_RUN_TAG=d1027_3b_parallel_confirmation_w2_full \
  FAASLORA_D1000_CONFIG=configs/ieee_tc/20261103_d1027_3b_parallel_confirmation_w2.yaml \
  bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

原入口文件名含W1，实际固定W2；不设prefix。外层tmux保持长运行，已有资源
入口在启动worker前建立服务72/80GiB、swap2、CPU4–23/28–47；辅助3/4GiB、
swap0、CPU2/3/26/27及独立watchdog。空机内存≥102GiB，磁盘≥150GiB且
覆盖32GiB预计新增量的规则，运行时16/100GiB保护。一次一个重型任务，
不与编译、维护、备份、远端配置变更并行。

预查远端：认证health为artifact_timing_v2/prepublished_gzip_v1，clock_id
b7a04b62f38c4ae6adabb4da5c0fd8dc，端口18080 PID1984520，沿用D1016–D1021
已记录进程；它不属于旧failed的primelora-artifact-tc-3b.service。本轮仅
只读确认，不错误重启旧unit。实际管理归属为远端session-8529.scope。
两端sysfs均1000Mb/s，远端full；不把D77历史100Mb/s当当前链路状态。
没有请求中打包、额外复制池、共享k3s操作或远端后台压力测试。

raw独占目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1027_20261103/d1027_3b_parallel_confirmation_w2_full`。
完整运行后先验证实际release，再复用D703 outcome审计入口显式指定本配置和
D1021 manifest，恢复SLO/生命周期/原生生成/双边传输和阶段证据。失败和
退化照实保留，不能删点或换尺子；不提前绘图。

本文件是当前执行交接，不覆盖用户dirty EXECUTION_STATUS；其SHA仍为
41a4db5bc464dfb885b20d2b5ac321a3631a1a02291d64ac823735505957e374。
所有未完范围仍保留：同版本两模型W0/W1/W2及G1/G2、独立数值正确性、
A1–A5、S1–S13、CE/Pareto、B1/B2/C1/C2/C3及两轮审稿回应、后期图表与
最终V2集成。当前目标active，不宣布3B完成或全项目完成。

## 启动与备份交接

代码检查点6099467a39f0f2a2c4d2db6b776cb0c442000d3d已推送并ls-remote核验：
origin/retry14_continuous_queue_v2，以及faaslora_origin的
backup/d1026-parallel-confirmation-20261103。后者是可回退备份，不冒称历史
分歧V2已最终集成；无force push。10文件明确暂存、diff/源SHA/凭据模式
检查完成，用户manifest/EXECUTION_STATUS及147旧投稿保护不变。

实际复用上述入口，外层tmux会话tc-d1027-full；额外显式unset旧prefix和
native-profile诊断环境变量。raw新目录由mkdir独占创建，无覆盖旧结果。
启动前磁盘161377443840B，MemAvailable110189088KiB，GPU compute为空。
启动回执确认服务scope为primelora-tc-svc-d2627e6ebd9b4ce9a2f1d3bd9fe96e93，
invocation03f8d2aa66234b408e65347ace11e4b6；独立辅助scope为
primelora-tc-aux-59c4a4725731464fa5eb15671f6a803e，watchdog PID2720453。
实际replay_ready count/source_count均4000，W2 view SHA为
5558de43ede0e2cf49c7478dcfb94640e7a8516406f53766fabfdf524bf91174；
source trace SHA为4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d。
当前VLLM EngineCore PID2722076已分配GPU，尚在初始准备；无结果时不填SLO。
发现旧D998辅助scope仍列active但cgls为空，不是第二个正在测量的GPU服务，
未在测量中调整该scope或共享服务。

下一步仅监控本轮及资源保护，完成后按既定顺序释放/审计/同合同对照；
不在运行期间改源码、调参数、压缩清理、备份或管理远端服务。本启动回执
作为后置文档随下一个正常检查点提交，不为更新文档再次打断性能测量。

### 02:46持续运行核查：第一组500请求已完成

本目标回合为verified wait：完整重读目标和PLAN，按monitor-experiment技能
只读核对原tmux、service InvocationID及真实watchdog/EngineCore进程，
对同一handle作12次间隔约45秒的轻量观察。没有重启、重跑、配置/源码改动、
维护、备份、远端操作或新GPU任务。

最新完整Live行：arrived529/4000、done528、ok528、fail0，两个runtime，
尚有1条处理中。第一组500请求已走完，这只是原生执行进度，不等于独立
数值正确性或最终SLO达标；不从界面旧5000ms统计推断共同联合要求。
watchdog样本738仍无warning/abort，high/max/OOM/kill/swap为0；服务当前
16011636736B、峰值17810493440B，主机可用99785392128B。该时刻磁盘
157885882368B，约147.05GiB：低于下一重任务的150GiB启动线，但高于本轮
100GiB运行底线，不因此中断有效测量或并行清理。完成释放后再评估维护。

原watchdog PID2720453和EngineCore PID2722076均live，service InvocationID
保持03f8d2aa66234b408e65347ace11e4b6。用户manifest/EXECUTION_STATUS SHA
仍与上节相等。继续同一D1027至终态，再做release、完整离线核验及对照；
尚无本轮完整性能结果，未删减全部后续范围，目标继续active。

### 03:05继续同一回放，接近半程

继续按原tmux/service InvocationID核查真实进程及新增日志；本轮再做24次
间隔约45秒的轻量观察，始终为同一运行，没有重启。最新Live为arrived1902、
done1901、ok1901、fail0（完整计划分母仍4000），两个runtime。期间自然
副本数在1–3间变化，实际经历缩容、到达集中和再扩容；不在运行中干预。
watchdog样本1856无warning/abort，high/max/OOM/kill为0，服务内存峰值
21.962356567GiB，磁盘141.171436310GiB。不是完整GPU积分或最终SLO结论。
继续等待完整终态；未启动任何第二候选、构建、维护、推送或远端管理。
