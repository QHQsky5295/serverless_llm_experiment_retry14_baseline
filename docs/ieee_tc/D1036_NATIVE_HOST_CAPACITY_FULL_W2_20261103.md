# D1036：预算推导 CPU adapter 容量的完整 W2 验证

2026-11-03。当前目标active，主线3B PrimeLoRA；**完整运行已闭合，83.00%联合达标，候选拒绝，禁止重复启动**。
本文件是最新交接，不改用户dirty的EXECUTION_STATUS.md。D1035画像已完整
结束并导出，不得重复，也不额外添加短Full。HydraServe隔离适配和D810基线
结果已存在，不接管共享k3s，不重开基础适配。

## 唯一问题与父点

D1033为开发父点：4000原生完成、3772联合达标（94.30%），物理GPU时间
8155.283735964913。它未达到3800，也未因GPU时间小幅减少被接受为改进。
D1034在完整日志中发现CPU容量32与GPU槽32相同、CPU缓存3868次达到上限、
实际原生CPU-only来源0次；现有2GiB子预算未充分利用。公式倒推得到110条，
保留两份工作区与观测非注册占用，仍执行实际字节准入；不作容量网格扫描。
历史、一级来源、推导和限制详见D1034。重复加载可能减少，状态检查可能变贵，
最终以完整结果判断，不预设能补齐28条联合违约。

## 冻结配置与输入

`configs/ieee_tc/20261103_d1036_3b_native_host_capacity_w2.yaml`
SHA f5435ebc10ca2e77cd14c3cb193a0e0afc0d6ce7c6260876a041d132e9cc746e。
相对D1033仅14处结构变化：CPU上限32→110、两个完成长度身份、两组新实测
画像路径/SHA/上下文及三个独占输出/缓存路径。严格运行配置装配后160服务类、
96准备类均能载入；保持启动政策、GPU32槽、并发8、token预算4096、1–3
副本、16GiB HOST、2GiB原生张量预算及IEEE九公式。实际物理分配日志新增
startup_memory仅为观测字段，不更改决策或计费，也不补写旧结果。

D1035画像manifest：c33175472fcc6739193808cdc7ffdbda4930693e6b297631efb28f3af9e409a7。
service：b1fca5b24c7ef91087851a744fa5b3cae038194df016a3a080ef0e20c184ebb1；
preparation：9849e90fd1b1a160baec3ee2e9be582d3b0aecb499f65787524e7f14083a5348。
24工件画像不能证明110条填满，Full须核对CPU-only复用、加载次数和相关
准备/控制开销，不把dispatch HOST误当原生CPU命中。

既有seed42开发trace，4000请求/500逻辑adapter；W2 rotation100、相同arrival/
prompt/target和fixed_length_greedy_v1，提前60秒部署，准备及释放持卡全计。
trace SHA 4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d；
map SHA 28c97dc7c1ffcbad49801ec19bec2a9c4cf9c4915c88a7e5163e223ece315e23。
真实远端使用已发布只读对象，实际读取/传输/竞争/本地物化计入，不打包或
扣除真实传输。远端快照不冒称连续监控，保留共同限制。

## 执行与判定

继续复用 `scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh`。
外部变量：FAASLORA_D1000_RUN_ID=d1036_20261103；
FAASLORA_D1000_RUN_TAG=d1036_3b_native_host_capacity_w2_full；
FAASLORA_D1000_CONFIG=上面配置路径。runner已固定W2，历史文件名不改。
显式取消三个TC详细timing、D1000_NATIVE_PROFILE、D1000_PREFIX_COUNT、
TC_DIAGNOSTIC_PREFIX_COUNT及VLLM_ENABLE_STARTUP_PLAN，不注入额外画像。
只启动一次完整键，run目录独占创建；不复制缓存、不强制命中或重测冷/热版本。

PLAN SHA d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83。
冻结V1/单轮V2及D648阈值不变：G1全部正确且至少3800联合达标后比GPU-s；
W2 G2还要求U≤12083.063870并比较P95 TTFT和观测联合差值。不用控制台
5000ms作最终判定，不作CI。独立数值正确性仍未知，不补造n_correct。

启动前PLAN、实际资源和既有147项保护检查；服务72/80GiB、swap2，辅助
3/4GiB、swap0，原CPU集合与独立watchdog不变。启动102GiB内存/150GiB
磁盘与预计增长，运行16GiB/100GiB底线；不得在测量时构建、压缩、推送或
远端变更。失败/退化保留，不扫描其他容量或重复求胜；先关闭本候选再决定
下一瓶颈。仅保存数据，达标前不为展示画图。

完整剩余范围：两模型同最终版本W0/W1/W2/G1/G2、独立数值正确性、外部
基线最终比较、A1–A5与S1–S13、CE/Pareto、B1/B2/C1/C2/C3及两轮审稿文档、
IEEE图表和最终V2集成。检查点备份不等于历史分歧FaaSLoRA V2已最终集成。

## 实际启动与运行中交接

上一目标轮为PROGRESS：完成D1035原生画像、严格导出、无损回收及D1036
配置准备。本轮先完整阅读目标文件和PLAN（长输出截断处另分段补读），
读取当前/历史状态、冻结V1/单轮V2；使用run-experiment和monitor-experiment。
不再跑已通过的74项测试或423请求，不改变候选、SLO和剩余范围。

2026-11-03 15:05:43，以源码da3ff76978745e5a6a2ff5a9d724e7bac9f29a87
在tmux `tc-d1036-full`启动唯一完整W2。源码此前已推送并ls-remote核验
origin/retry14_continuous_queue_v2和faaslora_origin/
backup/d1035-native-host-profile-20261103；后者仅备份，不强推分歧V2。
原runner复用，显式保持D747 descriptor、stdlib gzip、CUDA13工具链身份，
清除上述timing/prefix/原生额外startup-plan开关。配置SHA与冻结值相同。

启动前GPU计算进程为空；147旧保护项零变化，两个用户dirty SHA未变。
可用磁盘161309360128B，主机可用内存111750352896B；没有并行重任务。
远端认证health、PID1984520/start和两端1000/full只读确认，无服务变更。
raw目录：
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1036_20261103/d1036_3b_native_host_capacity_w2_full`。

实际service scope为primelora-tc-svc-3f13dda56288424e8e937fae5c912830.scope，
aux为primelora-tc-aux-74cad8184a2f479c93beea2b44773066.scope。
replay PID248900/start_ticks319679154；独立watchdog PID248951/
start_ticks319679317。共同deployment notice3196793.47991914、
业务t0=3196853.47991914，60秒未改，count4000/W2/map/view均符合冻结输入。

初始worker PID250500实际startup_memory=miss、publication=created，
KV8096582616B；后继PID253814/253937实际hit、producer250500，容量相同。
这些是同一真实physical journal中的回执，不凭启动快慢推断；activation分类
和完整开销仍待结束后关联。无复制缓存、强制命中或额外暖启动。

15:11:16只读核查：sample326，159条终态记录；两个PID及start_ticks均仍
匹配，日志采样年龄0.75秒，launch.json尚未形成，不能判为终态。服务峰值
17960034304B，主机可用98677600256B；high/max/OOM/kill、foreign compute、
逃逸均0，无保护告警或中止。磁盘159561052160B已低于下一轮启动150GiB，
但高于运行100GiB底线；不在测量中压缩，结束释放后再核查/维护。
这是运行中观察，不是完整4000请求达标或安全闭合声明。

15:15:40继续只读核查：终态记录增长至433，watchdog sample587，采样年龄
0.59秒；最终launch.json仍未产生。15:15左右已再次确认原publisher和watchdog
的PID/start_ticks均匹配，未重启。服务峰值仍17960034304B，主机可用
97916813312B，磁盘158133997568B；high/max/OOM/kill、foreign compute和
逃逸均0，无保护告警。仅更新运行中交接，不将部分请求外推为完整SLO结论。
本次上下文恢复后的推进为verified wait：实际任务存活且终态记录持续增长；
不重复画像、Full或HydraServe适配，不在测量期间开展磁盘维护。两个用户
dirty文件及冻结配置SHA再次一致，未改代码、输入或在线控制。

15:17:26–15:25:43：本次目标续轮为verified wait。使用monitor-experiment
按原本地tmux/进程身份进行12次只读检查，每次publisher PID248900与watchdog
PID248951的start_ticks均匹配；终态从516增至1067，已观测终态全部success
及native_contract_matched=true。sample692→1181，最后日志年龄0.91秒；
launch.json仍未产生。所查high/max/OOM/kill、foreign compute与escape均0，
无保护告警。最后主机可用86.92GiB、service峰值23.67GiB、磁盘144.47GiB，
仍高于运行底线，不在测量中压缩。原进程未重启、配置未修改、无并行实验。
原PLAN SHA保持d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83。
这只是约四分之一负载的完整性与资源观察，不是全量SLO、独立数值正确性、
GPU释放或性能改进验收。继续同一回放；所有未完成矩阵和文档范围不变。

15:27:14–15:44:32：又完成24次相隔约45秒的只读核查，原publisher/watchdog
PID和start_ticks每次一致，终态1154→2499且全部success/native合同匹配。
sample1272→2295，采样年龄最大0.99秒，最终launch.json仍未产生；没有重启
原任务或并行新实验。所查high/max/OOM/kill、foreign/escape均0、无保护告警。
这一监测段主机最低可用85.76GiB，service峰值28.87GiB，最后磁盘138.41GiB；
运行底线未触发，结束释放前不压缩。冻结配置与PLAN SHA再次核验未变。
本目标续轮是verified wait，不是新增性能结果；不提前评价部分负载的SLO、
物理GPU总量或CPU-only收益。下一动作仍是同一完整回放结束后的既定分析。

继续监测同一tmux/原进程和raw，不另起Full、不因观测等待而重启。结束后
真实资源释放→完整合同/冻结SLO/GPU-s/CPU-only复用与加载分析→保存结果→
必要压缩和备份。此交接只更新文档，测量时不提交推送或改代码/配置。

## 16:25完整结论：缓存复用增加，但系统目标明显退化

本轮最终由verified wait推进为PROGRESS：唯一4000请求回放结束、物理资源
释放、冻结指标/阶段/缓存/远端核验和13个worker的启动回执关联均已完成。
16:15:40已确认launch pass，service/watchdog/replay退出码全部0，服务资源域
消失、native GPU context释放；独立nvidia-smi无compute进程。不重跑冷/热版本。

| 指标 | D1033父点 | D1036候选 |
|---|---:|---:|
| 原生完成/失败 | 4000/0 | 4000/0 |
| 联合TTFT/TPOT达标 | 3772 (94.30%) | 3320 (83.00%) |
| TTFT达标 | 3850 | 3563 |
| TPOT达标 | 3913 | 3686 |
| 物理生命周期GPU-s | 8155.283735964913 | 8791.96267249249 |
| GPU-s/offered-request | 2.038820933991228 | 2.1979906681231225 |
| 平均TTFT/ms | 765.1991312011378 | 1190.6555377092445 |
| P95 TTFT/ms | 1956.042105332017 | 3098.928115796298 |
| 平均TPOT/ms | 15.610976187487047 | 19.36247007328033 |
| P95 TPOT/ms | 27.056324062868953 | 41.28023511212733 |
| 平均E2E/ms | 2481.9043227854418 | 3241.957105057547 |
| P95 E2E/ms | 4820.602268911898 | 6671.556550543755 |

联合少452条（−11.30百分点），GPU时间增加636.6789365275763（7.80695%），
P95 TTFT增加58.42850%。U虽低于W2预算12083.063870，联合约束未满足，
不能进入G1/G2达标排名。控制台旧5000ms的98.4%不采用；独立数值正确性
仍未闭合，n_correct=null，不将原生长度/adapter绑定检查冒充数值正确性。
这是已知seed42固定负载的单轮开发观测，不作CI、显著性或独立留出主张。

### 全部请求的阶段证据

- 同请求联合pass→fail520条、fail→pass68条；TTFT净少287、TPOT净少227。
- TTFT-only366、TPOT-only243、两者71；437条TTFT违约的最大互斥阶段为
  arrival→admission218、admission→acquired201、executable→engine11、
  engine-entry→queue7。不能把全部退化归给远端或GPU计算。
- 平均TTFT增量425.4564ms中，arrival→admission增加250.2335ms，
  admission→acquired增加62.4784ms，executable→engine增加65.2700ms，
  engine-entry→queue增加35.8155ms，engine queue增加2.9044ms，prefill增加8.7546ms。
  nested routing和runtime-slot等待是子阶段，不再重复相加。
- 最后一次native load均值39.8217→38.0604ms，仅减少1.7613ms；因此不能
  用局部加载收益推断整体SLO收益。runtime-slot等待增加173.3193ms，routing
  nested增加18.1335ms，属于条件性观测，不等于各自独立因果贡献。
- 八个500-request phase联合数452/379/411/419/421/400/411/427；后3500条
  2868达标，对比父点3307。退化并非只发生在初始启动，不能删开头请求解决。
- 时间分解、TTFT/TPOT/E2E重算误差均0；4000个dispatch来源字段完整，
  GPU2926/HOST242/NVMe344/Remote488；125服务类/75准备类与冻结画像匹配。

### 唯一缓存假设的检验

原生source：GPU2939、HOST236、file822、staging3；父点分别3005/0/991/4。
主机缓存确实产生236次CPU-only复用，110条容量在1314条请求后快照中达到，
最多78条CPU-only驻留，最大注册张量1394606080B。dispatch HOST与原生HOST
不是同一口径，不把242与236混用。文件来源减少169次不是全系统加载减少
169次的同义表述，也不能换算成假想减少的TTFT。

本轮观测非注册最大27577612B，高于父点18418980B；同一保守倒推函数会
给出109，但这是后验观测，不把配置改成109重新跑，也不能仅凭这一数值
宣称实际分配越界。真实混合工件注册量与实时字节准入仍成立，原先已明确
观测开销不是未来上界。110条候选保留为负结果，停止容量扫描。

自然scale-up12、scale-down11，父点3/2；本轮13个实际allocation全部释放。
GPU增量主要在共同arrival窗口（+638.2263 GPU-s）；pre-arrival −0.0749、
drain +0.0844、cleanup −1.5568。更多扩缩容、状态规模和等待同时出现，
尚不能仅靠相关性确定唯一根因；后续只读审计现有控制记录，不立即加入新策略。

### 启动、远端和保护证据

13个worker日志与实际lease、PID、请求owner和activation完成关联。initial
PID250500为唯一miss/created，native init74.29s、compile30.93s；12个natural
worker均有实际hit回执，producer250500，native init16.50–17.86s，全部
实际KV预算8096582616B。命中不是根据速度猜测；引擎激活37.98–50.41s是
包含其他阶段的总量，不与native init相加。原13份日志及一份不可变记录
归档14成员逐SHA验证，原文件保留；归档SHA
41559f44f0a3475f8957ad2bd3a3cca57ca3c6b84037d2afff8546f8e14b65a6。
同一initial worker的第二份保留日志已验证启动解析一致，另记路径，不当新worker。

505次真实远端传输、1172100920线上字节，双端ID/内容SHA/bytes一致，无
请求期打包。远端连续CPU/外部干扰监测缺失的既有限制继续披露，不虚构补齐。
4057个资源采样；service峰值36430598144B、host最低91591036928B、最低
磁盘144464052224B；high/max/OOM/kill/swap、foreign/escape均0。工作缓存
清空，147项旧投稿SHA及两个用户dirty SHA不变。CPU分析均在CUDA不可见、
固定辅助CPU、3/4GiB、swap0资源域内串行执行，不与GPU回放并行。

分析产物：paper_results/ieee_tc/p2_backend/d1036_3b_native_host_capacity_w2/。
保留离线接口问题：native-host CLI首次把已存相对config路径强转绝对路径，
在写输出前拒绝；随后直接调用同一既有函数、使用原相对身份完成，无改代码
或验证规则。启动关联辅助脚本首次误把物理lease owner与native adapter owner
当同一命名空间；保留attempt1快照，改用实际PID/lease和请求owner/activation
两条已有证据关联，复用已成功的13-worker审计而不覆盖。两者不是GPU运行失败。

### 决定和剩余主线

**D1036拒绝为改进，保留全部负结果。** 不推广110容量、不覆盖历史、不改
共同SLO，不重复同键，不改109/64/128碰运气。D1033本身也未达标；D1027、
D1029、D1033等历史取舍均保留，不把某一父点称为已接受最终版本。
下一步先封存/备份与恢复必要磁盘余量，然后利用已有完整控制/等待记录，
对照已关闭方向与原始来源寻找一个确有新证据的瓶颈；尚未启动新GPU候选。
全部两模型同最终版本W0/W1/W2、正确性、最终外部比较、A1–A5、S1–S13、
CE/Pareto、两轮审稿/设计文档、图表和V2最终集成仍待完成，整体目标active。

## 封存备份与测量后空间维护

结果提交4c9a3e353dd7b8b2ff90af347e8f71b25a52c158已推送并ls-remote核验
origin/retry14_continuous_queue_v2和faaslora_origin/
backup/d1036-native-host-capacity-outcome-20261103。18份指定文件，diff检查、
16项analysis bundle/10项completion SHA、JSON/脚本语法、凭据模式和staged
allowlist检查通过。所有离线分析实际完成作为本次smoke，不重跑未改的GPU
资格或74项旧生产测试。用户两处修改始终未stage；未force分歧FaaSLoRA V2，
备份不冒充最终集成。没有修改生产代码、测量配置或论文源文件。

仅在测量、全部分析及备份结束后，复用compress_closed_main_outcomes.py，
在固定辅助CPU、3/4GiB、swap0、低CPU/I/O优先级下分别压缩本次
main_outcome.json及唯一experiment_results_full*.json。每份两次独立解压
SHA与原源SHA一致，且再次对照已封存的source_refs，才移除未压缩副本；
保留原目录同名.json.gz及逐文件receipt。旧manifest的原路径/SHA不追改，
需要plain路径的旧reader须从对应gzip恢复完全相同字节，不能把路径缺失
误当证据丢失、自动更换来源或重跑实验。

回执paper_results/ieee_tc/storage/d1036_closed_outcome_20261103.json及
d1036_closed_result_20261103.json；分别实际回收538116096和348409856B，
合计886525952B。维护完成时available161075216384B，约150.013GiB，
只表示当时超过启动下限，不能保证下一重任务准入。没有维护期连续资源采样，
不虚构峰值/事件结论；保护范围、唯一工件、trace、环境与其他项目未动。
下一步按前节主线查已有控制/等待证据；所有未完成范围和active状态不变。
