# D1039：当前3B文件准备与原生HOST清点的有界诊断

**2026-11-03 追加身份更正：D1040逐请求审计确认，本轮2006个descriptor回执
均为Python路径；D1033完整父点8016个回执均为D747融合路径。下文清点CPU/
耗时数值属于本轮实际Python实现，不可作为当前D1033实现的开销或因果证据。
这是诊断启动环境遗漏，不是主系统丢失了已接受的优化；详见D1040文档。**

2026-11-03，接续D1038；完整目标仍active。**启动后发现文件路径前提不成立，
见末尾更正：本次实际仅能回答native HOST清点问题，不产生文件进程计时证据。**
这是测量接入及一次预登记诊断，
不是已经获得性能改进。EXECUTION_STATUS.md为用户dirty，继续不编辑、不暂存。

## 主线、问题和停止条件

已重读目标、完整PLAN、执行状态、冻结V1/单轮V2和D1038。上一目标轮为
PROGRESS（保存了已有证据与未决问题），本轮不重复纯文档推演。当前父配置
是D1033的32条CPU/GPU缓存、concurrency8、token4096、1–3实例，原生合同
4000完成、联合3772/4000、8155.283736 GPU-s，尚不满足3800，数值正确性
仍未闭合。D1036的110条缓存负结果保留，拒绝重新进行容量网格。

D1038的远端关联中，违约请求的文件发布包络均值461ms、发布后到native load
约412ms；它们不是纯CPU、锁或网络时间。待判定的两个问题：

1. 当前缓存填充后，文件命令的提交→执行、实际执行、执行→首次消费回执
   各占多少？重点分类publish_transfer、preparation_snapshot、acquire_best等，
   不将重叠命令包络相加作为请求延迟。
2. 当前native控制事务中，完整HOST清点占多少线程CPU/墙钟？使用原有
   fresh_inventory_v1，不以旧D863/D868配置替代当前实测。

固定复用原W2完整映射验证后的前1000条，不生成新trace/权重；分析时另列
ordinal>=500部分，但不删除前500条，不把诊断变成完整G1/G2。一个唯一执行键；
截断/缺失如实报告，不为凑样本追加相同运行。结束先确认资源释放，再按测量
决定单一优化方向；不能把诊断计时从旧主结果中减去来伪造新结果。

## 接入与测量语义

- 复用FileStateProcess已有_execute_timed和FileStateCall首次观察边界。
  默认关闭，仅command_spans_v1及既有nonformal/service/prefix门禁可启用。
- 每个父进程的文件日志上限64MiB/200000条，独占创建。只保存operation、
  owner/worker/sequence、成功标志和时钟；不保存命令参数、返回对象或权重。
- 提交→执行包含序列化、IPC、队列和调度；执行是完整owner命令；执行结束
  →首次观察包含IPC、调度及迟加入。不能分别冒称纯队列/纯IO/纯IPC。
  worker_cpu是整个进程CPU（含并行SHA线程），不与墙钟相加、不要求小于墙钟。
- 首次回执记录一次，包括失败、取消后重入；正常关闭记录提交/观察/保存数量。
  日志写入失败不得阻断已完成操作的引用/结果接管；关闭时报告诊断失败，
  不生成完整footer冒充成功。突然退出、未join和截断保持不完整身份。
- native host_spans_v1 + fresh_inventory_v1复用已有实现与分析器；不启用
  Python栈或step子组件。HOST清点嵌套于控制事务，不能再次叠加；主机span
  不是CUDA kernel时间。观察开销保留，不宣称与无观察Full的因果性能差。
- 不改IEEE公式、在线策略、远端配置或vLLM源码。源码包含已提交的D1037
  overload优先级修正，故不能把D1033旧成绩挂到本版本。

本轮使用run-experiment复用启动与资源保护，analyze-results限制解释范围，
serving-llms-vllm核对原生诊断边界，github-sync保存可回退检查点。技能中
通用吞吐/TTFT示例不替代本项目冻结阈值，也不引入调参或依赖升级。

已联网核查[Python 3.12 time文档](https://docs.python.org/3.12/library/time.html)：
本机跨进程单调时钟可用于上述差值；process CPU和thread CPU定义不同。
[vLLM 0.30.0 EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
在输入处理后执行engine step；本地已有atomic_control_yield，不能重做已关闭
的“排空全部utility导致饥饿”修复。官方结构支持测量边界，不证明本机因果。
历史D329跨调用图缓存、D628/631移锁hash、D864–866、D998等排除项见D1038，
没有本轮新证据前不重新实施。

## 唯一执行身份

配置：configs/ieee_tc/20261103_d1039_3b_file_native_prefix1000.yaml，SHA
6dc73cf4aebfd903fb409f5cc52d42c4b4eb1617f4c120f5b4fc2fdaba0355ac。
相对D1033仅三项输出/缓存路径变化；保留4000源合同，由现有prefix入口投影。

```bash
FAASLORA_D1000_RUN_ID=d1039_20261103 \
FAASLORA_D1000_RUN_TAG=d1039_3b_file_native_prefix1000 \
FAASLORA_D1000_CONFIG=configs/ieee_tc/20261103_d1039_3b_file_native_prefix1000.yaml \
FAASLORA_D1000_PREFIX_COUNT=1000 \
FAASLORA_D1000_NATIVE_PROFILE=host_spans_v1 \
bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

原始目录：/home/qhq/serverless_llm_experiment/results/ieee_tc/
p2_backend_qualification/d1039_20261103/d1039_3b_file_native_prefix1000。
tmux计划名称tc-d1039-file-native；不重用同键、不使用force。
当前为启动前登记，尚未写入GPU运行结果。

## 测前验证、资源和保护

24项不同的定向CPU测试通过（20项首轮；最终9项包含5项复查），覆盖默认关闭、
真实spawn、成功/原异常、取消后join、只写一次、日志边界、写失败后的结果接管、
HOST观察保真及配置仅路径变化。最终增加submitted_commands后的journal用例
已重跑。bash -n、git diff --check通过。测试在辅助3/4GiB、swap0、CPU2/3/26/27，
CUDA隐藏；不是GPU性能/数值正确性验收。未增加广泛资格矩阵。

独立旧投稿seal147项通过，用户manifest及EXECUTION_STATUS的SHA仍分别为
3b69bd0a6888aa56e98c3b9efb61234860d4b80f25729cfea6b79126d19127df、
41a4db5bc464dfb885b20d2b5ac321a3631a1a02291d64ac823735505957e374。

磁盘原可用161052160000B，略低150GiB。用已有压缩器无损归档D1033已关闭
request result，287843682→49664583B，两次解压SHA均为
1eeeb5f8613ab5baaf5b02df9df84de10ffb4aed61da09795656db91c346c6fd；
释放238178304B，可用161290317824B。回执：paper_results/ieee_tc/storage/
d1039_d1033_request_outcome_lossless_20261103.json。其余项目/唯一数据未删除。

推理service72/80GiB、swap2、CPU4–23/28–47，aux3/4GiB、swap0、CPU2/3/26/27。
既定启动102GiB内存、磁盘max(150GiB,100GiB+1.5×32GiB)，运行底线16/100GiB
及watchdog保持。测前GPU compute为空；启动再次由现有launcher核验。不在GPU
回放期间压缩、构建、备份、远端重启或进行额外传输压力测试。

17:22远端认证health通过，复用PID1984520、Oct31启动的既有3B服务，
prepublished_gzip_v1/artifact_timing_v2及clock ID与D1038一致。两端实际路由
NIC eno1np0/eno1均1000Mbps/full；不声称应用吞吐达到此值。最初enp*查询
无匹配，随后按ip route确定实际NIC；没有网络调整或服务重启。
共享k3s不属于项目，不修改；HydraServe隔离适配与D810基线证据继续复用。

PLAN SHA d7acd08127d4f4513b23af6dd57a9647cf047570f2b563e361e716ccb0b32b83；
V1 SHA 5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22；
V2 SHA 193bf1535af1aa50f83d98dc2e944ecbdc7bd27aa96baeca82f0654e02f9357d。
D648共同阈值仍不变。

## 完整剩余范围

两模型最终版本W0/W1/W2与G1/G2、独立数值正确性、最终外部对比、A1–A5
全部消融/confirmed-state/first-service、S1–S13、CE/Pareto/价格敏感性、
B1/B2/C1/C2/C3/新颖性与两轮审稿文档、达标后IEEE图表和最终V2集成全部保留。
此次诊断不关闭其中任何性能任务，不重新启动已完成的基线必要适配。

## 17:26启动与17:29前提更正（重要）

源码2430ac0501abe01b90dd9e2bdf75e27df2f41835已在启动前推送并ls-remote验证
origin/retry14_continuous_queue_v2、faaslora_origin/
backup/d1039-file-native-diagnostic-20261103。未强推分歧的FaaSLoRA V2。
17:26:03仅启动tmux tc-d1039-file-native一次；exec_receipt allow_exec=true。
service unit primelora-tc-svc-8724e870bb734ef4bbec5cb7ba7c0adc.scope，
InvocationID 7895512baa9c406db31fd295baf131fb；controller PID582322，
watchdog PID582261。prefix view SHA为
b6a5b999fee48149ee693e594aabd263b776414194516e634e75c7e4fe50023d；
source_count4000/count1000、W2完整map SHA与预登记一致。

**本次接入前错误地沿用了D1024/D1025/D1038关于当前已使用进程owner的叙述。**
运行中检查未生成diagnostic_file_owner后，核对了实际源码和配置：

- experiment_stack.py从preloading.file_state_process读取，默认false；
  residency_manager.py仅该值true时创建ProcessLocalSourceReferences。
- D1039（以及其D1033父YAML）未设置该选项；本次controller实际environ
  有FILE_OWNER_TIMING但没有FAASLORA_FILE_STATE_PROCESS覆盖。
- 因而本次执行的是LocalSourceReferences进程内路径，新文件journal没有
  被调用。不是零耗时、日志丢失或原生进程故障。
- 历史D897–D915确实存在显式process owner配置/环境；不能据此推定后续
  所有配置继续使用它。父点的历史实际身份仍须用其保留记录核查，不能仅凭
  当前env反推过去，也不能将组件D1025的执行时间解释为当前线上路径时间。

不运行中改变file_state_process，不把打开它伪装为单纯加计时；这会改变
执行架构和条件。保留本次预登记两个问题、其一未观测的事实，不覆盖失败
前提或为补齐数据重跑同键。继续既有native诊断：首个真实native PID583631、
start_ticks320524191的header已确认host_spans_v1/fresh_inventory_v1、服务
资源归属和共同clock。这是预登记的第二个独立问题，有有效采集理由；不是
把完整双问题诊断宣称成功。结束后先分析这一测量，再按实际进程内路径决定
是否仍需额外计时，避免再展开无依据的组件诊断或容量调参。

此次错误说明“文件进程隔离已经在当前工作点启用”这一历史推断不可靠；
后续不得继续以它排除或支持性能归因。新journal默认关闭，不影响本次进程内
路径，但其已通过单元测试不等于在本次GPU运行触发。
运行中不提交/推送、不压缩、不修改服务代码。当前结果仍in_progress，不是
完成或达标；下次从同一tmux/raw继续，勿重新启动。

## 接续核查：历史入口与现有分析器

本轮重新读取目标、完整PLAN和执行记录后，按monitor-experiment核对本地
tmux/controller/watchdog的真实存活，不把锁文件当运行证据。上一轮为
PROGRESS（测量接入/测试/备份及错误前提更正），不是已完成性能优化。

已找到更早的明确反证：D938_7B_CONTROL_PROVENANCE_AUDIT_20261030.md已经
区分D917启用文件进程/ISA-L与D930/D931未启用的环境差异；D953是7B同时
重开这两项的2204/4000中断结果，不能当单一文件进程的性能否定或成功。
3B的run_d934/run_d935入口直接沿用未启用该选项的D886 YAML。
故“实现存在/历史某轮启用”与“当前候选实际启用”必须分开，D1024/1025/1038
中泛化到当前执行的叙述已被这些原始入口反驳。没有新数据就不能把重开
文件进程视为已证明的优化，也不能因那些7B中断直接判定3B不可行。

结束后的分析复用raw d1016_20261102/analyze.py（SHA
a0743504ad6589def97eb22a04e1d175ebca3cf4c0b1fa680423d211d63a46d8），
其继续调用D723和当前analyze_core_host_timing。D1039的raw/analyze.py只
替换两个run-tag；不重建分析框架、不降低原生绑定或时钟/完整性要求。
既有native分析器已支持HOST清点嵌套区间和逐请求窗口交集。必须等实际
终态、资源释放后再运行；本轮准备脚本不表示已经产生分析结果。

## 17:47–17:52：诊断完成与定量结论

**当前已结束，不再是in_progress；不要重新启动本键。** 1000条原序请求全部
到达、提交、成功终结；launcher/service/replay/watchdog退出0，实际GPU
上下文释放、服务域移除及工作缓存为空均已核验。5个native进程的观察完整，
没有截断/孤立span；1000个首末token窗口全部覆盖。文件进程观察仍未触发，
本轮只回答预登记的native HOST问题，不宣称两个问题都已解决。

结果包：paper_results/ieee_tc/p2_backend/d1039_current_native_inventory/completion/。
manifest绑定20份小型结果/分析快照；完整大日志保留在原raw目录。

| 3B W2原序1000请求诊断 | 本轮观测 |
|---|---:|
| 原生请求/adapter/token合同通过 | 1000/1000 |
| TTFT / TPOT / 联合达标 | 983 / 936 / 924 |
| 仅TTFT / 仅TPOT / 两者违约 | 12 / 59 / 5 |
| 平均 / P95 TTFT | 605.128 / 1629.714 ms |
| 平均 / P95 TPOT | 18.072 / 39.345 ms |
| 平均 / P95 E2E | 2560.980 / 5158.558 ms |
| 完整物理GPU生命周期 | 2539.185860 GPU-s |
| 前500 / 后500联合达标 | 478 / 446 |

这些数字用于描述本诊断；不是4000请求性能点、不线性外推GPU成本、不与
D1033完整结果作胜负比较。observer开销包含在内，独立数值正确性仍未知，
n_correct=null、ranking_eligible=false；不计算CI、不更改D648冻结阈值。

### HOST清点究竟占了什么

共6942条嵌套HOST观察：3567次tensor/storage图清点，3375次allocator查询。
清点累计墙钟259956.504ms，所属线程CPU259457.379ms，CPU/墙钟99.808%。
这里是跨owner调用累计，不是服务独占墙钟，也不与GPU生命周期相加。

| 有推理工作时的清点所属控制操作 | 清点调用数 | 每次清点墙钟均值ms |
|---|---:|---:|
| pending GPU引用事务 | 437 | 77.186 |
| 按需加载并取得引用 | 322 | 74.480 |
| 完整来源快照 | 213 | 76.613 |
| 路由来源快照 | 534 | 76.629 |
| HOST准备 | 256 | 73.670 |
| 显式请求作用域快照 | 20 | 2.969 |

全部操作、idle-at-entry、registered/staged/requested作用域均保存在CSV，
上表不是所有控制工作，更不能由最后一行推断可把全局容量检查全部换成
单adapter检查；两种图具有不同预算/回收语义。

64条TPOT违约中63条输出<=32tokens，另一条36tokens。按逐请求首末token
窗口求交集，以下是各组请求均值，不是每条请求的因果归因：

| TPOT违约组 | 请求数 | decode窗口ms | client ms | engine step ms | HOST清点ms |
|---|---:|---:|---:|---:|---:|
| 全部 | 64 | 488.905 | 225.615 | 251.570 | 177.686 |
| ordinal<500 | 18 | 421.927 | 255.050 | 152.624 | 203.827 |
| ordinal>=500 | 46 | 515.114 | 214.097 | 290.289 | 167.457 |

全部组HOST清点占client交集78.756%、decode窗口36.344%；它**嵌套在**
client/input/step中，不能再加一次或从TTFT/TPOT直接扣掉。allocator查询在
这组仅0.193ms。native host span不是GPU kernel计时，也不测量GIL占用。

这补上了当前32条缓存工作点的子调用证据：重要的CPU工作确实位于新鲜HOST
图清点，而非仅把旧D868操作包络移植过来。它没有证明跨调用缓存安全，没有
证明既有C++整图方案在本配置下会改善完整SLO，也没有定位未观测的文件路径。

### 主线决策（不是新的性能接受）

关闭本次native profiling执行键，不追加相同前缀、不补跑文件观察以凑齐
原先两个问题。下一项首先用本次完整HOST调用分组核查当前图构建的重复工作
与消费者真正所需的作用域；仅在存在语义等价、可验证的新方向时形成一个
候选。D329跨调用缓存、D352事务内复用、D869暂存联合图、D180/181请求作用域、
D864–866整图实现和D998延后调用的历史均须先比对，不能重做已经完成的功能。
不因本次79%占比便重开旧候选；是否重开需要说明旧失败前提发生了什么变化。
若图路径不存在新的合法改动，再回到当前进程内文件路径，而不是假定文件
进程已经启用。下一轮不得以泛化“控制开销”为由再开没有具体问题的诊断。

### 交付、安全与分析身份

GPU释放后才读取远端日志；146次真实交付全部按transfer ID、adapter、SHA、
线上字节、服务端读取/发送关联成功，共338845661B，无请求时打包或临时归档。
复用既有远端服务，未重启/停止，没有新增远端资源监测；不声称完整排除远端
CPU外部干扰。原analysis1中的pending状态保留，delivery_closure.json给出闭环。

1195个资源样本：服务峰值22374473728B，high/max/OOM/OOM-kill/swap均0。
原生观察总量无截断；观察开销仍保留，不能把诊断视为低扰动正式性能运行。
离线分析/整理均在aux3/4GiB、swap0、CPU2/3/26/27、CUDA隐藏条件下完成。
首次分析入口因scope名称没有UUID而在写结果前拒绝；修正为既有UUID命名后
运行成功，不是模型失败或GPU重跑。随后整理峰值521027584B，相关事件/swap0。

分析复用D1016/D723；整理复用D1016成功attempt的curate.py和D863窗口交集。
D703当前SHA为3abcd9d766a7dc1c780244d8eedaac8440815aa85c0e45c6d9b81ef2cf35a070；
本次复核了所抽取的交付关联段，更新模板绑定，没有放松传输判据。新增的
inventory_groups和按前/后500分组只投影现有已核验CSV，不改变统计公式。
20项产物SHA逐项复核通过；147项旧投稿seal与两个用户dirty SHA均保持不变。

全部剩余范围仍为前述两模型同最终版本W0/W1/W2/G1/G2、数值正确性、最终
外部比较、A1–A5、S1–S13、成本/CE/Pareto、审稿设计文档、图表和V2集成。
共享k3s与既有HydraServe适配不重启；没有新baseline性能运行或论文源文件修改。

收尾时磁盘可用160952446976B，略低150GiB新重型任务启动线；GPU为空，
MemAvailable约104GiB。不得凭df的人类可读150G显示放行下一重型任务；
下次构建/回放前只对已关闭且可恢复的旧日志做既有无损归档，验证解压SHA和
引用回执，不删除唯一证据。本轮尚未对D1039原始文件压缩或删除。
