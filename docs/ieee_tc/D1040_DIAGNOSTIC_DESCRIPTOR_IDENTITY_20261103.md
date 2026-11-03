# D1040：修正诊断的实际HOST读取身份，不冒充新性能优化

**当前交接：唯一补测已结束，1000/1000成功，真实GPU释放及离线分析/交付核验通过。
冻结联合SLO为955/1000；这只是诊断前缀，不是完整G1/G2点。不要重新启动D1040。
当前融合读取下，完整HOST清点仍平均占用约50.862ms的工作中控制调用，下一步先
检查已拒绝方向和当前源码，不自动恢复整图候选、扩大缓存或再跑诊断。**

2026-11-03。当前主线仍是3B PrimeLoRA。目标与完整PLAN、冻结V1/单轮V2
已经重读；上一目标轮D1039完成诊断与备份为PROGRESS，但本轮发现其实现身份
与父点不一致，必须限制此前解释。本文件接续交接，不修改用户dirty的
EXECUTION_STATUS.md，不接管共享k3s。HydraServe隔离适配和D810基线证据复用。

## 新证据及对最初判断的纠正

按全部请求流式遍历实际native_host_descriptor回执，不仅检查shell默认值：

| 已有运行 | 请求数 | 缺回执请求 | 实际读取身份 | 回执出现数 |
|---|---:|---:|---|---:|
| D1033完整W2 | 4000 | 0 | fresh_native_tensor_descriptor_v1，D747 SHA | 8016 |
| D1039诊断 | 1000 | 0 | python_tensor_descriptor_v1 | 2006 |

D1033四个请求worker PID为4125618/4129029/4128955/4173524；各有
4039/3917/14/46个回执。D1039对应583631/585940/586060/618154，各有
1021/933/10/42个回执。它们是回执出现次数，不是独立清点调用数；仅覆盖
请求相关worker，不把无请求的额外诊断进程填入此集合。

因此，本轮最初“主系统漏掉历史已接受融合实现”的假设已被完整父日志否定。
run_d1000本身未设置该环境，并不证明D1033启动时也未传入它；D1033文档及
原有native_descriptor.json已经记录实际启用。本次错误是D1039启动命令遗漏
显式环境，不能用恢复开关重命名为新优化。D1039的Python清点99.808%线程CPU、
177.686ms等重叠结果全部保留，但不归因给当前D1033融合路径。

另一个此前已更正的事实继续有效：当前使用LocalSourceReferences进程内路径，
并未启用FileStateProcess。D1039没有文件进程计时不等于文件操作无开销。
本轮不为测量而打开另一种文件执行机制，不再启动未使用的文件观察器。

## 历史与一手源码

D747/D748的融合读取逐次观察当前Tensor/storage/pinning，只缓存函数入口；
D749完整W0曾从3727增至3787联合通过，明确接受为开发父版。该收益不直接
外推到当前W2。D864–866是不同的整图native候选，未被接受，不恢复graph。
D329跨调用缓存、D352同事务复用、D869联合图投影及D1036容量负结果保持原结论。

本轮核查官方[vLLM 0.30 worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
和[PyTorch 2.13 Memory.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)。
原生加载/替换和实际storage pinning仍要求读取当前状态，不能由逻辑ID推断
状态未变。这些源码约束观察语义，并不证明本地性能收益。

## 唯一修正与一次有界补测

只扩展现有runner的显式fresh_d747_v1选择和既有observer的D1040身份；
不修改生产调度、清点算法、九公式、原生库或任何性能参数。缺库、SHA/ABI
不符仍由已有loader报错，不静默回退。原来未指定的启动默认保持不变。

复用D747的195688B库，SHA
b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044；不重编译，
不重做9种CUDA pin状态和组件微测。修正后的问题仅是：**当前32条缓存、
真实融合读取下，完整HOST清点的墙钟/线程CPU及其decode重叠占比是多少？**
现有父运行没有fresh_inventory_v1分段，D1039则测了另一条实现，不能代替。

只跑一次源W2前1000请求的诊断修复身份；不是为寻找更好性能重复完整结果。
前500和后500全部保留，不挑快请求、不从延迟中减去观察开销、不推算G1/G2。
D1037仲裁修正已在当前源码，其完整性能尚未测量；不会给它挂上D1033成绩。

配置仅相对D1033改三个独占目录。执行命令：

```bash
FAASLORA_D1000_RUN_ID=d1040_20261103 \
FAASLORA_D1000_RUN_TAG=d1040_3b_native_descriptor_prefix1000 \
FAASLORA_D1000_CONFIG=configs/ieee_tc/20261103_d1040_3b_native_descriptor_prefix1000.yaml \
FAASLORA_D1000_HOST_DESCRIPTOR=fresh_d747_v1 \
FAASLORA_D1000_PREFIX_COUNT=1000 \
FAASLORA_D1000_NATIVE_PROFILE=host_spans_v1 \
bash scripts/ieee_tc/run_d1000_3b_route_confirmed_arrival_prefetch_w1.sh
```

启动后检查实际worker回执身份，不再只凭命令或YAML一致性宣称一致。
结束→真实释放→复用D1039/D1016/D723分析与已有核心计时器→保存来源和限制。
若清点并非主要开销，就转向既有阶段证据，不继续此局部优化；有明确瓶颈
才登记下一项可证伪候选，不自动追加prefix或容量扫描。

## 资源、封存与完整范围

先无损压缩D1029已关闭的请求JSON：288595652→49752773B，两次解压SHA
均8ca69c1d86b6843e87340f04cce3b05c2b68bf1891f0a071826b63c0492c6e7e；
释放238841856B，回执paper_results/ieee_tc/storage/d1040_d1029_request_lossless_20261103.json。
没有删除唯一权重、负载、原始内容或其他项目数据；原稿保护不变。

启动内存102GiB、磁盘150GiB并覆盖预测增长，service72/80GiB、swap2GiB、
CPU4–23/28–47；辅助3/4GiB、swap0、CPU2/3/26/27；运行底线16/100GiB。
不在测量中压缩/测试/构建/推送/重启远端。沿用3B真实远端已发布交付对象，
不请求时打包、不修改网络或工件。采用run-experiment和github-sync复用
启动/备份流程，analyze-results区分实际观察与可支持的主张；不采用通用
vLLM技能示例阈值替换冻结合同。

主点D1033仍3772/4000=94.30%、8155.283736GPU-s，不满足3800；独立数值
正确性未知，n_correct=null。冻结V1/单轮V2、D648和W2预算均不变。性能
接近目标前不画图，不重启已完成的baseline基础适配。

两模型同最终版本W0/W1/W2/G1/G2、独立数值正确性、最终外部对比、A1–A5
全部消融/confirmed-state/first-service、S1–S13、CE/Pareto/价格敏感性、
B1/B2/C1/C2/C3/新颖性及两轮审稿文档、IEEE图表、最终V2集成全部仍未闭合。
当前仅登记修复后的诊断，尚无D1040 GPU结果；备份分支不冒充已集成V2。

## 测前核验已完成

15项定向测试全部通过，包含实际shell子进程环境、默认不变/未知值拒绝、
配置仅三目录变化及既有loader/图语义检查；没有重复CUDA资格或组件微测。
现有实际库在当前Torch2.13/CPython3.12加载并返回正确SHA和descriptor模式。
最终辅助域峰值316260352B，high/max/OOM/swap均0；147项旧投稿保护及两个
用户dirty文件SHA未变。资格与两轮全部请求回执审计保存在
paper_results/ieee_tc/p2_backend/d1040_diagnostic_descriptor_identity/prelaunch/。

第一次离线封存因glob把压缩回执sidecar也当请求结果而停止；当时15项测试
已通过、GPU未启动。保留原prepare.py与测试日志，resume只修正结果文件
选择并复用已完成测试，不覆盖旧证据、不当作一次性能失败/重跑。

18:08远端认证health正常，clock ID仍为b7a04b62f38c4ae6adabb4da5c0fd8dc，
prepublished_gzip_v1/artifact_timing_v2。未重启服务或修改交付配置。

提交前仅删除新YAML末尾一个空行，测试时SHA
a5ebe34c05d880a97708205bffe7672748be72cd47574b6b8feda0befb53996c保留在原资格
记录；新文件加回一个换行字节与该SHA一致，证明无配置语义变更，不重跑测试。

## 实际启动与下一动作

源检查点7282564f9e4c310ca6a2b83866fa2607ba144a3e已推送且ls-remote一致：
origin/retry14_continuous_queue_v2及faaslora_origin/
backup/d1040-descriptor-diagnostic-20261103；分歧V2未force、未宣称集成。
实际配置SHA25b98d905efd2480dbc0f86ff8073dbdf4954bd9fea47338693a245e8798efe9。

18:10:56创建tmux tc-d1040-native-descriptor一次。exec_receipt allow_exec=true；
service scope primelora-tc-svc-dffed595e9a643a0b1528dcb002e0b4f.scope，
InvocationID 1a5defcf7c7d48aea77b6cfae684a09b；aux scope
primelora-tc-aux-03805ca1ce744248902efb8e1a45b70b.scope。
原始目录results/ieee_tc/p2_backend_qualification/d1040_20261103/
d1040_3b_native_descriptor_prefix1000（位于serverless_llm_experiment）。

18:12实际native PID691440，start_ticks320793416；/proc映射明确加载D747
原库，环境含正确library/SHA。诊断header已记录fresh_inventory_v1并实际
产生host_inventory span。模式未出现在/proc初始环境时不能杜撰：现有loader
默认descriptor且D747库只有read入口；最终仍以完整worker回执确认身份。
此时样本79、service峰值13136003072B，high/max/OOM/kill/swap均0，主机
可用100678283264B；没有资源告警/中止。仅为运行中快照，不是全程验收。

下一继续监测同一运行，不添加新的prefix或Full。完成后先释放，再复用已有
D1039/D1016/D723分析入口；全请求核查D747回执、host_inventory观察完整性、
自然扩缩容与资源/交付闭合。没有最终结果前不按局部SLO作结论。D1033仍是
既有完整父观测，D1039保留为实现不匹配的诊断；独立数值正确性和所有未完
主比较/消融/敏感性任务仍继续，不因本次补测缩减目标。

18:13–18:21本目标轮为verified wait：重复核对同一launcher/supervisor
PID689927/689938及start_ticks320790559/320790611，均live，监测样本从233
增至621，最新采样年龄0.40秒。454请求已成功终态、457已到达、0失败，
launch.json尚未生成；不是观察超时、停滞或终态，不可重新启动。
最新available92.30GiB、disk147.05GiB、service峰值16.18GiB，所查内存事件
及swap均0，无告警/中止。磁盘仅低于下一轮启动门槛，仍满足当前运行底线；
不在测量中进行清理。一次只读监测命令因括号语法错误未执行，随后重新读取
同一活进程成功；未因此重启或更改实验。

已在本轮raw目录准备analyze.py/curate.py，复用D1039/D1016/D723模板，仅
更换本轮路径和源身份，追加逐请求D747实际回执核查；**尚未执行或宣称通过**。
前者先要求launch及实际GPU释放，后者使用已有交付和重叠分析器，不新建
分析框架。结束后在既有aux3/4GiB、swap0、CPU2/3/26/27中依次运行，首次
失败保留，不能覆盖已产生的分析目录。当前无新生产改动、测试、压缩、
推送、远端管理或图表；本段运行交接尚未推送，待GPU释放后随结果备份。

## 完成结果：18:33离线核验通过

上一轮为verified wait；本轮完成同一运行及其离线分析，属于PROGRESS。
1000条均成功终态，launch/service/watchdog/replay均exit0，实际GPU上下文释放、
service资源域消失。launch SHA为
7499aeaf979659716968b984bc39039dafd9b10abebaded6c0e4346002011d97。
不把运行中超过95%的片段或最终前缀值冒充完整4000请求结果。

| 本轮诊断指标 | 观测值 |
|---|---:|
| TTFT通过 | 996/1000 |
| TPOT通过 | 959/1000 |
| 联合通过 | 955/1000（95.50%） |
| 前500 / 后500联合通过 | 488 / 467 |
| 平均 / P95 TTFT | 483.034678 / 1166.867872 ms |
| 平均 / P95 TPOT | 16.401686 / 32.296672 ms |
| 平均 / P95 E2E | 2328.129938 / 4757.417328 ms |
| 物理生命周期GPU占用 | 2596.520776 GPU-s |

全部原生token/adapter绑定及计时核查通过；三个时间指标重算误差均0。
独立数值正确性仍未由本次证明，n_correct=null，ranking_eligible=false。
本轮包含观察器开销，不据此接受新性能策略或取代D1033；也不将D1039与
D1040单轮闭环差异作为融合实现的新因果收益。

### 当前实际实现的瓶颈证据

- 全1000请求无缺失，共2004个实际回执均为D747的
  fresh_native_tensor_descriptor_v1，SHA正确、tensor_state_cached=false。
  请求相关4个worker与全部5个诊断进程分开记录；第五个扩容在收尾取消、无请求，
  不伪造其服务样本。初始activation 1个，自然扩容4个（其中1个未ready即取消）。
- 7123条分段观察完整，其中3686次HOST清点、3437次allocator统计。
  清点合计182788.424182ms墙钟、182412.671116ms线程CPU（99.794%）。
  工作中完整清点1857次，加权平均50.861702ms；另外20次请求限定清点单列，
  不能以请求局部清点替换物理总容量的语义。
- 41个TPOT违约请求的平均decode区间372.450740ms；其中控制调用交集
  160.511121ms、嵌套HOST清点114.184061ms。清点占控制交集71.138%、
  decode区间30.657%。这些区间存在共享/重叠，不能加到延迟上，也不能直接
  从延迟扣除推算“修复后”SLO。
- demand_load_and_acquire与pending引用操作占这组控制交集约35.572%和
  32.643%；routing/source snapshot随后。该结果支持继续检查实时清点的
  数据结构/调用语义，尚不证明任何具体替代方案安全或能改善完整指标。

结论：当前融合路径仍存在实测CPU密集的清点开销，但“缺失融合开关”的
解释已排除。D329缓存正确性问题、D864–866整图候选未接受、D869同事务联合
投影已实现、D1036扩大容量负结果继续有效；不能因本轮再次观测到开销，就
重复这些已关闭的候选。下一动作是对照当前源码与历史，寻找改变前提的具体
新证据；没有新证据就转向其他已测阶段，不追加同键运行或容量扫描。

### 交付、资源和来源保护

146次真实远程交付，共338845661线上字节，客户端与服务端ID/内容SHA/字节
逐项一致，无请求内打包、无服务重启。未新增远端资源监控，因此不宣称完整
排除远端CPU干扰。dispatch快照为GPU734、NVMe122、HOST1、remote143；
此条件分布不替代共同请求集合的机制对照。

1195个资源样本，服务峰值22296551424B，high/max/OOM/kill/swap均0。
分析与封存分两次串行辅助域运行（3/4GiB、swap0、CPU2/3/26/27、CUDA隐藏），
均exit0，保存的峰值分别1113567232B和617738240B，事件均0；封存快照不
虚构退出后全程峰值。147项旧投稿保护及两个用户dirty文件SHA均不变，工作
缓存已正常清空。后测可用磁盘160831639552B，略低于下一重型任务150GiB门槛；
下一构建/回放前须无损整理已关闭证据，不触碰唯一资产或旧投稿。

产物：paper_results/ieee_tc/p2_backend/d1040_diagnostic_descriptor_identity/completion/。
沿用D1039/D1016/D723及D863/D703分析入口，追加实际读取身份；没有改生产
公式/策略、冻结指标、阈值、权重、负载或论文源文件，没有新图表。完整两模型
最终比较、正确性补充、A1–A5、S1–S13、CE/Pareto及审稿文档交付仍未闭合。

## 备份与继续交接

完成检查点b8ec9ef3302297acfc5c1f0fa0d4ca1b4d84e5b8已实际推送并用ls-remote核验：
origin/retry14_continuous_queue_v2与faaslora_origin/
backup/d1040-descriptor-diagnostic-20261103均指向该提交。共23个明确路径，
21个manifest成员SHA核对通过、diff/凭据模式检查通过；两项用户dirty未stage。
实际FaaSLoRA V2仍存在分歧，未force或冒充已集成。该提交封存上述完整结果；
本备份回执段为提交后的本地交接，不自称已在该提交内。

下一主线：先依据已关闭证据作无损空间回收，恢复下一重型任务启动余量；
同时仅只读核查当前_ieee_lora_host_inventory在demand/pending事务中的完整
容量语义和已完成D352/D869复用，不能删掉新鲜检查来换低延迟。D864/D867
已读，整图候选历史典型组件收益约20.9%、完整结果仅多2个联合通过且多
625.277GPU-s，不能自动恢复。没有改变此前前提的新证据时，转向已测来源
发布/确认阶段，不追加同配置prefix或盲目调容量。本轮无新候选/新GPU任务；
独立数值正确性和最终同版本完整指标仍须后续完成。
