# D884：区分主动准备中改变与不改变 HOST 存储的路径

本轮只有既有数据复用和代码边界核查，没有生产修改或新 GPU 回放。D883
完整3B W2仍是3753/4000联合达标（93.825%）、11934.813584 GPU-s；未到
共同95%，独立数值正确性仍未知。所有原始失败/退化保留，不补同键重复。

## 1. 先排除已做过的方向

- D680已在GPU获取事务内交接HOST引用，不重新实现一次release RPC融合。
- D329/D747没有证明跨调用storage不可变，因此不采用epoch/model-ID/TTL图缓存。
- D864–D866整图原生化、D874–D881私有CPU构造及D882–D883小型定价融合
  没有建立所需的完整系统收益，不继续围绕这些组件微调/复测。
- D735处理controller/纯规划进程的重复验证；已登记objective在后续native
  动作中确实仍有重复处理，但本次不再开启这个小组件的性能实验。

直接复用D843已封存的三个core业务窗口采样：5910个样本中，nearest project
frame落在HOST tensor `add`的有483个，`validate_native_gpu_epoch`为40个。
这些是另一个旧版本诊断的栈出现次数，不是CPU时间比例；原采样受GIL影响，
三进程合计4个全阶段截断样本仍保留。不由40/483预测新版本收益，也不补新
profiling来强行得到归因。D863/D868的控制CPU/请求窗口证据继续作为背景。

## 2. 当前完整运行中该路径是否实际存在

复用D882权威journal流式读取与D868 HOST allocator回执分析，只读D883的
`mechanism_events._ieee_gpu_preparation_plans`，不累计shutdown复制件。
原始SHA：`be4d50d582bd4bc245e329eca56e02bb0ecf493c923ae708b5b812faad8bec3d`。

| D883实际对象 | 数量 |
|---|---:|
| GPU准备计划 / attempts | 2788 / 4010 |
| 已注册HOST→GPU成功 | 1383 |
| 其中预算前观察明确无staging、无私有构造预留 | 1303 |
| 其中预算前仍有其他staging或预留 | 80 |
| staging联合CPU/GPU提交成功 | 1462 |
| 已确认GPU复用 | 81 |
| stale_snapshot / benefit不足 / 无GPU victim | 1061 / 15 / 2 |
| 缺少回执 | 6 |

所有成功lease按owner/lease唯一校验。staging是否为空依据完整scope，而非
只看staged独占字节为零：共享storage也可能使其独占字节为零。1383个已注册
路径的HOST victim列表均为空。逐事务小表与缺失分类保存在curated目录。

**解释边界：**这1303次是预算前时点的范围观察，不是已保存的cost-provider
起始图，不能仅由它证明两次物理状态一致，更不能视为1303个请求必能提速。
没有从TTFT/TPOT扣除组件时间，没有重算或更换SLO阈值。

## 3. 与D674的关系：只考虑被明确分离的无CPU变更路径

D674的982个三清点事务中，127个在前两次改变范围/数量/字节，已经否决
无条件复用最早图。这个反证仍成立。D675只合并预算前与预算后的只读激活区间；
本次不是重复D675，也不把那些127个事务重新定义为安全。

当前源代码中的两条分支应分别处理：

| 同一次主动GPU准备 | cost-provider之后、预算前可能发生的操作 | 判断 |
|---|---|---|
| staged目标或任何HOST replacement | CPU删除、packing/注册、staging集合改变 | 必须重新完整清点 |
| 已注册目标，起点无任何staging/私有构造 | 原选择/准入只读计算、可选GPU-only victim移除、owner元数据刷新 | 允许进一步验证一个事务内复用候选；尚未实现 |

本轮同时核对本机实际vLLM0.30和对应官方源码。active cache的`pop`走
`AdapterLRUCache._on_remove`及`_deactivate_adapter`，后者清GPU slot映射；
这不同于registered cache的移除/注册。UniProc同步调用不提供多RPC原子性，
但本候选区间全部在一次worker同步调用内。以上仅支持限定分支的路径推理，
不是仅凭对象ID推断storage不可变。[官方model manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)、
[官方worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)、
[UniProc executor](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/executor/uniproc_executor.py)。

本地owner还包装了removal callback；后续资格必须检查这条实际包装链，不能
只验证上游函数名。PyTorch pinning读取底层storage属性，allocator后台状态
也不由owner epoch决定，因此每次新事务仍完整测量，预算时继续实时读取
allocator/workspace，不复用旧预算结果。[PyTorch pinning实现](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)。

## 4. 下一唯一候选及停止边界

候选是在**同一主动准备事务内、已证明无CPU存储变更的分支**，让新鲜的
replacement-cost图同时供后续预算前检查使用。不是缓存整个物理快照，也
不是把最早图无条件传给所有分支；D883当前生产源作为明确父版本，D882
等价重构仍在其中，不宣称其已经产生性能收益。

实施必须满足：

1. 起点实际无staging、无在途私有CPU构造，目标已注册；真实owner/thread、
   原生loader/manager/GPU-cache方法及removal包装链得到限定，不能由外部
   RPC布尔值授权。证明区间中没有await、CPU注册/移除/packing/转换。
2. 物理图在本次调用实时完整清点，含所有tensor/alias/pinning/隐藏字段检查；
   路径内部令牌只在这次同步调用有效，不跨请求、owner或epoch存活。
3. 保留动态source、引用、GPU/KV容量、原收益/victim与Eq.(8)判定；预算前后
   allocator和workspace继续新读。GPU-only移除不被误写为HOST释放。
4. 有staging/CPU变更的正常分支仍按现行完整观察执行；资格断言失败必须
   显式报错，不吞异常后使用旧图或添加重试。已有D329反例检测不放松。
5. 复用现有HOST事务、worker绑定与replacement fixtures，检验正常/no-victim/
   GPU-victim路径、staged排除、异常/allocator增长、真实removal callback链和
   每次调用重新测量。测试应能区分少一次完整清点与只是改回执名称。
6. 这是一个有界候选：若无法证明边界或组件调用链未减少工作，就不进入GPU
   回放。通过后一次完整3B W2新键，不加100/1000前缀、不重做广泛后端资格，
   保留D725 profile、同一配置/工件/trace/远端与冻结G1/G2。完整结果决定去留。

新前提是实际存在的1383次registered路径及明确排除CPU变更的分支边界；不是
推翻D674反例或重试D882定价微优化。本轮尚未完成新的资格证明，不能提前把
1303全部标为最终可复用次数或宣称任何性能提升。

## 5. 保存、安全与完整主线

raw：`results/ieee_tc/p2_backend_qualification/d884_20261026/`；curated：
`paper_results/ieee_tc/p2_backend/d884_registered_preparation_boundary/`。
既有stream/CSV来源分析方式复用，不新增测量框架或在线插桩。分析域使用
CPU2/3/26/27、3/4GiB、swap0，峰值53063680B，high/max/OOM/swap均0；已
inactive/dead。GPU、真实remote和共享k3s均未操作，未新增工件或负载。
可用磁盘约161GiB，未重复全盘扫描或删除其他项目；下个重型任务重新检查
150/100GiB及剩余增长量。旧147项、用户manifest和协议保护由备份检查回执确认。

本轮用vLLM技能核查原生执行边界，analyze-results区分观察与推断；不采用
通用技能阈值、不恢复CI/提前绘图。github-sync仅备份明确文件，最终V2分歧
不force push，备份不冒称最终同步。实际推送以raw/backup_completion.json为准。

整体目标继续active。两模型同版本W0/W1/W2/G1/G2、剩余baseline/dLoRA、
A1–A5/S1–S13、独立数值证明、两轮审稿图文和最终V2同步全部保留。
HydraServe已完成隔离适配，不因迟到的共享集群授权回复重新接管k3s。
