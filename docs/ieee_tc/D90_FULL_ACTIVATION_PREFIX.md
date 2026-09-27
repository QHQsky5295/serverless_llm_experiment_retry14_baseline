# D90：完整路径的有限联动检查

## 当前修正：分开内容身份与分配预留（2026-09-28）

原CPU反例已在修正后的同一owner路径复验；没有改权重、trace、远端服务或IEEE九式。

| 检查 | 结果 |
|---|---|
| 原 `code_lora_0039` 反例，三个临时目录 | 3/3通过，均仍观察到同样的+4096 B变化 |
| 每轮五个写入阶段 | 15/15容量检查通过；不改变原写入/fsync诊断顺序 |
| 内容/清理 | 每轮8个文件SHA相同；临时目录全部删除；147项旧结果不变 |
| 针对性回归 | 51项通过，1.522秒；首次命令的测试文件括号错误另行保留 |
| 容量/共享/替换相关回归 |最终71项通过，2.976秒；完整798项通过，48.055秒 |
| 加入draining退出修正后的最终回归 |799项通过，48.173秒；原生准备、取消、HTTP及basic smoke均包含 |
| 最终源版本的原工件反例复验 |1/1通过、五阶段无拒绝、8文件SHA相同、临时目录删除；4.582秒 |
| 真实GPU联动 | 尚未重跑，不宣称Full已经合格或性能改善 |

新容量合同保留 `st_blocks*512` 实测值，再加尚未消耗的分配预留，两者不重复计数。
对于本机plain ext4，令每文件的数据块数为 `n=ceil(size/block_size)`：最多4块时
extent可放入inode；更大文件按最多5层、每层不超过n个树节点预留保守的临时上界。
因此写入期上界为 `n*block_size + 5*n*block_size`，不是额外实际分配五倍数据，
也不是把额外预留写成真实磁盘消耗。tmpfs没有该extent树，仍按页面分配计算。
该上界刻意保守，可能暂时推迟准备；不会凭“此处通常只多4KiB”放过容量。

预留在写入之前、与其他传输/native HOST预留相同的锁内获得；随inode跨越发布
和传输退出，不能在rename后过早释放。只有写入者已关闭，且无SYNC的FIEMAP覆盖
完整普通已写extent时，才退还未用部分；仍unwritten/unknown时继续计量。
不添加生产fsync、sleep、整文件预写或失败重试。实际执行仍须满足原16GiB文件预算。
复验的临时目录容量随这个新保守上界计算，未修改生产预算。

内容身份仍由device/inode/mode/size/link/time及发布时完整SHA约束；分配块数不再
充当内容变化证据。分配变化单独检查已持有的容量上界，并更新实际footprint。
内容修改、缺失文件、非法扩长、超预留和未确认目录仍然拒绝。
替换只回收已无pending分配的副本，避免把可能仍需的元数据空间当可用容量。

依据：[extent-tree布局](https://docs.kernel.org/filesystems/ext4/ifork.html)、
[FIEMAP接口及非同步标志](https://docs.kernel.org/filesystems/fiemap.html)、
[Linux6.8的实际FIEMAP映射路径](https://github.com/torvalds/linux/blob/v6.8/fs/ext4/inode.c#L3430)。
仅适用于已核查的本机ext4/tmpfs与64位Linux接口，不宣称所有文件系统相同。
原始结果为 `allocation_bound_probe2.json`；原反例 `allocation_transition_probe1.json`
不覆盖。下一步完成相关回归与备份，再处理仍未闭合的native退出路径。

中间回归完整保留：第一次发现重新验证过的外部硬链接也被拒绝；恢复“共享inode
计一次、外部链接不可当回收空间”的既有规则。第二次并发替换暴露删除后的inode
身份记录需要在下一次create前退还；修正清理生命周期。随后针对性检查发现空
workspace清理会错误唤醒容量等待，以及合法根目录删除后不应按仍存在目录扫描。
分别保持原先无空间变化不唤醒的规则、区分受管删除后的清理视图。没有加重试、
延长测试超时或更改替换收益公式；失败日志仍为原名。

### GPU退出：可服务集合不是仍持有资源的集合

实际源码中，`_retain_runtime_request_reservation` 将未完成归还的副本标为
`draining`，这是正确的停止接单保护。问题在于最终退出遍历 `get_slots()`，该
接口只返回 `running` 副本，因此漏掉仍持有引擎的draining成员。

确定性CPU检查用真实InstancePool构造一个running和一个draining成员：原代码
只调用一次清理，断言 `1 != 2` 失败。修正新增完整成员视图，只用于最终退出；
路由、可服务数量、选卡和调度公式不变。继续调用原生shutdown、pidfd和物理归还
检查，不因状态变为draining而补记release。该CPU检查证明遗漏路径存在；没有
原运行的完整slot状态快照，仍不把它写成attempt6全部收尾问题的唯一已证原因。
取消/清理遵循 [Python官方语义](https://docs.python.org/3.12/library/asyncio-task.html#task-cancellation)。
首次组合回归有两个旧SimpleNamespace测试fixture缺少完整成员接口，补齐测试
接口后799项全部通过；没有在生产代码加“接口缺失就退回可服务集合”的兜底。

当前证据汇总为 `20260928_d90_bounded_allocation_and_cleanup.json`；最终源码
与测试日志SHA单独保存。旧三轮反例通过属于中间开发版本，最终版本另有一轮
复验，不合并成正式性能重复。下一步备份后恢复原3B有限Full联动，不重跑已完成
的工件发布、完整池下载或初始化profiling。是否真正消除Full失败由新运行决定。

## 最新结果：第六次联动定位到写入期分配增长（2026-09-28）

代码为已备份 `21f2258dcccd8b8e2c4790e8dc41e660ea4e8676`，配置、原100请求、
500逻辑adapter及D89初始化不变；本次只增加首异常证据，不声称已经修复根因。

| 检查 | 第六次观测 |
|---|---|
| 计划 /提交 /成功 /取消 /未提交 |100 /61 /47 /14 /39；成功行原生长度合同通过 |
| initial /controlled |均实际就绪；另两个 natural activation 取消 |
| 驻留周期 |60完成、13过期、2失败 |
| 首异常 |同一私有 staging 文件的实际占用由37,982,208增至37,986,304 B |
| 同时不变的字段 |device、inode、逻辑长度37,980,895 B、link count=1 |
| 关联工件 /传输 |`code_lora_0039` /`72c41c58f17e4eb488c13c29a56ceb11` |
| HTTP关联 |22个UUID全部匹配；51,098,232 B线上 /831,603,868 B已验证内容 |
| 发布 /未发布 /请求中打包 |21 /1 /0 |
| 内存 |340次采样；峰值19,138,220,032 B；主机最低97,015,042,048 B |
| high /max /OOM /swap |均0 |
| 收尾 |工作目录已由原运行删除；外置确认GPU/服务全部退出；远端和空辅助组已停止 |
| 进程内租约 |2已闭合、2未确认；不补造release时间；外置60秒后收尾仍属失败 |

错误发生在预分配后、解包写入中、内容发布前。此次明确证明“实际占用完全不变”
不成立，但尚未记录该文件当时的extent树，不能直接断言是哪种文件系统事件，
也不能据此把attempt5不同的错误归为同一原因。SHA已核验的是压缩归档；这次未
完成的payload不能写成已验证权重。后续应检验Linux ext4已分配但未写入extent的
转换是否会增加元数据块，同时保留来源验证、真实容量预算和并发预留。

准备过程中系统Python执行health命令因缺numpy失败，空结果保留。发现后使用
既有conda执行相同命令，两服务均通过；时间在启动后、runtime-ready/业务前。
不将其称为启动前健康检查；不重启服务，也不把本次作为正式性能资格。

汇总与完整首异常：`paper_results/ieee_tc/p2_backend/20260928_d90_3b_full_prefix_attempt6.json`。
147项旧结果保护通过。下一步仅做有界CPU文件诊断，不盲目启动第七次GPU回放。
Full、M1/M2、正式A/S均未合格/未开始；baseline继续暂停。

### CPU反例：合法写入可改变已预分配文件的块占用

复用D88实际owner诊断工具，在同一ext4文件系统的临时目录中，读取并复制既有
`code_lora_0039`内容。分别写入4、12、20 MiB位置的原4 KiB字节，并仅对诊断
文件fsync以观察已写/未写extent；最后写完整个原工件并核验8个文件SHA。
这些写入顺序/同步屏障不是原请求解包顺序，也不是生产优化方案。

| 观察阶段 | extent记录数 | 目标文件额外分配 | 原owner检查 |
|---|---:|---:|---|
| 预分配后 |1 |0 B |通过 |
| 第一个局部写入 |3 |0 B |通过 |
| 第二个局部写入 |5 |4,096 B |拒绝 |
| 第三个局部写入 |7 |4,096 B |拒绝 |
| 全部原内容写入并同步后 |1 |0 B |通过，8文件SHA相等 |

三个独立临时目录均得到相同反例；目录全部回收，147项保护内容不变。这里
extent数按filefrag逐条映射计数，不用其可能合并物理连续区间的末行摘要。
诊断12.099秒完成，3/4 GiB内存限制、swap0；没有GPU或远端操作。

[内核文档](https://docs.kernel.org/filesystems/ext4/ifork.html)说明inode内可容纳
前四个extent，更多映射需要树节点；[Linux6.8实现](https://raw.githubusercontent.com/torvalds/linux/v6.8/fs/ext4/extents.c)
中的`ext4_split_extent_at`和`ext4_ext_grow_indepth`明确包含写入转换、树扩展和
元数据块分配。[posix_fallocate合同](https://man7.org/linux/man-pages/man3/posix_fallocate.3.html)
并不承诺之后`st_blocks`完全不变。源码与本机反例支持容量不变量过强；它们不
反向补造真实失败瞬间的extent树，也不证明attempt5属于同一原因。

下一修正的约束：区分不可变内容身份、固定逻辑长度和可变文件系统分配；在写入
许可前保守预留、持续记账真实占用，并在终态核验。不能加“允许4 KiB差异”的
经验容差、忽略真实超预算、把文件存在当验证完成，或新增全文件预写/fsync来
碰巧消除现象。并发、取消、替换和native租约收尾均须先有确定性CPU覆盖。
当前生产代码未改，未启动第七次GPU回放；详细证据为
`paper_results/ieee_tc/p2_backend/20260928_d90_extent_transition_diagnostic.json`。

本阶段只推进 Prime IEEE Full。复用 D89 的实测初始化、D88 原始父级配置、
现有 500-adapter 清单及原 seed42 trace 前 100 请求，不生成新权重或负载。
D78/D80/D81 和 D88 测量均已完成，不重新执行。基线继续暂存。

## 检查问题与证据边界

执行顺序是实际初始启动、一次明确标为 controlled 的附加副本启动、原 100
请求的既有连续到达回放、实际引用/准备任务收尾与物理 GPU 归还。
直接复用 `ScenarioRunner`、`ExperimentStack`、真实子进程和原生 vLLM。
保留主实验的 Full 资格拦截，不添加“跳过资格”配置，不修改 IEEE 九式。

| 内容 | 本次检查 | 不能据此宣称 |
|---|---|---|
| 启动与扩容 | initial 与 controlled 两种实际所有权路径 | 自然扩容次数、A4 因果收益 |
| 数据 | 原始前 100 请求、原生 token、真实远程 | 全 4,000 请求正确完成、正式 SLO |
| 到达 | 沿用内部 open-loop，两个实例就绪后开始 | 共同 -60 秒部署通知下的性能 |
| 准备 | 实际需求、成本与所有者产生计划 | 零需求时空计划也算机制触发 |
| 成本 | 所有实际子进程的物理 acquire/release 保留 | 预启动开销免费或正式 G1/G2 排名 |
| 输出 | 原生计数与来源证据 | 原零权重池已具备数值辨别能力 |

每个输出必须保留请求 ID、adapter、目标/实际原生 token、实际 dispatch tier、
原生时间和引用证据。错误、取消、未触发与资源释放失败保留；不筛选成功请求。
本次成功也不自动解除 Full 正式运行拦截；外部回放、共同准备时钟及完整回放
仍须后续联合验证。

## 启动路径复用与准备入口一致性

主脚本原有局部工厂抽为 `_spawn_dedicated_scenario_engine`，主回放和诊断调用
同一函数。保留父配置、设备解析、原生子进程、协调器语义。若协调器构造失败，
已经创建的模型仍须被原所有者关闭；不能丢失物理租约。

接线检查发现两个旧准备入口仍可由请求回调触发：
`ExperimentStack.record_access` 的 NVMe-hit HOST promotion，以及请求引用释放
后的 opportunistic GPU forwarding。它们绕过 IEEE 收益目标/共同 movement
owner，不能和新路径同时作为 Full 的准备策略。因此 IEEE policy 下不再调用
这两个旧策略，仍保留新 handoff/residency、请求驱动加载和真实缓存复用。
旧 policy 的行为保留；这不是关闭论文的 HOST/GPU 机制或按结果删减开销。

历史参照为 `4af6a9c`（activation）和 `107724e`（GPU replacement）及现有
P1 公式合同。此次修正的可证伪检查是：同一个 NVMe-hit 回调，在旧策略下仍会
调度旧 promotion，在 IEEE 下不得产生该任务；IEEE 请求释放不能启动旧 GPU
准备。尚无真实性能增益结论。

取消与清理依据 [Python 3.12 官方任务/取消语义](https://docs.python.org/3.12/library/asyncio-task.html#shielding-from-cancellation)；
后端仍为已验证的 [vLLM 0.30 AsyncLLM](https://docs.vllm.ai/en/v0.30.0/api/vllm/v1/engine/async_llm/)。
这些参考只支持生命周期实现，不替本项目证明系统收益。

## 当前状态

新增最初 7 项路径测试通过；加入两项旧准备入口隔离测试后的最终回归
713 项通过（43.170 秒），包括请求生命周期、共享传输压力及基本检查。
第一轮回归命令误写两个不存在的测试模块，保留错误日志；第二轮正确模块
451 项通过（35.374 秒）。第三/四轮发现两个旧策略测试通过 `__new__`
绕过构造、遗漏策略身份；补充 fixture 的显式旧策略身份，未放松生产检查。
全部中间失败日志保留，最终依据为 `regression5.log`。
147 项旧投稿保护内容 SHA 无变化。当前没有真实性能增益结论。
原始目录为 `results/ieee_tc/p2_backend_qualification/d90_20260927/`。
源代码 `2f26d3233ed9d5723786d47d34830db4d441dd8a` 已推送并核对远端。

## 3B 第一次真实联动：受控扩容失败，已清理

| 检查 | 观测 |
|---|---|
| 初始副本 | 实际就绪，activation 46,791.79 ms |
| 第二次启动 | controlled；物理分配检查拒绝 |
| 计划 / 已执行推理请求 |100 /0 |
| 真实制品获取 |0；空需求初始计划不算准备收益 |
| 错误 |`physical GPU still has compute/owned/unknown contexts` |
| 本服务峰值内存 |5,673,623,552 B；71 次采样 |
| 主机最低可用内存 |110,412,017,664 B |
| high / max / OOM / swap |0 /0 /0 /0 |
| 实际模型租约 |`1a7fcfcfcd434f89bbb030b2daac0327` 已释放 |
| 最终状态 |GPU compute 清空，服务域消失，工作缓存删除，池为0 |
| 远端与外置组 |核对 invocation 后停止，无后台模型 |

监测记录显示控制进程 PID 3240430 在采样16时已经分别持有 GPU1/2/3 上的
268,435,456 B CUDA 上下文；此时模型工作进程尚未在 GPU0 上出现，后者首次
出现在采样40。第二次受控启动因此不能把 GPU1 视为无上下文设备。不能把
控制进程占卡从计量中删除，或豁免物理分配检查来绕过。

源码存在明确的不当观测路径：`GPUMemoryMonitor.get_current_memory_info`
先进入 `torch.cuda.device` 读取当前进程 allocator，再读取 NVML。
当前进程的 allocator 不是子进程模型的 KV/LoRA 状态，而且控制面观察不应
创建 CUDA 上下文。下一步以无 GPU 依赖替身测试隔离此路径，再将控制面
设备观测与工作进程 allocator 观测分开；不能把未知的 worker allocator
值填成零。实际 native admission 仍使用工作进程原生状态。该因果修正尚未
实施或通过反事实重测，不声称问题已经解决。

失败的 controlled activation 在进程内保留了2 GiB未确认 HOST reservation，
没有虚构成功释放；真实服务进程结束后，cgroup/GPU外置核验均清空。
所有失败原始数据保留。这不是可用 TTFT/GPU-s 性能点；不启动7B重跑同问题。
汇总：`paper_results/ieee_tc/p2_backend/20260927_d90_3b_full_prefix_attempt1.json`。

## 控制面被动观测修正（尚待联动复测）

失败证据已备份 `75a72bd68a3a9bce179d66da39b110176393f7aa`，远端 SHA 一致。
旧监控从初始同步 `b68eaeb` 就包含当前进程 CUDA allocator 采样；它在多子进程
执行中不具备所需的 worker 观测语义。IEEE policy 现在明确使用 `nvml_device`，
按配置的物理 GPU 索引读取设备总量/已用/剩余；构造及采样均不调用 Torch。
reserved/active/cached allocator 值记为未知（null），不填零，不传给旧估算器。
原生 worker 的 KV、LoRA、引用、有效容量和物理保护路径均不变；旧策略默认
仍为原 process-allocator 观测。无 NVML 或设备读取失败不退回 CUDA。

依据：[PyTorch 2.13 CUDA 源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/torch/cuda/__init__.py)、
[分配器源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/torch/cuda/memory.py)、
[NVIDIA NVML 设备查询](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html)。
前者涉及本进程 CUDA/allocator，后者提供设备级内存；两者不能互换为 worker KV。

| 检查 | 结果与范围 |
|---|---|
| 修正前四项替身测试 |3 failure /1 error，证明旧实现不满足被动观测合同；日志保留 |
| worker/launch/basic 回归 |404 项通过，30.271 秒 |
| 原生已安装环境、真实四卡 NVML 读取 |通过；进程3332000前后 `torch.cuda.is_initialized=False` |
| 正式 Full/性能 |仍未合格；不因 CPU 测试解除拦截 |

首次测试另有一个测试导入路径笔误，已纠正为 registry.schema；没有生产兼容
补丁。原生检查没有加载模型、没有远程传输，不是扩容反事实结果。
下一次复用原 driver、D89 profile、D88 配置和原100请求，新结果键 attempt2，
不重跑准备成本测量。只有真实联动及外置 GPU 所有权证据才能验证失败是否消除。

## 3B 第二次真实联动：启动通过，驻留报文失败

| 检查 | 观测 |
|---|---|
| initial / controlled |两者实际就绪；46.813 /48.296 秒 |
| 控制进程额外 GPU 上下文 |127 次资源采样均未观察到；仅模型工作进程持卡 |
| 业务 |进入首请求；完整可验证请求结果0/100，不作性能点 |
| 首次驻留计划 |两个副本均失败，内部 RPC 的逐行读取超过上限 |
| 真实 Remote 获取 |1 次，2,325,514 B 线上，42,695,980 B 原内容，UUID准确关联 |
| 打包 / 临时归档 |0 /0；请求采用既有发布缓存 |
| 内存峰值 /主机最低可用 |8,946,106,368 /107,181,121,536 B |
| high /max /OOM /swap |均为0 |
| 资源释放 |两张实际 GPU 租约均释放，服务域消失，远端/外置组均停止 |
| 文件收尾 |进程内保守拒绝未闭合引用；进程消失且无打开 fd 后回收本轮唯一工作副本 |

该反事实联动支持“控制面 CUDA 观测阻止第二副本启动”的归因，不能推广为
Full 已经合格。原生 KV/LoRA 控制没有替换为 NVML，九式和物理护栏未改。
新的根错误位于 `register_preparation_plan` 接收：`Separator is not found,
and chunk exceed the limit`。现有 dedicated worker 使用 `asyncio.start_server`
默认64 KiB reader 和 `readline`，父端发送却没有对应上限。官方
[Python 3.12 streams 文档](https://docs.python.org/3.12/library/asyncio-stream.html)
说明该默认限制；这是内部消息协议不一致，不能归因为显存不够或网络慢。

下一步保留完整候选和目标，明确已有 newline-JSON 协议的双向有界帧合同，
以实际 worker/proxy 的无 GPU loopback 测试先覆盖大计划、连续帧、超限及取消。
不对本次失败自动重试，不通过缩小500池、删计划字段或改变优化目标来规避。
错误读取远端 journal 的一次空文件保留；随后按实际 journal 身份获取完整记录，
没有修改原始记录。内容、清理及127次监测见
`paper_results/ieee_tc/p2_backend/20260927_d90_3b_full_prefix_attempt2.json`。

### 报文修正的验证与下一执行键

复用现有 newline-JSON 通道，不更换后端或另建通信框架。双方以共享的
8 MiB **单帧编码字节上限**接收/发送；它是内存保护的协议上限，不是模型
优化参数，不按测试点调大。父端检查超限回复，worker 的 reader 使用同一
上限；不能只单边加大读取缓冲。保留原有 native unknown-response 不重试、
取消后的所有权对账，以及 first/last token 的独立进度帧。每次回复记录
请求/响应线上字节，便于后续检查距离上限和观测开销。

使用真实 `_run_worker` 和 `SubprocessInferenceEngineProxy` loopback：500候选
测试报文超过1 MiB，修正前复现相同 `Separator` 错误；修正后完整字段往返
相等，随后下一条消息和 token 进度仍正确。另测编码字节边界、非ASCII膨胀、
有/无换行超限、分片及合并消息。该数据只是通信测试对象，不新增模型工件
或实验 workload，不作为准备算法/性能证据。

最终 worker、native retirement、request lifecycle、launch、basic 共571项通过
（31.604秒）；前一轮571项也通过（32.403秒），随后仅补充两项报文字节观测。
147项旧投稿保护清单无变化。所有原始失败、测试和清理日志保留。
新 `run_3b_full_prefix_attempt3.sh` 与远端 attempt3 monitor 启动文件已准备，
**尚未运行**。仍复用同一原100请求、原500池、D88配置和D89初始化测量，
下一步按原保护门槛做真实联动，不先跑7B或恢复baseline。Full正式拦截保留。

## 3B 第三次真实联动：并发业务中的计划登记拒绝

| 检查 | 观测 |
|---|---|
| 执行代码 |`f6ef5d6d0a440290117e52b02a3a750a403145bc`，已推送版本 |
| initial /controlled |两者就绪；业务开始后38个驻留周期完成，1个失败 |
| 请求证据 |最后 live 计数 arrived9/done5/ok5；中断未返回逐请求表，不能当完整性能结果 |
| 失败位置 |native `register_preparation_plan` 的完整当前来源检查 |
| 错误细分 |旧错误合并 epoch、slots、覆盖集合、来源身份；无法从该文本断定哪一项失败 |
| 真实获取 |5个 UUID 精确关联；11,609,599 B 线上 /200,896,988 B 内容；无打包 |
| 内存 |168次采样，峰值8,954,916,864 B；最低主机可用107,262,668,800 B |
| high /max /OOM /swap |全部0 |
| 收尾 |两物理租约释放；HOST reservation为0；工作缓存删除；远端/外置组停止 |

与 attempt2 不同，大计划已经能登记并执行；但这不等于完整 Full 已合格。
下一问题是并发请求改变后端状态时，过期规划如何得到确定的拒绝与终止。
必须区分来源身份错误、未知通信结果与明确未登记的过期快照，不能统一重试
或吞异常。先用真实 native owner 的无GPU缓存替身构造可控交错，再决定实现。
原拒绝保护和失败证据保留。38个周期完成不意味着38个非空机制收益样本。

`requests=[]` 是异常导致诊断器没有取得 runner 返回值，不表示未执行请求。
本轮保留 live 计数及其不足，不据此补造逐请求 token、TTFT 或正确率。
汇总：`paper_results/ieee_tc/p2_backend/20260927_d90_3b_full_prefix_attempt3.json`。

### 过期计划的显式拒绝合同（CPU验证完成，真实复测待做）

历史实现 `107724e` 已保护冻结目标，但把规划后正常业务产生的版本变化也
抛成不可恢复的 RPC 错误。实际 `_refresh` 的版本包含 slot、CPU集合和引用
pin状态；取得再释放同一引用，即使最终slot不变，版本也已经推进。现有
`MixedOwnedPreparation` 原测试实际上只断言这种交错会报错，没有验证在线
控制器能够在下一正常周期继续工作。

本次用真实 native owner、真实 mixed selector/执行器、实际引用操作及缓存
替身复现：修正前3项测试中2项错误；修正后扩展到全部相关回归788项通过
（41.934秒，独立6/8 GiB受限CPU测试服务已结束）。未加载模型或重新生成输入。
这证明可控交错的合同已修正，**不证明旧真实失败的合并条件一定只有epoch**。

实现只接受来源所有者明确返回的 `registered=false`、相同plan/owner/clock、
相同expected版本和严格推进的current版本，作为 `superseded`。旧目标不登记、
不提交GPU准备；执行器先join/close本轮实际所有权，然后该周期结束。下一
正常控制周期重新观察并重新求解，保留原公式、需求窗口、收益和控制cadence。
不覆盖旧目标，不循环重发，不把部分完成工作伪称零成本，不阻塞普通请求。
未知通信结果、错误owner/plan/版本、未知拒绝原因及关闭失败仍终止；取消仍
传播。handoff/controlled诊断不静默替换预定计划。

并发原则参考 [etcd原子条件事务](https://etcd.io/docs/v3.6/learning/api/#transaction)：
版本条件失败与未收到确定结果是不同状态；这里只借鉴该原则，没有引入etcd。
同时核对 [vLLM 0.30 LoRA worker](https://docs.vllm.ai/en/v0.30.0/api/vllm/lora/worker_manager/)：
实际adapter变更在单线程engine loop串行化。客户端先观察、后发登记请求，
二者之间仍可插入业务事件，不能把两次RPC当原子事务。此为实现解释，不是
新的算法贡献，也不提供性能领先证据。

为后续定位，诊断结果保留现有GPU准备计划账本及具体登记回复；同版本的
slot/覆盖/来源异常现在输出各项谓词，不将它们归并成正常过期。
下一步先保存该检查点，再以原D88配置/D89初始化和原100请求单独复测。
此前失败的逐请求返回缺失也须在下次运行前补齐失败现场的留存；不能从live
聚合值反推已丢失的逐请求测量。Full正式门槛仍未开放，baseline继续暂停。

### 中断回放证据留存（2026-09-28）

过期登记修正已备份 `c44167157a3634e6e420a26e66499a7e9f42b7dc` 并核对远端。
随后仅补计量：原连续回放器发生控制错误或整体取消时，先按原规则结束已
启动任务，再保留此前已收集结果与真实已启动任务的终态。未提交请求只列ID，
不补造timeout；无法恢复的结果单列collection error，原控制异常继续抛出。
诊断器在 `interrupted_replays` 中单独保存，不填入完整成功请求表，不改变
`pass=false`、分母或性能资格。正常成功路径不额外序列化逐请求结果。

修正前两项留存测试均错误；修改后的request/launch/basic共519项通过
（38.600秒）。覆盖全局取消、缺失后续到达、控制异常和诊断器不冒充成功。
原attempt3不能追补未保留的请求级证据。新attempt4脚本仅准备、语法检查通过，
尚未启动；仍为同一原100请求、500池和冻结开发配置。所有模型与远端服务停止。
## 2026-09-28：attempt4 并发回放失败，保留全部已提交请求

执行代码 `db317368466691160c245e5dd7ae5e2650d614a5`；仍为原始 100 请求前缀、
500 adapter 池及 D88/D89 冻结配置，不是正式 Full 性能比较。

| 检查 | 实测结果 | 解释 |
|---|---:|---|
| 计划 / 已提交 / 未提交 | 100 / 45 / 55 | 未提交不伪造 timeout |
| 成功 / 异常 / 中断取消 | 30 / 7 / 8 | 45 条均保留，无 collection error |
| ready activation | 2 | initial 和 controlled；另 2 个自然扩容被中断取消 |
| residency epoch | completed 56；superseded 5；cancelled 1；failed 1 | 不是 56 次独立实验 |
| 本机服务内存峰值 | 18,941,923,328 B | high/max/OOM/oom_kill/swap 均为 0 |
| 最低主机可用内存 | 97,875,075,072 B | 无保护性资源中止 |
| HTTP UUID 对齐 | 20 / 20 | 无请求打包；线上 46,437,100 B |
| 本地已确认发布 / 未发布 | 19 / 1 | 未发布项不得视为 cache hit；已核验逻辑字节 756,697,668 B |
| 进程内 physical lease 关闭 | 3 / 4 | 1 个未闭合事实保留，不补造 release 时间 |
| 外层退出后的实际 GPU / 服务进程 | 均已释放 | 与进程内账本关闭不同，不混为一项 |

七个请求异常分别是准备类别与 admitted source 不一致 5 次、远端准备区间缺少
相符的传输身份 1 次、缺失时间值参与减法 1 次；尚未单独确证各自原因。
这些错误不能作为“推理后端慢”或“层级机制无收益”的证据。

最终回放中断栈明确显示：native 注册返回合法 superseded，随后 file plan
关闭被仍存在的 physical materialization 拒绝。既有收尾只 join 自己记录的
intent；是否存在相同目标但未订阅的并发写入，需要用真实 owner/queue CPU
fixture 单独复现。保留资源/引用保护，不靠吞掉异常或取消所有请求规避。

本轮 local auxiliary scope `0a10c9f5fc8d46ebb6cf143487cafee1` 在实际
`cgroup.procs` 为空且 `populated=0` 后停止；远端 monitor
`0343b97b43e94671a42b10014d3eeb4f` 按身份停止，两个 artifact 服务及 monitor
最终均 inactive/MainPID=0/Result=success。没有运行中的模型或远端服务。

本轮 NVMe 临时缓存暂存：部分同 UID 进程的 `/proc` 检查被拒绝，未把不完整
引用检查当作可删除证明；空间充足，不删除原始证据或唯一工件。HOST 临时目录
已消失。原始收尾失败、保守未闭合 HOST 预留和 outer cleanup 回执均分别保留。
147 个历史保护项无变化。

来源：`paper_results/ieee_tc/p2_backend/20260928_d90_3b_full_prefix_attempt4.json`。
后续先 CPU 复现并发收尾与请求观测错误；不启动 7B、baseline 或正式矩阵。

## attempt4 后的三项最小并发修正（真实性能待重验）

| 已复现的问题 | 修正边界 | 不能据此声称 |
|---|---|---|
| superseded plan 未订阅同目标的另一实际写入，关闭失败 | 原 queue 按 owner/tier/adapter/content 等待已有执行；不新增订阅，不唤醒 deferred，不取消其他请求；原 owner 原子关闭检查保留 | 全系统资源收尾已经通过 |
| pending RPC 让出执行后读共享 `last_ieee_decision` | 选中时保存现有不可变 decision，后续始终使用本请求的 service class | confirmed routing 的性能收益已量化 |
| Remote 共享下载留下空的本请求 transfer dict | 只有实际拥有下载记录才登记本请求完整准备区间；共享等待仍计入 service D，物理传输日志仍保留 | 共享请求等待免费，或其 d=0 |

第一项在真实 file owner、queue、下载 fixture 与 native owner 的组合测试中复现；
包含重复取消，确保取消的计划不杀死仍有需求的共享下载，pending 保护到物理结束。
新 join 与关闭放在同一收尾 coroutine 内，最终无 await 缝隙；不取消引用检查。

第二项的控制性 interleaving 在旧实现上分别产生与真实日志相同的
`preparation class/profile differs...` 和 `NoneType - float`。其中 GPU 命中原本
由 GPU service class 初始化 D=0；错拿 Remote class 才没有 acquired timestamp。
因此不能用补一个默认时间值修理。这一修正保留 service class 与选中源的一致性。

第三项的真实 attempt4 queue 记录显示 `req_00007/translate_lora` 在 residency
写入开始后加入同一 job；它并非下载 creator。CPU 相同共享路径复现原始 identity
错误。修正后继续以 `shared_file_preparation_reused` 标注为不适合完整 d profile
更新，仍保留完整用户时间线；未把另一个任务的加载样本记到本请求。

参考边界：对照 [Python 3.12 任务、取消及 shield 语义](https://docs.python.org/3.12/library/asyncio-task.html)，
共享物理任务有独立寿命，取消一个等待者不代表操作已经终止。
[vLLM 0.30 LoRA worker manager](https://docs.vllm.ai/en/v0.30.0/api/vllm/lora/worker_manager/)
按实际 adapter 请求执行缓存加载/复用；本次不修改该原生缓存策略。
另核查 [vLLM 0.30 forward context 源码](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/forward_context.py)：
其 forward 范围的保存/恢复不能直接当作并发异步 router 的跨 await 请求状态，
所以这里复用不可变局部 decision，而不是新增全局状态或锁住全部请求。
这些文档只帮助限定正确的实现边界，不替本项目证明收益。

首次两个 closure 测试已有旧行为失败；第一轮 green fixture 误复用了上游不同
内容的 archive，第二轮计数误含 fixture 的 4 次 setup fetch。均只修正 fixture，
不松动正式内容 SHA 或下载计数校验。失败日志保留在 D90 raw 目录。
所有九式、D88/D89 配置/profile、原始 workload、工件及正式 Full 拦截均不变。

最终相关回归与基本检查 **757 项通过，39.889 秒**；完整日志为
`results/ieee_tc/p2_backend_qualification/d90_20260927/concurrent_regression1.log`。
当前只完成 CPU 正确性修正，下一次实际 3B 回放尚未启动。

## attempt5：完成更多实际请求，但文件发布边界检查失败

执行版本 `72971413137e192fa297b3a6b2d413cd989097a1`，原driver、100请求
前缀、500工件、D88配置和D89初始化不变。上述“尚未启动”是历史状态。

| 项目 | 本轮观测 |
|---|---:|
| 计划 / 已提交 / 尚未提交 | 100 / 74 / 26 |
| 成功 / 异常请求 / 中止取消 | 56 / 0 / 18 |
| 初始 / 受控 / 自然扩容实际ready | 1 / 1 / 2 |
| 驻留epoch completed / superseded / cancelled / failed | 61 / 16 / 1 / 3 |
| HTTP UUID准确关联 | 23/23 |
| 线上字节 / 内容核验后的逻辑字节 | 53,437,743 / 887,931,336 |
| 发布成功 / 未发布 | 22 / 1 |
| 每请求打包 / 临时归档 | 0 / 0 |
| 内存峰值 / 最低主机可用内存 | 19,172,200,448 / 97,284,222,976 B |
| high / max / OOM / OOM-kill / 实验swap | 全0 |

真实堆栈从 `preparation_snapshot → source_snapshot` 报告
`local copy exists without verified source publication`。这证明文件准备状态
检查拒绝了本次运行；尚不能仅凭这个组合错误确定具体adapter、rename窗口或
是哪一次失效。后续应在CPU控制性并发中定位物理文件出现与已核验状态发布的
原子边界，而不是把未验证目录当命中、放松检查或再盲跑。

56个成功请求的固定输出合同均通过；18个取消和26个未提交原样保留，不能称
56/56完整合格。先前三类请求错误本轮未复现，但不足以证明所有并发均正确。
两次自然扩容真正ready，是实际路径进展，不是A4样本门槛或性能贡献结论。

收尾仍有一个进程内GPU lease未闭合、2GiB native HOST预留保守保留；外置
收尾发现最后副本未自行退出，按既定60秒界限和明确身份释放本实验剩余进程。
最终所有GPU上下文和服务资源域已消失，不给原始未闭合lease补造释放时间。
service/watchdog为2/0。空aux `85cdb2b2e16c4184a602227a676f8ebf` 已停止；
推理全部结束后才停止匹配远端monitor `4587276762c849e9bb61fe4ccdf880bb`，
三个远端单元最终均inactive/MainPID0/success。

HOST临时目录消失；该attempt的NVMe缓存531,046,400B/112文件暂保留，因为
同UID的三个无关进程无法完整读取映射。未凭不完整占用审计删除；空间仍充足。
147个历史保护项均无变化。立即使用本状态表，不画虚假的完整性能图。
来源：`paper_results/ieee_tc/p2_backend/20260928_d90_3b_full_prefix_attempt5.json`。
下一步是文件发布边界与失败后的native lease收尾；不启动7B、baseline或主矩阵。

## attempt5 后的有界 CPU 排查：尚未证明根因

2026-09-28：用真实 `LocalSourceReferences`、原 runner 下载和发布入口，在
初次发布及替换时强制暂停于 rename 已完成、confirmed registry 尚未提交的
位置。读线程不能进入同一 owner 的观察区，恢复后只读到核验完成的副本。
另在真实预分配完成、正文写入前暂停，目标目录仍不存在，未把私有 workspace
当作已发布源。这些测试支持现有锁的原子边界，**不支持**简单的
“rename 与 registry 之间漏锁”假设；不能宣称已修复原 GPU 回放失败。

| 证据 | 结果 | 限制 |
|---|---|---|
| 发布窗口并发测试 |初次发布/替换均隔离中间态 |小型 CPU fixture，不是 GPU 回放 |
| 下载中预分配测试 |私有文件不成为 confirmed source |不覆盖所有真实文件系统时序 |
| attempt5 保留的14份 NVMe副本 |逻辑/分配字节及文件数与发布回执一致 |不是重新内容哈希，不能排除瞬时变化或已清理HOST副本 |
| 相关回归和基本检查 |759项通过，43.685秒 |Full仍不合格 |

原失败只保留每个驻留任务的异常类型；首个任务被取出后，收尾可能呈现另一
任务的后续 `unverified` 错误。因此本次仅补充错误分支的可观测性：每个失败
epoch保留原错误和堆栈；源变化记录首次不一致的路径、签名和footprint；未知
现存目录记录工件、tier及活动transfer。失败仍原样抛出，不重试、不重新扫描
覆盖首次观测、不把未知降级成Remote miss，也不增加正常请求的哈希/I/O。
首次red检查的两项失败是缺诊断字段；并发测试在修改前已经通过，完整日志保留。

边界依据：[Python RLock 官方语义](https://docs.python.org/3/library/threading.html#rlock-objects)
支持同一所有者的互斥；[vLLM 0.30 PEFT配置读取源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/peft_helper.py)
只证明该配置读取路径不回写文件，不代表已排除所有外部写入。
[Linux ext4分配说明](https://kernel.org/doc/html/next/filesystems/ext4/allocators.html)
提示物理分配有独立时序，但当前没有足够证据将此次失败归因于ext4。
不能因此加任意sleep/fsync或放宽签名/容量检查。

源证据：`paper_results/ieee_tc/p2_backend/20260928_d90_file_publication_diagnostic.json`。
下一步只允许一次同driver、同配置、同profile的3B原前缀诊断，以新记录保留
首次失败；不是验证已知修复或正式性能。7B和baseline继续等待，D78/D80/D81/
D88/D89不重做。异常后最后native lease归还问题仍待核查，不能用外置清理
冒充完整生命周期合格。
