# D90：完整路径的有限联动检查

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
