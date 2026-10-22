# D737：规划状态的解析位置，而非新的编码器

## 主线与唯一假设

D736完整3B W0为3706/4000联合达标（92.65%），尚少94条到95%；没有新的
性能合格点。D734业务controller样本中18/536位于native RPC解码；采样不是
CPU占比。D735已复用规划器的完整图验证结果，但controller仍先解码完整图、
再用pickle冻结交给同一个规划进程。该重复表示路径可从当前源码直接确认。

假设：保留同次新鲜JSON回复的不可变字节，仅在controller解开成功/错误外壳，
由既有规划worker解析完整图，可减少请求事件循环的同步工作；额外worker解析
与更大的IPC字节可能抵消收益。先用已有实际快照进行一次有界CPU组件研究，
记录两侧时间和总量，不把局部时间下降换算成TTFT/SLO改善。

## 历史排除与依据

- D172已采用msgspec直接从bytes解码，不能再次将其当新优化。
- D192的Python递归检查+新JSON编码原型更慢，已否决；本候选不改sender编码、
  字符/浮点表示、换行分帧、8MiB上限或后端消息格式，不重复该原型。
- D163已优化执行结果的key-only查询；本次不重复该问题。
- D329未证明物理表示缓存失效完整，D687/D688无完整收益后撤销：本次无跨epoch
  缓存、不删除分配/别名图，不改观察、HOST inventory或GPU执行视图。
- D735的已验证publication投影仍保留；观测capture时间不刷新为解析完成时间。

官方依据：[msgspec Raw延迟解析](https://raw.githubusercontent.com/jcrist/msgspec/main/docs/supported-types.rst)、
[Python asyncio阻塞边界](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)、
[vLLM 0.30.0序列化](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/serial_utils.py)。
只借鉴表示/执行隔离，不借用其性能数字，不移植MessagePack或unsafe pickle。

## 有界研究与推进条件

复用D192的精确类型/键顺序/float.hex比较与交替计时方式；复用D725已有3B
原生快照，另列D169历史7B大图作形态边界，不将历史大图冒称当前所有RPC。
按当前routing_source_snapshot省略的allocator/staging字段形成明确标注投影，
保留完整registered HOST图和GPU几何。不重跑模型、不生成权重/trace、不复制池。

两条组件路径均计入controller冻结和worker恢复：当前JSON→dict→pickle→dict；
候选JSON外壳→不可变result bytes→pickle→bytes→dict。JSON编码在两路径相同，
不作为收益。复用原有数值/Unicode边界fixture；非法数据必须在消费前拒绝，
如拒绝阶段变化则单列，不声称全域无条件等价。只测组件，不执行新的服务配置。

若parent工作没有下降或有不可接受的语义/总体代价，归档否决，不不断微调。
若值得推进，再实施唯一只读规划入口和针对性CPU/spawn/取消/状态时序检查，
之后一次完整4000回放。不得增加额外GPU短资格；完整性能决定接受与否。

使用既有aux CPU2/3/26/27、3/4GiB、swap0，不改共享k3s、远端、GPU或生产代码。
原始目录results/ieee_tc/p2_backend_qualification/d737_20261023；研究前完整
PLAN和最新状态已核对，V1/单轮V2不变。无新baseline优化、无图、无CI。
3B其余场景/基线、7B最终同版本、A1–A5/S1–S13和最终交付范围完整保留。

## 组件结果与实施决定

一次研究完成7种保存状态形态、42条计时行和13个边界案例。以下是同一CPU
组件研究的三组交替循环均值，单位ms；不是三次模型回放、不作服务CI。

| 来源/快照 | wire bytes | 原parent | 候选parent | 原worker恢复 | 候选worker恢复 | IPC字节比 |
|---|---:|---:|---:|---:|---:|---:|
| D725 3B/0（空sources） | 114475 | 1.421686 | 0.152354 | 1.013605 | 0.512980 | 1.794 |
| D725 3B/1（4 sources） | 218312 | 2.688331 | 0.443371 | 1.123859 | 1.248516 | 2.140 |
| D725 3B/2（驱逐后） | 114478 | 0.926972 | 0.158068 | 0.509578 | 1.073036 | 1.794 |
| D169历史7B/0 | 825838 | 8.703500 | 1.532509 | 5.544270 | 4.767287 | 1.916 |
| D169历史7B/1 | 1174717 | 13.088709 | 1.772039 | 7.213364 | 6.974765 | 1.928 |
| D169历史7B/2 | 1873470 | 20.327750 | 3.669282 | 13.334678 | 14.337616 | 1.941 |
| D169历史7B/3 | 2221326 | 25.689447 | 3.900935 | 15.724447 | 15.089234 | 1.945 |

parent是解析/冻结，worker是恢复/解析，两侧仍是同进程计时的组件模拟；实际
进程IPC、队列、planner运算未包含。传入worker的字节变大，不能只展示parent
减少。当前3B只有一个非空快照，不能拿历史7B大图外推当前完整消息分布。

所有有效样本精确值、类型、键顺序及float.hex一致。非有限数/截断/尾随字节等
拒绝；浮点溢出与非法UTF-8在候选中可能从parent延后到worker拒绝，仍在输入
验证和发布之前。不能称拒绝阶段完全一致。新入口的RPC外壳和timing header
按明确类型检查；它不是任意JSON协议的透明替代。

据此实施唯一候选：新增不可变`FrozenPlanningSnapshot`，仅native subprocess
的只读规划调用返回它；controller解析小timing header并沿用原RPC计时处理，
原始native capture/epoch/owner保持。原JSON编码与socket协议未改；旧dict入口、
legacy/in-process路径及其他RPC完全保留，无出错后回退。既有pure planner在
调用原输入验证前解码一次，执行D735原有同次验证与publication投影；无新增
进程、缓存、控制参数或物理状态捷径。

实现另作一次含真实timing-header与metadata冻结的有界检查：同D725非空快照，
parent 2.025842→0.614355ms（下降69.674%），worker恢复6.191286→1.150567ms，
IPC102044→218746bytes（2.144倍）。这是固定顺序40次循环的一次组件检查；
worker数值有分配/回收及调度影响，不作系统加速归因。第一原型msgspec0.21.1，
实现检查复用既有CPU环境msgspec0.20.0；各环境身份随回执保存，不混称相同运行条件。

结论：只接受为待完整回放的实现候选，尚无新的TTFT/SLO/GPU-s结果。使用结果
分析skill分开观察与解释；vLLM技能仅用于服务合同检查，不采用其示例参数、
通用500ms目标或宣传加速数字，不改既有vLLM0.30.0后端/profile。下一步在
针对性检查和备份后执行一次新的完整3B W0，不追加短GPU或再测同一组件。

## 针对性检查与限制

既有CPU/spawn、规划publication、传输/所有权和新增wire检查共122项通过，
91.057秒。随后新增native/legacy入口1项通过；部署环境msgspec0.21.1的2项
wire边界复核通过（与CPU环境检查内容重叠，不冒称额外独立科学样本）。
计划目标/预算/hash、GPU objective、不可变publication与原输入一致；文件
plan UUID仍每次新建。损坏图拒绝，取消join后丢弃结果，原RPC失败处理保留。

新增入口夹具首次失败是运行timing_contract与模拟profile身份不一致；修正
夹具为各自明确身份后通过，生产validate_runtime没有修改。一次重试启动的
aux UUID多一位，在测试前被原检查拒绝；失败/回执完整保留。未加兜底或放宽
校验，没有系统性能失败重试。当前没有新的GPU运行，不能宣称D737达标。

CPU研究峰值197345280B，122项测试峰值1815363584B，各受限任务high/max/
OOM/swap均0。原始输入、用户manifest、旧投稿保护结果未改；完整清理回执和
curated manifest在本次检查点保存。下一仅需新执行键的完整3B W0；旧D736
不重复，D711数值工件限制继续披露但不重做审计。共享k3s和远端均未操作。
