# D667：区分新鲜 HOST 清点与其外部等待的有界观测入口

2026-10-18。3B W0 主线，属于必要测量准备，不是新优化候选、性能实验或
G1/G2验收。本次没有改模型配置、缓存策略、准入判断、扩缩容或九个行间公式。

## 现有证据与缺口

D666复用两次完整运行，确认同HOST路径884条请求的acquired→snapshot
均值从83.110ms降至1.320ms。但D665联合达成仍为3550/4000=88.75%，
物理GPU占用15896.448 GPU-s，不能用局部收益替代整体目标。
GPU就绪请求也有95条TPOT违约，剩余问题不能都归因于真实远端传输。

目前每个需求事务的第一次HOST预算检查仍清点当前完整tensor/storage图。
D328旧7B栈样本出现is_pinned热点，D351旧7B原生核心观测发现控制工作占据
执行线程；这些是选择测量对象的依据，却不是当前3B内部函数耗时。
D666的pending→acquired、runtime-slot等区间包含其他工作和等待，不能直接
当作HOST清点耗时，也不能从用户TTFT扣除它们构造优化后的结果。

历史边界：D329已经否决仅凭owner epoch/model ID/TTL复用物理图；D352和
D663仅复用同一已证明事务内的后检查。D333的pending/HOST盲融合，D595的
并行压力屏障和D628的锁分拆失败证据保留。本次不重试上述候选，不堆叠新策略。

## 本轮原始资料核查

- [vLLM v0.30.0 EngineCore源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)：
  客户端控制处理与模型执行边界用于定位本机观测点，不引用其他版本的加速比例。
- [PyTorch v2.13.0 Memory.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)：
  CPU tensor的is_pinned查询底层storage pointer的accelerator属性，不能当作
  一个普通Python布尔字段；也不由此假定本机查询一定是剩余主要瓶颈。
- [vLLM官方CPU开销说明](https://vllm.ai/blog/2024-09-05-perf-update)：
  CPU侧处理可能影响推理推进，支持分开测控制与执行；其旧版本结果不直接
  外推到本机3B、LoRA和当前版本。

## 观测合同

复用已有`CoreHostTiming`，新增显式、默认关闭的
`FAASLORA_TC_HOST_INVENTORY_TIMING=fresh_inventory_v1`。必须同时启用
`host_spans_v1`并通过既有diagnostic-prefix实际服务资源域核验；正式运行、
孤立开关、未知值、错误资源域都拒绝，不静默降级。

只包裹两个完整函数：`_ieee_lora_host_inventory`和
`_ieee_pinned_host_allocator_stats`。不逐tensor插桩，不加CUDA同步、不改
查询次数，不替换返回值，不缓存清点结果。每次原函数仍恰好调用一次，原异常
保持同一对象。仅在本native core线程的活动观测span内记录；其他线程或span
外调用原样执行，不将其误记成core阻塞。

每条记录有本地monotonic起止、thread CPU起止、父span、returned/raised。
inventory额外记录已有结果中的scope、adapter/allocation数量和非负总字节，
不遍历tensor、不保留storage/model对象、不记录pointer、请求/adapter ID、
prompt、凭据或RPC参数。关闭时只恢复仍属自己的函数包装，不覆盖其他修改。

沿用每进程200000 spans/64MiB上限、显式truncation及关闭回执；不能截断后
宣称完整覆盖。嵌套时长不可直接相加。父span包含子观测写入开销，不能把观察
比例当作无扰动精确占比或可删除时延。观测运行不参加正式性能排名。

## 下一次且仅一次的原生测量

复用当前D665的3B配置、W0原trace前1000条、原500-adapter集合、真实远端
已发布只读压缩工件、同准备窗口和资源保护。仅换唯一输出/cache路径与显式
诊断开关，不改变arrival、token、SLO、容量、扩缩容或论文公式。沿用现有
诊断runner和远端管理，不新建负载、工件池或另一套profiling框架。

目的：分别计算完整/请求范围清点的次数、均值/P95、线程CPU占比，按父控制
操作分组；对照模型执行和短输出decode窗口，区分函数内工作与其外部等待。
完整覆盖、父子包含、已知scope、关闭/截断状态须验证。输出诊断数据表即可，
此时不作论文展示图，不由prefix的SLO或GPU-s决定候选胜负。

停止规则：测量一次完成后判断证据。若新鲜HOST清点确实占据大量控制工作，
才核查保持物理状态正确性的新设计；否则转向已测的其余关键路径，不继续为
该假设反复加样本。无完整storage生命周期证明，不引入跨请求图缓存。D663
仍是当前观测版本，局部机制有效但整体性能未获接受，不把它写成达标优化。

## 仪器变更清单

遵循`system-profile`的整函数包装、低扰动和变更可追溯要求。

| 文件 | 变化 | 默认服务行为 |
|---|---|---|
| `faaslora/utils/core_timing.py` | 可选整函数span、标量投影、恢复及开关核验 | 不安装包装 |
| `faaslora/scheduling/vllm_ieee_scheduler.py` | 让孤立新开关也进入诊断合同核验 | 原路径不变 |
| `tests/test_ieee_tc_core_timing.py` | 正常/异常、隐私、跨线程、上限、恢复和资源门槛测试 | 不进入服务 |
| `tests/test_ieee_tc_d456_diagnostic_contract.py` | 旧断言改为检查stack-only分支明确关闭core观测、不得启用core观测 | 不进入服务 |
| raw `d667_20261018/`中的资格、停止、关闭、备份辅助文件 | 复用已有受限检查与备份脚本；失败尝试不覆盖 | 不进入服务 |

未改`gpu_monitor.py`或`residency_manager.py`中的生产机制。无需为恢复普通
性能运行删除仪器代码；诊断开关必须关闭，并在正式执行身份中核验。

## 检查期间的两项异常（完整保留）

原生vLLM0.30环境中的22项focused检查通过。随后把旧大范围launch测试放入
该环境，产生辅助子进程与持续memory.high压力。只停止本次精确身份资源域，
停止前峰值3447791616 B，max/OOM/oom_kill/swap均为0；递归进程已释放。
第一次停止入口在该Python中缺少pidfd_open，尚未终止进程；保留首次回执，
随后用系统Python执行同一所有权停止流程成功，没有降低保护门槛。

切回D595已有的合格CPU测试环境，并沿用launch-first顺序后，237项中236项
通过，1项旧断言失败：它把`unset FAASLORA_TC_CORE_TIMING`也误判为启用
第二观测器。HEAD中的wrapper自D603起已有这个明确关闭动作，本轮没有修改
wrapper。只更新测试为“必须unset、不能export”，不改实际服务路径。
该次检查峰值1928179712 B，内存事件与swap全0，已关闭；修正后同一套237项
检查全部通过，另存analysis2，不覆盖失败。其峰值1935835136 B，内存事件/
swap全0，精确资源域递归空且关闭。加上原生环境22项，共259项最终检查通过，
不把前述失败尝试抹去；147个旧投稿保护文件和三项既有用户修改SHA未变。
当前没有GPU回放，没有新的性能提升或完整达标结论。

后续大范围CPU回归使用已验证的
`/home/qhq/anaconda3/envs/sllm_vllm0102_newserverless_20260518/bin/python`
及既有模块顺序；原生环境只执行明确有界的针对性检查，不能再次盲跑旧大套件。
带pidfd的所有权清理使用`/usr/bin/python3`。环境差异是已观测关联，不宣称
仅由某个包版本就解释了第一次子进程增长的完整原因。

其余3B W1/W2/基线/dLoRA、同版本7B、A1–A5/S1–S13、离线成本/CE、两轮
审稿文档和最终图表保持完整开放；不重启已完成HydraServe适配，不动原k3s。
