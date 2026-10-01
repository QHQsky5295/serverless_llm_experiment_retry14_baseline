# D149：所选来源复核的信息边界

2026-10-01。Prime 优先开发候选；基线继续暂停。本项不是新调度算法，
不改变 IEEE 九式、准备成本、并发容量、超时、输入、后端或远端交付。

## 问题及历史证据

D148 普通 7B Full 全部 4,000 条原生生成合同完成，但平均 TTFT 仍为
591.861802 秒，dispatch/admission 为 589.561252 秒。原生生成区间平均
4.879046 秒，入口至外层终态上包络为 9.998352 秒；上包络不是精确许可
持有时间，不能把所有非生成时间归因于某个函数。回复就绪至父进程读取
平均 1.094662 秒。D147 的显存查询优化未形成整体性能改进。

D143 控制器业务期 570 个栈样本中，最近项目帧 `_footprints` 出现 29 次；
这是采样计数，不是 CPU 时间份额。D132 已把路由的完整图验证放到独立
frontend，保留新鲜查询；但当前请求的两个后续来源查询仍把完整库存送到
父进程再验证。源码确认这些消费者只使用 `NativeSourceSnapshot` 中的
来源、容量、表示、owner/epoch/clock，而不消费张量别名边。
旧请求证据仅排除 `native_footprints`，仍保留 `native_staging_footprints`。
结果文件大小本身不是在线耗时证据。

## 单一候选及可证伪假设

将 `_acquire_runtime_gpu_reference.observe_source` 和非 native 来源准入
复核接到既有 `ieee_routing_sources`。它仍在每次调用中获取完整原生状态、
验证完整存储并集，随后发送类型化投影；父进程继续验证投影身份与新鲜度。
不添加缓存、TTL、旧快照复用、猜测容量或备用执行路径。

假设：减少这些请求路径上传往集中控制器的完整清单与重复验证，可减少
集中事件循环负担，从而改善实际请求推进。组件和静态信息足以支持尝试，
但不能证明它解释全部排队；必须以普通完整回放同时观察 TTFT、TPOT、
GPU-s 与失败，不能只凭文件变小或单函数变快接受整体收益。

| 不变量 | 本候选处理 |
|---|---|
| GPU/HOST 源身份、完成状态与真实 footprint | 每次完整验证后投影，父进程仍校验 |
| owner/epoch 冲突、成员更换、未知 source | 原拒绝与重新选择规则保留 |
| acquire/hold/release/取消/原生退役 | 实际有所有权的命令不改 |
| planner、replacement、物理 admission | 仍使用完整原始接口 |
| GPU core 构造完整库存 | 不改，不宣称其成本已经消除 |
| 每请求证据 | 保存来源投影及所选 footprint，不复制未消费库存 |

项目脚本、实现和测试搜索未发现后续代码消费三种 `snapshot_*` 证据中的 staging 图；
完整库存仍由实际需要它的观测/规划路径保留。证据 schema 的变化明确记录，
不改写旧结果，不混称执行键相同。

## 原始资料核查

[vLLM 的 CPU/GIL 分离分析](https://vllm.ai/blog/2024-09-05-perf-update)
支持把同步 CPU 工作与请求处理的执行边界分开；不借用其性能倍数。
[v0.30.0 UniProcExecutor](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/executor/uniproc_executor.py)
用于核查 worker RPC 的实际执行位置。
[Python 3.12 asyncio 说明](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)
说明同步工作可能延误同一事件循环中的其他任务。本候选复用已有独立
frontend，不临时新增线程、进程池或改变原生调度。

## 验证状态

两项新增针对性测试覆盖四 tier 的完整请求路径，以及获得 HOST 引用后
新鲜完整图损坏时的拒绝/释放。第一轮在旧代码上得到三个边界断言失败，
另一个错误来自测试原先误以为图损坏会返回失败对象：现有接口实际抛出
ValueError。修正测试对异常的预期，不修改错误处理；原测试 diff/回执保留。

候选两项测试通过（0.098 秒），四 tier 均保持预期来源，已获得引用均释放。
通用虚拟 worker fixture 补入生产实际已有的 device UUID 字段。完整受影响
回归 700 项通过（25.238 秒；整条命令 36.70 秒，RSS 1,210,360 KiB），
包括 D132 的同源投影等价性、跨进程协议、错误图、共享存储、过期 owner、
取消、退役、请求生命周期与基础 smoke。未重复旧组件性能微测。

| 本次检查 | 结果 | 解释边界 |
|---|---|---|
| 四 tier 请求及新鲜错误图拒绝 | 两项针对性测试通过 | 虚拟推理，不是性能值 |
| 受影响回归 | 700 / 700 通过 | 不证明 GPU Full 优于旧版 |
| 三个测试资源域 | 各 3/4 GiB、swap0、CPU2/3/26/27 | 实际限制读回，终态后关闭 |
| high / max / OOM / oom_kill | 全部 0 | 无 GPU/远端实验启动 |
| 普通 Full | 待执行 | 冻结配置，保留全部成功与失败 |

按计划采用状态表，不把正确性测试画成性能提升。下一步校验历史保护和
来源、提交备份，然后一个普通 7B W0 Full；暂不接受整体性能收益。

执行记录：`results/ieee_tc/p2_backend_qualification/d149_20261001/`。
冻结指标 V1 不变；数值 adapter、warm/Resident、3B 性能与正式矩阵仍开放。
