# D139：当前控制路径的下一项瓶颈诊断

2026-10-01。状态：既有记录的 RPC 离线恢复完成；没有 serving 修改、新计时、
profiler 或回放。D138 已封存并推送 `35b3e84c601ea99cda7762f3262ac7d9b2dae0e8`。
本项只扩展既有 analyzer，使用已生成的有界请求投影，不重新读取原始大文件。
这不是新的性能实验，也不是全局计划的替代。3B、7B 性能均保持 OPEN。

## 1. 从完整结果定位问题，而不是先增加容量

以下直接复用已封存的 D137/D138；各一轮，多项实现变化、不同模型，
不作两模型间因果比较，不重新解析原始大文件。

| 平均阶段，秒 | 7B D137 | 3B D138 |
|---|---:|---:|
| 计划到达到外层许可 | 599.531550 | 4.458793 |
| 许可到来源准入 | 1.745965 | 4.082005 |
| 来源准入到原生 dispatch | 2.049556 | 3.466771 |
| 原生 dispatch 到末 token | 4.540801 | 4.157320 |
| 末 token 到控制器完成 | 1.162485 | 1.352705 |
| 控制器完成到外层终态 | 0.477029 | 0.975116 |
| 许可到终态上包络 | 9.975837 | 14.033918 |

上包络不是精确许可释放时刻；native 区间也不是纯 GPU kernel 忙碌时间。
目前不能将上表的非 native 阶段全部归为某一个函数或远端网络。
7B 每副本峰值原生并发为 2，3B 为 8，已排除“整个后端始终串行”的描述。
3B 相对 D118 的 TPOT 退化与 252 条输出 hash 变化仍需核查，不当作已解决。

## 2. 本次已经确认的代码边界

当前源码与 D138 执行源码一致；此次没有修改 serving 文件。

1. `run_all_experiments.py::run_one.serve` 取得入口许可后，直到
   `_exec_request` 返回/异常清理才在 finally 中归还；D136 的 FIFO 所有权
   修正仍保留，不回退到可抢占旧等待者的实现。
2. `_exec_request` 的 finally 先调用 `_finish_runtime_request_reservation`。
   后者核对原生终态、pending demand 关闭、GPU/HOST 引用释放，最后提交
   控制器占用释放。不能在仅收到末 token 时假报这些资源已经归还。
3. `InferenceEngine.ieee_routing_sources` 的验证和消息投影在独立 frontend，
   但其前置 `source_snapshot` 仍由 GPU core 的 worker extension 构造。
   D132 减少了父控制器消息，不代表原生存储图构造已退出推理关键路径。
4. `IEEEWorkerObservationExtension.ieee_gpu_reference(source_snapshot)`
   每次构造 registered HOST inventory、GPU pool inventory，再构造含 staging
   的 HOST inventory。后者再次遍历注册对象，并读取当前 allocator。
   这是静态可见的重复遍历；尚未测得其当前 CPU 时间，不能直接声称它是主因。
5. 实测环境 vLLM 0.30.0 的 `EngineCoreProc.run_busy_loop` 先处理输入队列，
   再推进 engine step；`UniProcExecutor.collective_rpc` 默认直接调用 worker。
   因而这条同步观测路径可与解码争用 core 线程。已有 GPU 异步执行可能重叠，
   不把函数墙钟时长直接等同于 GPU 停顿时长。

## 3. 历史与文献约束

D123 旧版本仅采样父控制进程，不能证明当前 core 热点。D127 JSON decoder
微测已经归档，不重复；D128/D130 已隔离并封装纯规划计算；D132 已投影
新鲜路由状态；D134 绑定所选来源；D136 修正入口 FIFO。这些修正的测试
证据可复用，但不能用它们代替当前执行路径的性能测量。

[vLLM 原始性能分析](https://vllm.ai/blog/2024-09-05-perf-update)说明 CPU 工作与
执行串行化可能使 GPU 等待；这里只借鉴问题分解，不借用其加速倍数。
[0.30.0 core 源码](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/core.py)
与[UniProc 源码](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/executor/uniproc_executor.py)
用于核对实际调用边界。线上入口与原生调度容量不是同一种资源，不能通过
删掉引用/预算确认来消除等待。

[0.30.0 可复现性说明](https://docs.vllm.ai/en/v0.30.0/usage/reproducibility/)及
[batch invariance 说明](https://docs.vllm.ai/en/v0.30.0/features/batch_invariance/)
仅提示输出变化的一项可能来源；尚无证据将这 252 条逐条归因于批次。
不静默切换该开关，不把它的潜在性能影响混入控制路径优化。

## 4. 下一项可证伪问题与限界

问题：在当前完整控制路径中，原生存储观测的构造/验证是否占据足够的
CPU 执行时间，并与请求/解码推进重叠，以构成值得优先优化的原因？

下一步先核查现有记录是否能直接恢复每类调用时长和对象规模；不足时
复用 D123 的有界诊断入口，以当前版本、既有 prefix 索引及相同模型配置
采样父进程和实际原生 core，保留全部失败及诊断身份。未准备或启动此轮。
不得用旧父进程 profile 比例冒充当前 core 比例；不新造输入或整池。

若没有支持该方向的新证据，归档这一假设并回主线，不反复测相同小组件。
若支持，优先改进同一 owner 下的表示/消息/重复计算，保持新鲜状态、真实
footprint、共享存储并集、确认发布、物理 reservation 和取消/退役证明。
跨 epoch 缓存、减少检查频率、提高 cap 或改超时均不是本项已批准的候选。
任何实际候选仍须最小正确性/因果检查后回普通 Full，不由组件收益结项。

## 5. 本次检查的代码身份

| 文件 | SHA256 |
|---|---|
| `faaslora/memory/gpu_monitor.py` | `43c4082fbb83bea74761a65d3cca5db0629f9be8abb91442807bab3974c95216` |
| `faaslora/memory/residency_manager.py` | `bc13c9d4705e126ff9f5068e78db5a68e69ea6282b0ab75631b70e61ae14923b` |
| `faaslora/scheduling/vllm_ieee_scheduler.py` | `ccb7f1a79ef0b5db84b62f2bb8807cfc2490edca0104b051db230f97f5d8ce41` |
| `faaslora/preloading/planning_cpu.py` | `f0c4d7f34abaf336873bf97a26cb51e870972de08688cf53f89a5b1ba44d3779` |
| 已安装 vLLM `v1/engine/core.py` | `268bf40534a853812867ae8e80040f190983cb364b41182b55969aace20c93f5` |
| 已安装 vLLM `v1/executor/uniproc_executor.py` | `3cb039b80fe9b03dc1122f93c1f8a4098ef1355104b317ff6df25358f4349494` |

只读参考 serving-llms-vllm 技能的阶段划分；不采用其通用 QPS、TTFT 或
利用率示例作为本项目标准，冻结指标协议 V1 不变。上述代码检查本身未增加测试；
下述 analyzer 扩展另有 38 项针对性测试。

## 6. 已恢复的完整 RPC 测量（13:20 完成）

原 D138 curator 查询的是不存在的 `parent_response_pickup_delay_ms` 和
`parent_thread_resume_delay_ms`，实际字段带有 `parent_rpc_` 前缀。
旧 curated 文档中的这两项空值表示查询未找到字段，不表示原始观测丢失。
不改写封存文件；此次新增补充分析使用实际 `native_token_timing` 字段，
并逐条校验与外层副本值一致。

每模型均覆盖全部 4,000 个唯一请求，与封存 deployment/terminal 对齐。
22 个诊断字段均有有限非负存储值，没有缺失/null/非法值；字段存在不等于
各字段都是独立实测量，下面明确区分结构性零和遗留未计时字段。
这些不是两模型间因果比较；每模型 n=1，不计算 CI 或显著性。

| 生成调用相关观测 | 7B 平均 / P95 | 3B 平均 / P95 | 单位 |
|---|---:|---:|---|
| 申请父进程通信通道 | 0.017 / 0.027 | 0.082 / 0.022 | ms |
| 请求发送 flush | 0.120 / 0.205 | 0.133 / 0.215 | ms |
| worker 收到请求前间隔 | 1.711 / 1.360 | 3.954 / 2.302 | ms |
| 回复就绪至父进程读取 | 1,095.832 / 5,969.025 | 1,264.889 / 7,246.434 | ms |
| worker 完成至 controller 完成（单调时钟） | 1,096.603 / 5,969.813 | 1,265.690 / 7,246.740 | ms |
| 父进程 RPC 减 worker handler 的残差 | 1,097.766 / 5,970.202 | 1,269.129 / 7,248.093 | ms |
| RPC 响应大小 | 1,774.423 / 1,780 | 1,774.813 / 1,780 | bytes |
| 路由记录区间（含 await） | 779.620 / 1,607.048 | 1,138.745 / 2,385.619 | ms |

均值高于 P95 的小开销行不是错误：少量长尾可提高均值。
所有 P95 为 Type-1，表内仅显示舍入值，完整 CSV/JSON 保留原精度及最大值。

### 观察、解释与下一步

1. **观察：**两模型均有秒级回复获取尾延迟；申请通道均值低于 0.1 ms，
   单条回复约 1.8 KB。同一单调时钟记录的 worker→controller 完成间隔
   也接近该回复延迟，支持“存在实质性完成传递等待”，而不是只看墙钟差。
2. **解释边界：**这包含 worker 序列化/发送、loopback 和父事件循环读取，
   不能据此断言全部由某函数、JSON 或 GIL 引起，更不是远端 artifact 网络时间。
   小回复与短通道等待不支持把“扩大连接池”作为优先措施；不排除少数通道长尾。
3. **下一步：**优先测当前父进程不能及时处理已就绪回复的 CPU 路径，并同时
   观察真正的 GPU core、frontend 与 planner 子进程，避免再次只测父进程。
   复用 D123 既有前 1,000 请求诊断入口、当前 7B 配置与真实远端合同；
   不是新的 workload，也不把诊断 profile 当普通 Full 性能结果。
   尚未准备/启动该回放。没有证据时不添加吞吐补丁、不增加 cap。

[Python 3.12 asyncio 官方文档](https://docs.python.org/3.12/library/asyncio-dev.html)
解释同步 CPU 工作可推迟同一事件循环中的其他任务，作为待验证机制依据。
[py-spy 0.4.2 官方说明](https://github.com/benfred/py-spy/blob/v0.4.2/README.md)
提供 subprocess 跟踪和 GIL 采样；后者不覆盖释放 GIL 的原生计算。
因此应按实际 PID/角色分别解释采样，不能把样本占比当墙钟因果贡献；
不更改系统 ptrace 策略、不调用 locals、不采集凭据。

### 遗留计量边界不能被零值掩盖

- native-async 路径显式设置 `parent_rpc_thread_resume_delay_ms=0`，
  因为不再从阻塞 RPC 线程恢复；不是“测得完全没有调度延迟”。
- IEEE 路径的 `adapter_path_resolution_us`、`gpu_admission_decision_us`
  保留遗留初始化零；两模型均为 4,000 个零，不证明加载或 admission 免费。
- `routing_decision_us` 包含 await；遇到 snapshot 返回 None 的 continue
  发生在累计之前，所以它不是所有路由尝试/全部控制路径的 CPU 时间。
  `control_path_total_us` 此处等于该字段，也不能作为全系统控制开销。
- 这些请求级 RPC 数值仅对应 generate，不包含逐类 source_snapshot/
  preparation/retirement RPC 时长。已有数据不足以隔离 native 存储图遍历；
  因此当前采样诊断是受影响路径的必要新增测量，而非重复旧微测。
- 不将上述互相重叠的 span、残差或分位数相加，不从 E2E 中扣除它们造改善。

## 7. 产物、验证与资源

- `scripts/analyze_control_path_overhead.py --rpc-breakdown`：仅新增 opt-in
  模式，既有 native timeline/legacy 模式不变；显式输出目录，拒绝覆盖。
- seal SHA、projection/deployment/terminal SHA 验证后才能读取有界输入；
  保留全部 offered IDs。缺失或非法诊断值另列，不补零、不别名猜测。
- `tests/test_ieee_tc_rpc_breakdown.py` 14 项 + 原 timeline 24 项，共 38 项
  通过，涵盖 Type-1、零/缺失/非法值、重复/缺请求、时钟、生成身份、SHA
  和覆盖/大小限制。测试只因 analyzer 变更重跑，不重跑已封存 GPU 测试。
- 两模型离线计算分别 0.78/0.75 秒，峰值 RSS 129,024/130,944 KiB；
  3/4 GiB、swap=0、CPU 2/3/26/27 限制实际生效，内存事件全零。
- 数据：`paper_results/ieee_tc/p2_backend/20261001_d139_rpc_{3b,7b}/`，
  含 summary、精确指标表和完整逐请求诊断 CSV。大逐请求 CSV 保留本地；
  小汇总、脚本、测试、运行/清理回执进入备份。
- 本次采用分析/学术绘图技能的“精确多指标比较优先表格”规则，已交付本表。
  未生成把 n=1 描述性结果包装成新性能提升的图。
