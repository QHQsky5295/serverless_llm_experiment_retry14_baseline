# D474：HOST 准备执行视图的依赖证明

2026-10-12。主线仍为 7B PrimeLoRA。父版本
`fedd70afed64c8f1ee754ad6bfd939ce368d6b5d`；保留 D472 的 CPU 准备生命周期。
这不是完成的 GPU 性能实验，也不是 PrimeLoRA 已经最优的结论。

## 新证据与假设边界

D473 已完成的原始记录分析不重跑：D472 比 ElasticLocality 平均 E2E 高
902.659 ms，其中 decode 差 490.502 ms；不能将这种阶段差作因果分解。
45 个 TPOT 未达标请求中 33 个不与同 owner 的主动加载重叠，不能把所有
生成退化归咎于加载。D472 有 1,013 条 HOST 准备任务、2,427 个 file-held
attempt、967 次实际加载、1,457 个延后回执和 3 个缺终态回执的 attempt。
这些数值是旧运行的观测，不是新优化可减少多少毫秒或 GPU-s 的估计。

D463 只精简**入队前 owner/clock 查询**；D465 完整运行未改善性能。
保留这个否定结果，不回滚后藏掉，也不借 D472 的其他改进证明 D463 有效。
D473 因此禁止机械地把所有 full snapshot 换成 identity。本轮重新逐项检查
了另一个边界：`_queue_ieee_native_host_preparation.execute`，包括其不常见
的来源冲突分支。只有以下精确依赖证明支持本候选；其他查询不扩展。

假设：执行前查询没有物理图消费者，却在同步 core 中遍历全体 HOST storage。
改用**另一次新鲜的原生 identity 查询**，可能减少与生成共用的控制开销。
此假设只由后续单轮完整回放决定，不能由调用次数或 CPU fixture 接受。

## 消费者及必须保留的检查

| 执行步骤 | 消费的证据 | 本次处理 |
|---|---|---|
| 将执行绑定到入队 owner | 当前 owner、clock | 新鲜查询，严格一致；不复用入队返回值 |
| 来源冲突检查 | adapter、path、content、source incarnation、rank、当前 slot | 现有 live `owner.source_snapshot()`，不缓存 |
| 已在 GPU 的目标 | 当前 slot 与完成确认 | owner 原有 `_refresh`/确认不变量保留；不发起物化 |
| 替代文件副本导致计划失效 | native 身份、完整性、epoch/时钟、两个当前 SHA 确认文件 | 原 helper 与 parser 原样；不授权重标来源、加载或容量 |
| 保护文件 | 文件 owner 锁、当前 epoch、path/content 与引用 | 原样；锁不跨 await |
| 真正 HOST 准备 | 当前 native epoch、源绑定、预算、staging、allocator、workspace | 原实际执行事务原样；变化会延后或失败 |
| 完成、取消、丢失响应 | 真实 loader 回执和引用释放 | 原样；不加入重试或 fallback |

关键发现：`NativeSourceSnapshot.from_native` 本来就允许不含 footprint 的
readiness-only 视图；`supersede_confirmed_native_file_copy` 只使用身份和
文件发布证据。其返回是“旧计划失效”，不是加载/准入成功。相反，
`NativeAdapterSource.service_class` 明确拒绝无 footprint 的视图，因此它
不能冒充 Eq. (2) 的准备类别，或用于规划/容量计算。

这不是删除实际 admission 的 pinning/alias/allocator 检查，也不是以 GPU-hit
推断 HOST 安全。未被本只读消费者使用的物理表示不在这里额外验证；下一次
真实物理消费者仍执行完整检查。D329 的 storage resize、alias、pin registration
等跨调用缓存反例全部保留。Routing/planner 使用的完整注册图、GPU pool，
以及 demand/preparation 的实时物理检查都未改。

## 唯一生产修改及历史/原始来源

只有上述 `execute` 的 operation 从 `source_snapshot` 变为
`source_identity_snapshot`。全生产 AST 差异必须恰好为这个字符串。
worker、owner、parser、planner、admission、配置、生成、SLO 和基线均不改。
身份 endpoint 仍在同一原生线程执行新鲜 owner 观察，不新增 endpoint/cache。

- D178/D180/D181、D193/D194/D202 已有视图精简不重复实现。
- D312 文件锁、D330 planner 发布、D352 同事务 HOST 检查复用和 D355 原子
  控制交错保留；D462 pin-query 替代失败不重试。
- [vLLM 0.30 worker 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
  明确原生 LoRA 修改由单线程 core 驱动；这不证明跨调用物理状态不变。
- [vLLM 0.30 core 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
  与本地 D355 控制交错一起界定事务与模型推进边界。
- [PyTorch 2.13 Memory.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/aten/src/ATen/native/Memory.cpp)
  表明 pinning 为原生 pointer 查询，不是可随意删除的 Python 标签。
- [vLLM CPU 开销分析](https://vllm.ai/blog/2024-09-05-perf-update)
  支持检查 host 控制工作这一方向，其速度数字不用于估计本机收益。

以上官方资料于本轮在线核查；没有采用技能示例中的通用 500 ms/100 req/s
门槛、并发扫描或新依赖。`serving-llms-vllm`/`optimize-for-gpu` 仅用于核对
执行边界与“先正确性、后完整性能”流程，冻结 V1/V2 优先。

## CPU 资格、局限与下一步

新增测试直接拒绝在该执行路径调用 full snapshot；覆盖真实 fixture 的
加载/文件保护，以及**执行观察之后再次变旧**时保留 stale-epoch 延后，必须
有外部 owner wake 后才重新观察。另验证无 footprint 的 supersession 与全图
视图具有相同身份、拒绝错误 owner/clock/epoch/rank/source/完成确认，并且
不能构造 service class。原有取消、预算、GPU 已就绪、来源替换、pinning
和请求生命周期测试继续运行。

红测试在未改生产代码时运行：44 项中的 2 个新增依赖断言失败，另 42 通过；
其中 41 项是 unittest 自动发现了导入的既有 TestCase，并非 44 个新增测试。
已改成模块导入以避免重复发现，原测试源/日志保留。此红结果不是运行时
性能失败。首次候选 targeted 350 项全部通过，无失败/错误/跳过。
最终完整回归 **953 项全部通过**，无失败/错误/跳过；测试体 35.240 秒，含
发现/导入 40.366 秒。350/953 是重叠测试套件，不是独立实验。具体 SHA 和
资源回执以汇总目录 qualification.json 为准。CPU 测试不证明 GPU 数值或
完整性能改善。

通过 CPU 资格后：校验并备份 → 复用 D471 的原 100 请求原生资格（仅隔离
输出/缓存路径）→ 通过后原 D472 配置新键单轮完整 4,000 W0。
不重新生成负载/权重/发布缓存，不改变模型配置，不重跑相同键找赢家。
报告全部正确性、共同联合 SLO、物理 GPU-s 和尾延迟，旧候选全部保留。
若未改善，停止这条局部候选，不再机械扩展更多 identity 替换。

原始记录：`results/ieee_tc/p2_backend_qualification/d474_20261012/`；
汇总：`paper_results/ieee_tc/p2_backend/d474_host_execution/`。
本候选不关闭 7B、W1/W2/G2、3B、HydraServe、消融/机制/敏感性和文档矩阵。
无 GPU/remote 操作、基线优化、图表、论文修改或新数据。
