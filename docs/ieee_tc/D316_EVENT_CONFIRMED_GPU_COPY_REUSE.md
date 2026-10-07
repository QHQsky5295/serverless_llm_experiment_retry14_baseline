# D316：按目标 GPU 副本寿命保留确认记录

2026-10-07。7B 主线的单一开发候选；CPU 验证完成，GPU 性能验证尚未开始。
前一轮 D315 完整结果与回退点 `e6cb59e099021fa528be2865718c259bab3e8b4e`
已推送。计划 SHA `0c808509…`、冻结 V1 SHA `5f0732ef…` 未变；没有新图。

## 为什么继续处理等待，而不再调 token 预算

D315 平均到达至准入为 681.229 ms，占平均 TTFT 52.226%；其中 routing
177.688 ms 内嵌，不能再次相加。相对 D303 的物理 GPU-s 仍高 2.517%。
99.362% 的持卡时间在业务窗口内，单独缩短最终收尾不能解决主要资源差距。
本轮不改 1024-token 工作点、缩容阈值、38 秒 cooldown 或评价 SLO。

历史 D200 已测得查询在原生执行之前有明显等待，D201 核对了串行 utility
路径；不再重复“压缩响应即可消除全部等待”的假设。D216 的 fresh identity
RPC 候选已撤回：虽触发 scope-local 复用，均值却未改善。D289 改成原生事件
传播，D301 修正了同 epoch 新记录被发布动作立即清除的问题。现在的差别是：
**不增加另一轮 identity RPC，而是消费已经收到的原生事件，决定哪条既有
GPU-copy 测量记录确实失效。** 不是重新运行 D216 寻找更好结果。

## 复用 D315 全部日志得到的机会证据

现有 `analyze_control_path_overhead.py` 增加一个严格离线汇总函数，入口复用
`analyze_d297_full.py --route-copy-output`。仍先核对源结果 SHA、完整原生数量
合同和实际物理生命周期；没有生成新输入、权重或推理结果。

| 保留观察（不是独立 RPC） | 数量 |
|---|---:|
| 4,000 请求的最终副本目标视图 | 15,919 |
| 标为本次读取的目标视图 | 7,946 |
| 其中 GPU 目标 | 4,069 |
| 同目标副本身份/测量相同、epoch 推进后再次读取 | 2,357 |
| 上一行涉及的请求 | 1,261 |

按原生捕获时间排列同 `(replica, adapter)` 的保留视图，比较 owner、copy ID、
path、内容 SHA、footprint 与 representation。HOST-only、缺失来源和变化的
目标不列入候选。相同 wave 可能出现在多条最终请求记录中；中间状态、真实
cache 命中和所有重试没有完全重建。因此这些不是准确 RPC 数、可以省掉的
总次数、预测延迟或反事实 GPU-s。`exact_rpc_count` 和 `avoided_latency_ms`
均保持 null。这个信号支持进行一次新验证，不证明最终收益。

数据：`paper_results/ieee_tc/p2_backend/d316_route_copy_revisits/analysis1/`。
保存全部 2,357 条机会索引及输入/输出 SHA，不覆盖 D315 的原始审计。

## 生命周期依据与不变量

实际 `IEEEBackendGPUReferences._refresh` 在 slot、注册集合和 pin 集合变化
时推进全局 epoch；acquire/release 也会推进它。因此 epoch 不是一个目标
GPU 副本的寿命。原生 `_gpu_source_incarnations` 仅在新确认副本时创建，
原生移除回调先撤销它；同 copy 的另一次 pin 可以更新确认时间而不改变
copy ID。已有 `_load_and_acquire` 明确允许带 exact copy witness 的旧 epoch，
但拒绝未来 epoch、owner/内容/来源不符，并在同一线程内确认和取得引用。
本轮没有改这些原生边界。

核对本机 vLLM 0.30 与[官方 LoRA manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)：
其区分复用已有对象、加载新对象与 LRU 驱逐，实际操作处在单线程 core。
[官方 core](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
的 utility 路径也不能因调用方使用 async 就当成无代价后台工作。
这支持减少不必要的重复观察，但不直接证明 Prime 的新谓词或性能；正确性
仍依赖本项目 copy owner/epoch/lease 合同和针对性测试。vLLM skill 用于核查
后端执行边界，不采用其通用吞吐目标、并发建议或未经本机验证的加速比。

唯一服务修改位于 `_ieee_publish_identity_frontier`：

1. 同一完整身份的旧行为保留。
2. 新原生事件的 epoch 严格大于测量 epoch，且 owner/clock 相同；复用既有
   scope 检查，要求目标仍注册、已确认 GPU-ready，copy ID、slot、name、
   path、rank 和完整的目标测量均匹配，才保留原记录。
3. 保留的是**原对象、原 epoch、原时间戳、原测量**，不把它改写成刚测得，
   不用旧快照覆盖 InstanceSlot 的新观察，也不把它当物理 admission 快照。
4. HOST-only 测量仍随全局 epoch 失效，因为 alias/allocator 状态可能改变。
   target 改变、缺失、未知、未确认、owner/clock 改变或未来测量都不能保留。
5. 选中后 exact-copy 原子核验、引用保护、冲突重选、取消/丢回复收尾完全不变。
   同副本其他合法状态、队列、Q 排序、planner、admission 和扩缩容不改变。
6. 新增累计 `identity_frontier_retained_gpu_scopes` 只用于证明分支触发，不能
   当作独立 RPC 数或性能改善值。

不引入 TTL、经验等待、重试兜底、未来热点、另一个预测器或手工新系数。
IEEE 九式、D/T/O 更新、生成合同及真实传输计量都保持不变。

## 验证现状与下一步边界

未修改服务代码时，生产顺序的定向测试因原 scope 被删除而失败；失败日志
原样保留。修改后 419 项 route/原生引用/生命周期/footprint/identity/分析
回归通过，耗时 3.567 秒。包括不伪造新时间戳、目标变化、GPU→HOST、未知/
未确认、未来 epoch、取消、原生 copy 保护及历史分析口径。另有 27 项离线
汇总与原 route 测试先通过。这些不是 GPU 性能或数值资格。

下一项只做一次既有 100 请求索引的真实 7B 验证，另开 D317 执行键：

- 继承 D315 的 1024/cap8/slot8/rank16/.92、D298 已测 profiles、全部控制、
  60 秒通知、原 500 pool 和相同真实远端；只改变诊断计数/输出/冷缓存身份。
- 使用既有 runner/analyzer，验证 100/100 成功及原生数量、完整 binding、
  dispatch tier、物理 release 和保护清单；保持 N_correct 未知的边界。
- 累计新分支必须非零，保留所有目标冲突和重选；不能用不执行测量/核验来
  “改善”时间。若未触发或出现来源、计量或安全错误，不进入完整回放，先
  用日志判因；不能反复重跑同短测挑出胜者。
- 短测通过只能支持执行安全与触发。随后对 unchanged candidate 做一次
  完整 4,000 W0，与已封存 D315 比较 waiting、TTFT/TPOT、U、全部失败与
  输出差异。短测不决定 G1/G2 或全程性能，也不把历史不同配置当因果消融。
- 未证明整体收益就不关闭 7B，不开始 3B/外部基线，不再扩大本局部优化。

本轮没有新 GPU 运行。CPU 全部限制在 3/4 GiB、swap=0、CPU 2,3,26,27；
原始测试/分析日志位于 `results/ieee_tc/p2_backend_qualification/d316_20261007/`。
完成资源和保护核查、备份后再按完整准入门槛启动 D317；不是由 CPU 通过
直接宣称性能改进。最终 warm/Resident、数值资格和旧 Prime 新指标对照仍开放。
