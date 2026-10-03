# D193：规划快照最小依赖研究

状态：CPU-only 研究完成；支持进行单一生产候选的正确性验证，尚无整系统加速结论。
未改服务代码，未启动 GPU 或真实远端服务。

## 问题与唯一假设

主线仍为 Prime 7B 实际达标，然后才是 3B 和外部基线。D190 的 4000 请求
完整回放有 2024 次 owned planning，累计 worker CPU 1123.445 秒；同一暂定
共同参考下联合时延达成 84.375%，不能宣布满足 G1/G2。D187 的独立诊断
记录显示 source-view 冻结及 owned inputs 构造出现在规划 CPU 栈中；采样
次数不是 CPU 百分比，D187 也不能替代当前全部耗时归因。

唯一假设：初始化后的驻留规划搬运了不消费的 HOST 逐张量描述、staging 图和
allocator 诊断；使用现有 `routing_source_snapshot` 的完整 registered graph
可以减少传递、冻结和复制，但保持输入预算、源身份、选择及执行目标。
这里不是 `routing_wire` 的压缩摘要，也不是 request-scoped 单目标占用。

只研究描述表示，不改变需求、成本、控制频率、资源预算、SLO 或九个公式。
没有缓存旧层级状态，不改变 epoch/capture time，不移除执行时的物理检查。

## 依赖边界

| 内容 | 处理及理由 |
|---|---|
| owner、epoch、clock、完整 source identities、当前 slot mapping | 全保留；确认状态与失效判断 |
| registered HOST allocations、所有 adapter alias edges、distinct union、exclusive bytes | 全保留；不能用单目标占用替代全局容量或回收收益 |
| GPU pool/slot 实际容量、tensor view geometry/dtype | 全保留；profile class、物理布局和初始化绑定仍消费 |
| replacement-protected adapter IDs、staged source identities | 全保留；候选可替换性、原生 owner 不变量 |
| HOST tensor view 详细描述 | 研究不传递；原生 inventory 仍逐张量验证 shape/storage/pinning/alias |
| native staging footprint、allocator report | 研究不随规划传递；其他容量等待、执行及观测入口不变 |

代码依据：`gpu_monitor.py` 的两类 snapshot 共享同一 fresh owner read；既有
`routing_source_snapshot` 输出完整 registered graph，仅省去上述描述。当前
`owned_preparation_inputs`、native victim/objective 和 source parser 使用的
容量图仍保留。`_refresh_ieee_deferred_host_capacity` 确实读取 allocator，不能
替换其入口。规划摘要永远不是执行资源 reservation。

## 历史、外部参照与方法

- D151 改的是 routing 入口；当前 initialized planner 仍请求 full source snapshot。
- D181 是请求作用域 footprint；其全局容量限制在本研究中不放宽。
- D189 是不可变文件描述；D192 是已拒绝的 encoder 原型，本研究不再测试它。
- 复用 D192 的 D169 四个真实既存 source graph、三轮交错测量及受限 CPU wrapper。
- 使用当前 native parser 与 canonical JSON detach；投影开销计入候选时间。
  这是 parser＋一次 source-view 冻结的组件测量，不含原生 inventory 构造、RPC
  等待、整个 planner 或 GPU 服务耗时，不外推总 TTFT 改善。
- 复用现有 planning composition fixture，比较 handoff/residency、普通/受保护
  victim 的预算、selection 和 execution objectives。fixture 的额外描述是明确
  标识的测试元数据，不伪称实测物理图。移除字段会改变 source/plan/objective
  hashes；比较只忽略这些已知 provenance hashes，不忽略决策字段。
- 对真实图分别破坏 union、alias owners、exclusive credit、GPU capacity、slot
  mapping、owner identity 与 complete 标记；full/projected 均须同样拒绝。

vLLM 官方将 Python 调度和数据处理的 CPU 开销列为可能让 GPU 等待的原因，
并强调组件分离；本研究借鉴“减少控制路径不必要工作”，不借用其硬件收益。
[vLLM CPU 性能分析](https://vllm.ai/blog/2024-09-05-perf-update)
当前 v0.30 core client 的独立进程消息路径作为实现参照，并不是 Prime 的
状态传播正确性证明。[固定版本源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core_client.py)
CPU 工作会阻塞 asyncio 的其他任务，但移出或减少工作仍须保持状态所有权。
[Python asyncio 官方说明](https://docs.python.org/3.12/library/asyncio-dev.html#running-blocking-code)

## 结果与后续

4 个真实快照的完整 parsed state 一致；28 项破坏图/身份的检查同样拒绝；
4 个 fixture 情形的预算、选择和执行目标一致，只有预期的来源/计划 hash 改变。
24 条测量（四快照、每变体三轮、每轮五次调用）全部保留。以下为组件轮次
均值，不是独立服务运行，不计算 seed-level CI。

| 快照 | 已注册 adapter | 完整 bytes | 精简 bytes | 完整 ms | 精简 ms | 字节减少 | 时间减少 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 1388807 | 336182 | 116.792563 | 95.728060 | 75.793% | 18.036% |
| 1 | 12 | 2025691 | 444997 | 111.100663 | 52.221676 | 78.032% | 52.996% |
| 2 | 20 | 3301339 | 662970 | 182.087642 | 30.312095 | 79.918% | 83.353% |
| 3 | 24 | 3936172 | 771272 | 200.510064 | 35.570229 | 80.406% | 82.260% |

四组均值下降，但存在明显轮次波动：快照 0 的精简三轮为
226.768、30.618、29.798 ms；快照 1 为 19.269、18.481、118.914 ms。
不删除慢轮次，不据此推断波动原因。native inventory 构造及 RPC 没有在这里
测量；旧 D169 图也不是 D190 当前消息尺寸分布。原始 `includes_projection`
标记表示候选投影开销没有移出计时，full 本身不执行投影。

按照 academic-plotting 与计划 §11，本组件研究使用精确完整表，不制作容易
被误读为服务加速的性能图。结论是支持下一步单一候选，不是生产资格通过。
下一步复用现有完整 registered graph 的 snapshot 入口，仅替换 initialized
planner 的观察类型；其他 allocator/staging 消费入口不动。验证 fresh owner、
共享容量、保护/替换、取消和执行重检查后再跑一个普通 7B Full。
7B 的数值正确性、共同参考、旧/新 Prime 新指标对照和 G1/G2 仍未闭合，
不能推进 3B 或外部 baseline。

唯一 probe 退出 0：38.18 秒、峰值 RSS 1181496 KiB；实际资源域
`a0efe2a5f8b64cd185733befea55753b`，session 89380 已结束。
memory events/swap 全为 0，scope 已移除；147 项旧产物及 Plan/V1 校验通过。
Probe SHA：`71e1b074762109b95c4d23382f41859883ae3198488a7dc68e520c175be3e667`。

原始资料位于 `results/ieee_tc/p2_backend_qualification/d193_20261003/`。
新文件不覆盖历史结果。一次性 D78/D80 发布缓存不重建。
磁盘低于新重型任务 150 GiB 门槛；CPU study 使用 3/4 GiB、swap=0、
CPU 2,3,26,27，GPU/构建需另行经审计回收及新安全核查。
