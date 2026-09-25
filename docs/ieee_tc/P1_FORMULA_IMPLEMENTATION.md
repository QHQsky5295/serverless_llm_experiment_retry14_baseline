# P1 — IEEE 公式与实现合同（执行中）

本表不是性能结果，也不表示 Full 已完成 IEEE 对齐。主比较必须等所有关键
合同关闭；不得把旧代码的指标移植到新设计上。论文源文件未改动。

本文件前面的首次审计表保留历史发现；逐项最新进展见 D1–D12。测试通过不等于
真实模型资格，更不等于全部九式已在 Full 闭环接通。

## 规范来源

- IEEE PDF：`/home/qhq/storage_audit_20260915/PrimeLoRA_IEEE.pdf`，
  SHA256 `22d4272e78255f393d4d4f071670e7ca43f43771805784af75d9123a4e3912d1`。
- 对应用户提供的 LaTeX 附件：`0b113e57-b363-40fc-98e9-89336e793d16/pasted-text.txt`，
  SHA256 `35da2d89941452df2959b022aedddd4a984e45b8c15efb9951b5295deb198b90`。
- PDF 第 3–5 页和 LaTeX 设计部分核对：九式及所附观测、预算、引用语义一致。
- 首轮代码审计基于 `040296e6ee424a3f9ff37c40ca39da7ee4b8432c`。

## 九式逐项核查

| 论文规范语义 | 当前实现证据与差异 | 关闭条件 |
|---|---|---|
| (1) 最快有效 tier；GPU 必须可执行，保留较慢有效副本、引用和原子失效 | `instance_pool.py::mark_adapter_tier` 维护单一最快层提示，清空其余提示；`ExperimentStack` 另存共享路径。不能仅凭此提示证明物理副本和引用合同完整 | 后端 slot/epoch 与 registry 绑定；驱逐、取消、共享副本和 transfer 引用测试 |
| (2) 按 admission-time observation class 的 D+T+O、EWMA 和初始化 profile | `ObservedRequestCost.record` 是累计平均；类别主要为 tier/backbone，缺 IEEE 完整分箱。runner 在完成时记录 resolved `cache_tier`、I/O 和 runtime TTFT，不能直接当作 admission→acquisition→first→last 的三个区间 | 固定 class/profile；原生阶段事件；EWMA；GPU-hit D=0；不把完成通知延迟计入 O |
| (3) 可行集内按 `(floor(S/delta), Q, replica_id)`；无可行副本则排队 | `Router._routing_key` 还有 handoff 前缀、occupancy 项、未分桶成本及 created_at；capacity 是排序惩罚而非严格空集。不能当作式 (3) | 不可行候选排除、同快照、稳定 ID；取消前缀；选择+reservation 冲突重试 |
| (4) `h = 窗口到达占比`，`F = h * max(d_source-d_target,0)` | **h 已开始修正，见下文**。旧 planner 仍有 0.4/0.3/0.2/0.1 组合评分；forwarding utility 还乘压力；不能声称整个收益目标已经一致 | 所有计划使用同 epoch 的 h、实测表示类成本和真实 footprint；不得用静态 prior 或另一个 utility 混代 F |
| (5) 剩余层预算、正收益密度扫描、单 adapter 单最终目标、物理执行时预约 | `_select_scaleup_gpu_candidates` 仍含 preferred frontier、热度/新近性排序，并非跨层密度扫描 | 统一 handoff 候选/预算快照、staging 记账、唯一目标、transfer 去重 |
| (6) GPU→HOST→NVMe 条件性单层最大总 F | `PreloadingCandidate.priority_score` 不是 F；DP 用 `priority_score * rounded_weight`。启动路径先 NVMe 后 HOST，不能用启动搬运顺序证明正式规划顺序 | 候选价值=F；规划顺序与物理搬运顺序分开；排除更快层已选目标 |
| (7) ceil(m/MiB)、floor(B/MiB)，滚动 objective 行、受限 traceback；超限字节密度 scan | 旧 DP `cap_units=max(1,B//MiB)` 会让亚 MiB 剩余容量得到一单位；objective 全二维；大小门槛没有直接限制 `n*B` | 小规模穷举、亚 MiB/零预算/大表触发测试；原字节约束、确定性 tie-break |
| (8) `(reuse + max(avail-KV,0))*(1-max(batch,load))` | `_effective_capacity_mb` 没有独立 reuse；保留量取 KV 与 working-set gap 最大值；pressure 混入内存占用，另有 `utility > pressure` 门槛；KV 来自 token EWMA，并非每请求 uncovered block | 后端 iteration/KV block/adapter pool 观测；按式 (8) 原子候选及替换 admission；不把未知 KV 填零 |
| (9) physical used+reserved≤limit；池容量与物理扩容不重复计 | 有 `TierCapacity.reserved_bytes` 与物理监视，但共享池、并发 reservation 和实际 backend slot 的一致性尚未完成审计 | 并发、重复 transfer、取消、替换回滚、workspace、后端失效和释放压力测试 |

## P1-D1：到达需求分布修正

### 历史与可证伪假设

历史 `HotnessTracker` 来自 `b68eaeb` / `cd30b1b`，在本轮前没有实现 IEEE
窗口定义。旧路径在 LoRA resolve 后调用 `record_access`；繁忙或远端慢的请求
会更迟进入需求统计。具体假设：**h 不仅数值不归一，还滞后于真实 ingress，
因此新需求可能在 handoff 规划时被遗漏。** 这不是“改一个参数使 Prime 赢”。

无 GPU 的确定性重现（窗口 10 秒）：

| 输入与观测时刻 | 论文 h | 旧 get_hotness | 旧 registry mirror / top-k |
|---|---|---|---|
| t=100：a、b 各一次 | a=.5，b=.5 | a=.2，b=.2 | a=1，b=1 |
| t=111：没有新到达 | a=0，b=0 | a=.2，b=.2 | top-k 仍返回 a、b |

旧 getter 读的是单 adapter 的累计 EWMA，registry 又把窗口比例乘二。
旧 deque 最多 5,000 条，还会在高到达率下把时间窗口暗改为数量窗口。

### 修改边界

- 沿用 `HotnessTracker`、`ExperimentStack` 与现有 runner，不增加执行框架。
- 单调时钟精确维护 `(t-W,t]`，Counter 与到期队列提供摊销 O(1) 更新。
- 从已经到达的请求更新；排队/远端获取/推理完成不再决定需求观测时点。
- resolve 的统计仍保留，但不再次增加到达计数。
- `get_hotness` 不与 registry/static prior 取 max；查询和空闲时正确过期。
- planner 可从绑定的 provider 取得不可变 epoch snapshot；后续收益函数、
  filter 和 tier 规划仍须继续对齐，不能把完成 h 修正说成 (4)–(7) 全部完成。
- 不预读未来 trace，不新增权重或 workload；历史结果身份和文件不变。

### 参考与适用范围

[dLoRA 原文 §5](https://www.usenix.org/system/files/osdi24-wu-bingyang.pdf)
区分按工作负载进行的分发与对动态失衡的反应。它支持审视“实际需求在哪里、
状态何时更新”这一系统问题，但并未替本项目证明 `h*Delta d` 目标或窗口长度；
这里的精确定义来自 IEEE 稿件本身，不能冒称照搬 dLoRA 的公式。

### 验证与未验证

`tests/test_ieee_tc_demand.py`：九项无 GPU 合同测试通过，覆盖归一化、空窗、
边界、6,000 次到达不截断、不可变快照、stale prior、resolve 不重复计数、
arrival hook 先于 admission，以及输入/时钟校验。

完整 smoke 回归：288 项原有测试＋9 项需求测试，共 297 项通过、无 skip。
首轮五个 synthetic stack 缺少 provider 的 fixture 错误已记录；补充测试依赖，
没有给生产路径添加静态热度兜底。尚无真实模型收益比较，
不把这项修正标注为降低了 TTFT、GPU-s 或提高了 SLO。

## 下一步（主线 P1，不进行性能筛选）

1. 固定 demand/source/footprint 的 planning epoch，修正 F、DP 和多层目标。
2. 建立 admission-time class 与 D/T/O 原生事件，按式 (3) 实现选择与预约。
3. 接后端 iteration/KV/pool 观测后修正式 (8)/(9)，不可先用猜测数值填充。
4. 其间完成外置监控/实际 worker containment；通过后才进行 P2 模型资格。

## P1-D2：收益规划计算与容量边界（已测试，实测输入接入待完成）

历史 DP 的最近实质修改在 `9e49932`：从字节状态改为 MiB，但保留
`max(1, capacity // MiB)`。在本次修改前直接调用实际旧函数，得到：

| 检查项 | 旧实现实际输出 | 论文约束 / 新计算 |
|---|---:|---:|
| 剩余预算 | 524,288 bytes | 相同 |
| 单工件大小 | 786,432 bytes | 相同 |
| 被选择总字节 | 786,432 bytes | 0 bytes |
| 是否越预算 | 是 | 否 |

这是确定性容量违反，不是性能观测。不能靠执行端事后拒绝来把规划器称为正确。

沿用现有 `PreloadingPlanner` 与 `KnapsackItem`：

- `PreparationCandidate` 保存不可变的 source/target、footprint、h 与准备成本；
  `benefit_ms=h*max(d_source-d_target,0)`，`density=benefit/bytes`。
- `select_ieee_insertions` 处理 GPU→HOST→NVMe，排除更快层已选择目标。
- `select_ieee_handoff` 对正收益 pair 做密度扫描，按 adapter ID/tier 稳定决胜，
  每 adapter 只有一个最终目标。
- 两者要求三个层的显式剩余预算；同一 adapter 的矛盾需求/源状态或重复目标
  会报错，不自动合并猜测。执行时引用与 reservation 仍由资源 owner 负责。
- DP 使用 ceil(weight/MiB)、floor(capacity/MiB)、单个 packed objective 行和
  byte traceback；不再分配 n 个 Python-float objective 行。
- `max_dp_buffer_bytes` 是算表资源边界，默认 16 MiB，可在冻结前由 operator
  配置；不是优化性能排名的参数，也不是整个 planner 的 RSS 上限。
- 超过表缓冲预算走论文允许的原字节密度扫描，并记录实际 algorithm。
- 既有历史入口复用修正后的容量 kernel，但**保留显式历史 objective 身份**。
  它不因此成为 IEEE F 路径。TC 入口尚不能用旧 scorer 作为默默兜底。

十二项单元测试通过，包括 80 个八候选问题的所有子集枚举、非整数 MiB
占用、空/零预算、唯一性、稳定 tie-break、强制大表 scan、零收益和非法输入。
这里的数学测试样例不是新增 serving workload，未生成模型或 LoRA 工件。

实现参考：使用 [Python 官方 packed array 语义](https://docs.python.org/3/library/array.html)
降低 objective 表示的内存开销；算法语义来自 IEEE 式 (4)–(7)，不作为新算法
贡献。对 [vLLM 0.30.0 KV 管理接口](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/core/kv_cache_manager.py)
的初查确认后续必须对接真实块分配信息，不能把离线候选选择当作物理准入。

当前剩余：测量类/profile 来源、共享 footprint/remaining-budget owner、真实
pending-movement 执行路径的接入。没有接入完成就不放行 Full 正式实验。

## P1-D3：服务观测与严格路由（计算合同通过，原生事件接入待完成）

历史 `instance_pool.py` 的 `cb53f04` / `31a56f3` 路径保留在 legacy policy，
以便解释历史数据；它不是新的 IEEE Full。此次没有用修改后的类重标旧结果。
可证伪假设是：累计均值、完成时重分类、无可行集时仍返回副本，以及额外
handoff 前缀，会使路由行为偏离式 (2)/(3)。修正目标是定义正确，不预设收益。

在原 `instance_pool.py` 中增加并接入 Router 的显式 `ieee_confirmed` policy：

- `ServiceClassBins` 使用冻结的闭上界分箱；输入仅为 prompt、声明输出上限、
  rank、footprint、source representation、接受本请求后的 admitted 数。
- `ServiceCostModel` 按 admission-time class 保存 D/T/O；初始化必须有
  same-model/backend profile 身份，首个在线样本也按 beta 更新该初值。
  缺类报错，不以混合 tier 均值或零成本补齐。新副本继承冻结 profile，
  不继承上个正式运行已经学习到的状态。
- `ServiceIntervalObservation` 只接受同一单调时钟域中的 admission、首次
  executable acquisition、首/末 token。GPU-hit 在受保护 admission 时 D=0；
  完成通知不计入 O。每段完成即更新，取消不编造尚未完成的段。
- `ReplicaRoutingSnapshot` 不可变；同一次选择拒绝混合 epoch/request。
  严格排除无请求容量、active-adapter 不可行及未 ready 的副本；空集返回排队。
- 选择键严格为 `floor((D+T+O)/delta)` 后接 admitted、pending load、utilization、
  last dispatch 和 replica ID。不加 handoff 前缀、occupancy 或未分桶成本。
- 纯快照选择用于 A/A / shadow，不更新在线计数或 EWMA；live Router 入口
  记录选择身份，但**尚不等于完成物理 reference/reservation**。

十三项确定性测试通过，包括：同一服务时间桶中较空副本优先、下一个桶
不被负载项越过、GPU-hit D=0、单 token O=0、取消只保留完整段、profile
EWMA、稳定 ID、空可行集、错误快照拒绝、A/A 无副作用和 legacy 前缀不影响
IEEE 选择。它们是机制正确性表，不需要性能图，也没有生成 serving workload。

联网核查 [vLLM 指标设计](https://docs.vllm.ai/en/latest/design/metrics/)
及 [0.30.0 stats 源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/metrics/stats.py)：
其 engine/frontend 边界与本项目 admission 边界并不自动相同，输出事件也未必
逐 token。因此不能把聚合 TTFT 或 SSE chunk 间隔直接充当本论文三个区间。
本次沿用原生时间事件、单调时间差的原则，论文的 class 和公式仍以 IEEE 为准。

仍待关闭：runner 生成已提交的完整快照；后端提供 executable acquisition 事件；
跨进程时钟域确认；真实 profile 标定；原子选择/预约及发生冲突后的重新选择。
在这些条件满足前，不启用 `ieee_confirmed` 进行 Full 性能实验，不回退旧 policy。

## P1-D4：KV 需求、复用容量和准入公式（计算通过，原生 owner 接入待完成）

历史 `resource_coordinator.py` 的 `67a740c` 路径仍以 MB 近似 KV，并将
显存占用、working-set gap 与实际加载压力混合；另有 `utility > pressure`
条件。它与 IEEE 式 (8) 不同，不能通过阈值调参消除该语义差异。
本次扩展原协调器的 `evaluate_ieee_gpu_admission`，历史入口明确标为 legacy，
不悄悄改名，也不在原生信息缺失时回退历史公式。

| 论文规范语义 | 本次实现与检查 | 尚待真实路径提供的事实 |
|---|---|---|
| 完成请求的输入长度桶均值，窗口 `(t-W,t]` | 精确 sum/count、边界到期；空桶使用有身份的同模型 profile；重复 completion 拒绝 | 原生成功完成事件、冻结分桶/profile |
| 每请求剩余生成量与未处理 prompt，扣已预留空位，逐请求 ceil 到 block，最后扣 unreserved free blocks | 只接收不可变 admitted 状态；缺 bucket/KV layout 报错；不预读实际未来生成长度 | scheduler 的 admitted/processed/generated/allocated block 状态 |
| `p=max(batch,load)` | 当前 iteration tokens/token budget 与活动 transfer/load limit，分别 clip；没有内存利用率阈值或额外 utility | 同 epoch iteration/transfer 事件 |
| `E=(reuse+max(avail-KV,0))*(1-p)` | physical used 包含整个已分配 pool 一次；pool 的 occupied/reserved 不作为额外物理 bytes 重复扣除 | 原生 LoRA tensor storage、slot assignments、workspace 和预算 |
| 物理安全、兼容 slot | additional storage＋workspace 必须不超 unreserved headroom；CapacityOnly 仍保留这些检查 | slot 形状兼容、实际分配/释放完成事件 |
| 替换先接受再回收 | 计算只接受显式 after-victim-release proposal；不驱逐、不发布 GPU-ready | owner 原子 epoch 校验、victim/reference/slot/pool/physical reservation；取消释放 |

确定性反例：两个请求分别缺 11 和 2 个 token，block 大小 16、还有一个
unreserved free block。正确结果是 `ceil(11/16)+ceil(2/16)-1=1` 个新增 block；
先合并再取整会误算为零。预分配池例：physical used=800、limit=1000、
pool=400、occupied=100，idle 时 `avail=200,reuse=300,E=500`；复用 100 bytes
不增加 physical used。数值仅是数学测试，不是 GPU 测量或新增服务负载。

16 项检查通过，覆盖窗口不截断、单调时钟、快照不变、模型/profile 身份、
逐请求取整、declared limit、未完成最少一个预测 token、已预留容量、物理
边界和决策计数。接口返回可审计 decision，但 **True 不是 reservation**。
尚未完成实际 owner 事务，因此不能放行 Full 或用其说明 admission 已降低 TPOT。

联网核对 vLLM 0.30.0 的
[worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)、
[model manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)、
[KV manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/core/kv_cache_manager.py)
及 [scheduler](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/core/sched/scheduler.py)：
CPU adapter cache、GPU active slots 与 KV block pool 是不同状态，不能将
`list_adapters()`、文件大小或全局 GPU 使用率直接代入以上字段。新版后端
正在独立安装，未修改其安装文件；原生事件和 allocation owner 接入仍在主线上。

## P1-D5：原生 token 计量已接入原 engine/RPC/controller（模型资格待完成）

历史 runner 的 `_derive_vllm_latency_metrics` 使用完成通知求 decode，controller
又用 service E2E 减 TTFT 重算 TPOT；两处都可能把最后 token 之后的通知开销
计入解码。新版 V1 的字段也不同：`arrival_time` 是墙上时钟，`*_ts` 是
engine-core 单调时钟。不能直接相减，更不能缺字段时默默使用文本 chunk。

本次在原 metrics collector 与 runner 中接入 opt-in
`model.timing_contract=ieee_tc_native_v1`，旧实验身份仍为 `legacy`：

- 后端开启原生 stats；使用 queue/scheduled/first/last 的单调时间。
- 原生累积 token ID 的数量、前缀和 metrics count 必须一致；尚无完整 terminal
  或缺少原生字段就失败，不用 expected count、重分词或完成时间补齐。
- 每次有新 token 时保存 scalar 时间；无新 token 的完成通知不延后 last。
- 记录本机 boot/time-namespace 身份，并验证 worker/controller 边界顺序。
  跨机时间不直接拼接；实际 engine-core worker 的相同 namespace 仍需 launch
  census 证明，当前尚未宣称整个部署已通过时钟资格。
- controller 的 TTFT、TPOT 使用原生首末 token；单 token TPOT 在新请求结果和
  native payload 中为 null。历史数值 RPC tuple 保留 0 作为未观测占位，但
  `native_tpot_ms=null`、`tpot_observed=false` 决定新统计，不能把占位算成优值。
- `generate_prepared` 原先没有接受实际调用端传入的 `return_timing`；已修正
  为与原 `generate`、子进程入口一致，保留历史默认行为。
- native 合同禁止 engine 内递归隐藏重试；失败交给可记录 attempt/生命周期的
  外层 owner。此处未禁用后端原生 KV preemption/rescheduling。

| 测试问题 | 结果与边界 |
|---|---|
| 墙上时钟与单调时间混用 | 构造 wall arrival=1.8e9、native dispatch=100；只计算后者同域差 |
| decode 与完成通知分离 | first=100.5、last=101.1、通知=102：TPOT=300 ms（3 tokens），通知=900 ms，不能把通知加进 decode |
| 空 terminal 更新 | 保存最后一次新增 token 的时间，拒绝让 terminal 元数据延长 decode |
| 单 token | TPOT=null，RPC 保留 null；多 token 合法零值不因“必须 >0”被删除 |
| 原执行入口 | 实际 `generate_prepared`、RPC normalization 和 `_exec_request` 方法用 deterministic fake engine 验证，不是只测另一个参考实现 |
| 阶段加和 | admission→engine entry→queue→schedule→first→last→worker complete→controller complete，误差低于 1 ms |

13 个新增无 GPU 测试通过。第一次完整回归有一个旧 synthetic runner fixture
缺少 `model_cfg`，已在 fixture 显式指定 legacy 合同；没有给生产路径增加
缺失配置/时间的兜底。

这些是实测接线的正确性测试，**不是 13 个推理实验**。尚待真实后端检查
stats 的可变对象是否存在滞后、实际 clock namespace、完整调用开销与 token
合同。外置回放的计划到达/提交事件及受保护 executable acquisition 仍未接入；
pre-engine span 包含 resolve/transport，不能冒充纯 D。这些缺口关闭前不放行
C5、Full 或主比较。旧聚合器也不能据此被整体标记为 TC-qualified。

依据：vLLM 0.30.0
[RequestStateStats](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/metrics/stats.py)
和 [output processor](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/output_processor.py)。
采用其真实时间定义，并非复用 API 中名称相似但起点不同的 TTFT。

## P1-D6：原生 GPU 引用及真实生成入口绑定（已接线，CUDA 资格待完成）

本次假设：**router 的 resident hint 与原生 cache eviction 无共同 owner 时，
“选中时命中”不能保证“使用前仍可执行”。** 依据不是推测：历史入口的
slot hints、native `remove_lora` 及子进程调用分别维护状态；仅测最快 tier
或 `list_adapters()` 无法证明引用在整个 dispatch 路径有效。

联网核对 vLLM 0.30.0 的
[worker manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)、
[model manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)
和 [LRU cache](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/utils/cache.py)：
CPU/GPU 缓存分开；原生 pin 可阻止 LRU 驱逐，但显式 remove 仍能删除 pinned
项。因而“只加 pin 就完成 confirmed propagation”也不成立。实现采用原生
cache 的引用保护，并将显式卸载及真实 generation 绑定纳入同一所有权合同。

| 论文规范语义 | 当前实现证据 | 仍未关闭 |
|---|---|---|
| snapshot→选择→取得引用，冲突重选 | `IEEEBackendGPUReferences` 在 native worker 持有 incarnation/epoch；旧 epoch、CPU-only、已失效返回拒绝，不加载后假称原命中 | controller 全候选同 epoch 快照、active-request reservation 和冲突重选 |
| 引用有效期间不可回收 | native CPU/GPU LRU pin；多请求计数；explicit unload 走 owner；旁路失效使 owner invalidated | 所有主动准备/替换路径接入，同线程与真实模型压力资格 |
| executable acquisition 完成才发布 | worker 当前 CUDA stream event 完成后返回 acquire 时间；不以 `list_adapters` 代替 GPU | dense LoRA copy/execute 的实机 stream 资格与纯 D 成本更新 |
| 引用与实际请求相同 | existing generate/prepared/proxy 接受同 adapter lease，绑定 native request ID；native terminal 才允许释放 | 完整 controller 租约生命周期、取消的原生 abort 终态接入 |
| 取消、重复通信、释放 | acquire/release 重试不重复增减；不能复用已释放 dispatch ID；尚未终态不假释放；device 异常使 owner 不再可用 | backend 失效时整 worker 回收及生命周期积分 |

scope 明确限定 TP=PP=1 的单 runtime；这是主实验拓扑，不把 TP 多 worker
部分成功说成原子提交。该组件不隐式执行 cold-load、victim selection、KV
reservation 或式 (8) soft admission；回执明确 `request_admission_reserved=false`。
尚不能启用为正式 Full。引用接口默认关闭，不修改已封存运行的身份或数字。

16 项新增无 GPU 测试使用已安装旧后端的真实 LRU 容器、假 slot/device events
及实际 engine/prepared/RPC 方法，覆盖持有期间 LRU/显式驱逐、多引用、旧 epoch、
CPU-only、backend incarnation、已有 pin、CUDA 异常回滚、旁路删除、request-ID
绑定、未终态占用及传输拒绝。完整功能回归 **372 项通过**，无失败/skip；独立
安全回归 **19 项通过**。一次测试模块临时导入另一个 TestCase 导致其 6 项
被额外发现，已改为 module 引用；372 的最终计数不含该重复。

图表选择为本节合同状态表，而非性能图。真实 CUDA event、模型输出、引用开销
与性能影响尚未测量；没有声称本次检查减少 TTFT、降低 GPU-s 或提高联合 SLO。

## P1-D7：原生 scheduler/KV 观测（已接线，模型及事务资格待完成）

历史 D4 已证明 MB/working-set 近似不是论文的 KV 公式。本次进一步核查
0.30.0 的原生异步执行：worker 的 LoRA/显存 RPC 看不到 engine-core 持有的
请求队列和 block allocator；将两份独立采样直接拼成“同 epoch admission”
仍不正确。因此先接入真正的 scheduler owner，并明确它只提供观测。

沿用原 runner、子进程 worker/RPC，opt-in 配置为
`model.ieee_scheduler_observation=true`，必须提供冻结的
`model.ieee_input_upper_bounds`。以官方 `scheduler_cls` 扩展点继承
`AsyncScheduler`，保留其 schedule、preemption、async execution 和 allocation
实现，不通过退回同步调度取得易于解释但不同的性能路径。版本严格限定 0.30.0。

| 论文规范语义 | 当前实现证据 | 尚待资格 |
|---|---|---|
| iteration pressure，由调度事件更新，idle 为零 | 记录原生 SchedulerOutput 身份；多轮在途时以最新未收尾调度轮的 token 数计 pressure；较老完成不将其清零 | 真实 engine-core/worker 时间线、观测开销 |
| 实际完成与未处理 prompt | 从 computed 扣除仍属于当前 assignment 的在途 tokens；异步抢占的 stale share 单列，不误判负完成数 | 两模型 prefill/decode、抢占/恢复/取消实机检查 |
| 已预留未用位置、unreserved free blocks | 当前请求的非空有效 block assignment；free 来自原生 block pool，不用“总 block－逐请求求和”代替；共享 prefix 不重复扣除 | 原生 pool/实际 tensor 的对应与高压验证 |
| bytes/block | 一组 uniform FullAttentionSpec、TP/PP/CP=1；实际 page bytes×层数与单一 allocation size 相互核验；未知/混合/alias 布局拒绝 | 实际 7B/3B FP16 layout |
| 同一时间域和来源 | 在 owner thread 读取，记录 incarnation、PID、monotonic clock、时点；前端核验 clock；跨进程 RPC 不吞错误 | 实际进程 census/clock qualification |
| 原子 admission 与物理可行性 | 此接口不声称完成：`admission_reservation=false`；worker 分配和 scheduler 观察仍须进入同一事务 | 原子 owner、待提交请求的 reservation、physical workspace 与取消终态 |

抢占注意：原生 `num_computed_tokens` 被重置，但旧
`num_in_flight_tokens` 仍待返回，`num_stale_output_tokens` 标明其中不属于
当前 KV assignment 的部分。当前已完成位置为
`computed - (in_flight - stale)`。旧 block 的延迟释放仍由 native free pool
计量，不能先算为空闲。样本同时保留 generated、stale、preemptions 和
尚需重算的历史生成位置。后者**不偷偷加到论文的 unprocessed prompt**；
论文预测不是硬物理容量保证，实际分配仍须由后端约束，后续 S9 分析此近似边界。

private utility bridge 仅增加一个有冲突检查的 engine-core 观测方法；不覆写
vLLM 安装文件，也不把该私有协议宣称为稳定公共 API。未启用时保持旧路径。
原生队列未覆盖 controller 尚未提交的 admitted reservations，回执明确
`native_unfinished_requests_only`；后续 owner 整合不能遗漏或双计这些请求。

16 项无 GPU 检查覆盖旧轮完成、新轮压力、空调度、FIFO、线程身份、waiting/
finished/shared-prefix、在途 preemption/resume、未知布局、原生输出保留、
版本/命名冲突、实际构造参数、guarded launch、时钟与 RPC。它们使用假的
native scheduler/model 对象，不是实际后端或性能资格。本步骤交付此状态表，
不生成具有误导性的性能图。实际模型与完整回放仍待后端安装和资格完成。

依据：官方 0.30.0 的
[Request 状态](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/request.py)、
[scheduler](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/core/sched/scheduler.py)、
[AsyncScheduler](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/core/sched/async_scheduler.py)、
[KV layout](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/core/kv_cache_utils.py)、
[engine core](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
及其 [utility client](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core_client.py)。

## P1-D8：固定工作量的错误路径与跨进程输入（代码合同通过，共同比较待资格）

历史 `31a56f3` 已加入 fixed-output，但继续检查其实际调用链发现：

1. 外层禁止 fallback，并没有消除 `_prepare_vllm_prompt` 内部捕获分词错误后
   按字符/输入 hint 继续的路径。
2. 源 expected-output 缺失时还可能从另一个字段或默认 max-token 补值；
   prompt guard 和 model cap 又可能减少目标，变成“正确完成较少工作”。
3. parent 已准备好的 prompt 经 subprocess `generate_prepared` 后没有携带
   prepared 标记，worker 会重新 guard；decode/re-encode 并非必然幂等。
4. 仅在引用模式检查 LoRA，其他 fixed-output 调用若 LoRA 未启用可能生成基座输出。

修正在原 runner/worker 上完成，不另建 replay framework，不改历史结果：

| 共同协议要求 | 实际变更与证据 | 仍待验证 |
|---|---|---|
| `target=min(source_expected,cap)` | 源值/cap 必须为正整数；不能从其他字段补齐；模型/上下文不能偷偷改目标 | 既有完整 trace 的冻结输入审计 |
| canonical prompt | 真实分词、确定 decode/re-encode、非空 token、含特殊 token 的上下文检查；不能按字符兜底，退化边界不能编造 token | 最终后端 tokenizer 与所有 baseline 的逐请求 SHA |
| chat rendering 一致 | native fixed 合同显式冻结 `tokenizer_chat_template` 或 `role_lines_v1`；不按某次模板报错自动换格式 | 两模型共同 renderer 配置与 HTTP baseline 接入 |
| 子进程不改变输入 | 原 RPC 携带 prepared 三字段，worker 核验与请求一致后直接使用；不重复 guard | 原生 engine 真正接收到的 token 序列 |
| 正确 adapter | fixed-output 有 adapter/path 时必须构造 native LoRARequest；禁用 LoRA 不能当成功基座输出 | native adapter/weight 身份、正确加载及全池覆盖 |
| 原生输入/输出证据 | 原生 prompt token IDs 必须存在且在 stream 中不变，保存其 SHA；保留已有 native output SHA/count | 跨系统特殊 token 一致性，不只比较字符串 hash |

输入参考了本机 baseline replay 的既有 canonical decode/re-encode 语义，以及
官方 vLLM 0.30.0 [TokenizeParams](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/renderers/params.py)
的 context/special-token 检查。S-LoRA 的
[HTTP manager](https://raw.githubusercontent.com/S-LoRA/S-LoRA/main/slora/server/httpserver/manager.py)
和 [router](https://raw.githubusercontent.com/S-LoRA/S-LoRA/main/slora/server/router/manager.py)
分开传递 prompt IDs、adapter 与采样参数；这些身份不能由文本重分词或预期
输出数替代。最终仍须核验本地冻结 commit，而非用 upstream main 推断已运行行为。

新增 8 项无 GPU 合同测试，完整功能回归 434 项通过，无失败/skip。第一轮
37 项定向测试有一个旧 fixture 关闭 LoRA，因而更早被新基座替代检查拒绝；
已将该 fixture 显式设为有 LoRA、无引用，继续检验其原定引用缺失问题。
没有删除错误检查来使测试通过。

以上不代表已完成 C5 matched-output 性能实验。baseline replay 文件本身含有
用户未提交修改，本次没有覆写；其残留错误路径与最终共同输入协议须在 baseline
资格时通过独立可审计入口解决。不能因为 Prime 的单元测试通过就宣布全体系统
可比，也不能把旧 natural-output 结果重新标记为 fixed-output。

## P1-D9：原生按需加载与执行引用的同线程事务（非主动 admission）

本步骤继续 P1，不改变 IEEE 公式、不引入新调度策略。D6 的 hit-only `acquire`
只会保护已有 GPU 副本；若先经另一个 RPC 加载，再查询/保护，两个操作之间的
原生 LRU 更新仍可能改变目标。可证伪假设是：把加载、完成确认和 pin 合成同一个
串行 worker 操作，可消除这一局部间隙，但不会自动完成 controller 的请求准入。

对照本地历史 `cd5b68d`、D6/D4 合同和官方 vLLM 0.30.0 的
[worker loader](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)、
[native manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)
及 [dense slot layout](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/layers/base_linear.py)：
原生路径先校验/读入 CPU 权重，再按原 LRU 移除未 pin 项并激活 GPU；GPU 缓冲
按最大 rank/slot 预分配。因此，CPU 注册、GPU 可执行和权重文件大小不能混为一谈。

| 论文要求 / 风险 | 当前实际入口与验证 | 未完成的边界 |
|---|---|---|
| 加载期间保护已有请求 | `demand_load_and_acquire` 在同一 worker 线程调用原生 loader，随后完成 fence 和 CPU/GPU pin；保留原生 LRU | CUDA stream/thread 实机资格 |
| 满池不能驱逐正在用的副本 | 原生 GPU 或 CPU 全部被 pin 时返回容量冲突；未开始加载、未改 victim | controller 排队/重新选择与取消收尾 |
| 冷路径不能冒充原始 GPU hit | 回执分别记录 `gpu_resident_before_load`、`cpu_registered_before_load`、是否调用 loader 和加载至引用耗时 | reserve 后、resolve 前的全链路 dispatch 字段绑定 |
| 身份和重试不能替换权重 | 同一整数 ID 的 name/path 在 worker 生命周期中固定；拒绝未受管理的旧 cache ID、跨 source 的 begin-use、已释放 lease 重用；重复同事务不再加载 | 全 500 adapter 内容 SHA / native 输出资格 |
| 原生加载不能静默生成基座结果 | 禁用 load-in-place；加载后必须至少匹配一个可执行 LoRA module；异常不发布 ready，owner 失效 | 全 target-module/权重语义正确性，非仅非空模块 |
| footprint 是实际槽位占用 | 检查 dense tensor 完整 contiguous storage、offset、第一维 slot 数后才提供 `slot_capacity_bytes`；包含 padding，物理 pool 只计一次 | proactive proposal 与物理 owner 预算、workspace 的连接 |

同一池的空槽位字节仅表示**已分配池内可重新指派的容量**，不是新得到的 CUDA
空闲字节。任意 partial view / 非 contiguous / 非 slot-first 表示可以保留为诊断
inventory，但不能用总存储除 slot 数充当 admission footprint；新增按需路径拒绝
这种未经核验的表示，不回退到 rank 或 checkpoint-size 估计。

新增 13 项 CPU 合同检查（11 项原生缓存事务、2 项实际槽位布局），完整回归
447 项通过；独立 safety/census/replay 44 项通过。使用真实旧环境 LRU 类型和
假的加载/张量对象，未进行模型推理，不能宣称 vLLM 0.30 实机已兼容或性能提升。
147 项历史 seal 全部未变，四 GPU 仍空闲。以本表交付本步骤，无虚构性能曲线。

`request_admission_reserved=False` 和 `proactive_admission_evaluated=False` 明确保留：
本操作是请求驱动的原生加载，不执行 E(t)、不授予 KV/request slot、HOST 字节或
慢层级引用，也不代替 planner 的 victim/预算事务。全系统 Full 尚未连通此路径。
下一步是 controller 原子 request/adapter reservation 与原生 owner 的连接，之后
进行实际模型/时钟/stream/worker 资格；不得从单元测试直接跳到正式性能结论。

## P1-D10：请求预留从接纳到终态的完整生命周期

检查 D9 checkpoint `d22721b` 的实际 `_exec_request` 路径发现：请求和 adapter
计数在 resolve 前增加，而原 `try/finally` 只覆盖后续推理。制品获取失败、获取时
取消或预留后的记录异常，都可能使容量永远不归还。先以实际 runner 方法和假的
推理对象复现，三个检查均失败：结束后 `active_requests` 仍为 1（期望为 0）。
这不是 GPU 性能实验，但会影响队列可行性和后续请求的等待，必须在比较前解决。

可证伪假设：以成功预留为所有权起点，保留原始 request/adapter 身份，覆盖整个
resolve→generate→结果处理过程，可在未提交推理的失败路径准确归还自身份额；
已提交推理的取消则需要区分客户端终止与原生执行终态，不能直接归还容量。

| 论文 / 测量要求 | 已实现并验证的行为 | 不应外推的内容 |
|---|---|---|
| pending admission 计入 active 可行性 | 原 runner 记录一次 reservation，原 adapter 身份不随 resolve 返回值改变；同 adapter 两请求取消一个时计数 2→1，而非清零 | 不是原生 KV/物理字节 reservation |
| 失败、取消只释放自身所有权 | 最外层生命周期覆盖预留后所有操作；未开始生成时失败/取消会释放；重复 release 幂等，计数下溢报错 | 原生异步取消完成确认仍未接通 |
| batch 负载不漏减、不重复减 | 成功、取消、结果处理错误均经同一 batch 结束入口；不会在成功后又因统计异常减第二次 | 该计数不替代 D7 的原生 iteration token 压力 |
| 正确完成同一 adapter 工作 | fixed-output 拒绝 resolve 将请求变成 backbone 或丢失路径；在调用生成前失败 | 全池 native 权重正确性仍待实机资格 |
| 客户端结束不冒充 GPU 结束 | native 路径仅接受当前请求的原生 terminal 标记；不借用共享 `last_timing`；未确认工作保留计数、停止向该副本派新请求，并写入结果元数据 | draining 不是 abort 完成证明；还需原生收尾/实际 worker 退出确认 |
| 跨进程不改变证据类型 | 原生 timeline 只在 terminal 与计数校验后产生 Boolean；两个 RPC 归一化位置保留 Boolean/null/identity，不把 `1.0` 当作确认 | mock RPC 不是实际进程/时钟资格 |

原生取消语义核对了 vLLM 0.30.0 的
[AsyncLLM.generate/abort](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/async_llm.py)
和 S-LoRA 的 [HTTP manager abort](https://raw.githubusercontent.com/S-LoRA/S-LoRA/main/slora/server/httpserver/manager.py)。
取消信号的发送或前端 request map 删除，不能单独证明 worker 已停止访问 GPU 数据。
本次不增加假定等待时间来替代原生确认，也不把未知执行状态改成空闲以继续实验。

新增 13 项 reservation 检查和 1 项 typed-RPC 检查；完整功能回归 **461 项通过**，
零失败、错误、skip。测试覆盖真实 runner 方法，但推理对象是假的，GPU 没有执行
模型。中途先有 460 项通过，随后补查发现 Boolean 被通用 numeric conversion 转成
float，最终结果以包含该修正的 461 项为准。本表是本步骤的正确性证据交付。

边界仍明确保留：外层回放异常记录的原 request identity、原生取消后的真正终态
对账、native source snapshot 与 controller 的原子连接、慢层级引用和主动 E(t)
admission 尚待完成。`runtime_request_ownership` 只记录 controller 未结算项，不能
代替物理 GPU 生命周期计量，不能以此宣布 Full 已合格或 G1/G2 已达到。

## P1-D11：失败请求的输入身份与缺失测量

继续沿 D10 的实际失败路径追踪，而非另外建立回放器。`e238f60` 的最外层
`append_raw` 会把两个不同请求的异常写成相同 `request_id="error"`，adapter
和输入身份丢失；单个已启动 task 的 `CancelledError` 又会越过 `Exception`
分支，提前结束整个回放。两个真实 runner / 假推理检查先复现了这两种行为。

输入身份属于 offered trace，应在 task 创建时已知，不能根据是否生成成功决定
是否保留。当前连续回放器在收集已完成 task 时使用其原始 trace 索引，并校验
返回记录的身份；不再在整轮结束时凭一个匿名异常猜测请求。回放前拒绝空/重复
request ID，而不是静默跳过。已预处理的输入计划直接复用，不生成新负载。

| 情况 | 当前记录方式 | 证据含义 |
|---|---|---|
| 两个外层异常 | 各自保留 request、adapter、目标长度、canonical prompt SHA 与计划到达时间 | 仍是两个 offered 失败，不变成匿名结果或减少分母 |
| 单个 task 取消 | 记录 `CancelledError`；其余回放继续 | 客户端 task 终止，不推断 native/GPU 已释放 |
| 整轮取消 / publisher 提前结束 | 保持整轮中止；不制造未到达请求的 timeout 或补全成功列表 | 是截断运行，不能进入完整结果排名 |
| 无 native 输出的错误 | TTFT、TPOT、E2E、输出数、请求美元成本用 null；错误被观察的时间独立保存 | 观察到失败的耗时不是首 token 延迟，更不意味着零 GPU 成本 |
| 已 dispatch 后的 native 执行错误 | 保留已知 dispatch/tier、输入身份和 terminal 是否已观察；未观测指标仍为 null | first-dispatch 失败不能由下一请求的成功补齐 |
| 不完整结果或错误返回身份 | 明确拒绝完整回放输出 | 不用默认值制造可比较点 |

对照官方 [vLLM 0.30 benchmark](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/benchmarks/serve.py)
对成功时延、失败数和 goodput 的分开计算，本项目继续采用自己的已批准输入、
原生 token 与 joint-SLO 合同；不照搬它的文本重分词或零样本默认统计。

新增 9 项检查，最终完整功能回归 **470 项通过**，零失败、错误、skip。中途
469 项回归有一处旧 ingress fixture 返回 `SimpleNamespace` 而非实际结果类型，
已显式提供合法的测试结果；后续 native 错误记录扩展曾因输入摘要在成功分支才
定义而有四项错误，已将这些已知输入事实前移至生成前，最终回归包含该修正。
全部测试不运行模型，表中数据是正确性反例，不是性能收益。

仍需完成整轮安全中止的完整 partial-result journal、统一 offered/arrived/
submitted/native-terminal/good 事件账本，以及物理 GPU owner 对账。旧 aggregate
的全失败默认数值不是新的 G1/G2 合格统计；失败记录 null 也不是成本为零。
本次完成后返回 owner/source/admission 主线，不增加额外性能矩阵或调整 SLO。

## P1-D12：所选请求的原生引用连接与通信取消所有权

本步检查 `91c9feb` 的实际 runner：底层已有 D6/D9 的引用接口，但
`_exec_request` 没有取得或传入 `gpu_reference`。因此接口测试通过不能证明
请求真的受到了保护。三个修改前检查分别复现未取得引用、未释放引用，以及
取消路径根本未到达原生 acquisition 的情况；两项断言失败、一项等待超时。

可证伪假设：把 selected-request reservation、原生 load/reference、generation
和 terminal/release 绑定为同一请求生命周期，可以保护实际执行期间的副本；
丢失回执时必须保留未知所有权，不能把一次客户端异常当作物理释放。

对照官方 [vLLM 0.30.0 loader](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
与 [AsyncLLM](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/async_llm.py)：
沿用原生加载和缓存，不把 CPU 注册变成 GPU-ready。原生取消入口与实际 worker
执行仍是不同层次；本步骤不以调用 abort 代替 GPU 终态证明。

| 实际路径 / 风险 | 当前实现与验证 | 仍未完成的边界 |
|---|---|---|
| 执行前没有取得原生引用 | 在所选副本完成路径解析后，获取 worker/epoch/clock 回执，原生 load-and-acquire 后才传入生成；校验 adapter/name/path/lease | 这不是 reserve 后、resolve 前的 dispatch 快照；共同 routing/source epoch 尚待连接 |
| GPU hit 与本次加载混淆 | 保留 native 原始 GPU/CPU source 状态和 load 回执；不再在推理成功后凭旧规则补记 GPU-ready | HOST/NVMe 所有权和实际源类别仍待接入 |
| acquisition 期间取消或回执损坏 | 发出可变操作前保留 intent；无法确认结果时保留 controller 计数、引用身份并撤回该副本 | 后端 reconciliation/实际 worker 退出确认未完成 |
| 陈旧 epoch | 只使用原生明确拒绝且返回的新 epoch 重查；不重用旧 epoch、不 sleep 猜 ready | 物理容量冲突目前显式失败，尚需连接正式 dispatcher 排队/重选，不是已合格 Full |
| 多请求共享一个 adapter | 两个独立 lease；计数 2→1→0；第一请求释放不撤销另一请求的 pin | 同样需真实 CUDA stream 资格 |
| 其他请求的终态被误用 | terminal 必须匹配 owner、lease、native adapter ID；native owner 还可以拒绝 active-request release | 完整终态账本和安全中止 journal 未完成 |
| 结果处理失败或推理前失败 | 已知未开始生成、或匹配终态已返回时释放；只有原生 release 回执确认后归还 controller 容量 | 释放了引用不等于释放整张 GPU |
| release 回执丢失 | 即使测试中的 worker 实际已释放，controller 仍保留 `release_pending`，不捏造确认 | 后续明确对账，而非按超时清零 |

同一路径又复现两个通信层问题：原 `_rpc` 在丢失响应后再次执行操作；取消
`asyncio.to_thread` 后仍把正在收发的连接归还池。修改前分别观测到两次执行
（期望一次）和一次错误归还（期望零次）。Python 的
[Future 取消语义](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future.cancel)
也明确区分未开始的工作与已经运行的调用。

native 协议现已禁用通信层盲重试；取消连接先 shutdown，退出复用池并撤回该
transport，但不据此宣布 native work 已结束。真实本地 socket-pair 检查确认
阻塞接收被唤醒、连接未被复用；没有启动模型或远端服务。

本步新增 **15 项检查**（12 项 controller/native reference，3 项通信所有权）。
最终完整功能回归 **485 项通过**，独立 safety/census/replay **44 项通过**，均无
失败、错误或 skip。首次完整回归有一项测试 fixture 缺少 `_rpc_channels`；补齐
实际对象字段后通过，没有为生产对象增加容错默认值。147 项历史 seal 未变。
本表是正确性证据，不画性能曲线；仍未测得任何新增 TTFT/GPU-s/SLO 收益。

所有新行为仍属 opt-in native 路径；`confirmed_dispatch_snapshot=False`、
`proactive_admission_evaluated=False` 明确保留。不得把本次 resolve 后的 acquisition
时间强写为论文 GPU-hit 的 D=0，不能替代 admission-time class/区间与 committed
router snapshot。后续回到上述 owner/source/admission 连接，以及安装完成后的
实际模型、时钟、CUDA stream 和 worker 资格；Serverless 仍是 baseline 首项。

## P1-D13：原生副本身份、完成发布与不可变接收视图

`7b75ea5` 的 native owner 已绑定整数编号与 name/path，但没有绑定实际 CPU
LoRAModel 对象。修改前的检查复现：同一整数 ID 对应的对象被替换后，仍可取得
旧来源的引用；第二项检查确认尚无 source snapshot 接口（一项失败、一项错误）。
这是 A3/S2 中 confirmed-state 比较的前置语义问题，不是一个已测性能瓶颈。

本步可证伪假设：只有在实际对象身份一致、设备加载完成、原生回收尚未发生时，
该副本才能进入 confirmed GPU 视图；撤销 GPU 视图不应顺带否认仍有效的 CPU 副本。
对照 [vLLM 0.30.0 LoRA manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)
的分离 CPU/GPU 缓存及回收回调，保留原生 victim selection，在原回调前撤销发布。
[原生 LRU](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/utils/cache.py)
的只读 cache 视图用于观测，避免读取状态本身改变 LRU 顺序或命中计数。

| 状态变化 / 检查 | 实现证据 | 不得据此推断 |
|---|---|---|
| 相同编号换成另一个实际对象 | 对本 worker 拥有的对象保存弱引用；当前 CPU entry 不同即拒绝沿用身份 | 仅 name/path 相同不等于实际权重正确；工件 SHA/生成资格仍要检查 |
| GPU 加载已分配槽位但尚未完成 | 在已有 completion fence 成功后发布 GPU 确认 | 仅 slot map 不等于 executable；真实 CUDA stream 仍待资格验证 |
| 原生 GPU 驱逐 | 原生回调清槽位前撤销确认；仍在 CPU cache 的副本继续作为 HOST 来源 | 未宣称整个受管 HOST 层已连接 |
| 同 ID、同槽位移除后又激活 | 事件撤销旧确认，即使两次轮询的映射完全相同；新 acquisition 重新确认 | 不用最终映射相同掩盖中间失效 |
| CPU 权重驱逐与合法重载 | 元数据不持有强引用，测试确认旧对象可释放；本 owner 完成的新加载可建立新对象身份 | 不是增加一个隐藏的权重缓存 |
| 未经本 owner 确认的原生编号 | 明确记录 unknown source 与 unconfirmed GPU 两类集合 | 不按整数编号猜 adapter 名称，不将缺失 native source 直接判为 Remote |
| controller 收到状态 | 校验 owner/epoch/clock、slot/CPU 覆盖和完成时间，转为不可变副本；晚到旧 epoch 不覆盖新状态 | 只读快照不持引用，不等于跨副本物理同时快照或 dispatch reservation |
| 所选请求路径 | 通过既有 worker/engine RPC 获取并提交给对应 InstanceSlot，再走 D12 acquisition | 当前仍发生在 resolve 后，不冒称论文要求的 pre-dispatch snapshot |

本步新增 17 项检查；最终完整功能回归 **502 项通过**，独立 safety/census/replay
**44 项通过**，无失败、错误、skip。中途一次针对性回归暴露两处异常表达差异：
空 loader 未生成 CPU entry 时先抛 KeyError，以及更早发现非法删除后丢失原错误原因。
当前分别显式报告未注册对象、保留 invalidation 原因，没有增加加载/推理兜底。
147 项保护清单全部未变。本表作为本步骤交付；没有运行模型，不画性能收益图。

仍需完成：决策前组合各副本来源、实际 HOST/NVMe 引用、原生 footprint 与测得的
D/T/O 类别成本、容量冲突的排队/重选及主动 admission 原子事务。当前不可变视图
只接入了 selected-request 的观测路径，`Router.ieee_confirmed` 尚未由完整实测
候选集驱动。不能把 502 项通过写成 Full、A3 或 S2 实验已完成。

## P1-D14：实际 HOST 存储、GPU 槽位与观测类别

D9 已核对 dense GPU pool，D13 已核对实际 source identity，但尚无原生 CPU
tensor storage 清单。修改前的针对性检查因接口不存在而失败；本步补齐实际
容量来源，不以旧 `adapter_info.size_mb` 或默认文件大小生成 IEEE 成本类别。

官方 [vLLM 0.30.0 LoRAModel](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/lora_model.py)
允许 clone 共享底层 tensor；[packed LoRA weights](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/lora_weights.py)
也可复用 tensor 并保留缺省子模块。其
[worker loader](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
先建立 CPU 表示。因此可证伪的计量假设是：对 tensor view 求和或逐 adapter
直接求和，都可能误算 owner 的实际存储容量；单独删除一个 adapter 也未必能
回收其全部引用容量。

扩展现有 `gpu_monitor.py`，读取原生 `list_adapters()` 和 A/B 的 backing
storage；不读取新工件、不复制权重、不用假想的 rank→字节公式。下面均为小型
真实 PyTorch CPU tensor 的正确性检查，不是 7B/3B 的实测内存占用：

| 构造 / 条件 | 实测或应有容量（byte） | 需要区分的语义 |
|---|---:|---|
| adapter 7 引用共享 512＋独有 256 | 768 | 该 adapter 的完整表示 footprint |
| adapter 8 引用同一共享部分 | 512 | 两 adapter 的 footprint 和为 1280，但不是物理总量 |
| 同时注册 7/8 | **768** | 共享 allocation 只记一次 |
| 删除 7，仅 8 留存 | **512** | 删除 7 只减少 256，不能宣称回收 768 |
| 8 的两个很小 view 合计仅 16 | backing storage 仍为 **512** | view.numel 不等于保留的 storage capacity |
| 已验证的 GPU 两槽位表示（独立数学 fixture） | pool 2048，每槽 1024 | 空槽是池内逻辑容量，不是新增 CUDA free bytes |

原生观测保留 allocation、view、adapter 之间的关系与 dtype/pinning。A/B
不完整、空模块、未审计的 3D/非 CPU/稀疏表示、额外未计入的 tensor 字段明确
报错，不静默漏计。`exclusive_storage_bytes` 只表达不存在其他注册 adapter
引用，**不证明执行引用已经释放**。HOST allocator overhead、Python/RSS、
tmpfs/page cache、staging 和其他服务内存仍另行计量，不能由本表代替 80 GiB
资源包络或 HOST 物理 reservation。

已有 worker observation 和 source-snapshot RPC 现在返回同一原生状态下的
HOST/GPU footprint。controller 校验 storage union、共享边、exclusive 容量
及 GPU slot 对应，保存不可变的来源描述。`service_class()` 使用当前最快有效
native tier 的实测 footprint、rank、dtype、packed/pinning 表示，以及原有
prompt/declared-output/post-admission bins；继续调用 D3 的同一个分类器，未改
行间公式、EWMA 或 routing key。仅有 readiness 的旧视图不能生成 footprint
成本类别，缺失测量不会被补零或套入默认大小。

详细 tensor 清单留在资源/资格观测中；逐请求记录只保存已验证来源、选中副本的
footprint 与 owner 总量，不复制整个 tensor 清单数千次。真实 worker 的完整
观测/RPC 开销尚未测量；资格阶段需检验更新开销和 cadence，再冻结正式配置。

本步新增 **12 项检查**（7 项 HOST 实际 tensor、4 项类别/共享校验、1 项真实
runner 方法中的 footprint 运输与日志大小合同）。完整功能回归 **514 项通过**，
独立 safety/census/replay **44 项通过**，均无失败、错误或 skip；147 项历史
保护清单未变。中途 513 项通过的结果不包含最后的请求证据检查，以 514 为准。

仍未完成：HOST/NVMe managed-copy 所有权与实际文件引用、冷源 footprint/profile、
测得的 D/T/O 初始化、决策前多副本 composition、原子主动 admission、取消终态
对账与物理 GPU owner。没有据此启用 Full 或宣称吞吐/延迟/GPU-s 改善；下一步
回到实际可执行的来源/成本与请求调度连接，安装完成后优先真实模型资格。

## P1-D15：请求加载期间的受管文件副本引用

对照 IEEE §3.1 的“execution/transfer references protect physical copies”，
以及现有 `resolve_lora → native demand_load_and_acquire` 路径，发现旧实现返回
HOST/NVMe 目录后，并未让物理回收方知道该目录正在被读取。修改前的真实 runner
方法检查复现：模拟后端开始加载时，`ResidencyManager._delete_path` 已可删除
对应临时目录；检查失败在“源仍存在”，不是因生成速度发生变化。

本步假设是：只要读取与回收共享同一所有权同步，加载期间就不能删除或替换
其源；收到加载完成确认后，应解除文件引用，而不是无理由延长到整次生成。
核查 [vLLM 0.30.0 worker loader](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
的本地 checkpoint → CPU model → GPU activation 顺序；保留其原生加载流程，
由 D9/D12 已有的完成确认区分文件读取和随后 native tensor 的生命周期。

| 检查 / 条件 | 本步结果 | 边界 |
|---|---|---|
| 请求已解析路径、尚未调用可取消的 native RPC | 原 manager 发放带 owner/lease/path/device/inode 的文件引用 | 这是读引用，不是内容 SHA 验证或 confirmed tier 发布 |
| 加载未完成时删除目录、子文件或父目录 | 同一回收 owner 拒绝，原数据保留 | 不能阻止未接入该 owner 的外部进程或旧旁路直接写文件 |
| 两请求引用同一物理目录 | 各自释放一份；重复释放幂等，最后一份结束后才可回收 | 不按引用数多算一个物理副本；容量记账仍需独立接入 |
| 替换正在读取的目标 | 返回未执行；同一已存在副本的无修改复用仍可成功 | 未增加 sleep、降级路径或伪成功 |
| 同步 tier copy 与另一线程回收 source | 既有复制过程与回收共用锁，复制完成前源不能被删 | 该锁尚无模型级开销测量，不据此宣称异步复制优化 |
| native 完成加载/拷贝确认 | 立即解除文件引用；native CPU/GPU lease 继续保护生成 | 真实 CUDA stream 完成语义仍需 P2 模型资格 |
| 只读 source snapshot 失败 | 释放文件引用，不虚构已经启动 native load | 与 load 回复丢失不同 |
| load 回复丢失 / 取消 | 保留文件引用、native 未决所有权和原请求身份 | 仍需终态对账；不能用 timeout 自动当成已经停止读取 |
| 物理删除失败或引用冲突 | eviction 返回失败，原 tier 集合和 used bytes 不提前扣除 | 不把目录存在性当完整容量模型 |

实现扩展现有 `ResidencyManager` 与 runner 请求生命周期，不另建缓存/实验框架。
非本 owner 管理的 `StorageManager.LocalCache` 暂拒绝申请此引用，避免其独立
cleanup 删除文件却仍称“已保护”。legacy tier hints、文件内容验证、容量
reservation 和 confirmed publication 没有因此升级为 IEEE 合格状态。

新增 **11 项检查**，包括真实临时小文件、两个线程和实际 runner 方法；未生成
模型/adapter 池或负载。首次全请求检查遇到本项目 logger 不支持标准 logging
的多位置参数（47 项中一项错误），已改为其既有接口；最终 49 项请求/文件检查
与完整 **525 项功能回归全部通过**，独立 safety/census/replay **44 项通过**，
均无错误、失败、skip。147 项历史保护清单和计划 SHA 未变。本表是本步骤交付，
不制作性能增益图，不声称完成 A3/S2 或正式 Full。

下一主线仍包括：将真实远程 materialization、旧预加载直接 copy 路径与全部
合法回收统一到物理 owner；完成内容/容量/发布事务及冷源 profile，连接决策前
快照、路由与原子 admission；完成 native 取消对账和物理 GPU 生命周期。现有
`_ensure_local_async` 的 per-adapter fetch 锁不能替代上述所有权。不要在它们
完成前把本步合作式引用称为所有 HOST/NVMe 路径均已受到保护。

## P1-D16：完成后发布、失败保留旧副本、取消等待实际写入结束

延续 D15 的同一来源生命周期问题，检查实际 HTTP fetcher 和初始 tier copy。
修改前两项文件测试均复现旧副本已丢失：下载代码在解包前删除目标目录，因此
损坏 archive 或中途解包失败会毁掉旧的有效来源。另一个直接实现事实是，
名为 async 的 HTTP 路径仍在调用线程内执行 urllib 下载，阻塞同一调度循环。
这些是正确性/执行路径证据，不是已经量化的 TTFT 收益。

实现依据 [Python 3.12 文件重命名语义](https://docs.python.org/3.12/library/os.html#os.replace)、
[异步取消和 shield](https://docs.python.org/3.12/library/asyncio-task.html#shielding-from-cancellation)、
[已运行 Future 的取消语义](https://docs.python.org/3.12/library/concurrent.futures.html#concurrent.futures.Future.cancel)。
本步假设：把私有准备与受同步保护的发布分开，既能保留有效旧源，又不需要把
网络 I/O 锁在请求调度循环；取消必须按实际写入生命周期收尾。

| 条件 | 实现与检查结果 | 不能据此宣称 |
|---|---|---|
| 下载/解包尚未结束 | 同一文件系统的私有 sibling workspace；目标仍为旧有效副本 | 私有 staging 不等于免费容量，仍需纳入后续 budget reservation |
| 损坏/中断/空 archive | 不发布，不提前删除旧目标；清理本次私有临时文件 | 尚未验证完整 LoRA 内容 SHA、rank 或生成正确性 |
| 完成后的 publication | 真正 native 路径调用 manager 的同一个引用/回收 owner | 两次 rename 不是面向无锁 filesystem readers 的单一原子 exchange |
| 目标还有加载读引用 | publication 明确冲突，旧副本保留；不把错误变成零毫秒成功 | 当前冲突仍需上层排队/重选，不是完成了完整调度策略 |
| rename 失败 | 在 owner 临界区恢复旧副本；恢复也失败则保留 recovery 路径并报错 | 没有完成掉电持久性/crash recovery 资格 |
| HTTP 正在读取时取消请求 | 工作线程收到取消标志；等待实际 Future 终止后才释放 per-adapter 调用范围 | 发送 cancel 不是工作已经结束；阻塞系统调用仍由 timeout/watchdog 约束 |
| 重复取消 | 仍不释放活跃写入；线程结束后传播原取消 | 不把取消请求算作成功 fetch |
| tier 目录整体回收 | 活跃 materialization 保留其所属 tier；旧目标仍可取得读引用 | transfer 生命周期登记不是物理字节预留 |
| 普通 NVMe→HOST/初始 copy | 使用同一 staged publication；失败保留旧目标；初始 reset 遵守已有引用 | legacy path hints 仍不是 confirmed source registry |

改动复用现有 HTTP client、ResidencyManager、ExperimentStack 和 runner。真正
native HTTP 路径在原服务资源域内执行阻塞 I/O，不转移到外置监控资源域；旧
local-sim 路径没有被冒称真实远程。native fetch 异常直接保留，不落到本地工件
或 `(False,0ms)` 的错误掩盖路径。未改九个行间公式、公开 workload 或 baseline
策略。同期没有新的模型性能运行。

新增 **13 项检查**：8 项 HTTP/publication（含既有小型 loopback server 的
真实 HTTP roundtrip）、2 项本地所有权/失败 copy、3 项实际 runner 的线程、
取消与 owner 集成。取消测试用受控 response 对象，不冒称远端 174 已通过。
首次 system Python 测试因该解释器缺 numpy 而无法导入项目；随后使用既有稳定
推理环境，不安装依赖。修改前两个反例均失败；修改后 **538 项完整功能回归**
通过（含 62 项相关检查），零失败/错误/skip。安全与历史保护校验另见执行记录。

未完成的必要边界：内容 SHA/representation/profile、所有 tier 物理 used+
reserved+staging 字节、completed-source registry 与 epoch 的联合发布、文件
失效后原生 CPU/GPU 来源的协调，以及 native 取消终态和 GPU allocation owner。
删除路径不等于实际 backing storage 已释放：还需核查 mmap、page cache 和
原生 pinned CPU tensor 的存活关系，不能把 unlink 直接当作物理预算腾空。
同步本地 copy 仍未做异步性能优化；当前线程池/staging 峰值与 publication
开销必须在资格阶段测量。继续回到完整来源/成本—路由/admission 连接，不把
该表当作 A3、S1/S2、Full 或真实远端全池资格已完成。

## P1-D17：文件表示的实测占用、共享与观测范围

延续 D14 的原生 tensor footprint 与 D15/D16 的文件所有权，检查旧容量账本。
`TierCapacity.used_bytes` 仍按 `ArtifactMetadata.size_bytes` 在最快 tier 转移时
增减；这不能表达保留的低层副本、多个文件表示、共享 inode 和下载暂存。
不把这个旧账本直接当作 IEEE 的物理 used/reserved 证据。

只读核查既有 3B 工件 `code_lora_0015`，三个实际文件如下；未读取/复制整池：

| 文件 | 逻辑字节 | 已分配 512-byte blocks | 硬链接计数 |
|---|---:|---:|---:|
| adapter_config.json | 546 | 8 | 6 |
| adapter_data.bin | 28805398 | 56264 | 6 |
| adapter_model.safetensors | 18379976 | 35904 | 6 |

这证明工件的文件表示不能统一套用一个名义 adapter 大小，也不能根据逻辑
adapter 个数直接累加物理文件占用；并未推断这三个文件都是 native loader 的
实际读取集合。其余指向 backbone/tokenizer 的支持链接属于原冻结池，未修改。
受管物化目录的 inventory 不跟随这类链接，避免把外部模型算进本 owner。

依据 [Python 3.12 stat](https://docs.python.org/3.12/library/os.html#os.stat_result)
区分 `st_size` 与 `512*st_blocks`；依据
[Linux memory.stat](https://docs.kernel.org/admin-guide/cgroup-v2.html)
区分文件占用、page cache、tmpfs、mmap 和匿名 tensor。
[vLLM 0.30.0 的官方 loader](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/worker_manager.py)
先从本地 checkpoint 建立 CPU model 再激活 GPU；LRU 路径先成功加载新 adapter
才回收旧项，因此临时峰值还可能超过其 steady-state CPU adapter 个数上限。
本步只解决可观察量，不用最终缓存大小冒充加载峰值或 HOST 总内存。

| 条件 / 检查 | 本步结果 | 范围限制 |
|---|---|---|
| 低层副本仍存在 | 扫描原 owner 的 HOST/NVMe 目录，不只扫描 registry 的最快 tier | 实际 HOST tmpfs 资格仍需原部署检查 |
| 同 inode 的多个硬链接 | 按 device/inode 去重，共享分配只计一次 | 内容相同而 inode 不同不做猜测性去重 |
| 跨 tier 共享 inode | owner 总量去重，显式标记 tier 小计不可直接相加 | 具体 operator budget 归属与 reservation 仍待接入 |
| 稀疏文件 | 同时保留逻辑大小和实际 allocated blocks | 不以其中一种代替另一种，也不声称覆盖 reflink extent 去重 |
| 暂存/恢复目录 | 静止时与原目标一起计入，不能按隐藏目录名漏掉 | 活跃 transfer 没有剩余增长预留时拒绝完整容量快照 |
| 外部硬链接 | 记录 owner 外的 link 数 | unlink 不代表底层块或 mmap 已释放 |
| 选中请求读取文件 | 在现有读引用内测量此源，摘要进入原 request evidence | 不逐请求重复整个文件树，不等于内容 SHA 验证 |
| 符号链接、特殊文件、缺失根目录 | 明确拒绝，不填零、不追踪外部路径 | 原冻结池不因此被修改；受管目录需要合法物化 |
| 扫描期间非合作式修改 | stat 身份/大小/块数/时间发生变化则拒绝该次扫描 | 不是对任意外部写者的锁或完整内容防篡改保证 |

实现复用 `LocalSourceReferences` 和 `ResidencyManager`；native runner 原有
文件引用回执自然携带该源的占用摘要。完整 inventory 是需要时调用的静止
观测，不在每个请求中扫描全池。完整目录块纳入分配数，但 inode/journal
额外开销、已 unlink 仍打开的文件、共享 extent、CPU allocator、page cache
没有被虚构成已经测全；`physical_release_proven`、`content_verified` 和
`capacity_reserved` 均明确为 false。不能把这个接口的结果单独用于批准一项
可能继续增长的物化任务。

新增 **9 项检查**；前两项在修改前因为没有该观测接口而报错，不声称它们已
复现一次模型 OOM。修改后以小型实际文件验证双副本、暂存、hardlink、稀疏
文件、缺失/越界表示和修改冲突；实际 runner 测试验证摘要在文件释放后仍留在
请求证据中。未生成工件池、负载或真实模型输出。完整功能回归 **547 项通过**，
独立 safety/census/replay **44 项通过**，均无失败/错误/skip。本表为本次交付，
不画伪性能收益图。

下一步仍是把验证后的内容/表示和传输峰值连接到 owner 字节 reservation，
再在发布/释放时更新真实 used/reserved；接入冷源成本、决策前候选快照和
原子 routing/admission。未完成全套物理容量保证、Full 或任何正式性能实验。
