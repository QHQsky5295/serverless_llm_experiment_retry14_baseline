# P1 — IEEE 公式与实现合同（执行中）

本表不是性能结果，也不表示 Full 已完成 IEEE 对齐。主比较必须等所有关键
合同关闭；不得把旧代码的指标移植到新设计上。论文源文件未改动。

本文件前面的首次审计表保留历史发现；逐项最新进展见 D1–D34。测试通过不等于
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

## P1-D18：远端物化绑定冻结内容清单，逐文件限制写入并校验

检查 D17 后的下一依赖：旧 HTTP `/manifest` 只有 ID/可选 size；现存 3B 的
`.publicmix_generation_manifest.json` 是来源、名义大小、rank 和编号清单，
不是逐文件内容 SHA。不能把这两种清单当成物理写入大小或同工件证明。
原 fetcher 仅判断“解包后非空”，因此需要显式引入已有权重的静态内容索引，
而不是信任本次下载自行声明的权重身份。

实现依据 [Python 3.12 tar extraction filters](https://docs.python.org/3.12/library/tarfile.html#extraction-filters)：
路径过滤不等于内容或资源上限验证；档案可以包含重复成员、链接、稀疏表示。
同时核查 [vLLM 0.30.0 本地 LoRA 加载流程](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/worker_manager.py)，
文件身份校验和 loader 的 PEFT/模块合法性是两个不同前提，均不能代替实际生成
时的 adapter 正确性检查。本步不改 loader 的原生推理算法或论文公式。

| 情况 | 本步行为 | 不能据此宣称 |
|---|---|---|
| native true-remote 没有冻结内容清单 | 在网络/目标写入前拒绝下载 | 没有用远端 name-only 清单或名义 size 兜底 |
| 清单文件 | `artifact_content_v1`：每 adapter ID 的完整相对文件路径、整数 size_bytes、SHA256 | 内容索引是已有工件元数据，不是重建 adapter/负载 |
| 相同清单重排 | 规范排序后得到相同 SHA；配置后不能换成不同内容 | digest 是索引身份，不是本次压缩包 hash |
| HTTP body | 必须有正 Content-Length；记录真正读到的线上字节，长度不符则不发布 | 头部来自远端，不是已经取得的磁盘预算 |
| 普通文件 member | 大小先与冻结值一致才写；边写边 SHA，不再解包后额外读一遍权重 | 校验开销计入这次物化时间，不称免费工作 |
| 额外、重复、缺失、错大小或错 SHA | 拒绝，清理本次私有目录，旧有效目标保留 | 不把成功解包或同长度当正确内容 |
| 链接、稀疏文件、非规范路径 | strict representation 明确拒绝 | 该版本资格限于普通文件物化，不能假装已覆盖任意 archive 表示 |
| 校验成功但 publication 有引用冲突 | 保存 content_verified=true、state=not_published；旧目标保留 | 内容正确不等于已完成发布/变为可路由来源 |
| 成功发布 | 保存索引 SHA、线上/解包字节、发布状态与耗时 | 不是 selected-replica dispatch tier 快照 |

原 client 增加 `configure_content_manifest()`；native runner 从按模型配置的
`artifact_content_manifest_path` 读取静态 JSON，实际 HTTP 调用强制内容校验。
目录可见但尚未经过本路径下载的旧 cache 不因此自动得到 confirmed 状态。
原 native transfer 的异步线程、实际取消 join 和同 owner 发布保持不变；其
每次 transfer 回执进入本轮 coordination metadata，包含失败的 not_published。
全局中止时的增量 journal 仍未接入，不能承诺已有完整中止证据落盘。

本步没有替换远端原服务，没有启动 174，也没有复制或重哈希完整池。现有全池
清单不能直接冒充这个内容索引；后续资格步骤需优先复用可核验的历史 SHA，
缺失的才从既有权重按唯一文件去重、受限扫描产生小型元数据。各 baseline
的真实远程路径也须满足同一内容合同，不能将此 Prime 接入称为共同资格完成。

新增 8 项检查，含多种失败子例；原先两项检查在旧接口上失败，非模型 OOM。
35 项 HTTP/原 runner 检查通过，包含原 loopback server 的真实 HTTP 下载和
原先的取消/读引用冲突测试。小型 payload 不是合法训练权重，故只证明传输
合同；7B/3B 实际全池 LoRA 功能覆盖仍待执行。

明确未完成：`used+reserved` 的原子空间预留与排队，archive/temp/目录与
解包峰值、全局 cgroup 内存、已 unlink backing 的存活、publisher 的内容
epoch 与 pre-decision registry。当前可在写 archive 前取得线上声明大小，
并已有可信逐文件大小和受限写入，但它们仍必须连接物理 owner 的容量事务，
不能从本步推出“整个传输已保证不超预算”。继续沿这一主线完成，不新增
第二套 fetch/实验框架，也不以本检查替代 M1/M2、消融或 motivation。

## P1-D19：下载前预分配实际文件空间，统一计入旧副本和并发传输

D18 已获得可信 payload 文件大小，但收到 HTTP 长度仍不等于获得空间。
本步在原 owner/fetcher 内接入真正的写前分配：先在同一锁内检查受管 tier
全部已有普通文件（含旧目标、其他 transfer、未清理残留），再为压缩包及每个
payload 文件执行 `posix_fallocate`。开始读 body 前，核验每个实际 inode 的
逻辑长度和分配块数；不通过 `truncate` 或稀疏文件伪装预留。

依据 [Python 的 posix_fallocate 接口](https://docs.python.org/3.12/library/os.html#os.posix_fallocate)
及 [Linux/POSIX 预分配语义](https://man7.org/linux/man-pages/man3/posix_fallocate.3.html)，
私有文件独占写入，已有区间只覆盖、不截断、不增长。本机 `/home/qhq` 位于
ext4；逐文件分配粒度从该文件系统读取，并在分配后验证实际块数，未知或不匹配
的表示不准入。结合此前核对的 vLLM 原生本地加载流程，旧有效副本一直保留到
新内容校验及受管发布成功，不能提前拿旧副本空间去批准下载。

设分配粒度为 \(g\)，压缩包长度为 \(b\)，冻结文件长度为 \(s_f\)，本次新增
普通文件分配为
\[
P=g\lceil b/g\rceil+\sum_f g\lceil s_f/g\rceil.
\]
检查当前实际普通文件块数 \(U_{file}\) 满足 \(U_{file}+P\le B_{file}\)。
已获准 transfer 的空间已由文件系统实际分配，包含在后续 \(U_{file}\) 内，
不能再把其 `reserved_file_bytes` 加一次。此值是写前回执，不是完成后的额外
待分配量。文件预算按 owner 冻结；容量冲突不改变预算、不重新估计较小 footprint。

| 条件 | 实际检查结果 / 实现行为 | 边界 |
|---|---|---|
| 旧副本 + archive + payload 同时存在 | 三者同时进入文件预算 | 不用最终 payload 大小代替峰值 |
| 两线程竞争仅够一项 transfer 的容量 | 一项预分配成功，另一项容量冲突 | 不是按每个 fetch 各给一份剩余量 |
| 容量不足 | 原 runner 在第一次 body read 前拒绝，旧目标不变 | 已发 HTTP 请求可能触发远端打包，该成本不能抹去 |
| 已预分配的文件正在写 | 允许内容时间戳变化，仍严格检查 inode、长度、块数、link count | 不把未知 writer 的扫描当完整快照 |
| 同目标重复 transfer | owner 明确冲突，不建立第二 workspace | 原请求锁负责正常同 adapter 请求串行复用；通用 pending queue 尚待接入 |
| 取消后 writer 未结束 | 实际线程仍持有已分配空间，不能重用或清理其目录 | 延续 D16 的真实 Future join |
| cleanup 失败 | 残留文件被下次实际扫描继续计入 | 删除 reservation 记录不代表物理空间已释放 |
| 不支持预分配或块数不匹配 | 不接收 body，不采用稀疏/猜测性兜底 | 不是证明任意文件系统都已合格 |
| 发布 | 同 owner 校验容量身份、读引用和目标后，使用原发布协议 | 仍非 crash-durable registry transaction |

native runner 原有 NVMe ceiling 传给该 owner，不引入按模型名称估计 adapter
大小或按正式点调整并发的新系数。HTTP 写入使用已分配文件的 `r+b`，避免 `wb`
截断预留；strict SHA 和长度检查仍完整执行。transfer 证据增加文件预算、预留
块数、既有块数、owner/transfer 身份和分配粒度。

**范围明确限定为受管普通文件的实际 allocated blocks。**目录、inode、journal、
page cache、unlink 后仍存活的外部引用、native CPU tensors 和 GPU/KV 预算不能
由本接口替代。目录仍由 D17 inventory 观测，但不并入这个普通文件容量池。
预分配失败的部分文件由原 workspace 清理，失败证据保留；外置磁盘与 cgroup
保护继续适用。legacy local-copy/preload 的空间准入、跨 tier 共享预算归属、
native HOST/GPU 联合事务和容量冲突后的调度等待仍是 Full 资格的开放项。

新增 9 项检查，包含真实双线程与小型真实文件预分配，并加强原 HTTP runner
成功/取消回执检查。最先两项旧实现因缺接口失败；第一次相关回归有一项测试
fixture 缺失 `patch` 导入，已修正，未削弱运行时合同。最终完整功能回归
**564 项通过**，独立 safety/census/replay **44 项通过**，无失败/错误/skip。
本表是容量正确性证据，不是
性能收益图。扫描/预分配实际开销须在模型资格中测量，不能宣称免费或更快。

下一主线：原 P2 安装完成即优先验证真实后端、worker/clock/stream；继续把
内容 epoch、冷源成本和剩余预算送入决策前快照，并完成 native admission、
已知冲突等待、abort/release 与 GPU 生命周期。Serverless 仍是首个 baseline。

## P1-D20：完成区间的原生事件与跨进程在线更新

本步回到 D3 的 \(\widehat S=\widehat D+\widehat T+\widehat O\)。历史 runner
仍未创建 D3 的 admission-time observation，不能因为公式单测通过，就把旧路由
称为 IEEE Full。新增内容是**原生事件到该 observation 的桥接**，不是完整的
决策前快照、profile 资格或路由策略接入。

### 依据、假设与实现边界

历史 `cf01792` 已实现按完成区间更新的 EWMA；旧直接/子进程生成入口只返回
最终结果。如果仅在该返回值上补算 D/T/O，长请求的 T 更新将滞后到解码结束，
而取消请求已经完成的 T 会丢失。可证伪的检查是：阻止末 token 产生时，T 的
样本数必须已经增加、O 的样本数仍为零；首 token 后取消，保持这个状态。

复核 [vLLM 0.30 原生指标设计](https://docs.vllm.ai/en/v0.30.0/design/metrics/)
和已安装 `vllm/v1/metrics/stats.py` 的 `first_token_ts`、`last_token_ts` 更新路径，
沿用 EngineCore token event 时间，不把 callback 接收时间当生成时间。处理放在
现有 frontend/控制器路径，不修改 GPU 内循环或轮询计时。跨进程相减仍要求
已验证的本机同一 monotonic clock identity，不能推广为跨主机时钟一致。

| 论文规范语义 | 当前实现证据 | 尚未完成 |
|---|---|---|
| D：admission 至可执行引用取得 | 继续使用原 `ServiceIntervalObservation.acquire`，不改定义 | Full 原子接纳与决策前类别绑定 |
| T：取得引用至首 token | 直接后端首次原生 token 发出一次事件；同请求 RPC 可先传首 token 帧 | 实际模型端到端开销与下一次路由消费验证 |
| O：首至末 token | 仅正常原生终态、完整 token 合同通过后发送末 token；保留末 token 时间，不包含完成通知尾部 | 不作为独立性能或 warm profile |
| 更新 admission-time class | `NativeServiceIntervalObserver` 持有固定 observation，按完成区间更新同一个 EWMA | 正式多副本 profile 初始化、全源类别覆盖 |
| 请求取消或失败 | 保留已经完成的 D/T，拒绝取消后的迟到事件；不创建 O 样本 | Full 控制器持久结果 journal 与取消闭环 |
| 错误信息不可成为确认状态 | clock、request、adapter/reference、事件序号、token 数校验；最终结果再核对事件身份和时间 | 通信失败运行仍不具备正式性能资格 |

使用现有 dedicated worker TCP 通道，opt-in `native_service_events_v1`；成功
请求最多两帧，不传逐 token 流、不新增轮询进程。回调只更新有界内存状态，必须
同步返回 None；socket 线程将回调送回所属 event loop。取消立即撤下连接，迟到
回调不进入新运行，未知原生操作仍由既有 uncertainty/reference 规则保留。
无事件 observer 的历史 API 保持原返回合同；native observer 不允许 legacy
时间兜底。终态与事件不一致会拒绝整个请求成功资格，不能拿已收到事件代替终态。

### 本步验收表：不是性能实验

新增 12 个检查覆盖单次 EWMA 更新、错误身份/时钟/次序、单 token、直接生成
过程中的更新与取消，以及**实际 worker handler + 实际 proxy + loopback TCP**
的成功、取消、生成失败、重复帧、错误终态。推理输出由确定性 fixture 提供，
未加载模型；网络传输和 handler 本身不是 mock。

- 首轮 10 项：9 通过、1 收尾错误，原因是测试等待 server 退出前未关闭其池连接；
  修正测试的连接关闭顺序，未修改成功标准或添加运行时容错。
- 随后相关 131 项通过；补上重复帧检查后，最终完整功能回归 590 项通过，
  无失败、错误、skip。独立安全/census/replay 56 项通过。
- 147 个历史保护项和批准计划 SHA 均未变化。没有新增权重、负载或 GPU 性能点。

按计划第十一节，本步选择状态表而非“增益”图。此桥接尚未在实际模型上资格，
不重复旧同 prompt 控制；下一步推进 Full 决策前 source/cost/profile owner
和原子 reservation/admission，随后在同一次实际模型资格中检验桥接。不能把
测试初始 profile 数字填入正式配置，也不能把 resolve 后的 tier 冒称 admission
时刻的源状态。Serverless 仍保持 baseline 首位，M1/M2 等正式矩阵未开始。

## P1-D21：实际请求先保护原生缓存，再决定是否需要文件

### 历史依据与可证伪假设

`044873d` 中实际 `ScenarioRunner` 无条件先 `_resolve_lora`，然后才调用原生
引用取得路径。D12/D15 保护了文件加载与后续权重使用，却没有分离“缓存 tensor
仍有效”和“原始文件仍存在”。因此 selected replica 已有 GPU/HOST 权重也会
先进入文件来源解析。假设：文件已被回收、但原生权重仍有效时，应能持有正确
adapter 的原生引用，不调用文件解析或文件读引用取得。

对照 [vLLM 0.30 官方 LRU worker 实现](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/worker_manager.py)：
`LRUCacheWorkerLoRAManager.add_adapter` 在 CPU cache 命中且
`load_inplace=False` 时，使用现有 CPU 对象并激活，不调用 `_load_adapter`。
本机安装源码与该控制分支一致。仅借鉴这一有直接关系的原生行为，不改原生
LRU victim 选择、论文公式、生成合同或 artifact 内容。

| 论文规范语义 | 当前请求路径与验证证据 | 限制 |
|---|---|---|
| 最快有效副本是权重状态，不是文件名 | selected worker 先返回有身份的 GPU/HOST 来源；原生线程重新检查并保护后，跳过文件解析 | 仍不是全副本决策前快照 |
| HOST 命中仍有准备工作 | 复用已注册 CPU tensor，保留实际 promotion/fence 时长及原始 HOST 来源 | 未测实际模型净延迟收益 |
| 陈旧状态不能冒充命中 | acquisition 携带 required_source_tier；变化时先拒绝且不加载，再重新观察；缓存全失效才解析文件 | 尚非跨层完整原子 admission |
| GPU slot 不等于已确认 GPU-ready | remove/reactivate 后未确认 slot 仍以有效 HOST 来源处理，在 fence 后重新发布 GPU | 不把这次确认回填成原先 GPU hit |
| 独立来源生命周期 | native hit 不拿文件读引用；cold load 保留原文件保护；未知 acquisition 留下 native/controller ownership | 未完成全部物理预算/持久 journal |
| 不改变观测时间定义 | 保存来源、冲突与 receipt，实际时间不填零；保留 confirmed_dispatch_snapshot=False | 不据此宣称 admission 时刻 D=0 |

### 正确性状态表，不是性能结果

新增八项方法检查（含多种来源子情形）：GPU/HOST 已缓存但文件不可用、
GPU→HOST/全失效/未确认 slot 的并发变化、未知 acquisition 取消、adapter ID
碰撞、无效来源要求、引用重放身份和原始层级保持。使用实际 controller/owner
方法及确定性原生缓存 fixture，没有模型推理或真实远端下载。

- 修改前：实际请求检查在 GPU/HOST 两个子情形均错误进入文件解析；两项原生
  guard 检查因接口尚不存在报错。没有把此反例写成真实 CUDA 性能退化。
- 首次相关回归 111 项中两项旧测试期望错误发生于文件解析之后；现观测已移到
  之前。保留一个显式“取得文件读引用后的 snapshot 失败”检查，另一个加强为
  “错误时钟在文件读取之前拒绝”。没有生产 fallback 或放宽成功标准。
- 相关 116 项通过；完整功能回归 598 项、独立安全/census/replay 56 项通过。
  测试输出来自本轮实际执行。所有历史保护项及计划 SHA 不变。

按计划第十一节与 academic-plotting 的证据选择流程，本步采用状态表，未制作
虚构的延迟/收益曲线。真正的净收益还须把原生 snapshot 查询开销一并计入；不能
由少了一次文件解析推断 G1/G2 已改善。下一步直接返回 Full 的 admission-time
class/profile、全副本 source/cost 与原子准入接入，然后在该真实路径进行模型
资格，避免继续重复同 prompt 或无新问题的微测。

## P1-D22：实测服务 profile 的身份、初始化与扩容继承

### 缺口、依据与本次边界

`cb7bc80` 已有 D/T/O 的数学对象和完成区间事件，但实际 ScenarioRunner 没有
读取初始化测量，也没有为 InstanceSlot 建立各自的估计器。IEEE 正文要求同
模型/后端、代表性观测类的 profiling 初始化，并由新副本继承冻结 profile。
不能用旧按层级常量、缺失类填零或前一正式运行的学习状态来填这个缺口。

核对 [vLLM 0.30 原生统计源码](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/metrics/stats.py)：
`first_token_ts`/`last_token_ts` 来自 EngineCore 事件；解码区间由两者相减。
这支持复用原生事件边界，不支持用 HTTP 返回时间替代最后 token，也不替本系统
提供 adapter acquisition 时刻。D/T/O 仍按论文定义；不移植其他系统的目标函数。

可证伪假设：同一冻结测量应可复现相同初始化；任何模型、后端资源或输入合同
变化必须拒绝复用；一副本的 EWMA 更新不能改变另一新副本的初始化。

| 论文规范语义 | 当前实现证据 | 未完成事项 |
|---|---|---|
| 同模型/后端初始化 | profile 文件 SHA、实际 engine 配置、环境/资源/输入 SHA 显式匹配；只排除 GPU 放置编号 | 实际 campaign 身份生成与真实代表性 profiling 仍需资格 |
| 使用实测 D/T/O | 输入保存 admission/acquisition/first/last 原生边界，按固定 admission 类计算均值；不接受手填 latency 代替这些边界 | 文件合同验证不是来源真实性证明；没有提交生产用虚构样本 |
| 固定观测类 | 原始 prompt、declared output、rank、footprint、representation、post-admission count 重新分类 | 未测的类拒绝，不自动借用相邻类/尾桶 |
| 已保护 GPU hit 的 D=0 | 必须明确 protected_at_admission，且 acquisition 与 admission 同时 | selected-path 稍后取得引用仍不能冒充该条件 |
| 新副本继承冻结初始化 | 每个实际 pool slot 创建独立估计器；共享同一个 runtime 的假副本拒绝 | 实际 scale-out 模型路径还未资格 |
| 扩容配置一致 | 新 engine 配置先检查；不匹配则 shutdown 并报错，不进行 warmup | 完整 physical lifecycle 仍单独验收，shutdown 返回不是释放证明 |
| 路由使用显式 bin width | ScenarioRunner 向现有 IEEE Router 传递 service_bin_ms | 完整决策前快照、原子 admission 和请求观察对象接入仍未完成 |

接口放在现有 `resource_coordination.ieee_service_profile`：`path`、`sha256`、
`context`（backend_environment_sha256/resource_envelope_sha256/input_contract_sha256）
和 `ewma_beta`。实际模型配置包含后端最终解析设置，不以全局 GPU 可见列表代替
子 runtime 身份。汇总记录 profile SHA、支持类数、样本数、来源 run SHA 和 beta。
每模型配置在验证后冻结；本次没有新增生产配置、权重、负载或测量数据。

### 正确性状态表

新增 11 项确定性检查：从原生边界求均值、不可变 profile、SHA/配置/context
失配、错误时钟、重复/不正确样本、token 合同、时间顺序、GPU 保护、单 token、
缺失类、扩容学习隔离、实际 runner 初始化/摘要及配置失配的扩容收尾。
这些检查使用小型临时人工 fixture，明确不是模型 profiling 或性能数据。

- 首次 609 项功能回归有两项旧 `__new__` 测试缺初始化字段；明确补齐其 legacy
  `service_profiles=None`，未增加生产默认估计或放宽 IEEE 检查。
- 最终 609 项功能检查通过（23.285 秒），56 项独立安全/census/replay 检查通过
  （0.621 秒），无失败、错误或跳过。147 项历史保护清单与计划 SHA 不变。
- 本步用正确性状态表交付，不生成没有实测支持的收益图。尚无新的 G1/G2 结果，
  `Full` 不能从本步单独取得资格。

下一步是实际 pre-decision source/class/cost 组合和原子 admission，将 D20 的
观察对象绑定在真实接纳时刻；之后只做有这些完整边界的整合模型资格。不要把
fixture 导出成生产 profile，或再重复无新问题的同 prompt / 零权重检查。

## P1-D23：把已验证文件发布接入真实 HOST/NVMe 状态

`d324b26` 的原生 tensor 状态已有 owner/epoch，但文件 owner 只有路径读引用和
空间盘点。真实 HTTP 路径虽验证了 payload SHA，验证结果没有成为可供路由读取
的已确认本地状态。直接用目录存在或旧 tier hint 补齐快照，会混淆“存在”与
“已完成且可用”。本步复用原 HTTP writer、LocalSourceReferences 和层间复制，
没有建立新的下载框架或替代论文算法。

参考 [vLLM 0.30 的 atomic_writer 与下载锁](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/model_loader/weight_utils.py)：
完成后同文件系统发布与互斥下载是现有底层做法。本项目在已有机制上连接内容
身份和副本状态，不把原子 rename 本身称为新贡献。

| 论文规范语义 | 当前实现证据 | 限制 |
|---|---|---|
| 目的副本可用后才发布 | actual ScenarioRunner 的严格 HTTP 路径传入逐文件验证回执，owner 检查实际目的内容并发布 epoch/content SHA/footprint | 本轮小文件 loopback，不是 174 或完整池性能资格 |
| 位置不是缓存标签 | source_snapshot 返回已确认文件副本；现存但未验证的目录报 unknown，不能记为命中或 Remote | 旧缓存复用需单独验证；cold 主协议不自动导入未知旧目录 |
| 撤回先于存储复用 | managed mutation 先撤回，失败替换只恢复身份与内容记录仍一致的旧副本 | 非本 owner 的外部写入不受合作锁保护 |
| 引用绑定所选源 | acquire_confirmed 检查 owner、epoch、content identity 并在同一锁内持有读引用 | 未接入完整 pre-decision routing/atomic admission |
| 层间迁移保留内容身份 | 既有 NVMe→文件型 HOST 复制验证目的字节后保留同内容 SHA，回收 HOST 后 NVMe 仍可见 | 不把文件型 HOST 视作已注册 native CPU tensor；物理 HOST 内存与复制准入仍独立验收 |
| 观测不改变策略 | snapshot 不加载、不触碰 LRU、不重读整份权重；返回副本不可被调用方修改内部记录 | 元数据扫描/发布校验的真实开销尚未量化 |

### 反例及修正

首次八项检查有一项失败：测试在流式验证之后、发布之前等长改写目的文件，
原 stat 签名检查没有拒绝。本机此反例说明仅依赖大小/inode/时间戳不足以确认
最终目的字节。现发布边界对实际目的文件再做一次 SHA 校验，复用现有冻结
文件索引，不在每次路由查询重复哈希；该开销必须进入后续真实运行。

发布后的正确性依赖所有服务内写入、回收和复制使用共同 owner。元数据检查
能撤回观察到的外部变化，但不声称防范任意不合作进程或具有同 UID 的恶意
改写，也不声称 inode/stat 是内容哈希。外部修改属于协议破坏，而非允许的
后台更新路径。没有通过改 checksum、放宽错误或缩小生成目标消除该反例。

### 状态表与下一步

新增十项方法检查包括实际 runner 的发布/保护、未知目录、失败传输、撤回与
恢复、可观察外部变化、错误 owner/epoch/identity、校验后改写、真实 localhost
HTTP 往返、HOST 复制/下层保留和损坏复制。全部使用微小 fixture，不生成模型
工件或回放负载。真实 copytree 故障测试还断言破坏确实发生，避免仅因测试
包装器签名错误而“成功拒绝”。

最终功能回归 619 项通过（29.472 秒），独立安全/census/replay 56 项通过
（0.495 秒），无失败或跳过。完整旧 smoke 中一次 dummy-model 的 HF HEAD
发生 TLS 重试，未下载模型；该运行耗时不作性能证据。147 项历史保护清单和
源计划 SHA 不变。按计划以正确性状态表交付，没有制作虚构的延迟收益图。

下一步将原生 GPU/HOST 与此文件型 HOST/NVMe 状态组合为决策前 source/class/
cost 视图，再绑定真实 admission 观察和引用。REMOTE 身份/代表性 profile、
全层物理预算、准入与生命周期仍需整合；本步不能宣称 Full、主比较或 A3 已完成。

## P1-D24：真实请求的决策前全副本状态与服务成本

### 缺口与因果问题

`da9109b` 已有已确认原生源、已验证文件源、冻结实测 profile 和 IEEE Router，
但真实请求仍调用不带 IEEE snapshot 的 select_instance。分开的对象存在，
不等于论文 Eq.(2)–(3) 已决定实际目标。本步把这些输入接入原请求方法，
不另建回放器、不替换论文排序、不把历史缓存提示重命名为确认状态。

可证伪问题：旧 affinity/handoff 标签指向慢副本时，实际路由是否仍按同一
已接收视图的服务分桶与 Q 选目标？空可行集是否真正排队？接收过程中副本或
原生 epoch 改变时，是否会用被拒绝的旧观察做决策？

### 论文语义与当前接入

| 论文规范语义 | 当前实现证据 | 限制 / 后续验收 |
|---|---|---|
| 在选目标之前观察所有候选 | 原 runner 并行收集原生 source_snapshot；接收后合并文件源、当前 request/adapter counts 和 profile 估计；实际 Router 使用该 tuple | 是 controller 已接收视图，不是假称跨进程同时采样；必须单独复查所选源 |
| 最快有效源及表示相关成本 | native GPU → native HOST → 文件 HOST → NVMe → Remote；保持 rank/真实 footprint/representation；来源不明拒绝 | 原生名称/rank 绑定不等于已独立证明数值正确；native evidence 标 expected content，未冒充实际权重哈希证明 |
| Remote 具有固定内容身份 | 仅从既有小 PEFT config 读 rank，并核对冻结 content index；统计未压缩文件树字节 | 不进行本地权重 fallback；逻辑 payload 不是 gzip 线上字节或 GPU/CPU tensor 大小 |
| 同一收到的可行集合与 Q | 非目标提示不参与新排序；所有本地字段在最后一次 await 后组合；选择/计数预留间不 yield | 不等于原生资源已经原子预留；真实 physical admission 仍未整合 |
| 不能使用拒收的旧状态 | membership 改变或比已提交 epoch 旧时重新收集；未知 forwarding 不能假装 pending=0 | 后续需纳入有原生所有权的主动 preparation，未声称 Full 已启用全部机制 |
| GPU utilization 是执行忙碌率 | 查询 NVML gpu 时间占比，按冻结 hint cadence 保留采样时间；native CUDA UUID 定位物理 GPU | 不是即时精确 occupancy，也不是显存占比；模型下采样/RPC开销待测 |
| 副本位置须真实 | worker 发布 CUDA 实际 UUID；控制器按 UUID 查询，TP=1 下不同副本不能重复同一物理 UUID | 不把 worker-local CUDA ordinal 直接当主机 NVML index；TP>1 未资格 |
| 没有可行副本应等待 | 实际请求空可行集进入容量等待，不经过旧 slot=None 主副本兼容路径 | 取消等待不虚构 dispatch 或改变已持有计数 |
| load 任务有生命周期 | 保存 admitted request 的准备意图集合，完成 acquire 才清除；未知 mutation 保留并撤回副本 | 数的是请求准备意图，不是已合并 transfer 数/线上字节；不是完整主动 load ledger |

参照 [NVIDIA NVML 利用率定义](https://docs.nvidia.com/deploy/nvml-api/api/structnvmlUtilization__t.html)：
gpu 字段反映采样窗口内 kernel 执行的时间比例；本项目旧 GPU memory monitor 的
`utilization_percent` 是 used/total 显存比，不能替代该量。采样有窗口，时间戳和
cadence 进入证据，不将其包装成逐请求瞬时 SM occupancy。

参照 [PyTorch 2.13 CUDA device properties 实现](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/cuda/Module.cpp)：
设备属性暴露 UUID 字节，worker-local 编号可以转换为物理身份。本步复用该
现成事实，不通过主机编号相等假设解决专用进程 CUDA_VISIBLE_DEVICES 重映射。

### 正确性状态表

新增十五项方法测试使用微小文件/原生事件 fixture：冻结 config/内容身份、
四类源的优先级与 footprint、错误 rank/身份、实际请求选副本、live counts、
membership 与 stale epoch、未知 load 保留、无可行集等待、实际利用率字段与
UUID，以及重复物理 GPU 不得冒充 scale-out。现有 worker 方法测试也核对 UUID
输出。没有生成模型工件、trace 或生产 profile。

首次十四项尝试中八项 fixture 缺少显式 `profile_id`，在进入被测路径前报错；
补齐为 `test-fixture-only`，没有给生产代码增加默认估计。首次完整回归 633 项
通过，随后加入物理 UUID 绑定及重复设备检查；最终 634 项功能检查通过
（23.069 秒），56 项独立安全/census/replay 检查通过（0.451 秒），无跳过或失败。
147 项旧结果保护清单与原计划 SHA 不变。所有本轮数据是正确性检查，不是
G1/G2 性能点，因此以本状态表交付，不画没有模型实测支持的收益曲线。

### 不得跨越的结论边界与下一步

现在真实请求会产生 `ieee_predecision_received_view_v1`，包括所有候选类/估计、
owner/epoch、source 和 utilization 的采样事实。该记录明确
`selected_reference_acquired=false`：它证明“按哪些已收到的事实选目标”，
**不证明随后获取引用时源未变化**。现有 selected acquisition 仍重新查询，
完整 source revalidation/reselection 与 actual admission class 尚需合并。

下一步在同一真实请求边界完成所选源保护/冲突重选、接纳时固定观测类和
ServiceIntervalObservation；源变化不能沿用旧类给 EWMA 记账，稍后才拿到 GPU
引用不能追溯声称 admission 时 D=0。并继续全层物理预算及生命周期所有权。
没有经过这些边界和真实模型整合资格，不运行/宣称正式 Full、LastKnown 因果
消融或新 G1/G2 主结果。后续不再重复本轮各独立组合测试作为新的实验进展。

## P1-D25：将所选源保护与接纳时观测类接入实际请求

### 缺口、依据和决定

`e7a3b1e` 已使路由使用决策前状态，但之后的 cached-only acquisition 可以在
同一副本上重新发现另一层。若保留路由时的类来统计，会将 HOST 加载误记为
GPU 命中；若先加载 HOST 再开始接纳计时，又会把准备时间移出论文的 D。
本步在既有请求、原生引用、文件 owner 和事件回调中接通边界，不替换公式。

依据 [vLLM 0.30 原生 LoRA 加载路径](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/worker_manager.py)，
LRU loader 在 ID 已注册且未启用 inplace 时复用 CPU adapter，再进行 activation；
原生 engine-core 在线程内串行调用该路径。因此 HOST 来源可以先仅保护 CPU
cache 条目，再于接纳后加载 GPU。依据其
[LRU pin/unpin 接口](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/utils/cache.py)，
共享引用需区分本请求借用的 pin 与外部原有 pin，不能用统一 unpin 回收别人的保护。
不把这类引用管理包装成论文的新算法。

### 论文规范与实现证据

| 论文规范语义 | 当前实际请求路径 | 尚不能据此声称 |
|---|---|---|
| 选择后重新确认源 | GPU 用所选 owner/epoch 进行仅 GPU 的引用获取；HOST 用同一身份只保护 CPU 源；文件以 owner/epoch/content 获取读引用 | 跨所有进程、所有层的全局同时采样或原子事务 |
| 冲突重选 | 已知无副作用的 epoch/tier 冲突释放本次计数，并重新运行整个 Router；保留各次收到的视图与拒绝原因 | 未知 RPC 结果可当成无副作用冲突；未知结果仍保留所有权并撤回副本 |
| GPU 的 D=0 | 先保护可执行副本，随后提交接纳时刻；此前等待计入 admission wait | 迟到的 GPU acquisition 可追溯成接纳时已就绪 |
| HOST 的 D 包含准备 | CPU-only hold 不调用 loader/fence/GPU pin；接纳后才激活，按原生 acquisition 时间完成 D | HOST 来源保护等于 GPU-ready 或通过完整 HOST 物理预算准入 |
| 观测类在接纳固定 | 保护回执后用当前 admitted/pending 数重新确定 load bin；其余 source 表示不变；缺 profile 仍拒绝 | 路由旧 count 可覆盖接纳时的真实 count；fixture 可以充当生产 profile |
| 只更新完成区间 | 实际 generate 接收 NativeServiceIntervalObserver；首末 token 事件立即更新对应类；最终 token/时间须与回执一致 | 取消能补出未完成 O；仅完成通知可替代首 token |
| 文件引用覆盖读入 | HOST/NVMe 使用已验证副本；Remote 复用原 HTTP materializer，实际发布后取得同 owner 的引用，再加载 | 目录存在、旧 `_nvme_cache` 或本地 frozen fallback 可替代真实远端/内容确认 |
| 已持有状态不虚假释放 | HOST 释放与 GPU 引用分别确认；未知 hold/load/release 回执保留 controller/native 所有权 | 已发取消等于 CUDA 工作已结束或整卡已释放 |

文件/Remote 接纳前另复查选中副本的原生状态；若已出现更快的原生副本则重选。
这仍是各 owner 的已接收状态与所选有效源保护，不声称全局线性化的“最快层”。
下层副本受保护期间可以出现更快副本；接纳类固定，后续真实 acquisition 时间
决定 D。只有 guarded GPU admission 才使用 D=0。

IEEE 请求不再进入旧模拟加载/旧协调 resolve 路径。尚未接入的论文 E(t) 与
全层物理预算明确标记 `physical_capacity_qualified=false`，不能因此将这条
load/reference 路径作为已合格 Full 的性能结果。其它 legacy 实验路径未改名。

### 正确性状态表与失败保留

新增 16 个测试方法（多层、释放顺序和外部 pin 使用 subtests）：

- 原生 HOST-only hold、无加载/无 fence、共享 CPU/GPU 引用两种释放顺序、外部
  pin、GPU-only 撤回、CPU 非法撤回、stale/tier 冲突和 lease 身份冲突；
- 实际 runner + router + 引用 owner + 原生事件 fixture 的 GPU/HOST/文件 HOST/
  NVMe/Remote 全请求，包含 GPU→HOST 冲突重选、Remote→GPU 更快副本出现、
  文件 epoch 冲突、实际 admitted bin、hold/release 丢回执、接纳后取消、解码
  途中取消，以及最终回执时间与事件不符的拒绝。

首次 19 项定向运行有四个 fixture 初始化错误：测试未提供 routing_identity
必须的 config bytes。补齐实际微小配置输入，未在生产方法增加默认值。随后
10 项定向检查通过，再补一项解码取消检查。最终完整功能回归 **650 项通过**
（23.451 秒），独立安全/census/replay **56 项通过**（0.500 秒），无失败或跳过。
使用已有稳定环境、CPU/内存受限范围，无模型推理；HTTP 输入为微小内存响应，
不是 174 的真实链路资格。本节是正确性表，不生成收益图或生产延迟 profile。

### 回到主线

下一步不重复以上孤立检查：对现有模型资格入口接入这些真实边界并取得 native
整合证据，同时完成全层物理 admission 与 GPU 生命周期。特别保留两个已知边界：
原生 name/path 身份目前不能把同一已验证内容在 HOST/NVMe 的不同路径无条件
互换；接纳后原生容量冲突还需与真实 owner 的等待/唤醒协调。不能通过删除身份
校验、固定 sleep、猜测释放或旧模拟路径兜底解决。全池数值正确性、真实 profile、
远端磁盘门槛、SLO 标定、Serverless 优先的 baseline 对照和正式矩阵仍未完成。

## P1-D26：先测量，再初始化服务成本

### 本次问题与明确边界

D25 已把接纳边界接到真实方法，但此前只有受控事件测试。直接运行 Full 又需要
真正测得的初始化 profile；用测试常数填入路由器会形成循环论证。因此复用现有
`backend-model-check`，增加显式 `native_source_intervals` 资格方式。它调用原
ScenarioRunner 的 source protection / preparation 方法和原生 token observer，
但只选一个真实 worker，不创建虚构 Router、planner 或零延迟 profile。

`ServiceIntervalObservation.for_profiling` 仅收集已发生区间，不提供 estimate；
普通构造仍要求支持该观测类的 ServiceCostModel。生产路由不调用此资格方式。
九个行间公式、EWMA、原生 LRU/批处理及模型配置均不因这次测量而改变。

本次使用原 7B trace 的前 32 条、原权重和 adapter。历史前缀在第 21 条开始出现
原生 HOST 复用，故不增加人工驱逐。原生缺失时明确先 load/release，再开始这条
GPU/HOST 边界测量；priming 回执保留，不能当成 Remote 加载为零的主实验。
这是逐请求资格回放，不遵循主实验 open-loop 到达，亦无扩缩容或共同 SLO 结论。

### 依据与验收

核对 [vLLM 0.30.0 原生 LoRA manager](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/worker_manager.py)
的 CPU cache 复用/原生激活边界，以及
[AsyncLLM 原生输出](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/async_llm.py)。
本次不改官方调度策略，只检查已有缓存副本在接纳时的保护及实际 token 事件。

必须同时检查：固定原生输出数量、源身份、HOST-only 引用、GPU D=0、HOST D 的
非负顺序、事件与最终回复一致、D+T+O 恒等式、引用归还及实际 worker 退出。
原始 footprint/rank/prompt/output/admitted count 保留，资格用粗分桶不自动成为
正式冻结分桶。首次 JIT、串行并发度一、Remote/NVMe 缺失与样本覆盖限制必须
保留；不能把这些观测直接宣布为完整生产 profile 或系统性能收益。

新增三项确定性检查，GPU/HOST 子案例覆盖显式 profile-only 接纳、无初始化
估计、生产入口不接受空 model、GPU 零区间与取消不补 O；26 项定向检查通过。
真实模型与最终完整回归结果另列执行状态和 P2 资格表，未通过前不预填成功。

## D27 — 物理 GPU 所有权与实际退出（计量，不改九式）

历史 instance billing 的 ready/idle/shutdown 记录不能直接作为 G1 的物理占用。
在既有 dedicated subprocess 生命周期中增加真正的卡分配：分配者锁定物理 UUID，
子进程启动前得到对应 CUDA 可见集；实际 worker 返回 UUID，NVML 独立核对所属。
请求完成、LoRA 引用归还和物理卡归还是三个不同边界，不互相替代。

第一次真实 7B 检查完成 4/4 条目标输出，但在 shutdown 返回时仍检测到 GPU 上下文，
因此保留未关闭租约，未输出完整 GPU-s。旧退出顺序在收到停止应答后立即 TERM；
修正为先等待原生退出，用按出生身份固定的 pidfd 接收退出通知，再作原生 GPU 核验。
60 秒收尾上限与既有 TERM/KILL 保护仍成立，不添加等待 sleep 或伪造 release。

局部互斥锁不是集群资源管理器，测量范围仅为受保护服务内的独立 runtime；
同卡共享逻辑实例不得重复分配，TP 的多 UUID 分别计数。直接进程内 engine 和
外部 baseline 的分配者仍须各自接入，不能据此宣布 Full 生命周期全面合格。
详细状态、实际失败和修正运行见 `PHYSICAL_GPU_MEASUREMENT.md`。

## D28 — 原生容量冲突与真实引用释放事件

### 因果问题与选择

D25 已把源保护接入请求，D26/D27 分别验证源区间和整卡生命周期；但原生
`all_gpu_slots_pinned`/`all_cpu_entries_pinned` 仍直接成为请求失败。假设是：
对确由本服务在途引用造成的暂时容量冲突，等待实际引用释放、再重新观察并
执行原生加载，能够保持物理保护且避免不必要的失败。不是等待固定毫秒数，
不是 OOM 后重试，也不通过改变 IEEE 公式或 native LRU victim policy 实现。

本轮核对 [vLLM 0.30 worker manager](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/worker_manager.py)
的单线程 core 调用及 CPU cache/activation 路径，和
[原生 cache pin/unpin](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/utils/cache.py)。
原生 worker 不能阻塞等待尚需同一线程处理的 release RPC。因此在拒绝回执中
返回实际阻塞租约，由异步 controller 等待其完成；采用
[Python 3.12 asyncio.wait](https://docs.python.org/3.12/library/asyncio-task.html#asyncio.wait)
等待具体引用，不取消被等待的 owner，不引入周期轮询。

### 规范与当前实现

| 论文规范语义 | 实现与验收边界 |
|---|---|
| 不驱逐在用副本 | 原生回执列出每个候选的 GPU/HOST 引用及外部 pin；不改变原生 LRU |
| 释放后动态重检查 | 每个租约在发送获取前登记完成事件；一个候选的全部引用确认释放后，重新获取 native snapshot，再提交新 epoch |
| 不丢先到事件 | 完成见证保留至本次 runner 生命周期结束，迟到的冲突回执仍可看到先前释放 |
| 唤醒不是许可 | 其它请求先占用槽位时，再按新的真实阻塞引用等待；不把唤醒当作容量预留 |
| 取消不伪造释放 | 等待者取消不取消引用 owner；丢失获取/释放回执仍保留所有权并撤回副本 |
| D 包含接纳后的准备 | 已固定 HOST 类不改为 GPU 命中；容量等待起止保留在 acquisition 之前 |
| 外部持有不能猜测完成 | 没有任何完全由已知其它请求持有的可释放候选时，显式报错；不解除外部 pin |

这不等于全层物理字节 reservation 或论文主动 E(t) 已接通。源路径内容身份迁移、
KV/全层接纳、主动 planner/handoff 和正式 Full 的整合仍然未完成。

### 资格协议与预运行状态

无需 GPU 的定向检查覆盖共享引用最后一次释放、先到释放、取消、未知释放、
HOST CPU 容量、外部 pin、两等待者重新竞争和实际 Full 请求的 HOST-D 区间。
新增资格入口也由真实 runner/reference helpers 的微小 fixture 覆盖。共增加
10 项测试，最终功能回归 **676/676**、安全/census/replay **56/56**，无跳过。
过程中保留一个 fixture 错误：试图改只读 LRU capacity 属性；删除无必要的
fixture 容量修改后通过，生产策略没有因此放宽。

下一次真实 7B 检查只用旧 seed42 前32条中按出现顺序的前
`native_slot_count+1` 个不同 adapter 请求；不生成负载，不更改4个原生槽位。
先保护4个请求的实际槽位，第5个请求在第1个真实 decode 期间尝试加载。
只有确认确已进入容量等待后才归还第1个请求引用；第5个必须事件唤醒并完成，
随后完成所有选中请求并归还引用。无 sleep 注入、无性能排名、无完整 Full 声称。
复用受限 dedicated worker 与物理分配者、原有 native token 计量及外置 watchdog。
模型结果尚未预填；输出至新的 capacity-wait attempt，再交付状态表。

| 实际尝试 | 请求执行 | 状态 | 后续决定 |
|---|---:|---|---|
| capacity_wait_attempt1 | 0；模型未启动 | launcher 漏传 `FAASLORA_TC_NVML_SHA256`，受控 gate 明确拒绝；所属 gate 进程已释放 | 保留原回执，补齐已验证组件 SHA 后以 attempt2 执行相同诊断；不算模型失败或性能点 |
| capacity_wait_attempt2 | 5/5，755 native tokens | 真正4个槽位均持有时第5个请求等待3764.347ms；第1个请求完成并确认释放后唤醒，全部引用归还 | 已完成本问题所需 native 验证，不继续重复；回到 Full 接纳整合 |

### 实测结果、解释与主线归位

第5个请求 `req_00008` 的等待结束比 `req_00000` 最后一个 native token 晚
15.072ms；这是含终态/释放回执传播的时间，不称纯 CUDA fence 开销。等待时
原生快照确认4个 live leases；结束为0，随后 adapter caches 清空。选中请求
为00000/00002/00005/00007/00008，目标152/123/256/174/50全部匹配。
与 D26 source32 对应请求的 prompt、native input 和 output SHA 全部一致。
这不解决更早 req00005 与旧100前缀之间的差异，也不是独立数值正确性证明。

native TTFT/TPOT/E2E 重算误差均0ms；该诊断没有主实验到达时序，不能用来
报告完整用户 TTFT 或新 G1/G2 收益。真实物理占用64.852 GPU-s，含末 token
之后7.742s退出；正常退出码0，外置 NVML 证实上下文释放。77次资源采样，
服务峰值5699035136 bytes，high/max/OOM事件均0。两个所属辅助 scope 均在
确认空后停止，用户显示进程未动。全部147个保护项和源 plan SHA 保持不变。

实现检查点 `33ef68d0a1cd566211da91892f838f4a339cfc92` 在模型运行前推送并核实；
CSV/JSON状态表位于 `paper_results/ieee_tc/p2_backend/20260926_7b_capacity_wait.*`，
包含启动失败和成功尝试、原始记录/监控/所有权日志 SHA 与实际执行源码 SHA。
CPU cache 等待、多个等待者、取消和未知释放本轮只有 CPU fixture 证据；原生
模型证据只覆盖一个GPU-capacity等待者，不声称全路径容量资格。

下一个未完成项是实际 Full 的 KV/物理预算 E(t) 和代表性实测 profile 整合，
并完成内容绑定的跨路径源身份、主动规划/准备与生命周期聚合。不再重复本轮
5请求、D26 source32、D27 lifecycle4；正式矩阵、SLO/Resident标定与 Serverless
优先的 baseline 顺序不变。

## D29 — E(t) 接入原生 HOST→GPU 准备事务（CPU 合同通过，Full 仍未合格）

### 问题、历史证据与选择

D4 的式(8)/(9)计算器没有原生资源提交权；D7 的 KV 观察与 worker 观察各自
正确，但两次 RPC 不能提供同一次接纳事务。D28 已证实实际请求引用会阻止
原生 LRU 替换。因此本轮只解决一个问题：已有原生 HOST 副本向预分配 GPU
槽位提升时，让当前 KV/压力决策与受保护的替换发生在同一 owner 边界。
没有改论文公式，也没有把请求按需加载改走主动 E(t)。

核查的原始实现：
[vLLM 0.30 engine-core utility](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/core.py)、
[UniProc executor](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/executor/uniproc_executor.py)、
[native LoRA manager](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/model_manager.py)、
[dense slot copies](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/layers/base_linear.py)、
[packed column copies](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/layers/column_parallel_linear.py)
及 [LRU order/pins](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/utils/cache.py)。
这些源码与已安装0.30路径交叉检查；UniProc 的同步 utility 在 core 所在线程
调用 worker，不改变 AsyncScheduler 的原生调度、抢占或 block 分配策略。
[ELORA](https://arxiv.org/abs/2505.03756)说明 LoRA/KV 竞争值得独立处理，但不能
据其结果假设本项目的 admission 必然有性能收益。

### 论文规范语义与实现证据

| 论文规范语义 | 当前实现及明确边界 |
|---|---|
| 当前 KV、完成长度窗口、iteration pressure | core utility 现场采样，长度均值由成功 native completion 更新；取消/错误不更新，空桶使用显式冻结 profile |
| 评估与提交之间不被新调度穿插 | 精确限定0.30 UniProc、TP/PP/CP=1；同步 core→worker 调用，copy fence 完成后返回；不是把异步推理改为同步 scheduler |
| 复用容量不重复算物理增量 | 读取实际 uniform slot pool、device free/total；整个预分配 pool 已计 used，已有 HOST 的直接 slot copy 增量为0 |
| workspace 必须有证据 | 只接受已审计 native setter、TP1、同 dtype、连续 pinned CPU tensors及连续GPU目标切片；转换、未知 setter、packed expansion或非连续目标拒绝，不把任意模块 workspace 填0 |
| 替换基于 proposed after-victim state | 不触碰 LRU 顺序地读取原生 `order`，仅将实际将被替换的未pin victim计为可复用；成功后核对真正被移除的 ID |
| deferred 不改变缓存 | 先计算，再按原生 loader 执行；延后时无加载、驱逐、LRU touch或额外引用 |
| Full/CapacityOnly 只差软检查 | 共用同一原生 victim/loader/引用/完成fence；CapacityOnly仍不能替换被引用或外部pin的槽位 |
| 成功发布必须可执行 | 返回持有中的GPU准备租约，只有显式release回执才解除pin；失败不发布ready，未知RPC状态沿用保留机制 |
| 重试不重复执行 | 相同prepare attempt返回相同决定；改变source/policy或复用已释放租约拒绝；新的决策必须有新attempt |

接口沿用现有 `InferenceEngine`、dedicated worker、proxy 与 worker extension：
`ieee_prepare_host` → engine-core utility → `proactive_host_prepare_and_acquire`。
启用必须同时提供 `ieee_gpu_references`、`ieee_scheduler_observation` 和
`ieee_admission_profile`。profile 必须含实际 `model_backend_id`、`profile_id`、
论文窗口 `window_s` 和覆盖全部冻结输入桶的 `profile_means`。**本轮没有生成或
填入任何正式 profile 数值，也没有启用正式 Full。**

### 无 GPU 证据与状态表

| 检查 | 结果 | 可支持的结论 |
|---|---|---|
| Full满iteration压力，CapacityOnly相同状态 | Full不加载/驱逐；CapacityOnly按同一LRU完成并持有引用 | 软策略差异与物理保护可区分；非实测性能收益 |
| 全槽位pin、真实LRU顺序与字典顺序不同 | 正确拒绝或选择原生未pin victim | 不用假空闲或另一替换算法制造差异 |
| 陈旧epoch、跨进程快照、source/policy复用、失败copy | 明确拒绝或使owner失效；不发布ready | 不以兜底加载绕过证据缺失 |
| 成功/取消的完成长度窗口 | 成功更新所在桶；取消不更新；到期恢复显式profile | 没有未来长度、跨桶代替或新EWMA |
| 真实回环dedicated RPC | 观察、准备决定与后续native事件传输均通过 | 入口已接通，不是仅孤立公式测试 |
| 全功能/安全回归 | 691/691、56/56，无失败/skip；新增15项 | CPU/native-cache合同，不等于模型资格 |
| 旧D28 HOST/GPU布局离线复核 | 两种HOST布局均连续；HOST rank8但GPU pool max-rank64 | GPU目标B切片不连续，不能据CPU连续性声称零workspace |

按计划11.2使用状态表，不制作性能图。本轮未运行模型、未新增数据/权重/负载。
源plan和147项保护内容零变化；模型GPU保持15MiB/0%。未修改baseline仓库。

备份前进一步核查发现一个实质限制，已修正初版检查而没有启动模型冒险试错：
本环境Torch源码版本 `cf30153c4c131c8164ee7798e5022d810682e2cb` 的
[CUDA copy 实现](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/Copy.cu)
在非连续CPU→GPU路径使用临时GPU张量。D28日志中原生pool的B布局为
`[4,1,4096,64]` 等，注册adapter rank均为8；实际目标切片stride为`(64,1)`，
不是连续`(8,1)`。增加真实Torch meta tensor检查，无GPU/CPU数据分配。
因此当前零workspace分支**不能放行这些rank8→rank64的准备**；下一个实现项
必须覆盖其真实workspace或采用有证据、不增加隐式临时分配的等价拷贝。
不能填一个经验MB值、降低rank配置逃避问题，或把未知workspace当0。

### 尚未解决、下一主线

这个事务**只覆盖已经物化的 native HOST→GPU**。回执明确
`admitted_scope=native_unfinished_requests_only`、
`transfer_scope=serialized_native_host_to_gpu_only`、
`all_tier_admission_reserved=false`、`production_launch_authorized=false`。
native加载由同一线程串行且返回前完成，故该局部边界没有前一native transfer；
不能将这里的0活动传输冒充整个系统REMOTE/HOST准备压力。

下一步先闭合上述真实目标布局的workspace合同，再把控制器已接纳但未ADD的请求，与native请求作唯一身份交接，纳入同一
KV需求集合；把其他层transfer事件与预算接入，并连接实际planner/handoff。
同时完成代表性实测profile与内容绑定的跨路径来源。不能拿native范围的检查
替代Full所有层的资格，不能只打开新profile开关就继续使用旧warmup启发式。
原生模型验证必须在这些接口形成一个有意义的Full准备路径后进行，不重复
D26 source32、D27 lifecycle4或D28 capacity5。正式M1/M2、baseline及消融均未开始。

## D30：rank-sliced HOST→GPU 的显式 pitched-copy 路径

### 假设、依据和范围

D29发现的不是“显存预留值偏小”，而是PyTorch的非连续目标拷贝需要临时GPU
张量。为此增加一个明确的准备拷贝策略：保持原生vLLM0.30 reset/setter、LRU
victim、slot布局及rank64配置，使用`cudaMemcpy2DAsync`直接写入rank8目标矩形。
不重新训练或生成adapter，不改变九个公式，不新增经验workspace常量。

依据为当前Torch精确版本的Copy.cu（链接见D29）、
[CUDA13二维异步拷贝定义](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__MEMORY.html)、
[vLLM0.30原生linear setter](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/layers/base_linear.py)
及同版本merged setter。CUDA接口使用实际源/目标行距、字节宽度、行数和当前
stream。已安装cuda-bindings13.4.3提供该接口，未安装新依赖或修改原环境源码。

使用线程局部、限于一次native准备的TorchDispatchMode接管经过验证的copy操作，
不全局替换Torch/vLLM方法。所有预期源和目标地址、形状、stride、dtype必须
匹配，每个矩形恰好写入一次。未知/重复/遗漏copy均失败；没有copy_兜底重试。
零填充仍调用原生reset，缺失packed子模块保持原生语义。强引用保持CPU/GPU
视图直到所捕获stream的fence完成，包括异常退出；随后沿用原生引用确认。

| 论文规范语义 | D30当前实现证据 |
|---|---|
| workspace有实际依据 | 消除该已知路径的临时张量，而非把未知workspace填零；真实CUDA资格待下项 |
| Full/CapacityOnly物理策略相同 | 两者使用同一准备loader和原生LRU，只有软E(t)检查不同 |
| 普通请求加载不暗改 | demand和preparation显式分离，未安装准备loader不得回退到普通copy |
| 成功发布必须可执行 | 完成全部预期矩形并fence后才获得GPU引用；错误使事务失效 |
| 不把底层测试当完整资格 | 699项功能、56项安全检查通过；无模型/正式性能结论 |

新增8项CPU检查包含真实ATen局部dispatch、原生reset表达式、padding/其他slot
不变、CUDA错误/异常、未预期/重复/遗漏写入、stream切换及capture拒绝。
其中DMA以CPU mock验证控制合同，不冒称真实GPU验证。

下一项使用现有preflight的`backend-copy-check`，在原安全资源域运行一次真实
CUDA/native-setter检查：普通linear与包含缺失子模块的merged，rank8/maxrank64，
非零内存测试图样，不加载backbone、不新增权重文件或trace。比较全slot内容SHA、
零填充和额外GPU tensor峰值。它只回答拷贝问题，不是再次运行模型前缀，也不是
Full/performance资格。检查后交付状态表，再回到controller/native KV身份交接、
全层transfer/budget与planner/handoff主线。

### D30真实GPU验证结果（已完成，不再重复本微测）

执行代码`21e80f9b88fc32f9004c842c93da79b53d85af39`已先推送并核对远端SHA。
第一次调用未满足既有launcher的32位hex辅助scope命名、绝对路径要求，未进入
服务/CUDA；tmux早退未保留traceback，故保存的是实际调用和源检查依据，不伪造
原始异常。第二次按原入口规范执行成功，没有修改测试条件或重试GPU错误。

| 实际原生setter布局 | 原copy额外tensor峰值 | pitched copy额外tensor峰值 | 全槽位/零填充/非目标槽位 |
|---|---:|---:|---|
| linear，rank8/maxrank64，width4096 | 65,536 B | 0 B | 与预期、原生copy逐元素及SHA一致 |
| merged含缺失中间子模块，width4096/11008 | 176,128 B | 0 B | 与预期、原生copy逐元素及SHA一致 |

成功运行仅一次。scope正常退出0、12次外置采样，service peak729,321,472 B，
high/max/OOM事件均0，NVML终态无本实验上下文、服务scope已删除。没有生成token、
加载backbone或创造新LoRA文件。两个拷贝策略在所测完整pool内容上完全一致。

这里的0是**PyTorch额外GPU tensor分配峰值**。不能由此宣称任意CUDA内部资源
瞬时占用为0，也不能声称显存减少量等于reserved allocator arena变化。原生linear
先执行，增加2MiB allocator reservation；后续复用不能解释为稳定节省2MiB。
这不是时延优化实验证据，更不是Full/SLO/主比较资格。

按计划11.2交付此状态表及
`paper_results/ieee_tc/p2_backend/20260926_pitched_host_copy.{csv,json}`。
所有4行及来源SHA核验后归档；回到完整KV集合与实际主动准备集成，不继续
扩张这个局部微测矩阵。后续只有新的集成路径问题才需要新增针对性验证。

## D31：controller-pending 与 native KV 需求的唯一交接

### 问题与判断

D29只对后端已经收到的请求计算预测KV；控制器已经预留副本、但仍在准备
adapter的请求尚未进入native ADD。直接把这段时间当作零需求，会让主动准备
占用本应考虑的余量。这是观测集合不完整，不是九个公式需要增加经验系数。
D30已经解决rank-sliced准备拷贝的已知临时张量问题，本轮不重复其GPU微测。

对照[vLLM0.30 input processor](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/input_processor.py)、
[native frontend](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/async_llm.py)
及[Request转换](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/request.py)，
外部request ID会被原生随机化；EngineCoreRequest的external ID不会自动保留在
scheduler Request中。因此不能靠字符串前缀、cache salt或trace header猜测交接。
当前实现保留原生ID随机化，显式在ADD之前绑定两种身份。

### 已接入的调用链与不变量

1. 实际runner预留副本后、保护或准备所选source之前，生成一次性intent ID。
   实际engine/专用worker使用原生input processor得到prompt token（包含special
   tokens），登记数量、SHA、声明output limit和adapter ID；不使用trace token hint。
2. core在线程内登记的确认，是pending需求开始纳入同一快照的边界。该预留
   仍可能因source冲突被撤回，故是保守的pending集合，不冒称物理KV分配。
3. native frontend获得原生随机内部ID后，核对实际prompt SHA、长度上限和
   adapter，再将intent绑定到该ID；绑定/传输期间pending仍计入预测。
4. scheduler真正执行ADD时，在同一线程中交接：pending退出，native unfinished
   request接替。不是发送成功就移除，也不是两个集合长期各计一次。
5. 准备前取消需要core withdrawal确认，并留下不可复用的tombstone，防止迟到
   registration复活。已经绑定/ADD的请求不能假设不存在，必须走原生retirement
   和既有异步KV fence。未知结果保留所有权，不返回空闲容量。
6. Full/CapacityOnly仍用同一个集合和物理策略，仅原有软E(t)判断不同。pending
   请求尚未取得GPU adapter引用是合法状态；native请求仍必须持有可执行引用。

`admitted_scope=controller_pending_and_native_unfinished`区分新的快照范围。
pending请求的generated/reserved positions为0，未处理prompt为真实原生长度。
输出预测仍使用原完成窗口/冻结profile和声明上限，不读取未来实际长度。
原物理block allocator、demand加载、LRU、后端调度、完成窗口均未替换。
快照只遍历当前pending索引，不扫描已终态历史来重建活跃集合。

### 正确性状态表（不是性能图）

| 检查问题 | 当前证据及边界 |
|---|---|
| 准备期间是否漏计 | 原生token descriptor进入同一E(t)；确定性例子预测KV由0变500 B并正确defer，公式未改 |
| 发送到ADD之间是否双计/漏计 | 实际core utility与scheduler hook测试：bound仍计入，ADD后恰好一次native需求 |
| 内容/limit/adapter是否可偷换 | BOS差异、token SHA、limit、adapter差异均在原生发送/ADD前拒绝 |
| 取消是否允许迟到复活 | 登记中取消、先撤回后迟到登记、重复绑定、未关闭生成、丢失确认均覆盖 |
| 是否仅存在独立ledger | 实际runner HOST路径、engine方法、proxy、专用worker和真实loopback TCP控制/生成标识均接入并测试 |
| 后端API是否对应实际环境 | 已安装vLLM0.30 frontend/scheduler导入与方法签名检查通过，CUDA不可见、未加载模型 |
| 当前回归 | 717项功能检查、56项安全检查通过，无失败/跳过；新增18项pending检查 |

注册阶段调用一次原生预处理，正常生成继续原生输入路径，并严格比对实际
descriptor；这不是新的tokenizer或未来信息。其控制开销必须计入后续完整运行，
不据此声称TTFT改善。以上CPU/loopback结果不是新的真实GPU/Full资格。

### 下一主线

当前只闭合HOST→GPU事务使用的KV集合，**all-tier transfer/budget、实际
planner/handoff接入、代表性实测profile、跨路径内容身份及完整部署生命周期
仍未全部闭合**。继续这些集成项；形成有意义的Full路径后才做原生整体资格。
不重复D26 source32/D27 lifecycle4/D28 capacity5/D30 pitched-copy微测，不新增
局部微测矩阵。M1/M2、正式baseline、消融和敏感性均未开始。

## D32：真实文件层迁移与同一容量约束

### 主线问题、既有证据与本轮边界

D23的真实HTTP传输已经在写入前预分配archive与payload空间；D25保护并绑定
所选副本的文件内容；D29–D31处理native HOST→GPU及KV需求。代码历史中的
NVMe→HOST本地复制仍使用普通`copytree`，虽然发布内容经过校验，但临时新副本
没有同一物理文件预算。这会让规划的“剩余容量”和实际准备峰值不一致。

本轮可证伪判断是：**本地复制复用已验证下载的空间所有者，便能在执行时同时
约束旧副本、临时新副本和其他传输；无需改变论文的收益或预算公式。**
这里的约束仅是受管regular-file分配字节的子预算，不是整个HOST内存。

参考依据与范围：

- [ELORA论文](https://arxiv.org/abs/2505.03756)明确关注LoRA/KV使用依赖和换入换出；
  这支持检查真实共存资源，而不意味着本轮已经复现其算法或全部内存管理。
- [vLLM0.30 worker manager](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/worker_manager.py)
  中native CPU LoRA对象不是本地PEFT文件本身，不能把两者混为一份HOST占用。
- [POSIX预分配接口](https://docs.python.org/3/library/os.html#os.posix_fallocate)
  支持先取得真实文件空间；不以稀疏truncate或磁盘剩余量快照冒充预留。
- [Linux tmpfs说明](https://docs.kernel.org/filesystems/tmpfs.html)说明tmpfs可涉及swap，
  因此文件已分配不等于所有页面此刻驻留DRAM，更不等于整个服务RSS。

### 实际接入与不变量

1. 远程archive+payload与本地payload-only共用`_prepare_file_allocation`。
   本地复制不虚构archive；按实际文件大小与文件系统allocation unit预分配。
2. 复制前取得原内容身份的读引用，复制、校验、发布、清理结束后才释放。
   文件正文复制在所有者锁之外执行；所有文件已固定大小，不允许增长。
3. 发布前校验源身份、目标每文件SHA与签名；复用原有替换/恢复和读者保护。
   引用中的目标不可被覆盖；失败不退回普通copytree或改变容量阈值。
4. `local_file_budgets`提供文件子预算快照：实际预分配已在used中，不再重复
   加一份reserved；快照本身不预留容量，真正执行仍重新检查/取得空间。
5. 实际ResidencyManager确认路径与runner本地准备入口已接入。异步复制复用
   HTTP的取消等待规则：取消await不等于停止写线程，必须等待实际搬运退出。
6. 结果增加`local_artifact_transfers`，保留源确认、transfer ID、预算、字节、
   共同本地时间域、发布/失败和读引用释放。清理失败的真实路径继续占预算；
   错误链类型保留，不写入潜在敏感异常文本。

旧未确认路径仍是旧协议，不能仅凭目录存在成为IEEE confirmed命中。若源在
旧路径检查后变成confirmed，禁止它通过未预分配的复制分支，不静默绕行。

### 正确性状态表（微型文件测试，不是模型性能结果）

| 问题 | 实际检查与结果 |
|---|---|
| 本地准备是否虚构archive或少算payload | 两个微型文件占8,192 B，没有archive；准备中used=8,192、remaining=0、额外pending=0 |
| 两个方向是否保持内容 | 实际runner异步NVMe→HOST与HOST→NVMe，内容SHA一致；反向替换旧8,192 B时预分配峰值16,384 B，完成后8,192 B |
| 旧副本是否仍计入 | 16,383 B预算能保留一个副本，但拒绝第二份替换暂存，原副本和epoch不变 |
| 是否与别的传输共享额度 | 8,192 B预算已有4,096 B暂存时，新的8,192 B复制被拒绝，不能各自花一遍remaining |
| 取消是否提前释放 | 实际异步入口重复取消后仍等待写线程；期间源不可驱逐，退出后无残留引用/活动传输 |
| 错误是否变成命中 | 发布前损坏被拒绝；旧目标有读者时不得替换；源确认保持 |
| 清理失败是否虚增空闲 | 故障注入后残留8,192 B仍计入used，后续准备不能使用这份空间 |
| 实际HOST文件系统是否支持 | 同17项确认/复制检查在本机`/dev/shm`的tmpfs上通过；不据此宣称无swap或整个HOST资格 |

新增7项检查，扩展既有损坏与双向复制案例。最终完整功能回归和安全回归计数
记录于EXECUTION_STATUS；本轮不加载模型、不跑GPU微测、不新增LoRA/trace。
表中8,192等数值仅属于微型正确性fixture，不填入模型profile或论文性能表。

### 返回主线

本轮只闭合文件层复制的执行预算。native CPU tensor与文件/page-cache的总HOST
预算、跨路径内容身份、所有传输的实际压力、主动planner/handoff调用及实测
class profile仍需集成。已有旧preload流程调用预算化复制，不等于该流程的旧
planner已成为IEEE Full。保留这一区分，不因为局部检查通过启动正式M1/M2。
下一项应连接完整主动准备与物理所有者，不再扩展本地复制微测矩阵。

## D33：实际文件准备活动进入原生准入压力（非 Full 性能资格）

### 主线问题与依据

D29的原生HOST→GPU操作在同一调度线程内串行提交并等待完成，因此在下一次
准入决策时，没有上一笔未完成的该类复制。但D32的远端获取及HOST/NVMe文件
搬运可以同时发生；把loading pressure固定成0会遗漏它们。本轮假设是：将
这些实际操作的开始与结束纳入同一调度所有者，即可让原有E(t)公式使用非零
并发准备状态，而不改变公式或注入模拟等待。

核对了既有D29–D32代码与证据、IEEE定义，以及
[vLLM0.30原生core的utility处理路径](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/core.py)
和[对应异步客户端](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/core_client.py)。
这里使用明确版本的内部接口，不称为稳定公共API。
[dLoRA的联合请求/adapter编排](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang)
提供相关设计背景，但不用于证明本次修改有性能收益，也未移植其调度算法。

### 接入范围与不变量

- 每个已初始化的目标副本持有唯一传输活动表，与其KV/iteration状态由同一
  native调度线程观测。配置必须显式给出正整数`transfer_limit`；这是论文的
  压力分母，不是文件空间预留，也不是本轮新实现的并发执行限速器。
- 实际远端获取和confirmed本地搬运：开始通知得到确认后才能执行；IO线程、
  发布及清理结束后才发送结束通知。记录adapter、源/目标tier、文件所有者、
  副本所有者、transfer ID及同一机器时间域。重复开始不重复计数，已结束ID
  不得复活。actual remote miss在既有逐adapter锁内，等待同一复制的请求不
  获得另一个实际搬运入口。
- 原生准入读取活动数和配置额度，按原式计算load pressure。原生HOST→GPU
  仍在当前线程内完成；没有把另一个线程的缓存快照冒充原子KV/准入快照。
- 取消开始/结束通知的等待者仍须等待应答收敛；开始应答丢失时发送终止标记，
  防止迟到开始重新增加活动数。结束无法确认时保留活动压力和未确定记录。
  不以超时猜测“传输已经结束”。旧所有者的不确定记录不能由新所有者清除。
- `adapter_transfer_pressure`保留状态及IO结果。若IO已完成、调用者在等待
  结束应答时取消，分别记录`operation_outcome=completed`和
  `caller_cancelled=true`，不把它变成成功请求。错误只记录类型，不复制凭据。

活动区间包含通知与IO启动/退出的交接开销，是保守的准备所有权区间，**不是
纯网络活跃时间、H2D时间或planner的d profile**。IEEE的d排除开始加载前等待，
不能以本轮区间、含排队的D、或者TTFT直接填入。

### 正确性状态表（微型文件/原生接口替身，不是性能实验）

| 检查 | 证据与解释 |
|---|---|
| 活动数与准入是否相通 | 同一原生owner快照中1个文件准备、limit=2，实际worker准入返回load pressure=0.5；fixture值不用于模型配置 |
| 开始、重复、结束与迟到 | 同ID重复开始仅计1次；结束后计0；结束标记先到时拒绝迟到开始 |
| 真实文件入口是否调用 | 微型HTTP下载与NVMe→HOST异步入口执行期间活动数均为1，结束为0；保留内容校验 |
| 取消是否提前宣布空闲 | 重复取消开始等待，不运行IO；取消结束等待，IO结果和调用者取消分开，均等待实际结束确认 |
| 错误与不确定性 | body失败、开始应答丢失均保留失败；结束失败保留非零压力；不匹配的副本应答不能清除旧记录 |
| 调用链是否完整 | 实际loopback TCP代理/专用worker检查新增开始/结束命令，同时保留既有prepare/pending/generation检查 |

新增10项检查，并扩展既有worker准入和TCP测试。最终功能、安全和实际安装
接口验证结果见EXECUTION_STATUS。未加载模型、未新增GPU实验、未改变工件或
trace。根据计划11.2使用状态表，不为正确性fixture绘制性能优势图。

### 未完成项与返回主线

此版本仅覆盖**已初始化目标副本拥有的文件准备**。共享文件对其他副本的
压力归属、engine初始化前的handoff，以及整个服务的共享资源压力尚未闭合；
缺少目标engine时显式拒绝，不回退到0。该日志也不证明磁盘/总HOST预算完整。
全HOST/native tensor容量、IEEE planner/handoff实际执行、论文replacement
victim选择、实测d及D/T/O初始化、Full生命周期仍需集成。当前native LRU不能
称为论文收益驱逐。后续从这些真实路径继续，不增加独立transfer微测矩阵。
正式M1/M2、baseline、消融和敏感性均未因本轮通过而获准启动。

## D34：直接观测准备成本 d，不以含排队的服务 D 替代

### 为什么这是规划接入的前置项

检查实际`_preload_full_stack`发现它仍调用旧混合priority和阈值warmup，尚未
使用IEEE的需求加权准备收益。已有`select_ieee_insertions/handoff`数学检查
通过，不等于实际入口已经使用它们。准备收益的实测输入也不能从旧服务D
直接取得：IEEE明确规定d从开始加载到GPU激活，排除加载前等待，而D从副本
接纳到首次取得可执行adapter，包含其中等待。D26的HOST D及D33的保守活动
所有权区间都不是这个d。

本轮假设：在真实加载入口记录开始，并与已有完成同步/可执行引用关联，即可
提供直接可验证的d边界。无需改变收益公式、插入sleep或推测准备延迟。
核对了D25/D26/D29/D33记录和
[vLLM0.30 worker manager源码](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/worker_manager.py)：
native CPU复用与从文件加载不同，不能以目录存在或入队时间代表加载开始。
[dLoRA](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang)
的请求/adapter联合编排提供相关背景，不用于证明本轮存在性能优势。

### 实际边界与复用规则

1. 原生加载开始在同一worker通过源/容量检查、即将调用loader时记录；结束为
   原有completion fence后的可执行引用时间。真正GPU命中没有加载起止，
   不把引用RPC耗时记成d；GPU的d=0仍是论文定义。
2. 真实HTTP开始在已获得逐adapter搬运资格、创建暂存空间之后、发起请求之前。
   包含此后的远端打包/传输/预分配/解包/发布，而非纯网络时间。发布时刻与
   native加载开始/完成在同一推理机时间域比较，不与服务端时钟直接相减。
3. 请求自己的remote传输记录通过实际异步下载入口传递，不从全局日志按
   adapter名称猜最近的一次。request ID、transfer ID、源表示、native owner/
   lease与路径共同保留，目标路径规范化为绝对路径。
4. HOST/NVMe起点为真实native加载开始。Remote起点为本请求实际HTTP加载开始，
   后续阶段间的等待仍包含在d中；只排除第一段加载之前的等待。
5. 文件已由另一个请求准备好，或实际加载前出现native复用时，明确记录
   `profile_eligible=false`、`d_ms=null`及原因，不生成虚假的零成本完整加载
   样本。错误时钟、路径/owner、缺失完成或不合法顺序被拒绝。
6. 实际IEEE请求完成获取后生成`preparation_interval`，原有D/T/O更新保持不变。
   该证据是后续class profile的输入，不自动成为已冻结的代表性测量，不声称
   HOST/NVMe读取绕过page cache，也不把native加载总区间称为纯H2D。

### 正确性状态表（不含新模型性能测量）

| 问题 | 检查结果 |
|---|---|
| D与d是否分开 | 构造边界admission=100、native开始150、完成153：D=53s，d=3s，加载前等待50s；仅为数学fixture |
| Remote阶段间等待是否被删除 | 构造Remote开始111、发布140、native开始150、完成153：d=42s而非只相加传输+native；加载前11s排除 |
| 真正请求链能否提供字段 | 既有实际runner、微型HTTP/file源、native缓存替身的HOST/NVMe/Remote请求均检查d+加载前等待=D，remote transfer ID一致 |
| 复用是否误作完整冷加载 | shared文件、native源改变或GPU复用均为明确的不可用完整样本，而非d=0 |
| 非法边界是否默默修正 | 不同时钟、错误路径/身份、缺失native边界、倒序发布均拒绝；不clip到0 |
| GPU命中是否被污染 | 原有protected GPU D=0和服务事件保持，未伪造一个GPU加载样本 |

新增6项检查并扩展既有真实runner/file/native事务测试。初次222项定向检查有
一个测试代码漏导入clock helper的NameError，修复该测试后完整740项功能检查
通过，无失败或跳过；未修改判定门槛。安全回归结果见EXECUTION_STATUS。

### 返回主线与仍未完成的内容

本轮完成观测入口，**没有完成规划策略接入**，没有采集新的7B/3B profile，也
不以fixture或旧D填补它们。接下来应按实际表示/layout/footprint建立冻结实测
class初始化，并把收益、预算与实际planner/handoff/replacement共同接入。
完整HOST/native预算、共享/激活前压力及Full生命周期仍在该集成范围内。
不要再重复source32/capacity5或创建另一套测量框架。正式主比较仍未开始。

## D35：从需求与准备成本快照构造 IEEE 规划，并阻止旧策略冒充 Full

### 依据与可证伪问题

核对计划P1/P3、IEEE准备收益段、D34及其实现提交`6bdff25`后，本轮问题是：
既有数学selector是否真正收到同一需求窗口、同一版本的准备成本、真实目标
footprint和剩余预算，而不是旧registry热度及混合priority？实际启动入口仍
调用旧策略，因此不能仅凭selector检查通过就运行Full消融。

重新核查了[vLLM0.30原始实现](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/worker_manager.py)
中CPU LoRA复用、文件加载和原生LRU的边界，以及
[dLoRA原论文入口](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang)。
来源用于核对真实加载表示和联合编排的已有设计；不据此宣称IEEE规划有性能
优势。准备成本仍严格使用D34的加载开始→可执行完成，不使用含初始等待的D。

### 本轮实际实现范围

- `PreparationClass`单独表示tier/representation/layout/size class；不把请求
  prompt/output/admission类直接当作准备类。其layout和尺寸边界必须由后续
  合格实测profile生产者绑定，类型检查本身不是测量真实性证明。
- `PreparationCostModel`从明确class初值初始化，完整加载按固定beta更新；
  一次snapshot在同一锁内取得全部class估计。新副本只继承冻结初值，不继承
  其他测试轮学习状态。GPU剩余准备时间为论文定义的0。
- 更新接受带源类别、实际起止、native owner/lease的D34记录，拒绝重复、
  改类、把D填成d或不支持的class。共享/改变来源的无完整样本记录不更新，
  也不产生0成本。beta是明确冻结的更新规则，不按正式结果选取。
- `ExperimentStack.plan_ieee_preparation`直接取实际HotnessTracker窗口，
  再调用`generate_ieee_epoch`：冻结输入、由`h*(d_source-d_target)_+`构造
  candidate，handoff用密度扫描，residency用GPU→HOST→NVMe条件选择。
  每adapter一个目标；目标footprint和各层剩余预算由物理owner调用方提供。
- plan SHA绑定来源快照ID、需求计数/时刻、profile ID/更新序号、全部option
  和预算。返回值明确`physical_resources_reserved=false`；这是决策结果，
  不是物理资源预留。空窗口无需估计无收益class；正需求缺实测类直接报错。
- 实际`ScenarioRunner._preload_full_stack`在IEEE routing模式下，先于
  `stack.start()`、缓存重置和旧background warmup拒绝未合格执行。历史路径
  保留给旧协议；不会将旧priority悄悄用作IEEE Full。

### 正确性状态表（构造输入，不是性能数据）

| 要验证的问题 | 检查与结论 |
|---|---|
| 真实窗口还是静态热度 | 实际stack入口：空窗口不准备；a/b各一次到达时h_a=1/2；过窗后回到0 |
| d与D是否混用 | fixture初值10ms、实际加载3000ms、beta=1/2，更新1505ms；不使用含等待的53000ms |
| 是否使用统一epoch | 所有option只读取一次cost snapshot；旧snapshot不随后续更新变化；来源或预算变化改变plan SHA |
| 两种论文选择是否有区别 | 同fixture下handoff选高密度HOST，residency先选可容纳的GPU；同adapter无重复最终目标 |
| 是否靠缺测量兜底 | 正需求缺representation/layout类报错；重复观测、非法边界、冲突source和重复target拒绝 |
| 新副本是否继承测试结果 | new_replica恢复冻结初值，不继承在线更新 |
| 是否执行旧Full | 实际runner入口在启动后台任务之前拒绝；无文件重置或旧预加载副作用 |

定向152项通过；完整回归及资源收尾以EXECUTION_STATUS中的最终回执为准。
这些输入是CPU正确性fixture，没有建立新的7B/3B测量profile，没有GPU运行。

### 明确尚未完成，下一步不重复本轮检查

本轮接通**实际stack的规划入口**，但没有完成自动控制路径上的源/预算option
生产、实测profile加载与在线绑定，也没有接通统一pending movement queue。
因此Full执行目前被明确拦截；不能称已完成handoff/residency机制或论文实验。
接下来把代表性实测class、物理owner快照及统一迁移/替换执行接通，处理总HOST/
native tensor预算和共享/激活前压力。GPU原生LRU尚不是论文的loss-per-usable-
byte替换规则，不能忽略。完整验证回放须待上述合同闭合，不用另一项孤立微测
代替。baseline、M1/M2、消融和敏感性仍未开始。

## D36：实测准备成本初始化、实际请求更新与副本继承接通

### 问题与依据

D35已接上需求窗口与数学规划，但成本仍须由调用者提供；没有代表性实测
加载器、实际接纳时的class绑定和完成更新，仍不能作为在线系统。
本轮核对D34/D35、IEEE的d定义，以及
[vLLM0.30原始loader](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/worker_manager.py)
的原生CPU复用与文件加载边界；[dLoRA](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang)
仅作已有请求/adapter联合编排背景，不构成本轮性能证据。

可证伪问题：实际请求在接纳时固定的来源表示/尺寸类，是否能在加载完成后
收到正确d更新，并在下一次规划中使用，同时不把当前测试轮学习状态传给
新副本？本轮不修改九个公式，不以服务D补准备d，不启动另一模型短前缀。

### 已连接的实际路径

1. `confirmed_source_class`的来源描述保留真实`footprint_bytes`和
   `representation`：原生CPU/GPU使用原生占用，HOST/NVMe文件使用确认的
   allocated bytes，Remote使用冻结清单的payload bytes。不反推size bin
   中点，也不把压缩线上字节当作本地占用。
2. `FrozenPreparationProfiles.load`从原始source admission、native完成
   receipt和本请求remote receipt重新计算d，再形成class均值。文件SHA、
   模型配置、backend/resource/input context、固定生成和原生计时协议必须
   匹配。不同启动时钟的历史profile保留自己的时钟域，只在记录内部相减。
   重复request/attempt或同一native lease跨记录重复均拒绝。
3. 当前layout分区采用保守的`exact_content_v1`：只有完整已校验内容身份
   一致才允许共享layout类别，再区分来源表示和预先冻结的size bin。相同
   rank或字节数不自动代表相同layout。此分区比按tensor shape合并更细；
   不宣称不同权重内容的同形adapter已共享profile，也不改变500逻辑ID含义。
4. 实际runner支持`coordination.ieee_preparation_profile`，配置项与服务
   profile一致为path/SHA/context/ewma_beta，两种测量context必须一致。
   InstancePool给每个新runtime独立初始化cost model；backend配置不符则
   在warmup前拒绝并清理新engine。汇总保存准备profile身份。
5. 实际`_ieee_protect_selected_source`在任何source-hold/load RPC之前固定
   preparation class；缺测量或profile不符直接报错。来源接纳记录新增
   clock/class/profile身份。GPU命中和backbone不制造加载样本。
6. `_ieee_prepare_selected_adapter`实际取得可执行adapter后，将D34完整
   加载区间更新到同一slot、同一冻结class；共享/变化来源的不完整区间仍
   不更新。加载完成是更新事件，不等待生成结束才把服务D误写入d。

数据合同通过不等于原始测量正确性已独立合格；当前没有新的真实7B/3B
代表性profile文件，本轮测试文件均为临时fixture，不进入实验结果目录。

### 正确性状态表

| 问题 | 实际代码路径上的检查 |
|---|---|
| 初值是否由测量边界得出 | 历史时钟fixture的两个加载区间3000/1000ms形成2000ms均值，不使用含初始等待的D |
| 缺类能否借用相似类 | 内容、表示或size bin不匹配均拒绝，无rank-only、邻近类或固定延迟兜底 |
| 请求是否更新自己的来源类 | 实际runner＋native-cache替身＋临时文件/HTTP响应夹具覆盖native HOST、文件HOST/NVMe/Remote；仅本类更新，GPU命中不更新，非真实远端实验 |
| 缺类是否先加载再失败 | 真实请求入口在hold/load之前抛出缺类错误，未调用生成，未遗留native引用 |
| 更新能否影响下一规划 | 请求释放后，受控缓存替身退回HOST；重新读取来源，实际stack下一handoff计划使用更新后的d和序号 |
| 新副本是否继承测试学习 | 实际InstancePool第二副本恢复冻结均值；不同runtime配置/重复物理engine拒绝 |
| profile是否可追溯 | 实际runner加载SHA绑定文件并输出profile身份，服务/准备context不一致拒绝 |

新增9项检查。首次201项定向检查有两处测试假设错误：缺类实际向外抛异常，
而非返回失败请求；1024B预算按1MiB保守DP返回空集，不能假定强制greedy。
修正测试为期望异常，并用同一1024B物理fixture的handoff密度规则验证成本
传播，未改小预算DP或放宽生产语义。初次与最终760项功能回归均通过，安全
56项通过，无跳过。测试fixture时延不能作为任何模型性能或优势数值。

### 当前边界与主线下一步

本轮完成profile消费与实际请求更新链，不等于已经采集代表性profile，也
不等于自动规划/迁移执行已启用。仍须把真实owner的source/target footprint/
剩余预算组合成规划输入，将handoff与steady-state送入同一pending movement
queue，完成loss-per-usable-byte替换及总HOST/native预算、共享/激活前压力。
D35的Full启动拦截保留。之后才能做有意义的整体资格和完整回放；不得绕过
拦截、把fixture写进配置或重复孤立的cost/profile微测。正式矩阵仍未开始。

## D37：收益—损失替换接到原生 GPU 槽位所有者

### 瓶颈假设与依据

D29–D36的主动HOST→GPU路径仍按原生LRU选择victim。它不能验证IEEE正文
“相同h/d下按loss per usable byte替换”的机制，也会使后续CapacityOnly
消融缺少一致的替换基础。本轮不改公式，不改普通按需加载策略。

核对官方[vLLM0.30 model manager](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/lora/model_manager.py)
和[cache](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/utils/cache.py)：
原生active cache移除回调释放GPU槽位，CPU registered cache独立；普通
activate在满池时调用remove_oldest。采用已有GPU-only移除接口，不靠重排
LRU或临时给其他victim加假引用来强迫后端选中目标。
[ELORA](https://arxiv.org/abs/2505.03756)提供LoRA/KV联合管理的相关背景，
本轮替换目标以IEEE本文定义为准，不据此宣称新算法或性能优越性。
本地安装源码SHA：model_manager
`6695bb7d6373d8f29bb13d6ae31e2ef2a98f8c23a6f37f9147d67cafca5f3302`；cache
`397e993cebeeb44f37104e6f9a48ac7b4f3f878e0294fd8899e22f161de379c5`。

### 已接通的实际路径与范围

1. 实际ExperimentStack从HotnessTracker读取一次完整窗口，从每副本准备
   cost model读取一次h/d版本。原生source/footprint快照绑定owner、epoch、
   source path、内容类和slot bytes，生成可序列化objective及SHA。
   HOST fallback使用该adapter实际native CPU表示的准备成本，不用服务D。
   缺少正需求class、未知native身份或未确认GPU副本时拒绝；零需求不填假d。
2. objective通过既有engine/proxy/core preparation命令传给实际worker。
   worker提供当前unfinished/controller-pending需求的保护集合，并使用
   实际pool重新核验slot bytes。SHA是消息一致性，不是测量正确性认证。
3. native owner核对完整当前来源、epoch与GPU槽位。GPU/CPU引用、外部pin、
   活跃source transfer和当前pending demand排除victim。剩余victim按
   `h*d_HOST / usable_slot_bytes`排序，平局按adapter ID。
4. 当前限定为已资格的uniform dense GPU slots：一个完整槽位足以容纳
   incoming，满池最短覆盖前缀就是一个victim。文件大小或LoRA rank不充当
   GPU可复用字节。仅incoming收益严格大于loss才继续E(t)检查。
5. Full/CapacityOnly使用相同替换objective和物理保护。E(t)延后或净收益
   不正时，不驱逐、不touch LRU。接纳后，同一serialized owner在完成栅栏
   后移除指定GPU条目，再由既有pitched-copy/native setter填入释放槽位。
   不释放HOST fallback，完成后再核对fallback对象和所占slot。
6. ordinary demand继续使用原生LRU。没有objective的旧qualification调用
   显式标`native_lru_diagnostic`，不是缺profile时的IEEE兜底。Full旧入口
   仍被D35 guard拦截。失败/不确定copy使owner进入需恢复状态，不虚构回滚。

### 正确性状态表（构造输入，不是模型性能）

| 论文需要的性质 | 检查结果 |
|---|---|
| 与LRU是否真正不同 | 实际owner中LRU为2；同一h/d下选择loss更小的3，使用其slot1，保留其原CPU对象 |
| 净收益与严格边界 | fixture收益15ms、loss0.2ms可接纳；收益等于loss或为0时不进入资源准入 |
| 延后是否损害旧缓存 | E(t)拒绝前后source/epoch/slot/LRU完全不变 |
| 保护是否覆盖pending/transfer | 实际worker的controller-pending集合及CPU/GPU引用排除最优victim；全部受保护时不替换 |
| usable bytes是否真实 | worker实际slot101B与消息100B不一致时，在驱逐前拒绝（仅fixture字节） |
| 同分是否稳定 | 精确相同loss时按adapter ID，不追随native LRU顺序；后续普通需求仍按LRU |
| 重试与失败 | 相同lease/objective幂等；stale/hash错误拒绝；copy失败或fallback被丢弃不发布成功 |
| 成本是否同轮 | 实际stack只snapshot一次cost；正需求缺类拒绝，零需求缺类保留d=null |

新增13项检查。首轮定向114项、初次全量772项通过；随后把同分fixture改为
精确可表示的1.5并增加fallback丢失断言。已安装vLLM0.30环境运行上述13项
全部通过，CUDA未初始化。最终773项全量和56项安全检查全部通过，无跳过。
没有加载backbone、新生成权重/trace、采集真实profile或开展性能实验。

### 仍未闭合的主线

这是单次原生HOST→GPU物理事务的替换执行，不是全层级Full完成。统一
pending movement queue尚需负责候选密度顺序、pending target保护、跨执行
源变化和同规划epoch的剩余工作；HOST/NVMe可变大小多victim替换、总HOST/
native占用、共享/激活前压力和完整生命周期也仍需接通。代表性实测profile、
正确adapter数值资格、真实remote资格和完整回放尚未完成。不重复本轮
selector/cache fixture或旧短前缀微测；下一步进入统一迁移与资源所有者整合。

## D38：共享文件准备队列接入实际请求路径

### 问题、依据与实现边界

本轮沿D37主线整合执行，不修改九个公式。IEEE要求handoff与稳态规划共用
pending movement queue，复用已存在/正在准备的副本，并在依赖操作结束后
释放取消任务的资源。旧PreloadingManager的顺序执行、轮询和超时状态不能
提供这个所有权保证。因此在同一既有manager中增加owned queue，复用D32的
文件分配/发布和D33的真实加载压力入口，不另造下载器或实验框架。

实现依据：Python的[shield/cancellation规范](https://docs.python.org/3.12/library/asyncio-task.html#shielding-from-cancellation)
明确区分等待者取消与受保护任务；必须保留任务引用并等待其结束。
vLLM 0.30的[原生LoRA worker](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py)
仍由自己的执行路径管理缓存，本轮不将控制器Future当作GPU完成栅栏。
可证伪假设是：实际文件路径能共享一次准备，取消和重复订阅不导致重复
分配、提前回收或重复成本样本。这里不假定该改动已改善TTFT或GPU-s。

### 已接入的执行路径

1. 一个物理任务键绑定文件owner incarnation、目标tier、adapter和内容SHA。
   handoff/residency/demand保留各自plan、activation、target replica和订阅
   记录；相同任务复用原执行体，不启动第二个writer。
2. 队列先处理等待中的请求驱动任务，再按收益密度和稳定身份排序。
   不抢占已开始的物理操作。并发上限沿用既有显式配置；队列不假称自己
   预留了存储，实际容量仍由文件owner在写入前原子检查/预分配。
3. `ScenarioRunner._ensure_local_async`的真实HTTP miss现在进入此队列。
   同一入口接收显式handoff/residency文件准备，但自动planner触发尚未连接。
   Remote→NVMe和已确认HOST/NVMe互拷仍使用原有执行器与取消join。
4. 每次执行重新观察真实来源。终态Future不充当缓存：后续调用必须重新
   验证目标；驱逐后重新下载。互拷在source owner锁内核对冻结内容SHA，
   在不匹配时尚未分配目标，不依赖复制完成后才发现错误。
5. 一个订阅取消不会取消其他订阅的writer。最后一个取消必须等待真实
   reader/writer与压力finish结束；重复取消、取消开始后新订阅、协程尚未
   开始、取消与全局关闭并发都明确处理。失败任务保留，不自动盲重试。
6. deferred保持显式状态，仅接受对应物理owner的状态变化唤醒；其他owner
   不触发重试，执行期间发生的唤醒也不丢失。本轮真实文件复制不以此
   替代物理容量失败；原生GPU admission的自动唤醒连接仍待实现。
7. 仅实际操作创建者可将Remote span作为其准备成本样本，共享者不重复
   领取。输出保留一次真实传输、各订阅和attempt状态；summary可序列化，
   不携带活任务/engine。全局关机先join准备，再拆除对应推理实例。

### 正确性状态表（小型文件/HTTP响应fixture，不是远端性能实验）

| 要检查的问题 | 本轮观测 |
|---|---|
| handoff与请求同时准备同一NVMe目标 | 同一job、一次实际下载与一次文件发布；各自触发身份保留 |
| HOST并发复制 | 一次实际层间复制；目标文件内容一致；无残留materialization |
| 已完成Future与真实缓存是否混淆 | 后续命中重新核验；实际删除NVMe副本后再次下载 |
| 成本是否重复归属 | 创建者拥有Remote evidence，共享请求的成本样本字典保持空 |
| 源身份发生偏差 | owner锁内拒绝，HOST目标未创建，记录rejected |
| 取消一个共享等待者 | writer继续，其他订阅完成；正常完成不伪标withdrawn |
| 最后取消与关机并发 | 真实HTTP reader未退出时，文件预留和加载压力仍在，两个等待者均未提前返回 |
| 延后和排队顺序 | 只由同owner事件重试；请求优先、密度/身份稳定；无定时重试 |
| 关机先后关系 | queue任务终态先于实际runner移除实例 |

新增12项检查。相关166项、初次及最终785项全量、56项安全检查全部通过，
无失败/跳过。新增12项还在已安装vLLM0.30中通过，CUDA未初始化；其后
加强的并发close断言包含在最终785项中。这不是模型运行、真实174远端
下载或代表性profile，不提供新的性能数值。提交前147项历史保护清单与
计划SHA核验零变化；五个本轮资源域均为空、high/max/OOM为零并已关闭。
没有修改原图表、工件池或trace。

### 下一步与仍未通过的资格

Full guard保持：本轮不宣称自动handoff、原生GPU准备队列、pending-target
victim保护或全层级replacement完成。共享文件传输压力目前仍绑定发起
任务的已初始化target，不是所有副本/激活前的全局压力。单实例退役与跨
副本共享操作也需结合该压力所有权整合；只有全局关机的join顺序在本轮
接通。总HOST/native tensor预算、自动规划options、GPU待办执行/唤醒、
完整物理生命周期、实测初始化、数值及remote资格仍是主线未完成项。
下一步以这些集成为单位推进，不扩大相同文件/短前缀测试，不启动M1/M2。

## D39：原生GPU准备接入共同队列与pending-target保护

### 证据驱动的问题

D38已有真实文件队列，D37已有单次HOST→GPU准入/替换，但两者尚未连接。
此前worker只排除native需求、引用和传输pin；已规划未执行的目标不在
victim保护集合内。另外，把每个候选都要求为原始slot epoch，会让一轮
计划自己的第一项复制使后续项立即过期。这不是需要新评分公式的问题，
而是冻结优化目标与动态执行状态必须分离。

依据仍是IEEE的“同轮固定h/d、执行重查、pending target不可主动驱逐”。
本轮核查[vLLM0.30 LoRA manager](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/model_manager.py)
中CPU注册与GPU active cache/slot的联动，以及[EngineCore](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
的utility执行位置。登记与替换均落在实际native owner，不以控制器名单
冒充原子资源状态，也不改原生按需加载LRU。假设是完整选集先登记后执行，
可以在不改h/d及E(t)的前提下，使多个准备任务正确共享、延后和取消。

### 已完成连接（不是自动Full启动）

- 实际runner增加显式GPU计划执行入口，复用D38 queue。执行前一次登记
  整个selected set；内容身份与冻结工件索引核对，GPU任务按相同h*d/slot
  bytes排序。该入口接收已选集合，不自行发明planner目标。
- native owner登记plan incarnation、原始objective SHA与pending targets。
  首次登记要求完整当前source epoch，重复必须同身份；关闭留下tombstone。
  登记不占用GPU/CPU物理字节、不pin所有目标，也不伪称reservation。
- 每个执行attempt重新观察当前native来源，进入已有core/worker事务。
  已登记plan保留原始h/d，但允许本轮自身的slot/引用变化；仍检查完整
  当前source/fallback、live引用与E(t)。未登记的旧诊断路径仍执行原来的
  严格epoch检查，不能用新路径任意接受陈旧objective。
- pending targets加入主动replacement的排除集。目标已在GPU时，通过
  native acquire/fence确认复用，不虚构一次加载；其reference也关联plan。
  原生按需加载仍具有原策略和优先权，本轮不是修改整个LRU语义。
- successful准备释放临时GPU引用后才结束目标。延后任务保留，实际GPU/
  HOST引用释放、同engine文件压力finish和目标完成事件唤醒重查，无轮询。
  原生iteration变化尚未单独推送给此队列；当前由上述事件推进，不能
  将其描述为每个iteration都立即重试的完整调度实现。
- 一个任务供两个plan复用时，真实加载只做一次。若creator取消，其
  native冻结计划必须留到共享操作终态；不会撤掉其他订阅仍依赖的保护。
  取消后的等待是资源所有权join，不是成功请求，也不重置任何到达时刻。
- 丢失提交回复时，已有GPU lease和目标保护保持不确定状态；close拒绝
  带未释放操作的plan。代理仅凭同owner/plan的关闭证明清理对应RPC记录。
  不清除其他plan、旧owner或ordinary demand的不确定性。
- 实际全局shutdown先取消并join GPU plan，再关闭file queue及runtime。
  summary保存计划、attempt、真实准入/释放和关闭结果；不把重复订阅的
  返回值分别累加为两次物理加载。

### 正确性状态表（CPU原生cache fixture，不是模型性能）

| 论文性质 | 本轮观测 |
|---|---|
| 整个选集先于执行 | 错误目标集合原子拒绝；有效集合进入同一native owner |
| pending target是否可被主动替换 | 原最低loss目标受保护后不再被选；finish后才恢复资格 |
| 同轮执行是否改了目标 | GPU命中acquire/release改变实时epoch后，后续HOST→GPU仍用同一objective SHA/h/d |
| HOST→GPU和GPU复用 | 实际runner/common queue执行两目标，复用一项、加载一项；引用全部归还 |
| admission延后 | 旧GPU内容保留；实际引用释放后第二attempt重查并完成 |
| 共享与creator取消 | 两plan同一物理GPU任务、一次加载；creator取消不撤掉另一订阅保护 |
| 不确定回复 | 注入提交后丢回复：lease与pending target仍在，plan标closure_unresolved |
| proxy关闭范围 | 只清理同owner/plan准备RPC；其他身份和需求加载不被清除 |
| engine/core/worker链路 | 新plan命令、core字段及native owner回复贯通；错误字段拒绝 |
| 全局shutdown | GPU准备与临时引用/目标关闭先于runtime移除 |

新增11项检查。首轮定向130项、初次全量794项、最终796项以及56项安全
检查通过。最终在实际安装vLLM0.30中运行11项新增检查与11项core hook
检查，共22项通过，CUDA未初始化。没有新GPU/model运行、真实profile或
远端性能测量。构造输入不能证明实际数值adapter资格或G1/G2领先。

### 返回主线

Full guard不解除。自动planner/handoff触发及options生产、文件→native
HOST装载、总HOST预算、共享/激活前压力、文件pending target/可变大小
replacement、逐replica退役与完整生命周期仍需集成。主动加载完成如何
统一反馈d也需随自动规划路径接入；本轮只保留原始加载完成证据，没有
伪造profile。下一步推进这些完整资源/控制路径，不再重复本轮cache/queue
或旧模型短前缀检查。正式比较、消融、敏感性均未启动。

## D40：共享文件域与激活前传输压力

### 问题、依据与边界

D33只把一次文件准备通知发起它的已初始化engine。D38共享文件队列和
D39 GPU计划使用同一个HOST/NVMe物理所有者，但其他副本可能看不见该活动；
激活期间还不存在目标engine时，原入口直接拒绝。因此仅把目标engine的
活动数称为共享资源压力是不成立的。本轮假设：以实际文件所有者保存
唯一传输区间，在副本加入时同步未完成区间，可以消除这类状态遗漏，
无需改变IEEE的p_load、E(t)、h/d或准备选择公式。

核对IEEE handoff/residency的共享复制、执行重查和取消语义，以及D32–D39
历史实现；再次查阅[vLLM0.30 EngineCore源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/core.py)
的utility执行位置和[Python取消屏蔽语义](https://docs.python.org/3.12/library/asyncio-task.html#shielding-from-cancellation)。
采用实际native owner确认而非把控制端cached counter当作原子GPU状态。
这里的域只包含本runner共享同一文件所有者的副本，不把其他节点或无关
服务的传输强加给每个副本，也不声称实现全局带宽限速。

### 接入及正确性状态表

| 论文性质 | 本轮实现及验证范围 |
|---|---|
| 共享传输只计一次 | 真实文件入口建立一个transfer ID；两个native owner各观察同一活动，重复逻辑slot不增加活动数 |
| engine尚未初始化 | 文件域先持有区间；新engine在warmup/池发布/主动GPU计划前登记并重放全部未完成区间 |
| 加入与结束并发 | 同一控制锁序列化开始、加入、结束通知，不串行化实际文件复制；加入确认前不能参与主动准备 |
| 原生准入 | native core绑定不可变file-domain身份；worker接受该共享域快照，使用原有p_load/E(t)公式 |
| 错误与应答丢失 | 一副本finish失败保留不确定压力，但其他副本仍完成自己的finish；lost attach不允许IO或伪造空状态 |
| 取消 | 取消加入者不取消另一个副本的文件操作；IO/清理join后才结束；重复取消retirement仍等待实际压力终态 |
| 缩容清理 | 实际slot cleanup先join该engine准备计划，再退出共享压力域，最后调用engine shutdown；这不代替物理GPU释放证明 |
| 可追溯 | summary保留共享域成员及不确定状态；transfer保留每个native owner的start/finish回执、时钟和唯一IO结果 |

初轮定向144项通过。初次全量807项出现两个历史legacy测试fixture缺少
真实runner必有的model_cfg字段；补充空配置明确legacy身份，不在生产路径
加入缺字段静默回退。最终808项功能回归通过；新增12项及11项core hook
在已安装vLLM0.30下通过。另一个worker测试最初选择器类名写错，改用实际
类名后独立通过，共24项有效原生环境检查，CUDA未初始化。完整回执及
安全检查记录见EXECUTION_STATUS。
本轮使用实际runner/native接口与微型文件fixture，不是新GPU模型实验、
174真实服务实验、准备profile、性能对比或Full资格；按计划11.2交付状态表。

### 返回主线

该共享压力路径不等于自动planner/handoff已闭合。总HOST文件/native tensor
预算、文件到native HOST加载、选项生成及主动d反馈、文件pending-target/
可变大小替换、完整Full物理生命周期仍待接入。保留IEEE Full启动拒绝旧
warmup的限制。下轮沿总HOST/native所有权和自动规划主线继续，不再重复
本轮共享计数检查或旧模型短前缀。基线、M1/M2、消融和敏感性尚未启动。
