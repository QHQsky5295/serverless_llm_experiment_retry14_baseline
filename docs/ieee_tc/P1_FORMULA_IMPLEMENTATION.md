# P1 — IEEE 公式与实现合同（执行中）

本表不是性能结果，也不表示 Full 已完成 IEEE 对齐。主比较必须等所有关键
合同关闭；不得把旧代码的指标移植到新设计上。论文源文件未改动。

本文件前面的首次审计表保留历史发现；逐项最新进展见 D1–D6。测试通过不等于
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
