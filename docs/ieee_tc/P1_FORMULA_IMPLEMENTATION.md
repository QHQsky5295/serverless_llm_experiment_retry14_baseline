# P1 — IEEE 公式与实现合同（执行中）

本表不是性能结果，也不表示 Full 已完成 IEEE 对齐。主比较必须等所有关键
合同关闭；不得把旧代码的指标移植到新设计上。论文源文件未改动。

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
