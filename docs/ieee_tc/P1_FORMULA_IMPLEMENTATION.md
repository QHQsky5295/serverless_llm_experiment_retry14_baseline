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
