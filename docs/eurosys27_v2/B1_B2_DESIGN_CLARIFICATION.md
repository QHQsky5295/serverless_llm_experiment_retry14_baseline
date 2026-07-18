# B1/B2：placement、scale-out、residency 与 budget 的设计澄清

本文回答审稿人 B 的两组问题：Section 4.1 是否覆盖了完整服务场景；Section 4.2 中谁设置 budget、系统如何知道 adapter 在哪里、plan 是否最优，以及动态 input/output length、RPS、KV cache 和 model size 如何处理。

## 1. 论文规范语义

### 1.1 控制面的职责边界

PrimeLoRA 将一次请求的决策拆成四层，而不是求解一个覆盖未来全部请求的离线最优化问题：

1. **autoscaling** 决定是否增加/减少 runtime replica；
2. **placement** 在当前 active replicas 中选择一个能够接受请求且预计最快进入执行的 replica；
3. **residency/handoff planning** 在较慢时间尺度选择值得向更近 tier 移动的 adapter；
4. **GPU admission** 在执行 promotion 的当下重新检查瞬时 GPU feasibility。

因此，PrimeLoRA 的保证是：每个已提交 tier plan 和 GPU promotion 均满足当时的容量约束；系统不宣称在未知未来请求和运行时状态下得到全局最优解。

### 1.2 谁设置 budget

budget 分为 operator-provided physical envelope 与 runtime-derived residual budget：

- operator 配置每个 runtime 的 GPU memory envelope、HOST/NVMe cache capacity、minimum/maximum replicas、LoRA execution limit、model/runtime reserve 与 safety margin；
- backend/runtime 暴露 model weights、当前 KV/adapter memory、active batches 和可见 GPU memory；
- PrimeLoRA 在时刻 `t` 计算可用于 tier `l` 的残余 budget：

```text
B_i,l(t) = C_i,l - U_i,l(t) - R_i,l(t) - S_i,l
```

其中 `C` 是配置容量，`U` 是已使用容量，`R` 是为 model/KV/loading 等保留的动态 headroom，`S` 是静态 safety margin。对 GPU，最终 promotion 还要通过 [C3 的 momentary admission](C3_PLAN_VS_ADMISSION.md)。operator 不逐请求指定 adapter budget；runtime 也不会自行改变物理容量。

### 1.3 系统如何知道 adapter 在哪里

论文规范状态是 per-replica observable tier hint：

```text
T_i[a] = (tier, state, version, updated_at)
tier  in {GPU, HOST, NVMe, REMOTE}
state in {AVAILABLE, LOADING, EVICTING, FAILED, UNKNOWN}
```

- `REMOTE` 是 backing store/source-of-truth 或本地未命中。
- 只有成功 materialize/load 的完成事件才能把 hint 提升为 `AVAILABLE`。
- load/evict 开始先写 transition state，完成时原子发布新 tier/version；失败写 `FAILED` 并保留最后一个已验证的可服务 lower tier。
- routing 读取一个 versioned snapshot；缺失、过期、失败或冲突状态一律保守映射为较远 tier，不把未经验证的 hint 当作 GPU-ready。
- 单节点部署中 HOST/NVMe 可以是 node-local shared state；GPU executability 必须按 replica/runtime 记录。

该 registry 是 routing hint，不是分布式事务数据库。请求真正开始执行前，backend path resolution 仍负责验证 adapter 可由所选 runtime 地址化。

### 1.4 Placement 算法

令请求 `r` 的 adapter 为 `a(r)`。对每个 active replica `i`，控制面构造：

- `runtime_feasible_i`：runtime 有空 execution lane，且不处于 forwarding/retiring 状态；
- `active_lora_feasible_i`：满足 [C2 的 distinct active-LoRA 限制](C2_ACTIVE_LORA_FEASIBILITY.md)；
- `handoff_i`：adapter 是否属于该新 replica 的有界 first-service handoff set；
- `service_i`：根据 tier、已观测 I/O、runtime first-token 和 post-TTFT occupancy 估计的服务代价；
- `queue_i`：active requests、预计 lane release、adapter-load pressure、GPU utilization 和最近选择时间等轻量状态。

论文中应给出如下伪代码，而不是暗示对所有未来请求联合求最优：

```text
PLACE(r, t):
    repeat:
        candidates = snapshot(active replicas)
        for i in candidates:
            feasible[i] = runtime_feasible(i) AND active_lora_feasible(i, a(r))
            handoff[i]  = bounded_handoff_priority(i, a(r))
            service[i]  = readiness_cost(T_i[a(r)])
                          + observed_runtime_cost(i, a(r))
                          + predicted_occupancy(i, a(r))
            load[i]     = lightweight_runtime_hints(i)
        i* = lexicographic_min(
                 not feasible[i], handoff[i], service[i], load[i], stable_tiebreak(i))
        if reserve_execution_lane(i*, a(r)) succeeds:
            return i*
        wait until capacity/state changes
```

`reserve_execution_lane` 必须和 active-request/active-adapter shadow state 的增加发生在同一控制面临界区；否则并发请求可能都基于同一过期视图选中最后一个空 lane。

### 1.5 Scale-out handoff 算法

当 autoscaler 触发 scale-out 时，控制面只使用已经到达的 waiting queue 加上“预计在 runtime ready 前到达”的请求，不读取 held-out future trace。步骤为：

```text
BUILD_HANDOFF(new replica i, t):
    ready_at = t + predicted_runtime_startup + predicted_plan_load
    Q_ready  = currently_waiting_requests
               + arrivals_predicted_before(ready_at)
    Q_left   = subtract_requests_expected_to_start_on_incumbents(Q_ready)
    A_order  = adapters in first-service order of Q_left
    A_i      = longest prefix of A_order fitting B_i,handoff(t)
    publish (A_i, first_service_request_budget, expiry/version)
```

handoff priority 是有界的：只保护新 replica 的 first-service request budget，并为非 handoff 请求保留至少一个可用 lane。budget 消耗后，新 replica 回到普通 readiness/load routing，不形成永久 affinity。

### 1.6 Residency planning 与最优性边界

对 tier `l`，planner 先过滤当前 lower-tier candidates，再最大化 snapshot 上的总 readiness value：

```text
maximize   sum_a value_i,l(a,t) * x_a
subject to sum_a size(a) * x_a <= B_i,l(t)
           x_a in {0,1}
```

精确描述必须包含以下限定：

- 若 candidate count 与 budget state-space 在设定上限内，使用 MiB 粒度 0--1 knapsack；“最优”只针对这个**已过滤、离散化、单次 snapshot**。
- 若问题超过 planner 的时间/空间上限，使用 value-density/priority greedy；该路径不提供组合最优保证。
- 即便 DP 路径返回 snapshot optimum，arrival、KV、tier state 在 plan 生成后仍会变化，因此不构成 online/global optimum。
- 两条路径都必须满足 `sum size <= residual budget`；这才是系统需要声明的共同保证。

### 1.7 如何处理动态性

动态信号按变化速度分层处理：

| 信号 | 更新方式 | 使用位置 |
|---|---|---|
| request arrival、waiting queue、active request/adapter | 事件驱动 | placement、scale-out evidence、handoff frontier |
| adapter access/hotness | 每次访问更新，滑动窗口/EWMA | handoff/residency value |
| measured I/O、TTFT、tail occupancy | 请求完成时更新 | readiness-aware service estimate |
| batch input/output budget、active tokens | batch start/end | KV state与near-term KV growth |
| GPU memory/utilization、load pressure | 周期采样并在 admission 时刷新 | GPU effective capacity |
| tier movement | load/evict start、success、failure 事件 | location registry与routing hint |

planner 不需要准确预测最终输出长度；它使用 request metadata 与在线 observed distributions 构造 near-term estimate。任何预测误差都由执行时 admission、backend path validation 和下次 event/sample 更新吸收。模型大小在 runtime 启动时进入固定 model-weight reserve，不随请求改变。

### 1.8 场景覆盖表

| 场景 | Placement/scale-out 行为 | Adapter path 行为 | 必须记录的证据 |
|---|---|---|---|
| 有空 replica 且 GPU hit | 优先可行、低 service cost replica | 无额外 tier movement | pre-dispatch GPU tier、routing decision |
| GPU-cold，但 HOST/NVMe local | 与 queue/occupancy 联合比较，不只看 least-loaded | 从最近 verified local tier 地址化；promotion 另行 admission | pre-dispatch HOST/NVMe、I/O 与 admission decision |
| local miss/unknown/stale | 保守按 REMOTE 成本排序 | fetch 到 local tier，成功后发布新 hint | remote-cold、fetch success/failure、tier version |
| runtime lane 已满 | 标为 runtime-infeasible | 不提前进入 backend hidden queue | reserve retry/wait 时间 |
| distinct active LoRAs 达到上限，目标未 active | 标为 active-LoRA-infeasible | 等待某个 adapter reference 释放或选另一 replica | active-set size、cap、reserve retry |
| distinct cap 已满但目标 adapter 已 active | 仍可行；不会新增 distinct active slot | 复用当前 active adapter | active adapter refcount |
| 新 replica + planned adapter | 在 first-service budget 内给有界 handoff priority | ready 前 warmup，成功项才发布到 plan | plan/actual warmup/match/served |
| 新 replica + 非 planned adapter | 保留非 handoff lane；不允许永久阻塞 | 正常 tier resolve | handoff penalty与budget剩余 |
| GPU promotion momentarily rejected | 请求仍可被 placement | 使用 lower local path，不拒绝请求 | reject原因、source tier、请求成功 |
| scale-in | 先停止新 placement，等待/迁移 active work，再释放 runtime | 保留/降级可用 lower-tier artifact | draining state、lifecycle timestamps |
| load/replica failure | 删除或降级 hint，唤醒等待请求重新选 | fallback 至 verified lower tier；无路径则请求失败并留证 | failure、fallback、complete=false |

V2 的 A2/A3 累积消融和 readiness diagnostic 负责把这些机制映射到实际触发计数；场景未触发时，不得声称其被完整评估。

## 2. 当前实现证据

### 2.1 已实现并可定位的部分

| 论文概念 | 当前代码证据 | 说明 |
|---|---|---|
| per-slot runtime/tier hints | [`InstanceSlot`](../../faaslora/experiment/instance_pool.py) 的 `active_requests`、`active_adapter_counts`、三个 tier sets、`load_queue_depth`、`gpu_utilization_pct` | GPU/HOST/NVMe hint 在 controller process 内存中 |
| active-LoRA feasibility | `active_adapter_count()`、`can_accept_active_adapter()`、`begin_active_adapter()`、`end_active_adapter()` | cap 来自 `model_cfg.max_loras` |
| lexicographic placement | `Router._routing_key()` 与 `select_instance()` | 实际 key 顺序比论文 Eq. 4 更细；先 runtime capacity 与 active-LoRA，再 handoff、service/occupancy 和 load hints |
| atomic lane reserve | `ScenarioRunner._try_reserve_runtime_request_slot()` | 同一 asyncio loop 中、第一次 `await` 前增加 active state；失败后重试 |
| service observations | `ObservedRequestCost` 与 `InstanceSlot.record_request_cost()` | completion event 更新 arithmetic mean；不是 EWMA |
| pre-dispatch tier | `ScenarioRunner._exec_request()` 中 selection/reserve 后冻结 `readiness_tier_before_dispatch` | 发生在 `_resolve_lora()` 前，可用于 dispatch-time audit |
| scale-out projection | `_scale_up_ready_candidate_queue()`、`_scale_up_incumbent_started_request_count()`、`_build_scale_up_runtime_handoff_plans()` | 使用 live waiting queue、ready-delay projection、first-service prefix |
| dynamic handoff budget | `_scale_up_preload_budget_snapshot()` | 从 target runtime headroom 与 exact handoff prefix 推导；另有显式 legacy fallback |
| tier capacity | [`TierCapacity`](../../faaslora/memory/residency_manager.py) 与 `ResidencyManager._can_admit_artifact()` | GPU/HOST/NVMe 由配置容量和 safety margin约束 |
| bounded planner | [`PreloadingPlanner`](../../faaslora/preloading/preloading_planner.py) | DP 使用 MiB units；capacity >16 GiB 或 candidates >1000 回退 greedy；hybrid 在更小门槛内选 DP |
| dynamic hotness | [`HotnessTracker`](../../faaslora/experiment/hotness_tracker.py) 与 `ExperimentStack.record_access()` | 每次 resolved request access 更新 window/registry |
| GPU effective capacity | [`ResourceCoordinator.evaluate_gpu_admission()`](../../faaslora/scheduling/resource_coordinator.py) | admission 时考虑 free memory、current/predicted KV、load、working-set reserve和实际显存压力 |

### 2.2 不能直接写成已实现事实的部分

1. **不是强一致、多节点 per-replica registry。** 当前 formal testbed 在一个 inference node 上共享 `ExperimentStack` 的 HOST/NVMe maps 和一个 registry；dedicated slot 保持自己的 GPU hints。论文应描述“observable per-replica/runtime hints”，不要声称实现了分布式共识目录。
2. **metadata 只有一个主 tier。** `ArtifactMetadata.storage_tier` 是单值；`tier_artifacts` 与 runner 的 local path sets补充可访问路径。它不能独立表达同一 adapter 在多个 replica/tier 的所有副本。
3. **service estimate 并非 EWMA。** `ObservedRequestCost.record()` 保存累计算术平均。若不改代码，论文可写“online observed estimates”，不能把每一项都称为 EWMA-updated。
4. **`queued_loads` 不是当前 queue depth。** `CoordinationMetrics.queued_loads` 只增不减。它可以作为累计 load-pressure evidence，但不能支撑“瞬时 loading-queue depth”这一具体实现陈述。
5. **DP 最优性范围有限。** 只有 `_knapsack_dp_selection()` 的离散 snapshot 有局部组合最优；hybrid/large-case 回退 greedy。任何“the plan is optimal”都必须删除或加上述限定。
6. **GPU tier 是控制面状态。** `ResidencyManager._perform_load()` 明确指出 GPU target 只确保 local backing path，实际 adapter load 由 vLLM 首次 `LoRARequest` 完成。因此 paper 中的 “GPU-ready” 证据不能只依据 registry update。

### 2.3 当前实现与论文规范的处理原则

本轮按用户决定不修改 B1/B2 对应核心代码。论文修改应采用上面的规范语义，同时在 artifact/evaluation text 中限定当前单节点实现：HOST/NVMe 是 node-local shared cache、GPU state 按 runtime 跟踪、observed service cost 使用 online averages。若某一规范语义没有对应 instrumentation，则必须标为设计语义或 future implementation requirement，不能写成已经通过实验验证。

## 3. 可直接用于英文论文的文字

### 3.1 Placement and scenario coverage

> **Placement semantics.** For each request, PrimeLoRA takes a snapshot of the active replicas and constructs a lexicographic routing key. The first component checks runtime-lane and active-LoRA feasibility; the second assigns bounded priority to adapters in a fresh replica's scale-out handoff set; the remaining components estimate adapter-path and runtime service cost and use lightweight occupancy signals as tie breakers. The controller reserves the selected runtime lane and the adapter's active-set reference atomically before yielding the dispatch loop. If the reservation fails because the snapshot became stale, the request waits for a capacity notification and repeats placement. Missing or failed locality hints are treated conservatively as remote rather than as cache hits.

### 3.2 Budget ownership and location state

> **Budget and location state.** The operator configures physical tier capacities, model/runtime reserves, safety margins, and replica limits; PrimeLoRA does not require an operator to assign a per-request adapter budget. At runtime, the controller subtracts current occupancy and reserved headroom from each configured capacity to obtain a residual planning budget. The single-node prototype maintains node-local HOST/NVMe path maps and per-runtime GPU locality hints, and publishes a closer tier only after the corresponding movement succeeds. These hints guide routing but do not replace backend path validation; stale, missing, or failed state falls back to the nearest previously verified lower tier.

### 3.3 Optimality and dynamics

> **Optimization scope.** Residency planning is a bounded online heuristic, not a claim of global optimality under future arrivals. For a filtered candidate snapshot, PrimeLoRA uses a MiB-granularity 0--1 knapsack when the state space fits the planning bound and otherwise falls back to priority scanning. The former is optimal only for that discretized snapshot, while the latter has no combinatorial optimality guarantee; both preserve the residual-capacity constraint. Request arrivals and adapter accesses update demand state, request completions update observed service costs, batch events update KV estimates, and tier transitions update locality. A separate admission check revalidates momentary GPU feasibility immediately before promotion, absorbing prediction error and state changes after a residency plan was formed.

### 3.4 Compact algorithm text

> **Algorithm overview.** On a request arrival, the controller first filters or penalizes replicas that lack a runtime lane or cannot accept the target adapter without exceeding the backend's distinct active-LoRA limit. It then gives temporary priority to a matching scale-out handoff, estimates the cost of reaching execution from the replica's observed adapter tier, and accounts for current occupancy before reserving the best replica. On scale-out, the controller projects which already-arrived and near-term requests will remain after incumbent service, extracts their adapter order, and warms the longest prefix that fits the new runtime's headroom. The handoff reservation expires after a bounded first-service request budget, after which the replica participates in ordinary readiness-aware routing.
