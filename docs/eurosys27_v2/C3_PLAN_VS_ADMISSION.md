# C3：为何在 admission 时重新检查 budget

审稿人 C3 的核心问题是：如果 residency model 已经考虑 budget，为什么 PrimeLoRA 还要在 proactive GPU promotion 前检查一次；能否把它全部放进 model。这里的两次检查不是重复计算，而是不同时间尺度、不同问题。

## 1. 论文规范语义

### 1.1 Plan 决定“值得做什么”，admission 决定“现在能不能做”

Residency planner 使用较慢变化的 demand/locality state，回答：在一个 tier budget snapshot 下，哪些 adapter 的预期 readiness benefit 最大。它输出的是 candidate intent：

```text
plan entry = (adapter, source tier, target tier, expected value, size, plan version)
```

GPU admission 在 plan 即将执行或 request-path promotion 触发时，使用瞬时 inference state，回答：此时执行这次 movement 是否会和 KV growth、active batching 或正在进行的 adapter loads冲突。

```text
commit decision = ADMIT | DEFER/REJECT
```

一个 adapter 可同时满足“长期上值得留在 GPU”但不满足“此刻安全地搬入 GPU”。将两者分离，允许系统延迟 residency optimization而不拒绝用户请求。

### 1.2 为什么不能只在 planner 中检查一次

plan 生成与实际 promotion 之间，下列状态可能已经改变：

- 新请求进入并增长 active KV cache；
- 实际输出比预计更长；
- batch composition 与 concurrent sequence 数变化；
- 其他 adapter load 已开始或完成；
- backend/NVML 报告的真实 GPU free memory变化；
- candidate 已被另一个 event promote/evict；
- scale-out runtime 的实际 ready time 和 handoff working set偏离预测。

若只依赖 plan-time snapshot，系统要么在 stale state 下过度分配 GPU，要么必须每个 request/batch event 都完整重跑 knapsack。前者破坏安全性，后者把昂贵 planning 放进 inference critical path。PrimeLoRA 因而采用 plan/commit 分层：planner 可以过期，admission invariant 不能过期。

### 1.3 两层 budget 的关系

Planner 的 tier feasibility：

```text
sum(size(a) for a in selected candidates) <= residual_tier_budget_at_plan_time
```

Admission 的 momentary GPU feasibility：

```text
available(t)       = configured GPU envelope
                     - model/runtime allocation
                     - current KV and resident adapters
                     - loading reserve

future_reserve(t)  = max(predicted near-term KV growth,
                         uncovered recent adapter working set)

pressure(t)        = max(memory pressure,
                         current KV pressure,
                         predicted KV pressure,
                         in-flight load pressure,
                         measured physical GPU pressure)

effective(t)       = max(0, available(t) - future_reserve(t))
                     * (1 - pressure(t))

admit(a,t)         = measured_pressure_below_cutoff
                     AND value(a,t) > pressure(t)
                     AND size(a) <= effective(t)
```

planner 约束的是一个 candidate set 的总 budget；admission 检查单个操作的当前 residual capacity 与 contention。两者都通过后才提交 GPU residency。

### 1.4 请求不因 proactive promotion 被拒绝

GPU promotion 是优化，不是请求正确性的必要条件：

```text
PROMOTE_OR_SERVE(a):
    local_path = resolve_nearest_verified_local_tier(a)
    if local_path is absent:
        local_path = fetch_remote_to_local(a)

    if planner_or_online_policy nominates a for GPU:
        state = snapshot_current_gpu_state()
        if admission_allows(a, state):
            submit_backend_load(a, local_path)
            publish GPU-ready only after backend success
        else:
            keep candidate pending or discard stale plan entry

    serve request using the backend-addressable local_path
```

admission 拒绝意味着“此刻不主动固定 GPU residency”，而不是拒绝 inference。adapter 仍从 HOST/NVMe 的 backend-addressable path 按需服务；若本地不存在，先从 REMOTE materialize 到 local tier。

### 1.5 Failure 与重新规划

- `DEFER`：保留 plan intent，但下一次 admission 使用全新 snapshot；不得复用旧 effective capacity。
- `REJECT_STALE`：candidate value、tier 或 plan version 已失效，删除该 entry，等待下一轮 planner。
- backend load failure：不发布 GPU-ready，保留最后 verified lower tier并记录 failure。
- capacity race：residency manager 的最终 capacity check失败时视作 admission failure，不强制 `force=True` 绕过预算。
- user request path：若 lower local path有效则继续；只有 artifact 无法 materialize 或 backend invocation失败时，请求才失败。

## 2. 当前实现证据

### 2.1 Planner 路径

[`PreloadingPlanner`](../../faaslora/preloading/preloading_planner.py) 当前实现：

- 从 lower tiers 过滤 hotness、value、size 与 recent-access candidates；
- 可选择 greedy value、knapsack DP、hotness 或 hybrid strategy；
- DP 将 bytes 离散为 1 MiB units；capacity 超过 16 GiB 或 candidates 超过 1,000 时回退 greedy；
- hybrid 对较小 problem 使用 DP，对较大 problem 使用 priority greedy；
- `PreloadingPlanResult` 保存 selected artifacts、total size、capacity utilization、generation time 和 strategy。

[`ScenarioRunner`](../../scripts/run_all_experiments.py) 的 scale-out handoff 另有 instance-scoped fast path：

- `_scale_up_target_runtime_headroom_mb()` 从配置 GPU budget、model weights、reserve ratio 与 predicted KV growth推导 headroom；
- `_scale_up_exact_prefix_under_headroom()` 选择 first-service adapter order 中可放入 headroom 的 prefix；
- `_scale_up_preload_budget_snapshot()` 保存 exact prefix、live hotset、target headroom 与 plan metadata。

这两个 planning path 都使用 snapshot，不替代执行时 GPU admission。

### 2.2 Admission 路径

[`ResourceCoordinator.evaluate_gpu_admission()`](../../faaslora/scheduling/resource_coordinator.py) 当前返回：

- `pressure`：logical memory、current/predicted KV、in-flight load 和 actual GPU pressure 的最大值；
- `actual_gpu_pressure`：ResidencyManager capacity utilization，或 NVML/`nvidia-smi` 的 device-global probe；
- `utility`：recent hotness x source-tier locality，或 caller override；
- `effective_capacity_mb`；
- `predicted_kv_growth_mb`、`working_set_pressure`、`future_reserve_mb`；
- `should_attempt` 与 `admit`。

`request_lora_load()` 在 HOST/NVMe request path 调用该 decision：

- 已 GPU-resident则直接返回；
- 可 admission 时通过 loading semaphore 提交 residency mark；
- pressure 下可以短时等待并重算；
- `should_attempt=false` 或最终不能 admission 时不标记 GPU residency，request 仍持有 local path。

`ExperimentStack.warmup_gpu()` 也在 engine warmup 前调用 admission；`ScenarioRunner` 在 inference 后尝试把 HOST/NVMe served adapter 标为 GPU tier时再次调用 admission。

### 2.3 Final capacity 与 backend 边界

[`ResidencyManager.admit_artifact()`](../../faaslora/memory/residency_manager.py) 对 target tier 还有 final capacity/safety-margin check，并在必要时触发 eviction。它是最后的 control-plane capacity guard。

但当前实现有一个必须写清的边界：`ResidencyManager._perform_load()` 对 GPU target 只确保 local backing file存在；注释明确说明 actual GPU loading 由 vLLM 在第一条 `LoRARequest` 中完成。因此：

- `evaluate_gpu_admission()`/`admit_artifact()` 可证明 control-plane policy没有在已知 budget 下主动提交不合适的 residency intent；
- 不能仅凭 registry `GPU` 标签证明 backend 已完成物理 load；
- V2 的 `GPU-ready` evidence 必须同时检查实际 warmup/invocation success 和 pre-dispatch runtime hint，而不是只查 registry。

### 2.4 与论文规范的差异/限制

1. current coordinator 使用模型化 KV (`active_tokens * kv_per_1k_tokens_mb`) 和近期 batch-token EWMA，不是从 vLLM block allocator读取精确每请求 KV blocks。
2. actual GPU pressure probe有 cache interval；若 probe失败，代码可能回退为 0。正式实验需要记录 probe source/completeness，不能把 probe failure解释为“无压力”。
3. `request_lora_load()` 的 coordinated wait 最长通过固定轮询次数实现；它是实现策略，不应被论文描述为理论上的最优等待。
4. 存在某些 `force=True` 的 bookkeeping路径（例如 inference 后 tier mark）。这些路径不能用来证明严格 admission invariant；V2 instrumentation 应区分 policy-admitted、backend-on-demand load 与 bookkeeping reconciliation。
5. 本轮按用户决定不修改 C3 对应核心算法；论文应把保证限定为“budget-feasible admission under observed state”，而不是物理 allocator层的绝对无超配证明。

## 3. 回答 reviewer 的一句话版本

> Budget appears in both places because the planner solves a value-selection problem over a slower snapshot, whereas admission enforces a safety invariant over the GPU state at the instant a movement would be committed. Folding the second check into the model would either use stale KV/load state or require rerunning the planner on every inference event. A rejected promotion does not reject the request; the adapter remains serviceable through its verified HOST/NVMe path.

## 4. 可直接用于英文论文的文字

### 4.1 设计段落

> **Planning versus admission.** Residency planning and GPU admission operate at different time scales. The planner uses recent demand, locality, adapter size, and a residual tier-budget snapshot to identify adapters whose movement would have high expected readiness value. This plan is an optimization intent rather than a reservation of future GPU memory. Immediately before a proactive GPU promotion, PrimeLoRA recomputes available memory, near-term KV growth, uncovered working-set headroom, concurrent-load pressure, and measured GPU pressure. The promotion is committed only if its current utility exceeds contention and its size fits the resulting effective capacity.

### 4.2 为什么不合并段落

> Input/output lengths, batch composition, KV occupancy, and concurrent adapter loads can change between plan construction and execution. Checking feasibility only inside the planner would therefore commit from stale state; rerunning the full planner on every such event would move a bounded background optimization into the inference critical path. The second check is a commit-time safety guard, not a second solution of the residency objective.

### 4.3 fallback 段落

> A failed GPU-admission check does not reject inference. PrimeLoRA retains the candidate in, or resolves it from, the nearest verified lower local tier and passes that backend-addressable path to the serving runtime; explicit GPU residency is retried only after pressure changes or a later plan nominates the adapter. The prototype's GPU tier is control-plane state around the backend LoRA interface, so we publish a GPU-ready observation only after the corresponding backend warmup or invocation succeeds.
