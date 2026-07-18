# C2：active-LoRA feasibility 的严格定义

“Active-LoRA feasibility” 不是 adapter locality 的同义词。它回答的是：在 backend 对**同时执行的不同 adapter 数量**设有上限时，把目标请求送入某个 runtime 是否会超过该限制。

## 1. 论文规范语义

### 1.1 定义

对 runtime `i`，令：

- `A_i^active(t)`：时刻 `t` 至少有一个 in-flight request 正在使用的 distinct adapter 集合；
- `K_i`：backend/runtime 允许同时 active 的 distinct LoRA adapters 上限；
- `a(r)`：请求 `r` 的目标 adapter。

则 active-LoRA feasibility 为：

```text
active_lora_feasible(i,r,t) =
    true,                                      if r is backbone-only
    true,                                      if K_i <= 0 (unbounded/unspecified)
    true,                                      if a(r) in A_i^active(t)
    |A_i^active(t)| < K_i,                     otherwise
```

最后一种情况成立时，请求会引入一个新的 distinct active adapter；若集合已满，则该 runtime 暂时不可接收这个新 adapter。若目标 adapter 已经 active，即使集合大小等于 `K_i`，新请求也不会增加 distinct count，因此仍可行。

### 1.2 与其他概念的区别

| 概念 | 问题 | 是否等同 active-LoRA feasibility |
|---|---|---|
| GPU residency/readiness | adapter weights 是否已在该 runtime 的 GPU/executable path上 | 否；resident adapter 可以没有 active request |
| HOST/NVMe locality | adapter 是否有较近的 local backing path | 否；local 不代表已进入 active set |
| request concurrency | runtime 是否还有 request execution lane | 否；它是另一个独立 feasibility condition |
| queue length | 有多少请求等待 | 否；queue 可为 0，但 distinct active set 仍可能满 |
| adapter cache capacity | GPU/host 能缓存多少 adapter bytes | 否；这是 residency budget，不是 backend active-set slot |
| adapter refcount | 当前有多少 in-flight requests 使用某 adapter | refcount > 0 决定它是否属于 active set，但 feasibility 使用 distinct count |

因此论文 Eq. 4 中的 `Pi_i(r,t)` 应同时包含两个独立布尔项：runtime-lane feasibility 和 active-LoRA feasibility。tier/readiness 进入后续 service estimate，不应混入 active-set 定义。

### 1.3 状态更新与并发安全

规范实现维护 `active_adapter_refs_i[a]`：

```text
RESERVE(i, r):
    assert runtime_lane_available(i)
    assert active_lora_feasible(i, r, now)
    active_requests_i += 1
    if a(r) exists:
        active_adapter_refs_i[a(r)] += 1

RELEASE(i, r):
    active_requests_i -= 1
    if a(r) exists:
        active_adapter_refs_i[a(r)] -= 1
        if active_adapter_refs_i[a(r)] == 0:
            remove a(r) from active_adapter_refs_i
```

feasibility check 与 reserve 必须在同一 controller critical section、第一次异步等待之前完成；completion、failure、timeout/cancel 都必须走 `RELEASE`。否则两个并发请求可能都看到一个空 active slot并同时引入不同 adapter。

### 1.4 边界示例

假设 `K_i=2`：

| 当前 refcounts | 新请求 | 结果 | 原因 |
|---|---|---|---|
| `{a:2}` | `a` | 可行 | distinct count 仍为 1 |
| `{a:2}` | `b` | 可行 | 新 distinct count 为 2 |
| `{a:2,b:1}` | `a` | 可行 | `a` 已 active |
| `{a:2,b:1}` | `c` | 不可行 | 会把 distinct count 从 2 增至 3 |
| `{a:2,b:1}` | backbone-only | 可行 | 不占 LoRA active slot；仍需检查 request lane |
| `{}` 且 request lanes 已满 | `a` | active-LoRA 可行、runtime 不可行 | 两个 feasibility 条件必须分别计算 |

## 2. 当前实现证据

### 2.1 已实现路径

[`InstanceSlot`](../../faaslora/experiment/instance_pool.py) 中：

- `active_adapter_counts: Dict[str,int]` 保存 in-flight references；
- `active_adapter_count()` 只统计 refcount > 0 的 distinct adapter；
- `can_accept_active_adapter(adapter_id,max_active_loras)` 实现上述三种 LoRA 情况：cap 不设限、target 已 active、distinct count 仍有空间；
- `begin_active_adapter()` 增加 refcount；
- `end_active_adapter()` 减少 refcount并在 0 时删除 key。

`Router._routing_key()` 分别计算：

- `_runtime_capacity_penalty()`：execution lane 或 runtime forwarding 状态；
- `_capacity_penalty()`：active-LoRA feasibility。

它们排在 handoff、readiness/service cost 和 load hints 之前，因此可行 runtime 总是优先于仅具有更好 affinity 但 active set 已满的 runtime。

[`ScenarioRunner`](../../scripts/run_all_experiments.py) 中：

- `_runtime_max_active_loras()` 从 runner model config 的 `max_loras` 读取 `K_i`；
- `_try_reserve_runtime_request_slot()` 先检查 runtime lane 与 `can_accept_active_adapter()`，再同步增加 `active_requests` 和 adapter ref；该函数在 route selection 后、任何 `await` 之前调用；
- reserve 失败时 `_exec_request()` 等待 capacity condition 后重新选择；
- request success/failure 的公共 `finally` 调用 `end_active_adapter()` 并减少 `active_requests`。

### 2.2 当前实现的证据边界

1. `active_adapter_counts` 是 PrimeLoRA controller 的 shadow state，不是对 backend 内部所有 LoRA activity 的反向查询。它在 formal harness 中成立的前提是所有请求都经过同一 controller；外部旁路请求不会自动出现。
2. `max_loras` 是配置/后台接口给出的 runtime cap；它不是按空闲 GPU bytes动态计算的 residency capacity。GPU bytes 由独立的 ResourceCoordinator admission处理。
3. Router 采用 penalty + reserve retry，而不是预先删除所有 infeasible replicas。由于 feasibility penalty 是 lexicographic leading component，有可行 replica 时仍会优先选择可行者；若全部不可行，reserve 失败并等待状态改变。
4. active ref 在 route reserve 时增加，早于真正 backend generation。其语义是“已被该 runtime 接纳、占用 active-LoRA execution slot”，而不是“已经生成第一个 token”。论文应使用 admitted/in-flight，而不是 first-token-only定义。
5. 当前结果需要新增/核对 instrumentation，才能直接报告每次 decision 的 `distinct_active_count`、`max_loras` 和 target-already-active 分支；仅凭最终 latency 不能证明该 feasibility 分支触发。

## 3. 推荐写入论文的公式

可以把 Eq. 4 前的 feasibility 解释为：

```text
Pi_i(r,t) = (
    1[runtime_i is forwarding or active_requests_i >= K_i^req],
    1[a(r) notin A_i^active(t) and |A_i^active(t)| >= K_i^LoRA]
)
```

其中两个 indicator lexicographically 先于 handoff/service/load components。这里 `K_i^req` 是 request concurrency cap，`K_i^LoRA` 是 distinct active-LoRA cap，不能共用同一符号或描述。

## 4. 可直接用于英文论文的文字

### 4.1 定义段落

> **Active-LoRA feasibility.** This term refers to a backend execution limit, not to adapter residency. Let \(A_i^{\mathrm{active}}(t)\) be the set of distinct adapters referenced by requests already admitted to runtime \(i\), and let \(K_i^{\mathrm{LoRA}}\) be the backend's distinct active-adapter limit. A request for adapter \(a\) is feasible if \(a\in A_i^{\mathrm{active}}(t)\), because it does not consume another distinct adapter slot, or if \(|A_i^{\mathrm{active}}(t)|<K_i^{\mathrm{LoRA}}\). Backbone-only requests do not consume a LoRA slot. This check is separate from both request-concurrency feasibility and the adapter's GPU/HOST/NVMe residency.

### 4.2 实现段落

> The controller keeps a per-runtime reference count for every admitted adapter. It checks the request-lane limit and the distinct active-LoRA limit, increments both the runtime occupancy and the target adapter's reference count atomically at dispatch, and releases them on every success or failure path. Consequently, concurrent requests for an already active adapter can share the same distinct-adapter slot, whereas a request that would introduce a new adapter waits or is routed elsewhere when the active set is full.

### 4.3 限定段落

> In our prototype, this state is maintained by the PrimeLoRA controller and is valid because all evaluated requests pass through that controller. It is not inferred from GPU cache occupancy, and it does not account for requests that bypass the controller.
