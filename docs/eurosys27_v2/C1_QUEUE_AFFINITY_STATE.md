# C1：queue/affinity signals 的存储与更新

审稿人 C1 问的是一句具体实现问题：PrimeLoRA 所谓的 queue 和 affinity signals 到底是什么、存在哪里、何时更新，是每请求更新还是采样更新。答案不是二选一：当前设计使用 event-driven request/tier state，加上 sampled runtime state。

## 1. 论文规范语义

### 1.1 Signal 分类

论文中建议把笼统的 “queue and affinity signals” 改成以下三组：

1. **request occupancy signals**：该 runtime 已 reserve 的 active requests、每条 execution lane 的预计释放时间、可见 waiting work；
2. **adapter affinity/readiness signals**：目标 adapter 在该 replica 的已验证 tier、是否属于 scale-out handoff set、是否已经在 distinct active-LoRA set；
3. **sampled runtime pressure signals**：adapter-load backlog/pressure、GPU memory/utilization 和 backend/runtime health。

这三组状态的更新频率不同，不能全部称为“per request”，也不能全部称为“periodically sampled”。

### 1.2 规范状态表

| Signal | 规范含义 | Owner/存储 | 更新触发 | 更新模式 | stale/missing fallback |
|---|---|---|---|---|---|
| `active_requests_i` | 已 reserve 且未完成的请求数 | per-runtime controller slot | dispatch reserve、request finish/failure | event-driven，每请求 | stale-high 只会保守少调度；不得 stale-low |
| `lane_release_i[]` | 由 in-flight service estimate 得到的 lane 释放时间 | per-runtime controller slot | request reserve、finish、timeout pruning | event-driven + lazy prune | 无样本时只使用 active count/cap |
| `waiting_queue` | 已到达但尚未获得 runtime lane 的请求 | front-end/controller | arrival、admission、cancel | event-driven，每请求 | 缺失时不预测未到达请求 |
| `active_adapter_refs_i[a]` | runtime 当前执行 adapter `a` 的请求引用数 | per-runtime controller slot | lane reserve、request finish/failure | event-driven，每请求 | 异常清理必须释放；未知时保守视为占用 |
| `tier_i[a]` | adapter 在 replica/runtime 上最近一次成功验证的最近 tier | per-runtime tier hints；HOST/NVMe 可由 node-local path registry共享 | materialize/load/evict success、failure、显式 refresh | event-driven；后台 reconcile | 无/失败/冲突降级为 REMOTE/verified lower tier |
| `handoff_rank_i[a]` | fresh runtime 的有界 first-service adapter 优先级 | scale-out plan attached to slot | scale-out plan publish、request budget consume、expiry | 每次 scale-out + 每个落地 LoRA 请求 | 无 plan 时 neutral priority |
| `observed_cost_i[class]` | 对 tier/request class 的 I/O、runtime TTFT、tail occupancy 在线估计 | per-runtime controller slot | successful request completion | event-driven，每请求完成 | exact class无样本时回退到 LoRA-any/runtime/hardware model |
| `load_pressure_i` | 当前 outstanding/in-flight adapter load pressure | resource coordinator | load enqueue/start/finish/failure | event-driven，routing 可采样读取 | 无数据时设为 0，但 admission仍查 memory/KV |
| `gpu_pressure_i` | runtime GPU 的 logical/physical memory/utilization | resource coordinator/GPU monitor | monitor polling；routing/admission refresh | sampled | 使用可得的更保守值；probe失败时不得伪造低延迟命中 |
| `last_selected_i` | 最近一次 route reserve 时间 | per-runtime controller slot | successful reserve | event-driven，每请求 | 只作稳定 tie-breaker |

### 1.3 Routing 读取语义

每次 placement 必须读取一个快速 snapshot，但不要求对所有字段加分布式锁：

```text
snapshot_i = {
    active_requests,
    active_adapter_refs,
    verified_tier[target_adapter],
    handoff_rank/remaining_budget,
    observed_service_cost,
    most_recent_sampled_runtime_pressure,
    last_selected_at,
}
```

选择后，controller 在第一次异步等待之前原子 reserve `active_requests` 和 `active_adapter_refs[target]`。如果 reserve 发现 capacity 已改变，则本次 snapshot 作废并重选。tier/load 的过期状态通过保守 fallback 处理，而不是为每个请求同步扫描文件系统或阻塞查询 GPU。

### 1.4 更新时序示例

```text
request arrives
  -> insert into visible waiting queue
  -> refresh sampled hints only if refresh interval expired
  -> select replica from snapshot
  -> atomically reserve lane + active-adapter ref
  -> freeze readiness_tier_before_dispatch
  -> remove from waiting queue
  -> resolve nearest verified local path
  -> on successful tier change, publish new hint immediately
  -> invoke backend
  -> on completion, update observed service cost
  -> release active-adapter ref + lane and notify waiters
```

因此，request occupancy、active-LoRA 和完成统计是 per-request event-driven；GPU/runtime metrics 是 sampled；tier movement 在成功事件时立即更新，并辅以低频 reconciliation。

## 2. 当前实现证据

### 2.1 状态存储

当前 controller-process 的核心状态位于 [`faaslora/experiment/instance_pool.py`](../../faaslora/experiment/instance_pool.py)：

- `InstanceSlot.active_requests`：当前已 reserve 请求数；
- `active_adapter_counts`：adapter id 到 in-flight reference count；
- `inflight_request_deadlines`：用预计 busy time 维护的 lane release hints；
- `gpu_resident_adapters`、`host_cached_adapters`、`nvme_cached_adapters`：per-slot tier hints；
- `scaleup_handoff_planned_adapter_ranks`、`scaleup_handoff_request_budget`、`scaleup_handoff_assigned_requests`：handoff affinity；
- `observed_request_costs`：`backbone`、`lora_gpu`、`lora_host`、`lora_nvme`、`lora_remote`、`lora_any` buckets；
- `load_queue_depth`、`resident_lora_mb`、`gpu_utilization_pct`、`last_selected_at`：lightweight hints。

node-local registry/path 状态由 [`ExperimentStack`](../../faaslora/experiment/experiment_stack.py) 持有：

- `ArtifactRegistry` 保存 adapter 的 size、单值 `storage_tier`、path、status、access/load/hotness metadata；
- `_host_paths` 与 `_nvme_paths` 保存当前可见 local backing paths；
- `ResidencyManager.tier_artifacts` 保存各 tier 的控制面集合；
- `HotnessTracker` 保存 sliding access window 和 per-adapter EWMA 值。

### 2.2 事件驱动更新

| 事件 | 当前实现动作 | 代码证据 |
|---|---|---|
| dispatch reserve | 同步检查 runtime cap/active-LoRA cap，增加 `active_requests` 和 adapter ref，设置 `last_selected_at` | `ScenarioRunner._try_reserve_runtime_request_slot()` |
| completion/failure | `finally` 清除 lane deadline、减少 adapter ref/active requests、唤醒等待者 | `ScenarioRunner._exec_request()` 的 `finally` |
| request resolved | `ExperimentStack.record_access()` 更新 hotness 和 registry access stats | `ScenarioRunner._exec_request()`、`ExperimentStack.record_access()` |
| request completed | `slot.record_request_cost()` 更新 tier bucket 的 I/O、runtime TTFT 和 tail | `ScenarioRunner._exec_request()` |
| local path resolved | `_mark_slot_adapter_tier()` 发布该 adapter 的 slot tier hint | `ScenarioRunner._exec_request()` |
| scale-out warmup completed | 新 slot 对每个实际 warmed adapter 调用 `mark_adapter_tier(...,"gpu")` | `ScenarioRunner._add_dedicated_instance_slot()` |
| tier promotion/eviction | `ResidencyManager._update_artifact_tier()` 与 registry update | `ResidencyManager.admit_artifact()` / `evict_artifact()` |
| handoff request landed | `Router._consume_scaleup_handoff_budget_if_needed()` 增加 assigned count | `Router.select_instance()` |

### 2.3 采样与 reconciliation

- `ScenarioRunner._refresh_slot_runtime_hints()` 默认以 `runtime_hints_refresh_interval_s=0.5` 缓存 coordinator-derived hints。它在每轮 placement 前被调用，但间隔内直接复用 snapshot，避免每请求同步 bookkeeping 阻塞 replay event loop。
- 该 refresh 从 coordinator 读取 `current_lora_resident_mb` 与 logical GPU utilization，并在可用时取物理 GPU utilization 的更大值。
- `ExperimentStack.sync_local_tier_paths()` 默认 `local_tier_paths_sync_interval_s=60`，扫描 registry 与文件路径重建 HOST/NVMe maps；tier-changing caller 使用 `force=True` 立即发布。
- `GPUMemoryMonitor` 默认每 1 秒采样；`ResourceCoordinator` 自己的全局 memory probe 也有 interval/cache。

### 2.4 与投稿文字不完全一致的地方

1. **完成成本是 arithmetic mean。** `ObservedRequestCost.record()` 使用累计平均，不是 EWMA。论文可写“online observed service estimates”；若坚持写 EWMA，需要先修改并重新验证实现，本轮文档不把它写成事实。
2. **`load_queue_depth` 当前不是瞬时深度。** `ResourceCoordinator.get_summary_metrics()` 暴露的 `queued_loads` 来自累计 counter；`InstanceSlot.update_runtime_hints()` 把它赋给 `load_queue_depth`。因此当前结果只能把该字段称为 cumulative queued-load hint，不能把它解释为 current queue length。
3. **HOST/NVMe hints 在单节点上共享。** `_refresh_slot_runtime_hints()` 将 `ExperimentStack._host_paths/_nvme_paths` 复制到每个 slot；这符合 node-local shared cache，但不等于多节点 per-replica catalog。
4. **tier hint 不是物理 GPU load 的唯一证据。** `ResidencyManager` 的 GPU 状态与 backend LoRA executable state存在边界；V2 使用 `readiness_tier_before_dispatch`、实际 warmup/请求结果和不变量检查共同验证。
5. **异常采用 best-effort bookkeeping。** 部分 refresh/mark 操作捕获异常后继续。实验 analyzer 必须把 missing/conflicting readiness 当作 invalid evidence，不能静默补为 GPU hit。

## 3. 推荐的论文状态表述

为同时符合设计语义和当前单节点实现，建议使用：

- “per-runtime GPU hints and node-local HOST/NVMe path maps”，而不是笼统的 “global per-replica registry”；
- “online observed service estimates”，而不是 “all estimates are EWMA-updated”；
- “adapter-load pressure hint”，而不是在未修复 counter 前写 “instantaneous loading-queue depth”；
- “event-driven request/locality state plus periodically refreshed runtime pressure”，明确回答 reviewer 的 per-request vs sampled 问题。

## 4. 可直接用于英文论文的文字

### 4.1 主段落

> **Routing-state maintenance.** PrimeLoRA combines event-driven request and locality state with periodically refreshed runtime pressure. Each runtime slot stores its reserved request count, distinct active-LoRA reference counts, estimated lane-release times, per-runtime GPU locality hints, node-local HOST/NVMe path hints, bounded scale-out handoff ranks, and online service-cost observations. Request occupancy and active-LoRA references are updated at dispatch and completion; locality is published after a successful movement or path resolution; and observed I/O, first-token, and tail-service costs are updated when a request completes. GPU utilization and adapter-load pressure are refreshed from the coordinator at a short interval rather than synchronously probing the runtime for every request.

### 4.2 一致性与 fallback

> **Snapshot and fallback semantics.** Placement reads a lightweight snapshot of these signals and then reserves the selected runtime lane and adapter reference atomically before yielding the controller loop. A failed reservation invalidates the snapshot and triggers reselection. Missing, failed, or conflicting locality is treated as remote (or as the last verified lower tier), so a stale hint cannot manufacture a GPU hit. The single-node prototype shares HOST/NVMe path maps across runtimes on the inference node, while GPU locality and active-LoRA state remain runtime-specific.

### 4.3 若当前代码不修改时的更保守版本

> The prototype updates request occupancy and adapter references on every dispatch/completion event and refreshes coordinator-derived pressure hints at a 0.5-s interval. Its reported service estimates are online running averages. We use the cumulative queued-load counter only as a load-pressure hint, not as a measurement of instantaneous queue depth.
