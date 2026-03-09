# 动态扩缩容阈值设计（论文友好，未改代码）

本文档给出 scale-up / scale-down 阈值的**动态计算方式**，便于论文表述与后续实现。当前代码仍使用固定阈值；实现时按本设计接入。

---

## 一、目标与符号

- **目标**：用近期负载的统计量计算“何时触发 scale-up 预加载”和“何时触发 scale-down（并保留 warm pool）”，使策略 workload-adaptive，减少对固定常数的依赖。
- **输入（每批请求后可得）**：
  - `rps_batch`：本批次的 RPS = batch_size / Δt。
  - （可选）本批内命中的不同 LoRA 数量、队列长度、GPU 利用率等。
- **输出**：
  - 本批是否触发 **scale-up**（预加载 / 扩容）；
  - 本阶段结束时是否触发 **scale-down**（驱逐冷 LoRA、保留 warm pool）；
  - （可选）本阶段 **warm_pool_size** 的动态取值。

---

## 二、Scale-up 动态阈值（何时触发预加载/扩容）

### 2.1 公式（推荐用于论文）

- **基线 RPS**（指数移动平均，平滑瞬时抖动）  
  ```text
  baseline_rps ← (1 − β) · baseline_rps + β · rps_batch
  ```  
  其中 β ∈ (0, 1)，如 β = 0.2～0.3；初始可设 `baseline_rps = rps_batch` 或一较小常数（如 1.0）。

- **Scale-up 触发线**  
  ```text
  T_scale_up = max(T_min, baseline_rps · (1 + α_up))
  ```  
  - `α_up`：相对基线的上浮比例，建议 α_up ∈ [0.2, 0.5]（如 0.3）。  
  - `T_min`：绝对下限，避免基线极低时永不触发，如 T_min = 1.0。

- **触发条件**  
  ```text
  若 rps_batch > T_scale_up 且满足 cooldown、实例数上限等既有约束，则触发 scale-up（预加载 + 可选扩容）。
  ```

### 2.2 论文表述要点

- “我们采用 **EWMA 基线 + 相对上浮** 的动态阈值：仅当当前 batch RPS **超过** 近期基线 RPS 的 (1+α_up) 倍且不低于 T_min 时触发 scale-up，从而在高负载时提高触发线、在低负载时仍能响应，减少对固定 RPS 阈值的依赖。”

---

## 三、Scale-down 动态阈值（何时触发缩容/驱逐冷 LoRA）

### 3.1 公式（推荐用于论文）

- **Scale-down 触发线**（相对基线向下）  
  ```text
  T_scale_down = baseline_rps · β_down
  ```  
  - `β_down` ∈ (0, 1)，如 β_down = 0.3～0.5，表示“负载降到基线的 30%～50% 以下才考虑缩容”。

- **近期负载**（用于判断是否“持续低负载”）  
  - 用同一 EWMA 的 `baseline_rps` 即可；或单独维护“近期 RPS”的滑动平均：  
    ```text
    recent_rps ← (1 − β) · recent_rps + β · rps_batch
    ```  
  - 为简单起见，可与 scale-up 共用同一 `baseline_rps`（每次用 `rps_batch` 更新后，用当前 `baseline_rps` 与 `T_scale_down` 比较）。

- **持续低负载条件（防抖动）**  
  - 仅当 **recent_rps < T_scale_down** 的持续时间 ≥ `D_scale_down`（如 30～60 秒）时，才执行 scale-down（调用 `trigger_scale_down()` 并可选回收实例）。  
  - 实现方式：维护 `low_load_since`（首次满足 recent_rps < T_scale_down 的时间戳）；若当前仍满足且 `now - low_load_since ≥ D_scale_down`，则触发 scale-down；否则不触发。若某批 recent_rps ≥ T_scale_down，则重置 `low_load_since = None`。

### 3.2 论文表述要点

- “Scale-down 采用 **相对基线的下界** T_scale_down = baseline_rps · β_down，并要求 **持续低负载**（近期 RPS 低于 T_scale_down 且持续 D_scale_down 秒）后才执行，避免负载短暂回落导致的频繁缩容与抖动。”

---

## 四、动态 warm_pool_size（可选）

- **思路**：保留的 LoRA 数量与近期“活跃” LoRA 数相关，而非固定常数。  
  ```text
  active_loras_batch = 本批内请求命中的不同 adapter_id 的数量
  active_loras_ewma ← (1 − β) · active_loras_ewma + β · active_loras_batch
  warm_pool_size = clip(round(active_loras_ewma · (1 + γ)), min_warm, max_warm)
  ```  
  - γ 为余量（如 0.2），min_warm / max_warm 为上下界（如 2 与 8）。  
- **论文**：可写为 “warm pool 大小随近期活跃 LoRA 数的 EWMA 自适应，并做上下界裁剪”。

---

## 五、参数汇总（建议默认与可调范围）

| 符号 | 含义 | 建议默认 | 可调范围 |
|------|------|----------|----------|
| β | EWMA 更新系数（baseline_rps） | 0.2～0.3 | 0.1～0.5 |
| α_up | scale-up 相对基线上浮比例 | 0.3 | 0.2～0.5 |
| T_min | scale-up 绝对下限 (RPS) | 1.0 | 0.5～2.0 |
| β_down | scale-down 相对基线比例 | 0.4 | 0.3～0.5 |
| D_scale_down | 持续低负载时长 (秒) | 45 | 30～60 |
| γ (可选) | warm pool 相对 active_loras 余量 | 0.2 | 0.1～0.3 |
| min_warm / max_warm (可选) | warm pool 数量上下界 | 2, 8 | 1～4, 6～12 |

---

## 六、与现有代码的对接点（供实现 TODO 使用）

- **Scale-up**：当前在 `run_all_experiments.py` 中每批后用 `ScalingMetrics(requests_per_second=batch_rps, ...)` 调用 `autoscaler.make_scaling_decision_with_metrics`；AutoScaler 内 RPS 规则使用固定 `scale_up_threshold_rps` 与 `scale_down_threshold`。  
  - 实现时：在实验 runner 或 AutoScaler 内维护 `baseline_rps`，每批更新；将 **scale_up 判断** 改为 `rps_batch > T_scale_up`（T_scale_up 按上式计算）；可保留原有 cooldown、max_instances 等逻辑。
- **Scale-down**：当前在阶段结束后无条件调用 `coordinator.trigger_scale_down()`。  
  - 实现时：在阶段内每批更新 `recent_rps` 与 `T_scale_down`，维护 `low_load_since`；**仅在** 满足“持续低负载 ≥ D_scale_down”时在阶段末调用 `trigger_scale_down()`。
- **Warm pool 大小**：当前 `ResourceCoordinator` 与配置使用固定 `warm_pool_size`。  
  - 实现时：在 runner 或 coordinator 中按批更新 `active_loras_ewma`，在调用 `trigger_scale_down()` 前计算当次 `warm_pool_size` 并传入（或让 coordinator 从共享状态读取）。

以上计算方式可直接用于论文“方法”或“系统设计”小节；实现时按下方 TODO 逐步接入代码即可。
