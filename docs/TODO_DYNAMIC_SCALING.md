# TODO：动态扩缩容阈值实现

实现顺序按依赖关系排列；公式与参数见 `DYNAMIC_SCALING_THRESHOLDS_DESIGN.md`。

---

## 1. 状态与配置

- [ ] **1.1** 在配置中增加动态阈值开关与参数（如 `experiments.yaml` 的 `resource_coordination` 下）：
  - `dynamic_scaling: true/false`（关闭时沿用现有固定阈值）
  - `baseline_rps_ewma_beta`（默认 0.25）
  - `scale_up_alpha`（默认 0.3）、`scale_up_t_min`（默认 1.0）
  - `scale_down_beta`（默认 0.4）、`scale_down_duration_s`（默认 45）
  - （可选）`dynamic_warm_pool: true`、`warm_pool_gamma`、`warm_pool_min`、`warm_pool_max`

- [ ] **1.2** 在实验 runner 或 AutoScaler 可访问处维护运行时状态：
  - `baseline_rps`（float，初始可为 1.0 或首批 rps）
  - （scale-down）`low_load_since`（float | None，首次满足“低于 T_scale_down”的时间戳）
  - （可选）`active_loras_ewma`（float）、每批的 `active_loras_batch`（int）

---

## 2. Scale-up 动态阈值

- [ ] **2.1** 每批请求结束后，用当前 `rps_batch` 更新：
  - `baseline_rps = (1 - beta) * baseline_rps + beta * rps_batch`

- [ ] **2.2** 计算本批的 scale-up 触发线：
  - `T_scale_up = max(T_min, baseline_rps * (1 + alpha_up))`

- [ ] **2.3** 在调用 `make_scaling_decision_with_metrics` 前/内，若启用动态阈值：
  - 将 RPS 规则的 scale_up 判断改为：`rps_batch > T_scale_up`（或把 `scale_up_threshold_rps` 设为当次计算出的 `T_scale_up`，再走原有比较逻辑）
  - 保持 cooldown、max_instances 等既有约束不变

- [ ] **2.4** 若未启用动态阈值，保持现有固定 `scale_up_threshold_rps` 行为。

---

## 3. Scale-down 动态阈值

- [ ] **3.1** 每批请求结束后，计算：
  - `T_scale_down = baseline_rps * beta_down`
  - 判断本批是否“低负载”：`rps_batch < T_scale_down`（或使用 recent_rps 若单独维护）

- [ ] **3.2** 维护“持续低负载”计时：
  - 若当前批为低负载：若 `low_load_since` 为 None，则设为当前时间；否则不变
  - 若当前批非低负载：`low_load_since = None`

- [ ] **3.3** 在阶段结束原调用 `coordinator.trigger_scale_down()` 处：
  - 若启用动态 scale-down：仅当 `low_load_since` 非 None 且 `(now - low_load_since) >= D_scale_down` 时才调用 `trigger_scale_down()`
  - 若未启用：保持现有“阶段结束必调”行为

- [ ] **3.4** 多周期实验（E1）中，每个 phase 结束同样按上述条件决定是否调用 `trigger_scale_down()`；需保证每 phase 内能累计到足够的“低负载时长”或明确 phase 内是否用时间窗判断。

---

## 4. 动态 warm_pool_size（可选）

- [ ] **4.1** 每批统计本批内出现的不同 `adapter_id` 数量 → `active_loras_batch`

- [ ] **4.2** 更新 `active_loras_ewma = (1 - beta) * active_loras_ewma + beta * active_loras_batch`（可与 baseline_rps 共用 beta 或单独）

- [ ] **4.3** 在调用 `trigger_scale_down()` 时，若启用动态 warm pool：
  - 计算 `warm_pool_size = clip(round(active_loras_ewma * (1 + gamma)), min_warm, max_warm)`
  - 将该值传入 ResourceCoordinator（需扩展接口：`trigger_scale_down(warm_pool_size=...)` 或从配置/共享状态读取当次值）

- [ ] **4.4** ResourceCoordinator 内 `trigger_scale_down` 使用传入或读取的 `warm_pool_size`，而非固定配置值（当启用动态时）。

---

## 5. 实验与论文

- [ ] **5.1** 增加实验场景或配置：固定阈值 vs 动态阈值（同一 workload），对比 P99 TTFT、scale-up/scale-down 次数、eviction 次数、预加载浪费等

- [ ] **5.2** 在论文方法小节中引用 `DYNAMIC_SCALING_THRESHOLDS_DESIGN.md` 中的公式与符号，并说明参数默认值与可调范围

---

## 依赖关系简图

```
1.1 配置 + 1.2 状态
    → 2.1–2.4 scale-up 动态
    → 3.1–3.4 scale-down 动态
    → 4.1–4.4 动态 warm pool（可选）
    → 5.1–5.2 实验与论文
```

实现时建议先完成 1 → 2 → 3，验证无误后再做 4 和 5。
