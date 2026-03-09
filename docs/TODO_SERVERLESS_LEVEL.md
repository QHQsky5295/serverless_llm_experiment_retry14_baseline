# FaaSLoRA 对标 ServerlessLLM / ServerlessLora 级别 — 完整修复 TODO

本文档列出为达到与 ServerlessLLM (NSDI'24)、ServerlessLora 等**真实 Serverless 大模型推理系统**同级实现与评估所需完成的全部修复与增强。按模块与依赖关系组织，便于按序实施。**无仿真**：实验路径使用与线上一致的决策逻辑与真实加载路径。

---

## 一、扩缩容与预加载闭环

| ID | 任务 | 说明 | 依赖 | 状态 |
|----|------|------|------|------|
| **A1** | **实验路径接入真实扩缩决策** | 在 `run_all_experiments.py` 或统一入口中，不再“按场景一次性预加载”，而是引入扩缩决策：当队列长度/延迟/RPS 超过阈值时触发 scale-up，再触发预加载。可先复用 `faaslora/coordination/autoscaler.py` 的 `ScalingDecision`，在实验脚本里模拟“每 N 个请求或每 T 秒评估一次是否扩容”。 | 无 | **已完成**：实验路径每批用 `AutoScaler.make_scaling_decision_with_metrics(metrics, current_instances=pool.count())` 原样复用同一套规则与 cooldown，`SCALE_UP` 时调用 `trigger_scaling_preload`，若有 `engine_factory` 则 `add_instance`。 |
| **A2** | **扩缩事件触发预加载** | 在 scale-up 决策发生后，调用 `PreloadingManager.trigger_scaling_preload(scaling_event)`，将 `scaling_event` 包含 target_tier、capacity_bytes、type=scale_up 等，使预加载计划在“新实例/新容量”到来前生成并执行。 | A1 | **已完成**：ExperimentStack.trigger_scaling_preload 已传入 `type`、`target_tier`、`capacity_bytes`，与 PreloadingManager 约定一致。 |
| **A3** | **预加载输入改为在线热度** | `HotnessTracker` 在每次 `record_access` 后写回 `ArtifactRegistry` 的 `hotness_score`、`value_per_byte`、`access_count`；PreloadingPlanner 从 Registry 读取，scale-up 预加载使用请求流驱动的热度。 | 无 | **已完成** |

---

## 二、多实例与请求路由

| ID | 任务 | 说明 | 依赖 | 状态 |
|----|------|------|------|------|
| **B1** | **多实例/多 Worker 抽象** | 引入“实例”抽象：每个实例对应一个推理引擎（vLLM engine）+ 一份 ResourceCoordinator。支持实例池大小 > 1，scale-up = 增加实例、scale-down = 回收实例。 | 无 | **已完成**：InstancePool 已接入；scale-up 时若传入可选 `engine_factory` 则 `add_instance(新 engine, 新 coordinator)`；未传 engine_factory 则 scale-up 不增加新 engine。槽位数由 `max_instances` 决定，默认 2 槽位。 |
| **B2** | **请求路由** | 请求到达时经 Router 选择目标实例，再转发到对应 engine + coordinator。 | B1 | **已完成**：Router(pool) 已在 _exec_request 中通过 select_instance 选实例，请求使用 slot.engine / slot.coordinator。 |
| **B3** | **实例生命周期与 warm pool 可观测** | scale-down 时释放实例或保留暖池；记录“scale-down 后首次 scale-up 是否命中暖池”、“冷启动次数”等。 | A1, B1 | **已完成**：单/多周期已调用 `trigger_scale_down()` 并记录各项指标；多实例时对超出 min_instances 的实例调用 `remove_instance(id)` 回收。 |

---

## 三、实验路径与完整栈统一

| ID | 任务 | 说明 | 依赖 | 状态 |
|----|------|------|------|------|
| **C1** | **主实验使用 ResidencyManager 分层** | faaslora_* 场景下 tier 解析已由 `_stack.resolve_lora` → ResidencyManager.get_tier_status / admit_artifact 完成；ScenarioRunner 请求路径不再依赖手写 _gpu_warmed/_nvme_cache 判断 tier。 | 无 | **已完成** |
| **C2** | **主实验使用 PreloadingPlanner + PreloadingManager** | 预加载阶段已改为 `PreloadingPlanner.generate_preloading_plan` 生成计划，再按计划拷贝并 `admit_artifact` 到 NVME；与 faaslora/preloading 一致。 | A2 可选 | **已完成** |
| **C3** | **ResourceCoordinator 与 MemoryCoordinator / ResidencyManager 对齐** | 明确分工：ResourceCoordinator 负责“单实例内”的显存预算、排队、scale-down 暖池；ResidencyManager 负责 tier 间迁移与准入驱逐。当 ResourceCoordinator 构造时传入 `residency_manager` 时，`_available_mb`、`_get_resident_loras`、驱逐均使用 ResidencyManager.get_tier_status(GPU)，显存视图已对齐。 | C1 | **已完成** |

---

## 四、真实 LoRA 加载与协调效果

| ID | 任务 | 说明 | 依赖 | 状态 |
|----|------|------|------|------|
| **D1** | **FaaSLoRA 路径真实加载 LoRA 到 vLLM** | 在 faaslora 场景下，LoRA 需要加载到 GPU 时，调用 vLLM 的 load_lora / add_lora 等真实 API（若引擎支持），并测量实际耗时；而不是仅用 `asyncio.sleep(io_ms/1000)` 或只加 lora_io_ms。这样“协调”带来的排队与真实显存压力可在同一套加载路径上体现。 | 无 | **已完成**：warmup 与请求路径均用 `load_lora_to_gpu_and_measure` 做真实 vLLM 加载；测得的 `last_load_time_ms` / `predicted_load_time_ms` 写回 ArtifactRegistry；请求路径 NVMe→GPU 时用真实测量覆盖公式估计的 lora_io_ms，并 `admit_artifact(..., GPU)` 更新驻留。 |
| **D2** | **协调效果基于真实显存压力** | 保持当前逻辑：有协调时在显存不足则排队等待（defer）；无协调时强制驱逐。确保“显存不足”的判断来自真实状态（如 vLLM/KV 占用 + 当前 LoRA 占用），或来自与单卡显存上限一致的模型（gpu_budget_mb 等），而不是人为注入的随机惩罚。当前已去掉 simulate_burst_contention，此处只需确认 _available_mb 与真实或校准参数一致。 | 已部分完成 | 已部分完成 |

---

## 五、评估与可复现性

| ID | 任务 | 说明 | 依赖 | 状态 |
|----|------|------|------|------|
| **E1** | **多实例/多周期评估** | 实验设计支持“多轮 scale-up → 请求 → scale-down → 再 scale-up → 请求”，并汇报：每轮冷启动次数、暖池命中率、scale-up 事件与冷启动统计。 | B1, B3 | **已完成**：已支持 `multi_cycle_phases`、`idle_between_phases_s`，每 phase 结束 scale_down；ScenarioResult 含 `scale_up_events`、`cold_starts_after_scale_up`、`multi_cycle_phase_results`、`scale_down_events`、`warm_pool_retained_after_phase`，并写入 JSON 与终端输出。 |
| **E2** | **与 SOTA 的对比方式文档化** | 在 README 或 EXPERIMENT_GUIDE 中说明：与 S-LoRA、ServerlessLLM 的对比方式（同一代码库 baseline 或公式复现）。 | 无 | **已完成**：EXPERIMENT_GUIDE 第 6.6 节已说明采用同一代码库 baseline（slora_style / serverlessllm），不做延迟公式复现。 |
| **E3** | **公开配置与脚本** | 保证 `configs/experiments.yaml` 与主要脚本足以复现论文主表；单条命令与配置说明。 | 无 | **已完成**：EXPERIMENT_GUIDE 附录 A「论文主表复现（E3）」已给出单条命令、配置文件与结果路径，并指向 docs/ENVIRONMENT.md。 |

---

## 六、文档与代码一致性

| ID | 任务 | 说明 | 依赖 | 状态 |
|----|------|------|------|------|
| **F1** | **PROJECT_STRUCTURE / README 与实现一致** | 预加载由 `PreloadingPlanner` + `PreloadingManager` 完成；贡献 2 对应 ResidencyManager 的 GPU/Host/NVMe/Remote；完整栈默认开启。 | C1, C2 | **已完成**：PROJECT_STRUCTURE 已补充 coordination/、修正 serving/inference_engine.py、贡献2 与 LocalCache 表述；README 已更新完整栈默认与 --no-full-stack。 |
| **F2** | **贡献 3 表述** | 文档中明确：协调机制不依赖任何“概率注入”的竞争，仅依赖真实显存压力下的排队与驱逐；指标中的 contention_events / defer 等含义与代码一致。 | 已部分完成 | 已部分完成 |

---

## 七、建议实施顺序（按依赖与优先级）

1. **先统一实验与栈**：C1 → C2 → C3（实验路径用上 ResidencyManager + PreloadingManager，消除两套实现）。
2. **再打通扩缩与预加载**：A3（在线热度）→ A1（扩缩决策）→ A2（scale-up 触发预加载）。
3. **然后多实例与路由**：B1 → B2 → B3，并与 A1/A2 结合，使 scale-up 真正增加实例、预加载绑定到新实例。
4. **真实加载与评估**：D1（真实 LoRA 加载）→ D2 校验 → E1（多周期评估）→ E2/E3（文档与复现）。
5. **文档收尾**：F1 → F2。

---

## 八、与“最小可发表”的区别

- **最小可发表**：单实例 + 启动时预加载 + 延迟模型对比即可支撑贡献点。
- **ServerlessLLM/ServerlessLora 级别**：需要**真实扩缩语义**（多实例或至少多 worker）、**扩缩触发预加载**、**统一的分层驻留与预加载栈**、**真实加载路径上的协调效果**，以及**多周期/多实例下的评估与可复现说明**。本 TODO 覆盖上述全部项。

完成本 TODO 后，系统在“扩缩容 + 预加载 + 分层驻留 + 协调”上与论文描述一致，且评估与对标方式可经受审稿与复现要求。

---

## 九、统一 TODO 列表（真实 Serverless、无仿真）

在**真实 serverless 系统、无仿真**前提下，按优先级整理的下一步与未修复项。

### 已完成（本次/近期）

- **A1**：实验路径每批用 `AutoScaler.make_scaling_decision_with_metrics` 原样复用决策逻辑，SCALE_UP 时调用 `trigger_scaling_preload`；配置从 `coord_cfg` 注入 `coordination.autoscaling`，ExperimentStack 暴露 `autoscaler`（不 start 后台）。
- **A2**：ExperimentStack.trigger_scaling_preload 已传入 `type`、`target_tier`、`capacity_bytes`，与 PreloadingManager 约定一致。
- **A3**：HotnessTracker 写回 `hotness_score`、`value_per_byte`、`access_count`；PreloadingPlanner 从 Registry 读取。
- **B2**：Router(pool) 已在 _exec_request 中 select_instance，请求使用 slot.engine / slot.coordinator。
- **B1**：scale-up 时若提供可选 `engine_factory` 则 `add_instance`；否则 scale-up 不增加新 engine。槽位数由 `max_instances` 决定，默认 2 槽位。**B3**：多实例时 scale-down 后对超出 min_instances 的实例 `remove_instance`。
- **C1 / C2 / C3**：tier 由 ResidencyManager；预加载由 PreloadingPlanner 生成计划并执行；ResourceCoordinator 与 ResidencyManager 显存视图对齐。
- **D1**：warmup 与请求路径均用 `load_lora_to_gpu_and_measure`，耗时写回 Registry；NVMe→GPU 时用真实测量覆盖 lora_io_ms 并更新 GPU 驻留。
- **E1**：多周期 phase、scale_up_events、cold_starts_after_scale_up 已实现并写入结果与 JSON。
- **E2**：EXPERIMENT_GUIDE 6.6 已说明与 SOTA 的对比方式（同一代码库 baseline）。
- **E3**：EXPERIMENT_GUIDE 附录 A 已给出论文主表复现单条命令与配置说明。
- **F1**：PROJECT_STRUCTURE / README 已与实现一致（coordination/、serving/inference_engine、贡献2、完整栈默认）。

### 待办（可选扩展）

- 无；B1/B3 多实例路径已实现（需传入可选 `engine_factory` 才会真正增加实例，见下节）。

### 单实例 vs 多实例 / 对齐已有论文应采用哪种

- **默认多实例**：`configs/experiments.yaml` 中 `max_instances=2`，启动即多槽位，每槽位独立 ResourceCoordinator、共享同一 engine（单卡）；请求经 Router 选实例，tier 解析与协调按槽位使用对应 coordinator。
- **单实例**：将 `max_instances` 设为 1 即可；适合消融或资源受限对比。
- **多实例 + 新 engine**：scale-up 时若传入 `engine_factory` 则 `add_instance(新 engine, 新 coordinator)`，新实例会做热点 warmup；scale-down 时对超出 `min_instances` 的实例 `remove_instance(id)`。与 ServerlessLLM、S-LoRA 等多副本扩缩语义一致。

### 已知问题（无仿真前提下需注意）

- **默认多实例**：当前默认 2 槽位（同 engine、多 coordinator）；单实例请设 `max_instances=1`。
- **配置映射**：`configs/experiments.yaml` 的 `scale_up_threshold_rps` 已由 ExperimentStack 注入 `coordination.autoscaling`，AutoScaler 已增加 RPS 规则并读取该值，YAML 与扩缩决策一致。
