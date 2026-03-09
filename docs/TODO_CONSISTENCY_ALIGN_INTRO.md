# 整体一致性与论文引言对齐 TODO（扫描报告）

**说明**：按「论文引言」（README 研究背景与动机、三项贡献）与全项目单实例/多实例、预加载、协调路径做了一致性扫描。**1～6 已按“默认多实例”全部修改完成**（见第四节）。

---

## 一、论文引言要点（对齐基准）

- **观察1**：扩容实例缺乏工件命中感知，**新实例仍需冷加载 LoRA** → 贡献1 命中感知预加载  
- **贡献1**：扩容决策时，调度器将高频 LoRA 工件**提前加载至待启动实例**；**新实例启动后无需冷加载，直接命中缓存**，大幅降低 TTFT  
- **贡献2**：热–温–冷分层、动态迁移  
- **贡献3**：扩容阶段预留/渐进加载；负载回落阶段保留**暖池**；竞争来自真实显存压力  

实验架构：请求 → WorkloadDataset → **ScenarioRunner**（缓存层、预加载、ResourceCoordinator）→ **InferenceEngine** → 结果  

---

## 二、已一致、无需修改的部分

| 项目 | 说明 |
|------|------|
| 扩缩决策与 current_instances | 实验路径已用 `pool.count()` 传入 `make_scaling_decision_with_metrics`，与实例数一致。 |
| scale-up / scale-down 与实例池 | scale-up 时若有 `engine_factory` 则 `add_instance`；scale-down 时多实例则 `remove_instance(slots[-1])`，逻辑一致。 |
| 请求路径 engine/coordinator 使用 | B2 路由选 slot 后，`_exec_request` 使用 `slot.engine`、`slot.coordinator` 做推理与 `request_lora_load` 等，与多槽位设计一致。 |
| 预加载与热度 | PreloadingPlanner 从 Registry 读热度；HotnessTracker 写回；trigger_scaling_preload 在 scale-up 时触发，与贡献1 表述一致。 |
| 贡献2/3 实现位置 | ResidencyManager 分层、ResourceCoordinator 暖池与协调，与 README/PROJECT_STRUCTURE 描述一致。 |
| 配置 | `scale_up_threshold_rps` 等从 YAML 注入 coord_cfg → ExperimentStack → AutoScaler，一致。 |
| 文档 | TODO_SERVERLESS_LEVEL、EXPERIMENT_GUIDE 已说明单实例 vs 多实例及对齐论文建议。 |

---

## 三、存在的一致性问题（TODO）

**说明**：以下为修复前的描述（均已修复，见第四节）。若再次扫描后仍有遗留项，见 `docs/TODO_CONSISTENCY_RESCAN.md`。

### 1. 【多实例】faaslora 路径下 tier 解析未按实例分流（单 coordinator 假设）

- **现象**：`_resolve_lora` 在 `self._stack is not None` 且 faaslora_* 时，直接调用 `self._stack.resolve_lora(...)` 并**早退**，不接收、不使用参数 `coordinator=_coord`。  
- **结果**：ExperimentStack.resolve_lora 内部始终使用 **self.coordinator**（栈的唯一 coordinator），即所有请求的 tier 判断、request_lora_load、warm_pool_hit 均走**主实例**的 coordinator 与同一份 ResidencyManager/Registry。  
- **与引言关系**：引言未明确说「每实例独立 coordinator」，但 B2 已按 slot 选实例并传入 `_coord`；实现上 faaslora 路径**忽略了** slot.coordinator，多实例时语义仍为「单实例协调 + 多 engine 推理」。  
- **TODO**：  
  - 若坚持「每实例独立协调与显存视图」：需在 faaslora 路径支持按请求使用 `_coord` 做 tier/加载决策（例如 _stack.resolve_lora 接受可选 coordinator，或按 slot 走不同 resolve 分支）。  
  - 若接受「当前为单协调多 engine」：在文档（如 TODO_SERVERLESS_LEVEL 或本文件）中明确写明「faaslora_* 下 tier 与协调决策由单一 ExperimentStack.coordinator 负责，多实例时仅推理按 slot 分流」。

---

### 2. 【多实例】预加载与 warmup 仅作用于主实例

- **现象**：`preload` / `_preload_full_stack` 只对 **self.engine** 做 NVMe 预加载与 GPU warmup；通过 `engine_factory` 在 scale-up 时加入的实例**不参与**预加载阶段，也无独立 warmup。  
- **与引言关系**：引言称「新实例启动后无需冷加载，直接命中缓存」；当前实现下，**新加实例**（若有）启动时无预加载、无 warmup，与引言表述有差距。  
- **TODO**：  
  - **文档**：在 README 或 EXPERIMENT_GUIDE 中注明「默认单实例；多实例时由 engine_factory 提供的实例当前不参与预加载与 GPU warmup，新实例可能面临冷加载，与论文多副本语义的完全对齐需后续扩展」。  
  - **实现（可选）**：若需严格对齐「新实例直接命中」，可考虑新实例共享同一 NVMe/Registry 视图并在加入后执行一次 warmup，或由调用方在 engine_factory 内完成预热。

---

### 3. 【文档】README/引言未明确默认单实例与多实例可选性

- **现象**：README 使用「扩容实例」「新实例」「待启动实例」等用语，未说明当前**默认单实例**、多实例为**可选**（需传入 `engine_factory`）及其实验/资源含义。  
- **与引言关系**：避免审稿/读者误以为实现已是「多实例默认」或「新实例必然预加载」。  
- **TODO**：在 README「研究背景与动机」或「贡献1」下增加 1～2 句：当前实验默认**单实例**（一个 engine + 一个 coordinator），扩缩与预加载逻辑均生效；多实例为可选，需传入 `engine_factory`，详见 `docs/TODO_SERVERLESS_LEVEL.md` 中「单实例 vs 多实例」。

---

### 4. 【指标】贡献3 相关指标仅来自主实例 coordinator

- **现象**：`result.aggregate(elapsed, coord_m)` 的 `coord_m` 来自 **self.coordinator.get_summary_metrics()**，即主实例；多实例时其他 slot 的 coordinator 的 contention_events、warm_pool_hits、defer 等**未汇总**。  
- **与引言关系**：若论文只汇报「整体」协调效果，单实例或「主实例代表」可接受；若需「按实例」或「多实例汇总」，当前不足。  
- **TODO**：  
  - 若保持现状：在文档中写明「ScenarioResult 中贡献3 指标（contention、warm_pool_hits 等）来自主实例 coordinator」。  
  - 若需多实例汇总：在 run() 结束时遍历 pool.get_slots() 汇总各 slot.coordinator.get_summary_metrics() 再 aggregate。

---

### 5. 【配置】min_instances / max_instances 未从 YAML 统一注入

- **现象**：ExperimentStack 内 autoscaling 写死 `min_instances=1, max_instances=2`；InstancePool 构造时也写死 `min_instances=1, max_instances=2`；`configs/experiments.yaml` 未暴露实例池上下限。  
- **与引言关系**：影响复现与参数化实验（如跑 1 实例 vs 2 实例对比）。  
- **TODO**：可选。在 YAML 的 resource_coordination 或新字段中增加 `min_instances` / `max_instances`，并由 ExperimentStack 与 ScenarioRunner 的 InstancePool 构造统一读取，避免多处硬编码。

---

### 6. 【可选】README 架构图未体现多实例与路由

- **现象**：README「实验系统架构」仅画到 InferenceEngine，未画出 InstancePool / Router。  
- **与引言关系**：多实例为可选扩展，图中不画也可；若希望与「多实例可选」文档一致，可小幅补充。  
- **TODO**：在架构图中 InferenceEngine 上一层增加「（可选）InstancePool + Router」的说明或框，并标注「默认单实例，多实例时启用」。

---

## 四、已修复（本次修改，默认多实例）

| # | 原问题 | 修改内容 |
|---|--------|----------|
| 1 | faaslora 路径 tier 解析未按实例分流 | `ExperimentStack.resolve_lora` 增加可选参数 `coordinator`，按请求使用槽位 coordinator；`ScenarioRunner._resolve_lora` 传入 `coordinator=coord`。 |
| 2 | 预加载/warmup 仅主实例 | 新增 `_warmup_engine_hot_set(engine)`，在 scale-up 且 `engine_factory` 添加新实例后对该 engine 做热点 warmup。 |
| 3 | README 未明确默认单/多实例 | README 贡献1 增加「实验默认多实例」说明；单实例改为 `max_instances=1`。 |
| 4 | 贡献3 指标仅主实例 | 多实例时汇总各槽位：`_merge_coordinator_metrics(all_slots)`，再 `result.aggregate(elapsed, merged)`。 |
| 5 | min/max_instances 未从 YAML 统一 | `configs/experiments.yaml` 增加 `min_instances`、`max_instances`；ExperimentStack 与 ScenarioRunner 的 InstancePool 从 coord_cfg 读取；默认 `max_instances=2`。 |
| 6 | README 架构图未体现多实例 | 架构图增加「多实例与路由：InstancePool + Router」。 |
| 默认多实例 | — | 启动时若 `_stack` 且 `max_instances>1`，创建 `(max_instances-1)` 个额外 ResourceCoordinator（共享 residency_manager），每槽位独立 coordinator，同 engine；单实例设 `max_instances=1`。 |

---

## 五、无问题或已明确的设计选择

| 项目 | 说明 |
|------|------|
| scale-down 只对主实例调用 trigger_scale_down | 当前语义：仅主实例做暖池保留与释放；被 remove 的实例直接摘除，不单独调用其 coordinator.trigger_scale_down。实现与「主实例暖池可观测」一致。 |
| cold_starts_after_scale_up | 按请求窗口统计 cache_tier != "gpu"，不区分主/从实例；多实例下仍表示「scale-up 后整体冷启动数」，语义合理。 |
| 默认多实例 | 已改为默认多实例（max_instances=2）；单实例设 max_instances=1。 |

---

## 六、总结（实事求是）

- **与论文引言对齐**：贡献1/2/3 的机制与实现位置已对齐；README 已明确默认多实例及单实例配置方式。  
- **单实例 vs 多实例一致性**：  
  - **已修复**：1）faaslora 下 resolve_lora 按槽位使用 coordinator；2）新实例 scale-up 后做 warmup；3）README 与架构图已写默认多实例；4）贡献3 指标汇总各槽位；5）min/max_instances 从 YAML 统一；6）架构图含 InstancePool + Router。  
  - **默认多实例**：`max_instances=2` 时启动即多槽位、每槽位独立 coordinator；单实例设 `max_instances=1`。  

以上为完整扫描与修复结果；1～6 已全部修改，默认多实例。
