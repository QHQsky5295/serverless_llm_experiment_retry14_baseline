# 整体一致性扫描 TODO（对齐论文引言）— 最新

**说明**：对全项目进行单/多实例与论文引言一致性完整扫描。**不改代码**，仅给出完整 TODO；有问题则列出，没有则标明无，实事求是。对齐基准为 README 研究背景与动机、三项核心贡献。

---

## 一、论文引言要点（对齐基准）

- **观察1**：扩容实例缺乏工件命中感知，**新实例仍需冷加载 LoRA** → 贡献1 命中感知预加载  
- **贡献1**：扩容决策时将高频 LoRA **提前加载至待启动实例**；**新实例启动后无需冷加载，直接命中缓存**  
- **贡献2**：热–温–冷分层、动态迁移  
- **贡献3**：扩容预留/渐进加载；负载回落**暖池**；竞争来自真实显存压力  

实验架构：请求 → WorkloadDataset → **ScenarioRunner**（缓存、预加载、ResourceCoordinator、**InstancePool + Router**）→ InferenceEngine → 结果  

---

## 二、已一致、无需修改

### 代码

| 项目 | 说明 |
|------|------|
| 默认多实例 | `max_instances` 从 coord_cfg 读取（默认 2）；有 _stack 且 max_instances>1 时启动即创建多槽位、每槽位独立 ResourceCoordinator。 |
| 请求路径 | B2 选 slot；_exec_request 使用 slot.engine / slot.coordinator；notify_batch_start/end 使用 _coord；_resolve_lora 传入 coordinator=_coord。 |
| resolve_lora | ExperimentStack.resolve_lora(..., coordinator=...) 按请求使用传入的 coord（槽位 coordinator），与多实例一致。 |
| 扩缩 | current_instances = instance_pool.count()；scale-down 后按 min_instances 做 remove_instance(slots[-1])。 |
| 贡献3 指标 | 单周期与多周期结束时，若有 instance_pool 且 get_slots() 非空则 _merge_coordinator_metrics(all_slots) 再 aggregate。 |
| 多周期 phase warm_pool_hits | 每 phase 开始 prev_warm_hits = sum(各槽位 warm_pool_hits)；phase 结束 after_warm_hits = sum(各槽位 warm_pool_hits)；phase_warm_hits = after - prev，已按多槽位汇总。 |
| 预加载与 warmup | 主实例 preload；engine_factory 扩容后 _warmup_engine_hot_set(new_engine)。 |
| 非多槽位路径 | 无 _stack 时仅 1 槽位；fallback 同步推理用 self.engine/self.coordinator（无 router），语义正确。_warmup_gpu 仅在不走 full stack 时使用。 |

### 文档

| 项目 | 说明 |
|------|------|
| README | 贡献1 已写「实验默认多实例」、max_instances=2、单实例设 1；架构图含 InstancePool + Router。 |
| TODO_SERVERLESS_LEVEL | 表格 B1 与第九节「已完成」B1 表述一致（槽位数由 max_instances 决定，默认 2 槽位）。 |
| EXPERIMENT_GUIDE | 6.6 已说明默认多实例、实验实例数以 YAML 为准、utils/config 的 10 为 API 用。 |
| PROJECT_STRUCTURE | ResourceCoordinator 参数表含 min/max_instances；已注明实验以 YAML 为准。 |

---

## 三、待办（问题列表）

**无。** 经扫描，代码与文档在单/多实例及与论文引言对齐方面**未发现不一致或遗漏**。

---

## 四、可选（文档自洽，非必须）

| 项目 | 说明 |
|------|------|
| TODO_CONSISTENCY_RESCAN 总结句 | 该文档「总结」中曾写「仅多周期 phase 级 warm_pool_hits 仍用主实例（可选优化）」；当前 phase 已按多槽位汇总，该句已过时。可改为「phase 级 warm_pool_hits 已按多槽位汇总」或删除该句，使文档自洽。不影响实现与论文对齐。 |

---

## 五、总结（实事求是）

- **与论文引言对齐**：贡献1/2/3、默认多实例、新实例 warmup、架构图与 README 已对齐。  
- **单/多实例一致性**：请求路径、resolve_lora、指标汇总、phase 级 warm_pool_hits、扩缩与实例池、配置、预加载与 warmup 均已按多实例处理；**无代码或必须文档待办**。  
- **待办**：**无**。可选 1 条：RESCAN 文档中一句过时总结可改可删，仅文档自洽。

**结论**：整体一致；有问题就有、没有就没有——当前**没有**必须修复项。
