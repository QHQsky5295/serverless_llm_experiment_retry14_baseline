# 整体一致性最终扫描 TODO（对齐论文引言）

**说明**：对全项目进行第三轮单/多实例与论文引言一致性扫描。**不改代码**，仅给出完整 TODO；有问题则列出，没有则标明无，实事求是。对齐基准为 README 研究背景与动机、三项核心贡献。

---

## 一、论文引言要点（对齐基准）

- **观察1**：扩容实例缺乏工件命中感知，**新实例仍需冷加载 LoRA** → 贡献1 命中感知预加载  
- **贡献1**：扩容决策时将高频 LoRA **提前加载至待启动实例**；**新实例启动后无需冷加载，直接命中缓存**  
- **贡献2**：热–温–冷分层、动态迁移  
- **贡献3**：扩容预留/渐进加载；负载回落**暖池**；竞争来自真实显存压力  

实验架构：请求 → WorkloadDataset → **ScenarioRunner**（缓存、预加载、ResourceCoordinator、**InstancePool + Router**）→ InferenceEngine → 结果  

---

## 二、已一致、无需修改（代码 + 文档）

| 项目 | 说明 |
|------|------|
| 默认多实例 | YAML `max_instances=2`；启动时 _stack 且 max_instances>1 则创建多槽位、每槽位独立 coordinator；与 README 一致。 |
| 请求路径 | B2 选 slot，_exec_request 使用 slot.engine / slot.coordinator；notify_batch_start/end 使用 _coord；_resolve_lora 传入 coordinator=coord，ExperimentStack.resolve_lora 按请求使用对应 coordinator。 |
| 扩缩与实例数 | current_instances = pool.count()；scale-down 后按 min_instances 做 remove_instance。 |
| 贡献3 指标 | 单周期/多周期结束对 get_slots() 做 _merge_coordinator_metrics 再 aggregate。 |
| 多周期 phase warm_pool_hits | 每 phase 开始/结束对各槽位 warm_pool_hits 求和再算 delta，已按多槽位汇总。 |
| 预加载与 warmup | 主实例 preload；engine_factory 扩容后 _warmup_engine_hot_set(new_engine)。 |
| 配置 | min/max_instances 从 experiments.yaml → coord_cfg → ExperimentStack 与 InstancePool；PROJECT_STRUCTURE / EXPERIMENT_GUIDE 已注明实验以 YAML 为准、utils/config 的 10 为 API 用。 |
| README / 架构图 | 贡献1 已写默认多实例；架构图含 InstancePool + Router。 |
| 非多槽位路径 | 无 _stack 时仅 1 槽位；fallback 同步推理路径用 self.engine/self.coordinator，无 router，语义正确。_warmup_gpu 仅在不走 full stack 时使用，此时无多槽位。 |

---

## 三、仍存在的小问题（仅文档，可选）

### 1. 【文档】TODO_SERVERLESS_LEVEL「已完成」列表 B1 措辞 → **已修复**

- **修复**：第九节「已完成」中 B1 已改为「否则 scale-up 不增加新 engine。槽位数由 `max_instances` 决定，默认 2 槽位」，与表格 B1 一致。

---

## 四、无问题或已接受的设计

| 项目 | 说明 |
|------|------|
| trigger_scale_down 仅主实例 | 仅对 self.coordinator 调用；被 remove 的槽位直接摘除，不调用其 coordinator.trigger_scale_down，语义可接受。 |
| ResidencyManager 共享 | 多槽位共享同一 ResidencyManager/Registry，单卡下 GPU 视图一致。 |
| cold_starts_after_scale_up | 按请求窗口统计，不区分槽位，表示 scale-up 后整体冷启动数。 |
| C3「单实例内」 | ResourceCoordinator 负责「单实例内」显存/暖池，与多实例（每槽位一个 coordinator）语义一致。 |

---

## 五、总结（实事求是）

- **与论文引言对齐**：贡献1/2/3、默认多实例、新实例 warmup、架构图与 README 已对齐；无新增冲突。  
- **单/多实例一致性**：请求路径、resolve_lora、指标汇总、phase 级 warm_pool_hits、扩缩与实例池、配置注入、预加载与 warmup 均已按多实例处理；**未发现代码层面单/多实例混用或错误**。  
- **待办**：无；上述 1 条已修复。  

**结论**：整体一致性良好；可选文档措辞已与表格 B1 统一。
