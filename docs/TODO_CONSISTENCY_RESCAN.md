# 整体一致性再次扫描 TODO（对齐论文引言）

**说明**：在完成「默认多实例」及 1～6 项修改后，对全项目再次做单/多实例与论文引言一致性扫描。**不改代码**，仅给出完整 TODO；有问题则列出，没有则标明无，实事求是。对齐基准为 README 研究背景与动机、三项核心贡献。

---

## 一、论文引言要点（对齐基准）

- **观察1**：扩容实例缺乏工件命中感知，**新实例仍需冷加载 LoRA** → 贡献1 命中感知预加载  
- **贡献1**：扩容决策时将高频 LoRA **提前加载至待启动实例**；**新实例启动后无需冷加载，直接命中缓存**  
- **贡献2**：热–温–冷分层、动态迁移  
- **贡献3**：扩容预留/渐进加载；负载回落**暖池**；竞争来自真实显存压力  

实验架构：请求 → WorkloadDataset → **ScenarioRunner**（缓存、预加载、ResourceCoordinator、**InstancePool + Router**）→ InferenceEngine → 结果  

---

## 二、已一致、无需修改

| 项目 | 说明 |
|------|------|
| 默认多实例 | `max_instances=2` 从 YAML 注入；启动时若有 _stack 且 max_instances>1 则创建多槽位、每槽位独立 coordinator，与 README「实验默认多实例」一致。 |
| 扩缩与 current_instances | 两处 scale 决策均用 `current_instances = self.instance_pool.count() if self.instance_pool else 1`，与实例数一致。 |
| 请求路径按槽位 | B2 路由选 slot，`_exec_request` 使用 `slot.engine`、`slot.coordinator`；faaslora 下 `_resolve_lora` 传入 `coordinator=coord`，`ExperimentStack.resolve_lora` 按请求使用对应 coordinator。 |
| 贡献3 指标汇总 | 单周期与多周期结束时，若有 instance_pool 且 get_slots() 非空则 `_merge_coordinator_metrics(all_slots)` 再 aggregate，否则用主实例；与多实例一致。 |
| 预加载与 warmup | 主实例 preload 对 self.engine；scale-up 若 engine_factory 则对新实例调用 `_warmup_engine_hot_set(new_engine)`，与「新实例直接命中」对齐。 |
| scale-down 与 remove_instance | 单周期与多周期在 trigger_scale_down 后均检查 pool.count() > min_instances 并 remove_instance(slots[-1])，逻辑一致。 |
| min/max_instances 配置 | `configs/experiments.yaml` 有 min_instances、max_instances；ExperimentStack 与 ScenarioRunner 的 InstancePool 均从 coord_cfg 读取。 |
| README 与架构图 | README 贡献1 已写「实验默认多实例」及单实例配置方式；架构图已含 InstancePool + Router。 |
| EXPERIMENT_GUIDE / TODO_SERVERLESS_LEVEL | 已说明默认多实例、max_instances=2、单实例设 1。 |

---

## 三、仍存在或可优化的一致性问题（TODO）— 1～4 已实现

### 1. 【多周期】phase 级 warm_pool_hits 仅主实例 → **已实现**

- **实现**：多周期每 phase 开始用各槽位 `get_summary_metrics().warm_pool_hits` 求和作为 `prev_warm_hits`，phase 结束同样求和作为 `after_warm_hits`，`phase_warm_hits = after_warm_hits - prev_warm_hits` 写入 `multi_cycle_phase_results[].warm_pool_hits`。

---

### 2. 【文档】TODO_SERVERLESS_LEVEL 表格 B1 表述易歧义 → **已实现**

- **实现**：B1 状态已改为「未传 engine_factory 则 scale-up 不增加新 engine。槽位数由 max_instances 决定，默认 2 槽位」。

---

### 3. 【配置】faaslora/utils/config.py 与实验 YAML 默认值不同 → **已实现**

- **实现**：PROJECT_STRUCTURE 中 ResourceCoordinator 核心参数表增加 min/max_instances 及说明「实验实例数以 configs/experiments.yaml 为准；utils/config 的 max_instances=10 为 API/生产用」。EXPERIMENT_GUIDE 6.6 单实例 vs 多实例段落已注明实验实例数以 experiments.yaml 为准、utils/config 的 10 为 API/生产配置。

---

### 4. 【文档】TODO_CONSISTENCY_ALIGN_INTRO 第三节仍为修复前问题描述 → **已实现**

- **实现**：第三节开头已加说明「以下为修复前的描述（均已修复，见第四节）。若再次扫描后仍有遗留项，见 docs/TODO_CONSISTENCY_RESCAN.md」。

---

## 四、无问题或已接受的设计

| 项目 | 说明 |
|------|------|
| trigger_scale_down 仅主实例 | 仅对 self.coordinator 调用；被 remove 的槽位直接摘除不调用其 coordinator.trigger_scale_down，语义可接受（实例下线即丢弃状态）。 |
| ResidencyManager 共享 | 多槽位共享同一 ResidencyManager/Registry，单卡下 GPU tier 视图一致，设计合理。 |
| cold_starts_after_scale_up | 按请求窗口统计，不区分槽位，表示 scale-up 后整体冷启动数，语义合理。 |
| 非 faaslora 场景 | 无 _stack 时仅 1 槽位（add_instance 一次），与 baseline 单 engine 一致。 |

---

## 五、总结（实事求是）

- **与论文引言对齐**：贡献1/2/3 的机制、默认多实例、新实例 warmup、架构图与 README 表述已对齐；无新增冲突。  
- **单/多实例一致性**：请求路径、resolve_lora、指标汇总、扩缩与实例池、配置注入均已按多实例处理；**仅多周期 phase 级 warm_pool_hits 仍用主实例**（可选优化）。  
- **1～4 已全部实现**：phase 级 warm_pool_hits 多槽位汇总、B1 表述澄清、实验实例数说明（PROJECT_STRUCTURE + EXPERIMENT_GUIDE）、ALIGN_INTRO 第三节说明均已完成。

整体无单/多实例混用或与引言矛盾的严重问题；上述项为细节与文档层面的完善，现已全部落地。
