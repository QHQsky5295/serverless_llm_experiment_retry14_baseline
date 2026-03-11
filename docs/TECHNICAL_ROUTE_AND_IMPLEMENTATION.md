# FaaSLoRA 技术路线与实现说明

本文档是当前仓库的**权威技术说明文档**。内容对齐当前实现，而不是历史设计稿。若 README、实验指南与本文冲突，以本文为准。

## 1. 系统定位

FaaSLoRA 是一个面向多 LoRA 大模型推理的单节点 Serverless 研究原型。系统运行在单台双 GPU 服务器上，通过共享 backbone、工件感知请求放置、分层驻留和资源协同控制，研究以下问题：

1. 请求被放置到现有函数实例时，目标实例是否已经命中所需 LoRA 工件。
2. 系统扩展新的物理 GPU 执行实例时，新实例是否能够带着热点工件快速进入服务。
3. 在 LoRA 工件加载、KV 缓存和批推理共享 GPU 资源的条件下，如何控制争用带来的 TTFT 和尾延迟代价。

当前系统不是生产级云平台，而是一个**真实可运行的单节点双 GPU Serverless LoRA 原型**。它已经可以稳定运行 `shared / auto / dedicated` 三种模式，并支持 `100 / 300 / 500 / 1000` adapters 的实验矩阵。

## 2. 术语约定

论文与文档统一使用以下术语：

- **函数实例 / function instance**：请求调度与隔离的逻辑单位。
- **物理 GPU 执行实例 / physical GPU execution instance**：绑定 GPU 的独立推理 runtime。
- **worker node**：物理宿主机 / 计算节点。
- **shared execution slot**：仅在实现细节中使用，表示共享执行模式下同一物理 runtime 内的可路由槽位。

注意：`shared execution slot` 不是论文主术语，只用于解释当前原型实现。

## 3. 当前系统架构

```text
                 +--------------------------------------+
                 | Azure Trace + ShareGPT Prompt Pool   |
                 | arrivals / tokens / prompt sampling  |
                 +------------------+-------------------+
                                    |
                                    v
                      +-------------+-------------+
                      | ScenarioRunner            |
                      | scripts/run_all_experiments.py |
                      +-------------+-------------+
                                    |
                +-------------------+-------------------+
                |                                       |
                v                                       v
      +---------+---------+                  +----------+----------+
      | AutoScaler        |                  | Router / InstancePool|
      | arrival/backlog   |                  | shared / auto / dedicated |
      | busy/latency      |                  | adapter_affinity routing   |
      +---------+---------+                  +----------+----------+
                |                                       |
                +-------------------+-------------------+
                                    |
                                    v
                         +----------+-----------+
                         | ExperimentStack      |
                         | - PreloadingPlanner  |
                         | - PreloadingManager  |
                         | - ResidencyManager   |
                         | - ResourceCoordinator|
                         +----------+-----------+
                                    |
                     +--------------+---------------+
                     |                              |
                     v                              v
            +--------+--------+             +-------+--------+
            | GPU0 runtime    |             | GPU1 runtime   |
            | vLLM/Transformers|            | vLLM runtime   |
            +--------+--------+             +-------+--------+
                     \                              /
                      \                            /
                       +-----------+--------------+
                                   |
                                   v
                         GPU / HOST / NVME / REMOTE
```

## 4. 模块划分

### 4.1 工作负载生成

职责：
- 读取 Azure LLM trace
- 加载 ShareGPT prompt pool
- 构造带真实到达间隔和 token 分布的 replay workload

关键实现：
- `faaslora/datasets/dataset_loader.py`
- `faaslora/datasets/workload_generator.py`

当前方法学：
- Azure trace 提供请求到达时间、输入输出 token 统计与规模骨架
- ShareGPT 提供 prompt 文本池
- 小矩阵使用 representative trace replay
- 主结论实验计划使用 full Azure trace `28185`

## 4.2 路由与实例池

职责：
- 管理 `shared / auto / dedicated` 三种实例模式
- 在请求到达时选择目标实例
- 维护共享槽位与物理 runtime 的关系

关键实现：
- `faaslora/experiment/instance_pool.py`
- `scripts/run_all_experiments.py`

### 路由规则

当前默认路由策略是：`adapter_affinity`

核心思想：
- 优先把请求发到对目标 LoRA 工件具有更高缓存亲和性的实例
- 在缓存亲和性相近时，再考虑：
  - `active_requests`
  - `load_queue_depth`
  - `gpu_utilization_pct`
  - `last_selected_at`

这是一个**cache-affinity-aware routing heuristic**，不是全局最优放置器。

## 5. 实例模式

### 5.1 `shared`

语义：
- 单个物理 GPU runtime 上维护多个 `shared execution slot`
- 请求被路由到不同槽位，但底层共享同一个 engine、同一 GPU、同一 backbone

扩容行为：
- 触发的是 `logical_scale_up`
- 增加的是共享执行槽位，不新增物理 GPU runtime

适用角色：
- 共享执行基线
- 对标 shared-serving 系统的对照组

### 5.2 `auto`

语义：
- 系统优先扩展新的物理 GPU 执行实例
- 若物理扩容受限，再回退到共享执行路径

扩容行为：
- 触发的是 `physical_scale_up`
- 当前最适合作为主系统模式

适用角色：
- 论文主系统模式
- 兼顾真实扩容与部署可行性

### 5.3 `dedicated`

语义：
- 扩容时只允许创建新的独立物理 GPU runtime
- 不回退到 shared slot

扩容行为：
- 纯 `physical_scale_up`
- 在当前双 3090 环境下，物理上限就是 2 个 runtime

适用角色：
- 物理扩容真实性对照
- 性能上界 / 隔离上界对照

## 6. 扩缩容机制

### 6.1 扩缩容输入信号

当前扩缩决策使用以下主信号：

- `arrival_rps`
- `backlog`
- `instance_busy_ratio`
- `response_time`
- GPU 利用率与运行时负载快照

关键实现：
- `faaslora/coordination/autoscaler.py`
- `scripts/run_all_experiments.py`

### 6.2 阈值与动态规则

默认配置来自 `configs/experiments.yaml`：

- `dynamic_scaling: true`
- `scale_up_threshold_rps: 3.0`（固定模式下使用）
- `baseline_rps_ewma_beta: 0.25`
- `scale_up_alpha: 0.3`
- `scale_up_t_min: 1.0`
- `scale_down_beta: 0.4`
- `scale_down_duration_s: 45`
- `scale_decision_interval: 25`（preset 常覆盖为 `10`）

动态规则的含义：
- scale-up 不只依赖固定 RPS，而是结合 EWMA 基线与相对阈值
- scale-down 要求持续低负载，避免抖动

### 6.3 扩缩事件语义

当前结果中已经区分：

- `logical_scale_up`
- `physical_scale_up`
- `logical_scale_down`
- `physical_scale_down`

这使得 `shared` 的逻辑扩容和 `auto/dedicated` 的物理扩容可以在结果层显式区分。

## 7. 预加载与分层驻留

### 7.1 四层工件路径

系统统一维护四层工件状态：

- `GPU`
- `HOST`
- `NVME`
- `REMOTE`

关键实现：
- `faaslora/memory/residency_manager.py`
- `faaslora/experiment/experiment_stack.py`
- `faaslora/preloading/`

### 7.2 启动时预加载

当前完整栈支持三阶段预加载：

1. `REMOTE -> NVME`
2. `NVME -> HOST`
3. `HOST/NVME -> GPU warmup`

目的：
- 在实例真正处理请求之前，把最有价值的工件沿层次逐步前推
- 避免直接从 remote 冷装载到 GPU

### 7.3 scale-up 新实例 warmup

在 `auto/dedicated` 物理扩容时，系统会为新实例生成：
- `instance-scoped GPU warmup plan`

它基于：
- 近期访问统计
- 工件热度
- 当前层级位置
- 新实例本地可用性

这一步是当前系统与纯 shared-serving 基线拉开差距的关键机制之一。

## 8. 资源协同

职责：
- 协调 LoRA 工件加载、KV 缓存与批推理之间的资源竞争
- 在不直接破坏当前工作集的前提下完成 GPU admission / warmup / defer

关键实现：
- `faaslora/scheduling/resource_coordinator.py`

当前功能：
- admission / defer
- warmup 前资源检查
- contention / defer 统计
- 与 ResidencyManager 对接的 GPU 准入与层间状态更新

注意：
- 当前主线默认使用经过验证的稳定 admission 路径
- 一个“争用感知的有效容量准入机制”实验版已做过评估，但当前未作为主线默认机制启用

## 9. 推理后端

### 9.1 默认后端：`vllm`

当前默认后端是：`vllm`

原因：
- 更适合多 LoRA serving
- 支持更合理的共享执行和物理 runtime 扩展
- 是当前主线实验的标准路径

### 9.2 回退后端：`transformers`

`transformers` 保留为可切换回退后端，但不再作为主线默认路径。

原因：
- 当前项目中它更适合小规模、单并发、兼容性验证
- 不适合作为最终多 LoRA serverless 主实验后端

## 10. 当前实验状态

### 已完成的小矩阵（Qwen2.5-3B + vLLM）

已完成：
- `shared100`
- `auto100`
- `dedicated100`
- `shared300`
- `auto300`
- `dedicated300`
- `shared500`
- `auto500`
- `shared1000`
- `auto1000`

当前结论：
- `shared`：共享执行基线
- `dedicated`：小规模物理独立上界 / 对照
- `auto`：当前最合理主模式

### 当前仍未完成

- `auto1000 + full Azure trace 28185`
- CLI / packaging 断裂修复
- 稳定环境下测试闭环
- 文档口径全面对齐
- 跨模型家族与跨数据集扩展

## 11. 当前主线 TODO

### 主线 A：Qwen-3B 证据链
1. 以 `auto` 为主模式完成真正的 `28185` 全量主实验。
2. 汇总 `shared/auto/dedicated` 小矩阵和 full-trace 主实验的主表。

### 主线 B：工程硬伤修复
3. 修 CLI / packaging 断裂。
4. 补稳定环境下可执行的基础测试。
5. 清理 README / GUIDE / docs 与当前实现的漂移。

### 主线 C：模型扩展
6. Qwen 家族：`3B / 7B`
7. Meta-Llama 家族：`3B / 8B`
8. Gemma 家族：`4B / 12B`

### 主线 D：数据集扩展
9. `HuggingFaceH4/ultrachat_200k`
10. `lmsys/lmsys-chat-1m`

## 12. 已知边界

- 当前系统是单节点双 GPU 原型，不是完整多节点云平台
- ShareGPT 当前作为 prompt pool，而不是 full conversation replay
- 矩阵实验使用 representative trace replay；主模式需要 full trace 结论补齐
- `shared` 模式不是强隔离函数进程模型，而是共享 runtime + shared execution slot 的实现方式
