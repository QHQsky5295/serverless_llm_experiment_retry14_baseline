# FaaSLoRA：面向多 LoRA 大模型推理的扩缩容感知Serverless系统

FaaSLoRA 是一个面向真实云工作负载的多 LoRA Serverless 大模型推理研究原型。系统围绕三条主线展开：

- 工件命中感知的请求放置与扩缩容预加载
- GPU / 内存 / 本地存储 / 远端存储的多层工件驻留
- 工件加载、KV 缓存与批推理之间的资源协同控制

当前仓库已经具备可重复运行的实验主线，支持：

- 默认后端：`vllm`
- 实例模式：`shared` / `auto` / `dedicated`
- LoRA 规模：`100 / 300 / 500 / 1000`
- 工作负载：Azure LLM trace + ShareGPT prompt pool
- 当前推荐主模式：`auto`

---

## 项目定位

本项目不是生产级云平台，而是一个**真实可运行的单节点双 GPU Serverless LoRA 系统原型**。它的目标不是展示海量物理实例扩容，而是验证在共享 backbone 的多 LoRA 推理场景下，如何通过工件命中感知、分层驻留和资源协同，降低 TTFT、控制尾延迟，并在必要时扩展新的物理 GPU 执行实例。

论文与文档统一使用以下术语：

- **函数实例 / function instance**：请求调度与隔离的逻辑单位
- **物理 GPU 执行实例 / physical GPU execution instance**：绑定 GPU 的独立推理 runtime
- **worker 节点 / worker node**：物理宿主机 / 计算节点
- **共享执行槽位 / shared execution slot**：仅在实现细节中使用，用于描述 `shared` 模式下单个物理 runtime 内部的可路由槽位

---

## 研究背景与动机

在多 LoRA Serverless 推理中，系统面临三类核心问题：

| 观察 | 问题 | FaaSLoRA 对应机制 |
|---|---|---|
| 观察 1 | 请求被放置到现有函数实例时可能未命中所需 LoRA 工件；扩容到新实例时同样会发生工件冷加载 | 命中感知的请求放置与扩缩容预加载 |
| 观察 2 | 单层缓存策略难以兼顾命中率与显存利用率，工件迁移路径过长 | GPU / HOST / NVME / REMOTE 多层驻留 |
| 观察 3 | LoRA 工件加载、KV 缓存与批推理共享 GPU 显存和带宽，容易产生资源争用 | 资源协同控制与受控 warmup |

FaaSLoRA 的研究重点不是“为每个请求都创建新的物理 GPU 实例”，而是：

1. 优先把请求放到更可能已命中目标工件的函数实例。
2. 当现有实例无法满足负载或隔离需求、且有可用 GPU 资源时，再扩展新的物理 GPU 执行实例。
3. 在共享执行与扩容并存的条件下，控制工件加载对函数执行的副作用。

---

## 三项核心机制

### 1. 命中感知的请求放置与扩缩容预加载

- **请求放置**：请求优先路由到对目标 LoRA 具有更高缓存亲和性的函数实例。
- **扩容预加载**：当系统决定扩展新的物理 GPU 执行实例时，会依据近期访问统计、工件热度与层级位置，为新实例生成实例级热点工件 warmup 计划。
- **目标**：减少“请求已放置但工件未命中”和“扩容后仍需冷加载”的代价。

对应实现：

- `faaslora/experiment/instance_pool.py`
- `faaslora/preloading/preloading_planner.py`
- `faaslora/preloading/preloading_manager.py`
- `faaslora/experiment/experiment_stack.py`

### 2. 多层工件驻留控制

系统维护四层工件路径：

| 层级 | 作用 | 当前语义 |
|---|---|---|
| `GPU` | 热层 | 高频 LoRA 工件尽量常驻 GPU，优先保障 TTFT |
| `HOST` | 温层 | 缓存近期可能再次命中的工件，缩短再次装载路径 |
| `NVME` | 本地存储层 | 保留更大规模候选工件集合 |
| `REMOTE` | 远端源仓库 | 工件的最终来源 |

系统根据显存预算、访问热度与命中收益，在上述层级之间做迁移和保留，以缩短 LoRA 工件从远端存储到 GPU 的传输路径。

对应实现：

- `faaslora/memory/residency_manager.py`
- `faaslora/storage/local_cache.py`
- `faaslora/storage/remote_client.py`

### 3. 资源协同控制

系统协调以下三类竞争：

- LoRA 工件加载
- KV 缓存
- 批量推理执行

当前实现并不是简单“谁先抢到显存谁执行”，而是通过：

- warmup 前资源检查
- admission / defer
- 加载队列控制
- 实例级资源提示

来尽量减少工件加载对在线请求时延的副作用。

对应实现：

- `faaslora/scheduling/resource_coordinator.py`

> 注：一个“争用感知的有效容量准入机制”实验版本已做过评估，但当前未作为主线默认机制启用。

---

## 实验系统架构

```text
                 +--------------------------------------+
                 | Azure Trace + ShareGPT Prompt Pool   |
                 | 到达时间 / token 分布 / prompt 文本   |
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
      | 扩缩容控制        |                  | 路由与实例池        |
      | AutoScaler        |                  | InstancePool / Router|
      | arrival/backlog   |                  | shared/auto/dedicated|
      | busy/latency      |                  | adapter_affinity     |
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
            | vLLM / fallback |             | vLLM runtime   |
            +--------+--------+             +-------+--------+
                     \                              /
                      \                            /
                       +-----------+--------------+
                                   |
                                   v
                         GPU / HOST / NVME / REMOTE
```

---

## 核心模块与规则

### 模块总览

| 模块 | 作用 | 当前规则 / 实现要点 | 关键文件 |
|---|---|---|---|
| 工作负载生成 | 读取 Azure trace 和 ShareGPT，构造 replay workload | Azure trace 提供到达时间与 token 规模，ShareGPT 提供 prompt 文本池 | `faaslora/datasets/` |
| 路由与实例池 | 维护 `shared / auto / dedicated` 三种实例模式并路由请求 | 默认 `adapter_affinity`；先看缓存亲和性，再看 active、queue、GPU util | `faaslora/experiment/instance_pool.py` |
| 扩缩容 | 决定何时增加或回收实例 | 结合 `arrival_rps / backlog / busy_ratio / latency` 动态触发 | `faaslora/coordination/autoscaler.py` |
| 预加载 | 生成并执行初始预加载和 scale-up warmup 计划 | 启动时三阶段预加载；扩容时为新实例做 instance-scoped warmup | `faaslora/preloading/`, `faaslora/experiment/experiment_stack.py` |
| 分层驻留 | 管理工件在 GPU / HOST / NVME / REMOTE 之间的状态 | 根据预算、热度、层级位置决定保留、迁移和驱逐 | `faaslora/memory/residency_manager.py` |
| 资源协同 | 控制工件加载、KV 缓存与推理执行之间的竞争 | admission / defer / warmup 检查 / contention 统计 | `faaslora/scheduling/resource_coordinator.py` |
| 推理后端 | 执行真实模型推理 | 默认 `vllm`，保留 `transformers` 回退路径 | `faaslora/serving/`, `scripts/run_all_experiments.py` |
| 实验入口与结果 | 运行矩阵、显示 live 面板、落盘结果 | preset、结果独立命名、顶层 schema 扁平化 | `scripts/run_all_experiments.py`, `scripts/run_validation_bundle.sh` |

---

## 实例模式

### `shared`

- 单个物理 GPU runtime 上承载多个共享执行槽位
- scale-up 的语义是**逻辑扩容**
- 增加的是 shared slot，不新增物理 GPU runtime
- 适合作为 shared-serving 基线

### `auto`

- 优先扩展新的物理 GPU 执行实例
- 当物理扩容受限或失败时，再回退到共享执行路径
- 当前最适合作为论文主系统模式

### `dedicated`

- 扩容时只允许创建新的独立物理 GPU runtime
- 不回退到 shared slot
- 在当前双 3090 环境下，物理上限就是 2 个 runtime
- 适合作为物理扩容真实性与上界对照

---

## 扩缩容与调度规则

### 扩缩容信号

当前扩缩容主要使用：

- `arrival_rps`
- `backlog`
- `instance_busy_ratio`
- `response_time`
- GPU 运行时负载快照

### 事件类型

当前结果中已经区分：

- `logical_scale_up`
- `physical_scale_up`
- `logical_scale_down`
- `physical_scale_down`

因此：

- `shared` 的扩容是逻辑扩容
- `auto / dedicated` 的扩容是物理扩容

### 路由规则

当前默认路由策略是 `adapter_affinity`：

1. 优先把请求放到对目标 LoRA 工件具有更高缓存亲和性的实例。
2. 当亲和性相近时，再结合：
   - `active_requests`
   - `load_queue_depth`
   - `gpu_utilization_pct`
   - `last_selected_at`
   做 tie-break。

这是一个**cache-affinity-aware routing heuristic**，不是全局最优放置器。

---

## 缓存与存储

### 四层工件路径

| 层级 | 当前作用 |
|---|---|
| `REMOTE` | 远端源仓库，提供工件的最终来源 |
| `NVME` | 本地存储层，保留较大规模候选工件 |
| `HOST` | 内存层，缓存近期可能再次命中的工件 |
| `GPU` | 热层，优先保留关键高频 LoRA 工件 |

### 当前支持的预加载阶段

系统启动时支持三阶段预加载：

1. `REMOTE -> NVME`
2. `NVME -> HOST`
3. `HOST/NVME -> GPU warmup`

扩容到新物理实例时，系统还支持：

- `instance-scoped GPU warmup plan`

即按新实例视角生成热点工件 warmup 计划，而不是简单复用全局静态热集。

---

## 大模型推理侧

### 默认后端：`vllm`

当前主线实验全部基于 `vllm`，原因是：

- 更适合多 LoRA serving
- 更适合 `shared / auto / dedicated` 三种实例模式
- 更接近论文主线的真实推理路径

### 回退后端：`transformers`

`transformers` 仍保留，但目前定位为：

- 兼容性验证
- 小规模回退路径
- 非主线后端

因此：
- 当前论文主线不要再以 `transformers` 为主结果来源

---

## 数据与工作负载

| 数据来源 | 当前使用方式 |
|---|---|
| Azure LLM trace | 提供真实请求到达时间与 token 分布骨架 |
| ShareGPT | 提供 prompt 文本池，当前作为抽样 prompt pool 使用 |

注意：

- 当前 ShareGPT 不是 full conversation replay，而是 prompt pool
- 小矩阵使用 representative trace replay
- 当前主线下一步是：`auto1000 + Azure full trace 28185`

---

## 当前实验主线

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

- 真正的 `auto1000 + Azure full trace 28185`
- CLI / packaging 断裂修复
- 稳定环境下测试闭环
- 跨模型家族与跨数据集扩展

---

## 快速入口

### 主脚本入口

```bash
conda activate LLM_vllm0102
cd /home/qhq/serverless_llm_experiment
python scripts/run_all_experiments.py --config configs/experiments.yaml --preset auto1000
```

### 协作 / 矩阵验证入口

```bash
cd /home/qhq/serverless_llm_experiment
FAASLORA_PRESET=auto300 bash scripts/run_validation_bundle.sh custom
```

当前已内置 preset：

- `shared100`, `auto100`, `dedicated100`
- `shared300`, `auto300`, `dedicated300`
- `shared500`, `auto500`
- `shared1000`, `auto1000`

---

## 文档导航

| 文档 | 作用 |
|---|---|
| [`docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md`](docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md) | 当前实现的权威技术说明 |
| [`docs/PROJECT_PROGRESS.md`](docs/PROJECT_PROGRESS.md) | 当前进度、矩阵状态、TODO |
| [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) | 环境与依赖说明 |
| [`EXPERIMENT_GUIDE.md`](EXPERIMENT_GUIDE.md) | 实验执行操作手册 |
| [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) | 仓库文件结构说明 |

---

## 仓库同步约束

同步到 GitHub 时遵循以下规则：

- 代码和文档必须一起更新
- README、技术说明、实验指南、附属文档应与当前实现一致
- 模型目录只保留占位，不上传权重内容
- 项目标题固定为：
  - **FaaSLoRA：面向多 LoRA 大模型推理的扩缩容感知Serverless系统**

---

## 当前边界

- 当前系统是单节点双 GPU 原型，不是完整多节点云平台
- `shared` 模式不是强隔离函数进程模型，而是共享 runtime + shared execution slot 的实现方式
- ShareGPT 当前作为 prompt pool，而不是 full conversation replay
- 当前矩阵使用 representative trace replay；主模式仍需 full trace 结果补齐
