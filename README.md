# FaaSLoRA：面向多 LoRA 大模型推理的扩缩容感知Serverless系统

FaaSLoRA 是一个面向多 LoRA 大模型推理的单节点 Serverless 研究原型。系统聚焦三件事：

- 工件感知的请求放置与扩缩容预加载
- GPU / Host / NVMe / Remote 四层工件驻留
- 工件加载、KV 缓存与批推理之间的资源协同控制

当前仓库已经具备可重复运行的完整实验主线：

- 默认后端：`vllm`
- 实例模式：`shared` / `auto` / `dedicated`
- LoRA 规模：`100 / 300 / 500 / 1000`
- 真实工作负载：Azure LLM trace + ShareGPT prompt pool
- 当前推荐主模式：`auto`

## 项目定位

这个项目不是生产级云平台，而是一个**真实可运行的单节点双 GPU Serverless LoRA 系统原型**。它的目标不是展示海量物理实例扩容，而是验证在共享 backbone 的多 LoRA 推理场景下，如何通过工件命中感知、分层驻留和资源协同，降低 TTFT、控制尾延迟，并在必要时扩展新的物理 GPU 执行实例。

论文主术语使用以下分层：

- `函数实例 / function instance`：请求调度与隔离的逻辑单位
- `物理 GPU 执行实例 / physical GPU execution instance`：绑定 GPU 的独立推理 runtime
- `worker node`：物理宿主机 / 计算节点语义
- `shared execution slot`：仅在实现细节中用于描述 shared 模式的共享执行槽位

## 系统概览

```text
                    +----------------------------+
                    | Azure Trace + ShareGPT     |
                    | arrivals / tokens / prompts|
                    +-------------+--------------+
                                  |
                                  v
+----------------------+   +------+------------------------+
| AutoScaler           |   | Router / InstancePool        |
| - arrival_rps        |   | - shared / auto / dedicated  |
| - backlog            |   | - adapter_affinity routing   |
| - busy_ratio         |   | - logical vs physical scale  |
+----------+-----------+   +------+------------------------+
           |                          |
           |                          v
           |              +-----------+--------------------+
           |              | ExperimentStack               |
           |              | - PreloadingPlanner/Manager   |
           |              | - ResidencyManager            |
           |              | - ResourceCoordinator         |
           |              +-----------+-------------------+
           |                          |
           v                          v
  +--------+------------------------------+
  |       vLLM / Transformers Inference    |
  |  GPU0 runtime          GPU1 runtime    |
  +----------------+-----------------------+
                   |
                   v
         GPU / HOST / NVME / REMOTE tiers
```

## 核心模块

| 模块 | 作用 | 关键文件 |
|---|---|---|
| 工作负载生成 | 读取 Azure trace、ShareGPT prompt pool，构造 replay workload | `faaslora/datasets/` |
| 路由与实例池 | 维护 `shared/auto/dedicated` 三种实例模式与 adapter-affinity 路由 | `faaslora/experiment/instance_pool.py`, `scripts/run_all_experiments.py` |
| 扩缩容 | 根据 `arrival_rps / backlog / busy_ratio / latency` 做 logical 或 physical scale-up/down | `faaslora/coordination/autoscaler.py`, `scripts/run_all_experiments.py` |
| 预加载 | 规划并执行初始预加载与 scale-up warmup | `faaslora/preloading/`, `faaslora/experiment/experiment_stack.py` |
| 分层驻留 | 管理 GPU / HOST / NVME / REMOTE 四层容量与迁移 | `faaslora/memory/residency_manager.py` |
| 资源协同 | 控制工件加载、KV 缓存与推理执行的竞争 | `faaslora/scheduling/resource_coordinator.py` |
| 推理后端 | 默认 `vllm`，保留 `transformers` 回退路径 | `faaslora/serving/`, `scripts/run_all_experiments.py` |
| 实验入口 | 矩阵 preset、结果汇总、live 面板、结果落盘 | `scripts/run_all_experiments.py`, `scripts/run_validation_bundle.sh` |

## 实例模式

### `shared`

- 单个物理 GPU runtime 上承载多个共享执行槽位
- scale-up 是**逻辑扩容**：增加 shared slot
- 适合作为 shared-serving 基线

### `auto`

- 优先扩展新的物理 GPU 执行实例
- 物理扩容受限或失败时，回退到共享执行路径
- 当前是最适合作为论文主系统模式的策略

### `dedicated`

- 只允许扩展新的独立物理 GPU 执行实例
- 用于验证物理扩容真实性与性能上界
- 在当前双 3090 环境下，物理上限就是 2 个 runtime

## 扩缩容与调度规则

系统当前采用动态扩缩容阈值与工件亲和路由：

- 主要扩缩信号：`arrival_rps`、`backlog`、`instance_busy_ratio`、`response_time`
- `shared` 模式扩的是 `logical_scale_up`
- `auto/dedicated` 模式扩的是 `physical_scale_up`
- 路由策略默认是 `adapter_affinity`
  - 优先选择对目标 LoRA 具有更高缓存亲和性的实例
  - 结合 `active_requests`、`load_queue_depth`、`gpu_utilization` 做 tie-break

## 缓存 / 驻留层次

FaaSLoRA 使用四层工件路径：

- `GPU`：热层，关键高频 LoRA 常驻
- `HOST`：主存层，缓存近期可能再次命中的工件
- `NVME`：本地存储层，保留更大规模工件集合
- `REMOTE`：远端源仓库

系统支持：

- 启动时初始三阶段预加载
  - `REMOTE -> NVME`
  - `NVME -> HOST`
  - `HOST/NVME -> GPU warmup`
- scale-up 时的新实例级 warmup plan
- 根据热度、预算和实例局部性做层间迁移

## 当前实验方法

### 小矩阵

当前主线已跑通以下矩阵（Qwen2.5-3B + vLLM）：

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

当前主模式判断：`auto`

### 全量主实验

下一步主线是：

- `auto1000 + Azure full trace 28185`

注意：

- ShareGPT 当前作为 prompt pool 使用，而不是逐条 full replay
- Azure trace 作为真实到达与 token 统计骨架

## 快速入口

### 主脚本

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

当前矩阵 preset 已内置：

- `shared100`, `auto100`, `dedicated100`
- `shared300`, `auto300`, `dedicated300`
- `shared500`, `auto500`
- `shared1000`, `auto1000`

## 关键文档

- 技术实现与模块说明：[`docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md`](docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md)
- 当前项目进度：[`docs/PROJECT_PROGRESS.md`](docs/PROJECT_PROGRESS.md)
- 环境说明：[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)
- 实验指南：[`EXPERIMENT_GUIDE.md`](EXPERIMENT_GUIDE.md)
- 项目结构：[`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md)

## 仓库同步约束

本仓库同步到 GitHub 时遵循以下规则：

- 代码和文档必须一起更新
- README、技术说明、实验指南、附属文档应与当前实现一致
- 模型目录只保留占位，不上传权重内容
- 标题固定为：
  - **FaaSLoRA：面向多 LoRA 大模型推理的扩缩容感知Serverless系统**

## 当前边界

当前系统已经是一个可运行、可重复实验的单节点研究原型，但还不是完整云平台。当前仍在推进的工作包括：

- `auto1000 + 28185` 全量主实验
- CLI / packaging 断裂修复
- 稳定环境下的测试闭环
- 更多模型家族与对话数据集扩展
