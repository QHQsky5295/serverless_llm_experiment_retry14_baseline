# FaaSLoRA 项目结构详解

> 说明：本文用于说明仓库文件布局；当前系统语义、实例模式、扩缩容规则、缓存层次和实验进度，请以 [README.md](README.md)、[docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md](docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md) 和 [docs/PROJECT_PROGRESS.md](docs/PROJECT_PROGRESS.md) 为准。

本文档详细说明项目每个目录和文件的作用、实现内容及相互关系，适合初次接触本项目的研究人员参考。

---

## 一级目录总览

```
serverless_llm_experiment/
├── faaslora/               ← 核心系统模块（Python 包）
├── scripts/                ← 可执行脚本（实验运行、数据下载等）
├── configs/                ← 实验配置文件
├── data/                   ← 数据集存储（Azure 追踪、ShareGPT 缓存）
├── models/                 ← 大模型权重存储（下载后）
├── lora_adapters/          ← LoRA 适配器权重（合成或真实）
├── results/                ← 实验输出（JSON 结果、日志）
├── README.md               ← 项目概述与快速开始
├── PROJECT_STRUCTURE.md    ← 本文件：项目结构详解
├── EXPERIMENT_GUIDE.md     ← 实验执行完整指南
├── pyproject.toml          ← Python 包配置
└── requirements.txt        ← 依赖列表
```

---

## faaslora/ — 核心系统包

```
faaslora/
├── __init__.py             ← 包入口，导出主要类
├── preloading/             ← 贡献1：命中感知预加载
│   ├── __init__.py
│   ├── preloading_planner.py   ← PreloadingPlanner（热度/价值 0-1 背包与策略）
│   └── preloading_manager.py  ← PreloadingManager（执行预加载计划、扩缩事件触发）
├── memory/                 ← 贡献2：多层驻留控制
│   ├── __init__.py
│   ├── residency_manager.py   ← ResidencyManager（GPU/Host/NVMe/Remote 分层与准入驱逐）
│   ├── memory_coordinator.py ← MemoryCoordinator（推理与加载显存预算协调）
│   └── gpu_monitor.py         ← GPU 显存监控
├── storage/                ← 存储后端
│   ├── __init__.py
│   ├── local_cache.py      ← LoRA 本地缓存（NVMe 目录）
│   └── remote_client.py    ← 远端存储访问（带宽限速模拟）
├── scheduling/             ← 贡献3：资源协同调度
│   ├── __init__.py
│   └── resource_coordinator.py  ← ResourceCoordinator（预留、排队、scale-down 暖池；可选对接 ResidencyManager）
├── coordination/           ← 扩缩决策（与实验路径共用同一逻辑）
│   ├── __init__.py
│   └── autoscaler.py       ← AutoScaler（ScalingMetrics/ScalingDecision；make_scaling_decision_with_metrics 供实验复用）
├── experiment/             ← 实验集成（完整栈 + 在线热度 + 实例池）
│   ├── __init__.py
│   ├── experiment_config.py   ← 实验用 Config 包装（dot get）
│   ├── experiment_stack.py    ← ExperimentStack（Registry + ResidencyManager + PreloadingManager + Coordinator）
│   ├── hotness_tracker.py     ← HotnessTracker（在线热度更新 Registry）
│   └── instance_pool.py      ← InstancePool、Router（多实例与路由，B1/B2）
├── datasets/               ← 数据集加载与工作负载生成
│   ├── __init__.py
│   ├── dataset_loader.py   ← AzureTraceLoader、ShareGPTLoader、WorkloadDataset、AzureTraceReplay
│   └── workload_generator.py    ← WorkloadGenerator、WorkloadConfig、RequestTrace
├── registry/               ← 工件元数据
│   ├── schema.py           ← ArtifactMetadata、StorageTier、PreloadingPlan
│   └── artifact_registry.py   ← ArtifactRegistry
└── serving/                ← 推理引擎封装
    ├── __init__.py
    ├── inference_engine.py ← InferenceEngine（vLLM / 完整栈用）
    └── vllm_wrapper.py     ← vLLM 封装
```

说明：贡献2 的“热–温–冷”分层由 **memory/ResidencyManager** 实现；预加载入口为 ExperimentStack.preload / PreloadingManager.trigger_scaling_preload，实验完整栈默认开启（`--no-full-stack` 可关闭）。

### faaslora/preloading/preloading_planner.py 与 preloading_manager.py

**作用**：实现贡献1的命中感知预加载（与扩缩事件绑定）。

**关键类**：`PreloadingPlanner`、`PreloadingManager`

**功能**：
- 基于 Registry 的 hotness_score、value_per_byte 与扩缩事件生成预加载计划（0-1 背包/贪心）
- `PreloadingManager.trigger_scaling_preload(scaling_event)` 在 scale-up 时触发预加载
- 执行计划时通过 ResidencyManager 将工件准入到目标 tier（如 GPU/NVMe）

**主要方法**：
- `generate_preloading_plan(target_tier, capacity_bytes, scaling_event)` — 生成计划
- `trigger_scaling_preload(scaling_event)` — 扩缩事件触发预加载
- `execute_preloading_plan(plan_result)` — 执行计划

---

### faaslora/storage/local_cache.py

**作用**：存储层 LoRA 本地缓存（NVMe 目录），供下载与本地命中。**贡献2 的多层驻留由 memory/ResidencyManager 实现**；LocalCache 仅负责本地磁盘缓存与驱逐。

**关键类**：`LocalCache`

**分层设计**（与 ResidencyManager 的 GPU/Host/NVMe/Remote 配合）：
```
GPU 显存（热层）  ←→  NVMe SSD（温层）  ←→  远端存储（冷层）
  容量：gpu_budget_mb       由 nvme_capacity_mb 控制      无限容量，但高延迟
  延迟：0ms（已驻留）       延迟：5~50ms                   延迟：50~500ms
```

**驱逐策略**：LRU（最近最少使用），`_evict_lru()` 负责将低频工件降级到下一层

**主要方法**：
- `load_adapter(adapter_id)` — 按层次查找并加载，返回命中层级和耗时
- `evict_to_nvme(adapter_id)` — 将 GPU 显存中的工件降级到 NVMe
- `warm_pool_retain(hot_set)` — scale-down 时保留高频工件于 GPU

---

### faaslora/storage/remote_client.py

**作用**：模拟远端对象存储（S3/OSS）访问，通过 `bandwidth_mbps` 参数控制网络传输延迟。

**关键类**：`RemoteStorageClient`

**带宽模拟**：
```python
transfer_time_ms = size_mb / bandwidth_mbps * 8000   # Mbps→ms
```

默认带宽为 100 Mbps，与同类论文实验设置一致。

---

### faaslora/scheduling/resource_coordinator.py

**作用**：实现贡献3的扩缩容与显存协同调度核心逻辑。

**关键类**：`ResourceCoordinator`

**核心参数**（来自 `configs/experiments.yaml` 的 `resource_coordination` 与 `hardware`）：

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `min_instances` / `max_instances` | 实例池上下限（实验多实例默认 2） | 1 / 2 |
| `gpu_budget_mb` | GPU 显存总量 | 24576（3090 24GB） |
| `model_weights_mb` | backbone 权重占用 | 14336（7B 模型） |
| `kv_per_1k_tokens_mb` | 每千 token 的 KV cache 占用 | 2.0 MB |
| `lora_load_reserve_ratio` | 为 LoRA 加载预留的显存比例 | 0.15 |
| `warm_pool_size` | scale-down 后暖池保留 LoRA 数量 | 4 |

**说明**：实验实例数以 `configs/experiments.yaml` 的 `resource_coordination.min_instances` / `max_instances` 为准；`faaslora/utils/config.py` 中 `coordination.autoscaling` 默认 `max_instances=10` 为 API/生产用，与实验配置无关。

**三个核心行为**：

1. **scale-up 协调**：`request_lora_load()` — 检测显存压力，渐进式等待代替立即驱逐
2. **竞争检测**：当 LoRA 加载、KV cache、批推理同时占用显存时，
   - 无协调（`faaslora_no_coord`）：竞争惩罚延迟直接加入 TTFT（80~500ms）
   - 有协调（`faaslora_full`）：转化为可控的渐进等待（10~60ms）
3. **scale-down 暖池**：`trigger_scale_down()` — 保留热度最高的 `warm_pool_size` 个 LoRA 于 GPU，释放其余

---

### faaslora/datasets/dataset_loader.py

**作用**：加载真实数据集，驱动实验工作负载生成。

**关键类**：

**`AzureTraceLoader`**
- 读取 `data/` 目录下的 Azure LLM 推理追踪 CSV 文件
- 字段：时间戳、输入 token 数、输出 token 数
- 支持按工作负载类型过滤（`conv`/`code`/`mixed`）

**`ShareGPTLoader`**
- 优先从本地缓存 `data/sharegpt_cache.jsonl` 读取
- 不存在时尝试从 HuggingFace 下载（需网络）
- 失败时使用内嵌的 200 条对话作为 fallback

**`AzureTraceReplay`**
- 核心类：将 Azure 追踪的真实时间戳回放为实验请求序列
- `time_scale_factor=0.1`：将 1 小时追踪压缩为 6 分钟实验
- Zipf 分布选择 LoRA 适配器（`zipf_exponent=1.0`，模拟 80/20 热度分布）
- 用 ShareGPT 真实提示词填充每条请求

**`WorkloadDataset`**
- 组合 Azure + ShareGPT 数据
- `generate_traces()` 方法：返回 `List[RequestTrace]`

---

### faaslora/datasets/workload_generator.py

**作用**：工作负载生成器，作为 Azure 真实追踪不可用时的 Poisson 合成 fallback。

**关键类**：

**`RequestTrace`（数据类）**
```python
@dataclass
class RequestTrace:
    request_id: str          # 请求唯一 ID（"req_00001"）
    arrival_time: float      # 到达时间（秒，相对实验开始）
    prompt: str              # 提示词文本
    adapter_id: str          # 请求携带的 LoRA 适配器 ID
    adapter_domain: str      # 适配器领域（"finance"/"medical"/"code"...）
    expected_input_tokens: int
    expected_output_tokens: int
```

**`WorkloadConfig`**
- 控制合成工作负载的参数集合
- 包括到达率、请求数、Zipf 参数、热度演化等

**`WorkloadGenerator`**
- 基于 Poisson 过程生成请求到达时间
- 支持 `enable_hotness_evolution`：热度随时间演化，模拟真实用户行为变化

---

### faaslora/serving/inference_engine.py

**作用**：统一封装推理引擎接口，自动切换 vLLM（真实 GPU）和 Mock 模式。

**关键类**：`InferenceEngine`

**真实 GPU 模式**（vLLM 已安装 + CUDA 可用）：
```python
engine = vllm.AsyncLLMEngine.from_engine_args(
    EngineArgs(model=model_path, max_loras=8, ...)
)
```
- `max_loras=8`：vLLM 在 GPU 同时缓存 8 个 LoRA 适配器
- 调用 `engine.generate()` 执行真实 LLM 前向计算
- TTFT = 真实首 token 延迟（受 GPU 算力、批大小、序列长度影响）
- TPOT = 真实每 token 生成时间

**Mock 模式**（无 GPU 时）：
- 基于标定的延迟模型生成仿真延迟
- 延迟参数来源于真实 GPU 实验或论文数据
- 用于框架验证和参数调试，**不能代替真实 GPU 实验的结果数据**

---

## scripts/ — 可执行脚本

```
scripts/
├── run_all_experiments.py      ← 主实验脚本（入口点）
├── generate_lora_adapters.py   ← 生成合成 LoRA 适配器权重
├── download_datasets.py        ← 下载 ShareGPT 数据集
├── download_model.py           ← 下载大模型（Qwen/Llama/OPT）
└── setup_gpu.sh                ← GPU 环境一键安装
```

### run_all_experiments.py

**作用**：实验总入口，统一管理工作负载生成、场景执行、结果聚合和输出。

**运行流程**：
```
1. 解析 YAML 配置文件
2. 检测环境（vLLM / GPU / Mock）
3. 下载/验证数据集（Azure 追踪、ShareGPT）
4. 生成 LoRA 适配器（如未存在）
5. 生成请求序列（Azure 追踪回放 或 Poisson 合成）
6. 按顺序执行各实验场景（cold_start / slora_style / ... / faaslora_full）
7. 聚合指标，打印对比表格，保存 JSON 结果
```

**关键命令行参数**：
- `--config`：指定配置文件路径（默认 `configs/experiments.yaml`）
- `--quick`：快速模式，每个场景仅 30 个请求（调试用）
- `--scenario`：只运行指定场景（如 `--scenario faaslora_full`）

### generate_lora_adapters.py

**作用**：生成 LoRA 适配器文件，供实验加载使用。

**两种模式**：
- `--synthetic`：生成最小合成权重（仅包含正确的元数据结构，不依赖真实模型）
- `--peft`（默认）：使用 PEFT 库生成标准 LoRA 权重文件（需要基础模型）

**输出目录**：`lora_adapters/`，每个适配器一个子目录：
```
lora_adapters/
├── finance_lora/           # 金融领域（热度 0.9）
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── medical_lora/           # 医疗领域（热度 0.8）
├── code_lora/              # 代码领域（热度 0.75）
├── edu_lora/               # 教育领域（热度 0.6）
└── legal_lora/             # 法律领域（热度 0.4）
```

### download_model.py

**作用**：下载 HuggingFace 大模型至本地 `models/` 目录，并自动更新 `experiments.yaml`。

**支持模型**（均适用于 RTX 3090 24GB）：

| 模型 | 大小 | 显存 | 适用场景 |
|------|------|------|----------|
| Qwen/Qwen2.5-0.5B-Instruct | 1 GB | 2 GB | 快速验证 |
| Qwen/Qwen2.5-7B-Instruct | 15 GB | 18 GB | 论文实验推荐 |
| facebook/opt-1.3b | 2.6 GB | 4 GB | 无 Token 授权备选 |
| meta-llama/Llama-3.1-8B-Instruct | 16 GB | 20 GB | 与 S-LoRA 同基座 |

**使用方式**：
```bash
# 国内网络（需设置镜像）
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct
```

---

## configs/ — 实验配置

```
configs/
└── experiments.yaml        ← 主配置文件
```

### experiments.yaml 结构

```yaml
experiment:
  name: "..."               # 实验名称
  description: "..."        # 描述

model:                      # 基础大模型配置
  name: "Qwen/Qwen2.5-0.5B-Instruct"   # 模型路径（本地或 HF ID）
  tensor_parallel_size: 1   # 张量并行数（双 3090 改为 2）
  max_model_len: 2048        # 最大序列长度
  gpu_memory_utilization: 0.85          # vLLM GPU 显存使用率
  max_loras: 8              # vLLM 同时缓存的 LoRA 数量

hardware:                   # 硬件参数（用于调度模拟）
  gpu_budget_mb: 24576      # GPU 总显存（3090 = 24GB）
  model_weights_mb: 1000    # backbone 占用（0.5B≈1G，7B≈14G）
  kv_per_1k_tokens_mb: 0.5  # 每千 token KV cache（7B 约 2.0）

lora_adapters:              # LoRA 适配器定义列表
  - id: "finance_lora"
    hotness: 0.9            # 访问热度（越高越靠近 GPU 热层）
    domain: "finance"
    size_mb: 30

storage:                    # 存储层参数
  bandwidth_mbps: 100       # 远端存储→本地的网络带宽

workload:                   # 工作负载参数
  workload_type: "mixed"    # 请求类型（conv/code/mixed）
  time_scale_factor: 0.1    # 时间压缩比（0.1 = 1h 追踪→6min 实验）
  total_requests: 500       # 总请求数
  lora_request_ratio: 0.85  # 携带 LoRA 的请求比例
  zipf_exponent: 1.0        # LoRA 热度 Zipf 参数

scenarios:                  # 实验场景列表
  - name: "cold_start"
    baseline_type: "cold_start"
  - name: "slora_style"
    baseline_type: "slora_style"
  # ... 更多场景 ...
```

---

## data/ — 数据目录

```
data/
├── AzureLLMInferenceTrace_*.csv        ← Azure 生产推理追踪（已包含）
└── sharegpt_cache.jsonl                ← ShareGPT 对话缓存（运行后生成）
```

**Azure 追踪格式**（每行一条推理请求）：
```csv
TIMESTAMP,ContextTokens,GeneratedTokens,Model,Region
2023-04-01 00:00:01,512,128,GPT-4,...
```

---

## models/ — 模型目录

```
models/
└── Qwen--Qwen2.5-7B-Instruct/      ← 下载后的模型权重
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── model-000*.safetensors
```

**初始状态**：空目录（需运行 `python scripts/download_model.py` 下载）

---

## results/ — 实验结果

```
results/
└── experiment_results.json          ← 完整实验结果（每次运行覆盖）
```

### JSON 结构

```json
{
  "元数据": {
    "运行时间": "2024-...",
    "数据集": "Azure LLM 真实追踪",
    "vllm": true/false,
    "cuda": true/false
  },
  "对比结果汇总表": [
    {
      "场景名称": "faaslora_full",
      "系统类型": "faaslora_full",
      "TTFT均值_ms": 183.2,
      "TTFT_P99_ms": 620.1,
      "吞吐量_RPS": 2.48,
      "QPR性价比": 3578,
      "vs_baseline": {
        "TTFT改善%": 62.3,
        "P99改善%": 42.5,
        "RPS提升%": 19.6
      }
    }
  ],
  "SOTA重点对比": [...],
  "详细请求数据": {...}
}
```

---

## lora_adapters/ — LoRA 适配器

```
lora_adapters/
├── finance_lora/           ← 金融领域（热度最高，应驻留 GPU 热层）
├── medical_lora/           ← 医疗领域（高热度）
├── code_lora/              ← 代码生成
├── edu_lora/               ← 教育问答
└── legal_lora/             ← 法律文本（热度最低）
```

**当前内容**：合成适配器（`--synthetic` 模式生成的最小权重文件）

**真实适配器**：在真实 GPU 模式下，可用 `--peft` 模式基于下载的基础模型生成标准 LoRA 权重。

---

## 关键数据流

```
Azure Trace CSV
    └─→ AzureTraceLoader.load()
            └─→ AzureTraceReplay.replay()
                    │  按真实时间戳排列，Zipf LoRA 选择
                    └─→ List[RequestTrace]
                              │
                              ▼
                        ScenarioRunner._serve()
                              │  按 arrival_time 调度
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
              _resolve_lora()      engine.generate()
              （判断命中层：           （vLLM 推理）
               GPU/NVMe/Remote）
                    │
                    ▼
              ResourceCoordinator
              （贡献3 协调）
                    │
                    ▼
              记录延迟指标
              （TTFT / TPOT / E2E）
```
