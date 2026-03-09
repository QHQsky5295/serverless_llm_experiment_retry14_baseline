# FaaSLoRA：面向多 LoRA 大模型推理的扩缩容感知Serverless系统

FaaSLoRA 是一个面向真实云工作负载的 Serverless LLM 推理实验系统，完整实现了以下三项研究贡献，并提供与同类 SOTA 系统（S-LoRA、ServerlessLLM）的对比实验框架。

---

## 研究背景与动机

在 Serverless 架构中，LLM 推理面临三个核心挑战：

| 观察 | 问题 | FaaSLoRA 贡献 |
|------|------|--------------|
| 观察1 | 扩容实例缺乏工件命中感知，新实例仍需冷加载 LoRA | **贡献1**：命中感知预加载 |
| 观察2 | 单层缓存策略无法兼顾命中率与显存利用率 | **贡献2**：多层工件驻留控制 |
| 观察3 | 工件加载与 KV 缓存、批推理争抢显存 | **贡献3**：资源协同调度 |

---

## 三项核心贡献

### 贡献1：命中感知的工件预加载

- **机制**：扩容决策时，调度器基于近期请求分布和 Zipf 热度预测，将高频 LoRA 工件提前加载至待启动实例
- **效果**：新实例启动后无需冷加载，直接命中缓存，大幅降低 TTFT
- **实现**：`faaslora/preloading/` — `PreloadingPlanner` + `PreloadingManager`；实验可选完整栈 `faaslora/experiment/ExperimentStack`（含 ResidencyManager + 扩缩触发预加载）
- **实验默认**：**多实例**（`configs/experiments.yaml` 中 `max_instances=2`，启动即多槽位、每槽位独立 ResourceCoordinator；单实例可将 `max_instances` 设为 1）。新实例若通过 `engine_factory` 扩容，会做热点 warmup 以贴近「新实例直接命中缓存」。

### 贡献2：多层工件驻留控制（热–温–冷分层）

- **热层（Hot）**：高频工件驻留 GPU 显存，TTFT = 模型前向时间
- **温层（Warm）**：中频工件驻留 NVMe，加载时间 = NVMe 传输时间（约 20ms）
- **冷层（Cold）**：低频工件驻留远端存储，加载时间 = 网络传输时间（100+ ms）
- **动态迁移**：根据显存预算和工件访问频率，实时调整各层工件集合
- **实现**：`faaslora/memory/ResidencyManager`（GPU/Host/NVMe/Remote 分层与动态迁移）；`faaslora/storage/` — `LocalCache`、`RemoteStorageClient`

### 贡献3：扩缩容与显存协同调度

- **扩容阶段**：预留工件加载空间，渐进式加载策略，优先保障 KV 缓存与批推理
- **负载回落阶段**：逐步释放显存，保留高频工件于暖池（Warm Pool），快速响应后续请求
- **竞争**：来自真实显存压力；有协调时加载排队等待可用空间，无协调时强制驱逐冷 LoRA，不注入人为随机惩罚。
- **实现**：`faaslora/scheduling/` — `ResourceCoordinator`（可选对接 `ResidencyManager` 统一显存状态）

---

## 实验系统架构

```
请求输入（Azure LLM 真实追踪时间戳 + ShareGPT 提示词）
    │
    ▼
WorkloadDataset（工作负载生成器）
    │  AzureTraceReplay：按真实时间戳回放，Zipf LoRA 选择
    │
    ▼
ScenarioRunner（场景执行引擎）
    │
    ├── 缓存层：RemoteStorage → NVMe → GPU VRAM
    │           （贡献2：多层驻留）
    ├── 预加载：PreloadingPlanner + PreloadingManager / ExperimentStack
    │           （贡献1：命中感知；默认完整栈，--no-full-stack 可关闭）
    ├── 调度：ResourceCoordinator（贡献3：资源协同）
    └── 多实例与路由：InstancePool + Router
              （默认多槽位，每槽位独立 coordinator；请求经 Router 选实例）
    │
    ▼
InferenceEngine（vLLM AsyncLLMEngine / Mock）
    │
    ▼
实验结果：TTFT / TPOT / E2E / RPS / QPR / 成本
```

---

## 数据集

| 数据集 | 来源 | 用途 |
|--------|------|------|
| Azure LLM Inference Trace | Microsoft Research（学术公开） | 请求到达时间戳、输入输出 token 数 |
| ShareGPT | 开源对话数据集 | 真实用户提示词文本 |
| Zipf 分布 | 参数化生成 | LoRA 适配器访问热度模拟 |

Azure 追踪数据已包含在 `data/` 目录（28K 条生产记录）。ShareGPT 首次运行时自动下载（可选），未下载时使用内嵌的 200 条替代。

---

## 对比实验设计

| 场景名称 | 对应系统 | 描述 |
|----------|----------|------|
| `cold_start` | 基线 | 每次请求均冷加载 LoRA，无任何缓存 |
| `slora_style` | S-LoRA | GPU 显存 LRU 缓存，多 LoRA 批处理 |
| `serverlessllm` | ServerlessLLM | GPU 显存 + SSD 快速检查点恢复 |
| `faaslora_nvme` | FaaSLoRA 贡献1 | 命中感知预加载，NVMe 缓存 |
| `faaslora_no_coord` | FaaSLoRA 贡献1+2 | 多层驻留，无协调调度（消融） |
| `faaslora_full` | **FaaSLoRA 完整** | 贡献1+2+3，完整系统 |
| `backbone_only` | 下限参考 | 纯 backbone 推理，无 LoRA 开销 |

---

## 快速开始

### 方案A：Mock 模式（无 GPU，用于验证框架）

```bash
conda activate LLM
cd serverless_llm_experiment
pip install -e .
python scripts/generate_lora_adapters.py --synthetic
python scripts/run_all_experiments.py --config configs/experiments.yaml --quick
```

### 方案B：真实 GPU 推理（推荐，RTX 3090）

**推荐**：使用经联网查证、适配 RTX 3090 + vLLM LoRA 的**完整重装**（Python 3.11 + PyTorch 2.4+cu124 + vLLM 0.10.2），详见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。

```bash
cd serverless_llm_experiment

# 第一步：重装 GPU 环境（删除并重建 LLM 环境，推荐）
bash scripts/reinstall_env.sh
conda activate LLM

# 或仅在已有 LLM 内追加安装（不重装）
# conda activate LLM && bash scripts/setup_gpu.sh

# 第二步：下载大模型（二选一）
python scripts/download_model.py --model Qwen/Qwen2.5-0.5B-Instruct  # 快速验证
python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct    # 论文实验

# 第三步：下载 ShareGPT 数据集（可选）
python scripts/download_datasets.py

# 第四步：运行完整实验
python scripts/run_all_experiments.py --config configs/experiments.yaml

# 默认即使用完整栈（ResidencyManager + PreloadingManager + 扩缩触发预加载）；关闭完整栈加 --no-full-stack
python scripts/run_all_experiments.py --config configs/experiments.yaml
```

环境版本与重装说明见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)；若遇 vLLM LoRA 崩溃见 [docs/VLLM_RTX3090_LORA.md](docs/VLLM_RTX3090_LORA.md)。

---

## 输出结果示例

```
════════════════════════════════════════════════════════
  系统 / 场景       类型          TTFT均值  P99   RPS   QPR    命中%
  cold_start     [基线]            486ms 1078ms  2.08  1130     0%
  slora_style    [SOTA] S-LoRA     174ms  627ms  2.67  4032    87%
  serverlessllm  [SOTA] SlessLLM   176ms  626ms  2.58  3874    87%
  faaslora_full  [本文] 完整       183ms  620ms  2.48  3578    83% ◄
════════════════════════════════════════════════════════
```

实验结果同时保存至 `results/experiment_results.json`（含论文表格格式）。

---

## 项目结构

详见 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)（各文件夹和文件的详细说明）。

## 实验指南

详见 [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)（从环境配置到结果解读的完整步骤）。

---

## 参考文献

- **S-LoRA**：Sheng et al., "S-LoRA: Serving Thousands of Concurrent LoRA Adapters", MLSys 2024
- **ServerlessLLM**：Fu et al., "ServerlessLLM: Low-Latency Serverless Inference for Large Language Models", OSDI 2024
- **Azure LLM Trace**：Microsoft Azure 生产推理追踪数据集
