# FaaSLoRA 实验执行完整指南

> 说明：本文件保留为补充操作手册。当前仓库的**权威口径**以 [README.md](README.md)、[docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md](docs/TECHNICAL_ROUTE_AND_IMPLEMENTATION.md) 和 [docs/PROJECT_PROGRESS.md](docs/PROJECT_PROGRESS.md) 为准；若本文与当前实现冲突，以这三份文档为准。

本指南面向未接触过本项目的研究人员，按步骤说明如何从零配置环境、运行实验、解读结果，以及如何与同类论文系统进行对比。

---

## 目录

1. [系统要求](#1-系统要求)
2. [环境配置（GPU 真实推理）](#2-环境配置gpu-真实推理)
3. [数据集准备](#3-数据集准备)
4. [下载大模型](#4-下载大模型)
5. [生成 LoRA 适配器](#5-生成-lora-适配器)
6. [配置实验参数](#6-配置实验参数)
7. [运行实验](#7-运行实验)
8. [解读实验结果](#8-解读实验结果)
9. [Mock 模式（无 GPU）](#9-mock-模式无-gpu)
10. [常见问题](#10-常见问题)

---

## 1. 系统要求

### 硬件要求（推荐）

| 组件 | 最低 | 推荐 |
|------|------|------|
| GPU | 1× RTX 3090 (24GB) | 2× RTX 3090 |
| 系统内存 | 32 GB | 64 GB |
| 存储空间 | 100 GB（SSD） | 200 GB（NVMe SSD） |
| 操作系统 | Ubuntu 20.04+ | Ubuntu 22.04 |

### 软件要求

| 软件 | 版本 |
|------|------|
| CUDA | 11.8+ 或 12.1+ |
| Python | 3.12（当前稳定环境） |
| PyTorch | 2.8.0+cu128 |
| Conda 虚拟环境 | `LLM_vllm0102`（当前主线环境） |

> **说明**：当前主线实验在 **LLM_vllm0102** 环境中验证通过；精确版本见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。

---

## 2. 环境配置（GPU 真实推理）

### 2.1 激活虚拟环境

```bash
conda activate LLM_vllm0102
cd /home/qhq/serverless_llm_experiment
```

### 2.2 一键安装脚本（推荐）

```bash
bash scripts/setup_gpu.sh
```

该脚本自动完成：
- 安装核心依赖（peft、transformers、huggingface-hub 等）
- 安装 vLLM（用于真实 LLM 推理，约 5-10 分钟）
- 安装 FaaSLoRA 包（pip install -e .）
- 准备运行所需基础目录与依赖

> 说明：当前论文主线默认 LoRA 工件已切到 `PEFT+finetune`。即使完成环境安装，正式实验前仍建议按第 5 节单独生成与当前 base model 匹配的 PEFT 工件。

### 2.3 手动安装（如自动脚本失败）

```bash
# 基础依赖
pip install pyyaml safetensors peft transformers huggingface_hub accelerate

# vLLM（关键，与 PyTorch CUDA 版本需匹配）
pip install vllm

# 安装本项目包
pip install -e .
```

### 2.4 验证安装

```bash
python -c "
import torch, vllm
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('vLLM:', vllm.__version__)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name} {p.total_memory//1024**3}GB')
"
```

**预期输出**（双 3090）：
```
PyTorch: 2.8.0+cu128
CUDA: True
vLLM: 0.10.2
  GPU 0: NVIDIA GeForce RTX 3090 24GB
  GPU 1: NVIDIA GeForce RTX 3090 24GB
```

---

## 3. 数据集准备

### 3.1 Azure LLM 追踪（已内置）

Azure 追踪数据已包含在 `data/` 目录，无需下载：

```bash
ls data/
# AzureLLMInferenceTrace_1.csv
# AzureLLMInferenceTrace_2.csv
# ...
```

验证数据完整性：

```bash
python scripts/download_datasets.py --verify
```

### 3.2 ShareGPT 数据集（可选，提升提示词质量）

```bash
python scripts/download_datasets.py
```

> **说明**：ShareGPT 数据集提供真实用户对话文本作为提示词。未下载时，系统使用内嵌的 200 条对话 fallback，不影响 token 数量统计（仍来自 Azure 追踪），但提示词质量稍低。

> **国内网络**：如下载缓慢，设置镜像：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> python scripts/download_datasets.py
> ```

---

## 4. 下载大模型

这是真实 GPU 实验的**关键步骤**。当前 `models/` 目录为空，必须下载模型才能使用 vLLM 真实推理。

### 4.1 选择模型

```bash
python scripts/download_model.py --list
```

输出：
```
可用模型（适用于 RTX 3090 24 GB）：

  模型 ID                                        大小    显存  说明
  ─────────────────────────────────────────── ───── ─────  ──────────────
  Qwen/Qwen2.5-7B-Instruct                     14.5G   18G  论文实验推荐，RTX 3090 单卡可运行
  Qwen/Qwen2.5-14B-Instruct                    29.0G   30G  需要双 3090（tensor_parallel=2）
  mistralai/Mistral-7B-Instruct-v0.3          14.0G   18G  Qwen 之后的下一个家族，当前固定的小档位
  mistralai/Mistral-Nemo-Instruct-2407        24.0G   30G  后续大档位，双 3090 + TP=2
  gemma-3-12b-it                               24.0G   24G  暂挂计划列表，当前不配置
```

### 4.2 下载模型

**方案A：Qwen 中尺寸（Qwen2.5-7B，约 15 GB，30-60 分钟）**

```bash
export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像（可选）
python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct
```

**方案B：Qwen 13B+ 档（Qwen2.5-14B，双 3090 张量并行）**

```bash
export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像（可选）
python scripts/download_model.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --tensor-parallel 2
```

**方案C：已完成的小档位主线（Mistral-7B，单卡）**

```bash
export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像（可选）
python scripts/download_model.py --model mistralai/Mistral-7B-Instruct-v0.3
```

> 当前 Qwen 家族的 `3B / 7B / 14B` 主线验证已经完成；`mistralai/Mistral-7B-Instruct-v0.3` 的论文主线 `PEFT+finetune + 500 adapters + representative r1000` 也已完成。`Mistral-Nemo V2 publicmix r1000` 首轮稳定结果已完成，且 `opt1` 并发/显存温和优化已作为敏感性实验保留、未采纳为默认参数。`Mistral 7B V2 publicmix r1000` 首次尝试在当前环境下因 vLLM V1 + 异构 public LoRA 触发 `EngineCore / CUDA illegal memory access`，因此默认已切到更保守的 `V0 + no chunked prefill + no prefix caching + reduced concurrency` 路径后再重跑。`OPT` 已在本机验证为不支持 vLLM LoRA。Gemma 暂挂计划列表，当前不配置。

### 4.3 下载后配置提示

下载完成后，`download_model.py` 会打印推荐的 `FAASLORA_PROFILE_MODEL / FAASLORA_PROFILE_WORKLOAD` 与目标本地路径。
当前脚本**不会自动改写** `configs/experiments.yaml`，以免误改当前的 `profile_selection + model_profiles` 结构。
更推荐直接通过环境变量或 `profile_selection.model / dataset / workload` 统一切换主线实验组合，而不是分散手改多个字段。

---

## 5. 生成 LoRA 适配器

> **重要**：LoRA 适配器的权重维度必须与基础模型匹配。如果你更换了模型（如从 0.5B 切换到 7B），必须重新生成适配器。

```bash
# 论文主线默认：自动读取 experiments.yaml 中当前模型，生成 PEFT + 轻量 finetune 工件
python scripts/generate_lora_adapters.py --force

# 或手动指定模型
python scripts/generate_lora_adapters.py --model models/Qwen--Qwen2.5-7B-Instruct --force

# 仅 quick/debug 时，才显式回退到 synthetic
python scripts/generate_lora_adapters.py --model models/Qwen--Qwen2.5-7B-Instruct --synthetic --force
```

> **说明**：当前论文正式对比默认已经切到 `PEFT+finetune + two_phase`。也就是说，应先为每个 base model 预生成并冻结一整批 LoRA 工件，再启动正式实验。这样不同系统共享同一 base model 时，能够严格复用同一批工件、同一份负载和同一套热度轮换。若只是日常调试，也可以临时把 `lora_adapters.preparation_mode` 改成 `one_shot` 或 `bootstrap_once`。

> **当前默认解析方式**：若不显式传 `--model`，`scripts/generate_lora_adapters.py` 会跟随 `configs/experiments.yaml` 当前激活的 `profile_selection.model` 与对应 workload profile 的 adapter 数量来解析默认值，而不是只读取顶层兼容回退字段。

> **当前生成实现**：`PEFT+finetune` 模式现在会对同一轮 adapter 生成只加载一次 base model，再循环保存多个 adapter；因此重新启动后的新一轮生成会比旧版“每个 adapter 重新加载模型”更快。

当前下一步 `Mistral-Nemo + TP=2 + 500 adapters` 的推荐正式实验命令：

```bash
python scripts/download_model.py \
  --model mistralai/Mistral-Nemo-Instruct-2407

FAASLORA_PROFILE_MODEL=mistral_nemo_12b_tp2 \
FAASLORA_PROFILE_DATASET=azure_sharegpt_rep1000 \
FAASLORA_PROFILE_WORKLOAD=mistral_nemo_12b_tp2_main \
python scripts/run_all_experiments.py --config configs/experiments.yaml --scenario faaslora_full
```

如果你希望显式使用两阶段工作流，可先在 `configs/experiments.yaml` 中设置：

```yaml
lora_adapters:
  preparation_mode: "two_phase"
```

然后再手动执行：

```bash
python scripts/generate_lora_adapters.py \
  --model /home/qhq/serverless_llm_experiment/models/mistralai--Mistral-Nemo-Instruct-2407 \
  --num-adapters 500 \
  --use-peft \
  --finetune \
  --force
```

生成后检查：

```bash
ls artifacts/remote/
# finance_lora/  medical_lora/  code_lora/  support_lora/  legal_lora/  translate_lora/
```

---

## 6. 配置实验参数

编辑 `configs/experiments.yaml` 调整实验参数：

### 6.1 基础模型配置

```yaml
model:
  name: "models/Qwen--Qwen2.5-7B-Instruct"   # 模型本地路径
  tensor_parallel_size: 1   # 单卡；双 3090 改为 2
  max_model_len: 2048        # 最大序列长度
  gpu_memory_utilization: 0.85
  max_loras: 8              # vLLM 同时缓存的 LoRA 数
```

> **说明**：上面这段 `model:` 现在主要用于兼容旧路径与兜底读取。当前主线切换应优先通过 `profile_selection + model_profiles + dataset_profiles + workload_profiles` 完成，而不是散改顶层 `model/hardware/workload`。

### 6.2 硬件参数（贡献3 调度依赖）

```yaml
hardware:
  gpu_budget_mb: 24576      # 3090 = 24 × 1024 = 24576 MB
  model_weights_mb: 14336   # 7B 模型约 14 GB（fp16）
  kv_per_1k_tokens_mb: 2.0  # 7B 模型每千 token 约 2 MB KV cache
```

> **0.5B 模型对应值**：
> ```yaml
> model_weights_mb: 1000
> kv_per_1k_tokens_mb: 0.5
> ```

### 6.3 工作负载参数

```yaml
workload:
  workload_type: "mixed"        # 混合请求类型
  time_scale_factor: 0.1        # 时间压缩：0.1 = 1h 追踪→6min 实验
  total_requests: 500           # 每个场景处理的请求数
  lora_request_ratio: 0.85      # 85% 的请求携带 LoRA（参考 S-LoRA 论文）
  zipf_exponent: 1.0            # Zipf 参数：越大热度越集中
```

**时间压缩建议**：

| `time_scale_factor` | 实验时长 | 适用场景 |
|---------------------|----------|----------|
| 1.0 | 约 60 分钟 | 完整复现 |
| 0.1 | 约 6 分钟 | 论文实验（推荐） |
| 0.05 | 约 3 分钟 | 快速调试 |

### 6.4 存储带宽参数

```yaml
storage:
  bandwidth_mbps: 100   # 100 Mbps = 典型云存储（S3/OSS）
  nvme_capacity_mb: 20480
```

### 6.5 贡献3 协同调度参数

```yaml
scenarios:
  - name: "faaslora_no_coord"
    baseline_type: "faaslora_no_coord"
    resource_coordination:
      coordination_enabled: false

  - name: "faaslora_full"
    baseline_type: "faaslora_full"
    resource_coordination:
      coordination_enabled: true        # 开启协调：在显存紧张时排队等待可用空间
      effective_capacity_admission_enabled: true  # 当前默认：开启 P2.5 有效容量准入
      max_concurrent_loads: 2          # 渐进式启动：最多 2 个并发 LoRA 加载
      warm_pool_size: 4                # scale-down 后 GPU 暖池保留的 LoRA 数量
      idle_timeout_s: 10               # 低负载持续时间阈值，触发 scale-down
```

---

## 6.6 与 SOTA 的对比方式（E2）

本实验与 **S-LoRA**、**ServerlessLLM** 的对比采用 **同一代码库内的 baseline 场景**，而非延迟公式复现：

| 对比对象 | 本仓库中的对应场景 | 说明 |
|----------|-------------------|------|
| S-LoRA (SOSP'23/MLSys'24) | `slora_style` | 同一代码库：GPU 显存 LRU 缓存、多 LoRA 批处理；TTFT/TPOT 来自真实 vLLM 推理（或 Mock 标定）。 |
| ServerlessLLM (NSDI'24) | `serverlessllm` | 同一代码库：GPU + SSD 检查点恢复路径；延迟与命中率在本框架内测量。 |
| 无缓存基线 | `cold_start` | 每次请求冷加载 LoRA，用于计算改善百分比。 |

**不做**：使用论文中的延迟公式（如固定 compute_slora_load_ms）在代码外复现曲线；所有延迟与命中率均来自本仓库运行结果，便于审稿与复现。

**单实例 vs 多实例**：当前主线默认是 `instance_mode=auto`、`min_instances=1`、`max_instances=2`。也就是说，系统启动时先保留 1 个物理 runtime，在请求压力达到阈值时再 scale-up 到第 2 个物理 runtime，而不是“启动即双实例常驻”。单实例验证可将 `configs/experiments.yaml` 中 `resource_coordination.max_instances` 设为 1。实验实例数一律以 `configs/experiments.yaml` 的 `min_instances` / `max_instances` 为准；`faaslora/utils/config.py` 中的默认值属于 API/生产配置，不作为当前主实验默认口径。

---

## 7. 运行实验

### 7.1 快速验证（Mock 模式，无 GPU）

```bash
python scripts/run_all_experiments.py \
  --config configs/experiments.yaml \
  --quick
```

`--quick` 模式每个场景处理 30 个请求，约 2 分钟完成，用于验证框架正确性。

### 7.2 完整实验（真实 GPU 推理）

```bash
conda activate LLM_vllm0102
cd /home/qhq/serverless_llm_experiment
python scripts/run_all_experiments.py \
  --config configs/experiments.yaml
```

当前这条默认命令已经对齐到主线配置：
- `auto + 500 LoRA + representative 1000 requests`
- `effective_capacity_admission_enabled=true`
- `max_num_seqs=8`
- `max_loras=8`
- `max_num_batched_tokens=4096`
- `runtime_concurrency_cap=8`

在当前双 3090 机器上，`Qwen2.5-3B-Instruct` 这条主线完整运行的实测耗时约为 `46-47` 分钟。

> **建议**：从新的 SSH/TTY 会话启动这条命令，避免当前 session 处于 `closing` 状态时被 systemd 强制终止。

> **首次运行可能较慢**：vLLM 首次加载模型时需要编译内核，后续运行会使用缓存。

> **P2.5 开关**：当前仓库默认配置已将 `effective_capacity_admission_enabled` 切到开启状态。若要做对照实验，可通过环境变量显式关闭：
>
> ```bash
> FAASLORA_EFFECTIVE_CAPACITY_ADMISSION=0 bash scripts/run_validation_bundle.sh custom
> ```

> **7B 当前已验证参数**：`Qwen2.5-7B-Instruct` 现阶段已验证通过的默认参数为 `auto + 100 adapters + 1000 requests + concurrency=4 + runtime_concurrency_cap=4 + max_num_seqs=4 + max_loras=2 + max_num_batched_tokens=1024 + effective_capacity_admission_enabled=true`。当前主线正在补 `3B + P2.5 on` 复验，用于统一 3B/7B 的默认口径。

> **3B 当前复验结论**：`Qwen2.5-3B auto500 + representative1000 + P2.5 on` 已复验完成。相对旧的 `seq8_lora8` frozen baseline，TTFT/tail 只有小幅波动、`contention=0 / defer=0` 仍保持不变。因此对 3B 而言，P2.5 主要用于统一默认口径，而不是当前性能增益来源。

### 7.3 运行单个场景

```bash
cd /home/qhq/serverless_llm_experiment
python scripts/run_all_experiments.py \
  --config configs/experiments.yaml \
  --scenario faaslora_full
```

这条命令现在对应的也是当前主线默认场景。

### 7.4 实时进度

下列是当前主线路径的典型输出形态；具体数值会随复现实验略有波动：

```text
Scenario: faaslora_full  [faaslora_full]
[Phase 1] Preloading ...
  GPU warmup ...
[Phase 2] Serving 1000 requests ...
Live: done=625/1000  active=8  queued=0  inst=2  hit=94.6%
Done: 1000/1000  TTFT_avg=1.4s  P95=4.1s  P99=6.7s  RPS=0.36  Hit=94.6%
```

---

## 8. 解读实验结果

### 8.1 终端输出表格

实验结束后打印 4 张表格：

#### 表1：完整指标对比

```
系统 / 场景         类型          TTFT均值    P95     P99    TPOT   E2E_P99   RPS   成本/请求    QPR    命中%  GPU%  完成数
cold_start       [基线]            486ms  1004ms  1078ms  12.6ms    3812ms  2.08 $0.003777   1130     0%    0%  500/500
slora_style      [SOTA] S-LoRA     174ms   556ms   627ms  12.1ms    2747ms  2.67 $0.003794   4032    87%   70%  500/500
faaslora_full    [FaaSLoRA] 完整   183ms   558ms   620ms  12.2ms    3719ms  2.48 $0.003796   3578    83%   70%  500/500
```

**各指标含义**：

| 指标 | 含义 | 越 |
|------|------|---|
| TTFT均值 | 首 token 延迟均值（ms） | 低越好 |
| P95/P99 | 尾部延迟（ms） | 低越好 |
| TPOT | 每 token 生成时间（ms） | 低越好 |
| E2E_P99 | 端到端 P99 延迟 | 低越好 |
| RPS | 吞吐量（请求/秒） | 高越好 |
| 成本/请求 | 单请求计算成本（USD） | 低越好 |
| QPR | 性价比（请求质量/成本） | 高越好 |
| 命中% | LoRA 缓存命中率（无需远端加载） | 高越好 |
| GPU% | GPU 热层命中率（0ms 加载） | 高越好 |

#### 表2：相对改进（百分比）

```
slora_style    [SOTA]     TTFT ↓+64%  P95 ↓+45%  P99 ↓+42%  RPS ↑+28%  QPR ↑+257%
faaslora_full  [本文]     TTFT ↓+62%  P95 ↓+44%  P99 ↓+43%  RPS ↑+20%  QPR ↑+217%
```

> 负数 = 比基线差（如 `↓-5%` 表示比 S-LoRA 高 5%，即 TTFT 略高）

#### 表3：SOTA 重点对比（论文表格）

直接对应论文实验表格格式：

```
系统                  TTFT 均值  TTFT P99    TPOT   吞吐 RPS      QPR    命中率
[基线] 无缓存             486ms    1078ms  12.6ms      2.08     1130      0%
[SOTA] S-LoRA            174ms     627ms  12.1ms      2.67     4032     87%
[SOTA] ServerlessLLM     176ms     626ms  11.2ms      2.58     3874     87%
[FaaSLoRA] 完整          183ms     620ms  12.2ms      2.48     3578     83% ◄ 本文
```

#### 表4：贡献3 专项指标

```
场景               类型      P99_TTFT  竞争次数  竞争惩罚  调度延迟  暖池命中
faaslora_nvme    [贡献1]      671ms       -         -        -        0
faaslora_no_coord [消融]      740ms       2      406ms        -       21
faaslora_full    [本文]       620ms       -         -      17ms       21

贡献3 量化价值：P99 改善 120ms（无协调 740ms → 有协调 620ms）
```

**表4 各项含义**：

| 指标 | 含义 |
|------|------|
| 竞争次数 | 发生 KV-cache vs LoRA 加载显存竞争的次数 |
| 竞争惩罚 | 竞争导致的延迟惩罚均值（无协调时加入 TTFT） |
| 调度延迟 | 协调调度引入的渐进式等待时间（远小于竞争惩罚） |
| 暖池命中 | scale-down 后 GPU 暖池中的 LoRA 被命中的次数 |

### 8.2 JSON 结果文件

结果保存至 `results/experiment_results.json`，结构如下：

```json
{
  "元数据": { "运行时间": "...", "数据集": "Azure LLM 真实追踪", ... },
  "对比结果汇总表": [
    {
      "场景名称": "faaslora_full",
      "TTFT均值_ms": 183.2,
      "TTFT_P99_ms": 620.1,
      "vs_baseline": {
        "TTFT改善%": 62.3,
        "P99改善%": 42.5
      }
    }
  ],
  "SOTA重点对比": [...],
  "详细请求数据": { ... }
}
```

---

## 9. Mock 模式（无 GPU）

在没有 GPU 或 vLLM 未安装时，系统自动切换到 Mock 模式。

### Mock 模式与真实 GPU 模式的区别

| 方面 | Mock 模式 | 真实 GPU 模式 |
|------|-----------|--------------|
| TTFT 来源 | 标定的延迟模型（正态分布） | 真实 GPU 前向计算 |
| TPOT 来源 | 固定参数化估计 | 真实逐 token 生成时间 |
| LoRA 加载 | 基于 bandwidth_mbps 的时间模型 | 真实 safetensors 文件读取 |
| GPU warmup | `_gpu_warmed` 集合标记 | vLLM `engine.generate()` warmup 请求 |
| 贡献3 竞争 | ResourceCoordinator 在显存不足时对 LoRA 加载进行排队与驱逐 | 真实显存分配争抢 |
| 适合场景 | 框架验证、参数调试 | 论文实验数据 |

### Mock 模式延迟参数说明

Mock 模式的延迟来自 `configs/experiments.yaml` 中的 `scenarios[].latency_model`：

```yaml
scenarios:
  - name: "cold_start"
    latency_model:
      base_ttft_ms: 80        # 基础前向时间
      cold_load_extra_ms: 400 # 冷加载额外延迟
      tpot_ms: 12.0           # 每 token 时间
```

这些参数应根据真实 GPU 测量值进行标定。

---

## 10. 常见问题

### Q1：进程被杀死（"已杀死" / exit code 137）

**症状**：运行 `run_all_experiments.py` 后在 `Starting to load model...` 阶段被杀死

**原因**：Linux OOM Killer 发现系统内存（RAM）或 GPU 显存不足，杀死了进程。
vLLM 加载 7B 模型需要约 14 GB 系统 RAM + 约 18 GB 显存。

**解决方案**（在 `configs/experiments.yaml` 中逐步降低参数）：

```yaml
model:
  gpu_memory_utilization: 0.70   # 默认。如仍被杀死，降低到 0.60
  max_model_len: 1024             # 默认。降低到 512 可显著节省显存
  max_loras: 4                    # 默认。降低到 2 或 1
  enforce_eager: true             # 必须为 true，禁用 CUDA Graph 节省 2-4 GB
```

**如果仍然被杀死**：
1. 检查系统内存：`free -h`（建议至少 32 GB RAM）
2. 检查 GPU 显存：`nvidia-smi`（确认没有其他进程占用）
3. 如果 GPU 被其他进程占用：`nvidia-smi` → 找到占用显存的 PID → `kill PID`
4. 换用更小的快速验证模型（Qwen2.5-0.5B）：
   ```bash
   python scripts/download_model.py --model Qwen/Qwen2.5-0.5B-Instruct
   ```

### Q1b：显存不足（CUDA OOM）

**症状**：`torch.cuda.OutOfMemoryError`（进程没有被杀死，而是报 CUDA 错误）

**解决方案**：脚本内置了自动降级机制，会自动尝试 3 次递减的参数配置。
如果 3 次都失败，系统会自动切换到 Mock 模式。你也可以手动调整 YAML：

```yaml
model:
  gpu_memory_utilization: 0.60
  max_model_len: 512
  max_loras: 1
  enforce_eager: true
```

### Q2：vLLM 安装失败

```bash
# 方案1：优先使用当前主线稳定环境
conda activate LLM_vllm0102
python -c "import vllm; print(vllm.__version__)"

# 方案2：按 pyproject.toml 固定版本重装
pip install 'vllm==0.10.2'

# 方案3：使用 quick 模式做链路验证
python scripts/run_all_experiments.py --config configs/experiments.yaml --quick
```

### Q3：HuggingFace 下载超时

```bash
# 设置国内镜像
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct
```

### Q4：双 3090 如何配置

```yaml
# configs/experiments.yaml
model:
  tensor_parallel_size: 2       # 开启张量并行
  name: "models/Qwen--Qwen2.5-14B-Instruct"  # 14B 需要双卡
hardware:
  gpu_budget_mb: 49152          # 2 × 24576 MB
```

```bash
python scripts/download_model.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --tensor-parallel 2
```

### Q5：实验时间太长

```bash
# 减少请求数量
# 在 configs/experiments.yaml 中：
workload:
  total_requests: 200      # 默认 1000，改为 200

# 或使用 --quick 模式（每场景 30 请求）
python scripts/run_all_experiments.py --config configs/experiments.yaml --quick
```

### Q6：如何与更多 SOTA 论文对比

在 `configs/experiments.yaml` 的 `scenarios` 列表中添加新场景：

```yaml
scenarios:
  # ... 现有场景 ...
  
  # 添加 Punica 系统的 baseline
  - name: "punica_style"
    enabled: true
    baseline_type: "slora_style"    # 复用 S-LoRA 行为模型
    description: "Punica 风格：GPU 显存统一调度，CUDA 内核级多 LoRA 融合"
    preload:
      strategy: "gpu_lru"
      warm_fraction: 0.7
    latency_model:
      base_ttft_ms: 65
      lora_load_overhead_ms: 20
      tpot_ms: 10.0
```

### Q7：如何确认实验使用了真实 GPU 推理

运行时检查输出头部：

```
  推理模式: 真实 GPU 推理（vLLM）          ← 真实推理
  推理模式: Mock 推理引擎（标定延迟模型）  ← Mock 模式
```

也可以检查 JSON 结果：
```json
"元数据": {
  "cuda_available": true,
  "vllm_available": true,
  ...
}
```

---

## 附录 A：论文主表复现（E3）

**当前仓库默认入口**（复现 `configs/experiments.yaml` 当前默认 profile）：

```bash
conda activate LLM_vllm0102
cd /home/qhq/serverless_llm_experiment
python scripts/run_all_experiments.py --config configs/experiments.yaml
```

- **配置文件**：`configs/experiments.yaml`（通过 `profile_selection` 和各类 profile 选择当前主线组合）。
- **结果文件**：按 `backend / mode / adapters / requests / concurrency / results_tag` 落盘到 `results/` 目录，文件名取决于当前 profile 与环境变量覆盖。
- **环境**：当前主线稳定环境为 `LLM_vllm0102`，详见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。

**当前下一步（Mistral-Nemo + TP=2 + PEFT+finetune + 500 adapters）**：

```bash
conda activate LLM_vllm0102
cd /home/qhq/serverless_llm_experiment

python scripts/download_model.py \
  --model mistralai/Mistral-Nemo-Instruct-2407

FAASLORA_PROFILE_MODEL=mistral_nemo_12b_tp2 \
FAASLORA_PROFILE_DATASET=azure_sharegpt_rep1000 \
FAASLORA_PROFILE_WORKLOAD=mistral_nemo_12b_tp2_main \
python scripts/run_all_experiments.py --config configs/experiments.yaml --scenario faaslora_full
```

> 当前建议正式论文实验使用 `lora_adapters.preparation_mode = two_phase`：先单独生成冻结工件池，再用实验命令纯复用。若只是快速验证，可以临时切回 `bootstrap_once` 或 `one_shot`。

> 说明：`realistic_v2` 目前仅表示“内部异构生成模板”，不等同于当前正式论文使用的 `V2 publicmix` 路线。当前已确定的正式 `V2` 方案是：
> - `Qwen 7B / Mistral 7B`：优先下载公开 adapter，先验证本地兼容性，再纳入正式冻结工件池
> - `Qwen 14B / Mistral-Nemo`：先下载现有公开 adapter，再按统一规则补齐到 `500`
> - 同一 `base model` 下，不同系统对比必须严格复用同一批工件、同一份 workload、同一套 Zipf/热点轮换参数与 seed
> - 当前 `publicmix` 验证器会主动拒绝当前 vLLM 运行时不支持的公开工件，例如 `use_dora=true`

建议先用 `scripts/prepare_publicmix_pool.py` 做离线验收与建库清单：

```bash
cd /home/qhq/serverless_llm_experiment

python scripts/prepare_publicmix_pool.py validate \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --source-dir artifacts/public_candidates/mistral_7b \
  --output-json configs/generated/publicmix/mistral_7b_publicmix_validation.json

python scripts/prepare_publicmix_pool.py plan \
  --validated-report configs/generated/publicmix/mistral_7b_publicmix_validation.json \
  --output-json configs/generated/publicmix/mistral_7b_publicmix_manifest.json \
  --target-count 500 \
  --topup-profile realistic_v2 \
  --topup-seed 42

# 第三步：按 manifest 冻结成正式可复用的 V2 工件池
python scripts/prepare_publicmix_pool.py build \
  --manifest-json configs/generated/publicmix/mistral_7b_publicmix_manifest.json \
  --output-dir artifacts/frozen/mistral_7b_a500_v2_publicmix \
  --generation-mode peft_finetune \
  --python-bin /home/qhq/anaconda3/envs/LLM_vllm0102/bin/python
```

> `build` 会先复制已验通过的公开 adapter，再只对 `generated_fill` 缺口按 manifest 中记录的 `topup_profile + seed` 调用生成器补齐。这样冻结后的 `V2 publicmix` 目录可直接供正式对比实验复用，而不需要在实验启动时重新拼池。当前 `two_phase` 启动链路也已经能够自动修复冻结工件池在归档/恢复后留下的坏 `config.json / generation_config.json` 软链接，不需要再手工重拷支持文件；同时默认 runtime 不再全局启用 FlashInfer sampler，`Mistral` adapter/cache 目录会自动补齐 tokenizer 支持文件，从而尽量避免 fallback warning。

---

## 附录 B：实验重现清单

在提交论文前，建议完成以下检查：

- [ ] vLLM 已安装并验证（`vllm --version`）
- [ ] 真实 GPU 确认（`nvidia-smi`）
- [ ] 模型下载完整（`models/` 目录非空）
- [ ] Azure 追踪数据加载成功（`python scripts/download_datasets.py --verify`）
- [ ] ShareGPT 数据集已下载（若未下载，将使用 embedded fallback）
- [ ] LoRA 适配器已生成并冻结（当前模型专属 `artifacts/frozen/<model>_a500_v1/` 或后续 `v2_publicmix/` 非空，且与当前模型匹配）
- [ ] `configs/experiments.yaml` 中 `model.name` 指向本地路径
- [ ] 完整实验运行成功（非 `--quick` 模式）
- [ ] `results/experiment_results.json` 显示 `"cuda_available": true`
- [ ] 贡献3 表格中 `faaslora_full` 的 `Defer > 0` 且 `Contention = 0`
