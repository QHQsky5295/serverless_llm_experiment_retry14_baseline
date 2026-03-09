# FaaSLoRA 实验环境说明（RTX 3090 + vLLM LoRA）

本文档给出**经联网查证、可用于 RTX 3090 上 FaaSLoRA 完备实验**的环境组合与**完整重装**步骤。

---

## 1. 推荐环境组合（已验证可用）

| 组件 | 版本 | 说明 |
|------|------|------|
| **Python** | 3.11 | vLLM 0.10.2 支持 3.9–3.12，推荐 3.11 以兼容 PyTorch 2.4 |
| **PyTorch** | 2.4.x + **CUDA 12.4**（cu124） | 从 PyTorch 官方 cu124 源安装，适配 RTX 3090（Ampere） |
| **vLLM** | **0.10.2** | 项目固定版本；0.10.x 在消费级 GPU 上 LoRA 更稳定，避免 EngineCore 崩溃 |
| **CUDA 驱动** | 主机已安装 NVIDIA 驱动（≥12.x 推荐） | 与 PyTorch 自带 CUDA runtime 配合即可，无需单独装完整 CUDA Toolkit |

**为何不用 vLLM 0.16 + PyTorch 2.9+cu128？**  
在 RTX 3090 上使用 vLLM 0.16 做 LoRA 推理时，多次出现「首次 LoRA 请求后 EngineCore 子进程退出」；社区亦有类似反馈。cu128 主要面向 Blackwell 等新架构，对 3090 非必需。因此本项目采用 **vLLM 0.10.2 + PyTorch 2.4 + cu124** 作为推荐组合。

---

## 2. 完整重装步骤（推荐使用脚本）

**前提**：已安装 Conda（Miniconda/Anaconda），且本机有 NVIDIA GPU 与对应驱动。新建的 conda 环境名为 **LLM**。

### 方式 A：一键重装（推荐）

```bash
cd /path/to/serverless_llm_experiment
bash scripts/reinstall_env.sh
```

脚本会：

1. 删除已有 `LLM` 环境（若存在）
2. 新建 `LLM`，Python 3.11
3. 先安装 **PyTorch 2.4 + cu124**（官方 index-url）
4. 再安装 **vLLM 0.10.2**
5. 安装本项目及其余依赖（`pip install -e .`）
6. 自检并生成合成 LoRA 适配器

### 方式 B：手动重装

```bash
# 1. 删除旧环境
conda deactivate
conda env remove -n LLM -y

# 2. 新建环境（Python 3.11）
conda create -n LLM python=3.11 -y
conda activate LLM

# 3. 安装 PyTorch 2.4 + CUDA 12.4（必须先于 vLLM）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. 安装 vLLM（固定版本，避免 0.16 在 3090 LoRA 上崩溃）
pip install vllm==0.10.2

# 5. 进入项目并安装本项目及依赖
cd /path/to/serverless_llm_experiment
pip install -e .

# 6. 验证
python -c "
import torch
import vllm
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('vLLM:', vllm.__version__)
"
```

### 若出现「已杀死」或进程被 Kill

创建环境时若在 “Collecting package metadata” 阶段报错 **已杀死**（或进程被 kill），多为**内存不足**（OOM）。可尝试：

1. **先安装 libmamba 求解器**（更省内存、更快），再重跑脚本：
   ```bash
   conda install -n base conda-libmamba-solver -y
   bash scripts/reinstall_env.sh
   ```
   脚本会优先使用 `--solver=libmamba` 创建环境。

2. **关闭其他占内存的程序**，或换到内存更大的机器上执行。

3. **改用手动创建**：先 `conda create -n LLM python=3.11 -y --solver=libmamba`，再按「方式 B」后续步骤用 pip 安装 PyTorch 与 vLLM。

---

## 3. 验证环境

在激活 `LLM` 后执行：

```bash
python -c "
import torch
import vllm
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
print('vLLM:', vllm.__version__)
"
```

期望：`PyTorch: 2.4.x+cu124`、`CUDA available: True`、`vLLM: 0.10.2`。

---

## 4. 与项目配置的对应关系

- **pyproject.toml** 中已固定 `vllm==0.10.2`，与本文档一致。
- **configs/experiments.yaml** 中 `backend: "vllm"` 时，将使用上述 vLLM 环境。
- 若在 3090 上仍遇 vLLM LoRA 崩溃，可参考 [docs/VLLM_RTX3090_LORA.md](VLLM_RTX3090_LORA.md)（如增大 `/dev/shm`、改用 `backend: "transformers"` 等）。

---

## 5. 参考来源

- vLLM 0.10.2 安装与 GPU 要求：<https://docs.vllm.ai/en/v0.10.2/getting_started/installation/>
- PyTorch 2.4 + cu124 安装：<https://pytorch.org/get-started/locally/>
- 社区反馈：vLLM 0.10.x 在消费级 GPU 上 LoRA 更稳定；RTX 3090 多卡/EngineCore 崩溃见 GitHub issues #21339、#23517 等。
