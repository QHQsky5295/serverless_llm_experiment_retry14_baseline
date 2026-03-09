#!/usr/bin/env bash
# =============================================================================
# FaaSLoRA GPU 实验环境一键安装脚本
# 适用于：RTX 3090 × 2（或其他 NVIDIA GPU）
# 激活环境：conda activate LLM
# 运行方式：bash scripts/setup_gpu.sh
# =============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "================================================================"
echo "  FaaSLoRA GPU 实验环境安装"
echo "  工作目录: $REPO_ROOT"
echo "================================================================"

# 检查 conda 环境
if ! python -c "import torch" 2>/dev/null; then
    echo "[错误] 请先激活 LLM 环境：conda activate LLM"
    exit 1
fi

TORCH_VER=$(python -c "import torch; print(torch.__version__)")
echo "✓ PyTorch: $TORCH_VER"

# 检查 CUDA
CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_OK" = "True" ]; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEM=$(python -c "import torch; p=torch.cuda.get_device_properties(0); print(p.total_memory//1024**3)")
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
    echo "✓ CUDA 可用: ${GPU_COUNT}x ${GPU_NAME} (${GPU_MEM}GB each)"
else
    echo "[警告] CUDA 不可用，将运行 Mock 模式。如需真实推理请检查 GPU 驱动。"
fi

echo ""
echo "[1/4] 安装核心依赖..."
pip install pyyaml safetensors peft transformers huggingface_hub accelerate --quiet
echo "✓ 核心依赖安装完成"

echo ""
echo "[2/4] 安装 vLLM（用于真实 LLM 推理）..."
echo "  注意：vLLM 安装可能需要 5-10 分钟"
# vLLM 与 PyTorch CUDA 版本需匹配
pip install vllm --quiet && echo "✓ vLLM 安装成功" || echo "⚠ vLLM 安装失败，将使用 Mock 推理模式"

echo ""
echo "[3/4] 安装 FaaSLoRA 包..."
pip install -e "$REPO_ROOT" --quiet
echo "✓ FaaSLoRA 安装完成"

echo ""
echo "[4/4] 生成合成 LoRA 适配器..."
python scripts/generate_lora_adapters.py --synthetic
echo "✓ LoRA 适配器生成完成"

echo ""
echo "================================================================"
echo "  安装完成！"
echo "================================================================"
echo ""
echo "  下一步：下载推理模型（二选一）"
echo ""
echo "  方案A - 快速验证（Qwen2.5-0.5B，约 1 GB）："
echo "    python scripts/download_model.py --model Qwen/Qwen2.5-0.5B-Instruct"
echo ""
echo "  方案B - 论文级实验（Qwen2.5-7B，约 15 GB，推荐用于 RTX 3090）："
echo "    python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct"
echo ""
echo "  下载后运行实验："
echo "    python scripts/run_all_experiments.py --config configs/experiments.yaml"
echo ""
