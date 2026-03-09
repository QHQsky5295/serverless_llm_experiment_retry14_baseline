#!/usr/bin/env bash
# =============================================================================
# FaaSLoRA GPU 环境完整重装（RTX 3090 + vLLM 0.10.2 + PyTorch 2.4 cu124）
# 用法: bash scripts/reinstall_env.sh
# 每步若已完成则跳过；强制完整重装: FORCE_REINSTALL=1 bash scripts/reinstall_env.sh
# 详见: docs/ENVIRONMENT.md
# =============================================================================
set -e

ENV_NAME="${ENV_NAME:-LLM}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

_env_exists() {
    conda info --envs 2>/dev/null | awk '{print $1}' | grep -qx "$ENV_NAME"
}

if ! command -v conda &>/dev/null; then
    echo "[错误] 未找到 conda，请先安装 Miniconda 或 Anaconda。"
    exit 1
fi
[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ] && source "$HOME/anaconda3/etc/profile.d/conda.sh"
[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ] && source "$HOME/miniconda3/etc/profile.d/conda.sh"

echo "================================================================"
echo "  FaaSLoRA GPU 环境完整重装 - $ENV_NAME（已完成步骤将跳过）"
echo "  项目根目录: $REPO_ROOT"
echo "================================================================"

CURRENT=$(conda info --envs 2>/dev/null | grep "^*" | awk '{print $1}' || echo "")
if [ -n "$CURRENT" ] && [ "$CURRENT" = "$ENV_NAME" ]; then
    echo "[1/7] 当前处于 $ENV_NAME，先退出..."
    conda deactivate 2>/dev/null || true
fi

echo ""
echo "[2/7] 删除已有环境 $ENV_NAME（若存在）..."
if [ -n "${FORCE_REINSTALL}" ] && [ "${FORCE_REINSTALL}" != "0" ] && _env_exists; then
    conda env remove -n "$ENV_NAME" -y
elif _env_exists; then
    echo "  跳过（环境已存在，如需强制重装请: FORCE_REINSTALL=1 $0）"
else
    echo "  跳过（环境不存在）"
fi

echo ""
echo "[3/7] 创建新环境 $ENV_NAME（Python 3.11）..."
if _env_exists; then
    echo "  跳过（环境已存在）"
else
    # 优先用 libmamba 求解器（更省内存、更快；若出现「已杀死」多为内存不足，可先关掉其他程序或改用: conda install -n base conda-libmamba-solver）
    if conda create -n "$ENV_NAME" python=3.11 -y --solver=libmamba 2>/dev/null; then
        :
    else
        conda create -n "$ENV_NAME" python=3.11 -y
    fi
fi

echo ""
echo "[4/7] 安装 PyTorch 2.4 + CUDA 12.4（cu124）..."
if conda run -n "$ENV_NAME" python -c 'import torch; exit(0 if "cu124" in torch.__version__ else 1)' 2>/dev/null; then
    echo "  跳过（已安装 PyTorch cu124）"
else
    PYTORCH_INDEX="https://download.pytorch.org/whl/cu124"
    echo "  [4a] 安装 torch..."
    conda run -n "$ENV_NAME" pip install torch --index-url "$PYTORCH_INDEX"
    echo "  [4b] 安装 torchvision..."
    conda run -n "$ENV_NAME" pip install torchvision --index-url "$PYTORCH_INDEX"
    echo "  [4c] 安装 torchaudio..."
    conda run -n "$ENV_NAME" pip install torchaudio --index-url "$PYTORCH_INDEX"
fi

echo ""
echo "[5/7] 安装 vLLM 0.10.2（项目固定版本）..."
if conda run -n "$ENV_NAME" python -c 'import vllm; exit(0 if vllm.__version__ == "0.10.2" else 1)' 2>/dev/null; then
    echo "  跳过（已安装 vLLM 0.10.2）"
else
    conda run -n "$ENV_NAME" pip install "vllm==0.10.2"
fi

echo ""
echo "[6/7] 安装 FaaSLoRA 项目及依赖..."
if conda run -n "$ENV_NAME" pip show faaslora &>/dev/null; then
    echo "  跳过（已安装 FaaSLoRA）"
else
    conda run -n "$ENV_NAME" pip install -e "$REPO_ROOT"
fi

echo ""
echo "[7/7] 自检与生成合成 LoRA..."
conda run -n "$ENV_NAME" python -c 'import torch, vllm; print("PyTorch", torch.__version__, "vLLM", vllm.__version__, "CUDA", torch.cuda.is_available())'
if [ -d "$REPO_ROOT/lora_adapters" ] && [ -n "$(ls -A "$REPO_ROOT/lora_adapters" 2>/dev/null)" ]; then
    echo "  跳过（lora_adapters 已存在且非空）"
else
    conda run -n "$ENV_NAME" python "$REPO_ROOT/scripts/generate_lora_adapters.py" --synthetic 2>/dev/null || true
fi

echo ""
echo "================================================================"
echo "  重装完成！"
echo "================================================================"
echo ""
echo "  激活环境: conda activate $ENV_NAME"
echo "  运行实验: python scripts/run_all_experiments.py --config configs/experiments.yaml"
echo "  环境说明: docs/ENVIRONMENT.md"
echo ""
