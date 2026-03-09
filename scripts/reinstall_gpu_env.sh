#!/usr/bin/env bash
set -e
ENV_NAME="${ENV_NAME:-LLM}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command -v conda >/dev/null || { echo "Need conda."; exit 1; }
source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null
conda info --envs | grep -q "^*.*$ENV_NAME" && conda deactivate 2>/dev/null || true
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
conda create -n "$ENV_NAME" python=3.11 -y
conda run -n "$ENV_NAME" pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
conda run -n "$ENV_NAME" pip install --quiet "vllm==0.10.2"
conda run -n "$ENV_NAME" pip install --quiet -e "$REPO_ROOT"
conda run -n "$ENV_NAME" python -c "import torch, vllm; print(torch.__version__, vllm.__version__)"
echo "Done. conda activate $ENV_NAME"
da deactivate 2>/dev/null || true

echo "[2/7] 删除已有环境..."
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true

echo "[3/7] 创建新环境 Python 3.11..."
conda create -n "$ENV_NAME" python=3.11 -y

echo "[4/7] 安装 PyTorch 2.4 + cu124..."
conda run -n "$ENV_NAME" pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "[5/7] 安装 vLLM 0.10.2..."
conda run -n "$ENV_NAME" pip install --quiet "vllm==0.10.2"

echo "[6/7] 安装 FaaSLoRA 及依赖..."
conda run -n "$ENV_NAME" pip install --quiet -e "$REPO_ROOT"

echo "[7/7] 自检..."
conda run -n "$ENV_NAME" python -c "import torch; import vllm; print('PyTorch', torch.__version__, 'vLLM', vllm.__version__, 'CUDA', torch.cuda.is_available())"
conda run -n "$ENV_NAME" python "$REPO_ROOT/scripts/generate_lora_adapters.py" --synthetic 2>/dev/null || true

echo ""
echo "  完成。激活: conda activate $ENV_NAME"
echo "  运行实验: python scripts/run_all_experiments.py --config configs/experiments.yaml"
echo "  说明: docs/ENVIRONMENT.md"
