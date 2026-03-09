#!/usr/bin/env bash
# =============================================================================
# FaaSLoRA Environment Setup Script
# =============================================================================
# 用法：
#   conda activate LLM
#   bash scripts/setup_environment.sh
# =============================================================================
set -e

ENV_NAME="${1:-LLM}"
echo "========================================================"
echo "  FaaSLoRA Environment Setup"
echo "  Target conda env: $ENV_NAME"
echo "========================================================"

# Check that we are in the right environment
CURRENT_ENV=$(conda info --envs | grep "^*" | awk '{print $1}' || echo "unknown")
echo "  Current env: $CURRENT_ENV"

# ---------------------------------------------------------------------------
# 1. Core inference stack
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Installing vLLM ..."
pip install vllm --quiet

echo ""
echo "[2/5] Installing PEFT + Transformers ..."
pip install peft transformers safetensors accelerate --quiet

# ---------------------------------------------------------------------------
# 2. Dataset dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Installing dataset libraries ..."
pip install datasets huggingface_hub --quiet

# ---------------------------------------------------------------------------
# 3. Experiment & monitoring dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Installing experiment dependencies ..."
pip install pyyaml aiohttp prometheus_client redis boto3 s3fs pynvml --quiet

# ---------------------------------------------------------------------------
# 4. Install the project itself
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Installing FaaSLoRA project ..."
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pip install -e "$REPO_ROOT" --quiet

# ---------------------------------------------------------------------------
# 5. Quick smoke test
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo "  Running quick smoke test ..."
python -c "
import torch
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA     : {torch.cuda.is_available()}')

try:
    import vllm
    print(f'  vLLM     : {vllm.__version__}')
except ImportError:
    print('  vLLM     : NOT INSTALLED')

try:
    import peft
    print(f'  PEFT     : {peft.__version__}')
except ImportError:
    print('  PEFT     : NOT INSTALLED')

try:
    import transformers
    print(f'  Transformers: {transformers.__version__}')
except ImportError:
    print('  Transformers: NOT INSTALLED')

try:
    import datasets
    print(f'  Datasets : {datasets.__version__}')
except ImportError:
    print('  Datasets : NOT INSTALLED')

print('  Smoke test: OK')
"

# ---------------------------------------------------------------------------
# 6. Generate LoRA adapters
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo "  Generating LoRA adapter files ..."
echo "  (synthetic mode – no model download required)"
echo ""
cd "$REPO_ROOT"
python scripts/generate_lora_adapters.py --synthetic

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo "  Setup complete!"
echo ""
echo "  To run the experiments:"
echo "    conda activate $ENV_NAME"
echo "    cd $REPO_ROOT"
echo "    python scripts/run_all_experiments.py --config configs/experiments.yaml"
echo ""
echo "  To use a real GPU model (optional, ~1 GB download):"
echo "    python scripts/generate_lora_adapters.py --use-peft --model Qwen/Qwen2.5-0.5B-Instruct"
echo "========================================================"
