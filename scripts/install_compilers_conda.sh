#!/usr/bin/env bash
# =============================================================================
# Install C/C++ compiler toolchain inside the LLM conda env (for Triton/vLLM).
# 用法（在终端中执行）：
#   conda activate LLM
#   bash scripts/install_compilers_conda.sh
# 说明：
#   - 使用 conda-forge 提供的编译器包，避免依赖 apt 源（校园网常被重定向）。
#   - 安装完成后，Triton 在运行时即可找到 gcc/g++，不再报 "Failed to find C compiler"。
# =============================================================================
set -e

ENV_NAME="${CONDA_DEFAULT_ENV:-LLM}"
echo "Installing compiler toolchain into conda env: ${ENV_NAME}"

if [ -z "${CONDA_DEFAULT_ENV}" ]; then
  echo "[警告] 当前未激活任何 conda 环境。请先执行：conda activate LLM"
  exit 1
fi

echo ""
echo "[1/2] 安装编译器（gcc/g++ 等）..."
conda install -n "${ENV_NAME}" -c conda-forge compilers -y

echo ""
echo "[2/2] 确认 gcc 可用（供 Triton/vLLM 使用）..."
conda run -n "${ENV_NAME}" bash -c 'which gcc && gcc --version | head -n 1'

echo ""
echo "完成。之后在该环境中运行实验前，可显式指定："
echo "  export CC=\$(which gcc)"
echo "  python scripts/run_all_experiments.py --config configs/experiments.yaml"

