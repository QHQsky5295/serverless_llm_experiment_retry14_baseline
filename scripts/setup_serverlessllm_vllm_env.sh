#!/usr/bin/env bash
set -euo pipefail

REPO="${SLLM_REPO_ROOT:-/home/qhq/serverless_llm_baselines/repos/ServerlessLLM}"
SOURCE_ENV="${SLLM_VLLM_SOURCE_ENV:-LLM_vllm0102}"
TARGET_ENV="${SLLM_VLLM_ENV_NAME:-${SLLM_VLLM_ENV:-sllm_vllm0102_official}}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda not found" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
TARGET_PREFIX="${CONDA_BASE}/envs/${TARGET_ENV}"

echo "[1/4] Preparing dedicated ServerlessLLM vLLM worker env: ${TARGET_ENV}"
if [[ ! -d "${TARGET_PREFIX}" ]]; then
  conda create -y -n "${TARGET_ENV}" --clone "${SOURCE_ENV}"
else
  echo "[info] Reusing existing env: ${TARGET_ENV}"
fi

echo "[2/4] Installing ServerlessLLM worker package into the dedicated env"
env PYTHONNOUSERSITE=1 conda run -n "${TARGET_ENV}" pip install --no-deps "${REPO}"

echo "[3/4] Import smoke"
env PYTHONNOUSERSITE=1 conda run -n "${TARGET_ENV}" python - <<'PY'
import importlib

for name in [
    "sllm",
    "ray",
    "torch",
    "transformers",
    "peft",
    "vllm",
    "vllm.lora.request",
]:
    importlib.import_module(name)
print("serverlessllm_vllm_env_import_ok")
PY

echo "[4/4] Repo-source backend import smoke"
env PYTHONNOUSERSITE=1 conda run -n "${TARGET_ENV}" bash -lc "PYTHONPATH='${REPO}' python - <<'PY'
import importlib
importlib.import_module('sllm.backends.vllm_backend')
print('serverlessllm_vllm_backend_import_ok')
PY"

cat <<EOF

ServerlessLLM vLLM worker env is ready.
Env name: ${TARGET_ENV}

This env is isolated from your main project envs and is safe to use for baseline worker runs.
Head/API env remains sllm_head_official and store env remains sllm_worker_official.
EOF
