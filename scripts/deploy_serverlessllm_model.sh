#!/usr/bin/env bash
set -euo pipefail

HEAD_ENV="${SLLM_HEAD_ENV:-sllm_head_official}"
WORKER_ENV="${SLLM_WORKER_ENV:-sllm_worker_official}"
MODEL_NAME="${1:-facebook/opt-125m}"
BACKEND="${SLLM_BACKEND:-transformers}"
SERVER_URL="${LLM_SERVER_URL:-http://127.0.0.1:8343}"
CONFIG_PATH="${SLLM_DEPLOY_CONFIG:-}"
SLLM_REPO_ROOT="${SLLM_REPO_ROOT:-/home/qhq/serverless_llm_baselines/repos/ServerlessLLM}"
SLLM_BASELINES_ROOT="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
MATERIALIZE_SCRIPT="${SLLM_MATERIALIZE_LORAS_SCRIPT:-${SLLM_BASELINES_ROOT}/scripts/materialize_serverlessllm_loras.py}"
SLLM_EXTRA_PYTHONPATH="${SLLM_EXTRA_PYTHONPATH:-}"
PYTHONPATH_PREFIX="${SLLM_REPO_ROOT}${SLLM_EXTRA_PYTHONPATH:+:${SLLM_EXTRA_PYTHONPATH}}"
WORKER_ENV_PYTHON="/home/qhq/anaconda3/envs/${WORKER_ENV}/bin/python"
HEAD_ENV_PYTHON="/home/qhq/anaconda3/envs/${HEAD_ENV}/bin/python"

export LLM_SERVER_URL="${SERVER_URL}"

if [[ ! -x "${WORKER_ENV_PYTHON}" ]]; then
  echo "[ERROR] worker env python not found: ${WORKER_ENV_PYTHON}" >&2
  exit 1
fi

if [[ ! -x "${HEAD_ENV_PYTHON}" ]]; then
  echo "[ERROR] head env python not found: ${HEAD_ENV_PYTHON}" >&2
  exit 1
fi

if [[ -z "${CONFIG_PATH}" && $# -ge 1 && -f "${1}" ]]; then
  CONFIG_PATH="${1}"
fi

if [[ -n "${CONFIG_PATH}" ]]; then
  env PYTHONNOUSERSITE=1 \
    "${WORKER_ENV_PYTHON}" "${MATERIALIZE_SCRIPT}" --config "${CONFIG_PATH}"
  exec env PYTHONNOUSERSITE=1 \
    PYTHONPATH="${PYTHONPATH_PREFIX}" \
    env LLM_SERVER_URL="${LLM_SERVER_URL}" \
    "${HEAD_ENV_PYTHON}" -m sllm.cli.clic deploy --config "${CONFIG_PATH}"
fi

exec env PYTHONNOUSERSITE=1 \
  PYTHONPATH="${PYTHONPATH_PREFIX}" \
  env LLM_SERVER_URL="${LLM_SERVER_URL}" \
  "${HEAD_ENV_PYTHON}" -m sllm.cli.clic deploy --model "${MODEL_NAME}" --backend "${BACKEND}"
