#!/usr/bin/env bash
set -euo pipefail

HEAD_ENV="${SLLM_HEAD_ENV:-sllm_head_official}"
HOST="${SLLM_HOST:-127.0.0.1}"
PORT="${SLLM_PORT:-8343}"
RAY_PORT="${SLLM_RAY_PORT:-6389}"
RAY_ADDRESS="${SLLM_RAY_ADDRESS:-127.0.0.1:${RAY_PORT}}"
RAY_NODE_IP="${SLLM_RAY_NODE_IP:-127.0.0.1}"
SLLM_REPO_ROOT="${SLLM_REPO_ROOT:-/home/qhq/serverless_llm_baselines/repos/ServerlessLLM}"
SLLM_EXTRA_PYTHONPATH="${SLLM_EXTRA_PYTHONPATH:-}"
PYTHONPATH_PREFIX="${SLLM_REPO_ROOT}${SLLM_EXTRA_PYTHONPATH:+:${SLLM_EXTRA_PYTHONPATH}}"

exec env PYTHONNOUSERSITE=1 \
  SLLM_RAY_ADDRESS="${RAY_ADDRESS}" \
  SLLM_RAY_NODE_IP="${RAY_NODE_IP}" \
  PYTHONPATH="${PYTHONPATH_PREFIX}" \
  conda run --no-capture-output -n "${HEAD_ENV}" \
  python -m sllm.cli.clic start --host "${HOST}" --port "${PORT}"
