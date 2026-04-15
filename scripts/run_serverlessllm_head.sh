#!/usr/bin/env bash
set -euo pipefail

HEAD_ENV="${SLLM_HEAD_ENV:-sllm_head_official}"
RAY_PORT="${SLLM_RAY_PORT:-6389}"
RAY_NODE_IP="${SLLM_RAY_HEAD_HOST:-127.0.0.1}"
SLLM_HEAD_RESOURCES="${SLLM_HEAD_RESOURCES:-{\"control_node\": 1}}"
SLLM_REPO_ROOT="${SLLM_REPO_ROOT:-/home/qhq/serverless_llm_baselines/repos/ServerlessLLM}"
SLLM_EXTRA_PYTHONPATH="${SLLM_EXTRA_PYTHONPATH:-}"
PYTHONPATH_PREFIX="${SLLM_REPO_ROOT}${SLLM_EXTRA_PYTHONPATH:+:${SLLM_EXTRA_PYTHONPATH}}"

exec env PYTHONNOUSERSITE=1 \
  PYTHONPATH="${PYTHONPATH_PREFIX}" \
  conda run --no-capture-output -n "${HEAD_ENV}" \
  ray start --head --node-ip-address="${RAY_NODE_IP}" --port="${RAY_PORT}" --num-cpus=4 --num-gpus=0 \
  --resources="${SLLM_HEAD_RESOURCES}" --block
