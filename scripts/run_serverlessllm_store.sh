#!/usr/bin/env bash
set -euo pipefail

STORE_ENV="${SLLM_STORE_ENV:-${SLLM_WORKER_ENV:-sllm_worker_official}}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
STORE_PATH="${SLLM_STORE_PATH:-/home/qhq/serverless_llm_baselines/models}"
STORE_LOG="${SLLM_STORE_LOG:-/tmp/serverlessllm_store_smoke.log}"
STORE_MEM_POOL_SIZE="${SLLM_STORE_MEM_POOL_SIZE:-32GB}"
STORE_NUM_THREAD="${SLLM_STORE_NUM_THREAD:-4}"
STORE_CHUNK_SIZE="${SLLM_STORE_CHUNK_SIZE:-32MB}"

mkdir -p "${STORE_PATH}"
mkdir -p "$(dirname "${STORE_LOG}")"

echo "[serverlessllm-store] logging to ${STORE_LOG}"
echo "[serverlessllm-store] mem_pool_size=${STORE_MEM_POOL_SIZE} num_thread=${STORE_NUM_THREAD} chunk_size=${STORE_CHUNK_SIZE}"

exec env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  STORAGE_PATH="${STORE_PATH}" \
  conda run --no-capture-output -n "${STORE_ENV}" \
  sllm-store start \
  --storage-path "${STORE_PATH}" \
  --num-thread "${STORE_NUM_THREAD}" \
  --chunk-size "${STORE_CHUNK_SIZE}" \
  --mem-pool-size "${STORE_MEM_POOL_SIZE}" 2>&1 | tee -a "${STORE_LOG}"
