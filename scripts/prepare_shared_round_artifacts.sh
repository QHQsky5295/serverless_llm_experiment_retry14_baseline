#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
MAIN_REPO="${SLLM_MAIN_REPO:-/home/qhq/serverless_llm_experiment_retry14_baseline}"
OUTPUT_DIR="${SLLM_SHARED_ROUND_DIR:-${ROOT_DIR}/results/shared_rounds}"

MODEL_PROFILE="${SLLM_MODEL_PROFILE:?SLLM_MODEL_PROFILE is required}"
DATASET_PROFILE="${SLLM_DATASET_PROFILE:?SLLM_DATASET_PROFILE is required}"
WORKLOAD_PROFILE="${SLLM_WORKLOAD_PROFILE:?SLLM_WORKLOAD_PROFILE is required}"
TOTAL_REQUESTS="${SLLM_TOTAL_REQUESTS:?SLLM_TOTAL_REQUESTS is required}"
SELECTED_NUM_ADAPTERS="${SLLM_SELECTED_NUM_ADAPTERS:?SLLM_SELECTED_NUM_ADAPTERS is required}"
SAMPLING_SEED="${SLLM_SAMPLING_SEED:-42}"
SERVING_MODEL_NAME="${SLLM_SERVING_MODEL_NAME:-${MODEL_PROFILE}}"
STORAGE_REMOTE_DIR_OVERRIDE="${SLLM_STORAGE_REMOTE_DIR_OVERRIDE:-}"

RUN_TAG="${SLLM_RUN_TAG:-${MODEL_PROFILE}_r${TOTAL_REQUESTS}_a${SELECTED_NUM_ADAPTERS}_seed${SAMPLING_SEED}}"
TRACE_PATH="${OUTPUT_DIR}/${RUN_TAG}_trace.json"
ADAPTER_SUBSET_PATH="${OUTPUT_DIR}/${RUN_TAG}_adapter_subset.json"

mkdir -p "${OUTPUT_DIR}"

echo "[shared] Exporting authoritative trace + adapter subset from FaaSLoRA frozen pool"
CMD=(
  env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
  conda run --no-capture-output -n sllm_head_official
  python "${ROOT_DIR}/scripts/export_shared_faaslora_trace.py"
  --main-repo "${MAIN_REPO}"
  --model-profile "${MODEL_PROFILE}"
  --dataset-profile "${DATASET_PROFILE}"
  --workload-profile "${WORKLOAD_PROFILE}"
  --selected-num-adapters "${SELECTED_NUM_ADAPTERS}"
  --total-requests "${TOTAL_REQUESTS}"
  --seed "${SAMPLING_SEED}"
  --serving-model-name "${SERVING_MODEL_NAME}"
  --output "${TRACE_PATH}"
  --adapter-subset-output "${ADAPTER_SUBSET_PATH}"
)
if [[ -n "${STORAGE_REMOTE_DIR_OVERRIDE}" ]]; then
  CMD+=(--storage-remote-dir-override "${STORAGE_REMOTE_DIR_OVERRIDE}")
fi
"${CMD[@]}"

echo "trace          -> ${TRACE_PATH}"
echo "adapter_subset -> ${ADAPTER_SUBSET_PATH}"
