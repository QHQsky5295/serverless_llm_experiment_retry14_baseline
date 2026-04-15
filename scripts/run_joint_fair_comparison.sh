#!/usr/bin/env bash
set -euo pipefail

BASELINES_ROOT="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
MAIN_REPO="${SLLM_MAIN_REPO:-/home/qhq/serverless_llm_experiment_retry14_baseline}"
PROJECT_ROOT="${BASELINES_ROOT}/ServerlessLLM_project"
SHARED_DIR="${BASELINES_ROOT}/results/shared_rounds"
COMPARE_DIR="${BASELINES_ROOT}/results/comparisons"

MODEL_PROFILE="${SLLM_MODEL_PROFILE:?SLLM_MODEL_PROFILE is required}"
DATASET_PROFILE="${SLLM_DATASET_PROFILE:?SLLM_DATASET_PROFILE is required}"
WORKLOAD_PROFILE="${SLLM_WORKLOAD_PROFILE:?SLLM_WORKLOAD_PROFILE is required}"
TOTAL_REQUESTS="${SLLM_TOTAL_REQUESTS:?SLLM_TOTAL_REQUESTS is required}"
SELECTED_NUM_ADAPTERS="${SLLM_SELECTED_NUM_ADAPTERS:?SLLM_SELECTED_NUM_ADAPTERS is required}"
SAMPLING_SEED="${SLLM_SAMPLING_SEED:-42}"
RUN_TAG="${SLLM_RUN_TAG:-${MODEL_PROFILE}_r${TOTAL_REQUESTS}_a${SELECTED_NUM_ADAPTERS}_seed${SAMPLING_SEED}}"
WORKER_GPUS="${SLLM_WORKER_GPUS:-0,1,2,3}"

TRACE_PATH="${SLLM_SHARED_TRACE_PATH:-${SHARED_DIR}/${RUN_TAG}_trace.json}"
ADAPTER_SUBSET_PATH="${SLLM_SHARED_ADAPTER_SUBSET_PATH:-${SHARED_DIR}/${RUN_TAG}_adapter_subset.json}"
COMPARE_PATH="${COMPARE_DIR}/${RUN_TAG}_faaslora_vs_serverlessllm.json}"

mkdir -p "${COMPARE_DIR}"

if [[ ! -f "${TRACE_PATH}" || ! -f "${ADAPTER_SUBSET_PATH}" ]]; then
  (
    cd "${PROJECT_ROOT}"
    SLLM_MODEL_PROFILE="${MODEL_PROFILE}" \
    SLLM_DATASET_PROFILE="${DATASET_PROFILE}" \
    SLLM_WORKLOAD_PROFILE="${WORKLOAD_PROFILE}" \
    SLLM_TOTAL_REQUESTS="${TOTAL_REQUESTS}" \
    SLLM_SELECTED_NUM_ADAPTERS="${SELECTED_NUM_ADAPTERS}" \
    SLLM_SAMPLING_SEED="${SAMPLING_SEED}" \
    SLLM_RUN_TAG="${RUN_TAG}" \
    bash scripts/prepare_shared_round_artifacts.sh
  )
fi

(
  cd "${MAIN_REPO}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  FAASLORA_RESULTS_TAG="${RUN_TAG}_faaslora" \
  FAASLORA_PROFILE_MODEL="${MODEL_PROFILE}" \
  FAASLORA_PROFILE_DATASET="${DATASET_PROFILE}" \
  FAASLORA_PROFILE_WORKLOAD="${WORKLOAD_PROFILE}" \
  FAASLORA_SHARED_TRACE_PATH="${TRACE_PATH}" \
  FAASLORA_SHARED_ADAPTER_SUBSET_PATH="${ADAPTER_SUBSET_PATH}" \
  bash scripts/run_faaslora_shared_artifact_experiment.sh
)

FAASLORA_RESULT="$(find "${MAIN_REPO}/results" -maxdepth 3 -type f -name "*${RUN_TAG}_faaslora*.json" | sort | tail -n 1)"
if [[ -z "${FAASLORA_RESULT}" ]]; then
  echo "[ERROR] unable to locate FaasLoRA result JSON for ${RUN_TAG}_faaslora" >&2
  exit 1
fi

(
  cd "${PROJECT_ROOT}"
  SLLM_MODEL_PROFILE="${MODEL_PROFILE}" \
  SLLM_DATASET_PROFILE="${DATASET_PROFILE}" \
  SLLM_WORKLOAD_PROFILE="${WORKLOAD_PROFILE}" \
  SLLM_TOTAL_REQUESTS="${TOTAL_REQUESTS}" \
  SLLM_SELECTED_NUM_ADAPTERS="${SELECTED_NUM_ADAPTERS}" \
  SLLM_SAMPLING_SEED="${SAMPLING_SEED}" \
  SLLM_RUN_TAG="${RUN_TAG}_serverlessllm" \
  SLLM_SHARED_TRACE_PATH="${TRACE_PATH}" \
  SLLM_SHARED_ADAPTER_SUBSET_PATH="${ADAPTER_SUBSET_PATH}" \
  SLLM_WORKER_GPUS="${WORKER_GPUS}" \
  bash scripts/run_serverlessllm_fair_experiment.sh
)

SLLM_SUMMARY="${BASELINES_ROOT}/results/replay/${RUN_TAG}_serverlessllm_summary.json"
if [[ ! -f "${SLLM_SUMMARY}" ]]; then
  echo "[ERROR] missing ServerlessLLM summary JSON: ${SLLM_SUMMARY}" >&2
  exit 1
fi

python3 "${BASELINES_ROOT}/scripts/compare_fair_results.py" \
  --result "${FAASLORA_RESULT}" \
  --result "${SLLM_SUMMARY}" \
  --output "${COMPARE_PATH}"

echo "faaslora_result -> ${FAASLORA_RESULT}"
echo "serverlessllm_summary -> ${SLLM_SUMMARY}"
echo "comparison -> ${COMPARE_PATH}"
