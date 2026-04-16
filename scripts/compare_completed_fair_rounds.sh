#!/usr/bin/env bash
set -euo pipefail

BASELINES_ROOT="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
MAIN_REPO="${SLLM_MAIN_REPO:-/home/qhq/serverless_llm_experiment_retry14_baseline}"
RESULTS_DIR="${MAIN_REPO}/results"
REPLAY_DIR="${BASELINES_ROOT}/results/replay"
COMPARE_DIR="${BASELINES_ROOT}/results/comparisons"

if REAL_RESULTS_DIR="$(readlink -f "${RESULTS_DIR}" 2>/dev/null)"; then
  RESULTS_DIR="${REAL_RESULTS_DIR}"
fi

FAASLORA_RESULTS_TAG="${FAASLORA_RESULTS_TAG:?FAASLORA_RESULTS_TAG is required}"
COMPARE_RUN_TAG="${COMPARE_RUN_TAG:-${FAASLORA_RESULTS_TAG}}"
FAASLORA_RESULT_PATH="${FAASLORA_RESULT_PATH:-}"
SLLM_RUN_TAG="${SLLM_RUN_TAG:-}"
SGLANG_RUN_TAG="${SGLANG_RUN_TAG:-}"
PUNICA_RUN_TAG="${PUNICA_RUN_TAG:-}"

mkdir -p "${COMPARE_DIR}"

if [[ -n "${FAASLORA_RESULT_PATH}" ]]; then
  FAASLORA_RESULT="${FAASLORA_RESULT_PATH}"
else
  FAASLORA_RESULT="$(find "${RESULTS_DIR}" -maxdepth 3 -type f -name "*${FAASLORA_RESULTS_TAG}*.json" | sort | tail -n 1)"
fi
if [[ -z "${FAASLORA_RESULT}" || ! -f "${FAASLORA_RESULT}" ]]; then
  echo "[ERROR] unable to find FaaSLoRA result matching tag: ${FAASLORA_RESULTS_TAG}" >&2
  exit 1
fi

RESULT_ARGS=(--result "${FAASLORA_RESULT}")

if [[ -n "${SLLM_RUN_TAG}" ]]; then
  SLLM_SUMMARY="${REPLAY_DIR}/${SLLM_RUN_TAG}_summary.json"
  if [[ ! -f "${SLLM_SUMMARY}" ]]; then
    echo "[ERROR] missing ServerlessLLM summary JSON: ${SLLM_SUMMARY}" >&2
    exit 1
  fi
  RESULT_ARGS+=(--result "${SLLM_SUMMARY}")
fi

if [[ -n "${SGLANG_RUN_TAG}" ]]; then
  SGLANG_SUMMARY="${REPLAY_DIR}/${SGLANG_RUN_TAG}_summary.json"
  if [[ ! -f "${SGLANG_SUMMARY}" ]]; then
    echo "[ERROR] missing SGLang summary JSON: ${SGLANG_SUMMARY}" >&2
    exit 1
  fi
  RESULT_ARGS+=(--result "${SGLANG_SUMMARY}")
fi

if [[ -n "${PUNICA_RUN_TAG}" ]]; then
  PUNICA_SUMMARY="${REPLAY_DIR}/${PUNICA_RUN_TAG}_punica_summary.json"
  if [[ ! -f "${PUNICA_SUMMARY}" ]]; then
    echo "[ERROR] missing Punica summary JSON: ${PUNICA_SUMMARY}" >&2
    exit 1
  fi
  RESULT_ARGS+=(--result "${PUNICA_SUMMARY}")
fi

COMPARE_PATH="${COMPARE_DIR}/${COMPARE_RUN_TAG}.json"
python3 "${BASELINES_ROOT}/scripts/compare_fair_results.py" \
  "${RESULT_ARGS[@]}" \
  --output "${COMPARE_PATH}"

echo "faaslora_result -> ${FAASLORA_RESULT}"
if [[ -n "${SLLM_RUN_TAG}" ]]; then
  echo "serverlessllm_summary -> ${SLLM_SUMMARY}"
fi
if [[ -n "${SGLANG_RUN_TAG}" ]]; then
  echo "sglang_summary -> ${SGLANG_SUMMARY}"
fi
if [[ -n "${PUNICA_RUN_TAG}" ]]; then
  echo "punica_summary -> ${PUNICA_SUMMARY}"
fi
echo "comparison -> ${COMPARE_PATH}"
