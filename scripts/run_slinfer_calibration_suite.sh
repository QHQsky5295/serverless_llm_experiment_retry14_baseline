#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SLINFER_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
RELAY_ROOT="${RELAY_ROOT:-/home/qhq/relayserve_serverless_llm}"
RUN_PREFIX="${SLINFER_CALIBRATION_PREFIX:-20260614_slinfer_calibration512_v1}"
PYTHON="${SLINFER_ENV_DIR:-/home/qhq/anaconda3/envs/slinfer_official_20260612}/bin/python"
OUTPUT_DIR="${ROOT_DIR}/paper_artifacts/relayserve_v4"

export SLINFER_TRACE_ROLE="calibration512_rate100_disjoint"
export SLINFER_AUTO_STOP_STACK=1
export SLINFER_STORE_MEM_POOL_SIZE_GB="${SLINFER_STORE_MEM_POOL_SIZE_GB:-20}"
export SLINFER_MIN_AVAILABLE_MEMORY_GB="${SLINFER_MIN_AVAILABLE_MEMORY_GB:-32}"
export SLINFER_READY_TIMEOUT_S="${SLINFER_READY_TIMEOUT_S:-300}"
export SLINFER_ALLOW_FAILED_REQUESTS=1
export PYTHONUNBUFFERED=1

run_model() {
  local model_key="$1"
  local trace_path="$2"
  local ttft_slo_ms="$3"
  local tpot_slo_ms="$4"
  local candidates=()

  export SLINFER_TRACE_PATH_OVERRIDE="${trace_path}"
  for keep_alive_s in 1 10 30 60; do
    local tag="${RUN_PREFIX}_${model_key}_k${keep_alive_s}"
    local run_dir="${ROOT_DIR}/results/relayserve_continuation/slinfer/${tag}"
    if [[ -e "${run_dir}" ]]; then
      echo "refusing to overwrite calibration run: ${run_dir}" >&2
      exit 3
    fi
    export SLINFER_KEEP_ALIVE_S="${keep_alive_s}"
    bash "${ROOT_DIR}/scripts/run_slinfer_relayserve_continuation.sh" \
      "${model_key}" 0 "${tag}"
    candidates+=(--candidate "${keep_alive_s}=${run_dir}")
  done

  local table="${OUTPUT_DIR}/slinfer_${model_key}_calibration512.csv"
  "${PYTHON}" "${ROOT_DIR}/scripts/build_slinfer_calibration_table.py" \
    --model-key "${model_key}" \
    --ttft-slo-ms "${ttft_slo_ms}" \
    --tpot-slo-ms "${tpot_slo_ms}" \
    "${candidates[@]}" \
    --output "${table}"
  "${PYTHON}" "${ROOT_DIR}/scripts/verify_slinfer_calibration.py" "${table}"
}

mkdir -p "${OUTPUT_DIR}"

run_model \
  3b \
  "${RELAY_ROOT}/data_imports/traces/multi_turn_continuation/llama32_3b/llama32_3b_burstgptv2_chronological_sharegpt_multi_turn_calibration512_rate100_v4_trace.json" \
  180 \
  14

run_model \
  7b \
  "${RELAY_ROOT}/data_imports/traces/multi_turn_continuation/llama2_7b/llama2_7b_burstgptv2_chronological_sharegpt_multi_turn_calibration512_rate100_v4_trace.json" \
  440 \
  32

echo "SLINFER calibration suite completed and verified."
