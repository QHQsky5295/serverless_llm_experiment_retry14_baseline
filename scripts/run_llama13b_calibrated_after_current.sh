#!/usr/bin/env bash
set -euo pipefail

BASELINES_ROOT="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
CURRENT_RUN_TAG="${CURRENT_RUN_TAG:-llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1}"
CURRENT_QUEUE_ID="${CURRENT_QUEUE_ID:-20260507_llama13b_main_v2}"
CURRENT_ROUND_DIR="${CURRENT_ROUND_DIR:-${BASELINES_ROOT}/results/paper_experiments/03_main_comparison/${CURRENT_QUEUE_ID}_${CURRENT_RUN_TAG}}"
STATE_DIR="${CURRENT_ROUND_DIR}/state"
POLL_S="${POLL_S:-60}"

echo "[llama13b-calibrated-after-current] waiting for current PrimeLoRA marker: ${STATE_DIR}/50_faaslora.done"
while [[ ! -f "${STATE_DIR}/50_faaslora.done" ]]; do
  if [[ -f "${STATE_DIR}/50_faaslora.failed" ]]; then
    echo "[ERROR] current PrimeLoRA round failed; refusing to start calibrated round." >&2
    exit 1
  fi
  sleep "${POLL_S}"
done

echo "[llama13b-calibrated-after-current] current PrimeLoRA completed; waiting for GPUs to become idle."
while true; do
  active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l || true)"
  if [[ "${active}" == "0" ]]; then
    break
  fi
  echo "  GPUs still busy (${active} compute process entries); waiting ${POLL_S}s..."
  sleep "${POLL_S}"
done

PAPER_QUEUE_ID="${PAPER_QUEUE_ID:-20260507_llama13b_calibrated_s12_v1}" \
SLLM_RUN_TAG="${SLLM_RUN_TAG:-llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s12_mainv1}" \
SLLM_TIME_SCALE_FACTOR="${SLLM_TIME_SCALE_FACTOR:-12.0}" \
FAIR_ROUND_SYSTEMS="${FAIR_ROUND_SYSTEMS:-sglang serverlessllm vllm faaslora}" \
bash "${BASELINES_ROOT}/scripts/run_llama13b_calibrated_main_queue.sh"
