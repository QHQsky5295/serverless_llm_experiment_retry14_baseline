#!/usr/bin/env bash
set -euo pipefail

BASELINES_ROOT="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
RUN_TAG="${SLLM_RUN_TAG:-llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1}"
QUEUE_ID="${PAPER_QUEUE_ID:-20260507_llama13b_main_v2}"
ROUND_DIR="${FAIR_ROUND_DIR:-${BASELINES_ROOT}/results/paper_experiments/03_main_comparison/${QUEUE_ID}_${RUN_TAG}}"
STATE_DIR="${ROUND_DIR}/state"
POLL_S="${POLL_S:-60}"

echo "[llama13b-slora-after-faas] queue_id=${QUEUE_ID}"
echo "[llama13b-slora-after-faas] round_dir=${ROUND_DIR}"
echo "[llama13b-slora-after-faas] waiting for PrimeLoRA marker: ${STATE_DIR}/50_faaslora.done"

if [[ -f "${STATE_DIR}/40_slora.done" ]]; then
  echo "[llama13b-slora-after-faas] S-LoRA marker already exists; nothing to do."
  exit 0
fi

while [[ ! -f "${STATE_DIR}/50_faaslora.done" ]]; do
  if [[ -f "${STATE_DIR}/50_faaslora.failed" ]]; then
    echo "[ERROR] PrimeLoRA failed in this round; refusing to start S-LoRA." >&2
    exit 1
  fi
  sleep "${POLL_S}"
done

echo "[llama13b-slora-after-faas] PrimeLoRA completed; waiting for GPUs to become idle."
while true; do
  active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l || true)"
  if [[ "${active}" == "0" ]]; then
    break
  fi
  echo "  GPUs still busy (${active} compute process entries); waiting ${POLL_S}s..."
  sleep "${POLL_S}"
done

echo "[llama13b-slora-after-faas] starting S-LoRA Llama-13B diagnostic run."
echo "  Note: TP>1 uses the public-code-safe BMM path; see S-LoRA_project/docs/SLoRA_REPRO_PLAN.md."
echo "  Default topology is DP1/TP4 on 4 GPUs to avoid duplicating the 500-adapter host pool."

PAPER_QUEUE_ID="${QUEUE_ID}" \
SLLM_RUN_TAG="${RUN_TAG}" \
FAIR_ROUND_SYSTEMS=slora \
SLORA_TENSOR_PARALLEL_SIZE="${SLORA_TENSOR_PARALLEL_SIZE:-4}" \
SLORA_DATA_PARALLEL_REPLICAS="${SLORA_DATA_PARALLEL_REPLICAS:-1}" \
SLORA_USE_BMM="${SLORA_USE_BMM:-1}" \
SLORA_TIMEOUT_S="${SLORA_TIMEOUT_S:-21600}" \
bash "${BASELINES_ROOT}/scripts/run_llama13b_main_comparison_queue.sh"
