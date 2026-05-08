#!/usr/bin/env bash
set -euo pipefail

BASELINES_ROOT="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
FORMAL_RUN_TAG="${FORMAL_RUN_TAG:-llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1}"
FORMAL_QUEUE_ID="${FORMAL_QUEUE_ID:-20260507_llama13b_main_v2}"
FORMAL_ROUND_DIR="${FORMAL_ROUND_DIR:-${BASELINES_ROOT}/results/paper_experiments/03_main_comparison/${FORMAL_QUEUE_ID}_${FORMAL_RUN_TAG}}"
STATE_DIR="${FORMAL_ROUND_DIR}/state"
POLL_S="${POLL_S:-60}"

PROBE_QUEUE_ID="${PROBE_QUEUE_ID:-20260507_llama13b_vllm_seq4_probe}"
PROBE_RUN_TAG="${PROBE_RUN_TAG:-llama2_13b_vllm_seq4_probe256_a500_seed42_s8}"

echo "[llama13b-vllm-seq4-probe] waiting for formal PrimeLoRA marker: ${STATE_DIR}/50_faaslora.done"
while [[ ! -f "${STATE_DIR}/50_faaslora.done" ]]; do
  if [[ -f "${STATE_DIR}/50_faaslora.failed" ]]; then
    echo "[ERROR] PrimeLoRA failed in the formal round; refusing to start the vLLM probe." >&2
    exit 1
  fi
  sleep "${POLL_S}"
done

echo "[llama13b-vllm-seq4-probe] PrimeLoRA completed; waiting for GPUs to become idle."
while true; do
  active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l || true)"
  if [[ "${active}" == "0" ]]; then
    break
  fi
  echo "  GPUs still busy (${active} compute process entries); waiting ${POLL_S}s..."
  sleep "${POLL_S}"
done

echo "[llama13b-vllm-seq4-probe] starting 256-request vLLM health probe with max_num_seqs=4."
echo "  Purpose: check whether the completed formal vLLM 13B result was dominated by backend queueing from max_num_seqs=2."

PAPER_QUEUE_ID="${PROBE_QUEUE_ID}" \
SLLM_RUN_TAG="${PROBE_RUN_TAG}" \
FAIR_ROUND_SYSTEMS=vllm \
SLLM_TOTAL_REQUESTS="${SLLM_TOTAL_REQUESTS:-256}" \
SLLM_SELECTED_NUM_ADAPTERS="${SLLM_SELECTED_NUM_ADAPTERS:-500}" \
SLLM_MODEL_PROFILE="${SLLM_MODEL_PROFILE:-llama2_13b_tp2_v2_publicmix}" \
SLLM_DATASET_PROFILE="${SLLM_DATASET_PROFILE:-azure_sharegpt_rep4000}" \
SLLM_WORKLOAD_PROFILE="${SLLM_WORKLOAD_PROFILE:-llama2_13b_tp2_a500_formal4000_s8}" \
SLLM_TIME_SCALE_FACTOR="${SLLM_TIME_SCALE_FACTOR:-8.0}" \
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}" \
VLLM_MAX_LORAS="${VLLM_MAX_LORAS:-4}" \
VLLM_MAX_CPU_LORAS="${VLLM_MAX_CPU_LORAS:-32}" \
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-2048}" \
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}" \
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}" \
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-0}" \
VLLM_USE_V1="${VLLM_USE_V1:-0}" \
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" \
bash "${BASELINES_ROOT}/scripts/run_llama13b_main_comparison_queue.sh"
