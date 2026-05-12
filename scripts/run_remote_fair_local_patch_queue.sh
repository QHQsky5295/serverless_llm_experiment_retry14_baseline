#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SECTION="${REMOTE_FAIR_PATCH_SECTION:-11_remote_fair_main_local_sim_v4_patch}"
BANDWIDTH_MBPS="${REMOTE_FAIR_BANDWIDTH_MBPS:-250}"
STAGE_WORKERS="${REMOTE_FAIR_STAGE_WORKERS:-1}"
VLLM_MODE="${REMOTE_FAIR_VLLM_LORA_REGISTRATION_MODE:-static_remote}"

echo "[patch-queue] section=${SECTION}"
echo "[patch-queue] bandwidth_mib_s=${BANDWIDTH_MBPS}"
echo "[patch-queue] stage_workers=${STAGE_WORKERS}"
echo "[patch-queue] vllm_lora_registration_mode=${VLLM_MODE}"

echo
echo "[patch-queue] 1/3 Llama-2 7B: vLLM only"
REMOTE_FAIR_MODE=local-sim \
REMOTE_FAIR_SECTION="${SECTION}" \
REMOTE_FAIR_MODEL_LIST="llama2_7b" \
REMOTE_FAIR_SYSTEMS="vllm" \
REMOTE_FAIR_BANDWIDTH_MBPS="${BANDWIDTH_MBPS}" \
REMOTE_FAIR_STAGE_WORKERS="${STAGE_WORKERS}" \
REMOTE_FAIR_VLLM_LORA_REGISTRATION_MODE="${VLLM_MODE}" \
KILL_KNOWN_GPU_RESIDUALS="${KILL_KNOWN_GPU_RESIDUALS:-1}" \
bash scripts/run_remote_fair_main_rounds.sh

echo
echo "[patch-queue] 2/3 Llama-2 13B: vLLM + S-LoRA"
REMOTE_FAIR_MODE=local-sim \
REMOTE_FAIR_SECTION="${SECTION}" \
REMOTE_FAIR_MODEL_LIST="llama2_13b" \
REMOTE_FAIR_SYSTEMS="vllm slora" \
REMOTE_FAIR_BANDWIDTH_MBPS="${BANDWIDTH_MBPS}" \
REMOTE_FAIR_STAGE_WORKERS="${STAGE_WORKERS}" \
REMOTE_FAIR_VLLM_LORA_REGISTRATION_MODE="${VLLM_MODE}" \
KILL_KNOWN_GPU_RESIDUALS="${KILL_KNOWN_GPU_RESIDUALS:-1}" \
bash scripts/run_remote_fair_main_rounds.sh

echo
echo "[patch-queue] 3/3 Llama-3.2 3B: SGLang + ServerlessLLM + vLLM + S-LoRA"
REMOTE_FAIR_MODE=local-sim \
REMOTE_FAIR_SECTION="${SECTION}" \
REMOTE_FAIR_MODEL_LIST="llama32_3b" \
REMOTE_FAIR_SYSTEMS="sglang serverlessllm vllm slora" \
REMOTE_FAIR_BANDWIDTH_MBPS="${BANDWIDTH_MBPS}" \
REMOTE_FAIR_STAGE_WORKERS="${STAGE_WORKERS}" \
REMOTE_FAIR_VLLM_LORA_REGISTRATION_MODE="${VLLM_MODE}" \
REMOTE_FAIR_VLLM_LORA_REGISTRATION_MODE_LLAMA32_3B="${REMOTE_FAIR_VLLM_LORA_REGISTRATION_MODE_LLAMA32_3B:-dynamic_remote}" \
KILL_KNOWN_GPU_RESIDUALS="${KILL_KNOWN_GPU_RESIDUALS:-1}" \
bash scripts/run_remote_fair_main_rounds.sh

echo "[patch-queue] completed"
