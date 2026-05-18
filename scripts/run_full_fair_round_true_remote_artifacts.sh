#!/usr/bin/env bash
set -euo pipefail

# Wrap the existing fair-round runner with true remote-artifact settings.
# This script intentionally does not change workload generation, routing,
# backend parameters, metric definitions, or result collection. It only opens
# the same remote artifact path for every system in a round.

BASELINES_ROOT="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
MAIN_REPO="${SLLM_MAIN_REPO:-/home/qhq/serverless_llm_experiment_retry14_baseline}"
RUNNER="${REMOTE_FULL_FIGS_FAIR_RUNNER:-${BASELINES_ROOT}/scripts/run_full_fair_round.sh}"
BANDWIDTH_MBPS="${REMOTE_FULL_FIGS_BANDWIDTH_MBPS:-250}"
STAGE_WORKERS="${REMOTE_FULL_FIGS_STAGE_WORKERS:-1}"
MODEL_PROFILE="${SLLM_MODEL_PROFILE:-llama2_7b_main_v2_publicmix}"
ROUND_DIR="${FAIR_ROUND_DIR:-}"

endpoint_for_profile() {
  case "$1" in
    llama2_7b_main_v2_publicmix)
      printf '%s\n' "${REMOTE_FULL_FIGS_ENDPOINT_LLAMA2_7B:-http://192.168.4.174:18081}"
      ;;
    llama2_13b_tp2_v2_publicmix)
      printf '%s\n' "${REMOTE_FULL_FIGS_ENDPOINT_LLAMA2_13B:-http://192.168.4.174:18082}"
      ;;
    llama32_3b_main_modelscope)
      printf '%s\n' "${REMOTE_FULL_FIGS_ENDPOINT_LLAMA32_3B:-http://192.168.4.174:18080}"
      ;;
    *)
      printf '%s\n' "${REMOTE_FULL_FIGS_ENDPOINT:-}"
      ;;
  esac
}

vllm_registration_mode_for_profile() {
  case "$1" in
    llama32_3b_main_modelscope)
      printf 'dynamic_remote\n'
      ;;
    *)
      printf 'static_remote\n'
      ;;
  esac
}

endpoint="$(endpoint_for_profile "${MODEL_PROFILE}")"
if [[ -z "${endpoint}" ]]; then
  echo "[ERROR] no true-remote endpoint configured for SLLM_MODEL_PROFILE=${MODEL_PROFILE}" >&2
  exit 2
fi

cache_root="${ROUND_DIR:-${BASELINES_ROOT}/results/remote_artifact_cache/${MODEL_PROFILE}/true_remote_full_figs/${SLLM_RUN_TAG:-run}}/remote_cache"

REMOTE_NO_PROXY_HOSTS="192.168.4.174,10.199.227.174,127.0.0.1,localhost,::1"
if [[ -n "${NO_PROXY:-}" ]]; then
  export NO_PROXY="${NO_PROXY},${REMOTE_NO_PROXY_HOSTS}"
else
  export NO_PROXY="${REMOTE_NO_PROXY_HOSTS}"
fi
export no_proxy="${NO_PROXY}"
export HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy=

export BASELINE_REMOTE_ARTIFACT_ENDPOINT="${BASELINE_REMOTE_ARTIFACT_ENDPOINT:-${endpoint}}"
export BASELINE_REMOTE_ARTIFACT_STAGE_ENDPOINT="${BASELINE_REMOTE_ARTIFACT_STAGE_ENDPOINT:-${endpoint}}"
export BASELINE_REMOTE_ARTIFACT_BANDWIDTH_MBPS="${BASELINE_REMOTE_ARTIFACT_BANDWIDTH_MBPS:-${BANDWIDTH_MBPS}}"
export BASELINE_REMOTE_ARTIFACT_STAGE_WORKERS="${BASELINE_REMOTE_ARTIFACT_STAGE_WORKERS:-${STAGE_WORKERS}}"

export SLLM_REMOTE_ARTIFACT_STAGE_MODE="${SLLM_REMOTE_ARTIFACT_STAGE_MODE:-dynamic}"
export SLLM_REMOTE_ARTIFACT_STAGE_WORKERS="${SLLM_REMOTE_ARTIFACT_STAGE_WORKERS:-${STAGE_WORKERS}}"
export SLLM_REMOTE_ARTIFACT_STAGE_CACHE_DIR="${SLLM_REMOTE_ARTIFACT_STAGE_CACHE_DIR:-${cache_root}/serverlessllm}"

export SLORA_REMOTE_ARTIFACT_STAGE_WORKERS="${SLORA_REMOTE_ARTIFACT_STAGE_WORKERS:-${STAGE_WORKERS}}"
export SLORA_REMOTE_ARTIFACT_STAGE_CACHE_DIR="${SLORA_REMOTE_ARTIFACT_STAGE_CACHE_DIR:-${cache_root}/slora}"

export VLLM_LORA_REGISTRATION_MODE="${VLLM_LORA_REGISTRATION_MODE:-$(vllm_registration_mode_for_profile "${MODEL_PROFILE}")}"
export VLLM_REMOTE_ARTIFACT_STAGE_WORKERS="${VLLM_REMOTE_ARTIFACT_STAGE_WORKERS:-${STAGE_WORKERS}}"
export VLLM_REMOTE_ARTIFACT_CACHE_DIR="${VLLM_REMOTE_ARTIFACT_CACHE_DIR:-${cache_root}/vllm}"

export SGLANG_LORA_REGISTRATION_MODE="${SGLANG_LORA_REGISTRATION_MODE:-dynamic_remote}"
export SGLANG_REMOTE_ARTIFACT_CACHE_DIR="${SGLANG_REMOTE_ARTIFACT_CACHE_DIR:-${cache_root}/sglang}"

export FAASLORA_REMOTE_ARTIFACT_ENABLED="${FAASLORA_REMOTE_ARTIFACT_ENABLED:-1}"
export FAASLORA_REMOTE_ARTIFACT_ENDPOINT="${FAASLORA_REMOTE_ARTIFACT_ENDPOINT:-${endpoint}}"
export FAASLORA_REMOTE_ARTIFACT_BANDWIDTH_MBPS="${FAASLORA_REMOTE_ARTIFACT_BANDWIDTH_MBPS:-${BANDWIDTH_MBPS}}"

echo "[true-remote-wrapper] model_profile=${MODEL_PROFILE}"
echo "[true-remote-wrapper] endpoint=${endpoint}"
echo "[true-remote-wrapper] bandwidth_mib_s=${BANDWIDTH_MBPS}"
echo "[true-remote-wrapper] vllm_lora_registration_mode=${VLLM_LORA_REGISTRATION_MODE}"
echo "[true-remote-wrapper] cache_root=${cache_root}"

exec bash "${RUNNER}" "$@"
