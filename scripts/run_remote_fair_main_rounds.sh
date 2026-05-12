#!/usr/bin/env bash
set -euo pipefail

BASELINES_ROOT="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
MAIN_REPO="${SLLM_MAIN_REPO:-/home/qhq/serverless_llm_experiment_retry14_baseline}"
MODE="${REMOTE_FAIR_MODE:-local-sim}"
SECTION="${REMOTE_FAIR_SECTION:-11_remote_fair_main}"
SYSTEMS="${REMOTE_FAIR_SYSTEMS:-sglang serverlessllm vllm slora}"
GPU_IDS="${REMOTE_FAIR_GPU_IDS:-0,1,2,3}"
BANDWIDTH_MBPS="${REMOTE_FAIR_BANDWIDTH_MBPS:-250}"
STAGE_WORKERS="${REMOTE_FAIR_STAGE_WORKERS:-1}"
MODEL_LIST="${REMOTE_FAIR_MODEL_LIST:-llama2_7b llama2_13b llama32_3b}"

case "${MODE}" in
  local-sim|real-remote) ;;
  *)
    echo "[ERROR] REMOTE_FAIR_MODE must be local-sim or real-remote; got ${MODE}" >&2
    exit 1
    ;;
esac

timestamp() {
  date +%Y%m%d_%H%M%S
}

local_endpoint_for_model() {
  case "$1" in
    llama2_7b)
      printf 'file://%s/artifacts/frozen/llama2_7b_a500_v2_publicmix\n' "${MAIN_REPO}"
      ;;
    llama2_13b)
      printf 'file://%s/artifacts/frozen/llama2_13b_a500_v2_publicmix\n' "${MAIN_REPO}"
      ;;
    llama32_3b)
      printf 'file://%s/artifacts/frozen/llama32_3b_a500_v1_modelscope\n' "${MAIN_REPO}"
      ;;
    *)
      echo "[ERROR] unknown model key for local endpoint: $1" >&2
      return 1
      ;;
  esac
}

real_endpoint_for_model() {
  local model_key="$1"
  local endpoint_var=""
  case "${model_key}" in
    llama2_7b) endpoint_var="${REMOTE_FAIR_REAL_ENDPOINT_LLAMA2_7B:-}" ;;
    llama2_13b) endpoint_var="${REMOTE_FAIR_REAL_ENDPOINT_LLAMA2_13B:-}" ;;
    llama32_3b) endpoint_var="${REMOTE_FAIR_REAL_ENDPOINT_LLAMA32_3B:-}" ;;
  esac
  printf '%s\n' "${endpoint_var:-${REMOTE_FAIR_REAL_ENDPOINT:-}}"
}

vllm_registration_mode_for_model() {
  local model_key="$1"
  local model_specific=""
  case "${model_key}" in
    llama2_7b) model_specific="${REMOTE_FAIR_VLLM_LORA_REGISTRATION_MODE_LLAMA2_7B:-}" ;;
    llama2_13b) model_specific="${REMOTE_FAIR_VLLM_LORA_REGISTRATION_MODE_LLAMA2_13B:-}" ;;
    llama32_3b) model_specific="${REMOTE_FAIR_VLLM_LORA_REGISTRATION_MODE_LLAMA32_3B:-}" ;;
  esac
  if [[ -n "${model_specific}" ]]; then
    printf '%s\n' "${model_specific}"
    return 0
  fi
  if [[ -n "${REMOTE_FAIR_VLLM_LORA_REGISTRATION_MODE:-}" ]]; then
    printf '%s\n' "${REMOTE_FAIR_VLLM_LORA_REGISTRATION_MODE}"
    return 0
  fi
  case "${model_key}" in
    llama32_3b)
      # Llama-3.2 3B uses request-time remote materialization for vLLM:
      # static registration of the 500-adapter universe exhausts host-memory
      # headroom on the 4x3090 testbed, while dynamic_remote preserves the
      # same remote artifact semantics on first adapter touch.
      printf 'dynamic_remote\n'
      ;;
    *)
      printf 'static_remote\n'
      ;;
  esac
}

run_one_model() {
  local model_key="$1"
  local model_profile=""
  local workload_profile=""
  local run_tag=""
  local time_scale=""
  local timeout_s=""
  local extra_env=()

  case "${model_key}" in
    llama2_7b)
      model_profile="llama2_7b_main_v2_publicmix"
      workload_profile="llama2_7b_auto500_formal4000_s8"
      run_tag="llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_${MODE}_v1"
      timeout_s="${REMOTE_FAIR_LLAMA2_7B_TIMEOUT_S:-7200}"
      ;;
    llama2_13b)
      model_profile="llama2_13b_tp2_v2_publicmix"
      workload_profile="llama2_13b_tp2_a500_formal4000_s8"
      run_tag="llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_${MODE}_v1"
      time_scale="8.0"
      timeout_s="${REMOTE_FAIR_LLAMA2_13B_TIMEOUT_S:-21600}"
      extra_env+=(
        SLORA_TENSOR_PARALLEL_SIZE="${SLORA_TENSOR_PARALLEL_SIZE:-4}"
        SLORA_DATA_PARALLEL_REPLICAS="${SLORA_DATA_PARALLEL_REPLICAS:-1}"
      )
      ;;
    llama32_3b)
      model_profile="llama32_3b_main_modelscope"
      workload_profile="llama32_3b_auto500_formal4000_s8"
      run_tag="llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_${MODE}_v1"
      time_scale="8.0"
      timeout_s="${REMOTE_FAIR_LLAMA32_3B_TIMEOUT_S:-7200}"
      ;;
    *)
      echo "[ERROR] unknown model key: ${model_key}" >&2
      exit 1
      ;;
  esac

  local endpoint=""
  if [[ "${MODE}" == "local-sim" ]]; then
    endpoint="$(local_endpoint_for_model "${model_key}")"
  else
    endpoint="$(real_endpoint_for_model "${model_key}")"
    if [[ -z "${endpoint}" ]]; then
      echo "[ERROR] real-remote mode requires REMOTE_FAIR_REAL_ENDPOINT or model-specific endpoint for ${model_key}" >&2
      exit 1
    fi
  fi
  local vllm_registration_mode=""
  vllm_registration_mode="$(vllm_registration_mode_for_model "${model_key}")"

  local round_dir="${BASELINES_ROOT}/results/paper_experiments/${SECTION}/$(timestamp)_${run_tag}"
  mkdir -p "${round_dir}"

  echo "============================================================"
  echo "[remote-fair] model=${model_key}"
  echo "[remote-fair] mode=${MODE}"
  echo "[remote-fair] systems=${SYSTEMS}"
  echo "[remote-fair] endpoint=${endpoint}"
  echo "[remote-fair] round_dir=${round_dir}"
  echo "============================================================"

  env \
    SLLM_BASELINES_ROOT="${BASELINES_ROOT}" \
    SLLM_MAIN_REPO="${MAIN_REPO}" \
    FAIR_ROUND_DIR="${round_dir}" \
    FAIR_ROUND_SECTION="${SECTION}" \
    FAIR_ROUND_LABEL="${run_tag}" \
    FAIR_ROUND_SYSTEMS="${SYSTEMS}" \
    FAIR_ROUND_GPU_IDS="${GPU_IDS}" \
    SLLM_RUN_TAG="${run_tag}" \
    SLLM_MODEL_PROFILE="${model_profile}" \
    SLLM_DATASET_PROFILE="azure_sharegpt_rep4000" \
    SLLM_WORKLOAD_PROFILE="${workload_profile}" \
    SLLM_TOTAL_REQUESTS="4000" \
    SLLM_SELECTED_NUM_ADAPTERS="500" \
    SLLM_SAMPLING_SEED="42" \
    SLLM_TIME_SCALE_FACTOR="${time_scale}" \
    SLLM_TIMEOUT_S="${timeout_s}" \
    VLLM_TIMEOUT_S="${timeout_s}" \
    SGLANG_TIMEOUT_S="${timeout_s}" \
    SLORA_TIMEOUT_S="${timeout_s}" \
    BASELINE_REMOTE_ARTIFACT_ENDPOINT="${endpoint}" \
    BASELINE_REMOTE_ARTIFACT_STAGE_ENDPOINT="${endpoint}" \
    BASELINE_REMOTE_ARTIFACT_BANDWIDTH_MBPS="${BANDWIDTH_MBPS}" \
    SLLM_REMOTE_ARTIFACT_STAGE_MODE="dynamic" \
    SLLM_REMOTE_ARTIFACT_STAGE_WORKERS="${STAGE_WORKERS}" \
    SLORA_REMOTE_ARTIFACT_STAGE_WORKERS="${STAGE_WORKERS}" \
    VLLM_LORA_REGISTRATION_MODE="${vllm_registration_mode}" \
    VLLM_REMOTE_ARTIFACT_STAGE_WORKERS="${STAGE_WORKERS}" \
    SGLANG_LORA_REGISTRATION_MODE="dynamic_remote" \
    VLLM_REMOTE_ARTIFACT_CACHE_DIR="${round_dir}/remote_cache/vllm" \
    SGLANG_REMOTE_ARTIFACT_CACHE_DIR="${round_dir}/remote_cache/sglang" \
    SLLM_REMOTE_ARTIFACT_STAGE_CACHE_DIR="${round_dir}/remote_cache/serverlessllm" \
    SLORA_REMOTE_ARTIFACT_STAGE_CACHE_DIR="${round_dir}/remote_cache/slora" \
    "${extra_env[@]}" \
    bash "${BASELINES_ROOT}/scripts/run_full_fair_round.sh"
}

main() {
  cd "${BASELINES_ROOT}"
  local model_key=""
  for model_key in ${MODEL_LIST}; do
    run_one_model "${model_key}"
  done
}

main "$@"
