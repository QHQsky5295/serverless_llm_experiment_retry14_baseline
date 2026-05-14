#!/usr/bin/env bash
set -euo pipefail

# Run PrimeLoRA true-remote artifact experiments after the matching baseline
# remote-fair rounds have produced their shared trace/subset artifacts.
#
# This script is opt-in and non-overwriting: it writes new result tags suffixed
# with _real_remote_v1 and never modifies the closed-loop local-sim data.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_ROOT="${BASELINE_ROOT:-/home/qhq/serverless_llm_baselines}"
BASELINE_SECTION="${BASELINE_SECTION:-12_remote_fair_main_real_remote_v1}"
WAIT_FOR_TMUX="${WAIT_FOR_TMUX:-}"
MODEL_LIST="${FAASLORA_TRUE_REMOTE_MODEL_LIST:-llama2_7b llama2_13b llama32_3b}"
BACKENDS_BY_MODEL="${FAASLORA_TRUE_REMOTE_BACKENDS_BY_MODEL:-llama2_7b:vllm,sglang llama2_13b:vllm llama32_3b:vllm,sglang}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/remote_fair_real_faaslora_v1}"

mkdir -p "${LOG_DIR}"

export NO_PROXY="${NO_PROXY:-192.168.4.174,10.199.227.174,127.0.0.1,localhost,::1}"
export no_proxy="${no_proxy:-${NO_PROXY}}"
export HTTP_PROXY=
export HTTPS_PROXY=
export ALL_PROXY=
export http_proxy=
export https_proxy=
export all_proxy=

if [[ -n "${WAIT_FOR_TMUX}" ]]; then
  echo "[queue] waiting for tmux session ${WAIT_FOR_TMUX} to finish before starting PrimeLoRA true-remote runs"
  while tmux has-session -t "${WAIT_FOR_TMUX}" 2>/dev/null; do
    sleep 300
  done
fi

model_profile() {
  case "$1" in
    llama2_7b) echo "llama2_7b_main_v2_publicmix" ;;
    llama2_13b) echo "llama2_13b_tp2_v2_publicmix" ;;
    llama32_3b) echo "llama32_3b_main_modelscope" ;;
    *) echo "[ERROR] unknown model key $1" >&2; exit 2 ;;
  esac
}

workload_profile() {
  case "$1" in
    llama2_7b) echo "llama2_7b_auto500_formal4000_s8" ;;
    llama2_13b) echo "llama2_13b_tp2_a500_formal4000_s8" ;;
    llama32_3b) echo "llama32_3b_auto500_formal4000_s8" ;;
    *) echo "[ERROR] unknown model key $1" >&2; exit 2 ;;
  esac
}

endpoint_for_model() {
  case "$1" in
    llama2_7b) echo "${FAASLORA_REMOTE_ENDPOINT_LLAMA2_7B:-http://192.168.4.174:18081}" ;;
    llama2_13b) echo "${FAASLORA_REMOTE_ENDPOINT_LLAMA2_13B:-http://192.168.4.174:18082}" ;;
    llama32_3b) echo "${FAASLORA_REMOTE_ENDPOINT_LLAMA32_3B:-http://192.168.4.174:18080}" ;;
    *) echo "[ERROR] unknown model key $1" >&2; exit 2 ;;
  esac
}

round_glob_for_model() {
  case "$1" in
    llama2_7b) echo "*llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1" ;;
    llama2_13b) echo "*llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1" ;;
    llama32_3b) echo "*llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1" ;;
    *) echo "[ERROR] unknown model key $1" >&2; exit 2 ;;
  esac
}

find_round_dir() {
  local model_key="$1"
  local pattern
  pattern="$(round_glob_for_model "${model_key}")"
  find "${BASELINE_ROOT}/results/paper_experiments/${BASELINE_SECTION}" \
    -maxdepth 1 -mindepth 1 -type d -name "${pattern}" 2>/dev/null | sort | tail -n 1
}

run_tag_for_model() {
  case "$1" in
    llama2_7b) echo "llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1" ;;
    llama2_13b) echo "llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1" ;;
    llama32_3b) echo "llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1" ;;
    *) echo "[ERROR] unknown model key $1" >&2; exit 2 ;;
  esac
}

backends_for_model() {
  local model_key="$1"
  local item
  for item in ${BACKENDS_BY_MODEL}; do
    if [[ "${item%%:*}" == "${model_key}" ]]; then
      echo "${item#*:}" | tr ',' ' '
      return 0
    fi
  done
  echo "vllm"
}

run_one() {
  local model_key="$1"
  local backend="$2"
  local round_dir="$3"
  local run_tag
  local trace_path
  local subset_path
  local results_tag
  local script

  run_tag="$(run_tag_for_model "${model_key}")"
  trace_path="${round_dir}/shared_artifacts/${run_tag}_trace.json"
  subset_path="${round_dir}/shared_artifacts/${run_tag}_adapter_subset.json"
  if [[ ! -f "${trace_path}" || ! -f "${subset_path}" ]]; then
    echo "[ERROR] missing shared artifacts for ${model_key}: ${round_dir}" >&2
    exit 3
  fi

  case "${backend}" in
    vllm)
      script="${ROOT_DIR}/scripts/run_faaslora_shared_artifact_experiment.sh"
      results_tag="${run_tag}_faaslora_vllm_real_remote_v1"
      ;;
    sglang)
      script="${ROOT_DIR}/scripts/run_faaslora_sglang_shared_artifact_experiment.sh"
      results_tag="${run_tag}_faaslora_sglang_real_remote_v1"
      ;;
    *)
      echo "[ERROR] unknown backend ${backend}" >&2
      exit 4
      ;;
  esac

  echo "[run] ${model_key}/${backend}: ${results_tag}"
  (
    cd "${ROOT_DIR}"
    export FAASLORA_PROFILE_MODEL
    export FAASLORA_PROFILE_DATASET="azure_sharegpt_rep4000"
    export FAASLORA_PROFILE_WORKLOAD
    export FAASLORA_SHARED_TRACE_PATH="${trace_path}"
    export FAASLORA_SHARED_ADAPTER_SUBSET_PATH="${subset_path}"
    export FAASLORA_RESULTS_TAG="${results_tag}"
    export FAASLORA_REMOTE_ARTIFACT_ENABLED=1
    export FAASLORA_REMOTE_ARTIFACT_ENDPOINT
    export PYTHONUNBUFFERED=1
    FAASLORA_PROFILE_MODEL="$(model_profile "${model_key}")"
    FAASLORA_PROFILE_WORKLOAD="$(workload_profile "${model_key}")"
    FAASLORA_REMOTE_ARTIFACT_ENDPOINT="$(endpoint_for_model "${model_key}")"

    # Keep model-specific tuned envelopes documented in the paper logs.
    if [[ "${model_key}" == "llama32_3b" ]]; then
      export FAASLORA_MAX_INSTANCES="${FAASLORA_MAX_INSTANCES:-2}"
    fi
    if [[ "${model_key}" == "llama2_13b" ]]; then
      export FAASLORA_MIN_INSTANCES="${FAASLORA_MIN_INSTANCES:-2}"
      export FAASLORA_MAX_INSTANCES="${FAASLORA_MAX_INSTANCES:-2}"
    fi

    bash "${script}" --num-adapters 500 --full-stack
  ) 2>&1 | tee "${LOG_DIR}/${results_tag}.log"
}

for model_key in ${MODEL_LIST}; do
  round_dir="$(find_round_dir "${model_key}")"
  if [[ -z "${round_dir}" ]]; then
    echo "[ERROR] could not find baseline true-remote round for ${model_key} in ${BASELINE_SECTION}" >&2
    exit 5
  fi
  for backend in $(backends_for_model "${model_key}"); do
    run_one "${model_key}" "${backend}" "${round_dir}"
  done
done

echo "[done] PrimeLoRA true-remote queue completed."
