#!/usr/bin/env bash
set -euo pipefail

# End-to-end queue for a non-overwriting true-remote artifact reproduction of
# the paper figure/table set.  The queue reuses the existing fair-round runner
# and only enables true HTTP remote artifact materialization uniformly for
# baselines and PrimeLoRA.

MAIN_REPO="${REMOTE_FULL_FIGS_MAIN_REPO:-/home/qhq/serverless_llm_experiment_retry14_baseline}"
BASELINES_ROOT="${REMOTE_FULL_FIGS_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
QUEUE_ID="${REMOTE_FULL_FIGS_QUEUE_ID:-20260514_real_remote_fullfigs_v1}"
QUEUE_DIR="${REMOTE_FULL_FIGS_QUEUE_DIR:-${MAIN_REPO}/results/remote_full_figs_queues/${QUEUE_ID}}"
STATE_DIR="${QUEUE_DIR}/state"
LOG_DIR="${QUEUE_DIR}/logs"

FIGS_ROOT="${REMOTE_FULL_FIGS_OUTPUT_DIR:-${MAIN_REPO}/figs_remote_full_real_remote_v1}"
PAPER_FIGS_ROOT="${FIGS_ROOT}/paper"
PAPER_RESULTS_DIR="${REMOTE_FULL_FIGS_PAPER_RESULTS_DIR:-${MAIN_REPO}/paper_results/final_remote_full_real_remote_v1}"

TRUE_REMOTE_MAIN_SECTION="${TRUE_REMOTE_MAIN_SECTION:-12_remote_fair_main_real_remote_v1}"
BASELINE_WRAPPER="${REMOTE_FULL_FIGS_BASELINE_WRAPPER:-${BASELINES_ROOT}/scripts/run_full_fair_round_true_remote_artifacts.sh}"
PYTHON_BIN="${REMOTE_FULL_FIGS_PYTHON:-python3}"

ROUND_7B="${REMOTE_FULL_FIGS_ROUND_7B:-${BASELINES_ROOT}/results/paper_experiments/${TRUE_REMOTE_MAIN_SECTION}/20260513_012813_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1}"
ROUND_13B="${REMOTE_FULL_FIGS_ROUND_13B:-${BASELINES_ROOT}/results/paper_experiments/${TRUE_REMOTE_MAIN_SECTION}/20260513_074336_llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1}"
ROUND_3B="${REMOTE_FULL_FIGS_ROUND_3B:-${BASELINES_ROOT}/results/paper_experiments/${TRUE_REMOTE_MAIN_SECTION}/20260513_160342_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1}"

PRIME_VLLM_7B="${REMOTE_FULL_FIGS_PRIME_VLLM_7B:-${MAIN_REPO}/results/experiment_results_full_vllm_auto_a500_r4000_c4_faaslora_full_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real_remote_v1_faaslora_vllm_real_remote_v1.json}"
PRIME_VLLM_13B="${REMOTE_FULL_FIGS_PRIME_VLLM_13B:-${MAIN_REPO}/results/experiment_results_full_vllm_auto_a500_r4000_c2_faaslora_full_llama2_13b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real_remote_v1_faaslora_vllm_real_remote_v1.json}"
PRIME_VLLM_3B="${REMOTE_FULL_FIGS_PRIME_VLLM_3B:-${MAIN_REPO}/results/experiment_results_full_vllm_auto_a500_r4000_c4_faaslora_full_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real_remote_v1_faaslora_vllm_real_remote_v1.json}"
PRIME_SGLANG_7B="${REMOTE_FULL_FIGS_PRIME_SGLANG_7B:-${MAIN_REPO}/results/experiment_results_full_sglang_auto_a500_r4000_c4_faaslora_full_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real_remote_v1_faaslora_sglang_real_remote_v1.json}"
PRIME_SGLANG_3B="${REMOTE_FULL_FIGS_PRIME_SGLANG_3B:-${MAIN_REPO}/results/experiment_results_full_sglang_auto_a500_r4000_c4_faaslora_full_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real_remote_v1_faaslora_sglang_real_remote_v1.json}"

mkdir -p "${STATE_DIR}" "${LOG_DIR}" "${PAPER_FIGS_ROOT}" "${PAPER_RESULTS_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

stage_done_path() {
  printf '%s/%s.done\n' "${STATE_DIR}" "$1"
}

is_done() {
  [[ -f "$(stage_done_path "$1")" ]]
}

mark_done() {
  date '+%F %T' >"$(stage_done_path "$1")"
}

run_stage() {
  local stage="$1"
  shift
  local log_path="${LOG_DIR}/${stage}.log"
  if is_done "${stage}"; then
    log "skip ${stage}; marker exists"
    return 0
  fi
  log "stage=${stage} log=${log_path}"
  set +e
  "$@" 2>&1 | tee "${log_path}"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "${status}" -ne 0 ]]; then
    log "stage=${stage} failed status=${status}"
    return "${status}"
  fi
  mark_done "${stage}"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] required file missing: ${path}" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] required directory missing: ${path}" >&2
    exit 2
  fi
}

remote_health() {
  local remote_no_proxy_hosts="192.168.4.174,10.199.227.174,127.0.0.1,localhost,::1"
  if [[ -n "${NO_PROXY:-}" ]]; then
    export NO_PROXY="${NO_PROXY},${remote_no_proxy_hosts}"
  else
    export NO_PROXY="${remote_no_proxy_hosts}"
  fi
  export no_proxy="${NO_PROXY}"
  export HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy=
  local endpoint=""
  for endpoint in http://192.168.4.174:18081 http://192.168.4.174:18082 http://192.168.4.174:18080; do
    echo "[remote-health] ${endpoint}/health"
    curl -fsS "${endpoint}/health"
    echo
  done
}

run_load_queue() {
  cd "${BASELINES_ROOT}"
  env \
    SLLM_BASELINES_ROOT="${BASELINES_ROOT}" \
    SLLM_MAIN_REPO="${MAIN_REPO}" \
    PAPER_QUEUE_ID="${QUEUE_ID}_load" \
    PAPER_QUEUE_PROFILE="load_operating_p0" \
    PAPER_QUEUE_SYSTEMS="sglang serverlessllm vllm slora faaslora" \
    PAPER_QUEUE_RUNNER="${BASELINE_WRAPPER}" \
    bash "${BASELINES_ROOT}/scripts/run_paper_long_experiment_queue.sh"
}

run_adapter_pool_queue() {
  cd "${BASELINES_ROOT}"
  env \
    SLLM_BASELINES_ROOT="${BASELINES_ROOT}" \
    SLLM_MAIN_REPO="${MAIN_REPO}" \
    PAPER_QUEUE_ID="${QUEUE_ID}_adpool" \
    PAPER_QUEUE_PROFILE="adapter_pool_p0" \
    PAPER_QUEUE_SYSTEMS="sglang serverlessllm vllm slora faaslora" \
    PAPER_QUEUE_RUNNER="${BASELINE_WRAPPER}" \
    bash "${BASELINES_ROOT}/scripts/run_paper_long_experiment_queue.sh"
}

run_ablation_queue() {
  cd "${MAIN_REPO}"
  env \
    FAASLORA_MAIN_REPO="${MAIN_REPO}" \
    FAASLORA_BASELINES_ROOT="${BASELINES_ROOT}" \
    FAASLORA_PROFILE_MODEL="llama2_7b_main_v2_publicmix" \
    FAASLORA_PROFILE_DATASET="azure_sharegpt_rep4000" \
    FAASLORA_PROFILE_WORKLOAD="llama2_7b_auto500_formal4000_s8" \
    FAASLORA_TOTAL_REQUESTS="4000" \
    FAASLORA_SELECTED_NUM_ADAPTERS="500" \
    FAASLORA_SAMPLING_SEED="42" \
    FAASLORA_SOURCE_RUN_TAG="llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1" \
    FAASLORA_SOURCE_ROUND_DIR="${ROUND_7B}" \
    FAASLORA_PAPER_ABLATION_RUN_TAG="llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_ablation_real_remote_fullfigs_v1" \
    FAASLORA_PAPER_ABLATION_SECTION_ID="14_remote_fair_ablation_real_remote_full_figs_v1" \
    FAASLORA_PAPER_ABLATION_ROUND_DIR="${BASELINES_ROOT}/results/paper_experiments/14_remote_fair_ablation_real_remote_full_figs_v1/${QUEUE_ID}_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_ablation_real_remote_fullfigs_v1" \
    FAASLORA_REMOTE_ARTIFACT_ENABLED="1" \
    FAASLORA_REMOTE_ARTIFACT_ENDPOINT="http://192.168.4.174:18081" \
    FAASLORA_REMOTE_ARTIFACT_BANDWIDTH_MBPS="250" \
    PYTHONUNBUFFERED="1" \
    bash "${MAIN_REPO}/scripts/run_faaslora_paper_ablation_round.sh"
}

build_figures() {
  cd "${MAIN_REPO}"
  require_dir "${ROUND_7B}"
  require_dir "${ROUND_3B}"
  require_file "${PRIME_VLLM_7B}"
  require_file "${PRIME_VLLM_3B}"
  require_file "${PRIME_SGLANG_7B}"
  require_file "${PRIME_SGLANG_3B}"

  local load_s12="${BASELINES_ROOT}/results/paper_experiments/06_sensitivity_load_operating/${QUEUE_ID}_load_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s12_sensloadop_v1"
  local load_s10="${BASELINES_ROOT}/results/paper_experiments/06_sensitivity_load_operating/${QUEUE_ID}_load_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s10_sensloadop_v1"
  local ad_a100="${BASELINES_ROOT}/results/paper_experiments/07_sensitivity_adapter_pool/${QUEUE_ID}_adpool_llama2_7b_r4000_a100_seed42_z1p0_hot16_rot500_s8_sensadpool_v1"
  local ad_a200="${BASELINES_ROOT}/results/paper_experiments/07_sensitivity_adapter_pool/${QUEUE_ID}_adpool_llama2_7b_r4000_a200_seed42_z1p0_hot24_rot500_s8_sensadpool_v1"
  local ad_a300="${BASELINES_ROOT}/results/paper_experiments/07_sensitivity_adapter_pool/${QUEUE_ID}_adpool_llama2_7b_r4000_a300_seed42_z1p0_hot32_rot500_s8_sensadpool_v1"
  local ad_a400="${BASELINES_ROOT}/results/paper_experiments/07_sensitivity_adapter_pool/${QUEUE_ID}_adpool_llama2_7b_r4000_a400_seed42_z1p0_hot40_rot500_s8_sensadpool_v1"
  # a500 is the default main workload. Reuse the canonical true-remote main
  # round instead of rerunning an identical adapter-pool endpoint.
  local ad_a500="${ROUND_7B}"
  local ablation_round="${BASELINES_ROOT}/results/paper_experiments/14_remote_fair_ablation_real_remote_full_figs_v1/${QUEUE_ID}_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_ablation_real_remote_fullfigs_v1"

  require_dir "${load_s12}"
  require_dir "${load_s10}"
  require_dir "${ad_a100}"
  require_dir "${ad_a200}"
  require_dir "${ad_a300}"
  require_dir "${ad_a400}"
  require_dir "${ad_a500}"
  require_dir "${ablation_round}"

  "${PYTHON_BIN}" scripts/build_main_7b13b_artifacts.py \
    --round-7b "${ROUND_7B}" \
    --round-3b "${ROUND_3B}" \
    --system-summary-override "llama2_7b:faaslora:${PRIME_VLLM_7B}" \
    --system-summary-override "llama32_3b:faaslora:${PRIME_VLLM_3B}" \
    --out-dir "${PAPER_FIGS_ROOT}/main"

  # The combined main builder intentionally owns Table 1, TTFT decomposition,
  # and Fig. 7.  Generate the single-round teaser and normalized appendix-style
  # figure separately so the true-remote mirror remains complete without
  # overwriting the merged Llama-2-7B + Llama-3.2-3B artifacts.
  "${PYTHON_BIN}" scripts/plot_paper_figures.py \
    --round-dir "${ROUND_7B}" \
    --figure fig1_intro \
    --out-dir "${PAPER_FIGS_ROOT}/main"

  "${PYTHON_BIN}" scripts/plot_paper_figures.py \
    --round-dir "${ROUND_7B}" \
    --figure fig5_normalized \
    --out-dir "${PAPER_FIGS_ROOT}/main"

  "${PYTHON_BIN}" scripts/build_backend_portability_artifacts.py \
    --round-7b "${ROUND_7B}" \
    --round-3b "${ROUND_3B}" \
    --prime-7b-summary "${PRIME_VLLM_7B}" \
    --prime-3b-summary "${PRIME_VLLM_3B}" \
    --prime-sglang-7b-summary "${PRIME_SGLANG_7B}" \
    --prime-sglang-3b-summary "${PRIME_SGLANG_3B}" \
    --out-dir "${PAPER_FIGS_ROOT}/backend_portability"

  "${PYTHON_BIN}" scripts/plot_paper_figures.py \
    --round-dir "${ROUND_7B}" \
    --figure motivation_all \
    --out-dir "${PAPER_FIGS_ROOT}/motivation"

  "${PYTHON_BIN}" scripts/plot_paper_figures.py \
    --round-dir "${ablation_round}" \
    --figure ablation_all \
    --out-dir "${PAPER_FIGS_ROOT}/ablation"

  "${PYTHON_BIN}" scripts/analyze_service_readiness.py \
    --input "${ablation_round}" \
    --output "${PAPER_FIGS_ROOT}/readiness"

  "${PYTHON_BIN}" scripts/analyze_control_path_overhead.py \
    --input "${ablation_round}" \
    --output "${PAPER_FIGS_ROOT}/control_path"

  "${PYTHON_BIN}" scripts/plot_paper_sensitivity.py \
    --round-dir "${load_s12}" \
    --round-dir "${load_s10}" \
    --round-dir "${ad_a500}" \
    --system-summary-override "${ad_a500}:faaslora:${PRIME_VLLM_7B}" \
    --figure load \
    --out-dir "${PAPER_FIGS_ROOT}/sensitivity"

  "${PYTHON_BIN}" scripts/plot_paper_sensitivity.py \
    --round-dir "${ad_a100}" \
    --round-dir "${ad_a200}" \
    --round-dir "${ad_a300}" \
    --round-dir "${ad_a400}" \
    --round-dir "${ad_a500}" \
    --system-summary-override "${ad_a500}:faaslora:${PRIME_VLLM_7B}" \
    --figure adapter_pool \
    --out-dir "${PAPER_FIGS_ROOT}/sensitivity"

  mkdir -p "${PAPER_RESULTS_DIR}/figs" "${PAPER_RESULTS_DIR}/sources"
  rsync -a --delete "${FIGS_ROOT}/" "${PAPER_RESULTS_DIR}/figs/"
  gzip -c "${PRIME_VLLM_7B}" >"${PAPER_RESULTS_DIR}/sources/prime_vllm_7b.json.gz"
  gzip -c "${PRIME_VLLM_3B}" >"${PAPER_RESULTS_DIR}/sources/prime_vllm_3b.json.gz"
  gzip -c "${PRIME_SGLANG_7B}" >"${PAPER_RESULTS_DIR}/sources/prime_sglang_7b.json.gz"
  gzip -c "${PRIME_SGLANG_3B}" >"${PAPER_RESULTS_DIR}/sources/prime_sglang_3b.json.gz"
  {
    echo "# True-Remote Full Figure Snapshot"
    echo
    echo "- queue_id: ${QUEUE_ID}"
    echo "- figs_root: ${FIGS_ROOT}"
    echo "- paper_results: ${PAPER_RESULTS_DIR}"
    echo "- generated_at: $(date '+%F %T %Z')"
    echo
    echo "This snapshot is non-overwriting and uses real HTTP remote artifact endpoints for every system in the newly run figure experiments."
  } >"${PAPER_RESULTS_DIR}/README.md"
  (cd "${FIGS_ROOT}" && find . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
  (cd "${PAPER_RESULTS_DIR}" && find . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
}

main() {
  log "queue_id=${QUEUE_ID}"
  log "queue_dir=${QUEUE_DIR}"
  log "figs_root=${FIGS_ROOT}"
  log "paper_results_dir=${PAPER_RESULTS_DIR}"
  run_stage "00_remote_health" remote_health
  run_stage "10_load_queue" run_load_queue
  run_stage "20_adapter_pool_queue" run_adapter_pool_queue
  run_stage "30_ablation_queue" run_ablation_queue
  run_stage "40_build_figures" build_figures
  log "complete: ${QUEUE_DIR}"
}

main "$@"
