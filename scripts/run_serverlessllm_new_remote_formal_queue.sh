#!/usr/bin/env bash
set -euo pipefail

BASELINES_ROOT="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
MAIN_REPO="${SLLM_MAIN_REPO:-/home/qhq/serverless_llm_experiment_retry14_baseline}"
SLLM_REPO_ROOT="${SLLM_REPO_ROOT:-${BASELINES_ROOT}/vendor_new_baselines/ServerlessLLM_new_main_20260518}"
SLLM_ENV="${SLLM_NEW_ENV:-sllm_vllm0102_newserverless_20260518}"
SECTION_ID="${SLLM_NEW_SECTION_ID:-15_new_serverless_baselines_remote_v1}"
QUEUE_ID="${SLLM_NEW_QUEUE_ID:-$(date +%Y%m%d_%H%M%S)_serverlessllm_new_remote_v1}"
RESULT_ROOT="${BASELINES_ROOT}/results/paper_experiments/${SECTION_ID}"
LOG_ROOT="${BASELINES_ROOT}/results/logs/new_serverless_baselines_remote_v1/serverlessllm_new/formal/${QUEUE_ID}"
RUN_ONLY="${SLLM_NEW_ONLY:-all}"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"

REMOTE_NO_PROXY_HOSTS="192.168.4.174,10.199.227.174,127.0.0.1,localhost,::1"
if [[ -n "${NO_PROXY:-}" ]]; then
  export NO_PROXY="${NO_PROXY},${REMOTE_NO_PROXY_HOSTS}"
else
  export NO_PROXY="${REMOTE_NO_PROXY_HOSTS}"
fi
export no_proxy="${NO_PROXY}"
export HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy=

run_one() {
  local label="$1"
  local model_profile="$2"
  local workload_profile="$3"
  local run_tag="$4"
  local round_source="$5"
  local endpoint="$6"

  local round_dir="${RESULT_ROOT}/${QUEUE_ID}_${label}_${run_tag}_serverlessllm_new"
  local trace_path="${round_source}/shared_artifacts/${run_tag}_trace.json"
  local adapter_subset_path="${round_source}/shared_artifacts/${run_tag}_adapter_subset.json"
  local summary_path="${round_dir}/raw/replay/${run_tag}_serverlessllm_new_summary.json"

  mkdir -p "${round_dir}/shared_inputs" "${round_dir}/raw/replay" "${round_dir}/remote_cache/serverlessllm"

  if [[ -f "${summary_path}" ]]; then
    python "${BASELINES_ROOT}/scripts/validate_replay_results.py" \
      --system "ServerlessLLM-new" \
      --replay "${round_dir}/raw/replay/${run_tag}_serverlessllm_new_replay.json" \
      --expected-total 4000 >/dev/null 2>&1 && {
        echo "[skip] ${label}: existing valid summary ${summary_path}"
        return 0
      }
  fi

  echo "[run] ${label}"
  echo "      model_profile=${model_profile}"
  echo "      workload_profile=${workload_profile}"
  echo "      run_tag=${run_tag}"
  echo "      source_round=${round_source}"
  echo "      endpoint=${endpoint}"
  echo "      output_round=${round_dir}"

  SLLM_BASELINES_ROOT="${BASELINES_ROOT}" \
  SLLM_MAIN_REPO="${MAIN_REPO}" \
  SLLM_REPO_ROOT="${SLLM_REPO_ROOT}" \
  SLLM_HEAD_ENV="${SLLM_ENV}" \
  SLLM_WORKER_ENV="${SLLM_ENV}" \
  SLLM_STORE_ENV="${SLLM_ENV}" \
  SLLM_VLLM_ENV_NAME="${SLLM_ENV}" \
  SLLM_VLLM_SOURCE_ENV="${SLLM_ENV}" \
  SLLM_MODEL_PROFILE="${model_profile}" \
  SLLM_DATASET_PROFILE="azure_sharegpt_rep4000" \
  SLLM_WORKLOAD_PROFILE="${workload_profile}" \
  SLLM_TOTAL_REQUESTS="4000" \
  SLLM_SELECTED_NUM_ADAPTERS="500" \
  SLLM_SAMPLING_SEED="42" \
  SLLM_RUN_TAG="${run_tag}" \
  SLLM_RESULT_TAG="${run_tag}_serverlessllm_new" \
  SLLM_SHARED_TRACE_PATH="${trace_path}" \
  SLLM_SHARED_ADAPTER_SUBSET_PATH="${adapter_subset_path}" \
  SLLM_SHARED_INPUT_DIR="${round_dir}/shared_inputs" \
  SLLM_RESULT_DIR="${round_dir}/raw/replay" \
  SLLM_LOG_DIR="${LOG_ROOT}/${label}" \
  SLLM_REMOTE_ARTIFACT_STAGE_ENDPOINT="${endpoint}" \
  SLLM_REMOTE_ARTIFACT_STAGE_MODE="dynamic" \
  SLLM_REMOTE_ARTIFACT_STAGE_CACHE_DIR="${round_dir}/remote_cache/serverlessllm" \
  SLLM_REMOTE_ARTIFACT_STAGE_WORKERS="${SLLM_REMOTE_ARTIFACT_STAGE_WORKERS:-1}" \
  SLLM_REMOTE_ARTIFACT_STAGE_BANDWIDTH_MBPS="${SLLM_REMOTE_ARTIFACT_STAGE_BANDWIDTH_MBPS:-250}" \
  SLLM_BACKEND="vllm" \
  SLLM_WORKER_GPUS="${SLLM_WORKER_GPUS:-0,1,2,3}" \
  SLLM_TIMEOUT_S="${SLLM_TIMEOUT_S:-3600}" \
  SLLM_SLEEP_SCALE="1.0" \
  SLLM_EMPTY_SUCCESS_RETRIES="${SLLM_EMPTY_SUCCESS_RETRIES:-2}" \
  SLLM_STACK_SUFFIX="newsllm_${label}_${QUEUE_ID}" \
    bash "${BASELINES_ROOT}/scripts/run_serverlessllm_fair_experiment.sh"

  python - "${round_dir}" "${run_tag}" "${model_profile}" "${workload_profile}" "${trace_path}" "${adapter_subset_path}" "${endpoint}" "${SLLM_REPO_ROOT}" "${SLLM_ENV}" "${QUEUE_ID}" <<'PY'
import json
import sys
from pathlib import Path

round_dir = Path(sys.argv[1]).resolve()
run_tag = sys.argv[2]
manifest = {
    "metric_schema_version": "e2e_v3",
    "system": "serverlessllm_new",
    "queue_id": sys.argv[10],
    "run_tag": run_tag,
    "model_profile": sys.argv[3],
    "dataset_profile": "azure_sharegpt_rep4000",
    "workload_profile": sys.argv[4],
    "total_requests": 4000,
    "selected_num_adapters": 500,
    "sampling_seed": 42,
    "shared_trace_path": sys.argv[5],
    "shared_adapter_subset_path": sys.argv[6],
    "remote_artifact_endpoint": sys.argv[7],
    "serverlessllm_repo_root": sys.argv[8],
    "serverlessllm_env": sys.argv[9],
    "replay_path": str(round_dir / "raw" / "replay" / f"{run_tag}_serverlessllm_new_replay.json"),
    "summary_path": str(round_dir / "raw" / "replay" / f"{run_tag}_serverlessllm_new_summary.json"),
}
(round_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY
}

if [[ "${RUN_ONLY}" == "all" || "${RUN_ONLY}" == "llama2_7b" ]]; then
  run_one \
    "llama2_7b" \
    "llama2_7b_main_v2_publicmix" \
    "llama2_7b_auto500_formal4000_s8" \
    "llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1" \
    "${BASELINES_ROOT}/results/paper_experiments/12_remote_fair_main_real_remote_v1/20260513_012813_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1" \
    "http://192.168.4.174:18081"
fi

if [[ "${RUN_ONLY}" == "all" || "${RUN_ONLY}" == "llama32_3b" ]]; then
  run_one \
    "llama32_3b" \
    "llama32_3b_main_modelscope" \
    "llama32_3b_auto500_formal4000_s8" \
    "llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1" \
    "${BASELINES_ROOT}/results/paper_experiments/12_remote_fair_main_real_remote_v1/20260513_160342_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1" \
    "http://192.168.4.174:18080"
fi

echo "[done] ServerlessLLM-new true-remote formal queue completed: ${QUEUE_ID}"
