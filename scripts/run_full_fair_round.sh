#!/usr/bin/env bash
set -euo pipefail

BASELINES_ROOT="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
MAIN_REPO="${SLLM_MAIN_REPO:-/home/qhq/serverless_llm_experiment_retry14_baseline}"

MODEL_PROFILE="${SLLM_MODEL_PROFILE:-llama2_7b_main_v2_publicmix}"
DATASET_PROFILE="${SLLM_DATASET_PROFILE:-azure_sharegpt_rep4000}"
WORKLOAD_PROFILE="${SLLM_WORKLOAD_PROFILE:-llama2_7b_auto500_formal4000_s8}"
TOTAL_REQUESTS="${SLLM_TOTAL_REQUESTS:-4000}"
SELECTED_NUM_ADAPTERS="${SLLM_SELECTED_NUM_ADAPTERS:-500}"
SAMPLING_SEED="${SLLM_SAMPLING_SEED:-42}"

default_run_tag() {
  local model="$1" workload="$2" requests="$3" adapters="$4" seed="$5"
  if [[ "${model}" == "llama2_7b_main_v2_publicmix" && "${workload}" == "llama2_7b_auto500_formal4000_s8" && "${requests}" == "4000" && "${adapters}" == "500" && "${seed}" == "42" ]]; then
    printf '%s\n' "llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1"
    return
  fi
  printf '%s_r%s_a%s_seed%s_%s\n' "${model}" "${requests}" "${adapters}" "${seed}" "${workload}" \
    | tr -c 'A-Za-z0-9_.-' '_'
}

RUN_TAG="${SLLM_RUN_TAG:-$(default_run_tag "${MODEL_PROFILE}" "${WORKLOAD_PROFILE}" "${TOTAL_REQUESTS}" "${SELECTED_NUM_ADAPTERS}" "${SAMPLING_SEED}")}"
ROUND_SECTION="${FAIR_ROUND_SECTION:-03_main_comparison}"
ROUND_ROOT="${FAIR_ROUND_ROOT:-${BASELINES_ROOT}/results/paper_experiments/${ROUND_SECTION}}"
ROUND_LABEL="${FAIR_ROUND_LABEL:-${RUN_TAG}}"
ROUND_TIMESTAMP="${FAIR_ROUND_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
ROUND_DIR="${FAIR_ROUND_DIR:-${ROUND_ROOT}/${ROUND_TIMESTAMP}_${ROUND_LABEL}}"

SYSTEMS="${FAIR_ROUND_SYSTEMS:-sglang serverlessllm vllm slora faaslora}"
GPU_IDS="${FAIR_ROUND_GPU_IDS:-0,1,2,3}"
STRICT_GPU_IDLE="${FAIR_ROUND_STRICT_GPU_IDLE:-1}"
CLEANUP_TIMEOUT_S="${FAIR_ROUND_CLEANUP_TIMEOUT_S:-180}"
FORCE_RERUN="${FAIR_ROUND_FORCE:-0}"
KILL_KNOWN_GPU_RESIDUALS="${FAIR_ROUND_KILL_KNOWN_GPU_RESIDUALS:-1}"

TRACE_PATH="${ROUND_DIR}/shared_artifacts/${RUN_TAG}_trace.json"
ADAPTER_SUBSET_PATH="${ROUND_DIR}/shared_artifacts/${RUN_TAG}_adapter_subset.json"
RAW_REPLAY_DIR="${ROUND_DIR}/raw/replay"
RAW_LOG_DIR="${ROUND_DIR}/raw/logs"
RAW_SHARED_INPUT_DIR="${ROUND_DIR}/raw/shared_inputs"
RAW_FAAS_DIR="${ROUND_DIR}/raw/faaslora"
STEP_LOG_DIR="${ROUND_DIR}/logs"
STATE_DIR="${ROUND_DIR}/state"
COMPARE_DIR="${ROUND_DIR}/compare"

mkdir -p \
  "${ROUND_DIR}" \
  "${ROUND_DIR}/shared_artifacts" \
  "${RAW_REPLAY_DIR}" \
  "${RAW_LOG_DIR}" \
  "${RAW_SHARED_INPUT_DIR}" \
  "${RAW_FAAS_DIR}" \
  "${STEP_LOG_DIR}" \
  "${STATE_DIR}" \
  "${COMPARE_DIR}"

ROUND_ENV_FILE="${ROUND_DIR}/round.env"

write_round_env() {
  {
    printf 'export FAIR_ROUND_DIR=%q\n' "${ROUND_DIR}"
    printf 'export SLLM_RUN_TAG=%q\n' "${RUN_TAG}"
    printf 'export SLLM_MODEL_PROFILE=%q\n' "${MODEL_PROFILE}"
    printf 'export SLLM_DATASET_PROFILE=%q\n' "${DATASET_PROFILE}"
    printf 'export SLLM_WORKLOAD_PROFILE=%q\n' "${WORKLOAD_PROFILE}"
    printf 'export SLLM_TOTAL_REQUESTS=%q\n' "${TOTAL_REQUESTS}"
    printf 'export SLLM_SELECTED_NUM_ADAPTERS=%q\n' "${SELECTED_NUM_ADAPTERS}"
    printf 'export SLLM_SAMPLING_SEED=%q\n' "${SAMPLING_SEED}"
  } >"${ROUND_ENV_FILE}"
}

write_round_env

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

stage_done_path() {
  printf '%s/%s.done\n' "${STATE_DIR}" "$1"
}

is_done() {
  [[ "${FORCE_RERUN}" != "1" && -f "$(stage_done_path "$1")" ]]
}

mark_done() {
  date '+%F %T' >"$(stage_done_path "$1")"
}

run_logged() {
  local stage="$1"
  shift
  local log_path="${STEP_LOG_DIR}/${stage}.log"
  log "stage=${stage} log=${log_path}"
  set +e
  "$@" 2>&1 | tee "${log_path}"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "${status}" -ne 0 ]]; then
    log "stage=${stage} failed status=${status}"
    return "${status}"
  fi
}

kill_listener_port() {
  local port="$1"
  local pids=()
  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi
  mapfile -t pids < <(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null | awk '!seen[$0]++')
  if (( ${#pids[@]} == 0 )); then
    return 0
  fi
  log "clearing stale listener port=${port} pid=${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true
  for _ in $(seq 1 15); do
    sleep 1
    mapfile -t pids < <(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null | awk '!seen[$0]++')
    if (( ${#pids[@]} == 0 )); then
      return 0
    fi
  done
  log "forcing stale listener cleanup port=${port} pid=${pids[*]}"
  kill -9 "${pids[@]}" 2>/dev/null || true
}

cleanup_known_leftovers() {
  log "cleaning known leftover services"
  bash "${BASELINES_ROOT}/scripts/stop_serverlessllm_stack.sh" >/dev/null 2>&1 || true
  local ports=(
    8353
    8363 8373 8383 8393
    8463 8473 8483 8493
    8000 8001 8002 8003
  )
  local port
  for port in "${ports[@]}"; do
    kill_listener_port "${port}"
  done
  kill_known_gpu_residuals
}

gpu_residual_pids() {
  local gpu_csv="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  local gpu_ids=()
  local gpu=""
  IFS=',' read -r -a gpu_ids <<< "${gpu_csv}"
  for gpu in "${gpu_ids[@]}"; do
    gpu="$(printf '%s' "${gpu}" | xargs)"
    [[ -z "${gpu}" ]] && continue
    nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | sed '/^[[:space:]]*$/d' \
      | awk '{print $1}'
  done | sort -u
}

kill_known_gpu_residuals() {
  if [[ "${KILL_KNOWN_GPU_RESIDUALS}" != "1" ]]; then
    return 0
  fi
  local pids=()
  local pid=""
  mapfile -t pids < <(gpu_residual_pids "${GPU_IDS}" || true)
  if (( ${#pids[@]} == 0 )); then
    return 0
  fi
  for pid in "${pids[@]}"; do
    [[ -z "${pid}" ]] && continue
    local cmd=""
    cmd="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
    case "${cmd}" in
      *serverless_llm_baselines*|*serverless_llm_experiment*|*sglang*|*vllm*|*slora*|*sllm*|*dedicated_engine_worker*)
        log "killing known GPU residual pid=${pid} cmd=${cmd}"
        kill "${pid}" 2>/dev/null || true
        ;;
      *)
        log "leaving non-round GPU process pid=${pid} cmd=${cmd}"
        ;;
    esac
  done
  sleep 3
}

gpu_compute_rows() {
  local gpu_csv="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  local gpu_ids=()
  local gpu=""
  IFS=',' read -r -a gpu_ids <<< "${gpu_csv}"
  for gpu in "${gpu_ids[@]}"; do
    gpu="$(printf '%s' "${gpu}" | xargs)"
    [[ -z "${gpu}" ]] && continue
    nvidia-smi --id="${gpu}" --query-compute-apps=pid,used_gpu_memory,process_name --format=csv,noheader,nounits 2>/dev/null \
      | sed '/^[[:space:]]*$/d' \
      | awk -v gpu="${gpu}" '{print "gpu=" gpu " " $0}'
  done
}

wait_gpu_idle() {
  if [[ "${STRICT_GPU_IDLE}" != "1" ]]; then
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  local deadline=$((SECONDS + CLEANUP_TIMEOUT_S))
  local rows=""
  while true; do
    rows="$(gpu_compute_rows "${GPU_IDS}" || true)"
    if [[ -z "${rows}" ]]; then
      log "GPU compute state clean for gpu_ids=${GPU_IDS}"
      return 0
    fi
    if (( SECONDS >= deadline )); then
      echo "[ERROR] GPU compute processes remain after cleanup:" >&2
      printf '%s\n' "${rows}" >&2
      echo "        Re-run with the same FAIR_ROUND_DIR after cleaning them, or set FAIR_ROUND_STRICT_GPU_IDLE=0 only for diagnosis." >&2
      return 1
    fi
    log "waiting for GPU cleanup:"
    printf '%s\n' "${rows}"
    sleep 5
  done
}

pre_system_clean_check() {
  local system="$1"
  log "pre-clean before ${system}"
  cleanup_known_leftovers
  wait_gpu_idle
}

post_system_clean_check() {
  local system="$1"
  log "post-clean after ${system}"
  cleanup_known_leftovers
  wait_gpu_idle
}

summary_path_for_system() {
  case "$1" in
    sglang) printf '%s/%s_sglang_dp4_tp1_summary.json\n' "${RAW_REPLAY_DIR}" "${RUN_TAG}" ;;
    serverlessllm) printf '%s/%s_serverlessllm_summary.json\n' "${RAW_REPLAY_DIR}" "${RUN_TAG}" ;;
    vllm) printf '%s/%s_vllm_dp4_tp1_summary.json\n' "${RAW_REPLAY_DIR}" "${RUN_TAG}" ;;
    slora) printf '%s/%s_slora_dp4_tp1_summary.json\n' "${RAW_REPLAY_DIR}" "${RUN_TAG}" ;;
    faaslora) printf '%s/%s_faaslora_result.json\n' "${RAW_FAAS_DIR}" "${RUN_TAG}" ;;
    *) return 1 ;;
  esac
}

validate_summary() {
  local system="$1"
  local path="$2"
  python3 - "${system}" "${path}" "${TOTAL_REQUESTS}" <<'PY'
import json
import sys
from pathlib import Path

system = sys.argv[1]
path = Path(sys.argv[2])
expected_total = int(sys.argv[3])
if not path.exists():
    raise SystemExit(f"[ERROR] missing summary for {system}: {path}")
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("metric_schema_version") != "e2e_v3":
    raise SystemExit(f"[ERROR] {system} metric_schema_version is not e2e_v3: {data.get('metric_schema_version')}")
summaries = data.get("scenario_summaries")
if isinstance(summaries, dict) and summaries:
    summary = next(iter(summaries.values()))
elif isinstance(summaries, list) and summaries:
    summary = summaries[0]
else:
    table = data.get("comparison_table") or []
    summary = table[0] if table else data
completed = int(summary.get("completed_requests", summary.get("completed", -1)) or -1)
total = int(summary.get("total_requests", summary.get("total", expected_total)) or expected_total)
failed = int(summary.get("failed_requests", max(total - completed, 0)) or 0)
if total != expected_total:
    raise SystemExit(f"[ERROR] {system} total mismatch: expected {expected_total}, got {total}")
if completed != total or failed != 0:
    raise SystemExit(f"[ERROR] {system} incomplete result: completed={completed}, total={total}, failed={failed}")
for source_key in ("prompt_token_source_counts", "completion_token_source_counts", "metrics_source_counts"):
    counts = summary.get(source_key) or {}
    if isinstance(counts, dict) and counts.get("trace_expected", 0):
        raise SystemExit(f"[ERROR] {system} used trace_expected token fallback in {source_key}: {counts}")
table = data.get("comparison_table") or []
row = table[0] if table else {}
checks = {
    "TTFT_e2e_avg_ms": row.get("TTFT_e2e_avg_ms", summary.get("avg_overall_ttft_ms")),
    "E2E_e2e_avg_ms": row.get("E2E_e2e_avg_ms", summary.get("avg_overall_e2e_ms")),
    "Cost/req": row.get("avg_cost_USD", summary.get("avg_cost_usd")),
    "CE": row.get("CE", summary.get("ce") or summary.get("monetary_ce")),
}
for key, value in checks.items():
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise SystemExit(f"[ERROR] {system} missing numeric {key}: {value!r}")
    if numeric <= 0.0:
        raise SystemExit(f"[ERROR] {system} non-positive {key}: {numeric}")
print(f"[validated] {system}: completed={completed}/{total} path={path}")
PY
}

run_prep() {
  local stage="00_prep"
  if is_done "${stage}"; then
    log "skip ${stage}; marker exists"
    return 0
  fi
  run_logged "${stage}" env \
    SLLM_BASELINES_ROOT="${BASELINES_ROOT}" \
    SLLM_MAIN_REPO="${MAIN_REPO}" \
    SLLM_SHARED_ROUND_DIR="${ROUND_DIR}/shared_artifacts" \
    SLLM_MODEL_PROFILE="${MODEL_PROFILE}" \
    SLLM_DATASET_PROFILE="${DATASET_PROFILE}" \
    SLLM_WORKLOAD_PROFILE="${WORKLOAD_PROFILE}" \
    SLLM_TOTAL_REQUESTS="${TOTAL_REQUESTS}" \
    SLLM_SELECTED_NUM_ADAPTERS="${SELECTED_NUM_ADAPTERS}" \
    SLLM_SAMPLING_SEED="${SAMPLING_SEED}" \
    SLLM_RUN_TAG="${RUN_TAG}" \
    bash "${BASELINES_ROOT}/scripts/prepare_shared_round_artifacts.sh"
  [[ -f "${TRACE_PATH}" ]] || { echo "[ERROR] missing trace: ${TRACE_PATH}" >&2; return 1; }
  [[ -f "${ADAPTER_SUBSET_PATH}" ]] || { echo "[ERROR] missing adapter subset: ${ADAPTER_SUBSET_PATH}" >&2; return 1; }
  mark_done "${stage}"
}

run_sglang() {
  local stage="10_sglang"
  if is_done "${stage}"; then
    log "skip ${stage}; marker exists"
    return 0
  fi
  pre_system_clean_check "SGLang"
  run_logged "${stage}" env \
    SLLM_BASELINES_ROOT="${BASELINES_ROOT}" \
    SLLM_MAIN_REPO="${MAIN_REPO}" \
    SLLM_RESULT_DIR="${RAW_REPLAY_DIR}" \
    SLLM_LOG_DIR="${RAW_LOG_DIR}" \
    SLLM_SHARED_INPUT_DIR="${RAW_SHARED_INPUT_DIR}" \
    SLLM_MODEL_PROFILE="${MODEL_PROFILE}" \
    SLLM_DATASET_PROFILE="${DATASET_PROFILE}" \
    SLLM_WORKLOAD_PROFILE="${WORKLOAD_PROFILE}" \
    SLLM_TOTAL_REQUESTS="${TOTAL_REQUESTS}" \
    SLLM_SELECTED_NUM_ADAPTERS="${SELECTED_NUM_ADAPTERS}" \
    SLLM_SAMPLING_SEED="${SAMPLING_SEED}" \
    SLLM_RUN_TAG="${RUN_TAG}" \
    SLLM_SHARED_TRACE_PATH="${TRACE_PATH}" \
    SLLM_SHARED_ADAPTER_SUBSET_PATH="${ADAPTER_SUBSET_PATH}" \
    SGLANG_GPU_IDS="${GPU_IDS}" \
    SGLANG_DATA_PARALLEL_REPLICAS="${SGLANG_DATA_PARALLEL_REPLICAS:-4}" \
    SGLANG_TENSOR_PARALLEL_SIZE="${SGLANG_TENSOR_PARALLEL_SIZE:-1}" \
    bash "${BASELINES_ROOT}/scripts/run_sglang_fair_experiment.sh"
  validate_summary "SGLang" "$(summary_path_for_system sglang)"
  post_system_clean_check "SGLang"
  mark_done "${stage}"
}

run_serverlessllm() {
  local stage="20_serverlessllm"
  if is_done "${stage}"; then
    log "skip ${stage}; marker exists"
    return 0
  fi
  pre_system_clean_check "ServerlessLLM"
  run_logged "${stage}" env \
    SLLM_BASELINES_ROOT="${BASELINES_ROOT}" \
    SLLM_MAIN_REPO="${MAIN_REPO}" \
    SLLM_RESULT_DIR="${RAW_REPLAY_DIR}" \
    SLLM_LOG_DIR="${RAW_LOG_DIR}" \
    SLLM_SHARED_INPUT_DIR="${RAW_SHARED_INPUT_DIR}" \
    SLLM_MODEL_PROFILE="${MODEL_PROFILE}" \
    SLLM_DATASET_PROFILE="${DATASET_PROFILE}" \
    SLLM_WORKLOAD_PROFILE="${WORKLOAD_PROFILE}" \
    SLLM_TOTAL_REQUESTS="${TOTAL_REQUESTS}" \
    SLLM_SELECTED_NUM_ADAPTERS="${SELECTED_NUM_ADAPTERS}" \
    SLLM_SAMPLING_SEED="${SAMPLING_SEED}" \
    SLLM_RUN_TAG="${RUN_TAG}" \
    SLLM_SHARED_TRACE_PATH="${TRACE_PATH}" \
    SLLM_SHARED_ADAPTER_SUBSET_PATH="${ADAPTER_SUBSET_PATH}" \
    SLLM_BACKEND="${SLLM_BACKEND:-vllm}" \
    SLLM_WORKER_GPUS="${GPU_IDS}" \
    bash "${BASELINES_ROOT}/scripts/run_serverlessllm_fair_experiment.sh"
  validate_summary "ServerlessLLM" "$(summary_path_for_system serverlessllm)"
  post_system_clean_check "ServerlessLLM"
  mark_done "${stage}"
}

run_vllm() {
  local stage="30_vllm"
  if is_done "${stage}"; then
    log "skip ${stage}; marker exists"
    return 0
  fi
  pre_system_clean_check "vLLM"
  run_logged "${stage}" env \
    SLLM_BASELINES_ROOT="${BASELINES_ROOT}" \
    SLLM_MAIN_REPO="${MAIN_REPO}" \
    SLLM_RESULT_DIR="${RAW_REPLAY_DIR}" \
    SLLM_LOG_DIR="${RAW_LOG_DIR}" \
    SLLM_SHARED_INPUT_DIR="${RAW_SHARED_INPUT_DIR}" \
    SLLM_MODEL_PROFILE="${MODEL_PROFILE}" \
    SLLM_DATASET_PROFILE="${DATASET_PROFILE}" \
    SLLM_WORKLOAD_PROFILE="${WORKLOAD_PROFILE}" \
    SLLM_TOTAL_REQUESTS="${TOTAL_REQUESTS}" \
    SLLM_SELECTED_NUM_ADAPTERS="${SELECTED_NUM_ADAPTERS}" \
    SLLM_SAMPLING_SEED="${SAMPLING_SEED}" \
    SLLM_RUN_TAG="${RUN_TAG}" \
    SLLM_SHARED_TRACE_PATH="${TRACE_PATH}" \
    SLLM_SHARED_ADAPTER_SUBSET_PATH="${ADAPTER_SUBSET_PATH}" \
    VLLM_GPU_IDS="${GPU_IDS}" \
    VLLM_DATA_PARALLEL_REPLICAS="${VLLM_DATA_PARALLEL_REPLICAS:-4}" \
    VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}" \
    bash "${BASELINES_ROOT}/scripts/run_vllm_fair_experiment.sh"
  validate_summary "vLLM" "$(summary_path_for_system vllm)"
  post_system_clean_check "vLLM"
  mark_done "${stage}"
}

run_slora() {
  local stage="40_slora"
  if is_done "${stage}"; then
    log "skip ${stage}; marker exists"
    return 0
  fi
  pre_system_clean_check "S-LoRA"
  run_logged "${stage}" env \
    SLLM_BASELINES_ROOT="${BASELINES_ROOT}" \
    SLLM_MAIN_REPO="${MAIN_REPO}" \
    SLLM_RESULT_DIR="${RAW_REPLAY_DIR}" \
    SLLM_LOG_DIR="${RAW_LOG_DIR}" \
    SLLM_SHARED_INPUT_DIR="${RAW_SHARED_INPUT_DIR}" \
    SLLM_MODEL_PROFILE="${MODEL_PROFILE}" \
    SLLM_DATASET_PROFILE="${DATASET_PROFILE}" \
    SLLM_WORKLOAD_PROFILE="${WORKLOAD_PROFILE}" \
    SLLM_TOTAL_REQUESTS="${TOTAL_REQUESTS}" \
    SLLM_SELECTED_NUM_ADAPTERS="${SELECTED_NUM_ADAPTERS}" \
    SLLM_SAMPLING_SEED="${SAMPLING_SEED}" \
    SLLM_RUN_TAG="${RUN_TAG}" \
    SLLM_SHARED_TRACE_PATH="${TRACE_PATH}" \
    SLLM_SHARED_ADAPTER_SUBSET_PATH="${ADAPTER_SUBSET_PATH}" \
    SLORA_GPU_IDS="${GPU_IDS}" \
    SLORA_DATA_PARALLEL_REPLICAS="${SLORA_DATA_PARALLEL_REPLICAS:-4}" \
    SLORA_TENSOR_PARALLEL_SIZE="${SLORA_TENSOR_PARALLEL_SIZE:-1}" \
    bash "${BASELINES_ROOT}/scripts/run_slora_fair_experiment.sh"
  validate_summary "S-LoRA" "$(summary_path_for_system slora)"
  post_system_clean_check "S-LoRA"
  mark_done "${stage}"
}

find_latest_faaslora_result() {
  local tag="$1"
  find "${MAIN_REPO}/results" -maxdepth 3 -type f -name "*${tag}*.json" -printf '%T@ %p\n' 2>/dev/null \
    | sort -n \
    | awk 'END{print substr($0, index($0,$2))}'
}

run_faaslora() {
  local stage="50_faaslora"
  local faas_tag="${RUN_TAG}_faaslora"
  local faas_copy
  faas_copy="$(summary_path_for_system faaslora)"
  if is_done "${stage}"; then
    log "skip ${stage}; marker exists"
    return 0
  fi
  pre_system_clean_check "FaaSLoRA"
  run_logged "${stage}" env \
    FAASLORA_PROFILE_MODEL="${MODEL_PROFILE}" \
    FAASLORA_PROFILE_DATASET="${DATASET_PROFILE}" \
    FAASLORA_PROFILE_WORKLOAD="${WORKLOAD_PROFILE}" \
    FAASLORA_TOTAL_REQUESTS="${TOTAL_REQUESTS}" \
    FAASLORA_SHARED_TRACE_PATH="${TRACE_PATH}" \
    FAASLORA_SHARED_ADAPTER_SUBSET_PATH="${ADAPTER_SUBSET_PATH}" \
    FAASLORA_RESULTS_TAG="${faas_tag}" \
    PYTHONUNBUFFERED=1 \
    bash "${MAIN_REPO}/scripts/run_faaslora_shared_artifact_experiment.sh" \
      --num-adapters "${SELECTED_NUM_ADAPTERS}" \
      --full-stack
  local latest
  latest="$(find_latest_faaslora_result "${faas_tag}")"
  if [[ -z "${latest}" || ! -f "${latest}" ]]; then
    echo "[ERROR] unable to locate FaaSLoRA result for tag=${faas_tag}" >&2
    return 1
  fi
  cp -f "${latest}" "${faas_copy}"
  printf '%s\n' "${latest}" >"${RAW_FAAS_DIR}/${RUN_TAG}_faaslora_source_path.txt"
  validate_summary "FaaSLoRA" "${faas_copy}"
  post_system_clean_check "FaaSLoRA"
  mark_done "${stage}"
}

run_compare() {
  local stage="90_compare"
  if is_done "${stage}"; then
    log "skip ${stage}; marker exists"
    return 0
  fi
  local compare_json="${COMPARE_DIR}/${RUN_TAG}_five_system_compare.json"
  local compare_txt="${COMPARE_DIR}/${RUN_TAG}_five_system_compare.txt"
  local log_path="${STEP_LOG_DIR}/${stage}.log"
  local args=(python3 "${BASELINES_ROOT}/scripts/compare_fair_results.py")
  local system=""
  for system in faaslora sglang serverlessllm vllm slora; do
    case " ${SYSTEMS} " in
      *" ${system} "*) args+=(--result "$(summary_path_for_system "${system}")") ;;
    esac
  done
  args+=(--output "${compare_json}")
  log "stage=${stage} log=${log_path}"
  set +e
  "${args[@]}" 2>&1 | tee "${compare_txt}" | tee "${log_path}"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "${status}" -ne 0 ]]; then
    log "stage=${stage} failed status=${status}"
    return "${status}"
  fi
  mark_done "${stage}"
}

write_manifest() {
  python3 - "${ROUND_DIR}" "${RUN_TAG}" "${MODEL_PROFILE}" "${DATASET_PROFILE}" "${WORKLOAD_PROFILE}" "${TOTAL_REQUESTS}" "${SELECTED_NUM_ADAPTERS}" "${SAMPLING_SEED}" "${TRACE_PATH}" "${ADAPTER_SUBSET_PATH}" "${SYSTEMS}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

round_dir = Path(sys.argv[1])
payload = {
    "run_tag": sys.argv[2],
    "model_profile": sys.argv[3],
    "dataset_profile": sys.argv[4],
    "workload_profile": sys.argv[5],
    "total_requests": int(sys.argv[6]),
    "selected_num_adapters": int(sys.argv[7]),
    "sampling_seed": int(sys.argv[8]),
    "shared_trace_path": sys.argv[9],
    "shared_adapter_subset_path": sys.argv[10],
    "systems": sys.argv[11].split(),
    "metric_schema_version": "e2e_v3",
    "round_dir": str(round_dir),
    "state_dir": str(round_dir / "state"),
    "compare_json": str(round_dir / "compare" / f"{sys.argv[2]}_five_system_compare.json"),
    "compare_txt": str(round_dir / "compare" / f"{sys.argv[2]}_five_system_compare.txt"),
}
for name, cwd in (("baseline_git", "/home/qhq/serverless_llm_baselines"), ("faaslora_git", "/home/qhq/serverless_llm_experiment_retry14_baseline")):
    try:
        commit = subprocess.check_output(["git", "-C", cwd, "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(["git", "-C", cwd, "branch", "--show-current"], text=True).strip()
    except Exception:
        commit = ""
        branch = ""
    payload[name] = {"path": cwd, "branch": branch, "commit": commit}
(round_dir / "MANIFEST.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(round_dir / "MANIFEST.json")
PY
}

run_system_if_selected() {
  local system="$1"
  case " ${SYSTEMS} " in
    *" ${system} "*) "run_${system}" ;;
    *) log "skip ${system}; not listed in FAIR_ROUND_SYSTEMS=${SYSTEMS}" ;;
  esac
}

main() {
  log "round_dir=${ROUND_DIR}"
  log "run_tag=${RUN_TAG}"
  log "resume_file=${ROUND_ENV_FILE}"
  run_prep
  run_system_if_selected sglang
  run_system_if_selected serverlessllm
  run_system_if_selected vllm
  run_system_if_selected slora
  run_system_if_selected faaslora
  run_compare
  write_manifest
  log "round complete: ${ROUND_DIR}"
  log "comparison: ${COMPARE_DIR}/${RUN_TAG}_five_system_compare.txt"
}

main "$@"
