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
TIME_SCALE_FACTOR="${SLLM_TIME_SCALE_FACTOR:-}"
GENERATION_CONTRACT="${FAIR_GENERATION_CONTRACT:-legacy}"
FIXED_OUTPUT_MAX_TOKENS="${FAIR_FIXED_OUTPUT_MAX_TOKENS:-256}"
FIXED_PROMPT_MAX_TOKENS="${FAIR_FIXED_PROMPT_MAX_TOKENS:-759}"
STORAGE_BANDWIDTH_MIB_S="${FAIR_STORAGE_BANDWIDTH_MIB_S:-${BASELINE_REMOTE_ARTIFACT_BANDWIDTH_MIB_S:-${BASELINE_REMOTE_ARTIFACT_BANDWIDTH_MBPS:-250}}}"
ZIPF_EXPONENT="${SLLM_ZIPF_EXPONENT:-}"
ACTIVE_ADAPTER_CAP="${SLLM_ACTIVE_ADAPTER_CAP:-}"
HOTSET_ROTATION_REQUESTS="${SLLM_HOTSET_ROTATION_REQUESTS:-}"
HOTSET_ROTATION_MODE="${SLLM_HOTSET_ROTATION_MODE:-}"
HOTSET_OVERLAP_FRACTION="${SLLM_HOTSET_OVERLAP_FRACTION:-}"
FAASLORA_SCENARIO="${FAIR_FAASLORA_SCENARIO:-faaslora_full}"
FORMAL_RUN="${FAIR_FORMAL_RUN:-0}"
CAMPAIGN_KIND="${FAIR_CAMPAIGN_KIND:-}"
RESOLVED_CONFIG_FAMILY="${FAIR_RESOLVED_CONFIG_FAMILY:-}"

case "${FORMAL_RUN}" in
  0|1) ;;
  *)
    echo "[ERROR] FAIR_FORMAL_RUN must be 0 or 1; got ${FORMAL_RUN}" >&2
    exit 1
    ;;
esac

infer_trace_role() {
  if [[ "${FAASLORA_SCENARIO}" != v2_* ]]; then
    printf '%s\n' legacy
    return 0
  fi
  case "${SAMPLING_SEED}" in
    41) printf '%s\n' validation ;;
    42) printf '%s\n' smoke ;;
    43|44|45) printf '%s\n' heldout ;;
    *) printf '%s\n' invalid ;;
  esac
}

TRACE_ROLE="${FAIR_TRACE_ROLE:-$(infer_trace_role)}"

case "${GENERATION_CONTRACT}" in
  legacy|fixed_length_greedy_v1)
    ;;
  *)
    echo "[ERROR] unsupported FAIR_GENERATION_CONTRACT=${GENERATION_CONTRACT}" >&2
    exit 1
    ;;
esac
if [[ "${GENERATION_CONTRACT}" == "fixed_length_greedy_v1" ]]; then
  if ! [[ "${FIXED_OUTPUT_MAX_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] FAIR_FIXED_OUTPUT_MAX_TOKENS must be a positive integer" >&2
    exit 1
  fi
  if ! [[ "${FIXED_PROMPT_MAX_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] FAIR_FIXED_PROMPT_MAX_TOKENS must be a positive integer" >&2
    exit 1
  fi
fi
if ! [[ "${STORAGE_BANDWIDTH_MIB_S}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "[ERROR] FAIR_STORAGE_BANDWIDTH_MIB_S must be a non-negative number" >&2
  exit 1
fi

default_run_tag() {
  local model="$1" workload="$2" requests="$3" adapters="$4" seed="$5"
  if [[ "${model}" == "llama2_7b_main_v2_publicmix" && "${workload}" == "llama2_7b_auto500_formal4000_s8" && "${requests}" == "4000" && "${adapters}" == "500" && "${seed}" == "42" ]]; then
    printf '%s\n' "llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1"
    return
  fi
  if [[ "${model}" == "llama32_3b_main_modelscope" && "${workload}" == "llama32_3b_auto500_formal4000_s8" && "${requests}" == "4000" && "${adapters}" == "500" && "${seed}" == "42" ]]; then
    printf '%s\n' "llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv2_seq8"
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
RESOLVED_CONFIG_REGISTRY="${FAIR_RESOLVED_CONFIG_REGISTRY:-${ROUND_ROOT}/_protocol/system_resolved_config_registry.json}"

SYSTEMS="${FAIR_ROUND_SYSTEMS:-sglang serverlessllm vllm slora faaslora}"
EXECUTION_ORDER="${FAIR_ROUND_EXECUTION_ORDER:-sglang serverlessllm vllm slora faaslora}"
GPU_IDS="${FAIR_ROUND_GPU_IDS:-0,1,2,3}"
STRICT_GPU_IDLE="${FAIR_ROUND_STRICT_GPU_IDLE:-1}"
CLEANUP_TIMEOUT_S="${FAIR_ROUND_CLEANUP_TIMEOUT_S:-180}"
FORCE_RERUN="${FAIR_ROUND_FORCE:-0}"
KILL_KNOWN_GPU_RESIDUALS="${FAIR_ROUND_KILL_KNOWN_GPU_RESIDUALS:-1}"
DRY_RUN="${FAIR_ROUND_DRY_RUN:-${PAPER_QUEUE_DRY_RUN:-0}}"

TRACE_PATH="${ROUND_DIR}/shared_artifacts/${RUN_TAG}_trace.json"
ADAPTER_SUBSET_PATH="${ROUND_DIR}/shared_artifacts/${RUN_TAG}_adapter_subset.json"
RAW_REPLAY_DIR="${ROUND_DIR}/raw/replay"
RAW_LOG_DIR="${ROUND_DIR}/raw/logs"
RAW_SHARED_INPUT_DIR="${ROUND_DIR}/raw/shared_inputs"
RAW_FAAS_DIR="${ROUND_DIR}/raw/faaslora"
STEP_LOG_DIR="${ROUND_DIR}/logs"
STATE_DIR="${ROUND_DIR}/state"
COMPARE_DIR="${ROUND_DIR}/compare"
PROTOCOL_DIR="${ROUND_DIR}/protocol"
RESOLVED_CONFIG_PATH="${PROTOCOL_DIR}/system_resolved_config.json"

mkdir -p \
  "${ROUND_DIR}" \
  "${ROUND_DIR}/shared_artifacts" \
  "${RAW_REPLAY_DIR}" \
  "${RAW_LOG_DIR}" \
  "${RAW_SHARED_INPUT_DIR}" \
  "${RAW_FAAS_DIR}" \
  "${STEP_LOG_DIR}" \
  "${STATE_DIR}" \
  "${COMPARE_DIR}" \
  "${PROTOCOL_DIR}"

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
    printf 'export SLLM_TIME_SCALE_FACTOR=%q\n' "${TIME_SCALE_FACTOR}"
    printf 'export FAIR_GENERATION_CONTRACT=%q\n' "${GENERATION_CONTRACT}"
    printf 'export FAIR_FIXED_OUTPUT_MAX_TOKENS=%q\n' "${FIXED_OUTPUT_MAX_TOKENS}"
    printf 'export FAIR_FIXED_PROMPT_MAX_TOKENS=%q\n' "${FIXED_PROMPT_MAX_TOKENS}"
    printf 'export FAIR_STORAGE_BANDWIDTH_MIB_S=%q\n' "${STORAGE_BANDWIDTH_MIB_S}"
    printf 'export SLLM_ZIPF_EXPONENT=%q\n' "${ZIPF_EXPONENT}"
    printf 'export SLLM_ACTIVE_ADAPTER_CAP=%q\n' "${ACTIVE_ADAPTER_CAP}"
    printf 'export SLLM_HOTSET_ROTATION_REQUESTS=%q\n' "${HOTSET_ROTATION_REQUESTS}"
    printf 'export SLLM_HOTSET_ROTATION_MODE=%q\n' "${HOTSET_ROTATION_MODE}"
    printf 'export SLLM_HOTSET_OVERLAP_FRACTION=%q\n' "${HOTSET_OVERLAP_FRACTION}"
    printf 'export FAIR_FAASLORA_SCENARIO=%q\n' "${FAASLORA_SCENARIO}"
    printf 'export FAIR_ROUND_EXECUTION_ORDER=%q\n' "${EXECUTION_ORDER}"
    printf 'export FAIR_CAMPAIGN_KIND=%q\n' "${CAMPAIGN_KIND}"
    printf 'export FAIR_FORMAL_RUN=%q\n' "${FORMAL_RUN}"
    printf 'export FAIR_TRACE_ROLE=%q\n' "${TRACE_ROLE}"
    printf 'export FAIR_RESOLVED_CONFIG_FAMILY=%q\n' "${RESOLVED_CONFIG_FAMILY}"
    printf 'export FAIR_RESOLVED_CONFIG_REGISTRY=%q\n' "${RESOLVED_CONFIG_REGISTRY}"
  } >"${ROUND_ENV_FILE}"
}

validate_or_write_round_env() {
  if [[ "${FORCE_RERUN}" != "1" && -f "${STATE_DIR}/00_prep.done" && -f "${ROUND_ENV_FILE}" ]]; then
    local existing=()
    mapfile -t existing < <(
      bash -c '
        source "$1"
        printf "%s\n" \
          "${SLLM_RUN_TAG:-}" \
          "${SLLM_MODEL_PROFILE:-}" \
          "${SLLM_DATASET_PROFILE:-}" \
          "${SLLM_WORKLOAD_PROFILE:-}" \
          "${SLLM_TOTAL_REQUESTS:-}" \
          "${SLLM_SELECTED_NUM_ADAPTERS:-}" \
          "${SLLM_SAMPLING_SEED:-}" \
          "${SLLM_TIME_SCALE_FACTOR:-}" \
          "${FAIR_GENERATION_CONTRACT:-legacy}" \
          "${FAIR_FIXED_OUTPUT_MAX_TOKENS:-256}" \
          "${FAIR_FIXED_PROMPT_MAX_TOKENS:-759}" \
          "${FAIR_STORAGE_BANDWIDTH_MIB_S:-250}" \
          "${SLLM_ZIPF_EXPONENT:-}" \
          "${SLLM_ACTIVE_ADAPTER_CAP:-}" \
          "${SLLM_HOTSET_ROTATION_REQUESTS:-}" \
          "${SLLM_HOTSET_ROTATION_MODE:-}" \
          "${SLLM_HOTSET_OVERLAP_FRACTION:-}" \
          "${FAIR_FAASLORA_SCENARIO:-faaslora_full}" \
          "${FAIR_ROUND_EXECUTION_ORDER:-sglang serverlessllm vllm slora faaslora}" \
          "${FAIR_CAMPAIGN_KIND:-}" \
          "${FAIR_FORMAL_RUN:-0}" \
          "${FAIR_TRACE_ROLE:-legacy}" \
          "${FAIR_RESOLVED_CONFIG_FAMILY:-}" \
          "${FAIR_RESOLVED_CONFIG_REGISTRY:-}"
      ' bash "${ROUND_ENV_FILE}"
    )
    local names=(
      SLLM_RUN_TAG
      SLLM_MODEL_PROFILE
      SLLM_DATASET_PROFILE
      SLLM_WORKLOAD_PROFILE
      SLLM_TOTAL_REQUESTS
      SLLM_SELECTED_NUM_ADAPTERS
      SLLM_SAMPLING_SEED
      SLLM_TIME_SCALE_FACTOR
      FAIR_GENERATION_CONTRACT
      FAIR_FIXED_OUTPUT_MAX_TOKENS
      FAIR_FIXED_PROMPT_MAX_TOKENS
      FAIR_STORAGE_BANDWIDTH_MIB_S
      SLLM_ZIPF_EXPONENT
      SLLM_ACTIVE_ADAPTER_CAP
      SLLM_HOTSET_ROTATION_REQUESTS
      SLLM_HOTSET_ROTATION_MODE
      SLLM_HOTSET_OVERLAP_FRACTION
      FAIR_FAASLORA_SCENARIO
      FAIR_ROUND_EXECUTION_ORDER
      FAIR_CAMPAIGN_KIND
      FAIR_FORMAL_RUN
      FAIR_TRACE_ROLE
      FAIR_RESOLVED_CONFIG_FAMILY
      FAIR_RESOLVED_CONFIG_REGISTRY
    )
    local current=(
      "${RUN_TAG}"
      "${MODEL_PROFILE}"
      "${DATASET_PROFILE}"
      "${WORKLOAD_PROFILE}"
      "${TOTAL_REQUESTS}"
      "${SELECTED_NUM_ADAPTERS}"
      "${SAMPLING_SEED}"
      "${TIME_SCALE_FACTOR}"
      "${GENERATION_CONTRACT}"
      "${FIXED_OUTPUT_MAX_TOKENS}"
      "${FIXED_PROMPT_MAX_TOKENS}"
      "${STORAGE_BANDWIDTH_MIB_S}"
      "${ZIPF_EXPONENT}"
      "${ACTIVE_ADAPTER_CAP}"
      "${HOTSET_ROTATION_REQUESTS}"
      "${HOTSET_ROTATION_MODE}"
      "${HOTSET_OVERLAP_FRACTION}"
      "${FAASLORA_SCENARIO}"
      "${EXECUTION_ORDER}"
      "${CAMPAIGN_KIND}"
      "${FORMAL_RUN}"
      "${TRACE_ROLE}"
      "${RESOLVED_CONFIG_FAMILY}"
      "${RESOLVED_CONFIG_REGISTRY}"
    )
    local i
    for i in "${!names[@]}"; do
      if [[ "${existing[$i]:-}" != "${current[$i]}" ]]; then
        echo "[ERROR] ${ROUND_DIR} already has prepared shared artifacts, but ${names[$i]} differs." >&2
        echo "        round.env has '${existing[$i]:-}', current run requested '${current[$i]}'." >&2
        echo "        Resume with the original round.env values, or use FAIR_ROUND_FORCE=1 with a clean/new round directory." >&2
        exit 1
      fi
    done
    return 0
  fi
  write_round_env
}

validate_or_write_round_env

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

validate_trace_role_and_formal_sources() {
  PYTHONPATH="${BASELINES_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 - "${BASELINES_ROOT}" "${MAIN_REPO}" "${SAMPLING_SEED}" \
      "${TRACE_ROLE}" "${FAASLORA_SCENARIO}" "${TOTAL_REQUESTS}" \
      "${FORMAL_RUN}" <<'PY'
import sys
from pathlib import Path

from fair_system_resolved_config import (
    require_formal_source_cleanliness,
    source_cleanliness,
    validate_trace_role,
)

baselines_root = Path(sys.argv[1])
main_repo = Path(sys.argv[2])
role = validate_trace_role(int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6]))
formal_run = bool(int(sys.argv[7]))
status = (
    require_formal_source_cleanliness(baselines_root, main_repo)
    if formal_run
    else source_cleanliness(baselines_root, main_repo)
)
print(
    f"[protocol-gate] formal_run={int(formal_run)} trace_role={role} "
    f"source_clean_for_formal={str(status['source_clean_for_formal']).lower()}"
)
PY
}

prepare_resolved_config_gate() {
  python3 "${BASELINES_ROOT}/scripts/fair_system_resolved_config.py" \
    --baselines-root "${BASELINES_ROOT}" \
    --main-repo "${MAIN_REPO}" \
    --model-profile "${MODEL_PROFILE}" \
    --dataset-profile "${DATASET_PROFILE}" \
    --workload-profile "${WORKLOAD_PROFILE}" \
    --total-requests "${TOTAL_REQUESTS}" \
    --selected-num-adapters "${SELECTED_NUM_ADAPTERS}" \
    --sampling-seed "${SAMPLING_SEED}" \
    --time-scale-factor "${TIME_SCALE_FACTOR}" \
    --formal-run "${FORMAL_RUN}" \
    --trace-role "${TRACE_ROLE}" \
    --generation-contract "${GENERATION_CONTRACT}" \
    --fixed-output-max-tokens "${FIXED_OUTPUT_MAX_TOKENS}" \
    --fixed-prompt-max-tokens "${FIXED_PROMPT_MAX_TOKENS}" \
    --storage-bandwidth-mib-s "${STORAGE_BANDWIDTH_MIB_S}" \
    --zipf-exponent "${ZIPF_EXPONENT}" \
    --active-adapter-cap "${ACTIVE_ADAPTER_CAP}" \
    --hotset-rotation-requests "${HOTSET_ROTATION_REQUESTS}" \
    --hotset-rotation-mode "${HOTSET_ROTATION_MODE}" \
    --hotset-overlap-fraction "${HOTSET_OVERLAP_FRACTION}" \
    --faaslora-scenario "${FAASLORA_SCENARIO}" \
    --gpu-ids "${GPU_IDS}" \
    --configuration-family "${RESOLVED_CONFIG_FAMILY}" \
    --run-tag "${RUN_TAG}" \
    --trace-path "${TRACE_PATH}" \
    --adapter-subset-path "${ADAPTER_SUBSET_PATH}" \
    --execution-order "${EXECUTION_ORDER}" \
    --campaign-kind "${CAMPAIGN_KIND}" \
    --output "${RESOLVED_CONFIG_PATH}" \
    --registry "${RESOLVED_CONFIG_REGISTRY}"
}

resolved_config_sha256() {
  python3 - "${RESOLVED_CONFIG_PATH}" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
value = str(payload.get("system_resolved_config_sha256") or "")
if not re.fullmatch(r"[0-9a-f]{64}", value):
    raise SystemExit(f"[ERROR] invalid system_resolved_config_sha256 in {path}: {value!r}")
print(value)
PY
}

mark_successful_validation_if_complete() {
  if [[ "${FORMAL_RUN}" != "1" || "${TRACE_ROLE}" != "validation" ]]; then
    return 0
  fi
  python3 "${BASELINES_ROOT}/scripts/fair_system_resolved_config.py" \
    mark-validation-complete \
    --sidecar "${RESOLVED_CONFIG_PATH}" \
    --registry "${RESOLVED_CONFIG_REGISTRY}" \
    --manifest "${ROUND_DIR}/MANIFEST.json"
}

validate_v2_protocol_controls() {
  case "${FAASLORA_SCENARIO}" in
    faaslora_full|v2_elastic_only|v2_hit_aware_preparation|v2_hierarchical_no_coord|v2_full)
      ;;
    *)
      echo "[ERROR] unsupported FAIR_FAASLORA_SCENARIO=${FAASLORA_SCENARIO}" >&2
      return 1
      ;;
  esac
  if [[ "${FAASLORA_SCENARIO}" == v2_* && "${FORCE_RERUN}" == "1" ]]; then
    echo "[ERROR] V2 protocol forbids FAIR_ROUND_FORCE=1; use a fresh unique round directory." >&2
    return 1
  fi

  local allowed=" sglang serverlessllm vllm slora faaslora "
  local seen=" "
  local system=""
  for system in ${EXECUTION_ORDER}; do
    if [[ "${allowed}" != *" ${system} "* ]]; then
      echo "[ERROR] unknown system in FAIR_ROUND_EXECUTION_ORDER: ${system}" >&2
      return 1
    fi
    if [[ "${seen}" == *" ${system} "* ]]; then
      echo "[ERROR] duplicate system in FAIR_ROUND_EXECUTION_ORDER: ${system}" >&2
      return 1
    fi
    seen+="${system} "
  done
  for system in ${SYSTEMS}; do
    if [[ "${allowed}" != *" ${system} "* ]]; then
      echo "[ERROR] unknown system in FAIR_ROUND_SYSTEMS: ${system}" >&2
      return 1
    fi
    if [[ "${seen}" != *" ${system} "* ]]; then
      echo "[ERROR] selected system ${system} is absent from FAIR_ROUND_EXECUTION_ORDER" >&2
      return 1
    fi
  done

  PYTHONPATH="${BASELINES_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 - "${CAMPAIGN_KIND}" "${FORMAL_RUN}" "${TRACE_ROLE}" \
      "${MODEL_PROFILE}" "${SAMPLING_SEED}" "${FAASLORA_SCENARIO}" \
      "${SYSTEMS}" "${EXECUTION_ORDER}" "${GENERATION_CONTRACT}" <<'PY'
import sys

from fair_system_resolved_config import validate_campaign_protocol

result = validate_campaign_protocol(
    campaign_kind=sys.argv[1],
    formal_run=bool(int(sys.argv[2])),
    trace_role=sys.argv[3],
    model_profile=sys.argv[4],
    sampling_seed=int(sys.argv[5]),
    faaslora_scenario=sys.argv[6],
    systems=sys.argv[7].split(),
    execution_order=sys.argv[8].split(),
    generation_contract=sys.argv[9],
)
print(
    "[campaign-gate] "
    f"kind={result['campaign_kind'] or 'exploratory'} "
    f"publication_protocol_enforced="
    f"{str(result['publication_protocol_enforced']).lower()}"
)
PY
}

default_client_timeout_s() {
  case "${MODEL_PROFILE}" in
    *13b*|*14b*|*13B*|*14B*)
      printf '%s\n' 21600
      ;;
    *)
      printf '%s\n' 3600
      ;;
  esac
}

stage_done_path() {
  printf '%s/%s.done\n' "${STATE_DIR}" "$1"
}

stage_unsupported_path() {
  printf '%s/%s.unsupported\n' "${STATE_DIR}" "$1"
}

is_done() {
  [[ "${FORCE_RERUN}" != "1" ]] && {
    [[ -f "$(stage_done_path "$1")" ]] || [[ -f "$(stage_unsupported_path "$1")" ]]
  }
}

mark_done() {
  date '+%F %T' >"$(stage_done_path "$1")"
}

stage_for_system() {
  case "$1" in
    sglang) printf '10_sglang\n' ;;
    serverlessllm) printf '20_serverlessllm\n' ;;
    vllm) printf '30_vllm\n' ;;
    slora) printf '40_slora\n' ;;
    faaslora) printf '50_faaslora\n' ;;
    *)
      printf 'unknown_%s\n' "$1"
      ;;
  esac
}

unsupported_reason_for_system() {
  local system="$1"
  if [[ "${system}" == "slora" && "${MODEL_PROFILE}" == qwen_* ]]; then
    printf 'S-LoRA upstream only provides Llama/Llama2 model backends in this harness; Qwen-family profiles expose model_type=qwen2 and are unsupported without a new core model implementation.'
    return 0
  fi
  return 1
}

system_supported() {
  local system="$1"
  if unsupported_reason_for_system "${system}" >/dev/null; then
    return 1
  fi
  return 0
}

mark_unsupported_system() {
  local system="$1"
  local stage=""
  local reason=""
  stage="$(stage_for_system "${system}")"
  reason="$(unsupported_reason_for_system "${system}")"
  log "skip ${stage}; ${reason}"
  {
    printf 'timestamp=%s\n' "$(date '+%F %T')"
    printf 'system=%s\n' "${system}"
    printf 'model_profile=%s\n' "${MODEL_PROFILE}"
    printf 'reason=%s\n' "${reason}"
  } >"$(stage_unsupported_path "${stage}")"
}

selected_supported_systems() {
  local system=""
  for system in "$@"; do
    case " ${SYSTEMS} " in
      *" ${system} "*)
        if system_supported "${system}"; then
          printf '%s\n' "${system}"
        fi
        ;;
    esac
  done
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
    if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      echo "[ERROR] listener remains on port=${port}, but its PID is not visible to the current user." >&2
      echo "        It is likely owned by root or a container; stop it before running a fair round." >&2
      return 1
    fi
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
  sleep 1
  if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    echo "[ERROR] listener remains on port=${port} after cleanup." >&2
    echo "        Stop the owning process before running a fair round." >&2
    return 1
  fi
}

cleanup_known_leftovers() {
  log "cleaning known leftover services"
  bash "${BASELINES_ROOT}/scripts/stop_serverlessllm_stack.sh" >/dev/null 2>&1 || true
  local ports=(
    8353
    8363 8373 8383 8393
    8463 8473 8483 8493
    8000 8001 8002 8003
    8080
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
  local query_output=""
  IFS=',' read -r -a gpu_ids <<< "${gpu_csv}"
  for gpu in "${gpu_ids[@]}"; do
    gpu="$(printf '%s' "${gpu}" | xargs)"
    [[ -z "${gpu}" ]] && continue
    if ! query_output="$(nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
      continue
    fi
    printf '%s\n' "${query_output}" \
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
  local query_output=""
  IFS=',' read -r -a gpu_ids <<< "${gpu_csv}"
  for gpu in "${gpu_ids[@]}"; do
    gpu="$(printf '%s' "${gpu}" | xargs)"
    [[ -z "${gpu}" ]] && continue
    if ! query_output="$(nvidia-smi --id="${gpu}" --query-compute-apps=pid,used_gpu_memory,process_name --format=csv,noheader,nounits 2>/dev/null)"; then
      continue
    fi
    while IFS=',' read -r pid used_mem process_name; do
      pid="$(printf '%s' "${pid}" | xargs)"
      used_mem="$(printf '%s' "${used_mem}" | xargs)"
      process_name="$(printf '%s' "${process_name}" | xargs)"
      [[ -z "${pid}" ]] && continue
      local user=""
      local cmd=""
      user="$(ps -p "${pid}" -o user= 2>/dev/null | xargs || true)"
      cmd="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
      if (( ${#cmd} > 240 )); then
        cmd="${cmd:0:240}..."
      fi
      if [[ -n "${cmd}" ]]; then
        printf 'gpu=%s pid=%s mem=%sMiB process=%s user=%s cmd=%s\n' "${gpu}" "${pid}" "${used_mem}" "${process_name}" "${user:-unknown}" "${cmd}"
      else
        printf 'gpu=%s pid=%s mem=%sMiB process=%s user=%s cmd=<unavailable>\n' "${gpu}" "${pid}" "${used_mem}" "${process_name}" "${user:-unknown}"
      fi
    done < <(printf '%s\n' "${query_output}" | sed '/^[[:space:]]*$/d')
  done
}

wait_gpu_idle() {
  if [[ "${STRICT_GPU_IDLE}" != "1" ]]; then
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi is unavailable; cannot verify GPU idle state under strict mode." >&2
    echo "        Check the NVIDIA driver, or set FAIR_ROUND_STRICT_GPU_IDLE=0 only for diagnosis." >&2
    return 1
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

ROUND_EXIT_CLEANUP_RUNNING=0
cleanup_on_round_exit() {
  local status=$?
  if (( ROUND_EXIT_CLEANUP_RUNNING != 0 )); then
    return "${status}"
  fi
  ROUND_EXIT_CLEANUP_RUNNING=1
  set +e
  cleanup_known_leftovers >/dev/null 2>&1 || true
  ROUND_EXIT_CLEANUP_RUNNING=0
  return "${status}"
}
trap cleanup_on_round_exit EXIT INT TERM HUP

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

latest_summary_match() {
  local pattern="$1"
  local fallback="$2"
  python3 - "${pattern}" "${fallback}" <<'PY'
import glob
import os
import sys

pattern = sys.argv[1]
fallback = sys.argv[2]
matches = [path for path in glob.glob(pattern) if os.path.isfile(path)]
if not matches:
    print(fallback)
else:
    matches.sort(key=lambda path: os.path.getmtime(path))
    print(matches[-1])
PY
}

summary_path_for_system() {
  case "$1" in
    sglang) latest_summary_match "${RAW_REPLAY_DIR}/${RUN_TAG}_sglang_*_summary.json" "${RAW_REPLAY_DIR}/${RUN_TAG}_sglang_dp4_tp1_summary.json" ;;
    serverlessllm) printf '%s/%s_serverlessllm_summary.json\n' "${RAW_REPLAY_DIR}" "${RUN_TAG}" ;;
    vllm) latest_summary_match "${RAW_REPLAY_DIR}/${RUN_TAG}_vllm_*_summary.json" "${RAW_REPLAY_DIR}/${RUN_TAG}_vllm_dp4_tp1_summary.json" ;;
    slora) latest_summary_match "${RAW_REPLAY_DIR}/${RUN_TAG}_slora_*_summary.json" "${RAW_REPLAY_DIR}/${RUN_TAG}_slora_dp4_tp1_summary.json" ;;
    faaslora) printf '%s/%s_faaslora_result.json\n' "${RAW_FAAS_DIR}" "${RUN_TAG}" ;;
    *) return 1 ;;
  esac
}

validate_summary() {
  local system="$1"
  local path="$2"
  local expected_total="${TOTAL_REQUESTS}"
  if [[ "${system}" == "SGLang" && "${SGLANG_MAX_REPLAY_REQUESTS:-0}" != "0" ]]; then
    expected_total="${SGLANG_MAX_REPLAY_REQUESTS}"
  fi
  python3 - "${system}" "${path}" "${expected_total}" "${TRACE_PATH}" "${ADAPTER_SUBSET_PATH}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

system = sys.argv[1]
path = Path(sys.argv[2])
expected_total = int(sys.argv[3])
trace_path = Path(sys.argv[4]).resolve()
subset_path = Path(sys.argv[5]).resolve()

def sha256(candidate: Path) -> str:
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if not path.exists():
    raise SystemExit(f"[ERROR] missing summary for {system}: {path}")
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("metric_schema_version") != "e2e_v3":
    raise SystemExit(f"[ERROR] {system} metric_schema_version is not e2e_v3: {data.get('metric_schema_version')}")
if system == "ServerlessLLM":
    metadata = data.get("metadata") or {}
    for field, expected in (
        ("shared_trace_sha256", sha256(trace_path)),
        ("shared_adapter_subset_sha256", sha256(subset_path)),
    ):
        observed = str(metadata.get(field) or "")
        if observed != expected:
            raise SystemExit(
                f"[ERROR] ServerlessLLM {field} mismatch: "
                f"expected={expected} observed={observed or '<missing>'}"
            )
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
if total < expected_total:
    raise SystemExit(f"[ERROR] {system} total mismatch: expected at least {expected_total}, got {total}")
if completed != expected_total or failed != 0:
    raise SystemExit(
        f"[ERROR] {system} incomplete result: completed={completed}, expected_completed={expected_total}, total={total}, failed={failed}"
    )
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
print(f"[validated] {system}: completed={completed}/{expected_total} trace_total={total} path={path}")
PY
}

validate_faaslora_generation_contract() {
  local path="$1"
  if [[ "${GENERATION_CONTRACT}" != "fixed_length_greedy_v1" ]]; then
    return 0
  fi
  python3 - "${path}" "${TOTAL_REQUESTS}" "${FIXED_OUTPUT_MAX_TOKENS}" "${FIXED_PROMPT_MAX_TOKENS}" <<'PY'
import hashlib
import json
import math
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_total = int(sys.argv[2])
output_cap = int(sys.argv[3])
prompt_cap = int(sys.argv[4])
data = json.loads(path.read_text(encoding="utf-8"))
metadata = data.get("metadata") or {}
contract = str(metadata.get("generation_contract") or "")
if contract != "fixed_length_greedy_v1":
    raise SystemExit(
        f"[ERROR] FaaSLoRA generation_contract mismatch: {contract!r}"
    )
if int(metadata.get("fixed_output_max_tokens") or 0) != output_cap:
    raise SystemExit("[ERROR] FaaSLoRA fixed output cap mismatch")
if int(metadata.get("fixed_prompt_max_tokens") or 0) != prompt_cap:
    raise SystemExit("[ERROR] FaaSLoRA fixed prompt cap mismatch")

detailed = data.get("detailed_results") or {}
if len(detailed) != 1:
    raise SystemExit(
        f"[ERROR] fixed-contract FaaSLoRA result must contain one scenario, got {list(detailed)}"
    )
scenario_name, scenario = next(iter(detailed.items()))
requests = list((scenario or {}).get("requests") or [])
if len(requests) != expected_total:
    raise SystemExit(
        f"[ERROR] FaaSLoRA fixed-contract request count mismatch: "
        f"observed={len(requests)} expected={expected_total}"
    )

sha256_re = re.compile(r"^[0-9a-f]{64}$")
errors = []
seen = set()
for request in requests:
    request_id = str(request.get("request_id") or "")
    prefix = f"request_id={request_id or '<missing>'}"
    if not request_id or request_id in seen:
        errors.append(f"{prefix}: missing or duplicate request id")
    seen.add(request_id)
    if not bool(request.get("success")):
        errors.append(f"{prefix}: request is not successful")
    if str(request.get("generation_contract") or "") != contract:
        errors.append(f"{prefix}: generation contract mismatch")
    try:
        source = int(request.get("source_expected_output_tokens"))
        requested = int(request.get("requested_completion_tokens"))
        completion = int(request.get("completion_tokens"))
        prompt_tokens = int(request.get("canonical_prompt_tokens"))
    except (TypeError, ValueError):
        errors.append(f"{prefix}: invalid token-count field")
        continue
    expected_target = min(source, output_cap)
    if source <= 0 or requested != expected_target or completion != requested:
        errors.append(
            f"{prefix}: source/requested/completion mismatch "
            f"source={source} requested={requested} completion={completion}"
        )
    if prompt_tokens <= 0 or prompt_tokens > prompt_cap:
        errors.append(f"{prefix}: prompt tokens outside [1, {prompt_cap}]")
    if str(request.get("completion_token_source") or "") != "vllm_token_ids":
        errors.append(f"{prefix}: completion token source is not vllm_token_ids")
    if request.get("output_contract_match") is not True:
        errors.append(f"{prefix}: output_contract_match is not true")
    for field in ("canonical_prompt_sha256", "completion_token_ids_sha256"):
        if not sha256_re.fullmatch(str(request.get(field) or "")):
            errors.append(f"{prefix}: invalid {field}")

    def finite(field):
        try:
            value = float(request.get(field))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    e2e = finite("e2e_ms")
    dispatch = finite("dispatch_admission_wait_ms")
    service_e2e = finite("service_e2e_ms")
    if e2e is None or dispatch is None or service_e2e is None:
        errors.append(f"{prefix}: missing E2E decomposition")
    elif abs(e2e - (dispatch + service_e2e)) > 1.0:
        errors.append(f"{prefix}: E2E decomposition error exceeds 1 ms")

    if completion > 1:
        tpot = finite("tpot_ms")
        service_e2e_for_tpot = finite("service_e2e_ms")
        service_ttft_for_tpot = finite("service_ttft_ms")
        if tpot is None or service_e2e_for_tpot is None or service_ttft_for_tpot is None:
            errors.append(f"{prefix}: missing TPOT recomputation fields")
        else:
            recomputed = max(
                0.0, service_e2e_for_tpot - service_ttft_for_tpot
            ) / (completion - 1)
            if abs(tpot - recomputed) > 1.0:
                errors.append(f"{prefix}: TPOT recomputation error exceeds 1 ms")

request_map = [
    {
        "request_id": request.get("request_id"),
        "adapter_id": request.get("adapter_id"),
        "arrival_time_s": request.get("scheduled_arrival_offset_s"),
        "source_expected_output_tokens": request.get("source_expected_output_tokens"),
        "requested_completion_tokens": request.get("requested_completion_tokens"),
        "canonical_prompt_sha256": request.get("canonical_prompt_sha256"),
        "canonical_prompt_tokens": request.get("canonical_prompt_tokens"),
    }
    for request in requests
]
expected_map_sha = hashlib.sha256(
    json.dumps(
        request_map,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
observed_map_sha = str(
    (metadata.get("generation_contract_request_map_sha256") or {}).get(scenario_name)
    or ""
)
if observed_map_sha != expected_map_sha:
    errors.append("generation contract request-map SHA-256 mismatch")

if errors:
    print(
        f"[ERROR] FaaSLoRA fixed generation contract failed: errors={len(errors)}",
        file=sys.stderr,
    )
    for error in errors[:20]:
        print(f"  {error}", file=sys.stderr)
    raise SystemExit(1)
print(
    f"[check] FaaSLoRA fixed generation contract: {len(requests)} requests "
    "match target/token/hash/decomposition gates."
)
PY
}

validate_faaslora_identity() {
  local path="$1"
  local expected_tag="$2"
  local expected_settings_sha="$3"
  python3 - "${path}" "${FAASLORA_SCENARIO}" "${expected_tag}" \
    "${TRACE_PATH}" "${ADAPTER_SUBSET_PATH}" "${TOTAL_REQUESTS}" \
    "${expected_settings_sha}" "${STORAGE_BANDWIDTH_MIB_S}" \
    "$(resolved_config_sha256)" "${TRACE_ROLE}" "${FORMAL_RUN}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_scenario = sys.argv[2]
expected_tag = sys.argv[3]
trace_path = Path(sys.argv[4])
subset_path = Path(sys.argv[5])
expected_total = int(sys.argv[6])
expected_settings_sha = sys.argv[7]
expected_bandwidth_mib_s = float(sys.argv[8])
expected_resolved_config_sha = sys.argv[9]
expected_trace_role = sys.argv[10]
expected_formal_run = bool(int(sys.argv[11]))

def sha256(candidate: Path) -> str:
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

data = json.loads(path.read_text(encoding="utf-8"))
metadata = data.get("metadata") or {}
if str(metadata.get("results_tag") or "") != expected_tag:
    raise SystemExit(
        f"[ERROR] FaaSLoRA result tag mismatch: expected={expected_tag!r} "
        f"observed={metadata.get('results_tag')!r}"
    )
for field, expected in (
    ("shared_trace_sha256", sha256(trace_path)),
    ("shared_adapter_subset_sha256", sha256(subset_path)),
):
    observed = str(metadata.get(field) or "")
    if observed != expected:
        raise SystemExit(
            f"[ERROR] FaaSLoRA {field} mismatch: expected={expected} observed={observed or '<missing>'}"
        )
if int(metadata.get("total_requests") or -1) != expected_total:
    raise SystemExit("[ERROR] FaaSLoRA metadata total_requests mismatch")
if str(metadata.get("run_frozen_settings_sha256") or "") != expected_settings_sha:
    raise SystemExit("[ERROR] FaaSLoRA frozen-settings SHA-256 mismatch")
if str(metadata.get("system_resolved_config_sha256") or "") != expected_resolved_config_sha:
    raise SystemExit("[ERROR] FaaSLoRA system-resolved-config SHA-256 mismatch")
if str(metadata.get("trace_role") or "") != expected_trace_role:
    raise SystemExit("[ERROR] FaaSLoRA trace_role metadata mismatch")
if metadata.get("formal_run") is not expected_formal_run:
    raise SystemExit("[ERROR] FaaSLoRA formal_run metadata mismatch")
if float(metadata.get("bandwidth_mib_s", -1)) != expected_bandwidth_mib_s:
    raise SystemExit("[ERROR] FaaSLoRA bandwidth metadata mismatch")

for key in ("scenario_summaries", "detailed_results"):
    names = sorted((data.get(key) or {}).keys())
    if names != [expected_scenario]:
        raise SystemExit(
            f"[ERROR] FaaSLoRA {key} must contain exactly {expected_scenario!r}; got {names}"
        )
coordination = (metadata.get("scenario_coordination") or {}).get(expected_scenario) or {}
if expected_scenario.startswith("v2_") and not bool(
    coordination.get("cold_cache_reset_before_run")
):
    raise SystemExit("[ERROR] V2 FaaSLoRA result lacks cold-cache reset evidence")
print(
    f"[check] FaaSLoRA identity: scenario={expected_scenario} tag={expected_tag} "
    "trace/subset/frozen-settings/resolved-config SHA-256 matched"
)
PY
}

trace_remote_endpoint() {
  python3 - "${TRACE_PATH}" <<'PY'
import json
import sys
from pathlib import Path

trace_path = Path(sys.argv[1])
payload = json.loads(trace_path.read_text(encoding="utf-8"))
remote_dir = str(payload.get("remote_dir") or "").strip()
if not remote_dir:
    raise SystemExit(f"[ERROR] trace has no remote_dir: {trace_path}")
path = Path(remote_dir).expanduser().resolve()
if not path.is_dir():
    raise SystemExit(f"[ERROR] trace remote_dir is not a directory: {path}")
print(path.as_uri())
PY
}

faaslora_frozen_settings_sha256() {
  python3 - \
    "${MAIN_REPO}" "${BASELINES_ROOT}" "${TRACE_PATH}" "${ADAPTER_SUBSET_PATH}" \
    "${MODEL_PROFILE}" "${DATASET_PROFILE}" "${WORKLOAD_PROFILE}" \
    "${TOTAL_REQUESTS}" "${SELECTED_NUM_ADAPTERS}" "${SAMPLING_SEED}" \
    "${TIME_SCALE_FACTOR}" "${GENERATION_CONTRACT}" "${FIXED_OUTPUT_MAX_TOKENS}" \
    "${FIXED_PROMPT_MAX_TOKENS}" "${STORAGE_BANDWIDTH_MIB_S}" \
    "${ZIPF_EXPONENT}" "${ACTIVE_ADAPTER_CAP}" "${HOTSET_ROTATION_REQUESTS}" \
    "${HOTSET_ROTATION_MODE}" "${HOTSET_OVERLAP_FRACTION}" \
    "${FAASLORA_SCENARIO}" "${EXECUTION_ORDER}" "${GPU_IDS}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

main_repo = Path(sys.argv[1])
baselines_root = Path(sys.argv[2])

def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def git_commit(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()

payload = {
    "faaslora_git_commit": git_commit(main_repo),
    "baseline_git_commit": git_commit(baselines_root),
    "faaslora_config_sha256": file_sha(main_repo / "configs" / "experiments.yaml"),
    "shared_trace_sha256": file_sha(sys.argv[3]),
    "shared_adapter_subset_sha256": file_sha(sys.argv[4]),
    "model_profile": sys.argv[5],
    "dataset_profile": sys.argv[6],
    "workload_profile": sys.argv[7],
    "total_requests": int(sys.argv[8]),
    "selected_num_adapters": int(sys.argv[9]),
    "sampling_seed": int(sys.argv[10]),
    "time_scale_factor": sys.argv[11] or None,
    "generation_contract": sys.argv[12],
    "fixed_output_max_tokens": int(sys.argv[13]),
    "fixed_prompt_max_tokens": int(sys.argv[14]),
    "storage_bandwidth_mib_s": float(sys.argv[15]),
    "zipf_exponent": sys.argv[16] or None,
    "active_adapter_cap": sys.argv[17] or None,
    "hotset_rotation_requests": sys.argv[18] or None,
    "hotset_rotation_mode": sys.argv[19] or None,
    "hotset_overlap_fraction": sys.argv[20] or None,
    "faaslora_scenario": sys.argv[21],
    "execution_order": sys.argv[22].split(),
    "gpu_ids": sys.argv[23],
}
canonical = json.dumps(
    payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
).encode("utf-8")
print(hashlib.sha256(canonical).hexdigest())
PY
}

validate_serverlessllm_bandwidth_evidence() {
  local path="$1"
  python3 - "${path}" "${STORAGE_BANDWIDTH_MIB_S}" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
configured = float(sys.argv[2])
data = json.loads(path.read_text(encoding="utf-8"))
metadata = data.get("metadata") or {}
aggregate = metadata.get("aggregate_bandwidth") or {}
observed = float(aggregate.get("configured_mib_s", -1))
if not math.isclose(observed, configured, rel_tol=0.0, abs_tol=1e-9):
    raise SystemExit(
        f"[ERROR] ServerlessLLM bandwidth mismatch: expected={configured} observed={observed}"
    )
transfer_count = int(aggregate.get("transfer_count", 0) or 0)
total_bytes = int(aggregate.get("total_bytes", 0) or 0)
mode = str(aggregate.get("limit_mode") or "")
expected_mode = "file_no_delay" if configured == 0.0 else "file_aggregate_reservation"
if mode != expected_mode:
    raise SystemExit(
        f"[ERROR] ServerlessLLM limiter mode mismatch: expected={expected_mode} observed={mode!r}"
    )
if transfer_count <= 0 or total_bytes <= 0:
    raise SystemExit(
        "[ERROR] ServerlessLLM dynamic remote fetch did not trigger; refusing bandwidth evidence"
    )
print(
    f"[check] ServerlessLLM aggregate fetch: mode={mode} transfers={transfer_count} "
    f"bytes={total_bytes} configured={configured} MiB/s"
)
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
    SLLM_TIME_SCALE_FACTOR="${TIME_SCALE_FACTOR}" \
    SLLM_ZIPF_EXPONENT="${ZIPF_EXPONENT}" \
    SLLM_ACTIVE_ADAPTER_CAP="${ACTIVE_ADAPTER_CAP}" \
    SLLM_HOTSET_ROTATION_REQUESTS="${HOTSET_ROTATION_REQUESTS}" \
    SLLM_HOTSET_ROTATION_MODE="${HOTSET_ROTATION_MODE}" \
    SLLM_HOTSET_OVERLAP_FRACTION="${HOTSET_OVERLAP_FRACTION}" \
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
    SGLANG_DATA_PARALLEL_REPLICAS="${SGLANG_DATA_PARALLEL_REPLICAS:-}" \
    SGLANG_TENSOR_PARALLEL_SIZE="${SGLANG_TENSOR_PARALLEL_SIZE:-}" \
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
  local serverless_timeout_s="${SLLM_TIMEOUT_S:-$(default_client_timeout_s)}"
  local remote_endpoint="${SLLM_REMOTE_ARTIFACT_STAGE_ENDPOINT:-}"
  if [[ -z "${remote_endpoint}" ]]; then
    remote_endpoint="$(trace_remote_endpoint)"
  fi
  local remote_cache_dir="${ROUND_DIR}/cache/serverlessllm"
  log "ServerlessLLM client timeout=${serverless_timeout_s}s for model_profile=${MODEL_PROFILE}"
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
    SLLM_REMOTE_ARTIFACT_STAGE_ENDPOINT="${remote_endpoint}" \
    SLLM_REMOTE_ARTIFACT_STAGE_MODE=dynamic \
    SLLM_REMOTE_ARTIFACT_STAGE_CACHE_DIR="${remote_cache_dir}" \
    SLLM_REMOTE_ARTIFACT_STAGE_BANDWIDTH_MIB_S="${STORAGE_BANDWIDTH_MIB_S}" \
    FAIR_SYSTEM_RESOLVED_CONFIG_SHA256="$(resolved_config_sha256)" \
    FAIR_TRACE_ROLE="${TRACE_ROLE}" \
    FAIR_FORMAL_RUN="${FORMAL_RUN}" \
    SLLM_BACKEND="${SLLM_BACKEND:-vllm}" \
    SLLM_WORKER_GPUS="${GPU_IDS}" \
    SLLM_TIMEOUT_S="${serverless_timeout_s}" \
    bash "${BASELINES_ROOT}/scripts/run_serverlessllm_fair_experiment.sh"
  validate_summary "ServerlessLLM" "$(summary_path_for_system serverlessllm)"
  validate_serverlessllm_bandwidth_evidence "$(summary_path_for_system serverlessllm)"
  post_system_clean_check "ServerlessLLM"
  mark_done "${stage}"
}

run_vllm() {
  local stage="30_vllm"
  if is_done "${stage}"; then
    log "skip ${stage}; marker exists"
    return 0
  fi
  local vllm_dp="${VLLM_DATA_PARALLEL_REPLICAS:-}"
  local vllm_tp="${VLLM_TENSOR_PARALLEL_SIZE:-}"
  local vllm_max_loras="${VLLM_MAX_LORAS:-}"
  local vllm_max_num_seqs="${VLLM_MAX_NUM_SEQS:-}"
  local vllm_max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
  local vllm_max_cpu_loras="${VLLM_MAX_CPU_LORAS:-}"
  local vllm_lora_registration_mode="${VLLM_LORA_REGISTRATION_MODE:-}"
  local vllm_dynamic_lora_routing="${VLLM_DYNAMIC_LORA_ROUTING:-}"
  local vllm_dynamic_lora_hot_pair_threshold="${VLLM_DYNAMIC_LORA_HOT_PAIR_THRESHOLD:-}"
  local vllm_dynamic_lora_hot_pair_max_adapters="${VLLM_DYNAMIC_LORA_HOT_PAIR_MAX_ADAPTERS:-}"
  local vllm_dynamic_lora_max_loaded_per_endpoint="${VLLM_DYNAMIC_LORA_MAX_LOADED_PER_ENDPOINT:-}"
  local vllm_disable_frontend_mp="${VLLM_DISABLE_FRONTEND_MULTIPROCESSING:-}"
  local vllm_enforce_eager="${VLLM_ENFORCE_EAGER:-}"
  local vllm_enable_chunked_prefill="${VLLM_ENABLE_CHUNKED_PREFILL:-}"
  local vllm_enable_prefix_caching="${VLLM_ENABLE_PREFIX_CACHING:-}"
  local vllm_use_v1="${VLLM_USE_V1:-}"
  local vllm_use_flashinfer_sampler="${VLLM_USE_FLASHINFER_SAMPLER:-}"
  local vllm_attention_backend="${VLLM_ATTENTION_BACKEND:-}"
  local vllm_timeout_s="${VLLM_TIMEOUT_S:-$(default_client_timeout_s)}"
  if [[ -z "${vllm_dp}" && -z "${vllm_tp}" && "${MODEL_PROFILE}" == "qwen_7b_main_v2_publicmix" ]]; then
    # Qwen2.5-7B V2 uses the vLLM V0/eager path in the current environment.
    # Static registration preloads all sampled LoRA modules into every OpenAI
    # API replica and leaves a large host-side footprint before the replay even
    # reaches the long-tail adapter set. Keep four independent TP=1 service
    # replicas for throughput, but load LoRA modules through vLLM's runtime API
    # on first use. This preserves the 500-adapter universe and 100% LoRA-bound
    # replay while avoiding static per-replica registration of all adapters.
    vllm_dp="${VLLM_QWEN7_SAFE_DP:-4}"
    vllm_tp="${VLLM_QWEN7_SAFE_TP:-1}"
    vllm_max_num_seqs="${vllm_max_num_seqs:-${VLLM_QWEN7_SAFE_MAX_NUM_SEQS:-8}}"
    vllm_max_loras="${vllm_max_loras:-${VLLM_QWEN7_SAFE_MAX_LORAS:-8}}"
    vllm_max_num_batched_tokens="${vllm_max_num_batched_tokens:-${VLLM_QWEN7_SAFE_MAX_NUM_BATCHED_TOKENS:-4096}}"
    vllm_max_cpu_loras="${vllm_max_cpu_loras:-${VLLM_QWEN7_SAFE_MAX_CPU_LORAS:-24}}"
    vllm_lora_registration_mode="${vllm_lora_registration_mode:-${VLLM_QWEN7_LORA_REGISTRATION_MODE:-dynamic}}"
    vllm_dynamic_lora_routing="${VLLM_QWEN7_DYNAMIC_LORA_ROUTING:-adapter_hash}"
    vllm_dynamic_lora_hot_pair_threshold="${vllm_dynamic_lora_hot_pair_threshold:-${VLLM_QWEN7_DYNAMIC_LORA_HOT_PAIR_THRESHOLD:-8}}"
    vllm_dynamic_lora_hot_pair_max_adapters="${vllm_dynamic_lora_hot_pair_max_adapters:-${VLLM_QWEN7_DYNAMIC_LORA_HOT_PAIR_MAX_ADAPTERS:-32}}"
    vllm_dynamic_lora_max_loaded_per_endpoint="${vllm_dynamic_lora_max_loaded_per_endpoint:-${VLLM_QWEN7_DYNAMIC_LORA_MAX_LOADED_PER_ENDPOINT:-${vllm_max_cpu_loras}}}"
    vllm_disable_frontend_mp="${vllm_disable_frontend_mp:-${VLLM_QWEN7_DISABLE_FRONTEND_MULTIPROCESSING:-1}}"
    log "vLLM Qwen2.5-7B safe topology override: dp=${vllm_dp} tp=${vllm_tp} max_num_seqs=${vllm_max_num_seqs} max_loras=${vllm_max_loras} max_cpu_loras=${vllm_max_cpu_loras} max_batched_tokens=${vllm_max_num_batched_tokens} lora_registration_mode=${vllm_lora_registration_mode} dynamic_lora_routing=${vllm_dynamic_lora_routing} hot_pair_threshold=${vllm_dynamic_lora_hot_pair_threshold} hot_pair_max=${vllm_dynamic_lora_hot_pair_max_adapters} dynamic_lora_max_loaded_per_endpoint=${vllm_dynamic_lora_max_loaded_per_endpoint} disable_frontend_mp=${vllm_disable_frontend_mp} on gpu_ids=${GPU_IDS}"
  fi
  if [[ -z "${vllm_dp}" && -z "${vllm_tp}" && "${MODEL_PROFILE}" == llama32_*_modelscope ]]; then
    # Llama-3.2 small-model profiles are evaluated with the same 500-adapter
    # universe as the other systems. Static --lora-modules registration exposes
    # all sampled adapters to every OpenAI API replica and leaves a large
    # host-side footprint throughout the replay; on the 4x3090 testbed this can
    # reduce MemAvailable to the fail-fast guard even though requests are still
    # succeeding. Use vLLM's runtime LoRA API instead, preserving the trace,
    # adapter subset, DP/TP topology, and per-replica serving caps while loading
    # only the adapters that are actually reached by each endpoint.
    vllm_dp="${VLLM_LLAMA32_SAFE_DP:-4}"
    vllm_tp="${VLLM_LLAMA32_SAFE_TP:-1}"
    vllm_max_num_seqs="${vllm_max_num_seqs:-${VLLM_LLAMA32_SAFE_MAX_NUM_SEQS:-8}}"
    vllm_max_loras="${vllm_max_loras:-${VLLM_LLAMA32_SAFE_MAX_LORAS:-8}}"
    vllm_max_num_batched_tokens="${vllm_max_num_batched_tokens:-${VLLM_LLAMA32_SAFE_MAX_NUM_BATCHED_TOKENS:-4096}}"
    vllm_max_cpu_loras="${vllm_max_cpu_loras:-${VLLM_LLAMA32_SAFE_MAX_CPU_LORAS:-24}}"
    vllm_lora_registration_mode="${vllm_lora_registration_mode:-${VLLM_LLAMA32_LORA_REGISTRATION_MODE:-dynamic}}"
    vllm_dynamic_lora_routing="${VLLM_LLAMA32_DYNAMIC_LORA_ROUTING:-adapter_hash}"
    vllm_dynamic_lora_hot_pair_threshold="${vllm_dynamic_lora_hot_pair_threshold:-${VLLM_LLAMA32_DYNAMIC_LORA_HOT_PAIR_THRESHOLD:-8}}"
    vllm_dynamic_lora_hot_pair_max_adapters="${vllm_dynamic_lora_hot_pair_max_adapters:-${VLLM_LLAMA32_DYNAMIC_LORA_HOT_PAIR_MAX_ADAPTERS:-32}}"
    vllm_dynamic_lora_max_loaded_per_endpoint="${vllm_dynamic_lora_max_loaded_per_endpoint:-${VLLM_LLAMA32_DYNAMIC_LORA_MAX_LOADED_PER_ENDPOINT:-${vllm_max_cpu_loras}}}"
    vllm_disable_frontend_mp="${vllm_disable_frontend_mp:-${VLLM_LLAMA32_DISABLE_FRONTEND_MULTIPROCESSING:-1}}"
    log "vLLM Llama-3.2 safe topology override: dp=${vllm_dp} tp=${vllm_tp} max_num_seqs=${vllm_max_num_seqs} max_loras=${vllm_max_loras} max_cpu_loras=${vllm_max_cpu_loras} max_batched_tokens=${vllm_max_num_batched_tokens} lora_registration_mode=${vllm_lora_registration_mode} dynamic_lora_routing=${vllm_dynamic_lora_routing} hot_pair_threshold=${vllm_dynamic_lora_hot_pair_threshold} hot_pair_max=${vllm_dynamic_lora_hot_pair_max_adapters} dynamic_lora_max_loaded_per_endpoint=${vllm_dynamic_lora_max_loaded_per_endpoint} disable_frontend_mp=${vllm_disable_frontend_mp} on gpu_ids=${GPU_IDS}"
  fi
  if [[ -z "${vllm_dp}" && -z "${vllm_tp}" && "${MODEL_PROFILE}" == "llama2_13b_tp2_v2_publicmix" ]]; then
    # Llama-2-13B uses two TP=2 vLLM replicas on the four-GPU testbed.  The
    # V0 eager path is correct but excessively serializes long-prefill
    # multi-LoRA replay at this scale.  Use the same static LoRA universe for
    # fairness, while allowing vLLM's graph/chunked-prefill path so the baseline
    # is not penalized by a conservative smoke-test configuration.  The earlier
    # seq/lora=2 and seq/lora=4 envelopes left each backend with low KV-cache
    # occupancy and persistent internal Pending queues on the fixed s8 replay.
    # A same-load 1000-request probe that crosses the 500-request hot-set
    # rotation stayed failure-free at seq/lora=8, with vLLM reporting maximum
    # concurrency 11.69x for 1024-token requests.  Therefore seq/lora=8 is the
    # current stable 13B serving envelope on this 4x3090 testbed.
    vllm_dp="${VLLM_LLAMA13B_SAFE_DP:-2}"
    vllm_tp="${VLLM_LLAMA13B_SAFE_TP:-2}"
    vllm_max_num_seqs="${vllm_max_num_seqs:-${VLLM_LLAMA13B_SAFE_MAX_NUM_SEQS:-8}}"
    vllm_max_loras="${vllm_max_loras:-${VLLM_LLAMA13B_SAFE_MAX_LORAS:-8}}"
    vllm_max_num_batched_tokens="${vllm_max_num_batched_tokens:-${VLLM_LLAMA13B_SAFE_MAX_NUM_BATCHED_TOKENS:-4096}}"
    vllm_max_cpu_loras="${vllm_max_cpu_loras:-${VLLM_LLAMA13B_SAFE_MAX_CPU_LORAS:-64}}"
    vllm_lora_registration_mode="${vllm_lora_registration_mode:-${VLLM_LLAMA13B_LORA_REGISTRATION_MODE:-static}}"
    vllm_enforce_eager="${vllm_enforce_eager:-${VLLM_LLAMA13B_ENFORCE_EAGER:-0}}"
    vllm_enable_chunked_prefill="${vllm_enable_chunked_prefill:-${VLLM_LLAMA13B_ENABLE_CHUNKED_PREFILL:-1}}"
    vllm_enable_prefix_caching="${vllm_enable_prefix_caching:-${VLLM_LLAMA13B_ENABLE_PREFIX_CACHING:-0}}"
    log "vLLM Llama-2-13B topology override: dp=${vllm_dp} tp=${vllm_tp} max_num_seqs=${vllm_max_num_seqs} max_loras=${vllm_max_loras} max_cpu_loras=${vllm_max_cpu_loras} max_batched_tokens=${vllm_max_num_batched_tokens} lora_registration_mode=${vllm_lora_registration_mode} enforce_eager=${vllm_enforce_eager} chunked_prefill=${vllm_enable_chunked_prefill} prefix_caching=${vllm_enable_prefix_caching} on gpu_ids=${GPU_IDS}"
  fi
  if [[ -z "${vllm_lora_registration_mode}" ]]; then
    if [[ "${MODEL_PROFILE}" == qwen_* || "${MODEL_PROFILE}" == llama32_*_modelscope ]]; then
      vllm_lora_registration_mode="dynamic"
    else
      vllm_lora_registration_mode="static"
    fi
  fi
  if [[ -z "${vllm_dynamic_lora_routing}" ]]; then
    if [[ "${vllm_lora_registration_mode}" == "dynamic" && ( "${MODEL_PROFILE}" == qwen_* || "${MODEL_PROFILE}" == llama32_*_modelscope ) ]]; then
      vllm_dynamic_lora_routing="adapter_hash"
    else
      vllm_dynamic_lora_routing="round_robin"
    fi
  fi
  vllm_dynamic_lora_hot_pair_threshold="${vllm_dynamic_lora_hot_pair_threshold:-8}"
  vllm_dynamic_lora_hot_pair_max_adapters="${vllm_dynamic_lora_hot_pair_max_adapters:-32}"
  if [[ -z "${vllm_dynamic_lora_max_loaded_per_endpoint}" ]]; then
    if [[ "${vllm_lora_registration_mode}" == "dynamic" ]]; then
      vllm_dynamic_lora_max_loaded_per_endpoint="${VLLM_QWEN_DYNAMIC_LORA_MAX_LOADED_PER_ENDPOINT:-auto}"
    else
      vllm_dynamic_lora_max_loaded_per_endpoint="auto"
    fi
  fi
  if [[ -z "${vllm_disable_frontend_mp}" ]]; then
    if [[ "${vllm_lora_registration_mode}" == "dynamic" && ( "${MODEL_PROFILE}" == qwen_* || "${MODEL_PROFILE}" == llama32_*_modelscope ) ]]; then
      # Qwen-family profiles use vLLM's V0/eager OpenAI API path in this
      # environment; Llama-3.2 small-model profiles follow the same runtime
      # LoRA path to avoid static per-replica registration of the 500-adapter
      # pool. The single-process frontend avoids an extra host-side footprint
      # while keeping the same DP/TP topology and serving caps.
      vllm_disable_frontend_mp="${VLLM_QWEN_DISABLE_FRONTEND_MULTIPROCESSING:-1}"
    else
      vllm_disable_frontend_mp="0"
    fi
  fi
  pre_system_clean_check "vLLM"
  log "vLLM client timeout=${vllm_timeout_s}s for model_profile=${MODEL_PROFILE}"
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
    VLLM_DATA_PARALLEL_REPLICAS="${vllm_dp}" \
    VLLM_TENSOR_PARALLEL_SIZE="${vllm_tp}" \
    VLLM_MAX_NUM_SEQS="${vllm_max_num_seqs}" \
    VLLM_MAX_LORAS="${vllm_max_loras}" \
    VLLM_MAX_NUM_BATCHED_TOKENS="${vllm_max_num_batched_tokens}" \
    VLLM_MAX_CPU_LORAS="${vllm_max_cpu_loras}" \
    VLLM_LORA_REGISTRATION_MODE="${vllm_lora_registration_mode}" \
    VLLM_DYNAMIC_LORA_ROUTING="${vllm_dynamic_lora_routing}" \
    VLLM_DYNAMIC_LORA_HOT_PAIR_THRESHOLD="${vllm_dynamic_lora_hot_pair_threshold}" \
    VLLM_DYNAMIC_LORA_HOT_PAIR_MAX_ADAPTERS="${vllm_dynamic_lora_hot_pair_max_adapters}" \
    VLLM_DYNAMIC_LORA_MAX_LOADED_PER_ENDPOINT="${vllm_dynamic_lora_max_loaded_per_endpoint}" \
    VLLM_DISABLE_FRONTEND_MULTIPROCESSING="${vllm_disable_frontend_mp}" \
    VLLM_ENFORCE_EAGER="${vllm_enforce_eager}" \
    VLLM_ENABLE_CHUNKED_PREFILL="${vllm_enable_chunked_prefill}" \
    VLLM_ENABLE_PREFIX_CACHING="${vllm_enable_prefix_caching}" \
    VLLM_USE_V1="${vllm_use_v1}" \
    VLLM_USE_FLASHINFER_SAMPLER="${vllm_use_flashinfer_sampler}" \
    VLLM_ATTENTION_BACKEND="${vllm_attention_backend}" \
    VLLM_TIMEOUT_S="${vllm_timeout_s}" \
    bash "${BASELINES_ROOT}/scripts/run_vllm_fair_experiment.sh"
  if [[ "${VLLM_SMOKE_ONLY:-0}" == "1" ]]; then
    log "vLLM smoke-only run completed; skipping formal summary validation"
  else
    validate_summary "vLLM" "$(summary_path_for_system vllm)"
  fi
  post_system_clean_check "vLLM"
  mark_done "${stage}"
}

run_slora() {
  local stage="40_slora"
  if is_done "${stage}"; then
    log "skip ${stage}; marker exists"
    return 0
  fi
  local slora_timeout_s="${SLORA_TIMEOUT_S:-$(default_client_timeout_s)}"
  log "S-LoRA client timeout=${slora_timeout_s}s for model_profile=${MODEL_PROFILE}"
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
    SLLM_GENERATION_CONTRACT="${GENERATION_CONTRACT}" \
    SLLM_FIXED_OUTPUT_MAX_TOKENS="${FIXED_OUTPUT_MAX_TOKENS}" \
    SLLM_FIXED_PROMPT_MAX_TOKENS="${FIXED_PROMPT_MAX_TOKENS}" \
    FAIR_SYSTEM_RESOLVED_CONFIG_SHA256="$(resolved_config_sha256)" \
    FAIR_TRACE_ROLE="${TRACE_ROLE}" \
    FAIR_FORMAL_RUN="${FORMAL_RUN}" \
    SLORA_GPU_IDS="${GPU_IDS}" \
    SLORA_DATA_PARALLEL_REPLICAS="${SLORA_DATA_PARALLEL_REPLICAS:-}" \
    SLORA_TENSOR_PARALLEL_SIZE="${SLORA_TENSOR_PARALLEL_SIZE:-}" \
    SLORA_TIMEOUT_S="${slora_timeout_s}" \
    bash "${BASELINES_ROOT}/scripts/run_slora_fair_experiment.sh"
  validate_summary "S-LoRA" "$(summary_path_for_system slora)"
  post_system_clean_check "S-LoRA"
  mark_done "${stage}"
}

find_latest_faaslora_result() {
  local tag="$1"
  # MAIN_REPO/results is a symlink in the current FaaSLoRA checkout. Use -L so
  # post-run collection follows the real result directory instead of treating the
  # symlink itself as a terminal file.
  find -L "${MAIN_REPO}/results" -maxdepth 3 -type f -name "*_${tag}.json" -printf '%T@ %p\n' 2>/dev/null \
    | sort -n \
    | awk 'END{print substr($0, index($0,$2))}'
}

collect_faaslora_result() {
  local stage="$1"
  local faas_tag="$2"
  local faas_copy="$3"
  local frozen_settings_sha="$4"
  local latest
  latest="$(find_latest_faaslora_result "${faas_tag}")"
  if [[ -z "${latest}" || ! -f "${latest}" ]]; then
    return 1
  fi
  log "collecting FaaSLoRA result: ${latest}"
  cp -f "${latest}" "${faas_copy}"
  printf '%s\n' "${latest}" >"${RAW_FAAS_DIR}/${RUN_TAG}_faaslora_source_path.txt"
  if validate_faaslora_identity "${faas_copy}" "${faas_tag}" "${frozen_settings_sha}" \
    && validate_summary "FaaSLoRA" "${faas_copy}" \
    && validate_faaslora_generation_contract "${faas_copy}"; then
    post_system_clean_check "FaaSLoRA"
    mark_done "${stage}"
    return 0
  fi
  return 1
}

run_faaslora() {
  local stage="50_faaslora"
  local frozen_settings_sha=""
  frozen_settings_sha="$(faaslora_frozen_settings_sha256)"
  local faas_tag="${RUN_TAG}_faaslora_${FAASLORA_SCENARIO}_cfg${frozen_settings_sha:0:12}"
  if [[ "${GENERATION_CONTRACT}" == "fixed_length_greedy_v1" ]]; then
    faas_tag="${RUN_TAG}_faaslora_${FAASLORA_SCENARIO}_fixedlen_greedy_v1_cfg${frozen_settings_sha:0:12}"
  fi
  local faas_copy
  faas_copy="$(summary_path_for_system faaslora)"
  if is_done "${stage}"; then
    log "skip ${stage}; marker exists"
    return 0
  fi
  if collect_faaslora_result "${stage}" "${faas_tag}" "${faas_copy}" "${frozen_settings_sha}"; then
    log "skip ${stage}; valid existing FaaSLoRA result was collected"
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
    FAASLORA_RUN_FROZEN_SETTINGS_SHA256="${frozen_settings_sha}" \
    FAASLORA_SYSTEM_RESOLVED_CONFIG_SHA256="$(resolved_config_sha256)" \
    FAASLORA_TRACE_ROLE="${TRACE_ROLE}" \
    FAASLORA_FORMAL_RUN="${FORMAL_RUN}" \
    FAASLORA_GENERATION_CONTRACT="${GENERATION_CONTRACT}" \
    FAASLORA_FIXED_OUTPUT_MAX_TOKENS="${FIXED_OUTPUT_MAX_TOKENS}" \
    FAASLORA_FIXED_PROMPT_MAX_TOKENS="${FIXED_PROMPT_MAX_TOKENS}" \
    FAASLORA_STORAGE_BANDWIDTH_MIB_S="${STORAGE_BANDWIDTH_MIB_S}" \
    FAASLORA_NVME_CACHE_DIR="${ROUND_DIR}/cache/faaslora_nvme" \
    FAASLORA_HOST_CACHE_DIR="/dev/shm/faaslora_eurosys27_v2/${RUN_TAG}_${FAASLORA_SCENARIO}" \
    FAASLORA_ZIPF_EXPONENT="${ZIPF_EXPONENT}" \
    FAASLORA_ACTIVE_ADAPTER_CAP="${ACTIVE_ADAPTER_CAP}" \
    FAASLORA_HOTSET_ROTATION_REQUESTS="${HOTSET_ROTATION_REQUESTS}" \
    FAASLORA_HOTSET_ROTATION_MODE="${HOTSET_ROTATION_MODE}" \
    FAASLORA_HOTSET_OVERLAP_FRACTION="${HOTSET_OVERLAP_FRACTION}" \
    FAASLORA_SCENARIO="${FAASLORA_SCENARIO}" \
    PYTHONUNBUFFERED=1 \
    bash "${MAIN_REPO}/scripts/run_faaslora_shared_artifact_experiment.sh" \
      --num-adapters "${SELECTED_NUM_ADAPTERS}" \
      --full-stack
  if ! collect_faaslora_result "${stage}" "${faas_tag}" "${faas_copy}" "${frozen_settings_sha}"; then
    echo "[ERROR] unable to locate FaaSLoRA result for tag=${faas_tag}" >&2
    post_system_clean_check "FaaSLoRA" || true
    return 1
  fi
}

validate_cross_system_generation_contract() {
  if [[ "${GENERATION_CONTRACT}" != "fixed_length_greedy_v1" ]]; then
    return 0
  fi
  case " ${SYSTEMS} " in
    *" slora "*) ;;
    *) return 0 ;;
  esac
  case " ${SYSTEMS} " in
    *" faaslora "*) ;;
    *) return 0 ;;
  esac
  local slora_replay=""
  local faaslora_result=""
  slora_replay="$(latest_summary_match \
    "${RAW_REPLAY_DIR}/${RUN_TAG}_slora_*fixedlen*_replay.json" "")"
  faaslora_result="$(summary_path_for_system faaslora)"
  python3 - "${slora_replay}" "${faaslora_result}" <<'PY'
import json
import sys
from pathlib import Path

slora_path = Path(sys.argv[1])
faaslora_path = Path(sys.argv[2])
if not slora_path.is_file() or not faaslora_path.is_file():
    raise SystemExit(
        f"[ERROR] missing fixed-contract cross-system input: "
        f"slora={slora_path} faaslora={faaslora_path}"
    )
slora = json.loads(slora_path.read_text(encoding="utf-8"))
faaslora = json.loads(faaslora_path.read_text(encoding="utf-8"))
slora_sha = str(slora.get("generation_contract_request_map_sha256") or "")
faas_map = (faaslora.get("metadata") or {}).get(
    "generation_contract_request_map_sha256"
) or {}
if len(faas_map) != 1:
    raise SystemExit(
        f"[ERROR] FaaSLoRA generation-contract map is not single-scenario: {faas_map}"
    )
faaslora_sha = str(next(iter(faas_map.values())) or "")
if not slora_sha or slora_sha != faaslora_sha:
    raise SystemExit(
        "[ERROR] PrimeLoRA/S-LoRA request target/prompt map mismatch: "
        f"slora={slora_sha!r} faaslora={faaslora_sha!r}"
    )
print(
    "[check] PrimeLoRA/S-LoRA fixed generation contract uses an identical "
    f"request target/prompt map: {slora_sha}"
)
PY
}

run_compare() {
  local stage="90_compare"
  if [[ "${VLLM_SMOKE_ONLY:-0}" == "1" ]]; then
    log "skip ${stage}; VLLM_SMOKE_ONLY=1 does not produce a formal summary"
    mark_done "${stage}"
    return 0
  fi
  validate_cross_system_generation_contract
  local expected_systems=""
  expected_systems="$(selected_supported_systems faaslora sglang serverlessllm vllm slora | xargs)"
  if is_done "${stage}"; then
    local compare_json="${COMPARE_DIR}/${RUN_TAG}_five_system_compare.json"
    if python3 - "${compare_json}" "${expected_systems}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
systems = sys.argv[2].split()
if not path.exists():
    raise SystemExit(1)
data = json.loads(path.read_text(encoding="utf-8"))
rows = data.get("strict_rows") or []
seen = set()
for row in rows:
    name = str(row[0]).lower() if row else ""
    if "faaslora" in name:
        seen.add("faaslora")
    elif "sglang" in name:
        seen.add("sglang")
    elif "serverlessllm" in name:
        seen.add("serverlessllm")
    elif "vllm" in name:
        seen.add("vllm")
    elif "s-lora" in name or "slora" in name:
        seen.add("slora")
missing = [system for system in systems if system not in seen]
if missing:
    raise SystemExit(1)
PY
    then
      log "skip ${stage}; marker exists and compare contains selected systems"
      return 0
    fi
    log "${stage} marker exists but compare is missing selected systems; regenerating compare"
  fi
  local compare_json="${COMPARE_DIR}/${RUN_TAG}_five_system_compare.json"
  local compare_txt="${COMPARE_DIR}/${RUN_TAG}_five_system_compare.txt"
  local log_path="${STEP_LOG_DIR}/${stage}.log"
  local args=(python3 "${BASELINES_ROOT}/scripts/compare_fair_results.py")
  local system=""
  for system in faaslora sglang serverlessllm vllm slora; do
    case " ${SYSTEMS} " in
      *" ${system} "*)
        if system_supported "${system}"; then
          args+=(--result "$(summary_path_for_system "${system}")")
        fi
        ;;
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
  local supported_systems=""
  local unsupported_systems=""
  supported_systems="$(selected_supported_systems faaslora sglang serverlessllm vllm slora | xargs)"
  unsupported_systems="$(
    for system in ${SYSTEMS}; do
      if ! system_supported "${system}"; then
        printf '%s\n' "${system}"
      fi
    done | xargs
  )"
  local frozen_settings_sha=""
  frozen_settings_sha="$(faaslora_frozen_settings_sha256)"
  python3 - "${ROUND_DIR}" "${RUN_TAG}" "${MODEL_PROFILE}" "${DATASET_PROFILE}" "${WORKLOAD_PROFILE}" "${TOTAL_REQUESTS}" "${SELECTED_NUM_ADAPTERS}" "${SAMPLING_SEED}" "${TRACE_PATH}" "${ADAPTER_SUBSET_PATH}" "${SYSTEMS}" "${supported_systems}" "${unsupported_systems}" "${GENERATION_CONTRACT}" "${FIXED_OUTPUT_MAX_TOKENS}" "${FIXED_PROMPT_MAX_TOKENS}" "${STORAGE_BANDWIDTH_MIB_S}" "${ZIPF_EXPONENT}" "${ACTIVE_ADAPTER_CAP}" "${HOTSET_ROTATION_REQUESTS}" "${HOTSET_ROTATION_MODE}" "${HOTSET_OVERLAP_FRACTION}" "${FAASLORA_SCENARIO}" "${EXECUTION_ORDER}" "${frozen_settings_sha}" "${CAMPAIGN_KIND}" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

round_dir = Path(sys.argv[1])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

trace_path = Path(sys.argv[9])
subset_path = Path(sys.argv[10])
payload = {
    "run_tag": sys.argv[2],
    "model_profile": sys.argv[3],
    "dataset_profile": sys.argv[4],
    "workload_profile": sys.argv[5],
    "total_requests": int(sys.argv[6]),
    "selected_num_adapters": int(sys.argv[7]),
    "sampling_seed": int(sys.argv[8]),
    "shared_trace_path": str(trace_path),
    "shared_trace_sha256": sha256(trace_path),
    "shared_adapter_subset_path": str(subset_path),
    "shared_adapter_subset_sha256": sha256(subset_path),
    "systems": sys.argv[11].split(),
    "supported_systems": sys.argv[12].split(),
    "unsupported_systems": sys.argv[13].split(),
    "metric_schema_version": "e2e_v3",
    "round_dir": str(round_dir),
    "state_dir": str(round_dir / "state"),
    "compare_json": str(round_dir / "compare" / f"{sys.argv[2]}_five_system_compare.json"),
    "compare_txt": str(round_dir / "compare" / f"{sys.argv[2]}_five_system_compare.txt"),
    "generation_contract": sys.argv[14],
    "fixed_output_max_tokens": int(sys.argv[15]),
    "fixed_prompt_max_tokens": int(sys.argv[16]),
    "bandwidth_mib_s": float(sys.argv[17]),
    "bandwidth_gbit_s": float(sys.argv[17]) * 1024.0 * 1024.0 * 8.0 / 1_000_000_000.0,
    "bandwidth_limit_mode": (
        "no-delay" if float(sys.argv[17]) == 0.0 else "aggregate_application_layer_local_sim"
    ),
    "workload_overrides": {
        "zipf_exponent": sys.argv[18] or None,
        "active_adapter_cap": sys.argv[19] or None,
        "hotset_rotation_requests": sys.argv[20] or None,
        "hotset_rotation_mode": sys.argv[21] or None,
        "hotset_overlap_fraction": sys.argv[22] or None,
    },
    "faaslora_scenario": sys.argv[23],
    "execution_order": sys.argv[24].split(),
    "faaslora_frozen_settings_sha256": sys.argv[25],
    "campaign_kind": sys.argv[26] or None,
}
resolved_config_path = round_dir / "protocol" / "system_resolved_config.json"
if not resolved_config_path.is_file():
    raise SystemExit(f"[ERROR] missing resolved-config sidecar: {resolved_config_path}")
resolved_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
if int(resolved_config.get("sampling_seed", -1)) != payload["sampling_seed"]:
    raise SystemExit("[ERROR] resolved-config sidecar sampling seed mismatch")
if str(resolved_config.get("model_profile") or "") != payload["model_profile"]:
    raise SystemExit("[ERROR] resolved-config sidecar model profile mismatch")
if (resolved_config.get("campaign_kind") or None) != payload["campaign_kind"]:
    raise SystemExit("[ERROR] resolved-config sidecar campaign kind mismatch")
resolved_hash = str(resolved_config.get("system_resolved_config_sha256") or "")
if len(resolved_hash) != 64 or any(ch not in "0123456789abcdef" for ch in resolved_hash):
    raise SystemExit("[ERROR] invalid system_resolved_config_sha256 in sidecar")
full_run_hash = str(resolved_config.get("full_run_identity_sha256") or "")
if len(full_run_hash) != 64 or any(ch not in "0123456789abcdef" for ch in full_run_hash):
    raise SystemExit("[ERROR] invalid full_run_identity_sha256 in sidecar")
payload.update(
    {
        "formal_run": bool(resolved_config.get("formal_run")),
        "trace_role": str(resolved_config.get("trace_role") or ""),
        "source_clean_for_formal": bool(resolved_config.get("source_clean_for_formal")),
        "source_cleanliness": resolved_config.get("source_cleanliness") or {},
        "system_resolved_config_sha256": resolved_hash,
        "system_resolved_config_path": str(resolved_config_path),
        "system_resolved_config_sidecar_sha256": sha256(resolved_config_path),
        "system_resolved_config_sidecar_bytes": resolved_config_path.stat().st_size,
        "system_resolved_config_family_id": str(
            resolved_config.get("configuration_family_id") or ""
        ),
        "full_run_identity_sha256": full_run_hash,
        "system_resolved_config_registry_path": str(
            resolved_config.get("registry_path") or ""
        ),
    }
)
if payload["formal_run"] and not payload["source_clean_for_formal"]:
    raise SystemExit("[ERROR] formal manifest refuses non-clean tracked source state")
for name, cwd in (("baseline_git", "/home/qhq/serverless_llm_baselines"), ("faaslora_git", "/home/qhq/serverless_llm_experiment_retry14_baseline")):
    try:
        commit = subprocess.check_output(["git", "-C", cwd, "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(["git", "-C", cwd, "branch", "--show-current"], text=True).strip()
        status = subprocess.check_output(
            ["git", "-C", cwd, "status", "--short", "--untracked-files=no"],
            text=True,
        ).strip().splitlines()
    except Exception:
        commit = ""
        branch = ""
        status = []
    payload[name] = {
        "path": cwd,
        "branch": branch,
        "commit": commit,
        "tracked_dirty_paths": status,
    }

source_files = {}
raw_sources = []
for root_name in ("raw", "compare", "logs"):
    root = round_dir / root_name
    if not root.exists():
        continue
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        source_files[str(path.relative_to(round_dir))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        if root_name == "raw":
            raw_sources.append(
                {
                    "path": str(path.resolve()),
                    "relative_path": str(path.relative_to(round_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
payload["source_files"] = source_files
payload["raw_sources"] = raw_sources

state_files = sorted(path.name for path in (round_dir / "state").glob("*.done"))
payload["state_markers"] = state_files
stage_by_system = {
    "sglang": "10_sglang.done",
    "serverlessllm": "20_serverlessllm.done",
    "vllm": "30_vllm.done",
    "slora": "40_slora.done",
    "faaslora": "50_faaslora.done",
}
selected_supported = payload["supported_systems"]
required_markers = ["00_prep.done"] + [stage_by_system[item] for item in selected_supported]
if selected_supported:
    required_markers.append("90_compare.done")
payload["required_state_markers"] = required_markers
payload["status"] = (
    "complete" if all(marker in state_files for marker in required_markers) else "incomplete"
)

try:
    gpu_rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip().splitlines()
except Exception:
    gpu_rows = []
payload["hardware"] = {
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "gpu_rows": gpu_rows,
    "topology_note": "See the campaign preservation hardware-topology preflight; runtime TP/DP is recorded in each system summary.",
}
(round_dir / "MANIFEST.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(round_dir / "MANIFEST.json")
PY
}

run_system_if_selected() {
  local system="$1"
  case " ${SYSTEMS} " in
    *" ${system} "*)
      if system_supported "${system}"; then
        "run_${system}"
      else
        mark_unsupported_system "${system}"
      fi
      ;;
    *) log "skip ${system}; not listed in FAIR_ROUND_SYSTEMS=${SYSTEMS}" ;;
  esac
}

main() {
  validate_v2_protocol_controls
  validate_trace_role_and_formal_sources
  log "round_dir=${ROUND_DIR}"
  log "run_tag=${RUN_TAG}"
  log "formal_run=${FORMAL_RUN} trace_role=${TRACE_ROLE}"
  log "campaign_kind=${CAMPAIGN_KIND:-exploratory}"
  log "execution_order=${EXECUTION_ORDER}"
  log "resume_file=${ROUND_ENV_FILE}"
  run_prep
  prepare_resolved_config_gate
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "dry-run complete after shared artifact preparation; selected systems=${SYSTEMS}"
    log "dry-run does not launch serving systems, replay requests, validate summaries, or write compare tables"
    validate_trace_role_and_formal_sources
    write_manifest
    return 0
  fi
  local system=""
  for system in ${EXECUTION_ORDER}; do
    run_system_if_selected "${system}"
  done
  run_compare
  validate_trace_role_and_formal_sources
  write_manifest
  mark_successful_validation_if_complete
  log "round complete: ${ROUND_DIR}"
  log "comparison: ${COMPARE_DIR}/${RUN_TAG}_five_system_compare.txt"
}

main "$@"
