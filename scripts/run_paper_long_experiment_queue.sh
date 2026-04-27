#!/usr/bin/env bash
set -euo pipefail

BASELINES_ROOT="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
MAIN_REPO="${SLLM_MAIN_REPO:-/home/qhq/serverless_llm_experiment_retry14_baseline}"
RUNNER="${PAPER_QUEUE_RUNNER:-${BASELINES_ROOT}/scripts/run_full_fair_round.sh}"

QUEUE_PROFILE="${PAPER_QUEUE_PROFILE:-load_p0}"
QUEUE_ID="${PAPER_QUEUE_ID:-$(date +%Y%m%d_%H%M%S)_${QUEUE_PROFILE}}"
QUEUE_DIR="${PAPER_QUEUE_DIR:-${BASELINES_ROOT}/results/paper_experiments/00_queues/${QUEUE_ID}}"
STATE_DIR="${QUEUE_DIR}/state"
LOG_DIR="${QUEUE_DIR}/logs"
DRY_RUN="${PAPER_QUEUE_DRY_RUN:-0}"

# Default queue is the complete five-system comparison. Use PAPER_QUEUE_SYSTEMS
# only for explicit diagnostic/fast partial rounds; paper-facing sensitivity
# rounds should keep ServerlessLLM unless the manuscript states otherwise.
DEFAULT_SYSTEMS="${PAPER_QUEUE_SYSTEMS:-sglang serverlessllm vllm slora faaslora}"

mkdir -p "${STATE_DIR}" "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

sanitize_label() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
}

stage_done_path() {
  printf '%s/%s.done\n' "${STATE_DIR}" "$(sanitize_label "$1")"
}

is_done() {
  [[ -f "$(stage_done_path "$1")" ]]
}

mark_done() {
  date '+%F %T' >"$(stage_done_path "$1")"
}

round_compare_contains_systems() {
  local round_dir="$1"
  local run_tag="$2"
  local systems="$3"
  local compare_json="${round_dir}/compare/${run_tag}_five_system_compare.json"
  python3 - "${compare_json}" "${systems}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
required = set(sys.argv[2].split())
if not path.exists():
    raise SystemExit(1)

payload = json.loads(path.read_text(encoding="utf-8"))
headers = payload.get("strict_headers") or []
rows = payload.get("strict_rows") or []
if "System" not in headers:
    raise SystemExit(1)
idx = headers.index("System")
seen = set()
for row in rows:
    name = str(row[idx]).lower() if idx < len(row) else ""
    if "faaslora" in name or "primelora" in name:
        seen.add("faaslora")
    if "sglang" in name:
        seen.add("sglang")
    if "serverlessllm" in name:
        seen.add("serverlessllm")
    if "vllm" in name:
        seen.add("vllm")
    if "s-lora" in name or "slora" in name:
        seen.add("slora")
missing = required - seen
raise SystemExit(0 if not missing else 1)
PY
}

write_queue_env() {
  {
    printf 'export PAPER_QUEUE_ID=%q\n' "${QUEUE_ID}"
    printf 'export PAPER_QUEUE_DIR=%q\n' "${QUEUE_DIR}"
    printf 'export PAPER_QUEUE_PROFILE=%q\n' "${QUEUE_PROFILE}"
    printf 'export PAPER_QUEUE_SYSTEMS=%q\n' "${DEFAULT_SYSTEMS}"
  } >"${QUEUE_DIR}/queue.env"
}

validate_runner() {
  if [[ ! -x "${RUNNER}" ]]; then
    echo "[ERROR] fair-round runner is not executable: ${RUNNER}" >&2
    return 1
  fi
  bash -n "${RUNNER}"
}

run_round() {
  local label="$1"
  local section="$2"
  local run_tag="$3"
  local model_profile="$4"
  local dataset_profile="$5"
  local workload_profile="$6"
  local total_requests="$7"
  local adapters="$8"
  local seed="$9"
  local time_scale="${10}"
  local systems="${11:-${DEFAULT_SYSTEMS}}"

  local round_dir="${BASELINES_ROOT}/results/paper_experiments/${section}/${QUEUE_ID}_${run_tag}"
  local stage_label="${section}_${run_tag}"
  local log_path="${LOG_DIR}/$(sanitize_label "${stage_label}").log"

  if is_done "${stage_label}"; then
    if round_compare_contains_systems "${round_dir}" "${run_tag}" "${systems}"; then
      log "skip ${stage_label}; queue marker exists and compare contains selected systems"
      return 0
    fi
    log "${stage_label} marker exists but compare is missing selected systems; resuming round"
  fi

  log "queue stage=${stage_label}"
  log "round_dir=${round_dir}"
  log "systems=${systems}"
  log "time_scale=${time_scale}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    cat <<EOF
[dry-run] ${label}
  FAIR_ROUND_DIR=${round_dir}
  SLLM_RUN_TAG=${run_tag}
  SLLM_MODEL_PROFILE=${model_profile}
  SLLM_DATASET_PROFILE=${dataset_profile}
  SLLM_WORKLOAD_PROFILE=${workload_profile}
  SLLM_TOTAL_REQUESTS=${total_requests}
  SLLM_SELECTED_NUM_ADAPTERS=${adapters}
  SLLM_SAMPLING_SEED=${seed}
  SLLM_TIME_SCALE_FACTOR=${time_scale}
  FAIR_ROUND_SYSTEMS=${systems}
EOF
    return 0
  fi

  set +e
  env \
    SLLM_BASELINES_ROOT="${BASELINES_ROOT}" \
    SLLM_MAIN_REPO="${MAIN_REPO}" \
    FAIR_ROUND_DIR="${round_dir}" \
    FAIR_ROUND_SECTION="${section}" \
    FAIR_ROUND_LABEL="${run_tag}" \
    SLLM_RUN_TAG="${run_tag}" \
    SLLM_MODEL_PROFILE="${model_profile}" \
    SLLM_DATASET_PROFILE="${dataset_profile}" \
    SLLM_WORKLOAD_PROFILE="${workload_profile}" \
    SLLM_TOTAL_REQUESTS="${total_requests}" \
    SLLM_SELECTED_NUM_ADAPTERS="${adapters}" \
    SLLM_SAMPLING_SEED="${seed}" \
    SLLM_TIME_SCALE_FACTOR="${time_scale}" \
    FAIR_ROUND_SYSTEMS="${systems}" \
    bash "${RUNNER}" 2>&1 | tee "${log_path}"
  local status=${PIPESTATUS[0]}
  set -e

  if [[ "${status}" -ne 0 ]]; then
    log "failed stage=${stage_label} status=${status}; rerun this queue with PAPER_QUEUE_ID=${QUEUE_ID} to resume"
    return "${status}"
  fi

  mark_done "${stage_label}"
}

run_profile_load_p0() {
  # P0: long-running, low-risk sensitivity data. Same model/profile/adapters as
  # the closed Llama-2 7B main round; only the trace time scale changes.
  run_round \
    "Llama-2 7B load sensitivity s6" \
    "06_sensitivity_load" \
    "llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s6_sensload_v1" \
    "llama2_7b_main_v2_publicmix" \
    "azure_sharegpt_rep4000" \
    "llama2_7b_auto500_formal4000_s8" \
    "4000" "500" "42" "6.0" "${DEFAULT_SYSTEMS}"

  run_round \
    "Llama-2 7B load sensitivity s4" \
    "06_sensitivity_load" \
    "llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s4_sensload_v1" \
    "llama2_7b_main_v2_publicmix" \
    "azure_sharegpt_rep4000" \
    "llama2_7b_auto500_formal4000_s8" \
    "4000" "500" "42" "4.0" "${DEFAULT_SYSTEMS}"
}

run_profile_load_p1() {
  run_profile_load_p0
  run_round \
    "Llama-2 7B load sensitivity s2" \
    "06_sensitivity_load" \
    "llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s2_sensload_v1" \
    "llama2_7b_main_v2_publicmix" \
    "azure_sharegpt_rep4000" \
    "llama2_7b_auto500_formal4000_s8" \
    "4000" "500" "42" "2.0" "${DEFAULT_SYSTEMS}"
}

run_profile_load_operating_p0() {
  # Low/medium-load sensitivity for the normal serverless operating regime.
  # These points keep the same Llama-2 7B formal workload as the closed s8 main
  # round and only lower arrival pressure by increasing the replay time scale.
  # The existing s8 main round is the nominal-load point; s6/s4 are stress.
  run_round \
    "Llama-2 7B low-load sensitivity s12" \
    "06_sensitivity_load_operating" \
    "llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s12_sensloadop_v1" \
    "llama2_7b_main_v2_publicmix" \
    "azure_sharegpt_rep4000" \
    "llama2_7b_auto500_formal4000_s8" \
    "4000" "500" "42" "12.0" "${DEFAULT_SYSTEMS}"

  run_round \
    "Llama-2 7B medium-load sensitivity s10" \
    "06_sensitivity_load_operating" \
    "llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s10_sensloadop_v1" \
    "llama2_7b_main_v2_publicmix" \
    "azure_sharegpt_rep4000" \
    "llama2_7b_auto500_formal4000_s8" \
    "4000" "500" "42" "10.0" "${DEFAULT_SYSTEMS}"
}

run_profile_load_low_p0() {
  # Backward-compatible alias kept for any existing queue commands.
  run_profile_load_operating_p0
}

main() {
  write_queue_env
  validate_runner
  log "queue_id=${QUEUE_ID}"
  log "queue_dir=${QUEUE_DIR}"
  log "queue_env=${QUEUE_DIR}/queue.env"
  log "profile=${QUEUE_PROFILE}"

  case "${QUEUE_PROFILE}" in
    load_p0) run_profile_load_p0 ;;
    load_p1) run_profile_load_p1 ;;
    load_operating_p0) run_profile_load_operating_p0 ;;
    load_low_p0) run_profile_load_low_p0 ;;
    *)
      echo "[ERROR] unknown PAPER_QUEUE_PROFILE=${QUEUE_PROFILE}; supported: load_p0, load_p1, load_operating_p0, load_low_p0" >&2
      return 2
      ;;
  esac

  log "queue complete: ${QUEUE_DIR}"
}

main "$@"
