#!/usr/bin/env bash
set -euo pipefail

MAIN_REPO="${FAASLORA_MAIN_REPO:-/home/qhq/serverless_llm_experiment_retry14_baseline}"
BASELINES_ROOT="${FAASLORA_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
RUNNER="${MAIN_REPO}/scripts/run_all_experiments_user_scope.sh"
PYTHON_BIN="${FAASLORA_PYTHON:-/home/qhq/anaconda3/envs/LLM_vllm0102/bin/python}"
CONFIG_PATH="${FAASLORA_PAPER_ABLATION_CONFIG:-${MAIN_REPO}/configs/experiments.yaml}"

MODEL_PROFILE="${FAASLORA_PROFILE_MODEL:-llama2_7b_main_v2_publicmix}"
DATASET_PROFILE="${FAASLORA_PROFILE_DATASET:-azure_sharegpt_rep4000}"
WORKLOAD_PROFILE="${FAASLORA_PROFILE_WORKLOAD:-llama2_7b_auto500_formal4000_s8}"
TOTAL_REQUESTS="${FAASLORA_TOTAL_REQUESTS:-4000}"
SELECTED_NUM_ADAPTERS="${FAASLORA_SELECTED_NUM_ADAPTERS:-500}"
SAMPLING_SEED="${FAASLORA_SAMPLING_SEED:-42}"
STORAGE_BANDWIDTH_MIB_S="${FAASLORA_STORAGE_BANDWIDTH_MIB_S:-250}"
ZIPF_EXPONENT="${FAASLORA_ZIPF_EXPONENT:-1.0}"
ACTIVE_ADAPTER_CAP="${FAASLORA_ACTIVE_ADAPTER_CAP:-48}"
HOTSET_ROTATION_REQUESTS="${FAASLORA_HOTSET_ROTATION_REQUESTS:-500}"
HOTSET_ROTATION_MODE="$(printf '%s' "${FAASLORA_HOTSET_ROTATION_MODE:-legacy}" | tr '[:upper:]' '[:lower:]')"
HOTSET_OVERLAP_FRACTION="${FAASLORA_HOTSET_OVERLAP_FRACTION:-0.75}"
TIME_SCALE_FACTOR="${FAASLORA_TIME_SCALE_FACTOR:-8.0}"

SOURCE_RUN_TAG="${FAASLORA_SOURCE_RUN_TAG:-llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1}"
SOURCE_ROUND_DIR="${FAASLORA_SOURCE_ROUND_DIR:-${BASELINES_ROOT}/results/paper_experiments/03_main_comparison/20260424_104050_${SOURCE_RUN_TAG}}"
TRACE_PATH="${FAASLORA_SHARED_TRACE_PATH:-${SOURCE_ROUND_DIR}/shared_artifacts/${SOURCE_RUN_TAG}_trace.json}"
ADAPTER_SUBSET_PATH="${FAASLORA_SHARED_ADAPTER_SUBSET_PATH:-${SOURCE_ROUND_DIR}/shared_artifacts/${SOURCE_RUN_TAG}_adapter_subset.json}"

RUN_TAG="${FAASLORA_PAPER_ABLATION_RUN_TAG:-llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_ablation_v1}"
SECTION_ID="${FAASLORA_PAPER_ABLATION_SECTION_ID:-04_ablation}"
ROUND_PURPOSE="${FAASLORA_PAPER_ABLATION_PURPOSE:-fig2_fig3_fig6_faaslora_cumulative_ablation_and_coordination}"
FIGURE_TARGETS="${FAASLORA_PAPER_ABLATION_FIGURES:-Fig2 Fig3 Fig6 CoordinationSubfigure}"
ROUND_ROOT="${FAASLORA_PAPER_ABLATION_ROOT:-${BASELINES_ROOT}/results/paper_experiments/${SECTION_ID}}"
ROUND_TIMESTAMP="${FAASLORA_PAPER_ABLATION_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
ROUND_DIR="${FAASLORA_PAPER_ABLATION_ROUND_DIR:-${ROUND_ROOT}/${ROUND_TIMESTAMP}_${RUN_TAG}}"
SCENARIOS_RAW="${FAASLORA_PAPER_ABLATION_SCENARIOS:-v2_elastic_only v2_hit_aware_preparation v2_hierarchical_no_coord v2_full}"
FORCE_RERUN="${FAASLORA_PAPER_ABLATION_FORCE:-0}"
GPU_IDS="${FAASLORA_PAPER_ABLATION_GPU_IDS:-0,1,2,3}"
REQUIRE_GPU_IDLE="${FAASLORA_PAPER_ABLATION_REQUIRE_GPU_IDLE:-1}"
KILL_KNOWN_GPU_RESIDUALS="${FAASLORA_PAPER_ABLATION_KILL_KNOWN_GPU_RESIDUALS:-1}"
CLEANUP_TIMEOUT_S="${FAASLORA_PAPER_ABLATION_CLEANUP_TIMEOUT_S:-180}"
DRY_RUN="${FAASLORA_PAPER_ABLATION_DRY_RUN:-0}"
ALLOW_INTERNAL_BASELINES="${FAASLORA_PAPER_ABLATION_ALLOW_INTERNAL_BASELINES:-0}"
REQUIRE_FEATURE_TRIGGER="${FAASLORA_PAPER_ABLATION_REQUIRE_FEATURE_TRIGGER:-1}"
FORMAL_RUN="${FAASLORA_PAPER_ABLATION_FORMAL:-0}"
TRACE_ROLE="${FAASLORA_TRACE_ROLE:-auto}"
EXPECTED_NON_FEATURE_FROZEN_CONFIG_SHA256="${FAASLORA_EXPECTED_NON_FEATURE_FROZEN_CONFIG_SHA256:-}"
VALIDATION_REGISTRY="${FAASLORA_PAPER_ABLATION_VALIDATION_REGISTRY:-${ROUND_ROOT}/_protocol/non_feature_validation_registry.json}"
VALIDATION_EVIDENCE_PATH="${ROUND_DIR}/protocol/seed41_validation_evidence.json"
VALIDATION_REGISTRY_TOOL="${MAIN_REPO}/scripts/faaslora_ablation_validation_registry.py"
MULTI_CYCLE_PHASES="${FAASLORA_MULTI_CYCLE_PHASES:-}"
IDLE_BETWEEN_PHASES_S="${FAASLORA_IDLE_BETWEEN_PHASES_S:-}"
TUNING_ENV_SHA256=""

if [[ "${TRACE_ROLE}" == "auto" ]]; then
  case "${SAMPLING_SEED}" in
    41) TRACE_ROLE="validation" ;;
    42) TRACE_ROLE="smoke" ;;
    43|44|45) TRACE_ROLE="heldout" ;;
    *) TRACE_ROLE="exploratory" ;;
  esac
fi

RAW_DIR="${ROUND_DIR}/raw/faaslora"
LOG_DIR="${ROUND_DIR}/logs"
STATE_DIR="${ROUND_DIR}/state"
SHARED_DIR="${ROUND_DIR}/shared_artifacts"
PROTOCOL_DIR="${ROUND_DIR}/protocol"

mkdir -p "${RAW_DIR}" "${LOG_DIR}" "${STATE_DIR}" "${SHARED_DIR}" "${PROTOCOL_DIR}"

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

sanitize_label() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
}

gpu_residual_pids() {
  local gpu_csv="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  local gpu_ids=()
  local gpu=""
  local output=""
  IFS=',' read -r -a gpu_ids <<< "${gpu_csv}"
  for gpu in "${gpu_ids[@]}"; do
    gpu="$(printf '%s' "${gpu}" | xargs)"
    [[ -z "${gpu}" ]] && continue
    if output="$(nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
      printf '%s\n' "${output}" | sed '/^[[:space:]]*$/d' | awk '{print $1}'
    fi
  done | sort -u
}

check_gpu_idle() {
  if [[ "${REQUIRE_GPU_IDLE}" != "1" ]]; then
    return 0
  fi
  local pids=()
  mapfile -t pids < <(gpu_residual_pids "${GPU_IDS}" || true)
  if (( ${#pids[@]} == 0 )); then
    return 0
  fi
  log "[ERROR] GPUs are not idle on ids=${GPU_IDS}; refusing to start a formal ablation stage."
  for pid in "${pids[@]}"; do
    ps -fp "${pid}" || true
  done
  log "Stop unrelated GPU jobs first, or set FAASLORA_PAPER_ABLATION_REQUIRE_GPU_IDLE=0 if this is intentional."
  return 1
}

kill_known_gpu_residuals() {
  if [[ "${KILL_KNOWN_GPU_RESIDUALS}" != "1" ]]; then
    return 0
  fi
  local pids=()
  local pid=""
  mapfile -t pids < <(gpu_residual_pids "${GPU_IDS}" || true)
  for pid in "${pids[@]}"; do
    [[ -z "${pid}" ]] && continue
    local cmd=""
    cmd="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
    case "${cmd}" in
      *serverless_llm_experiment_retry14_baseline*|*faaslora*|*dedicated_engine_worker*)
        log "cleaning known FaaSLoRA GPU residual pid=${pid}"
        kill "${pid}" 2>/dev/null || true
        ;;
      *)
        log "leaving non-round GPU process untouched pid=${pid} cmd=${cmd}"
        ;;
    esac
  done
}

wait_gpu_idle() {
  if [[ "${REQUIRE_GPU_IDLE}" != "1" ]]; then
    return 0
  fi
  local deadline=$((SECONDS + CLEANUP_TIMEOUT_S))
  local pids=()
  while true; do
    mapfile -t pids < <(gpu_residual_pids "${GPU_IDS}" || true)
    if (( ${#pids[@]} == 0 )); then
      log "GPU post-clean gate passed for ids=${GPU_IDS}"
      return 0
    fi
    if (( SECONDS >= deadline )); then
      log "[ERROR] GPU processes remain after cleanup timeout ids=${GPU_IDS} pids=${pids[*]}"
      for pid in "${pids[@]}"; do
        ps -fp "${pid}" || true
      done
      return 1
    fi
    sleep 3
  done
}

validate_shared_artifacts() {
  "${PYTHON_BIN}" - "${TRACE_PATH}" "${ADAPTER_SUBSET_PATH}" "${MODEL_PROFILE}" "${DATASET_PROFILE}" "${WORKLOAD_PROFILE}" "${TOTAL_REQUESTS}" "${SELECTED_NUM_ADAPTERS}" "${SAMPLING_SEED}" \
    "${ZIPF_EXPONENT}" "${ACTIVE_ADAPTER_CAP}" "${HOTSET_ROTATION_REQUESTS}" \
    "${HOTSET_ROTATION_MODE}" "${HOTSET_OVERLAP_FRACTION}" "${TIME_SCALE_FACTOR}" \
    "${FORMAL_RUN}" <<'PY'
import json
import math
import sys
from pathlib import Path

trace_path = Path(sys.argv[1])
subset_path = Path(sys.argv[2])
model_profile, dataset_profile, workload_profile = sys.argv[3:6]
total_requests = int(sys.argv[6])
selected_num_adapters = int(sys.argv[7])
sampling_seed = int(sys.argv[8])
zipf_exponent = float(sys.argv[9])
active_adapter_cap = int(sys.argv[10])
hotset_rotation_requests = int(sys.argv[11])
hotset_rotation_mode = sys.argv[12].strip().lower()
hotset_overlap_fraction = float(sys.argv[13])
time_scale_factor = float(sys.argv[14])
formal_run = sys.argv[15] == "1"

if not trace_path.exists():
    raise SystemExit(f"shared trace artifact not found: {trace_path}")
if not subset_path.exists():
    raise SystemExit(f"shared adapter subset artifact not found: {subset_path}")

trace = json.loads(trace_path.read_text(encoding="utf-8"))
subset = json.loads(subset_path.read_text(encoding="utf-8"))

for field, expected in (
    ("model_profile", model_profile),
    ("dataset_profile", dataset_profile),
    ("workload_profile", workload_profile),
):
    if trace.get(field) != expected:
        raise SystemExit(f"trace {field} mismatch: expected {expected}, got {trace.get(field)}")
    if subset.get(field) != expected:
        raise SystemExit(f"subset {field} mismatch: expected {expected}, got {subset.get(field)}")

if len(trace.get("requests", [])) != total_requests:
    raise SystemExit(f"trace request count mismatch: expected {total_requests}, got {len(trace.get('requests', []))}")
if int(trace.get("selected_num_adapters", -1)) != selected_num_adapters:
    raise SystemExit("trace selected_num_adapters mismatch")
if int(subset.get("selected_num_adapters", -1)) != selected_num_adapters:
    raise SystemExit("subset selected_num_adapters mismatch")
if int(trace.get("sampling_seed", -1)) != sampling_seed:
    raise SystemExit("trace sampling_seed mismatch")
if int(subset.get("sampling_seed", -1)) != sampling_seed:
    raise SystemExit("subset sampling_seed mismatch")

load_profile = trace.get("load_profile") or {}

def require_float(field, expected, *, root_field=None):
    values = [(f"load_profile.{field}", load_profile.get(field))]
    if root_field:
        values.append((root_field, trace.get(root_field)))
    present = [(name, value) for name, value in values if value is not None]
    if not present:
        raise SystemExit(f"shared trace missing workload axis {field}")
    for name, value in present:
        if not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-9):
            raise SystemExit(
                f"shared trace {name} mismatch: expected {expected}, got {value}"
            )

def require_int(field, expected):
    value = load_profile.get(field)
    if value is None:
        raise SystemExit(f"shared trace missing workload axis {field}")
    if int(value) != expected:
        raise SystemExit(
            f"shared trace {field} mismatch: expected {expected}, got {value}"
        )

require_float("zipf_exponent", zipf_exponent)
require_int("active_adapter_cap", active_adapter_cap)
require_int("hotset_rotation_requests", hotset_rotation_requests)
require_float(
    "configured_time_scale_factor",
    time_scale_factor,
    root_field="configured_time_scale_factor",
)
require_float(
    "effective_time_scale_factor",
    time_scale_factor,
    root_field="effective_time_scale_factor",
)

trace_rotation_mode = load_profile.get("hotset_rotation_mode")
trace_overlap = load_profile.get("hotset_overlap_fraction")
# Old non-formal compatibility traces predate these two explicit fields. Their
# generator semantics were exactly legacy/0.75; formal V2 traces must record
# both axes rather than relying on that inference.
if trace_rotation_mode is None or trace_overlap is None:
    if formal_run:
        missing = [
            name
            for name, value in (
                ("hotset_rotation_mode", trace_rotation_mode),
                ("hotset_overlap_fraction", trace_overlap),
            )
            if value is None
        ]
        raise SystemExit(
            "formal shared trace missing workload axes: " + ", ".join(missing)
        )
    trace_rotation_mode = trace_rotation_mode or "legacy"
    trace_overlap = 0.75 if trace_overlap is None else trace_overlap
if str(trace_rotation_mode).strip().lower() != hotset_rotation_mode:
    raise SystemExit(
        "shared trace hotset_rotation_mode mismatch: "
        f"expected {hotset_rotation_mode}, got {trace_rotation_mode}"
    )
if not math.isclose(
    float(trace_overlap), hotset_overlap_fraction, rel_tol=0.0, abs_tol=1e-9
):
    raise SystemExit(
        "shared trace hotset_overlap_fraction mismatch: "
        f"expected {hotset_overlap_fraction}, got {trace_overlap}"
    )

subset_ids = {str(item["id"]) for item in subset.get("adapters", []) if "id" in item}
if len(subset_ids) != selected_num_adapters:
    raise SystemExit(f"subset adapter cardinality mismatch: expected {selected_num_adapters}, got {len(subset_ids)}")

trace_ids = {str(req.get("adapter_id")) for req in trace.get("requests", []) if req.get("adapter_id") is not None}
missing = sorted(trace_ids - subset_ids)
if missing:
    raise SystemExit(f"trace references adapters outside subset: {missing[:8]}")
PY
}

validate_scenarios() {
  local scenario=""
  local allowed_faaslora=" faaslora_nvme faaslora_no_coord faaslora_full v2_elastic_only v2_hit_aware_preparation v2_hierarchical_no_coord v2_full "
  local allowed_internal=" cold_start slora_style serverlessllm "
  if [[ "${FORCE_RERUN}" == "1" && " ${SCENARIOS_RAW} " == *" v2_"* ]]; then
    log "[ERROR] V2 protocol forbids force-overwriting an ablation round; use a new unique round directory."
    return 1
  fi
  for scenario in "${SCENARIOS[@]}"; do
    [[ -z "${scenario}" ]] && continue
    if [[ "${allowed_faaslora}" == *" ${scenario} "* ]]; then
      continue
    fi
    if [[ "${ALLOW_INTERNAL_BASELINES}" == "1" && "${allowed_internal}" == *" ${scenario} "* ]]; then
      continue
    fi
    log "[ERROR] scenario=${scenario} is not allowed in this paper ablation script."
    log "Allowed by default: legacy FaaSLoRA ablations and explicit v2_* cumulative scenarios."
    log "Internal legacy references require FAASLORA_PAPER_ABLATION_ALLOW_INTERNAL_BASELINES=1 and must not be mixed with official baseline claims."
    return 1
  done
}

validate_source_and_seed_protocol() {
  if [[ -n "${EXPECTED_NON_FEATURE_FROZEN_CONFIG_SHA256}" && ! "${EXPECTED_NON_FEATURE_FROZEN_CONFIG_SHA256}" =~ ^[0-9a-fA-F]{64}$ ]]; then
    log "[ERROR] FAASLORA_EXPECTED_NON_FEATURE_FROZEN_CONFIG_SHA256 must be a 64-character SHA-256"
    return 1
  fi
  case "${TRACE_ROLE}" in
    validation|smoke|heldout|exploratory) ;;
    *)
      log "[ERROR] unsupported FAASLORA_TRACE_ROLE=${TRACE_ROLE}"
      return 1
      ;;
  esac
  case "${TRACE_ROLE}:${SAMPLING_SEED}" in
    validation:41|smoke:42|heldout:43|heldout:44|heldout:45|exploratory:*) ;;
    *)
      log "[ERROR] trace_role=${TRACE_ROLE} is incompatible with seed=${SAMPLING_SEED}"
      return 1
      ;;
  esac
  if [[ "${FORMAL_RUN}" != "1" ]]; then
    return 0
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "[ERROR] formal ablation cannot be a dry-run; use FORMAL=0 for protocol smoke"
    return 1
  fi
  case "${TRACE_ROLE}:${TOTAL_REQUESTS}" in
    validation:1000|heldout:4000) ;;
    *)
      log "[ERROR] formal validation requires seed41/1000; heldout requires seeds43-45/4000"
      return 1
      ;;
  esac
  if [[ "${TRACE_ROLE}" == "validation" ]]; then
    if (( ${#SCENARIOS[@]} != 1 )) || [[ "${SCENARIOS[0]}" != "v2_full" ]]; then
      log "[ERROR] formal seed41 ablation validation must run exactly v2_full"
      return 1
    fi
    if [[ -n "${EXPECTED_NON_FEATURE_FROZEN_CONFIG_SHA256}" ]]; then
      log "[ERROR] seed41 validation must measure, not pre-impose, a frozen hash"
      return 1
    fi
  fi
  "${PYTHON_BIN}" - "${MAIN_REPO}" <<'PY'
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1])
allowed = {"configs/generated/lora_manifest_1000.json"}
rows = subprocess.check_output(
    ["git", "-C", str(repo), "status", "--short", "--untracked-files=no"],
    text=True,
).splitlines()
dirty = []
for row in rows:
    path = row[3:].strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    if path not in allowed:
        dirty.append(row)
if dirty:
    raise SystemExit(
        "formal ablation refuses tracked dirty source files (only the user-owned "
        "configs/generated/lora_manifest_1000.json is allowlisted):\n"
        + "\n".join(dirty)
    )
PY

  if [[ "${TRACE_ROLE}" == "heldout" ]]; then
    EXPECTED_NON_FEATURE_FROZEN_CONFIG_SHA256="$(
      "${PYTHON_BIN}" "${VALIDATION_REGISTRY_TOOL}" resolve-heldout \
        --registry "${VALIDATION_REGISTRY}" \
        --evidence "${VALIDATION_EVIDENCE_PATH}" \
        --repo "${MAIN_REPO}" \
        --config "${CONFIG_PATH}" \
        --sampling-seed "${SAMPLING_SEED}" \
        --total-requests "${TOTAL_REQUESTS}" \
        --round-dir "${ROUND_DIR}" \
        --expected-non-feature-sha256 "${EXPECTED_NON_FEATURE_FROZEN_CONFIG_SHA256}" \
        --model-profile "${MODEL_PROFILE}" \
        --dataset-profile "${DATASET_PROFILE}" \
        --workload-profile "${WORKLOAD_PROFILE}" \
        --selected-num-adapters "${SELECTED_NUM_ADAPTERS}" \
        --gpu-ids "${GPU_IDS}" \
        --generation-contract legacy
    )"
    if [[ ! "${EXPECTED_NON_FEATURE_FROZEN_CONFIG_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
      log "[ERROR] validation registry returned an invalid non-feature hash"
      return 1
    fi
    log "seed41 validation gate selected non_feature_sha=${EXPECTED_NON_FEATURE_FROZEN_CONFIG_SHA256}"
  fi
}

register_successful_validation_if_complete() {
  if [[ "${FORMAL_RUN}" != "1" || "${TRACE_ROLE}" != "validation" ]]; then
    return 0
  fi
  "${PYTHON_BIN}" "${VALIDATION_REGISTRY_TOOL}" register-validation \
    --registry "${VALIDATION_REGISTRY}" \
    --manifest "${ROUND_DIR}/MANIFEST.json" \
    --repo "${MAIN_REPO}" \
    --config "${CONFIG_PATH}" \
    --model-profile "${MODEL_PROFILE}" \
    --dataset-profile "${DATASET_PROFILE}" \
    --workload-profile "${WORKLOAD_PROFILE}" \
    --selected-num-adapters "${SELECTED_NUM_ADAPTERS}" \
    --gpu-ids "${GPU_IDS}" \
    --generation-contract legacy
}

faaslora_tuning_env_sha256() {
  "${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import os

# These fields identify the trace, sensitivity point, output location, or
# protocol role.  They belong to the full-run identity, not to the frozen
# system tuning configuration shared across bandwidth/workload points.
excluded_exact = {
    "FAASLORA_RESULTS_TAG",
    "FAASLORA_RUN_FROZEN_SETTINGS_SHA256",
    "FAASLORA_SYSTEM_RESOLVED_CONFIG_SHA256",
    "FAASLORA_EXPECTED_NON_FEATURE_FROZEN_CONFIG_SHA256",
    "FAASLORA_TRACE_ROLE",
    "FAASLORA_FORMAL_RUN",
    "FAASLORA_SAMPLING_SEED",
    "FAASLORA_WORKLOAD_SEED",
    "FAASLORA_GENERATION_SEED",
    "FAASLORA_SHARED_TRACE_PATH",
    "FAASLORA_SHARED_ADAPTER_SUBSET_PATH",
    "FAASLORA_NVME_CACHE_DIR",
    "FAASLORA_HOST_CACHE_DIR",
    "FAASLORA_TOTAL_REQUESTS",
    "FAASLORA_SELECTED_NUM_ADAPTERS",
    "FAASLORA_STORAGE_BANDWIDTH_MIB_S",
    "FAASLORA_STORAGE_BANDWIDTH_MBPS",
    "FAASLORA_ZIPF_EXPONENT",
    "FAASLORA_ACTIVE_ADAPTER_CAP",
    "FAASLORA_HOTSET_ROTATION_REQUESTS",
    "FAASLORA_HOTSET_ROTATION_MODE",
    "FAASLORA_HOTSET_OVERLAP_FRACTION",
    "FAASLORA_MULTI_CYCLE_PHASES",
    "FAASLORA_IDLE_BETWEEN_PHASES_S",
    "FAASLORA_GENERATION_CONTRACT",
    "FAASLORA_FIXED_OUTPUT_MAX_TOKENS",
    "FAASLORA_FIXED_PROMPT_MAX_TOKENS",
    "FAASLORA_TIME_SCALE_FACTOR",
    "FAASLORA_PROFILE_DATASET",
    "FAASLORA_PROFILE_WORKLOAD",
    "FAASLORA_V2_TUNING_ENV_SHA256",
}
excluded_fragments = (
    "RUN_TAG", "ROUND_DIR", "ROUND_ROOT", "ROUND_TIMESTAMP",
    "SOURCE_ROUND", "SOURCE_RUN", "PAPER_ABLATION_ROOT",
    "VALIDATION_REGISTRY", "DRY_RUN", "FORCE",
)
payload = {
    key: value
    for key, value in sorted(os.environ.items())
    if key.startswith(("FAASLORA_", "VLLM_"))
    and not key.startswith("FAASLORA_PAPER_ABLATION_")
    and key not in excluded_exact
    and not any(fragment in key for fragment in excluded_fragments)
}
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(canonical).hexdigest())
PY
}

faaslora_system_resolved_config_sha256() {
  local scenario="$1"
  "${PYTHON_BIN}" - \
    "${MAIN_REPO}" "${CONFIG_PATH}" "${scenario}" \
    "${MODEL_PROFILE}" "${GPU_IDS}" "${TUNING_ENV_SHA256}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1])
config = Path(sys.argv[2])
commit = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
).strip()
payload = {
    "schema": "faaslora_system_resolved_config_v2",
    "source_commit": commit,
    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "scenario": sys.argv[3],
    "model_profile": sys.argv[4],
    "gpu_ids": sys.argv[5],
    "tuning_env_sha256": sys.argv[6],
}
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(canonical).hexdigest())
PY
}

validate_result_json() {
  local result_path="$1"
  local scenario="$2"
  local resolved_config_sha=""
  resolved_config_sha="$(faaslora_system_resolved_config_sha256 "${scenario}")"
  "${PYTHON_BIN}" - "${result_path}" "${scenario}" "${TOTAL_REQUESTS}" \
    "${REQUIRE_FEATURE_TRIGGER}" "${TRACE_PATH}" "${ADAPTER_SUBSET_PATH}" \
    "${RUN_TAG}_${scenario}" "${STORAGE_BANDWIDTH_MIB_S}" \
    "${resolved_config_sha}" "${TRACE_ROLE}" "${FORMAL_RUN}" \
    "${EXPECTED_NON_FEATURE_FROZEN_CONFIG_SHA256}" "${SAMPLING_SEED}" \
    "${ZIPF_EXPONENT}" "${ACTIVE_ADAPTER_CAP}" "${HOTSET_ROTATION_REQUESTS}" \
    "${HOTSET_ROTATION_MODE}" "${HOTSET_OVERLAP_FRACTION}" \
    "${TIME_SCALE_FACTOR}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
scenario = sys.argv[2]
expected_total = int(sys.argv[3])
require_trigger = sys.argv[4] == "1"
trace_path = Path(sys.argv[5])
subset_path = Path(sys.argv[6])
expected_result_tag = sys.argv[7]
expected_bandwidth_mib_s = float(sys.argv[8])
expected_resolved_config_sha = sys.argv[9]
expected_trace_role = sys.argv[10]
expected_formal = sys.argv[11] == "1"
expected_non_feature_sha = sys.argv[12].strip().lower()
expected_generation_seed = int(sys.argv[13])
expected_zipf_exponent = float(sys.argv[14])
expected_active_adapter_cap = int(sys.argv[15])
expected_hotset_rotation_requests = int(sys.argv[16])
expected_hotset_rotation_mode = sys.argv[17].strip().lower()
expected_hotset_overlap_fraction = float(sys.argv[18])
expected_time_scale_factor = float(sys.argv[19])
obj = json.loads(path.read_text(encoding="utf-8"))

def sha256(candidate):
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

metadata = obj.get("metadata") or {}
if metadata.get("system_resolved_config_sha256") != expected_resolved_config_sha:
    raise SystemExit(
        f"{path}: resolved-config SHA mismatch: "
        f"expected={expected_resolved_config_sha} "
        f"actual={metadata.get('system_resolved_config_sha256')!r}"
    )
non_feature_sha = str(metadata.get("non_feature_frozen_config_sha256") or "")
if scenario.startswith("v2_") and not (
    len(non_feature_sha) == 64
    and all(ch in "0123456789abcdef" for ch in non_feature_sha.lower())
):
    raise SystemExit(
        f"{path}: missing or invalid non_feature_frozen_config_sha256"
    )
if expected_non_feature_sha and non_feature_sha.lower() != expected_non_feature_sha:
    raise SystemExit(
        f"{path}: frozen non-feature SHA mismatch: "
        f"expected={expected_non_feature_sha} actual={non_feature_sha.lower()}"
    )
if str(metadata.get("trace_role") or "") != expected_trace_role:
    raise SystemExit(f"{path}: trace_role mismatch")
if bool(metadata.get("formal_run")) is not expected_formal:
    raise SystemExit(f"{path}: formal_run mismatch")
if int(metadata.get("generation_seed", -1)) != expected_generation_seed:
    raise SystemExit(f"{path}: generation_seed mismatch")
if int(metadata.get("workload_seed", -1)) != expected_generation_seed:
    raise SystemExit(f"{path}: workload_seed mismatch")
if str(metadata.get("results_tag") or "") != expected_result_tag:
    raise SystemExit(
        f"{path}: results_tag mismatch: expected={expected_result_tag!r} "
        f"observed={metadata.get('results_tag')!r}"
    )
for field, expected in (
    ("shared_trace_sha256", sha256(trace_path)),
    ("shared_adapter_subset_sha256", sha256(subset_path)),
):
    if str(metadata.get(field) or "") != expected:
        raise SystemExit(f"{path}: {field} mismatch or missing")
if not math.isclose(
    float(metadata.get("bandwidth_mib_s", -1)),
    expected_bandwidth_mib_s,
    rel_tol=0.0,
    abs_tol=1e-9,
):
    raise SystemExit(f"{path}: aggregate bandwidth metadata mismatch")

def require_metadata_float(field, expected):
    value = metadata.get(field)
    if value is None or not math.isclose(
        float(value), expected, rel_tol=0.0, abs_tol=1e-9
    ):
        raise SystemExit(
            f"{path}: workload axis {field} mismatch: "
            f"expected={expected} actual={value!r}"
        )

def require_metadata_int(field, expected):
    value = metadata.get(field)
    if value is None or int(value) != expected:
        raise SystemExit(
            f"{path}: workload axis {field} mismatch: "
            f"expected={expected} actual={value!r}"
        )

require_metadata_float("zipf_exponent", expected_zipf_exponent)
require_metadata_int("active_adapter_cap", expected_active_adapter_cap)
require_metadata_int("hotset_rotation_requests", expected_hotset_rotation_requests)
require_metadata_float(
    "hotset_overlap_fraction", expected_hotset_overlap_fraction
)
require_metadata_float(
    "configured_time_scale_factor", expected_time_scale_factor
)
require_metadata_float("effective_time_scale_factor", expected_time_scale_factor)
if str(metadata.get("hotset_rotation_mode") or "").strip().lower() != expected_hotset_rotation_mode:
    raise SystemExit(
        f"{path}: workload axis hotset_rotation_mode mismatch: "
        f"expected={expected_hotset_rotation_mode!r} "
        f"actual={metadata.get('hotset_rotation_mode')!r}"
    )

shared_load_profile = metadata.get("shared_trace_load_profile") or {}
for field, expected in (
    ("zipf_exponent", expected_zipf_exponent),
    ("hotset_overlap_fraction", expected_hotset_overlap_fraction),
):
    value = shared_load_profile.get(field)
    if value is None or not math.isclose(
        float(value), expected, rel_tol=0.0, abs_tol=1e-9
    ):
        raise SystemExit(
            f"{path}: shared-trace workload axis {field} mismatch: "
            f"expected={expected} actual={value!r}"
        )
for field, expected in (
    ("active_adapter_cap", expected_active_adapter_cap),
    ("hotset_rotation_requests", expected_hotset_rotation_requests),
):
    value = shared_load_profile.get(field)
    if value is None or int(value) != expected:
        raise SystemExit(
            f"{path}: shared-trace workload axis {field} mismatch: "
            f"expected={expected} actual={value!r}"
        )
if str(shared_load_profile.get("hotset_rotation_mode") or "").strip().lower() != expected_hotset_rotation_mode:
    raise SystemExit(
        f"{path}: shared-trace workload axis hotset_rotation_mode mismatch"
    )
for field in (
    "shared_trace_configured_time_scale_factor",
    "shared_trace_effective_time_scale_factor",
):
    value = metadata.get(field)
    if value is None or not math.isclose(
        float(value), expected_time_scale_factor, rel_tol=0.0, abs_tol=1e-9
    ):
        raise SystemExit(
            f"{path}: {field} mismatch: "
            f"expected={expected_time_scale_factor} actual={value!r}"
        )

schema = obj.get("metric_schema_version")
if schema != "e2e_v3":
    raise SystemExit(f"{path}: metric_schema_version must be e2e_v3, got {schema!r}")

summaries = obj.get("scenario_summaries") or {}
if sorted(summaries) != [scenario]:
    raise SystemExit(
        f"{path}: scenario_summaries must contain exactly {scenario!r}; got {sorted(summaries)}"
    )
summary = summaries[scenario]

total = int(summary.get("total_requests", -1))
completed = int(summary.get("completed_requests", -1))
failed = int(summary.get("failed_requests", 0) or 0)
if total != expected_total:
    raise SystemExit(f"{path}: total_requests mismatch: expected {expected_total}, got {total}")
if completed != total or failed != 0:
    raise SystemExit(f"{path}: invalid completion state total={total} completed={completed} failed={failed}")
if str(summary.get("backend", "")).lower() != "vllm":
    raise SystemExit(f"{path}: backend must be vllm, got {summary.get('backend')!r}")

required_metrics = [
    "avg_overall_ttft_ms",
    "p95_overall_ttft_ms",
    "avg_overall_e2e_ms",
    "p95_overall_e2e_ms",
    "avg_tpot_ms",
    "throughput_tok_per_s",
    "monetary_cost_per_request_usd",
    "monetary_ce",
]
for key in required_metrics:
    value = summary.get(key)
    if value is None:
        raise SystemExit(f"{path}: missing required metric {key}")
    try:
        f = float(value)
    except Exception as exc:
        raise SystemExit(f"{path}: non-numeric metric {key}={value!r}") from exc
    if not math.isfinite(f) or f < 0:
        raise SystemExit(f"{path}: invalid metric {key}={value!r}")

host_required = bool(summary.get("host_cache_memory_backed_required", False))
if scenario in {"faaslora_no_coord", "faaslora_full"}:
    host_required = True
if host_required and not bool(summary.get("host_cache_memory_backed", False)):
    raise SystemExit(
        f"{path}: HOST tier is not memory-backed; host_cache_memory_backed={summary.get('host_cache_memory_backed')!r}"
    )

details = obj.get("detailed_results") or {}
if sorted(details) != [scenario]:
    raise SystemExit(
        f"{path}: detailed_results must contain exactly {scenario!r}; got {sorted(details)}"
    )
requests = details[scenario].get("requests") or []
if len(requests) != total:
    raise SystemExit(f"{path}: detailed request count mismatch: expected {total}, got {len(requests)}")

if scenario.startswith("v2_"):
    expected_gates = {
        "v2_elastic_only": (False, False, False, False, False),
        "v2_hit_aware_preparation": (True, True, False, False, False),
        "v2_hierarchical_no_coord": (True, True, True, False, False),
        "v2_full": (True, True, True, True, True),
    }
    coord = (((metadata.get("scenario_coordination") or {}).get(scenario)) or {})
    if not bool(coord.get("cold_cache_reset_before_run")):
        raise SystemExit(f"{path}: missing cold-cache reset evidence")
    gates = coord.get("feature_gates") or {}
    actual = (
        bool(gates.get("readiness_routing_enabled")),
        bool(gates.get("scale_up_handoff_enabled")),
        bool(gates.get("hierarchical_residency_enabled")),
        bool(gates.get("coordination_enabled")),
        bool(gates.get("effective_capacity_admission_enabled")),
    )
    if actual != expected_gates[scenario]:
        raise SystemExit(
            f"{path}: feature gate mismatch for {scenario}: "
            f"expected={expected_gates[scenario]} actual={actual}"
        )
    successful_lora = [row for row in requests if row.get("success") and row.get("adapter_id")]
    legal_tiers = {"gpu", "host", "nvme", "remote"}
    invalid_tiers = [
        row.get("request_id") for row in successful_lora
        if str(row.get("readiness_tier_before_dispatch", "")).lower() not in legal_tiers
    ]
    if invalid_tiers:
        raise SystemExit(
            f"{path}: incomplete dispatch-time tier evidence for {len(invalid_tiers)} requests"
        )
    activation = coord.get("feature_activation") or {}
    if require_trigger and expected_total >= 1000:
        if int(activation.get("routing_decision_count", 0) or 0) != expected_total:
            raise SystemExit(f"{path}: routing decision count did not cover every request")
        readiness_count = int(
            activation.get("readiness_aware_routing_decision_count", 0) or 0
        )
        load_only_count = int(
            activation.get("load_only_routing_decision_count", 0) or 0
        )
        if gates.get("readiness_routing_enabled"):
            if readiness_count < expected_total or load_only_count != 0:
                raise SystemExit(f"{path}: readiness-aware routing trigger count mismatch")
        elif load_only_count < expected_total or readiness_count != 0:
            raise SystemExit(f"{path}: ElasticOnly did not use pure load-only routing")
        if gates.get("scale_up_handoff_enabled") and int(
            activation.get("scale_up_events_with_planned_adapters", 0) or 0
        ) <= 0:
            raise SystemExit(f"{path}: handoff enabled but no planned-adapter scale-up event triggered")
        if gates.get("scale_up_handoff_enabled"):
            first_service = int(
                activation.get("scaleup_first_service_request_count", 0) or 0
            )
            planned_matches = int(
                activation.get("scaleup_first_service_planned_match_count", 0) or 0
            )
            if first_service <= 0 or planned_matches <= 0:
                raise SystemExit(
                    f"{path}: handoff was planned but no matched first-service request was served"
                )
        elif any(
            int(activation.get(name, 0) or 0) != 0
            for name in (
                "scale_up_events_with_planned_adapters",
                "scaleup_first_service_request_count",
                "scaleup_first_service_planned_match_count",
            )
        ):
            raise SystemExit(f"{path}: disabled handoff recorded served/planned activity")
        if gates.get("hierarchical_residency_enabled"):
            host_count = int(activation.get("initial_or_current_host_adapter_count", 0) or 0)
            nvme_count = int(activation.get("initial_or_current_nvme_adapter_count", 0) or 0)
            transition_count = int(
                activation.get("hierarchy_specific_online_transition_count", 0) or 0
            )
            if host_count <= 0 or nvme_count <= 0:
                raise SystemExit(
                    f"{path}: hierarchy enabled but HOST/NVMe tiers were not both populated"
                )
            if transition_count <= 0:
                raise SystemExit(
                    f"{path}: hierarchy enabled but no online HOST/GPU promotion completed"
                )
        elif any(
            int(activation.get(name, 0) or 0) != 0
            for name in (
                "initial_or_current_host_adapter_count",
                "host_promotion_scheduled_count",
                "host_promotion_completed_count",
                "runtime_gpu_forward_attempt_count",
                "runtime_gpu_forward_success_count",
                "hierarchy_specific_online_transition_count",
            )
        ):
            raise SystemExit(f"{path}: disabled hierarchy recorded hierarchy-specific activity")
        if gates.get("effective_capacity_admission_enabled"):
            if int(activation.get("gpu_admission_decision_count", 0) or 0) <= 0:
                raise SystemExit(f"{path}: admission enabled but no admission decision was observed")
            if int(
                activation.get("gpu_admission_observed_request_count", 0) or 0
            ) <= 0:
                raise SystemExit(
                    f"{path}: admission decisions were not observed on an online request"
                )
        elif any(
            int(activation.get(name, 0) or 0) != 0
            for name in (
                "gpu_admission_observed_request_count",
                "gpu_admission_decision_count",
            )
        ):
            raise SystemExit(f"{path}: disabled admission recorded admission activity")

print(
    f"validated {scenario}: TTFT_avg={summary['avg_overall_ttft_ms']:.3f}ms "
    f"E2E_avg={summary['avg_overall_e2e_ms']:.3f}ms "
    f"Cost/req=${summary['monetary_cost_per_request_usd']:.6f} "
    f"CE={summary['monetary_ce']:.3f}"
)
PY
}

find_result_json() {
  local scenario="$1"
  local result_tag="$2"
  local sanitized_tag=""
  sanitized_tag="$(sanitize_label "${result_tag}")"
  # MAIN_REPO/results is a symlink in the retry14 workspace. Use -L so a
  # successfully written FaaSLoRA result is recoverable after a harness failure.
  find -L "${MAIN_REPO}/results" -type f -name "*_${sanitized_tag}.json" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR==1 {sub(/^[^ ]+ /, ""); print}'
}

run_logged() {
  local stage="$1"
  shift
  local log_path="${LOG_DIR}/${stage}.log"
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

write_round_env() {
  {
    printf 'export FAASLORA_PAPER_ABLATION_ROUND_DIR=%q\n' "${ROUND_DIR}"
    printf 'export FAASLORA_PAPER_ABLATION_RUN_TAG=%q\n' "${RUN_TAG}"
    printf 'export FAASLORA_PAPER_ABLATION_SECTION_ID=%q\n' "${SECTION_ID}"
    printf 'export FAASLORA_PAPER_ABLATION_PURPOSE=%q\n' "${ROUND_PURPOSE}"
    printf 'export FAASLORA_PAPER_ABLATION_FIGURES=%q\n' "${FIGURE_TARGETS}"
    printf 'export FAASLORA_PROFILE_MODEL=%q\n' "${MODEL_PROFILE}"
    printf 'export FAASLORA_PROFILE_DATASET=%q\n' "${DATASET_PROFILE}"
    printf 'export FAASLORA_PROFILE_WORKLOAD=%q\n' "${WORKLOAD_PROFILE}"
    printf 'export FAASLORA_TOTAL_REQUESTS=%q\n' "${TOTAL_REQUESTS}"
    printf 'export FAASLORA_SELECTED_NUM_ADAPTERS=%q\n' "${SELECTED_NUM_ADAPTERS}"
    printf 'export FAASLORA_SAMPLING_SEED=%q\n' "${SAMPLING_SEED}"
    printf 'export FAASLORA_STORAGE_BANDWIDTH_MIB_S=%q\n' "${STORAGE_BANDWIDTH_MIB_S}"
    printf 'export FAASLORA_ZIPF_EXPONENT=%q\n' "${ZIPF_EXPONENT}"
    printf 'export FAASLORA_ACTIVE_ADAPTER_CAP=%q\n' "${ACTIVE_ADAPTER_CAP}"
    printf 'export FAASLORA_HOTSET_ROTATION_REQUESTS=%q\n' "${HOTSET_ROTATION_REQUESTS}"
    printf 'export FAASLORA_HOTSET_ROTATION_MODE=%q\n' "${HOTSET_ROTATION_MODE}"
    printf 'export FAASLORA_HOTSET_OVERLAP_FRACTION=%q\n' "${HOTSET_OVERLAP_FRACTION}"
    printf 'export FAASLORA_TIME_SCALE_FACTOR=%q\n' "${TIME_SCALE_FACTOR}"
    printf 'export FAASLORA_SOURCE_ROUND_DIR=%q\n' "${SOURCE_ROUND_DIR}"
    printf 'export FAASLORA_SOURCE_RUN_TAG=%q\n' "${SOURCE_RUN_TAG}"
    printf 'export FAASLORA_SHARED_TRACE_PATH=%q\n' "${TRACE_PATH}"
    printf 'export FAASLORA_SHARED_ADAPTER_SUBSET_PATH=%q\n' "${ADAPTER_SUBSET_PATH}"
    printf 'export FAASLORA_PAPER_ABLATION_SCENARIOS=%q\n' "${SCENARIOS_RAW}"
    printf 'export FAASLORA_PAPER_ABLATION_FORMAL=%q\n' "${FORMAL_RUN}"
    printf 'export FAASLORA_TRACE_ROLE=%q\n' "${TRACE_ROLE}"
    printf 'export FAASLORA_EXPECTED_NON_FEATURE_FROZEN_CONFIG_SHA256=%q\n' "${EXPECTED_NON_FEATURE_FROZEN_CONFIG_SHA256}"
    printf 'export FAASLORA_PAPER_ABLATION_VALIDATION_REGISTRY=%q\n' "${VALIDATION_REGISTRY}"
    printf 'export FAASLORA_PAPER_ABLATION_CONFIG=%q\n' "${CONFIG_PATH}"
    printf 'export FAASLORA_PAPER_ABLATION_GPU_IDS=%q\n' "${GPU_IDS}"
    printf 'export FAASLORA_PAPER_ABLATION_REQUIRE_FEATURE_TRIGGER=%q\n' "${REQUIRE_FEATURE_TRIGGER}"
    printf 'export FAASLORA_MULTI_CYCLE_PHASES=%q\n' "${MULTI_CYCLE_PHASES}"
    printf 'export FAASLORA_IDLE_BETWEEN_PHASES_S=%q\n' "${IDLE_BETWEEN_PHASES_S}"
    printf 'export FAASLORA_V2_TUNING_ENV_SHA256=%q\n' "${TUNING_ENV_SHA256}"
  } >"${ROUND_DIR}/round.env"
}

validate_or_write_round_env() {
  local env_path="${ROUND_DIR}/round.env"
  if [[ "${FORCE_RERUN}" != "1" && -f "${env_path}" ]]; then
    local existing=()
    mapfile -t existing < <(
      bash -c '
        source "$1"
        printf "%s\n" \
          "${FAASLORA_PAPER_ABLATION_RUN_TAG:-}" \
          "${FAASLORA_PROFILE_MODEL:-}" \
          "${FAASLORA_PROFILE_DATASET:-}" \
          "${FAASLORA_PROFILE_WORKLOAD:-}" \
          "${FAASLORA_TOTAL_REQUESTS:-}" \
          "${FAASLORA_SELECTED_NUM_ADAPTERS:-}" \
          "${FAASLORA_SAMPLING_SEED:-}" \
          "${FAASLORA_STORAGE_BANDWIDTH_MIB_S:-}" \
          "${FAASLORA_ZIPF_EXPONENT:-}" \
          "${FAASLORA_ACTIVE_ADAPTER_CAP:-}" \
          "${FAASLORA_HOTSET_ROTATION_REQUESTS:-}" \
          "${FAASLORA_HOTSET_ROTATION_MODE:-}" \
          "${FAASLORA_HOTSET_OVERLAP_FRACTION:-}" \
          "${FAASLORA_TIME_SCALE_FACTOR:-}" \
          "${FAASLORA_SHARED_TRACE_PATH:-}" \
          "${FAASLORA_SHARED_ADAPTER_SUBSET_PATH:-}" \
          "${FAASLORA_PAPER_ABLATION_SCENARIOS:-}" \
          "${FAASLORA_PAPER_ABLATION_FORMAL:-0}" \
          "${FAASLORA_TRACE_ROLE:-auto}" \
          "${FAASLORA_EXPECTED_NON_FEATURE_FROZEN_CONFIG_SHA256:-}" \
          "${FAASLORA_PAPER_ABLATION_VALIDATION_REGISTRY:-}" \
          "${FAASLORA_PAPER_ABLATION_GPU_IDS:-}" \
          "${FAASLORA_PAPER_ABLATION_REQUIRE_FEATURE_TRIGGER:-}" \
          "${FAASLORA_MULTI_CYCLE_PHASES:-}" \
          "${FAASLORA_IDLE_BETWEEN_PHASES_S:-}" \
          "${FAASLORA_V2_TUNING_ENV_SHA256:-}"
      ' bash "${env_path}"
    )
    local names=(
      RUN_TAG MODEL_PROFILE DATASET_PROFILE WORKLOAD_PROFILE TOTAL_REQUESTS
      SELECTED_NUM_ADAPTERS SAMPLING_SEED STORAGE_BANDWIDTH_MIB_S
      ZIPF_EXPONENT ACTIVE_ADAPTER_CAP HOTSET_ROTATION_REQUESTS
      HOTSET_ROTATION_MODE HOTSET_OVERLAP_FRACTION TIME_SCALE_FACTOR
      TRACE_PATH ADAPTER_SUBSET_PATH SCENARIOS
      FORMAL_RUN TRACE_ROLE
      EXPECTED_NON_FEATURE_FROZEN_CONFIG_SHA256
      VALIDATION_REGISTRY
      GPU_IDS REQUIRE_FEATURE_TRIGGER MULTI_CYCLE_PHASES IDLE_BETWEEN_PHASES_S
      TUNING_ENV_SHA256
    )
    local current=(
      "${RUN_TAG}" "${MODEL_PROFILE}" "${DATASET_PROFILE}" "${WORKLOAD_PROFILE}"
      "${TOTAL_REQUESTS}" "${SELECTED_NUM_ADAPTERS}" "${SAMPLING_SEED}"
      "${STORAGE_BANDWIDTH_MIB_S}" "${ZIPF_EXPONENT}" "${ACTIVE_ADAPTER_CAP}"
      "${HOTSET_ROTATION_REQUESTS}" "${HOTSET_ROTATION_MODE}"
      "${HOTSET_OVERLAP_FRACTION}" "${TIME_SCALE_FACTOR}"
      "${TRACE_PATH}" "${ADAPTER_SUBSET_PATH}"
      "${SCENARIOS_RAW}"
      "${FORMAL_RUN}" "${TRACE_ROLE}"
      "${EXPECTED_NON_FEATURE_FROZEN_CONFIG_SHA256}"
      "${VALIDATION_REGISTRY}"
      "${GPU_IDS}" "${REQUIRE_FEATURE_TRIGGER}" "${MULTI_CYCLE_PHASES}"
      "${IDLE_BETWEEN_PHASES_S}" "${TUNING_ENV_SHA256}"
    )
    local i
    for i in "${!names[@]}"; do
      if [[ "${existing[$i]:-}" != "${current[$i]}" ]]; then
        log "[ERROR] frozen round mismatch for ${names[$i]}: existing='${existing[$i]:-}' current='${current[$i]}'"
        log "Use a new round directory; do not overwrite an existing run-key."
        return 1
      fi
    done
    return 0
  fi
  write_round_env
}

write_manifest() {
  "${PYTHON_BIN}" - \
    "${ROUND_DIR}" \
    "${RUN_TAG}" \
    "${SCENARIOS_RAW}" \
    "${TRACE_PATH}" \
    "${ADAPTER_SUBSET_PATH}" \
    "${SOURCE_ROUND_DIR}" \
    "${SOURCE_RUN_TAG}" \
    "${MODEL_PROFILE}" \
    "${DATASET_PROFILE}" \
    "${WORKLOAD_PROFILE}" \
    "${SECTION_ID}" \
    "${ROUND_PURPOSE}" \
    "${FIGURE_TARGETS}" \
    "${MAIN_REPO}" \
    "${STORAGE_BANDWIDTH_MIB_S}" \
    "${FORMAL_RUN}" \
    "${TRACE_ROLE}" \
    "${CONFIG_PATH}" \
    "${VALIDATION_REGISTRY}" \
    "${VALIDATION_EVIDENCE_PATH}" \
    "${EXPECTED_NON_FEATURE_FROZEN_CONFIG_SHA256}" \
    "${GPU_IDS}" \
    "${SELECTED_NUM_ADAPTERS}" \
    "${REQUIRE_FEATURE_TRIGGER}" \
    "${MULTI_CYCLE_PHASES}" \
    "${IDLE_BETWEEN_PHASES_S}" \
    "${TUNING_ENV_SHA256}" \
    "${ZIPF_EXPONENT}" \
    "${ACTIVE_ADAPTER_CAP}" \
    "${HOTSET_ROTATION_REQUESTS}" \
    "${HOTSET_ROTATION_MODE}" \
    "${HOTSET_OVERLAP_FRACTION}" \
    "${TIME_SCALE_FACTOR}" <<'PY'
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

round_dir = Path(sys.argv[1])
run_tag = sys.argv[2]
scenarios = sys.argv[3].split()
trace_path = Path(sys.argv[4])
subset_path = Path(sys.argv[5])
source_round_dir = sys.argv[6]
source_run_tag = sys.argv[7]
model_profile = sys.argv[8]
dataset_profile = sys.argv[9]
workload_profile = sys.argv[10]
section_id = sys.argv[11]
purpose = sys.argv[12]
figure_targets = sys.argv[13].split()
main_repo = Path(sys.argv[14])
bandwidth_mib_s = float(sys.argv[15])
formal_run = sys.argv[16] == "1"
trace_role = sys.argv[17]
config_path = Path(sys.argv[18]).resolve()
validation_registry_path = Path(sys.argv[19]).resolve()
validation_evidence_path = Path(sys.argv[20]).resolve()
expected_non_feature_hash = sys.argv[21].strip().lower()
gpu_ids = [item.strip() for item in sys.argv[22].split(",") if item.strip()]
selected_num_adapters = int(sys.argv[23])
require_feature_trigger = sys.argv[24] == "1"
multi_cycle_phases = int(sys.argv[25]) if sys.argv[25] else None
idle_between_phases_s = float(sys.argv[26]) if sys.argv[26] else None
tuning_env_sha256 = sys.argv[27]
zipf_exponent = float(sys.argv[28])
active_adapter_cap = int(sys.argv[29])
hotset_rotation_requests = int(sys.argv[30])
hotset_rotation_mode = sys.argv[31].strip().lower()
hotset_overlap_fraction = float(sys.argv[32])
time_scale_factor = float(sys.argv[33])
raw_dir = round_dir / "raw" / "faaslora"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git_value(args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(main_repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None

def build_consistency_audit(entries):
    metrics = {
        "ttft_avg_ms": ("lower", 0.02),
        "ttft_p95_ms": ("lower", 0.03),
        "e2e_avg_ms": ("lower", 0.02),
        "e2e_p95_ms": ("lower", 0.03),
        "cost_per_req_usd": ("lower", 0.03),
        "ce": ("higher", 0.03),
        "gpu_hit_rate": ("higher", 0.01),
        "avg_lora_io_ms": ("lower", 0.05),
    }
    by_scenario = {
        entry["scenario"]: entry
        for entry in entries
        if entry.get("exists") and entry.get("summary")
    }
    audit = {
        "baseline": "v2_full",
        "policy": (
            "Compare only scenarios from the same ablation round. "
            "Warnings are not automatic failures; they require explanation before plotting."
        ),
        "status": "incomplete",
        "warnings": [],
        "comparisons": [],
    }
    full = by_scenario.get("v2_full")
    if not full:
        audit["missing"] = ["v2_full"]
        return audit
    full_summary = full["summary"]
    audit["status"] = "ok"
    for scenario, entry in sorted(by_scenario.items()):
        if scenario == "v2_full":
            continue
        row = {"scenario": scenario, "metrics": {}}
        for metric, (direction, tolerance) in metrics.items():
            full_value = full_summary.get(metric)
            scenario_value = entry["summary"].get(metric)
            if full_value is None or scenario_value is None:
                continue
            try:
                full_f = float(full_value)
                scenario_f = float(scenario_value)
            except Exception:
                continue
            if full_f == 0:
                rel = None
            else:
                rel = (scenario_f - full_f) / abs(full_f)
            scenario_beats_full = (
                scenario_f < full_f if direction == "lower" else scenario_f > full_f
            )
            over_tolerance = False
            if rel is not None:
                over_tolerance = (
                    rel < -tolerance if direction == "lower" else rel > tolerance
                )
            metric_row = {
                "direction": direction,
                "full": full_f,
                "scenario": scenario_f,
                "relative_delta_vs_full": rel,
                "scenario_beats_full": scenario_beats_full,
                "warning": bool(scenario_beats_full and over_tolerance),
            }
            row["metrics"][metric] = metric_row
            if metric_row["warning"]:
                audit["warnings"].append(
                    {
                        "scenario": scenario,
                        "metric": metric,
                        "direction": direction,
                        "full": full_f,
                        "scenario": scenario_f,
                        "relative_delta_vs_full": rel,
                        "tolerance": tolerance,
                    }
                )
        audit["comparisons"].append(row)
    if audit["warnings"]:
        audit["status"] = "warning"
    return audit

trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
subset_payload = json.loads(subset_path.read_text(encoding="utf-8"))
trace_load_profile = trace_payload.get("load_profile") or {}
entries = []
csv_rows = []
for scenario in scenarios:
    result = raw_dir / f"{run_tag}_{scenario}_result.json"
    source = raw_dir / f"{run_tag}_{scenario}_source_path.txt"
    entry = {"scenario": scenario, "result_json": str(result), "exists": result.exists()}
    if source.exists():
        entry["source_path"] = source.read_text(encoding="utf-8").strip()
    if result.exists():
        obj = json.loads(result.read_text(encoding="utf-8"))
        entry["bytes"] = result.stat().st_size
        entry["sha256"] = sha256(result)
        entry["system_resolved_config_sha256"] = (
            (obj.get("metadata") or {}).get("system_resolved_config_sha256")
        )
        entry["non_feature_frozen_config_sha256"] = (
            (obj.get("metadata") or {}).get("non_feature_frozen_config_sha256")
        )
        summary = (obj.get("scenario_summaries") or {}).get(scenario, {})
        entry["summary"] = {
            "ttft_avg_ms": summary.get("avg_overall_ttft_ms"),
            "ttft_p95_ms": summary.get("p95_overall_ttft_ms"),
            "e2e_avg_ms": summary.get("avg_overall_e2e_ms"),
            "e2e_p95_ms": summary.get("p95_overall_e2e_ms"),
            "tpot_ms": summary.get("avg_tpot_ms"),
            "tok_per_s": summary.get("throughput_tok_per_s"),
            "cost_per_req_usd": summary.get("monetary_cost_per_request_usd"),
            "ce": summary.get("monetary_ce"),
            "gpu_hit_rate": summary.get("gpu_hit_rate"),
            "avg_lora_io_ms": summary.get("avg_lora_io_ms"),
            "host_cache_memory_backed": summary.get("host_cache_memory_backed"),
        }
        csv_rows.append({"scenario": scenario, **entry["summary"]})
    else:
        csv_rows.append({"scenario": scenario})
    entries.append(entry)

tracked_status = (
    git_value(["status", "--short", "--untracked-files=no"]) or ""
).splitlines()
allowed_dirty = {"configs/generated/lora_manifest_1000.json"}
unexpected_dirty = []
for row in tracked_status:
    path = row[3:].strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    if path not in allowed_dirty:
        unexpected_dirty.append(row)
expected_state_markers = [f"scenario_{scenario}.done" for scenario in scenarios]
actual_state_markers = sorted(path.name for path in (round_dir / "state").glob("*.done"))
failure_marker = round_dir / "state" / "round_failed.current"
campaign_complete = bool(entries) and all(entry.get("exists") for entry in entries) and all(
    marker in actual_state_markers for marker in expected_state_markers
) and not failure_marker.exists()
v2_entries = [entry for entry in entries if str(entry.get("scenario", "")).startswith("v2_")]
non_feature_hashes = {
    str(entry.get("non_feature_frozen_config_sha256") or "").lower()
    for entry in v2_entries
}
non_feature_hash_valid = bool(v2_entries) and all(
    len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)
    for value in non_feature_hashes
)
non_feature_hash_consistent = non_feature_hash_valid and len(non_feature_hashes) == 1
if v2_entries:
    campaign_complete = campaign_complete and non_feature_hash_consistent
configuration_family = {
    "campaign_kind": "v2_a2_a3_ablation",
    "model_profile": model_profile,
    "dataset_profile": dataset_profile,
    "workload_profile": workload_profile,
    "selected_num_adapters": selected_num_adapters,
    "gpu_ids": gpu_ids,
    "generation_contract": "legacy",
}
validation_evidence = None
if validation_evidence_path.is_file():
    evidence_payload = json.loads(validation_evidence_path.read_text(encoding="utf-8"))
    selected_hash = str(
        evidence_payload.get("selected_non_feature_frozen_config_sha256") or ""
    ).lower()
    if selected_hash != expected_non_feature_hash:
        raise SystemExit("held-out validation evidence selected hash mismatch")
    validation_evidence = {
        "path": str(validation_evidence_path),
        "sha256": sha256(validation_evidence_path),
        "bytes": validation_evidence_path.stat().st_size,
        "selected_non_feature_frozen_config_sha256": selected_hash,
        "successful_validation_manifest": (
            (evidence_payload.get("successful_validation") or {}).get("manifest")
        ),
        "successful_validation_manifest_sha256": (
            (evidence_payload.get("successful_validation") or {}).get("manifest_sha256")
        ),
        "successful_validation_manifest_bytes": (
            (evidence_payload.get("successful_validation") or {}).get("manifest_bytes")
        ),
        "source_commit": evidence_payload.get("source_commit"),
        "config_path": evidence_payload.get("config_path"),
        "config_sha256": evidence_payload.get("config_sha256"),
        "configuration_family_id": evidence_payload.get("configuration_family_id"),
        "registry_path": str(validation_registry_path),
        "registry_sha256_after_freeze": evidence_payload.get(
            "registry_sha256_after_freeze"
        ),
    }
if formal_run and trace_role == "heldout" and validation_evidence is None:
    raise SystemExit("formal held-out manifest requires immutable seed41 validation evidence")
if formal_run and trace_role == "heldout" and (
    not non_feature_hash_consistent
    or next(iter(non_feature_hashes)) != expected_non_feature_hash
):
    raise SystemExit("formal held-out results do not match the seed41 frozen non-feature hash")

manifest = {
    "status": "complete" if campaign_complete else "incomplete",
    "formal_run": formal_run,
    "trace_role": trace_role,
    "run_tag": run_tag,
    "section_id": section_id,
    "purpose": purpose,
    "figure_targets": figure_targets,
    "round_dir": str(round_dir),
    "source_round_dir": source_round_dir,
    "source_run_tag": source_run_tag,
    "model_profile": model_profile,
    "dataset_profile": dataset_profile,
    "workload_profile": workload_profile,
    "configuration_family": configuration_family,
    "config_snapshot": {
        "path": str(config_path),
        "sha256": sha256(config_path),
        "bytes": config_path.stat().st_size,
    },
    "seed41_validation_evidence": validation_evidence,
    "code_snapshot": {
        "repo": str(main_repo),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_branch": git_value(["branch", "--show-current"]),
        "git_status_short": git_value(["status", "--short"]),
        "tracked_dirty_paths": tracked_status,
        "unexpected_tracked_dirty_paths": unexpected_dirty,
        "source_clean_for_formal": not unexpected_dirty,
    },
    "shared_trace": {
        "path": str(trace_path),
        "sha256": sha256(trace_path),
        "requests": len(trace_payload.get("requests", [])),
        "selected_num_adapters": trace_payload.get("selected_num_adapters"),
        "sampling_seed": trace_payload.get("sampling_seed"),
        "generation_seed": trace_payload.get("sampling_seed"),
        "active_adapter_cap": trace_payload.get("active_adapter_cap")
            or (trace_payload.get("load_profile") or {}).get("active_adapter_cap"),
        "hotset_rotation_requests": trace_payload.get("hotset_rotation_requests")
            or (trace_payload.get("load_profile") or {}).get("hotset_rotation_requests"),
        "zipf_exponent": trace_load_profile.get("zipf_exponent"),
        "hotset_rotation_mode": trace_load_profile.get("hotset_rotation_mode"),
        "hotset_overlap_fraction": trace_load_profile.get(
            "hotset_overlap_fraction"
        ),
        "configured_time_scale_factor": trace_payload.get(
            "configured_time_scale_factor",
            trace_load_profile.get("configured_time_scale_factor"),
        ),
        "effective_time_scale_factor": trace_payload.get(
            "effective_time_scale_factor",
            trace_load_profile.get("effective_time_scale_factor"),
        ),
    },
    "shared_adapter_subset": {
        "path": str(subset_path),
        "sha256": sha256(subset_path),
        "selected_num_adapters": subset_payload.get("selected_num_adapters"),
        "sampling_seed": subset_payload.get("sampling_seed"),
        "adapter_count": len(subset_payload.get("adapters", [])),
    },
    "scenarios": scenarios,
    "bandwidth_mib_s": bandwidth_mib_s,
    "workload_axes": {
        "zipf_exponent": zipf_exponent,
        "active_adapter_cap": active_adapter_cap,
        "hotset_rotation_requests": hotset_rotation_requests,
        "hotset_rotation_mode": hotset_rotation_mode,
        "hotset_overlap_fraction": hotset_overlap_fraction,
        "time_scale_factor": time_scale_factor,
    },
    "mechanism_trigger_gate_required": require_feature_trigger,
    "multi_cycle_phases": multi_cycle_phases,
    "idle_between_phases_s": idle_between_phases_s,
    "tuning_env_sha256": tuning_env_sha256,
    "non_feature_frozen_config_sha256": (
        next(iter(non_feature_hashes)) if non_feature_hash_consistent else None
    ),
    "non_feature_frozen_config_consistent": non_feature_hash_consistent,
    "entries": entries,
    "state_markers": actual_state_markers,
    "required_state_markers": expected_state_markers,
    "runner_failure": (
        failure_marker.read_text(encoding="utf-8").strip()
        if failure_marker.is_file()
        else None
    ),
}
(round_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
audit = build_consistency_audit(entries)
(round_dir / "ablation_consistency_audit.json").write_text(
    json.dumps(audit, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
csv_path = round_dir / "summary_metrics.csv"
fieldnames = [
    "scenario",
    "ttft_avg_ms",
    "ttft_p95_ms",
    "e2e_avg_ms",
    "e2e_p95_ms",
    "tpot_ms",
    "tok_per_s",
    "cost_per_req_usd",
    "ce",
    "gpu_hit_rate",
    "avg_lora_io_ms",
    "host_cache_memory_backed",
]
with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in csv_rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
print(round_dir / "MANIFEST.json")
PY
}

ROUND_EXIT_CLEANUP_RUNNING=0
cleanup_on_round_exit() {
  local status=$?
  if (( ROUND_EXIT_CLEANUP_RUNNING != 0 )); then
    return "${status}"
  fi
  ROUND_EXIT_CLEANUP_RUNNING=1
  trap - EXIT INT TERM HUP
  set +e
  if (( status != 0 )); then
    printf 'exit_status=%s timestamp=%s\n' \
      "${status}" "$(date --iso-8601=seconds)" >"${STATE_DIR}/round_failed.current"
    if [[ -f "${TRACE_PATH}" && -f "${ADAPTER_SUBSET_PATH}" ]]; then
      write_manifest >>"${LOG_DIR}/failure_manifest.log" 2>&1 || true
    fi
  fi
  kill_known_gpu_residuals || true
  wait_gpu_idle
  local cleanup_status=$?
  if (( status == 0 && cleanup_status != 0 )); then
    status="${cleanup_status}"
  fi
  ROUND_EXIT_CLEANUP_RUNNING=0
  exit "${status}"
}
trap cleanup_on_round_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

if [[ ! -x "${RUNNER}" ]]; then
  log "[ERROR] runner not found or not executable: ${RUNNER}"
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  log "[ERROR] config not found: ${CONFIG_PATH}"
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  log "[ERROR] Python not executable: ${PYTHON_BIN}"
  exit 1
fi
if [[ ! -f "${VALIDATION_REGISTRY_TOOL}" ]]; then
  log "[ERROR] seed41 validation registry tool not found: ${VALIDATION_REGISTRY_TOOL}"
  exit 1
fi

validate_shared_artifacts
read -r -a SCENARIOS <<< "${SCENARIOS_RAW}"
TUNING_ENV_SHA256="$(faaslora_tuning_env_sha256)"
if [[ ! "${TUNING_ENV_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  log "[ERROR] failed to resolve frozen tuning-environment digest"
  exit 1
fi
validate_scenarios
validate_source_and_seed_protocol
validate_or_write_round_env
cp -f "${TRACE_PATH}" "${SHARED_DIR}/$(basename "${TRACE_PATH}")"
cp -f "${ADAPTER_SUBSET_PATH}" "${SHARED_DIR}/$(basename "${ADAPTER_SUBSET_PATH}")"
log "round_dir=${ROUND_DIR}"
log "run_tag=${RUN_TAG}"
log "section=${SECTION_ID} purpose=${ROUND_PURPOSE} figures=${FIGURE_TARGETS}"
log "scenarios=${SCENARIOS[*]}"
log "trace=${TRACE_PATH}"
log "adapter_subset=${ADAPTER_SUBSET_PATH}"
log "trace_role=${TRACE_ROLE} formal=${FORMAL_RUN}"

# A retry preserves the earlier incomplete MANIFEST/logs but starts with a
# fresh current-attempt failure marker.
rm -f "${STATE_DIR}/round_failed.current"

if [[ "${DRY_RUN}" == "1" ]]; then
  for scenario in "${SCENARIOS[@]}"; do
    [[ -z "${scenario}" ]] && continue
    log "[dry-run] would run scenario=${scenario} result_tag=${RUN_TAG}_${scenario}"
  done
  write_manifest
  log "[dry-run] shared artifacts and round layout validated: ${ROUND_DIR}"
  exit 0
fi

for scenario in "${SCENARIOS[@]}"; do
  [[ -z "${scenario}" ]] && continue
  stage="scenario_${scenario}"
  result_tag="${RUN_TAG}_${scenario}"
  copied_result="${RAW_DIR}/${RUN_TAG}_${scenario}_result.json"

  if is_done "${stage}" && [[ -f "${copied_result}" ]]; then
    log "stage=${stage} already done; validating copied result and skipping"
    validate_result_json "${copied_result}" "${scenario}"
    continue
  fi

  recovered_result="$(find_result_json "${scenario}" "${result_tag}")"
  if [[ -n "${recovered_result}" && -f "${recovered_result}" ]]; then
    log "stage=${stage} has existing source result; validating and marking done without rerun"
    validate_result_json "${recovered_result}" "${scenario}"
    cp -f "${recovered_result}" "${copied_result}"
    printf '%s\n' "${recovered_result}" >"${RAW_DIR}/${RUN_TAG}_${scenario}_source_path.txt"
    mark_done "${stage}"
    continue
  fi

  check_gpu_idle
  log "running scenario=${scenario} result_tag=${result_tag}"
  (
    export FAASLORA_PROFILE_MODEL="${MODEL_PROFILE}"
    export FAASLORA_PROFILE_DATASET="${DATASET_PROFILE}"
    export FAASLORA_PROFILE_WORKLOAD="${WORKLOAD_PROFILE}"
    export FAASLORA_TOTAL_REQUESTS="${TOTAL_REQUESTS}"
    export FAASLORA_WORKLOAD_SEED="${SAMPLING_SEED}"
    export FAASLORA_GENERATION_SEED="${SAMPLING_SEED}"
    export FAASLORA_SHARED_TRACE_PATH="${TRACE_PATH}"
    export FAASLORA_SHARED_ADAPTER_SUBSET_PATH="${ADAPTER_SUBSET_PATH}"
    export FAASLORA_RESULTS_TAG="${result_tag}"
    export FAASLORA_STORAGE_BANDWIDTH_MIB_S="${STORAGE_BANDWIDTH_MIB_S}"
    export FAASLORA_ZIPF_EXPONENT="${ZIPF_EXPONENT}"
    export FAASLORA_ACTIVE_ADAPTER_CAP="${ACTIVE_ADAPTER_CAP}"
    export FAASLORA_HOTSET_ROTATION_REQUESTS="${HOTSET_ROTATION_REQUESTS}"
    export FAASLORA_HOTSET_ROTATION_MODE="${HOTSET_ROTATION_MODE}"
    export FAASLORA_HOTSET_OVERLAP_FRACTION="${HOTSET_OVERLAP_FRACTION}"
    export FAASLORA_TIME_SCALE_FACTOR="${TIME_SCALE_FACTOR}"
    export FAASLORA_NVME_CACHE_DIR="${ROUND_DIR}/cache/nvme"
    export FAASLORA_HOST_CACHE_DIR="/dev/shm/faaslora_eurosys27_v2/${RUN_TAG}"
    export FAASLORA_SYSTEM_RESOLVED_CONFIG_SHA256="$(
      faaslora_system_resolved_config_sha256 "${scenario}"
    )"
    export FAASLORA_EXPECTED_NON_FEATURE_FROZEN_CONFIG_SHA256="${EXPECTED_NON_FEATURE_FROZEN_CONFIG_SHA256}"
    export FAASLORA_TRACE_ROLE="${TRACE_ROLE}"
    export FAASLORA_FORMAL_RUN="${FORMAL_RUN}"
    if [[ -n "${MULTI_CYCLE_PHASES}" ]]; then
      export FAASLORA_MULTI_CYCLE_PHASES="${MULTI_CYCLE_PHASES}"
    fi
    if [[ -n "${IDLE_BETWEEN_PHASES_S}" ]]; then
      export FAASLORA_IDLE_BETWEEN_PHASES_S="${IDLE_BETWEEN_PHASES_S}"
    fi
    export PYTHONUNBUFFERED=1
    cd "${MAIN_REPO}"
    run_logged "${stage}" "${RUNNER}" \
      --config "${CONFIG_PATH}" \
      --scenario "${scenario}" \
      --backend vllm \
      --model-profile "${MODEL_PROFILE}" \
      --dataset-profile "${DATASET_PROFILE}" \
      --workload-profile "${WORKLOAD_PROFILE}" \
      --num-adapters "${SELECTED_NUM_ADAPTERS}"
  )

  kill_known_gpu_residuals
  wait_gpu_idle

  result_path="$(find_result_json "${scenario}" "${result_tag}")"
  if [[ -z "${result_path}" || ! -f "${result_path}" ]]; then
    log "[ERROR] unable to locate result JSON for scenario=${scenario} result_tag=${result_tag}"
    exit 1
  fi
  validate_result_json "${result_path}" "${scenario}"
  cp -f "${result_path}" "${copied_result}"
  printf '%s\n' "${result_path}" >"${RAW_DIR}/${RUN_TAG}_${scenario}_source_path.txt"
  mark_done "${stage}"
  log "stage=${stage} done result=${copied_result}"
done

rm -f "${STATE_DIR}/round_failed.current"
write_manifest
register_successful_validation_if_complete
log "FaaSLoRA paper ablation round complete: ${ROUND_DIR}"
