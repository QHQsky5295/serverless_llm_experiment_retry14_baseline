#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
MAIN_REPO="${SLLM_MAIN_REPO:-/home/qhq/serverless_llm_experiment_retry14_baseline}"
RESULT_DIR="${SLLM_RESULT_DIR:-${ROOT_DIR}/results/replay}"
LOG_DIR="${SLLM_LOG_DIR:-${ROOT_DIR}/results/logs}"
SHARED_INPUT_DIR="${SLLM_SHARED_INPUT_DIR:-${ROOT_DIR}/results/shared_inputs}"
CONFIG_PATH="${SLLM_CONFIG_PATH:-${MAIN_REPO}/configs/experiments.yaml}"
VLLM_PYTHON="${VLLM_PYTHON:-/home/qhq/anaconda3/envs/LLM_vllm0102/bin/python}"

MODEL_PROFILE="${SLLM_MODEL_PROFILE:?SLLM_MODEL_PROFILE is required}"
DATASET_PROFILE="${SLLM_DATASET_PROFILE:?SLLM_DATASET_PROFILE is required}"
WORKLOAD_PROFILE="${SLLM_WORKLOAD_PROFILE:?SLLM_WORKLOAD_PROFILE is required}"
TOTAL_REQUESTS="${SLLM_TOTAL_REQUESTS:?SLLM_TOTAL_REQUESTS is required}"
SELECTED_NUM_ADAPTERS="${SLLM_SELECTED_NUM_ADAPTERS:?SLLM_SELECTED_NUM_ADAPTERS is required}"
SAMPLING_SEED="${SLLM_SAMPLING_SEED:-42}"
GENERATION_SEED="${SLLM_GENERATION_SEED:-${SAMPLING_SEED}}"
RUN_TAG="${SLLM_RUN_TAG:-${MODEL_PROFILE}_r${TOTAL_REQUESTS}_a${SELECTED_NUM_ADAPTERS}_seed${SAMPLING_SEED}}"
SHARED_TRACE_PATH="${SLLM_SHARED_TRACE_PATH:?SLLM_SHARED_TRACE_PATH is required}"
SHARED_ADAPTER_SUBSET_PATH="${SLLM_SHARED_ADAPTER_SUBSET_PATH:?SLLM_SHARED_ADAPTER_SUBSET_PATH is required}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8363}"
VLLM_PORT_STRIDE="${VLLM_PORT_STRIDE:-10}"
VLLM_GPU_IDS="${VLLM_GPU_IDS:-0,1,2,3}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-}"
VLLM_DATA_PARALLEL_REPLICAS="${VLLM_DATA_PARALLEL_REPLICAS:-}"
VLLM_SLEEP_SCALE="${VLLM_SLEEP_SCALE:-1.0}"
VLLM_TIMEOUT_S="${VLLM_TIMEOUT_S:-3600}"
VLLM_DRY_RUN="${VLLM_DRY_RUN:-0}"
VLLM_MIN_OUTPUT_TOKENS="${VLLM_MIN_OUTPUT_TOKENS:-1}"
VLLM_INCLUDE_STREAM_USAGE="${VLLM_INCLUDE_STREAM_USAGE:-1}"
VLLM_EMPTY_SUCCESS_RETRIES="${VLLM_EMPTY_SUCCESS_RETRIES:-2}"
VLLM_EMPTY_SUCCESS_RETRY_DELAY_S="${VLLM_EMPTY_SUCCESS_RETRY_DELAY_S:-0.5}"
VLLM_HOST_MIN_MEM_GB="${VLLM_HOST_MIN_MEM_GB:-32}"
VLLM_MEM_WATCH_INTERVAL_S="${VLLM_MEM_WATCH_INTERVAL_S:-2}"
VLLM_SMOKE_ONLY="${VLLM_SMOKE_ONLY:-0}"
VLLM_SMOKE_MAX_TOKENS="${VLLM_SMOKE_MAX_TOKENS:-1}"
VLLM_MAX_LORAS="${VLLM_MAX_LORAS:-}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
VLLM_MAX_REPLAY_REQUESTS="${VLLM_MAX_REPLAY_REQUESTS:-0}"
VLLM_LORA_REGISTRATION_MODE="${VLLM_LORA_REGISTRATION_MODE:-static}"
VLLM_DYNAMIC_LORA_ROUTING="${VLLM_DYNAMIC_LORA_ROUTING:-round_robin}"
VLLM_DYNAMIC_LORA_HOT_PAIR_THRESHOLD="${VLLM_DYNAMIC_LORA_HOT_PAIR_THRESHOLD:-8}"
VLLM_DYNAMIC_LORA_HOT_PAIR_MAX_ADAPTERS="${VLLM_DYNAMIC_LORA_HOT_PAIR_MAX_ADAPTERS:-32}"
VLLM_DISABLE_FRONTEND_MULTIPROCESSING="${VLLM_DISABLE_FRONTEND_MULTIPROCESSING:-0}"
VLLM_ABORT_AFTER_FAILURES="${VLLM_ABORT_AFTER_FAILURES:-8}"
VLLM_ABORT_FAILURES_MIN_DONE="${VLLM_ABORT_FAILURES_MIN_DONE:-32}"
VLLM_EXPECTED_REPLAY_TOTAL="${TOTAL_REQUESTS}"
if [[ "${VLLM_MAX_REPLAY_REQUESTS}" =~ ^[0-9]+$ ]] && (( VLLM_MAX_REPLAY_REQUESTS > 0 && VLLM_MAX_REPLAY_REQUESTS < TOTAL_REQUESTS )); then
  VLLM_EXPECTED_REPLAY_TOTAL="${VLLM_MAX_REPLAY_REQUESTS}"
fi

mkdir -p "${RESULT_DIR}" "${LOG_DIR}" "${SHARED_INPUT_DIR}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] experiments config not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SHARED_TRACE_PATH}" ]]; then
  echo "[ERROR] shared trace artifact not found: ${SHARED_TRACE_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SHARED_ADAPTER_SUBSET_PATH}" ]]; then
  echo "[ERROR] shared adapter subset artifact not found: ${SHARED_ADAPTER_SUBSET_PATH}" >&2
  exit 1
fi
if [[ ! -x "${VLLM_PYTHON}" ]]; then
  echo "[ERROR] vLLM python not found or not executable: ${VLLM_PYTHON}" >&2
  exit 1
fi
case "${VLLM_LORA_REGISTRATION_MODE}" in
  static|dynamic) ;;
  *)
    echo "[ERROR] VLLM_LORA_REGISTRATION_MODE must be static or dynamic, got ${VLLM_LORA_REGISTRATION_MODE}" >&2
    exit 1
    ;;
esac
case "${VLLM_DYNAMIC_LORA_ROUTING}" in
  round_robin|adapter_hash|adapter_pair_hash|adaptive_hot_pair_hash) ;;
  *)
    echo "[ERROR] VLLM_DYNAMIC_LORA_ROUTING must be round_robin, adapter_hash, adapter_pair_hash, or adaptive_hot_pair_hash; got ${VLLM_DYNAMIC_LORA_ROUTING}" >&2
    exit 1
    ;;
esac

stop_vllm_servers() {
  if [[ -n "${VLLM_SERVER_PIDS:-}" ]]; then
    for pid in ${VLLM_SERVER_PIDS}; do
      kill -- "-${pid}" 2>/dev/null || true
      kill "${pid}" 2>/dev/null || true
    done
    for pid in ${VLLM_SERVER_PIDS}; do
      wait "${pid}" 2>/dev/null || true
    done
    VLLM_SERVER_PIDS=""
  fi
}

cleanup() {
  local status=$?
  stop_vllm_servers
  return "${status}"
}
trap cleanup EXIT

mem_available_kib() {
  awk '/^MemAvailable:/ {print $2; exit}' /proc/meminfo
}

mem_threshold_kib() {
  PYTHONNOUSERSITE=1 "${VLLM_PYTHON}" - "${VLLM_HOST_MIN_MEM_GB}" <<'PY'
import sys
value = float(sys.argv[1])
print(int(value * 1024 * 1024))
PY
}

require_host_memory() {
  local context="$1"
  local threshold_kib
  threshold_kib="$(mem_threshold_kib)"
  if (( threshold_kib <= 0 )); then
    return 0
  fi
  local available_kib
  available_kib="$(mem_available_kib)"
  if [[ -z "${available_kib}" ]]; then
    echo "[WARN] unable to read MemAvailable while checking host memory for ${context}" >&2
    return 0
  fi
  if (( available_kib < threshold_kib )); then
    echo "[ERROR] host memory guard tripped during ${context}: MemAvailable=$((available_kib / 1024)) MiB < required=$((threshold_kib / 1024)) MiB." >&2
    echo "        Aborting vLLM stage before Linux OOM can destabilize SSH/system services." >&2
    stop_vllm_servers
    return 1
  fi
}

monitor_replay_and_servers() {
  local replay_pid="$1"
  local replica_pid
  while kill -0 "${replay_pid}" 2>/dev/null; do
    if ! require_host_memory "vLLM replay"; then
      kill "${replay_pid}" 2>/dev/null || true
      return 1
    fi
    for replica_pid in ${VLLM_SERVER_PIDS:-}; do
      if ! kill -0 "${replica_pid}" 2>/dev/null; then
        echo "[ERROR] vLLM server pid=${replica_pid} exited during replay. Aborting this stage instead of continuing with invalid partial results." >&2
        kill "${replay_pid}" 2>/dev/null || true
        stop_vllm_servers
        return 1
      fi
    done
    sleep "${VLLM_MEM_WATCH_INTERVAL_S}"
  done
  return 0
}

ensure_port_is_free() {
  local port="$1"
  local label="$2"
  local pids=()
  mapfile -t pids < <(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null | awk '!seen[$0]++')
  if (( ${#pids[@]} == 0 )); then
    return 0
  fi
  echo "      clearing stale ${label} listener(s) on port=${port}: ${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true
  for _ in $(seq 1 15); do
    sleep 1
    mapfile -t pids < <(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null | awk '!seen[$0]++')
    if (( ${#pids[@]} == 0 )); then
      return 0
    fi
  done
  echo "[ERROR] stale ${label} listener still occupies port ${port}: ${pids[*]}" >&2
  exit 1
}

ensure_gpu_set_idle() {
  local gpu_csv="$1"
  local label="$2"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  local gpu_ids=()
  local gpu=""
  IFS=',' read -r -a gpu_ids <<< "${gpu_csv}"
  for gpu in "${gpu_ids[@]}"; do
    gpu="$(echo "${gpu}" | xargs)"
    [[ -z "${gpu}" ]] && continue
    local procs=()
    local query_output=""
    query_output="$(nvidia-smi --id="${gpu}" --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true)"
    mapfile -t procs < <(printf '%s\n' "${query_output}" | sed '/^[[:space:]]*$/d' | grep -E '^[[:space:]]*[0-9]+[[:space:]]*,')
    if (( ${#procs[@]} > 0 )); then
      echo "[ERROR] ${label} target GPU ${gpu} is not idle; active compute processes detected:" >&2
      printf '  %s\n' "${procs[@]}" >&2
      echo "        Clean up earlier experiments before starting a formal run." >&2
      exit 1
    fi
  done
}

echo "[1/5] Validating shared trace and adapter subset"
PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${VLLM_PYTHON}" - "${SHARED_TRACE_PATH}" "${SHARED_ADAPTER_SUBSET_PATH}" "${MODEL_PROFILE}" "${DATASET_PROFILE}" "${WORKLOAD_PROFILE}" "${TOTAL_REQUESTS}" "${SELECTED_NUM_ADAPTERS}" "${SAMPLING_SEED}" <<'PY'
import json
import sys
from pathlib import Path

trace_path = Path(sys.argv[1])
subset_path = Path(sys.argv[2])
model_profile = sys.argv[3]
dataset_profile = sys.argv[4]
workload_profile = sys.argv[5]
total_requests = int(sys.argv[6])
selected_num_adapters = int(sys.argv[7])
sampling_seed = int(sys.argv[8])

trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
subset_payload = json.loads(subset_path.read_text(encoding="utf-8"))

for field, expected in (
    ("model_profile", model_profile),
    ("dataset_profile", dataset_profile),
    ("workload_profile", workload_profile),
):
    observed = trace_payload.get(field)
    if observed != expected:
        raise SystemExit(f"trace {field} mismatch: expected {expected}, got {observed}")
    observed_subset = subset_payload.get(field)
    if observed_subset != expected:
        raise SystemExit(f"subset {field} mismatch: expected {expected}, got {observed_subset}")

if int(trace_payload.get("total_requests", -1)) != total_requests:
    raise SystemExit(
        f"trace total_requests mismatch: expected {total_requests}, got {trace_payload.get('total_requests')}"
    )
if int(trace_payload.get("selected_num_adapters", -1)) != selected_num_adapters:
    raise SystemExit(
        f"trace selected_num_adapters mismatch: expected {selected_num_adapters}, got {trace_payload.get('selected_num_adapters')}"
    )
if int(subset_payload.get("selected_num_adapters", -1)) != selected_num_adapters:
    raise SystemExit(
        f"subset selected_num_adapters mismatch: expected {selected_num_adapters}, got {subset_payload.get('selected_num_adapters')}"
    )
if int(trace_payload.get("sampling_seed", -1)) != sampling_seed:
    raise SystemExit(
        f"trace sampling_seed mismatch: expected {sampling_seed}, got {trace_payload.get('sampling_seed')}"
    )
if int(subset_payload.get("sampling_seed", -1)) != sampling_seed:
    raise SystemExit(
        f"subset sampling_seed mismatch: expected {sampling_seed}, got {subset_payload.get('sampling_seed')}"
    )

subset_ids = {str(item["id"]) for item in subset_payload.get("adapters", [])}
trace_ids = {
    str(req.get("adapter_id"))
    for req in trace_payload.get("requests", [])
    if req.get("adapter_id") is not None
}
if len(subset_ids) != selected_num_adapters:
    raise SystemExit(
        f"subset adapter cardinality mismatch: expected {selected_num_adapters}, got {len(subset_ids)}"
    )
if not trace_ids.issubset(subset_ids):
    missing = sorted(trace_ids - subset_ids)
    raise SystemExit(f"trace references adapters outside subset: {missing[:8]}")
PY

readarray -t _CFG < <(
  PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${VLLM_PYTHON}" - "${ROOT_DIR}" "${CONFIG_PATH}" "${MODEL_PROFILE}" "${DATASET_PROFILE}" "${WORKLOAD_PROFILE}" "${VLLM_TENSOR_PARALLEL_SIZE}" "${VLLM_DATA_PARALLEL_REPLICAS}" "${VLLM_GPU_IDS}" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]) / "scripts"))
from summarize_serverlessllm_replay import _load_yaml, _resolve_profiles  # type: ignore

cfg = _load_yaml(Path(sys.argv[2]))
model_cfg, _adapters_cfg, _datasets_cfg, workload_cfg, coord_cfg = _resolve_profiles(
    cfg,
    sys.argv[3],
    sys.argv[4],
    sys.argv[5],
)
tp_override = str(sys.argv[6] or "").strip()
dp_override = str(sys.argv[7] or "").strip()
gpu_ids = [item.strip() for item in str(sys.argv[8]).split(",") if item.strip()]
cost_model = dict(cfg.get("cost_model", {}) or {})

def as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    return default

tp = int(tp_override) if tp_override else int(model_cfg.get("tensor_parallel_size", 1) or 1)
if tp <= 0:
    raise SystemExit(f"invalid tensor parallel size: {tp}")
if dp_override:
    dp = int(dp_override)
else:
    dp = max(1, len(gpu_ids) // tp)
if dp <= 0:
    raise SystemExit(f"invalid data parallel replicas: {dp}")
required_gpus = dp * tp
if len(gpu_ids) < required_gpus:
    raise SystemExit(
        f"vLLM DP/TP topology requires {required_gpus} GPU ids, got {len(gpu_ids)}"
    )

print(float(cost_model.get("base_cost_usd", 0.001)))
print(float(cost_model.get("input_token_cost_usd", 0.0000015)))
print(float(cost_model.get("output_token_cost_usd", 0.000002)))
print(float(workload_cfg.get("ttft_slo_ms", coord_cfg.get("ttft_slo_ms", 5000.0)) or 5000.0))
print(str(model_cfg.get("name", "")))
print(int(model_cfg.get("max_model_len", 0) or 0))
print(int(model_cfg.get("max_input_len", 0) or 0))
print(int(model_cfg.get("max_output_tokens_cap", 0) or 0))
print(tp)
print(dp)
print(float(model_cfg.get("gpu_memory_utilization", 0.7) or 0.7))
print(str(model_cfg.get("dtype", "auto") or "auto"))
print(int(model_cfg.get("max_loras", 1) or 1))
print(int(model_cfg.get("max_lora_rank", 16) or 16))
print(int(model_cfg.get("max_num_seqs", 1) or 1))
print(int(model_cfg.get("max_num_batched_tokens", 1024) or 1024))
print(1 if as_bool(model_cfg.get("enable_chunked_prefill"), False) else 0)
print(1 if as_bool(model_cfg.get("enable_prefix_caching"), False) else 0)
print(1 if as_bool(model_cfg.get("enforce_eager"), False) else 0)
print(1 if as_bool(model_cfg.get("vllm_use_v1"), False) else 0)
print(str(model_cfg.get("vllm_attention_backend", "") or ""))
print(1 if as_bool(model_cfg.get("vllm_use_flashinfer_sampler"), False) else 0)
PY
)

if (( ${#_CFG[@]} != 22 )); then
  echo "[ERROR] failed to resolve vLLM fair-run parameters from ${CONFIG_PATH}" >&2
  exit 1
fi

BASE_COST_USD="${_CFG[0]}"
INPUT_TOKEN_COST_USD="${_CFG[1]}"
OUTPUT_TOKEN_COST_USD="${_CFG[2]}"
TTFT_SLO_MS="${_CFG[3]}"
MODEL_PATH="${_CFG[4]}"
PROMPT_GUARD_TOKENIZER_MODEL="${MODEL_PATH}"
PROMPT_GUARD_MAX_MODEL_LEN="${_CFG[5]}"
PROMPT_GUARD_MAX_INPUT_LEN="${_CFG[6]}"
PROMPT_GUARD_MAX_OUTPUT_TOKENS_CAP="${_CFG[7]}"
TP_EFFECTIVE="${_CFG[8]}"
DP_REPLICAS="${_CFG[9]}"
GPU_MEMORY_UTILIZATION="${_CFG[10]}"
DTYPE="${_CFG[11]}"
MAX_LORAS="${_CFG[12]}"
MAX_LORA_RANK="${_CFG[13]}"
MAX_NUM_SEQS="${_CFG[14]}"
MAX_NUM_BATCHED_TOKENS="${_CFG[15]}"
ENABLE_CHUNKED_PREFILL="${_CFG[16]}"
ENABLE_PREFIX_CACHING="${_CFG[17]}"
ENFORCE_EAGER="${_CFG[18]}"
VLLM_USE_V1_EFFECTIVE="${_CFG[19]}"
VLLM_ATTENTION_BACKEND_EFFECTIVE="${_CFG[20]}"
VLLM_USE_FLASHINFER_SAMPLER_EFFECTIVE="${_CFG[21]}"

apply_positive_int_override() {
  local env_name="$1"
  local current_value="$2"
  local override_value="${!env_name:-}"
  if [[ -z "${override_value}" || "${override_value}" == "0" ]]; then
    printf '%s\n' "${current_value}"
    return 0
  fi
  if ! [[ "${override_value}" =~ ^[0-9]+$ ]] || (( override_value <= 0 )); then
    echo "[ERROR] ${env_name} must be a positive integer, got ${override_value}" >&2
    return 1
  fi
  printf '%s\n' "${override_value}"
}

MAX_LORAS="$(apply_positive_int_override VLLM_MAX_LORAS "${MAX_LORAS}")"
MAX_NUM_SEQS="$(apply_positive_int_override VLLM_MAX_NUM_SEQS "${MAX_NUM_SEQS}")"
MAX_NUM_BATCHED_TOKENS="$(apply_positive_int_override VLLM_MAX_NUM_BATCHED_TOKENS "${MAX_NUM_BATCHED_TOKENS}")"
if (( MAX_LORAS < MAX_NUM_SEQS )); then
  echo "[ERROR] max_loras=${MAX_LORAS} is smaller than max_num_seqs=${MAX_NUM_SEQS}." >&2
  echo "        Set VLLM_MAX_LORAS>=VLLM_MAX_NUM_SEQS for formal multi-LoRA runs." >&2
  exit 1
fi

VLLM_TOPOLOGY_LABEL="dp${DP_REPLICAS}_tp${TP_EFFECTIVE}"
RESULT_TAG="${VLLM_RESULT_TAG:-${RUN_TAG}_vllm_${VLLM_TOPOLOGY_LABEL}}"
REPLAY_PATH="${RESULT_DIR}/${RESULT_TAG}_replay.json"
SUMMARY_PATH="${RESULT_DIR}/${RESULT_TAG}_summary.json"
LAUNCH_SPEC_PATH="${SHARED_INPUT_DIR}/${RESULT_TAG}_launch.yaml"
FLEET_SPEC_PATH="${SHARED_INPUT_DIR}/${RESULT_TAG}_fleet.yaml"
LORA_MODULES_JSON="${SHARED_INPUT_DIR}/${RESULT_TAG}_lora_modules.json"
LORA_MODULES_TXT="${SHARED_INPUT_DIR}/${RESULT_TAG}_lora_modules.txt"
SERVER_LOG_PREFIX="${LOG_DIR}/${RESULT_TAG}_server"

resolve_max_cpu_loras() {
  # Keep the experiment adapter universe unchanged: every replay request remains
  # LoRA-bound against the same SELECTED_NUM_ADAPTERS pool. In static mode those
  # adapters are exposed through --lora-modules; in dynamic mode the same module
  # map is used by replay-time /v1/load_lora_adapter calls. This value only
  # limits vLLM's per-replica CPU LoRA cache.
  local requested="${VLLM_MAX_CPU_LORAS:-auto}"
  local minimum="${MAX_LORAS}"
  if (( MAX_NUM_SEQS > minimum )); then
    minimum="${MAX_NUM_SEQS}"
  fi

  local value
  if [[ "${requested}" == "auto" || -z "${requested}" || "${requested}" == "0" ]]; then
    value=$(( MAX_LORAS * 4 ))
    if (( value < 16 )); then
      value=16
    fi
    if (( value < minimum )); then
      value="${minimum}"
    fi
    if (( value > SELECTED_NUM_ADAPTERS )); then
      value="${SELECTED_NUM_ADAPTERS}"
    fi
  else
    if ! [[ "${requested}" =~ ^[0-9]+$ ]]; then
      echo "[ERROR] VLLM_MAX_CPU_LORAS must be a positive integer or auto, got ${requested}" >&2
      return 1
    fi
    value="${requested}"
    if (( value < minimum )); then
      echo "[ERROR] VLLM_MAX_CPU_LORAS=${value} is smaller than required minimum ${minimum} (max(max_loras,max_num_seqs))" >&2
      return 1
    fi
    if (( value > SELECTED_NUM_ADAPTERS )); then
      value="${SELECTED_NUM_ADAPTERS}"
    fi
  fi
  printf '%s\n' "${value}"
}

MAX_CPU_LORAS="$(resolve_max_cpu_loras)"

echo "[2/5] Building vLLM launch spec from shared subset"
PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${VLLM_PYTHON}" - "${SHARED_ADAPTER_SUBSET_PATH}" "${MODEL_PATH}" "${LAUNCH_SPEC_PATH}" "${LORA_MODULES_JSON}" "${LORA_MODULES_TXT}" "${TP_EFFECTIVE}" "${DP_REPLICAS}" "${VLLM_GPU_IDS}" "${VLLM_HOST}" "${VLLM_PORT}" "${VLLM_PORT_STRIDE}" "${GPU_MEMORY_UTILIZATION}" "${DTYPE}" "${MAX_LORAS}" "${MAX_CPU_LORAS}" "${MAX_LORA_RANK}" "${MAX_NUM_SEQS}" "${MAX_NUM_BATCHED_TOKENS}" "${ENABLE_CHUNKED_PREFILL}" "${ENABLE_PREFIX_CACHING}" "${ENFORCE_EAGER}" "${VLLM_USE_V1_EFFECTIVE}" "${VLLM_ATTENTION_BACKEND_EFFECTIVE}" "${VLLM_USE_FLASHINFER_SAMPLER_EFFECTIVE}" "${PROMPT_GUARD_MAX_MODEL_LEN}" "${VLLM_LORA_REGISTRATION_MODE}" "${VLLM_DYNAMIC_LORA_ROUTING}" "${VLLM_DYNAMIC_LORA_HOT_PAIR_THRESHOLD}" "${VLLM_DYNAMIC_LORA_HOT_PAIR_MAX_ADAPTERS}" "${VLLM_DISABLE_FRONTEND_MULTIPROCESSING}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

subset_path = Path(sys.argv[1]).resolve()
model_path = sys.argv[2]
launch_spec_path = Path(sys.argv[3]).resolve()
lora_modules_json = Path(sys.argv[4]).resolve()
lora_modules_txt = Path(sys.argv[5]).resolve()
tp = int(sys.argv[6])
dp = int(sys.argv[7])
gpu_ids = [item.strip() for item in sys.argv[8].split(",") if item.strip()]
host = sys.argv[9]
base_port = int(sys.argv[10])
port_stride = int(sys.argv[11])
gpu_memory_utilization = float(sys.argv[12])
dtype = sys.argv[13]
max_loras = int(sys.argv[14])
max_cpu_loras = int(sys.argv[15])
max_lora_rank = int(sys.argv[16])
max_num_seqs = int(sys.argv[17])
max_num_batched_tokens = int(sys.argv[18])
enable_chunked_prefill = bool(int(sys.argv[19]))
enable_prefix_caching = bool(int(sys.argv[20]))
enforce_eager = bool(int(sys.argv[21]))
vllm_use_v1 = bool(int(sys.argv[22]))
attention_backend = sys.argv[23]
use_flashinfer_sampler = bool(int(sys.argv[24]))
max_model_len = int(sys.argv[25])
lora_registration_mode = sys.argv[26]
dynamic_lora_routing = sys.argv[27]
dynamic_lora_hot_pair_threshold = int(sys.argv[28])
dynamic_lora_hot_pair_max_adapters = int(sys.argv[29])
disable_frontend_multiprocessing = bool(int(sys.argv[30]))
if port_stride < 4:
    raise SystemExit(f"VLLM_PORT_STRIDE must leave room for vLLM internal ports, got {port_stride}")

payload = json.loads(subset_path.read_text(encoding="utf-8"))
remote_dir = Path(payload["remote_dir"]).resolve()
modules = []
for item in payload.get("adapters", []):
    adapter_id = str(item["id"])
    adapter_path = remote_dir / adapter_id
    if not adapter_path.exists():
        raise SystemExit(f"missing adapter path for vLLM launch: {adapter_path}")
    modules.append(f"{adapter_id}={adapter_path}")

required = dp * tp
if len(gpu_ids) < required:
    raise SystemExit(f"topology requires {required} GPU ids, got {len(gpu_ids)}")

replica_ports = []
replica_gpu_masks = []
base_urls = []
for replica_idx in range(dp):
    gpu_slice = gpu_ids[replica_idx * tp : (replica_idx + 1) * tp]
    port = base_port + replica_idx * port_stride
    replica_ports.append(port)
    replica_gpu_masks.append(",".join(gpu_slice))
    base_urls.append(f"http://{host}:{port}")

launch = {
    "model": model_path,
    "served_model_name": "base",
    "trust_remote_code": True,
    "host": host,
    "base_port": base_port,
    "port_stride": port_stride,
    "tensor_parallel_size": tp,
    "data_parallel_replicas": dp,
    "num_gpus": required,
    "gpu_per_request": tp,
    "parallelism_topology": (
        f"data_parallel_dp{dp}_tp{tp}" if dp > 1 else (f"model_parallel_tp{tp}" if tp > 1 else "single_gpu")
    ),
    "gpu_memory_utilization": gpu_memory_utilization,
    "dtype": dtype,
    "enable_lora": True,
    "max_loras": max_loras,
    "max_cpu_loras": max_cpu_loras,
    "max_lora_rank": max_lora_rank,
    "max_num_seqs": max_num_seqs,
    "max_num_batched_tokens": max_num_batched_tokens,
    "max_model_len": max_model_len,
    "enable_chunked_prefill": enable_chunked_prefill,
    "enable_prefix_caching": enable_prefix_caching,
    "enforce_eager": enforce_eager,
    "vllm_use_v1": vllm_use_v1,
    "vllm_attention_backend": attention_backend,
    "vllm_use_flashinfer_sampler": use_flashinfer_sampler,
    "lora_modules_count": len(modules),
    "lora_registration_mode": lora_registration_mode,
    "dynamic_lora_routing": dynamic_lora_routing,
    "dynamic_lora_hot_pair_threshold": dynamic_lora_hot_pair_threshold,
    "dynamic_lora_hot_pair_max_adapters": dynamic_lora_hot_pair_max_adapters,
    "disable_frontend_multiprocessing": disable_frontend_multiprocessing,
    "base_urls": base_urls,
    "replica_ports": replica_ports,
    "replica_gpu_masks": replica_gpu_masks,
    "static_startup_sec": 0.0,
}
launch_spec_path.write_text(yaml.safe_dump(launch, sort_keys=False), encoding="utf-8")
lora_modules_json.write_text(json.dumps(modules, ensure_ascii=False, indent=2), encoding="utf-8")
lora_modules_txt.write_text("\n".join(modules) + "\n", encoding="utf-8")
print(launch_spec_path)
PY

echo "      cost_model(base/in/out)=${BASE_COST_USD}/${INPUT_TOKEN_COST_USD}/${OUTPUT_TOKEN_COST_USD}"
echo "      ttft_slo_ms=${TTFT_SLO_MS}"
echo "      model=${MODEL_PATH}"
echo "      topology=${VLLM_TOPOLOGY_LABEL} gpu_ids=${VLLM_GPU_IDS}"
echo "      max_loras=${MAX_LORAS} max_cpu_loras=${MAX_CPU_LORAS}/${SELECTED_NUM_ADAPTERS} max_lora_rank=${MAX_LORA_RANK}"
echo "      max_num_seqs=${MAX_NUM_SEQS} max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
echo "      lora_registration_mode=${VLLM_LORA_REGISTRATION_MODE} dynamic_lora_routing=${VLLM_DYNAMIC_LORA_ROUTING} hot_pair_threshold=${VLLM_DYNAMIC_LORA_HOT_PAIR_THRESHOLD} hot_pair_max=${VLLM_DYNAMIC_LORA_HOT_PAIR_MAX_ADAPTERS}"
echo "      disable_frontend_multiprocessing=${VLLM_DISABLE_FRONTEND_MULTIPROCESSING}"
echo "      host_memory_guard=${VLLM_HOST_MIN_MEM_GB}GiB watch_interval=${VLLM_MEM_WATCH_INTERVAL_S}s"
echo "      smoke_only=${VLLM_SMOKE_ONLY} smoke_max_tokens=${VLLM_SMOKE_MAX_TOKENS}"
echo "      max_replay_requests=${VLLM_MAX_REPLAY_REQUESTS} expected_replay_total=${VLLM_EXPECTED_REPLAY_TOTAL} abort_after_failures=${VLLM_ABORT_AFTER_FAILURES} abort_failures_min_done=${VLLM_ABORT_FAILURES_MIN_DONE}"
echo "      min_output_tokens=${VLLM_MIN_OUTPUT_TOKENS} include_stream_usage=${VLLM_INCLUDE_STREAM_USAGE} empty_success_retries=${VLLM_EMPTY_SUCCESS_RETRIES}"
echo "      launch_spec=${LAUNCH_SPEC_PATH}"
echo "      lora_modules=${LORA_MODULES_TXT}"
echo "      replay=${REPLAY_PATH}"
echo "      summary=${SUMMARY_PATH}"

mapfile -t VLLM_LORA_MODULES < "${LORA_MODULES_TXT}"
IFS=',' read -r -a GPU_ID_ARRAY <<< "${VLLM_GPU_IDS}"
REQUIRED_GPU_COUNT=$(( DP_REPLICAS * TP_EFFECTIVE ))
if (( ${#GPU_ID_ARRAY[@]} < REQUIRED_GPU_COUNT )); then
  echo "[ERROR] vLLM DP/TP topology requires ${REQUIRED_GPU_COUNT} GPU ids, got ${#GPU_ID_ARRAY[@]} from VLLM_GPU_IDS=${VLLM_GPU_IDS}" >&2
  exit 1
fi
ensure_gpu_set_idle "${VLLM_GPU_IDS}" "vLLM"
require_host_memory "before vLLM launch"

write_fleet_spec() {
  local startup_sec="$1"
  shift
  PYTHONNOUSERSITE=1 "${VLLM_PYTHON}" - "${LAUNCH_SPEC_PATH}" "${FLEET_SPEC_PATH}" "${startup_sec}" "$@" <<'PY'
import sys
from pathlib import Path

import yaml

launch = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
startup_sec = float(sys.argv[3])
base_urls = [item for item in sys.argv[4].split(",") if item]
ports = [int(item) for item in sys.argv[5].split() if item]
gpu_masks = [item for item in sys.argv[6].split() if item]
fleet = dict(launch)
fleet.update({
    "served_model_name": str(launch.get("served_model_name") or "base"),
    "static_startup_sec": startup_sec,
    "base_urls": base_urls,
    "replica_ports": ports,
    "replica_gpu_masks": gpu_masks,
})
Path(sys.argv[2]).write_text(yaml.safe_dump(fleet, sort_keys=False), encoding="utf-8")
PY
}

VLLM_BASE_URLS=()
VLLM_REPLICA_PORTS=()
VLLM_REPLICA_GPU_MASKS=()
for replica_idx in $(seq 0 $((DP_REPLICAS - 1))); do
  replica_port=$((VLLM_PORT + replica_idx * VLLM_PORT_STRIDE))
  gpu_slice=()
  for local_idx in $(seq 0 $((TP_EFFECTIVE - 1))); do
    gpu_slice+=("${GPU_ID_ARRAY[$((replica_idx * TP_EFFECTIVE + local_idx))]}")
  done
  replica_gpu_mask="$(IFS=,; echo "${gpu_slice[*]}")"
  VLLM_BASE_URLS+=("http://${VLLM_HOST}:${replica_port}")
  VLLM_REPLICA_PORTS+=("${replica_port}")
  VLLM_REPLICA_GPU_MASKS+=("${replica_gpu_mask}")
done
VLLM_BASE_URL_LIST="$(IFS=,; echo "${VLLM_BASE_URLS[*]}")"
write_fleet_spec "0.0" "${VLLM_BASE_URL_LIST}" "${VLLM_REPLICA_PORTS[*]}" "${VLLM_REPLICA_GPU_MASKS[*]}"

if [[ "${VLLM_DRY_RUN}" == "1" ]]; then
  echo "[dry-run] vLLM launch/replay/summary specs generated; skipping server launch."
  exit 0
fi

echo "[3/5] Starting isolated vLLM OpenAI server(s)"
rm -f "${SERVER_LOG_PREFIX}"_r*.log
VLLM_SERVER_PIDS=""
VLLM_STARTUP_SECS=()

for replica_idx in $(seq 0 $((DP_REPLICAS - 1))); do
  replica_port="${VLLM_REPLICA_PORTS[${replica_idx}]}"
  replica_gpu_mask="${VLLM_REPLICA_GPU_MASKS[${replica_idx}]}"
  ensure_port_is_free "${replica_port}" "vLLM API"
  replica_log="${SERVER_LOG_PREFIX}_r${replica_idx}.log"
  launch_epoch="$(PYTHONNOUSERSITE=1 "${VLLM_PYTHON}" -c 'import time; print(f"{time.time():.6f}")')"
  env_args=(
    PYTHONNOUSERSITE=1
    PYTHONUNBUFFERED=1
    CUDA_VISIBLE_DEVICES="${replica_gpu_mask}"
    VLLM_USE_V1="${VLLM_USE_V1_EFFECTIVE}"
    VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER_EFFECTIVE}"
  )
  if [[ -n "${VLLM_ATTENTION_BACKEND_EFFECTIVE}" ]]; then
    env_args+=(VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND_EFFECTIVE}")
  fi
  if [[ "${VLLM_LORA_REGISTRATION_MODE}" == "dynamic" ]]; then
    env_args+=(VLLM_ALLOW_RUNTIME_LORA_UPDATING=True)
  fi
  server_cmd=(
    "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server
    --model "${MODEL_PATH}"
    --served-model-name "base"
    --host "${VLLM_HOST}"
    --port "${replica_port}"
    --trust-remote-code
    --tensor-parallel-size "${TP_EFFECTIVE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --dtype "${DTYPE}"
    --max-model-len "${PROMPT_GUARD_MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --enable-lora
    --max-loras "${MAX_LORAS}"
    --max-cpu-loras "${MAX_CPU_LORAS}"
    --max-lora-rank "${MAX_LORA_RANK}"
    --disable-log-requests
  )
  if [[ "${VLLM_LORA_REGISTRATION_MODE}" == "static" ]]; then
    server_cmd+=(--lora-modules "${VLLM_LORA_MODULES[@]}")
  fi
  if [[ "${VLLM_DISABLE_FRONTEND_MULTIPROCESSING}" == "1" ]]; then
    server_cmd+=(--disable-frontend-multiprocessing)
  fi
  if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
    server_cmd+=(--enable-chunked-prefill)
  else
    server_cmd+=(--no-enable-chunked-prefill)
  fi
  if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
    server_cmd+=(--enable-prefix-caching)
  else
    server_cmd+=(--no-enable-prefix-caching)
  fi
  if [[ "${ENFORCE_EAGER}" == "1" ]]; then
    server_cmd+=(--enforce-eager)
  fi

  setsid env "${env_args[@]}" "${server_cmd[@]}" > "${replica_log}" 2>&1 &
  replica_pid=$!
  VLLM_SERVER_PIDS="${VLLM_SERVER_PIDS} ${replica_pid}"
  echo "      replica=${replica_idx} pid=${replica_pid} port=${replica_port} gpu_mask=${replica_gpu_mask} log=${replica_log}"

  ready=0
  for _ in $(seq 1 360); do
    if curl -s "http://${VLLM_HOST}:${replica_port}/v1/models" >/tmp/vllm_models_${RESULT_TAG}_${replica_idx}.json 2>/dev/null; then
      ready=1
      break
    fi
    if ! kill -0 "${replica_pid}" 2>/dev/null; then
      echo "[ERROR] vLLM replica ${replica_idx} exited before becoming ready. Tail log:" >&2
      tail -n 100 "${replica_log}" >&2 || true
      exit 1
    fi
    sleep 2
  done
  if [[ "${ready}" != "1" ]]; then
    echo "[ERROR] timed out waiting for vLLM replica ${replica_idx} /v1/models readiness. Tail log:" >&2
    tail -n 100 "${replica_log}" >&2 || true
    exit 1
  fi
  ready_epoch="$(PYTHONNOUSERSITE=1 "${VLLM_PYTHON}" -c 'import time; print(f"{time.time():.6f}")')"
  startup_sec="$(
    PYTHONNOUSERSITE=1 "${VLLM_PYTHON}" -c 'import sys; print(f"{max(0.0, float(sys.argv[2]) - float(sys.argv[1])):.6f}")' \
      "${launch_epoch}" "${ready_epoch}"
  )"
  VLLM_STARTUP_SECS+=("${startup_sec}")
  echo "      replica=${replica_idx} startup_sec=${startup_sec}"
  require_host_memory "after vLLM replica ${replica_idx} startup"
done

startup_max_sec="$(
  PYTHONNOUSERSITE=1 "${VLLM_PYTHON}" -c '
import sys
values = [float(v) for v in sys.argv[1:]]
print(f"{(max(values) if values else 0.0):.6f}")
' "${VLLM_STARTUP_SECS[@]}"
)"
VLLM_SERVER_STARTUP_SEC="${startup_max_sec:-0.0}"
write_fleet_spec "${VLLM_SERVER_STARTUP_SEC}" "${VLLM_BASE_URL_LIST}" "${VLLM_REPLICA_PORTS[*]}" "${VLLM_REPLICA_GPU_MASKS[*]}"
echo "      vllm_startup_sec=${VLLM_SERVER_STARTUP_SEC}"
echo "      vllm_base_urls=${VLLM_BASE_URL_LIST}"

if [[ "${VLLM_SMOKE_ONLY}" == "1" ]]; then
  echo "[smoke] Sending one short LoRA request to each vLLM replica"
  PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${VLLM_PYTHON}" - "${VLLM_BASE_URL_LIST}" "${LORA_MODULES_TXT}" "${VLLM_SMOKE_MAX_TOKENS}" "${VLLM_LORA_REGISTRATION_MODE}" <<'PY'
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

base_urls = [item for item in sys.argv[1].split(",") if item]
modules = [
    tuple(line.split("=", 1))
    for line in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
max_tokens = int(sys.argv[3])
registration_mode = sys.argv[4]
if not base_urls:
    raise SystemExit("no vLLM base URLs for smoke test")
if not modules:
    raise SystemExit("no LoRA modules for smoke test")

for idx, base_url in enumerate(base_urls):
    adapter, adapter_path = modules[idx % len(modules)]
    if registration_mode == "dynamic":
        load_payload = {"lora_name": adapter, "lora_path": adapter_path}
        load_req = urllib.request.Request(
            base_url.rstrip("/") + "/v1/load_lora_adapter",
            data=json.dumps(load_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(load_req, timeout=180) as resp:
                if resp.status not in (200, 201, 204):
                    body = resp.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"HTTP {resp.status}: {body[:500]}")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if "already" not in body.lower() or "loaded" not in body.lower():
                raise SystemExit(f"smoke dynamic load failed for {base_url} adapter={adapter}: HTTP {exc.code}: {body[:800]}")
        except Exception as exc:
            raise SystemExit(f"smoke dynamic load failed for {base_url} adapter={adapter}: {type(exc).__name__}: {exc}")
    payload = {
        "model": adapter,
        "prompt": "Hello",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {body[:500]}")
            parsed = json.loads(body)
            choices = parsed.get("choices") or []
            if not choices:
                raise RuntimeError(f"no completion choices: {body[:500]}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"smoke request failed for {base_url} adapter={adapter}: HTTP {exc.code}: {body[:800]}")
    except Exception as exc:
        raise SystemExit(f"smoke request failed for {base_url} adapter={adapter}: {type(exc).__name__}: {exc}")
    print(f"[smoke] ok replica={idx} base_url={base_url} adapter={adapter}")
PY
  require_host_memory "after vLLM smoke"
  echo "[smoke] vLLM startup and short LoRA requests succeeded; skipping formal replay."
  exit 0
fi

echo "[4/5] Replaying shared trace with unified live metrics"
REPLAY_EXTRA_ARGS=()
if [[ "${VLLM_INCLUDE_STREAM_USAGE}" == "1" ]]; then
  REPLAY_EXTRA_ARGS+=(--include-stream-usage)
fi
if [[ "${VLLM_LORA_REGISTRATION_MODE}" == "dynamic" ]]; then
  REPLAY_EXTRA_ARGS+=(
    --dynamic-lora-modules "${LORA_MODULES_TXT}"
    --dynamic-lora-routing "${VLLM_DYNAMIC_LORA_ROUTING}"
    --dynamic-lora-hot-pair-threshold "${VLLM_DYNAMIC_LORA_HOT_PAIR_THRESHOLD}"
    --dynamic-lora-hot-pair-max-adapters "${VLLM_DYNAMIC_LORA_HOT_PAIR_MAX_ADAPTERS}"
  )
fi
set +e
PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${VLLM_PYTHON}" \
  "${ROOT_DIR}/scripts/replay_openai_trace.py" \
  --trace "${SHARED_TRACE_PATH}" \
  --base-url "${VLLM_BASE_URLS[0]}" \
  --base-url-list "${VLLM_BASE_URL_LIST}" \
  --endpoint-path "/v1/completions" \
  --convert-chat-to-prompt \
  --prompt-guard-tokenizer-model "${PROMPT_GUARD_TOKENIZER_MODEL}" \
  --prompt-guard-max-model-len "${PROMPT_GUARD_MAX_MODEL_LEN}" \
  --prompt-guard-max-input-len "${PROMPT_GUARD_MAX_INPUT_LEN}" \
  --prompt-guard-max-output-tokens-cap "${PROMPT_GUARD_MAX_OUTPUT_TOKENS_CAP}" \
  --sleep-scale "${VLLM_SLEEP_SCALE}" \
  --timeout-s "${VLLM_TIMEOUT_S}" \
  --base-cost-usd "${BASE_COST_USD}" \
  --input-token-cost-usd "${INPUT_TOKEN_COST_USD}" \
  --output-token-cost-usd "${OUTPUT_TOKEN_COST_USD}" \
  --ttft-slo-ms "${TTFT_SLO_MS}" \
  --generation-seed "${GENERATION_SEED}" \
  --min-output-tokens "${VLLM_MIN_OUTPUT_TOKENS}" \
  --empty-success-retries "${VLLM_EMPTY_SUCCESS_RETRIES}" \
  --empty-success-retry-delay-s "${VLLM_EMPTY_SUCCESS_RETRY_DELAY_S}" \
  --max-requests "${VLLM_MAX_REPLAY_REQUESTS}" \
  --abort-after-failures "${VLLM_ABORT_AFTER_FAILURES}" \
  --abort-failures-min-done "${VLLM_ABORT_FAILURES_MIN_DONE}" \
  --adapter-source-field "adapter_id" \
  --adapter-target-field "model" \
  --drop-body-field "request_id" \
  --drop-body-field "lora_adapter_name" \
  --label "${RESULT_TAG}" \
  --output "${REPLAY_PATH}" \
  "${REPLAY_EXTRA_ARGS[@]}" &
REPLAY_PID=$!

MONITOR_STATUS=0
monitor_replay_and_servers "${REPLAY_PID}" || MONITOR_STATUS=$?
wait "${REPLAY_PID}"
REPLAY_STATUS=$?
set -e
if [[ "${MONITOR_STATUS}" -ne 0 ]]; then
  echo "[ERROR] vLLM replay monitor failed status=${MONITOR_STATUS}; rejecting this run." >&2
  exit "${MONITOR_STATUS}"
fi
if [[ "${REPLAY_STATUS}" -ne 0 ]]; then
  echo "[ERROR] vLLM replay failed status=${REPLAY_STATUS}; rejecting this run." >&2
  stop_vllm_servers
  exit "${REPLAY_STATUS}"
fi
require_host_memory "after vLLM replay"

PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${VLLM_PYTHON}" \
  "${ROOT_DIR}/scripts/validate_replay_results.py" \
  --system "vLLM" \
  --replay "${REPLAY_PATH}" \
  --expected-total "${VLLM_EXPECTED_REPLAY_TOTAL}"

PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${VLLM_PYTHON}" - "${REPLAY_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
results = list(payload.get("results", []))
total = len(results)
ok = [r for r in results if bool(r.get("success"))]
if total <= 0:
    raise SystemExit(f"[ERROR] vLLM replay wrote no request results: {path}")
if len(ok) != total:
    failed = [r for r in results if not bool(r.get("success"))]
    print(
        f"[ERROR] vLLM replay success mismatch: ok={len(ok)} total={total}. "
        "This is a serving/replay failure, not a valid performance result.",
        file=sys.stderr,
    )
    for item in failed[:8]:
        print(
            "  "
            f"request_id={item.get('request_id')} "
            f"adapter_id={item.get('adapter_id')} "
            f"status={item.get('status_code')} "
            f"error={str(item.get('error') or '')[:500]}",
            file=sys.stderr,
        )
    raise SystemExit(1)
bad_token_source = [
    r for r in ok
    if str(r.get("prompt_token_source") or "") == "trace_expected"
    or str(r.get("completion_token_source") or "") == "trace_expected"
]
if bad_token_source:
    print(
        f"[ERROR] vLLM replay still fell back to trace expected token counts: "
        f"bad={len(bad_token_source)} total_ok={len(ok)}. "
        "This would contaminate TPOT/token-cost diagnostics, so the run is rejected.",
        file=sys.stderr,
    )
    for item in bad_token_source[:8]:
        print(
            "  "
            f"request_id={item.get('request_id')} "
            f"adapter_id={item.get('adapter_id')} "
            f"prompt_source={item.get('prompt_token_source')} "
            f"completion_source={item.get('completion_token_source')}",
            file=sys.stderr,
        )
    raise SystemExit(1)
print(f"[check] vLLM replay success: ok={len(ok)} total={total}")
print(
    "[check] vLLM token sources are observed/local; no trace_expected fallback "
    "entered formal token diagnostics."
)
PY

echo "[5/5] Summarizing replay into the shared paper metric schema"
PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${VLLM_PYTHON}" \
  "${ROOT_DIR}/scripts/summarize_serverlessllm_replay.py" \
  --main-repo "${MAIN_REPO}" \
  --config "${CONFIG_PATH}" \
  --model-profile "${MODEL_PROFILE}" \
  --dataset-profile "${DATASET_PROFILE}" \
  --workload-profile "${WORKLOAD_PROFILE}" \
  --trace "${SHARED_TRACE_PATH}" \
  --adapter-subset "${SHARED_ADAPTER_SUBSET_PATH}" \
  --deploy "${FLEET_SPEC_PATH}" \
  --replay "${REPLAY_PATH}" \
  --scenario-name "vllm_fair" \
  --baseline-type "vllm" \
  --backend-label "vllm_openai_official" \
  --system-name "vLLM" \
  --instance-mode "static_runtime" \
  --routing-policy "fcfs_batching" \
  --static-startup-sec "${VLLM_SERVER_STARTUP_SEC}" \
  --output "${SUMMARY_PATH}"

echo
echo "vLLM fair replay complete."
echo "  Replay : ${REPLAY_PATH}"
echo "  Summary: ${SUMMARY_PATH}"
echo "  Fleet  : ${FLEET_SPEC_PATH}"
echo "  Logs   : ${SERVER_LOG_PREFIX}_r*.log"
