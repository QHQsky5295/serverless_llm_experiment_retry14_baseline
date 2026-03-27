#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$ROOT_DIR"
DEFAULT_CONFIG="$PROJECT_ROOT/configs/experiments.yaml"
PYTHON_BIN="${FAASLORA_PYTHON:-/home/qhq/anaconda3/envs/LLM_vllm0102/bin/python}"
LOG_DIR="${FAASLORA_LOG_DIR:-/tmp/faaslora_validation}"
MODE="${1:-quick100}"
RUNNER_WRAPPER="$PROJECT_ROOT/scripts/run_all_experiments_user_scope.sh"
shift || true

mkdir -p "$LOG_DIR"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

run_precheck() {
  local precheck_log="$1"
  {
    echo "[precheck] $(timestamp)"
    echo "python=$PYTHON_BIN"
    echo "mode=$MODE"
    echo "cwd=$PROJECT_ROOT"
    echo "session_cgroup=$(cat /proc/self/cgroup | tr '\n' ' ')"
    echo
    echo "[nvidia-smi -L]"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi -L || true
    else
      echo "nvidia-smi: not found"
    fi
    echo
    echo "[torch cuda check]"
    "$PYTHON_BIN" - <<'PY'
import json
info = {}
try:
    import torch
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["device_count"] = int(torch.cuda.device_count())
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        info["device_name_0"] = torch.cuda.get_device_name(0)
except Exception as exc:
    info["error"] = str(exc)
print(json.dumps(info, ensure_ascii=False))
PY
    echo
    echo "[session]"
    loginctl session-status "${XDG_SESSION_ID:-}" 2>/dev/null | sed -n '1,12p' || true
  } >"$precheck_log" 2>&1
}

build_temp_config() {
  local src_cfg="$1"
  local dst_cfg="$2"
  local backend="${FAASLORA_BACKEND:-}"
  local instance_mode="${FAASLORA_INSTANCE_MODE:-}"
  local max_instances="${FAASLORA_MAX_INSTANCES:-}"
  local min_instances="${FAASLORA_MIN_INSTANCES:-}"
  local runtime_cap="${FAASLORA_RUNTIME_CONCURRENCY_CAP:-}"
  local total_requests="${FAASLORA_TOTAL_REQUESTS:-}"
  local concurrency="${FAASLORA_CONCURRENCY:-}"
  local quick_total_requests="${FAASLORA_QUICK_TOTAL_REQUESTS:-}"
  local quick_concurrency="${FAASLORA_QUICK_CONCURRENCY:-}"
  local time_scale_factor="${FAASLORA_TIME_SCALE_FACTOR:-}"
  local scale_decision_interval="${FAASLORA_SCALE_DECISION_INTERVAL:-}"
  local scale_up_threshold_rps="${FAASLORA_SCALE_UP_THRESHOLD_RPS:-}"
  local effective_capacity_admission="${FAASLORA_EFFECTIVE_CAPACITY_ADMISSION:-}"
  local max_model_len="${FAASLORA_MAX_MODEL_LEN:-}"
  local max_num_seqs="${FAASLORA_MAX_NUM_SEQS:-}"
  local max_loras="${FAASLORA_MAX_LORAS:-}"
  local max_num_batched_tokens="${FAASLORA_MAX_NUM_BATCHED_TOKENS:-}"

  "$PYTHON_BIN" - <<PY
from pathlib import Path
import yaml

src = Path(${src_cfg@Q})
dst = Path(${dst_cfg@Q})
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))

backend = ${backend@Q}
instance_mode = ${instance_mode@Q}
max_instances = ${max_instances@Q}
min_instances = ${min_instances@Q}
runtime_cap = ${runtime_cap@Q}
total_requests = ${total_requests@Q}
concurrency = ${concurrency@Q}
quick_total_requests = ${quick_total_requests@Q}
quick_concurrency = ${quick_concurrency@Q}
time_scale_factor = ${time_scale_factor@Q}
scale_decision_interval = ${scale_decision_interval@Q}
scale_up_threshold_rps = ${scale_up_threshold_rps@Q}
effective_capacity_admission = ${effective_capacity_admission@Q}
max_model_len = ${max_model_len@Q}
max_num_seqs = ${max_num_seqs@Q}
max_loras = ${max_loras@Q}
max_num_batched_tokens = ${max_num_batched_tokens@Q}

if backend:
    cfg.setdefault("model", {})["backend"] = backend
if runtime_cap:
    cfg.setdefault("model", {})["runtime_concurrency_cap"] = int(runtime_cap)
if max_model_len:
    cfg.setdefault("model", {})["max_model_len"] = int(max_model_len)
if max_num_seqs:
    cfg.setdefault("model", {})["max_num_seqs"] = int(max_num_seqs)
if max_loras:
    cfg.setdefault("model", {})["max_loras"] = int(max_loras)
if max_num_batched_tokens:
    cfg.setdefault("model", {})["max_num_batched_tokens"] = int(max_num_batched_tokens)
if instance_mode:
    cfg.setdefault("resource_coordination", {})["instance_mode"] = instance_mode
if max_instances:
    cfg.setdefault("resource_coordination", {})["max_instances"] = int(max_instances)
if min_instances:
    cfg.setdefault("resource_coordination", {})["min_instances"] = int(min_instances)
if scale_decision_interval:
    cfg.setdefault("resource_coordination", {})["scale_decision_interval"] = int(scale_decision_interval)
if scale_up_threshold_rps:
    cfg.setdefault("resource_coordination", {})["scale_up_threshold_rps"] = float(scale_up_threshold_rps)
if effective_capacity_admission:
    cfg.setdefault("resource_coordination", {})["effective_capacity_admission_enabled"] = (
        str(effective_capacity_admission).strip().lower() in {"1", "true", "yes", "on"}
    )
if total_requests:
    cfg.setdefault("workload", {})["total_requests"] = int(total_requests)
if concurrency:
    cfg.setdefault("workload", {})["concurrency"] = int(concurrency)
if quick_total_requests:
    cfg.setdefault("workload", {})["quick_total_requests"] = int(quick_total_requests)
if quick_concurrency:
    cfg.setdefault("workload", {})["quick_concurrency"] = int(quick_concurrency)
if time_scale_factor:
    cfg.setdefault("workload", {})["time_scale_factor"] = float(time_scale_factor)

dst.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
print(dst)
PY
}

MODE_TAG="$MODE"
NUM_ADAPTERS="${FAASLORA_NUM_ADAPTERS:-}"
PRESET_NAME="${FAASLORA_PRESET:-}"
SCENARIO="${FAASLORA_SCENARIO:-faaslora_full}"
QUICK_FLAG=()

case "$MODE" in
  gpu-check)
    PRECHECK_LOG="$LOG_DIR/gpu_check.log"
    run_precheck "$PRECHECK_LOG"
    echo "precheck_log=$PRECHECK_LOG"
    exit 0
    ;;
  quick100)
    NUM_ADAPTERS="${NUM_ADAPTERS:-100}"
    QUICK_FLAG=(--quick)
    ;;
  quick1000)
    NUM_ADAPTERS="${NUM_ADAPTERS:-1000}"
    QUICK_FLAG=(--quick)
    ;;
  full1000)
    NUM_ADAPTERS="${NUM_ADAPTERS:-1000}"
    ;;
  custom)
    if [[ -z "${NUM_ADAPTERS:-}" && -z "${PRESET_NAME:-}" ]]; then
      echo "FAASLORA_NUM_ADAPTERS or FAASLORA_PRESET is required when MODE=custom" >&2
      exit 2
    fi
    if [[ "${FAASLORA_QUICK:-0}" == "1" ]]; then
      QUICK_FLAG=(--quick)
    fi
    ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    echo "Supported modes: gpu-check, quick100, quick1000, full1000, custom" >&2
    exit 2
    ;;
esac

RUN_TAG="${FAASLORA_LOG_TAG:-${MODE_TAG}}"
PRECHECK_LOG="$LOG_DIR/${RUN_TAG}.precheck.log"
RUN_LOG="$LOG_DIR/${RUN_TAG}.run.log"
EXIT_FILE="$LOG_DIR/${RUN_TAG}.exit"
TMP_CONFIG="$LOG_DIR/${RUN_TAG}.yaml"

rm -f "$PRECHECK_LOG" "$RUN_LOG" "$EXIT_FILE" "$TMP_CONFIG"

run_precheck "$PRECHECK_LOG"
build_temp_config "$DEFAULT_CONFIG" "$TMP_CONFIG" >/dev/null

CMD=(
  "$RUNNER_WRAPPER"
  --config "$TMP_CONFIG"
  --scenario "$SCENARIO"
)
if [[ -n "${PRESET_NAME:-}" ]]; then
  CMD+=(--preset "$PRESET_NAME")
fi
if [[ -n "${NUM_ADAPTERS:-}" ]]; then
  CMD+=(--num-adapters "$NUM_ADAPTERS")
fi
if [[ -n "${FAASLORA_BACKEND:-}" ]]; then
  CMD+=(--backend "$FAASLORA_BACKEND")
fi
CMD+=("${QUICK_FLAG[@]}")
CMD+=("$@")

{
  echo "[command] $(timestamp)"
  printf ' %q' "${CMD[@]}"
  echo
  echo
} >"$RUN_LOG"

set +e
FAASLORA_PYTHON="$PYTHON_BIN" "${CMD[@]}" >>"$RUN_LOG" 2>&1
rc=$?
set -e

printf '%s\n' "$rc" >"$EXIT_FILE"

echo "precheck_log=$PRECHECK_LOG"
echo "run_log=$RUN_LOG"
echo "exit_file=$EXIT_FILE"
echo "temp_config=$TMP_CONFIG"
echo "exit_code=$rc"
