#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${LLUMNIX_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
RELAY_ROOT="${RELAY_ROOT:-/home/qhq/relayserve_serverless_llm}"
PROJECT_ROOT="${LLUMNIX_PROJECT_ROOT:-${ROOT_DIR}/workspaces/llumnix_official_20260615}"
ENV_DIR="${LLUMNIX_ENV_DIR:-/home/qhq/anaconda3/envs/llumnix_official_20260615}"
MODEL_KEY="${1:?usage: $0 3b|7b [max_requests] [run_tag]}"
MAX_REQUESTS="${2:-0}"
RUN_TAG="${3:-$(date -u +%Y%m%dT%H%M%SZ)_llumnix_${MODEL_KEY}_r${MAX_REQUESTS}}"
TRACE_ROLE="${LLUMNIX_TRACE_ROLE:-formal4000}"
PORT="${LLUMNIX_PORT:-8010}"
INITIAL_INSTANCES="${LLUMNIX_INITIAL_INSTANCES:-4}"
GPU_MEMORY_UTILIZATION="${LLUMNIX_GPU_MEMORY_UTILIZATION:-0.84}"
MIGRATION_BUFFER_BLOCKS="${LLUMNIX_MIGRATION_BUFFER_BLOCKS:-128}"
MAX_NUM_SEQS="${LLUMNIX_MAX_NUM_SEQS:-64}"
MAX_MODEL_LEN="${LLUMNIX_MAX_MODEL_LEN:-3072}"
ENFORCE_EAGER="${LLUMNIX_ENFORCE_EAGER:-1}"
ENABLE_ROUTINE_MIGRATION="${LLUMNIX_ENABLE_ROUTINE_MIGRATION:-1}"
SERVICE_READY_TIMEOUT_S="${LLUMNIX_SERVICE_READY_TIMEOUT_S:-900}"
SERVICE_STABILIZATION_S="${LLUMNIX_SERVICE_STABILIZATION_S:-10}"
FULL_PATH_PROBE_TIMEOUT_S="${LLUMNIX_FULL_PATH_PROBE_TIMEOUT_S:-15}"
FULL_PATH_PROBE_ATTEMPTS="${LLUMNIX_FULL_PATH_PROBE_ATTEMPTS:-3}"
REQUEST_TIMEOUT_S="${LLUMNIX_REQUEST_TIMEOUT_S:-1800}"
INIT_INSTANCES_TIMEOUT_S="${LLUMNIX_INIT_INSTANCES_TIMEOUT:-900}"
INIT_WORKER_RPC_TIMEOUT_S="${LLUMNIX_INIT_WORKER_RPC_TIMEOUT:-300}"
SCALE_UP_RPC_TIMEOUT_S="${LLUMNIX_SCALE_UP_RPC_TIMEOUT:-300}"
INSTANCE_READY_TIMEOUT_S="${LLUMNIX_INSTANCE_READY_TIMEOUT:-600}"
PLACEMENT_GROUP_TIMEOUT_S="${LLUMNIX_WAIT_PLACEMENT_GROUP_TIMEOUT:-60}"
UTILITY_CALL_TIMEOUT_S="${LLUMNIX_UTILITY_CALL_TIMEOUT:-300}"
MIN_AVAILABLE_MEMORY_GB="${LLUMNIX_MIN_AVAILABLE_MEMORY_GB:-32}"
MAX_GPU_TEMPERATURE_C="${LLUMNIX_MAX_GPU_TEMPERATURE_C:-88}"
RESOURCE_SAMPLE_INTERVAL_S="${LLUMNIX_RESOURCE_SAMPLE_INTERVAL_S:-2}"
GPU_COST_PER_SECOND_USD="${LLUMNIX_GPU_COST_PER_SECOND_USD:-0.0008}"
ALLOW_FAILED_REQUESTS="${LLUMNIX_ALLOW_FAILED_REQUESTS:-0}"

case "${MODEL_KEY}" in
  3b)
    MODEL_PATH="/home/qhq/serverless_llm_experiment_retry14_baseline/models/LLM-Research--Llama-3.2-3B-Instruct"
    TRACE_PATH="${RELAY_ROOT}/data_imports/traces/multi_turn_continuation/llama32_3b/llama32_3b_burstgptv2_chronological_sharegpt_multi_turn_formal4000_v1_trace.json"
    TTFT_SLO_MS="180"
    TPOT_SLO_MS="14"
    ;;
  7b)
    MODEL_PATH="/home/qhq/serverless_llm_experiment/models/meta-llama--Llama-2-7b-hf"
    TRACE_PATH="${RELAY_ROOT}/data_imports/traces/multi_turn_continuation/llama2_7b/llama2_7b_burstgptv2_chronological_sharegpt_multi_turn_formal4000_v1_trace.json"
    TTFT_SLO_MS="440"
    TPOT_SLO_MS="32"
    ;;
  *)
    echo "model must be 3b or 7b" >&2
    exit 2
    ;;
esac

MODEL_PATH="${LLUMNIX_MODEL_PATH_OVERRIDE:-${MODEL_PATH}}"
TRACE_PATH="${LLUMNIX_TRACE_PATH_OVERRIDE:-${TRACE_PATH}}"
RUN_DIR="${LLUMNIX_RUN_DIR:-${ROOT_DIR}/results/relayserve_continuation/llumnix/${RUN_TAG}}"
RAW_PATH="${RUN_DIR}/raw_records.json"
SUMMARY_PATH="${RUN_DIR}/source_summary.json"
MANIFEST_PATH="${RUN_DIR}/manifest.json"
LOG_DIR="${RUN_DIR}/logs"
SNAPSHOT_DIR="${RUN_DIR}/frozen_config"
RESOURCE_GUARD_PATH="${LOG_DIR}/resource_guard.csv"
SERVICE_HEALTH_PATH="${LOG_DIR}/service_health.json"
FULL_PATH_PROBE_PATH="${LOG_DIR}/full_path_probe.json"
SERVICE_PID=""
RESOURCE_GUARD_PID=""
STARTUP_SEC="0"

if [[ -e "${RUN_DIR}" ]]; then
  echo "refusing to overwrite existing Llumnix run directory: ${RUN_DIR}" >&2
  exit 3
fi
for required in \
  "${TRACE_PATH}" \
  "${MODEL_PATH}/config.json" \
  "${PROJECT_ROOT}/configs/vllm.yml" \
  "${ENV_DIR}/bin/python"
do
  if [[ ! -e "${required}" ]]; then
    echo "missing required Llumnix input: ${required}" >&2
    exit 4
  fi
done

mkdir -p "${LOG_DIR}" "${SNAPSHOT_DIR}"

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ -n "${RESOURCE_GUARD_PID}" ]]; then
    kill "${RESOURCE_GUARD_PID}" 2>/dev/null || true
    wait "${RESOURCE_GUARD_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SERVICE_PID}" ]]; then
    kill -TERM -- "-${SERVICE_PID}" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-${SERVICE_PID}" 2>/dev/null || true
    wait "${SERVICE_PID}" 2>/dev/null || true
  fi
  "${ENV_DIR}/bin/ray" stop --force >"${LOG_DIR}/ray_stop.log" 2>&1 || true
  pkill -f "llumnix.entrypoints.vllm.api_server.*--port ${PORT}" \
    >>"${LOG_DIR}/ray_stop.log" 2>&1 || true
  printf '%s\n' "${exit_code}" >"${RUN_DIR}/.exit"
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

start_resource_guard() {
  local runner_pid="$$"
  (
    echo "timestamp_epoch,mem_available_kb,mem_total_kb,swap_free_kb,max_gpu_temp_c"
    while true; do
      read -r available_kb total_kb swap_free_kb < <(
        awk '
          /MemTotal:/ {total=$2}
          /MemAvailable:/ {available=$2}
          /SwapFree:/ {swapfree=$2}
          END {print available, total, swapfree}
        ' /proc/meminfo
      )
      max_gpu_temp="$(
        nvidia-smi --query-gpu=temperature.gpu \
          --format=csv,noheader,nounits | sort -nr | head -n1
      )"
      printf '%s,%s,%s,%s,%s\n' \
        "$(date +%s)" "${available_kb}" "${total_kb}" \
        "${swap_free_kb}" "${max_gpu_temp}"
      if (( available_kb < MIN_AVAILABLE_MEMORY_GB * 1024 * 1024 )); then
        echo \
          "memory_guard_breach: available_kb=${available_kb}, " \
          "reserve_gb=${MIN_AVAILABLE_MEMORY_GB}" >&2
        kill -TERM "${runner_pid}" 2>/dev/null || true
        exit 90
      fi
      if (( max_gpu_temp >= MAX_GPU_TEMPERATURE_C )); then
        echo \
          "temperature_guard_breach: max_gpu_temp_c=${max_gpu_temp}, " \
          "limit_c=${MAX_GPU_TEMPERATURE_C}" >&2
        kill -TERM "${runner_pid}" 2>/dev/null || true
        exit 91
      fi
      sleep "${RESOURCE_SAMPLE_INTERVAL_S}"
    done
  ) >"${RESOURCE_GUARD_PATH}" 2>"${LOG_DIR}/resource_guard.err" &
  RESOURCE_GUARD_PID=$!
}

echo "Verify reproducible Llumnix host compatibility patch"
LLUMNIX_PROJECT_ROOT="${PROJECT_ROOT}" \
  bash "${ROOT_DIR}/scripts/apply_llumnix_relayserve_patch.sh"

echo "[0/7] Verify isolated official Llumnix environment"
PYTHONNOUSERSITE=1 "${ENV_DIR}/bin/python" - <<'PY'
import llumnix
import ray
import vllm

print("llumnix", getattr(llumnix, "__version__", "source-checkout"))
print("ray", ray.__version__)
print("vllm", vllm.__version__)
PY

echo "[1/7] Freeze provenance and launch parameters"
cp "${PROJECT_ROOT}/configs/vllm.yml" "${SNAPSHOT_DIR}/official_vllm.yml"
git -C "${PROJECT_ROOT}" rev-parse HEAD >"${SNAPSHOT_DIR}/llumnix_git_commit.txt"
git -C "${PROJECT_ROOT}" status --short >"${SNAPSHOT_DIR}/llumnix_git_status.txt"
"${ENV_DIR}/bin/python" -m pip freeze >"${SNAPSHOT_DIR}/pip_freeze.txt"
nvidia-smi -q >"${SNAPSHOT_DIR}/nvidia_smi_q.txt"

SERVICE_ARGS=(
  -m llumnix.entrypoints.vllm.api_server
  --config-file "${PROJECT_ROOT}/configs/vllm.yml"
  --host 127.0.0.1
  --port "${PORT}"
  --initial-instances "${INITIAL_INSTANCES}"
  --launch-ray-cluster
  --model "${MODEL_PATH}"
  --worker-use-ray
  --migration-backend rayrpc
  --migration-buffer-blocks "${MIGRATION_BUFFER_BLOCKS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
)
if [[ "${ENABLE_ROUTINE_MIGRATION}" == "1" ]]; then
  SERVICE_ARGS+=(--enable-routine-migration)
fi
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  SERVICE_ARGS+=(--enforce-eager)
fi
printf '%q ' "${ENV_DIR}/bin/python" "${SERVICE_ARGS[@]}" \
  >"${SNAPSHOT_DIR}/service_command.sh"
printf '\n' >>"${SNAPSHOT_DIR}/service_command.sh"
cat >"${SNAPSHOT_DIR}/service_environment.txt" <<EOF
PYTHONNOUSERSITE=1
PATH_PREFIX=${ENV_DIR}/bin
HEAD_NODE=1
HEAD_NODE_IP=127.0.0.1
LLUMNIX_INIT_INSTANCES_TIMEOUT=${INIT_INSTANCES_TIMEOUT_S}
LLUMNIX_INIT_WORKER_RPC_TIMEOUT=${INIT_WORKER_RPC_TIMEOUT_S}
LLUMNIX_SCALE_UP_RPC_TIMEOUT=${SCALE_UP_RPC_TIMEOUT_S}
LLUMNIX_INSTANCE_READY_TIMEOUT=${INSTANCE_READY_TIMEOUT_S}
LLUMNIX_WAIT_PLACEMENT_GROUP_TIMEOUT=${PLACEMENT_GROUP_TIMEOUT_S}
LLUMNIX_UTILITY_CALL_TIMEOUT=${UTILITY_CALL_TIMEOUT_S}
LLUMNIX_SERVICE_STABILIZATION_S=${SERVICE_STABILIZATION_S}
LLUMNIX_FULL_PATH_PROBE_TIMEOUT_S=${FULL_PATH_PROBE_TIMEOUT_S}
LLUMNIX_FULL_PATH_PROBE_ATTEMPTS=${FULL_PATH_PROBE_ATTEMPTS}
EOF

"${ENV_DIR}/bin/python" - \
  "${MANIFEST_PATH}" "${TRACE_PATH}" "${MODEL_PATH}" "${PROJECT_ROOT}" \
  "${ROOT_DIR}" "${RELAY_ROOT}" "${RUN_TAG}" "${MODEL_KEY}" \
  "${MAX_REQUESTS}" "${TRACE_ROLE}" "${PORT}" "${INITIAL_INSTANCES}" \
  "${GPU_MEMORY_UTILIZATION}" "${MIGRATION_BUFFER_BLOCKS}" \
  "${MAX_NUM_SEQS}" "${MAX_MODEL_LEN}" "${ENFORCE_EAGER}" \
  "${ENABLE_ROUTINE_MIGRATION}" "${MIN_AVAILABLE_MEMORY_GB}" \
  "${MAX_GPU_TEMPERATURE_C}" "${INIT_INSTANCES_TIMEOUT_S}" \
  "${INIT_WORKER_RPC_TIMEOUT_S}" "${SCALE_UP_RPC_TIMEOUT_S}" \
  "${INSTANCE_READY_TIMEOUT_S}" "${PLACEMENT_GROUP_TIMEOUT_S}" \
  "${UTILITY_CALL_TIMEOUT_S}" \
  "${SERVICE_STABILIZATION_S}" "${FULL_PATH_PROBE_TIMEOUT_S}" \
  "${FULL_PATH_PROBE_ATTEMPTS}" "${SNAPSHOT_DIR}" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    manifest_path, trace_path, model_path, project_root, root, relay_root,
    run_tag, model_key, max_requests, trace_role, port, initial_instances,
    gpu_memory_utilization, migration_buffer_blocks, max_num_seqs,
    max_model_len, enforce_eager, enable_routine_migration,
    min_available_memory_gb, max_gpu_temperature_c,
    init_instances_timeout_s, init_worker_rpc_timeout_s,
    scale_up_rpc_timeout_s, instance_ready_timeout_s,
    placement_group_timeout_s, utility_call_timeout_s, service_stabilization_s,
    full_path_probe_timeout_s, full_path_probe_attempts, snapshot_dir,
) = sys.argv[1:]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git_head(path):
    return subprocess.check_output(
        ["git", "-C", path, "rev-parse", "HEAD"], text=True
    ).strip()

root_path = Path(root)
source_paths = [
    root_path / "patches/llumnix_relayserve_compat.patch",
    root_path / "scripts/apply_llumnix_relayserve_patch.sh",
    root_path / "scripts/replay_llumnix_trace.py",
    root_path / "scripts/probe_llumnix_service.py",
    root_path / "scripts/run_llumnix_relayserve_continuation.sh",
    root_path / "scripts/summarize_llumnix_replay.py",
    root_path / "scripts/validate_llumnix_service_health.py",
]
snapshot_path = Path(snapshot_dir)
payload = {
    "schema": "relayserve_external_llumnix_run_v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_tag": run_tag,
    "system": "Llumnix",
    "model_key": model_key,
    "max_requests": int(max_requests),
    "trace_path": trace_path,
    "trace_sha256": sha(trace_path),
    "trace_role": trace_role,
    "model_path": model_path,
    "model_config_sha256": sha(Path(model_path) / "config.json"),
    "llumnix_repo": project_root,
    "llumnix_git_commit": git_head(project_root),
    "baseline_harness_git_commit": git_head(root),
    "relayserve_git_commit": git_head(relay_root),
    "harness_source_sha256": {
        str(path.relative_to(root_path)): sha(path) for path in source_paths
    },
    "runtime_env": "/home/qhq/anaconda3/envs/llumnix_official_20260615",
    "launch_profile": {
        "port": int(port),
        "initial_instances": int(initial_instances),
        "dispatch_policy": "load",
        "migration_backend": "rayrpc",
        "migration_buffer_blocks": int(migration_buffer_blocks),
        "enable_routine_migration": enable_routine_migration == "1",
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "max_num_seqs": int(max_num_seqs),
        "max_model_len": int(max_model_len),
        "enforce_eager": enforce_eager == "1",
        "init_instances_timeout_s": float(init_instances_timeout_s),
        "init_worker_rpc_timeout_s": float(init_worker_rpc_timeout_s),
        "scale_up_rpc_timeout_s": float(scale_up_rpc_timeout_s),
        "instance_ready_timeout_s": float(instance_ready_timeout_s),
        "placement_group_timeout_s": float(placement_group_timeout_s),
        "utility_call_timeout_s": float(utility_call_timeout_s),
        "service_stabilization_s": float(service_stabilization_s),
        "full_path_probe_timeout_s": float(full_path_probe_timeout_s),
        "full_path_probe_attempts": int(full_path_probe_attempts),
    },
    "resource_guard": {
        "min_available_memory_gb": float(min_available_memory_gb),
        "max_gpu_temperature_c": int(max_gpu_temperature_c),
    },
    "comparison_contract": {
        "workload": f"RelayServe frozen continuation trace ({trace_role}), rate=1.00x",
        "slo_profile": "paper_nominal",
        "gpu_budget": 4,
        "base_model_only": True,
        "lora_enabled": False,
        "system_mode": "official Llumnix load dispatch and request migration",
        "cost_model": "static four-GPU deployment lifecycle",
    },
    "frozen_config_dir": snapshot_dir,
    "frozen_config_sha256": {
        path.name: sha(path)
        for path in sorted(snapshot_path.iterdir())
        if path.is_file()
    },
}
Path(manifest_path).write_text(json.dumps(payload, indent=2) + "\n")
PY

start_resource_guard

echo "[2/7] Start official Llumnix four-instance service"
START_MONOTONIC="$("${ENV_DIR}/bin/python" -c 'import time; print(time.monotonic())')"
setsid env \
  PYTHONNOUSERSITE=1 \
  PATH="${ENV_DIR}/bin:${PATH}" \
  LLUMNIX_INIT_INSTANCES_TIMEOUT="${INIT_INSTANCES_TIMEOUT_S}" \
  LLUMNIX_INIT_WORKER_RPC_TIMEOUT="${INIT_WORKER_RPC_TIMEOUT_S}" \
  LLUMNIX_SCALE_UP_RPC_TIMEOUT="${SCALE_UP_RPC_TIMEOUT_S}" \
  LLUMNIX_INSTANCE_READY_TIMEOUT="${INSTANCE_READY_TIMEOUT_S}" \
  LLUMNIX_WAIT_PLACEMENT_GROUP_TIMEOUT="${PLACEMENT_GROUP_TIMEOUT_S}" \
  LLUMNIX_UTILITY_CALL_TIMEOUT="${UTILITY_CALL_TIMEOUT_S}" \
  HEAD_NODE=1 \
  HEAD_NODE_IP=127.0.0.1 \
  "${ENV_DIR}/bin/python" "${SERVICE_ARGS[@]}" \
  >"${LOG_DIR}/service.log" 2>&1 &
SERVICE_PID=$!

deadline=$((SECONDS + SERVICE_READY_TIMEOUT_S))
while true; do
  if ! kill -0 "${SERVICE_PID}" 2>/dev/null; then
    echo "Llumnix service exited before readiness" >&2
    tail -n 200 "${LOG_DIR}/service.log" >&2 || true
    exit 20
  fi
  if curl -fsS "http://127.0.0.1:${PORT}/is_ready" | grep -qi true; then
    set +e
    "${ENV_DIR}/bin/python" \
      "${ROOT_DIR}/scripts/validate_llumnix_service_health.py" \
      --service-log "${LOG_DIR}/service.log" \
      --expected-instances "${INITIAL_INSTANCES}" \
      --phase preflight
    HEALTH_PREFLIGHT_EXIT_CODE=$?
    set -e
    if [[ "${HEALTH_PREFLIGHT_EXIT_CODE}" == "0" ]]; then
      OBSERVED_READY_INSTANCES="${INITIAL_INSTANCES}"
      break
    fi
    if (( HEALTH_PREFLIGHT_EXIT_CODE >= 20 )); then
      echo "Llumnix service failed the four-instance preflight gate" >&2
      exit 24
    fi
  fi
  if (( SECONDS >= deadline )); then
    echo "Llumnix service readiness timed out" >&2
    tail -n 200 "${LOG_DIR}/service.log" >&2 || true
    exit 21
  fi
  sleep 2
done

echo "Wait ${SERVICE_STABILIZATION_S}s for the official output queue loop"
sleep "${SERVICE_STABILIZATION_S}"
echo "Verify full request/output path before measured replay"
set +e
PYTHONNOUSERSITE=1 "${ENV_DIR}/bin/python" \
  "${ROOT_DIR}/scripts/probe_llumnix_service.py" \
  --base-url "http://127.0.0.1:${PORT}" \
  --timeout-s "${FULL_PATH_PROBE_TIMEOUT_S}" \
  --attempts "${FULL_PATH_PROBE_ATTEMPTS}" \
  --output "${FULL_PATH_PROBE_PATH}"
FULL_PATH_PROBE_EXIT_CODE=$?
set -e
if [[ "${FULL_PATH_PROBE_EXIT_CODE}" != "0" ]]; then
  echo "Llumnix full request/output path probe failed" >&2
  exit 25
fi

MEASUREMENT_SERVICE_LOG_OFFSET="$(stat -c '%s' "${LOG_DIR}/service.log")"
STARTUP_SEC="$(
  "${ENV_DIR}/bin/python" -c \
    "import time; print(time.monotonic() - float('${START_MONOTONIC}'))"
)"
"${ENV_DIR}/bin/python" - \
  "${MANIFEST_PATH}" "${STARTUP_SEC}" "${SERVICE_PID}" \
  "${OBSERVED_READY_INSTANCES}" "${FULL_PATH_PROBE_PATH}" \
  "${MEASUREMENT_SERVICE_LOG_OFFSET}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
payload["initial_runtime_startup_sec"] = float(sys.argv[2])
payload["service_pid"] = int(sys.argv[3])
payload["observed_ready_instances"] = int(sys.argv[4])
probe_path = Path(sys.argv[5])
payload["full_path_probe_path"] = str(probe_path)
payload["full_path_probe_sha256"] = hashlib.sha256(
    probe_path.read_bytes()
).hexdigest()
payload["measurement_service_log_offset_bytes"] = int(sys.argv[6])
path.write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "[3/7] Replay frozen continuation workload"
REPLAY_ARGS=()
if [[ "${MAX_REQUESTS}" != "0" ]]; then
  REPLAY_ARGS+=(--max-requests "${MAX_REQUESTS}")
fi
if [[ "${ALLOW_FAILED_REQUESTS}" == "1" ]]; then
  REPLAY_ARGS+=(--allow-failures)
fi
set +e
PYTHONNOUSERSITE=1 "${ENV_DIR}/bin/python" \
  "${ROOT_DIR}/scripts/replay_llumnix_trace.py" \
  --trace "${TRACE_PATH}" \
  --base-url "http://127.0.0.1:${PORT}" \
  --tokenizer "${MODEL_PATH}" \
  --output "${RAW_PATH}" \
  --label "${RUN_TAG}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --ttft-slo-ms "${TTFT_SLO_MS}" \
  --tpot-slo-ms "${TPOT_SLO_MS}" \
  --timeout-s "${REQUEST_TIMEOUT_S}" \
  "${REPLAY_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/replay.log"
REPLAY_EXIT_CODE=${PIPESTATUS[0]}
set -e

set +e
"${ENV_DIR}/bin/python" \
  "${ROOT_DIR}/scripts/validate_llumnix_service_health.py" \
  --service-log "${LOG_DIR}/service.log" \
  --expected-instances "${INITIAL_INSTANCES}" \
  --phase final \
  --runtime-offset "${MEASUREMENT_SERVICE_LOG_OFFSET}" \
  --output "${SERVICE_HEALTH_PATH}"
SERVICE_HEALTH_EXIT_CODE=$?
set -e

EXPECTED_TOTAL="$(
  "${ENV_DIR}/bin/python" - "${TRACE_PATH}" "${MAX_REQUESTS}" <<'PY'
import json
import sys
from pathlib import Path

total = len(json.loads(Path(sys.argv[1]).read_text())["requests"])
limit = int(sys.argv[2])
print(min(total, limit) if limit > 0 else total)
PY
)"

echo "[4/7] Validate raw records"
VALIDATE_ARGS=()
if [[ "${ALLOW_FAILED_REQUESTS}" == "1" ]]; then
  VALIDATE_ARGS+=(--allow-failures)
fi
set +e
"${ENV_DIR}/bin/python" "${ROOT_DIR}/scripts/validate_replay_results.py" \
  --system Llumnix \
  --replay "${RAW_PATH}" \
  --expected-total "${EXPECTED_TOTAL}" \
  "${VALIDATE_ARGS[@]}"
VALIDATION_EXIT_CODE=$?
set -e

echo "[5/7] Summarize static four-GPU lifecycle cost"
"${ENV_DIR}/bin/python" "${ROOT_DIR}/scripts/summarize_llumnix_replay.py" \
  --replay "${RAW_PATH}" \
  --output "${SUMMARY_PATH}" \
  --model "${MODEL_PATH}" \
  --model-key "${MODEL_KEY}" \
  --gpu-budget 4 \
  --startup-sec "${STARTUP_SEC}" \
  --gpu-cost-per-second-usd "${GPU_COST_PER_SECOND_USD}"

echo "[6/7] Finalize artifact hashes"
"${ENV_DIR}/bin/python" - \
  "${MANIFEST_PATH}" "${RAW_PATH}" "${SUMMARY_PATH}" \
  "${RESOURCE_GUARD_PATH}" "${REPLAY_EXIT_CODE}" \
  "${VALIDATION_EXIT_CODE}" "${SERVICE_HEALTH_PATH}" \
  "${SERVICE_HEALTH_EXIT_CODE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, raw_path, summary_path, guard_path = map(Path, sys.argv[1:5])
replay_exit_code = int(sys.argv[5])
validation_exit_code = int(sys.argv[6])
service_health_path = Path(sys.argv[7])
service_health_exit_code = int(sys.argv[8])

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

rows = []
for line in guard_path.read_text().splitlines()[1:]:
    if line.strip():
        rows.append(line.split(","))
payload = json.loads(manifest_path.read_text())
payload.update({
    "raw_records_path": str(raw_path),
    "raw_records_sha256": sha(raw_path),
    "source_summary_path": str(summary_path),
    "source_summary_sha256": sha(summary_path),
    "resource_guard_path": str(guard_path),
    "resource_guard_sha256": sha(guard_path),
    "resource_guard_min_available_kb": min(int(row[1]) for row in rows),
    "resource_guard_max_gpu_temperature_c": max(int(row[4]) for row in rows),
    "service_health_path": str(service_health_path),
    "service_health_sha256": sha(service_health_path),
    "service_health_exit_code": service_health_exit_code,
    "replay_exit_code": replay_exit_code,
    "validation_exit_code": validation_exit_code,
    "strict_zero_failure_pass": (
        replay_exit_code == 0
        and validation_exit_code == 0
        and service_health_exit_code == 0
    ),
})
manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "[7/7] Complete"
echo "run_dir=${RUN_DIR}"
echo "raw_records_path=${RAW_PATH}"
echo "source_summary_path=${SUMMARY_PATH}"
if [[ "${REPLAY_EXIT_CODE}" != "0" ]]; then
  exit "${REPLAY_EXIT_CODE}"
fi
if [[ "${VALIDATION_EXIT_CODE}" != "0" ]]; then
  exit "${VALIDATION_EXIT_CODE}"
fi
if [[ "${SERVICE_HEALTH_EXIT_CODE}" != "0" ]]; then
  exit "${SERVICE_HEALTH_EXIT_CODE}"
fi
