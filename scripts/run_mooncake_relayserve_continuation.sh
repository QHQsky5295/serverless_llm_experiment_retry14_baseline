#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${MOONCAKE_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
RELAY_ROOT="${RELAY_ROOT:-/home/qhq/relayserve_serverless_llm}"
VLLM_ROOT="${MOONCAKE_VLLM_ROOT:-${ROOT_DIR}/workspaces/mooncake_vllm_official_20260615}"
MOONCAKE_ROOT="${MOONCAKE_SOURCE_ROOT:-${ROOT_DIR}/vendor_new_baselines/Mooncake_main_20260615}"
ENV_DIR="${MOONCAKE_ENV_DIR:-/home/qhq/.venvs/mooncake_official_20260615}"
if [[ -x "${ENV_DIR}/bin/vllm" ]]; then
  VLLM_CMD=("${ENV_DIR}/bin/vllm")
else
  VLLM_CMD=("${ENV_DIR}/bin/python" -m vllm.entrypoints.cli.main)
fi
MODEL_KEY="${1:?usage: $0 3b|7b [max_requests] [run_tag]}"
MAX_REQUESTS="${2:-0}"
RUN_TAG="${3:-$(date -u +%Y%m%dT%H%M%SZ)_mooncake_${MODEL_KEY}_r${MAX_REQUESTS}}"
TRACE_ROLE="${MOONCAKE_TRACE_ROLE:-formal4000}"
PROXY_PORT="${MOONCAKE_PROXY_PORT:-8100}"
PREFILL_PORTS=(${MOONCAKE_PREFILL_PORTS:-8110 8111})
BOOTSTRAP_PORTS=(${MOONCAKE_BOOTSTRAP_PORTS:-8998 8999})
DECODE_PORTS=(${MOONCAKE_DECODE_PORTS:-8120 8121})
PREFILL_GPUS=(${MOONCAKE_PREFILL_GPUS:-0 1})
DECODE_GPUS=(${MOONCAKE_DECODE_GPUS:-2 3})
GPU_MEMORY_UTILIZATION="${MOONCAKE_GPU_MEMORY_UTILIZATION:-0.84}"
MAX_NUM_SEQS="${MOONCAKE_MAX_NUM_SEQS:-64}"
MAX_MODEL_LEN="${MOONCAKE_MAX_MODEL_LEN:-3072}"
DTYPE="${MOONCAKE_DTYPE:-float16}"
SERVICE_READY_TIMEOUT_S="${MOONCAKE_SERVICE_READY_TIMEOUT_S:-900}"
SERVICE_STABILIZATION_S="${MOONCAKE_SERVICE_STABILIZATION_S:-10}"
REQUEST_TIMEOUT_S="${MOONCAKE_REQUEST_TIMEOUT_S:-1800}"
MIN_AVAILABLE_MEMORY_GB="${MOONCAKE_MIN_AVAILABLE_MEMORY_GB:-40}"
MAX_GPU_TEMPERATURE_C="${MOONCAKE_MAX_GPU_TEMPERATURE_C:-88}"
RESOURCE_SAMPLE_INTERVAL_S="${MOONCAKE_RESOURCE_SAMPLE_INTERVAL_S:-2}"
GPU_COST_PER_SECOND_USD="${MOONCAKE_GPU_COST_PER_SECOND_USD:-0.0008}"
ALLOW_FAILED_REQUESTS="${MOONCAKE_ALLOW_FAILED_REQUESTS:-0}"

case "${MODEL_KEY}" in
  3b)
    MODEL_PATH="/home/qhq/serverless_llm_experiment_retry14_baseline/models/LLM-Research--Llama-3.2-3B-Instruct"
    SERVED_MODEL_NAME="mooncake-relayserve-llama32-3b"
    TRACE_PATH="${RELAY_ROOT}/data_imports/traces/multi_turn_continuation/llama32_3b/llama32_3b_burstgptv2_chronological_sharegpt_multi_turn_formal4000_v1_trace.json"
    TTFT_SLO_MS="180"
    TPOT_SLO_MS="14"
    ;;
  7b)
    MODEL_PATH="/home/qhq/serverless_llm_experiment/models/meta-llama--Llama-2-7b-hf"
    SERVED_MODEL_NAME="mooncake-relayserve-llama2-7b"
    TRACE_PATH="${RELAY_ROOT}/data_imports/traces/multi_turn_continuation/llama2_7b/llama2_7b_burstgptv2_chronological_sharegpt_multi_turn_formal4000_v1_trace.json"
    TTFT_SLO_MS="440"
    TPOT_SLO_MS="32"
    ;;
  *)
    echo "model must be 3b or 7b" >&2
    exit 2
    ;;
esac

MODEL_PATH="${MOONCAKE_MODEL_PATH_OVERRIDE:-${MODEL_PATH}}"
TRACE_PATH="${MOONCAKE_TRACE_PATH_OVERRIDE:-${TRACE_PATH}}"
RUN_DIR="${MOONCAKE_RUN_DIR:-${ROOT_DIR}/results/relayserve_continuation/mooncake/${RUN_TAG}}"
RAW_PATH="${RUN_DIR}/raw_records.json"
SUMMARY_PATH="${RUN_DIR}/source_summary.json"
MANIFEST_PATH="${RUN_DIR}/manifest.json"
LOG_DIR="${RUN_DIR}/logs"
SNAPSHOT_DIR="${RUN_DIR}/frozen_config"
RESOURCE_GUARD_PATH="${LOG_DIR}/resource_guard.csv"
PROBE_PATH="${LOG_DIR}/full_path_probe.json"
PROXY_SCRIPT="${VLLM_ROOT}/examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py"
RESOURCE_GUARD_PID=""
SERVICE_PIDS=()
STARTUP_SEC="0"

if [[ -e "${RUN_DIR}" ]]; then
  echo "refusing to overwrite existing Mooncake run directory: ${RUN_DIR}" >&2
  exit 3
fi
if [[ "${#PREFILL_PORTS[@]}" != "2" || "${#BOOTSTRAP_PORTS[@]}" != "2" \
   || "${#DECODE_PORTS[@]}" != "2" || "${#PREFILL_GPUS[@]}" != "2" \
   || "${#DECODE_GPUS[@]}" != "2" ]]; then
  echo "Mooncake formal profile requires exactly 2 prefiller and 2 decoder instances" >&2
  exit 4
fi
for required in \
  "${TRACE_PATH}" \
  "${MODEL_PATH}/config.json" \
  "${ENV_DIR}/bin/python" \
  "${PROXY_SCRIPT}"
do
  if [[ ! -e "${required}" ]]; then
    echo "missing required Mooncake input: ${required}" >&2
    exit 5
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
  for pid in "${SERVICE_PIDS[@]}"; do
    kill -TERM -- "-${pid}" 2>/dev/null || true
  done
  sleep 2
  for pid in "${SERVICE_PIDS[@]}"; do
    kill -KILL -- "-${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  done
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
        echo "memory_guard_breach: available_kb=${available_kb}" >&2
        kill -TERM "${runner_pid}" 2>/dev/null || true
        exit 90
      fi
      if (( max_gpu_temp >= MAX_GPU_TEMPERATURE_C )); then
        echo "temperature_guard_breach: max_gpu_temp_c=${max_gpu_temp}" >&2
        kill -TERM "${runner_pid}" 2>/dev/null || true
        exit 91
      fi
      sleep "${RESOURCE_SAMPLE_INTERVAL_S}"
    done
  ) >"${RESOURCE_GUARD_PATH}" 2>"${LOG_DIR}/resource_guard.err" &
  RESOURCE_GUARD_PID=$!
}

echo "[0/8] Verify isolated official Mooncake/vLLM environment"
PYTHONNOUSERSITE=1 "${ENV_DIR}/bin/python" - <<'PY'
import mooncake.engine
import torch
import vllm

print("vllm", vllm.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("mooncake.engine", mooncake.engine.__file__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
PY

echo "[1/8] Freeze provenance and launch parameters"
git -C "${VLLM_ROOT}" rev-parse HEAD >"${SNAPSHOT_DIR}/vllm_git_commit.txt"
git -C "${VLLM_ROOT}" status --short >"${SNAPSHOT_DIR}/vllm_git_status.txt"
git -C "${MOONCAKE_ROOT}" rev-parse HEAD >"${SNAPSHOT_DIR}/mooncake_git_commit.txt"
git -C "${MOONCAKE_ROOT}" status --short >"${SNAPSHOT_DIR}/mooncake_git_status.txt"
"${ENV_DIR}/bin/python" -m pip freeze >"${SNAPSHOT_DIR}/pip_freeze.txt"
nvidia-smi -q >"${SNAPSHOT_DIR}/nvidia_smi_q.txt"

KV_PRODUCER_CONFIG='{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_protocol":"tcp","num_workers":10}}'
KV_CONSUMER_CONFIG='{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"mooncake_protocol":"tcp","num_workers":10}}'
cat >"${SNAPSHOT_DIR}/launch_profile.json" <<EOF
{
  "topology": "2P+2D",
  "protocol": "tcp",
  "prefill_gpus": [${PREFILL_GPUS[0]}, ${PREFILL_GPUS[1]}],
  "decode_gpus": [${DECODE_GPUS[0]}, ${DECODE_GPUS[1]}],
  "prefill_ports": [${PREFILL_PORTS[0]}, ${PREFILL_PORTS[1]}],
  "decode_ports": [${DECODE_PORTS[0]}, ${DECODE_PORTS[1]}],
  "bootstrap_ports": [${BOOTSTRAP_PORTS[0]}, ${BOOTSTRAP_PORTS[1]}],
  "proxy_port": ${PROXY_PORT},
  "gpu_memory_utilization": ${GPU_MEMORY_UTILIZATION},
  "max_num_seqs": ${MAX_NUM_SEQS},
  "max_model_len": ${MAX_MODEL_LEN},
  "dtype": "${DTYPE}",
  "enforce_eager": true
}
EOF

"${ENV_DIR}/bin/python" - \
  "${MANIFEST_PATH}" "${TRACE_PATH}" "${MODEL_PATH}" "${VLLM_ROOT}" \
  "${MOONCAKE_ROOT}" "${ROOT_DIR}" "${RELAY_ROOT}" "${RUN_TAG}" \
  "${MODEL_KEY}" "${MAX_REQUESTS}" "${TRACE_ROLE}" "${SNAPSHOT_DIR}" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    manifest_path, trace_path, model_path, vllm_root, mooncake_root,
    root, relay_root, run_tag, model_key, max_requests, trace_role,
    snapshot_dir,
) = sys.argv[1:]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git_head(path):
    return subprocess.check_output(
        ["git", "-C", path, "rev-parse", "HEAD"], text=True
    ).strip()

root_path = Path(root)
source_paths = [
    root_path / "scripts/replay_openai_trace.py",
    root_path / "scripts/run_mooncake_relayserve_continuation.sh",
    root_path / "scripts/summarize_mooncake_replay.py",
    root_path / "scripts/validate_replay_results.py",
]
snapshot_path = Path(snapshot_dir)
payload = {
    "schema": "relayserve_external_mooncake_run_v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_tag": run_tag,
    "system": "Mooncake",
    "model_key": model_key,
    "max_requests": int(max_requests),
    "trace_path": trace_path,
    "trace_sha256": sha(trace_path),
    "trace_role": trace_role,
    "model_path": model_path,
    "model_config_sha256": sha(Path(model_path) / "config.json"),
    "vllm_repo": vllm_root,
    "vllm_git_commit": git_head(vllm_root),
    "mooncake_repo": mooncake_root,
    "mooncake_git_commit": git_head(mooncake_root),
    "baseline_harness_git_commit": git_head(root),
    "relayserve_git_commit": git_head(relay_root),
    "harness_source_sha256": {
        str(path.relative_to(root_path)): sha(path) for path in source_paths
    },
    "runtime_env": "/home/qhq/.venvs/mooncake_official_20260615",
    "launch_profile": json.loads(
        (snapshot_path / "launch_profile.json").read_text()
    ),
    "comparison_contract": {
        "workload": f"RelayServe frozen continuation trace ({trace_role}), rate=1.00x",
        "slo_profile": "paper_nominal",
        "gpu_budget": 4,
        "base_model_only": True,
        "lora_enabled": False,
        "system_mode": "official MooncakeConnector 2P+2D over TCP",
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
START_MONOTONIC="$("${ENV_DIR}/bin/python" -c 'import time; print(time.monotonic())')"

COMMON_ARGS=(
  serve "${MODEL_PATH}"
  --host 127.0.0.1
  --served-model-name "${SERVED_MODEL_NAME}"
  --max-model-len "${MAX_MODEL_LEN}"
  --dtype "${DTYPE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --enforce-eager
  --disable-log-requests
  --generation-config vllm
)

echo "[2/8] Start two official Mooncake prefillers"
for index in 0 1; do
  setsid env \
    PYTHONNOUSERSITE=1 \
    CUDA_VISIBLE_DEVICES="${PREFILL_GPUS[$index]}" \
    VLLM_MOONCAKE_BOOTSTRAP_PORT="${BOOTSTRAP_PORTS[$index]}" \
    "${VLLM_CMD[@]}" "${COMMON_ARGS[@]}" \
    --port "${PREFILL_PORTS[$index]}" \
    --kv-transfer-config "${KV_PRODUCER_CONFIG}" \
    >"${LOG_DIR}/prefill$((index + 1)).log" 2>&1 &
  SERVICE_PIDS+=("$!")
done

echo "[3/8] Start two official Mooncake decoders"
for index in 0 1; do
  setsid env \
    PYTHONNOUSERSITE=1 \
    CUDA_VISIBLE_DEVICES="${DECODE_GPUS[$index]}" \
    "${VLLM_CMD[@]}" "${COMMON_ARGS[@]}" \
    --port "${DECODE_PORTS[$index]}" \
    --kv-transfer-config "${KV_CONSUMER_CONFIG}" \
    >"${LOG_DIR}/decode$((index + 1)).log" 2>&1 &
  SERVICE_PIDS+=("$!")
done

echo "[4/8] Wait for all four workers and start the official proxy"
deadline=$((SECONDS + SERVICE_READY_TIMEOUT_S))
for port in "${PREFILL_PORTS[@]}" "${DECODE_PORTS[@]}"; do
  until curl -fsS "http://127.0.0.1:${port}/health" >/dev/null; do
    for pid in "${SERVICE_PIDS[@]}"; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        echo "Mooncake worker exited before readiness" >&2
        tail -n 160 "${LOG_DIR}"/*.log >&2 || true
        exit 20
      fi
    done
    if (( SECONDS >= deadline )); then
      echo "Mooncake worker readiness timed out on port ${port}" >&2
      exit 21
    fi
    sleep 2
  done
done

for log in "${LOG_DIR}"/prefill*.log "${LOG_DIR}"/decode*.log; do
  if ! rg -q "Mooncake Transfer Engine is using tcp as its protocol" "${log}"; then
    echo "Mooncake TCP protocol evidence missing from ${log}" >&2
    exit 22
  fi
done

setsid env PYTHONNOUSERSITE=1 \
  "${ENV_DIR}/bin/python" "${PROXY_SCRIPT}" \
  --host 127.0.0.1 \
  --port "${PROXY_PORT}" \
  --prefill "http://127.0.0.1:${PREFILL_PORTS[0]}" "${BOOTSTRAP_PORTS[0]}" \
  --prefill "http://127.0.0.1:${PREFILL_PORTS[1]}" "${BOOTSTRAP_PORTS[1]}" \
  --decode "http://127.0.0.1:${DECODE_PORTS[0]}" \
  --decode "http://127.0.0.1:${DECODE_PORTS[1]}" \
  >"${LOG_DIR}/proxy.log" 2>&1 &
SERVICE_PIDS+=("$!")

sleep "${SERVICE_STABILIZATION_S}"
echo "Verify the full official prefill-transfer-decode path"
set +e
PYTHONNOUSERSITE=1 "${ENV_DIR}/bin/python" \
  "${ROOT_DIR}/scripts/replay_openai_trace.py" \
  --trace "${TRACE_PATH}" \
  --base-url "http://127.0.0.1:${PROXY_PORT}" \
  --output "${PROBE_PATH}" \
  --model-override "${SERVED_MODEL_NAME}" \
  --prompt-guard-tokenizer-model "${MODEL_PATH}" \
  --prompt-guard-max-model-len "${MAX_MODEL_LEN}" \
  --force-stream \
  --include-stream-usage \
  --min-output-tokens 2 \
  --max-requests 1 \
  --timeout-s 120 \
  --ttft-slo-ms "${TTFT_SLO_MS}" \
  --generation-seed 42 \
  --label "${RUN_TAG}_probe" \
  >"${LOG_DIR}/probe.log" 2>&1
PROBE_EXIT_CODE=$?
set -e
if [[ "${PROBE_EXIT_CODE}" != "0" ]]; then
  echo "Mooncake full-path probe process failed" >&2
  exit 23
fi
"${ENV_DIR}/bin/python" "${ROOT_DIR}/scripts/validate_replay_results.py" \
  --system Mooncake \
  --replay "${PROBE_PATH}" \
  --expected-total 1

STARTUP_SEC="$(
  "${ENV_DIR}/bin/python" -c \
    "import time; print(time.monotonic() - float('${START_MONOTONIC}'))"
)"
"${ENV_DIR}/bin/python" - \
  "${MANIFEST_PATH}" "${STARTUP_SEC}" "${PROBE_PATH}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
probe_path = Path(sys.argv[3])
payload["initial_runtime_startup_sec"] = float(sys.argv[2])
payload["full_path_probe_path"] = str(probe_path)
payload["full_path_probe_sha256"] = hashlib.sha256(
    probe_path.read_bytes()
).hexdigest()
path.write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "[5/8] Replay frozen continuation workload"
REPLAY_ARGS=()
if [[ "${MAX_REQUESTS}" != "0" ]]; then
  REPLAY_ARGS+=(--max-requests "${MAX_REQUESTS}")
fi
set +e
PYTHONNOUSERSITE=1 "${ENV_DIR}/bin/python" \
  "${ROOT_DIR}/scripts/replay_openai_trace.py" \
  --trace "${TRACE_PATH}" \
  --base-url "http://127.0.0.1:${PROXY_PORT}" \
  --output "${RAW_PATH}" \
  --model-override "${SERVED_MODEL_NAME}" \
  --prompt-guard-tokenizer-model "${MODEL_PATH}" \
  --prompt-guard-max-model-len "${MAX_MODEL_LEN}" \
  --force-stream \
  --include-stream-usage \
  --min-output-tokens 2 \
  --sleep-scale 1.0 \
  --timeout-s "${REQUEST_TIMEOUT_S}" \
  --ttft-slo-ms "${TTFT_SLO_MS}" \
  --generation-seed 42 \
  --label "${RUN_TAG}" \
  --abort-after-failures 8 \
  --abort-failures-min-done 8 \
  "${REPLAY_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/replay.log"
REPLAY_EXIT_CODE=${PIPESTATUS[0]}
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
VALIDATE_ARGS=()
if [[ "${ALLOW_FAILED_REQUESTS}" == "1" ]]; then
  VALIDATE_ARGS+=(--allow-failures)
fi
set +e
"${ENV_DIR}/bin/python" "${ROOT_DIR}/scripts/validate_replay_results.py" \
  --system Mooncake \
  --replay "${RAW_PATH}" \
  --expected-total "${EXPECTED_TOTAL}" \
  "${VALIDATE_ARGS[@]}"
VALIDATION_EXIT_CODE=$?
set -e

echo "[6/8] Summarize static four-GPU lifecycle cost"
"${ENV_DIR}/bin/python" "${ROOT_DIR}/scripts/summarize_mooncake_replay.py" \
  --replay "${RAW_PATH}" \
  --output "${SUMMARY_PATH}" \
  --model "${MODEL_PATH}" \
  --model-key "${MODEL_KEY}" \
  --gpu-budget 4 \
  --startup-sec "${STARTUP_SEC}" \
  --gpu-cost-per-second-usd "${GPU_COST_PER_SECOND_USD}" \
  --tpot-slo-ms "${TPOT_SLO_MS}"

echo "[7/8] Verify final service health and artifact hashes"
SERVICE_HEALTH_EXIT_CODE=0
for pid in "${SERVICE_PIDS[@]}"; do
  if ! kill -0 "${pid}" 2>/dev/null; then
    SERVICE_HEALTH_EXIT_CODE=1
  fi
done
if rg -n -i \
  "Traceback|Mooncake Transfer Engine initialization failed|Engine core initialization failed|CUDA out of memory|NVRM: Xid" \
  "${LOG_DIR}"/*.log >"${LOG_DIR}/service_health_errors.txt"; then
  SERVICE_HEALTH_EXIT_CODE=1
fi

"${ENV_DIR}/bin/python" - \
  "${MANIFEST_PATH}" "${RAW_PATH}" "${SUMMARY_PATH}" \
  "${RESOURCE_GUARD_PATH}" "${REPLAY_EXIT_CODE}" \
  "${VALIDATION_EXIT_CODE}" "${SERVICE_HEALTH_EXIT_CODE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, raw_path, summary_path, guard_path = map(Path, sys.argv[1:5])
replay_exit_code = int(sys.argv[5])
validation_exit_code = int(sys.argv[6])
service_health_exit_code = int(sys.argv[7])

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

rows = [
    line.split(",")
    for line in guard_path.read_text().splitlines()[1:]
    if line.strip()
]
payload = json.loads(manifest_path.read_text())
summary = json.loads(summary_path.read_text())
payload.update({
    "raw_records_path": str(raw_path),
    "raw_records_sha256": sha(raw_path),
    "source_summary_path": str(summary_path),
    "source_summary_sha256": sha(summary_path),
    "resource_guard_path": str(guard_path),
    "resource_guard_sha256": sha(guard_path),
    "resource_guard_min_available_kb": min(int(row[1]) for row in rows),
    "resource_guard_max_gpu_temperature_c": max(int(row[4]) for row in rows),
    "service_health_exit_code": service_health_exit_code,
    "replay_exit_code": replay_exit_code,
    "validation_exit_code": validation_exit_code,
    "strict_zero_failure_pass": (
        replay_exit_code == 0
        and validation_exit_code == 0
        and service_health_exit_code == 0
        and bool(summary["strict_zero_failure_pass"])
    ),
})
manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "[8/8] Complete"
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
