#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${DISTSERVE_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
RELAY_ROOT="${RELAY_ROOT:-/home/qhq/relayserve_serverless_llm}"
PROJECT_ROOT="${DISTSERVE_PROJECT_ROOT:-${ROOT_DIR}/workspaces/distserve_official_20260615}"
SOURCE_ROOT="${DISTSERVE_SOURCE_ROOT:-${ROOT_DIR}/vendor_new_baselines/DistServe_main_20260614}"
SWIFT_ROOT="${DISTSERVE_SWIFT_ROOT:-${PROJECT_ROOT}/SwiftTransformer}"
ENV_DIR="${DISTSERVE_ENV_DIR:-/home/qhq/anaconda3/envs/distserve_official_20260615}"
REFERENCE_ENV_DIR="${DISTSERVE_REFERENCE_ENV_DIR:-/home/qhq/anaconda3/envs/llumnix_official_20260615}"
MODEL_KEY="${1:?usage: $0 3b|7b [max_requests] [run_tag]}"
MAX_REQUESTS="${2:-0}"
RUN_TAG="${3:-$(date -u +%Y%m%dT%H%M%SZ)_distserve_${MODEL_KEY}_r${MAX_REQUESTS}}"
TRACE_ROLE="${DISTSERVE_TRACE_ROLE:-formal4000}"
PORT="${DISTSERVE_PORT:-8200}"
GPU_MEMORY_UTILIZATION="${DISTSERVE_GPU_MEMORY_UTILIZATION:-0.84}"
BLOCK_SIZE="${DISTSERVE_BLOCK_SIZE:-16}"
MAX_NUM_BLOCKS_PER_REQ="${DISTSERVE_MAX_NUM_BLOCKS_PER_REQ:-192}"
SWAP_SPACE_GB="${DISTSERVE_SWAP_SPACE_GB:-16}"
CONTEXT_MAX_BATCH_SIZE="${DISTSERVE_CONTEXT_MAX_BATCH_SIZE:-64}"
CONTEXT_MAX_TOKENS_PER_BATCH="${DISTSERVE_CONTEXT_MAX_TOKENS_PER_BATCH:-8192}"
DECODING_MAX_BATCH_SIZE="${DISTSERVE_DECODING_MAX_BATCH_SIZE:-256}"
DECODING_MAX_TOKENS_PER_BATCH="${DISTSERVE_DECODING_MAX_TOKENS_PER_BATCH:-65536}"
SERVICE_READY_TIMEOUT_S="${DISTSERVE_SERVICE_READY_TIMEOUT_S:-1800}"
REQUEST_TIMEOUT_S="${DISTSERVE_REQUEST_TIMEOUT_S:-1800}"
CHECKPOINT_INTERVAL="${DISTSERVE_CHECKPOINT_INTERVAL:-256}"
SEMANTIC_SAMPLE_COUNT="${DISTSERVE_SEMANTIC_SAMPLE_COUNT:-16}"
SEMANTIC_NEW_TOKENS="${DISTSERVE_SEMANTIC_NEW_TOKENS:-8}"
MIN_AVAILABLE_MEMORY_GB="${DISTSERVE_MIN_AVAILABLE_MEMORY_GB:-40}"
MAX_GPU_TEMPERATURE_C="${DISTSERVE_MAX_GPU_TEMPERATURE_C:-88}"
RESOURCE_SAMPLE_INTERVAL_S="${DISTSERVE_RESOURCE_SAMPLE_INTERVAL_S:-2}"
GPU_COST_PER_SECOND_USD="${DISTSERVE_GPU_COST_PER_SECOND_USD:-0.0008}"
DISTSERVE_CACHE="${DISTSERVE_CACHE:-${ROOT_DIR}/cache/distserve_official_20260615}"

case "${MODEL_KEY}" in
  3b)
    MODEL_PATH="/home/qhq/serverless_llm_experiment_retry14_baseline/models/LLM-Research--Llama-3.2-3B-Instruct"
    TRACE_PATH="${RELAY_ROOT}/data_imports/traces/multi_turn_continuation/llama32_3b/llama32_3b_burstgptv2_chronological_sharegpt_multi_turn_formal4000_v1_trace.json"
    TTFT_SLO_MS="180"
    TPOT_SLO_MS="14"
    MAX_MODEL_LEN="${DISTSERVE_MAX_MODEL_LEN:-3072}"
    ;;
  7b)
    MODEL_PATH="/home/qhq/serverless_llm_experiment/models/meta-llama--Llama-2-7b-hf"
    TRACE_PATH="${RELAY_ROOT}/data_imports/traces/multi_turn_continuation/llama2_7b/llama2_7b_burstgptv2_chronological_sharegpt_multi_turn_formal4000_v1_trace.json"
    TTFT_SLO_MS="440"
    TPOT_SLO_MS="32"
    MAX_MODEL_LEN="${DISTSERVE_MAX_MODEL_LEN:-3072}"
    ;;
  *)
    echo "model must be 3b or 7b" >&2
    exit 2
    ;;
esac

MODEL_PATH="${DISTSERVE_MODEL_PATH_OVERRIDE:-${MODEL_PATH}}"
TRACE_PATH="${DISTSERVE_TRACE_PATH_OVERRIDE:-${TRACE_PATH}}"
RUN_DIR="${DISTSERVE_RUN_DIR:-${ROOT_DIR}/results/relayserve_continuation/distserve/${RUN_TAG}}"
RAW_PATH="${RUN_DIR}/raw_records.json"
SUMMARY_PATH="${RUN_DIR}/source_summary.json"
MANIFEST_PATH="${RUN_DIR}/manifest.json"
LOG_DIR="${RUN_DIR}/logs"
SNAPSHOT_DIR="${RUN_DIR}/frozen_config"
REFERENCE_PATH="${LOG_DIR}/semantic_reference.json"
SEMANTIC_REPORT_PATH="${LOG_DIR}/semantic_validation.json"
RESOURCE_GUARD_PATH="${LOG_DIR}/resource_guard.csv"
SERVICE_PID=""
RESOURCE_GUARD_PID=""
STARTUP_SEC="0"

if [[ -e "${RUN_DIR}" ]]; then
  echo "refusing to overwrite existing DistServe run directory: ${RUN_DIR}" >&2
  exit 3
fi
for required in \
  "${TRACE_PATH}" \
  "${MODEL_PATH}/config.json" \
  "${PROJECT_ROOT}/distserve/api_server/distserve_api_server.py" \
  "${SWIFT_ROOT}/build/lib/libst_pybinding.so" \
  "${ENV_DIR}/bin/python" \
  "${ENV_DIR}/bin/ray" \
  "${REFERENCE_ENV_DIR}/bin/python"
do
  if [[ ! -e "${required}" ]]; then
    echo "missing required DistServe input: ${required}" >&2
    exit 4
  fi
done

mkdir -p "${LOG_DIR}" "${SNAPSHOT_DIR}" "${DISTSERVE_CACHE}"

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ -n "${RESOURCE_GUARD_PID}" ]]; then
    kill "${RESOURCE_GUARD_PID}" 2>/dev/null || true
    wait "${RESOURCE_GUARD_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SERVICE_PID}" ]]; then
    kill -TERM -- "-${SERVICE_PID}" 2>/dev/null || true
    sleep 3
    kill -KILL -- "-${SERVICE_PID}" 2>/dev/null || true
    wait "${SERVICE_PID}" 2>/dev/null || true
  fi
  PYTHONNOUSERSITE=1 "${ENV_DIR}/bin/ray" stop --force \
    >"${LOG_DIR}/ray_stop.log" 2>&1 || true
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

echo "[0/9] Verify isolated official DistServe environment"
PYTHONNOUSERSITE=1 "${ENV_DIR}/bin/python" - <<'PY'
import pathlib
import torch

import distserve

library = (
    pathlib.Path(distserve.__file__).resolve().parents[1]
    / "SwiftTransformer/build/lib/libst_pybinding.so"
)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("distserve", distserve.__file__)
print("swifttransformer", library)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if not library.exists():
    raise SystemExit(f"SwiftTransformer library is missing: {library}")
PY

echo "[1/9] Freeze source provenance and launch parameters"
git -C "${SOURCE_ROOT}" rev-parse HEAD >"${SNAPSHOT_DIR}/distserve_source_git_commit.txt"
git -C "${SOURCE_ROOT}" status --short >"${SNAPSHOT_DIR}/distserve_source_git_status.txt"
git -C "${PROJECT_ROOT}" rev-parse HEAD >"${SNAPSHOT_DIR}/distserve_workspace_git_commit.txt"
git -C "${PROJECT_ROOT}" status --short >"${SNAPSHOT_DIR}/distserve_workspace_git_status.txt"
git -C "${SWIFT_ROOT}" rev-parse HEAD >"${SNAPSHOT_DIR}/swifttransformer_git_commit.txt"
git -C "${SWIFT_ROOT}" status --short >"${SNAPSHOT_DIR}/swifttransformer_git_status.txt"
PYTHONNOUSERSITE=1 "${ENV_DIR}/bin/python" -m pip freeze \
  >"${SNAPSHOT_DIR}/pip_freeze.txt"
nvidia-smi -q >"${SNAPSHOT_DIR}/nvidia_smi_q.txt"
cp "${ROOT_DIR}/patches/distserve_relayserve_compat.patch" \
  "${SNAPSHOT_DIR}/distserve_relayserve_compat.patch"

PYTHONNOUSERSITE=1 "${ENV_DIR}/bin/python" - \
  "${SNAPSHOT_DIR}/launch_profile.json" <<PY
import json
import pathlib

payload = {
    "topology": "2 context TP + 2 decoding TP",
    "gpu_budget": 4,
    "context_tensor_parallel_size": 2,
    "context_pipeline_parallel_size": 1,
    "decoding_tensor_parallel_size": 2,
    "decoding_pipeline_parallel_size": 1,
    "block_size": int("${BLOCK_SIZE}"),
    "max_num_blocks_per_req": int("${MAX_NUM_BLOCKS_PER_REQ}"),
    "gpu_memory_utilization": float("${GPU_MEMORY_UTILIZATION}"),
    "swap_space_gb": int("${SWAP_SPACE_GB}"),
    "context_max_batch_size": int("${CONTEXT_MAX_BATCH_SIZE}"),
    "context_max_tokens_per_batch": int("${CONTEXT_MAX_TOKENS_PER_BATCH}"),
    "decoding_max_batch_size": int("${DECODING_MAX_BATCH_SIZE}"),
    "decoding_max_tokens_per_batch": int("${DECODING_MAX_TOKENS_PER_BATCH}"),
    "max_model_len_comparison_guard": int("${MAX_MODEL_LEN}"),
    "semantic_sample_count": int("${SEMANTIC_SAMPLE_COUNT}"),
    "semantic_new_tokens": int("${SEMANTIC_NEW_TOKENS}"),
}
pathlib.Path("${SNAPSHOT_DIR}/launch_profile.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY

echo "[2/9] Generate frozen Hugging Face semantic reference"
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  "${REFERENCE_ENV_DIR}/bin/python" \
  "${ROOT_DIR}/scripts/validate_distserve_semantics.py" reference \
  --trace "${TRACE_PATH}" \
  --model "${MODEL_PATH}" \
  --tokenizer "${MODEL_PATH}" \
  --output "${REFERENCE_PATH}" \
  --sample-count "${SEMANTIC_SAMPLE_COUNT}" \
  --new-tokens "${SEMANTIC_NEW_TOKENS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype float16 \
  --device cuda:0 \
  2>&1 | tee "${LOG_DIR}/semantic_reference.log"

start_resource_guard
PYTHONNOUSERSITE=1 "${ENV_DIR}/bin/ray" stop --force \
  >"${LOG_DIR}/ray_preclean.log" 2>&1 || true
START_MONOTONIC="$("${ENV_DIR}/bin/python" -c 'import time; print(time.monotonic())')"

echo "[3/9] Start official DistServe 2P+2D topology"
setsid env \
  PYTHONNOUSERSITE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  DISTSERVE_CACHE="${DISTSERVE_CACHE}" \
  "${ENV_DIR}/bin/python" -m distserve.api_server.distserve_api_server \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --model "${MODEL_PATH}" \
  --tokenizer "${MODEL_PATH}" \
  --seed 42 \
  --context-tensor-parallel-size 2 \
  --context-pipeline-parallel-size 1 \
  --decoding-tensor-parallel-size 2 \
  --decoding-pipeline-parallel-size 1 \
  --block-size "${BLOCK_SIZE}" \
  --max-num-blocks-per-req "${MAX_NUM_BLOCKS_PER_REQ}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --swap-space "${SWAP_SPACE_GB}" \
  --context-sched-policy fcfs \
  --context-max-batch-size "${CONTEXT_MAX_BATCH_SIZE}" \
  --context-max-tokens-per-batch "${CONTEXT_MAX_TOKENS_PER_BATCH}" \
  --decoding-sched-policy fcfs \
  --decoding-max-batch-size "${DECODING_MAX_BATCH_SIZE}" \
  --decoding-max-tokens-per-batch "${DECODING_MAX_TOKENS_PER_BATCH}" \
  >"${LOG_DIR}/service.log" 2>&1 &
SERVICE_PID=$!

echo "[4/9] Wait for API readiness and verify exact token semantics"
deadline=$((SECONDS + SERVICE_READY_TIMEOUT_S))
until curl -fsS "http://127.0.0.1:${PORT}/docs" >/dev/null; do
  if ! kill -0 "${SERVICE_PID}" 2>/dev/null; then
    echo "DistServe exited before readiness" >&2
    tail -n 200 "${LOG_DIR}/service.log" >&2 || true
    exit 20
  fi
  if (( SECONDS >= deadline )); then
    echo "DistServe readiness timed out" >&2
    tail -n 200 "${LOG_DIR}/service.log" >&2 || true
    exit 21
  fi
  sleep 2
done

PYTHONNOUSERSITE=1 "${ENV_DIR}/bin/python" \
  "${ROOT_DIR}/scripts/validate_distserve_semantics.py" compare \
  --reference "${REFERENCE_PATH}" \
  --base-url "http://127.0.0.1:${PORT}" \
  --output "${SEMANTIC_REPORT_PATH}" \
  --timeout-s 300 \
  --required-exact-fraction 1.0 \
  --required-first-token-fraction 1.0 \
  --required-token-fraction 1.0 \
  2>&1 | tee "${LOG_DIR}/semantic_validation.log"

STARTUP_SEC="$(
  "${ENV_DIR}/bin/python" -c \
    "import time; print(time.monotonic() - float('${START_MONOTONIC}'))"
)"

echo "[5/9] Build frozen run manifest"
PYTHONNOUSERSITE=1 "${ENV_DIR}/bin/python" - \
  "${MANIFEST_PATH}" "${TRACE_PATH}" "${MODEL_PATH}" "${PROJECT_ROOT}" \
  "${SOURCE_ROOT}" "${SWIFT_ROOT}" "${ROOT_DIR}" "${RELAY_ROOT}" \
  "${RUN_TAG}" "${MODEL_KEY}" "${MAX_REQUESTS}" "${TRACE_ROLE}" \
  "${SNAPSHOT_DIR}" "${REFERENCE_PATH}" "${SEMANTIC_REPORT_PATH}" \
  "${STARTUP_SEC}" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    manifest_path, trace_path, model_path, project_root, source_root,
    swift_root, root, relay_root, run_tag, model_key, max_requests,
    trace_role, snapshot_dir, reference_path, semantic_report_path,
    startup_sec,
) = sys.argv[1:]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git_head(path):
    return subprocess.check_output(
        ["git", "-C", path, "rev-parse", "HEAD"], text=True
    ).strip()

root_path = Path(root)
source_paths = [
    root_path / "scripts/replay_distserve_trace.py",
    root_path / "scripts/run_distserve_relayserve_continuation.sh",
    root_path / "scripts/summarize_distserve_replay.py",
    root_path / "scripts/validate_distserve_semantics.py",
]
snapshot_path = Path(snapshot_dir)
payload = {
    "schema": "relayserve_external_distserve_run_v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_tag": run_tag,
    "system": "DistServe",
    "model_key": model_key,
    "max_requests": int(max_requests),
    "trace_path": trace_path,
    "trace_sha256": sha(trace_path),
    "trace_role": trace_role,
    "model_path": model_path,
    "model_config_sha256": sha(Path(model_path) / "config.json"),
    "distserve_workspace": project_root,
    "distserve_git_commit": git_head(project_root),
    "distserve_source_git_commit": git_head(source_root),
    "swifttransformer_git_commit": git_head(swift_root),
    "baseline_harness_git_commit": git_head(root),
    "relayserve_git_commit": git_head(relay_root),
    "harness_source_sha256": {
        str(path.relative_to(root_path)): sha(path) for path in source_paths
    },
    "runtime_env": "/home/qhq/anaconda3/envs/distserve_official_20260615",
    "launch_profile": json.loads(
        (snapshot_path / "launch_profile.json").read_text()
    ),
    "comparison_contract": {
        "workload": f"RelayServe frozen continuation trace ({trace_role}), rate=1.00x",
        "slo_profile": "paper_nominal",
        "gpu_budget": 4,
        "base_model_only": True,
        "lora_enabled": False,
        "system_mode": "official DistServe 2-context-TP + 2-decoding-TP",
        "cost_model": "static four-GPU deployment lifecycle",
    },
    "semantic_reference_path": reference_path,
    "semantic_reference_sha256": sha(reference_path),
    "semantic_validation_report": semantic_report_path,
    "semantic_validation_sha256": sha(semantic_report_path),
    "initial_runtime_startup_sec": float(startup_sec),
    "frozen_config_dir": snapshot_dir,
    "frozen_config_sha256": {
        path.name: sha(path)
        for path in sorted(snapshot_path.iterdir())
        if path.is_file()
    },
}
Path(manifest_path).write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "[6/9] Replay frozen continuation workload"
REPLAY_ARGS=()
if [[ "${MAX_REQUESTS}" != "0" ]]; then
  REPLAY_ARGS+=(--max-requests "${MAX_REQUESTS}")
fi
set +e
PYTHONNOUSERSITE=1 "${ENV_DIR}/bin/python" \
  "${ROOT_DIR}/scripts/replay_distserve_trace.py" \
  --trace "${TRACE_PATH}" \
  --base-url "http://127.0.0.1:${PORT}" \
  --tokenizer "${MODEL_PATH}" \
  --output "${RAW_PATH}" \
  --label "${RUN_TAG}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --ttft-slo-ms "${TTFT_SLO_MS}" \
  --tpot-slo-ms "${TPOT_SLO_MS}" \
  --timeout-s "${REQUEST_TIMEOUT_S}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
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
set +e
"${ENV_DIR}/bin/python" "${ROOT_DIR}/scripts/validate_replay_results.py" \
  --system DistServe \
  --replay "${RAW_PATH}" \
  --expected-total "${EXPECTED_TOTAL}"
VALIDATION_EXIT_CODE=$?
set -e

echo "[7/9] Summarize static four-GPU lifecycle cost"
"${ENV_DIR}/bin/python" "${ROOT_DIR}/scripts/summarize_distserve_replay.py" \
  --replay "${RAW_PATH}" \
  --output "${SUMMARY_PATH}" \
  --model "${MODEL_PATH}" \
  --model-key "${MODEL_KEY}" \
  --semantic-validation-report "${SEMANTIC_REPORT_PATH}" \
  --gpu-budget 4 \
  --startup-sec "${STARTUP_SEC}" \
  --gpu-cost-per-second-usd "${GPU_COST_PER_SECOND_USD}"

echo "[8/9] Verify runtime health and freeze artifact hashes"
SERVICE_HEALTH_EXIT_CODE=0
if ! kill -0 "${SERVICE_PID}" 2>/dev/null; then
  SERVICE_HEALTH_EXIT_CODE=1
fi
if rg -n -i \
  "Traceback|CUDA out of memory|NVRM: Xid|WorkerCrashedError|RayActorError|core dumped|segmentation fault" \
  "${LOG_DIR}/service.log" >"${LOG_DIR}/service_health_errors.txt"; then
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
        and bool(summary["semantic_validation_pass"])
    ),
})
manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "[9/9] Complete"
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
