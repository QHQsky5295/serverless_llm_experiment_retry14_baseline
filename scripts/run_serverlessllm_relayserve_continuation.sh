#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
RELAY_ROOT="${RELAY_ROOT:-/home/qhq/relayserve_serverless_llm}"
CONFIG_PATH="${SLLM_CONFIG_PATH:-${ROOT_DIR}/configs/relayserve_continuation_baselines.yaml}"
PYTHON_ENV="${SLLM_RUNTIME_ENV:-sllm_vllm0102_newserverless_20260518}"
PYTHON_BIN="/home/qhq/anaconda3/envs/${PYTHON_ENV}/bin/python"
MODEL_KEY="${1:?usage: $0 3b|7b [max_requests] [run_tag]}"
MAX_REQUESTS="${2:-64}"
RUN_TAG="${3:-$(date -u +%Y%m%dT%H%M%SZ)_serverlessllm_${MODEL_KEY}_r${MAX_REQUESTS}}"
WORKER_GPUS="${SLLM_WORKER_GPUS:-0,1,2,3}"
MIN_INSTANCES="${SLLM_MIN_INSTANCES:-1}"
MAX_INSTANCES="${SLLM_MAX_INSTANCES:-4}"
TARGET_CONCURRENCY="${SLLM_TARGET_CONCURRENCY:-16}"
KEEP_ALIVE_S="${SLLM_KEEP_ALIVE_S:-300}"
TIMEOUT_S="${SLLM_TIMEOUT_S:-1800}"
READY_TIMEOUT_S="${SLLM_READY_TIMEOUT_S:-900}"
AUTO_STOP="${SLLM_AUTO_STOP_STACK:-1}"

case "${MODEL_KEY}" in
  3b)
    MODEL_PROFILE="relayserve_llama32_3b"
    WORKLOAD_PROFILE="relayserve_3b_paper_nominal"
    SERVING_MODEL_NAME="serverlessllm-relayserve-llama32-3b"
    TRACE_PATH="${RELAY_ROOT}/data_imports/traces/multi_turn_continuation/llama32_3b/llama32_3b_burstgptv2_chronological_sharegpt_multi_turn_formal4000_v1_trace.json"
    TOKENIZER_PATH="/home/qhq/serverless_llm_experiment_retry14_baseline/models/LLM-Research--Llama-3.2-3B-Instruct"
    TTFT_SLO_MS="180"
    ;;
  7b)
    MODEL_PROFILE="relayserve_llama2_7b"
    WORKLOAD_PROFILE="relayserve_7b_paper_nominal"
    SERVING_MODEL_NAME="serverlessllm-relayserve-llama2-7b"
    TRACE_PATH="${RELAY_ROOT}/data_imports/traces/multi_turn_continuation/llama2_7b/llama2_7b_burstgptv2_chronological_sharegpt_multi_turn_formal4000_v1_trace.json"
    TOKENIZER_PATH="/home/qhq/serverless_llm_experiment/models/meta-llama--Llama-2-7b-hf"
    TTFT_SLO_MS="440"
    ;;
  *)
    echo "model must be 3b or 7b, got: ${MODEL_KEY}" >&2
    exit 2
    ;;
esac

TRACE_PATH="${SLLM_TRACE_PATH_OVERRIDE:-${TRACE_PATH}}"
TRACE_ROLE="${SLLM_TRACE_ROLE:-formal4000}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "missing ServerlessLLM runtime Python: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${TRACE_PATH}" ]]; then
  echo "missing frozen trace: ${TRACE_PATH}" >&2
  exit 2
fi

RUN_DIR="${SLLM_RUN_DIR:-${ROOT_DIR}/results/relayserve_continuation/serverlessllm/${RUN_TAG}}"
DEPLOY_PATH="${RUN_DIR}/deploy.json"
REPLAY_PATH="${RUN_DIR}/raw_records.json"
SUMMARY_PATH="${RUN_DIR}/source_summary.json"
MANIFEST_PATH="${RUN_DIR}/manifest.json"
LOG_DIR="${RUN_DIR}/logs"
STACK_SUFFIX="$(printf '%s' "${RUN_TAG}" | tr -c 'A-Za-z0-9_.-' '_')"

mkdir -p "${LOG_DIR}"

export SLLM_HEAD_ENV="${PYTHON_ENV}"
export SLLM_WORKER_ENV="${PYTHON_ENV}"
export SLLM_STORE_ENV="${PYTHON_ENV}"
export SLLM_DIRECT_PATH_MODE="${SLLM_DIRECT_PATH_MODE:-1}"
export SLLM_WORKER_GPUS="${WORKER_GPUS}"
export SLLM_HEAD_SESSION="sllm_head_${STACK_SUFFIX}"
export SLLM_STORE_SESSION="sllm_store_${STACK_SUFFIX}"
export SLLM_SERVE_SESSION="sllm_serve_${STACK_SUFFIX}"
export SLLM_WORKER_SESSION_PREFIX="sllm_worker_${STACK_SUFFIX}"
export SLLM_SERVE_LOG_PATH="${LOG_DIR}/serve.log"
export SLLM_DEPLOY_CONFIG="${DEPLOY_PATH}"
export SLLM_REQUEST_POLL_INTERVAL_S="${SLLM_REQUEST_POLL_INTERVAL_S:-0.005}"
export VLLM_NO_USAGE_STATS=1

cleanup() {
  local exit_code=$?
  for session in \
    "${SLLM_HEAD_SESSION}" \
    "${SLLM_STORE_SESSION}" \
    "${SLLM_SERVE_SESSION}" \
    "${SLLM_WORKER_SESSION_PREFIX}_0"
  do
    tmux capture-pane -pJt "${session}" -S -4000 \
      >"${LOG_DIR}/${session}.log" 2>/dev/null || true
  done
  if [[ "${AUTO_STOP}" == "1" ]]; then
    bash "${ROOT_DIR}/scripts/stop_serverlessllm_stack.sh" \
      >"${LOG_DIR}/stop.log" 2>&1 || true
  fi
  exit "${exit_code}"
}
trap cleanup EXIT

echo "[1/7] Generate base-model-only ServerlessLLM deployment"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/generate_serverlessllm_deploy_config.py" \
  --main-repo /home/qhq/serverless_llm_experiment_retry14_baseline \
  --config "${CONFIG_PATH}" \
  --model-profile "${MODEL_PROFILE}" \
  --workload-profile "${WORKLOAD_PROFILE}" \
  --backend vllm \
  --base-model-only \
  --serving-model-name "${SERVING_MODEL_NAME}" \
  --available-worker-gpus "${WORKER_GPUS}" \
  --min-instances "${MIN_INSTANCES}" \
  --max-instances "${MAX_INSTANCES}" \
  --target "${TARGET_CONCURRENCY}" \
  --keep-alive "${KEEP_ALIVE_S}" \
  --output "${DEPLOY_PATH}"

readarray -t VLLM_ENV_LINES < <(
  "${PYTHON_BIN}" - "${DEPLOY_PATH}" <<'PY'
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
for key, value in (cfg.get("backend_config", {}).get("vllm_runtime_env", {}) or {}).items():
    if value not in (None, ""):
        print(f"{key}={value}")
PY
)
for line in "${VLLM_ENV_LINES[@]}"; do
  export "${line}"
done

echo "[2/7] Record immutable inputs"
"${PYTHON_BIN}" - \
  "${MANIFEST_PATH}" "${TRACE_PATH}" "${DEPLOY_PATH}" "${CONFIG_PATH}" \
  "${ROOT_DIR}" "${RELAY_ROOT}" "${RUN_TAG}" "${MODEL_KEY}" "${MAX_REQUESTS}" \
  "${PYTHON_ENV}" "${WORKER_GPUS}" "${SLLM_REQUEST_POLL_INTERVAL_S}" \
  "${TRACE_ROLE}" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    manifest_path,
    trace_path,
    deploy_path,
    config_path,
    root,
    relay_root,
    run_tag,
    model_key,
    max_requests,
    python_env,
    worker_gpus,
    request_poll_interval_s,
    trace_role,
) = sys.argv[1:]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git_head(path):
    return subprocess.check_output(
        ["git", "-C", path, "rev-parse", "HEAD"], text=True
    ).strip()

payload = {
    "schema": "relayserve_external_serverlessllm_run_v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_tag": run_tag,
    "system": "ServerlessLLM",
    "model_key": model_key,
    "max_requests": int(max_requests),
    "trace_path": trace_path,
    "trace_sha256": sha(trace_path),
    "trace_role": trace_role,
    "deploy_path": deploy_path,
    "deploy_sha256": sha(deploy_path),
    "config_path": config_path,
    "config_sha256": sha(config_path),
    "serverlessllm_repo": str(Path(root) / "repos/ServerlessLLM"),
    "serverlessllm_git_commit": git_head(str(Path(root) / "repos/ServerlessLLM")),
    "baseline_harness_git_commit": git_head(root),
    "relayserve_git_commit": git_head(relay_root),
    "runtime_env": python_env,
    "worker_gpus": worker_gpus,
    "request_poll_interval_s": float(request_poll_interval_s),
    "comparison_contract": {
        "workload": f"RelayServe frozen continuation trace ({trace_role}), rate=1.00x",
        "slo_profile": "paper_nominal",
        "gpu_budget": 4,
        "base_model_only": True,
        "lora_enabled": False,
        "cost_model": "RelayServe lifecycle monetary model",
    },
}
Path(manifest_path).write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "[3/7] Start official ServerlessLLM control plane and workers"
bash "${ROOT_DIR}/scripts/start_serverlessllm_stack.sh" \
  >"${LOG_DIR}/stack_start.log" 2>&1

echo "[4/7] Deploy model"
PREDEPLOY_START_NS="$(date +%s%N)"
bash "${ROOT_DIR}/scripts/deploy_serverlessllm_model.sh" "${DEPLOY_PATH}" \
  >"${LOG_DIR}/deploy.log" 2>&1

echo "[4.5/7] Wait for the initial runtime to become ready"
READY_DEADLINE=$((SECONDS + READY_TIMEOUT_S))
READY_PATTERN="Instance .* is ready for model ${SERVING_MODEL_NAME}"
while ! rg -q "${READY_PATTERN}" "${SLLM_SERVE_LOG_PATH}" 2>/dev/null; do
  if (( SECONDS >= READY_DEADLINE )); then
    echo "timed out waiting for initial ServerlessLLM runtime readiness" >&2
    exit 70
  fi
  sleep 1
done
PREDEPLOY_END_NS="$(date +%s%N)"
PREDEPLOY_STARTUP_SEC="$(
  "${PYTHON_BIN}" - "${PREDEPLOY_START_NS}" "${PREDEPLOY_END_NS}" <<'PY'
import sys
print(f"{(int(sys.argv[2]) - int(sys.argv[1])) / 1_000_000_000.0:.6f}")
PY
)"
echo "initial_runtime_startup_sec=${PREDEPLOY_STARTUP_SEC}"

echo "[5/7] Replay frozen continuation workload"
REPLAY_ARGS=()
if [[ "${MAX_REQUESTS}" != "0" ]]; then
  REPLAY_ARGS+=(--max-requests "${MAX_REQUESTS}")
fi
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/replay_openai_trace.py" \
  --trace "${TRACE_PATH}" \
  --base-url http://127.0.0.1:8343 \
  --model-override "${SERVING_MODEL_NAME}" \
  --convert-chat-to-prompt \
  --prompt-guard-tokenizer-model "${TOKENIZER_PATH}" \
  --prompt-guard-max-model-len 3072 \
  --sleep-scale 1.0 \
  --timeout-s "${TIMEOUT_S}" \
  --empty-success-retries 2 \
  --empty-success-retry-delay-s 1.0 \
  --include-stream-usage \
  --require-server-metrics \
  --ttft-slo-ms "${TTFT_SLO_MS}" \
  --generation-seed 42 \
  --label "${RUN_TAG}" \
  --output "${REPLAY_PATH}" \
  "${REPLAY_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/replay.log"

EXPECTED_TOTAL="$(
  "${PYTHON_BIN}" - "${TRACE_PATH}" "${MAX_REQUESTS}" <<'PY'
import json
import sys
from pathlib import Path

total = len(json.loads(Path(sys.argv[1]).read_text())["requests"])
limit = int(sys.argv[2])
print(min(total, limit) if limit > 0 else total)
PY
)"

echo "[6/7] Validate and summarize"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/validate_replay_results.py" \
  --system ServerlessLLM \
  --replay "${REPLAY_PATH}" \
  --expected-total "${EXPECTED_TOTAL}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/summarize_serverlessllm_replay.py" \
  --main-repo /home/qhq/serverless_llm_experiment_retry14_baseline \
  --config "${CONFIG_PATH}" \
  --model-profile "${MODEL_PROFILE}" \
  --dataset-profile relayserve_formal4000 \
  --workload-profile "${WORKLOAD_PROFILE}" \
  --trace "${TRACE_PATH}" \
  --replay "${REPLAY_PATH}" \
  --deploy "${DEPLOY_PATH}" \
  --predeploy-startup-sec "${PREDEPLOY_STARTUP_SEC}" \
  --scenario-name "serverlessllm_relayserve_continuation" \
  --baseline-type serverlessllm \
  --backend-label serverlessllm_official_vllm \
  --system-name ServerlessLLM \
  --output "${SUMMARY_PATH}"

echo "[7/7] Finalize manifest hashes"
"${PYTHON_BIN}" - \
  "${MANIFEST_PATH}" "${REPLAY_PATH}" "${SUMMARY_PATH}" \
  "${PREDEPLOY_STARTUP_SEC}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, replay_path, summary_path = map(Path, sys.argv[1:4])
payload = json.loads(manifest_path.read_text())
for name, path in (("raw_records", replay_path), ("source_summary", summary_path)):
    payload[f"{name}_path"] = str(path)
    payload[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
payload["initial_runtime_startup_sec"] = float(sys.argv[4])
manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "run_dir=${RUN_DIR}"
echo "raw_records_path=${REPLAY_PATH}"
echo "source_summary_path=${SUMMARY_PATH}"
