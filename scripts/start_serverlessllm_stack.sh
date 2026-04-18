#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
SCRIPTS_DIR="${ROOT_DIR}/scripts"

HEAD_SESSION="${SLLM_HEAD_SESSION:-sllm_head_formal}"
STORE_SESSION="${SLLM_STORE_SESSION:-sllm_store_formal}"
SERVE_SESSION="${SLLM_SERVE_SESSION:-sllm_serve_formal}"
WORKER_SESSION_PREFIX="${SLLM_WORKER_SESSION_PREFIX:-sllm_worker_formal}"
SLLM_SINGLE_HOST_MULTI_GPU="${SLLM_SINGLE_HOST_MULTI_GPU:-1}"
HEAD_ENV="${SLLM_HEAD_ENV:-sllm_head_official}"
WORKER_ENV="${SLLM_WORKER_ENV:-sllm_worker_official}"
STORE_ENV="${SLLM_STORE_ENV:-${WORKER_ENV}}"
DIRECT_PATH_MODE="${SLLM_DIRECT_PATH_MODE:-0}"

RAY_PORT="${SLLM_RAY_PORT:-6389}"
SLLM_HOST="${SLLM_HOST:-127.0.0.1}"
SLLM_PORT="${SLLM_PORT:-8343}"
WORKER_GPUS="${SLLM_WORKER_GPUS:-0,1,2,3}"
STORE_READY_PATTERN="${SLLM_STORE_READY_PATTERN:-Starting gRPC server on 0\\.0\\.0\\.0:8073}"
SERVE_LOG_PATH="${SLLM_SERVE_LOG_PATH:-/tmp/serverlessllm_serve_formal.log}"

detect_ray_head_host() {
  if [[ -n "${SLLM_RAY_HEAD_HOST:-}" ]]; then
    printf '%s\n' "${SLLM_RAY_HEAD_HOST}"
    return 0
  fi
  if command -v hostname >/dev/null 2>&1; then
    local first_ipv4
    first_ipv4="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' | grep -v '^127\\.' | head -n 1 || true)"
    if [[ -n "${first_ipv4}" ]]; then
      printf '%s\n' "${first_ipv4}"
      return 0
    fi
  fi
  printf '127.0.0.1\n'
}

RAY_HEAD_HOST="$(detect_ray_head_host)"
RAY_ADDRESS="${RAY_HEAD_HOST}:${RAY_PORT}"

IFS=',' read -r -a GPU_LIST <<< "${WORKER_GPUS}"
GPU_COUNT="${#GPU_LIST[@]}"
if (( GPU_COUNT <= 0 )); then
  echo "SLLM_WORKER_GPUS must contain at least one GPU id" >&2
  exit 1
fi

if [[ "${SLLM_SINGLE_HOST_MULTI_GPU}" == "1" ]]; then
  EXPECTED_WORKERS=1
else
  EXPECTED_WORKERS="${GPU_COUNT}"
fi

pane_contains() {
  local session_name="$1"
  local pattern="$2"
  local pane_text
  pane_text="$(tmux capture-pane -pJt "${session_name}" -S -160 2>/dev/null || true)"
  if [[ -z "${pane_text}" ]]; then
    return 1
  fi
  if command -v rg >/dev/null 2>&1; then
    printf '%s\n' "${pane_text}" | rg -q "${pattern}"
  else
    printf '%s\n' "${pane_text}" | grep -Eq "${pattern}"
  fi
}

wait_for_head() {
  local attempt=0
  while (( attempt < 30 )); do
    if tmux has-session -t "${HEAD_SESSION}" 2>/dev/null && \
      pane_contains "${HEAD_SESSION}" "Ray runtime started\\."; then
      return 0
    fi
    sleep 2
    attempt=$((attempt + 1))
  done
  echo "Timed out waiting for Ray head to become reachable" >&2
  return 1
}

wait_for_workers() {
  local expected="$1"
  local attempt=0
  while (( attempt < 60 )); do
    local ready=0
    for idx in "${!GPU_LIST[@]}"; do
      local session_name="${WORKER_SESSION_PREFIX}_${idx}"
      if tmux has-session -t "${session_name}" 2>/dev/null && \
        pane_contains "${session_name}" "Ray runtime started\\."; then
        ready=$((ready + 1))
      fi
    done
    if (( ready >= expected )); then
      return 0
    fi
    sleep 2
    attempt=$((attempt + 1))
  done
  echo "Timed out waiting for ${expected} worker nodes to register" >&2
  return 1
}

wait_for_store() {
  local attempt=0
  while (( attempt < 60 )); do
    if tmux has-session -t "${STORE_SESSION}" 2>/dev/null && \
      pane_contains "${STORE_SESSION}" "${STORE_READY_PATTERN}"; then
      return 0
    fi
    sleep 2
    attempt=$((attempt + 1))
  done
  echo "Timed out waiting for store service to become ready" >&2
  echo "===== ${STORE_SESSION} =====" >&2
  tmux capture-pane -pJt "${STORE_SESSION}" -S -200 2>/dev/null | tail -n 120 >&2 || true
  return 1
}

wait_for_serve() {
  local attempt=0
  while (( attempt < 120 )); do
    if ! tmux has-session -t "${SERVE_SESSION}" 2>/dev/null; then
      if (( attempt >= 2 )); then
        echo "ServerlessLLM serve session exited before the API became ready" >&2
        if [[ -f "${SERVE_LOG_PATH}" ]]; then
          echo "===== ${SERVE_LOG_PATH} =====" >&2
          tail -n 120 "${SERVE_LOG_PATH}" >&2 || true
        fi
        return 1
      fi
    fi
    if curl --noproxy '*' -fsS "http://${SLLM_HOST}:${SLLM_PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
    attempt=$((attempt + 1))
  done
  echo "Timed out waiting for ServerlessLLM API" >&2
  for session_name in "${HEAD_SESSION}" "${WORKER_SESSION_PREFIX}_0" "${STORE_SESSION}" "${SERVE_SESSION}"; do
    echo "===== ${session_name} =====" >&2
    tmux capture-pane -pJt "${session_name}" -S -200 2>/dev/null | tail -n 120 >&2 || true
  done
  return 1
}

bash "${SCRIPTS_DIR}/sync_serverlessllm_runtime_sources.sh"
bash "${SCRIPTS_DIR}/stop_serverlessllm_stack.sh" >/dev/null 2>&1 || true

tmux kill-session -t "${HEAD_SESSION}" 2>/dev/null || true
tmux kill-session -t "${STORE_SESSION}" 2>/dev/null || true
tmux kill-session -t "${SERVE_SESSION}" 2>/dev/null || true
for idx in "${!GPU_LIST[@]}"; do
  tmux kill-session -t "${WORKER_SESSION_PREFIX}_${idx}" 2>/dev/null || true
done

tmux new-session -d -s "${HEAD_SESSION}" -c "${ROOT_DIR}" \
  "env SLLM_HEAD_ENV=${HEAD_ENV} SLLM_DIRECT_PATH_MODE=${DIRECT_PATH_MODE} SLLM_RAY_HEAD_HOST=${RAY_HEAD_HOST} SLLM_RAY_NODE_IP=${RAY_HEAD_HOST} SLLM_RAY_ADDRESS=${RAY_ADDRESS} bash ${SCRIPTS_DIR}/run_serverlessllm_head.sh"
wait_for_head

if [[ "${SLLM_SINGLE_HOST_MULTI_GPU}" == "1" ]]; then
  tmux new-session -d -s "${WORKER_SESSION_PREFIX}_0" -c "${ROOT_DIR}" \
    "env SLLM_WORKER_ENV=${WORKER_ENV} SLLM_DIRECT_PATH_MODE=${DIRECT_PATH_MODE} SLLM_RAY_HEAD_HOST=${RAY_HEAD_HOST} SLLM_RAY_NODE_IP=${RAY_HEAD_HOST} SLLM_RAY_ADDRESS=${RAY_ADDRESS} CUDA_VISIBLE_DEVICES=${WORKER_GPUS} SLLM_WORKER_ID=0 SLLM_WORKER_NUM_GPUS=${GPU_COUNT} SLLM_WORKER_CPUS=$((GPU_COUNT * 4)) bash ${SCRIPTS_DIR}/run_serverlessllm_worker.sh"
else
  for idx in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[$idx]}"
    tmux new-session -d -s "${WORKER_SESSION_PREFIX}_${idx}" -c "${ROOT_DIR}" \
      "env SLLM_WORKER_ENV=${WORKER_ENV} SLLM_DIRECT_PATH_MODE=${DIRECT_PATH_MODE} SLLM_RAY_HEAD_HOST=${RAY_HEAD_HOST} SLLM_RAY_NODE_IP=${RAY_HEAD_HOST} SLLM_RAY_ADDRESS=${RAY_ADDRESS} CUDA_VISIBLE_DEVICES=${gpu} SLLM_WORKER_ID=${idx} bash ${SCRIPTS_DIR}/run_serverlessllm_worker.sh"
  done
fi
wait_for_workers "${EXPECTED_WORKERS}"

if [[ "${DIRECT_PATH_MODE}" != "1" ]]; then
  tmux new-session -d -s "${STORE_SESSION}" -c "${ROOT_DIR}" \
    "env SLLM_STORE_ENV=${STORE_ENV} SLLM_DIRECT_PATH_MODE=${DIRECT_PATH_MODE} CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} bash ${SCRIPTS_DIR}/run_serverlessllm_store.sh"
  wait_for_store
fi

rm -f "${SERVE_LOG_PATH}"
tmux new-session -d -s "${SERVE_SESSION}" -c "${ROOT_DIR}" \
  "bash -lc 'env SLLM_HEAD_ENV=${HEAD_ENV} SLLM_DIRECT_PATH_MODE=${DIRECT_PATH_MODE} SLLM_RAY_HEAD_HOST=${RAY_HEAD_HOST} SLLM_RAY_NODE_IP=${RAY_HEAD_HOST} SLLM_RAY_ADDRESS=${RAY_ADDRESS} bash ${SCRIPTS_DIR}/run_serverlessllm_serve.sh 2>&1 | tee ${SERVE_LOG_PATH}'"
wait_for_serve

echo "ServerlessLLM stack is ready."
echo "  head   : ${HEAD_SESSION}"
if [[ "${SLLM_SINGLE_HOST_MULTI_GPU}" == "1" ]]; then
  echo "  worker : ${WORKER_SESSION_PREFIX}_0 (gpus=${WORKER_GPUS})"
else
  for idx in "${!GPU_LIST[@]}"; do
    echo "  worker : ${WORKER_SESSION_PREFIX}_${idx} (gpu=${GPU_LIST[$idx]})"
  done
fi
if [[ "${DIRECT_PATH_MODE}" != "1" ]]; then
  echo "  store  : ${STORE_SESSION} (gpu=${GPU_LIST[0]})"
else
  echo "  store  : skipped (direct-path mode)"
fi
echo "  serve  : ${SERVE_SESSION}"
echo "  api    : http://${SLLM_HOST}:${SLLM_PORT}/v1/models"
