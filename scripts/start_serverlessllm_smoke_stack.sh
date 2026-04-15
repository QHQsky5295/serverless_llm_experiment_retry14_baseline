#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/qhq/serverless_llm_baselines"
SCRIPTS_DIR="${ROOT_DIR}/scripts"

HEAD_SESSION="${SLLM_HEAD_SESSION:-sllm_head_smoke}"
WORKER_SESSION="${SLLM_WORKER_SESSION:-sllm_worker_smoke}"
STORE_SESSION="${SLLM_STORE_SESSION:-sllm_store_smoke}"
SERVE_SESSION="${SLLM_SERVE_SESSION:-sllm_serve_smoke}"

RAY_PORT="${SLLM_RAY_PORT:-6389}"
SLLM_HOST="${SLLM_HOST:-127.0.0.1}"
SLLM_PORT="${SLLM_PORT:-8343}"

wait_for_head() {
  local attempt=0
  while (( attempt < 30 )); do
    if env PYTHONNOUSERSITE=1 conda run -n sllm_head_official \
      python -c "import ray; ray.init(address='auto', ignore_reinit_error=True); print('ok')" \
      >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
    attempt=$((attempt + 1))
  done
  echo "Timed out waiting for Ray head to become reachable" >&2
  return 1
}

wait_for_worker() {
  local attempt=0
  while (( attempt < 45 )); do
    if env PYTHONNOUSERSITE=1 conda run -n sllm_head_official \
      python - <<'PY' >/dev/null 2>&1
import ray
ray.init(address="auto", ignore_reinit_error=True)
for node in ray.nodes():
    resources = node.get("Resources", {})
    if resources.get("control_node", 0) > 0:
        continue
    if any(key.startswith("worker_id_") for key in resources):
        raise SystemExit(0)
raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep 2
    attempt=$((attempt + 1))
  done
  echo "Timed out waiting for worker node registration" >&2
  return 1
}

wait_for_serve() {
  local attempt=0
  while (( attempt < 45 )); do
    if curl -fsS "http://${SLLM_HOST}:${SLLM_PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
    attempt=$((attempt + 1))
  done
  echo "Timed out waiting for ServerlessLLM API" >&2
  return 1
}

for session in "${SERVE_SESSION}" "${STORE_SESSION}" "${WORKER_SESSION}" "${HEAD_SESSION}"; do
  tmux kill-session -t "${session}" 2>/dev/null || true
done

tmux new-session -d -s "${HEAD_SESSION}" -c "${ROOT_DIR}" \
  "bash ${SCRIPTS_DIR}/run_serverlessllm_head.sh"
wait_for_head

tmux new-session -d -s "${WORKER_SESSION}" -c "${ROOT_DIR}" \
  "bash ${SCRIPTS_DIR}/run_serverlessllm_worker.sh"
wait_for_worker

tmux new-session -d -s "${STORE_SESSION}" -c "${ROOT_DIR}" \
  "bash ${SCRIPTS_DIR}/run_serverlessllm_store.sh"

tmux new-session -d -s "${SERVE_SESSION}" -c "${ROOT_DIR}" \
  "bash ${SCRIPTS_DIR}/run_serverlessllm_serve.sh"
wait_for_serve

echo "ServerlessLLM smoke stack is ready."
echo "  head   : ${HEAD_SESSION}"
echo "  worker : ${WORKER_SESSION}"
echo "  store  : ${STORE_SESSION}"
echo "  serve  : ${SERVE_SESSION}"
echo "  api    : http://${SLLM_HOST}:${SLLM_PORT}/v1/models"
