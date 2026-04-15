#!/usr/bin/env bash
set -euo pipefail

HEAD_SESSION="${SLLM_HEAD_SESSION:-sllm_head_smoke}"
WORKER_SESSION="${SLLM_WORKER_SESSION:-sllm_worker_smoke}"
STORE_SESSION="${SLLM_STORE_SESSION:-sllm_store_smoke}"
SERVE_SESSION="${SLLM_SERVE_SESSION:-sllm_serve_smoke}"

for session in "${SERVE_SESSION}" "${STORE_SESSION}" "${WORKER_SESSION}" "${HEAD_SESSION}"; do
  tmux kill-session -t "${session}" 2>/dev/null || true
done

pkill -f 'conda run -n sllm_head_official ray start --head --port=6389' 2>/dev/null || true
pkill -f 'conda run -n sllm_worker_official ray start --address=127.0.0.1:6389' 2>/dev/null || true
pkill -f 'conda run -n sllm_worker_official sllm-store start --storage-path /home/qhq/serverless_llm_baselines/models' 2>/dev/null || true
pkill -f 'conda run -n sllm_head_official sllm start --host 127.0.0.1 --port 8343' 2>/dev/null || true
pkill -f 'conda run -n sllm_head_official env LLM_SERVER_URL=http://127.0.0.1:8343 sllm deploy' 2>/dev/null || true

sleep 2

pkill -9 -f 'conda run -n sllm_head_official ray start --head --port=6389' 2>/dev/null || true
pkill -9 -f 'conda run -n sllm_worker_official ray start --address=127.0.0.1:6389' 2>/dev/null || true
pkill -9 -f 'conda run -n sllm_worker_official sllm-store start --storage-path /home/qhq/serverless_llm_baselines/models' 2>/dev/null || true
pkill -9 -f 'conda run -n sllm_head_official sllm start --host 127.0.0.1 --port 8343' 2>/dev/null || true
pkill -9 -f 'conda run -n sllm_head_official env LLM_SERVER_URL=http://127.0.0.1:8343 sllm deploy' 2>/dev/null || true

echo "Stopped isolated ServerlessLLM smoke stack."
