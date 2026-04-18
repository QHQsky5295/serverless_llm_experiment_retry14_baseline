#!/usr/bin/env bash
set -euo pipefail

HEAD_SESSION="${SLLM_HEAD_SESSION:-sllm_head_formal}"
STORE_SESSION="${SLLM_STORE_SESSION:-sllm_store_formal}"
SERVE_SESSION="${SLLM_SERVE_SESSION:-sllm_serve_formal}"
WORKER_SESSION_PREFIX="${SLLM_WORKER_SESSION_PREFIX:-sllm_worker_formal}"
WORKER_GPUS="${SLLM_WORKER_GPUS:-0,1,2,3}"
HEAD_ENV="${SLLM_HEAD_ENV:-sllm_head_official}"
WORKER_ENV="${SLLM_WORKER_ENV:-sllm_worker_official}"

IFS=',' read -r -a GPU_LIST <<< "${WORKER_GPUS}"

if tmux ls >/dev/null 2>&1; then
  while IFS= read -r session_name; do
    [[ -z "${session_name}" ]] && continue
    tmux kill-session -t "${session_name}" 2>/dev/null || true
  done < <(tmux ls 2>/dev/null | cut -d: -f1 | grep -E '^sllm_(head|store|serve|worker)_' || true)
fi

for session in "${SERVE_SESSION}" "${STORE_SESSION}" "${HEAD_SESSION}"; do
  tmux kill-session -t "${session}" 2>/dev/null || true
done

for idx in "${!GPU_LIST[@]}"; do
  tmux kill-session -t "${WORKER_SESSION_PREFIX}_${idx}" 2>/dev/null || true
done

# Tear down any lingering Ray processes that may survive outside tmux and
# otherwise leak cluster state across supposedly isolated baseline runs.
env PYTHONNOUSERSITE=1 conda run -n "${HEAD_ENV}" ray stop --force >/dev/null 2>&1 || true
env PYTHONNOUSERSITE=1 conda run -n "${WORKER_ENV}" ray stop --force >/dev/null 2>&1 || true

# Best-effort cleanup for detached store/serve processes that can survive the
# tmux session if a child process re-parented before teardown.
pkill -f "sllm-store start --storage-path /home/qhq/serverless_llm_baselines/models" 2>/dev/null || true
pkill -f "uvicorn app.main:app --host 0.0.0.0 --port 8080" 2>/dev/null || true
pkill -f "sllm.cli.clic start --host 127.0.0.1 --port 8343" 2>/dev/null || true

# Clear legacy temp logs from the old non-namespaced launcher path so they do
# not get mistaken for the active run's failure evidence.
rm -f /tmp/serverlessllm_serve_formal.log /tmp/serverlessllm_store_smoke.log 2>/dev/null || true

echo "Stopped isolated ServerlessLLM stack."
