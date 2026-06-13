#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="${SLINFER_MODEL_KEY:-3b}"
STACK_PREFIX="${SLINFER_STACK_PREFIX:-slinfer_${MODEL_KEY}}"

for session in \
  "${STACK_PREFIX}_gateway" \
  "${STACK_PREFIX}_gpu0" \
  "${STACK_PREFIX}_gpu1" \
  "${STACK_PREFIX}_gpu2" \
  "${STACK_PREFIX}_gpu3" \
  "${STACK_PREFIX}_store"
do
  tmux kill-session -t "${session}" 2>/dev/null || true
done

sleep 2
pkill -f "SLINFER_core/tools/vllm_backend_starter.py" 2>/dev/null || true
pkill -f "SLINFER_core/scheduler/gateway.py" 2>/dev/null || true
pkill -f "SLINFER_main_20260323.*vllm.entrypoints.openai.api_server" 2>/dev/null || true

echo "Stopped isolated SLINFER stack: ${STACK_PREFIX}."
