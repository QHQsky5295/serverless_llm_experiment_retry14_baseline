#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-facebook/opt-125m}"
SERVER_URL="${LLM_SERVER_URL:-http://127.0.0.1:8343}"
LORA_ADAPTER_NAME="${LORA_ADAPTER_NAME:-${2:-}}"

if [[ -n "${LORA_ADAPTER_NAME}" ]]; then
  LORA_FIELD=",\n    \"lora_adapter_name\": \"${LORA_ADAPTER_NAME}\""
else
  LORA_FIELD=""
fi

exec curl -s "${SERVER_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"What is ServerlessLLM?\"}
    ]${LORA_FIELD}
  }"
