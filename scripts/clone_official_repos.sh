#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/qhq/serverless_llm_baselines/repos"
mkdir -p "$ROOT"

clone_if_missing() {
  local url="$1"
  local name="$2"
  if [[ -d "$ROOT/$name/.git" ]]; then
    echo "[skip] $name already exists"
  else
    git clone "$url" "$ROOT/$name"
  fi
}

clone_if_missing "https://github.com/ServerlessLLM/ServerlessLLM.git" "ServerlessLLM"
clone_if_missing "https://github.com/S-LoRA/S-LoRA.git" "S-LoRA"
clone_if_missing "https://github.com/skypilot-org/skypilot.git" "skypilot"

echo
echo "Done. Repos are under: $ROOT"
