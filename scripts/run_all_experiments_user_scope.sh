#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${FAASLORA_PYTHON:-/home/qhq/anaconda3/envs/LLM_vllm0102/bin/python}"
SCRIPT_PATH="$ROOT_DIR/scripts/run_all_experiments.py"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "[ERROR] Runner not found: $SCRIPT_PATH" >&2
  exit 1
fi

cd "$ROOT_DIR"

SYSTEMD_ENV_ARGS=()
for name in $(compgen -e); do
  case "$name" in
    FAASLORA_*|CUDA_VISIBLE_DEVICES|VLLM_*|PYTHONUNBUFFERED)
      SYSTEMD_ENV_ARGS+=(--setenv="$name=${!name}")
      ;;
  esac
done

exec systemd-run --user --scope --collect \
  "${SYSTEMD_ENV_ARGS[@]}" \
  "$PYTHON_BIN" "$SCRIPT_PATH" "$@"
