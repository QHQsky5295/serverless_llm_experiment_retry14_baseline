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

can_use_systemd_user_scope() {
  if [[ "${FAASLORA_DISABLE_SYSTEMD_SCOPE:-0}" == "1" ]]; then
    return 1
  fi
  if ! command -v systemd-run >/dev/null 2>&1; then
    return 1
  fi
  if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" && -z "${XDG_RUNTIME_DIR:-}" ]]; then
    return 1
  fi
  if ! systemd-run --user --scope --collect /bin/true >/dev/null 2>&1; then
    return 1
  fi
  return 0
}

if can_use_systemd_user_scope; then
  exec systemd-run --user --scope --collect \
    "${SYSTEMD_ENV_ARGS[@]}" \
    "$PYTHON_BIN" "$SCRIPT_PATH" "$@"
fi

echo "[warn] systemd user scope unavailable; running without systemd-run --user --scope" >&2
exec "$PYTHON_BIN" "$SCRIPT_PATH" "$@"
