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

# TC qualification never takes the historical unbounded fallback. The auxiliary
# scope covers supervisor + watcher (and the later external replay), while the
# actual runner/descendants enter a separately verified service scope.
if [[ "${FAASLORA_TC_QUALIFICATION:-0}" == "1" ]]; then
  : "${FAASLORA_TC_LAUNCH_OUTPUT:?TC qualification requires a unique absolute receipt path}"
  if [[ "${FAASLORA_DISABLE_SYSTEMD_SCOPE:-0}" == "1" ]]; then
    echo "[ERROR] TC qualification cannot disable resource containment" >&2
    exit 1
  fi
  TC_AUX_UNIT="primelora-tc-aux-$(/usr/bin/python3 -c 'import uuid; print(uuid.uuid4().hex)').scope"
  exec systemd-run --user --scope --collect --unit="$TC_AUX_UNIT" \
    -p MemoryHigh=3G -p MemoryMax=4G -p MemorySwapMax=0 \
    taskset -c 2,3,26,27 /usr/bin/python3 "$ROOT_DIR/scripts/ieee_tc_preflight.py" \
    gated-launch --output "$FAASLORA_TC_LAUNCH_OUTPUT" \
    --predicted-growth-gib "${FAASLORA_TC_PREDICTED_GROWTH_GIB:-0}" \
    --exec "$PYTHON_BIN" "$SCRIPT_PATH" "$@"
fi

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
