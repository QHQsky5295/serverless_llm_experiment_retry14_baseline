#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
MAIN_REPO="${SLLM_MAIN_REPO:-/home/qhq/serverless_llm_experiment_retry14_baseline}"
CONFIG_PATH="${SLLM_CONFIG_PATH:-${MAIN_REPO}/configs/experiments.yaml}"
SANITIZED_POOL_ROOT="${SLLM_SANITIZED_POOL_ROOT:-${ROOT_DIR}/artifacts/frozen_sanitized}"
MODEL_PROFILE="${SLLM_MODEL_PROFILE:?SLLM_MODEL_PROFILE is required}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] missing config: ${CONFIG_PATH}" >&2
  exit 1
fi

SANITIZED_REMOTE_DIR="$(
  python3 - "${CONFIG_PATH}" "${MAIN_REPO}" "${MODEL_PROFILE}" "${SANITIZED_POOL_ROOT}" <<'PY'
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1]).resolve()
main_repo = Path(sys.argv[2]).resolve()
profile_name = sys.argv[3]
sanitized_root = Path(sys.argv[4]).resolve()
cfg = yaml.safe_load(cfg_path.read_text()) or {}
profile = dict((cfg.get("model_profiles", {}) or {}).get(profile_name) or {})
if not profile:
    raise SystemExit(f"unknown model profile: {profile_name}")
storage_cfg = dict(cfg.get("storage", {}) or {})
for bucket in (profile,):
    storage_cfg.update(bucket.get("storage", {}) or {})
remote_dir = Path(storage_cfg.get("remote_dir", "artifacts/remote"))
if not remote_dir.is_absolute():
    remote_dir = (main_repo / remote_dir).resolve()
else:
    remote_dir = remote_dir.resolve()
print((sanitized_root / remote_dir.name).resolve())
PY
)"

if [[ ! -d "${SANITIZED_REMOTE_DIR}" ]]; then
  echo "[ERROR] missing sanitized pool for ${MODEL_PROFILE}: ${SANITIZED_REMOTE_DIR}" >&2
  exit 1
fi

echo "[sanitized] Sampling this round from sanitized frozen pool"
echo "  model_profile  : ${MODEL_PROFILE}"
echo "  sanitized_pool : ${SANITIZED_REMOTE_DIR}"

SLLM_STORAGE_REMOTE_DIR_OVERRIDE="${SANITIZED_REMOTE_DIR}" \
bash "${ROOT_DIR}/scripts/prepare_shared_round_artifacts.sh"
