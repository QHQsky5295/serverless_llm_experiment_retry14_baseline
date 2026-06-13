#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SLINFER_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
PROJECT_BASE="${SLINFER_PROJECT_BASE:-${ROOT_DIR}/vendor_new_baselines/SLINFER_main_20260323}"
PATCH_PATH="${ROOT_DIR}/patches/slinfer_relayserve_compat.patch"

if git -C "${PROJECT_BASE}" apply --reverse --check "${PATCH_PATH}" >/dev/null 2>&1; then
  echo "SLINFER RelayServe compatibility patch is already applied."
  exit 0
fi

git -C "${PROJECT_BASE}" apply --check "${PATCH_PATH}"
git -C "${PROJECT_BASE}" apply "${PATCH_PATH}"
echo "Applied SLINFER RelayServe compatibility patch."
