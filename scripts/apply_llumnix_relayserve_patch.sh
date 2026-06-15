#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${LLUMNIX_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
PROJECT_ROOT="${LLUMNIX_PROJECT_ROOT:-${ROOT_DIR}/workspaces/llumnix_official_20260615}"
PATCH_PATH="${ROOT_DIR}/patches/llumnix_relayserve_compat.patch"

if git -C "${PROJECT_ROOT}" apply --check "${PATCH_PATH}" >/dev/null 2>&1; then
  git -C "${PROJECT_ROOT}" apply "${PATCH_PATH}"
  echo "Applied Llumnix RelayServe host compatibility patch."
elif git -C "${PROJECT_ROOT}" apply --reverse --check "${PATCH_PATH}" >/dev/null 2>&1; then
  echo "Llumnix RelayServe host compatibility patch is already applied."
else
  echo "Llumnix compatibility patch does not match the frozen source checkout." >&2
  exit 1
fi
