#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${DISTSERVE_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
PROJECT_ROOT="${DISTSERVE_PROJECT_ROOT:-${ROOT_DIR}/workspaces/distserve_official_20260615}"
PATCH_PATH="${ROOT_DIR}/patches/distserve_relayserve_compat.patch"

if git -C "${PROJECT_ROOT}" apply --check "${PATCH_PATH}" >/dev/null 2>&1; then
  git -C "${PROJECT_ROOT}" apply "${PATCH_PATH}"
  echo "Applied DistServe RelayServe checkpoint compatibility patch."
elif git -C "${PROJECT_ROOT}" apply --reverse --check "${PATCH_PATH}" >/dev/null 2>&1; then
  echo "DistServe RelayServe checkpoint compatibility patch is already applied."
else
  echo "DistServe compatibility patch does not match the frozen source checkout." >&2
  exit 1
fi
