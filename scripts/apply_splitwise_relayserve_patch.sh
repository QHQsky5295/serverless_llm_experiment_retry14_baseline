#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SPLITWISE_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
PROJECT_ROOT="${SPLITWISE_PROJECT_ROOT:-${ROOT_DIR}/workspaces/splitwise_official_20260615}"
PATCH_PATH="${ROOT_DIR}/patches/splitwise_relayserve_compat.patch"
EXPECTED_COMMIT="8f99e7dc9b407f4ce2488d03dd44c0b8b946dab0"

for required in "${PROJECT_ROOT}/.git" "${PATCH_PATH}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing Splitwise compatibility input: ${required}" >&2
    exit 2
  fi
done

actual_commit="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo \
    "unexpected Splitwise commit: ${actual_commit}; expected ${EXPECTED_COMMIT}" \
    >&2
  exit 3
fi

if git -C "${PROJECT_ROOT}" apply --reverse --check "${PATCH_PATH}" \
  >/dev/null 2>&1
then
  echo "Splitwise compatibility patch already applied"
elif git -C "${PROJECT_ROOT}" apply --check "${PATCH_PATH}"; then
  git -C "${PROJECT_ROOT}" apply "${PATCH_PATH}"
  echo "Applied Splitwise compatibility patch"
else
  echo "Splitwise compatibility patch does not apply cleanly" >&2
  exit 4
fi
