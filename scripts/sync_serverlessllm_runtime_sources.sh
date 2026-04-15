#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
REPO_ROOT="${SLLM_REPO_ROOT:-${ROOT_DIR}/repos/ServerlessLLM}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
HEAD_ENV="${SLLM_HEAD_ENV:-sllm_head_official}"
WORKER_ENV="${SLLM_WORKER_ENV:-sllm_worker_official}"
WORKER_ENV_ROOT="${SLLM_WORKER_ENV_ROOT:-${CONDA_BASE}/envs/${WORKER_ENV}}"
HEAD_ENV_ROOT="${SLLM_HEAD_ENV_ROOT:-${CONDA_BASE}/envs/${HEAD_ENV}}"

copy_if_needed() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "${src}" ]]; then
    echo "[sync] missing source: ${src}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${dst}")"
  if [[ -f "${dst}" ]] && cmp -s "${src}" "${dst}"; then
    echo "[sync] up-to-date ${dst}"
    return 0
  fi
  install -m 0644 "${src}" "${dst}"
  echo "[sync] updated ${dst}"
}

resolve_site_packages() {
  local env_root="$1"
  local candidate
  candidate="$(find "${env_root}/lib" -maxdepth 2 -type d -name site-packages | head -n 1)"
  if [[ -z "${candidate}" ]]; then
    echo "[sync] unable to resolve site-packages under ${env_root}" >&2
    exit 1
  fi
  printf '%s\n' "${candidate}"
}

WORKER_SITE_PACKAGES="$(resolve_site_packages "${WORKER_ENV_ROOT}")"
HEAD_SITE_PACKAGES="$(resolve_site_packages "${HEAD_ENV_ROOT}")"

copy_if_needed \
  "${REPO_ROOT}/sllm/backends/transformers_backend.py" \
  "${WORKER_SITE_PACKAGES}/sllm/backends/transformers_backend.py"

copy_if_needed \
  "${REPO_ROOT}/sllm/backends/__init__.py" \
  "${WORKER_SITE_PACKAGES}/sllm/backends/__init__.py"

copy_if_needed \
  "${REPO_ROOT}/sllm/backends/vllm_backend.py" \
  "${WORKER_SITE_PACKAGES}/sllm/backends/vllm_backend.py"

copy_if_needed \
  "${REPO_ROOT}/sllm/routers/roundrobin_router.py" \
  "${WORKER_SITE_PACKAGES}/sllm/routers/roundrobin_router.py"

copy_if_needed \
  "${REPO_ROOT}/sllm/controller.py" \
  "${WORKER_SITE_PACKAGES}/sllm/controller.py"

copy_if_needed \
  "${REPO_ROOT}/sllm/store_manager.py" \
  "${WORKER_SITE_PACKAGES}/sllm/store_manager.py"

copy_if_needed \
  "${REPO_ROOT}/sllm/app_lib.py" \
  "${WORKER_SITE_PACKAGES}/sllm/app_lib.py"

copy_if_needed \
  "${REPO_ROOT}/sllm/hardware_info_collector.py" \
  "${WORKER_SITE_PACKAGES}/sllm/hardware_info_collector.py"

copy_if_needed \
  "${REPO_ROOT}/sllm/utils.py" \
  "${WORKER_SITE_PACKAGES}/sllm/utils.py"

copy_if_needed \
  "${REPO_ROOT}/sllm_store/sllm_store/transformers.py" \
  "${WORKER_SITE_PACKAGES}/sllm_store/transformers.py"

copy_if_needed \
  "${REPO_ROOT}/sllm/backends/transformers_backend.py" \
  "${HEAD_SITE_PACKAGES}/sllm/backends/transformers_backend.py"

copy_if_needed \
  "${REPO_ROOT}/sllm/backends/__init__.py" \
  "${HEAD_SITE_PACKAGES}/sllm/backends/__init__.py"

copy_if_needed \
  "${REPO_ROOT}/sllm/backends/vllm_backend.py" \
  "${HEAD_SITE_PACKAGES}/sllm/backends/vllm_backend.py"

copy_if_needed \
  "${REPO_ROOT}/sllm/routers/roundrobin_router.py" \
  "${HEAD_SITE_PACKAGES}/sllm/routers/roundrobin_router.py"

copy_if_needed \
  "${REPO_ROOT}/sllm/controller.py" \
  "${HEAD_SITE_PACKAGES}/sllm/controller.py"

copy_if_needed \
  "${REPO_ROOT}/sllm/store_manager.py" \
  "${HEAD_SITE_PACKAGES}/sllm/store_manager.py"

copy_if_needed \
  "${REPO_ROOT}/sllm/app_lib.py" \
  "${HEAD_SITE_PACKAGES}/sllm/app_lib.py"

copy_if_needed \
  "${REPO_ROOT}/sllm/hardware_info_collector.py" \
  "${HEAD_SITE_PACKAGES}/sllm/hardware_info_collector.py"

copy_if_needed \
  "${REPO_ROOT}/sllm/utils.py" \
  "${HEAD_SITE_PACKAGES}/sllm/utils.py"
