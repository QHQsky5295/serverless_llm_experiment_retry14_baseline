#!/usr/bin/env bash
set -euo pipefail

# Stage LoRA adapter directories onto a remote artifact node.
#
# This helper deliberately does not accept or store passwords.  Use interactive
# SSH password input, SSH keys, or an external secret manager.  It is separate
# from the formal experiment harness and does not run inference.

usage() {
  cat >&2 <<'USAGE'
Usage:
  scripts/stage_remote_artifacts.sh \
    --source artifacts/frozen/llama32_3b_a500_v1_modelscope \
    --remote-user lab14 \
    --remote-host 10.199.227.174 \
    --remote-port 8122 \
    --remote-dir /data/primelora_remote_artifacts/llama32_3b_a500_v1_modelscope \
    [--adapter-list path/to/adapter_ids.txt]

If --adapter-list is omitted, all first-level adapter directories are staged.
USAGE
}

SOURCE=""
REMOTE_USER=""
REMOTE_HOST=""
REMOTE_PORT="22"
REMOTE_DIR=""
ADAPTER_LIST=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --remote-user) REMOTE_USER="$2"; shift 2 ;;
    --remote-host) REMOTE_HOST="$2"; shift 2 ;;
    --remote-port) REMOTE_PORT="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --adapter-list) ADAPTER_LIST="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${SOURCE}" || -z "${REMOTE_USER}" || -z "${REMOTE_HOST}" || -z "${REMOTE_DIR}" ]]; then
  usage
  exit 2
fi

if [[ ! -d "${SOURCE}" ]]; then
  echo "source directory not found: ${SOURCE}" >&2
  exit 1
fi

ssh_cmd=(ssh -p "${REMOTE_PORT}" -o StrictHostKeyChecking=accept-new)
# -L dereferences adapter support-file symlinks.  The frozen pools often store
# model config/tokenizer support files as absolute symlinks into the local model
# cache; preserving those links on a separate remote node would make the served
# artifact non-portable.
rsync_cmd=(rsync -aL --info=progress2 -e "${ssh_cmd[*]}")

"${ssh_cmd[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_DIR}'"

if [[ -n "${ADAPTER_LIST}" ]]; then
  if [[ ! -f "${ADAPTER_LIST}" ]]; then
    echo "adapter list not found: ${ADAPTER_LIST}" >&2
    exit 1
  fi
  tmp_include="$(mktemp)"
  trap 'rm -f "${tmp_include}"' EXIT
  while IFS= read -r adapter_id; do
    [[ -z "${adapter_id}" || "${adapter_id}" =~ ^# ]] && continue
    printf '+ /%s/***\n' "${adapter_id}" >> "${tmp_include}"
  done < "${ADAPTER_LIST}"
  printf '- *\n' >> "${tmp_include}"
  "${rsync_cmd[@]}" --include-from="${tmp_include}" "${SOURCE}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
else
  "${rsync_cmd[@]}" "${SOURCE}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
fi

echo "remote staging complete: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
