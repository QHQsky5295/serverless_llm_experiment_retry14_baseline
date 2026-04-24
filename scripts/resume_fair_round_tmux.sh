#!/usr/bin/env bash
set -euo pipefail

BASELINES_ROOT="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
ROUND_SECTION="${FAIR_ROUND_SECTION:-03_main_comparison}"
ROUND_ROOT="${FAIR_ROUND_ROOT:-${BASELINES_ROOT}/results/paper_experiments/${ROUND_SECTION}}"
RUNNER="${FAIR_ROUND_RUNNER:-${BASELINES_ROOT}/scripts/run_full_fair_round.sh}"
SESSION_NAME="${FAIR_ROUND_TMUX_SESSION:-}"
ROUND_DIR="${FAIR_ROUND_DIR:-}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  resume_fair_round_tmux.sh [--round-dir DIR] [--round-root DIR] [--session NAME] [--dry-run]

Purpose:
  Resume a fair-comparison round from anywhere. The script automatically:
    1. selects the most advanced unfinished round when --round-dir is omitted,
    2. creates or reuses a tmux session,
    3. sources the selected round.env,
    4. runs run_full_fair_round.sh so completed stages are skipped.

Environment overrides:
  FAIR_ROUND_DIR           Explicit round directory to resume.
  FAIR_ROUND_ROOT          Directory containing timestamped round directories.
  FAIR_ROUND_TMUX_SESSION  tmux session name.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --round-dir)
      ROUND_DIR="${2:?--round-dir requires a value}"
      shift 2
      ;;
    --round-root)
      ROUND_ROOT="${2:?--round-root requires a value}"
      shift 2
      ;;
    --session)
      SESSION_NAME="${2:?--session requires a value}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

expected_stages=(
  00_prep
  10_sglang
  20_serverlessllm
  30_vllm
  40_slora
  50_faaslora
  90_compare
)

done_count_for_round() {
  local dir="$1"
  local count=0
  local stage=""
  for stage in "${expected_stages[@]}"; do
    if [[ -f "${dir}/state/${stage}.done" ]]; then
      count=$((count + 1))
    fi
  done
  printf '%s\n' "${count}"
}

select_round_dir() {
  if [[ ! -d "${ROUND_ROOT}" ]]; then
    echo "[ERROR] round root not found: ${ROUND_ROOT}" >&2
    return 1
  fi

  local candidates
  candidates="$(
    find "${ROUND_ROOT}" -maxdepth 1 -mindepth 1 -type d -print 2>/dev/null \
      | while IFS= read -r dir; do
          [[ -f "${dir}/round.env" ]] || continue
          local done_count complete_flag mtime
          done_count="$(done_count_for_round "${dir}")"
          complete_flag=0
          [[ -f "${dir}/state/90_compare.done" ]] && complete_flag=1
          mtime="$(stat -c '%Y' "${dir}")"
          printf '%s\t%s\t%s\t%s\n' "${complete_flag}" "${done_count}" "${mtime}" "${dir}"
        done \
      | sort -t $'\t' -k1,1n -k2,2nr -k3,3nr
  )"

  if [[ -z "${candidates}" ]]; then
    echo "[ERROR] no resumable rounds with round.env found under ${ROUND_ROOT}" >&2
    return 1
  fi

  printf '%s\n' "${candidates}" | head -n 1 | cut -f4-
}

if [[ -z "${ROUND_DIR}" ]]; then
  ROUND_DIR="$(select_round_dir)"
fi

ROUND_ENV="${ROUND_DIR}/round.env"
if [[ ! -f "${ROUND_ENV}" ]]; then
  echo "[ERROR] round.env not found: ${ROUND_ENV}" >&2
  exit 1
fi
if [[ ! -x "${RUNNER}" ]]; then
  echo "[ERROR] fair round runner is not executable: ${RUNNER}" >&2
  exit 1
fi

round_base="$(basename "${ROUND_DIR}")"
if [[ -z "${SESSION_NAME}" ]]; then
  SESSION_NAME="fair_${round_base}"
  SESSION_NAME="${SESSION_NAME//[^A-Za-z0-9_.-]/_}"
  SESSION_NAME="${SESSION_NAME:0:80}"
fi

echo "[resume] round_dir=${ROUND_DIR}"
echo "[resume] round_env=${ROUND_ENV}"
echo "[resume] tmux_session=${SESSION_NAME}"
echo "[resume] completed markers:"
find "${ROUND_DIR}/state" -maxdepth 1 -type f -name '*.done' -printf '  %f\n' 2>/dev/null | sort || true

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[resume] dry-run only; no tmux session was created or entered"
  exit 0
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "[resume] tmux session already exists; entering it"
else
  tmux new-session -d -s "${SESSION_NAME}" "
source /home/qhq/anaconda3/etc/profile.d/conda.sh >/dev/null 2>&1 || true
source '${ROUND_ENV}'
bash '${RUNNER}'
status=\$?
echo
if [ \"\$status\" -eq 0 ]; then
  echo '[finished] resumed fair round completed successfully.'
else
  echo \"[failed] resumed fair round failed with status=\$status.\"
fi
exec bash
"
  echo "[resume] created tmux session and started runner"
fi

if [[ -n "${TMUX:-}" ]]; then
  tmux switch-client -t "${SESSION_NAME}"
else
  tmux attach -t "${SESSION_NAME}"
fi
