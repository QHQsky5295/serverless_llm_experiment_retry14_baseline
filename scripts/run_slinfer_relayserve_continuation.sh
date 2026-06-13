#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SLINFER_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
RELAY_ROOT="${RELAY_ROOT:-/home/qhq/relayserve_serverless_llm}"
PROJECT_BASE="${SLINFER_PROJECT_BASE:-${ROOT_DIR}/vendor_new_baselines/SLINFER_main_20260323}"
ENV_DIR="${SLINFER_ENV_DIR:-/home/qhq/anaconda3/envs/slinfer_official_20260612}"
CONFIG_PATH="${SLINFER_CONFIG_PATH:-${ROOT_DIR}/configs/relayserve_continuation_baselines.yaml}"
MODEL_KEY="${1:?usage: $0 3b|7b [max_requests] [run_tag]}"
MAX_REQUESTS="${2:-0}"
RUN_TAG="${3:-$(date -u +%Y%m%dT%H%M%SZ)_slinfer_${MODEL_KEY}_r${MAX_REQUESTS}}"
AUTO_STOP="${SLINFER_AUTO_STOP_STACK:-1}"
NODE_MEMORY_GB="${SLINFER_NODE_MEMORY_GB:-23.0}"
KEEP_ALIVE_S="${SLINFER_KEEP_ALIVE_S:-1}"
TIMEOUT_S="${SLINFER_TIMEOUT_S:-1800}"
MONITOR_TAIL_S="${SLINFER_MONITOR_TAIL_S:-$((KEEP_ALIVE_S + 2))}"

case "${MODEL_KEY}" in
  3b)
    MODEL_TYPE="llama-3.2-3b"
    TOKENIZER_PATH="/home/qhq/serverless_llm_experiment_retry14_baseline/models/LLM-Research--Llama-3.2-3B-Instruct"
    TRACE_PATH="${RELAY_ROOT}/data_imports/traces/multi_turn_continuation/llama32_3b/llama32_3b_burstgptv2_chronological_sharegpt_multi_turn_formal4000_v1_trace.json"
    TTFT_SLO_MS="180"
    TPOT_SLO_MS="14"
    ;;
  7b)
    MODEL_TYPE="llama-2-7b"
    TOKENIZER_PATH="/home/qhq/serverless_llm_experiment/models/meta-llama--Llama-2-7b-hf"
    TRACE_PATH="${RELAY_ROOT}/data_imports/traces/multi_turn_continuation/llama2_7b/llama2_7b_burstgptv2_chronological_sharegpt_multi_turn_formal4000_v1_trace.json"
    TTFT_SLO_MS="440"
    TPOT_SLO_MS="32"
    ;;
  *)
    echo "model must be 3b or 7b" >&2
    exit 2
    ;;
esac

TRACE_PATH="${SLINFER_TRACE_PATH_OVERRIDE:-${TRACE_PATH}}"
TRACE_ROLE="${SLINFER_TRACE_ROLE:-formal4000}"
RUN_DIR="${SLINFER_RUN_DIR:-${ROOT_DIR}/results/relayserve_continuation/slinfer/${RUN_TAG}}"
RAW_PATH="${RUN_DIR}/raw_records.json"
SUMMARY_PATH="${RUN_DIR}/source_summary.json"
MANIFEST_PATH="${RUN_DIR}/manifest.json"
LOG_DIR="${RUN_DIR}/logs"
SNAPSHOT_DIR="${RUN_DIR}/frozen_config"
STACK_PREFIX="$(printf '%s' "slinfer_${RUN_TAG}" | tr -c 'A-Za-z0-9_.-' '_')"

if [[ -e "${RUN_DIR}" ]]; then
  echo "refusing to overwrite existing SLINFER run directory: ${RUN_DIR}" >&2
  exit 3
fi
if [[ ! -f "${TRACE_PATH}" ]]; then
  echo "missing frozen trace: ${TRACE_PATH}" >&2
  exit 4
fi
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "missing SLINFER Python environment: ${ENV_DIR}" >&2
  exit 5
fi

mkdir -p "${LOG_DIR}" "${SNAPSHOT_DIR}"

export SLINFER_BASELINES_ROOT="${ROOT_DIR}"
export SLINFER_PROJECT_BASE="${PROJECT_BASE}"
export SLINFER_ENV_DIR="${ENV_DIR}"
export SLINFER_MODEL_KEY="${MODEL_KEY}"
export SLINFER_STACK_PREFIX="${STACK_PREFIX}"
export SLINFER_LOG_DIR="${LOG_DIR}"

cleanup() {
  local exit_code=$?
  for session in \
    "${STACK_PREFIX}_gateway" \
    "${STACK_PREFIX}_gpu0" \
    "${STACK_PREFIX}_gpu1" \
    "${STACK_PREFIX}_gpu2" \
    "${STACK_PREFIX}_gpu3" \
    "${STACK_PREFIX}_store"
  do
    tmux capture-pane -pJt "${session}" -S -4000 \
      >"${LOG_DIR}/${session}.tmux.log" 2>/dev/null || true
  done
  if [[ "${AUTO_STOP}" == "1" ]]; then
    bash "${ROOT_DIR}/scripts/stop_slinfer_stack.sh" \
      >"${LOG_DIR}/stop.log" 2>&1 || true
  fi
  exit "${exit_code}"
}
trap cleanup EXIT

echo "[0/7] Verify reproducible SLINFER compatibility patch"
bash "${ROOT_DIR}/scripts/apply_slinfer_relayserve_patch.sh"

echo "[1/7] Materialize audited 4x3090 SLINFER topology"
"${ENV_DIR}/bin/python" "${ROOT_DIR}/scripts/prepare_slinfer_relayserve.py" \
  --project-root "${PROJECT_BASE}" \
  --model-key "${MODEL_KEY}" \
  --node-memory-gb "${NODE_MEMORY_GB}" \
  --snapshot-dir "${SNAPSHOT_DIR}"

echo "[2/7] Record immutable inputs"
"${ENV_DIR}/bin/python" - \
  "${MANIFEST_PATH}" "${TRACE_PATH}" "${CONFIG_PATH}" "${PROJECT_BASE}" \
  "${ROOT_DIR}" "${RELAY_ROOT}" "${RUN_TAG}" "${MODEL_KEY}" "${MAX_REQUESTS}" \
  "${TRACE_ROLE}" "${NODE_MEMORY_GB}" "${KEEP_ALIVE_S}" \
  "${MONITOR_TAIL_S}" "${SNAPSHOT_DIR}" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    manifest_path,
    trace_path,
    config_path,
    project_base,
    root,
    relay_root,
    run_tag,
    model_key,
    max_requests,
    trace_role,
    node_memory_gb,
    keep_alive_s,
    monitor_tail_s,
    snapshot_dir,
) = sys.argv[1:]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git_head(path):
    return subprocess.check_output(
        ["git", "-C", path, "rev-parse", "HEAD"], text=True
    ).strip()

def git_diff_sha(path):
    diff = subprocess.check_output(
        ["git", "-C", path, "diff", "--binary"], stderr=subprocess.DEVNULL
    )
    return hashlib.sha256(diff).hexdigest()

root_path = Path(root)
source_paths = [
    root_path / "scripts/prepare_slinfer_relayserve.py",
    root_path / "scripts/apply_slinfer_relayserve_patch.sh",
    root_path / "scripts/replay_slinfer_trace.py",
    root_path / "scripts/run_slinfer_relayserve_continuation.sh",
    root_path / "scripts/start_slinfer_stack.sh",
    root_path / "scripts/stop_slinfer_stack.sh",
    root_path / "scripts/summarize_slinfer_replay.py",
    root_path / "patches/slinfer_relayserve_compat.patch",
]
snapshot_path = Path(snapshot_dir)
payload = {
    "schema": "relayserve_external_slinfer_run_v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_tag": run_tag,
    "system": "SLINFER",
    "model_key": model_key,
    "max_requests": int(max_requests),
    "trace_path": trace_path,
    "trace_sha256": sha(trace_path),
    "trace_role": trace_role,
    "config_path": config_path,
    "config_sha256": sha(config_path),
    "slinfer_repo": project_base,
    "slinfer_git_commit": git_head(project_base),
    "slinfer_worktree_diff_sha256": git_diff_sha(project_base),
    "baseline_harness_git_commit": git_head(root),
    "harness_source_sha256": {
        str(path.relative_to(root_path)): sha(path)
        for path in source_paths
    },
    "relayserve_git_commit": git_head(relay_root),
    "runtime_env": "/home/qhq/anaconda3/envs/slinfer_official_20260612",
    "node_memory_gb": float(node_memory_gb),
    "keep_alive_s": float(keep_alive_s),
    "monitor_tail_s": float(monitor_tail_s),
    "frozen_config_dir": snapshot_dir,
    "frozen_config_sha256": {
        path.name: sha(path)
        for path in sorted(snapshot_path.iterdir())
        if path.is_file()
    },
    "comparison_contract": {
        "workload": f"RelayServe frozen continuation trace ({trace_role}), rate=1.00x",
        "slo_profile": "paper_nominal",
        "gpu_budget": 4,
        "base_model_only": True,
        "lora_enabled": False,
        "system_mode": "official sota scheduler, GPU-only",
        "cost_model": "RelayServe lifecycle monetary model",
    },
}
Path(manifest_path).write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "[3/7] Start official SLINFER GPU-only stack"
STACK_START_MONOTONIC="$("${ENV_DIR}/bin/python" -c 'import time; print(time.monotonic())')"
bash "${ROOT_DIR}/scripts/start_slinfer_stack.sh" \
  >"${LOG_DIR}/stack_start.log" 2>&1
"${ENV_DIR}/bin/python" - \
  "${MANIFEST_PATH}" "${STACK_START_MONOTONIC}" <<'PY'
import json
import sys
import time
from pathlib import Path

manifest_path = Path(sys.argv[1])
started = float(sys.argv[2])
manifest = json.loads(manifest_path.read_text())
manifest["initial_runtime_startup_sec"] = time.monotonic() - started
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
PY

echo "[4/7] Replay frozen continuation workload"
REPLAY_ARGS=()
if [[ "${MAX_REQUESTS}" != "0" ]]; then
  REPLAY_ARGS+=(--max-requests "${MAX_REQUESTS}")
fi
"${ENV_DIR}/bin/python" "${ROOT_DIR}/scripts/replay_slinfer_trace.py" \
  --trace "${TRACE_PATH}" \
  --output "${RAW_PATH}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --model-type "${MODEL_TYPE}" \
  --model-id 0 \
  --gateway-url http://127.0.0.1:7000 \
  --label "${RUN_TAG}" \
  --max-model-len 3072 \
  --ttft-slo-ms "${TTFT_SLO_MS}" \
  --tpot-slo-ms "${TPOT_SLO_MS}" \
  --keep-alive-s "${KEEP_ALIVE_S}" \
  --timeout-s "${TIMEOUT_S}" \
  --monitor-tail-s "${MONITOR_TAIL_S}" \
  "${REPLAY_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/replay.log"

EXPECTED_TOTAL="$(
  "${ENV_DIR}/bin/python" - "${TRACE_PATH}" "${MAX_REQUESTS}" <<'PY'
import json
import sys
from pathlib import Path

total = len(json.loads(Path(sys.argv[1]).read_text())["requests"])
limit = int(sys.argv[2])
print(min(total, limit) if limit > 0 else total)
PY
)"

echo "[5/7] Validate raw records"
"${ENV_DIR}/bin/python" "${ROOT_DIR}/scripts/validate_replay_results.py" \
  --system SLINFER \
  --replay "${RAW_PATH}" \
  --expected-total "${EXPECTED_TOTAL}"

echo "[6/7] Summarize with the frozen lifecycle cost model"
"${ENV_DIR}/bin/python" "${ROOT_DIR}/scripts/summarize_slinfer_replay.py" \
  --replay "${RAW_PATH}" \
  --config "${CONFIG_PATH}" \
  --model-key "${MODEL_KEY}" \
  --scenario-name slinfer_relayserve_continuation \
  --output "${SUMMARY_PATH}"

echo "[7/7] Finalize manifest hashes"
"${ENV_DIR}/bin/python" - \
  "${MANIFEST_PATH}" "${RAW_PATH}" "${SUMMARY_PATH}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, raw_path, summary_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
for key, path in [
    ("raw_records", raw_path),
    ("source_summary", summary_path),
]:
    manifest[f"{key}_path"] = str(path.resolve())
    manifest[f"{key}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
PY

echo "run_dir=${RUN_DIR}"
echo "raw_records_path=${RAW_PATH}"
echo "source_summary_path=${SUMMARY_PATH}"
