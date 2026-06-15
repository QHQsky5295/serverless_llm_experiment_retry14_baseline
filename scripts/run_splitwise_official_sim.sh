#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SPLITWISE_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
RELAY_ROOT="${RELAY_ROOT:-/home/qhq/relayserve_serverless_llm}"
PROJECT_ROOT="${SPLITWISE_PROJECT_ROOT:-${ROOT_DIR}/vendor_new_baselines/splitwise-sim_main_20260614}"
ENV_DIR="${SPLITWISE_ENV_DIR:-/home/qhq/anaconda3/envs/splitwise_official_20260615}"
MODEL_KEY="${1:?usage: $0 3b|7b [max_requests] [run_tag]}"
MAX_REQUESTS="${2:-4000}"
RUN_TAG="${3:-$(date -u +%Y%m%dT%H%M%SZ)_splitwise_${MODEL_KEY}_r${MAX_REQUESTS}}"
TRACE_ROLE="${SPLITWISE_TRACE_ROLE:-formal4000}"
END_TIME_S="${SPLITWISE_END_TIME_S:-86400}"
ARRIVAL_OFFSET_S="${SPLITWISE_ARRIVAL_OFFSET_S:-1.0}"

case "${MODEL_KEY}" in
  3b)
    TRACE_PATH="${RELAY_ROOT}/data_imports/traces/multi_turn_continuation/llama32_3b/llama32_3b_burstgptv2_chronological_sharegpt_multi_turn_formal4000_v1_trace.json"
    ;;
  7b)
    TRACE_PATH="${RELAY_ROOT}/data_imports/traces/multi_turn_continuation/llama2_7b/llama2_7b_burstgptv2_chronological_sharegpt_multi_turn_formal4000_v1_trace.json"
    ;;
  *)
    echo "model must be 3b or 7b" >&2
    exit 2
    ;;
esac

RUN_DIR="${SPLITWISE_RUN_DIR:-${ROOT_DIR}/results/relayserve_continuation/splitwise/${RUN_TAG}}"
INPUT_DIR="${RUN_DIR}/input"
SIM_DIR="${RUN_DIR}/simulator"
SNAPSHOT_DIR="${RUN_DIR}/frozen_config"
LOG_DIR="${RUN_DIR}/logs"
TRACE_CSV="${INPUT_DIR}/relayserve_formal.csv"
TRACE_METADATA="${INPUT_DIR}/trace_conversion.json"
MANIFEST_PATH="${RUN_DIR}/manifest.json"
SUMMARY_PATH="${RUN_DIR}/source_summary.json"

if [[ -e "${RUN_DIR}" ]]; then
  echo "refusing to overwrite existing Splitwise run directory: ${RUN_DIR}" >&2
  exit 3
fi
for required in \
  "${TRACE_PATH}" \
  "${PROJECT_ROOT}/run.py" \
  "${PROJECT_ROOT}/data/perf_model.csv" \
  "${ENV_DIR}/bin/python"
do
  if [[ ! -e "${required}" ]]; then
    echo "missing required Splitwise input: ${required}" >&2
    exit 4
  fi
done

mkdir -p "${INPUT_DIR}" "${SIM_DIR}" "${SNAPSHOT_DIR}" "${LOG_DIR}"

echo "[1/5] Convert the frozen RelayServe trace to the official CSV schema"
"${ENV_DIR}/bin/python" "${ROOT_DIR}/scripts/prepare_splitwise_trace.py" \
  --input "${TRACE_PATH}" \
  --output "${TRACE_CSV}" \
  --metadata-output "${TRACE_METADATA}" \
  --max-requests "${MAX_REQUESTS}" \
  --arrival-offset-s "${ARRIVAL_OFFSET_S}"

echo "[2/5] Freeze official-source and environment provenance"
git -C "${PROJECT_ROOT}" rev-parse HEAD >"${SNAPSHOT_DIR}/splitwise_git_commit.txt"
git -C "${PROJECT_ROOT}" status --short >"${SNAPSHOT_DIR}/splitwise_git_status.txt"
"${ENV_DIR}/bin/python" -m pip freeze >"${SNAPSHOT_DIR}/pip_freeze.txt"
cp "${PROJECT_ROOT}/requirements.txt" "${SNAPSHOT_DIR}/requirements.txt"
cp "${PROJECT_ROOT}/configs/config.yaml" "${SNAPSHOT_DIR}/official_config.yaml"
cp "${PROJECT_ROOT}/configs/cluster/half_half.yaml" "${SNAPSHOT_DIR}/official_cluster.yaml"
cp "${PROJECT_ROOT}/configs/start_state/splitwise.yaml" "${SNAPSHOT_DIR}/official_start_state.yaml"
cp "${PROJECT_ROOT}/configs/performance_model/db.yaml" "${SNAPSHOT_DIR}/official_performance_model.yaml"

COMMAND=(
  "${ENV_DIR}/bin/python"
  run.py
  cluster=half_half
  start_state=splitwise
  performance_model=db
  applications.0.model_architecture=llama2-70b
  applications.0.model_size=llama2-70b-fp16
  applications.0.scheduler=kv_token_jsq
  "trace.dir=${INPUT_DIR}"
  trace.filename=relayserve_formal
  "end_time=${END_TIME_S}"
  debug=False
  seed=0
  "output_dir=${SIM_DIR}"
  "hydra.run.dir=${SIM_DIR}"
)
printf '%q ' "${COMMAND[@]}" >"${SNAPSHOT_DIR}/simulator_command.sh"
printf '\n' >>"${SNAPSHOT_DIR}/simulator_command.sh"

"${ENV_DIR}/bin/python" - \
  "${MANIFEST_PATH}" "${TRACE_PATH}" "${TRACE_CSV}" "${TRACE_METADATA}" \
  "${PROJECT_ROOT}" "${ROOT_DIR}" "${RELAY_ROOT}" "${MODEL_KEY}" \
  "${MAX_REQUESTS}" "${TRACE_ROLE}" "${RUN_TAG}" "${SIM_DIR}" \
  "${ARRIVAL_OFFSET_S}" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    manifest_path, source_trace, converted_trace, trace_metadata,
    project_root, baseline_root, relay_root, model_key, max_requests,
    trace_role, run_tag, sim_dir, arrival_offset_s,
) = sys.argv[1:]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git_head(path):
    return subprocess.check_output(
        ["git", "-C", path, "rev-parse", "HEAD"], text=True
    ).strip()

payload = {
    "schema": "relayserve_external_splitwise_sim_run_v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_tag": run_tag,
    "system": "Splitwise",
    "runtime_class": "official_discrete_event_simulator",
    "source_trace_model_key": model_key,
    "max_requests": int(max_requests),
    "trace_role": trace_role,
    "source_trace_path": source_trace,
    "source_trace_sha256": sha(source_trace),
    "converted_trace_path": converted_trace,
    "converted_trace_sha256": sha(converted_trace),
    "trace_conversion_path": trace_metadata,
    "trace_conversion_sha256": sha(trace_metadata),
    "arrival_offset_s": float(arrival_offset_s),
    "splitwise_repo": project_root,
    "splitwise_git_commit": git_head(project_root),
    "baseline_harness_git_commit": git_head(baseline_root),
    "relayserve_git_commit": git_head(relay_root),
    "simulator_output_dir": sim_dir,
    "official_profile": {
        "model_architecture": "llama2-70b",
        "model_size": "llama2-70b-fp16",
        "cluster": "half_half",
        "start_state": "splitwise",
        "prompt_instance": "1x DGX-H100 TP8",
        "token_instance": "1x DGX-A100 TP8",
        "scheduler": "kv_token_jsq",
        "performance_model": "official database",
    },
    "formal_main_comparison_eligible": False,
    "eligibility_class": "simulator_due_diligence_only",
}
Path(manifest_path).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

echo "[3/5] Run the unmodified official Splitwise simulator"
(
  cd "${PROJECT_ROOT}"
  PYTHONNOUSERSITE=1 "${COMMAND[@]}"
) >"${LOG_DIR}/simulator.log" 2>&1

echo "[4/5] Summarize simulator evidence with a non-testbed eligibility label"
"${ENV_DIR}/bin/python" "${ROOT_DIR}/scripts/summarize_splitwise_sim.py" \
  --detailed "${SIM_DIR}/detailed/0.csv" \
  --manifest "${MANIFEST_PATH}" \
  --output "${SUMMARY_PATH}" \
  --model-key "${MODEL_KEY}" \
  --expected-requests "${MAX_REQUESTS}"

echo "[5/5] Verify the official checkout remained untouched"
git -C "${PROJECT_ROOT}" status --short >"${SNAPSHOT_DIR}/splitwise_git_status_after.txt"
cmp \
  "${SNAPSHOT_DIR}/splitwise_git_status.txt" \
  "${SNAPSHOT_DIR}/splitwise_git_status_after.txt"
printf '0\n' >"${RUN_DIR}/.exit"
echo "Splitwise simulator evidence complete: ${RUN_DIR}"
