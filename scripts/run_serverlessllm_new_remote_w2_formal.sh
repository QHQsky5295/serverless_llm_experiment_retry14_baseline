#!/usr/bin/env bash
set -euo pipefail

# Official ServerlessLLM runner with the approved W2 adapter-only view.
# Arrivals, prompts, targets, artifacts, scheduler and backend policy remain unchanged.

BASELINES_ROOT="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
MAIN_REPO="${SLLM_MAIN_REPO:-/home/qhq/serverless_llm_experiment_retry14_baseline}"
SLLM_REPO_ROOT="${SLLM_REPO_ROOT:-${BASELINES_ROOT}/vendor_new_baselines/ServerlessLLM_new_main_20260518}"
SLLM_ENV="${SLLM_NEW_ENV:-sllm_vllm0102_newserverless_20260518}"
SECTION_ID="${SLLM_NEW_SECTION_ID:-16_serverlessllm_new_remote_w2_v1}"
QUEUE_ID="${SLLM_NEW_QUEUE_ID:-$(date +%Y%m%d_%H%M%S)_serverlessllm_new_w2_v1}"
RESULT_ROOT="${BASELINES_ROOT}/results/paper_experiments/${SECTION_ID}"
LOG_ROOT="${BASELINES_ROOT}/results/logs/new_serverless_baselines_remote_w2_v1/serverlessllm_new/formal/${QUEUE_ID}"
SOURCE_ROUND="${SLLM_W2_SOURCE_ROUND:-${BASELINES_ROOT}/results/paper_experiments/11_remote_fair_main_local_sim_v4/20260512_123619_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1}"
SOURCE_TAG="llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1"
W2_MAP="${SLLM_W2_ADAPTER_MAP:-${MAIN_REPO}/paper_results/ieee_tc/inputs/20261015_7b_w2_adapter_map_v1.json}"
ENDPOINT="${SLLM_REMOTE_ARTIFACT_STAGE_ENDPOINT:-http://192.168.4.174:18081}"
RUN_TAG="${SLLM_RUN_TAG:-llama2_7b_r4000_a500_seed42_w2_rot100_s8_remote_fair_real-remote_v1}"

TRACE_SOURCE="${SOURCE_ROUND}/shared_artifacts/${SOURCE_TAG}_trace.json"
ADAPTER_SUBSET="${SOURCE_ROUND}/shared_artifacts/${SOURCE_TAG}_adapter_subset.json"
ROUND_DIR="${RESULT_ROOT}/${QUEUE_ID}_llama2_7b_${RUN_TAG}_serverlessllm_new"
INPUT_DIR="${ROUND_DIR}/shared_inputs"
TRACE_W2="${INPUT_DIR}/${RUN_TAG}_trace_w2.json"
SUMMARY_PATH="${ROUND_DIR}/raw/replay/${RUN_TAG}_serverlessllm_new_summary.json"

mkdir -p "${INPUT_DIR}" "${ROUND_DIR}/raw/replay" "${ROUND_DIR}/remote_cache/serverlessllm" "${LOG_ROOT}/llama2_7b"
if [[ -f "${SUMMARY_PATH}" ]]; then
  echo "[refuse] output already exists: ${SUMMARY_PATH}" >&2
  exit 2
fi

python "${BASELINES_ROOT}/scripts/derive_frozen_w2_trace.py" \
  --source-trace "${TRACE_SOURCE}" --adapter-map "${W2_MAP}" --output "${TRACE_W2}" \
  > "${INPUT_DIR}/w2_projection.json"

python - "${ROUND_DIR}" "${RUN_TAG}" "${TRACE_SOURCE}" "${TRACE_W2}" "${ADAPTER_SUBSET}" "${W2_MAP}" "${ENDPOINT}" "${SLLM_REPO_ROOT}" "${SLLM_ENV}" "${QUEUE_ID}" <<'PY'
import hashlib, json, sys
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

round_dir = Path(sys.argv[1]).resolve()
tag = sys.argv[2]
manifest = {
    'metric_schema_version': 'e2e_v3', 'system': 'ServerlessLLM',
    'queue_id': sys.argv[10], 'run_tag': tag,
    'model_profile': 'llama2_7b_main_v2_publicmix',
    'dataset_profile': 'azure_sharegpt_rep4000',
    'workload_profile': 'llama2_7b_auto500_formal4000_s8',
    'workload_view': 'W2', 'total_requests': 4000,
    'selected_num_adapters': 500, 'sampling_seed': 42,
    'source_trace_path': sys.argv[3], 'source_trace_sha256': sha(sys.argv[3]),
    'w2_trace_path': sys.argv[4], 'w2_trace_sha256': sha(sys.argv[4]),
    'shared_adapter_subset_path': sys.argv[5], 'shared_adapter_subset_sha256': sha(sys.argv[5]),
    'adapter_map_path': sys.argv[6], 'adapter_map_sha256': sha(sys.argv[6]),
    'endpoint': sys.argv[7], 'serverlessllm_repo_root': sys.argv[8], 'serverlessllm_env': sys.argv[9],
    'replay_path': str(round_dir / 'raw' / 'replay' / f'{tag}_serverlessllm_new_replay.json'),
    'summary_path': str(round_dir / 'raw' / 'replay' / f'{tag}_serverlessllm_new_summary.json'),
}
(round_dir / 'MANIFEST_PRE_RUN.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
PY

export SLLM_BASELINES_ROOT="${BASELINES_ROOT}" SLLM_MAIN_REPO="${MAIN_REPO}" SLLM_REPO_ROOT="${SLLM_REPO_ROOT}"
export SLLM_HEAD_ENV="${SLLM_ENV}" SLLM_WORKER_ENV="${SLLM_ENV}" SLLM_STORE_ENV="${SLLM_ENV}"
export SLLM_VLLM_ENV_NAME="${SLLM_ENV}" SLLM_VLLM_SOURCE_ENV="${SLLM_ENV}"
export SLLM_MODEL_PROFILE="llama2_7b_main_v2_publicmix" SLLM_DATASET_PROFILE="azure_sharegpt_rep4000"
export SLLM_WORKLOAD_PROFILE="llama2_7b_auto500_formal4000_s8" SLLM_TOTAL_REQUESTS=4000
export SLLM_SELECTED_NUM_ADAPTERS=500 SLLM_SAMPLING_SEED=42
export SLLM_GENERATION_CONTRACT=fixed_length_greedy_v1 SLLM_FIXED_OUTPUT_MAX_TOKENS=256 SLLM_FIXED_PROMPT_MAX_TOKENS=759
export SLLM_RUN_TAG="${RUN_TAG}" SLLM_RESULT_TAG="${RUN_TAG}_serverlessllm_new"
export SLLM_SHARED_TRACE_PATH="${TRACE_W2}" SLLM_SHARED_ADAPTER_SUBSET_PATH="${ADAPTER_SUBSET}"
export SLLM_SHARED_INPUT_DIR="${INPUT_DIR}" SLLM_RESULT_DIR="${ROUND_DIR}/raw/replay" SLLM_LOG_DIR="${LOG_ROOT}/llama2_7b"
export SLLM_REMOTE_ARTIFACT_STAGE_ENDPOINT="${ENDPOINT}" SLLM_REMOTE_ARTIFACT_STAGE_MODE=dynamic
export SLLM_REMOTE_ARTIFACT_STAGE_CACHE_DIR="${ROUND_DIR}/remote_cache/serverlessllm"
export SLLM_REMOTE_ARTIFACT_STAGE_WORKERS="${SLLM_REMOTE_ARTIFACT_STAGE_WORKERS:-1}"
export SLLM_REMOTE_ARTIFACT_STAGE_BANDWIDTH_MIB_S="${SLLM_REMOTE_ARTIFACT_STAGE_BANDWIDTH_MIB_S:-250}"
export SLLM_VLLM_PROBE_TIMEOUT_S="${SLLM_VLLM_PROBE_TIMEOUT_S:-300}"
export SLLM_BACKEND=vllm SLLM_WORKER_GPUS="${SLLM_WORKER_GPUS:-0,1,2,3}"
export SLLM_TIMEOUT_S="${SLLM_TIMEOUT_S:-3600}" SLLM_SLEEP_SCALE=1.0
export SLLM_EMPTY_SUCCESS_RETRIES="${SLLM_EMPTY_SUCCESS_RETRIES:-2}" SLLM_STACK_SUFFIX="newsllm_w2_${QUEUE_ID}"

bash "${BASELINES_ROOT}/scripts/run_serverlessllm_fair_experiment.sh"

python - "${ROUND_DIR}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / 'MANIFEST_PRE_RUN.json').read_text())
manifest['run_completed'] = True
manifest['summary_exists'] = Path(manifest['summary_path']).is_file()
(root / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
PY
echo "[done] ServerlessLLM W2 formal run: ${ROUND_DIR}"
