#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${ROOT_DIR}/scripts/run_all_experiments_user_scope.sh"

if [[ ! -x "${RUNNER}" ]]; then
  echo "[ERROR] runner not found: ${RUNNER}" >&2
  exit 1
fi

export FAASLORA_PROFILE_MODEL="${FAASLORA_PROFILE_MODEL:?FAASLORA_PROFILE_MODEL is required}"
export FAASLORA_PROFILE_DATASET="${FAASLORA_PROFILE_DATASET:?FAASLORA_PROFILE_DATASET is required}"
export FAASLORA_PROFILE_WORKLOAD="${FAASLORA_PROFILE_WORKLOAD:?FAASLORA_PROFILE_WORKLOAD is required}"
export FAASLORA_SHARED_TRACE_PATH="${FAASLORA_SHARED_TRACE_PATH:?FAASLORA_SHARED_TRACE_PATH is required}"
export FAASLORA_SHARED_ADAPTER_SUBSET_PATH="${FAASLORA_SHARED_ADAPTER_SUBSET_PATH:?FAASLORA_SHARED_ADAPTER_SUBSET_PATH is required}"
export FAASLORA_RESULTS_TAG="${FAASLORA_RESULTS_TAG:?FAASLORA_RESULTS_TAG is required}"

if [[ ! -f "${FAASLORA_SHARED_TRACE_PATH}" ]]; then
  echo "[ERROR] shared trace artifact not found: ${FAASLORA_SHARED_TRACE_PATH}" >&2
  exit 1
fi
if [[ ! -f "${FAASLORA_SHARED_ADAPTER_SUBSET_PATH}" ]]; then
  echo "[ERROR] shared adapter subset artifact not found: ${FAASLORA_SHARED_ADAPTER_SUBSET_PATH}" >&2
  exit 1
fi

/home/qhq/anaconda3/envs/LLM_vllm0102/bin/python - "${FAASLORA_SHARED_TRACE_PATH}" "${FAASLORA_SHARED_ADAPTER_SUBSET_PATH}" "${FAASLORA_PROFILE_MODEL}" "${FAASLORA_PROFILE_DATASET}" "${FAASLORA_PROFILE_WORKLOAD}" <<'PY'
import json
import sys
from pathlib import Path

trace_path = Path(sys.argv[1])
subset_path = Path(sys.argv[2])
model_profile = sys.argv[3]
dataset_profile = sys.argv[4]
workload_profile = sys.argv[5]

trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
subset_payload = json.loads(subset_path.read_text(encoding="utf-8"))

for field, expected in (
    ("model_profile", model_profile),
    ("dataset_profile", dataset_profile),
    ("workload_profile", workload_profile),
):
    observed = trace_payload.get(field)
    if observed != expected:
        raise SystemExit(f"trace {field} mismatch: expected {expected}, got {observed}")
    observed_subset = subset_payload.get(field)
    if observed_subset != expected:
        raise SystemExit(f"subset {field} mismatch: expected {expected}, got {observed_subset}")

trace_selected = int(trace_payload.get("selected_num_adapters", -1))
subset_selected = int(subset_payload.get("selected_num_adapters", -1))
if trace_selected != subset_selected:
    raise SystemExit(
        f"trace/subset selected_num_adapters mismatch: trace={trace_selected}, subset={subset_selected}"
    )

trace_seed = int(trace_payload.get("sampling_seed", -1))
subset_seed = int(subset_payload.get("sampling_seed", -1))
if trace_seed != subset_seed:
    raise SystemExit(
        f"trace/subset sampling_seed mismatch: trace={trace_seed}, subset={subset_seed}"
    )

subset_ids = {str(item["id"]) for item in subset_payload.get("adapters", [])}
if len(subset_ids) != subset_selected:
    raise SystemExit(
        f"subset adapter cardinality mismatch: expected {subset_selected}, got {len(subset_ids)}"
    )

trace_ids = {
    str(req.get("adapter_id"))
    for req in trace_payload.get("requests", [])
    if req.get("adapter_id") is not None
}
if not trace_ids.issubset(subset_ids):
    missing = sorted(trace_ids - subset_ids)
    raise SystemExit(f"trace references adapters outside subset: {missing[:8]}")
PY

cd "${ROOT_DIR}"
exec "${RUNNER}" --config configs/experiments.yaml --scenario faaslora_full --backend vllm "$@"
