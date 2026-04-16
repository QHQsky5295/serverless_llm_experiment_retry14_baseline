#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
MAIN_REPO="${SLLM_MAIN_REPO:-/home/qhq/serverless_llm_experiment_retry14_baseline}"
RESULT_DIR="${SLLM_RESULT_DIR:-${ROOT_DIR}/results/replay}"
PUNICA_LORA_ROOT="${PUNICA_LORA_ROOT:-${ROOT_DIR}/results/punica_loras}"
CONFIG_PATH="${SLLM_CONFIG_PATH:-${MAIN_REPO}/configs/experiments.yaml}"
PUNICA_VENV="${PUNICA_VENV:-/home/qhq/.venvs/punica_cu121_py310}"

MODEL_PROFILE="${SLLM_MODEL_PROFILE:?SLLM_MODEL_PROFILE is required}"
DATASET_PROFILE="${SLLM_DATASET_PROFILE:?SLLM_DATASET_PROFILE is required}"
WORKLOAD_PROFILE="${SLLM_WORKLOAD_PROFILE:?SLLM_WORKLOAD_PROFILE is required}"
TOTAL_REQUESTS="${SLLM_TOTAL_REQUESTS:?SLLM_TOTAL_REQUESTS is required}"
SELECTED_NUM_ADAPTERS="${SLLM_SELECTED_NUM_ADAPTERS:?SLLM_SELECTED_NUM_ADAPTERS is required}"
SAMPLING_SEED="${SLLM_SAMPLING_SEED:-42}"
RUN_TAG="${SLLM_RUN_TAG:-${MODEL_PROFILE}_r${TOTAL_REQUESTS}_a${SELECTED_NUM_ADAPTERS}_seed${SAMPLING_SEED}}"
SHARED_TRACE_PATH="${SLLM_SHARED_TRACE_PATH:?SLLM_SHARED_TRACE_PATH is required}"
SHARED_ADAPTER_SUBSET_PATH="${SLLM_SHARED_ADAPTER_SUBSET_PATH:?SLLM_SHARED_ADAPTER_SUBSET_PATH is required}"

PUNICA_GPU_ID="${PUNICA_GPU_ID:-0}"
PUNICA_BATCH_SIZE="${PUNICA_BATCH_SIZE:-4}"
PUNICA_CPU_LORA_CACHE_SIZE="${PUNICA_CPU_LORA_CACHE_SIZE:-128}"
PUNICA_GPU_LORA_CACHE_SIZE="${PUNICA_GPU_LORA_CACHE_SIZE:-8}"
PUNICA_LIVE_INTERVAL_S="${PUNICA_LIVE_INTERVAL_S:-2.0}"

REPLAY_PATH="${RESULT_DIR}/${RUN_TAG}_punica_replay.json"
SUMMARY_PATH="${RESULT_DIR}/${RUN_TAG}_punica_summary.json"
PUNICA_LORA_DIR="${PUNICA_LORA_ROOT}/${RUN_TAG}"
PUNICA_LORA_MANIFEST="${PUNICA_LORA_DIR}/punica_lora_manifest.json"

mkdir -p "${RESULT_DIR}" "${PUNICA_LORA_ROOT}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] experiments config not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SHARED_TRACE_PATH}" ]]; then
  echo "[ERROR] shared trace artifact not found: ${SHARED_TRACE_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SHARED_ADAPTER_SUBSET_PATH}" ]]; then
  echo "[ERROR] shared adapter subset artifact not found: ${SHARED_ADAPTER_SUBSET_PATH}" >&2
  exit 1
fi
if [[ ! -x "${PUNICA_VENV}/bin/python" ]]; then
  echo "[ERROR] Punica venv python not found: ${PUNICA_VENV}/bin/python" >&2
  exit 1
fi

echo "[1/4] Validating shared trace and adapter subset"
"${PUNICA_VENV}/bin/python" - "${SHARED_TRACE_PATH}" "${SHARED_ADAPTER_SUBSET_PATH}" "${MODEL_PROFILE}" "${DATASET_PROFILE}" "${WORKLOAD_PROFILE}" "${TOTAL_REQUESTS}" "${SELECTED_NUM_ADAPTERS}" "${SAMPLING_SEED}" <<'PY'
import json
import sys
from pathlib import Path

trace_path = Path(sys.argv[1])
subset_path = Path(sys.argv[2])
model_profile = sys.argv[3]
dataset_profile = sys.argv[4]
workload_profile = sys.argv[5]
total_requests = int(sys.argv[6])
selected_num_adapters = int(sys.argv[7])
sampling_seed = int(sys.argv[8])

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

if int(trace_payload.get("total_requests", -1)) != total_requests:
    raise SystemExit(
        f"trace total_requests mismatch: expected {total_requests}, got {trace_payload.get('total_requests')}"
    )
if int(trace_payload.get("selected_num_adapters", -1)) != selected_num_adapters:
    raise SystemExit(
        f"trace selected_num_adapters mismatch: expected {selected_num_adapters}, got {trace_payload.get('selected_num_adapters')}"
    )
if int(subset_payload.get("selected_num_adapters", -1)) != selected_num_adapters:
    raise SystemExit(
        f"subset selected_num_adapters mismatch: expected {selected_num_adapters}, got {subset_payload.get('selected_num_adapters')}"
    )
if int(trace_payload.get("sampling_seed", -1)) != sampling_seed:
    raise SystemExit(
        f"trace sampling_seed mismatch: expected {sampling_seed}, got {trace_payload.get('sampling_seed')}"
    )
if int(subset_payload.get("sampling_seed", -1)) != sampling_seed:
    raise SystemExit(
        f"subset sampling_seed mismatch: expected {sampling_seed}, got {subset_payload.get('sampling_seed')}"
    )

subset_ids = {str(item["id"]) for item in subset_payload.get("adapters", [])}
trace_ids = {
    str(req.get("adapter_id"))
    for req in trace_payload.get("requests", [])
    if req.get("adapter_id") is not None
}
if not trace_ids.issubset(subset_ids):
    missing = sorted(trace_ids - subset_ids)
    raise SystemExit(f"trace references adapters outside subset: {missing[:8]}")
PY

echo "[2/4] Materializing the sampled LoRA subset into Punica format"
"${PUNICA_VENV}/bin/python" "${ROOT_DIR}/scripts/materialize_punica_loras.py" \
  --adapter-subset "${SHARED_ADAPTER_SUBSET_PATH}" \
  --output-dir "${PUNICA_LORA_DIR}"

echo "[3/4] Replaying shared trace on Punica"
CUDA_VISIBLE_DEVICES="${PUNICA_GPU_ID}" "${PUNICA_VENV}/bin/python" "${ROOT_DIR}/scripts/replay_punica_trace.py" \
  --main-repo "${MAIN_REPO}" \
  --config "${CONFIG_PATH}" \
  --trace "${SHARED_TRACE_PATH}" \
  --punica-lora-manifest "${PUNICA_LORA_MANIFEST}" \
  --model-profile "${MODEL_PROFILE}" \
  --dataset-profile "${DATASET_PROFILE}" \
  --workload-profile "${WORKLOAD_PROFILE}" \
  --run-tag "${RUN_TAG}" \
  --batch-size "${PUNICA_BATCH_SIZE}" \
  --cpu-lora-cache-size "${PUNICA_CPU_LORA_CACHE_SIZE}" \
  --gpu-lora-cache-size "${PUNICA_GPU_LORA_CACHE_SIZE}" \
  --live-interval-s "${PUNICA_LIVE_INTERVAL_S}" \
  --output "${REPLAY_PATH}"

echo "[4/4] Summarizing Punica replay into the shared paper metric schema"
"${PUNICA_VENV}/bin/python" "${ROOT_DIR}/scripts/summarize_punica_replay.py" \
  --main-repo "${MAIN_REPO}" \
  --config "${CONFIG_PATH}" \
  --replay "${REPLAY_PATH}" \
  --trace "${SHARED_TRACE_PATH}" \
  --model-profile "${MODEL_PROFILE}" \
  --dataset-profile "${DATASET_PROFILE}" \
  --workload-profile "${WORKLOAD_PROFILE}" \
  --output "${SUMMARY_PATH}" \
  --scenario-name punica_fair

echo
echo "Punica fair replay complete."
echo "  Replay : ${REPLAY_PATH}"
echo "  Summary: ${SUMMARY_PATH}"
