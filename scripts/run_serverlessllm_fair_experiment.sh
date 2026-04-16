#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
MAIN_REPO="${SLLM_MAIN_REPO:-/home/qhq/serverless_llm_experiment_retry14_baseline}"
SHARED_INPUT_DIR="${SLLM_SHARED_INPUT_DIR:-${ROOT_DIR}/results/shared_inputs}"
RESULT_DIR="${SLLM_RESULT_DIR:-${ROOT_DIR}/results/replay}"
CONFIG_PATH="${SLLM_CONFIG_PATH:-${MAIN_REPO}/configs/experiments.yaml}"

MODEL_PROFILE="${SLLM_MODEL_PROFILE:?SLLM_MODEL_PROFILE is required}"
DATASET_PROFILE="${SLLM_DATASET_PROFILE:?SLLM_DATASET_PROFILE is required}"
WORKLOAD_PROFILE="${SLLM_WORKLOAD_PROFILE:?SLLM_WORKLOAD_PROFILE is required}"
TOTAL_REQUESTS="${SLLM_TOTAL_REQUESTS:?SLLM_TOTAL_REQUESTS is required}"
SELECTED_NUM_ADAPTERS="${SLLM_SELECTED_NUM_ADAPTERS:?SLLM_SELECTED_NUM_ADAPTERS is required}"
SAMPLING_SEED="${SLLM_SAMPLING_SEED:-42}"
SERVING_MODEL_NAME="${SLLM_SERVING_MODEL_NAME:-${MODEL_PROFILE}}"
WORKER_GPUS="${SLLM_WORKER_GPUS:-0,1,2,3}"
BACKEND="${SLLM_BACKEND:-auto}"
VLLM_ENV_NAME="${SLLM_VLLM_ENV_NAME:-sllm_vllm0102_official}"
VLLM_SOURCE_ENV="${SLLM_VLLM_SOURCE_ENV:-LLM_vllm0102}"
SLEEP_SCALE="${SLLM_SLEEP_SCALE:-1.0}"
TIMEOUT_S="${SLLM_TIMEOUT_S:-3600}"
LIMIT_ADAPTERS="${SLLM_LIMIT_ADAPTERS:-}"
SHARED_TRACE_PATH="${SLLM_SHARED_TRACE_PATH:?SLLM_SHARED_TRACE_PATH is required}"
SHARED_ADAPTER_SUBSET_PATH="${SLLM_SHARED_ADAPTER_SUBSET_PATH:?SLLM_SHARED_ADAPTER_SUBSET_PATH is required}"

RUN_TAG="${SLLM_RUN_TAG:-${SERVING_MODEL_NAME}_r${TOTAL_REQUESTS}_a${SELECTED_NUM_ADAPTERS}_seed${SAMPLING_SEED}}"
TRACE_PATH="${SHARED_TRACE_PATH}"
ADAPTER_SUBSET_PATH="${SHARED_ADAPTER_SUBSET_PATH}"
DEPLOY_PATH="${SHARED_INPUT_DIR}/${RUN_TAG}_deploy.json"
REPLAY_PATH="${RESULT_DIR}/${RUN_TAG}_replay.json"
SUMMARY_PATH="${RESULT_DIR}/${RUN_TAG}_summary.json"

mkdir -p "${SHARED_INPUT_DIR}" "${RESULT_DIR}"

ORIGINAL_HEAD_ENV="${SLLM_HEAD_ENV:-}"
ORIGINAL_WORKER_ENV="${SLLM_WORKER_ENV:-}"
ORIGINAL_STORE_ENV="${SLLM_STORE_ENV:-}"
ORIGINAL_DIRECT_PATH_MODE="${SLLM_DIRECT_PATH_MODE:-}"

generate_deploy() {
  local backend="$1"
  local -a gen_cmd=(
    python "${ROOT_DIR}/scripts/generate_serverlessllm_deploy_config.py"
    --main-repo "${MAIN_REPO}"
    --model-profile "${MODEL_PROFILE}"
    --workload-profile "${WORKLOAD_PROFILE}"
    --backend "${backend}"
    --selected-num-adapters "${SELECTED_NUM_ADAPTERS}"
    --serving-model-name "${SERVING_MODEL_NAME}"
    --available-worker-gpus "${WORKER_GPUS}"
    --output "${DEPLOY_PATH}"
  )
  if [[ -n "${ADAPTER_SUBSET_PATH}" ]]; then
    gen_cmd+=(--adapter-subset-path "${ADAPTER_SUBSET_PATH}")
  fi
  if [[ -n "${LIMIT_ADAPTERS}" ]]; then
    gen_cmd+=(--limit-adapters "${LIMIT_ADAPTERS}")
  fi
  env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 conda run --no-capture-output -n sllm_head_official "${gen_cmd[@]}"
}

probe_vllm_backend() {
  local use_v1="$1"
  env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${WORKER_GPUS}" VLLM_USE_V1="${use_v1}" STORAGE_PATH="${ROOT_DIR}/models" SLLM_STORAGE_PATH="${ROOT_DIR}/models" \
    conda run --no-capture-output -n "${VLLM_RUNTIME_ENV}" \
    python "${ROOT_DIR}/scripts/probe_vllm_lora_runtime.py" \
      --deploy "${DEPLOY_PATH}" \
      --trace "${TRACE_PATH}"
}

configure_runtime_for_backend() {
  local backend="$1"
  if [[ "${backend}" == "vllm" ]]; then
    export SLLM_HEAD_ENV="${ORIGINAL_HEAD_ENV:-${VLLM_RUNTIME_ENV}}"
    export SLLM_WORKER_ENV="${ORIGINAL_WORKER_ENV:-${VLLM_RUNTIME_ENV}}"
    export SLLM_STORE_ENV="${ORIGINAL_STORE_ENV:-sllm_worker_official}"
    export SLLM_DIRECT_PATH_MODE="${ORIGINAL_DIRECT_PATH_MODE:-1}"
  else
    export SLLM_HEAD_ENV="${ORIGINAL_HEAD_ENV:-sllm_head_official}"
    export SLLM_WORKER_ENV="${ORIGINAL_WORKER_ENV:-sllm_worker_official}"
    export SLLM_STORE_ENV="${ORIGINAL_STORE_ENV:-${SLLM_WORKER_ENV}}"
    export SLLM_DIRECT_PATH_MODE="${ORIGINAL_DIRECT_PATH_MODE:-0}"
    unset VLLM_USE_V1 || true
  fi
}

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] experiments config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${TRACE_PATH}" ]]; then
  echo "[ERROR] shared trace artifact not found: ${TRACE_PATH}" >&2
  exit 1
fi
if [[ ! -f "${ADAPTER_SUBSET_PATH}" ]]; then
  echo "[ERROR] shared adapter subset artifact not found: ${ADAPTER_SUBSET_PATH}" >&2
  exit 1
fi

echo "[1/5] Validating shared trace and adapter subset"
env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 conda run --no-capture-output -n sllm_head_official \
  python - "${TRACE_PATH}" "${ADAPTER_SUBSET_PATH}" "${MODEL_PROFILE}" "${DATASET_PROFILE}" "${WORKLOAD_PROFILE}" "${TOTAL_REQUESTS}" "${SELECTED_NUM_ADAPTERS}" "${SAMPLING_SEED}" <<'PY'
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
if int(subset_payload.get("selected_num_adapters", -1)) != selected_num_adapters:
    raise SystemExit(
        "subset selected_num_adapters mismatch: "
        f"expected {selected_num_adapters}, got {subset_payload.get('selected_num_adapters')}"
    )
if int(trace_payload.get("selected_num_adapters", -1)) != selected_num_adapters:
    raise SystemExit(
        "trace selected_num_adapters mismatch: "
        f"expected {selected_num_adapters}, got {trace_payload.get('selected_num_adapters')}"
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
if len(subset_ids) != selected_num_adapters:
    raise SystemExit(
        f"subset adapter cardinality mismatch: expected {selected_num_adapters}, got {len(subset_ids)}"
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

readarray -t _METRIC_CFG < <(
  env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 conda run --no-capture-output -n sllm_head_official \
    python - "${ROOT_DIR}" "${CONFIG_PATH}" "${MODEL_PROFILE}" "${DATASET_PROFILE}" "${WORKLOAD_PROFILE}" <<'PY'
import sys
from pathlib import Path

root_dir = Path(sys.argv[1])
config_path = Path(sys.argv[2])
model_profile = sys.argv[3]
dataset_profile = sys.argv[4]
workload_profile = sys.argv[5]

sys.path.insert(0, str(root_dir / "scripts"))
from summarize_serverlessllm_replay import _load_yaml, _resolve_profiles  # type: ignore

cfg = _load_yaml(config_path)
_model_cfg, _adapters_cfg, _datasets_cfg, workload_cfg, coord_cfg = _resolve_profiles(
    cfg,
    model_profile,
    dataset_profile,
    workload_profile,
)
cost_model = dict(cfg.get("cost_model", {}) or {})
print(float(cost_model.get("base_cost_usd", 0.001)))
print(float(cost_model.get("input_token_cost_usd", 0.0000015)))
print(float(cost_model.get("output_token_cost_usd", 0.000002)))
print(float(workload_cfg.get("ttft_slo_ms", coord_cfg.get("ttft_slo_ms", 5000.0)) or 5000.0))
PY
)

if (( ${#_METRIC_CFG[@]} != 4 )); then
  echo "[ERROR] failed to resolve shared metric parameters from ${CONFIG_PATH}" >&2
  exit 1
fi

BASE_COST_USD="${_METRIC_CFG[0]}"
INPUT_TOKEN_COST_USD="${_METRIC_CFG[1]}"
OUTPUT_TOKEN_COST_USD="${_METRIC_CFG[2]}"
TTFT_SLO_MS="${_METRIC_CFG[3]}"

echo "      cost_model(base/in/out)=${BASE_COST_USD}/${INPUT_TOKEN_COST_USD}/${OUTPUT_TOKEN_COST_USD}"
echo "      ttft_slo_ms=${TTFT_SLO_MS}"
echo "      backend=${BACKEND}"
echo "      replay_timeout_s=${TIMEOUT_S}"

if [[ "${BACKEND}" == "vllm" || "${BACKEND}" == "auto" ]]; then
  VLLM_RUNTIME_ENV="${VLLM_ENV_NAME}"
  if [[ ! -d "/home/qhq/anaconda3/envs/${VLLM_RUNTIME_ENV}" ]]; then
    VLLM_RUNTIME_ENV="${VLLM_SOURCE_ENV}"
  fi
  echo "      vllm_runtime_env=${VLLM_RUNTIME_ENV}"
fi

ACTUAL_BACKEND="${BACKEND}"
if [[ "${ACTUAL_BACKEND}" == "auto" ]]; then
  ACTUAL_BACKEND="vllm"
fi
configure_runtime_for_backend "${ACTUAL_BACKEND}"

echo "[2/5] Generating ServerlessLLM deploy config"
generate_deploy "${ACTUAL_BACKEND}"

if [[ "${ACTUAL_BACKEND}" == "vllm" ]]; then
  echo "[2.5/5] Probing vLLM LoRA correctness on a real shared-trace request"
  if probe_vllm_backend 1; then
    export VLLM_USE_V1=1
    echo "      selected_vllm_engine=V1"
  elif probe_vllm_backend 0; then
    export VLLM_USE_V1=0
    echo "      selected_vllm_engine=V0"
  else
    if [[ "${BACKEND}" == "vllm" ]]; then
      echo "[ERROR] vLLM failed correctness probing on the sampled many-LoRA workload; refusing to run an invalid baseline." >&2
      exit 1
    fi
    echo "      vllm_probe_failed=true -> falling back to transformers for correctness"
    ACTUAL_BACKEND="transformers"
    configure_runtime_for_backend "${ACTUAL_BACKEND}"
    echo "[2.6/5] Regenerating deploy config for transformers fallback"
    generate_deploy "${ACTUAL_BACKEND}"
  fi
fi

BACKEND="${ACTUAL_BACKEND}"
echo "      runtime_backend=${BACKEND} head_env=${SLLM_HEAD_ENV} worker_env=${SLLM_WORKER_ENV} store_env=${SLLM_STORE_ENV} direct_path_mode=${SLLM_DIRECT_PATH_MODE}"

echo "[3/5] Starting isolated ServerlessLLM stack"
SLLM_WORKER_GPUS="${WORKER_GPUS}" bash "${ROOT_DIR}/scripts/start_serverlessllm_stack.sh"
echo "[4/5] Deploying model + sampled LoRA subset"
bash "${ROOT_DIR}/scripts/deploy_serverlessllm_model.sh" "${DEPLOY_PATH}"

echo "[5/5] Replaying shared trace with live metrics"
env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 conda run --no-capture-output -n sllm_head_official \
  python "${ROOT_DIR}/scripts/replay_openai_trace.py" \
  --trace "${TRACE_PATH}" \
  --base-url "http://127.0.0.1:8343" \
  --sleep-scale "${SLEEP_SCALE}" \
  --timeout-s "${TIMEOUT_S}" \
  --base-cost-usd "${BASE_COST_USD}" \
  --input-token-cost-usd "${INPUT_TOKEN_COST_USD}" \
  --output-token-cost-usd "${OUTPUT_TOKEN_COST_USD}" \
  --ttft-slo-ms "${TTFT_SLO_MS}" \
  --require-server-metrics \
  --label "${RUN_TAG}" \
  --output "${REPLAY_PATH}"

echo "[post] Summarizing replay into the shared paper metric schema"
env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 conda run --no-capture-output -n sllm_head_official \
  python "${ROOT_DIR}/scripts/summarize_serverlessllm_replay.py" \
  --main-repo "${MAIN_REPO}" \
  --config "${CONFIG_PATH}" \
  --model-profile "${MODEL_PROFILE}" \
  --dataset-profile "${DATASET_PROFILE}" \
  --workload-profile "${WORKLOAD_PROFILE}" \
  --trace "${TRACE_PATH}" \
  --replay "${REPLAY_PATH}" \
  --deploy "${DEPLOY_PATH}" \
  --scenario-name "serverlessllm_fair" \
  --output "${SUMMARY_PATH}"

echo "trace  -> ${TRACE_PATH}"
echo "deploy -> ${DEPLOY_PATH}"
echo "replay -> ${REPLAY_PATH}"
echo "summary -> ${SUMMARY_PATH}"
