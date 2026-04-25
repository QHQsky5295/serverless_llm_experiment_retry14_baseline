#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
MAIN_REPO="${SLLM_MAIN_REPO:-/home/qhq/serverless_llm_experiment_retry14_baseline}"
RESULT_DIR="${SLLM_RESULT_DIR:-${ROOT_DIR}/results/replay}"
LOG_DIR="${SLLM_LOG_DIR:-${ROOT_DIR}/results/logs}"
SHARED_INPUT_DIR="${SLLM_SHARED_INPUT_DIR:-${ROOT_DIR}/results/shared_inputs}"
CONFIG_PATH="${SLLM_CONFIG_PATH:-${MAIN_REPO}/configs/experiments.yaml}"
SGLANG_VENV="${SGLANG_VENV:-/home/qhq/.venvs/sglang_py310}"

MODEL_PROFILE="${SLLM_MODEL_PROFILE:?SLLM_MODEL_PROFILE is required}"
DATASET_PROFILE="${SLLM_DATASET_PROFILE:?SLLM_DATASET_PROFILE is required}"
WORKLOAD_PROFILE="${SLLM_WORKLOAD_PROFILE:?SLLM_WORKLOAD_PROFILE is required}"
TOTAL_REQUESTS="${SLLM_TOTAL_REQUESTS:?SLLM_TOTAL_REQUESTS is required}"
SELECTED_NUM_ADAPTERS="${SLLM_SELECTED_NUM_ADAPTERS:?SLLM_SELECTED_NUM_ADAPTERS is required}"
SAMPLING_SEED="${SLLM_SAMPLING_SEED:-42}"
GENERATION_SEED="${SLLM_GENERATION_SEED:-${SAMPLING_SEED}}"
RUN_TAG="${SLLM_RUN_TAG:-${MODEL_PROFILE}_r${TOTAL_REQUESTS}_a${SELECTED_NUM_ADAPTERS}_seed${SAMPLING_SEED}}"
SHARED_TRACE_PATH="${SLLM_SHARED_TRACE_PATH:?SLLM_SHARED_TRACE_PATH is required}"
SHARED_ADAPTER_SUBSET_PATH="${SLLM_SHARED_ADAPTER_SUBSET_PATH:?SLLM_SHARED_ADAPTER_SUBSET_PATH is required}"

SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"
SGLANG_PORT="${SGLANG_PORT:-8353}"
SGLANG_GPU_IDS="${SGLANG_GPU_IDS:-0,1,2,3}"
SGLANG_TENSOR_PARALLEL_SIZE="${SGLANG_TENSOR_PARALLEL_SIZE:-}"
SGLANG_DATA_PARALLEL_REPLICAS="${SGLANG_DATA_PARALLEL_REPLICAS:-1}"
SGLANG_SLEEP_SCALE="${SGLANG_SLEEP_SCALE:-1.0}"
SGLANG_TIMEOUT_S="${SGLANG_TIMEOUT_S:-3600}"

SGLANG_TOPOLOGY_RESULT_LABEL="dp${SGLANG_DATA_PARALLEL_REPLICAS}_tp${SGLANG_TENSOR_PARALLEL_SIZE:-profile}"
RESULT_TAG="${SGLANG_RESULT_TAG:-${RUN_TAG}_sglang_${SGLANG_TOPOLOGY_RESULT_LABEL}}"
REPLAY_PATH="${RESULT_DIR}/${RESULT_TAG}_replay.json"
SUMMARY_PATH="${RESULT_DIR}/${RESULT_TAG}_summary.json"
SERVER_LOG_PATH="${LOG_DIR}/${RUN_TAG}_sglang_server.log"
METRICS_DIR="${LOG_DIR}/${RUN_TAG}_sglang_metrics"
LAUNCH_SPEC_PATH="${SHARED_INPUT_DIR}/${RUN_TAG}_sglang_launch.yaml"
FLEET_SPEC_PATH="${SHARED_INPUT_DIR}/${RUN_TAG}_sglang_fleet.yaml"
LORA_PATHS_JSON="${SHARED_INPUT_DIR}/${RUN_TAG}_sglang_lora_paths.json"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}" "${SHARED_INPUT_DIR}" "${METRICS_DIR}"

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
if [[ ! -x "${SGLANG_VENV}/bin/python" ]]; then
  echo "[ERROR] SGLang venv python not found: ${SGLANG_VENV}/bin/python" >&2
  exit 1
fi

export PATH="${SGLANG_VENV}/bin:${PATH}"

cleanup() {
  local status=$?
  if [[ -n "${SGLANG_SERVER_PIDS:-}" ]]; then
    for pid in ${SGLANG_SERVER_PIDS}; do
      kill "${pid}" 2>/dev/null || true
    done
    for pid in ${SGLANG_SERVER_PIDS}; do
      wait "${pid}" 2>/dev/null || true
    done
  fi
  if [[ -n "${SGLANG_SERVER_PID:-}" ]]; then
    kill "${SGLANG_SERVER_PID}" 2>/dev/null || true
    wait "${SGLANG_SERVER_PID}" 2>/dev/null || true
  fi
  return "${status}"
}
trap cleanup EXIT

echo "[1/5] Validating shared trace and adapter subset"
"${SGLANG_VENV}/bin/python" - "${SHARED_TRACE_PATH}" "${SHARED_ADAPTER_SUBSET_PATH}" "${MODEL_PROFILE}" "${DATASET_PROFILE}" "${WORKLOAD_PROFILE}" "${TOTAL_REQUESTS}" "${SELECTED_NUM_ADAPTERS}" "${SAMPLING_SEED}" <<'PY'
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

readarray -t _METRIC_CFG < <(
  PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${SGLANG_VENV}/bin/python" - "${ROOT_DIR}" "${CONFIG_PATH}" "${MODEL_PROFILE}" "${DATASET_PROFILE}" "${WORKLOAD_PROFILE}" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]) / "scripts"))
from summarize_serverlessllm_replay import _load_yaml, _resolve_profiles  # type: ignore

cfg = _load_yaml(Path(sys.argv[2]))
model_cfg, _adapters_cfg, _datasets_cfg, workload_cfg, coord_cfg = _resolve_profiles(
    cfg,
    sys.argv[3],
    sys.argv[4],
    sys.argv[5],
)
cost_model = dict(cfg.get("cost_model", {}) or {})
print(float(cost_model.get("base_cost_usd", 0.001)))
print(float(cost_model.get("input_token_cost_usd", 0.0000015)))
print(float(cost_model.get("output_token_cost_usd", 0.000002)))
print(float(workload_cfg.get("ttft_slo_ms", coord_cfg.get("ttft_slo_ms", 5000.0)) or 5000.0))
print(str(model_cfg.get("name", "")))
print(int(model_cfg.get("max_model_len", 0) or 0))
print(int(model_cfg.get("max_input_len", 0) or 0))
print(int(model_cfg.get("max_output_tokens_cap", 0) or 0))
PY
)

if (( ${#_METRIC_CFG[@]} != 8 )); then
  echo "[ERROR] failed to resolve shared metric parameters from ${CONFIG_PATH}" >&2
  exit 1
fi

BASE_COST_USD="${_METRIC_CFG[0]}"
INPUT_TOKEN_COST_USD="${_METRIC_CFG[1]}"
OUTPUT_TOKEN_COST_USD="${_METRIC_CFG[2]}"
TTFT_SLO_MS="${_METRIC_CFG[3]}"
PROMPT_GUARD_TOKENIZER_MODEL="${_METRIC_CFG[4]}"
PROMPT_GUARD_MAX_MODEL_LEN="${_METRIC_CFG[5]}"
PROMPT_GUARD_MAX_INPUT_LEN="${_METRIC_CFG[6]}"
PROMPT_GUARD_MAX_OUTPUT_TOKENS_CAP="${_METRIC_CFG[7]}"

echo "[2/5] Building SGLang launch spec from shared subset"
PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${SGLANG_VENV}/bin/python" - "${CONFIG_PATH}" "${MAIN_REPO}" "${MODEL_PROFILE}" "${DATASET_PROFILE}" "${WORKLOAD_PROFILE}" "${SHARED_ADAPTER_SUBSET_PATH}" "${LAUNCH_SPEC_PATH}" "${LORA_PATHS_JSON}" "${SGLANG_HOST}" "${SGLANG_PORT}" "${SGLANG_TENSOR_PARALLEL_SIZE}" <<'PY'
import json
import sys
from pathlib import Path

import yaml


def _deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_profiles(cfg, model_profile, dataset_profile, workload_profile):
    model_cfg = dict(cfg.get("model", {}) or {})
    adapters_cfg = dict(cfg.get("lora_adapters", {}) or {})
    datasets_cfg = dict(cfg.get("datasets", {}) or {})
    workload_cfg = dict(cfg.get("workload", {}) or {})
    coord_cfg = dict(cfg.get("resource_coordination", {}) or {})
    for bucket_name, selected in [
        ("model_profiles", model_profile),
        ("dataset_profiles", dataset_profile),
        ("workload_profiles", workload_profile),
    ]:
        profile = dict((cfg.get(bucket_name, {}) or {}).get(selected) or {})
        if not profile:
            raise KeyError(f"unknown profile '{selected}' in {bucket_name}")
        model_cfg = _deep_merge(model_cfg, profile.get("model", {}) or {})
        adapters_cfg = _deep_merge(adapters_cfg, profile.get("lora_adapters", {}) or {})
        datasets_cfg = _deep_merge(datasets_cfg, profile.get("datasets", {}) or {})
        workload_cfg = _deep_merge(workload_cfg, profile.get("workload", {}) or {})
        coord_cfg = _deep_merge(coord_cfg, profile.get("resource_coordination", {}) or {})
    return model_cfg, adapters_cfg, datasets_cfg, workload_cfg, coord_cfg


cfg_path = Path(sys.argv[1])
main_repo = Path(sys.argv[2]).resolve()
model_profile = sys.argv[3]
dataset_profile = sys.argv[4]
workload_profile = sys.argv[5]
subset_path = Path(sys.argv[6]).resolve()
output_path = Path(sys.argv[7]).resolve()
lora_paths_json = Path(sys.argv[8]).resolve()
host = sys.argv[9]
port = int(sys.argv[10])
tp_override = str(sys.argv[11] or "").strip()

cfg = yaml.safe_load(cfg_path.read_text()) or {}
model_cfg, _adapters_cfg, _datasets_cfg, _workload_cfg, _coord_cfg = _resolve_profiles(
    cfg, model_profile, dataset_profile, workload_profile
)
subset_payload = json.loads(subset_path.read_text(encoding="utf-8"))
remote_dir = Path(subset_payload["remote_dir"]).resolve()

lora_entries = []
for item in subset_payload.get("adapters", []):
    adapter_id = str(item["id"])
    adapter_path = remote_dir / adapter_id
    if not adapter_path.exists():
        raise SystemExit(f"missing adapter path for SGLang launch: {adapter_path}")
    lora_entries.append(f"{adapter_id}={adapter_path}")

tp_size = int(tp_override) if tp_override else int(model_cfg.get("tensor_parallel_size", 1) or 1)
launch = {
    "model-path": str(model_cfg["name"]),
    "host": host,
    "port": port,
    "served-model-name": model_profile,
    "trust-remote-code": True,
    "tp": tp_size,
    "context-length": int(model_cfg.get("max_model_len", 1024) or 1024),
    "mem-fraction-static": float(model_cfg.get("gpu_memory_utilization", 0.7) or 0.7),
    "dtype": str(model_cfg.get("dtype", "auto") or "auto"),
    "enable-lora": True,
    "max-lora-rank": int(model_cfg.get("max_lora_rank", 64) or 64),
    "lora-target-modules": "all",
    "max-loras-per-batch": min(
        len(lora_entries),
        int(model_cfg.get("max_loras", 4) or 4),
    ) or 1,
    "max-loaded-loras": len(lora_entries),
    "enable-metrics": True,
    "enable-request-time-stats-logging": True,
}

if bool(model_cfg.get("enable_chunked_prefill", True)):
    launch["chunked-prefill-size"] = int(model_cfg.get("max_num_batched_tokens", 4096) or 4096)
else:
    launch["chunked-prefill-size"] = -1

if not bool(model_cfg.get("enable_prefix_caching", True)):
    launch["disable-radix-cache"] = True

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(yaml.safe_dump(launch, sort_keys=False), encoding="utf-8")
lora_paths_json.write_text(json.dumps(lora_entries, ensure_ascii=False, indent=2), encoding="utf-8")
print(output_path)
PY

echo "      cost_model(base/in/out)=${BASE_COST_USD}/${INPUT_TOKEN_COST_USD}/${OUTPUT_TOKEN_COST_USD}"
echo "      ttft_slo_ms=${TTFT_SLO_MS}"
echo "      launch_spec=${LAUNCH_SPEC_PATH}"
echo "      lora_paths_json=${LORA_PATHS_JSON}"
echo "      server_log=${SERVER_LOG_PATH}"
echo "      sglang_gpu_ids=${SGLANG_GPU_IDS}"
echo "      sglang_tensor_parallel_size=${SGLANG_TENSOR_PARALLEL_SIZE:-profile_default}"

echo "[3/5] Starting isolated SGLang server(s)"
rm -f "${SERVER_LOG_PATH}" "${LOG_DIR}/${RUN_TAG}_sglang_server_"*.log
rm -rf "${METRICS_DIR}" "${LOG_DIR}/${RUN_TAG}_sglang_metrics_"*
mkdir -p "${METRICS_DIR}"
mapfile -t SGLANG_LORA_PATHS < <(
  PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${SGLANG_VENV}/bin/python" - "${LORA_PATHS_JSON}" <<'PY'
import json
import sys
from pathlib import Path

items = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for item in items:
    print(str(item))
PY
)
TP_EFFECTIVE="$(
  PYTHONNOUSERSITE=1 "${SGLANG_VENV}/bin/python" - "${LAUNCH_SPEC_PATH}" <<'PY'
import sys
from pathlib import Path
import yaml

payload = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print(int(payload.get("tp", payload.get("tensor_parallel_size", 1)) or 1))
PY
)"
DP_REPLICAS="$(PYTHONNOUSERSITE=1 "${SGLANG_VENV}/bin/python" -c 'import sys; print(max(1, int(sys.argv[1])))' "${SGLANG_DATA_PARALLEL_REPLICAS}")"
IFS=',' read -r -a GPU_ID_ARRAY <<< "${SGLANG_GPU_IDS}"
REQUIRED_GPU_COUNT=$(( DP_REPLICAS * TP_EFFECTIVE ))
if (( ${#GPU_ID_ARRAY[@]} < REQUIRED_GPU_COUNT )); then
  echo "[ERROR] SGLang DP/TP topology requires ${REQUIRED_GPU_COUNT} GPU ids, got ${#GPU_ID_ARRAY[@]} from SGLANG_GPU_IDS=${SGLANG_GPU_IDS}" >&2
  exit 1
fi

SGLANG_SERVER_PIDS=""
SGLANG_BASE_URLS=()
SGLANG_REPLICA_PORTS=()
SGLANG_REPLICA_GPU_MASKS=()
SGLANG_STARTUP_SECS=()

for replica_idx in $(seq 0 $((DP_REPLICAS - 1))); do
  replica_port=$((SGLANG_PORT + replica_idx))
  replica_spec="${SHARED_INPUT_DIR}/${RUN_TAG}_sglang_launch_r${replica_idx}.yaml"
  replica_log="${LOG_DIR}/${RUN_TAG}_sglang_server_r${replica_idx}.log"
  replica_metrics="${LOG_DIR}/${RUN_TAG}_sglang_metrics_r${replica_idx}"
  gpu_slice=()
  for local_idx in $(seq 0 $((TP_EFFECTIVE - 1))); do
    gpu_slice+=("${GPU_ID_ARRAY[$((replica_idx * TP_EFFECTIVE + local_idx))]}")
  done
  replica_gpu_mask="$(IFS=,; echo "${gpu_slice[*]}")"
  PYTHONNOUSERSITE=1 "${SGLANG_VENV}/bin/python" - "${LAUNCH_SPEC_PATH}" "${replica_spec}" "${replica_port}" <<'PY'
import sys
from pathlib import Path
import yaml

payload = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
payload["port"] = int(sys.argv[3])
Path(sys.argv[2]).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY
  rm -f "${replica_log}"
  rm -rf "${replica_metrics}"
  mkdir -p "${replica_metrics}"
  launch_epoch="$(PYTHONNOUSERSITE=1 "${SGLANG_VENV}/bin/python" -c 'import time; print(f"{time.time():.6f}")')"
  CUDA_VISIBLE_DEVICES="${replica_gpu_mask}" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
    "${SGLANG_VENV}/bin/python" -m sglang.launch_server \
    --config "${replica_spec}" \
    --lora-paths "${SGLANG_LORA_PATHS[@]}" \
    --export-metrics-to-file \
    --export-metrics-to-file-dir "${replica_metrics}" \
    > "${replica_log}" 2>&1 &
  replica_pid=$!
  SGLANG_SERVER_PIDS="${SGLANG_SERVER_PIDS} ${replica_pid}"
  SGLANG_BASE_URLS+=("http://${SGLANG_HOST}:${replica_port}")
  SGLANG_REPLICA_PORTS+=("${replica_port}")
  SGLANG_REPLICA_GPU_MASKS+=("${replica_gpu_mask}")
  echo "      replica=${replica_idx} pid=${replica_pid} port=${replica_port} gpu_mask=${replica_gpu_mask}"
  ready=0
  for _ in $(seq 1 240); do
    if curl -s "http://${SGLANG_HOST}:${replica_port}/v1/models" >/tmp/sglang_models_${RUN_TAG}_${replica_idx}.json 2>/dev/null; then
      ready=1
      break
    fi
    if ! kill -0 "${replica_pid}" 2>/dev/null; then
      echo "[ERROR] SGLang replica ${replica_idx} exited before becoming ready. Tail log:" >&2
      tail -n 80 "${replica_log}" >&2 || true
      exit 1
    fi
    sleep 2
  done
  if [[ "${ready}" != "1" ]]; then
    echo "[ERROR] timed out waiting for SGLang replica ${replica_idx} /v1/models readiness. Tail log:" >&2
    tail -n 80 "${replica_log}" >&2 || true
    exit 1
  fi
  ready_epoch="$(PYTHONNOUSERSITE=1 "${SGLANG_VENV}/bin/python" -c 'import time; print(f"{time.time():.6f}")')"
  startup_sec="$(
    PYTHONNOUSERSITE=1 "${SGLANG_VENV}/bin/python" -c 'import sys; print(f"{max(0.0, float(sys.argv[2]) - float(sys.argv[1])):.6f}")' \
      "${launch_epoch}" "${ready_epoch}"
  )"
  SGLANG_STARTUP_SECS+=("${startup_sec}")
  echo "      replica=${replica_idx} startup_sec=${startup_sec}"
done

SGLANG_BASE_URL_LIST="$(IFS=,; echo "${SGLANG_BASE_URLS[*]}")"
SGLANG_SERVER_STARTUP_SEC="$(
  PYTHONNOUSERSITE=1 "${SGLANG_VENV}/bin/python" - "${SGLANG_STARTUP_SECS[@]}" <<'PY'
import sys
values = [float(v) for v in sys.argv[1:]]
print(f"{(max(values) if values else 0.0):.6f}")
PY
)"
PYTHONNOUSERSITE=1 "${SGLANG_VENV}/bin/python" - "${LAUNCH_SPEC_PATH}" "${FLEET_SPEC_PATH}" "${DP_REPLICAS}" "${TP_EFFECTIVE}" "${SGLANG_BASE_URL_LIST}" "${SGLANG_SERVER_STARTUP_SEC}" "${SGLANG_REPLICA_PORTS[*]}" "${SGLANG_REPLICA_GPU_MASKS[*]}" <<'PY'
import sys
from pathlib import Path
import yaml

base = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
dp = int(sys.argv[3])
tp = int(sys.argv[4])
base_urls = [item for item in sys.argv[5].split(",") if item]
startup_sec = float(sys.argv[6])
ports = [int(item) for item in sys.argv[7].split() if item]
gpu_masks = [item for item in sys.argv[8].split() if item]
fleet = dict(base)
fleet.update({
    "data_parallel_replicas": dp,
    "tp": tp,
    "tensor_parallel_size": tp,
    "num_gpus": dp * tp,
    "gpu_per_request": tp,
    "parallelism_topology": f"data_parallel_dp{dp}_tp{tp}" if dp > 1 else (f"model_parallel_tp{tp}" if tp > 1 else "single_gpu"),
    "base_urls": base_urls,
    "replica_ports": ports,
    "replica_gpu_masks": gpu_masks,
    "static_startup_sec": startup_sec,
})
Path(sys.argv[2]).write_text(yaml.safe_dump(fleet, sort_keys=False), encoding="utf-8")
PY
if [[ "${DP_REPLICAS}" -gt 1 ]]; then
  SGLANG_TOPOLOGY_LABEL="dp${DP_REPLICAS}_tp${TP_EFFECTIVE}"
else
  SGLANG_TOPOLOGY_LABEL="tp${TP_EFFECTIVE}"
fi
echo "      sglang_topology=${SGLANG_TOPOLOGY_LABEL}"
echo "      sglang_startup_sec=${SGLANG_SERVER_STARTUP_SEC}"
echo "      sglang_base_urls=${SGLANG_BASE_URL_LIST}"

echo "[4/5] Replaying shared trace with unified live metrics"
PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${SGLANG_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/replay_openai_trace.py" \
  --trace "${SHARED_TRACE_PATH}" \
  --base-url "${SGLANG_BASE_URLS[0]}" \
  --base-url-list "${SGLANG_BASE_URL_LIST}" \
  --endpoint-path "/generate" \
  --convert-chat-to-prompt \
  --sglang-native-generate \
  --prompt-guard-tokenizer-model "${PROMPT_GUARD_TOKENIZER_MODEL}" \
  --prompt-guard-max-model-len "${PROMPT_GUARD_MAX_MODEL_LEN}" \
  --prompt-guard-max-input-len "${PROMPT_GUARD_MAX_INPUT_LEN}" \
  --prompt-guard-max-output-tokens-cap "${PROMPT_GUARD_MAX_OUTPUT_TOKENS_CAP}" \
  --sleep-scale "${SGLANG_SLEEP_SCALE}" \
  --timeout-s "${SGLANG_TIMEOUT_S}" \
  --base-cost-usd "${BASE_COST_USD}" \
  --input-token-cost-usd "${INPUT_TOKEN_COST_USD}" \
  --output-token-cost-usd "${OUTPUT_TOKEN_COST_USD}" \
  --ttft-slo-ms "${TTFT_SLO_MS}" \
  --generation-seed "${GENERATION_SEED}" \
  --adapter-source-field "adapter_id" \
  --adapter-target-field "lora_path" \
  --drop-body-field "request_id" \
  --drop-body-field "lora_adapter_name" \
  --label "${RUN_TAG}" \
  --output "${REPLAY_PATH}"

PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${SGLANG_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/validate_replay_results.py" \
  --system "SGLang" \
  --replay "${REPLAY_PATH}" \
  --expected-total "${TOTAL_REQUESTS}"

echo "[5/5] Summarizing replay into the shared paper metric schema"
PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${SGLANG_VENV}/bin/python" \
  "${ROOT_DIR}/scripts/summarize_serverlessllm_replay.py" \
  --main-repo "${MAIN_REPO}" \
  --config "${CONFIG_PATH}" \
  --model-profile "${MODEL_PROFILE}" \
  --dataset-profile "${DATASET_PROFILE}" \
  --workload-profile "${WORKLOAD_PROFILE}" \
  --trace "${SHARED_TRACE_PATH}" \
  --adapter-subset "${SHARED_ADAPTER_SUBSET_PATH}" \
  --deploy "${FLEET_SPEC_PATH}" \
  --replay "${REPLAY_PATH}" \
  --scenario-name "sglang_fair" \
  --baseline-type "sglang" \
  --backend-label "sglang_official" \
  --system-name "SGLang" \
  --instance-mode "static_runtime" \
  --routing-policy "fcfs_batching" \
  --static-startup-sec "${SGLANG_SERVER_STARTUP_SEC}" \
  --output "${SUMMARY_PATH}"

echo
echo "SGLang fair replay complete."
echo "  Replay : ${REPLAY_PATH}"
echo "  Summary: ${SUMMARY_PATH}"
echo "  Server log: ${SERVER_LOG_PATH}"
