#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
MAIN_REPO="${SLLM_MAIN_REPO:-/home/qhq/serverless_llm_experiment_retry14_baseline}"
RESULT_DIR="${SLLM_RESULT_DIR:-${ROOT_DIR}/results/replay}"
LOG_DIR="${SLLM_LOG_DIR:-${ROOT_DIR}/results/logs}"
SHARED_INPUT_DIR="${SLLM_SHARED_INPUT_DIR:-${ROOT_DIR}/results/shared_inputs}"
CONFIG_PATH="${SLLM_CONFIG_PATH:-${MAIN_REPO}/configs/experiments.yaml}"
HELPER_PYTHON="${SLORA_HELPER_PYTHON:-/home/qhq/anaconda3/envs/LLM_vllm0102/bin/python}"
SLORA_REPO="${SLORA_REPO:-${ROOT_DIR}/repos/S-LoRA}"
SLORA_PYTHON="${SLORA_PYTHON:-/home/qhq/anaconda3/envs/slora_official_cu118/bin/python}"
SLORA_ENV_DIR="${SLORA_ENV_DIR:-$(dirname "$(dirname "${SLORA_PYTHON}")")}"
SLORA_TORCH_LIB_DIR="${SLORA_TORCH_LIB_DIR:-${SLORA_ENV_DIR}/lib/python3.9/site-packages/torch/lib}"
SLORA_LD_LIBRARY_PATH="${SLORA_TORCH_LIB_DIR}:${SLORA_ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
SLORA_CUDA_HOME="${SLORA_CUDA_HOME:-${SLORA_ENV_DIR}}"

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

SLORA_HOST="${SLORA_HOST:-127.0.0.1}"
SLORA_PORT="${SLORA_PORT:-8463}"
SLORA_PORT_STRIDE="${SLORA_PORT_STRIDE:-10}"
SLORA_NCCL_PORT="${SLORA_NCCL_PORT:-29630}"
SLORA_GPU_IDS="${SLORA_GPU_IDS:-0,1,2,3}"
SLORA_TENSOR_PARALLEL_SIZE="${SLORA_TENSOR_PARALLEL_SIZE:-1}"
SLORA_DATA_PARALLEL_REPLICAS="${SLORA_DATA_PARALLEL_REPLICAS:-}"
SLORA_MAX_TOTAL_TOKEN_NUM="${SLORA_MAX_TOTAL_TOKEN_NUM:-14000}"
SLORA_SLEEP_SCALE="${SLORA_SLEEP_SCALE:-1.0}"
SLORA_TIMEOUT_S="${SLORA_TIMEOUT_S:-3600}"
SLORA_DRY_RUN="${SLORA_DRY_RUN:-0}"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}" "${SHARED_INPUT_DIR}"

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
if [[ ! -x "${HELPER_PYTHON}" ]]; then
  echo "[ERROR] helper python not found or not executable: ${HELPER_PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${SLORA_REPO}" ]]; then
  echo "[ERROR] S-LoRA repo not found: ${SLORA_REPO}" >&2
  exit 1
fi

cleanup() {
  local status=$?
  if [[ -n "${SLORA_SERVER_PIDS:-}" ]]; then
    for pid in ${SLORA_SERVER_PIDS}; do
      kill "${pid}" 2>/dev/null || true
    done
    for pid in ${SLORA_SERVER_PIDS}; do
      wait "${pid}" 2>/dev/null || true
    done
  fi
  return "${status}"
}
trap cleanup EXIT

ensure_port_is_free() {
  local port="$1"
  local label="$2"
  local pids=()
  mapfile -t pids < <(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null | awk '!seen[$0]++')
  if (( ${#pids[@]} == 0 )); then
    return 0
  fi
  echo "      clearing stale ${label} listener(s) on port=${port}: ${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true
  for _ in $(seq 1 15); do
    sleep 1
    mapfile -t pids < <(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null | awk '!seen[$0]++')
    if (( ${#pids[@]} == 0 )); then
      return 0
    fi
  done
  echo "[ERROR] stale ${label} listener still occupies port ${port}: ${pids[*]}" >&2
  exit 1
}

ensure_gpu_set_idle() {
  local gpu_csv="$1"
  local label="$2"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  local gpu_ids=()
  local gpu=""
  IFS=',' read -r -a gpu_ids <<< "${gpu_csv}"
  for gpu in "${gpu_ids[@]}"; do
    gpu="$(echo "${gpu}" | xargs)"
    [[ -z "${gpu}" ]] && continue
    local procs=()
    local query_output=""
    query_output="$(nvidia-smi --id="${gpu}" --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true)"
    mapfile -t procs < <(printf '%s\n' "${query_output}" | sed '/^[[:space:]]*$/d' | rg '^[[:space:]]*[0-9]+[[:space:]]*,')
    if (( ${#procs[@]} > 0 )); then
      echo "[ERROR] ${label} target GPU ${gpu} is not idle; active compute processes detected:" >&2
      printf '  %s\n' "${procs[@]}" >&2
      echo "        Clean up earlier experiments before starting a formal run." >&2
      exit 1
    fi
  done
}

echo "[1/5] Validating shared trace and adapter subset"
PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${HELPER_PYTHON}" - "${SHARED_TRACE_PATH}" "${SHARED_ADAPTER_SUBSET_PATH}" "${MODEL_PROFILE}" "${DATASET_PROFILE}" "${WORKLOAD_PROFILE}" "${TOTAL_REQUESTS}" "${SELECTED_NUM_ADAPTERS}" "${SAMPLING_SEED}" <<'PY'
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
    observed_subset = subset_payload.get(field)
    if observed != expected:
        raise SystemExit(f"trace {field} mismatch: expected {expected}, got {observed}")
    if observed_subset != expected:
        raise SystemExit(f"subset {field} mismatch: expected {expected}, got {observed_subset}")

if int(trace_payload.get("total_requests", -1)) != total_requests:
    raise SystemExit(f"trace total_requests mismatch: expected {total_requests}, got {trace_payload.get('total_requests')}")
if int(trace_payload.get("selected_num_adapters", -1)) != selected_num_adapters:
    raise SystemExit(f"trace selected_num_adapters mismatch: expected {selected_num_adapters}, got {trace_payload.get('selected_num_adapters')}")
if int(subset_payload.get("selected_num_adapters", -1)) != selected_num_adapters:
    raise SystemExit(f"subset selected_num_adapters mismatch: expected {selected_num_adapters}, got {subset_payload.get('selected_num_adapters')}")
if int(trace_payload.get("sampling_seed", -1)) != sampling_seed:
    raise SystemExit(f"trace sampling_seed mismatch: expected {sampling_seed}, got {trace_payload.get('sampling_seed')}")
if int(subset_payload.get("sampling_seed", -1)) != sampling_seed:
    raise SystemExit(f"subset sampling_seed mismatch: expected {sampling_seed}, got {subset_payload.get('sampling_seed')}")

subset_ids = {str(item["id"]) for item in subset_payload.get("adapters", [])}
trace_ids = {
    str(req.get("adapter_id"))
    for req in trace_payload.get("requests", [])
    if req.get("adapter_id") is not None
}
if len(subset_ids) != selected_num_adapters:
    raise SystemExit(f"subset adapter cardinality mismatch: expected {selected_num_adapters}, got {len(subset_ids)}")
if not trace_ids.issubset(subset_ids):
    missing = sorted(trace_ids - subset_ids)
    raise SystemExit(f"trace references adapters outside subset: {missing[:8]}")
PY

readarray -t _CFG < <(
  PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${HELPER_PYTHON}" - "${ROOT_DIR}" "${CONFIG_PATH}" "${MODEL_PROFILE}" "${DATASET_PROFILE}" "${WORKLOAD_PROFILE}" "${SLORA_TENSOR_PARALLEL_SIZE}" "${SLORA_DATA_PARALLEL_REPLICAS}" "${SLORA_GPU_IDS}" <<'PY'
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
tp_override = str(sys.argv[6] or "").strip()
dp_override = str(sys.argv[7] or "").strip()
gpu_ids = [item.strip() for item in str(sys.argv[8]).split(",") if item.strip()]
cost_model = dict(cfg.get("cost_model", {}) or {})

tp = int(tp_override) if tp_override else 1
if tp <= 0:
    raise SystemExit(f"invalid tensor parallel size: {tp}")
if dp_override:
    dp = int(dp_override)
else:
    dp = max(1, len(gpu_ids) // tp)
if dp <= 0:
    raise SystemExit(f"invalid data parallel replicas: {dp}")
required_gpus = dp * tp
if len(gpu_ids) < required_gpus:
    raise SystemExit(f"S-LoRA DP/TP topology requires {required_gpus} GPU ids, got {len(gpu_ids)}")

max_model_len = int(model_cfg.get("max_model_len", 1024) or 1024)
max_output_cap = int(model_cfg.get("max_output_tokens_cap", 0) or 0)
reserve = max(32, max_output_cap if max_output_cap > 0 else 32)
max_req_input_len = int(model_cfg.get("max_input_len") or max(32, max_model_len - reserve - 8))
max_req_total_len = max(max_model_len, max_req_input_len + max(1, reserve))

print(float(cost_model.get("base_cost_usd", 0.001)))
print(float(cost_model.get("input_token_cost_usd", 0.0000015)))
print(float(cost_model.get("output_token_cost_usd", 0.000002)))
print(float(workload_cfg.get("ttft_slo_ms", coord_cfg.get("ttft_slo_ms", 5000.0)) or 5000.0))
print(str(model_cfg.get("name", "")))
print(max_model_len)
print(int(model_cfg.get("max_input_len") or 0))
print(int(model_cfg.get("max_output_tokens_cap") or 0))
print(tp)
print(dp)
print(max_req_input_len)
print(max_req_total_len)
PY
)

if (( ${#_CFG[@]} != 12 )); then
  echo "[ERROR] failed to resolve S-LoRA fair-run parameters from ${CONFIG_PATH}" >&2
  exit 1
fi

BASE_COST_USD="${_CFG[0]}"
INPUT_TOKEN_COST_USD="${_CFG[1]}"
OUTPUT_TOKEN_COST_USD="${_CFG[2]}"
TTFT_SLO_MS="${_CFG[3]}"
MODEL_PATH="${_CFG[4]}"
PROMPT_GUARD_TOKENIZER_MODEL="${MODEL_PATH}"
PROMPT_GUARD_MAX_MODEL_LEN="${_CFG[5]}"
PROMPT_GUARD_MAX_INPUT_LEN="${_CFG[6]}"
PROMPT_GUARD_MAX_OUTPUT_TOKENS_CAP="${_CFG[7]}"
TP_EFFECTIVE="${_CFG[8]}"
DP_REPLICAS="${_CFG[9]}"
SLORA_MAX_REQ_INPUT_LEN="${SLORA_MAX_REQ_INPUT_LEN:-${_CFG[10]}}"
SLORA_MAX_REQ_TOTAL_LEN="${SLORA_MAX_REQ_TOTAL_LEN:-${_CFG[11]}}"
SLORA_BATCH_MAX_TOKENS="${SLORA_BATCH_MAX_TOKENS:-${SLORA_MAX_REQ_TOTAL_LEN}}"

SLORA_TOPOLOGY_LABEL="dp${DP_REPLICAS}_tp${TP_EFFECTIVE}"
RESULT_TAG="${SLORA_RESULT_TAG:-${RUN_TAG}_slora_${SLORA_TOPOLOGY_LABEL}}"
REPLAY_PATH="${RESULT_DIR}/${RESULT_TAG}_replay.json"
SUMMARY_PATH="${RESULT_DIR}/${RESULT_TAG}_summary.json"
LAUNCH_SPEC_PATH="${SHARED_INPUT_DIR}/${RESULT_TAG}_launch.yaml"
FLEET_SPEC_PATH="${SHARED_INPUT_DIR}/${RESULT_TAG}_fleet.yaml"
ADAPTER_VALUE_MAP_PATH="${SHARED_INPUT_DIR}/${RESULT_TAG}_adapter_value_map.json"
LORA_DIRS_TXT="${SHARED_INPUT_DIR}/${RESULT_TAG}_lora_dirs.txt"
SERVER_LOG_PREFIX="${LOG_DIR}/${RESULT_TAG}_server"

echo "[2/5] Building S-LoRA launch spec from shared subset"
PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${HELPER_PYTHON}" - "${SHARED_ADAPTER_SUBSET_PATH}" "${MODEL_PATH}" "${LAUNCH_SPEC_PATH}" "${FLEET_SPEC_PATH}" "${ADAPTER_VALUE_MAP_PATH}" "${LORA_DIRS_TXT}" "${TP_EFFECTIVE}" "${DP_REPLICAS}" "${SLORA_GPU_IDS}" "${SLORA_HOST}" "${SLORA_PORT}" "${SLORA_PORT_STRIDE}" "${SLORA_MAX_TOTAL_TOKEN_NUM}" "${SLORA_MAX_REQ_INPUT_LEN}" "${SLORA_MAX_REQ_TOTAL_LEN}" "${SLORA_BATCH_MAX_TOKENS}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

subset_path = Path(sys.argv[1]).resolve()
model_path = sys.argv[2]
launch_spec_path = Path(sys.argv[3]).resolve()
fleet_spec_path = Path(sys.argv[4]).resolve()
adapter_value_map_path = Path(sys.argv[5]).resolve()
lora_dirs_txt = Path(sys.argv[6]).resolve()
tp = int(sys.argv[7])
dp = int(sys.argv[8])
gpu_ids = [item.strip() for item in sys.argv[9].split(",") if item.strip()]
host = sys.argv[10]
base_port = int(sys.argv[11])
port_stride = int(sys.argv[12])
max_total_token_num = int(sys.argv[13])
max_req_input_len = int(sys.argv[14])
max_req_total_len = int(sys.argv[15])
batch_max_tokens = int(sys.argv[16])
if port_stride < 4:
    raise SystemExit(f"SLORA_PORT_STRIDE must leave room for internal ports, got {port_stride}")

payload = json.loads(subset_path.read_text(encoding="utf-8"))
remote_dir = Path(payload["remote_dir"]).resolve()
adapter_map = {}
lora_dirs = []
for item in payload.get("adapters", []):
    adapter_id = str(item["id"])
    adapter_path = remote_dir / adapter_id
    if not adapter_path.exists():
        raise SystemExit(f"missing adapter path for S-LoRA launch: {adapter_path}")
    adapter_map[adapter_id] = str(adapter_path)
    lora_dirs.append(str(adapter_path))

required = dp * tp
if len(gpu_ids) < required:
    raise SystemExit(f"topology requires {required} GPU ids, got {len(gpu_ids)}")

replica_ports = []
replica_gpu_masks = []
base_urls = []
for replica_idx in range(dp):
    gpu_slice = gpu_ids[replica_idx * tp : (replica_idx + 1) * tp]
    port = base_port + replica_idx * port_stride
    replica_ports.append(port)
    replica_gpu_masks.append(",".join(gpu_slice))
    base_urls.append(f"http://{host}:{port}")

launch = {
    "model": model_path,
    "host": host,
    "base_port": base_port,
    "port_stride": port_stride,
    "tensor_parallel_size": tp,
    "data_parallel_replicas": dp,
    "num_gpus": required,
    "gpu_per_request": tp,
    "parallelism_topology": (
        f"data_parallel_dp{dp}_tp{tp}" if dp > 1 else (f"model_parallel_tp{tp}" if tp > 1 else "single_gpu")
    ),
    "max_total_token_num": max_total_token_num,
    "max_req_input_len": max_req_input_len,
    "max_req_total_len": max_req_total_len,
    "batch_max_tokens": batch_max_tokens,
    "lora_modules_count": len(lora_dirs),
    "base_urls": base_urls,
    "replica_ports": replica_ports,
    "replica_gpu_masks": replica_gpu_masks,
    "static_startup_sec": 0.0,
}
launch_spec_path.write_text(yaml.safe_dump(launch, sort_keys=False), encoding="utf-8")
fleet_spec_path.write_text(yaml.safe_dump(launch, sort_keys=False), encoding="utf-8")
adapter_value_map_path.write_text(json.dumps(adapter_map, ensure_ascii=False, indent=2), encoding="utf-8")
lora_dirs_txt.write_text("\n".join(lora_dirs) + "\n", encoding="utf-8")
print(launch_spec_path)
PY

echo "      cost_model(base/in/out)=${BASE_COST_USD}/${INPUT_TOKEN_COST_USD}/${OUTPUT_TOKEN_COST_USD}"
echo "      ttft_slo_ms=${TTFT_SLO_MS}"
echo "      model=${MODEL_PATH}"
echo "      topology=${SLORA_TOPOLOGY_LABEL} gpu_ids=${SLORA_GPU_IDS}"
echo "      launch_spec=${LAUNCH_SPEC_PATH}"
echo "      adapter_map=${ADAPTER_VALUE_MAP_PATH}"
echo "      replay=${REPLAY_PATH}"
echo "      summary=${SUMMARY_PATH}"

if [[ "${SLORA_DRY_RUN}" == "1" ]]; then
  echo "[dry-run] S-LoRA launch/replay/summary specs generated; skipping server launch."
  exit 0
fi

if [[ ! -x "${SLORA_PYTHON}" ]]; then
  echo "[ERROR] S-LoRA python not found or not executable: ${SLORA_PYTHON}" >&2
  echo "        Create the isolated environment first; see S-LoRA_project/docs/SLoRA_REPRO_PLAN.md." >&2
  exit 1
fi

CUDA_HOME="${SLORA_CUDA_HOME}" \
PATH="${SLORA_ENV_DIR}/bin:${PATH}" \
LD_LIBRARY_PATH="${SLORA_LD_LIBRARY_PATH}" \
PYTHONPATH="${SLORA_REPO}:${PYTHONPATH:-}" \
PYTHONNOUSERSITE=1 \
"${SLORA_PYTHON}" - <<'PY'
import importlib
import numpy
import sys
mods = ["torch", "triton", "uvloop", "fastapi", "rpyc", "slora.server.api_server"]
failed = []
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception as exc:
        failed.append(f"{mod}: {type(exc).__name__}: {exc}")
if int(str(numpy.__version__).split(".", 1)[0]) >= 2:
    failed.append(
        f"numpy: incompatible version {numpy.__version__}; "
        "S-LoRA torch/cu118 runtime requires numpy<2 for stable startup"
    )
if failed:
    print("[ERROR] S-LoRA environment preflight failed:", file=sys.stderr)
    for item in failed:
        print("  " + item, file=sys.stderr)
    raise SystemExit(1)
PY

echo "[3/5] Starting isolated S-LoRA server(s)"
rm -f "${SERVER_LOG_PREFIX}"_r*.log
SLORA_SERVER_PIDS=""
SLORA_STARTUP_SECS=()
mapfile -t SLORA_LORA_DIRS < "${LORA_DIRS_TXT}"
IFS=',' read -r -a GPU_ID_ARRAY <<< "${SLORA_GPU_IDS}"
ensure_gpu_set_idle "${SLORA_GPU_IDS}" "S-LoRA"

for replica_idx in $(seq 0 $((DP_REPLICAS - 1))); do
  replica_port=$((SLORA_PORT + replica_idx * SLORA_PORT_STRIDE))
  replica_nccl_port=$((SLORA_NCCL_PORT + replica_idx))
  ensure_port_is_free "${replica_port}" "S-LoRA API"
  gpu_slice=()
  for local_idx in $(seq 0 $((TP_EFFECTIVE - 1))); do
    gpu_slice+=("${GPU_ID_ARRAY[$((replica_idx * TP_EFFECTIVE + local_idx))]}")
  done
  replica_gpu_mask="$(IFS=,; echo "${gpu_slice[*]}")"
  replica_log="${SERVER_LOG_PREFIX}_r${replica_idx}.log"
  launch_epoch="$(PYTHONNOUSERSITE=1 "${HELPER_PYTHON}" -c 'import time; print(f"{time.time():.6f}")')"
  server_cmd=(
    "${SLORA_PYTHON}" -m slora.server.api_server
    --model_dir "${MODEL_PATH}"
    --tokenizer_mode auto
    --host "${SLORA_HOST}"
    --port "${replica_port}"
    --tp "${TP_EFFECTIVE}"
    --nccl_port "${replica_nccl_port}"
    --max_total_token_num "${SLORA_MAX_TOTAL_TOKEN_NUM}"
    --max_req_input_len "${SLORA_MAX_REQ_INPUT_LEN}"
    --max_req_total_len "${SLORA_MAX_REQ_TOTAL_LEN}"
    --batch_max_tokens "${SLORA_BATCH_MAX_TOKENS}"
    --trust_remote_code
    --swap
    --disable_log_stats
  )
  for lora_dir in "${SLORA_LORA_DIRS[@]}"; do
    server_cmd+=(--lora-dirs "${lora_dir}")
  done
  CUDA_VISIBLE_DEVICES="${replica_gpu_mask}" \
  CUDA_HOME="${SLORA_CUDA_HOME}" \
  PATH="${SLORA_ENV_DIR}/bin:${PATH}" \
  LD_LIBRARY_PATH="${SLORA_LD_LIBRARY_PATH}" \
  PYTHONPATH="${SLORA_REPO}:${PYTHONPATH:-}" \
  PYTHONNOUSERSITE=1 \
  PYTHONUNBUFFERED=1 \
  "${server_cmd[@]}" > "${replica_log}" 2>&1 &
  replica_pid=$!
  SLORA_SERVER_PIDS="${SLORA_SERVER_PIDS} ${replica_pid}"
  echo "      replica=${replica_idx} pid=${replica_pid} port=${replica_port} gpu_mask=${replica_gpu_mask} log=${replica_log}"

  ready=0
  for _ in $(seq 1 360); do
    if curl -s "http://${SLORA_HOST}:${replica_port}/health" >/tmp/slora_health_${RESULT_TAG}_${replica_idx}.txt 2>/dev/null; then
      ready=1
      break
    fi
    if ! kill -0 "${replica_pid}" 2>/dev/null; then
      echo "[ERROR] S-LoRA replica ${replica_idx} exited before becoming ready. Tail log:" >&2
      tail -n 120 "${replica_log}" >&2 || true
      exit 1
    fi
    sleep 2
  done
  if [[ "${ready}" != "1" ]]; then
    echo "[ERROR] timed out waiting for S-LoRA replica ${replica_idx} /health readiness. Tail log:" >&2
    tail -n 120 "${replica_log}" >&2 || true
    exit 1
  fi
  ready_epoch="$(PYTHONNOUSERSITE=1 "${HELPER_PYTHON}" -c 'import time; print(f"{time.time():.6f}")')"
  startup_sec="$(
    PYTHONNOUSERSITE=1 "${HELPER_PYTHON}" -c 'import sys; print(f"{max(0.0, float(sys.argv[2]) - float(sys.argv[1])):.6f}")' \
      "${launch_epoch}" "${ready_epoch}"
  )"
  SLORA_STARTUP_SECS+=("${startup_sec}")
  echo "      replica=${replica_idx} startup_sec=${startup_sec}"
done

SLORA_SERVER_STARTUP_SEC="$(
  PYTHONNOUSERSITE=1 "${HELPER_PYTHON}" - "${SLORA_STARTUP_SECS[@]}" <<'PY'
import sys
values = [float(v) for v in sys.argv[1:]]
print(f"{(max(values) if values else 0.0):.6f}")
PY
)"
PYTHONNOUSERSITE=1 "${HELPER_PYTHON}" - "${LAUNCH_SPEC_PATH}" "${FLEET_SPEC_PATH}" "${SLORA_SERVER_STARTUP_SEC}" <<'PY'
import sys
from pathlib import Path
import yaml
fleet = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
fleet["static_startup_sec"] = float(sys.argv[3])
Path(sys.argv[2]).write_text(yaml.safe_dump(fleet, sort_keys=False), encoding="utf-8")
PY
echo "      slora_startup_sec=${SLORA_SERVER_STARTUP_SEC}"

echo "[4/5] Replaying shared trace with unified live metrics"
PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${HELPER_PYTHON}" \
  "${ROOT_DIR}/scripts/replay_openai_trace.py" \
  --trace "${SHARED_TRACE_PATH}" \
  --base-url "http://${SLORA_HOST}:${SLORA_PORT}" \
  --base-url-list "$(PYTHONNOUSERSITE=1 "${HELPER_PYTHON}" - "${FLEET_SPEC_PATH}" <<'PY'
import sys, yaml
from pathlib import Path
fleet = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print(",".join(str(x) for x in fleet.get("base_urls", [])))
PY
)" \
  --endpoint-path "/generate_stream" \
  --convert-chat-to-prompt \
  --slora-native-generate \
  --prompt-guard-tokenizer-model "${PROMPT_GUARD_TOKENIZER_MODEL}" \
  --prompt-guard-max-model-len "${PROMPT_GUARD_MAX_MODEL_LEN}" \
  --prompt-guard-max-input-len "${PROMPT_GUARD_MAX_INPUT_LEN}" \
  --prompt-guard-max-output-tokens-cap "${PROMPT_GUARD_MAX_OUTPUT_TOKENS_CAP}" \
  --sleep-scale "${SLORA_SLEEP_SCALE}" \
  --timeout-s "${SLORA_TIMEOUT_S}" \
  --base-cost-usd "${BASE_COST_USD}" \
  --input-token-cost-usd "${INPUT_TOKEN_COST_USD}" \
  --output-token-cost-usd "${OUTPUT_TOKEN_COST_USD}" \
  --ttft-slo-ms "${TTFT_SLO_MS}" \
  --generation-seed "${GENERATION_SEED}" \
  --adapter-source-field "adapter_id" \
  --adapter-target-field "lora_dir" \
  --adapter-value-map "${ADAPTER_VALUE_MAP_PATH}" \
  --drop-body-field "request_id" \
  --drop-body-field "lora_adapter_name" \
  --label "${RESULT_TAG}" \
  --output "${REPLAY_PATH}"

PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${HELPER_PYTHON}" \
  "${ROOT_DIR}/scripts/validate_replay_results.py" \
  --system "S-LoRA" \
  --replay "${REPLAY_PATH}" \
  --expected-total "${TOTAL_REQUESTS}"

PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${HELPER_PYTHON}" - "${REPLAY_PATH}" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
results = list(payload.get("results", []))
total = len(results)
ok = [r for r in results if bool(r.get("success"))]
if total <= 0:
    raise SystemExit(f"[ERROR] S-LoRA replay wrote no request results: {path}")
if len(ok) != total:
    failed = [r for r in results if not bool(r.get("success"))]
    print(f"[ERROR] S-LoRA replay success mismatch: ok={len(ok)} total={total}.", file=sys.stderr)
    for item in failed[:8]:
        print(
            f"  request_id={item.get('request_id')} adapter_id={item.get('adapter_id')} "
            f"status={item.get('status_code')} error={str(item.get('error') or '')[:500]}",
            file=sys.stderr,
        )
    raise SystemExit(1)
bad_token_source = [
    r for r in ok
    if str(r.get("prompt_token_source") or "") == "trace_expected"
    or str(r.get("completion_token_source") or "") == "trace_expected"
]
if bad_token_source:
    print(
        f"[ERROR] S-LoRA replay fell back to trace expected token counts: "
        f"bad={len(bad_token_source)} total_ok={len(ok)}. "
        "This would contaminate TPOT/token-cost diagnostics, so the run is rejected.",
        file=sys.stderr,
    )
    for item in bad_token_source[:8]:
        print(
            f"  request_id={item.get('request_id')} adapter_id={item.get('adapter_id')} "
            f"prompt_source={item.get('prompt_token_source')} "
            f"completion_source={item.get('completion_token_source')}",
            file=sys.stderr,
        )
    raise SystemExit(1)
print(f"[check] S-LoRA replay success: ok={len(ok)} total={total}")
print(
    "[check] S-LoRA token sources are observed/local; no trace_expected fallback "
    "entered formal token diagnostics."
)
PY

echo "[5/5] Summarizing replay into the shared paper metric schema"
PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${HELPER_PYTHON}" \
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
  --scenario-name "slora_fair" \
  --baseline-type "slora" \
  --backend-label "slora_official" \
  --system-name "S-LoRA" \
  --instance-mode "static_runtime" \
  --routing-policy "fcfs_batching" \
  --static-startup-sec "${SLORA_SERVER_STARTUP_SEC}" \
  --output "${SUMMARY_PATH}"

echo
echo "S-LoRA fair replay complete."
echo "  Replay : ${REPLAY_PATH}"
echo "  Summary: ${SUMMARY_PATH}"
echo "  Fleet  : ${FLEET_SPEC_PATH}"
echo "  Logs   : ${SERVER_LOG_PREFIX}_r*.log"
