#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
MAIN_REPO="${SLLM_MAIN_REPO:-/home/qhq/serverless_llm_experiment_retry14_baseline}"
SHARED_INPUT_DIR="${SLLM_SHARED_INPUT_DIR:-${ROOT_DIR}/results/shared_inputs}"
RESULT_DIR="${SLLM_RESULT_DIR:-${ROOT_DIR}/results/replay}"
LOG_DIR="${SLLM_LOG_DIR:-${ROOT_DIR}/results/logs}"
CONFIG_PATH="${SLLM_CONFIG_PATH:-${MAIN_REPO}/configs/experiments.yaml}"

MODEL_PROFILE="${SLLM_MODEL_PROFILE:?SLLM_MODEL_PROFILE is required}"
DATASET_PROFILE="${SLLM_DATASET_PROFILE:?SLLM_DATASET_PROFILE is required}"
WORKLOAD_PROFILE="${SLLM_WORKLOAD_PROFILE:?SLLM_WORKLOAD_PROFILE is required}"
TOTAL_REQUESTS="${SLLM_TOTAL_REQUESTS:?SLLM_TOTAL_REQUESTS is required}"
SELECTED_NUM_ADAPTERS="${SLLM_SELECTED_NUM_ADAPTERS:?SLLM_SELECTED_NUM_ADAPTERS is required}"
SAMPLING_SEED="${SLLM_SAMPLING_SEED:-42}"
GENERATION_SEED="${SLLM_GENERATION_SEED:-${SAMPLING_SEED}}"
SERVING_MODEL_NAME="${SLLM_SERVING_MODEL_NAME:-${MODEL_PROFILE}}"
WORKER_GPUS="${SLLM_WORKER_GPUS:-0,1,2,3}"
BACKEND="${SLLM_BACKEND:-auto}"
VLLM_ENV_NAME="${SLLM_VLLM_ENV_NAME:-sllm_vllm0102_official}"
VLLM_SOURCE_ENV="${SLLM_VLLM_SOURCE_ENV:-LLM_vllm0102}"
CONDA_ROOT="${SLLM_CONDA_ROOT:-/home/qhq/anaconda3}"
SLEEP_SCALE="${SLLM_SLEEP_SCALE:-1.0}"
default_client_timeout_s() {
  case "${MODEL_PROFILE}" in
    *13b*|*14b*|*13B*|*14B*)
      printf '%s\n' 21600
      ;;
    *)
      printf '%s\n' 3600
      ;;
  esac
}
TIMEOUT_S="${SLLM_TIMEOUT_S:-$(default_client_timeout_s)}"
EMPTY_SUCCESS_RETRIES="${SLLM_EMPTY_SUCCESS_RETRIES:-2}"
EMPTY_SUCCESS_RETRY_DELAY_S="${SLLM_EMPTY_SUCCESS_RETRY_DELAY_S:-1.0}"
VLLM_PROBE_TIMEOUT_S="${SLLM_VLLM_PROBE_TIMEOUT_S:-120}"
LIMIT_ADAPTERS="${SLLM_LIMIT_ADAPTERS:-}"
DEPLOY_MIN_INSTANCES="${SLLM_DEPLOY_MIN_INSTANCES:-}"
DEPLOY_MAX_INSTANCES="${SLLM_DEPLOY_MAX_INSTANCES:-}"
DEPLOY_TARGET="${SLLM_DEPLOY_TARGET:-}"
DEPLOY_KEEP_ALIVE="${SLLM_DEPLOY_KEEP_ALIVE:-}"
REPLAY_MAX_REQUESTS="${SLLM_REPLAY_MAX_REQUESTS:-}"
POST_DEPLOY_WAIT_S="${SLLM_POST_DEPLOY_WAIT_S:-0}"
RUN_TAG="${SLLM_RUN_TAG:-${SERVING_MODEL_NAME}_r${TOTAL_REQUESTS}_a${SELECTED_NUM_ADAPTERS}_seed${SAMPLING_SEED}}"
REMOTE_ARTIFACT_STAGE_ENDPOINT="${SLLM_REMOTE_ARTIFACT_STAGE_ENDPOINT:-${BASELINE_REMOTE_ARTIFACT_STAGE_ENDPOINT:-}}"
REMOTE_ARTIFACT_STAGE_CACHE_DIR="${SLLM_REMOTE_ARTIFACT_STAGE_CACHE_DIR:-${ROOT_DIR}/results/remote_artifact_cache/${MODEL_PROFILE}/serverlessllm/${RUN_TAG}}"
REMOTE_ARTIFACT_STAGE_WORKERS="${SLLM_REMOTE_ARTIFACT_STAGE_WORKERS:-1}"
REMOTE_ARTIFACT_STAGE_BANDWIDTH_MBPS="${SLLM_REMOTE_ARTIFACT_STAGE_BANDWIDTH_MBPS:-${BASELINE_REMOTE_ARTIFACT_BANDWIDTH_MBPS:-250}}"
REMOTE_ARTIFACT_STAGE_MODE="${SLLM_REMOTE_ARTIFACT_STAGE_MODE:-dynamic}"
SHARED_TRACE_PATH="${SLLM_SHARED_TRACE_PATH:?SLLM_SHARED_TRACE_PATH is required}"
SHARED_ADAPTER_SUBSET_PATH="${SLLM_SHARED_ADAPTER_SUBSET_PATH:?SLLM_SHARED_ADAPTER_SUBSET_PATH is required}"

RESULT_TAG="${SLLM_RESULT_TAG:-${RUN_TAG}_serverlessllm}"
TRACE_PATH="${SHARED_TRACE_PATH}"
ADAPTER_SUBSET_PATH="${SHARED_ADAPTER_SUBSET_PATH}"
DEPLOY_PATH="${SHARED_INPUT_DIR}/${RESULT_TAG}_deploy.json"
REPLAY_PATH="${RESULT_DIR}/${RESULT_TAG}_replay.json"
SUMMARY_PATH="${RESULT_DIR}/${RESULT_TAG}_summary.json"
STACK_SUFFIX_RAW="${SLLM_STACK_SUFFIX:-${RUN_TAG}}"
STACK_SUFFIX="$(printf '%s' "${STACK_SUFFIX_RAW}" | tr -c 'A-Za-z0-9_.-' '_')"

mkdir -p "${SHARED_INPUT_DIR}" "${RESULT_DIR}" "${LOG_DIR}"

ORIGINAL_HEAD_ENV="${SLLM_HEAD_ENV:-}"
ORIGINAL_WORKER_ENV="${SLLM_WORKER_ENV:-}"
ORIGINAL_STORE_ENV="${SLLM_STORE_ENV:-}"
ORIGINAL_DIRECT_PATH_MODE="${SLLM_DIRECT_PATH_MODE:-}"
STACK_STARTED=0

env_prefix() {
  local env_name="$1"
  printf '%s/envs/%s' "${CONDA_ROOT}" "${env_name}"
}

env_python_bin() {
  local env_name="$1"
  printf '%s/bin/python' "$(env_prefix "${env_name}")"
}

run_python_in_env() {
  local env_name="$1"
  shift
  env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "$(env_python_bin "${env_name}")" "$@"
}

cleanup_stack() {
  if [[ "${STACK_STARTED}" == "1" && "${SLLM_AUTO_STOP_STACK:-1}" == "1" ]]; then
    bash "${ROOT_DIR}/scripts/stop_serverlessllm_stack.sh" >/dev/null 2>&1 || true
  fi
}

trap cleanup_stack EXIT

generate_deploy() {
  local backend="$1"
  local -a gen_cmd=(
    "${ROOT_DIR}/scripts/generate_serverlessllm_deploy_config.py"
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
  if [[ -n "${DEPLOY_MIN_INSTANCES}" ]]; then
    gen_cmd+=(--min-instances "${DEPLOY_MIN_INSTANCES}")
  fi
  if [[ -n "${DEPLOY_MAX_INSTANCES}" ]]; then
    gen_cmd+=(--max-instances "${DEPLOY_MAX_INSTANCES}")
  fi
  if [[ -n "${DEPLOY_TARGET}" ]]; then
    gen_cmd+=(--target "${DEPLOY_TARGET}")
  fi
  if [[ -n "${DEPLOY_KEEP_ALIVE}" ]]; then
    gen_cmd+=(--keep-alive "${DEPLOY_KEEP_ALIVE}")
  fi
  run_python_in_env sllm_head_official "${gen_cmd[@]}"
}

env_has_vllm() {
  local env_name="$1"
  timeout 20s run_python_in_env "${env_name}" \
    -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("vllm") else 1)' \
    >/dev/null 2>&1
}

probe_vllm_backend() {
  local use_v1="$1"
  local probe_log
  probe_log="$(mktemp)"
  if timeout "${VLLM_PROBE_TIMEOUT_S}s" \
    env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${WORKER_GPUS}" VLLM_USE_V1="${use_v1}" STORAGE_PATH="${ROOT_DIR}/models" SLLM_STORAGE_PATH="${ROOT_DIR}/models" \
      "$(env_python_bin "${VLLM_RUNTIME_ENV}")" "${ROOT_DIR}/scripts/probe_vllm_lora_runtime.py" \
      --deploy "${DEPLOY_PATH}" \
      --trace "${TRACE_PATH}" >"${probe_log}" 2>&1; then
    rm -f "${probe_log}"
    return 0
  fi
  echo "      vllm_probe_engine=V${use_v1} failed; tail follows:" >&2
  grep -Ev '^(ERROR conda\\.cli\\.main_run:execute|TRACE conda\\.|==> )' "${probe_log}" | tail -n 40 >&2 || true
  rm -f "${probe_log}"
  return 1
}

prefetch_serverlessllm_probe_adapter() {
  if [[ -z "${SLLM_REQUEST_REMOTE_ADAPTER_MAP}" ]]; then
    return 0
  fi
  run_python_in_env sllm_head_official - "${ROOT_DIR}" "${TRACE_PATH}" "${SLLM_REQUEST_REMOTE_ADAPTER_MAP}" "${REMOTE_ARTIFACT_STAGE_ENDPOINT}" "${REMOTE_ARTIFACT_STAGE_BANDWIDTH_MBPS}" "${TIMEOUT_S}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
trace_path = Path(sys.argv[2]).resolve()
adapter_map_path = Path(sys.argv[3]).resolve()
endpoint = sys.argv[4]
bandwidth_mbps = float(sys.argv[5] or 0.0)
timeout_s = float(sys.argv[6] or 300.0)

sys.path.insert(0, str(root / "scripts"))
from replay_openai_trace import RemoteArtifactFetcher  # type: ignore

trace = json.loads(trace_path.read_text(encoding="utf-8"))
adapter_map = json.loads(adapter_map_path.read_text(encoding="utf-8"))
adapter_id = ""
best_tokens = -1
for req in trace.get("requests", []) or []:
    body = req.get("body") or {}
    candidate = str(body.get("lora_adapter_name") or req.get("adapter_id") or "").strip()
    if not candidate or candidate not in adapter_map:
        continue
    prompt_tokens = int(req.get("prompt_input_tokens", 0) or 0)
    if prompt_tokens > best_tokens:
        adapter_id = candidate
        best_tokens = prompt_tokens
if not adapter_id:
    raise SystemExit("cannot find a probe adapter in trace and remote adapter map")
fetcher = RemoteArtifactFetcher(
    endpoint=endpoint,
    timeout_s=timeout_s,
    token_env="PRIME_REMOTE_TOKEN",
    bandwidth_mbps=bandwidth_mbps,
)
metrics = fetcher.ensure(adapter_id, adapter_map[adapter_id])
print(json.dumps({"adapter_id": adapter_id, **metrics}, indent=2))
PY
}

reset_serverlessllm_request_remote_cache() {
  if [[ -z "${SLLM_REQUEST_REMOTE_ADAPTER_MAP}" ]]; then
    return 0
  fi
  run_python_in_env sllm_head_official - "${SLLM_REQUEST_REMOTE_ADAPTER_MAP}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

adapter_map = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
parents = {str(Path(path).expanduser().resolve().parent) for path in adapter_map.values()}
for parent_s in parents:
    parent = Path(parent_s)
    if parent.exists():
        shutil.rmtree(parent)
    parent.mkdir(parents=True, exist_ok=True)
print(f"reset_request_remote_cache_dirs={len(parents)}")
PY
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
    unset VLLM_ATTENTION_BACKEND || true
    unset VLLM_USE_FLASHINFER_SAMPLER || true
    unset VLLM_NO_USAGE_STATS || true
  fi
}

export_vllm_runtime_env_from_deploy() {
  if [[ ! -f "${DEPLOY_PATH}" ]]; then
    return 0
  fi
  readarray -t _VLLM_RUNTIME_ENV < <(
    run_python_in_env sllm_head_official - "${DEPLOY_PATH}" <<'PY'
import json
import sys
from pathlib import Path

deploy = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
backend_config = deploy.get("backend_config", {}) or {}
runtime_env = backend_config.get("vllm_runtime_env", {}) or {}
if not isinstance(runtime_env, dict):
    runtime_env = {}
for key in (
    "VLLM_USE_V1",
    "VLLM_ATTENTION_BACKEND",
    "VLLM_USE_FLASHINFER_SAMPLER",
    "VLLM_NO_USAGE_STATS",
):
    value = runtime_env.get(key)
    if value not in (None, ""):
        print(f"{key}={value}")
PY
  )
  local item key value
  for item in "${_VLLM_RUNTIME_ENV[@]}"; do
    key="${item%%=*}"
    value="${item#*=}"
    if [[ -n "${key}" ]]; then
      export "${key}=${value}"
    fi
  done
}

update_deploy_vllm_use_v1() {
  local use_v1="$1"
  run_python_in_env sllm_head_official - "${DEPLOY_PATH}" "${use_v1}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
use_v1 = str(sys.argv[2]).strip()
deploy = json.loads(path.read_text(encoding="utf-8"))
backend_config = deploy.setdefault("backend_config", {})
runtime_env = backend_config.setdefault("vllm_runtime_env", {})
runtime_env["VLLM_USE_V1"] = use_v1
backend_config["vllm_use_v1"] = use_v1 == "1"
path.write_text(json.dumps(deploy, indent=2), encoding="utf-8")
PY
  export_vllm_runtime_env_from_deploy
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
run_python_in_env sllm_head_official \
  - "${TRACE_PATH}" "${ADAPTER_SUBSET_PATH}" "${MODEL_PROFILE}" "${DATASET_PROFILE}" "${WORKLOAD_PROFILE}" "${TOTAL_REQUESTS}" "${SELECTED_NUM_ADAPTERS}" "${SAMPLING_SEED}" <<'PY'
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
  run_python_in_env sllm_head_official \
    - "${ROOT_DIR}" "${CONFIG_PATH}" "${MODEL_PROFILE}" "${DATASET_PROFILE}" "${WORKLOAD_PROFILE}" <<'PY'
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
model_cfg = dict(_model_cfg or {})
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

echo "      cost_model(base/in/out)=${BASE_COST_USD}/${INPUT_TOKEN_COST_USD}/${OUTPUT_TOKEN_COST_USD}"
echo "      ttft_slo_ms=${TTFT_SLO_MS}"
echo "      prompt_guard(model/max_len/max_input/output_cap)=${PROMPT_GUARD_TOKENIZER_MODEL}/${PROMPT_GUARD_MAX_MODEL_LEN}/${PROMPT_GUARD_MAX_INPUT_LEN}/${PROMPT_GUARD_MAX_OUTPUT_TOKENS_CAP}"
echo "      backend=${BACKEND}"
echo "      replay_timeout_s=${TIMEOUT_S}"
echo "      replay_max_requests=${REPLAY_MAX_REQUESTS:-0}"
echo "      deploy_overrides(min/max/target/keep_alive)=${DEPLOY_MIN_INSTANCES:-default}/${DEPLOY_MAX_INSTANCES:-default}/${DEPLOY_TARGET:-default}/${DEPLOY_KEEP_ALIVE:-default}"
echo "      post_deploy_wait_s=${POST_DEPLOY_WAIT_S}"
echo "      empty_success_retries=${EMPTY_SUCCESS_RETRIES} delay_s=${EMPTY_SUCCESS_RETRY_DELAY_S}"

REMOTE_STAGE_SEC="0.0"
SLLM_REQUEST_REMOTE_ADAPTER_MAP=""
if [[ -n "${REMOTE_ARTIFACT_STAGE_ENDPOINT}" ]]; then
  case "${REMOTE_ARTIFACT_STAGE_MODE}" in
    dynamic|stage) ;;
    *)
      echo "[ERROR] SLLM_REMOTE_ARTIFACT_STAGE_MODE must be dynamic or stage; got ${REMOTE_ARTIFACT_STAGE_MODE}" >&2
      exit 1
      ;;
  esac
  if [[ "${REMOTE_ARTIFACT_STAGE_MODE}" == "stage" ]]; then
    echo "[remote-stage] Materializing ServerlessLLM adapter subset from ${REMOTE_ARTIFACT_STAGE_ENDPOINT}"
    rm -rf "${REMOTE_ARTIFACT_STAGE_CACHE_DIR}"
    STAGED_ADAPTER_SUBSET_PATH="${SHARED_INPUT_DIR}/${RESULT_TAG}_remote_staged_adapter_subset.json"
    run_python_in_env sllm_head_official \
      "${ROOT_DIR}/scripts/materialize_remote_adapter_subset.py" \
      --main-repo "${MAIN_REPO}" \
      --adapter-subset "${ADAPTER_SUBSET_PATH}" \
      --endpoint "${REMOTE_ARTIFACT_STAGE_ENDPOINT}" \
      --output-dir "${REMOTE_ARTIFACT_STAGE_CACHE_DIR}" \
      --output-subset "${STAGED_ADAPTER_SUBSET_PATH}" \
      --workers "${REMOTE_ARTIFACT_STAGE_WORKERS}" \
      --bandwidth-mbps "${REMOTE_ARTIFACT_STAGE_BANDWIDTH_MBPS}"
    ADAPTER_SUBSET_PATH="${STAGED_ADAPTER_SUBSET_PATH}"
    REMOTE_STAGE_SEC="$(
      run_python_in_env sllm_head_official - "${ADAPTER_SUBSET_PATH}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
ms = float((payload.get("remote_materialization") or {}).get("elapsed_ms") or 0.0)
print(f"{ms / 1000.0:.6f}")
PY
    )"
    echo "      staged_adapter_subset=${ADAPTER_SUBSET_PATH}"
    echo "      staged_adapter_cache=${REMOTE_ARTIFACT_STAGE_CACHE_DIR}"
    echo "      staged_adapter_bandwidth_mib_s=${REMOTE_ARTIFACT_STAGE_BANDWIDTH_MBPS}"
    echo "      staged_adapter_sec=${REMOTE_STAGE_SEC}"
  else
    echo "[remote-dynamic] Configuring ServerlessLLM request-path adapter materialization from ${REMOTE_ARTIFACT_STAGE_ENDPOINT}"
    rm -rf "${REMOTE_ARTIFACT_STAGE_CACHE_DIR}"
    DYNAMIC_ADAPTER_SUBSET_PATH="${SHARED_INPUT_DIR}/${RESULT_TAG}_remote_dynamic_adapter_subset.json"
    SLLM_REQUEST_REMOTE_ADAPTER_MAP="${SHARED_INPUT_DIR}/${RESULT_TAG}_remote_dynamic_adapter_map.json"
    run_python_in_env sllm_head_official - "${ADAPTER_SUBSET_PATH}" "${REMOTE_ARTIFACT_STAGE_CACHE_DIR}" "${DYNAMIC_ADAPTER_SUBSET_PATH}" "${SLLM_REQUEST_REMOTE_ADAPTER_MAP}" "${REMOTE_ARTIFACT_STAGE_ENDPOINT}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

subset_path = Path(sys.argv[1]).resolve()
cache_dir = Path(sys.argv[2]).resolve()
out_subset = Path(sys.argv[3]).resolve()
out_map = Path(sys.argv[4]).resolve()
endpoint = sys.argv[5]

payload = json.loads(subset_path.read_text(encoding="utf-8"))
if cache_dir.exists():
    shutil.rmtree(cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)
adapter_map = {}
for item in payload.get("adapters", []) or []:
    adapter_id = str(item.get("id") or "").strip()
    if not adapter_id:
        continue
    target = cache_dir / adapter_id
    target.mkdir(parents=True, exist_ok=True)
    adapter_map[adapter_id] = str(target)
if not adapter_map:
    raise SystemExit("adapter subset contains no adapter ids")
repaired = dict(payload)
repaired["remote_dir"] = str(cache_dir)
repaired["source_remote_dir"] = payload.get("remote_dir")
repaired["remote_artifact_endpoint"] = endpoint
repaired["remote_materialization"] = {
    "mode": "request_path_dynamic",
    "adapter_count": len(adapter_map),
    "output_dir": str(cache_dir),
}
out_subset.parent.mkdir(parents=True, exist_ok=True)
out_subset.write_text(json.dumps(repaired, indent=2), encoding="utf-8")
out_map.write_text(json.dumps(adapter_map, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"adapter_count": len(adapter_map), "cache_dir": str(cache_dir), "map": str(out_map)}))
PY
    ADAPTER_SUBSET_PATH="${DYNAMIC_ADAPTER_SUBSET_PATH}"
    echo "      request_remote_subset=${ADAPTER_SUBSET_PATH}"
    echo "      request_remote_map=${SLLM_REQUEST_REMOTE_ADAPTER_MAP}"
    echo "      request_remote_cache=${REMOTE_ARTIFACT_STAGE_CACHE_DIR}"
    echo "      request_remote_bandwidth_mib_s=${REMOTE_ARTIFACT_STAGE_BANDWIDTH_MBPS}"
  fi
fi

if [[ "${BACKEND}" == "vllm" || "${BACKEND}" == "auto" ]]; then
  VLLM_RUNTIME_ENV="${VLLM_ENV_NAME}"
  if [[ ! -d "/home/qhq/anaconda3/envs/${VLLM_RUNTIME_ENV}" ]] || ! env_has_vllm "${VLLM_RUNTIME_ENV}"; then
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
  export_vllm_runtime_env_from_deploy
  echo "      vllm_runtime_env=VLLM_USE_V1=${VLLM_USE_V1:-unset} VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-unset} VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-unset}"
fi

if [[ "${ACTUAL_BACKEND}" == "vllm" ]]; then
  echo "[2.5/5] Probing vLLM LoRA correctness on a real shared-trace request"
  prefetch_serverlessllm_probe_adapter
  if probe_vllm_backend 1; then
    update_deploy_vllm_use_v1 1
    echo "      selected_vllm_engine=V1"
  elif probe_vllm_backend 0; then
    update_deploy_vllm_use_v1 0
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
  reset_serverlessllm_request_remote_cache
fi

BACKEND="${ACTUAL_BACKEND}"
echo "      runtime_backend=${BACKEND} head_env=${SLLM_HEAD_ENV} worker_env=${SLLM_WORKER_ENV} store_env=${SLLM_STORE_ENV} direct_path_mode=${SLLM_DIRECT_PATH_MODE}"
export SLLM_HEAD_SESSION="${SLLM_HEAD_SESSION:-sllm_head_${STACK_SUFFIX}}"
export SLLM_STORE_SESSION="${SLLM_STORE_SESSION:-sllm_store_${STACK_SUFFIX}}"
export SLLM_SERVE_SESSION="${SLLM_SERVE_SESSION:-sllm_serve_${STACK_SUFFIX}}"
export SLLM_WORKER_SESSION_PREFIX="${SLLM_WORKER_SESSION_PREFIX:-sllm_worker_${STACK_SUFFIX}}"
export SLLM_SERVE_LOG_PATH="${SLLM_SERVE_LOG_PATH:-${LOG_DIR}/${RUN_TAG}_serve.log}"
echo "      stack_sessions=head:${SLLM_HEAD_SESSION} store:${SLLM_STORE_SESSION} serve:${SLLM_SERVE_SESSION} worker_prefix:${SLLM_WORKER_SESSION_PREFIX}"
echo "      serve_log=${SLLM_SERVE_LOG_PATH}"

echo "[3/5] Starting isolated ServerlessLLM stack"
STACK_STARTED=1
SLLM_WORKER_GPUS="${WORKER_GPUS}" bash "${ROOT_DIR}/scripts/start_serverlessllm_stack.sh"
echo "[4/5] Deploying model + sampled LoRA subset"
bash "${ROOT_DIR}/scripts/deploy_serverlessllm_model.sh" "${DEPLOY_PATH}"
if [[ "${POST_DEPLOY_WAIT_S}" != "0" && "${POST_DEPLOY_WAIT_S}" != "0.0" ]]; then
  echo "[4.5/5] Waiting ${POST_DEPLOY_WAIT_S}s for pre-created backends to finish loading"
  sleep "${POST_DEPLOY_WAIT_S}"
fi

echo "[5/5] Replaying shared trace with live metrics"
SLLM_REPLAY_EXTRA_ARGS=()
if [[ -n "${SLLM_REQUEST_REMOTE_ADAPTER_MAP}" ]]; then
  SLLM_REPLAY_EXTRA_ARGS+=(
    --request-remote-adapter-map "${SLLM_REQUEST_REMOTE_ADAPTER_MAP}"
    --request-remote-endpoint "${REMOTE_ARTIFACT_STAGE_ENDPOINT}"
    --request-remote-timeout-s "${TIMEOUT_S}"
    --request-remote-bandwidth-mbps "${REMOTE_ARTIFACT_STAGE_BANDWIDTH_MBPS}"
  )
fi
if [[ -n "${REPLAY_MAX_REQUESTS}" && "${REPLAY_MAX_REQUESTS}" != "0" ]]; then
  SLLM_REPLAY_EXTRA_ARGS+=(--max-requests "${REPLAY_MAX_REQUESTS}")
fi
run_python_in_env sllm_head_official \
  "${ROOT_DIR}/scripts/replay_openai_trace.py" \
  --trace "${TRACE_PATH}" \
  --base-url "http://127.0.0.1:8343" \
  --convert-chat-to-prompt \
  --prompt-guard-tokenizer-model "${PROMPT_GUARD_TOKENIZER_MODEL}" \
  --prompt-guard-max-model-len "${PROMPT_GUARD_MAX_MODEL_LEN}" \
  --prompt-guard-max-input-len "${PROMPT_GUARD_MAX_INPUT_LEN}" \
  --prompt-guard-max-output-tokens-cap "${PROMPT_GUARD_MAX_OUTPUT_TOKENS_CAP}" \
  --sleep-scale "${SLEEP_SCALE}" \
  --timeout-s "${TIMEOUT_S}" \
  --empty-success-retries "${EMPTY_SUCCESS_RETRIES}" \
  --empty-success-retry-delay-s "${EMPTY_SUCCESS_RETRY_DELAY_S}" \
  --base-cost-usd "${BASE_COST_USD}" \
  --input-token-cost-usd "${INPUT_TOKEN_COST_USD}" \
  --output-token-cost-usd "${OUTPUT_TOKEN_COST_USD}" \
  --ttft-slo-ms "${TTFT_SLO_MS}" \
  --generation-seed "${GENERATION_SEED}" \
  --require-server-metrics \
  --label "${RUN_TAG}" \
  --output "${REPLAY_PATH}" \
  "${SLLM_REPLAY_EXTRA_ARGS[@]}"

VALIDATE_EXPECTED_TOTAL="${TOTAL_REQUESTS}"
if [[ -n "${REPLAY_MAX_REQUESTS}" && "${REPLAY_MAX_REQUESTS}" != "0" ]]; then
  VALIDATE_EXPECTED_TOTAL="${REPLAY_MAX_REQUESTS}"
fi
run_python_in_env sllm_head_official \
  "${ROOT_DIR}/scripts/validate_replay_results.py" \
  --system "ServerlessLLM" \
  --replay "${REPLAY_PATH}" \
  --expected-total "${VALIDATE_EXPECTED_TOTAL}"

echo "[post] Summarizing replay into the shared paper metric schema"
SUMMARY_PREDEPLOY_STARTUP_SEC="$(
  run_python_in_env sllm_head_official - "${REMOTE_STAGE_SEC}" "${POST_DEPLOY_WAIT_S}" <<'PY'
import sys

remote_stage = float(sys.argv[1] or 0.0)
post_deploy_wait = float(sys.argv[2] or 0.0)
print(f"{remote_stage + post_deploy_wait:.6f}")
PY
)"
run_python_in_env sllm_head_official \
  "${ROOT_DIR}/scripts/summarize_serverlessllm_replay.py" \
  --main-repo "${MAIN_REPO}" \
  --config "${CONFIG_PATH}" \
  --model-profile "${MODEL_PROFILE}" \
  --dataset-profile "${DATASET_PROFILE}" \
  --workload-profile "${WORKLOAD_PROFILE}" \
  --trace "${TRACE_PATH}" \
  --adapter-subset "${ADAPTER_SUBSET_PATH}" \
  --replay "${REPLAY_PATH}" \
  --deploy "${DEPLOY_PATH}" \
  --predeploy-startup-sec "${SUMMARY_PREDEPLOY_STARTUP_SEC}" \
  --scenario-name "serverlessllm_fair" \
  --backend-label "serverlessllm_${BACKEND}" \
  --output "${SUMMARY_PATH}"

echo "trace  -> ${TRACE_PATH}"
echo "deploy -> ${DEPLOY_PATH}"
echo "replay -> ${REPLAY_PATH}"
echo "summary -> ${SUMMARY_PATH}"
