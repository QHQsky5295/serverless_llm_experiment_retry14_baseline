#!/usr/bin/env bash
set -euo pipefail

BASELINES_ROOT="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
MAIN_REPO="${SLLM_MAIN_REPO:-/home/qhq/serverless_llm_experiment_retry14_baseline}"
DLORA_REPO="${DLORA_REPO:-${BASELINES_ROOT}/vendor_new_baselines/dLoRA_artifact_main_20260519}"
DLORA_ENV="${DLORA_ENV:-dlora_medusa_clone_20260519}"
SECTION_ID="${DLORA_SECTION_ID:-15_new_serverless_baselines_remote_v1}"
QUEUE_ID="${DLORA_QUEUE_ID:-$(date +%Y%m%d_%H%M%S)_dlora_remote_formal_v1}"
RUN_ONLY="${DLORA_ONLY:-llama32_3b}"
RESULT_ROOT="${BASELINES_ROOT}/results/paper_experiments/${SECTION_ID}"
LOG_ROOT="${BASELINES_ROOT}/results/logs/new_serverless_baselines_remote_v1/dlora/formal/${QUEUE_ID}"
CONDA_BIN="${CONDA_BIN:-/home/qhq/anaconda3/bin/conda}"
DLORA_MIGRATION_TYPE="${DLORA_MIGRATION_TYPE:-3}"
case "${DLORA_MIGRATION_TYPE}" in
  1) DLORA_MIGRATION_LABEL="dispatch_only" ;;
  2) DLORA_MIGRATION_LABEL="dispatch_mig" ;;
  3) DLORA_MIGRATION_LABEL="period_mig" ;;
  *) DLORA_MIGRATION_LABEL="migration_${DLORA_MIGRATION_TYPE}" ;;
esac
DLORA_ROUTING_POLICY="${DLORA_ROUTING_POLICY:-dlora_${DLORA_MIGRATION_LABEL}}"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"

REMOTE_NO_PROXY_HOSTS="192.168.4.174,10.199.227.174,127.0.0.1,localhost,::1"
if [[ -n "${NO_PROXY:-}" ]]; then
  export NO_PROXY="${NO_PROXY},${REMOTE_NO_PROXY_HOSTS}"
else
  export NO_PROXY="${REMOTE_NO_PROXY_HOSTS}"
fi
export no_proxy="${NO_PROXY}"
export HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy=

wait_for_http() {
  local url="$1"
  local timeout_s="$2"
  local server_pid="${3:-}"
  python - "$url" "$timeout_s" "$server_pid" <<'PY'
import os
import sys
import time
import urllib.request

url = sys.argv[1].rstrip("/") + "/openapi.json"
deadline = time.time() + float(sys.argv[2])
pid = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else 0
while time.time() < deadline:
    if pid > 0:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print(f"[ERROR] server process exited before {url} became ready", file=sys.stderr)
            raise SystemExit(1)
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            if 200 <= int(resp.status) < 500:
                print(f"[ready] {url} status={resp.status}", flush=True)
                raise SystemExit(0)
    except Exception:
        time.sleep(2)
print(f"[ERROR] timed out waiting for {url}", file=sys.stderr)
raise SystemExit(1)
PY
}

stop_dlora_runtime() {
  local server_pid="${1:-}"
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" >/dev/null 2>&1; then
    kill "${server_pid}" >/dev/null 2>&1 || true
    sleep 5
    kill -9 "${server_pid}" >/dev/null 2>&1 || true
  fi
  env PYTHONNOUSERSITE=1 "${CONDA_BIN}" run -n "${DLORA_ENV}" ray stop --force >/dev/null 2>&1 || true
}

write_adapter_map() {
  local subset_path="$1"
  local output_path="$2"
  python - "$subset_path" "$output_path" <<'PY'
import json
import sys
from pathlib import Path

subset = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
adapters = subset.get("adapters") or []
mapping = {str(item["id"]): str(idx) for idx, item in enumerate(adapters)}
Path(sys.argv[2]).write_text(json.dumps(mapping, indent=2), encoding="utf-8")
PY
}

write_deploy_json() {
  local output_path="$1"
  local model_profile="$2"
  local model_path="$3"
  local num_groups="$4"
  local tensor_parallel="$5"
  local gpu_ids="$6"
  python - "$output_path" "$model_profile" "$model_path" "$num_groups" "$tensor_parallel" "$gpu_ids" <<'PY'
import json
import sys
from pathlib import Path

num_groups = int(sys.argv[4])
tp = int(sys.argv[5])
gpu_ids = [x for x in sys.argv[6].split(",") if x != ""]
payload = {
    "system": "dlora",
    "model_profile": sys.argv[2],
    "model_path": sys.argv[3],
    "runtime_gpu_count": max(1, len(gpu_ids)),
    "num_gpus": max(1, len(gpu_ids)),
    "data_parallel_replicas": num_groups,
    "dp": num_groups,
    "tp": tp,
    "tensor_parallel_size": tp,
    "parallelism_topology": f"dlora_num_groups{num_groups}_tp{tp}",
    "gpu_ids": gpu_ids,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

write_manifest() {
  local output_path="$1"
  local label="$2"
  local model_profile="$3"
  local workload_profile="$4"
  local run_tag="$5"
  local source_round="$6"
  local trace_path="$7"
  local original_subset_path="$8"
  local materialized_subset_path="$9"
  local endpoint="${10}"
  local replay_path="${11}"
  local summary_path="${12}"
  local deploy_path="${13}"
  local server_log="${14}"
  local replay_log="${15}"
  local materialize_log="${16}"
  local startup_sec="${17}"
  local predeploy_sec="${18}"
  python - "$output_path" "$label" "$model_profile" "$workload_profile" "$run_tag" \
    "$source_round" "$trace_path" "$original_subset_path" "$materialized_subset_path" \
    "$endpoint" "$replay_path" "$summary_path" "$deploy_path" "$server_log" "$replay_log" \
    "$materialize_log" "$startup_sec" "$predeploy_sec" "$QUEUE_ID" "$DLORA_REPO" \
    "$DLORA_ENV" "$DLORA_MIGRATION_TYPE" "$DLORA_ROUTING_POLICY" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "metric_schema_version": "e2e_v3",
    "system": "dlora",
    "queue_id": sys.argv[19],
    "label": sys.argv[2],
    "model_profile": sys.argv[3],
    "dataset_profile": "azure_sharegpt_rep4000",
    "workload_profile": sys.argv[4],
    "run_tag": sys.argv[5],
    "total_requests": 4000,
    "selected_num_adapters": 500,
    "sampling_seed": 42,
    "source_round": sys.argv[6],
    "shared_trace_path": sys.argv[7],
    "shared_adapter_subset_path": sys.argv[8],
    "materialized_adapter_subset_path": sys.argv[9],
    "remote_artifact_endpoint": sys.argv[10],
    "replay_path": sys.argv[11],
    "summary_path": sys.argv[12],
    "deploy_path": sys.argv[13],
    "server_log": sys.argv[14],
    "replay_log": sys.argv[15],
    "materialize_log": sys.argv[16],
    "static_startup_sec": float(sys.argv[17]),
    "predeploy_startup_sec": float(sys.argv[18]),
    "dlora_repo": sys.argv[20],
    "dlora_env": sys.argv[21],
    "dlora_exec_type": 3,
    "dlora_migration_type": int(sys.argv[22]),
    "routing_policy": sys.argv[23],
    "adaptation_boundary": (
        "hardware/workload wrapper only: real remote materialization, adapter-id map, "
        "real PEFT loader, e2e_v3 replay compatibility, and selected-adapter LoRA "
        "einsum to avoid materializing unused one-hot adapter columns on 24GB GPUs; "
        "dLoRA scheduling, migration, and adapter orchestration are not replaced"
    ),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

run_one() {
  local label="$1"
  local model_profile="$2"
  local workload_profile="$3"
  local run_tag="$4"
  local source_round="$5"
  local endpoint="$6"
  local model_path="$7"
  local port="$8"
  local num_groups="$9"
  local tensor_parallel="${10}"
  local gpu_ids="${11}"
  local gpu_memory_utilization="${12}"
  local max_num_seqs="${13}"
  local max_num_batched_tokens="${14}"
  local startup_timeout_s="${15}"

  local round_dir="${RESULT_ROOT}/${QUEUE_ID}_${label}_${run_tag}_dlora"
  local shared_dir="${round_dir}/shared_inputs"
  local raw_dir="${round_dir}/raw/replay"
  local remote_cache="${round_dir}/remote_cache/dlora"
  local trace_path="${source_round}/shared_artifacts/${run_tag}_trace.json"
  local original_subset_path="${source_round}/shared_artifacts/${run_tag}_adapter_subset.json"
  local materialized_subset_path="${shared_dir}/${run_tag}_dlora_remote_adapter_subset.json"
  local adapter_map_path="${shared_dir}/${run_tag}_dlora_adapter_value_map.json"
  local deploy_path="${shared_dir}/${run_tag}_dlora_deploy.json"
  local replay_path="${raw_dir}/${run_tag}_dlora_replay.json"
  local summary_path="${raw_dir}/${run_tag}_dlora_summary.json"
  local materialize_log="${LOG_ROOT}/${label}_remote_materialize.log"
  local server_log="${LOG_ROOT}/${label}_server.log"
  local replay_log="${LOG_ROOT}/${label}_replay.log"
  local summary_log="${LOG_ROOT}/${label}_summary.log"

  mkdir -p "${shared_dir}" "${raw_dir}" "${remote_cache}"

  if [[ -f "${summary_path}" ]]; then
    python "${BASELINES_ROOT}/scripts/validate_replay_results.py" \
      --system "dLoRA" \
      --replay "${replay_path}" \
      --expected-total 4000 >/dev/null 2>&1 && {
        echo "[skip] ${label}: existing valid summary ${summary_path}"
        return 0
      }
  fi

  echo "[run] ${label}"
  echo "      run_tag=${run_tag}"
  echo "      source_round=${source_round}"
  echo "      endpoint=${endpoint}"
  echo "      output_round=${round_dir}"
  echo "      dlora_topology=num_groups${num_groups}_tp${tensor_parallel}_gpus${gpu_ids}_${DLORA_MIGRATION_LABEL}"

  local pre_t0 pre_t1 predeploy_sec
  pre_t0="$(date +%s.%N)"
  local remote_force_args=()
  if [[ "${DLORA_REMOTE_FORCE:-0}" == "1" ]]; then
    remote_force_args=(--force)
  fi

  python "${BASELINES_ROOT}/scripts/materialize_remote_adapter_subset.py" \
    --main-repo "${MAIN_REPO}" \
    --adapter-subset "${original_subset_path}" \
    --endpoint "${endpoint}" \
    --output-dir "${remote_cache}" \
    --output-subset "${materialized_subset_path}" \
    --workers "${DLORA_REMOTE_WORKERS:-4}" \
    --timeout-s "${DLORA_REMOTE_TIMEOUT_S:-600}" \
    "${remote_force_args[@]}" 2>&1 | tee "${materialize_log}"
  pre_t1="$(date +%s.%N)"
  predeploy_sec="$(python - "$pre_t0" "$pre_t1" <<'PY'
import sys
print(max(0.0, float(sys.argv[2]) - float(sys.argv[1])))
PY
)"

  write_adapter_map "${materialized_subset_path}" "${adapter_map_path}"
  write_deploy_json "${deploy_path}" "${model_profile}" "${model_path}" "${num_groups}" "${tensor_parallel}" "${gpu_ids}"

  env PYTHONNOUSERSITE=1 "${CONDA_BIN}" run -n "${DLORA_ENV}" ray stop --force >/dev/null 2>&1 || true

  local server_pid startup_t0 startup_t1 startup_sec
  startup_t0="$(date +%s.%N)"
  (
    cd "${DLORA_REPO}"
    env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${gpu_ids}" \
      RAY_DEDUP_LOGS=0 TOKENIZERS_PARALLELISM=true \
      "${CONDA_BIN}" run --no-capture-output -n "${DLORA_ENV}" \
        python -m vllm.entrypoints.api_server \
          --model "${model_path}" \
          --tokenizer "${model_path}" \
          --swap-space "${DLORA_SWAP_SPACE_GB:-8}" \
          --disable-log-requests \
          --num-models 500 \
          --num-groups "${num_groups}" \
          -tp "${tensor_parallel}" \
          --worker-use-ray \
          --engine-use-ray \
          --exec-type 3 \
          --migration-type "${DLORA_MIGRATION_TYPE}" \
          --host 127.0.0.1 \
          --port "${port}" \
          --no-use-dummy-weights \
          --lora-adapter-subset "${materialized_subset_path}" \
          --max-r "${DLORA_MAX_R:-16}" \
          --gpu-capacity "${DLORA_GPU_CAPACITY:-8}" \
          --max-num-seqs "${max_num_seqs}" \
          --max-num-batched-tokens "${max_num_batched_tokens}" \
          --gpu-memory-utilization "${gpu_memory_utilization}"
  ) >"${server_log}" 2>&1 &
  server_pid=$!
  trap 'stop_dlora_runtime "${server_pid:-}"' EXIT

  wait_for_http "http://127.0.0.1:${port}" "${startup_timeout_s}" "${server_pid}"
  startup_t1="$(date +%s.%N)"
  startup_sec="$(python - "$startup_t0" "$startup_t1" <<'PY'
import sys
print(max(0.0, float(sys.argv[2]) - float(sys.argv[1])))
PY
)"

  env PYTHONNOUSERSITE=1 "${CONDA_BIN}" run --no-capture-output -n "${DLORA_ENV}" \
    python "${BASELINES_ROOT}/scripts/replay_openai_trace.py" \
      --trace "${trace_path}" \
      --base-url "http://127.0.0.1:${port}" \
      --output "${replay_path}" \
      --sleep-scale "${DLORA_SLEEP_SCALE:-1.0}" \
      --timeout-s "${DLORA_REPLAY_TIMEOUT_S:-7200}" \
      --label "dlora_${label}_${DLORA_MIGRATION_LABEL}_remote_formal" \
      --endpoint-path "/generate" \
      --convert-chat-to-prompt \
      --prompt-guard-tokenizer-model "${model_path}" \
      --prompt-guard-max-model-len "${DLORA_PROMPT_GUARD_MAX_MODEL_LEN:-1024}" \
      --prompt-guard-max-input-len "${DLORA_PROMPT_GUARD_MAX_INPUT_LEN:-0}" \
      --prompt-guard-max-output-tokens-cap "${DLORA_OUTPUT_TOKENS_CAP:-0}" \
      --adapter-source-field "adapter_id" \
      --adapter-target-field "model_id" \
      --adapter-value-map "${adapter_map_path}" \
      --drop-body-field "model" \
      --drop-body-field "request_id" \
      --drop-body-field "lora_adapter_name" \
      --abort-after-failures "${DLORA_ABORT_AFTER_FAILURES:-1}" \
      --abort-failures-min-done "${DLORA_ABORT_FAILURES_MIN_DONE:-1}" \
      2>&1 | tee "${replay_log}"

  python "${BASELINES_ROOT}/scripts/validate_replay_results.py" \
    --system "dLoRA" \
    --replay "${replay_path}" \
    --expected-total 4000

  python "${BASELINES_ROOT}/scripts/summarize_serverlessllm_replay.py" \
    --main-repo "${MAIN_REPO}" \
    --replay "${replay_path}" \
    --trace "${trace_path}" \
    --adapter-subset "${materialized_subset_path}" \
    --deploy "${deploy_path}" \
    --model-profile "${model_profile}" \
    --dataset-profile "azure_sharegpt_rep4000" \
    --workload-profile "${workload_profile}" \
    --output "${summary_path}" \
    --scenario-name "dlora_${label}_${DLORA_MIGRATION_LABEL}_remote_formal" \
    --baseline-type "vllm" \
    --backend-label "dlora" \
    --system-name "dLoRA" \
    --instance-mode "static_dlora" \
    --routing-policy "${DLORA_ROUTING_POLICY}" \
    --static-startup-sec "${startup_sec}" \
    --predeploy-startup-sec "${predeploy_sec}" \
    2>&1 | tee "${summary_log}"

  write_manifest "${round_dir}/MANIFEST.json" "${label}" "${model_profile}" "${workload_profile}" \
    "${run_tag}" "${source_round}" "${trace_path}" "${original_subset_path}" \
    "${materialized_subset_path}" "${endpoint}" "${replay_path}" "${summary_path}" \
    "${deploy_path}" "${server_log}" "${replay_log}" "${materialize_log}" \
    "${startup_sec}" "${predeploy_sec}"

  stop_dlora_runtime "${server_pid}"
  trap - EXIT
}

if [[ "${RUN_ONLY}" == "all" || "${RUN_ONLY}" == "llama32_3b" ]]; then
  run_one \
    "llama32_3b" \
    "llama32_3b_main_modelscope" \
    "llama32_3b_auto500_formal4000_s8" \
    "llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1" \
    "${BASELINES_ROOT}/results/paper_experiments/12_remote_fair_main_real_remote_v1/20260513_160342_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1" \
    "http://192.168.4.174:18080" \
    "${MAIN_REPO}/models/LLM-Research--Llama-3.2-3B-Instruct" \
    "${DLORA_LLAMA32_PORT:-18280}" \
    "${DLORA_LLAMA32_NUM_GROUPS:-4}" \
    "${DLORA_LLAMA32_TP:-1}" \
    "${DLORA_LLAMA32_GPU_IDS:-0,1,2,3}" \
    "${DLORA_LLAMA32_GPU_MEMORY_UTILIZATION:-0.57}" \
    "${DLORA_LLAMA32_MAX_NUM_SEQS:-1}" \
    "${DLORA_LLAMA32_MAX_NUM_BATCHED_TOKENS:-1024}" \
    "${DLORA_LLAMA32_STARTUP_TIMEOUT_S:-1800}"
fi

if [[ "${RUN_ONLY}" == "all" || "${RUN_ONLY}" == "llama2_7b" ]]; then
  run_one \
    "llama2_7b" \
    "llama2_7b_main_v2_publicmix" \
    "llama2_7b_auto500_formal4000_s8" \
    "llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1" \
    "${BASELINES_ROOT}/results/paper_experiments/12_remote_fair_main_real_remote_v1/20260513_012813_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1" \
    "http://192.168.4.174:18081" \
    "${DLORA_LLAMA2_MODEL_PATH:-/home/qhq/serverless_llm_experiment/models/meta-llama--Llama-2-7b-hf}" \
    "${DLORA_LLAMA2_PORT:-18281}" \
    "${DLORA_LLAMA2_NUM_GROUPS:-2}" \
    "${DLORA_LLAMA2_TP:-2}" \
    "${DLORA_LLAMA2_GPU_IDS:-0,1,2,3}" \
    "${DLORA_LLAMA2_GPU_MEMORY_UTILIZATION:-0.88}" \
    "${DLORA_LLAMA2_MAX_NUM_SEQS:-1}" \
    "${DLORA_LLAMA2_MAX_NUM_BATCHED_TOKENS:-256}" \
    "${DLORA_LLAMA2_STARTUP_TIMEOUT_S:-2400}"
fi

echo "[done] dLoRA true-remote formal queue completed: ${QUEUE_ID}"
