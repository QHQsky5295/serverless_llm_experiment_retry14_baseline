#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/qhq/serverless_llm_experiment_retry14_baseline
RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1100_20261107/7b_busy_service_rate_w2
PROFILE=W2
TAG=d1100_7b_busy_service_rate_w2
NVME=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1100_20261107/7b_nvme_busy_service_rate_w2
HOST=/dev/shm/tc-d1100-7b-busy-service-rate-w2
# Reuse the exact D1100 configuration for missing cross-workload points.
# Explicit paths prevent accidental reuse of the original result or cache.
if (( $# )); then
  if (( $# != 5 )); then
    echo 'usage: runner [W0|W1|W2 RAW TAG NVME HOST]' >&2; exit 2
  fi
  PROFILE=$1; RAW=$2; TAG=$3; NVME=$4; HOST=$5
fi
case "$PROFILE" in W0|W1|W2) ;; *) exit 2 ;; esac
[[ "$RAW" == /* && "$NVME" == /* && "$HOST" == /dev/shm/* ]]
[[ "$TAG" =~ ^[A-Za-z0-9_]+$ ]]
for path in "$RAW" "$NVME" "$HOST"; do
  if [[ -e "$path" || -L "$path" ]]; then
    echo "refusing existing result/cache: $path" >&2; exit 2
  fi
done
cd "$ROOT"
mkdir -p "$RAW"
exec >"$RAW/launcher.log" 2>&1

REMOTE=primelora-artifact-174
REMOTE_UNIT=primelora-artifact-tc-7b.service
REMOTE_TOKEN=/home/lab14/.config/primelora-tc-d75/artifact.token
REMOTE_INVOCATION=
REMOTE_MONITOR_PID=
stop_remote() {
  if [[ -n "$REMOTE_INVOCATION" ]]; then
    ssh "$REMOTE" bash -s -- "$REMOTE_UNIT" "$REMOTE_INVOCATION" > "$RAW/remote_service_stop.txt" 2>&1 <<'SH' || true
if [[ "$(systemctl --user show "$1" -p InvocationID --value)" == "$2" ]]; then
  systemctl --user stop "$1"
fi
systemctl --user show "$1" -p ActiveState -p MainPID -p InvocationID
SH
  fi
  if [[ -n "$REMOTE_MONITOR_PID" ]]; then
    wait "$REMOTE_MONITOR_PID" || true
  fi
}
trap stop_remote EXIT

[[ "$(ssh "$REMOTE" "systemctl --user show $REMOTE_UNIT -p ActiveState --value")" == inactive ]]
ssh "$REMOTE" "systemctl --user start $REMOTE_UNIT && systemctl --user is-active $REMOTE_UNIT" > "$RAW/remote_service_start.txt"
REMOTE_INVOCATION=$(ssh "$REMOTE" "systemctl --user show $REMOTE_UNIT -p InvocationID --value")
[[ "$REMOTE_INVOCATION" =~ ^[a-f0-9]{32}$ ]]
ssh "$REMOTE" bash -s -- "$REMOTE_UNIT" "$REMOTE_INVOCATION" \
  < scripts/ieee_tc/monitor_remote_artifact_service.sh > "$RAW/remote_monitor.log" 2>&1 &
REMOTE_MONITOR_PID=$!
for attempt in $(seq 1 30); do
  TOKEN=$(ssh "$REMOTE" "cat $REMOTE_TOKEN")
  if NO_PROXY=192.168.4.174,10.199.227.174,127.0.0.1,localhost,::1 \
     no_proxy=192.168.4.174,10.199.227.174,127.0.0.1,localhost,::1 \
     curl -fsS --max-time 3 -H "Authorization: Bearer $TOKEN" \
     http://192.168.4.174:18081/health > "$RAW/remote_pre_health.json"; then
    break
  fi
  if [[ "$attempt" == 30 ]]; then
    echo 'remote artifact health did not become ready' >&2
    exit 1
  fi
  sleep 1
done
python3 - "$RAW/remote_pre_health.json" <<'PY'
import json, sys
row=json.load(open(sys.argv[1]))
assert row.get('ok') is True
assert row.get('delivery_mode') == 'prepublished_gzip_v1'
assert row.get('timing_contract') == 'artifact_timing_v2'
print(json.dumps({'remote_contract_verified': True, 'delivery_mode': row['delivery_mode'],
                  'timing_contract': row['timing_contract']}, sort_keys=True))
PY

export PATH="/home/qhq/.venvs/primelora_vllm0300_tc_20260925/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.0
export FLASHINFER_NVCC=/usr/local/cuda-13.0/bin/nvcc
export MAX_JOBS=2
IFS= read -r PRIME_REMOTE_TOKEN < /home/qhq/.config/primelora-tc-d75/artifact.token
export PRIME_REMOTE_TOKEN
export FAASLORA_REMOTE_ARTIFACT_ENABLED=1
export FAASLORA_REMOTE_ARTIFACT_ENDPOINT=http://192.168.4.174:18081
export FAASLORA_REMOTE_ARTIFACT_DELIVERY_MODE=prepublished_gzip_v1
export FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY=uncached_background_v1
export PYTORCH_ALLOC_CONF=pinned_max_cached_size_mb:0,pinned_use_background_threads:True
export VLLM_CACHE_ROOT=/home/qhq/.cache/vllm
export FAASLORA_PYTHON=/home/qhq/.venvs/primelora_vllm0300_tc_20260925/bin/python
export FAASLORA_TC_QUALIFICATION=1
export FAASLORA_TC_EXTERNAL_REPLAY=1
export FAASLORA_TC_REPLAY_PROFILE="$PROFILE"
unset FAASLORA_TC_ADAPTER_MAP FAASLORA_TC_ADAPTER_MAP_SHA256
if [[ "$PROFILE" == W2 ]]; then
  export FAASLORA_TC_ADAPTER_MAP=/home/qhq/serverless_llm_experiment_retry14_baseline/paper_results/ieee_tc/inputs/20261015_7b_w2_adapter_map_v1.json
  export FAASLORA_TC_ADAPTER_MAP_SHA256=48c72e39bfff13874de70a73a8a7ca26ff7be5246d5ee5a3a59dfb06bd502300
fi
export FAASLORA_SHARED_TRACE_PATH=/home/qhq/serverless_llm_baselines/results/paper_experiments/11_remote_fair_main_local_sim_v4/20260512_123619_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1/shared_artifacts/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1_trace.json
export FAASLORA_SHARED_ADAPTER_SUBSET_PATH="${FAASLORA_SHARED_TRACE_PATH%_trace.json}_adapter_subset.json"
export FAASLORA_TRACE_ROLE=exploratory
export FAASLORA_FORMAL_RUN=0
export FAASLORA_RESULTS_TAG="$TAG"
export FAASLORA_OUTPUT_DIR="$RAW"
export FAASLORA_NVME_CACHE_DIR="$NVME"
export FAASLORA_HOST_CACHE_DIR="$HOST"
export FAASLORA_TC_LAUNCH_OUTPUT="$RAW/launch.json"
export FAASLORA_TC_PREDICTED_GROWTH_GIB=32
export FAASLORA_TC_NVML_BINDING=/home/qhq/anaconda3/envs/LLM_vllm0102/lib/python3.12/site-packages/pynvml.py
export FAASLORA_TC_NVML_SHA256=4251429c25f1615a4166f395d3c09fe0732bfb023864c4b7f12813373e5696ea
export FAASLORA_WORKER_START_TIMEOUT_S=600
export FAASLORA_KEEP_DEDICATED_WORKER_LOGS=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1

bash scripts/run_all_experiments_user_scope.sh \
  --config configs/ieee_tc/20261107_d1100_7b_busy_service_rate_w2.yaml \
  --scenario tc_ieee_full --full-stack
