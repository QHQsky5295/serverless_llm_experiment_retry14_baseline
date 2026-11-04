#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/qhq/serverless_llm_experiment_retry14_baseline
RUN_ID="${FAASLORA_D1000_RUN_ID:-d1000_20261101}"
RUN_TAG="${FAASLORA_D1000_RUN_TAG:-d1000_3b_route_confirmed_arrival_prefetch_w1}"
CONFIG="${FAASLORA_D1000_CONFIG:-configs/ieee_tc/20261101_d1000_3b_route_confirmed_arrival_prefetch_w1.yaml}"
SCENARIO="${FAASLORA_D1000_SCENARIO:-tc_ieee_full}"
case "$SCENARIO" in
  tc_ieee_full|tc_ieee_elastic_only|tc_ieee_elastic_locality) ;;
  *) echo 'Unknown owned IEEE scenario' >&2; exit 2 ;;
esac
RAW_ROOT="${FAASLORA_D1000_RAW_ROOT:-/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification}"
RAW="$RAW_ROOT/$RUN_ID/$RUN_TAG"
cd "$ROOT"
test -f "$CONFIG"
mkdir -p "$RAW_ROOT/$RUN_ID"
mkdir "$RAW"  # A frozen execution key must never truncate an earlier launcher log.
exec >"$RAW/launcher.log" 2>&1

export PATH="/home/qhq/.venvs/primelora_vllm0300_tc_20260925/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.0
export FLASHINFER_NVCC=/usr/local/cuda-13.0/bin/nvcc
export MAX_JOBS=2
export NO_PROXY="192.168.4.174,10.199.227.174,127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"
IFS= read -r PRIME_REMOTE_TOKEN < /home/qhq/.config/primelora-tc-d75/artifact.token
export PRIME_REMOTE_TOKEN
export FAASLORA_REMOTE_ARTIFACT_ENABLED=1
export FAASLORA_REMOTE_ARTIFACT_ENDPOINT=http://192.168.4.174:18080
export FAASLORA_REMOTE_ARTIFACT_DELIVERY_MODE=prepublished_gzip_v1
export FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY=uncached_background_v1
export PYTORCH_ALLOC_CONF=pinned_max_cached_size_mb:0,pinned_use_background_threads:True
export VLLM_CACHE_ROOT=/home/qhq/.cache/vllm
export FAASLORA_PYTHON=/home/qhq/.venvs/primelora_vllm0300_tc_20260925/bin/python
export FAASLORA_TC_QUALIFICATION=1
export FAASLORA_TC_EXTERNAL_REPLAY=1
# Select an already defined frozen view; never change the serving configuration.
# Historical callers remain W2. W0/W1 must not inherit W2's adapter rotation.
export FAASLORA_TC_REPLAY_PROFILE="${FAASLORA_D1000_WORKLOAD:-W2}"
case "$FAASLORA_TC_REPLAY_PROFILE" in
  W0|W1)
    unset FAASLORA_TC_ADAPTER_MAP FAASLORA_TC_ADAPTER_MAP_SHA256
    ;;
  W2)
    export FAASLORA_TC_ADAPTER_MAP="$ROOT/paper_results/ieee_tc/inputs/20261018_3b_w2_adapter_map_v1.json"
    export FAASLORA_TC_ADAPTER_MAP_SHA256=28c97dc7c1ffcbad49801ec19bec2a9c4cf9c4915c88a7e5163e223ece315e23
    ;;
  *) echo 'Unknown frozen full-workload view' >&2; exit 2 ;;
esac
export FAASLORA_SHARED_TRACE_PATH=/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260509_134259_llama32_3b_main_s8/shared_artifacts/llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_trace.json
export FAASLORA_SHARED_ADAPTER_SUBSET_PATH="${FAASLORA_SHARED_TRACE_PATH%_trace.json}_adapter_subset.json"
export FAASLORA_TRACE_ROLE=exploratory
export FAASLORA_FORMAL_RUN=0
export FAASLORA_RESULTS_TAG="$RUN_TAG"
export FAASLORA_OUTPUT_DIR="$RAW"
export FAASLORA_NVME_CACHE_DIR="$RAW_ROOT/$RUN_ID/${RUN_TAG}_nvme"
export FAASLORA_HOST_CACHE_DIR="/dev/shm/tc-$RUN_TAG"
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

# A diagnostic must select the same executable HOST reader as its parent run.
# Keep historical/default callers unchanged; record this opt-in in the run key.
case "${FAASLORA_D1000_HOST_DESCRIPTOR:-}" in
  '') ;;
  fresh_d747_v1)
    export FAASLORA_IEEE_HOST_INVENTORY_MODE=descriptor
    export FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d747_20261023/_ieee_native_host_descriptor.cpython-312-x86_64-linux-gnu.so
    export FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256=b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044
    ;;
  *) echo 'Unknown qualified HOST descriptor' >&2; exit 2 ;;
esac

if [[ -n "${FAASLORA_D1000_PREFIX_COUNT:-}" ]]; then
  export FAASLORA_TC_DIAGNOSTIC_PREFIX_COUNT="$FAASLORA_D1000_PREFIX_COUNT"
fi

case "${FAASLORA_D1000_NATIVE_PROFILE:-}" in
  '') ;;
  host_spans_v1)
    # Reuse the bounded native observer only for the registered prefix diagnostic.
    test "${FAASLORA_D1000_PREFIX_COUNT:-}" = 1000
    export FAASLORA_PYTHON="$ROOT/scripts/ieee_tc/profile_d328_controller_python.sh"
    ;;
  *) echo 'Unknown native diagnostic observer' >&2; exit 2 ;;
esac

case "${FAASLORA_D1000_LOCAL_FILE_PROFILE:-}" in
  '') ;;
  local_lock_spans_v1)
    # Observe the parent's actual in-process file owner, without native profiling.
    test "${FAASLORA_D1000_PREFIX_COUNT:-}" = 1000
    test -z "${FAASLORA_D1000_NATIVE_PROFILE:-}"
    export FAASLORA_FILE_STATE_PROCESS=0
    export FAASLORA_TC_FILE_OWNER_TIMING=local_lock_spans_v1
    unset FAASLORA_TC_CORE_TIMING FAASLORA_TC_CORE_STEP_TIMING
    unset FAASLORA_TC_HOST_INVENTORY_TIMING
    ;;
  *) echo 'Unknown local file diagnostic observer' >&2; exit 2 ;;
esac

exec bash scripts/run_all_experiments_user_scope.sh \
  --config "$CONFIG" \
  --scenario "$SCENARIO" --full-stack
