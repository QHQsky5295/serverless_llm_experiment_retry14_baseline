#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SLINFER_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
PROJECT_BASE="${SLINFER_PROJECT_BASE:-${ROOT_DIR}/vendor_new_baselines/SLINFER_main_20260323}"
ENV_DIR="${SLINFER_ENV_DIR:-/home/qhq/anaconda3/envs/slinfer_official_20260612}"
MODEL_KEY="${SLINFER_MODEL_KEY:?set SLINFER_MODEL_KEY=3b|7b}"
STACK_PREFIX="${SLINFER_STACK_PREFIX:-slinfer_${MODEL_KEY}}"
LOG_DIR="${SLINFER_LOG_DIR:-${ROOT_DIR}/logs/${STACK_PREFIX}}"
READY_TIMEOUT_S="${SLINFER_READY_TIMEOUT_S:-900}"
STORE_MEM_POOL_SIZE_GB="${SLINFER_STORE_MEM_POOL_SIZE_GB:-20}"
MIN_AVAILABLE_MEMORY_GB="${SLINFER_MIN_AVAILABLE_MEMORY_GB:-32}"

case "${MODEL_KEY}" in
  3b)
    MODEL_TYPE="llama-3.2-3b"
    DEFAULT_WORKER_NUM=2
    MAX_WORKER_NUM=8
    ;;
  7b)
    MODEL_TYPE="llama-2-7b"
    DEFAULT_WORKER_NUM=1
    MAX_WORKER_NUM=4
    ;;
  *)
    echo "SLINFER_MODEL_KEY must be 3b or 7b" >&2
    exit 2
    ;;
esac

REQUESTED_WORKER_NUM="${SLINFER_WORKERS_PER_GPU:-0}"
if [[ "${REQUESTED_WORKER_NUM}" == "0" ]]; then
  WORKER_NUM="${DEFAULT_WORKER_NUM}"
else
  WORKER_NUM="${REQUESTED_WORKER_NUM}"
fi
if [[ ! "${WORKER_NUM}" =~ ^[1-9][0-9]*$ ]] \
  || (( WORKER_NUM > MAX_WORKER_NUM )); then
  echo \
    "SLINFER_WORKERS_PER_GPU must be an integer in [1, ${MAX_WORKER_NUM}] " \
    "for ${MODEL_KEY}; got ${WORKER_NUM}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"

available_memory_gb() {
  awk '/MemAvailable:/ {printf "%d\n", $2 / 1024 / 1024}' /proc/meminfo
}

require_memory_headroom() {
  local stage="$1"
  local available_gb
  available_gb="$(available_memory_gb)"
  if (( available_gb < MIN_AVAILABLE_MEMORY_GB )); then
    echo \
      "SLINFER memory guard failed at ${stage}: " \
      "${available_gb}GB available < ${MIN_AVAILABLE_MEMORY_GB}GB reserve" \
      >&2
    return 1
  fi
}

export PROJECT_BASE
export PATH="${ENV_DIR}/bin:${PATH}"
export CUDA_HOME="${ENV_DIR}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export VLLM_NO_USAGE_STATS=1

NVIDIA_LIBS=(
  "${ENV_DIR}/lib"
  "${ENV_DIR}/lib/python3.11/site-packages/nvidia/cublas/lib"
  "${ENV_DIR}/lib/python3.11/site-packages/nvidia/cuda_cupti/lib"
  "${ENV_DIR}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
  "${ENV_DIR}/lib/python3.11/site-packages/nvidia/cudnn/lib"
  "${ENV_DIR}/lib/python3.11/site-packages/nvidia/cufft/lib"
  "${ENV_DIR}/lib/python3.11/site-packages/nvidia/curand/lib"
  "${ENV_DIR}/lib/python3.11/site-packages/nvidia/cusolver/lib"
  "${ENV_DIR}/lib/python3.11/site-packages/nvidia/cusparse/lib"
  "${ENV_DIR}/lib/python3.11/site-packages/nvidia/nccl/lib"
  "${ENV_DIR}/lib/python3.11/site-packages/nvidia/nvjitlink/lib"
  "${ENV_DIR}/lib/python3.11/site-packages/nvidia/nvtx/lib"
  "/usr/lib/x86_64-linux-gnu"
)
export LD_LIBRARY_PATH="$(IFS=:; echo "${NVIDIA_LIBS[*]}")"

sessions=(
  "${STACK_PREFIX}_store"
  "${STACK_PREFIX}_gpu0"
  "${STACK_PREFIX}_gpu1"
  "${STACK_PREFIX}_gpu2"
  "${STACK_PREFIX}_gpu3"
  "${STACK_PREFIX}_gateway"
)
for session in "${sessions[@]}"; do
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "SLINFER tmux session already exists: ${session}" >&2
    exit 3
  fi
done

require_memory_headroom "preflight"

tmux new-session -d -s "${STACK_PREFIX}_store" \
  "export PATH='${PATH}' PROJECT_BASE='${PROJECT_BASE}' LD_LIBRARY_PATH='${LD_LIBRARY_PATH}'; \
   sllm-store-server --storage_path '${PROJECT_BASE}/gpu_models' \
   --mem_pool_size '${STORE_MEM_POOL_SIZE_GB}' \
   > '${LOG_DIR}/store.log' 2>&1"

store_deadline=$((SECONDS + READY_TIMEOUT_S))
while ! rg -q "Server listening on .*8073" "${LOG_DIR}/store.log" 2>/dev/null; do
  require_memory_headroom "model-store-startup"
  if (( SECONDS >= store_deadline )); then
    echo "timed out waiting for SLINFER model store" >&2
    exit 70
  fi
  sleep 1
done

require_memory_headroom "post-model-store"

for gpu in 0 1 2 3; do
  base_port=$((8000 + gpu * 100))
  tmux new-session -d -s "${STACK_PREFIX}_gpu${gpu}" \
    "export PATH='${PATH}' PROJECT_BASE='${PROJECT_BASE}' LD_LIBRARY_PATH='${LD_LIBRARY_PATH}' \
     OMP_NUM_THREADS='${OMP_NUM_THREADS}' VLLM_NO_USAGE_STATS=1; \
     cd '${PROJECT_BASE}/SLINFER_core/tools'; \
     python vllm_batch_starter.py --model '${MODEL_TYPE}' --device gpu \
       --worker_num '${WORKER_NUM}' --port '${base_port}' --gpu '${gpu}' \
       > '${LOG_DIR}/gpu${gpu}_wrapper.log' 2>&1"
done

worker_deadline=$((SECONDS + READY_TIMEOUT_S))
while true; do
  all_workers_ready=1
  for gpu in 0 1 2 3; do
    base_port=$((8000 + gpu * 100))
    for ((worker = 0; worker < WORKER_NUM; worker++)); do
      if ! curl -sf "http://127.0.0.1:$((base_port + worker))/health" \
        >/dev/null; then
        all_workers_ready=0
      fi
    done
    if ! tmux has-session -t "${STACK_PREFIX}_gpu${gpu}" 2>/dev/null; then
      echo "SLINFER GPU ${gpu} worker wrapper exited before readiness" >&2
      exit 73
    fi
  done
  if (( all_workers_ready == 1 )); then
    break
  fi
  require_memory_headroom "worker-api-startup"
  if (( SECONDS >= worker_deadline )); then
    echo "timed out waiting for SLINFER worker APIs" >&2
    exit 74
  fi
  sleep 2
done

tmux new-session -d -s "${STACK_PREFIX}_gateway" \
  "export PATH='${PATH}' PROJECT_BASE='${PROJECT_BASE}' LD_LIBRARY_PATH='${LD_LIBRARY_PATH}'; \
   cd '${PROJECT_BASE}/SLINFER_core/scheduler'; \
   python gateway.py > '${LOG_DIR}/gateway.log' 2>&1"

gateway_deadline=$((SECONDS + READY_TIMEOUT_S))
while ! curl -sf -X POST http://127.0.0.1:7000/get_config >/dev/null; do
  require_memory_headroom "worker-gateway-startup"
  if (( SECONDS >= gateway_deadline )); then
    echo "timed out waiting for SLINFER gateway" >&2
    exit 71
  fi
  if ! tmux has-session -t "${STACK_PREFIX}_gateway" 2>/dev/null; then
    echo "SLINFER gateway exited before becoming ready" >&2
    exit 72
  fi
  sleep 2
done

require_memory_headroom "stack-ready"
echo "SLINFER ${MODEL_KEY} GPU-only stack is ready at http://127.0.0.1:7000."
