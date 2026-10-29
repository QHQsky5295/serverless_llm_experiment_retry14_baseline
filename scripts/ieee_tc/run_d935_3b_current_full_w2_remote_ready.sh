#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/qhq/serverless_llm_experiment_retry14_baseline
RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d935_20261029/3b_current_full_w2_remote_ready
cd "$ROOT"
mkdir -p "$RAW"
exec >"$RAW/launcher.log" 2>&1

export PATH="/home/qhq/.venvs/primelora_vllm0300_tc_20260925/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.0
export FLASHINFER_NVCC=/usr/local/cuda-13.0/bin/nvcc
export MAX_JOBS=2
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
export FAASLORA_TC_REPLAY_PROFILE=W2
export FAASLORA_TC_ADAPTER_MAP="$ROOT/paper_results/ieee_tc/inputs/20261018_3b_w2_adapter_map_v1.json"
export FAASLORA_TC_ADAPTER_MAP_SHA256=28c97dc7c1ffcbad49801ec19bec2a9c4cf9c4915c88a7e5163e223ece315e23
export FAASLORA_SHARED_TRACE_PATH=/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260509_134259_llama32_3b_main_s8/shared_artifacts/llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_trace.json
export FAASLORA_SHARED_ADAPTER_SUBSET_PATH="${FAASLORA_SHARED_TRACE_PATH%_trace.json}_adapter_subset.json"
export FAASLORA_TRACE_ROLE=exploratory
export FAASLORA_FORMAL_RUN=0
export FAASLORA_RESULTS_TAG=d935_3b_current_full_w2_remote_ready
export FAASLORA_OUTPUT_DIR="$RAW"
export FAASLORA_NVME_CACHE_DIR=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d935_20261029/3b_nvme_current_full_w2_remote_ready
export FAASLORA_HOST_CACHE_DIR=/dev/shm/tc-d935-3b-current-full-w2-remote-ready
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

exec bash scripts/run_all_experiments_user_scope.sh \
  --config configs/ieee_tc/20261026_d886_3b_max3_full_w2.yaml \
  --scenario tc_ieee_full --full-stack
