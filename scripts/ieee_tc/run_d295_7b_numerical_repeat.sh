#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
ROOT=/home/qhq/serverless_llm_experiment_retry14_baseline
BLOCK=${1:?explicit block 1, 2 or 3 required}
case "$BLOCK" in 1|2|3) ;; *) exit 2 ;; esac
ATTEMPT=${2:?explicit fresh attempt number required}
[[ "$ATTEMPT" =~ ^[1-9][0-9]*$ ]]
RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d295_20261006/block_${BLOCK}_attempt_${ATTEMPT}
test ! -e "$RAW"
mkdir -p "$RAW"
exec >"$RAW/launcher.log" 2>&1
cd "$ROOT"
unset FAASLORA_REMOTE_ARTIFACT_ENABLED FAASLORA_REMOTE_ARTIFACT_ENDPOINT
unset FAASLORA_REMOTE_ARTIFACT_DELIVERY_MODE PRIME_REMOTE_TOKEN
unset VLLM_BATCH_INVARIANT
export PATH="/home/qhq/.venvs/primelora_vllm0300_tc_20260925/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.0
export FLASHINFER_NVCC=/usr/local/cuda-13.0/bin/nvcc
export MAX_JOBS=2
export FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY=uncached_background_v1
export PYTORCH_ALLOC_CONF=pinned_max_cached_size_mb:0,pinned_use_background_threads:True
export VLLM_CACHE_ROOT=/home/qhq/.cache/vllm
export FAASLORA_KEEP_DEDICATED_WORKER_LOGS=1
export FAASLORA_WORKER_START_TIMEOUT_S=600
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export FAASLORA_TC_NVML_BINDING=/home/qhq/anaconda3/envs/LLM_vllm0102/lib/python3.12/site-packages/pynvml.py
export FAASLORA_TC_NVML_SHA256=4251429c25f1615a4166f395d3c09fe0732bfb023864c4b7f12813373e5696ea
git rev-parse HEAD > "$RAW/source_commit.txt"
sha256sum scripts/ieee_tc_preflight.py scripts/run_all_experiments.py \
  configs/ieee_tc/20261006_d295_7b_numerical_repeat.yaml \
  docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md \
  /home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md > "$RAW/sources.sha256"
IFS= read -r UNIT_ID < /proc/sys/kernel/random/uuid
exec systemd-run --user --scope --collect --unit="primelora-tc-aux-${UNIT_ID//-/}.scope" \
  -p MemoryHigh=3G -p MemoryMax=4G -p MemorySwapMax=0 \
  taskset -c 2,3,26,27 /usr/bin/python3 scripts/ieee_tc_preflight.py gated-launch \
  --output "$RAW/launch.json" --predicted-growth-gib 2 \
  --exec /home/qhq/.venvs/primelora_vllm0300_tc_20260925/bin/python \
  "$ROOT/scripts/ieee_tc_preflight.py" backend-model-check \
  --runtime-receipt "$ROOT/results/ieee_tc/p2_backend_qualification/runtime_20260926/cuda_import_attempt3.json" \
  --config "$ROOT/configs/ieee_tc/20261006_d295_7b_numerical_repeat.yaml" \
  --model-profile llama2_7b_main_v2_publicmix \
  --replay-trace /home/qhq/serverless_llm_baselines/results/paper_experiments/11_remote_fair_main_local_sim_v4/20260512_123619_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1/shared_artifacts/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1_trace.json \
  --request-count 100 --qualification-mode native_numerical_repeat \
  --output "$RAW/numerical_repeat.json"
