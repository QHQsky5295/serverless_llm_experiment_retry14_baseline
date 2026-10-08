#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/qhq/serverless_llm_experiment_retry14_baseline
RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d293_20261006/7b_prefetch_validation_100
CONFIG=configs/ieee_tc/20261006_d293_7b_prefetch_validation_100.yaml
TAG=d293_7b_prefetch_validation_100
SCENARIO=tc_ieee_prefetch_validation_100
PREFIX_COUNT=100
case "${1:-d293}" in
  d293) ;;
  d296)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d296_20261007/7b_lifecycle_validation_100
    CONFIG=configs/ieee_tc/20261007_d296_7b_lifecycle_validation_100.yaml
    TAG=d296_7b_lifecycle_validation_100
    ;;
  d297)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d297_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d297_7b_full_w0.yaml
    TAG=d297_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d298)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d298_20261007/7b_capacity_validation_100
    CONFIG=configs/ieee_tc/20261007_d298_7b_capacity_validation_100.yaml
    TAG=d298_7b_capacity_validation_100
    ;;
  d299)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d299_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d299_7b_full_w0.yaml
    TAG=d299_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d300)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d300_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d300_7b_full_w0.yaml
    TAG=d300_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d302)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d302_20261007/7b_frontier_validation_100
    CONFIG=configs/ieee_tc/20261007_d302_7b_frontier_validation_100.yaml
    TAG=d302_7b_frontier_validation_100
    SCENARIO=tc_ieee_frontier_validation_100
    ;;
  d303)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d303_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d303_7b_full_w0.yaml
    TAG=d303_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d308)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d308_20261007/7b_prefill_validation_100
    CONFIG=configs/ieee_tc/20261007_d308_7b_prefill_validation_100.yaml
    TAG=d308_7b_prefill_validation_100
    SCENARIO=tc_ieee_prefill_validation_100
    ;;
  d309)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d309_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d309_7b_full_w0.yaml
    TAG=d309_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d310)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d310_20261007/7b_prefill_validation_100
    CONFIG=configs/ieee_tc/20261007_d310_7b_prefill_validation_100.yaml
    TAG=d310_7b_prefill_validation_100
    SCENARIO=tc_ieee_prefill_validation_100
    ;;
  d311)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d311_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d311_7b_full_w0.yaml
    TAG=d311_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d312)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d312_20261007/7b_prefill_validation_100
    CONFIG=configs/ieee_tc/20261007_d312_7b_prefill_validation_100.yaml
    TAG=d312_7b_prefill_validation_100
    SCENARIO=tc_ieee_prefill_validation_100
    ;;
  d313)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d313_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d313_7b_full_w0.yaml
    TAG=d313_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d314)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d314_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d314_7b_full_w0.yaml
    TAG=d314_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d315)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d315_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d315_7b_full_w0.yaml
    TAG=d315_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d317)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d317_20261007/7b_copy_validation_100
    CONFIG=configs/ieee_tc/20261007_d317_7b_copy_validation_100.yaml
    TAG=d317_7b_copy_validation_100
    SCENARIO=tc_ieee_copy_validation_100
    ;;
  d318)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d318_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d318_7b_full_w0.yaml
    TAG=d318_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d322)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d322_20261007/7b_length_validation_100
    CONFIG=configs/ieee_tc/20261007_d322_7b_length_validation_100.yaml
    TAG=d322_7b_length_validation_100
    SCENARIO=tc_ieee_length_validation_100
    ;;
  d322-attempt2)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d322_20261007/7b_length_validation_100_attempt2
    CONFIG=configs/ieee_tc/20261007_d322_7b_length_validation_100_attempt2.yaml
    TAG=d322_7b_length_validation_100_attempt2
    SCENARIO=tc_ieee_length_validation_100
    ;;
  d323)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d323_20261007/7b_full_w0
    CONFIG=configs/ieee_tc/20261007_d323_7b_full_w0.yaml
    TAG=d323_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d326)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d326_20261008/7b_full_w0
    CONFIG=configs/ieee_tc/20261008_d326_7b_full_w0.yaml
    TAG=d326_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d328)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d328_20261008/7b_full_w0_prefix1
    CONFIG=configs/ieee_tc/20261008_d328_7b_full_w0_prefix1.yaml
    TAG=d328_7b_full_w0_prefix1
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=1000
    ;;
  d331)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d331_20261008/7b_planning_validation_100
    CONFIG=configs/ieee_tc/20261008_d331_7b_planning_validation_100.yaml
    TAG=d331_7b_planning_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d331-attempt2)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d331_20261008/7b_planning_validation_100_attempt2
    CONFIG=configs/ieee_tc/20261008_d331_7b_planning_validation_100_attempt2.yaml
    TAG=d331_7b_planning_validation_100_attempt2
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d332)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d332_20261008/7b_full_w0
    CONFIG=configs/ieee_tc/20261008_d332_7b_full_w0.yaml
    TAG=d332_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d334)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d334_20261008/7b_native_pending_validation_100
    CONFIG=configs/ieee_tc/20261008_d334_7b_native_pending_validation_100.yaml
    TAG=d334_7b_native_pending_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d335)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d335_20261008/7b_full_w0
    CONFIG=configs/ieee_tc/20261008_d335_7b_full_w0.yaml
    TAG=d335_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d336)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d336_20261008/7b_full_w0
    CONFIG=configs/ieee_tc/20261008_d336_7b_full_w0.yaml
    TAG=d336_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d340)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d340_20261008/7b_prebuild_validation_100
    CONFIG=configs/ieee_tc/20261008_d340_7b_prebuild_validation_100.yaml
    TAG=d340_7b_prebuild_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    export FAASLORA_IEEE_QUANTIZATION_PREBUILD_POLICY=on_demand_v1
    export HUMMING_DISABLE_PARALLEL_BUILD=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    ;;
  d341)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d341_20261008/7b_full_w0
    CONFIG=configs/ieee_tc/20261008_d341_7b_full_w0.yaml
    TAG=d341_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    export FAASLORA_IEEE_QUANTIZATION_PREBUILD_POLICY=on_demand_v1
    export HUMMING_DISABLE_PARALLEL_BUILD=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    ;;
  d345)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d345_20261008/7b_capacity_validation_100
    CONFIG=configs/ieee_tc/20261008_d345_7b_capacity_validation_100.yaml
    TAG=d345_7b_capacity_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d346)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d346_20261008/7b_full_w0
    CONFIG=configs/ieee_tc/20261008_d346_7b_full_w0.yaml
    TAG=d346_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d349)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d349_20261008/7b_full_w0
    CONFIG=configs/ieee_tc/20261008_d349_7b_full_w0.yaml
    TAG=d349_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d351)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d351_20261008/7b_full_w0_prefix1
    CONFIG=configs/ieee_tc/20261008_d351_7b_full_w0_prefix1.yaml
    TAG=d351_7b_full_w0_prefix1
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=1000
    ;;
  d352)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d352_20261008/7b_hit_validation_100
    CONFIG=configs/ieee_tc/20261008_d352_7b_hit_validation_100.yaml
    TAG=d352_7b_hit_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d353)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d353_20261008/7b_full_w0
    CONFIG=configs/ieee_tc/20261008_d353_7b_full_w0.yaml
    TAG=d353_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d355)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d355_20261008/7b_interleave_validation_100
    CONFIG=configs/ieee_tc/20261008_d355_7b_interleave_validation_100.yaml
    TAG=d355_7b_interleave_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d357)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d357_20261008/7b_interleave_validation_100
    CONFIG=configs/ieee_tc/20261008_d357_7b_interleave_validation_100.yaml
    TAG=d357_7b_interleave_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d358)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d358_20261008/7b_full_w0
    CONFIG=configs/ieee_tc/20261008_d358_7b_full_w0.yaml
    TAG=d358_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  d363)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d363_20261009/7b_file_demand_validation_100
    CONFIG=configs/ieee_tc/20261009_d363_7b_file_demand_validation_100.yaml
    TAG=d363_7b_file_demand_validation_100
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=100
    ;;
  d364)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d364_20261009/7b_full_w0
    CONFIG=configs/ieee_tc/20261009_d364_7b_full_w0.yaml
    TAG=d364_7b_full_w0
    SCENARIO=tc_ieee_full
    PREFIX_COUNT=
    ;;
  *) echo 'Unknown frozen campaign identity; no arbitrary output override' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || exit 2
cd "$ROOT"
[[ ! -e "$RAW" ]] || { echo "Refusing existing run: $RAW" >&2; exit 2; }
mkdir -p "$(dirname "$RAW")"
mkdir "$RAW"
set -o noclobber
exec >"$RAW/launcher.log" 2>&1
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
if [[ "${1:-d293}" == d328 || "${1:-d293}" == d351 ]]; then
  # Observer is enabled by this wrapper only AFTER the normal resource gate.
  export FAASLORA_PYTHON="$ROOT/scripts/ieee_tc/profile_d328_controller_python.sh"
fi
export FAASLORA_TC_QUALIFICATION=1
export FAASLORA_TC_EXTERNAL_REPLAY=1
export FAASLORA_TC_REPLAY_PROFILE=W0
if [[ -n "$PREFIX_COUNT" ]]; then
  export FAASLORA_TC_DIAGNOSTIC_PREFIX_COUNT="$PREFIX_COUNT"
else
  unset FAASLORA_TC_DIAGNOSTIC_PREFIX_COUNT
fi
export FAASLORA_TC_LAUNCH_OUTPUT="$RAW/launch.json"
export FAASLORA_TC_PREDICTED_GROWTH_GIB=32
export FAASLORA_TC_NVML_BINDING=/home/qhq/anaconda3/envs/LLM_vllm0102/lib/python3.12/site-packages/pynvml.py
export FAASLORA_TC_NVML_SHA256=4251429c25f1615a4166f395d3c09fe0732bfb023864c4b7f12813373e5696ea
export FAASLORA_SHARED_TRACE_PATH=/home/qhq/serverless_llm_baselines/results/paper_experiments/11_remote_fair_main_local_sim_v4/20260512_123619_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1/shared_artifacts/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1_trace.json
export FAASLORA_SHARED_ADAPTER_SUBSET_PATH="${FAASLORA_SHARED_TRACE_PATH%_trace.json}_adapter_subset.json"
export FAASLORA_TRACE_ROLE=exploratory
export FAASLORA_FORMAL_RUN=0
export FAASLORA_RESULTS_TAG="$TAG"
export FAASLORA_WORKER_START_TIMEOUT_S=600
export FAASLORA_KEEP_DEDICATED_WORKER_LOGS=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
exec bash scripts/run_all_experiments_user_scope.sh \
  --config "$CONFIG" \
  --scenario "$SCENARIO" --full-stack
