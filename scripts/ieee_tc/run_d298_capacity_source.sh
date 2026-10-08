#!/usr/bin/env bash
# Same D157 native_source_matrix and gated-launch; no replacement runner.
set -euo pipefail
ROOT=/home/qhq/serverless_llm_experiment_retry14_baseline
case "${1:-d298}" in
  d344-gaps)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d344_20261008/gaps1
    CONFIG=configs/ieee_tc/20261008_d343_7b_capacity_probe14.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d344_capacity_profiles/gaps1/source_spec.json
    ;;
  d344)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d344_20261008
    CONFIG=configs/ieee_tc/20261008_d343_7b_capacity_probe14.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d344_capacity_profiles/source_spec.json
    ;;
  d343-cap14)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d343_20261008/cap14
    CONFIG=configs/ieee_tc/20261008_d343_7b_capacity_probe14.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d343_capacity_probe/cap14/source_spec.json
    ;;
  d343-probe)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d343_20261008
    CONFIG=configs/ieee_tc/20261008_d343_7b_capacity_probe.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d343_capacity_probe/source_spec.json
    ;;
  d298)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d298_20261007
    CONFIG=configs/ieee_tc/20261007_d298_7b_capacity_source.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d298_capacity/source_spec.json
    ;;
  d307)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d307_20261007
    CONFIG=configs/ieee_tc/20261007_d307_7b_prefill_source.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d307_prefill/source_spec.json
    ;;
  d307-attempt2)
    # Same candidate/cases; explicit new execution after the atomic file-pin fix.
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d307_20261007/attempt2
    CONFIG=configs/ieee_tc/20261007_d307_7b_prefill_source.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d307_prefill/source_spec.json
    ;;
  d321-gaps)
    # Same qualified backend/collector; only19 missing output classes.
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d321_20261007/gaps1
    CONFIG=configs/ieee_tc/20261007_d298_7b_capacity_source.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d321_output_classes/gaps1/source_spec.json
    ;;
  *) exit 2 ;;
esac
[[ $# -le 1 ]] || exit 2
cd "$ROOT"
[[ ! -e "$RAW/source1_launch.json" && ! -e "$RAW/source1.json" ]] || exit 2
set -o noclobber
exec >"$RAW/source1_launcher.log" 2>&1
git rev-parse HEAD > "$RAW/source_commit.txt"
export PATH="/home/qhq/.venvs/primelora_vllm0300_tc_20260925/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.0 FLASHINFER_NVCC=/usr/local/cuda-13.0/bin/nvcc MAX_JOBS=2
IFS= read -r PRIME_REMOTE_TOKEN < /home/qhq/.config/primelora-tc-d75/artifact.token
export PRIME_REMOTE_TOKEN
export FAASLORA_REMOTE_ARTIFACT_ENABLED=1 FAASLORA_REMOTE_ARTIFACT_ENDPOINT=http://192.168.4.174:18081
export FAASLORA_REMOTE_ARTIFACT_DELIVERY_MODE=prepublished_gzip_v1
export FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY=uncached_background_v1
export PYTORCH_ALLOC_CONF=pinned_max_cached_size_mb:0,pinned_use_background_threads:True
export VLLM_CACHE_ROOT=/home/qhq/.cache/vllm
export FAASLORA_TC_NVML_BINDING=/home/qhq/anaconda3/envs/LLM_vllm0102/lib/python3.12/site-packages/pynvml.py
export FAASLORA_TC_NVML_SHA256=4251429c25f1615a4166f395d3c09fe0732bfb023864c4b7f12813373e5696ea
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
unit="primelora-tc-aux-$(tr -d '-' < /proc/sys/kernel/random/uuid).scope"
exec systemd-run --user --scope --collect --unit="$unit" \
  -p MemoryHigh=3G -p MemoryMax=4G -p MemorySwapMax=0 \
  taskset -c 2,3,26,27 /usr/bin/python3 scripts/ieee_tc_preflight.py gated-launch \
  --output "$RAW/source1_launch.json" --predicted-growth-gib 8 \
  --exec /home/qhq/.venvs/primelora_vllm0300_tc_20260925/bin/python \
  scripts/ieee_tc_preflight.py backend-model-check \
  --runtime-receipt results/ieee_tc/p2_backend_qualification/runtime_20260926/cuda_import_attempt3.json \
  --config "$CONFIG" --model-profile llama2_7b_main_v2_publicmix \
  --replay-trace /home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260424_131528_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1/shared_artifacts/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_trace.json \
  --request-count 1000 --qualification-mode native_source_matrix \
  --source-profile-spec "$SPEC" \
  --output "$RAW/source1.json"
