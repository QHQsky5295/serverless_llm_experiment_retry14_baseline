#!/usr/bin/env bash
set -euo pipefail

# D385 is a new, non-overwriting attempt after D384 reference attempt was
# interrupted by the launcher.  It consumes the immutable D384 native witness.
ROOT=/home/qhq/serverless_llm_experiment_retry14_baseline
OUT=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d385_20261010/7b_peft_sequence_reference_100_retry1.json
LAUNCH=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d385_20261010/7b_peft_sequence_reference_100_retry1_launch.json
TRACE=/home/qhq/serverless_llm_baselines/results/paper_experiments/11_remote_fair_main_local_sim_v4/20260512_123619_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1/shared_artifacts/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1_trace.json
CONFIG="$ROOT/configs/ieee_tc/20261010_d384_7b_native_token_witness_100.yaml"
NATIVE=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d384_20261010/7b_native_token_witness_100/experiment_results_full_vllm_dedicated_a500_r100_c2_tc_ieee_native_token_witness_100_d384_7b_native_token_witness_100.json
ADAPTER_ROOT=/home/qhq/serverless_llm_experiment/artifacts/frozen/llama2_7b_a500_v2_publicmix
NVML_BINDING=/home/qhq/anaconda3/envs/LLM_vllm0102/lib/python3.12/site-packages/pynvml.py
NVML_SHA256=4251429c25f1615a4166f395d3c09fe0732bfb023864c4b7f12813373e5696ea
cd "$ROOT"
if [[ -e "$OUT" || -e "$LAUNCH" ]]; then
  echo "refusing to reuse existing D385 reference evidence" >&2
  exit 70
fi
test -s "$NATIVE"
mkdir -p "$(dirname "$OUT")"
export PATH="/home/qhq/anaconda3/envs/LLM_vllm0102/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export VLLM_NO_USAGE_STATS=1
export FAASLORA_TC_NVML_BINDING="$NVML_BINDING"
export FAASLORA_TC_NVML_SHA256="$NVML_SHA256"
AUX_UNIT="primelora-tc-aux-$(/usr/bin/python3 -c 'import uuid; print(uuid.uuid4().hex)').scope"
exec systemd-run --user --scope --collect --unit="$AUX_UNIT" \
  -p MemoryHigh=3G -p MemoryMax=4G -p MemorySwapMax=0 \
  taskset -c 2,3,26,27 /usr/bin/python3 scripts/ieee_tc_preflight.py \
  gated-launch --output "$LAUNCH" --predicted-growth-gib 16 \
  --exec /home/qhq/anaconda3/envs/LLM_vllm0102/bin/python scripts/ieee_tc_preflight.py \
    backend-peft-sequence-reference \
    --native-result "$NATIVE" \
    --replay-trace "$TRACE" \
    --config "$CONFIG" \
    --adapter-root "$ADAPTER_ROOT" \
    --output "$OUT"
