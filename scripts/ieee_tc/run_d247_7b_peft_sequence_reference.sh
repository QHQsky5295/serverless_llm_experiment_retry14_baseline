#!/usr/bin/env bash
set -euo pipefail

# D247 is a correctness-only independent reference.  Refuse to overwrite any
# result or launch receipt; no replay service or remote artifact node is used.
ROOT=/home/qhq/serverless_llm_experiment_retry14_baseline
OUT=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d247_20261006/7b_peft_sequence_reference.json
LAUNCH=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d247_20261006/7b_peft_sequence_reference_launch.json
TRACE=/home/qhq/serverless_llm_baselines/results/paper_experiments/11_remote_fair_main_local_sim_v4/20260512_123619_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1/shared_artifacts/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1_trace.json
CONFIG=$ROOT/configs/ieee_tc/20261006_d243_7b_token_capture_22req_retry2.yaml
ADAPTER_ROOT=/home/qhq/serverless_llm_experiment/artifacts/frozen/llama2_7b_a500_v2_publicmix
cd "$ROOT"
if [[ -e "$OUT" || -e "$LAUNCH" ]]; then
  echo "refusing to reuse D247 evidence path" >&2
  exit 70
fi
mkdir -p "$(dirname "$OUT")"

export PATH="/home/qhq/anaconda3/envs/LLM_vllm0102/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export VLLM_NO_USAGE_STATS=1

exec /home/qhq/anaconda3/envs/LLM_vllm0102/bin/python scripts/ieee_tc_preflight.py \
  gated-launch --output "$LAUNCH" --predicted-growth-gib 16 \
  --exec /home/qhq/anaconda3/envs/LLM_vllm0102/bin/python scripts/ieee_tc_preflight.py \
    backend-peft-sequence-reference \
    --native-result /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d243_retry2_20261006/7b_token_capture_22req/experiment_results_full_vllm_dedicated_a500_r22_c2_tc_ieee_token_capture_22_retry2_d243_7b_token_capture_22req_retry2.json \
    --replay-trace "$TRACE" \
    --config "$CONFIG" \
    --adapter-root "$ADAPTER_ROOT" \
    --output "$OUT"
