#!/usr/bin/env bash
set -euo pipefail

# D247 retry3 is a fresh correctness-only independent reference identity after
# launch-gate failures before GPU loading.  Refuse any pre-existing output.
ROOT=/home/qhq/serverless_llm_experiment_retry14_baseline
OUT=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d247_retry3_20261006/7b_peft_sequence_reference.json
LAUNCH=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d247_retry3_20261006/7b_peft_sequence_reference_launch.json
TRACE=/home/qhq/serverless_llm_baselines/results/paper_experiments/11_remote_fair_main_local_sim_v4/20260512_123619_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1/shared_artifacts/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_local-sim_v1_trace.json
CONFIG=$ROOT/configs/ieee_tc/20261006_d243_7b_token_capture_22req_retry2.yaml
ADAPTER_ROOT=/home/qhq/serverless_llm_experiment/artifacts/frozen/llama2_7b_a500_v2_publicmix
NVML_BINDING=/home/qhq/anaconda3/envs/LLM_vllm0102/lib/python3.12/site-packages/pynvml.py
NVML_SHA256=4251429c25f1615a4166f395d3c09fe0732bfb023864c4b7f12813373e5696ea
cd "$ROOT"
if [[ -e "$OUT" || -e "$LAUNCH" ]]; then
  echo "refusing to reuse D247 retry3 evidence path" >&2
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
export FAASLORA_TC_NVML_BINDING="$NVML_BINDING"
export FAASLORA_TC_NVML_SHA256="$NVML_SHA256"

# The outer supervisor uses system Python for PID-handle signaling and enters
# the shared 4-GiB auxiliary scope.  The CUDA environment is only the child
# reference runner inside the gated service scope.
AUX_UNIT="primelora-tc-aux-$(/usr/bin/python3 -c 'import uuid; print(uuid.uuid4().hex)').scope"
exec systemd-run --user --scope --collect --unit="$AUX_UNIT" \
  -p MemoryHigh=3G -p MemoryMax=4G -p MemorySwapMax=0 \
  taskset -c 2,3,26,27 /usr/bin/python3 scripts/ieee_tc_preflight.py \
  gated-launch --output "$LAUNCH" --predicted-growth-gib 16 \
  --exec /home/qhq/anaconda3/envs/LLM_vllm0102/bin/python scripts/ieee_tc_preflight.py \
    backend-peft-sequence-reference \
    --native-result /home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d243_retry2_20261006/7b_token_capture_22req/experiment_results_full_vllm_dedicated_a500_r22_c2_tc_ieee_token_capture_22_retry2_d243_7b_token_capture_22req_retry2.json \
    --replay-trace "$TRACE" \
    --config "$CONFIG" \
    --adapter-root "$ADAPTER_ROOT" \
    --output "$OUT"
