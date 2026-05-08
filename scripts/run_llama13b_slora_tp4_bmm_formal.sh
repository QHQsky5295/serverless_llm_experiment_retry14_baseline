#!/usr/bin/env bash
set -euo pipefail

cd /home/qhq/serverless_llm_baselines

SESSION="${SESSION:-paper_llama13b_slora_tp4_bmm_formal}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session ${SESSION} already exists; attach with: tmux attach -t ${SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" "cd /home/qhq/serverless_llm_baselines && \
  PAPER_QUEUE_ID=20260507_llama13b_slora_tp4_bmm_formal \
  SLLM_RUN_TAG=llama2_13b_slora_tp4_bmm_r4000_a500_seed42_s8 \
  FAIR_ROUND_SYSTEMS='slora' \
  SLLM_TOTAL_REQUESTS=4000 \
  SLLM_MODEL_PROFILE=llama2_13b_tp2_v2_publicmix \
  SLLM_DATASET_PROFILE=azure_sharegpt_rep4000 \
  SLLM_WORKLOAD_PROFILE=llama2_13b_tp2_a500_formal4000_s8 \
  SLLM_TIME_SCALE_FACTOR=8.0 \
  SLLM_SELECTED_NUM_ADAPTERS=500 \
  SLORA_TENSOR_PARALLEL_SIZE=4 \
  SLORA_DATA_PARALLEL_REPLICAS=1 \
  SLORA_USE_BMM=1 \
  SLORA_TIMEOUT_S=21600 \
  bash scripts/run_llama13b_main_comparison_queue.sh 2>&1 | tee /tmp/paper_llama13b_slora_tp4_bmm_formal.log; exec bash"

echo "started ${SESSION}"
echo "attach: tmux attach -t ${SESSION}"
echo "tail:   tail -f /tmp/paper_llama13b_slora_tp4_bmm_formal.log"
