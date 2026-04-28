#!/usr/bin/env bash
set -euo pipefail

BASELINES_ROOT="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"

# Default queue fills the missing 100/200/300/400-adapter points and reuses the
# closed 500-adapter s8 main round as the right endpoint. Set
# PAPER_QUEUE_PROFILE=adapter_pool_full_p0 to rerun the 500-adapter point too.
export PAPER_QUEUE_PROFILE="${PAPER_QUEUE_PROFILE:-adapter_pool_p0}"
export PAPER_QUEUE_SYSTEMS="${PAPER_QUEUE_SYSTEMS:-sglang serverlessllm vllm slora faaslora}"

exec "${BASELINES_ROOT}/scripts/run_paper_long_experiment_queue.sh" "$@"
