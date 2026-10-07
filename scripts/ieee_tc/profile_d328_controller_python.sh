#!/usr/bin/env bash
set -euo pipefail
# Reuse D200's bounded observer wrapper; no serving policy or kernel changes.
ROOT=/home/qhq/serverless_llm_experiment_retry14_baseline
RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d328_20261008
test "$1" = "$ROOT/scripts/run_all_experiments.py"
test -n "${FAASLORA_TC_LAUNCH_RECEIPT:-}"
test -f "$FAASLORA_TC_LAUNCH_RECEIPT"
test ! -e "$RAW/controller_prefix1_time.txt"
test ! -e "$(dirname "$FAASLORA_TC_LAUNCH_RECEIPT")/diagnostic_stacks"
test "${FAASLORA_FORMAL_RUN:-}" = 0
export FAASLORA_TC_STACK_SAMPLING=python_frames_v1
export FAASLORA_PYTHON=/home/qhq/.venvs/primelora_vllm0300_tc_20260925/bin/python
exec /usr/bin/time -v -o "$RAW/controller_prefix1_time.txt" "$FAASLORA_PYTHON" "$@"
