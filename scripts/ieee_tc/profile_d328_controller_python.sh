#!/usr/bin/env bash
set -euo pipefail
# Reuse D200's bounded observer wrapper; no serving policy or kernel changes.
ROOT=/home/qhq/serverless_llm_experiment_retry14_baseline
RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d328_20261008
case "${FAASLORA_RESULTS_TAG:-}" in
  d328_7b_full_w0_prefix1) ;;
  d456_7b_full_w0_prefix1)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d456_20261012
    ;;
  d351_7b_full_w0_prefix1)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d351_20261008
    ;;
  *) echo 'Unregistered diagnostic observer campaign' >&2; exit 2 ;;
esac
test "$1" = "$ROOT/scripts/run_all_experiments.py"
test -n "${FAASLORA_TC_LAUNCH_RECEIPT:-}"
test -f "$FAASLORA_TC_LAUNCH_RECEIPT"
test ! -e "$RAW/controller_prefix1_time.txt"
test "${FAASLORA_FORMAL_RUN:-}" = 0
if [[ "$FAASLORA_RESULTS_TAG" == d328_7b_full_w0_prefix1 || "$FAASLORA_RESULTS_TAG" == d456_7b_full_w0_prefix1 ]]; then
  test ! -e "$(dirname "$FAASLORA_TC_LAUNCH_RECEIPT")/diagnostic_stacks"
  export FAASLORA_TC_STACK_SAMPLING=python_frames_v1
else
  test ! -e "$(dirname "$FAASLORA_TC_LAUNCH_RECEIPT")/diagnostic_core"
  unset FAASLORA_TC_STACK_SAMPLING
  export FAASLORA_TC_CORE_TIMING=host_spans_v1
fi
export FAASLORA_PYTHON=/home/qhq/.venvs/primelora_vllm0300_tc_20260925/bin/python
exec /usr/bin/time -v -o "$RAW/controller_prefix1_time.txt" "$FAASLORA_PYTHON" "$@"
