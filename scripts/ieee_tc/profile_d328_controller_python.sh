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
  d512_7b_full_w0_prefix1)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d512_20261014
    ;;
  d519_7b_full_w0_prefix1)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d519_20261014
    ;;
  d603_7b_controller_full_w2)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d603_20261017
    ;;
  d668_3b_full_w0_prefix1)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d668_20261018
    ;;
  d684_3b_full_w0_prefix1)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d684_20261018
    ;;
  d734_3b_controller_prefix1000)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d734_20261022
    ;;
  d843_3b_max3_full_w2)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d843_20261025
    ;;
  d863_attempt2_3b_native_prefix1000)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d863_20261026_attempt2
    ;;
  d863_3b_native_prefix1000)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d863_20261026
    ;;
  d723_3b_step_components100)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d723_20261022
    ;;
  *) echo 'Unregistered diagnostic observer campaign' >&2; exit 2 ;;
esac
test "$1" = "$ROOT/scripts/run_all_experiments.py"
test -n "${FAASLORA_TC_LAUNCH_RECEIPT:-}"
test -f "$FAASLORA_TC_LAUNCH_RECEIPT"
test ! -e "$RAW/controller_prefix1_time.txt"
test "${FAASLORA_FORMAL_RUN:-}" = 0
unset FAASLORA_TC_STACK_REPLAY_SCOPE
if [[ "$FAASLORA_RESULTS_TAG" == d328_7b_full_w0_prefix1 || "$FAASLORA_RESULTS_TAG" == d456_7b_full_w0_prefix1 || "$FAASLORA_RESULTS_TAG" == d603_7b_controller_full_w2 || "$FAASLORA_RESULTS_TAG" == d734_3b_controller_prefix1000 || "$FAASLORA_RESULTS_TAG" == d843_3b_max3_full_w2 ]]; then
  test ! -e "$(dirname "$FAASLORA_TC_LAUNCH_RECEIPT")/diagnostic_stacks"
  export FAASLORA_TC_STACK_SAMPLING=python_frames_v1
  unset FAASLORA_TC_CORE_TIMING
  if [[ "$FAASLORA_RESULTS_TAG" == d603_7b_controller_full_w2 || "$FAASLORA_RESULTS_TAG" == d843_3b_max3_full_w2 ]]; then
    export FAASLORA_TC_STACK_REPLAY_SCOPE=full_source_v1
  fi
else
  test ! -e "$(dirname "$FAASLORA_TC_LAUNCH_RECEIPT")/diagnostic_core"
  unset FAASLORA_TC_STACK_SAMPLING
  export FAASLORA_TC_CORE_TIMING=host_spans_v1
fi
if [[ "$FAASLORA_RESULTS_TAG" == d519_7b_full_w0_prefix1 || "$FAASLORA_RESULTS_TAG" == d723_3b_step_components100 ]]; then
  export FAASLORA_TC_CORE_STEP_TIMING=v2_components_v1
else
  unset FAASLORA_TC_CORE_STEP_TIMING
fi
export FAASLORA_PYTHON=/home/qhq/.venvs/primelora_vllm0300_tc_20260925/bin/python
if [[ "$FAASLORA_RESULTS_TAG" == d668_3b_full_w0_prefix1 || "$FAASLORA_RESULTS_TAG" == d684_3b_full_w0_prefix1 ]]; then
  export FAASLORA_TC_HOST_INVENTORY_TIMING=fresh_inventory_v1
else
  unset FAASLORA_TC_HOST_INVENTORY_TIMING
fi
exec /usr/bin/time -v -o "$RAW/controller_prefix1_time.txt" "$FAASLORA_PYTHON" "$@"
