#!/usr/bin/env bash
# Continue the existing D1063 post-run sequence; offline comparison only.
set -euo pipefail
set -o noclobber
ROOT=/home/qhq/serverless_llm_experiment_retry14_baseline
RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1063_20261105
PY=/home/qhq/.venvs/primelora_vllm0300_tc_20260925/bin/python
exec > "$RAW/compare_after_closure.log" 2>&1
trap 'rc=$?; printf "comparison_exit_code=%s\n" "$rc"' EXIT
cd "$ROOT"
date --iso-8601=seconds
deadline=$((SECONDS + 7200))
while tmux has-session -t tc-d1063-nohost-postrun 2>/dev/null; do
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 15
done
date --iso-8601=seconds
rg -q '^pipeline_exit_code=0$' "$RAW/postrun_pipeline.log"
jq -e '.passed == true and .requests == 4000 and .work_caches_empty == true' \
    "$ROOT/paper_results/ieee_tc/ablation/d1063_3b_nohost_w1/completion/closure.json"
unit="primelora-tc-test-$(/usr/bin/python3 -c 'import uuid; print(uuid.uuid4().hex)')"
systemd-run --user --scope --collect --unit="$unit" \
    -p MemoryHigh=3G -p MemoryMax=4G -p MemorySwapMax=0 -p AllowedCPUs=2,3,26,27 \
    env -u PYTHONPATH CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    taskset -c 2,3,26,27 "$PY" "$RAW/compare.py"
date --iso-8601=seconds
