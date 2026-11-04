#!/usr/bin/env bash
# Reuse D1062's post-run audit sequence. No inference/retries/remote mutations.
set -euo pipefail
set -o noclobber
ROOT=/home/qhq/serverless_llm_experiment_retry14_baseline
RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1063_20261105
PY=/home/qhq/.venvs/primelora_vllm0300_tc_20260925/bin/python
exec > "$RAW/postrun_pipeline.log" 2>&1
trap 'rc=$?; printf "pipeline_exit_code=%s\n" "$rc"' EXIT
cd "$ROOT"
date --iso-8601=seconds
deadline=$((SECONDS + 7200))
while tmux has-session -t tc-d1063-nohost-3b-w1 2>/dev/null; do
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 15
done
date --iso-8601=seconds
jq -e '.pass == true and .service_path_removed == true and .native_gpu_context_release_confirmed == true' \
    "$RAW/d1063_3b_nohost_w1/launch.json"
unit="primelora-tc-test-$(/usr/bin/python3 -c 'import uuid; print(uuid.uuid4().hex)')"
systemd-run --user --scope --collect --unit="$unit" \
    -p MemoryHigh=3G -p MemoryMax=4G -p MemorySwapMax=0 -p AllowedCPUs=2,3,26,27 \
    env -u PYTHONPATH CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    taskset -c 2,3,26,27 "$PY" "$RAW/collect_closed_evidence.py"
date --iso-8601=seconds
