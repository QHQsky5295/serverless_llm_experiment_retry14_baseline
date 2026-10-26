#!/usr/bin/env bash
# Same D157 native_source_matrix and gated-launch; no replacement runner.
set -euo pipefail
ROOT=/home/qhq/serverless_llm_experiment_retry14_baseline
case "${1:-d298}" in
  d877)
    # D725's exact 423 cases; new measurements for private CPU construction.
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d877_20261026
    CONFIG=configs/ieee_tc/20261026_d877_3b_private_host_source.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d877_private_host_profile/source_spec.json
    ;;
  d821)
    # Existing D725 first six waves, only the D820 pressure transport differs.
    # This checks the real process boundary, not a new service profile.
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d821_20261025
    CONFIG=configs/ieee_tc/20261022_d724_3b_graph_native.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d821_shared_pressure_native/source_spec.json
    ;;
  d725)
    # D701 cases reused; graph configuration and measured timing identity are new.
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d725_20261022
    CONFIG=configs/ieee_tc/20261022_d724_3b_graph_native.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d725_3b_graph_profile/source_spec.json
    ;;
  d647|d648)
    # Existing native warm collector; D646 indices select old inputs, not timings.
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/${1}_20261017
    CONFIG=configs/ieee_tc/20261017_d641_3b_native_source.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d646_3b_warm_inputs/warm_input_index.json
    ;;
  d644)
    # D642 completed lengths initialize admission; old timing samples stay old.
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d644_20261017
    CONFIG=configs/ieee_tc/20261017_d641_3b_native_source.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d643_3b_initializer/source_spec.json
    ;;
  d699)
    # D698's 30 missing 3B output/source/rank/admission classes only.  The
    # source spec is CPU-prepared from existing prefix requests; the native
    # collector must still close coverage before any profile is accepted.
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d699_20261019
    CONFIG=configs/ieee_tc/20261017_d641_3b_native_source.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d699_3b_output_profile_gaps/source_spec.json
    ;;
  d700)
    # New execution key after D699's launcher identity failure.  The source
    # specification is unchanged; only the explicit 3B runner selection is
    # corrected and the raw directory is fresh.
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d700_20261019
    CONFIG=configs/ieee_tc/20261017_d641_3b_native_source.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d699_3b_output_profile_gaps/source_spec.json
    ;;
  d701)
    # Same-spec full 3B profile collection: D645's 24 identities and waves
    # plus the pre-registered D700 output-class gap waves.  This is a new
    # append-only execution key; it never mutates D644/D645/D700 evidence.
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d701_20261019
    CONFIG=configs/ieee_tc/20261017_d641_3b_native_source.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d701_3b_same_spec_profile/source_spec.json
    ;;
  d642)
    # Same D641 3B cases/model; align only with the current native launch contract.
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d642_20261017
    CONFIG=configs/ieee_tc/20261017_d641_3b_native_source.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d640_3b_migration/source_spec.json
    ;;
  d641)
    # 3B current-runtime migration. Cases come from D88; no old timings reused.
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d641_20261017
    CONFIG=configs/ieee_tc/20261017_d641_3b_native_source.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d640_3b_migration/source_spec.json
    ;;
  d526)
    # D525 fixes the native integer contract; reuse frozen cases, not old timings.
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d526_20261014
    CONFIG=configs/ieee_tc/20261014_d524_7b_metadata_source.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d524_metadata_profiles/source_spec.json
    ;;
  d524)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d524_20261014
    CONFIG=configs/ieee_tc/20261014_d524_7b_metadata_source.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d524_metadata_profiles/source_spec.json
    ;;
  d362)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d362_20261009
    CONFIG=configs/ieee_tc/20261009_d362_7b_file_demand_source.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d362_file_demand_profiles/source_spec.json
    ;;
  d356-gaps)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d356_20261008/gaps1
    CONFIG=configs/ieee_tc/20261008_d356_7b_control_source.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d356_control_profiles/gaps1/source_spec.json
    ;;
  d356)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d356_20261008
    CONFIG=configs/ieee_tc/20261008_d356_7b_control_source.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d356_control_profiles/source_spec.json
    ;;
  d344-gaps)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d344_20261008/gaps1
    CONFIG=configs/ieee_tc/20261008_d343_7b_capacity_probe14.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d344_capacity_profiles/gaps1/source_spec.json
    ;;
  d344)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d344_20261008
    CONFIG=configs/ieee_tc/20261008_d343_7b_capacity_probe14.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d344_capacity_profiles/source_spec.json
    ;;
  d343-cap14)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d343_20261008/cap14
    CONFIG=configs/ieee_tc/20261008_d343_7b_capacity_probe14.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d343_capacity_probe/cap14/source_spec.json
    ;;
  d343-probe)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d343_20261008
    CONFIG=configs/ieee_tc/20261008_d343_7b_capacity_probe.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d343_capacity_probe/source_spec.json
    ;;
  d298)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d298_20261007
    CONFIG=configs/ieee_tc/20261007_d298_7b_capacity_source.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d298_capacity/source_spec.json
    ;;
  d307)
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d307_20261007
    CONFIG=configs/ieee_tc/20261007_d307_7b_prefill_source.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d307_prefill/source_spec.json
    ;;
  d307-attempt2)
    # Same candidate/cases; explicit new execution after the atomic file-pin fix.
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d307_20261007/attempt2
    CONFIG=configs/ieee_tc/20261007_d307_7b_prefill_source.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d307_prefill/source_spec.json
    ;;
  d321-gaps)
    # Same qualified backend/collector; only19 missing output classes.
    RAW=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d321_20261007/gaps1
    CONFIG=configs/ieee_tc/20261007_d298_7b_capacity_source.yaml
    SPEC=paper_results/ieee_tc/p2_backend/d321_output_classes/gaps1/source_spec.json
    ;;
  *) exit 2 ;;
esac
[[ $# -le 1 ]] || exit 2
cd "$ROOT"
mkdir -p "$RAW"
[[ ! -e "$RAW/source1_launch.json" && ! -e "$RAW/source1.json" ]] || exit 2
set -o noclobber
exec >"$RAW/source1_launcher.log" 2>&1
git rev-parse HEAD > "$RAW/source_commit.txt"
export PATH="/home/qhq/.venvs/primelora_vllm0300_tc_20260925/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.0 FLASHINFER_NVCC=/usr/local/cuda-13.0/bin/nvcc MAX_JOBS=2
IFS= read -r PRIME_REMOTE_TOKEN < /home/qhq/.config/primelora-tc-d75/artifact.token
export PRIME_REMOTE_TOKEN
export FAASLORA_REMOTE_ARTIFACT_ENABLED=1 FAASLORA_REMOTE_ARTIFACT_ENDPOINT=http://192.168.4.174:18081
MODEL_PROFILE=llama2_7b_main_v2_publicmix
REPLAY_TRACE=/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260424_131528_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1/shared_artifacts/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_trace.json
if [[ "${1:-d298}" == d641 || "${1:-d298}" == d642 || "${1:-d298}" == d644 || "${1:-d298}" == d647 || "${1:-d298}" == d648 || "${1:-d298}" == d699 || "${1:-d298}" == d700 || "${1:-d298}" == d701 || "${1:-d298}" == d725 || "${1:-d298}" == d821 ]]; then
  export FAASLORA_REMOTE_ARTIFACT_ENDPOINT=http://192.168.4.174:18080
  MODEL_PROFILE=llama32_3b_main_modelscope
  REPLAY_TRACE=/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260509_134259_llama32_3b_main_s8/shared_artifacts/llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_trace.json
fi
if [[ "${1:-d298}" == d642 || "${1:-d298}" == d644 || "${1:-d298}" == d647 || "${1:-d298}" == d648 || "${1:-d298}" == d699 || "${1:-d298}" == d700 || "${1:-d298}" == d701 || "${1:-d298}" == d725 || "${1:-d298}" == d821 ]]; then
  export FAASLORA_TC_NATIVE_TOOLCHAIN_CONTRACT="$ROOT/configs/ieee_tc/native_toolchain_cuda130_v1.json"
  export FAASLORA_TC_NATIVE_TOOLCHAIN_SHA256=328b202599d90b9e037ebad53bc7696a9997be2f0f412aaaeda2c1a0cc9c5b1c
  export FAASLORA_WORKER_START_TIMEOUT_S=600
  export FAASLORA_KEEP_DEDICATED_WORKER_LOGS=1
fi
export FAASLORA_REMOTE_ARTIFACT_DELIVERY_MODE=prepublished_gzip_v1
if [[ "${1:-d298}" == d877 ]]; then
  export FAASLORA_REMOTE_ARTIFACT_ENDPOINT=http://192.168.4.174:18080
  MODEL_PROFILE=llama32_3b_main_modelscope
  REPLAY_TRACE=/home/qhq/serverless_llm_baselines/results/paper_experiments/03_main_comparison/20260509_134259_llama32_3b_main_s8/shared_artifacts/llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_mainv1_trace.json
  export FAASLORA_TC_NATIVE_TOOLCHAIN_CONTRACT="$ROOT/configs/ieee_tc/native_toolchain_cuda130_v1.json"
  export FAASLORA_TC_NATIVE_TOOLCHAIN_SHA256=328b202599d90b9e037ebad53bc7696a9997be2f0f412aaaeda2c1a0cc9c5b1c
  export FAASLORA_WORKER_START_TIMEOUT_S=600 FAASLORA_KEEP_DEDICATED_WORKER_LOGS=1
  export FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY=/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d747_20261023/_ieee_native_host_descriptor.cpython-312-x86_64-linux-gnu.so
  export FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256=b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044
  export FAASLORA_REMOTE_ARTIFACT_GZIP_DECODER=isal
  export PYTHONPATH="/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d838_20261025/dependency:$ROOT"
fi
export FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY=uncached_background_v1
export PYTORCH_ALLOC_CONF=pinned_max_cached_size_mb:0,pinned_use_background_threads:True
export VLLM_CACHE_ROOT=/home/qhq/.cache/vllm
export FAASLORA_TC_NVML_BINDING=/home/qhq/anaconda3/envs/LLM_vllm0102/lib/python3.12/site-packages/pynvml.py
export FAASLORA_TC_NVML_SHA256=4251429c25f1615a4166f395d3c09fe0732bfb023864c4b7f12813373e5696ea
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
QUALIFICATION_ARGS=(--request-count 1000 --qualification-mode native_source_matrix --source-profile-spec "$SPEC")
if [[ "${1:-d298}" == d647 || "${1:-d298}" == d648 ]]; then
  # Pure GPU-ready reference excludes network, router, planner and admission.
  unset FAASLORA_REMOTE_ARTIFACT_ENABLED FAASLORA_REMOTE_ARTIFACT_ENDPOINT
  unset FAASLORA_REMOTE_ARTIFACT_DELIVERY_MODE PRIME_REMOTE_TOKEN
  PHASE=probe
  [[ "$1" != d648 ]] || PHASE=measure
  INDEX_SHA=$(sha256sum "$SPEC")
  INDEX_SHA=${INDEX_SHA%% *}
  QUALIFICATION_ARGS=(--request-count 4000 --qualification-mode native_warm_reference
    --warm-reference-index "$SPEC" --warm-reference-index-sha256 "$INDEX_SHA" --warm-reference-phase "$PHASE")
fi
unit="primelora-tc-aux-$(tr -d '-' < /proc/sys/kernel/random/uuid).scope"
exec systemd-run --user --scope --collect --unit="$unit" \
  -p MemoryHigh=3G -p MemoryMax=4G -p MemorySwapMax=0 \
  taskset -c 2,3,26,27 /usr/bin/python3 scripts/ieee_tc_preflight.py gated-launch \
  --output "$RAW/source1_launch.json" --predicted-growth-gib 8 \
  --exec /home/qhq/.venvs/primelora_vllm0300_tc_20260925/bin/python \
  scripts/ieee_tc_preflight.py backend-model-check \
  --runtime-receipt results/ieee_tc/p2_backend_qualification/runtime_20260926/cuda_import_attempt3.json \
  --config "$CONFIG" --model-profile "$MODEL_PROFILE" \
  --replay-trace "$REPLAY_TRACE" \
  "${QUALIFICATION_ARGS[@]}" \
  --output "$RAW/source1.json"
