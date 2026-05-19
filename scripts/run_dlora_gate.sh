#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/qhq/serverless_llm_baselines}
DLORA_REPO=${DLORA_REPO:-${ROOT}/vendor_new_baselines/dLoRA_artifact_main_20260519}
DLORA_PATCH=${DLORA_PATCH:-${ROOT}/DLoRA_project/patches/modern_ray_import_compat.patch}
LOG_ROOT=${LOG_ROOT:-${ROOT}/results/logs/new_serverless_baselines_remote_v1/dlora/gate}
GATE_ROOT=${GATE_ROOT:-${ROOT}/results/paper_experiments/15_new_serverless_baselines_remote_v1/gates/dlora}
ENV_NAME=${ENV_NAME:-dlora_20260519}
CLONE_SOURCE_ENV=${CLONE_SOURCE_ENV:-medusa_20260518}
CONDA_BIN=${CONDA_BIN:-/home/qhq/anaconda3/bin/conda}
STAGE=${1:-source}
PIP_INSTALL_ARGS=${PIP_INSTALL_ARGS:-}
CUDA_COMPAT_INCLUDES=${CUDA_COMPAT_INCLUDES:-/home/qhq/anaconda3/pkgs/cuda-cudart-dev-12.1.105-0/include:/home/qhq/anaconda3/pkgs/libcublas-dev-12.1.3.1-0/include:/home/qhq/anaconda3/pkgs/libcusparse-dev-12.1.0.106-0/include:/home/qhq/anaconda3/pkgs/libcusolver-dev-11.4.5.107-0/include}
CUDA_COMPAT_LIBS=${CUDA_COMPAT_LIBS:-/home/qhq/anaconda3/pkgs/cuda-cudart-12.1.105-0/lib:/home/qhq/anaconda3/pkgs/libcublas-12.1.3.1-0/lib:/home/qhq/anaconda3/pkgs/libcusparse-12.1.0.106-0/lib:/home/qhq/anaconda3/pkgs/libcusolver-11.4.5.107-0/lib}

LLAMA2_ROUND=${LLAMA2_ROUND:-${ROOT}/results/paper_experiments/12_remote_fair_main_real_remote_v1/20260513_012813_llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1}
LLAMA32_ROUND=${LLAMA32_ROUND:-${ROOT}/results/paper_experiments/12_remote_fair_main_real_remote_v1/20260513_160342_llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1}

mkdir -p "${LOG_ROOT}" "${GATE_ROOT}"

run_logged() {
  local name=$1
  shift
  {
    echo "[cmd] $*"
    "$@"
  } 2>&1 | tee "${LOG_ROOT}/${name}.log"
}

source_gate() {
  run_logged git_state git -C "${DLORA_REPO}" rev-parse HEAD
  run_logged environment bash -lc "python3 --version && nvcc --version && nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
  run_logged source_scan bash -lc "rg -n --no-heading 'use-dummy-weights|use_dummy_weights|torch\\.randn|torch\\.zeros|adapter_model\\.safetensors|safetensors|PeftModel|load_adapter|LoRARequest|load_lora_adapter|Llama-3|Llama-2' '${DLORA_REPO}'/ae_scripts '${DLORA_REPO}'/eval_scripts '${DLORA_REPO}'/vllm '${DLORA_REPO}'/examples '${DLORA_REPO}'/tests '${DLORA_REPO}'/docs || true"

  python3 "${ROOT}/scripts/probe_dlora_gate.py" \
    --repo "${DLORA_REPO}" \
    --trace "${LLAMA2_ROUND}/shared_artifacts/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1_trace.json" \
    --adapter-subset "${LLAMA2_ROUND}/shared_artifacts/llama2_7b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1_adapter_subset.json" \
    --model-label llama2_7b \
    --out-json "${GATE_ROOT}/dlora_llama2_7b_source_gate.json" \
    2>&1 | tee "${LOG_ROOT}/probe_llama2_7b.log"

  python3 "${ROOT}/scripts/probe_dlora_gate.py" \
    --repo "${DLORA_REPO}" \
    --trace "${LLAMA32_ROUND}/shared_artifacts/llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1_trace.json" \
    --adapter-subset "${LLAMA32_ROUND}/shared_artifacts/llama32_3b_r4000_a500_seed42_z1p0_hot48_rot500_s8_remote_fair_real-remote_v1_adapter_subset.json" \
    --model-label llama32_3b \
    --out-json "${GATE_ROOT}/dlora_llama32_3b_source_gate.json" \
    2>&1 | tee "${LOG_ROOT}/probe_llama32_3b.log"
}

apply_local_patches() {
  if [[ ! -f "${DLORA_PATCH}" ]]; then
    return
  fi
  if git -C "${DLORA_REPO}" apply --check --unidiff-zero "${DLORA_PATCH}" >/dev/null 2>&1; then
    run_logged patch_modern_ray git -C "${DLORA_REPO}" apply --unidiff-zero "${DLORA_PATCH}"
  else
    {
      echo "[skip] local patch is already applied or no longer matches: ${DLORA_PATCH}"
      git -C "${DLORA_REPO}" diff -- vllm/engine/ray_utils.py vllm/engine/llm_engine.py vllm/engine/engine_manager.py vllm/entrypoints/llm.py
    } 2>&1 | tee "${LOG_ROOT}/patch_modern_ray.log"
  fi
}

env_gate() {
  if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    run_logged conda_create "${CONDA_BIN}" create -n "${ENV_NAME}" python=3.9 -y
  fi
  run_logged conda_cuda "${CONDA_BIN}" install -n "${ENV_NAME}" nvidia/label/cuda-12.2.0::cuda-toolkit -y
  run_logged conda_torch "${CONDA_BIN}" install -n "${ENV_NAME}" pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
  run_logged pip_base "${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install -U pip setuptools wheel ninja packaging
}

env_clone_gate() {
  if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    run_logged conda_clone "${CONDA_BIN}" create -n "${ENV_NAME}" --clone "${CLONE_SOURCE_ENV}" -y
  fi
  run_logged clone_environment bash -lc "'${CONDA_BIN}' run -n '${ENV_NAME}' python -c \"import sys, torch; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('torch_cuda', torch.version.cuda)\" && '${CONDA_BIN}' run -n '${ENV_NAME}' nvcc --version"
}

build_gate() {
  local prefix
  local pip_args=()
  if [[ -n "${PIP_INSTALL_ARGS}" ]]; then
    read -r -a pip_args <<< "${PIP_INSTALL_ARGS}"
  fi
  prefix=$("${CONDA_BIN}" run -n "${ENV_NAME}" python -c 'import os; print(os.environ["CONDA_PREFIX"])')
  apply_local_patches
  run_logged submodule git -C "${DLORA_REPO}" submodule update --init --recursive
  run_logged pip_install env CUDA_HOME="${prefix}" CUDACXX="${prefix}/bin/nvcc" \
    CPATH="${CUDA_COMPAT_INCLUDES}:${prefix}/targets/x86_64-linux/include:${prefix}/include:${CPATH:-}" \
    LIBRARY_PATH="${CUDA_COMPAT_LIBS}:${prefix}/targets/x86_64-linux/lib:${prefix}/lib:${LIBRARY_PATH:-}" \
    LD_LIBRARY_PATH="${CUDA_COMPAT_LIBS}:${prefix}/targets/x86_64-linux/lib:${prefix}/lib:${LD_LIBRARY_PATH:-}" \
    "${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install -e "${DLORA_REPO}" "${pip_args[@]}"
  run_logged import_vllm "${CONDA_BIN}" run -n "${ENV_NAME}" python -c "import vllm; print(vllm.__version__)"
}

import_gate() {
  run_logged import_vllm "${CONDA_BIN}" run -n "${ENV_NAME}" python -c "import vllm; print(vllm.__version__)"
}

case "${STAGE}" in
  source)
    source_gate
    ;;
  env)
    env_gate
    ;;
  env_clone)
    env_clone_gate
    ;;
  build)
    build_gate
    ;;
  import)
    import_gate
    ;;
  all)
    source_gate
    env_gate
    build_gate
    ;;
  *)
    echo "usage: $0 [source|env|env_clone|build|import|all]" >&2
    exit 2
    ;;
esac
