#!/usr/bin/env bash
set -euo pipefail

REPO="/home/qhq/serverless_llm_baselines/repos/ServerlessLLM"
HEAD_ENV="${SLLM_HEAD_ENV:-sllm_head_official}"
WORKER_ENV="${SLLM_WORKER_ENV:-sllm_worker_official}"
PYTHON_VERSION="${SLLM_PYTHON_VERSION:-3.10}"
FILTERED_REQS="$(mktemp)"
trap 'rm -f "${FILTERED_REQS}"' EXIT

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda not found"
  exit 1
fi

grep -v '^serverless-llm-store$' "${REPO}/requirements.txt" > "${FILTERED_REQS}"

CONDA_BASE="$(conda info --base)"
HEAD_PREFIX="${CONDA_BASE}/envs/${HEAD_ENV}"
WORKER_PREFIX="${CONDA_BASE}/envs/${WORKER_ENV}"

echo "[1/6] Creating head env: ${HEAD_ENV} (python=${PYTHON_VERSION})"
if [ ! -d "${HEAD_PREFIX}" ]; then
  conda create -y -n "${HEAD_ENV}" "python=${PYTHON_VERSION}"
else
  echo "[info] Reusing existing env: ${HEAD_ENV}"
fi
conda install -y -n "${HEAD_ENV}" ninja

echo "[2/6] Creating worker env: ${WORKER_ENV} (python=${PYTHON_VERSION})"
if [ ! -d "${WORKER_PREFIX}" ]; then
  conda create -y -n "${WORKER_ENV}" "python=${PYTHON_VERSION}"
else
  echo "[info] Reusing existing env: ${WORKER_ENV}"
fi
conda install -y -n "${WORKER_ENV}" ninja

echo "[3/6] Installing head dependencies from source"
env PYTHONNOUSERSITE=1 conda run -n "${HEAD_ENV}" pip install -r "${FILTERED_REQS}"
env PYTHONNOUSERSITE=1 conda run -n "${HEAD_ENV}" pip install \
  "torch>=2.7.0" \
  "transformers>=4.52.4" \
  "accelerate>=1.7.0" \
  "peft>=0.16.0" \
  "grpcio==1.76.0" \
  "grpcio-tools==1.76.0" \
  "httpx>=0.27.0" \
  "attrs>=22.2.0" \
  "pillow>=10.3.0"
env PYTHONNOUSERSITE=1 conda run -n "${HEAD_ENV}" pip install "click==8.1.7"
env PYTHONNOUSERSITE=1 conda run -n "${HEAD_ENV}" pip install --no-deps "${REPO}"

echo "[4/6] Installing worker dependencies from source"
env PYTHONNOUSERSITE=1 conda run -n "${WORKER_ENV}" pip install -r "${REPO}/requirements-worker.txt"
env PYTHONNOUSERSITE=1 conda run -n "${WORKER_ENV}" pip install \
  "grpcio==1.76.0" \
  "grpcio-tools==1.76.0"
env PYTHONNOUSERSITE=1 conda run -n "${WORKER_ENV}" pip install "click==8.1.7"
env PYTHONNOUSERSITE=1 conda run -n "${WORKER_ENV}" pip install "cmake>=3.20,<4.0" ninja
python3 - <<PY
import shutil
from pathlib import Path

build_dir = Path("${REPO}") / "sllm_store" / "build"
if build_dir.exists():
    shutil.rmtree(build_dir)
PY
env PYTHONNOUSERSITE=1 conda run -n "${WORKER_ENV}" bash -lc "cd '${REPO}/sllm_store' && PYTHONNOUSERSITE=1 python setup.py install"
env PYTHONNOUSERSITE=1 conda run -n "${WORKER_ENV}" pip install --no-deps "${REPO}"

echo "[4.5/6] Mirroring sllm_store package from worker env into head env"
python3 - <<PY
import shutil
from pathlib import Path

worker_site = Path("${WORKER_PREFIX}") / "lib/python3.10/site-packages"
head_site = Path("${HEAD_PREFIX}") / "lib/python3.10/site-packages"
for name in ["sllm_store", "serverless_llm_store.egg-info"]:
    src = worker_site / name
    dst = head_site / name
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.exists():
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
PY

echo "[5/6] Import smoke for head env"
env PYTHONNOUSERSITE=1 conda run -n "${HEAD_ENV}" python - <<'PY'
import importlib
for name in ["sllm", "fastapi", "ray", "torch", "transformers", "grpc", "sllm_store.client", "sllm_store._C"]:
    importlib.import_module(name)
print("serverlessllm_head_import_ok")
PY

echo "[6/6] Import smoke for worker env"
env PYTHONNOUSERSITE=1 conda run -n "${WORKER_ENV}" python - <<'PY'
import importlib
for name in ["sllm", "sllm_store", "sllm_store.server", "ray", "transformers", "peft", "torch", "grpc"]:
    importlib.import_module(name)
print("serverlessllm_worker_import_ok")
PY

cat <<EOF

ServerlessLLM isolated environments are ready.
Head env   : ${HEAD_ENV}
Worker env : ${WORKER_ENV}

Suggested next steps:
  1. bash /home/qhq/serverless_llm_baselines/scripts/run_serverlessllm_head.sh
  2. bash /home/qhq/serverless_llm_baselines/scripts/run_serverlessllm_worker.sh
  3. bash /home/qhq/serverless_llm_baselines/scripts/run_serverlessllm_store.sh
  4. bash /home/qhq/serverless_llm_baselines/scripts/run_serverlessllm_serve.sh
EOF
