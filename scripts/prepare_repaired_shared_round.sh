#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${SLLM_BASELINES_ROOT:-/home/qhq/serverless_llm_baselines}"
RUN_TAG="${SLLM_RUN_TAG:?SLLM_RUN_TAG is required}"

bash "${ROOT_DIR}/scripts/prepare_shared_round_artifacts.sh"

TRACE_PATH="${ROOT_DIR}/results/shared_rounds/${RUN_TAG}_trace.json"
SUBSET_PATH="${ROOT_DIR}/results/shared_rounds/${RUN_TAG}_adapter_subset.json"
REPAIRED_REMOTE_DIR="${ROOT_DIR}/results/repaired_loras/${RUN_TAG}"
REPAIRED_SUBSET_PATH="${ROOT_DIR}/results/shared_rounds/${RUN_TAG}_adapter_subset_repaired.json"
PUNICA_VENV="${PUNICA_VENV:-/home/qhq/.venvs/punica_cu121_py310}"

if [[ ! -f "${TRACE_PATH}" ]]; then
  echo "[ERROR] missing generated trace artifact: ${TRACE_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SUBSET_PATH}" ]]; then
  echo "[ERROR] missing generated adapter subset artifact: ${SUBSET_PATH}" >&2
  exit 1
fi
if [[ ! -x "${PUNICA_VENV}/bin/python" ]]; then
  echo "[ERROR] missing Punica venv python: ${PUNICA_VENV}/bin/python" >&2
  exit 1
fi

echo
echo "[repair] Repairing non-finite LoRA weights for this shared round"
"${PUNICA_VENV}/bin/python" "${ROOT_DIR}/scripts/repair_shared_adapter_subset.py" \
  --adapter-subset "${SUBSET_PATH}" \
  --output-dir "${REPAIRED_REMOTE_DIR}" \
  --output-subset "${REPAIRED_SUBSET_PATH}"

echo
echo "Shared round prepared."
echo "  Trace           : ${TRACE_PATH}"
echo "  Raw subset      : ${SUBSET_PATH}"
echo "  Repaired subset : ${REPAIRED_SUBSET_PATH}"
echo "  Repaired remote : ${REPAIRED_REMOTE_DIR}"
