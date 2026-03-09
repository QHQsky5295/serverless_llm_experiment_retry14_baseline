#!/usr/bin/env bash
# =========================================================================
# Download Qwen2.5-3B-Instruct to local models/ directory
# =========================================================================
# Usage:
#   conda activate LLM
#   cd ~/serverless_llm_experiment
#   bash scripts/download_3b_model.sh
# =========================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MODEL_DIR="${REPO_ROOT}/models/Qwen--Qwen2.5-3B-Instruct"

mkdir -p "${MODEL_DIR}"

# Check if weight files already exist (not just config.json)
SAFETENSOR_COUNT=$(find "${MODEL_DIR}" -name "*.safetensors" 2>/dev/null | wc -l)
if [ "${SAFETENSOR_COUNT}" -gt 0 ]; then
    echo "[OK] Model weights already exist at ${MODEL_DIR} (${SAFETENSOR_COUNT} safetensor files)"
    python3 -c "
import json, pathlib
cfg = json.loads(pathlib.Path('${MODEL_DIR}/config.json').read_text())
print(f'  hidden_size  = {cfg.get(\"hidden_size\", \"?\")}')
print(f'  num_layers   = {cfg.get(\"num_hidden_layers\", \"?\")}')
print(f'  vocab_size   = {cfg.get(\"vocab_size\", \"?\")}')
"
    exit 0
fi

echo "============================================================"
echo "  Downloading Qwen/Qwen2.5-3B-Instruct"
echo "  Target: ${MODEL_DIR}"
echo "  Method: one file at a time (memory-safe)"
echo "============================================================"

python3 << 'PYEOF'
import os, json
from huggingface_hub import hf_hub_download, list_repo_files

repo_id = "Qwen/Qwen2.5-3B-Instruct"
local_dir = os.environ["MODEL_DIR"]

files = list_repo_files(repo_id)
total = len(files)
for i, fname in enumerate(files, 1):
    dest = os.path.join(local_dir, fname)
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  [{i}/{total}] SKIP (exists, {size_mb:.1f} MB): {fname}")
        continue
    print(f"  [{i}/{total}] Downloading: {fname} ...")
    hf_hub_download(
        repo_id=repo_id,
        filename=fname,
        local_dir=local_dir,
    )
    size_mb = os.path.getsize(os.path.join(local_dir, fname)) / 1e6
    print(f"           OK ({size_mb:.1f} MB)")

print("\nAll files downloaded!")
PYEOF

echo ""
echo "[OK] Model downloaded to ${MODEL_DIR}"
python3 -c "
import json, pathlib
cfg = json.loads(pathlib.Path('${MODEL_DIR}/config.json').read_text())
print(f'  hidden_size  = {cfg.get(\"hidden_size\", \"?\")}')
print(f'  num_layers   = {cfg.get(\"num_hidden_layers\", \"?\")}')
print(f'  vocab_size   = {cfg.get(\"vocab_size\", \"?\")}')
"
