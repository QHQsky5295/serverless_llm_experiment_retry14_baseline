# PrimeLoRA Remote Artifact Node

This directory contains a sanitized two-node artifact-store component for
PrimeLoRA.  It is intended for deployment demonstrations and reproducibility
packages where LoRA adapters should be transferred from a separate remote
machine instead of being read from a local folder.

The remote node is intentionally lightweight.  It does not run model inference.
It only serves LoRA adapter directories as authenticated HTTP tarballs.  The
local inference node keeps the existing PrimeLoRA experiment path by default;
remote transfer is enabled only when `FAASLORA_REMOTE_ARTIFACT_ENABLED=1` is set.

## Remote Node

Prepare a directory containing adapter subdirectories:

```bash
mkdir -p /data/primelora_remote_artifacts/llama32_3b
```

From the local node, adapters can be staged with:

```bash
scripts/stage_remote_artifacts.sh \
  --source artifacts/frozen/llama32_3b_a500_v1_modelscope \
  --remote-user lab14 \
  --remote-host 10.199.227.174 \
  --remote-port 8122 \
  --remote-dir /data/primelora_remote_artifacts/llama32_3b_a500_v1_modelscope
```

The staging helper uses interactive SSH or SSH keys.  It never stores a
password in the repository.  It dereferences support-file symlinks while
copying, because the frozen local adapter pools may contain absolute symlinks
into the GPU node's model cache.

Start the server:

```bash
cd /path/to/PrimeLoRA
python3 remote_artifact_node/server.py \
  --root /data/primelora_remote_artifacts/llama32_3b \
  --host 0.0.0.0 \
  --port 18080
```

Optional token authentication:

```bash
export PRIME_REMOTE_TOKEN='<runtime-token>'
python3 remote_artifact_node/server.py \
  --root /data/primelora_remote_artifacts/llama32_3b \
  --host 0.0.0.0 \
  --port 18080
```

Do not commit passwords or runtime tokens.  The server accepts either
`Authorization: Bearer <token>` or `X-PrimeLoRA-Token: <token>`.

## Local Inference Node

Smoke-test the remote store:

```bash
export FAASLORA_REMOTE_ARTIFACT_ENDPOINT=http://192.168.4.174:18080
python3 scripts/remote_artifact_client.py health
python3 scripts/remote_artifact_client.py list --limit 5
python3 scripts/remote_artifact_client.py smoke --dst-root /tmp/primelora_remote_fetch
```

Enable remote transfer in an opt-in run:

```bash
export FAASLORA_REMOTE_ARTIFACT_ENABLED=1
export FAASLORA_REMOTE_ARTIFACT_ENDPOINT=http://192.168.4.174:18080
python3 scripts/run_all_experiments.py --config configs/experiments.yaml --quick
```

When the switch is not set, all existing paper experiments keep using their
local frozen artifact directories.

## API

- `GET /health`: server health and root directory.
- `GET /manifest`: available adapter IDs.
- `HEAD /artifacts/<adapter_id>.tar.gz`: existence check.
- `GET /artifacts/<adapter_id>.tar.gz`: download one adapter directory.

The tarball contains the contents of the adapter directory, not an extra nested
top-level directory.  The local client extracts it directly into the requested
NVMe cache path.
