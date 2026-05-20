# AIBrix Reproduction Gate

Status: build/runtime-LoRA component gate closed on 2026-05-20; not a formal
table row on this machine.

- Upstream: `https://github.com/vllm-project/aibrix`
- Upstream tag: `v0.6.0`
- Upstream commit: `52405d78c63c68df5be9e8bdc5e21ccd6c4abde2`
- Local source: `vendor_new_baselines/AIBrix_v0.6.0_20260520`
- Go build env: `aibrix_20260520`
- Python runtime env: `aibrix_py_20260520`
- Gate evidence: `evidence/gate_2026-05-20.json`

## Local Adaptation Boundary

AIBrix is a Kubernetes-native vLLM control plane, not a single-process serving
engine. The local adaptation therefore stays at the deployment and compatibility
boundary:

- build official controller-manager and gateway-plugins from upstream source;
- use local user-space Go/ZMQ/Python dependencies instead of changing system
  binaries or closed experiment environments;
- validate the AIBrix Python runtime sidecar against the local vLLM
  `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` APIs with a real
  closed-workload Llama-3.2 3B LoRA adapter;
- record Kubernetes/Docker/GPU-pod blockers rather than replacing AIBrix's
  controller, gateway, or CRD workflow with a custom runner.

## Current Result

Passed:

- `controller-manager` builds from upstream source.
- `gateway-plugins` builds in no-ZMQ mode.
- `gateway-plugins` builds in ZMQ mode after adding local user-space
  `zeromq`, `libsodium`, and `pkg-config` to the isolated env.
- Python `aibrix` runtime installs/imports from source.
- AIBrix runtime sidecar can load and unload the real
  `llama32_3b_a500_v1_modelscope/code_lora` adapter into a local vLLM 0.10.2
  server and a `code_lora` completion succeeds.

Formal blocker:

- Full AIBrix requires a Kubernetes cluster with GPU-enabled vLLM pods,
  AIBrix CRDs/controller/gateway, runtime sidecar, and model adapter resources.
  This machine currently exposes no usable user-level Kubernetes/Docker path:
  `/usr/local/bin/kubectl` and `/usr/local/bin/helm` are root-only,
  `/var/run/docker.sock` is not accessible to `qhq`, `kind` is absent, and
  passwordless sudo is unavailable.

Conclusion: AIBrix is valid related appendix/gate evidence, but it should not
enter the formal comparison table until a same-hardware GPU Kubernetes runtime
can run the closed true-remote 3B/7B LoRA workload without replacing AIBrix's
control plane.
