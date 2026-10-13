# HydraServe Reproduction Gate

## Current IEEE TC work (2026-10-13)

The May decision below is historical, not the current qualification outcome.
The request-identity omission does not by itself require replacing scheduler
semantics. Necessary isolated platform and LoRA-coordinate adaptation is ongoing;
the unrelated existing k3s cluster is not used or modified. Retain native modified
vLLM 0.4.2, pipeline, overlap and consolidation; no baseline performance tuning.
See [D479 adaptation evidence](../docs/ieee_tc/D479_HYDRASERVE_LORA_BINDING_ADAPTATION.md).
The [D480 platform seam](../docs/ieee_tc/D480_HYDRASERVE_PLATFORM_PORT.md) preserves
native policies and actual physical-node topology; its CPU checks pass, but the
concrete isolated lifecycle and model workers still require qualification.
The [D481 IPC/transport witness](../docs/ieee_tc/D481_HYDRASERVE_IPC_AND_TRANSPORT.md)
now demonstrates real private shared IPC, host/container HTTP and CPU execution
of native two-stage TCP connections with explicit replica source addresses.
It is not a CUDA, numerical, consolidation or performance qualification.
CPU ownership tests and single-GPU NVML access have passed in their limited
scopes. Native GPU/numerical correctness and performance are NOT yet qualified.
Official source remains clean; the new runtime component is not installed yet.

## Historical May gate (retained)

HydraServe is tracked as a new serverless LLM inference candidate for the
2026-05-18 baseline campaign.

- Upstream: `https://github.com/LLMServe/hydraserve`
- Upstream commit: `8ae605de354ccfa2e9095514cdb4a9e9c56aa56b`
- Venue: NSDI 2026
- Local source: `vendor_new_baselines/HydraServe_main_20260520`
- Control-plane env: `hydraserve_py_20260520`

Current decision: gate-only appendix evidence on this machine. The upstream
control modules import in an isolated Python environment and the embedded
vLLM 0.4.2 source still exposes static LoRA module arguments, but the complete
HydraServe system requires a Kubernetes GPU runtime, storage-server pods,
Docker images, node labels, and GPU-share scheduling that are not available to
the current user account. In addition, HydraServe's scheduler request path
does not preserve per-request adapter identity, so supporting the closed
PrimeLoRA LoRA workload would require changing scheduler semantics rather than
only adapting paths or local libraries.

Do not enter HydraServe into the formal comparison table unless a real GPU
Kubernetes deployment can replay the unchanged true-remote 3B/7B LoRA workload
and emit `e2e_v3`.
