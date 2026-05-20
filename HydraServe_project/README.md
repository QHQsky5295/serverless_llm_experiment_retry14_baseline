# HydraServe Reproduction Gate

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
