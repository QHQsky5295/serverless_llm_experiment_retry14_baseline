# Medusa Baseline Project

This directory is the project entry point for the 2026-05-19 Medusa
reproduction gate. Medusa is kept separate from the existing formal baseline
projects because its official artifact is a modified vLLM/Medusa tree with
CUDA Graph and SPDK materialization dependencies.

## Current Status

- Paper/system: Medusa, ASPLOS 2025.
- Upstream source: `/home/qhq/serverless_llm_baselines/vendor_new_baselines/Medusa_main_20260518`
- Project symlink: `repo`
- Upstream commit: `6581d2e5ec8fa4ecdabcdb50560982a78ea3ca89`
- Runtime env attempted: `medusa_20260518`
- Gate result: official build did not complete on the current machine.

## Gate Summary

The official build gate reached `vllm._C` compilation after CUDA include-path
repair, but failed in `csrc/cuda_graph.cu` on a CUDA Graph API signature
mismatch. Static inspection also shows the official build hard-codes SPDK,
DPDK, and GDRCopy paths under `/home/zsx/spdk`, which are absent on this
machine.

Because the official Medusa runtime cannot be built here, no formal
true-remote LoRA replay was launched. The result is a closed feasibility gate,
not a performance row.

Detailed notes:

```text
Medusa_project/docs/GATE_2026-05-19.md
```
