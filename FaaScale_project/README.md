# FaaScale / LambdaScale Baseline Project

This directory is the project entry point for the 2026-05-19
FaaScale/LambdaScale reproduction gate. It is kept separate from the closed
paper baseline projects and from the ServerlessLLM-new and Medusa gate work.

## Current Status

- Paper/system: FaaScale / LambdaScale.
- Upstream source:
  `/home/qhq/serverless_llm_baselines/vendor_new_baselines/lambda-scale_main_20260518`
- Upstream commit: `9db210fcb6979f7c1f73f9819a77e0edb6c5e343`
- RDMA helper source:
  `/home/qhq/serverless_llm_baselines/vendor_new_baselines/rdma-p2p_main_20260518`
- RDMA helper commit: `ed83237439d2103141fbc7c9b97f348055b6cb53`
- Local adaptation env: `faascale_20260519`
- Local adaptation patch:
  `patches/RDMA_P2P_localadapt_20260519.patch`
- Gate result: local imports, IPC extension, and RDMA-P2P Python binding can
  be built/imported after isolated dependency repair, but formal true-remote
  LoRA replay is blocked on this machine.

## Gate Summary

The official LambdaScale checkout does not import directly because the package
imports `test_bed_local` while the repository directory is named
`lambda-scale`. A local symlink shim inside an isolated clone fixes that import
layout. The generated protobuf files also require `protobuf==3.20.x`; after
installing minimal dependencies into `faascale_20260519`, the core utility
import succeeds and reports Llama-2 7B as supported but Llama-3.2 3B as
unsupported.

The local IPC extension builds and imports with the correct torch library
runtime path. The RDMA-P2P helper required local build repairs: missing Derecho
CMake/template files were copied from Derecho v2.4.0 into the local adaptation
copy, `pybind11` was supplied locally, `rdma-core` headers came from the conda
env, and CUDA 13 required a narrow `cuCtxCreate` API patch. The targeted
`rdmc_shn` Python binding then built and imported.

Runtime still cannot enter a formal paper-equivalent run here: `/dev/infiniband`
is absent, `/sys/class/infiniband` has no usable devices, `wrapper_initialize`
finds zero IB devices and returns `False`, and passwordless sudo is unavailable
for driver/device setup. Static capability checks also found no LoRA/PEFT path
and no Llama-3.2 3B model config.

The 2026-05-21 runtime recheck confirms the same boundary on the clean machine:
`/dev/infiniband` is still absent, `/sys/class/infiniband` exposes no usable
device, only PCI-level Broadcom RDMA-capable Ethernet controllers are visible,
and `sudo -n true` still requires a password.

Detailed notes:

```text
FaaScale_project/docs/GATE_2026-05-19.md
FaaScale_project/docs/RUNTIME_RECHECK_2026-05-21.md
```
