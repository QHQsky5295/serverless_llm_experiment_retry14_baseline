# FaaScale / LambdaScale Gate 2026-05-19

This folder stores non-overwriting evidence for the FaaScale/LambdaScale
reproduction gate. It is part of the new serverless baseline campaign and is
not part of the default paper data snapshot.

## Source

- System: FaaScale / LambdaScale.
- LambdaScale upstream commit:
  `9db210fcb6979f7c1f73f9819a77e0edb6c5e343`.
- RDMA-P2P upstream commit:
  `ed83237439d2103141fbc7c9b97f348055b6cb53`.
- Baseline project entry:
  `/home/qhq/serverless_llm_baselines/FaaScale_project/`.
- Local adaptation env: `faascale_20260519`.

## Gate Result

Decision: do not enter FaaScale/LambdaScale into the formal comparison table
or figures on the current machine.

Evidence:

- Raw official import failed with `ModuleNotFoundError: test_bed_local`.
- An isolated symlink shim fixed the package-layout mismatch.
- Generated protobuf files required `protobuf==3.20.x`; after minimal
  dependency repair, utility import passed.
- Capability probe reports Llama-2 7B support but no Llama-3.2 3B config:
  `llm7 True`, `llama32 False`.
- Targeted LoRA/PEFT/QLoRA source search found no workload path.
- LambdaScale IPC extension built and imported after setting the torch library
  runtime path.
- RDMA-P2P required local Derecho metadata, pybind11, conda `rdma-core`
  headers, and a CUDA 13 `cuCtxCreate` patch.
- The targeted `rdmc_shn` Python binding built and imported.
- RDMA runtime initialization still finds zero IB devices and returns `False`.
  `/dev/infiniband` is absent, the InfiniBand sysfs class exposes no usable
  device, and passwordless sudo is unavailable for driver/device setup.
- A 2026-05-21 clean-machine recheck confirmed `/dev/infiniband` is still
  absent; `/sys/class/infiniband` has no usable device; only PCI-level Broadcom
  RDMA-capable Ethernet controllers are visible; and passwordless sudo is
  unavailable.

## Files

- `20260519_faascale_official_import_raw.log.gz`: raw official import failure.
- `20260519_faascale_import_symlink_utils.log.gz`: symlink import plus
  protobuf runtime failure.
- `20260519_faascale_pip_min_deps.log.gz`: minimal dependency installation.
- `20260519_faascale_import_symlink_utils_afterdeps.log.gz`: post-dependency
  utility import and Llama-2/Llama-3.2 capability check.
- `20260519_faascale_ipc_build.log.gz`: IPC extension build.
- `20260519_faascale_import_ipc_modelinfo*.log.gz`: IPC/model-loader import
  checks before and after torch library path repair.
- `20260519_faascale_model_capability_probe.log.gz`: model support probe.
- `20260519_faascale_lora_peft_search.log.gz` and
  `20260519_faascale_lora_adapter_search.log.gz`: LoRA/adapter source search.
- `20260519_faascale_llama2_empty_model_probe.log.gz`: empty Llama-2 7B model
  construction smoke test.
- `20260519_faascale_env_explicit_final.txt.gz`: final explicit conda env.
- `20260519_rdmap2p_install_*.log.gz`: local RDMA-P2P dependency installs.
- `20260519_rdmap2p_build_*.log.gz`: RDMA-P2P build attempts and targeted
  `rdmc_shn` build success.
- `20260519_rdmap2p_import_pyrdmc.log.gz`: `pyrdmc` import smoke.
- `20260519_rdmap2p_runtime_init_no_infiniband.log.gz`: runtime gate showing
  zero IB devices.
- `RDMA_P2P_localadapt_20260519.patch`: CUDA 13 local adaptation patch.
- `RUNTIME_RECHECK_2026-05-21.md`: clean-machine runtime/device recheck.
- `SHA256SUMS`: checksums for this gate bundle.
