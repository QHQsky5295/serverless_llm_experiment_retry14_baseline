# Medusa Gate 2026-05-19

This folder stores the non-overwriting evidence for the Medusa reproduction
gate. It is part of the new serverless baseline campaign, not part of the
default paper data snapshot.

## Source

- System: Medusa, ASPLOS 2025.
- Upstream commit: `6581d2e5ec8fa4ecdabcdb50560982a78ea3ca89`.
- Local source:
  `/home/qhq/serverless_llm_baselines/vendor_new_baselines/Medusa_main_20260518`.
- Baseline project entry:
  `/home/qhq/serverless_llm_baselines/Medusa_project/`.

## Gate Result

Decision: do not enter Medusa into the formal comparison table or figures on
the current machine.

Evidence:

- `import vllm` failed before build with missing `vllm._C`.
- The first official build attempt failed because `cusparse.h` was not visible.
- After adding conda CUDA target include/library paths, `vllm._moe_C` compiled
  and the build reached `vllm._C`, but failed at `csrc/cuda_graph.cu(131)` with
  a CUDA Graph API signature mismatch.
- Static inspection shows the official build also hard-codes SPDK/DPDK/GDRCopy
  paths under `/home/zsx/spdk`, while this machine has no such stack.
- Official scripts hard-code `/home/zsx` model paths and OpenLLaMA/Qwen/Yi/Falcon
  model names; Llama-2 7B and Llama-3.2 3B true-remote LoRA replay would require
  nontrivial adaptation after the build problem is solved.

Local adaptation continuation:

- `Medusa_localadapt_20260519.patch` patches the CUDA Graph API call,
  parameterizes SPDK/DPDK/GDRCopy paths, passes local include paths to nvcc,
  and adds a default-off `MEDUSA_SPDK_NO_HUGE=1` smoke switch.
- Local SPDK-Medusa commit:
  `3dd897f8406f0824d6de08d0cb21df9e0f9ed76c`.
- Local GDRCopy commit:
  `737708f3b5955cb6ef6b47bba35600a31bce4222`.
- With `medusa_localadapt_20260519`, local GDRCopy userspace lib and
  SPDK-Medusa shared libs build successfully.
- Medusa `vllm._C` and `_moe_C` build and import successfully.
- Default SPDK runtime still fails because `HugePages_Total=0`.
- With `MEDUSA_SPDK_NO_HUGE=1`, `init_spdk_daemon()` / `fini_spdk_daemon()`
  pass as a smoke test, but this is not a paper-equivalent Medusa run.
- Current machine still has no visible NVMe/Optane device, no `/dev/gdrdrv`,
  and no passwordless sudo for hugepage/driver/PCI binding.

## Files

- `20260519_medusa_build_gate_cpath_continue.log.gz`: compressed build-gate log.
- `20260519_gdrcopy_lib_install.log.gz`: GDRCopy userspace library build log.
- `20260519_spdk_medusa_configure_localadapt7_vfio.log.gz`: successful local
  SPDK-Medusa configure log.
- `20260519_spdk_medusa_make_localadapt5_dpdklibpath.log.gz`: successful local
  SPDK-Medusa build log.
- `20260519_medusa_build_gate_localadapt3_nvccincludes.log.gz`: successful
  Medusa `_C` build log after nvcc include adaptation.
- `20260519_medusa_build_gate_localadapt4_nohugeflag.log.gz`: incremental
  rebuild log for the default-off no-huge smoke switch.
- `20260519_medusa_import_localadapt1.log.gz`: import verification for `_C`
  and `_moe_C`.
- `20260519_medusa_runtime_init_spdk_localadapt1.log.gz`: default SPDK init
  failure with no hugepages.
- `20260519_medusa_runtime_init_spdk_nohuge_localadapt1.log.gz`: no-huge smoke
  init/fini pass.
- `20260519_medusa_runtime_test_spdk_gdb_localadapt1.log.gz`: gdb backtrace for
  the raw `test_spdk.test()` helper, which is not a valid initialized runtime
  entry.
- `20260519_medusa_localadapt_env_explicit_after.txt.gz`: conda explicit env
  list for `medusa_localadapt_20260519`.
- `Medusa_localadapt_20260519.patch`: local Medusa source patch.
- `SHA256SUMS`: checksum for the compressed log.
