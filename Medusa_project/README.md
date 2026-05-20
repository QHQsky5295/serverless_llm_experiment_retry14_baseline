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
- Local adaptation env: `medusa_localadapt_20260519`
- Gate result: local adapted build/import succeeds; formal runtime remains
  blocked by current-machine SPDK runtime requirements.

## Gate Summary

The official build gate reached `vllm._C` compilation after CUDA include-path
repair, but failed in `csrc/cuda_graph.cu` on a CUDA Graph API signature
mismatch. The continuation patched that CUDA Graph call and replaced the
hard-coded SPDK/DPDK/GDRCopy paths with local environment variables. With
locally built SPDK-Medusa and GDRCopy userspace libraries, `vllm._C` and
`vllm._moe_C` build and import successfully.

The remaining blocker is runtime/system state, not path editing: the machine
has zero configured hugepages by default, no visible NVMe/Optane device for
SPDK I/O, no `/dev/gdrdrv`, and no passwordless sudo for hugepage/driver/PCI
binding setup. Therefore no formal true-remote LoRA replay was launched. The
result is a closed build/import gate, not a performance row.

The 2026-05-21 runtime recheck confirms the blocker is still present after the
dLoRA run cleaned up: `HugePages_Total=0`, `/dev/gdrdrv` is absent, no NVMe or
Optane device is visible, `/sys/class/uio` is absent, `/dev/vfio/vfio` has no
bound device node, and `sudo -n true` still requires a password.

Detailed notes:

```text
Medusa_project/docs/GATE_2026-05-19.md
Medusa_project/docs/LOCALADAPT_2026-05-19.md
Medusa_project/docs/RUNTIME_RECHECK_2026-05-21.md
```
