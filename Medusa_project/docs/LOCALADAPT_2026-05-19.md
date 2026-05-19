# Medusa Local Adaptation Gate: 2026-05-19

This note continues the official Medusa gate after the user clarified that
SPDK, DPDK, and GDRCopy path problems should be treated as local adaptation
work rather than an immediate impossibility claim.

## Isolation Boundary

- Official Medusa clone:
  `/home/qhq/serverless_llm_baselines/vendor_new_baselines/Medusa_main_20260518`
- Upstream commit: `6581d2e5ec8fa4ecdabcdb50560982a78ea3ca89`
- Local adaptation patch:
  `patches/Medusa_localadapt_20260519.patch`
- New isolated env for the continuation:
  `medusa_localadapt_20260519`
- Local SPDK-Medusa clone:
  `/home/qhq/serverless_llm_baselines/vendor_new_baselines/SPDK-Medusa_20260519`
- SPDK-Medusa commit: `3dd897f8406f0824d6de08d0cb21df9e0f9ed76c`
- Local GDRCopy clone:
  `/home/qhq/serverless_llm_baselines/vendor_new_baselines/gdrcopy_20260519`
- GDRCopy commit: `737708f3b5955cb6ef6b47bba35600a31bce4222`

The continuation did not modify the closed ServerlessLLM-new env, old
ServerlessLLM code, `figs/`, `paper_results/final_v2/`, or the true-remote
mirror artifacts.

## What Was Adapted

The local patch is intentionally narrow:

- Adds CUDA 12.3+ compatible `cuGraphGetEdges(..., nullptr, &savedEdges)`.
- Replaces hard-coded `/home/zsx/spdk` and `/usr/local/lib` build/link paths
  with:
  - `MEDUSA_SPDK_ROOT`
  - `MEDUSA_GDRCOPY_ROOT`
  - `MEDUSA_EXTRA_INCLUDE_DIRS`
  - `MEDUSA_EXTRA_LIB_DIRS`
- Passes SPDK/DPDK/GDRCopy include paths to both C++ and nvcc compilation.
- Keeps Medusa's SPDK/DPDK/GDRCopy link semantics; no SPDK stubs were used.
- Adds a default-off smoke-test switch, `MEDUSA_SPDK_NO_HUGE=1`, to verify
  whether runtime can pass DPDK hugepage initialization. Formal Medusa runs
  should not use this switch unless explicitly documented as a non-paper
  deviation.

## Build Evidence

Local dependencies were installed only into `medusa_localadapt_20260519`:
`meson`, `nasm`, `autoconf`, `automake`, `libtool`, `yasm`, `libfuse3`,
`libaio`, `json-c`, `cmocka`, `pyelftools`, and `numactl`.

Build progression:

1. GDRCopy userspace `libgdrapi.so` built under
   `gdrcopy_20260519/local_install`.
2. SPDK-Medusa shared build succeeded after local conda dependency injection.
   Key produced libraries include:
   - `build/lib/libspdk_env_dpdk.so`
   - `build/lib/libspdk_vfio_user.so`
   - `dpdk/build/lib/librte_vhost.so`
   - `isa-l-crypto/.libs/libisal_crypto.so`
3. Medusa `python setup.py build_ext --inplace` succeeded with the local
   adaptation patch.
4. Import verification succeeded:

```text
IMPORT_OK .../vllm/_C.cpython-39-x86_64-linux-gnu.so .../vllm/_moe_C.cpython-39-x86_64-linux-gnu.so
C_ATTRS ['ops', 'tensor_ops', 'test_spdk']
```

## Runtime Evidence

Default SPDK initialization fails on this machine because no 2 MB hugepages are
available:

```text
EAL: No free 2048 kB hugepages reported on node 0
EAL: No free 2048 kB hugepages reported on node 1
EAL: FATAL: Cannot get hugepage information.
spdk_env_init: *ERROR*: Failed to initialize DPDK
Unable to initialize SPDK env
```

With `MEDUSA_SPDK_NO_HUGE=1`, `init_spdk_daemon()` and `fini_spdk_daemon()`
complete, but this is only a smoke check. The machine still has no visible NVMe
or Optane device for Medusa's SPDK materialization path:

- `lspci` / `/sys/class/nvme` / `/dev/nvme*`: no NVMe device visible.
- `lsblk`: only `PERC H330 Adp` backed `sda` is visible.
- `/dev/gdrdrv`: absent.
- `sudo -n true`: unavailable, so hugepage provisioning, driver loading, and
  PCI binding cannot be performed by this process.

## Decision

Medusa is now a successful local build/import gate, but it is still not a
formal comparison candidate on the current machine. The missing formal
requirements are runtime/system requirements, not path-editing requirements:

- provisioned hugepages or an explicitly justified no-huge deviation,
- SPDK-accessible NVMe/Optane devices matching Medusa's materialization model,
- any required GDRCopy kernel device setup,
- model/workload adapter for the closed Llama-2 7B and Llama-3.2 3B LoRA
  true-remote traces.

Do not add Medusa to the main comparison table or figures unless those runtime
requirements are satisfied and the closed true-remote workload can be replayed
without changing variables.
