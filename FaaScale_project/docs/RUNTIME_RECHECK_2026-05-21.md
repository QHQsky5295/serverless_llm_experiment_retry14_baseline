# FaaScale / LambdaScale Runtime Recheck: 2026-05-21

This recheck was run after the dLoRA 7B final cache gate, on a clean machine
state, to verify whether the remaining FaaScale/LambdaScale blocker is still
only a path/library issue or a real runtime/device/workload issue.

## Commands Checked

- `/dev/infiniband`.
- `/sys/class/infiniband`.
- `lspci` for RDMA-capable devices.
- loaded RDMA-related modules via `lsmod`.
- passwordless sudo via `sudo -n true`.
- the previously recorded source capability probes for Llama-3.2 3B and LoRA.

## Current Machine State

- `/dev/infiniband` is absent.
- `/sys/class/infiniband` exists only as a class path; no usable IB device is
  exposed under it.
- `lspci` shows Broadcom BCM57416 RDMA-capable Ethernet controllers, but the
  userspace verbs device is not exposed.
- `lsmod` shows `ib_core` and `bnxt_en`, but no usable `/dev/infiniband`
  device path.
- `sudo -n true` requires a password.
- The existing source probe still shows no Llama-3.2 3B config and no
  LoRA/PEFT request path in the official tree.

## Decision

The local package/import/IPC/RDMA-binding adaptation remains successful, but a
paper-equivalent FaaScale/LambdaScale formal replay is still blocked on this
machine. The remaining blockers are not Python package paths: the RDMA device
stack is not exposed to userspace, the current user cannot provision or bind
devices without a password, and the official source lacks the required
Llama-3.2 3B and many-LoRA workload path.

Do not start a true-remote formal FaaScale replay on this machine unless a
usable RDMA device appears under `/dev/infiniband`, `wrapper_initialize`
succeeds, and the model/LoRA workload path is adapted without changing the
closed remote workload variables.
