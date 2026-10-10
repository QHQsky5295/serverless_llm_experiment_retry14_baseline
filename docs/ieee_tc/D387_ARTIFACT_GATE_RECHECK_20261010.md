# D387 — artifact-node independent disk gate recheck

Date: 2026-10-10.  This is a read-only policy and environment recheck; it
does not start inference, an artifact service, packing, cleanup, or a remote
stress workload.

## Decision

The artifact-only node does not use the inference host's fixed 150-GiB start
floor.  The implemented gate is the frozen independent rule

\[
P_{pack}=\sum_i q_i g_i,\qquad
F_{required}=S+\lceil1.5(P_{pack}+G)\rceil,
\]

where `q_i` is the evidence-backed maximum concurrent temporary archives,
`g_i` is the remaining byte-growth bound per archive, `G` is remaining log and
failure-close growth, and `S>0` is the explicitly frozen safety reserve.  The
gate rejects missing, non-positive, or unknown bounds; it does not reduce
application concurrency to make a check pass.

The implementation and CLI were already committed in the branch history
(`f1f36110`, `3ae5fba`) and are ancestors of the current pushed HEAD.  The
inference-host `disk_required()` function remains unchanged and is not called
by `artifact-disk-check`.

## Current remote observation

The authenticated read-only check used the existing management alias
`primelora-artifact-174` and did not modify the host:

| observation | value |
|---|---:|
| filesystem | `/dev/mapper/ubuntu--vg-ubuntu--lv`, ext4 |
| free bytes | 139,212,648,448 |
| free inodes | 30,883,816 (statfs readback) |
| frozen required bytes | 97,407,250,432 |
| margin | 41,805,398,016 bytes |
| artifact 7B service | inactive |
| artifact 3B service | inactive |
| artifact port 18081 | closed |
| remote NIC | 1000 Mb/s, full duplex, carrier up |

The bound is the previously qualified two-model serial archive bound with
64-MiB log growth and a 16-GiB safety reserve.  This recheck does not claim a
new packing workload or alter that bound.  Any future increase in concurrent
packing or archive growth requires a new explicit bound and a new read-only
check.

## Local qualification evidence

The focused artifact-gate test subset passed (`5 passed`).  A full run under
the model-serving virtual environment had 84 passed and 4 failures, all in
unrelated watchdog/PID-handle protocol tests: that interpreter lacks the
qualified `signal.pidfd_send_signal` primitive, so those failures do not
exercise the artifact arithmetic or filesystem gate.  No serving code was
changed to bypass them; the qualified launcher continues to use its existing
system-Python watchdog path.

## Experimental boundary

This policy change affects only whether the artifact node can safely host
temporary delivery work.  It does not change inference-host disk, memory,
swap, GPU, or OOM protections; it does not delete unique weights, adapters,
traces, raw results, failure evidence, or source history.

The remote link state is recorded separately from the old D77 100-Mbps
observation.  Current main runs use `link_state=1000_full`; historical
`100_full` is not pooled with them.  Request-triggered remote read, transfer,
receive, extraction, and materialization remain in user TTFT/E2E.  One-time
administrative packing, publication, health checks, and unrelated remote work
are recorded as deployment-management spans and excluded from request E2E;
they are never subtracted from an already observed E2E value.
