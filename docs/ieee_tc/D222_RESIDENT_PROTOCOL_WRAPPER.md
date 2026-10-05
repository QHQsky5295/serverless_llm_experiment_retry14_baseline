# D222 — Resident-vLLM protocol wrapper (retired prototype)

Status: **RETIRED / NOT QUALIFIED**.  This document records an exploratory
prototype and its later D222--D224 smoke attempts.  It is not an implementation
checkpoint for a Resident reference and must not be used to authorize a GPU
campaign or a resource comparison.

## What changed

The baseline harness briefly had a separate
`run_vllm_resident_reference_v1.sh` entrypoint.  It reused the ordinary vLLM
runner, frozen trace, frozen 500-adapter subset, native
`fixed_length_greedy_v1` token path, and the prepublished read-only gzip
artifact endpoint.  It did not regenerate weights, traces, or extracted
adapter pools.  The entrypoint and the `VLLM_RESIDENT_PROTOCOL_V1=1` path are
now fail-closed and retained only as historical source evidence.

When `VLLM_RESIDENT_PROTOCOL_V1=1`, the existing runner additionally:

1. starts service and replay in distinct user-systemd slices with explicit
   memory/task limits;
2. starts each vLLM replica as a service unit in the shared service slice and
   the replay publisher in the auxiliary slice;
3. records a common monotonic clock identity and publishes one deployment
   notice exactly `VLLM_RESIDENT_LEAD_S` seconds before replay business time;
4. requires `dynamic_remote` and records the `prepublished_gzip_v1` delivery
   mode, with no request-time archive packing;
5. records per-GPU CUDA UUID acquisition/hold/release intervals and closes the
   receipt only after an independent NVML census is at the idle baseline.

The receipt helper is
`serverless_llm_baselines/scripts/ieee_tc/resident_protocol_receipt.py`.
The prototype source is backed up at commits `0d01e9a`, `65185e2`, `c3a0215`,
and `0a8e5f7` on the baseline `main` remote branch.  Existing user-dirty
baseline files were not staged.

## Validation boundary

The prototype's syntax and token-contract checks passed, but those checks did
not establish the required protocol.  D222--D224 exposed the following
disqualifying facts:

* the runner used the inherited vLLM 0.10.2 environment rather than the frozen
  Prime/vLLM 0.30.0 backend;
* the effective service cap was 96 GiB with no verified `MemoryHigh`, rather
  than the 72/80 GiB contract, and the limits were attached after anchors were
  started;
* the deployment notice was published only after all replicas were ready, and
  the replay created a separate time origin after tokenizer preparation;
* the receipt treated an initial `nvidia-smi` inventory as GPU acquisition and
  closed all intervals from a final memory-only census, without a physical
  lease, owner identity, or overlap-union proof;
* remote transfer/content identity and service/auxiliary accounting were not
  independently joined to the request path; and
* the launch used `dynamic_remote`, whereas the repository's ordinary formal
  remote-fair vLLM path requires round-local `static_remote` staging unless a
  separately resource-bounded exception has been qualified; and
* native output-token agreement proves the generation contract only, not
  numerical adapter correctness.

Accordingly, D224's 100/100 completion is retained as a **generation-contract
diagnostic** only.  D222 and D223 are launcher/protocol failures.  None of the
three attempts is eligible for G1/G2, joint-SLO qualification, lifecycle
GPU-s/request, or a Resident baseline ranking.  No 1,000-request follow-up is
authorized from this prototype.  A future Resident reference must reuse the
already-qualified common preflight, resource-domain, physical lease, common
clock, and cleanup infrastructure instead of extending this wrapper.
