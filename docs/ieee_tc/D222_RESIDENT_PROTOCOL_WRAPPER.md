# D222 — Resident-vLLM protocol wrapper (implementation checkpoint)

Status: protocol-layer implementation and CPU/contract validation complete;
no model replay or performance result was produced in D222.

## What changed

The baseline harness now has a separate `run_vllm_resident_reference_v1.sh`
entrypoint. It reuses the ordinary vLLM runner, frozen trace, frozen
500-adapter subset, native `fixed_length_greedy_v1` token path, and the
prepublished read-only gzip artifact endpoint. It does not regenerate weights,
traces, or extracted adapter pools.

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
The baseline implementation is backed up at commit `0d01e9a` on its `main`
remote branch. Existing user-dirty baseline files were not staged.

## Validation boundary

`bash -n`, Python byte-compilation, an actual four-GPU receipt init/notice/
finalize round-trip, systemd slice creation/cleanup, and a vLLM dry-run with
the historical 7B W0 trace/subset all passed. The dry-run does not create a
physical lifecycle receipt or start a service. No remote service was started
and no GPU inference was run in D222.

The wrapper is therefore a qualification prerequisite, not a Resident result.
The next step is a fresh 100-request 7B smoke using the remote 7B endpoint,
followed by the prescribed 1,000-request compatibility run only if all
identity, token, delivery, resource-domain, and release gates pass.
