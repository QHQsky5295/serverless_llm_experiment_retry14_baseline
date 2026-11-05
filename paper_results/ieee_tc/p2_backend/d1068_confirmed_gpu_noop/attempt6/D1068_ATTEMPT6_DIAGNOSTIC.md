# D1068 attempt 6: confirmed-GPU no-preparation diagnostic

## Scope

This run tests whether the D1068 no-preparation proof is exercised in the full Llama-3.2-3B W1 replay. It is a mechanism and lifecycle diagnostic, not a publishable performance comparison.

The replay completed 4,000/4,000 requests with zero failed requests. It used the frozen W1 trace, the existing 500-adapter subset, `fixed_length_greedy_v1`, real HTTP delivery with prepublished gzip archives, and no request-time remote packing. The raw result is retained under the path recorded in `sources.json`.

## Evidence observed

- 744 `confirmed_gpu_no_preparation` proof receipts were observed.
- 89 `ieee_residency_zero_demand_v1` rounds were observed in the coordination record.
- 8 confirmed file-copy no-op receipts were observed.
- The result contained 132 remote artifact transfers and zero request-time pack operations.
- The service completed 9 scale-up and 7 scale-down events.
- Cleanup completed and the launch receipt confirmed release of the native GPU context.

The JSON outcome contains public/private copies of some event arrays. Counts above deliberately use the proof flag and the first `coordination_after_shutdown.ieee_residency_noop_rounds` array once; duplicated serialization is not treated as additional execution.

## Why it is not a canonical performance result

Attempt 6 was started directly from the user-scope runner. It did not set the frozen native HOST descriptor variables used by the qualified D1047 wrapper:

```text
FAASLORA_IEEE_HOST_INVENTORY_MODE=descriptor
FAASLORA_IEEE_HOST_DESCRIPTOR_LIBRARY=.../_ieee_native_host_descriptor.cpython-312-x86_64-linux-gnu.so
FAASLORA_IEEE_HOST_DESCRIPTOR_SHA256=b76733db910a0041409ce375e7b5a281dabadd6d0e5796699bf8fe574e293044
```

The recorded descriptor in `main_outcome.json` is `python_tensor_descriptor_v1`, whereas the frozen comparison path requires the native descriptor. This changes the host-observation implementation and therefore invalidates a latency or physical-GPU comparison with D1047. The run is consequently marked `formal_comparison_qualified=false`; its TTFT, E2E, SLO, and physical-GPU values are diagnostic only. No figure or G1/G2 claim may consume them.

This is an execution-contract mismatch, not evidence that the no-preparation proof is ineffective. The proof did trigger, so the mechanism question has useful evidence. The raw run and all failed earlier attempts remain untouched.

## Next canonical action

Run exactly one proper D1068 3B W1 replay with the existing qualified wrapper (or equivalent explicit native descriptor exports), keeping the same code head, config, trace, adapter subset, remote delivery contract, and generation contract. It must be a fresh run key and must not overwrite attempt 6. Do not create a short substitute run or tune from this ineligible performance result.

## Source integrity

See `diagnostic_result.json` and `sources.json` for the raw paths, SHA-256 values, code/config/trace identities, and the exact boundary between diagnostic evidence and canonical comparison evidence.
