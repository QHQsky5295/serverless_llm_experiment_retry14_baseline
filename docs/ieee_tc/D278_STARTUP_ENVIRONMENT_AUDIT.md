# D278 startup-environment audit

This is an offline comparison of two already completed 7B PrimeLoRA
diagnostics.  It does not modify either run, promote a diagnostic to G1/G2, or
authorize a new replay.

## Evidence

The curated table is
`paper_results/ieee_tc/p2_backend_qualification/20261006_d278_startup_environment_audit/d278_vs_d219_startup_audit.csv`.
The source result and launch-receipt SHA-256 values are stored in that table.
Both runs use the same 7B model, seed-42 source trace, 500-adapter subset,
fixed-output contract, vLLM environment and prepublished gzip delivery.  D278
adds the eight-phase readiness diagnostic; it is not a matched performance
replicate of D219.

The raw source paths are the existing files under
`results/ieee_tc/p2_backend_qualification/d219_20261005/7b_confirmed_snapshot_cache_full_replay/`
and
`results/ieee_tc/p2_backend_qualification/d278_20261006/7b_readiness_8phase_full_retry9/`.
No raw result was copied into Git.

## Observed comparison

| Quantity | D219 | D278 |
|---|---:|---:|
| engine/preload interval (s) | 53.61 | 204.52 |
| mean user TTFT (ms) | 1,508.36 | 6,895.51 |
| P95 user TTFT (ms) | 3,620.02 | 22,698.65 |
| first-100 mean TTFT (ms) | 2,574.95 | 142,568.83 |
| mean dispatch/admission wait (ms) | 814.07 | 6,213.82 |
| mean service TTFT (ms) | 694.29 | 681.68 |
| physical GPU-s | 15,911.71 | 15,349.59 |
| parent swap at preflight (bytes) | 220,192,768 | 12,049,293,312 |
| parent memory PSI avg10 | 0.00 | 0.13 |
| natural first-service observations | 0 | 10 |

The native service TTFT is essentially unchanged (681.68 ms in D278 versus
694.29 ms in D219), while the first 100 requests in D278 are dominated by the
initial engine/preload backlog.  D278's preflight also records approximately
12 GiB of swap in the parent resource domain and non-zero memory pressure;
D219 records approximately 0.22 GiB and zero short-window pressure.  This is
evidence of a different host-resource state, not evidence that confirmed-tier
routing or GPU decode became intrinsically 4--5x slower.

## Decision

D278 is retained as a valid runtime/readiness diagnostic with 4,000 successful
requests and complete dispatch snapshots, but its mean/P95 user latency is not
used as a PrimeLoRA optimization target or as a regression claim against D219.
The next Prime optimization must first establish a clean startup/resource
stratum under the Plan's memory, swap, CPU and disk gates.  No code change is
justified by this audit alone.  A future matched run may test a source-level
readiness hypothesis only after a falsifiable CPU contract check and a clean
preflight; it must preserve the same generation, trace, remote-cache and
resource contracts.
