# D378 7B capacity-revalidation diagnostic: closure

Date: 2026-10-10

## Eligibility

D378 is a bounded seed-42, 100-request development diagnostic. It used the
existing 7B trace and adapter subset, fixed-length greedy generation, the C=8
D321/D298 profile-aligned configuration, and the authenticated remote 7B
prepublished gzip service. It is not a formal run (`formal_run=false`), does
not contain a frozen correctness field (`n_correct=null`), and is not eligible
for G1/G2 ranking or figure generation.

The run completed 100/100 terminal requests with zero terminal failures and
100/100 `output_contract_match=true` records. Completion tokens came from
native vLLM token IDs; each request's completion count equaled its requested
count. This is generation-contract evidence, not numerical adapter correctness.

Raw evidence hashes:

| artifact | SHA-256 |
|---|---|
| `launch.json` | `22c5136aeae617e0c9455ae6c16c386637417d80c73f9aa5f8e333eac4ace628` |
| result JSON | `6888356b7d64910d73d4517541b8830b9b63139f28b97f950f9f22abf4f746d6` |
| physical summary | `46d5b05180222c8b785bf2b197dbef711f7ebb30a14125d5ff80eec594abbeb1` |
| request terminals | `24f3732cd2999996752ae1792f8800849f787995a4324b0d9f29d9a8ec6af5f5` |

## Descriptive observations

The run-level summary reports mean/P95 TTFT `903.6/2710.8 ms`, mean/P95 E2E
`5664.8/9070.9 ms`, mean TPOT `26.29 ms`, 99% TTFT-SLO attainment, and
`0.5255` requests/s over the 190.3-second diagnostic window. The native
physical deployment ledger observed `800.0264` GPU-seconds across four
devices, with no open leases at closure. The legacy scenario accounting reports
`541.5024` lifecycle GPU-seconds; because these are different accounting
contracts, neither is used as a formal comparison value here. The diagnostic
physical ledger is retained for later audit.

Readiness tiers immediately before dispatch were GPU 51, remote 29, HOST 15
and NVMe 5. The run recorded 29 successful remote transfers, 71 non-remote
dispatches, 49 adapter-replica mismatch flags, and three scale-up events that
created four total dedicated activations. These counts demonstrate that the
capacity path and tier metadata were exercised; they do not establish a causal
benefit or a controller optimum.

After the completion receipt was captured, the remote 7B service was stopped;
its port was absent, all local GPUs were idle, and the user-owned manifest
remained untouched. No figure was generated.
