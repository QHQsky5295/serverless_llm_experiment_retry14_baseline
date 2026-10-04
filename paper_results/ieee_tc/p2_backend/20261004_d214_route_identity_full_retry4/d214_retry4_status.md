# D214 7B route identity-recheck full replay (retry4)

- Population: 4000/4000 terminal, 4000 successful, 4000 native-contract complete; target values different from 256: 3055; unique request IDs: 4000.
- Lifecycle: physical GPU-s observed **15907.670119**, 4 journals with exact acquire → worker_spawn → native_workers → native_workers_exited → release order; open leases: 0.
- Delivery: 132 outcome transfers joined one-to-one to the copied remote journal; the file contains 266 lines, of which 134 are retained out-of-run records excluded from retry4 bytes/counts. Joined records report prepublished gzip and no request-time packing; joined server bytes: 131480060.
- Descriptive service values: E2E TTFT mean/P95/P99 1953.0/4233.0/6262.0 ms; service TTFT 686.0/1875.0/3868.0 ms; TPOT 37.2 ms; E2E 6285.0/12522.0/14907.0 ms.
- Qualification: launch and cleanup passed, but formal comparison remains **not qualified**: `n_correct` is unknown and the displayed 5000-ms SLO is TTFT-only, not the frozen joint TTFT/TPOT SLO.
- Provenance: `d214_retry4_curated.json` contains source hashes; raw evidence remains immutable under the qualification result directory.
