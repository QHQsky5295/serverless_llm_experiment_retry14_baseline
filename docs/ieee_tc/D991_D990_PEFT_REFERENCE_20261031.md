# D991 — PEFT numerical reference for D990 (2026-10-31)

D991 reuses the existing bounded PEFT reference protocol, but points it at the
D990 native token-ID witness and the exact D990 trace/config. It compares the
same request IDs, adapter IDs, canonical prompt hashes and fixed target lengths
without changing the production serving path. The output is a bounded witness
audit only: it can report exact sequence agreement and length/contract failures,
but it cannot certify the 4,000-request D989 performance run.

Observed outcome: the native/PEFT witness matched exactly for 94 of 100
requests. Prompt identity and request mapping were complete; the six remaining
requests are retained as mismatches rather than silently discarded. This is a
bounded numerical witness (and is therefore `pass=false` and
`semantic_full_pool_qualification=false`), not evidence that the D989
4,000-request run is formally correct. The 94% agreement is modestly above the
previous 7B bounded witness (92/100), but the independent HF/PEFT reference
remains a diagnostic rather than the production-backend oracle.

