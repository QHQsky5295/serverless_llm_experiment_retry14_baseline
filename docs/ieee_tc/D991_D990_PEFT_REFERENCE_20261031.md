# D991 — PEFT numerical reference for D990 (2026-10-31)

D991 reuses the existing bounded PEFT reference protocol, but points it at the
D990 native token-ID witness and the exact D990 trace/config. It compares the
same request IDs, adapter IDs, canonical prompt hashes and fixed target lengths
without changing the production serving path. The output is a bounded witness
audit only: it can report exact sequence agreement and length/contract failures,
but it cannot certify the 4,000-request D989 performance run.

