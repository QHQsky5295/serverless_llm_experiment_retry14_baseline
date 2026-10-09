# D379 7B C=8 full W0 development replay

**D380 clarification:** the launcher enables real HTTP artifact transfer; the
legacy `local_sim_no_delay` label describes the zero-delay limiter setting and
does not identify the transport. D379 also differs from D371 in profiles and
two policy opt-ins, so it is not a capacity-only causal comparison. See
[D380](D380_PROTOCOL_RECONCILIATION.md). The historical registration below and
the running configuration are otherwise retained.

D379 extends the successful D378 profile-aligned C=8 diagnostic to the frozen
4,000-request W0 trace. It keeps the D321/D298 C=8 profile identities, current
source-identity code, fixed-length greedy contract, 500-adapter subset,
no-delay local-sim transport and all controller values unchanged. The remote
7B service must be authenticated and healthy before launch and remain active
until the completion receipt is captured.

D379 is a development replay, not a formal G1/G2 result: the configuration
does not provide an independent numerical correctness denominator, and it uses
the existing seed-42 trace rather than a recovered seed-41 validation trace.
It is intended to measure the full-trace lifecycle/latency behavior of the
capacity-isolated C=8 envelope and to compare it descriptively with D371; it
does not authorize a figure, baseline ranking, or 3B work by itself.

The run has a fresh output and cache identity. All startup, disk, memory, GPU,
remote-health and cleanup gates remain unchanged. Any incomplete or failed run
is retained and excluded rather than resumed in place.
