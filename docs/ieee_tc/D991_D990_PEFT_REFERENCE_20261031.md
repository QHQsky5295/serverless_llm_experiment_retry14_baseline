# D991 — PEFT numerical reference for D990 (2026-10-31)

D991 reuses the existing bounded PEFT reference protocol, but points it at the
D990 native token-ID witness and the exact D990 trace/config. It compares the
same request IDs, adapter IDs, canonical prompt hashes and fixed target lengths
without changing the production serving path. The output is a bounded witness
audit only: it can report exact sequence agreement and length/contract failures,
but it cannot certify the 4,000-request D989 performance run.

Observed outcome: the native/PEFT witness matched exactly for 94 of 100
requests. Prompt identity and request mapping were complete; the six remaining
requests are retained as mismatches rather than silently discarded. Exact
sequence consistency is therefore `pass=false`; independently, the bounded
scope requires `semantic_full_pool_qualification=false`. This is not evidence that the D989
4,000-request run is formally correct. The previous bounded witness recorded
92/100; these different executions do not establish an improvement in numerical
correctness. The independent HF/PEFT reference remains a diagnostic rather
than the production-backend oracle.

At the first divergence, all six native decisions select one of the PEFT top-two tokens:
four have a reported top-two logit gap of 0.015625 and two have a gap of zero.
All native and reference lengths equal the targets. This is consistent with
numerical near-ties, but does not prove numerical equivalence or exclude wrong
adapter execution; the six exact mismatches and unknown full-run correctness
remain. No repeat witness or new performance replay is warranted solely to seek
a higher match percentage. Use the existing native binding evidence and retain
the numerical limitation in the complete experiment comparison.

