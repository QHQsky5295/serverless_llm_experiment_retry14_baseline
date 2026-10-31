# D990 — 7B bounded correctness witness for D989 (2026-10-31)

D990 reuses the frozen W0 trace, canonical prompts, target map and 500-adapter
pool, but executes only the pre-registered first 100 requests with native
token-ID capture. Its control path is the D989 busy-runtime service-rate and
independent forecast-dwell candidate. The run is a correctness diagnostic;
its latency and resource values must not enter the performance comparison.

The captured native sequences will be checked against the existing independent
HF/PEFT reference protocol with the same request IDs, prompt hashes, adapter
identities and target lengths. Exact matches, length mismatches and numerical
near-ties remain separately reported. Even a complete 100-request witness
cannot set `n_correct=4000` for D989 or promote the 4,000-request run to formal
G1/G2 qualification.

Observed outcome: 100/100 requests completed without failure; all 100 satisfied
the fixed-length contract, exposed a native terminal event, and emitted a
native token-ID SHA-256. The bounded run used 43 distinct canonical prompts and
46 target lengths. Its physical owner-union GPU time was 335.415 GPU-s, but this
number is diagnostic only and is not normalized into the formal comparison.

