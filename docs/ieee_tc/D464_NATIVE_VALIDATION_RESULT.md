# D464 native100 result: correctness gate passed, full performance still open

The original first100 requests were run once using the D463 production candidate,
D458 configuration (exclusive paths only), and existing real published-gzip
delivery. Execution checkpoint1a081bc7c1ea216c01f320f2767d05f36a2118d4.

| Evidence | Observed value |
|---|---:|
| Terminal/native/input/adapter-bound requests | 100/100 |
| Native output tokens | 17369 |
| Common joint SLO | 99/100 |
| Mean/P95 TTFT | 825.733/1898.895 ms |
| Mean/P95 TPOT | 28.512/36.520 ms |
| HOST queue preparation records | 63 |
| Dispatch tiers: GPU/HOST/NVMe/remote | 65/3/3/29 |
| Initial/natural ready activations | 1/3 |
| Remote transfers/wire bytes | 29/50650865 |
| Request-time packing | 0 |
| Observed service memory peak | 18753241088 B |
| high/max/OOM/swap | all zero |

Native TTFT/TPOT reconstruction errors are zero. Both-end transfer IDs, byte
counts, published archive and content evidence agree. All dispatch fields are
complete; the specific queue path executes. Independent numerical correctness
remains open and raw n_correct stays null. Binding/count evidence is not a
claim that all kernels have been independently numerically verified.

The short-prefix lifecycle total656.881695521 GPU-s includes startup and release;
it is NOT a4000-request estimate or ranking. No comparison to full-run resource
efficiency, no repeated-run mean/CI and no figure. Hypothesis acceptance remains
pending one new-key full4000 replay with unchanged D458 parameters.

GPU contexts and the owned service group were released before remote shutdown.
Auxiliary closure used captured InvocationID/inode and actual recursive member
reads. Remote journals were copied only after measurement and SHA checked.
Source result SHA5ac6dc4e5e08a7143ac0aefc539ee35f107080860f0f9db4fcb02e67879c6784.

One inherited optional analysis label read the absent `reason` field; curated
analysis uses the actual `trigger_reason`, with the original analysis retained.
No raw record or count changed. This is offline derivation, not a rerun.

Disk headroom was near the new-run threshold, so the already closed D458
main_outcome was losslessly compressed with the existing tool: source SHA and
two independent decompression SHAs must agree before removing the uncompressed
duplicate. No weights, trace, pool, historical protected result or environment
was deleted. The dedicated storage receipt preserves the old path/SHA mapping.
