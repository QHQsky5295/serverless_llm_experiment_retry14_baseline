# D948: initial-NVMe preload outcome

The completed candidate is:

`d947_20261030/7b_initial_nvme_preload_barrier_w2`

It used the same 7B model, W2 trace, 500-adapter map, fixed-length greedy
contract, vLLM runtime, and IEEE confirmed controller as D931.  The explicit
candidate difference was a deployment-stage materialization of all 500
published adapters into the 24 GiB NVMe tier, with replay delayed until that
stage and initial runtime activation had completed.

The run completed 4,000/4,000 requests with zero failures.  The native output
contract matched 4,000/4,000 requests and vLLM token IDs were present for all
requests.  Request-level numerical correctness remains `n_correct=null`; no
oracle is inferred from output-contract compliance.

| metric | D931 current control | D947 initial NVMe preload |
|---|---:|---:|
| average TTFT (ms) | 1215.95 | 1144.78 |
| P95 TTFT (ms) | 2997.74 | 2464.45 |
| average service TTFT (ms) | 685.05 | 548.52 |
| P95 service TTFT (ms) | 2144.92 | 1262.77 |
| TTFT-SLO attainment | 98.1% | 98.5% |
| average TPOT (ms) | 31.54 | 31.19 |
| average LoRA I/O (ms) | 237.60 | 82.90 |
| remote transfer count | 497 | 500 (deployment stage) |
| cache-hit rate | 87.33% | 100% |
| physical GPU-seconds | 11716.06 | 13296.43 |
| physical GPU-seconds/request | 2.9290 | 3.3241 |

D947 therefore reduces request latency and eliminates request-time remote
misses, but increases physical lifecycle GPU usage by approximately 13.5%.
Under the frozen G1 ordering (joint SLO feasibility, then minimize physical
GPU-seconds/request), it is not the PrimeLoRA final candidate.  It is retained
as a mechanism trade-off/ablation result and must not be presented as a
headline improvement.

The normal ownership cleanup removed the unpacked NVMe cache after completion;
the authoritative raw evidence remains in the result JSON, physical
allocation journals, request terminals, and the 500 verified remote-transfer
records in `physical_deployment/main_outcome.json`.  No fabricated cache
archive is used.
