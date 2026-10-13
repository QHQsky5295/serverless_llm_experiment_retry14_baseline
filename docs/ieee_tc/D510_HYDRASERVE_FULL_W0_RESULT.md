# D510: full HydraServe W0 result and seven-system comparison

Completed 2026-10-14, one preregistered run, no parameter search or repeated-run
selection. Configuration and limitations: [preregistration](D510_HYDRASERVE_FULL_W0.md).

## Actual result

4,000/4,000 offered, arrived and terminal; 4,000 native contract responses,
zero failures, 458,224 actual output tokens and 132 observed adapter IDs.
All source/prompt/adapter/index/path/content/native-count/clock joins passed.
132 real remote transfers, 131,480,060 wire bytes; no request-path packing.
The two existing nonzero witnesses retained the D502 output sequences.
Independent numerical qualification remains open; raw n_correct stays null.

| Quantity | Observed value |
|---|---:|
| Joint common TTFT/TPOT SLO | 2,749/4,000 = 68.725% |
| TTFT marginal attainment | 89.775% |
| TPOT conditional attainment | 74.675% (4,000 requests) |
| Mean / Type-1 P95 TTFT | 1,505.569 / 8,851.575 ms |
| Mean / Type-1 P95 TPOT | 69.884 / 108.214 ms |
| Mean E2E | 8,863.097 ms |
| Actual physical GPU union | 9,662.152610210469 GPU-s |
| GPU-s / V1-bound request | 2.4155381525526174 |

The point is outside the frozen 95% feasible set. Low GPU time alone is not a
G1 win. This is a known-workload development observation, not an independent
holdout, a baseline optimum, a repeated-run claim, or W1/W2/G2 completion.

Mean phases are submit lag 3.944966 ms, submit-to-instance assignment 883.935365
ms, assignment-to-first-token 617.688312 ms, decode 7,310.705142 ms and completion
notification 46.823236 ms. Adapter resolution averages 27.848080 ms but is nested,
not additive. These measurements identify queue/decode contributions; they do
not prove every regression is inherent to the published system or establish a
single causal explanation. No timeout or common SLO was relaxed.

## Actual lifecycle and mechanism boundary

34 acquires, 34 worker spawns, 34 native GPU context confirmations and 34 actual
releases; every native launch requests PP_SIZE=DEST_PP_SIZE=1 and BATCH_SIZE=8.
The native controller creates/retires actual replicas on one four-GPU host.
Full pipeline/KV migration/consolidation is not demonstrated by this pp1 run.
No claim of physical multi-node scale-out or newer accelerator performance.

Mutually exclusive GPU-s windows: pre-arrival 0, arrival 9,450.354225911666,
drain 20.74031947203912, cleanup 191.05806482676417. Their sum equals the actual
device union. Peak service memory 51,459,432,448 bytes; high/max/OOM/OOMkill and
service swap all zero. Original native idle retirement completed, then exact
local service/auxiliary closure and owned remote-service stop passed. tmux exit0.
No concurrent build, compression or remote maintenance occurred during replay.

## Same-work seven-system W0 observation

All 4,000 requests matched source, adapter content, canonical prompt, native
prompt IDs/count, target, generation contract and relative arrival (1 us clock
subtraction allowance only). Contract SHA:
384b63425402ced97961e91081265d3cbe340b6a3b4febde242de28f62cd7399.
Common resource policy, SLO and Resident reference remain unchanged. PrimeLoRA
uses D476, not the older D371 row retained in the historical comparison.

| System | Joint SLO | GPU-s / bound request | P95 TTFT (ms) | W0 feasible |
|---|---:|---:|---:|---|
| PrimeLoRA | 97.425% | 2.694612 | 2,315.363 | yes |
| ServerlessLLM | 98.125% | 4.022441 | 1,052.246 | yes |
| vLLM | 99.475% | 4.028844 | 1,200.507 | yes |
| S-LoRA | 94.425% | 3.789269 | 17,699.067 | no |
| Loquetier | 4.350% | 4.249596 | 555,110.032 | no |
| ElasticLocality | 95.775% | 2.473179 | 2,420.608 | yes |
| HydraServe | 68.725% | 2.415538 | 8,851.575 | no |

Among observed feasible points, ElasticLocality uses less GPU time than
PrimeLoRA. PrimeLoRA uses 8.9534% more resource than ElasticLocality, while its
joint SLO is 1.65 percentage points higher and its P95 TTFT is 4.3479% lower.
Thus PrimeLoRA has not met the requested best-resource objective; optimization
must resume from this explicit gap, without discarding stronger comparisons.
PrimeLoRA uses about 33% less GPU time than ServerlessLLM/vLLM but has worse
raw tail TTFT. Do not turn that trade-off into an all-metric superiority claim.

## Source recovery and retained analysis failures

The completed comparison checked 301 sources. An exhaustive inventory of 217
historical references found exactly four representation/version differences:
two later-updated tracked files and two losslessly compressed logs. Exact old
Git bytes and decompressed original SHA values were checked, never substituted
by current content. The final flat executed source and receipts are saved.
Earlier analysis failures (historical-path assumptions, full-vs-summary receipt,
wrong CPU environment and source-wrapper preparation) are retained separately.
None changed an experiment or required a GPU replay. Use compare_final.py in the
existing PrimeLoRA streaming-analysis environment, not the failed entrypoints.

## Next main-line task

Seal and back up this result and the complete available comparison. Then return
to evidence-driven PrimeLoRA optimization against the feasible ElasticLocality
resource gap; retain all IEEE equations and common metrics. 3B, W1/W2/G2,
ablation, mechanism, sensitivity and later publication work remain outstanding.
The 500-ID pool's limited weight diversity and the D509 96-equal/4-changed
same-backend audit with unproved numerical cause remain explicit limitations.
