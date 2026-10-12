# D465 — complete HOST queue candidate, common SLO not met

2026-10-12. One full original 4,000-request 7B W0 replay, execution commit
`601bcd39da9bf61c496494dcdccd6e407ea669a4`. This is a retained development
observation, not an independent holdout or a new canonical performance winner.
The production delta is D463's one initial owner/clock observation; configuration
equals D458 except output/cache paths. Nine IEEE formulas and frozen metrics
V1 + single-run V2 are unchanged. No baseline rerun or performance tuning.

## Complete matched observation

| Observation | PrimeLoRA D458 | PrimeLoRA D465 | ElasticLocality D453 |
|---|---:|---:|---:|
| Native/source/adapter-bound requests | 4,000 | 4,000 | 4,000 |
| Joint TTFT/TPOT attainment | 96.45% | 94.75% | 95.775% |
| Physical GPU-s / bound request | 2.994085 | 2.996037 | 2.473179 |
| P95 TTFT, ms | 2,847.386 | 3,519.901 | 2,420.608 |

D465 has 3,790 joint-good requests, **10 below** the fixed 3,800 threshold.
TTFT passes 3,827; TPOT passes 3,937; TTFT-only failures 147, TPOT-only 37,
both 26. All requests complete the native fixed-output contract; failures of
SLO are not execution failures. Total physical occupancy is 11,984.148429 GPU-s.
The four disjoint windows sum to that total: pre-arrival 36.392555,
arrival 11,769.849737, drain 11.849582, cleanup 166.056555 GPU-s.

Against D458, GPU occupancy increases 0.0652%, joint SLO decreases 1.70
percentage points, and P95 TTFT increases 23.6187%. Mean TTFT rises from
1,145.752 to 1,397.648 ms. Its additive decomposition changes from
513.687 to 753.010 ms before native dispatch, and 632.065 to 644.638 ms
in service TTFT. This locates the measured mean increase, not its cause.
Mean TPOT is 33.982953 ms and mean E2E 5,204.356846 ms.

The W0 budget ratio is 0.991529, but budget compliance alone is insufficient:
joint SLO fails and required W1/W2 evidence remains open. Do not call this G2.
The descriptive resource gap to ElasticLocality is +21.1411%; a constrained
G1 winning claim is ineligible for this candidate.

## Mechanism, input, delivery and safety evidence

- The independent S-LoRA D435 input witness matches all 4,000 request IDs,
  adapters, targets, prompt hashes, native input-token hashes and source items.
  Common contract SHA `384b63425402ced97961e91081265d3cbe340b6a3b4febde242de28f62cd7399`.
- 458,224 native output tokens, 132 encountered logical adapters. Original pool
  diversity limitations remain: 500 logical IDs are not 500 independent models.
- 1,129 HOST queue records: 1,006 residency and 123 handoff. Initial activation
  becomes ready; 21 natural scale-out triggers yield 18 ready and 3 not ready.
  Counts do not establish per-call savings or causal latency contributions.
- 132 transfers and 131,480,060 wire bytes agree at both endpoints; all 132 are
  also joined to request preparation records. Request-time packing is zero.
  Existing D78/D80 published gzip was reused; actual link was 1 Gbit/s/full.
- TTFT/TPOT/E2E native reconstruction errors are zero. Physical journals close,
  actual GPU release is confirmed, and exact-owned local scopes and remote
  services are stopped before post-run analysis.
- 4,075 watchdog resource samples: peak service memory 22,273,388,544 bytes;
  high/max/OOM/oom_kill/swap all zero; minimum host available memory
  91,419,357,184 bytes. This does not prove absence of all environmental noise.
  Two systemd inotify warnings are retained; no zero-impact or causal claim.
- Same-backend output digests differ in 202 requests versus D458, 223 versus
  D371, and 193 versus ElasticLocality. All differences and bindings are retained;
  no unsupported rounding/batching explanation or independent kernel numerical
  correctness claim. Raw `n_correct=null` remains unchanged.

## Decision and next mainline

**Do not accept D463/D465 as a demonstrated performance improvement.** Preserve
this complete nonfeasible observation; no retry of the same key, threshold
relaxation, winner filtering or silent replacement by D458. D458 remains the
previous feasible development checkpoint, with its own resource gap unresolved.
The small redundant-read hypothesis has reached its full-run decision point;
do not spend another microbenchmark/prefix/full replay on it without new evidence.

Next examine the retained matched request/activation/queue timeline to identify
a falsifiable cause of delayed dispatch and excess within-window GPU occupancy.
Reconcile D459/D460/D461 and previous rejected optimizations before another
production change. Whether retaining or reverting the semantically valid D463
read, declare the next version explicitly; do not silently mix code and results.
Use current primary-source implementation evidence for any new optimization.
W1/W2/G2, 3B, HydraServe independent qualification, dLoRA3B, A1–A5, S1–S13,
cost analysis, review documents and later publication figures remain required.

## Provenance and storage

Raw campaign: `results/ieee_tc/p2_backend_qualification/d465_20261012/`.
Curated: `paper_results/ieee_tc/p2_backend/d465_full_w0/`.
Result SHA `cb83dc960641a164be351680c8d69a54873491186eb1825b48afc0407393dd91`.
Existing analyzers and binding/physical-integral primitives were reused. The
large mechanism log is streamed under the 3/4-GiB CPU-only envelope. No figure
or run-level CI was generated. Protected 147 historical entries are unchanged.
The inherited raw completion `kind` has a cosmetic `host_queue000` typo; its
4,000 counts/source identities are correct and original bytes are preserved.
Storage receipt maps the closed main_outcome to lossless gzip after two
decompression SHA checks; this is not loss or alteration of original evidence.
Final verification checked 94 source references and protected147. Its first
helper used a nonexistent plural warning-field name; the corrected helper reads
the actual singular boolean `warning`. The failed helper/log/closure remain,
and all 4,075 samples report false. No measurement, criterion or run changed.
