# D170 — 7B common warm reference measurement

Status: COMPLETE and independently curated, 2026-10-02 20:13 +08. One run,
19:40–20:06; physical GPU release and exact process absence confirmed. All
1536 measurements and16 warmup calls passed native input/token/timing checks.
This is metrology evidence; G1/G2 and per-model acceptance remain OPEN.

Reuses D168's sealed input index and D169's qualified batch-8 runtime exactly.
No new workload, adapter pool, output contract, serving optimizer or remote cache.
1536 measured calls = two input-length groups × 256 original requests × three
rounds, plus one eight-request warmup batch per group (1552 calls total).
No Prime routing/planning/admission in this GPU-ready native reference.
Compute each round's arithmetic request means, then equal-weight the three
rounds; candidate thresholds are 5× native TTFT and 2× native TPOT means.
Do not claim a frozen cross-model reference before common-batch qualification,
or numerical adapter identity/G1/G2 from native counts alone.

## Output and resource check

D169's retained 38,068,749-byte probe has 10,662,731 compact bytes in batch
observations, with bounded native CPU-cache capacity. Linear 194/4 extrapolation
is 1,846,334,326.5 bytes (a sizing estimate, not a proven memory bound).
Retain original observations; use bounded streaming analysis if needed.
No reduction of memory/watchdog/disk protections and no extra inference probe.
The full service remains 72/80 GiB, swap 2 GiB; analysis 3/4 GiB, swap zero.
Prelaunch disk ~151.6 GiB; running floor remains 100 GiB.

Preparation attempts 1/2 failed before GPU launch: generator len(), then an
incorrect stat.st_size() call. Both scripts/logs retained; attempt 3 passed.
They did not change measured data or serving code. The small D169 probe was
parsed in these checks only to assess output growth, not re-rank its performance.

## Measured reference table

Groups use executed native input-token counts (including the model's special
token), not untruncated source lengths. D168 merged coincident quartile bounds;
the common boundary is616. Each row has256 original requests, batch8.

| Input group | Round | Native TTFT mean (ms) | Native TPOT mean (ms) |
|---|---:|---:|---:|
| ≤616 | 1 | 597.729408 | 29.811045 |
| ≤616 | 2 | 598.496698 | 29.829912 |
| ≤616 | 3 | 597.829028 | 29.828341 |
| >616 | 1 | 926.948554 | 42.703325 |
| >616 | 2 | 927.990718 | 42.585696 |
| >616 | 3 | 926.255827 | 42.728890 |

Per V1, average the three round means equally, then multiply by5/2:

| Input group | Warm TTFT (ms) | Warm TPOT (ms) | Candidate TTFT SLO (ms) | Candidate TPOT SLO (ms) |
|---|---:|---:|---:|---:|
| ≤616 | 598.018378 | 29.823099 | 2990.091890 | 59.646198 |
| >616 | 927.065033 | 42.672637 | 4635.325166 | 85.345274 |

Exact values, all1552 sample rows, all194 batch rows and provenance are in
`paper_results/ieee_tc/p2_backend/20261002_d170_warm_*`.
Tables are the appropriate artifact: this experiment establishes reference
values, not a system ranking or an advantage figure. Same-process rounds are
not independent Full-run replicates; no t-CI or superiority claim is attached.

## Validation and resource evidence

- Exact selected source indices, request/adapter/target map, canonical prompt
  and native prompt IDs matched the sealed index in every round.
- Native actual output counts equal targets; first/last/dispatch clocks and
  recomputed TTFT/TPOT agree within the unchanged1ms metrology bound.
- Every batch starts GPU-ready, with no prior admitted/unretired work and
  sufficient observed full-context KV blocks. All references released and
  scheduler work drained; final native cache/slots empty.
- One physical lease,1429.3385943299509 GPU-s, released. This is calibration
  resource use, not a Full resource result or a Resident budget reference.
- 1499 resource samples: observed peak12596813824B (~11.73GiB), minimum host
  available99741794304B, all memory events/swap/warnings/aborts zero.
- Service/watchdog exit0; exact cleanup verified20:06:54. No remote operations.
- Raw JSON1763970424B retained unchanged. Reused streaming projection emits
  29031137B, after two structural fixtures. It excludes only repeated batch
  tensor inventories while retaining required ownership/slot/cache facts;
  original evidence remains available. CPU analysis limit3/4GiB/swap0 retained.
- Post-write disk160967503872B is slightly below the150GiB new-heavy gate;
  sampled disk minima before final serialization are not the post-write value.
  No next heavy run until audited space recovery; no threshold relaxation.
- Protected147 historical entries unchanged. No serving source/config changes.

## Interpretation and next action

1. Three round means are close; the longer-input group has higher native warm
   TTFT and TPOT. This supports input-conditioned reference values, not a causal
   claim about Prime's routing or hierarchy.
2. These7B candidates can support explicitly provisional offline diagnostics.
   The cross-model common-batch contract is not yet frozen. Per-token numerical
   adapter execution identity/full-pool qualification remain open; native counts
   and reference names alone do not close them.
3. After evidence backup, reuse the latest retained7B Full request projection to
   quantify the conditional timing gap under these candidates. Keep unknown
   correctness unknown and do not label provisional diagnostics formal G1/G2.
4. Complete the necessary matched old/new Prime and resource-reference evidence,
   then continue evidence-based7B optimization. Do not restart external baselines
   or advance3B optimization on the strength of this calibration.

The prior Full remains D164; this isolated reference is not a new Full result.
