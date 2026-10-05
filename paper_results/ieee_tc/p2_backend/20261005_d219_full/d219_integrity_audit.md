# D219 integrity and metric audit

## Verdict

`WARN — descriptive exploratory evidence only.` The run is complete and the
curated values agree with the source result, but it is not eligible for a formal
G1/G2 or cross-system claim.

## Evidence checks

- **Result existence and agreement — PASS.** The curated JSON, status table and
  figure manifest point to the same D219 source run and retain source hashes.
- **Ground-truth/normalization fraud — no issue observed.** The target output
  count is derived from the frozen trace/configuration contract, and CE uses the
  declared latency/cost product. No self-normalizing denominator was found.
- **Completion scope — PASS for execution, not for correctness.** The run has
  4,000/4,000 terminal records, native output-contract matches, and closed
  physical allocation journals. `n_correct` remains `null`; a native token
  count is not numerical adapter correctness.
- **Population denominators — explicit but incomplete for quality.** The
  curated record exports `N_plan=N_arrived=N_submitted=N_terminal=4,000` and
  explicitly records `N_correct=null` and `N_good=null`; the latter is unknown
  because D219 has no frozen joint TTFT/TPOT SLO.
- **Percentiles — corrected offline.** The producer uses linear interpolation,
  while the frozen protocol requires Type-1 (x_{(\lceil pm\rceil)}). The
  curated `protocol_type1_metrics` section recomputes P50/P95/P99 from all
  successful request records. Producer percentiles remain retained as raw source
  fields, not formal values.
- **SLO — not qualified.** The producer's 98.275% field is TTFT-only at the
  5,000-ms legacy threshold. D219 has no frozen common warm TTFT/TPOT threshold,
  so joint SLO attainment is explicitly `null`.
- **Timing coverage — PASS for this diagnostic.** All requests contain planned
  arrival, client-submit, native dispatch, first-token, last-token and worker
  completion timestamps. The audit checks timestamp ordering, timestamp
  algebra, stored TTFT/E2E/service fields against their clock formulas, and
  native TPOT; each has zero maximum absolute error over 4,000 rows. This
  validates timing-field consistency, but does not establish numerical adapter
  correctness or a joint SLO.
- **Remote journal reconciliation — PASS for replay scope, WARN for all events.**
  All 132 replay-linked transfers pass artifact, SHA, byte, delivery-mode,
  no-pack, outcome, timing-contract and span-order checks. The journal also
  contains one earlier preload/initialization event that is not request-bound
  by the main outcome; it is retained and classified in
  `d219_remote_unmatched.csv`, but excluded from the replay denominator.
- **Mechanism scope — conditional WARN.** The confirmed-snapshot route is
  active and its request-level owner/epoch/tier invariants pass. Metadata also
  records `effective_capacity_admission_enabled=false`; GPU admission,
  hierarchy transition and background planning counters are zero. Therefore the
  run qualifies the route/full-replay path only, not physical-capacity admission
  or every mechanism implied by the paper's Full label.

## Claim boundary

D219 supports a complete single-seed route/full-replay diagnostic and a measured
control-path reduction relative to D210. It does not support a statistically
significant PrimeLoRA improvement, a numerical-correctness claim, a joint-SLO
claim, a G1/G2 resource ranking, or a baseline comparison.

## Required next evidence

1. Complete the token-source/numerical adapter-correctness audit; the
   replay-linked remote transfer audit is complete for its declared scope.
2. Complete the common warm/Resident reference under the frozen remote,
   lifecycle, resource-domain and timeout contract.
3. Use held-out paired blocks before attributing the P95 TTFT trade-off or
   selecting the next PrimeLoRA optimization.
