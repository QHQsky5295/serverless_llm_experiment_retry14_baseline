# D993 — retained-metric audit preparation (2026-10-31)

## Status and purpose

Completion update: D992 and all four retained-data analyses have finished.
Authoritative outcomes and next action are in
`D993_RETAINED_CONTROL_OUTCOME_20261031.md`. The following preparation text
records the earlier in-progress state; do not restart the completed run.

Analysis preparation only, while the registered D992 3B W2 replay continues.
No production control code, current configuration, threshold, artifact service,
or existing Kubernetes workload was changed. No new GPU experiment or figure.

The next comparison must use the approved common joint TTFT/TPOT criterion,
Type-1 quantiles and physical GPU ownership union. D989's 99.8% value is only
the legacy 5-second TTFT indicator; it does not establish joint feasibility.
D972's narrative lifecycle value 12,194.16 is not its physical owner-union
value (the retained physical summary records 11,310.910081950948 GPU-s).
These reporting issues require retained-data analysis, not new replay.

## Existing analyzer extension and checks

Reuse `scripts/ieee_tc/analyze_d455_stages.py:analyze_retained_controls`:

- Add explicit model selection for the already-frozen D394 7B or D648 3B
  thresholds and the corresponding existing content index.
- Select the input-length group from the frozen manifest (616 or 506 boundary,
  respectively), without copying the 7B boundary into the 3B evaluation.
- Accept explicit launch receipts in source order. D989 must use
  `launch_retry5.json`, not a failed default attempt or a newest-file heuristic.
- Preserve native bindings, original clocks, 4,000-request coverage and physical
  interval/window checks; add control-action reason counts to the existing report.
- Preserve `n_correct=null`, no independent holdout and no G1/G2 qualification.

The historical `null` is preserved as a source field, not interpreted as a new
requirement to compare all 4,000 sequences exactly against PEFT. D393 and D637
already rejected that stronger interpretation of V1. Cross-backend exact
sequence equality and the V1 request/adapter/native-accounting contract are
different questions. The bounded D981/D991 witnesses remain diagnostics;
their mismatches do not alone invalidate a performance run or authorize a new
full reference replay. A later derived V1 correctness count must explicitly
name its evidence and must not overwrite the original field. Formal identity,
cross-scenario coverage and numerical-arithmetic proof remain separate.

Seven focused `RetainedControlAnalysisInputs` tests passed in the existing
vLLM 0.30 environment with CUDA devices hidden, auxiliary CPU affinity and
3/4-GiB memory limits. The first invocation used the old PEFT environment and
failed at import because it lacks `ijson`; no test body passed in that attempt.
No package installation or serving change was made. `git diff --check` passed.
Actual retained-result analysis and checkpoint push wait until D992 releases
its resources, to avoid heavy parsing/hashing or backup during measurement.

## Exact next work

1. Finish the same D992 run; keep safety/failed status if it terminates early.
2. Reuse the extended analyzer on D972/D992 with `model='3b'` and on the
   explicitly retained D983/D988/D989 W0 runs with `model='7b'`.
3. Store new summaries under `paper_results/ieee_tc/p2_backend/d993_*`; no old
   raw JSON or prior curated result is overwritten. Record source/protocol SHA.
4. Correct the D972/D989 narrative using the reconstructed joint timing and
   physical ownership values; do not label the 3B W2 run W0.
5. Evaluate the one current controller candidate, then return to outstanding
   same-config W0/W1/W2, comparisons, A1–A5 and S1–S13. No duplicate witness or
   repeat completed key merely to seek a better result.

## HydraServe / shared-cluster clarification

The user confirmed the existing k3s belongs to another project. Reuse the
already completed rootless Podman adaptation and retained HydraServe D510,
D546 and D574 evidence. Do not obtain its kubeconfig, restart k3s, modify its
workloads/network, or repeat basic deployment. Existing PP=1 observations do
not prove multi-node pipeline/consolidation performance; retain that limitation.

The latest checkpoint remains `87d3003f`, already pushed before D992 started.
This analysis preparation is not yet committed; preserve user edits to the
generated manifest and EXECUTION_STATUS, and never stage them wholesale.

Protocol interpretation references: D393_7B_CORRECTNESS_AND_REMOTE_METRIC_ALIGNMENT.md
and D637_REPAIR_REUSE_AND_CORRECTNESS_SCOPE.md. Do not revive the superseded
D390/D391 exact-reference hold after context compaction.
