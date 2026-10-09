# D367: complete W0 after the immutable file-identity repair

2026-10-09. This is the predeclared complete validation after D365/D366, not
another tuning round. Reuse the existing runner, D362 measured initialization,
7B artifact pool, original 4,000-request W0 and published remote objects.

## Final evaluation criteria and evidence required

The final metric definitions are frozen, not tentative. Under V1:
- G1 requires all 4,000 requests correct and joint TTFT/TPOT attainment >=95%;
  then compare lifecycle physical GPU-s/request, lower is better.
- G2 requires all correct, the common resource envelope, joint attainment >=95%
  and U <= 0.75 U_ref; compare W1/W2 P95 TTFT and separately test zero-margin
  joint-SLO noninferiority against the comparators.
- Raw mean/P95 TPOT need not independently improve. A TPOT regression is retained
  and checked against the joint constraint; by itself it does not reject G1/G2.
- CE, cost and raw mean latency remain auxiliary, not alternative acceptance rules.

Missing factual evidence is not a change to these final criteria. D170 provides
measured timing samples but does not close the common numerical/reference gates.
Keep its historical identity; do not silently certify it or invent N_correct or
Resident budgets. Complete W0 is necessary for this candidate decision but is not
the entire W0/W1/W2 qualification or formal repeated baseline comparison.

Protocol SHA: 5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22.
Plan SHA: 0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7.

## Prior evidence and fixed comparison

D366 completed 100/100 native fixed-output returns, no failures. Relative to
D363 its mean/P95 TTFT fell 23.63%/36.99% and physical GPU-s fell 58.21%;
mean/P95 TPOT rose 11.38%/29.43%. D170 timing-only checks counted 99/100 joint
passes and zero TPOT misses. This is one prefix, not a CI or extrapolation.
The reason to continue is missing complete-run evidence, not TPOT alone.

D364's 3,999/4,000 successful full run remains a failure. D365 deterministically
reproduced and repaired immutable-publication comparison with bounded changing
physical allocation; no retry/fallback or weakened content/capacity protection.
The specific historical filesystem event is unproven. Preserve that distinction.

Compare complete D367 with retained D358/D353/D336 and retain D364's failure.
Report native correctness/binding and remaining numerical qualification, full
offered population, lifecycle U, TTFT/TPOT/E2E, joint timing checks and missing
final reference evidence, phase waiting, source tiers and file-demand triggers.
Do not infer causality from one natural closed-loop replay or discard regressions.

## Execution contract

D367 YAML differs from D366 only in output/NVMe/HOST paths; disable the prefix
through the existing explicit full-run case. No serving changes, profile refresh,
concurrency/capacity tuning, new weights/trace, thresholds or IEEE-equation edits.
The existing D362 strict child binding must still cover 200 service/24 preparation
classes. Reuse D365's two-environment serving tests; run only existing launch,
timing/analysis and OS-guard fixtures required for this new execution identity.

Use system Python for OS/scope guards and native model Python for imported health
and binding checks. This explicitly avoids the retained D366 missing-numpy setup
failure without adding packages. Check whitespace before sealing new records.

One heavy job, guarded open-loop replay and watchdog; host >=102 GiB and disk
>=150 GiB to start, running disk floor100 GiB, predicted growth32 GiB. Unchanged
service72/80 GiB/swap2 and auxiliary3/4 GiB/swap0 with separated CPU sets.
Verify D366 closure/backup, immutable inputs and 147 protected artifacts, with
only the documented baseline-script exception. Never stage user manifest or
unrelated untracked work.

Back up source before launch. Start unchanged authenticated published-gzip
services with new monitor path; record actual negotiated NIC speed. No request
packing, remote reconfiguration, tests, compression or hash scans during replay.
After actual resource release, stop owned remote units once, collect journals,
audit, retain failures, seal and back up. Never repeat completed old cleanups.

Continue 7B -> 3B -> paused baselines -> M1/M2 -> ablations/sensitivities.
No figures yet. D365 records the relevant Linux/vLLM primary-source checks;
this run introduces no further serving optimization.
