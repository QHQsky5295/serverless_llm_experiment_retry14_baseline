# D363 — first100 validation before unchanged-candidate Full W0

2026-10-09. Mainline: 7B optimization, then 3B, then paused baselines and the
approved main comparisons/ablations/sensitivities. No figures or paper edits.

## Question and retained evidence

D360 recovered a rise in stale owner-epoch rechecks from 593 to 1,609 between
D353 and D358, while final native load time was nearly unchanged. D361 introduced
one candidate: a held immutable-file witness with validation/load/acquisition in
the native owner's serialized turn. All live ownership, capacity and pin checks
remain. GPU/native-HOST paths retain their existing exact-copy guards.

This follows the relevant implementation boundary in the pinned
[official vLLM 0.30.0 worker](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py):
the single-threaded core handles adapter validation, cache-capacity enforcement
and activation. Rechecked 2026-10-09. That source motivates the ownership boundary,
not a performance claim or the assumption that every model step changes an epoch.

D362 has now completed 1,669 native calls, covering 200 service classes and 24
preparation classes. Its 969 file transactions include 191 with an intervening
owner-epoch advance. Fresh strict-profile loading passed; no old timings are
relabeled. Measurement SHA:
`7f53d261ed66ae4c5386237da3b4c425197dda3c18c4764e7dbfbcc3d15a457e`.

## Registered execution

Reuse the existing `run_d293_7b_prefetch_validation_100.sh` entry point, qualified
native vLLM environment, published remote cache, original 500 logical adapter
pool and W0 trace. Replay the first 100 requests by the existing prefix switch;
do not generate a new trace. The config retains total_requests=4000 as the base
trace identity, while the diagnostic offered set is exactly the first 100.

Relative to D358, change only output/cache paths, the D361 file-demand policy,
its identity-bound completed-length initializer IDs and D362 measured profile
bindings. Keep the same C14 runtime, token budget 1024, control interleave,
autoscaler, HOST budgets, generation, preparation window and timeout. Strictly
assemble the child runtime model and require equality with D362's contract.

Before launch: tests, exact profile SHA/loader checks, protected-result checks,
bounded resources, idle GPUs, frozen remote configuration, actual link speed and
pushed source identity. Use one heavy job, separate replay/watchdog and owned
cleanup. Do not alter remote services or hash/compress during the measurement.

After native release: stop only this run's owned remote units; collect matched
journals; validate 100 terminal requests, native token counts, adapter/content
and file-witness receipts, dispatch tiers, timing identities and resource
accounting. Preserve failed evidence without inventing missing values.

## Decision and next experiment

This prefix qualifies execution and measurement, not the performance target.
Report any untriggered path rather than claiming a mechanism benefit. If valid,
proceed directly to complete 4,000-request W0 with this unchanged candidate and
compare D358, D353 and D336. Inspect all-request/phase-2 TTFT tails, TPOT, stale
rechecks, admission-to-acquisition waits and physical GPU-s. Do not extrapolate
prefix GPU-s or retune a second mechanism before the full replay.

D170 timing bounds remain provisional. Numerical `N_correct`, the final common
warm reference and complete Resident budget reference remain open. Successful
execution is not formal G1/G2 qualification, and single development runs do not
produce an independent-repeat confidence interval.

Plan SHA: `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`.
Metric V1 SHA: `5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`.
