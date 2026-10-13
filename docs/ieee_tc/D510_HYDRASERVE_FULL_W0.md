# D510: qualified HydraServe recipe, one full W0 replay

Preregistered 2026-10-13; prepared, not yet launched. This is a development
observation, not an independent holdout or numerical-correctness claim.

## Question and unchanged treatment

Does the qualified HydraServe implementation correctly execute the original
4,000-request W0, and what are its common joint SLO and physical GPU lifetime?
D509 proved affected mixed-LoRA concurrency and bounded resource release on the
first 100 requests. That qualification is not a full performance result.

Reuse D509's original Scheduler, placement, native idle retirement (30 s),
modified vLLM 0.4.2 and D508 necessary cancellation/capacity adaptations. Native
Model.batch=8 determines max_loras=batch*current PP_SIZE. No new policy or
performance parameter is selected. The 16 rendered overlays must match D509.
One physical host, four RTX 3090 GPUs; rootless isolated Podman, no access to the
unrelated Kubernetes cluster. Untriggered pipeline/consolidation is not evidence
of their correctness or performance.

Reuse the byte-identical D507 full HTTP config, original seed42 trace and frozen
500-ID pool. No new weights, workload, arrival resampling or whole-pool copy.
Expected full view SHA: a5331be2e2204483f18206825d9aaa18cbaa259fe0babfc303f989055819430d.
Generation stays fixed_length_greedy_v1; full contract expects 458,224 actual
native output tokens and 132 observed adapter IDs. All first-touch transfers
use the same remote published immutable gzip objects, without request packing.

## Evaluation, limitations and failure handling

Common deployment notice 60 s, request guard 1,800 s; model registration
(13,9000,200,1000000,1), cache=0 and slow_expr=0 unchanged. Native control
thresholds are not the shared evaluation SLO. Frozen metric V1 and single-run V2
remain authoritative. G1 requires all contracts plus at least 3,800 joint SLO
passes; then compare actual lifecycle GPU-s. W0 is not W1/W2/G2 closure.

The D509/D503 same-backend audit found 96 identical and four changed output
sequences, with identical canonical work and native counts; the numerical cause
is not independently proved. Preserve this audit, zero-weight pool limitations,
raw n_correct=null and numerical qualification=false. Report only the explicitly
audited V1 binding/accounting observation, not an invented numerical proof.

Run once for this frozen key. Retain failures, nonfeasibility and actual resource
events; do not increase deadlines, filter requests, repeat for a winner, or use
an incomplete GPU ledger as a precise resource ranking. Success-only audit runs
only after complete workload and actual closure; otherwise use failure-aware
analysis. D507 failure and D509 qualification stay immutable.

## Safety and sequence

Existing 72/80 GiB service high/max, 2 GiB swap, service CPU 4–23/28–47 and
independent 3/4 GiB auxiliary domain apply. Start requires 102 GiB MemAvailable
and disk max(150 GiB,100 GiB+1.5*20 GiB); runtime stop floors unchanged.
One heavy job; no builds, scans, compression or remote maintenance during replay.
Back up preregistration and tests, explicitly await remote start/authenticated
health, read protected-source/resource PASS, close preparation CPU scopes, launch
in tmux, then verify native retirement and exact-owned local/remote closure.

After saving this result, compare all available 7B baselines with PrimeLoRA D476,
then resume evidence-driven PrimeLoRA optimization. All 3B, W1/W2, ablation,
mechanism, sensitivity and final publication work remains in the approved plan.
