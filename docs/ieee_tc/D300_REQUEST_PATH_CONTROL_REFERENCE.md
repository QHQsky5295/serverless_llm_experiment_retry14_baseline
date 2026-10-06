# D300 — isolate the scale-in latency reference boundary

2026-10-07. Development candidate, **not** a change to the common SLO, IEEE
controller equations, or a claim of lower resource use. 7B remains the main line.
No new reference measurement, weights, trace, figures, 3B or baseline campaign.

## Evidence and one falsifiable hypothesis

D299 completed 4000 requests with 15845.533076 physical GPU-s, three scale-outs
and no scale-in. All 1335 low-queue/low-saturation observations failed the latency
predicate. Six low-queue/low-saturation/no-pending spans exceed the unchanged
38-second cooldown. D297 did not have such spans; the new capacity package is
why revisiting this hypothesis now is justified, not a reason to rerun D297.

D88 deliberately selected a development lower threshold from D84's protected
GPU-ready acquisition-to-first-token P95 (281.278192 ms). The actual controller
observes arrival-to-first-token, including routing, admission and preparation.
This is not an implementation violation of the paper's configurable limits;
it is a potentially unsuitable operating point, which D88 already warned about.

**Hypothesis:** replacing only that short-path lower reference with an already
measured Full request-path reference permits occasional safe reclamation and
reduces physical occupancy without losing the current timing quality. Actual
closed-loop outcomes may refute this: delayed reactivation can worsen TTFT,
decode contention can worsen TPOT, and repeated startup can erase GPU savings.

## One predeclared configuration, not a threshold sweep

Set `ttft_lower_ms = Q95_Type1(overall_ttft_ms of all 100 D298 Full requests)`
= **2339.924886357039 ms**. Use the already completed D298 calibration/smoke
run, not a percentile chosen from D299's low-pressure periods. Retain all 100
requests, including preparation and waiting; no tier, output, or latency filter.

Source: `d298_20261007/7b_capacity_validation_100/experiment_results_full_vllm_dedicated_a500_r100_c2_tc_ieee_prefetch_validation_100_d298_7b_capacity_validation_100.json`,
SHA `52666d436be444924980c26d86a01dd3e1e244bb823b17bbba1f5a0a01f5ed24`.
Existing audit SHA `42b6b895b9e255115d72160787ce2bbc760decabf645878a62359fea5e0d90bc`.
Recompute the percentile from native-audited requests before accepting the config.

This is an empirical development reference, not a guaranteed low-load bound or
an optimality derivation. D298 has only 100 requests, uses the old controller,
and belongs to the same already-seen seed42 source trace; no held-out claim.
The rule takes a measured end-to-end operating envelope as the scale-in guard
and relies additionally on empty queue, low active fraction, no pending
activation and sustained cooldown. It introduces no coefficient or new online
predictor. The selected number is frozen before the new run; it will not adapt
to that run's measured P95 or be raised again merely to produce scale-in.

Everything else remains D299: queue upper/lower 2/1, active upper/lower .5/.25,
TTFT upper5000ms, window45s, cooldown38s, cap8/slots8/rank16/.92memory,
D298 service/preparation initialization, EWMA, routing, handoff and admission.
The C2-derived soft concurrency targets are a separate candidate, intentionally
not changed here. Physical capacity and soft targets need not be equal.

The controller still consumes causal completed overall TTFT; unknown is not
zero, all three lower conditions must hold, and only owned drained replicas
may be reclaimed. No force-scale command, artificial idle gap, or free cleanup.
Frozen V1 evaluation thresholds/95%/75% and lifecycle counting remain unchanged.

## Primary-source and implementation cross-check

- IEEE supplied source §3.1 explicitly permits configured upper/lower limits,
  max-normalized-score scale-out and sustained all-low scale-in. Local
  `IEEEReplicaControl` and `_evaluate_ieee_live_control` implement this;
  controller code was last introduced by `faf5c85`, not edited for this test.
- [Ray Serve advanced autoscaling](https://docs.ray.io/en/latest/serve/advanced-guides/advanced-autoscaling.html)
  distinguishes the service target from hard capacity, calls for load testing
  against end-to-end latency, and uses a separate downscale delay. This supports
  calibrating a service-path target, not copying Ray defaults or promising
  equivalence to its policy.
- [Ray 2.49.0 policy source](https://github.com/ray-project/ray/blob/ray-2.49.0/python/ray/serve/autoscaling_policy.py)
  computes request pressure against target×replicas and counts consecutive
  scale-down periods. We retain IEEE's max-score/all-low rule, not Ray's formula.
- [Knative concurrency](https://knative.dev/docs/serving/autoscaling/concurrency/)
  explicitly separates soft targets and hard limits. Capacity8 therefore does
  not by itself prove that old queue/active targets are erroneous.

Two guessed Ray `_private/autoscaling_policy.py` URLs returned404 during
read-only research; the public path above was fetched successfully. No serving
dependency or environment was changed.

## Validation and interpretation

1. Reuse existing native audit to verify D298 and the percentile; assert the
   candidate has exactly one non-identity config difference from D299.
2. Existing controller/actual-lifecycle tests: strict boundaries, missing
   samples, pending reset, minimum replicas, drained retirement and cleanup.
3. Inspect the single predeclared predicate on D299's saved observations only
   as a feasibility diagnostic. Do not convert it to predicted GPU-s or
   counterfactual latency. If no eligible span exists, record that limitation;
   do not sweep thresholds to select a winning historical trajectory.
4. Since no runtime/loader/contract changes, the above are the minimal checks;
   do not repeat source248 or warm1536. Run one complete4000 W0 development
   replay with the existing protected launcher after backup and resource gates.
5. Compare D299/D300 full workload, all failures, physical GPU-s, mean/P95 TTFT,
   TPOT, unchanged provisional joint timing, scale-up/down and actual release.
   Lower GPU-s alone is not success. Numerical correctness and final shared
   references remain open, so no formal G1/G2 acceptance or CI from one run.

Only one candidate is active. After completion, save data and decide from the
resource/latency tradeoff; do not leave this running as an unbounded local sweep.

## Prelaunch checks completed

Existing native audit recomputed all100 requests and the exact reference P95.
Config SHA `25debb773cc004a134e026b148c23d9af2565a8a7041db98e74f438efa65a378`;
only the declared lower control reference differs after excluding run-owned
paths. All56 control/lifecycle/native-timing tests passed (31.139s); shell syntax
and diff checks passed. Config audit peak30,662,656bytes, high/max/OOM0.

The fixed candidate predicate is low in420 saved D299 observations, with one
continuous sampled span43.252538s exceeding38s. This is an opportunity check,
not predicted scale-in, performance, or resource savings.

Five completed historical main_outcome logs were losslessly compressed; two
source/archive round trips verified before removing only raw duplicates.
Original request files, curated data and protected submission files unchanged.
Receipt: `paper_results/ieee_tc/storage/20261007_completed_outcomes_lossless_compression.json`.
Available162918813696bytes afterwards, above the unchanged150GiB start floor.
Protected seal remains146/147 unchanged, the sole known baseline script edit
unchanged; no blanket protected-pass claim. Test failure-path log warnings
are expected unit fixtures, not an inference failure or GPU attempt.
