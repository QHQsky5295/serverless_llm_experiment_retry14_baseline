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

## Launch receipt

Checkpoint `26ec812ba5b15e8c4f5b18d1e99a58445b7c39b2` was pushed and verified
before launch at05:05:32+08. Existing protected runner in tmux
`primelora-d300-7b` has original4000W0, no diagnostic truncation. Authenticated
direct remote health passed before deployment; same1GbE/published-cache protocol.
Actual owner identities and current status are in EXECUTION_STATUS. At sample33,
service peak5,046,140,928bytes, hostavailable107,976,048,640bytes and no memory
events/warnings. This is a launch observation, not completion or an outcome.
Do not launch another D300 or modify this candidate while it runs.

## Completed full-run evidence — 2026-10-07

All4000 native-contract requests completed, with no reported failure. Six
physical leases on four distinct GPUs closed, including two natural retirements
and two reacquisitions. Native GPU census is clear; service/replay/watchdog
returned0 and the service scope was removed. Exact-owned remote services were
stopped after local release. No performance rerun or intervening configuration
change occurred.

| Metric | D299 full W0 | D300 full W0 |
|---|---:|---:|
| Mean TTFT (ms) | 1306.468257 | 1390.790816 |
| P95 TTFT (ms) | 2884.701929 | 3064.463936 |
| Mean TPOT (ms) | 32.918675 | 33.284821 |
| P95 TPOT (ms) | 62.482613 | 61.798980 |
| Mean E2E (s) | 4.904404 | 5.055340 |
| Physical GPU-s | 15845.533076 | 15569.192088 |
| Provisional D170 joint timing upper bound | 3828/4000 (95.70%) | 3806/4000 (95.15%) |
| Natural scale-up / scale-down actions | 3 / 0 | 5 / 2 |

Each column is one complete, already-seen W0 development run. No CI or
statistical superiority claim. GPU occupancy is1.743968%lower; mean/P95TTFT
are6.454237%/6.231563%higher, meanE2E3.077567%higher, while P95TPOT is1.094118%
lower. The joint timing upper bound falls0.55percentagepoints. These are a
resource/latency tradeoff, not an across-metric improvement. Both provisional
timing bounds exceed95%, but neither establishes formal G1/G2: numerical
qualification, globally frozen common references and Resident budget remain
open. Per-correct resource remainsnull. A small resource reduction is not
rejected merely because TTFT rises, but this point cannot yet be promoted to a
qualified resource-optimal or budget-latency-optimal configuration.

The actual unheld intervals are222.209195328s onGPU2 and56.481796434s onGPU1.
They are observed, not counterfactual savings. The complete physical integral
includes all starts/restarts and final release: pre-arrival37.447073,
arrival15485.627607, drain3.905246, cleanup42.212162 GPU-s. Do not substitute
the terminal's simulated billing projection or sum overlapping activity spans.

### Audit and mechanism observations

- All4000 request/adapter/arrival/target/prompt/generation/token-source fields
  matchD299; shared trace/subset identities match. All4000 binding, native-slot,
  lease, content and clock checks pass; maximum TTFT-stage identity error is0.
-193 output-token hashes differ fromD299. Binding consistency is not kernel
  arithmetic qualification. Preserve the differences, without automatically
  reopening the already bounded D295 numerical diagnostic.
-132 remote transfers join by UUID, clock, archive/content identity and bytes;
 131480060wire bytes, exactly the same totals asD299. No request-time packing
  or temporary archive. Common totals do not prove equal transfer-time spans.
- Dispatch tiers GPU1868/HOST1136/NVMe859/remote137; tier-conditioned groups
  are not common-request causal comparisons.
-4038 resource samples; peak20565889024B, minimum hostavailable94115979264B,
  high/max/OOM/swap0. Protected seal remains146/147unchanged; sole previously
  known baseline replay-script edit has the same recorded SHA. Raw seal pass
  remainsfalse, not replaced with an all-clean claim.
-1860 control observations:5scale-ups,2retirements,1256lowqueue+lowactive
  samples,1006of those still blocked only by the latency predicate. All-low
  maximum sampled span57.843228s under the unchanged38s cooldown.
- Mean arrival-to-admission increases from0.686342s to0.756334s; the remaining
  means are admission-to-dispatch0.341237s, engine-entry-to-queue0.078388s,
  nativequeue0.012781s and prefill0.202050s. Waiting before admission remains
  the largest mean component. This decomposition localizes a question; it
  does not prove that a particular routing or admission mechanism caused it.

### Decision and return to the main line

The reclamation/occupancy part of the hypothesis is observed; preservation of
the preceding timing quality is not. Retain D300 as a complete exploratory
resource-saving tradeoff, not a final accepted winner. Do not raise the lower
threshold again just to create more scale-ins, and do not rerun this unchanged
candidate seeking a better realization.

Next7B question: use the saved complete request/control evidence to distinguish
avoidable arrival-to-admission delay from necessary readiness/active-capacity
constraints. Cross-check the existing C2-derived soft targets, the now-cap8
runtime and earlier failed approaches against IEEE semantics and primary
implementations before selecting ONE next candidate. Soft and hard capacities
need not be equal; this is not permission to set both to8 by fiat. No new
warm/source/numerical calibration,3B,baseline or figures merely to fill the gap.
FinalG1/G2 and the full approved campaign remainactive and incomplete.

### Reproducibility

Curated outputs: `paper_results/ieee_tc/p2_backend/d300_full_w0/validated1/`
and `context1/`. The existing native/context drivers gained an opt-inD300
identity only; historical defaults and output directories remain unchanged.
Native analysis peak734461952B; context peak1062809600B, both under3/4GiB
bounds with no high/max/OOM; their empty owned scopes were stopped.
The35existing native-timing/provisional-bound tests pass; test scope peak
494972928B with no high/max/OOM. No runtime algorithm was edited for this audit.

- Source SHA: `0964d7a58feeebadf607e72f6f6ea7a09b949d6df44da684bb86d0c27e8be974`.
- Native audit SHA: `1ba3f6f5daae169f4618f413a51c6bf20c2c9833474bdf8e6c55a6f5a6f6ae99`.
- Control CSV SHA: `08185e80608d37a54737f79173f9b3ab0901df4d45ea2395aa6659b940c94be8`.
- Context SHA: `7339967323cc176f8c5deba102c4c9e73b232295f4aef3b8dc6f47855a62b67a`.
- Remote journal `transfers-4fed620e4e2a48518c284170fb890217.jsonl`, SHA
  `0ad4dd4f7161e1a56804431757c17442cc6a2e04c9f7e3e92e9360f391110b6a`.
- Remote monitor SHA: `42b131ba1c1083f7dd6951c38213877ab00973b817cb1ef548d47c6bf7c9d253`.
  Both remote/local pairs match; journal located by its processclock contents.

During post-release collection, remote `rg` was unavailable, so the equivalent
read-only `grep -l -F` located the journal. Two guessed local auxiliary launcher
paths were absent; the already established systemd bounded invocation was used.
Neither observation failure restarted inference or changed measured conditions.
