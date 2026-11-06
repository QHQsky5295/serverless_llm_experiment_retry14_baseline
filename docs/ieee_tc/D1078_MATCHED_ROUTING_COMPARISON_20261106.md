# D1078 — matched service-bin control for D1077 (2026-11-06)

> Correction (D1080): both runs use a 240-second preparation window, not the
> frozen 60-second main protocol. D1077 is NoRouting, not paper Full.
> Physical lease-union GPU-seconds are 6596.153619553894 (D1077) and
> 6746.289175793994 (D1078); the earlier table used scenario infrastructure
> estimates. Its physical-resource percentage and interpolated quantiles are
> superseded by [D1080](D1080_DIAGNOSTIC_IDENTITY_AND_RESOURCE_CORRECTION_20261106.md).
> Preserve the original measurements; do not promote load_only to paper Full.

## Purpose and status

D1078 closes the confound identified in the D1077 audit.  It uses the same
3B model, W1 trace, fixed-length greedy generation, remote delivery contract,
profiles, resource envelope, confirmed-arrival prefetch, handoff, hierarchy,
and **effective-capacity admission enabled**.  The only intended feature
difference is the replica selection key:

| Run | Selection key | Admission |
|---|---|---|
| D1077 | `load_only` | enabled |
| D1078 | `service_bins` | enabled |

Both are seed-42 exploratory runs.  They are mechanism evidence, not repeated
seed estimates or formal ranking points; `n_correct` remains unknown for both.

The first D1078 launch key was rejected before request execution because of a
manually mistyped preparation-profile context digest.  It is retained as an
invalid prelaunch record.  `retry1` is the only completed D1078 run.

## Provenance

| Item | D1077 | D1078 |
|---|---|---|
| Run key | `d1077_3b_load_only_full_w1_replay240_retry2` | `d1078_3b_service_bins_admission_on_w1_replay240_retry1` |
| Config commit | `39e65069` | `a106cbd5` |
| Result SHA-256 | `bec4b261076e065d42629954452e1e82e46175b17e236251b341a872ca2af075` | `e854aefb67c2cc9f9fe8f4a40a6fc8898e544cef6f109314efdd2823c67abcb8` |
| Physical summary SHA-256 | `efbd8a4be6f079630e05ceb271fb9849ee876391ec3696540e66cfc318ac3` | `a6c5401536290e06a2ce1560c972928f9cb172179ccd54c1a223f55737410d6d` |
| Native audit | `d1077_load_only_full_w1_audit/native_timeline/summary.json` | `d1078_service_bins_admission_on_w1_audit/native_timeline/summary.json` |
| Native audit SHA-256 | `d306fa2051b61f5029610be3a0add0199863da552798333c931c5196211aaa6a` | `a33f5e8b2d7c987882b3c464066dcfc04c0ed2ad635f122d336f829a8d76918a` |

Both native audits found 4,000 terminal records, one timing domain, 4,000
native-success records, and zero failed requests.  The audits are occupancy
and contract checks; they do not certify numerical adapter correctness or
replace the physical GPU lifecycle summaries.

## Matched outcome

Lower is better for latency, E2E, and GPU-seconds; higher is better for CE.

| Metric | D1077 load-only | D1078 service-bins | Load-only change |
|---|---:|---:|---:|
| Average TTFT (ms) | 460.859 | 485.532 | **−5.08%** |
| P95 TTFT (ms) | 1,069.542 | 1,112.310 | **−3.85%** |
| Average TPOT (ms) | 16.943 | 17.688 | **−4.21%** |
| Average E2E (ms) | 2,287.304 | 2,378.689 | **−3.84%** |
| P95 E2E (ms) | 4,964.606 | 5,130.168 | **−3.23%** |
| Physical GPU-seconds | 6,952.670 | 7,101.375 | **−2.09%** |
| Monetary CE | 380.073 | 360.049 | **+5.56%** |
| Infrastructure CE | 314.409 | 295.999 | **+6.22%** |
| Average dispatch/admission wait (ms) | 208.888 | 214.542 | **−2.64%** |
| P95 ingress queue wait (ms) | 370.297 | 428.173 | **−13.52%** |
| Cache hit / GPU hit | 96.70% / 81.00% | 96.73% / 83.63% | descriptive |

D1077 completed both scale-up events at request indices 24 and 293.  D1078
created its second and third replicas at the same request index 24.  Thus the
service-bin control held three GPUs for more of the lifecycle, producing
7,101.375 versus 6,952.670 physical GPU-seconds despite its higher GPU hit
rate.  This is consistent with a placement/scaling interaction, not evidence
that higher GPU-hit rate alone is sufficient.

## Common D648 timing gate

Using the frozen, common thresholds (prompt tokens `<=506`: TTFT 1466.0508 ms,
TPOT 29.9219 ms; prompt tokens `>506`: TTFT 2539.6487 ms, TPOT 36.2155 ms):

| Run | TTFT pass | TPOT pass | Joint pass | Timing fraction |
|---|---:|---:|---:|---:|
| D1077 load-only | 3,983 | 3,858 | 3,844 | 96.10% |
| D1078 service-bins | 3,982 | 3,828 | 3,812 | 95.30% |

The common minimum is 3,800/4,000.  Both runs clear the timing gate, but D1077
has 32 more joint passes.  The short-prompt joint counts are 975 versus 970;
the long-prompt counts are 2,869 versus 2,842.  These are descriptive
request-level results from one seed, not confidence intervals.

## Interpretation

Under the admission-matched comparison, load-only routing is the better
candidate on every primary direction observed here: lower mean/P95 TTFT, lower
TPOT and E2E, lower physical GPU-seconds, higher monetary/infrastructure CE,
and more joint SLO passes.  The difference is not a generic “cache hit” effect:
D1078 has a slightly higher GPU-hit rate, yet its more aggressive early scale-out
increases lifecycle GPU time and its service-bin key leaves more queueing.

The evidence supports the following narrow claim:

> For this fixed 3B W1 workload and admission-enabled implementation, the
> existing service-time-bin-first replica key is inferior to the owned
> admitted-load key in both tail service and lifecycle resource use.

It does not, by itself, prove a universal routing theorem, establish numerical
adapter correctness, or justify a multi-seed significance claim.  The next
PrimeLoRA step is to preserve the load-only candidate while auditing the
request-level service-bin/load and selected-replica admitted-load fields for
the short-output cohort.  No figure is generated until that mechanism audit is
complete.
