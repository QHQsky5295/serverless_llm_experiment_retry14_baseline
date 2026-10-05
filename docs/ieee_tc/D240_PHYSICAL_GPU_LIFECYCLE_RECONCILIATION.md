# D240 — Physical GPU lifecycle reconciliation for old PrimeLoRA runs

Date: 2026-10-06

Status: **measurement correction and code-gate definition; no replay**.

## Finding

The frozen G1 metric is the union of verified physical GPU allocation intervals:

\[
U=\sum_d\int a_d(t)\,dt,\qquad R=U/N_{correct}.
\]

The existing result JSON also contains `infra_gpu_seconds_total` and an
`instance_lifecycle_log`. Those fields are a legacy billing projection. They are
not interchangeable with the physical owner-union summary. In the D210 and D219
7B runs, the legacy view contains only the primary runtime lifecycle even though
each run has three scale-up events and four released physical allocation
journals.

## Evidence

| run | legacy `infra_gpu_seconds_total` | physical `summary.json` GPU-s | physical journals | formal status |
|---|---:|---:|---:|---|
| D210 ordinary Full | 4,317.0107 | 15,916.8239 | 4, all released | exploratory, correctness not qualified |
| D219 confirmed snapshot Full | 4,317.0500 | 15,911.7125 | 4, all released | exploratory, correctness not qualified |
| D138 confirmed 3B Full | 4,331.3333 | 15,861.3305 | 4, all released | diagnostic, not a qualification point |

The D210→D219 physical change is −5.1114 GPU-s (−0.0321%), not the apparent
zero change obtained by comparing only the legacy primary-runtime values. The
correct physical quantity therefore does not alter the qualitative conclusion:
the confirmed snapshot improved mean/control-path latency, worsened P95 TTFT,
and did not yet demonstrate a material lifecycle reduction. It does change the
numeric value that any future G1 table must use.

## Why this is not a result rewrite

No raw result, allocation journal, request denominator, execution key, or paper
source was changed. The correction only selects the resource quantity already
sealed by the run's `physical_deployment/summary.json`. The legacy value remains
available as a separate billing diagnostic. It must not be labelled “lifecycle
GPU-s” or used for G1/G2 eligibility.

## Required implementation gate before the next formal replay

The runner/analyzer must expose a canonical field such as
`physical_gpu_seconds_total` with the following invariants:

1. it is read from the finalized physical deployment summary, not inferred from
   request completion or the logical instance list;
2. all allocation journals referenced by the summary exist, hash-verify, and
   have a release receipt;
3. the value equals the union-weighted device intervals, including scale-out,
   startup, drain, and cleanup intervals, without double-counting overlaps;
4. `physical_gpu_seconds_per_correct_request` is N/A until
   `N_correct=N_plan` and the native correctness gate passes;
5. formal figure/table generators reject a record that has only legacy billing
   GPU-s or whose physical summary is missing/incomplete;
6. both physical and legacy fields are printed with unambiguous names so later
   cost sensitivity cannot silently reuse the wrong one.

This is a measurement-integrity fix, not a performance optimization. It must be
completed and fixture-tested before interpreting any PrimeLoRA or baseline point
as G1. The next performance hypothesis remains readiness/first-service tail;
the metric gate is a prerequisite and does not authorize a new full replay by
itself.
