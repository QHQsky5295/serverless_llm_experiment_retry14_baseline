# D274 — 3B historical bottleneck and current-profile boundary audit

Status: CPU-only, exploratory, not a performance result and not a replay authorization.

Date: 2026-10-06

## Question

The retained D138 Llama-3.2-3B run reported a mean TTFT of 12.589 s. Before
changing the serving path, we must determine whether this is evidence of a
model-compute limit or of a control/readiness configuration that is no longer
the current PrimeLoRA path.

## Evidence and provenance

The only input used here is the bounded 33 MiB request projection from
`d138_20261001/full_full1_request_projection.json` (4,000 requests, seed 42).
The run is already classified as exploratory: native completion and request
success are present, but numerical correctness, common warm SLO, and physical
GPU lifecycle qualification are absent. It is therefore not used for G1/G2 or
for comparison with any baseline.

The D138 projection SHA and source metadata remain in the existing D138
manifest; the projection was read as a bounded artifact and the 9+ GiB raw
result was not loaded or rewritten.

## What D138 actually shows

| Quantity | D138 value |
|---|---:|
| Requests / successful requests | 4,000 / 4,000 |
| Mean overall TTFT | 12,588.6 ms |
| P95 overall TTFT | 45,713.7 ms |
| Mean service TTFT | 4,047.8 ms |
| Mean vLLM TTFT | 581.1 ms |
| Mean dispatch/admission wait | 8,540.8 ms |
| Mean ingress queue wait | 7,032.2 ms |
| Mean runtime-slot wait | 4,082.0 ms |
| Mean pre-runtime service shell | 3,466.8 ms |
| Mean dispatch-window wait | 2,950.2 ms |
| Mean LoRA I/O | 2,762.8 ms |
| Mean parent-RPC overhead | 1,269.1 ms |
| Mean TPOT | 38.58 ms |
| Physical GPU-s / request | 3.9653 (descriptive only) |

The native vLLM component is much smaller than the user-visible delay, while
queueing, dispatch/admission, runtime-slot and preparation/control spans are
large. This supports a *control/readiness dominated* hypothesis for D138; it
does not establish that the same cause remains in the current implementation.

## Current 3B profile is not D138's profile

The current YAML profile resolution was audited without starting a model. The
canonical resolved-profile SHA is
`61e4931aacc42068a21e256fde1b0e81c547cfe94b656bf86081fbdc6b55227c` (source
YAML SHA `f923c36d541046639ba70691106cf7841b322f850377bd8d719004adc959e45d`).
The selected profiles were `llama32_3b_main_modelscope`,
`azure_sharegpt_rep4000`, and `llama32_3b_auto500_formal4000_s8`.

Important differences from D138 are:

| Dimension | D138 | Current profile resolution |
|---|---|---|
| Workload concurrency | 8 | 4 |
| Runtime coordination | dedicated, `ieee_confirmed` | auto, `adapter_affinity` |
| Admission | explicit measured IEEE profile | enabled, but no D138 profile supplied by the profile selector |
| Scale-up | queue/active IEEE thresholds | predictive target + automatic startup parallelism |
| Output/input bounds | explicit 256 / 759 | inherited workload contract; model fields are null before run-time contract assembly |
| Storage bandwidth | no-delay in D138 | profile fallback is 250 MiB/s unless an explicit V2 override is supplied |
| Remote/cache roots | D138 run-specific roots | current profile's frozen artifact root and runtime-generated cache roots |

Consequently, directly “optimizing the D138 bottleneck” would confound
algorithm changes with profile changes. A current 3B result must first record
the fully assembled run contract, including generation, confirmed routing,
admission profile, cache state, and remote bandwidth.

## Falsifiable next hypothesis (not yet executed)

**H3B-1.** Under the current IEEE-confirmed PrimeLoRA profile, if the old
3B problem is still present, a bounded prefix will show a large gap between
user TTFT and native vLLM TTFT, with the excess concentrated in dispatch /
readiness / slot spans. If the gap collapses, D138 was configuration/state
specific and no control-path rewrite is justified.

The minimal validation is one fresh, explicitly witnessed 3B prefix using the
same frozen trace and adapter subset, with no parameter search. It must record
the assembled profile hash and the stage fields above. It is *not* authorized
by this document: the execution gate remains the unresolved 7B correctness,
common-warm reference, and physical whole-GPU allocation/release evidence in
the current status ledger. Once that gate is cleared, the prefix can be
considered before a 4,000-request 3B replay.

## Decision

No code change is justified by D138 alone. No 3B GPU replay, baseline run, or
ranking claim is started here. The audit narrows the next 3B experiment to a
controlled profile-bound validation and prevents the old 12.6-s figure from
being mistaken for a current-model compute limit.
