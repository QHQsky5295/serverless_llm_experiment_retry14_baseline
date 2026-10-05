# D242 — 7B readiness-publication hypothesis after D235

Date: 2026-10-06

Status: **offline hypothesis registration; no replay and no baseline**.

## Observation

D219 activates three new dedicated replicas after approximately 44.91–46.57 s.
The confirmed dispatch records cover all 3,003 requests served by those
replicas, but only 10 of the first 60 selected requests were confirmed GPU
sources; the remaining 50 were HOST/NVMe/Remote. The earliest new-replica
requests include an 11,288.8-ms TTFT. D235 also shows that the native
`ieee_confirmed` path publishes the engine as soon as the runtime is initialized
while its activation preparation task may still be pending. The legacy
`scaleup_first_service_*` counters are undefined for this path and will not be
reused.

## Falsifiable hypothesis

H1: the measured scale-out tail is partly caused by publishing a new runtime to
the confirmed router before its activation-time preparation epoch has reached a
terminal, evidence-bound state. A readiness-publication barrier (or an
equivalent ready-time preparation epoch) should reduce first-service
activation-to-token and scale-up-affected P95 TTFT, at the cost of a possible
shorter period in which only the incumbent replicas carry load.

H0: the first-service tail is not materially changed by the publication
boundary; the dominant delay remains engine startup, queue admission, or
request-specific remote materialization. Under H0, delaying publication should
not improve the scale-up group after the common workload and SLO gates.

## Existing mechanism to test first

The pre-registered `ieee_handoff_policy=delayed` path is the controlled first
test. It keeps the already frozen plan, source observations, target adapters and
budget, and changes only the start boundary of the handoff work until the
initialized engine is available. It must not be described as a new algorithm
or as proof of GPU readiness. The current `full` path is the paired reference;
`no_handoff` is a mechanism boundary only, not a replacement Full system.

If `delayed` does not close the tail, the next candidate is a ready-time
replanning epoch that is explicitly bound to a post-startup native snapshot and
the current visible demand frontier. It is not permitted to reuse a stale
pre-start plan, attach the old legacy handoff fields, or mark a non-GPU source
as GPU-ready. That candidate requires a separate code checkpoint and bounded
unit evidence before any replay.

## Next admissible evidence

No GPU action is authorized while the current inference-host preflight fails.
The authoritative current memory requirement is

\[
A_0\ge 80+4+16+2=102\ \mathrm{GiB},
\]

from the approved plan and `ieee_tc_preflight.py`; older 109.5-GiB prose is
historical and will not be silently mixed into a run. When the unchanged gate
passes, the order is:

1. complete the already registered numerical/output-reference gate;
2. run the explicit-admission bounded diagnostic required by D226, retaining
   all typed decisions and physical owner evidence;
3. run paired 7B `full` and `delayed` natural-scaleout diagnostics using the
   existing trace and adapter subset, with the same physical-resource sidecar;
4. require at least 20 natural-scaleout epochs with a real first dispatch before
   interpreting a first-service distribution; otherwise expand once under the
   same key and report insufficiency; and
5. only if the mechanism and correctness gates pass, evaluate a fresh held-out
   7B block and compare G1/G2 under the frozen common SLO and physical budget.

The diagnostic records `engine_ready`, `preparation_start`,
`preparation_terminal`, first dispatch, first token, selected confirmed tier,
source/owner/epoch identity, and whether the first request was admitted or
deferred. A failed or unused epoch is reported as such; another request or
another replica cannot fill its first-service slot.

This registration narrows the next optimization to one causal boundary. It does
not authorize a full replay, claim a PrimeLoRA win, or advance to 3B/baselines.
