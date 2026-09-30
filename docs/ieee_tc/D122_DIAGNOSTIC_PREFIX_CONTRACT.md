# D122 — Explicit diagnostic prefix through the integrated control path

2026-09-30. Measurement-support change, not a serving optimization or a GPU
performance result. Baselines remain paused. The nine IEEE equations, online
policy, runtime capacities, 1,800-second development deadline and resource
limits are unchanged.

## Why this is needed

D119's current 7B Full completed only 2,890 of 4,000 requests under the native
generation contract. D120 located substantial waiting before the dispatch gate,
but did not identify its cause. D121 measured one historical native-state copy
component; it cannot establish current 7B control-path cost. Repeating an
unchanged Full or changing concurrency on that evidence would not isolate the
bottleneck. The next diagnostic therefore profiles the predeclared first 1,000
original 7B requests through the real integrated path.

`FrozenReplayPlan.load(count=...)` already supported immutable index views.
However, the publisher, main runner and shared-trace loader reconstructed the
whole source. Setting a configuration's request count to 1,000 alone did not
truncate a shared trace. D122 connects that existing index view end-to-end; it
does not introduce another workload generator or serving framework.

## Execution contract

| Property | Default full-source replay | Explicit diagnostic prefix |
|---|---|---|
| Source file | Existing immutable trace | Same existing immutable trace |
| Requests | Entire source | First K in original arrival/ID order |
| Arrival offsets and rate | Existing W0/W1, rate 1 | Same prefix offsets, rate 1 |
| Receipt identity | `full_source_v1` (also legacy absent marker) | `diagnostic_prefix_v1` and exact K |
| Integrated contract | `ieee_full_integrated_replay_v1` | `ieee_diagnostic_prefix_replay_v1` |
| Population identity | Full SHA, view SHA, count, source count | Full SHA, prefix view SHA, K, source count |
| Formal-run flag | Existing provenance checks still apply | Rejected before model initialization |
| Preparation/arrival/lifecycle | Existing owners and 60-second notice | Same owners and notice |
| Qualification | Execution receipt alone is insufficient | Never Full or formal-comparison evidence |

Opt-in interface: `FAASLORA_TC_DIAGNOSTIC_PREFIX_COUNT=1000`, together with
the existing guarded external-replay wrapper. The supervisor passes the count
to the original publisher; its ready receipt must match the requested count.
The service reconstructs and verifies the exact source/view, and the shared
loader selects only the bound prefix. All 500 adapters remain available;
selection does not reduce the static artifact universe. It creates no new trace,
weights, prompts or remote delivery pool.

Missing/unknown scope, changed input SHA/view, improper count (including bool),
full-source count labelled as a prefix, tiny witness, rate-scaled input, HTTP
transport substitution and formal-prefix use are rejected. The unchanged
full-source default cannot silently accept a shorter plan. Prefix terminal
accounting uses K; source count stays independently recorded. A successful
1,000-request diagnostic must never be reported as 4,000-request completion.

## Verification and limitations

| Check | Result |
|---|---|
| System-Python launcher/protection tests | 66 passed, 1.036 s |
| Existing replay, launch, physical lifecycle and basic smoke tests | 455 passed, 37.085 s |
| Actual Unix fixture transport before model readiness | Full and prefix retain original IDs/arrival clock |
| Physical deployment fixture | Original source count and offered prefix both preserved |
| Notice, generation, timeout and owner checks | Still enforced |
| Formal prefix / unmarked shortened input | Rejected |
| GPU or remote experiment in D122 | None |

The initial mixed-interpreter suite had 228 passes and four failures because
the model environment lacks `signal.pidfd_send_signal`, required by the actual
launcher. Its wrapper, full log and time receipt are retained. Launcher tests
were then run with their qualified `/usr/bin/python3`; protection code was not
weakened or mocked to bypass the missing primitive. One publisher-propagation
test was added before the final 66-test system-Python suite.

The fixture tests validate measurement wiring, not inference throughput or
profiler overhead. This diagnostic is not a new independent trace, not a paired
performance comparison with D119, and not warm/SLO or numeric-adapter evidence.
Detailed profiling can perturb timings; sampled Python/GIL stacks are not
whole-system wall-time attribution. Reuse the already qualified py-spy 0.4.2
wrapper; its official documentation describes these sampling limitations:
[py-spy 0.4.2](https://github.com/benfred/py-spy/blob/v0.4.2/README.md).

## Evidence and next action

Raw: `results/ieee_tc/p2_backend_qualification/d122_20260930`.
Curated: `paper_results/ieee_tc/p2_backend/20260930_d122_diagnostic_prefix`.
This status table, rather than a fabricated speedup plot, is the appropriate
deliverable. Source identities, protected-result verification, test logs and
exact-owned empty-scope cleanup are saved with it. No legacy result is replaced.

Next: one current 7B first-1,000 W0 controller profile, same D119 configuration
and D89 initialization, fresh owned paths, existing published remote cache.
Only after its evidence select one causal optimization; any accepted candidate
still requires ordinary full 4,000-request replay. Warm/Resident, numerical
adapter qualification, baselines, M1/M2, A1–A5 and S1–S13 remain outstanding.
