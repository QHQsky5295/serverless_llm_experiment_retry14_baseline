# D388 — 7B incumbent audit and next-gate decision

Date: 2026-10-10.  This is an offline, read-only comparison of the closed
D371, D379 and D386 development replays.  It launches no GPU workload, does
not alter a result, and does not advance the Prime stage.

## Decision

D379 is the current 7B development incumbent for resource-oriented follow-up:
it has the smallest recent physical GPU lifecycle ledger (12,076.449893
GPU-s) while completing the native request and fixed-output contracts.  It is
not a formal G1/G2 result because `n_correct` and the common warm/Resident
references remain unresolved.

D386 is rejected as an optimization candidate.  It changed only
`queue_upper: 8 -> 4` relative to D379, but physical GPU lifecycle time rose
to 14,914.097249 GPU-s (+23.50%), mean/P95 user TTFT rose to
1419.680/3495.195 ms, dispatch/admission wait rose to 854.473 ms, and the
provisional timing attainment fell from 0.9778 to 0.9725.  The service-TTFT
decrease is therefore not a Pareto improvement.

No new capacity or threshold sweep is authorized from this audit.  The next
7B action must first close the frozen correctness/common-reference boundary,
or introduce one separately registered mechanism hypothesis supported by new
stage evidence.  A lower queue trigger is not that hypothesis.

## Evidence and confounding audit

| replay | runtime cap / max LoRA | active thresholds | policy/profile identity | physical GPU-s | mean/P95 user TTFT (ms) |
|---|---:|---|---|---:|---:|
| D371 | 14 / 14 | 0.50 / 0.25 | atomic interleave + protected file demand; D362 profiles | 14,482.524 | 1150.798 / 2838.109 |
| D379 | 8 / 8 | 0.875 / 0.4375 | no opt-in policy fields; D321/D298 profiles | 12,076.450 | 1313.421 / 3135.892 |
| D386 | 8 / 8 | 0.875 / 0.4375 | same as D379; only queue trigger changed | 14,914.097 | 1419.680 / 3495.195 |

D371 cannot be used as a one-factor comparison against D379: it changes
capacity, active hysteresis, preparation/service profiles, and two policy
switches together.  D386 is the valid one-factor test of the queue trigger and
provides the negative result above.

The request-stage counters are consistent with that interpretation.  D386
caused earlier scale-up events (for example, the second activation at request
298 rather than D379's request 588), but its scale-up first-service GPU-hit
rate fell from 0.2222 to 0.1111 and its scale-up cold-start distribution did
not improve.  Earlier activation alone is therefore not evidence of useful
readiness; it can extend GPU lifetime without reducing the user-visible
admission path.

## Correctness boundary

Both D379 and D386 have 4,000/4,000 terminal/native-contract records and no
cleanup or resource-scope failures, but both deliberately retain
`n_correct=null`.  D385's 100-request HF/PEFT witness matched 92/100; all
eight divergences were top-2 near ties (margin 0 or 0.015625) with no adapter,
prompt, length, or resource fault.  This is useful numerical-sensitivity
evidence, not permission to relax the exact witness or infer full-workload
correctness.  The current metric protocol also requires the common warm and
Resident reference before G1/G2.  No ranking, baseline, 3B run, ablation,
sensitivity result, or figure is derived from D388.

## Protected next step

Before another heavy replay, perform only the already-defined offline closure
checks for the correctness/common-reference boundary and re-read the source
plan.  If a serving candidate is later justified, register exactly one
one-factor configuration against D379 with a fresh result/cache identity and
the same trace, adapter subset, generation contract, profiles, memory gates,
and remote delivery semantics.  Preserve D371/D379/D386 as separate evidence;
do not relabel D379 as formal, do not select a seed by outcome, and do not
change the SLO or correctness ruler.

## Source hashes

- Plan SHA: `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`
- D379 result SHA: `772ae04ec266e1b0e69c7a1075b7c8bf8a93eb2fc322a2f8ff356a99bc9874d6`
- D386 result SHA: `f0214e0dfe978fdbb120e16b0a7984daa60f5c19f6313c3d5a8bcd3f7f1b0b66`
- D386 config SHA: `3b60dc468132b9c2e7ffd4bbe23ccac06b3f9b3827465250cc2af4de0541a6fb`
- D386 runner SHA: `2f297847efccbb84b97e1612793b6be46058ea2e1053b9731ed440c6fade5858`
