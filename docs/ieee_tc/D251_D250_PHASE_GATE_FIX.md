# D251 — D250 launcher-gate mismatch and bounded repair

Date: 2026-10-06

## Failed attempt

D250 Full attempt `d250_20261006/7b_readiness_8phase_full` was rejected before
model initialization. The guarded launcher created the auxiliary replay and
service scopes, but `run_all_experiments.py` raised:

`ValueError: external replay requires one explicit scenario/run and the frozen shared trace`

The rejection was caused by an old guard that required
`multi_cycle_phases == 1`. It was not a GPU, vLLM, remote artifact, request,
memory, or correctness failure. The launch receipt records `N_arrived=0`,
`service_returncode=1`, `watchdog_returncode=0`, and complete GPU-context and
scope cleanup. The raw failed receipt remains preserved and is not a result
denominator.

## First-principles repair

The guarded external replay already supplies one immutable W0 arrival view,
with one explicit scenario and one run. Multi-cycle execution only partitions
that same view into registered phase chunks and invokes the existing phase
boundary scale-down/scale-up path; it does not resample arrivals, reset
request IDs, or create a second replay. The `multi_cycle_phases == 1` check was
therefore stricter than the execution model and contradicted the approved
8-phase readiness protocol.

The repair removes only that incorrect equality check and retains all other
external-replay requirements: one run, one explicit scenario, a frozen shared
trace, a guarded replay receipt, matching trace SHA, and a positive phase count.
No timing, routing, admission, readiness, backend, or resource formula is
changed.

The repaired source is covered by Python compilation and `git diff --check`
before retry. A new unique launch identity and fresh result root are required;
the failed attempt is never overwritten.
