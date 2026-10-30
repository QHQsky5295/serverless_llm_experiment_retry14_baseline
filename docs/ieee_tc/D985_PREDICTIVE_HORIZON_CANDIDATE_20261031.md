# D985 — Predictive-horizon scale-up candidate (2026-10-31)

## Status

This is a development candidate, not a formal comparison. It is evaluated on
the frozen 7B W0 trace used by D983. The candidate is enabled only by the new
explicit configuration field `ieee_predictive_scaling`; D955, D983, historical
results, and all paper result directories are unchanged.

## Falsifiable hypothesis

D984 found that the severe TTFT tail is concentrated while a new replica is in
the measured activation interval. The current controller scales only after
queue/active/TTFT pressure is visible, so the first queued requests can already
be outside the service target. A controller that uses only observations already
available at a control round can reduce this activation debt by starting a
replica when the measured arrival rate over the readiness horizon would exceed
the measured service capacity.

The candidate does not read the future trace, use a request schedule, or alter
the adapter/residency/admission contract. It uses:

1. the causal arrival and completion counters observed at the controller;
2. a 45-second trailing observation window already frozen by the IEEE TTFT
   control configuration;
3. the measured per-ready-instance completion rate over that window; and
4. an EWMA of completed activation latencies, with the same `ewma_beta=0.5`
   already present in the frozen service/preparation profiles.

With arrival rate `lambda`, service rate per ready instance `mu`, queue depth
`q`, and measured readiness horizon `T`, the candidate estimates

```
forecast_work = q + lambda*T
target = ceil(forecast_work / (mu*T))
```

and clips the target only to the already configured physical instance range.
It can increase a scale-up target or cancel a scale-down that would leave the
forecast target unmet. It never invents a capacity observation when no
activation or completion rate has yet been measured; the original IEEE control
then remains the only decision rule.

## Why this is not a threshold patch

There is no new fixed queue threshold, arrival threshold, startup constant, or
future-trace hint. The 45-second window and EWMA coefficient are explicit
frozen configuration, and the readiness horizon is learned from the runtime's
own completed activations. The candidate is therefore falsifiable: on the same
trace it must either cover later activation debt with no uncontrolled increase
in lifecycle GPU seconds, or it must be rejected.

## Offline D983 replay audit (read-only)

The first audit uses the D983 request counters and completed activation
latencies without changing that result. At each two-second control point it
updates the trailing arrival/completion rates, uses only activations completed
before that point to update `T`, and evaluates the formula above. The completed
D983 trajectory is used only to score how early the candidate would have
decided; no future request row is supplied to the candidate.

Under this audit, the candidate would have exposed an earlier capacity target
for 14 of the 26 observed scale-up events. The two substantial lead times are
approximately 171.5 seconds before the second scale-up wave and 163.8 seconds
before the later wave. Several events are predicted only 0.5–5.3 seconds before
the original controller, and the first scale-up cannot be predicted because no
completed activation has yet supplied a readiness horizon. This is evidence
that the hypothesis is testable, not evidence that the candidate improves the
end-to-end result: the GPU replay is still required to measure residency,
activation overlap, and joint SLO.

## Validation sequence

1. Run unit tests and static compilation before any GPU work.
2. Perform a read-only replay audit on D983 using the event counters and
   observed activation durations. The audit must report each predicted target,
   the first causal time it would have exceeded the current target, and the
   number of additional scale-up decisions. It may use D983's completed trace
   for evaluation, but the online candidate itself may not access it.
3. If the audit shows a reduction in activation-debt exposure without an
   obviously persistent four-GPU reservation, run one GPU development replay
   with the new D985 execution key. The run remains nonformal until an
   independent numerical correctness oracle is available.
4. Compare D985 against D983 on physical owner-union GPU-s/request, overall and
   activation-affected P95 TTFT, joint SLO attainment, scale-up/scale-down
   counts, readiness horizon observations, and the forecast decision log.

## Rejection criteria

Reject the candidate and do not tune it on the test trace if it causes any of
the following: (a) a lower joint SLO with no compensating lifecycle reduction,
(b) persistent max-instance residency after the demand wave has ended, (c) a
new physical accounting inconsistency, (d) activation decisions without the
required causal counters, or (e) any correctness/oracle failure. A rejected
candidate is retained as an auditable development result and is not silently
folded into a later run.

## Literature boundary

The principle is consistent with recent autoscaling work that treats long
startup and queueing as part of the SLO path, but PrimeLoRA does not claim the
hardware mechanisms of those systems: [BlitzScale, OSDI 2025](https://www.usenix.org/system/files/osdi25-zhang-dingyan.pdf)
and [Chiron, arXiv 2501.08090](https://arxiv.org/abs/2501.08090).
