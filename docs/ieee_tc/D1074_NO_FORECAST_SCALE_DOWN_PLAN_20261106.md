# D1074 no-forecast-scale-down diagnostic

## Question

D1073 reached the common 240 s startup window but still showed 3,744/4,000
joint TTFT/TPOT-good requests. Its controller released seven replicas using
`ieee_forecast_capacity_release` and then reactivated capacity during later
bursts. D1074 tests whether that declared predictive-release mechanism, rather
than the request path or cache state, is the dominant source of tail loss.

## One-factor intervention

Keep the D1047 configuration, W1 trace, seed 42, fixed-length greedy contract,
local-sim/no-delay storage, resource limits, and 240 s replay gate unchanged.
Override exactly one existing feature gate at launch:

```
FAASLORA_IEEE_ALLOW_FORECAST_SCALE_DOWN=false
```

This does not change any paper equation, routing semantics, adapter set, model,
or threshold. It disables only proactive forecast-capacity release; ordinary
reactive scale-down remains available. The run is a diagnostic and is not a
formal held-out result.

## Acceptance and rejection

Record contract integrity, startup, joint TTFT/TPOT attainment under D648,
GPU-seconds, scale-up/scale-down count, dispatch/admission, cache/GPU hits,
and request-window behavior. Accept the mechanism for the next frozen
candidate only if joint attainment improves without an unjustified increase in
GPU occupancy/cost relative to D1073. If the tail does not improve or the
resource objective regresses materially, reject this intervention and retain
the existing policy. Do not choose seeds or thresholds after observing the
result.

## Provenance

The run must use a new immutable run key under
`results/ieee_tc/p2_backend_qualification/d1074_20261106/`; D1073 and all
previous raw outcomes remain untouched. Store the exact environment override,
configuration SHA, trace SHA, adapter-subset SHA, and source commit in the
launcher record.

