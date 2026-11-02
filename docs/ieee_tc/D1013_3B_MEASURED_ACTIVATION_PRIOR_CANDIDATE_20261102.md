# D1013 3B measured-activation prior candidate

## Hypothesis

D1000's first two scale-out submissions were reactive: they were issued at
request indices 217 and 221 only after the TTFT signal crossed the controller
limit.  The corresponding measured activation times were 35.9377 s and
38.0473 s (the later two were 37.5491 s and 38.2545 s).  During this delay the
run recorded 127 and 7 scale-up-affected requests.  D1007's fixed two-instance
floor improved the joint SLO only from 92.25% to 93.00% while increasing
physical GPU time by 4.97%, so a permanent floor is not a suitable fix.

The candidate supplies the previously measured maximum activation horizon as a
causal warm-start prior to the IEEE predictive controller.  The controller
still requires online arrival-rate and busy-instance service-rate observations;
the prior by itself cannot trigger scaling.  Once an activation completes, the
measured latency replaces the prior through the existing EWMA update.  No
future trace rows or adapter hotness are consulted.

## Frozen change

Only `resource_coordination.ieee_predictive_scaling.initial_activation_horizon_s`
is added to the D1000 3B W2 configuration.  Its value is the Type-1 maximum of
the four D1000 physical activation observations:

```
max(35.9376669740, 38.0473159272, 37.5491194869, 38.2545367524)
= 38.2545367524 s
```

All model, backend, trace, adapter map, service/preparation profiles, cache
policy, limits, workload, and remote delivery settings remain unchanged.

## Expected interpretation

The candidate is accepted only if the full 4,000-request audit shows correct
completion, the common joint TTFT/TPOT SLO reaches at least 95%, and no
unexplained provenance or lifecycle violation is present.  A resource increase
is reported explicitly; it is not hidden by the prior.  If the joint SLO does
not improve enough, this candidate is rejected and its raw result is retained.
No figure is generated before qualification.

## Provenance

- Parent development reference: D1000 3B route-confirmed arrival prefetch W2.
- Prior source: D1000 `E1_scale_up_events.engine_activation_ms`.
- New execution key: `d1013_20261102/d1013_3b_measured_activation_prior_w1`.
- This is an optimization candidate, not a final result or a baseline.
