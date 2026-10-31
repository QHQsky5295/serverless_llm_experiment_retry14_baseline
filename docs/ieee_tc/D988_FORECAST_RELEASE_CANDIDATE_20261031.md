# D988 — forecast-based capacity release candidate (2026-10-31)

D987 demonstrated that causal readiness-horizon forecasting removes the W0
activation debt, but it kept four replicas for the complete replay and raised
physical GPU-s/request by 59.6%. D988 keeps the same arrival/service/horizon
forecast and adds one explicit control rule: when the forecast target remains
below the current ready-plus-pending capacity, the existing frozen IEEE
scale-down cooldown must elapse before one replica is released. The action is
one-at-a-time and never runs while an activation is pending.

This is not a new idle threshold or a trace-specific timer. The release target
comes from the same measured capacity forecast; the delay is the existing
`scale_down_cooldown_s=38` configuration. The candidate must be rejected if it
reduces joint SLO, releases capacity while the forecast target is unmet, or
still leaves physical GPU-s/request materially above the D983 reference without
an equivalent tail/SLO gain.

The run is a new nonformal development key on the unchanged frozen 7B W0
trace. It requires the same correctness caveat as D987 and must produce a
complete physical owner-union summary before interpretation.
