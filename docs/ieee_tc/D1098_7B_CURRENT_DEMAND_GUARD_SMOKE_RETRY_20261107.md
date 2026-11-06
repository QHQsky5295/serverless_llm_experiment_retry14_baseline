# D1098 — 7B current-demand guard smoke retry

D1097 passed the resource preflight but terminated before replay because the
launcher labeled a seed-42 trace as `validation`; the runner correctly permits
seed 42 only under its `smoke` role. No request, model output, or performance
measurement was produced. The D1097 launch receipt and zero-arrival replay
remain retained as a protocol failure.

D1098 repeats the same bounded current-demand guard experiment with the runner's
permitted `smoke` role. The model, first-1,000-request limit, trace, adapter
subset, generation contract, remote delivery and guard setting are unchanged.
This is a protocol correction, not a performance-selected rerun.
