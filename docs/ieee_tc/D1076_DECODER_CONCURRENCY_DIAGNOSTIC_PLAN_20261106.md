# D1076 decoder-concurrency diagnostic

## Motivation

D1074's remaining joint failures are dominated by short-output TPOT: 190 of
196 TPOT failures have at most 64 output tokens. The transfer-limit test cannot
be performed without a new native admission profile. The next reversible,
profile-consistent diagnostic therefore tests the serving runtime's declared
sequence concurrency, while retaining the no-forecast-scale-down policy.

## One-factor intervention

Keep the D1074 model, trace, seed, 240 s replay gate, generation contract,
storage semantics, and resource limits. Set both existing runtime caps to four:

```
FAASLORA_MAX_NUM_SEQS=4
FAASLORA_RUNTIME_CONCURRENCY_CAP=4
FAASLORA_IEEE_ALLOW_FORECAST_SCALE_DOWN=false
```

The two runtime-cap overrides are one logical intervention (decoder batch
concurrency), and keeping them equal avoids an accidental cap mismatch. The
first run is a 1,000-request registered prefix diagnostic. It is not a formal
comparison; it only tests whether short-output TPOT tail improves without an
unacceptable TTFT/dispatch regression. A full run will be preregistered only
if the prefix provides evidence in that direction.

## Acceptance evidence

Record contract integrity, D648 length-conditioned TTFT/TPOT attainment,
short-output TPOT, TTFT and dispatch/admission, native decode timing, GPU
seconds, scale events, OOM/watchdog status, and GPU release. No threshold or
seed selection is permitted after observation.

