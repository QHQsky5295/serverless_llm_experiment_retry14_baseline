# D1075 adapter-transfer concurrency diagnostic

## Motivation

D1074 removed forecast scale-down oscillation and improved joint D648
attainment from 93.60% to 94.65%, but 190 of the 196 TPOT failures remain in
the <=64-output-token cohort. D1074 also permits three concurrent movement
lanes (`max_concurrent_loads=3`). D1075 tests whether this existing transfer
concurrency contributes to short-decode tail latency.

## One-factor intervention

Keep D1074's model, W1 trace, seed, 240 s replay gate, generation contract,
local-sim/no-delay storage, resource limits, and
`FAASLORA_IEEE_ALLOW_FORECAST_SCALE_DOWN=false`. Override only:

```
FAASLORA_MAX_CONCURRENT_LOADS=1
```

The first run is a 1,000-request prefix diagnostic using the existing runner's
registered prefix mechanism. It is not a formal comparison and cannot replace
the full trace. It is used only to determine whether serializing transfer
movement reduces TPOT tail without an obviously unacceptable dispatch/TTFT
regression. If the diagnostic is promising, a new full 4,000-request run will
be preregistered with the same one-factor change; otherwise the candidate is
rejected and D1074 remains the current control-policy candidate.

## Acceptance evidence

Record native contract integrity, joint D648 TTFT/TPOT attainment, short-output
TPOT distribution, dispatch/admission wait, LoRA I/O, transfer concurrency,
GPU seconds, scale events, OOM/watchdog status, and release confirmation. No
thresholds or seeds may be changed after seeing the result.

## D1075 prefix attempt outcome

The registered 1,000-request prefix attempt was rejected before serving by the
existing online-binding invariant:

```
ValueError: IEEE transfer pressure limit must match the actual shared movement owner
```

The attempted override changed `max_concurrent_loads` to 1 while the frozen
native admission profile and shared movement owner were both calibrated for
`transfer_limit=3`. No request was served, no performance data was collected,
and the failed launch is retained as a failed run key. We do not rewrite the
profile or bypass this invariant. A valid concurrency experiment would first
need a separately measured native profile with the new transfer limit, as
required by the existing D1011/D1012 precedent; it is therefore not part of
the current optimization loop.

