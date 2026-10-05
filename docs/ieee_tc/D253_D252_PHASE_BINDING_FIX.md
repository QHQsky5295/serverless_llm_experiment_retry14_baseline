# D253 — D252 multi-cycle replay binding failure

Date: 2026-10-06

## Failed attempt

D252 Full retry1 passed the guarded launch and entered the existing runner, but
it stopped before model initialization. The scenario constructor rejected the
same 8-phase external replay with:

`ValueError: model replay requires a continuous frozen-input origin`

The second stale guard was inside `_configure_external_replay`; it repeated the
old `multi_cycle_phases == 1` assumption after the top-level launcher guard had
been repaired. D252 therefore has `N_arrived=0`, no generated result, and no
performance or correctness denominator. The service scope, watchdog, GPU
context, and temporary roots were cleaned up.

## Repair boundary

The constructor now rejects only the genuinely incompatible `tiny_witness`
origin and verifies a positive phase count. It continues to require the same
guarded receipt, exact frozen trace request order, matching replay context, and
successful replay pump. No request timing, phase partition, scaling,
handoff/readiness, or backend behavior is changed.

A new D254 identity is required for the next retry; D252 remains immutable
failure evidence.
