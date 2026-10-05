# D255 — D254 phase-window replay repair

Date: 2026-10-06

## Failed attempt

D254 entered the real vLLM/Full stack and created the initial GPU runtime, but
the first 8-phase window stopped after six offered requests. The dispatcher
raised `external replay cannot reset at phase/drain boundaries`. This was the
third stale single-window assumption in the same path: the external ingress
allowed only one consumer call and the dispatcher expected that call to contain
all 4,000 entries.

The attempt is therefore incomplete (`N_plan=4000`, `N_arrived=6`), not a
performance or correctness result. The raw service log, replay journal,
watchdog evidence, worker-failure workdir and physical cleanup receipt remain
under the unique D254 root. No partial latency or resource value is promoted.

## First-principles repair

The external replay is one immutable ordered stream. The service now retains a
single consumer and an explicit receive cursor. A multi-cycle window requests a
contiguous `(start_index, count)` interval; the next window must start exactly
at the previous cursor. The pump, deployment clock, request IDs, source hashes,
arrival timestamps and terminal denominator remain global and unchanged.

The default no-argument `receive()` API remains one-shot for existing paths.
Non-contiguous windows, a second consumer, gaps, and incomplete windows fail
closed. The runner maps each global ingress index back to the local phase slice
without changing the request's scheduled timestamp.

CPU-only tests cover the new two-window cursor, legacy one-shot reception,
publisher failure, startup buffering and dispatcher integration. A fresh
launch identity is required for the next retry; D254 is never overwritten.
