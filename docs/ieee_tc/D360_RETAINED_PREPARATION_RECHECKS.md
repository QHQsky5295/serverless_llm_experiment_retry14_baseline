# D360: retained preparation receipts identify global-epoch retry amplification

2026-10-09. Development diagnosis only, not a new GPU experiment or a formal
qualification. This follows D359's additive TTFT partition, using every request
in the same D353/D358 7B W0 runs (4,000 each). Plan and frozen V1 are unchanged.
The published remote delivery cache already exists; the user's reconfirmed
approval does not require generating another copy.

## Observed results

| Retained measurement | D353 | D358 |
|---|---:|---:|
| Requests with stale-state rechecks | 267 | 263 |
| Total stale-state rechecks | 593 | 1,609 |
| Maximum rechecks for one request | 12 | 105 |
| Requests whose final transaction actually loads | 1,416 | 1,403 |
| Mean admission → executable acquisition, all requests (ms) | 154.966 | 272.560 |
| Mean executable acquisition → engine entry, all requests (ms) | 108.973 | 132.087 |
| Mean admission → engine entry, all requests (ms) | 263.939 | 404.647 |
| Mean admission → final load start, loaded subset (ms) | 386.488 | 580.420 |
| Mean final native load, loaded subset (ms) | 41.888 | 42.492 |

Second 500-request phase: rechecks increase from 108 to 1,023, affecting 39
versus 64 requests. Mean admission-to-acquisition increases from 157.351 to
925.850 ms; mean final native load is 45.670 versus 44.183 ms. All eight phases,
all selected tiers, and zero/positive recheck cohorts remain in `groups.csv`.
These selected-tier cohorts are post-routing descriptions, not matched causal
effects. Neither the phase nor the loaded subset is a replacement for full-run
G1/G2 evaluation.

D358 request `req_00615` has 105 stale rechecks, 42,814.410 ms from admission to
acquisition and 43,186.485 ms to engine entry. It was selected from NVMe, but
the final successful transaction found its adapter already on GPU and invoked
no load. Therefore it has no final-load duration, not a fabricated zero-cost
file load. Its last RPC duration describes only that RPC, not all retries.

## What the evidence supports

1. D359's increased pre-engine time is not explained by slower final native
   loading alone. The count of affected requests barely changes; repetition
   per affected request grows substantially.
2. `IEEEBackendGPUReferences._load_and_acquire` rejects a request without a
   native source-copy ID when the caller's worker epoch differs from the live
   epoch. The controller refreshes that epoch and retries across another RPC.
   Other adapters' reference/pin/cache changes can also advance this global
   epoch; the check is broader than this file's immutable content identity.
3. Exact GPU/HOST copy acquisitions already distinguish copy identity from
   unrelated global-epoch progress and recheck current capacity locally.
   File-demand loading still uses the global snapshot equality rule.
4. D355 allows model/control progress between atomic commands. D358 shows more
   repeated stale-state outcomes under that policy, consistent with a controller
   chasing a changing worker-wide version. This is an implementation hypothesis
   with direct retry evidence, not a proven counterfactual speedup.

Do not claim every model iteration increments the epoch. Do not call all time
before the final load RPC waiting: it may include file preparation, transfer,
competing work, multiple control calls and capacity waiting. `last_conflict`
is a final observation, not a full conflict journal. One run per version gives
no run-level CI. Numerical `N_correct` and formal qualification remain unknown.

## Primary-source and history crosscheck

Official vLLM 0.30.0's LRU worker performs adapter lookup, native loading, cache
capacity handling and activation on its single-threaded core loop. This supports
examining worker-local atomic operations, not removing live safety checks.
[Official worker manager, add_adapter](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/lora/worker_manager.py).

Existing project history includes exact source-copy protection, causal identity
frontier publication, bounded source-read pins, D333 fused pending/GPU-reference
acquisition, and D352 same-transaction GPU-hit HOST observation. Reuse their
ownership/cancellation contracts; do not introduce cross-request cached HOST
capacity, known unsafe in D329, or a blind stale-command fallback.

## Single next hypothesis, acceptance gates and non-goals

For a held, content-verified local file source, perform live native validation
and request-priority load/acquire in one serialized owner transaction, rather
than reuse a controller-side capacity snapshot and repeatedly chase its epoch.
The file witness, native owner, logical/content binding, live slot/pin/HOST
budget checks, completion fence, lease replay and cancellation ownership must
remain mandatory. GPU/HOST exact-copy selection, proactive Eq. (8), router
snapshots, allocation accounting and all nine paper equations stay unchanged.

First add opt-in contract tests: unrelated version progress, wrong owner/content
or source witness, live capacity blocking, pinned victims, replay/cancellation,
and unchanged existing cached/proactive operations. Then a qualified smoke and
complete W0 compare against retained D358/D353/D336. Test the mechanism through
explicit transaction receipts and retry counts; test its performance through
full-run joint timing SLO, P95/P99 TTFT, TPOT and physical GPU-s. No threshold,
autoscaler, concurrency or timeout retuning is bundled with this hypothesis.
No performance gain is claimed before those runs.

## Evidence and safety

- Analyzer: existing `scripts/ieee_tc/analyze_d297_full.py`, opt-in
  `--preparation-wait-output`; default historical paths unchanged.
- 61 selected CPU tests passed. No new GPU inference, remote operation or figure.
- Analysis peak 962,719,744 bytes; memory high/max/OOM and swap use zero.
- Data: `paper_results/ieee_tc/p2_backend/d360_preparation_wait/analysis1/`.
- All 8,000 request projections preserve missing-load spans and native identity
  joins; source/output hashes are retained. Protected historical data are checked
  before checkpoint publication; only explicit new paths are staged.
- Mainline remains 7B optimization → 3B → paused baselines → M1/M2 → ablations
  and sensitivities. This diagnosis does not close 7B.
