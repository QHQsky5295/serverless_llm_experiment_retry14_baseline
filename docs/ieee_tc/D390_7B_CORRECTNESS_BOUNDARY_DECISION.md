# D390 — 7B correctness-boundary offline decision

Date: 2026-10-10.  This is a read-only protocol decision; it launches no GPU
or remote service and does not change any result or correctness definition.

## Decision

The numerical adapter-correctness gate cannot be closed from the existing
evidence.  `n_correct` remains unknown for D379, D386 and the earlier Full
replays.  No historical run may be promoted to G1/G2, and no baseline or 3B
comparison may use a correctness-normalized resource ranking yet.

This is not evidence that PrimeLoRA is numerically wrong.  It is the result of
the frozen protocol refusing to infer semantic correctness from a weaker
contract.  The strict ruler is retained unchanged.

## Evidence that is sufficient for narrower claims

| evidence | result | what it establishes |
|---|---:|---|
| D384 native-token witness | 100/100 fixed-length native contracts | request, prompt, target-length, and native-token accounting |
| D385 independent HF/PEFT witness | 92/100 exact; 8/8 native tokens in reference top-2 | bounded checkpoint/adapter/input consistency and near-tie sensitivity |
| D295 same-backend controlled repeats | cross-launch exact pairs 22/27, 24/27, 23/27 | output instability exists under the tested vLLM execution conditions |
| D379/D386 full replay | 4,000/4,000 terminal/native-contract records | complete serving and timing/resource contract |

None of these rows proves that every generated sequence is the intended
adapter computation.  D385's near ties cannot be converted into exact matches
by tolerance, and D295's repeated outputs cannot be selected by a favorable
launch.  D248 explicitly prevents imposing cross-backend token identity as a
different, stronger comparison rule, but that does not create a positive
semantic oracle for the full replay.

## Why no offline promotion is lawful

The frozen metric protocol requires `N_correct` for G1/G2.  A native token
count, adapter tensor SHA, request/adapter binding, and successful terminal
state are necessary conditions, not a numerical reference.  The only direct
independent sequence witness is bounded and has strict mismatches.  The
pre-registered same-backend repeat is complete and explicitly prohibits
additional screening repeats.  Therefore setting `N_correct=4,000` now would
change the denominator by assertion rather than evidence.

## Consequence for the main line

The Prime-first hold remains in force.  No new controller candidate, 3B run,
external baseline, ablation, sensitivity experiment, ranking, or figure is
authorized from this offline decision.  The common warm and Resident reference
are also still pending: D170 is a provisional same-process warm measurement,
and D213's Resident repeats violate the remote-delivery, common-notice,
resource-domain, physical-lifecycle, and timeout contracts.  They must not be
used to freeze a budget or final SLO threshold.

The next permitted action is a separately reviewed qualification proposal that
can satisfy the existing correctness ruler without relaxing it.  Until such a
proposal is accepted and executed, the existing Prime measurements remain
development diagnostics only.

## Provenance

- D384: `docs/ieee_tc/D384_NATIVE_TOKEN_WITNESS_PROTOCOL.md`
- D385: `docs/ieee_tc/D385_PEFT_SEQUENCE_REFERENCE_RESULT.md`
- D295: `docs/ieee_tc/D295_CONTROLLED_NUMERICAL_REPEAT.md`
- D248: `docs/ieee_tc/D248_CROSS_BACKEND_SEQUENCE_RECONCILIATION.md`
- D379: `docs/ieee_tc/D379_FULL_REPLAY_RESULT.md`
- D386: `docs/ieee_tc/D386_FULL_REPLAY_RESULT.md`
- Frozen ruler: `docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md`

