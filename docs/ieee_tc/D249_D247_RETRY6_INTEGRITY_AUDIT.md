# D249 — Fresh integrity audit of D247 retry6

Date: 2026-10-06

## Decision

The fresh audit confirms that D247 retry6 is an internally consistent bounded
cross-backend sequence diagnostic, but it is not a semantic qualification pass.
The raw result remains immutable and is not admitted to any performance,
correctness, SLO, G1/G2, or baseline denominator.

## Evidence

| item | value |
|---|---|
| result | `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d247_retry6_20261006/7b_peft_sequence_reference.json` |
| result SHA-256 | `db24af3e2c60dde77c783ceef0aba09da4af772ad549990d0b3098b6bfda0949` |
| measurement | complete, 22/22 requests |
| exact sequence match | 19/22 |
| strict pass | false |
| launch receipt | `53be573932cd0f9f609809d0ed81dbddf0516ac44505ab25a6268d20660efa21` |
| cleanup | GPU context released; service path removed; no open owned lease |

The three divergences occur at positions 27, 37, and 12. At each first
divergence the native token is in the independent HF/PEFT top-two set; the
top-one/top-two FP16 margins are 0.015625, 0, and 0.015625. This supports a
near-tie numerical-path diagnostic only. It does not justify a token tolerance,
success-subset filtering, or rewriting the strict result as a pass.

## Retry4 quarantine

The same audit found a stale semantic inconsistency in the earlier retry4
artifact: it records 17/22 exact matches while reporting `pass=true`, contrary
to the current sequence predicate and to the retry5/retry6 schema. Retry4 is
quarantined and cannot be used as a successful reference. The raw file and its
launch evidence remain preserved for provenance; no historical result is
rewritten.

## Scope boundary

D247 retry6 establishes only: fixed-output transport completed, prompt/target
identity checks completed, adapter tensor checks completed for the registered
diagnostic, and three cross-kernel near-tie divergences were observed. It does
not establish full-pool correctness, 4,000-request correctness, physical
admission, common SLO, lifecycle efficiency, or a comparison win.

The next admissible work remains the pre-registered 7B physical/admission/
readiness bounded diagnostics. External baselines and 3B remain paused.
