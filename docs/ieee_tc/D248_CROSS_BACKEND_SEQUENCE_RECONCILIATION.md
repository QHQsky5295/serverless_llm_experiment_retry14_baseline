# D248 — cross-backend sequence reconciliation

## Decision

D247 retry6 is a completed bounded diagnostic, not a 22/22 exact-sequence pass:

- native D246 result: 22/22 fixed-length requests completed with non-negative
  native token IDs and `actual_tokens == target_tokens`;
- D247 retry6: all 22 prompt hashes, prompt-token-ID hashes, target maps and
  adapter tensor keys/dtypes/values matched the frozen inputs;
- independent HF/PEFT sequence equality: 19/22;
- the three differences are `req_00009` (finance), `req_00013` (finance), and
  `req_00018` (ecommerce);
- at each first difference, the native token is in the HF top-2; the HF top-1
  minus top-2 margins are respectively `0.015625`, `0.0`, and `0.015625` in
  FP16 score units;
- retry6 launch returned a nonzero diagnostic status because the strict
  sequence predicate is false, while the service, watchdog, GPU-context and
  scope cleanup receipts all pass.

The raw result remains `pass=false` and is not rewritten.  The three near-tie
records are not discarded or counted as exact matches.

## Why this does not invalidate the native task contract

The authoritative frozen metric protocol explicitly states that cross-backend
executions are not required to produce identical token-by-token text, while
same-backend changes must be audited.  The native path is vLLM with
`FLASH_ATTN` and FlashInfer sampling; the independent witness is
HuggingFace/PEFT with eager attention.  The witness therefore validates
checkpoint/adapter/input identity and reveals whether a disagreement is a
gross semantic mismatch, but it cannot impose a cross-kernel exact-token rule
that is stricter than the frozen comparison contract.

The evidence supports the following bounded statement: for the 22-request
diagnostic, the native service honored the fixed-length output contract for all
requests, all 12 used adapter directories were independently tensor-verified,
all prompt/target identities matched, and 19 sequences were exactly equal;
the remaining three are consistent with FP16 near ties in the independent
witness; native-side logits and tie-breaking were not captured, so the cause is
not proven to be numerical alone.
This is a semantic/input-contract witness, not a 4,000-request correctness
denominator and not a performance or ranking point.

## Claim boundary and next gate

The strict sequence result stays in the diagnostic appendix and cannot be
reported as 100% cross-backend agreement.  Conversely, it is not valid to
label the three near ties as adapter-replica mismatch or as a PrimeLoRA
correctness failure without a same-backend counterfactual.  The next 7B gate
may proceed to the already planned bounded physical/admission/readiness
diagnostic, followed by a protected Full qualification replay.  That replay
still requires 4,000/4,000 native requests, correct adapter identity,
joint-SLO accounting, complete dispatch fields, physical allocation/release
sidecars, and the frozen G1/G2 rules.  No 3B, external baseline, or comparison
claim is authorized by D248.

## Immutable evidence

| Item | SHA-256 |
|---|---|
| D247 retry6 result | `db24af3e2c60dde77c783ceef0aba09da4af772ad549990d0b3098b6bfda0949` |
| D247 retry6 launch receipt | `53be573932cd0f9f609809d0ed81dbddf0516ac44505ab25a6268d20660efa21` |
| D246 native result | `0c4ff53a029113531e5b8604ea33edd0ff1d765772ccf8bc15fd354344dcc076` |
| retry6 configuration | `e0719a9d788060a95d1b18d51798fb66d698a55564051d9df786bd3b1f6ad09a` |
| retry6 launcher | `3eeaed9313f7e876895bb299211ae1bc786c028279b1291cccb42ad93c3e7f73` |
| preflight source at retry6 | `803832d6217cbc62f92eb8da93c259f6281154a8acb5adb0861ffd85d0846ed4` |

The frozen trace SHA remains
`efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`.
