# D220 — 7B correctness and reference evidence inventory

Status: offline inventory complete, no GPU or remote operation. This record
prevents repeating already completed diagnostics and keeps the formal gate
conservative.

## Evidence already available

| Evidence | Scope | What it establishes | What it does not establish |
|---|---|---|---|
| D166/D167 | 44 slot/content snapshots, four content classes | checkpoint bytes and registered slot contents agree | per-request execution, all 500 IDs, numerical Full correctness |
| D207 | 22 real requests, 3,917 nonempty native iterations | intended adapter/owner is bound to observed forward/logits rows | all 4,000 requests, all 500 IDs, arithmetic equivalence of a Full replay |
| D208 | 512 pre-registered coordinate/operator cases, normal and CUDA-graph paths | installed LoRA shrink/expand operators use the selected tensor contents | arbitrary model inputs, scheduler/routing correctness, Full `N_correct` |
| D63 native + independent PEFT | one canonical prompt, nonzero A/B, zero and explicit base controls | nonzero weights have an observable numerical effect; tensor loading and the independent reference are byte-consistent | a general error bound, 500-ID coverage, concurrent Full correctness |
| D170 | 1,536 GPU-ready vLLM measurements, batch 8 | input-conditioned warm TTFT/TPOT candidates and viable batch geometry | frozen cross-model reference, Resident lifecycle budget, adapter numerical identity |
| D219 | 4,000 fixed-output Prime requests | terminal/native token contracts, request timing, route evidence and physical lifecycle | numerical adapter correctness (`N_correct` is unknown), joint SLO, admission/full-mechanism qualification |

The 7B audit pool contains 500 inspected adapters and four distinct weight
classes: two nonzero singleton adapters (`finance_lora`, `medical_lora`), one
zero-update class covering the remaining rank-8 adapters, and one rank-16
nonzero class. The existing D63/D208 controls cover the four content classes;
this is coverage of tensor classes, not proof that every request in D219 used
the intended physical slot.

## Formal interpretation

No new numerical GPU run is justified solely to repeat D63, D207 or D208. The
existing evidence is a layered implementation audit, not a semantic oracle for
the 4,000-request replay. In particular, output-token length, output hashes,
logical adapter IDs and native execution counts cannot be promoted to
`N_correct`. D219 therefore remains descriptive and `n_correct=null`.

The remaining qualification gap is not “whether LoRA arithmetic exists”; that
is covered for the audited tensor classes. It is the absence of a complete,
request-bound semantic correctness oracle under the exact Full replay and the
absence of a formally frozen Resident resource reference. Those are separate
gates and must not be conflated.

## Next permitted evidence

1. Reuse D170's measured warm candidates as provisional inputs only; do not
   rewrite them as a frozen cross-model reference.
2. Build/qualify the protocol-compliant Resident reference with the same
   workload, common notice, resource-domain separation, per-GPU lifecycle
   ledger and 1,800-second qualification protection. No performance ranking is
   allowed until the reference is complete.
3. If a new correctness experiment is proposed, it must add a falsifiable
   request-level semantic oracle or full-pool execution binding; repeating a
   single prompt, token-count contract, or coordinate arithmetic is not a new
   hypothesis.

## Provenance hashes

- `D166_NATIVE_SLOT_CONTENT_RUN.md`: `f327cef114b215c84d4d7a3308a485b50c0d7a827f8f96f6c339b99f227e5c05`
- `D167_CHECKPOINT_SLOT_CONTENT.md`: `5111ff4b65ca668dc4da8d1388b4e739639bf9b97cca35c4dfaa88cca75c277d`
- `D207_NATIVE_EXECUTION_PREFIX.md`: `6934aa43e44a204ef92b1b3b42d0be7de92d15dde2fe36a2775c8569415c4867`
- `D208_LORA_COORDINATE_ARITHMETIC.md`: `e1b92d06c65c35322e24d2e51118cc2717e6a5aa85e0fbe8165ed724604057fd`
- `NATIVE_ADAPTER_NUMERIC_CONTROL.md`: `074d994aab7dbba80e2db4a4fa21fb096b666fe03b33fb14a1957e266a4d1cd5`
- `20260927_7b_peft_reference.json`: `ffa2e575a16dd313c2227b9deceeefd2dc95e635375f185a8513dfadc4f9e1aa`
- `20261002_d170_warm_measurement.json`: `0ebc7587effe293dcb986d33fb932c100cbcb3384c56ab2c0b0bec73b2bdd88c`
- `20261003_d208_coordinate_arithmetic.json`: `bca88d6cdd8555632175f891f8b176428247be49a4c65c63b15782b1a1434451`
