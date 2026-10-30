# D981: 3B native/PEFT numerical witness audit (2026-10-31)

## Purpose and boundary

D981 is a bounded correctness diagnostic for the existing 3B PrimeLoRA request path. It does not replace the D972 4,000-request development performance result, does not enter the formal M1/M2 matrix, and does not establish a 4,000-request `N_correct` denominator. The native witness and the independent reference use the same frozen 3B trace prefix, canonical prompt renderer, target lengths, base checkpoint, and 500-adapter pool.

The distinction is intentional:

- D972 proves complete 4,000-request execution, native timing, and fixed-output contract integrity, but stores only completion-token hashes.
- D980 reruns the first 100 requests with opt-in native token-ID capture.
- D981 compares those captured native token IDs against an independent offline HF/PEFT reference.

No performance value from D980 is substituted for D972, and no bounded witness is promoted to full-pool semantic qualification.

## Inputs and provenance

- Native witness: `d980_20261031/3b_native_token_witness_100`
- Reference audit: `d981_20261031/3b_peft_sequence_reference_100_from_d980`
- Model: local Llama-3.2-3B-Instruct checkpoint used by D972
- Trace: the existing 3B W2 shared trace, first 100 entries, source SHA-256 `4ea5d026da3820301e753ad6b03ea776e25a5c3f01921933bd124598eb26018d`
- Adapter pool: existing 500-adapter `llama32_3b_a500_v1_modelscope`
- Generation contract: `fixed_length_greedy_v1`, native vLLM token IDs, no EOS termination
- Native witness: 100/100 requests completed, 100/100 output contracts matched, 100/100 token-ID captures present
- Reference loaded 29 adapters appearing in the 100-request prefix

## Native witness integrity

D980 completed all 100 offered requests without failure. Every request has:

- `success=true`;
- `output_contract_match=true`;
- requested completion tokens equal observed completion tokens;
- native TPOT timing;
- `completion_token_ids_capture_contract=fixed_length_greedy_v1_native_ids_opt_in_v1`;
- a native completion-token-ID SHA-256 and the corresponding integer sequence.

The token capture is opt-in and isolated to D980; ordinary performance results retain only the digest, avoiding a schema and storage change to D972 or future Full runs.

## Independent sequence comparison

The PEFT reference uses the same prompt identity and target map for all 100 requests. It loaded the local base model and each adapter with `autocast_adapter_dtype=false`, generated exactly the requested number of tokens with a portable no-EOS setting, and compared integer token sequences rather than text re-tokenization.

| Outcome | Count |
|---|---:|
| Exact native/reference sequence matches | 95 / 100 |
| Length mismatches | 0 |
| Sequence mismatches | 5 |
| Native token in reference top-2 at first divergence | 5 / 5 |

The five mismatches occur at positions 4, 18, 39, and 42. Two finance requests have an HF top-1/top-2 score difference of 0; the remaining divergences have a small margin (0.0078125 or 0.015625 in the recorded FP16 reference scores). Thus the observed bounded mismatch pattern is consistent with close FP16 backend/reference tie-breaking, not with a missing adapter or a length/route failure. This is an interpretation of the recorded top-2 evidence, not a claim that all sequences are semantically interchangeable.

## Correctness conclusion

D980+D981 establish a strong bounded numerical witness for the 3B path: all sampled requests execute the intended adapter and fixed-output contract, and 95% match the independent PEFT sequence exactly. The five near-tie mismatches remain visible and are not relabeled as correct.

The evidence is insufficient to set `n_correct=4000`, to claim full-pool semantic qualification, or to admit D972 to a formal G1/G2 ranking. Until a broader correctness protocol is approved and completed, the proper status is:

`contract_complete = true`, `bounded_numerical_witness = 95/100 exact`, `full_semantic_qualification = unknown`.

## Audit implementation fixes used by this diagnostic

The preflight reference audit was corrected in append-only commits to (i) snapshot fast-tokenizer-only checkpoints such as Llama-3.2 without requiring an absent `tokenizer.model`, (ii) use the native prompt decode cleanup contract, (iii) record length mismatches without indexing past the shorter sequence, and (iv) use a portable no-EOS fixed-length reference generation. These are audit correctness fixes; they do not alter PrimeLoRA serving behavior or any D972 performance measurement.

