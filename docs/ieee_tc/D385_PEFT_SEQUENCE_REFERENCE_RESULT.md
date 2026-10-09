# D385 — bounded HF/PEFT sequence reference result

Date: 2026-10-10. This is a correctness-boundary diagnostic for the existing
7B native-token witness. It is not a serving-performance run, does not create
`N_correct`, and is not eligible for G1/G2, ranking, or a figure.

## Frozen inputs and execution boundary

- Source native witness: D384, the first 100 requests of the frozen W0 trace.
- Same 7B backbone, tokenizer, adapter-root, prompt renderer, target-token
  contract, and adapter bytes as D384.
- Independent reference: offline Hugging Face/PEFT FP16 model on one qualified
  RTX 3090, eager attention, `use_cache=True`, sequential greedy generation.
- 100 requests; 29 distinct adapters; all four materialized weight/config
  content classes and both input-length groups are represented.
- No request trace, adapter pool, or model weight was regenerated or modified.

The complete machine-readable result is
`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d385_20261010/7b_peft_sequence_reference_100_retry1.json`.
The gated launch receipt is the corresponding
`7b_peft_sequence_reference_100_retry1_launch.json`.

## Observed result

| quantity | value |
|---|---:|
| completed reference requests | 100/100 |
| exact native/reference sequences | 92/100 |
| first-divergence sequences | 8/100 |
| target-length contract | 100/100 |
| adapter tensor identity checks | all loaded adapters passed |
| native token present in reference top-2 at first divergence | 8/8 |
| first-divergence top-1 minus top-2 margin | 0 or 0.015625 for all 8 |
| service-scope OOM/max/high events | 0 |
| GPU context release and service-scope removal | confirmed |

The launch receipt has `service_returncode=2` and classifies the diagnostic as
`qualification_command_failure` because the strict exact-sequence predicate is
false. This is an expected protocol outcome for an exact witness that records
near ties; it is not an infrastructure crash. The host/user cgroup contains a
pre-existing historical OOM counter, but no D385 service-scope OOM or OOM-kill
event occurred, so that counter is not attributed to D385.

## Interpretation

The strict witness predicate is intentionally not relaxed: D385 therefore does
not pass its exact-sequence gate. However, every mismatch is numerically local
to a top-2 near tie in the independent reference, with margin 0 or 0.015625,
and no missing adapter, tensor-identity failure, prompt mismatch, length
mismatch, or resource fault. This supports the narrower conclusion that the
observed differences are consistent with FP16/reduction-order/backend
numerical sensitivity. It does not prove that all 4,000 requests are
semantically correct, and it does not justify replacing the frozen strict
correctness ruler with a tolerance or a success-only denominator.

The production PrimeLoRA result remains unqualified until the plan's full
correctness/common-reference evidence is available. D385 closes the bounded
diagnostic branch; it does not authorize 3B, baseline, G1/G2, or figure work.
The next action must be a documented 7B decision on the strict full-workload
correctness boundary and, if necessary, a separately registered stability
candidate. `VLLM_BATCH_INVARIANT=1` is not silently enabled by this result: it
is a different execution mode whose overhead and full-stack behavior would
need an independent qualification.

## Provenance

- D385 launch-receipt SHA-256:
  `6c6350f9a6eed5580ee210b685dd9de230cb333a4835ccb95e18af23a893e258`.
- Plan SHA-256 at launch:
  `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`.
- The failed first launcher attempt remains separately retained and is not
  combined with this completed retry.
