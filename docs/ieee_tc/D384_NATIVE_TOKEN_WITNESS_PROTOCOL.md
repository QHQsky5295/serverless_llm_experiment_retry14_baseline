# D384 — pre-registered 7B native-token witness

Date: 2026-10-10.  This diagnostic is registered before execution and is
not a performance, SLO, G1, G2, or baseline-comparison run.

## Why this run is admissible

D379 completed the fixed-length native generation contract for all 4,000
requests, but its result stores only one native completion-token hash per
request.  A hash is sufficient for the length and transport audit; it cannot
establish the frozen exact-sequence `n_correct` contract.  Repeating the old
22-request probe would not add coverage, because D295 already shows that the
same near-tie behavior is nondeterministic across launches.

D384 therefore uses the first 100 records of the already frozen W0 trace.  The
prefix is selected by request order before observing D384 outcomes; no request
is added, removed, reordered, or regenerated.  The prefix contains both D170
input-length groups and all four materialized `(adapter_model.safetensors,
adapter_config.json)` content classes:

| witness property | prefix-100 count |
|---|---:|
| requests | 100 |
| short input (`expected_input_tokens <= 616`) | 37 |
| long input (`expected_input_tokens > 616`) | 63 |
| content class 1 | 47 |
| content class 2 | 24 |
| content class 3 | 16 |
| content class 4 | 13 |

The source trace SHA, adapter-content-index SHA, canonical prompt binding and
request IDs are recorded in the run receipt.  The serving configuration keeps
D379's vLLM backend, C=8 runtime envelope, confirmed routing, handoff policy,
profiles, adapter pool and remote delivery path; the only evidence changes
are the diagnostic prefix and `ieee_token_id_capture=true`.

## Comparison contract

Every completed request must contain native output token IDs, their SHA-256,
the canonical prompt hash, the fixed-length greedy contract, and the exact
request/adapter binding.  `backend-peft-sequence-reference` then evaluates the
same frozen prompt and adapter binding using the independently qualified PEFT
reference.  A request is a witness match only when the native token-ID
sequence is exactly equal to the reference sequence.  A length match, text
similarity, or top-2 probability proximity is not promoted to correctness.

The diagnostic reports, without threshold laundering:

* request-binding and content-class coverage;
* native token capture completeness and terminal-contract completeness;
* exact sequence matches and first-divergence positions;
* top-2 margin classification only as a numerical warning;
* per-length-group and per-content-class counts.

It does not set `n_correct` for D379 or any later formal run.  Even a clean
100-request witness is bounded evidence that the four content classes and two
length groups are executable under the native contract; it is not a proof of
4,000/4,000 exact correctness.  A divergence remains a failure of the exact
contract, regardless of whether it is later explained as an FP16 near tie.

## Stop and interpretation rules

The run is invalid if the prefix identity, adapter-content index, model/backend
identity, generation contract, or prompt binding differs from the frozen
inputs.  It is also invalid if token IDs are missing, if the remote service is
left running after cleanup, or if the physical GPU/memory/disk gates fail.

If all 100 requests match exactly, the result supports a larger pre-registered
coverage witness but still leaves full-workload `n_correct` open.  If any
request diverges, the result is retained as a correctness boundary and no
formal Prime ranking is started.  In either case, no 3B run, baseline run,
ablation, or figure is launched from D384 alone.

The one-time remote gzip delivery cache remains outside request E2E.  Any
request-triggered remote read, HTTP transfer, receive, extraction, or
materialization remains part of the measured request path; no administrative
packing time is subtracted.
