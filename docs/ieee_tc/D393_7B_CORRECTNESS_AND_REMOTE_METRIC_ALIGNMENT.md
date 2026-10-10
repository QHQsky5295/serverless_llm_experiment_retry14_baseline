# D393 — 7B correctness and remote-delivery metric alignment

Date: 2026-10-10.  This is an offline evidence decision.  It does not
rewrite D379/D386, does not change the frozen metric protocol, and launches no
GPU or remote service.

## Decision in one paragraph

The frozen V1 protocol does **not** require token-for-token equality between
different inference backends.  It requires a normal terminal state, the fixed
generation contract, the correct request/adapter identity, and complete native
measurement.  The later D391 proposal treated exact equality with an
independent HF/PEFT execution as the only possible full-workload correctness
route.  That is a stricter cross-backend reproducibility test, not the V1
definition, and it conflicts with V1's explicit cross-backend rule.  D379's
offline audit therefore closes the V1 *binding/accounting* evidence for all
4,000 requests, while leaving D379 a development run because it was not a
formal frozen execution and the common warm/Resident references are absent.
No historical result is relabelled as a formal G1/G2 result.

## Frozen rules that control this decision

The authoritative protocol is
`primelora_tc_metrics_v1` (`METRIC_PROTOCOL_FROZEN_V1.md`, SHA-256
`5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22`).  Its
correctness predicate is request-level: normal terminal state, target output
count, correct request/adapter identity, and complete native accounting.  It
also says that independent backends are not required to produce identical
token sequences; same-backend changes still require an audit.

The protocol does not permit any of the following shortcuts:

* treating a successful HTTP response or a matching length as adapter
  correctness without source and lease evidence;
* counting a missing or fallback adapter as correct because the adapter name
  was present;
* treating HF/PEFT output as a performance reference, or using a cross-kernel
  near-tie as a latency/SLO failure;
* deleting divergent rows, changing the denominator, or adding a tolerance
  after seeing the result.

The independent HF/PEFT witness remains valuable diagnostic evidence.  D385
found 92/100 exact sequences and eight first divergences at top-2 margins of
0 or 0.015625.  D295 found same-vLLM cross-launch variation.  These results
show that exact sequence equality is not a stable cross-backend oracle on the
current RTX 3090/vLLM setup.  The vLLM 0.30 reproducibility documentation also
states that reproducibility is not guaranteed by default and is limited to the
same hardware and vLLM version; online execution can only use its batch
invariance mode for this purpose:
<https://docs.vllm.ai/en/v0.30.0/usage/reproducibility/>.

## D379 full-workload audit

Input result:

`/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d379_20261010/7b_capacity_revalidation_full_w0/experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_full_d379_7b_capacity_revalidation_full_w0.json`

Result SHA-256: `772ae04ec266e1b0e69c7a1075b7c8bf8a93eb2fc322a2f8ff356a99bc9874d6`.

The audit checked every request, not a prefix or a sampled subset:

| V1 evidence field | Count | Interpretation |
|---|---:|---|
| fixed-length output contract (`completion_tokens = requested = min(source_expected, 256)`) | 4,000/4,000 | target work was completed without shortening the denominator |
| native terminal observation and native output count | 4,000/4,000 | native generation and accounting are present |
| native token source (`vllm_token_ids`) | 4,000/4,000 | text re-tokenization did not replace native counting |
| confirmed dispatch snapshot | 4,000/4,000 | selected-replica state was captured before dispatch |
| request ID to adapter ID binding | 4,000/4,000 | request identity was retained through the path |
| acquisition lease and adapter integer identity | 4,000/4,000 | the selected native reference was acquired or the source path was recorded |
| expected content SHA copied into the acquisition receipt | 4,000/4,000 | content identity was not inferred from a directory name |
| source-admission content SHA agrees with the acquisition expectation | 4,000/4,000 | file/HOST/NVMe/remote source identity is coherent |
| released native request/reference receipt | 4,000/4,000 | every request closed its native reference |
| prepublished remote transfer verification for remote rows | 132/132 remote rows | published object, content verification, and no request-time packing |

The run contains 1,994 GPU, 1,032 HOST, 841 NVMe and 133 remote source
selections (the audit's remote verification count is 132 because one row has a
legacy missing timing subrecord; that row remains retained and is not silently
repaired).  This difference is a diagnostic field-quality issue, not a reason
to erase a request or claim zero remote transfers.

The content audit also remains part of the interpretation.  The existing 7B
pool contains 498 all-zero A/B adapters and two non-zero content classes; the
3B pool is all-zero.  A zero LoRA update is mathematically a valid zero update,
but it cannot establish that 500 logical IDs are 500 independently trained
adapters.  We therefore describe the pool as a structural adapter workload,
retain all source/transport/cache costs, and do not make a trained-adapter
quality claim.  This limitation is separate from whether the serving path
used the requested source rather than silently falling back to the base model.

### What this audit does and does not certify

It supports assigning the V1 request-level correctness indicator only when the
same fields are present in a **new formal run**.  It does not promote the old
D379 record, because its launch receipt says `formal_run=false`, its resource
and common-reference gates were not frozen, and its result used development
configuration metadata.  It also does not claim cross-backend semantic
identity, trained-adapter diversity, or a G1/G2 ranking.

The next formal run may use the existing native identity chain and a bounded
independent numerical witness as diagnostics.  It must not require exact
HF/PEFT sequence equality as an unannounced replacement for V1.  A gross
source/adapter mismatch, missing lease, missing native measurement, fallback,
or incorrect target length remains a correctness failure.

## Remote delivery and latency accounting — final rule

The detailed scope is frozen in
`METRIC_PROTOCOL_REMOTE_SCOPE_FROZEN_V2.md` (`primelora_tc_remote_scope_v2`,
SHA-256 `485a6b7cf9fd34db5fdc87959ebbe9f757054cd7aa26a6a3efea210d91615a29`).
The classification is:

| Event | Counts in user TTFT/E2E? | Reason |
|---|---:|---|
| request-triggered remote open/read, HTTP response, receive, local write, extraction, and materialization | yes | this is work caused by serving the request |
| cache miss, eviction, tier promotion, handoff, and contention selected by a system | yes | these are the mechanisms under comparison |
| one-time prepublished gzip creation, archive hashing, manifest generation, and bulk scan before deployment | no | deployment administration outside the serving workload |
| remote service start/stop/restart, health polling, unrelated cleanup or logging | no | external orchestration; never subtracted after a run |
| an administrator task overlapping a formal run | no correction | mark `external_interference`, retain raw data, exclude or repeat the execution key |

The user-clock definitions remain

\[
TTFT_r=f_r-a_r,\qquad E2E_r=c_r-a_r.
\]

The remote transfer is not removed merely because the link is shared.  Prime's
fewer misses and lower transfer concurrency are legitimate outcomes of its
GPU/HOST/NVMe residency policy.  All systems receive the same immutable
published object and content manifest, but they are **not** forced to perform
the same number of misses, fetches, or bytes.  Equalizing those quantities
would remove the mechanism being evaluated.  The transfer count, logical bytes,
wire bytes, cache tier, and client/server spans are reported separately so a
latency improvement cannot be hidden behind a cache label.

The approved D78/D80 one-time gzip cache is shared deployment state, not a
Prime-only optimization.  It contains about 1.47 GiB across both models and
has already passed 500/500 adapter functional coverage for each model with zero
request-time packing.  The old dynamic-packing records remain delivery
diagnostics and are never corrected by subtracting their packing time from
E2E.

## Link evidence and external-validity statement

The historical D77 snapshot observed 100 Mb/s/full because the remote peer
advertised only 10/100.  It did not establish that the NIC itself was limited
to 100 Mb/s; the peer, switch port, cable, or autonegotiation path was not
isolated.  A fresh read-only check on 2026-10-10 found the actual experiment
route `192.168.4.178 (eno1np0) ↔ 192.168.4.174 (eno1)` at 1000 Mb/s/full on
both endpoints; the artifact service was stopped and port 18081 was closed.
The immediate cause of the old downshift remains unknown.  No 100/400-Gb/s or
H100 conclusion is made.

Consequently, request-triggered transfer remains in the measured latency, and
the paper reports the link as a testbed stratum plus bandwidth sensitivity.  A
100-Mb/s historical stratum may magnify the value of residency, but deleting
transfer would destroy the causal question.  The defensible external-validity
statement is: results establish behavior on the measured 1-Gb/s/full path and
the registered application-layer rate points; they do not establish absolute
performance on a modern 100/400-Gb/s fabric.  This is a limitation to state,
not a reason to mix administrative pack time into the serving metric.

## Gate and next action

The D379 audit closes no performance goal.  It changes the next action from a
strict, cross-backend exact-sequence replay to:

1. retain the native identity evidence and bounded HF/PEFT result as
   diagnostics;
2. run the common warm reference under the final 7B backend/resource envelope;
3. run the three Resident reference blocks needed for `U_ref` and the G2
   budget; and
4. run a new formal 7B Prime Full replay with the published remote contract,
   complete request-level identity fields, and the frozen G1/G2 calculations.

No 3B, baseline, ablation, sensitivity, ranking, or figure is opened by this
offline decision.  If the new formal run has a source/adapter or output
contract failure, it remains a qualification failure; it is not repaired by
discarding rows or by changing the metric.

## Provenance

* Plan SHA-256: `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`
* D379 result SHA-256: `772ae04ec266e1b0e69c7a1075b7c8bf8a93eb2fc322a2f8ff356a99bc9874d6`
* D385 HF/PEFT witness: `D385_PEFT_SEQUENCE_REFERENCE_RESULT.md`
* D295 same-backend repeat: `D295_CONTROLLED_NUMERICAL_REPEAT.md`
* Artifact content limitation: `ARTIFACT_CONTENT_AUDIT.md`
* Remote delivery scope: `METRIC_PROTOCOL_REMOTE_SCOPE_FROZEN_V2.md`
