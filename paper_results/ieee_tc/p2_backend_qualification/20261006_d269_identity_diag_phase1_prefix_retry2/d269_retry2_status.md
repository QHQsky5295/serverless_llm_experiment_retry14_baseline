# D269 retry2: phase-preserving identity diagnostic

Date: 2026-10-06
Status: complete diagnostic; excluded from formal comparison, readiness, G1/G2, and ranking

## Purpose

D266 stopped at `req_00198` in the first 500-request phase with
`native integer ID reused for a different adapter source`. D269 retry2 was a
fresh, phase-preserving diagnostic: the same frozen 7B trace prefix of 500
requests was replayed as one phase so that request 198 was not moved across
the seven phase boundaries used by D268. The structured validator from
`3153655`/`6b6dcbe` was active. No validator relaxation, retry, fallback, or
changed routing objective was introduced.

## Frozen identities and raw evidence

| item | value |
|---|---|
| raw result root | `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d269_20261006/7b_identity_diag_phase1_prefix_retry2/` |
| result JSON | `experiment_results_full_vllm_dedicated_a500_r4000_c2_tc_ieee_identity_diag_phase1_prefix_retry2_d269_7b_identity_diag_phase1_prefix_retry2.json` |
| result SHA-256 | `82ae8cc73d1a085fee37163378d302ce4950df80ea7aa3dd7fb36a7cb45eed5d` |
| launch receipt SHA-256 | `437d8448147c5dd7ac6f5cd801929b4d99db32cb531c60314e9a130c05f4200e` |
| config SHA-256 | `baf54c741e9d49012a99a1db60fa9690d7d95fae4633521f0d2e94c2146a06bd` |
| launcher SHA-256 | `2aedb7b9af728bb277fcc0175d94e68801329fb15b1c5389948de4554a18f80a` |
| plan SHA-256 | `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7` |
| result schema | `e2e_v3`, exploratory diagnostic |
| trace | same 7B W0 source trace as D266; 500-request view, source count 4000 |
| delivery | `prepublished_gzip_v1`; request-time remote packing absent |

The launch receipt reports `pass=true`, service/replay/watchdog return code
zero, native GPU-context release confirmed, and service-path removal. It also
reports `production_launch_authorized=false`; this is intentional for this
diagnostic. The watchdog sampled the service cgroup without `high`, `max`,
`oom`, or `oom_kill` events. Final physical GPU release is operationally
clean, but the allocator field remains `proves_physical_lease_release=false`;
this is a release-evidence warning, not a failure of the request run.

## Categorical outcome

| quantity | observed |
|---|---:|
| requested diagnostic prefix | 500 |
| arrived / terminal | 500 / 500 |
| success / failure | 500 / 0 |
| fixed-length/native token contract | 500 / 500 |
| request 198 (`education_lora_0007`) | terminal success, target 9 tokens |
| natural scale-up / scale-down | 3 / 0 |
| terminal source-binding conflict | 0 observed |
| cgroup high/max/OOM/kill | 0 / 0 / 0 / 0 |
| first-service observations | 3 (below readiness minimum 20) |

`req_00198` has the same source-request SHA as D266's failed request position
(`4517901c4dc45baa9743da9140157213c90e3ba4174db79ce3b0c191da89d8fc`) and
completed successfully. The retained `main_outcome.json` also records a
verified `education_lora_0007` NVMe source, immutable content digest,
preparation plan, and completed native host preparation. This establishes
that the adapter/source path was exercised somewhere in the run, but does
not map that preparation to request 198 or expose the exact native
transition: the terminal schema has no adapter ID, source path, retry, or
conflict counter.

## Descriptive measurements (not ranking evidence)

The run used real RTX 3090/vLLM LoRA workers despite the analyzer metadata
field `vllm_available=false`; the provenance mismatch is retained rather than
silently corrected. Descriptive values are:

| metric | value |
|---|---:|
| mean / P95 TTFT from scheduled arrival | 1205.5 / 2802.7 ms |
| mean / P95 service TTFT | 692.5 / 1882.7 ms |
| mean / P95 E2E | 6302.1 / 12317.7 ms |
| mean TPOT | 37.36 ms |
| TTFT SLO attainment (`<=5000 ms`) | 99.6% |
| physical GPU-seconds | 968.009 |
| GPU-s per completed request | 1.936 |
| remote delivery mode | prepublished gzip, no request-time pack |

These values describe a 500-request exploratory prefix and must not be
compared with 4000-request formal runs. The analyzer's generic “headline”
block is not promoted to a paper claim.

## Interpretation and next gate

D269 retry2 removes D268's phase-partition confound and weakens the hypothesis
that the conflict is inevitable at this request position. It does **not** prove
that the validator repair is correct or that the original D266 state was
recreated: no structured rejection occurred, no direct adapter/source witness
was recorded, and native residency counters are not an independent identity
oracle. The fail-closed validator therefore remains unchanged.

The next admissible action is a read-only source/state comparison of D266 and
D269 (including selected source, content digest, live-copy and preparation-plan
state). A new heavy readiness replay is not authorized by this diagnostic
alone. Until that comparison and the D265/D266 readiness aggregation are
resolved, Prime has no formal readiness, G1/G2, or baseline-ranking result.

No performance figure is generated for this diagnostic; the appropriate
artifact is this categorical status table and its provenance manifest.
