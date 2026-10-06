# D268 identity diagnostic — bounded prefix, not a repair claim

Date: 2026-10-06  
Status: diagnostic complete; excluded from formal comparison

## Frozen identities and evidence

- Raw result root: `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d268_20261006/7b_identity_diag_prefix/`
- Result SHA-256: `e34ea23a3673df7f842edc515b667461128076e6f18e7bace2a4707c095ab750`
- Final launch receipt: `launch.json`
- Launch receipt SHA-256: `d9e0614e73680fc40df9bbb8c82af8e01b1486881f10b63e832b9cfa0f7156d1`
- Configuration SHA-256: `0ad7a97d7d18861c2324aa5d1c7f6a66838c3481d18a490c5de7977a45fd3f60`
- Launcher SHA-256: `47d78e1d85e6f2e71942404089edb5ea44d32a7f2e2fd73bea119d832f1b8d59`
- Plan SHA-256: `0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7`
- Trace SHA-256: `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`
- Adapter-subset SHA-256: `aa94b21e1295aefde664e5b29c030a9e42e02af946369bfe9ae15336b89b3016`
- Delivery contract: `prepublished_gzip_v1`; request-time remote packing was absent.

## Categorical outcome

| quantity | observed |
|---|---:|
| requests in diagnostic prefix | 210 |
| terminal successes / failures | 210 / 0 |
| native fixed-length/token-ID contracts | 210 / 210 |
| scale-up / scale-down events | 10 / 7 |
| identity-conflict failures | 0 |
| service return / replay return / watchdog return | 0 / 0 / 0 |
| native GPU-context release / service-path removal | confirmed / confirmed |
| formal comparison qualified | no |

## Why this is not D266's replication

D266 replayed the full 4,000-request trace in eight 500-request phases. D268
used an externally bounded 210-request prefix while retaining
`multi_cycle_phases=8`; the runner therefore created seven 26-request phases
and one 28-request phase. The phase means rose from 2.89 s to 88.62 s TTFT,
with overall mean/P95 TTFT 36.91/88.95 s and mean/P95 dispatch-window wait
35.56/87.26 s. This is a queue/drain observation caused by the prefix-phase
combination, not an acceptable PrimeLoRA latency result and not evidence that
the D266 identity failure was repaired.

The run did cross the request-index neighborhood of D266's failure without a
repeat, but the changed prefix length and phase boundaries are a confound.
The metadata also retains `trace_role=exploratory`, `formal_run=false`,
`run_frozen_settings_sha256=""`, `system_resolved_config_sha256=""`, and
`vllm_available=false`; these provenance fields prevent a formal correctness
or performance claim.

## Interpretation and next gate

D268 establishes only that this bounded, differently partitioned prefix can
complete cleanly under the instrumented validator. The operational cleanup
receipt is complete, but the watchdog/exec evidence still records
`proves_physical_lease_release=false`; this is a strict physical-release
warning, not permission to claim allocator-level lease release. D268 does not
establish readiness, SLO, G1/G2, ranking, or causality for the D266 failure.
The next action is a read-only source and replay-semantics audit comparing
D266 and D268 state transitions; no further GPU replay is authorized until
that audit identifies a falsifiable hypothesis. The validator must not be
relaxed and the identity conflict must not be reclassified as fixed.
