# IEEE TC execution status

## NEXT GATE — D278 post-repair 7B readiness diagnostic registered; launch held

2026-10-06. After the CPU red/green reproducer and the 489 focused tests
passed, D278 is registered as one fresh, same-configuration cold diagnostic.
It changes only the D276 run/result/cache identity and tests whether the
lease-to-plan receipt repair prevents the observed capacity-wait failure.  The
existing seed-42 W0 trace, 500-adapter subset, fixed-output contract, vLLM
environment, remote prepublished delivery and all resource/control profiles
are frozen.  The config and launcher are
`configs/ieee_tc/20261006_d278_7b_readiness_8phase_full_retry9.yaml` and
`scripts/ieee_tc/run_d278_7b_readiness_8phase_full_retry9.sh`; their hashes and
acceptance boundary are sealed in `docs/ieee_tc/D278_7B_READINESS_POST_D277.md`.

No GPU replay has started in this registration step.  The final preflight must
repeat memory/disk/process/GPU cleanup, verify the existing remote service
without repacking, and preserve the exploratory/non-ranking identity.

## CURRENT — D277 identifies and repairs the proactive-lease witness schema defect

2026-10-06. D276 showed that a selected-adapter capacity failure could leave an
empty `gpu_reference_evidence` object because the exception escaped before the
inner RequestResult envelope.  The new bounded
`native_capacity_conflict_v1` diagnostic copies only the final blocker rows,
lease IDs, external-pin state, preparation-plan IDs, owner/epoch and request
identity onto the exception.  It does not retry, evict, relax fail-closed
checks, or infer a release.  Full native snapshots and cache contents remain
excluded.

The red/green CPU reproducer used a real proactive preparation lease and a
request-driven acquisition against the same native capacity.  Without the
plan-task binding it fails closed as before; with the exact lease-to-plan
binding it waits for the acknowledged preparation release and then succeeds.
The worker now emits `preparation_leases=[{lease_id,plan_id}]`, and the
controller accepts it only with one live engine-scoped plan task.  Unknown,
duplicate or mismatched identities remain rejected.  This is the minimal
source-level repair suggested by D276's observed failure path; it does not
change routing, eviction, admission thresholds or retry policy.

The request-lifecycle suite passes 185/185, and the GPU-reference plus
transfer-pressure suites pass 304/304 after the schema change (489 tests
across the three suites).  This is not yet a
readiness or performance result.  After the post-fix suites pass, one new
same-configuration cold readiness replay may test whether the D276 failure
reappears and whether the first-service sample gate can be met.
The protected-artifact seal check remains false only for the pre-existing
change to the sealed baseline file
`serverless_llm_baselines/scripts/replay_openai_trace.py`; the file was not
edited or staged here.

## CURRENT — D276 7B readiness replicate closes with one localized-but-unproven capacity-wait failure

2026-10-06. The registered same-configuration cold replicate completed its
4,000-request replay envelope after D275, with 4,000 terminal records. 3,999
requests succeeded and one (`req_00862`, `translate_lora_0030`) failed before
dispatch with `RuntimeError: native capacity conflict has no acknowledged
release path`. The retained traceback reaches `_wait_native_reference_capacity`,
but the failed terminal has an empty `gpu_reference_evidence` object. Thus the
artifact identifies where the controller rejected the wait, not which
candidate lease blocked it; it does not prove that D275 caused or fixed the
failure and does not authorize a blind replay.

Among dispatched records, pre-dispatch tier coverage is 3,999/3,999 and tier
invariant conflicts are zero. The run exercised ten natural scale-ups, seven
scale-downs, and ten first-service observations, so it misses the
pre-registered first-service minimum of twenty. The physical sidecar is
complete (15,878.133222 GPU-s, no open leases, `n_correct=null`) and cleanup
is complete; service/replay/watchdog return code is zero, native GPU-context
release is confirmed, and the owned service path was removed. No service or
auxiliary OOM/oom-kill occurred; service memory peaked at 21,792,223,232 bytes
and host `MemAvailable` remained 102.91--122.41 GiB.

D276 is consequently diagnostic-only and is excluded from readiness, SLO,
lifecycle, G1/G2 and baseline denominators. Its compact status table is
`paper_results/ieee_tc/p2_backend_qualification/20261006_d276_readiness_retry8/d276_status.csv`;
the full claim boundary and artifact hashes are in
`docs/ieee_tc/D276_7B_READINESS_RECHECK.md`. The next falsifiable action is
CPU-only: exercise a proactive GPU-preparation lease and a request-driven GPU
acquisition against the same native capacity, retain the exact candidate and
release receipts, and test whether preparation leases are represented in the
acknowledged-release witness set. No new full replay is authorized until that
contract is either rejected or repaired with focused tests.

## CURRENT — D275 stale selected GPU copy now re-routes without relaxing identity checks

2026-10-06. A new CPU-only, falsifiable contract test isolated a race class
that was not distinguished by D266's incomplete witness.  It selected a
confirmed GPU copy, retired it, installed the same logical adapter and
verified immutable content from a new managed path, and then submitted the old
selection.  Before the change, the request raised the native source-conflict
exception at `_validate_source_binding`; this red observation was captured
before modifying production behavior.

`IEEEBackendGPUReferences._load_and_acquire` now returns the existing typed
`required_source_changed` result when (and only when) an observed source ID is
stale while the current live copy proves the same logical adapter name and
content digest.  The controller's existing whole-request re-resolution path
therefore remains authoritative.  Future-epoch and lease checks remain strict;
wrong logical identity, wrong content, staged-source mismatch, and callers
without a verified digest still reach the unchanged fail-closed validator.  No
native cache, pin, lease, or epoch is mutated by the typed conflict.

The focused native-reference suite passes **135/135**; the request-lifecycle,
pending-GPU-pipeline, and transfer-pressure suites pass **365/365** under the
bounded CPU scope.  Evidence and the compact table are in
`docs/ieee_tc/D275_STALE_GPU_SELECTION_RECHECK.md` and
`paper_results/ieee_tc/p2_backend_qualification/20261006_d275_stale_selection_recheck/d275_cpu_contract.csv`.
This closes a demonstrated stale-selection contract bug class, but does not
reconstruct D266 (which lacks old/requested/live source and plan witnesses),
and is not a GPU readiness, performance, SLO, G1/G2, or baseline result.  The
next gate is rechecking the existing numerical/common-warm/physical-release
requirements before any new 7B GPU readiness attempt; no replay is authorized
solely by D275.

Checkpoint `8cff52a` contains the implementation, test, and compact table; it
was pushed to `faaslora_origin/retry14_continuous_queue_v2`.  The pre-existing
dirty manifest and unrelated untracked review/figure artifacts were not staged.

## CURRENT — D274 3B historical bottleneck is profile-bound; no replay authorized

2026-10-06. A CPU-only audit compared the retained exploratory D138
Llama-3.2-3B projection with the fully resolved current 3B profile. D138's
12,588.6-ms mean TTFT is not a model-compute measurement: mean native vLLM
TTFT is 581.1 ms, while dispatch/admission, ingress queue, slot and
pre-runtime spans account for most of the user-visible delay. However, D138
used concurrency 8, dedicated `ieee_confirmed` coordination, explicit measured
service/preparation profiles and no-delay storage; the current profile resolves
to concurrency 4, automatic `adapter_affinity` coordination, predictive
scale-up and an effective-capacity gate. The historical value therefore
cannot be used to justify a current control-path rewrite or a formal 3B claim.

The audit and resolved-profile digest are sealed in
`docs/ieee_tc/D274_3B_HISTORICAL_BOTTLENECK_AUDIT.md` and
`paper_results/ieee_tc/p2_backend/20261006_d274_3b_profile_audit.json`.
They register falsifiable hypothesis H3B-1: a fresh current-profile prefix,
when authorized, must distinguish residual dispatch/readiness/slot delay from
native inference before any 3B full replay. This CPU-only step does not
authorize a GPU replay, 3B run, baseline, SLO, G1/G2 point or ranking. The
active gate remains the 7B numerical/common-warm/physical-release evidence;
the user-visible 3B 12.6-s figure remains historical and unqualified.

## CURRENT — D273 source-plan boundary audited; no validator change retained

2026-10-06. A CPU-only falsification check examined whether the verified
same-content path-migration shortcut could bypass an active preparation plan.
The direct validator call accepts such a source only when no plan identity is
carried; this is the request-side input that the controller is designed to use
to supersede a stale plan and advance the source epoch. The complete
transfer-pressure suite then passed 169/169 tests, including staging,
native-HOST, handoff, movement, and native-commit supersession boundaries; the
focused GPU-reference suite passed 135/135. The exploratory source edit was
fully reverted and no production validator change remains.

The evidence therefore distinguishes an unplanned demand that must trigger
plan supersession from an operation explicitly owned by a frozen preparation
plan. D273 does not identify D266's missing old/requested source state and does
not authorize a GPU replay, readiness point, G1/G2 point, or baseline
comparison. The audit is retained at
`paper_results/ieee_tc/p2_backend_qualification/20261006_d273_source_plan_supersession_audit/d273_status.md`.

## CURRENT — D272 separates source-path migration from content identity

2026-10-06. A CPU-only contract qualification inspected the unchanged native
source validator after the D271 partial diagnostic. Six focused source/identity
tests passed, covering live-copy relabelling, verified same-content migration
after complete retirement, preparation-plan ownership, host guards, and the
distinct CPU/GPU source views. The retained evidence is
`paper_results/ieee_tc/p2_backend_qualification/20261006_d272_source_identity_contract/d272_status.md`.

This establishes the implementation's intended boundary: a new path is legal
only for the same logical adapter after all old native copies and preparation
plans retire and the immutable content SHA agrees; a live copy, active plan,
different logical name, or different content remains fail-closed. It does not
identify which predicate fired in D266, whose pre-instrumentation record lacks
the old/requested source, content, live-state, and preparation-plan fields.
Therefore D266 remains unresolved, no validator relaxation or retry is
justified, and D272 authorizes no GPU replay, readiness point, G1/G2 point, or
baseline comparison.

## CURRENT — D271 does not reproduce D266 and remains a partial diagnostic

2026-10-06. D271 replayed the registered 7B eight-phase diagnostic with the
bounded source-binding witness enabled. It observed 231 arrivals and 224
successful terminations before a deliberate SIGINT after the historical D266
position `req_00198` had completed. The native integer-ID reuse marker,
structured source-binding conflict, and retained failure observation were all
absent in the observed prefix. This rules out a claim of an immediate
deterministic failure at that position under this replay state, but it does not
reproduce D266, identify its old/requested/live source transition, or prove a
repair.

The run is recorded in
`paper_results/ieee_tc/p2_backend_qualification/20261007_d271_identity_diag_8phase/d271_status.md`.
It is classified as a manual-stop partial diagnostic with
`pass=false` and `production_launch_authorized=false`; the raw evidence is
retained and excluded from readiness, performance, SLO, G1/G2, and all ranking
denominators. The owned service path was removed and cleanup reported
`released=true`, but the terminal census did not prove allocator-level/native
context release. No OOM, cgroup high/max/oom/oom_kill event, remote mutation, or
historical-result overwrite occurred. The intermediate latency values are not
formal measurements.

D266 therefore remains unresolved. The fail-closed validator is unchanged, and
no further heavy identity replay is authorized without a new falsifiable
hypothesis and a witness that links old/requested/live source, content identity,
and preparation-plan state at the native rejection. The active execution gate
is the complete common warm reference plus physical whole-GPU allocation and
release evidence for Prime; baseline comparison and G1/G2 ranking remain
blocked until that gate is met.

## CURRENT — HydraServe/remote metric boundary audit frozen as a non-mutating addendum

2026-10-06. The authoritative metric calculation remains
[`METRIC_PROTOCOL_FINAL_V2.md`](METRIC_PROTOCOL_FINAL_V2.md), with the frozen V1 and
remote-scope V2 parents unchanged. A read-only audit of the HydraServe NSDI'26 paper,
the official repository, and the D77/D78 link records is retained in
[`METRIC_PROTOCOL_HYDRASERVE_REMOTE_AUDIT_20261006.md`](METRIC_PROTOCOL_HYDRASERVE_REMOTE_AUDIT_20261006.md)
(SHA-256 `ea5b82c0ddfdfcb4ac736f8fe76862a976cccb9ad081a4ec26eae4efe7d022ac`). It
confirms that HydraServe's published primary evidence is cold-start TTFT,
TTFT/TPOT SLO attainment, and a resource/cost trade-off—not CE—and that its
memory-time cost model is not a substitute for this project's physical GPU-s
lifecycle metric. The addendum also freezes the classification of the approved
one-time read-only gzip delivery cache as deployment-time work, while all
request-triggered remote read/send/receive/materialization remains in user E2E.
D77's 100-Mbps negotiation and D78/current 1-Gbps/full negotiation remain separate
execution states; no historical 100-Mbps coverage run is promoted to a performance
or G1/G2 point. This documentation step authorizes no new GPU replay.
The read-only 2026-10-06 state (both endpoints 1000/full, remote boot ID retained,
no service or link mutation) is recorded at
[`20261006_d270_link_readonly.json`](../../paper_results/ieee_tc/remote_qualification/20261006_d270_link_readonly.json)
(SHA-256 `c03f299d2a3c58c44cf4e9384058d3c3731dd2eb0850f2211c0ef4523aca8240`).

## CURRENT — request-level source-binding witness is instrumented and unit-tested

2026-10-06. The controller now retains a bounded, request-local diagnostic when
the unchanged native rejection marker `native integer ID reused for a different
adapter source` escapes selected-source preparation. The evidence binds the
offered request ID and adapter to the selected instance, submitted native
intent, selected source identity, owner/epoch and the original bounded worker
error text. Full routing snapshots are deliberately excluded to avoid
duplicating the adapter pool in every failure record. The exception is still
re-raised unchanged; there is no retry, fallback, validator relaxation, route
change, or ownership decision based on this field. The normal reservation
cleanup and unresolved-native-work handling therefore remain authoritative.

The same evidence is emitted in the request failure observation when the
exception is collected outside the inner execution path, closing the evidence
gap seen in D266 without retrospectively altering D266. A focused regression
test verifies that a synthetic D266-shaped error keeps request/source identity
and drops unrelated full-pool metadata; the complete request-lifecycle and
native-reference suites pass: **317 passed, 19 warnings**. This is an
instrumentation qualification, not a readiness, performance, G1/G2, or repair
result. A future bounded replay may use it only to capture the exact rejection
predicate; no replay is authorized solely because this instrumentation exists.

## CURRENT — D269 retry2 completes a phase-preserving identity diagnostic; no repair claim

2026-10-06. D269 retry2 replayed a fresh 500-request 7B prefix as **one
phase**, preserving the phase structure relevant to D266's `req_00198` rather
than repartitioning the prefix into D268's eight mini-phases. The structured
source-binding diagnostics were active, with no retry, fallback, validator
relaxation, or changed routing objective. The raw result and audit are stored
at `paper_results/ieee_tc/p2_backend_qualification/20261006_d269_identity_diag_phase1_prefix_retry2/`.

All 500/500 requests terminated successfully (500 successes, zero failures),
including request 198 with the same source-request SHA as the D266 failure
position. The run observed three natural scale-ups and no scale-downs. No
terminal source-binding exception appears in the service or retained worker
logs. The launch receipt reports zero service/replay/watchdog return codes,
native context release and service-path removal; cgroup `high`, `max`, `oom`,
and `oom_kill` counts stayed zero. The physical-release proof field remains
`proves_physical_lease_release=false`, so the allocator-level release claim is
not promoted.

This is stronger than D268 for replay-state control, but it still does **not**
prove that the D266 identity transition was reproduced or repaired. The
terminal schema has no adapter ID/source path, retry, or transient-conflict
counter. The retained `main_outcome.json` does verify an
`education_lora_0007` NVMe source, content digest, preparation plan, and
completed native host preparation, but it does not link that preparation to
request 198. The analyzer also reports `vllm_available=false` although worker
logs show real vLLM 0.30.0/LoRA execution. Consequently the result is
exploratory, not a readiness, performance, G1/G2, or ranking point. Descriptive mean/P95
TTFT is 1205.5/2802.7 ms and mean/P95 E2E is 6302.1/12317.7 ms for the 500
request prefix; these numbers are not compared with formal runs.

The initial D269 attempt remains separately classified as a
`protocol_or_launcher_error` caused by a copied `source_audit_sha256` typo;
its GPU request count was zero and its receipt is retained in
`paper_results/ieee_tc/p2_backend_qualification/20261006_d269_identity_diag_phase1_prefix/`.

The validator remains fail-closed and unchanged. The next gate is a
read-only comparison of D266 and D269 source-selection, content-digest,
live-copy, and preparation-plan state. D269 does not authorize a new full
readiness replay or any formal Prime optimization claim.

The read-only comparison is now complete. D269's native outcome contains a
verified `education_lora_0007` NVMe source and content digest, one deferred
`stale_snapshot` preparation, and a later completed preparation from the same
source. D266 has no structured source witness at its rejection, and neither
run maps the native preparation record to request 198. Thus the comparison
confirms that D269 exercised the adapter path and that ordinary stale-snapshot
revalidation can complete, but it cannot identify D266's rejection predicate
or establish a repair. The validator remains unchanged; no new heavy replay
is authorized until a future witness can capture old/requested source,
content, live-copy, and preparation-plan state at the exact rejection. The
side-by-side evidence and the controlled-difference table are retained in
`paper_results/ieee_tc/p2_backend_qualification/20261006_d269_identity_diag_phase1_prefix_retry2/d266_d269_source_state_comparison.md`.

## CURRENT — D266 identity conflict unresolved; D267 configuration failure; D268 prefix diagnostic is non-causal

2026-10-06. The authorized D266 fresh cold rerun did not produce a formal
result. At request `req_00198` (227 arrivals observed before manual stop), the
native worker rejected adapter `education_lora_0007` with
`ValueError: native integer ID reused for a different adapter source` during
`_validate_source_binding`, called from `demand_load_and_acquire`. The
terminal journal records the request as failed with no instance, prompt hash,
or completion tokens. The worker traceback and all raw journals remain at
`results/ieee_tc/p2_backend_qualification/d266_20261006/7b_readiness_8phase_full_retry7/`.
The run is incomplete and is excluded from readiness, performance, SLO,
G1/G2, and ranking denominators. Watchdog samples show no OOM/high event and
the exact owned scopes were cleaned; the manual interruption means D266 has
no final successful launch receipt suitable for a formal release claim.

The failure is not an integer-ID hash collision: the frozen adapter IDs are
unique. D257 had an analogous source-binding failure for a different adapter,
and D265 succeeded for the same `education_lora_0007` request position. This
supports a path/source-state hypothesis, but the old and requested live
bindings at the D266 failure are not yet jointly captured. The strict live
relabeling validator therefore remains unchanged; no retry, fallback, or
relaxed source policy is justified.

The post-D269 non-GPU source-state contract check selected 31 source/identity
tests from `tests/test_ieee_tc_gpu_references.py`; all 31 passed (103 tests
deselected). It confirms the intended semantics: a live native copy cannot be
relabelled to another path, while an old retired copy can be reloaded from a
new tier when the immutable content identity agrees. This is a code-contract
check, not a reconstruction of D266's missing old/requested source witness.

D267 was a fresh configuration-only attempt. Its service profile rejected a
hand-copied `source_audit_sha256` identity before model startup, with no GPU
inference. The failure and its receipt are retained in
`paper_results/ieee_tc/p2_backend_qualification/20261006_d267_identity_diag_prefix/d267_status.md`.

D268 then corrected the configuration and ran a fresh 210-request bounded
prefix. It completed 210/210 requests with no terminal identity-conflict
failure, ten scale-ups, seven scale-downs, clean native-context release, and
owned service-path removal. There is no direct native identity-conflict
counter in the terminal schema, so this is an indirect non-recurrence
observation. The strict cleanup field
`proves_physical_lease_release=false` remains a physical-release warning.
This is not a repair result: the prefix retained eight
multi-cycle phases, so it used seven 26-request phases and one 28-request
phase rather than D266's eight 500-request phases. Its later phases reached
88.62 s mean TTFT and 87.26 s P95 dispatch-window wait, which is a queue/drain
effect of this diagnostic partition and not an acceptable final performance
point. D268 metadata remains exploratory (`formal_run=false`), with empty
frozen-settings/source-config hashes and `vllm_available=false`; its full
evidence and caveats are summarized in
`paper_results/ieee_tc/p2_backend_qualification/20261006_d268_identity_diag_prefix/d268_status.md`.

The current decision is therefore **no repair claim and no new GPU replay**.
The next gate is a source-level comparison of D266 and D268 replay/phase
semantics and a controlled, falsifiable identity-state diagnostic. Only after
that audit identifies the changed state needed to reproduce or rule out the
conflict may another heavy readiness run be authorized. The readiness gate,
G1/G2, and the user's latency target remain open.

## CURRENT — D265 7B eight-cycle Full readiness diagnostic completes; one gate remains open

2026-10-06. The fresh 7B `d265_7b_readiness_8phase_full_retry7` diagnostic
completed its pre-registered 8-cycle workload without changing the frozen
trace, 500-adapter subset, model, fixed-output contract, vLLM environment,
profiles, resource envelope, or read-only prepublished remote delivery. The
raw result is retained at
`results/ieee_tc/p2_backend_qualification/d265_20261006/7b_readiness_8phase_full_retry7/`.
Its result SHA256 is
`545da871f734a0a7b6dbf25fa97af66465a6cdc9db33ca67182d535c0ca1256a`, launch
receipt SHA256 is
`faf08552233d7869926e562e32f3ba2dc5b44860cbb7b73d9f56453100e8c834`, config
SHA256 is
`95717b4b1a6a53c0f0f1184f552672c909bcc91f6bbc5db40a8eec3dd84c754a`, and
launcher SHA256 is
`d24565ae4e02d56fb56c0c6a0a18c040c729d5f9bb54aac5b4a73dcfd77d6027`.

The categorical execution contract is complete: 4,000/4,000 requests
arrived and terminated successfully, with zero failures, 4,000/4,000 native
fixed-length/token-ID contracts, 4,000/4,000 canonical prompt hashes and
native timing observations, 100% pre-dispatch tier coverage, and zero tier
invariant conflicts. The tier partition is GPU 1,190, HOST 1,684, NVMe 994,
and Remote 132; each tier's readiness booleans are internally consistent.
The run exercised eight 500-request phases, ten natural scale-up events and
seven natural scale-down events. The terminal journal contains no nonempty
`failure_observation`, and the service, replay, watchdog, physical allocation
journals, GPU-context release and owned service-path removal all closed
successfully. Physical owner-union accounting reports 15,883.368 GPU-s
(3.970842 GPU-s per offered request), with no open lease; this is the canonical
physical field and must not be replaced by the legacy lifecycle estimate.

The readiness diagnostic gate is **not yet passed** solely because the
pre-registered first-service sample minimum is 20 per model/variant. D265 has
10 natural scale-out first-service observations (all ten are dispatch-time
Remote/cold and none has a planned-adapter match). The existing analyzer
therefore reports `passed=false`, `rerun_required=true`, while all other gate
components pass. The protocol explicitly allows one additional identically
configured cold run; D266 is authorized only to supply the missing independent
first-service observations, then the two runs will be aggregated without
pooling requests as independent repetitions.

The result remains a **diagnostic**, not a formal PrimeLoRA comparison or
G1/G2 point: metadata intentionally marks `trace_role=exploratory` and
`formal_run=false`, `n_correct=null`, and the frozen-settings/source commit
fields are empty. The metadata also reports `vllm_available=false` despite
the native vLLM token-timing path; this provenance inconsistency is retained
as an audit caveat rather than silently repaired. The descriptive overall
TTFT/E2E values (2,159.8/6,510.8 ms mean; 7,581.6/14,457.7 ms P95) are not
used as ranking evidence. A readiness summary table and diagnostic figures
are stored in
`paper_results/ieee_tc/p2_backend_qualification/20261006_d265_readiness_retry7/`
and `figs/ieee_tc/p2_backend_qualification/20261006_d265_readiness_retry7/`;
the first-service figure is correctly omitted because n=10 is below the
published minimum. The request-local routing hardening has no recurrence in
this run, but D265 does not prove that it caused the earlier D258 exception.

No remote configuration, packing, cache regeneration, or unrelated workload
was performed during measurement. The remote records report the approved
`prepublished_gzip_v1` delivery with `remote_pack_performed=false`; request-
triggered transfer remains part of the application timeline, while one-time
publication and unrelated remote work remain outside it under the metric
protocol. D265 is therefore closed as a successful execution diagnostic with
one sample-count gate outstanding, not as evidence that PrimeLoRA has already
met its final optimization target.

## CURRENT — D258 retry6 stopped after an unlocalized controller exception; routing hardening is diagnostic

2026-10-06. The fresh 7B eight-phase Full diagnostic
`d258_7b_readiness_8phase_full_retry6` was intentionally interrupted after
approximately 74 arrivals. Scale-out reached four instances and host/GPU
memory remained healthy, but one request terminated before reservation binding
with `AttributeError: 'NoneType' object has no attribute 'replica_id'`. The
terminal record has no instance, dispatch snapshot, prompt hash, or completion
tokens. No native-RPC failure, worker transport error, or OOM was observed.

This run is an incomplete diagnostic and is excluded from every readiness,
performance, SLO, G1/G2, and ranking denominator. The manual interruption means
its final release proof is incomplete; the raw terminal journal, service log,
watchdog trace, and launch receipt remain at the unique retry6 result root.
Post-cleanup GPU/process checks are clear.

The exact controller source line is not established by this artifact. The
request was observed before reservation binding, but the collector retained
only the exception type/message and the enclosing run was manually
interrupted. In particular, the old selection call and the subsequent read of
`last_ieee_decision` had no intervening `await`; the shared diagnostic field is
therefore a plausible robustness hazard, not a demonstrated cause of this
failure. It must not be reported as the root cause.

The repair keeps the immutable decision returned by the selection call in the
request-local execution context and retains `last_ieee_decision` only for
diagnostics and legacy tests. This is defensible request-local semantic
hardening, not a causal explanation, and it adds no retry, fallback, altered
SLO, or changed routing objective. Focused routing, pending-GPU, and
request-lifecycle tests pass; a fresh 100-request bounded smoke also completed
with 100/100 terminal successes and complete dispatch-tier observations. That
smoke is still diagnostic only. Before another 4,000-request readiness run,
the collector now preserves a bounded Python traceback for controller task
exceptions so any recurrence can be localized from the immutable raw result.

## CURRENT — D260 bounded RPC reproducer confirms a repairable combined-reference schema defect

2026-10-06. The fresh 7B eight-request bounded reproducer
`d260_7b_rpc_prefix8` completed with 8 arrivals and 8 terminal records (7
successes, 1 failure). It is a diagnostic only and is excluded from all
performance, SLO, G1/G2, and ranking denominators. The launch receipt passed
service, watchdog, replay, cleanup, GPU-context release, and service-path
removal checks; the failed request and all worker logs remain at the fresh
result root
`results/ieee_tc/p2_backend_qualification/d260_20261006/7b_rpc_prefix8`.

The new worker-side error instrumentation identifies the failure before native
GPU mutation: the controller supplied a combined GPU reference containing the
immutable `expected_content_sha256` field introduced by D258, while
`InferenceEngine.ieee_register_pending_gpu_reference` still required the old
eight-field set. The worker therefore raised
`ValueError: combined registration requires an exact selected GPU reference`,
the parent correctly treated the reply as ownership-uncertain, and the worker
was retired. vLLM initialization completed normally and no OOM or transport
failure was observed. This is a data-flow contract mismatch, not evidence of
an algorithmic readiness or capacity failure.

The minimal repair accepts exactly the legacy reference or that reference plus
one validated lower-case SHA-256 content identity; all other fields and invalid
digests remain rejected before registration. The repair is covered by the
pending-GPU pipeline and native-RPC ownership tests. D260 config and launcher
identities are respectively
`077bb73938e778b5dbad38bcdb33098a0cb53e36276a710a133a726a6b317ade` and
`a9075a8710a3eb5fccbb1a48d4dcb9a6a0c8f386f42739da`.

No new 4,000-request GPU replay is authorized until this repair is committed
and a fresh bounded smoke confirms that the digest-bearing combined path reaches
the native owner and returns an acknowledged receipt without uncertainty.

## CURRENT — D261 post-repair bounded smoke passes the native combined path

2026-10-06. The first D261 launch was a gate-only rejection because its fresh
configuration omitted the required preparation-profile `ewma_beta`; it created
no request result and passed GPU-context and scope cleanup. That evidence is
retained at `d261_20261006/7b_rpc_prefix8` and is excluded from all denominators.

The corrected retry used a new result/cache identity
`d261_20261006/7b_rpc_prefix8_retry2` with the same frozen trace, adapter
subset, model, vLLM environment, and remote delivery contract. All 8/8
requests completed successfully, including four requests routed with a
confirmed GPU source and the digest-bearing combined pending/admission path;
all eight native intents carried `expected_content_sha256`. The result contains
zero `subprocess_native_rpc_failed_no_retry` records and zero ownership-uncertain
guards. Every request's native acquisition receipt echoed the expected content
identity, and the launch receipt confirms service, watchdog, replay, cleanup,
GPU-context release, and service-path removal.

This bounded smoke is diagnostic rather than a performance result. It clears
the schema-repair gate for a fresh D258 4,000-request readiness diagnostic, but
it does not by itself establish readiness, SLO, or G1/G2 performance. The
corrected config and launcher hashes are
`85e94ff08cce8cfa2aaebf999ed122b2464047802d5631caeeb8917063342663` and
`0a5cafe0121a7fa3bf3cf28e1d00701041d6340a68e59e405978d289795e7af8`.

## CURRENT — D258 retry5 is an interrupted diagnostic; native RPC ownership failures remain unresolved

2026-10-06. The fresh 7B 8-phase Full diagnostic
`d258_7b_readiness_8phase_full_retry5` was started with the registered config and
launcher identities below, but was stopped after the failure mode became
decisive. At interruption, 177 requests had arrived, 100 had terminal records
(81 successes and 19 failures), and the run had already recorded 15
`subprocess_native_rpc_failed_no_retry` failures plus four
`native RPC ownership unresolved; new generation withheld` guards. The partial
run therefore fails the 4,000/4,000 readiness gate and is excluded from every
performance, SLO, G1/G2, and ranking denominator. Its raw logs, watchdog trace,
terminal journal, and worker-failure workdirs are retained at the fresh result
root; no old result was overwritten.

The failure is not attributable to host memory pressure or GPU OOM: watchdog
samples show zero cgroup OOM/high events, host memory remained available, and
the vLLM workers reached `engine ready` before their parent-side native RPC
channel became unresolvable. The strict no-retry behavior is intentional and
correct for safety, but the repeated empty/closed native exchange is a new
runtime defect to localize before another readiness run. The interruption also
caused the watchdog's final physical-release proof to be incomplete; current
GPU/process checks after cleanup are clear, but this partial run cannot provide
release evidence for a readiness claim.

The next action is a source-level RPC/worker-lifecycle audit using this raw
evidence (including the first failing request and its worker log), followed by a
minimal focused reproducer and tests. A new GPU replay is not authorized until
the cause is explained and the repair is test-qualified.

## CURRENT — D258 native identity/capacity repair is test-qualified; rerun still gated

2026-10-06. D257 remains excluded from formal readiness because it completed
3,993/4,000 requests and exposed two separate runtime defects: six requests
could not identify an acknowledged release owner for a proactive native
preparation lease, and one request rejected a path change for the same
immutable adapter content. The original D257 evidence is unchanged.

D258 repair code now carries the manifest `content_sha256` through demand,
proactive preparation, file-to-native-HOST preparation, the vLLM bridge, and
the worker admission payload. A path change is accepted only for the same
logical adapter, with a verified equal digest and after the old native copy is
no longer live; legacy commands without a digest retain strict path identity.
Native capacity receipts expose optional preparation `plan_id` witnesses, and
the controller waits only for the matching same-engine preparation task. A
failed or cancelled task is not treated as capacity. No polling, forced
eviction, retry of an unresolved native RPC, or external-pin release was
introduced.

Validation completed in the qualified vLLM 0.30.0 environment:

- 484 native GPU-reference, request-lifecycle, and transfer-pressure tests;
- 177 scheduler-observation and service-event tests;
- 289 basic smoke tests and 19 external-replay contract tests;
- compile and `git diff --check` checks.

The added tests cover digest-preserving path relocation after retirement,
rejection of a mismatched digest, and worker-side digest publication. No GPU
replay was started after this repair. The next gate is a fresh D258 7B
8-phase diagnostic with the existing 4,000-request contract; it must satisfy
4,000/4,000 terminal success, complete dispatch tiers, zero tier conflicts,
at least eight natural scale-ups and at least 20 true first-dispatch samples
before any readiness claim. D257's seven failures remain visible and are not
reclassified.

The fresh diagnostic is registered as
`d258_7b_readiness_8phase_full_retry5` with config SHA-256
`fbb61e70edf83c89d1c272374a54c08e87b1ea04e3378e9eea9cc05cf50988b8` and
launcher SHA-256
`f561b5980f3aff4c8d8fd713b9b0f1da87d553dfe087873026307a8647d54349`.
Its result, NVMe, and `/dev/shm` roots are fresh and its fields match D257
after identity substitution; no old result directory is reused.

## CURRENT — D249 fresh audit closes retry6 provenance and quarantines stale retry4

2026-10-06. A fresh read-only integrity audit of D247 retry6 confirms an
internally consistent bounded diagnostic: 22/22 requests completed, 19/22
native/HF sequences match exactly, and the three remaining first divergences
are HF top-two near ties (margins 0.015625, 0, and 0.015625). The strict result
therefore remains `pass=false`; no tolerance, post-hoc rewrite, or successful
subset enters the correctness denominator. Cleanup, GPU-context release, and
owned-service removal are confirmed.

The overall D247 evidence set is `FAIL` for integrity because retry4 predates
the strict predicate fix and contains a contradictory `pass=true` field.
Retry6 alone is an internally consistent `WARN` diagnostic; it does not become
a semantic pass through the fresh audit.

The audit also found that the older retry4 artifact reports 17/22 exact matches
but `pass=true`, which contradicts the current strict predicate. Retry4 is now
explicitly quarantined as stale contradictory evidence. Its raw files remain
unchanged and are retained only for provenance. D249 permits no semantic,
performance, SLO, G1/G2, or baseline claim from D247; the next admissible step
is the already registered 7B physical/admission/readiness bounded diagnostic.
Details and immutable hashes are in
`docs/ieee_tc/D249_D247_RETRY6_INTEGRITY_AUDIT.md`.

## NEXT GATE — D247 independent 7B PEFT sequence reference prepared; no GPU run yet

2026-10-06（D246 后的独立语义门槛）. D246 已经证明固定输出的原生 token-ID
序列可以完整穿过真实 worker、JSON/RPC、controller 和 terminal sidecar，但其
`n_correct` 仍为 null，不能直接作为正确性或正式资格证据。D247 注册一个只读的
22-request 原始 checkpoint + PEFT 对照：复用 D246 的结果、W0 冻结 trace、同一
canonical renderer、现有 7B checkpoint 和 500-adapter 池；不生成新权重、工件、负载
或服务结果，也不进入性能分母。

D247 在单张可见 GPU 上以离线 HuggingFace/PEFT、FP16、eager attention、同一
`fixed_length_greedy_v1` 目标长度和关闭 EOS 的语义逐请求生成，并在生成前校验
prompt hash、prompt token-ID hash、target map、adapter safetensors keys/dtype/value。
验收要求 22/22 完成且 native token-ID 序列逐元素完全一致；任何不一致都保留首个
分歧位置并停止，不使用文本重分词、容差或成功子集筛选。D236 的首位置参考仅作
辅助证据，不能替代该序列级门槛。

本条目前仅登记协议、脚本和静态检查，尚未启动 GPU。启动前仍需重新执行内存、磁盘、
GPU 空闲、进程清理和结果路径唯一性检查；若 D247 通过，下一步才是按计划恢复受保护的
7B Full/资格路径；3B、外部 baseline、G1/G2 和比较图仍暂停。详细协议见
`docs/ieee_tc/D247_PEFT_SEQUENCE_REFERENCE.md`。

两次启动尝试都在 GPU 加载前被门控拒绝，且没有创建结果或 launch receipt：第一次是
CUDA/PEFT 环境的 Python 3.12.12 缺少 `signal.pidfd_send_signal`，第二次虽换用系统
Python，但没有从预期的共享 auxiliary cgroup 启动。两者都不是请求失败或序列不一致，
空路径也没有计入实验。脚本现在同时固定两项启动契约：由系统 Python 承担外层
watchdog，并通过 4-GiB systemd auxiliary scope 启动它；CUDA 环境仅作为受限子进程
执行 PEFT。资源门、输入、checkpoint、adapter、生成语义和输出路径均不变，修正后需
重新通过同一启动门运行一次。

随后一次启动已通过内存/磁盘预检，但因缺少显式 NVML binding/SHA 在 GPU 加载前被
拒绝；该 launch receipt 作为 `d247_20261006` 的失败门控证据保留，不进入分母。检查
确认 11 个旧 `primelora-tc-aux-*` scope 均 `TasksCurrent=0`，随后只停止这些空 scope
以释放 systemd/inotify 资源；没有触碰任何有任务的 scope。下一次使用新的
`d247_retry3_20261006` 身份，并显式固定已验证的 NVML 路径和 SHA；若通过，才可进入
独立 PEFT 序列比较。

D247 retry3 随后通过了 containment、NVML、watchdog 和清理门，并生成了唯一结果，
但子进程在加载模型前报 `ModuleNotFoundError: scripts`；结果为
`measurement_complete=false`、请求数 0，没有 token 或 PEFT 序列，故不构成语义失败。
根因是 systemd 子进程 cwd 不受保证，而参考函数隐含依赖仓库根目录。修正已改为在
导入既有 renderer/workload plan 前显式加入仓库根目录，retry4 使用新的
`d247_retry4_20261006` 身份；retry3 结果和 launch 证据均保留。

D247 retry4 完成了独立原始 checkpoint + PEFT 生成并安全清理，但 22 条序列只有
17 条逐元素一致，`matched_reference_consistency_pass=false`；5 条差异中有两条是
HF 参考在 EOS token 处提前停止，而 native `fixed_length_greedy_v1` 明确要求
`ignore_eos`，另有少数请求在更早位置出现后端数值分歧。因此 retry4 是有效的
语义诊断、不是资格通过，也不进入任何性能分母。实现中的顶层 `pass` 已修正为只有
逐序列全匹配才为 true，避免把“测量完成”误读为“语义通过”。下一步 retry5 使用
显式空 EOS 集合和与服务一致的 KV cache，在全新身份上复核；3B、baseline、G1/G2
仍暂停。

D247 retry5 按修正后的 empty-EOS + KV-cache 语义完成并安全清理，结果为 19/22
逐元素一致，`measurement_complete=true` 但 `pass=false`。剩余 3 条均在目标长度
内生成完成，差异分别发生于 finance/ecommerce 请求的第 12–28 个位置，不再是 EOS
提前停止；这说明跨 vLLM FLASH_ATTN/FlashInfer 与 HF eager/PEFT 的 FP16 数值路径
可能在近 tie 处选择不同 token，但尚未足以把它客观归类为纯数值差异。下一步只做
一次 retry6 margin diagnostic：记录 HF 首分歧点 top-2 分数、margin 和 native token
是否位于 top-2，不改变生成或验收语义。若为近 tie，则将精确序列门降级为诊断证据并
以共同 fixed-length/adapter/prompt contract 作为资格前提；若不是，则继续停在语义
问题定位，绝不进入正式 Full/G1/G2。

D247 retry6 完成了预登记的 margin diagnostic。严格谓词仍为 19/22、顶层
`pass=false`；3 个差异位置的 native token 均在 HF top-2，FP16 top-1/top-2
margin 为 `0.015625/0/0.015625`。所有 prompt/tokenizer/target/adapter tensor
identity 检查、GPU-context release、service scope cleanup 通过。依据冻结 V1
“跨后端不要求逐 token 相同、同后端变化必须审计”的条款，D248 将严格序列结果保留为
诊断而不把近 tie 误判为 adapter 或请求语义错误；同时不把它改写成 22/22 exact pass。
D248 允许下一步进入已登记的 7B physical/admission/readiness bounded diagnostic，
随后才可进行 4,000-request Full qualification。3B、外部 baseline、G1/G2 和比较图
仍暂停。完整裁决与 immutable hashes 见
`docs/ieee_tc/D248_CROSS_BACKEND_SEQUENCE_RECONCILIATION.md`。

D249 新鲜审计随后确认 retry6 内部证据一致，但发现旧 retry4 产物把 17/22
精确匹配错误标成 `pass=true`。retry4 已隔离，不进入任何正确性分母；retry6
仍保持严格 `pass=false`。因此下一项只执行预登记的 D250 7B Full/Delayed
8-phase readiness 诊断，规格见 `docs/ieee_tc/D250_7B_READINESS_8PHASE_DIAGNOSTIC.md`。

D250 Full 的第一次启动在模型加载前被拒绝：外部 replay 校验遗留了
`multi_cycle_phases == 1` 限制，和已批准的 8-phase readiness 协议冲突。该
attempt 的 `N_arrived=0`、`service_returncode=1`，watchdog、GPU context 和
scope cleanup 均正常，不能归类为服务或性能失败。D251 记录了最小修正：保留
单一冻结 replay、单一 scenario/run、trace SHA 和 guarded receipt，仅移除错误
的 phase-count equality；修正后必须使用新的 launch identity 重试，不覆盖这次
门控失败证据。

D252 retry1 通过了外部 replay 顶层门，但在场景构造器内再次触发了相同的
`multi_cycle_phases == 1` 旧限制，仍在模型初始化前结束，`N_arrived=0`。D253
记录了第二处边界修正：构造器只拒绝 tiny-witness，继续绑定同一 guarded receipt、
trace request order、context 和正 phase count。D252 仍是独立失败证据，后续重试
必须使用新的 D254 identity。

D254 retry2 已进入真实 vLLM Full stack，但在首个 phase 收到 6 条请求后触发
`external replay cannot reset at phase/drain boundaries`，随后安全退出；这是
同一路径中第三个把单窗口 replay 假定为全 4,000 条的旧守门器。D255 记录了
最小修正：单一 ingress consumer 加连续全局 cursor，8 个窗口严格按
`(start_index,count)` 接续，默认一次性 API 保持不变；非连续窗口、第二 consumer
和不完整窗口 fail-closed。D254 的 6-request partial attempt 不进入任何分母，
下一次必须使用新 launch identity。

## CURRENT — D246 passes real-worker token transport; independent semantic gate remains

2026-10-06（D243 retry2，7B 22 请求固定输出诊断）. 在 D245 修复后，新的唯一
evidence root 完成 22/22 请求，失败 0。所有请求均有
`output_contract_match=true`、`timing_contract=ieee_tc_native_v1`、真实非负整数
token-ID 序列及匹配 digest，序列长度逐请求等于目标长度；终端 journal 也为
22/22 `success=true/native_contract_matched=true`。这证明类型修复已穿过真实
worker、JSON/RPC、controller 和 terminal sidecar，不是单元测试假象。

该运行仍不是 Full 正确性或性能结果。物理 sidecar 绑定结果 SHA，记录
`n_plan=22,n_terminal=22,n_native_contract_complete=22,n_correct=null`、无 open
lease、294.1641511830967 physical GPU-s；`n_correct` 必须等待独立原始 checkpoint
PEFT 序列比较。D236 的首位置参考不能替代这一步。watchdog 无 OOM/high/swap
事件并确认 GPU context 释放；远端 7B 服务已停止。详细证据见
`docs/ieee_tc/D246_TOKEN_CAPTURE_RETRY2.md`。3B、外部 baseline、G1/G2 和正式
Full replay 继续暂停。

## CURRENT — D245 localizes the D243 retry1 token-capture failure

2026-10-06（数值诊断失败后的进程间证据修复）. D243 retry1 completed the
protected 22-request replay and released all four allocation owners, but all
22 requests failed with `opt-in completion token capture is invalid`.  The
source-level worker contract attaches a token-ID list; the dedicated parent
proxy and the parent RPC timing annotator independently applied scalar
coercion, turning that list into a numeric fallback before controller
validation.  Focused tests reproduce this path, but the GPU-level causal
interpretation remains provisional until retry2.  The retry is excluded from
correctness, SLO, G1/G2 and ranking denominators.

D245 adds one shared validator-preserving normalizer for both parent boundaries.
It preserves only the opt-in `completion_token_ids` list after checking
non-negative integer IDs; ordinary timing fields keep their legacy scalar
normalization and the default-off digest-only schema is unchanged.  A new
encode/decode → parent-normalization → proxy-return integration test passes.
Pinned tests pass: 58 service-routing, 4 token-capture, and 25 native-timing
tests (87 combined).  Full evidence and the claim boundary are in
`docs/ieee_tc/D245_TOKEN_CAPTURE_RPC_FIX.md`.

The retry1 physical sidecar is cleanup-complete (`n_plan=22`, `n_terminal=22`,
no open leases, 318.4623162969947 observed physical GPU-s) but reports
`n_native_contract_complete=0` and `n_correct=null`; no low-cost or performance
conclusion is admissible.  Its metadata also reports `vllm_available=false`
while the service log records dedicated vLLM/GPU execution attempts; this is a
provenance warning to resolve by recording worker runtime identity in retry2,
not evidence that the retry was eligible.  The next action is a fresh guarded
retry with a new result identity, followed by the existing bounded
numerical-reference scope; 3B and all external baselines remain paused.

## CURRENT — D244 closes a profile-identity contract failure; D243 retry is held

2026-10-06（D243 首次启动失败，已完成因果审计）. D243 在第一条模型请求完成前
被冻结 service profile 校验拒绝：`ieee_token_id_capture` 是 D228 明确的纯观测
开关，但旧的 `model_identity` 将它当作运行语义字段，导致 profile/configuration
上下文不匹配。一次 replay header 已建立后，runner 安全停止；没有 request
terminal、GPU allocation、token sequence、SLO 或性能数据。远端 7B 发布服务已
停止，失败原始证据保留在唯一 D243 launch root，不进入任何分母。

该问题不通过手工改 profile SHA 或放宽校验解决。D244 对第一性原则的修正是让
`FrozenServiceProfiles.model_identity` 及 stdlib preflight identity 共同排除唯一的
诊断字段 `ieee_token_id_capture`，与已存在的 placement-only 排除保持一致；模型、
权重、采样、路由、准入、profile 文件、环境/资源/input context 和论文公式均不
变。普通 backend/model 配置变化仍 fail-closed。D244 checkpoint 的服务路由 58
项和 token-capture 3 项聚焦测试通过；下一次重试必须使用新结果身份，并重新执行全部本地主机/
GPU 门槛。详情见 `D244_OBSERVATION_FLAG_PROFILE_IDENTITY_FIX.md`。

## CURRENT — D243 7B native token-capture diagnostic prepared; GPU launch held for final gates

2026-10-06（D228 后的数值正确性诊断，尚未启动 GPU）. 已登记新的 22-request
Llama-2-7B 配置 `configs/ieee_tc/20261006_d243_7b_token_capture_22req.yaml`，其
SHA-256 为 `b57acd9c6e2e61b20535afac697be6ad81fc39767cd672262221d91cd61cb213`。
它复用冻结的 D166/D167/D207/D208 请求与内容索引、D227/D238 的模型/profile、
`fixed_length_greedy_v1` 合同和 D78/D80 已发布 gzip 工件；仅显式开启
`ieee_token_id_capture`，并使用新的结果/NVMe/HOST 根目录。没有生成权重、LoRA
工件、trace 或第二份解压池。

离线 YAML、路径和 SHA 检查通过。清理遗留离线扫描后，当前一次 preflight 观察到
`issues=[]`、`MemAvailable=121,837,080,576` 字节、推理目录剩余
`172,863,201,280` 字节，均高于未改变的 109,521,666,048 字节内存和
161,061,273,600 字节磁盘起始门槛；GPU 空闲状态仍需在启动前再次读取。远端
D233 的一次性发布 gzip 缓存、内容 SHA 和独立 artifact-disk 峰值规则已具备，
但服务 invocation、结果 sidecar 和独立 PEFT 对照仍不存在。

D243 只允许作为数值诊断：必须得到 22/22 成功的固定输出、逐请求原生非负整数
token IDs、完整 dispatch/admission/physical sidecar，并与原始 checkpoint PEFT
参考做精确序列比较；它不能进入 G1/G2、SLO、性能排名或 baseline 分母。启动前
会重新做本地内存/磁盘/GPU/cleanup 与远端服务状态检查；任一失败均保留证据并停
止，不降低阈值。详细协议见 `D243_TOKEN_CAPTURE_22_DIAGNOSTIC.md/.json`。

## CURRENT — D242 registers one readiness-publication hypothesis; host gate still closed

2026-10-06（离线瓶颈登记，无 GPU 重跑）. D235 的证据把下一处可证伪瓶颈
收敛为一个边界：新 dedicated runtime 在 engine 初始化后即可进入 confirmed
router，而 activation preparation 可能仍在执行；D219 新副本首 60 条中只有
10 条是 confirmed GPU source，最早首批 TTFT 达 11,288.8 ms。D242 将已存在的
`ieee_handoff_policy=delayed` 作为第一受控变量，Full 为配对参考，NoHandoff
只作机制边界；不复用 legacy handoff 字段，也不把非 GPU source 改称 GPU-ready。

当前 inference-host `MemAvailable` 在清理遗留离线扫描进程后约 93 GiB，仍低于
批准计划和 preflight 的 102 GiB（80+4+16+2）要求，且 swap 已有使用。因此没有
启动模型、远端服务或 replay。D242 规定内存门通过后的顺序：数值/输出 gate、
D226 显式 admission bounded diagnostic、Full/Delayed natural-scaleout 诊断，
然后才考虑新的 7B 留出块。3B 和外部 baseline 继续暂停。

## CURRENT — D241 canonical physical-resource sidecar gate implemented

2026-10-06（测量完整性实现，无 GPU 重跑）. D240 要求的物理生命周期验收已
落实：受保护运行在结果 JSON 写入且所有 allocation owner 完成清理后，生成
`physical_resource_summary.json`，其中的 `physical_gpu_seconds_total` 只来自
最终 owner-union summary，并绑定 allocation journal、terminal、summary 和
结果文件的 SHA-256。旧的 `infra_gpu_seconds_total` 与
`instance_lifecycle_log` 明确不能替代该字段。正式 Fig. 9 的
`--formal-matrix` 路径会拒绝缺少 sidecar、未完成释放、分母不完整或哈希不匹配
的记录；旧结果没有被重写。

物理生命周期单元测试 32/32 通过，Python 编译和 diff 检查通过。更大的 launch
测试中有 7 个环境阻塞（当前 Python 无 NVML、测试环境缺少可选 vLLM），不是本次
sidecar 变更引起的协议失败。D241 记录实现与验收；没有启动模型、远程 artifact
服务、baseline 或正式 replay。PrimeLoRA 7B qualification/optimization 仍是
唯一下一步活动，3B 与外部 baseline 继续暂停。

## CURRENT — D239 freezes remote latency scope, 100-Mbps interpretation, and HydraServe-aligned metrics

2026-10-06（只读指标与结果审计）. A new protocol audit,
`docs/ieee_tc/D239_REMOTE_METRIC_HYDRA_PRIME_AUDIT.md`, records the current
evidence and does not start a GPU or remote service. The current artifact route
is 1000 Mb/s/full on both observed endpoints with zero reported NIC errors. The
historical D77 100-Mbps/full state remains a separately labelled negotiation
state: its physical cause was not established, so it is neither silently
relabelled as 1-Gbps nor treated as proof of a permanent 100-Mbps device. The
formal contract keeps request-caused remote read/send/receive/write/extract and
contention in user TTFT/E2E, while one-time prepublished gzip construction,
administration, scans, cleanup, and request-external packing stay outside the
request result and are separately audited.

The metric decision is now explicit: HydraServe's warm-derived TTFT/TPOT SLO
logic is a disclosed reference, while PrimeLoRA's primary comparison remains
G1 (verified physical GPU-s/request under common correctness and joint SLO) and
G2 (scale-out/churn P95 user TTFT under a common GPU-time budget and joint SLO).
CE and idle-ready pricing remain secondary sensitivity quantities. The audit
also records that D210→D219 7B confirmed-snapshot Full reduced mean TTFT and
control-path time but increased P95 TTFT and changed the physical owner-union
GPU lifecycle only from 15,916.8239 to 15,911.7125 GPU-s (−0.032%). The
previous 4,317-GPU-s values were legacy primary-instance billing fields, not
the frozen G1 resource quantity. The next optimization target is therefore
readiness/first-service tail, not another unvalidated routing rewrite. Existing
3B D138 is a diagnostic, not a qualification point: its remaining mean TTFT is
dominated by observable dispatch/admission, runtime-slot, and LoRA-I/O terms.

No metric threshold, denominator, execution key, historical result, or paper
source was changed. PrimeLoRA 7B qualification remains the active gate; 3B and
external baselines stay paused. The corrected physical-lifecycle reconciliation
is documented in D240; deterministic JSON/hash/whitespace checks pass;
the fresh same-family D239 reviewer did not return within the audit window and
is recorded as `BLOCKED_REVIEWER_TIMEOUT`, not PASS.

## CURRENT — D238 retry4–6 expose natural proactive-coverage conditions; no admission result

2026-10-06（受控诊断审计）. Three fresh Llama-2-7B, seed-42, 22-request
prefixes completed with the native fixed-output contract and complete cleanup.
Retry4 ran the old post-drain observation and found seven HOST sources but no
GPU target selected by the live planner. Retry5 moved the diagnostic into
natural in-flight demand (`arrived=4`, `active=1`, `backlog=1`) and found no
native HOST source on the selected owner. Retry6 used a larger in-flight
window (`arrived=12`, `active=4`, `backlog=6`), found one HOST source, and
observed a frozen Remote→GPU target for a different adapter; the diagnostic
correctly stopped at `mixed_gpu_sources` rather than relabeling it as HOST→GPU.

All three report zero proactive admission decisions, zero admit/defer/reject
events, and `physical_kv_reservation=false`. Thus the retry sequence is a
coverage result, not proof of an ordinary request-path defect and not a
qualification of (E_i(t)). The run-level TTFT/E2E and GPU-s values remain
diagnostic only; no G1/G2, joint SLO, lifecycle ranking, or baseline comparison
is admissible. Full evidence, hashes, and the claim boundary are sealed in
`docs/ieee_tc/D238_RETRY456_DIAGNOSTIC_AUDIT.md` and its JSON sidecar. The
integrity audit is `docs/ieee_tc/D238_RETRY456_EXPERIMENT_AUDIT.md`; the fresh
same-family reviewer did not return within the audit window and is recorded as
BLOCKED, not PASS.

The next action is not another arbitrary short prefix. First pre-register a
legitimate later frozen-trace source window (or report that proactive coverage
is unavailable), then use paired Full/CapacityOnly frozen owner/epoch evidence
with typed decision, allocator/workspace, cancellation and release receipts.
The remote artifact services are stopped and no local GPU compute process
remains.

## CURRENT — D238 retry3 completes the bounded diagnostic but still does not exercise proactive admission

2026-10-06（受控诊断）. A fresh 22-request prefix was run with the protected
`diagnostic_prefix_v1` replay scope after fixing the witness UUID binding and
making the prefix explicit. All 22 requests completed with the native
fixed-output terminal contract; the service, replay, watchdog, lease cleanup,
and GPU-context release checks passed. The remote artifact unit was stopped
after the run and no local GPU compute process remained.

This is still a diagnostic-only result. The machine-readable diagnostic reports
`state=not_eligible` and
`reason=requires_one_initialized_ieee_runtime_and_profiles`; proactive GPU
admission decisions and physical KV reservations are both zero. Three natural
scale-up events occurred, but none had a planned adapter. Thus the run confirms
bounded request-path execution and a remaining proactive-coverage gap; it does
not qualify physical admission, ordinary demand-route correctness, G1/G2, SLO,
lifecycle ranking, or any baseline comparison. The observed 3,926.38-ms mean
TTFT and 10,375.84-ms mean E2E are retained only as startup/queue diagnostics.

The immutable result SHA is
`cefb72492f4c533b9c5cb813a63a20a817cc6ec2d8180507276b8a06091ff355`; the
launch receipt SHA is
`021b6d635259af906a4ca851c66231846026503b84d4ad1b0493294a6add0399`.
Full evidence and the claim boundary are sealed in
`docs/ieee_tc/D238_RETRY3_DIAGNOSTIC_AUDIT.md` and its JSON sidecar. The next
diagnostic must create an initialized runtime/profile and use a frozen-state
Full-versus-CapacityOnly counterfactual; sequential evaluation on one mutable
owner is not valid.

## CURRENT — D238 corrects the D227 admission interpretation

2026-10-06（语义复核）. A fresh read-only review of the IEEE PDF, the
formula-to-code record, and the D227 trace found that \(E_i(t)\) is a proactive
GPU-residency budget. The paper explicitly gives request-driven loading
priority over proactive preparation. D227's 22 request-driven source
resolutions (14 remote, 6 GPU, 2 HOST) submitted no proactive residency
candidate; its zero
admission decisions therefore show insufficient proactive coverage, not a
proven defect in ordinary demand loading. The raw D227 contract, cleanup,
pressure and non-ranking boundaries remain unchanged.

The current checkpoint is being kept within that boundary: when the explicit
gate is active it can emit a serialized core-to-worker observational capacity witness for a
request-driven load, using `capacity_only=true`; it leaves
`proactive_admission_evaluated=false` and `physical_kv_reservation=false`.
When the gate is absent, ordinary native demand loading is preserved. This
witness is diagnostic only and is not a formal physical-admission or G1/G2
qualification. The next bounded action, after code tests and a checkpoint, is a
proactive HOST→GPU decision diagnostic; the full replay remains held.

The detailed correction is `docs/ieee_tc/D238_ADMISSION_SEMANTIC_REASSESSMENT.md`
and its JSON sidecar.

## HISTORICAL — D237 D227 retry2 had no proactive-admission coverage

2026-10-06（受控诊断）. The original D227 launcher first stopped before model
construction because the guarded environment omitted the published-remote
delivery mode variable; that attempt is retained as a protocol/launcher
failure. After restoring the exact approved delivery environment, retry2
completed a bounded 22-request prefix with 22/22 native fixed-output terminal
contracts, return code 0, and confirmed GPU-context cleanup. It is still not a
performance or ranking result.

The important raw observation is that the YAML gate was `true`, but the native
request-driven route recorded zero proactive GPU-admission decisions and zero
physical KV reservations. Every inspected request reports
`physical_capacity_qualified=false`; the native pending journal explicitly
declares that it reserves no physical KV blocks. The original D237 wording
called this a request-path integration issue. D238 supersedes that causal
interpretation: the run exercised demand-driven loading but did not submit a
proactive HOST-to-GPU candidate, so it proves insufficient proactive coverage,
not a defect in request-priority demand loading.
The full replay remains held until a physical owner/epoch reservation
transaction, release semantics, and bounded pressure tests are implemented.

The retry's user TTFT is dominated by 134.4-s engine initialization and queue
wait (mean TTFT 101.27 s; mean service TTFT 0.758 s), so no latency, SLO,
CE, or lifecycle ranking is inferred. The physical owner-union deployment
journal reports 348.72 GPU-s; the separate 602.51 GPU-s billing field includes
a 300-s idle tail and is not the lifecycle metric. All leases were released.
The current Plan startup equation is `80+4+16+2=102 GiB`; older status prose
said 109.5 GiB as a stale byte-equivalent description. D227 used the current
Plan SHA and 102-GiB equation; future manifests must carry the reconciled
canonical value explicitly.

Full evidence and source-line analysis are sealed in
`docs/ieee_tc/D237_D227_RETRY2_ADMISSION_AUDIT.md` and its JSON sidecar.

## CURRENT — D236 independent PEFT reference passes; full Prime qualification remains open

2026-10-05（受控数值参考）. A single guarded 7B `backend-peft-reference`
run completed with `pass=true`, `measurement_complete=true`, and
`matched_reference_consistency_pass=true`. It evaluated the five pre-registered
roles (nonzero-A, zero, nonzero-B, base, and nonzero-A-repeat) against the
existing native receipt and the byte-verified frozen W0 trace. Prompt/tokenizer
identity matched; all three adapters passed exact tensor-value/dtype checks.
The independent full-vocabulary reference reports zero-vs-base difference 0,
nonzero-A and nonzero-B differences, and an identical repeated-A result.

This closes only the bounded numerical reference gate. The result explicitly
remains `semantic_full_pool_qualification=false` and
`production_launch_authorized=false`; it is not a 4,000-request correctness
denominator, SLO result, lifecycle-budget measurement, or ranking point. The
launch receipt and result hashes, plus the exact interpretation boundary, are
sealed in `docs/ieee_tc/D236_PEFT_REFERENCE_RESULT_AUDIT.md` and its JSON sidecar.
The service and auxiliary scopes were stopped, the GPU compute context was
released, and no remote service or replay was started.

The current Plan SHA `0c808509…` contains the explicit no-service startup
equation `80+4+16+2=102 GiB`, while older status prose still says 109.5 GiB.
D236 used the current Plan equation; this discrepancy must be reconciled and
recorded in the next run manifest before the D227 physical-admission or any
performance run.

## CURRENT — D235 confirmed scale-out first-service audit separates schema gap from real tail cause

2026-10-05（离线代码—结果核对）. The three D219 scale-up events were
triggered at visible request count 4 and took 44.91–46.57 s to activate. The
legacy `scaleup_first_service_*` counters are zero by construction in the
`ieee_confirmed` path: activation rejects legacy handoff plans and the event
registrar strips their fields, so those counters cannot be interpreted as
first-service readiness evidence. In contrast, all 3,003 requests served by
the new replicas carry `confirmed_dispatch_snapshot=true`; the first 20
requests per new replica contain only 10 GPU-ready cases out of 60 (16.7%),
with 23 HOST, 12 NVMe and 15 Remote cases. The earliest new-replica requests
were confirmed NVMe/Remote/NVMe and included a 11,288.8-ms TTFT.

The result is diagnostic only: it does not qualify G1/G2 and does not justify
reusing the legacy handoff. It does establish that the confirmed snapshot is
being captured at dispatch, while a newly published replica may still be
selected with a non-GPU source and perform demand loading. The next
first-principles optimization must therefore create a native, evidence-bound
ready-time preparation epoch or a measured admission objective that accounts
for the activation source state; it must not relabel a prediction as GPU-ready.
Details and exact hashes are in
`docs/ieee_tc/D235_SCALEUP_FIRST_SERVICE_AUDIT.md`.

## CURRENT — D234 old/new 7B Prime audit identifies a partial gain and an unresolved tail

2026-10-05（离线结果核对）. The existing 4,000-request ordinary Full (D210)
and confirmed-snapshot Full (D219) were compared field by field without
relabeling either as a formal G1/G2 point. D219 improves mean TTFT from
1650.5 ms to 1508.4 ms (−8.6%), mean E2E from 6034.1 ms to 5848.0 ms
(−3.1%), dispatch-admission wait from 938.4 ms to 814.1 ms (−13.2%), runtime
slot wait from 615.8 ms to 485.9 ms (−21.1%), and mean control-path time from
325.9 ms to 84.7 ms (−74.0%). P95 E2E improves 3.0%.

The same comparison is not uniformly better: P95 TTFT rises from 3415.8 ms to
3620.0 ms (+6.0%), legacy TTFT attainment falls by 0.10 percentage points,
GPU lifecycle time is unchanged (4317.01 vs 4317.05 GPU-s), and GPU cache-hit
rate is not higher. Thus the confirmed-snapshot path has a measured control
path benefit but has not yet demonstrated G1/G2 or a lifecycle reduction. The
100-request D217 retry10 prefix is not a regression counterexample: its cold
prefix cache hit rate is only 74% and its dispatch-window P95 is 10.67 s, so it
is a separate readiness/qualification diagnostic.

This evidence narrows the next first-principles optimization target to
scale-out/readiness tail behavior, not another blind routing rewrite. Numerical
correctness and physical admission evidence remain prerequisites; no code or
metric threshold is changed based on these descriptive differences.

Detailed values and calculation are in
`docs/ieee_tc/D234_7B_OLD_NEW_PRIME_AUDIT.md`.

## CURRENT — D233 remote boundary and independent artifact gate rechecked read-only

2026-10-05 20:09+08. A read-only recheck found the current direct experiment
route at `1000 Mb/s/full` on both the remote `eno1` and local `eno1np0`; both
interfaces are up. The remote 3B/7B artifact units are inactive and no
18080/18081 listener is present. No remote configuration, link setting,
service restart, cache mutation, or traffic-generation action was performed.

The historical D77 observation of remote `100 Mb/s/full` remains a separate
diagnostic. Its recorded peer advertisement was limited to 10/100; the exact
switch/port/cable cause was never established. It is not merged with the
current 1-Gbps state and cannot be used to claim that the machine-room link is
physically limited to 100 Mbps. The metric protocol therefore keeps
request-triggered transfer in user TTFT/E2E, while excluding one-time
prepublished packing, administration, scans, cleanup and unrelated remote
work. A 100-Mbps historical point, a current 1-Gbps point and application
no-delay sensitivity points must remain separately labelled.

The approved immutable gzip delivery caches are present on the remote node:
both manifests report `complete=true`, 500 artifacts, mode
`published_gzip_delivery_v1`, and `preparation_outside_inference=true`; total
archive bytes are 1,581,215,261 (about 1.47 GiB). Current remote free space is
143,522,885,632 bytes (133.67 GiB). The previously sealed independent bound
is 97,407,250,432 bytes (90.72 GiB), derived from the declared concurrent-pack
peak, 64-MiB log allowance, 16-GiB safety reserve and 1.5 margin. Thus the
artifact node passes its own measured gate despite being below the old 150-GiB
floor; the inference host's 150/100-GiB disk and memory/OOM gates are unchanged.

Evidence and calculation details are in
`docs/ieee_tc/D233_REMOTE_BOUNDARY_RECHECK.md`. This recheck is a prerequisite
confirmation only; it does not qualify Prime 7B or authorize a new GPU run.

## CURRENT — D232 independent PEFT-reference inputs are CPU-audited and ready

2026-10-05（本轮离线核查）. The existing completed native five-arm control
receipt `results/ieee_tc/p2_backend_qualification/model_20260926/llama2_7b_nativenumeric5_attempt1.json`
was checked as an input to the already implemented `backend-peft-reference`
gate; no model was loaded and no GPU or remote service was started. It is a
completed, exploratory observation with one same-prompt source request and the
roles nonzero-A, zero, nonzero-B, base, and nonzero-A-repeat. Its trace SHA is
identical to the current frozen W0 trace. All three referenced adapter files
and the local Llama-2-7B safetensors index exist and match the recorded bytes.

The old native control's identical generated token sequences are not treated as
correctness evidence. The independent reference must still recompute the
first-position logits with the original checkpoint and PEFT, then compare them
with the captured native first-token probabilities. A CPU-only input audit is
sealed in `docs/ieee_tc/D232_PEFT_REFERENCE_INPUT_AUDIT.md`. Once the unchanged
109.5-GiB memory gate passes, the next GPU action is this bounded reference
gate—not another 4,000-request replay or a baseline.

## CURRENT — D231 D217 retry10 bounded route qualification completed; formal gate still open

2026-10-05（本轮核查）. The retained retry9 receipt remains unambiguous: its
replay receipt and service ingress both declared the full 4,000-request source,
so it is a launcher/protocol failure with no performance denominator. The
retry10 directory also contained a later bounded run that had not yet been
recorded in this status file; it is now audited below rather than rerun.

The new launcher
`scripts/ieee_tc/run_d217_7b_confirmed_snapshot_cache_validation_retry10.sh`
sets `FAASLORA_TC_DIAGNOSTIC_PREFIX_COUNT=100` before invoking the guarded
external-replay wrapper, passes the retry10 YAML explicitly, and preserves the
qualified NVML binding, allocator policy, W0 trace, fixed-output contract,
confirmed-snapshot route gate, and fresh output/NVMe/HOST roots. A CPU-only
static audit verified 100 requests, seed 42, the route gate, fixed generation
contract, qualification/external-replay flags, prefix value, and fresh paths;
`bash -n`, Python YAML validation, and `git diff --check` passed. No additional
GPU worker was started by this audit.

The bounded retry10 run itself completed 100/100 requests with zero failures,
100/100 fixed-output contract matches, 100/100 native token-timing records and
100/100 confirmed-dispatch snapshots. Its detailed audit is
`docs/ieee_tc/D231_D217_RETRY10_BOUNDARY_AUDIT.md`. It is still diagnostic-only:
the launch receipt records `production_launch_authorized=false`, every request
has `physical_capacity_qualified=false`, no GPU-admission decision was made,
and no independent numerical reference establishes `N_correct`. It therefore
cannot enter the common-warm SLO, Resident budget, G1/G2 or any ranking
denominator. The 109.5-GiB gate remains unchanged for the next authorized
diagnostic.

## CURRENT — D229 final remote metric and artifact-disk protocol frozen; 7B mainline still held

2026-10-05（本轮）. Added `docs/ieee_tc/METRIC_PROTOCOL_FINAL_V2.md`, a single
versioned calculation entry that composes the immutable V1 and Remote Scope V2
protocols. It freezes the HydraServe-aligned warm-reference 5x TTFT/2x TPOT
thresholds, joint correctness/SLO denominator, physical GPU lifecycle G1,
common-budget P95 TTFT G2, E2E decomposition, true-remote inclusion/exclusion,
CE sensitivity, paired-run statistics, and publication/manifest checks. HydraServe
primary sources were rechecked: its NSDI'26 paper reports warm-derived TTFT/TPOT
SLO attainment and GPU memory-time cost, not CE as a universal primary metric;
the official repository identifies its modified vLLM 0.4.2 implementation and
separate remote storage server. No old protocol or result was edited.

The remote artifact node was queried read-only after all services were stopped:
`eno1` reported 1000 Mb/s/full, auto-negotiation on, link detected, and zero
reported RX FCS/RX errors/TX errors/collisions. This is kept separate from the
historical D77 100-Mb/s negotiation state; the cause of that earlier state is not
claimed. Request-triggered remote read/send/receive/write/extract and contention
remain in the user timeline; one-time published-cache creation, administration,
hash scans, cleanup and request-external packing remain outside the request path
and are never subtracted from E2E. No remote service was started or modified.

The artifact-node gate is confirmed as an independent peak rule already present
in `scripts/ieee_tc_preflight.py`: `required = safety_reserve + ceil(1.5 *
(concurrent_pack_peak + log_growth))`, with explicit free-byte/inode and later
quota evidence. It deliberately does not call the inference-host
150/100-GiB `disk_required` rule; inference memory/OOM/disk gates are unchanged.
The five targeted artifact-gate CPU tests passed, as did Python compilation and
`git diff --check`; the unchanged
109.5-GiB physical-startup memory gate still forbids the next D225/D227 GPU
diagnostic. Prime 7B numerical correctness, common warm SLO, Resident budget and
G1/G2 remain open; no 3B or baseline run is authorized.

## CURRENT — D228 opt-in native token capture CPU-qualified; numerical GPU gate still held

2026-10-05 19:45+08. The runner now has a default-off
`ieee_token_id_capture` diagnostic flag.  When explicitly enabled it carries
the already observed native completion `token_ids` through the existing
`native_token_timing` evidence object, validates non-negative integer IDs, and
retains the compact SHA.  Ordinary Full results do not change schema or
behavior.  The new three-case unit test passes; the existing 43 native-timing
and EuroSys runner tests pass; Python compilation and diff checks pass.

This is only preparation for the D225 bounded numerical gate.  No D228 config,
GPU worker, remote artifact service, or token-capture result was started.  A
future fresh 22-request config must opt in explicitly, use distinct roots, and
compare sequences against an independent original-checkpoint PEFT reference.
Until the unchanged 109.5-GiB physical-startup gate passes, `N_correct` remains
unknown and no Prime replay, 3B run, baseline, SLO, or ranking is authorized.

## CURRENT — D227 admission-gate diagnostic configuration prepared; execution held

2026-10-05 19:35+08. A CPU-only audit created the new bounded 7B diagnostic
configuration `configs/ieee_tc/20261006_d227_7b_admission_gate_22req.yaml`.
It differs from immutable D219 only by an explicit
`effective_capacity_admission_enabled=true`, a 22-request bounded prefix,
fresh output/NVMe/HOST roots, and a diagnostic scenario identity.  The same
model, frozen trace/adapter content index, generation contract, vLLM backend,
profiles, resource envelope and remote delivery contract are retained.  YAML
parsing, recursive intended-difference comparison, scenario-map gate merge,
fresh-path absence and `git diff --check` all pass.  The configuration SHA is
`76c9aa04f96d100211395e4267158ffc83e1ded5787f90caa470779103273c73`; D219's
reference SHA remains
`5072d681dc961a44b03ce9f3794b5c13d878f902c6f962cfba13999d5d3a0b63`.

No GPU worker, local result directory, or remote artifact service was started
for D227.  The unchanged physical-startup memory gate remains below threshold
(about 105.6 GiB available versus 109.5 GiB required), so the 22-request run
is not authorized yet.  The design note and status table are
`docs/ieee_tc/D227_ADMISSION_GATE_22_DIAGNOSTIC.md` and
`paper_results/ieee_tc/p2_backend_qualification/20261006_d227_admission_gate/d227_status.md`.
Even after execution, D227 is diagnostic-only and cannot by itself establish
numeric adapter correctness, common SLO, G1/G2, or a full-system ranking.

## CURRENT — post-audit safe hold; Prime 7B acceptance remains the main line

2026-10-05 18:40+08. The D222–D224 Resident audit and quarantine are complete
and backed up. No local GPU compute process, owned experiment scope, or remote
artifact service is active. The current host has about 105 GiB `MemAvailable`
against the unchanged 109.5 GiB physical-startup gate and about 163 GiB free
disk; no memory-cache purge, unrelated-process stop, threshold relaxation, or
GPU retry is authorized. This is a safety hold, not a completed experiment.

The Prime 7B line remains before 3B and external baselines. D219 is a complete
single-seed route/full diagnostic, but numerical adapter correctness, common
warm SLO, Resident budget, and G1/G2 remain open. The next admissible work is
bounded CPU/offline correctness and protocol preparation, followed by one
freshly gated 7B validation only when the unchanged resource gate passes.

## CURRENT — D225 output-difference audit completed; numerical gate remains open

2026-10-05 19:10+08. A read-only audit paired the immutable D210 and D219 7B
request records by all 4,000 request IDs. Adapter IDs, canonical prompt hashes,
input lengths, requested/actual output lengths, native token source and fixed
generation contract match for 4,000/4,000. Completion token-ID SHA differs for
136/4,000 (3.40%). This is an observation about two separate runtime attempts,
not evidence of an incorrect adapter: all 136 rows use a different activation
instance, while cache-tier and ready-state changes also occur among rows whose
hash is unchanged. The logs retain only token-ID digests, not the sequences or an
independent semantic reference, so `N_correct` remains unknown.

The audit and a falsifiable H0/H1 next gate are sealed in
`docs/ieee_tc/D225_OUTPUT_DIFF_AND_NUMERICAL_GATE.md` and the status table in
`paper_results/ieee_tc/p2_backend/20261005_d225_output_diff/`. The next GPU
action, only after the unchanged memory gate passes, is a bounded 22-request
diagnostic that reuses D166/D167/D207/D208 inputs and adds opt-in token/reference
capture; it is not a performance run. No full replay, 3B run or external baseline
is authorized by this audit. No local GPU process or remote artifact service was
started for D225.

## CURRENT — D226 found a disabled admission gate in D219

2026-10-05 19:25+08. A read-only source/config audit found that D219's
`tc_ieee_full` YAML omits `effective_capacity_admission_enabled`; the runner and
`ResourceCoordinator` both default the missing key to `false`. D219's result
metadata therefore records `false` and zero admission decisions/admit/defer/reject
events. This is a static configuration fact, not evidence that the workload had
no capacity pressure. D219 is retained as confirmed-snapshot route/full diagnostic
only, not promoted to a complete Full mechanism result or G1/G2 denominator.

The root-cause chain and the sole next variable (a new config with the gate
explicitly true, first checked by a bounded 22-request diagnostic) are sealed in
`docs/ieee_tc/D226_D219_ADMISSION_GATE_AUDIT.md` and its status table. The new
diagnostic remains gated by D225 numerical correctness preparation and the
unchanged memory/cleanup protocol; no GPU or remote service was started for D226.

## CURRENT — D222–D224 Resident-vLLM prototype RETIRED; no qualified Resident result

2026-10-05 18:25+08. A post-run protocol audit found that the D222 prototype
wrapper did not satisfy the frozen Resident reference contract.  The baseline
runner used the inherited vLLM 0.10.2 environment, not the Prime/vLLM 0.30.0
backend; its effective service cap was 96 GiB without a verified 72/80 GiB
`MemoryHigh`/`MemoryMax` contract; the deployment notice was emitted only after
all replicas became ready; replay used a later independent time origin; and
the receipt recorded an initial `nvidia-smi` inventory as “acquisition” and
closed intervals using only a final memory threshold, without a physical lease,
owner identity, or overlap-union proof.  Dynamic remote fetch/load was also
placed in the replay auxiliary domain.  These are protocol defects, not
performance observations.

The launch also selected `dynamic_remote`; the repository's ordinary
remote-fair vLLM reproduction rule requires round-local `static_remote` staging
unless a separately resource-bounded exception is qualified.  D222–D224 did
not qualify that exception, and their receipts do not capture a vLLM package or
source identity.

The prototype entrypoint and `VLLM_RESIDENT_PROTOCOL_V1=1` path are now
fail-closed in baseline commits `0a8e5f7` and `70bbbfc`.  Historical source and
raw logs remain untouched.  D222 is a
CPU/syntax and dry-run artifact only; D223 is a launcher/protocol failure; D224
completed 100/100 requests with the native fixed-output token contract, but
that contract does not prove numerical adapter correctness or physical
resource accounting.  D224 is therefore retained solely as a generation-
contract diagnostic, not a Resident baseline, lifecycle GPU-s/request sample,
joint-SLO result, G1/G2 input, or ranking point.  No 1,000-request follow-up is
authorized from this wrapper.

The next Resident reference, if needed, must reuse the existing qualified
common preflight, resource-domain, physical lease, common-clock, and cleanup
infrastructure.  The D222–D224 audit is recorded in
`docs/ieee_tc/D222_RESIDENT_PROTOCOL_WRAPPER.md`; the numeric diagnostic table
is `paper_results/ieee_tc/p2_backend_qualification/20261005_d222_d224_resident_audit/d224_diagnostic_table.md`.
No remote service or GPU is running after cleanup.

## CURRENT — D219 7B confirmed-snapshot full replay COMPLETE; formal acceptance remains open

2026-10-05 15:45+08. D219 completed the frozen 7B W0 exploratory full replay
with 4,000/4,000 terminal requests, 4,000/4,000 native fixed-output contract
matches, zero terminal failures, and four physical allocation journals closed
in the required acquire → worker_spawn → native_workers → native_workers_exited
→ release order. The remote artifact service was stopped after local cleanup;
the local GPU census is idle and no owned leases remain. The remote journal
contains 133 artifact-transfer events: 132 replay-linked records all use the
prepublished read-only gzip contract with `pack_performed=false`; one earlier
`finance_lora` preload/initialization event is retained and classified in
`d219_remote_unmatched.csv`, but is excluded from the replay denominator. No
request-time packing was present.

The confirmed-snapshot route evidence is complete at request level: dispatch
snapshots, dispatch tiers, owner/epoch checks, source-tier equality, prompt
hashes and timing-contract fields each cover 4,000/4,000 requests. The source
collection mode is `confirmed_snapshot_cache_v1` for all observed rows, with no
reported full fallback or identity-only rows. There are 252 requests with at
least one stale recheck (530 observed rechecks); these are selected-state
validation observations, not a causal performance estimate. Run metadata sets
`effective_capacity_admission_enabled=false`, and observed admission,
planning, promotion and hierarchy-transition counters are zero; D219 therefore
does not qualify the full physical-capacity/admission mechanism. The producer
also labels `readiness_routing_enabled=false` for the `ieee_confirmed` policy,
although 6,404 readiness-aware decisions are observed; this is recorded as a
metadata predicate/schema mismatch, not as evidence that confirmed routing was
disabled.

Descriptive values, using the frozen V1 Type-1 percentile recomputation, are
mean/P95 TTFT 1,508.362/3,619.799 ms, service TTFT 694.289/1,843.186 ms,
mean/P95 E2E 5,848.015/12,025.636 ms, mean TPOT 37.566 ms, throughput
1.00853 requests/s, and legacy SLO display 98.275%.
Observed physical GPU time is 15,911.713 GPU-s (3.977928 GPU-s per offered
request). Relative to the immediately preceding D210 descriptive run, mean
TTFT is lower by about 8.62% and mean E2E by about 3.08%, while P95 TTFT is
higher by about 5.98%; physical GPU time is effectively unchanged (about
0.00091 higher). Thus D219 is not an across-the-board improvement and does not
close the optimization gate.

The run is deliberately **not** a formal G1/G2 or cross-system result:
`n_correct` is unknown because the native token contract does not establish
numerical adapter correctness; common warm-derived thresholds and Resident
budget references are not yet frozen; and there is only one exploratory seed
with no paired held-out blocks. The 4,000-row timing audit has zero maximum
absolute error for timestamp algebra, stored TTFT/E2E/service fields, and
native TPOT recomputation; these checks validate timing-field consistency, not
numerical adapter correctness. The 98.275% value is retained as a descriptive
legacy field, not as the frozen
joint TTFT/TPOT SLO attainment. The full
curation, status table and diagnostic figure are in
`paper_results/ieee_tc/p2_backend/20261005_d219_full/` and
`figs/ieee_tc/p2_backend/20261005_d219_full/`; the curation correction is in
checkpoint `4e25e39`, pushed
to `faaslora_origin/retry14_continuous_queue_v2`.

**Next:** perform the remaining 7B numerical/output correctness audit and the
common warm/Resident reference qualification using the frozen protocol before
starting 3B or any external baseline. If those gates pass, use D219's frozen
candidate only as a development input for held-out paired blocks; if they do
not, keep Prime-only optimization focused on the measured tail/control-path
bottleneck. Do not claim a PrimeLoRA win or start the baseline matrix from
D219 alone.

Offline inventory D220 confirms that D166/D167, D207, D208 and the D63 native /
independent-PEFT controls already cover the four inspected 7B tensor classes;
repeating those single-prompt or coordinate checks would add no new hypothesis.
They still do not provide a request-bound semantic oracle for all 4,000 D219
requests, so `N_correct` remains unknown. D170 warm measurements remain
provisional and are not a frozen Resident budget. The next evidence must close
one of these two distinct gaps rather than relabel existing diagnostics.

## CURRENT — D217 retry8 launch correction prepared; retry7 evidence reclassified conservatively

2026-10-05 13:12+08. The retry7 log audit does not establish a distinct
post-initialization handoff deadlock. It establishes only that the dedicated
worker did not publish `ready.json` before the 600-s deadline: the checkpoint
load completed, the first CUDA-graph profiling pass and kernel warmup completed,
and the process was then terminated without an engine-ready receipt. The strict
allocation journal therefore remains correctly unqualified; no route,
latency, SLO, or GPU-s/request value is inferred. The earlier wording that
called this a proven handoff problem is superseded by this narrower statement.

The source/config audit found that retry7 used the default shared vLLM cache,
and its cold run populated a new AOT artifact (`b7a8b0...`) immediately before
termination. The known-good D214 launcher explicitly exported the cache roots;
the vLLM source and official compilation documentation describe the cache as a
reusable, configuration-keyed startup artifact. A fresh retry is therefore
justified only as a one-variable startup validation: same D217 route, trace,
generation, resource envelope, and 100-request prefix; fresh output/NVMe/HOST
roots; explicit reuse of the already populated cache; unchanged 600-s timeout.
It is not a performance run and cannot authorize a full replay by itself.

The retry8 configuration was corrected to the authoritative native-host source
SHA and committed/pushed as `42fb789`; no user files were staged. The remote 7B
artifact service was started and authenticated against the immutable published
cache. Local disk, memory, and GPU gates passed immediately before launch.

The first retry8 launch was rejected by the unchanged preflight before any
model process or GPU worker was started: `MemAvailable` was
107,886,551,040 bytes versus the required 109,521,666,048 bytes. This is a
`protocol_or_launcher_error`, not a system failure or performance observation.
Two transient auxiliary scopes created by the abandoned detached launch were
stopped by their exact unit identities; the remote 7B service was stopped and
the local GPU process census is empty. The failed receipt is retained under the
retry8 raw directory and is excluded from every denominator. No timeout,
cache, route, or generation conclusion is drawn from it.

The next cache-startup attempt (retry9) did reach native vLLM readiness and
reused a previously compiled AOT artifact, but its launcher omitted the
guarded `FAASLORA_TC_DIAGNOSTIC_PREFIX_COUNT=100` setting. The external
publisher therefore began the full 4,000-request trace (`arrived=118/4000`)
instead of the intended bounded prefix. I stopped the exact owned
auxiliary/service scopes before treating it as a campaign; remote 7B was
stopped, GPU processes are absent, and the partial receipt/logs remain under
the retry9 raw directory. This is a launcher/protocol failure with no
performance denominator, not a PrimeLoRA result. The corrected retry will add
only the already-supported bounded-prefix environment variable; no
algorithmic, cache, or resource change is authorized by this correction.


## CURRENT — D217 retry7 reached native vLLM initialization but did not qualify worker readiness; route candidate remains pending

2026-10-05 12:56+08. A fresh D217 bounded qualification run was attempted with
the confirmed-snapshot route gate enabled, the unchanged W0 first-100-request
prefix, byte-verified shared trace/subset paths, the qualified NVML binding,
the frozen allocator policy, and `FAASLORA_WORKER_START_TIMEOUT_S=600`. The
remote 7B service authenticated successfully and delivered `finance_lora`
from the immutable `prepublished_gzip_v1` cache: 14,718,801 archive bytes,
`remote_pack_performed=false`, 470.49 ms end-to-end client transfer stage,
and 20,938,675 verified extracted bytes. This is valid delivery-path evidence,
not a request-performance sample.

The run reached an owned native vLLM EngineCore on GPU0. The cold 12.55-GiB
checkpoint load completed in 258.91 s (model load reported 300.36 s), followed
by torch.compile (82.69 s), CUDA graph capture, KV-cache construction and
kernel warmup. However, the dedicated worker never emitted its required
ready-file before the 600-s startup gate; no request was admitted and
`request_terminals.jsonl` is empty. The runner terminated with
`RuntimeError: native worker lifetime unqualified; physical lease retained`;
the launch receipt classifies this as `qualification_command_failure`, not an
algorithmic or latency result. The receipt SHA256 is
`5c858a9646b3398451a4a3a8e1e0db09a415ee5a714a9405cc9d7a89e47e231a`.

The strict physical journal records acquire → worker_spawn →
release_deferred(`native_worker_lifetime_unqualified`), so no GPU-s/request,
route hit rate, SLO, G1/G2 value, or cross-system denominator is inferred.
The watchdog observed the native context disappear and reported no OOM/high/max
event; the owned local auxiliary scope and remote artifact service were then
stopped by exact unit name. The run is retained under
`results/ieee_tc/p2_backend_qualification/d217_20261005/7b_confirmed_snapshot_cache_validation_retry7/`
and excluded from all rankings. This establishes that increasing the timeout
from 180 s to 600 s was necessary but not sufficient: the remaining issue is
the final worker-readiness/engine handoff after model initialization, not a
proven route or workload effect. Before another heavy run, audit that
readiness protocol and its prior successful D216 counterpart; do not simply
increase the timeout again or advance to 3B/baselines.

## CURRENT — D217 retry2/retry3 launch-contract failures retained; confirmed-snapshot GPU validation still pending

2026-10-05 12:15+08. Two fresh D217 retries were completed under distinct
campaign directories and are retained as protocol evidence only. Retry2 passed
the unchanged physical-memory preflight but stopped before model startup because
the launcher did not provide the required explicit NVML binding and binding SHA;
its receipt SHA256 is
`517ae75931f8917816d367bafc633dddfbca5d586f6d263c42d0b212ae864834`.
Retry3 supplied the qualified NVML binding and passed the same preflight, but the
runner stopped during initial deployment before a native worker/GPU context was
created. The frozen `uncached_background_v1` HOST allocator candidate conflicted
with the runner's inherited `expandable_segments:False` aliases; its receipt
SHA256 is
`cec1e5665f910e467891dba73404cef9532cdf1b6682d2acabb04128d89baa1d`.

Neither retry emitted request terminals, lifecycle allocations, route evidence,
or performance measurements. Both are therefore excluded from candidate
selection, G1/G2, figures, and all cross-system denominators. The service cgroup
reported no OOM/high/max event, all owned scopes were stopped by exact unit name,
the GPU terminal census confirmed no owned native context, and the remote
published-cache service was stopped after local cleanup. The authenticated
remote artifact path and immutable `prepublished_gzip_v1` cache were unchanged.

The allocator test selection passes 16/16. It establishes the next launch
contract without changing the algorithm: export the model's frozen
`FAASLORA_IEEE_NATIVE_HOST_ALLOCATOR_POLICY=uncached_background_v1` before the
fresh runner process imports `run_all_experiments.py`, so the import-time legacy
aliases are not introduced; the child then derives the verified
`pinned_max_cached_size_mb:0,pinned_use_background_threads:True` setting. A
fresh retry is authorized only with this environment, the already-qualified
NVML binding, unchanged D217 config semantics, new output/NVMe/HOST roots, and
the unchanged memory/disk gates. No 4,000-request replay, 3B run, or external
baseline is authorized until the 100-request route candidate produces native
GPU evidence and passes its pre-registered mechanism and control-path checks.

The immediately following retry4 also stopped before deployment and is retained
separately. Its receipt SHA256 is
`0f7bde9cc045042b873596f56bb87d95da14be2ce22cdfe7013a9916273a1b6c`. The
unchanged preflight passed, but the qualification supervisor did not receive the
NVML binding because the launch command used the non-existent variable
`FAASLORA_TC_NVML_BINDING_SHA256`; the code contract is
`FAASLORA_TC_NVML_SHA256`. No service log, request, model, or GPU evidence from
retry4 is admissible. This naming error is now corrected in the next launch
command; no runner or algorithm code is changed.

Retry5 then passed the NVML and allocator launch contracts and reached the
runner, but the supplied adapter-subset path was not present on this checkout.
The run therefore stopped during input resolution before model/GPU startup; its
receipt SHA256 is
`fe6cfa85ac21510170ee75e87b3d8c63b44b9ec3705cccf1c2a94853a3d4ffc7`.
The replay trace was otherwise the intended frozen 4,000-request trace. A
read-only content audit found the same trace bytes (SHA256
`efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`) and the
same adapter subset bytes (SHA256
`aa94b21e1295aefde664e5b29c030a9e42e02af946369bfe9ae15336b89b3016`) in the
retained `11_remote_fair_main_local_sim_v3/v4` shared-artifact directories.
Retry5 is excluded from all performance evidence; the next fresh retry uses
those existing byte-verified paths and new result/cache roots. No artifact is
regenerated or overwritten.

Retry6 reached the native deployment path with all input and launch contracts
valid. It verified one real remote `prepublished_gzip_v1` transfer
(`finance_lora`, 14,718,801 archive bytes, 547.7 ms client stage) and created a
GPU context of about 13.8 GiB on the owned RTX 3090. The vLLM cold checkpoint
load did not publish the worker-ready file within the existing 180-s
`FAASLORA_WORKER_START_TIMEOUT_S` window; the parent then retired the worker and
the strict physical journal correctly refused a release because no native-worker
confirmation had been recorded. The receipt SHA256 is
`b2a0cd61e5005538102dab1c38c126010c0d497441eff4f2257ce16dd3bd541c`.
There were zero request terminals, zero route observations and no performance
sample; retry6 is excluded from G1/G2 and all rankings. The remote service and
local scopes were stopped after the exact cleanup path.

This establishes a launch-time, not algorithmic, constraint: D216 succeeded
with a warm page cache, whereas retry6's cold 12.55-GiB checkpoint load exceeded
180 s. The next fresh retry keeps the same model, vLLM, route candidate,
workload, resource envelope, and remote cache, and exports
`FAASLORA_WORKER_START_TIMEOUT_S=600` so the pre-existing cold-start evidence can
complete. This is an explicit startup contract, not a performance tuning knob;
no request is admitted until the same native-worker confirmation, lifecycle,
generation and dispatch evidence gates pass.

## CURRENT — Remote-scope metric protocol V2 frozen; artifact-node disk gate separated from inference host

2026-10-05 11:55+08. A new protocol document
`docs/ieee_tc/METRIC_PROTOCOL_REMOTE_SCOPE_FROZEN_V2.md` was added without
modifying the frozen V1 or reclassifying any historical run. Its SHA256 is
`485a6b7cf9fd34db5fdc87959ebbe9f757054cd7aa26a6a3efea210d91615a29`.
It freezes the distinction required for the real-remote experiment: request-
caused remote open/read/send/receive/write/extract, shared bandwidth, miss
count/bytes/concurrency and legitimate residency remain in the measured service
path; one-time published-cache construction, remote admin, hash scans, cleanup,
logs and request-external packing do not. Those excluded activities remain
audited and are never subtracted from E2E. All systems use the same immutable
published gzip content while retaining their legitimate cache/residency behavior.

The protocol makes the primary claim independent of CE: correct completion and
common warm-derived joint TTFT/TPOT SLO (HydraServe-aligned 5x/2x reference
倍率) precede G1 lifecycle physical GPU-s/request and G2 common-budget P95 TTFT.
CE, idle-ready factors and prices are supplementary sensitivity/Pareto evidence.
It records that the historical D77 100-Mbps observation was a separate peer-
advertisement state; the current read-only check on 174 found local/remote
1-Gbps full-duplex links and no NIC errors. No 10/100-Gbps or H100 claim is made.

The preflight code now exposes `artifact-disk-check`, a read-only gate that
computes the independent bound
`safety_reserve + ceil(1.5 * (concurrent_pack_peak + log_growth))` and checks
explicit artifact-node free bytes/inodes. It does not call the inference-host
150/100-GiB rule, start services, alter transfer concurrency, pack files or
delete data; inference-host memory/disk gates are unchanged. The pure arithmetic,
filesystem and CLI gate tests pass (5/5 targeted). A remote read-only census found
143,676,846,080 free bytes and 30,895,524 free inodes on `/home/lab14`; this is
reported as capacity evidence only, not a launch authorization because formal
delivery uses the already published cache and still requires explicit bound,
quota and service checks.

No D217 GPU validation was started: the unchanged inference startup gate still
has insufficient physical headroom. Prime-only optimization remains at the
confirmed-snapshot candidate decision; 3B and external baselines remain paused.

## CURRENT — D217 confirmed-snapshot route candidate CPU-qualified; GPU validation held by startup memory gate

2026-10-05 11:25+08. A new Prime-only 7B candidate was implemented behind the
disabled-by-default `ieee_route_confirmed_snapshot_cache` gate.  A scoped route
footprint is populated only by the authoritative native request endpoint.  On
later candidate views, non-selected replicas may reuse that immutable confirmed
observation; the selected source is still checked by its owner/epoch/source
transaction (or by one selected-replica identity read for a non-native source).
Any selected conflict invalidates only that `(replica, adapter-scope)` entry and
forces a fresh authoritative read.  This is not a TTL, polling, future-demand
oracle, or unconditional last-known tier decision.  IEEE equations, admission,
generation, workload, profiles and resource limits are unchanged.

The route-identity suite passes 14/14 under the qualified vLLM Python
environment, including authoritative-only cache creation, no duplicate full
read on reuse, missing device identity fail-closed behavior, epoch/source
conflict handling, membership rejection and Type-1 route-evidence checks.
Python compilation, YAML parsing and `git diff --check` also pass.
The broader request-footprint, service-routing, scheduler-observation and
request-lifecycle regression selection passes 292/292 in the bounded auxiliary
3/4-GiB scope.

One fresh D217 diagnostic configuration and distinct output/NVMe/HOST roots were
created.  The existing remote 7B artifact unit was started through its managed
systemd user service and authenticated health/manifest checks reported
`prepublished_gzip_v1` and the frozen 500-adapter set.  The first launcher
attempt stopped before execution because the new output parent did not exist;
the second stopped in preflight with `insufficient_physical_startup_headroom`
(`MemAvailable` about 106--108 GB versus the frozen 109.5 GB requirement).
The failed auxiliary scopes were stopped by their exact unit names.  No model,
GPU worker, replay request, or new result was produced; the D217 directory only
contains the preflight receipt.  A no-write page-cache release did not clear
the margin, so the experiment remains held rather than weakening the gate or
terminating unrelated services.
Because no local inference worker was admitted, the remote 7B artifact unit was
stopped again after the health check; its managed unit and published cache
remain unchanged for the next gated attempt.

**Next:** when the host passes the unchanged startup gate, rerun the same D217
command from its existing empty campaign directory.  Compare its bounded
control-path mean/P95 with D215/D216, audit selected-conflict invalidations and
all lifecycle/contract fields, then decide whether a 4,000-request replay is
permitted.  Do not start 3B or any external baseline before that decision.

## CURRENT — D216 retry7 bounded validation COMPLETE; mechanism active but candidate revoked for full-replay selection

2026-10-04 16:00+08. The fresh D216 retry7 was run only after stopping and
restarting the owned 3B/7B artifact services. The authenticated health check
passed after the service bind completed, and the exact gated replay then used
the immutable published 7B cache. This removes the retry5/retry6 launch-time
authentication mismatch; neither failed attempt is treated as algorithmic or
performance evidence.

The bounded W0 prefix completed 100/100 planned, arrived, submitted,
terminal, successful and native-contract-matched requests. All 100 actual
output lengths equaled their per-request targets, all prompt hashes and
dispatch-time readiness fields were present, and four physical allocation
journals closed in the required order with zero open leases. Launch, service,
replay and watchdog returned 0; native GPU contexts and the owned service path
were released. Observed lifecycle usage was 852.302629 GPU-s (8.523026 per
offered request). The fixed-output contract does not establish numerical
adapter correctness, so `n_correct` remains unknown and the attempt is not a
formal G1/G2 or cross-system result.

The D216 source-level predicate was active: the cumulative `cache_hits`
counter reached 30 in the final route collection, with zero identity failures.
The emitted counters are cumulative and are retained as mechanism evidence,
not converted to a request-level hit rate. The run contained 337 source
snapshots/106 collection IDs; aggregate fields were 3,509 cache hits, 1,386
identity-only observations, 1,136 full authoritative reads, 732 full
fallbacks and 404 cold full reads. These counts are not independent samples.

The primary control-path check is mixed. D216 mean/P95 control path was
238.515/692.622 ms. Relative to D215's 236.004/947.056 ms, the mean is 1.06%
higher while P95 is 26.84% lower. Relative to D214's 462.790-ms mean, D216 is
lower, but that comparison alone cannot satisfy the pre-registered candidate
gate. Because the primary mean did not decrease against the immediately prior
candidate and only one bounded block exists, **D216 is revoked for full-replay
selection**; no 4,000-request replay, 3B run or external baseline is
authorized. The lower P95 is retained as a diagnostic signal for the next
source/log audit, not as a selected advantage.

The 29 joined remote transfer records are all `sent` under
`prepublished_gzip_v1`, with `pack_performed=false`, no temporary archive and
50,650,865 server-written bytes. The result therefore has no retry7
authentication or request-time packaging failure. The full bounded curation
and status table are in
`paper_results/ieee_tc/p2_backend/20261004_d216_route_identity_scope_validation_retry7/`.

**Next:** preserve retry7 and its status table, audit why scope-local reuse did
not reduce the primary mean (collection-wave multiplicity and fallback
population first), and propose one new falsifiable candidate only after that
audit. D216 must not be blindly retried; 3B and external baselines remain
paused.

## CURRENT — D216 route-evidence audit corrected; candidate remains revoked

Backup checkpoint `32487b1` (pushed to `faaslora_origin/retry14_continuous_queue_v2`)
seals the retained retry7 curation, corrected route-evidence audit, wave-ID
provenance fix, and remote/metric decision document. The protected manifest,
old figures and unrelated user files were excluded.

2026-10-04 22:20+08. A bounded CPU-only audit of the retained D215/D216 request
evidence found that the producer-side collection counter and the emitted
collection label are not interchangeable. D216 reports 106 producer collections
after shutdown but only 83 labels in request evidence; D215 reports 107 and 87.
Seven D215 labels and nine D216 labels contain conflicting per-wave counters.
The old collector read the global collection counter after an asynchronous native
await, so a concurrent scope could advance it before the waiting wave emitted its
metadata. Snapshot sums and cumulative counter sums therefore cannot be presented
as RPC totals. Exact RPC count remains unknown.

The runtime fix is narrow evidence provenance: a wave captures its ID at creation
and all joined waiters return that ID. It does not change routing, cache
acceptance, admission, any IEEE equation, workload, SLO, or resource profile.
The route-identity suite plus request-footprint/service-routing/lifecycle/timing
tests passed 275/275 under the auxiliary 3/4-GiB scope, with no GPU or remote
service active.

The corrected Type-1 control P95 values are D215 945.799 ms and D216 690.333 ms;
the D216 mean remains 238.515 ms versus D215 236.004 ms (−1.064% lower-better
direction, i.e. a 1.064% regression). D216 has 337 observed source snapshots
versus 304, 76/100 requests with four source snapshots versus 65/100, and
scaleup_affected 51/100 versus 39/100. Fallback-conditioned control means are
306.126 ms for D216's 67 reported-fallback requests and 101.243 ms for its 33
without reported fallback; these groups are diagnostic, not causal contrasts.

The corrected audit is in
paper_results/ieee_tc/p2_backend/20261004_d216_route_evidence_audit_v2/ and
explicitly records the exact-RPC unknown status, Type-1 recomputation and
population mismatch. The status-table decision is unchanged: mechanism and
lifecycle gates pass, but D216 is revoked for full replay. No 4,000-request,
3B or baseline run is authorized until a new falsifiable candidate explains
fallback/membership cost without relying on future demand or test-point tuning.

## HISTORICAL — D216 scope-local identity candidate CPU QUALIFIED; bounded GPU validation pending

2026-10-04 14:25+08. The D215 source audit produced one falsifiable next
hypothesis: global owner epochs advance for unrelated adapter churn, while the
requested GPU copy can retain the same owner/source identity. D216 keeps a
fresh identity RPC but permits reuse only for a currently confirmed GPU target
whose adapter identity, path/rank, GPU slot, registration and cached scoped
footprint all match. Host-only targets, target-copy changes, unknown or
unconfirmed entries, identity failure and membership changes remain fail-closed
to the authoritative scoped read. The nine equations, admission/budget checks,
workload, generation contract, lifecycle accounting and resource envelope are
unchanged.

The route-identity, request-footprint, identity-recheck, service-routing,
scheduler-observation and request-lifecycle suites pass **298 tests** under the
bounded auxiliary resource domain. Two new tests specifically prove that an
unrelated source incarnation/epoch change can reuse the target GPU footprint,
while a target source-id change still forces a full read. This is semantic
qualification only; no GPU performance result exists yet.

The next authorized action is one fresh 100-request 7B W0 validation using the
exact D215 trace, adapter subset, published delivery cache, D157 profiles and
resource envelope. It must show a nonzero `scope_local_hits` path, reduced
control-path cost, and no lifecycle/coverage regression before a 4,000-request
replay can be considered. 3B and external baselines remain paused.

## CURRENT — D215 REVOKED after bounded route-identity validation; Prime gate remains open

2026-10-04 14:05+08. The D215 cold-scope candidate completed its authorized
100-request 7B diagnostic, but it did not establish the intended low-cost
identity-recheck path. The run is retained as mechanism evidence and is
excluded from every G1/G2, joint-SLO, resource-saving, ranking, and baseline
denominator.

The strict execution contract passed: 100/100 terminal records, 100/100
successful requests, 100/100 fixed-output/native contract matches, four exact
physical allocation journals, zero open leases, and 779.891374 GPU-s observed.
The run was nevertheless not a formal comparison: the token contract does not
prove numerical adapter correctness, and the legacy 5,000-ms display is a
TTFT-only diagnostic rather than the frozen joint TTFT/TPOT SLO.

The decisive mechanism audit is in
`paper_results/ieee_tc/p2_backend/20261004_d215_route_identity_cold_validation/`.
The coordination record reports 107 collections, 7 stale rejections and 1
joined collection. Request evidence contains 87 unique collection IDs; 37
include cold full reads (88 replica reads) and 60 include authoritative full
fallbacks (142 replica reads). The emitted `cache_hits` and `identity_only`
fields are cumulative counters, so they are not converted into a reuse rate.
Repeated authoritative fallback on scopes that had already been cached means
the proposed path was not dominant enough to justify a 4,000-request replay.

Descriptive diagnostic values were routing/control mean/P95 236.004/947.056
ms, dispatch/admission mean/P95 1,719.253/10,809.149 ms, E2E TTFT mean/P95
2,372.378/11,132.852 ms, service TTFT mean/P95 653.125/1,805.050 ms, E2E
mean/P95 8,622.024/19,239.451 ms, and legacy TTFT-only attainment 90%.
These values are diagnostic only and are not compared as accepted performance
improvements.

**Decision:** revoke D215; do not run its 4,000-request replay, 3B, or any
external baseline. The next step is a source-level audit of why the cached
footprint is rejected after a fresh identity observation, followed by one new
CPU-qualified hypothesis. No stacked optimizer or blind retry is authorized.

## HISTORICAL — D215 cold-scope candidate CPU QUALIFIED; superseded by the terminal validation block above

2026-10-04 13:25+08. D214's stage audit identified a specific control-path
candidate rather than a generic latency tuning opportunity: on a cold
`(replica, requested-adapter-scope)`, the identity-recheck implementation could
perform a compact identity RPC and then an authoritative scoped RPC, although
the compact endpoint does not carry the scoped footprint required for routing.
D215 changes only this cold path. It performs one authoritative scoped read on a
cold scope, publishes that measured footprint, and starts fresh identity
rechecks only after the exact scope is cached. Identity mismatch, malformed
coverage, epoch change, or RPC failure still fails closed to a per-replica
authoritative read. No last-known tier, TTL, admission/profile/formula,
workload, generation, or resource-envelope change is included.

The candidate passed 296 related CPU tests (route identity, request footprints,
identity recheck, service routing, scheduler observation and request lifecycle)
under the bounded auxiliary scope. Python compilation and `git diff --check`
also pass. This is semantic qualification only; it is not evidence of GPU
performance improvement.

The fresh bounded validation configuration is
`configs/ieee_tc/20261004_d215_7b_route_identity_cold_validation.yaml`. It
reuses the exact D214 W0 prefix, trace and adapter-subset identity, D157
profiles, fixed-output contract, published read-only artifact delivery, and
resource envelope, with fresh D215 output/NVMe/HOST/receipt roots. The next
authorized action is one 100-request 7B validation. It must pass the existing
request, native timing, physical allocation/release and transfer audits before
its routing/control, dispatch, service-TTFT, E2E and GPU-s observations are
compared with D210/D214. A full 4,000-request replay is allowed only if the
bounded run reduces the predicted control-path cost without a new correctness,
lifecycle or resource failure. 3B and external baselines remain paused.

## CURRENT — D214 retry4 COMPLETE; lifecycle-qualified diagnostic, formal G1/G2 still open

2026-10-04 12:45+08. The D214 route-identity recheck full replay completed and
was closed before any new task was started. The launch, replay and watchdog
returned 0; native GPU-context release was confirmed, the owned service path
was removed, local `nvidia-smi` was empty, and the remote artifact service was
stopped only after local release. The protected artifact credential was then
rotated on both endpoints without restarting the service; the token value is
not stored in the repository or status log.

The strict lifecycle/request audit has 4,000/4,000 terminal records,
4,000/4,000 successful requests, 4,000/4,000 native fixed-output contract
matches, 4,000 unique request IDs, zero terminal errors, and no missing
canonical-prompt/source/token-ID/clock/instance fields. The target is
request-specific (`min(source_expected,256)`), so 3,055 requests have a target
other than 256; the run must not be described as a uniform 256-token replay.
All four physical allocation journals have exactly
`acquire → worker_spawn → native_workers → native_workers_exited → release`,
with zero open leases. Observed lifecycle usage is **15,907.670119 GPU-s**
(3.976917530 GPU-s per offered request).

The immutable delivery evidence contains 132 retry4 transfer records. The
copied remote journal contains 266 lines: 132 join one-to-one with retry4
outcome IDs, while 134 retained out-of-run records are excluded from retry4
counts and bytes. Every joined record reports `prepublished_gzip_v1`,
`pack_performed=false`, and no temporary archive. The main outcome records
`remote_pack_ms=0` and
`remote_pack_duration_basis=absent_by_published_contract`; request-time
packaging therefore did not enter this replay. The joined server-written bytes
are 131,480,060. The raw transfer and lifecycle sources remain unchanged.

Descriptive end-of-run values are E2E TTFT mean/P95/P99
1,953/4,233/6,262 ms, service TTFT 686/1,875/3,868 ms, TPOT 37.2 ms, and E2E
6,285/12,522/14,907 ms. The log's 97% `slo@5000ms` is a legacy TTFT-only
display, not the frozen joint TTFT/TPOT SLO. Relative to D210's descriptive
1,650.511 ms mean and 3,415.442 ms P95 TTFT, retry4 is not an accepted latency
improvement; its lifecycle value is only a diagnostic observation and is not a
formal saving claim. The identity-recheck mechanism was active (5,160 source
collections, 492 joins, 1,602 stale rejections), but these counters do not
establish causal G1/G2 benefit.

Compact, bounded curation (3/4 GiB auxiliary scope) is stored at
`paper_results/ieee_tc/p2_backend/20261004_d214_route_identity_full_retry4/`;
the JSON/CSV/status files retain source hashes and the raw evidence remains
immutable. `formal_comparison_qualified=false` remains intentional:
`n_correct` is not inferred from the native token contract, and common warm
SLO/Resident reference values are not frozen. No figure or ranking is generated
for this diagnostic; the status table is the appropriate paper artifact.

The curated JSON SHA256 is
`69b7865e299d02e9c185d77f0a175be0258702d5b1dfaaa75972bf0208ae5ccf`; the
metrics CSV SHA256 is
`f9422421b45c69b4e749b0ef4618da8f3ed0fc3980b3c01d497fd5c46100f685`.
The D210/D214 stage comparison table and interpretation are in
`d214_vs_d210_stage_comparison.{csv,md}`. It shows routing/control mean rising
from 325.863 ms to 462.790 ms while service TTFT falls from 712.107 ms to
686.015 ms; the added control path dominates the user-visible regression.

**Decision:** D214 passes lifecycle and serving-contract integrity, but does not
close the 7B Prime optimization gate. Before 3B or any external baseline, the
next action is a single new, evidence-backed 7B bottleneck hypothesis (after
the required old/new comparison and numerical-correctness/SLO audits), followed
by a fresh validation/full replay only if that hypothesis predicts improvement.
No blind retry or stacked optimizer is authorized.

## HISTORICAL — D214 retry4 LIVE; superseded by the terminal block above

2026-10-04 11:36+08 continuation: previous turn PROGRESS/VERIFIED WAIT,
current turn VERIFIED WAIT on the same live tool session `46064`. Independent
PID and systemd checks confirm service invocation unchanged and active; the
replay has not terminated. Bounded terminal audit now reads 1,279 terminal /
1,279 success / 1,279 native-contract matches, zero duplicate request IDs and
no terminal error types. This is a partial planned-4,000 population, not
numerical-adapter qualification, formal SLO or G1/G2 acceptance. Latest watchdog
sample 1401: service 18,072,084,480 bytes, host available 97,054,093,312 bytes,
free disk 177,581,068,288 bytes, service swap/events zero, no warning/abort,
foreign compute or escaped worker. Plan/V1 SHA unchanged. No runtime edits,
new experiment, heavy analysis, remote configuration or cleanup occurred.
Continue THIS run to terminal; do not restart on an observation timeout. The
credential-rotation follow-up was pending at that time and is now complete.

2026-10-04 11:16+08. Goal ACTIVE/current turn VERIFIED WAIT. User's once-only
read-only compressed delivery-cache authorization remains fulfilled by D78/D80;
the immutable published cache is reused, not regenerated. The full Plan, status
and V1 protocol were reread in this continuation. No baseline/3B is running.

Lifecycle fix checkpoint `28b810d` and the fresh retry4 configuration `c40a823`
were pushed to `faaslora_origin/retry14_continuous_queue_v2`; unrelated user
changes and `configs/generated/lora_manifest_1000.json` were not staged.
The same W0 trace SHA `efb903254fcddc320b6765144f4118883d3d057267c5d516ee88927d4504957c`
and subset SHA `aa94b21e1295aefde664e5b29c030a9e42e02af946369bfe9ae15336b89b3016`
are reused; only output/NVMe/HOST/receipt roots changed from retry3.

The protected replay is LIVE in tool session `46064`, service scope
`primelora-tc-svc-8fc8702482d74551bf58db5771ff9db1.scope`, invocation
`98a92d8e38b942408d811b5c299d4612`, and auxiliary scope
`primelora-tc-aux-e1e337b26fca4c59ae2bc3acc4d4fef4.scope`. Raw receipt root:
`results/ieee_tc/p2_backend_qualification/d214_20261004/7b_route_identity_full_retry4/`.
At watchdog sample 242, 93 terminal records existed (partial, not acceptance).
Four native GPU PIDs 2792275/2796895/2797323/2797552 were service-owned with
the frozen CPU affinity. Service memory was 16,560,996,352 bytes, host available
97,482,391,552 bytes, service swap and memory events all zero; no warning,
abort, foreign compute or escaped worker. Free disk was 178,698,461,184 bytes.

Do NOT start another replay, change runtime source/config, or alter remote
delivery while this run is live. Next: monitor THIS session to terminal,
verify local physical release, then exact-owned remote stop/copy, strict
bounded analysis, status table/figure as appropriate, and backup. Formal
numeric correctness/common-warm/Resident reference/G1G2 remain open; a raw
development 5-second attainment value is not the frozen joint SLO.

Historical security note (resolved in the terminal block above): an authenticated pre-run curl diagnostic mistakenly echoed
the artifact access token into tool output. Its value is not stored here or in
Git. After the live replay ends and remote evidence is retained, rotate the
local/remote protected token together and verify authenticated access before
the next run. Do not change the credential mid-measurement. Avoid shell tracing,
verbose authenticated curl, or credential-bearing process-argument output.

## CURRENT — D214 7B full replay retry3 completed requests but failed strict physical-owner journal audit

2026-10-04 11:05+08. Goal ACTIVE/current turn PROGRESS. The cached-startup
retry3 completed the entire PrimeLoRA-only W0 replay: 4,000/4,000 requests
arrived, submitted, terminated and succeeded under the frozen
`fixed_length_greedy_v1` contract, with native token IDs and per-request targets
capped at 256 (not a uniform 256-token workload).
The replay process returned 0 and all owned GPU contexts were released. The
raw result and launch evidence remain immutable under
`results/ieee_tc/p2_backend_qualification/d214_20261004/7b_outputs_full_retry3/`
and `.../7b_route_identity_full_retry3/`.

The run is nevertheless **not formally qualified** and is excluded from G1,
G2, SLO, resource and ranking tables. The service returned 1 during final
physical-deployment reduction because one of the four allocation journals
contained a second `native_workers_exited` event after its `release` event.
The other journals had the required acquire → worker_spawn → native_workers →
native_workers_exited → release order. This is a concurrent retirement
accounting race (two shutdown paths waited on the same native worker), not an
OOM, model, workload, remote-delivery or algorithmic failure. The reducer
correctly failed closed rather than converting an ambiguous lifecycle into a
GPU-s/request value. The run recorded no cgroup max/oom/oom_kill event and
preserved its incomplete qualification receipt.

The fix is limited to lifecycle evidence: `wait_workers()` now serializes
concurrent waiters and records `native_workers_exited` exactly once; it does
not weaken release checks or alter any PrimeLoRA routing/admission/profile
logic. The new regression test and the lifecycle/native-retirement/launch
suite passed (178 tests, 0 failures). A fresh retry4 will reuse only the
already-populated compile cache and the exact trace, adapter subset, backend,
resource envelope and D214 feature gate, while using new output/NVMe/HOST and
receipt roots. No baseline or figure is authorized until retry4 passes the
strict request, native-generation, dispatch/readiness, journal-order and
cleanup gates.

## CURRENT — D214 7B full replay retry3 launch evidence retained; retry4 pending

The retry3 launch receipt is retained with `formal_comparison_qualified=false`
and `pass=false` solely because of the physical-owner journal-order reducer
failure described above; this does not invalidate the 4,000 completed request
records as raw diagnostics. The four journal IDs and the unmodified result
files are kept for audit. No result directory is overwritten, and no
performance interpretation has been generated.

## CURRENT — D214 7B full replay retry1 rejected before model initialization; launcher-only retry authorized

2026-10-04 09:24+08. Goal ACTIVE/current turn PROGRESS. The first 4,000-request
D214 full replay was retained as an incomplete launch attempt and is excluded
from every performance, SLO, resource, and ranking table. Its receipt is
`results/ieee_tc/p2_backend_qualification/d214_20261004/7b_route_identity_full_retry1/launch.json`:
the service and replay returned 1, while native GPU-context release was
confirmed and the owned cleanup path was already gone. No model worker reached
inference and no request entered the algorithmic result denominator.

The fail-closed launcher rejected the run during the initial background
scale-up because the parent environment did not carry the frozen D157 native
HOST allocator contract (`uncached_background_v1` with
`pinned_max_cached_size_mb:0,pinned_use_background_threads:True`). This is a
protocol/launcher error, not evidence against D214, the published remote
delivery, or PrimeLoRA latency. The remote 7B artifact unit was stopped after
the attempt and read back inactive; local `nvidia-smi` showed only the normal
driver baseline (15 MiB per GPU), with no owned worker processes.

The retry preserves the exact model, vLLM backend, D157 service/preparation
profiles, W0 trace, 500-adapter subset, fixed-output contract, resource
envelope, and D214 feature gate. It changes only the fresh output/NVMe/HOST
roots and supplies the already-qualified allocator/PATH environment before
the protected user-scope launch. The failed retry1 directories remain
immutable; no old result is overwritten. The full replay remains a Prime-only
candidate check; baseline comparison and formal G1/G2 claims remain blocked
until a complete replay passes the strict correctness, release, and evidence
gates.

The immediately following retry2 launch also exited before the guarded
service scope was created: the preflight attempted to stat the configured
receipt parent `.../d214_20261004/7b_route_identity_full_retry2/`, which had
not yet been created. No model, cache, GPU context, request, or remote
artifact transfer was started. This second result is therefore another
`protocol_or_launcher_error`, retained in the execution log but excluded from
all denominators. The retry remains valid with the same fresh cache/workspace
roots; only the empty receipt-parent directory is created before the next
invocation.

That guarded retry then reached model initialization but failed before any
worker-ready event. The worker log records weight loading, torch.compile,
FlashAttention, CUDA-graph capture and KV-cache setup; no native request
terminal was produced. The parent reached its existing 180-second startup
deadline and teardown retained the physical lease because no native worker
confirmation/exit evidence had been recorded. The launch receipt reports
`service_returncode=-15`, `replay_returncode=1`, native GPU-context release
confirmed, and zero cgroup `max`, `oom`, or `oom_kill` events. The physical
deployment summary therefore remains incomplete with
`gpu_seconds_observed=182.01148`, `n_terminal=0`, `n_correct=null`, and an
open lease journal retained as failure evidence; it is not a resource or
performance sample.

The only falsifiable working explanation is initialization variance around
the first build of the fresh vLLM/Inductor cache: D214 bounded retry7 recorded
an engine activation of about 155.4 seconds under the same frozen profile,
whereas this fresh-cache attempt did not publish ready before 180 seconds.
This is not yet established as the cause. The next attempt reuses the
now-populated read-only compile-cache contents, changes only fresh output,
NVMe, HOST and receipt roots, and keeps the timeout and all algorithmic
settings unchanged. If cached initialization still misses the deadline, the
startup qualification path will be audited before another replay.

## CURRENT — D214 7B route identity-recheck bounded validation PASS; full replay decision pending

2026-10-04 09:10+08. Goal ACTIVE/current turn PROGRESS. The seventh D214
bounded validation is the first attempt that reached the native 7B worker under
the frozen D157 service profile. It used the existing W0 trace prefix (100 of
4,000), the existing 500-adapter subset, `fixed_length_greedy_v1`, the
published remote delivery contract, and only the D214
`resource_coordination.ieee_route_identity_recheck_cache` gate. The venv
`PATH` was made explicit so the already-qualified FlashInfer `ninja` tool was
visible; no algorithmic or workload parameter was changed.

All 100 planned requests arrived, submitted, terminated and passed the native
output contract. The strict counts are 100/100 target-length matches,
100/100 native `vllm_token_ids`, prompt hashes, `ieee_tc_native_v1` timing,
native terminal observations, confirmed dispatch snapshots, and complete
pre-dispatch readiness fields. The route evidence contains 291 per-replica
source observations, all in `identity_recheck_cache_v1` mode. The final
monotone mechanism counters recovered from concurrent evidence are 54 cached
full-state reuses and 198 identity-only observations, with 0 identity
failures and 0 membership rejections. The per-observation
`full_fallback_replica_count` field was observed at most 4 concurrently, but it
is an instantaneous snapshot and must not be interpreted as a total number of
authoritative fallbacks. These are mechanism counters, not request-level
denominators; the auditable extraction and caveat are in
`paper_results/ieee_tc/p2_backend_qualification/d214_20261004_route_identity_validation/d214_7b_retry7_curated.json`.

The launch receipt passed, native GPU-context release was confirmed, the owned
service path was removed, local `nvidia-smi` was empty after completion, and
the D214-owned remote artifact units were explicitly stopped and read back as
inactive. Raw result SHA256 is
`1163f60f4092a2d2ea907630d80057da184cadb15ceb65d0a8f6b772a6269a9e`, and the
launch receipt SHA256 is
`b1bb5ad48ffd896f7d9484b6c7997d4e342e0dd2517f4e9e8e7a268826edfd49`.

The diagnostic had mean overall TTFT 65,663.985 ms and zero 5-second SLO
attainment because the 100-request prefix overlapped model startup; these
values are explicitly non-formal and are not a performance claim. Service-path
TTFT was 593.653 ms mean / 1,471.231 ms P95, while the measured control path
was 138,920.583 us mean / 348,461.808 us P95. A complete 4,000-request replay
is now scientifically justified as a candidate check of the control-path
hypothesis, but it is not yet started: it must use fresh output/cache roots,
the same frozen inputs and environment, and the full replay acceptance gates.
No baseline, 3B, comparison, or ranking is authorized before that decision is
audited. The bounded run remains diagnostic and does not establish G1/G2.

## CURRENT — D214 route identity-recheck candidate CPU QUALIFIED; no GPU replay yet

2026-10-04 08:35+08. Goal ACTIVE/current turn PROGRESS. After the D213
protocol audit, the next PrimeLoRA-only candidate is a feature-gated route
observation optimization. D210 request projection identified the remaining
control-path cost: each confirmed route performed a full scoped native
footprint read even when the owner/source epoch was unchanged. D214 keeps a
fresh identity-only owner read for every replica and reuses a prior measured
scoped footprint only on exact owner/epoch/clock/source/slot identity match;
identity failure, unknown/unconfirmed native state, epoch change or incomplete
footprint falls back to the original full endpoint. No IEEE equation, profile,
admission, lease or SLO definition changed, and the gate defaults OFF.

The implementation is in `scripts/run_all_experiments.py`, documented in
`docs/ieee_tc/D214_ROUTE_IDENTITY_RECHECK_CACHE.md`, and tested by the new
CPU-only `tests/test_ieee_tc_route_identity_cache.py`. The affected request
footprint/lifecycle suites plus the two D214 tests ran in the declared
auxiliary CPU domain: 199 tests passed, 0 failed, 0 swap/OOM evidence. This
qualifies only state-machine correctness; it does not claim a latency gain.

The bounded candidate configuration is now
`configs/ieee_tc/20261004_d214_7b_route_identity_validation.yaml`; it reuses the
D210 7B W0 trace, D157 profiles and published remote delivery, changes only
the feature gate plus new output/cache paths, and leaves the user manifest
untouched. Next is one 100-request W0 validation replay with strict cleanup/
dispatch evidence, followed by a decision from the measured control-path and
TTFT change about whether one complete replay is scientifically justified. No
new baseline, 3B, comparison, or long vLLM run is authorized before this Prime
candidate gate is resolved.

The first launch attempt reached the runner but was rejected before model
initialization because the controller-only gate had been placed in `model`,
changing the frozen service-profile identity. It produced no GPU inference and
is retained as a failed launch diagnostic under the first D214 validation
directory. The gate is now stored under `resource_coordination`, so the model
profile identity remains exactly D157; the retry must use a new output
directory and must not overwrite that failed attempt.

The second bounded attempt is also retained without scientific interpretation.
It passed the external-replay preflight and loaded the shared W0 trace prefix
(100 of 4,000 requests), but the runner stopped before model/GPU initialization
because the configuration still referenced the first attempt's already-created
NVMe and `/dev/shm` workspace roots. The runner's fail-closed contract correctly
raised `FileExistsError: IEEE replay requires fresh per-run workspace roots`;
`N_arrived=N_submitted=0` and no GPU inference occurred. Its launch receipt and
service evidence are under
`results/ieee_tc/p2_backend_qualification/d214_20261004/7b_route_identity_validation_retry2/`.
This is a launch hygiene failure, not evidence against the identity-recheck
candidate. A third bounded validation must use a new output directory and new
workspace roots, and must leave both earlier attempts immutable.

The third bounded attempt also stopped before model/GPU initialization. The
resource preflight passed and the external replay published the same 100-request
diagnostic prefix, but the guarded launcher rejected the attempt because the
required SHA-locked NVML binding variables were not supplied. It therefore
reported `explicit NVML binding and SHA required before native launch`, released
the service scope, and left `N_arrived=N_submitted=0`; there is no algorithmic or
performance interpretation. The retained receipt is under
`results/ieee_tc/p2_backend_qualification/d214_20261004/7b_route_identity_validation_retry3/`.
The next attempt adds only the already-qualified binding
`/home/qhq/anaconda3/envs/LLM_vllm0102/lib/python3.12/site-packages/pynvml.py`
with SHA256
`4251429c25f1615a4166f395d3c09fe0732bfb023864c4b7f12813373e5696ea`, plus
fresh cache roots; the feature gate and workload remain unchanged.

The fourth bounded attempt passed the NVML/resource gates and reached the
runner, but stopped during remote-storage setup because the command omitted the
already frozen 500-adapter subset path. The fail-closed error was
`IEEE input setup requires nonempty unique adapter IDs`; no cache root was
created, no model was initialized, and the watchdog confirmed native GPU
contexts were released. Its complete receipt is retained at
`results/ieee_tc/p2_backend_qualification/d214_20261004/7b_route_identity_validation_retry4/`.
This is a command/provenance error, not a route-candidate result. The next
attempt supplies the existing subset JSON paired with the existing W0 trace;
it does not regenerate or alter the adapter pool.

The fifth bounded attempt supplied both frozen inputs and reached the full-stack
preload path. It then failed before model initialization because the parent
process had imported the runner's legacy allocator default
(`expandable_segments:False`) while the frozen D157 profile requires the
explicit `uncached_background_v1` native HOST allocator. The runner correctly
rejected this inherited conflict (`native HOST allocator candidate conflicts
with inherited allocator tuning`). No request was served; the service scope and
native GPU context were released. The evidence remains at
`results/ieee_tc/p2_backend_qualification/d214_20261004/7b_route_identity_validation_retry5/`.
The next attempt will launch the same profile with its already-specified
allocator policy preconfigured in the parent environment, not by changing the
profile or candidate algorithm.

The sixth bounded attempt reached the real 7B worker and began GPU model
initialization, but FlashInfer's first sampling-kernel JIT failed with
`FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`. The qualified
venv contains the `ninja` executable, but its `bin` directory was not present in
the inherited worker `PATH`; this is an environment propagation failure. The
external publisher observed 13 arrivals before the service connection closed,
with no terminal requests, so the attempt is incomplete and cannot support a
performance or route-semantic claim. The worker, service scope and GPU context
were released; raw worker logs are retained under the retry6 launch evidence.
The next attempt adds the venv `bin` directory to `PATH` only, preserving the
same code, allocator policy, trace, adapter subset and fresh workspace rule.

## CURRENT — D213 Resident repeats COMPLETE as serving diagnostics; V1 reference REJECTED by protocol audit

2026-10-04 08:20+08. Goal ACTIVE/current turn PROGRESS. Before this
classification the full Plan, this status file and METRIC_PROTOCOL_FROZEN_V1
were reread; `analyze-results`, `run-experiment`, `academic-plotting` and
`github-sync` instructions were reread as applicable. D213 repeat3 completed
4000/4000 successful requests, so all three retained repeats pass the fixed
output/native-token serving gates. Repeat3 descriptive values are mean TTFT
230.662 ms, Type-1 P95 TTFT 275.283 ms, mean E2E 3089.306 ms, Type-1 P95 E2E
6969.348 ms and mean TPOT 25.271 ms. GPU processes and the owned unit are
stopped; post-run `nvidia-smi` is empty. The new repeat3 compact receipt is
`paper_results/ieee_tc/p2_backend_qualification/d213_vllm_7b_w0_resident_repeat3_seed42_fixed.json`.

The bounded CPU analysis was run in the auxiliary 3/4 GiB, CPU 2,3,26,27
resource domain. It produced the repeat table and the fail-closed audit at
`paper_results/ieee_tc/p2_backend_qualification/d213_vllm_7b_resident_reference_audit_v2/`;
audit SHA `8d6948a0cedfbdf648a572c67e08eec142c43c3182d60e7e50c5abd71171de72`.
The three replays are **not** eligible to define V1 `U_ref` or the G2 budget:
they used a local static adapter pool rather than the common published remote
delivery, launched four replicas sequentially and started replay only after
all were ready rather than using the common `t=-60s` notice, placed playback
and service in the same resource domain, and recorded only a whole-unit
envelope—not per-GPU allocation/release integrals. The baseline summary also
uses a synthetic startup-plus-300-second-tail lifecycle model, and the runner
used a 7200-second timeout rather than V1's 1800-second qualification guard.

This is a protocol/launcher mismatch discovered in our own reference path,
not a vLLM performance failure and not a reason to hide the successful
serving evidence. The prior repeat1/2 compact receipts remain immutable
historical files; the new audit manifest supersedes only their earlier
“eligible reference candidate” interpretation. No fourth long replay will be
started under the same invalid contract. No figure or PrimeLoRA ranking is
generated; a status table is the scientifically appropriate deliverable.

Next: first construct and CPU-audit a protocol-compliant ordinary-vLLM
reference launcher with common notice, published remote delivery, separated
playback/monitor resources, 1800-second protection and per-GPU lifecycle
events. Only after that gate passes may the three-repeat Resident reference
be rerun. PrimeLoRA 7B formal acceptance, 3B, baselines, M1/M2, ablations and
sensitivity experiments remain open; no configuration or SLO was changed.

## CURRENT — D213 7B Resident repeat2 COMPLETE and eligible; reference mean pending

2026-10-04 06:47+08. Goal ACTIVE/current turn PROGRESS. The full Plan, this
status file and METRIC_PROTOCOL_FROZEN_V1 were reread before acceptance. The
ordinary vLLM Resident candidate completed the same existing W0 trace as
repeat1: 4000/4000 records, 4000/4000 successful requests, all HTTP 200, and
strict `fixed_length_greedy_v1` validation passed for native `vllm_token_ids`,
target length, prompt hash and timing decomposition. Descriptive values are
mean TTFT 231.091 ms, P95 TTFT 274.771 ms, mean E2E 3089.880 ms, P95 E2E
6968.416 ms, and mean TPOT 25.268 ms. These values are a second seed-42
repeat under the same trace/subset and are not a formal ranking.

The systemd journal records the owned four-GPU service envelope from
05:35:44 to 06:46:21 (4237 s). The runner stopped all four owned vLLM trees;
the unit became inactive and post-run `nvidia-smi` showed no compute apps.
The conservative lifecycle envelope is therefore 16948 GPU-s for this
repeat. The compact receipt is
`paper_results/ieee_tc/p2_backend_qualification/d213_vllm_7b_w0_resident_repeat2_seed42_fixed.json`.

Paper-evidence interpretation: D213 now has two complete ordinary-vLLM
Resident repeats eligible for the three-repeat reference set. The lifecycle
envelopes remain conservative per-repeat observations; they are not yet a
frozen `U_ref`, G2 budget, or acceptance threshold. Repeat3 must use the same
trace, subset, generation contract and resource envelope. No performance
figure or PrimeLoRA ranking is justified before that reference protocol and
the common warm-SLO audit are complete.

## CURRENT — D212 ordinary vLLM fixed-output qualification PASS; Resident replay still pending

2026-10-04 04:16+08. Goal ACTIVE/current turn PROGRESS. The full Plan, this
status file and METRIC_PROTOCOL_FROZEN_V1 were reread before qualification
analysis. After the CPU audit and two retained launch-only failures, the
third short 7B ordinary-vLLM qualification completed with four DP=1/TP=1
servers, static registration of the existing 500-adapter subset, and the
existing W0 trace (seed 42, no new trace or weights). The run completed
100/100 requests successfully under `fixed_length_greedy_v1`; the strict
validator passed target length, native `vllm_token_ids`, prompt hash and
decomposition gates for every request. Replay elapsed time was 189.8485 s.
All four services exited normally and the owned GPU/cgroup scopes were
released. Compact provenance is in
`paper_results/ieee_tc/p2_backend_qualification/d212_vllm_7b_seed42_q100_static_fixed_attempt3.json`.

Paper-evidence interpretation: this closes the ordinary-vLLM measurement
qualification precondition, but it is not a Resident reference, warm-SLO
sample, G1/G2 budget, or performance comparison. The next action is a
read-only semantic audit and then the required complete Resident reference
runs, with common delivery and strict 4000-request acceptance. No figure is
generated for this short qualification; a status record is the appropriate
artifact.

## CURRENT — D213 ordinary vLLM Resident semantic audit PASS; complete reference pending

2026-10-04 04:22+08. Goal ACTIVE/current turn PROGRESS. The full Plan, this
status file and METRIC_PROTOCOL_FROZEN_V1 were reread before the audit. The
ordinary baseline runner, replay client and generated DP4/TP1 launch
specification were checked read-only. Neither source contains
`faaslora_full`, `ieee_confirmed`, confirmed-tier propagation, Prime handoff,
hierarchical residency, forwarding or Prime admission/scale-control logic.
The specification contains only ordinary vLLM LoRA, batching, KV/cache and
round-robin endpoint settings; its four services use the existing static
500-adapter pool. The audit is documented in
`docs/ieee_tc/D213_VLLM_RESIDENT_SEMANTIC_AUDIT.md`.

Paper-evidence interpretation: the ordinary vLLM implementation is now a
valid Resident candidate in semantics, and D212 supplies its fixed-output
measurement qualification. This does not freeze a numeric reference or a G2
budget. The next run must use the existing 7B W0 trace, complete 4000/4000
under the fixed contract, release all service scopes, and then be repeated for
the V1 reference protocol; no ranking or figure is justified before that.

## CURRENT — D213 7B Resident repeat1 COMPLETE and eligible; reference mean pending

2026-10-04 05:36+08. Goal ACTIVE/current turn PROGRESS. The full Plan, this
status file and METRIC_PROTOCOL_FROZEN_V1 were reread before acceptance. The
ordinary vLLM Resident candidate completed the existing W0 trace in full:
4000/4000 records, 4000/4000 successful requests, all HTTP 200, and strict
`fixed_length_greedy_v1` validation passed for native `vllm_token_ids`, target
length, prompt hash and timing decomposition. Descriptive values are mean
TTFT 232.078 ms, P95 TTFT 276.110 ms, mean E2E 3092.043 ms, P95 E2E
6977.803 ms, and mean TPOT 25.300 ms. These are one seed-42 repeat, not a
formal ranking.

The systemd journal records the owned four-GPU service envelope from
04:21:49 to 05:32:17 (4228 s). The runner stopped all four owned vLLM trees;
the unit became inactive and post-run `nvidia-smi` showed no compute apps.
The conservative lifecycle envelope is therefore 16912 GPU-s for this repeat.
The compact receipt is
`paper_results/ieee_tc/p2_backend_qualification/d213_vllm_7b_w0_resident_repeat1_seed42_fixed.json`.

Paper-evidence interpretation: D213 is the first complete ordinary-vLLM
Resident repeat eligible for the three-repeat reference set. Its lifecycle
envelope is not yet a frozen `U_ref`, and no G2 budget or ranking is derived
from one repeat. Repeat2 and repeat3 must use the same trace, subset,
generation contract and resource envelope, then the reference mean and
SLO/budget audit can be computed. No performance figure is generated yet;
the single-repeat status table is the appropriate artifact.

## CURRENT — D212 vLLM fixed-output path CPU QUALIFIED; Resident replay still pending

2026-10-04 03:47+08. Goal ACTIVE/current turn PROGRESS. Before any comparison,
the full Plan, this status file and METRIC_PROTOCOL_FROZEN_V1 were reread.
D212 audited the ordinary vLLM runner after the D211 Resident candidate was
rejected for retaining `faaslora_full`/`ieee_confirmed` semantics. The old
runner did not propagate `fixed_length_greedy_v1` to replay and the replay
client had no vLLM-native token-ID source. No GPU, remote service, model,
trace, adapter or paper result was changed in this checkpoint.

Baseline-harness commit
`a98ee47d174e5830d1b08da56ad8b10c4e23a671` was pushed to its `origin/main`.
It adds vLLM `return_token_ids=true`, `vllm_token_ids` validation, explicit
generation-contract propagation, and the empty-text/native-ID observation
fix. The pre-existing dirty Serverless changes in that repository remain
unstaged and untouched. CPU-scoped tests passed: fixed-contract 9/9 and
runner/config 23/23; syntax, shell and diff checks passed. A fixed-contract
dry-run with the existing W0 trace confirmed the 256-output/759-prompt caps
and contract reached the vLLM launch/replay specification; it is not a
performance run.

Paper-evidence interpretation: D212 repairs a measurement precondition for a
future vLLM comparison. It does not qualify ordinary vLLM Resident, establish
numeric adapter correctness, freeze warm SLO or budget references, or compare
PrimeLoRA with vLLM. Next construct and read-only-audit the ordinary vLLM path
without Prime control semantics, then run the short qualification replay before
the required three complete Resident runs. See
`docs/ieee_tc/D212_VLLM_FIXED_OUTPUT_PATH_AUDIT.md`.

## CURRENT — D211 Resident reference candidate rejected before formal use

2026-10-04 03:12+08. Goal ACTIVE/current turn PROGRESS. A fixed-four-runtime
7B Resident reference candidate was stopped after semantic audit, before it
could be used to define (U_{ref}) or a G2 budget. The candidate disabled
handoff, hierarchical residency, forwarding, scale control and effective
capacity admission, but still ran through `faaslora_full` with
`routing_policy=ieee_confirmed`. Its selected-replica path therefore retained
Prime's confirmed-tier/physical IEEE routing machinery and was not the
ordinary vLLM reference required by V1 §7. Continuing would contaminate the
reference rather than answer the budget question. This is a protocol
qualification rejection, not a performance result.

The candidate used vLLM 0.30.0, four TP=1 dedicated runtimes, 7B W0, the
existing 4,000-request/500-adapter seed-42 trace, and the fixed-length greedy
contract. Config SHA `da13a1ec5c9e0954138ec9287b7ff9c341ab718ea7985ed07609f0c57c19bae9`;
launch receipt SHA `f2eb5c1aac936cdf23df59ff8f6008000ff67ae9f7d8b35a43cc7318dc9ee0e4`.
The first attempt was rejected before GPU startup because `max_concurrent_loads`
did not match the measured shared movement owner; that failure is retained.
After restoring the measured limit 3, the second attempt reached the serving
phase, observed 1,642 received/dequeued requests and 1,624 successful live
completions (planned denominator 4,000, no request failures observed), then
was stopped. It ended `service_returncode=-15`, `replay_returncode=-2`,
`classification=safety_abort_unattributed`, and watchdog return 1 because the
forced stop did not prove native context release at the stop instant. The GPU
was subsequently empty, the owned scopes were gone, and no partial value is
eligible for ranking or budget estimation.

All raw replay, ingress, service, watchdog and launch evidence is archived at
`results/ieee_tc/p2_backend_qualification/d211_20261004_resident_ref1/`;
the semantically invalid attempt is under
`7b_resident_ref1/launch.launch.attempt2_semantic_invalid/`. The exact remote
3B/7B transfer journals and monitor log were copied only after local release;
remote services are now inactive, with the recorded remote NIC still
1000/full. No remote repacking, hash scan or reconfiguration occurred.

Paper-evidence interpretation: D211 does not establish a Resident reference,
does not freeze a warm/budget manifest, and does not advance 7B acceptance.
The next step is to construct and read-only-audit an ordinary vLLM path that
does not import `ieee_confirmed` or `faaslora_full` control semantics, then run
a short qualification replay before the required three complete Resident
runs. See `docs/ieee_tc/D211_RESIDENT_REFERENCE_ATTEMPT2.md`. No figure is
generated for this incomplete/invalid candidate; the plan calls for a status
table rather than a misleading performance plot.

## CURRENT — D210 ordinary 7B Full COMPLETE, analyzed; 7B formal acceptance still open

2026-10-04 01:46+08. Goal ACTIVE/current turn PROGRESS. D210 completed and
closed; do NOT repeat this replay or add another optimizer before the next
acceptance evidence. Full Plan/status/V1 reread; analyze-results and
academic-plotting read. No baselines/3B started. The D210 checkpoint is
committed and pushed at `2f366779b8a8388a550b5f03e088187ff1da294b`, with the
user's manifest modification and unrelated dirty paths excluded.

D210 ordinary same-config 7B W0 replay (D209 candidate, current HEAD
be7c2951a64d7f381eb0a4345f5701561dcf3549) completed 4000/4000 terminal,
4000/4000 success, 4000/4000 native contract, zero terminal errors. Launch
pass/service/replay/watchdog return 0; native GPU context released and local
scope/GPU cleanup verified. Remote stopped only after local release; exact
transfer journal copied and SHA-checked. No remote reconfiguration or packing.

D210 vs prior D203 one-run descriptive evidence: mean TTFT 1650.511 vs
1741.609ms (5.23% lower); P95 TTFT 3415.442 vs 3853.586ms (11.37% lower);
mean E2E 6034.115 vs 6173.824ms (2.26% lower); mean TPOT 38.397 vs
38.587ms (0.49% lower), but P95 TPOT 63.920 vs 63.039ms (1.40% higher).
Physical GPU U=15916.823905 GPU-s vs D203 15926.245594 (-0.059%); this is not
yet a lifecycle saving claim. Remote 132 pairs/131480060 client and server
bytes/3300789780 verified logical bytes/packing0. Service peak20272316416B,
minimum host available92068065280B, swap/high/max/OOM/watchdog warnings all0.

Offline projection and occupancy audit ran under 3/4GiB CPU scopes with source
and compact SHA retained. Curated result JSON SHA
597620d2ef2d927c971136a56be337887e5bd201895320ea7bf21d6e43028c66; CSV SHA
7ae260cb0b5d2827e3a0f318487a80a51d10879959727a643f2aef5f84fc0eca.
Figure PDF SHA fe72e897f468fe53c0a01e695a2dfdb361c99482da35ed2e95ac2020a3b8a043;
figure visually inspected, single-column/TNR layout, no overlap. D210 report:
docs/ieee_tc/D210_7B_ORDINARY_FULL.md. The first 128MiB metadata projection
attempt is retained as attempt1; final 165266073B compact projection uses the
previously qualified 192MiB bounded path, original 498984004B outcome remains.

This is observation/contract evidence, not formal G1/G2 acceptance: numerical
adapter correctness, common warm SLO thresholds, Resident GPU budget reference,
and paired multi-block evidence remain open. P95 TPOT trade-off is retained.
NEXT: perform the necessary 7B common warm/Resident reference and correctness
acceptance work before 3B or baselines; do not declare the model accepted from
this single improvement.

### D210 live/replay history (superseded by COMPLETE above)

2026-10-04 00:29+08. This turn PROGRESS: one qualified ordinary replay launched
00:11:47 in tmux tc_d210_full1, now VERIFIED WAIT, goal ACTIVE. Do NOT launch
another replay/optimizer/qualification or stop based on observation timeout.
Service primelora-tc-svc-77e8c69b01e547a78275658afd562ee0.scope,
invocation b3359f20fd7e42199afba473684542ca/inode16524917,72/80GiBswap2.
Aux primelora-tc-aux-e41fba3b3ec440ce8804ea3fa07fbf1d.scope,
invocation42585c2ff6e1458db99213aeeb966b37,3/4GiBswap0CPU2,3,26,27.
Deploymentnotice551157.263463389/businessstart551217.263463389;
plannedarrivalend555181.172366511. Detailed observers OFF,4000W0 unchanged.

Latest physical_deployment/request_terminals.jsonl:876terminal/876success/
876native_contract_matched/error_types empty. Partial ONLY; numerical/full/SLO/
performance acceptance unknown. Early monitor looked for request_terminal in
replay.jsonl (which only logs creation/submission); those zero counts were NOT
actual absence of service terminal records. Correct source now used above.
Watchdog1035/monotonic552205.321853412:service17849155584B,hostavailable
95135272960B,events/swap0,no warning/abort/foreign/escaped. ActualGPUworkers
4049851/4054776/4055069/4055288 all service-owned, originalCPUaffinity preserved.
Freshdisk161290977280B; running100GiB safe. No other heavy/analysis/tool sessions.
Prelaunch/health scopes inactive/identity+paths empty. No remote operations since
activation/health; readonly published cache reused without repacking.
7Bhealthclock remote-process-monotonic:254154de56494625aaba44191e5bd06a.
Exact-owned remote stop/copy wrappers prepared/bashed-n PASS, NOT executed;
must wait local terminal + physical release. No local cleanup wrapper prepared
yet; adapt existing D203 wrapper to exact live identities/PIDs after terminal.

NEXT monitor SAME tmux/service/watchdog -> terminal/local release -> exact remote
stop/copy -> bounded existing D203 metadata/projection/curation/occupancy/timing
analysis -> table/interpretation/scoped backup. Avoid whole-loading large source
JSON or reprojecting historic D20334MB projection. No Git mutation while running.
All numerical/commonreference/legacy G1G2 and subsequent matrix gaps remain OPEN.

### D210 prelaunch history (superseded by LIVE above)

2026-10-04 00:11+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status/V1
reread after compaction; combined-output Plan-tail truncation repaired by
explicit tail read. run-experiment/monitor-experiment read; no subagents.
D209 already SEALED/BACKED, no repeat tests/probe or additional optimizer.
Once-only published cache permission fulfilled D78/D80; reused, not recreated.

Raw results/ieee_tc/p2_backend_qualification/d210_20261004. Reused D203 ordinary
4000 W0 launch/config/prelaunch; only freshowned paths/current source identities.
D157cap4/profile/generation/60snotice/1800s qualification unchanged; detailed
frame/control/execution observers OFF. Current HEADbe7c2951; no runtime edits.
Prelaunch078ec3e435bb6fcca8b9c21c9150d97db0fa521b4752c408e5981f5ec42c986e:
614source refs/147protected/resourcePASS, session90968 CLOSED0. Health CLOSED0.
Remote activated only BEFORE measurement; NIC1000/full and authenticated health
PASS. 3Bpid1813549/id de107771cd5544868a3210f635259bfb;
7Bpid1813551/id97ca094dd9de49e1b5d0aee40ca4dd78;
monitor-d210full1pid1813554/id72aa8b760743474f9c707a572f749370.
NEXT ONE ordinaryFull in tmux -> actualworker/resource checks -> terminal/local
release -> exact-owned remote stop/copy -> bounded reused analysis/table/backup.
No remote config/hash/cleanup/restart during replay; no second experiment.
Numerical/output-difference/commonwarm/Resident/old-newG1G2 remain OPEN. Current
candidate is CPU-qualified only, not performance/model acceptance. 7B then3B then
baselines; all M1M2/A1–A5/S1–S13 retained.

## Previous checkpoint — D209 acquisition identity CPU QUALIFIED, SEALED and BACKED

Backup verified2026-10-04 00:01+08:
be7c2951a64d7f381eb0a4345f5701561dcf3549. Push44490 CLOSED0; independent
ls-remote42250 CLOSED0 exactmatch.10explicitfiles/126staged+archiveentries
scope/secretsPASS; checksum/diff/syntax/smoke/qualificationPASS; usermanifest
excluded. No stagedfiles, GPUcompute, experiment scopes/tmux or live tools.
Remote remains stopped/untouched. D209 DONE, doNOTrepeat tests/probe/backup.
NEXT one ordinary same-config7B4000Full after fresh resource/source gate, then
cleanup/table/interpretation. GoalACTIVE; no performance/model/G1G2 acceptance.

2026-10-03 23:59+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status/V1 read after compaction;
optimize-for-gpu, vLLM+optimization reference, run-experiment, analyze-results and
academic-plotting read. No subagents/GPU/remote operation. Published delivery cache
permission fulfilled D78/D80, not regenerated. D208 COMPLETE, not repeated.

One candidate only: `_acquire_runtime_gpu_reference.observe_source` consumes fresh
identity/tier/epoch/path, not physical footprint. Reuse qualified identity-only
endpoint and monotonic identity frontier. Original native demand transaction
keeps before/after full HOST allocation checks, slot/copy guards and references;
initial routing classes/planner/admission/config/formulas/SLO unchanged.
Acquisition evidence explicitly identity-only; no invented 0 capacity.
Official vLLM0.30 worker/model manager and CPU optimization docs checked;
D178/D181/D202 histories distinguished. No immutable physical-descriptor cache
implemented: consumer audit found this simpler redundant query. Runtime change
confined to ONE runner method; native owner/worker/planner/admission unchanged.

Raw results/ieee_tc/p2_backend_qualification/d209_20261003. All CPU3/4GiBswap0/
CPUs2,3,26,27; terminal events/swap0, actual units/paths absent:
- tests1 session79687 CLOSED1/id3cfeefd1977746bd94f4dc1b8a7df4c8:
  222tests/1error, old fixture native_source_state.owner_id now identity-only.
- tests2 session83761 CLOSED1: same222/1error, next old .sources assertion;
  both corrected to explicit identity record, production candidate unchanged.
- tests3 session2940 CLOSED0:222PASS2.862s, including3new budget/clock checks.
- regression1 tmux ENDED/id36b025de7b0c4809b4d402b30a112882:
  1144PASS157.105s, peak2050699264B; tests overlap, not independent repeats.
- probe1 session60898 CLOSED0/id312059a7ac2c4147b00c1b2767e236a4:
  exact-parent five-source pairs/10records PASS; D20334MBprojection reused.
- curate1 session27710 CLOSED0/id85a9f923ea1b4c7e9430c2b2228acf44:
  147protected/PlanV1/unchanged-runtime AST/sourceSHA/syntax/shell PASS.
Failure/source snapshots retained. No GPU/inference/remote operation or live
tool sessions/scopes/tmux. No hardware/profile/config/threshold changes.

GPU/HOST component paths unchanged. HOSTfile/NVMe/Remote fullgraph1->0,
identity1->2, initial targetgraph1->1 and totalfreshreads3->3. Each1demand
transaction, same tier/holds/releases, no remaining references. D203 source
populations:GPU1189/nativeHOST1766/HOSTfile36/NVMe873/Remote136. Lastthree1045/
4000=26.125%; population coverage, NOT actualRPCcounts or latency attribution.

Doc D209_ACQUISITION_IDENTITY.md SEALED35cec3282f9d6c972288d74ee44d1c7d015f198e00b2ba185f00778206476415;
curated8946a278312da1d2e9a20878ed9f92432cd81d14017702d3b14a1c76ed08985d;
60member2721576B evidencebundlea21feb4e99b370fafbc96d058d91554a8685a574a560cc10b433217e175b6d0c.
Qualification table delivered; no GPU timing gain/CI claimed. DoNOTedit sealed
doc/results. github-sync read. NEXT exact10path scoped checked backup, then
ONE ordinary7B4000Full same D203/D157cap4config/profile, freshownedpaths,
diagnostic observers OFF, current source/Plan/V1 identity. Reuse D203 launch/
preflight/remote/cleanup/analysis. Lastdisk162226733056B, fresh gate beforeheavy.
No repeat D207/D208/D209 or additional optimizer before ordinary validation.
7B numeric/output differences/commonreference/old-newG1G2 remain OPEN, then3B,
thenbaselines; allM1M2/A1–A5/S1–S13 retained. No additional optimization stack.

## Previous checkpoint — D208 isolated LoRA coordinate arithmetic QUALIFIED, SEALED and BACKED

Backup verified2026-10-03 23:34+08:
a19365ef15f9d03885d1981b2e28c4fbbc445955. Push69434 CLOSED0; independent
ls-remote44602 CLOSED0 exactmatch.7explicitfiles/31staged+archive entries
scope/secretsPASS; diff/checksum/syntax/GPUqualification PASS; usermanifest
excluded. No stagedfiles, GPUcompute, experiment scopes/tmux or tool sessions;
remote remains stopped/untouched. Curate actuala835f487f84e4ccba3f5280586eb8812
scope/path absent, events/swap0. Lastdisk162263064576B, freshgate before heavy.
Curated bca88d6cdd8555632175f891f8b176428247be49a4c65c63b15782b1a1434451;
doc e1b92d06c65c35322e24d2e51118cc2717e6a5aa85e0fbe8165ed724604057fd.
D208 DONE; doNOTrepeatprobe/curation/backup. NEXT remaining full-request
correctness/output-difference + common-reference/old-newPrime G1G2, as below.
Goal ACTIVE, no model acceptance or mainmatrix completion.

2026-10-03 23:31+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status/V1
read after compaction; grouped output truncation repaired by separate fullread.
analyze-results/vLLM+troubleshooting/run-experiment/monitor-experiment/
academic-plotting/github-sync read; no subagents. No production code/config/
formula/SLO/native-package change, remote operation or cache generation.
Once-only cache permission fulfilled D78/D80. D207 already backed, not repeated.

ONE minimal arithmetic question following D63/D167/D207: installed native
shrink/expand on coordinate inputs must equal directly read checkpoint columns.
Official v0.30 test_punica_ops.py/utils.py fetched/read; generic broad tolerance
NOT imported. Four existing content classes, all32layers qkv/o, FP16/4slots/
rank64padding,16even A-input coordinates/all64 B-rank coordinates, mixed base
rows and two slot assignments. Native eager + CUDA Graph replay checked.
No new model/adapter/workload, no new observation framework or Full run.

Raw results/ieee_tc/p2_backend_qualification/d208_20261003. arithmetic1 PASS,
512 combinations/673710080 compared elements/0 mismatches, maxabs0. Each128
layer/module/op reference group distinguishes nonzero-adapter swap and base;
combined16782968 differing elements. Elements/padding/repeated coordinates NOT
independent samples, no performance/CI/Full/n_correct upgrade. Arbitrary-input
roundoff/fullmodel arithmetic and D203126hashdifferences remain unproven.

Service b741410a96a44636b72b4a2a705ac28e / invocation240e1426d709418b9cf9430459fd389b,
inode16495613, actualGPU0worker3849963 service-owned CPUs4–23,28–47.
Auxe03a7ae7d7234f4aa3edacbb8a73a234. 22watchdogsamples/peak782970880B/
minhost111552618496B, allguardevents/swap0. Service/watchdog0, physicalcontext
release/pathremovedtrue; exactcleanup23:26:54. No GPU/tmux/scopes/remote live.
GPU script20.305s diagnosticwall, NOT inference timing or lifecycle saving.

Curate1 session6376 CLOSED0, CPU3/4GiBswap0/CPUs2,3,26,27; terminalevents/swap0,
unitidentity/pathabsent. Independently checked full512keycoverage/128controls/
sourceSHA/actualworker/resource/147protected/PlanV1/runtimeunchanged, syntax and
shell smokePASS. No GPU/curation failure. One launcher-preparation JS index
typo before writes and one extra-newline archivalcopy corrected before curation;
not measurement attempts. Exact prelaunchprotocol retained and SHA matched.

Doc D208_LORA_COORDINATE_ARITHMETIC.md/table SEALED, doNOT edit after binding.
24member bundle SHA b479466b36c2b9118575e38de62e5436546f75be323f627105b3a9470cfffe61.
NEXT exact7path checked backup, then RETURN mainline. Do NOT repeat D166/D167/
D207/D208 or broaden kernel proof absent a specific new failure/hypothesis.
Remaining: audit output-difference using evidence capable of distinguishing
batching arithmetic from wrong adapter; commonwarm/Resident and necessary
matched old-newPrime evidence for actual7BG1G2, then3B, thenbaselines. Existing
126outputhashes alone cannot recover firstdivergenttokens/logits. No silent
batch-invariance toggle, tolerance change or speculative Full. AllM1M2/A1–A5/
S1–S13 remain pending. Native-count completion != formal numerical acceptance.

## Previous checkpoint — D207 actual 22-request execution metadata QUALIFIED, SEALED and BACKED

Backup verified2026-10-03 23:11+08:
8efaad5dc7c169881f1ea21d3b5cac96d319c4be. Push25309 CLOSED0; independent
ls-remote97019 CLOSED0 exactmatch.13explicitfiles/120staged+archive entries
scope/secretsPASS; diff/checksumPASS; usermanifestexcluded. No stagedfiles,
GPUcompute/runningexperiment scopes/tmux/tool sessions; remote untouched/stopped.
Latestdisk162280996864B; freshgate before newheavy, floorsunchanged.
D207 qualification and backup DONE; do NOT repeat them. NEXT remaining
correctness/output-difference and common-reference evidence for actual old-new
7B G1G2 acceptance, not a further observation framework or speculative Full.
Read V1 FULL before comparison/config selection. 3B/baselines/latermatrix remain
pending; goal ACTIVE, not model completion. Details and retained failures below.

2026-10-03 23:10+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status read
after compaction; no subagents. vLLM/troubleshooting/run/monitor/academic-plotting/
github-sync read; grouped academic read truncation repaired by individual fullread.
Once-only published-cache approval fulfilled D78/D80, no regeneration or remote
operation. No optimizer/configuration/equation/SLO or native site-package changes.
No system comparison/config selection, V1 unchanged. Production Full still has
NO execution observer. Changes only observer class + opt-in preflight qualification.

Actual V2 source matched official v0.30 hash; previous D206 assumed older runner.
Forward and sample are separate native calls. FULL graph replay / PIECEWISE outer
call / NONE model forward explicitly observed. Internal request IDs retained
verbatim, joined offline to input-side lease via EXISTING frontend exact-ID +
core retirement acknowledgement. No suffix removal, adapter/order guessing,
disabled randomization, eager bypass or loosened arithmetic tolerance.

D207 attempt4 COMPLETE/pass:22/22 correct native-count contracts,3917tokens;
3961iterations=3917nonempty+44empty,31468events/24drainedsnapshots. Independent
native scheduled/completed deltas both3917. Every real forward/logits mapping
matches intended adapter, physical slot and GPU metadata/layer/buffer identity.
Actual graph modes FULL/NONE only; PIECEWISE still CPU-only. Four content classes
reuse D166inputs; no newweights/trace/contentreaudit. Does NOT qualify numerical
arithmetic, concurrentFull/fullpool/n_correct,126outputchanges or formalG1G2.
Raw35555149B SHA993c2956ee2c007dd678327932dc5368e4f454f43595587e8d3197088a46467d;
launchd8a47fdcad9ce9f9336f94f4bce50c708c7f626f658e2c0ba4aa153fc1541489.
Service24c20e6f2a2e45ae87251d87393528b9/invocation405173e431a146c3b32d32f300ee4626,
auxf17bf7b4769440279cf3a3f3a270ad95, nativeworker3749672 actualservice-owned/
CPUs4–23,28–47. return0/watchdog0, physicallease released/pathsgone/GPUempty,
exactcleanup23:06:14. No liveGPU/tmux/remote/tool sessions.

All three failures retained: attempt1 pre-request admitted=[] schema;
attempt2 pre-request absent oldrunner _model_forward; attempt3 firstrequest
observer compared external/internalID, nativecoreexception,0completed/partial
journal=startonly. Not numeric/servingperformance failures. All released, guards0.
FinalCPU tests4 416PASS20.225s/id2089ff8f4ee7437b919a0edf45de6f2e; earlier
tests1/2/3/regression retained (overlap, notindependent).3/4GiBswap0events0.
Bounded inspect4 session93325 CLOSED0; actualmetadata printed, scopegone.
Curate1 session40444 CLOSED0/id8da81aa9213547c8801c125596fed75e;3/4GiBswap0,
protected147/actualinput-SHA/independentjournal/ASTscope/unchanged serving
policy and frontend/scheduler/owner/config checksPASS. No curation failure.

Doc D207_NATIVE_EXECUTION_PREFIX.md SEALED6934aa43e44a204ef92b1b3b42d0be7de92d15dde2fe36a2775c8569415c4867;
curatedc55595d89796854604502b911a71856a3a3c626383e53c872b7d3917ded3db2c;
80member2041377B bundleac2f3dd3f3809ca2295eb9b06d053c80cff6f4b8947783e877664a5d093101a7.
Qualification/attempt/request tables delivered, no speedupfigure/ranking. Do NOT
repeat D205/D206/D207 probes or22requestrun absent a new actual hypothesis.
NEXT exact13path checked backup, then remaining correctness/output-difference
evidence and common-reference/newmetric old-newPrime acceptance. Read V1 FULL
before comparing or selecting configuration. Return mainline, not more observer
framework: actual7BG1G2acceptance→3B→baselines. Commonwarm/Resident/numeric and
legacycomparison stillOPEN; all M1M2/A1–A5/S1–S13 retained, not finished.

### D207 preparation and failed-attempt history (superseded above)

23:02+08 update: attempt3 installed observer, but first native request rejected
because external req_1 != vLLM randomized req_1-ab14385f; adapter963725 unchanged.
Completed0/22; no numeric mismatch concluded. Native core died from observer
exception, partialjournal=startonly; failure/tail retained. launch285a6985a4ca515fe2224641b64cfb3ede351f7811ae9e9c8786e3f1925dea82,
service8f62138b8cde4944b1ad73f9013e25c4/invocation4cf9895be6474e8784bea51f92fdcc03;
return2/watchdog0, release/pathgone true, exactcleanup22:56:10. AllGPU/scopesgone.
Correction reuses existing idempotent native retirement receipt (frontend exact
external/internal map + core retired acknowledgement). Observer collects internal
IDs verbatim in separate counters; offline checker joins ONLY independent receipt,
not suffix removal/adapter/order inference. Existing frontend/owner unchanged.
Tests4 416PASS20.225s/id2089ff8f4ee7437b919a0edf45de6f2e/session4760 CLOSED0,
3/4GiBswap0/events0/pathabsent,147protectedPASS. NEXT preparedattempt4 sameinputs/
config aftergate, newpaths. Noobserver data accepted yet; allG1G2gapsremainOPEN.

2026-10-03 22:55+08 continuation. Full Plan/status reread after compaction;
skills vLLM/troubleshooting/run/monitor/academic-plotting/github-sync read.
Attempt1 native admitted=[] schema error before requests, fixed; tests2
34PASS1.681s/id2ae3289130c84c498780f845d4feffbc. Attempt2 observer assumed
older runner `_model_forward`, absent in actual `vllm.v1.worker.gpu.model_runner`.
No requests submitted/no journal. launch877ba8d843e9ef1ef30f2c859fe5e3cf35ddd2f8f1b646909a1d8ef214fa22d1,
service d5d2fbb8eaa341bfa0265d02225bd7a7/invocation8dd62442608b4f6c93bde6645a0371ea,
return2/watchdog0, physicalrelease/pathremoved true. Exact cleanup verified
22:45:58, service+aux72adc254e61b42b98d0e4f75c491e6a0/path absent/GPUempty.

One diagnostic-only correction targets actual nativeV2 explicitly, no default
serving policy/site-package/config/formula/SLO change. FULL graphmanager replay,
PIECEWISE outer call and NONE forward observed; native prepare_inputs batch
bound across separate execute_model/sample_tokens; no double-counted PW child.
Execution contract vllm_v2_split_forward_sample_v1; checker requires split
events and actual request-state/lora identity. Official V2 runner exactSHA
174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce matchesinstalled.
Tests3 414PASS20.539s/session24652 CLOSED0/id3f90f697428c42ce9af64b2cb38fc243,
3/4GiBswap0CPU2,3,26,27, events/swap0, scopeidentity/path absent. Same tests
overlap preceding groups; no numeric/actualCUDA qualification yet. Source
curl/check session4710 CLOSED0. No remote/cache generation or subagents.
NEXT fresh source/resource gates and SAME22request config under new attempt3,
not anotherFull/optimizer. D207doc DRAFT, noD207commit yet. Allcorrectness/G1G2/
7Bacceptance then3B/baseline/formalmatrixgaps OPEN.

### D207 preparation history (superseded above)

2026-10-03 continuation. Goal ACTIVE/current turn PROGRESS. Full Plan/status,
vLLM/troubleshooting, run-experiment, monitor-experiment, academic-plotting and
github-sync read. No subagents. Existing preflight gained only opt-in
native_execution_metadata plus journal checker; production serving unchanged.
Independent input leases/owner/adapter/target, native begin/end, all ordered
forward/logits boundaries and independent scheduled/completed counter deltas
checked. No timing/arithmetic/fullpool/n_correct upgrade. Reuse D16622request
configuration/content evidence; no content re-audit or optimizer/config change.

Raw results/ieee_tc/p2_backend_qualification/d207_20261003.
CPU3/4GiBswap0CPUs2,3,26,27, events/swap0, units/paths absent:
tests1 session46537 CLOSED0/idb49d47a9b7e64acda59565664ed074f4,
33PASS1.677s (9new tests); regression1 session13055 CLOSED0/
id33603cdeb68546a5a3f1964362491bd1,376PASS19.617s (groups overlap).
One premature read of terminal regression files, later captured/verified.
147protectedPASS. Doc D207_NATIVE_EXECUTION_PREFIX.md DRAFT/table.

Disk belowNEWheavy150GiB; cache1 PyPI-only5–200MiB read-only audit found only
187609088B; NOT APPLIED. session25966 CLOSED0/idd2772e4ffed64b70a1fd9cb83e61aa5c.
cache2 one official PyTorch CUDA2.0.1+cu118 cp39 archive SHA matched index,
but urllibHEAD403, session84639 CLOSED1; no deletion/outputaudit, log/source kept.
cache3 same public URL via curlHEAD200/size, exactSHA
eb55f29db5744eda8a96f5594e637daed0d52278273005de759970e67cfa6a5a,
fresh reference/owner/open-file guards. audit25447 CLOSED0/
id5e9887208165418dbcc1b50f4794519a; audited2267332608allocatedB.
Reviewed auditSHAa750b307d1acb75d8ad0faafa96486b257bd4d4bf4d0ac1a2eac30e0993cfe63.
Apply43193 CLOSED0/id80c28ab4422a411883e1125384a6273e: exact one archive+header
removed,2267332608B freed, four tracked repos/installed environments unchanged.
Unit/path absent, events/swap0. Disk162355134464B aboveNEWheavy150GiB.
All cache scopes same3/4GiBswap0; no model/results changes. Original audits kept.

NEXT freshgate then ONE existing22request actualGPU diagnostic in tmux,
prepared run_7b_execution1.sh (new paths only, D166 config unchanged).
No remote service/cache regeneration needed. Whole numerical/126outputdifferences/
commonwarm/Resident/old-newG1G2 remainOPEN. 7Bacceptance->3B->baselines unchanged.
AllM1M2/A1–A5/S1–S13 retained. Do NOT redo D205/D206 or these CPU tests.

## Previous checkpoint — D206 native execution observer CPU QUALIFIED, SEALED and BACKED

Backup verified2026-10-03 22:09+08:
82ac6b5e21457ddfb0a667597a39e874d1561974. Push21943 CLOSED0;
independent ls-remote36584 CLOSED0 exactmatch.11explicitfiles/62staged+archive
entries scope/secretsPASS; bundle/diff(cr-at-eol)PASS; usermanifestexcluded.
Verification6acc1bcf7ab46880c01afdd779e91f2327a112be74222dd9d8b1cf22945d9458.
No stagedfiles, live experiment scopes/GPU compute/tool sessions; remote remains
stopped/untouched. Lastdisk160026431488B belowNEWheavy150GiB. Do NOT repeat
D206 tests/backup or D205 probe. NEXT existing preflight integration -> matching
lightweight check -> fresh resource/reclaim gate -> ONE22request actualGPU
diagnostic as below. All correctness/acceptance/later-matrix gaps remainOPEN.

2026-10-03 22:07+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status read;
vLLM+troubleshooting, run-experiment, academic-plotting, github-sync read.
No subagents, GPU replay, optimizer/config/formula/SLO changes, remote operation
or cache creation. Latest once-only permission already fulfilled D78/D80.
No system comparison/config selection; V1 unchanged. No model acceptance.

ONE isolated observer added in gpu_monitor + existing observation RPC optional
argument through InferenceEngine/proxy. Default None installs NOTHING in Full.
Native begin/end binding independent of batch; execute/forward/logits boundaries,
actual GPU metadata readback/stream fence, padding/graph mode, all real scheduled
rows, actual executed model-tree module identity/Punica/A-B and metadata pointers.
Bounded64bindings/<=10000iterations/<=32MiB serialized buffer; explicit fail,
no silent truncation. Drain retains counters/sequence/failure; install rollback,
atomic owner check before restore. Dense TP/PP/DP1/nonspec/no ubatching only.
No site-package patch or default inference policy change. Metadata collector
does NOT certify arithmetic/graph replay/fullpool/n_correct. Ordinary runtime
last measured remains D203/D202. Official fixed-v0.30 runner fetched/read;
Punica URL fetch failed, installed exact D205 source SHA reused, not invented.

Raw results/ieee_tc/p2_backend_qualification/d206_20261003. Sequential scopes
3/4GiBswap0CPU2,3,26,27; terminal events/swap0, actual paths+units absent:
- tests1 session71031 CLOSED0/id69b86c389dc64f0e93ea9a18dd5c8a84;
  494PASS19.793s (initial12 new tests). Source/log retained.
- tests2 session63695 CLOSED0/idf90f839be9274b119075df39c076af67;
  496PASS20.416s after model-tree identity/atomic-unhook/rollback additions;
  final14newtests. CPU fake tensors/graph label, NOT actual GPU execution.
- curate1 session23740 CLOSED0/id52d05dcbf78b4646bcae2d9f44a2e14c;
 147protected/sourceSHA/syntax/unchanged AST outside observer+three opt-in RPCs
 PASS; owner/planner/instance/dedicated-worker/preflight unchanged.
No failed test/curation attempts. One initial read-only scope absence check
guessed uid1000; corrected to actual recorded uid1001 and verified explicitly.
No live GPU/experiment/tool sessions. Remote stopped/untouched. Disk160034369536B
belowNEWheavy150GiB; no GPU/build or deletion/relaxed floor this turn.

Doc D206_NATIVE_EXECUTION_OBSERVER.md SEALED; do NOT edit after binding.
Curated7f44ff456079a7646dca4da50c2dcccd6cd3abec525c8c4d688b1d18f5e501df;
39member1164889B bundle19c4f473800ed0daffaed3bd7463a333dfcfed18661f994b6ee4f33d54d19f36.
CPU qualification/status CSV table, no performance figure or ranking.
NEXT exact11path tested backup, then connect existing preflight short-prefix
qualification to start/read/stop + external lease/adapter/sequence/whole-iteration
checks; lightweight integration check -> fresh disk/resource gate and ONE existing
22request/four-content-class actualGPU diagnostic. Reuse D166/D167 content proof;
no repeated D205CPU probe, D206tests or speculativeFull/new framework. Isolated
local-artifact diagnostic needs no remote service. Preserve failures/missing.
All numeric/126output differences/commonwarm/Resident/legacyG1G2 remainOPEN.
7Bacceptance->3B->baselines; all M1M2/A1–A5/S1–S13 remain outstanding.

## Previous checkpoint — D205 execution-metadata checker CPU QUALIFIED, SEALED and BACKED

Backup verified2026-10-03 21:40+08:
0a6c22ebf666a9f97d36c90639927ea1e8b0ad65. Push46148 CLOSED0;
independent ls-remote33898 CLOSED0 exactmatch.11explicitfiles/128staged+archive
entries scope/secretsPASS; bundle checksum/diff(cr-at-eol)PASS, usermanifest
excluded. No staged files, experiment scopes, GPU compute or live tool sessions.
Remote remains stopped/untouched. Plan/V1 SHA unchanged. Lastdisk160057401344B
belowNEWheavy150GiB; no GPU/build before freshgate and audited reclaim.
NEXT isolated execution observer qualification below; do NOT repeat D205 CPU
checker/probe or backup. All numeric/G1G2/model/formal gaps stay OPEN.

2026-10-03 21:38+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status read;
vLLM+troubleshooting, run-experiment, academic-plotting, github-sync read.
No subagents, GPU replay, optimizer, configuration/formula/SLO changes, remote
operation or cache creation. Latest once-only cache approval fulfilled D78/D80.
No system comparison/config selection; V1 unchanged, no new acceptance claim.

Actual vLLM0.30 request->batch->slot->Punica->kernel path audited using installed
files + official fixed-version source. Existing manager already invalidates on
slot-layout change; DO NOT reimplement it. Kernel consumes grouped sorted rows,
counts/starts/active slots and flags, separately for token and sampler. Logical
callID and before/after slot contents alone do not cover this execution boundary.

Added ONE offline preflight validator, no production hook/caller. All previous
preflight AST nodes and serving sources unchanged. External expected backend
request bindings, batch order/counts, physical slot uniqueness, token/sampler
indices, groups/permutation/counts/starts/grid/no-LoRA checked. All-base stale
arrays explicitly unused per native early return. DenseTP1/nonspec scope only;
pure checker NEVER self-certifies GPU collection/arithmetic/fullpool/n_correct.
Current installed LoRAKernelMeta CPU probe8cases/34wrongcontrols rejected;
CPU Torch2.8, NOT nativeTorch2.13/CUDA/compiled-graph qualification. New10tests.

Rawresults/ieee_tc/p2_backend_qualification/d205_20261003. All3/4GiBswap0,
CPU2,3,26,27, terminalevents/swap0, paths/unitsabsent/GPUempty:
- tests1 session89400 CLOSED1/id94c7ee8174e64410875bd1e30872b440;
  384run/2failure3error: four old OSguard tests used model Python withoutpidfd;
  one old dynamic-importfixture lacked sys.modules registration for dataclass.
- tests2 session70196 CLOSED0/idcec5ab5190044d01a676da1c1034137c;
  337PASS21.462s. Only fixture normal-import registration repaired, guards unchanged.
- os1 session63454 CLOSED0/id7908320ad53440b4be48c0eca25f6e42;
  remaining47PASS.096s under qualified /usr/bin/python3. Tests2 final readonly
  seal verification and os1 briefly overlapped; no concurrent GPU/build/inference.
- probe1 session97917 CLOSED0/id43bcb4ac254f484aa06657db7d04aa5a;
  9.73s/RSS1101660KiB,8cases34negativecontrols,CUDAuninitialized.
- curate1 session75120 CLOSED1/ida043f8ac98e24de7b242da775ecc3740;
  JSON parser rejected existing INFO stdout prefix before any curated write.
- curate2 session85357 CLOSED1/id6a7fb690f92642dbaad125460a816e38;
  overescaped prefix regex rejected same retained stdout before curated writes.
- curate3 session7690 CLOSED0/idbbf3fc760b7c4d8588410b8dd081b3c5;
  strict known-prefix parser corrected only in separate third analysis. No tests/
  GPU rerun for curation. All scripts/logs/source snapshots/failures retained.
Some readonly rg/sed searches used missing guessed paths; corrected, no measured
GPU failure. No live tool/GPU/remote jobs. Disk~160068288512B stillbelowNEWheavy
150GiB; no lowered floors or deleted data.147protectedPASS.

Doc D205_NATIVE_EXECUTION_METADATA.md SEALED; do NOT edit after binding.
Curatedfd3fc3e95c1419ae60e1ddc1dbf4b7b8c25256fdb93350be30e07b668c532385;
verification49266635ff76d14b3a7870262406c6d73bffb196efd04d031016b9f2d1b01ba4;
84member1171392B bundlebb90a790b3b14d10d7c4615a76b568e1f7c9134e6fa3398cf65921d06f921379.
Correctness status table/CSV delivered, no performance ranking. Tests1/curate1/
curate2 failures are not omitted or reclassified as GPU failures.

NEXT exact11path tested backup, then isolated native forward-boundary observer
qualification (existing worker extension/preflight, no production overhead),
then ONE existing22request/4contentclass diagnostic only after disk/resource
gate. Bind independent begin_use/backend IDs, actual device metadata, all real
iterations/padding/graph path and layer wrapper/buffer identity. Python-only
operator hook may miss CUDA graph replay; don't call eager-only evidence graph
qualification. Reuse D166/D167 content proof; no repeated loose D63 or speculative
Full. D204126changedoutputs/numeric/commonwarm/Resident/legacyG1G2 stillOPEN.
7Bacceptance->3B->baselines; all M1M2/A1–A5/S1–S13 remain outstanding.

## Previous checkpoint — D204 retained-data diagnosis COMPLETE, SEALED and BACKED

Backup verified2026-10-03 21:10+08:
8d45f0775842136032a5f803cfbd301c0467289d. Push9201 CLOSED0; independent
ls-remote37620 CLOSED0 exactmatch.18explicitfiles/35staged+archiveentries
scope/secretPASS; checksum/diff(cr-at-eol)PASS; usermanifestexcluded.
No stagedfiles, GPUcompute or experiment scopes remain; remote untouched/stopped.
VerificationSHAed367dd51fe40366c77e72acef3c4fce55137af5e4ce0225a50571c987903512.
Lastdisk160086003712B stillbelownewheavy150GiB; freshgatebeforeGPU/build.
NEXT identity-execution observation boundary audit below, not repeatD204,
anotherFull or cachecreation. All numeric/G1G2/model/formalmatrixgaps OPEN.

2026-10-03 21:08+08. Goal ACTIVE/current turn PROGRESS. No serving/code/config/
formula/SLO changes, GPU replay, remote management, new weights or traces.
D203 retained34MBprojection + D195projection + D196table reused; no big-log
reprojection. Once-only cache permission fulfilled D78/D80, not regenerated.
analyze-results/vLLM+reference/run-experiment/academic-plotting/github-sync read.
Full Plan/status/V1 read, hashes unchanged, no subagents.

D203 preengine1447.966992ms. D195->D203 six nonoverlap meanms:
arrival223.603735->194.293671;gate198.569185->123.963843;
routing519.377555->341.223842;residual414.846893->296.681148;
source-to-handoff539.380491->441.574187;handoff-to-native65.913325->50.230300.
One run each, noCI or causal attribution.205TTFT-onlymisses have2050.550355ms
source-to-handoff vs350.114900ms in3711timingpasses; conditionalnotcausal.
Eight500-request index groups timingpass470/469/469/471/463/469/439/461,
so not merely startup. Same161preengine-only thresholdcrossings;
3711/4000timing-only still89short3800, NOformalSLO/modelacceptance.

Outputchanged126same3874:finance11/103,medical2/66,other113/3831.
RetainednativeadapterintIDs matchlogicalmapping4000/4000 inbothversions;
notper-tokenexecution/numericproof. SourcecontentSHAequal672pairs;3328native
sourcepairshavenocomparablefileSHA, explicitUNKNOWN, notfilled. Outputhashes
alone cannot recover firstdivergenttoken/logits. OfficialvLLM0.30 defaults do
notguaranteereproducibility; batchinvariance supportsCC>=8.0 andhasLoRA branch,
but not testedhere andNOTproofwhy126changed. No toggledbackendflag/tolerance.

AllCPU3/4GiBswap0CPUs2,3,26,27; scopes/pathsgone,GPUempty,events/swap0:
analyze1 session11110 CLOSED1/id7dd3cfe6b37546b68983f1a4b129eab3,
10.68s/RSS1107260KiB; incorrectuniversalfileSHAassumption raisedKeyError on
native source; originalscript/log/partialtables retained. analyze2 separate
outputdir, session71305 CLOSED0/idd976272d79854f8dad5d84c78269486a,
11.15s/RSS1110264KiB; only offline schema coverage corrected. finalize1 session
16583 CLOSED0/idc4f8f5b0677a437ba01ff2fa78e1106d;147protectedPASS, independent
CSVsummary/documenttables/sourceSHA/syntaxsmokePASS, productiontreeunchanged.
Read-only searches with guessed missingpaths failed then corrected; no serving
or GPU measurement failure. No live tool/GPU/remote jobs. Disk160080211968B,
belowNEWheavy150GiB; CPUanalysisonly, safetyfloorsunchanged.

Doc D204_REMAINING_MISSES_AND_IDENTITY.md SEALED (doNOTeditafterbinding).
Curated39c0a6d608dc0f2d652a75620ef066fb7ccb35092e5421f85c65cc5eded0e1ab;
17member28611B bundle024a903561b4af739c5146df6c437967ba7b83cf5c4b88ae64a245fe93bb731b.
Rawresults/ieee_tc/p2_backend_qualification/d204_20261003; validcurated
paper_results/ieee_tc/p2_backend/20261003_d204_control_path_attempt2.
Failedfirst20261003_d204_control_path remainslocalprotected, notformalresult.

NEXT exact18path backup, then inspect native token-to-adapter/slot execution
observation boundary using existing22requests/4contentclasses/checkpointslot
evidence; one minimal falsifiable identity diagnostic, not repeatedlooseD63
probabilitytest. Performance question remains selected-source preparation
queue vsactualwork; doNOTspeculatively rerunFull or repeatD176/D184/D202.
7Bnumeric/commonwarm/Resident/old-newG1G2 OPEN, then3B,thenbaselines.
All M1M2/A1–A5/S1–S13 retained; no goal/modelcompletion.

## D204 preparation history — superseded by COMPLETE above

2026-10-03 continuation. Full Plan/status/V1 read; analyze-results, vLLM and
optimization reference, run-experiment and academic-plotting read. No subagents.
Latest once-only remote cache approval already fulfilled by D78/D80; no rebuild,
remote operation or GPU replay. D203 remains COMPLETE/SEALED/BACKED below.
Only bounded D196 analysis wrapper adapted to D203, adding retained timing-outcome
and output-hash/adapter-identity joins against D195; no raw log reprojection.
Current official vLLM0.30 reproducibility/batch-invariance and LoRA kernel source
read: default output reproducibility is not guaranteed; NOT proof of the cause
of126differences. No backend flag, numerical tolerance or SLO changes.
NEXT one CPU-only3/4GiBswap0CPU2,3,26,27 analysis -> cleanup -> exact table and
interpretation -> scoped backup. New GPU/build below150GiB remains disallowed;
latestdisk160098562048B, availablememory112895528960B, GPU/scopesempty.
All7Bnumeric/commonwarm/Resident/old-newG1G2,3B and later matrix gaps OPEN.

## Previous checkpoint — D203 ordinary 7B Full COMPLETE, analyzed, SEALED and BACKED

Backup verified2026-10-03 20:46+08:
81984d41240b09b8d3615ecf4e4b18fa204cbc28. Push46638 CLOSED0;
independent ls-remote51034 CLOSED0 exactmatch.16explicitpaths/103staged+archive
entries passed scope/secrets; bundle checksum and diff(cr-at-eol)PASS.
Usermanifest/unrelated work excluded; no staged files remain. No live GPU,
experiment scopes/tmux/tool sessions; remote remains stopped. No need to repeat
D203 qualification, analysis or backup. Lastdisk160103985152B; fresh gate before
newheavy, currently below150GiB. Goal ACTIVE, no model acceptance.
NEXT retained-data current-stage/remaining-miss and output-identity audit with
history/current official implementation; one falsifiable hypothesis before
new optimization/replay. SevenB must satisfy actual G1/G2 old-new target before
3B/baselines. All numerical/reference/legacy comparison and full matrix gaps
below remain OPEN; D203 merely improved over D195, not final acceptance.

2026-10-03 20:45+08. Goal ACTIVE/current turn PROGRESS. Same D203 Full ended,
all cleanup/analysis complete; no GPU/remote/tmux/analysis/tool sessions live.
FULL Plan/status read for this task; V1 FULL reread before comparisons; hashes
unchanged. monitor-experiment/analyze-results/run-experiment/academic-plotting/
github-sync read; no subagents. No serving/code/threshold/configuration edits.

D195 -> D203 (one development replay each, NOT causal/CI/model acceptance):
meanTTFT2.290592642511532->1.7416089653320668s; P955.207960044150241->
3.8535859030671418s; P998.059697074699216->5.9235012114513665s;
meanTPOT40.26261691433204->38.58740864479467ms;
P95TPOT66.60925991794404->63.038562519337844ms;
U15915.263361512974->15926.245593711035GPU-s (+.069%, no resource saving).
Same provisional D170 timing joint3476->3711/4000=92.775%, +235/+5.875pp;
still89short3800. TTFT-only205/TPOT-only75/both9/failure0. n_correct UNKNOWN,
commonwarm/Resident not frozen, NOformalG1G2 or legacy-Prime superiority.
15nonoutputfields identical;3874outputhashsame/126changed OPEN. Dispatch4000
complete/conflicts0:GPU1189/HOST1802/NVMe873/remote136. AlltimeidentitiesPASS.
Source4337reads/4145collections/16511RPC/192joins/274stale rejects.
Ownedplanner2797complete/708.411902008CPU-s (+sideeffect); init2complete.
Meanpreengine1.447966992s;161requests' preengine span alone exceeds threshold.
Retain D202 candidate; still 7B UNACCEPTED. No blind Full without new hypothesis.

Successful offline scopes3/4GiBswap0CPU2,3,26,27, terminal events/swap0,
identity/path absence verified:
- metadata5:97583 CLOSED0/894e5a9fe73a4ec1841af9c5e762c0d9;
- projection:73044 CLOSED0/327728052db74524a778ee2e4d004bf6;
- curation:7872 CLOSED0/c68e6246a4f7402b96caace77c78d86d;
- occupancy:38953 CLOSED0/12bf3659a50a45f48de1667b533ec027;
- timing:71332 CLOSED0/ab9e76eb3f2f42458b33b81403d859d1;
- evidence:10665 CLOSED0/a8d302de3707458a859d40c2fef39d1e;
- bundle:23272 CLOSED0/0d77932f0b724b009054fcaee867c6a9.
Metadata4 size-guard failure retained as below, not GPU failure; no rerun of
projection or inference. One template-preparation assertion and a doc patch
context mismatch occurred before writes; corrected. One bundle receipt read
used wrong cwd, corrected; checksumPASS. No additional measurement failure.

Doc D203_7B_ORDINARY_FULL.md exacttable and caveats SEALED; do NOT edit after
verification binding. Curated8b92e298f7093295a5b062f45e8502d4c68b9d71e93aa606049116be044763e3;
requestprojection34389917B SHA3e5250e0b34c152f06e7c520bd84469cfa483e2f6414da5d428f848929ffc8cc;
verification10149d2e36f6b3690c09f14cf5a60f43c5c86be28eebe05210610c581177c1be;
87member54106B bundlebe6cdb5f7e2d651e480f5ae0b3b92e6c2a9eff1a6a35fa4a118d25cf6f8fa5e0.
548frozenrefs/147protected PASS; current analyzer matches D199 qualification,
all old helper ASTs and timing tests unchanged. Prior31/71 tests reused and
actual D203 full-data checks passed. No new production tests needed.
NEXT exact16path scoped backup, then retained-data remaining-control/identity
audit against history + official source; one hypothesis/minimal validation.
No repeated D200/D201 or same optimizer/Full without new evidence. Keep
7Bacceptance->3B->baselines and allM1M2/A1–A5/S1–S13 outstanding. Latestdisk
160106127360B BELOW NEWheavy150GiB; fresh resource gate/reclaim audit required
before any GPU/build, not a reason to relax floors or delete unique evidence.

## D203 collection history — superseded by COMPLETE above

2026-10-03 20:29+08 analysis continuation, goal ACTIVE/PROGRESS.
Metadata4 session89280 CLOSED1 after successful stream projection exceeded
128MiB input-size guard (173117160B). No Python collector ran, no OOM evidence;
jq exit0/peak728448KiB. Original wrapper/source/output retained; actualscope
4c10f0f1e1cc4514967b4e459657dcf9 absent; final cgroup counters unavailable.
Metadata5 reuses same projection, no reprojection/GPU rerun. New per-run
192MiB cap justified by previous same-schema RSS and conservative expansion;
3/4GiBswap0 unchanged. Exact two-field control/quarantine projection817902B
keeps shared analyzer128MiB guard unchanged; parent full evidence retained.
97583 CLOSED0, actual894e5a9fe73a4ec1841af9c5e762c0d9, peak610660KiB/events-swap0;
cleanup_metadata5_full1.sh verified scope+path absent/GPUempty.
Preliminary4000success/native contracts; fourleasesreleased;
U15926.245593711035GPU-s,132UUID/allpublished/131480060wireB/
3300789780logicalB/requestpacking0/no mismatches. 4063watchdogsamples,
servicepeak20148785152B/minhost92553940992B/events-swap0/warnings0.
Numerical/SLO/performance qualification remains OPEN.
Request projection LIVE session73044,
primelora-d203-full1-project-20261003.scope actual327728052db74524a778ee2e4d004bf6,
3/4GiBswap0CPU2,3,26,27. Existing D96 streaming jq unchanged.
NEXT wait SAME projection->prepared cleanup_projection_full1.sh->existing
curation/occupancy/timing->table->backup. No optimizer or new GPU run.

2026-10-03 20:22+08. Current turn PROGRESS after verified wait. SAME Full ended,
4000terminal/4000success/4000native-contract, error_types empty; numeric correctness
and formal SLO/G1G2 still UNKNOWN. launch pass/service-replay-watchdog0,
native_gpu_context_release_confirmed/service_path_removed true. tmux absent,
exact service/aux invocation and cgroup paths absent; six recorded PIDs absent,
GPU compute empty. Local cleanup receipt recorded20:19:30. No new GPU replay.
Only after local release, exact-owned remote3B/7B/monitor stopped; stop21985
CLOSED0, allinactive/MainPID0. Journal found by frozen healthclock:
transfers-258535c3794b42d4bdb15bcaadab0c72.jsonl. Copy31392 CLOSED0; localjournal
SHA342e454b70324433f179729283a5a0ee95b602c717346dc26229ee26ab5cfa1a,
monitor35a46797fcf41ff801a4f2241992fb763b0a33402aca065581a3c38d5a0c1052.
No remote management during inference; no cache regeneration.

Plan/status already FULL read for continuous D203 task; hashes unchanged,
V1 FULL reread before analysis. Existing D195 metadata/request projection,
curation/failure/occupancy/timing wrappers adapted only to D203 paths/identities
and previous D195 comparison; no serving/formula/configuration change. Metadata
scope primelora-d203-full1-metadata4-20261003.scope LIVE, session89280;
3/4GiBswap0CPU2,3,26,27, tinyfixturesPASS. Source main_outcome522819369B;
streaming projection before bounded JSON load. All original sources retained.
NEXT wait SAME analysis; verify cleanup; bounded request projection -> full
curation/occupancy/timing -> exact table/interpretation -> scoped backup.
No further optimizer/3B/baseline before D203 closure. Lastdisk160405495808B,
NEWheavy150GiB not met; CPU bounded analysis allowed, safety floors unchanged.
All numerical/commonwarm/Resident/old-newG1G2 and later matrix remain OPEN.

## D203 live-monitor history — superseded by TERMINAL above

2026-10-03 20:07+08 same ongoing monitor, previous/current VERIFIED WAIT;
goal ACTIVE. FULL Plan/status/AGENTS/V1 read after compaction; unchanged SHA.
monitor-experiment/analyze-results/run-experiment/academic-plotting read;
grouped skill output truncation repaired by separate missing-span read.
Latest3563terminal/3563success/3563native-contract/error_types empty, still
partial and NOT numeric/full/SLO/performance acceptance. Same live tmux,
service/aux invocation IDs and four native GPU workers service-owned.
Watchdog3356/monotonic536472.716791287:service20003020800B,
hostavailable95990538240B, events/swap0/no warning/abort/foreign/escaped.
Disk158055075840B: running100GiB safe, NEWheavy150GiB not met. No new
experiment/optimizer, large-log analysis, remote management, cleanup or live
tool session. D195 existing bounded metadata/request projection, curator,
timing/occupancy and evidence wrappers read only; 15 templates also cached
in functions store d195_scripts, NOT adapted or executed for D203 yet.
Once-only readonly delivery-cache approval already fulfilled D78/D80; reuse,
no regeneration. NEXT continue SAME Full; confirmed terminal/local release
-> prepared exact remote stop/copy -> adapt existing bounded D195 analysis
-> full table/interpretation/backup. All acceptance and later matrix gaps OPEN.

2026-10-03 19:56+08 same ongoing monitor, previous/current VERIFIED WAIT;
goal ACTIVE. Same live tmux/service/aux IDs and four owned native workers.
Plan/V1 SHA unchanged from full task read; monitor-experiment reread. Latest
3040terminal/3040success/3040native-contract/error_types empty. Still partial,
not numerical/full/SLO/performance acceptance. Watchdog2749/monotonic
535857.757065129:service19541774336B, hostavailable94885326848B, events/swap0,
no warning/abort/foreign/escaped. Disk158530756608B; running100GiB safe,
newheavy150GiB not met. No new experiment, code, analysis, remote management,
cleanup or live tool session. Continue SAME Full, then local terminal/release,
prepared exact remote stop/copy, bounded existing analysis/table/backup.
All numerical/reference/old-new/G1G2 and remaining matrix gaps stay OPEN.

2026-10-03 19:49+08 same ongoing monitor, previous/current VERIFIED WAIT;
goal ACTIVE. Same tmux/service/aux identities and four owned GPU workers live.
Plan/V1 SHA unchanged from full task read; monitor-experiment reread. Latest
2517terminal/2517success/2517native-contract, error_types empty; NOT full,
numeric, SLO or performance acceptance. Watchdog2317/monotonic535420.08886423:
service19164942336B, hostavailable95112548352B, events/swap0/no warning/abort/
foreign/escaped. Disk158814269440B, running100GiB safe/newheavy150GiB not met.
No new experiments/code/analysis/remote management/cleanup/tool sessions.
NEXT keep this exact replay, then verified localrelease -> prepared remote
stop/copy -> bounded D195 analysis/table/backup. All acceptance gaps remain OPEN.

2026-10-03 19:43+08 same-task monitor continuation; previous/current VERIFIED
WAIT, goal ACTIVE. Plan/V1 hashes remain unchanged from full task read;
monitor-experiment reread. No new task, optimizer, analysis or remote changes.
Latest2024terminal/2024success/2024native-contract, error_types empty; still
partial, not numeric/full/SLO/performance acceptance. tmux tc_d203_full1 and
exact service/aux IDs below active; same four GPU workers service-owned.
Watchdog1958/monotonic535056.356041721:service18536136704B, hostavailable
95256092672B, events/swap0/no warning/abort/foreign/escaped. Disk159137865728B,
running100GiB safe/newheavy150GiB not passed. No live tool sessions, no cleanup
or extra workload. NEXT continue SAME run to terminal and verified release;
then prepared remote stop/copy -> bounded D195 analysis/table/backup. All
7B acceptance/3B/baseline/formal/ablation/sensitivity work remains OPEN.

2026-10-03 19:37+08 ongoing-monitor continuation, previous/current VERIFIED
WAIT; goal ACTIVE. Same live tmux tc_d203_full1 and exact service/aux invocation
IDs below, no new replay or production changes. Plan/V1 SHA unchanged from full
read of this continuous task; monitor-experiment reread. Latest1525terminal/
1525success/1525native-contract/error_types empty. Partial only, not full/numeric/
SLO/performance acceptance. Watchdog1600/monotonic534693.630221785: service
18148175872B, hostavailable95432777728B, events/swap0/no warning/abort/foreign/
escaped. Four same native GPU PIDs still service-owned. Disk159427059712B above
running100GiB but below newheavy150GiB; no extra heavy work or cleanup. No tool
sessions or remote management. NEXT same replay monitoring; ONLY after local
terminal/release execute prepared exact-owned remote stop/copy, then bounded
D195 analysis adaptation/table/backup. No baseline/3B resumed; all gaps OPEN.

2026-10-03 19:30+08 ongoing-monitor continuation: previous/current turns
VERIFIED WAIT on same live handle; goal ACTIVE. Plan/V1 SHA rechecked unchanged
from full read in this continuous monitoring task; monitor-experiment reread.
No new task/replay/optimizer/analysis/remote management/cleanup/backup. Latest
1069terminal/1069success/1069native-contract, error_types empty; partial only,
numeric/full/SLO/performance acceptance NOT established. Same four GPU PIDs
and service/aux invocation IDs below, both active; tmux tc_d203_full1 alive.
Watchdog1186/monotonic534274.213385683:service18105544704B, hostavailable
95398572032B, events/swap0/no warning/abort/foreign/escaped. Disk159549440000B:
above running100GiB but below NEWheavy150GiB; no other heavy work. No live tool
sessions. Continue exact Full; terminal->local release->remote stop/copy->bounded
existing analysis/table/backup. All 7B/3B and later acceptance gaps remain OPEN.

2026-10-03 19:22+08 continuation: previous turn PROGRESS (qualified ordinary
Full launched); this turn VERIFIED WAIT on the same live tmux/service/aux.
Full Plan/status and monitor-experiment read, no truncation. No new replay,
optimizer, analysis, remote operation, cleanup or backup during measurement.
Only D195 exact local cleanup wrapper adapted to current identities, bash-n
PASS, NOT executed. analyze-results skill inspected; no results interpretation.
Live sample696/monotonic533777.731856753:517terminal/517success/517native
contracts, no error_type. Numeric correctness and full completion UNKNOWN;
do NOT label this partial population as performance acceptance. Four native
PIDs2744241/2749026/2749402/2749657 still service-owned. Service/aux invocation
IDs below unchanged; both active, tmux tc_d203_full1 alive. Service17176203264B,
hostavailable95960543232B, events/swap0/no warning/abort/foreign/escaped.
Disk160102776832B, running100GiB floor safe, NEWheavy150GiB not passed.
No live tool sessions. Remote remains untouched during this run. NEXT continue
this exact replay -> confirm local terminal/releases -> exact remote stop/copy
-> bounded existing D195 analysis/table/backup. Goal ACTIVE, all later work OPEN.

19:13+08 monitor: sample168/monotonic533242.800717055,46terminal records;
NOT complete/native/numeric/performance acceptance. Four actual GPU workers
2744241/2749026/2749402/2749657 all service_member=true, CPUs4–23,28–47.
Service15317975040B/hostavailable97029894144B/events-swap0/no warning/abort/
foreigncompute/escapedworkers. tmux alive. Disk160672710656B below NEWheavy
150GiB but above unchanged RUNNING100GiB floor; no other heavy work allowed.
No live tool sessions; only this GPU replay/aux and owned remote services live.
No remote management since health/start; cleanup scripts are prepared but
must NOT run until confirmed local terminal and physical release. Continue
monitoring this exact run, no second replay/optimization/qualification.

Launched2026-10-03 19:10+08 in tmux tc_d203_full1. Raw
results/ieee_tc/p2_backend_qualification/d203_20261003. Existing D195 ordinary
4000 config unchanged except owned paths; diagnostic prefix/frame/control OFF.
Prelaunch0586567f3ea5c862ed2e49e7180804f1de6458fc6c34e46c17e2a122a7b28a63:
548source references/147protected/resource pass; session60778 CLOSED0.
Prelaunch actual13a35835166d4e418e7ea1f7ab6804e4; health actual
3fc0bb56b64d41cea35488d238c44f97 immediateCLOSED0. Both scopes absent.
Service primelora-tc-svc-b45ff477137e448999b428d39a8e14f0.scope,
invocationbcd562dc2b5148a59aed8c38d466e933/inode16336682,72/80GiBswap2.
Aux primelora-tc-aux-7fb06b5a677d4b62b58776aefd5653ac.scope,
invocation85dc6c7dc6fa4137b23d912186204f66,3/4GiBswap0CPU2,3,26,27.
Deploymentnotice533073.53751656/businessstart533133.53751656. Native GPU
worker2744241 observed; verify all scale-out workers from live watchdog next.
Remote ACTIVE, both NIC1000/full and health checked before measurement:
3Bpid1536346/id987612c9cfd2405eadc6e096b071535d;
7Bpid1536348/ide8904853bdeb472595e5a9f54718339c;
monitor-d203full1pid1536351/id0845c03e49ad4254ab473f32ef132b61.
7Bclockremote-process-monotonic:a211cea09c844dcfa7e40d11d0a5ef83.
Exact stop/copy scripts prepared, NOT executed; only after local terminal and
physical release. No remote management/cleanup/hash scans during replay.
NEXT monitor actual workers/resources -> existing cleanup -> bounded D195
analysis adaptation -> exact table/interpretation -> scoped backup. No result
or model qualification yet. All later acceptance/matrix gaps remain OPEN.

2026-10-03 19:09+08 preparation history. Full Plan/status/V1 reread; grouped output truncation in
status tail corrected by explicit bounded reread. run-experiment/monitor-experiment
read; no subagents. D202 checkpoint below already backed; no repeated tests or
optimizer. Reuse D195 ordinary4000 config/launcher/preflight with only fresh
d203_20261003 output paths and current source/evidence identity. Prefix and
detailed frame/control observer OFF. Candidate/evidence HEAD2fc3630d.
Latest census GPUempty/no experiment scopes/tmux; remote was stopped. Disk
161182912512B and hostavailable112587755520B, fresh prelaunch still required.
User once-only cache permission already fulfilled D78/D80, not regenerated.
NEXT bounded prelaunch + owned remote activation/health, then ONE ordinaryFull;
monitor actual workers, retain failures, cleanup->table->backup. No performance
claim; all numeric/commonwarm/Resident/old-new/G1G2 and later work remain OPEN.

## Previous checkpoint — D202 physical HOST representation CPU QUALIFIED, SEALED and BACKED

Backup verified2026-10-03 19:00+08:2fc3630d684e73f5bfbae1b49f1ac937488444bf.
Push69680 CLOSED0; independent ls-remote44698 CLOSED0 exactmatch.10explicitpaths/
114staged+archiveentries scope/secretPASS; checksum/diffPASS. Usermanifest and
unrelated files excluded. No live GPU/tmux/analysis/tool sessions; remote still
stopped. Lastdisk161197293568B, freshgate before newheavy. NEXT ordinary4000Full
below; backup DONE, no need to repeat qualification or add an optimizer.

2026-10-03 18:59+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status/V1
read; Plan/status reread before qualification (one grouped output truncated,
missing span then reread in bounded chunks). Plan/V1 hashes unchanged.
optimize-for-gpu, vLLM+optimization reference, run-experiment, analyze-results,
academic-plotting and github-sync fully read; no subagents. User once-only
delivery cache authorization already fulfilled D78/D80, not recreated; no
remote operation or GPU replay this turn. Parent5fadd45fb6a8be47c7a6e39ed65d8b1f4f2ecc4d.

ONE candidate: five physical HOST consumer calls use existing graph-only
projection. All current storage/alias/dtype/pinning/staged/allocator checks,
GPU pool_tensor_views, owner, admission, equations, config and return isolation
retained; isolated worker observation/default helper full detail unchanged.
Official vLLM0.30 UniProc/model manager/CPU optimization read; D144/D174/D194
history and direct dynamic consumers audited. No stale cache/extra concurrency/
timeout change. Runtime change only gpu_monitor.ieee_gpu_reference plus helper
documentation; other runtime AST and owner/planner/admission source unchanged.

Rawresults/ieee_tc/p2_backend_qualification/d202_20261003. AllCPU3/4GiBswap0,
CPUs2,3,26,27; terminal events/swap0, scopes+paths absent/GPUempty:
- tests1 79run/2fail1error, session95372 CLOSED1,
  actualc8f34f31684140e988af5708199d6640. New fixture missed configure call and
  valid same-observation length state; tests corrected only, runtime unchanged.
- tests2 79PASS .047s, session27662 CLOSED0,
  actual7cc5a12ca100432fa248b6e58cca017f.
- regression1 1122PASS156.731s, tmux ended,
  actual1a24cf33f08e4995aa845074942a9113; peak2052943872B. Groups overlap.
- probe1 fixture namespace error before measurements, session38069 CLOSED1,
  actuala807c3ac302d4df9b0a0aa7a47d9f6db. Exact parent globals lacked mocked GPU
  pool observer. Original script retained; probe2 explicitly shares fixture.
- probe2 PASS, session74384 CLOSED0, actuald286e5a9132f4d878dfc5302e137a3f1.
- curate1 PASS, session84672 CLOSED0, actuale750606de0dd484b9793150f7097b7e9.
All failure/source snapshots retained. No production safety check weakened.

7exact-parent extension tinyCPU cases preserve all other fields/errors;
same pinningcounts6(nonempty)/14(stagedalias), description queries0. Four retained
D169 physical snapshots exact NativeSourceSnapshot/allocator equal. JSONbefore/
after1388807/550023,2025691/766427,3301339/1199835,3936172/1415212B.
HOSTonly deepcopy ms26.140382/9.928758,75.266975/14.411368,
96.688978/60.121510,111.970420/30.927552; all24blocktimings preserved.
CPU Torch2.8/three alternating blocks ten copies each, not CUDA/RPC/wallrequest,
no n3workload CI/performance acceptance. Retained-state source SHA verified.
Doc D202_PHYSICAL_HOST_REPRESENTATION.md + table SEALED;147protectedPASS.
Curated0590051fcffff3eaaed25eb107789a95a51923d49db6deccbcb909ffbb3296ae;
62member1017908B bundleb42256a2ab62dd9a266dc5e2ffc962b9943a2ce9fa783c7a323a56d024007f31.
Lastdisk161210089472B (~150.14GiB), freshgate before newheavy. No cleanup/install.

NEXT exact scoped tested backup, then ONE ordinary7B4000Full same D195/D157cap4
config/protocol/profile, fresh owned dirs, diagnostic observer OFF. No repeat
D200/D201 or another optimizer. Follow existing ordinary launcher/prelaunch/
cleanup/analysis/table. Numeric correctness/128hashchanges/commonwarm/Resident/
old-newG1G2 still OPEN. 7Bacceptance→3B→baselines; allM1M2/A1–A5/S1–S13retained.

## Previous checkpoint — D201 native-wait association COMPLETE, SEALED and BACKED

Backup verified2026-10-03 18:35+08:5fadd45fb6a8be47c7a6e39ed65d8b1f4f2ecc4d.
Push58926 CLOSED0; independent ls-remote34944 CLOSED0 exactmatch.8explicitpaths/
19staged+archiveentries passed scope/secret checks; checksum/diffPASS. No
serving edits; CPU intervalfixtures and actual retained-data analysis passed.
Usermanifest/unrelated files excluded. Both analysispaths absent/GPUempty;
no live tool sessions/remote service. NEXT one audited physical-HOST
representation candidate below; no need to repeat D200/D201 or backup them.

2026-10-03 18:33+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status/V1
reread; unchanged. analyze-results/run-experiment/academic-plotting/vLLM/
github-sync read; no subagents. User once-only remote delivery cache permission
already fulfilled D78/D80, not regenerated. No GPU replay/remote operation,
serving change or optimizer added. Current runtime still D198 diagnostics,
last ordinary Full D195. Latest evidence HEAD before backup1852c43.

Official vLLM0.30 core/UniProc/client and local call-chain audited. Synchronous
utility executes in core before returning Future; client async does not move
owner work off-thread. Same-thread Prime preparation/inventory can occupy the
query path, not proof all wait is queueing or native inference bottleneck.
Reused exact SHA-sealed D200 bounded frame reader/control qualification;
no new framework or big-log projection. Capture interval must entirely fit
[frontend_native_send,native_begin); process incarnation and main thread match.
159 unique process/sample pairs cover226/4204calls; 1boundary-ambiguous capture,
0missing/truncated in-intervalmain. HOST deepcopyline940 occurs48times
(11/12/10/15); HOST pointercheckadd103 occurs59times. NOTCPU/wall percentages,
not latency attribution/critical-path savings. Full counts and allotherframes
retained; unsampled calls not zero. 8boundaryfixturesPASS.

Association1 session32248 CLOSED0,actual85a69431dba240a883b2ebc9ff29518d,
3/4GiBswap0CPU2,3,26,27,peak109821952B/events0/scope+pathabsent.
Curate1 immediateCLOSED0,actualaf02bac61f2744e8875d216a63695e30,sameenvelope,
events/swap0/scopeidentityempty. No live jobs/tool sessions.147protectedPASS.
One read-only receipt command used wrong cwd; corrected; checksumPASS and
no analysis failure/artifact rewrite. No production tests needed (no edits).
Doc D201_NATIVE_WAIT_ASSOCIATION.md + primary exacttable SEALED.
Curatedc55ee1c3190a26724240d35ab21847187b0d314d6fc7d252bab0d54d6152604d;
verificationf8ed73d05081aa8da95fb2cc62300c35ed023584213b34bdbb885b6897bcd2dc;
11member31930B bundleccb5c2788b2a97aae291a1920d925831576bf7b810f1d3beb104391685c092eb.
Rawresults/ieee_tc/p2_backend_qualification/d201_20261003. Disk161228591104B,
freshgate before heavy task. No cleanup/deletion or environment install.

NEXT exact scoped backup, then ONE physical-HOST-consumer representation
candidate. Auditdynamicconsumers; keep fresh fullallocation/alias/dtype/pinning/
staged/allocator graph, GPU pool_tensor_views, owner/admission/formulas and
D144 mutable-return isolation. Existing host_tensor_views has no explicit
runtimeconsumer outsideproducer; remove only proven unused work, not physical
checks or stale-cache shortcut. Reuseactualsnapshots/tinyfixtures then ordinary
4000Full if qualified. D174 applied onlyrouting; D144 duplicate-pin/traversal,
D181targetscope/D194planning alreadydone, don'trepeat. No speculativeFull,
off-threadowner mutation, capacity/timeoutincrease or D197encoderrevisit.
SevenBnumeric/128hashchanges/commonwarm/Resident/old-new/G1G2remainOPEN;
7Bacceptance→3B→baselines; allM1M2/A1–A5/S1–S13retained.

## Previous checkpoint — D200 detailed 7B prefix COMPLETE, SEALED and BACKED

Backup verified2026-10-03 18:14+08:1852c43aa55b89923e23c591ecf68137e3d0c965.
Pushsession97990 CLOSED0; independent ls-remotesession81035 CLOSED0 exactmatch.
9explicitpaths/90staged+archive entries passed scope/secrets; usermanifest and
unrelated files excluded. Bundle checksum PASS; standard CSV CRLF triggered
plain git whitespace diagnostic, rerun explicit cr-at-eol check PASS with other
whitespace rules retained; sealed CSV bytes NOT rewritten. No production change
or new smoke needed; existing qualified analyzers passed actual D200 data.
All experiment scopes and GPU compute absent, no live tool sessions. Remote
owned services stopped and copied; no runtime to restart. Finaldisk161245675520B.
NEXT official/local vLLM utility scheduling/response audit, one evidence-backed
hypothesis, then minimal qualification; no speculative Full. Goal remainsACTIVE.

2026-10-03 18:11+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status/V1
reread after compaction; hashes unchanged. monitor-experiment/analyze-results/
run-experiment/academic-plotting/github-sync read. No subagents or optimizer.
Latest once-only cache permission already fulfilled D78/D80; reused, not rebuilt.

D200 ended:1000/1000 terminal/native-contract successes, zero failures, numeric
n_correct UNKNOWN. launch pass/service-replay-watchdog0, physical release true;
launchSHAc8f083b721368eb9fa0e5bf4908a81b29a9ef3d64127a68a6e3f691557577d03.
Four physical leases/4452.071693249978GPU-s. Exact local service/aux paths and
ten observed PIDs absent; GPU census empty; tmux ended. Only then exact-owned
three remote services stopped (57398 CLOSED0), allinactive/MainPID0. Transfer
journal transfers-981699c3da6f4a13a75dfb14e95c6f1b.jsonl copied with SHA check
(91954 CLOSED0). 60UUID pairs/allpublished/74953014wireB/1474429899logicalB,
requestpacking0/no byte mismatches. No remote changes during inference.
1191watchdogsamples/servicepeak18999349248B/minhost92882001920B/events-swap0,
no warning/foreigncompute/escaped workers. Remote/local runtime now STOPPED.

Reuse D187 collection/curation, only D200 identities/paths and D199 boundaries
added. No production code changes. Analysis sequential3/4GiBswap0CPU2,3,26,27:
- boundaries1 actual4f58160d4d854163afd2a6d4f5f49f3c/session15587 CLOSED0;
- metadata actual245057c1874643929bb63dd7676a3693/session31782 CLOSED0;
- frames actualb372a59811a14a1cb71db16c25bf77f9/immediate CLOSED0;
- curation actual2750a8509b5443a8aba318f8a342e493/session72138 CLOSED0.
Allterminalmemory.events0; actual identity/path absence checked in seal.
46244boundaryevents/4204attempts ALLcomplete-success, no partial/orphan/cancel/
error; count agrees with source_stats4204RPC. ALLbusiness includingdrain.
Means ms:parenttotal179.796957; frontend-send→native-begin96.328020;
nativewall12.315801/nativeCPU9.520286; native-ready→frontendreceive24.726790;
frontend-ready→parentreceive35.193664. NOT pure queue/transport times, not
request critical-path attribution; fourconcurrentRPCs not additiveasTTFT.
Tenprocessframe populations:controller/planner/fourfrontends/fourGPUcores;
6truncatedstacks/0missingmain/0partial. OccurrencesNOTCPUpercentages.
Controller537businesssamples, nearestproject decode45/fileinventory26/planning17.
No ordinaryFull comparison/n1CI/performance/model/SLO qualification inferred.

Doc D200_CONTROL_BOUNDARY_PREFIX.md and exact-stage Markdown/CSV tables SEALED.
Curated23ffb565ccfb93f74bc4283bca82b56eab417989e4eeb4b4a4626176819e683d;
verification56c032f3f2160c258fc86918ab949ab1647e0bbbee9c2895e4b760c89dee9691.
Bundle81members736572B SHAd4c8a06afaf5a0aab48158deaa9cc54c56db9d77988ee282b5af3844d2aead1d.
147protected/537prelaunchrefs PASS. Full raw request JSON not reparsed; no new
native-timeline/dispatch completeness or E2E identity claim from this prefix.
seal1 session16563 CLOSED1: path cache1 nonexistent (actual cache), before
result artifacts; original helper/log retained, no GPU rerun. Failedscope
c7a68fa362cc4e508f8b633c89185ad4 absent; terminal counters not captured.
seal2 corrected only exact archive path/new attempt, session59160 CLOSED0,
actualbf7bbe7b25954331ad39479e3b7ce662; events0/scopeidentity/path absent.
No live tool sessions. Lastdisk161249439744B; freshgate before any nextheavy.

NEXT backup exact D200 checkpoint, then read actual vLLM0.30 utility scheduling/
response + Prime caller against official docs/code and historical optimizations;
one falsifiable hypothesis before any serving change. D197 summary production
still DEFERRED; no repeat serializer study/stale confirmed-state shortcut or
speculative Full. SevenB numeric/commonwarm/Resident/old-new/G1G2 gaps OPEN.
7B acceptance→3B→baselines; entire M1M2/A1–A5/S1–S13 still retained.

## D200 launch history — superseded by COMPLETE above

17:46+08 last read:34terminal records,968control-event lines across tenprocess
files (one startup-only process has0events); not complete or success acceptance.
watchdogsample152: service14967316480B/hostavailable97511907328B/events-swap0,
no foreigncompute/escaped/guardabort. tmux live. Prior D199curate/D200cache1/
cache2audit-apply/prelaunch/health scopes all not-found/identity+path empty.
No live tool sessions; ONLY GPUdiagnostic+ownaux+three remote services active.

17:45+08 actual worker check: watchdogsample108,monotonic527952.964302317;
GPUcomputePIDs2364655/2368679/2369096/2369310 all service_member=true and
affinity4–23,28–47. Service memory17061773312B/hostavailable95008206848B;
memory.events all0/swap0/guardwarningfalse/no abort. Actual workers verified,
not merely launcher. No analysis/remote management during live measurement.

2026-10-03 17:44+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status/V1
read; full Plan/status reread before GPU launch (large grouped output truncated,
801–1537/status reread separately to EOF). Plan/V1 unchanged. No new optimizer.
D199 backup c75ef3750b4fc739f9351f713578fb2bbe9f3ca1 verified below.
D200 ONE detailed diagnostic launched in tmux tc_d200_prefix1; raw
results/ieee_tc/p2_backend_qualification/d200_20261003. Do NOT launch another.
Same D195 config/D157 cap4/60s/source42/fixedoutput/1800s; only prefix1000 and
three fresh owned paths differ. D198 boundaries + existing 2s frame observer.
Prelaunch4e04fad348e3e3f4c35c86870ac59a35592622cb9e8a1abdeff825b796d02362,
537refs/147protected PASS; prelaunchsession60394 CLOSED0, health CLOSED0.
Actual service primelora-tc-svc-5c3eb2f2a2c14d249e224f5311a2b94e.scope,
invocation03661341910f44188c81ffede5224537, inode16281283, 72/80GiBswap2.
Aux primelora-tc-aux-dfbd3ac771f348919fc489b5daaba34f.scope,
invocationb7721e1d3b674280a2825cbf4079c77d,3/4GiBswap0CPU2,3,26,27.
Deploymentnotice527844.502615776/businessstart527904.502615776.
Initial live samples: no guard events/foreign compute/escaped workers; GPU
workers still starting at this record. Need verify actual GPU processes next.
Once-only D78/D80 remote cache reused, no repacking. BothNIC1000/full prelaunch.
Remote services ACTIVE: 3Bpid1451251/id2bfcb78c0818438daa6d1a03cc42d170;
7Bpid1451253/id14ae44c2bd5f461db9227a961fe3cc57;
monitor-d200prefix1 pid1451256/ide5020271cb6c4780905090375efacf15.
7B healthclockremote-process-monotonic:d79dce1802504d40b319489cb4591e6f.
Exact remote stop/copy scripts prepared, MUST NOT execute before local terminal
and physical release; no remote operations, hash/cleanup/config change during run.
Cache2 audit/apply completed49806/45551 CLOSED0; identities
a10851c61fa24ff1b0218eab094252ba/c8d4d676fea74ce3ba990a2ce9ba76d4.
52publicwheelarchives/104files875450368allocatedB reclaimed after exact public
SHA/size/HEAD/reference/ownership/openfile checks; fourtrackedprojects unchanged,
installedenvs/models/results/custombuilds untouched. Both3/4GiB/events-swap0.
AuditSHA706b48b37ff17689f204bc15ccc16667cbce6cb2045f41665e09db264164e06d;
newlaunchdisk161602117632B above150GiB; running100GiBfloor unchanged.
Cache1 read-only9archive audit not applied, superseded. No stale deletion retry.
NEXT monitor -> exact localrelease -> exact remotestop/copy -> bounded D199
analysis + existing stack/native terminal analysis -> table/interpretation/backup.
No claim of current completeness, numericalcorrectness, SLO or performance gain.
7B acceptance -> 3B -> baselines; all later matrix retained. D200doc DRAFT.

## D199 completed — offline control-boundary analysis CPU QUALIFIED and SEALED

Backup verified2026-10-03 17:36+08: c75ef3750b4fc739f9351f713578fb2bbe9f3ca1,
pushsession51612 CLOSED0, independent ls-remotesession28448 CLOSED0 exactmatch.
9explicitpaths/42staged+archiveentries passed scope/secret check; checksum/diff
PASS. First checksum invocation used wrong cwd (read-only missing file), rerun
in bundle directory PASS; no evidence or test failure. D199 doc/results sealed.
D200 prep history: cache1 read-only10–30MiB audit ended96663 CLOSED0; only158302208B
eligible under previous package allowlist, not applied. Fresh metadata inspection
identified additional exact public scientific/build-tool wheel names; cache2
read-only audit started with expanded explicit names, same SHA/size/HEAD/ref/
ownership/open-file checks. No GPU/remote changes. Do not apply stale cache1.

2026-10-03 17:34+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status/V1
reread; hashes unchanged. analyze-results/run-experiment/academic-plotting/
github-sync read. Latest once-only remote cache permission acknowledged as
already fulfilled D78/D80; no recreation, no remote operations this turn.
Existing analyzer only: strict bounded ID/clock/process/role/ordering joins,
Type-1 complete-success summaries, explicit missing/error/cancelled/orphan and
partial-last-line counts. No serving/formula/SLO/configuration change. No
performance gain inferred; no GPU launch. D199 doc and curated evidence SEALED.
Tests1 31PASS .234s, actual98860922cc054a6aa5835a1e6cd84d8e/session86164 CLOSED0.
Regression1 71PASS .357s, actualad9f576254a844a4a8664ef676563cd5/session97731 CLOSED0.
17newtests; test groups overlap. Both3/4GiBswap0CPU2,3,26,27/events0; scope/path
absence verified by curation. Source archives retain exact test wrapper history.
Curate1 actual97257bc145c5498fa68e5c9cc77922c4/session40264 CLOSED0,events/swap0.
147protected/PlanV1 unchanged; existing analyzer bodies AST unchanged; only one
offline script changed in runtime tree. CuratedSHA
48a9a62805f40ded7a86979b7e06066eb07b1c8fbde98c7b89b37e204009d7f1.
24member79329B bundleSHA299c0c16fc0ba5b2b7dec429daecede04e266928798e17b70f960de0f14c36c6.
NEXT exact scoped backup, audited disposable-cache recovery/fresh gate, then
ONE current-source1000-prefix detailed diagnostic. Disk160749322240B last
check is below NEWheavy150GiB. Do not relax floor, repeat D198 tests, or run
speculative Full/new optimizer. 7B acceptance -> 3B -> baselines remains order;
all numerical/commonreference/old-new/G1G2 and full matrix gaps retained below.

## Previous checkpoint — D198 control-boundary diagnostic CPU QUALIFIED, SEALED and BACKED

Backup verified2026-10-03 17:15+08:
79dc8f12668c7d255846e97c7a461c95e8d580d4; pushsession52715 CLOSED0 and
independent ls-remotesession95840 CLOSED0 match. Exact13paths/70staged+archive
entries passed scope/credential checks; checksum/AST/shell/diff checks PASS.
Usermanifest excluded. D198 sealed doc/results unchanged. Final GPU census
empty; no experiment scopes/tool handles (unrelated podman untouched).
NEXT strict existing-analyzer grouping qualification below, then one detailed
prefix after fresh resource gate. No new optimizer/Full or repeated D198 tests.
Disk160754200576B below NEWheavy150GiB; no floor relaxation or data deletion.
Goal ACTIVE/current turn PROGRESS; all model/G1G2 and later work remain OPEN.

2026-10-03 17:13+08. Goal ACTIVE/current turn PROGRESS. Full Plan/status/V1
read after compaction; Plan/V1 hashes unchanged. optimize-for-gpu, vLLM and
its optimization reference, run-experiment, academic-plotting, github-sync read.
No subagents/independent semantic-review claim. User once-only cache permission
already fulfilled by D78/D80; cache reused, not recreated, no remote operation.
Current HEAD81f998c9c36f98cbbfece56807d87c0d05129a74; last ordinary runtimea0c4a9d.

ONE diagnostic-only change in logger/native worker/runner/dedicated worker:
existing qualified nonformal python_frames_v1 prefix gate now emits read-only
request-source RPC boundaries with one attempt ID/common local monotonic clock.
Default emits no fields/files. No scheduling/owner/physical inventory/formula/
configuration/timeout/SLO change, no new optimizer. Detailed code and caveats in
D198_CONTROL_BOUNDARY_DIAGNOSTIC.md (SEALED; do NOT edit). New10 CPU tests; actual
native physical computation stubbed in loopback, separate real-owner fixtures;
not CUDA/model correctness/performance qualification. Missing/failed/cancelled
events remain incomplete, not filled zero; async CPU deltas not request CPU.

Targeted40PASS49.586s, actual63d3ac2a573843668c7295e326df82f1/session66942 CLOSED0,
3/4GiBswap0CPU2,3,26,27; events/swap0/scopeabsence verified. One premature
read of terminal memory files occurred while tests still active; no experiment
failure or relaunch, terminal receipts later present. Exact source snapshots
retained in raw results/ieee_tc/p2_backend_qualification/d198_20261003.
Regression1 1100PASS156.099s, actual29cd7d6a3c03410687302bdd4caa1613,
same envelope/events-swap0; scope/path absent and tmux ended. No failed tests;
groups overlap, not independent repeats. Exact sources unchanged after tests.
Curation1 actual993191de83674736861148be55ec23de/session62949 CLOSED0,
same envelope/events-swap0/scope absence verified. 147protected/PlanV1/sourceSHA/
allowed-function AST and unchanged owner/planner/admission checks PASS.
CuratedSHA81ce5ed6e4c6de5313b4e62586cb643c3850255b5275593b520442a3549f73b8.
29member1127008B bundleSHA79210323e4539b0c9e026046fdfbd2bb69f71aac65d5ee5c22ae0237a6f924de.
Doc/curated SEALED; do NOT edit after evidence binding. Qualification table
delivered; no misleading speedup graph. No live GPU/remote/analysis/tool jobs.
No CUDA/model/numeric/performance acceptance. Exact13path backup verified above.

NEXT qualify strict bounded offline boundary grouping in
existing analyzer (ID/clock/completeness/order/cancel; no filled zero) -> audited
disk recovery/fresh gate -> ONE current-source1000-prefix detailed diagnostic.
No speculative Full/new optimizer or repeated D198 tests before new evidence.
GPU/build blocked by NEWheavy150GiB disk floor, NOT an overall goal blocker:
160772681728B last read; CPU bounded work allowed; running100GiB/memory floors
unchanged. Actual7B numerical/128hashchanges/commonwarm/Resident/oldPrime-newmetrics/
G1G2 OPEN before3B/baselines. Entire M1M2/A1–A5/S1–S13 retained.

## Most recent retained evidence — do not rerun

- D197 request GPU summary component study SEALED/BACKED81f998c9.
  Eleven cases/66 timing rows/110 negatives/7 CPU fixtures; source states identical.
  Bytes -76.925–95.274%, ten component deltas .557–.636ms, last1.146ms.
  NOT native inventory/actual RPC/CUDA/end-to-end. Production candidate DEFERRED;
  no summary implementation, repeated encoding study or speculative Full.
- D196 control-path retained-data diagnosis SEALED/BACKEDbac5f8f.
  D190→D195 preengine2101.355→1961.691ms (85.641% current meanTTFT);
  routing563.304→519.378ms, residual415.139→414.847ms NOT one RPC.
  Four source queries concurrent, not serial. Fresh control connections, not
  generation channel pool. No source-RPC boundary retention before D198.
  GPU-selected1165 conditional population differs from previous1232; don't
  infer causal improvement from those groups. Native source admission occurs
  before source→handoff1.219ms. Local ingress not remote174HTTP.
- D195 ordinary4000Full SEALED/BACKEDa42f250, runtimea0c4a9d. Complete native
  countcontracts4000/failure0; numerical n_correct UNKNOWN.
  D190→D195 meanTTFT2.433103089→2.290592642511532s, P95
  5.351091139→5.207960044150241s; P997.788593317→8.059697074699216s REGRESSION.
  MeanTPOT39.830671→40.26261691433204ms REGRESSION; P9567.727418→66.609259918ms.
  InternalE2E7.069757720→6.895221204s. PhysicalU15919.807325829→
  15915.263361513GPU-s(-.02854%), no established resource saving.
  SAME provisional D170 joint3375→3476/4000=86.9%, still324short3800.
  All15nonoutputfields identical;3872outputhashsame/128changes OPEN.
  Dispatch4000complete/conflicts0:GPU1165/HOST1773/NVMe928/remote134.
  4520source reads/4218collections/16806RPC/302joins/442stale rejects.
  2196plans=2init+2194owned;2193complete+1cancelled-discarded; CPU565.902328697s.
  132remote UUIDpairs/allpublished/131480060wireB/3300789780logicalB/packing0.
  Four physical leases released; local/remote/analysis jobs CLOSED, doNOTrestart.
  Reusable34407428B projection SHA805a6b8c5aa8f89b7394a4eba8b593f5ada95c6f3048f5a9f2d6c1c06e759697.
  CuratedSHA31594e530d42341c2eb876e18daf2d7a3f39a4c9cb728cd14f3b4bda45da4134.
  Metadata121405834B within128MiBguard; never wholeload249/361MBoriginals.
- D194 planner uses existing complete registered routing_source_snapshot,
  not target-scoped routing wire; qualified75targeted/1070regression, BACKEDa0c4a9d.
- D192 guarded encoder REJECTED; current compact messages show no consistent gain.
  D189 immutable static descriptions already implemented; no static-cache redo.
  D184 pending+GPU frontend pipeline already implemented; D181 target scoped
  routing already implemented. Don't wholesale remove legacy hints: scale-down
  still consumes queue hints. No removed physical invariants or TTL stale cache.

## Verbatim ledger archive

Previous2542-line status preserved verbatim in
docs/ieee_tc/EXECUTION_HISTORY_THROUGH_D197_20261003.md,
SHA3ff837acaf1d7ca1fe513afbf969fc12fb6034a665253004baefc50d4ae43639.
All old LIVE/NEXT text is historical; CURRENT above supersedes it.
No old docs/results/failed evidence rewritten or deleted.

## Completed history and acceptance gaps — do not repeat

Previous ledger preserved VERBATIM in
docs/ieee_tc/EXECUTION_HISTORY_THROUGH_D178_QUALIFICATION_20261003.md,
SHA3828059ffcb4822821f60acb11c3c169341e20d4031fc3a20b87d9eb6cfb6eb4.
OldLIVE/NEXT text there is historical, superseded by CURRENT above.
D152 archive d18c53f9e63b75a240dc09aa67badf8f501efc827b11dc56a8f4d48296d4aaa2
and all earlier archives remain unchanged. No old evidence deleted.

D176 held-HOST transaction qualified45targeted/947regression/8parentfixtures,
backedb4b10c1; D177providesperformanceobservation, notmodelacceptance.
D175 table/projection/backed95de3c3; D174 representation/newmeasurements done.
D172/D173 decode and D163/D164 membership qualified but no establishednetgain;
do not replay those hypotheses. D159/D160 CPUworker fusion improved waiting.
D162 independent currentcap4prefix CPUframe counts NOTpercentages:
controller538businesssamples/71JSONloads/23fileinventory/20execution_copy/
17execution_bundle; GPUHOSTinventory128/552,116/531,89/529,98/527.
D144inflightsharing,D151omitstaging,D153singlefileinventory already implemented.

D1707Bwarm1536+16measured, backed ef38f257; commonreference NOTgloballyfrozen.
Input≤616/>616 thresholds2.990091890/4.635325166s,TPOT.059646198/.085345274s.
Referencebatch8/slots8/seqs8/rank16/.92 differsfromFullcap4/slots4/rank64/.70,
disclose; do not relabel it Full or relax thresholds. 3Breferencepending.
D166/D16744snapshots19888tensors matchcheckpoint+GPUslot; NOTper-token
executionmap/fullpool/n_correct. D63looseprobabilitycontrols acceptwrongadapter
andbase, do not repeat or pretend numericalqualification.
3B500zero,7B498zero+2nonzero/4weightSHA; noauthoritynewweights.
Legacy7BmeanTTFT.564local/.657remote,3B.881local/1.087remote; contracts incomplete,
cannot reconstruct physicalU/n_correct fromCE or discounts.
Current3BD138mean12.588627/P9545.713688s stillUNACCEPTED. Newmetricold/newPrime,
numeric, commonwarm/Residentbudget/G1G2 remainOPEN, notjustpreviouscandidategain.
D145fourRPCtimeoutcauses unresolved; laterzero failures do not close cause.
Neverwholeload/reproject D137/D145/D14815GB original data.

## Authority and frozen protocols

- Read FULL /home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md and this ledger
  before task/experiment and after compaction. No subagents authorized.
- Plan1537lines SHA0c8085098ab25edb17b17362cee5a78cc84009a029196147b6f614fc26f998b7.
  Full mirror: docs/ieee_tc/PLAN_APPROVED_20261002_PRIME_FIRST.md.
  Oldfe6c05b... retained in PLAN_APPROVED_20260927_PUBLISHED_ARTIFACT.md;
  do not rewrite D152/earlier run provenance to new Plan identity.
  Execution authorization supersedes historical plan-mode/noexecution wording.
- Before comparison/configuration selection/comparative figure read FULL
  docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md,312lines,
  SHA5f0732ef54d629b40cece42bdbf536e8e74c6505e3ec5d4c7e8742408ad80f22.
  G1=allcorrect+jointSLO95% thenminimumphysicalGPU-s; G2=commonbudget+SLO,
  thenP95TTFT. CE supplementary. Warm/reference numerical thresholds NOTfrozen;
  development5000ms/liveCE are NOTformal SLO qualification.
- Preserve nineIEEE equations/core. Each optimizer: historical evidence +
  current primary literature/code + falsifiable hypothesis + minimal causal
  validation + ordinaryFull. One candidate; no future knowledge/test-point
  tuning/winner filtering, newweight/trace generation or13B.
- Eachrun cleanup→validation→table/figure→interpretation→next. Academic plotting:
  plansection11,IEEE singlecolumn3.45in,TimesNewRoman,nooverlap,below bold(a).
  Failure/status tables appropriate; no n1CI/rankingfabrication.
- Chinese updates≤60s, paper/evidence terms, completed/current/outstanding.
  Backup testedcheckpoints to faaslora_origin/retry14_continuous_queue_v2.
  Noforcepush/credentials/rawlargeGit. Baseline changes inownrepo.

## Current mainline and reusable evidence

- BaselinesPAUSEDHEAD9e2cf28903ed11bc8ee891dd4cc9636b94307573. D74Serverless3B
  repaired complete/analyzed/plotted/backed; preparedoriginal NEVERrun.
  ResumeafterPrimeFull:Serverless→vLLM→S-LoRA→dLoRA3B→Loquetier→HydraServe.
  DisplayServerless, not-new. AllformalM1M2,ablations,sensitivities outstanding.
- Numericadapter distinction pending:3B500/500zero,7B498/500zero,2/4distinct
  weightSHA. Nativecount≠numeric proof; no authoritynewweights,don'trepeatedlyask.
- D122 diagnostic-prefix contract optionalFAASLORA_TC_DIAGNOSTIC_PREFIX_COUNT:
  explicitindexview/sourceSHA/count/defaultFullguard; formalprefix forbidden.
  521finaltestsPASS,backedd4825f6. Not a servingoptimizer; doNOTrepeat tests.
- D121 actualsavedstateCPUcopy/JSON/SHA component only,backed716e6a5; D120
  D119queue/conditionaloccupancy,backeda7d8673. Neitherproveswholequeuecause.
- D1197B4000:2890native-success,1086Timeout+24returnedownershiperrors;
  22866.332606GPU-s,61leases/57quarantinesreleased. Firsttimeout precedes
  ownershipfailure; earlierbacklog cannot be attributedsolelytolaterquarantine.
  Original9.48GB NEVERwholeload/reprojectagain. Boundedprojection29MB reusable.
  FinaldocD119_FULL_W0_FULL1.md,backed4ee00af.
- D1183B4000:all4000nativecontractsuccess,0failure,15866.211830GPU-s,
 4leasesreleased/0quarantines; largewaiting,notnumericadapter/SLOqualified.
  Original9.18GB neverwholeload. Backedb4c194b.
- D11748f808 avoidsIEEE legacyterminalscale-down; settlespreparationownersbefore
  retire. D1160ef466c non-targetmembership fix. D1142697758 actualasyncTCP avoids
  shared32-executor starvation. D1115442e62 parses samein-flight snapshotonce.
  Alreadyinruntime; no repeatedobsoleteprobes or safetyguardremoval.
- D115 all4000native butshutdownfailure; ownedHOST/NVMe roots remain RETAINED:
  /dev/shm/tc-d115-3b-full10/tc_ieee_full and d115_20260929/3b_nvme_full10/tc_ieee_full.
  DoNOTdelete or claimworkspacecleanup. Historicalquarantine/error evidence kept.
- D88profiles3B368/368,7B92/92;D89initializers
  paper_results/ieee_tc/p2_backend/d89_{3b,7b}_initialization/manifest.json.
  Use requested_model_config PARENT,notresolvedchild; no reprofile.
- cap2 historyApr13/Apr22maxloras8 KV differsfromcurrent7Bmaxloras4/.30;
  no cap4qualification. DoNOTblindcapacitytune orraise1800squalificationdeadline.
- Prior warmerreference interface audit WARM_REFERENCE_REUSE_AUDIT_20260930.md.
  D88/D89 profiles are not commonwarmSLO samples; don'tinflatewarmreference with
  Primeprotected-source/controlwall. Reference protocol remains pending.

## Environment, remote and safety

- Repo /home/qhq/serverless_llm_experiment_retry14_baseline,
  branchretry14_continuous_queue_v2,remotefaaslora_origin. Baselineownrepo/main.
  results symlink=/home/qhq/serverless_llm_experiment/results.
- Nativeenv /home/qhq/.venvs/primelora_vllm0300_tc_20260925:
  vLLM.30/torch2.13/CUDA13/Python3.12.12/SM86.
  CPUenv /home/qhq/anaconda3/envs/LLM_vllm0102/bin/python;
  OSguards/usr/bin/python3 (modelenvlacks pidfd_send_signal).
  py-spy/home/qhq/anaconda3/envs/splitwise_official_20260615/bin/py-spy.
  Noinstall/driver/globalptrace change.
- Current3Bcap8/slots8/cpu32/gpu.72; current7Bcap4/slots4/cpu24/gpu.70;
  7B D157 service/preparation/length profiles replace oldD89cap2 timings.
  HOST16GiB/native2/NVMe16,W5/movement3,scale2s/beta.5,min1max4.
  3Bservicebin28.114717726756954ms,7B27.4336ms. Developmentonly.
- Remotealiasprimelora-artifact-174 lab14@192.168.4.174:8122,strictBatchMode,
  key~/.ssh/primelora_artifact_174_ed25519_20260925(0600);
  token~/.config/primelora-tc-d75/artifact.token NEVERprint/stage.
  FingerprintSHA256:wkvfU2qJWd6V7TCPYpot5PHXJlgBChI03Npu4y7bb40.
  Artifactonly, noGPUinference. HTTP3B18080/7B18081 direct,noSSHdatatunnel.
- Immutablecache/home/lab14/primelora_remote/tc/d78_20260927/published/{3b,7b},
  both500/500,total1581215261B. Once-onlycreationfulfilledD78/D80; NEVERrebuild.
  Localeno1np0/remoteeno1 1000/full verifiedbeforeD123; historical100Mbps
  notcurrent. NoagentNICchanges. Actualtransfers/competition retained,packing0.
- Service72/80GiB/swap2,CPUs4–23,28–47;aux/CPUanalysis3/4GiB/swap0,
  CPUs2,3,26,27. ONEheavyjob; limitactualworkersBEFORElaunch.
  No global kill/ray-stop/reset/reboot. Only exact-owned identity cleanup.
  No remoteconfig/restart/hash/cleanup during inference.
  Inference disk150/100GiB unchanged;artifactnodeindependentincrementalrule.
- Contentindices inputs/20260927_3b_remote_content_index.json
  SHAbd1826c58f00ea30dee1d3829dc11727a975127db701a1eee6c6c86b4a38f275;
  inputs/20260927_7b_materialized_content_index.json
  SHAe85cce3c3611da15530f9e663549231d3493282c85913344fe41c2a67044260c.
  Do not use misnamed7b_remote_content_index diagnostic4998filelist.

## Protection and archives

- Seal147protected:paper_results/ieee_tc/safety/20260925_execution_start_protected.json,
  SHAfa8f001aaa139017762a1cc7e3cb8d090f9d28483166724947246594e0d6a2a6.
  Verify via scripts.ieee_tc_preflight.verify_seal; oldfinal_v2/figs/paperunchanged.
- NEVERstage configs/generated/lora_manifest_1000.json,unrelatedAAAIarchive,
  oldfigures,regenerate_motivation_figs.py,rejectedpreviews. Explicitpaths only.
- Prior1439-line ledger archived VERBATIM in
  docs/ieee_tc/EXECUTION_HISTORY_THROUGH_D123_20260930.md,
  SHAdf8bc100fe6eaa4bb00ccbc857b7a469ecaca5e56e4ce28f6f76b883cd692c70.
  Historical LIVE/NEXT handles are superseded, not commands to execute.
- Earlier THROUGH_D110,D91_D100,D81_D91,D78_D80,D67_D77,THROUGH_D26,D27_D52,
  D53_D66 archives remain intact. Currentstate above has priority.
