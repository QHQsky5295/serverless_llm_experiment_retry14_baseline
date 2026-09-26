# IEEE TC execution status

## Authority and recovery

- Authoritative plan: `/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md`
- Approved snapshot: `PLAN_APPROVED_20260925.md`
- SHA256: `a9ed2d8136073f5a5d5c8088489e1d8642b916a610fc7e9d3b3ff8eaf35301b0`
- User authorized execution, milestone Git backups, and recurring paper-oriented
  progress reports after approving the plan. Never infer execution completion.
- Main starting commit: `7ebb1ca9688362736710d94db78ec416a88a2384`.
- Baseline starting commit: `d85263e00e976cc61c8938e662d77a2b249cdf58`.
- No running tmux experiment was present at initial execution inspection.

## Mainline ledger

| Block | Status | Evidence / next action |
|---|---|---|
| Approved plan persistence | Recorded | Full snapshot + source hash + AGENTS instructions |
| Resource containment / safety | Native Ray inheritance, external replay and7B dedicated physical owner/exit witness tested; full-deployment gates pending | RESOURCE_QUALIFICATION.md + EXTERNAL_REPLAY_QUALIFICATION.md + PHYSICAL_GPU_MEASUREMENT.md D27. Other owner paths and Full lifecycle aggregation remain open. CPU proof is affinity, not delegated cpuset |
| Protected historical artifacts | Sealed and verified | `paper_results/ieee_tc/safety/20260925_execution_start_protected.json`; old results and selected user modifications unchanged |
| Remote authentication / management | Key login verified; service qualification pending | Strict host checking, dedicated restricted key; remote disk 138.9 GiB below 150 GiB floor, user decision pending. See REMOTE_ACCESS.md |
| P0 main-table / Full provenance | 7B source-pair audit complete | Same trace/subset SHA, different execution; 202 scalar fields preserved. P0_FULL_PROVENANCE.md; no performance rerun needed for this finding |
| Serverless wait audit | Historical audit + no-GPU control-path tests complete | Clean 7B/3B logs reused; six real-method AST tests pass; incremental ready-before-wait patch preserved. Model pair pending |
| P1 IEEE semantic alignment | Explicit file→native HOST connected; automatic Full open | P1_FORMULA_IMPLEMENTATION.md D1–D41. CPU-only preparation retains file/source identity and allocator-accounted tensor sub-budget, never implicitly activates GPU. Full HOST budget, automatic planner/handoff, file-tier replacement and complete lifecycle remain open. IEEE startup still rejects legacy warmup; no new GPU/Full qualification |
| P2 backend qualification | Mechanical sequential/batch/cancel evidence retained; full-pool content scan complete; semantic qualification OPEN | ARTIFACT_CONTENT_AUDIT.md: current 3B 500/500 all-zero, 7B 498/500 all-zero, only 2/4 weight SHAs respectively. 3B native same-prompt/different-SHA outputs identical; zero controls cannot distinguish adapter application. No formal/remote performance qualification |
| Baseline qualification | Pending | Serverless, vLLM, S-LoRA, dLoRA 3B, Loquetier, HydraServe |
| M1 / M2 | Not started | No new formal performance claims |
| A1–A5, S1–S3 | Not started | Shared frozen policies required |
| S4–S13 + offline supplements | Not started | Follow core experiments |
| Documents / final curated figures | Not started | Per-run figures/tables start with the first evidence block |

## Reporting and backup cadence

- While actively working, concise user updates at least once per minute.
- After each completed experiment group and at least every 30 minutes of active
  work: completed evidence / current research question and reason / remaining work.
- Tested milestone commits: safety/protocol, Serverless audit/repair, IEEE + backend,
  baseline qualification, each main/ablation/sensitivity evidence block.
- Stage named task files only. Verify pushed remote commit IDs. Preserve failures.
- Resume by reading the plan and this ledger, inspecting processes and current Git
  state, then doing the next unmet gate; do not restart live work from stale notes.

## Pre-existing changes (do not stage)

Main: generated LoRA manifest, AAAI review archive/directory, old figs/paper
untracked artifacts, scripts/regenerate_motivation_figs.py.

Baseline: scripts/replay_openai_trace.py,
scripts/run_serverlessllm_relayserve_continuation.sh, cache/, installs/, repos/,
configs/relayserve_motivation_serverlessllm.yaml.

## Immediate next actions

LATEST IMPLEMENTATION: D41 connects confirmed file→native CPU-only loading,
registration and source pins through the actual runner/common queue/worker.
It checks an immutable accounted-tensor sub-budget using process pinned allocator
retention, reachable pageable tensors and conservative checkpoint/cast/pin peak.
This is explicitly NOT total HOST memory. Cache-full preparation defers; no
implicit LRU or GPU activation. Cancel joins actual reads/releases; lost reply
retains source and shared pressure. Pressure-finish wakes GPU, not its own HOST
budget deferral. D40 shared/preactivation ownership remains. No new model/profile.
Next integrate total HOST/native/file/workspace ownership with automatic planner
options/handoff and proactive d feedback; do not repeat this narrow CPU path.
Final frozen-source822 functional and56 safety checks pass. All13 new checks
and11 native-hook checks pass under installedvLLM0.30, CUDA uninitialized.
File-tier pending protection/multi-victim replacement and full per-replica
physical lifecycle remain open. D39 native plan/queue invariants are retained.
Queue wakes on actual reference release/file-pressure finish/target closure,
not yet direct native iteration notifications. _preload_full_stack still
rejects ieee_confirmed BEFORE legacy work. Do not bypass or repeat this gate.

LATEST COMPLETED MODEL RUN: `llama2_7b_capacity_wait_attempt2`, finished and cleaned.
Five first-distinct requests from old32-prefix,755 native tokens, all targets met.
The fifth request waited3764.347ms for a real referenced slot; release witnessed,
state re-observed, generation completed and all leases returned. Prompt/native
input/output SHAs match corresponding D26 source32 rows. Native timing
recomputation errors0ms. Controlled ownership evidence, NOT Full/remote/SLO or
performance qualification. Do not repeat. Curated CSV/JSON and P1 D28 table saved.

PREVIOUS MODEL RUN: `llama2_7b_physical_lifecycle_attempt3`, finished and
cleaned. Four original7B requests/551 tokens; normal parent exit0, actual native
worker exit event, UUID census and physical lease return.60.738 GPU-s including
7.642s after the last token. Attempt1 censored; attempt2 forced-exit with48.769s
tail; both preserved. Same request/adapter/prompt/output hashes across all three.
Do not rerun this prefix. Curated CSV/JSON + status table delivered. No formal
performance or complete Full lifecycle claim. D26 source32 remains completed;
its one historical SHA difference and independent numerical gate remain open.

0. P2 installation finished; do NOT reinstall. All current runs finished/cleaned. Native adapter reference, full 3B/7B content scan and 7B five-arm native numeric diagnostic COMPLETE; do not repeat same-prompt/zero controls. NATIVE_ADAPTER_NUMERIC_CONTROL.md: all five 217-token outputs identical; A/Z probability max difference .02608, A/A also .006367. Descriptive evidence, not independent numeric/full semantic qualification. Return to Full source/cost and owner integration next. Current 3B pool has 500 all-zero LoRAs (2 SHAs); 7B has 498 zero plus finance/medical nonzero (4 SHAs). Plan forbids new weights; user confirmation requested before adding a few trained nonzero 3B correctness fixtures, no answer yet. Keep pools/traces/history intact. Targeted 2,521 existing configs yielded no 3B candidate, not proof of server-wide absence. Independent numerical verification remains an open gate, not erased by this diagnostic. Review official 0.30 warning on old 3B chunked_prefill=false before freezing. Reuse environment/cache, KEEP_DEDICATED_WORKER_LOGS for future proxy runs.
1. D29 connects E(t) to native core/worker HOST→GPU; D30 qualifies rank-sliced direct copies. D31 completes pending→native demand identity handoff. D32 binds actual HOST/NVMe file migration to verified content, preallocated peak space, read references and joined cancellation. D40 extends D33 pressure to actual shared/preactivation file-owner intervals and native startup joins. Next integrate total HOST/native tensor budgets, actual IEEE planner/handoff/replacement and measured class initialization. Do not repeat source32/capacity5/lifecycle4, pitched-copy or transfer-only checks.
2. D27 now qualifies the7B dedicated runtime's physical allocation and normal exit. Do not re-run its four-request prefix. Connect owner coverage and aggregation for the actual Full deployment (shared/direct/multi-runtime paths are not qualified here), while completing physical tier admission and representative measured profiles. Qualify Serverless native checkpoint path before its original/repaired model pair.
3. Continue remote setup after its disk gate; no heavy GPU run
   until actual process containment and watchdog gates are satisfied.

## Latest verified backups and evidence index

- Main tested implementation: `34d8fabab094bd49201f9ea000fc54f8ec051ec3`, pushed to
  `faaslora_origin/retry14_continuous_queue_v2`; remote SHA verified.
  This backup receipt is a subsequent documentation-only commit.
- Baselines: `16570c023a439c884624e7a5bdfa0d8577faf7a3`, pushed to
  `origin/main`; remote SHA verified at its checkpoint. No baseline edits in D28.
- All earlier checkpoints, failures, qualification attempts and exact backups
  remain verbatim in [EXECUTION_HISTORY_THROUGH_D26.md](EXECUTION_HISTORY_THROUGH_D26.md).
  Archive SHA256: `4e91643d959967f1f0dc8642fe969d35380f2156cf8c820964019539d3b415cb`.
  This is archival organization, not new experimental evidence.
- Read the relevant historical section before each optimization; do not treat
  old chronological “next” items as current instructions.
- P1 formulas / native sources / ownership: `P1_FORMULA_IMPLEMENTATION.md` D1–D41.
- Actual model qualifications and source32 table: `P2_BACKEND_QUALIFICATION.md`.
- Full-pool SHA / zero weights: `ARTIFACT_CONTENT_AUDIT.md`.
- Native five-arm limits: `NATIVE_ADAPTER_NUMERIC_CONTROL.md`.
- Safety / external replay: `RESOURCE_QUALIFICATION.md`,
  `EXTERNAL_REPLAY_QUALIFICATION.md`, `PHYSICAL_GPU_MEASUREMENT.md`.
- P0: `P0_FULL_PROVENANCE.md`; Serverless: its dedicated audit docs and
  `paper_results/ieee_tc/serverless_audit/` (inspect actual paths before use).
- Remote key access, management and floor: `REMOTE_ACCESS.md`.

## Current evidence and non-negotiable open gates

1. Actual 0.30.0 / torch2.13 / CUDA13 environment installed once:
   `/home/qhq/.venvs/primelora_vllm0300_tc_20260925`.
   Stable no-GPU tests use `/home/qhq/anaconda3/envs/LLM_vllm0102/bin/python`.
   Do not reinstall, duplicate weights/traces, upgrade driver or clear compile caches.
2. Both models’ old100 sequential prefixes reach all fixed targets; ordinary
   batch pairs and 7B cancelled-RPC ownership qualify their narrow cases.
   Full500 execution, numerical correctness and remote main qualification remain open.
   Earlier failures and 3B cancellation output differences remain preserved.
3. D26 real7B32: 5967 tokens;28 GPU/4 HOST;16 explicit first-miss priming loads.
   Source-admission acquisition/token events and references pass; both recomputation
   errors0ms. HOST D 93.910/103.719/108.989/118.300ms includes control/loading,
   not pure H2D. One output SHA mismatch at req_00005; cause unestablished.
   No ready-time cost profile, SLO, system ranking or Full qualification follows.
   Artifacts: `paper_results/ieee_tc/p2_backend/20260926_7b_source32.{json,csv}`.
4. Full received-view routing and actual D25 source protection exist. D32 adds
   content-bound, budgeted multiple file paths. Total HOST/native tensor budgets,
   proactive planning/handoff connection and complete
   deployment physical GPU lifecycle still require integration. Do not silently
   call the legacy planner or partial file sub-budget IEEE Full.
5. Actual profile measurements must cover their frozen class/configuration and
   environment; neither test constants nor neighboring-class latency can fill gaps.
   Replica inheritance is from frozen measurements, never a previous test block.
6. D27 adds opt-in dedicated-runtime physical allocation/return journals and
   actual7B qualification. Full/all-baseline owner binding is NOT complete.
   Physical GPU possession is not GPU utilization,
   instance “ready”, request completion, empty_cache or shutdown return.
7. Serverless official one-second ready-path polling is supported by old raw
   logs and6 method tests. Original/repaired1000-request model pairs remain pending.
   Native checkpoint loader path and contained launch must qualify first.
   Existing broad Ray-stop/pkill/global-tmux launcher is not safe for TC.
8. Remote key-only login works, last free disk137.91GiB below150GiB floor;
   services not qualified/started. Do not delete unique/unrelated data or silently
   relax the floor. Additional nonzero3B correctness artifacts need user authority.
9. Physical CPU isolation proof is actual affinity, not unavailable delegated
   cpuset. Services72/80GiB high/max, swap2GiB; auxiliary4GiB; one heavy at a time.
   Read source plan for exact admission, disk, failure/timeout and statistical rules.
10. All147 protected entries and source plan SHA verified unchanged after D26.
    D26 functional653 and safety/census/replay56 tests pass, no skips/failures.
    Final D26 GPUs15MiB/0%, MemAvailable108586104KiB, disk353938874368 bytes.
    Scope `primelora-tc-aux-adff572906f44b3c9872a65ecab8feb8.scope` verified empty
    and stopped. Recheck live handles/resources; a stale record is not a live run.

## This continuation

### D41 explicit file→native HOST; total HOST qualification remains open

- Actual runner/common queue binds the existing confirmed file SHA/read lease
  to a CPU-only native load/registration/pin. It keeps official parsing/mapping/
  packing/scaling but never invokes the worker's GPU-activating add_adapter.
  CPU-full returns host_replacement_required rather than hidden LRU. Registered
  native source is published only after load; GPU slots remain unchanged.
- Process pinned allocator allocated/active/cached counters now accompany actual
  worker/source observations. Registered pinned tensors are not double-counted;
  missing statistics are unknown. Explicit immutable tensor sub-budget charges
  reported pinned retention, registered pageable storage, source file and
  conservative converted/pinned tensor peak. This is NOT total HOST physical
  memory: allocator overhead/pageable retention/reserved segments, files and
  all other workers still need unified ownership. Demand policy unchanged.
- Cancellation joins real reader and pin release before file release. Lost native
  replies retain file/CPU ownership and shared pressure, without fabricated
  IO-joined/finish timestamps; retirement rejects unsettled state. Pressure-finish
  only wakes GPU E(t), not a self-triggered retry of the same HOST budget check.
  Full guard remains. This is explicit empty-CPU-slot preparation, not automatic
  Full, complete HOST replacement or physical memory qualification.
-13 new checks. First targeted151 had one fixture error: preservedvLLM0.10 request
  lacks load_inplace; explicit0.30 request fixture fixes the test without changing
  production semantics. Initial full821 passed, subsequent822 passed; final
  frozen-source regression and safety receipts are recorded below. Native0.30:
  all13 new checks pass, CUDA uninitialized, actual LoRARequest schema checked.
  An extra native-hook selector was initially misspelled (NativeCoreHookContract),
  generating one FailedTest; corrected NativeHookWiring passes11 separately.
  Thus24 actual native-environment checks pass, not14+11 successful checks.
  No GPU/model, real174, new profile, numerical correctness or performance run.
  Required correctness table and primary-source rationale are in P1 D41.
- Final frozen-source822 functional and56 safety tests pass, no failures/skips.
  All seven owned D41 scopes verified TasksCurrent0, empty cgroup.procs and
  high/max/OOM0, then stopped. GPUs15MiB/0%, MemAvailable108726280KiB,
  disk353576480768B after cleanup. Source plan SHA and147 protected entries
  unchanged. No new weights/traces, no historical-result changes. Baseline
  repository remains16570c023a439c884624e7a5bdfa0d8577faf7a3, no D41 edits there.
- Next: unify total HOST files/native/allocator/workspace budget and remaining
  option production, automatic handoff/planning, HOST/NVMe pending protection/
  multi-victim replacement, proactive d feedback and full physical lifecycle.
  Native tensor sub-budget must not be relabeled total HOST. Do not repeat this
  local path or old model prefixes. Formal baselines/M1/M2/ablations/sensitivities
  remain not started. Goal active; outstanding remote disk and3B fixture authority
  gates remain unchanged. Preserve both repositories' pre-existing user dirt.
- Implementation34d8fabab094bd49201f9ea000fc54f8ec051ec3 pushed to
  faaslora_origin/retry14_continuous_queue_v2; full remote SHA verified. This
  subsequent backup receipt is documentation-only. All seven owned D41 scopes
  are stopped, no model or test job remains live. Recheck plan/status/resources
  on continuation, then integrate total HOST ownership and automatic planning,
  not another isolated native-HOST/pressure or old-model-prefix qualification.

### D40 shared/preactivation file-owner pressure

- The actual file preparation path uses one service-owned physical file domain.
  All initialized participants acknowledge start before IO; engine activation
  replays active intervals before warmup/pool/GPU preparation. Duplicate logical
  slots do not multiply transfers. Core binding, current native observation and
  worker E(t) retain exact domain/clock identity; no formula or profile changed.
- Notifications serialize under one domain lock; file operations remain concurrent.
  Joining/finish races and repeated cancellation settle actual RPCs. Lost attach
  stops IO; a lost finish keeps that owner's pressure while other owners settle.
  Actual slot removal joins its GPU plans, then existing file pressure, then
  invokes shutdown. This is not complete physical GPU lifecycle qualification.
- Initial targeted144 passed. First full807 had two legacy smoke fixture errors:
  model_cfg was absent on __new__-constructed runners. Fixtures now explicitly
  provide empty legacy configuration; no production missing-field fallback added.
  Initial12 new checks plus extensions; full808 passed, no failures/skips.
  InstalledvLLM0.30:12 new +11 native-hook checks pass; one further worker test
  initially had an invalid selector (NativeProactivePreparation). Correct selector
  NativeProactiveTransactions passes separately:24 actual checks pass in total,
  CUDA uninitialized. The selector error is not a backend/runtime failure.
 56 safety checks pass. Initial six owned D40 scopes checked TasksCurrent0/empty pids,
  high,max,OOM0, then stopped. No check process or GPU context remains.
  P1 D40 contains the correctness table and primary-source rationale. No GPU/model,
  real174, numerical adapter, representative profile or performance run occurred.
- Source plan SHA and147 protected entries checked unchanged after final tests.
  Final GPUs15MiB/0%, MemAvailable109966216KiB, disk353608052736B. No new
  weights/traces, no old-results mutation. Initial implementation79e0a6e6343d8b6cf9c4bd45f1bdc7c2714ed19c
  pushed and remote full SHA verified.
  Baseline repository remains16570c023a439c884624e7a5bdfa0d8577faf7a3. Preserve user dirt.
- Closure review found that cancellation after a native join acknowledgement
  must retire that subscription BEFORE shutting down the new engine. Actual
  scale-out cancellation now joins old file pressure; the other replica's copy
  still completes. One additional real-entrypoint check, total13 new checks;
  final809 functional pass. All13 new checks pass in installedvLLM0.30 with CUDA
  uninitialized. Both additional owned scopes checked empty/high,max,OOM0 and
  stopped (eight D40 scopes total). Final GPUs15MiB/0%, MemAvailable108830624KiB,
  disk353606467584B; plan SHA and147 protected entries unchanged. No running job.
- Next: total HOST/native budgets and file→native HOST, automatic options/handoff,
  proactive d feedback, file pending-target/multi-victim replacement, complete
  Full physical lifecycle. Full guard stays; baselines/M1/M2/ablations/sensitivities
  remain not started. Do not repeat the shared-pressure or old model-prefix gate.
- Final implementationc27351347c1685070514e39a028c6b4ef8185939 pushed to
  faaslora_origin/retry14_continuous_queue_v2; remote full SHA verified. This
  subsequent receipt is documentation-only. All eight D40 owned scopes stopped;
  only original user dirt remains. Goal active, not complete or blocked. Recheck
  plan/status/resources, then continue total HOST/native ownership and automatic
  planning integration. Do not repeat D40 or old GPU/model prefixes.

### D39 selected native GPU plans on the shared queue

- Actual runner→core/worker/owner path registers the complete selected set before
  execution. Targets exclude proactive victims. Frozen h/d and objective SHA
  remain unchanged across the plan's own slot/reference changes; every attempt
  checks actual source/fallback, references and E(t). GPU-ready reuse is fenced,
  not a made-up load. Native demand remains native policy. Full guard remains.
- Multiple plans reuse one physical job. Creator cancellation retains its plan
  until other subscribers' operation settles. Lease release precedes target
  finish/close. Lost native replies retain unknown ownership; proxy clears only
  exact plan/owner preparation uncertainty. Actual shutdown joins these plan
  tasks before file queue and runtime removal. No blind retry or cache fallback.
-11 new checks. Targeted130, initial794 and final796 functional plus56 safety
  pass, no failures/skips. Actual installedvLLM0.30 final22 (11 new+11 core hook)
  checks pass, CUDA uninitialized. Earlier installed20 also passed. These use
  native cache/worker fixtures, not CUDA/model/real remote or profile measurements.
  P1 D39 contains the correctness table, source references and exact scope.
- Six owned D39 scopes verified TasksCurrent0/empty pids/high,max,OOM0 and stopped.
  Final GPUs15MiB each, utilization0/0/1/0%; MemAvailable108921392KiB,
  disk353628155904B. All147 protected entries and source-plan SHA unchanged.
  No new weights/traces, GPU/model run or old-results mutation. Baseline remains
  16570c023a439c884624e7a5bdfa0d8577faf7a3; pre-existing user dirt is preserved.
- Next: shared/preactivation pressure, file→native HOST and total HOST/native
  budget ownership, automatic options/handoff and proactive d feedback. File
  pending-target/replacement, per-replica retirement and complete Full lifecycle
  remain open. Native iteration notifications are not yet queue wake events;
  current release/transfer/target events provide actual rechecks. Do not repeat
  this local gate or source32/capacity5/lifecycle4. Baselines/M1/M2/ablations/
  sensitivities remain not started. Full goal active, not complete or blocked.
- Implementation391f1613fecc573c82667179f286d98092d842b4 pushed to
  faaslora_origin/retry14_continuous_queue_v2; full remote SHA verified. All six
  D39 scopes stopped and no longer listed; only original user dirt remains.
  This backup receipt is documentation-only. Recheck plan/status/resources,
  then continue shared/preactivation pressure and total HOST/native ownership
  with automatic planner integration. Do not repeat this narrow qualification.

### D38 actual shared file preparation; automatic Full still open

- The existing PreloadingManager now owns a common event-driven movement queue.
  Actual request HTTP misses and explicit handoff/residency HOST/NVMe paths use
  one physical operation per owner/tier/adapter/content. Demand precedes pending
  proactive work; density/identity are stable; running work is not preempted.
  Actual owner checks/rechecks storage, source SHA and publication. Future results
  never replace fresh source observations. No new planner formula or downloader.
- Cancellation withdraws one interest; last-interest cancellation joins actual
  file work and native pressure settlement. Concurrent shutdown also joins, then
  removes engines. Failed/uncertain work is not automatically retried. Deferred
  queue work needs a same-owner state event. Only the creator gets Remote cost
  evidence; per-intent metadata and attempts are recorded in actual summary.
-12 added checks; targeted166, initial/final785 functional and56 safety pass,
  no failures/skips.12 also pass under actual installedvLLM0.30, CUDA uninitialized;
  subsequent concurrent-close assertion is covered by final785. Real file and
  HTTP-response fixtures, not actual174 service, CUDA/model, representative
  profile or performance evidence. P1 D38 gives correctness table and sources.
- Five owned D38 scopes verified TasksCurrent0, empty cgroup.procs and
  high/max/OOM events0, then stopped. Final GPUs15MiB/0%, MemAvailable108974684KiB,
  disk353645719552B. All147 protected entries and source-plan SHA unchanged.
  Baseline repository remains16570c023a439c884624e7a5bdfa0d8577faf7a3; original
  user changes are unstaged. No new weights/traces or old-results mutation.
- Full guard retained. Automatic handoff/options, native GPU queue and pending
  target protection, HOST/NVMe replacement, total HOST/native accounting,
  shared/preactivation pressure, per-replica retirement and Full physical
  lifecycle remain. D38 initialized-target file pressure is not global pressure.
  Do not treat queued file correctness as Full qualification or repeat this gate.
  Formal baselines/M1/M2/ablations/sensitivities remain not started; goal active.
- Implementation06132c59a0ebbd527afa830cdc95b97d39d1b60e pushed to
  faaslora_origin/retry14_continuous_queue_v2; full remote SHA verified. No D38
  scopes remain listed; only original user dirt remains. This receipt is a
  documentation-only commit. Reread plan/status and inspect live resources next,
  then continue integrated GPU movement/shared pressure/total-budget ownership,
  not another isolated file queue or model-prefix qualification.

### D37 native GPU objective replacement; Full queue still open

- Actual stack freezes one ingress demand window and one preparation-cost
  snapshot with complete native source/footprint/slot evidence. Unknown or
  unconfirmed sources and missing positive-demand classes reject. Zero-demand
  missing classes carry null d, not an invented zero latency. Hash is message
  integrity, not measurement/numerical correctness proof.
- Actual core accepts the objective; worker contributes live pending demand
  protection and checks real uniform slot bytes. Owner chooses loss/usable-byte
  victim with stable identity tie-break, checks strict positive net benefit,
  then evaluates E(t). Deferral leaves caches untouched. Accepted GPU-only
  reclamation preserves native HOST fallback and uses the existing copy/fence.
  Failed copy or lost fallback poisons the owner, not a claimed rollback.
  Full/CapacityOnly share replacement; ordinary demand remains native LRU.
-13 new tests. Targeted114, first full772, final773 functional +56 safety pass.
  Exact-tie fixture refined to exactly representable1.5; fallback post-copy
  invariant added. All13 new checks pass under actual installedvLLM0.30 with
  CUDA uninitialized. These are native-cache/worker fixtures, not CUDA copy,
  model latency, real representative profiles or Full performance evidence.
  P1 D37 includes the required correctness table and primary-source rationale.
- Five owned scopes checked TasksCurrent0/empty pids/high,max,OOM0. Final GPUs
  15MiB/0%, MemAvailable109057500KiB, disk353680191488B. All147 protected entries
  and plan SHA unchanged; baseline remains16570c023a439c884624e7a5bdfa0d8577faf7a3.
- Next is common owned movement/handoff queue, pending-target/transfer reuse,
  total HOST/native accounting, shared/preactivation pressure and full lifecycle.
  This single uniform GPU-slot transaction is NOT HOST/NVMe multi-victim or
  automatic Full integration. Representative profiles, numerical correctness
  and remote/backend qualification remain open. No baselines/M1/M2/ablations/
  sensitivities started. Do not repeat this local gate; full goal remains active.
- Implementation fdf87d59f5fa99fbdb6d4fb444bc3b9bd1fc22a1 pushed to
  faaslora_origin/retry14_continuous_queue_v2; full remote SHA verified. All five
  D37 scopes stopped after empty checks; none remain listed. Only original user
  dirt remains. This backup receipt is documentation-only. Recheck plan/status
  and live resources before continuing the common owned movement integration.

### D36 actual preparation profile/request feedback connection

- FrozenPreparationProfiles recomputes d from raw source-admission/native/remote
  boundaries, binds file SHA/model/environment/resource/input context, rejects
  duplicate request/lease and incomplete/reused sources. Prior-boot profile
  clocks are compared only within their original measurement. It is a data
  contract check, not independent proof of backend/adapter numerical correctness.
- Exact-content layout partition conservatively shares only identical verified
  stored content, further split by source representation and size bin. Actual
  confirmed_source_class now preserves observed footprint/representation. No
  adjacent-bin, rank-only or service-D substitution. No new real profiles created.
- Real runner loads preparation configuration; actual InstancePool initializes
  each runtime from frozen means, validates runtime before warmup, and reports
  profile identity. Preparation/service measurement contexts must match.
  Actual source admission fixes the preparation class before hold/load RPCs;
  actual completed loading updates that slot/class. GPU hits do not invent loads.
- Nine added tests. Initial targeted201 had two test-assumption errors: missing
  class propagates KeyError rather than returning a failed result; tiny1024B
  budget correctly yields empty conservative1MiB DP, so cost propagation uses
  the actual handoff raw-byte rule. Tests corrected without altering production
  acceptance. Initial and final760 functional +56 safety pass, no failures/skips.
  Real runner/cache fixtures, temporary files/HTTP response fixtures and next
  stack epoch exercised; not a real remote run, CUDA/model/Full performance or
  representative measurement.
- Four owned D36 scopes checked TasksCurrent0/no pids/high,max,OOM0, then stopped.
  Final GPU memory15MiB each, utilization0/0/1/0%; MemAvailable108926216KiB,
  disk353696034816 B. All147 protected entries and source plan SHA unchanged.
  Baseline repo remains16570c023a439c884624e7a5bdfa0d8577faf7a3, user dirt preserved.
- Next: actual owner source/target footprint/remaining-budget option production,
  common pending movement queue, loss-per-usable-byte replacement, total HOST/
  native/shared/preactivation accounting and Full lifecycle. Representative real
  profile collection and complete qualification still open. D35 startup guard
  retained. Baselines/M1/M2/ablations/sensitivities not started; full goal active.
- Implementation f3039c35184287e584b615659c9dd63d443f65cf pushed to
  faaslora_origin/retry14_continuous_queue_v2; full remote SHA verified. All D36
  owned scopes checked empty and stopped. Only original user dirt remains.
  This backup receipt is documentation-only. Recheck current state and plan
  before continuing actual owner/movement integration; do not repeat this gate.

### D35 class-cost planning entry; Full execution still unqualified

- PreparationClass separates representation/layout/size from request D/T/O.
  Cost updates require source-bound actual loading intervals, reject repeated
  native lease samples and unsupported classes, and do not treat shared-source
  ineligible records as zero. New replicas inherit frozen initial profiles.
  Typed inputs alone do not certify the eventual measured profile provenance.
- Actual ExperimentStack planning entry reads its HotnessTracker once, freezes
  class estimates once, derives F=h*[d_source-d_target]+, then calls the existing
  IEEE handoff/residency selectors. Input-order-stable plan SHA binds all options,
  demand, cost version, received source ID and remaining budgets. It explicitly
  is NOT a physical reservation. Positive-demand missing classes fail, while
  zero-demand options do not invent measurements.
- Actual runner's ieee_confirmed preload path rejects legacy priority/warmup
  BEFORE stack.start and cache reset. No fabricated fallback/legacy Full. This
  guard remains until actual movement-owner/queue integration is complete.
-11 added tests; targeted152, full751 and safety56 pass, no failures/skips.
  Correctness table in P1 D35; no GPU/model run or representative profile. Three
  owned test scopes verified TasksCurrent0, no pids, high/max/OOM events0, stopped.
  Final GPUs15MiB/0%, MemAvailable108992372KiB, disk353713020928 B. Plan SHA and147
  protected entries unchanged. Baseline remains16570c023a439c884624e7a5bdfa0d8577faf7a3,
  its user dirt untouched. No adapter/trace generation or old-result mutation.
- Next unresolved mainline: frozen real preparation class producer/loader and
  binding of actual completed loads; received owner source/footprint/remaining
  budget options; common pending movement queue and loss-per-usable-byte
  replacement; total HOST/native tensor and shared/preactivation pressure;
  complete Full lifecycle/remote/backend qualification. Do not treat the stack
  planning API as automatic control-path execution. M1/M2, formal baselines,
  ablations and sensitivities remain not started. Goal remains active.
- Implementation dba7014570a982cca892ca5937fbe611246cdb4d pushed to
  faaslora_origin/retry14_continuous_queue_v2 and full remote SHA verified.
  All D35 owned scopes stopped; only original user dirt remains. This backup
  receipt is a subsequent documentation-only commit. Resume by rereading plan/
  ledger and current resources; do not repeat the completed planning checks.

### D34 preparation-cost measurement boundaries; no new performance experiment

- Actual native loading records start after source/capacity checks and completion
  at the existing executable-reference fence. Actual HTTP records request-start
  and publication in the inference host clock. The real async/request path carries
  its own transfer evidence, not a nearest-adapter global-log lookup.
- Preparation d excludes waiting before its first loading stage; Remote d retains
  gaps after that start through native activation. D/T/O remains unchanged.
  GPU hits are not invented load samples. Shared-file or changed-native sources
  have d=null and an explicit ineligible reason; bad identity/clock/order fails.
- Six added checks and extended real runner/file/native-cache-fixture cases.
  Initial targeted222 had one test-only NameError (missing clock-helper import),
  corrected without changing acceptance. Final740 functional and56 safety pass,
  no failures/skips. P1 D34 contains correctness table and source rationale.
  Fixtures are not representative model profiles, CUDA/model performance or
  proof that LoRA weights are applied numerically.
- All three owned D34 scopes verified empty, high/max/OOM events0, then stopped.
  Final GPUs15MiB/0%, MemAvailable109106632KiB, disk353723465728 B. Protected147
  entries and source-plan SHA unchanged. Baseline repo remains16570c023a439c884624e7a5bdfa0d8577faf7a3;
  no adapter/trace regeneration, model load or old-result mutation.
- Important mainline finding retained: _preload_full_stack still calls legacy
  mixed priority/warmup, not the tested IEEE mathematical selectors. Actual
  planner/handoff/replacement, measured class/layout/footprint initialization,
  total HOST/native/shared ownership and Full lifecycle remain open. Do not use
  D26 HOST D, D33 activity intervals or these fixtures as preparation profiles.
  Baseline performance, M1/M2, ablations and sensitivities remain not started.
  Goal remains active; resume integration, not an expanded timing-only campaign.
- Implementation6bdff25aadefb0d42086083844b4040c5fde8514 pushed to
  faaslora_origin/retry14_continuous_queue_v2; full remote SHA verified. This
  receipt is documentation-only. All D34 scopes were checked empty and stopped;
  only original user dirt remains. Recheck live resources before the next task.

### D33 implementation evidence; no new model or performance experiment

- Actual file-preparation start/finish now reaches the initialized target replica's
  native scheduling owner. E(t) uses that activity count and an explicit loading
  limit instead of constant zero. Native HOST→GPU remains serialized/fenced.
  Event ownership is conservative, not pure wire time or a planner d profile.
- Repeated events are idempotent; cancelled/lost start replies require a terminal
  tombstone; unfinished/uncertain finish retains pressure. Completed IO and a
  caller cancelled during finish are separate outcomes. Owner mismatch cannot
  clear old uncertainty. Actual remote/local async entries and dedicated TCP
  command routing are exercised. Scope is NOT shared/preactivation/global Full.
- Ten added tests. Targeted112, initial/final full734 and safety56 pass, no skips
  or failures. Actual installedvLLM0.30/torch2.13+cu130 imports and utility signature
  pass, CUDA uninitialized, no model loaded. P1 D33 has the required correctness
  table and primary-source rationale; fixture pressure0.5 is not a model result.
- All five owned D33 test scopes verified at TasksCurrent0, high/max/OOM events0,
  then stopped. Forty-two older listed scopes were also inspected read-only and
  contained no tasks; their active label is not evidence of a live experiment.
- Final GPUs15MiB/0%, MemAvailable109079732KiB, disk353736339456 B. All147
  protected entries and the plan SHA unchanged. No new weights/traces, no old
  results overwritten. Baseline repo unchanged at16570c023a439c884624e7a5bdfa0d8577faf7a3.
- Return to actual planner/handoff/replacement, shared pressure, full HOST/native
  resource ownership and correctly measured initialization profiles. Do not turn
  this narrow journal into a new microtest campaign. M1/M2, baseline performance,
  ablation and sensitivity remain not started; goal active.
- Implementationec1416e5f49a979ed40193852ae69bb23353eaf8 pushed to
  faaslora_origin/retry14_continuous_queue_v2; remote full SHA verified. This
  receipt is documentation-only. Recheck live resources and reread plan/status
  before continuing; no D33 task remains active and no performance slot completed.

### D32 implementation evidence; no new model or performance experiment

- Actual confirmed local HOST/NVMe movement now shares preallocation with HTTP
  archive transfers, payload-only for local copies. Old destination, staging and
  concurrent transfer bytes count in one frozen file budget. No sparse allocation,
  hidden eviction, relaxed budget or copytree fallback qualifies the native path.
- Source lease spans copy/publication/cleanup. Actual async runner uses the same
  join-on-cancel fence as remote transfer. Content identity is retained both ways;
  publication errors/cleanup failures and real residual space are not hidden.
  Planning snapshot is explicitly not a reservation and not total HOST RAM.
- Seven added checks; final724 functional +56 safety pass, no failures/skips.
  The17 confirmed-publication/copy checks also pass with actual tmpfs /dev/shm
  temporary files. These are bounded CPU/filesystem checks, not model performance
  or resident-DRAM qualification. The required correctness table is in P1 D32.
- Source plan SHA and147 protected artifacts verified unchanged. No adapter,
  trace or model regenerated; no GPU test/model loaded. Baseline repo unchanged
  at16570c023a439c884624e7a5bdfa0d8577faf7a3. Do not stage either repo's user dirt.
- Remaining mainline: total HOST/native accounting, global transfer pressure,
  actual proactive planner/handoff, measured class initialization, complete Full
  lifecycle and backend/remote qualifications. M1/M2, formal baseline, ablation
  and sensitivity matrices remain not started. Do not expand local-copy tests.
- Implementation05b1b366fdf407e6be378cf73f868f54d022f9ce pushed and full remote
  SHA verified. All five D32 test scopes checked at TasksCurrent0 and stopped;
  none remain listed. Final GPUs15MiB/0%, MemAvailable109176836KiB,
  disk353765814272 B.147 protected entries/source plan SHA unchanged.
  Only pre-existing user dirt remains. This backup receipt is documentation-only;
  recheck current resources before the next launch. Full goal remains active.

### D31 implementation evidence; no new model or performance experiment

- Complete prediction set now includes explicitly registered controller-pending
  demand and native unfinished requests on the same scheduling owner thread.
  Real native preprocessing supplies token SHA/count, declared limit and adapter;
  native random IDs are preserved. Binding/sending does not drop pending demand;
  actual ADD performs unique handoff. No physical KV allocation is invented.
- Actual runner registration precedes selected-source protection. Engine/proxy/
  dedicated-worker and frontend lifecycle paths are wired; a lost response keeps
  the intent, cancellation needs owner proof, and tombstones reject late revival.
  Bound/native requests still need real retirement/deferred-KV fences. Unknown
  ownership is retained, not turned into empty capacity or a successful request.
- 18 new tests;717 functional and56 safety checks pass, no failures/skips. Actual
  installed0.30 frontend/scheduler imports and native API signatures pass with
  CUDA hidden, no model loaded. Actual loopback TCP tests cover the new commands
  and generation identity; this is not CUDA or complete Full qualification.
- P1 D31 includes the required status table and primary-source rationale. Source
  plan SHA and147 protected artifacts unchanged. No model/LoRA/trace regenerated,
  old results untouched; baseline repo unchanged at16570c023a439c884624e7a5bdfa0d8577faf7a3.
- Return to all-tier transfer/budget, actual planner/handoff and measured profiles.
  Do not start another isolated handoff/pitched-copy/model-prefix microcampaign.
  A meaningful integrated Full qualification is still required before M1/M2,
  baseline performance, ablations or sensitivities. Goal remains active.
- Implementation d6a6a158ab8de288a64456707e88859ab7926a82 pushed and full remote
  SHA verified. All six D31 test scopes checked at TasksCurrent0 and stopped.
  Final plan/protected seal unchanged; GPUs15MiB/0%, MemAvailable109188416KiB,
  disk353778409472 B. Only original user dirt remains. This backup receipt is
  documentation-only; recheck live resources before the next launch.

### D30 completed implementation and real native-setter copy qualification

- D29 workspace finding now has an explicit pitched-copy strategy, using actual
  CUDA row geometry inside unchanged native setters/LRU. No max-rank reduction,
  guessed workspace, global monkeypatch, fallback copy or new adapter artifacts.
  Ordinary demand loading remains distinct; Full/CapacityOnly share this strategy.
- 699 functional and56 safety checks passed, no failures/skips. Plan SHA and147
  protected entries unchanged before execution. Actual CUDA/native-setter test
  completed once: linear and merged/missing-middle full-pool contents exactly match
  native copy; extra tensor peaks0 vs65,536/176,128 B. No backbone/model-prefix rerun.
- Attempt1 was a launcher name/path contract error before service; note retained,
  no invented traceback. Attempt2 normal exit0,12 resource samples, peak729321472 B,
  no high/max/OOM; actual native contexts clear and service scope removed. Auxiliary
  scope18791d4b43c847298cb13ece0ab9fc29 checked empty and stopped; owned tmux closed.
  State table plus curated CSV/JSON delivered, not a latency or system ranking result.
- All8 curated source/evidence SHA entries and4 CSV rows checked against raw output.
  Final plan SHA and147 protected entries unchanged. All five remaining D30 test
  scopes verified at TasksCurrent0 and stopped. GPUs15MiB/0%, MemAvailable109355868KiB,
  disk353817423872 B at final check. Baseline repo unchanged; preserve its user dirt.
  Implementation21e80f9 was pushed and remote SHA verified before the CUDA test;
  this result/receipt is a subsequent documentation/data-only checkpoint.
- Then resume controller-pending/native KV identity handoff, all-tier budget and
  transfer accounting, actual planner/handoff and representative measured profiles.
  M1/M2, formal baselines, ablations/sensitivities remain not started.

### D29 implementation evidence; no new model experiment

- The actual engine/proxy/dedicated/core/worker entry points now execute one
  same-owner KV observation, E(t) decision and fenced native HOST promotion.
  No controller-sampled state is mislabeled atomic. UniProc0.30 only;
  current async scheduler, actual native victim order and demand policy retained.
- Whole LoRA pool counted once. Zero additional GPU storage/workspace is allowed
  only for inspected dense pinned-CPU/same-dtype/contiguous source AND target slices.
  Unknown conversion/staging/setters fail before eviction. Full deferral changes
  no cache; CapacityOnly keeps the same physical protection and LRU.
- Native successful completions feed the exact window; cancelled/error requests
  do not. Profile is explicit and must cover all frozen buckets. No formal profile
  data or new weights/traces created. The new switch does not enable legacy warmup
  as IEEE Full or authorize any formal launch.
- 15 added checks;691 functional +56 safety pass, no failures/skips. Real loopback
  RPC tests include both new preparation and existing observation commands.
  Old D28 inventory reused offline:2 contiguous HOST layouts, rank8 vs GPU max-rank64.
  Actual Torch source and meta tensors show the rank-sliced B destination needs a
  temporary. The zero-workspace path rejects it; this is a concrete open integration
  item, not a qualified mixed-rank preparation. No GPU run was used to discover it.
  This is not new CUDA-copy, all-tier admission or Full performance qualification.
- Plan SHA and147 protected entries unchanged. No model launched this block;
  all GPUs15MiB/0%, MemAvailable109264220KiB, disk353834336256 bytes at check.
  Recheck before launch; baseline repo unchanged. State table delivered in P1 D29.
- Next: actual rank-sliced copy workspace contract, controller/native identity handoff for complete admitted-KV set,
  other-tier transfer/budget integration, actual planner/preparation, representative
  frozen measurements. No new isolated capacity5/source32/lifecycle4 diagnostics.
- Implementation4ca84c3 pushed and remote full SHA verified. All five D29 auxiliary
  scopes checked at TasksCurrent0 and stopped; no model/context left. Final protected
  seal147/147 and plan SHA unchanged; GPUs15MiB/0%, MemAvailable109386324KiB,
  disk353836335104 bytes. Only the original user dirt remains unstaged. This backup
  receipt is a subsequent documentation-only commit; the full goal is not complete.

### D28 completed evidence block; full goal remains active

- Native pin-capacity conflicts now carry exact blocking lease identities.
  The actual runner waits for acknowledged reference releases, then re-observes
  the native epoch. No polling, artificial delay, external unpin or OOM retry.
- 10 additional tests;676 functional +56 safety pass. Source plan and147-entry
  protected seal unchanged. One corrected fixture-only read-only capacity error
  is recorded in P1 D28. No new formal performance result.
- Real controlled7B qualification completed once:5/5 targets,755 tokens, one
  native GPU-capacity wait3764.347ms, all five references returned. Full table
  and CSV/JSON delivered. Attempt1 was a missing launcher NVML SHA, before model
  execution; retained and classified separately. No test seed/config selection.
- Attempt2:64.852 physical GPU-s,7.742s post-token tail, normal worker exit0;
  service peak5699035136 bytes,77 samples,no high/max/OOM. All native contexts
  cleared, service scope removed, both owned empty auxiliary scopes stopped.
  Post-run four GPUs15MiB/0%, MemAvailable109437428KiB. Recheck live resources
  before next launch. Plan SHA and147 protected entries unchanged.
- Source implementation33ef68d remotely verified before model execution.
  All9 curated evidence/executed-source SHAs and all5 CSV rows recomputed
  against preserved raw outputs. D28 after-run report is a subsequent
  documentation/data-only checkpoint, not another model execution.
  Mainline next: Full physical KV/tier admission and representative measured
  profiles; no repeated capacity5/source32/lifecycle4 or zero-weight controls.

### Previous D27 checkpoint (retained context)

- D27 is progress: real7B physical ownership qualification and a causally identified
  RPC-close/exit-order correction; no formal matrix slot completed.
- Final regression:666 functional and56 safety tests passed, no failures/skips.
- Three preserved runs are indexed in `20260926_7b_physical_lifecycle.json`.
  Attempt3 runtime hashes are recorded there; startup-failure ownership without
  native-worker lifetime proof remains censored, not automatically released.
- All three experiment scopes cleaned; their owned empty auxiliary scopes stopped.
  No model job remains live. Recheck resources/processes before resuming.
- D27 final resource check: all four GPUs15MiB/0%, no live service scope,
  MemAvailable108399768KiB; disk353880449024 bytes at the post-run seal check.
  All147 protected entries and all12 raw/receipt/watchdog/journal hashes plus
  five executed-source hashes verified. User dirt remains unstaged; baseline
  repository unchanged. Code backup0e2adf9 remotely verified before this receipt.
- Next work: Full physical tier admission and representative measured profiles,
  with lifecycle owner coverage/aggregation integrated into that deployment.
  Do not repeat source32, lifecycle4 or same-prompt/zero controls.
- M1/M2, A1–A5, S1–S13, common SLO/Resident calibration, new baselines and final
  paper-oriented design/curated figure package remain pending. The full goal is active.
