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
| P1 IEEE semantic alignment | Native E(t)/HOST preparation transaction connected and CPU-tested; Full integration open | P1_FORMULA_IMPLEMENTATION.md D1–D29. Source32 and capacity5 measured earlier. D29 adds same-owner KV/slot decision/commit, not a new model or Full performance result. Controller-pending KV, all-tier transfer/budget and proactive planner integration remain open |
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

LATEST CODE CHECKPOINT: D29 adds opt-in native HOST→preallocated-GPU preparation
with core-owned E(t) evaluation and completion-length window.691 functional and
56 safety tests pass. No new model execution or profile numbers. Existing model
evidence remains D28 below. Do not repeat capacity5/source32/lifecycle4.

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
1. D29 now connects E(t) to the native core/worker HOST→GPU operation, with exact native LRU victim and real pool/headroom reads. NEW concrete gap: D28 native rank8 CPU adapters copy into max-rank64 GPU B slices; PyTorch allocates a temporary for this noncontiguous target. The zero-workspace branch correctly REJECTS this path. First close its actual workspace or equivalent no-hidden-allocation copy contract; do not lower rank config or guess MB. Then bridge controller-admitted/not-yet-ADD requests into the KV set with unique identity handoff; include all-tier transfer events/budgets, attach actual planner/handoff and measured class initialization. Current receipts cover native unfinished requests and serialized HOST→GPU transfers only. Content-bound HOST/NVMe migration remains open. No repeated source32/capacity5/lifecycle4.
2. D27 now qualifies the7B dedicated runtime's physical allocation and normal exit. Do not re-run its four-request prefix. Connect owner coverage and aggregation for the actual Full deployment (shared/direct/multi-runtime paths are not qualified here), while completing physical tier admission and representative measured profiles. Qualify Serverless native checkpoint path before its original/repaired model pair.
3. Continue remote setup after its disk gate; no heavy GPU run
   until actual process containment and watchdog gates are satisfied.

## Latest verified backups and evidence index

- Main tested implementation: `4ca84c375d1b17029cfb8807650bfe86a631ede6`, pushed to
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
- P1 formulas / native sources / ownership: `P1_FORMULA_IMPLEMENTATION.md` D1–D29.
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
4. Full received-view routing and actual D25 source protection exist, but
   all-tier physical admission/E(t), content-bound
   multiple file paths, proactive planning/handoff connection and physical GPU
   lifecycle still require real integration. Do not silently use legacy paths.
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

### D30 implementation, real copy qualification next

- D29 workspace finding now has an explicit pitched-copy strategy, using actual
  CUDA row geometry inside unchanged native setters/LRU. No max-rank reduction,
  guessed workspace, global monkeypatch, fallback copy or new adapter artifacts.
  Ordinary demand loading remains distinct; Full/CapacityOnly share this strategy.
- 699 functional and56 safety checks passed, no failures/skips. Plan SHA and147
  protected entries unchanged. Current CUDA qualification still pending: next is
  one guarded native-setter/copy microcheck, no backbone/model-prefix rerun.
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
