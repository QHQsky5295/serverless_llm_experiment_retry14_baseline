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
| Resource containment / safety | Native Ray inheritance, external replay and7B dedicated physical owner/exit witness tested; Full deployment aggregation connected with CPU tests | RESOURCE_QUALIFICATION.md + EXTERNAL_REPLAY_QUALIFICATION.md + PHYSICAL_GPU_MEASUREMENT.md D27/D55. Actual multi-activation/other-owner qualification and A4 remain open. CPU proof is affinity, not delegated cpuset |
| Protected historical artifacts | Sealed and verified | `paper_results/ieee_tc/safety/20260925_execution_start_protected.json`; old results and selected user modifications unchanged |
| Remote authentication / management | Key login verified; service qualification pending | Strict host checking, dedicated restricted key; remote disk 138.9 GiB below 150 GiB floor, user decision pending. See REMOTE_ACCESS.md |
| P0 main-table / Full provenance | 7B source-pair audit complete | Same trace/subset SHA, different execution; 202 scalar fields preserved. P0_FULL_PROVENANCE.md; no performance rerun needed for this finding |
| Serverless wait audit | Historical audit + no-GPU control-path tests complete | Clean 7B/3B logs reused; six real-method AST tests pass; incremental ready-before-wait patch preserved. Model pair pending |
| P1 IEEE semantic alignment | Automatic GPU/final-file replacement, budgeted CPU-staging joint commit and proactive d feedback connected; Full remains open | P1_FORMULA_IMPLEMENTATION.md through D54. Same frozen h/d and actual completion samples; physical-byte-full HOST/staging capacity, profiles, allocator qualification and Full lifecycle remain open. No GPU/Full qualification; startup still rejects legacy warmup |
| P2 backend qualification | Mechanical sequential/batch/cancel evidence retained; full-pool content scan complete; semantic qualification OPEN | ARTIFACT_CONTENT_AUDIT.md: current 3B 500/500 all-zero, 7B 498/500 all-zero, only 2/4 weight SHAs respectively. 3B native same-prompt/different-SHA outputs identical; zero controls cannot distinguish adapter application. No formal/remote performance qualification |
| Baseline qualification | Pending | Serverless, vLLM, S-LoRA, dLoRA 3B, Loquetier, HydraServe |
| M1 / M2 | Not started | No new formal performance claims |
| A1–A5, S1–S3 | Not started | Shared frozen policies required |
| S4–S13 + offline supplements | Not started | Follow core experiments |
| Documents / final curated figures | Design/qualification documents and diagnostic tables in progress; formal performance figures pending | Current measurement table: PHYSICAL_GPU_MEASUREMENT.md D55; no formal performance figure |

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

D57 connects an explicit pre-import native allocator candidate to the existing
dedicated worker, checks actual native readback, and separates resident R from
temporary W in the existing loading contract. No production model config changed.
Offline six-row workspace table reuses D56; no third allocator microtest or old
request-prefix repetition. See P2_BACKEND_QUALIFICATION D57. Next choose/freeze
the resident/staging partition INSIDE the existing HOST allowance, including
actual non-LoRA occupancy and concurrent staged lifetimes, then qualify actual
Full/profile/lifecycle. An allowance that cannot fit the working set plus loading
workspace is infeasible, not a reason to increase it silently or wait forever.
The opt-in candidate is not selected as a performance winner; complete model
validation and matching vLLM opportunity remain required. Full guard remains.
Current tests:928 functional,58 system-Python safety,26 installed-native CPU
checks pass; zero CUDA initialization. The earlier combined698-test attempt
retains2failures/2errors from using the wrong interpreter for pidfd safety tests,
not a system-result failure. No guard or test was weakened. All5 owned scopes
are empty with memory high/max/OOM0; cleanup/backup receipt below follows.
Tested code/data checkpoint81fc5c47410b43f93142fefee1577a9cd395d7d5 is pushed;
fresh full remote SHA matches. All5 owned scopes stopped. No experiment remains
running. This subsequent receipt is documentation only; baseline16570c unchanged
with fresh origin/main verification. Do not claim complete Full qualification.

D56 measured the actual native CPU checkpoint allocator for all six existing
content/rank/module classes, default vs official no-caching configuration, in
two separately guarded processes. Default retains106168320B after all objects
are removed; same-shape reload uses cached blocks, but cached total does not
guarantee a different shape fits. Uncached returns blocks but loses allocation
reuse (and changes size rounding). Results/table: P2_BACKEND_QUALIFICATION D56
and paper_results/ieee_tc/p2_backend/20260927_native_host_allocator.{json,csv}.
Do not repeat these two microchecks or turn them into an SLO/profile claim.
Next: budgeted workspace and physical-byte-full replacement in actual Full,
using observed allocator semantics, then representative profile/Full lifecycle.
Production allocator settings/equations/budgets unchanged; Full remains guarded.
D56 code/data checkpointcbfbe188f719ca16457133c90c75ed5ab973da33 is pushed and
its full remote SHA verified. Both native observations and all CPU tests ended;
their owned service/aux/test groups are released or empty and stopped. This
subsequent receipt is documentation only. Baseline remains16570c, unchanged.

D55 connects all dedicated owner journals and actual request terminals to one
external-replay deployment reducer, including retired/failed-start runtimes.
It does not qualify Full GPU execution or infer correct LoRA application from
token counts. Final-source937 functional,70 installed-native and56 safety checks
pass; all nine owned scopes are empty and stopped. Implementation7f27bd3fed7d842dac68d5e032b082ea558976b6
is pushed and its full remote SHA verified. This receipt is documentation only; see
PHYSICAL_GPU_MEASUREMENT D55 for the correctness table and scope. Native HOST physical-byte-full capacity,
measured profiles/allocator, actual Full multi-activation/A4 and the two external
authority choices still remain. Baseline/performance stages have not started.

D54 implementation checks complete: budgeted unregistered CPU staging, joint
CPU/GPU loss/admission/commit and demand takeover are connected to the actual
runner/core/worker path. Original three-entry/two-GPU-target fixture now finishes;
deferral leaves old residents unchanged. Physical-byte-full/multi-victim staging,
measured profiles/allocator and Full lifecycle remain OPEN. Do not claim Full
qualification or repeat old model prefixes. User decisions remain pending.
Final-source931 functional,53 installed-native and56 safety checks pass. All
nine owned test scopes are empty and stopped. No experiment runs in background.
D54 tested implementation774bd7f5d0dbdd415ad3ded148152cefd092b5fd is pushed;
fresh full remote SHA verified. This subsequent documentation-only receipt does
not change runtime/test sources. Previous D53:327fa3ba3a145ca5bbec02f43f1e152bcefba4a8
preserves history/status; D52 runtime checkpoint remains7c4337d59fd687ebabf7acfed6f30dbfd75b2a2f.
Main branch retry14_continuous_queue_v2. Baseline remains
16570c023a439c884624e7a5bdfa0d8577faf7a3, origin/main.

Continue the actual Full physical-byte-full native-HOST/staging path. D54 fixes
D52's three-entry/fourth-materialization stall when conservative staging peak
fits the tensor allowance, keeping the original two GPU targets. It does not
solve exhausted physical bytes or prove reusable allocator capacity. Do not hide
those remaining gaps by enlarging the budget or assuming tensor deletion frees
pinned allocator bytes. Joint claim/physical admission must precede reclamation;
E(t) deferral must leave victims resident.

Already connected: automatic frozen h/d GPU and final-file selection/replacement,
native/file ownership, initial/natural activation and IEEE control, actual
proactive completion feedback into the next cost epoch. Full still rejects
legacy warmup. Still open: physical-byte-full native-HOST/staging replacement,
representative profiles/allocator qualification, Full physical lifecycle/A4,
backend numerical/500-pool and real-remote qualifications.

Do not repeat completed source32/capacity5/lifecycle4, zero controls, D52 feedback
or isolated prior prefix tests. No formal baseline/M1/M2/ablation/sensitivity run
has started. Correctness checks are not performance experiments. User has not
yet answered requests about nonzero3B correctness fixtures / remote disk gate.

## Evidence archives and recovery

- [EXECUTION_HISTORY_THROUGH_D26.md](EXECUTION_HISTORY_THROUGH_D26.md) preserves
  earlier full ledger; SHA256 4e91643d959967f1f0dc8642fe969d35380f2156cf8c820964019539d3b415cb.
- [EXECUTION_HISTORY_D27_D52.md](EXECUTION_HISTORY_D27_D52.md) preserves the complete
  pre-D53 ledger verbatim, including all failures, test counts and backup receipts.
  SHA256: bd56dec4a43d7e03f35e8a9b1b26ac598378df4ca75429e0d1344d237e0d1dfe.
  This organization does not change experiment status or the authoritative plan.
- Read relevant original experiment logs/history before each optimization, not
  every obsolete chronological next-action paragraph. Current next actions above
  supersede those historical instructions.
- P1_FORMULA_IMPLEMENTATION.md through D54: equations, implementation and state tables.
- P2_BACKEND_QUALIFICATION.md, ARTIFACT_CONTENT_AUDIT.md,
  NATIVE_ADAPTER_NUMERIC_CONTROL.md: model and artifact qualification limits.
- RESOURCE_QUALIFICATION.md, EXTERNAL_REPLAY_QUALIFICATION.md,
  PHYSICAL_GPU_MEASUREMENT.md: process/resource ownership.
- P0_FULL_PROVENANCE.md and Serverless audit docs: historical evidence.
- REMOTE_ACCESS.md: strict key access and service/disk qualification.

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
   content-bound file paths; D42 combines shared file capacity and per-native-owner
   reserved tensor allowances. Actual allocator configuration/total-service
   observations, proactive planning/handoff and complete deployment physical GPU
   lifecycle still require qualification. Do not silently call the legacy planner
   or the explicit managed-memory envelope qualified IEEE Full.
5. Actual profile measurements must cover their frozen class/configuration and
   environment; neither test constants nor neighboring-class latency can fill gaps.
   Replica inheritance is from frozen measurements, never a previous test block.
6. D27 adds opt-in dedicated-runtime physical allocation/return journals and
   actual7B qualification. D55 connects launch-wide Full journal/terminal
   reduction; actual Full/all-baseline owner coverage is NOT qualified.
   Physical GPU possession is not GPU utilization,
   instance “ready”, request completion, empty_cache or shutdown return.
7. Serverless official one-second ready-path polling is supported by old raw
   logs and6 method tests. Original/repaired1000-request model pairs remain pending.
   Native checkpoint loader path and contained launch must qualify first.
   Existing broad Ray-stop/pkill/global-tmux launcher is not safe for TC.
8. Remote key-only login works, D53 free disk138.52GiB below150GiB floor;
   services not qualified/started. Do not delete unique/unrelated data or silently
   relax the floor. Additional nonzero3B correctness artifacts need user authority.
9. Physical CPU isolation proof is actual affinity, not unavailable delegated
   cpuset. Services72/80GiB high/max, swap2GiB; auxiliary4GiB; one heavy at a time.
   Read source plan for exact admission, disk, failure/timeout and statistical rules.
10. All147 protected entries and source plan SHA verified unchanged after D55.
    D55 functional937, installed-native70 and safety/census/replay56 pass, no
    skips/failures/errors; these overlapping checks are not independent repeats.
    Final D55 GPUs15MiB/0%, MemAvailable108544020KiB, disk353458348032 bytes.
    All nine D55 owned scopes are empty with high/max/oom/oom_kill0, then stopped.
    Recheck live handles/resources; a stale record is not a live run.

## This continuation

### D57 explicit allocator candidate and workspace accounting

- Startdec4136b637ff9142d2cb7de6d672cf1506366c5, baseline16570c unchanged.
  Full plan/status/AGENTS and run-experiment/github-sync read after recovery.
  Protected147 and plan SHA unchanged; original user dirt preserved. No agent,
  model/GPU, remote174 or formal performance run launched this continuation.
- Examined IEEE staging/budget/replacement text, D54/D56 history, actual multi-
  target staging execution, officialvLLM0.30 load-before-evict and dense packing,
  and PyTorch2.13 allocator. Rejected early eviction, aggregate cached-byte credit
  and arbitrarily enlarged budgets. The actual native policy candidate is now
  explicit in model config, applied before subprocess imports, with legacy alias
  conflicts rejected and one actual worker readback. Default config unchanged;
  profile identity rejects reuse across allocator-policy changes.
- Existing header-based loading contract separates resident pinned R and
  temporaryW=source+converted, without dropping either from peak admission.
  Verified uncached policy uses native exact-size allocation; unconfigured
  policy retains original rounded conservative bound. All actual retained and
  staged bytes continue to count. No immediate release/complete RAM guarantee.
- Reused D56 rawSHA08f7083ec24bfc2d6605187da7eabc8b5558e1dcb1f7ce1fe99809ae0a14e698
  to produce six-row CSV/JSON and immediate P2 table. 7B rank16 final32MiB vs
  conservative extra load peak100697616B;3B rank16 final17.5MiB vs55080136B.
  Peak is derived, NOT measured; no SLO/profile/performance inference.
- Initial492 targeted checks pass. Combined698 discovery uses the old model
  Python for safety tests and returns2failures/2errors because pidfd_send_signal
  is absent. Preserved cpu_regression.log. Correct separation gives928 functional
  and58 qualified system-Python safety checks,0failure/error/skip. Installed
  vLLM0.30/torch2.13 environment26 focused CPU checks pass; CUDA uninitialized.
  Logs under results/ieee_tc/p2_backend_qualification/d57_20260927. Counts overlap.
- Full cache/workspace partition and multi-plan liveness, measured profiles,
  actual Full/A4, numerical500-pool and real-remote remain open. No native budget,
  formal model config, equations, old results, trace or artifact was changed.
  Baseline repair pairs/M1/M2/ablations/sensitivities have not started. Pending
  user choices on nonzero3B fixtures and remote disk remain unchanged.
- Final5 owned scopes actual cgroup.procs empty; high/max/oom/oom_kill0.
  GPUs15MiB/0%, MemAvailable108633516KiB, disk353415098368B. Protected147 and plan
  SHA unchanged; offline six rows independently reconciled to raw SHA/formula.
  Tested milestone backup follows; goal remains active, not complete.
- Tested code/data81fc5c47410b43f93142fefee1577a9cd395d7d5 pushed to
  faaslora_origin/retry14_continuous_queue_v2; fresh full remote SHA verified.
  Eleven named task files only, diff/protected-name/added-text secret checks pass.
  All5 test scopes empty and stopped; no live model/test. Baseline no task edits,
  HEAD16570c023a439c884624e7a5bdfa0d8577faf7a3 matches fresh origin/main.
  This documentation-only receipt changes no measurement or runtime source.

### D56 actual native HOST allocator evidence

- Previous goal turn is progress: D55 connected/tested/backed up complete
  deployment physical measurement. Startd0f07d0fc522e9f8e4d3c75dce868261a3ec1c08;
  full plan/status/AGENTS and run-experiment/github-sync reread. Protected147,
  plan SHA and original user dirt unchanged.42 historical listed scopes were
  checked read-only: actual cgroup.procs all empty; no running model to restart.
- Inspected D41/D42/D54 histories, actual native loader/budget/allocator code,
  officialvLLM0.30 LoRAModel/worker, PyTorch2.13 allocator/config/statistics and
  ELORA paper. Default stats cannot establish shape-compatible reusable bytes;
  rejected both early victim credit and subtracting aggregate cached bytes.
- Extended existing preflight with backend-host-check, no new framework or
  pool/trace. Before measurement select first existing content/rank/modules
  representative in weight-byte/ID order:3B2 and7B4. Exact weights/config SHA
  checked before/after. Actual native CPU checkpoint materialization; no backbone,
  native registry/packing, H2D, generation or inferred numerical correctness.
- Two fresh guarded services (72/80GiB, swap2GiB, common CPU/aux/watchdog) run
  default and official pinned_max_cached_size_mb:0, same six controls and six
  steps each. Both complete, service/watchdog exit0, actual context release and
  service removal confirmed. Each17 resource samples; peaks782364672/707137536B,
  high/max/oom/oom_kill0. Auxiliary groups verified empty and stopped.
- Default finalactive0 butallocated106168320B; same-shape reload adds0 native
  allocations;3B rank transition still grows the pool despite cached bytes.
  Uncached returns all six objects' blocks, finalallocated0; reload adds224
  allocations for3B and256 for7B. Its size-rounding also differs. These are
  mechanism observations, not performance repetitions or optimality evidence.
- Exact72-row CSV + curated JSON record raw/launch/watchdog/source SHA, actual
  config readback, resource state and limitations. P2 document supplies the
  immediate result table before advancing. First arm cleaned/table written
  before second. No production allocator/budget/policy change, no Full guard
  removal, no model prefix repetition or formal baseline/main/ablation result.
- Two added selector/guard checks;58 safety/census/replay and288 existing basic
  smoke checks pass0fail/error/skip. Final cleanup/protection/backup follows.
  Remote disk/nonzero3B authority choices remain pending; main goal stays active.
- Final two auxiliary and two test scopes checked TasksCurrent0/actual empty
  cgroup.procs, high/max/oom/oom_kill0, then stopped; both service groups already
  gone under the existing watchdog. GPUs15MiB/0%, MemAvailable108489688KiB,
  disk353422700544B. Protected147 and authoritative plan unchanged. All72 curated
  rows independently reconciled with raw counters; raw/launch/watchdog/source
  hashes match. No model/test remains running. Only original user dirt remains.
- Code/data checkpointcbfbe188f719ca16457133c90c75ed5ab973da33 pushed; fresh
  full faaslora_origin/retry14_continuous_queue_v2 SHA matches. Six named task
  files only, whitespace/secrets checks pass. Baseline no edits at16570c with
  fresh origin/main full SHA checked. This documentation-only receipt adds no
  measurement and does not certify Full or formal comparison eligibility.

### D55 whole-deployment physical GPU measurement

- Previous goal turn is progress: D54 implemented/tested/backed up budgeted
  CPU staging and joint native HOST/GPU commit. Current start748f78ec7f4e313dbd6e8fbced7e83e643c3e0cc,
  baseline16570c unchanged. Full authoritative plan, status and skills reread;
  protected147 and source-plan SHA unchanged, original user dirt preserved.
- Examined actual native-HOST budget/loader and IEEE replacement text. Physical
  byte exhaustion cannot be fixed by pretending deleted pinned tensors are
  reusable. No budget increase, early eviction or allocator-flush workaround.
  Advanced the independent Full physical accounting requirement, based on D27
  history and current official NVML/vLLM sources, rather than repeat prefixes.
- Actual external-replay entry now owns a fresh measurement bound to frozen
  input/clock/notice; mandatory dedicated allocator configuration propagates
  through initial/scale-out/reinit. All allocations use the existing launch-wide
  locks. Final sidecar reduces every owner journal after cleanup, independent
  of the surviving pool; incomplete/malformed evidence is not a complete score.
  Scenario metadata links physical evidence separately from old billing.
- Actual request terminal path records success/failure; global cancellation is
  interruption, not a terminal/timeout. Native token-contract matches are a
  separate count, never inferred numerical LoRA correctness. n_correct and
  per-correct resource remain null, comparison eligibility false pending proof.
  One runtime return failure no longer skips all remaining runtime cleanups.
-9 added tests. First70 had1 fixture tuple/list JSON comparison failure, fixed
  by canonical serialization in the assertion. Next70, initial919 across19
  modules and expanded937 across20 modules pass. Final frozen-source/native/
  safety results follow. Counts overlap, not independent repetitions.
- No model/GPU/performance/remote174 run, new artifacts/traces, manuscript or
  old-result edits. Full startup guard stays. Physical-byte-full HOST/staging,
  measured profiles/allocator, actual Full multi-activation/A4, numerical and
  remote gates remain. P1/P2 has progressed; formal baseline/M1/M2/ablations/
  sensitivities have NOT started. No full-completion claim.
- Final frozen-source20-module937 functional, installedvLLM0.30.0/torch2.13.0+
  cu13070 and safety/census/replay56 checks pass with no failures/errors/skips.
  CUDA remains uninitialized. Test selections/counts overlap; no performance or
  independent-repeat claim. Final runtime SHA256: metrics741d0781bb9ccb53114a6267e4c90ac0c806320f382f4e87bb122e29e2b2e8d2;
  runnerb488f2fa2ea9e63b10b3e364ba4d0ba285204b5fa9be8d7a82be50081afa4f45.
- All nine D55 scopes have TasksCurrent0 and actual cgroup.procs empty,
  high/max/oom/oom_kill0, then stopped. No test/model remains running. Final
  GPUs15MiB/0%, MemAvailable108544020KiB, disk353458348032 bytes. All147
  protected entries and plan SHA unchanged. Baseline unchanged at16570c with
  fresh origin/main full-SHA verification; no unrelated changes staged. Named
  file diff/secrets checks and tested implementation milestone backup follow.
- Tested implementation7f27bd3fed7d842dac68d5e032b082ea558976b6 pushed to
  faaslora_origin/retry14_continuous_queue_v2; fresh full remote SHA matches.
  Exactly six task files committed, generated manifest/unrelated dirt excluded.
  Initial added-text secret scan falsely matched the hexadecimal validation
  alphabet; inspection identified only that hit, boundary-aware scan passed
  before push. No credential was introduced. Baseline remains16570c unchanged
  with fresh remote verification. This subsequent receipt changes documentation
  only. Mainline remains representative measured HOST/allocator capacity and
  actual Full multi-activation/A4 qualification; no new performance claim.

### D54 budgeted staging and joint native HOST/GPU commit

- D53 archived authoritative history and refreshed external-gate evidence; it
  did not implement replacement. D54 reread full plan/status/skills, IEEE history,
  actual installedvLLM0.30 worker/model code and primary onlinevLLM/ELORA sources.
  Start main327fa3ba3a145ca5bbec02f43f1e152bcefba4a8, baseline16570c unchanged.
  Protected147 and source-plan SHA unchanged; original user dirt preserved.
- Actual full-cache path now stages a budgeted CPU object without registration,
  holds fastest real file fallbacks, evaluates joint CPU/GPU loss and E(t), then
  commits only after acceptance. CPU removal's GPU invalidation is counted once.
  Pins/joint targets protected, loss of replies retains references. Native demand
  can consume staging under its original LRU policy and invalidates the old
  proactive observation; no duplicate file load or complete-load cost sample.
  No hidden LRU in proactive path, cache enlargement or future eviction credit.
- Scope is entry-full with sufficient actual peak allowance, not physical-byte-
  full multi-victim capacity/reuse proof. Full guard stays. P1 D54 gives the
  required correctness state table and exact limits. No GPU/model/174/profile/
  performance campaign, new weights/traces or manuscript/old-result changes.
- Initial16 checks had2 errors from missing new native CPU cost classes in the
  mixed fixture, corrected explicitly; next16 pass. Expanded144 had2 old mock
  call-argument assertions and7 errors from requiring file owner on native-only
  paths; move that dependency to staged operations, no fallback/relaxed guard.
  Next144 and initial930 full regression pass with no failure/error/skip.
  Four added integrated checks cover original a/d targets at CPU capacity3,
  E(t) deferral/cancel, demand takeover, and lost commit reply. Final frozen-source
  native/regression/safety, cleanup and backup receipts follow.
- Final review distinguishes registered and staging-only tensor bytes without
  double-charging shared storage; all process-retained pinned bytes still count.
  Adds one accounting check. Frozen-source931 functional checks,53 checks under
  installedvLLM0.30.0/torch2.13.0+cu130, and56 safety/census/replay checks pass
  with no failures/errors/skips. CUDA stays uninitialized. These are overlapping
  correctness checks, not performance observations or independent repeats.
- All nine D54 scopes have TasksCurrent0 and actual cgroup.procs empty, with
  high/max/oom/oom_kill0, then stopped. Final GPUs15MiB/0%, available memory
  108692732KiB, available disk353486036992 bytes. Protected147 and plan SHA
  unchanged. No model/test remains running. Named-file Git checks and milestone
  backup follow; baseline unchanged and no formal performance claim is made.
- Tested implementation774bd7f5d0dbdd415ad3ded148152cefd092b5fd pushed to
  faaslora_origin/retry14_continuous_queue_v2; fresh remote SHA matches exactly.
  Only10 named task files staged, diff/secret-pattern checks pass, generated
  manifest and unrelated dirt excluded. Baseline has no task changes. This
  receipt is documentation only, not another experiment. Next mainline remains
  actual physical HOST/staging capacity, representative profiles/allocator and
  Full lifecycle qualification; do not rerun completed prefixes or treat entry-
  capacity success as physical-byte-full success. Goal remains active.

### D53 mainline review and explicit external decisions

- Read the full authoritative plan/status, AGENTS and run-experiment/github-sync
  instructions after recovery. Main a8e3926, baseline16570c; original user dirt
  intact. No model, GPU performance, remote-service start or runtime code change.
  The previous ledger is archived byte-for-byte (cmp against HEAD passed), not
  deleted or summarized in place without its original evidence.
- Inspected the actual worker/core/native-owner/runner full-cache path. A staged
  unregistered CPU object could permit observing the incoming copy geometry
  before GPU admission, but it must remain budgeted and owned across cancellation
  and shared consumers. Registration/eviction then needs a joint HOST/GPU loss
  calculation with protected actual fallback copies. This design is NOT yet
  implemented or qualified. Increasing CPU entry count, replacing by native LRU,
  or evicting before E(t) are not accepted as a fix.
- Official PyTorch host allocator documentation and installed torch2.13 source
  confirm that allocated bytes include retained pinned blocks; the public API is
  torch.accelerator.empty_host_cache(), not torch.cuda.empty_host_cache(). Its
  existence does not prove that a future victim yields reusable capacity. No
  flush, allocator configuration change or fake eviction credit was introduced.
  Sources: https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.host_memory_stats.html
  and https://docs.pytorch.org/devlogs/eager/2026-08-09-pinned-memory-allocator/ .
- Fresh strict-key read-only remote174 check:148735832064 bytes available
  (138.52GiB); no listener on18080/18081. Existing7B/3B upload archives total
  1967102690 bytes (1.83GiB). Deleting them cannot satisfy the150GiB floor.
  Nothing deleted and no service started. Asked user to keep the floor and free
  space, or authorize an independently measured artifact-node peak-space rule.
  This supersedes the insufficient suggestion that archive cleanup alone could
  unblock the node. Inference-machine limits remain unchanged.
- Re-asked the outstanding authority question: a few trained nonzero3B
  correctness fixtures, without modifying the original500 pool or trace, versus
  explicitly limiting the claim to the zero-weight pool. No new weight was
  acquired/generated and no current correctness gate was waived.
- Read-only current scopes showed no processes; all147 protected artifacts and
  authoritative plan SHA unchanged. GPUs15MiB/0%, MemAvailable108664392KiB,
  local available353393827840B at check. Mainline report explicitly distinguishes
  implementation checks from formal performance: baseline/M1/M2/ablations/
  sensitivities have not started. D53 has no new performance claim. Await the
  two user choices before any action changing the approved artifact/disk rules;
  full goal remains incomplete. Final documentation validation/backup follows.
- Documentation-only checkpoint validation:288 existing basic smoke checks pass
  with no failures/errors/skips. The owned CPU-only scope has TasksCurrent0,
  actual cgroup.procs empty and high/max/oom/oom_kill0, then was stopped. Protected
  seal147/147 and plan SHA reverified unchanged after the checks. No experiment
  runs in the background. Only these three documentation files are being backed
  up; baseline/runtime code and original user changes remain untouched.

### D52 actual proactive preparation-cost feedback

- Reread full authoritative plan/status and AGENTS/skills; inspected IEEE source,
  D34/D36/D41/D46/D51 history and officialvLLM0.30 worker/model source plus dLoRA.
  Start main328492967c966ab48ac70d8a5aee815931c31521, baseline16570c unchanged.
  Protected147 and plan SHA unchanged. GPUs15MiB/0%, MemAvailable108733024KiB,
  disk353419440128B at live recheck; original dirt preserved. No agents/model/GPU/
  remote174/performance campaign. All42 older listed scopes report TasksCurrent0.
- Native HOST replacement inspection confirmed CPU removal can deactivate GPU
  and pinned allocator retention is not freed usable bytes. Did not add eviction
  or relax budget. Advanced another required Full gate: actual mixed preparation
  now feeds complete own-operation loading intervals to source-class EWMA.
  First-stage prewait excluded; interstage wait retained. CPU/Remote sharing and
  GPU reuse do not create zero or duplicate samples. Frozen epoch unchanged;
  new replicas retain frozen initialization. P1 D52 gives correctness table.
-7 new checks plus actual GPU-reuse identity receipt. Initial45 had1 error due
  that missing source identity, corrected. A larger check timed out55s in new
  multi-target NVMe fixture; bounded3s diagnosis confirmed existing native HOST
  full-cache deferral for the fourth CPU object. No capacity/selector change:
  feedback fixture uses one source target; replacement remains explicitly open.
 131 focused and initial925 full checks pass0fail/error/skip. Closure review adds
  native CPU object incarnation across loading stages: same ID/path reloaded is
  not the original CPU stage. Reuse preserves identity; reload changes it and
  disqualifies a composed sample. Final-source full/native/safety receipts follow.
- Formal baselines/M1/M2/ablation/sensitivity remain not started. Native HOST and
  intermediate staging replacement, representative profiles/allocator qualification,
  Full physical lifecycle/A4 and backend/remote qualification remain. Full guard
  retained. No weights/traces/new profile/manuscript/old-result changes. Baseline
  untouched; remote disk/nonzero3B authority gates unchanged. Goal active.
- Final frozen-source926 functional checks,42 installedvLLM0.30.0/torch2.13.0+
  cu130 checks and56 safety/census/replay checks pass0fail/error/skip. CUDA remains
  uninitialized. All147 protected entries and source plan SHA reverified unchanged.
  Prior main3284929 and baseline16570c fresh full remote SHAs verified. Final
  owned-scope cleanup/resource and implementation backup receipts follow.
- All eight D52 scopes verified TasksCurrent0, actual cgroup.procs empty and
  high/max/oom/oom_kill0, then stopped. Forty-two older listed scopes were also
  read-only inspected: all actual cgroup.procs empty. No model/test remains.
  Final GPUs15MiB/0%, MemAvailable108494868KiB, disk353405702144B. No capacity
  relaxation, no artifacts regenerated. Tested implementation backup follows.
- Implementation7c4337d59fd687ebabf7acfed6f30dbfd75b2a2f pushed and fresh full
  remote SHA verified. All eight D52 scopes are stopped; only original user dirt
  remains. Plan and protected147 reverified unchanged after backup. This subsequent
  documentation-only receipt is not another experiment. Resume integrated native
  HOST/staging capacity/admission, representative profiles/allocator qualification
  and Full physical lifecycle/A4; do not repeat feedback or old model prefixes.
  Full goal remains active, no complete/performance qualification claim.
