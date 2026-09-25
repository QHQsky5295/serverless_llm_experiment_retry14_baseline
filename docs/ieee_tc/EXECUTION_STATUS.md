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
| Resource containment / safety | Native Ray inheritance, guarded external replay/startup ingress and idle native GPU census witnesses tested; model gates pending | RESOURCE_QUALIFICATION.md + EXTERNAL_REPLAY_QUALIFICATION.md + PHYSICAL_GPU_MEASUREMENT.md. Physical owner-event/native model lifetime integration remains open. CPU proof is affinity, not delegated cpuset |
| Protected historical artifacts | Sealed and verified | `paper_results/ieee_tc/safety/20260925_execution_start_protected.json`; old results and selected user modifications unchanged |
| Remote authentication / management | Key login verified; service qualification pending | Strict host checking, dedicated restricted key; remote disk 138.9 GiB below 150 GiB floor, user decision pending. See REMOTE_ACCESS.md |
| P0 main-table / Full provenance | 7B source-pair audit complete | Same trace/subset SHA, different execution; 202 scalar fields preserved. P0_FULL_PROVENANCE.md; no performance rerun needed for this finding |
| Serverless wait audit | Historical audit + no-GPU control-path tests complete | Clean 7B/3B logs reused; six real-method AST tests pass; incremental ready-before-wait patch preserved. Model pair pending |
| P1 IEEE semantic alignment | Mathematical contracts, native observations/references, cooperative file ownership and staged HTTP/local publication tested; physical integration open | P1_FORMULA_IMPLEMENTATION.md D1–D16. Selected native requests protect file reads; true-remote writes publish through the same owner, cancel joins actual writer. Content/physical budgets/registry epoch, pre-decision source/cost composition, CUDA/clock/stream qualification, abort reconciliation and proactive atomic admission remain open; no Full performance qualification |
| P2 backend qualification | Dependency dry-run passed; isolated installation running, no model run yet | P2_BACKEND_QUALIFICATION.md. 198 hash-locked binary packages; private tmux `tc-p2-0925-01:install`, 3/4 GiB build scope. Poll before any new heavy work |
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

0. Check the existing P2 install before starting anything heavy: `tmux -L tc-p2-0925-01 list-sessions`; log `paper_results/ieee_tc/p2_backend/vllm0300_install_attempt1.install.log`; final receipt same prefix `.json`. Do not recreate the venv, repeat downloads or run another model concurrently.
1. Continue native owner integration: measured source-class costs, resource-owner remaining budgets, atomic routing/admission. Native token/reference interfaces and startup-parallel external arrival/submission are opt-in and unit/witness tested, but actual engine clocks/streams/controller owners remain open. Do not declare the historical scorer IEEE-aligned.
2. Qualify actual model/GPU workers and lifecycle under the existing guarded launcher; native single-raylet and tiny replay witnesses are not the full deployment gate. Qualify Serverless native checkpoint path before its original/repaired model pair.
3. Continue remote setup after its disk gate; no heavy GPU run
   until actual process containment and watchdog gates are satisfied.

## Verified backups

- Main `a022340f014ab2a115bbab79841638ec6c15d1ae` pushed to
  `faaslora_origin/retry14_continuous_queue_v2`; remote SHA checked.
- Baseline `2353e7e0c7af7cfd0bfb43a4913c9178da391e34` pushed to `origin/main`;
  remote SHA checked. No pre-existing user changes staged.
- Baseline audit/repair checkpoint `f40eff06ad1a31d17fe05b12405d74b8a6a7bf70`
  pushed to `origin/main`; main audit bundle corresponds to this implementation.
- Main diagnostic bundle `9e7a73209b8da4a9006c1e2951fc26038301b167`
  pushed to V2; remote SHA verified.
- Main native-worker safety checkpoint `040296e6ee424a3f9ff37c40ca39da7ee4b8432c`
  and baseline loader audit `45698053fb48ab2219ccde9253e4754e28d68a63`
  pushed to their respective branches; remote SHAs verified.
- Main exact-demand checkpoint `b9c326958d864c3a061b7eb7049220a7aedadb67`
  pushed to V2; remote SHA verified.
- Main exact-planning checkpoint `1a6c3096d355ac5659c391b75c9bca89cf47d228`
  pushed to V2; remote SHA verified.
- Main IEEE service/routing contract `cf017928b051e603a458f1a758a6e1a1c248bb4e`
  and baseline native-asset audit `16570c023a439c884624e7a5bdfa0d8577faf7a3`
  pushed to respective repositories; remote SHAs verified.
- Main independent-watchdog checkpoint `09e910a6faa908c0f9e6e66d7604056e53656671`
  pushed to V2; remote SHA verified.
- Main backend dependency checkpoint `cf774c1c7bb8f6c352d9993eb7401e70975f679c`
  and byte/block admission checkpoint `a08b4943b5826028fd5f4ec1f1774eb610c1aa56`
  pushed to V2; remote SHAs verified.
- Native token timing checkpoint `0013fca361c26643184689c34d0b147dc3176d57`
  pushed to V2; remote SHA verified.
- Native observation checkpoint `ea439a0234ba9d151d90a6d17b20596b54621c47`
  pushed to V2; remote SHA verified.
- Native GPU reference checkpoint `cd5b68deb1e0d37764bb2b7b2bbf4cc07085c3fd`
  pushed to V2; remote SHA verified.
- Guarded-launch checkpoint `e6e76955a8ba20a9ab5b96bb2cc204db9b734426`
  pushed to V2; remote SHA verified.
- Independent frozen-replay checkpoint `5bcb6a333c8ff2e809077af87c58be363c7f14b9`
  pushed to V2; remote SHA verified.
- Startup-parallel ingress checkpoint `4fe05a5cab78bb6dc91ad6e2f85301cf9f330dbc`
  pushed to V2; remote SHA verified.
- Physical GPU union/native census checkpoint `4ab1d1e9fbe7f2f64f0e0f8ce1e8aa954b71d2c7`
  pushed to V2; remote SHA verified.
- Native async scheduler/KV checkpoint `aaa006f57c3af63ec64c6fba1f28a29f5df3a89f`
  pushed to V2; remote SHA verified.
- Fixed-work boundary checkpoint `6ec86c6676ba9dbf39a69c12b697ce51a664af64`
  pushed to V2; remote SHA verified.
- Native demand-load checkpoint `d22721bd6f8966a55e657a3e1dd93e414b94651d`
  pushed to V2; remote SHA verified.
- Controller request-lifetime checkpoint `e238f603281223eae411202a38962488b3e14000`
  pushed to V2; remote SHA verified.
- Offered-failure identity checkpoint `91c9feb258a1c5e649cb808286446b85a7b65c2e`
  pushed to V2; remote SHA verified.
- Selected-request reference checkpoint `7b75ea53636c86b69221047bc5778a703ad2cee6`
  pushed to V2; remote SHA verified.
- Native source-publication checkpoint `4b2adaf3f7e79c303a2d963fe02ed34093dcf76a`
  pushed to V2; remote SHA verified.
- Native storage/observation-class checkpoint `c0724e91316af7b7721a9b6f03b5756db5c32487`
  pushed to V2; remote SHA verified.
- Managed file-read checkpoint `92ad342553dc82f35fbd6677e98f32a1908cfb93`
  pushed to V2; remote SHA verified.

## Serverless evidence checkpoint (provisional model-level attribution)

- Recomputed clean 7B and 3B raw logs, 4,000 successful recorded requests each.
  Median backend-start gaps: 1.002267 s / 1.002198 s; router queue mean:
  236.666 s / 237.200 s. Historical runs are R2 relative to the TC contract.
- Official method with two ready replicas incurs one wait per request;
  minimal repaired method preserves a,b,a,b RR and has no ready-path wait.
- Six no-GPU method tests pass (ready, empty, full, cancellation). Full inference
  cancellation/scale-down and real model pairs are still required.
- Source environment not synchronized/overwritten; vendor's pre-existing staged
  changes preserved; only incremental router change is new in its worktree.
- Existing summarizer extended, not replaced. Plotting extends existing figure
  script. Initial render attempt found matplotlib absent in the inference env;
  use an existing plotting environment, do not install into frozen inference env.
- Plotting uses existing conda base with matplotlib 3.10.0; installed Times New
  Roman faces registered explicitly. Two r2 figures rendered and visually checked,
  3.45-inch width, source data/hashes retained; old figures untouched.
- Existing Serverless launcher uses global tmux and cleanup includes global
  Ray-stop/pkill paths. Do NOT use that path for TC: isolate the session server,
  contain actual workers and scope cleanup to this run before any model pilot.

## Completed evidence: foundational safety (not performance)

- Eight resource-policy/protection unit tests passed.
- Existing stable-environment basic smoke suite: 288 tests passed in 26.917 s.
- First tiny allocation test was throttled at `memory.high` and timed out before
  reaching `memory.max`: 1,737 high events, no OOM. Failure preserved.
- Second test separated hard-limit behavior (128 MiB high=max), verified an
  in-scope OOM kill, child inheritance and cleanup. Production limits unchanged.
- Delegated controllers currently expose memory/pids, not cpuset; do not claim
  CPU controller isolation from accepted systemd properties alone.
- Native Ray 2.54.0 witness: two distinct actors in a 2/3 GiB high/max scope,
  128 MiB object store, no GPUs; actual PID cgroups and affinity agree. Both
  worker PIDs and their scope were absent after cleanup. First attempt failed
  from AF_UNIX path length before worker launch; retained as launcher error.
- No formal GPU run has started. No performance claim follows from these checks.

## P1 demand checkpoint

- Exact ingress-based window fractions replace per-adapter accumulating EWMA /
  doubled registry values; immutable epoch snapshot available to the planner.
- No-GPU reproduction: two equal arrivals previously returned .2/.2 from getter
  and 1/1 from registry; expiry did not clear getter/top-k. Expected .5/.5 then 0/0.
- First full regression had five fixture errors: synthetic stacks bypassed
  construction and omitted the authoritative demand provider. Fixtures now provide
  it explicitly; no production fallback was added to make tests pass.
- 288 existing smoke + 9 exact-demand tests = 297 passed, zero skips/failures.
- All 147 sealed historical entries unchanged after tests. Models were mocked;
  four real GPUs remained idle. No workload/adapter regeneration occurred.
- Nine-formula gap table is an evidence table, not an experimental performance
  figure. No claim that P1, Full qualification or optimization target is complete.

## P1 planning primitive checkpoint

- Reproduced old sub-MiB capacity error: selected 786432 bytes with only 524288
  bytes remaining. New conservative kernel selects none for this counterexample.
- Existing planner extended with immutable measured-cost/demand candidates,
  conditional tier insertion and one-target handoff scan. No new experiment runner.
- Twelve tests passed, including exhaustive subsets for 80 small mathematical
  problems. Combined regression: 309 tests passed, zero errors/failures/skips.
- These are planning primitives, not full live integration. Runtime path still
  needs confirmed per-replica source profiles/footprints and owner reservations.
  Retained legacy scorer is explicitly not IEEE-qualified; no fallback inference
  of missing costs or physical state is permitted for TC qualification.

## P1 service / routing checkpoint

- Thirteen new deterministic tests pass: admission-time classes, EWMA from
  explicit profiles, nonoverlapping intervals, completed-only updates on cancel,
  GPU-hit D=0, strict feasible set, exact binned key and A/A purity.
- Combined regression: 322 passed, zero failures/errors/skips. The 147 sealed
  historical files remain unchanged; all four GPUs idle after the no-GPU checks.
- Existing Router now has an explicit `ieee_confirmed` policy requiring a
  committed snapshot. Legacy affinity/cumulative metrics remain separately named;
  no hidden fallback or performance requalification of old runs.
- Not yet live-qualified: native acquisition and token hooks, snapshot owner,
  physical references and atomic reservations. No GPU replay started.
- Historical Serverless native loader reuse identified: successful four-request
  3B smoke, Python 3.12 store wheels and existing converted checkpoint. Reuse is
  compatibility/asset evidence only (different code and workload), not TC ranking.
  All six existing vLLM files match restored preimages; seventh added loader is
  absent as expected. Do not overwrite the shared environment or borrow global
  cleanup commands from that other project's historical launcher.

## Independent safety monitor checkpoint

- Existing preflight script now provides an OS-resource watchdog in a separate
  bounded auxiliary scope; production runner handshake not yet wired.
- Sixteen policy tests pass. First live witness failed JSONL parsing (final pretty
  JSON mixed into stream); failure receipt preserved, stream contract corrected.
- Second witness: independent watcher, three actual samples, synthetic alarm,
  service parent/child terminated gracefully; resource directory removed.
- Stubborn witness: parent and child ignore TERM, owned cgroup.kill releases them
  after the 10-second grace. No remaining test PIDs. No host pressure induced.
- This is foundational safety evidence only. Actual model/GPU workers, replay
  separation, native Ray telemetry, GPU lifecycle and launch handshake remain open.

## P1 byte/block admission checkpoint

- Existing coordinator has a separately named IEEE entry point with explicit
  immutable backend state; old MB/working-set heuristics are not a fallback.
- Sixteen no-GPU tests pass: trailing completion means, per-request KV rounding,
  unreserved free blocks, distinct pool reuse and physical reservations, exact
  batch/load pressure, native slot and workspace bounds, CapacityOnly safety.
- Functional regression: 338 tests pass in the inference Python, zero errors or
  skips. Safety regression: 18 tests pass in system Python (the qualified watcher
  interpreter). A first combined invocation had one fixture error because conda
  Python lacks `signal.pidfd_send_signal`; no runtime fallback was introduced and
  conda Python is not qualified to launch the external watchdog.
- Watchdog now checks required PID-handle primitives before a witness launch or
  emitting readiness; unsupported interpreters fail closed. Nineteen safety
  tests pass in system Python. This is not a fallback to broad PID-based signals.
- Admission evaluation does not claim a slot, evict victims or publish readiness.
  Native allocation owner/epoch transactions and scheduler hooks remain open.
- At 23:42 local time, vLLM install is still active under the existing bounded
  build scope, progressing through dependencies; no final receipt yet. GPUs have
  not been used for performance. Do not launch a duplicate installation.
- Next: native token-event timing/return path and resource-owner integration;
  actual production launch/replay separation gates still precede any GPU pilot.

## P1 native token timing checkpoint

- Thirteen new deterministic tests exercise native V1 first/last-token boundaries,
  unchanged cumulative token IDs, explicit terminal, missing fields, clock mismatch,
  completion-tail separation, single-token null, numeric RPC transport and actual
  engine/prepared/controller methods. No GPU inference was run.
- Opt-in `timing_contract=ieee_tc_native_v1` enables native stats and rejects
  timing/count fallback. Native mode does not silently retry inside the engine.
- First full regression had one failure from a legacy `__new__` fixture missing
  `model_cfg`; fixture now explicitly supplies the legacy contract. Production
  behavior was not weakened to accept missing timing.
- Final functional regression: 351 tests pass, zero failures/errors/skips. Safety
  suite: 19 pass in the qualified system interpreter. All 147 protected entries
  remain unchanged; four GPUs show 15 MiB and zero utilization after the checks.
- Remaining: native model event qualification, executable acquisition, atomic
  owner/reference/reservation, external replay, aggregate/goodput/lifecycle gates.
- Remote disk rechecked: 149094010880 bytes (~138.9 GiB), still below the agreed
  150 GiB floor; no remote artifact service started or unrelated data deleted.

## P2 native worker observation wiring

- Existing GPU monitor now supplies a read-only official vLLM worker extension;
  engine/RPC/proxy transport preserves nested identity and physical storage facts.
- Five no-GPU tests pass. Whole tensor storage is deduplicated across views;
  registered CPU adapters and GPU slot assignments are explicitly distinct.
- Functional regression after wiring: 356 pass, zero failures/errors/skips;
  147 protected historical entries unchanged, all four GPUs idle (15 MiB each).
- Unsynchronized slot maps are not confirmed readiness; even barrier diagnostic
  observations do not hold dispatch references/reservations or grant launch.
- Real CUDA observation, model identity, clock/cgroup census and resource-owner
  integration remain pending. No performance experiment or additional workload
  was generated for these tests.
- At Sep 26 00:02 local, existing P2 installation is still downloading its
  hash-locked CUDA dependency. Scope is active, all memory high/max/OOM counters
  remain zero; no second install or model run started.

## Native GPU reference checkpoint

- Sixteen added no-GPU checks pass. Native cache pinning is paired with explicit
  unload protection, incarnation/epoch conflicts, shared request references and
  actual generate/prepared/RPC request-ID binding. A CPU-only adapter is never
  implicitly loaded and relabelled as an original GPU hit.
- A bound request without native terminal retains its reference; failure does
  not fabricate release. Device or referenced-cache invalidation poisons the
  owner rather than silently returning to a stale hint. Cold load/admission,
  controller ownership, slow-tier references and abort acknowledgement are open.
- Worker event fences cover its current CUDA stream, not all-device barriers.
  This must pass real stream/thread/model qualification before any confirmed
  readiness claim. Current reference API is TP=PP=1 only and opt-in.
- Final regression: 372 functional tests, 19 safety tests, zero failures/skips.
  Final test count excludes six temporarily duplicated imported TestCase tests.
  All 147 protected entries unchanged; all GPUs remain idle (15 MiB each).
- Sep 26 00:23 local: existing bounded P2 installation still active on the
  hash-locked CUDA wheel. Disk ~338 GiB free; available RAM ~104 GiB. No GPU
  performance run, second installation, model/adapter duplication or cleanup of
  unique evidence was performed.
- Next useful local work: wire bounded service launch + independent watchdog
  readiness handshake and genuinely external replay. Continue native scheduler
  KV/iteration observations and controller reservations after that gate; do not
  stop the active installation or declare P1/Full complete from unit tests.

## Pre-exec launch handshake checkpoint

- Existing wrapper/preflight extended: one bounded auxiliary domain owns the
  supervisor + independent watcher; model process waits in bounded service
  domain until real watcher attachment and first valid resource sample.
- Native engine initialization checks the receipt/worker identity before model
  creation; managed path forbids global process cleanup and hidden fallback
  config retries. Heavy launch refuses an active TC build/service.
- First tiny witness correctly retained as failed: empty scope persisted after
  both processes exited. Fixed lifecycle detection using cgroup populated state,
  then safely stopped only that empty owned scope. Second witness passes; raw
  receipts/logs preserved and hashed. Actual model/GPU qualification still open.
- 24 safety and 377 functional tests pass, zero failures/skips; historical seal
  verifies 147 unchanged entries; four GPUs idle, 15 MiB each.
- Sep 26 00:40 local: installation is demonstrably progressing, now CUDA NVRTC
  wheel (dependency 96 rather than prior 90); original bounded build remains
  active. No duplicate downloads/environment or concurrent GPU run started.
- Next: genuine auxiliary-process open-loop replay and fixed deployment-notice
  origin. Do not merely move a sleeping timer while leaving the replay blocked
  by service admission. Also keep actual native clock/resource census, live
  scheduler/KV owner wiring and remote disk approval on the mainline checklist.

## Independent frozen-arrival checkpoint

- Existing workload module/launcher/runner extended, no new experiment framework.
  Fixed deployment notice + 60-second production origin; separately bounded
  publisher, full original request transport, source/view/content SHA, same local
  native clock. W0/W1 supported; W2 frozen adapter map still required.
- Historical full-trace idle-floor and future-ready adapter look-ahead disabled
  on this TC path. Backlog/demand-rate/candidate control uses received arrivals;
  transport failure never silently returns to a service-local timer.
- Actual 32-request prefix of existing seed42 trace, 8x diagnostic speed: first
  witness throttled parsing under an empty-process 64 MiB high limit; owned stop,
  failed receipt preserved. Replay-specific 192/256 MiB witness passes, 32/32,
  peak 119590912 bytes, high/max/OOM zero, processes released.
- During a 1.50-second receiver stall the independent process still created and
  submitted the due request; max creation lateness 2.527 ms. This is a measurement
  precondition, NOT a model latency/throughput result. Table/summary delivered.
- Final functional regression: 391 passed, zero failures/errors/skips. Safety
  suite: 24 passed; the eight frozen-replay tests also pass in system Python.
  Receipt/raw SHA checks and all 147 protected historical entries verified.
- Sep 26 00:59 local: original candidate installation still active on cuDNN
  dependency 98; current memory ~1.04 GiB, no duplicate install. Four GPUs idle,
  15 MiB each; disk ~338 GiB free. No model campaign started.
- Next: receive requests inside the service domain during initialization, then
  native model/process/clock qualification and live admission owner integration.
  HTTP baselines need identical input timing but separately audited transport;
  current IPC witness is not full common-protocol qualification. See dedicated doc.

## Startup-parallel ingress checkpoint

- Existing main entry now connects and validates service ingress before executing
  the original initialization path. Reception and consumption are independent
  tasks in the service budget, not auxiliary service work hidden from accounting.
- Actual observed arrivals feed the demand window at original receipt times;
  startup backlog is replayed to one observer once, not made artificially fresh
  or counted again at request dispatch. Future/unsorted timestamp inputs reject.
- Actual no-GPU witness: 32/32 existing requests, three received before simulated
  ready, all consumed afterwards; max queued wait 2207.611 ms retained. Peak
  service memory 119566336 bytes, no high/max/OOM; all owned processes released.
- Final functional regression 398 pass, no failures/errors/skips. System safety
  24 pass; all 11 external-replay tests also pass in system Python. Source/raw
  receipt SHA checks pass. Qualification table/curated summary delivered.
- Sep 26 01:11 local: P2 install still live on dependency 98, memory ~1.17 GiB;
  disk ~338 GiB. No heavy concurrent model/build, data regeneration or performance
  campaign. Retain prior failed replay witness; this is a different startup gate.
- Next: native resource/scheduler owner integration and physical GPU lifetime
  measurement, then actual 0.30 compatibility/worker/clock/stream qualification
  after installation completes. Common HTTP input transport, remote disk decision,
  Serverless native qualification and main experiments remain open.

## Physical-time algebra and native census checkpoint

- Existing metrics collector now has physical UUID lease-union arithmetic and
  four mutually exclusive windows. Twelve tests pass, including overlap, TP,
  retained context/lease, zero correct, censored counts and absent owner evidence.
  It is not yet connected to a qualified actual device allocator; legacy billing
  is unchanged and cannot supply new G1/G2 results.
- Existing independent watchdog reads native NVML v3 compute/graphics processes,
  PID birth/cgroup/affinity and raw v2 memory. The exact existing binding is
  SHA-locked and reused without importing CUDA or modifying environments.
- Actual idle/no-model witness: four physical UUIDs, 32/32 old requests received,
  11 periodic GPU observations, 2.868/3.906/5.218 ms min/median/max query duration;
  initial observation 43.799 ms. Peak service 120496128 bytes, high/max/OOM zero,
  owned scope gone and native contexts clear. This is not loaded-model overhead
  or model-release evidence. State table/curated summary and raw SHA delivered.
- First direct binding import failure (missing sys.modules registration) and
  v1/v2 memory-field distinction are documented, not hidden. Nine census tests,
  24 safety tests, 11 independent replay tests; 410 functional tests pass with no
  failures/errors/skips. Preserve previous evidence and all historical data.
- Next: native scheduler/KV owner, allocation-owner events and atomic admission;
  actual model/clock/stream/worker qualification after the live P2 installation.
  Original install has finished the cuFFT download (dependency 100); no duplicate
  install or concurrent model. Remote disk decision and formal matrices pending.

## Native scheduler/KV observation checkpoint

- Existing runner/RPC now opt into an exact-version AsyncScheduler observer.
  Official scheduling/allocation policy remains native; no synchronous scheduler
  substitution. It reads owner-thread unfinished requests, real KV assignments,
  block-pool free count and the latest unretired iteration's scheduled tokens.
- Source review identified async preemption retaining stale in-flight output
  after resetting computed progress. Observation now subtracts only current
  in-flight work; deferred blocks remain owned in the native free pool. Historical
  generated-prefix recomputation is separately visible, not silently added to
  the paper's prompt term or claimed as a physical safety guarantee.
- Sixteen no-GPU tests pass. Final full functional suite: 426 pass, zero errors,
  failures or skips. System safety/census/replay suite: 44 pass. An earlier 425
  pass run preceded the additional stale-preemption/resume case; use 426 as final.
  All 147 sealed historical entries remain unchanged; four GPUs idle at 15 MiB.
- This is not atomic admission: controller-side pending reservations, allocation
  owner, slow-tier references and native model/stream/clock qualification are open.
  P1-D7 provides the evidence/limitation table; no performance figure is fabricated.
- Sep 26 01:55 local: original bounded P2 installation active on cuSOLVER (103),
  ~1.64 GiB current memory, ~337 GiB local disk free, ~104 GiB available host RAM.
  No duplicate installation, concurrent model run or regenerated workload.
- Next mainline: actual backend qualification once install completes, plus owner
  transaction integration and common generation/measurement qualification. Keep
  Serverless first in baseline order; remote disk decision remains pending.

## Fixed-work boundary checkpoint

- Closed legacy fallback inside fixed-output prompt preparation, missing source
  target substitution, output-target shrink and disabled-LoRA base-model output.
  Native chat rendering is explicitly frozen rather than chosen after exception.
  Existing subprocess RPC now preserves the canonical prepared request unchanged.
- Native input token IDs are required and SHA-recorded alongside output IDs.
  This is input evidence, not proof of adapter correctness or cross-system parity.
  HTTP transport/frozen-tokenizer/full-pool qualification remain open. Baseline's
  dirty replay file was inspected, never modified or staged in this checkpoint.
- Eight added no-GPU tests; full functional suite 434 passed without errors,
  failures or skips. First targeted run: one fixture expected a later missing-
  reference error while LoRA was disabled; fixture now tests enabled LoRA and a
  missing reference explicitly. Safety/census/replay tests and historical seal
  are verified separately before backup. No model performance run occurred.
- Original install has progressed to cuSPARSE dependency 104 after completing
  cuSOLVER. Keep its live scope; do not restart, duplicate or overlap model work.
- Next: finish native owner/controller integration, then actual backend/clock/
  stream/worker qualification. Inspect common frozen input with the final
  tokenizer before baseline runs; Serverless remains first. Formal M1/M2,
  ablations, motivation and sensitivity evidence remain unmeasured.

## Native demand-load transaction checkpoint

- Existing native reference owner now has a separately named demand-load path:
  original native loader/LRU -> current-stream completion -> CPU/GPU pin, without
  yielding the worker thread between operations. Existing hit-only acquire still
  never loads. Capacity conflicts have no eviction/load side effect.
- Original source facts survive in the receipt; CPU-registered promotion is not
  a GPU hit. Integer/name/path and dispatch lease identity are checked, including
  retries, generation binding and post-eviction reuse. Native zero-module load
  is rejected; failure never publishes readiness or pretends rollback succeeded.
- Actual dense buffer layout is checked before deriving padded slot footprint.
  Empty slot bytes are logical reuse inside a physically allocated pool, not
  additional free device memory. Unknown views are not approximated by file size.
- Thirteen added tests; full functional regression 447 pass with zero failures,
  errors or skips. Separate system safety/census/replay 44 pass. All 147 protected
  entries unchanged; four GPUs remain idle, 15 MiB each. No model inference.
- Sep 26 02:23 local: original P2 installer still active at dependency 105,
  ~1.91 GiB memory; local disk ~336 GiB, host available memory ~104 GiB. No second
  install or concurrent model. Keep the same live installation and receipt paths.
- This is not proactive admission or Full integration. Controller-side request/
  adapter reservations, native physical/KV ownership, slow-tier references and
  actual worker/clock/stream/model qualification remain the next mainline tasks.
  Baseline order remains Serverless first; remote disk decision remains pending.

## Controller request-lifetime checkpoint

- Three actual-runner/fake-inference regressions reproduced capacity leaks on
  resolution failure, resolution cancellation and an immediate post-reserve error.
  All failed on the prior implementation (active count remained 1 instead of 0).
- Request ownership now starts at reserve and retains original adapter identity;
  shared requests release only their own counts, batch completion occurs once,
  fixed-output resolution cannot silently switch to backbone inference.
- Native cancellation without terminal evidence retains counts and withdraws the
  replica; per-request unresolved ownership survives raw/aggregate summaries.
  Shared `last_timing` cannot supply terminal acknowledgement. This does NOT
  implement native abort completion or prove physical GPU release.
- A follow-up transport check caught Boolean terminal flags being coerced to
  floats. Both parent RPC timing sites now preserve typed evidence; numeric/null
  values do not become true acknowledgements. Fourteen added checks in total;
  final functional regression 461 pass, no errors/failures/skips. The earlier 460
  pass result preceded this transport check. Separate system safety/census/replay
  suite: 44 pass. No real inference was performed.
- Sep 26 02:46 local: original bounded installer still active on NCCL (112),
  memory ~2.15 GiB; disk ~336 GiB, available RAM ~104 GiB. All four GPUs idle at
  15 MiB; 147 protected entries verified unchanged. No duplicate installation,
  regenerated workload, old result overwrite or new GPU performance claim.
- Next: preserve original offered request identity through outer replay failures,
  then finish owner/source/admission integration and actual backend qualification
  once installation completes. Keep Serverless first in baseline order. Remote
  service disk decision, M1/M2, ablations and sensitivities remain pending.

## Offered failure identity checkpoint

- Two prechange checks reproduced anonymous `error/error` rows and one-request
  cancellation escaping as whole-replay cancellation. Continuous collection now
  binds each task to its original trace and prepared input. Empty/duplicate input
  IDs and wrong returned identities reject; no new trace/artifact generation.
- Outer exceptions and native execution failures retain original input identity;
  known dispatch/tier evidence is kept for the latter. Unobserved token/latency/
  cost values are null, not zero. Failure-observed time is not a first token or
  native completion, and does not imply physical GPU release.
- Global cancellation still propagates; a prematurely ended publisher cannot
  manufacture outcomes for unoffered requests. Durable partial-result journal
  and common native-terminal/resource accounting remain open.
- Nine added checks; final functional regression 470 pass, no failures/errors/
  skips; separate safety/census/replay suite 44 pass. Intermediate fixture mismatch and failed-input initialization errors
  are documented in P1-D11, not hidden. The 147-entry historical seal remains
  unchanged; no model performance run, new baseline point or ranking claim.
- Sep 26 02:59 local: original isolated backend installation remains active on
  NCCL (112); local free disk ~336 GiB, available RAM ~103 GiB. No second installer
  or concurrent model. Continue the same install; quiet wheel output is not a
  reason to restart it.
- Next: connect controller execution to native source/reference ownership and
  atomic admission, including cancellation reconciliation. Actual backend/
  worker/clock/stream qualification follows installation. Serverless remains
  first among baseline model runs. Remote disk decision and all formal matrices
  remain outstanding; do not treat correctness tests as completed experiments.

## Selected-request native reference checkpoint

- Three prechange actual-runner checks reproduced the missing controller-to-
  native reference path (two failures and one timeout). Selected requests now
  acquire a source-bound native lease before generation, validate per-request
  terminal identity, and return controller capacity only after native release.
- Stale epochs use the explicitly returned native epoch; unknown acquisition/
  release replies retain ownership and withdraw the replica. Two same-adapter
  requests retain independent shares. Resolve-after-acquisition evidence is NOT
  a committed pre-dispatch snapshot; no GPU-hit D=0 or Full qualification claim.
- Two more prechange checks reproduced transport duplicate execution and reuse
  of a cancelled in-flight connection. Native RPC no longer blindly retries;
  shutdown removes the interrupted socket from reuse, not native work from the
  physical ledger. Actual local socket-pair cancellation check passes.
- Fifteen added checks; final functional regression 485 pass, no failures/errors/
  skips. Separate safety/census/replay suite: 44 pass. First full regression had
  one incomplete test fixture; corrected explicitly, no production fallback.
  All 147 historical seal entries unchanged. D12 contains the evidence table;
  no new model performance experiment, trace, adapter or baseline ranking.
- Sep 26 03:17 local: original bounded P2 installer remains active, now OpenCV
  dependency 120 after NVVM; current memory about 2.5 GiB, local free disk about
  336 GiB, available RAM about 104 GiB. Four GPUs idle at 15 MiB. No duplicate
  install or concurrent model; continue observing the same attempt.
- Next: committed native source/cost snapshots and controller routing/admission
  integration. Known capacity conflicts still need dispatcher wait/reselection;
  unknown native work needs actual terminal/release reconciliation, not guessed
  timeouts. Keep slow-tier references, proactive E(t), native model/stream/clock
  qualification and physical GPU allocation-owner events open. Serverless first
  among baseline runs; remote disk decision and formal matrices still pending.

## Native source identity/publication checkpoint

- Before modification, one test reproduced silent same-ID CPU object replacement;
  another failed because completed source snapshots did not exist. Native owner
  now binds weak object identity, publishes GPU sources after its completion
  fence and withdraws them before the original native removal callback clears
  the slot. Native victim choice is unchanged; valid CPU sources remain visible.
- Remove/reactivate with the same ID/slot cannot keep its prior GPU confirmation.
  Metadata does not retain evicted CPU weights. Snapshots do not pin, touch LRU
  order or infer source names for unowned native IDs; absence is not Remote.
- Existing worker/engine RPC and selected-request path commit an immutable,
  validated owner/epoch/clock view to InstanceSlot. Delayed old views cannot
  replace newer state. This is still resolve-after-selection observation, NOT
  pre-dispatch routing or D=0 evidence. Multi-replica source/cost assembly and
  live routing/admission remain unfinished; old hints are not relabelled.
- Seventeen added no-GPU checks; final full functional regression 502 pass,
  zero failures/errors/skips; separate safety/census/replay 44 pass. Intermediate
  missing-entry exception and invalidation-reason differences are documented in
  D13. The 147 protected entries are unchanged. No model performance result,
  new trace/adapter, formal comparison or measured ranking was produced.
- Sep 26 03:36 local: original P2 installation active at tokenspeed-triton (182),
  after completing TileLang and tokenizers. All memory high/max/OOM counters
  remain zero; local free disk ~335.5 GiB, available RAM ~103.5 GiB, four GPUs
  idle at 15 MiB. No duplicate installation or concurrent model.
- Next: actual source-class footprint/cost composition and pre-decision routing,
  owner reservations and known-conflict re-selection. Retain native abort/release
  reconciliation, slow-tier references, proactive E(t), physical GPU allocation
  events and actual 0.30 model/clock/stream/worker qualification on the mainline.
  Serverless remains first among baseline model runs. Remote service disk gate
  and all formal M1/M2, ablation and sensitivity matrices remain pending.

## Native source footprint / observation-class checkpoint

- Existing worker monitor inventories actual CPU A/B tensor storage and sharing,
  alongside the previously checked GPU dense pool. Tiny real CPU tensors verify
  768 bytes total for adapters whose individual footprints sum to 1280, only
  256 bytes removed with the first adapter, and 512 bytes retained by 16 bytes of
  views. These are arithmetic witnesses, NOT actual 7B/3B footprint measurements.
- Source RPC carries the native inventory into the immutable controller view.
  Source classes use actual HOST storage or padded GPU slot bytes, rank, dtype,
  packed/pinning representation and the existing request/load bins. Missing
  measured footprint rejects cost classification; no default file-size estimate.
  HOST storage capacity is not RSS, allocator overhead or a budget reservation.
- Detailed tensor inventories are not duplicated into every request's evidence;
  selected-source description and owner totals remain. Actual observation/RPC
  cost and update cadence still require model-level qualification before freezing
  a Full configuration. Multi-replica routing is not yet driven by this view.
- Twelve added checks; final functional regression 514 pass, no failures/errors/
  skips; separate safety/census/replay 44 pass. Prior 513 pass did not include the
  final request-evidence transport check. All 147 protected entries unchanged,
  four GPUs idle at 15 MiB; no inference or new adapter/workload data generated.
- Sep 26 03:53 local: same P2 installer alive at torch dependency 183, following
  completed tokenspeed-triton. Memory high/max/OOM events remain zero; disk about
  335 GiB free, host available memory about 104 GiB. No second install, driver
  change or concurrent model job. Remote disk decision remains pending.
- Next mainline: managed HOST/NVMe source references and cold-source metadata,
  real source-cost initialization/composition, pre-decision routing and atomic
  reservation/admission. Native abort/release reconciliation and physical GPU
  owner integration remain required. After setup, qualify actual 0.30 model,
  worker, clock and stream. Serverless stays first among baseline model runs;
  formal M1/M2, ablations, motivation and sensitivities are still unmeasured.

## Managed file-read reference checkpoint

- A prechange actual-runner check reproduced source deletion while its native
  loader was about to read the directory. The existing manager now shares a
  cooperative path owner between reference acquisition and physical mutation.
  Shared readers, parent/child deletion, replacement, synchronous copy versus
  another reclaim thread, deletion failure and unchanged eviction accounting
  are tested with tiny temporary files, not model inference.
- Selected native requests acquire file references before cancellable RPCs.
  Acknowledged completed load releases the file reference; native tensor leases
  protect subsequent generation. Lost load replies retain the unresolved file
  and native ownership. Read-only snapshot failure does not fabricate work.
- This is NOT complete managed-tier qualification: content SHA, copy publication,
  physical budgets, remote materialization/direct-preload writers, and all-tier
  registry transactions remain open. External LocalCache with an independent
  reclaimer is explicitly unsupported by this source-reference entry point.
- Eleven added checks; final full functional regression 525 pass with no failures,
  errors or skips; independent safety/census/replay 44 pass. An intermediate
  47-test run had one logger API error, corrected to the existing logger contract.
  The 147 protected entries and original plan SHA are unchanged. No formal model
  comparison, ablation, regenerated trace/adapter pool or performance claim.
- Sep 26 04:08 local: original P2 install remains active downloading torch (183).
  The 3 GiB high limit has now caused 136 reclaim/throttle events; max/OOM/OOM-kill
  remain zero. Its sampled memory is mostly file cache (~2.77 GiB), anonymous
  memory ~151 MiB. Host available memory ~104 GiB, free disk ~335 GiB; four GPUs
  idle at 15 MiB. Preserve this same attempt, no duplicate install or model run.
- Next: continue physical source-owner integration and native request/abort
  reconciliation, then actual backend/clock/stream/worker qualification as soon
  as setup completes. Pre-decision measured costs/routing and atomic admission
  remain necessary before Full. Keep Serverless first among baseline model runs;
  remote disk gate and all formal M1/M2, ablation and sensitivity matrices pending.

## Completed-file publication / actual transfer lifetime checkpoint

- Two prechange tests reproduced loss of the previous valid directory on invalid
  archive and interrupted extraction. HTTP and managed local directory copies
  now stage privately on the destination filesystem, then publish through the
  same source owner. Rename failure restores the previous copy; failed restore
  preserves its recovery directory and reports failure rather than deleting it.
- Native HTTP uses an in-service worker thread, so blocking urllib is no longer
  directly run in the request event loop. Cancellation is cooperative and joins
  the actual executor Future; repeated cancellation cannot release a live writer.
  Active materialization prevents whole-tier reset, while old copies remain
  readable and active read leases can reject replacement. No simulated delay.
- Initial full-stack tier reset/copy and ExperimentStack's direct preload copy
  now use existing manager ownership/publication. Caller-provided legacy copy
  callbacks, old tier hints and unqualified legacy scenarios are not promoted
  to confirmed-state evidence. Native transfer exceptions are not replaced with
  local fallback or a zero-time failure value.
- Thirteen added checks; final functional regression 538 pass, no errors/failures/
  skips. The related 62 tests include a tiny real loopback HTTP roundtrip; other
  cancellation tests control the response object. This is not 174 remote-server
  qualification or a GPU performance experiment. First system-Python import
  failed on absent numpy; stable existing env used without dependency installation.
- Independent safety/census/replay 44 pass. The 147 protected entries and plan
  SHA remain unchanged; all four GPUs idle at 15 MiB after validation.
- Sep 26 04:18 local: original P2 scope active at torch dependency 183; 304 high
  events, zero max/OOM/OOM-kill. Free disk ~335 GiB, host available memory ~104
  GiB. Same installer retained, no duplicate environment or concurrent model.
- Next: physical source capacity/content/registry publication, measured cold
  source classes and pre-decision routing/admission; native abort reconciliation
  and physical GPU ownership remain open. Account retained mmap/page-cache/tensor
  storage separately from unlink. Qualify real backend/worker/clock/stream when
  installation completes. Serverless remains first in baseline order; remote
  disk decision, formal comparisons, ablations and sensitivities still pending.
