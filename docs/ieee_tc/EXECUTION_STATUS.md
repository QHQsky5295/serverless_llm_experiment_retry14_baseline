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
| Resource containment / safety | Primitive, native Ray inheritance and independent-watchdog tests passed; production gates pending | RESOURCE_QUALIFICATION.md. Both graceful and stubborn owned trees released. Model workers, launch handshake and replay separation still pending. CPU proof is affinity, not delegated cpuset |
| Protected historical artifacts | Sealed and verified | `paper_results/ieee_tc/safety/20260925_execution_start_protected.json`; old results and selected user modifications unchanged |
| Remote authentication / management | Key login verified; service qualification pending | Strict host checking, dedicated restricted key; remote disk 138.9 GiB below 150 GiB floor, user decision pending. See REMOTE_ACCESS.md |
| P0 main-table / Full provenance | 7B source-pair audit complete | Same trace/subset SHA, different execution; 202 scalar fields preserved. P0_FULL_PROVENANCE.md; no performance rerun needed for this finding |
| Serverless wait audit | Historical audit + no-GPU control-path tests complete | Clean 7B/3B logs reused; six real-method AST tests pass; incremental ready-before-wait patch preserved. Model pair pending |
| P1 IEEE semantic alignment | Mathematical contracts + native token/reference interfaces tested; controller/resource integration open | P1_FORMULA_IMPLEMENTATION.md D1–D6. GPU references bind actual engine requests and native LRU; CUDA/clock/stream qualification, slow-tier references and atomic admission remain open; no Full performance qualification |
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
1. Continue native owner integration: measured source-class costs, resource-owner remaining budgets, then atomic routing/admission. Native token-event timing is wired opt-in (P1-D5), but executable acquisition, actual engine clock census and external arrival/submission remain open. Do not declare the historical scorer IEEE-aligned.
2. Integrate effective service-worker and external-watchdog checks into existing launchers; native single-raylet witness is not the full deployment gate. Qualify Serverless native checkpoint path before its original/repaired model pair.
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
