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
| Serverless wait audit | D65 actual two-raylet containment and D66 native-input identity pass | Existing3B170 packed tensors exactly match254HF tensors; complete96-file store bundle and actual native imports verified. Native store/model loading and original/repaired model pairs pending; no performance claim |
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

D66 completes reusable native Serverless input qualification. All170 packed
FP16 checkpoint tensors match all254 source tensors, complete6,425,499,648B;
metadata/tokenizers equal. Main CSV170rows and JSON delivered.24 launcher/router/
layout and3 native CPU fixture checks pass. Native-input preflight also passes:
6overlay preimages/official patch, complete96-file existing store bundle,
actual9f50241 backend/native imports and native-format EngineArgs. No install,
model load, new pool/trace or remote service. One pre-service relative-output
launcher error and one missing-libglog import retained; both corrected without
guard or baseline policy changes. Explicit bundled-library selection fixes its
obsolete build RUNPATH. All owned scopes cleaned; backup receipt below.

Next actual guarded native store/model loading, not another preflight/identity/
Ray-only audit. Reuse exact checkpoint, qualified native launch view, existing
compiled store and reversible loader-only port. Exact selection/readback is in
native_inputs_preflight JSON and baseline CONTAINED_LAUNCH. Install with new
backup/receipt; restore only after actual owned worker exit. Verify store UUIDs,
registration/load confirmation, actual model-worker containment and cleanup.
Then proceed to the approved original/repaired pairs. Full startup/remote/
correctness gates below remain; no formal baseline/M1/M2/ablation/sensitivity
performance result or optimality claim. Do not rerun successful checkpoint,
allocator, old prefixes or infrastructure witnesses.

D65 actual Serverless infrastructure witness now passes (attempt4), cleaned and
immediately tabulated. Do not repeat it or treat it as a model/performance run.
Both live raylet capacities total8GiB; five native workers and their children
share the actual72/80GiB service group and40CPU affinity. All cleanup passes.
Three launcher-error attempts are retained, separately classified from baseline
failures. The current source fixes argv abbreviation, results-parent symlink
identity and explicit-resource JSON expansion; original native scripts untouched.
Private dead-pane retention preserves early failures. Next native checkpoint/
overlay identity and actual store/model loading, then the two-model1000-request
original/repaired pairs. Native Ray memory-monitor inspection is complete: both
logs show0.99of134626840576B host RAM, not shared80GiB. Leave native semantics
recorded/enabled and retain the qualified OS cap/watchdog; no Ray rebuild or
extra infrastructure-only rerun is justified. Logical memory is neither an
enforceable budget nor actual allocation. Full/remote/correctness gates
below stay open. No baseline/M1/M2/ablation/sensitivity performance result yet.
Final source/evidence backup details are recorded in D65 below.

D64 rechecked actual Full profile dependencies: current serial GPU/HOST samples
cannot initialize unmeasured file/remote/concurrency classes, particularly after
the allocator change. No fabricated profiles or startup-guard removal. Fresh
remote174 read-only check:148732485632B available (about138.52GiB), no listeners
on18080/18081; both authority questions were asked again and remain unanswered.
Do not start the remote service or replace the all-zero3B correctness pool.

While those Full prerequisites remain blocked, a bounded Serverless prerequisite
advanced: exact native launch scripts now have an exclusive TC view, without
modifying historical originals/installed packages or using global cleanup.
The real selected environment is sllm_vllm0102_newserverless_20260518 / Ray2.54,
not the legacy default Ray2.48 environments. Baseline checkpoint
ef49691e76c5fd1a48db11b158ab1c3177d93940 is pushed and fresh remote SHA verified.
Read ServerlessLLM_new_project/ieee_tc/CONTAINED_LAUNCH.md in the baseline repo.
Next qualify actual two-raylet worker containment/capacity and native loader
using this view and the existing gated launcher; this is not permission to
skip native overlay/checkpoint identity or run formal comparisons. No more
standalone prefix, allocator or index repeats. Return to representative Full
profiles/lifecycle once remote/correctness requirements can actually be met.
No native Ray/model/remote/performance service was started in D64; the two CPU
test domains are empty/stopped. Baseline/M1/M2/ablations remain not started.

D63 independent7B HF/PEFT reference completes and is cleaned. Exact256tensors
per adapter match existing weights; input393tokens matches old native SHA.
All5 matched references close, but all20 correct/wrong comparisons also close
under the same preregistered tolerance. This test cannot identify native adapter
application; do not tune tolerance or repeat the same prompt. HF itself shows
nonzero A/B effects and exact zero/base and A/A at this position. Immediate
20-cell table/CSV/JSON is delivered in NATIVE_ADAPTER_NUMERIC_CONTROL. Full,
500-ID semantic and real-remote qualifications stay OPEN. Current backbone SHA
is now available; historical native SHA cannot be backfilled. Next actual
Full/profile integration, with native identity/slot/arithmetic evidence and
representative preparation/service measurements; unqualified samples are not
correct-request claims. Use ONLY the two D62 candidate content indices in
inputs/README. No more content rescans, allocator-only checks or old prefixes.
D60 allocator candidate/fixed HOST allowance remains; Full guard stays until
its actual unmet requirements are satisfied. Remote174 below150GiB and nonzero3B
authority choices remain. No formal baseline/M1/M2/ablation/sensitivity begun.
70 safety and288 smoke checks pass; all D63 owned scopes empty/stopped. Tested
diagnosticd81f78cbdb200cdc7375cf6ad7daee8d32aaf10c pushed/full remote SHA checked;
evidence backup follows below. Goal active, no formal optimality claim.

D61 connects actual asynchronous HOST-capacity observations to existing deferred
work on the frozen control cadence. No waiters means no RPC; unchanged bytes
mean no new physical load. Actual A (budget) or A-X (protected workspace) must
improve; cancellation/owner/clock checked, all original admission checks remain.
943 functional,16 installed-native CPU and62 safety checks pass; no GPU/profile/
Full experiment this turn. P2 D61 contains the immediate correctness table and
limits. No more allocator-only microtests or old prefixes. Next representative
measured profiles and actual Full/multi-plan/lifecycle qualification, using the
explicit D60 candidate and fixed HOST allowance. Full guard and both unanswered
artifact/remote authority choices remain. Formal baselines/M1/M2/ablations are
not started. Tested checkpoint34ac569cf8d3ee38013efe499193a8febbd3d6fd is pushed
with fresh full remote SHA verification. All four owned scopes are empty/stopped;
no experiment remains running. Receipt below; goal active.

D60 second/final local comparison completed and cleaned: official background
event processing returns all6classes' native-copy pinned bytes at the original
post-delete observations, before the next real load. Exact GPU content matches;
all queries completed, no in-flight/private-pool/latency-bound claim. See P2 D60
and curated20260927_host_copy_background CSV/JSON alongside unchanged D59.
Opt-in uncached_background_v1 now has a distinct frozen model/worker/readback
identity and uses the same fixed HOST partition/actual-byte checks. No production
config or B/C selected; old profiles cannot be relabeled as the new candidate.
No more allocator-only microtests or old request prefixes. Next use this candidate
in representative Full/profile integration, explicitly address actual observed
HOST return waking deferred work, then actual Full/lifecycle/A4. Current release
events are not proof of asynchronous allocator-return notification; don't poll
unconditionally, assume future bytes or raise the budget. Full guard remains.
Tests/backup receipts below; formal baselines/M1/M2/ablations remain not started.
The two unanswered artifact/remote authority choices remain. Goal active.

D58 connects an opt-in native HOST workspace partition inside the existing
allowance, based on all audited existing artifact classes. Actual total occupancy
protects native cache growth and one demand load from proactive staging; no
early eviction, budget/cache growth or assumed allocator release. Controller
aliases share one frozen contract; acknowledged plan closure wakes waiters.
935 functional,59 system-Python safety and30 installed-native CPU checks pass;
CUDA uninitialized. Two-pool bounds independently recompute from D56/audit SHA.
See P2_BACKEND_QUALIFICATION D58 for the formulas, correctness table and limits.
Next is actual native copy/in-flight HOST return and representative Full/profile
qualification, not another default/uncached CPU microtest or old request prefix.
Actual multi-plan liveness and Full are not qualified; no production B/C chosen.
Full guard and the two unanswered artifact/remote authority choices remain.
All5 owned scopes empty and stopped; protected147 and plan unchanged. Tested
checkpoint e2ba9f6f8a76d0e0ce02e87f2f097161dd64bd83 is pushed with fresh full remote
SHA verification. No experiment is running; this subsequent receipt is docs only.

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

### D66 reusable native checkpoint and loader composition

- Full plan/status/AGENTS reread after recovery; run-experiment/github-sync and
  academic-plotting used. Start main ebfbc0594468299530bf6c01b0cbdc117ec64460,
  baseline03a9fd7e92261673bda748ac9c897065fc0067bb; original user dirt preserved.
  No agents. D65 remains passed, not repeated. Formal baseline/M1/M2/ablations
  and sensitivities not started. Two external authority choices stay open.
- Read actual TC9f50241 backend/store and historical0fd00ca native-loader smoke,
  exact loader-only reversible overlay, compiled store receipt and native tensor
  format. Browsed primary Serverless loader/backend and vLLM0.10.2 Llama source.
  Other project's routing overlay and ordinary direct loader are not substitutes.
- Existing native3B file6,425,499,648B has170 packed FP16 tensors versus254 source
  HF tensors. Added complete-element bounded CPU identity audit to the existing
  source adapter; no conversion/pool generation.24 launcher/router/layout tests
  and3 pinned-native CPU fixtures pass, no failure/error/skip. Wrong/reordered
  weights, missing source keys and changed tokenizer are rejected. Actual large
  checkpoint audit and loader composition preflight follow; no model yet.
- Baseline comparator checkpoint2aec04be304c1314f8145ee48524bf8ff15a73e7 pushed;
  fresh origin/main full SHA matches. First audit launch stopped before service:
  outer receipt path was relative, rejected by existing absolute-output rule.
  Preserve console/exit1 as protocol_or_launcher_error; no native file audit
  executed, no source fix/guard relaxation. Retry uses explicit absolute paths.
- Actual attempt2 complete:170/170 native packed tensors,254/254 source tensors,
  6,425,499,648B exact after FP16 conversion, all5 shared small files equal.
  Native SHA3f937cdc2c3b637cf61a19809670a0e6146b943a049ba407fa5dc5e2b46f9979.
  Audit93.656s is not startup latency.95watchdog samples,peak6769774592B,
  minimum host available110954188800B,swap/high/max/OOM0,CUDA uninitialized.
  Service/watchdog0/0, service released; aux4704cd42cbc54798a30f3c8a4d67eee3
  actual process list empty/events0 then stopped. First-error aux likewise.
  Complete170-row CSV and summaryJSON delivered before next task. No native
  model/LoRA/performance qualification follows; reused data unchanged.
- Loader-only preflight passes6preimages/official patch, with no installation.
  Complete96-file native store bundle exactly matches preservedSHA/member closure.
  First actual CPU import fails missinglibglog.so.1; ELF RUNPATH points to obsolete
  build staging. Existing bundled library selected explicitly (no rebuild/copy)
  and second import succeeds. ActualTC9f50241 source, native storeTorch/C/C++,
  grpc1.76,torch2.8+cu128,vLLM0.10.2,Ray2.54 imported; nativeFP16/TP1 EngineArgs
  selects audited checkpoint, engine None andCUDA uninitialized. Both CPU groups
  actual empty/events0 then stopped. Curated input-preflight JSON and immediate
  table include paths/SHA/all attempts. No actual model/native load qualified.
  Next actual owned native loading, not another CPU/Ray/checkpoint repetition.
- Final main288 smoke checks pass,0failure/error/skip; owned group actual empty,
  memory.high19612 under its CPU-test1GiB soft limit, max/OOM0, then stopped.
  This is CPU-test reclaim pressure, not native-audit or model performance.
  All D66 owned groups cleaned, no model/TMUX
  remains. Complete170 curated rows/totals independently reconcile to original
  native audit; raw/launch/watchdog/source/import file hashes all match.147
  protected artifacts and authoritative plan unchanged. GPUs15MiB/0%, available
  host108887676KiB, available disk353281323008B; no artifact conversion/copy.
- Baseline source checkpoint2aec04be304c1314f8145ee48524bf8ff15a73e7 and evidence
  document9b12427ca7bcf8b7f20771fe0ecd671930029cf6 are pushed; fresh full origin/main
  SHA matches9b12427. Original dirty replay/relayserve files excluded. Main backup
  includes only this status and three new curated CSV/JSON files, after diff/
  secrets/protected-name checks. Goal remains active; no formal performance,
  native loader or numerical LoRA qualification is claimed.

### D65 actual Serverless two-raylet qualification

- Full plan/status/AGENTS and run/backup/academic-plotting skills read, including
  recovery. No agents, new model/pool/trace, remote service or formal replay.
  Existing native source adapter reused; exact Ray2.54/commit import required.
  Preregistered two live raylets,4+4GiB capacity,1head+4worker actors and children,
  actual cgroup/affinity, private cleanup. No CUDA/model/store/API inference.
- Attempt1 failed outer argparse abbreviation of child --host before service
  start. Main parser now allow_abbrev=False; actual outer+nested argv test and
  71 safety/census/replay checks pass. Fixc8fd70c702a36e5cecb133f373a2304c06d85002
  pushed/full remote SHA checked. Attempt2 stopped before Ray: logical/physical
  output alias mismatch. Canonicalize once, preserve alias, test real symlink;
  21 baseline checks pass. Baselineff2b268df048e90a44c4adbfa751d97869c0eeba
  pushed/full origin/main SHA checked (earlier witness source3a642e7 also saved).
- Attempt3 head exits before ready. Shell-only reproduction proves explicit
  resource JSON gains an extra closing brace; original native stderr was lost
  when its pane exited, so do not quote that run as direct error-text proof.
  Generated leaf now separates defaults from parameter expansion, tests both
  explicit/default actual argv. One hash-bound private tmux config preserves
  exited panes. No scheduling/loader/resource-budget changes. 22 CPU checks pass.
  Attempts1–3 are protocol_or_launcher_error, not performance/system failures.
  All owned service/auxiliary groups clean, memory high/max/OOM0, then stopped.
- Attempt4 actual witness and guarded launch both pass:2live raylets, each
  4294967296B object-store capacity;5distinct native actors and5children share
  group/40CPU affinity; all4logical GPU IDs covered. Actual raylet command lines
  agree. No CUDA/model correctness follows. Service/watchdog0/0, private tmux0,
  service path removed and native contexts clear. Auxa5c0a5cf531f4ae5963877337c7c6baf
  actual process list empty/events0, then stopped. No experiment left running.
-11 watchdog samples, peak1324113920B, minimum host available110455627776B,
  swap0/high/max/OOM0. Separate witness peak1347006464B; no inference-footprint
  claim. Actual Ray logical memory107262640128/106816192512B exceeds shared80GiB
  cap; native memory-monitor semantics need inspection, OS cap not waived.
  Immediate exact table in baseline CONTAINED_LAUNCH plus main curatedCSV/JSON;
  no artificial performance plot. Rawwitnessf6d16cbee35c000ffc00b5898bc3b6791dfcd439aadaf27fedb074dd1b331511,
  launch8a8ddbe98be63248fbe38fc5dfd3ffed6db0082b5786d07e974ef978e14e2d5c.
- Next native loader/store/checkpoint and real model-worker qualification, then
  original/repaired pairs; do not repeat this infrastructure-only witness. Remote
  disk/nonzero3B authority choices remain, Full guard stays. Formal comparisons,
  ablations and sensitivities not started. Final protection/backup follows.
- Read same actual raylet logs and officialRay2.54 memory-monitor source: native
  threshold133280571392B,total134626840576B (host-based), not service-aware.
  Preserve native monitor and documented OS/external safety; no new memory-loop
  prerequisite. Both original logs retained in attempt4 with exact SHA. Reuse
  existing official-loader port/checkpoint; no installed-source mutation here.
  Other project's M4 routing overlay is NOT a loader and must not be imported.
  Its historical0fd00ca witness differs from TC audited9f50241 source; qualify
  the actual selected composition, never relabel the other project's success.
- Final baseline checkpoint03a9fd7e92261673bda748ac9c897065fc0067bb is pushed;
  fresh origin/main full SHA matches. Exactly3 named baseline task files;
  original dirty replay/relayserve work excluded. Final main288 smoke checks
  pass0failure/error/skip (earlier71 safety checks and baseline22 pass); all
  test domains actual empty/events0 and stopped. Protected147/plan unchanged.
  Curated values reconcile with raw witness/receipt; all4 launch-evidence SHA
  values and actual measured helperSHA7340b753af523ea675d81418edbafa2d6c6deb9f019b48d0a36a4aafea00a7cd
  match. Original raylet logs retained with exact SHA. Final GPUs15MiB/0%,
  MemAvailable109140492KiB, local available353304219648B. No TMUX/model remains.
  This main data/status checkpoint contains only the two curated files and this
  ledger; named-stage/secrets/diff checks precede its non-force V2 push. No old
  result, pool, trace, manuscript or remote service changed. Goal remains active.

### D64 Full prerequisites and contained Serverless native-launch adaptation

- Main starts6826b9663afaf698f6f19566d6671120af8d65ae; baseline starts16570c.
  Full plan/status/AGENTS and execution/backup skills reread, including after
  compaction. Academic-plotting read before the immediate qualification table.
  Protected147/plan unchanged; original dirty files preserved. No agents.
- Inspected FrozenServiceProfiles/FrozenPreparationProfiles and actual native
  acquisition/initial-deployment paths. Full still needs representative measured
  profiles tied to its configuration and content classes. The D26 serial GPU/
  HOST observations cannot fill missing file/remote/concurrency classes. No
  new constants/profiles, new artifact, algorithm change or guard bypass.
- Fresh strict-key read-only174 query gives148732485632B available, below150GiB;
  no18080/18081 listeners. Existing server creates temporary compressed archives
  per fetch, so a separate artifact-node space rule needs actual peak evidence
  and user authority, not an assumption of zero temporary space. Asked again
  about that rule versus freeing space, and about independent nonzero3B
  correctness fixtures versus limited claims. No answer, mutation or remote
  service launch. Both decisions remain pending.
- Read historical new-Serverless wrapper, native-loader qualification and all
  five existing launch scripts. Legacy start synchronizes installed code and
  calls global cleanup; common per-node object-store8GiB would total16GiB for
  head+worker. This is a conditional code finding, not historical allocation
  measurement. Selected new/native environment actually has Ray2.54.0; old
  defaults have2.48.0. Read installed/official2.54 CLI and spill implementation.
- Baseline adds prepare_ieee_tc_serverless_stack.py, a hash-bound renderer of
  those same scripts, not another supervisor/framework. Originals remain exact.
  Generated stack requires the existing guarded TC service; private tmux/Ray/
  spill/log paths, no source-install/global-stop/log-overwrite, one worker-raylet,
  4GiB+4GiB object store, full store GPU visibility and native storage-aware CLI.
  Direct-path fallback and conflicting legacy memory override rejected. Ports
  checked before launch. Equal partition is qualification-only, not M1 optimum.
  Actual worker capacity/ownership, imports, loader overlay and model correctness
  remain unqualified; generated readiness strings do not establish those facts.
- Initial15 and final17 CPU checks pass, zero failures/errors/skips; overlapping
  selections, not performance repetitions. Eleven new launcher checks plus six
  original method tests. Generated native CLI arguments are captured with a
  non-Ray recorder; no Ray/store/model is started. Immediate qualification table
  in baseline CONTAINED_LAUNCH separates evidence from all remaining gates.
- Both named test domains (58c4186b4f944717a485624194291efe and
  24d7e95dcfb349b1a7f43318831e6f66) have actual empty cgroup.procs, TasksCurrent0
  and high/max/oom/oom_kill0, then stopped. GPUs15MiB/0%; no TMUX model remains.
  Main runtime unchanged, D63 smoke evidence retained, no unnecessary GPU rerun.
- Exactly four baseline task files committed/pushed as
  ef49691e76c5fd1a48db11b158ab1c3177d93940; fresh origin/main full SHA matches.
  Named-stage, diff and added-text credential checks pass; pre-existing dirty
  replay/relayserve work excluded. Protected147 and source-plan SHA reverified.
  This main-repo status update records cross-repository progress; no formal
  baseline/M1/M2/ablation/sensitivity result or optimality claim. Goal active.

### D63 independent existing-weight numerical reference

- Previous D62 is progress: actual two-pool indices completed/validated/pushed.
  Startcef22afeb3bdf172b91b7959f7ff3fb5fa6c9e64; original user dirt retained.
  Full plan/status/AGENTS and run/backup/plotting skills read. Protected147 and
  source plan unchanged. GPUs15MiB/0%,MemAvailable109187448KiB,local free353278058496B.
- Return to independent7B correctness, not another index/allocator/prefix run.
  Read prior five-control probability observations, installed HF/PEFT source,
  exactvLLM0.30 sample-logprob tests and officialPEFT API. Existing old environment
  has PEFT0.18.1; newvLLM environment unchanged. Original nonzero7B weights and
  source prompt are reused. No new3B control or remote174 service is authorized.
- Existing preflight gains opt-in independent HF/PEFT first-position reference,
  exact adapter tensor checks and current backbone/input hashes. Positive AND
  wrong controls use the same predeclared official numeric criterion; closeness
  alone does not establish identity. Historical native lacks backboneSHA, and
  this limited reference does not certify Full or500ID behavior. Protocol is in
  NATIVE_ADAPTER_NUMERIC_CONTROL before launch; tests/actual observation follow.
-70 safety/reference/index/census/replay checks and288 basic smoke checks pass,
 0failures/errors/skips. Actual old-environment imports and393-token prompt
 reconstruction match native SHA without initializing CUDA. Three owned CPU
 scopes are empty with high/max/OOM0 and stopped. Protected147 and plan unchanged.
 Tested diagnostic source backup precedes the one independent model observation.
- Diagnosticd81f78cbdb200cdc7375cf6ad7daee8d32aaf10c pushed/fresh full remote SHA
  matches before launch. One guarded TMUX reference observation completes,
  measurement/launcher pass, service/watchdog0, actual GPU context/domain gone.
  HF loaded3x256tensors exactly; current backbone/input identities saved. All5
  matching comparisons close, but all20 wrong/correct matrix cells also close;
  adapter discrimination remains unestablished. HF nonzero full-vocab effects
  differ from base; zero/base and A/A are0 at this single output position.
  Immediate exact result table and curated20-row CSV/JSON delivered; no plot
  or full qualification/performance claim. No threshold change/second attempt.
- Watchdog57samples, peak1207103488B,minimum host available110319460352B,
  high/max/OOM0; actual auxiliary process list empty/events0 then stopped.
  FourGPUs15MiB/0%; no TMUX/model remains. RawSHA
  ad1ee5427b37f2c272d90633ffa6e571b554fbd5824102405ccad1094a79802d.
  Next representative profile/Full integration, not another same-prompt check.
  External choices/Full guard unchanged. Final protection/evidence backup follows.
- Complete20-row curated matrix independently reconciled to raw values; all
  raw/launch/watchdog/log and checker SHAs match. Protected147 and authoritative
  plan unchanged. Final MemAvailable109097772KiB,local free353354043392B.
  Evidence checkpointc497d2ee1969f3f13e6daa0a93f8df2da84e5a8d pushed; fresh full
  remote SHA matches. Four named data/docs files only, whitespace/secrets/name
  checks pass. Baseline16570c023a439c884624e7a5bdfa0d8577faf7a3 unchanged, fresh
  origin/main verified. This subsequent receipt adds no experiment or gate.

### D62 complete existing artifact identity for Full/profile input

- Previous D61 is progress: actual HOST-capacity propagation implemented/tested/
  pushed. Start61d0fde74806fa68d1ba459d2cc1ffaa6aaae2c8, baseline16570c unchanged.
  Full plan/status/AGENTS and execution/backup/plotting skills read. Protected147
  and plan unchanged; original dirt intact. No live TMUX/GPU work. Local available
  353312870400B,MemAvailable109203312KiB,GPUs15MiB/0% at initial check.
- Current-source audit: Full still lacks actual representative profiles and
  complete consumed artifact content indices. D26 serial GPU/HOST intervals
  cannot fill file/remote/concurrency classes or changed allocator identity.
  Read D18/D22/D36 and actual client/profile admission source; browsed original
  Python tarfile/vLLM0.30 loader. No fake profile or Full guard bypass.
- Fresh read-only remote check:148728238080B available,below150GiB;18080/18081
  have no listeners. Key access works. No remote service, cleanup or rule change.
  Both artifact/remote authority questions remain pending.
- Existing preflight gains index export bound to completed tensor audit. All
  current bytes verified, unchanged hardlinks read once; no model/tensor reload,
  pool/trace regeneration or weight/algorithm/profile changes. Four new tests.
  First66 safety checks have1import error: system Python lacks NumPy pulled in by
  the legacy storage package facade. Test now loads the actual stdlib HTTP module
  directly; endpoint keyword corrected. Second66 pass0failure/error/skip.
  Bounded smoke and actual per-pool indexing/validation receipts follow.
- 288 basic smoke checks pass. Initial indexer checkpoint
  860a066deaae9c12513661244eb84c65dd595889 pushed/full remote SHA verified.
  First3B index task terminates on existing external support symlinks; no output
  or artifact mutation. Exact first log retained. Read-only enumeration finds
  2500 such links (five per adapter). Actual remote server SHA a365072244512f4880432d7f4198cf3e45897ff19063a2b25eb141c8ca4e2a02
  already skips outside/dangling links. Indexer now explicitly mirrors that
  selection, records excluded link bytes, rejects unsupported internal expansion.
  Required weight/config/padding hashes still match old audit; no server change.
  Added real loopback archive-versus-index test;67 safety checks pass. Immediate
  failure/correction table delivered in ARTIFACT_CONTENT_AUDIT before next task.
- Local-pack3B index completes:500adapters,1500files,14911799296B,24exact classes;
  excludes2500support links/4570774000B. Scope empty/stopped,peak1074524160B,
  high23683,max/OOM0. SHA ba57a73918851669c28af0e383df9cacb38d2217fbbbaa9d8dd16662c00a6d16.
  Subsequent read-only174 metadata inspection finds NO symlinks:3B auxiliary
  files already materialized,4000files/19482573296B;7B5000files/12985984450B.
  Thus local-pack index is diagnostic only, NOT the remote3B contract. No remote
  bytes hashed/service started. New explicit allowed-local-support-root mode
  describes those remote relative files without copying or rewriting pools.
  Added shared-target mutation check; fresh tests and new indices follow.
  safety4 retained:58 checks execute and1 nonexistent census-module import fails;
  launcher corrected to the existing gpu_census module, no production failure.
- Final68 safety/index/census/replay checks pass. Materialized3B index completes:
  500IDs/4000files/19482573296B,24exact classes;1505inodes/14920940844B hashed.
  Existing actualHTTP client freezes manifest and validates all500 routing IDs.
  File SHA bd1826c58f00ea30dee1d3829dc11727a975127db701a1eee6c6c86b4a38f275;
  remote contents still unverified. Scope empty/stopped,peak1074520064B,
  high22769,max/OOM0. Both safety4/5 scopes empty/events0/stopped. Immediate
  evidence table saved before advancing to the7B existing-pool index.
- Final-source288 basic smoke checks pass; owned scope empty/events0/stopped.
  7B defaultskip scan completes but finds2external auxiliary links in finance_lora
  (797B), contrary to the earlier readable-file enumeration assumption. Output
  4998files/12985983653B,6classes;actualHTTP client validates500IDs but this does
  NOT match the remote5000file layout. Preserve as local-pack diagnostic only.
  Scope empty/stopped,peak1074266112B,high18811,max/OOM0. Immediate table recorded;
  next same tested explicit-support mode includes those two existing7B targets,
  new output/no overwrite. Baseline16570c full origin/main SHA reverified.
- Final7B materialized index completes:500IDs/5000files/12985984450B,6classes,
  2approved support targets, no exclusions. Both candidate indices revalidated
  through actualHTTP consumer/all500 localPEFT/routing IDs.7B file SHA
  e85cce3c3611da15530f9e663549231d3493282c85913344fe41c2a67044260c;
  peak1074266112B,high17056,max/OOM0;actual process list empty then scope stopped.
  Those2remote7B auxiliary files additionally have matching read-onlySSH SHAs,
  not complete remote qualification. Candidate-vs-diagnostic paths and all
  hashes documented in inputs/README; immediate final two-model table delivered.
- All D62 owned scopes empty/stopped; no live model/TMUX/service/test. Final
  GPUs15MiB/0%,MemAvailable109255308KiB,available353285074944B. Protected147 and
  authoritative plan SHA unchanged. No pool/trace/weight copied or regenerated,
  no semantic/remote/Full/performance gate waived. Tested named-file backup
  follows; baseline unchanged16570c verified against fresh origin/main.
- Tested code/indices/documents checkpoint5c89e3824c7f2055af8e59b9013567876e95f22b
  pushed; fresh faaslora_origin/retry14_continuous_queue_v2 full SHA matches.
  Exactly9 named files; whitespace, protected-name and added-text credential
  checks pass. Current git status contains only original user dirt; no D62
  scope or TMUX session remains. This subsequent documentation-only receipt
  adds no measurements and does not qualify remote/Full/semantic correctness.

### D61 observed HOST return and deferred preparation

- Start e7f9a4d711863495fb1d44aa9700278a1a7b42d6; baseline16570c unchanged.
  Full plan/status/AGENTS and execution/backup/plotting skills read, including
  after recovery. Protected147 and plan unchanged; original dirt preserved.
  Reviewed actual queue/worker/allocator path, D54/D58/D60, IEEE budget text and
  matching official PyTorch2.13/vLLM0.30 sources. No agent/model/GPU/remote run.
- Existing deferred work retains its actual byte-refusal snapshot. On existing
  frozen control samples, only owners with live HOST byte-pressure waiters are
  observed, once per owner. A or A-X improvement wakes existing work; original
  executor rechecks capacity/references/E(t). Wrong owner/clock/unknown bytes
  rejected; post-RPC attempt recheck excludes withdrawn/obsolete refusals.
  No retry timer, dummy allocation, future-free credit, budget increase, early
  eviction, paper equation/profile/model-config change or Full guard removal.
- Five added checks. Initial61 targeted checks, final943 functional,16 installed
  native CPU and62 system-Python safety/census/replay checks pass,0fail/error/skip.
  Native CPU checks assert CUDA uninitialized; counts overlap. Correctness table
  in P2 D61 explicitly distinguishes controlled CPU observations from actual D60
  allocator measurement. Logs:results/ieee_tc/p2_backend_qualification/d61_20260927.
- Next is representative actual profiles and Full/multi-plan/lifecycle, not
  another allocator-only or old-prefix check. Full numerical/500-pool and remote
  authority gates remain. No formal baseline/main/ablation/sensitivity started.
  Final owned-scope cleanup, protection and named-file backup follows.
- All four D61 scopes have actual cgroup.procs empty,TasksCurrent0 and
  high/max/oom/oom_kill0, then stopped. No model/test remains running. Final
  GPUs15MiB/0%,MemAvailable108888372KiB,available disk353312206848B. Protected147
  and authoritative plan reverified unchanged. Tested named-file backup follows;
  baseline has no task edits. These checks add no performance qualification.
- Tested implementation34ac569cf8d3ee38013efe499193a8febbd3d6fd pushed; fresh
  faaslora_origin/retry14_continuous_queue_v2 full SHA matches. Exactly six named
  files, whitespace/protected-name/added-text secrets checks pass. Baseline
  16570c023a439c884624e7a5bdfa0d8577faf7a3 unchanged and fresh origin/main SHA
  matches. This subsequent receipt is documentation only, not another run.

### D60 official background event handling

- Start e2d4c886bd6d4588d07e2b4f6d0db3195c5dd8be; baseline16570c unchanged.
  Full plan/status/AGENTS and execution/backup/plotting skills read. Protected147
  and plan SHA unchanged; no model/TMUX active, GPUs15MiB/0%. Read actual D59
  evidence and installed allocator/native setter; browsed matching official
  sources. No production policy or budget change. Existing diagnostic gains a
  strictly checked background-only flag; second/final local hypothesis comparison
  preregistered in P2. No new artifacts/traces or formal performance run.
- 62 safety and288 smoke checks pass. Diagnostic checkpoint
  e9ebcbe7b1415e2dea565ff1eed071dadde04e78 pushed with full remote verification.
  Actual one-process background_attempt1 completes all6classes/2arms/96states.
  All native-copy post-delete and second-fence allocated bytes0 before next load;
  all slot content matches. All stream queries completed, no in-flight deletion
  guarantee. Peak727416832B over42 samples, high/max/OOM0, service/watchdog0.
  Actual native context/domain released; aux950712bccfc44b3f8574df3738f59330
  cgroup.procs empty then stopped. Two CPU test scopes TasksCurrent0/events0,
  stopped. Immediate P2 table and complete CSV/JSON delivered before next task.
  Accept official background handling as the Full integration candidate, not a
  performance winner. No further allocator microloop; production identity and
  actual queue progress/profile/Full remain next.
- Existing runner/worker/native readback and fixed workspace gate now accept
  distinct opt-in uncached_background_v1. Default/model configs unchanged; old
  profiles and inherited allocator conflicts rejected. Actual occupancy remains
  authoritative; no future-return credit, B/C increase or Full guard removal.
  Initial215 targeted checks, final938 functional,22 installed-native CPU and
  62 system-Python safety checks pass with no failures/errors/skips. CUDA remains
  uninitialized in CPU checks. Counts overlap; not performance repetitions.
- All96 curated rows independently reconciled to raw counters; all12 cross-run
  GPU slot SHAs match D59. Raw/launch/watchdog/comparison/source SHAs verified.
  Four final integration/test scopes actual cgroup.procs empty, TasksCurrent0,
  high/max/OOM0, stopped. No test/model/TMUX remains. Protected147 and plan SHA
  unchanged; GPUs15MiB/0%, MemAvailable109306740KiB, available353322582016B.
  Next: integrate observed HOST capacity changes with actual deferred-work
  progress and representative measured profiles/Full, not another isolated
  allocator or old-prefix check. Numerical/500pool/remote authority gates remain;
  formal baseline/main/ablation/sensitivity experiments have not started.
  Named-file evidence/integration backup completed below; goal remains active.
- Measured evidence and explicit candidate committed as
  32972897965e9ed3728dae6bb1917581d068f4e2; pushed and fresh full remote SHA
  matches. Exactly12 named task files, diff/secrets/protected-name checks pass.
  Baseline16570c023a439c884624e7a5bdfa0d8577faf7a3 unchanged and fresh origin/main
  SHA matches. Protected147 and source plan reverified after backup. This final
  documentation-only receipt changes no runtime, measurement or qualification.

### D59 real-copy HOST lifetime

- Previous goal turn D58 is progress: implemented/tested/backed up fixed-budget
  workspace partition. Current startcb2b1e3f08265837016c5556732735f6399c3e1b;
  baseline16570c unchanged. Full plan/status/AGENTS and execution/backup skills
  reread. Protected147/plan SHA unchanged, original user dirt retained.
- Inspected actual native setters/allocator, D29/D56/D58 evidence and IEEE
  staging/replacement text; browsed officialvLLM0.30 Base/Merged setter sources
  and PyTorch2.13 CachingHostAllocator. Hypothesis: real H2D stream references
  retain blocks even after a fence, unlike D56 CPU-only deletion. No guessed
  release credit or production workaround. Extended existing diagnostic only.
- New measurement protocol is fixed before launch in P2 document.60 system-
  Python safety/census/replay checks pass; smoke/model observation/cleanup and
  backup receipts follow. Formal baselines/M1/M2/ablations remain not started.
- 288 smoke checks pass. Diagnostic source d9aba9e9a63e34e1d6e9fff5a4217d68e7cf404e
  pushed and fresh full remote SHA verified before native launch. Actual TMUX
  tc_d59_host_copy executed one fresh guarded process; lifetime_attempt1 and
  launch receipt complete/pass, service/watchdog0, actual context/group released.
  No backbone/inference or performance comparison. Same six existing classes,
  12 copy arms,96 states; all original SHA and complete GPU slot contents match.
- Native post-fence deletion retains R (3B9,175,040/18,350,080;7B16,777,216/
  33,554,432B). A second fence leaves it retained; the next genuine load processes
  224/256 frees. Pitched post-fence deletion returns all blocks. Every setter-
  return query is already complete, so do not claim in-flight source deletion
  safety or latency. Next no-copy object removal leaves all arms at0.
- 43 resource samples, service peak723648512B, high/max/oom/oom_kill0. Aux UUID
  7adc4c78c67e4c8396faeb6efa59bcf1 and two owned test scopes empty, events0, stopped.
  GPU15MiB/0%. Immediate table and complete curated CSV/JSON plus raw/launch/
  watchdog hashes written before next experiment. Production config unchanged.
- Final all96 curated rows independently reconciled to original native counters;
  raw/launch/watchdog and measurement source SHA match. Protected147 and plan
  unchanged. MemAvailable109101688KiB, available disk353353383936B, GPUs15MiB/0%.
  No test/model/TMUX remains running. Actual pending-event retention is now
  evidence, not speculation; do not repeat this first observation. Next focused
  alternative is official background event processing, then return to actual
  Full/profile qualification. No production candidate selected yet. Result
  backup follows; baseline/performance/numerical/remote gates remain open.
- Curated96-row evidence and explanatory table committed as
  4922db1a0b2dc6ce4e827ca91b2ae02cfe2050d8, pushed and full remote SHA verified.
  Exactly4 named result/document files staged; diff/secret/name checks pass.
  Baseline16570c023a439c884624e7a5bdfa0d8577faf7a3 unchanged, fresh origin/main
  full SHA verified. This final receipt changes no runtime or measurement.

### D58 native HOST workspace partition

- Start8856bdfff20dc0b9be8c5c888fdc669afadb54e0, baseline16570c unchanged.
  Full plan/status/AGENTS and run-experiment/github-sync reread after recovery.
  Protected147/plan SHA unchanged; all pre-existing user dirt preserved. No
  agents, model/GPU, remote174 or formal performance run in this continuation.
- Read D54/D56/D57 source/history, IEEE staging/budget text, actual installed
  vLLM0.30 loader and dense packing, official worker_manager/PyTorch2.13 allocator
  and ELORA sources. Derived conservative resident/growth/demand protection from
  actual serialized native loading, not a new paper objective or manual retry.
  New optional contract uses actual native C, actual baseline/current occupancy,
  and existing full-pool R/W bounds. It rejects undersized allowances and changed
  layouts; no B/C increase, hidden LRU, predicted victim credit or flush.
- Runtime worker separates actual exclusively registered storage from staged
  aliases, applies protected-space check only to new proactive loads, and keeps
  actual total-byte checks for demand/reuse. Controller forwards/validates one
  immutable contract per physical owner. Acknowledged plan close wakes waiters.
  Opt-in/default model config and Full startup guard remain unchanged.
- Reused six-class D56 observations and complete audit to derive 3B/7B bounds,
  no weights loaded/generated. Curated JSON and immediate correctness table in
  P2 document; source SHA and independent offline recomputation match. These
  tensor upper bounds are not measured peaks or whole-service RAM guarantees.
- Initial281 and focused22 checks pass. Final-source935 functional,59 system-
  Python safety/census/replay and30 installedvLLM0.30/torch2.13 CPU checks pass,
  zero failures/errors/skips and CUDA uninitialized. Logs under
  results/ieee_tc/p2_backend_qualification/d58_20260927. Counts overlap.
- Actual in-flight pinned return, full multi-plan behavior, measured profiles,
  actual Full/A4, numerical500-pool and realremote gates remain open. No new
  baseline/M1/M2/ablation/sensitivity results or optimality claims. Next proceeds
  to actual copy/Full evidence instead of repeating completed CPU measurements.
  Final owned-scope cleanup/protection/backup receipt follows; goal stays active.
- All5 D58 owned scopes verified actual cgroup.procs empty, TasksCurrent0 and
  high/max/oom/oom_kill0, then stopped. Final GPUs15MiB/0%, MemAvailable108651780KiB,
  available disk353387175936B. Protected147 and source plan SHA unchanged;
  no model/test remains running. Named-file tested milestone backup follows.
- Tested code/data checkpoint e2ba9f6f8a76d0e0ce02e87f2f097161dd64bd83 pushed to
  faaslora_origin/retry14_continuous_queue_v2 and fresh full remote SHA matches.
  Exactly11 named task files, diff/protected-name/added-text secret checks pass.
  Baseline has no task edits at16570c023a439c884624e7a5bdfa0d8577faf7a3; fresh
  origin/main full SHA also matches. This additional
  documentation-only receipt adds no measurements or qualification claims.

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
