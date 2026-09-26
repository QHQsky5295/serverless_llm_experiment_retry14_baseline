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
| P1 IEEE semantic alignment | Actual routing/source admission connected; 7B native source boundary measured; Full physical integration open | P1_FORMULA_IMPLEMENTATION.md D1–D26. 32-request actual GPU/HOST qualification through D25 helpers; no legacy simulation fallback or production profile fabricated. Native multi-path content identity/capacity waiting, all-tier physical admission/lifecycle and representative Full qualification remain open; no Full performance qualification |
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

LATEST COMPLETED MODEL RUN: `llama2_7b_physical_lifecycle_attempt3`, finished and
cleaned. Four original7B requests/551 tokens; normal parent exit0, actual native
worker exit event, UUID census and physical lease return.60.738 GPU-s including
7.642s after the last token. Attempt1 censored; attempt2 forced-exit with48.769s
tail; both preserved. Same request/adapter/prompt/output hashes across all three.
Do not rerun this prefix. Curated CSV/JSON + status table delivered. No formal
performance or complete Full lifecycle claim. D26 source32 remains completed;
its one historical SHA difference and independent numerical gate remain open.

0. P2 installation finished; do NOT reinstall. All current runs finished/cleaned. Native adapter reference, full 3B/7B content scan and 7B five-arm native numeric diagnostic COMPLETE; do not repeat same-prompt/zero controls. NATIVE_ADAPTER_NUMERIC_CONTROL.md: all five 217-token outputs identical; A/Z probability max difference .02608, A/A also .006367. Descriptive evidence, not independent numeric/full semantic qualification. Return to Full source/cost and owner integration next. Current 3B pool has 500 all-zero LoRAs (2 SHAs); 7B has 498 zero plus finance/medical nonzero (4 SHAs). Plan forbids new weights; user confirmation requested before adding a few trained nonzero 3B correctness fixtures, no answer yet. Keep pools/traces/history intact. Targeted 2,521 existing configs yielded no 3B candidate, not proof of server-wide absence. Independent numerical verification remains an open gate, not erased by this diagnostic. Review official 0.30 warning on old 3B chunked_prefill=false before freezing. Reuse environment/cache, KEEP_DEDICATED_WORKER_LOGS for future proxy runs.
1. D26 collected actual native D25 source-admission/token-boundary evidence for 7B through the existing backend-model-check. Explicit profile-only collection requires no fictitious initial estimate; production Router still requires measured profiles. Next complete physical admission/lifecycle and representative class initialization/Full integration, not another repetition of this serial prefix. Known native capacity waiting and content-bound HOST/NVMe path migration remain explicit gaps; never remove identity checks or introduce sleep/fallback. Do not repeat isolated source-composition/same-prompt controls. This is not full global atomic admission or Full performance qualification.
2. D27 now qualifies the7B dedicated runtime's physical allocation and normal exit. Do not re-run its four-request prefix. Connect owner coverage and aggregation for the actual Full deployment (shared/direct/multi-runtime paths are not qualified here), while completing physical tier admission and representative measured profiles. Qualify Serverless native checkpoint path before its original/repaired model pair.
3. Continue remote setup after its disk gate; no heavy GPU run
   until actual process containment and watchdog gates are satisfied.

## Latest verified backups and evidence index

- Main tested implementation: `0e2adf9dc938b0a9e2c2aa5447585685d60d5a3c`, pushed to
  `faaslora_origin/retry14_continuous_queue_v2`; remote SHA verified.
  This backup receipt is a subsequent documentation-only commit.
- Baselines: `16570c023a439c884624e7a5bdfa0d8577faf7a3`, pushed to
  `origin/main`; remote SHA verified at its checkpoint. No baseline edits in D26.
- All earlier checkpoints, failures, qualification attempts and exact backups
  remain verbatim in [EXECUTION_HISTORY_THROUGH_D26.md](EXECUTION_HISTORY_THROUGH_D26.md).
  Archive SHA256: `4e91643d959967f1f0dc8642fe969d35380f2156cf8c820964019539d3b415cb`.
  This is archival organization, not new experimental evidence.
- Read the relevant historical section before each optimization; do not treat
  old chronological “next” items as current instructions.
- P1 formulas / native sources / ownership: `P1_FORMULA_IMPLEMENTATION.md` D1–D27.
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
   all-tier physical admission/E(t), capacity-conflict waiting, content-bound
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
