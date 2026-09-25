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
| Resource containment / safety | Basic primitives passed; production gates pending | Tiny scope inheritance, OOM and cleanup pass; failed soft-throttled attempt retained. CPU proof is affinity, not delegated cpuset |
| Protected historical artifacts | Sealed and verified | `paper_results/ieee_tc/safety/20260925_execution_start_protected.json`; old results and selected user modifications unchanged |
| Remote authentication / management | Key login verified; service qualification pending | Strict host checking, dedicated restricted key; remote disk 138.9 GiB below 150 GiB floor, user decision pending. See REMOTE_ACCESS.md |
| P0 main-table / Full provenance | 7B source-pair audit complete | Same trace/subset SHA, different execution; 202 scalar fields preserved. P0_FULL_PROVENANCE.md; no performance rerun needed for this finding |
| Serverless wait audit | Historical audit + no-GPU control-path tests complete | Clean 7B/3B logs reused; six real-method AST tests pass; incremental ready-before-wait patch preserved. Model pair pending |
| P1 IEEE semantic alignment | Pending | Audit existing V2 implementation rather than rebuilding |
| P2 backend qualification | Pending | vLLM candidate must be checked locally |
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

1. Back up tested Serverless audit/repair, P0 source table and inspected diagnostic figures.
2. Integrate effective service-worker and external-watchdog checks into existing launchers.
3. Continue remote setup after its disk gate and P0/P1 read-only audits; no heavy GPU run
   until actual process containment and watchdog gates are satisfied.

## Verified backups

- Main `a022340f014ab2a115bbab79841638ec6c15d1ae` pushed to
  `faaslora_origin/retry14_continuous_queue_v2`; remote SHA checked.
- Baseline `2353e7e0c7af7cfd0bfb43a4913c9178da391e34` pushed to `origin/main`;
  remote SHA checked. No pre-existing user changes staged.
- Baseline audit/repair checkpoint `f40eff06ad1a31d17fe05b12405d74b8a6a7bf70`
  pushed to `origin/main`; main audit bundle corresponds to this implementation.

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
- No formal GPU run has started. No performance claim follows from these checks.
