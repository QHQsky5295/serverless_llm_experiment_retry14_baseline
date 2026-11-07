# PrimeLoRA IEEE TC execution contract

Current handoff (2026-11-07): D1116 offline W1 budget diagnosis COMPLETE.
Read `docs/ieee_tc/D1116_W1_BUDGET_DIAGNOSIS_20261107.md` then
`docs/ieee_tc/D1117_7B_BUDGET_CAP3_REGISTRATION_20261107.md`.
No GPU experiment is running. D1115 arrival held four GPUs for 1061.571773 s,
mean 3.447244 physical GPUs; non-arrival cost only 64.779336 GPU-s. Twenty
closed leases, 19 successful scale-ups/17 releases; 9 releases followed by
another activation within 38 s. Request interval complement is NOT idle time
or guaranteed savings. Existing source inputs/results remain unchanged.
Next is registered D1117 7B G2 cap3 W1, NOT another audit/replay of D1115.
Config `20261107_d1117_7b_budget_cap3.yaml` SHA
34562e22c517fc62033da1c3437ef95df616fd54c8ae0dcb20a3e59db289eb53.
Only max ready+pending 4->3 plus unique output/cache roots; all 4 physical
devices visible, same D1100 serving code/controller/equations. Runner now
accepts an optional sixth explicit config path, old 0/5-argument mode unchanged.
53 targeted tests and shell syntax pass. Before actual launch, backup and
normal resource/remote checks; launch once in tmux per D1117 instructions.
If W1 budget+joint timing feasible, use the SAME configuration for W2; do not
tune per scenario, replace resource_opt, conflate cap3 with physical budget,
change thresholds, suppress failed history, or repeat rejected guard/no-release
directions. D619 W1 success and D621 W2 failure both retained as context.
All M1/M2/A1-A5/S1-S13/docs/figures remain active; D1117 not yet started.

Completed handoff (2026-11-07): D1115 current-version 7B W1 COMPLETE/released.
Read `docs/ieee_tc/D1115_W1_RESULT_AND_THREE_SCENARIO_20261107.md` first.
4000 native-contract requests, joint SLO 98.55%, 7603.874678 GPU-s,
P95 TTFT 2331.329520 ms. No repeat; tmux ended and owned remote is inactive.
Raw: `/home/qhq/serverless_llm_experiment/results/ieee_tc/p2_backend_qualification/d1115_20261107/7b_same_version_w1`.
D1114 W0 / D1115 W1 / D1100 W2 now form the same-config 7B three-scenario
comparison. All three meet the timing SLO, but W1 exceeds frozen G2 budget
6756.571050 GPU-s by 847.303628 (12.5404% of budget; 11.1431% reduction needed).
Next: investigate this specific 7B G2 gap using existing ownership/queue/control
logs and prior budget candidates/rejected directions, then literature and one
falsifiable hypothesis if a new budget working point is necessary. Do not repeat
W0/W1/W2, restart baseline reproduction, switch back to 3B, change thresholds,
or declare all objectives complete. Existing A1-A5/S1-S13/docs/figures remain open.
Five W1 baseline audits reused; all inputs matched, 218 output digests differ
from D536 without independent numerical attribution. All differences retained.
50 source hashes and 144 protected publication files verified; user manifest unchanged.

Completed handoff (2026-11-07): D1114 7B W0 is COMPLETE and released.
Read `docs/ieee_tc/D1114_W0_RESULT_20261107.md` first. Do not rerun it.
4000 native-contract requests, 3954 joint SLO (98.85%), 10541.548234 GPU-s,
P95 TTFT 1868.328236 ms. Matched existing W0 baselines; 164 output digests
differ from vLLM D423, retained without an assumed numerical explanation.
144 protected publication files and user manifest unchanged. No serving changes.
Its result backup f8a52faa was pushed and verified before D1115 started.
W2 D1100 already exists; no W0/W2 rerun or new baseline reproduction.
D989 is older-source W0; full main/ablation/sensitivity scope remains open.

Current handoff (2026-11-07): read
`docs/ieee_tc/D1113_BASELINE_CONTRACT_RECONCILIATION_20261107.md`, then
`docs/ieee_tc/D1112_W2_OUTCOME_AND_HANDOFF_20261107.md`, before older headings.
D1103/D1106/D1108 are R2 diagnostics, NOT common-contract main comparisons.
Reuse already completed D555/D557/D591; D1113 matched all 4,000 inputs/native
contracts to D1100 and verified the common notice/resources. D593/D810 already
contain both-model baseline matrices; do not restart baseline reproduction.
Next missing current-configuration work is 7B D1100 W0, then W1, not another W2.
D1084 and D1112 both completed same-version 3B W2; do not rerun either or
select by victory. D1112 is below the frozen joint SLO and remains disclosed.

Latest user clarification (2026-10-25, concurrent main-line progress): storage
auditing/reclamation must progress alongside the experiment main line, not become
an indefinite replacement for it. Perform proven-safe compression/cleanup between
GPU measurements; use existing logs/code for the current 3B optimization question
while CPU/I/O maintenance runs. Never overlap heavy maintenance with a measured
GPU replay. Unreadable/shared or unresolved other-project data remain protected;
do not wait for a perfect whole-disk inventory before making in-scope progress.

Latest user update (2026-10-25): explain whole-disk storage attribution, not only
this project's usage. Audit other projects and shared/system permission gaps;
delete only proven obsolete/rebuildable contents, or losslessly compress closed
evidence with restoration/source records. Size, age, failed-run labels and cache
names alone are insufficient. Do not present a timed-out/partial scan as a full
inventory, modify shared services, or change filesystem reserves to mask growth.
See `docs/ieee_tc/STORAGE_PRIORITY_20261025.md`. Storage first, then resume the
same experiment main line without removing outstanding scope.

Latest user clarification (2026-10-22): explicitly check prior rejected and
ineffective optimizations before every new candidate. Do not repeat a closed
direction unless a concrete new observation changes its premise; document that
new evidence. Preserve all failed measurements, keep one falsifiable candidate,
and return to the complete experiment main line after its bounded evaluation.

Latest user update (2026-10-22): avoid excessive defensive decisions and useless
extra gates. Reuse valid completed checks; choose tests proportional to the changed
path. Do not repeat broad qualifications or add diagnostic replays without a
specific unresolved question. Preserve the established memory/disk containment,
correctness and measurement contract. If a native mixed-batch profile collection
already validates the changed backend path, do not add a redundant short Full
replay before the required complete workload. Record limitations, not new hurdles.

Before any task, after context compaction, and before every experiment, read
`/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md` in full and
`docs/ieee_tc/EXECUTION_STATUS.md`. The source plan is authoritative; its
historical "plan mode / no execution this turn" sentences describe approval
time, not the user's subsequent explicit execution authorization.

Latest user update (2026-10-14): read
`docs/ieee_tc/BASELINE_ADAPTATION_STOP_RULE_20261016.md` first for the newer
bounded-adaptation rule. Once necessary correctness, measurement and machine
adaptation are verified and observed performance has mechanism/workload evidence,
stop baseline performance optimization. Low SLO alone is not a reason for another
search. Complete the frozen matrix, then focus on PrimeLoRA; retain all limitations
and unresolved concrete defects. This does not remove any planned experiment.

For the 2026-10-14 baseline scope also read
`docs/ieee_tc/BASELINE_SCOPE_20261014.md` and
`docs/ieee_tc/SLO_BUDGET_RATIONALE_20261014.md`.
External baseline tables contain published-paper systems only. ElasticLocality
is NOT an external baseline; preserve historical data and use it only as a
PrimeLoRA composite ablation variant. No relabeling as another paper system,
deleting evidence, or claiming performance improved because the table changed.
Core-mechanism preservation/evidence matters, not an exact hardware replica.
95%/75% are frozen project points, not universal industry norms; keep the
approved threshold/budget sensitivity without changing online policy or history.

Latest user order (2026-10-12): read
`docs/ieee_tc/STORAGE_PRIORITY_20261013.md` FIRST for the newer storage-first
instruction: audit and safely reclaim proven obsolete/rebuildable data or
losslessly compress closed evidence before another build/new experiment.
Retain unique assets, failed evidence, user dirt and unrelated projects.
After this maintenance, resume the baseline order below; no experiment scope
is removed. Read
`docs/ieee_tc/BASELINE_COMPLETION_ORDER_20261012.md`. Finish/audit/save the already
running D476 once, then complete HydraServe necessary isolated adaptation/native
qualification and obtain actual results BEFORE another PrimeLoRA optimization
candidate. Compare the full available baseline set, then optimize PrimeLoRA.
This supersedes older priorities deferring HydraServe, not metrics or full scope.

Latest user goal update (2026-10-11): also read
`docs/ieee_tc/BASELINE_EXECUTION_RULES_20261011.md`. Except necessary compatibility,
correctness/measurement and safety adaptations, do not optimize non-PrimeLoRA
systems or search their performance configurations. Retain their observed results;
focus optimization on PrimeLoRA. This supersedes older baseline-tuning permission,
not the full matrix, frozen metric rules or historical execution identities.

- Work on `retry14_continuous_queue_v2`; preserve pre-existing dirty files.
- Never stage `configs/generated/lora_manifest_1000.json` or unrelated user work.
- Use existing runners, environments, artifacts and traces. Do not regenerate
  adapter pools, duplicate full datasets, overwrite old results, or run 13B.
- G1: correct full workload + common joint SLO, then minimum lifecycle GPU-s.
  G2: common budget + SLO, then tail TTFT. CE is supplementary.
- Before every system comparison, configuration selection or comparative figure,
  read docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md in full. Record its SHA and the
  separately frozen warm/reference manifests. Development 5000ms/live CE is not
  final SLO qualification. Do not alter V1 in place after its freeze; revisions
  require a new protocol identity and an explicit impact assessment.
- Every optimization: historical logs/code + current primary-source literature
  and code + falsifiable bottleneck hypothesis + validation + full replay.
- Latest user direction (2026-10-10): single performance run per frozen key,
  no repeated-run mean/CI; preserve all old attempts and development identity.
  Read METRIC_PROTOCOL_SINGLE_RUN_V2.md alongside frozen V1. First close a
  reusable 7B PrimeLoRA point and compare 7B baselines (ServerlessLLM audit first),
  then 3B. W0 success alone does not close W1/W2 or G2.
- Preserve IEEE's nine equations and semantics. Prefer measured, causal online
  state; model-specific frozen configuration is allowed, test-point tuning is not.
- First validate resource containment and protected results. No GPU performance
  campaign until actual workers, playback separation, watchdog and cleanup pass.
- ServerlessLLM audit comes first among baseline tasks. Latest user naming rule
  (2026-10-10): always write PrimeLoRA and ServerlessLLM in full in reports,
  new documentation and figures. Preserve historical raw identities/paths.
  Authorized polling repair must preserve RR/scaling/loading semantics.
- One heavy run at a time. No blind OOM retries. Do not stop unrelated processes.
- Latest user direction (2026-10-06) supersedes plan section 11's immediate
  plotting cadence: before PrimeLoRA meets the performance targets, do not generate
  figures. Each run: cleanup -> validate/save data -> concise interpretation ->
  next main-line task. Do not spend optimization time on presentation work.
  Later publication figures still follow section 11's typography/layout rules.
- Report periodically in Chinese as paper evidence: completed experiments,
  current question and reason for remaining here, outstanding experiments.
  Do not disguise engineering checks as completed performance experiments.
- Commit/push tested checkpoints to faaslora_origin/retry14_continuous_queue_v2;
  baseline work goes to its own repo. No force push, credentials, or raw large data.
- Record evidence and next action in EXECUTION_STATUS before yielding. Keep the
  full goal active until all specified experiments and deliverables are verified.
