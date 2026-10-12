# PrimeLoRA IEEE TC execution contract

Before any task, after context compaction, and before every experiment, read
`/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md` in full and
`docs/ieee_tc/EXECUTION_STATUS.md`. The source plan is authoritative; its
historical "plan mode / no execution this turn" sentences describe approval
time, not the user's subsequent explicit execution authorization.

Latest user order (2026-10-12): read
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
