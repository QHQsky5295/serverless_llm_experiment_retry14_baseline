# PrimeLoRA IEEE TC execution contract

Before any task, after context compaction, and before every experiment, read
`/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md` in full and
`docs/ieee_tc/EXECUTION_STATUS.md`. The source plan is authoritative; its
historical "plan mode / no execution this turn" sentences describe approval
time, not the user's subsequent explicit execution authorization.

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
- Preserve IEEE's nine equations and semantics. Prefer measured, causal online
  state; model-specific frozen configuration is allowed, test-point tuning is not.
- First validate resource containment and protected results. No GPU performance
  campaign until actual workers, playback separation, watchdog and cleanup pass.
- Serverless audit comes first among baseline tasks. Display name is Serverless.
  Authorized polling repair must preserve RR/scaling/loading semantics.
- One heavy run at a time. No blind OOM retries. Do not stop unrelated processes.
- Each run: cleanup -> validation -> figure/table -> interpretation -> next task.
  Follow plan section 11, Times New Roman, IEEE single column, no overlap.
- Report periodically in Chinese as paper evidence: completed experiments,
  current question and reason for remaining here, outstanding experiments.
  Do not disguise engineering checks as completed performance experiments.
- Commit/push tested checkpoints to faaslora_origin/retry14_continuous_queue_v2;
  baseline work goes to its own repo. No force push, credentials, or raw large data.
- Record evidence and next action in EXECUTION_STATUS before yielding. Keep the
  full goal active until all specified experiments and deliverables are verified.
