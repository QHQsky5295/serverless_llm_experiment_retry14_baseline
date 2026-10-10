# PrimeLoRA IEEE TC baseline execution

Read `/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md` in full and
`/home/qhq/serverless_llm_experiment_retry14_baseline/docs/ieee_tc/EXECUTION_STATUS.md`
before work, after context compaction, and before every experiment.

Preserve existing dirty replay_openai_trace.py and relayserve work. Stage only
explicitly created/changed task files. No dataset/pool regeneration or old-result
overwrite. Reuse prior valid logs per R0–R3. One heavy experiment at a time;
actual worker memory/CPU restrictions and safe cleanup are mandatory.

Baseline priority: ServerlessLLM audit/authorized minimal polling fix, vLLM,
S-LoRA, dLoRA 3B, Loquetier, HydraServe. Preserve official core algorithms;
record compatibility patches and native versions. Latest user direction
(2026-10-10): use full names PrimeLoRA and ServerlessLLM in every report,
new document and figure. Preserve internal IDs, source paths and historical
raw names; retain official commit and patch identities in provenance.

Latest user direction (2026-10-10): read the main repository's
docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md alongside frozen V1. One
performance run per frozen execution key; no repeated-run means or CIs.
Preserve all attempts and do not select the best repeat. Resume 7B baselines
after a reusable 7B PrimeLoRA point, then 3B. The 2026-10-06 direction defers
figures until PrimeLoRA is near target; save data/status and interpretation now.
Report progress from the perspective of paper evidence, not implementation jargon.
Commit/push tested milestones to this repository's origin/main without force,
secrets, raw large data or unrelated changes. The main repo tracks cross-project
checkpoint hashes. Do not treat formal failures as disposable retry attempts.
