# PrimeLoRA IEEE TC baseline execution

Read `/home/qhq/storage_audit_20260915/PrimeLoRA-PLAN.md` in full and
`/home/qhq/serverless_llm_experiment_retry14_baseline/docs/ieee_tc/EXECUTION_STATUS.md`
before work, after context compaction, and before every experiment.

Preserve existing dirty replay_openai_trace.py and relayserve work. Stage only
explicitly created/changed task files. No dataset/pool regeneration or old-result
overwrite. Reuse prior valid logs per R0–R3. One heavy experiment at a time;
actual worker memory/CPU restrictions and safe cleanup are mandatory.

Baseline priority: Serverless audit/authorized minimal polling fix, vLLM,
S-LoRA, dLoRA 3B, Loquetier, HydraServe. Preserve official core algorithms;
record compatibility patches and native versions. Display Serverless without
"-new", but retain official commit and patch identities in provenance.

Each run ends with a figure/table and interpretation before the next run.
Report progress from the perspective of paper evidence, not implementation jargon.
Commit/push tested milestones to this repository's origin/main without force,
secrets, raw large data or unrelated changes. The main repo tracks cross-project
checkpoint hashes. Do not treat formal failures as disposable retry attempts.
