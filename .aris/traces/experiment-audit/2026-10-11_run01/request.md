# Reviewer invocation

Tool: collaboration.spawn_agent. Task: d420_vllm_evidence_review.
fork_turns: none. Model: gpt-5.6-sol. reasoning_effort: ultra.

Read-only experiment integrity review. Do not edit files, launch experiments, access remote machines, or spawn further agents. Review objectively: test for evidence errors, do not presume either correctness or misconduct. Scope is existing vLLM 7B evidence and justified reuse boundary, not a new performance experiment. Follow the experiment-audit A–F checklist: (A) reference/target provenance and labeling, (B) metric denominators/self-normalization, (C) existence and numerical agreement of result claims, (D) actually executed metric paths, (E) scope including actual distinct adapter IDs versus nominal pool and short qualification versus complete replay, (F) evaluation type. Report PASS/WARN/FAIL with exact file:line evidence, concrete claim impact and bounded next checks. Mark semantic review same-family/provisional; deterministic checks separately. Required skill referenced shared-references are absent locally; disclose this limitation. Read these files directly (and directly linked source evidence as needed), not executor summaries:

- /home/qhq/serverless_llm_experiment_retry14_baseline/docs/ieee_tc/METRIC_PROTOCOL_FROZEN_V1.md
- /home/qhq/serverless_llm_experiment_retry14_baseline/docs/ieee_tc/METRIC_PROTOCOL_SINGLE_RUN_V2.md
- /home/qhq/serverless_llm_experiment_retry14_baseline/docs/ieee_tc/D419_VLLM_7B_ENVIRONMENT_PATH.md
- /home/qhq/serverless_llm_experiment_retry14_baseline/paper_results/ieee_tc/baseline_audit/d419_7b_native_path/audit.json
- /home/qhq/serverless_llm_experiment_retry14_baseline/paper_results/ieee_tc/baseline_audit/d419_7b_native_path/evidence_manifest.json
- /home/qhq/serverless_llm_experiment_retry14_baseline/scripts/ieee_tc/audit_d402_serverlessllm_path.py
- /home/qhq/serverless_llm_experiment_retry14_baseline/docs/ieee_tc/D412_VLLM_MEASUREMENT_REUSE_AUDIT.md
- /home/qhq/serverless_llm_experiment_retry14_baseline/docs/ieee_tc/D397_RESIDENT_AUDIT_CORRECTION.md
- /home/qhq/serverless_llm_experiment_retry14_baseline/docs/ieee_tc/D213_VLLM_RESIDENT_SEMANTIC_AUDIT.md
- /home/qhq/serverless_llm_experiment_retry14_baseline/docs/ieee_tc/D213_RESIDENT_REFERENCE_PROTOCOL_AUDIT.md
- /home/qhq/serverless_llm_baselines/scripts/replay_vllm_engine_trace.py
- /home/qhq/serverless_llm_baselines/scripts/ieee_tc_serverless_measurement.py
- /home/qhq/serverless_llm_baselines/configs/ieee_tc/vllm_7b_resident_native_v1.json
- /home/qhq/serverless_llm_experiment/results/ieee_tc/vllm_qualification/d419_20261011/

Prioritize any issue that actually prevents safe reuse; do not demand stricter output equality than the frozen protocol. Final answer sufficient for executor to save an advisory audit report and trace.

## Follow-up message 1

A GPU qualification is now running. Limit remaining review to small retained source/result reads; no whole-tree hashing, CPU tests, imports of serving backends, compression, cleanup or remote operations. Please send bounded findings when ready. Your review is advisory and must not turn into a new benchmark requirement.

## Interim response

Bounded interim: deterministic D419 core evidence is internally consistent. Manifest has 103/103 unique entries; all existed and matched recorded byte count/SHA when checked before your new GPU-run notice. Raw replay independently reproduces 100 contracts/responses, 29 actual adapter IDs, 17,369 target/native tokens, TTFT 18,634.9075 ms mean / 74,980.1629 ms Type-1 P95, TPOT 25.3491 / 27.6129 ms, E2E 23,055.6557 ms, and frozen-SLO 47/100 TTFT, 100/100 TPOT, 47/100 joint. Physical intervals/windows sum exactly to 1,019.258640 GPU-s; n_correct remains null. Likely overall WARN only for bounded reuse/scope (100-request prefix, 29 of nominal 500 IDs, 4 weight hashes, one run) and missing retained test-execution receipt for the doc’s “27 CPU tests passed” claim; no issue found that blocks narrow affected-path/measurement reuse. Full W0/U_ref/ranking remains unsupported. I’m stopping broad reads and will finish from retained evidence.

## Follow-up message 2

Thanks; please conclude the advisory A–F report now from your checked evidence, with exact citations and limits. No further tests/scans needed. Your receipt concern will be recorded; any new verification receipt can be captured after the live gate ends, not during inference.
