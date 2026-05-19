# New Serverless Baselines Remote V1

This is a non-overwriting curated bundle for the 2026-05-18
ServerlessLLM-new true-remote candidate experiment. It does not replace the
default paper data in `paper_results/final_v2/` or the default figures in
`figs/`.

## Scope

- System: `ServerlessLLM-new`
- Upstream commit: `9f50241baa5386e06a9321c51f19a9ef5f964c2b`
- Harness: `/home/qhq/serverless_llm_baselines`
- Result section:
  `/home/qhq/serverless_llm_baselines/results/paper_experiments/15_new_serverless_baselines_remote_v1/`
- Metric schema: `e2e_v3`
- Workloads: Llama-2 7B and Llama-3.2 3B, 4,000 requests, 500 adapters, s8,
  true remote adapter artifacts.

## Files

- `tables/serverlessllm_new_metrics.json`: compact metrics and provenance.
- `tables/serverlessllm_new_metrics.csv`: table-ready copy of the same metrics.
- `source_summaries/*.json.gz`: compressed source summary JSON files from the
  formal replays.
- `SHA256SUMS`: checksums for the bundle.

## Inclusion Decision

These results are valid candidate rows for a future formal comparison table or
appendix because both backbones completed `4000/4000` requests with no replay
failures and no `trace_expected` token fallback. They should be labeled
`ServerlessLLM-new` and kept separate from the older `ServerlessLLM` baseline.

The old Llama-2 7B first run is superseded by
`20260518_serverlessllm_new_remote_v1_clean7b` because that rerun had no
concurrent dependency/build activity. No old figures, tables, or default paper
snapshots were overwritten.
