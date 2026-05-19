# New Serverless Baselines Remote V1

This is a non-overwriting curated bundle for the 2026-05-18 new serverless
baseline campaign. It contains the ServerlessLLM-new true-remote candidate
experiment and gate evidence for later systems. It does not replace the default
paper data in `paper_results/final_v2/` or the default figures in `figs/`.

## Scope

- Formal candidate system: `ServerlessLLM-new`
- Gate-only systems: `Medusa`, `FaaScale/LambdaScale`
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
- `gates/medusa/`: Medusa official and local-adaptation gate evidence. Medusa
  builds/imports after local path adaptation, but is not a formal result row on
  this machine because the SPDK runtime prerequisites are absent.
- `gates/faascale/`: FaaScale/LambdaScale official and local-adaptation gate
  evidence. LambdaScale imports and IPC/RDMA-P2P bindings build/import after
  isolated repair, but it is not a formal result row on this machine because
  the RDMA device stack is not exposed and the source lacks ready
  Llama-3.2 3B plus LoRA/PEFT workload support.
- `SHA256SUMS`: checksums for the bundle.

## Inclusion Decision

The ServerlessLLM-new results are valid candidate rows for a future formal
comparison table or appendix because both backbones completed `4000/4000`
requests with no replay failures and no `trace_expected` token fallback. They
should be labeled `ServerlessLLM-new` and kept separate from the older
`ServerlessLLM` baseline.

Medusa and FaaScale/LambdaScale are gate-only evidence on this machine. Do not
promote them to formal rows unless their runtime prerequisites are satisfied
and the closed true-remote LoRA workload can be replayed into `e2e_v3` without
changing workload variables.

The old Llama-2 7B first run is superseded by
`20260518_serverlessllm_new_remote_v1_clean7b` because that rerun had no
concurrent dependency/build activity. No old figures, tables, or default paper
snapshots were overwritten.
