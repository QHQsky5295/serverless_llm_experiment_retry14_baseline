# New Serverless Baselines Remote V1

This is a non-overwriting curated bundle for the 2026-05-18 new serverless
baseline campaign. It contains the ServerlessLLM-new true-remote candidate
experiment and gate evidence for later systems. It does not replace the default
paper data in `paper_results/final_v2/` or the default figures in `figs/`.

## Scope

- Formal candidate system: `ServerlessLLM-new`
- Limited single-backbone evidence: `dLoRA` Llama-3.2 3B official
  period-migration full replay.
- Gate-only systems: `Loquetier`, `AIBrix`, `HydraServe`, `Sarathi-Serve`,
  `Medusa`, `FaaScale/LambdaScale`; `dLoRA` Llama-2 7B is closed as an
  infeasible wrapper-only gate on this machine.
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
- `tables/new_serverless_baselines_inclusion_status_2026-05-21.csv`: current
  paper-use decision matrix for ServerlessLLM-new, dLoRA, Loquetier, AIBrix,
  HydraServe, Sarathi-Serve, Medusa, and FaaScale/LambdaScale.
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
- `gates/dlora/`: dLoRA build/import, real-adapter scale gates, dispatch-only
  full replay evidence, official period-migration short gates, the selected
  Llama-3.2 3B official full replay, and Llama-2 7B infeasibility gates. dLoRA
  can consume real PEFT adapters through the local compatibility layer.
  `migration_type=3` completed the 3B true-remote workload with `4000/4000`
  requests and no token fallback under the best wrapper-only DP2/TP1 envelope.
  The matching 7B path was driven through DP2/TP2 and G1/TP4 wrapper envelopes;
  the final G1/TP4 `gpu_capacity=1`, `gpu_memory_utilization=0.99` gate still
  produced `# GPU blocks: 0, # CPU blocks: 1024` after all `500/500` remote
  adapters materialized and loaded. dLoRA therefore is limited/appendix
  evidence here, not a full 3B+7B main comparison row.
- `gates/loquetier/`: Loquetier local-adaptation patch, compact gate outputs,
  and real-adapter evidence. Loquetier currently passes Llama-3.2 3B through
  256 adapters / 1024 filtered requests and Llama-2 7B through 128 adapters /
  256 filtered requests, but is not a formal result row because the 3B/500
  adapter preflight OOMs while materializing mixed-LoRA adapter weights on a
  24GB RTX 3090.
- `gates/aibrix/`: AIBrix build and runtime-LoRA component gate evidence.
  AIBrix controller-manager, gateway-plugins, and Python runtime build/import
  locally, and the runtime sidecar can load/unload a real Llama-3.2 3B LoRA
  adapter through vLLM's dynamic LoRA API. It is not a formal result row because
  full AIBrix requires a Kubernetes GPU control plane that this machine cannot
  run from the current user account.
- `gates/hydraserve/`: HydraServe control-plane import, platform gate, and
  LoRA-interface audit evidence. HydraServe's embedded vLLM source exposes
  static LoRA arguments, but the full system requires an unavailable
  Docker/Kubernetes GPU runtime, and the scheduler path does not preserve
  per-request adapter identity for the closed PrimeLoRA LoRA workload.
- `gates/sarathi_serve/`: Sarathi-Serve package/source and LoRA-workload audit.
  The faithful OSDI branch has no LoRA/adapter/PEFT path; the newer main branch
  has only an unused `LoRAModulePath` dataclass and lacks `enable_lora`,
  `LoRARequest`, PEFT loading, and adapter-aware scheduling. Adding the closed
  PrimeLoRA workload would be a new LoRA serving implementation, so this remains
  appendix/gate evidence only.
- `SHA256SUMS`: checksums for the bundle.

## Inclusion Decision

The ServerlessLLM-new results are valid candidate rows for a future formal
comparison table or appendix because both backbones completed `4000/4000`
requests with no replay failures and no `trace_expected` token fallback. They
should be labeled `ServerlessLLM-new` and kept separate from the older
`ServerlessLLM` baseline.

dLoRA, Loquetier, AIBrix, HydraServe, Sarathi-Serve, Medusa, and
FaaScale/LambdaScale are not full formal main rows in this bundle. dLoRA now
has a valid Llama-3.2 3B official period-migration full replay (`4000/4000`, no
token fallback, CE `45.5240`), but the Llama-2 7B formal path cannot reach HTTP
readiness under the unchanged 500-adapter true-remote workload without changing
dLoRA/vLLM core memory layout. Treat dLoRA as appendix or limited
single-backbone evidence, not as a full 3B+7B comparison row.
Do not promote any of these systems to formal rows unless their runtime and
scale requirements are satisfied and the closed true-remote LoRA workload can
be replayed into `e2e_v3` without changing workload variables.

The old Llama-2 7B first run is superseded by
`20260518_serverlessllm_new_remote_v1_clean7b` because that rerun had no
concurrent dependency/build activity. No old figures, tables, or default paper
snapshots were overwritten.
