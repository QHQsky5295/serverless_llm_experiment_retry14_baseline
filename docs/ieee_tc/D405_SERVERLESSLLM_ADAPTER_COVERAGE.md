# D405: ServerlessLLM whole-pool functional coverage view

2026-10-11. CPU implementation and historical inventory, not a GPU result.
Main plan SHA f851e6da5237cb706f5a167dd1180984d5d02694fa6568f1720e0e5aada0b1f5;
frozen V1 plus single-run V2 unchanged. PrimeLoRA D371 remains sealed.

## Evidence and interpretation

| Existing evidence | Completed requests | Actual adapter IDs | Reuse decision |
|---|---:|---:|---|
| May clean 7B replay | 4000 reported success | 132 | R2, historical contract |
| May earlier 7B replay | 4000 reported success | 132 | R2; does not supersede clean run |
| May 3B replay | 4000 reported success | 132 | R2, not a new 3B run |
| D404 native 7B qualification | 100 native contracts | 29 | Reuse, no repeat |
| Original trace first 500 / first 1000 | not new executions | 46 / 60 | Inputs only |

Thus replaying another ordinary prefix cannot close 500-ID engine coverage.
D80's full HTTP/content qualification is reused, not repeated or mislabelled as
engine execution. Source files and hashes are in main repository
`paper_results/ieee_tc/baseline_audit/d405_coverage_inventory.json`.

D404 assignment-to-worker-terminal intervals overlap at most twice per engine,
eight overall; mean duration11.077381962453947s. These intervals end before the
router decrements its request count: they are NOT exact router occupancy.
The frozen official router uses target both for desired instance count and
per-instance capacity. This supports checking public configuration, not proving
GPU saturation or predicting throughput by a short-prefix extrapolation.
[Official fixed source](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/routers/roundrobin_router.py).
Native LoRA requests carry explicit adapter identity; a directory or successful
download alone does not verify model execution.
[Versioned backend documentation](https://docs.vllm.ai/en/v0.10.2/features/lora.html).

## Narrow implementation

The existing prepare helper supports opt-in `request_view=adapter_coverage_v1`,
exactly500 requests, `qualification_only=true`, `formal_run=false`, and
`experiment_role=adapter_pool_functional_coverage`. All are checked; ambiguous
or performance-labelled500 views are rejected. Defaults remain byte-equivalent
to the old100/1000/4000 source-prefix construction.

Bind lexically sorted IDs from the SHA-locked existing500 pool to the existing
trace's first500 ordered rows. Keep request IDs, arrival offsets, prompts and
declared output targets. Derived rows contain original row SHA/adapter ID and
pool SHA; derived SHA and profile ADAPTER_COVERAGE_V1 distinguish them from W0.
No trace/pool files are generated or overwritten. Publisher, physical ledger
and deadline reconstruct the same view. Contracts retain derivation metadata;
physical native binding is checked against the mapped adapter, not the old one.

7B view SHA286f7547a8129fd6e75885379428b84c9a5c23a7eb02bf1d7546f49bcf2cee5d.
The500 mapped IDs exactly equal the existing pool. Total materialized payload
12,985,984,450B is recorded for disk planning, not as compressed wire size.

87 CPU tests passed in7.478s under4GiB/zero-swap auxiliary scope. Includes six
new tests for preserved work, provenance, unchanged defaults, invalid identity,
pool/source drift, public ingress and physical adapter binding. The preceding
measurement-only suite53 tests also passed in5.681s; no GPU test claimed.

## Next affected gate, D406

Run this500-ID view once, real prepublished HTTP, cold private cache, same fast
loader, native RR/autoscaler/storage scheduler and repaired request identities.
Use public target4, matching already-present max_num_seqs4/max_loras4; keep
min1/max4/keep10 and all other D404 settings. This is a functional candidate,
not a selected optimum and not a target2-vs4 performance comparison (adapter
mapping differs). Do not port PrimeLoRA policies or repeat the old polling pair.

Require500 distinct native bindings, exact output targets, prompt/provenance,
remote content/receipt joins, live scheduler and actual ready worker identities,
complete physical release and no hidden fallback. Numerical distinguishability
is a separate qualification: repeated/zero weights remain disclosed. Failure is
retained; it must not be bypassed by reducing the pool. No figures or ranking.
Afterwards return to original1000 compatibility/public-config validation, then
one full4000 W0 with frozen configuration. No new source trace or weights.
