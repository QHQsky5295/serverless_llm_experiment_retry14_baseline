# D762: content-verified shared tokenizer representation

This is a necessary representation adaptation following the D760 3B native
full-workload failure and D761 CPU reproduction, not a baseline parameter search.
The native vLLM 0.10.2 tokenizer LRU can repeatedly construct identical costly
tokenizers; D761 observed delayed memory reclamation despite capacity four.

Explicit `shared_tokenizer=True` adds one startup call to a new source view;
the installed native checkout and default generated path remain unchanged.
`verified_published_base_v1` verifies every permitted published adapter's five
JSON tokenizer/config files against the actual engine tokenizer source. It
supports the audited artifact layout only and rejects heterogeneous contents.
Normal per-request remote resolution/content verification remains mandatory.

Only TokenizerGroup's per-adapter tokenizer selection is disabled, after checking
its empty cache and identity in input/preprocessor/output consumers. Backend and
engine LoRA enablement, native LoRARequest/weights, RR, store, autoscaling, limits,
sampling, real transfer bytes and cache policies are untouched. No periodic GC,
fallback, process restart, prefetch, backend upgrade or runtime parameter search.
The source-view and backend-ready receipts record the contract and content proof.

Evidence: 33 affected unit/regression tests passed. Native CPU check: 628 sync and
628 async lookups (original 128-access sequence plus all 500 IDs), one tokenizer,
zero per-adapter LRU entries; RSS +253952 B during lookups. Eight encode/decode
unit cases matched one independently loaded original adapter tokenizer. Auxiliary
cgroup peak 599142400 B, no high/max/OOM/swap. No model/GPU experiment yet; neither
numerical LoRA correctness nor complete performance recovery is proved by this.

Sources: [v0.10.2 TokenizerGroup](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/transformers_utils/tokenizer_group.py),
[v0.11.0 common-tokenizer design](https://raw.githubusercontent.com/vllm-project/vllm/v0.11.0/vllm/v1/engine/processor.py).
The latter is a representation reference, not evidence that another version fixes
this machine's full-run failure. Native runtime remains v0.10.2.

Next: one existing 100-request changed-path qualification, then a new complete W0
key and unchanged W1/W2. Do not repeat the old 500/1000 qualification sequence.
Full evidence and source identities are in the main repository's
`paper_results/ieee_tc/baseline_audit/d762_verified_common_tokenizer/`.
