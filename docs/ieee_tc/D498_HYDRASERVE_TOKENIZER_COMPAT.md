# D497 failure and D498 lossless tokenizer compatibility

This is a compatibility/correctness checkpoint, **not** a HydraServe performance
result. Original backend vLLM 0.4.2, controller policies, initialization overlap,
LoRA kernels, batching and timeouts remain unchanged. Existing unrelated k3s is
not accessed or modified. The two input requests are original trace req_00000
(finance_lora, 152 tokens) and req_00021 (medical_lora, 89 tokens).

## D497 actual observation

Original CPU provider -> private shared memory -> native C++ GPU loading ->
original app initialization succeeded. The actual Request/Instance/ChatEngine
path submitted req_00000 with the frozen prompt IDs, greedy generation,
ignore_eos and correct finance LoRA name/index/path. HTTP headers returned 200,
but the engine died before any native output event. The stream failed; medical
was not submitted. Zero requests completed. No latency/resource ranking follows.

The server traceback is at native TokenizerGroup.get_lora_tokenizer, then
AutoTokenizer/TokenizerFast.from_file. The exact error is `ModelWrapper` JSON
deserialization at line 277138. This is not a network timeout, GPU OOM or proof
that HydraServe's scheduling mechanism performs poorly.

Existing adapter tokenizer JSON has BPE merges as pairs, whereas pinned
tokenizers 0.19.1 expects strings. The backbone tokenizer uses the legacy format.
All three existing tokenizer.model files have identical SHA
9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347;
this fact alone is not treated as sufficient behavioral equivalence.

Primary code evidence: [tokenizers 0.19.1 serializer/deserializer](https://raw.githubusercontent.com/huggingface/tokenizers/v0.19.1/tokenizers/src/models/bpe/serialization.rs)
and [0.20.0 pair/legacy schema](https://raw.githubusercontent.com/huggingface/tokenizers/v0.20.0/tokenizers/src/models/bpe/serialization.rs).
The old schema cannot represent merge tokens containing ASCII spaces
unambiguously; the adapter rejects that case instead of guessing.

D497 whole-service memory peak 42,372,599,808 bytes; high/max/OOM/OOM-kill events
and service swap were zero. All owned worker/provider/storage/pause processes
were released; GPU compute census empty and actual service group absent. Failed
source overlays, native logs, empty token events and exact scope closures remain.

## D498 necessary adaptation

`runtime/tokenizer_compat.py` converts only pairs `[a,b]` to legacy `a b`, in
the original order, with an exact JSON semantic inverse check. It leaves all
other fields unchanged. Mixed/malformed/ambiguous formats fail explicitly.
Already-legacy bytes remain unchanged. It does not catch runtime tokenizer
errors, select another tokenizer, change vocabulary or use a base-model fallback.

An immutable metadata view symlinks original files and writes only the converted
tokenizer.json. No weight bytes are copied, modified or generated; source pools
are untouched. This is input-format compatibility, not a new trained artifact or
performance configuration. Native adapter registration must use the view path
and mount its original symlink targets read-only. Formal real-remote integration
must account for any request-triggered local metadata preparation; it cannot
pre-materialize weights for free or substitute the local diagnostic for remote.

Qualification compares existing transformers/tokenizers 0.22.2 reading original
adapters against pinned transformers 4.43.1/tokenizers 0.19.1 reading the view.
For each of the two adapters: all 4,000 original trace texts' encoded ID sequences
and decoded text, full vocabulary mapping, all 32,000 single-token decodes,
special token map/IDs/BOS/EOS. Digests and exact comparison are saved. This is
strong coverage of the frozen inputs plus a reversible format transformation,
not a universal proof of two tokenizer libraries or numerical LoRA inference.

## Next gate and scope

Run the prepared D498 actual GPU witness once, retaining D497 unchanged. Only
tokenizer metadata delivery differs. Then qualify request-driven real-remote
resolution, full original Scheduler and 100 -> 500-ID -> 1,000 -> 4,000 replay.
Do not start another PrimeLoRA optimization before HydraServe actual results
and all available baseline comparison.

SHM control connection ownership remains a separate open correctness issue:
startup asyncio.run closes the writer's event loop before later release. Repair
its actual owner lifetime and prove prompt buffer release before repeated or
multi-replica loading. Pipeline/KV migration/concurrent consolidation numerical
correctness also remains open; this checkpoint does not claim those mechanisms.

Raw immutable evidence: original experiment results/ieee_tc/
hydraserve_qualification/d497_20261013 and d498_20261013. Curated evidence and
source manifest: paper_results/ieee_tc/hydraserve/d498/summary.json.
