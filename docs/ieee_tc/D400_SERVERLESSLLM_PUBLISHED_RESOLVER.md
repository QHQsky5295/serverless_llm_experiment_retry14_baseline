# D400 — ServerlessLLM service-owned published LoRA resolution

2026-10-10. CPU transport qualification only; no new model/performance run.
This extends the existing D399 contained qualification path. It does not
qualify ServerlessLLM for the main comparison by itself.

## Evidence and adaptation decision

The audited native controller normally registers LoRA artifacts before router
creation. For a missing path its downloader creates PEFT/store-format output,
whereas the qualified native vLLM backend consumes standard PEFT directories.
Pointing registration at nonexistent cache paths is therefore incorrect.
The historical fair runner already has `dynamic_remote` delivery and the local
controller exposes `skip_store_lora_registration`; however its fetch previously
ran in the replay client. That resource attribution is not acceptable here.

The new explicit `published_gzip_ondemand_peft_v1` adaptation moves generic
on-demand PEFT transport into the native backend service. It is not presented
as unmodified upstream behavior or as ServerlessLLM's native LoRA checkpoint
loader. Native backbone store registration/fast loading stays enabled, and
RR, queue capacity, autoscaling and engine sampling are unchanged. It adds no
PrimeLoRA routing, preparation planner, admission policy or prefetch.

This boundary is consistent with the download-before-LoRARequest pattern in
the version-matched vLLM resolver interface, but that interface does not supply
this ServerlessLLM integration automatically. The service embeds the engine,
not the vLLM OpenAI server. References checked on 2026-10-10:
[vLLM 0.10.2 LoRA resolvers](https://docs.vllm.ai/en/v0.10.2/features/lora.html),
[fixed ServerlessLLM controller](https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/controller.py).
Local compatibility changes and the upstream identity remain separately hashed.

## Execution contract

The existing replay configuration accepts an optional `remote_artifacts` object
with exactly `mode`, `endpoint`, `token_env`, `timeout_s`. It must also select
`service_pre_ready_v1`. Existing `pool_index` and `pool_index_sha256` identify
the frozen content index. No new pool, weights or trace is generated.

- The launcher audits existing pool metadata, creates a fresh run-local cache,
  and maps the same full adapter set to that cache. It disables only store LoRA
  conversion; native backbone store registration cannot be disabled here.
- The service backend resolves the requested adapter before constructing its
  native LoRARequest. It uses the already-tested shared strict HTTP downloader.
  The replay client reads only frozen metadata and hashes, never weights or
  remotely fetched artifacts. No hidden local-pool fallback exists.
- The cache is ordinary node-shared on-demand materialization, not a tiered
  policy. Only touched IDs are fetched; there is no bulk prestage or new
  permanent extracted pool. Cache scope and lack of eviction are explicit.
- Objects must use the approved prepublished-gzip protocol. Actual transfer,
  archive/file SHA verification, extraction and local publication are in the
  service resource domain and request latency. No request-time remote packing,
  simulated delay, posthoc subtraction or enforced equal fetch count.
- Each adapter has an interprocess publication lock. Different adapters are
  independent. Lock waiting occurs off the native async event loop; its 50-ms
  cancellation-check interval is disclosed implementation overhead, not a
  request retry, bandwidth limit or tuning parameter.
- Every cache hit requires a previously verified receipt plus matching file
  set/stat identity. A bare directory, changed content/stat identity or missing
  receipt fails explicitly, without automatic repair/refetch. The stat guard
  assumes this owned immutable cache; it is not an adversarial security proof.
- Each backend checks actual cgroup membership and records its PID, cgroup and
  affinity. Per-process exclusive journals retain failures, transfer evidence,
  request binding and cache hits. Tokens and credentials are never journaled.
- Cancellation signals and drains the I/O worker; ordinary failures have no
  hidden retry. The external service watchdog remains responsible for fatal
  process/host failures. Unverified partial publication is never a valid hit.
- Request observations bind adapter, content/manifest SHA, publication receipt
  SHA and source transfer ID. Cache hits report zero new wire bytes, retaining
  the source transfer ID. Native backend entry <= resolution <= engine dispatch
  is checked. Resolution is a subspan, not an extra additive E2E component.
- New source views hash the shared downloader/clock plus existing measured
  native sources/helper. Old source views and results are not overwritten.

`remote_qualified` deliberately remains false in the launcher result until
real model-path and journal validation is performed. Configuration selection
or successful HTTP fixture tests alone must not set it true. Native token
binding still does not prove numerical LoRA distinguishability.

## Validation and next boundary

77 CPU tests passed in 7.600 seconds in `ieee-tc-d400-cpu-final.scope`,
MemoryHigh=3 GiB, MemoryMax=4 GiB, swap=0, CPUs 2/3/26/27, CPUQuota=400%.
Eight new tests plus expanded existing assertions cover actual loopback
published-object HTTP, three separate processes coalescing one miss, independent
objects, cancellation/draining, corrupt/bare caches, HTTP failure without retry,
wrong identity/path/cgroup, cold-cache reuse rejection, and the native hook.
The existing native token/timing, source-view, startup admission, router and
launcher tests remain passing. Synthetic fixture TTFT/CE printed by an old
summary unit test are not experiment results. Tiny temporary files are unit
fixtures, not regenerated experimental LoRA artifacts.

No GPU model was launched; no remote production service was started/restarted,
no D78/D80 cache was republished, and no figure was made. Preserve all D72/D73
diagnostics; do not repeat them. User-dirty replay/relay scripts are unchanged.

Next: reuse the existing physical allocation/census/deployment interfaces to
cover the native store and actual engine processes over their lifetimes,
including retired actors, with verified final release. End-of-run actor lookup
alone remains insufficient. Then recheck disk-growth admission, qualify this
affected 7B path once, and perform the matched full workload under a frozen
configuration. Other 7B baselines, valid Resident reference, 3B, ablations,
mechanisms and sensitivities remain open.
