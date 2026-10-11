# D416 — vLLM native service integration (CPU checkpoint)

2026-10-11. This is implementation/CPU validation, **not** a GPU qualification,
Resident budget reference, new performance result or baseline optimum.

## Problem and source evidence

D395/D396 cannot define U_ref: client first-payload/EOF timestamps do not
recover native first/last token events, the external replay owned service-side
artifact loading, and publication waited for all workers. D412 preserves the
evidence and D415 validates the pinned vLLM 0.30.0 native output snapshots.
No old result is replaced or repaired by inventing missing events.

History inspected: fair-runner commits dd1ca72, effb149, 1ebe507 and 1115b16;
the exact D395 launch/fleet YAML; D397 correction; D412 and D415. D395 used
DP4/TP1, round_robin, max_num_seqs2, max_loras4, max_cpu_loras16, token budget
1024, context1024, FP16, fraction0.7, chunked prefill, no prefix cache, non-eager.
These are the next qualification's inherited public workpoint, not a new
parameter search. Recheck environment and actual backend readback before GPU work.

Primary sources checked online and against installed interfaces:

- [v0.30.0 AsyncLLM](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/v1/engine/async_llm.py):
  token-ID prompts, LoRARequest, cumulative outputs, generator cancellation and
  shutdown are the reused native interfaces. Native shutdown alone is not a
  physical release witness.
- [v0.30.0 LoRA documentation](https://docs.vllm.ai/en/v0.30.0/features/lora/):
  use native per-request LoRA binding and ordinary native caching, not a
  PrimeLoRA scheduler or hidden base-model fallback.

Falsifiable hypothesis: an explicit service-owned entry can preserve ordinary
round robin/native serving while satisfying the already frozen input, time and
resource boundaries. CPU tests cover control contracts, not GPU performance.

## Existing entry points extended

`scripts/replay_vllm_engine_trace.py` now has opt-in `native-http-service` and
`native-http-worker` modes. The legacy synchronous replay remains unchanged;
its dirty-module import is lazy and is not used by the new modes.

- Reuse `prepare_ieee_tc_serverless_stack.py`'s independent HTTP publisher,
  frozen request builder, native validator, remote publication resolver and
  terminal recorder. New explicit consumer schema is
  `ieee_tc_vllm_http_replay_v1`; lifecycle is `vllm_resident_pool_v1`.
  A vLLM consumer cannot claim ServerlessLLM's `native_store_pool_v1`.
- Reuse the existing common guarded launcher/watchdog. Its allowed publisher
  helper remains unchanged. The service cannot start without the actual
  admission receipt, matching clock/input config and containment.
- Pre-ready ingress listens before backend imports. The service-owned router
  uses the original request ordinal modulo replica count, including if HTTP
  arrivals reorder. It waits only for that selected worker, never reroutes by
  readiness/adapter/load, never waits for all workers before accepting arrivals.
  The ordinal map only defines the existing deterministic RR policy; future
  adapter popularity, prompts and targets are not scheduling inputs.
- Spawn one native AsyncLLM per allocated GPU under the service domain. The
  resolver, local materialization and LoRARequest execute there, not in the
  publisher. CUMULATIVE output plus D415 snapshots feed the common native
  timeline. Input IDs, target, adapter name/ID/path and terminal are validated;
  no text re-tokenization, payload timestamp, EOF timing or output fallback.
- Each GPU owner is allocated before its worker spawn, bound to process birth
  and actual NVML contexts, and returned only after process/pidfd exit and a
  fresh native census. Startup failures retain an unqualified/open lease rather
  than fabricate release. All four leases and external terminals feed the
  common physical deployment summary.
- Ready receipts record source/config SHA, actual process/affinity/cgroup,
  device, pinned snapshot source and native API files. Atomic no-overwrite
  publication prevents a reader from consuming half a ready receipt.

Deployment difference disclosed: the old client had an API-registration LRU
wrapper in addition to vLLM's cache. The native entry passes LoRARequest directly
to the engine and uses its own ordinary HOST/GPU LRU at the same recorded
capacity. It does not copy that client-side wrapper into the engine, nor label
the new deployment byte-identical to D395. No PrimeLoRA placement policy is
introduced. The existing user-owned replay and relayserve files are untouched.

## Verification and limits

The initial combined suite passed 122 tests in 7.997s in a 4-GiB, zero-swap
auxiliary scope on CPUs2,3,26,27. Ten new service tests cover schema/lifecycle
separation, unknown routing/local fallback rejection, atomic publication,
selected-worker queueing, arrival reordering, cancellation/duplicate IDs,
native token observations, invalid generation, failed remote resolution,
generator closure and real-loopback pre-ready HTTP through the common validator.
The synthetic CPU tokens are explicitly fixtures, not real inference evidence.
The existing main-repository physical lifecycle suite passed32 tests in1.316s.
Final regression counts after review are recorded in EXECUTION_STATUS.

The actual vLLM0.30.0 environment accepted inherited AsyncEngineArgs and the
CUMULATIVE SamplingParams with GPU visibility disabled; generate/shutdown
signatures match. No engine constructed, no weights loaded, no GPU context.
This does not prove live worker containment, complete 500-ID backend coverage,
independent numerical LoRA correctness, remote byte joins or GPU release.

## Next main-line action

Freeze the inherited 7B qualification config and current source/environment,
prepare a new exclusive run key using the existing source100 prefix, check
the full plan/protocol, input/protected hashes and fresh memory/disk/GPU gates.
Reuse the already published D78/D80 cache; do not regenerate it. Then one
affected native GPU-path qualification (pre-ready arrivals, actual remote
bytes/content, all native clocks and four physical releases), audit/reuse
prior backend qualification as applicable, and one full original W0.

Do not repeat D395/D396, retune ServerlessLLM, jump3B or mark U_ref closed.
PrimeLoRA D371 and ServerlessLLM D414 remain sealed. Failure/partial observations
stay separate; no best-repeat selection, new trace/pool, figure or paper edit.
