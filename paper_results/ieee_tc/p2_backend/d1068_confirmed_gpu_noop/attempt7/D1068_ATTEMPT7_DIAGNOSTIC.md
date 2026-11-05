# D1068 attempt 7: native-descriptor full replay

## Result boundary

This is the corrected full Llama-3.2-3B W1 replay for the D1068 mechanism question. It is a completed diagnostic, not a formal G1/G2 comparison point. The run used the existing wrapper, the same code/configuration/trace/adapter subset as attempt 6, and the qualified native HOST descriptor. The wrapper records D1000 mechanism replays as `formal_run=false` and `trace_role=exploratory`; `n_correct` is consequently unknown. Those facts are preserved rather than silently promoting the result to a ranking.

## What completed

- 4,000/4,000 requests completed; 0 failed requests.
- `fixed_length_greedy_v1`, 3,053,162 total tokens, 132 real HTTP prepublished-gzip transfers, and 0 request-time remote packing operations.
- Native descriptor evidence is present as `fresh_native_tensor_descriptor_v1`; the expected D747 library SHA is recorded in `sources.json`.
- Physical lifecycle measurement completed at 6,054.364375863224 GPU-s; all allocation journals were released, cleanup completed, and the native GPU context was released.
- The run recorded 8 scale-ups and 7 scale-downs. A worker failure directory was retained as failure evidence and is not mixed into the success count.

## Mechanism evidence

The corrected replay produced 794 `confirmed_gpu_no_preparation` proof flags, 87 logical zero-demand residency rounds, and 6 logical confirmed-file-copy no-op events. The JSON contains a public/private duplicate serialization of event-kind records; the serialized counts are 1,588, 174, and 12 respectively. Counts in this document remove that duplication.

The proof is therefore exercised under the native descriptor path. It is evidence that the exact current-state proof is reachable in a full replay, not evidence that all planned preparation work is removable or that G1/G2 has been won.

## Diagnostic observations (not ranking claims)

The legacy 5,000-ms SLO field reports 96.15%, average TTFT is 2,235.297 ms, P95 TTFT is 1,667.349 ms, average E2E is 4,154.920 ms, P95 E2E is 5,956.076 ms, and the result schema reports 4,000 completed requests. These values are retained to diagnose the current path; they are not used as a formal comparison because the run is exploratory and independent arithmetic correctness is unavailable. In particular, no causal improvement against D1047 is inferred from the lower physical-GPU total.

## Follow-up boundary

Do not run another short replay or tune from this diagnostic. The next formal performance point, if needed, requires a new registered formal protocol key with formal trace identity and an independent correctness contract. Attempt 6 and attempt 7 remain separate raw evidence and must not overwrite one another.
