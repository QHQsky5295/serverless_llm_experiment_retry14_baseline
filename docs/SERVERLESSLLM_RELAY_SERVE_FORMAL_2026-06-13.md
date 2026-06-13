# ServerlessLLM RelayServe-Workload Formal Closure

Date: 2026-06-13

ServerlessLLM completed the frozen RelayServe continuation workload for both
paper models. The runs use the official ServerlessLLM scheduler, autoscaler,
model lifecycle, and worker control plane. The only control-plane timing
adaptation is the preregistered 5 ms request-allocation poll; the official
autoscaler interval remains unchanged.

## Frozen selection

The selected configurations came from disjoint 512-request calibration traces:

| Model | Target concurrency | Keep alive |
|---|---:|---:|
| Llama-3.2 3B | 16 | 60 s |
| Llama-2 7B | 8 | 60 s |

No formal4000 request was used to select either configuration.

## Formal4000 results

| Model | Completed | TTFT P95 | TPOT P95 | Joint attainment | E2E avg | CE | Paper P95 gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Llama-3.2 3B | 4000/4000 | 259.466 ms | 15.486 ms | 74.875% | 2518.666 ms | 284.838983 | fail |
| Llama-2 7B | 4000/4000 | 1108.989 ms | 32.249 ms | 83.150% | 5831.871 ms | 76.818968 | fail |

The 3B service TTFT remained fast, but burst recovery and request allocation
introduced deadline misses after idle intervals. The 7B result shows the same
effect more strongly: model execution is viable, while lifecycle recovery and
dispatch waiting dominate the formal TTFT tail.

## Evidence

- Formal table:
  `paper_artifacts/relayserve_v4/serverlessllm_formal4000.csv`
- 3B raw run:
  `results/relayserve_continuation/serverlessllm/20260613_serverlessllm_3b_formal4000_t16_k60_v1`
- 7B raw run:
  `results/relayserve_continuation/serverlessllm/20260613_serverlessllm_7b_formal4000_t8_k60_v1`
- Builder:
  `scripts/build_serverlessllm_formal_table.py`
- Independent verifier:
  `scripts/verify_serverlessllm_formal.py`

The formal table records absolute evidence paths and SHA-256 hashes for each
raw record file, source summary, manifest, trace, and source revision.
