# D214 versus D210: stage-level diagnostic comparison

This is a descriptive, single-run comparison under the same 7B W0 trace,
adapter subset, vLLM backend and physical resource envelope. It is not a CI,
formal SLO, G1/G2, or baseline-ranking result. The D210 and D214 source-output
SHA256 values are respectively
`942948559cea51aebecbcc6f8fdfd55acfb455e4afc33dfa01782b86b2b3ecb1` and
`6db6f93e3cc7e790a7af880ba095541f970edfd1ecc1f9ce20f459355c6290eb`.

| Metric | D210 | D214 | Relative change (D210−D214)/D210 |
|---|---:|---:|---:|
| Routing/control mean (µs) | 325,862.938 | 462,790.392 | −42.01% |
| Routing/control P95 (µs) | 1,075,641.501 | 1,463,889.913 | −36.09% |
| Dispatch wait mean (ms) | 938.404 | 1,267.387 | −35.05% |
| Dispatch wait P95 (ms) | 2,361.264 | 3,182.245 | −34.77% |
| Service TTFT mean (ms) | 712.107 | 686.015 | +3.67% |
| Service TTFT P95 (ms) | 1,892.091 | 1,875.025 | +0.90% |
| Overall TTFT mean (ms) | 1,650.511 | 1,953.402 | −18.35% |
| Overall TTFT P95 (ms) | 3,415.837 | 4,232.688 | −23.91% |
| Physical GPU-seconds (descriptive) | 15,916.824 | 15,907.670 | +0.058% |

The identity-recheck candidate slightly reduced the service-side TTFT but added
approximately 137 ms to mean routing/control and 329 ms to mean dispatch wait.
The net result is a worse user-visible TTFT despite essentially unchanged
physical GPU time. This falsifies the narrow hypothesis that replacing the full
scoped observation with a fresh identity read is a beneficial optimization on
this implementation. The likely mechanism is that the identity RPC is not a
cheap event-driven publication: it still traverses the native GPU-reference
path, while the candidate also increases collection activity and records stale
rejections. This is a hypothesis for the next minimal audit, not a causal claim
from one run.

Next optimization must first measure the identity endpoint's own span and the
full-observation span on the same requests, then test one event/version-based
confirmed-state publication design only if that audit shows avoidable duplicate
work. A last-known tier shortcut, TTL cache, or test-point-specific bypass is
not authorized.
