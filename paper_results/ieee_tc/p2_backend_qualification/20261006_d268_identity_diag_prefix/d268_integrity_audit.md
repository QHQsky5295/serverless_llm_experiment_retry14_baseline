# D268 independent integrity audit

The audit is read-only and classifies evidence, not system quality.

| axis | classification | boundary |
|---|---|---|
| A. provenance and semantic ground truth | WARN | Trace, prefix count, request hashes, fixed-output contract, plan SHA, and launch sidecars agree. No semantic ground truth or sealed source-commit/config digest is present in the result metadata; native token/hash contracts are not correctness labels. |
| B. metric normalization | PASS for descriptive raw fields; WARN for ranking | No self-output normalization was found in the preserved terminal path. GPU-second/CE and TokenProxy values are explicitly simulated infrastructure-cost/output-token proxies and remain supplementary; no formal denominator or comparison claim is allowed. |
| C. existence and cleanup | PASS for 210/210 operational closure; FAIL for strict physical-release proof | Replay, ingress, terminals, service, watchdog, native context, and owned service path agree. The launch/watchdog evidence still has `proves_physical_lease_release=false`; the root result hash is not sealed inside the launch receipt. |
| D. mechanism activity | PASS for configured observer/native worker activity; WARN for identity recurrence evidence | vLLM workers and IEEE observer hooks were live. The terminal schema has no direct native identity-conflict counter, so zero recurrence is inferred from 210 successful native contracts and no error text, not directly measured. |
| E. scope and phase semantics | WARN | 210 requests were repartitioned into 8 mini-phases, not D266's 8x500. `trace_role=exploratory`, `formal_run=false`, and `formal_comparison_qualified=false`. |
| F. evaluation type | PASS as bounded operational proxy | This is a native fixed-length/token-ID contract and cleanup diagnostic on real RTX 3090/vLLM, closest to a self-supervised operational proxy. It is not semantic correctness, performance qualification, readiness, G1/G2, or ranking evidence. |

The gated launcher field `pass=true` means the bounded execution and cleanup
workflow completed. It does not override `production_launch_authorized=false`
or `formal_comparison_qualified=false`. The service-log TTFT/CE aggregates
are descriptive runner telemetry because the permitted terminal sidecar has no
per-request timing fields for independent recomputation.
